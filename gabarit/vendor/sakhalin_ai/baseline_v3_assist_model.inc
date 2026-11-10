// Generated from models/baseline_v3_runtime_policy.json. Do not edit by hand.
constexpr double kBaselineV3AssistScoreThreshold = 0.80645161290322576;
constexpr const char* kBaselineV3AssistScoreName = "baseline_v3_assist_score";

double BaselineV3AssistScoreTree0(const std::array<double, 8>& features) {
  if (features[4] <= 0.2405722239418025) {
    return 0;
  } else {
    if (features[4] <= 0.99075833152796644) {
      if (features[6] <= 55.019496972978843) {
        return 0;
      } else {
        if (features[6] <= 56.635885751466859) {
          return 0.99331423113658068;
        } else {
          return 0;
        }
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree1(const std::array<double, 8>& features) {
  if (features[1] <= 0.64376556873321533) {
    return 0;
  } else {
    if (features[4] <= 1.0390564421527346) {
      if (features[6] <= 56.659900879576597) {
        if (features[4] <= 0.91116745485137007) {
          return 1;
        } else {
          return 0.93023255813953476;
        }
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree2(const std::array<double, 8>& features) {
  if (features[3] <= 0.61948281526565552) {
    return 0;
  } else {
    if (features[3] <= 1.6678162664175034) {
      if (features[0] <= 108.5) {
        return 0.83333333333333337;
      } else {
        if (features[6] <= 55.817378043813974) {
          return 0.98765432098765438;
        } else {
          return 1;
        }
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree3(const std::array<double, 8>& features) {
  if (features[1] <= 0.62267351150512695) {
    return 0;
  } else {
    if (features[3] <= 1.9294230937957764) {
      if (features[0] <= 74) {
        return 0;
      } else {
        if (features[0] <= 112.5) {
          return 0.98765432098765438;
        } else {
          return 1;
        }
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree4(const std::array<double, 8>& features) {
  if (features[3] <= 0.4982905387878418) {
    return 0;
  } else {
    if (features[3] <= 1.468310983851552) {
      if (features[4] <= 0.20042878722145585) {
        return 0;
      } else {
        return 1;
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree5(const std::array<double, 8>& features) {
  if (features[1] <= 0.63983064889907837) {
    return 0;
  } else {
    if (features[4] <= 0.94571887957256573) {
      if (features[2] <= 0.31847763061523438) {
        return 0;
      } else {
        return 1;
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree6(const std::array<double, 8>& features) {
  if (features[1] <= 0.65225982666015625) {
    if (features[1] <= 0.63709843158721924) {
      return 0;
    } else {
      if (features[2] <= 0.50532341003417969) {
        return 0.83333333333333337;
      } else {
        return 0;
      }
    }
  } else {
    if (features[0] <= 171.5) {
      if (features[0] <= 111) {
        if (features[0] <= 105.5) {
          return 0.83333333333333337;
        } else {
          return 0;
        }
      } else {
        if (features[2] <= 0.46001434326171875) {
          return 0.99415204678362579;
        } else {
          return 0.87671232876712335;
        }
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree7(const std::array<double, 8>& features) {
  if (features[3] <= 0.43592306971549988) {
    return 0;
  } else {
    if (features[3] <= 1.824202761054039) {
      if (features[3] <= 0.67402970790863037) {
        return 0;
      } else {
        if (features[2] <= 0.31877899169921875) {
          return 0;
        } else {
          return 0.99712368168744014;
        }
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree8(const std::array<double, 8>& features) {
  if (features[3] <= 0.43389829993247986) {
    return 0;
  } else {
    if (features[3] <= 1.9438272640109062) {
      if (features[2] <= 0.95899772644042969) {
        if (features[0] <= 81) {
          return 0;
        } else {
          return 0.99903938520653224;
        }
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree9(const std::array<double, 8>& features) {
  if (features[3] <= 0.47281622886657715) {
    return 0;
  } else {
    if (features[3] <= 1.3727716524153948) {
      if (features[0] <= 81) {
        return 0;
      } else {
        if (features[3] <= 0.86080271005630493) {
          return 0.98765432098765438;
        } else {
          return 1;
        }
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree10(const std::array<double, 8>& features) {
  if (features[5] <= 394.06683227881535) {
    if (features[3] <= 1.9075610265135765) {
      if (features[6] <= 59.694974016304371) {
        if (features[6] <= 53.078439118596876) {
          return 0;
        } else {
          return 1;
        }
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

double BaselineV3AssistScoreTree11(const std::array<double, 8>& features) {
  if (features[3] <= 0.5187225341796875) {
    return 0;
  } else {
    if (features[3] <= 1.8108683973550797) {
      if (features[4] <= 0.25856169460259659) {
        return 0;
      } else {
        if (features[3] <= 1.5080595016479492) {
          return 1;
        } else {
          return 0.93023255813953476;
        }
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree12(const std::array<double, 8>& features) {
  if (features[1] <= 0.63468855619430542) {
    return 0;
  } else {
    if (features[3] <= 1.5284046977758408) {
      if (features[0] <= 112.5) {
        return 0.967741935483871;
      } else {
        return 1;
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree13(const std::array<double, 8>& features) {
  if (features[1] <= 0.6429750919342041) {
    return 0;
  } else {
    if (features[3] <= 1.8862583041191101) {
      if (features[7] <= 63.124003759455363) {
        return 1;
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree14(const std::array<double, 8>& features) {
  if (features[3] <= 0.44766530394554138) {
    return 0;
  } else {
    if (features[3] <= 1.9438272640109062) {
      if (features[3] <= 0.67192453145980835) {
        return 0;
      } else {
        if (features[6] <= 58.837184919165821) {
          return 0.99712368168744014;
        } else {
          return 0;
        }
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree15(const std::array<double, 8>& features) {
  if (features[1] <= 0.63312959671020508) {
    return 0;
  } else {
    if (features[4] <= 0.95572084571048121) {
      if (features[6] <= 56.659900879576597) {
        return 1;
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreTree16(const std::array<double, 8>& features) {
  if (features[1] <= 0.62540161609649658) {
    return 0;
  } else {
    if (features[3] <= 1.8499920666217804) {
      if (features[0] <= 111.5) {
        return 0.83333333333333337;
      } else {
        return 1;
      }
    } else {
      return 0;
    }
  }
}

double BaselineV3AssistScoreScore(const std::array<double, 8>& features) {
  double score = 0.0;
  score += BaselineV3AssistScoreTree0(features);
  score += BaselineV3AssistScoreTree1(features);
  score += BaselineV3AssistScoreTree2(features);
  score += BaselineV3AssistScoreTree3(features);
  score += BaselineV3AssistScoreTree4(features);
  score += BaselineV3AssistScoreTree5(features);
  score += BaselineV3AssistScoreTree6(features);
  score += BaselineV3AssistScoreTree7(features);
  score += BaselineV3AssistScoreTree8(features);
  score += BaselineV3AssistScoreTree9(features);
  score += BaselineV3AssistScoreTree10(features);
  score += BaselineV3AssistScoreTree11(features);
  score += BaselineV3AssistScoreTree12(features);
  score += BaselineV3AssistScoreTree13(features);
  score += BaselineV3AssistScoreTree14(features);
  score += BaselineV3AssistScoreTree15(features);
  score += BaselineV3AssistScoreTree16(features);
  return score / 17.0;
}
