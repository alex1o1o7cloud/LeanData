import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_S7_geometric_sequence_k_l363_36369

noncomputable def S_n (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_S7 (a_4 : ℕ) (h : a_4 = 8) : S_n a_4 1 7 = 56 := by
  sorry

def Sn_formula (k : ℕ) : ℕ := k^2 + k
def a (i d : ℕ) := i * d

theorem geometric_sequence_k (a_1 k : ℕ) (h1 : a_1 = 2) (h2 : (2 * k + 2)^2 = 6 * (k^2 + k)) :
  k = 2 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_S7_geometric_sequence_k_l363_36369


namespace NUMINAMATH_GPT_cube_volume_l363_36320

theorem cube_volume (A : ℝ) (hA : A = 294) : ∃ (V : ℝ), V = 343 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_l363_36320


namespace NUMINAMATH_GPT_factor_polynomial_l363_36366

theorem factor_polynomial :
  4 * (x + 5) * (x + 6) * (x + 10) * (x + 12) - 3 * x^2 = 
  (2 * x^2 + 35 * x + 120) * (x + 8) * (2 * x + 15) := 
by sorry

end NUMINAMATH_GPT_factor_polynomial_l363_36366


namespace NUMINAMATH_GPT_two_painters_days_l363_36305

-- Define the conditions and the proof problem
def five_painters_days : ℕ := 5
def days_per_five_painters : ℕ := 2
def total_painter_days : ℕ := five_painters_days * days_per_five_painters -- Total painter-days for the original scenario
def two_painters : ℕ := 2
def last_day_painter_half_day : ℕ := 1 -- Indicating that one painter works half a day on the last day
def last_day_work : ℕ := two_painters - last_day_painter_half_day / 2 -- Total work on the last day is equivalent to 1.5 painter-days

theorem two_painters_days : total_painter_days = 5 :=
by
  sorry -- Mathematical proof goes here

end NUMINAMATH_GPT_two_painters_days_l363_36305


namespace NUMINAMATH_GPT_equations_have_one_contact_point_l363_36347

theorem equations_have_one_contact_point (c : ℝ):
  (∃ x : ℝ, x^2 + 1 = 4 * x + c) ∧ (∀ x1 x2 : ℝ, (x1 ≠ x2) → ¬(x1^2 + 1 = 4 * x1 + c ∧ x2^2 + 1 = 4 * x2 + c)) ↔ c = -3 :=
by
  sorry

end NUMINAMATH_GPT_equations_have_one_contact_point_l363_36347


namespace NUMINAMATH_GPT_factors_of_180_l363_36336

theorem factors_of_180 : ∃ n : ℕ, n = 18 ∧ ∀ p q r : ℕ, 180 = p^2 * q^2 * r^1 → 
  n = (p + 1) * (q + 1) * (r) :=
by
  sorry

end NUMINAMATH_GPT_factors_of_180_l363_36336


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l363_36357

theorem quadratic_two_distinct_real_roots (m : ℝ) (h : -4 * m > 0) : m = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l363_36357


namespace NUMINAMATH_GPT_average_last_4_matches_l363_36319

theorem average_last_4_matches 
  (avg_10 : ℝ) (avg_6 : ℝ) (result : ℝ)
  (h1 : avg_10 = 38.9)
  (h2 : avg_6 = 42)
  (h3 : result = 34.25) :
  let total_runs_10 := avg_10 * 10
  let total_runs_6 := avg_6 * 6
  let total_runs_4 := total_runs_10 - total_runs_6
  let avg_4 := total_runs_4 / 4
  avg_4 = result :=
  sorry

end NUMINAMATH_GPT_average_last_4_matches_l363_36319


namespace NUMINAMATH_GPT_coordinates_of_point_M_l363_36378

theorem coordinates_of_point_M :
    ∀ (M : ℝ × ℝ),
      (M.1 < 0 ∧ M.2 > 0) → -- M is in the second quadrant
      dist (M.1, M.2) (M.1, 0) = 1 → -- distance to x-axis is 1
      dist (M.1, M.2) (0, M.2) = 2 → -- distance to y-axis is 2
      M = (-2, 1) :=
by
  intros M in_second_quadrant dist_to_x_axis dist_to_y_axis
  sorry

end NUMINAMATH_GPT_coordinates_of_point_M_l363_36378


namespace NUMINAMATH_GPT_fernanda_savings_calc_l363_36342

noncomputable def aryan_debt : ℝ := 1200
noncomputable def kyro_debt : ℝ := aryan_debt / 2
noncomputable def aryan_payment : ℝ := 0.60 * aryan_debt
noncomputable def kyro_payment : ℝ := 0.80 * kyro_debt
noncomputable def initial_savings : ℝ := 300
noncomputable def total_payment_received : ℝ := aryan_payment + kyro_payment
noncomputable def total_savings : ℝ := initial_savings + total_payment_received

theorem fernanda_savings_calc : total_savings = 1500 := by
  sorry

end NUMINAMATH_GPT_fernanda_savings_calc_l363_36342


namespace NUMINAMATH_GPT_other_train_length_l363_36326

-- Define a theorem to prove that the length of the other train (L) is 413.95 meters
theorem other_train_length (length_first_train : ℝ) (speed_first_train_kmph : ℝ) 
                           (speed_second_train_kmph: ℝ) (time_crossing_seconds : ℝ) : 
                           length_first_train = 350 → 
                           speed_first_train_kmph = 150 →
                           speed_second_train_kmph = 100 →
                           time_crossing_seconds = 11 →
                           ∃ (L : ℝ), L = 413.95 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_other_train_length_l363_36326


namespace NUMINAMATH_GPT_platform_length_259_9584_l363_36327

noncomputable def length_of_platform (speed_kmph time_sec train_length_m : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600  -- conversion from kmph to m/s
  let distance_covered := speed_mps * time_sec
  distance_covered - train_length_m

theorem platform_length_259_9584 :
  length_of_platform 72 26 260.0416 = 259.9584 :=
by sorry

end NUMINAMATH_GPT_platform_length_259_9584_l363_36327


namespace NUMINAMATH_GPT_neg_p_l363_36343

-- Define the initial proposition p
def p : Prop := ∀ (m : ℝ), m ≥ 0 → 4^m ≥ 4 * m

-- State the theorem to prove the negation of p
theorem neg_p : ¬p ↔ ∃ (m_0 : ℝ), m_0 ≥ 0 ∧ 4^m_0 < 4 * m_0 :=
by
  sorry

end NUMINAMATH_GPT_neg_p_l363_36343


namespace NUMINAMATH_GPT_oranges_left_l363_36334

-- Main theorem statement: number of oranges left after specified increases and losses
theorem oranges_left (Mary Jason Tom Sarah : ℕ)
  (hMary : Mary = 122)
  (hJason : Jason = 105)
  (hTom : Tom = 85)
  (hSarah : Sarah = 134) 
  (round : ℝ → ℕ) 
  : round (round ( (Mary : ℝ) * 1.1) 
         + round ((Jason : ℝ) * 1.1) 
         + round ((Tom : ℝ) * 1.1) 
         + round ((Sarah : ℝ) * 1.1) 
         - round (0.15 * (round ((Mary : ℝ) * 1.1) 
                         + round ((Jason : ℝ) * 1.1)
                         + round ((Tom : ℝ) * 1.1) 
                         + round ((Sarah : ℝ) * 1.1)) )) = 417  := 
sorry

end NUMINAMATH_GPT_oranges_left_l363_36334


namespace NUMINAMATH_GPT_Alyssa_puppies_l363_36359

theorem Alyssa_puppies (initial_puppies give_away_puppies : ℕ) (h_initial : initial_puppies = 12) (h_give_away : give_away_puppies = 7) :
  initial_puppies - give_away_puppies = 5 :=
by
  sorry

end NUMINAMATH_GPT_Alyssa_puppies_l363_36359


namespace NUMINAMATH_GPT_verify_p_q_l363_36337

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-5, 2]]

def p : ℤ := 5
def q : ℤ := -26

theorem verify_p_q :
  N * N = p • N + q • (1 : Matrix (Fin 2) (Fin 2) ℤ) :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_verify_p_q_l363_36337


namespace NUMINAMATH_GPT_ordered_pairs_m_n_l363_36394

theorem ordered_pairs_m_n :
  ∃ (s : Finset (ℕ × ℕ)), 
  (∀ p ∈ s, p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≥ p.2 ∧ (p.1 ^ 2 - p.2 ^ 2 = 72)) ∧ s.card = 3 :=
by
  sorry

end NUMINAMATH_GPT_ordered_pairs_m_n_l363_36394


namespace NUMINAMATH_GPT_triangle_cosine_theorem_l363_36372

def triangle_sums (a b c : ℝ) : ℝ := 
  b^2 + c^2 - a^2 + a^2 + c^2 - b^2 + a^2 + b^2 - c^2

theorem triangle_cosine_theorem (a b c : ℝ) (cos_A cos_B cos_C : ℝ) :
  a = 2 → b = 3 → c = 4 → 2 * b * c * cos_A + 2 * c * a * cos_B + 2 * a * b * cos_C = 29 :=
by
  intros h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_triangle_cosine_theorem_l363_36372


namespace NUMINAMATH_GPT_travel_time_home_to_community_center_l363_36315

-- Definitions and assumptions based on the conditions
def time_to_library := 30 -- in minutes
def distance_to_library := 5 -- in miles
def time_spent_at_library := 15 -- in minutes
def distance_to_community_center := 3 -- in miles
noncomputable def cycling_speed := time_to_library / distance_to_library -- in minutes per mile

-- Time calculation to reach the community center from the library
noncomputable def time_from_library_to_community_center := distance_to_community_center * cycling_speed -- in minutes

-- Total time spent to travel from home to the community center
noncomputable def total_time_home_to_community_center :=
  time_to_library + time_spent_at_library + time_from_library_to_community_center

-- The proof statement verifying the total time
theorem travel_time_home_to_community_center : total_time_home_to_community_center = 63 := by
  sorry

end NUMINAMATH_GPT_travel_time_home_to_community_center_l363_36315


namespace NUMINAMATH_GPT_count_divisible_by_35_l363_36351

theorem count_divisible_by_35 : 
  ∃! (n : ℕ), n = 13 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 ∧ (∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ ab = 10 * a + b) →
    (ab * 100 + 35) % 35 = 0 ↔ ab % 7 = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_count_divisible_by_35_l363_36351


namespace NUMINAMATH_GPT_candy_distribution_l363_36381

theorem candy_distribution (candy : ℕ) (people : ℕ) (hcandy : candy = 30) (hpeople : people = 5) :
  ∃ k : ℕ, candy - k = people * (candy / people) ∧ k = 0 := 
by
  sorry

end NUMINAMATH_GPT_candy_distribution_l363_36381


namespace NUMINAMATH_GPT_median_CD_eq_altitude_from_C_eq_centroid_G_eq_l363_36314

namespace Geometry

/-- Vertices of the triangle -/
def A : ℝ × ℝ := (4, 4)
def B : ℝ × ℝ := (-4, 2)
def C : ℝ × ℝ := (2, 0)

/-- Proof of the equation of the median CD on the side AB -/
theorem median_CD_eq : ∀ (x y : ℝ), 3 * x + 2 * y - 6 = 0 :=
sorry

/-- Proof of the equation of the altitude from C to AB -/
theorem altitude_from_C_eq : ∀ (x y : ℝ), 4 * x + y - 8 = 0 :=
sorry

/-- Proof of the coordinates of the centroid G of triangle ABC -/
theorem centroid_G_eq : ∃ (x y : ℝ), x = 2 / 3 ∧ y = 2 :=
sorry

end Geometry

end NUMINAMATH_GPT_median_CD_eq_altitude_from_C_eq_centroid_G_eq_l363_36314


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l363_36333

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = -1) 
  (h2 : a 2 + a 3 = -2) 
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  q = -2 ∨ q = 1 := 
by sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l363_36333


namespace NUMINAMATH_GPT_turnip_mixture_l363_36349

theorem turnip_mixture (cups_potatoes total_turnips : ℕ) (h_ratio : 20 = 5 * 4) (h_turnips : total_turnips = 8) :
    cups_potatoes = 2 :=
by
    have ratio := h_ratio
    have turnips := h_turnips
    sorry

end NUMINAMATH_GPT_turnip_mixture_l363_36349


namespace NUMINAMATH_GPT_can_restore_axes_l363_36348

noncomputable def restore_axes (A : ℝ×ℝ) (hA : A.snd = 3 ^ A.fst) : Prop :=
  ∃ (B C D : ℝ×ℝ),
    (B.fst = A.fst ∧ B.snd = 0) ∧
    (C.fst = A.fst ∧ C.snd = A.snd) ∧
    (D.fst = A.fst ∧ D.snd = 3 ^ C.fst) ∧
    (∃ (extend_perpendicular : ∀ (x: ℝ), ℝ→ℝ), extend_perpendicular A.snd B.fst = D.snd)

theorem can_restore_axes (A : ℝ×ℝ) (hA : A.snd = 3 ^ A.fst) : restore_axes A hA :=
  sorry

end NUMINAMATH_GPT_can_restore_axes_l363_36348


namespace NUMINAMATH_GPT_shopkeeper_standard_weight_l363_36371

theorem shopkeeper_standard_weight
    (cost_price : ℝ)
    (actual_weight_used : ℝ)
    (profit_percentage : ℝ)
    (standard_weight : ℝ)
    (H1 : actual_weight_used = 800)
    (H2 : profit_percentage = 25) :
    standard_weight = 1000 :=
by 
    sorry

end NUMINAMATH_GPT_shopkeeper_standard_weight_l363_36371


namespace NUMINAMATH_GPT_weight_of_balls_l363_36379

theorem weight_of_balls (x y : ℕ) (h1 : 5 * x + 3 * y = 42) (h2 : 5 * y + 3 * x = 38) :
  x = 6 ∧ y = 4 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_balls_l363_36379


namespace NUMINAMATH_GPT_tan_A_tan_B_l363_36312

theorem tan_A_tan_B (A B C : ℝ) (R : ℝ) (H F : ℝ)
  (HF : H + F = 26) (h1 : 2 * R * Real.cos A * Real.cos B = 8)
  (h2 : 2 * R * Real.sin A * Real.sin B = 26) :
  Real.tan A * Real.tan B = 13 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_A_tan_B_l363_36312


namespace NUMINAMATH_GPT_Micheal_completion_time_l363_36304

variable (W M A : ℝ)

-- Conditions
def condition1 (W M A : ℝ) : Prop := M + A = W / 20
def condition2 (W M A : ℝ) : Prop := A = (W - 14 * (M + A)) / 10

-- Goal
theorem Micheal_completion_time :
  (condition1 W M A) →
  (condition2 W M A) →
  M = W / 50 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_Micheal_completion_time_l363_36304


namespace NUMINAMATH_GPT_optionA_is_multiple_of_5_optionB_is_multiple_of_5_optionC_is_multiple_of_5_optionD_is_multiple_of_5_optionE_is_not_multiple_of_5_l363_36329

-- Definitions of the options
def optionA : ℕ := 2019^2 - 2014^2
def optionB : ℕ := 2019^2 * 10^2
def optionC : ℕ := 2020^2 / 101^2
def optionD : ℕ := 2010^2 - 2005^2
def optionE : ℕ := 2015^2 / 5^2

-- Statements to be proven
theorem optionA_is_multiple_of_5 : optionA % 5 = 0 := by sorry
theorem optionB_is_multiple_of_5 : optionB % 5 = 0 := by sorry
theorem optionC_is_multiple_of_5 : optionC % 5 = 0 := by sorry
theorem optionD_is_multiple_of_5 : optionD % 5 = 0 := by sorry
theorem optionE_is_not_multiple_of_5 : optionE % 5 ≠ 0 := by sorry

end NUMINAMATH_GPT_optionA_is_multiple_of_5_optionB_is_multiple_of_5_optionC_is_multiple_of_5_optionD_is_multiple_of_5_optionE_is_not_multiple_of_5_l363_36329


namespace NUMINAMATH_GPT_gena_hits_target_l363_36353

-- Definitions from the problem conditions
def initial_shots : ℕ := 5
def total_shots : ℕ := 17
def shots_per_hit : ℕ := 2

-- Mathematical equivalent proof statement
theorem gena_hits_target (G : ℕ) (H : G * shots_per_hit + initial_shots = total_shots) : G = 6 :=
by
  sorry

end NUMINAMATH_GPT_gena_hits_target_l363_36353


namespace NUMINAMATH_GPT_max_n_for_factored_polynomial_l363_36318

theorem max_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * 72 + n = 6 * B + A ∧ A * B = 72 → n ≤ 433) ∧
             (∃ (A B : ℤ), 6 * B + A = 433 ∧ A * B = 72) :=
by sorry

end NUMINAMATH_GPT_max_n_for_factored_polynomial_l363_36318


namespace NUMINAMATH_GPT_age_problem_l363_36392

theorem age_problem :
  (∃ (x y : ℕ), 
    (3 * x - 7 = 5 * (x - 7)) ∧ 
    (42 + y = 2 * (14 + y)) ∧ 
    (2 * x = 28) ∧ 
    (x = 14) ∧ 
    (3 * 14 = 42) ∧ 
    (42 - 14 = 28) ∧ 
    (y = 14)) :=
by
  sorry

end NUMINAMATH_GPT_age_problem_l363_36392


namespace NUMINAMATH_GPT_minutes_after_2017_is_0554_l363_36382

theorem minutes_after_2017_is_0554 :
  let initial_time := (20, 17) -- time in hours and minutes
  let total_minutes := 2017
  let hours_passed := total_minutes / 60
  let minutes_passed := total_minutes % 60
  let days_passed := hours_passed / 24
  let remaining_hours := hours_passed % 24
  let resulting_hours := (initial_time.fst + remaining_hours) % 24
  let resulting_minutes := initial_time.snd + minutes_passed
  let final_hours := if resulting_minutes >= 60 then resulting_hours + 1 else resulting_hours
  let final_minutes := if resulting_minutes >= 60 then resulting_minutes - 60 else resulting_minutes
  final_hours % 24 = 5 ∧ final_minutes = 54 := by
  sorry

end NUMINAMATH_GPT_minutes_after_2017_is_0554_l363_36382


namespace NUMINAMATH_GPT_no_natural_numbers_satisfy_conditions_l363_36377

theorem no_natural_numbers_satisfy_conditions : 
  ¬ ∃ (a b : ℕ), 
    (∃ (k : ℕ), k^2 = a^2 + 2 * b^2) ∧ 
    (∃ (m : ℕ), m^2 = b^2 + 2 * a) :=
by {
  -- Proof steps and logical deductions can be written here.
  sorry
}

end NUMINAMATH_GPT_no_natural_numbers_satisfy_conditions_l363_36377


namespace NUMINAMATH_GPT_hexagon_colorings_correct_l363_36339

def valid_hexagon_colorings : Prop :=
  ∃ (colors : Fin 6 → Fin 7),
    (colors 0 ≠ colors 1) ∧
    (colors 1 ≠ colors 2) ∧
    (colors 2 ≠ colors 3) ∧
    (colors 3 ≠ colors 4) ∧
    (colors 4 ≠ colors 5) ∧
    (colors 5 ≠ colors 0) ∧
    (colors 0 ≠ colors 2) ∧
    (colors 1 ≠ colors 3) ∧
    (colors 2 ≠ colors 4) ∧
    (colors 3 ≠ colors 5) ∧
    ∃! (n : Nat), n = 12600

theorem hexagon_colorings_correct : valid_hexagon_colorings :=
sorry

end NUMINAMATH_GPT_hexagon_colorings_correct_l363_36339


namespace NUMINAMATH_GPT_initial_sentences_today_l363_36360

-- Definitions of the given conditions
def typing_rate : ℕ := 6
def initial_typing_time : ℕ := 20
def additional_typing_time : ℕ := 15
def erased_sentences : ℕ := 40
def post_meeting_typing_time : ℕ := 18
def total_sentences_end_of_day : ℕ := 536

def sentences_typed_before_break := initial_typing_time * typing_rate
def sentences_typed_after_break := additional_typing_time * typing_rate
def sentences_typed_post_meeting := post_meeting_typing_time * typing_rate
def sentences_today := sentences_typed_before_break + sentences_typed_after_break - erased_sentences + sentences_typed_post_meeting

theorem initial_sentences_today : total_sentences_end_of_day - sentences_today = 258 := by
  -- proof here
  sorry

end NUMINAMATH_GPT_initial_sentences_today_l363_36360


namespace NUMINAMATH_GPT_geometric_sequence_a8_eq_pm1_l363_36344

variable {R : Type*} [LinearOrderedField R]

theorem geometric_sequence_a8_eq_pm1 :
  ∀ (a : ℕ → R), (∀ n : ℕ, ∃ r : R, r ≠ 0 ∧ a n = a 0 * r ^ n) → 
  (a 4 + a 12 = -3) ∧ (a 4 * a 12 = 1) → 
  (a 8 = 1 ∨ a 8 = -1) := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a8_eq_pm1_l363_36344


namespace NUMINAMATH_GPT_surface_area_of_sphere_l363_36398

/-- Given a right prism with all vertices on a sphere, a height of 4, and a volume of 64,
    the surface area of this sphere is 48π -/
theorem surface_area_of_sphere (h : ℝ) (V : ℝ) (S : ℝ) :
  h = 4 → V = 64 → S = 48 * Real.pi := by
  sorry

end NUMINAMATH_GPT_surface_area_of_sphere_l363_36398


namespace NUMINAMATH_GPT_simplify_and_evaluate_l363_36388

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_and_evaluate :
  ( (x + 3) / x - 1 ) / ( (x^2 - 1) / (x^2 + x) ) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l363_36388


namespace NUMINAMATH_GPT_initial_percentage_water_l363_36387

theorem initial_percentage_water (W_initial W_final N_initial N_final : ℝ) (h1 : W_initial = 100) 
    (h2 : N_initial = W_initial - W_final) (h3 : W_final = 25) (h4 : W_final / N_final = 0.96) : N_initial / W_initial = 0.99 := 
by
  sorry

end NUMINAMATH_GPT_initial_percentage_water_l363_36387


namespace NUMINAMATH_GPT_abs_value_expression_l363_36362

theorem abs_value_expression (m n : ℝ) (h1 : m < 0) (h2 : m * n < 0) :
  |n - m + 1| - |m - n - 5| = -4 :=
sorry

end NUMINAMATH_GPT_abs_value_expression_l363_36362


namespace NUMINAMATH_GPT_solve_system_l363_36397

theorem solve_system :
  (∃ x y : ℝ, 4 * x - 3 * y = -3 ∧ 8 * x + 5 * y = 11 + x ^ 2 ∧ (x, y) = (14.996, 19.994)) ∨
  (∃ x y : ℝ, 4 * x - 3 * y = -3 ∧ 8 * x + 5 * y = 11 + x ^ 2 ∧ (x, y) = (0.421, 1.561)) :=
  sorry

end NUMINAMATH_GPT_solve_system_l363_36397


namespace NUMINAMATH_GPT_particle_speed_interval_l363_36356

noncomputable def particle_position (t : ℝ) : ℝ × ℝ :=
  (3 * t + 5, 5 * t - 7)

theorem particle_speed_interval (k : ℝ) :
  let start_pos := particle_position k
  let end_pos := particle_position (k + 2)
  let delta_x := end_pos.1 - start_pos.1
  let delta_y := end_pos.2 - start_pos.2
  let speed := Real.sqrt (delta_x^2 + delta_y^2)
  speed = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_GPT_particle_speed_interval_l363_36356


namespace NUMINAMATH_GPT_fraction_pos_integer_l363_36307

theorem fraction_pos_integer (p : ℕ) (hp : 0 < p) : (∃ (k : ℕ), k = 1 + (2 * p + 53) / (3 * p - 8)) ↔ p = 3 := 
by
  sorry

end NUMINAMATH_GPT_fraction_pos_integer_l363_36307


namespace NUMINAMATH_GPT_determine_n_between_sqrt3_l363_36355

theorem determine_n_between_sqrt3 (n : ℕ) (hpos : 0 < n)
  (hineq : (n + 3) / n < Real.sqrt 3 ∧ Real.sqrt 3 < (n + 4) / (n + 1)) :
  n = 4 :=
sorry

end NUMINAMATH_GPT_determine_n_between_sqrt3_l363_36355


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l363_36338

noncomputable def necessary_but_not_sufficient_condition (x : ℝ) : Prop :=
  -2 < x ∧ x < 3 → x^2 - 2 * x - 3 < 0

theorem necessary_but_not_sufficient (x : ℝ) : 
-2 < x ∧ x < 3 → x^2 - 2 * x - 3 < 0 := 
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l363_36338


namespace NUMINAMATH_GPT_possible_measures_of_angle_X_l363_36345

theorem possible_measures_of_angle_X : 
  ∃ n : ℕ, n = 17 ∧ (∀ (X Y : ℕ), 
    X > 0 ∧ Y > 0 ∧ X + Y = 180 ∧ 
    ∃ m : ℕ, m ≥ 1 ∧ X = m * Y) :=
sorry

end NUMINAMATH_GPT_possible_measures_of_angle_X_l363_36345


namespace NUMINAMATH_GPT_ordered_pairs_count_l363_36330

theorem ordered_pairs_count :
  (∃ (a b : ℝ), (∃ (x y : ℤ),
    a * (x : ℝ) + b * (y : ℝ) = 1 ∧
    (x : ℝ)^2 + (y : ℝ)^2 = 65)) →
  ∃ (n : ℕ), n = 128 :=
by
  sorry

end NUMINAMATH_GPT_ordered_pairs_count_l363_36330


namespace NUMINAMATH_GPT_commutating_matrices_l363_36313

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=  ![![2, 3], ![4, 5]]
noncomputable def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![x, y], ![z, w]]

theorem commutating_matrices (x y z w : ℝ) (h1 : A * (B x y z w) = (B x y z w) * A) (h2 : 4 * y ≠ z) : 
  (x - w) / (z - 4 * y) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_commutating_matrices_l363_36313


namespace NUMINAMATH_GPT_part1_part2_part3_l363_36393

-- Part 1
theorem part1 (B_count : ℕ) : 
  (1 * 100) + (B_count * 68) + (4 * 20) = 520 → 
  B_count = 5 := 
by sorry

-- Part 2
theorem part2 (A_count B_count : ℕ) : 
  A_count + B_count = 5 → 
  (100 * A_count) + (68 * B_count) = 404 → 
  A_count = 2 ∧ B_count = 3 := 
by sorry

-- Part 3
theorem part3 : 
  ∃ (A_count B_count C_count : ℕ), 
  (A_count <= 16) ∧ (B_count <= 16) ∧ (C_count <= 16) ∧ 
  (A_count + B_count + C_count <= 16) ∧ 
  (100 * A_count + 68 * B_count = 708 ∨ 
   68 * B_count + 20 * C_count = 708 ∨ 
   100 * A_count + 20 * C_count = 708) → 
  ((A_count = 3 ∧ B_count = 6 ∧ C_count = 0) ∨ 
   (A_count = 0 ∧ B_count = 6 ∧ C_count = 15)) := 
by sorry

end NUMINAMATH_GPT_part1_part2_part3_l363_36393


namespace NUMINAMATH_GPT_total_cost_is_67_15_l363_36352

noncomputable def calculate_total_cost : ℝ :=
  let caramel_cost := 3
  let candy_bar_cost := 2 * caramel_cost
  let cotton_candy_cost := (candy_bar_cost * 4) / 2
  let chocolate_bar_cost := candy_bar_cost + caramel_cost
  let lollipop_cost := candy_bar_cost / 3

  let candy_bar_total := 6 * candy_bar_cost
  let caramel_total := 3 * caramel_cost
  let cotton_candy_total := 1 * cotton_candy_cost
  let chocolate_bar_total := 2 * chocolate_bar_cost
  let lollipop_total := 2 * lollipop_cost

  let discounted_candy_bar_total := candy_bar_total * 0.9
  let discounted_caramel_total := caramel_total * 0.85
  let discounted_cotton_candy_total := cotton_candy_total * 0.8
  let discounted_chocolate_bar_total := chocolate_bar_total * 0.75
  let discounted_lollipop_total := lollipop_total -- No additional discount

  discounted_candy_bar_total +
  discounted_caramel_total +
  discounted_cotton_candy_total +
  discounted_chocolate_bar_total +
  discounted_lollipop_total

theorem total_cost_is_67_15 : calculate_total_cost = 67.15 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_67_15_l363_36352


namespace NUMINAMATH_GPT_no_int_b_exists_l363_36385

theorem no_int_b_exists (k n a : ℕ) (hk3 : k ≥ 3) (hn3 : n ≥ 3) (hk_odd : k % 2 = 1) (hn_odd : n % 2 = 1)
  (ha1 : a ≥ 1) (hka : k ∣ (2^a + 1)) (hna : n ∣ (2^a - 1)) :
  ¬ ∃ b : ℕ, b ≥ 1 ∧ k ∣ (2^b - 1) ∧ n ∣ (2^b + 1) :=
sorry

end NUMINAMATH_GPT_no_int_b_exists_l363_36385


namespace NUMINAMATH_GPT_reciprocal_real_roots_l363_36399

theorem reciprocal_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 * x2 = 1 ∧ x1 + x2 = 2 * (m + 2)) ∧ 
  (x1^2 - 2 * (m + 2) * x1 + (m^2 - 4) = 0) → m = Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_reciprocal_real_roots_l363_36399


namespace NUMINAMATH_GPT_cosine_of_five_pi_over_three_l363_36396

theorem cosine_of_five_pi_over_three :
  Real.cos (5 * Real.pi / 3) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_cosine_of_five_pi_over_three_l363_36396


namespace NUMINAMATH_GPT_curves_intersection_four_points_l363_36331

theorem curves_intersection_four_points (b : ℝ) :
  (∃ x1 x2 x3 x4 y1 y2 y3 y4 : ℝ,
    x1^2 + y1^2 = b^2 ∧ y1 = x1^2 - b + 1 ∧
    x2^2 + y2^2 = b^2 ∧ y2 = x2^2 - b + 1 ∧
    x3^2 + y3^2 = b^2 ∧ y3 = x3^2 - b + 1 ∧
    x4^2 + y4^2 = b^2 ∧ y4 = x4^2 - b + 1 ∧
    (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x1, y1) ≠ (x4, y4) ∧
    (x2, y2) ≠ (x3, y3) ∧ (x2, y2) ≠ (x4, y4) ∧
    (x3, y3) ≠ (x4, y4)) →
  b > 2 :=
sorry

end NUMINAMATH_GPT_curves_intersection_four_points_l363_36331


namespace NUMINAMATH_GPT_central_angle_star_in_polygon_l363_36350

theorem central_angle_star_in_polygon (n : ℕ) (h : 2 < n) : 
  ∃ C, C = 720 / n :=
by sorry

end NUMINAMATH_GPT_central_angle_star_in_polygon_l363_36350


namespace NUMINAMATH_GPT_problem_value_l363_36328

theorem problem_value :
  4 * (8 - 3) / 2 - 7 = 3 := 
by
  sorry

end NUMINAMATH_GPT_problem_value_l363_36328


namespace NUMINAMATH_GPT_calories_per_cookie_l363_36341

theorem calories_per_cookie (C : ℝ) (h1 : ∀ cracker, cracker = 15)
    (h2 : ∀ cookie, cookie = C)
    (h3 : 7 * C + 10 * 15 = 500) :
    C = 50 :=
  by
    sorry

end NUMINAMATH_GPT_calories_per_cookie_l363_36341


namespace NUMINAMATH_GPT_hamburger_cost_l363_36390

def annie's_starting_money : ℕ := 120
def num_hamburgers_bought : ℕ := 8
def price_milkshake : ℕ := 3
def num_milkshakes_bought : ℕ := 6
def leftover_money : ℕ := 70

theorem hamburger_cost :
  ∃ (H : ℕ), 8 * H + 6 * price_milkshake = annie's_starting_money - leftover_money ∧ H = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_hamburger_cost_l363_36390


namespace NUMINAMATH_GPT_triangle_length_product_square_l363_36300

theorem triangle_length_product_square 
  (a1 : ℝ) (b1 : ℝ) (c1 : ℝ) (a2 : ℝ) (b2 : ℝ) (c2 : ℝ) 
  (h1 : a1 * b1 / 2 = 3)
  (h2 : a2 * b2 / 2 = 4)
  (h3 : a1 = a2)
  (h4 : c1 = 2 * c2) 
  (h5 : c1^2 = a1^2 + b1^2)
  (h6 : c2^2 = a2^2 + b2^2) :
  (b1 * b2)^2 = (2304 / 25 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_triangle_length_product_square_l363_36300


namespace NUMINAMATH_GPT_min_value_of_function_l363_36309

open Real

theorem min_value_of_function (x y : ℝ) (h : 2 * x + 8 * y = 3) : ∃ (min_value : ℝ), min_value = -19 / 20 ∧ ∀ (x y : ℝ), 2 * x + 8 * y = 3 → x^2 + 4 * y^2 - 2 * x ≥ -19 / 20 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_function_l363_36309


namespace NUMINAMATH_GPT_g_at_8_l363_36354

def g (x : ℝ) : ℝ := sorry

axiom g_property : ∀ x y : ℝ, x * g y = y * g x

axiom g_at_24 : g 24 = 12

theorem g_at_8 : g 8 = 4 := by
  sorry

end NUMINAMATH_GPT_g_at_8_l363_36354


namespace NUMINAMATH_GPT_constant_function_of_inequality_l363_36380

theorem constant_function_of_inequality
  (f : ℤ → ℝ)
  (h_bound : ∃ M : ℝ, ∀ n : ℤ, f n ≤ M)
  (h_ineq : ∀ n : ℤ, f n ≤ (f (n - 1) + f (n + 1)) / 2) :
  ∀ m n : ℤ, f m = f n := by
  sorry

end NUMINAMATH_GPT_constant_function_of_inequality_l363_36380


namespace NUMINAMATH_GPT_value_of_g_l363_36395

-- Defining the function g and its property
def g (x : ℝ) : ℝ := 5

-- Theorem to prove g(x - 3) = 5 for any real number x
theorem value_of_g (x : ℝ) : g (x - 3) = 5 := by
  sorry

end NUMINAMATH_GPT_value_of_g_l363_36395


namespace NUMINAMATH_GPT_inequality_proof_l363_36391

variable {x1 x2 y1 y2 z1 z2 : ℝ}

theorem inequality_proof (hx1 : x1 > 0) (hx2 : x2 > 0)
   (hxy1 : x1 * y1 - z1^2 > 0) (hxy2 : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
  sorry

end NUMINAMATH_GPT_inequality_proof_l363_36391


namespace NUMINAMATH_GPT_fence_poles_placement_l363_36358

def total_bridges_length (bridges : List ℕ) : ℕ :=
  bridges.sum

def effective_path_length (path_length : ℕ) (bridges_length : ℕ) : ℕ :=
  path_length - bridges_length

def poles_on_one_side (effective_length : ℕ) (interval : ℕ) : ℕ :=
  effective_length / interval

def total_poles (path_length : ℕ) (interval : ℕ) (bridges : List ℕ) : ℕ :=
  let bridges_length := total_bridges_length bridges
  let effective_length := effective_path_length path_length bridges_length
  let poles_one_side := poles_on_one_side effective_length interval
  2 * poles_one_side + 2

theorem fence_poles_placement :
  total_poles 2300 8 [48, 58, 62] = 534 := by
  sorry

end NUMINAMATH_GPT_fence_poles_placement_l363_36358


namespace NUMINAMATH_GPT_tan_alpha_through_point_l363_36322

theorem tan_alpha_through_point (α : ℝ) (x y : ℝ) (h : (x, y) = (3, 4)) : Real.tan α = 4 / 3 :=
sorry

end NUMINAMATH_GPT_tan_alpha_through_point_l363_36322


namespace NUMINAMATH_GPT_ratio_of_screams_to_hours_l363_36389

-- Definitions from conditions
def hours_hired : ℕ := 6
def current_babysitter_rate : ℕ := 16
def new_babysitter_rate : ℕ := 12
def extra_charge_per_scream : ℕ := 3
def cost_difference : ℕ := 18

-- Calculate necessary costs
def current_babysitter_cost : ℕ := current_babysitter_rate * hours_hired
def new_babysitter_base_cost : ℕ := new_babysitter_rate * hours_hired
def new_babysitter_total_cost : ℕ := current_babysitter_cost - cost_difference
def screams_cost : ℕ := new_babysitter_total_cost - new_babysitter_base_cost
def number_of_screams : ℕ := screams_cost / extra_charge_per_scream

-- Theorem to prove the ratio
theorem ratio_of_screams_to_hours : number_of_screams / hours_hired = 1 := by
  sorry

end NUMINAMATH_GPT_ratio_of_screams_to_hours_l363_36389


namespace NUMINAMATH_GPT_parabola_directrix_l363_36325

theorem parabola_directrix (x y : ℝ) (h : x^2 = 8 * y) : y = -2 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l363_36325


namespace NUMINAMATH_GPT_yunjeong_locker_problem_l363_36361

theorem yunjeong_locker_problem
  (l r f b : ℕ)
  (h_l : l = 7)
  (h_r : r = 13)
  (h_f : f = 8)
  (h_b : b = 14)
  (same_rows : ∀ pos1 pos2 : ℕ, pos1 = pos2) :
  (l - 1) + (r - 1) + (f - 1) + (b - 1) = 399 := sorry

end NUMINAMATH_GPT_yunjeong_locker_problem_l363_36361


namespace NUMINAMATH_GPT_x_squared_y_cubed_eq_200_l363_36384

theorem x_squared_y_cubed_eq_200 (x y : ℕ) (h : 2^x * 9^y = 200) : x^2 * y^3 = 200 := by
  sorry

end NUMINAMATH_GPT_x_squared_y_cubed_eq_200_l363_36384


namespace NUMINAMATH_GPT_marbles_remainder_l363_36310

theorem marbles_remainder 
  (g r p : ℕ) 
  (hg : g % 8 = 5) 
  (hr : r % 7 = 2) 
  (hp : p % 7 = 4) : 
  (r + p + g) % 7 = 4 := 
sorry

end NUMINAMATH_GPT_marbles_remainder_l363_36310


namespace NUMINAMATH_GPT_max_value_at_log2_one_l363_36340

noncomputable def f (x : ℝ) : ℝ := 2 * x + 2 - 3 * (4 : ℝ) ^ x
def domain (x : ℝ) : Prop := x < 1 ∨ x > 3

theorem max_value_at_log2_one :
  (∃ x, domain x ∧ f x = 0) ∧ (∀ y, domain y → f y ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_max_value_at_log2_one_l363_36340


namespace NUMINAMATH_GPT_minimum_expression_l363_36335

variable (a b : ℝ)

theorem minimum_expression (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 3) :
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 3 ∧ (∀ x y : ℝ, 0 < x → 0 < y → x + y = 3 → 
  x = a ∧ y = b  → ∃ m : ℝ, m ≥ 1 ∧ (m = (1/(a+1)) + 1/b))) := sorry

end NUMINAMATH_GPT_minimum_expression_l363_36335


namespace NUMINAMATH_GPT_compute_f_g_2_l363_36370

def f (x : ℝ) : ℝ := 5 - 4 * x
def g (x : ℝ) : ℝ := x^2 + 2

theorem compute_f_g_2 : f (g 2) = -19 := 
by {
  sorry
}

end NUMINAMATH_GPT_compute_f_g_2_l363_36370


namespace NUMINAMATH_GPT_find_ab_l363_36323

variables (a b c : ℝ)

-- Defining the conditions
def cond1 : Prop := a - b = 5
def cond2 : Prop := a^2 + b^2 = 34
def cond3 : Prop := a^3 - b^3 = 30
def cond4 : Prop := a^2 + b^2 - c^2 = 50

theorem find_ab (h1 : cond1 a b) (h2 : cond2 a b) (h3 : cond3 a b) (h4 : cond4 a b c) :
  a * b = 4.5 :=
sorry

end NUMINAMATH_GPT_find_ab_l363_36323


namespace NUMINAMATH_GPT_negation_P1_is_false_negation_P2_is_false_l363_36332

-- Define the propositions
def isMultiDigitNumber (n : ℕ) : Prop := n >= 10
def lastDigitIsZero (n : ℕ) : Prop := n % 10 = 0
def isMultipleOfFive (n : ℕ) : Prop := n % 5 = 0
def isEven (n : ℕ) : Prop := n % 2 = 0

-- The propositions
def P1 (n : ℕ) : Prop := isMultiDigitNumber n → (lastDigitIsZero n → isMultipleOfFive n)
def P2 : Prop := ∀ n, isEven n → n % 2 = 0

-- The negations
def notP1 (n : ℕ) : Prop := isMultiDigitNumber n ∧ lastDigitIsZero n → ¬isMultipleOfFive n
def notP2 : Prop := ∃ n, isEven n ∧ ¬(n % 2 = 0)

-- The proof problems
theorem negation_P1_is_false (n : ℕ) : notP1 n → False := by
  sorry

theorem negation_P2_is_false : notP2 → False := by
  sorry

end NUMINAMATH_GPT_negation_P1_is_false_negation_P2_is_false_l363_36332


namespace NUMINAMATH_GPT_scientific_notation_of_274M_l363_36303

theorem scientific_notation_of_274M :
  274000000 = 2.74 * 10^8 := 
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_of_274M_l363_36303


namespace NUMINAMATH_GPT_value_of_f_at_7_l363_36368

theorem value_of_f_at_7
  (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_periodic : ∀ x, f (x + 4) = f x)
  (h_definition : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) :
  f 7 = 2 :=
by
  -- Proof will be filled here
  sorry

end NUMINAMATH_GPT_value_of_f_at_7_l363_36368


namespace NUMINAMATH_GPT_train_and_car_combined_time_l363_36363

noncomputable def combined_time (car_time : ℝ) (extra_time : ℝ) : ℝ :=
  car_time + (car_time + extra_time)

theorem train_and_car_combined_time : 
  ∀ (car_time : ℝ) (extra_time : ℝ), car_time = 4.5 → extra_time = 2.0 → combined_time car_time extra_time = 11 :=
by
  intros car_time extra_time hcar hextra
  sorry

end NUMINAMATH_GPT_train_and_car_combined_time_l363_36363


namespace NUMINAMATH_GPT_degrees_for_lemon_pie_l363_36324

theorem degrees_for_lemon_pie 
    (total_students : ℕ)
    (chocolate_lovers : ℕ)
    (apple_lovers : ℕ)
    (blueberry_lovers : ℕ)
    (remaining_students : ℕ)
    (lemon_pie_degrees : ℝ) :
    total_students = 42 →
    chocolate_lovers = 15 →
    apple_lovers = 9 →
    blueberry_lovers = 7 →
    remaining_students = total_students - (chocolate_lovers + apple_lovers + blueberry_lovers) →
    lemon_pie_degrees = (remaining_students / 2 / total_students * 360) →
    lemon_pie_degrees = 47.14 :=
by
  intros _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_degrees_for_lemon_pie_l363_36324


namespace NUMINAMATH_GPT_jogging_walking_ratio_l363_36317

theorem jogging_walking_ratio (total_time walk_time jog_time: ℕ) (h1 : total_time = 21) (h2 : walk_time = 9) (h3 : jog_time = total_time - walk_time) :
  (jog_time : ℚ) / walk_time = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_jogging_walking_ratio_l363_36317


namespace NUMINAMATH_GPT_budget_spent_on_research_and_development_l363_36367

theorem budget_spent_on_research_and_development:
  (∀ budget_total : ℝ, budget_total > 0) →
  (∀ transportation : ℝ, transportation = 15) →
  (∃ research_and_development : ℝ, research_and_development ≥ 0) →
  (∀ utilities : ℝ, utilities = 5) →
  (∀ equipment : ℝ, equipment = 4) →
  (∀ supplies : ℝ, supplies = 2) →
  (∀ salaries_degrees : ℝ, salaries_degrees = 234) →
  (∀ total_degrees : ℝ, total_degrees = 360) →
  (∀ percentage_salaries : ℝ, percentage_salaries = (salaries_degrees / total_degrees) * 100) →
  (∀ known_percentages : ℝ, known_percentages = transportation + utilities + equipment + supplies + percentage_salaries) →
  (∀ rnd_percent : ℝ, rnd_percent = 100 - known_percentages) →
  (rnd_percent = 9) :=
  sorry

end NUMINAMATH_GPT_budget_spent_on_research_and_development_l363_36367


namespace NUMINAMATH_GPT_max_value_of_sin_l363_36376

theorem max_value_of_sin (x : ℝ) : (2 * Real.sin x) ≤ 2 :=
by
  -- this theorem directly implies that 2sin(x) has a maximum value of 2.
  sorry

end NUMINAMATH_GPT_max_value_of_sin_l363_36376


namespace NUMINAMATH_GPT_trapezoid_possible_and_area_sum_l363_36373

theorem trapezoid_possible_and_area_sum (a b c d : ℕ) (h1 : a = 4) (h2 : b = 6) (h3 : c = 8) (h4 : d = 12) :
  ∃ (S : ℚ), S = 72 := 
by
  -- conditions ensure one pair of sides is parallel
  -- area calculation based on trapezoid properties
  sorry

end NUMINAMATH_GPT_trapezoid_possible_and_area_sum_l363_36373


namespace NUMINAMATH_GPT_simple_interest_years_l363_36374

noncomputable def simple_interest (P r t : ℕ) : ℕ :=
  P * r * t / 100

noncomputable def compound_interest (P r n : ℕ) : ℕ :=
  P * (1 + r / 100)^n - P

theorem simple_interest_years
  (P_si r_si P_ci r_ci n_ci si_half_ci si_si : ℕ)
  (h_si : simple_interest P_si r_si si_si = si_half_ci)
  (h_ci : compound_interest P_ci r_ci n_ci = si_half_ci * 2) :
  si_si = 2 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_years_l363_36374


namespace NUMINAMATH_GPT_triangle_inequality_example_l363_36346

theorem triangle_inequality_example {x : ℝ} (h1: 3 + 4 > x) (h2: abs (3 - 4) < x) : 1 < x ∧ x < 7 :=
  sorry

end NUMINAMATH_GPT_triangle_inequality_example_l363_36346


namespace NUMINAMATH_GPT_modular_inverse_of_17_mod_800_l363_36311

    theorem modular_inverse_of_17_mod_800 :
      ∃ x : ℤ, 0 ≤ x ∧ x < 800 ∧ (17 * x) % 800 = 1 :=
    by
      use 47
      sorry
    
end NUMINAMATH_GPT_modular_inverse_of_17_mod_800_l363_36311


namespace NUMINAMATH_GPT_find_starting_number_l363_36308

theorem find_starting_number (x : ℝ) (h : ((x - 2 + 4) / 1) / 2 * 8 = 77) : x = 17.25 := by
  sorry

end NUMINAMATH_GPT_find_starting_number_l363_36308


namespace NUMINAMATH_GPT_num_tents_needed_l363_36375

def count_people : ℕ :=
  let matts_family := 1 + 1 + 1 + 1 + 1 + 4 + 1 + 1 + 2 + 2
  let joes_family := 1 + 1 + 3 + 1
  matts_family + joes_family

def house_capacity : ℕ := 6

def tent_capacity : ℕ := 2

theorem num_tents_needed : (count_people - house_capacity) / tent_capacity = 7 := by
  sorry

end NUMINAMATH_GPT_num_tents_needed_l363_36375


namespace NUMINAMATH_GPT_number_of_shirts_made_today_l363_36321

-- Define the rate of shirts made per minute.
def shirts_per_minute : ℕ := 6

-- Define the number of minutes the machine worked today.
def minutes_today : ℕ := 12

-- Define the total number of shirts made today.
def shirts_made_today : ℕ := shirts_per_minute * minutes_today

-- State the theorem for the number of shirts made today.
theorem number_of_shirts_made_today : shirts_made_today = 72 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_number_of_shirts_made_today_l363_36321


namespace NUMINAMATH_GPT_square_prism_surface_area_eq_volume_l363_36386

theorem square_prism_surface_area_eq_volume :
  ∃ (a b : ℕ), (a > 0) ∧ (2 * a^2 + 4 * a * b = a^2 * b)
  ↔ (a = 12 ∧ b = 3) ∨ (a = 8 ∧ b = 4) ∨ (a = 6 ∧ b = 6) ∨ (a = 5 ∧ b = 10) :=
by
  sorry

end NUMINAMATH_GPT_square_prism_surface_area_eq_volume_l363_36386


namespace NUMINAMATH_GPT_points_in_quadrants_l363_36306

theorem points_in_quadrants (x y : ℝ) (h₁ : y > 3 * x) (h₂ : y > 6 - x) : 
  (0 <= x ∧ 0 <= y) ∨ (x <= 0 ∧ 0 <= y) :=
by
  sorry

end NUMINAMATH_GPT_points_in_quadrants_l363_36306


namespace NUMINAMATH_GPT_stuffed_dogs_count_l363_36383

theorem stuffed_dogs_count (D : ℕ) (h1 : 14 + D % 7 = 0) : D = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_stuffed_dogs_count_l363_36383


namespace NUMINAMATH_GPT_correct_answer_is_B_l363_36365

def lack_of_eco_friendly_habits : Prop := true
def major_global_climate_change_cause (s : String) : Prop :=
  s = "cause"

theorem correct_answer_is_B :
  major_global_climate_change_cause "cause" ∧ lack_of_eco_friendly_habits → "B" = "cause" :=
by
  sorry

end NUMINAMATH_GPT_correct_answer_is_B_l363_36365


namespace NUMINAMATH_GPT_original_number_doubled_added_trebled_l363_36316

theorem original_number_doubled_added_trebled (x : ℤ) : 3 * (2 * x + 9) = 75 → x = 8 :=
by
  intro h
  -- The proof is omitted as instructed.
  sorry

end NUMINAMATH_GPT_original_number_doubled_added_trebled_l363_36316


namespace NUMINAMATH_GPT_large_pizza_slices_l363_36302

-- Definitions and conditions based on the given problem
def slicesEatenByPhilAndre : ℕ := 9 * 2
def slicesLeft : ℕ := 2 * 2
def slicesOnSmallCheesePizza : ℕ := 8
def totalSlices : ℕ := slicesEatenByPhilAndre + slicesLeft

-- The theorem to be proven
theorem large_pizza_slices (slicesEatenByPhilAndre slicesLeft slicesOnSmallCheesePizza : ℕ) :
  slicesEatenByPhilAndre = 18 ∧ slicesLeft = 4 ∧ slicesOnSmallCheesePizza = 8 →
  totalSlices - slicesOnSmallCheesePizza = 14 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_large_pizza_slices_l363_36302


namespace NUMINAMATH_GPT_roots_lost_extraneous_roots_l363_36364

noncomputable def f1 (x : ℝ) := Real.arcsin x
noncomputable def g1 (x : ℝ) := 2 * Real.arcsin (x / Real.sqrt 2)
noncomputable def f2 (x : ℝ) := x
noncomputable def g2 (x : ℝ) := 2 * x

theorem roots_lost :
  ∃ x : ℝ, f1 x = g1 x ∧ ¬ ∃ y : ℝ, Real.tan (f1 y) = Real.tan (g1 y) :=
sorry

theorem extraneous_roots :
  ∃ x : ℝ, ¬ f2 x = g2 x ∧ ∃ y : ℝ, Real.tan (f2 y) = Real.tan (g2 y) :=
sorry

end NUMINAMATH_GPT_roots_lost_extraneous_roots_l363_36364


namespace NUMINAMATH_GPT_polynomial_evaluation_l363_36301

theorem polynomial_evaluation (x : ℝ) (h1 : x^2 - 3 * x - 10 = 0) (h2 : 0 < x) : 
  x^3 - 3 * x^2 - 10 * x + 5 = 5 :=
sorry

end NUMINAMATH_GPT_polynomial_evaluation_l363_36301
