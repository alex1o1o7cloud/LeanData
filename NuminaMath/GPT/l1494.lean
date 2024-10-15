import Mathlib

namespace NUMINAMATH_GPT_largest_n_unique_k_l1494_149445

theorem largest_n_unique_k : 
  ∃ n : ℕ, (∀ k : ℤ, (5 / 12 : ℚ) < n / (n + k) ∧ n / (n + k) < (4 / 9 : ℚ) → k = 9) ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_unique_k_l1494_149445


namespace NUMINAMATH_GPT_expand_expression_l1494_149497

variable (x y z : ℝ)

theorem expand_expression :
  (x + 8) * (3 * y + 12) * (2 * z + 4) =
  6 * x * y * z + 12 * x * z + 24 * y * z + 12 * x * y + 48 * x + 96 * y + 96 * z + 384 := 
  sorry

end NUMINAMATH_GPT_expand_expression_l1494_149497


namespace NUMINAMATH_GPT_find_a15_l1494_149411

namespace ArithmeticSequence

variable {a : ℕ → ℝ}

def arithmetic_sequence (an : ℕ → ℝ) := ∃ (a₁ d : ℝ), ∀ n, an n = a₁ + (n - 1) * d

theorem find_a15
  (h_seq : arithmetic_sequence a)
  (h_a3 : a 3 = 4)
  (h_a9 : a 9 = 10) :
  a 15 = 16 := 
sorry

end ArithmeticSequence

end NUMINAMATH_GPT_find_a15_l1494_149411


namespace NUMINAMATH_GPT_greatest_x_value_l1494_149498

theorem greatest_x_value :
  ∃ x : ℝ, (x ≠ 2 ∧ (x^2 - 5 * x - 14) / (x - 2) = 4 / (x + 4)) ∧ x = -2 ∧ 
           ∀ y, (y ≠ 2 ∧ (y^2 - 5 * y - 14) / (y - 2) = 4 / (y + 4)) → y ≤ x :=
by
  sorry

end NUMINAMATH_GPT_greatest_x_value_l1494_149498


namespace NUMINAMATH_GPT_xiao_ming_polygon_l1494_149470

theorem xiao_ming_polygon (n : ℕ) (h : (n - 2) * 180 = 2185) : n = 14 :=
by sorry

end NUMINAMATH_GPT_xiao_ming_polygon_l1494_149470


namespace NUMINAMATH_GPT_altitude_of_isosceles_triangle_l1494_149469

noncomputable def radius_X (C : ℝ) := C / (2 * Real.pi)
noncomputable def radius_Y (radius_X : ℝ) := radius_X
noncomputable def a (radius_Y : ℝ) := radius_Y / 2

-- Define the theorem to be proven
theorem altitude_of_isosceles_triangle (C : ℝ) (h_C : C = 14 * Real.pi) (radius_X := radius_X C) (radius_Y := radius_Y radius_X) (a := a radius_Y) :
  ∃ h : ℝ, h = a * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_altitude_of_isosceles_triangle_l1494_149469


namespace NUMINAMATH_GPT_range_of_a_l1494_149434

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (x^2 + (a + 2) * x + 1) * ((3 - 2 * a) * x^2 + 5 * x + (3 - 2 * a)) ≥ 0) : a ∈ Set.Icc (-4 : ℝ) 0 := sorry

end NUMINAMATH_GPT_range_of_a_l1494_149434


namespace NUMINAMATH_GPT_solve_y_l1494_149463

theorem solve_y (y : ℚ) (h : (3 * y) / 7 = 14) : y = 98 / 3 := 
by sorry

end NUMINAMATH_GPT_solve_y_l1494_149463


namespace NUMINAMATH_GPT_age_difference_l1494_149415

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 11) : C + 11 = A :=
by {
  sorry
}

end NUMINAMATH_GPT_age_difference_l1494_149415


namespace NUMINAMATH_GPT_line_through_point_and_parallel_l1494_149429

def point_A : ℝ × ℝ × ℝ := (-2, 3, 1)

def plane1 (x y z : ℝ) := x - 2*y - z - 2 = 0
def plane2 (x y z : ℝ) := 2*x + 3*y - z + 1 = 0

theorem line_through_point_and_parallel (x y z t : ℝ) :
  ∃ t, 
    x = 5 * t - 2 ∧
    y = -t + 3 ∧
    z = 7 * t + 1 :=
sorry

end NUMINAMATH_GPT_line_through_point_and_parallel_l1494_149429


namespace NUMINAMATH_GPT_stan_needs_more_minutes_l1494_149420

/-- Stan has 10 songs each of 3 minutes and 15 songs each of 2 minutes. His run takes 100 minutes.
    Prove that he needs 40 more minutes of songs in his playlist. -/
theorem stan_needs_more_minutes 
    (num_3min_songs : ℕ) 
    (num_2min_songs : ℕ) 
    (time_per_3min_song : ℕ) 
    (time_per_2min_song : ℕ) 
    (total_run_time : ℕ) 
    (given_minutes_3min_songs : num_3min_songs = 10)
    (given_minutes_2min_songs : num_2min_songs = 15)
    (given_time_per_3min_song : time_per_3min_song = 3)
    (given_time_per_2min_song : time_per_2min_song = 2)
    (given_total_run_time : total_run_time = 100)
    : num_3min_songs * time_per_3min_song + num_2min_songs * time_per_2min_song = 60 →
      total_run_time - (num_3min_songs * time_per_3min_song + num_2min_songs * time_per_2min_song) = 40 := 
by
    sorry

end NUMINAMATH_GPT_stan_needs_more_minutes_l1494_149420


namespace NUMINAMATH_GPT_train_speed_calculation_l1494_149417

open Real

noncomputable def train_speed_in_kmph (V : ℝ) : ℝ := V * 3.6

theorem train_speed_calculation (L V : ℝ) (h1 : L = 16 * V) (h2 : L + 280 = 30 * V) :
  train_speed_in_kmph V = 72 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_calculation_l1494_149417


namespace NUMINAMATH_GPT_mike_remaining_cards_l1494_149490

def initial_cards (mike_cards : ℕ) : ℕ := 87
def sam_cards (sam_bought : ℕ) : ℕ := 13
def alex_cards (alex_bought : ℕ) : ℕ := 15

theorem mike_remaining_cards (mike_cards sam_bought alex_bought : ℕ) :
  mike_cards - (sam_bought + alex_bought) = 59 :=
by
  let mike_cards := initial_cards 87
  let sam_cards := sam_bought
  let alex_cards := alex_bought
  sorry

end NUMINAMATH_GPT_mike_remaining_cards_l1494_149490


namespace NUMINAMATH_GPT_Mr_A_financial_outcome_l1494_149407

def home_worth : ℝ := 200000
def profit_percent : ℝ := 0.15
def loss_percent : ℝ := 0.05

def selling_price := (1 + profit_percent) * home_worth
def buying_price := (1 - loss_percent) * selling_price

theorem Mr_A_financial_outcome : 
  selling_price - buying_price = 11500 :=
by
  sorry

end NUMINAMATH_GPT_Mr_A_financial_outcome_l1494_149407


namespace NUMINAMATH_GPT_ajax_store_price_l1494_149416

theorem ajax_store_price (original_price : ℝ) (first_discount_rate : ℝ) (second_discount_rate : ℝ)
    (h_original: original_price = 180)
    (h_first_discount : first_discount_rate = 0.5)
    (h_second_discount : second_discount_rate = 0.2) :
    let first_discount_price := original_price * (1 - first_discount_rate)
    let saturday_price := first_discount_price * (1 - second_discount_rate)
    saturday_price = 72 :=
by
    sorry

end NUMINAMATH_GPT_ajax_store_price_l1494_149416


namespace NUMINAMATH_GPT_equal_diagonals_implies_quad_or_pent_l1494_149410

-- Define a convex polygon with n edges and equal diagonals
structure ConvexPolygon (n : ℕ) :=
(edges : ℕ)
(convex : Prop)
(diagonalsEqualLength : Prop)

-- State the theorem to prove
theorem equal_diagonals_implies_quad_or_pent (n : ℕ) (poly : ConvexPolygon n) 
    (h1 : poly.convex) 
    (h2 : poly.diagonalsEqualLength) :
    (n = 4) ∨ (n = 5) :=
sorry

end NUMINAMATH_GPT_equal_diagonals_implies_quad_or_pent_l1494_149410


namespace NUMINAMATH_GPT_Robert_older_than_Elizabeth_l1494_149484

-- Define the conditions
def Patrick_half_Robert (Patrick Robert : ℕ) : Prop := Patrick = Robert / 2
def Robert_turn_30_in_2_years (Robert : ℕ) : Prop := Robert + 2 = 30
def Elizabeth_4_years_younger_than_Patrick (Elizabeth Patrick : ℕ) : Prop := Elizabeth = Patrick - 4

-- The theorem we need to prove
theorem Robert_older_than_Elizabeth
  (Patrick Robert Elizabeth : ℕ)
  (h1 : Patrick_half_Robert Patrick Robert)
  (h2 : Robert_turn_30_in_2_years Robert)
  (h3 : Elizabeth_4_years_younger_than_Patrick Elizabeth Patrick) :
  Robert - Elizabeth = 18 :=
sorry

end NUMINAMATH_GPT_Robert_older_than_Elizabeth_l1494_149484


namespace NUMINAMATH_GPT_intersection_M_N_l1494_149461

-- Define sets M and N
def M := { x : ℝ | ∃ t : ℝ, x = 2^(-t) }
def N := { y : ℝ | ∃ x : ℝ, y = Real.sin x }

-- Prove the intersection of M and N
theorem intersection_M_N : M ∩ N = { y : ℝ | 0 < y ∧ y ≤ 1 } :=
by sorry

end NUMINAMATH_GPT_intersection_M_N_l1494_149461


namespace NUMINAMATH_GPT_minimum_value_is_81_l1494_149428

noncomputable def minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) : ℝ :=
a^2 + 9 * a * b + 9 * b^2 + 3 * c^2

theorem minimum_value_is_81 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minimum_value a b c h1 h2 h3 h4 = 81 :=
sorry

end NUMINAMATH_GPT_minimum_value_is_81_l1494_149428


namespace NUMINAMATH_GPT_problem_l1494_149493

theorem problem (a b c d : ℝ) (h₁ : a + b = 0) (h₂ : c * d = 1) : 
  (5 * a + 5 * b - 7 * c * d) / (-(c * d) ^ 3) = 7 := 
by
  sorry

end NUMINAMATH_GPT_problem_l1494_149493


namespace NUMINAMATH_GPT_total_marbles_correct_l1494_149478

-- Define the number of marbles Mary has
def MaryYellowMarbles := 9
def MaryBlueMarbles := 7
def MaryGreenMarbles := 6

-- Define the number of marbles Joan has
def JoanYellowMarbles := 3
def JoanBlueMarbles := 5
def JoanGreenMarbles := 4

-- Define the total number of marbles for Mary and Joan combined
def TotalMarbles := MaryYellowMarbles + MaryBlueMarbles + MaryGreenMarbles + JoanYellowMarbles + JoanBlueMarbles + JoanGreenMarbles

-- We want to prove that the total number of marbles is 34
theorem total_marbles_correct : TotalMarbles = 34 := by
  -- The proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_total_marbles_correct_l1494_149478


namespace NUMINAMATH_GPT_problem_inequality_l1494_149403

theorem problem_inequality (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_abc : a * b * c = 1) :
  (a - 1 + (1 / b)) * (b - 1 + (1 / c)) * (c - 1 + (1 / a)) ≤ 1 :=
sorry

end NUMINAMATH_GPT_problem_inequality_l1494_149403


namespace NUMINAMATH_GPT_image_of_3_5_pre_image_of_3_5_l1494_149474

def f (x y : ℤ) : ℤ × ℤ := (x - y, x + y)

theorem image_of_3_5 : f 3 5 = (-2, 8) :=
by
  sorry

theorem pre_image_of_3_5 : ∃ (x y : ℤ), f x y = (3, 5) ∧ x = 4 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_image_of_3_5_pre_image_of_3_5_l1494_149474


namespace NUMINAMATH_GPT_graph_of_conic_section_is_straight_lines_l1494_149492

variable {x y : ℝ}

theorem graph_of_conic_section_is_straight_lines:
  (x^2 - 9 * y^2 = 0) ↔ (x = 3 * y ∨ x = -3 * y) := by
  sorry

end NUMINAMATH_GPT_graph_of_conic_section_is_straight_lines_l1494_149492


namespace NUMINAMATH_GPT_find_cement_used_lexi_l1494_149453

def cement_used_total : ℝ := 15.1
def cement_used_tess : ℝ := 5.1
def cement_used_lexi : ℝ := cement_used_total - cement_used_tess

theorem find_cement_used_lexi : cement_used_lexi = 10 := by
  sorry

end NUMINAMATH_GPT_find_cement_used_lexi_l1494_149453


namespace NUMINAMATH_GPT_original_triangle_area_l1494_149402

theorem original_triangle_area (A_new : ℝ) (scale_factor : ℝ) (A_original : ℝ) 
  (h1: scale_factor = 5) (h2: A_new = 200) (h3: A_new = scale_factor^2 * A_original) : 
  A_original = 8 :=
by
  sorry

end NUMINAMATH_GPT_original_triangle_area_l1494_149402


namespace NUMINAMATH_GPT_length_increase_100_l1494_149456

theorem length_increase_100 (n : ℕ) (h : (n + 2) / 2 = 100) : n = 198 :=
sorry

end NUMINAMATH_GPT_length_increase_100_l1494_149456


namespace NUMINAMATH_GPT_cos_of_theta_l1494_149476

theorem cos_of_theta
  (A : ℝ) (a : ℝ) (m : ℝ) (θ : ℝ) 
  (hA : A = 40) 
  (ha : a = 12) 
  (hm : m = 10) 
  (h_area: A = (1/2) * a * m * Real.sin θ) 
  : Real.cos θ = (Real.sqrt 5) / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_of_theta_l1494_149476


namespace NUMINAMATH_GPT_min_value_of_z_l1494_149459

theorem min_value_of_z : ∀ (x : ℝ), ∃ z : ℝ, z = 5 * x^2 - 20 * x + 45 ∧ z ≥ 25 :=
by sorry

end NUMINAMATH_GPT_min_value_of_z_l1494_149459


namespace NUMINAMATH_GPT_find_a_value_l1494_149477

/-- Given the distribution of the random variable ξ as p(ξ = k) = a (1/3)^k for k = 1, 2, 3, 
    prove that the value of a that satisfies the probabilities summing to 1 is 27/13. -/
theorem find_a_value (a : ℝ) :
  (a * (1 / 3) + a * (1 / 3)^2 + a * (1 / 3)^3 = 1) → a = 27 / 13 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_find_a_value_l1494_149477


namespace NUMINAMATH_GPT_value_of_b_minus_d_squared_l1494_149443

variable {a b c d : ℤ}

theorem value_of_b_minus_d_squared (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 3) : (b - d) ^ 2 = 25 := 
by
  sorry

end NUMINAMATH_GPT_value_of_b_minus_d_squared_l1494_149443


namespace NUMINAMATH_GPT_fourth_power_nested_sqrt_l1494_149480

noncomputable def nested_sqrt := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))

theorem fourth_power_nested_sqrt :
  (nested_sqrt ^ 4) = 6 + 4 * Real.sqrt (2 + Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_fourth_power_nested_sqrt_l1494_149480


namespace NUMINAMATH_GPT_largest_fraction_l1494_149491

theorem largest_fraction
  (a b c d : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d) :
  (c + d) / (a + b) ≥ (a + b) / (c + d)
  ∧ (c + d) / (a + b) ≥ (a + d) / (b + c)
  ∧ (c + d) / (a + b) ≥ (b + c) / (a + d)
  ∧ (c + d) / (a + b) ≥ (b + d) / (a + c) :=
by
  sorry

end NUMINAMATH_GPT_largest_fraction_l1494_149491


namespace NUMINAMATH_GPT_johns_total_money_l1494_149486

-- Defining the given conditions
def initial_amount : ℕ := 5
def amount_spent : ℕ := 2
def allowance : ℕ := 26

-- Constructing the proof statement
theorem johns_total_money : initial_amount - amount_spent + allowance = 29 :=
by
  sorry

end NUMINAMATH_GPT_johns_total_money_l1494_149486


namespace NUMINAMATH_GPT_base8_addition_l1494_149473

theorem base8_addition (X Y : ℕ) 
  (h1 : 5 * 8 + X + Y + 3 * 8 + 2 = 6 * 64 + 4 * 8 + X) :
  X + Y = 16 := by
  sorry

end NUMINAMATH_GPT_base8_addition_l1494_149473


namespace NUMINAMATH_GPT_incorrect_statement_C_l1494_149462

theorem incorrect_statement_C (x : ℝ) (h : x > -2) : (6 / x) > -3 :=
sorry

end NUMINAMATH_GPT_incorrect_statement_C_l1494_149462


namespace NUMINAMATH_GPT_simplify_expression_frac_l1494_149471

theorem simplify_expression_frac (a b k : ℤ) (h : (6*k + 12) / 6 = a * k + b) : a = 1 ∧ b = 2 → a / b = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_frac_l1494_149471


namespace NUMINAMATH_GPT_find_a_l1494_149494

theorem find_a (a : ℝ) (k_l : ℝ) (h1 : k_l = -1)
  (h2 : a ≠ 3) 
  (h3 : (2 - (-1)) / (3 - a) * k_l = -1) : a = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1494_149494


namespace NUMINAMATH_GPT_area_of_ABCM_l1494_149405

-- Definitions of the problem conditions
def length_of_sides (P : ℕ) := 4
def forms_right_angle (P : ℕ) := True
def M_intersection (AG CH : ℝ) := True

-- Proposition that quadrilateral ABCM has the correct area
theorem area_of_ABCM (a b c m : ℝ) :
  (length_of_sides 12 = 4) ∧
  (forms_right_angle 12) ∧
  (M_intersection a b) →
  ∃ area_ABCM : ℝ, area_ABCM = 88/5 :=
by
  sorry

end NUMINAMATH_GPT_area_of_ABCM_l1494_149405


namespace NUMINAMATH_GPT_average_score_l1494_149485

variable (T : ℝ) -- Total number of students
variable (M : ℝ) -- Number of male students
variable (F : ℝ) -- Number of female students

variable (avgM : ℝ) -- Average score for male students
variable (avgF : ℝ) -- Average score for female students

-- Conditions
def M_condition : Prop := M = 0.4 * T
def F_condition : Prop := F = 0.6 * T
def avgM_condition : Prop := avgM = 75
def avgF_condition : Prop := avgF = 80

theorem average_score (h1 : M_condition T M) (h2 : F_condition T F) 
    (h3 : avgM_condition avgM) (h4 : avgF_condition avgF) :
    (75 * M + 80 * F) / T = 78 := by
  sorry

end NUMINAMATH_GPT_average_score_l1494_149485


namespace NUMINAMATH_GPT_value_of_frac_mul_l1494_149413

theorem value_of_frac_mul (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 2 * d) :
  (a * c) / (b * d) = 8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_frac_mul_l1494_149413


namespace NUMINAMATH_GPT_device_failure_probability_l1494_149452

noncomputable def probability_fail_device (p1 p2 p3 : ℝ) (p_one p_two p_three : ℝ) : ℝ :=
  0.006 * p3 + 0.092 * p_two + 0.398 * p_one

theorem device_failure_probability
  (p1 p2 p3 : ℝ) (p_one p_two p_three : ℝ)
  (h1 : p1 = 0.1)
  (h2 : p2 = 0.2)
  (h3 : p3 = 0.3)
  (h4 : p_one = 0.25)
  (h5 : p_two = 0.6)
  (h6 : p_three = 0.9) :
  probability_fail_device p1 p2 p3 p_one p_two p_three = 0.1601 :=
by
  sorry

end NUMINAMATH_GPT_device_failure_probability_l1494_149452


namespace NUMINAMATH_GPT_number_of_arrangements_SEES_l1494_149401

theorem number_of_arrangements_SEES : 
  ∃ n : ℕ, 
    (∀ (total_letters E S : ℕ), 
      total_letters = 4 ∧ E = 2 ∧ S = 2 → 
      n = Nat.factorial total_letters / (Nat.factorial E * Nat.factorial S)) → 
    n = 6 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_arrangements_SEES_l1494_149401


namespace NUMINAMATH_GPT_ratio_bob_to_jason_l1494_149496

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := 35

theorem ratio_bob_to_jason : bob_grade / jason_grade = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_bob_to_jason_l1494_149496


namespace NUMINAMATH_GPT_last_digit_2_to_2010_l1494_149487

theorem last_digit_2_to_2010 : (2 ^ 2010) % 10 = 4 := 
by
  -- proofs and lemmas go here
  sorry

end NUMINAMATH_GPT_last_digit_2_to_2010_l1494_149487


namespace NUMINAMATH_GPT_min_value_objective_l1494_149444

variable (x y : ℝ)

def constraints : Prop :=
  3 * x + y - 6 ≥ 0 ∧ x - y - 2 ≤ 0 ∧ y - 3 ≤ 0

def objective (x y : ℝ) : ℝ := y - 2 * x

theorem min_value_objective :
  constraints x y → ∃ x y, objective x y = -7 :=
by
  sorry

end NUMINAMATH_GPT_min_value_objective_l1494_149444


namespace NUMINAMATH_GPT_separate_curves_l1494_149448

variable {A : Type} [CommRing A]

def crossing_characteristic (ε : A → ℤ) (A1 A2 A3 A4 : A) : Prop :=
  ε A1 + ε A2 + ε A3 + ε A4 = 0

theorem separate_curves {A : Type} [CommRing A]
  {ε : A → ℤ} {A1 A2 A3 A4 : A} 
  (h : ε A1 + ε A2 + ε A3 + ε A4 = 0)
  (h1 : ε A1 = 1 ∨ ε A1 = -1)
  (h2 : ε A2 = 1 ∨ ε A2 = -1)
  (h3 : ε A3 = 1 ∨ ε A3 = -1)
  (h4 : ε A4 = 1 ∨ ε A4 = -1) :
  (∃ B1 B2 : A, B1 ≠ B2 ∧  ∀ (A : A), ((ε A = 1) → (A = B1)) ∨ ((ε A = -1) → (A = B2))) :=
  sorry

end NUMINAMATH_GPT_separate_curves_l1494_149448


namespace NUMINAMATH_GPT_investment_in_real_estate_l1494_149442

def total_investment : ℝ := 200000
def ratio_real_estate_to_mutual_funds : ℝ := 7

theorem investment_in_real_estate (mutual_funds_investment real_estate_investment: ℝ) 
  (h1 : mutual_funds_investment + real_estate_investment = total_investment)
  (h2 : real_estate_investment = ratio_real_estate_to_mutual_funds * mutual_funds_investment) :
  real_estate_investment = 175000 := sorry

end NUMINAMATH_GPT_investment_in_real_estate_l1494_149442


namespace NUMINAMATH_GPT_find_y_value_l1494_149481

variable (x y z k : ℝ)

-- Conditions
def inverse_relation_y (x y : ℝ) (k : ℝ) : Prop := 5 * y = k / (x^2)
def direct_relation_z (x z : ℝ) : Prop := 3 * z = x

-- Constant from conditions
def k_constant := 500

-- Problem statement
theorem find_y_value (h1 : inverse_relation_y 2 25 k_constant) (h2 : direct_relation_z 4 6) :
  y = 6.25 :=
by
  sorry

-- Auxiliary instance to fulfill the proof requirement
noncomputable def y_value : ℝ := 6.25

end NUMINAMATH_GPT_find_y_value_l1494_149481


namespace NUMINAMATH_GPT_initial_number_of_friends_is_six_l1494_149419

theorem initial_number_of_friends_is_six
  (car_cost : ℕ)
  (car_wash_earnings : ℕ)
  (F : ℕ)
  (additional_cost_when_one_friend_leaves : ℕ)
  (h1 : car_cost = 1700)
  (h2 : car_wash_earnings = 500)
  (remaining_cost := car_cost - car_wash_earnings)
  (cost_per_friend_before := remaining_cost / F)
  (cost_per_friend_after := remaining_cost / (F - 1))
  (h3 : additional_cost_when_one_friend_leaves = 40)
  (h4 : cost_per_friend_after = cost_per_friend_before + additional_cost_when_one_friend_leaves) :
  F = 6 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_friends_is_six_l1494_149419


namespace NUMINAMATH_GPT_base_conversion_and_addition_l1494_149436

theorem base_conversion_and_addition :
  let a₈ : ℕ := 3 * 8^2 + 5 * 8^1 + 6 * 8^0
  let c₁₄ : ℕ := 4 * 14^2 + 12 * 14^1 + 3 * 14^0
  a₈ + c₁₄ = 1193 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_and_addition_l1494_149436


namespace NUMINAMATH_GPT_total_students_is_45_l1494_149404

-- Define the initial conditions with the definitions provided
def drunk_drivers : Nat := 6
def speeders : Nat := 7 * drunk_drivers - 3
def total_students : Nat := drunk_drivers + speeders

-- The theorem to prove that the total number of students is 45
theorem total_students_is_45 : total_students = 45 :=
by
  sorry

end NUMINAMATH_GPT_total_students_is_45_l1494_149404


namespace NUMINAMATH_GPT_people_in_room_l1494_149430

/-- 
   Problem: Five-sixths of the people in a room are seated in five-sixths of the chairs.
   The rest of the people are standing. If there are 10 empty chairs, 
   prove that there are 60 people in the room.
-/
theorem people_in_room (people chairs : ℕ) 
  (h_condition1 : 5 / 6 * people = 5 / 6 * chairs) 
  (h_condition2 : chairs = 60) :
  people = 60 :=
by
  sorry

end NUMINAMATH_GPT_people_in_room_l1494_149430


namespace NUMINAMATH_GPT_number_of_friends_l1494_149451

def has14_pokemon_cards (x : String) : Prop :=
  x = "Sam" ∨ x = "Dan" ∨ x = "Tom" ∨ x = "Keith"

theorem number_of_friends :
  ∃ n, n = 4 ∧
        ∀ x, has14_pokemon_cards x ↔ x = "Sam" ∨ x = "Dan" ∨ x = "Tom" ∨ x = "Keith" :=
by
  sorry

end NUMINAMATH_GPT_number_of_friends_l1494_149451


namespace NUMINAMATH_GPT_decreasing_interval_f_l1494_149464

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 4 * x - 3

-- Statement to prove that the interval where f is monotonically decreasing is [2, +∞)
theorem decreasing_interval_f : (∀ x₁ x₂ : ℝ, 2 ≤ x₁ ∧ x₁ ≤ x₂ → f x₁ ≥ f x₂) :=
by
  sorry

end NUMINAMATH_GPT_decreasing_interval_f_l1494_149464


namespace NUMINAMATH_GPT_number_of_roses_l1494_149437

theorem number_of_roses 
  (R L T : ℕ)
  (h1 : R + L + T = 100)
  (h2 : R = L + 22)
  (h3 : R = T - 20) : R = 34 := 
sorry

end NUMINAMATH_GPT_number_of_roses_l1494_149437


namespace NUMINAMATH_GPT_trigonometric_ratio_l1494_149427

open Real

theorem trigonometric_ratio (u v : ℝ) (h1 : sin u / sin v = 4) (h2 : cos u / cos v = 1 / 3) :
  (sin (2 * u) / sin (2 * v) + cos (2 * u) / cos (2 * v)) = 389 / 381 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_ratio_l1494_149427


namespace NUMINAMATH_GPT_material_left_eq_l1494_149457

theorem material_left_eq :
  let a := (4 / 17 : ℚ)
  let b := (3 / 10 : ℚ)
  let total_bought := a + b
  let used := (0.23529411764705882 : ℚ)
  total_bought - used = (51 / 170 : ℚ) :=
by
  let a := (4 / 17 : ℚ)
  let b := (3 / 10 : ℚ)
  let total_bought := a + b
  let used := (0.23529411764705882 : ℚ)
  show total_bought - used = (51 / 170)
  sorry

end NUMINAMATH_GPT_material_left_eq_l1494_149457


namespace NUMINAMATH_GPT_max_bk_at_k_l1494_149438
open Nat Real

theorem max_bk_at_k :
  let B_k (k : ℕ) := (choose 2000 k) * (0.1 : ℝ) ^ k
  ∃ k : ℕ, (k = 181) ∧ (∀ m : ℕ, B_k m ≤ B_k k) :=
sorry

end NUMINAMATH_GPT_max_bk_at_k_l1494_149438


namespace NUMINAMATH_GPT_smallest_prime_with_digit_sum_23_l1494_149499

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end NUMINAMATH_GPT_smallest_prime_with_digit_sum_23_l1494_149499


namespace NUMINAMATH_GPT_hammer_nail_cost_l1494_149475

variable (h n : ℝ)

theorem hammer_nail_cost (h n : ℝ)
    (h1 : 4 * h + 5 * n = 10.45)
    (h2 : 3 * h + 9 * n = 12.87) :
  20 * h + 25 * n = 52.25 :=
sorry

end NUMINAMATH_GPT_hammer_nail_cost_l1494_149475


namespace NUMINAMATH_GPT_bob_homework_time_l1494_149467

variable (T_Alice T_Bob : ℕ)

theorem bob_homework_time (h_Alice : T_Alice = 40) (h_Bob : T_Bob = (3 * T_Alice) / 8) : T_Bob = 15 :=
by
  rw [h_Alice] at h_Bob
  norm_num at h_Bob
  exact h_Bob

-- Assuming T_Alice represents the time taken by Alice to complete her homework
-- and T_Bob represents the time taken by Bob to complete his homework,
-- we prove that T_Bob is 15 minutes given the conditions.

end NUMINAMATH_GPT_bob_homework_time_l1494_149467


namespace NUMINAMATH_GPT_avg_rate_first_half_l1494_149409

theorem avg_rate_first_half (total_distance : ℕ) (avg_rate : ℕ) (first_half_distance : ℕ) (second_half_distance : ℕ)
  (rate_first_half : ℕ) (time_first_half : ℕ) (time_second_half : ℕ) (total_time : ℕ) :
  total_distance = 640 →
  second_half_distance = first_half_distance →
  time_second_half = 3 * time_first_half →
  avg_rate = 40 →
  total_distance = first_half_distance + second_half_distance →
  total_time = time_first_half + time_second_half →
  avg_rate = total_distance / total_time →
  time_first_half = first_half_distance / rate_first_half →
  rate_first_half = 80
  :=
  sorry

end NUMINAMATH_GPT_avg_rate_first_half_l1494_149409


namespace NUMINAMATH_GPT_unique_function_solution_l1494_149450

theorem unique_function_solution :
  ∀ f : ℕ+ → ℕ+, (∀ x y : ℕ+, f (x + y * f x) = x * f (y + 1)) → (∀ x : ℕ+, f x = x) :=
by
  sorry

end NUMINAMATH_GPT_unique_function_solution_l1494_149450


namespace NUMINAMATH_GPT_deshaun_read_books_over_summer_l1494_149412

theorem deshaun_read_books_over_summer 
  (summer_days : ℕ)
  (average_pages_per_book : ℕ)
  (ratio_closest_person : ℝ)
  (pages_read_per_day_second_person : ℕ)
  (books_read : ℕ)
  (total_pages_second_person_read : ℕ)
  (h1 : summer_days = 80)
  (h2 : average_pages_per_book = 320)
  (h3 : ratio_closest_person = 0.75)
  (h4 : pages_read_per_day_second_person = 180)
  (h5 : total_pages_second_person_read = pages_read_per_day_second_person * summer_days)
  (h6 : books_read * average_pages_per_book = total_pages_second_person_read / ratio_closest_person) :
  books_read = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_deshaun_read_books_over_summer_l1494_149412


namespace NUMINAMATH_GPT_turnover_june_l1494_149421

variable (TurnoverApril TurnoverMay : ℝ)

theorem turnover_june (h1 : TurnoverApril = 10) (h2 : TurnoverMay = 12) :
  TurnoverMay * (1 + (TurnoverMay - TurnoverApril) / TurnoverApril) = 14.4 := by
  sorry

end NUMINAMATH_GPT_turnover_june_l1494_149421


namespace NUMINAMATH_GPT_find_first_number_of_sequence_l1494_149423

theorem find_first_number_of_sequence
    (a : ℕ → ℕ)
    (h1 : ∀ n, 3 ≤ n → a n = a (n-1) * a (n-2))
    (h2 : a 8 = 36)
    (h3 : a 9 = 1296)
    (h4 : a 10 = 46656) :
    a 1 = 60466176 := 
sorry

end NUMINAMATH_GPT_find_first_number_of_sequence_l1494_149423


namespace NUMINAMATH_GPT_prize_calculations_l1494_149424

-- Definitions for the conditions
def total_prizes := 50
def first_prize_unit_price := 20
def second_prize_unit_price := 14
def third_prize_unit_price := 8
def num_second_prize (x : ℕ) := 3 * x - 2
def num_third_prize (x : ℕ) := total_prizes - x - num_second_prize x
def total_cost (x : ℕ) := first_prize_unit_price * x + second_prize_unit_price * num_second_prize x + third_prize_unit_price * num_third_prize x

-- Proof problem statement
theorem prize_calculations (x : ℕ) (h : num_second_prize x = 22) : 
  num_second_prize x = 3 * x - 2 ∧ 
  num_third_prize x = 52 - 4 * x ∧ 
  total_cost x = 30 * x + 388 ∧ 
  total_cost 8 = 628 :=
by
  sorry

end NUMINAMATH_GPT_prize_calculations_l1494_149424


namespace NUMINAMATH_GPT_line_AC_eqn_l1494_149440

-- Define points A and B
structure Point where
  x : ℝ
  y : ℝ

-- Define point A
def A : Point := { x := 3, y := 1 }

-- Define point B
def B : Point := { x := -1, y := 2 }

-- Define the line equation y = x + 1
def line_eq (p : Point) : Prop := p.y = p.x + 1

-- Define the bisector being on line y=x+1 as a condition
axiom bisector_on_line (C : Point) : 
  line_eq C → (∃ k : ℝ, (C.y - B.y) = k * (C.x - B.x))

-- Define the final goal to prove the equation of line AC
theorem line_AC_eqn (C : Point) :
  line_eq C → ((A.x - C.x) * (B.y - C.y) = (B.x - C.x) * (A.y - C.y)) → C.x = -3 ∧ C.y = -2 → 
  (A.x - 2 * A.y = 1) := sorry

end NUMINAMATH_GPT_line_AC_eqn_l1494_149440


namespace NUMINAMATH_GPT_sqrt_product_l1494_149468

theorem sqrt_product : (Real.sqrt 121) * (Real.sqrt 49) * (Real.sqrt 11) = 77 * (Real.sqrt 11) := by
  -- This is just the theorem statement as requested.
  sorry

end NUMINAMATH_GPT_sqrt_product_l1494_149468


namespace NUMINAMATH_GPT_calculate_expression_l1494_149479

theorem calculate_expression (x : ℕ) (h : x = 3) : x + x * x^(x - 1) = 30 := by
  rw [h]
  -- Proof steps would go here but we are including only the statement
  sorry

end NUMINAMATH_GPT_calculate_expression_l1494_149479


namespace NUMINAMATH_GPT_race_positions_l1494_149433

theorem race_positions
  (positions : Fin 15 → String) 
  (h_quinn_lucas : ∃ n : Fin 15, positions n = "Quinn" ∧ positions (n + 4) = "Lucas")
  (h_oliver_quinn : ∃ n : Fin 15, positions (n - 1) = "Oliver" ∧ positions n = "Quinn")
  (h_naomi_oliver : ∃ n : Fin 15, positions n = "Naomi" ∧ positions (n + 3) = "Oliver")
  (h_emma_lucas : ∃ n : Fin 15, positions n = "Lucas" ∧ positions (n + 1) = "Emma")
  (h_sara_naomi : ∃ n : Fin 15, positions n = "Naomi" ∧ positions (n + 1) = "Sara")
  (h_naomi_4th : ∃ n : Fin 15, n = 3 ∧ positions n = "Naomi") :
  positions 6 = "Oliver" :=
by
  sorry

end NUMINAMATH_GPT_race_positions_l1494_149433


namespace NUMINAMATH_GPT_sum_a_b_l1494_149458

theorem sum_a_b (a b : ℕ) (h1 : 2 + 2 / 3 = 2^2 * (2 / 3))
(h2: 3 + 3 / 8 = 3^2 * (3 / 8)) 
(h3: 4 + 4 / 15 = 4^2 * (4 / 15)) 
(h_n : ∀ n, n + n / (n^2 - 1) = n^2 * (n / (n^2 - 1)) → 
(a = 9^2 - 1) ∧ (b = 9)) : 
a + b = 89 := 
sorry

end NUMINAMATH_GPT_sum_a_b_l1494_149458


namespace NUMINAMATH_GPT_trig_cos_sum_l1494_149426

open Real

theorem trig_cos_sum :
  cos (37 * (π / 180)) * cos (23 * (π / 180)) - sin (37 * (π / 180)) * sin (23 * (π / 180)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trig_cos_sum_l1494_149426


namespace NUMINAMATH_GPT_range_of_3a_minus_b_l1494_149406

theorem range_of_3a_minus_b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3)
                             (h3 : 2 < a - b) (h4 : a - b < 4) :
    ∃ (x : ℝ), 3 ≤ x ∧ x ≤ 11 ∧ x = 3 * a - b :=
sorry

end NUMINAMATH_GPT_range_of_3a_minus_b_l1494_149406


namespace NUMINAMATH_GPT_exist_circle_tangent_to_three_circles_l1494_149482

variable (h1 k1 r1 h2 k2 r2 h3 k3 r3 h k r : ℝ)

def condition1 : Prop := (h - h1)^2 + (k - k1)^2 = (r + r1)^2
def condition2 : Prop := (h - h2)^2 + (k - k2)^2 = (r + r2)^2
def condition3 : Prop := (h - h3)^2 + (k - k3)^2 = (r + r3)^2

theorem exist_circle_tangent_to_three_circles : 
  ∃ (h k r : ℝ), condition1 h1 k1 r1 h k r ∧ condition2 h2 k2 r2 h k r ∧ condition3 h3 k3 r3 h k r :=
by
  sorry

end NUMINAMATH_GPT_exist_circle_tangent_to_three_circles_l1494_149482


namespace NUMINAMATH_GPT_rolls_sold_to_grandmother_l1494_149432

theorem rolls_sold_to_grandmother (t u n s g : ℕ) 
  (h1 : t = 45)
  (h2 : u = 10)
  (h3 : n = 6)
  (h4 : s = 28)
  (total_sold : t - s = g + u + n) : 
  g = 1 := 
  sorry

end NUMINAMATH_GPT_rolls_sold_to_grandmother_l1494_149432


namespace NUMINAMATH_GPT_find_third_number_in_second_set_l1494_149449

theorem find_third_number_in_second_set (x y: ℕ) 
    (h1 : (28 + x + 42 + 78 + 104) / 5 = 90) 
    (h2 : (128 + 255 + y + 1023 + x) / 5 = 423) 
: y = 511 := 
sorry

end NUMINAMATH_GPT_find_third_number_in_second_set_l1494_149449


namespace NUMINAMATH_GPT_floor_x_mul_x_eq_54_l1494_149488

def positive_real (x : ℝ) : Prop := x > 0

theorem floor_x_mul_x_eq_54 (x : ℝ) (h_pos : positive_real x) : ⌊x⌋ * x = 54 ↔ x = 54 / 7 :=
by
  sorry

end NUMINAMATH_GPT_floor_x_mul_x_eq_54_l1494_149488


namespace NUMINAMATH_GPT_polynomial_identity_l1494_149455

theorem polynomial_identity (x y : ℝ) (h : x - y = 1) : 
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 := 
  sorry

end NUMINAMATH_GPT_polynomial_identity_l1494_149455


namespace NUMINAMATH_GPT_candy_bar_cost_l1494_149414

theorem candy_bar_cost (initial_amount change : ℕ) (h : initial_amount = 50) (hc : change = 5) : 
  initial_amount - change = 45 :=
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_candy_bar_cost_l1494_149414


namespace NUMINAMATH_GPT_math_problem_l1494_149454

theorem math_problem (a b c k : ℝ) (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h2 : a + b + c = 0) (h3 : a^2 = k * b^2) (hk : k ≠ 0) :
  (a^2 * b^2) / ((a^2 - b * c) * (b^2 - a * c)) + (a^2 * c^2) / ((a^2 - b * c) * (c^2 - a * b)) + (b^2 * c^2) / ((b^2 - a * c) * (c^2 - a * b)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1494_149454


namespace NUMINAMATH_GPT_number_of_zookeepers_12_l1494_149435

theorem number_of_zookeepers_12 :
  let P := 30 -- number of penguins
  let Zr := 22 -- number of zebras
  let T := 8 -- number of tigers
  let A_heads := P + Zr + T -- total number of animal heads
  let A_feet := (2 * P) + (4 * Zr) + (4 * T) -- total number of animal feet
  ∃ Z : ℕ, -- number of zookeepers
  (A_heads + Z) + 132 = A_feet + (2 * Z) → Z = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_zookeepers_12_l1494_149435


namespace NUMINAMATH_GPT_frank_cookies_l1494_149447

theorem frank_cookies :
  ∀ (F M M_i L : ℕ),
    (F = M / 2 - 3) →
    (M = 3 * M_i) →
    (M_i = 2 * L) →
    (L = 5) →
    F = 12 :=
by
  intros F M M_i L h1 h2 h3 h4
  rw [h4] at h3
  rw [h3] at h2
  rw [h2] at h1
  sorry

end NUMINAMATH_GPT_frank_cookies_l1494_149447


namespace NUMINAMATH_GPT_Tom_Brady_passing_yards_l1494_149431

-- Definitions
def record := 5999
def current_yards := 4200
def games_left := 6

-- Proof problem statement
theorem Tom_Brady_passing_yards :
  (record + 1 - current_yards) / games_left = 300 := by
  sorry

end NUMINAMATH_GPT_Tom_Brady_passing_yards_l1494_149431


namespace NUMINAMATH_GPT_factor_difference_of_squares_l1494_149495

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l1494_149495


namespace NUMINAMATH_GPT_markup_percentage_is_ten_l1494_149400

theorem markup_percentage_is_ten (S C : ℝ)
  (h1 : S - C = 0.0909090909090909 * S) :
  (S - C) / C * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_markup_percentage_is_ten_l1494_149400


namespace NUMINAMATH_GPT_portion_of_profit_divided_equally_l1494_149422

-- Definitions for the given conditions
def total_investment_mary : ℝ := 600
def total_investment_mike : ℝ := 400
def total_profit : ℝ := 7500
def profit_diff : ℝ := 1000

-- Main statement
theorem portion_of_profit_divided_equally (E P : ℝ) 
  (h1 : total_profit = E + P)
  (h2 : E + (3/5) * P = E + (2/5) * P + profit_diff) :
  E = 2500 :=
by
  sorry

end NUMINAMATH_GPT_portion_of_profit_divided_equally_l1494_149422


namespace NUMINAMATH_GPT_arithmetic_sequence_n_value_l1494_149489

def arithmetic_seq_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_n_value :
  ∀ (a1 d n an : ℕ), a1 = 3 → d = 2 → an = 25 → arithmetic_seq_nth_term a1 d n = an → n = 12 :=
by
  intros a1 d n an ha1 hd han h
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_value_l1494_149489


namespace NUMINAMATH_GPT_evaluate_square_of_sum_l1494_149446

theorem evaluate_square_of_sum (x y : ℕ) (h1 : x + y = 20) (h2 : 2 * x + y = 27) : (x + y) ^ 2 = 400 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_square_of_sum_l1494_149446


namespace NUMINAMATH_GPT_solve_equation_l1494_149441

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x = -4/3) ↔ (x^2 + 2 * x + 2) / (x + 2) = x + 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1494_149441


namespace NUMINAMATH_GPT_minimum_ab_bc_ca_l1494_149465

theorem minimum_ab_bc_ca {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = a^3) (h5 : a * b * c = a^3) : 
  ab + bc + ca ≥ 9 :=
sorry

end NUMINAMATH_GPT_minimum_ab_bc_ca_l1494_149465


namespace NUMINAMATH_GPT_sin_subtract_of_obtuse_angle_l1494_149466

open Real -- Open the Real namespace for convenience.

theorem sin_subtract_of_obtuse_angle (α : ℝ) 
  (h1 : (π / 2) < α) (h2 : α < π)
  (h3 : sin (π / 4 + α) = 3 / 4)
  : sin (π / 4 - α) = - (sqrt 7) / 4 := 
by 
  sorry -- Proof placeholder.

end NUMINAMATH_GPT_sin_subtract_of_obtuse_angle_l1494_149466


namespace NUMINAMATH_GPT_problem_statement_l1494_149425

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x + 4
def g (x : ℝ) : ℝ := 2*x - 1

-- State the theorem and provide the necessary conditions
theorem problem_statement : f (g 5) - g (f 5) = 381 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1494_149425


namespace NUMINAMATH_GPT_trig_identity_proof_l1494_149418

theorem trig_identity_proof (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) :
  Real.sin (2 * α - π / 6) + Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l1494_149418


namespace NUMINAMATH_GPT_abc_le_sqrt2_div_4_l1494_149460

variable {a b c : ℝ}
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
variable (h : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) = 1)

theorem abc_le_sqrt2_div_4 (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) = 1) :
  a * b * c ≤ (Real.sqrt 2) / 4 := 
sorry

end NUMINAMATH_GPT_abc_le_sqrt2_div_4_l1494_149460


namespace NUMINAMATH_GPT_max_area_of_garden_l1494_149483

theorem max_area_of_garden (p : ℝ) (h : p = 36) : 
  ∃ A : ℝ, (∀ l w : ℝ, l + l + w + w = p → l * w ≤ A) ∧ A = 81 :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_garden_l1494_149483


namespace NUMINAMATH_GPT_jelly_bean_count_l1494_149472

variable (b c : ℕ)
variable (h1 : b = 3 * c)
variable (h2 : b - 5 = 5 * (c - 15))

theorem jelly_bean_count : b = 105 := by
  sorry

end NUMINAMATH_GPT_jelly_bean_count_l1494_149472


namespace NUMINAMATH_GPT_remaining_students_correct_l1494_149408

def initial_groups : Nat := 3
def students_per_group : Nat := 8
def students_left_early : Nat := 2

def total_students (groups students_per_group : Nat) : Nat := groups * students_per_group

def remaining_students (total students_left_early : Nat) : Nat := total - students_left_early

theorem remaining_students_correct :
  remaining_students (total_students initial_groups students_per_group) students_left_early = 22 := by
  sorry

end NUMINAMATH_GPT_remaining_students_correct_l1494_149408


namespace NUMINAMATH_GPT_find_d_h_l1494_149439

theorem find_d_h (a b c d g h : ℂ) (h1 : b = 4) (h2 : g = -a - c) (h3 : a + c + g = 0) (h4 : b + d + h = 3) : 
  d + h = -1 := 
by
  sorry

end NUMINAMATH_GPT_find_d_h_l1494_149439
