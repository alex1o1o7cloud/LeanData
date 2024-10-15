import Mathlib

namespace NUMINAMATH_GPT_tangent_function_property_l2271_227191

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.tan (ϕ - x)

theorem tangent_function_property 
  (ϕ a : ℝ) 
  (h1 : π / 2 < ϕ) 
  (h2 : ϕ < 3 * π / 2) 
  (h3 : f 0 ϕ = 0) 
  (h4 : f (-a) ϕ = 1/2) : 
  f (a + π / 4) ϕ = -3 := by
  sorry

end NUMINAMATH_GPT_tangent_function_property_l2271_227191


namespace NUMINAMATH_GPT_value_of_PQRS_l2271_227103

theorem value_of_PQRS : 
  let P := 2 * (Real.sqrt 2010 + Real.sqrt 2011)
  let Q := 3 * (-Real.sqrt 2010 - Real.sqrt 2011)
  let R := 2 * (Real.sqrt 2010 - Real.sqrt 2011)
  let S := 3 * (Real.sqrt 2011 - Real.sqrt 2010)
  P * Q * R * S = -36 :=
by
  sorry

end NUMINAMATH_GPT_value_of_PQRS_l2271_227103


namespace NUMINAMATH_GPT_find_a_l2271_227193

noncomputable def f (x a : ℝ) := x^2 + (a + 1) * x + (a + 2)

def g (x a : ℝ) := (a + 1) * x
def h (x a : ℝ) := x^2 + a + 2

def p (a : ℝ) := ∀ x ≥ (a + 1)^2, f x a ≤ x
def q (a : ℝ) := ∀ x, g x a < 0

theorem find_a : 
  (¬p a) → (p a ∨ q a) → a ≥ -1 := sorry

end NUMINAMATH_GPT_find_a_l2271_227193


namespace NUMINAMATH_GPT_solve_inequality_l2271_227133

theorem solve_inequality (x : ℝ) : 2 * x^2 - x - 1 > 0 ↔ x < -1/2 ∨ x > 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2271_227133


namespace NUMINAMATH_GPT_shorter_piece_is_28_l2271_227137

noncomputable def shorter_piece_length (x : ℕ) : Prop :=
  x + (x + 12) = 68 → x = 28

theorem shorter_piece_is_28 (x : ℕ) : shorter_piece_length x :=
by
  intro h
  have h1 : 2 * x + 12 = 68 := by linarith
  have h2 : 2 * x = 56 := by linarith
  have h3 : x = 28 := by linarith
  exact h3

end NUMINAMATH_GPT_shorter_piece_is_28_l2271_227137


namespace NUMINAMATH_GPT_solve_for_x_l2271_227175

theorem solve_for_x (x : ℤ) : 27 - 5 = 4 + x → x = 18 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l2271_227175


namespace NUMINAMATH_GPT_maria_green_beans_l2271_227101

theorem maria_green_beans
    (potatoes : ℕ)
    (carrots : ℕ)
    (onions : ℕ)
    (green_beans : ℕ)
    (h1 : potatoes = 2)
    (h2 : carrots = 6 * potatoes)
    (h3 : onions = 2 * carrots)
    (h4 : green_beans = onions / 3) :
  green_beans = 8 := 
sorry

end NUMINAMATH_GPT_maria_green_beans_l2271_227101


namespace NUMINAMATH_GPT_zeros_of_g_l2271_227184

theorem zeros_of_g (a b : ℝ) (h : 2 * a + b = 0) :
  (∃ x : ℝ, (b * x^2 - a * x = 0) ∧ (x = 0 ∨ x = -1 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_zeros_of_g_l2271_227184


namespace NUMINAMATH_GPT_sum_of_digits_l2271_227157

theorem sum_of_digits (N : ℕ) (h : N * (N + 1) / 2 = 3003) : (7 + 7) = 14 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_l2271_227157


namespace NUMINAMATH_GPT_sequence_converges_to_zero_and_N_for_epsilon_l2271_227145

theorem sequence_converges_to_zero_and_N_for_epsilon :
  (∀ ε > 0, ∃ N : ℕ, ∀ n > N, |1 / (n : ℝ) - 0| < ε) ∧ 
  (∃ N : ℕ, ∀ n > N, |1 / (n : ℝ)| < 0.001) :=
by
  sorry

end NUMINAMATH_GPT_sequence_converges_to_zero_and_N_for_epsilon_l2271_227145


namespace NUMINAMATH_GPT_regions_of_diagonals_formula_l2271_227152

def regions_of_diagonals (n : ℕ) : ℕ :=
  ((n - 1) * (n - 2) * (n * n - 3 * n + 12)) / 24

theorem regions_of_diagonals_formula (n : ℕ) (h : 3 ≤ n) :
  ∃ (fn : ℕ), fn = regions_of_diagonals n := by
  sorry

end NUMINAMATH_GPT_regions_of_diagonals_formula_l2271_227152


namespace NUMINAMATH_GPT_arithmetic_lemma_l2271_227161

theorem arithmetic_lemma : 45 * 52 + 48 * 45 = 4500 := by
  sorry

end NUMINAMATH_GPT_arithmetic_lemma_l2271_227161


namespace NUMINAMATH_GPT_range_of_m_l2271_227155

variable {f : ℝ → ℝ}

theorem range_of_m 
  (even_f : ∀ x : ℝ, f x = f (-x))
  (mono_f : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h : f (m + 1) < f (3 * m - 1)) :
  m > 1 ∨ m < 0 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2271_227155


namespace NUMINAMATH_GPT_binomial_coefficient_sum_l2271_227178

theorem binomial_coefficient_sum {n : ℕ} (h : (1 : ℝ) + 1 = 128) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_sum_l2271_227178


namespace NUMINAMATH_GPT_least_integer_nk_l2271_227158

noncomputable def min_nk (k : ℕ) : ℕ :=
  (5 * k + 1) / 2

theorem least_integer_nk (k : ℕ) (S : Fin 5 → Finset ℕ) :
  (∀ j : Fin 5, (S j).card = k) →
  (∀ i : Fin 4, (S i ∩ S (i + 1)).card = 0) →
  (S 4 ∩ S 0).card = 0 →
  (∃ nk, (∃ (U : Finset ℕ), (∀ j : Fin 5, S j ⊆ U) ∧ U.card = nk) ∧ nk = min_nk k) :=
by
  sorry

end NUMINAMATH_GPT_least_integer_nk_l2271_227158


namespace NUMINAMATH_GPT_population_in_scientific_notation_l2271_227156

theorem population_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 1370540000 = a * 10^n ∧ a = 1.37054 ∧ n = 9 :=
by
  sorry

end NUMINAMATH_GPT_population_in_scientific_notation_l2271_227156


namespace NUMINAMATH_GPT_multiple_of_eight_l2271_227174

theorem multiple_of_eight (x y : ℤ) (h : ∀ (k : ℤ), 24 + 16 * k = 8) : ∃ (k : ℤ), x + 16 * y = 8 * k := 
by
  sorry

end NUMINAMATH_GPT_multiple_of_eight_l2271_227174


namespace NUMINAMATH_GPT_intersection_P_Q_l2271_227121

def P (y : ℝ) : Prop :=
  ∃ (x : ℝ), y = -x^2 + 2

def Q (y : ℝ) : Prop :=
  ∃ (x : ℝ), y = x

theorem intersection_P_Q :
  { y : ℝ | P y } ∩ { y : ℝ | Q y } = { y : ℝ | y ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l2271_227121


namespace NUMINAMATH_GPT_no_n_geq_2_makes_10101n_prime_l2271_227125

theorem no_n_geq_2_makes_10101n_prime : ∀ n : ℕ, n ≥ 2 → ¬ Prime (n^4 + n^2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_no_n_geq_2_makes_10101n_prime_l2271_227125


namespace NUMINAMATH_GPT_number_of_pens_l2271_227190

theorem number_of_pens (num_pencils : ℕ) (total_cost : ℝ) (avg_price_pencil : ℝ) (avg_price_pen : ℝ) : ℕ :=
  sorry

example : number_of_pens 75 690 2 18 = 30 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_pens_l2271_227190


namespace NUMINAMATH_GPT_arithmetic_seq_question_l2271_227127

theorem arithmetic_seq_question (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a n = a 1 + (n - 1) * d)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_question_l2271_227127


namespace NUMINAMATH_GPT_total_distance_covered_l2271_227183

theorem total_distance_covered :
  ∀ (r j w total : ℝ),
    r = 40 →
    j = (3 / 5) * r →
    w = 5 * j →
    total = r + j + w →
    total = 184 := by
  sorry

end NUMINAMATH_GPT_total_distance_covered_l2271_227183


namespace NUMINAMATH_GPT_find_a9_l2271_227188

variable (a : ℕ → ℤ)  -- Arithmetic sequence
variable (S : ℕ → ℤ)  -- Sum of the first n terms

-- Conditions provided in the problem
axiom Sum_condition : S 8 = 4 * a 3
axiom Term_condition : a 7 = -2
axiom Sum_def : ∀ n, S n = (n * (a 1 + a n)) / 2

-- Hypothesis for common difference
def common_diff (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Proving that a_9 equals -6 given the conditions
theorem find_a9 (d : ℤ) : common_diff a d → a 9 = -6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_a9_l2271_227188


namespace NUMINAMATH_GPT_marbles_exceed_200_on_sunday_l2271_227194

theorem marbles_exceed_200_on_sunday:
  ∃ n : ℕ, 3 * 2^n > 200 ∧ (n % 7) = 0 :=
by
  sorry

end NUMINAMATH_GPT_marbles_exceed_200_on_sunday_l2271_227194


namespace NUMINAMATH_GPT_fg_of_2_eq_0_l2271_227197

def f (x : ℝ) : ℝ := 4 - x^2
def g (x : ℝ) : ℝ := 3 * x - x^3

theorem fg_of_2_eq_0 : f (g 2) = 0 := by
  sorry

end NUMINAMATH_GPT_fg_of_2_eq_0_l2271_227197


namespace NUMINAMATH_GPT_num_valid_N_l2271_227113

theorem num_valid_N : 
  ∃ n : ℕ, n = 4 ∧ ∀ (N : ℕ), (N > 0) → (∃ k : ℕ, 60 = (N+3) * k ∧ k % 2 = 0) ↔ (N = 1 ∨ N = 9 ∨ N = 17 ∨ N = 57) :=
sorry

end NUMINAMATH_GPT_num_valid_N_l2271_227113


namespace NUMINAMATH_GPT_total_cookies_prepared_l2271_227168

-- State the conditions as definitions
def num_guests : ℕ := 10
def cookies_per_guest : ℕ := 18

-- The theorem stating the problem
theorem total_cookies_prepared (num_guests cookies_per_guest : ℕ) : 
  num_guests * cookies_per_guest = 180 := 
by 
  -- Here, we would have the proof, but we're using sorry to skip it
  sorry

end NUMINAMATH_GPT_total_cookies_prepared_l2271_227168


namespace NUMINAMATH_GPT_option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l2271_227176

theorem option_A_incorrect (a : ℝ) : (a^2) * (a^3) ≠ a^6 :=
by sorry

theorem option_B_incorrect (a : ℝ) : (a^2)^3 ≠ a^5 :=
by sorry

theorem option_C_incorrect (a : ℝ) : (a^6) / (a^2) ≠ a^3 :=
by sorry

theorem option_D_correct (a b : ℝ) : (a + 2 * b) * (a - 2 * b) = a^2 - 4 * b^2 :=
by sorry

end NUMINAMATH_GPT_option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l2271_227176


namespace NUMINAMATH_GPT_kilometers_to_chains_l2271_227119

theorem kilometers_to_chains :
  (1 * 10 * 50 = 500) :=
by
  sorry

end NUMINAMATH_GPT_kilometers_to_chains_l2271_227119


namespace NUMINAMATH_GPT_cone_generatrix_length_is_2sqrt2_l2271_227182

noncomputable def cone_generatrix_length (r : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let l := (2 * Real.pi * r) / Real.pi
  l

theorem cone_generatrix_length_is_2sqrt2 :
  cone_generatrix_length (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_cone_generatrix_length_is_2sqrt2_l2271_227182


namespace NUMINAMATH_GPT_polar_to_cartesian_conversion_l2271_227111

noncomputable def polarToCartesian (ρ θ : ℝ) : ℝ × ℝ :=
  let x := ρ * Real.cos θ
  let y := ρ * Real.sin θ
  (x, y)

theorem polar_to_cartesian_conversion :
  polarToCartesian 4 (Real.pi / 3) = (2, 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_conversion_l2271_227111


namespace NUMINAMATH_GPT_inequality_on_abc_l2271_227189

variable (a b c : ℝ)

theorem inequality_on_abc (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  (a^4 + b^4) / (a^6 + b^6) + (b^4 + c^4) / (b^6 + c^6) + (c^4 + a^4) / (c^6 + a^6) ≤ 1 / (a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_on_abc_l2271_227189


namespace NUMINAMATH_GPT_unique_real_root_iff_a_eq_3_l2271_227143

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * abs x + a^2 - 9

theorem unique_real_root_iff_a_eq_3 {a : ℝ} (hu : ∃! x : ℝ, f x a = 0) : a = 3 :=
sorry

end NUMINAMATH_GPT_unique_real_root_iff_a_eq_3_l2271_227143


namespace NUMINAMATH_GPT_greater_number_l2271_227195

theorem greater_number (x: ℕ) (h1 : 3 * x + 4 * x = 21) : 4 * x = 12 := by
  sorry

end NUMINAMATH_GPT_greater_number_l2271_227195


namespace NUMINAMATH_GPT_minimum_teachers_needed_l2271_227122

theorem minimum_teachers_needed
  (math_teachers : ℕ) (physics_teachers : ℕ) (chemistry_teachers : ℕ)
  (max_subjects_per_teacher : ℕ) :
  math_teachers = 7 →
  physics_teachers = 6 →
  chemistry_teachers = 5 →
  max_subjects_per_teacher = 3 →
  ∃ t : ℕ, t = 5 ∧ (t * max_subjects_per_teacher ≥ math_teachers + physics_teachers + chemistry_teachers) :=
by
  repeat { sorry }

end NUMINAMATH_GPT_minimum_teachers_needed_l2271_227122


namespace NUMINAMATH_GPT_distances_product_eq_l2271_227159

-- Define the distances
variables (d_ab d_ac d_bc d_ba d_cb d_ca : ℝ)

-- State the theorem
theorem distances_product_eq : d_ab * d_bc * d_ca = d_ac * d_ba * d_cb :=
sorry

end NUMINAMATH_GPT_distances_product_eq_l2271_227159


namespace NUMINAMATH_GPT_tshirt_cost_l2271_227123

theorem tshirt_cost (initial_amount sweater_cost shoes_cost amount_left spent_on_tshirt : ℕ) 
  (h_initial : initial_amount = 91) 
  (h_sweater : sweater_cost = 24) 
  (h_shoes : shoes_cost = 11) 
  (h_left : amount_left = 50)
  (h_spent : spent_on_tshirt = initial_amount - amount_left - sweater_cost - shoes_cost) :
  spent_on_tshirt = 6 :=
sorry

end NUMINAMATH_GPT_tshirt_cost_l2271_227123


namespace NUMINAMATH_GPT_average_speed_round_trip_l2271_227150

-- Define average speed calculation for round trip

open Real

theorem average_speed_round_trip (S : ℝ) (hS : S > 0) :
  let t1 := S / 6
  let t2 := S / 4
  let total_distance := 2 * S
  let total_time := t1 + t2
  let average_speed := total_distance / total_time
  average_speed = 4.8 :=
  by
    sorry

end NUMINAMATH_GPT_average_speed_round_trip_l2271_227150


namespace NUMINAMATH_GPT_minimum_value_x_plus_y_l2271_227169

theorem minimum_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y * (x - y)^2 = 1) : x + y ≥ 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_x_plus_y_l2271_227169


namespace NUMINAMATH_GPT_average_of_consecutive_odds_is_24_l2271_227128

theorem average_of_consecutive_odds_is_24 (a b c d : ℤ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : d = 27) 
  (h5 : b = d - 2) (h6 : c = d - 4) (h7 : a = d - 6) 
  (h8 : ∀ x : ℤ, x % 2 = 1) :
  ((a + b + c + d) / 4) = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_of_consecutive_odds_is_24_l2271_227128


namespace NUMINAMATH_GPT_complete_square_result_l2271_227138

theorem complete_square_result (x : ℝ) :
  (x^2 - 4 * x - 3 = 0) → ((x - 2) ^ 2 = 7) :=
by sorry

end NUMINAMATH_GPT_complete_square_result_l2271_227138


namespace NUMINAMATH_GPT_rational_square_plus_one_positive_l2271_227146

theorem rational_square_plus_one_positive (x : ℚ) : x^2 + 1 > 0 :=
sorry

end NUMINAMATH_GPT_rational_square_plus_one_positive_l2271_227146


namespace NUMINAMATH_GPT_cost_price_of_article_l2271_227105

theorem cost_price_of_article (C SP1 SP2 G1 G2 : ℝ) 
  (h_SP1 : SP1 = 160) 
  (h_SP2 : SP2 = 220) 
  (h_gain_relation : G2 = 1.05 * G1) 
  (h_G1 : G1 = SP1 - C) 
  (h_G2 : G2 = SP2 - C) : C = 1040 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l2271_227105


namespace NUMINAMATH_GPT_common_root_polynomials_l2271_227163

theorem common_root_polynomials (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 1 = 0 ∧ x^2 + x + a = 0) ↔ (a = 1 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_common_root_polynomials_l2271_227163


namespace NUMINAMATH_GPT_value_of_p10_l2271_227120

def p (d e f x : ℝ) : ℝ := d * x^2 + e * x + f

theorem value_of_p10 (d e f : ℝ) 
  (h1 : p d e f 3 = p d e f 4)
  (h2 : p d e f 2 = p d e f 5)
  (h3 : p d e f 0 = 2) :
  p d e f 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_p10_l2271_227120


namespace NUMINAMATH_GPT_linear_function_not_in_second_quadrant_l2271_227144

-- Define the linear function y = x - 1.
def linear_function (x : ℝ) : ℝ := x - 1

-- Define the condition for a point to be in the second quadrant.
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- State that for any point (x, y) in the second quadrant, it does not satisfy y = x - 1.
theorem linear_function_not_in_second_quadrant {x y : ℝ} (h : in_second_quadrant x y) : linear_function x ≠ y :=
sorry

end NUMINAMATH_GPT_linear_function_not_in_second_quadrant_l2271_227144


namespace NUMINAMATH_GPT_find_f_l2271_227167

noncomputable def f : ℝ → ℝ := sorry

theorem find_f (f : ℝ → ℝ) (h₀ : f 0 = 2) 
  (h₁ : ∀ x y : ℝ, f (x * y) = f ((x^2 + y^2) / 2) + (x + y)^2) :
  ∀ x : ℝ, f x = 2 - 2 * x :=
sorry

end NUMINAMATH_GPT_find_f_l2271_227167


namespace NUMINAMATH_GPT_problem_fraction_eq_l2271_227186

theorem problem_fraction_eq (x : ℝ) :
  (x * (3 / 4) * (1 / 2) * 5060 = 759.0000000000001) ↔ (x = 0.4) :=
by
  sorry

end NUMINAMATH_GPT_problem_fraction_eq_l2271_227186


namespace NUMINAMATH_GPT_ned_did_not_wash_10_items_l2271_227118

theorem ned_did_not_wash_10_items :
  let short_sleeve_shirts := 9
  let long_sleeve_shirts := 21
  let pairs_of_pants := 15
  let jackets := 8
  let total_items := short_sleeve_shirts + long_sleeve_shirts + pairs_of_pants + jackets
  let washed_items := 43
  let not_washed_Items := total_items - washed_items
  not_washed_Items = 10 := by
sorry

end NUMINAMATH_GPT_ned_did_not_wash_10_items_l2271_227118


namespace NUMINAMATH_GPT_inequality_division_by_two_l2271_227177

theorem inequality_division_by_two (x y : ℝ) (h : x > y) : (x / 2) > (y / 2) := 
sorry

end NUMINAMATH_GPT_inequality_division_by_two_l2271_227177


namespace NUMINAMATH_GPT_maximum_value_problem_l2271_227187

theorem maximum_value_problem (x : ℝ) (h : 0 < x ∧ x < 4/3) : ∃ M, M = (4 / 3) ∧ ∀ y, 0 < y ∧ y < 4/3 → x * (4 - 3 * x) ≤ M :=
sorry

end NUMINAMATH_GPT_maximum_value_problem_l2271_227187


namespace NUMINAMATH_GPT_max_path_length_correct_l2271_227131

noncomputable def maxFlyPathLength : ℝ :=
  2 * Real.sqrt 2 + Real.sqrt 6 + 6

theorem max_path_length_correct :
  ∀ (fly_path_length : ℝ), (fly_path_length = maxFlyPathLength) :=
by
  intro fly_path_length
  sorry

end NUMINAMATH_GPT_max_path_length_correct_l2271_227131


namespace NUMINAMATH_GPT_does_not_determine_shape_l2271_227170

-- Definition of a function that checks whether given data determine the shape of a triangle
def determines_shape (data : Type) : Prop := sorry

-- Various conditions about data
def ratio_two_angles_included_side : Type := sorry
def ratios_three_angle_bisectors : Type := sorry
def ratios_three_side_lengths : Type := sorry
def ratio_angle_bisector_opposite_side : Type := sorry
def three_angles : Type := sorry

-- The main theorem stating that the ratio of an angle bisector to its corresponding opposite side does not uniquely determine the shape of a triangle.
theorem does_not_determine_shape :
  ¬determines_shape ratio_angle_bisector_opposite_side := sorry

end NUMINAMATH_GPT_does_not_determine_shape_l2271_227170


namespace NUMINAMATH_GPT_fruit_bowl_remaining_l2271_227126

-- Define the initial conditions
def oranges : Nat := 3
def lemons : Nat := 6
def fruits_eaten : Nat := 3

-- Define the total count of fruits initially
def total_fruits : Nat := oranges + lemons

-- The goal is to prove remaining fruits == 6
theorem fruit_bowl_remaining : total_fruits - fruits_eaten = 6 := by
  sorry

end NUMINAMATH_GPT_fruit_bowl_remaining_l2271_227126


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2271_227148

noncomputable def f (a x : ℝ) : ℝ := a^(x-1)

theorem problem_1 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  f a 3 = 4 → a = 2 :=
sorry

theorem problem_2 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  f a (Real.log a) = 100 → (a = 100 ∨ a = 1 / 10) :=
sorry

theorem problem_3 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  (a > 1 → f a (Real.log (1 / 100)) > f a (-2.1)) ∧
  (0 < a ∧ a < 1 → f a (Real.log (1 / 100)) < f a (-2.1)) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2271_227148


namespace NUMINAMATH_GPT_direct_proportion_function_l2271_227153

theorem direct_proportion_function (k : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = k * x) (h2 : f 3 = 6) : ∀ x, f x = 2 * x := by
  sorry

end NUMINAMATH_GPT_direct_proportion_function_l2271_227153


namespace NUMINAMATH_GPT_matrix_equation_l2271_227162

open Matrix

-- Define matrix N and the identity matrix I
def N : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 8], ![-4, -2]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1

-- Scalars p and q
def p : ℤ := 1
def q : ℤ := -26

-- Theorem statement
theorem matrix_equation :
  N * N = p • N + q • I :=
  by
    sorry

end NUMINAMATH_GPT_matrix_equation_l2271_227162


namespace NUMINAMATH_GPT_units_digit_7_pow_1995_l2271_227181

theorem units_digit_7_pow_1995 : 
  ∃ a : ℕ, a = 3 ∧ ∀ n : ℕ, (7^n % 10 = a) → ((n % 4) + 1 = 3) := 
by
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_1995_l2271_227181


namespace NUMINAMATH_GPT_find_m_l2271_227106

def is_good (n : ℤ) : Prop :=
  ¬ (∃ k : ℤ, |n| = k^2)

theorem find_m (m : ℤ) : (m % 4 = 3) → 
  (∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_good a ∧ is_good b ∧ is_good c ∧ (a * b * c) % 2 = 1 ∧ a + b + c = m) :=
sorry

end NUMINAMATH_GPT_find_m_l2271_227106


namespace NUMINAMATH_GPT_perfect_square_trinomial_implies_possible_m_values_l2271_227114

theorem perfect_square_trinomial_implies_possible_m_values (m : ℝ) :
  (∃ a : ℝ, ∀ x : ℝ, (x - a)^2 = x^2 - 2*m*x + 16) → (m = 4 ∨ m = -4) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_implies_possible_m_values_l2271_227114


namespace NUMINAMATH_GPT_f_eq_f_inv_implies_x_eq_0_l2271_227142

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1
noncomputable def f_inv (x : ℝ) : ℝ := (-1 + Real.sqrt (3 * x + 4)) / 3

theorem f_eq_f_inv_implies_x_eq_0 (x : ℝ) : f x = f_inv x → x = 0 :=
by
  sorry

end NUMINAMATH_GPT_f_eq_f_inv_implies_x_eq_0_l2271_227142


namespace NUMINAMATH_GPT_temperature_rise_per_hour_l2271_227160

-- Define the conditions
variables (x : ℕ) -- temperature rise per hour

-- Assume the given conditions
axiom power_outage : (3 : ℕ) * x = (6 : ℕ) * 4

-- State the proposition
theorem temperature_rise_per_hour : x = 8 :=
sorry

end NUMINAMATH_GPT_temperature_rise_per_hour_l2271_227160


namespace NUMINAMATH_GPT_average_screen_time_per_player_l2271_227124

def video_point_guard : ℕ := 130
def video_shooting_guard : ℕ := 145
def video_small_forward : ℕ := 85
def video_power_forward : ℕ := 60
def video_center : ℕ := 180
def total_video_time : ℕ := 
  video_point_guard + video_shooting_guard + video_small_forward + video_power_forward + video_center
def total_video_time_minutes : ℕ := total_video_time / 60
def number_of_players : ℕ := 5

theorem average_screen_time_per_player : total_video_time_minutes / number_of_players = 2 :=
  sorry

end NUMINAMATH_GPT_average_screen_time_per_player_l2271_227124


namespace NUMINAMATH_GPT_probability_of_sum_at_least_10_l2271_227185

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 6

theorem probability_of_sum_at_least_10 :
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_probability_of_sum_at_least_10_l2271_227185


namespace NUMINAMATH_GPT_parallel_to_l3_through_P_perpendicular_to_l3_through_P_l2271_227172

-- Define the lines l1, l2, and l3
def l1 (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0
def l2 (x y : ℝ) : Prop := x + 2 * y - 3 = 0
def l3 (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the intersection point P
def P := (1, 1)

-- Define the parallel line equation to l3 passing through P
def parallel_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Define the perpendicular line equation to l3 passing through P
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Prove the parallel line through P is 2x + y - 3 = 0
theorem parallel_to_l3_through_P : 
  ∀ (x y : ℝ), l1 x y → l2 x y → (parallel_line 1 1) := 
by 
  sorry

-- Prove the perpendicular line through P is x - 2y + 1 = 0
theorem perpendicular_to_l3_through_P : 
  ∀ (x y : ℝ), l1 x y → l2 x y → (perpendicular_line 1 1) := 
by 
  sorry

end NUMINAMATH_GPT_parallel_to_l3_through_P_perpendicular_to_l3_through_P_l2271_227172


namespace NUMINAMATH_GPT_math_problem_l2271_227100

open Classical

theorem math_problem (s x y : ℝ) (h₁ : s > 0) (h₂ : x^2 + y^2 ≠ 0) (h₃ : x * s^2 < y * s^2) :
  ¬(-x^2 < -y^2) ∧ ¬(-x^2 < y^2) ∧ ¬(x^2 < -y^2) ∧ ¬(x^2 > y^2) := by
  sorry

end NUMINAMATH_GPT_math_problem_l2271_227100


namespace NUMINAMATH_GPT_clock_hands_meeting_duration_l2271_227165

noncomputable def angle_between_clock_hands (h m : ℝ) : ℝ :=
  abs ((30 * h + m / 2) - (6 * m) % 360)

theorem clock_hands_meeting_duration : 
  ∃ n m : ℝ, 0 <= n ∧ n < m ∧ m < 60 ∧ angle_between_clock_hands 5 n = 120 ∧ angle_between_clock_hands 5 m = 120 ∧ m - n = 44 :=
sorry

end NUMINAMATH_GPT_clock_hands_meeting_duration_l2271_227165


namespace NUMINAMATH_GPT_find_smallest_m_l2271_227112

theorem find_smallest_m : ∃ m : ℕ, m > 0 ∧ (790 * m ≡ 1430 * m [MOD 30]) ∧ ∀ n : ℕ, n > 0 ∧ (790 * n ≡ 1430 * n [MOD 30]) → m ≤ n :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_m_l2271_227112


namespace NUMINAMATH_GPT_factor_expression_l2271_227109

theorem factor_expression (y : ℝ) : 84 * y ^ 13 + 210 * y ^ 26 = 42 * y ^ 13 * (2 + 5 * y ^ 13) :=
by sorry

end NUMINAMATH_GPT_factor_expression_l2271_227109


namespace NUMINAMATH_GPT_number_of_grouping_methods_l2271_227199

theorem number_of_grouping_methods : 
  let males := 5
  let females := 3
  let groups := 2
  let select_males := Nat.choose males groups
  let select_females := Nat.choose females groups
  let permute := Nat.factorial groups
  select_males * select_females * permute * permute = 60 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_grouping_methods_l2271_227199


namespace NUMINAMATH_GPT_transmit_data_time_l2271_227115

def total_chunks (blocks: ℕ) (chunks_per_block: ℕ) : ℕ := blocks * chunks_per_block

def transmit_time (total_chunks: ℕ) (chunks_per_second: ℕ) : ℕ := total_chunks / chunks_per_second

def time_in_minutes (transmit_time_seconds: ℕ) : ℕ := transmit_time_seconds / 60

theorem transmit_data_time :
  ∀ (blocks chunks_per_block chunks_per_second : ℕ),
    blocks = 150 →
    chunks_per_block = 256 →
    chunks_per_second = 200 →
    time_in_minutes (transmit_time (total_chunks blocks chunks_per_block) chunks_per_second) = 3 := by
  intros
  sorry

end NUMINAMATH_GPT_transmit_data_time_l2271_227115


namespace NUMINAMATH_GPT_minimum_rounds_l2271_227171

-- Given conditions based on the problem statement
variable (m : ℕ) (hm : m ≥ 17)
variable (players : Fin (2 * m)) -- Representing 2m players
variable (rounds : Fin (2 * m - 1)) -- Representing 2m - 1 rounds
variable (pairs : Fin m → Fin (2 * m) × Fin (2 * m)) -- Pairing for each of the m pairs in each round

-- Statement of the proof problem
theorem minimum_rounds (h1 : ∀ i j, i ≠ j → ∃! (k : Fin m), pairs k = (i, j) ∨ pairs k = (j, i))
(h2 : ∀ k : Fin m, (pairs k).fst ≠ (pairs k).snd)
(h3 : ∀ i j, i ≠ j → ∃ r : Fin (2 * m - 1), (∃ k : Fin m, pairs k = (i, j)) ∧ (∃ k : Fin m, pairs k = (j, i))) :
∃ (n : ℕ), n = m - 1 ∧ ∀ s : Fin 4 → Fin (2 * m), (∀ i j, i ≠ j → ¬ ∃ r : Fin n, ∃ k : Fin m, pairs k = (s i, s j)) ∨ (∃ r1 r2 : Fin n, ∃ i j, i ≠ j ∧ ∃ k1 k2 : Fin m, pairs k1 = (s i, s j) ∧ pairs k2 = (s j, s i)) :=
sorry

end NUMINAMATH_GPT_minimum_rounds_l2271_227171


namespace NUMINAMATH_GPT_A_subset_B_l2271_227149

def A : Set ℤ := { x : ℤ | ∃ k : ℤ, x = 4 * k + 1 }
def B : Set ℤ := { x : ℤ | ∃ k : ℤ, x = 2 * k - 1 }

theorem A_subset_B : A ⊆ B :=
  sorry

end NUMINAMATH_GPT_A_subset_B_l2271_227149


namespace NUMINAMATH_GPT_probability_of_five_3s_is_099_l2271_227102

-- Define conditions
def number_of_dice : ℕ := 15
def rolled_value : ℕ := 3
def probability_of_3 : ℚ := 1 / 8
def number_of_successes : ℕ := 5
def probability_of_not_3 : ℚ := 7 / 8

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability calculation
def probability_exactly_five_3s : ℚ :=
  binomial_coefficient number_of_dice number_of_successes *
  probability_of_3 ^ number_of_successes *
  probability_of_not_3 ^ (number_of_dice - number_of_successes)

theorem probability_of_five_3s_is_099 :
  probability_exactly_five_3s = 0.099 := by
  sorry -- Proof to be filled in later

end NUMINAMATH_GPT_probability_of_five_3s_is_099_l2271_227102


namespace NUMINAMATH_GPT_quadratic_eq_transformed_l2271_227154

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2 * x - 7 = 0

-- Define the form to transform to using completing the square method
def transformed_eq (x : ℝ) : Prop := (x - 1)^2 = 8

-- The theorem to be proved
theorem quadratic_eq_transformed (x : ℝ) :
  quadratic_eq x → transformed_eq x :=
by
  intros h
  -- here we would use steps of completing the square to transform the equation
  sorry

end NUMINAMATH_GPT_quadratic_eq_transformed_l2271_227154


namespace NUMINAMATH_GPT_homework_total_time_l2271_227147

theorem homework_total_time :
  ∀ (j g p : ℕ),
  j = 18 →
  g = j - 6 →
  p = 2 * g - 4 →
  j + g + p = 50 :=
by
  intros j g p h1 h2 h3
  sorry

end NUMINAMATH_GPT_homework_total_time_l2271_227147


namespace NUMINAMATH_GPT_factorial_divisibility_l2271_227164

theorem factorial_divisibility 
  {n : ℕ} 
  (hn : bit0 (n.bits.count 1) == 1995) : 
  (2^(n-1995)) ∣ n! := 
sorry

end NUMINAMATH_GPT_factorial_divisibility_l2271_227164


namespace NUMINAMATH_GPT_minimum_additional_small_bottles_needed_l2271_227173

-- Definitions from the problem conditions
def small_bottle_volume : ℕ := 45
def large_bottle_total_volume : ℕ := 600
def initial_volume_in_large_bottle : ℕ := 90

-- The proof problem: How many more small bottles does Jasmine need to fill the large bottle?
theorem minimum_additional_small_bottles_needed : 
  (large_bottle_total_volume - initial_volume_in_large_bottle + small_bottle_volume - 1) / small_bottle_volume = 12 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_additional_small_bottles_needed_l2271_227173


namespace NUMINAMATH_GPT_probability_diagonals_intersect_inside_decagon_l2271_227135

/-- Two diagonals of a regular decagon are chosen. 
  What is the probability that their intersection lies inside the decagon?
-/
theorem probability_diagonals_intersect_inside_decagon : 
  let num_diagonals := 35
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210
  let probability := num_intersecting_pairs / num_pairs
  probability = 42 / 119 :=
by
  -- Definitions based on the conditions
  let num_diagonals := (10 * (10 - 3)) / 2
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210

  -- Simplified probability
  let probability := num_intersecting_pairs / num_pairs

  -- Sorry used to skip the proof
  sorry

end NUMINAMATH_GPT_probability_diagonals_intersect_inside_decagon_l2271_227135


namespace NUMINAMATH_GPT_gcd_polynomial_even_multiple_of_97_l2271_227141

theorem gcd_polynomial_even_multiple_of_97 (b : ℤ) (k : ℤ) (h_b : b = 2 * 97 * k) :
  Int.gcd (3 * b^2 + 41 * b + 74) (b + 19) = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_even_multiple_of_97_l2271_227141


namespace NUMINAMATH_GPT_area_of_large_hexagon_eq_270_l2271_227196

noncomputable def area_large_hexagon (area_shaded : ℝ) (n_small_hexagons_shaded : ℕ) (n_small_hexagons_large : ℕ): ℝ :=
  let area_one_small_hexagon := area_shaded / n_small_hexagons_shaded
  area_one_small_hexagon * n_small_hexagons_large

theorem area_of_large_hexagon_eq_270 :
  area_large_hexagon 180 6 7 = 270 := by
  sorry

end NUMINAMATH_GPT_area_of_large_hexagon_eq_270_l2271_227196


namespace NUMINAMATH_GPT_dividend_50100_l2271_227129

theorem dividend_50100 (D Q R : ℕ) (h1 : D = 20 * Q) (h2 : D = 10 * R) (h3 : R = 100) : 
    D * Q + R = 50100 := by
  sorry

end NUMINAMATH_GPT_dividend_50100_l2271_227129


namespace NUMINAMATH_GPT_prime_solution_exists_l2271_227132

theorem prime_solution_exists (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  p^2 + 1 = 74 * (q^2 + r^2) → (p = 31 ∧ q = 2 ∧ r = 3) :=
by
  sorry

end NUMINAMATH_GPT_prime_solution_exists_l2271_227132


namespace NUMINAMATH_GPT_product_of_a_and_c_l2271_227117

theorem product_of_a_and_c (a b c : ℝ) (h1 : a + b + c = 100) (h2 : a - b = 20) (h3 : b - c = 30) : a * c = 378.07 :=
by
  sorry

end NUMINAMATH_GPT_product_of_a_and_c_l2271_227117


namespace NUMINAMATH_GPT_min_value_expression_l2271_227192

theorem min_value_expression (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (min ((1 / ((1 - x) * (1 - y) * (1 - z))) + (1 / ((1 + x) * (1 + y) * (1 + z))) + (x * y * z)) 2) = 2 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_expression_l2271_227192


namespace NUMINAMATH_GPT_number_of_pieces_correct_l2271_227107

-- Define the dimensions of the pan
def pan_length : ℕ := 30
def pan_width : ℕ := 24

-- Define the dimensions of each piece of brownie
def piece_length : ℕ := 3
def piece_width : ℕ := 2

-- Calculate the area of the pan
def pan_area : ℕ := pan_length * pan_width

-- Calculate the area of each piece of brownie
def piece_area : ℕ := piece_length * piece_width

-- The proof problem statement
theorem number_of_pieces_correct : (pan_area / piece_area) = 120 :=
by sorry

end NUMINAMATH_GPT_number_of_pieces_correct_l2271_227107


namespace NUMINAMATH_GPT_std_dev_samples_l2271_227134

def sample_A := [82, 84, 84, 86, 86, 86, 88, 88, 88, 88]
def sample_B := [84, 86, 86, 88, 88, 88, 90, 90, 90, 90]

noncomputable def std_dev (l : List ℕ) :=
  let n := l.length
  let mean := (l.sum : ℚ) / n
  let variance := (l.map (λ x => (x - mean) * (x - mean))).sum / n
  variance.sqrt

theorem std_dev_samples :
  std_dev sample_A = std_dev sample_B := 
sorry

end NUMINAMATH_GPT_std_dev_samples_l2271_227134


namespace NUMINAMATH_GPT_valentines_left_l2271_227108

theorem valentines_left (initial_valentines given_away : ℕ) (h_initial : initial_valentines = 30) (h_given : given_away = 8) :
  initial_valentines - given_away = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_valentines_left_l2271_227108


namespace NUMINAMATH_GPT_smallest_value_of_k_l2271_227116

theorem smallest_value_of_k (k : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + k = 5) ↔ k >= 9 := 
sorry

end NUMINAMATH_GPT_smallest_value_of_k_l2271_227116


namespace NUMINAMATH_GPT_what_to_do_first_l2271_227151

-- Definition of the conditions
def eat_or_sleep_to_survive (days_without_eat : ℕ) (days_without_sleep : ℕ) : Prop :=
  (days_without_eat = 7 → days_without_sleep ≠ 7) ∨ (days_without_sleep = 7 → days_without_eat ≠ 7)

-- Theorem statement based on the problem and its conditions
theorem what_to_do_first (days_without_eat days_without_sleep : ℕ) :
  days_without_eat = 7 ∨ days_without_sleep = 7 →
  eat_or_sleep_to_survive days_without_eat days_without_sleep :=
by sorry

end NUMINAMATH_GPT_what_to_do_first_l2271_227151


namespace NUMINAMATH_GPT_cosine_difference_l2271_227130

theorem cosine_difference (A B : ℝ) (h1 : Real.sin A + Real.sin B = 3/2) (h2 : Real.cos A + Real.cos B = 2) :
  Real.cos (A - B) = 17 / 8 :=
by
  sorry

end NUMINAMATH_GPT_cosine_difference_l2271_227130


namespace NUMINAMATH_GPT_find_sum_of_abc_l2271_227139

noncomputable def m (a b c : ℕ) : ℝ := a - b * Real.sqrt c

theorem find_sum_of_abc (a b c : ℕ) (ha : ¬ (c % 2 = 0) ∧ ∀ p : ℕ, Prime p → ¬ p * p ∣ c) 
  (hprob : ((30 - m a b c) ^ 2 / 30 ^ 2 = 0.75)) : a + b + c = 48 := 
by
  sorry

end NUMINAMATH_GPT_find_sum_of_abc_l2271_227139


namespace NUMINAMATH_GPT_problem_l2271_227180

def is_arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (M m : ℕ)

-- Conditions
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 1 ≥ 1
axiom h3 : a 2 ≤ 5
axiom h4 : a 5 ≥ 8

-- Sum function for arithmetic sequence
axiom h5 : ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)

-- Definition of M and m based on S_15
axiom hM : M = max (S 15)
axiom hm : m = min (S 15)

theorem problem (h : S 15 = M + m) : M + m = 600 :=
  sorry

end NUMINAMATH_GPT_problem_l2271_227180


namespace NUMINAMATH_GPT_machine_probabilities_at_least_one_first_class_component_l2271_227166

theorem machine_probabilities : 
  (∃ (PA PB PC : ℝ), 
  PA * (1 - PB) = 1/4 ∧ 
  PB * (1 - PC) = 1/12 ∧ 
  PA * PC = 2/9 ∧ 
  PA = 1/3 ∧ 
  PB = 1/4 ∧ 
  PC = 2/3) 
:=
sorry

theorem at_least_one_first_class_component : 
  ∃ (PA PB PC : ℝ), 
  PA * (1 - PB) = 1/4 ∧ 
  PB * (1 - PC) = 1/12 ∧ 
  PA * PC = 2/9 ∧ 
  PA = 1/3 ∧ 
  PB = 1/4 ∧ 
  PC = 2/3 ∧ 
  1 - (1 - PA) * (1 - PB) * (1 - PC) = 5/6
:=
sorry

end NUMINAMATH_GPT_machine_probabilities_at_least_one_first_class_component_l2271_227166


namespace NUMINAMATH_GPT_solution_set_equiv_l2271_227179

def solution_set (x : ℝ) : Prop := 2 * x - 6 < 0

theorem solution_set_equiv (x : ℝ) : solution_set x ↔ x < 3 := by
  sorry

end NUMINAMATH_GPT_solution_set_equiv_l2271_227179


namespace NUMINAMATH_GPT_Sidney_JumpJacks_Tuesday_l2271_227104

variable (JumpJacksMonday JumpJacksTuesday JumpJacksWednesday JumpJacksThursday : ℕ)
variable (SidneyTotalJumpJacks BrookeTotalJumpJacks : ℕ)

-- Given conditions
axiom H1 : JumpJacksMonday = 20
axiom H2 : JumpJacksWednesday = 40
axiom H3 : JumpJacksThursday = 50
axiom H4 : BrookeTotalJumpJacks = 3 * SidneyTotalJumpJacks
axiom H5 : BrookeTotalJumpJacks = 438

-- Prove Sidney's JumpJacks on Tuesday
theorem Sidney_JumpJacks_Tuesday : JumpJacksTuesday = 36 :=
by
  sorry

end NUMINAMATH_GPT_Sidney_JumpJacks_Tuesday_l2271_227104


namespace NUMINAMATH_GPT_max_a3_in_arith_geo_sequences_l2271_227140

theorem max_a3_in_arith_geo_sequences
  (a1 a2 a3 : ℝ) (b1 b2 b3 : ℝ)
  (h1 : a1 + a2 + a3 = 15)
  (h2 : a2 = ((a1 + a3) / 2))
  (h3 : b1 * b2 * b3 = 27)
  (h4 : (a1 + b1) * (a3 + b3) = (a2 + b2) ^ 2)
  (h5 : a1 + b1 > 0)
  (h6 : a2 + b2 > 0)
  (h7 : a3 + b3 > 0) :
  a3 ≤ 59 := sorry

end NUMINAMATH_GPT_max_a3_in_arith_geo_sequences_l2271_227140


namespace NUMINAMATH_GPT_find_x_l2271_227136

def vector (α : Type*) := α × α

def parallel (a b : vector ℝ) : Prop :=
a.1 * b.2 - a.2 * b.1 = 0

theorem find_x (x : ℝ) (a b : vector ℝ)
  (ha : a = (1, 2))
  (hb : b = (x, 4))
  (h : parallel a b) : x = 2 :=
by sorry

end NUMINAMATH_GPT_find_x_l2271_227136


namespace NUMINAMATH_GPT_eval_F_at_4_f_5_l2271_227198

def f (a : ℤ) : ℤ := 3 * a - 6
def F (a : ℤ) (b : ℤ) : ℤ := 2 * b ^ 2 + 3 * a

theorem eval_F_at_4_f_5 : F 4 (f 5) = 174 := by
  sorry

end NUMINAMATH_GPT_eval_F_at_4_f_5_l2271_227198


namespace NUMINAMATH_GPT_contact_prob_correct_l2271_227110

-- Define the conditions.
def m : ℕ := 6
def n : ℕ := 7
variable (p : ℝ)

-- Define the probability computation.
def prob_contact : ℝ := 1 - (1 - p)^(m * n)

-- Formal statement of the problem.
theorem contact_prob_correct : prob_contact p = 1 - (1 - p)^42 := by
  sorry

end NUMINAMATH_GPT_contact_prob_correct_l2271_227110
