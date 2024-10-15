import Mathlib

namespace NUMINAMATH_GPT_correct_operation_only_l2351_235138

theorem correct_operation_only (a b x y : ℝ) : 
  (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) ∧ 
  (4 * x^2 * y - x^2 * y ≠ 3) ∧ 
  ((a + b)^2 ≠ a^2 + b^2) ∧
  ((ab)^3 = a^3 * b^3) := 
by 
  sorry

end NUMINAMATH_GPT_correct_operation_only_l2351_235138


namespace NUMINAMATH_GPT_expansion_coeff_l2351_235170

theorem expansion_coeff (a b : ℝ) (x : ℝ) (h : (1 + a * x) ^ 5 = 1 + 10 * x + b * x^2 + a^5 * x^5) :
  b = 40 :=
sorry

end NUMINAMATH_GPT_expansion_coeff_l2351_235170


namespace NUMINAMATH_GPT_scientific_notation_correct_l2351_235112

def num : ℝ := 1040000000

theorem scientific_notation_correct : num = 1.04 * 10^9 := by
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l2351_235112


namespace NUMINAMATH_GPT_initial_erasers_count_l2351_235192

noncomputable def erasers_lost := 42
noncomputable def erasers_ended_up_with := 53

theorem initial_erasers_count (initial_erasers : ℕ) : 
  initial_erasers_ended_up_with = initial_erasers - erasers_lost → initial_erasers = 95 :=
by
  sorry

end NUMINAMATH_GPT_initial_erasers_count_l2351_235192


namespace NUMINAMATH_GPT_range_of_m_l2351_235164

theorem range_of_m {m : ℝ} (h : ∃ x : ℝ, 2 < x ∧ x < 3 ∧ x^2 + 2 * x - m = 0) : 8 < m ∧ m < 15 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2351_235164


namespace NUMINAMATH_GPT_competition_score_l2351_235199

theorem competition_score
    (x : ℕ)
    (h1 : 20 ≥ x)
    (h2 : 5 * x - (20 - x) = 70) :
    x = 15 :=
sorry

end NUMINAMATH_GPT_competition_score_l2351_235199


namespace NUMINAMATH_GPT_distance_to_directrix_l2351_235114

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 64 + y^2 / 28 = 1

noncomputable def left_focus : ℝ × ℝ := (-6, 0)

noncomputable def right_focus : ℝ × ℝ := (6, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_to_directrix (P : ℝ × ℝ) (hP : ellipse P.1 P.2) (hPF1 : distance P left_focus = 4) :
  distance P right_focus * 4 / 3 = 16 :=
sorry

end NUMINAMATH_GPT_distance_to_directrix_l2351_235114


namespace NUMINAMATH_GPT_prove_avg_mark_of_batch3_l2351_235135

noncomputable def avg_mark_of_batch3 (A1 A2 A3 : ℕ) (Marks1 Marks2 Marks3 : ℚ) : Prop :=
  A1 = 40 ∧ A2 = 50 ∧ A3 = 60 ∧ Marks1 = 45 ∧ Marks2 = 55 ∧ 
  (A1 * Marks1 + A2 * Marks2 + A3 * Marks3) / (A1 + A2 + A3) = 56.333333333333336 → 
  Marks3 = 65

theorem prove_avg_mark_of_batch3 : avg_mark_of_batch3 40 50 60 45 55 65 :=
by
  unfold avg_mark_of_batch3
  sorry

end NUMINAMATH_GPT_prove_avg_mark_of_batch3_l2351_235135


namespace NUMINAMATH_GPT_installation_quantities_l2351_235158

theorem installation_quantities :
  ∃ x1 x2 x3 : ℕ, x1 = 22 ∧ x2 = 88 ∧ x3 = 22 ∧
  (x1 + x2 + x3 ≥ 100) ∧
  (x2 = 4 * x1) ∧
  (∃ k : ℕ, x3 = k * x1) ∧
  (5 * x3 = x2 + 22) :=
  by {
    -- We are simply stating the equivalence and supporting conditions.
    -- Here, we will use 'sorry' as a placeholder.
    sorry
  }

end NUMINAMATH_GPT_installation_quantities_l2351_235158


namespace NUMINAMATH_GPT_problem_1_problem_2_l2351_235169

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Conditions
axiom h1 : ∀ n : ℕ, 2 * S n = a (n + 1) - 2^(n + 1) + 1
axiom h2 : a 2 + 5 = a 1 + (a 3 - a 2)

-- Problem 1: Prove the value of a₁
theorem problem_1 : a 1 = 1 := sorry

-- Problem 2: Find the general term formula for the sequence {aₙ}
theorem problem_2 : ∀ n : ℕ, a n = 3^n - 2^n := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2351_235169


namespace NUMINAMATH_GPT_find_lost_card_number_l2351_235148

theorem find_lost_card_number (n : ℕ) (S : ℕ) (x : ℕ) 
  (h1 : S = n * (n + 1) / 2) 
  (h2 : S - x = 101) 
  (h3 : n = 14) : 
  x = 4 := 
by sorry

end NUMINAMATH_GPT_find_lost_card_number_l2351_235148


namespace NUMINAMATH_GPT_mean_second_set_l2351_235163

theorem mean_second_set (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 62) :
  (48 + 62 + 98 + 124 + x) / 5 = 78 :=
sorry

end NUMINAMATH_GPT_mean_second_set_l2351_235163


namespace NUMINAMATH_GPT_represents_not_much_different_l2351_235107

def not_much_different_from (x : ℝ) (c : ℝ) : Prop := x - c ≤ 0

theorem represents_not_much_different {x : ℝ} :
  (not_much_different_from x 2023) = (x - 2023 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_represents_not_much_different_l2351_235107


namespace NUMINAMATH_GPT_sum_of_a6_and_a7_l2351_235127

theorem sum_of_a6_and_a7 (a : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 2 + a 3 + a 10 + a 11 = 48) : a 6 + a 7 = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_a6_and_a7_l2351_235127


namespace NUMINAMATH_GPT_minimize_sum_of_reciprocals_l2351_235123

theorem minimize_sum_of_reciprocals (a b : ℕ) (h : 4 * a + b = 6) : 
  a = 1 ∧ b = 2 ∨ a = 2 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_minimize_sum_of_reciprocals_l2351_235123


namespace NUMINAMATH_GPT_complement_union_eq_l2351_235155

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def S : Set ℕ := {1, 3, 5}
def T : Set ℕ := {3, 6}

theorem complement_union_eq : (U \ (S ∪ T)) = {2, 4, 7, 8} :=
by {
  sorry
}

end NUMINAMATH_GPT_complement_union_eq_l2351_235155


namespace NUMINAMATH_GPT_charity_dinner_cost_l2351_235145

def cost_of_rice_per_plate : ℝ := 0.10
def cost_of_chicken_per_plate : ℝ := 0.40
def number_of_plates : ℕ := 100

theorem charity_dinner_cost : 
  cost_of_rice_per_plate + cost_of_chicken_per_plate * number_of_plates = 50 :=
by
  sorry

end NUMINAMATH_GPT_charity_dinner_cost_l2351_235145


namespace NUMINAMATH_GPT_license_plates_count_l2351_235119

theorem license_plates_count :
  let num_vowels := 5
  let num_letters := 26
  let num_odd_digits := 5
  let num_even_digits := 5
  num_vowels * num_letters * num_letters * num_odd_digits * num_even_digits = 84500 :=
by
  sorry

end NUMINAMATH_GPT_license_plates_count_l2351_235119


namespace NUMINAMATH_GPT_maximum_candy_leftover_l2351_235179

theorem maximum_candy_leftover (x : ℕ) 
  (h1 : ∀ (bags : ℕ), bags = 12 → x ≥ bags * 10)
  (h2 : ∃ (leftover : ℕ), leftover < 12 ∧ leftover = (x - 120) % 12) : 
  ∃ (leftover : ℕ), leftover = 11 :=
by
  sorry

end NUMINAMATH_GPT_maximum_candy_leftover_l2351_235179


namespace NUMINAMATH_GPT_sum_of_number_is_8_l2351_235131

theorem sum_of_number_is_8 (x v : ℝ) (h1 : 0.75 * x + 2 = v) (h2 : x = 8.0) : v = 8.0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_number_is_8_l2351_235131


namespace NUMINAMATH_GPT_find_k_value_l2351_235159

theorem find_k_value (x y k : ℝ) 
  (h1 : x - 3 * y = k + 2) 
  (h2 : x - y = 4) 
  (h3 : 3 * x + y = -8) : 
  k = 12 := 
  by {
    sorry
  }

end NUMINAMATH_GPT_find_k_value_l2351_235159


namespace NUMINAMATH_GPT_max_value_A_l2351_235111

noncomputable def A (x y : ℝ) : ℝ :=
  ((x^2 - y) * Real.sqrt (y + x^3 - x * y) + (y^2 - x) * Real.sqrt (x + y^3 - x * y) + 1) /
  ((x - y)^2 + 1)

theorem max_value_A (x y : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) :
  A x y ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_A_l2351_235111


namespace NUMINAMATH_GPT_find_angle_A_l2351_235132

theorem find_angle_A (A B C : ℝ) (a b c : ℝ) :
  (a * Real.sin A = b * Real.sin B + (c - b) * Real.sin C)
  → (A = π / 3) :=
sorry

end NUMINAMATH_GPT_find_angle_A_l2351_235132


namespace NUMINAMATH_GPT_exponentiation_properties_l2351_235157

theorem exponentiation_properties:
  (10^6) * (10^2)^3 / 10^4 = 10^8 :=
by
  sorry

end NUMINAMATH_GPT_exponentiation_properties_l2351_235157


namespace NUMINAMATH_GPT_no_such_set_exists_l2351_235180

open Nat Set

theorem no_such_set_exists (M : Set ℕ) : 
  (∀ m : ℕ, m > 1 → ∃ a b : ℕ, a ∈ M ∧ b ∈ M ∧ a + b = m) →
  (∀ a b c d : ℕ, a ∈ M → b ∈ M → c ∈ M → d ∈ M → 
    a > 10 → b > 10 → c > 10 → d > 10 → a + b = c + d → a = c ∨ a = d) → 
  False := by
  sorry

end NUMINAMATH_GPT_no_such_set_exists_l2351_235180


namespace NUMINAMATH_GPT_some_value_correct_l2351_235121

theorem some_value_correct (w x y : ℝ) (some_value : ℝ)
  (h1 : 3 / w + some_value = 3 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  some_value = 6 := by
  sorry

end NUMINAMATH_GPT_some_value_correct_l2351_235121


namespace NUMINAMATH_GPT_percent_research_and_development_is_9_l2351_235197

-- Define given percentages
def percent_transportation := 20
def percent_utilities := 5
def percent_equipment := 4
def percent_supplies := 2

-- Define degree representation and calculate percent for salaries
def degrees_in_circle := 360
def degrees_salaries := 216
def percent_salaries := (degrees_salaries * 100) / degrees_in_circle

-- Define the total percentage representation
def total_percent := 100
def known_percent := percent_transportation + percent_utilities + percent_equipment + percent_supplies + percent_salaries

-- Calculate the percent for research and development
def percent_research_and_development := total_percent - known_percent

-- Theorem statement
theorem percent_research_and_development_is_9 : percent_research_and_development = 9 :=
by 
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_percent_research_and_development_is_9_l2351_235197


namespace NUMINAMATH_GPT_romanov_family_savings_l2351_235178

theorem romanov_family_savings :
  let cost_multi_tariff_meter := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let night_consumption := 230
  let day_consumption := monthly_consumption - night_consumption
  let night_rate := 3.4
  let day_rate := 5.2
  let standard_rate := 4.6
  let yearly_cost_multi_tariff :=
    (night_consumption * night_rate * 12) +
    (day_consumption * day_rate * 12)
  let total_cost_multi_tariff :=
    cost_multi_tariff_meter + installation_cost + (yearly_cost_multi_tariff * 3)
  let yearly_cost_standard :=
    monthly_consumption * standard_rate * 12
  let total_cost_standard :=
    yearly_cost_standard * 3
  total_cost_standard - total_cost_multi_tariff = 3824 := 
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_romanov_family_savings_l2351_235178


namespace NUMINAMATH_GPT_alice_ate_more_l2351_235139

theorem alice_ate_more (cookies : Fin 8 → ℕ) (h_alice : cookies 0 = 8) (h_tom : cookies 7 = 1) :
  cookies 0 - cookies 7 = 7 :=
by
  -- Placeholder for the actual proof, which is not required here
  sorry

end NUMINAMATH_GPT_alice_ate_more_l2351_235139


namespace NUMINAMATH_GPT_each_person_bid_count_l2351_235191

-- Define the conditions and initial values
noncomputable def auctioneer_price_increase : ℕ := 5
noncomputable def initial_price : ℕ := 15
noncomputable def final_price : ℕ := 65
noncomputable def number_of_bidders : ℕ := 2

-- Define the proof statement
theorem each_person_bid_count : 
  (final_price - initial_price) / auctioneer_price_increase / number_of_bidders = 5 :=
by sorry

end NUMINAMATH_GPT_each_person_bid_count_l2351_235191


namespace NUMINAMATH_GPT_gcd_example_l2351_235190

-- Define the two numbers
def a : ℕ := 102
def b : ℕ := 238

-- Define the GCD of a and b
def gcd_ab : ℕ :=
  Nat.gcd a b

-- The expected result of the GCD
def expected_gcd : ℕ := 34

-- Prove that the GCD of a and b is equal to the expected GCD
theorem gcd_example : gcd_ab = expected_gcd := by
  sorry

end NUMINAMATH_GPT_gcd_example_l2351_235190


namespace NUMINAMATH_GPT_count_numbers_with_remainder_7_dividing_65_l2351_235142

theorem count_numbers_with_remainder_7_dividing_65 : 
  (∃ n : ℕ, n > 7 ∧ n ∣ 58 ∧ 65 % n = 7) ∧ 
  (∀ m : ℕ, m > 7 ∧ m ∣ 58 ∧ 65 % m = 7 → m = 29 ∨ m = 58) :=
sorry

end NUMINAMATH_GPT_count_numbers_with_remainder_7_dividing_65_l2351_235142


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2351_235118

theorem solution_set_of_inequality (x : ℝ) :
  x * |x - 1| > 0 ↔ (0 < x ∧ x < 1) ∨ (x > 1) :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2351_235118


namespace NUMINAMATH_GPT_weight_difference_l2351_235113

theorem weight_difference :
  let Box_A := 2.4
  let Box_B := 5.3
  let Box_C := 13.7
  let Box_D := 7.1
  let Box_E := 10.2
  let Box_F := 3.6
  let Box_G := 9.5
  max Box_A (max Box_B (max Box_C (max Box_D (max Box_E (max Box_F Box_G))))) -
  min Box_A (min Box_B (min Box_C (min Box_D (min Box_E (min Box_F Box_G))))) = 11.3 :=
by
  sorry

end NUMINAMATH_GPT_weight_difference_l2351_235113


namespace NUMINAMATH_GPT_min_value_is_neg_one_l2351_235125

noncomputable def find_min_value (a b c : ℝ) (h : 4 * a^2 - 2 * a * b + b^2 - c = 0) (h1 : 0 < c) (h2 : |2 * a + b| = sorry) : ℝ :=
  1 / a + 2 / b + 4 / c

theorem min_value_is_neg_one (a b c : ℝ) (h : 4 * a^2 - 2 * a * b + b^2 - c = 0) (h1 : 0 < c) (h2 : |2 * a + b| = sorry) : 
  find_min_value a b c h h1 h2 = -1 :=
sorry

end NUMINAMATH_GPT_min_value_is_neg_one_l2351_235125


namespace NUMINAMATH_GPT_max_value_of_y_over_x_l2351_235167

theorem max_value_of_y_over_x {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 2 * y = 3) :
  y / x ≤ 9 / 8 :=
sorry

end NUMINAMATH_GPT_max_value_of_y_over_x_l2351_235167


namespace NUMINAMATH_GPT_fraction_simplify_l2351_235101

theorem fraction_simplify :
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_fraction_simplify_l2351_235101


namespace NUMINAMATH_GPT_inequality_solution_l2351_235105

theorem inequality_solution (x y : ℝ) (h : 5 * x > -5 * y) : x + y > 0 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2351_235105


namespace NUMINAMATH_GPT_probability_x_plus_2y_lt_6_l2351_235185

noncomputable def prob_x_plus_2y_lt_6 : ℚ :=
  let rect_area : ℚ := (4 : ℚ) * 3
  let quad_area : ℚ := (4 : ℚ) * 1 + (1 / 2 : ℚ) * 4 * 2
  quad_area / rect_area

theorem probability_x_plus_2y_lt_6 :
  prob_x_plus_2y_lt_6 = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_x_plus_2y_lt_6_l2351_235185


namespace NUMINAMATH_GPT_person_B_spheres_needed_l2351_235104

-- Translate conditions to Lean definitions
def sum_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6
def sum_triangulars (m : ℕ) : ℕ := (m * (m + 1) * (m + 2)) / 6

-- Define the main theorem
theorem person_B_spheres_needed (n m : ℕ) (hA : sum_squares n = 2109)
    (hB : m ≥ 25) : sum_triangulars m = 2925 :=
    sorry

end NUMINAMATH_GPT_person_B_spheres_needed_l2351_235104


namespace NUMINAMATH_GPT_polar_to_cartesian_circle_l2351_235116

theorem polar_to_cartesian_circle :
  ∀ (r : ℝ) (x y : ℝ), r = 3 → r = Real.sqrt (x^2 + y^2) → x^2 + y^2 = 9 :=
by
  intros r x y hr h
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_circle_l2351_235116


namespace NUMINAMATH_GPT_no_integer_solution_l2351_235144

theorem no_integer_solution (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_no_integer_solution_l2351_235144


namespace NUMINAMATH_GPT_triangle_ab_length_triangle_roots_quadratic_l2351_235154

open Real

noncomputable def right_angled_triangle_length_ab (p s : ℝ) : ℝ :=
  (p / 2) - sqrt ((p / 2)^2 - 2 * s)

noncomputable def right_angled_triangle_quadratic (p s : ℝ) : Polynomial ℝ :=
  Polynomial.X^2 - Polynomial.C ((p / 2) + sqrt ((p / 2)^2 - 2 * s)) * Polynomial.X
    + Polynomial.C (2 * s)

theorem triangle_ab_length (p s : ℝ) :
  ∃ (AB : ℝ), AB = right_angled_triangle_length_ab p s ∧
    ∃ (AC BC : ℝ), (AC + BC + AB = p) ∧ (1 / 2 * BC * AC = s) :=
by
  use right_angled_triangle_length_ab p s
  sorry

theorem triangle_roots_quadratic (p s : ℝ) :
  ∃ (AC BC : ℝ), AC + BC = (p / 2) + sqrt ((p / 2)^2 - 2 * s) ∧
    AC * BC = 2 * s ∧
    (Polynomial.aeval AC (right_angled_triangle_quadratic p s) = 0) ∧
    (Polynomial.aeval BC (right_angled_triangle_quadratic p s) = 0) :=
by
  sorry

end NUMINAMATH_GPT_triangle_ab_length_triangle_roots_quadratic_l2351_235154


namespace NUMINAMATH_GPT_find_g_inverse_75_l2351_235147

noncomputable def g (x : ℝ) : ℝ := 3 * x^3 - 6

theorem find_g_inverse_75 : g⁻¹ 75 = 3 := sorry

end NUMINAMATH_GPT_find_g_inverse_75_l2351_235147


namespace NUMINAMATH_GPT_establish_model_steps_correct_l2351_235194

-- Define each step as a unique identifier
inductive Step : Type
| observe_pose_questions
| propose_assumptions
| express_properties
| test_or_revise

open Step

-- The sequence of steps to establish a mathematical model for population change
def correct_model_steps : List Step :=
  [observe_pose_questions, propose_assumptions, express_properties, test_or_revise]

-- The correct answer is the sequence of steps in the correct order
theorem establish_model_steps_correct :
  correct_model_steps = [observe_pose_questions, propose_assumptions, express_properties, test_or_revise] :=
  by sorry

end NUMINAMATH_GPT_establish_model_steps_correct_l2351_235194


namespace NUMINAMATH_GPT_arccos_one_eq_zero_l2351_235137

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_GPT_arccos_one_eq_zero_l2351_235137


namespace NUMINAMATH_GPT_inequality_1_inequality_2_inequality_4_l2351_235130

variable {a b : ℝ}

def condition (a b : ℝ) : Prop := (1/a < 1/b) ∧ (1/b < 0)

theorem inequality_1 (ha : a < 0) (hb : b < 0) (hc : condition a b) : a + b < a * b :=
sorry

theorem inequality_2 (hc : condition a b) : |a| < |b| :=
sorry

theorem inequality_4 (hc : condition a b) : (b / a) + (a / b) > 2 :=
sorry

end NUMINAMATH_GPT_inequality_1_inequality_2_inequality_4_l2351_235130


namespace NUMINAMATH_GPT_probability_sum_less_than_product_l2351_235173

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end NUMINAMATH_GPT_probability_sum_less_than_product_l2351_235173


namespace NUMINAMATH_GPT_batsman_sixes_l2351_235103

theorem batsman_sixes (total_runs : ℕ) (boundaries : ℕ) (running_percentage : ℝ) (score_per_boundary : ℕ) (score_per_six : ℕ)
  (h1 : total_runs = 150)
  (h2 : boundaries = 5)
  (h3 : running_percentage = 66.67)
  (h4 : score_per_boundary = 4)
  (h5 : score_per_six = 6) :
  ∃ (sixes : ℕ), sixes = 5 :=
by
  -- Calculations omitted
  existsi 5
  sorry

end NUMINAMATH_GPT_batsman_sixes_l2351_235103


namespace NUMINAMATH_GPT_remaining_paint_fraction_l2351_235102

theorem remaining_paint_fraction :
  ∀ (initial_paint : ℝ) (half_usage : ℕ → ℝ → ℝ),
    initial_paint = 2 →
    half_usage 0 (2 : ℝ) = 1 →
    half_usage 1 (1 : ℝ) = 0.5 →
    half_usage 2 (0.5 : ℝ) = 0.25 →
    half_usage 3 (0.25 : ℝ) = (0.25 / initial_paint) := by
  sorry

end NUMINAMATH_GPT_remaining_paint_fraction_l2351_235102


namespace NUMINAMATH_GPT_brenda_has_eight_l2351_235149

-- Define the amounts each friend has
def emma_money : ℕ := 8
def daya_money : ℕ := emma_money + (emma_money / 4)
def jeff_money : ℕ := (2 * daya_money) / 5
def brenda_money : ℕ := jeff_money + 4

-- Define the theorem to prove Brenda's money is 8
theorem brenda_has_eight : brenda_money = 8 := by
  sorry

end NUMINAMATH_GPT_brenda_has_eight_l2351_235149


namespace NUMINAMATH_GPT_intersection_complement_l2351_235162

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement : A ∩ (U \ B) = {1, 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_complement_l2351_235162


namespace NUMINAMATH_GPT_arithmetic_sequence_8th_term_l2351_235129

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_8th_term_l2351_235129


namespace NUMINAMATH_GPT_math_problem_l2351_235181

theorem math_problem (p q : ℕ) (hp : p % 13 = 7) (hq : q % 13 = 7) (hp_lower : 1000 ≤ p) (hp_upper : p < 10000) (hq_lower : 10000 ≤ q) (min_p : ∀ n, n % 13 = 7 → 1000 ≤ n → n < 10000 → p ≤ n) (min_q : ∀ n, n % 13 = 7 → 10000 ≤ n → q ≤ n) : 
  q - p = 8996 := 
sorry

end NUMINAMATH_GPT_math_problem_l2351_235181


namespace NUMINAMATH_GPT_combustion_moles_l2351_235189

-- Chemical reaction definitions
def balanced_equation : Prop :=
  ∀ (CH4 Cl2 O2 CO2 HCl H2O : ℝ),
  1 * CH4 + 4 * Cl2 + 4 * O2 = 1 * CO2 + 4 * HCl + 2 * H2O

-- Moles of substances
def moles_CH4 := 24
def moles_Cl2 := 48
def moles_O2 := 96
def moles_CO2 := 24
def moles_HCl := 48
def moles_H2O := 48

-- Prove the conditions based on the balanced equation
theorem combustion_moles :
  balanced_equation →
  (moles_O2 = 4 * moles_CH4) ∧
  (moles_H2O = 2 * moles_CH4) :=
by {
  sorry
}

end NUMINAMATH_GPT_combustion_moles_l2351_235189


namespace NUMINAMATH_GPT_least_subtract_for_divisibility_l2351_235184

theorem least_subtract_for_divisibility (n : ℕ) (hn : n = 427398) : 
  (∃ m : ℕ, n - m % 10 = 0 ∧ m = 2) :=
by
  sorry

end NUMINAMATH_GPT_least_subtract_for_divisibility_l2351_235184


namespace NUMINAMATH_GPT_remainder_when_divided_l2351_235136
-- First, import the necessary library.

-- Define the problem conditions and the goal.
theorem remainder_when_divided (P Q Q' R R' S T D D' D'' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + D'' * R' + R')
  (h3 : S = D'' * T)
  (h4 : R' = S + T) :
  P % (D * D' * D'') = D * R' + R := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_l2351_235136


namespace NUMINAMATH_GPT_point_coordinates_l2351_235156

-- We assume that the point P has coordinates (2, 4) and prove that the coordinates with respect to the origin in Cartesian system are indeed (2, 4).
theorem point_coordinates (x y : ℝ) (h : x = 2 ∧ y = 4) : (x, y) = (2, 4) :=
by
  sorry

end NUMINAMATH_GPT_point_coordinates_l2351_235156


namespace NUMINAMATH_GPT_second_smallest_odd_number_l2351_235143

-- Define the conditions
def four_consecutive_odd_numbers_sum (n : ℕ) : Prop := 
  n % 2 = 1 ∧ (n + (n + 2) + (n + 4) + (n + 6) = 112)

-- State the theorem
theorem second_smallest_odd_number (n : ℕ) (h : four_consecutive_odd_numbers_sum n) : n + 2 = 27 :=
sorry

end NUMINAMATH_GPT_second_smallest_odd_number_l2351_235143


namespace NUMINAMATH_GPT_ellipse_equation_midpoint_coordinates_l2351_235172

noncomputable def ellipse_c := {x : ℝ × ℝ | (x.1^2 / 25) + (x.2^2 / 16) = 1}

theorem ellipse_equation (a b : ℝ) (h1 : a = 5) (h2 : b = 4) :
    ∀ x y : ℝ, x = 0 → y = 4 → (y^2 / b^2 = 1) ∧ (e = 3 / 5) → 
      (a > b ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1) := 
sorry

theorem midpoint_coordinates (a b : ℝ) (h1 : a = 5) (h2 : b = 4) :
    ∀ x y x1 x2 y1 y2 : ℝ, 
    (y = 4 / 5 * (x - 3)) → 
    (y1 = 4 / 5 * (x1 - 3)) ∧ (y2 = 4 / 5 * (x2 - 3)) ∧ 
    (x1^2 / a^2) + ((y1 - 3)^2 / b^2) = 1 ∧ (x2^2 / a^2) + ((y2 - 3)^2 / b^2) = 1 ∧ 
    (x1 + x2 = 3) → 
    ((x1 + x2) / 2 = 3 / 2) ∧ ((y1 + y2) / 2 = -6 / 5) := 
sorry

end NUMINAMATH_GPT_ellipse_equation_midpoint_coordinates_l2351_235172


namespace NUMINAMATH_GPT_total_number_of_books_l2351_235195

theorem total_number_of_books (history_books geography_books math_books : ℕ)
  (h1 : history_books = 32) (h2 : geography_books = 25) (h3 : math_books = 43) :
  history_books + geography_books + math_books = 100 :=
by
  -- the proof would go here but we use sorry to skip it
  sorry

end NUMINAMATH_GPT_total_number_of_books_l2351_235195


namespace NUMINAMATH_GPT_pushing_car_effort_l2351_235151

theorem pushing_car_effort (effort constant : ℕ) (people1 people2 : ℕ) 
  (h1 : constant = people1 * effort)
  (h2 : people1 = 4)
  (h3 : effort = 120)
  (h4 : people2 = 6) :
  effort * people1 = constant → constant = people2 * 80 :=
by
  sorry

end NUMINAMATH_GPT_pushing_car_effort_l2351_235151


namespace NUMINAMATH_GPT_gcd_1151_3079_l2351_235182

def a : ℕ := 1151
def b : ℕ := 3079

theorem gcd_1151_3079 : gcd a b = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_1151_3079_l2351_235182


namespace NUMINAMATH_GPT_sally_received_quarters_l2351_235186

theorem sally_received_quarters : 
  ∀ (original_quarters total_quarters received_quarters : ℕ), 
  original_quarters = 760 → 
  total_quarters = 1178 → 
  received_quarters = total_quarters - original_quarters → 
  received_quarters = 418 :=
by 
  intros original_quarters total_quarters received_quarters h_original h_total h_received
  rw [h_original, h_total] at h_received
  exact h_received

end NUMINAMATH_GPT_sally_received_quarters_l2351_235186


namespace NUMINAMATH_GPT_eight_in_M_nine_in_M_ten_not_in_M_l2351_235176

def M (a : ℤ) : Prop := ∃ b c : ℤ, a = b^2 - c^2

theorem eight_in_M : M 8 := by
  sorry

theorem nine_in_M : M 9 := by
  sorry

theorem ten_not_in_M : ¬ M 10 := by
  sorry

end NUMINAMATH_GPT_eight_in_M_nine_in_M_ten_not_in_M_l2351_235176


namespace NUMINAMATH_GPT_selling_price_of_book_l2351_235124

theorem selling_price_of_book (cost_price : ℝ) (profit_percentage : ℝ) (profit : ℝ) (selling_price : ℝ) 
  (h₁ : cost_price = 60) 
  (h₂ : profit_percentage = 25) 
  (h₃ : profit = (profit_percentage / 100) * cost_price) 
  (h₄ : selling_price = cost_price + profit) : 
  selling_price = 75 := 
by
  sorry

end NUMINAMATH_GPT_selling_price_of_book_l2351_235124


namespace NUMINAMATH_GPT_initial_parking_hours_proof_l2351_235174

noncomputable def initial_parking_hours (total_cost : ℝ) (excess_hourly_rate : ℝ) (average_cost : ℝ) (total_hours : ℕ) : ℝ :=
  let h := (total_hours * average_cost - total_cost) / excess_hourly_rate
  h

theorem initial_parking_hours_proof : initial_parking_hours 21.25 1.75 2.361111111111111 9 = 2 :=
by
  sorry

end NUMINAMATH_GPT_initial_parking_hours_proof_l2351_235174


namespace NUMINAMATH_GPT_abs_inequality_l2351_235161

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end NUMINAMATH_GPT_abs_inequality_l2351_235161


namespace NUMINAMATH_GPT_hexagon_sequences_l2351_235171

theorem hexagon_sequences : ∃ n : ℕ, n = 7 ∧ 
  ∀ (x d : ℕ), 6 * x + 15 * d = 720 ∧ (2 * x + 5 * d = 240) ∧ 
  (x + 5 * d < 160) ∧ (0 < x) ∧ (0 < d) ∧ (d % 2 = 0) ↔ (∃ k < n, (∃ x, ∃ d, x = 85 - 2*k ∧ d = 2 + 2*k)) :=
by
  sorry

end NUMINAMATH_GPT_hexagon_sequences_l2351_235171


namespace NUMINAMATH_GPT_prime_transformation_l2351_235146

theorem prime_transformation (p : ℕ) (prime_p : Nat.Prime p) (h : p = 3) : ∃ q : ℕ, q = 13 * p + 2 ∧ Nat.Prime q :=
by
  use 41
  sorry

end NUMINAMATH_GPT_prime_transformation_l2351_235146


namespace NUMINAMATH_GPT_find_m_l2351_235188

variable {α : Type*} [DecidableEq α]

-- Definitions and conditions
def A (m : ℤ) : Set ℤ := {-1, 3, m ^ 2}
def B : Set ℤ := {3, 4}

theorem find_m (m : ℤ) (h : B ⊆ A m) : m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_GPT_find_m_l2351_235188


namespace NUMINAMATH_GPT_quadrilateral_offset_l2351_235165

theorem quadrilateral_offset (d A h₂ x : ℝ)
  (h_da: d = 40)
  (h_A: A = 400)
  (h_h2 : h₂ = 9)
  (h_area : A = 1/2 * d * (x + h₂)) : 
  x = 11 :=
by sorry

end NUMINAMATH_GPT_quadrilateral_offset_l2351_235165


namespace NUMINAMATH_GPT_arithmetic_seq_sum_2017_l2351_235198

theorem arithmetic_seq_sum_2017 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (a1 : a 1 = -2017) 
  (h1 : ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1))
  (h2 : (S 2014) / 2014 - (S 2008) / 2008 = 6) : 
  S 2017 = -2017 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_2017_l2351_235198


namespace NUMINAMATH_GPT_MrFletcherPaymentPerHour_l2351_235120

theorem MrFletcherPaymentPerHour :
  (2 * (10 + 8 + 15)) * x = 660 → x = 10 :=
by
  -- This is where you'd provide the proof, but we skip it as per instructions.
  sorry

end NUMINAMATH_GPT_MrFletcherPaymentPerHour_l2351_235120


namespace NUMINAMATH_GPT_missing_fraction_l2351_235152

-- Defining all the given fractions
def f1 : ℚ := 1 / 3
def f2 : ℚ := 1 / 2
def f3 : ℚ := 1 / 5
def f4 : ℚ := 1 / 4
def f5 : ℚ := -9 / 20
def f6 : ℚ := -5 / 6

-- Defining the total sum in decimal form
def total_sum : ℚ := 5 / 6  -- Since 0.8333333333333334 is equivalent to 5/6

-- Defining the sum of the given fractions
def given_sum : ℚ := f1 + f2 + f3 + f4 + f5 + f6

-- The Lean 4 statement to prove the missing fraction
theorem missing_fraction : ∃ x : ℚ, (given_sum + x = total_sum) ∧ x = 5 / 6 :=
by
  use 5 / 6
  constructor
  . sorry
  . rfl

end NUMINAMATH_GPT_missing_fraction_l2351_235152


namespace NUMINAMATH_GPT_additional_carpet_needed_l2351_235110

-- Definitions according to the given conditions
def length_feet := 18
def width_feet := 12
def covered_area := 4 -- in square yards
def feet_per_yard := 3

-- Prove that the additional square yards needed to cover the remaining part of the floor is 20
theorem additional_carpet_needed : 
  ((length_feet / feet_per_yard) * (width_feet / feet_per_yard) - covered_area) = 20 := 
by
  sorry

end NUMINAMATH_GPT_additional_carpet_needed_l2351_235110


namespace NUMINAMATH_GPT_ariel_fish_l2351_235140

theorem ariel_fish (total_fish : ℕ) (male_fraction female_fraction : ℚ) (h1 : total_fish = 45) (h2 : male_fraction = 2 / 3) (h3 : female_fraction = 1 - male_fraction) : total_fish * female_fraction = 15 :=
by
  sorry

end NUMINAMATH_GPT_ariel_fish_l2351_235140


namespace NUMINAMATH_GPT_quadratic_term_elimination_l2351_235106

theorem quadratic_term_elimination (m : ℝ) :
  (3 * (x : ℝ) ^ 2 - 10 - 2 * x - 4 * x ^ 2 + m * x ^ 2) = -(x : ℝ) * (2 * x + 10) ↔ m = 1 := 
by sorry

end NUMINAMATH_GPT_quadratic_term_elimination_l2351_235106


namespace NUMINAMATH_GPT_exists_nat_with_digit_sum_l2351_235150

-- Definitions of the necessary functions
def digit_sum (n : ℕ) : ℕ := sorry -- Assume this is the sum of the digits of n

theorem exists_nat_with_digit_sum :
  ∃ n : ℕ, digit_sum n = 1000 ∧ digit_sum (n^2) = 1000000 :=
by
  sorry

end NUMINAMATH_GPT_exists_nat_with_digit_sum_l2351_235150


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l2351_235177

theorem eccentricity_of_hyperbola {a b c e : ℝ} (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : b = 2 * a)
  (h₄ : c^2 = a^2 + b^2) :
  e = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l2351_235177


namespace NUMINAMATH_GPT_river_depth_mid_may_l2351_235117

-- Definitions corresponding to the conditions
def depth_mid_june (D : ℕ) : ℕ := D + 10
def depth_mid_july (D : ℕ) : ℕ := 3 * (depth_mid_june D)

-- The theorem statement
theorem river_depth_mid_may (D : ℕ) (h : depth_mid_july D = 45) : D = 5 :=
by
  sorry

end NUMINAMATH_GPT_river_depth_mid_may_l2351_235117


namespace NUMINAMATH_GPT_cs_share_l2351_235128

-- Definitions for the conditions
def daily_work (days: ℕ) : ℚ := 1 / days

def total_work_contribution (a_days: ℕ) (b_days: ℕ) (c_days: ℕ): ℚ := 
  daily_work a_days + daily_work b_days + daily_work c_days

def total_payment (payment: ℕ) (work_contribution: ℚ) : ℚ := 
  payment * work_contribution

-- The mathematically equivalent proof problem
theorem cs_share (a_days: ℕ) (b_days: ℕ) (total_days : ℕ) (payment: ℕ) : 
  a_days = 6 → b_days = 8 → total_days = 3 → payment = 1200 →
  total_payment payment (daily_work total_days - (daily_work a_days + daily_work b_days)) = 50 :=
sorry

end NUMINAMATH_GPT_cs_share_l2351_235128


namespace NUMINAMATH_GPT_time_to_fill_pot_l2351_235160

def pot_volume : ℕ := 3000  -- in ml
def rate_of_entry : ℕ := 60 -- in ml/minute

-- Statement: Prove that the time required for the pot to be full is 50 minutes.
theorem time_to_fill_pot : (pot_volume / rate_of_entry) = 50 := by
  sorry

end NUMINAMATH_GPT_time_to_fill_pot_l2351_235160


namespace NUMINAMATH_GPT_Kuwabara_class_girls_percentage_l2351_235100

variable (num_girls num_boys : ℕ)

def total_students (num_girls num_boys : ℕ) : ℕ :=
  num_girls + num_boys

def girls_percentage (num_girls num_boys : ℕ) : ℚ :=
  (num_girls : ℚ) / (total_students num_girls num_boys : ℚ) * 100

theorem Kuwabara_class_girls_percentage (num_girls num_boys : ℕ) (h1: num_girls = 10) (h2: num_boys = 15) :
  girls_percentage num_girls num_boys = 40 := 
by
  sorry

end NUMINAMATH_GPT_Kuwabara_class_girls_percentage_l2351_235100


namespace NUMINAMATH_GPT_set_intersection_complement_l2351_235187

def setA : Set ℝ := {-2, -1, 0, 1, 2}
def setB : Set ℝ := { x : ℝ | x^2 + 2*x < 0 }
def complementB : Set ℝ := { x : ℝ | x ≥ 0 ∨ x ≤ -2 }

theorem set_intersection_complement :
  setA ∩ complementB = {-2, 0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l2351_235187


namespace NUMINAMATH_GPT_pattern_continues_for_max_8_years_l2351_235134

def is_adult_age (age : ℕ) := 18 ≤ age ∧ age < 40

def fits_pattern (p1 p2 n : ℕ) : Prop := 
  is_adult_age p1 ∧
  is_adult_age p2 ∧ 
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → 
    (k % (p1 + k) = 0 ∨ k % (p2 + k) = 0) ∧ ¬ (k % (p1 + k) = 0 ∧ k % (p2 + k) = 0))

theorem pattern_continues_for_max_8_years (p1 p2 : ℕ) : 
  fits_pattern p1 p2 8 := 
sorry

end NUMINAMATH_GPT_pattern_continues_for_max_8_years_l2351_235134


namespace NUMINAMATH_GPT_burger_cost_l2351_235108

theorem burger_cost {B : ℝ} (sandwich_cost : ℝ) (smoothies_cost : ℝ) (total_cost : ℝ)
  (H1 : sandwich_cost = 4)
  (H2 : smoothies_cost = 8)
  (H3 : total_cost = 17)
  (H4 : B + sandwich_cost + smoothies_cost = total_cost) :
  B = 5 :=
sorry

end NUMINAMATH_GPT_burger_cost_l2351_235108


namespace NUMINAMATH_GPT_maria_nickels_l2351_235196

theorem maria_nickels (dimes quarters_initial quarters_additional : ℕ) (total_amount : ℚ) 
  (Hd : dimes = 4) (Hqi : quarters_initial = 4) (Hqa : quarters_additional = 5) (Htotal : total_amount = 3) : 
  (dimes * 0.10 + quarters_initial * 0.25 + quarters_additional * 0.25 + n/20) = total_amount → n = 7 :=
  sorry

end NUMINAMATH_GPT_maria_nickels_l2351_235196


namespace NUMINAMATH_GPT_chord_count_l2351_235193

theorem chord_count {n : ℕ} (h : n = 2024) : 
  ∃ k : ℕ, k ≥ 1024732 ∧ ∀ (i j : ℕ), (i < n → j < n → i ≠ j → true) := sorry

end NUMINAMATH_GPT_chord_count_l2351_235193


namespace NUMINAMATH_GPT_triangle_area_is_correct_l2351_235141

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_is_correct :
  area_of_triangle (0, 3) (4, -2) (9, 6) = 16.5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_correct_l2351_235141


namespace NUMINAMATH_GPT_complex_purely_imaginary_l2351_235109

theorem complex_purely_imaginary (x : ℝ) :
  (x^2 - 1 = 0) → (x - 1 ≠ 0) → x = -1 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_complex_purely_imaginary_l2351_235109


namespace NUMINAMATH_GPT_hulk_jump_distance_l2351_235126

theorem hulk_jump_distance :
  ∃ n : ℕ, 3^n > 1500 ∧ ∀ m < n, 3^m ≤ 1500 := 
sorry

end NUMINAMATH_GPT_hulk_jump_distance_l2351_235126


namespace NUMINAMATH_GPT_two_leq_one_add_one_div_n_pow_n_lt_three_l2351_235183

theorem two_leq_one_add_one_div_n_pow_n_lt_three :
  ∀ (n : ℕ), 2 ≤ (1 + (1 : ℝ) / n) ^ n ∧ (1 + (1 : ℝ) / n) ^ n < 3 := 
by 
  sorry

end NUMINAMATH_GPT_two_leq_one_add_one_div_n_pow_n_lt_three_l2351_235183


namespace NUMINAMATH_GPT_total_students_l2351_235153

-- Lean statement: Prove the number of students given the conditions.
theorem total_students (num_classrooms : ℕ) (num_buses : ℕ) (seats_per_bus : ℕ) 
  (students : ℕ) (h1 : num_classrooms = 87) (h2 : num_buses = 29) 
  (h3 : seats_per_bus = 2) (h4 : students = num_classrooms * num_buses * seats_per_bus) :
  students = 5046 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l2351_235153


namespace NUMINAMATH_GPT_quadratic_root_l2351_235133

theorem quadratic_root (a : ℝ) : (∃ x : ℝ, x = 1 ∧ a * x^2 + x - 2 = 0) → a = 1 := by
  sorry

end NUMINAMATH_GPT_quadratic_root_l2351_235133


namespace NUMINAMATH_GPT_tank_fill_time_l2351_235175

theorem tank_fill_time :
  let fill_rate_A := 1 / 8
  let empty_rate_B := 1 / 24
  let combined_rate := fill_rate_A - empty_rate_B
  let time_with_both_pipes := 66
  let partial_fill := time_with_both_pipes * combined_rate
  let remaining_fill := 1 - (partial_fill % 1)
  let additional_time_A := remaining_fill / fill_rate_A
  time_with_both_pipes + additional_time_A = 70 :=
by
  let fill_rate_A := 1 / 8
  let empty_rate_B := 1 / 24
  let combined_rate := fill_rate_A - empty_rate_B
  let time_with_both_pipes := 66
  let partial_fill := time_with_both_pipes * combined_rate
  let remaining_fill := 1 - (partial_fill % 1)
  let additional_time_A := remaining_fill / fill_rate_A
  have h : time_with_both_pipes + additional_time_A = 70 := sorry
  exact h

end NUMINAMATH_GPT_tank_fill_time_l2351_235175


namespace NUMINAMATH_GPT_number_of_leap_years_l2351_235168

noncomputable def is_leap_year (year : ℕ) : Prop :=
  (year % 1300 = 300 ∨ year % 1300 = 700) ∧ 2000 ≤ year ∧ year ≤ 5000

noncomputable def leap_years : List ℕ :=
  [2900, 4200, 3300, 4600]

theorem number_of_leap_years : leap_years.length = 4 ∧ ∀ y ∈ leap_years, is_leap_year y := by
  sorry

end NUMINAMATH_GPT_number_of_leap_years_l2351_235168


namespace NUMINAMATH_GPT_kangaroo_mob_has_6_l2351_235115

-- Define the problem conditions
def mob_of_kangaroos (W : ℝ) (k : ℕ) : Prop :=
  ∃ (two_lightest three_heaviest remaining : ℝ) (n_two n_three n_rem : ℕ),
    two_lightest = 0.25 * W ∧
    three_heaviest = 0.60 * W ∧
    remaining = 0.15 * W ∧
    n_two = 2 ∧
    n_three = 3 ∧
    n_rem = 1 ∧
    k = n_two + n_three + n_rem

-- The theorem to be proven
theorem kangaroo_mob_has_6 (W : ℝ) : ∃ k, mob_of_kangaroos W k ∧ k = 6 :=
by
  sorry

end NUMINAMATH_GPT_kangaroo_mob_has_6_l2351_235115


namespace NUMINAMATH_GPT_optimal_washing_effect_l2351_235166

noncomputable def total_capacity : ℝ := 20 -- kilograms
noncomputable def weight_clothes : ℝ := 5 -- kilograms
noncomputable def weight_detergent_existing : ℝ := 2 * 0.02 -- kilograms
noncomputable def optimal_concentration : ℝ := 0.004 -- kilograms per kilogram of water

theorem optimal_washing_effect :
  ∃ (additional_detergent additional_water : ℝ),
    additional_detergent = 0.02 ∧ additional_water = 14.94 ∧
    weight_clothes + additional_water + weight_detergent_existing + additional_detergent = total_capacity ∧
    weight_detergent_existing + additional_detergent = optimal_concentration * additional_water :=
by
  sorry

end NUMINAMATH_GPT_optimal_washing_effect_l2351_235166


namespace NUMINAMATH_GPT_a7_equals_21_l2351_235122

-- Define the sequence {a_n} recursively
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | (n + 2) => seq n + seq (n + 1)

-- Statement to prove that a_7 = 21
theorem a7_equals_21 : seq 6 = 21 := 
  sorry

end NUMINAMATH_GPT_a7_equals_21_l2351_235122
