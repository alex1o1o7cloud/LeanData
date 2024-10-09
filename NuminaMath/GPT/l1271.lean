import Mathlib

namespace no_nat_fun_satisfying_property_l1271_127112

theorem no_nat_fun_satisfying_property :
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 :=
by
  sorry

end no_nat_fun_satisfying_property_l1271_127112


namespace prob_mc_tf_correct_prob_at_least_one_mc_correct_l1271_127169

-- Define the total number of questions and their types
def total_questions : ℕ := 5
def multiple_choice_questions : ℕ := 3
def true_false_questions : ℕ := 2
def total_outcomes : ℕ := total_questions * (total_questions - 1)

-- Probability calculation for one drawing a multiple-choice and the other drawing a true/false question
def prob_mc_tf : ℚ := (multiple_choice_questions * true_false_questions + true_false_questions * multiple_choice_questions) / total_outcomes

-- Probability calculation for at least one drawing a multiple-choice question
def prob_at_least_one_mc : ℚ := 1 - (true_false_questions * (true_false_questions - 1)) / total_outcomes

theorem prob_mc_tf_correct : prob_mc_tf = 3/5 := by
  sorry

theorem prob_at_least_one_mc_correct : prob_at_least_one_mc = 9/10 := by
  sorry

end prob_mc_tf_correct_prob_at_least_one_mc_correct_l1271_127169


namespace max_ab_under_constraint_l1271_127114

theorem max_ab_under_constraint (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 3 * a + 2 * b = 1) : 
  ab ≤ (1 / 24) ∧ (ab = 1 / 24 ↔ a = 1 / 6 ∧ b = 1 / 4) :=
sorry

end max_ab_under_constraint_l1271_127114


namespace factorize_polynomial_l1271_127137

theorem factorize_polynomial (x : ℝ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) := 
sorry

end factorize_polynomial_l1271_127137


namespace inverse_f_486_l1271_127192

-- Define the function f with given properties.
def f : ℝ → ℝ := sorry

-- Condition 1: f(5) = 2
axiom f_at_5 : f 5 = 2

-- Condition 2: f(3x) = 3f(x) for all x
axiom f_scale : ∀ x, f (3 * x) = 3 * f x

-- Proposition: f⁻¹(486) = 1215
theorem inverse_f_486 : (∃ x, f x = 486) → ∀ x, f x = 486 → x = 1215 :=
by sorry

end inverse_f_486_l1271_127192


namespace decrease_percent_in_revenue_l1271_127132

theorem decrease_percent_in_revenue
  (T C : ℝ)
  (h_pos_T : 0 < T)
  (h_pos_C : 0 < C)
  (h_new_tax : T_new = 0.80 * T)
  (h_new_consumption : C_new = 1.20 * C) :
  let original_revenue := T * C
  let new_revenue := 0.80 * T * 1.20 * C
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 4 := by
sorry

end decrease_percent_in_revenue_l1271_127132


namespace repeating_decimal_exceeds_finite_decimal_by_l1271_127188

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l1271_127188


namespace find_a_l1271_127193

theorem find_a (a : ℝ) (h₁ : ¬ (a = 0)) (h_perp : (∀ x y : ℝ, (a * x + 1 = 0) 
  -> (a - 2) * x + y + a = 0 -> ∀ x₁ y₁, (a * x₁ + 1 = 0) -> y = y₁)) : a = 2 := 
by 
  sorry

end find_a_l1271_127193


namespace collinear_points_b_value_l1271_127119

theorem collinear_points_b_value (b : ℝ)
    (h : let slope1 := (4 - (-6)) / ((2 * b + 1) - 4)
         let slope2 := (1 - (-6)) / ((-3 * b + 2) - 4)
         slope1 = slope2) :
    b = -1 / 44 :=
by
  have slope1 := (4 - (-6)) / ((2 * b + 1) - 4)
  have slope2 := (1 - (-6)) / ((-3 * b + 2) - 4)
  have := h
  sorry

end collinear_points_b_value_l1271_127119


namespace find_n_l1271_127198

noncomputable def objects_per_hour (n : ℕ) : ℕ := n

theorem find_n (n : ℕ) (h₁ : 1 + (2 / 3) + (1 / 3) + (1 / 3) = 7 / 3) 
  (h₂ : objects_per_hour n * 7 / 3 = 28) : n = 12 :=
by
  have total_hours := h₁ 
  have total_objects := h₂
  sorry

end find_n_l1271_127198


namespace max_n_possible_l1271_127189

theorem max_n_possible (k : ℕ) (h_k : k > 1) : ∃ n : ℕ, n = k - 1 :=
by
  sorry

end max_n_possible_l1271_127189


namespace mean_score_is_82_l1271_127141

noncomputable def mean_score 
  (M A m a : ℝ) 
  (hM : M = 90) 
  (hA : A = 75) 
  (hm : m / a = 4 / 5) : ℝ := 
  (M * m + A * a) / (m + a)

theorem mean_score_is_82 
  (M A m a : ℝ) 
  (hM : M = 90) 
  (hA : A = 75) 
  (hm : m / a = 4 / 5) : 
  mean_score M A m a hM hA hm = 82 := 
    sorry

end mean_score_is_82_l1271_127141


namespace range_of_b_l1271_127191

theorem range_of_b (b : ℝ) (h : Real.sqrt ((b-2)^2) = 2 - b) : b ≤ 2 :=
by {
  sorry
}

end range_of_b_l1271_127191


namespace min_value_fraction_l1271_127196

theorem min_value_fraction (x y : ℝ) (hx : x > -1) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  ∃ m, (∀ z, z = (1 / (x + 1) + 1 / y) → z ≥ m) ∧ m = (3 + 2 * Real.sqrt 2) / 2 :=
by
  sorry

end min_value_fraction_l1271_127196


namespace dealer_can_determine_values_l1271_127136

def card_value_determined (a : Fin 100 → Fin 100) : Prop :=
  (∀ i j : Fin 100, i > j → a i > a j) ∧ (a 0 > a 99) ∧
  (∀ k : Fin 100, a k = k + 1)

theorem dealer_can_determine_values :
  ∃ (messages : Fin 100 → Fin 100), card_value_determined messages :=
sorry

end dealer_can_determine_values_l1271_127136


namespace ott_fraction_is_3_over_13_l1271_127186

-- Defining the types and quantities involved
noncomputable def moes_original_money (amount_given: ℚ) := amount_given * 5
noncomputable def lokis_original_money (amount_given: ℚ) := amount_given * 5
noncomputable def nicks_original_money (amount_given: ℚ) := amount_given * 3

-- Total original money of the group (excluding Ott)
noncomputable def total_original_money (amount_given: ℚ) :=
  moes_original_money amount_given + lokis_original_money amount_given + nicks_original_money amount_given

-- Total money received by Ott
noncomputable def otts_received_money (amount_given: ℚ) := 3 * amount_given

-- Fraction of the group's total money Ott now has
noncomputable def otts_fraction_of_total_money (amount_given: ℚ) : ℚ :=
  otts_received_money amount_given / total_original_money amount_given

-- The theorem to be proved
theorem ott_fraction_is_3_over_13 :
  otts_fraction_of_total_money 1 = 3 / 13 :=
by
  -- The body of the proof is skipped with sorry
  sorry

end ott_fraction_is_3_over_13_l1271_127186


namespace triangle_dimensions_l1271_127130

theorem triangle_dimensions (a b c : ℕ) (h1 : a > b) (h2 : b > c)
  (h3 : a = 2 * c) (h4 : b - 2 = c) (h5 : 2 * a / 3 = b) :
  a = 12 ∧ b = 8 ∧ c = 6 :=
by
  sorry

end triangle_dimensions_l1271_127130


namespace tetrahedron_inscribed_in_pyramid_edge_length_l1271_127126

noncomputable def edge_length_of_tetrahedron := (Real.sqrt 2) / 2

theorem tetrahedron_inscribed_in_pyramid_edge_length :
  let A := (0,0,0)
  let B := (1,0,0)
  let C := (1,1,0)
  let D := (0,1,0)
  let E := (0.5, 0.5, 1)
  let v₁ := (0.5, 0, 0)
  let v₂ := (1, 0.5, 0)
  let v₃ := (0, 0.5, 0)
  dist (v₁ : ℝ × ℝ × ℝ) v₂ = edge_length_of_tetrahedron ∧
  dist v₂ v₃ = edge_length_of_tetrahedron ∧
  dist v₃ v₁ = edge_length_of_tetrahedron ∧
  dist E v₁ = dist E v₂ ∧
  dist E v₂ = dist E v₃ :=
by
  sorry

end tetrahedron_inscribed_in_pyramid_edge_length_l1271_127126


namespace sqrt_x_plus_inv_sqrt_x_eq_sqrt_152_l1271_127153

-- Conditions
variable (x : ℝ) (h₀ : 0 < x) (h₁ : x + 1 / x = 150)

-- Statement to prove
theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_152 : (Real.sqrt x + Real.sqrt (1 / x) = Real.sqrt 152) := 
sorry -- Proof not needed, skip with sorry

end sqrt_x_plus_inv_sqrt_x_eq_sqrt_152_l1271_127153


namespace bill_pays_sales_tax_correct_l1271_127184

def take_home_salary : ℝ := 40000
def property_tax : ℝ := 2000
def gross_salary : ℝ := 50000
def income_tax (gs : ℝ) : ℝ := 0.10 * gs
def total_taxes_paid (gs th : ℝ) : ℝ := gs - th
def sales_tax (ttp it pt : ℝ) : ℝ := ttp - it - pt

theorem bill_pays_sales_tax_correct :
  sales_tax
    (total_taxes_paid gross_salary take_home_salary)
    (income_tax gross_salary)
    property_tax = 3000 :=
by sorry

end bill_pays_sales_tax_correct_l1271_127184


namespace necessary_but_not_sufficient_condition_l1271_127161

theorem necessary_but_not_sufficient_condition (x : ℝ) : x^2 - 4 = 0 → x + 2 = 0 :=
by
  sorry

end necessary_but_not_sufficient_condition_l1271_127161


namespace closest_perfect_square_to_314_l1271_127197

theorem closest_perfect_square_to_314 :
  ∃ n : ℤ, n^2 = 324 ∧ ∀ m : ℤ, m^2 ≠ 324 → |m^2 - 314| > |324 - 314| :=
by
  sorry

end closest_perfect_square_to_314_l1271_127197


namespace sum_of_squares_mul_l1271_127164

theorem sum_of_squares_mul (a b c d : ℝ) :
(a^2 + b^2) * (c^2 + d^2) = (a * c + b * d)^2 + (a * d - b * c)^2 :=
by
  sorry

end sum_of_squares_mul_l1271_127164


namespace find_constant_a_l1271_127123

theorem find_constant_a (x y a : ℝ) (h1 : (ax + 4 * y) / (x - 2 * y) = 13) (h2 : x / (2 * y) = 5 / 2) : a = 7 :=
sorry

end find_constant_a_l1271_127123


namespace peter_initial_erasers_l1271_127118

theorem peter_initial_erasers (E : ℕ) (h : E + 3 = 11) : E = 8 :=
by {
  sorry
}

end peter_initial_erasers_l1271_127118


namespace length_of_shorter_angle_trisector_l1271_127140

theorem length_of_shorter_angle_trisector (BC AC : ℝ) (h1 : BC = 3) (h2 : AC = 4) :
  let AB := Real.sqrt (BC^2 + AC^2)
  let x := 2 * (12 / (4 * Real.sqrt 3 + 3))
  let PC := 2 * x
  AB = 5 ∧ PC = (32 * Real.sqrt 3 - 24) / 13 :=
by
  sorry

end length_of_shorter_angle_trisector_l1271_127140


namespace anne_wandering_time_l1271_127124

theorem anne_wandering_time (distance speed : ℝ) (h_dist : distance = 3.0) (h_speed : speed = 2.0) : 
  distance / speed = 1.5 :=
by
  rw [h_dist, h_speed]
  norm_num

end anne_wandering_time_l1271_127124


namespace total_area_of_field_l1271_127194

theorem total_area_of_field 
  (A_s : ℕ) 
  (h₁ : A_s = 315)
  (A_l : ℕ) 
  (h₂ : A_l - A_s = (1/5) * ((A_s + A_l) / 2)) : 
  A_s + A_l = 700 := 
  by 
    sorry

end total_area_of_field_l1271_127194


namespace cost_of_7_cubic_yards_of_topsoil_is_1512_l1271_127108

-- Definition of the given conditions
def cost_per_cubic_foot : ℕ := 8
def cubic_yards : ℕ := 7
def cubic_yards_to_cubic_feet : ℕ := 27

-- Problem definition
def cost_of_topsoil (cubic_yards : ℕ) (cost_per_cubic_foot : ℕ) (cubic_yards_to_cubic_feet : ℕ) : ℕ :=
  cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot

-- The proof statement
theorem cost_of_7_cubic_yards_of_topsoil_is_1512 :
  cost_of_topsoil cubic_yards cost_per_cubic_foot cubic_yards_to_cubic_feet = 1512 := by
  sorry

end cost_of_7_cubic_yards_of_topsoil_is_1512_l1271_127108


namespace perfect_square_iff_n_eq_5_l1271_127187

theorem perfect_square_iff_n_eq_5 (n : ℕ) (h_pos : 0 < n) :
  ∃ m : ℕ, n * 2^(n-1) + 1 = m^2 ↔ n = 5 := by
  sorry

end perfect_square_iff_n_eq_5_l1271_127187


namespace eq_has_infinite_solutions_l1271_127156

theorem eq_has_infinite_solutions (b : ℤ) :
  (∀ x : ℤ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by 
  sorry

end eq_has_infinite_solutions_l1271_127156


namespace minimum_n_for_i_pow_n_eq_neg_i_l1271_127199

open Complex

theorem minimum_n_for_i_pow_n_eq_neg_i : ∃ (n : ℕ), 0 < n ∧ (i^n = -i) ∧ ∀ (m : ℕ), 0 < m ∧ (i^m = -i) → n ≤ m :=
by
  sorry

end minimum_n_for_i_pow_n_eq_neg_i_l1271_127199


namespace relay_race_total_time_l1271_127145

theorem relay_race_total_time :
  let t1 := 55
  let t2 := t1 + 10
  let t3 := t2 - 15
  let t4 := t1 - 25
  t1 + t2 + t3 + t4 = 200 := by
    sorry

end relay_race_total_time_l1271_127145


namespace positive_iff_sum_and_product_positive_l1271_127135

theorem positive_iff_sum_and_product_positive (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) :=
by
  sorry

end positive_iff_sum_and_product_positive_l1271_127135


namespace min_trips_is_157_l1271_127128

theorem min_trips_is_157 :
  ∃ x y : ℕ, 31 * x + 32 * y = 5000 ∧ x + y = 157 :=
sorry

end min_trips_is_157_l1271_127128


namespace prob_triangle_inequality_l1271_127181

theorem prob_triangle_inequality (x y z : ℕ) (h1 : 1 ≤ x ∧ x ≤ 6) (h2 : 1 ≤ y ∧ y ≤ 6) (h3 : 1 ≤ z ∧ z ≤ 6) : 
  (∃ (p : ℚ), p = 37 / 72) := 
sorry

end prob_triangle_inequality_l1271_127181


namespace cos_pi_minus_alpha_l1271_127167

theorem cos_pi_minus_alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : Real.cos (π - α) = - (1 / 3) :=
by
  sorry

end cos_pi_minus_alpha_l1271_127167


namespace cages_used_l1271_127105

-- Define the initial conditions
def total_puppies : ℕ := 18
def puppies_sold : ℕ := 3
def puppies_per_cage : ℕ := 5

-- State the theorem to prove the number of cages used
theorem cages_used : (total_puppies - puppies_sold) / puppies_per_cage = 3 := by
  sorry

end cages_used_l1271_127105


namespace joe_paint_usage_l1271_127148

theorem joe_paint_usage :
  ∀ (total_paint initial_remaining_paint final_remaining_paint paint_first_week paint_second_week total_used : ℕ),
  total_paint = 360 →
  initial_remaining_paint = total_paint - paint_first_week →
  final_remaining_paint = initial_remaining_paint - paint_second_week →
  paint_first_week = (2 * total_paint) / 3 →
  paint_second_week = (1 * initial_remaining_paint) / 5 →
  total_used = paint_first_week + paint_second_week →
  total_used = 264 :=
by
  sorry

end joe_paint_usage_l1271_127148


namespace arithmetic_sequence_sum_l1271_127180

theorem arithmetic_sequence_sum (c d : ℕ) 
  (h1 : ∀ (a1 a2 a3 a4 a5 a6 : ℕ), a1 = 3 → a2 = 10 → a3 = 17 → a6 = 38 → (a2 - a1 = a3 - a2) → (a3 - a2 = c - a3) → (c - a3 = d - c) → (d - c = a6 - d)) : 
  c + d = 55 := 
by 
  sorry

end arithmetic_sequence_sum_l1271_127180


namespace sequence_general_term_l1271_127120

theorem sequence_general_term (n : ℕ) : 
  (2 * n - 1) / (2 ^ n) = a_n := 
sorry

end sequence_general_term_l1271_127120


namespace necessary_condition_l1271_127160

variable (P Q : Prop)

/-- If the presence of the dragon city's flying general implies that
    the horses of the Hu people will not cross the Yin Mountains,
    then "not letting the horses of the Hu people cross the Yin Mountains"
    is a necessary condition for the presence of the dragon city's flying general. -/
theorem necessary_condition (h : P → Q) : ¬Q → ¬P :=
by sorry

end necessary_condition_l1271_127160


namespace range_of_x_l1271_127102

theorem range_of_x (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) : x^2 + a * x > 4 * x + a - 3 ↔ (x > 3 ∨ x < -1) := by
  sorry

end range_of_x_l1271_127102


namespace average_cookies_per_package_is_fifteen_l1271_127170

def average_cookies_count (cookies : List ℕ) (n : ℕ) : ℕ :=
  (cookies.sum / n : ℕ)

theorem average_cookies_per_package_is_fifteen :
  average_cookies_count [5, 12, 18, 20, 21] 5 = 15 :=
by
  sorry

end average_cookies_per_package_is_fifteen_l1271_127170


namespace value_of_k_l1271_127149

theorem value_of_k (a b k : ℝ) (h1 : 2 * a = k) (h2 : 3 * b = k) (h3 : 2 * a + b = a * b) (h4 : k ≠ 1) : k = 8 := 
sorry

end value_of_k_l1271_127149


namespace compound_interest_semiannual_l1271_127125

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_semiannual :
  compound_interest 150 0.20 2 1 = 181.50 :=
by
  sorry

end compound_interest_semiannual_l1271_127125


namespace min_value_expression_l1271_127101

/-- 
Given real numbers a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p such that 
abcd = 16, efgh = 16, ijkl = 16, and mnop = 16, prove that the minimum value of 
(aeim)^2 + (bfjn)^2 + (cgko)^2 + (dhlp)^2 is 1024. 
-/
theorem min_value_expression (a b c d e f g h i j k l m n o p : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 16) 
  (h3 : i * j * k * l = 16) 
  (h4 : m * n * o * p = 16) : 
  (a * e * i * m) ^ 2 + (b * f * j * n) ^ 2 + (c * g * k * o) ^ 2 + (d * h * l * p) ^ 2 ≥ 1024 :=
by 
  sorry


end min_value_expression_l1271_127101


namespace percentage_chain_l1271_127179

theorem percentage_chain (n : ℝ) (h : n = 6000) : 0.1 * (0.3 * (0.5 * n)) = 90 := by
  sorry

end percentage_chain_l1271_127179


namespace thomas_blocks_total_l1271_127104

theorem thomas_blocks_total 
  (first_stack : ℕ)
  (second_stack : ℕ)
  (third_stack : ℕ)
  (fourth_stack : ℕ)
  (fifth_stack : ℕ) 
  (h1 : first_stack = 7)
  (h2 : second_stack = first_stack + 3)
  (h3 : third_stack = second_stack - 6)
  (h4 : fourth_stack = third_stack + 10)
  (h5 : fifth_stack = 2 * second_stack) :
  first_stack + second_stack + third_stack + fourth_stack + fifth_stack = 55 :=
by
  sorry

end thomas_blocks_total_l1271_127104


namespace negation_of_all_have_trap_consumption_l1271_127159

-- Definitions for the conditions
def domestic_mobile_phone : Type := sorry

def has_trap_consumption (phone : domestic_mobile_phone) : Prop := sorry

def all_have_trap_consumption : Prop := ∀ phone : domestic_mobile_phone, has_trap_consumption phone

-- Statement of the problem
theorem negation_of_all_have_trap_consumption :
  ¬ all_have_trap_consumption ↔ ∃ phone : domestic_mobile_phone, ¬ has_trap_consumption phone :=
sorry

end negation_of_all_have_trap_consumption_l1271_127159


namespace total_books_to_read_l1271_127109

theorem total_books_to_read (books_per_week : ℕ) (weeks : ℕ) (total_books : ℕ) 
  (h1 : books_per_week = 6) 
  (h2 : weeks = 5) 
  (h3 : total_books = books_per_week * weeks) : 
  total_books = 30 :=
by
  rw [h1, h2] at h3
  exact h3

end total_books_to_read_l1271_127109


namespace find_m_l1271_127176

theorem find_m (x₁ x₂ y₁ y₂ : ℝ) (m : ℝ) 
  (h_parabola_A : y₁ = 2 * x₁^2) 
  (h_parabola_B : y₂ = 2 * x₂^2) 
  (h_symmetry : y₂ - y₁ = 2 * (x₂^2 - x₁^2)) 
  (h_product : x₁ * x₂ = -1/2) 
  (h_midpoint : (y₂ + y₁) / 2 = (x₂ + x₁) / 2 + m) :
  m = 3 / 2 :=
by
  sorry

end find_m_l1271_127176


namespace sharpening_cost_l1271_127168

theorem sharpening_cost
  (trees_chopped : ℕ)
  (trees_per_sharpening : ℕ)
  (total_cost : ℕ)
  (min_trees_chopped : trees_chopped ≥ 91)
  (trees_per_sharpening_eq : trees_per_sharpening = 13)
  (total_cost_eq : total_cost = 35) :
  total_cost / (trees_chopped / trees_per_sharpening) = 5 := by
  sorry

end sharpening_cost_l1271_127168


namespace average_of_remaining_numbers_l1271_127158

theorem average_of_remaining_numbers 
  (numbers : List ℝ)
  (h_len : numbers.length = 15)
  (h_avg : (numbers.sum / 15) = 100)
  (h_remove : [80, 90, 95] ⊆ numbers) :
  ((numbers.sum - 80 - 90 - 95) / 12) = (1235 / 12) :=
sorry

end average_of_remaining_numbers_l1271_127158


namespace average_velocity_mass_flow_rate_available_horsepower_l1271_127152

/-- Average velocity of water flowing out of the sluice gate. -/
theorem average_velocity (g h₁ h₂ : ℝ) (h1_5m : h₁ = 5) (h2_5_4m : h₂ = 5.4) (g_9_81 : g = 9.81) :
    (1 / 2) * (Real.sqrt (2 * g * h₁) + Real.sqrt (2 * g * h₂)) = 10.1 :=
by
  sorry

/-- Mass flow rate of water per second when given average velocity and opening dimensions. -/
theorem mass_flow_rate (v A : ℝ) (v_10_1 : v = 10.1) (A_0_6 : A = 0.4 * 1.5) (rho : ℝ) (rho_1000 : rho = 1000) :
    ρ * A * v = 6060 :=
by
  sorry

/-- Available horsepower through turbines given mass flow rate and average velocity. -/
theorem available_horsepower (m v : ℝ) (m_6060 : m = 6060) (v_10_1 : v = 10.1 ) (hp : ℝ)
    (hp_735_5 : hp = 735.5 ) :
    (1 / 2) * m * v^2 / hp = 420 :=
by
  sorry

end average_velocity_mass_flow_rate_available_horsepower_l1271_127152


namespace joe_first_lift_weight_l1271_127174

theorem joe_first_lift_weight (x y : ℕ) 
  (h1 : x + y = 900)
  (h2 : 2 * x = y + 300) :
  x = 400 :=
by
  sorry

end joe_first_lift_weight_l1271_127174


namespace parallelogram_side_length_sum_l1271_127121

theorem parallelogram_side_length_sum (x y z : ℚ) 
  (h1 : 3 * x - 1 = 12)
  (h2 : 4 * z + 2 = 7 * y + 3) :
  x + y + z = 121 / 21 :=
by
  sorry

end parallelogram_side_length_sum_l1271_127121


namespace hyperbola_foci_coords_l1271_127172

theorem hyperbola_foci_coords :
  let a := 5
  let b := 2
  let c := Real.sqrt (a^2 + b^2)
  ∀ x y : ℝ, 4 * y^2 - 25 * x^2 = 100 →
  (x = 0 ∧ (y = c ∨ y = -c)) := by
  intros a b c x y h
  have h1 : 4 * y^2 = 100 + 25 * x^2 := by linarith
  have h2 : y^2 = 25 + 25/4 * x^2 := by linarith
  have h3 : x = 0 := by sorry
  have h4 : y = c ∨ y = -c := by sorry
  exact ⟨h3, h4⟩

end hyperbola_foci_coords_l1271_127172


namespace most_entries_with_80_yuan_is_c_pass_pass_a_is_cost_effective_after_30_entries_l1271_127178

noncomputable def most_entries_with_80_yuan : Nat :=
let cost_a := 120
let cost_b := 60
let cost_c := 40
let entry_b := 2
let entry_c := 3
let budget := 80
let entries_b := (budget - cost_b) / entry_b
let entries_c := (budget - cost_c) / entry_c
let entries_no_pass := budget / 10
if cost_a <= budget then 
  0
else
  max entries_b (max entries_c entries_no_pass)

theorem most_entries_with_80_yuan_is_c_pass : most_entries_with_80_yuan = 13 :=
by
  sorry

noncomputable def is_pass_a_cost_effective (x : Nat) : Prop :=
let cost_a := 120
let cost_b_entries := 60 + 2 * x
let cost_c_entries := 40 + 3 * x
let cost_no_pass := 10 * x
x > 30 → cost_a < cost_b_entries ∧ cost_a < cost_c_entries ∧ cost_a < cost_no_pass

theorem pass_a_is_cost_effective_after_30_entries : ∀ x : Nat, is_pass_a_cost_effective x :=
by
  sorry

end most_entries_with_80_yuan_is_c_pass_pass_a_is_cost_effective_after_30_entries_l1271_127178


namespace percentage_design_black_is_57_l1271_127134

noncomputable def circleRadius (n : ℕ) : ℝ :=
  3 * (n + 1)

noncomputable def circleArea (n : ℕ) : ℝ :=
  Real.pi * (circleRadius n) ^ 2

noncomputable def totalArea : ℝ :=
  circleArea 6

noncomputable def blackAreas : ℝ :=
  circleArea 0 + (circleArea 2 - circleArea 1) +
  (circleArea 4 - circleArea 3) +
  (circleArea 6 - circleArea 5)

noncomputable def percentageBlack : ℝ :=
  (blackAreas / totalArea) * 100

theorem percentage_design_black_is_57 :
  percentageBlack = 57 := 
by
  sorry

end percentage_design_black_is_57_l1271_127134


namespace certain_number_is_213_l1271_127139

theorem certain_number_is_213 (n : ℕ) (h : n * 16 = 3408) : n = 213 :=
sorry

end certain_number_is_213_l1271_127139


namespace license_plate_count_is_correct_l1271_127171

/-- Define the number of consonants in the English alphabet --/
def num_consonants : Nat := 20

/-- Define the number of possibilities for 'A' --/
def num_A : Nat := 1

/-- Define the number of even digits --/
def num_even_digits : Nat := 5

/-- Define the total number of valid four-character license plates --/
def total_license_plate_count : Nat :=
  num_consonants * num_A * num_consonants * num_even_digits

/-- Theorem stating that the total number of license plates is 2000 --/
theorem license_plate_count_is_correct : 
  total_license_plate_count = 2000 :=
  by
    -- The proof is omitted
    sorry

end license_plate_count_is_correct_l1271_127171


namespace ratio_perimeters_of_squares_l1271_127138

theorem ratio_perimeters_of_squares (a b : ℝ) (h_diag : (a * Real.sqrt 2) / (b * Real.sqrt 2) = 2.5) : (4 * a) / (4 * b) = 10 :=
by
  sorry

end ratio_perimeters_of_squares_l1271_127138


namespace anne_cleans_in_12_hours_l1271_127146

theorem anne_cleans_in_12_hours (B A C : ℝ) (h1 : B + A + C = 1/4)
    (h2 : B + 2 * A + 3 * C = 1/3) (h3 : B + C = 1/6) : 1 / A = 12 :=
by
    sorry

end anne_cleans_in_12_hours_l1271_127146


namespace carousel_ratio_l1271_127113

theorem carousel_ratio (P : ℕ) (h : 3 + P + 2*P + P/3 = 33) : P / 3 = 3 := 
by 
  sorry

end carousel_ratio_l1271_127113


namespace simplify_2A_minus_B_twoA_minusB_value_when_a_neg2_b_1_twoA_minusB_independent_of_a_l1271_127185

def A (a b : ℝ) := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) := 4 * a^2 + 6 * a * b + 8 * a

theorem simplify_2A_minus_B {a b : ℝ} :
  2 * A a b - B a b = -16 * a * b + 6 * b - 8 * a :=
by
  sorry

theorem twoA_minusB_value_when_a_neg2_b_1 :
  2 * A (-2) (1) - B (-2) (1) = 54 :=
by
  sorry

theorem twoA_minusB_independent_of_a {b : ℝ} :
  (∀ a : ℝ, 2 * A a b - B a b = 6 * b - 8 * a) → b = -1 / 2 :=
by
  sorry

end simplify_2A_minus_B_twoA_minusB_value_when_a_neg2_b_1_twoA_minusB_independent_of_a_l1271_127185


namespace problem1_problem2_l1271_127154

-- Problem 1 Lean statement
theorem problem1 (x y : ℝ) (hx : x ≠ 1) (hx' : x ≠ -1) (hy : y ≠ 0) :
    (x^2 - 1) / y / ((x + 1) / y^2) = y * (x - 1) :=
sorry

-- Problem 2 Lean statement
theorem problem2 (m n : ℝ) (hm1 : m ≠ n) (hm2 : m ≠ -n) :
    m / (m + n) + n / (m - n) - 2 * m^2 / (m^2 - n^2) = -1 :=
sorry

end problem1_problem2_l1271_127154


namespace fedya_initial_deposit_l1271_127175

theorem fedya_initial_deposit (n k : ℕ) (h₁ : k < 30) (h₂ : n * (100 - k) = 84700) : 
  n = 1100 :=
by
  sorry

end fedya_initial_deposit_l1271_127175


namespace households_without_car_or_bike_l1271_127190

/--
In a neighborhood having 90 households, some did not have either a car or a bike.
If 16 households had both a car and a bike and 44 had a car, and
there were 35 households with a bike only.
Prove that there are 11 households that did not have either a car or a bike.
-/
theorem households_without_car_or_bike
  (total_households : ℕ)
  (both_car_and_bike : ℕ)
  (car : ℕ)
  (bike_only : ℕ)
  (H1 : total_households = 90)
  (H2 : both_car_and_bike = 16)
  (H3 : car = 44)
  (H4 : bike_only = 35) :
  ∃ N : ℕ, N = total_households - (car - both_car_and_bike + bike_only + both_car_and_bike) ∧ N = 11 :=
by {
  sorry
}

end households_without_car_or_bike_l1271_127190


namespace total_spending_in_4_years_is_680_l1271_127166

-- Define the spending of each person
def trevor_spending_yearly : ℕ := 80
def reed_spending_yearly : ℕ := trevor_spending_yearly - 20
def quinn_spending_yearly : ℕ := reed_spending_yearly / 2

-- Define the total spending in one year
def total_spending_yearly : ℕ := trevor_spending_yearly + reed_spending_yearly + quinn_spending_yearly

-- Let's state what we need to prove
theorem total_spending_in_4_years_is_680 : 
  4 * total_spending_yearly = 680 := 
by 
  -- this is where the proof steps would go
  sorry

end total_spending_in_4_years_is_680_l1271_127166


namespace complement_union_of_M_and_N_l1271_127129

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l1271_127129


namespace bob_pennies_l1271_127100

theorem bob_pennies (a b : ℤ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  sorry

end bob_pennies_l1271_127100


namespace distance_covered_at_40kmph_l1271_127115

def total_distance : ℝ := 250
def speed_40 : ℝ := 40
def speed_60 : ℝ := 60
def total_time : ℝ := 5.2

theorem distance_covered_at_40kmph :
  ∃ (x : ℝ), (x / speed_40 + (total_distance - x) / speed_60 = total_time) ∧ x = 124 :=
  sorry

end distance_covered_at_40kmph_l1271_127115


namespace required_run_rate_l1271_127165

theorem required_run_rate (run_rate_first_10_overs : ℝ) (target_runs total_overs first_overs : ℕ) :
  run_rate_first_10_overs = 4.2 ∧ target_runs = 282 ∧ total_overs = 50 ∧ first_overs = 10 →
  (target_runs - run_rate_first_10_overs * first_overs) / (total_overs - first_overs) = 6 :=
by
  sorry

end required_run_rate_l1271_127165


namespace proof_f_2008_l1271_127163

theorem proof_f_2008 {f : ℝ → ℝ} 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (3 * x + 1) = f (3 * (x + 1) + 1))
  (h3 : f (-1) = -1) : 
  f 2008 = 1 := 
by
  sorry

end proof_f_2008_l1271_127163


namespace completing_the_square_l1271_127131

theorem completing_the_square :
  ∃ d, (∀ x: ℝ, (x^2 - 6 * x + 5 = 0) → ((x - 3)^2 = d)) ∧ d = 4 :=
by
  -- proof goes here
  sorry

end completing_the_square_l1271_127131


namespace average_words_per_minute_l1271_127116

theorem average_words_per_minute 
  (total_words : ℕ) 
  (total_hours : ℕ) 
  (h_words : total_words = 30000) 
  (h_hours : total_hours = 100) : 
  (total_words / total_hours / 60 = 5) := by
  sorry

end average_words_per_minute_l1271_127116


namespace train_crossing_time_l1271_127144

def length_of_train : ℕ := 120
def speed_of_train_kmph : ℕ := 54
def length_of_bridge : ℕ := 660

def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600
def total_distance : ℕ := length_of_train + length_of_bridge
def time_to_cross_bridge : ℕ := total_distance / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross_bridge = 52 :=
sorry

end train_crossing_time_l1271_127144


namespace parallelogram_area_l1271_127127

theorem parallelogram_area (base height : ℝ) (h_base : base = 25) (h_height : height = 15) :
  base * height = 375 :=
by
  subst h_base
  subst h_height
  sorry

end parallelogram_area_l1271_127127


namespace max_S_n_value_arithmetic_sequence_l1271_127183

-- Definitions and conditions
def S_n (n : ℕ) : ℤ := 3 * n - n^2

def a_n (n : ℕ) : ℤ := 
if n = 0 then 0 else S_n n - S_n (n - 1)

-- Statement of the first part of the proof problem
theorem max_S_n_value (n : ℕ) (h : n = 1 ∨ n = 2) : S_n n = 2 :=
sorry

-- Statement of the second part of the proof problem
theorem arithmetic_sequence :
  ∀ n : ℕ, n ≥ 1 → a_n (n + 1) - a_n n = -2 :=
sorry

end max_S_n_value_arithmetic_sequence_l1271_127183


namespace distance_between_stations_l1271_127111

-- Definitions based on conditions in step a):
def speed_train1 : ℝ := 20  -- speed of the first train in km/hr
def speed_train2 : ℝ := 25  -- speed of the second train in km/hr
def extra_distance : ℝ := 55  -- one train has traveled 55 km more

-- Definition of the proof problem
theorem distance_between_stations :
  ∃ D1 D2 T : ℝ, D1 = speed_train1 * T ∧ D2 = speed_train2 * T ∧ D2 = D1 + extra_distance ∧ D1 + D2 = 495 :=
by
  sorry

end distance_between_stations_l1271_127111


namespace geometric_seq_reciprocal_sum_l1271_127143

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop := ∀ n, a (n + 1) = a n * r

theorem geometric_seq_reciprocal_sum
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : geometric_sequence a r)
  (h1 : a 2 * a 5 = -3/4)
  (h2 : a 2 + a 3 + a 4 + a 5 = 5/4) :
  (1 / a 2) + (1 / a 3) + (1 / a 4) + (1 / a 5) = -5/3 := sorry

end geometric_seq_reciprocal_sum_l1271_127143


namespace age_ratio_l1271_127155

theorem age_ratio (Tim_age : ℕ) (John_age : ℕ) (ratio : ℚ) 
  (h1 : Tim_age = 79) 
  (h2 : John_age = 35) 
  (h3 : Tim_age = ratio * John_age - 5) : 
  ratio = 2.4 := 
by sorry

end age_ratio_l1271_127155


namespace y_intercept_of_line_l1271_127103

theorem y_intercept_of_line (m x y b : ℝ) (h1 : m = 4) (h2 : x = 50) (h3 : y = 300) (h4 : y = m * x + b) : b = 100 := by
  sorry

end y_intercept_of_line_l1271_127103


namespace hypotenuse_length_l1271_127173

noncomputable def side_lengths_to_hypotenuse (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 + b^2)

theorem hypotenuse_length 
  (AB BC : ℝ) 
  (h1 : Real.sqrt (AB * BC) = 8) 
  (h2 : (1 / 2) * AB * BC = 48) :
  side_lengths_to_hypotenuse AB BC = 4 * Real.sqrt 13 :=
by
  sorry

end hypotenuse_length_l1271_127173


namespace cubic_inequality_l1271_127142

theorem cubic_inequality (a : ℝ) (h : a ≠ -1) : 
  (1 + a^3) / (1 + a)^3 ≥ 1 / 4 :=
by sorry

end cubic_inequality_l1271_127142


namespace chef_michel_total_pies_l1271_127106

theorem chef_michel_total_pies
  (shepherd_pie_pieces : ℕ)
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ)
  (chicken_pot_pie_customers : ℕ)
  (H1 : shepherd_pie_pieces = 4)
  (H2 : chicken_pot_pie_pieces = 5)
  (H3 : shepherd_pie_customers = 52)
  (H4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) + (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by
  sorry

end chef_michel_total_pies_l1271_127106


namespace fuel_cost_per_liter_l1271_127182

def service_cost_per_vehicle : ℝ := 2.20
def num_minivans : ℕ := 3
def num_trucks : ℕ := 2
def total_cost : ℝ := 347.7
def mini_van_tank_capacity : ℝ := 65
def truck_tank_increase : ℝ := 1.2
def truck_tank_capacity : ℝ := mini_van_tank_capacity * (1 + truck_tank_increase)

theorem fuel_cost_per_liter : 
  let total_service_cost := (num_minivans + num_trucks) * service_cost_per_vehicle
  let total_capacity_minivans := num_minivans * mini_van_tank_capacity
  let total_capacity_trucks := num_trucks * truck_tank_capacity
  let total_fuel_capacity := total_capacity_minivans + total_capacity_trucks
  let fuel_cost := total_cost - total_service_cost
  let cost_per_liter := fuel_cost / total_fuel_capacity
  cost_per_liter = 0.70 := 
  sorry

end fuel_cost_per_liter_l1271_127182


namespace scientific_notation_of_great_wall_l1271_127110

theorem scientific_notation_of_great_wall : 
  ∀ n : ℕ, (6700010 : ℝ) = 6.7 * 10^6 :=
by
  sorry

end scientific_notation_of_great_wall_l1271_127110


namespace longest_side_AB_l1271_127162

-- Definitions of angles in the quadrilateral
def angle_ABC := 65
def angle_BCD := 70
def angle_CDA := 60

/-- In a quadrilateral ABCD with angles as specified, prove that AB is the longest side. -/
theorem longest_side_AB (AB BC CD DA : ℝ) : 
  (angle_ABC = 65 ∧ angle_BCD = 70 ∧ angle_CDA = 60) → 
  AB > DA ∧ AB > BC ∧ AB > CD :=
by
  intros h
  sorry

end longest_side_AB_l1271_127162


namespace mean_points_scored_is_48_l1271_127150

def class_points : List ℤ := [50, 57, 49, 57, 32, 46, 65, 28]

theorem mean_points_scored_is_48 : (class_points.sum / class_points.length) = 48 := by
  sorry

end mean_points_scored_is_48_l1271_127150


namespace min_value_fraction_condition_l1271_127147

noncomputable def minValue (a b : ℝ) := 1 / (2 * a) + a / (b + 1)

theorem min_value_fraction_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  minValue a b = 5 / 4 :=
by
  sorry

end min_value_fraction_condition_l1271_127147


namespace xy_inequality_l1271_127122

theorem xy_inequality (x y : ℝ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  x * y ≤ (x^(n+2) + y^(n+2)) / (x^n + y^n) :=
sorry

end xy_inequality_l1271_127122


namespace min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l1271_127151

-- Part (a): For n = 12:
theorem min_sticks_to_break_for_square_12 : ∀ (n : ℕ), n = 12 → 
  (∃ (sticks : Finset ℕ), sticks.card = 12 ∧ sticks.sum id = 78 ∧ (¬ (78 % 4 = 0) → 
  ∃ (b : ℕ), b = 2)) := 
by sorry

-- Part (b): For n = 15:
theorem can_form_square_without_breaking_15 : ∀ (n : ℕ), n = 15 → 
  (∃ (sticks : Finset ℕ), sticks.card = 15 ∧ sticks.sum id = 120 ∧ (120 % 4 = 0)) :=
by sorry

end min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l1271_127151


namespace inequality_x2_gt_y2_plus_6_l1271_127107

theorem inequality_x2_gt_y2_plus_6 (x y : ℝ) (h1 : x > y) (h2 : y > 3 / (x - y)) : x^2 > y^2 + 6 :=
sorry

end inequality_x2_gt_y2_plus_6_l1271_127107


namespace division_remainder_l1271_127177

theorem division_remainder (dividend divisor quotient remainder : ℕ)
  (h₁ : dividend = 689)
  (h₂ : divisor = 36)
  (h₃ : quotient = 19)
  (h₄ : dividend = divisor * quotient + remainder) :
  remainder = 5 :=
by
  sorry

end division_remainder_l1271_127177


namespace playground_total_l1271_127117

def boys : ℕ := 44
def girls : ℕ := 53

theorem playground_total : boys + girls = 97 := by
  sorry

end playground_total_l1271_127117


namespace prob_club_then_diamond_then_heart_l1271_127133

noncomputable def prob_first_card_club := 13 / 52
noncomputable def prob_second_card_diamond_given_first_club := 13 / 51
noncomputable def prob_third_card_heart_given_first_club_second_diamond := 13 / 50

noncomputable def overall_probability := 
  prob_first_card_club * 
  prob_second_card_diamond_given_first_club * 
  prob_third_card_heart_given_first_club_second_diamond

theorem prob_club_then_diamond_then_heart :
  overall_probability = 2197 / 132600 :=
by
  sorry

end prob_club_then_diamond_then_heart_l1271_127133


namespace solve_for_x_l1271_127157

theorem solve_for_x (x : ℚ) : 
  5*x + 9*x = 450 - 10*(x - 5) -> x = 125/6 :=
by
  sorry

end solve_for_x_l1271_127157


namespace cubic_root_relation_l1271_127195

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem cubic_root_relation
  (x1 x2 x3 : ℝ)
  (hx1x2 : x1 < x2)
  (hx2x3 : x2 < 0)
  (hx3pos : 0 < x3)
  (hfx1 : f x1 = 0)
  (hfx2 : f x2 = 0)
  (hfx3 : f x3 = 0) :
  x3^2 - x2^2 = x3 - x1 :=
sorry

end cubic_root_relation_l1271_127195
