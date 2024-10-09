import Mathlib

namespace geometric_sequence_common_ratio_l542_54271

variables {a_n : ℕ → ℝ} {S_n q : ℝ}

axiom a1_eq : a_n 1 = 2
axiom an_eq : ∀ n, a_n n = if n > 0 then 2 * q^(n-1) else 0
axiom Sn_eq : ∀ n, a_n n = -64 → S_n = -42 → q = -2

theorem geometric_sequence_common_ratio (q : ℝ) :
  (∀ n, a_n n = if n > 0 then 2 * q^(n-1) else 0) →
  a_n 1 = 2 →
  (∀ n, a_n n = -64 → S_n = -42 → q = -2) :=
by intros _ _ _; sorry

end geometric_sequence_common_ratio_l542_54271


namespace find_two_digit_number_l542_54286

theorem find_two_digit_number : ∃ (y : ℕ), (10 ≤ y ∧ y < 100) ∧ (∃ x : ℕ, x = (y / 10) + (y % 10) ∧ x^3 = y^2) ∧ y = 27 := 
by
  sorry

end find_two_digit_number_l542_54286


namespace abs_eq_abs_implies_l542_54232

theorem abs_eq_abs_implies (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 := 
sorry

end abs_eq_abs_implies_l542_54232


namespace initial_noodles_l542_54217

variable (d w e r : ℕ)

-- Conditions
def gave_to_william (w : ℕ) := w = 15
def gave_to_emily (e : ℕ) := e = 20
def remaining_noodles (r : ℕ) := r = 40

-- The statement to be proven
theorem initial_noodles (h1 : gave_to_william w) (h2 : gave_to_emily e) (h3 : remaining_noodles r) : d = w + e + r := by
  -- Proof will be filled in later.
  sorry

end initial_noodles_l542_54217


namespace cubic_sum_identity_l542_54238

theorem cubic_sum_identity
  (x y z : ℝ)
  (h1 : x + y + z = 8)
  (h2 : x * y + x * z + y * z = 17)
  (h3 : x * y * z = -14) :
  x^3 + y^3 + z^3 = 62 :=
sorry

end cubic_sum_identity_l542_54238


namespace complex_roots_eqn_l542_54262

open Complex

theorem complex_roots_eqn (a b c d e k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) 
  (hk1 : a * k^3 + b * k^2 + c * k + d = e)
  (hk2 : b * k^3 + c * k^2 + d * k + e = a) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I := 
sorry

end complex_roots_eqn_l542_54262


namespace no_nat_solution_for_exp_eq_l542_54267

theorem no_nat_solution_for_exp_eq (n x y z : ℕ) (hn : n > 1) (hx : x ≤ n) (hy : y ≤ n) :
  ¬ (x^n + y^n = z^n) :=
by
  sorry

end no_nat_solution_for_exp_eq_l542_54267


namespace quadratic_roots_diff_square_l542_54296

theorem quadratic_roots_diff_square :
  ∀ (d e : ℝ), (∀ x : ℝ, 4 * x^2 + 8 * x - 48 = 0 → (x = d ∨ x = e)) → (d - e)^2 = 49 :=
by
  intros d e h
  sorry

end quadratic_roots_diff_square_l542_54296


namespace sandra_age_l542_54294

theorem sandra_age (S : ℕ) (h1 : ∀ x : ℕ, x = 14) (h2 : S - 3 = 3 * (14 - 3)) : S = 36 :=
by sorry

end sandra_age_l542_54294


namespace rectangle_length_l542_54248

theorem rectangle_length (b l : ℝ) 
  (h1 : l = 2 * b)
  (h2 : (l - 5) * (b + 5) = l * b + 75) : l = 40 := by
  sorry

end rectangle_length_l542_54248


namespace more_cabbages_produced_l542_54219

theorem more_cabbages_produced
  (square_garden : ∀ n : ℕ, ∃ s : ℕ, s ^ 2 = n)
  (area_per_cabbage : ∀ cabbages : ℕ, cabbages = 11236 → ∃ s : ℕ, s ^ 2 = cabbages) :
  11236 - 105 ^ 2 = 211 := by
sorry

end more_cabbages_produced_l542_54219


namespace shaded_area_of_rectangle_l542_54224

theorem shaded_area_of_rectangle :
  let length := 5   -- Length of the rectangle in cm
  let width := 12   -- Width of the rectangle in cm
  let base := 2     -- Base of each triangle in cm
  let height := 5   -- Height of each triangle in cm
  let rect_area := length * width
  let triangle_area := (1 / 2) * base * height
  let unshaded_area := 2 * triangle_area
  let shaded_area := rect_area - unshaded_area
  shaded_area = 50 :=
by
  -- Calculation follows solution steps.
  sorry

end shaded_area_of_rectangle_l542_54224


namespace arithmetic_seq_num_terms_l542_54256

theorem arithmetic_seq_num_terms (a1 : ℕ := 1) (S_odd S_even : ℕ) (n : ℕ) 
  (h1 : S_odd = 341) (h2 : S_even = 682) : 2 * n = 10 :=
by
  sorry

end arithmetic_seq_num_terms_l542_54256


namespace probability_diamond_or_ace_l542_54246

theorem probability_diamond_or_ace (total_cards : ℕ) (diamonds : ℕ) (aces : ℕ) (jokers : ℕ)
  (not_diamonds_nor_aces : ℕ) (p_not_diamond_nor_ace : ℚ) (p_both_not_diamond_nor_ace : ℚ) : 
  total_cards = 54 →
  diamonds = 13 →
  aces = 4 →
  jokers = 2 →
  not_diamonds_nor_aces = 38 →
  p_not_diamond_nor_ace = 19 / 27 →
  p_both_not_diamond_nor_ace = (19 / 27) ^ 2 →
  1 - p_both_not_diamond_nor_ace = 368 / 729 :=
by 
  intros
  sorry

end probability_diamond_or_ace_l542_54246


namespace find_m_n_calculate_expression_l542_54218

-- Define the polynomials A and B
def A (m x : ℝ) := 5 * x^2 - m * x + 1
def B (n x : ℝ) := 2 * x^2 - 2 * x - n

-- The conditions
variable (x : ℝ) (m n : ℝ)
def no_linear_or_constant_terms (m : ℝ) (n : ℝ) : Prop :=
  ∀ x : ℝ, 3 * x^2 + (2 - m) * x + (1 + n) = 3 * x^2

-- The target theorem
theorem find_m_n 
  (h : no_linear_or_constant_terms m n) : 
  m = 2 ∧ n = -1 := sorry

-- Calculate the expression when m = 2 and n = -1
theorem calculate_expression
  (hm : m = 2)
  (hn : n = -1) : 
  m^2 + n^2 - 2 * m * n = 9 := sorry

end find_m_n_calculate_expression_l542_54218


namespace kanul_spent_on_raw_materials_eq_500_l542_54253

variable (total_amount : ℕ)
variable (machinery_cost : ℕ)
variable (cash_percentage : ℕ)

def amount_spent_on_raw_materials (total_amount machinery_cost cash_percentage : ℕ) : ℕ :=
  total_amount - machinery_cost - (total_amount * cash_percentage / 100)

theorem kanul_spent_on_raw_materials_eq_500 :
  total_amount = 1000 →
  machinery_cost = 400 →
  cash_percentage = 10 →
  amount_spent_on_raw_materials total_amount machinery_cost cash_percentage = 500 :=
by
  intros
  sorry

end kanul_spent_on_raw_materials_eq_500_l542_54253


namespace total_area_of_map_l542_54266

def level1_area : ℕ := 40 * 20
def level2_area : ℕ := 15 * 15
def level3_area : ℕ := (25 * 12) / 2

def total_area : ℕ := level1_area + level2_area + level3_area

theorem total_area_of_map : total_area = 1175 := by
  -- Proof to be completed
  sorry

end total_area_of_map_l542_54266


namespace missing_jar_size_l542_54255

theorem missing_jar_size (x : ℕ) (h₁ : 3 * 16 + 3 * x + 3 * 40 = 252) 
                          (h₂ : 3 + 3 + 3 = 9) : x = 28 := 
by 
  sorry

end missing_jar_size_l542_54255


namespace correct_substitution_l542_54243

theorem correct_substitution (x y : ℝ) 
  (h1 : y = 1 - x) 
  (h2 : x - 2 * y = 4) : x - 2 + 2 * x = 4 :=
by
  sorry

end correct_substitution_l542_54243


namespace x_minus_y_possible_values_l542_54269

theorem x_minus_y_possible_values (x y : ℝ) (hx : x^2 = 9) (hy : |y| = 4) (hxy : x < y) : x - y = -1 ∨ x - y = -7 := 
sorry

end x_minus_y_possible_values_l542_54269


namespace find_f_l542_54216

noncomputable def f (x : ℕ) : ℚ := (1/4) * x * (x + 1) * (2 * x + 1)

lemma f_initial_condition : f 1 = 3 / 2 := by
  sorry

lemma f_functional_equation (x y : ℕ) :
  f (x + y) = (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2 := by
  sorry

theorem find_f (x : ℕ) : f x = (1 / 4) * x * (x + 1) * (2 * x + 1) := by
  sorry

end find_f_l542_54216


namespace expenditure_ratio_l542_54206

variable {I : ℝ} -- Income in the first year

-- Conditions
def first_year_savings (I : ℝ) : ℝ := 0.5 * I
def first_year_expenditure (I : ℝ) : ℝ := I - first_year_savings I
def second_year_income (I : ℝ) : ℝ := 1.5 * I
def second_year_savings (I : ℝ) : ℝ := 2 * first_year_savings I
def second_year_expenditure (I : ℝ) : ℝ := second_year_income I - second_year_savings I

-- Condition statement in Lean
theorem expenditure_ratio (I : ℝ) : 
  let total_expenditure := first_year_expenditure I + second_year_expenditure I
  (total_expenditure / first_year_expenditure I) = 2 :=
  by 
    sorry

end expenditure_ratio_l542_54206


namespace intersection_A_B_l542_54200

def is_log2 (y x : ℝ) : Prop := y = Real.log x / Real.log 2

def set_A (y : ℝ) : Set ℝ := { x | ∃ y, is_log2 y x}
def set_B : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem intersection_A_B : (set_A 1) ∩ set_B = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_A_B_l542_54200


namespace arithmetic_seq_solution_l542_54239

theorem arithmetic_seq_solution (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) 
  (h_nonzero : ∀ n, a n ≠ 0) 
  (h_arith : ∀ n ≥ 2, a (n+1) - a n ^ 2 + a (n-1) = 0) 
  (h_sum : ∀ k, S k = (k * (a 1 + a k)) / 2) :
  S (2 * n - 1) - 4 * n = -2 := 
sorry

end arithmetic_seq_solution_l542_54239


namespace hyperbola_foci_property_l542_54275

noncomputable def hyperbola (x y b : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / b^2) = 1

theorem hyperbola_foci_property (x y b : ℝ) (h : hyperbola x y b) (b_pos : b > 0) (PF1 : ℝ) (PF2 : ℝ) (hPF1 : PF1 = 5) :
  PF2 = 11 :=
by
  sorry

end hyperbola_foci_property_l542_54275


namespace xyz_expression_l542_54249

theorem xyz_expression (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
    (h4 : x + y + z = 0) (h5 : xy + xz + yz ≠ 0) :
    (x^3 + y^3 + z^3) / (xyz * (xy + xz + yz)) = -3 / (2 * (x^2 + y^2 + xy)) :=
by sorry

end xyz_expression_l542_54249


namespace complement_U_A_l542_54257

open Set

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {x | x^2 < 3}

theorem complement_U_A :
  (U \ A) = {-2, 2} :=
sorry

end complement_U_A_l542_54257


namespace systematic_sampling_interval_l542_54231

-- Definition of the population size and sample size
def populationSize : Nat := 800
def sampleSize : Nat := 40

-- The main theorem stating that the interval k in systematic sampling is 20
theorem systematic_sampling_interval : populationSize / sampleSize = 20 := by
  sorry

end systematic_sampling_interval_l542_54231


namespace cheaper_candy_price_l542_54229

theorem cheaper_candy_price
    (mix_total_weight : ℝ) (mix_price_per_pound : ℝ)
    (cheap_weight : ℝ) (expensive_weight : ℝ) (expensive_price_per_pound : ℝ)
    (cheap_total_value : ℝ) (expensive_total_value : ℝ) (total_mix_value : ℝ) :
    mix_total_weight = 80 →
    mix_price_per_pound = 2.20 →
    cheap_weight = 64 →
    expensive_weight = mix_total_weight - cheap_weight →
    expensive_price_per_pound = 3.00 →
    cheap_total_value = cheap_weight * x →
    expensive_total_value = expensive_weight * expensive_price_per_pound →
    total_mix_value = mix_total_weight * mix_price_per_pound →
    total_mix_value = cheap_total_value + expensive_total_value →
    x = 2 := 
sorry

end cheaper_candy_price_l542_54229


namespace derivative_at_x1_is_12_l542_54260

theorem derivative_at_x1_is_12 : 
  (deriv (fun x : ℝ => (2 * x + 1) ^ 2) 1) = 12 :=
by
  sorry

end derivative_at_x1_is_12_l542_54260


namespace num_unique_seven_digit_integers_l542_54281

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

noncomputable def unique_seven_digit_integers : ℕ :=
  factorial 7 / (factorial 2 * factorial 2 * factorial 2)

theorem num_unique_seven_digit_integers : unique_seven_digit_integers = 630 := by
  sorry

end num_unique_seven_digit_integers_l542_54281


namespace division_remainder_l542_54293

def remainder (x y : ℕ) : ℕ := x % y

theorem division_remainder (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : (x : ℚ) / y = 96.15) (h4 : y = 20) : remainder x y = 3 :=
by
  sorry

end division_remainder_l542_54293


namespace student_failed_by_l542_54201

theorem student_failed_by :
  ∀ (total_marks obtained_marks passing_percentage : ℕ),
  total_marks = 700 →
  obtained_marks = 175 →
  passing_percentage = 33 →
  (passing_percentage * total_marks) / 100 - obtained_marks = 56 :=
by
  intros total_marks obtained_marks passing_percentage h1 h2 h3
  sorry

end student_failed_by_l542_54201


namespace third_median_length_l542_54241

noncomputable def triangle_median_length (m₁ m₂ : ℝ) (area : ℝ) : ℝ :=
  if m₁ = 5 ∧ m₂ = 4 ∧ area = 6 * Real.sqrt 5 then
    3 * Real.sqrt 7
  else
    0

theorem third_median_length (m₁ m₂ : ℝ) (area : ℝ)
  (h₁ : m₁ = 5) (h₂ : m₂ = 4) (h₃ : area = 6 * Real.sqrt 5) :
  triangle_median_length m₁ m₂ area = 3 * Real.sqrt 7 :=
by
  -- Proof is skipped
  sorry

end third_median_length_l542_54241


namespace relationship_among_a_b_c_l542_54210

theorem relationship_among_a_b_c :
  let a := (1/6) ^ (1/2)
  let b := Real.log (1/3) / Real.log 6
  let c := Real.log (1/7) / Real.log (1/6)
  c > a ∧ a > b :=
by
  sorry

end relationship_among_a_b_c_l542_54210


namespace suitable_comprehensive_survey_l542_54298

-- Definitions based on conditions

def heights_of_students (n : Nat) : Prop := n = 45
def disease_rate_wheat (area : Type) : Prop := True
def love_for_chrysanthemums (population : Type) : Prop := True
def food_safety_hotel (time : Type) : Prop := True

-- The theorem to prove

theorem suitable_comprehensive_survey : 
  (heights_of_students 45 → True) ∧ 
  (disease_rate_wheat ℕ → False) ∧ 
  (love_for_chrysanthemums ℕ → False) ∧ 
  (food_safety_hotel ℕ → False) →
  heights_of_students 45 :=
by
  intros
  sorry

end suitable_comprehensive_survey_l542_54298


namespace num_shoes_sold_sab_dane_sold_6_pairs_of_shoes_l542_54211

theorem num_shoes_sold (price_shoes : ℕ) (num_shirts : ℕ) (price_shirts : ℕ) (total_earn_per_person : ℕ) : ℕ :=
  let total_earnings_shirts := num_shirts * price_shirts
  let total_earnings := total_earn_per_person * 2
  let earnings_from_shoes := total_earnings - total_earnings_shirts
  let num_shoes_sold := earnings_from_shoes / price_shoes
  num_shoes_sold

theorem sab_dane_sold_6_pairs_of_shoes :
  num_shoes_sold 3 18 2 27 = 6 :=
by
  sorry

end num_shoes_sold_sab_dane_sold_6_pairs_of_shoes_l542_54211


namespace broken_marbles_total_l542_54273

theorem broken_marbles_total :
  let broken_set_1 := 0.10 * 50
  let broken_set_2 := 0.20 * 60
  let broken_set_3 := 0.30 * 70
  let broken_set_4 := 0.15 * 80
  let total_broken := broken_set_1 + broken_set_2 + broken_set_3 + broken_set_4
  total_broken = 50 :=
by
  sorry


end broken_marbles_total_l542_54273


namespace log_diff_decreases_l542_54226

-- Define the natural number n
variable (n : ℕ)

-- Proof statement
theorem log_diff_decreases (hn : 0 < n) : 
  (Real.log (n + 1) - Real.log n) = Real.log (1 + 1 / n) ∧ 
  ∀ m : ℕ, ∀ hn' : 0 < m, m > n → Real.log (m + 1) - Real.log m < Real.log (n + 1) - Real.log n := by
  sorry

end log_diff_decreases_l542_54226


namespace pamela_skittles_l542_54274

variable (initial_skittles : Nat) (given_to_karen : Nat)

def skittles_after_giving (initial_skittles given_to_karen : Nat) : Nat :=
  initial_skittles - given_to_karen

theorem pamela_skittles (h1 : initial_skittles = 50) (h2 : given_to_karen = 7) :
  skittles_after_giving initial_skittles given_to_karen = 43 := by
  sorry

end pamela_skittles_l542_54274


namespace compare_variables_l542_54264

theorem compare_variables (a b c : ℝ) (h1 : a = 2 ^ (1 / 2)) (h2 : b = Real.log 3 / Real.log π) (h3 : c = Real.log (1 / 3) / Real.log 2) : 
  a > b ∧ b > c :=
by
  sorry

end compare_variables_l542_54264


namespace set_intersection_l542_54270

theorem set_intersection (M N : Set ℝ) 
  (hM : M = {x | 2 * x - 3 < 1}) 
  (hN : N = {x | -1 < x ∧ x < 3}) : 
  (M ∩ N) = {x | -1 < x ∧ x < 2} := 
by 
  sorry

end set_intersection_l542_54270


namespace circle_properties_l542_54204

noncomputable def circle_center_and_radius (x y: ℝ) : Prop :=
  (x^2 + 8*x + y^2 - 10*y = 11)

theorem circle_properties :
  (∃ (a b r : ℝ), (a, b) = (-4, 5) ∧ r = 2 * Real.sqrt 13 ∧ circle_center_and_radius x y → a + b + r = 1 + 2 * Real.sqrt 13) :=
  sorry

end circle_properties_l542_54204


namespace parabola_line_intersection_l542_54209

theorem parabola_line_intersection (x1 x2 : ℝ) (h1 : x1 * x2 = 1) (h2 : x1 + 1 = 4) : x2 + 1 = 4 / 3 :=
by
  sorry

end parabola_line_intersection_l542_54209


namespace linear_function_of_additivity_l542_54242

theorem linear_function_of_additivity (f : ℝ → ℝ) 
  (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end linear_function_of_additivity_l542_54242


namespace other_store_pools_l542_54292

variable (P A : ℕ)
variable (three_times : P = 3 * A)
variable (total_pools : P + A = 800)

theorem other_store_pools (three_times : P = 3 * A) (total_pools : P + A = 800) : A = 266 := 
by
  sorry

end other_store_pools_l542_54292


namespace new_average_mark_of_remaining_students_l542_54250

def new_average (total_students : ℕ) (excluded_students : ℕ) (avg_marks : ℕ) (excluded_avg_marks : ℕ) : ℕ :=
  ((total_students * avg_marks) - (excluded_students * excluded_avg_marks)) / (total_students - excluded_students)

theorem new_average_mark_of_remaining_students 
  (total_students : ℕ) (excluded_students : ℕ) (avg_marks : ℕ) (excluded_avg_marks : ℕ)
  (h1 : total_students = 33)
  (h2 : excluded_students = 3)
  (h3 : avg_marks = 90)
  (h4 : excluded_avg_marks = 40) : 
  new_average total_students excluded_students avg_marks excluded_avg_marks = 95 :=
by
  sorry

end new_average_mark_of_remaining_students_l542_54250


namespace part1_part2_l542_54297

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  k - |x - 3|

theorem part1 (k : ℝ) (h : ∀ x, f (x + 3) k ≥ 0 ↔ x ∈ [-1, 1]) : k = 1 :=
sorry

variable (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c)

theorem part2 (h : (1 / a) + (1 / (2 * b)) + (1 / (3 * c)) = 1) : 
  (1 / 9) * a + (2 / 9) * b + (3 / 9) * c ≥ 1 :=
sorry

end part1_part2_l542_54297


namespace typing_time_l542_54299

theorem typing_time (typing_speed : ℕ) (words_per_page : ℕ) (number_of_pages : ℕ) (h1 : typing_speed = 90) (h2 : words_per_page = 450) (h3 : number_of_pages = 10) : 
  (words_per_page / typing_speed) * number_of_pages = 50 := 
by
  sorry

end typing_time_l542_54299


namespace change_received_l542_54287

-- Define the given conditions
def num_apples : ℕ := 5
def cost_per_apple : ℝ := 0.75
def amount_paid : ℝ := 10.00

-- Prove the change is equal to $6.25
theorem change_received :
  amount_paid - (num_apples * cost_per_apple) = 6.25 :=
by
  sorry

end change_received_l542_54287


namespace slope_of_perpendicular_line_l542_54237

theorem slope_of_perpendicular_line (x y : ℝ) (h : 5 * x - 4 * y = 20) : 
  ∃ m : ℝ, m = -4 / 5 :=
sorry

end slope_of_perpendicular_line_l542_54237


namespace simplify_expression_l542_54233

variable (x y : ℝ)

theorem simplify_expression : (3 * x + 4 * x + 5 * y + 2 * y) = 7 * x + 7 * y :=
by
  sorry

end simplify_expression_l542_54233


namespace find_k_l542_54290

noncomputable def proof_problem (x1 x2 x3 x4 : ℝ) (k : ℝ) : Prop :=
  (x1 + x2) / (x3 + x4) = k ∧
  (x3 + x4) / (x1 + x2) = k ∧
  (x1 + x3) / (x2 + x4) = k ∧
  (x2 + x4) / (x1 + x3) = k ∧
  (x1 + x4) / (x2 + x3) = k ∧
  (x2 + x3) / (x1 + x4) = k ∧
  x1 ≠ x2 ∨ x2 ≠ x3 ∨ x3 ≠ x4 ∨ x4 ≠ x1

theorem find_k (x1 x2 x3 x4 : ℝ) (h : proof_problem x1 x2 x3 x4 k) : k = -1 :=
  sorry

end find_k_l542_54290


namespace cos_value_of_tan_third_quadrant_l542_54285

theorem cos_value_of_tan_third_quadrant (x : ℝ) (h1 : Real.tan x = 4 / 3) (h2 : π < x ∧ x < 3 * π / 2) : 
  Real.cos x = -3 / 5 := 
sorry

end cos_value_of_tan_third_quadrant_l542_54285


namespace arithmetic_mean_of_sixty_integers_starting_from_3_l542_54215

def arithmetic_mean_of_sequence (a d n : ℕ) : ℚ :=
  let a_n := a + (n - 1) * d
  let S_n := n * (a + a_n) / 2
  S_n / n

theorem arithmetic_mean_of_sixty_integers_starting_from_3 : arithmetic_mean_of_sequence 3 1 60 = 32.5 :=
by 
  sorry

end arithmetic_mean_of_sixty_integers_starting_from_3_l542_54215


namespace factorize_expression_l542_54240

theorem factorize_expression (x : ℝ) : 9 * x^3 - 18 * x^2 + 9 * x = 9 * x * (x - 1)^2 := 
by 
    sorry

end factorize_expression_l542_54240


namespace min_area_of_rectangle_with_perimeter_100_l542_54221

theorem min_area_of_rectangle_with_perimeter_100 :
  ∃ (length width : ℕ), 
    (length + width = 50) ∧ 
    (length * width = 49) := 
by
  sorry

end min_area_of_rectangle_with_perimeter_100_l542_54221


namespace points_lie_on_line_l542_54214

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
    let x := (t + 2) / t
    let y := (t - 2) / t
    x + y = 2 :=
by
  let x := (t + 2) / t
  let y := (t - 2) / t
  sorry

end points_lie_on_line_l542_54214


namespace complement_of_M_in_U_is_14_l542_54276

def U : Set ℕ := {x | x < 5 ∧ x > 0}

def M : Set ℕ := {x | x^2 - 5 * x + 6 = 0}

theorem complement_of_M_in_U_is_14 : 
  {x | x ∈ U ∧ x ∉ M} = {1, 4} :=
by
  sorry

end complement_of_M_in_U_is_14_l542_54276


namespace range_of_a_l542_54258

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + 2 * x - Real.exp x

theorem range_of_a (h : ∀ m n : ℝ, 0 < m → 0 < n → m > n → (f a m - f a n) / (m - n) < 2) :
  a ≤ Real.exp 1 / (2 * 1) := 
sorry

end range_of_a_l542_54258


namespace length_of_side_of_pentagon_l542_54280

-- Assuming these conditions from the math problem:
-- 1. The perimeter of the regular polygon is 125.
-- 2. The polygon is a pentagon (5 sides).

-- Let's define the conditions:
def perimeter := 125
def sides := 5
def regular_polygon (perimeter : ℕ) (sides : ℕ) := (perimeter / sides : ℕ)

-- Statement to be proved:
theorem length_of_side_of_pentagon : regular_polygon perimeter sides = 25 := 
by sorry

end length_of_side_of_pentagon_l542_54280


namespace max_lambda_inequality_l542_54268

theorem max_lambda_inequality (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (1 / Real.sqrt (20 * a + 23 * b) + 1 / Real.sqrt (23 * a + 20 * b)) ≥ (2 / Real.sqrt 43 / Real.sqrt (a + b)) :=
by
  sorry

end max_lambda_inequality_l542_54268


namespace max_rectangle_area_under_budget_l542_54208

/-- 
Let L and W be the length and width of a rectangle, respectively, where:
1. The length L is made of materials priced at 3 yuan per meter.
2. The width W is made of materials priced at 5 yuan per meter.
3. Both L and W are integers.
4. The total cost 3L + 5W does not exceed 100 yuan.

Prove that the maximum area of the rectangle that can be made under these constraints is 40 square meters.
--/
theorem max_rectangle_area_under_budget :
  ∃ (L W : ℤ), 3 * L + 5 * W ≤ 100 ∧ 0 ≤ L ∧ 0 ≤ W ∧ L * W = 40 :=
sorry

end max_rectangle_area_under_budget_l542_54208


namespace total_students_in_halls_l542_54244

theorem total_students_in_halls :
  let S_g := 30
  let S_b := 2 * S_g
  let S_m := 3 / 5 * (S_g + S_b)
  S_g + S_b + S_m = 144 :=
by
  sorry

end total_students_in_halls_l542_54244


namespace abs_eq_4_reciprocal_eq_self_l542_54235

namespace RationalProofs

-- Problem 1
theorem abs_eq_4 (x : ℚ) : |x| = 4 ↔ x = 4 ∨ x = -4 :=
by sorry

-- Problem 2
theorem reciprocal_eq_self (x : ℚ) : x ≠ 0 → x⁻¹ = x ↔ x = 1 ∨ x = -1 :=
by sorry

end RationalProofs

end abs_eq_4_reciprocal_eq_self_l542_54235


namespace chess_tournament_participants_l542_54295

open Int

theorem chess_tournament_participants (n : ℕ) (h_games: n * (n - 1) / 2 = 190) : n = 20 :=
by
  sorry

end chess_tournament_participants_l542_54295


namespace evaluateExpression_at_3_l542_54277

noncomputable def evaluateExpression (x : ℚ) : ℚ :=
  (x - 1 + (2 - 2 * x) / (x + 1)) / ((x * x - x) / (x + 1))

theorem evaluateExpression_at_3 : evaluateExpression 3 = 2 / 3 := by
  sorry

end evaluateExpression_at_3_l542_54277


namespace swimming_speed_still_water_l542_54284

theorem swimming_speed_still_water 
  (v t : ℝ) 
  (h1 : 3 = (v + 3) * t / (v - 3)) 
  (h2 : t ≠ 0) :
  v = 9 :=
by
  sorry

end swimming_speed_still_water_l542_54284


namespace proof_problem_l542_54234

noncomputable def problem : ℚ :=
  let a := 1
  let b := 2
  let c := 1
  let d := 0
  a + 2 * b + 3 * c + 4 * d

theorem proof_problem : problem = 8 := by
  -- All computations are visible here
  unfold problem
  rfl

end proof_problem_l542_54234


namespace max_angle_C_l542_54247

-- Define the necessary context and conditions
variable {a b c : ℝ}

-- Condition that a^2 + b^2 = 2c^2 in a triangle
axiom triangle_condition : a^2 + b^2 = 2 * c^2

-- Theorem statement
theorem max_angle_C (h : a^2 + b^2 = 2 * c^2) : ∃ C : ℝ, C = Real.pi / 3 := sorry

end max_angle_C_l542_54247


namespace greatest_integer_x_l542_54212

theorem greatest_integer_x (x : ℤ) : 
  (∃ k : ℤ, (x - 4) = k ∧ x^2 - 3 * x + 4 = k * (x - 4) + 8) →
  x ≤ 12 :=
by
  sorry

end greatest_integer_x_l542_54212


namespace max_min_values_f_l542_54225

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem max_min_values_f :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ 2) ∧
  (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 2) ∧
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ Real.sqrt 3) ∧
  (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = Real.sqrt 3) :=
by
  sorry

end max_min_values_f_l542_54225


namespace circle_range_of_m_l542_54230

theorem circle_range_of_m (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x + 2 * m * y + 2 * m^2 + m - 1 = 0 → (2 * m^2 + m - 1 = 0)) → (-2 < m) ∧ (m < 2/3) :=
by
  sorry

end circle_range_of_m_l542_54230


namespace percentage_of_second_division_l542_54227

theorem percentage_of_second_division
  (total_students : ℕ)
  (students_first_division : ℕ)
  (students_just_passed : ℕ)
  (h1: total_students = 300)
  (h2: students_first_division = 75)
  (h3: students_just_passed = 63) :
  (total_students - (students_first_division + students_just_passed)) * 100 / total_students = 54 := 
by
  -- Proof will be added later
  sorry

end percentage_of_second_division_l542_54227


namespace coffee_cost_l542_54205

def ounces_per_donut : ℕ := 2
def ounces_per_pot : ℕ := 12
def cost_per_pot : ℕ := 3
def dozens_of_donuts : ℕ := 3
def donuts_per_dozen : ℕ := 12

theorem coffee_cost :
  let total_donuts := dozens_of_donuts * donuts_per_dozen
  let total_ounces := ounces_per_donut * total_donuts
  let total_pots := total_ounces / ounces_per_pot
  let total_cost := total_pots * cost_per_pot
  total_cost = 18 := by
  sorry

end coffee_cost_l542_54205


namespace math_problem_l542_54282

theorem math_problem : 3 * 3^4 + 9^60 / 9^59 - 27^3 = -19431 := by
  sorry

end math_problem_l542_54282


namespace smallest_m_plus_n_l542_54263

theorem smallest_m_plus_n (m n : ℕ) (hmn : m > n) (hid : (2012^m : ℕ) % 1000 = (2012^n) % 1000) : m + n = 104 :=
sorry

end smallest_m_plus_n_l542_54263


namespace times_older_l542_54222

-- Conditions
variables (H S : ℕ)
axiom hold_age : H = 36
axiom hold_son_relation : H = 3 * S

-- Statement of the problem
theorem times_older (H S : ℕ) (h1 : H = 36) (h2 : H = 3 * S) : (H - 8) / (S - 8) = 7 :=
by
  -- Proof will be provided here
  sorry

end times_older_l542_54222


namespace find_x_in_terms_of_a_b_l542_54203

variable (a b x : ℝ)
variable (ha : a > 0) (hb : b > 0) (hx : x > 0) (r : ℝ)
variable (h1 : r = (4 * a)^(3 * b))
variable (h2 : r = a ^ b * x ^ b)

theorem find_x_in_terms_of_a_b 
  (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h1 : (4 * a)^(3 * b) = r)
  (h2 : r = a^b * x^b) :
  x = 64 * a^2 :=
by
  sorry

end find_x_in_terms_of_a_b_l542_54203


namespace div_by_30_l542_54220

theorem div_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end div_by_30_l542_54220


namespace no_real_intersection_l542_54213

def parabola_line_no_real_intersection : Prop :=
  let a := 3
  let b := -6
  let c := 5
  (b^2 - 4 * a * c) < 0

theorem no_real_intersection (h : parabola_line_no_real_intersection) : 
  ∀ x : ℝ, 3*x^2 - 4*x + 2 ≠ 2*x - 3 :=
by sorry

end no_real_intersection_l542_54213


namespace polynomial_remainder_l542_54288

theorem polynomial_remainder (p q r : Polynomial ℝ) (h1 : p.eval 2 = 6) (h2 : p.eval 4 = 14)
  (r_deg : r.degree < 2) :
  p = q * (X - 2) * (X - 4) + r → r = 4 * X - 2 :=
by
  sorry

end polynomial_remainder_l542_54288


namespace log_ordering_correct_l542_54261

noncomputable def log_ordering : Prop :=
  let a := 20.3
  let b := 0.32
  let c := Real.log b
  (0 < b ∧ b < 1) ∧ (c < 0) ∧ (c < b ∧ b < a)

theorem log_ordering_correct : log_ordering :=
by
  -- skipped proof
  sorry

end log_ordering_correct_l542_54261


namespace common_solutions_y_values_l542_54291

theorem common_solutions_y_values :
  (∃ x : ℝ, x^2 + y^2 - 16 = 0 ∧ x^2 - 3*y - 12 = 0) ↔ (y = -4 ∨ y = 1) :=
by {
  sorry
}

end common_solutions_y_values_l542_54291


namespace red_peppers_weight_correct_l542_54223

def weight_of_red_peppers : Prop :=
  ∀ (T G : ℝ), (T = 0.66) ∧ (G = 0.33) → (T - G = 0.33)

theorem red_peppers_weight_correct : weight_of_red_peppers :=
  sorry

end red_peppers_weight_correct_l542_54223


namespace sequence_sum_l542_54254

-- Definitions representing the given conditions
variables (A H M O X : ℕ)

-- Assuming the conditions as hypotheses
theorem sequence_sum (h₁ : A + 9 + H = 19) (h₂ : 9 + H + M = 19) (h₃ : H + M + O = 19)
  (h₄ : M + O + X = 19) : A + H + M + O = 26 :=
sorry

end sequence_sum_l542_54254


namespace remainder_zero_l542_54236

theorem remainder_zero {n : ℕ} (h : n > 0) : 
  (2013^n - 1803^n - 1781^n + 1774^n) % 203 = 0 :=
by {
  sorry
}

end remainder_zero_l542_54236


namespace kite_initial_gain_percentage_l542_54272

noncomputable def initial_gain_percentage (MP CP : ℝ) : ℝ :=
  ((MP - CP) / CP) * 100

theorem kite_initial_gain_percentage :
  ∃ MP CP : ℝ,
    SP = 30 ∧
    SP = MP * 0.9 ∧
    1.035 * CP = SP ∧
    initial_gain_percentage MP CP = 15 :=
sorry

end kite_initial_gain_percentage_l542_54272


namespace range_of_m_l542_54283

theorem range_of_m (α : ℝ) (m : ℝ) (h : (α > π ∧ α < 3 * π / 2) ∨ (α > 3 * π / 2 ∧ α < 2 * π)) :
  -1 < (Real.sin α) ∧ (Real.sin α) < 0 ∧ (Real.sin α) = (2 * m - 3) / (4 - m) → 
  m ∈ Set.Ioo (-1 : ℝ) (3 / 2 : ℝ) :=
by
  sorry

end range_of_m_l542_54283


namespace part1_min_value_part2_min_value_l542_54279

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem part1_min_value :
  ∃ (m : ℝ), m = 2 ∧ (∀ (x : ℝ), f x ≥ m) :=
sorry

theorem part2_min_value (a b : ℝ) (h : a^2 + b^2 = 2) :
  ∃ (y : ℝ), y = (1 / (a^2 + 1) + 4 / (b^2 + 1)) ∧ y = 9 / 4 :=
sorry

end part1_min_value_part2_min_value_l542_54279


namespace smith_family_seating_problem_l542_54265

theorem smith_family_seating_problem :
  let total_children := 8
  let boys := 4
  let girls := 4
  (total_children.factorial - (boys.factorial * girls.factorial)) = 39744 :=
by
  sorry

end smith_family_seating_problem_l542_54265


namespace probability_of_team_with_2_girls_2_boys_l542_54245

open Nat

-- Define the combinatorics function for binomial coefficients
def binomial (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem probability_of_team_with_2_girls_2_boys :
  let total_women := 8
  let total_men := 6
  let team_size := 4
  let ways_to_choose_2_girls := binomial total_women 2
  let ways_to_choose_2_boys := binomial total_men 2
  let total_ways_to_form_team := binomial (total_women + total_men) team_size
  let favorable_outcomes := ways_to_choose_2_girls * ways_to_choose_2_boys
  (favorable_outcomes : ℚ) / total_ways_to_form_team = 60 / 143 := 
by sorry

end probability_of_team_with_2_girls_2_boys_l542_54245


namespace triangular_weight_l542_54251

noncomputable def rectangular_weight := 90
variables {C T : ℕ}

-- Conditions
axiom cond1 : C + T = 3 * C
axiom cond2 : 4 * C + T = T + C + rectangular_weight

-- Question: How much does the triangular weight weigh?
theorem triangular_weight : T = 60 :=
sorry

end triangular_weight_l542_54251


namespace teddy_has_8_cats_l542_54289

theorem teddy_has_8_cats (dogs_teddy : ℕ) (cats_teddy : ℕ) (dogs_total : ℕ) (pets_total : ℕ)
  (h1 : dogs_teddy = 7)
  (h2 : dogs_total = dogs_teddy + (dogs_teddy + 9) + (dogs_teddy - 5))
  (h3 : pets_total = dogs_total + cats_teddy + (cats_teddy + 13))
  (h4 : pets_total = 54) :
  cats_teddy = 8 := by
  sorry

end teddy_has_8_cats_l542_54289


namespace evaluate_expression_l542_54278

theorem evaluate_expression (x : ℝ) (h1 : x^5 + 1 ≠ 0) (h2 : x^5 - 1 ≠ 0) :
  ( ((x^2 - 2*x + 2)^2 * (x^3 - x^2 + 1)^2 / (x^5 + 1)^2)^2 *
    ((x^2 + 2*x + 2)^2 * (x^3 + x^2 + 1)^2 / (x^5 - 1)^2)^2 )
  = 1 := 
by 
  sorry

end evaluate_expression_l542_54278


namespace correct_statement_about_residuals_l542_54228

-- Define the properties and characteristics of residuals as per the definition
axiom residuals_definition : Prop
axiom residuals_usefulness : residuals_definition → Prop

-- The theorem to prove that the correct statement about residuals is that they can be used to assess the effectiveness of model fitting
theorem correct_statement_about_residuals (h : residuals_definition) : residuals_usefulness h :=
sorry

end correct_statement_about_residuals_l542_54228


namespace inequality_proof_l542_54259

variable (x y : ℝ)
variable (h1 : x ≥ 0)
variable (h2 : y ≥ 0)
variable (h3 : x + y ≤ 1)

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y ≤ 1) : 
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := 
by 
  sorry

end inequality_proof_l542_54259


namespace quadrilateral_with_exactly_two_axes_of_symmetry_is_either_rectangle_or_rhombus_l542_54252

structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

def has_exactly_two_axes_of_symmetry (q : Quadrilateral) : Prop :=
  -- Definition to be developed further based on symmetry conditions
  sorry

def is_rectangle (q : Quadrilateral) : Prop :=
  -- Definition to be developed further based on properties of rectangle
  sorry

def is_rhombus (q : Quadrilateral) : Prop :=
  -- Definition to be developed further based on properties of rhombus
  sorry

theorem quadrilateral_with_exactly_two_axes_of_symmetry_is_either_rectangle_or_rhombus
  (q : Quadrilateral)
  (h : has_exactly_two_axes_of_symmetry q) :
  is_rectangle q ∨ is_rhombus q := by
  sorry

end quadrilateral_with_exactly_two_axes_of_symmetry_is_either_rectangle_or_rhombus_l542_54252


namespace sufficient_but_not_necessary_l542_54207

theorem sufficient_but_not_necessary {a b : ℝ} (h1 : a > 1) (h2 : b > 2) :
  (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 1 ∧ b > 2)) :=
by
  sorry

end sufficient_but_not_necessary_l542_54207


namespace problem1_problem2_l542_54202

-- Problem 1
theorem problem1 : 5*Real.sqrt 2 - (Real.sqrt 18 + Real.sqrt (1/2)) = (3/2) * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem2 : (2*Real.sqrt 3 - 1)^2 + (Real.sqrt 24) / (Real.sqrt 2) = 13 - 2*Real.sqrt 3 := by
  sorry

end problem1_problem2_l542_54202
