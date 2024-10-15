import Mathlib

namespace NUMINAMATH_GPT_part1_intersection_when_a_is_zero_part2_range_of_a_l1732_173236

-- Definitions of sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | 2 * a - 1 ≤ x ∧ x < a + 5}

-- Part (1): When a = 0, find A ∩ B
theorem part1_intersection_when_a_is_zero :
  A ∩ B 0 = {x : ℝ | -1 < x ∧ x < 5} :=
sorry

-- Part (2): If A ∪ B = A, find the range of values for a
theorem part2_range_of_a (a : ℝ) :
  (A ∪ B a = A) → (0 < a ∧ a ≤ 1) ∨ (6 ≤ a) :=
sorry

end NUMINAMATH_GPT_part1_intersection_when_a_is_zero_part2_range_of_a_l1732_173236


namespace NUMINAMATH_GPT_find_number_l1732_173234

theorem find_number (x : ℝ) (h : (4 / 3) * x = 48) : x = 36 :=
sorry

end NUMINAMATH_GPT_find_number_l1732_173234


namespace NUMINAMATH_GPT_reading_time_difference_l1732_173279

theorem reading_time_difference
  (tristan_speed : ℕ := 120)
  (ella_speed : ℕ := 40)
  (book_pages : ℕ := 360) :
  let tristan_time := book_pages / tristan_speed
  let ella_time := book_pages / ella_speed
  let time_difference_hours := ella_time - tristan_time
  let time_difference_minutes := time_difference_hours * 60
  time_difference_minutes = 360 :=
by
  sorry

end NUMINAMATH_GPT_reading_time_difference_l1732_173279


namespace NUMINAMATH_GPT_max_sum_x_y_l1732_173221

theorem max_sum_x_y {x y a b : ℝ} 
  (hx : 0 < x) (hy : 0 < y) (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ y)
  (h1 : a^2 + y^2 = 2) (h2 : b^2 + x^2 = 1) (h3 : a * x + b * y = 1) : 
  x + y ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_sum_x_y_l1732_173221


namespace NUMINAMATH_GPT_local_minimum_at_2_l1732_173231

noncomputable def f (x m : ℝ) : ℝ := x * (x - m)^2

theorem local_minimum_at_2 (m : ℝ) (h : 2 * (2 - m)^2 + 2 * 4 * (2 - m) = 0) : m = 6 :=
by
  sorry

end NUMINAMATH_GPT_local_minimum_at_2_l1732_173231


namespace NUMINAMATH_GPT_consumer_installment_credit_l1732_173250

theorem consumer_installment_credit (C : ℝ) (A : ℝ) (h1 : A = 0.36 * C) 
    (h2 : 75 = A / 2) : C = 416.67 :=
by
  sorry

end NUMINAMATH_GPT_consumer_installment_credit_l1732_173250


namespace NUMINAMATH_GPT_simplify_fraction_l1732_173272

theorem simplify_fraction (n : ℕ) : 
  (3 ^ (n + 3) - 3 * (3 ^ n)) / (3 * 3 ^ (n + 2)) = 8 / 9 :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1732_173272


namespace NUMINAMATH_GPT_total_right_handed_players_l1732_173281

theorem total_right_handed_players
  (total_players : ℕ)
  (total_throwers : ℕ)
  (left_handed_throwers_perc : ℕ)
  (right_handed_thrower_runs : ℕ)
  (left_handed_thrower_runs : ℕ)
  (total_runs : ℕ)
  (batsmen_to_allrounders_run_ratio : ℕ)
  (proportion_left_right_non_throwers : ℕ)
  (left_handed_non_thrower_runs : ℕ)
  (left_handed_batsmen_eq_allrounders : Prop)
  (left_handed_throwers : ℕ)
  (right_handed_throwers : ℕ)
  (total_right_handed_thrower_runs : ℕ)
  (total_left_handed_thrower_runs : ℕ)
  (total_throwers_runs : ℕ)
  (total_non_thrower_runs : ℕ)
  (allrounder_runs : ℕ)
  (batsmen_runs : ℕ)
  (left_handed_batsmen : ℕ)
  (left_handed_allrounders : ℕ)
  (total_left_handed_non_throwers : ℕ)
  (right_handed_non_throwers : ℕ)
  (total_right_handed_players : ℕ) :
  total_players = 120 →
  total_throwers = 55 →
  left_handed_throwers_perc = 20 →
  right_handed_thrower_runs = 25 →
  left_handed_thrower_runs = 30 →
  total_runs = 3620 →
  batsmen_to_allrounders_run_ratio = 2 →
  proportion_left_right_non_throwers = 5 →
  left_handed_non_thrower_runs = 720 →
  left_handed_batsmen_eq_allrounders →
  left_handed_throwers = total_throwers * left_handed_throwers_perc / 100 →
  right_handed_throwers = total_throwers - left_handed_throwers →
  total_right_handed_thrower_runs = right_handed_throwers * right_handed_thrower_runs →
  total_left_handed_thrower_runs = left_handed_throwers * left_handed_thrower_runs →
  total_throwers_runs = total_right_handed_thrower_runs + total_left_handed_thrower_runs →
  total_non_thrower_runs = total_runs - total_throwers_runs →
  allrounder_runs = total_non_thrower_runs / (batsmen_to_allrounders_run_ratio + 1) →
  batsmen_runs = batsmen_to_allrounders_run_ratio * allrounder_runs →
  left_handed_batsmen = left_handed_non_thrower_runs / (left_handed_thrower_runs * 2) →
  left_handed_allrounders = left_handed_non_thrower_runs / (left_handed_thrower_runs * 2) →
  total_left_handed_non_throwers = left_handed_batsmen + left_handed_allrounders →
  right_handed_non_throwers = total_left_handed_non_throwers * proportion_left_right_non_throwers →
  total_right_handed_players = right_handed_throwers + right_handed_non_throwers →
  total_right_handed_players = 164 :=
by sorry

end NUMINAMATH_GPT_total_right_handed_players_l1732_173281


namespace NUMINAMATH_GPT_exponential_fixed_point_l1732_173276

theorem exponential_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (a^(4-4) + 5 = 6) :=
sorry

end NUMINAMATH_GPT_exponential_fixed_point_l1732_173276


namespace NUMINAMATH_GPT_johns_original_number_l1732_173210

def switch_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  10 * units + tens

theorem johns_original_number :
  ∃ x : ℕ, (10 ≤ x ∧ x < 100) ∧ (∃ y : ℕ, y = 5 * x + 13 ∧ 82 ≤ switch_digits y ∧ switch_digits y ≤ 86 ∧ x = 11) :=
by
  sorry

end NUMINAMATH_GPT_johns_original_number_l1732_173210


namespace NUMINAMATH_GPT_cost_price_of_one_ball_l1732_173274

theorem cost_price_of_one_ball (x : ℝ) (h : 11 * x - 720 = 5 * x) : x = 120 :=
sorry

end NUMINAMATH_GPT_cost_price_of_one_ball_l1732_173274


namespace NUMINAMATH_GPT_slices_needed_l1732_173285

def number_of_sandwiches : ℕ := 5
def slices_per_sandwich : ℕ := 3
def total_slices_required (n : ℕ) (s : ℕ) : ℕ := n * s

theorem slices_needed : total_slices_required number_of_sandwiches slices_per_sandwich = 15 :=
by
  sorry

end NUMINAMATH_GPT_slices_needed_l1732_173285


namespace NUMINAMATH_GPT_circle_radius_of_square_perimeter_eq_area_l1732_173200

theorem circle_radius_of_square_perimeter_eq_area (r : ℝ) (s : ℝ) (h1 : 2 * r = s) (h2 : 4 * s = 8 * r) (h3 : π * r ^ 2 = 8 * r) : r = 8 / π := by
  sorry

end NUMINAMATH_GPT_circle_radius_of_square_perimeter_eq_area_l1732_173200


namespace NUMINAMATH_GPT_find_x_l1732_173257

theorem find_x (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1732_173257


namespace NUMINAMATH_GPT_adult_tickets_sold_l1732_173283

theorem adult_tickets_sold (A C : ℕ) (h1 : A + C = 85) (h2 : 5 * A + 2 * C = 275) : A = 35 := by
  sorry

end NUMINAMATH_GPT_adult_tickets_sold_l1732_173283


namespace NUMINAMATH_GPT_molecular_weight_of_6_moles_Al2_CO3_3_l1732_173242

noncomputable def molecular_weight_Al2_CO3_3: ℝ :=
  let Al_weight := 26.98
  let C_weight := 12.01
  let O_weight := 16.00
  let CO3_weight := C_weight + 3 * O_weight
  let one_mole_weight := 2 * Al_weight + 3 * CO3_weight
  6 * one_mole_weight

theorem molecular_weight_of_6_moles_Al2_CO3_3 : 
  molecular_weight_Al2_CO3_3 = 1403.94 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_6_moles_Al2_CO3_3_l1732_173242


namespace NUMINAMATH_GPT_roots_equation_l1732_173205

theorem roots_equation (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) : p + q = 69 :=
sorry

end NUMINAMATH_GPT_roots_equation_l1732_173205


namespace NUMINAMATH_GPT_james_chore_time_l1732_173216

-- Definitions for the conditions
def t_vacuum : ℕ := 3
def t_chores : ℕ := 3 * t_vacuum
def t_total : ℕ := t_vacuum + t_chores

-- Statement
theorem james_chore_time : t_total = 12 := by
  sorry

end NUMINAMATH_GPT_james_chore_time_l1732_173216


namespace NUMINAMATH_GPT_tan_subtraction_l1732_173246

theorem tan_subtraction (α β : ℝ) (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_tan_subtraction_l1732_173246


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l1732_173299

theorem sum_of_roots_of_quadratic :
  ∀ x1 x2 : ℝ, (∃ a b c, a = -1 ∧ b = 2 ∧ c = 4 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) → (x1 + x2 = 2) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l1732_173299


namespace NUMINAMATH_GPT_rectangle_perimeter_l1732_173263

theorem rectangle_perimeter (u v : ℝ) (π : ℝ) (major minor : ℝ) (area_rect area_ellipse : ℝ) 
  (inscribed : area_ellipse = 4032 * π ∧ area_rect = 4032 ∧ major = 2 * (u + v)) :
  2 * (u + v) = 128 := by
  -- Given: the area of the rectangle, the conditions of the inscribed ellipse, and the major axis constraint.
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1732_173263


namespace NUMINAMATH_GPT_problem_statement_l1732_173244

theorem problem_statement (a b c x : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0)
  (eq1 : (a * x^4 / b * c)^3 = x^3)
  (sum_eq : a + b + c = 9) :
  (x = 1 ∨ x = -1) ∧ a = 1 ∧ b = 4 ∧ c = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1732_173244


namespace NUMINAMATH_GPT_original_average_age_l1732_173287

theorem original_average_age (N : ℕ) (A : ℝ) (h1 : A = 50) (h2 : 12 * 32 + N * 50 = (N + 12) * (A - 4)) : A = 50 := by
  sorry 

end NUMINAMATH_GPT_original_average_age_l1732_173287


namespace NUMINAMATH_GPT_unique_pair_l1732_173298

theorem unique_pair (m n : ℕ) (h1 : m < n) (h2 : n ∣ m^2 + 1) (h3 : m ∣ n^2 + 1) : (m, n) = (1, 1) :=
sorry

end NUMINAMATH_GPT_unique_pair_l1732_173298


namespace NUMINAMATH_GPT_polynomial_inequality_l1732_173229

theorem polynomial_inequality (f : ℝ → ℝ) (h1 : f 0 = 1)
    (h2 : ∀ (x y : ℝ), f (x - y) + f x ≥ 2 * x^2 - 2 * x * y + y^2 + 2 * x - y + 2) :
    f = λ x => x^2 + x + 1 := by
  sorry

end NUMINAMATH_GPT_polynomial_inequality_l1732_173229


namespace NUMINAMATH_GPT_natural_number_sum_of_coprimes_l1732_173223

theorem natural_number_sum_of_coprimes (n : ℕ) (h : n ≥ 2) : ∃ a b : ℕ, n = a + b ∧ Nat.gcd a b = 1 :=
by
  use (n - 1), 1
  sorry

end NUMINAMATH_GPT_natural_number_sum_of_coprimes_l1732_173223


namespace NUMINAMATH_GPT_geometric_sequence_sum_of_first_five_l1732_173247

theorem geometric_sequence_sum_of_first_five :
  (∃ (a : ℕ → ℝ) (r : ℝ),
    (∀ n, n > 0 → a n > 0) ∧
    a 2 = 2 ∧
    a 4 = 8 ∧
    r = 2 ∧
    a 1 = 1 ∧
    a 3 = a 1 * r^2 ∧
    a 5 = a 1 * r^4 ∧
    (a 1 + a 2 + a 3 + a 4 + a 5 = 31)
  ) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_of_first_five_l1732_173247


namespace NUMINAMATH_GPT_problem_statement_l1732_173288

variable (f : ℝ → ℝ) 

def prop1 (f : ℝ → ℝ) : Prop := ∃T > 0, T ≠ 3 / 2 ∧ ∀ x, f (x + T) = f x
def prop2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 3 / 4) = f (-x + 3 / 4)
def prop3 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def prop4 (f : ℝ → ℝ) : Prop := Monotone f

theorem problem_statement (h₁ : ∀ x, f (x + 3 / 2) = -f x)
                          (h₂ : ∀ x, f (x - 3 / 4) = -f (-x - 3 / 4)) : 
                          (¬prop1 f) ∧ (prop2 f) ∧ (prop3 f) ∧ (¬prop4 f) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1732_173288


namespace NUMINAMATH_GPT_focus_coordinates_correct_l1732_173203
noncomputable def ellipse_focus : Real × Real :=
  let center : Real × Real := (4, -1)
  let a : Real := 4
  let b : Real := 1.5
  let c : Real := Real.sqrt (a^2 - b^2)
  (center.1 + c, center.2)

theorem focus_coordinates_correct : 
  ellipse_focus = (7.708, -1) := 
by 
  sorry

end NUMINAMATH_GPT_focus_coordinates_correct_l1732_173203


namespace NUMINAMATH_GPT_pairs_satisfying_equation_l1732_173207

theorem pairs_satisfying_equation (a b : ℝ) : 
  (∀ n : ℕ, n > 0 → a * ⌊b * n⌋ = b * ⌊a * n⌋) ↔ 
  (a = 0 ∨ b = 0 ∨ a = b ∨ ∃ k : ℤ, a = k ∧ b = k) := 
by
  sorry

end NUMINAMATH_GPT_pairs_satisfying_equation_l1732_173207


namespace NUMINAMATH_GPT_mode_is_3_5_of_salaries_l1732_173280

def salaries : List ℚ := [30, 14, 9, 6, 4, 3.5, 3]
def frequencies : List ℕ := [1, 2, 3, 4, 5, 6, 4]

noncomputable def mode_of_salaries (salaries : List ℚ) (frequencies : List ℕ) : ℚ :=
by
  sorry

theorem mode_is_3_5_of_salaries :
  mode_of_salaries salaries frequencies = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_mode_is_3_5_of_salaries_l1732_173280


namespace NUMINAMATH_GPT_christmas_distribution_l1732_173227

theorem christmas_distribution :
  ∃ (n x : ℕ), 
    (240 + 120 + 1 = 361) ∧
    (n * x = 361) ∧
    (n = 19) ∧
    (x = 19) ∧
    ∃ (a b : ℕ), (a + b = 19) ∧ (a * 5 + b * 6 = 100) :=
by
  sorry

end NUMINAMATH_GPT_christmas_distribution_l1732_173227


namespace NUMINAMATH_GPT_smallest_c_for_f_inverse_l1732_173239

noncomputable def f (x : ℝ) : ℝ := (x - 3)^2 - 4

theorem smallest_c_for_f_inverse :
  ∃ c : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≥ c → x₂ ≥ c → f x₁ = f x₂ → x₁ = x₂) ∧ (∀ d : ℝ, d < c → ∃ x₁ x₂ : ℝ, x₁ ≥ d ∧ x₂ ≥ d ∧ f x₁ = f x₂ ∧ x₁ ≠ x₂) ∧ c = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_c_for_f_inverse_l1732_173239


namespace NUMINAMATH_GPT_inv_sum_mod_l1732_173293

theorem inv_sum_mod 
  : (∃ (x y : ℤ), (3 * x ≡ 1 [ZMOD 25]) ∧ (3^2 * y ≡ 1 [ZMOD 25]) ∧ (x + y ≡ 6 [ZMOD 25])) :=
sorry

end NUMINAMATH_GPT_inv_sum_mod_l1732_173293


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1732_173217

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (0 ≤ x ∧ x ≤ 1) → |x| ≤ 1 :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1732_173217


namespace NUMINAMATH_GPT_max_value_expression_l1732_173270

theorem max_value_expression : 
  ∃ x_max : ℝ, 
    (∀ x : ℝ, -3 * x^2 + 15 * x + 9 ≤ -3 * x_max^2 + 15 * x_max + 9) ∧
    (-3 * x_max^2 + 15 * x_max + 9 = 111 / 4) :=
by
  sorry

end NUMINAMATH_GPT_max_value_expression_l1732_173270


namespace NUMINAMATH_GPT_math_proof_problem_l1732_173296

noncomputable def M : ℝ :=
  let x := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / (Real.sqrt (Real.sqrt 7 + 2))
  let y := Real.sqrt (5 - 2 * Real.sqrt 6)
  x - y

theorem math_proof_problem :
  M = (Real.sqrt 57 - 6 * Real.sqrt 6 + 4) / 3 :=
by
  sorry

end NUMINAMATH_GPT_math_proof_problem_l1732_173296


namespace NUMINAMATH_GPT_explicit_form_of_f_l1732_173255

noncomputable def f (x : ℝ) : ℝ := sorry

theorem explicit_form_of_f :
  (∀ x : ℝ, f x + f (x + 3) = 0) →
  (∀ x : ℝ, -1 < x ∧ x ≤ 1 → f x = 2 * x - 3) →
  (∀ x : ℝ, 2 < x ∧ x ≤ 4 → f x = -2 * x + 9) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_explicit_form_of_f_l1732_173255


namespace NUMINAMATH_GPT_cost_per_gallon_l1732_173273

theorem cost_per_gallon (weekly_spend : ℝ) (two_week_usage : ℝ) (weekly_spend_eq : weekly_spend = 36) (two_week_usage_eq : two_week_usage = 24) : 
  (2 * weekly_spend / two_week_usage) = 3 :=
by sorry

end NUMINAMATH_GPT_cost_per_gallon_l1732_173273


namespace NUMINAMATH_GPT_heartsuit_example_l1732_173254

def heartsuit (a b : ℤ) : ℤ := a * b^3 - 2 * b + 3

theorem heartsuit_example : heartsuit 2 3 = 51 :=
by
  sorry

end NUMINAMATH_GPT_heartsuit_example_l1732_173254


namespace NUMINAMATH_GPT_odd_squares_diff_divisible_by_8_l1732_173232

theorem odd_squares_diff_divisible_by_8 (m n : ℤ) (a b : ℤ) (hm : a = 2 * m + 1) (hn : b = 2 * n + 1) : (a^2 - b^2) % 8 = 0 := sorry

end NUMINAMATH_GPT_odd_squares_diff_divisible_by_8_l1732_173232


namespace NUMINAMATH_GPT_quadratic_function_is_explicit_form_l1732_173267

-- Conditions
variable {f : ℝ → ℝ}
variable (H1 : f (-1) = 0)
variable (H2 : ∀ x : ℝ, x ≤ f x ∧ f x ≤ (x^2 + 1) / 2)

-- The quadratic function we aim to prove
def quadratic_function_form_proof (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = (1/4) * x^2 + (1/2) * x + (1/4)

-- Main theorem statement
theorem quadratic_function_is_explicit_form : quadratic_function_form_proof f :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_quadratic_function_is_explicit_form_l1732_173267


namespace NUMINAMATH_GPT_inequality_not_always_true_l1732_173269

theorem inequality_not_always_true
  (x y w : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x^2 > y^2) (hw : w ≠ 0) :
  ∃ w, w ≠ 0 ∧ x^2 * w ≤ y^2 * w :=
sorry

end NUMINAMATH_GPT_inequality_not_always_true_l1732_173269


namespace NUMINAMATH_GPT_find_a_for_even_function_l1732_173233

theorem find_a_for_even_function (a : ℝ) : 
  (∀ x : ℝ, (x + a) * (x - 4) = ((-x) + a) * ((-x) - 4)) → a = 4 :=
by sorry

end NUMINAMATH_GPT_find_a_for_even_function_l1732_173233


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1732_173292

theorem quadratic_inequality_solution_set (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ (b^2 - 4 * a * c) < 0) :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1732_173292


namespace NUMINAMATH_GPT_elizabeth_fruits_l1732_173202

def total_fruits (initial_bananas initial_apples initial_grapes eaten_bananas eaten_apples eaten_grapes : Nat) : Nat :=
  let bananas_left := initial_bananas - eaten_bananas
  let apples_left := initial_apples - eaten_apples
  let grapes_left := initial_grapes - eaten_grapes
  bananas_left + apples_left + grapes_left

theorem elizabeth_fruits : total_fruits 12 7 19 4 2 10 = 22 := by
  sorry

end NUMINAMATH_GPT_elizabeth_fruits_l1732_173202


namespace NUMINAMATH_GPT_product_of_roots_of_quadratic_l1732_173286

   -- Definition of the quadratic equation used in the condition
   def quadratic (x : ℝ) : ℝ := x^2 - 2 * x - 8

   -- Problem statement: Prove that the product of the roots of the given quadratic equation is -8.
   theorem product_of_roots_of_quadratic : 
     (∀ x : ℝ, quadratic x = 0 → (x = 4 ∨ x = -2)) → (4 * -2 = -8) :=
   by
     sorry
   
end NUMINAMATH_GPT_product_of_roots_of_quadratic_l1732_173286


namespace NUMINAMATH_GPT_even_positive_factors_count_l1732_173218

theorem even_positive_factors_count (n : ℕ) (h : n = 2^4 * 3^3 * 7) : 
  ∃ k : ℕ, k = 32 := 
by
  sorry

end NUMINAMATH_GPT_even_positive_factors_count_l1732_173218


namespace NUMINAMATH_GPT_budget_allocation_genetically_modified_microorganisms_l1732_173211

theorem budget_allocation_genetically_modified_microorganisms :
  let microphotonics := 14
  let home_electronics := 19
  let food_additives := 10
  let industrial_lubricants := 8
  let total_percentage := 100
  let basic_astrophysics_percentage := 25
  let known_percentage := microphotonics + home_electronics + food_additives + industrial_lubricants + basic_astrophysics_percentage
  let genetically_modified_microorganisms := total_percentage - known_percentage
  genetically_modified_microorganisms = 24 := 
by
  sorry

end NUMINAMATH_GPT_budget_allocation_genetically_modified_microorganisms_l1732_173211


namespace NUMINAMATH_GPT_function_monotonicity_l1732_173204

theorem function_monotonicity (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 1 → (3 * x^2 + a) < 0) ∧ 
  (∀ x, 1 < x → (3 * x^2 + a) > 0) → 
  (a = -3 ∧ ∃ b : ℝ, true) :=
by {
  sorry
}

end NUMINAMATH_GPT_function_monotonicity_l1732_173204


namespace NUMINAMATH_GPT_sqrt_neg2023_squared_l1732_173248

theorem sqrt_neg2023_squared : Real.sqrt ((-2023 : ℝ)^2) = 2023 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_neg2023_squared_l1732_173248


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_k4_l1732_173289

theorem arithmetic_geometric_sequence_k4 (a : ℕ → ℝ) (d : ℝ) (h_d_ne_zero : d ≠ 0)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_geo_seq : ∃ k : ℕ → ℕ, k 0 = 1 ∧ k 1 = 2 ∧ k 2 = 6 ∧ ∀ i, a (k i + 1) / a (k i) = a (k i + 2) / a (k i + 1)) :
  ∃ k4 : ℕ, k4 = 22 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_k4_l1732_173289


namespace NUMINAMATH_GPT_age_of_replaced_person_is_46_l1732_173222

variable (age_of_replaced_person : ℕ)
variable (new_person_age : ℕ := 16)
variable (decrease_in_age_per_person : ℕ := 3)
variable (number_of_people : ℕ := 10)

theorem age_of_replaced_person_is_46 :
  age_of_replaced_person - new_person_age = decrease_in_age_per_person * number_of_people → 
  age_of_replaced_person = 46 :=
by
  sorry

end NUMINAMATH_GPT_age_of_replaced_person_is_46_l1732_173222


namespace NUMINAMATH_GPT_apples_minimum_count_l1732_173253

theorem apples_minimum_count :
  ∃ n : ℕ, n ≡ 2 [MOD 3] ∧ n ≡ 2 [MOD 4] ∧ n ≡ 2 [MOD 5] ∧ n = 62 := by
sorry

end NUMINAMATH_GPT_apples_minimum_count_l1732_173253


namespace NUMINAMATH_GPT_inscribed_square_area_after_cutting_l1732_173209

theorem inscribed_square_area_after_cutting :
  let original_side := 5
  let cut_side := 1
  let remaining_side := original_side - 2 * cut_side
  let largest_inscribed_square_area := remaining_side ^ 2
  largest_inscribed_square_area = 9 :=
by
  let original_side := 5
  let cut_side := 1
  let remaining_side := original_side - 2 * cut_side
  let largest_inscribed_square_area := remaining_side ^ 2
  show largest_inscribed_square_area = 9
  sorry

end NUMINAMATH_GPT_inscribed_square_area_after_cutting_l1732_173209


namespace NUMINAMATH_GPT_ratio_of_cereal_boxes_l1732_173240

variable (F : ℕ) (S : ℕ) (T : ℕ) (k : ℚ)

def boxes_cereal : Prop :=
  F = 14 ∧
  F + S + T = 33 ∧
  S = k * (F : ℚ) ∧
  S = T - 5 → 
  S / F = 1 / 2

theorem ratio_of_cereal_boxes (F S T : ℕ) (k : ℚ) : 
  boxes_cereal F S T k :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_cereal_boxes_l1732_173240


namespace NUMINAMATH_GPT_cousins_room_distributions_l1732_173284

theorem cousins_room_distributions : 
  let cousins := 5
  let rooms := 4
  let possible_distributions := (1 + 5 + 10 + 10 + 15 + 10 : ℕ)
  possible_distributions = 51 :=
by
  sorry

end NUMINAMATH_GPT_cousins_room_distributions_l1732_173284


namespace NUMINAMATH_GPT_average_salary_all_workers_l1732_173249

-- Define the given conditions as constants
def num_technicians : ℕ := 7
def avg_salary_technicians : ℕ := 12000

def num_workers_total : ℕ := 21
def num_workers_remaining := num_workers_total - num_technicians
def avg_salary_remaining_workers : ℕ := 6000

-- Define the statement we need to prove
theorem average_salary_all_workers :
  let total_salary_technicians := num_technicians * avg_salary_technicians
  let total_salary_remaining_workers := num_workers_remaining * avg_salary_remaining_workers
  let total_salary_all_workers := total_salary_technicians + total_salary_remaining_workers
  let avg_salary_all_workers := total_salary_all_workers / num_workers_total
  avg_salary_all_workers = 8000 :=
by
  sorry

end NUMINAMATH_GPT_average_salary_all_workers_l1732_173249


namespace NUMINAMATH_GPT_height_at_10inches_l1732_173220

theorem height_at_10inches 
  (a : ℚ)
  (h : 20 = (- (4 / 125) * 25 ^ 2 + 20))
  (span_eq : 50 = 50)
  (height_eq : 20 = 20)
  (y_eq : ∀ x : ℚ, - (4 / 125) * x ^ 2 + 20 = 16.8) :
  (- (4 / 125) * 10 ^ 2 + 20) = 16.8 :=
by
  sorry

end NUMINAMATH_GPT_height_at_10inches_l1732_173220


namespace NUMINAMATH_GPT_miki_sandcastle_height_correct_l1732_173251

namespace SandcastleHeight

def sister_sandcastle_height := 0.5
def difference_in_height := 0.3333333333333333
def miki_sandcastle_height := sister_sandcastle_height + difference_in_height

theorem miki_sandcastle_height_correct : miki_sandcastle_height = 0.8333333333333333 := by
  unfold miki_sandcastle_height sister_sandcastle_height difference_in_height
  simp
  sorry

end SandcastleHeight

end NUMINAMATH_GPT_miki_sandcastle_height_correct_l1732_173251


namespace NUMINAMATH_GPT_total_time_correct_l1732_173228

-- Conditions
def minutes_per_story : Nat := 7
def weeks : Nat := 20

-- Total time calculation
def total_minutes : Nat := minutes_per_story * weeks

-- Conversion to hours and minutes
def total_hours : Nat := total_minutes / 60
def remaining_minutes : Nat := total_minutes % 60

-- The proof problem
theorem total_time_correct :
  total_minutes = 140 ∧ total_hours = 2 ∧ remaining_minutes = 20 := by
  sorry

end NUMINAMATH_GPT_total_time_correct_l1732_173228


namespace NUMINAMATH_GPT_italian_clock_hand_coincidence_l1732_173265

theorem italian_clock_hand_coincidence :
  let hour_hand_rotation := 1 / 24
  let minute_hand_rotation := 1
  ∃ (t : ℕ), 0 ≤ t ∧ t < 24 ∧ (t * hour_hand_rotation) % 1 = (t * minute_hand_rotation) % 1
:= sorry

end NUMINAMATH_GPT_italian_clock_hand_coincidence_l1732_173265


namespace NUMINAMATH_GPT_part1_l1732_173224

theorem part1 (P Q R : Polynomial ℝ) : 
  ¬ ∃ (P Q R : Polynomial ℝ), (∀ x y z : ℝ, (x - y + 1)^3 * P.eval x + (y - z - 1)^3 * Q.eval y + (z - 2 * x + 1)^3 * R.eval z = 1) := sorry

end NUMINAMATH_GPT_part1_l1732_173224


namespace NUMINAMATH_GPT_lights_on_bottom_layer_l1732_173262

theorem lights_on_bottom_layer
  (a₁ : ℕ)
  (q : ℕ := 3)
  (S₅ : ℕ := 242)
  (n : ℕ := 5)
  (sum_formula : S₅ = (a₁ * (q^n - 1)) / (q - 1)) :
  (a₁ * q^(n-1) = 162) :=
by
  sorry

end NUMINAMATH_GPT_lights_on_bottom_layer_l1732_173262


namespace NUMINAMATH_GPT_ellipse_foci_distance_l1732_173201

noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∃ (a b : ℝ), (a = 6) ∧ (b = 3) ∧ distance_between_foci a b = 6 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_distance_l1732_173201


namespace NUMINAMATH_GPT_gage_needs_to_skate_l1732_173275

noncomputable def gage_average_skating_time (d1 d2: ℕ) (t1 t2 t8: ℕ) : ℕ :=
  let total_time := (d1 * t1) + (d2 * t2) + t8
  (total_time / (d1 + d2 + 1))

theorem gage_needs_to_skate (t1 t2: ℕ) (d1 d2: ℕ) (avg: ℕ) 
  (t1_minutes: t1 = 80) (t2_minutes: t2 = 105) 
  (days1: d1 = 4) (days2: d2 = 3) (avg_goal: avg = 95) :
  gage_average_skating_time d1 d2 t1 t2 125 = avg :=
by
  sorry

end NUMINAMATH_GPT_gage_needs_to_skate_l1732_173275


namespace NUMINAMATH_GPT_coin_order_correct_l1732_173258

-- Define the coins
inductive Coin
| A | B | C | D | E
deriving DecidableEq

open Coin

-- Define the conditions
def covers (x y : Coin) : Prop :=
  (x = A ∧ y = B) ∨
  (x = C ∧ (y = A ∨ y = D)) ∨
  (x = D ∧ y = B) ∨
  (y = E ∧ x = C)

-- Define the order of coins from top to bottom as a list
def coinOrder : List Coin := [C, E, A, D, B]

-- Prove that the order is correct
theorem coin_order_correct :
  ∀ c₁ c₂ : Coin, c₁ ≠ c₂ → List.indexOf c₁ coinOrder < List.indexOf c₂ coinOrder ↔ covers c₁ c₂ :=
by
  sorry

end NUMINAMATH_GPT_coin_order_correct_l1732_173258


namespace NUMINAMATH_GPT_absolute_value_is_four_l1732_173212

-- Given condition: the absolute value of a number equals 4
theorem absolute_value_is_four (x : ℝ) : abs x = 4 → (x = 4 ∨ x = -4) :=
by
  sorry

end NUMINAMATH_GPT_absolute_value_is_four_l1732_173212


namespace NUMINAMATH_GPT_rad_to_deg_eq_l1732_173214

theorem rad_to_deg_eq : (4 / 3) * 180 = 240 := by
  sorry

end NUMINAMATH_GPT_rad_to_deg_eq_l1732_173214


namespace NUMINAMATH_GPT_subcommittees_with_at_least_one_coach_l1732_173213

-- Definitions based on conditions
def total_members : ℕ := 12
def total_coaches : ℕ := 5
def subcommittee_size : ℕ := 5

-- Lean statement of the problem
theorem subcommittees_with_at_least_one_coach :
  (Nat.choose total_members subcommittee_size) - (Nat.choose (total_members - total_coaches) subcommittee_size) = 771 := by
  sorry

end NUMINAMATH_GPT_subcommittees_with_at_least_one_coach_l1732_173213


namespace NUMINAMATH_GPT_distance_between_points_l1732_173215

theorem distance_between_points :
  let point1 := (2, -3)
  let point2 := (8, 9)
  dist point1 point2 = 6 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_l1732_173215


namespace NUMINAMATH_GPT_house_value_l1732_173226

open Nat

-- Define the conditions
variables (V x : ℕ)
variables (split_amount money_paid : ℕ)
variables (houses_brothers youngest_received : ℕ)
variables (y1 y2 : ℕ)

-- Hypotheses from the conditions
def conditions (V x split_amount money_paid houses_brothers youngest_received y1 y2 : ℕ) :=
  (split_amount = V / 5) ∧
  (houses_brothers = 3) ∧
  (money_paid = 2000) ∧
  (youngest_received = 3000) ∧
  (3 * houses_brothers * money_paid = 6000) ∧
  (y1 = youngest_received) ∧
  (y2 = youngest_received) ∧
  (3 * x + 6000 = V)

-- Main theorem stating the value of one house
theorem house_value (V x : ℕ) (split_amount money_paid houses_brothers youngest_received y1 y2: ℕ) :
  conditions V x split_amount money_paid houses_brothers youngest_received y1 y2 →
  x = 3000 :=
by
  intros
  simp [conditions] at *
  sorry

end NUMINAMATH_GPT_house_value_l1732_173226


namespace NUMINAMATH_GPT_number_of_men_l1732_173294

variable (W M : ℝ)
variable (N_women N_men : ℕ)

theorem number_of_men (h1 : M = 2 * W)
  (h2 : N_women * W * 30 = 21600) :
  (N_men * M * 20 = 14400) → N_men = N_women / 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_men_l1732_173294


namespace NUMINAMATH_GPT_total_photos_newspaper_l1732_173290

-- Define the conditions
def num_pages1 := 12
def photos_per_page1 := 2
def num_pages2 := 9
def photos_per_page2 := 3

-- Define the total number of photos
def total_photos := (num_pages1 * photos_per_page1) + (num_pages2 * photos_per_page2)

-- Prove that the total number of photos is 51
theorem total_photos_newspaper : total_photos = 51 := by
  -- We will fill the proof later
  sorry

end NUMINAMATH_GPT_total_photos_newspaper_l1732_173290


namespace NUMINAMATH_GPT_system_a_l1732_173243

theorem system_a (x y z : ℝ) (h1 : x + y + z = 6) (h2 : 1/x + 1/y + 1/z = 11/6) (h3 : x*y + y*z + z*x = 11) :
  x = 1 ∧ y = 2 ∧ z = 3 ∨ x = 1 ∧ y = 3 ∧ z = 2 ∨ x = 2 ∧ y = 1 ∧ z = 3 ∨ x = 2 ∧ y = 3 ∧ z = 1 ∨ x = 3 ∧ y = 1 ∧ z = 2 ∨ x = 3 ∧ y = 2 ∧ z = 1 :=
sorry

end NUMINAMATH_GPT_system_a_l1732_173243


namespace NUMINAMATH_GPT_total_population_l1732_173271

-- Defining the populations of Springfield and the difference in population
def springfield_population : ℕ := 482653
def population_difference : ℕ := 119666

-- The definition of Greenville's population in terms of Springfield's population
def greenville_population : ℕ := springfield_population - population_difference

-- The statement that we want to prove: the total population of Springfield and Greenville
theorem total_population :
  springfield_population + greenville_population = 845640 := by
  sorry

end NUMINAMATH_GPT_total_population_l1732_173271


namespace NUMINAMATH_GPT_bananas_eaten_l1732_173264

variable (initial_bananas : ℕ) (remaining_bananas : ℕ)

theorem bananas_eaten (initial_bananas remaining_bananas : ℕ) (h_initial : initial_bananas = 12) (h_remaining : remaining_bananas = 10) : initial_bananas - remaining_bananas = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_bananas_eaten_l1732_173264


namespace NUMINAMATH_GPT_pass_rate_eq_l1732_173261

theorem pass_rate_eq (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : (1 - a) * (1 - b) = ab - a - b + 1 :=
by
  sorry

end NUMINAMATH_GPT_pass_rate_eq_l1732_173261


namespace NUMINAMATH_GPT_find_initial_investment_l1732_173295

-- Define the necessary parameters for the problem
variables (P r : ℝ)

-- Given conditions
def condition1 : Prop := P * (1 + r * 3) = 240
def condition2 : Prop := 150 * (1 + r * 6) = 210

-- The statement to be proved
theorem find_initial_investment (h1 : condition1 P r) (h2 : condition2 r) : P = 200 :=
sorry

end NUMINAMATH_GPT_find_initial_investment_l1732_173295


namespace NUMINAMATH_GPT_sector_angle_l1732_173266

-- Define the conditions
def perimeter (r l : ℝ) : ℝ := 2 * r + l
def arc_length (α r : ℝ) : ℝ := α * r

-- Define the problem statement
theorem sector_angle (perimeter_eq : perimeter 1 l = 4) (arc_length_eq : arc_length α 1 = l) : α = 2 := 
by 
  -- remainder of the proof can be added here 
  sorry

end NUMINAMATH_GPT_sector_angle_l1732_173266


namespace NUMINAMATH_GPT_volume_common_part_equal_quarter_volume_each_cone_l1732_173230

theorem volume_common_part_equal_quarter_volume_each_cone
  (r h : ℝ) (V_cone : ℝ)
  (h_cone_volume : V_cone = (1 / 3) * π * r^2 * h) :
  ∃ V_common, V_common = (1 / 4) * V_cone :=
by
  -- Main structure of the proof skipped
  sorry

end NUMINAMATH_GPT_volume_common_part_equal_quarter_volume_each_cone_l1732_173230


namespace NUMINAMATH_GPT_math_problem_l1732_173256

theorem math_problem (x y : ℝ) :
  let A := x^3 + 3*x^2*y + y^3 - 3*x*y^2
  let B := x^2*y - x*y^2
  A - 3*B = x^3 + y^3 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1732_173256


namespace NUMINAMATH_GPT_smallest_piece_length_l1732_173291

theorem smallest_piece_length (x : ℕ) :
  (9 - x) + (14 - x) ≤ (16 - x) → x ≥ 7 :=
by
  sorry

end NUMINAMATH_GPT_smallest_piece_length_l1732_173291


namespace NUMINAMATH_GPT_max_2ab_plus_2bc_sqrt2_l1732_173268

theorem max_2ab_plus_2bc_sqrt2 (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * Real.sqrt 2 ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_max_2ab_plus_2bc_sqrt2_l1732_173268


namespace NUMINAMATH_GPT_function_passes_through_fixed_point_l1732_173235

variables {a : ℝ}

/-- Given the function f(x) = a^(x-1) (a > 0 and a ≠ 1), prove that the function always passes through the point (1, 1) -/
theorem function_passes_through_fixed_point (h1 : a > 0) (h2 : a ≠ 1) :
  (a^(1-1) = 1) :=
by
  sorry

end NUMINAMATH_GPT_function_passes_through_fixed_point_l1732_173235


namespace NUMINAMATH_GPT_number_of_white_dogs_l1732_173252

noncomputable def number_of_brown_dogs : ℕ := 20
noncomputable def number_of_black_dogs : ℕ := 15
noncomputable def total_number_of_dogs : ℕ := 45

theorem number_of_white_dogs : total_number_of_dogs - (number_of_brown_dogs + number_of_black_dogs) = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_white_dogs_l1732_173252


namespace NUMINAMATH_GPT_garden_width_is_correct_l1732_173219

noncomputable def width_of_garden : ℝ :=
  let w := 12 -- We will define the width to be 12 as the final correct answer.
  w

theorem garden_width_is_correct (h_length : ∀ {w : ℝ}, 3 * w = 432 / w) : width_of_garden = 12 := by
  sorry

end NUMINAMATH_GPT_garden_width_is_correct_l1732_173219


namespace NUMINAMATH_GPT_passengers_in_each_car_l1732_173297

theorem passengers_in_each_car (P : ℕ) (h1 : 20 * (P + 2) = 80) : P = 2 := 
by
  sorry

end NUMINAMATH_GPT_passengers_in_each_car_l1732_173297


namespace NUMINAMATH_GPT_non_trivial_solution_exists_l1732_173241

theorem non_trivial_solution_exists (a b c : ℤ) (p : ℕ) [Fact (Nat.Prime p)] :
  ∃ x y z : ℤ, (a * x^2 + b * y^2 + c * z^2) % p = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :=
sorry

end NUMINAMATH_GPT_non_trivial_solution_exists_l1732_173241


namespace NUMINAMATH_GPT_train_length_l1732_173206

theorem train_length (speed_kph : ℕ) (tunnel_length_m : ℕ) (time_s : ℕ) : 
  speed_kph = 54 → 
  tunnel_length_m = 1200 → 
  time_s = 100 → 
  ∃ train_length_m : ℕ, train_length_m = 300 := 
by
  intros h1 h2 h3
  have speed_mps : ℕ := (speed_kph * 1000) / 3600 
  have total_distance_m : ℕ := speed_mps * time_s
  have train_length_m : ℕ := total_distance_m - tunnel_length_m
  use train_length_m
  sorry

end NUMINAMATH_GPT_train_length_l1732_173206


namespace NUMINAMATH_GPT_proof_problem_l1732_173277

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 4 * x^2) - 2 * x) + 3

theorem proof_problem : f (Real.log 2) + f (Real.log (1 / 2)) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l1732_173277


namespace NUMINAMATH_GPT_triangle_area_l1732_173259

-- Define the points P, Q, R and the conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def PQR_right_triangle (P Q R : Point) : Prop := 
  (P.x - R.x)^2 + (P.y - R.y)^2 = 24^2 ∧  -- Length PR
  (Q.x - R.x)^2 + (Q.y - R.y)^2 = 73^2 ∧  -- Length RQ
  (P.x - Q.x)^2 + (P.y - Q.y)^2 = 75^2 ∧  -- Hypotenuse PQ
  (P.y = 3 * P.x + 4) ∧                   -- Median through P
  (Q.y = -Q.x + 5)                        -- Median through Q


noncomputable def area (P Q R : Point) : ℝ := 
  0.5 * abs (P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y))

theorem triangle_area (P Q R : Point) (h : PQR_right_triangle P Q R) : 
  area P Q R = 876 :=
sorry

end NUMINAMATH_GPT_triangle_area_l1732_173259


namespace NUMINAMATH_GPT_total_time_is_11_l1732_173245

-- Define the times each person spent in the pool
def Jerry_time : Nat := 3
def Elaine_time : Nat := 2 * Jerry_time
def George_time : Nat := Elaine_time / 3
def Kramer_time : Nat := 0

-- Define the total time spent in the pool by all friends
def total_time : Nat := Jerry_time + Elaine_time + George_time + Kramer_time

-- Prove that the total time is 11 minutes
theorem total_time_is_11 : total_time = 11 := sorry

end NUMINAMATH_GPT_total_time_is_11_l1732_173245


namespace NUMINAMATH_GPT_perpendicular_lines_b_eq_neg_six_l1732_173282

theorem perpendicular_lines_b_eq_neg_six
    (b : ℝ) :
    (∀ x y : ℝ, 3 * y + 2 * x - 4 = 0 → y = (-2/3) * x + 4/3) →
    (∀ x y : ℝ, 4 * y + b * x - 6 = 0 → y = (-b/4) * x + 3/2) →
    - (2/3) * (-b/4) = -1 →
    b = -6 := 
sorry

end NUMINAMATH_GPT_perpendicular_lines_b_eq_neg_six_l1732_173282


namespace NUMINAMATH_GPT_Liam_homework_assignments_l1732_173237

theorem Liam_homework_assignments : 
  let assignments_needed (points : ℕ) : ℕ := match points with
    | 0     => 0
    | n+1 =>
        if n+1 <= 4 then 1
        else (4 + (((n+1) - 1)/4 - 1))

  30 <= 4 + 8 + 12 + 16 + 20 + 24 + 28 + 16 → ((λ points => List.sum (List.map assignments_needed (List.range points))) 30) = 128 :=
by
  sorry

end NUMINAMATH_GPT_Liam_homework_assignments_l1732_173237


namespace NUMINAMATH_GPT_nickels_count_l1732_173260

theorem nickels_count (original_nickels : ℕ) (additional_nickels : ℕ) 
                        (h₁ : original_nickels = 7) 
                        (h₂ : additional_nickels = 5) : 
    original_nickels + additional_nickels = 12 := 
by sorry

end NUMINAMATH_GPT_nickels_count_l1732_173260


namespace NUMINAMATH_GPT_standard_deviation_is_2_l1732_173208

noncomputable def dataset := [51, 54, 55, 57, 53]

noncomputable def mean (l : List ℝ) : ℝ :=
  ((l.sum : ℝ) / (l.length : ℝ))

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  ((l.map (λ x => (x - m)^2)).sum : ℝ) / (l.length : ℝ)

noncomputable def std_dev (l : List ℝ) : ℝ :=
  Real.sqrt (variance l)

theorem standard_deviation_is_2 :
  mean dataset = 54 →
  std_dev dataset = 2 := by
  intro h_mean
  sorry

end NUMINAMATH_GPT_standard_deviation_is_2_l1732_173208


namespace NUMINAMATH_GPT_number_of_persons_l1732_173238

-- Definitions of the given conditions
def average : ℕ := 15
def average_5 : ℕ := 14
def sum_5 : ℕ := 5 * average_5
def average_9 : ℕ := 16
def sum_9 : ℕ := 9 * average_9
def age_15th : ℕ := 41
def total_sum : ℕ := sum_5 + sum_9 + age_15th

-- The main theorem stating the equivalence
theorem number_of_persons (N : ℕ) (h_average : average * N = total_sum) : N = 17 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_of_persons_l1732_173238


namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_l1732_173225

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1)
noncomputable def g (x : ℝ) : ℝ := Real.exp x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f (x + 1) a + g x

-- Problem Part (1)
theorem problem_part_1 (a : ℝ) (h_pos : 0 < a) :
  (Real.exp 1 - 1) / Real.exp 1 < a ∧ a < (Real.exp 2 - 1) / Real.exp 1 :=
sorry

-- Problem Part (2)
theorem problem_part_2 (a : ℝ) (h_cond : ∀ x, 0 ≤ x → h x a ≥ 1) :
  a ≤ 2 :=
sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_l1732_173225


namespace NUMINAMATH_GPT_number_of_dogs_l1732_173278

theorem number_of_dogs
    (total_animals : ℕ)
    (dogs_ratio : ℕ) (bunnies_ratio : ℕ) (birds_ratio : ℕ)
    (h_total : total_animals = 816)
    (h_ratio : dogs_ratio = 3 ∧ bunnies_ratio = 9 ∧ birds_ratio = 11) :
    (total_animals / (dogs_ratio + bunnies_ratio + birds_ratio) * dogs_ratio = 105) :=
by
    sorry

end NUMINAMATH_GPT_number_of_dogs_l1732_173278
