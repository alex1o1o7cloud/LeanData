import Mathlib

namespace NUMINAMATH_GPT_vectors_parallel_x_squared_eq_two_l774_77426

theorem vectors_parallel_x_squared_eq_two (x : ℝ) 
  (a : ℝ × ℝ := (x+2, 1+x)) 
  (b : ℝ × ℝ := (x-2, 1-x)) 
  (parallel : (a.1 * b.2 - a.2 * b.1) = 0) : x^2 = 2 :=
sorry

end NUMINAMATH_GPT_vectors_parallel_x_squared_eq_two_l774_77426


namespace NUMINAMATH_GPT_fewer_bees_than_flowers_l774_77462

theorem fewer_bees_than_flowers :
  (5 - 3 = 2) :=
by
  sorry

end NUMINAMATH_GPT_fewer_bees_than_flowers_l774_77462


namespace NUMINAMATH_GPT_trigonometric_bound_l774_77446

open Real

theorem trigonometric_bound (x y : ℝ) : 
  -1/2 ≤ (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ∧ 
  (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ≤ 1/2 :=
by 
  sorry

end NUMINAMATH_GPT_trigonometric_bound_l774_77446


namespace NUMINAMATH_GPT_inequality_for_pos_reals_equality_condition_l774_77433

open Real

theorem inequality_for_pos_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / c + c / b ≥ 4 * a / (a + b) :=
by
  -- Theorem Statement Proof Skeleton
  sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / c + c / b = 4 * a / (a + b)) ↔ (a = b ∧ b = c) :=
by
  -- Theorem Statement Proof Skeleton
  sorry

end NUMINAMATH_GPT_inequality_for_pos_reals_equality_condition_l774_77433


namespace NUMINAMATH_GPT_descending_order_of_numbers_l774_77481

theorem descending_order_of_numbers :
  let a := 62
  let b := 78
  let c := 64
  let d := 59
  b > c ∧ c > a ∧ a > d :=
by
  let a := 62
  let b := 78
  let c := 64
  let d := 59
  sorry

end NUMINAMATH_GPT_descending_order_of_numbers_l774_77481


namespace NUMINAMATH_GPT_base_number_is_two_l774_77424

theorem base_number_is_two (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^22)
  (h2 : n = 21) : x = 2 :=
sorry

end NUMINAMATH_GPT_base_number_is_two_l774_77424


namespace NUMINAMATH_GPT_rationalize_simplify_l774_77427

theorem rationalize_simplify :
  3 / (Real.sqrt 75 + Real.sqrt 3) = Real.sqrt 3 / 6 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_simplify_l774_77427


namespace NUMINAMATH_GPT_negation_proof_l774_77477

theorem negation_proof (a b : ℝ) : 
  (¬ (a > b → 2 * a > 2 * b - 1)) = (a ≤ b → 2 * a ≤ 2 * b - 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_proof_l774_77477


namespace NUMINAMATH_GPT_samantha_spends_36_dollars_l774_77468

def cost_per_toy : ℝ := 12.00
def discount_factor : ℝ := 0.5
def num_toys_bought : ℕ := 4

def total_spent (cost_per_toy : ℝ) (discount_factor : ℝ) (num_toys_bought : ℕ) : ℝ :=
  let pair_cost := cost_per_toy + (cost_per_toy * discount_factor)
  let num_pairs := num_toys_bought / 2
  num_pairs * pair_cost

theorem samantha_spends_36_dollars :
  total_spent cost_per_toy discount_factor num_toys_bought = 36.00 :=
sorry

end NUMINAMATH_GPT_samantha_spends_36_dollars_l774_77468


namespace NUMINAMATH_GPT_root_exists_in_interval_l774_77401

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

theorem root_exists_in_interval :
  ∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0 := 
sorry

end NUMINAMATH_GPT_root_exists_in_interval_l774_77401


namespace NUMINAMATH_GPT_find_x_squared_minus_y_squared_l774_77429

theorem find_x_squared_minus_y_squared 
  (x y : ℝ)
  (h1 : x + y = 5)
  (h2 : x - y = 1) :
  x^2 - y^2 = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_x_squared_minus_y_squared_l774_77429


namespace NUMINAMATH_GPT_Raine_steps_to_school_l774_77482

-- Define Raine's conditions
variable (steps_total : ℕ) (days : ℕ) (round_trip_steps : ℕ)

-- Given conditions
def Raine_conditions := steps_total = 1500 ∧ days = 5 ∧ round_trip_steps = steps_total / days

-- Prove that the steps to school is 150 given Raine's conditions
theorem Raine_steps_to_school (h : Raine_conditions 1500 5 300) : (300 / 2) = 150 :=
by
  sorry

end NUMINAMATH_GPT_Raine_steps_to_school_l774_77482


namespace NUMINAMATH_GPT_application_methods_l774_77459

variables (students : Fin 6) (colleges : Fin 3)

def total_applications_without_restriction : ℕ := 3^6
def applications_missing_one_college : ℕ := 2^6
def overcounted_applications_missing_two_college : ℕ := 1

theorem application_methods (h1 : total_applications_without_restriction = 729)
    (h2 : applications_missing_one_college = 64)
    (h3 : overcounted_applications_missing_two_college = 1) :
    ∀ (students : Fin 6), ∀ (colleges : Fin 3),
      (total_applications_without_restriction - 3 * applications_missing_one_college + 3 * overcounted_applications_missing_two_college = 540) :=
by {
  sorry
}

end NUMINAMATH_GPT_application_methods_l774_77459


namespace NUMINAMATH_GPT_age_difference_l774_77463

-- Let D denote the daughter's age and M denote the mother's age
variable (D M : ℕ)

-- Conditions given in the problem
axiom h1 : M = 11 * D
axiom h2 : M + 13 = 2 * (D + 13)

-- The main proof statement to show the difference in their current ages
theorem age_difference : M - D = 40 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l774_77463


namespace NUMINAMATH_GPT_radius_of_circumscribed_sphere_eq_a_l774_77460

-- Assume a to be a real number representing the side length of the base and height of the hexagonal pyramid
variables (a : ℝ)

-- Representing the base as a regular hexagon and the pyramid as having equal side length and height
def regular_hexagonal_pyramid (a : ℝ) : Type := {b : ℝ // b = a}

-- The radius of the circumscribed sphere to a given regular hexagonal pyramid
def radius_of_circumscribed_sphere (a : ℝ) : ℝ := a

-- Theorem stating that the radius of the sphere circumscribed around a regular hexagonal pyramid 
-- with side length and height both equal to a is a
theorem radius_of_circumscribed_sphere_eq_a (a : ℝ) :
  radius_of_circumscribed_sphere a = a :=
by {
  sorry
}

end NUMINAMATH_GPT_radius_of_circumscribed_sphere_eq_a_l774_77460


namespace NUMINAMATH_GPT_integral_sqrt_1_minus_x_sq_plus_2x_l774_77492

theorem integral_sqrt_1_minus_x_sq_plus_2x :
  ∫ x in (0 : Real)..1, (Real.sqrt (1 - x^2) + 2 * x) = (Real.pi + 4) / 4 := by
  sorry

end NUMINAMATH_GPT_integral_sqrt_1_minus_x_sq_plus_2x_l774_77492


namespace NUMINAMATH_GPT_average_age_of_dance_group_l774_77451

theorem average_age_of_dance_group
  (avg_age_children : ℕ)
  (avg_age_adults : ℕ)
  (num_children : ℕ)
  (num_adults : ℕ)
  (total_num_members : ℕ)
  (total_sum_ages : ℕ)
  (average_age : ℚ)
  (h_children : avg_age_children = 12)
  (h_adults : avg_age_adults = 40)
  (h_num_children : num_children = 8)
  (h_num_adults : num_adults = 12)
  (h_total_members : total_num_members = 20)
  (h_total_ages : total_sum_ages = 576)
  (h_average_age : average_age = 28.8) :
  average_age = (total_sum_ages : ℚ) / total_num_members :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_dance_group_l774_77451


namespace NUMINAMATH_GPT_parabola_coefficients_l774_77443

theorem parabola_coefficients 
  (a b c : ℝ) 
  (h_vertex : ∀ x : ℝ, (2 - (-2))^2 * a + (-2 * 2 * a + b) * (2 - (-2)) + (c - 5) = 0)
  (h_point : 9 = a * (2:ℝ)^2 + b * (2:ℝ) + c) : 
  a = 1 / 4 ∧ b = 1 ∧ c = 6 := 
by 
  sorry

end NUMINAMATH_GPT_parabola_coefficients_l774_77443


namespace NUMINAMATH_GPT_measure_of_angle_l774_77464

theorem measure_of_angle (x : ℝ) (h1 : 90 - x = 3 * x - 10) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_l774_77464


namespace NUMINAMATH_GPT_find_angle_B_l774_77497

theorem find_angle_B (a b c : ℝ) (h : a^2 + c^2 - b^2 = a * c) : 
  ∃ B : ℝ, 0 < B ∧ B < 180 ∧ B = 60 :=
by 
  sorry

end NUMINAMATH_GPT_find_angle_B_l774_77497


namespace NUMINAMATH_GPT_triangle_inequality_l774_77495

variable {α β γ a b c : ℝ}

theorem triangle_inequality (h1: α ≥ β) (h2: β ≥ γ) (h3: a ≥ b) (h4: b ≥ c) (h5: α ≥ γ) (h6: a ≥ c) :
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l774_77495


namespace NUMINAMATH_GPT_find_a1_l774_77487

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {a_n} is a geometric sequence with a common ratio q > 0
axiom geom_seq : (∀ n, a (n + 1) = a n * q)

-- Given conditions of the problem
def condition1 : q > 0 := sorry
def condition2 : a 5 * a 7 = 4 * (a 4) ^ 2 := sorry
def condition3 : a 2 = 1 := sorry

-- Prove that a_1 = sqrt 2 / 2
theorem find_a1 : a 1 = (Real.sqrt 2) / 2 := sorry

end NUMINAMATH_GPT_find_a1_l774_77487


namespace NUMINAMATH_GPT_wednesday_tips_value_l774_77493

-- Definitions for the conditions
def hourly_wage : ℕ := 10
def monday_hours : ℕ := 7
def tuesday_hours : ℕ := 5
def wednesday_hours : ℕ := 7
def monday_tips : ℕ := 18
def tuesday_tips : ℕ := 12
def total_earnings : ℕ := 240

-- Hourly earnings
def monday_earnings := monday_hours * hourly_wage
def tuesday_earnings := tuesday_hours * hourly_wage
def wednesday_earnings := wednesday_hours * hourly_wage

-- Total wage earnings
def total_wage_earnings := monday_earnings + tuesday_earnings + wednesday_earnings

-- Total earnings with known tips
def known_earnings := total_wage_earnings + monday_tips + tuesday_tips

-- Prove that Wednesday tips is $20
theorem wednesday_tips_value : (total_earnings - known_earnings) = 20 := by
  sorry

end NUMINAMATH_GPT_wednesday_tips_value_l774_77493


namespace NUMINAMATH_GPT_percentageReduction_l774_77484

variable (R P : ℝ)

def originalPrice (R : ℝ) (P : ℝ) : Prop :=
  2400 / R - 2400 / P = 8 ∧ R = 120

theorem percentageReduction : 
  originalPrice 120 P → ((P - 120) / P) * 100 = 40 := 
by
  sorry

end NUMINAMATH_GPT_percentageReduction_l774_77484


namespace NUMINAMATH_GPT_minimize_sum_pos_maximize_product_pos_l774_77431

def N : ℕ := 10^1001 - 1

noncomputable def find_min_sum_position : ℕ := 996

noncomputable def find_max_product_position : ℕ := 995

theorem minimize_sum_pos :
  ∀ m : ℕ, (m ≠ find_min_sum_position) → 
      (2 * 10^m + 10^(1001-m) - 10) ≥ (2 * 10^find_min_sum_position + 10^(1001-find_min_sum_position) - 10) := 
sorry

theorem maximize_product_pos :
  ∀ m : ℕ, (m ≠ find_max_product_position) → 
      ((2 * 10^m - 1) * (10^(1001 - m) - 9)) ≤ ((2 * 10^find_max_product_position - 1) * (10^(1001 - find_max_product_position) - 9)) :=
sorry

end NUMINAMATH_GPT_minimize_sum_pos_maximize_product_pos_l774_77431


namespace NUMINAMATH_GPT_find_d_l774_77499

theorem find_d (c : ℝ) (d : ℝ) (h1 : c = 7)
  (h2 : (2, 6) ∈ { p : ℝ × ℝ | ∃ d, (p = (2, 6) ∨ p = (5, c) ∨ p = (d, 0)) ∧
           ∃ m, m = (0 - 6) / (d - 2) ∧ m = (c - 6) / (5 - 2) }) : 
  d = -16 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l774_77499


namespace NUMINAMATH_GPT_incorrect_connection_probability_l774_77456

noncomputable def probability_of_incorrect_connection (p : ℝ) : ℝ :=
  let r2 := 1 / 9
  let r3 := (8 / 9) * (1 / 9)
  (3 * p^2 * (1 - p) * r2) + (1 * p^3 * r3)

theorem incorrect_connection_probability : probability_of_incorrect_connection 0.02 = 0.000131 :=
by
  sorry

end NUMINAMATH_GPT_incorrect_connection_probability_l774_77456


namespace NUMINAMATH_GPT_simplify_expression_l774_77410

theorem simplify_expression (x : ℝ) : 8 * x + 15 - 3 * x + 27 = 5 * x + 42 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l774_77410


namespace NUMINAMATH_GPT_minimum_value_expression_l774_77461

theorem minimum_value_expression {x1 x2 x3 x4 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) (hx4 : 0 < x4) (h_sum : x1 + x2 + x3 + x4 = Real.pi) :
  (2 * (Real.sin x1)^2 + 1 / (Real.sin x1)^2) * (2 * (Real.sin x2)^2 + 1 / (Real.sin x2)^2) * (2 * (Real.sin x3)^2 + 1 / (Real.sin x3)^2) * (2 * (Real.sin x4)^2 + 1 / (Real.sin x4)^2) ≥ 81 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_expression_l774_77461


namespace NUMINAMATH_GPT_total_cost_is_correct_l774_77428

def cost_per_pound : ℝ := 0.45
def weight_sugar : ℝ := 40
def weight_flour : ℝ := 16

theorem total_cost_is_correct :
  weight_sugar * cost_per_pound + weight_flour * cost_per_pound = 25.20 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_is_correct_l774_77428


namespace NUMINAMATH_GPT_percentage_passed_all_three_l774_77458

variable (F_H F_E F_M F_HE F_EM F_HM F_HEM : ℝ)

theorem percentage_passed_all_three :
  F_H = 0.46 →
  F_E = 0.54 →
  F_M = 0.32 →
  F_HE = 0.18 →
  F_EM = 0.12 →
  F_HM = 0.1 →
  F_HEM = 0.06 →
  (100 - (F_H + F_E + F_M - F_HE - F_EM - F_HM + F_HEM)) = 2 :=
by sorry

end NUMINAMATH_GPT_percentage_passed_all_three_l774_77458


namespace NUMINAMATH_GPT_average_weight_proof_l774_77400

variables (W_A W_B W_C W_D W_E : ℝ)

noncomputable def final_average_weight (W_A W_B W_C W_D W_E : ℝ) : ℝ := (W_B + W_C + W_D + W_E) / 4

theorem average_weight_proof
  (h1 : (W_A + W_B + W_C) / 3 = 84)
  (h2 : W_A = 77)
  (h3 : (W_A + W_B + W_C + W_D) / 4 = 80)
  (h4 : W_E = W_D + 5) :
  final_average_weight W_A W_B W_C W_D W_E = 97.25 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_proof_l774_77400


namespace NUMINAMATH_GPT_jerrie_minutes_l774_77442

-- Define the conditions
def barney_situps_per_minute := 45
def carrie_situps_per_minute := 2 * barney_situps_per_minute
def jerrie_situps_per_minute := carrie_situps_per_minute + 5
def barney_total_situps := 1 * barney_situps_per_minute
def carrie_total_situps := 2 * carrie_situps_per_minute
def combined_total_situps := 510

-- Define the question and required proof
theorem jerrie_minutes :
  ∃ J : ℕ, barney_total_situps + carrie_total_situps + J * jerrie_situps_per_minute = combined_total_situps ∧ J = 3 :=
  by
  sorry

end NUMINAMATH_GPT_jerrie_minutes_l774_77442


namespace NUMINAMATH_GPT_magnitude_squared_complex_l774_77436

noncomputable def complex_number := Complex.mk 3 (-4)
noncomputable def squared_complex := complex_number * complex_number

theorem magnitude_squared_complex : Complex.abs squared_complex = 25 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_squared_complex_l774_77436


namespace NUMINAMATH_GPT_inverse_value_exists_l774_77466

noncomputable def f (a x : ℝ) := a^x - 1

theorem inverse_value_exists (a : ℝ) (h : f a 1 = 1) : (f a)⁻¹ 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_inverse_value_exists_l774_77466


namespace NUMINAMATH_GPT_items_left_in_store_l774_77452

def restocked : ℕ := 4458
def sold : ℕ := 1561
def storeroom : ℕ := 575

theorem items_left_in_store : restocked - sold + storeroom = 3472 := by
  sorry

end NUMINAMATH_GPT_items_left_in_store_l774_77452


namespace NUMINAMATH_GPT_rook_attack_expectation_correct_l774_77425

open ProbabilityTheory

noncomputable def rook_attack_expectation : ℝ := sorry

theorem rook_attack_expectation_correct :
  rook_attack_expectation = 35.33 := sorry

end NUMINAMATH_GPT_rook_attack_expectation_correct_l774_77425


namespace NUMINAMATH_GPT_solve_for_t_l774_77420

theorem solve_for_t : ∃ t : ℝ, 3 * 3^t + Real.sqrt (9 * 9^t) = 18 ∧ t = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_t_l774_77420


namespace NUMINAMATH_GPT_average_of_11_results_l774_77412

theorem average_of_11_results (a b c : ℝ) (avg_first_6 avg_last_6 sixth_result avg_all_11 : ℝ)
  (h1 : avg_first_6 = 58)
  (h2 : avg_last_6 = 63)
  (h3 : sixth_result = 66) :
  avg_all_11 = 60 :=
by
  sorry

end NUMINAMATH_GPT_average_of_11_results_l774_77412


namespace NUMINAMATH_GPT_find_second_term_of_ratio_l774_77438

theorem find_second_term_of_ratio
  (a b c d : ℕ)
  (h1 : a = 6)
  (h2 : b = 7)
  (h3 : c = 3)
  (h4 : (a - c) * 4 < a * d) :
  d = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_second_term_of_ratio_l774_77438


namespace NUMINAMATH_GPT_solve_for_x_l774_77455

theorem solve_for_x (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l774_77455


namespace NUMINAMATH_GPT_cosine_identity_l774_77448

theorem cosine_identity (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 4) : 
  Real.cos (2 * α - Real.pi / 3) = 7 / 8 := by 
  sorry

end NUMINAMATH_GPT_cosine_identity_l774_77448


namespace NUMINAMATH_GPT_relationship_cannot_be_determined_l774_77465

noncomputable def point_on_parabola (a b c x y : ℝ) : Prop :=
  y = a * x^2 + b * x + c

theorem relationship_cannot_be_determined
  (a b c x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) (h1 : a ≠ 0) 
  (h2 : point_on_parabola a b c x1 y1) 
  (h3 : point_on_parabola a b c x2 y2) 
  (h4 : point_on_parabola a b c x3 y3) 
  (h5 : point_on_parabola a b c x4 y4)
  (h6 : x1 + x4 - x2 + x3 = 0) : 
  ¬( ∃ m n : ℝ, ((y4 - y1) / (x4 - x1) = m ∧ (y2 - y3) / (x2 - x3) = m) ∨ 
                     ((y4 - y1) / (x4 - x1) * (y2 - y3) / (x2 - x3) = -1) ∨ 
                     ((y4 - y1) / (x4 - x1) ≠ m ∧ (y2 - y3) / (x2 - x3) ≠ m ∧ 
                      (y4 - y1) / (x4 - x1) * (y2 - y3) / (x2 - x3) ≠ -1)) :=
sorry

end NUMINAMATH_GPT_relationship_cannot_be_determined_l774_77465


namespace NUMINAMATH_GPT_n_squared_plus_3n_is_perfect_square_iff_l774_77496

theorem n_squared_plus_3n_is_perfect_square_iff (n : ℕ) : 
  ∃ k : ℕ, n^2 + 3 * n = k^2 ↔ n = 1 :=
by 
  sorry

end NUMINAMATH_GPT_n_squared_plus_3n_is_perfect_square_iff_l774_77496


namespace NUMINAMATH_GPT_noah_small_paintings_sold_last_month_l774_77489

theorem noah_small_paintings_sold_last_month
  (large_painting_price small_painting_price : ℕ)
  (large_paintings_sold_last_month : ℕ)
  (total_sales_this_month : ℕ)
  (sale_multiplier : ℕ)
  (x : ℕ)
  (h1 : large_painting_price = 60)
  (h2 : small_painting_price = 30)
  (h3 : large_paintings_sold_last_month = 8)
  (h4 : total_sales_this_month = 1200)
  (h5 : sale_multiplier = 2) :
  (2 * ((large_paintings_sold_last_month * large_painting_price) + (x * small_painting_price)) = total_sales_this_month) → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_noah_small_paintings_sold_last_month_l774_77489


namespace NUMINAMATH_GPT_science_books_have_9_copies_l774_77435

theorem science_books_have_9_copies :
  ∃ (A B C D : ℕ), A + B + C + D = 35 ∧ A + B = 17 ∧ B + C = 16 ∧ A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ B = 9 :=
by
  sorry

end NUMINAMATH_GPT_science_books_have_9_copies_l774_77435


namespace NUMINAMATH_GPT_number_of_recipes_l774_77478

-- Let's define the necessary conditions.
def cups_per_recipe : ℕ := 2
def total_cups_needed : ℕ := 46

-- Prove that the number of recipes required is 23.
theorem number_of_recipes : total_cups_needed / cups_per_recipe = 23 :=
by
  sorry

end NUMINAMATH_GPT_number_of_recipes_l774_77478


namespace NUMINAMATH_GPT_Sara_house_size_l774_77472

theorem Sara_house_size (nada_size : ℕ) (h1 : nada_size = 450) (h2 : Sara_size = 2 * nada_size + 100) : Sara_size = 1000 :=
by sorry

end NUMINAMATH_GPT_Sara_house_size_l774_77472


namespace NUMINAMATH_GPT_range_of_a_l774_77422

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 0 → a * 4^x - 2^x + 2 > 0) → a > -1 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l774_77422


namespace NUMINAMATH_GPT_circumscribed_circle_radius_l774_77457

noncomputable def radius_of_circumscribed_circle (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 + b^2)) / 2

theorem circumscribed_circle_radius (a r l b R : ℝ)
  (h1 : r = 1)
  (h2 : a = 2 * Real.sqrt 3)
  (h3 : b = 3)
  (h4 : l = a)
  (h5 : R = radius_of_circumscribed_circle l b) :
  R = Real.sqrt 21 / 2 :=
by
  sorry

end NUMINAMATH_GPT_circumscribed_circle_radius_l774_77457


namespace NUMINAMATH_GPT_number_of_red_balls_l774_77416

theorem number_of_red_balls (total_balls : ℕ) (prob_red : ℚ) (h : total_balls = 20 ∧ prob_red = 0.25) : ∃ x : ℕ, x = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_red_balls_l774_77416


namespace NUMINAMATH_GPT_inequality_solution_l774_77411

theorem inequality_solution (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (x ∈ Set.Ioo (-2 : ℝ) (-1) ∨ x ∈ Set.Ioi 2) ↔ 
  (∃ x : ℝ, (x^2 + x - 2) / (x + 2) ≥ (3 / (x - 2)) + (3 / 2)) := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l774_77411


namespace NUMINAMATH_GPT_no_nontrivial_integer_solutions_l774_77449

theorem no_nontrivial_integer_solutions (x y z : ℤ) : x^3 + 2*y^3 + 4*z^3 - 6*x*y*z = 0 -> x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_nontrivial_integer_solutions_l774_77449


namespace NUMINAMATH_GPT_probability_exactly_two_singers_same_province_l774_77475

-- Defining the number of provinces and number of singers per province
def num_provinces : ℕ := 6
def singers_per_province : ℕ := 2

-- Total number of singers
def num_singers : ℕ := num_provinces * singers_per_province

-- Define the total number of ways to choose 4 winners from 12 contestants
def total_combinations : ℕ := Nat.choose num_singers 4

-- Define the number of favorable ways to select exactly two singers from the same province and two from two other provinces
def favorable_combinations : ℕ := 
  (Nat.choose num_provinces 1) *  -- Choose one province for the pair
  (Nat.choose (num_provinces - 1) 2) *  -- Choose two remaining provinces
  (Nat.choose singers_per_province 1) *
  (Nat.choose singers_per_province 1)

-- Calculate the probability
def probability : ℚ := favorable_combinations / total_combinations

-- Stating the theorem to be proved
theorem probability_exactly_two_singers_same_province : probability = 16 / 33 :=
by
  sorry

end NUMINAMATH_GPT_probability_exactly_two_singers_same_province_l774_77475


namespace NUMINAMATH_GPT_scientific_notation_of_274000000_l774_77491

theorem scientific_notation_of_274000000 :
  (274000000 : ℝ) = 2.74 * 10 ^ 8 :=
by
    sorry

end NUMINAMATH_GPT_scientific_notation_of_274000000_l774_77491


namespace NUMINAMATH_GPT_exists_k_l774_77404

-- Define P as a non-constant homogeneous polynomial with real coefficients
def homogeneous_polynomial (n : ℕ) (P : ℝ → ℝ → ℝ) :=
  ∀ (a b : ℝ), P (a * a) (b * b) = (a * a) ^ n * (b * b) ^ n

-- Define the main problem
theorem exists_k (P : ℝ → ℝ → ℝ) (hP : ∃ n : ℕ, homogeneous_polynomial n P)
  (h : ∀ t : ℝ, P (Real.sin t) (Real.cos t) = 1) :
  ∃ k : ℕ, ∀ x y : ℝ, P x y = (x^2 + y^2) ^ k :=
sorry

end NUMINAMATH_GPT_exists_k_l774_77404


namespace NUMINAMATH_GPT_election_total_valid_votes_l774_77469

theorem election_total_valid_votes (V B : ℝ) 
    (hA : 0.45 * V = B * V + 250) 
    (hB : 2.5 * B = 62.5) :
    V = 1250 :=
by
  sorry

end NUMINAMATH_GPT_election_total_valid_votes_l774_77469


namespace NUMINAMATH_GPT_total_money_l774_77471

theorem total_money 
  (n_pennies n_nickels n_dimes n_quarters n_half_dollars : ℝ) 
  (h_pennies : n_pennies = 9) 
  (h_nickels : n_nickels = 4) 
  (h_dimes : n_dimes = 3) 
  (h_quarters : n_quarters = 7) 
  (h_half_dollars : n_half_dollars = 5) : 
  0.01 * n_pennies + 0.05 * n_nickels + 0.10 * n_dimes + 0.25 * n_quarters + 0.50 * n_half_dollars = 4.84 :=
by 
  sorry

end NUMINAMATH_GPT_total_money_l774_77471


namespace NUMINAMATH_GPT_monthly_price_reduction_rate_l774_77434

-- Let's define the given conditions
def initial_price_March : ℝ := 23000
def price_in_May : ℝ := 16000

-- Define the monthly average price reduction rate
variable (x : ℝ)

-- Define the statement to be proven
theorem monthly_price_reduction_rate :
  23 * (1 - x) ^ 2 = 16 :=
sorry

end NUMINAMATH_GPT_monthly_price_reduction_rate_l774_77434


namespace NUMINAMATH_GPT_distance_sum_identity_l774_77414

noncomputable def squared_distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem distance_sum_identity
  (a b c x y : ℝ)
  (A B C P G : ℝ × ℝ)
  (hA : A = (a, b))
  (hB : B = (-c, 0))
  (hC : C = (c, 0))
  (hG : G = (a / 3, b / 3))
  (hP : P = (x, y))
  (hG_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) :
  squared_distance A P + squared_distance B P + squared_distance C P =
  squared_distance A G + squared_distance B G + squared_distance C G + 3 * squared_distance G P :=
by sorry

end NUMINAMATH_GPT_distance_sum_identity_l774_77414


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l774_77437

theorem simplify_and_evaluate_expression (a b : ℝ) (h1 : a = -1) (h2 : a * b = 2) :
  3 * (2 * a^2 * b + a * b^2) - (3 * a * b^2 - a^2 * b) = -14 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l774_77437


namespace NUMINAMATH_GPT_count_ways_to_choose_and_discard_l774_77419

theorem count_ways_to_choose_and_discard :
  let suits := 4 
  let cards_per_suit := 13
  let ways_to_choose_4_different_suits := Nat.choose 4 4
  let ways_to_choose_4_cards := cards_per_suit ^ 4
  let ways_to_discard_1_card := 4
  1 * ways_to_choose_4_cards * ways_to_discard_1_card = 114244 :=
by
  sorry

end NUMINAMATH_GPT_count_ways_to_choose_and_discard_l774_77419


namespace NUMINAMATH_GPT_negation_exists_l774_77418

theorem negation_exists (h : ∀ x : ℝ, 0 < x → Real.sin x < x) : ∃ x : ℝ, 0 < x ∧ Real.sin x ≥ x :=
by
  sorry

end NUMINAMATH_GPT_negation_exists_l774_77418


namespace NUMINAMATH_GPT_line_through_A_area_1_l774_77467

def line_equation : Prop :=
  ∃ k : ℚ, ∀ x y : ℚ, (y = k * (x + 2) + 2) ↔ 
    (x + 2 * y - 2 = 0 ∨ 2 * x + y + 2 = 0) ∧ 
    (2 * (k * 0 + 2) * (-2 - 2 / k) = 2)

theorem line_through_A_area_1 : line_equation :=
by
  sorry

end NUMINAMATH_GPT_line_through_A_area_1_l774_77467


namespace NUMINAMATH_GPT_largest_four_digit_perfect_cube_is_9261_l774_77423

-- Define the notion of a four-digit number and perfect cube
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

-- The main theorem statement
theorem largest_four_digit_perfect_cube_is_9261 :
  ∃ n, is_four_digit n ∧ is_perfect_cube n ∧ (∀ m, is_four_digit m ∧ is_perfect_cube m → m ≤ n) ∧ n = 9261 :=
sorry -- Proof is omitted

end NUMINAMATH_GPT_largest_four_digit_perfect_cube_is_9261_l774_77423


namespace NUMINAMATH_GPT_min_value_of_sum_l774_77480

theorem min_value_of_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 1 / x + 1 / y + 1 / z = 1) :
  x + 4 * y + 9 * z ≥ 36 ∧ (x + 4 * y + 9 * z = 36 ↔ x = 6 ∧ y = 3 ∧ z = 2) := 
sorry

end NUMINAMATH_GPT_min_value_of_sum_l774_77480


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_thirty_l774_77417

-- Definitions according to the conditions
def arithmetic_seq_sums (S : ℕ → ℤ) : Prop :=
  ∃ a d : ℤ, ∀ n : ℕ, S n = a + n * d

-- Main statement we need to prove
theorem arithmetic_sequence_sum_thirty (S : ℕ → ℤ)
  (h1 : S 10 = 10)
  (h2 : S 20 = 30)
  (h3 : arithmetic_seq_sums S) : 
  S 30 = 50 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_thirty_l774_77417


namespace NUMINAMATH_GPT_scientific_notation_560000_l774_77407

theorem scientific_notation_560000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 560000 = a * 10 ^ n ∧ a = 5.6 ∧ n = 5 :=
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_560000_l774_77407


namespace NUMINAMATH_GPT_other_discount_l774_77405

theorem other_discount (list_price : ℝ) (final_price : ℝ) (first_discount : ℝ) (other_discount : ℝ) :
  list_price = 70 → final_price = 61.74 → first_discount = 10 → (list_price * (1 - first_discount / 100) * (1 - other_discount / 100) = final_price) → other_discount = 2 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_other_discount_l774_77405


namespace NUMINAMATH_GPT_distance_to_conference_l774_77498

theorem distance_to_conference (t d : ℝ) 
  (h1 : d = 40 * (t + 0.75))
  (h2 : d - 40 = 60 * (t - 1.25)) :
  d = 160 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_conference_l774_77498


namespace NUMINAMATH_GPT_alpha_when_beta_neg4_l774_77406

theorem alpha_when_beta_neg4 :
  (∀ (α β : ℝ), (β ≠ 0) → α = 5 → β = 2 → α * β^2 = α * 4) →
   ∃ (α : ℝ), α = 5 → ∃ β, β = -4 → α = 5 / 4 :=
  by
    intros h
    use 5 / 4
    sorry

end NUMINAMATH_GPT_alpha_when_beta_neg4_l774_77406


namespace NUMINAMATH_GPT_pipe_fill_time_with_leak_l774_77488

theorem pipe_fill_time_with_leak (A L : ℝ) (hA : A = 1 / 2) (hL : L = 1 / 6) :
  (1 / (A - L)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_pipe_fill_time_with_leak_l774_77488


namespace NUMINAMATH_GPT_leah_coins_value_l774_77415

theorem leah_coins_value
  (p n : ℕ)
  (h₁ : n + p = 15)
  (h₂ : n + 2 = p) : p + 5 * n = 38 :=
by
  -- definitions used in converting conditions
  sorry

end NUMINAMATH_GPT_leah_coins_value_l774_77415


namespace NUMINAMATH_GPT_area_of_pentagon_AEDCB_l774_77432

structure Rectangle (A B C D : Type) :=
  (AB BC AD CD : ℕ)

def is_perpendicular (A E E' D : Type) : Prop := sorry

def area_of_triangle (AE DE : ℕ) : ℕ :=
  (AE * DE) / 2

def area_of_rectangle (length width : ℕ) : ℕ :=
  length * width

def area_of_pentagon (area_rect area_triangle : ℕ) : ℕ :=
  area_rect - area_triangle

theorem area_of_pentagon_AEDCB
  (A B C D E : Type)
  (h_rectangle : Rectangle A B C D)
  (h_perpendicular : is_perpendicular A E E D)
  (AE DE : ℕ)
  (h_ae : AE = 9)
  (h_de : DE = 12)
  : area_of_pentagon (area_of_rectangle 15 12) (area_of_triangle AE DE) = 126 := 
  sorry

end NUMINAMATH_GPT_area_of_pentagon_AEDCB_l774_77432


namespace NUMINAMATH_GPT_percent_decrease_l774_77447

theorem percent_decrease (p_original p_sale : ℝ) (h₁ : p_original = 100) (h₂ : p_sale = 50) :
  ((p_original - p_sale) / p_original * 100) = 50 := by
  sorry

end NUMINAMATH_GPT_percent_decrease_l774_77447


namespace NUMINAMATH_GPT_area_of_sector_l774_77444

theorem area_of_sector (r : ℝ) (θ : ℝ) (h1 : r = 10) (h2 : θ = π / 5) : 
  (1 / 2) * r * r * θ = 10 * π :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_area_of_sector_l774_77444


namespace NUMINAMATH_GPT_train_speed_is_correct_l774_77421

-- Definitions of the problem
def length_of_train : ℕ := 360
def time_to_pass_bridge : ℕ := 25
def length_of_bridge : ℕ := 140
def conversion_factor : ℝ := 3.6

-- Distance covered by the train plus the length of the bridge
def total_distance : ℕ := length_of_train + length_of_bridge

-- Speed calculation in m/s
def speed_in_m_per_s := total_distance / time_to_pass_bridge

-- Conversion to km/h
def speed_in_km_per_h := speed_in_m_per_s * conversion_factor

-- The proof goal: the speed of the train is 72 km/h
theorem train_speed_is_correct : speed_in_km_per_h = 72 := by
  sorry

end NUMINAMATH_GPT_train_speed_is_correct_l774_77421


namespace NUMINAMATH_GPT_smallest_p_l774_77439

theorem smallest_p 
  (p q : ℕ) 
  (h1 : (5 : ℚ) / 8 < p / (q : ℚ) ∧ p / (q : ℚ) < 7 / 8)
  (h2 : p + q = 2005) : p = 772 :=
sorry

end NUMINAMATH_GPT_smallest_p_l774_77439


namespace NUMINAMATH_GPT_car_mileage_city_l774_77494

theorem car_mileage_city (h c t : ℕ) 
  (h_eq_tank_mileage : 462 = h * t) 
  (c_eq_tank_mileage : 336 = c * t) 
  (mileage_diff : c = h - 3) : 
  c = 8 := 
by
  sorry

end NUMINAMATH_GPT_car_mileage_city_l774_77494


namespace NUMINAMATH_GPT_range_of_a_l774_77408

def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2) : a ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l774_77408


namespace NUMINAMATH_GPT_evaluateExpression_correct_l774_77440

open Real

noncomputable def evaluateExpression : ℝ :=
  (-2)^2 + 2 * sin (π / 3) - tan (π / 3)

theorem evaluateExpression_correct : evaluateExpression = 4 :=
  sorry

end NUMINAMATH_GPT_evaluateExpression_correct_l774_77440


namespace NUMINAMATH_GPT_max_product_of_sum_300_l774_77402

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end NUMINAMATH_GPT_max_product_of_sum_300_l774_77402


namespace NUMINAMATH_GPT_greatest_percentage_increase_l774_77476

def pop1970_F := 30000
def pop1980_F := 45000
def pop1970_G := 60000
def pop1980_G := 75000
def pop1970_H := 40000
def pop1970_I := 20000
def pop1980_combined_H := 70000
def pop1970_J := 90000
def pop1980_J := 120000

def percentage_increase (pop1970 pop1980 : ℕ) : ℚ :=
  ((pop1980 - pop1970 : ℚ) / pop1970) * 100

theorem greatest_percentage_increase :
  ∀ (city : ℕ), (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase pop1970_G pop1980_G) ∧
                (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase (pop1970_H + pop1970_I) pop1980_combined_H) ∧
                (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase pop1970_J pop1980_J) := by 
  sorry

end NUMINAMATH_GPT_greatest_percentage_increase_l774_77476


namespace NUMINAMATH_GPT_find_value_of_m_l774_77453

theorem find_value_of_m (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 4*x + 2*y + m = 0 ∧ (x - 2)^2 + (y + 1)^2 = 4) →
  m = 1 :=
sorry

end NUMINAMATH_GPT_find_value_of_m_l774_77453


namespace NUMINAMATH_GPT_initial_acidic_liquid_quantity_l774_77479

theorem initial_acidic_liquid_quantity
  (A : ℝ) -- initial quantity of the acidic liquid in liters
  (W : ℝ) -- quantity of water to be removed in liters
  (h1 : W = 6)
  (h2 : (0.40 * A) = 0.60 * (A - W)) : 
  A = 18 :=
by sorry

end NUMINAMATH_GPT_initial_acidic_liquid_quantity_l774_77479


namespace NUMINAMATH_GPT_parallel_vectors_l774_77485

theorem parallel_vectors (m : ℝ) :
  let a : (ℝ × ℝ × ℝ) := (2, -1, 2)
  let b : (ℝ × ℝ × ℝ) := (-4, 2, m)
  (∀ k : ℝ, a = (k * -4, k * 2, k * m)) →
  m = -4 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_l774_77485


namespace NUMINAMATH_GPT_add_fractions_l774_77486

theorem add_fractions : (2 / 3 : ℚ) + (7 / 8) = 37 / 24 := 
by sorry

end NUMINAMATH_GPT_add_fractions_l774_77486


namespace NUMINAMATH_GPT_xy_product_range_l774_77473

theorem xy_product_range (x y : ℝ) (h : x^2 * y^2 + x^2 - 10 * x * y - 8 * x + 16 = 0) :
  0 ≤ x * y ∧ x * y ≤ 10 := 
sorry

end NUMINAMATH_GPT_xy_product_range_l774_77473


namespace NUMINAMATH_GPT_valid_triplets_l774_77403

theorem valid_triplets (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_leq1 : a ≤ b) (h_leq2 : b ≤ c)
  (h_div1 : a ∣ (b + c)) (h_div2 : b ∣ (a + c)) (h_div3 : c ∣ (a + b)) :
  (a = b ∧ b = c) ∨ (a = b ∧ c = 2 * a) ∨ (b = 2 * a ∧ c = 3 * a) :=
sorry

end NUMINAMATH_GPT_valid_triplets_l774_77403


namespace NUMINAMATH_GPT_max_area_and_length_l774_77441

def material_cost (x y : ℝ) : ℝ :=
  900 * x + 400 * y + 200 * x * y

def area (x y : ℝ) : ℝ := x * y

theorem max_area_and_length (x y : ℝ) (h₁ : material_cost x y ≤ 32000) :
  ∃ (S : ℝ) (x : ℝ), S = 100 ∧ x = 20 / 3 :=
sorry

end NUMINAMATH_GPT_max_area_and_length_l774_77441


namespace NUMINAMATH_GPT_single_bill_value_l774_77409

theorem single_bill_value 
  (total_amount : ℕ) 
  (num_5_dollar_bills : ℕ) 
  (amount_5_dollar_bills : ℕ) 
  (single_bill : ℕ) : 
  total_amount = 45 → 
  num_5_dollar_bills = 7 → 
  amount_5_dollar_bills = 5 → 
  total_amount = num_5_dollar_bills * amount_5_dollar_bills + single_bill → 
  single_bill = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_single_bill_value_l774_77409


namespace NUMINAMATH_GPT_number_of_attempted_problems_l774_77470

-- Lean statement to define the problem setup
def student_assignment_problem (x y : ℕ) : Prop :=
  8 * x - 5 * y = 13 ∧ x + y ≤ 20

-- The Lean statement asserting the solution to the problem
theorem number_of_attempted_problems : ∃ x y : ℕ, student_assignment_problem x y ∧ x + y = 13 := 
by
  sorry

end NUMINAMATH_GPT_number_of_attempted_problems_l774_77470


namespace NUMINAMATH_GPT_totalSandwiches_l774_77450

def numberOfPeople : ℝ := 219.0
def sandwichesPerPerson : ℝ := 3.0

theorem totalSandwiches : numberOfPeople * sandwichesPerPerson = 657.0 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_totalSandwiches_l774_77450


namespace NUMINAMATH_GPT_central_angle_of_regular_hexagon_l774_77474

theorem central_angle_of_regular_hexagon:
  ∀ (α : ℝ), 
  (∃ n : ℕ, n = 6 ∧ n * α = 360) →
  α = 60 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_regular_hexagon_l774_77474


namespace NUMINAMATH_GPT_number_of_boxes_initially_l774_77413

theorem number_of_boxes_initially (B : ℕ) (h1 : ∃ B, 8 * B - 17 = 15) : B = 4 :=
  by
  sorry

end NUMINAMATH_GPT_number_of_boxes_initially_l774_77413


namespace NUMINAMATH_GPT_recurring_decimal_mul_seven_l774_77454

-- Declare the repeating decimal as a definition
def recurring_decimal_0_3 : ℚ := 1 / 3

-- Theorem stating that the product of 0.333... and 7 is 7/3
theorem recurring_decimal_mul_seven : recurring_decimal_0_3 * 7 = 7 / 3 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_recurring_decimal_mul_seven_l774_77454


namespace NUMINAMATH_GPT_decreasing_intervals_tangent_line_eq_l774_77490

-- Define the function f and its derivative.
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + 1
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Part 1: Prove intervals of monotonic decreasing.
theorem decreasing_intervals :
  (∀ x, f' x < 0 → x < -1 ∨ x > 3) := 
sorry

-- Part 2: Prove the tangent line equation.
theorem tangent_line_eq :
  15 * (-2) + (-13) + 27 = 0 :=
sorry

end NUMINAMATH_GPT_decreasing_intervals_tangent_line_eq_l774_77490


namespace NUMINAMATH_GPT_hank_donated_percentage_l774_77430

variable (A_c D_c A_b D_b A_l D_t D_l p : ℝ) (h1 : A_c = 100) (h2 : D_c = 0.90 * A_c)
variable (h3 : A_b = 80) (h4 : D_b = 0.75 * A_b) (h5 : A_l = 50) (h6 : D_t = 200)

theorem hank_donated_percentage :
  D_l = D_t - (D_c + D_b) → 
  p = (D_l / A_l) * 100 → 
  p = 100 :=
by
  sorry

end NUMINAMATH_GPT_hank_donated_percentage_l774_77430


namespace NUMINAMATH_GPT_circle_placement_possible_l774_77445

theorem circle_placement_possible
  (length : ℕ)
  (width : ℕ)
  (n : ℕ)
  (area_ci : ℕ)
  (ne_int_lt : length = 20)
  (ne_wid_lt : width = 25)
  (ne_squares : n = 120)
  (sm_area_lt : area_ci = 456) :
  120 * (1 + (Real.pi / 4)) < area_ci :=
by sorry

end NUMINAMATH_GPT_circle_placement_possible_l774_77445


namespace NUMINAMATH_GPT_mn_minus_n_values_l774_77483

theorem mn_minus_n_values (m n : ℝ) (h1 : |m| = 4) (h2 : |n| = 2.5) (h3 : m * n < 0) :
  m * n - n = -7.5 ∨ m * n - n = -12.5 :=
sorry

end NUMINAMATH_GPT_mn_minus_n_values_l774_77483
