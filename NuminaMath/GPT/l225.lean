import Mathlib

namespace NUMINAMATH_GPT_part_a_part_b_l225_22545

-- Assuming existence of function S satisfying certain properties
variable (S : Type → Type → Type → ℝ)

-- Part (a)
theorem part_a (A B C : Type) : 
  S A B C = -S B A C ∧ S A B C = S B C A :=
sorry

-- Part (b)
theorem part_b (A B C D : Type) : 
  S A B C = S D A B + S D B C + S D C A :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l225_22545


namespace NUMINAMATH_GPT_xiao_ming_correctly_answered_question_count_l225_22564

-- Define the given conditions as constants and variables
def total_questions : ℕ := 20
def points_per_correct : ℕ := 8
def points_deducted_per_incorrect : ℕ := 5
def total_score : ℕ := 134

-- Prove that the number of correctly answered questions is 18
theorem xiao_ming_correctly_answered_question_count :
  ∃ (correct_count incorrect_count : ℕ), 
      correct_count + incorrect_count = total_questions ∧
      correct_count * points_per_correct - 
      incorrect_count * points_deducted_per_incorrect = total_score ∧
      correct_count = 18 :=
by
  sorry

end NUMINAMATH_GPT_xiao_ming_correctly_answered_question_count_l225_22564


namespace NUMINAMATH_GPT_perimeter_difference_l225_22581

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end NUMINAMATH_GPT_perimeter_difference_l225_22581


namespace NUMINAMATH_GPT_value_of_x_l225_22568

theorem value_of_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l225_22568


namespace NUMINAMATH_GPT_Karlsson_eats_more_than_half_l225_22575

open Real

theorem Karlsson_eats_more_than_half
  (D : ℝ) (S : ℕ → ℝ)
  (a b : ℕ → ℝ)
  (cut_and_eat : ∀ n, S (n + 1) = S n - (S n * a n) / (a n + b n))
  (side_conditions : ∀ n, max (a n) (b n) ≤ D) :
  ∃ n, S n < (S 0) / 2 := sorry

end NUMINAMATH_GPT_Karlsson_eats_more_than_half_l225_22575


namespace NUMINAMATH_GPT_height_ratio_l225_22571

noncomputable def Anne_height := 80
noncomputable def Bella_height := 3 * Anne_height
noncomputable def Sister_height := Bella_height - 200

theorem height_ratio : Anne_height / Sister_height = 2 :=
by
  /-
  The proof here is omitted as requested.
  -/
  sorry

end NUMINAMATH_GPT_height_ratio_l225_22571


namespace NUMINAMATH_GPT_students_spend_185_minutes_in_timeout_l225_22547

variable (tR tF tS t_total : ℕ)

-- Conditions
def running_timeouts : ℕ := 5
def food_timeouts : ℕ := 5 * running_timeouts - 1
def swearing_timeouts : ℕ := food_timeouts / 3
def total_timeouts : ℕ := running_timeouts + food_timeouts + swearing_timeouts
def timeout_duration : ℕ := 5

-- Total time spent in time-out
def total_timeout_minutes : ℕ := total_timeouts * timeout_duration

theorem students_spend_185_minutes_in_timeout :
  total_timeout_minutes = 185 :=
by
  -- The answer is directly given by the conditions and the correct answer identified.
  sorry

end NUMINAMATH_GPT_students_spend_185_minutes_in_timeout_l225_22547


namespace NUMINAMATH_GPT_coloring_equilateral_triangle_l225_22520

theorem coloring_equilateral_triangle :
  ∀ (A B C : Type) (color : A → Type) (d : A → A → ℝ),
  (∀ x y, d x y = 1 → color x = color y) :=
by sorry

end NUMINAMATH_GPT_coloring_equilateral_triangle_l225_22520


namespace NUMINAMATH_GPT_time_difference_is_16_point_5_l225_22541

noncomputable def time_difference : ℝ :=
  let danny_to_steve : ℝ := 33
  let steve_to_danny := 2 * danny_to_steve -- Steve takes twice the time as Danny
  let emma_to_houses : ℝ := 40
  let danny_halfway := danny_to_steve / 2 -- Halfway point for Danny
  let steve_halfway := steve_to_danny / 2 -- Halfway point for Steve
  let emma_halfway := emma_to_houses / 2 -- Halfway point for Emma
  -- Additional times to the halfway point
  let steve_additional := steve_halfway - danny_halfway
  let emma_additional := emma_halfway - danny_halfway
  -- The final result is the maximum of these times
  max steve_additional emma_additional

theorem time_difference_is_16_point_5 : time_difference = 16.5 :=
  by
  sorry

end NUMINAMATH_GPT_time_difference_is_16_point_5_l225_22541


namespace NUMINAMATH_GPT_log_function_passes_through_point_l225_22515

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (x - 1) / Real.log a - 1

theorem log_function_passes_through_point {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = -1 :=
by
  -- To complete the proof, one would argue about the properties of logarithms in specific bases.
  sorry

end NUMINAMATH_GPT_log_function_passes_through_point_l225_22515


namespace NUMINAMATH_GPT_snow_at_least_once_prob_l225_22502

-- Define the conditions for the problem
def prob_snow_day1_to_day4 : ℚ := 1 / 2
def prob_no_snow_day1_to_day4 : ℚ := 1 - prob_snow_day1_to_day4

def prob_snow_day5_to_day7 : ℚ := 1 / 3
def prob_no_snow_day5_to_day7 : ℚ := 1 - prob_snow_day5_to_day7

-- Define the probability of no snow during the first week of February
def prob_no_snow_week : ℚ := (prob_no_snow_day1_to_day4 ^ 4) * (prob_no_snow_day5_to_day7 ^ 3)

-- Define the probability that it snows at least once during the first week of February
def prob_snow_at_least_once : ℚ := 1 - prob_no_snow_week

-- The theorem we want to prove
theorem snow_at_least_once_prob : prob_snow_at_least_once = 53 / 54 :=
by
  sorry

end NUMINAMATH_GPT_snow_at_least_once_prob_l225_22502


namespace NUMINAMATH_GPT_quadratic_root_form_eq_l225_22500

theorem quadratic_root_form_eq (c : ℚ) : 
  (∀ x : ℚ, x^2 - 7 * x + c = 0 → x = (7 + Real.sqrt (9 * c)) / 2 ∨ x = (7 - Real.sqrt (9 * c)) / 2) →
  c = 49 / 13 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_root_form_eq_l225_22500


namespace NUMINAMATH_GPT_calculate_sum_of_powers_l225_22578

theorem calculate_sum_of_powers :
  (6^2 - 3^2)^4 + (7^2 - 2^2)^4 = 4632066 :=
by
  sorry

end NUMINAMATH_GPT_calculate_sum_of_powers_l225_22578


namespace NUMINAMATH_GPT_employee_salary_percentage_l225_22588

theorem employee_salary_percentage (A B : ℝ)
    (h1 : A + B = 450)
    (h2 : B = 180) : (A / B) * 100 = 150 := by
  sorry

end NUMINAMATH_GPT_employee_salary_percentage_l225_22588


namespace NUMINAMATH_GPT_bus_travel_fraction_l225_22559

theorem bus_travel_fraction :
  ∃ D : ℝ, D = 30.000000000000007 ∧
            (1 / 3) * D + 2 + (18 / 30) * D = D ∧
            (18 / 30) = (3 / 5) :=
by
  sorry

end NUMINAMATH_GPT_bus_travel_fraction_l225_22559


namespace NUMINAMATH_GPT_integral_cos_2x_eq_half_l225_22556

theorem integral_cos_2x_eq_half :
  ∫ x in (0:ℝ)..(Real.pi / 4), Real.cos (2 * x) = 1 / 2 := by
sorry

end NUMINAMATH_GPT_integral_cos_2x_eq_half_l225_22556


namespace NUMINAMATH_GPT_half_dollar_difference_l225_22565

theorem half_dollar_difference (n d h : ℕ) 
  (h1 : n + d + h = 150) 
  (h2 : 5 * n + 10 * d + 50 * h = 1500) : 
  ∃ h_max h_min, (h_max - h_min = 16) :=
by sorry

end NUMINAMATH_GPT_half_dollar_difference_l225_22565


namespace NUMINAMATH_GPT_inequality_half_l225_22598

-- The Lean 4 statement for the given proof problem
theorem inequality_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry


end NUMINAMATH_GPT_inequality_half_l225_22598


namespace NUMINAMATH_GPT_missing_digit_divisibility_l225_22580

theorem missing_digit_divisibility (x : ℕ) (h1 : x < 10) :
  3 ∣ (1 + 3 + 5 + 7 + x + 2) ↔ x = 0 ∨ x = 3 ∨ x = 6 ∨ x = 9 := by
  sorry

end NUMINAMATH_GPT_missing_digit_divisibility_l225_22580


namespace NUMINAMATH_GPT_lcm_105_360_eq_2520_l225_22549

theorem lcm_105_360_eq_2520 :
  Nat.lcm 105 360 = 2520 :=
by
  have h1 : 105 = 3 * 5 * 7 := by norm_num
  have h2 : 360 = 2^3 * 3^2 * 5 := by norm_num
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_lcm_105_360_eq_2520_l225_22549


namespace NUMINAMATH_GPT_ratio_of_plums_to_peaches_is_three_l225_22596

theorem ratio_of_plums_to_peaches_is_three :
  ∃ (L P W : ℕ), W = 1 ∧ P = W + 12 ∧ L = 3 * P ∧ W + P + L = 53 ∧ (L / P) = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_plums_to_peaches_is_three_l225_22596


namespace NUMINAMATH_GPT_roots_greater_than_half_iff_l225_22570

noncomputable def quadratic_roots (a : ℝ) (x1 x2 : ℝ) : Prop :=
  (2 - a) * x1^2 - 3 * a * x1 + 2 * a = 0 ∧ 
  (2 - a) * x2^2 - 3 * a * x2 + 2 * a = 0 ∧
  x1 > 1/2 ∧ x2 > 1/2

theorem roots_greater_than_half_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_roots a x1 x2) ↔ (16 / 17 < a ∧ a < 2) :=
sorry

end NUMINAMATH_GPT_roots_greater_than_half_iff_l225_22570


namespace NUMINAMATH_GPT_train_speed_and_length_l225_22516

theorem train_speed_and_length 
  (x y : ℝ)
  (h1 : 60 * x = 1000 + y)
  (h2 : 40 * x = 1000 - y) :
  x = 20 ∧ y = 200 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_and_length_l225_22516


namespace NUMINAMATH_GPT_lowest_price_l225_22579

theorem lowest_price (cost_per_component shipping_cost_per_unit fixed_costs number_of_components produced_cost total_variable_cost total_cost lowest_price : ℝ)
  (h1 : cost_per_component = 80)
  (h2 : shipping_cost_per_unit = 2)
  (h3 : fixed_costs = 16200)
  (h4 : number_of_components = 150)
  (h5 : total_variable_cost = cost_per_component + shipping_cost_per_unit)
  (h6 : produced_cost = total_variable_cost * number_of_components)
  (h7 : total_cost = produced_cost + fixed_costs)
  (h8 : lowest_price = total_cost / number_of_components) :
  lowest_price = 190 :=
  by
  sorry

end NUMINAMATH_GPT_lowest_price_l225_22579


namespace NUMINAMATH_GPT_exists_identical_coordinates_l225_22544

theorem exists_identical_coordinates
  (O O' : ℝ × ℝ)
  (Ox Oy O'x' O'y' : ℝ → ℝ)
  (units_different : ∃ u v : ℝ, u ≠ v)
  (O_ne_O' : O ≠ O')
  (Ox_not_parallel_O'x' : ∃ θ : ℝ, θ ≠ 0 ∧ θ ≠ π) :
  ∃ S : ℝ × ℝ, (S.1 = Ox S.1 ∧ S.2 = Oy S.2) ∧ (S.1 = O'x' S.1 ∧ S.2 = O'y' S.2) :=
sorry

end NUMINAMATH_GPT_exists_identical_coordinates_l225_22544


namespace NUMINAMATH_GPT_pizza_slices_left_l225_22513

-- Lean definitions for conditions
def total_slices : ℕ := 24
def slices_eaten_dinner : ℕ := total_slices / 3
def slices_after_dinner : ℕ := total_slices - slices_eaten_dinner

def slices_eaten_yves : ℕ := slices_after_dinner / 5
def slices_after_yves : ℕ := slices_after_dinner - slices_eaten_yves

def slices_eaten_oldest_siblings : ℕ := 3 * 3
def slices_after_oldest_siblings : ℕ := slices_after_yves - slices_eaten_oldest_siblings

def num_remaining_siblings : ℕ := 7 - 3
def slices_eaten_remaining_siblings : ℕ := num_remaining_siblings * 2
def slices_final : ℕ := if slices_after_oldest_siblings < slices_eaten_remaining_siblings then 0 else slices_after_oldest_siblings - slices_eaten_remaining_siblings

-- Proposition to prove
theorem pizza_slices_left : slices_final = 0 := by sorry

end NUMINAMATH_GPT_pizza_slices_left_l225_22513


namespace NUMINAMATH_GPT_power_of_two_l225_22522

theorem power_of_two (m n : ℕ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
  (h_prime : Prime (m^(4^n + 1) - 1)) : 
  ∃ t : ℕ, n = 2^t :=
sorry

end NUMINAMATH_GPT_power_of_two_l225_22522


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l225_22518

theorem arithmetic_sequence_property 
  (a : ℕ → ℤ) 
  (h₁ : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l225_22518


namespace NUMINAMATH_GPT_remaining_marbles_l225_22548

theorem remaining_marbles (initial_marbles : ℕ) (num_customers : ℕ) (marble_range : List ℕ)
  (h_initial : initial_marbles = 2500)
  (h_customers : num_customers = 50)
  (h_range : marble_range = List.range' 1 50)
  (disjoint_range : ∀ (a b : ℕ), a ∈ marble_range → b ∈ marble_range → a ≠ b → a + b ≤ 50) :
  initial_marbles - (num_customers * (50 + 1) / 2) = 1225 :=
by
  sorry

end NUMINAMATH_GPT_remaining_marbles_l225_22548


namespace NUMINAMATH_GPT_box_contains_1600_calories_l225_22560

theorem box_contains_1600_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  total_calories = 1600 :=
by
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  show total_calories = 1600
  sorry

end NUMINAMATH_GPT_box_contains_1600_calories_l225_22560


namespace NUMINAMATH_GPT_solve_for_z_l225_22582

variable {z : ℂ}
def complex_i := Complex.I

theorem solve_for_z (h : 1 - complex_i * z = -1 + complex_i * z) : z = -complex_i := by
  sorry

end NUMINAMATH_GPT_solve_for_z_l225_22582


namespace NUMINAMATH_GPT_sheila_tue_thu_hours_l225_22550

def hours_mwf : ℕ := 8
def days_mwf : ℕ := 3
def total_hours_mwf : ℕ := hours_mwf * days_mwf

def weekly_earnings : ℕ := 360
def hourly_rate : ℕ := 10
def earnings_mwf : ℕ := total_hours_mwf * hourly_rate

def earnings_tue_thu : ℕ := weekly_earnings - earnings_mwf
def hours_tue_thu : ℕ := earnings_tue_thu / hourly_rate

theorem sheila_tue_thu_hours : hours_tue_thu = 12 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_sheila_tue_thu_hours_l225_22550


namespace NUMINAMATH_GPT_problem1_l225_22599

theorem problem1 (x y : ℝ) (h1 : x * (x + y) = 27) (h2 : y * (x + y) = 54) : (x + y)^2 = 81 := 
by
  sorry

end NUMINAMATH_GPT_problem1_l225_22599


namespace NUMINAMATH_GPT_choose_president_and_secretary_same_gender_l225_22509

theorem choose_president_and_secretary_same_gender :
  let total_members := 25
  let boys := 15
  let girls := 10
  ∃ (total_ways : ℕ), total_ways = (boys * (boys - 1)) + (girls * (girls - 1)) := sorry

end NUMINAMATH_GPT_choose_president_and_secretary_same_gender_l225_22509


namespace NUMINAMATH_GPT_maximum_reduced_price_l225_22589

theorem maximum_reduced_price (marked_price : ℝ) (cost_price : ℝ) (reduced_price : ℝ) 
    (h1 : marked_price = 240) 
    (h2 : marked_price = cost_price * 1.6) 
    (h3 : reduced_price - cost_price ≥ cost_price * 0.1) : 
    reduced_price ≤ 165 :=
sorry

end NUMINAMATH_GPT_maximum_reduced_price_l225_22589


namespace NUMINAMATH_GPT_poly_solution_l225_22563

-- Definitions for the conditions of the problem
def poly1 (d g : ℚ) := 5 * d ^ 2 - 4 * d + g
def poly2 (d h : ℚ) := 4 * d ^ 2 + h * d - 5
def product (d g h : ℚ) := 20 * d ^ 4 - 31 * d ^ 3 - 17 * d ^ 2 + 23 * d - 10

-- Statement of the problem: proving g + h = 7/2 given the conditions.
theorem poly_solution
  (g h : ℚ)
  (cond : ∀ d : ℚ, poly1 d g * poly2 d h = product d g h) :
  g + h = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_poly_solution_l225_22563


namespace NUMINAMATH_GPT_sum_of_first_six_terms_geometric_sequence_l225_22514

theorem sum_of_first_six_terms_geometric_sequence :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * (1 - r ^ n) / (1 - r)
  S_n = 1365 / 4096 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_six_terms_geometric_sequence_l225_22514


namespace NUMINAMATH_GPT_analysis_method_sufficient_conditions_l225_22583

theorem analysis_method_sufficient_conditions (P : Prop) (analysis_method : ∀ (Q : Prop), (Q → P) → Q) :
  ∀ Q, (Q → P) → Q :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_analysis_method_sufficient_conditions_l225_22583


namespace NUMINAMATH_GPT_triangle_area_is_9_l225_22506

-- Define the vertices of the triangle
def x1 : ℝ := 1
def y1 : ℝ := 2
def x2 : ℝ := 4
def y2 : ℝ := 5
def x3 : ℝ := 6
def y3 : ℝ := 1

-- Define the area calculation formula for the triangle
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- The proof statement
theorem triangle_area_is_9 :
  triangle_area x1 y1 x2 y2 x3 y3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_9_l225_22506


namespace NUMINAMATH_GPT_value_of_x_squared_add_reciprocal_squared_l225_22526

theorem value_of_x_squared_add_reciprocal_squared (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_squared_add_reciprocal_squared_l225_22526


namespace NUMINAMATH_GPT_find_x_of_orthogonal_vectors_l225_22529

variable (x : ℝ)

def a : ℝ × ℝ × ℝ := (2, -3, 1)
def b : ℝ × ℝ × ℝ := (-4, 2, x)

theorem find_x_of_orthogonal_vectors (h : (2 * -4 + -3 * 2 + 1 * x) = 0) : x = 14 := by
  sorry

end NUMINAMATH_GPT_find_x_of_orthogonal_vectors_l225_22529


namespace NUMINAMATH_GPT_set_union_example_l225_22577

open Set

theorem set_union_example :
  let A := ({1, 3, 5, 6} : Set ℤ)
  let B := ({-1, 5, 7} : Set ℤ)
  A ∪ B = ({-1, 1, 3, 5, 6, 7} : Set ℤ) :=
by
  intros
  sorry

end NUMINAMATH_GPT_set_union_example_l225_22577


namespace NUMINAMATH_GPT_quadratic_root_a_l225_22566

theorem quadratic_root_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 4 = 0 ∧ x = 1) → a = -5 :=
by
  intro h
  have h1 : (1:ℝ)^2 + a * (1:ℝ) + 4 = 0 := sorry
  linarith

end NUMINAMATH_GPT_quadratic_root_a_l225_22566


namespace NUMINAMATH_GPT_rectangle_extraction_l225_22539

theorem rectangle_extraction (m : ℤ) (h1 : m > 12) : 
  ∃ (x y : ℤ), x ≤ y ∧ x * y > m ∧ x * (y - 1) < m :=
by
  sorry

end NUMINAMATH_GPT_rectangle_extraction_l225_22539


namespace NUMINAMATH_GPT_father_cards_given_l225_22527

-- Defining the conditions
def Janessa_initial_cards : Nat := 4
def eBay_cards : Nat := 36
def bad_cards : Nat := 4
def dexter_cards : Nat := 29
def janessa_kept_cards : Nat := 20

-- Proving the number of cards father gave her
theorem father_cards_given : ∃ n : Nat, n = 13 ∧ (Janessa_initial_cards + eBay_cards - bad_cards + n = dexter_cards + janessa_kept_cards) := 
by
  sorry

end NUMINAMATH_GPT_father_cards_given_l225_22527


namespace NUMINAMATH_GPT_union_card_ge_165_l225_22555

open Finset

variable (A : Finset ℕ) (A_i : Fin (11) → Finset ℕ)
variable (hA : A.card = 225)
variable (hA_i_card : ∀ i, (A_i i).card = 45)
variable (hA_i_intersect : ∀ i j, i < j → ((A_i i) ∩ (A_i j)).card = 9)

theorem union_card_ge_165 : (Finset.biUnion Finset.univ A_i).card ≥ 165 := by sorry

end NUMINAMATH_GPT_union_card_ge_165_l225_22555


namespace NUMINAMATH_GPT_clock_correct_time_fraction_l225_22532

/-- A 12-hour digital clock problem:
A 12-hour digital clock displays the hour and minute of a day.
Whenever it is supposed to display a '1' or a '2', it mistakenly displays a '9'.
The fraction of the day during which the clock shows the correct time is 7/24.
-/
theorem clock_correct_time_fraction : (7 : ℚ) / 24 = 7 / 24 :=
by sorry

end NUMINAMATH_GPT_clock_correct_time_fraction_l225_22532


namespace NUMINAMATH_GPT_train_speed_correct_l225_22585

def train_length : ℝ := 2500  -- Length of the train in meters.
def crossing_time : ℝ := 100  -- Time to cross the electric pole in seconds.
def expected_speed : ℝ := 25  -- Expected speed of the train in meters/second.

theorem train_speed_correct :
  (train_length / crossing_time) = expected_speed :=
by
  sorry

end NUMINAMATH_GPT_train_speed_correct_l225_22585


namespace NUMINAMATH_GPT_quadratic_decomposition_l225_22503

theorem quadratic_decomposition (a b c : ℝ) :
  (∀ x : ℝ, 6 * x^2 + 72 * x + 432 = a * (x + b)^2 + c) → a + b + c = 228 :=
sorry

end NUMINAMATH_GPT_quadratic_decomposition_l225_22503


namespace NUMINAMATH_GPT_sphere_surface_area_l225_22584

theorem sphere_surface_area (a : ℝ) (l R : ℝ)
  (h₁ : 6 * l^2 = a)
  (h₂ : l * Real.sqrt 3 = 2 * R) :
  4 * Real.pi * R^2 = (Real.pi / 2) * a :=
sorry

end NUMINAMATH_GPT_sphere_surface_area_l225_22584


namespace NUMINAMATH_GPT_percentage_decrease_l225_22562

variable {a b x m : ℝ} (p : ℝ)

theorem percentage_decrease (h₁ : a / b = 4 / 5)
                          (h₂ : x = 1.25 * a)
                          (h₃ : m = b * (1 - p / 100))
                          (h₄ : m / x = 0.8) :
  p = 20 :=
sorry

end NUMINAMATH_GPT_percentage_decrease_l225_22562


namespace NUMINAMATH_GPT_purely_imaginary_complex_number_l225_22542

theorem purely_imaginary_complex_number (a : ℝ) (i : ℂ)
  (h₁ : i * i = -1)
  (h₂ : ∃ z : ℂ, z = (a + i) / (1 - i) ∧ z.im ≠ 0 ∧ z.re = 0) :
  a = 1 :=
sorry

end NUMINAMATH_GPT_purely_imaginary_complex_number_l225_22542


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_product_2730_eq_42_l225_22528

theorem sum_of_consecutive_integers_product_2730_eq_42 :
  ∃ x : ℤ, x * (x + 1) * (x + 2) = 2730 ∧ x + (x + 1) + (x + 2) = 42 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_product_2730_eq_42_l225_22528


namespace NUMINAMATH_GPT_speed_against_current_l225_22597

noncomputable def man's_speed_with_current : ℝ := 20
noncomputable def current_speed : ℝ := 1

theorem speed_against_current :
  (man's_speed_with_current - 2 * current_speed) = 18 := by
sorry

end NUMINAMATH_GPT_speed_against_current_l225_22597


namespace NUMINAMATH_GPT_acute_triangle_sin_sum_gt_two_l225_22593

theorem acute_triangle_sin_sum_gt_two 
  {α β γ : ℝ} 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : 0 < γ ∧ γ < π / 2) 
  (h4 : α + β + γ = π) :
  (Real.sin α + Real.sin β + Real.sin γ > 2) :=
sorry

end NUMINAMATH_GPT_acute_triangle_sin_sum_gt_two_l225_22593


namespace NUMINAMATH_GPT_start_time_is_10_am_l225_22551

-- Definitions related to the problem statements
def distance_AB : ℝ := 600
def speed_A_to_B : ℝ := 70
def speed_B_to_A : ℝ := 80
def meeting_time : ℝ := 14  -- using 24-hour format, 2 pm as 14

-- Prove that the starting time is 10 am given the conditions
theorem start_time_is_10_am (t : ℝ) :
  (speed_A_to_B * t + speed_B_to_A * t = distance_AB) →
  (meeting_time - t = 10) :=
sorry

end NUMINAMATH_GPT_start_time_is_10_am_l225_22551


namespace NUMINAMATH_GPT_least_number_to_subtract_l225_22567

theorem least_number_to_subtract (n : ℕ) : 
  ∃ k : ℕ, k = 762429836 % 17 ∧ k = 15 := 
by sorry

end NUMINAMATH_GPT_least_number_to_subtract_l225_22567


namespace NUMINAMATH_GPT_problem_a_l225_22501

theorem problem_a (nums : Fin 101 → ℤ) : ∃ i j : Fin 101, i ≠ j ∧ (nums i - nums j) % 100 = 0 := sorry

end NUMINAMATH_GPT_problem_a_l225_22501


namespace NUMINAMATH_GPT_quilt_patch_cost_is_correct_l225_22521

noncomputable def quilt_area : ℕ := 16 * 20

def patch_area : ℕ := 4

def first_10_patch_cost : ℕ := 10

def discount_patch_cost : ℕ := 5

def total_patches (quilt_area patch_area : ℕ) : ℕ := quilt_area / patch_area

def cost_for_first_10 (first_10_patch_cost : ℕ) : ℕ := 10 * first_10_patch_cost

def cost_for_discounted (total_patches first_10_patch_cost discount_patch_cost : ℕ) : ℕ :=
  (total_patches - 10) * discount_patch_cost

def total_cost (cost_for_first_10 cost_for_discounted : ℕ) : ℕ :=
  cost_for_first_10 + cost_for_discounted

theorem quilt_patch_cost_is_correct :
  total_cost (cost_for_first_10 first_10_patch_cost)
             (cost_for_discounted (total_patches quilt_area patch_area) first_10_patch_cost discount_patch_cost) = 450 :=
by
  sorry

end NUMINAMATH_GPT_quilt_patch_cost_is_correct_l225_22521


namespace NUMINAMATH_GPT_A_or_B_not_A_or_C_A_and_C_A_and_B_or_C_l225_22592

-- Definitions of events
def A : Prop := sorry -- event that the part is of the first grade
def B : Prop := sorry -- event that the part is of the second grade
def C : Prop := sorry -- event that the part is of the third grade

-- Mathematically equivalent proof problems
theorem A_or_B : A ∨ B ↔ (A ∨ B) :=
by sorry

theorem not_A_or_C : ¬(A ∨ C) ↔ B :=
by sorry

theorem A_and_C : (A ∧ C) ↔ false :=
by sorry

theorem A_and_B_or_C : ((A ∧ B) ∨ C) ↔ C :=
by sorry

end NUMINAMATH_GPT_A_or_B_not_A_or_C_A_and_C_A_and_B_or_C_l225_22592


namespace NUMINAMATH_GPT_total_number_of_ways_is_144_l225_22519

def count_ways_to_place_letters_on_grid : Nat :=
  16 * 9

theorem total_number_of_ways_is_144 :
  count_ways_to_place_letters_on_grid = 144 :=
  by
    sorry

end NUMINAMATH_GPT_total_number_of_ways_is_144_l225_22519


namespace NUMINAMATH_GPT_units_digit_quotient_eq_one_l225_22531

theorem units_digit_quotient_eq_one :
  (2^2023 + 3^2023) / 5 % 10 = 1 := by
  sorry

end NUMINAMATH_GPT_units_digit_quotient_eq_one_l225_22531


namespace NUMINAMATH_GPT_sqrt_addition_l225_22590

theorem sqrt_addition :
  (Real.sqrt (49 + 81) + Real.sqrt (36 - 9) = Real.sqrt 130 + 3 * Real.sqrt 3) := 
by sorry

end NUMINAMATH_GPT_sqrt_addition_l225_22590


namespace NUMINAMATH_GPT_bowls_remaining_l225_22558

def initial_bowls : ℕ := 250

def customers_purchases : List (ℕ × ℕ) :=
  [(5, 7), (10, 15), (15, 22), (5, 36), (7, 46), (8, 0)]

def reward_ranges (bought : ℕ) : ℕ :=
  if bought >= 5 && bought <= 9 then 1
  else if bought >= 10 && bought <= 19 then 3
  else if bought >= 20 && bought <= 29 then 6
  else if bought >= 30 && bought <= 39 then 8
  else if bought >= 40 then 12
  else 0

def total_free_bowls : ℕ :=
  List.foldl (λ acc (n, b) => acc + n * reward_ranges b) 0 customers_purchases

theorem bowls_remaining :
  initial_bowls - total_free_bowls = 1 := by
  sorry

end NUMINAMATH_GPT_bowls_remaining_l225_22558


namespace NUMINAMATH_GPT_prob_equals_two_yellow_marbles_l225_22534

noncomputable def probability_two_yellow_marbles : ℚ :=
  let total_marbles : ℕ := 3 + 4 + 8
  let yellow_marbles : ℕ := 4
  let first_draw_prob : ℚ := yellow_marbles / total_marbles
  let second_total_marbles : ℕ := total_marbles - 1
  let second_yellow_marbles : ℕ := yellow_marbles - 1
  let second_draw_prob : ℚ := second_yellow_marbles / second_total_marbles
  first_draw_prob * second_draw_prob

theorem prob_equals_two_yellow_marbles :
  probability_two_yellow_marbles = 2 / 35 :=
by
  sorry

end NUMINAMATH_GPT_prob_equals_two_yellow_marbles_l225_22534


namespace NUMINAMATH_GPT_evaluate_f_3_minus_f_neg_3_l225_22573

def f (x : ℝ) : ℝ := x^4 + x^2 + 7 * x

theorem evaluate_f_3_minus_f_neg_3 : f 3 - f (-3) = 42 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_3_minus_f_neg_3_l225_22573


namespace NUMINAMATH_GPT_problem_statement_l225_22524

theorem problem_statement (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : 
  x^12 - 7 * x^8 + x^4 = 343 :=
sorry

end NUMINAMATH_GPT_problem_statement_l225_22524


namespace NUMINAMATH_GPT_factory_a_min_hours_l225_22525

theorem factory_a_min_hours (x : ℕ) :
  (550 * x + (700 - 55 * x) / 45 * 495 ≤ 7260) → (8 ≤ x) :=
by
  sorry

end NUMINAMATH_GPT_factory_a_min_hours_l225_22525


namespace NUMINAMATH_GPT_find_number_l225_22595

variable (a b x : ℕ)

theorem find_number
    (h1 : x * a = 7 * b)
    (h2 : x * a = 20)
    (h3 : 7 * b = 20) :
    x = 1 :=
sorry

end NUMINAMATH_GPT_find_number_l225_22595


namespace NUMINAMATH_GPT_three_xy_eq_24_l225_22535

variable {x y : ℝ}

theorem three_xy_eq_24 (h : x * (x + 3 * y) = x^2 + 24) : 3 * x * y = 24 :=
sorry

end NUMINAMATH_GPT_three_xy_eq_24_l225_22535


namespace NUMINAMATH_GPT_complement_of_M_in_U_l225_22537

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5}
noncomputable def M : Set ℕ := {0, 1}

theorem complement_of_M_in_U : (U \ M) = {2, 3, 4, 5} :=
by
  -- The proof is omitted here.
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_l225_22537


namespace NUMINAMATH_GPT_maximum_f_value_l225_22553

noncomputable def otimes (a b : ℝ) : ℝ :=
if a ≤ b then a else b

noncomputable def f (x : ℝ) : ℝ :=
otimes (3 * x^2 + 6) (23 - x^2)

theorem maximum_f_value : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = 4 :=
sorry

end NUMINAMATH_GPT_maximum_f_value_l225_22553


namespace NUMINAMATH_GPT_units_digit_powers_difference_l225_22576

theorem units_digit_powers_difference (p : ℕ) 
  (h1: p > 0) 
  (h2: p % 2 = 0) 
  (h3: (p % 10 + 2) % 10 = 8) : 
  ((p ^ 3) % 10 - (p ^ 2) % 10) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_powers_difference_l225_22576


namespace NUMINAMATH_GPT_required_raise_percentage_l225_22517

theorem required_raise_percentage (S : ℝ) (hS : S > 0) : 
  ((S - (0.85 * S - 50)) / (0.85 * S - 50) = 0.1875) :=
by
  -- Proof of this theorem can be carried out here
  sorry

end NUMINAMATH_GPT_required_raise_percentage_l225_22517


namespace NUMINAMATH_GPT_no_such_integers_exists_l225_22510

theorem no_such_integers_exists :
  ∀ (P : ℕ → ℕ), (∀ x, P x = x ^ 2000 - x ^ 1000 + 1) →
  ¬(∃ (a : Fin 8002 → ℕ), Function.Injective a ∧
  (∀ i j k : Fin 8002, i ≠ j → j ≠ k → i ≠ k → a i * a j * a k ∣ P (a i) * P (a j) * P (a k))) := 
by
  intro P hP notExists
  have contra : ∃ (a : Fin 8002 → ℕ), Function.Injective a ∧
    (∀ i j k : Fin 8002, i ≠ j → j ≠ k → i ≠ k → a i * a j * a k ∣ P (a i) * P (a j) * P (a k)) := notExists
  sorry

end NUMINAMATH_GPT_no_such_integers_exists_l225_22510


namespace NUMINAMATH_GPT_max_value_3x_plus_4y_l225_22508

theorem max_value_3x_plus_4y (x y : ℝ) : x^2 + y^2 = 14 * x + 6 * y + 6 → 3 * x + 4 * y ≤ 73 :=
sorry

end NUMINAMATH_GPT_max_value_3x_plus_4y_l225_22508


namespace NUMINAMATH_GPT_zach_needs_more_money_zach_more_money_needed_l225_22561

/-!
# Zach's Bike Savings Problem
Zach needs $100 to buy a brand new bike.
Weekly allowance: $5.
Earnings from mowing the lawn: $10.
Earnings from babysitting: $7 per hour.
Zach has already saved $65.
He will receive weekly allowance on Friday.
He will mow the lawn and babysit for 2 hours this Saturday.
Prove that Zach needs $6 more to buy the bike.
-/

def zach_current_savings : ℕ := 65
def bike_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def mowing_earnings : ℕ := 10
def babysitting_rate : ℕ := 7
def babysitting_hours : ℕ := 2

theorem zach_needs_more_money : zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours)) = 94 :=
by sorry

theorem zach_more_money_needed : bike_cost - (zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours))) = 6 :=
by sorry

end NUMINAMATH_GPT_zach_needs_more_money_zach_more_money_needed_l225_22561


namespace NUMINAMATH_GPT_arc_intercept_length_l225_22594

noncomputable def side_length : ℝ := 4
noncomputable def diagonal_length : ℝ := Real.sqrt (side_length^2 + side_length^2)
noncomputable def radius : ℝ := diagonal_length / 2
noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def arc_length_one_side : ℝ := circumference / 4

theorem arc_intercept_length :
  arc_length_one_side = Real.sqrt 2 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_arc_intercept_length_l225_22594


namespace NUMINAMATH_GPT_safety_rent_a_car_cost_per_mile_l225_22552

/-
Problem:
Prove that the cost per mile for Safety Rent-a-Car is 0.177 dollars, given that the total cost of renting an intermediate-size car for 150 miles is the same for Safety Rent-a-Car and City Rentals, with their respective pricing schemes.
-/

theorem safety_rent_a_car_cost_per_mile :
  let x := 21.95
  let y := 18.95
  let z := 0.21
  (x + 150 * real_safety_per_mile) = (y + 150 * z) ↔ real_safety_per_mile = 0.177 :=
by
  sorry

end NUMINAMATH_GPT_safety_rent_a_car_cost_per_mile_l225_22552


namespace NUMINAMATH_GPT_product_is_correct_l225_22536

noncomputable def IKS := 521
noncomputable def KSI := 215
def product := 112015

theorem product_is_correct : IKS * KSI = product :=
by
  -- Proof yet to be constructed
  sorry

end NUMINAMATH_GPT_product_is_correct_l225_22536


namespace NUMINAMATH_GPT_area_of_region_is_12_l225_22504

def region_area : ℝ :=
  let f1 (x : ℝ) : ℝ := |x - 2|
  let f2 (x : ℝ) : ℝ := 5 - |x + 1|
  let valid_region (x y : ℝ) : Prop := f1 x ≤ y ∧ y ≤ f2 x
  12

theorem area_of_region_is_12 :
  ∃ (area : ℝ), region_area = 12 := by
  use 12
  sorry

end NUMINAMATH_GPT_area_of_region_is_12_l225_22504


namespace NUMINAMATH_GPT_fencing_required_l225_22530

theorem fencing_required {length width : ℝ} 
  (uncovered_side : length = 20)
  (field_area : length * width = 50) :
  2 * width + length = 25 :=
by
  sorry

end NUMINAMATH_GPT_fencing_required_l225_22530


namespace NUMINAMATH_GPT_find_a_in_terms_of_x_l225_22538

theorem find_a_in_terms_of_x (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a^3 - b^3 = 22 * x^3) (h₃ : a - b = 2 * x) : 
  a = x * (1 + (Real.sqrt (40 / 3)) / 2) ∨ a = x * (1 - (Real.sqrt (40 / 3)) / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_in_terms_of_x_l225_22538


namespace NUMINAMATH_GPT_smallest_k_49_divides_binom_l225_22586

theorem smallest_k_49_divides_binom : 
  ∃ k : ℕ, 0 < k ∧ 49 ∣ Nat.choose (2 * k) k ∧ (∀ m : ℕ, 0 < m ∧ 49 ∣ Nat.choose (2 * m) m → k ≤ m) ∧ k = 25 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_49_divides_binom_l225_22586


namespace NUMINAMATH_GPT_union_M_N_is_U_l225_22533

-- Defining the universal set as the set of real numbers
def U : Set ℝ := Set.univ

-- Defining the set M
def M : Set ℝ := {x | x > 0}

-- Defining the set N
def N : Set ℝ := {x | x^2 >= x}

-- Stating the theorem that M ∪ N = U
theorem union_M_N_is_U : M ∪ N = U :=
  sorry

end NUMINAMATH_GPT_union_M_N_is_U_l225_22533


namespace NUMINAMATH_GPT_greatest_number_remainder_l225_22574

theorem greatest_number_remainder (G R : ℕ) (h1 : 150 % G = 50) (h2 : 230 % G = 5) (h3 : 175 % G = R) (h4 : ∀ g, g ∣ 100 → g ∣ 225 → g ∣ (175 - R) → g ≤ G) : R = 0 :=
by {
  -- This is the statement only; the proof is omitted as per the instructions.
  sorry
}

end NUMINAMATH_GPT_greatest_number_remainder_l225_22574


namespace NUMINAMATH_GPT_chi_square_confidence_l225_22546

theorem chi_square_confidence (chi_square : ℝ) (df : ℕ) (critical_value : ℝ) :
  chi_square = 6.825 ∧ df = 1 ∧ critical_value = 6.635 → confidence_level = 0.99 := 
by
  sorry

end NUMINAMATH_GPT_chi_square_confidence_l225_22546


namespace NUMINAMATH_GPT_a_n_general_term_b_n_general_term_l225_22505

noncomputable def seq_a (n : ℕ) : ℕ :=
  2 * n - 1

theorem a_n_general_term (n : ℕ) (Sn : ℕ → ℕ) (S_property : ∀ n : ℕ, 4 * Sn n = (seq_a n) ^ 2 + 2 * seq_a n + 1) :
  seq_a n = 2 * n - 1 :=
sorry

noncomputable def geom_seq (q : ℕ) (n : ℕ) : ℕ :=
  q ^ (n - 1)

theorem b_n_general_term (n m q : ℕ) (a1 am am3 : ℕ) (b_property : ∀ n : ℕ, geom_seq q n = q ^ (n - 1))
  (a_property : ∀ n : ℕ, seq_a n = 2 * n - 1)
  (b1_condition : geom_seq q 1 = seq_a 1) (bm_condition : geom_seq q m = seq_a m)
  (bm1_condition : geom_seq q (m + 1) = seq_a (m + 3)) :
  q = 3 ∨ q = 7 ∧ (∀ n : ℕ, geom_seq q n = 3 ^ (n - 1) ∨ geom_seq q n = 7 ^ (n - 1)) :=
sorry

end NUMINAMATH_GPT_a_n_general_term_b_n_general_term_l225_22505


namespace NUMINAMATH_GPT_num_int_solutions_l225_22511

theorem num_int_solutions (x : ℤ) : 
  (x^4 - 39 * x^2 + 140 < 0) ↔ (x = 3 ∨ x = -3 ∨ x = 4 ∨ x = -4 ∨ x = 5 ∨ x = -5) := 
sorry

end NUMINAMATH_GPT_num_int_solutions_l225_22511


namespace NUMINAMATH_GPT_negation_of_quadratic_statement_l225_22507

variable {x a b : ℝ}

theorem negation_of_quadratic_statement (h : x = a ∨ x = b) : x^2 - (a + b) * x + ab = 0 := sorry

end NUMINAMATH_GPT_negation_of_quadratic_statement_l225_22507


namespace NUMINAMATH_GPT_find_a_l225_22572

open Nat

-- Define the conditions and the proof goal
theorem find_a (a b : ℕ) (h1 : 2019 = a^2 - b^2) (h2 : a < 1000) : a = 338 :=
sorry

end NUMINAMATH_GPT_find_a_l225_22572


namespace NUMINAMATH_GPT_students_passed_both_tests_l225_22587

theorem students_passed_both_tests :
  ∀ (total students_passed_long_jump students_passed_shot_put students_failed_both x : ℕ),
    total = 50 →
    students_passed_long_jump = 40 →
    students_passed_shot_put = 31 →
    students_failed_both = 4 →
    (students_passed_long_jump - x) + (students_passed_shot_put - x) + x + students_failed_both = total →
    x = 25 :=
by intros total students_passed_long_jump students_passed_shot_put students_failed_both x
   intro total_eq students_passed_long_jump_eq students_passed_shot_put_eq students_failed_both_eq sum_eq
   sorry

end NUMINAMATH_GPT_students_passed_both_tests_l225_22587


namespace NUMINAMATH_GPT_max_k_range_minus_five_l225_22512

theorem max_k_range_minus_five :
  ∃ k : ℝ, (∀ x : ℝ, x^2 + 5 * x + k = -5) → k = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_k_range_minus_five_l225_22512


namespace NUMINAMATH_GPT_b101_mod_49_l225_22543

-- Definitions based on conditions
def b (n : ℕ) : ℕ := 5^n + 7^n

-- The formal statement of the proof problem
theorem b101_mod_49 : b 101 % 49 = 12 := by
  sorry

end NUMINAMATH_GPT_b101_mod_49_l225_22543


namespace NUMINAMATH_GPT_frosting_problem_equivalent_l225_22523

/-
Problem:
Cagney can frost a cupcake every 15 seconds.
Lacey can frost a cupcake every 40 seconds.
Mack can frost a cupcake every 25 seconds.
Prove that together they can frost 79 cupcakes in 10 minutes.
-/

def cupcakes_frosted_together_in_10_minutes (rate_cagney rate_lacey rate_mack : ℕ) (time_minutes : ℕ) : ℕ :=
  let time_seconds := time_minutes * 60
  let rate_cagney := 1 / 15
  let rate_lacey := 1 / 40
  let rate_mack := 1 / 25
  let combined_rate := rate_cagney + rate_lacey + rate_mack
  combined_rate * time_seconds

theorem frosting_problem_equivalent:
  cupcakes_frosted_together_in_10_minutes 1 1 1 10 = 79 := by
  sorry

end NUMINAMATH_GPT_frosting_problem_equivalent_l225_22523


namespace NUMINAMATH_GPT_value_subtracted_is_five_l225_22569

variable (N x : ℕ)

theorem value_subtracted_is_five
  (h1 : (N - x) / 7 = 7)
  (h2 : (N - 14) / 10 = 4) : x = 5 := by
  sorry

end NUMINAMATH_GPT_value_subtracted_is_five_l225_22569


namespace NUMINAMATH_GPT_crayons_initially_l225_22540

theorem crayons_initially (crayons_left crayons_lost : ℕ) (h_left : crayons_left = 134) (h_lost : crayons_lost = 345) :
  crayons_left + crayons_lost = 479 :=
by
  sorry

end NUMINAMATH_GPT_crayons_initially_l225_22540


namespace NUMINAMATH_GPT_print_output_l225_22591

-- Conditions
def a : Nat := 10

/-- The print statement with the given conditions should output "a=10" -/
theorem print_output : "a=" ++ toString a = "a=10" :=
sorry

end NUMINAMATH_GPT_print_output_l225_22591


namespace NUMINAMATH_GPT_dogwood_trees_initial_count_l225_22554

theorem dogwood_trees_initial_count 
  (dogwoods_today : ℕ) 
  (dogwoods_tomorrow : ℕ) 
  (final_dogwoods : ℕ)
  (total_planted : ℕ := dogwoods_today + dogwoods_tomorrow)
  (initial_dogwoods := final_dogwoods - total_planted)
  (h : dogwoods_today = 41)
  (h1 : dogwoods_tomorrow = 20)
  (h2 : final_dogwoods = 100) : 
  initial_dogwoods = 39 := 
by sorry

end NUMINAMATH_GPT_dogwood_trees_initial_count_l225_22554


namespace NUMINAMATH_GPT_johns_new_weekly_earnings_l225_22557

-- Definition of the initial weekly earnings
def initial_weekly_earnings := 40

-- Definition of the percent increase in earnings
def percent_increase := 100

-- Definition for the final weekly earnings after the raise
def final_weekly_earnings (initial_earnings : Nat) (percentage : Nat) := 
  initial_earnings + (initial_earnings * percentage / 100)

-- Theorem stating John’s final weekly earnings after the raise
theorem johns_new_weekly_earnings : final_weekly_earnings initial_weekly_earnings percent_increase = 80 :=
  by
  sorry

end NUMINAMATH_GPT_johns_new_weekly_earnings_l225_22557
