import Mathlib

namespace NUMINAMATH_GPT_dice_probability_l2154_215412

theorem dice_probability :
  let outcomes : List ℕ := [2, 3, 4, 5]
  let total_possible_outcomes := 6 * 6 * 6
  let successful_outcomes := 4 * 4 * 4
  (successful_outcomes / total_possible_outcomes : ℚ) = 8 / 27 :=
by
  sorry

end NUMINAMATH_GPT_dice_probability_l2154_215412


namespace NUMINAMATH_GPT_different_values_of_t_l2154_215437

-- Define the conditions on the numbers
variables (p q r s t : ℕ)

-- Define the constraints: p, q, r, s, and t are distinct single-digit numbers
def valid_single_digit (x : ℕ) := x > 0 ∧ x < 10
def distinct_single_digits (p q r s t : ℕ) := 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t

-- Define the relationships given in the problem
def conditions (p q r s t : ℕ) :=
  valid_single_digit p ∧
  valid_single_digit q ∧
  valid_single_digit r ∧
  valid_single_digit s ∧
  valid_single_digit t ∧
  distinct_single_digits p q r s t ∧
  p - q = r ∧
  r - s = t

-- Theorem to be proven
theorem different_values_of_t : 
  ∃! (count : ℕ), count = 6 ∧ (∃ p q r s t, conditions p q r s t) := 
sorry

end NUMINAMATH_GPT_different_values_of_t_l2154_215437


namespace NUMINAMATH_GPT_unique_solution_for_4_circ_20_l2154_215410

def operation (x y : ℝ) : ℝ := 3 * x - 2 * y + 2 * x * y

theorem unique_solution_for_4_circ_20 : ∃! y : ℝ, operation 4 y = 20 :=
by 
  sorry

end NUMINAMATH_GPT_unique_solution_for_4_circ_20_l2154_215410


namespace NUMINAMATH_GPT_cranes_in_each_flock_l2154_215409

theorem cranes_in_each_flock (c : ℕ) (h1 : ∃ n : ℕ, 13 * n = 221)
  (h2 : ∃ n : ℕ, c * n = 221) :
  c = 221 :=
by sorry

end NUMINAMATH_GPT_cranes_in_each_flock_l2154_215409


namespace NUMINAMATH_GPT_least_number_of_faces_l2154_215475

def faces_triangular_prism : ℕ := 5
def faces_quadrangular_prism : ℕ := 6
def faces_triangular_pyramid : ℕ := 4
def faces_quadrangular_pyramid : ℕ := 5
def faces_truncated_quadrangular_pyramid : ℕ := 6

theorem least_number_of_faces : faces_triangular_pyramid < faces_triangular_prism ∧
                                faces_triangular_pyramid < faces_quadrangular_prism ∧
                                faces_triangular_pyramid < faces_quadrangular_pyramid ∧
                                faces_triangular_pyramid < faces_truncated_quadrangular_pyramid 
                                :=
by {
  sorry
}

end NUMINAMATH_GPT_least_number_of_faces_l2154_215475


namespace NUMINAMATH_GPT_range_f_3_l2154_215421

section

variables (a c : ℝ) (f : ℝ → ℝ)
def quadratic_function := ∀ x, f x = a * x^2 - c

-- Define the constraints given in the problem
axiom h1 : -4 ≤ f 1 ∧ f 1 ≤ -1
axiom h2 : -1 ≤ f 2 ∧ f 2 ≤ 5

-- Prove that the correct range for f(3) is -1 ≤ f(3) ≤ 20
theorem range_f_3 (a c : ℝ) (f : ℝ → ℝ) (h1 : -4 ≤ f 1 ∧ f 1 ≤ -1) (h2 : -1 ≤ f 2 ∧ f 2 ≤ 5):
  -1 ≤ f 3 ∧ f 3 ≤ 20 :=
sorry

end

end NUMINAMATH_GPT_range_f_3_l2154_215421


namespace NUMINAMATH_GPT_unique_solution_l2154_215425

theorem unique_solution (n : ℕ) (h1 : n > 0) (h2 : n^2 ∣ 3^n + 1) : n = 1 :=
sorry

end NUMINAMATH_GPT_unique_solution_l2154_215425


namespace NUMINAMATH_GPT_sample_size_is_correct_l2154_215427

-- Define the conditions
def total_students : ℕ := 40 * 50
def students_selected : ℕ := 150

-- Theorem: The sample size is 150 given that 150 students are selected
theorem sample_size_is_correct : students_selected = 150 := by
  sorry  -- Proof to be completed

end NUMINAMATH_GPT_sample_size_is_correct_l2154_215427


namespace NUMINAMATH_GPT_rectangle_right_triangle_max_area_and_hypotenuse_l2154_215439

theorem rectangle_right_triangle_max_area_and_hypotenuse (x y h : ℝ) (h_triangle : h^2 = x^2 + y^2) (h_perimeter : 2 * (x + y) = 60) :
  (x * y ≤ 225) ∧ (x = 15) ∧ (y = 15) ∧ (h = 15 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_right_triangle_max_area_and_hypotenuse_l2154_215439


namespace NUMINAMATH_GPT_billy_age_l2154_215419

theorem billy_age (B J : ℕ) (h1 : B = 3 * J) (h2 : B + J = 64) : B = 48 :=
by
  sorry

end NUMINAMATH_GPT_billy_age_l2154_215419


namespace NUMINAMATH_GPT_expr1_is_91_expr2_is_25_expr3_is_49_expr4_is_1_l2154_215407

-- Definitions to add parentheses in the given expressions to achieve the desired results.
def expr1 := 7 * (9 + 12 / 3)
def expr2 := (7 * 9 + 12) / 3
def expr3 := 7 * (9 + 12) / 3
def expr4 := (48 * 6) / (48 * 6)

-- Proof statements
theorem expr1_is_91 : expr1 = 91 := 
by sorry

theorem expr2_is_25 : expr2 = 25 :=
by sorry

theorem expr3_is_49 : expr3 = 49 :=
by sorry

theorem expr4_is_1 : expr4 = 1 :=
by sorry

end NUMINAMATH_GPT_expr1_is_91_expr2_is_25_expr3_is_49_expr4_is_1_l2154_215407


namespace NUMINAMATH_GPT_distribute_money_equation_l2154_215471

theorem distribute_money_equation (x : ℕ) (hx : x > 0) : 
  (10 : ℚ) / x = (40 : ℚ) / (x + 6) := 
sorry

end NUMINAMATH_GPT_distribute_money_equation_l2154_215471


namespace NUMINAMATH_GPT_range_of_a_l2154_215481

theorem range_of_a (a : ℝ) : 
  4 * a^2 - 12 * (a + 6) > 0 ↔ a < -3 ∨ a > 6 := 
by sorry

end NUMINAMATH_GPT_range_of_a_l2154_215481


namespace NUMINAMATH_GPT_ribbon_total_length_l2154_215430

theorem ribbon_total_length (R : ℝ)
  (h_first : R - (1/2)*R = (1/2)*R)
  (h_second : (1/2)*R - (1/3)*((1/2)*R) = (1/3)*R)
  (h_third : (1/3)*R - (1/2)*((1/3)*R) = (1/6)*R)
  (h_remaining : (1/6)*R = 250) :
  R = 1500 :=
sorry

end NUMINAMATH_GPT_ribbon_total_length_l2154_215430


namespace NUMINAMATH_GPT_ninth_group_number_l2154_215482

-- Conditions
def num_workers : ℕ := 100
def sample_size : ℕ := 20
def group_size : ℕ := num_workers / sample_size
def fifth_group_number : ℕ := 23

-- Theorem stating the result for the 9th group number.
theorem ninth_group_number : ∃ n : ℕ, n = 43 :=
by
  -- We calculate the numbers step by step.
  have interval : ℕ := group_size
  have difference : ℕ := 9 - 5
  have increment : ℕ := difference * interval
  have ninth_group_num : ℕ := fifth_group_number + increment
  use ninth_group_num
  sorry

end NUMINAMATH_GPT_ninth_group_number_l2154_215482


namespace NUMINAMATH_GPT_stuffed_animals_total_l2154_215404

theorem stuffed_animals_total :
  let McKenna := 34
  let Kenley := 2 * McKenna
  let Tenly := Kenley + 5
  McKenna + Kenley + Tenly = 175 :=
by
  sorry

end NUMINAMATH_GPT_stuffed_animals_total_l2154_215404


namespace NUMINAMATH_GPT_min_sum_xy_l2154_215438

theorem min_sum_xy (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_ne_hy : x ≠ y)
  (h_eq : (1 / x : ℝ) + (1 / y) = 1 / 12) : x + y = 49 :=
sorry

end NUMINAMATH_GPT_min_sum_xy_l2154_215438


namespace NUMINAMATH_GPT_remainder_div_84_l2154_215442

def a := (5 : ℕ) * 10 ^ 2015 + (5 : ℕ)

theorem remainder_div_84 (a : ℕ) (h : a = (5 : ℕ) * 10 ^ 2015 + (5 : ℕ)) : a % 84 = 63 := 
by 
  -- Placeholder for the actual steps to prove
  sorry

end NUMINAMATH_GPT_remainder_div_84_l2154_215442


namespace NUMINAMATH_GPT_product_of_differences_l2154_215428

theorem product_of_differences (p q p' q' α β α' β' : ℝ)
  (h1 : α + β = -p) (h2 : α * β = q)
  (h3 : α' + β' = -p') (h4 : α' * β' = q') :
  ((α - α') * (α - β') * (β - α') * (β - β') = (q - q')^2 + (p - p') * (q' * p - p' * q)) :=
sorry

end NUMINAMATH_GPT_product_of_differences_l2154_215428


namespace NUMINAMATH_GPT_sum_of_positive_integers_l2154_215456

theorem sum_of_positive_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 272) : x + y = 32 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_positive_integers_l2154_215456


namespace NUMINAMATH_GPT_point_quadrant_l2154_215458

theorem point_quadrant (a b : ℝ) (h : |a - 4| + (b + 3)^2 = 0) : b < 0 ∧ a > 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_point_quadrant_l2154_215458


namespace NUMINAMATH_GPT_scalene_polygon_exists_l2154_215401

theorem scalene_polygon_exists (n: ℕ) (a: Fin n → ℝ) (h: ∀ i, 1 ≤ a i ∧ a i ≤ 2013) (h_geq: n ≥ 13):
  ∃ (A B C : Fin n), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ a A + a B > a C ∧ a A + a C > a B ∧ a B + a C > a A :=
sorry

end NUMINAMATH_GPT_scalene_polygon_exists_l2154_215401


namespace NUMINAMATH_GPT_income_is_12000_l2154_215476

theorem income_is_12000 (P : ℝ) : (P * 1.02 = 12240) → (P = 12000) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_income_is_12000_l2154_215476


namespace NUMINAMATH_GPT_total_laps_jogged_l2154_215485

-- Defining the conditions
def jogged_PE_class : ℝ := 1.12
def jogged_track_practice : ℝ := 2.12

-- Statement to prove
theorem total_laps_jogged : jogged_PE_class + jogged_track_practice = 3.24 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_total_laps_jogged_l2154_215485


namespace NUMINAMATH_GPT_intersection_equal_l2154_215491

noncomputable def M := { y : ℝ | ∃ x : ℝ, y = Real.log (x + 1) / Real.log (1 / 2) ∧ x ≥ 3 }
noncomputable def N := { x : ℝ | x^2 + 2 * x - 3 ≤ 0 }

theorem intersection_equal : M ∩ N = {a : ℝ | -3 ≤ a ∧ a ≤ -2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_equal_l2154_215491


namespace NUMINAMATH_GPT_fractional_part_of_students_who_walk_home_l2154_215466

def fraction_bus := 1 / 3
def fraction_automobile := 1 / 5
def fraction_bicycle := 1 / 8
def fraction_scooter := 1 / 10

theorem fractional_part_of_students_who_walk_home :
  (1 : ℚ) - (fraction_bus + fraction_automobile + fraction_bicycle + fraction_scooter) = 29 / 120 :=
by
  sorry

end NUMINAMATH_GPT_fractional_part_of_students_who_walk_home_l2154_215466


namespace NUMINAMATH_GPT_andrea_reaches_lauren_in_25_minutes_l2154_215414

noncomputable def initial_distance : ℝ := 30
noncomputable def decrease_rate : ℝ := 90
noncomputable def Lauren_stop_time : ℝ := 10 / 60

theorem andrea_reaches_lauren_in_25_minutes :
  ∃ v_L v_A : ℝ, v_A = 2 * v_L ∧ v_A + v_L = decrease_rate ∧ ∃ remaining_distance remaining_time final_time : ℝ, 
  remaining_distance = initial_distance - decrease_rate * Lauren_stop_time ∧ 
  remaining_time = remaining_distance / v_A ∧ 
  final_time = Lauren_stop_time + remaining_time ∧ 
  final_time * 60 = 25 :=
sorry

end NUMINAMATH_GPT_andrea_reaches_lauren_in_25_minutes_l2154_215414


namespace NUMINAMATH_GPT_find_a_l2154_215415

def E (a b c : ℤ) : ℤ := a * b * b + c

theorem find_a (a : ℤ) : E a 3 1 = E a 5 11 → a = -5 / 8 := 
by sorry

end NUMINAMATH_GPT_find_a_l2154_215415


namespace NUMINAMATH_GPT_number_of_divisors_of_n_l2154_215462

def n : ℕ := 2^3 * 3^4 * 5^3 * 7^2

theorem number_of_divisors_of_n : ∃ d : ℕ, d = 240 ∧ ∀ k : ℕ, k ∣ n ↔ ∃ a b c d : ℕ, 0 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 4 ∧ 0 ≤ c ∧ c ≤ 3 ∧ 0 ≤ d ∧ d ≤ 2 := 
sorry

end NUMINAMATH_GPT_number_of_divisors_of_n_l2154_215462


namespace NUMINAMATH_GPT_number_of_equilateral_triangles_in_lattice_l2154_215454

-- Definitions representing the conditions of the problem
def is_unit_distance (a b : ℕ) : Prop :=
  true -- Assume true as we are not focusing on the definition

def expanded_hexagonal_lattice (p : ℕ) : Prop :=
  true -- Assume true as the specific construction details are abstracted

-- The target theorem statement
theorem number_of_equilateral_triangles_in_lattice 
  (lattice : ℕ → Prop) (dist : ℕ → ℕ → Prop) 
  (h₁ : ∀ p, lattice p → dist p p) 
  (h₂ : ∀ p, (expanded_hexagonal_lattice p) ↔ lattice p ∧ dist p p) : 
  ∃ n, n = 32 :=
by 
  existsi 32
  sorry

end NUMINAMATH_GPT_number_of_equilateral_triangles_in_lattice_l2154_215454


namespace NUMINAMATH_GPT_candy_boxes_system_l2154_215464

-- Given conditions and definitions
def sheets_total (x y : ℕ) : Prop := x + y = 35
def sheet_usage (x y : ℕ) : Prop := 20 * x = 30 * y / 2

-- Statement
theorem candy_boxes_system (x y : ℕ) (h1 : sheets_total x y) (h2 : sheet_usage x y) : 
  (x + y = 35) ∧ (20 * x = 30 * y / 2) := 
by
sorry

end NUMINAMATH_GPT_candy_boxes_system_l2154_215464


namespace NUMINAMATH_GPT_sum_six_consecutive_integers_l2154_215494

-- Statement of the problem
theorem sum_six_consecutive_integers (n : ℤ) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5)) = 6 * n + 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_six_consecutive_integers_l2154_215494


namespace NUMINAMATH_GPT_solve_for_b_l2154_215452

theorem solve_for_b (b : ℝ) (m : ℝ) (h : b > 0)
  (h1 : ∀ x : ℝ, x^2 + b * x + 54 = (x + m) ^ 2 + 18) : b = 12 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_b_l2154_215452


namespace NUMINAMATH_GPT_cube_triangulation_impossible_l2154_215411

theorem cube_triangulation_impossible (vertex_sum : ℝ) (triangle_inter_sum : ℝ) (triangle_sum : ℝ) :
  vertex_sum = 270 ∧ triangle_inter_sum = 360 ∧ triangle_sum = 180 → ∃ (n : ℕ), n = 3 ∧ ∀ (m : ℕ), m ≠ 3 → false :=
by
  sorry

end NUMINAMATH_GPT_cube_triangulation_impossible_l2154_215411


namespace NUMINAMATH_GPT_cost_of_painting_new_room_l2154_215451

theorem cost_of_painting_new_room
  (L B H : ℝ)    -- Dimensions of the original room
  (c : ℝ)        -- Cost to paint the original room
  (h₁ : c = 350) -- Given that the cost of painting the original room is Rs. 350
  (A : ℝ)        -- Area of the walls of the original room
  (h₂ : A = 2 * (L + B) * H) -- Given the area calculation for the original room
  (newA : ℝ)     -- Area of the walls of the new room
  (h₃ : newA = 18 * (L + B) * H) -- Given the area calculation for the new room
  : (350 / (2 * (L + B) * H)) * (18 * (L + B) * H) = 3150 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_painting_new_room_l2154_215451


namespace NUMINAMATH_GPT_negation_equivalence_l2154_215420

-- Define the proposition P stating 'there exists an x in ℝ such that x^2 - 2x + 4 > 0'
def P : Prop := ∃ x : ℝ, x^2 - 2*x + 4 > 0

-- Define the proposition Q which is the negation of proposition P
def Q : Prop := ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0

-- State the proof problem: Prove that the negation of proposition P is equivalent to proposition Q
theorem negation_equivalence : ¬ P ↔ Q := by
  -- Proof to be provided.
  sorry

end NUMINAMATH_GPT_negation_equivalence_l2154_215420


namespace NUMINAMATH_GPT_even_func_decreasing_on_neg_interval_l2154_215431

variable {f : ℝ → ℝ}

theorem even_func_decreasing_on_neg_interval
  (h_even : ∀ x, f x = f (-x))
  (h_increasing : ∀ (a b : ℝ), 3 ≤ a → a < b → b ≤ 7 → f a < f b)
  (h_min_val : ∀ x, 3 ≤ x → x ≤ 7 → f x ≥ 2) :
  (∀ (a b : ℝ), -7 ≤ a → a < b → b ≤ -3 → f a > f b) ∧ (∀ x, -7 ≤ x → x ≤ -3 → f x ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_even_func_decreasing_on_neg_interval_l2154_215431


namespace NUMINAMATH_GPT_number_machine_output_l2154_215499

def number_machine (n : ℕ) : ℕ :=
  let step1 := n * 3
  let step2 := step1 + 20
  let step3 := step2 / 2
  let step4 := step3 ^ 2
  let step5 := step4 - 45
  step5

theorem number_machine_output : number_machine 90 = 20980 := by
  sorry

end NUMINAMATH_GPT_number_machine_output_l2154_215499


namespace NUMINAMATH_GPT_value_of_f_x_plus_5_l2154_215468

open Function

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem
theorem value_of_f_x_plus_5 (x : ℝ) : f (x + 5) = 3 * x + 16 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_x_plus_5_l2154_215468


namespace NUMINAMATH_GPT_cement_bought_l2154_215453

-- Define the three conditions given in the problem
def original_cement : ℕ := 98
def son_contribution : ℕ := 137
def total_cement : ℕ := 450

-- Using those conditions, state that the amount of cement he bought is 215 lbs
theorem cement_bought :
  original_cement + son_contribution = 235 ∧ total_cement - (original_cement + son_contribution) = 215 := 
by {
  sorry
}

end NUMINAMATH_GPT_cement_bought_l2154_215453


namespace NUMINAMATH_GPT_packs_sold_in_other_villages_l2154_215408

theorem packs_sold_in_other_villages
  (packs_v1 : ℕ) (packs_v2 : ℕ) (h1 : packs_v1 = 23) (h2 : packs_v2 = 28) :
  packs_v1 + packs_v2 = 51 := 
by {
  sorry
}

end NUMINAMATH_GPT_packs_sold_in_other_villages_l2154_215408


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_4_and_5_l2154_215455

theorem smallest_four_digit_divisible_by_4_and_5 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ ∀ m, 1000 ≤ m ∧ m < 10000 ∧ m % 4 = 0 ∧ m % 5 = 0 → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_4_and_5_l2154_215455


namespace NUMINAMATH_GPT_probability_reach_correct_l2154_215495

noncomputable def probability_reach (n : ℕ) : ℚ :=
  (2/3) + (1/12) * (1 - (-1/3)^(n-1))

theorem probability_reach_correct (n : ℕ) (P_n : ℚ) :
  P_n = probability_reach n :=
by
  sorry

end NUMINAMATH_GPT_probability_reach_correct_l2154_215495


namespace NUMINAMATH_GPT_eval_expr_l2154_215467

theorem eval_expr : (3 : ℚ) / (2 - (5 / 4)) = 4 := by
  sorry

end NUMINAMATH_GPT_eval_expr_l2154_215467


namespace NUMINAMATH_GPT_water_usage_in_May_l2154_215493

theorem water_usage_in_May (x : ℝ) (h_cost : 45 = if x ≤ 12 then 2 * x 
                                                else if x ≤ 18 then 24 + 2.5 * (x - 12) 
                                                else 39 + 3 * (x - 18)) : x = 20 :=
sorry

end NUMINAMATH_GPT_water_usage_in_May_l2154_215493


namespace NUMINAMATH_GPT_actual_distance_traveled_l2154_215486

theorem actual_distance_traveled (D : ℕ) (h : D / 10 = (D + 20) / 15) : D = 40 := 
sorry

end NUMINAMATH_GPT_actual_distance_traveled_l2154_215486


namespace NUMINAMATH_GPT_max_profit_price_l2154_215489

-- Define the initial conditions
def purchase_price : ℝ := 80
def initial_selling_price : ℝ := 90
def initial_sales_volume : ℝ := 400
def price_increase_effect : ℝ := 1
def sales_volume_decrease : ℝ := 20

-- Define the profit function
def profit (x : ℝ) : ℝ :=
  let selling_price := initial_selling_price + x
  let sales_volume := initial_sales_volume - x * sales_volume_decrease
  let profit_per_item := selling_price - purchase_price
  profit_per_item * sales_volume

-- The statement that needs to be proved
theorem max_profit_price : ∃ x : ℝ, x = 10 ∧ (initial_selling_price + x = 100) := by
  sorry

end NUMINAMATH_GPT_max_profit_price_l2154_215489


namespace NUMINAMATH_GPT_simplify_expression_l2154_215426

theorem simplify_expression : 3000 * (3000 ^ 3000) + 3000 * (3000 ^ 3000) = 2 * 3000 ^ 3001 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2154_215426


namespace NUMINAMATH_GPT_correct_relation_l2154_215405

-- Define the set A
def A : Set ℤ := { x | x^2 - 4 = 0 }

-- The statement that 2 is an element of A
theorem correct_relation : 2 ∈ A :=
by 
    -- We skip the proof here
    sorry

end NUMINAMATH_GPT_correct_relation_l2154_215405


namespace NUMINAMATH_GPT_sixteen_pow_five_eq_four_pow_p_l2154_215479

theorem sixteen_pow_five_eq_four_pow_p (p : ℕ) (h : 16^5 = 4^p) : p = 10 := 
  sorry

end NUMINAMATH_GPT_sixteen_pow_five_eq_four_pow_p_l2154_215479


namespace NUMINAMATH_GPT_john_saved_120_dollars_l2154_215496

-- Defining the conditions
def num_machines : ℕ := 10
def ball_bearings_per_machine : ℕ := 30
def total_ball_bearings : ℕ := num_machines * ball_bearings_per_machine
def regular_price_per_bearing : ℝ := 1
def sale_price_per_bearing : ℝ := 0.75
def bulk_discount : ℝ := 0.20
def discounted_price_per_bearing : ℝ := sale_price_per_bearing - (bulk_discount * sale_price_per_bearing)

-- Calculate total costs
def total_cost_without_sale : ℝ := total_ball_bearings * regular_price_per_bearing
def total_cost_with_sale : ℝ := total_ball_bearings * discounted_price_per_bearing

-- Calculate the savings
def savings : ℝ := total_cost_without_sale - total_cost_with_sale

-- The theorem we want to prove
theorem john_saved_120_dollars : savings = 120 := by
  sorry

end NUMINAMATH_GPT_john_saved_120_dollars_l2154_215496


namespace NUMINAMATH_GPT_n_n_plus_one_div_by_2_l2154_215472

theorem n_n_plus_one_div_by_2 (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 99) : 2 ∣ n * (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_n_n_plus_one_div_by_2_l2154_215472


namespace NUMINAMATH_GPT_parabola_directrix_l2154_215480

theorem parabola_directrix (p : ℝ) (h_focus : ∃ x y : ℝ, y^2 = 2*p*x ∧ 2*x + 3*y - 4 = 0) : 
  ∀ x y : ℝ, y^2 = 2*p*x → x = -p/2 := 
sorry

end NUMINAMATH_GPT_parabola_directrix_l2154_215480


namespace NUMINAMATH_GPT_larger_number_is_50_l2154_215470

theorem larger_number_is_50 (x y : ℤ) (h1 : 4 * y = 5 * x) (h2 : y - x = 10) : y = 50 :=
sorry

end NUMINAMATH_GPT_larger_number_is_50_l2154_215470


namespace NUMINAMATH_GPT_quadrilateral_property_indeterminate_l2154_215477

variable {α : Type*}
variable (Q A : α → Prop)

theorem quadrilateral_property_indeterminate :
  (¬ ∀ x, Q x → A x) → ¬ ((∃ x, Q x ∧ A x) ↔ False) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadrilateral_property_indeterminate_l2154_215477


namespace NUMINAMATH_GPT_total_value_correct_l2154_215418

-- Define conditions
def import_tax_rate : ℝ := 0.07
def tax_paid : ℝ := 109.90
def tax_exempt_value : ℝ := 1000

-- Define total value
def total_value (V : ℝ) : Prop :=
  V - tax_exempt_value = tax_paid / import_tax_rate

-- Theorem stating that the total value is $2570
theorem total_value_correct : total_value 2570 := by
  sorry

end NUMINAMATH_GPT_total_value_correct_l2154_215418


namespace NUMINAMATH_GPT_scale_division_l2154_215463

theorem scale_division (total_feet : ℕ) (inches_extra : ℕ) (part_length : ℕ) (total_parts : ℕ) :
  total_feet = 6 → inches_extra = 8 → part_length = 20 → 
  total_parts = (6 * 12 + 8) / 20 → total_parts = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_scale_division_l2154_215463


namespace NUMINAMATH_GPT_four_x_plus_t_odd_l2154_215413

theorem four_x_plus_t_odd (x t : ℤ) (hx : 2 * x - t = 11) : ¬(∃ n : ℤ, 4 * x + t = 2 * n) :=
by
  -- Since we need to prove the statement, we start a proof block
  sorry -- skipping the actual proof part for this statement

end NUMINAMATH_GPT_four_x_plus_t_odd_l2154_215413


namespace NUMINAMATH_GPT_arcsin_cos_arcsin_rel_arccos_sin_arccos_l2154_215440

theorem arcsin_cos_arcsin_rel_arccos_sin_arccos (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) :
    let α := Real.arcsin (Real.cos (Real.arcsin x))
    let β := Real.arccos (Real.sin (Real.arccos x))
    (Real.arcsin x + Real.arccos x = π / 2) → α + β = π / 2 :=
by
  let α := Real.arcsin (Real.cos (Real.arcsin x))
  let β := Real.arccos (Real.sin (Real.arccos x))
  intro h_arcsin_arccos_eq
  sorry

end NUMINAMATH_GPT_arcsin_cos_arcsin_rel_arccos_sin_arccos_l2154_215440


namespace NUMINAMATH_GPT_baseball_card_value_decrease_l2154_215484

theorem baseball_card_value_decrease (V0 : ℝ) (V1 V2 : ℝ) :
  V1 = V0 * 0.5 → V2 = V1 * 0.9 → (V0 - V2) / V0 * 100 = 55 :=
by 
  intros hV1 hV2
  sorry

end NUMINAMATH_GPT_baseball_card_value_decrease_l2154_215484


namespace NUMINAMATH_GPT_john_bought_3_tshirts_l2154_215459

theorem john_bought_3_tshirts (T : ℕ) (h : 20 * T + 50 = 110) : T = 3 := 
by 
  sorry

end NUMINAMATH_GPT_john_bought_3_tshirts_l2154_215459


namespace NUMINAMATH_GPT_rate_per_sq_meter_l2154_215422

theorem rate_per_sq_meter
  (length : Float := 9)
  (width : Float := 4.75)
  (total_cost : Float := 38475)
  : (total_cost / (length * width)) = 900 := 
by
  sorry

end NUMINAMATH_GPT_rate_per_sq_meter_l2154_215422


namespace NUMINAMATH_GPT_solutionSet_l2154_215433

def passesThroughQuadrants (a b : ℝ) : Prop :=
  a > 0

def intersectsXAxisAt (a b : ℝ) : Prop :=
  b = 2 * a

theorem solutionSet (a b x : ℝ) (hq : passesThroughQuadrants a b) (hi : intersectsXAxisAt a b) :
  (a * x > b) ↔ (x > 2) :=
by
  sorry

end NUMINAMATH_GPT_solutionSet_l2154_215433


namespace NUMINAMATH_GPT_paula_candies_l2154_215465

def candies_per_friend (total_candies : ℕ) (number_of_friends : ℕ) : ℕ :=
  total_candies / number_of_friends

theorem paula_candies :
  let initial_candies := 20
  let additional_candies := 4
  let total_candies := initial_candies + additional_candies
  let number_of_friends := 6
  candies_per_friend total_candies number_of_friends = 4 :=
by
  sorry

end NUMINAMATH_GPT_paula_candies_l2154_215465


namespace NUMINAMATH_GPT_parabola_equation_l2154_215444

variables (x y : ℝ)

def parabola_passes_through_point (x y : ℝ) : Prop :=
(x = 2 ∧ y = 7)

def focus_x_coord_five (x : ℝ) : Prop :=
(x = 5)

def axis_of_symmetry_parallel_to_y : Prop := True

def vertex_lies_on_x_axis (x y : ℝ) : Prop :=
(x = 5 ∧ y = 0)

theorem parabola_equation
  (h1 : parabola_passes_through_point x y)
  (h2 : focus_x_coord_five x)
  (h3 : axis_of_symmetry_parallel_to_y)
  (h4 : vertex_lies_on_x_axis x y) :
  49 * x + 3 * y^2 - 245 = 0
:= sorry

end NUMINAMATH_GPT_parabola_equation_l2154_215444


namespace NUMINAMATH_GPT_sum_of_number_and_square_is_306_l2154_215424

theorem sum_of_number_and_square_is_306 (n : ℕ) (h : n = 17) : n + n^2 = 306 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_number_and_square_is_306_l2154_215424


namespace NUMINAMATH_GPT_find_difference_l2154_215400

theorem find_difference (x y : ℕ) (hx : ∃ k : ℕ, x = k^2) (h_sum_prod : x + y = x * y - 2006) : y - x = 666 :=
sorry

end NUMINAMATH_GPT_find_difference_l2154_215400


namespace NUMINAMATH_GPT_angle_alpha_range_l2154_215402

/-- Given point P (tan α, sin α - cos α) is in the first quadrant, 
and 0 ≤ α ≤ 2π, then the range of values for angle α is (π/4, π/2) ∪ (π, 5π/4). -/
theorem angle_alpha_range (α : ℝ) 
  (h0 : 0 ≤ α) (h1 : α ≤ 2 * Real.pi) 
  (h2 : Real.tan α > 0) (h3 : Real.sin α - Real.cos α > 0) : 
  (Real.pi / 4 < α ∧ α < Real.pi / 2) ∨ 
  (Real.pi < α ∧ α < 5 * Real.pi / 4) :=
sorry

end NUMINAMATH_GPT_angle_alpha_range_l2154_215402


namespace NUMINAMATH_GPT_smallest_whole_number_l2154_215403

theorem smallest_whole_number (a b c d : ℤ)
  (h₁ : a = 3 + 1 / 3)
  (h₂ : b = 4 + 1 / 4)
  (h₃ : c = 5 + 1 / 6)
  (h₄ : d = 6 + 1 / 8)
  (h₅ : a + b + c + d - 2 > 16)
  (h₆ : a + b + c + d - 2 < 17) :
  17 > 16 + (a + b + c + d - 18) - 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 8 :=
  sorry

end NUMINAMATH_GPT_smallest_whole_number_l2154_215403


namespace NUMINAMATH_GPT_number_of_true_statements_l2154_215474

theorem number_of_true_statements 
  (a b c : ℝ) 
  (Hc : c ≠ 0) : 
  ((a > b → a * c^2 > b * c^2) ∧ (a * c^2 ≤ b * c^2 → a ≤ b)) ∧ 
  ¬((a * c^2 > b * c^2 → a > b) ∨ (a ≤ b → a * c^2 ≤ b * c^2)) :=
by
  sorry

end NUMINAMATH_GPT_number_of_true_statements_l2154_215474


namespace NUMINAMATH_GPT_tan_sum_half_l2154_215461

theorem tan_sum_half (a b : ℝ) (h1 : Real.cos a + Real.cos b = 3/5) (h2 : Real.sin a + Real.sin b = 1/5) :
  Real.tan ((a + b) / 2) = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_tan_sum_half_l2154_215461


namespace NUMINAMATH_GPT_sin_593_l2154_215478

theorem sin_593 (h : Real.sin (37 * Real.pi / 180) = 3/5) : 
  Real.sin (593 * Real.pi / 180) = -3/5 :=
by
sorry

end NUMINAMATH_GPT_sin_593_l2154_215478


namespace NUMINAMATH_GPT_fraction_product_eq_one_l2154_215416

theorem fraction_product_eq_one :
  (7 / 4 : ℚ) * (8 / 14) * (21 / 12) * (16 / 28) * (49 / 28) * (24 / 42) * (63 / 36) * (32 / 56) = 1 := by
  sorry

end NUMINAMATH_GPT_fraction_product_eq_one_l2154_215416


namespace NUMINAMATH_GPT_technician_completion_percentage_l2154_215429

noncomputable def percentage_completed (D : ℝ) : ℝ :=
  let total_distance := 2.20 * D
  let completed_distance := 1.12 * D
  (completed_distance / total_distance) * 100

theorem technician_completion_percentage (D : ℝ) (hD : D > 0) :
  percentage_completed D = 50.91 :=
by
  sorry

end NUMINAMATH_GPT_technician_completion_percentage_l2154_215429


namespace NUMINAMATH_GPT_magnitude_of_angle_B_value_of_k_l2154_215460

-- Define the conditions and corresponding proofs

variable {a b c : ℝ}
variable {A B C : ℝ} -- Angles in the triangle
variable (k : ℝ) -- Define k
variable (h1 : (2 * a - c) * Real.cos B = b * Real.cos C) -- Given condition for part 1
variable (h2 : (A + B + C) = Real.pi) -- Angle sum in triangle
variable (h3 : k > 1) -- Condition for part 2
variable (m_dot_n_max : ∀ (t : ℝ), 4 * k * t + Real.cos (2 * Real.arcsin t) = 5) -- Given condition for part 2

-- Proofs Required

theorem magnitude_of_angle_B (hA : 0 < A ∧ A < Real.pi) : B = Real.pi / 3 :=
by 
  sorry -- proof to be completed

theorem value_of_k : k = 3 / 2 :=
by 
  sorry -- proof to be completed

end NUMINAMATH_GPT_magnitude_of_angle_B_value_of_k_l2154_215460


namespace NUMINAMATH_GPT_soldiers_in_groups_l2154_215497

theorem soldiers_in_groups (x : ℕ) (h1 : x % 2 = 1) (h2 : x % 3 = 2) (h3 : x % 5 = 3) : x % 30 = 23 :=
by
  sorry

end NUMINAMATH_GPT_soldiers_in_groups_l2154_215497


namespace NUMINAMATH_GPT_total_weight_of_arrangement_l2154_215449

def original_side_length : ℤ := 4
def original_weight : ℤ := 16
def larger_side_length : ℤ := 10

theorem total_weight_of_arrangement :
  let original_area := original_side_length ^ 2
  let larger_area := larger_side_length ^ 2
  let number_of_pieces := (larger_area / original_area : ℤ)
  let total_weight := (number_of_pieces * original_weight)
  total_weight = 96 :=
by
  let original_area := original_side_length ^ 2
  let larger_area := larger_side_length ^ 2
  let number_of_pieces := (larger_area / original_area : ℤ)
  let total_weight := (number_of_pieces * original_weight)
  sorry

end NUMINAMATH_GPT_total_weight_of_arrangement_l2154_215449


namespace NUMINAMATH_GPT_triangle_area_proof_l2154_215483

noncomputable def segment_squared (a b : ℝ) : ℝ := a ^ 2 - b ^ 2

noncomputable def triangle_conditions (a b c : ℝ): Prop :=
  segment_squared b a = a ^ 2 - c ^ 2

noncomputable def area_triangle_OLK (r a b c : ℝ) (cond : triangle_conditions a b c): ℝ :=
  (a / (2 * Real.sqrt 3)) * Real.sqrt (r^2 - (a^2 / 3))

theorem triangle_area_proof (r a b c : ℝ) (cond : triangle_conditions a b c) :
  area_triangle_OLK r a b c cond = (a / (2 * Real.sqrt 3)) * Real.sqrt (r^2 - (a^2 / 3)) :=
sorry

end NUMINAMATH_GPT_triangle_area_proof_l2154_215483


namespace NUMINAMATH_GPT_total_items_deleted_l2154_215435

-- Define the initial conditions
def initial_apps : Nat := 17
def initial_files : Nat := 21
def remaining_apps : Nat := 3
def remaining_files : Nat := 7
def transferred_files : Nat := 4

-- Prove the total number of deleted items
theorem total_items_deleted : (initial_apps - remaining_apps) + (initial_files - (remaining_files + transferred_files)) = 24 :=
by
  sorry

end NUMINAMATH_GPT_total_items_deleted_l2154_215435


namespace NUMINAMATH_GPT_vector_CB_correct_l2154_215443

-- Define the vectors AB and AC
def AB : ℝ × ℝ := (2, 3)
def AC : ℝ × ℝ := (-1, 2)

-- Define the vector CB as the difference of AB and AC
def CB (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

-- Prove that CB = (3, 1) given AB and AC
theorem vector_CB_correct : CB AB AC = (3, 1) :=
by
  sorry

end NUMINAMATH_GPT_vector_CB_correct_l2154_215443


namespace NUMINAMATH_GPT_TruckCapacities_RentalPlanExists_MinimumRentalCost_l2154_215492

-- Problem 1
theorem TruckCapacities (x y : ℕ) (h1: 2 * x + y = 10) (h2: x + 2 * y = 11) :
  x = 3 ∧ y = 4 :=
by
  sorry

-- Problem 2
theorem RentalPlanExists (a b : ℕ) (h: 3 * a + 4 * b = 31) :
  (a = 9 ∧ b = 1) ∨ (a = 5 ∧ b = 4) ∨ (a = 1 ∧ b = 7) :=
by
  sorry

-- Problem 3
theorem MinimumRentalCost (a b : ℕ) (h1: 3 * a + 4 * b = 31) 
  (h2: 100 * a + 120 * b = 940) :
  ∃ a b, a = 1 ∧ b = 7 :=
by
  sorry

end NUMINAMATH_GPT_TruckCapacities_RentalPlanExists_MinimumRentalCost_l2154_215492


namespace NUMINAMATH_GPT_remaining_cooking_time_l2154_215457

-- Define the recommended cooking time in minutes and the time already cooked in seconds
def recommended_cooking_time_min := 5
def time_cooked_seconds := 45

-- Define the conversion from minutes to seconds
def minutes_to_seconds (min : Nat) : Nat := min * 60

-- Define the total recommended cooking time in seconds
def total_recommended_cooking_time_seconds := minutes_to_seconds recommended_cooking_time_min

-- State the theorem to prove the remaining cooking time
theorem remaining_cooking_time :
  (total_recommended_cooking_time_seconds - time_cooked_seconds) = 255 :=
by
  sorry

end NUMINAMATH_GPT_remaining_cooking_time_l2154_215457


namespace NUMINAMATH_GPT_fixed_cost_calculation_l2154_215423

theorem fixed_cost_calculation (TC MC n FC : ℕ) (h1 : TC = 16000) (h2 : MC = 200) (h3 : n = 20) (h4 : TC = FC + MC * n) : FC = 12000 :=
by
  sorry

end NUMINAMATH_GPT_fixed_cost_calculation_l2154_215423


namespace NUMINAMATH_GPT_square_side_length_l2154_215498

variable (s : ℝ)
variable (k : ℝ := 6)

theorem square_side_length :
  s^2 = k * 4 * s → s = 24 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_square_side_length_l2154_215498


namespace NUMINAMATH_GPT_jason_total_amount_l2154_215441

def shorts_price : ℝ := 14.28
def jacket_price : ℝ := 4.74
def shoes_price : ℝ := 25.95
def socks_price : ℝ := 6.80
def tshirts_price : ℝ := 18.36
def hat_price : ℝ := 12.50
def swimsuit_price : ℝ := 22.95
def sunglasses_price : ℝ := 45.60
def wristbands_price : ℝ := 9.80

def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def discounted_price (price : ℝ) (discount : ℝ) : ℝ := price - (price * discount)

def total_discounted_price : ℝ := 
  (discounted_price shorts_price discount1) + 
  (discounted_price jacket_price discount1) + 
  (discounted_price hat_price discount1) + 
  (discounted_price shoes_price discount2) + 
  (discounted_price socks_price discount2) + 
  (discounted_price tshirts_price discount2) + 
  (discounted_price swimsuit_price discount2) + 
  (discounted_price sunglasses_price discount2) + 
  (discounted_price wristbands_price discount2)

def total_with_tax : ℝ := total_discounted_price + (total_discounted_price * sales_tax_rate)

theorem jason_total_amount : total_with_tax = 153.07 := by
  sorry

end NUMINAMATH_GPT_jason_total_amount_l2154_215441


namespace NUMINAMATH_GPT_halfway_fraction_l2154_215448

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end NUMINAMATH_GPT_halfway_fraction_l2154_215448


namespace NUMINAMATH_GPT_room_length_l2154_215417

theorem room_length (L : ℝ) (width height door_area window_area cost_per_sq_ft total_cost : ℝ) 
    (num_windows : ℕ) (door_w window_w door_h window_h : ℝ)
    (h_width : width = 15) (h_height : height = 12) 
    (h_cost_per_sq_ft : cost_per_sq_ft = 9)
    (h_door_area : door_area = door_w * door_h)
    (h_window_area : window_area = window_w * window_h)
    (h_num_windows : num_windows = 3)
    (h_door_dim : door_w = 6 ∧ door_h = 3)
    (h_window_dim : window_w = 4 ∧ window_h = 3)
    (h_total_cost : total_cost = 8154) :
    (2 * height * (L + width) - (door_area + num_windows * window_area)) * cost_per_sq_ft = total_cost →
    L = 25 := 
by
  intros h_cost_eq
  sorry

end NUMINAMATH_GPT_room_length_l2154_215417


namespace NUMINAMATH_GPT_polynomial_solution_l2154_215432

theorem polynomial_solution (x : ℂ) (h : x^3 + x^2 + x + 1 = 0) : x^4 + 2 * x^3 + 2 * x^2 + 2 * x + 1 = 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_solution_l2154_215432


namespace NUMINAMATH_GPT_percentage_fractions_l2154_215450

theorem percentage_fractions : (3 / 8 / 100) * (160 : ℚ) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_fractions_l2154_215450


namespace NUMINAMATH_GPT_gcd_3375_9180_l2154_215447

-- Definition of gcd and the problem condition
theorem gcd_3375_9180 : Nat.gcd 3375 9180 = 135 := by
  sorry -- Proof can be filled in with the steps using the Euclidean algorithm

end NUMINAMATH_GPT_gcd_3375_9180_l2154_215447


namespace NUMINAMATH_GPT_jeff_corrected_mean_l2154_215406

def initial_scores : List ℕ := [85, 90, 87, 93, 89, 84, 88]

def corrected_scores : List ℕ := [85, 90, 92, 93, 89, 89, 88]

noncomputable def arithmetic_mean (scores : List ℕ) : ℝ :=
  (scores.sum : ℝ) / (scores.length : ℝ)

theorem jeff_corrected_mean :
  arithmetic_mean corrected_scores = 89.42857142857143 := 
by
  sorry

end NUMINAMATH_GPT_jeff_corrected_mean_l2154_215406


namespace NUMINAMATH_GPT_smallest_GCD_value_l2154_215488

theorem smallest_GCD_value (a b c d N : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
    (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : N > 5)
    (hc1 : Nat.gcd a b = 1 ∨ Nat.gcd a c = 1 ∨ Nat.gcd a d = 1 ∨ Nat.gcd b c = 1 ∨ Nat.gcd b d = 1 ∨ Nat.gcd c d = 1)
    (hc2 : Nat.gcd a b = 2 ∨ Nat.gcd a c = 2 ∨ Nat.gcd a d = 2 ∨ Nat.gcd b c = 2 ∨ Nat.gcd b d = 2 ∨ Nat.gcd c d = 2)
    (hc3 : Nat.gcd a b = 3 ∨ Nat.gcd a c = 3 ∨ Nat.gcd a d = 3 ∨ Nat.gcd b c = 3 ∨ Nat.gcd b d = 3 ∨ Nat.gcd c d = 3)
    (hc4 : Nat.gcd a b = 4 ∨ Nat.gcd a c = 4 ∨ Nat.gcd a d = 4 ∨ Nat.gcd b c = 4 ∨ Nat.gcd b d = 4 ∨ Nat.gcd c d = 4)
    (hc5 : Nat.gcd a b = 5 ∨ Nat.gcd a c = 5 ∨ Nat.gcd a d = 5 ∨ Nat.gcd b c = 5 ∨ Nat.gcd b d = 5 ∨ Nat.gcd c d = 5)
    (hcN : Nat.gcd a b = N ∨ Nat.gcd a c = N ∨ Nat.gcd a d = N ∨ Nat.gcd b c = N ∨ Nat.gcd b d = N ∨ Nat.gcd c d = N):
    N = 14 :=
sorry

end NUMINAMATH_GPT_smallest_GCD_value_l2154_215488


namespace NUMINAMATH_GPT_probability_one_side_is_side_of_decagon_l2154_215445

theorem probability_one_side_is_side_of_decagon :
  let decagon_vertices := 10
  let total_triangles := Nat.choose decagon_vertices 3
  let favorable_one_side :=
    decagon_vertices * (decagon_vertices - 3) / 2
  let favorable_two_sides := decagon_vertices
  let favorable_outcomes := favorable_one_side + favorable_two_sides
  let probability := favorable_outcomes / total_triangles
  total_triangles = 120 ∧ favorable_outcomes = 60 ∧ probability = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_probability_one_side_is_side_of_decagon_l2154_215445


namespace NUMINAMATH_GPT_sqrt_144000_simplified_l2154_215446

theorem sqrt_144000_simplified : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_GPT_sqrt_144000_simplified_l2154_215446


namespace NUMINAMATH_GPT_cos_alpha_add_beta_over_two_l2154_215436

theorem cos_alpha_add_beta_over_two (
  α β : ℝ) 
  (h1 : 0 < α ∧ α < (Real.pi / 2)) 
  (h2 : - (Real.pi / 2) < β ∧ β < 0) 
  (hcos1 : Real.cos (α + (Real.pi / 4)) = 1 / 3) 
  (hcos2 : Real.cos ((β / 2) - (Real.pi / 4)) = Real.sqrt 3 / 3) : 
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_GPT_cos_alpha_add_beta_over_two_l2154_215436


namespace NUMINAMATH_GPT_milk_cost_correct_l2154_215434

-- Definitions of the given conditions
def bagelCost : ℝ := 0.95
def orangeJuiceCost : ℝ := 0.85
def sandwichCost : ℝ := 4.65
def lunchExtraCost : ℝ := 4.0

-- Total cost of breakfast
def breakfastCost : ℝ := bagelCost + orangeJuiceCost

-- Total cost of lunch
def lunchCost : ℝ := breakfastCost + lunchExtraCost

-- Cost of milk
def milkCost : ℝ := lunchCost - sandwichCost

-- Theorem to prove the cost of milk
theorem milk_cost_correct : milkCost = 1.15 :=
by
  sorry

end NUMINAMATH_GPT_milk_cost_correct_l2154_215434


namespace NUMINAMATH_GPT_remainder_8357_to_8361_div_9_l2154_215473

theorem remainder_8357_to_8361_div_9 :
  (8357 + 8358 + 8359 + 8360 + 8361) % 9 = 3 := 
by
  sorry

end NUMINAMATH_GPT_remainder_8357_to_8361_div_9_l2154_215473


namespace NUMINAMATH_GPT_diana_age_is_8_l2154_215490

noncomputable def age_of_grace_last_year : ℕ := 3
noncomputable def age_of_grace_today : ℕ := age_of_grace_last_year + 1
noncomputable def age_of_diana_today : ℕ := 2 * age_of_grace_today

theorem diana_age_is_8 : age_of_diana_today = 8 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_diana_age_is_8_l2154_215490


namespace NUMINAMATH_GPT_unique_two_digit_integer_solution_l2154_215487

variable {s : ℕ}

-- Conditions
def is_two_digit_positive_integer (s : ℕ) : Prop :=
  10 ≤ s ∧ s < 100

def last_two_digits_of_13s_are_52 (s : ℕ) : Prop :=
  13 * s % 100 = 52

-- Theorem statement
theorem unique_two_digit_integer_solution (h1 : is_two_digit_positive_integer s)
                                          (h2 : last_two_digits_of_13s_are_52 s) :
  s = 4 :=
sorry

end NUMINAMATH_GPT_unique_two_digit_integer_solution_l2154_215487


namespace NUMINAMATH_GPT_combined_area_of_walls_l2154_215469

theorem combined_area_of_walls (A : ℕ) 
  (h1: ∃ (A : ℕ), A ≥ 0)
  (h2 : (A - 2 * 40 - 40 = 180)) :
  A = 300 := 
sorry

end NUMINAMATH_GPT_combined_area_of_walls_l2154_215469
