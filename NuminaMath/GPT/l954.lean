import Mathlib

namespace NUMINAMATH_GPT_annual_interest_rate_l954_95497

theorem annual_interest_rate (r : ℝ) :
  (6000 * r + 4000 * 0.09 = 840) → r = 0.08 :=
by sorry

end NUMINAMATH_GPT_annual_interest_rate_l954_95497


namespace NUMINAMATH_GPT_cannot_achieve_141_cents_l954_95442
-- Importing the required library

-- Definitions corresponding to types of coins and their values
def penny := 1
def nickel := 5
def dime := 10
def half_dollar := 50

-- The main statement to prove
theorem cannot_achieve_141_cents :
  ¬∃ (x y z : ℕ), x + y + z = 3 ∧ 
    x * penny + y * nickel + z * dime + (3 - x - y - z) * half_dollar = 141 := 
by
  -- Currently leaving the proof as a sorry
  sorry

end NUMINAMATH_GPT_cannot_achieve_141_cents_l954_95442


namespace NUMINAMATH_GPT_total_tennis_balls_used_l954_95405

theorem total_tennis_balls_used 
  (round1_games : Nat := 8) 
  (round2_games : Nat := 4) 
  (round3_games : Nat := 2) 
  (finals_games : Nat := 1)
  (cans_per_game : Nat := 5) 
  (balls_per_can : Nat := 3) : 

  3 * (5 * (8 + 4 + 2 + 1)) = 225 := 
by
  sorry

end NUMINAMATH_GPT_total_tennis_balls_used_l954_95405


namespace NUMINAMATH_GPT_fuel_used_l954_95425

theorem fuel_used (x : ℝ) (h1 : x + 0.8 * x = 27) : x = 15 :=
sorry

end NUMINAMATH_GPT_fuel_used_l954_95425


namespace NUMINAMATH_GPT_ratio_proof_l954_95480

theorem ratio_proof (a b x : ℝ) (h : a > b) (h_b_pos : b > 0)
  (h_x : x = 0.5 * Real.sqrt (a / b) + 0.5 * Real.sqrt (b / a)) :
  2 * b * Real.sqrt (x^2 - 1) / (x - Real.sqrt (x^2 - 1)) = a - b := 
sorry

end NUMINAMATH_GPT_ratio_proof_l954_95480


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l954_95462

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1_solution_set (a : ℝ) (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x | x ≤ 0} ∪ {x | x ≥ 5} :=
by 
  -- proof goes here
  sorry

theorem part2_range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : 
  a ≤ -3 ∨ a ≥ 5 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l954_95462


namespace NUMINAMATH_GPT_math_problem_l954_95426

theorem math_problem (a b c d x y : ℝ) (h1 : a = -b) (h2 : c * d = 1) 
  (h3 : (x + 3)^2 + |y - 2| = 0) : 2 * (a + b) - 2 * (c * d)^4 + (x + y)^2022 = -1 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l954_95426


namespace NUMINAMATH_GPT_find_solutions_l954_95419

noncomputable def solution_exists (x y z p : ℝ) : Prop :=
  (x^2 - 1 = p * (y + z)) ∧
  (y^2 - 1 = p * (z + x)) ∧
  (z^2 - 1 = p * (x + y))

theorem find_solutions (x y z p : ℝ) :
  solution_exists x y z p ↔
  (x = (p + Real.sqrt (p^2 + 1)) ∧ y = (p + Real.sqrt (p^2 + 1)) ∧ z = (p + Real.sqrt (p^2 + 1)) ∨
   x = (p - Real.sqrt (p^2 + 1)) ∧ y = (p - Real.sqrt (p^2 + 1)) ∧ z = (p - Real.sqrt (p^2 + 1))) ∨
  (x = (Real.sqrt (1 - p^2)) ∧ y = (Real.sqrt (1 - p^2)) ∧ z = (-p - Real.sqrt (1 - p^2)) ∨
   x = (-Real.sqrt (1 - p^2)) ∧ y = (-Real.sqrt (1 - p^2)) ∧ z = (-p + Real.sqrt (1 - p^2))) :=
by
  -- Proof starts here
  sorry

end NUMINAMATH_GPT_find_solutions_l954_95419


namespace NUMINAMATH_GPT_log_xy_eq_5_over_11_l954_95428

-- Definitions of the conditions
axiom log_xy4_eq_one {x y : ℝ} : Real.log (x * y^4) = 1
axiom log_x3y_eq_one {x y : ℝ} : Real.log (x^3 * y) = 1

-- The statement to be proven
theorem log_xy_eq_5_over_11 {x y : ℝ} (h1 : Real.log (x * y^4) = 1) (h2 : Real.log (x^3 * y) = 1) :
  Real.log (x * y) = 5 / 11 :=
by
  sorry

end NUMINAMATH_GPT_log_xy_eq_5_over_11_l954_95428


namespace NUMINAMATH_GPT_greatest_multiple_of_3_lt_1000_l954_95459

theorem greatest_multiple_of_3_lt_1000 :
  ∃ (x : ℕ), (x % 3 = 0) ∧ (x > 0) ∧ (x^3 < 1000) ∧ ∀ (y : ℕ), (y % 3 = 0) ∧ (y > 0) ∧ (y^3 < 1000) → y ≤ x := 
sorry

end NUMINAMATH_GPT_greatest_multiple_of_3_lt_1000_l954_95459


namespace NUMINAMATH_GPT_force_magnitudes_ratio_l954_95402

theorem force_magnitudes_ratio (a d : ℝ) (h1 : (a + 2 * d)^2 = a^2 + (a + d)^2) :
  ∃ k : ℝ, k > 0 ∧ (a + d) = a * (4 / 3) ∧ (a + 2 * d) = a * (5 / 3) :=
by
  sorry

end NUMINAMATH_GPT_force_magnitudes_ratio_l954_95402


namespace NUMINAMATH_GPT_find_principal_l954_95467

theorem find_principal
  (SI : ℝ)
  (R : ℝ)
  (T : ℝ)
  (h_SI : SI = 4025.25)
  (h_R : R = 0.09)
  (h_T : T = 5) : 
  (SI / (R * T / 100)) = 8950 :=
by
  rw [h_SI, h_R, h_T]
  sorry

end NUMINAMATH_GPT_find_principal_l954_95467


namespace NUMINAMATH_GPT_cabbage_price_l954_95440

theorem cabbage_price
  (earnings_wednesday : ℕ)
  (earnings_friday : ℕ)
  (earnings_today : ℕ)
  (total_weight : ℕ)
  (h1 : earnings_wednesday = 30)
  (h2 : earnings_friday = 24)
  (h3 : earnings_today = 42)
  (h4 : total_weight = 48) :
  (earnings_wednesday + earnings_friday + earnings_today) / total_weight = 2 := by
  sorry

end NUMINAMATH_GPT_cabbage_price_l954_95440


namespace NUMINAMATH_GPT_mixed_candy_price_l954_95473

noncomputable def price_per_pound (a b c : ℕ) (pa pb pc : ℝ) : ℝ :=
  (a * pa + b * pb + c * pc) / (a + b + c)

theorem mixed_candy_price :
  let a := 30
  let b := 15
  let c := 20
  let pa := 10.0
  let pb := 12.0
  let pc := 15.0
  price_per_pound a b c pa pb pc * 0.9 = 10.8 := by
  sorry

end NUMINAMATH_GPT_mixed_candy_price_l954_95473


namespace NUMINAMATH_GPT_find_c_work_rate_l954_95423

variables (A B C : ℚ)   -- Using rational numbers for the work rates

theorem find_c_work_rate (h1 : A + B = 1/3) (h2 : B + C = 1/4) (h3 : C + A = 1/6) : 
  C = 1/24 := 
sorry 

end NUMINAMATH_GPT_find_c_work_rate_l954_95423


namespace NUMINAMATH_GPT_garden_area_l954_95477

theorem garden_area (w l A : ℕ) (h1 : w = 12) (h2 : l = 3 * w) (h3 : A = l * w) : A = 432 := by
  sorry

end NUMINAMATH_GPT_garden_area_l954_95477


namespace NUMINAMATH_GPT_max_type_A_stationery_l954_95447

-- Define the variables and constraints
variables (x y : ℕ)

-- Define the conditions as hypotheses
def condition1 : Prop := 3 * x + 2 * (x - 2) + y = 66
def condition2 : Prop := 3 * x ≤ 33

-- The statement to prove
theorem max_type_A_stationery : condition1 x y ∧ condition2 x → x ≤ 11 :=
by sorry

end NUMINAMATH_GPT_max_type_A_stationery_l954_95447


namespace NUMINAMATH_GPT_xyz_solution_l954_95434

theorem xyz_solution (x y z : ℂ) (h1 : x * y + 5 * y = -20) 
                                 (h2 : y * z + 5 * z = -20) 
                                 (h3 : z * x + 5 * x = -20) :
  x * y * z = 200 / 3 := 
sorry

end NUMINAMATH_GPT_xyz_solution_l954_95434


namespace NUMINAMATH_GPT_shortest_side_15_l954_95466

theorem shortest_side_15 (b c : ℕ) (h : ℕ) (hb : b < c)
  (h_perimeter : 24 + b + c = 66)
  (h_area_int : ∃ A : ℕ, A*A = 33 * 9 * (33 - b) * (b - 9))
  (h_altitude_int : ∃ A : ℕ, 24 * h = 2 * A) : b = 15 :=
sorry

end NUMINAMATH_GPT_shortest_side_15_l954_95466


namespace NUMINAMATH_GPT_Jack_heavier_than_Sam_l954_95469

def total_weight := 96 -- total weight of Jack and Sam in pounds
def jack_weight := 52 -- Jack's weight in pounds

def sam_weight := total_weight - jack_weight

theorem Jack_heavier_than_Sam : jack_weight - sam_weight = 8 := by
  -- Here we would provide a proof, but we leave it as sorry for now.
  sorry

end NUMINAMATH_GPT_Jack_heavier_than_Sam_l954_95469


namespace NUMINAMATH_GPT_larger_number_is_38_l954_95424

theorem larger_number_is_38 (x y : ℕ) (h1 : x + y = 64) (h2 : y = x + 12) : y = 38 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_38_l954_95424


namespace NUMINAMATH_GPT_equilateral_triangle_perimeter_l954_95488

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_perimeter_l954_95488


namespace NUMINAMATH_GPT_units_digit_calculation_l954_95408

-- Define a function to compute the units digit of a number in base 10
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_calculation :
  units_digit (8 * 18 * 1988 - 8^3) = 0 := by
  sorry

end NUMINAMATH_GPT_units_digit_calculation_l954_95408


namespace NUMINAMATH_GPT_max_of_expression_l954_95446

theorem max_of_expression (a b c : ℝ) (hbc : b > c) (hca : c > a) (ha : a > 0) (hb : b > 0) (hc : c > 0) (ha_nonzero : a ≠ 0) :
  ∃ (max_val : ℝ), max_val = 44 ∧ (∀ x, x = (2*a + b)^2 + (b - 2*c)^2 + (c - a)^2 → x ≤ max_val) := 
sorry

end NUMINAMATH_GPT_max_of_expression_l954_95446


namespace NUMINAMATH_GPT_warehouse_problem_l954_95496

/-- 
Problem Statement:
A certain unit decides to invest 3200 yuan to build a warehouse (in the shape of a rectangular prism) with a constant height.
The back wall will be built reusing the old wall at no cost, the front will be made of iron grilles at a cost of 40 yuan per meter in length,
and the two side walls will be built with bricks at a cost of 45 yuan per meter in length.
The top will have a cost of 20 yuan per square meter.
Let the length of the iron grilles be x meters and the length of one brick wall be y meters.
Find:
1. Write down the relationship between x and y.
2. Determine the maximum allowable value of the warehouse area S. In order to maximize S without exceeding the budget, how long should the front iron grille be designed
-/

theorem warehouse_problem (x y : ℝ) :
    (40 * x + 90 * y + 20 * x * y = 3200 ∧ 0 < x ∧ x < 80) →
    (y = (320 - 4 * x) / (9 + 2 * x) ∧ x = 15 ∧ y = 20 / 3 ∧ x * y = 100) :=
by
  sorry

end NUMINAMATH_GPT_warehouse_problem_l954_95496


namespace NUMINAMATH_GPT_find_a_l954_95470

theorem find_a :
  ∃ a : ℝ, (2 * x - (a * Real.exp x + x) + 1 = 0) = (a = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l954_95470


namespace NUMINAMATH_GPT_randy_blocks_l954_95481

theorem randy_blocks (total_blocks house_blocks diff_blocks tower_blocks : ℕ) 
  (h_total : total_blocks = 90)
  (h_house : house_blocks = 89)
  (h_diff : house_blocks = tower_blocks + diff_blocks)
  (h_diff_value : diff_blocks = 26) :
  tower_blocks = 63 :=
by
  -- sorry is placed here to skip the proof.
  sorry

end NUMINAMATH_GPT_randy_blocks_l954_95481


namespace NUMINAMATH_GPT_PQRS_product_eq_one_l954_95444

noncomputable def P := Real.sqrt 2011 + Real.sqrt 2012
noncomputable def Q := -Real.sqrt 2011 - Real.sqrt 2012
noncomputable def R := Real.sqrt 2011 - Real.sqrt 2012
noncomputable def S := Real.sqrt 2012 - Real.sqrt 2011

theorem PQRS_product_eq_one : P * Q * R * S = 1 := by
  sorry

end NUMINAMATH_GPT_PQRS_product_eq_one_l954_95444


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l954_95458

theorem hyperbola_asymptotes (p : ℝ) (h : (p / 2, 0) ∈ {q : ℝ × ℝ | q.1 ^ 2 / 8 - q.2 ^ 2 / p = 1}) :
  (y = x) ∨ (y = -x) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l954_95458


namespace NUMINAMATH_GPT_total_cost_of_toys_l954_95431

def cost_of_toy_cars : ℝ := 14.88
def cost_of_toy_trucks : ℝ := 5.86

theorem total_cost_of_toys :
  cost_of_toy_cars + cost_of_toy_trucks = 20.74 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_toys_l954_95431


namespace NUMINAMATH_GPT_kenya_more_peanuts_l954_95457

-- Define the number of peanuts Jose has
def Jose_peanuts : ℕ := 85

-- Define the number of peanuts Kenya has
def Kenya_peanuts : ℕ := 133

-- The proof problem: Prove that Kenya has 48 more peanuts than Jose
theorem kenya_more_peanuts : Kenya_peanuts - Jose_peanuts = 48 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_kenya_more_peanuts_l954_95457


namespace NUMINAMATH_GPT_min_trials_to_ensure_pass_l954_95443

theorem min_trials_to_ensure_pass (p : ℝ) (n : ℕ) (h₁ : p = 3 / 4) (h₂ : n ≥ 1): 
  (1 - (1 - p) ^ n) > 0.99 → n ≥ 4 :=
by sorry

end NUMINAMATH_GPT_min_trials_to_ensure_pass_l954_95443


namespace NUMINAMATH_GPT_kimberly_initial_skittles_l954_95453

theorem kimberly_initial_skittles (total new initial : ℕ) (h1 : total = 12) (h2 : new = 7) (h3 : total = initial + new) : initial = 5 :=
by {
  -- Using the given conditions to form the proof
  sorry
}

end NUMINAMATH_GPT_kimberly_initial_skittles_l954_95453


namespace NUMINAMATH_GPT_arithmetic_sequence_k_value_l954_95445

theorem arithmetic_sequence_k_value 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) = a n + d) 
  (h_first_term : a 1 = 0) 
  (h_nonzero_diff : d ≠ 0) 
  (h_sum : ∃ k, a k = a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7) : 
  ∃ k, k = 22 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_k_value_l954_95445


namespace NUMINAMATH_GPT_ratio_of_side_lengths_of_frustum_l954_95433

theorem ratio_of_side_lengths_of_frustum (L1 L2 H : ℚ) (V_prism V_frustum : ℚ)
  (h1 : V_prism = L1^2 * H)
  (h2 : V_frustum = (1/3) * (L1^2 * (H * (L1 / (L1 - L2))) - L2^2 * (H * (L2 / (L1 - L2)))))
  (h3 : V_frustum = (2/3) * V_prism) :
  L1 / L2 = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_side_lengths_of_frustum_l954_95433


namespace NUMINAMATH_GPT_last_four_digits_of_5_pow_2011_l954_95404

theorem last_four_digits_of_5_pow_2011 :
  (5^2011) % 10000 = 8125 := 
by
  -- Using modular arithmetic and periodicity properties of powers of 5.
  sorry

end NUMINAMATH_GPT_last_four_digits_of_5_pow_2011_l954_95404


namespace NUMINAMATH_GPT_correct_choice_is_C_l954_95400

-- Define the proposition C.
def prop_C : Prop := ∃ x : ℝ, |x - 1| < 0

-- The problem statement in Lean 4.
theorem correct_choice_is_C : ¬ prop_C :=
by
  sorry

end NUMINAMATH_GPT_correct_choice_is_C_l954_95400


namespace NUMINAMATH_GPT_minimum_value_expr_l954_95439

theorem minimum_value_expr (x y : ℝ) : 
  (xy - 2)^2 + (x^2 + y^2)^2 ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_expr_l954_95439


namespace NUMINAMATH_GPT_minimum_value_ineq_l954_95435

theorem minimum_value_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
    (1 / (a + 2 * b)) + (1 / (b + 2 * c)) + (1 / (c + 2 * a)) ≥ 3 := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_ineq_l954_95435


namespace NUMINAMATH_GPT_intersection_eq_l954_95422

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem intersection_eq : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l954_95422


namespace NUMINAMATH_GPT_sheets_in_stack_l954_95485

theorem sheets_in_stack (thickness_per_500_sheets : ℝ) (stack_height : ℝ) (total_sheets : ℕ) :
  thickness_per_500_sheets = 4 → stack_height = 10 → total_sheets = 1250 :=
by
  intros h1 h2
  -- We will provide the mathematical proof steps here.
  sorry

end NUMINAMATH_GPT_sheets_in_stack_l954_95485


namespace NUMINAMATH_GPT_relationship_of_y_values_l954_95413

theorem relationship_of_y_values (b y1 y2 y3 : ℝ) (h1 : y1 = 3 * (-3) - b)
                                (h2 : y2 = 3 * 1 - b)
                                (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_of_y_values_l954_95413


namespace NUMINAMATH_GPT_incorrect_expression_l954_95468

variable {x y : ℚ}

theorem incorrect_expression (h : x / y = 5 / 3) : (x - 2 * y) / y ≠ 1 / 3 := by
  have h1 : x / y = 5 / 3 := h
  have h2 : (x - 2 * y) / y = (x / y) - (2 * y) / y := by sorry
  have h3 : (x - 2 * y) / y = (5 / 3) - 2 := by sorry
  have h4 : (x - 2 * y) / y = (5 / 3) - (6 / 3) := by sorry
  have h5 : (x - 2 * y) / y = -1 / 3 := by sorry
  exact sorry

end NUMINAMATH_GPT_incorrect_expression_l954_95468


namespace NUMINAMATH_GPT_train_meeting_distance_l954_95401

theorem train_meeting_distance :
  let distance := 150
  let time_x := 4
  let time_y := 3.5
  let speed_x := distance / time_x
  let speed_y := distance / time_y
  let relative_speed := speed_x + speed_y
  let time_to_meet := distance / relative_speed
  let distance_x_at_meeting := time_to_meet * speed_x
  distance_x_at_meeting = 70 := by
sorry

end NUMINAMATH_GPT_train_meeting_distance_l954_95401


namespace NUMINAMATH_GPT_minValue_l954_95465

theorem minValue (x y z : ℝ) (h : 1/x + 2/y + 3/z = 1) : x + y/2 + z/3 ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_minValue_l954_95465


namespace NUMINAMATH_GPT_largest_tangential_quadrilaterals_l954_95491

-- Definitions and conditions
def convex_ngon {n : ℕ} (h : n ≥ 5) : Type := sorry -- Placeholder for defining a convex n-gon with ≥ 5 sides
def tangential_quadrilateral {n : ℕ} (h : n ≥ 5) (k : ℕ) : Prop := 
  -- Placeholder for the property that exactly k quadrilaterals out of all possible ones 
  -- in a convex n-gon have an inscribed circle
  sorry

theorem largest_tangential_quadrilaterals {n : ℕ} (h : n ≥ 5) : 
  ∃ k : ℕ, tangential_quadrilateral h k ∧ k = n / 2 :=
sorry

end NUMINAMATH_GPT_largest_tangential_quadrilaterals_l954_95491


namespace NUMINAMATH_GPT_solve_for_h_l954_95464

-- Define the given polynomials
def p1 (x : ℝ) : ℝ := 2*x^5 + 4*x^3 - 3*x^2 + x + 7
def p2 (x : ℝ) : ℝ := -x^3 + 2*x^2 - 5*x + 4

-- Define h(x) as the unknown polynomial to solve for
def h (x : ℝ) : ℝ := -2*x^5 - x^3 + 5*x^2 - 6*x - 3

-- The theorem to prove
theorem solve_for_h : 
  (∀ (x : ℝ), p1 x + h x = p2 x) → (∀ (x : ℝ), h x = -2*x^5 - x^3 + 5*x^2 - 6*x - 3) :=
by
  intro h_cond
  sorry

end NUMINAMATH_GPT_solve_for_h_l954_95464


namespace NUMINAMATH_GPT_hotel_bill_amount_l954_95474

-- Definition of the variables used in the conditions
def each_paid : ℝ := 124.11
def friends : ℕ := 9

-- The Lean 4 theorem statement
theorem hotel_bill_amount :
  friends * each_paid = 1116.99 := sorry

end NUMINAMATH_GPT_hotel_bill_amount_l954_95474


namespace NUMINAMATH_GPT_find_angle_B_l954_95451

noncomputable def angle_B (a b c : ℝ) (A B C : ℝ) (h : b * Real.cos A - c * Real.cos B = (c - a) * Real.cos B) (h_sum : A + B + C = Real.pi) : ℝ :=
  B

theorem find_angle_B (a b c : ℝ) (A B C : ℝ) (h : b * Real.cos A - c * Real.cos B = (c - a) * Real.cos B) (h_sum : A + B + C = Real.pi) :
  B = Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_find_angle_B_l954_95451


namespace NUMINAMATH_GPT_jello_cost_l954_95475

def cost_to_fill_tub_with_jello (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : ℕ :=
  water_volume_cubic_feet * gallons_per_cubic_foot * pounds_per_gallon * tablespoons_per_pound * cost_per_tablespoon

theorem jello_cost (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : 
    water_volume_cubic_feet = 6 ∧ gallons_per_cubic_foot = 7 ∧ pounds_per_gallon = 8 ∧ 
    tablespoons_per_pound = 1 ∧ cost_per_tablespoon = 1 →
    cost_to_fill_tub_with_jello water_volume_cubic_feet gallons_per_cubic_foot pounds_per_gallon tablespoons_per_pound cost_per_tablespoon = 270 :=
  by 
    sorry

end NUMINAMATH_GPT_jello_cost_l954_95475


namespace NUMINAMATH_GPT_problem_statement_l954_95494

theorem problem_statement (
  a b c d x y z t : ℝ
) (habcd : 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1) 
  (hxyz : 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ 1 ≤ t)
  (h_sum : a + b + c + d + x + y + z + t = 8) :
  a^2 + b^2 + c^2 + d^2 + x^2 + y^2 + z^2 + t^2 ≤ 28 := 
sorry

end NUMINAMATH_GPT_problem_statement_l954_95494


namespace NUMINAMATH_GPT_find_a_l954_95498

noncomputable def f (x : ℝ) : ℝ := 5^(abs x)

noncomputable def g (a x : ℝ) : ℝ := a*x^2 - x

theorem find_a (a : ℝ) (h : f (g a 1) = 1) : a = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l954_95498


namespace NUMINAMATH_GPT_max_cables_used_eq_375_l954_95406

-- Conditions for the problem
def total_employees : Nat := 40
def brand_A_computers : Nat := 25
def brand_B_computers : Nat := 15

-- The main theorem we want to prove
theorem max_cables_used_eq_375 
  (h_employees : total_employees = 40)
  (h_brand_A_computers : brand_A_computers = 25)
  (h_brand_B_computers : brand_B_computers = 15)
  (cables_connectivity : ∀ (a : Fin brand_A_computers) (b : Fin brand_B_computers), Prop)
  (no_initial_connections : ∀ (a : Fin brand_A_computers) (b : Fin brand_B_computers), ¬ cables_connectivity a b)
  (each_brand_B_connected : ∀ (b : Fin brand_B_computers), ∃ (a : Fin brand_A_computers), cables_connectivity a b)
  : ∃ (n : Nat), n = 375 := 
sorry

end NUMINAMATH_GPT_max_cables_used_eq_375_l954_95406


namespace NUMINAMATH_GPT_problem_statement_l954_95490

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l954_95490


namespace NUMINAMATH_GPT_determine_digit_l954_95441

theorem determine_digit (Θ : ℕ) (hΘ : Θ > 0 ∧ Θ < 10) (h : 630 / Θ = 40 + 3 * Θ) : Θ = 9 :=
sorry

end NUMINAMATH_GPT_determine_digit_l954_95441


namespace NUMINAMATH_GPT_problem_inequality_l954_95448

theorem problem_inequality 
  (a b c : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) : 
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) :=
by
  sorry

end NUMINAMATH_GPT_problem_inequality_l954_95448


namespace NUMINAMATH_GPT_bamboo_pole_is_10_l954_95452

noncomputable def bamboo_pole_length (x : ℕ) : Prop :=
  (x - 4)^2 + (x - 2)^2 = x^2

theorem bamboo_pole_is_10 : bamboo_pole_length 10 :=
by
  -- The proof is not provided
  sorry

end NUMINAMATH_GPT_bamboo_pole_is_10_l954_95452


namespace NUMINAMATH_GPT_cylinder_volume_l954_95493

theorem cylinder_volume (r h : ℝ) (hr : r = 1) (hh : h = 1) : (π * r^2 * h) = π :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_l954_95493


namespace NUMINAMATH_GPT_initial_number_of_girls_l954_95499

theorem initial_number_of_girls (b g : ℤ) 
  (h1 : b = 3 * (g - 20)) 
  (h2 : 3 * (b - 30) = g - 20) : 
  g = 31 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_girls_l954_95499


namespace NUMINAMATH_GPT_total_onions_grown_l954_95455

-- Given conditions
def onions_grown_by_Nancy : ℕ := 2
def onions_grown_by_Dan : ℕ := 9
def onions_grown_by_Mike : ℕ := 4
def days_worked : ℕ := 6

-- Statement we need to prove
theorem total_onions_grown : onions_grown_by_Nancy + onions_grown_by_Dan + onions_grown_by_Mike = 15 :=
by sorry

end NUMINAMATH_GPT_total_onions_grown_l954_95455


namespace NUMINAMATH_GPT_correct_equation_l954_95410

theorem correct_equation (x Planned : ℝ) (h1 : 6 * x = Planned + 7) (h2 : 5 * x = Planned - 13) :
  6 * x - 7 = 5 * x + 13 :=
by
  sorry

end NUMINAMATH_GPT_correct_equation_l954_95410


namespace NUMINAMATH_GPT_statement_B_not_true_l954_95450

def op_star (x y : ℝ) := x^2 - 2*x*y + y^2

theorem statement_B_not_true (x y : ℝ) : 3 * (op_star x y) ≠ op_star (3 * x) (3 * y) :=
by
  have h1 : 3 * (op_star x y) = 3 * (x^2 - 2 * x * y + y^2) := rfl
  have h2 : op_star (3 * x) (3 * y) = (3 * x)^2 - 2 * (3 * x) * (3 * y) + (3 * y)^2 := rfl
  sorry

end NUMINAMATH_GPT_statement_B_not_true_l954_95450


namespace NUMINAMATH_GPT_compute_expression_l954_95412

theorem compute_expression : (-9 * 3 - (-7 * -4) + (-11 * -6) = 11) := by
  sorry

end NUMINAMATH_GPT_compute_expression_l954_95412


namespace NUMINAMATH_GPT_find_a_value_l954_95482

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x + 1)

theorem find_a_value :
  ∃ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a 0 + f a 1 = a) → a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_value_l954_95482


namespace NUMINAMATH_GPT_parameterized_line_l954_95461

noncomputable def g (t : ℝ) : ℝ := 9 * t + 10

theorem parameterized_line (t : ℝ) :
  let x := g t
  let y := 18 * t - 10
  y = 2 * x - 30 :=
by
  sorry

end NUMINAMATH_GPT_parameterized_line_l954_95461


namespace NUMINAMATH_GPT_set_d_pythagorean_triple_l954_95476

theorem set_d_pythagorean_triple : (9^2 + 40^2 = 41^2) :=
by sorry

end NUMINAMATH_GPT_set_d_pythagorean_triple_l954_95476


namespace NUMINAMATH_GPT_sara_schavenger_hunt_l954_95483

theorem sara_schavenger_hunt :
  let monday := 1 -- Sara rearranges the books herself
  let tuesday := 2 -- Sara can choose from Liam or Mia
  let wednesday := 4 -- There are 4 classmates
  let thursday := 3 -- There are 3 new volunteers
  let friday := 1 -- Sara and Zoe do it together
  monday * tuesday * wednesday * thursday * friday = 24 :=
by
  sorry

end NUMINAMATH_GPT_sara_schavenger_hunt_l954_95483


namespace NUMINAMATH_GPT_solve_problem_l954_95432

theorem solve_problem (nabla odot : ℕ) 
  (h1 : 0 < nabla) 
  (h2 : nabla < 20) 
  (h3 : 0 < odot) 
  (h4 : odot < 20) 
  (h5 : nabla ≠ odot) 
  (h6 : nabla * nabla * nabla = nabla) : 
  nabla * nabla = 64 :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l954_95432


namespace NUMINAMATH_GPT_value_of_a_l954_95438

theorem value_of_a (a : ℝ) (x y : ℝ) : 
  (x + a^2 * y + 6 = 0 ∧ (a - 2) * x + 3 * a * y + 2 * a = 0) ↔ a = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l954_95438


namespace NUMINAMATH_GPT_circle_passing_through_pole_l954_95456

noncomputable def equation_of_circle (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sqrt 2 * Real.cos θ

theorem circle_passing_through_pole :
  equation_of_circle 2 θ := 
sorry

end NUMINAMATH_GPT_circle_passing_through_pole_l954_95456


namespace NUMINAMATH_GPT_expression_pos_intervals_l954_95479

theorem expression_pos_intervals :
  ∀ x : ℝ, (x > -1 ∧ x < 1) ∨ (x > 3) ↔ (x + 1) * (x - 1) * (x - 3) > 0 := by
  sorry

end NUMINAMATH_GPT_expression_pos_intervals_l954_95479


namespace NUMINAMATH_GPT_find_next_score_l954_95415

def scores := [95, 85, 75, 65, 90]
def current_avg := (95 + 85 + 75 + 65 + 90) / 5
def target_avg := current_avg + 4

theorem find_next_score (s : ℕ) (h : (95 + 85 + 75 + 65 + 90 + s) / 6 = target_avg) : s = 106 :=
by
  -- Proof steps here
  sorry

end NUMINAMATH_GPT_find_next_score_l954_95415


namespace NUMINAMATH_GPT_table_covered_with_three_layers_l954_95430

theorem table_covered_with_three_layers (A T table_area two_layers : ℕ)
    (hA : A = 204)
    (htable : table_area = 175)
    (hcover : 140 = 80 * table_area / 100)
    (htwo_layers : two_layers = 24) :
    3 * T + 2 * two_layers + (140 - two_layers - T) = 204 → T = 20 := by
  sorry

end NUMINAMATH_GPT_table_covered_with_three_layers_l954_95430


namespace NUMINAMATH_GPT_ram_first_year_balance_l954_95437

-- Given conditions
def initial_deposit : ℝ := 1000
def interest_first_year : ℝ := 100

-- Calculate end of the first year balance
def balance_first_year := initial_deposit + interest_first_year

-- Prove that balance_first_year is $1100
theorem ram_first_year_balance :
  balance_first_year = 1100 :=
by 
  sorry

end NUMINAMATH_GPT_ram_first_year_balance_l954_95437


namespace NUMINAMATH_GPT_product_of_sequence_is_256_l954_95487

-- Definitions for conditions
def seq : List ℚ := [1 / 4, 16 / 1, 1 / 64, 256 / 1, 1 / 1024, 4096 / 1, 1 / 16384, 65536 / 1]

-- The main theorem
theorem product_of_sequence_is_256 : (seq.prod = 256) :=
by
  sorry

end NUMINAMATH_GPT_product_of_sequence_is_256_l954_95487


namespace NUMINAMATH_GPT_rowing_speed_still_water_l954_95486

theorem rowing_speed_still_water (v r : ℕ) (h1 : r = 18) (h2 : 1 / (v - r) = 3 * (1 / (v + r))) : v = 36 :=
by sorry

end NUMINAMATH_GPT_rowing_speed_still_water_l954_95486


namespace NUMINAMATH_GPT_solve_for_a_l954_95417

open Complex

noncomputable def question (a : ℝ) : Prop :=
  ∃ z : ℂ, z = (a + I) / (1 - I) ∧ z.im ≠ 0 ∧ z.re = 0

theorem solve_for_a (a : ℝ) (h : question a) : a = 1 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l954_95417


namespace NUMINAMATH_GPT_prob_male_given_obese_correct_l954_95478

-- Definitions based on conditions
def ratio_male_female : ℚ := 3 / 2
def prob_obese_male : ℚ := 1 / 5
def prob_obese_female : ℚ := 1 / 10

-- Definition of events
def total_employees : ℚ := ratio_male_female + 1

-- Probability calculations
def prob_male : ℚ := ratio_male_female / total_employees
def prob_female : ℚ := 1 / total_employees

def prob_obese_and_male : ℚ := prob_male * prob_obese_male
def prob_obese_and_female : ℚ := prob_female * prob_obese_female

def prob_obese : ℚ := prob_obese_and_male + prob_obese_and_female

def prob_male_given_obese : ℚ := prob_obese_and_male / prob_obese

-- Theorem statement
theorem prob_male_given_obese_correct : prob_male_given_obese = 3 / 4 := sorry

end NUMINAMATH_GPT_prob_male_given_obese_correct_l954_95478


namespace NUMINAMATH_GPT_train_speed_on_time_l954_95454

theorem train_speed_on_time :
  ∃ (v : ℝ), 
  (∀ (d : ℝ) (t : ℝ),
    d = 133.33 ∧ 
    80 * (t + 1/3) = d ∧ 
    v * t = d) → 
  v = 100 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_on_time_l954_95454


namespace NUMINAMATH_GPT_henry_total_cost_l954_95492

def henry_initial_figures : ℕ := 3
def henry_total_needed_figures : ℕ := 15
def cost_per_figure : ℕ := 12

theorem henry_total_cost :
  (henry_total_needed_figures - henry_initial_figures) * cost_per_figure = 144 :=
by
  sorry

end NUMINAMATH_GPT_henry_total_cost_l954_95492


namespace NUMINAMATH_GPT_number_of_adult_males_l954_95463

def population := 480
def ratio_children := 1
def ratio_adult_males := 2
def ratio_adult_females := 2
def total_ratio_parts := ratio_children + ratio_adult_males + ratio_adult_females

theorem number_of_adult_males : 
  (population / total_ratio_parts) * ratio_adult_males = 192 :=
by
  sorry

end NUMINAMATH_GPT_number_of_adult_males_l954_95463


namespace NUMINAMATH_GPT_train_passes_in_two_minutes_l954_95449

noncomputable def time_to_pass_through_tunnel : ℕ := 
  let train_length := 100 -- Length of the train in meters
  let train_speed := 72 * 1000 / 60 -- Speed of the train in m/min (converted)
  let tunnel_length := 2300 -- Length of the tunnel in meters (converted from 2.3 km to meters)
  let total_distance := train_length + tunnel_length -- Total distance to travel
  total_distance / train_speed -- Time in minutes (total distance divided by speed)

theorem train_passes_in_two_minutes : time_to_pass_through_tunnel = 2 := 
  by
  -- proof would go here, but for this statement, we use 'sorry'
  sorry

end NUMINAMATH_GPT_train_passes_in_two_minutes_l954_95449


namespace NUMINAMATH_GPT_gcd_2703_1113_l954_95429

theorem gcd_2703_1113 : Nat.gcd 2703 1113 = 159 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_2703_1113_l954_95429


namespace NUMINAMATH_GPT_fraction_is_five_sixths_l954_95403

-- Define the conditions as given in the problem
def number : ℝ := -72.0
def target_value : ℝ := -60

-- The statement we aim to prove
theorem fraction_is_five_sixths (f : ℝ) (h : f * number = target_value) : f = 5/6 :=
  sorry

end NUMINAMATH_GPT_fraction_is_five_sixths_l954_95403


namespace NUMINAMATH_GPT_no_positive_integer_n_has_perfect_square_form_l954_95495

theorem no_positive_integer_n_has_perfect_square_form (n : ℕ) (h : 0 < n) : 
  ¬ ∃ k : ℕ, n^4 + 2 * n^3 + 2 * n^2 + 2 * n + 1 = k^2 := 
sorry

end NUMINAMATH_GPT_no_positive_integer_n_has_perfect_square_form_l954_95495


namespace NUMINAMATH_GPT_simplify_and_evaluate_l954_95472

-- Definitions and conditions 
def x := ℝ
def given_condition (x: ℝ) : Prop := x + 2 = Real.sqrt 2

-- The problem statement translated into Lean 4
theorem simplify_and_evaluate (x: ℝ) (h: given_condition x) :
  ((x^2 + 1) / x + 2) / ((x - 3) * (x + 1) / (x^2 - 3 * x)) = Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l954_95472


namespace NUMINAMATH_GPT_compute_expression_l954_95489

theorem compute_expression :
  2 * 2^5 - 8^58 / 8^56 = 0 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l954_95489


namespace NUMINAMATH_GPT_seashells_left_sam_seashells_now_l954_95418

-- Problem conditions
def initial_seashells : ℕ := 35
def seashells_given : ℕ := 18

-- Proof problem statement
theorem seashells_left (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

-- The required statement
theorem sam_seashells_now : seashells_left initial_seashells seashells_given = 17 := by 
  sorry

end NUMINAMATH_GPT_seashells_left_sam_seashells_now_l954_95418


namespace NUMINAMATH_GPT_valentine_day_spending_l954_95421

structure DogTreatsConfig where
  heart_biscuits_count_A : Nat
  puppy_boots_count_A : Nat
  small_toy_count_A : Nat
  heart_biscuits_count_B : Nat
  puppy_boots_count_B : Nat
  large_toy_count_B : Nat
  heart_biscuit_price : Nat
  puppy_boots_price : Nat
  small_toy_price : Nat
  large_toy_price : Nat
  heart_biscuits_discount : Float
  large_toy_discount : Float

def treats_config : DogTreatsConfig :=
  { heart_biscuits_count_A := 5
    puppy_boots_count_A := 1
    small_toy_count_A := 1
    heart_biscuits_count_B := 7
    puppy_boots_count_B := 2
    large_toy_count_B := 1
    heart_biscuit_price := 2
    puppy_boots_price := 15
    small_toy_price := 10
    large_toy_price := 20
    heart_biscuits_discount := 0.20
    large_toy_discount := 0.15 }

def total_discounted_amount_spent (cfg : DogTreatsConfig) : Float :=
  let heart_biscuits_total_cost := (cfg.heart_biscuits_count_A + cfg.heart_biscuits_count_B) * cfg.heart_biscuit_price
  let puppy_boots_total_cost := (cfg.puppy_boots_count_A * cfg.puppy_boots_price) + (cfg.puppy_boots_count_B * cfg.puppy_boots_price)
  let small_toy_total_cost := cfg.small_toy_count_A * cfg.small_toy_price
  let large_toy_total_cost := cfg.large_toy_count_B * cfg.large_toy_price
  let total_cost_without_discount := Float.ofNat (heart_biscuits_total_cost + puppy_boots_total_cost + small_toy_total_cost + large_toy_total_cost)
  let heart_biscuits_discount_amount := cfg.heart_biscuits_discount * Float.ofNat heart_biscuits_total_cost
  let large_toy_discount_amount := cfg.large_toy_discount * Float.ofNat large_toy_total_cost
  let total_discount_amount := heart_biscuits_discount_amount + large_toy_discount_amount
  total_cost_without_discount - total_discount_amount

theorem valentine_day_spending : total_discounted_amount_spent treats_config = 91.20 := by
  sorry

end NUMINAMATH_GPT_valentine_day_spending_l954_95421


namespace NUMINAMATH_GPT_find_a_l954_95414

theorem find_a
  (x1 x2 a : ℝ)
  (h1 : x1^2 + 4 * x1 - 3 = 0)
  (h2 : x2^2 + 4 * x2 - 3 = 0)
  (h3 : 2 * x1 * (x2^2 + 3 * x2 - 3) + a = 2) :
  a = -4 :=
sorry

end NUMINAMATH_GPT_find_a_l954_95414


namespace NUMINAMATH_GPT_incorrect_transformation_l954_95427

theorem incorrect_transformation :
  ¬ ∀ (a b c : ℝ), ac = bc → a = b :=
by
  sorry

end NUMINAMATH_GPT_incorrect_transformation_l954_95427


namespace NUMINAMATH_GPT_xy_inequality_l954_95416

theorem xy_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = 2) : 
  x^2 * y^2 * (x^2 + y^2) ≤ 2 := 
sorry

end NUMINAMATH_GPT_xy_inequality_l954_95416


namespace NUMINAMATH_GPT_angle_A_in_triangle_l954_95436

noncomputable def is_angle_A (a b : ℝ) (B A: ℝ) : Prop :=
  a = 2 * Real.sqrt 3 ∧ b = 2 * Real.sqrt 2 ∧ B = Real.pi / 4 ∧
  (A = Real.pi / 3 ∨ A = 2 * Real.pi / 3)

theorem angle_A_in_triangle (a b A B : ℝ) (h : is_angle_A a b B A) : A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_angle_A_in_triangle_l954_95436


namespace NUMINAMATH_GPT_min_sum_log_geq_four_l954_95484

theorem min_sum_log_geq_four (m n : ℝ) (hm : 0 < m) (hn : 0 < n) 
  (hlog : Real.log m / Real.log 3 + Real.log n / Real.log 3 ≥ 4) : 
  m + n ≥ 18 :=
sorry

end NUMINAMATH_GPT_min_sum_log_geq_four_l954_95484


namespace NUMINAMATH_GPT_a_n_bound_l954_95407

theorem a_n_bound (a : ℕ → ℝ) :
  a 1 = 1 ∧ (∀ m n : ℕ, 0 < m ∧ 0 < n → (m + n) * a (m + n) ≤ a m + a n) →
  1 / a 200 > 4 * 10^7 := 
sorry

end NUMINAMATH_GPT_a_n_bound_l954_95407


namespace NUMINAMATH_GPT_inequality_abc_l954_95460

theorem inequality_abc (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a^2 + b^2 = 1/2) :
  (1 / (1 - a) + 1 / (1 - b) >= 4)
  ∧ ((1 / (1 - a) + 1 / (1 - b) = 4) ↔ (a = 1/2 ∧ b = 1/2)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l954_95460


namespace NUMINAMATH_GPT_total_lines_to_write_l954_95411

theorem total_lines_to_write (lines_per_page pages_needed : ℕ) (h1 : lines_per_page = 30) (h2 : pages_needed = 5) : lines_per_page * pages_needed = 150 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_lines_to_write_l954_95411


namespace NUMINAMATH_GPT_B_investment_time_l954_95471

theorem B_investment_time (x : ℝ) (m : ℝ) :
  let A_share := x * 12
  let B_share := 2 * x * (12 - m)
  let C_share := 3 * x * 4
  let total_gain := 18600
  let A_gain := 6200
  let ratio := A_gain / total_gain
  ratio = 1 / 3 →
  A_share = 1 / 3 * (A_share + B_share + C_share) →
  m = 6 := by
sorry

end NUMINAMATH_GPT_B_investment_time_l954_95471


namespace NUMINAMATH_GPT_p_sq_plus_q_sq_l954_95409

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 := 
by 
  sorry

end NUMINAMATH_GPT_p_sq_plus_q_sq_l954_95409


namespace NUMINAMATH_GPT_lateral_surface_area_truncated_cone_l954_95420

theorem lateral_surface_area_truncated_cone :
  let r := 1
  let R := 4
  let h := 4
  let l := Real.sqrt ((R - r)^2 + h^2)
  let S := Real.pi * (r + R) * l
  S = 25 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_truncated_cone_l954_95420
