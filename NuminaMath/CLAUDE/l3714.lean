import Mathlib

namespace NUMINAMATH_CALUDE_least_k_correct_l3714_371447

/-- Sum of reciprocal values of non-zero digits of all positive integers up to and including n -/
def S (n : ℕ) : ℚ := sorry

/-- The least positive integer k such that k! * S_2016 is an integer -/
def least_k : ℕ := 7

theorem least_k_correct :
  (∀ m : ℕ, m < least_k → ¬(∃ z : ℤ, z = (m.factorial : ℚ) * S 2016)) ∧
  (∃ z : ℤ, z = (least_k.factorial : ℚ) * S 2016) := by sorry

end NUMINAMATH_CALUDE_least_k_correct_l3714_371447


namespace NUMINAMATH_CALUDE_circle_tangency_distance_l3714_371478

theorem circle_tangency_distance (r_O r_O' d_external : ℝ) : 
  r_O = 5 → 
  d_external = 9 → 
  r_O + r_O' = d_external → 
  |r_O' - r_O| = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_distance_l3714_371478


namespace NUMINAMATH_CALUDE_shopkeeper_profit_days_l3714_371427

/-- Proves that given the specified mean profits, the total number of days is 30 -/
theorem shopkeeper_profit_days : 
  ∀ (total_days : ℕ) (mean_profit mean_first_15 mean_last_15 : ℚ),
  mean_profit = 350 →
  mean_first_15 = 225 →
  mean_last_15 = 475 →
  mean_profit * total_days = mean_first_15 * 15 + mean_last_15 * 15 →
  total_days = 30 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_days_l3714_371427


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3714_371480

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 - 2*x₁ - 8 = 0) ∧ 
  (x₂^2 - 2*x₂ - 8 = 0) ∧ 
  x₁ = 4 ∧ 
  x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3714_371480


namespace NUMINAMATH_CALUDE_no_solution_exists_l3714_371451

theorem no_solution_exists : ¬ ∃ (a b c d : ℤ), a^4 + b^4 + c^4 + 2016 = 10*d := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3714_371451


namespace NUMINAMATH_CALUDE_max_triangles_correct_l3714_371494

/-- The number of points on the hypotenuse of the right triangle -/
def num_points : ℕ := 8

/-- The maximum number of triangles that can be formed -/
def max_triangles : ℕ := 28

/-- Theorem stating that the maximum number of triangles is correct -/
theorem max_triangles_correct :
  (num_points.choose 2) = max_triangles := by sorry

end NUMINAMATH_CALUDE_max_triangles_correct_l3714_371494


namespace NUMINAMATH_CALUDE_sum_floor_value_l3714_371467

theorem sum_floor_value (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_squares : a^2 + b^2 = 2016 ∧ c^2 + d^2 = 2016)
  (products : a * c = 1024 ∧ b * d = 1024) : 
  ⌊a + b + c + d⌋ = 127 := by
sorry

end NUMINAMATH_CALUDE_sum_floor_value_l3714_371467


namespace NUMINAMATH_CALUDE_min_sum_abc_l3714_371460

-- Define the properties of a, b, and c
def is_valid_abc (a b c : ℕ+) : Prop :=
  a * b * c = 2310 ∧ ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ c = p * q

-- State the theorem
theorem min_sum_abc :
  ∀ a b c : ℕ+, is_valid_abc a b c →
  ∀ a' b' c' : ℕ+, is_valid_abc a' b' c' →
  a + b + c ≤ a' + b' + c' ∧
  ∃ a₀ b₀ c₀ : ℕ+, is_valid_abc a₀ b₀ c₀ ∧ a₀ + b₀ + c₀ = 88 :=
sorry

end NUMINAMATH_CALUDE_min_sum_abc_l3714_371460


namespace NUMINAMATH_CALUDE_expand_product_l3714_371462

theorem expand_product (x : ℝ) : (2*x + 3) * (x + 6) = 2*x^2 + 15*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3714_371462


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3714_371413

def A : Set ℝ := {x : ℝ | x^2 - x - 2 = 0}

def B : Set ℝ := {y : ℝ | ∃ x ∈ A, y = x + 3}

theorem union_of_A_and_B : A ∪ B = {-1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3714_371413


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_l3714_371414

theorem product_of_roots_cubic (x : ℝ) : 
  (∃ p q r : ℝ, x^3 - 9*x^2 + 27*x - 5 = (x - p) * (x - q) * (x - r)) → 
  (∃ p q r : ℝ, x^3 - 9*x^2 + 27*x - 5 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 5) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_l3714_371414


namespace NUMINAMATH_CALUDE_equation_roots_l3714_371465

theorem equation_roots : 
  {x : ℝ | (x + 1) * (x - 2) = x + 1} = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_equation_roots_l3714_371465


namespace NUMINAMATH_CALUDE_oil_price_reduction_l3714_371434

/-- Proves that given a 40% reduction in oil price, if 8 kg more oil can be bought for Rs. 2400 after the reduction, then the reduced price per kg is Rs. 120. -/
theorem oil_price_reduction (original_price : ℝ) : 
  let reduced_price := original_price * 0.6
  let original_quantity := 2400 / original_price
  let new_quantity := 2400 / reduced_price
  (new_quantity - original_quantity = 8) → reduced_price = 120 := by
  sorry


end NUMINAMATH_CALUDE_oil_price_reduction_l3714_371434


namespace NUMINAMATH_CALUDE_unique_triple_divisibility_l3714_371491

theorem unique_triple_divisibility (a b c : ℤ) :
  1 < a ∧ a < b ∧ b < c ∧ 
  (a * b - 1) * (b * c - 1) * (c * a - 1) % (a * b * c) = 0 →
  a = 2 ∧ b = 3 ∧ c = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_triple_divisibility_l3714_371491


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_in_binomial_difference_x_cubed_coefficient_is_negative_ten_l3714_371433

theorem x_cubed_coefficient_in_binomial_difference : ℤ :=
  let n₁ : ℕ := 5
  let n₂ : ℕ := 6
  let k : ℕ := 3
  let coeff₁ : ℤ := (Nat.choose n₁ k : ℤ)
  let coeff₂ : ℤ := (Nat.choose n₂ k : ℤ)
  coeff₁ - coeff₂

theorem x_cubed_coefficient_is_negative_ten :
  x_cubed_coefficient_in_binomial_difference = -10 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_in_binomial_difference_x_cubed_coefficient_is_negative_ten_l3714_371433


namespace NUMINAMATH_CALUDE_x_plus_y_equals_fifteen_l3714_371439

theorem x_plus_y_equals_fifteen (x y : ℝ) 
  (h1 : (3 : ℝ)^x = 27^(y + 1)) 
  (h2 : (16 : ℝ)^y = 4^(x - 6)) : 
  x + y = 15 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_fifteen_l3714_371439


namespace NUMINAMATH_CALUDE_max_product_constraint_l3714_371445

theorem max_product_constraint (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 40) :
  x * y ≤ 400 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l3714_371445


namespace NUMINAMATH_CALUDE_second_smallest_pack_count_l3714_371485

def hot_dogs_per_pack : ℕ := 12
def buns_per_pack : ℕ := 10
def leftover_hot_dogs : ℕ := 6

def is_valid_pack_count (n : ℕ) : Prop :=
  (hot_dogs_per_pack * n) % buns_per_pack = leftover_hot_dogs

theorem second_smallest_pack_count : 
  ∃ (n : ℕ), is_valid_pack_count n ∧ 
    (∃ (m : ℕ), m < n ∧ is_valid_pack_count m) ∧
    (∀ (k : ℕ), k < n → is_valid_pack_count k → k ≤ m) ∧
    n = 8 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_pack_count_l3714_371485


namespace NUMINAMATH_CALUDE_bug_positions_l3714_371454

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Set of positions reachable by the bug in at most n steps -/
def reachablePositions (n : ℕ) : Set ℚ :=
  {x | ∃ (k : ℕ), k ≤ n ∧ ∃ (steps : List (ℚ → ℚ)),
    steps.length = k ∧
    (∀ step ∈ steps, step = (· + 2) ∨ step = (· / 2)) ∧
    x = (steps.foldl (λ acc f => f acc) 1)}

/-- The main theorem -/
theorem bug_positions (n : ℕ) :
  (reachablePositions n).ncard = fib (n + 4) - (n + 4) :=
sorry

end NUMINAMATH_CALUDE_bug_positions_l3714_371454


namespace NUMINAMATH_CALUDE_inverse_mod_103_l3714_371425

theorem inverse_mod_103 (h : (7⁻¹ : ZMod 103) = 55) : (49⁻¹ : ZMod 103) = 38 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_103_l3714_371425


namespace NUMINAMATH_CALUDE_ratio_chain_l3714_371400

theorem ratio_chain (a b c d : ℚ) 
  (h1 : a / b = 1 / 4)
  (h2 : b / c = 13 / 9)
  (h3 : c / d = 5 / 13) :
  a / d = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ratio_chain_l3714_371400


namespace NUMINAMATH_CALUDE_birds_in_tree_l3714_371481

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29)
  (h2 : final_birds = 42) :
  final_birds - initial_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l3714_371481


namespace NUMINAMATH_CALUDE_range_of_a_l3714_371405

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def q (a : ℝ) : Prop := StrictMono (fun x => Real.log x / Real.log a)

-- Define the theorem
theorem range_of_a :
  (∃ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  {a : ℝ | (-2 < a ∧ a ≤ 1) ∨ (a ≥ 2)} = {a : ℝ | a ∈ Set.Ioc (-2) 1 ∪ Set.Ici 2} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3714_371405


namespace NUMINAMATH_CALUDE_math_score_calculation_math_score_is_83_l3714_371479

theorem math_score_calculation (average_three : ℝ) (average_decrease : ℝ) : ℝ :=
  let total_three := 3 * average_three
  let new_average := average_three - average_decrease
  let total_four := 4 * new_average
  total_four - total_three

theorem math_score_is_83 :
  math_score_calculation 95 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_math_score_calculation_math_score_is_83_l3714_371479


namespace NUMINAMATH_CALUDE_sin_2000_in_terms_of_tan_160_l3714_371443

theorem sin_2000_in_terms_of_tan_160 (a : ℝ) (h : Real.tan (160 * π / 180) = a) :
  Real.sin (2000 * π / 180) = -a / Real.sqrt (1 + a^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_2000_in_terms_of_tan_160_l3714_371443


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3714_371430

theorem solution_set_of_inequality (x : ℝ) : 
  x * (x + 2) < 3 ↔ -3 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3714_371430


namespace NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l3714_371410

theorem sqrt_inequality_solution_set (x : ℝ) : 
  (Real.sqrt (x + 3) < 2) ↔ (x ∈ Set.Icc (-3) 1) :=
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l3714_371410


namespace NUMINAMATH_CALUDE_watermelon_customers_l3714_371424

theorem watermelon_customers (total : ℕ) (one_melon : ℕ) (three_melons : ℕ) :
  total = 46 →
  one_melon = 17 →
  three_melons = 3 →
  ∃ (two_melons : ℕ),
    two_melons * 2 + one_melon * 1 + three_melons * 3 = total ∧
    two_melons = 10 :=
by sorry

end NUMINAMATH_CALUDE_watermelon_customers_l3714_371424


namespace NUMINAMATH_CALUDE_final_seashell_count_l3714_371486

def seashell_transactions (initial : ℝ) (friend_gift : ℝ) (brother_gift : ℝ) 
  (buy_percent : ℝ) (sell_fraction : ℝ) (damage_percent : ℝ) (trade_fraction : ℝ) : ℝ :=
  let remaining_after_gifts := initial - friend_gift - brother_gift
  let after_buying := remaining_after_gifts + (buy_percent * remaining_after_gifts)
  let after_selling := after_buying - (sell_fraction * after_buying)
  let after_damage := after_selling - (damage_percent * after_selling)
  after_damage - (trade_fraction * after_damage)

theorem final_seashell_count : 
  seashell_transactions 385.5 45.75 34.25 0.2 (2/3) 0.1 (1/4) = 82.485 := by
  sorry

end NUMINAMATH_CALUDE_final_seashell_count_l3714_371486


namespace NUMINAMATH_CALUDE_min_combines_to_goal_l3714_371482

/-- Represents the state of rock stacks -/
structure RockStacks :=
  (stacks : List ℕ)

/-- Allowed operations on rock stacks -/
inductive Operation
  | Split (i : ℕ) (a b : ℕ)
  | Combine (i j : ℕ)

/-- Applies an operation to the rock stacks -/
def applyOperation (s : RockStacks) (op : Operation) : RockStacks :=
  sorry

/-- Checks if the goal state is reached -/
def isGoalReached (s : RockStacks) (n : ℕ) : Prop :=
  sorry

/-- Theorem: Minimum number of combines to reach the goal state -/
theorem min_combines_to_goal (n : ℕ) :
  ∃ (ops : List Operation),
    (ops.filter (λ op => match op with
      | Operation.Combine _ _ => true
      | _ => false)).length = 4 ∧
    isGoalReached (ops.foldl applyOperation ⟨[3 * 2^n]⟩) n :=
  sorry

end NUMINAMATH_CALUDE_min_combines_to_goal_l3714_371482


namespace NUMINAMATH_CALUDE_power_multiplication_l3714_371453

theorem power_multiplication (m : ℝ) : m^2 * m^3 = m^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3714_371453


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_squared_l3714_371431

/-- Represents a parallelogram ABCD with specific properties -/
structure Parallelogram where
  -- Area of the parallelogram
  area : ℝ
  -- Length of PQ (projections of A and C onto BD)
  pq : ℝ
  -- Length of RS (projections of B and D onto AC)
  rs : ℝ
  -- Ensures area is positive
  area_pos : area > 0
  -- Ensures PQ is positive
  pq_pos : pq > 0
  -- Ensures RS is positive
  rs_pos : rs > 0

/-- The main theorem about the longer diagonal of the parallelogram -/
theorem parallelogram_diagonal_squared
  (abcd : Parallelogram)
  (h1 : abcd.area = 24)
  (h2 : abcd.pq = 8)
  (h3 : abcd.rs = 10) :
  ∃ (d : ℝ), d > 0 ∧ d^2 = 62 + 20 * Real.sqrt 61 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_squared_l3714_371431


namespace NUMINAMATH_CALUDE_six_couples_handshakes_l3714_371472

/-- The number of handshakes in a gathering of couples where each person shakes hands
    with everyone except their spouse -/
def handshakes (n : ℕ) : ℕ :=
  let total_people := 2 * n
  let handshakes_per_person := total_people - 2
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 6 couples, the total number of handshakes is 60 -/
theorem six_couples_handshakes :
  handshakes 6 = 60 := by
  sorry


end NUMINAMATH_CALUDE_six_couples_handshakes_l3714_371472


namespace NUMINAMATH_CALUDE_shorter_leg_length_is_five_l3714_371422

/-- A right triangle that can be cut and reassembled into a square -/
structure CuttableRightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  is_right_triangle : shorter_leg^2 + longer_leg^2 = hypotenuse^2
  can_form_square : hypotenuse = 2 * shorter_leg

/-- The theorem stating that if a right triangle with longer leg 10 can be cut and
    reassembled into a square, then its shorter leg has length 5 -/
theorem shorter_leg_length_is_five
  (triangle : CuttableRightTriangle)
  (h : triangle.longer_leg = 10) :
  triangle.shorter_leg = 5 := by
  sorry


end NUMINAMATH_CALUDE_shorter_leg_length_is_five_l3714_371422


namespace NUMINAMATH_CALUDE_probability_in_painted_cube_l3714_371421

/-- Represents a 5x5x5 cube with three adjacent faces painted -/
structure PaintedCube :=
  (size : ℕ)
  (total_cubes : ℕ)
  (three_face_cubes : ℕ)
  (no_face_cubes : ℕ)

/-- The probability of selecting one cube with three painted faces and one with no painted faces -/
def probability_three_and_none (cube : PaintedCube) : ℚ :=
  (cube.three_face_cubes * cube.no_face_cubes : ℚ) / (cube.total_cubes * (cube.total_cubes - 1) / 2)

/-- The theorem to be proved -/
theorem probability_in_painted_cube :
  ∃ (cube : PaintedCube),
    cube.size = 5 ∧
    cube.total_cubes = 125 ∧
    cube.three_face_cubes = 1 ∧
    cube.no_face_cubes = 76 ∧
    probability_three_and_none cube = 2 / 205 :=
sorry

end NUMINAMATH_CALUDE_probability_in_painted_cube_l3714_371421


namespace NUMINAMATH_CALUDE_riza_son_age_l3714_371461

/-- Represents the age difference between Riza and her son -/
def age_difference : ℕ := 25

/-- Represents the sum of Riza's and her son's current ages -/
def current_age_sum : ℕ := 105

/-- Represents Riza's son's current age -/
def son_age : ℕ := (current_age_sum - age_difference) / 2

theorem riza_son_age : son_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_riza_son_age_l3714_371461


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_l3714_371418

theorem sum_of_solutions_quadratic (a b c d e : ℝ) : 
  (∀ x, x^2 - a*x - b = c*x + d) → 
  (∃ x₁ x₂, x₁^2 - a*x₁ - b = c*x₁ + d ∧ 
            x₂^2 - a*x₂ - b = c*x₂ + d ∧ 
            x₁ ≠ x₂) →
  (x₁ + x₂ = a + c) :=
by sorry

-- Specific instance
theorem sum_of_solutions_specific : 
  (∀ x, x^2 - 6*x - 8 = 4*x + 20) → 
  (∃ x₁ x₂, x₁^2 - 6*x₁ - 8 = 4*x₁ + 20 ∧ 
            x₂^2 - 6*x₂ - 8 = 4*x₂ + 20 ∧ 
            x₁ ≠ x₂) →
  (x₁ + x₂ = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_l3714_371418


namespace NUMINAMATH_CALUDE_sticker_distribution_l3714_371420

theorem sticker_distribution (d : ℕ) (h : d > 0) :
  let total_stickers : ℕ := 72
  let friends : ℕ := d
  let stickers_per_friend : ℚ := total_stickers / friends
  stickers_per_friend = 72 / d :=
by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l3714_371420


namespace NUMINAMATH_CALUDE_notebook_cost_l3714_371436

/-- Given a notebook and its cover with a total pre-tax cost of $3.00,
    where the notebook costs $2 more than its cover,
    prove that the pre-tax cost of the notebook is $2.50. -/
theorem notebook_cost (notebook_cost cover_cost : ℝ) : 
  notebook_cost + cover_cost = 3 →
  notebook_cost = cover_cost + 2 →
  notebook_cost = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l3714_371436


namespace NUMINAMATH_CALUDE_root_equation_value_l3714_371495

theorem root_equation_value (a : ℝ) (h : a^2 + 3*a - 5 = 0) :
  a^2 + 3*a + 2021 = 2026 := by
sorry

end NUMINAMATH_CALUDE_root_equation_value_l3714_371495


namespace NUMINAMATH_CALUDE_males_with_college_degree_only_count_l3714_371457

/-- Represents the employee demographics of a company -/
structure CompanyDemographics where
  total_employees : Nat
  total_females : Nat
  employees_with_advanced_degrees : Nat
  females_with_advanced_degrees : Nat

/-- Calculates the number of males with a college degree only -/
def males_with_college_degree_only (demo : CompanyDemographics) : Nat :=
  let total_males := demo.total_employees - demo.total_females
  let males_with_advanced_degrees := demo.employees_with_advanced_degrees - demo.females_with_advanced_degrees
  total_males - males_with_advanced_degrees

/-- Theorem stating the number of males with a college degree only -/
theorem males_with_college_degree_only_count 
  (demo : CompanyDemographics) 
  (h1 : demo.total_employees = 180)
  (h2 : demo.total_females = 110)
  (h3 : demo.employees_with_advanced_degrees = 90)
  (h4 : demo.females_with_advanced_degrees = 55) :
  males_with_college_degree_only demo = 35 := by
  sorry

#eval males_with_college_degree_only { 
  total_employees := 180, 
  total_females := 110, 
  employees_with_advanced_degrees := 90, 
  females_with_advanced_degrees := 55 
}

end NUMINAMATH_CALUDE_males_with_college_degree_only_count_l3714_371457


namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_one_l3714_371483

/-- A polynomial of degree 5 with integer coefficients -/
def polynomial (a₁ a₂ a₃ a₄ a₅ : ℤ) (x : ℝ) : ℝ :=
  x^5 + a₁ * x^4 + a₂ * x^3 + a₃ * x^2 + a₄ * x + a₅

/-- Theorem stating the value of f(-1) given specific conditions -/
theorem polynomial_value_at_negative_one
  (a₁ a₂ a₃ a₄ a₅ : ℤ)
  (h1 : polynomial a₁ a₂ a₃ a₄ a₅ (Real.sqrt 3 + Real.sqrt 2) = 0)
  (h2 : polynomial a₁ a₂ a₃ a₄ a₅ 1 + polynomial a₁ a₂ a₃ a₄ a₅ 3 = 0) :
  polynomial a₁ a₂ a₃ a₄ a₅ (-1) = 24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_negative_one_l3714_371483


namespace NUMINAMATH_CALUDE_inequality_proof_l3714_371444

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ∧
  ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3714_371444


namespace NUMINAMATH_CALUDE_absolute_value_equality_l3714_371490

theorem absolute_value_equality (x : ℝ) (h : x < -2) :
  |x - Real.sqrt ((x + 2)^2)| = -2*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l3714_371490


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_l3714_371493

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem tangent_line_at_point_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  let tangent_line (x : ℝ) : ℝ := m * (x - x₀) + y₀
  (∀ x, tangent_line x = -3 * x + 2) ∧ y₀ = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_l3714_371493


namespace NUMINAMATH_CALUDE_circle_properties_correct_l3714_371409

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

/-- Given a circle equation, compute its center and radius -/
def computeCircleProperties (eq : CircleEquation) : CircleProperties :=
  sorry

theorem circle_properties_correct (eq : CircleEquation) 
  (h : eq = CircleEquation.mk 4 (-8) 4 (-16) 20) : 
  computeCircleProperties eq = CircleProperties.mk (1, 2) 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_correct_l3714_371409


namespace NUMINAMATH_CALUDE_greatest_odd_factors_below_100_l3714_371438

/-- A number has an odd number of positive factors if and only if it is a perfect square. -/
def has_odd_factors (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- The greatest whole number less than 100 that has an odd number of positive factors is 81. -/
theorem greatest_odd_factors_below_100 : 
  (∀ m : ℕ, m < 100 → has_odd_factors m → m ≤ 81) ∧ has_odd_factors 81 ∧ 81 < 100 := by
  sorry

end NUMINAMATH_CALUDE_greatest_odd_factors_below_100_l3714_371438


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l3714_371435

theorem stratified_sampling_female_count 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (female_employees : ℕ) 
  (male_sampled : ℕ) :
  total_employees = 140 →
  male_employees = 80 →
  female_employees = 60 →
  male_sampled = 16 →
  (female_employees : ℚ) * (male_sampled : ℚ) / (male_employees : ℚ) = 12 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l3714_371435


namespace NUMINAMATH_CALUDE_square_root_equation_l3714_371401

theorem square_root_equation (x : ℝ) : (x + 1)^2 = 9 → x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l3714_371401


namespace NUMINAMATH_CALUDE_square_of_binomial_l3714_371403

theorem square_of_binomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 14*x + k = (x - a)^2) ↔ k = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l3714_371403


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3714_371497

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1/3)
  (h3 : a 2 + a 5 = 4)
  (h4 : ∃ n : ℕ, a n = 33) :
  (∃ n : ℕ, a n = 33 ∧ n = 50) ∧
  (∃ S : ℚ, S = (50 * 51) / 3 ∧ S = 850) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3714_371497


namespace NUMINAMATH_CALUDE_smallest_x_value_l3714_371471

theorem smallest_x_value (x y : ℝ) : 
  4 ≤ x ∧ x < 6 →
  6 < y ∧ y < 10 →
  (∃ (n : ℤ), n = ⌊y - x⌋ ∧ n ≤ 5 ∧ ∀ (m : ℤ), m = ⌊y - x⌋ → m ≤ n) →
  x ≥ 4 ∧ ∀ (z : ℝ), (4 ≤ z ∧ z < 6 ∧ 
    (∃ (w : ℝ), 6 < w ∧ w < 10 ∧ 
      (∃ (n : ℤ), n = ⌊w - z⌋ ∧ n ≤ 5 ∧ ∀ (m : ℤ), m = ⌊w - z⌋ → m ≤ n))) →
    z ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3714_371471


namespace NUMINAMATH_CALUDE_imaginary_part_product_l3714_371440

theorem imaginary_part_product : Complex.im ((1 + Complex.I) * (3 - Complex.I)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_product_l3714_371440


namespace NUMINAMATH_CALUDE_crazy_silly_school_theorem_l3714_371488

/-- Represents the 'Crazy Silly School' series collection --/
structure CrazySillyCollection where
  books : ℕ
  movies : ℕ
  videoGames : ℕ
  audiobooks : ℕ

/-- Represents the completed items in the collection --/
structure CompletedItems where
  booksRead : ℕ
  moviesWatched : ℕ
  gamesPlayed : ℕ
  audiobooksListened : ℕ
  halfReadBooks : ℕ
  halfWatchedMovies : ℕ

/-- Calculates the portions left to complete in the collection --/
def portionsLeftToComplete (collection : CrazySillyCollection) (completed : CompletedItems) : ℚ :=
  let totalPortions := collection.books + collection.movies + collection.videoGames + collection.audiobooks
  let completedPortions := completed.booksRead - completed.halfReadBooks / 2 +
                           completed.moviesWatched - completed.halfWatchedMovies / 2 +
                           completed.gamesPlayed +
                           completed.audiobooksListened
  totalPortions - completedPortions

/-- Theorem stating the number of portions left to complete in the 'Crazy Silly School' series --/
theorem crazy_silly_school_theorem (collection : CrazySillyCollection) (completed : CompletedItems) :
  collection.books = 22 ∧
  collection.movies = 10 ∧
  collection.videoGames = 8 ∧
  collection.audiobooks = 15 ∧
  completed.booksRead = 12 ∧
  completed.moviesWatched = 6 ∧
  completed.gamesPlayed = 3 ∧
  completed.audiobooksListened = 7 ∧
  completed.halfReadBooks = 2 ∧
  completed.halfWatchedMovies = 1 →
  portionsLeftToComplete collection completed = 28.5 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_theorem_l3714_371488


namespace NUMINAMATH_CALUDE_sandis_initial_amount_l3714_371474

/-- Proves that Sandi's initial amount was $300 given the conditions of the problem -/
theorem sandis_initial_amount (sandi_initial : ℝ) : 
  (3 * sandi_initial + 150 = 1050) → sandi_initial = 300 := by
  sorry

end NUMINAMATH_CALUDE_sandis_initial_amount_l3714_371474


namespace NUMINAMATH_CALUDE_age_problem_l3714_371459

theorem age_problem (oleg serezha misha : ℕ) : 
  serezha = oleg + 1 →
  misha = serezha + 1 →
  40 < oleg + serezha + misha →
  oleg + serezha + misha < 45 →
  oleg = 13 ∧ serezha = 14 ∧ misha = 15 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l3714_371459


namespace NUMINAMATH_CALUDE_sally_lemonade_sales_l3714_371470

/-- Calculates the total number of lemonade cups sold over two weeks -/
def total_lemonade_cups (last_week : ℕ) (percent_increase : ℕ) : ℕ :=
  let this_week := last_week + (last_week * percent_increase) / 100
  last_week + this_week

/-- Proves that given the conditions, Sally sold 46 cups of lemonade in total -/
theorem sally_lemonade_sales : total_lemonade_cups 20 30 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sally_lemonade_sales_l3714_371470


namespace NUMINAMATH_CALUDE_nickel_probability_is_one_fourth_l3714_371458

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in cents -/
def totalValue : Coin → ℕ
  | Coin.Dime => 1000
  | Coin.Nickel => 500
  | Coin.Penny => 200

/-- The number of coins of each type -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Dime + coinCount Coin.Nickel + coinCount Coin.Penny

/-- The probability of selecting a nickel -/
def nickelProbability : ℚ := coinCount Coin.Nickel / totalCoins

theorem nickel_probability_is_one_fourth :
  nickelProbability = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_nickel_probability_is_one_fourth_l3714_371458


namespace NUMINAMATH_CALUDE_remaining_potatoes_l3714_371475

def initial_potatoes : ℕ := 8
def eaten_potatoes : ℕ := 3

theorem remaining_potatoes : initial_potatoes - eaten_potatoes = 5 := by
  sorry

end NUMINAMATH_CALUDE_remaining_potatoes_l3714_371475


namespace NUMINAMATH_CALUDE_cube_sum_implies_sum_l3714_371463

theorem cube_sum_implies_sum (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_implies_sum_l3714_371463


namespace NUMINAMATH_CALUDE_max_value_of_a_l3714_371419

def f (x a : ℝ) : ℝ := |8 * x^3 - 12 * x - a| + a

theorem max_value_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≤ 0) ∧ (∃ x ∈ Set.Icc 0 1, f x a = 0) →
  a ≤ -2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3714_371419


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l3714_371417

def A : Set ℝ := {x | x * (x - 2) < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (Set.univ \ B) = Set.Icc 1 2 ∩ Set.Iio 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l3714_371417


namespace NUMINAMATH_CALUDE_catering_weight_calculation_mason_catering_weight_l3714_371402

/-- Calculates the total weight of silverware and plates for a catering event. -/
theorem catering_weight_calculation (silverware_weight plate_weight : ℕ)
  (silverware_per_setting plates_per_setting : ℕ)
  (tables settings_per_table backup_settings : ℕ) : ℕ :=
  let total_settings := tables * settings_per_table + backup_settings
  let weight_per_setting := silverware_per_setting * silverware_weight + plates_per_setting * plate_weight
  total_settings * weight_per_setting

/-- Proves that the total weight of all settings for Mason's catering event is 5040 ounces. -/
theorem mason_catering_weight :
  catering_weight_calculation 4 12 3 2 15 8 20 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_catering_weight_calculation_mason_catering_weight_l3714_371402


namespace NUMINAMATH_CALUDE_tom_balloons_remaining_l3714_371441

/-- Given that Tom has 30 violet balloons initially and gives away 16 balloons,
    prove that he has 14 violet balloons remaining. -/
theorem tom_balloons_remaining (initial : ℕ) (given_away : ℕ) (h1 : initial = 30) (h2 : given_away = 16) :
  initial - given_away = 14 := by
  sorry

end NUMINAMATH_CALUDE_tom_balloons_remaining_l3714_371441


namespace NUMINAMATH_CALUDE_third_purchase_total_l3714_371498

/-- Represents the clothing purchase scenario -/
structure ClothingPurchase where
  initialCost : ℕ
  typeAIncrease : ℕ
  typeBIncrease : ℕ
  secondCostIncrease : ℕ
  averageIncrease : ℕ
  profitMargin : ℚ
  thirdTypeBCost : ℕ

/-- Theorem stating the total number of pieces in the third purchase -/
theorem third_purchase_total (cp : ClothingPurchase)
  (h1 : cp.initialCost = 3600)
  (h2 : cp.typeAIncrease = 20)
  (h3 : cp.typeBIncrease = 5)
  (h4 : cp.secondCostIncrease = 400)
  (h5 : cp.averageIncrease = 8)
  (h6 : cp.profitMargin = 35 / 100)
  (h7 : cp.thirdTypeBCost = 3000) :
  ∃ (x y : ℕ),
    x + y = 50 ∧
    20 * x + 5 * y = 400 ∧
    8 * (x + y) = 400 ∧
    (3600 + 400) * (1 + cp.profitMargin) = 5400 ∧
    x * 60 + y * 75 = 3600 ∧
    3000 / 75 = (5400 - 3000) / 60 ∧
    (3000 / 75 + 3000 / 75) = 80 :=
  sorry


end NUMINAMATH_CALUDE_third_purchase_total_l3714_371498


namespace NUMINAMATH_CALUDE_sin_105_times_sin_15_l3714_371416

theorem sin_105_times_sin_15 : Real.sin (105 * π / 180) * Real.sin (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_105_times_sin_15_l3714_371416


namespace NUMINAMATH_CALUDE_digit_457_of_17_53_l3714_371456

/-- The decimal expansion of 17/53 -/
def decimal_expansion : ℕ → ℕ := sorry

/-- The length of the repeating part in the decimal expansion of 17/53 -/
def cycle_length : ℕ := 20

/-- The 457th digit after the decimal point in the expansion of 17/53 is 1 -/
theorem digit_457_of_17_53 : decimal_expansion 457 = 1 := by sorry

end NUMINAMATH_CALUDE_digit_457_of_17_53_l3714_371456


namespace NUMINAMATH_CALUDE_maurice_prior_rides_eq_eight_l3714_371412

/-- The number of times Maurice rode during his visit -/
def maurice_visit_rides : ℕ := 8

/-- The number of times Matt rode without Maurice -/
def matt_solo_rides : ℕ := 16

/-- The total number of times Matt rode -/
def matt_total_rides : ℕ := maurice_visit_rides + matt_solo_rides

/-- The number of times Maurice rode before his visit -/
def maurice_prior_rides : ℕ := matt_total_rides / 3

theorem maurice_prior_rides_eq_eight :
  maurice_prior_rides = 8 := by sorry

end NUMINAMATH_CALUDE_maurice_prior_rides_eq_eight_l3714_371412


namespace NUMINAMATH_CALUDE_newspaper_pages_l3714_371449

/-- Represents a newspaper with a certain number of pages -/
structure Newspaper where
  num_pages : ℕ

/-- Predicate indicating that two pages are on the same sheet -/
def on_same_sheet (n : Newspaper) (p1 p2 : ℕ) : Prop :=
  p1 ≤ n.num_pages ∧ p2 ≤ n.num_pages ∧ p1 + p2 = n.num_pages + 1

/-- The theorem stating the number of pages in the newspaper -/
theorem newspaper_pages : 
  ∃ (n : Newspaper), n.num_pages = 28 ∧ on_same_sheet n 8 21 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_pages_l3714_371449


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_is_96_l3714_371468

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  -- The shorter base of the trapezoid
  short_base : ℝ
  -- The perimeter of the trapezoid
  perimeter : ℝ
  -- The diagonal bisects the obtuse angle
  diagonal_bisects_obtuse_angle : Bool

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating that an isosceles trapezoid with given properties has an area of 96 -/
theorem isosceles_trapezoid_area_is_96 (t : IsoscelesTrapezoid) 
  (h1 : t.short_base = 3)
  (h2 : t.perimeter = 42)
  (h3 : t.diagonal_bisects_obtuse_angle = true) :
  area t = 96 := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_is_96_l3714_371468


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l3714_371423

theorem complex_arithmetic_equality : (9 - 8 + 7)^2 * 6 + 5 - 4^2 * 3 + 2^3 - 1 = 347 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l3714_371423


namespace NUMINAMATH_CALUDE_value_added_to_reach_new_average_l3714_371450

theorem value_added_to_reach_new_average (n : ℕ) (initial_avg final_avg : ℝ) (h1 : n = 15) (h2 : initial_avg = 40) (h3 : final_avg = 55) :
  ∃ x : ℝ, (n : ℝ) * initial_avg + n * x = n * final_avg ∧ x = 15 :=
by sorry

end NUMINAMATH_CALUDE_value_added_to_reach_new_average_l3714_371450


namespace NUMINAMATH_CALUDE_chocolate_bars_per_box_l3714_371476

theorem chocolate_bars_per_box (total_bars : ℕ) (total_boxes : ℕ) 
  (h1 : total_bars = 849) (h2 : total_boxes = 170) :
  total_bars / total_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_per_box_l3714_371476


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l3714_371426

theorem arithmetic_mean_geq_geometric_mean (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l3714_371426


namespace NUMINAMATH_CALUDE_diamond_symmetry_points_l3714_371492

/-- The diamond operation -/
def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

/-- The set of points (x, y) satisfying x ⋄ y = y ⋄ x -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1}

/-- Two lines in ℝ² -/
def two_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = p.2 ∨ p.1 = -p.2}

theorem diamond_symmetry_points :
  S = two_lines ∪ ({0} : Set ℝ).prod Set.univ ∪ Set.univ.prod ({0} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_diamond_symmetry_points_l3714_371492


namespace NUMINAMATH_CALUDE_square_partition_exists_l3714_371499

/-- A square is a four-sided polygon with all sides equal and all angles equal to 90 degrees. -/
structure Square where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  sides_equal : ∀ i j, sides i = sides j
  angles_right : ∀ i, angles i = 90

/-- A convex pentagon is a five-sided polygon with all interior angles less than 180 degrees. -/
structure ConvexPentagon where
  sides : Fin 5 → ℝ
  angles : Fin 5 → ℝ
  angles_convex : ∀ i, angles i < 180

/-- A partition of a square into convex pentagons -/
structure SquarePartition where
  square : Square
  pentagons : List ConvexPentagon
  is_partition : Square → List ConvexPentagon → Prop

/-- Theorem: There exists a partition of a square into a finite number of convex pentagons -/
theorem square_partition_exists : ∃ p : SquarePartition, p.pentagons.length > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_partition_exists_l3714_371499


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3714_371484

theorem greatest_divisor_four_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → 
    12 ∣ (k * (k + 1) * (k + 2) * (k + 3)) ∧
    ∀ (m : ℕ), m > 12 → 
      ∃ (j : ℕ), j > 0 ∧ ¬(m ∣ (j * (j + 1) * (j + 2) * (j + 3)))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3714_371484


namespace NUMINAMATH_CALUDE_angle_trig_sum_l3714_371473

theorem angle_trig_sum (a : ℝ) (ha : a ≠ 0) :
  let α := Real.arctan (3*a / (-4*a))
  if a > 0 then
    Real.sin α + Real.cos α - Real.tan α = 11/20
  else
    Real.sin α + Real.cos α - Real.tan α = 19/20 := by
  sorry

end NUMINAMATH_CALUDE_angle_trig_sum_l3714_371473


namespace NUMINAMATH_CALUDE_initial_apples_count_l3714_371429

/-- Represents the number of apple trees Rachel has -/
def total_trees : ℕ := 52

/-- Represents the number of apples picked from one tree -/
def apples_picked : ℕ := 2

/-- Represents the number of apples remaining on the tree after picking -/
def apples_remaining : ℕ := 7

/-- Theorem stating that the initial number of apples on the tree is equal to
    the sum of apples remaining and apples picked -/
theorem initial_apples_count : 
  ∃ (initial_apples : ℕ), initial_apples = apples_remaining + apples_picked :=
by sorry

end NUMINAMATH_CALUDE_initial_apples_count_l3714_371429


namespace NUMINAMATH_CALUDE_fib_150_mod_9_l3714_371404

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Fibonacci sequence modulo 9 repeats every 24 terms -/
axiom fib_mod_9_period : ∀ n, fib n % 9 = fib (n % 24) % 9

/-- The 6th Fibonacci number modulo 9 is 8 -/
axiom fib_6_mod_9 : fib 6 % 9 = 8

/-- The 150th Fibonacci number modulo 9 -/
theorem fib_150_mod_9 : fib 150 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_fib_150_mod_9_l3714_371404


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3714_371477

theorem min_value_trig_expression (α β : Real) :
  ∃ (min : Real), 
    (∀ α β : Real, (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ min) ∧ 
    (∃ α β : Real, (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = min) ∧
    min = 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3714_371477


namespace NUMINAMATH_CALUDE_inequality_proof_l3714_371415

theorem inequality_proof (a b c : ℝ) (k : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 2) (habc : a * b * c = 1) :
  (a^k / (a + b)) + (b^k / (b + c)) + (c^k / (c + a)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3714_371415


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3714_371455

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_condition : a 1 * a 3 = 6 * a 2) :
  a 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3714_371455


namespace NUMINAMATH_CALUDE_reciprocal_and_abs_of_negative_one_sixth_l3714_371432

theorem reciprocal_and_abs_of_negative_one_sixth :
  let x : ℚ := -1/6
  let reciprocal : ℚ := 1/x
  (reciprocal = -6) ∧ (abs reciprocal = 6) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_abs_of_negative_one_sixth_l3714_371432


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3714_371487

-- Define the conditions for a hyperbola and an ellipse
def is_hyperbola (m : ℝ) : Prop := (m + 3) * (2 * m + 1) < 0
def is_ellipse_with_y_intersection (m : ℝ) : Prop := -(2 * m - 1) > m + 2 ∧ m + 2 > 0

-- Define the condition given in the problem
def given_condition (m : ℝ) : Prop := -2 < m ∧ m < -1/3

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ m : ℝ, is_hyperbola m ∧ is_ellipse_with_y_intersection m → given_condition m) ∧
  (∃ m : ℝ, given_condition m ∧ ¬(is_hyperbola m ∧ is_ellipse_with_y_intersection m)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3714_371487


namespace NUMINAMATH_CALUDE_kyle_age_l3714_371496

def age_problem (casey shelley kyle julian frederick tyson : ℕ) : Prop :=
  (shelley + 3 = kyle) ∧
  (shelley = julian + 4) ∧
  (julian + 20 = frederick) ∧
  (frederick = 2 * tyson) ∧
  (tyson = 2 * casey) ∧
  (casey = 15)

theorem kyle_age :
  ∀ casey shelley kyle julian frederick tyson : ℕ,
  age_problem casey shelley kyle julian frederick tyson →
  kyle = 47 :=
by
  sorry

end NUMINAMATH_CALUDE_kyle_age_l3714_371496


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3714_371448

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 3^5 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes :
  ways_to_put_balls_in_boxes 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3714_371448


namespace NUMINAMATH_CALUDE_janet_needs_775_l3714_371406

/-- The amount of additional money Janet needs to rent an apartment -/
def additional_money_needed (savings : ℕ) (monthly_rent : ℕ) (advance_months : ℕ) (deposit : ℕ) : ℕ :=
  (monthly_rent * advance_months + deposit) - savings

/-- Proof that Janet needs $775 more to rent the apartment -/
theorem janet_needs_775 : 
  additional_money_needed 2225 1250 2 500 = 775 := by
  sorry

end NUMINAMATH_CALUDE_janet_needs_775_l3714_371406


namespace NUMINAMATH_CALUDE_second_exam_sleep_duration_l3714_371428

/-- Represents the relationship between sleep duration and test score -/
structure SleepScoreRelation where
  sleep : ℝ
  score : ℝ
  constant : ℝ
  inv_relation : sleep * score = constant

/-- Proves the required sleep duration for the second exam -/
theorem second_exam_sleep_duration 
  (first_exam : SleepScoreRelation)
  (h_first_exam : first_exam.sleep = 9 ∧ first_exam.score = 75)
  (target_average : ℝ)
  (h_target_average : target_average = 85) :
  ∃ (second_exam : SleepScoreRelation),
    second_exam.constant = first_exam.constant ∧
    (first_exam.score + second_exam.score) / 2 = target_average ∧
    second_exam.sleep = 135 / 19 := by
  sorry

end NUMINAMATH_CALUDE_second_exam_sleep_duration_l3714_371428


namespace NUMINAMATH_CALUDE_graph_passes_through_quadrants_l3714_371469

-- Define the function
def f (x : ℝ) : ℝ := -3 * x + 1

-- Theorem statement
theorem graph_passes_through_quadrants :
  (∃ x y, x > 0 ∧ y > 0 ∧ f x = y) ∧  -- First quadrant
  (∃ x y, x < 0 ∧ y > 0 ∧ f x = y) ∧  -- Second quadrant
  (∃ x y, x > 0 ∧ y < 0 ∧ f x = y) :=  -- Fourth quadrant
by sorry

end NUMINAMATH_CALUDE_graph_passes_through_quadrants_l3714_371469


namespace NUMINAMATH_CALUDE_winter_olympics_volunteer_allocation_l3714_371407

theorem winter_olympics_volunteer_allocation :
  let n : ℕ := 5  -- number of volunteers
  let k : ℕ := 4  -- number of projects
  let allocation_schemes : ℕ := (n.choose 2) * k.factorial
  allocation_schemes = 240 := by sorry

end NUMINAMATH_CALUDE_winter_olympics_volunteer_allocation_l3714_371407


namespace NUMINAMATH_CALUDE_wage_increase_percentage_l3714_371489

theorem wage_increase_percentage (old_wage new_wage : ℝ) (h1 : old_wage = 20) (h2 : new_wage = 28) :
  (new_wage - old_wage) / old_wage * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_wage_increase_percentage_l3714_371489


namespace NUMINAMATH_CALUDE_difference_of_squares_l3714_371446

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3714_371446


namespace NUMINAMATH_CALUDE_arccos_lt_arcsin_iff_l3714_371411

theorem arccos_lt_arcsin_iff (x : ℝ) : Real.arccos x < Real.arcsin x ↔ x ∈ Set.Ioo (1 / Real.sqrt 2) 1 := by
  sorry

end NUMINAMATH_CALUDE_arccos_lt_arcsin_iff_l3714_371411


namespace NUMINAMATH_CALUDE_saheed_kayla_earnings_ratio_l3714_371437

/-- Proves that the ratio of Saheed's earnings to Kayla's earnings is 4:1 -/
theorem saheed_kayla_earnings_ratio :
  let vika_earnings : ℕ := 84
  let kayla_earnings : ℕ := vika_earnings - 30
  let saheed_earnings : ℕ := 216
  (saheed_earnings : ℚ) / kayla_earnings = 4 := by
  sorry

end NUMINAMATH_CALUDE_saheed_kayla_earnings_ratio_l3714_371437


namespace NUMINAMATH_CALUDE_sum_digits_inequality_l3714_371464

/-- S(n) represents the sum of digits of a natural number n -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n, S(8n) ≥ (1/8) * S(n) -/
theorem sum_digits_inequality (n : ℕ) : S (8 * n) ≥ (1 / 8) * S n := by sorry

end NUMINAMATH_CALUDE_sum_digits_inequality_l3714_371464


namespace NUMINAMATH_CALUDE_rental_cost_is_165_l3714_371442

/-- Calculates the total cost of renting a car given the daily rate, per-mile rate, number of days, and miles driven. -/
def total_rental_cost (daily_rate : ℝ) (mile_rate : ℝ) (days : ℕ) (miles : ℕ) : ℝ :=
  daily_rate * (days : ℝ) + mile_rate * (miles : ℝ)

/-- Theorem stating that under the given conditions, the total rental cost is $165. -/
theorem rental_cost_is_165 :
  let daily_rate : ℝ := 30
  let mile_rate : ℝ := 0.15
  let days : ℕ := 3
  let miles : ℕ := 500
  total_rental_cost daily_rate mile_rate days miles = 165 := by
sorry


end NUMINAMATH_CALUDE_rental_cost_is_165_l3714_371442


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l3714_371408

theorem factor_implies_c_value (c : ℝ) : 
  (∀ x : ℝ, (2 * x + 5) ∣ (4 * x^3 + 19 * x^2 + c * x + 45)) → c = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l3714_371408


namespace NUMINAMATH_CALUDE_exponent_addition_l3714_371466

theorem exponent_addition (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_addition_l3714_371466


namespace NUMINAMATH_CALUDE_age_of_fifteenth_person_l3714_371452

theorem age_of_fifteenth_person (total_persons : ℕ) (avg_all : ℕ) (group1_size : ℕ) (avg_group1 : ℕ) (group2_size : ℕ) (avg_group2 : ℕ) :
  total_persons = 20 →
  avg_all = 15 →
  group1_size = 5 →
  avg_group1 = 14 →
  group2_size = 9 →
  avg_group2 = 16 →
  ∃ (age_15th : ℕ), age_15th = 86 ∧
    total_persons * avg_all = group1_size * avg_group1 + group2_size * avg_group2 + age_15th :=
by sorry

end NUMINAMATH_CALUDE_age_of_fifteenth_person_l3714_371452
