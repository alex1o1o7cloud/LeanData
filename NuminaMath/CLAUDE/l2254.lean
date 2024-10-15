import Mathlib

namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l2254_225499

theorem point_in_first_quadrant (x y : ℝ) : 
  (|3*x - 2*y - 1| + Real.sqrt (x + y - 2) = 0) → (x > 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l2254_225499


namespace NUMINAMATH_CALUDE_farmer_water_capacity_l2254_225458

/-- Calculates the total water capacity for a farmer's trucks -/
def total_water_capacity (num_trucks : ℕ) (tanks_per_truck : ℕ) (liters_per_tank : ℕ) : ℕ :=
  num_trucks * tanks_per_truck * liters_per_tank

/-- Theorem stating the total water capacity for the farmer's specific setup -/
theorem farmer_water_capacity :
  total_water_capacity 3 3 150 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_farmer_water_capacity_l2254_225458


namespace NUMINAMATH_CALUDE_angle_measure_problem_l2254_225483

theorem angle_measure_problem (C D : ℝ) : 
  C + D = 180 →  -- angles are supplementary
  C = 12 * D →   -- C is 12 times D
  C = 2160 / 13  -- measure of angle C
  := by sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l2254_225483


namespace NUMINAMATH_CALUDE_shirt_pricing_theorem_l2254_225461

/-- Represents the monthly sales volume as a function of price -/
def sales_volume (x : ℝ) : ℝ := -20 * x + 2600

/-- Represents the profit as a function of price -/
def profit (x : ℝ) : ℝ := (x - 50) * (sales_volume x)

/-- The cost price of each shirt -/
def cost_price : ℝ := 50

/-- The constraint that selling price is not less than cost price -/
def price_constraint (x : ℝ) : Prop := x ≥ cost_price

/-- The constraint that profit per unit should not exceed 30% of cost price -/
def profit_constraint (x : ℝ) : Prop := (x - cost_price) / cost_price ≤ 0.3

theorem shirt_pricing_theorem :
  ∃ (x : ℝ), price_constraint x ∧ profit x = 24000 ∧ x = 70 ∧
  ∃ (y : ℝ), price_constraint y ∧ profit_constraint y ∧
    (∀ z, price_constraint z → profit_constraint z → profit z ≤ profit y) ∧
    y = 65 ∧ profit y = 19500 := by sorry

end NUMINAMATH_CALUDE_shirt_pricing_theorem_l2254_225461


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2254_225424

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 4 * a * x + 3 > 0) → 0 ≤ a ∧ a < 3/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2254_225424


namespace NUMINAMATH_CALUDE_longest_side_l2254_225430

-- Define the triangle
def triangle (x : ℝ) := {a : ℝ // a = 7 ∨ a = x^2 + 4 ∨ a = 3*x + 1}

-- Define the perimeter condition
def perimeter_condition (x : ℝ) : Prop := 7 + (x^2 + 4) + (3*x + 1) = 45

-- State the theorem
theorem longest_side (x : ℝ) (h : perimeter_condition x) : 
  ∀ (side : triangle x), side.val ≤ x^2 + 4 :=
sorry

end NUMINAMATH_CALUDE_longest_side_l2254_225430


namespace NUMINAMATH_CALUDE_dryer_ball_savings_l2254_225403

/-- Calculates the savings from using wool dryer balls instead of dryer sheets over two years -/
theorem dryer_ball_savings :
  let loads_per_month : ℕ := 4 + 5 + 6 + 7
  let loads_per_year : ℕ := loads_per_month * 12
  let sheets_per_box : ℕ := 104
  let boxes_per_year : ℕ := (loads_per_year + sheets_per_box - 1) / sheets_per_box
  let initial_box_price : ℝ := 5.50
  let price_increase_rate : ℝ := 0.025
  let dryer_ball_price : ℝ := 15

  let first_year_cost : ℝ := boxes_per_year * initial_box_price
  let second_year_cost : ℝ := boxes_per_year * (initial_box_price * (1 + price_increase_rate))
  let total_sheet_cost : ℝ := first_year_cost + second_year_cost

  let savings : ℝ := total_sheet_cost - dryer_ball_price

  savings = 18.4125 := by sorry

end NUMINAMATH_CALUDE_dryer_ball_savings_l2254_225403


namespace NUMINAMATH_CALUDE_point_movement_and_linear_function_l2254_225401

theorem point_movement_and_linear_function (k : ℝ) : 
  let initial_point : ℝ × ℝ := (5, 3)
  let new_point : ℝ × ℝ := (initial_point.1 - 4, initial_point.2 - 1)
  new_point.2 = k * new_point.1 - 2 → k = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_point_movement_and_linear_function_l2254_225401


namespace NUMINAMATH_CALUDE_tulip_petals_l2254_225478

/-- Proves that each tulip has 3 petals given the conditions in Elena's garden --/
theorem tulip_petals (num_lilies : ℕ) (num_tulips : ℕ) (lily_petals : ℕ) (total_petals : ℕ)
  (h1 : num_lilies = 8)
  (h2 : num_tulips = 5)
  (h3 : lily_petals = 6)
  (h4 : total_petals = 63)
  (h5 : total_petals = num_lilies * lily_petals + num_tulips * (total_petals - num_lilies * lily_petals) / num_tulips) :
  (total_petals - num_lilies * lily_petals) / num_tulips = 3 := by
  sorry

#eval (63 - 8 * 6) / 5  -- This should output 3

end NUMINAMATH_CALUDE_tulip_petals_l2254_225478


namespace NUMINAMATH_CALUDE_smallest_a_value_l2254_225413

theorem smallest_a_value (a b d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) :
  (∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x + d)) →
  ∃ k : ℤ, a = 17 - 2 * Real.pi * ↑k ∧ 
    ∀ a' : ℝ, (0 ≤ a' ∧ ∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x + d)) → 
      17 - 2 * Real.pi * ↑k ≤ a' :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2254_225413


namespace NUMINAMATH_CALUDE_cube_properties_l2254_225481

/-- A cube is a convex polyhedron with specific properties -/
structure Cube where
  vertices : ℕ
  faces : ℕ
  edges : ℕ

/-- Euler's formula for convex polyhedrons -/
def euler_formula (c : Cube) : Prop :=
  c.vertices - c.edges + c.faces = 2

/-- Theorem stating the properties of a cube -/
theorem cube_properties : ∃ (c : Cube), c.vertices = 8 ∧ c.faces = 6 ∧ c.edges = 12 ∧ euler_formula c := by
  sorry

end NUMINAMATH_CALUDE_cube_properties_l2254_225481


namespace NUMINAMATH_CALUDE_domain_of_g_l2254_225474

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f (x : ℝ) : Prop := 0 ≤ x + 1 ∧ x + 1 ≤ 2

-- Define the function g
def g (x : ℝ) : ℝ := f (x + 3)

-- Theorem statement
theorem domain_of_g :
  (∀ x, domain_f x ↔ 0 ≤ x + 1 ∧ x + 1 ≤ 2) →
  (∀ x, g x = f (x + 3)) →
  (∀ x, g x ≠ 0 ↔ -3 ≤ x ∧ x ≤ -1) :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l2254_225474


namespace NUMINAMATH_CALUDE_cos_2018pi_over_3_l2254_225492

theorem cos_2018pi_over_3 : Real.cos (2018 * Real.pi / 3) = -(1 / 2) := by sorry

end NUMINAMATH_CALUDE_cos_2018pi_over_3_l2254_225492


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_3150_l2254_225488

theorem sum_of_prime_factors_3150 : (Finset.sum (Finset.filter Nat.Prime (Finset.range (3150 + 1))) id) = 17 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_3150_l2254_225488


namespace NUMINAMATH_CALUDE_square_root_of_64_l2254_225489

theorem square_root_of_64 : ∃ x : ℝ, x^2 = 64 ∧ (x = 8 ∨ x = -8) :=
  sorry

end NUMINAMATH_CALUDE_square_root_of_64_l2254_225489


namespace NUMINAMATH_CALUDE_marble_count_l2254_225415

/-- Represents a bag of marbles with red, blue, and green colors -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Theorem: Given a bag of marbles with the specified conditions, 
    prove the total number of marbles and the number of red marbles -/
theorem marble_count (bag : MarbleBag) 
  (ratio : bag.red * 3 * 4 = bag.blue * 2 * 4 ∧ bag.blue * 2 * 4 = bag.green * 2 * 3)
  (green_count : bag.green = 36) :
  bag.red + bag.blue + bag.green = 81 ∧ bag.red = 18 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_l2254_225415


namespace NUMINAMATH_CALUDE_least_clock_equivalent_is_nine_l2254_225498

/-- A number is clock equivalent to its square if their difference is divisible by 12 -/
def ClockEquivalent (n : ℕ) : Prop :=
  (n ^ 2 - n) % 12 = 0

/-- The least whole number greater than 4 that is clock equivalent to its square -/
def LeastClockEquivalent : ℕ := 9

theorem least_clock_equivalent_is_nine :
  (LeastClockEquivalent > 4) ∧
  ClockEquivalent LeastClockEquivalent ∧
  ∀ n : ℕ, (n > 4 ∧ n < LeastClockEquivalent) → ¬ClockEquivalent n :=
by sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_is_nine_l2254_225498


namespace NUMINAMATH_CALUDE_mode_most_relevant_for_sales_volume_l2254_225431

/-- Represents a shoe size -/
def ShoeSize := ℕ

/-- Represents a list of shoe sizes sold -/
def SalesList := List ShoeSize

/-- Calculates the mode of a list of shoe sizes -/
def mode (sales : SalesList) : ShoeSize :=
  sorry

/-- Represents the relevance of a statistical measure for determining the shoe size with highest sales volume -/
inductive Relevance
| Low : Relevance
| Medium : Relevance
| High : Relevance

/-- Determines the relevance of a statistical measure for sales volume prediction -/
def relevanceForSalesVolume (measure : String) : Relevance :=
  sorry

theorem mode_most_relevant_for_sales_volume :
  relevanceForSalesVolume "mode" = Relevance.High ∧
  (∀ m : String, m ≠ "mode" → relevanceForSalesVolume m ≠ Relevance.High) :=
sorry

end NUMINAMATH_CALUDE_mode_most_relevant_for_sales_volume_l2254_225431


namespace NUMINAMATH_CALUDE_jessica_probability_is_37_966_l2254_225449

/-- Represents the problem of distributing textbooks into boxes. -/
structure TextbookDistribution where
  total_books : Nat
  english_books : Nat
  box1_capacity : Nat
  box2_capacity : Nat
  box3_capacity : Nat
  box4_capacity : Nat

/-- The specific textbook distribution problem given in the question. -/
def jessica_distribution : TextbookDistribution :=
  { total_books := 15
  , english_books := 4
  , box1_capacity := 3
  , box2_capacity := 4
  , box3_capacity := 5
  , box4_capacity := 3
  }

/-- Calculates the probability of all English textbooks ending up in the third box. -/
def probability_all_english_in_third_box (d : TextbookDistribution) : Rat :=
  sorry

/-- Theorem stating that the probability for Jessica's distribution is 37/966. -/
theorem jessica_probability_is_37_966 :
  probability_all_english_in_third_box jessica_distribution = 37 / 966 := by
  sorry

end NUMINAMATH_CALUDE_jessica_probability_is_37_966_l2254_225449


namespace NUMINAMATH_CALUDE_janet_cat_collars_l2254_225421

/-- The number of inches of nylon needed for a dog collar -/
def dog_collar_nylon : ℕ := 18

/-- The number of inches of nylon needed for a cat collar -/
def cat_collar_nylon : ℕ := 10

/-- The total number of inches of nylon Janet needs -/
def total_nylon : ℕ := 192

/-- The number of dog collars Janet needs to make -/
def num_dog_collars : ℕ := 9

/-- Theorem stating that Janet needs to make 3 cat collars -/
theorem janet_cat_collars : 
  (total_nylon - num_dog_collars * dog_collar_nylon) / cat_collar_nylon = 3 := by
  sorry

end NUMINAMATH_CALUDE_janet_cat_collars_l2254_225421


namespace NUMINAMATH_CALUDE_quadratic_solutions_solution_at_minus_four_solution_at_minus_three_no_solution_at_minus_five_l2254_225471

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - (a - 12) * x + 36 - 5 * a

-- Define the condition for x
def x_condition (x : ℝ) : Prop := -6 < x ∧ x ≤ -2 ∧ x ≠ -5 ∧ x ≠ -4 ∧ x ≠ -3

-- Define the range for a
def a_range (a : ℝ) : Prop := (4 < a ∧ a < 4.5) ∨ (4.5 < a ∧ a ≤ 16/3)

-- Main theorem
theorem quadratic_solutions (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x_condition x₁ ∧ x_condition x₂ ∧ 
   quadratic a x₁ = 0 ∧ quadratic a x₂ = 0) ↔ a_range a :=
sorry

-- Theorems for specific points
theorem solution_at_minus_four :
  quadratic 4 (-4) = 0 :=
sorry

theorem solution_at_minus_three :
  quadratic 4.5 (-3) = 0 :=
sorry

theorem no_solution_at_minus_five (a : ℝ) :
  quadratic a (-5) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_solutions_solution_at_minus_four_solution_at_minus_three_no_solution_at_minus_five_l2254_225471


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2254_225418

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_difference : 
  let a := -8  -- First term
  let d := 7   -- Common difference (derived from -1 - (-8))
  let seq := arithmeticSequence a d
  (seq 110 - seq 100).natAbs = 70 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2254_225418


namespace NUMINAMATH_CALUDE_jelly_bean_division_l2254_225457

theorem jelly_bean_division (initial_amount : ℕ) (eaten_amount : ℕ) (num_piles : ℕ) :
  initial_amount = 36 →
  eaten_amount = 6 →
  num_piles = 3 →
  (initial_amount - eaten_amount) / num_piles = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_division_l2254_225457


namespace NUMINAMATH_CALUDE_complex_power_of_sqrt2i_l2254_225453

theorem complex_power_of_sqrt2i :
  ∀ z : ℂ, z = Complex.I * Real.sqrt 2 → z^4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_of_sqrt2i_l2254_225453


namespace NUMINAMATH_CALUDE_expression_bounds_l2254_225459

theorem expression_bounds (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  let expr := |x + y + z| / (|x| + |y| + |z|)
  (∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 → |a + b + c| / (|a| + |b| + |c|) ≤ 1) ∧
  (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ |a + b + c| / (|a| + |b| + |c|) = 0) ∧
  (1 - 0 = 1) := by
sorry

end NUMINAMATH_CALUDE_expression_bounds_l2254_225459


namespace NUMINAMATH_CALUDE_sum_of_sqrt_greater_than_one_l2254_225422

theorem sum_of_sqrt_greater_than_one 
  (x y z t : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0)
  (hxy : x ≠ y) (hxz : x ≠ z) (hxt : x ≠ t)
  (hyz : y ≠ z) (hyt : y ≠ t) (hzt : z ≠ t)
  (hsum : x + y + z + t = 1) :
  (Real.sqrt x + Real.sqrt y > 1) ∨
  (Real.sqrt x + Real.sqrt z > 1) ∨
  (Real.sqrt x + t > 1) ∨
  (Real.sqrt y + Real.sqrt z > 1) ∨
  (Real.sqrt y + Real.sqrt t > 1) ∨
  (Real.sqrt z + Real.sqrt t > 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_greater_than_one_l2254_225422


namespace NUMINAMATH_CALUDE_find_extremes_l2254_225448

/-- Represents the result of a weighing -/
inductive CompareResult
  | Less : CompareResult
  | Equal : CompareResult
  | Greater : CompareResult

/-- Represents a weight -/
structure Weight where
  id : Nat

/-- Represents a weighing operation -/
def weighing (w1 w2 : Weight) : CompareResult := sorry

/-- Represents the set of 5 weights -/
def Weights : Type := Fin 5 → Weight

/-- The heaviest weight in the set -/
def heaviest (ws : Weights) : Weight := sorry

/-- The lightest weight in the set -/
def lightest (ws : Weights) : Weight := sorry

/-- Axiom: Three weights have the same weight -/
axiom three_same_weight (ws : Weights) : 
  ∃ (i j k : Fin 5), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    weighing (ws i) (ws j) = CompareResult.Equal ∧ 
    weighing (ws j) (ws k) = CompareResult.Equal

/-- Axiom: One weight is heavier than the three identical weights -/
axiom one_heavier (ws : Weights) : 
  ∃ (h : Fin 5), ∀ (i : Fin 5), 
    weighing (ws i) (ws h) = CompareResult.Less ∨ 
    weighing (ws i) (ws h) = CompareResult.Equal

/-- Axiom: One weight is lighter than the three identical weights -/
axiom one_lighter (ws : Weights) : 
  ∃ (l : Fin 5), ∀ (i : Fin 5), 
    weighing (ws l) (ws i) = CompareResult.Less ∨ 
    weighing (ws l) (ws i) = CompareResult.Equal

/-- Theorem: It's possible to determine the heaviest and lightest weights in at most three weighings -/
theorem find_extremes (ws : Weights) : 
  ∃ (w1 w2 w3 w4 w5 w6 : Weight), 
    (weighing w1 w2 = CompareResult.Less ∨ 
     weighing w1 w2 = CompareResult.Equal ∨ 
     weighing w1 w2 = CompareResult.Greater) ∧
    (weighing w3 w4 = CompareResult.Less ∨ 
     weighing w3 w4 = CompareResult.Equal ∨ 
     weighing w3 w4 = CompareResult.Greater) ∧
    (weighing w5 w6 = CompareResult.Less ∨ 
     weighing w5 w6 = CompareResult.Equal ∨ 
     weighing w5 w6 = CompareResult.Greater) →
    (heaviest ws = heaviest ws ∧ lightest ws = lightest ws) :=
  sorry

end NUMINAMATH_CALUDE_find_extremes_l2254_225448


namespace NUMINAMATH_CALUDE_cistern_problem_solution_l2254_225482

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width water_breadth : ℝ) : ℝ :=
  let bottom_area := length * width
  let longer_side_area := 2 * length * water_breadth
  let shorter_side_area := 2 * width * water_breadth
  bottom_area + longer_side_area + shorter_side_area

/-- Theorem stating that the wet surface area of the given cistern is 121.5 m² -/
theorem cistern_problem_solution :
  cistern_wet_surface_area 9 6 2.25 = 121.5 := by
  sorry

#eval cistern_wet_surface_area 9 6 2.25

end NUMINAMATH_CALUDE_cistern_problem_solution_l2254_225482


namespace NUMINAMATH_CALUDE_patio_length_l2254_225463

theorem patio_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = 4 * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 100 →
  length = 40 := by
sorry

end NUMINAMATH_CALUDE_patio_length_l2254_225463


namespace NUMINAMATH_CALUDE_element_in_set_l2254_225407

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l2254_225407


namespace NUMINAMATH_CALUDE_solution_set_range_no_k_exists_positive_roots_k_range_l2254_225472

/-- The quadratic function y(x) = kx² - 2kx + 2k - 1 -/
def y (k x : ℝ) : ℝ := k * x^2 - 2 * k * x + 2 * k - 1

/-- The solution set of y ≥ 4k - 2 is all real numbers iff k ∈ [0, 1/3] -/
theorem solution_set_range (k : ℝ) :
  (∀ x, y k x ≥ 4 * k - 2) ↔ k ∈ Set.Icc 0 (1/3) := by sorry

/-- No k ∈ (0, 1) satisfies x₁² + x₂² = 3x₁x₂ - 4 for roots of y(x) = 0 -/
theorem no_k_exists (k : ℝ) (hk : k ∈ Set.Ioo 0 1) :
  ¬∃ x₁ x₂ : ℝ, y k x₁ = 0 ∧ y k x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + x₂^2 = 3*x₁*x₂ - 4 := by sorry

/-- If roots of y(x) = 0 are positive, then k ∈ (1/2, 1) -/
theorem positive_roots_k_range (k : ℝ) :
  (∃ x₁ x₂ : ℝ, y k x₁ = 0 ∧ y k x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0) →
  k ∈ Set.Ioo (1/2) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_range_no_k_exists_positive_roots_k_range_l2254_225472


namespace NUMINAMATH_CALUDE_silverware_probability_l2254_225476

/-- The number of each type and color of silverware in the drawer -/
def num_each : ℕ := 8

/-- The total number of pieces of silverware in the drawer -/
def total_pieces : ℕ := 6 * num_each

/-- The number of ways to choose any 3 items from the drawer -/
def total_ways : ℕ := Nat.choose total_pieces 3

/-- The number of ways to choose one fork, one spoon, and one knife of different colors -/
def favorable_ways : ℕ := 2 * (num_each * num_each * num_each)

/-- The probability of selecting one fork, one spoon, and one knife of different colors -/
def probability : ℚ := favorable_ways / total_ways

theorem silverware_probability :
  probability = 32 / 541 := by sorry

end NUMINAMATH_CALUDE_silverware_probability_l2254_225476


namespace NUMINAMATH_CALUDE_power_of_power_l2254_225493

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2254_225493


namespace NUMINAMATH_CALUDE_prob_square_divisor_15_factorial_l2254_225411

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of positive integer divisors of n that are perfect squares -/
def num_square_divisors (n : ℕ) : ℕ := sorry

/-- The total number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The probability of choosing a perfect square divisor from the positive integer divisors of n -/
def prob_square_divisor (n : ℕ) : ℚ :=
  (num_square_divisors n : ℚ) / (num_divisors n : ℚ)

theorem prob_square_divisor_15_factorial :
  prob_square_divisor (factorial 15) = 1 / 36 := by sorry

end NUMINAMATH_CALUDE_prob_square_divisor_15_factorial_l2254_225411


namespace NUMINAMATH_CALUDE_sweets_distribution_l2254_225412

theorem sweets_distribution (total_children : Nat) (absent_children : Nat) (extra_sweets : Nat) 
  (h1 : total_children = 256)
  (h2 : absent_children = 64)
  (h3 : extra_sweets = 12) :
  let original_sweets := (total_children - absent_children) * extra_sweets / absent_children
  original_sweets = 36 := by
sorry

end NUMINAMATH_CALUDE_sweets_distribution_l2254_225412


namespace NUMINAMATH_CALUDE_inverse_function_point_sum_l2254_225417

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverse functions
axiom inverse_relation : ∀ x, f_inv (f x) = x

-- Define the condition that (1,2) is on the graph of y = f(x)/2
axiom point_on_graph : f 1 = 4

-- Theorem to prove
theorem inverse_function_point_sum :
  ∃ a b : ℝ, f_inv a = 2*b ∧ a + b = 9/2 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_point_sum_l2254_225417


namespace NUMINAMATH_CALUDE_inequality_and_range_l2254_225475

-- Define the function f
def f (x : ℝ) : ℝ := |3 * x + 2|

-- Define the theorem
theorem inequality_and_range :
  -- Part I: Solution set of f(x) < 4 - |x-1|
  (∀ x : ℝ, f x < 4 - |x - 1| ↔ x > -5/4 ∧ x < 1/2) ∧
  -- Part II: Range of a
  (∀ m n a : ℝ, m > 0 → n > 0 → m + n = 1 → a > 0 →
    (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) →
    a ≤ 10/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_range_l2254_225475


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_range_l2254_225419

/-- A function f with an extreme value only at x = 0 -/
def f (a b x : ℝ) : ℝ := x^4 + a*x^3 + 2*x^2 + b

/-- f has an extreme value only at x = 0 -/
def has_extreme_only_at_zero (a b : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → ¬(∀ y : ℝ, f a b y ≤ f a b x ∨ ∀ y : ℝ, f a b y ≥ f a b x)

/-- The main theorem: if f has an extreme value only at x = 0, then -8/3 ≤ a ≤ 8/3 -/
theorem extreme_value_implies_a_range (a b : ℝ) :
  has_extreme_only_at_zero a b → -8/3 ≤ a ∧ a ≤ 8/3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_range_l2254_225419


namespace NUMINAMATH_CALUDE_unique_solution_mod_151_l2254_225420

theorem unique_solution_mod_151 :
  ∃! n : ℤ, 0 ≤ n ∧ n < 151 ∧ (150 * n) % 151 = 93 % 151 ∧ n = 58 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mod_151_l2254_225420


namespace NUMINAMATH_CALUDE_pyramid_base_side_length_l2254_225406

/-- A square pyramid with a hemisphere resting on its base -/
structure PyramidWithHemisphere where
  /-- Height of the pyramid -/
  pyramidHeight : ℝ
  /-- Radius of the hemisphere -/
  hemisphereRadius : ℝ
  /-- The hemisphere is tangent to all four faces and the base of the pyramid -/
  isTangent : Bool

/-- Calculate the side length of the square base of the pyramid -/
def calculateBaseSideLength (p : PyramidWithHemisphere) : ℝ :=
  sorry

/-- Theorem stating that for a pyramid of height 9 and hemisphere of radius 3,
    the side length of the base is 9 -/
theorem pyramid_base_side_length 
  (p : PyramidWithHemisphere) 
  (h1 : p.pyramidHeight = 9) 
  (h2 : p.hemisphereRadius = 3) 
  (h3 : p.isTangent = true) : 
  calculateBaseSideLength p = 9 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_base_side_length_l2254_225406


namespace NUMINAMATH_CALUDE_sum_of_x1_and_x2_l2254_225408

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- State the theorem
theorem sum_of_x1_and_x2 (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ → f x₁ = 101 → f x₂ = 101 → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x1_and_x2_l2254_225408


namespace NUMINAMATH_CALUDE_fraction_simplification_l2254_225402

theorem fraction_simplification :
  (1 - 2 - 4 + 8 + 16 + 32 - 64 + 128 - 256) /
  (2 - 4 - 8 + 16 + 32 + 64 - 128 + 256) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2254_225402


namespace NUMINAMATH_CALUDE_log_sum_two_five_equals_one_l2254_225409

theorem log_sum_two_five_equals_one : Real.log 2 + Real.log 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_two_five_equals_one_l2254_225409


namespace NUMINAMATH_CALUDE_squares_in_figure_100_l2254_225497

-- Define the sequence function
def f (n : ℕ) : ℕ := 2 * n^3 + 2 * n^2 + 4 * n + 1

-- State the theorem
theorem squares_in_figure_100 :
  f 0 = 1 ∧ f 1 = 9 ∧ f 2 = 29 ∧ f 3 = 65 → f 100 = 2020401 :=
by
  sorry


end NUMINAMATH_CALUDE_squares_in_figure_100_l2254_225497


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_8250_l2254_225484

theorem largest_prime_factor_of_8250 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 8250 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 8250 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_8250_l2254_225484


namespace NUMINAMATH_CALUDE_complement_of_A_l2254_225447

def A : Set ℝ := {x | (x - 1) / (x - 2) ≥ 0}

theorem complement_of_A : (Set.univ \ A) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2254_225447


namespace NUMINAMATH_CALUDE_stratified_sampling_elderly_count_l2254_225440

def total_population : ℕ := 180
def elderly_population : ℕ := 30
def sample_size : ℕ := 36

theorem stratified_sampling_elderly_count :
  (elderly_population * sample_size) / total_population = 6 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_elderly_count_l2254_225440


namespace NUMINAMATH_CALUDE_warehouse_capacity_is_510_l2254_225468

/-- The total capacity of a grain-storage warehouse --/
def warehouse_capacity (total_bins : ℕ) (large_bins : ℕ) (large_capacity : ℕ) (small_capacity : ℕ) : ℕ :=
  large_bins * large_capacity + (total_bins - large_bins) * small_capacity

/-- Theorem: The warehouse capacity is 510 tons --/
theorem warehouse_capacity_is_510 :
  warehouse_capacity 30 12 20 15 = 510 :=
by sorry

end NUMINAMATH_CALUDE_warehouse_capacity_is_510_l2254_225468


namespace NUMINAMATH_CALUDE_one_common_root_sum_l2254_225427

theorem one_common_root_sum (a b : ℝ) :
  (∃! x, x^2 + a*x + b = 0 ∧ x^2 + b*x + a = 0) →
  a + b = -1 :=
by sorry

end NUMINAMATH_CALUDE_one_common_root_sum_l2254_225427


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l2254_225462

-- Define the set of people
inductive Person : Type
  | A
  | B
  | C

-- Define the set of cards
inductive Card : Type
  | Red
  | Yellow
  | Blue

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "Person A gets the red card"
def event_A (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Person B gets the red card"
def event_B (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(event_A d ∧ event_B d)) ∧
  (∃ d : Distribution, ¬event_A d ∧ ¬event_B d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l2254_225462


namespace NUMINAMATH_CALUDE_factor_calculation_l2254_225436

theorem factor_calculation (original : ℝ) (factor : ℝ) : 
  original = 5 → 
  (2 * original + 9) * factor = 57 → 
  factor = 3 := by
sorry

end NUMINAMATH_CALUDE_factor_calculation_l2254_225436


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2254_225456

-- Define the equations
def equation1 (x : ℝ) : Prop := 6 * x - 7 = 4 * x - 5
def equation2 (x : ℝ) : Prop := (x + 1) / 2 - 1 = 2 + (2 - x) / 4

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1 := by
  sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2254_225456


namespace NUMINAMATH_CALUDE_possible_regions_l2254_225425

/-- The number of lines dividing the plane -/
def num_lines : ℕ := 99

/-- The function that calculates the number of regions based on the number of parallel lines -/
def num_regions (k : ℕ) : ℕ := (k + 1) * (100 - k)

/-- The theorem stating the possible values of n less than 199 -/
theorem possible_regions :
  ∀ n : ℕ, n < 199 →
    (∃ k : ℕ, k ≤ num_lines ∧ n = num_regions k) →
    n = 100 ∨ n = 198 := by
  sorry

end NUMINAMATH_CALUDE_possible_regions_l2254_225425


namespace NUMINAMATH_CALUDE_at_op_four_nine_l2254_225487

-- Define the operation @
def at_op (a b : ℝ) : ℝ := a * b ^ (1 / 2)

-- Theorem statement
theorem at_op_four_nine : at_op 4 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_at_op_four_nine_l2254_225487


namespace NUMINAMATH_CALUDE_square_difference_l2254_225464

theorem square_difference (x y : ℝ) : (x - y) * (x - y) = x^2 - 2*x*y + y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2254_225464


namespace NUMINAMATH_CALUDE_simplify_expression_l2254_225416

theorem simplify_expression (x y : ℝ) : 7 * x + 8 - 3 * x + 15 - 2 * y = 4 * x - 2 * y + 23 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2254_225416


namespace NUMINAMATH_CALUDE_triangle_sine_problem_l2254_225410

theorem triangle_sine_problem (D E F : ℝ) (h_area : (1/2) * D * E * Real.sin F = 100) 
  (h_geom_mean : Real.sqrt (D * E) = 15) : Real.sin F = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_problem_l2254_225410


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l2254_225495

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 4) (h2 : x^2 + y^2 = 8) : x^3 + y^3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l2254_225495


namespace NUMINAMATH_CALUDE_balloon_difference_l2254_225465

theorem balloon_difference (yellow_balloons : ℕ) (total_balloons : ℕ) (school_balloons : ℕ) :
  yellow_balloons = 3414 →
  total_balloons % 10 = 0 →
  total_balloons / 10 = school_balloons →
  school_balloons = 859 →
  total_balloons > 2 * yellow_balloons →
  total_balloons - 2 * yellow_balloons = 1762 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l2254_225465


namespace NUMINAMATH_CALUDE_arthur_arrival_speed_l2254_225429

theorem arthur_arrival_speed :
  ∀ (distance : ℝ) (n : ℝ),
    (distance / 60 = distance / n + 1/12) →
    (distance / 90 = distance / n - 1/12) →
    n = 72 := by
  sorry

end NUMINAMATH_CALUDE_arthur_arrival_speed_l2254_225429


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_l2254_225434

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 72 ∣ m^3) : 
  ∀ k : ℕ, k ∣ m → k ≤ 6 ∧ 6 ∣ m :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_l2254_225434


namespace NUMINAMATH_CALUDE_ramanujan_identity_a_l2254_225496

theorem ramanujan_identity_a : 
  (((2 : ℝ) ^ (1/3) - 1) ^ (1/3) = (1/9 : ℝ) ^ (1/3) - (2/9 : ℝ) ^ (1/3) + (4/9 : ℝ) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_identity_a_l2254_225496


namespace NUMINAMATH_CALUDE_fraction_value_l2254_225486

theorem fraction_value (a b : ℝ) (h : 1/a - 1/b = 4) :
  (a - 2*a*b - b) / (2*a - 2*b + 7*a*b) = 6 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l2254_225486


namespace NUMINAMATH_CALUDE_price_change_l2254_225494

theorem price_change (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * 1.3
  let final_price := increased_price * 0.75
  final_price = original_price * 0.975 :=
by sorry

end NUMINAMATH_CALUDE_price_change_l2254_225494


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l2254_225450

/-- The equation represents two lines if it can be rewritten in the form of two linear equations -/
def represents_two_lines (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), ∀ x y, f x y = 0 ↔ (x = a * y + b ∨ x = c * y + d)

/-- The given equation -/
def equation (x y : ℝ) : ℝ :=
  x^2 - 25 * y^2 - 10 * x + 50

theorem equation_represents_two_lines :
  represents_two_lines equation :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l2254_225450


namespace NUMINAMATH_CALUDE_pentagon_angle_sum_l2254_225444

theorem pentagon_angle_sum (P Q R a b : ℝ) : 
  P = 34 → Q = 82 → R = 30 → 
  (P + Q + (360 - a) + 90 + (120 - b) = 540) → 
  a + b = 146 := by sorry

end NUMINAMATH_CALUDE_pentagon_angle_sum_l2254_225444


namespace NUMINAMATH_CALUDE_best_approximation_log5_10_l2254_225470

/-- Approximation of log₁₀2 -/
def log10_2 : ℝ := 0.301

/-- Approximation of log₁₀3 -/
def log10_3 : ℝ := 0.477

/-- The set of possible fractions for approximating log₅10 -/
def fraction_options : List ℚ := [8/7, 9/7, 10/7, 11/7, 12/7]

/-- Statement: The fraction 10/7 is the closest approximation to log₅10 among the given options -/
theorem best_approximation_log5_10 : 
  ∃ (x : ℚ), x ∈ fraction_options ∧ 
  ∀ (y : ℚ), y ∈ fraction_options → |x - (1 / (1 - log10_2))| ≤ |y - (1 / (1 - log10_2))| ∧
  x = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_best_approximation_log5_10_l2254_225470


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l2254_225404

/-- Prove that the number of boys who are neither happy nor sad is 5 -/
theorem boys_neither_happy_nor_sad (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neither_children : ℕ) (total_boys : ℕ) (total_girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neither_children = 20 →
  total_boys = 17 →
  total_girls = 43 →
  happy_boys = 6 →
  sad_girls = 4 →
  total_children = happy_children + sad_children + neither_children →
  total_children = total_boys + total_girls →
  (total_boys - happy_boys - (sad_children - sad_girls) : ℤ) = 5 := by
sorry


end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l2254_225404


namespace NUMINAMATH_CALUDE_line_l_satisfies_conditions_l2254_225467

-- Define the points
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (5, 4)
def C : ℝ × ℝ := (3, 7)
def D : ℝ × ℝ := (7, 1)
def E : ℝ × ℝ := (10, 2)
def F : ℝ × ℝ := (8, 6)

-- Define the line l
def l (x y : ℝ) : Prop := 10 * x - 2 * y - 55 = 0

-- Define the line DF
def DF (x y : ℝ) : Prop := y = 5 * x - 34

-- Define the triangles ABC and DEF
def triangle_ABC : Set (ℝ × ℝ) :=
  {p | ∃ (t₁ t₂ t₃ : ℝ), t₁ ≥ 0 ∧ t₂ ≥ 0 ∧ t₃ ≥ 0 ∧ t₁ + t₂ + t₃ = 1 ∧
    p = (t₁ * A.1 + t₂ * B.1 + t₃ * C.1, t₁ * A.2 + t₂ * B.2 + t₃ * C.2)}

def triangle_DEF : Set (ℝ × ℝ) :=
  {p | ∃ (t₁ t₂ t₃ : ℝ), t₁ ≥ 0 ∧ t₂ ≥ 0 ∧ t₃ ≥ 0 ∧ t₁ + t₂ + t₃ = 1 ∧
    p = (t₁ * D.1 + t₂ * E.1 + t₃ * F.1, t₁ * D.2 + t₂ * E.2 + t₃ * F.2)}

-- Define the distance function
def distance (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  sorry -- Implementation of distance function

theorem line_l_satisfies_conditions :
  (∀ x y, l x y → DF x y) ∧ -- l is parallel to DF
  (∃ d : ℝ, 
    (∀ p ∈ triangle_ABC, distance p l ≥ d) ∧
    (∃ p₁ ∈ triangle_ABC, distance p₁ l = d) ∧
    (∀ p ∈ triangle_DEF, distance p l ≥ d) ∧
    (∃ p₂ ∈ triangle_DEF, distance p₂ l = d)) :=
by sorry

end NUMINAMATH_CALUDE_line_l_satisfies_conditions_l2254_225467


namespace NUMINAMATH_CALUDE_caps_first_week_l2254_225405

/-- The number of caps made in the first week -/
def first_week : ℕ := sorry

/-- The number of caps made in the second week -/
def second_week : ℕ := 400

/-- The number of caps made in the third week -/
def third_week : ℕ := 300

/-- The total number of caps made in four weeks -/
def total_caps : ℕ := 1360

theorem caps_first_week : 
  first_week = 320 ∧
  second_week = 400 ∧
  third_week = 300 ∧
  first_week + second_week + third_week + (first_week + second_week + third_week) / 3 = total_caps :=
by sorry

end NUMINAMATH_CALUDE_caps_first_week_l2254_225405


namespace NUMINAMATH_CALUDE_largest_square_area_l2254_225423

theorem largest_square_area (x y z : ℝ) (h1 : x^2 + y^2 = z^2) 
  (h2 : x^2 + y^2 + 2*z^2 = 722) : z^2 = 722/3 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l2254_225423


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2254_225400

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q ^ (n - 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  a 3 + a 5 = 6 →
  a 5 + a 7 + a 9 = 28 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2254_225400


namespace NUMINAMATH_CALUDE_right_triangle_circle_and_trajectory_l2254_225441

/-- Right triangle ABC with hypotenuse AB, where A(-1,0) and B(3,0) -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  hA : A = (-1, 0)
  hB : B = (3, 0)
  isRightTriangle : sorry -- Assume this triangle is right-angled

/-- The general equation of a circle -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The equation of a trajectory -/
def TrajectoryEquation (a b r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

theorem right_triangle_circle_and_trajectory 
  (triangle : RightTriangle) (x y : ℝ) (hy : y ≠ 0) :
  (CircleEquation 1 0 2 x y ↔ x^2 + y^2 - 2*x - 3 = 0) ∧
  (TrajectoryEquation 2 0 1 x y ↔ (x-2)^2 + y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circle_and_trajectory_l2254_225441


namespace NUMINAMATH_CALUDE_parabola_translation_l2254_225428

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally -/
def translate_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c }

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b
  , c := p.c + v }

theorem parabola_translation :
  let p1 : Parabola := { a := 2, b := 4, c := -3 }  -- y = 2(x+1)^2 - 3
  let p2 : Parabola := translate_vertical (translate_horizontal p1 1) 3
  p2 = { a := 2, b := 0, c := 0 }  -- y = 2x^2
  := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2254_225428


namespace NUMINAMATH_CALUDE_triangle_radius_inequalities_l2254_225454

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define perimeter, circumradius, and inradius
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c
def circumradius (t : Triangle) : ℝ := sorry
def inradius (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_radius_inequalities :
  ∃ t1 t2 t3 : Triangle,
    ¬(perimeter t1 > circumradius t1 + inradius t1) ∧
    ¬(perimeter t2 ≤ circumradius t2 + inradius t2) ∧
    ¬(perimeter t3 / 6 < circumradius t3 + inradius t3 ∧ circumradius t3 + inradius t3 < 6 * perimeter t3) :=
  sorry

end NUMINAMATH_CALUDE_triangle_radius_inequalities_l2254_225454


namespace NUMINAMATH_CALUDE_yellas_computer_usage_l2254_225473

theorem yellas_computer_usage (last_week_hours : ℕ) (reduction : ℕ) : 
  last_week_hours = 91 → 
  reduction = 35 → 
  (last_week_hours - reduction) / 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_yellas_computer_usage_l2254_225473


namespace NUMINAMATH_CALUDE_polynomial_equality_l2254_225479

theorem polynomial_equality (a b c : ℤ) :
  (∀ x : ℝ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) →
  (a = 3 ∨ a = 7) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2254_225479


namespace NUMINAMATH_CALUDE_expand_expression_l2254_225460

theorem expand_expression (x : ℝ) : (x + 3) * (4 * x - 8) - 2 * x = 4 * x^2 + 2 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2254_225460


namespace NUMINAMATH_CALUDE_no_real_solutions_l2254_225491

theorem no_real_solutions : ¬∃ (x y : ℝ), x^2 + 3*y^2 - 4*x - 6*y + 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2254_225491


namespace NUMINAMATH_CALUDE_not_divisible_by_100_l2254_225442

theorem not_divisible_by_100 : ∀ n : ℕ, ¬(100 ∣ (n^2 + 6*n + 2019)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_100_l2254_225442


namespace NUMINAMATH_CALUDE_exponential_max_greater_than_min_l2254_225438

theorem exponential_max_greater_than_min (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), x ∈ Set.Icc 1 2 ∧ y ∈ Set.Icc 1 2 ∧ a^x > a^y :=
sorry

end NUMINAMATH_CALUDE_exponential_max_greater_than_min_l2254_225438


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l2254_225469

/-- Three points in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point2D) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

theorem collinear_points_x_value :
  let p : Point2D := ⟨1, 1⟩
  let a : Point2D := ⟨2, -4⟩
  let b : Point2D := ⟨x, -9⟩
  collinear p a b → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_x_value_l2254_225469


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2254_225439

theorem rectangular_to_polar_conversion :
  ∀ (x y r θ : ℝ),
  x = 8 ∧ y = 2 * Real.sqrt 3 →
  r = Real.sqrt (x^2 + y^2) →
  θ = Real.arctan (y / x) →
  r > 0 →
  0 ≤ θ ∧ θ < 2 * Real.pi →
  r = 2 * Real.sqrt 19 ∧ θ = Real.arctan (Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2254_225439


namespace NUMINAMATH_CALUDE_arithmetic_sequence_value_increasing_sequence_set_l2254_225414

def sequence_sum (a : ℝ) (n : ℕ) : ℝ := sorry

def sequence_term (a : ℝ) (n : ℕ) : ℝ := sorry

axiom sequence_sum_property (a : ℝ) (n : ℕ) :
  n ≥ 2 → (sequence_sum a n)^2 = 3 * n^2 * (sequence_term a n) + (sequence_sum a (n-1))^2

axiom nonzero_terms (a : ℝ) (n : ℕ) : sequence_term a n ≠ 0

def is_arithmetic_sequence (a : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → sequence_term a (n+1) - sequence_term a n = sequence_term a n - sequence_term a (n-1)

def is_increasing_sequence (a : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → sequence_term a n < sequence_term a (n+1)

theorem arithmetic_sequence_value (a : ℝ) :
  is_arithmetic_sequence a → a = 3 := sorry

theorem increasing_sequence_set :
  {a : ℝ | is_increasing_sequence a} = Set.Ioo (9/4) (15/4) := sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_value_increasing_sequence_set_l2254_225414


namespace NUMINAMATH_CALUDE_cricket_problem_l2254_225432

/-- Represents the runs scored by each batsman in a cricket match -/
structure CricketScores where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ

/-- Theorem representing the cricket problem -/
theorem cricket_problem (scores : CricketScores) : scores.E = 20 :=
  by
  have h1 : scores.A + scores.B + scores.C + scores.D + scores.E = 180 := 
    sorry -- Average score is 36, so total is 5 * 36 = 180
  have h2 : scores.D = scores.E + 5 := 
    sorry -- D scored 5 more than E
  have h3 : scores.E = scores.A - 8 := 
    sorry -- E scored 8 fewer than A
  have h4 : scores.B = scores.D + scores.E := 
    sorry -- B scored as many as D and E combined
  have h5 : scores.B + scores.C = 107 := 
    sorry -- B and C scored 107 between them
  sorry -- Proof that E = 20

end NUMINAMATH_CALUDE_cricket_problem_l2254_225432


namespace NUMINAMATH_CALUDE_obrien_hats_count_l2254_225490

/-- The number of hats Fire chief Simpson has -/
def simpson_hats : ℕ := 15

/-- The initial number of hats Policeman O'Brien had -/
def obrien_initial_hats : ℕ := 2 * simpson_hats + 5

/-- The number of hats Policeman O'Brien lost -/
def obrien_lost_hats : ℕ := 1

/-- The current number of hats Policeman O'Brien has -/
def obrien_current_hats : ℕ := obrien_initial_hats - obrien_lost_hats

theorem obrien_hats_count : obrien_current_hats = 34 := by
  sorry

end NUMINAMATH_CALUDE_obrien_hats_count_l2254_225490


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2254_225480

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given conditions
  c = Real.sqrt 2 →
  Real.cos C = 3/4 →
  2 * c * Real.sin A = b * Real.sin C →
  -- Conclusions
  b = 2 ∧
  Real.sin A = Real.sqrt 14 / 8 ∧
  Real.sin (2 * A + π/6) = (5 * Real.sqrt 21 + 9) / 32 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2254_225480


namespace NUMINAMATH_CALUDE_no_point_satisfies_both_systems_l2254_225485

/-- A point in the 2D plane satisfies System I if it meets all these conditions -/
def satisfies_system_I (x y : ℝ) : Prop :=
  y < 3 ∧ x - y < 3 ∧ x + y < 4

/-- A point in the 2D plane satisfies System II if it meets all these conditions -/
def satisfies_system_II (x y : ℝ) : Prop :=
  (y - 3) * (x - y - 3) ≥ 0 ∧
  (y - 3) * (x + y - 4) ≤ 0 ∧
  (x - y - 3) * (x + y - 4) ≤ 0

/-- There is no point that satisfies both System I and System II -/
theorem no_point_satisfies_both_systems :
  ¬ ∃ (x y : ℝ), satisfies_system_I x y ∧ satisfies_system_II x y :=
by sorry

end NUMINAMATH_CALUDE_no_point_satisfies_both_systems_l2254_225485


namespace NUMINAMATH_CALUDE_abs_equation_solution_set_l2254_225426

theorem abs_equation_solution_set (x : ℝ) :
  |2*x - 1| = |x| + |x - 1| ↔ x ≤ 0 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_set_l2254_225426


namespace NUMINAMATH_CALUDE_sum_of_roots_l2254_225455

theorem sum_of_roots (k d x₁ x₂ : ℝ) 
  (h₁ : x₁ ≠ x₂)
  (h₂ : 5 * x₁^2 - k * x₁ = d)
  (h₃ : 5 * x₂^2 - k * x₂ = d) :
  x₁ + x₂ = k / 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2254_225455


namespace NUMINAMATH_CALUDE_quadratic_one_root_l2254_225452

/-- A quadratic function f(x) = x^2 - 2x + m has exactly one root if and only if m = 1 -/
theorem quadratic_one_root (m : ℝ) : 
  (∃! x, x^2 - 2*x + m = 0) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l2254_225452


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2254_225446

theorem intersection_of_lines : ∃! p : ℚ × ℚ, 
  8 * p.1 - 5 * p.2 = 20 ∧ 6 * p.1 + 2 * p.2 = 18 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l2254_225446


namespace NUMINAMATH_CALUDE_boris_bowls_l2254_225451

def candy_distribution (initial_candy : ℕ) (daughter_eats : ℕ) (boris_takes : ℕ) (remaining_in_bowl : ℕ) : ℕ :=
  let remaining_candy := initial_candy - daughter_eats
  let pieces_per_bowl := remaining_in_bowl + boris_takes
  remaining_candy / pieces_per_bowl

theorem boris_bowls :
  candy_distribution 100 8 3 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_boris_bowls_l2254_225451


namespace NUMINAMATH_CALUDE_f_equals_g_l2254_225477

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l2254_225477


namespace NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l2254_225443

theorem no_real_roots_for_nonzero_k :
  ∀ k : ℝ, k ≠ 0 → ¬∃ x : ℝ, x^2 + k*x + k^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l2254_225443


namespace NUMINAMATH_CALUDE_wang_loss_is_97_l2254_225437

-- Define the relevant quantities
def gift_cost : ℕ := 18
def gift_price : ℕ := 21
def payment : ℕ := 100
def change_given : ℕ := 79
def counterfeit_bill : ℕ := 100
def neighbor_repayment : ℕ := 100

-- Define Mr. Wang's loss
def wang_loss : ℕ := change_given + gift_cost + neighbor_repayment - payment

-- Theorem statement
theorem wang_loss_is_97 : wang_loss = 97 := by
  sorry

end NUMINAMATH_CALUDE_wang_loss_is_97_l2254_225437


namespace NUMINAMATH_CALUDE_circular_garden_radius_l2254_225445

theorem circular_garden_radius (r : ℝ) : r > 0 → 2 * Real.pi * r = (1 / 5) * Real.pi * r^2 → r = 10 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l2254_225445


namespace NUMINAMATH_CALUDE_fourth_term_is_27_l2254_225466

def S (n : ℕ) : ℤ := 4 * n^2 - n - 8

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem fourth_term_is_27 : a 4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_27_l2254_225466


namespace NUMINAMATH_CALUDE_teacher_assignment_count_l2254_225435

/-- The number of ways to assign teachers to classes -/
def assign_teachers (n : ℕ) (m : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- The number of intern teachers -/
def num_teachers : ℕ := 4

/-- The number of classes -/
def num_classes : ℕ := 3

/-- Theorem stating that the number of ways to assign 4 teachers to 3 classes,
    with each class having at least 1 teacher, is 36 -/
theorem teacher_assignment_count :
  assign_teachers num_teachers num_classes = 36 :=
sorry

end NUMINAMATH_CALUDE_teacher_assignment_count_l2254_225435


namespace NUMINAMATH_CALUDE_parabola_equation_l2254_225433

/-- A parabola passing through two points on the x-axis -/
structure Parabola where
  a : ℝ
  b : ℝ
  eval : ℝ → ℝ := fun x => a * x^2 + b * x - 5

/-- The parabola passes through the points (-1,0) and (5,0) -/
def passes_through (p : Parabola) : Prop :=
  p.eval (-1) = 0 ∧ p.eval 5 = 0

/-- The theorem stating that the parabola passing through (-1,0) and (5,0) has the equation y = x² - 4x - 5 -/
theorem parabola_equation (p : Parabola) (h : passes_through p) :
  p.a = 1 ∧ p.b = -4 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2254_225433
