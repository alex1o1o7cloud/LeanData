import Mathlib

namespace NUMINAMATH_CALUDE_problem_1_problem_2_l333_33392

def A : Set ℝ := {x | 6 / (x + 1) ≥ 1}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

theorem problem_1 : A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

theorem problem_2 : ∃ m : ℝ, A ∩ B m = {x | -1 < x ∧ x < 4} ∧ m = 8 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l333_33392


namespace NUMINAMATH_CALUDE_expression_simplification_l333_33300

theorem expression_simplification (x : ℝ) :
  (3*x^3 + 4*x^2 + 5)*(2*x - 1) - (2*x - 1)*(x^2 + 2*x - 8) + (x^2 - 2*x + 3)*(2*x - 1)*(x - 2) =
  8*x^4 - 2*x^3 - 5*x^2 + 32*x - 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l333_33300


namespace NUMINAMATH_CALUDE_fib_arithmetic_seq_solution_l333_33312

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Property of three consecutive Fibonacci numbers forming an arithmetic sequence -/
def is_fib_arithmetic_seq (a b c : ℕ) : Prop :=
  fib b - fib a = fib c - fib b ∧ fib a < fib b ∧ fib b < fib c

theorem fib_arithmetic_seq_solution :
  ∃ a b c : ℕ, is_fib_arithmetic_seq a b c ∧ a + b + c = 3000 ∧ a = 998 := by
  sorry


end NUMINAMATH_CALUDE_fib_arithmetic_seq_solution_l333_33312


namespace NUMINAMATH_CALUDE_furniture_shop_cost_price_l333_33308

/-- Proves that if a shop owner charges 20% more than the cost price,
    and a customer paid 3600 for an item, then the cost price was 3000. -/
theorem furniture_shop_cost_price 
  (markup_percentage : ℝ) 
  (selling_price : ℝ) 
  (cost_price : ℝ) : 
  markup_percentage = 0.20 →
  selling_price = 3600 →
  selling_price = cost_price * (1 + markup_percentage) →
  cost_price = 3000 := by
sorry

end NUMINAMATH_CALUDE_furniture_shop_cost_price_l333_33308


namespace NUMINAMATH_CALUDE_bug_path_tiles_l333_33331

/-- Represents the number of tiles visited by a bug walking diagonally across a rectangular grid -/
def tilesVisited (width : ℕ) (length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- The playground dimensions in tile units -/
def playground_width : ℕ := 6
def playground_length : ℕ := 13

theorem bug_path_tiles :
  tilesVisited playground_width playground_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l333_33331


namespace NUMINAMATH_CALUDE_factory_bulb_reliability_l333_33346

theorem factory_bulb_reliability 
  (factory_x_reliability : ℝ) 
  (factory_x_supply : ℝ) 
  (total_reliability : ℝ) 
  (h1 : factory_x_reliability = 0.59) 
  (h2 : factory_x_supply = 0.60) 
  (h3 : total_reliability = 0.62) :
  let factory_y_supply := 1 - factory_x_supply
  let factory_y_reliability := (total_reliability - factory_x_supply * factory_x_reliability) / factory_y_supply
  factory_y_reliability = 0.665 := by
sorry

end NUMINAMATH_CALUDE_factory_bulb_reliability_l333_33346


namespace NUMINAMATH_CALUDE_opposite_of_abs_neg_pi_l333_33345

theorem opposite_of_abs_neg_pi : -(|-π|) = -π := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_abs_neg_pi_l333_33345


namespace NUMINAMATH_CALUDE_triangle_lines_correct_l333_33362

/-- Triangle with vertices A(-5,0), B(3,-3), and C(0,2) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def triangle : Triangle := { A := (-5, 0), B := (3, -3), C := (0, 2) }

/-- The equation of the line containing side BC -/
def line_BC : LineEquation := { a := 5, b := 3, c := -6 }

/-- The equation of the line containing the altitude from A to side BC -/
def altitude_A : LineEquation := { a := 5, b := 2, c := 25 }

theorem triangle_lines_correct (t : Triangle) (bc : LineEquation) (alt : LineEquation) :
  t = triangle → bc = line_BC → alt = altitude_A := by sorry

end NUMINAMATH_CALUDE_triangle_lines_correct_l333_33362


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_1_a_range_for_interval_containment_l333_33385

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x + 4
def g (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem for part (1)
theorem solution_set_for_a_eq_1 :
  let a := 1
  ∃ (S : Set ℝ), S = {x | f a x ≥ g x} ∧ S = Set.Icc (-1) ((Real.sqrt 17 - 1) / 2) :=
sorry

-- Theorem for part (2)
theorem a_range_for_interval_containment :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (-1) 1, f a x ≥ g x) → a ∈ Set.Icc (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_1_a_range_for_interval_containment_l333_33385


namespace NUMINAMATH_CALUDE_ones_digit_of_9_pow_47_l333_33386

-- Define a function to get the ones digit of an integer
def ones_digit (n : ℕ) : ℕ := n % 10

-- Theorem stating that the ones digit of 9^47 is 9
theorem ones_digit_of_9_pow_47 : ones_digit (9^47) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_9_pow_47_l333_33386


namespace NUMINAMATH_CALUDE_binary_matrix_sum_theorem_l333_33333

/-- A 5x5 matrix with entries 0 or 1 -/
def BinaryMatrix := Matrix (Fin 5) (Fin 5) Bool

/-- Get the 24 sequences from a BinaryMatrix as specified in the problem -/
def getSequences (X : BinaryMatrix) : Finset (List Bool) := sorry

/-- The sum of all entries in a BinaryMatrix -/
def matrixSum (X : BinaryMatrix) : ℕ := sorry

/-- Main theorem -/
theorem binary_matrix_sum_theorem (X : BinaryMatrix) :
  (getSequences X).card = 24 → matrixSum X = 12 ∨ matrixSum X = 13 := by sorry

end NUMINAMATH_CALUDE_binary_matrix_sum_theorem_l333_33333


namespace NUMINAMATH_CALUDE_exists_valid_selection_l333_33377

/-- A vertex of a polygon with two distinct numbers -/
structure Vertex :=
  (num1 : ℕ)
  (num2 : ℕ)
  (distinct : num1 ≠ num2)

/-- A convex 100-gon with two numbers at each vertex -/
def Polygon := Fin 100 → Vertex

/-- A selection of numbers from the vertices -/
def Selection := Fin 100 → ℕ

/-- Predicate to check if a selection is valid (no adjacent vertices have the same number) -/
def ValidSelection (p : Polygon) (s : Selection) : Prop :=
  ∀ i : Fin 100, s i ≠ s (i + 1)

/-- Theorem stating that for any 100-gon with two distinct numbers at each vertex,
    there exists a valid selection of numbers -/
theorem exists_valid_selection (p : Polygon) :
  ∃ s : Selection, ValidSelection p s ∧ ∀ i : Fin 100, s i = (p i).num1 ∨ s i = (p i).num2 :=
sorry

end NUMINAMATH_CALUDE_exists_valid_selection_l333_33377


namespace NUMINAMATH_CALUDE_b_win_probability_l333_33302

/-- Represents the outcome of a single die roll -/
def DieRoll := Fin 6

/-- Represents the state of the game after each roll -/
structure GameState where
  rolls : List DieRoll
  turn : Bool  -- true for A's turn, false for B's turn

/-- Checks if a number is a multiple of 2 -/
def isMultipleOf2 (n : ℕ) : Bool := n % 2 = 0

/-- Checks if a number is a multiple of 3 -/
def isMultipleOf3 (n : ℕ) : Bool := n % 3 = 0

/-- Sums the last n rolls in the game state -/
def sumLastNRolls (state : GameState) (n : ℕ) : ℕ :=
  (state.rolls.take n).map (fun x => x.val + 1) |>.sum

/-- Determines if the game has ended and who the winner is -/
def gameResult (state : GameState) : Option Bool :=
  if state.rolls.length < 2 then
    none
  else if state.rolls.length < 3 then
    if isMultipleOf3 (sumLastNRolls state 2) then some false else none
  else
    let lastThreeSum := sumLastNRolls state 3
    let lastTwoSum := sumLastNRolls state 2
    if isMultipleOf2 lastThreeSum && !isMultipleOf3 lastTwoSum then
      some true  -- A wins
    else if isMultipleOf3 lastTwoSum && !isMultipleOf2 lastThreeSum then
      some false  -- B wins
    else
      none  -- Game continues

/-- The probability that player B wins the game -/
def probabilityBWins : ℚ := 5/9

theorem b_win_probability :
  probabilityBWins = 5/9 := by sorry

end NUMINAMATH_CALUDE_b_win_probability_l333_33302


namespace NUMINAMATH_CALUDE_polygon_120_degree_angle_l333_33323

/-- A triangular grid of equilateral triangles with unit sides -/
structure TriangularGrid where
  -- Add necessary fields here

/-- A non-self-intersecting polygon on a triangular grid -/
structure Polygon (grid : TriangularGrid) where
  vertices : List (ℕ × ℕ)
  is_non_self_intersecting : Bool
  perimeter : ℕ

/-- Checks if a polygon has a 120-degree angle (internal or external) -/
def has_120_degree_angle (grid : TriangularGrid) (p : Polygon grid) : Prop :=
  sorry

theorem polygon_120_degree_angle 
  (grid : TriangularGrid) 
  (p : Polygon grid) 
  (h1 : p.is_non_self_intersecting = true) 
  (h2 : p.perimeter = 1399) : 
  has_120_degree_angle grid p := by
  sorry

end NUMINAMATH_CALUDE_polygon_120_degree_angle_l333_33323


namespace NUMINAMATH_CALUDE_uncle_jerry_tomatoes_l333_33356

theorem uncle_jerry_tomatoes (day1 day2 total : ℕ) 
  (h1 : day2 = day1 + 50)
  (h2 : day1 + day2 = total)
  (h3 : total = 290) : 
  day1 = 120 := by
sorry

end NUMINAMATH_CALUDE_uncle_jerry_tomatoes_l333_33356


namespace NUMINAMATH_CALUDE_function_convexity_concavity_l333_33301

-- Function convexity/concavity theorem
theorem function_convexity_concavity :
  -- x² is convex everywhere
  (∀ (x₁ x₂ q₁ q₂ : ℝ), q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ * x₁^2 + q₂ * x₂^2 - (q₁ * x₁ + q₂ * x₂)^2 ≥ 0) ∧
  -- √x is concave everywhere
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ > 0 → x₂ > 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ * Real.sqrt x₁ + q₂ * Real.sqrt x₂ - Real.sqrt (q₁ * x₁ + q₂ * x₂) ≤ 0) ∧
  -- x³ is convex for x > 0 and concave for x < 0
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ > 0 → x₂ > 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ * x₁^3 + q₂ * x₂^3 - (q₁ * x₁ + q₂ * x₂)^3 ≥ 0) ∧
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ < 0 → x₂ < 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ * x₁^3 + q₂ * x₂^3 - (q₁ * x₁ + q₂ * x₂)^3 ≤ 0) ∧
  -- 1/x is convex for x > 0 and concave for x < 0
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ > 0 → x₂ > 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ / x₁ + q₂ / x₂ - 1 / (q₁ * x₁ + q₂ * x₂) ≥ 0) ∧
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ < 0 → x₂ < 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ / x₁ + q₂ / x₂ - 1 / (q₁ * x₁ + q₂ * x₂) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_convexity_concavity_l333_33301


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l333_33381

theorem arithmetic_calculation : (21 / (6 + 1 - 4)) * 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l333_33381


namespace NUMINAMATH_CALUDE_no_common_multiple_in_factors_of_600_l333_33368

theorem no_common_multiple_in_factors_of_600 : 
  ∀ n : ℕ, n ∣ 600 → ¬(30 ∣ n ∧ 42 ∣ n ∧ 56 ∣ n) :=
by
  sorry

end NUMINAMATH_CALUDE_no_common_multiple_in_factors_of_600_l333_33368


namespace NUMINAMATH_CALUDE_pairball_playing_time_l333_33360

theorem pairball_playing_time (num_children : ℕ) (total_time : ℕ) : 
  num_children = 6 →
  total_time = 90 →
  ∃ (time_per_child : ℕ), 
    time_per_child * num_children = 2 * total_time ∧
    time_per_child = 30 :=
by sorry

end NUMINAMATH_CALUDE_pairball_playing_time_l333_33360


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l333_33399

theorem min_value_trig_expression :
  ∃ (x : ℝ), ∀ (y : ℝ),
    (Real.sin y)^8 + (Real.cos y)^8 + 1
    ≤ ((Real.sin x)^8 + (Real.cos x)^8 + 1) / ((Real.sin x)^6 + (Real.cos x)^6 + 1)
    ∧ ((Real.sin x)^8 + (Real.cos x)^8 + 1) / ((Real.sin x)^6 + (Real.cos x)^6 + 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l333_33399


namespace NUMINAMATH_CALUDE_shopping_total_proof_l333_33353

def toy_count : ℕ := 5
def toy_price : ℚ := 3
def toy_discount : ℚ := 0.20

def book_count : ℕ := 3
def book_price : ℚ := 8
def book_discount : ℚ := 0.15

def shirt_count : ℕ := 2
def shirt_price : ℚ := 12
def shirt_discount : ℚ := 0.25

def total_paid : ℚ := 50.40

theorem shopping_total_proof :
  (toy_count : ℚ) * toy_price * (1 - toy_discount) +
  (book_count : ℚ) * book_price * (1 - book_discount) +
  (shirt_count : ℚ) * shirt_price * (1 - shirt_discount) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_shopping_total_proof_l333_33353


namespace NUMINAMATH_CALUDE_f_properties_l333_33395

def f (x : ℝ) : ℝ := -x - x^3

theorem f_properties (x₁ x₂ : ℝ) (h : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧ (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l333_33395


namespace NUMINAMATH_CALUDE_triangle_tangent_sum_inequality_l333_33335

/-- For any acute-angled triangle ABC with perimeter p and inradius r,
    the sum of the tangents of its angles is greater than or equal to
    the ratio of its perimeter to twice its inradius. -/
theorem triangle_tangent_sum_inequality (A B C : ℝ) (p r : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute angles
  A + B + C = Real.pi ∧    -- Sum of angles in a triangle
  p > 0 ∧ r > 0 →          -- Positive perimeter and inradius
  Real.tan A + Real.tan B + Real.tan C ≥ p / (2 * r) := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_sum_inequality_l333_33335


namespace NUMINAMATH_CALUDE_parabola_tangent_and_circle_l333_33382

/-- Given a parabola y = x^2 and point P (1, -1), this theorem proves:
    1. The x-coordinates of the tangent points M and N, where x₁ < x₂, are x₁ = 1 - √2 and x₂ = 1 + √2.
    2. The area of a circle with center P tangent to line MN is 16π/5. -/
theorem parabola_tangent_and_circle (x₁ x₂ : ℝ) :
  let P : ℝ × ℝ := (1, -1)
  let T₀ : ℝ → ℝ := λ x => x^2
  let is_tangent (x : ℝ) := T₀ x = (x - 1)^2 - 1 ∧ 2*x = (x^2 + 1) / (x - 1)
  x₁ < x₂ ∧ is_tangent x₁ ∧ is_tangent x₂ →
  (x₁ = 1 - Real.sqrt 2 ∧ x₂ = 1 + Real.sqrt 2) ∧
  (π * ((2 * 1 + 1 + 1) / Real.sqrt (4 + 1))^2 = 16 * π / 5) := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_and_circle_l333_33382


namespace NUMINAMATH_CALUDE_investment_growth_l333_33380

/-- The monthly interest rate for an investment that grows from $300 to $363 in 2 months -/
def monthly_interest_rate : ℝ :=
  0.1

theorem investment_growth (initial_investment : ℝ) (final_amount : ℝ) (months : ℕ) :
  initial_investment = 300 →
  final_amount = 363 →
  months = 2 →
  final_amount = initial_investment * (1 + monthly_interest_rate) ^ months :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l333_33380


namespace NUMINAMATH_CALUDE_boxes_sold_l333_33337

theorem boxes_sold (initial boxes_left : ℕ) (h : initial ≥ boxes_left) :
  initial - boxes_left = initial - boxes_left :=
by sorry

end NUMINAMATH_CALUDE_boxes_sold_l333_33337


namespace NUMINAMATH_CALUDE_only_cone_cannot_have_rectangular_cross_section_l333_33324

-- Define the types of geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | PentagonalPrism
  | Cube

-- Define a function that determines if a solid can have a rectangular cross-section
def canHaveRectangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => true
  | GeometricSolid.Cone => false
  | GeometricSolid.PentagonalPrism => true
  | GeometricSolid.Cube => true

-- Theorem statement
theorem only_cone_cannot_have_rectangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveRectangularCrossSection solid) ↔ solid = GeometricSolid.Cone :=
by sorry

end NUMINAMATH_CALUDE_only_cone_cannot_have_rectangular_cross_section_l333_33324


namespace NUMINAMATH_CALUDE_xy_inequality_l333_33398

theorem xy_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = x + y + 3) :
  (x + y ≥ 6) ∧ (x * y ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_xy_inequality_l333_33398


namespace NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l333_33366

theorem power_of_product_equals_product_of_powers (b : ℝ) : (2 * b^2)^3 = 8 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l333_33366


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_twenty_eight_l333_33343

theorem reciprocal_of_negative_twenty_eight :
  (1 : ℚ) / (-28 : ℚ) = -1 / 28 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_twenty_eight_l333_33343


namespace NUMINAMATH_CALUDE_no_real_roots_l333_33394

theorem no_real_roots : ∀ x : ℝ, x^2 + 3*x + 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l333_33394


namespace NUMINAMATH_CALUDE_tea_brewing_time_proof_l333_33332

/-- The time needed to wash the kettle and fill it with cold water -/
def wash_kettle_time : ℕ := 2

/-- The time needed to wash the teapot and cups -/
def wash_teapot_cups_time : ℕ := 2

/-- The time needed to get tea leaves -/
def get_tea_leaves_time : ℕ := 1

/-- The time needed to boil water -/
def boil_water_time : ℕ := 15

/-- The time needed to brew the tea -/
def brew_tea_time : ℕ := 1

/-- The shortest operation time for brewing a pot of tea -/
def shortest_operation_time : ℕ := 18

theorem tea_brewing_time_proof :
  shortest_operation_time = max wash_kettle_time (max boil_water_time brew_tea_time) :=
by sorry

end NUMINAMATH_CALUDE_tea_brewing_time_proof_l333_33332


namespace NUMINAMATH_CALUDE_non_negative_integer_solutions_of_inequality_l333_33306

theorem non_negative_integer_solutions_of_inequality : 
  {x : ℕ | -2 * (x : ℤ) > -4} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_non_negative_integer_solutions_of_inequality_l333_33306


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l333_33372

theorem complex_fraction_calculation : 
  ∃ ε > 0, |((9/20 : ℚ) - 11/30 + 13/42 - 15/56 + 17/72) * 120 - (1/3) / (1/4) - 42| < ε :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l333_33372


namespace NUMINAMATH_CALUDE_arithmetic_sequence_implication_l333_33310

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_implication
  (a b : ℕ → ℝ)
  (h : ∀ n : ℕ, b n = a n + a (n + 1)) :
  is_arithmetic_sequence a → is_arithmetic_sequence b ∧
  ¬(is_arithmetic_sequence b → is_arithmetic_sequence a) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_implication_l333_33310


namespace NUMINAMATH_CALUDE_problem_statement_l333_33384

theorem problem_statement (a b : ℝ) (h : a - 3*b = 3) : 
  (a + 2*b) - (2*a - b) = -3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l333_33384


namespace NUMINAMATH_CALUDE_cone_volume_l333_33342

/-- The volume of a cone with base radius 1 and slant height 2 is (√3/3)π -/
theorem cone_volume (r h l : ℝ) : 
  r = 1 → l = 2 → h^2 + r^2 = l^2 → (1/3 * π * r^2 * h) = (Real.sqrt 3 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l333_33342


namespace NUMINAMATH_CALUDE_three_squares_decomposition_l333_33348

theorem three_squares_decomposition (n : ℤ) (h : n > 5) :
  3 * (n - 1)^2 + 32 = (n - 5)^2 + (n - 1)^2 + (n + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_three_squares_decomposition_l333_33348


namespace NUMINAMATH_CALUDE_circle_equation_l333_33355

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def is_in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def is_tangent_to_line (C : Circle) (a b c : ℝ) : Prop :=
  let (x, y) := C.center
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2) = C.radius

def is_tangent_to_x_axis (C : Circle) : Prop :=
  C.center.2 = C.radius

-- State the theorem
theorem circle_equation (C : Circle) :
  C.radius = 1 →
  is_in_first_quadrant C.center →
  is_tangent_to_line C 4 (-3) 0 →
  is_tangent_to_x_axis C →
  ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 1 ↔ (x, y) ∈ {p | (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l333_33355


namespace NUMINAMATH_CALUDE_debby_water_bottles_l333_33370

def water_bottle_problem (initial_bottles : ℕ) (bottles_per_day : ℕ) (remaining_bottles : ℕ) : Prop :=
  let days : ℕ := (initial_bottles - remaining_bottles) / bottles_per_day
  days = 1

theorem debby_water_bottles :
  water_bottle_problem 301 144 157 :=
sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l333_33370


namespace NUMINAMATH_CALUDE_edward_board_game_cost_l333_33319

def board_game_cost (total_cost : ℕ) (num_figures : ℕ) (figure_cost : ℕ) : ℕ :=
  total_cost - (num_figures * figure_cost)

theorem edward_board_game_cost :
  board_game_cost 30 4 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_edward_board_game_cost_l333_33319


namespace NUMINAMATH_CALUDE_division_theorem_l333_33338

theorem division_theorem (M q : ℤ) (h : M = 54 * q + 37) :
  ∃ (k : ℤ), M = 18 * k + 1 ∧ k = 3 * q + 2 := by
  sorry

end NUMINAMATH_CALUDE_division_theorem_l333_33338


namespace NUMINAMATH_CALUDE_negation_existence_equivalence_l333_33347

theorem negation_existence_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔ (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_existence_equivalence_l333_33347


namespace NUMINAMATH_CALUDE_mashed_potatoes_vs_bacon_difference_l333_33320

/-- The number of students who suggested adding bacon to the menu. -/
def bacon_students : ℕ := 269

/-- The number of students who suggested adding mashed potatoes to the menu. -/
def mashed_potatoes_students : ℕ := 330

/-- The number of students who suggested adding tomatoes to the menu. -/
def tomato_students : ℕ := 76

/-- The theorem states that the difference between the number of students who suggested
    mashed potatoes and the number of students who suggested bacon is 61. -/
theorem mashed_potatoes_vs_bacon_difference :
  mashed_potatoes_students - bacon_students = 61 := by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_vs_bacon_difference_l333_33320


namespace NUMINAMATH_CALUDE_rain_stop_time_l333_33387

def rain_duration (start_time : ℕ) (day1_duration : ℕ) : ℕ → ℕ
  | 1 => day1_duration
  | 2 => day1_duration + 2
  | 3 => 2 * (day1_duration + 2)
  | _ => 0

theorem rain_stop_time (start_time : ℕ) (day1_duration : ℕ) :
  start_time = 7 ∧ 
  (rain_duration start_time day1_duration 1 + 
   rain_duration start_time day1_duration 2 + 
   rain_duration start_time day1_duration 3 = 46) →
  start_time + day1_duration = 17 := by
  sorry

end NUMINAMATH_CALUDE_rain_stop_time_l333_33387


namespace NUMINAMATH_CALUDE_polynomial_expansion_l333_33334

theorem polynomial_expansion (t : ℝ) : 
  (3 * t^3 - 2 * t^2 + 4 * t - 1) * (-2 * t^2 + 3 * t + 6) = 
  -6 * t^5 + 5 * t^4 + 4 * t^3 + 22 * t^2 + 27 * t - 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l333_33334


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l333_33317

theorem sqrt_D_irrational (x : ℤ) : 
  let a : ℤ := x
  let b : ℤ := x + 2
  let c : ℤ := a * b
  let d : ℤ := b + c
  let D : ℤ := a^2 + b^2 + c^2 + d^2
  Irrational (Real.sqrt D) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l333_33317


namespace NUMINAMATH_CALUDE_ratio_of_x_to_y_l333_33350

theorem ratio_of_x_to_y (x y : ℝ) (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) :
  (1/5 * x) / (1/6 * y) = 18/25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_x_to_y_l333_33350


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l333_33336

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v1 v2 : Fin 2 → ℝ) : Prop :=
  v1 0 * v2 1 = v1 1 * v2 0

/-- The problem statement -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  are_parallel (λ i => if i = 0 then 1 else 2) (λ i => if i = 0 then 2*x else -3) →
  x = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l333_33336


namespace NUMINAMATH_CALUDE_stamps_ratio_l333_33396

def stamps_problem (bert ernie peggy : ℕ) : Prop :=
  bert = 4 * ernie ∧
  ∃ k : ℕ, ernie = k * peggy ∧
  peggy = 75 ∧
  bert = peggy + 825

theorem stamps_ratio (bert ernie peggy : ℕ) 
  (h : stamps_problem bert ernie peggy) : ernie / peggy = 3 := by
  sorry

end NUMINAMATH_CALUDE_stamps_ratio_l333_33396


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l333_33340

theorem circle_tangent_to_line (m : ℝ) (hm : m ≥ 0) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = m}
  let line := {(x, y) : ℝ × ℝ | x + y = Real.sqrt (2 * m)}
  ∃ (p : ℝ × ℝ), p ∈ circle ∧ p ∈ line ∧
    ∀ (q : ℝ × ℝ), q ∈ circle → q ∈ line → q = p :=
by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l333_33340


namespace NUMINAMATH_CALUDE_hexagon_semicircles_area_l333_33388

/-- The area of the region inside a regular hexagon with side length 4, 
    but outside eight semicircles (where each semicircle's diameter 
    coincides with each side of the hexagon) -/
theorem hexagon_semicircles_area : 
  let s : ℝ := 4 -- side length of the hexagon
  let r : ℝ := s / 2 -- radius of each semicircle
  let hexagon_area : ℝ := 3 * Real.sqrt 3 / 2 * s^2
  let semicircle_area : ℝ := 8 * (Real.pi * r^2 / 2)
  hexagon_area - semicircle_area = 24 * Real.sqrt 3 - 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_hexagon_semicircles_area_l333_33388


namespace NUMINAMATH_CALUDE_shoe_price_is_50_l333_33318

/-- Represents the original price of a pair of shoes -/
def original_shoe_price : ℝ := sorry

/-- Represents the discount rate for shoes -/
def shoe_discount : ℝ := 0.4

/-- Represents the discount rate for dresses -/
def dress_discount : ℝ := 0.2

/-- Represents the number of pairs of shoes bought -/
def num_shoes : ℕ := 2

/-- Represents the original price of the dress -/
def dress_price : ℝ := 100

/-- Represents the total amount spent -/
def total_spent : ℝ := 140

/-- Theorem stating that the original price of a pair of shoes is $50 -/
theorem shoe_price_is_50 : 
  (num_shoes : ℝ) * original_shoe_price * (1 - shoe_discount) + 
  dress_price * (1 - dress_discount) = total_spent → 
  original_shoe_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_shoe_price_is_50_l333_33318


namespace NUMINAMATH_CALUDE_apple_tree_problem_l333_33371

/-- The number of apples initially on the tree -/
def initial_apples : ℕ := 11

/-- The number of apples picked from the tree -/
def apples_picked : ℕ := 7

/-- The number of new apples that grew on the tree -/
def new_apples : ℕ := 2

/-- The number of apples currently on the tree -/
def current_apples : ℕ := 6

theorem apple_tree_problem :
  initial_apples - apples_picked + new_apples = current_apples :=
by sorry

end NUMINAMATH_CALUDE_apple_tree_problem_l333_33371


namespace NUMINAMATH_CALUDE_selection_theorem_l333_33376

/-- The number of ways to select 3 people from 4 male and 3 female students, ensuring both genders are represented. -/
def selection_ways (male_count female_count : ℕ) (total_selected : ℕ) : ℕ :=
  Nat.choose (male_count + female_count) total_selected -
  Nat.choose male_count total_selected -
  Nat.choose female_count total_selected

/-- Theorem stating that the number of ways to select 3 people from 4 male and 3 female students, ensuring both genders are represented, is 30. -/
theorem selection_theorem :
  selection_ways 4 3 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l333_33376


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l333_33311

/-- Given vectors a and b in ℝ², prove that the magnitude of their difference is 5 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  a = (2, 1) →
  b = (-2, 4) →
  ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l333_33311


namespace NUMINAMATH_CALUDE_consecutive_draws_count_l333_33354

/-- The number of ways to draw 4 consecutively numbered balls from a set of 20 balls. -/
def consecutiveDraws : ℕ := 17

/-- The total number of balls in the bin. -/
def totalBalls : ℕ := 20

/-- The number of balls to be drawn. -/
def ballsDrawn : ℕ := 4

theorem consecutive_draws_count :
  consecutiveDraws = totalBalls - ballsDrawn + 1 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_draws_count_l333_33354


namespace NUMINAMATH_CALUDE_shirt_cost_theorem_l333_33303

theorem shirt_cost_theorem :
  let total_shirts : ℕ := 5
  let cheap_shirts : ℕ := 3
  let expensive_shirts : ℕ := total_shirts - cheap_shirts
  let cheap_price : ℕ := 15
  let expensive_price : ℕ := 20
  
  (cheap_shirts * cheap_price + expensive_shirts * expensive_price : ℕ) = 85
  := by sorry

end NUMINAMATH_CALUDE_shirt_cost_theorem_l333_33303


namespace NUMINAMATH_CALUDE_solution_approximation_l333_33339

/-- The solution to the equation (0.625 * 0.0729 * 28.9) / (x * 0.025 * 8.1) = 382.5 is approximately 0.01689 -/
theorem solution_approximation : ∃ x : ℝ, 
  (0.625 * 0.0729 * 28.9) / (x * 0.025 * 8.1) = 382.5 ∧ 
  abs (x - 0.01689) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_solution_approximation_l333_33339


namespace NUMINAMATH_CALUDE_circle_equation_through_point_l333_33397

/-- The equation of a circle with center (1, 0) passing through (1, -1) -/
theorem circle_equation_through_point :
  let center : ℝ × ℝ := (1, 0)
  let point : ℝ × ℝ := (1, -1)
  let equation (x y : ℝ) := (x - center.1)^2 + (y - center.2)^2 = (point.1 - center.1)^2 + (point.2 - center.2)^2
  ∀ x y : ℝ, equation x y ↔ (x - 1)^2 + y^2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_equation_through_point_l333_33397


namespace NUMINAMATH_CALUDE_fifteen_switches_connections_l333_33374

/-- The number of unique connections in a network of switches -/
def uniqueConnections (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a network of 15 switches, where each switch connects to 
    exactly 4 other switches, the total number of unique connections is 30. -/
theorem fifteen_switches_connections : 
  uniqueConnections 15 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_switches_connections_l333_33374


namespace NUMINAMATH_CALUDE_largest_n_less_than_2023_l333_33365

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 2 * n

-- Define the sequence b_n
def b (n : ℕ) : ℕ := 2^n

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℕ := n^2 + n

-- Define T_n
def T (n : ℕ) : ℕ := (n - 1) * 2^(n + 2) + 4

theorem largest_n_less_than_2023 :
  (∀ n : ℕ, S n = n^2 + n) →
  (∀ n : ℕ, b n = 2^n) →
  (∀ n : ℕ, T n = (n - 1) * 2^(n + 2) + 4) →
  (∃ m : ℕ, m = 6 ∧ T m < 2023 ∧ ∀ k > m, T k ≥ 2023) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_less_than_2023_l333_33365


namespace NUMINAMATH_CALUDE_fraction_sum_l333_33305

theorem fraction_sum : (1 : ℚ) / 4 + 2 / 9 + 3 / 6 = 35 / 36 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_l333_33305


namespace NUMINAMATH_CALUDE_inequality_proof_l333_33369

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  2 * x + 1 / (x^2 - 2*x*y + y^2) ≥ 2 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l333_33369


namespace NUMINAMATH_CALUDE_sandwich_problem_l333_33307

theorem sandwich_problem (total : ℕ) (bologna : ℕ) (x : ℕ) :
  total = 80 →
  bologna = 35 →
  bologna = 7 * (total / (1 + 7 + x)) →
  x * (total / (1 + 7 + x)) = 40 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_problem_l333_33307


namespace NUMINAMATH_CALUDE_direct_square_variation_problem_l333_33304

/-- A function representing direct variation with the square of x -/
def direct_square_variation (k : ℝ) (x : ℝ) : ℝ := k * x^2

theorem direct_square_variation_problem (y : ℝ → ℝ) :
  (∃ k : ℝ, ∀ x, y x = direct_square_variation k x) →
  y 3 = 18 →
  y 6 = 72 := by sorry

end NUMINAMATH_CALUDE_direct_square_variation_problem_l333_33304


namespace NUMINAMATH_CALUDE_correct_subtraction_l333_33357

theorem correct_subtraction (x : ℤ) (h : x - 21 = 52) : x - 40 = 33 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l333_33357


namespace NUMINAMATH_CALUDE_divisible_by_900_l333_33330

theorem divisible_by_900 (n : ℕ) : ∃ k : ℤ, 6^(2*(n+1)) - 2^(n+3) * 3^(n+2) + 36 = 900 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_900_l333_33330


namespace NUMINAMATH_CALUDE_six_people_circular_table_l333_33375

/-- The number of distinct circular permutations of n elements -/
def circularPermutations (n : ℕ) : ℕ := (n - 1).factorial

/-- Two seating arrangements are considered the same if one is a rotation of the other -/
axiom rotation_equivalence : ∀ n : ℕ, n > 0 → circularPermutations n = (n - 1).factorial

theorem six_people_circular_table : circularPermutations 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_six_people_circular_table_l333_33375


namespace NUMINAMATH_CALUDE_unit_digit_of_seven_to_fourteen_l333_33389

theorem unit_digit_of_seven_to_fourteen (n : ℕ) : n = 7^14 → n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_seven_to_fourteen_l333_33389


namespace NUMINAMATH_CALUDE_circle_sum_of_center_and_radius_l333_33373

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 16*y + 81 = -y^2 + 14*x

-- Define the center and radius of the circle
def circle_center_radius (a b r : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_sum_of_center_and_radius :
  ∃ a b r, circle_center_radius a b r ∧ a + b + r = 15 + 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_circle_sum_of_center_and_radius_l333_33373


namespace NUMINAMATH_CALUDE_sum_squares_inequality_l333_33393

theorem sum_squares_inequality (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  a + b + c ≥ a^2*b^2 + b^2*c^2 + c^2*a^2 := by
sorry

end NUMINAMATH_CALUDE_sum_squares_inequality_l333_33393


namespace NUMINAMATH_CALUDE_fifteenth_term_geometric_sequence_l333_33322

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

theorem fifteenth_term_geometric_sequence :
  geometric_sequence 12 (1/3) 15 = 4/1594323 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_geometric_sequence_l333_33322


namespace NUMINAMATH_CALUDE_intersection_P_Q_l333_33316

def P : Set ℝ := {x | -x^2 + 3*x + 4 < 0}
def Q : Set ℝ := {x | 2*x - 5 > 0}

theorem intersection_P_Q : P ∩ Q = {x | x > 4} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l333_33316


namespace NUMINAMATH_CALUDE_equation_solution_l333_33358

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x^2 + 6*x + 6*x * Real.sqrt (x + 5) - 52
  ∃ (x₁ x₂ : ℝ), x₁ = (9 - Real.sqrt 5) / 2 ∧ x₂ = (9 + Real.sqrt 5) / 2 ∧
  f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l333_33358


namespace NUMINAMATH_CALUDE_helicopter_rental_cost_l333_33379

/-- Calculates the total cost of helicopter rental given the specified conditions -/
theorem helicopter_rental_cost : 
  let hours_per_day : ℕ := 2
  let num_days : ℕ := 3
  let rate_day1 : ℚ := 85
  let rate_day2 : ℚ := 75
  let rate_day3 : ℚ := 65
  let discount_rate : ℚ := 0.05
  let cost_before_discount : ℚ := hours_per_day * (rate_day1 + rate_day2 + rate_day3)
  let discount : ℚ := discount_rate * cost_before_discount
  let total_cost : ℚ := cost_before_discount - discount
  total_cost = 427.5 := by sorry

end NUMINAMATH_CALUDE_helicopter_rental_cost_l333_33379


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_with_diff_217_l333_33351

theorem smallest_sum_of_squares_with_diff_217 :
  ∃ (x y : ℕ), 
    x^2 - y^2 = 217 ∧
    ∀ (a b : ℕ), a^2 - b^2 = 217 → x^2 + y^2 ≤ a^2 + b^2 ∧
    x^2 + y^2 = 505 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_with_diff_217_l333_33351


namespace NUMINAMATH_CALUDE_minimum_guests_proof_l333_33391

-- Define the total food consumed
def total_food : ℝ := 319

-- Define the maximum individual consumption limits
def max_meat : ℝ := 1.5
def max_vegetables : ℝ := 0.3
def max_dessert : ℝ := 0.2

-- Define the consumption ratio
def meat_ratio : ℝ := 3
def vegetables_ratio : ℝ := 1
def dessert_ratio : ℝ := 1

-- Define the minimum number of guests
def min_guests : ℕ := 160

-- Theorem statement
theorem minimum_guests_proof :
  ∃ (guests : ℕ), guests ≥ min_guests ∧
  (guests : ℝ) * (max_meat + max_vegetables + max_dessert) ≥ total_food ∧
  ∀ (g : ℕ), g < guests →
    (g : ℝ) * (max_meat + max_vegetables + max_dessert) < total_food :=
sorry

end NUMINAMATH_CALUDE_minimum_guests_proof_l333_33391


namespace NUMINAMATH_CALUDE_point_B_in_third_quadrant_l333_33309

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the third quadrant -/
def is_in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If A(m, -n) is in the second quadrant, then B(-mn, m) is in the third quadrant -/
theorem point_B_in_third_quadrant 
  (m n : ℝ) 
  (h : is_in_second_quadrant ⟨m, -n⟩) : 
  is_in_third_quadrant ⟨-m*n, m⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_B_in_third_quadrant_l333_33309


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l333_33341

theorem fractional_equation_solution :
  ∃ x : ℝ, (x + 1) / x = 2 / 3 ∧ x = -3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l333_33341


namespace NUMINAMATH_CALUDE_derivative_of_f_l333_33367

noncomputable def f (x : ℝ) := Real.cos (x^2 + x)

theorem derivative_of_f (x : ℝ) :
  deriv f x = -(2 * x + 1) * Real.sin (x^2 + x) := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l333_33367


namespace NUMINAMATH_CALUDE_perpendicular_line_tangent_cubic_l333_33359

/-- Given a line ax - by - 2 = 0 perpendicular to the tangent of y = x^3 at (1,1), prove a/b = -1/3 -/
theorem perpendicular_line_tangent_cubic (a b : ℝ) : 
  (∀ x y : ℝ, a * x - b * y - 2 = 0 → 
    (x - 1) * (3 * (1 : ℝ)^2) + (y - 1) = 0) → 
  a / b = -1/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_tangent_cubic_l333_33359


namespace NUMINAMATH_CALUDE_product_of_x_values_l333_33329

theorem product_of_x_values (x : ℝ) : 
  (|18 / x - 6| = 3) → (∃ y : ℝ, y ≠ x ∧ |18 / y - 6| = 3 ∧ x * y = 12) :=
by sorry

end NUMINAMATH_CALUDE_product_of_x_values_l333_33329


namespace NUMINAMATH_CALUDE_perimeter_is_twentyone_l333_33315

/-- A figure with 3 vertices where the distance between any 2 vertices is 7 -/
structure ThreeVertexFigure where
  vertices : Fin 3 → ℝ × ℝ
  distance_eq_seven : ∀ (i j : Fin 3), i ≠ j → Real.sqrt ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 7

/-- The perimeter of a ThreeVertexFigure is 21 -/
theorem perimeter_is_twentyone (f : ThreeVertexFigure) : 
  (Real.sqrt ((f.vertices 0).1 - (f.vertices 1).1)^2 + ((f.vertices 0).2 - (f.vertices 1).2)^2) +
  (Real.sqrt ((f.vertices 1).1 - (f.vertices 2).1)^2 + ((f.vertices 1).2 - (f.vertices 2).2)^2) +
  (Real.sqrt ((f.vertices 2).1 - (f.vertices 0).1)^2 + ((f.vertices 2).2 - (f.vertices 0).2)^2) = 21 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_is_twentyone_l333_33315


namespace NUMINAMATH_CALUDE_dot_product_equals_four_l333_33327

def a : ℝ × ℝ := (1, 2)

theorem dot_product_equals_four (b : ℝ × ℝ) 
  (h : (2 • a) - b = (4, 1)) : a • b = 4 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_equals_four_l333_33327


namespace NUMINAMATH_CALUDE_sum_of_symmetric_roots_l333_33361

theorem sum_of_symmetric_roots (f : ℝ → ℝ) 
  (h_sym : ∀ x : ℝ, f (-x) = f x) 
  (h_roots : ∃! (s : Finset ℝ), s.card = 2009 ∧ ∀ x ∈ s, f x = 0) : 
  ∃ (s : Finset ℝ), s.card = 2009 ∧ (∀ x ∈ s, f x = 0) ∧ (s.sum id = 0) := by
sorry

end NUMINAMATH_CALUDE_sum_of_symmetric_roots_l333_33361


namespace NUMINAMATH_CALUDE_remainder_of_binary_div_8_l333_33363

/-- The binary representation of the number --/
def binary_num : List Bool := [true, true, false, true, true, true, false, false, true, false, true, true]

/-- Convert a binary number to decimal --/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Get the last three digits of a binary number --/
def last_three_digits (binary : List Bool) : List Bool :=
  binary.reverse.take 3

theorem remainder_of_binary_div_8 :
  binary_to_decimal (last_three_digits binary_num) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_div_8_l333_33363


namespace NUMINAMATH_CALUDE_p_squared_plus_18_composite_l333_33352

theorem p_squared_plus_18_composite (p : ℕ) (hp : Prime p) : ¬ Prime (p^2 + 18) := by
  sorry

end NUMINAMATH_CALUDE_p_squared_plus_18_composite_l333_33352


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_reciprocals_ge_four_l333_33383

theorem product_of_sum_and_sum_of_reciprocals_ge_four (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (1/a + 1/b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_reciprocals_ge_four_l333_33383


namespace NUMINAMATH_CALUDE_gcd_of_256_196_560_l333_33349

theorem gcd_of_256_196_560 : Nat.gcd 256 (Nat.gcd 196 560) = 28 := by sorry

end NUMINAMATH_CALUDE_gcd_of_256_196_560_l333_33349


namespace NUMINAMATH_CALUDE_chemical_quantity_problem_l333_33344

theorem chemical_quantity_problem (x : ℤ) : 
  532 * x - 325 * x = 1065430 → x = 5148 := by sorry

end NUMINAMATH_CALUDE_chemical_quantity_problem_l333_33344


namespace NUMINAMATH_CALUDE_tangent_identity_l333_33328

theorem tangent_identity (α β : ℝ) 
  (h1 : Real.tan (α + β) ≠ 0) (h2 : Real.tan (α - β) ≠ 0) :
  (Real.tan α + Real.tan β) / Real.tan (α + β) + 
  (Real.tan α - Real.tan β) / Real.tan (α - β) + 
  2 * (Real.tan α)^2 = 2 / (Real.cos α)^2 := by
sorry

end NUMINAMATH_CALUDE_tangent_identity_l333_33328


namespace NUMINAMATH_CALUDE_min_knights_in_village_l333_33321

theorem min_knights_in_village (total_people : Nat) (total_statements : Nat) (liar_statements : Nat) :
  total_people = 7 →
  total_statements = total_people * (total_people - 1) →
  total_statements = 42 →
  liar_statements = 24 →
  ∃ (knights : Nat), knights ≥ 3 ∧ 
    knights + (total_people - knights) = total_people ∧
    2 * knights * (total_people - knights) = liar_statements :=
by sorry

end NUMINAMATH_CALUDE_min_knights_in_village_l333_33321


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l333_33313

theorem sqrt_equation_solutions :
  {x : ℝ | Real.sqrt (3 * x^2 + 2 * x + 1) = 3} = {4/3, -2} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l333_33313


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_S_l333_33314

/-- The product of non-zero digits of a positive integer -/
def p (n : ℕ+) : ℕ :=
  sorry

/-- The sum of p(n) from 1 to 999 -/
def S : ℕ :=
  (Finset.range 999).sum (λ i => p ⟨i + 1, Nat.succ_pos i⟩)

/-- The largest prime factor of S -/
theorem largest_prime_factor_of_S :
  ∃ (q : ℕ), Nat.Prime q ∧ q ∣ S ∧ ∀ (p : ℕ), Nat.Prime p → p ∣ S → p ≤ q :=
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_S_l333_33314


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l333_33378

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose n 2 = Nat.choose n 5) → n = 7 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l333_33378


namespace NUMINAMATH_CALUDE_dumbbell_system_total_weight_l333_33325

/-- The weight of a dumbbell system with three pairs of dumbbells -/
def dumbbell_system_weight (weight1 weight2 weight3 : ℕ) : ℕ :=
  2 * weight1 + 2 * weight2 + 2 * weight3

/-- Theorem: The weight of the specific dumbbell system is 32 lbs -/
theorem dumbbell_system_total_weight :
  dumbbell_system_weight 3 5 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_dumbbell_system_total_weight_l333_33325


namespace NUMINAMATH_CALUDE_system_solution_l333_33326

theorem system_solution : ∃ (x y z : ℝ), 
  (x^2 - y - z = 8) ∧ 
  (4*x + y^2 + 3*z = -11) ∧ 
  (2*x - 3*y + z^2 = -11) ∧ 
  (x = -3) ∧ (y = 2) ∧ (z = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l333_33326


namespace NUMINAMATH_CALUDE_parabola_distance_property_l333_33364

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -2

-- Define the condition for Q
def Q_condition (Q P : ℝ × ℝ) : Prop :=
  let (qx, qy) := Q
  let (px, py) := P
  (qx - 2, qy) = -4 * (px - 2, py)

-- Theorem statement
theorem parabola_distance_property (Q P : ℝ × ℝ) :
  directrix P.1 →
  parabola Q.1 Q.2 →
  Q_condition Q P →
  Real.sqrt ((Q.1 - 2)^2 + Q.2^2) = 20 := by sorry

end NUMINAMATH_CALUDE_parabola_distance_property_l333_33364


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l333_33390

/-- A quadratic function f(x) = x^2 + px + q -/
def f (p q x : ℝ) : ℝ := x^2 + p*x + q

/-- The theorem stating the conditions for f(p) = f(q) = 0 -/
theorem quadratic_roots_condition (p q : ℝ) :
  (f p q p = 0 ∧ f p q q = 0) ↔ 
  ((p = 0 ∧ q = 0) ∨ (p = -1/2 ∧ q = -1/2) ∨ (p = 1 ∧ q = -2)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l333_33390
