import Mathlib

namespace NUMINAMATH_CALUDE_hexagon_exterior_angles_sum_l1082_108257

/-- A hexagon is a polygon with 6 sides -/
def Hexagon : Type := Unit

/-- The sum of exterior angles of a polygon -/
def sum_exterior_angles (p : Type) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of a hexagon is 360 degrees -/
theorem hexagon_exterior_angles_sum :
  sum_exterior_angles Hexagon = 360 :=
sorry

end NUMINAMATH_CALUDE_hexagon_exterior_angles_sum_l1082_108257


namespace NUMINAMATH_CALUDE_mode_most_relevant_for_restocking_l1082_108275

/-- Represents a shoe size -/
def ShoeSize := ℕ

/-- Represents the inventory of shoes -/
def Inventory := List ShoeSize

/-- A statistical measure for shoe sizes -/
class StatisticalMeasure where
  measure : Inventory → ℝ

/-- Variance of shoe sizes -/
def variance : StatisticalMeasure := sorry

/-- Mode of shoe sizes -/
def mode : StatisticalMeasure := sorry

/-- Median of shoe sizes -/
def median : StatisticalMeasure := sorry

/-- Mean of shoe sizes -/
def mean : StatisticalMeasure := sorry

/-- Relevance of a statistical measure for restocking -/
def relevance (m : StatisticalMeasure) : ℝ := sorry

/-- The shoe store -/
structure ShoeStore where
  inventory : Inventory

/-- Theorem: Mode is the most relevant statistical measure for restocking -/
theorem mode_most_relevant_for_restocking (store : ShoeStore) :
  ∀ m : StatisticalMeasure, m ≠ mode → relevance mode > relevance m :=
sorry

end NUMINAMATH_CALUDE_mode_most_relevant_for_restocking_l1082_108275


namespace NUMINAMATH_CALUDE_second_meeting_time_correct_l1082_108274

/-- Represents a vehicle on a race track -/
structure Vehicle where
  name : String
  lap_time : ℕ  -- lap time in seconds

/-- Calculates the time until two vehicles meet at the starting point for the second time -/
def timeToSecondMeeting (v1 v2 : Vehicle) : ℚ :=
  (Nat.lcm v1.lap_time v2.lap_time : ℚ) / 60

/-- The main theorem to prove -/
theorem second_meeting_time_correct 
  (magic : Vehicle) 
  (bull : Vehicle) 
  (h1 : magic.lap_time = 150)
  (h2 : bull.lap_time = 3600 / 40) :
  timeToSecondMeeting magic bull = 7.5 := by
  sorry

#eval timeToSecondMeeting 
  { name := "The Racing Magic", lap_time := 150 } 
  { name := "The Charging Bull", lap_time := 3600 / 40 }

end NUMINAMATH_CALUDE_second_meeting_time_correct_l1082_108274


namespace NUMINAMATH_CALUDE_sum_max_at_5_l1082_108218

/-- An arithmetic sequence with its first term and sum properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_positive : a 1 > 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  sum : ℕ → ℝ
  sum_def : ∀ n : ℕ, sum n = (n : ℝ) / 2 * (a 1 + a n)
  sum_9_positive : sum 9 > 0
  sum_10_negative : sum 10 < 0

/-- The sum of the arithmetic sequence is maximized at n = 5 -/
theorem sum_max_at_5 (seq : ArithmeticSequence) :
  ∃ (n : ℕ), ∀ (m : ℕ), seq.sum m ≤ seq.sum n ∧ n = 5 :=
sorry

end NUMINAMATH_CALUDE_sum_max_at_5_l1082_108218


namespace NUMINAMATH_CALUDE_cube_sum_equals_275_l1082_108217

theorem cube_sum_equals_275 (a b : ℝ) (h1 : a + b = 11) (h2 : a * b = 32) :
  a^3 + b^3 = 275 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equals_275_l1082_108217


namespace NUMINAMATH_CALUDE_base_difference_in_right_trapezoid_l1082_108290

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- The measure of the largest angle in degrees -/
  largest_angle : ℝ
  /-- The length of the shorter leg -/
  shorter_leg : ℝ
  /-- The length of the longer base -/
  longer_base : ℝ
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- Condition that the largest angle is 135° -/
  largest_angle_eq : largest_angle = 135
  /-- Condition that the shorter leg is 18 -/
  shorter_leg_eq : shorter_leg = 18
  /-- Condition that it's a right trapezoid (one angle is 90°) -/
  is_right : True

/-- Theorem stating the difference between bases in a right trapezoid with specific properties -/
theorem base_difference_in_right_trapezoid (t : RightTrapezoid) : 
  t.longer_base - t.shorter_base = 18 := by
  sorry

end NUMINAMATH_CALUDE_base_difference_in_right_trapezoid_l1082_108290


namespace NUMINAMATH_CALUDE_fifteen_degrees_to_radians_l1082_108241

/-- Conversion of 15 degrees to radians -/
theorem fifteen_degrees_to_radians : 
  (15 : ℝ) * π / 180 = π / 12 := by sorry

end NUMINAMATH_CALUDE_fifteen_degrees_to_radians_l1082_108241


namespace NUMINAMATH_CALUDE_abc_product_value_l1082_108227

theorem abc_product_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a + 1/b = 5) (eq2 : b + 1/c = 2) (eq3 : c + 1/a = 3) :
  a * b * c = 10 + 3 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_value_l1082_108227


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1082_108250

-- Define the line l
def line_l (c : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x + 2*y + c = 0

-- Define the circle C
def circle_C : ℝ → ℝ → Prop :=
  fun x y => x^2 + y^2 + 2*x - 4*y = 0

-- Define the translated line l'
def line_l_prime (c : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x + 2*y + c + 5 = 0

-- Define the tangency condition
def is_tangent (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, l x y ∧ C x y ∧ ∀ x' y', l x' y' ∧ C x' y' → (x = x' ∧ y = y')

theorem line_tangent_to_circle (c : ℝ) :
  is_tangent (line_l_prime c) circle_C → c = -3 ∨ c = -13 :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1082_108250


namespace NUMINAMATH_CALUDE_hypotenuse_segment_ratio_l1082_108232

/-- Represents a right triangle with a perpendicular from the right angle to the hypotenuse -/
structure RightTriangleWithAltitude where
  /-- Length of the shorter leg -/
  short_leg : ℝ
  /-- Length of the longer leg -/
  long_leg : ℝ
  /-- The longer leg is 3 times the shorter leg -/
  leg_ratio : long_leg = 3 * short_leg
  /-- Length of the segment of the hypotenuse adjacent to the shorter leg -/
  hyp_short : ℝ
  /-- Length of the segment of the hypotenuse adjacent to the longer leg -/
  hyp_long : ℝ
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : short_leg ^ 2 + long_leg ^ 2 = (hyp_short + hyp_long) ^ 2

/-- The main theorem: the ratio of hypotenuse segments is 9:1 -/
theorem hypotenuse_segment_ratio (t : RightTriangleWithAltitude) : 
  t.hyp_long / t.hyp_short = 9 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_segment_ratio_l1082_108232


namespace NUMINAMATH_CALUDE_inverse_proportion_l1082_108297

/-- Given that x is inversely proportional to y, if x = 4 when y = -2, 
    then x = 4/5 when y = -10 -/
theorem inverse_proportion (x y : ℝ) (h : ∃ k : ℝ, ∀ x y, x * y = k) 
  (h1 : 4 * (-2) = x * y) : 
  x * (-10) = 4/5 * (-10) := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_l1082_108297


namespace NUMINAMATH_CALUDE_min_ratio_digit_difference_l1082_108203

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def digit_sum (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def ratio (n : ℕ) : ℚ := n / (digit_sum n)

def ten_thousands_digit (n : ℕ) : ℕ := n / 10000

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10

theorem min_ratio_digit_difference :
  ∃ (n : ℕ), is_five_digit n ∧
  (∀ (m : ℕ), is_five_digit m → ratio n ≤ ratio m) ∧
  (thousands_digit n - ten_thousands_digit n = 8) :=
sorry

end NUMINAMATH_CALUDE_min_ratio_digit_difference_l1082_108203


namespace NUMINAMATH_CALUDE_scooter_profit_percentage_l1082_108289

/-- Calculates the profit percentage for a scooter sale given specific conditions -/
theorem scooter_profit_percentage 
  (initial_price : ℝ)
  (initial_repair_rate : ℝ)
  (additional_maintenance : ℝ)
  (safety_upgrade_rate : ℝ)
  (sales_tax_rate : ℝ)
  (selling_price : ℝ)
  (h1 : initial_price = 4700)
  (h2 : initial_repair_rate = 0.1)
  (h3 : additional_maintenance = 500)
  (h4 : safety_upgrade_rate = 0.05)
  (h5 : sales_tax_rate = 0.12)
  (h6 : selling_price = 5800) :
  let initial_repair := initial_price * initial_repair_rate
  let total_repair := initial_repair + additional_maintenance
  let safety_upgrade := total_repair * safety_upgrade_rate
  let total_cost := initial_price + total_repair + safety_upgrade
  let sales_tax := selling_price * sales_tax_rate
  let total_selling_price := selling_price + sales_tax
  let profit := total_selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  ∃ ε > 0, abs (profit_percentage - 13.60) < ε :=
by sorry

end NUMINAMATH_CALUDE_scooter_profit_percentage_l1082_108289


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_decomposition_l1082_108280

/-- A cyclic quadrilateral is a quadrilateral that can be circumscribed about a circle. -/
def CyclicQuadrilateral : Type := sorry

/-- A decomposition of a quadrilateral into n smaller quadrilaterals. -/
def Decomposition (Q : CyclicQuadrilateral) (n : ℕ) : Type := sorry

/-- Predicate to check if all quadrilaterals in a decomposition are cyclic. -/
def AllCyclic (d : Decomposition Q n) : Prop := sorry

theorem cyclic_quadrilateral_decomposition (n : ℕ) (Q : CyclicQuadrilateral) 
  (h : n ≥ 4) : 
  ∃ (d : Decomposition Q n), AllCyclic d := by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_decomposition_l1082_108280


namespace NUMINAMATH_CALUDE_square_area_ratio_l1082_108252

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 2 * b) : b^2 = 4 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1082_108252


namespace NUMINAMATH_CALUDE_equation_solution_l1082_108226

theorem equation_solution (x : ℝ) : 
  3 / (x + 2) = 2 / (x - 1) → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1082_108226


namespace NUMINAMATH_CALUDE_cube_root_of_square_l1082_108272

theorem cube_root_of_square (x : ℝ) : x > 0 → (x^2)^(1/3) = x^(2/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_of_square_l1082_108272


namespace NUMINAMATH_CALUDE_clara_weight_l1082_108225

/-- Given two weights satisfying certain conditions, prove that one of them is 88 pounds. -/
theorem clara_weight (alice_weight clara_weight : ℝ) 
  (h1 : alice_weight + clara_weight = 220)
  (h2 : clara_weight - alice_weight = clara_weight / 3) : 
  clara_weight = 88 := by
  sorry

end NUMINAMATH_CALUDE_clara_weight_l1082_108225


namespace NUMINAMATH_CALUDE_initial_average_marks_l1082_108234

theorem initial_average_marks 
  (n : ℕ) 
  (incorrect_mark correct_mark : ℕ) 
  (correct_average : ℚ) : 
  n = 10 → 
  incorrect_mark = 90 → 
  correct_mark = 10 → 
  correct_average = 92 → 
  (n * correct_average + (incorrect_mark - correct_mark)) / n = 100 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_marks_l1082_108234


namespace NUMINAMATH_CALUDE_all_dice_even_probability_l1082_108224

/-- The probability of a single standard six-sided die showing an even number -/
def prob_single_even : ℚ := 1 / 2

/-- The number of dice being tossed simultaneously -/
def num_dice : ℕ := 5

/-- The probability of all dice showing an even number -/
def prob_all_even : ℚ := (prob_single_even) ^ num_dice

theorem all_dice_even_probability :
  prob_all_even = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_all_dice_even_probability_l1082_108224


namespace NUMINAMATH_CALUDE_sqrt_x_minus_4_real_range_l1082_108262

theorem sqrt_x_minus_4_real_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 4) ↔ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_4_real_range_l1082_108262


namespace NUMINAMATH_CALUDE_withdrawal_amount_l1082_108246

def initial_balance : ℕ := 65
def deposit : ℕ := 15
def final_balance : ℕ := 76

theorem withdrawal_amount : 
  initial_balance + deposit - final_balance = 4 := by
  sorry

end NUMINAMATH_CALUDE_withdrawal_amount_l1082_108246


namespace NUMINAMATH_CALUDE_grid_shading_theorem_l1082_108277

/-- Represents a square on the grid -/
structure Square where
  row : Fin 6
  col : Fin 6

/-- Determines if a square is shaded based on its position -/
def is_shaded (s : Square) : Prop :=
  (s.row % 2 = 0 ∧ s.col % 2 = 1) ∨ (s.row % 2 = 1 ∧ s.col % 2 = 0)

/-- The total number of squares in the grid -/
def total_squares : Nat := 36

/-- The number of shaded squares in the grid -/
def shaded_squares : Nat := 21

/-- The fraction of shaded squares in the grid -/
def shaded_fraction : Rat := 7 / 12

theorem grid_shading_theorem :
  (shaded_squares : Rat) / total_squares = shaded_fraction := by
  sorry

end NUMINAMATH_CALUDE_grid_shading_theorem_l1082_108277


namespace NUMINAMATH_CALUDE_three_digit_sum_9_l1082_108215

/-- A function that generates all three-digit numbers using digits 1 to 5 -/
def generateNumbers : List (Fin 5 × Fin 5 × Fin 5) := sorry

/-- A function that checks if the sum of digits in a three-digit number is 9 -/
def sumIs9 (n : Fin 5 × Fin 5 × Fin 5) : Bool := sorry

/-- The theorem to be proved -/
theorem three_digit_sum_9 : 
  (generateNumbers.filter sumIs9).length = 19 := by sorry

end NUMINAMATH_CALUDE_three_digit_sum_9_l1082_108215


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l1082_108264

theorem roots_quadratic_equation (x₁ x₂ : ℝ) : 
  (x₁^2 + 3*x₁ - 2 = 0) → 
  (x₂^2 + 3*x₂ - 2 = 0) → 
  (x₁^2 + 2*x₁ - x₂ = 5) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l1082_108264


namespace NUMINAMATH_CALUDE_adjacent_removal_unequal_sums_l1082_108296

theorem adjacent_removal_unequal_sums (arrangement : List ℕ) : 
  arrangement.length = 2005 → 
  ∃ (i : Fin 2005), 
    ¬∃ (partition : List ℕ → List ℕ × List ℕ), 
      let remaining := arrangement.removeNth i.val ++ arrangement.removeNth ((i.val + 1) % 2005)
      (partition remaining).1.sum = (partition remaining).2.sum :=
by sorry

end NUMINAMATH_CALUDE_adjacent_removal_unequal_sums_l1082_108296


namespace NUMINAMATH_CALUDE_abc_product_equals_k_l1082_108214

theorem abc_product_equals_k (a b c k : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → k ≠ 0 →
  a ≠ b → b ≠ c → a ≠ c →
  (a + k / b = b + k / c) → (b + k / c = c + k / a) →
  |a * b * c| = |k| := by
sorry

end NUMINAMATH_CALUDE_abc_product_equals_k_l1082_108214


namespace NUMINAMATH_CALUDE_binomial_square_condition_l1082_108291

/-- If 9x^2 + 30x + a is the square of a binomial, then a = 25 -/
theorem binomial_square_condition (a : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, 9*x^2 + 30*x + a = (b*x + c)^2) → a = 25 := by
sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l1082_108291


namespace NUMINAMATH_CALUDE_parallelogram_external_bisectors_rectangle_diagonal_l1082_108279

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a quadrilateral is a rectangle -/
def isRectangle (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the length of a line segment between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Represents the intersection points of external angle bisectors -/
structure ExternalBisectorPoints :=
  (P Q R S : Point)

/-- Checks if given points are formed by intersection of external angle bisectors -/
def areExternalBisectorPoints (q : Quadrilateral) (e : ExternalBisectorPoints) : Prop :=
  sorry

/-- Main theorem -/
theorem parallelogram_external_bisectors_rectangle_diagonal
  (ABCD : Quadrilateral)
  (PQRS : ExternalBisectorPoints) :
  isParallelogram ABCD →
  areExternalBisectorPoints ABCD PQRS →
  isRectangle ⟨PQRS.P, PQRS.Q, PQRS.R, PQRS.S⟩ →
  distance PQRS.P PQRS.R = distance ABCD.A ABCD.B + distance ABCD.B ABCD.C :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_external_bisectors_rectangle_diagonal_l1082_108279


namespace NUMINAMATH_CALUDE_parabola_properties_l1082_108286

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the focus F
def focus : ℝ × ℝ := (4, 0)

-- Define a point M on the parabola
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

-- Define N as a point on the y-axis
def point_on_y_axis (N : ℝ × ℝ) : Prop :=
  N.1 = 0

-- Define M as the midpoint of FN
def is_midpoint (F M N : ℝ × ℝ) : Prop :=
  M.1 = (F.1 + N.1) / 2 ∧ M.2 = (F.2 + N.2) / 2

-- Main theorem
theorem parabola_properties (M N : ℝ × ℝ) 
  (h1 : point_on_parabola M)
  (h2 : point_on_y_axis N)
  (h3 : is_midpoint focus M N) :
  (∀ x y, y^2 = 16 * x → x = -4 → False) ∧  -- Directrix equation
  (Real.sqrt ((focus.1 - N.1)^2 + (focus.2 - N.2)^2) = 12) ∧  -- |FN| = 12
  (1/2 * focus.1 * N.2 = 16 * Real.sqrt 2) :=  -- Area of triangle ONF
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1082_108286


namespace NUMINAMATH_CALUDE_consecutive_product_divisible_by_six_l1082_108253

theorem consecutive_product_divisible_by_six (n : ℤ) : ∃ k : ℤ, n * (n + 1) * (n + 2) = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_divisible_by_six_l1082_108253


namespace NUMINAMATH_CALUDE_sugar_solution_sweetness_l1082_108240

theorem sugar_solution_sweetness (a b t : ℝ) (ha : a > 0) (hb : b > 0) (ht : t > 0) (hab : a > b) :
  (b + t) / (a + t) > b / a :=
by sorry

end NUMINAMATH_CALUDE_sugar_solution_sweetness_l1082_108240


namespace NUMINAMATH_CALUDE_landscape_breadth_l1082_108223

theorem landscape_breadth (length width : ℝ) (playground_area : ℝ) : 
  width = 8 * length →
  playground_area = 3200 →
  playground_area = (1 / 9) * (length * width) →
  width = 480 := by
sorry

end NUMINAMATH_CALUDE_landscape_breadth_l1082_108223


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1082_108248

theorem quadratic_transformation (a b c : ℝ) :
  (∃ m q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = 5 * (x - 3)^2 + 7) →
  (∃ m p q : ℝ, ∀ x : ℝ, 4 * a * x^2 + 4 * b * x + 4 * c = m * (x - p)^2 + q) →
  (∃ m p q : ℝ, ∀ x : ℝ, 4 * a * x^2 + 4 * b * x + 4 * c = m * (x - p)^2 + q ∧ p = 3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1082_108248


namespace NUMINAMATH_CALUDE_single_tile_replacement_impossible_l1082_108245

/-- Represents the two types of tiles used for paving -/
inductive TileType
  | Rectangle4x1
  | Square2x2

/-- Represents a rectangular floor -/
structure Floor :=
  (width : ℕ)
  (height : ℕ)
  (tiling : List (TileType))

/-- Checks if a tiling is valid for the given floor -/
def is_valid_tiling (floor : Floor) : Prop :=
  sorry

/-- Represents the operation of replacing a single tile -/
def replace_single_tile (floor : Floor) (old_type new_type : TileType) : Floor :=
  sorry

/-- The main theorem stating that replacing a single tile
    with a different type always results in an invalid tiling -/
theorem single_tile_replacement_impossible (floor : Floor) :
  ∀ (old_type new_type : TileType),
    old_type ≠ new_type →
    is_valid_tiling floor →
    ¬(is_valid_tiling (replace_single_tile floor old_type new_type)) :=
  sorry

end NUMINAMATH_CALUDE_single_tile_replacement_impossible_l1082_108245


namespace NUMINAMATH_CALUDE_unique_solution_geometric_series_l1082_108249

theorem unique_solution_geometric_series :
  ∃! x : ℝ, |x| < 1 ∧ x = (1 : ℝ) / (1 + x) ∧ x = (-1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_geometric_series_l1082_108249


namespace NUMINAMATH_CALUDE_toms_total_cost_is_48_l1082_108233

/-- Represents the fruit prices and quantities --/
structure FruitPurchase where
  lemon_price : ℝ
  papaya_price : ℝ
  mango_price : ℝ
  orange_price : ℝ
  apple_price : ℝ
  pineapple_price : ℝ
  lemon_qty : ℕ
  papaya_qty : ℕ
  mango_qty : ℕ
  orange_qty : ℕ
  apple_qty : ℕ
  pineapple_qty : ℕ

/-- Calculates the total cost after all discounts --/
def totalCostAfterDiscounts (purchase : FruitPurchase) (customer_number : ℕ) : ℝ :=
  sorry

/-- Theorem stating that Tom's total cost after all discounts is $48 --/
theorem toms_total_cost_is_48 :
  let purchase : FruitPurchase := {
    lemon_price := 2,
    papaya_price := 1,
    mango_price := 4,
    orange_price := 3,
    apple_price := 1.5,
    pineapple_price := 5,
    lemon_qty := 8,
    papaya_qty := 6,
    mango_qty := 5,
    orange_qty := 3,
    apple_qty := 8,
    pineapple_qty := 2
  }
  totalCostAfterDiscounts purchase 7 = 48 := by sorry

end NUMINAMATH_CALUDE_toms_total_cost_is_48_l1082_108233


namespace NUMINAMATH_CALUDE_line_point_sum_l1082_108238

/-- The line equation y = -2/5 * x + 10 -/
def line_equation (x y : ℝ) : Prop := y = -2/5 * x + 10

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (25, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 10)

/-- Point T is on line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
  r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- The area of triangle POQ is 4 times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop := 
  abs ((P.1 * Q.2 - Q.1 * P.2) / 2) = 4 * abs ((P.1 * s - r * P.2) / 2)

/-- Main theorem -/
theorem line_point_sum (r s : ℝ) 
  (h1 : line_equation r s) 
  (h2 : T_on_PQ r s) 
  (h3 : area_condition r s) : 
  r + s = 21.25 := by sorry

end NUMINAMATH_CALUDE_line_point_sum_l1082_108238


namespace NUMINAMATH_CALUDE_f_properties_l1082_108219

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/4)^x - 8 * (1/2)^x - 1
  else if x = 0 then 0
  else -4^x + 8 * 2^x + 1

theorem f_properties :
  (∀ x, f x + f (-x) = 0) →
  (∀ x < 0, f x = (1/4)^x - 8 * (1/2)^x - 1) →
  (∀ x > 0, f x = -4^x + 8 * 2^x + 1) ∧
  (∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, f y ≤ f x) ∧
  f 2 = 17 ∧
  (∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, f x ≤ f y) ∧
  f 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1082_108219


namespace NUMINAMATH_CALUDE_sales_executive_target_earning_l1082_108231

/-- Calculates the target monthly earning for a sales executive --/
def target_monthly_earning (fixed_salary : ℝ) (commission_rate : ℝ) (required_sales : ℝ) : ℝ :=
  fixed_salary + commission_rate * required_sales

/-- Proves that the target monthly earning is $5000 given the specified conditions --/
theorem sales_executive_target_earning :
  target_monthly_earning 1000 0.05 80000 = 5000 := by
sorry

end NUMINAMATH_CALUDE_sales_executive_target_earning_l1082_108231


namespace NUMINAMATH_CALUDE_lisa_caffeine_over_goal_l1082_108292

/-- The amount of caffeine Lisa consumed over her goal -/
def caffeine_over_goal (caffeine_per_cup : ℕ) (daily_goal : ℕ) (cups_consumed : ℕ) : ℕ :=
  max ((caffeine_per_cup * cups_consumed) - daily_goal) 0

/-- Theorem stating that Lisa consumed 40 mg of caffeine over her goal -/
theorem lisa_caffeine_over_goal :
  caffeine_over_goal 80 200 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_lisa_caffeine_over_goal_l1082_108292


namespace NUMINAMATH_CALUDE_sequence_formula_l1082_108281

theorem sequence_formula (a b : ℕ → ℝ) : 
  (∀ n, a n > 0 ∧ b n > 0) →  -- Each term is positive
  (∀ n, 2 * b n = a n + a (n + 1)) →  -- Arithmetic sequence condition
  (∀ n, (a (n + 1))^2 = b n * b (n + 1)) →  -- Geometric sequence condition
  a 1 = 1 →  -- Initial condition
  a 2 = 3 →  -- Initial condition
  (∀ n, a n = (n^2 + n) / 2) :=  -- General term formula
by sorry

end NUMINAMATH_CALUDE_sequence_formula_l1082_108281


namespace NUMINAMATH_CALUDE_marble_prism_weight_l1082_108200

/-- Represents the properties of a rectangular prism -/
structure RectangularPrism where
  height : ℝ
  baseLength : ℝ
  density : ℝ

/-- Calculates the weight of a rectangular prism -/
def weight (prism : RectangularPrism) : ℝ :=
  prism.height * prism.baseLength * prism.baseLength * prism.density

/-- Theorem: The weight of the specified marble rectangular prism is 86400 kg -/
theorem marble_prism_weight :
  let prism : RectangularPrism := {
    height := 8,
    baseLength := 2,
    density := 2700
  }
  weight prism = 86400 := by
  sorry

end NUMINAMATH_CALUDE_marble_prism_weight_l1082_108200


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l1082_108276

theorem smallest_sum_B_plus_c : ∃ (B c : ℕ),
  (0 ≤ B ∧ B ≤ 4) ∧
  (c > 6) ∧
  (31 * B = 4 * (c + 1)) ∧
  (∀ (B' c' : ℕ), (0 ≤ B' ∧ B' ≤ 4) ∧ (c' > 6) ∧ (31 * B' = 4 * (c' + 1)) → B + c ≤ B' + c') ∧
  B + c = 34 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l1082_108276


namespace NUMINAMATH_CALUDE_determine_dracula_status_l1082_108216

/-- Represents the types of Transylvanians -/
inductive TransylvanianType
| Truthful
| Liar

/-- Represents the possible answers to a yes/no question -/
inductive Answer
| Yes
| No

/-- Represents Dracula's status -/
inductive DraculaStatus
| Alive
| NotAlive

/-- A Transylvanian's response to the question -/
def response (t : TransylvanianType) (d : DraculaStatus) : Answer :=
  match t, d with
  | TransylvanianType.Truthful, DraculaStatus.Alive => Answer.Yes
  | TransylvanianType.Truthful, DraculaStatus.NotAlive => Answer.No
  | TransylvanianType.Liar, DraculaStatus.Alive => Answer.Yes
  | TransylvanianType.Liar, DraculaStatus.NotAlive => Answer.No

/-- The main theorem: The question can determine Dracula's status -/
theorem determine_dracula_status :
  ∀ (t : TransylvanianType) (d : DraculaStatus),
    response t d = Answer.Yes ↔ d = DraculaStatus.Alive :=
by sorry

end NUMINAMATH_CALUDE_determine_dracula_status_l1082_108216


namespace NUMINAMATH_CALUDE_monthly_revenue_is_4000_l1082_108206

/-- A store's financial data -/
structure StoreFinancials where
  initial_investment : ℕ
  monthly_expenses : ℕ
  payback_period : ℕ

/-- Calculate the monthly revenue required to break even -/
def calculate_monthly_revenue (store : StoreFinancials) : ℕ :=
  (store.initial_investment + store.monthly_expenses * store.payback_period) / store.payback_period

/-- Theorem: Given the store's financial data, the monthly revenue is $4000 -/
theorem monthly_revenue_is_4000 (store : StoreFinancials) 
    (h1 : store.initial_investment = 25000)
    (h2 : store.monthly_expenses = 1500)
    (h3 : store.payback_period = 10) :
  calculate_monthly_revenue store = 4000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_revenue_is_4000_l1082_108206


namespace NUMINAMATH_CALUDE_probability_purple_face_l1082_108236

/-- The probability of rolling a purple face on a 10-sided die with 3 purple faces is 3/10. -/
theorem probability_purple_face (total_faces : ℕ) (purple_faces : ℕ) 
  (h1 : total_faces = 10) (h2 : purple_faces = 3) : 
  (purple_faces : ℚ) / total_faces = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_purple_face_l1082_108236


namespace NUMINAMATH_CALUDE_running_speed_calculation_l1082_108255

def walking_speed : ℝ := 4
def total_distance : ℝ := 20
def total_time : ℝ := 3.75

theorem running_speed_calculation (R : ℝ) :
  (total_distance / 2 / walking_speed) + (total_distance / 2 / R) = total_time →
  R = 8 := by
  sorry

end NUMINAMATH_CALUDE_running_speed_calculation_l1082_108255


namespace NUMINAMATH_CALUDE_sum_of_four_digit_odd_and_multiples_of_ten_l1082_108298

/-- The number of four-digit odd numbers -/
def A : ℕ := 4500

/-- The number of four-digit multiples of 10 -/
def B : ℕ := 900

/-- The sum of four-digit odd numbers and four-digit multiples of 10 is 5400 -/
theorem sum_of_four_digit_odd_and_multiples_of_ten : A + B = 5400 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_odd_and_multiples_of_ten_l1082_108298


namespace NUMINAMATH_CALUDE_line_through_circle_center_l1082_108273

theorem line_through_circle_center (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 ∧ 3*x + y + a = 0) → 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 → 3*x + y + a = 0 → x = -1 ∧ y = 2) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l1082_108273


namespace NUMINAMATH_CALUDE_focus_of_given_parabola_l1082_108263

/-- The parabola equation is y = (1/8)x^2 -/
def parabola_equation (x y : ℝ) : Prop := y = (1/8) * x^2

/-- The focus of a parabola with equation x^2 = 4py is at (0, p) -/
def focus_of_standard_parabola (p : ℝ) : ℝ × ℝ := (0, p)

/-- The theorem stating that the focus of the parabola y = (1/8)x^2 is at (0, 2) -/
theorem focus_of_given_parabola :
  ∃ (p : ℝ), (∀ x y : ℝ, parabola_equation x y ↔ x^2 = 4*p*y) ∧
             focus_of_standard_parabola p = (0, 2) := by sorry

end NUMINAMATH_CALUDE_focus_of_given_parabola_l1082_108263


namespace NUMINAMATH_CALUDE_apartment_utilities_cost_l1082_108247

/-- Represents the monthly costs and driving distance for an apartment --/
structure Apartment where
  rent : ℝ
  utilities : ℝ
  driveMiles : ℝ

/-- Calculates the total monthly cost for an apartment --/
def totalMonthlyCost (apt : Apartment) (workdays : ℝ) (driveCostPerMile : ℝ) : ℝ :=
  apt.rent + apt.utilities + (apt.driveMiles * workdays * driveCostPerMile)

/-- The problem statement --/
theorem apartment_utilities_cost 
  (apt1 : Apartment)
  (apt2 : Apartment)
  (workdays : ℝ)
  (driveCostPerMile : ℝ)
  (totalCostDifference : ℝ)
  (h1 : apt1.rent = 800)
  (h2 : apt1.utilities = 260)
  (h3 : apt1.driveMiles = 31)
  (h4 : apt2.rent = 900)
  (h5 : apt2.driveMiles = 21)
  (h6 : workdays = 20)
  (h7 : driveCostPerMile = 0.58)
  (h8 : totalMonthlyCost apt1 workdays driveCostPerMile - 
        totalMonthlyCost apt2 workdays driveCostPerMile = totalCostDifference)
  (h9 : totalCostDifference = 76) :
  apt2.utilities = 200 := by
  sorry


end NUMINAMATH_CALUDE_apartment_utilities_cost_l1082_108247


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1082_108266

theorem arithmetic_expression_evaluation :
  37 + (87 / 29) + (15 * 19) - 100 - (450 / 15) + 13 = 208 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1082_108266


namespace NUMINAMATH_CALUDE_min_n_for_divisibility_by_20_l1082_108251

theorem min_n_for_divisibility_by_20 :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (S : Finset ℕ), S.card = n →
    ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℕ), T.card = m ∧
    ¬∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  n = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_divisibility_by_20_l1082_108251


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_23_l1082_108201

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_four_digit_sum_23 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 23 → n ≤ 9950 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_23_l1082_108201


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1082_108207

/-- Sum of first n terms of an arithmetic sequence -/
def T (b : ℚ) (n : ℕ) : ℚ := n * (2 * b + (n - 1) * 5) / 2

/-- The theorem statement -/
theorem arithmetic_sequence_first_term :
  (∃ (k : ℚ), ∀ (n : ℕ), n > 0 → T b (4 * n) / T b n = k) →
  b = 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1082_108207


namespace NUMINAMATH_CALUDE_chairs_difference_l1082_108229

theorem chairs_difference (initial : ℕ) (remaining : ℕ) : 
  initial = 15 → remaining = 3 → initial - remaining = 12 := by
  sorry

end NUMINAMATH_CALUDE_chairs_difference_l1082_108229


namespace NUMINAMATH_CALUDE_fruit_arrangements_l1082_108260

/-- The number of distinct arrangements of 9 items, where there are 4 indistinguishable items of type A, 3 indistinguishable items of type B, and 2 indistinguishable items of type C. -/
def distinct_arrangements (total : Nat) (a : Nat) (b : Nat) (c : Nat) : Nat :=
  Nat.factorial total / (Nat.factorial a * Nat.factorial b * Nat.factorial c)

/-- Theorem stating that the number of distinct arrangements of 9 items, 
    where there are 4 indistinguishable items of type A, 
    3 indistinguishable items of type B, and 2 indistinguishable items of type C, 
    is equal to 1260. -/
theorem fruit_arrangements : distinct_arrangements 9 4 3 2 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangements_l1082_108260


namespace NUMINAMATH_CALUDE_cube_sum_simplification_l1082_108282

theorem cube_sum_simplification (a b c : ℝ) :
  (a^3 + b^3) / (c^3 + b^3) = (a + b) / (c + b) ↔ a + c = b :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_simplification_l1082_108282


namespace NUMINAMATH_CALUDE_box_cubes_required_l1082_108295

theorem box_cubes_required (length width height cube_volume : ℕ) 
  (h1 : length = 12)
  (h2 : width = 16)
  (h3 : height = 6)
  (h4 : cube_volume = 3) : 
  (length * width * height) / cube_volume = 384 := by
  sorry

end NUMINAMATH_CALUDE_box_cubes_required_l1082_108295


namespace NUMINAMATH_CALUDE_sequence_problem_l1082_108237

theorem sequence_problem (n : ℕ+) (b : ℕ → ℝ)
  (h0 : b 0 = 25)
  (h1 : b 1 = 56)
  (hn : b n = 0)
  (hk : ∀ k : ℕ, 1 ≤ k → k < n → b (k + 1) = b (k - 1) - 7 / b k) :
  n = 201 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1082_108237


namespace NUMINAMATH_CALUDE_exactly_two_pass_probability_l1082_108265

def prob_A : ℚ := 4/5
def prob_B : ℚ := 3/4
def prob_C : ℚ := 3/4

theorem exactly_two_pass_probability : 
  let prob_AB := prob_A * prob_B * (1 - prob_C)
  let prob_AC := prob_A * (1 - prob_B) * prob_C
  let prob_BC := (1 - prob_A) * prob_B * prob_C
  prob_AB + prob_AC + prob_BC = 33/80 :=
by sorry

end NUMINAMATH_CALUDE_exactly_two_pass_probability_l1082_108265


namespace NUMINAMATH_CALUDE_abs_neg_2023_l1082_108204

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l1082_108204


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2985_l1082_108294

theorem smallest_prime_factor_of_2985 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2985 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2985 → p ≤ q :=
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2985_l1082_108294


namespace NUMINAMATH_CALUDE_juku_exit_position_l1082_108278

/-- Represents the state of Juku on the escalator -/
structure EscalatorState where
  time : ℕ
  position : ℕ

/-- The escalator system with Juku's movement -/
def escalator_system (total_steps : ℕ) (start_position : ℕ) : ℕ → EscalatorState
| 0 => ⟨0, start_position⟩
| t + 1 => 
  let prev := escalator_system total_steps start_position t
  let new_pos := 
    if t % 3 == 0 then prev.position - 1
    else if t % 3 == 1 then prev.position + 1
    else prev.position - 2
  ⟨t + 1, new_pos⟩

/-- Theorem: Juku exits at the 23rd step relative to the ground -/
theorem juku_exit_position : 
  ∃ (t : ℕ), (escalator_system 75 38 t).position + (t / 2) = 23 := by
  sorry

#eval (escalator_system 75 38 45).position + 45 / 2

end NUMINAMATH_CALUDE_juku_exit_position_l1082_108278


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1082_108287

def polynomial (x : ℝ) : ℝ :=
  -3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (2*x^4 + 3*x^2 - x) - 2 * (3*x^6 - 7)

theorem sum_of_coefficients : 
  (polynomial 1) = 37 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1082_108287


namespace NUMINAMATH_CALUDE_power_sum_division_l1082_108267

theorem power_sum_division (x y : ℕ) (hx : x = 3) (hy : y = 4) : (x^5 + 3*y^3) / 9 = 48 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_division_l1082_108267


namespace NUMINAMATH_CALUDE_alice_current_age_l1082_108288

/-- Alice's current age -/
def alice_age : ℕ := 30

/-- Beatrice's current age -/
def beatrice_age : ℕ := 11

/-- In 8 years, Alice will be twice as old as Beatrice -/
axiom future_age_relation : alice_age + 8 = 2 * (beatrice_age + 8)

/-- Ten years ago, the sum of their ages was 21 -/
axiom past_age_sum : (alice_age - 10) + (beatrice_age - 10) = 21

theorem alice_current_age : alice_age = 30 := by
  sorry

end NUMINAMATH_CALUDE_alice_current_age_l1082_108288


namespace NUMINAMATH_CALUDE_tim_total_sleep_l1082_108244

/-- Tim's weekly sleep schedule -/
structure SleepSchedule where
  weekdays : Nat -- Number of weekdays
  weekdaySleep : Nat -- Hours of sleep on weekdays
  weekends : Nat -- Number of weekend days
  weekendSleep : Nat -- Hours of sleep on weekends

/-- Calculate total sleep based on a sleep schedule -/
def totalSleep (schedule : SleepSchedule) : Nat :=
  schedule.weekdays * schedule.weekdaySleep + schedule.weekends * schedule.weekendSleep

/-- Tim's actual sleep schedule -/
def timSchedule : SleepSchedule :=
  { weekdays := 5
    weekdaySleep := 6
    weekends := 2
    weekendSleep := 10 }

/-- Theorem: Tim's total sleep per week is 50 hours -/
theorem tim_total_sleep : totalSleep timSchedule = 50 := by
  sorry

end NUMINAMATH_CALUDE_tim_total_sleep_l1082_108244


namespace NUMINAMATH_CALUDE_abs_neg_nine_equals_nine_l1082_108256

theorem abs_neg_nine_equals_nine : abs (-9 : ℤ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_nine_equals_nine_l1082_108256


namespace NUMINAMATH_CALUDE_math_reading_homework_difference_l1082_108284

theorem math_reading_homework_difference (reading_pages math_pages : ℕ) 
  (h1 : reading_pages = 12) 
  (h2 : math_pages = 23) : 
  math_pages - reading_pages = 11 := by
  sorry

end NUMINAMATH_CALUDE_math_reading_homework_difference_l1082_108284


namespace NUMINAMATH_CALUDE_problem_solution_l1082_108242

theorem problem_solution : 
  (Real.sqrt (7^2 + 24^2)) / (Real.sqrt (49 + 16)) = (5 * Real.sqrt 65) / 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1082_108242


namespace NUMINAMATH_CALUDE_quadratic_root_reciprocal_sum_l1082_108211

theorem quadratic_root_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ - 2 = 0 → 
  x₂^2 - 4*x₂ - 2 = 0 → 
  x₁ ≠ x₂ →
  1/x₁ + 1/x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_reciprocal_sum_l1082_108211


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1082_108209

def complex_number : ℂ := 2 - Complex.I

theorem point_in_fourth_quadrant (z : ℂ) (h : z = complex_number) :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1082_108209


namespace NUMINAMATH_CALUDE_gcf_of_24_and_16_l1082_108212

theorem gcf_of_24_and_16 :
  let n : ℕ := 24
  let m : ℕ := 16
  let lcm_nm : ℕ := 48
  lcm n m = lcm_nm →
  Nat.gcd n m = 8 := by
sorry

end NUMINAMATH_CALUDE_gcf_of_24_and_16_l1082_108212


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1082_108261

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x - 1 ≤ 0) → -4 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1082_108261


namespace NUMINAMATH_CALUDE_quadratic_crosses_origin_l1082_108208

/-- Given a quadratic function g(x) = ax^2 + bx where a ≠ 0 and b ≠ 0,
    the graph crosses the x-axis at the origin. -/
theorem quadratic_crosses_origin (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let g : ℝ → ℝ := λ x ↦ a * x^2 + b * x
  (g 0 = 0) ∧ (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε \ {0}, g x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_crosses_origin_l1082_108208


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l1082_108283

theorem cubic_roots_sum_of_cubes (p q : ℝ) (r s : ℂ) : 
  (r^3 - p*r^2 + q*r - p = 0) → 
  (s^3 - p*s^2 + q*s - p = 0) → 
  r^3 + s^3 = p^3 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l1082_108283


namespace NUMINAMATH_CALUDE_probability_of_vowel_in_four_consecutive_letters_l1082_108213

/-- Represents the English alphabet --/
def Alphabet : Finset Char := sorry

/-- Represents the vowels in the English alphabet --/
def Vowels : Finset Char := sorry

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The number of vowels --/
def vowel_count : ℕ := 5

/-- The number of possible sets of 4 consecutive letters --/
def consecutive_sets : ℕ := 23

/-- The number of sets of 4 consecutive letters without a vowel --/
def sets_without_vowel : ℕ := 5

/-- Theorem: The probability of selecting at least one vowel when choosing 4 consecutive letters at random from the English alphabet is 18/23 --/
theorem probability_of_vowel_in_four_consecutive_letters :
  (consecutive_sets - sets_without_vowel : ℚ) / consecutive_sets = 18 / 23 :=
sorry

end NUMINAMATH_CALUDE_probability_of_vowel_in_four_consecutive_letters_l1082_108213


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_two_l1082_108270

theorem no_solution_iff_n_eq_neg_two (n : ℤ) :
  (∀ x y : ℚ, 2 * x = 1 + n * y ∧ n * x = 1 + 2 * y) ↔ n = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_two_l1082_108270


namespace NUMINAMATH_CALUDE_score_difference_is_seven_l1082_108202

-- Define the score distribution
def score_distribution : List (Float × Float) := [
  (0.20, 60),
  (0.30, 70),
  (0.25, 85),
  (0.25, 95)
]

-- Define the mean score
def mean_score : Float :=
  (score_distribution.map (λ p => p.1 * p.2)).sum

-- Define the median score
def median_score : Float := 85

-- Theorem statement
theorem score_difference_is_seven :
  median_score - mean_score = 7 := by
  sorry


end NUMINAMATH_CALUDE_score_difference_is_seven_l1082_108202


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1082_108293

theorem perfect_square_condition (k : ℝ) : 
  (∃ (p : ℝ → ℝ), ∀ x, x^2 + 6*x + k^2 = (p x)^2) → (k = 3 ∨ k = -3) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1082_108293


namespace NUMINAMATH_CALUDE_fraction_lower_bound_l1082_108258

theorem fraction_lower_bound (p q r s : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) :
  ((p^2 + p + 1) * (q^2 + q + 1) * (r^2 + r + 1) * (s^2 + s + 1)) / (p * q * r * s) ≥ 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_lower_bound_l1082_108258


namespace NUMINAMATH_CALUDE_total_boxes_eq_sum_l1082_108222

/-- The total number of boxes Kaylee needs to sell -/
def total_boxes : ℕ := sorry

/-- The number of lemon biscuit boxes sold -/
def lemon_boxes : ℕ := 12

/-- The number of chocolate biscuit boxes sold -/
def chocolate_boxes : ℕ := 5

/-- The number of oatmeal biscuit boxes sold -/
def oatmeal_boxes : ℕ := 4

/-- The additional number of boxes Kaylee needs to sell -/
def additional_boxes : ℕ := 12

/-- Theorem stating that the total number of boxes is equal to the sum of all sold boxes and additional boxes -/
theorem total_boxes_eq_sum :
  total_boxes = lemon_boxes + chocolate_boxes + oatmeal_boxes + additional_boxes := by
  sorry

end NUMINAMATH_CALUDE_total_boxes_eq_sum_l1082_108222


namespace NUMINAMATH_CALUDE_intersection_M_N_l1082_108220

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 - 3*x = 0}

theorem intersection_M_N : M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1082_108220


namespace NUMINAMATH_CALUDE_expression_evaluation_l1082_108228

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := -8
  (Real.sqrt (9 * x * y) - 2 * Real.sqrt (x^3 * y) + Real.sqrt (x * y^3)) = 20 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1082_108228


namespace NUMINAMATH_CALUDE_correct_sum_exists_l1082_108254

def num1 : ℕ := 3742586
def num2 : ℕ := 4829430
def given_sum : ℕ := 72120116

def replace_digit (n : ℕ) (d e : ℕ) : ℕ :=
  -- Function to replace all occurrences of d with e in n
  sorry

theorem correct_sum_exists : ∃ (d e : ℕ), d ≠ e ∧ 
  d < 10 ∧ e < 10 ∧ 
  replace_digit num1 d e + replace_digit num2 d e = given_sum ∧
  d + e = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_sum_exists_l1082_108254


namespace NUMINAMATH_CALUDE_larger_number_problem_l1082_108230

theorem larger_number_problem (x y : ℤ) : 
  x + y = 56 → y = x + 12 → y = 34 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1082_108230


namespace NUMINAMATH_CALUDE_complex_expressions_equality_l1082_108239

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the equality to be proved
theorem complex_expressions_equality :
  ((-1/2 : ℂ) + (Real.sqrt 3/2)*i) * (2 - i) * (3 + i) = 
    (-3/2 : ℂ) + (5*Real.sqrt 3/2) + ((7*Real.sqrt 3 + 1)/2)*i ∧
  ((Real.sqrt 2 + Real.sqrt 2*i)^2 * (4 + 5*i)) / ((5 - 4*i) * (1 - i)) = 
    (62/41 : ℂ) + (80/41)*i :=
by sorry

-- Axiom for i^2 = -1
axiom i_squared : i^2 = -1

end NUMINAMATH_CALUDE_complex_expressions_equality_l1082_108239


namespace NUMINAMATH_CALUDE_max_product_sum_300_l1082_108235

theorem max_product_sum_300 : 
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 ∧ ∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l1082_108235


namespace NUMINAMATH_CALUDE_angle_sum_at_point_l1082_108243

theorem angle_sum_at_point (y : ℝ) : 
  y > 0 ∧ y + y + 140 = 360 → y = 110 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_at_point_l1082_108243


namespace NUMINAMATH_CALUDE_hat_problem_l1082_108259

/-- The number of customers -/
def n : ℕ := 5

/-- The probability that no customer gets their own hat when n customers randomly take hats -/
def prob_no_own_hat (n : ℕ) : ℚ :=
  sorry

theorem hat_problem : prob_no_own_hat n = 11/30 := by
  sorry

end NUMINAMATH_CALUDE_hat_problem_l1082_108259


namespace NUMINAMATH_CALUDE_inequality_solution_l1082_108205

theorem inequality_solution (x : ℝ) :
  (4 ≤ x^2 - 3*x - 6 ∧ x^2 - 3*x - 6 ≤ 2*x + 8) ↔ (5 ≤ x ∧ x ≤ 7) ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1082_108205


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1082_108221

/-- The equation (x + 3) / (kx - 2) = x + 1 has exactly one solution if and only if k = -7 + 2√10 or k = -7 - 2√10 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (x + 3) / (k * x - 2) = x + 1 ∧ k * x - 2 ≠ 0) ↔ 
  (k = -7 + 2 * Real.sqrt 10 ∨ k = -7 - 2 * Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1082_108221


namespace NUMINAMATH_CALUDE_tomato_difference_l1082_108271

theorem tomato_difference (initial_tomatoes picked_tomatoes : ℕ) 
  (h1 : initial_tomatoes = 17)
  (h2 : picked_tomatoes = 9) :
  initial_tomatoes - picked_tomatoes = 8 := by
  sorry

end NUMINAMATH_CALUDE_tomato_difference_l1082_108271


namespace NUMINAMATH_CALUDE_prism_with_21_edges_has_9_faces_l1082_108210

/-- A prism is a polyhedron with two congruent parallel faces (bases) and whose other faces (lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ :=
  let lateral_faces := p.edges / 3
  lateral_faces + 2

theorem prism_with_21_edges_has_9_faces (p : Prism) (h : p.edges = 21) : num_faces p = 9 := by
  sorry

end NUMINAMATH_CALUDE_prism_with_21_edges_has_9_faces_l1082_108210


namespace NUMINAMATH_CALUDE_reflection_matrix_conditions_l1082_108299

def reflection_matrix (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, b; -3/4, 1/4]

theorem reflection_matrix_conditions (a b : ℚ) :
  reflection_matrix a b * reflection_matrix a b = 1 ↔ a = -1/4 ∧ b = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_reflection_matrix_conditions_l1082_108299


namespace NUMINAMATH_CALUDE_probability_four_blue_marbles_l1082_108285

def total_marbles : ℕ := 20
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 12
def num_trials : ℕ := 8
def num_blue_picked : ℕ := 4

theorem probability_four_blue_marbles :
  (Nat.choose num_trials num_blue_picked) *
  (blue_marbles / total_marbles : ℚ) ^ num_blue_picked *
  (red_marbles / total_marbles : ℚ) ^ (num_trials - num_blue_picked) =
  90720 / 390625 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_blue_marbles_l1082_108285


namespace NUMINAMATH_CALUDE_add_fractions_simplest_form_l1082_108268

theorem add_fractions_simplest_form :
  (7 : ℚ) / 8 + (3 : ℚ) / 5 = (59 : ℚ) / 40 ∧ 
  ∀ n d : ℤ, (d ≠ 0 ∧ (59 : ℚ) / 40 = (n : ℚ) / d) → n.gcd d = 1 := by
  sorry

end NUMINAMATH_CALUDE_add_fractions_simplest_form_l1082_108268


namespace NUMINAMATH_CALUDE_equation_has_two_solutions_l1082_108269

-- Define the equation
def f (x : ℝ) : Prop := Real.sqrt (5 - x) = x * Real.sqrt (5 - x)

-- Theorem statement
theorem equation_has_two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ f a ∧ f b ∧ ∀ (x : ℝ), f x → (x = a ∨ x = b) :=
sorry

end NUMINAMATH_CALUDE_equation_has_two_solutions_l1082_108269
