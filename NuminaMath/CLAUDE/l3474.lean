import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_side_relation_l3474_347433

theorem right_triangle_side_relation (a d : ℝ) :
  (a > 0) →
  (d > 0) →
  (a ≤ a + 2*d) →
  (a + 2*d ≤ a + 4*d) →
  (a + 4*d)^2 = a^2 + (a + 2*d)^2 →
  a = d*(1 + Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_side_relation_l3474_347433


namespace NUMINAMATH_CALUDE_tv_screen_horizontal_length_l3474_347490

/-- Represents a rectangular TV screen --/
structure TVScreen where
  horizontal : ℝ
  vertical : ℝ
  diagonal : ℝ

/-- Theorem: Given a TV screen with horizontal to vertical ratio of 9:12 and
    diagonal of 32 inches, the horizontal length is 25.6 inches --/
theorem tv_screen_horizontal_length 
  (tv : TVScreen) 
  (ratio : tv.horizontal / tv.vertical = 9 / 12) 
  (diag : tv.diagonal = 32) :
  tv.horizontal = 25.6 := by
  sorry

#check tv_screen_horizontal_length

end NUMINAMATH_CALUDE_tv_screen_horizontal_length_l3474_347490


namespace NUMINAMATH_CALUDE_gretchen_earnings_l3474_347480

/-- Calculates the total earnings for Gretchen's caricature drawings over a weekend -/
def weekend_earnings (price_per_drawing : ℕ) (saturday_sales : ℕ) (sunday_sales : ℕ) : ℕ :=
  price_per_drawing * (saturday_sales + sunday_sales)

/-- Proves that Gretchen's earnings for the weekend are $800 -/
theorem gretchen_earnings :
  weekend_earnings 20 24 16 = 800 := by
sorry

end NUMINAMATH_CALUDE_gretchen_earnings_l3474_347480


namespace NUMINAMATH_CALUDE_average_visitors_is_288_l3474_347478

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def average_visitors_per_day (sunday_visitors : ℕ) (other_day_visitors : ℕ) : ℚ :=
  let num_sundays : ℕ := 30 / 7
  let num_other_days : ℕ := 30 - num_sundays
  let total_visitors : ℕ := sunday_visitors * num_sundays + other_day_visitors * num_other_days
  (total_visitors : ℚ) / 30

/-- Theorem stating that the average number of visitors per day is 288 -/
theorem average_visitors_is_288 :
  average_visitors_per_day 600 240 = 288 := by
  sorry

end NUMINAMATH_CALUDE_average_visitors_is_288_l3474_347478


namespace NUMINAMATH_CALUDE_calculate_expression_l3474_347471

theorem calculate_expression : (Real.pi - Real.sqrt 3) ^ 0 - 2 * Real.sin (45 * π / 180) + |-Real.sqrt 2| + Real.sqrt 8 = 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3474_347471


namespace NUMINAMATH_CALUDE_hidden_primes_average_l3474_347434

-- Define the type for our cards
structure Card where
  visible : ℕ
  hidden : ℕ

-- Define the property of being consecutive primes
def ConsecutivePrimes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ (p > q) ∧ ∀ k, q < k → k < p → ¬Nat.Prime k

-- State the theorem
theorem hidden_primes_average (card1 card2 : Card) :
  card1.visible = 18 →
  card2.visible = 27 →
  card1.visible + card1.hidden = card2.visible + card2.hidden →
  ConsecutivePrimes card1.hidden card2.hidden →
  card1.hidden - card2.hidden = 9 →
  (card1.hidden + card2.hidden) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_hidden_primes_average_l3474_347434


namespace NUMINAMATH_CALUDE_punch_water_calculation_l3474_347440

/-- Calculates the amount of water needed for a punch mixture -/
def water_needed (total_volume : ℚ) (water_parts : ℕ) (juice_parts : ℕ) : ℚ :=
  (total_volume * water_parts) / (water_parts + juice_parts)

/-- Theorem stating the amount of water needed for the specific punch recipe -/
theorem punch_water_calculation :
  water_needed 3 5 3 = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_punch_water_calculation_l3474_347440


namespace NUMINAMATH_CALUDE_branch_A_more_profitable_choose_branch_A_l3474_347423

/-- Represents the grades of products --/
inductive Grade
| A
| B
| C
| D

/-- Represents the branches of the factory --/
inductive Branch
| A
| B

/-- Processing fee for each grade --/
def processingFee (g : Grade) : Int :=
  match g with
  | Grade.A => 90
  | Grade.B => 50
  | Grade.C => 20
  | Grade.D => -50

/-- Processing cost for each branch --/
def processingCost (b : Branch) : Int :=
  match b with
  | Branch.A => 25
  | Branch.B => 20

/-- Frequency distribution for each branch --/
def frequency (b : Branch) (g : Grade) : Int :=
  match b, g with
  | Branch.A, Grade.A => 40
  | Branch.A, Grade.B => 20
  | Branch.A, Grade.C => 20
  | Branch.A, Grade.D => 20
  | Branch.B, Grade.A => 28
  | Branch.B, Grade.B => 17
  | Branch.B, Grade.C => 34
  | Branch.B, Grade.D => 21

/-- Calculate average profit for a branch --/
def averageProfit (b : Branch) : Int :=
  (processingFee Grade.A - processingCost b) * frequency b Grade.A +
  (processingFee Grade.B - processingCost b) * frequency b Grade.B +
  (processingFee Grade.C - processingCost b) * frequency b Grade.C +
  (processingFee Grade.D - processingCost b) * frequency b Grade.D

/-- Theorem: Branch A has higher average profit than Branch B --/
theorem branch_A_more_profitable :
  averageProfit Branch.A > averageProfit Branch.B :=
by sorry

/-- Corollary: Factory should choose Branch A --/
theorem choose_branch_A :
  ∀ b : Branch, b ≠ Branch.A → averageProfit Branch.A > averageProfit b :=
by sorry

end NUMINAMATH_CALUDE_branch_A_more_profitable_choose_branch_A_l3474_347423


namespace NUMINAMATH_CALUDE_problem_statement_l3474_347424

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 3 * a * (b^2 - 1) = b * (1 - a^2)) : 
  (1 / a + 3 / b = a + 3 * b) ∧ 
  (a^(3/2) * b^(1/2) + 3 * a^(1/2) * b^(3/2) ≥ 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3474_347424


namespace NUMINAMATH_CALUDE_gcd_upper_bound_l3474_347491

theorem gcd_upper_bound (a b : ℕ+) : Nat.gcd a.val b.val ≤ Real.sqrt (a.val + b.val : ℝ) := by sorry

end NUMINAMATH_CALUDE_gcd_upper_bound_l3474_347491


namespace NUMINAMATH_CALUDE_no_integer_solution_l3474_347493

theorem no_integer_solution : ¬∃ (a b c : ℤ), a^2 + b^2 + 1 = 4*c := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3474_347493


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3474_347425

/-- Given vectors a and b, find the unique value of t such that a is perpendicular to (t * a + b) -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (1, -1)) (h2 : b = (6, -4)) :
  ∃! t : ℝ, (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) ∧ t = -5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3474_347425


namespace NUMINAMATH_CALUDE_parallelogram_sum_l3474_347430

/-- A parallelogram with sides 12, 4z + 2, 3x - 1, and 7y + 3 -/
structure Parallelogram (x y z : ℚ) where
  side1 : ℚ := 12
  side2 : ℚ := 4 * z + 2
  side3 : ℚ := 3 * x - 1
  side4 : ℚ := 7 * y + 3
  opposite_sides_equal1 : side1 = side3
  opposite_sides_equal2 : side2 = side4

/-- The sum of x, y, and z in the parallelogram equals 121/21 -/
theorem parallelogram_sum (x y z : ℚ) (p : Parallelogram x y z) : x + y + z = 121/21 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_sum_l3474_347430


namespace NUMINAMATH_CALUDE_supermarket_purchase_cost_l3474_347431

/-- Calculates the total cost of items with given quantities, prices, and discounts -/
def totalCost (quantities : List ℕ) (prices : List ℚ) (discounts : List ℚ) : ℚ :=
  List.sum (List.zipWith3 (fun q p d => q * p * (1 - d)) quantities prices discounts)

/-- The problem statement -/
theorem supermarket_purchase_cost : 
  let quantities : List ℕ := [24, 6, 5, 3]
  let prices : List ℚ := [9/5, 17/10, 17/5, 56/5]
  let discounts : List ℚ := [1/5, 1/5, 0, 1/10]
  totalCost quantities prices discounts = 4498/50
  := by sorry

end NUMINAMATH_CALUDE_supermarket_purchase_cost_l3474_347431


namespace NUMINAMATH_CALUDE_solution_set_theorem_l3474_347458

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, f x + x * (deriv f x) > 0)

-- Define the theorem
theorem solution_set_theorem :
  {x : ℝ | (deriv f (Real.sqrt (x + 1))) > Real.sqrt (x - 1) * f (Real.sqrt (x^2 - 1))} =
  {x : ℝ | 1 ≤ x ∧ x < 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l3474_347458


namespace NUMINAMATH_CALUDE_units_digit_sum_squares_odd_plus_7_1011_l3474_347445

/-- The units digit of the sum of squares of the first n odd positive integers plus 7 -/
def units_digit_sum_squares_odd_plus_7 (n : ℕ) : ℕ :=
  (((List.range n).map (fun i => (2 * i + 1) ^ 2)).sum + 7) % 10

/-- Theorem stating that the units digit of the sum of squares of the first 1011 odd positive integers plus 7 is 2 -/
theorem units_digit_sum_squares_odd_plus_7_1011 :
  units_digit_sum_squares_odd_plus_7 1011 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_squares_odd_plus_7_1011_l3474_347445


namespace NUMINAMATH_CALUDE_complex_calculations_l3474_347465

theorem complex_calculations :
  (∃ (i : ℂ), i * i = -1) →
  (∃ (z₁ z₂ : ℂ),
    (1 - i) * (-1/2 + (Real.sqrt 3)/2 * i) * (1 + i) = z₁ ∧
    z₁ = -1 + Real.sqrt 3 * i ∧
    (2 + 2*i) / (1 - i)^2 + (Real.sqrt 2 / (1 + i))^2010 = z₂ ∧
    z₂ = -1 - 2*i) :=
by sorry

end NUMINAMATH_CALUDE_complex_calculations_l3474_347465


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3474_347442

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point (given_line : Line) (p : Point) :
  ∃ (result_line : Line),
    pointOnLine p result_line ∧
    parallel result_line given_line ∧
    result_line.a = 2 ∧
    result_line.b = 1 ∧
    result_line.c = -1 :=
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3474_347442


namespace NUMINAMATH_CALUDE_goods_train_length_l3474_347444

/-- Calculate the length of a goods train given the speeds of two trains moving in opposite directions and the time taken for the goods train to pass an observer in the other train. -/
theorem goods_train_length (man_train_speed goods_train_speed : ℝ) (passing_time : ℝ) :
  man_train_speed = 30 →
  goods_train_speed = 82 →
  passing_time = 9 →
  ∃ (length : ℝ), abs (length - 279.99) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_goods_train_length_l3474_347444


namespace NUMINAMATH_CALUDE_two_tangent_lines_l3474_347446

/-- A line that passes through a point and intersects a parabola at only one point. -/
structure TangentLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The point through which the line passes -/
  point : ℝ × ℝ
  /-- The parabola equation in the form y^2 = ax -/
  parabola_coeff : ℝ

/-- The number of lines passing through a given point and tangent to a parabola -/
def count_tangent_lines (point : ℝ × ℝ) (parabola_coeff : ℝ) : ℕ :=
  sorry

/-- Theorem: There are exactly two lines that pass through point M(2, 4) 
    and intersect the parabola y^2 = 8x at only one point -/
theorem two_tangent_lines : count_tangent_lines (2, 4) 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_tangent_lines_l3474_347446


namespace NUMINAMATH_CALUDE_scientific_notation_of_830_billion_l3474_347450

theorem scientific_notation_of_830_billion :
  (830 : ℝ) * (10^9 : ℝ) = 8.3 * (10^11 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_830_billion_l3474_347450


namespace NUMINAMATH_CALUDE_sqrt_180_simplification_l3474_347495

theorem sqrt_180_simplification : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_180_simplification_l3474_347495


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l3474_347447

theorem reciprocal_sum_of_roots (a b c : ℚ) (α β : ℚ) :
  a ≠ 0 →
  (∃ x y : ℚ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y) →
  (∀ x : ℚ, a * x^2 + b * x + c = 0 → (α = 1/x ∨ β = 1/x)) →
  a = 6 ∧ b = 5 ∧ c = 7 →
  α + β = -5/7 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l3474_347447


namespace NUMINAMATH_CALUDE_sheersCost_is_40_l3474_347417

/-- The cost of window treatments for a house with 3 windows, where each window
    requires a pair of sheers and a pair of drapes. -/
def WindowTreatmentsCost (sheersCost : ℚ) : ℚ :=
  3 * (sheersCost + 60)

/-- Theorem stating that the cost of a pair of sheers is $40, given the conditions. -/
theorem sheersCost_is_40 :
  ∃ (sheersCost : ℚ), WindowTreatmentsCost sheersCost = 300 ∧ sheersCost = 40 :=
sorry

end NUMINAMATH_CALUDE_sheersCost_is_40_l3474_347417


namespace NUMINAMATH_CALUDE_select_four_with_girl_l3474_347466

/-- The number of ways to select 4 people from 4 boys and 2 girls with at least one girl -/
def select_with_girl (total : ℕ) (boys : ℕ) (girls : ℕ) (to_select : ℕ) : ℕ :=
  Nat.choose total to_select - Nat.choose boys to_select

theorem select_four_with_girl :
  select_with_girl 6 4 2 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_select_four_with_girl_l3474_347466


namespace NUMINAMATH_CALUDE_negative_expressions_l3474_347435

theorem negative_expressions (x : ℝ) (h : x < 0) : x^3 < 0 ∧ -x^4 < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_expressions_l3474_347435


namespace NUMINAMATH_CALUDE_negation_equivalence_l3474_347405

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3474_347405


namespace NUMINAMATH_CALUDE_pearl_string_value_l3474_347402

/-- Represents a string of pearls with a given middle pearl value and decreasing rates on each side. -/
structure PearlString where
  middleValue : ℕ
  decreaseRate1 : ℕ
  decreaseRate2 : ℕ

/-- Calculates the total value of the pearl string. -/
def totalValue (ps : PearlString) : ℕ :=
  ps.middleValue + 16 * ps.middleValue - 16 * 17 * ps.decreaseRate1 / 2 +
  16 * ps.middleValue - 16 * 17 * ps.decreaseRate2 / 2

/-- Calculates the value of the fourth pearl from the middle on the more expensive side. -/
def fourthPearlValue (ps : PearlString) : ℕ :=
  ps.middleValue - 4 * min ps.decreaseRate1 ps.decreaseRate2

/-- The main theorem stating the conditions and the result to be proven. -/
theorem pearl_string_value (ps : PearlString) :
  ps.decreaseRate1 = 3000 →
  ps.decreaseRate2 = 4500 →
  totalValue ps = 25 * fourthPearlValue ps →
  ps.middleValue = 90000 := by
  sorry

end NUMINAMATH_CALUDE_pearl_string_value_l3474_347402


namespace NUMINAMATH_CALUDE_phone_selling_price_l3474_347437

theorem phone_selling_price 
  (total_phones : ℕ) 
  (initial_investment : ℚ) 
  (profit_ratio : ℚ) :
  total_phones = 200 →
  initial_investment = 3000 →
  profit_ratio = 1/3 →
  (initial_investment + profit_ratio * initial_investment) / total_phones = 20 := by
sorry

end NUMINAMATH_CALUDE_phone_selling_price_l3474_347437


namespace NUMINAMATH_CALUDE_bowling_ball_volume_l3474_347427

/-- The remaining volume of a bowling ball after drilling holes -/
theorem bowling_ball_volume (π : ℝ) (h : π > 0) : 
  let sphere_volume := (4/3) * π * (12^3)
  let small_hole_volume := π * (3/2)^2 * 10
  let large_hole_volume := π * 2^2 * 10
  sphere_volume - (2 * small_hole_volume + large_hole_volume) = 2219 * π := by
  sorry

#check bowling_ball_volume

end NUMINAMATH_CALUDE_bowling_ball_volume_l3474_347427


namespace NUMINAMATH_CALUDE_max_page_number_proof_l3474_347498

def max_page_number (ones : ℕ) (twos : ℕ) : ℕ :=
  let digits : List ℕ := [0, 3, 4, 5, 6, 7, 8, 9]
  199

theorem max_page_number_proof (ones twos : ℕ) :
  ones = 25 → twos = 30 → max_page_number ones twos = 199 := by
  sorry

end NUMINAMATH_CALUDE_max_page_number_proof_l3474_347498


namespace NUMINAMATH_CALUDE_valid_monomial_l3474_347414

def is_valid_monomial (m : ℤ → ℤ → ℤ) : Prop :=
  ∃ (a b : ℕ), ∀ x y, m x y = -2 * x^a * y^b ∧ a + b = 3

theorem valid_monomial : 
  is_valid_monomial (fun x y ↦ -2 * x^2 * y) := by sorry

end NUMINAMATH_CALUDE_valid_monomial_l3474_347414


namespace NUMINAMATH_CALUDE_room_dimension_is_15_l3474_347449

/-- Represents the dimensions and costs related to whitewashing a room --/
structure RoomWhitewash where
  length : ℝ
  width : ℝ
  height : ℝ
  doorLength : ℝ
  doorWidth : ℝ
  windowLength : ℝ
  windowWidth : ℝ
  numWindows : ℕ
  costPerSquareFoot : ℝ
  totalCost : ℝ

/-- Theorem stating that the unknown dimension of the room is 15 feet --/
theorem room_dimension_is_15 (r : RoomWhitewash) 
  (h1 : r.length = 25)
  (h2 : r.height = 12)
  (h3 : r.doorLength = 6)
  (h4 : r.doorWidth = 3)
  (h5 : r.windowLength = 4)
  (h6 : r.windowWidth = 3)
  (h7 : r.numWindows = 3)
  (h8 : r.costPerSquareFoot = 4)
  (h9 : r.totalCost = 3624)
  : r.width = 15 := by
  sorry

end NUMINAMATH_CALUDE_room_dimension_is_15_l3474_347449


namespace NUMINAMATH_CALUDE_red_balls_in_box_l3474_347443

/-- Given a box with an initial number of red balls and a number of red balls added,
    calculate the final number of red balls in the box. -/
def final_red_balls (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The final number of red balls in the box is 7 when starting with 5 and adding 2. -/
theorem red_balls_in_box : final_red_balls 5 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_in_box_l3474_347443


namespace NUMINAMATH_CALUDE_expression_evaluation_l3474_347451

theorem expression_evaluation :
  let a : ℚ := -1/3
  (3*a - 1)^2 + 3*a*(3*a + 2) = 3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3474_347451


namespace NUMINAMATH_CALUDE_order_of_sqrt_differences_l3474_347401

theorem order_of_sqrt_differences :
  let m : ℝ := Real.sqrt 6 - Real.sqrt 5
  let n : ℝ := Real.sqrt 7 - Real.sqrt 6
  let p : ℝ := Real.sqrt 8 - Real.sqrt 7
  m > n ∧ n > p :=
by sorry

end NUMINAMATH_CALUDE_order_of_sqrt_differences_l3474_347401


namespace NUMINAMATH_CALUDE_pen_price_problem_l3474_347499

theorem pen_price_problem (price : ℝ) (quantity : ℝ) : 
  (price * quantity = (price - 1) * (quantity + 100)) →
  (price * quantity = (price + 2) * (quantity - 100)) →
  price = 4 := by
sorry

end NUMINAMATH_CALUDE_pen_price_problem_l3474_347499


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_min_l3474_347468

/-- Given a parabola y = ax^2 + bx + c with positive integer coefficients that intersects
    the x-axis at two distinct points within distance 1 of the origin, 
    the sum of its coefficients is at least 11. -/
theorem parabola_coefficient_sum_min (a b c : ℕ+) 
  (h_distinct : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0)
  (h_distance : ∀ x : ℝ, a * x^2 + b * x + c = 0 → |x| < 1) :
  a + b + c ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_min_l3474_347468


namespace NUMINAMATH_CALUDE_root_equation_q_value_l3474_347416

theorem root_equation_q_value (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 2/b)^2 - p*(a + 2/b) + q = 0) →
  ((b + 2/a)^2 - p*(b + 2/a) + q = 0) →
  (q = 25/3) := by
sorry

end NUMINAMATH_CALUDE_root_equation_q_value_l3474_347416


namespace NUMINAMATH_CALUDE_point_b_value_l3474_347441

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- The distance between two points on a number line -/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_b_value (a b : Point) :
  a.value = 1 → distance a b = 3 → b.value = 4 ∨ b.value = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_b_value_l3474_347441


namespace NUMINAMATH_CALUDE_min_club_members_l3474_347472

theorem min_club_members (N : ℕ) : N < 80 ∧ 
  ((N - 5) % 8 = 0 ∨ (N - 5) % 7 = 0) ∧ 
  N % 9 = 7 → 
  N ≥ 61 :=
by sorry

end NUMINAMATH_CALUDE_min_club_members_l3474_347472


namespace NUMINAMATH_CALUDE_candy_cost_l3474_347489

theorem candy_cost (initial_amount pencil_cost remaining_amount : ℕ) 
  (h1 : initial_amount = 43)
  (h2 : pencil_cost = 20)
  (h3 : remaining_amount = 18) :
  initial_amount - pencil_cost - remaining_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_l3474_347489


namespace NUMINAMATH_CALUDE_lily_total_books_l3474_347484

def mike_books_tuesday : ℕ := 45
def corey_books_tuesday : ℕ := 2 * mike_books_tuesday
def mike_gave_to_lily : ℕ := 10
def corey_gave_to_lily : ℕ := mike_gave_to_lily + 15

theorem lily_total_books : mike_gave_to_lily + corey_gave_to_lily = 35 :=
by sorry

end NUMINAMATH_CALUDE_lily_total_books_l3474_347484


namespace NUMINAMATH_CALUDE_range_of_a_l3474_347432

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3474_347432


namespace NUMINAMATH_CALUDE_g_of_5_equals_18_l3474_347479

/-- Given a function g where g(x) = 4x - 2 for all x, prove that g(5) = 18 -/
theorem g_of_5_equals_18 (g : ℝ → ℝ) (h : ∀ x, g x = 4 * x - 2) : g 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_equals_18_l3474_347479


namespace NUMINAMATH_CALUDE_mass_of_Al2O3_solution_l3474_347407

-- Define the atomic masses
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_O : ℝ := 16.00

-- Define the volume and concentration of the solution
def volume : ℝ := 2.5
def concentration : ℝ := 4

-- Define the molecular weight of Al2O3
def molecular_weight_Al2O3 : ℝ := 2 * atomic_mass_Al + 3 * atomic_mass_O

-- State the theorem
theorem mass_of_Al2O3_solution :
  let moles : ℝ := volume * concentration
  let mass : ℝ := moles * molecular_weight_Al2O3
  mass = 1019.6 := by sorry

end NUMINAMATH_CALUDE_mass_of_Al2O3_solution_l3474_347407


namespace NUMINAMATH_CALUDE_vanessa_camera_pictures_l3474_347454

/-- The number of pictures Vanessa uploaded from her camera -/
def camera_pictures (phone_pictures album_count pictures_per_album : ℕ) : ℕ :=
  album_count * pictures_per_album - phone_pictures

/-- Proof that Vanessa uploaded 7 pictures from her camera -/
theorem vanessa_camera_pictures :
  camera_pictures 23 5 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_camera_pictures_l3474_347454


namespace NUMINAMATH_CALUDE_pen_cost_proof_l3474_347410

theorem pen_cost_proof (total_students : ℕ) (total_cost : ℚ) : ∃ (buyers pens_per_student cost_per_pen : ℕ),
  total_students = 40 ∧
  total_cost = 2091 / 100 ∧
  buyers > total_students / 2 ∧
  buyers ≤ total_students ∧
  pens_per_student % 2 = 1 ∧
  pens_per_student > 1 ∧
  Nat.Prime cost_per_pen ∧
  buyers * pens_per_student * cost_per_pen = 2091 ∧
  cost_per_pen = 47 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_proof_l3474_347410


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l3474_347453

/-- Given a shopkeeper selling cloth with a total selling price, loss per metre, and cost price per metre,
    prove that the number of metres sold is as calculated. -/
theorem cloth_sale_calculation
  (total_selling_price : ℕ)
  (loss_per_metre : ℕ)
  (cost_price_per_metre : ℕ)
  (h1 : total_selling_price = 36000)
  (h2 : loss_per_metre = 10)
  (h3 : cost_price_per_metre = 70) :
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 600 := by
  sorry

#check cloth_sale_calculation

end NUMINAMATH_CALUDE_cloth_sale_calculation_l3474_347453


namespace NUMINAMATH_CALUDE_cubic_resonance_intervals_sqrt_resonance_interval_l3474_347404

-- Definition of a resonance interval
def is_resonance_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a ≤ b ∧
  Monotone f ∧
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y)

-- Theorem for the cubic function
theorem cubic_resonance_intervals :
  (is_resonance_interval (fun x ↦ x^3) (-1) 0) ∧
  (is_resonance_interval (fun x ↦ x^3) (-1) 1) ∧
  (is_resonance_interval (fun x ↦ x^3) 0 1) :=
sorry

-- Theorem for the square root function
theorem sqrt_resonance_interval (k : ℝ) :
  (∃ a b, is_resonance_interval (fun x ↦ Real.sqrt (x + 1) - k) a b) ↔
  (1 ≤ k ∧ k < 5/4) :=
sorry

end NUMINAMATH_CALUDE_cubic_resonance_intervals_sqrt_resonance_interval_l3474_347404


namespace NUMINAMATH_CALUDE_dentist_bill_calculation_dentist_cleaning_cost_l3474_347492

theorem dentist_bill_calculation (filling_cost : ℕ) (extraction_cost : ℕ) : ℕ :=
  let total_bill := 5 * filling_cost
  let cleaning_cost := total_bill - (2 * filling_cost + extraction_cost)
  cleaning_cost

theorem dentist_cleaning_cost : dentist_bill_calculation 120 290 = 70 := by
  sorry

end NUMINAMATH_CALUDE_dentist_bill_calculation_dentist_cleaning_cost_l3474_347492


namespace NUMINAMATH_CALUDE_valid_numbers_l3474_347439

def is_valid_number (N : ℕ) : Prop :=
  ∃ (a b k : ℕ),
    10 ≤ a ∧ a ≤ 99 ∧
    0 ≤ b ∧ b < 10^k ∧
    N = 10^k * a + b ∧
    Odd N ∧
    10^k * a + b = 149 * b

theorem valid_numbers :
  ∀ N : ℕ, is_valid_number N → (N = 745 ∨ N = 3725) :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l3474_347439


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_quadratic_l3474_347488

theorem factorization_cubic_minus_quadratic (x y : ℝ) :
  y^3 - 4*x^2*y = y*(y+2*x)*(y-2*x) := by sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_quadratic_l3474_347488


namespace NUMINAMATH_CALUDE_extreme_points_range_l3474_347463

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - a*x + a + 1) * Real.exp x

theorem extreme_points_range (a : ℝ) (x₁ x₂ : ℝ) (h₁ : a > 0) (h₂ : x₁ < x₂)
  (h₃ : ∀ x, f a x = 0 → x = x₁ ∨ x = x₂)
  (h₄ : ∀ m : ℝ, m * x₁ - f a x₂ / Real.exp x₁ > 0) :
  ∀ m : ℝ, m ≥ 2 ↔ m * x₁ - f a x₂ / Real.exp x₁ > 0 :=
by sorry

end NUMINAMATH_CALUDE_extreme_points_range_l3474_347463


namespace NUMINAMATH_CALUDE_matrix_equality_l3474_347422

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = 2 * A * B)
  (h2 : A * B = !![5, 1; -2, 4]) :
  B * A = !![10, 2; -4, 8] := by
sorry

end NUMINAMATH_CALUDE_matrix_equality_l3474_347422


namespace NUMINAMATH_CALUDE_booklet_word_count_l3474_347412

theorem booklet_word_count (total_pages : Nat) (max_words_per_page : Nat) (remainder : Nat) (modulus : Nat) : 
  total_pages = 154 →
  max_words_per_page = 120 →
  remainder = 207 →
  modulus = 221 →
  ∃ (words_per_page : Nat),
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % modulus = remainder ∧
    words_per_page = 100 := by
  sorry

end NUMINAMATH_CALUDE_booklet_word_count_l3474_347412


namespace NUMINAMATH_CALUDE_topsoil_cost_calculation_l3474_347452

/-- The cost of topsoil in dollars per cubic foot -/
def topsoil_cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The amount of topsoil in cubic yards -/
def topsoil_amount : ℝ := 8

/-- The total cost of topsoil in dollars -/
def total_cost : ℝ := topsoil_amount * cubic_yards_to_cubic_feet * topsoil_cost_per_cubic_foot

theorem topsoil_cost_calculation : total_cost = 1728 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_calculation_l3474_347452


namespace NUMINAMATH_CALUDE_swimmer_speed_l3474_347403

/-- A swimmer's speed in still water, given stream conditions -/
theorem swimmer_speed (v s : ℝ) (h1 : s = 1.5) (h2 : (v - s)⁻¹ = 2 * (v + s)⁻¹) : v = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_l3474_347403


namespace NUMINAMATH_CALUDE_system_solution_l3474_347413

theorem system_solution : 
  ∃ (x y : ℚ), 4 * x - 35 * y = -1 ∧ 3 * y - x = 5 ∧ x = -172/23 ∧ y = -19/23 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3474_347413


namespace NUMINAMATH_CALUDE_prime_factorization_of_large_number_l3474_347411

theorem prime_factorization_of_large_number :
  1007021035035021007001 = 7^7 * 11^7 * 13^7 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_of_large_number_l3474_347411


namespace NUMINAMATH_CALUDE_angle_trisector_theorem_l3474_347496

/-- 
Given a triangle ABC with angle γ = ∠ACB, if the trisectors of γ divide 
the opposite side AB into segments d, e, f, then cos²(γ/3) = ((d+e)(e+f))/(4df)
-/
theorem angle_trisector_theorem (d e f : ℝ) (γ : ℝ) 
  (h1 : d > 0) (h2 : e > 0) (h3 : f > 0) (h4 : γ > 0) (h5 : γ < π) :
  (Real.cos (γ / 3))^2 = ((d + e) * (e + f)) / (4 * d * f) :=
sorry

end NUMINAMATH_CALUDE_angle_trisector_theorem_l3474_347496


namespace NUMINAMATH_CALUDE_distance_calculation_l3474_347461

/-- The speed of light in km/s -/
def speed_of_light : ℝ := 3 * 10^5

/-- The time it takes for light to reach Earth from Proxima Centauri in years -/
def travel_time : ℝ := 4

/-- The number of seconds in a year -/
def seconds_per_year : ℝ := 3 * 10^7

/-- The distance from Proxima Centauri to Earth in km -/
def distance_to_proxima_centauri : ℝ := speed_of_light * travel_time * seconds_per_year

theorem distance_calculation :
  distance_to_proxima_centauri = 3.6 * 10^13 := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l3474_347461


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3474_347409

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) :
  Real.sqrt (a * b) < (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3474_347409


namespace NUMINAMATH_CALUDE_test_maximum_marks_l3474_347481

theorem test_maximum_marks :
  let passing_percentage : ℚ := 60 / 100
  let student_score : ℕ := 80
  let marks_needed_to_pass : ℕ := 100
  let maximum_marks : ℕ := 300
  passing_percentage * maximum_marks = student_score + marks_needed_to_pass →
  maximum_marks = 300 := by
sorry

end NUMINAMATH_CALUDE_test_maximum_marks_l3474_347481


namespace NUMINAMATH_CALUDE_opposite_silver_is_orange_l3474_347473

/-- Represents the colors of the cube faces -/
inductive Color
  | Blue
  | Orange
  | Black
  | Yellow
  | Silver
  | Violet

/-- Represents the positions of the cube faces -/
inductive Position
  | Top
  | Bottom
  | Front
  | Back
  | Left
  | Right

/-- Represents a view of the cube -/
structure View where
  top : Color
  front : Color
  right : Color

/-- The cube with its colored faces -/
structure Cube where
  faces : Position → Color

def first_view : View :=
  { top := Color.Blue, front := Color.Yellow, right := Color.Violet }

def second_view : View :=
  { top := Color.Blue, front := Color.Silver, right := Color.Violet }

def third_view : View :=
  { top := Color.Blue, front := Color.Black, right := Color.Violet }

theorem opposite_silver_is_orange (c : Cube) :
  (c.faces Position.Front = Color.Silver) →
  (c.faces Position.Top = Color.Blue) →
  (c.faces Position.Right = Color.Violet) →
  (c.faces Position.Back = Color.Orange) :=
by sorry

end NUMINAMATH_CALUDE_opposite_silver_is_orange_l3474_347473


namespace NUMINAMATH_CALUDE_circumscribed_iff_similar_when_moved_l3474_347420

/-- A polygon represented by its vertices -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- A function to check if a polygon is convex -/
def isConvex (p : Polygon) : Prop :=
  sorry

/-- A function to move all sides of a polygon outward by a distance -/
def moveOutward (p : Polygon) (distance : ℝ) : Polygon :=
  sorry

/-- A function to check if two polygons are similar -/
def areSimilar (p1 p2 : Polygon) : Prop :=
  sorry

/-- A function to check if a polygon is circumscribed -/
def isCircumscribed (p : Polygon) : Prop :=
  sorry

/-- Theorem: A convex polygon is circumscribed if and only if 
    moving all its sides outward by a distance of 1 results 
    in a polygon similar to the original one -/
theorem circumscribed_iff_similar_when_moved (p : Polygon) :
  isConvex p →
  isCircumscribed p ↔ areSimilar p (moveOutward p 1) :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_iff_similar_when_moved_l3474_347420


namespace NUMINAMATH_CALUDE_davids_physics_marks_l3474_347438

theorem davids_physics_marks :
  let english_marks : ℕ := 51
  let math_marks : ℕ := 65
  let chemistry_marks : ℕ := 67
  let biology_marks : ℕ := 85
  let average_marks : ℕ := 70
  let total_subjects : ℕ := 5

  let total_marks : ℕ := average_marks * total_subjects
  let known_marks : ℕ := english_marks + math_marks + chemistry_marks + biology_marks
  let physics_marks : ℕ := total_marks - known_marks

  physics_marks = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l3474_347438


namespace NUMINAMATH_CALUDE_intersection_A_B_l3474_347494

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x : ℝ | x^2 - 2*x ≤ 0}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3474_347494


namespace NUMINAMATH_CALUDE_simplest_quadratic_root_l3474_347457

theorem simplest_quadratic_root (x : ℝ) : 
  (∃ (k : ℚ), Real.sqrt (x + 1) = k * Real.sqrt (5 / 2)) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_simplest_quadratic_root_l3474_347457


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3474_347470

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1}

theorem complement_of_A_in_U : (U \ A) = {0, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3474_347470


namespace NUMINAMATH_CALUDE_compute_expression_l3474_347482

theorem compute_expression : 25 * (216 / 3 + 49 / 7 + 16 / 25 + 2) = 2041 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3474_347482


namespace NUMINAMATH_CALUDE_blake_poured_out_02_gallons_l3474_347487

/-- The amount of water Blake poured out, given initial and remaining amounts -/
def water_poured_out (initial : Real) (remaining : Real) : Real :=
  initial - remaining

/-- Theorem: Blake poured out 0.2 gallons of water -/
theorem blake_poured_out_02_gallons :
  let initial := 0.8
  let remaining := 0.6
  water_poured_out initial remaining = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_blake_poured_out_02_gallons_l3474_347487


namespace NUMINAMATH_CALUDE_inequality_proof_l3474_347415

theorem inequality_proof (x y : ℝ) (h : x > y) : -2 * x < -2 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3474_347415


namespace NUMINAMATH_CALUDE_prime_even_intersection_l3474_347429

def P : Set ℕ := {n : ℕ | Nat.Prime n}
def Q : Set ℕ := {n : ℕ | Even n}

theorem prime_even_intersection : P ∩ Q = {2} := by
  sorry

end NUMINAMATH_CALUDE_prime_even_intersection_l3474_347429


namespace NUMINAMATH_CALUDE_tinas_career_win_loss_difference_l3474_347419

/-- Represents Tina's boxing career -/
structure BoxingCareer where
  initial_wins : ℕ
  additional_wins_before_first_loss : ℕ
  wins_doubled : Bool

/-- Calculates the total number of wins in Tina's career -/
def total_wins (career : BoxingCareer) : ℕ :=
  let wins_before_doubling := career.initial_wins + career.additional_wins_before_first_loss
  if career.wins_doubled then
    2 * wins_before_doubling
  else
    wins_before_doubling

/-- Calculates the total number of losses in Tina's career -/
def total_losses (career : BoxingCareer) : ℕ :=
  if career.wins_doubled then 2 else 1

/-- Theorem stating the difference between wins and losses in Tina's career -/
theorem tinas_career_win_loss_difference :
  ∀ (career : BoxingCareer),
    career.initial_wins = 10 →
    career.additional_wins_before_first_loss = 5 →
    career.wins_doubled = true →
    total_wins career - total_losses career = 28 := by
  sorry

end NUMINAMATH_CALUDE_tinas_career_win_loss_difference_l3474_347419


namespace NUMINAMATH_CALUDE_not_right_triangle_l3474_347469

theorem not_right_triangle (a b c : ℝ) : 
  (a = 3 ∧ b = 4 ∧ c = 5) ∨ 
  (a = 1 ∧ b = Real.sqrt 3 ∧ c = 2) ∨ 
  (a = Real.sqrt 11 ∧ b = 2 ∧ c = 4) ∨ 
  (a^2 = (c+b)*(c-b)) →
  (¬(a^2 + b^2 = c^2) ↔ a = Real.sqrt 11 ∧ b = 2 ∧ c = 4) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l3474_347469


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_squares_possible_l3474_347474

/-- Given three real numbers that form a geometric sequence and are not all equal,
    it's possible for their squares to form an arithmetic sequence. -/
theorem geometric_to_arithmetic_squares_possible (a b c : ℝ) :
  (∃ r : ℝ, r ≠ 1 ∧ b = a * r ∧ c = b * r) →  -- Geometric sequence condition
  (a ≠ b ∨ b ≠ c) →                          -- Not all equal condition
  ∃ x y z : ℝ, x = a^2 ∧ y = b^2 ∧ z = c^2 ∧  -- Squares of a, b, c
            y - x = z - y                    -- Arithmetic sequence condition
    := by sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_squares_possible_l3474_347474


namespace NUMINAMATH_CALUDE_annies_initial_apples_l3474_347459

theorem annies_initial_apples (initial_apples total_apples apples_from_nathan : ℕ) :
  total_apples = initial_apples + apples_from_nathan →
  apples_from_nathan = 6 →
  total_apples = 12 →
  initial_apples = 6 := by
sorry

end NUMINAMATH_CALUDE_annies_initial_apples_l3474_347459


namespace NUMINAMATH_CALUDE_cubic_odd_and_increasing_l3474_347436

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem cubic_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_odd_and_increasing_l3474_347436


namespace NUMINAMATH_CALUDE_base8_square_unique_l3474_347418

/-- Converts a base-10 number to base-8 --/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 8) :: aux (m / 8)
  aux n |>.reverse

/-- Checks if a list contains each number from 0 to 7 exactly once --/
def containsEachDigitOnce (l : List ℕ) : Prop :=
  ∀ d, d ∈ Finset.range 8 → (l.count d = 1)

/-- The main theorem --/
theorem base8_square_unique : 
  ∃! n : ℕ, 
    (toBase8 n).length = 3 ∧ 
    containsEachDigitOnce (toBase8 n) ∧
    containsEachDigitOnce (toBase8 (n * n)) ∧
    n = 256 := by sorry

end NUMINAMATH_CALUDE_base8_square_unique_l3474_347418


namespace NUMINAMATH_CALUDE_bus_ride_time_l3474_347485

def total_trip_time : ℕ := 8 * 60  -- 8 hours in minutes
def walk_time : ℕ := 15
def train_ride_time : ℕ := 6 * 60  -- 6 hours in minutes

def wait_time : ℕ := 2 * walk_time

def time_without_bus : ℕ := train_ride_time + walk_time + wait_time

theorem bus_ride_time : total_trip_time - time_without_bus = 75 := by
  sorry

end NUMINAMATH_CALUDE_bus_ride_time_l3474_347485


namespace NUMINAMATH_CALUDE_pie_sugar_percentage_l3474_347428

/-- Given a pie weighing 200 grams with 50 grams of sugar, 
    prove that 75% of the pie is not sugar. -/
theorem pie_sugar_percentage 
  (total_weight : ℝ) 
  (sugar_weight : ℝ) 
  (h1 : total_weight = 200) 
  (h2 : sugar_weight = 50) : 
  (total_weight - sugar_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_pie_sugar_percentage_l3474_347428


namespace NUMINAMATH_CALUDE_olivine_stones_difference_l3474_347460

theorem olivine_stones_difference (agate_stones olivine_stones diamond_stones : ℕ) : 
  agate_stones = 30 →
  olivine_stones > agate_stones →
  diamond_stones = olivine_stones + 11 →
  agate_stones + olivine_stones + diamond_stones = 111 →
  olivine_stones = agate_stones + 5 := by
sorry

end NUMINAMATH_CALUDE_olivine_stones_difference_l3474_347460


namespace NUMINAMATH_CALUDE_sine_equation_solution_l3474_347408

theorem sine_equation_solution (x : ℝ) (h1 : 3 * Real.sin (2 * x) = 2 * Real.sin x) 
  (h2 : 0 < x ∧ x < π) : x = Real.arccos (1/3) := by
  sorry

end NUMINAMATH_CALUDE_sine_equation_solution_l3474_347408


namespace NUMINAMATH_CALUDE_area_of_annular_region_area_of_specific_annular_region_l3474_347406

/-- The area of an annular region between two concentric circles -/
theorem area_of_annular_region (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > r₁) : 
  π * r₂^2 - π * r₁^2 = π * (r₂^2 - r₁^2) :=
by sorry

/-- The area of the annular region between two concentric circles with radii 4 and 7 is 33π -/
theorem area_of_specific_annular_region : 
  π * 7^2 - π * 4^2 = 33 * π :=
by sorry

end NUMINAMATH_CALUDE_area_of_annular_region_area_of_specific_annular_region_l3474_347406


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l3474_347475

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (t x y : ℝ) : Prop := x = 1 + (2/Real.sqrt 5)*t ∧ y = 1 + (1/Real.sqrt 5)*t

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    line_l t₁ A.1 A.2 ∧ curve_C A.1 A.2 ∧
    line_l t₂ B.1 B.2 ∧ curve_C B.1 B.2 ∧
    t₁ ≠ t₂

-- State the theorem
theorem intersection_distance_sum :
  ∀ A B : ℝ × ℝ,
  intersection_points A B →
  Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
  Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
  4 * Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l3474_347475


namespace NUMINAMATH_CALUDE_work_problem_solution_l3474_347421

def work_problem (a_days b_days remaining_days : ℚ) : Prop :=
  let a_rate : ℚ := 1 / a_days
  let b_rate : ℚ := 1 / b_days
  let combined_rate : ℚ := a_rate + b_rate
  let x : ℚ := 2  -- Days A and B worked together
  combined_rate * x + b_rate * remaining_days = 1

theorem work_problem_solution :
  work_problem 4 8 2 = true :=
sorry

end NUMINAMATH_CALUDE_work_problem_solution_l3474_347421


namespace NUMINAMATH_CALUDE_inverse_proportionality_l3474_347400

theorem inverse_proportionality (x y : ℝ) (P : ℝ) : 
  (x + y = 30) → (x - y = 12) → (x * y = P) → (3 * (P / 3) = 63) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l3474_347400


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3474_347497

theorem quadratic_inequality_solution_set (x : ℝ) :
  3 * x^2 + 7 * x + 2 < 0 ↔ -1 < x ∧ x < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3474_347497


namespace NUMINAMATH_CALUDE_binomial_expansions_l3474_347455

theorem binomial_expansions (a b : ℝ) : 
  ((a + b)^3 = a^3 + 3*a^2*b + 3*a*b^2 + b^3) ∧ 
  ((a + b)^4 = a^4 + 4*a^3*b + 6*a^2*b^2 + 4*a*b^3 + b^4) ∧
  ((a + b)^5 = a^5 + 5*a^4*b + 10*a^3*b^2 + 10*a^2*b^3 + 5*a*b^4 + b^5) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansions_l3474_347455


namespace NUMINAMATH_CALUDE_prime_sum_squares_l3474_347486

theorem prime_sum_squares (p q m : ℕ) : 
  p.Prime → q.Prime → p ≠ q →
  p^2 - 2001*p + m = 0 →
  q^2 - 2001*q + m = 0 →
  p^2 + q^2 = 3996005 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l3474_347486


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l3474_347476

theorem correct_quotient_proof (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 12 = 63) : D / 21 = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l3474_347476


namespace NUMINAMATH_CALUDE_exponential_regression_model_l3474_347464

/-- Given a model y = ce^(kx) and its transformed linear equation z = 0.3x + 4 where z = ln y,
    prove that c = e^4 and k = 0.3 -/
theorem exponential_regression_model (c k : ℝ) : 
  (∀ x y : ℝ, y = c * Real.exp (k * x)) →
  (∀ x z : ℝ, z = 0.3 * x + 4) →
  (∀ y : ℝ, z = Real.log y) →
  c = Real.exp 4 ∧ k = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_regression_model_l3474_347464


namespace NUMINAMATH_CALUDE_book_cost_l3474_347462

theorem book_cost (initial_money : ℕ) (notebooks : ℕ) (notebook_cost : ℕ) (books : ℕ) (money_left : ℕ) : 
  initial_money = 56 →
  notebooks = 7 →
  notebook_cost = 4 →
  books = 2 →
  money_left = 14 →
  (initial_money - money_left - notebooks * notebook_cost) / books = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l3474_347462


namespace NUMINAMATH_CALUDE_max_area_enclosure_l3474_347483

/-- Represents a rectangular enclosure. -/
structure Enclosure where
  length : ℝ
  width : ℝ

/-- The perimeter of the enclosure is exactly 420 feet. -/
def perimeterConstraint (e : Enclosure) : Prop :=
  2 * e.length + 2 * e.width = 420

/-- The length of the enclosure is at least 100 feet. -/
def lengthConstraint (e : Enclosure) : Prop :=
  e.length ≥ 100

/-- The width of the enclosure is at least 60 feet. -/
def widthConstraint (e : Enclosure) : Prop :=
  e.width ≥ 60

/-- The area of the enclosure. -/
def area (e : Enclosure) : ℝ :=
  e.length * e.width

/-- The theorem stating that the maximum area is achieved when length = width = 105 feet. -/
theorem max_area_enclosure :
  ∀ e : Enclosure,
    perimeterConstraint e → lengthConstraint e → widthConstraint e →
    area e ≤ 11025 ∧
    (area e = 11025 ↔ e.length = 105 ∧ e.width = 105) :=
by sorry

end NUMINAMATH_CALUDE_max_area_enclosure_l3474_347483


namespace NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l3474_347456

theorem condition_p_sufficient_not_necessary :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2 ∧ x * y > 1) ∧
  (∃ x y : ℝ, x + y > 2 ∧ x * y > 1 ∧ ¬(x > 1 ∧ y > 1)) := by
  sorry

end NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l3474_347456


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3474_347426

theorem cubic_root_sum (a b c : ℝ) : 
  (a^3 = a + 1) → (b^3 = b + 1) → (c^3 = c + 1) →
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = -9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3474_347426


namespace NUMINAMATH_CALUDE_samantha_routes_l3474_347448

/-- Represents a location on a grid --/
structure Location :=
  (x : ℤ) (y : ℤ)

/-- Calculates the number of shortest paths between two locations --/
def num_shortest_paths (start finish : Location) : ℕ :=
  sorry

/-- Samantha's home location relative to the southwest corner of City Park --/
def home : Location :=
  { x := -1, y := -3 }

/-- Southwest corner of City Park --/
def park_sw : Location :=
  { x := 0, y := 0 }

/-- Northeast corner of City Park --/
def park_ne : Location :=
  { x := 0, y := 0 }

/-- Samantha's school location relative to the northeast corner of City Park --/
def school : Location :=
  { x := 3, y := 1 }

/-- Library location relative to the school --/
def library : Location :=
  { x := 2, y := 1 }

/-- Total number of routes Samantha can take --/
def total_routes : ℕ :=
  (num_shortest_paths home park_sw) *
  (num_shortest_paths park_ne school) *
  (num_shortest_paths school library)

theorem samantha_routes :
  total_routes = 48 :=
sorry

end NUMINAMATH_CALUDE_samantha_routes_l3474_347448


namespace NUMINAMATH_CALUDE_landscape_breadth_l3474_347477

/-- Proves that the breadth of a rectangular landscape is 480 meters given the specified conditions -/
theorem landscape_breadth :
  ∀ (length breadth : ℝ),
  breadth = 8 * length →
  3200 = (1 / 9) * (length * breadth) →
  breadth = 480 :=
by
  sorry

end NUMINAMATH_CALUDE_landscape_breadth_l3474_347477


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3474_347467

theorem quadratic_equation_solution :
  ∃ (x1 x2 : ℝ), 
    x1 > 0 ∧ x2 > 0 ∧
    (1/2 * (4 * x1^2 - 1) = (x1^2 - 75*x1 - 15) * (x1^2 + 50*x1 + 10)) ∧
    (1/2 * (4 * x2^2 - 1) = (x2^2 - 75*x2 - 15) * (x2^2 + 50*x2 + 10)) ∧
    x1 = (75 + Real.sqrt 5773) / 2 ∧
    x2 = (-50 + Real.sqrt 2356) / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3474_347467
