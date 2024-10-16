import Mathlib

namespace NUMINAMATH_CALUDE_symmetric_quadratic_comparison_l375_37575

/-- A quadratic function that opens upward and is symmetric about x = 2013 -/
class SymmetricQuadratic (f : ℝ → ℝ) :=
  (opens_upward : ∃ (a b c : ℝ), a > 0 ∧ ∀ x, f x = a * x^2 + b * x + c)
  (symmetric : ∀ x, f (2013 + x) = f (2013 - x))

/-- Theorem: For a symmetric quadratic function f that opens upward,
    f(2011) is greater than f(2014) -/
theorem symmetric_quadratic_comparison
  (f : ℝ → ℝ) [SymmetricQuadratic f] :
  f 2011 > f 2014 :=
sorry

end NUMINAMATH_CALUDE_symmetric_quadratic_comparison_l375_37575


namespace NUMINAMATH_CALUDE_problem_solution_l375_37551

def f (x : ℝ) := |2*x - 7| + 1

def g (x : ℝ) := f x - 2*|x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x ≤ x ↔ (8/3 ≤ x ∧ x ≤ 6)) ∧
  (∀ x : ℝ, g x ≥ -4) ∧
  (∀ a : ℝ, (∃ x : ℝ, g x ≤ a) ↔ a ≥ -4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l375_37551


namespace NUMINAMATH_CALUDE_four_point_circle_theorem_l375_37540

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define what it means for three points to be collinear
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

-- Define what it means for a point to be on a circle
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define what it means for a point to be inside a circle
def insideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

-- The main theorem
theorem four_point_circle_theorem (A B C D : Point) 
  (h : ¬collinear A B C ∧ ¬collinear A B D ∧ ¬collinear A C D ∧ ¬collinear B C D) :
  ∃ (c : Circle), 
    (onCircle A c ∧ onCircle B c ∧ onCircle C c ∧ (onCircle D c ∨ insideCircle D c)) ∨
    (onCircle A c ∧ onCircle B c ∧ onCircle D c ∧ (onCircle C c ∨ insideCircle C c)) ∨
    (onCircle A c ∧ onCircle C c ∧ onCircle D c ∧ (onCircle B c ∨ insideCircle B c)) ∨
    (onCircle B c ∧ onCircle C c ∧ onCircle D c ∧ (onCircle A c ∨ insideCircle A c)) :=
  sorry

end NUMINAMATH_CALUDE_four_point_circle_theorem_l375_37540


namespace NUMINAMATH_CALUDE_remainder_problem_l375_37508

theorem remainder_problem (N : ℕ) (R : ℕ) (h1 : R < 100) (h2 : ∃ k : ℕ, N = 100 * k + R) (h3 : ∃ m : ℕ, N = R * m + 1) : R = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l375_37508


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_set_l375_37595

-- Define the cubic polynomial function
def f (x : ℝ) : ℝ := -3 * x^3 + 5 * x^2 - 2 * x + 1

-- State the theorem
theorem cubic_inequality_solution_set :
  ∀ x : ℝ, f x > 0 ↔ (x > -1 ∧ x < 1/3) ∨ x > 1 :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_set_l375_37595


namespace NUMINAMATH_CALUDE_chongqing_population_scientific_notation_l375_37573

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The population of Chongqing at the end of 2022 -/
def chongqing_population : ℕ := 32000000

/-- Converts a natural number to its scientific notation representation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem chongqing_population_scientific_notation :
  to_scientific_notation chongqing_population =
    ScientificNotation.mk 3.2 7 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_chongqing_population_scientific_notation_l375_37573


namespace NUMINAMATH_CALUDE_least_cans_required_l375_37500

def maaza_volume : ℕ := 20
def pepsi_volume : ℕ := 144
def sprite_volume : ℕ := 368

theorem least_cans_required :
  let gcd := Nat.gcd (Nat.gcd maaza_volume pepsi_volume) sprite_volume
  maaza_volume / gcd + pepsi_volume / gcd + sprite_volume / gcd = 133 := by
  sorry

end NUMINAMATH_CALUDE_least_cans_required_l375_37500


namespace NUMINAMATH_CALUDE_hash_2_3_4_l375_37566

/-- The # operation defined on three real numbers -/
def hash (a b c : ℝ) : ℝ := (b + 1)^2 - 4*a*(c - 1)

/-- Theorem stating that #(2, 3, 4) = -8 -/
theorem hash_2_3_4 : hash 2 3 4 = -8 := by
  sorry

end NUMINAMATH_CALUDE_hash_2_3_4_l375_37566


namespace NUMINAMATH_CALUDE_mMobile_first_two_lines_cost_l375_37589

/-- The cost of a mobile phone plan for a family of 5 -/
structure MobilePlan where
  firstTwoLines : ℕ  -- Cost for first two lines
  additionalLine : ℕ  -- Cost for each additional line

/-- Calculate the total cost for 5 lines -/
def totalCost (plan : MobilePlan) : ℕ :=
  plan.firstTwoLines + 3 * plan.additionalLine

theorem mMobile_first_two_lines_cost : 
  ∃ (mMobile : MobilePlan),
    mMobile.additionalLine = 14 ∧
    ∃ (tMobile : MobilePlan),
      tMobile.firstTwoLines = 50 ∧
      tMobile.additionalLine = 16 ∧
      totalCost tMobile - totalCost mMobile = 11 ∧
      mMobile.firstTwoLines = 45 := by
  sorry

end NUMINAMATH_CALUDE_mMobile_first_two_lines_cost_l375_37589


namespace NUMINAMATH_CALUDE_incorrect_locus_proof_l375_37586

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 25

/-- The set of points satisfying the circle equation -/
def circle_set : Set (ℝ × ℝ) := {p | circle_equation p.1 p.2}

/-- The statement to be proven false -/
def incorrect_statement : Prop :=
  (∀ p : ℝ × ℝ, ¬(circle_equation p.1 p.2) → p ∉ circle_set) →
  (circle_set = {p : ℝ × ℝ | (p.1 - 0)^2 + (p.2 - 0)^2 = 5^2})

theorem incorrect_locus_proof : ¬incorrect_statement := by
  sorry

end NUMINAMATH_CALUDE_incorrect_locus_proof_l375_37586


namespace NUMINAMATH_CALUDE_watermelon_weight_calculation_l375_37584

/-- The weight of a single watermelon in pounds -/
def watermelon_weight : ℝ := 23

/-- The price per pound of watermelon in dollars -/
def price_per_pound : ℝ := 2

/-- The number of watermelons sold -/
def num_watermelons : ℕ := 18

/-- The total revenue from selling the watermelons in dollars -/
def total_revenue : ℝ := 828

theorem watermelon_weight_calculation :
  watermelon_weight = total_revenue / (price_per_pound * num_watermelons) :=
by sorry

end NUMINAMATH_CALUDE_watermelon_weight_calculation_l375_37584


namespace NUMINAMATH_CALUDE_fraction_sum_l375_37531

theorem fraction_sum : (3 : ℚ) / 8 + (9 : ℚ) / 14 = (33 : ℚ) / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l375_37531


namespace NUMINAMATH_CALUDE_five_heads_in_nine_flips_l375_37547

/-- The probability of getting exactly k heads when flipping n fair coins -/
def coinFlipProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

/-- Theorem: The probability of getting exactly 5 heads when flipping 9 fair coins is 63/256 -/
theorem five_heads_in_nine_flips :
  coinFlipProbability 9 5 = 63 / 256 := by
  sorry

end NUMINAMATH_CALUDE_five_heads_in_nine_flips_l375_37547


namespace NUMINAMATH_CALUDE_greater_number_proof_l375_37598

theorem greater_number_proof (x y : ℝ) (h_sum : x + y = 40) (h_diff : x - y = 12) (h_greater : x > y) : x = 26 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l375_37598


namespace NUMINAMATH_CALUDE_no_valid_sequence_for_arrangement_D_l375_37544

/-- Represents a cell in the 2x4 grid -/
inductive Cell
| topLeft | topMidLeft | topMidRight | topRight
| bottomLeft | bottomMidLeft | bottomMidRight | bottomRight

/-- Checks if two cells are adjacent (share a common vertex) -/
def adjacent (c1 c2 : Cell) : Prop :=
  match c1, c2 with
  | Cell.topLeft, Cell.topMidLeft | Cell.topLeft, Cell.bottomLeft | Cell.topLeft, Cell.bottomMidLeft => True
  | Cell.topMidLeft, Cell.topLeft | Cell.topMidLeft, Cell.topMidRight | Cell.topMidLeft, Cell.bottomLeft | Cell.topMidLeft, Cell.bottomMidLeft | Cell.topMidLeft, Cell.bottomMidRight => True
  | Cell.topMidRight, Cell.topMidLeft | Cell.topMidRight, Cell.topRight | Cell.topMidRight, Cell.bottomMidLeft | Cell.topMidRight, Cell.bottomMidRight | Cell.topMidRight, Cell.bottomRight => True
  | Cell.topRight, Cell.topMidRight | Cell.topRight, Cell.bottomMidRight | Cell.topRight, Cell.bottomRight => True
  | Cell.bottomLeft, Cell.topLeft | Cell.bottomLeft, Cell.topMidLeft | Cell.bottomLeft, Cell.bottomMidLeft => True
  | Cell.bottomMidLeft, Cell.topLeft | Cell.bottomMidLeft, Cell.topMidLeft | Cell.bottomMidLeft, Cell.topMidRight | Cell.bottomMidLeft, Cell.bottomLeft | Cell.bottomMidLeft, Cell.bottomMidRight => True
  | Cell.bottomMidRight, Cell.topMidLeft | Cell.bottomMidRight, Cell.topMidRight | Cell.bottomMidRight, Cell.topRight | Cell.bottomMidRight, Cell.bottomMidLeft | Cell.bottomMidRight, Cell.bottomRight => True
  | Cell.bottomRight, Cell.topMidRight | Cell.bottomRight, Cell.topRight | Cell.bottomRight, Cell.bottomMidRight => True
  | _, _ => False

/-- Represents a sequence of cell selections -/
def CellSequence := List Cell

/-- Checks if a cell sequence is valid according to the rules -/
def validSequence (seq : CellSequence) : Prop :=
  match seq with
  | [] => True
  | [_] => True
  | c1 :: c2 :: rest => adjacent c1 c2 ∧ validSequence (c2 :: rest)

/-- Represents the arrangement D -/
def arrangementD : List Cell :=
  [Cell.topLeft, Cell.topMidLeft, Cell.topMidRight, Cell.topRight,
   Cell.bottomLeft, Cell.bottomMidRight, Cell.bottomMidLeft, Cell.bottomRight]

/-- Theorem stating that no valid sequence can produce arrangement D -/
theorem no_valid_sequence_for_arrangement_D :
  ¬∃ (seq : CellSequence), validSequence seq ∧ seq.map (λ c => c) = arrangementD := by
  sorry


end NUMINAMATH_CALUDE_no_valid_sequence_for_arrangement_D_l375_37544


namespace NUMINAMATH_CALUDE_complex_expression_equality_l375_37549

theorem complex_expression_equality : 
  (2 + 1/4)^(1/2) - (-9.6)^0 - (3 + 3/8)^(2/3) + (1.5)^2 + (Real.sqrt 2 * 43)^4 = 5/4 + 4 * 43^4 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l375_37549


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l375_37570

-- Define the trajectory C
def C (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y + Real.sqrt 3)^2) + Real.sqrt (x^2 + (y - Real.sqrt 3)^2) = 4

-- Define the intersection line
def intersectionLine (x y : ℝ) : Prop := y = (1/2) * x

theorem trajectory_and_intersection :
  -- The equation of trajectory C
  (∀ x y, C x y ↔ x^2 + y^2/4 = 1) ∧
  -- The length of chord AB
  (∃ x₁ y₁ x₂ y₂, 
    C x₁ y₁ ∧ C x₂ y₂ ∧ 
    intersectionLine x₁ y₁ ∧ intersectionLine x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l375_37570


namespace NUMINAMATH_CALUDE_uncovered_cells_less_than_mn_div_4_uncovered_cells_less_than_mn_div_5_l375_37545

/-- Represents a rectangular board with dominoes -/
structure DominoBoard where
  m : ℕ  -- width of the board
  n : ℕ  -- height of the board
  uncovered_cells : ℕ  -- number of uncovered cells

/-- The number of uncovered cells is less than mn/4 -/
theorem uncovered_cells_less_than_mn_div_4 (board : DominoBoard) :
  board.uncovered_cells < (board.m * board.n) / 4 := by
  sorry

/-- The number of uncovered cells is less than mn/5 -/
theorem uncovered_cells_less_than_mn_div_5 (board : DominoBoard) :
  board.uncovered_cells < (board.m * board.n) / 5 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_cells_less_than_mn_div_4_uncovered_cells_less_than_mn_div_5_l375_37545


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_implies_m_greater_than_two_l375_37591

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point P as a function of m -/
def P (m : ℝ) : Point2D :=
  { x := m - 1, y := 2 - m }

theorem point_in_fourth_quadrant_implies_m_greater_than_two :
  ∀ m : ℝ, isInFourthQuadrant (P m) → m > 2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_implies_m_greater_than_two_l375_37591


namespace NUMINAMATH_CALUDE_geometric_sequence_min_a3_l375_37505

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_min_a3 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 - a 1 = 1 →
  (∀ q : ℝ, q > 0 → a 3 ≤ (a 1) * q^2) →
  ∀ n : ℕ, a n = 2^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_a3_l375_37505


namespace NUMINAMATH_CALUDE_calf_grazing_area_increase_calf_grazing_area_increase_value_l375_37530

/-- The additional area a calf can graze when its rope is increased from 12 m to 25 m -/
theorem calf_grazing_area_increase : ℝ :=
  let initial_length : ℝ := 12
  let final_length : ℝ := 25
  let initial_area := Real.pi * initial_length ^ 2
  let final_area := Real.pi * final_length ^ 2
  final_area - initial_area

/-- The additional area a calf can graze when its rope is increased from 12 m to 25 m is 481π m² -/
theorem calf_grazing_area_increase_value : 
  calf_grazing_area_increase = 481 * Real.pi := by sorry

end NUMINAMATH_CALUDE_calf_grazing_area_increase_calf_grazing_area_increase_value_l375_37530


namespace NUMINAMATH_CALUDE_equilateral_triangle_not_centrally_symmetric_l375_37536

-- Define the shape type
inductive Shape
  | Parallelogram
  | LineSegment
  | EquilateralTriangle
  | Rhombus

-- Define the property of being centrally symmetric
def is_centrally_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => True
  | Shape.LineSegment => True
  | Shape.EquilateralTriangle => False
  | Shape.Rhombus => True

-- Theorem statement
theorem equilateral_triangle_not_centrally_symmetric :
  ∀ s : Shape, ¬(is_centrally_symmetric s) ↔ s = Shape.EquilateralTriangle :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_not_centrally_symmetric_l375_37536


namespace NUMINAMATH_CALUDE_max_slices_formula_l375_37526

/-- Represents a triangular cake with candles -/
structure TriangularCake where
  numCandles : ℕ
  candlesNotCollinear : True  -- Placeholder for the condition that no three candles are collinear

/-- The maximum number of triangular slices for a given cake -/
def maxSlices (cake : TriangularCake) : ℕ :=
  2 * cake.numCandles - 5

/-- Theorem stating the maximum number of slices for a cake with k candles -/
theorem max_slices_formula (k : ℕ) (h : k ≥ 3) :
  ∀ (cake : TriangularCake), cake.numCandles = k →
    maxSlices cake = 2 * k - 5 := by
  sorry

end NUMINAMATH_CALUDE_max_slices_formula_l375_37526


namespace NUMINAMATH_CALUDE_ticket_sales_l375_37504

theorem ticket_sales (total : ℕ) (full_price : ℕ) (reduced_price : ℕ) :
  total = 25200 →
  full_price = 16500 →
  full_price = 5 * reduced_price →
  reduced_price = 3300 := by
sorry

end NUMINAMATH_CALUDE_ticket_sales_l375_37504


namespace NUMINAMATH_CALUDE_mayoral_election_votes_l375_37509

theorem mayoral_election_votes (candidate_x candidate_y other_candidate : ℕ) : 
  candidate_x = candidate_y + (candidate_y / 2) →
  candidate_y = other_candidate - (other_candidate * 2 / 5) →
  candidate_x = 22500 →
  other_candidate = 25000 := by
sorry

end NUMINAMATH_CALUDE_mayoral_election_votes_l375_37509


namespace NUMINAMATH_CALUDE_weight_of_new_person_l375_37506

/-- Calculates the weight of a new person in a group replacement scenario. -/
def newPersonWeight (groupSize : ℕ) (avgWeightIncrease : ℝ) (replacedPersonWeight : ℝ) : ℝ :=
  replacedPersonWeight + groupSize * avgWeightIncrease

/-- Proves that the weight of the new person is 108 kg given the specified conditions. -/
theorem weight_of_new_person :
  let groupSize : ℕ := 15
  let avgWeightIncrease : ℝ := 2.2
  let replacedPersonWeight : ℝ := 75
  newPersonWeight groupSize avgWeightIncrease replacedPersonWeight = 108 := by
  sorry

#eval newPersonWeight 15 2.2 75

end NUMINAMATH_CALUDE_weight_of_new_person_l375_37506


namespace NUMINAMATH_CALUDE_pizza_piece_volume_l375_37561

/-- The volume of a piece of pizza -/
theorem pizza_piece_volume (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) :
  thickness = 1/4 →
  diameter = 16 →
  num_pieces = 16 →
  (π * (diameter/2)^2 * thickness) / num_pieces = π := by
  sorry

end NUMINAMATH_CALUDE_pizza_piece_volume_l375_37561


namespace NUMINAMATH_CALUDE_sticker_collection_size_l375_37542

theorem sticker_collection_size : ∃! (N : ℕ), N < 100 ∧ N % 6 = 2 ∧ N % 8 = 3 ∧ N = 83 := by
  sorry

end NUMINAMATH_CALUDE_sticker_collection_size_l375_37542


namespace NUMINAMATH_CALUDE_additional_hovering_time_l375_37564

/-- Represents the hovering time of a plane in different time zones over two days. -/
structure PlaneHoveringTime where
  mountain_day1 : ℕ
  central_day1 : ℕ
  eastern_day1 : ℕ
  mountain_day2 : ℕ
  central_day2 : ℕ
  eastern_day2 : ℕ

/-- Theorem stating that given the conditions of the problem, the additional hovering time
    in each time zone on the second day is 5 hours. -/
theorem additional_hovering_time
  (h : PlaneHoveringTime)
  (h_mountain_day1 : h.mountain_day1 = 3)
  (h_central_day1 : h.central_day1 = 4)
  (h_eastern_day1 : h.eastern_day1 = 2)
  (h_total_time : h.mountain_day1 + h.central_day1 + h.eastern_day1 +
                  h.mountain_day2 + h.central_day2 + h.eastern_day2 = 24)
  (h_equal_additional : h.mountain_day2 = h.central_day2 ∧ h.central_day2 = h.eastern_day2) :
  h.mountain_day2 = 5 ∧ h.central_day2 = 5 ∧ h.eastern_day2 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_additional_hovering_time_l375_37564


namespace NUMINAMATH_CALUDE_total_marbles_l375_37541

/-- Given that Rhonda has 80 marbles and Amon has 55 more marbles than Rhonda,
    prove that they have 215 marbles combined. -/
theorem total_marbles (rhonda_marbles : ℕ) (amon_extra_marbles : ℕ) 
    (h1 : rhonda_marbles = 80)
    (h2 : amon_extra_marbles = 55) : 
  rhonda_marbles + (rhonda_marbles + amon_extra_marbles) = 215 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l375_37541


namespace NUMINAMATH_CALUDE_steve_juice_consumption_l375_37510

theorem steve_juice_consumption (don_juice : ℚ) (steve_fraction : ℚ) :
  don_juice = 1/4 →
  steve_fraction = 3/4 →
  steve_fraction * don_juice = 3/16 := by
sorry

end NUMINAMATH_CALUDE_steve_juice_consumption_l375_37510


namespace NUMINAMATH_CALUDE_johann_oranges_l375_37588

def oranges_problem (initial_oranges eaten_oranges stolen_fraction returned_oranges : ℕ) : Prop :=
  let remaining_after_eating := initial_oranges - eaten_oranges
  let stolen := (remaining_after_eating / 2 : ℕ)
  let final_oranges := remaining_after_eating - stolen + returned_oranges
  final_oranges = 30

theorem johann_oranges :
  oranges_problem 60 10 2 5 := by sorry

end NUMINAMATH_CALUDE_johann_oranges_l375_37588


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l375_37592

/-- Two parabolas intersecting coordinate axes to form a kite -/
def parabola_kite (a b : ℝ) : Prop :=
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    -- Parabola equations
    (∀ x, a * x^2 - 4 = 6 - b * x^2 → x = x₁ ∨ x = x₂) ∧
    (∀ y, y = a * 0^2 - 4 → y = y₁) ∧
    (∀ y, y = 6 - b * 0^2 → y = y₂) ∧
    -- Four distinct intersection points
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    -- Kite area
    (1/2) * (x₂ - x₁) * (y₂ - y₁) = 18

/-- Theorem: If two parabolas form a kite with area 18, then a + b = 125/36 -/
theorem parabola_kite_sum (a b : ℝ) :
  parabola_kite a b → a + b = 125/36 := by
  sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l375_37592


namespace NUMINAMATH_CALUDE_range_of_a_for_union_equality_intersection_A_B_union_A_B_l375_37529

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 7}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 6}

-- State the theorem
theorem range_of_a_for_union_equality :
  ∀ a : ℝ, (A ∪ C a = C a) ↔ (2 ≤ a ∧ a < 3) :=
by sorry

-- Additional theorems for intersection and union of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 7} :=
by sorry

theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_union_equality_intersection_A_B_union_A_B_l375_37529


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_175_l375_37520

theorem smallest_prime_factor_of_175 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 175 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 175 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_175_l375_37520


namespace NUMINAMATH_CALUDE_inequality_proof_l375_37538

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ((a^3) / (a^3 + 15*b*c*d))^(1/2) ≥ (a^(15/8)) / (a^(15/8) + b^(15/8) + c^(15/8) + d^(15/8)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l375_37538


namespace NUMINAMATH_CALUDE_complex_number_simplification_l375_37527

theorem complex_number_simplification :
  let i : ℂ := Complex.I
  (5 : ℂ) / (2 - i) - i = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l375_37527


namespace NUMINAMATH_CALUDE_element_correspondence_l375_37524

-- Define the mapping f from A to B
def f (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem element_correspondence : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_element_correspondence_l375_37524


namespace NUMINAMATH_CALUDE_solutions_exist_and_finite_l375_37514

theorem solutions_exist_and_finite :
  ∃ (n : ℕ) (S : Finset ℝ),
    (∀ θ ∈ S, 0 < θ ∧ θ < 2 * Real.pi) ∧
    (∀ θ ∈ S, Real.sin (7 * Real.pi * Real.cos θ) = Real.cos (7 * Real.pi * Real.sin θ)) ∧
    S.card = n :=
by sorry

end NUMINAMATH_CALUDE_solutions_exist_and_finite_l375_37514


namespace NUMINAMATH_CALUDE_rectangle_diagonal_perimeter_ratio_l375_37571

theorem rectangle_diagonal_perimeter_ratio :
  ∀ (long_side : ℝ),
  long_side > 0 →
  let short_side := (1/3) * long_side
  let diagonal := Real.sqrt (short_side^2 + long_side^2)
  let perimeter := 2 * (short_side + long_side)
  let saved_distance := (1/3) * long_side
  diagonal + saved_distance = long_side →
  diagonal / perimeter = Real.sqrt 10 / 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_perimeter_ratio_l375_37571


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l375_37569

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h : c ≠ 0) : a / c = b / c → a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l375_37569


namespace NUMINAMATH_CALUDE_vector_collinearity_l375_37572

/-- Given vectors in ℝ², prove that if 3a + b is collinear with c, then x = -4 -/
theorem vector_collinearity (a b c : ℝ × ℝ) (x : ℝ) :
  a = (-2, 0) →
  b = (2, 1) →
  c = (x, 1) →
  ∃ (k : ℝ), k • (3 • a + b) = c →
  x = -4 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l375_37572


namespace NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l375_37522

theorem tan_ratio_from_sin_sum_diff (p q : Real) 
  (h1 : Real.sin (p + q) = 0.6) 
  (h2 : Real.sin (p - q) = 0.3) : 
  Real.tan p / Real.tan q = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l375_37522


namespace NUMINAMATH_CALUDE_line_through_points_l375_37562

/-- A line passing through three given points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The y-coordinate of a point on the line given its x-coordinate -/
def y_coord (l : Line) (x : ℝ) : ℝ :=
  sorry

theorem line_through_points (l : Line) : 
  l.point1 = (2, 8) ∧ l.point2 = (4, 14) ∧ l.point3 = (6, 20) → 
  y_coord l 50 = 152 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l375_37562


namespace NUMINAMATH_CALUDE_tshirt_sale_revenue_per_minute_l375_37552

/-- Calculates the money made per minute in a t-shirt sale. -/
theorem tshirt_sale_revenue_per_minute 
  (total_shirts : ℕ) 
  (sale_duration : ℕ) 
  (black_shirt_price : ℕ) 
  (white_shirt_price : ℕ) : 
  total_shirts = 200 →
  sale_duration = 25 →
  black_shirt_price = 30 →
  white_shirt_price = 25 →
  (total_shirts / 2 * black_shirt_price + total_shirts / 2 * white_shirt_price) / sale_duration = 220 :=
by
  sorry

#check tshirt_sale_revenue_per_minute

end NUMINAMATH_CALUDE_tshirt_sale_revenue_per_minute_l375_37552


namespace NUMINAMATH_CALUDE_equation_solution_l375_37521

theorem equation_solution :
  let f (x : ℝ) := (3 * x^2) / (x - 2) - (3 * x + 8) / 4 + (5 - 9 * x) / (x - 2) + 2
  ∀ x : ℝ, x ≠ 2 → (f x = 0 ↔ x = (5 + Real.sqrt 5) / 3 ∨ x = (5 - Real.sqrt 5) / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l375_37521


namespace NUMINAMATH_CALUDE_units_digit_of_27_times_36_l375_37599

theorem units_digit_of_27_times_36 : (27 * 36) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_27_times_36_l375_37599


namespace NUMINAMATH_CALUDE_minimum_houses_with_more_than_five_floors_l375_37535

theorem minimum_houses_with_more_than_five_floors (n : ℕ) : 
  (n > 0) → 
  (∃ x : ℕ, x < n ∧ (n - x : ℚ) / n > 47/50) → 
  (∀ m : ℕ, m < n → ∃ y : ℕ, y < m ∧ (m - y : ℚ) / m ≤ 47/50) → 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_minimum_houses_with_more_than_five_floors_l375_37535


namespace NUMINAMATH_CALUDE_geometry_propositions_l375_37513

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the main theorem
theorem geometry_propositions
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : subset m β) :
  -- Exactly two of the following propositions are correct
  ∃! (correct : Fin 4 → Prop),
    (∀ i, correct i ↔ i.val < 2) ∧
    correct 0 = (parallel α β → line_perpendicular l m) ∧
    correct 1 = (line_perpendicular l m → parallel α β) ∧
    correct 2 = (plane_perpendicular α β → line_parallel l m) ∧
    correct 3 = (line_parallel l m → plane_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_propositions_l375_37513


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l375_37579

theorem complex_modulus_theorem : Complex.abs (-6 + (9/4) * Complex.I) = (Real.sqrt 657) / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l375_37579


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l375_37539

def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

theorem monotonic_decreasing_interval_of_f :
  ∀ a b : ℝ, a = -1 ∧ b = 11 →
  (∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f y < f x) ∧
  ¬(∃ c d : ℝ, (c < a ∨ b < d) ∧
    (∀ x y : ℝ, c < x ∧ x < y ∧ y < d → f y < f x)) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l375_37539


namespace NUMINAMATH_CALUDE_total_tomatoes_l375_37597

def tomato_problem (plant1 plant2 plant3 : ℕ) : Prop :=
  plant1 = 24 ∧
  plant2 = (plant1 / 2) + 5 ∧
  plant3 = plant2 + 2 ∧
  plant1 + plant2 + plant3 = 60

theorem total_tomatoes :
  ∃ plant1 plant2 plant3 : ℕ, tomato_problem plant1 plant2 plant3 :=
sorry

end NUMINAMATH_CALUDE_total_tomatoes_l375_37597


namespace NUMINAMATH_CALUDE_stratified_sampling_athletes_l375_37576

theorem stratified_sampling_athletes (total_male : ℕ) (total_female : ℕ) 
  (drawn_male : ℕ) (drawn_female : ℕ) : 
  total_male = 64 → total_female = 56 → drawn_male = 8 →
  (drawn_male : ℚ) / total_male = (drawn_female : ℚ) / total_female →
  drawn_female = 7 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_athletes_l375_37576


namespace NUMINAMATH_CALUDE_right_triangle_area_l375_37557

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 12 →
  angle = 30 * π / 180 →
  let shortest_side := hypotenuse / 2
  let longest_side := hypotenuse / 2 * Real.sqrt 3
  let area := shortest_side * longest_side / 2
  area = 18 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l375_37557


namespace NUMINAMATH_CALUDE_lattice_points_5_11_to_35_221_l375_37563

/-- The number of lattice points on a line segment --/
def lattice_points_on_segment (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (5, 11) to (35, 221) is 31 --/
theorem lattice_points_5_11_to_35_221 :
  lattice_points_on_segment 5 11 35 221 = 31 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_5_11_to_35_221_l375_37563


namespace NUMINAMATH_CALUDE_probability_all_male_students_l375_37518

theorem probability_all_male_students (total_male : ℕ) (total_female : ℕ) 
  (selected : ℕ) (prob_at_least_one_female : ℚ) :
  total_male = 4 →
  total_female = 2 →
  selected = 3 →
  prob_at_least_one_female = 4/5 →
  1 - prob_at_least_one_female = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_male_students_l375_37518


namespace NUMINAMATH_CALUDE_marks_score_ratio_l375_37503

theorem marks_score_ratio (highest_score range marks_score : ℕ) : 
  highest_score = 98 →
  range = 75 →
  marks_score = 46 →
  marks_score % (highest_score - range) = 0 →
  marks_score / (highest_score - range) = 2 :=
by sorry

end NUMINAMATH_CALUDE_marks_score_ratio_l375_37503


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_reciprocal_l375_37507

theorem cubic_roots_sum_of_cubes_reciprocal (a b c d r s : ℝ) :
  a ≠ 0 →
  c ≠ 0 →
  a * r^3 + b * r^2 + c * r + d = 0 →
  a * s^3 + b * s^2 + c * s + d = 0 →
  r ≠ 0 →
  s ≠ 0 →
  (1 / r^3) + (1 / s^3) = (b^3 - 3 * a * b * c) / c^3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_reciprocal_l375_37507


namespace NUMINAMATH_CALUDE_group_size_is_16_l375_37546

/-- The number of children whose height increases -/
def num_taller_children : ℕ := 12

/-- The height increase for each of the taller children in cm -/
def height_increase : ℕ := 8

/-- The total height increase in cm -/
def total_height_increase : ℕ := num_taller_children * height_increase

/-- The mean height increase in cm -/
def mean_height_increase : ℕ := 6

theorem group_size_is_16 :
  ∃ n : ℕ, n > 0 ∧ (total_height_increase : ℚ) / n = mean_height_increase ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_group_size_is_16_l375_37546


namespace NUMINAMATH_CALUDE_lunch_cost_before_tip_l375_37528

/-- Given a 20% tip and a total spending of $60.6, prove that the original cost of the lunch before the tip was $50.5. -/
theorem lunch_cost_before_tip (tip_percentage : Real) (total_spent : Real) (lunch_cost : Real) : 
  tip_percentage = 0.20 →
  total_spent = 60.6 →
  lunch_cost * (1 + tip_percentage) = total_spent →
  lunch_cost = 50.5 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_before_tip_l375_37528


namespace NUMINAMATH_CALUDE_parabola_coefficient_l375_37555

/-- The value of 'a' for a parabola y = ax^2 - 2x + 3 passing through the point (1, 2) -/
theorem parabola_coefficient (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 - 2 * x + 3) → 
  (2 : ℝ) = a * 1^2 - 2 * 1 + 3 → 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l375_37555


namespace NUMINAMATH_CALUDE_infinite_nonzero_digit_sum_equality_l375_37516

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a natural number contains zero in its digits -/
def contains_zero (n : ℕ) : Prop := sorry

theorem infinite_nonzero_digit_sum_equality :
  ∀ k : ℕ, ∃ f : ℕ → ℕ,
    (∀ n : ℕ, ¬contains_zero (f n)) ∧
    (∀ n : ℕ, sum_of_digits (f n) = sum_of_digits (k * f n)) ∧
    (∀ n : ℕ, f n < f (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_nonzero_digit_sum_equality_l375_37516


namespace NUMINAMATH_CALUDE_circle_square_tangency_l375_37537

theorem circle_square_tangency (r : ℝ) (s : ℝ) 
  (hr : r = 13) (hs : s = 18) : 
  let d := Real.sqrt (r^2 - (s - r)^2)
  (s - d = 1) ∧ d = 17 := by sorry

end NUMINAMATH_CALUDE_circle_square_tangency_l375_37537


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l375_37502

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l375_37502


namespace NUMINAMATH_CALUDE_function_characterization_l375_37577

-- Define the property that the function f must satisfy
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a^2) - f (b^2) ≤ (f a + b) * (a - f b)

-- State the theorem
theorem function_characterization (f : ℝ → ℝ) (h : SatisfiesInequality f) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l375_37577


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l375_37556

/-- The price ratio of a muffin to a banana is 2 -/
theorem muffin_banana_price_ratio :
  ∀ (m b S : ℝ),
  (3 * m + 5 * b = S) →
  (5 * m + 7 * b = 3 * S) →
  m = 2 * b :=
by sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l375_37556


namespace NUMINAMATH_CALUDE_largest_integer_less_than_150_with_remainder_2_mod_9_l375_37559

theorem largest_integer_less_than_150_with_remainder_2_mod_9 : ∃ n : ℕ, n < 150 ∧ n % 9 = 2 ∧ ∀ m : ℕ, m < 150 ∧ m % 9 = 2 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_150_with_remainder_2_mod_9_l375_37559


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l375_37560

/-- The cost price of one meter of cloth given the selling price, quantity, and profit per meter -/
theorem cost_price_per_meter
  (selling_price : ℕ)
  (quantity : ℕ)
  (profit_per_meter : ℕ)
  (h1 : selling_price = 8925)
  (h2 : quantity = 85)
  (h3 : profit_per_meter = 20) :
  (selling_price - quantity * profit_per_meter) / quantity = 85 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l375_37560


namespace NUMINAMATH_CALUDE_rearrangement_time_theorem_l375_37553

/-- The number of letters in the name -/
def name_length : ℕ := 9

/-- The number of times the repeated letter appears -/
def repeated_letter_count : ℕ := 2

/-- The number of rearrangements that can be written per minute -/
def rearrangements_per_minute : ℕ := 15

/-- Calculate the number of unique rearrangements -/
def unique_rearrangements : ℕ := name_length.factorial / repeated_letter_count.factorial

/-- Calculate the total time in hours to write all rearrangements -/
def total_time_hours : ℚ :=
  (unique_rearrangements / rearrangements_per_minute : ℚ) / 60

/-- Theorem stating the time required to write all rearrangements -/
theorem rearrangement_time_theorem :
  total_time_hours = 201.6 := by sorry

end NUMINAMATH_CALUDE_rearrangement_time_theorem_l375_37553


namespace NUMINAMATH_CALUDE_exists_greatest_n_leq_2008_l375_37517

/-- Checks if a number is a perfect square -/
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

/-- The sum of squares formula for natural numbers -/
def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The formula for the sum of squares from n+1 to 3n -/
def sumOfSquaresNTo3N (n : ℕ) : ℕ := (26 * n^3 + 12 * n^2 + n) / 3

/-- The main theorem statement -/
theorem exists_greatest_n_leq_2008 :
  ∃ n : ℕ, n ≤ 2008 ∧ 
    isPerfectSquare (sumOfSquares n * sumOfSquaresNTo3N n) ∧
    ∀ m : ℕ, m > n → m ≤ 2008 → 
      ¬ isPerfectSquare (sumOfSquares m * sumOfSquaresNTo3N m) := by
  sorry

end NUMINAMATH_CALUDE_exists_greatest_n_leq_2008_l375_37517


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l375_37582

/-- Given a line segment CD with midpoint N and endpoint C, proves that the product of D's coordinates is 39 -/
theorem midpoint_coordinate_product (C N D : ℝ × ℝ) : 
  C = (5, 3) → N = (4, 8) → N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → D.1 * D.2 = 39 := by
  sorry

#check midpoint_coordinate_product

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l375_37582


namespace NUMINAMATH_CALUDE_f_satisfies_condition_l375_37596

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(1 / (log x))

-- State the theorem
theorem f_satisfies_condition (a : ℝ) (h_a : a > 1) :
  ∀ (x y u v : ℝ), x > 1 → y > 1 → u > 0 → v > 0 →
    f a (x^u * y^v) ≤ (f a x)^(1/(1*u)) * (f a y)^(1/10) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_condition_l375_37596


namespace NUMINAMATH_CALUDE_geometric_sequence_partial_sums_zero_property_l375_37594

/-- A geometric sequence of real numbers -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Partial sums of a sequence -/
def partial_sums (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => partial_sums a n + a (n + 1)

/-- The main theorem -/
theorem geometric_sequence_partial_sums_zero_property
  (a : ℕ → ℝ) (h : geometric_sequence a) :
  (∀ n : ℕ, partial_sums a n ≠ 0) ∨
  (∀ m : ℕ, ∃ n : ℕ, n ≥ m ∧ partial_sums a n = 0) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_partial_sums_zero_property_l375_37594


namespace NUMINAMATH_CALUDE_probability_red_is_half_l375_37554

def bag_contents : ℕ × ℕ := (3, 3)

def probability_red (contents : ℕ × ℕ) : ℚ :=
  contents.1 / (contents.1 + contents.2)

theorem probability_red_is_half : 
  probability_red bag_contents = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_red_is_half_l375_37554


namespace NUMINAMATH_CALUDE_multiples_of_12_around_negative_150_l375_37543

theorem multiples_of_12_around_negative_150 :
  ∀ n m : ℤ,
  (∀ k : ℤ, 12 * k < -150 → k ≤ n) →
  (∀ j : ℤ, 12 * j > -150 → m ≤ j) →
  12 * n = -156 ∧ 12 * m = -144 :=
by
  sorry

end NUMINAMATH_CALUDE_multiples_of_12_around_negative_150_l375_37543


namespace NUMINAMATH_CALUDE_parallel_vectors_l375_37548

/-- Given vectors a and b, if a is parallel to (a + b), then the second component of b is -3. -/
theorem parallel_vectors (a b : ℝ × ℝ) (h : ∃ (k : ℝ), a = k • (a + b)) : 
  a.1 = -1 ∧ a.2 = 1 ∧ b.1 = 3 → b.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l375_37548


namespace NUMINAMATH_CALUDE_equation_equivalence_l375_37533

theorem equation_equivalence (x y : ℝ) :
  y^2 - 2*x*y + x^2 - 1 = 0 ↔ (y = x + 1 ∨ y = x - 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l375_37533


namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l375_37583

-- Part 1
theorem calculation_proof :
  |Real.sqrt 5 - 3| + (1/2)⁻¹ - Real.sqrt 20 + Real.sqrt 3 * Real.cos (30 * π / 180) = 13/2 - 3 * Real.sqrt 5 := by
  sorry

-- Part 2
theorem inequality_system_solution (x : ℝ) :
  (-3 * (x - 2) ≥ 4 - x ∧ (1 + 2*x) / 3 > x - 1) ↔ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l375_37583


namespace NUMINAMATH_CALUDE_line_through_points_l375_37525

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a given line -/
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The theorem states that the line x - 2y + 1 = 0 passes through points A(-1, 0) and B(3, 2) -/
theorem line_through_points :
  let A : Point2D := ⟨-1, 0⟩
  let B : Point2D := ⟨3, 2⟩
  let line : Line2D := ⟨1, -2, 1⟩
  point_on_line A line ∧ point_on_line B line :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l375_37525


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l375_37558

/-- Theorem: A cube with surface area approximately 600 square cc has a volume of 1000 cubic cc. -/
theorem cube_volume_from_surface_area :
  ∃ (s : ℝ), s > 0 ∧ 6 * s^2 = 599.9999999999998 → s^3 = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l375_37558


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l375_37581

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 - i) / (1 + i) = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l375_37581


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l375_37580

/-- The number of people sitting at the round table -/
def total_people : ℕ := 9

/-- The number of math majors -/
def math_majors : ℕ := 4

/-- The number of ways to choose seats for math majors -/
def total_arrangements : ℕ := Nat.choose total_people math_majors

/-- The number of ways for math majors to sit in consecutive seats -/
def consecutive_arrangements : ℕ := total_people

/-- The probability that all math majors sit in consecutive seats -/
def probability : ℚ := consecutive_arrangements / total_arrangements

theorem math_majors_consecutive_probability :
  probability = 1 / 14 := by sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l375_37580


namespace NUMINAMATH_CALUDE_equation_solution_l375_37523

theorem equation_solution (x : ℝ) : 
  x ≠ 3 → ((2 - x) / (x - 3) = 1 / (x - 3) - 2) → x = 5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l375_37523


namespace NUMINAMATH_CALUDE_problem_I3_1_l375_37512

theorem problem_I3_1 (w x y z : ℝ) (hw : w > 0) 
  (h1 : w * x * y * z = 4) (h2 : w - x * y * z = 3) : w = 4 := by
sorry


end NUMINAMATH_CALUDE_problem_I3_1_l375_37512


namespace NUMINAMATH_CALUDE_total_marks_difference_l375_37578

theorem total_marks_difference (P C M : ℝ) 
  (h1 : P + C + M > P) 
  (h2 : (C + M) / 2 = 75) : 
  P + C + M - P = 150 := by
sorry

end NUMINAMATH_CALUDE_total_marks_difference_l375_37578


namespace NUMINAMATH_CALUDE_motel_room_rate_l375_37550

theorem motel_room_rate (total_rent : ℕ) (lower_rate : ℕ) (reduction_percentage : ℚ) 
  (num_rooms_changed : ℕ) (h1 : total_rent = 2000) (h2 : lower_rate = 40) 
  (h3 : reduction_percentage = 1/10) (h4 : num_rooms_changed = 10) : 
  ∃ (higher_rate : ℕ), 
    (∃ (num_lower_rooms num_higher_rooms : ℕ), 
      total_rent = lower_rate * num_lower_rooms + higher_rate * num_higher_rooms ∧
      total_rent - (reduction_percentage * total_rent) = 
        lower_rate * (num_lower_rooms + num_rooms_changed) + 
        higher_rate * (num_higher_rooms - num_rooms_changed)) ∧
    higher_rate = 60 := by
  sorry

end NUMINAMATH_CALUDE_motel_room_rate_l375_37550


namespace NUMINAMATH_CALUDE_vector_subtraction_l375_37574

/-- Given two vectors OM and ON in ℝ², prove that the vector MN has coordinates (-8, 1) -/
theorem vector_subtraction (OM ON : ℝ × ℝ) (h1 : OM = (3, -2)) (h2 : ON = (-5, -1)) :
  ON - OM = (-8, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l375_37574


namespace NUMINAMATH_CALUDE_polynomial_factorization_l375_37593

theorem polynomial_factorization (y : ℝ) : 
  (20 * y^4 + 100 * y - 10) - (5 * y^3 - 15 * y + 10) = 5 * (4 * y^4 - y^3 + 23 * y - 4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l375_37593


namespace NUMINAMATH_CALUDE_train_length_calculation_l375_37501

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 30 → time_s = 6 → 
  ∃ (length_m : ℝ), (abs (length_m - 50) < 1 ∧ length_m = speed_kmh * (1000 / 3600) * time_s) :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l375_37501


namespace NUMINAMATH_CALUDE_exact_five_green_probability_l375_37590

def total_marbles : ℕ := 12
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 4
def total_draws : ℕ := 8
def green_draws : ℕ := 5

def prob_green : ℚ := green_marbles / total_marbles
def prob_purple : ℚ := purple_marbles / total_marbles

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem exact_five_green_probability :
  (binomial_coefficient total_draws green_draws : ℚ) * 
  (prob_green ^ green_draws) * 
  (prob_purple ^ (total_draws - green_draws)) =
  56 * (2/3)^5 * (1/3)^3 := by sorry

end NUMINAMATH_CALUDE_exact_five_green_probability_l375_37590


namespace NUMINAMATH_CALUDE_unique_function_solution_l375_37534

theorem unique_function_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (1 + x * y) - f (x + y) = f x * f y) ∧ 
  (f (-1) ≠ 0) → 
  (∀ x : ℝ, f x = x - 1) :=
sorry

end NUMINAMATH_CALUDE_unique_function_solution_l375_37534


namespace NUMINAMATH_CALUDE_line_A2A3_tangent_to_circle_M_l375_37565

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = x

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point on the parabola
def point_on_parabola (A : ℝ × ℝ) : Prop := parabola_C A.1 A.2

-- Define a line tangent to the circle M
def line_tangent_to_circle_M (A B : ℝ × ℝ) : Prop :=
  let d := abs ((B.2 - A.2) * 2 - (B.1 - A.1) * 0 + (A.1 * B.2 - B.1 * A.2)) /
            Real.sqrt ((B.2 - A.2)^2 + (B.1 - A.1)^2)
  d = 1

-- State the theorem
theorem line_A2A3_tangent_to_circle_M 
  (A₁ A₂ A₃ : ℝ × ℝ)
  (h₁ : point_on_parabola A₁)
  (h₂ : point_on_parabola A₂)
  (h₃ : point_on_parabola A₃)
  (h₄ : line_tangent_to_circle_M A₁ A₂)
  (h₅ : line_tangent_to_circle_M A₁ A₃) :
  line_tangent_to_circle_M A₂ A₃ := by sorry

end NUMINAMATH_CALUDE_line_A2A3_tangent_to_circle_M_l375_37565


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l375_37515

theorem solve_system_of_equations :
  ∃ (x y : ℝ), (x / 6) * 12 = 11 ∧ 4 * (x - y) + 5 = 11 ∧ x = 5.5 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l375_37515


namespace NUMINAMATH_CALUDE_small_cube_edge_length_l375_37567

/-- Given a cube with volume 1000 cm³, prove that cutting off 8 small cubes
    of equal size from its corners, resulting in a remaining volume of 488 cm³,
    yields small cubes with edge length 4 cm. -/
theorem small_cube_edge_length 
  (initial_volume : ℝ) 
  (remaining_volume : ℝ) 
  (num_small_cubes : ℕ) 
  (h_initial : initial_volume = 1000)
  (h_remaining : remaining_volume = 488)
  (h_num_cubes : num_small_cubes = 8) :
  ∃ (edge_length : ℝ), 
    edge_length = 4 ∧ 
    initial_volume - num_small_cubes * edge_length ^ 3 = remaining_volume :=
by sorry

end NUMINAMATH_CALUDE_small_cube_edge_length_l375_37567


namespace NUMINAMATH_CALUDE_range_of_a_l375_37519

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ 2 * x * (x - a) < 1) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l375_37519


namespace NUMINAMATH_CALUDE_square_semicircle_perimeter_l375_37568

theorem square_semicircle_perimeter : 
  let square_side : ℝ := 2 / Real.pi
  let semicircle_diameter : ℝ := square_side
  let full_circle_circumference : ℝ := Real.pi * semicircle_diameter
  let region_perimeter : ℝ := 2 * full_circle_circumference
  region_perimeter = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_semicircle_perimeter_l375_37568


namespace NUMINAMATH_CALUDE_min_value_expression_l375_37532

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (min : ℝ), min = 4 * Real.sqrt 3 ∧
  ∀ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0),
    x^2 + y^2 + 16 / x^2 + 4 * y / x ≥ min ∧
    ∃ (a₀ b₀ : ℝ) (ha₀ : a₀ ≠ 0) (hb₀ : b₀ ≠ 0),
      a₀^2 + b₀^2 + 16 / a₀^2 + 4 * b₀ / a₀ = min :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l375_37532


namespace NUMINAMATH_CALUDE_book_sales_ratio_l375_37585

theorem book_sales_ratio : 
  ∀ (T : ℕ), -- Number of books sold on Thursday
  15 + T + T / 5 = 69 → -- Total sales equation
  T / 15 = 3 -- Ratio of Thursday to Wednesday sales
  := by sorry

end NUMINAMATH_CALUDE_book_sales_ratio_l375_37585


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l375_37587

-- Define the function types
variable (f g : ℝ → ℝ)

-- Define the functional equation condition
def functional_equation : Prop :=
  ∀ x y : ℝ, f (x - f y) = x * f y - y * f x + g x

-- State the theorem
theorem functional_equation_solutions :
  functional_equation f g →
  ((∀ x, f x = 0 ∧ g x = 0) ∨ (∀ x, f x = x ∧ g x = 0)) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l375_37587


namespace NUMINAMATH_CALUDE_dentists_age_l375_37511

theorem dentists_age : ∃ (x : ℕ), 
  (x - 8) / 6 = (x + 8) / 10 ∧ x = 32 := by
  sorry

end NUMINAMATH_CALUDE_dentists_age_l375_37511
