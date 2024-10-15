import Mathlib

namespace NUMINAMATH_CALUDE_constant_term_of_polynomial_product_l698_69808

theorem constant_term_of_polynomial_product :
  let p : Polynomial ℤ := X^3 + 2*X + 7
  let q : Polynomial ℤ := 2*X^4 + 3*X^2 + 10
  (p * q).coeff 0 = 70 := by sorry

end NUMINAMATH_CALUDE_constant_term_of_polynomial_product_l698_69808


namespace NUMINAMATH_CALUDE_range_of_a_l698_69818

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2 ≥ 0) ↔ a ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l698_69818


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l698_69890

theorem polynomial_division_theorem (x : ℝ) : 
  (x + 5) * (x^4 - 5*x^3 + 2*x^2 + x - 19) + 105 = x^5 - 23*x^3 + 11*x^2 - 14*x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l698_69890


namespace NUMINAMATH_CALUDE_sum_f_positive_l698_69897

def f (x : ℝ) := x^3 + x

theorem sum_f_positive (a b c : ℝ) (hab : a + b > 0) (hbc : b + c > 0) (hca : c + a > 0) :
  f a + f b + f c > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l698_69897


namespace NUMINAMATH_CALUDE_profit_is_eight_percent_l698_69883

/-- Given a markup percentage and a discount percentage, calculate the profit percentage. -/
def profit_percentage (markup : ℝ) (discount : ℝ) : ℝ :=
  let marked_price := 1 + markup
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100

/-- Theorem stating that given a 30% markup and 16.92307692307692% discount, the profit is 8%. -/
theorem profit_is_eight_percent :
  profit_percentage 0.3 0.1692307692307692 = 8 := by
  sorry

end NUMINAMATH_CALUDE_profit_is_eight_percent_l698_69883


namespace NUMINAMATH_CALUDE_ball_placement_count_l698_69862

-- Define the number of balls
def num_balls : ℕ := 3

-- Define the number of available boxes (excluding box 1)
def num_boxes : ℕ := 3

-- Theorem statement
theorem ball_placement_count : (num_boxes ^ num_balls) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ball_placement_count_l698_69862


namespace NUMINAMATH_CALUDE_second_group_women_l698_69817

/-- The work rate of one man -/
def man_rate : ℝ := sorry

/-- The work rate of one woman -/
def woman_rate : ℝ := sorry

/-- The number of women in the second group -/
def x : ℕ := sorry

/-- The work rate of 3 men and 8 women equals the work rate of 6 men and x women -/
axiom work_rate_equality : 3 * man_rate + 8 * woman_rate = 6 * man_rate + x * woman_rate

/-- The work rate of 4 men and 5 women is 0.9285714285714286 times the work rate of 3 men and 8 women -/
axiom work_rate_fraction : 4 * man_rate + 5 * woman_rate = 0.9285714285714286 * (3 * man_rate + 8 * woman_rate)

/-- The number of women in the second group is 14 -/
theorem second_group_women : x = 14 := by sorry

end NUMINAMATH_CALUDE_second_group_women_l698_69817


namespace NUMINAMATH_CALUDE_investment_problem_l698_69801

/-- Proves that the initial investment is $698 given the conditions of Peter and David's investments -/
theorem investment_problem (peter_amount : ℝ) (david_amount : ℝ) (peter_years : ℝ) (david_years : ℝ)
  (h_peter : peter_amount = 815)
  (h_david : david_amount = 854)
  (h_peter_years : peter_years = 3)
  (h_david_years : david_years = 4)
  (h_same_principal : ∃ (P : ℝ), P > 0 ∧ 
    ∃ (r : ℝ), r > 0 ∧ 
      peter_amount = P * (1 + r * peter_years) ∧
      david_amount = P * (1 + r * david_years)) :
  ∃ (P : ℝ), P = 698 ∧ 
    ∃ (r : ℝ), r > 0 ∧ 
      peter_amount = P * (1 + r * peter_years) ∧
      david_amount = P * (1 + r * david_years) :=
sorry


end NUMINAMATH_CALUDE_investment_problem_l698_69801


namespace NUMINAMATH_CALUDE_powerjet_pump_volume_l698_69886

/-- The Powerjet pump rate in gallons per hour -/
def pump_rate : ℝ := 350

/-- The time period in hours -/
def time_period : ℝ := 1.5

/-- The total volume of water pumped -/
def total_volume : ℝ := pump_rate * time_period

theorem powerjet_pump_volume : total_volume = 525 := by
  sorry

end NUMINAMATH_CALUDE_powerjet_pump_volume_l698_69886


namespace NUMINAMATH_CALUDE_valid_drawing_probability_l698_69841

/-- The number of white balls in the box -/
def white_balls : ℕ := 5

/-- The number of black balls in the box -/
def black_balls : ℕ := 5

/-- The number of red balls in the box -/
def red_balls : ℕ := 1

/-- The total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls + red_balls

/-- The number of valid drawing sequences -/
def valid_sequences : ℕ := 2

/-- The probability of drawing the balls in a valid sequence -/
def probability : ℚ := valid_sequences / (Nat.factorial total_balls / (Nat.factorial white_balls * Nat.factorial black_balls * Nat.factorial red_balls))

theorem valid_drawing_probability : probability = 1 / 231 := by
  sorry

end NUMINAMATH_CALUDE_valid_drawing_probability_l698_69841


namespace NUMINAMATH_CALUDE_repeating_decimal_47_l698_69800

theorem repeating_decimal_47 : ∃ (x : ℚ), x = 47 / 99 ∧ 
  (∀ (n : ℕ), (100 * x - ⌊100 * x⌋) * 10^n = 47 / 100) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_47_l698_69800


namespace NUMINAMATH_CALUDE_triangle_problem_l698_69892

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sine_law : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C)
  (h_a : a = Real.sqrt 5)
  (h_b : b = 3)
  (h_sin_C : Real.sin C = 2 * Real.sin A) : 
  c = 2 * Real.sqrt 5 ∧ Real.sin (2 * A - Real.pi / 4) = Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l698_69892


namespace NUMINAMATH_CALUDE_computer_ownership_increase_l698_69857

/-- The percentage of families owning a personal computer in 1992 -/
def percentage_1992 : ℝ := 30

/-- The increase in the number of families owning a computer from 1992 to 1999 -/
def increase_1992_to_1999 : ℝ := 50

/-- The percentage of families owning at least one personal computer in 1999 -/
def percentage_1999 : ℝ := 45

theorem computer_ownership_increase :
  percentage_1999 = percentage_1992 * (1 + increase_1992_to_1999 / 100) := by
  sorry

end NUMINAMATH_CALUDE_computer_ownership_increase_l698_69857


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l698_69843

theorem two_digit_number_problem :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧
  ∃ k : ℤ, 3 * n - 4 = 10 * k ∧
  60 < 4 * n - 15 ∧ 4 * n - 15 < 100 ∧
  n = 28 :=
sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l698_69843


namespace NUMINAMATH_CALUDE_minimize_quadratic_l698_69894

/-- The quadratic function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- The theorem states that x = 3 minimizes the quadratic function f(x) = 3x^2 - 18x + 7 -/
theorem minimize_quadratic :
  ∃ (x_min : ℝ), x_min = 3 ∧ ∀ (x : ℝ), f x ≥ f x_min :=
sorry

end NUMINAMATH_CALUDE_minimize_quadratic_l698_69894


namespace NUMINAMATH_CALUDE_hippopotamus_crayons_l698_69829

/-- The number of crayons eaten by a hippopotamus --/
def crayonsEaten (initial final : ℕ) : ℕ := initial - final

/-- Theorem: The number of crayons eaten by the hippopotamus is the difference between 
    the initial and final number of crayons --/
theorem hippopotamus_crayons (initial final : ℕ) (h : initial ≥ final) :
  crayonsEaten initial final = initial - final := by
  sorry

/-- Given Jane's initial and final crayon counts, calculate how many were eaten --/
def janesCrayons : ℕ := 
  let initial := 87
  let final := 80
  crayonsEaten initial final

#eval janesCrayons  -- Should output 7

end NUMINAMATH_CALUDE_hippopotamus_crayons_l698_69829


namespace NUMINAMATH_CALUDE_divisibility_theorem_l698_69877

/-- The set of natural numbers m for which 3^m - 1 is divisible by 2^m -/
def S : Set ℕ := {m : ℕ | ∃ k : ℕ, 3^m - 1 = k * 2^m}

/-- The set of natural numbers m for which 31^m - 1 is divisible by 2^m -/
def T : Set ℕ := {m : ℕ | ∃ k : ℕ, 31^m - 1 = k * 2^m}

theorem divisibility_theorem :
  S = {1, 2, 4} ∧ T = {1, 2, 4, 6, 8} := by sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l698_69877


namespace NUMINAMATH_CALUDE_set_M_value_l698_69845

def M : Set ℤ := {a | ∃ (n : ℕ+), 6 / (5 - a) = n}

theorem set_M_value : M = {-1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_M_value_l698_69845


namespace NUMINAMATH_CALUDE_probability_two_zeros_not_adjacent_l698_69807

/-- The number of ways to arrange n ones and k zeros in a row -/
def totalArrangements (n k : ℕ) : ℕ :=
  Nat.choose (n + k) k

/-- The number of ways to arrange n ones and k zeros in a row where the zeros are not adjacent -/
def favorableArrangements (n k : ℕ) : ℕ :=
  Nat.choose (n + 1) k

/-- The probability that k zeros are not adjacent when arranged with n ones in a row -/
def probabilityNonAdjacentZeros (n k : ℕ) : ℚ :=
  (favorableArrangements n k : ℚ) / (totalArrangements n k : ℚ)

theorem probability_two_zeros_not_adjacent :
  probabilityNonAdjacentZeros 4 2 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_zeros_not_adjacent_l698_69807


namespace NUMINAMATH_CALUDE_consistent_coloring_pattern_l698_69811

-- Define a hexagonal board
structure HexBoard where
  size : ℕ
  traversal : List ℕ

-- Define the coloring function
def color (n : ℕ) : String :=
  if n % 3 = 0 then "Black"
  else if n % 3 = 1 then "Red"
  else "White"

-- Define a property that checks if two boards have the same coloring pattern
def sameColoringPattern (board1 board2 : HexBoard) : Prop :=
  board1.traversal.map color = board2.traversal.map color

-- Theorem statement
theorem consistent_coloring_pattern 
  (board1 board2 : HexBoard) 
  (h : board1.traversal.length = board2.traversal.length) : 
  sameColoringPattern board1 board2 := by
  sorry

end NUMINAMATH_CALUDE_consistent_coloring_pattern_l698_69811


namespace NUMINAMATH_CALUDE_parabola_properties_l698_69828

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * (x^2 - 4*x + 3)

-- Define the theorem
theorem parabola_properties (a : ℝ) (h_a : a > 0) :
  -- 1. Axis of symmetry
  (∀ x : ℝ, parabola a (2 + x) = parabola a (2 - x)) ∧
  -- 2. When PQ = QA, C is at (0, 3)
  (∃ m : ℝ, m > 2 ∧ parabola a m = 3 ∧ 3 = m - 1 → parabola a 0 = 3) ∧
  -- 3. When PQ > QA, 3 < m < 4
  (∀ m : ℝ, m > 2 ∧ parabola a m = 3 ∧ 3 > m - 1 → 3 < m ∧ m < 4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l698_69828


namespace NUMINAMATH_CALUDE_distance_A_to_yoz_l698_69879

/-- The distance from a point to the yoz plane is the absolute value of its x-coordinate. -/
def distance_to_yoz (p : ℝ × ℝ × ℝ) : ℝ :=
  |p.1|

/-- Point A with coordinates (-3, 1, -4) -/
def A : ℝ × ℝ × ℝ := (-3, 1, -4)

/-- Theorem: The distance from point A to the yoz plane is 3 -/
theorem distance_A_to_yoz : distance_to_yoz A = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_to_yoz_l698_69879


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l698_69876

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x * (x - 1) = 0}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l698_69876


namespace NUMINAMATH_CALUDE_redWhiteJellyBeansCount_l698_69881

/-- Represents the number of jelly beans of each color in one bag -/
structure JellyBeanBag where
  red : ℕ
  black : ℕ
  green : ℕ
  purple : ℕ
  yellow : ℕ
  white : ℕ

/-- Calculates the total number of red and white jelly beans in the fishbowl -/
def totalRedWhiteInFishbowl (bag : JellyBeanBag) (bagsToFill : ℕ) : ℕ :=
  (bag.red + bag.white) * bagsToFill

/-- Theorem: The total number of red and white jelly beans in the fishbowl is 126 -/
theorem redWhiteJellyBeansCount : 
  let bag : JellyBeanBag := {
    red := 24,
    black := 13,
    green := 36,
    purple := 28,
    yellow := 32,
    white := 18
  }
  let bagsToFill : ℕ := 3
  totalRedWhiteInFishbowl bag bagsToFill = 126 := by
  sorry


end NUMINAMATH_CALUDE_redWhiteJellyBeansCount_l698_69881


namespace NUMINAMATH_CALUDE_sum_of_roots_of_special_quadratic_l698_69859

-- Define a quadratic polynomial
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

-- State the theorem
theorem sum_of_roots_of_special_quadratic (a b c : ℝ) :
  (∀ x : ℝ, QuadraticPolynomial a b c (x^3 + 2*x) ≥ QuadraticPolynomial a b c (x^2 + 3)) →
  (b / a = -4 / 5) :=
by sorry

-- The sum of roots is -b/a, so if b/a = -4/5, then the sum of roots is 4/5

end NUMINAMATH_CALUDE_sum_of_roots_of_special_quadratic_l698_69859


namespace NUMINAMATH_CALUDE_shifted_linear_to_proportional_l698_69899

/-- A linear function y = ax + b -/
structure LinearFunction where
  a : ℝ
  b : ℝ

/-- Shift a linear function to the left by h units -/
def shift_left (f : LinearFunction) (h : ℝ) : LinearFunction :=
  { a := f.a, b := f.a * h + f.b }

/-- A function is directly proportional if it passes through the origin -/
def is_directly_proportional (f : LinearFunction) : Prop :=
  f.b = 0

/-- The main theorem -/
theorem shifted_linear_to_proportional (m : ℝ) : 
  let f : LinearFunction := { a := 2, b := m - 1 }
  let shifted_f := shift_left f 3
  is_directly_proportional shifted_f → m = -5 := by
sorry

end NUMINAMATH_CALUDE_shifted_linear_to_proportional_l698_69899


namespace NUMINAMATH_CALUDE_ellipse_a_value_l698_69837

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) : Type :=
  (h_pos : 0 < b ∧ b < a)

/-- The foci of an ellipse -/
def foci (e : Ellipse a b) : ℝ × ℝ := sorry

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse a b) : Type :=
  (x y : ℝ)
  (on_ellipse : x^2 / a^2 + y^2 / b^2 = 1)

/-- The area of a triangle formed by a point on the ellipse and the foci -/
def triangle_area (e : Ellipse a b) (p : PointOnEllipse e) : ℝ := sorry

/-- The tangent of the angle PF₁F₂ -/
def tan_angle_PF1F2 (e : Ellipse a b) (p : PointOnEllipse e) : ℝ := sorry

/-- The tangent of the angle PF₂F₁ -/
def tan_angle_PF2F1 (e : Ellipse a b) (p : PointOnEllipse e) : ℝ := sorry

/-- Theorem: If there exists a point P on the ellipse satisfying the given conditions, 
    then the semi-major axis a equals √15/2 -/
theorem ellipse_a_value (a b : ℝ) (e : Ellipse a b) :
  (∃ p : PointOnEllipse e, 
    triangle_area e p = 1 ∧ 
    tan_angle_PF1F2 e p = 1/2 ∧ 
    tan_angle_PF2F1 e p = -2) →
  a = Real.sqrt 15 / 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_a_value_l698_69837


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l698_69896

theorem quadratic_equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 3 ∧ x₂ = -2) ∧ 
  ((2 * x₁ - 1)^2 - 25 = 0) ∧ 
  ((2 * x₂ - 1)^2 - 25 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l698_69896


namespace NUMINAMATH_CALUDE_calculation_proof_l698_69858

theorem calculation_proof : 19 * 0.125 + 281 * (1/8) - 12.5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l698_69858


namespace NUMINAMATH_CALUDE_black_piece_position_l698_69864

-- Define the structure of a piece
structure Piece :=
  (cubes : Fin 4 → Unit)
  (shape : String)

-- Define the rectangular prism
structure RectangularPrism :=
  (pieces : Fin 4 → Piece)
  (visible : Fin 4 → Bool)
  (bottom_layer : Fin 2 → Piece)

-- Define the positions
inductive Position
  | A | B | C | D

-- Define the properties of the black piece
def is_black_piece (p : Piece) : Prop :=
  p.shape = "T" ∧ 
  (∃ (i : Fin 4), i.val = 3 → p.cubes i = ())

-- Theorem statement
theorem black_piece_position (prism : RectangularPrism) 
  (h1 : ∃ (i : Fin 4), ¬prism.visible i)
  (h2 : ∃ (i : Fin 2), is_black_piece (prism.bottom_layer i))
  (h3 : ∃ (i : Fin 2), prism.bottom_layer i = prism.pieces 3) :
  ∃ (p : Piece), is_black_piece p ∧ p = prism.pieces 2 :=
sorry

end NUMINAMATH_CALUDE_black_piece_position_l698_69864


namespace NUMINAMATH_CALUDE_lehmer_mean_properties_l698_69823

noncomputable def A (a b : ℝ) : ℝ := (a + b) / 2

noncomputable def G (a b : ℝ) : ℝ := Real.sqrt (a * b)

noncomputable def L (p a b : ℝ) : ℝ := (a^p + b^p) / (a^(p-1) + b^(p-1))

theorem lehmer_mean_properties (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  L 0.5 a b ≤ A a b ∧
  L 0 a b ≥ G a b ∧
  L 2 a b ≥ L 1 a b ∧
  ∃ n, L (n + 1) a b > L n a b :=
sorry

end NUMINAMATH_CALUDE_lehmer_mean_properties_l698_69823


namespace NUMINAMATH_CALUDE_rectangle_length_given_perimeter_and_breadth_l698_69861

/-- The perimeter of a rectangle given its length and breadth -/
def rectanglePerimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: For a rectangular garden with perimeter 500 m and breadth 100 m, the length is 150 m -/
theorem rectangle_length_given_perimeter_and_breadth :
  ∀ length : ℝ, rectanglePerimeter length 100 = 500 → length = 150 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_given_perimeter_and_breadth_l698_69861


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l698_69851

/-- Proves that for a regular polygon with an exterior angle of 36°, the sum of its interior angles is 1440°. -/
theorem regular_polygon_interior_angle_sum (n : ℕ) (ext_angle : ℝ) : 
  ext_angle = 36 → 
  n * ext_angle = 360 →
  (n - 2) * 180 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l698_69851


namespace NUMINAMATH_CALUDE_spinner_final_direction_l698_69832

/-- Represents the four cardinal directions --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a spinner with its current direction --/
structure Spinner :=
  (direction : Direction)

/-- Calculates the new direction after a given number of quarter turns clockwise --/
def new_direction_after_quarter_turns (initial : Direction) (quarter_turns : Int) : Direction :=
  sorry

/-- Converts revolutions to quarter turns --/
def revolutions_to_quarter_turns (revolutions : Rat) : Int :=
  sorry

theorem spinner_final_direction :
  let initial_spinner := Spinner.mk Direction.South
  let clockwise_turns := revolutions_to_quarter_turns (7/2)
  let counterclockwise_turns := revolutions_to_quarter_turns (9/4)
  let net_turns := clockwise_turns - counterclockwise_turns
  let final_direction := new_direction_after_quarter_turns initial_spinner.direction net_turns
  final_direction = Direction.West := by sorry

end NUMINAMATH_CALUDE_spinner_final_direction_l698_69832


namespace NUMINAMATH_CALUDE_total_length_is_6000_feet_l698_69834

/-- Represents a path on a scale drawing -/
structure ScalePath where
  length : ℝ  -- length of the path on the drawing in inches
  scale : ℝ   -- scale factor (feet represented by 1 inch)

/-- Calculates the actual length of a path in feet -/
def actualLength (path : ScalePath) : ℝ := path.length * path.scale

/-- Theorem: The total length represented by two paths on a scale drawing is 6000 feet -/
theorem total_length_is_6000_feet (path1 path2 : ScalePath)
  (h1 : path1.length = 6 ∧ path1.scale = 500)
  (h2 : path2.length = 3 ∧ path2.scale = 1000) :
  actualLength path1 + actualLength path2 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_total_length_is_6000_feet_l698_69834


namespace NUMINAMATH_CALUDE_negation_equivalence_l698_69833

theorem negation_equivalence :
  (¬ (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l698_69833


namespace NUMINAMATH_CALUDE_office_officers_count_l698_69891

/-- Represents the number of officers in an office. -/
def num_officers : ℕ := 15

/-- Represents the number of non-officers in the office. -/
def num_non_officers : ℕ := 525

/-- Represents the average salary of all employees in rupees per month. -/
def avg_salary_all : ℕ := 120

/-- Represents the average salary of officers in rupees per month. -/
def avg_salary_officers : ℕ := 470

/-- Represents the average salary of non-officers in rupees per month. -/
def avg_salary_non_officers : ℕ := 110

/-- Theorem stating that the number of officers is 15, given the conditions. -/
theorem office_officers_count :
  (num_officers * avg_salary_officers + num_non_officers * avg_salary_non_officers) / (num_officers + num_non_officers) = avg_salary_all ∧
  num_officers = 15 :=
sorry

end NUMINAMATH_CALUDE_office_officers_count_l698_69891


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l698_69863

theorem condition_necessary_not_sufficient :
  (∀ a b : ℝ, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  (∃ a b : ℝ, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l698_69863


namespace NUMINAMATH_CALUDE_inequality_solution_l698_69805

theorem inequality_solution (m n : ℝ) :
  (∀ x, 2 * m * x + 3 < 3 * x + n ↔
    ((2 * m - 3 > 0 ∧ x < (n - 3) / (2 * m - 3)) ∨
     (2 * m - 3 < 0 ∧ x > (n - 3) / (2 * m - 3)) ∨
     (m = 3 / 2 ∧ n > 3) ∨
     (m = 3 / 2 ∧ n ≤ 3 ∧ False))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l698_69805


namespace NUMINAMATH_CALUDE_wholesale_price_is_90_l698_69813

def retail_price : ℝ := 120

def discount_rate : ℝ := 0.1

def profit_rate : ℝ := 0.2

def selling_price (retail : ℝ) (discount : ℝ) : ℝ :=
  retail * (1 - discount)

def profit (wholesale : ℝ) (rate : ℝ) : ℝ :=
  wholesale * rate

theorem wholesale_price_is_90 :
  ∃ (wholesale : ℝ),
    selling_price retail_price discount_rate = wholesale + profit wholesale profit_rate ∧
    wholesale = 90 := by sorry

end NUMINAMATH_CALUDE_wholesale_price_is_90_l698_69813


namespace NUMINAMATH_CALUDE_only_sixteen_seventeen_not_divide_l698_69826

/-- A number satisfying the conditions of the problem -/
def special_number (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range 30, k + 2 ∣ n ∨ k + 3 ∣ n

/-- The theorem stating that 16 and 17 are the only consecutive numbers
    that don't divide the special number -/
theorem only_sixteen_seventeen_not_divide (n : ℕ) (h : special_number n) :
    ∃! (a : ℕ), a ∈ Finset.range 30 ∧ ¬(a + 2 ∣ n) ∧ ¬(a + 3 ∣ n) ∧ a = 14 := by
  sorry

#check only_sixteen_seventeen_not_divide

end NUMINAMATH_CALUDE_only_sixteen_seventeen_not_divide_l698_69826


namespace NUMINAMATH_CALUDE_correct_calculation_l698_69872

theorem correct_calculation (x : ℝ) : 2 * x * x^2 = 2 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l698_69872


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l698_69821

open Set

theorem union_of_A_and_complement_of_B (A B : Set ℝ) : 
  A = {x : ℝ | x^2 - 4*x - 12 < 0} →
  B = {x : ℝ | x < 2} →
  A ∪ (univ \ B) = {x : ℝ | x > -2} := by
sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l698_69821


namespace NUMINAMATH_CALUDE_circle_transformation_l698_69810

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point vertically by a given amount -/
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ := (p.1, p.2 + dy)

/-- The initial coordinates of the center of circle S -/
def initial_point : ℝ × ℝ := (3, -4)

/-- The vertical translation amount -/
def translation_amount : ℝ := 5

theorem circle_transformation :
  translate_y (reflect_x initial_point) translation_amount = (3, 9) := by
  sorry

end NUMINAMATH_CALUDE_circle_transformation_l698_69810


namespace NUMINAMATH_CALUDE_least_clock_equivalent_hour_l698_69830

theorem least_clock_equivalent_hour : ∃ (h : ℕ), 
  h > 3 ∧ 
  (∀ k : ℕ, k > 3 ∧ k < h → ¬(12 ∣ (k^2 - k))) ∧ 
  (12 ∣ (h^2 - h)) :=
by sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_hour_l698_69830


namespace NUMINAMATH_CALUDE_dave_won_ten_tickets_l698_69898

/-- Calculates the number of tickets Dave won later at the arcade --/
def tickets_won_later (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : ℕ :=
  final_tickets - (initial_tickets - spent_tickets)

/-- Proves that Dave won 10 tickets later at the arcade --/
theorem dave_won_ten_tickets :
  tickets_won_later 11 5 16 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dave_won_ten_tickets_l698_69898


namespace NUMINAMATH_CALUDE_triangle_trigonometry_l698_69839

theorem triangle_trigonometry (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  2 * a * Real.sin B = Real.sqrt 3 * b →
  Real.cos C = 5 / 13 →
  Real.sin A = Real.sqrt 3 / 2 ∧
  Real.cos B = (12 * Real.sqrt 3 - 5) / 26 := by
sorry

end NUMINAMATH_CALUDE_triangle_trigonometry_l698_69839


namespace NUMINAMATH_CALUDE_min_sum_squares_l698_69885

def S : Finset Int := {-9, -6, -3, 0, 1, 3, 6, 10}

theorem min_sum_squares (a b c d e f g h : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S) 
  (he : e ∈ S) (hf : f ∈ S) (hg : g ∈ S) (hh : h ∈ S)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
               f ≠ g ∧ f ≠ h ∧
               g ≠ h) :
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 2 ∧ 
  ∃ (a' b' c' d' e' f' g' h' : Int), 
    a' ∈ S ∧ b' ∈ S ∧ c' ∈ S ∧ d' ∈ S ∧ e' ∈ S ∧ f' ∈ S ∧ g' ∈ S ∧ h' ∈ S ∧
    a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ a' ≠ e' ∧ a' ≠ f' ∧ a' ≠ g' ∧ a' ≠ h' ∧
    b' ≠ c' ∧ b' ≠ d' ∧ b' ≠ e' ∧ b' ≠ f' ∧ b' ≠ g' ∧ b' ≠ h' ∧
    c' ≠ d' ∧ c' ≠ e' ∧ c' ≠ f' ∧ c' ≠ g' ∧ c' ≠ h' ∧
    d' ≠ e' ∧ d' ≠ f' ∧ d' ≠ g' ∧ d' ≠ h' ∧
    e' ≠ f' ∧ e' ≠ g' ∧ e' ≠ h' ∧
    f' ≠ g' ∧ f' ≠ h' ∧
    g' ≠ h' ∧
    (a' + b' + c' + d')^2 + (e' + f' + g' + h')^2 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l698_69885


namespace NUMINAMATH_CALUDE_lemonade_scaling_l698_69850

/-- Represents a lemonade recipe -/
structure LemonadeRecipe where
  lemons : ℚ
  sugar : ℚ
  gallons : ℚ

/-- The original recipe -/
def originalRecipe : LemonadeRecipe :=
  { lemons := 30
  , sugar := 5
  , gallons := 40 }

/-- Calculate the amount of an ingredient needed for a given number of gallons -/
def calculateIngredient (original : LemonadeRecipe) (ingredient : ℚ) (targetGallons : ℚ) : ℚ :=
  (ingredient / original.gallons) * targetGallons

/-- The theorem to prove -/
theorem lemonade_scaling (recipe : LemonadeRecipe) (targetGallons : ℚ) :
  let scaledLemons := calculateIngredient recipe recipe.lemons targetGallons
  let scaledSugar := calculateIngredient recipe recipe.sugar targetGallons
  recipe.gallons = 40 ∧ recipe.lemons = 30 ∧ recipe.sugar = 5 ∧ targetGallons = 10 →
  scaledLemons = 7.5 ∧ scaledSugar = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_scaling_l698_69850


namespace NUMINAMATH_CALUDE_existence_of_n_l698_69852

theorem existence_of_n (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hcd : c * d = 1) :
  ∃ n : ℕ, (a * b ≤ n^2) ∧ (n^2 ≤ (a + c) * (b + d)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l698_69852


namespace NUMINAMATH_CALUDE_walts_earnings_l698_69846

/-- Proves that Walt's total earnings from his part-time job were $9000 -/
theorem walts_earnings (interest_rate_1 interest_rate_2 : ℝ) 
  (investment_2 total_interest : ℝ) :
  interest_rate_1 = 0.09 →
  interest_rate_2 = 0.08 →
  investment_2 = 4000 →
  total_interest = 770 →
  ∃ (investment_1 : ℝ),
    interest_rate_1 * investment_1 + interest_rate_2 * investment_2 = total_interest ∧
    investment_1 + investment_2 = 9000 :=
by
  sorry

#check walts_earnings

end NUMINAMATH_CALUDE_walts_earnings_l698_69846


namespace NUMINAMATH_CALUDE_angle_C_is_pi_over_3_side_c_is_sqrt_6_l698_69816

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a + t.b = Real.sqrt 3 * t.c ∧
  2 * (Real.sin t.C)^2 = 3 * Real.sin t.A * Real.sin t.B

-- Define the area condition
def hasAreaSqrt3 (t : Triangle) : Prop :=
  1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 3

-- Theorem 1
theorem angle_C_is_pi_over_3 (t : Triangle) 
  (h : satisfiesConditions t) : t.C = π/3 :=
sorry

-- Theorem 2
theorem side_c_is_sqrt_6 (t : Triangle) 
  (h1 : satisfiesConditions t) 
  (h2 : hasAreaSqrt3 t) : t.c = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_angle_C_is_pi_over_3_side_c_is_sqrt_6_l698_69816


namespace NUMINAMATH_CALUDE_b_worked_five_days_l698_69825

/-- Represents the number of days it takes for a person to complete the entire work alone -/
def total_days : ℕ := 15

/-- Represents the number of days it takes A to complete the remaining work after B leaves -/
def remaining_days : ℕ := 5

/-- Represents the fraction of work completed by one person in one day -/
def daily_work_rate : ℚ := 1 / total_days

/-- Represents the number of days B worked before leaving -/
def days_b_worked : ℕ := sorry

theorem b_worked_five_days :
  (days_b_worked : ℚ) * (2 * daily_work_rate) + remaining_days * daily_work_rate = 1 :=
sorry

end NUMINAMATH_CALUDE_b_worked_five_days_l698_69825


namespace NUMINAMATH_CALUDE_product_of_digits_l698_69849

def is_valid_number (a b c : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧
  a + b + c = 11 ∧
  (100 * a + 10 * b + c) % 5 = 0 ∧
  a = 2 * b

theorem product_of_digits (a b c : ℕ) :
  is_valid_number a b c → a * b * c = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_l698_69849


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l698_69893

theorem equilateral_triangle_perimeter (s : ℝ) (h_positive : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l698_69893


namespace NUMINAMATH_CALUDE_fixed_point_theorem_dot_product_range_l698_69819

-- Define the curves and line
def curve_C (x y : ℝ) : Prop := y^2 = 4*x
def curve_M (x y : ℝ) : Prop := (x-1)^2 + y^2 = 4 ∧ x ≥ 1
def line_l (m n x y : ℝ) : Prop := x = m*y + n

-- Define the dot product
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

-- Part I
theorem fixed_point_theorem (m n x1 y1 x2 y2 : ℝ) :
  curve_C x1 y1 ∧ curve_C x2 y2 ∧
  line_l m n x1 y1 ∧ line_l m n x2 y2 ∧
  dot_product x1 y1 x2 y2 = -4 →
  n = 2 :=
sorry

-- Part II
theorem dot_product_range (m n x1 y1 x2 y2 : ℝ) :
  curve_C x1 y1 ∧ curve_C x2 y2 ∧
  line_l m n x1 y1 ∧ line_l m n x2 y2 ∧
  curve_M 1 0 ∧
  (∀ x y, curve_M x y → ¬(line_l m n x y ∧ (x, y) ≠ (1, 0))) →
  dot_product (x1-1) y1 (x2-1) y2 ≤ -8 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_dot_product_range_l698_69819


namespace NUMINAMATH_CALUDE_uniform_rv_expected_value_l698_69815

/-- A random variable uniformly distributed in the interval (a, b) -/
def UniformRV (a b : ℝ) : Type := ℝ

/-- The expected value of a random variable -/
def ExpectedValue (X : Type) : ℝ := sorry

/-- Theorem: The expected value of a uniformly distributed random variable -/
theorem uniform_rv_expected_value (a b : ℝ) (h : a < b) :
  ExpectedValue (UniformRV a b) = (a + b) / 2 := by sorry

end NUMINAMATH_CALUDE_uniform_rv_expected_value_l698_69815


namespace NUMINAMATH_CALUDE_spider_count_l698_69840

theorem spider_count (total_legs : ℕ) (legs_per_spider : ℕ) (h1 : total_legs = 40) (h2 : legs_per_spider = 8) :
  total_legs / legs_per_spider = 5 := by
  sorry

end NUMINAMATH_CALUDE_spider_count_l698_69840


namespace NUMINAMATH_CALUDE_surface_area_specific_parallelepiped_l698_69880

/-- The surface area of a rectangular parallelepiped with given face areas -/
def surface_area_parallelepiped (a b c : ℝ) : ℝ :=
  2 * (a + b + c)

/-- Theorem: The surface area of a rectangular parallelepiped with face areas 4, 3, and 6 is 26 -/
theorem surface_area_specific_parallelepiped :
  surface_area_parallelepiped 4 3 6 = 26 := by
  sorry

#check surface_area_specific_parallelepiped

end NUMINAMATH_CALUDE_surface_area_specific_parallelepiped_l698_69880


namespace NUMINAMATH_CALUDE_free_fall_time_l698_69855

/-- The time taken for an object to fall from a height of 490m, given the relationship h = 4.9t² -/
theorem free_fall_time : ∃ (t : ℝ), t > 0 ∧ 490 = 4.9 * t^2 ∧ t = 10 := by
  sorry

end NUMINAMATH_CALUDE_free_fall_time_l698_69855


namespace NUMINAMATH_CALUDE_sum_of_real_and_imaginary_parts_of_one_plus_i_squared_l698_69854

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_real_and_imaginary_parts_of_one_plus_i_squared (a b : ℝ) : 
  (1 + i)^2 = a + b * i → a + b = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_real_and_imaginary_parts_of_one_plus_i_squared_l698_69854


namespace NUMINAMATH_CALUDE_hyperbola_min_eccentricity_asymptote_l698_69822

/-- The asymptotic equation of a hyperbola with minimum eccentricity -/
theorem hyperbola_min_eccentricity_asymptote (m : ℝ) (h : m > 0) :
  let e := Real.sqrt (m + 4 / m + 1)
  let hyperbola := fun (x y : ℝ) => x^2 / m - y^2 / (m^2 + 4) = 1
  let asymptote := fun (x : ℝ) => (2 * x, -2 * x)
  (∀ m' > 0, e ≤ Real.sqrt (m' + 4 / m' + 1)) →
  (∃ t : ℝ, hyperbola (asymptote t).1 (asymptote t).2) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_min_eccentricity_asymptote_l698_69822


namespace NUMINAMATH_CALUDE_f_neither_odd_nor_even_l698_69875

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- Statement: f is neither odd nor even
theorem f_neither_odd_nor_even :
  (∃ x : ℝ, f (-x) ≠ f x) ∧ (∃ x : ℝ, f (-x) ≠ -f x) := by
  sorry

end NUMINAMATH_CALUDE_f_neither_odd_nor_even_l698_69875


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l698_69848

theorem polynomial_evaluation : 
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 5
  x^2 + y^2 - z^2 + 3*x*y - z = -35 := by
sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l698_69848


namespace NUMINAMATH_CALUDE_line_up_arrangements_l698_69856

def number_of_people : ℕ := 5
def number_of_youngest : ℕ := 2

theorem line_up_arrangements :
  (number_of_people.factorial - 
   (number_of_youngest * (number_of_people - 1).factorial)) = 72 :=
by sorry

end NUMINAMATH_CALUDE_line_up_arrangements_l698_69856


namespace NUMINAMATH_CALUDE_alices_age_l698_69887

theorem alices_age (alice : ℕ) (eve : ℕ) 
  (h1 : alice = 2 * eve) 
  (h2 : alice = eve + 10) : 
  alice = 20 := by
sorry

end NUMINAMATH_CALUDE_alices_age_l698_69887


namespace NUMINAMATH_CALUDE_chemistry_marks_proof_l698_69844

def english_marks : ℕ := 76
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def biology_marks : ℕ := 85
def average_marks : ℕ := 75
def total_subjects : ℕ := 5

theorem chemistry_marks_proof :
  ∃ (chemistry_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / total_subjects = average_marks ∧
    chemistry_marks = 67 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_marks_proof_l698_69844


namespace NUMINAMATH_CALUDE_game_theorems_l698_69865

/-- Game with three possible point values and their probabilities --/
structure Game where
  p : ℝ
  prob_5 : ℝ := 2 * p
  prob_10 : ℝ := p
  prob_20 : ℝ := 1 - 3 * p
  h_p_pos : 0 < p
  h_p_bound : p < 1/3
  h_prob_sum : prob_5 + prob_10 + prob_20 = 1

/-- A round consists of three games --/
def Round := Fin 3 → Game

/-- The probability of total points not exceeding 25 in one round --/
def prob_not_exceed_25 (r : Round) : ℝ := sorry

/-- The expected value of total points in one round --/
def expected_value (r : Round) : ℝ := sorry

theorem game_theorems (r : Round) (h_same_p : ∀ i j : Fin 3, (r i).p = (r j).p) :
  (∃ (p : ℝ), prob_not_exceed_25 r = 26 * p^3) ∧
  (∃ (p : ℝ), p = 1/9 → expected_value r = 140/3) :=
sorry

end NUMINAMATH_CALUDE_game_theorems_l698_69865


namespace NUMINAMATH_CALUDE_fourth_power_difference_not_prime_l698_69869

theorem fourth_power_difference_not_prime (p q : ℕ) (hp : Prime p) (hq : Prime q) (hne : p ≠ q) :
  ¬ Prime (p^4 - q^4) := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_difference_not_prime_l698_69869


namespace NUMINAMATH_CALUDE_factorization_equality_l698_69820

theorem factorization_equality (x y : ℝ) :
  (5 * x - 4 * y) * (x + 2 * y) = 5 * x^2 + 6 * x * y - 8 * y^2 := by sorry

end NUMINAMATH_CALUDE_factorization_equality_l698_69820


namespace NUMINAMATH_CALUDE_rotation_sum_l698_69866

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Represents a rotation transformation -/
structure Rotation where
  angle : ℝ
  center : Point

/-- Checks if a rotation transforms one triangle to another -/
def rotates (r : Rotation) (t1 t2 : Triangle) : Prop := sorry

theorem rotation_sum (t1 t2 : Triangle) (r : Rotation) :
  t1.p1 = Point.mk 2 2 ∧
  t1.p2 = Point.mk 2 14 ∧
  t1.p3 = Point.mk 18 2 ∧
  t2.p1 = Point.mk 32 26 ∧
  t2.p2 = Point.mk 44 26 ∧
  t2.p3 = Point.mk 32 10 ∧
  rotates r t1 t2 ∧
  0 < r.angle ∧ r.angle < 180 →
  r.angle + r.center.x + r.center.y = 124 := by
  sorry

end NUMINAMATH_CALUDE_rotation_sum_l698_69866


namespace NUMINAMATH_CALUDE_triangle_dissection_theorem_l698_69842

/-- A triangle in a 2D plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Represents a dissection of a triangle -/
def Dissection (t : Triangle) := List (List (ℝ × ℝ))

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : Triangle) : Prop := sorry

/-- Checks if one triangle is a reflection of another -/
def is_reflection (t1 t2 : Triangle) : Prop := sorry

/-- Checks if a dissection can transform one triangle to another using only translations -/
def can_transform_by_translation (d : Dissection t1) (t1 t2 : Triangle) : Prop := sorry

theorem triangle_dissection_theorem (t1 t2 : Triangle) :
  are_congruent t1 t2 → is_reflection t1 t2 →
  ∃ (d : Dissection t1), can_transform_by_translation d t1 t2 ∧ d.length ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_dissection_theorem_l698_69842


namespace NUMINAMATH_CALUDE_intersection_midpoint_l698_69836

/-- Given a straight line x - y = 2 intersecting a parabola y² = 4x at points A and B,
    the midpoint M of line segment AB has coordinates (4, 2). -/
theorem intersection_midpoint (A B M : ℝ × ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    A = (x₁, y₁) ∧ B = (x₂, y₂) ∧
    x₁ - y₁ = 2 ∧ x₂ - y₂ = 2 ∧
    y₁^2 = 4*x₁ ∧ y₂^2 = 4*x₂ ∧
    M = ((x₁ + x₂)/2, (y₁ + y₂)/2)) →
  M = (4, 2) := by
sorry


end NUMINAMATH_CALUDE_intersection_midpoint_l698_69836


namespace NUMINAMATH_CALUDE_ellipse_equation_l698_69870

/-- The standard equation of an ellipse with given foci and a point on the ellipse -/
theorem ellipse_equation (P A B : ℝ × ℝ) (h_P : P = (5/2, -3/2)) (h_A : A = (-2, 0)) (h_B : B = (2, 0)) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 ↔
    (x - A.1)^2 + (y - A.2)^2 + (x - B.1)^2 + (y - B.2)^2 = 4 * a^2 ∧
    (x - P.1)^2 + (y - P.2)^2 = ((x - A.1)^2 + (y - A.2)^2)^(1/2) * ((x - B.1)^2 + (y - B.2)^2)^(1/2)) ∧
  a^2 = 10 ∧ b^2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l698_69870


namespace NUMINAMATH_CALUDE_rook_placement_exists_l698_69814

/-- Represents a chessboard with rook placements -/
structure Chessboard (n : ℕ) :=
  (placements : Fin n → Fin n)
  (colors : Fin n → Fin n → Fin (n^2 / 2))

/-- Predicate to check if rook placements are valid -/
def valid_placements (n : ℕ) (board : Chessboard n) : Prop :=
  (∀ i j : Fin n, i ≠ j → board.placements i ≠ board.placements j) ∧
  (∀ i j : Fin n, i ≠ j → board.colors i (board.placements i) ≠ board.colors j (board.placements j))

/-- Predicate to check if the coloring is valid -/
def valid_coloring (n : ℕ) (board : Chessboard n) : Prop :=
  ∀ c : Fin (n^2 / 2), ∃! (i j k l : Fin n), 
    (i, j) ≠ (k, l) ∧ board.colors i j = c ∧ board.colors k l = c

/-- Main theorem -/
theorem rook_placement_exists (n : ℕ) (h_even : Even n) (h_gt_2 : n > 2) :
  ∃ (board : Chessboard n), valid_placements n board ∧ valid_coloring n board :=
sorry

end NUMINAMATH_CALUDE_rook_placement_exists_l698_69814


namespace NUMINAMATH_CALUDE_simplified_expansion_terms_l698_69878

/-- The number of terms in the simplified expansion of (x+y+z)^2008 + (x-y-z)^2008 -/
def num_terms : ℕ :=
  (Finset.range 1005).card + (Finset.range 1006).card

theorem simplified_expansion_terms :
  num_terms = 505815 :=
sorry

end NUMINAMATH_CALUDE_simplified_expansion_terms_l698_69878


namespace NUMINAMATH_CALUDE_earlier_usage_time_correct_l698_69895

/-- Represents a beer barrel with two taps -/
structure BeerBarrel where
  capacity : ℕ
  midwayTapRate : ℕ  -- minutes per litre
  bottomTapRate : ℕ  -- minutes per litre

/-- Calculates how much earlier the lower tap was used than usual -/
def earlierUsageTime (barrel : BeerBarrel) (usageTime : ℕ) : ℕ :=
  let drawnAmount := usageTime / barrel.bottomTapRate
  let midwayAmount := barrel.capacity / 2
  let remainingAmount := barrel.capacity - drawnAmount
  let excessAmount := remainingAmount - midwayAmount
  excessAmount * barrel.midwayTapRate

theorem earlier_usage_time_correct (barrel : BeerBarrel) (usageTime : ℕ) :
  barrel.capacity = 36 ∧ 
  barrel.midwayTapRate = 6 ∧ 
  barrel.bottomTapRate = 4 ∧ 
  usageTime = 16 →
  earlierUsageTime barrel usageTime = 84 := by
  sorry

#eval earlierUsageTime ⟨36, 6, 4⟩ 16

end NUMINAMATH_CALUDE_earlier_usage_time_correct_l698_69895


namespace NUMINAMATH_CALUDE_two_balls_different_color_weight_l698_69806

-- Define the types for color and weight
inductive Color : Type
| Red : Color
| Blue : Color

inductive Weight : Type
| Light : Weight
| Heavy : Weight

-- Define the Ball type
structure Ball :=
  (color : Color)
  (weight : Weight)

-- Define the theorem
theorem two_balls_different_color_weight 
  (balls : Set Ball)
  (h1 : ∀ b : Ball, b ∈ balls → (b.color = Color.Red ∨ b.color = Color.Blue))
  (h2 : ∀ b : Ball, b ∈ balls → (b.weight = Weight.Light ∨ b.weight = Weight.Heavy))
  (h3 : ∃ b : Ball, b ∈ balls ∧ b.color = Color.Red)
  (h4 : ∃ b : Ball, b ∈ balls ∧ b.color = Color.Blue)
  (h5 : ∃ b : Ball, b ∈ balls ∧ b.weight = Weight.Light)
  (h6 : ∃ b : Ball, b ∈ balls ∧ b.weight = Weight.Heavy)
  : ∃ b1 b2 : Ball, b1 ∈ balls ∧ b2 ∈ balls ∧ b1.color ≠ b2.color ∧ b1.weight ≠ b2.weight :=
by
  sorry

end NUMINAMATH_CALUDE_two_balls_different_color_weight_l698_69806


namespace NUMINAMATH_CALUDE_prism_volume_l698_69889

/-- The volume of a right rectangular prism given its face areas -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 30)
  (h2 : a * c = 40)
  (h3 : b * c = 60) :
  a * b * c = 120 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l698_69889


namespace NUMINAMATH_CALUDE_inequality_proof_l698_69827

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  x / (1 + x^2) + y / (1 + y^2) + z / (1 + z^2) ≤ 3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l698_69827


namespace NUMINAMATH_CALUDE_simplify_expression_l698_69835

theorem simplify_expression (a : ℝ) (h : a < (1/4)) : 4*(4*a - 1)^2 = (1 - 4*a) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l698_69835


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l698_69868

theorem lcm_from_product_and_hcf (a b : ℕ+) 
  (h_product : a * b = 82500)
  (h_hcf : Nat.gcd a b = 55) :
  Nat.lcm a b = 1500 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l698_69868


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l698_69871

-- Define the inverse variation relationship
def inverse_variation (y z : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ y^4 * z^(1/4) = k

-- State the theorem
theorem inverse_variation_problem (y z : ℝ) :
  inverse_variation y z →
  (3 : ℝ)^4 * 16^(1/4) = 6^4 * z^(1/4) →
  z = 1 / 4096 :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l698_69871


namespace NUMINAMATH_CALUDE_hugo_roll_four_given_win_l698_69812

-- Define the number of players
def num_players : ℕ := 5

-- Define the number of sides on the die
def die_sides : ℕ := 6

-- Define Hugo's winning probability
def hugo_win_prob : ℚ := 1 / num_players

-- Define the probability of rolling a 4
def roll_four_prob : ℚ := 1 / die_sides

-- Define the probability of Hugo winning given his first roll was 4
def hugo_win_given_four : ℚ := 145 / 1296

-- Theorem statement
theorem hugo_roll_four_given_win (
  num_players : ℕ) (die_sides : ℕ) (hugo_win_prob : ℚ) (roll_four_prob : ℚ) (hugo_win_given_four : ℚ) :
  num_players = 5 →
  die_sides = 6 →
  hugo_win_prob = 1 / num_players →
  roll_four_prob = 1 / die_sides →
  hugo_win_given_four = 145 / 1296 →
  (roll_four_prob * hugo_win_given_four) / hugo_win_prob = 145 / 1552 :=
by sorry

end NUMINAMATH_CALUDE_hugo_roll_four_given_win_l698_69812


namespace NUMINAMATH_CALUDE_train_speed_calculation_l698_69874

/-- Calculate the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : crossing_time = 25) :
  (train_length + bridge_length) / crossing_time = 16 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l698_69874


namespace NUMINAMATH_CALUDE_seventh_alignment_time_l698_69809

/-- Represents a standard clock with 12 divisions -/
structure Clock :=
  (divisions : Nat)
  (minute_hand_speed : Nat)
  (hour_hand_speed : Nat)

/-- Represents a time in hours and minutes -/
structure Time :=
  (hours : Nat)
  (minutes : Nat)

/-- Calculates the time until the nth alignment of clock hands -/
def time_until_nth_alignment (c : Clock) (start : Time) (n : Nat) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem seventh_alignment_time (c : Clock) (start : Time) :
  c.divisions = 12 →
  c.minute_hand_speed = 12 →
  c.hour_hand_speed = 1 →
  start.hours = 16 →
  start.minutes = 45 →
  time_until_nth_alignment c start 7 = 435 :=
sorry

end NUMINAMATH_CALUDE_seventh_alignment_time_l698_69809


namespace NUMINAMATH_CALUDE_S_is_line_l698_69882

-- Define the set of points satisfying the equation
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 5 * Real.sqrt ((p.1 - 1)^2 + (p.2 - 2)^2) = |3 * p.1 + 4 * p.2 - 11|}

-- Theorem stating that S is a line
theorem S_is_line : ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ S = {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0} :=
sorry

end NUMINAMATH_CALUDE_S_is_line_l698_69882


namespace NUMINAMATH_CALUDE_a_squared_plus_b_squared_eq_25_l698_69884

/-- Definition of the sequence S_n -/
def S (n : ℕ) : ℕ := sorry

/-- The guessed formula for S_{2n-1} -/
def S_odd (n a b : ℕ) : ℕ := (4*n - 3) * (a*n + b)

/-- Theorem stating the relation between a, b and the sequence S -/
theorem a_squared_plus_b_squared_eq_25 (a b : ℕ) :
  S 1 = 1 ∧ S 3 = 25 ∧ (∀ n, S (2*n - 1) = S_odd n a b) →
  a^2 + b^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_a_squared_plus_b_squared_eq_25_l698_69884


namespace NUMINAMATH_CALUDE_polynomial_inequality_conditions_l698_69873

/-- A polynomial function of degree 3 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The main theorem stating the conditions on a, b, and c -/
theorem polynomial_inequality_conditions
  (a b c : ℝ)
  (h : ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → f a b c (x + y) ≥ f a b c x + f a b c y) :
  a ≥ (3/2) * (9*c)^(1/3) ∧ c ≤ 0 ∧ b ∈ Set.univ :=
sorry

end NUMINAMATH_CALUDE_polynomial_inequality_conditions_l698_69873


namespace NUMINAMATH_CALUDE_rahims_average_book_price_l698_69867

/-- Calculates the average price per book given two separate book purchases -/
def average_price_per_book (books1 : ℕ) (price1 : ℕ) (books2 : ℕ) (price2 : ℕ) : ℚ :=
  (price1 + price2) / (books1 + books2)

/-- Theorem stating that the average price per book for Rahim's purchases is 20 -/
theorem rahims_average_book_price :
  average_price_per_book 50 1000 40 800 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rahims_average_book_price_l698_69867


namespace NUMINAMATH_CALUDE_compute_expression_l698_69803

theorem compute_expression : 20 * ((150 / 5) - (40 / 8) + (16 / 32) + 3) = 570 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l698_69803


namespace NUMINAMATH_CALUDE_rebus_solution_l698_69802

theorem rebus_solution : ∃! (K I S : Nat), 
  K < 10 ∧ I < 10 ∧ S < 10 ∧
  K ≠ I ∧ K ≠ S ∧ I ≠ S ∧
  100 * K + 10 * I + S + 100 * K + 10 * S + I = 100 * I + 10 * S + K := by
  sorry

end NUMINAMATH_CALUDE_rebus_solution_l698_69802


namespace NUMINAMATH_CALUDE_attendants_using_pen_pen_users_count_l698_69847

theorem attendants_using_pen (total_pencil : ℕ) (only_one_tool : ℕ) (both_tools : ℕ) : ℕ :=
  let pencil_only := total_pencil - both_tools
  let pen_only := only_one_tool - pencil_only
  pen_only + both_tools

theorem pen_users_count : attendants_using_pen 25 20 10 = 15 := by
  sorry

end NUMINAMATH_CALUDE_attendants_using_pen_pen_users_count_l698_69847


namespace NUMINAMATH_CALUDE_eliminate_denominators_l698_69804

theorem eliminate_denominators (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1) :
  (3 / (2 * x) = 1 / (x - 1)) ↔ (3 * x - 3 = 2 * x) :=
sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l698_69804


namespace NUMINAMATH_CALUDE_polynomial_product_equality_l698_69853

theorem polynomial_product_equality (x : ℝ) : 
  (1 + x^3) * (1 - 2*x + x^4) = 1 - 2*x + x^3 - x^4 + x^7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_equality_l698_69853


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_two_l698_69831

theorem negation_of_universal_positive_square_plus_two :
  (¬ ∀ x : ℝ, x^2 + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_two_l698_69831


namespace NUMINAMATH_CALUDE_divisibility_condition_l698_69860

theorem divisibility_condition (n : ℕ+) : (n^2 + 1) ∣ (n + 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l698_69860


namespace NUMINAMATH_CALUDE_max_value_of_t_l698_69888

theorem max_value_of_t (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → min a (b / (a^2 + b^2)) ≤ 1) ∧ 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ min a (b / (a^2 + b^2)) = 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_t_l698_69888


namespace NUMINAMATH_CALUDE_smallest_y_absolute_equation_l698_69838

theorem smallest_y_absolute_equation : 
  let y₁ := -46 / 5
  let y₂ := 64 / 5
  (∀ y : ℚ, |5 * y - 9| = 55 → y ≥ y₁) ∧ 
  |5 * y₁ - 9| = 55 ∧ 
  |5 * y₂ - 9| = 55 ∧
  y₁ < y₂ :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_absolute_equation_l698_69838


namespace NUMINAMATH_CALUDE_candy_sharing_l698_69824

theorem candy_sharing (hugh tommy melany : ℕ) (h1 : hugh = 8) (h2 : tommy = 6) (h3 : melany = 7) :
  (hugh + tommy + melany) / 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_sharing_l698_69824
