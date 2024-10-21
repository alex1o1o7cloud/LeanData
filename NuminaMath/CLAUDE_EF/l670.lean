import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_tax_rate_l670_67001

-- Define the tax rates and thresholds
noncomputable def tax_free_threshold : ℝ := 5000
noncomputable def first_bracket_threshold : ℝ := 30000
noncomputable def first_bracket_rate (p : ℝ) : ℝ := p + 1
noncomputable def second_bracket_rate (p : ℝ) : ℝ := p + 3

-- Define the function to calculate tax
noncomputable def calculate_tax (p : ℝ) (income : ℝ) : ℝ :=
  if income ≤ tax_free_threshold then 0
  else if income ≤ first_bracket_threshold then
    (first_bracket_rate p / 100) * (income - tax_free_threshold)
  else
    (first_bracket_rate p / 100) * (first_bracket_threshold - tax_free_threshold) +
    (second_bracket_rate p / 100) * (income - first_bracket_threshold)

-- Define James's income
noncomputable def james_income : ℝ := 180240

-- Theorem statement
theorem james_tax_rate (p : ℝ) :
  calculate_tax p james_income = (p + 0.5) / 100 * james_income := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_tax_rate_l670_67001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l670_67032

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (a + 2) * Real.log x - 2 / x + 2

/-- Theorem stating the extreme values of f(x) when a = 4 and the conditions for no zeros on (1,e) -/
theorem f_properties :
  (∃ (x : ℝ), ∀ y, f 4 x ≤ f 4 y) ∧
  (∃ (x : ℝ), ∀ y, f 4 x ≥ f 4 y) ∧
  (∀ a : ℝ, a ≤ 0 ∨ a ≥ 2 / (Real.exp 1 * (Real.exp 1 - 1)) →
    ∀ x : ℝ, 1 < x ∧ x < Real.exp 1 → f a x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l670_67032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_a_eq_zero_l670_67070

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = -x^3 + (a+1)cos(x) + x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + (a+1) * Real.cos x + x

theorem f_of_a_eq_zero (a : ℝ) (h : IsOdd (f a)) : f a a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_a_eq_zero_l670_67070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_condition_rectangle_diagonal_condition_l670_67075

/-- Represents a quadratic equation of the form x^2 - (k+1)x + (1/4)k^2 + 1 = 0 -/
def QuadraticEquation (k : ℝ) (x : ℝ) : Prop :=
  x^2 - (k+1)*x + (1/4)*k^2 + 1 = 0

/-- The discriminant of the quadratic equation -/
noncomputable def Discriminant (k : ℝ) : ℝ :=
  (k+1)^2 - 4*((1/4)*k^2 + 1)

theorem quadratic_roots_condition (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ QuadraticEquation k x₁ ∧ QuadraticEquation k x₂) ↔ k ≥ 3/2 := by
  sorry

theorem rectangle_diagonal_condition (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ QuadraticEquation k x₁ ∧ QuadraticEquation k x₂ ∧ x₁^2 + x₂^2 = 5) →
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_condition_rectangle_diagonal_condition_l670_67075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_approx_l670_67095

-- Define the initial loan amount
def initial_loan : ℚ := 15000

-- Define the interest rates
def rate1 : ℚ := 1 / 10  -- 10% for Plan 1
def rate2 : ℚ := 3 / 25  -- 12% for Plan 2

-- Define the compounding frequencies
def n1 : ℕ := 2  -- biannually for Plan 1
def n2 : ℕ := 1  -- annually for Plan 2

-- Define the loan term in years
def term : ℕ := 10

-- Function to calculate compound interest
noncomputable def compound_interest (principal : ℚ) (rate : ℚ) (n : ℕ) (t : ℕ) : ℚ :=
  principal * (1 + rate / n) ^ (n * t)

-- Calculate the balance after 5 years in Plan 1
noncomputable def balance_5_years : ℚ := compound_interest initial_loan rate1 n1 5

-- Calculate the balance after 7 years in Plan 1
noncomputable def balance_7_years : ℚ := compound_interest (balance_5_years / 2) rate1 n1 2

-- Calculate the final balance in Plan 1
noncomputable def final_balance_plan1 : ℚ := compound_interest (balance_7_years * 2 / 3) rate1 n1 3

-- Calculate the total payment in Plan 1
noncomputable def total_payment_plan1 : ℚ := (balance_5_years / 2) + (balance_7_years / 3) + final_balance_plan1

-- Calculate the total payment in Plan 2
noncomputable def total_payment_plan2 : ℚ := compound_interest initial_loan rate2 n2 term

-- Theorem statement
theorem payment_difference_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |total_payment_plan2 - total_payment_plan1 - 16155| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_approx_l670_67095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_triangular_prism_l670_67042

/-- Represents a triangular prism with base sides a and b, included angle θ, and height h -/
structure TriangularPrism where
  a : ℝ
  b : ℝ
  θ : ℝ
  h : ℝ

/-- Conditions for the triangular prism -/
def PrismConditions (p : TriangularPrism) : Prop :=
  0 < p.a ∧ 0 < p.b ∧ 0 < p.θ ∧ 0 < p.h ∧
  p.θ ≠ Real.pi / 2 ∧
  p.a * p.b * Real.sin p.θ + p.a * p.h = 36

/-- Volume of the triangular prism -/
noncomputable def volume (p : TriangularPrism) : ℝ :=
  1/2 * p.a * p.b * p.h * Real.sin p.θ

/-- Theorem stating the maximum volume of the triangular prism -/
theorem max_volume_triangular_prism (p : TriangularPrism) 
  (h : PrismConditions p) : 
  ∃ (max_vol : ℝ), volume p ≤ max_vol ∧ max_vol = 9 * p.a * Real.sin p.θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_triangular_prism_l670_67042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_condition_l670_67058

-- Define the circle
def circle_center : ℝ × ℝ := (2, 1)
noncomputable def circle_radius : ℝ := Real.sqrt 10

-- Define the points P and Q
def point_P : ℝ × ℝ := (1, -2)
def point_Q : ℝ × ℝ := (3, 4)

-- Define the line y = 2x + b
def line (b : ℝ) (x : ℝ) : ℝ := 2 * x + b

-- Define the distance from a point to the line
noncomputable def distance_to_line (b : ℝ) (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p
  abs (2 * x - y + b) / Real.sqrt 5

-- Main theorem
theorem chord_length_condition (b : ℝ) : 
  (distance_to_line b circle_center)^2 + 5 = circle_radius^2 ↔ b = 2 ∨ b = -8 := by
  sorry

#check chord_length_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_condition_l670_67058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloons_given_l670_67053

theorem balloons_given (initial : ℝ) (final : ℕ) (given : ℕ) : 
  initial = 7.0 → final = 12 → given = final - Int.floor initial := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloons_given_l670_67053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l670_67011

/-- The slope of a line given by the equation 4y - 8x = 16 is 2 -/
theorem slope_of_line (x y x' y' : ℝ) (h : x ≠ x') :
  4 * y - 8 * x = 16 → (y - y') / (x - x') = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l670_67011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l670_67081

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

-- Define the derivative of f
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 2 * a * x - (a + 2) + 1 / x

-- Theorem for part 1
theorem part_one (a : ℝ) : f_prime a 1 = 0 → a = 1 := by sorry

-- Theorem for part 2
theorem part_two (a : ℝ) (x : ℝ) (h1 : a ≥ 1) (h2 : 1 ≤ x) (h3 : x ≤ Real.exp 1) :
  f_prime a x ≥ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l670_67081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_is_six_l670_67038

/-- A right triangle with given side lengths and its incenter -/
structure RightTriangleWithIncenter where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  is_right_triangle : PQ^2 + PR^2 = QR^2
  pq_positive : PQ > 0
  pr_positive : PR > 0
  qr_positive : QR > 0

/-- The length of the angle bisector from P to the incenter I -/
noncomputable def angle_bisector_length (t : RightTriangleWithIncenter) : ℝ :=
  (t.PQ + t.PR - t.QR) / 2

/-- Theorem: In the given right triangle, PI = 6 -/
theorem angle_bisector_length_is_six (t : RightTriangleWithIncenter) 
    (h1 : t.PQ = 20) (h2 : t.PR = 21) (h3 : t.QR = 29) : 
    angle_bisector_length t = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_is_six_l670_67038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l670_67016

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + 2 * Real.pi / 3)

theorem min_value_of_f :
  ∃ (min : ℝ), min = -Real.sqrt 3 ∧
  ∀ x, x ∈ Set.Icc (-Real.pi / 2) 0 → f x ≥ min := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l670_67016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l670_67019

open Set
open Function

-- Define the function f on the real numbers
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x > 0}

-- State the condition on f and its second derivative
axiom f_condition (x : ℝ) (h : x ∈ domain_f) : f x + x * (deriv^[2] f x) > 0

-- State the theorem
theorem inequality_equivalence :
  ∀ x ∈ domain_f, ((x - 1) * f (x^2 - 1) < f (x + 1)) ↔ (1 < x ∧ x < 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l670_67019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_perimeter_225_deg_14_cm_l670_67047

/-- The perimeter of a circular sector with given radius and central angle -/
noncomputable def sectorPerimeter (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  2 * radius + (centralAngle / 360) * 2 * Real.pi * radius

/-- Theorem: The perimeter of a circular sector with radius 14 cm and central angle 225 degrees -/
theorem sector_perimeter_225_deg_14_cm :
  sectorPerimeter 14 225 = 28 + 17.5 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_perimeter_225_deg_14_cm_l670_67047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_production_increase_l670_67002

/-- The monthly increase in salt production for a company -/
noncomputable def monthly_increase (jan_prod : ℝ) (avg_daily_prod : ℝ) : ℝ :=
  let total_prod := avg_daily_prod * 365
  let sum_formula := λ x ↦ 12 / 2 * (2 * jan_prod + (12 - 1) * x)
  (total_prod - sum_formula 0) / (12 * 11 / 2)

/-- Theorem stating the monthly increase in salt production -/
theorem salt_production_increase 
  (jan_prod : ℝ) 
  (avg_daily_prod : ℝ) 
  (h_jan : jan_prod = 3000) 
  (h_avg : avg_daily_prod = 116.71232876712328) : 
  monthly_increase jan_prod avg_daily_prod = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_production_increase_l670_67002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_yes_answers_is_21_yes_answers_21_achievable_l670_67029

/-- Represents a row of people -/
structure Row where
  knights : Nat
  liars : Nat
  total : knights + liars = 5

/-- Represents the arrangement of people -/
structure Arrangement where
  rows : Finset Row
  total_rows : rows.card = 6
  total_people : (rows.sum fun r => r.knights + r.liars) = 30

/-- Defines a blue row (more than half are liars) -/
def isBlueRow (r : Row) : Bool := r.liars > r.knights

/-- The question asked by the journalist -/
def journalistQuestion (a : Arrangement) : Bool :=
  (a.rows.filter (fun r => isBlueRow r)).card ≥ 4

/-- A function that counts the maximum number of "yes" answers -/
def maxYesAnswers (a : Arrangement) : Nat :=
  max
    (a.rows.sum fun r => if isBlueRow r then r.liars else r.knights)
    (a.rows.sum fun r => if not (journalistQuestion a) then r.liars else r.knights)

/-- The main theorem to prove -/
theorem max_yes_answers_is_21 (a : Arrangement) : maxYesAnswers a ≤ 21 := by
  sorry

/-- The theorem that 21 is achievable -/
theorem yes_answers_21_achievable : ∃ a : Arrangement, maxYesAnswers a = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_yes_answers_is_21_yes_answers_21_achievable_l670_67029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_overlapping_square_l670_67049

/-- Represents a square on the grid -/
structure Square where
  x : Fin 8
  y : Fin 8
  size : Nat
  h_size : size > 0

/-- Checks if two squares overlap -/
def overlaps (s1 s2 : Square) : Prop :=
  ∃ (x y : Fin 8), x ≥ s1.x ∧ x < s1.x + s1.size ∧ y ≥ s1.y ∧ y < s1.y + s1.size ∧
                   x ≥ s2.x ∧ x < s2.x + s2.size ∧ y ≥ s2.y ∧ y < s2.y + s2.size

/-- A list of 8 non-overlapping 2x2 squares -/
def colored_squares : List Square := sorry

/-- The theorem to be proved -/
theorem exists_non_overlapping_square :
  ∃ (s : Square), s.size = 2 ∧ ∀ (cs : Square), cs ∈ colored_squares → ¬overlaps s cs :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_overlapping_square_l670_67049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_half_l670_67045

def sequence_a : ℕ → ℚ
  | 0 => 1/2  -- Add this case for 0
  | 1 => 1/2
  | (n+1) => 1 / (1 - sequence_a n)

theorem a_2017_equals_half : sequence_a 2017 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_half_l670_67045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_shaded_area_coefficient_sum_l670_67054

/-- Represents a grid with circles placed on top --/
structure GridWithCircles where
  gridSize : ℕ
  squareSize : ℝ
  smallCircleCount : ℕ
  smallCircleRadius : ℝ
  largeCircleCount : ℕ
  largeCircleRadius : ℝ

/-- Calculates the total area of the grid --/
def totalGridArea (g : GridWithCircles) : ℝ :=
  (g.gridSize * g.squareSize) ^ 2

/-- Calculates the total area covered by circles --/
noncomputable def totalCircleArea (g : GridWithCircles) : ℝ :=
  g.smallCircleCount * Real.pi * g.smallCircleRadius ^ 2 +
  g.largeCircleCount * Real.pi * g.largeCircleRadius ^ 2

/-- Calculates the visible shaded area coefficients --/
noncomputable def visibleShadedAreaCoefficients (g : GridWithCircles) : ℝ × ℝ :=
  (totalGridArea g, totalCircleArea g / Real.pi)

/-- The main theorem --/
theorem visible_shaded_area_coefficient_sum :
  let g : GridWithCircles := {
    gridSize := 4,
    squareSize := 2,
    smallCircleCount := 4,
    smallCircleRadius := 1,
    largeCircleCount := 2,
    largeCircleRadius := 1.5
  }
  let (A, B) := visibleShadedAreaCoefficients g
  A + B = 72.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_shaded_area_coefficient_sum_l670_67054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_cubed_coeff_is_zero_x_cubed_coeff_is_zero_explicit_l670_67013

/-- Given a polynomial q(x) = x^4 - 4x^2 + 3, 
    the coefficient of x^3 in (q(x))^3 is 0 -/
theorem x_cubed_coeff_is_zero (x : ℝ) : 
  let q : ℝ → ℝ := fun x ↦ x^4 - 4*x^2 + 3
  ∃ a b c d : ℝ, (q x)^3 = a*x^12 + b*x^8 + c*x^4 + d := by
  sorry

/-- The coefficient of x^3 in (q(x))^3 is indeed 0 -/
theorem x_cubed_coeff_is_zero_explicit : 
  let q : ℝ → ℝ := fun x ↦ x^4 - 4*x^2 + 3
  ∀ x : ℝ, ∃ a b c d : ℝ, (q x)^3 = a*x^12 + b*x^8 + c*x^4 + d ∧ 
  (∀ e : ℝ, (q x)^3 ≠ a*x^12 + b*x^8 + c*x^4 + d + e*x^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_cubed_coeff_is_zero_x_cubed_coeff_is_zero_explicit_l670_67013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_properties_l670_67009

-- Define the logarithmic function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 9 = 2) :
  (a = 3) ∧
  (∀ x : ℝ, f a (x + 1) < 1 ↔ -1 < x ∧ x < 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_properties_l670_67009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_empty_squares_l670_67022

/-- Represents a chessboard with cockroaches -/
structure Chessboard :=
  (size : Nat)
  (initial_cockroaches : Nat)

/-- Represents a movement of cockroaches on the chessboard -/
def valid_movement (board : Chessboard) : Prop :=
  board.size = 8 ∧
  board.initial_cockroaches = 2 ∧
  ∀ (x y : Nat), x < board.size → y < board.size →
    ∃ (new_x new_y : Nat),
      (new_x < board.size ∧ new_y < board.size) ∧
      ((new_x = x + 1 ∧ new_y = y) ∨
       (new_x = x - 1 ∧ new_y = y) ∨
       (new_x = x ∧ new_y = y + 1) ∨
       (new_x = x ∧ new_y = y - 1))

/-- Function to calculate the number of empty squares after movement -/
def number_of_empty_squares (board : Chessboard) (movement : valid_movement board) : Nat :=
  sorry

/-- The theorem stating the maximum number of empty squares after movement -/
theorem max_empty_squares (board : Chessboard) :
  valid_movement board →
  ∃ (empty_squares : Nat), empty_squares ≤ 24 ∧
    ∀ (n : Nat), (∃ (movement : valid_movement board),
      number_of_empty_squares board movement = n) →
      n ≤ empty_squares :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_empty_squares_l670_67022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_in_cone_and_cylinder_l670_67036

theorem sphere_volume_in_cone_and_cylinder (cone_base_diameter : ℝ) 
  (cone_vertex_angle : ℝ) (cylinder_diameter : ℝ) (cylinder_height : ℝ) :
  cone_base_diameter = 16 →
  cone_vertex_angle = 45 →
  cylinder_diameter = cone_base_diameter / 2 →
  cylinder_height = cone_base_diameter / 2 →
  (4 / 3) * Real.pi * ((cylinder_diameter / 2) ^ 3) = (256 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_in_cone_and_cylinder_l670_67036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_equal_diagonals_rhombus_perpendicular_diagonals_rectangle_circumscribable_rhombus_inscribable_l670_67076

/-- A structure representing a parallelogram -/
structure Parallelogram where
  /-- The length of one pair of parallel sides -/
  side1 : ℝ
  /-- The length of the other pair of parallel sides -/
  side2 : ℝ
  /-- The angle between adjacent sides (in radians) -/
  angle : ℝ
  /-- Assumption that side lengths are positive -/
  side1_pos : side1 > 0
  /-- Assumption that side lengths are positive -/
  side2_pos : side2 > 0
  /-- Assumption that the angle is between 0 and π -/
  angle_range : 0 < angle ∧ angle < Real.pi

/-- A function to determine if a parallelogram is a rectangle -/
def is_rectangle (p : Parallelogram) : Prop :=
  p.angle = Real.pi / 2

/-- A function to determine if a parallelogram is a rhombus -/
def is_rhombus (p : Parallelogram) : Prop :=
  p.side1 = p.side2

/-- A function to determine if a parallelogram is a square -/
def is_square (p : Parallelogram) : Prop :=
  is_rectangle p ∧ is_rhombus p

/-- Theorem stating that rectangles have equal diagonals -/
theorem rectangle_equal_diagonals (p : Parallelogram) :
  is_rectangle p → ∃ d : ℝ, d > 0 ∧ d^2 = p.side1^2 + p.side2^2 := by
  sorry

/-- Theorem stating that rhombi have perpendicular diagonals -/
theorem rhombus_perpendicular_diagonals (p : Parallelogram) :
  is_rhombus p → ∃ d1 d2 : ℝ, d1 > 0 ∧ d2 > 0 ∧ d1 * d2 = 2 * p.side1^2 * Real.sin p.angle := by
  sorry

/-- Theorem stating that only rectangles among parallelograms can be circumscribed by a circle -/
theorem rectangle_circumscribable (p : Parallelogram) :
  (∃ r : ℝ, r > 0 ∧ r^2 = (p.side1^2 + p.side2^2) / 4) → is_rectangle p := by
  sorry

/-- Theorem stating that only rhombi among parallelograms can be inscribed in a circle -/
theorem rhombus_inscribable (p : Parallelogram) :
  (∃ r : ℝ, r > 0 ∧ r = p.side1 * Real.sin (p.angle / 2)) → is_rhombus p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_equal_diagonals_rhombus_perpendicular_diagonals_rectangle_circumscribable_rhombus_inscribable_l670_67076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_function_zero_l670_67061

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A function assigning a real number to each point in a plane -/
def PlaneFunction := Point → ℝ

/-- The centroid (intersection of medians) of a triangle -/
noncomputable def centroid (A B C : Point) : Point :=
  { x := (A.x + B.x + C.x) / 3, y := (A.y + B.y + C.y) / 3 }

/-- Theorem: If f(M) = f(A) + f(B) + f(C) for any triangle ABC where M is its centroid,
    then f(A) = 0 for all points A -/
theorem plane_function_zero (f : PlaneFunction) :
  (∀ A B C : Point, f (centroid A B C) = f A + f B + f C) →
  (∀ A : Point, f A = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_function_zero_l670_67061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_water_tower_height_scaled_height_approx_l670_67004

/-- Represents the height of a water tower model given the original height, original volume, and model volume -/
noncomputable def scaled_height (original_height : ℝ) (original_volume : ℝ) (model_volume : ℝ) : ℝ :=
  original_height * (model_volume / original_volume) ^ (1/3)

/-- Theorem stating that a scaled model of a 40m water tower holding 100,000L, 
    when scaled to hold 0.1L, should be 0.4m tall -/
theorem scaled_water_tower_height :
  scaled_height 40 100000 0.1 = 0.4 := by
  sorry

/-- Approximate evaluation of the scaled height -/
theorem scaled_height_approx :
  ∃ ε > 0, |scaled_height 40 100000 0.1 - 0.4| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_water_tower_height_scaled_height_approx_l670_67004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_configuration_exists_l670_67091

/-- A collection of unordered triples is n-admissible if for each 1 ≤ k ≤ n - 2 and 
    each choice of k distinct elements from the collection, 
    the union of these elements has at least k+2 elements. -/
def IsNAdmissible (n : ℕ) (S : Finset (Finset ℕ)) : Prop :=
  ∀ (k : ℕ) (A : Finset (Finset ℕ)), 1 ≤ k → k ≤ n - 2 → A ⊆ S → A.card = k →
    (A.biUnion id).card ≥ k + 2

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The angle between three points -/
noncomputable def angle (A B C : Point) : ℝ := sorry

/-- The statement of the problem -/
theorem configuration_exists (n : ℕ) (S : Finset (Finset ℕ)) 
    (hn : n > 3) (hS : IsNAdmissible n S) (hScard : S.card = n - 2) :
  ∃ (P : ℕ → Point), (∀ i j, i ≠ j → P i ≠ P j) ∧
    (∀ t ∈ S, ∀ i j k, i ∈ t → j ∈ t → k ∈ t → angle (P i) (P j) (P k) < 61 * π / 180) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_configuration_exists_l670_67091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_primitive_root_of_unity_l670_67014

open Complex

/-- A complex number z is a primitive nth root of unity if z^n = 1 and z^k ≠ 1 for all 1 ≤ k < n -/
def isPrimitiveRoot (z : ℂ) (n : ℕ) : Prop :=
  z^n = 1 ∧ ∀ k : ℕ, 1 ≤ k → k < n → z^k ≠ 1

/-- The nth roots of unity function -/
noncomputable def nthRoot (n : ℕ) (m : ℤ) : ℂ :=
  Complex.exp (2 * Real.pi * I * (m : ℂ) / (n : ℂ))

theorem primitive_root_of_unity (n : ℕ) (m : ℤ) (h : Nat.Coprime n m.natAbs) :
  isPrimitiveRoot (nthRoot n m) n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_primitive_root_of_unity_l670_67014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_over_b_l670_67034

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_decreasing : ∀ x y : ℝ, 0 ≤ x ∧ x < y → f y < f x

-- Define the inequality condition
axiom inequality_condition : ∀ a b : ℝ, a > 0 ∧ b > 0 → 
  f (Real.log (a / b)) + f (Real.log (b / a)) - 2 * f 1 < 0

-- Theorem statement
theorem range_of_a_over_b :
  ∀ a b : ℝ, a > 0 ∧ b > 0 →
  (f (Real.log (a / b)) + f (Real.log (b / a)) - 2 * f 1 < 0) →
  (a / b ∈ Set.Ioo 0 (1 / Real.exp 1) ∪ Set.Ioi (Real.exp 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_over_b_l670_67034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncool_parents_count_l670_67057

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) 
  (h1 : total = 40)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 20)
  (h4 : both_cool = 10) :
  total - (cool_dads + cool_moms - both_cool) = 12 := by
  sorry

#check uncool_parents_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncool_parents_count_l670_67057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_consumption_proof_l670_67068

noncomputable def original_wattages : List ℝ := [60, 80, 100, 120]
def increase_factor : ℝ := 1.25
def hours_per_day : ℝ := 6
def days_per_month : ℝ := 30

noncomputable def new_wattages : List ℝ := original_wattages.map (· * increase_factor)

noncomputable def combined_wattage : ℝ := new_wattages.sum

noncomputable def kilowatts : ℝ := combined_wattage / 1000

noncomputable def energy_consumption : ℝ := kilowatts * hours_per_day * days_per_month

theorem energy_consumption_proof : energy_consumption = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_consumption_proof_l670_67068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sine_inequality_l670_67007

theorem negation_of_sine_inequality :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sine_inequality_l670_67007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_range_l670_67046

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then (2 - a/2) * x + 2 else a^(x-1)

theorem f_increasing_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 3 ≤ a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_range_l670_67046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_periodic_no_smallest_period_infinite_values_l670_67027

-- Define the property of being periodic
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x : ℝ, f (x + p) = f x

-- Define the property of having no smallest period
def HasNoSmallestPeriod (f : ℝ → ℝ) : Prop :=
  ∀ p : ℝ, IsPeriodic f → ∃ q : ℝ, 0 < q ∧ q < p ∧ IsPeriodic f

-- Define the property of taking infinitely many different values
def TakesInfinitelyManyValues (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ (S : Finset ℝ), S.card = n ∧ (∀ x y, x ∈ S → y ∈ S → x ≠ y → f x ≠ f y)

-- State the theorem
theorem exists_periodic_no_smallest_period_infinite_values :
  ∃ f : ℝ → ℝ, IsPeriodic f ∧ HasNoSmallestPeriod f ∧ TakesInfinitelyManyValues f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_periodic_no_smallest_period_infinite_values_l670_67027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_agreement_ratio_rounded_is_point_eight_l670_67048

/-- The ratio of agreement required for a proposal to pass in a committee --/
def agreement_ratio : ℚ := 15 / 20

/-- Rounding a rational number to the nearest tenth --/
def round_to_nearest_tenth (q : ℚ) : ℚ :=
  (q * 10).floor / 10 + if (q * 10 - (q * 10).floor ≥ 1/2) then 1/10 else 0

/-- Theorem stating that the agreement ratio rounded to the nearest tenth is 0.8 --/
theorem agreement_ratio_rounded_is_point_eight :
  round_to_nearest_tenth agreement_ratio = 4/5 := by
  sorry

#eval round_to_nearest_tenth agreement_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_agreement_ratio_rounded_is_point_eight_l670_67048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l670_67000

/-- A regular polygon with perimeter 180 and side length 15 has 12 sides -/
theorem regular_polygon_sides (p : ℕ) (perimeter side_length : ℝ) 
  (h_perimeter : perimeter = 180)
  (h_side_length : side_length = 15)
  (h_regular : p * side_length = perimeter) : 
  p = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l670_67000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_zero_l670_67050

/-- The function f(x) = (x+a)e^x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.exp x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := (x + a + 1) * Real.exp x

/-- The slope of the tangent line to f(x) at x=0 -/
noncomputable def tangent_slope (a : ℝ) : ℝ := f_derivative a 0

/-- The slope of the line x+y+1=0 -/
def perpendicular_line_slope : ℝ := -1

theorem tangent_perpendicular_implies_a_zero (a : ℝ) :
  (tangent_slope a) * perpendicular_line_slope = -1 → a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_zero_l670_67050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_integer_iterate_f_l670_67089

noncomputable def ceiling (x : ℝ) : ℤ := Int.ceil x

noncomputable def f (r : ℝ) : ℝ := r * (ceiling r)

noncomputable def iterate_f (r : ℝ) : ℕ → ℝ
  | 0 => r
  | n + 1 => f (iterate_f r n)

theorem exists_integer_iterate_f (k : ℤ) (hk : k > 0) :
  ∃ m : ℕ, ∃ n : ℤ, iterate_f (↑k + 0.5) m = ↑n := by
  sorry

#check exists_integer_iterate_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_integer_iterate_f_l670_67089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_250_l670_67067

/-- The length of a bridge crossed by a train -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Theorem: The length of the bridge is 250 meters -/
theorem bridge_length_is_250 :
  bridge_length 125 45 30 = 250 := by
  -- Unfold the definition of bridge_length
  unfold bridge_length
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_250_l670_67067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_square_root_equality_l670_67088

theorem nested_square_root_equality : 
  Real.sqrt (25 * Real.sqrt (16 * Real.sqrt 9)) = 10 * (3 : Real).rpow (1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_square_root_equality_l670_67088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l670_67037

-- Define the nabla operation as noncomputable
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Theorem statement
theorem nabla_calculation :
  nabla 1 (nabla 2 (nabla 3 4)) = 1 := by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l670_67037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_buffy_breath_holding_time_l670_67021

/-- Represents the time in seconds that a person can hold their breath underwater. -/
def BreathHoldingTime : Type := ℕ

/-- Kelly's breath-holding time in minutes. -/
def kelly_time_minutes : ℕ := 3

/-- Converts minutes to seconds. -/
def minutes_to_seconds (minutes : ℕ) : ℕ := minutes * 60

/-- The difference in seconds between Kelly's and Brittany's breath-holding times. -/
def kelly_brittany_diff : ℕ := 20

/-- The difference in seconds between Brittany's and Buffy's breath-holding times. -/
def brittany_buffy_diff : ℕ := 40

/-- Calculates Buffy's breath-holding time given the conditions. -/
def buffy_time : ℕ :=
  minutes_to_seconds kelly_time_minutes - kelly_brittany_diff - brittany_buffy_diff

theorem buffy_breath_holding_time :
  buffy_time = (120 : ℕ) := by
  -- Unfold definitions
  unfold buffy_time
  unfold minutes_to_seconds
  unfold kelly_time_minutes
  -- Perform calculation
  simp [Nat.mul_comm]
  -- The proof is complete
  rfl

#eval buffy_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_buffy_breath_holding_time_l670_67021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_sum_l670_67012

/-- Represents a rectangular box with given dimensions --/
structure Box where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the area of the triangle formed by center points of three faces meeting at a corner --/
noncomputable def triangleArea (b : Box) : ℝ :=
  let side1 := b.length / 2
  let side2 := Real.sqrt ((b.height / 2) ^ 2 + (b.width / 2) ^ 2)
  let side3 := Real.sqrt ((b.height / 2) ^ 2 + (b.length / 2) ^ 2)
  let s := (side1 + side2 + side3) / 2
  Real.sqrt (s * (s - side1) * (s - side2) * (s - side3))

/-- Main theorem --/
theorem box_dimensions_sum (m n : ℕ) (h_coprime : Nat.Coprime m n) :
  let b := Box.mk 10 20 (m / n : ℝ)
  triangleArea b = 40 → m + n = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_sum_l670_67012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l670_67062

def sequenceRec (a₀ a₁ a₂ : ℕ) : ℕ → ℕ
| 0 => a₀
| 1 => a₁
| 2 => a₂
| n + 3 => Int.natAbs (sequenceRec a₀ a₁ a₂ (n + 2) - sequenceRec a₀ a₁ a₂ n)

theorem sequence_properties (a₀ a₁ a₂ : ℕ) (h : a₀ + a₁ + a₂ > 0) :
  ∃ (c : ℕ), ∀ (i : ℕ), sequenceRec a₀ a₁ a₂ i ≤ c ∧
  ∃ (p : ℕ), p > 0 ∧ ∀ (n : ℕ), ∃ (m : ℕ), m ≥ n ∧ ∀ (k : ℕ),
    sequenceRec a₀ a₁ a₂ (m + k) = sequenceRec a₀ a₁ a₂ (m + k + p) ∧
  ∀ (b₀ b₁ b₂ : ℕ) (h' : b₀ + b₁ + b₂ > 0),
    ∃ (q : ℕ), q > 0 ∧ ∀ (n : ℕ), ∃ (m : ℕ), m ≥ n ∧ ∀ (k : ℕ),
      sequenceRec b₀ b₁ b₂ (m + k) = sequenceRec b₀ b₁ b₂ (m + k + q) ∧ q = p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l670_67062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l670_67040

noncomputable section

-- Define the expression
def f (x : ℝ) : ℝ := Real.sqrt (2 - x) / (x + 3)

-- Define the set of valid x values
def valid_x : Set ℝ := {x | x ≤ 2 ∧ x ≠ -3}

-- Theorem statement
theorem range_of_f : 
  {x : ℝ | ∃ y, f x = y ∧ y ∈ Set.univ} = valid_x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l670_67040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l670_67008

/-- The eccentricity of a hyperbola with given equation and asymptote -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b / a = 3 / 4) :
  Real.sqrt ((a^2 + b^2) / a^2) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l670_67008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_perimeter_hexagon_plane_l670_67005

/-- A cube in 3D space -/
structure Cube where
  -- Add necessary fields for a cube
  side : ℝ

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields for a plane
  normal : Point
  d : ℝ

/-- A line in 3D space -/
structure Line where
  direction : Point
  point : Point

/-- Represents a space diagonal of a cube -/
noncomputable def spaceDiagonal (c : Cube) : Line :=
  { direction := { x := 1, y := 1, z := 1 },
    point := { x := 0, y := 0, z := 0 } }

/-- Checks if a point is on the edge of a cube -/
def isOnEdge (p : Point) (c : Cube) : Prop :=
  sorry

/-- Checks if a plane intersects a cube to form a hexagon -/
def formsHexagon (pl : Plane) (c : Cube) : Prop :=
  sorry

/-- Calculates the perimeter of the hexagonal cross-section -/
noncomputable def hexagonPerimeter (pl : Plane) (c : Cube) : ℝ :=
  sorry

/-- Checks if a plane is perpendicular to a line -/
def isPerpendicularTo (pl : Plane) (l : Line) : Prop :=
  sorry

theorem minimal_perimeter_hexagon_plane
  (c : Cube) (p : Point) (h : isOnEdge p c) :
  ∃ (pl : Plane),
    formsHexagon pl c ∧
    (∀ (pl' : Plane), formsHexagon pl' c →
      hexagonPerimeter pl c ≤ hexagonPerimeter pl' c) ∧
    isPerpendicularTo pl (spaceDiagonal c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_perimeter_hexagon_plane_l670_67005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_integers_with_remainder_three_l670_67033

theorem two_digit_integers_with_remainder_three : 
  (Finset.filter (fun n => n ≥ 10 ∧ n < 100 ∧ n % 9 = 3) (Finset.range 100)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_integers_with_remainder_three_l670_67033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_x_and_z_l670_67051

/-- Represents the time taken to complete a work -/
def WorkTime := ℝ

/-- Represents the rate of work (proportion of work done per unit time) -/
def WorkRate := ℝ

theorem work_time_x_and_z 
  (time_x : WorkTime) 
  (time_yz : WorkTime) 
  (time_y : WorkTime) 
  (h1 : time_x = (8 : ℝ))
  (h2 : time_yz = (6 : ℝ))
  (h3 : time_y = (24 : ℝ)) :
  ∃ (time_xz : WorkTime), time_xz = (4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_x_and_z_l670_67051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_with_conditions_l670_67015

theorem min_sum_with_conditions (a b : ℕ) 
  (ha : a > 0)
  (hb : b > 0)
  (h1 : Nat.gcd (a + b) 330 = 1)
  (h2 : ∃ k : ℕ, a^a = k * b^b)
  (h3 : ¬ ∃ m : ℕ, a = m * b) :
  ∀ (x y : ℕ), 
    x > 0 →
    y > 0 →
    (Nat.gcd (x + y) 330 = 1) → 
    (∃ k : ℕ, x^x = k * y^y) → 
    (¬ ∃ m : ℕ, x = m * y) → 
    (a + b ≤ x + y) ∧ 
    (a + b = 520) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_with_conditions_l670_67015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_item_list_price_l670_67030

theorem item_list_price : ∃ (x : ℝ), 
  0.15 * (x - 15) = 0.25 * (x - 25) ∧ x = 40 := by
  use 40
  apply And.intro
  · norm_num
  · rfl

#check item_list_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_item_list_price_l670_67030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_meat_given_same_filling_l670_67023

/-- The total number of dumplings -/
def total_dumplings : ℕ := 5

/-- The number of meat-filled dumplings -/
def meat_dumplings : ℕ := 2

/-- The number of red bean paste-filled dumplings -/
def bean_dumplings : ℕ := 3

/-- The probability of selecting two dumplings with the same filling -/
noncomputable def prob_same_filling : ℚ :=
  (Nat.choose meat_dumplings 2 + Nat.choose bean_dumplings 2) / Nat.choose total_dumplings 2

/-- The probability of selecting two meat-filled dumplings -/
noncomputable def prob_both_meat : ℚ := 
  Nat.choose meat_dumplings 2 / Nat.choose total_dumplings 2

/-- Theorem: The probability of both dumplings being meat-filled, given that they have the same filling, is 1/4 -/
theorem prob_meat_given_same_filling :
  prob_both_meat / prob_same_filling = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_meat_given_same_filling_l670_67023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_periodicity_l670_67024

/-- A cubic polynomial with rational coefficients -/
def CubicPolynomial (a b c d : ℚ) : ℚ → ℚ := λ x ↦ a * x^3 + b * x^2 + c * x + d

/-- A sequence of rational numbers defined by a recurrence relation -/
def RationalSequence (P : ℚ → ℚ) (q₁ : ℚ) : ℕ → ℚ
  | 0 => q₁
  | n + 1 => P (RationalSequence P q₁ n)

/-- The main theorem: eventual periodicity of the sequence -/
theorem eventual_periodicity
  (a b c d : ℚ) (q₁ : ℚ) :
  ∃ (k : ℕ), k > 0 ∧ ∀ (n : ℕ), n ≥ 1 →
    RationalSequence (CubicPolynomial a b c d) q₁ (n + k) =
    RationalSequence (CubicPolynomial a b c d) q₁ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_periodicity_l670_67024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_4a_plus_c_l670_67082

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

-- Define the angle ABC to be 120°
noncomputable def angle_ABC_120 (t : Triangle) : Prop :=
  let angle_ABC := Real.arccos ((t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c))
  angle_ABC = 2 * Real.pi / 3

-- Define the angle bisector property
def angle_bisector_property (t : Triangle) (D : ℝ × ℝ) : Prop :=
  ∃ (lambda : ℝ), D = ((lambda * t.A.1 + t.C.1) / (lambda + 1), (lambda * t.A.2 + t.C.2) / (lambda + 1))

-- Define BD = 1
def BD_equals_one (t : Triangle) (D : ℝ × ℝ) : Prop :=
  Real.sqrt ((t.B.1 - D.1)^2 + (t.B.2 - D.2)^2) = 1

-- Theorem statement
theorem min_value_4a_plus_c (t : Triangle) (D : ℝ × ℝ) :
  is_valid_triangle t →
  angle_ABC_120 t →
  angle_bisector_property t D →
  BD_equals_one t D →
  ∀ a c, a > 0 → c > 0 → 4 * a + c ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_4a_plus_c_l670_67082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_locus_is_parabola_l670_67006

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in the 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The distance from a point to a line -/
noncomputable def distanceToLine (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- Checks if a circle passes through a point -/
def passesThrough (c : Circle) (p : Point) : Prop :=
  distance c.center p = c.radius

/-- Checks if a circle is tangent to a line -/
def isTangent (c : Circle) (l : Line) : Prop :=
  distanceToLine c.center l = c.radius

/-- The given point F -/
def F : Point := { x := 0, y := 3 }

/-- The given line y + 3 = 0 -/
def L : Line := { a := 0, b := 1, c := 3 }

/-- The theorem stating that x^2 = 12y is the locus of circle centers -/
theorem circle_locus_is_parabola :
  ∀ (p : Point),
    (∃ (c : Circle), passesThrough c F ∧ isTangent c L ∧ c.center = p) ↔
    p.x^2 = 12 * p.y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_locus_is_parabola_l670_67006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_zero_to_e_l670_67085

open Real

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (log x) / x

-- State the theorem
theorem f_increasing_on_zero_to_e :
  ∀ x : ℝ, 0 < x → x < Real.exp 1 → deriv f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_zero_to_e_l670_67085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_duration_is_48_minutes_l670_67072

/-- Represents the painting rate in terms of percentage of house painted per hour -/
structure PaintingRate where
  rate : ℝ
  nonneg : rate ≥ 0

/-- Represents a day's work -/
structure WorkDay where
  startTime : ℝ
  endTime : ℝ
  percentPainted : ℝ
  numPainters : ℕ
  validTime : startTime < endTime
  validPercent : percentPainted ≥ 0 ∧ percentPainted ≤ 1

def lunchBreakDuration : ℝ → Prop := sorry

/-- Paula's painting rate -/
noncomputable def paulaRate : PaintingRate := sorry

/-- Assistant's painting rate -/
noncomputable def assistantRate : PaintingRate := sorry

/-- Friday's work -/
def fridayWork : WorkDay :=
  { startTime := 8
    endTime := 16
    percentPainted := 0.6
    numPainters := 2
    validTime := by sorry
    validPercent := by sorry }

/-- Saturday's work -/
def saturdayWork : WorkDay :=
  { startTime := 8
    endTime := 17
    percentPainted := 0.35
    numPainters := 1
    validTime := by sorry
    validPercent := by sorry }

/-- Sunday's work -/
def sundayWork : WorkDay :=
  { startTime := 8
    endTime := 13
    percentPainted := 0.1
    numPainters := 1
    validTime := by sorry
    validPercent := by sorry }

theorem lunch_break_duration_is_48_minutes :
  lunchBreakDuration (48 / 60) ∧
  ∀ x, lunchBreakDuration x →
    fridayWork.percentPainted = (fridayWork.endTime - fridayWork.startTime - x) * (paulaRate.rate + assistantRate.rate) ∧
    saturdayWork.percentPainted = (saturdayWork.endTime - saturdayWork.startTime - x) * paulaRate.rate ∧
    sundayWork.percentPainted = (sundayWork.endTime - sundayWork.startTime - x) * assistantRate.rate :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_duration_is_48_minutes_l670_67072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l670_67074

-- Define the power function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^a

-- State the theorem
theorem power_function_theorem (a : ℝ) :
  f a 2 = Real.sqrt 2 →
  (a = 1/2) ∧ (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x < y → f a x < f a y) :=
by
  -- The proof is omitted and replaced with 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l670_67074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_height_l670_67066

/-- The height of a square pyramid given its base edge length and angle between adjacent faces -/
noncomputable def pyramid_height (a : ℝ) (α : ℝ) : ℝ :=
  a / Real.sqrt (2 * Real.tan (α / 2) ^ 2 - 2)

/-- Theorem: The height of a square pyramid with base edge length a and adjacent faces forming an angle α with each other is given by a / √(2 tan²(α/2) - 2) -/
theorem square_pyramid_height (a α : ℝ) (h_a : a > 0) (h_α : 0 < α ∧ α < π) :
  ∃ m : ℝ, m > 0 ∧ m = pyramid_height a α :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_height_l670_67066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l670_67096

noncomputable def f (x : ℝ) : ℝ := sorry

noncomputable def triangle_area (a : ℝ) : ℝ := 
  if a ≤ 3 then (a^2 - 2*a + 1) / 4 else a - 2

theorem max_triangle_area 
  (h_even : ∀ x, f x = f (-x))
  (h_periodic : ∀ x, f (x + 2) = f x)
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x - 1)
  (h_a : 2 < a) :
  ∃ A B : ℝ × ℝ, 
    A.1 < B.1 ∧ 
    A.1 ∈ Set.Icc 1 3 ∧ 
    B.1 ∈ Set.Icc 1 3 ∧
    f A.1 = f B.1 ∧
    triangle_area a = 
      (1/2) * abs (B.1 - A.1) * abs (a - f A.1) ∧
    ∀ A' B' : ℝ × ℝ, 
      A'.1 < B'.1 → 
      A'.1 ∈ Set.Icc 1 3 → 
      B'.1 ∈ Set.Icc 1 3 →
      f A'.1 = f B'.1 →
      (1/2) * abs (B'.1 - A'.1) * abs (a - f A'.1) ≤ triangle_area a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l670_67096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_five_coprime_two_digit_composites_l670_67099

theorem no_five_coprime_two_digit_composites :
  ¬ ∃ (a b c d e : ℕ),
    (10 ≤ a ∧ a ≤ 99) ∧
    (10 ≤ b ∧ b ≤ 99) ∧
    (10 ≤ c ∧ c ≤ 99) ∧
    (10 ≤ d ∧ d ≤ 99) ∧
    (10 ≤ e ∧ e ≤ 99) ∧
    (¬ Nat.Prime a) ∧
    (¬ Nat.Prime b) ∧
    (¬ Nat.Prime c) ∧
    (¬ Nat.Prime d) ∧
    (¬ Nat.Prime e) ∧
    (Nat.Coprime a b) ∧
    (Nat.Coprime a c) ∧
    (Nat.Coprime a d) ∧
    (Nat.Coprime a e) ∧
    (Nat.Coprime b c) ∧
    (Nat.Coprime b d) ∧
    (Nat.Coprime b e) ∧
    (Nat.Coprime c d) ∧
    (Nat.Coprime c e) ∧
    (Nat.Coprime d e) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_five_coprime_two_digit_composites_l670_67099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_from_home_growth_is_exponential_l670_67025

/-- Represents the percentage of working adults working at home in Metroville --/
def WorkFromHomePercentage : ℕ → ℝ
  | 2000 => 10
  | 2005 => 13
  | 2010 => 20
  | 2015 => 40
  | _ => 0  -- Default value for other years

/-- Represents an exponential growth model --/
def ExponentialGrowthModel (a b : ℝ) (t : ℕ) : ℝ :=
  a * (1 + b) ^ (t - 2000)

/-- Theorem stating that the WorkFromHomePercentage is best represented by an exponential growth model --/
theorem work_from_home_growth_is_exponential :
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧
  (∀ t : ℕ, t ∈ ({2000, 2005, 2010, 2015} : Set ℕ) →
    |WorkFromHomePercentage t - ExponentialGrowthModel a b t| < 1) :=
sorry

#check work_from_home_growth_is_exponential

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_from_home_growth_is_exponential_l670_67025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l670_67060

theorem triangle_property (A B C : Real) (a b c : Real) :
  (0 < A) ∧ (A < π) ∧
  (0 < B) ∧ (B < π) ∧
  (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  (Real.sin A ^ 2 - Real.sin B ^ 2 - Real.sin C ^ 2 = Real.sin B * Real.sin C) ∧
  (a = 3) ∧
  (a = b * Real.sin C / Real.sin B) ∧
  (a = c * Real.sin B / Real.sin C) ∧
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →
  (A = 2 * π / 3) ∧
  (∃ (P : Real), P = a + b + c ∧ P ≤ 3 + 2 * Real.sqrt 3) ∧
  (∃ (P_max : Real), P_max = 3 + 2 * Real.sqrt 3 ∧
    ∀ (b' c' : Real),
      (3 ^ 2 = b' ^ 2 + c' ^ 2 - 2 * b' * c' * Real.cos (2 * π / 3)) →
      (3 + b' + c' ≤ P_max)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l670_67060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l670_67063

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3)

theorem f_properties :
  (¬ ∀ x : ℝ, f x = f (-Real.pi/6 - x)) ∧ 
  (∀ x y : ℝ, -Real.pi/6 ≤ x ∧ x < y ∧ y ≤ 0 → f x > f y) ∧
  ((¬ ∀ x : ℝ, f x = f (-Real.pi/6 - x)) ∨ 
   (∀ x y : ℝ, -Real.pi/6 ≤ x ∧ x < y ∧ y ≤ 0 → f x > f y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l670_67063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_lucky_multiple_of_8_l670_67098

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_lucky (n : ℕ) : Prop :=
  n % (sum_of_digits n) = 0

theorem smallest_non_lucky_multiple_of_8 :
  ∀ k : ℕ, k > 0 ∧ k < 11 → is_lucky (8 * k) ∧
  ¬ is_lucky 88 ∧
  (∀ m : ℕ, m > 0 ∧ m < 11 → 8 * m ≠ 88) :=
by
  sorry

#eval sum_of_digits 88  -- This will help verify that sum_of_digits is working
#eval 88 % (sum_of_digits 88)  -- This will help verify that is_lucky is working correctly for 88

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_lucky_multiple_of_8_l670_67098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_path_difference_l670_67087

theorem garden_path_difference : 
  let east_side : ℝ := 3
  let north_side : ℝ := 4
  let jerry_path : ℝ := east_side + north_side
  let silvia_path : ℝ := Real.sqrt (east_side^2 + north_side^2)
  let difference_percentage : ℝ := (jerry_path - silvia_path) / jerry_path * 100
  ∃ (closest_percentage : ℝ), closest_percentage = 30 ∧ 
    ∀ (other_percentage : ℝ), other_percentage ∈ ({20, 25, 35, 40} : Set ℝ) → 
      |difference_percentage - closest_percentage| < |difference_percentage - other_percentage| :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_path_difference_l670_67087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_equilibrium_l670_67079

/-- A curved, thin homogeneous rod ABC with weights at its ends is in equilibrium. -/
theorem rod_equilibrium (m₁ m₂ lambda AB BC BO OC : ℝ) : 
  m₁ = 2 →
  lambda = 2 →
  AB = 7 →
  BC = 5 →
  BO = 4 →
  OC = 3 →
  m₁ * AB + lambda * AB * (AB / 2) = m₂ * BO + lambda * BC * (BO / 2) →
  m₂ = 10.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_equilibrium_l670_67079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l670_67073

theorem cos_alpha_value (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos (α + β) = 12/13) (h4 : Real.cos (2*α + β) = 3/5) : Real.cos α = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l670_67073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sharp_64_nested_l670_67093

-- Define the # operation
noncomputable def sharp (N : ℝ) : ℝ := (1/2) * (N - 2) + 2

-- Theorem statement
theorem sharp_64_nested : sharp (sharp (sharp 64)) = 9.75 := by
  -- Evaluate sharp 64
  have h1 : sharp 64 = 33 := by
    unfold sharp
    simp
    norm_num
  
  -- Evaluate sharp 33
  have h2 : sharp 33 = 17.5 := by
    unfold sharp
    simp
    norm_num

  -- Evaluate sharp 17.5
  have h3 : sharp 17.5 = 9.75 := by
    unfold sharp
    simp
    norm_num

  -- Combine the steps
  rw [h1, h2, h3]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sharp_64_nested_l670_67093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l670_67097

-- Define the constants
noncomputable def a : ℝ := Real.pi ^ (1/3)
noncomputable def b : ℝ := Real.log 3 / Real.log Real.pi
noncomputable def c : ℝ := Real.log (Real.sqrt 3 - 1)

-- State the theorem
theorem relationship_abc : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l670_67097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l670_67059

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  a = 3 →
  Real.cos A = Real.sqrt 6 / 3 →
  B = A + Real.pi / 2 →
  b = 3 * Real.sqrt 2 ∧
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l670_67059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_edge_sum_l670_67069

/-- Represents a rectangular solid with dimensions in geometric progression -/
structure GeometricRectangularSolid where
  a : ℝ
  r : ℝ

/-- The volume of a GeometricRectangularSolid -/
noncomputable def volume (solid : GeometricRectangularSolid) : ℝ :=
  solid.a^3

/-- The surface area of a GeometricRectangularSolid -/
noncomputable def surfaceArea (solid : GeometricRectangularSolid) : ℝ :=
  2 * (solid.a^2 / solid.r + solid.a^2 * solid.r + solid.a^2)

/-- The sum of all edge lengths of a GeometricRectangularSolid -/
noncomputable def sumOfEdges (solid : GeometricRectangularSolid) : ℝ :=
  4 * (solid.a / solid.r + solid.a + solid.a * solid.r)

theorem rectangular_solid_edge_sum :
  ∃ (solid : GeometricRectangularSolid),
    volume solid = 343 ∧
    surfaceArea solid = 294 ∧
    sumOfEdges solid = 84 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_edge_sum_l670_67069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_position_l670_67077

/-- The distance from the center of a circle to the centroid of an arc -/
noncomputable def centroid_distance (R : ℝ) (α : ℝ) : ℝ :=
  (2 * R / α) * Real.sin (α / 2)

/-- The theorem stating the position of the centroid for an arc and a semicircle -/
theorem centroid_position (R : ℝ) (α : ℝ) (h_R : R > 0) (h_α : α > 0) :
  centroid_distance R α = (2 * R / α) * Real.sin (α / 2) ∧
  centroid_distance R Real.pi = 2 * R / Real.pi := by
  sorry

#check centroid_position

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_position_l670_67077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l670_67035

noncomputable section

/-- Helper function to calculate the area of a triangle given its vertices -/
def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  let a := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let b := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let c := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Given a triangle PQR with PQ = 12 and PR : QR = 25 : 24, 
    the maximum possible area is 2601 -/
theorem triangle_max_area (P Q R : ℝ × ℝ) : 
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  pq = 12 ∧ pr / qr = 25 / 24 →
  area_triangle P Q R ≤ 2601 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l670_67035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_plus_minus_one_l670_67084

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := 
  (2 * Real.cos θ - 4 * Real.sin θ * Real.cos θ, 
   2 * Real.cos θ * Real.sin θ - 4 * Real.sin θ * Real.sin θ)

noncomputable def line_l (α t : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, -1 + t * Real.sin α)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_slope_is_plus_minus_one 
  (α : ℝ) 
  (t1 t2 : ℝ) 
  (θ1 θ2 : ℝ) 
  (h1 : curve_C θ1 = line_l α t1) 
  (h2 : curve_C θ2 = line_l α t2) 
  (h3 : distance (curve_C θ1) (curve_C θ2) = 3 * Real.sqrt 2) :
  Real.tan α = 1 ∨ Real.tan α = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_plus_minus_one_l670_67084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l670_67018

-- Define the interval [1,2]
def I : Set ℝ := Set.Icc 1 2

-- Define the functions f and g
noncomputable def f (p q : ℝ) (x : ℝ) := x^2 + p*x + q
noncomputable def g (x : ℝ) := x + 1/x^2

-- State the theorem
theorem max_value_f (p q : ℝ) :
  (∃ x ∈ I, (∀ y ∈ I, f p q y ≥ f p q x) ∧
                     (∀ y ∈ I, g y ≥ g x) ∧
                     (∀ y ∈ I, f p q y = g y → y = x)) →
  (∃ x ∈ I, ∀ y ∈ I, f p q y ≤ f p q x) ∧
  (∃ x ∈ I, f p q x = 4 - (5/2) * Real.rpow 2 (1/3) + Real.rpow 4 (1/3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l670_67018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_race_path_l670_67090

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- The race configuration -/
structure RaceConfig where
  wall_length : ℝ
  start : Point
  finish : Point
  wall_start : Point
  wall_end : Point

/-- Reflects a point across a vertical line -/
def reflect_point (p : Point) (line_x : ℝ) : Point :=
  { x := 2 * line_x - p.x, y := p.y }

theorem shortest_race_path (config : RaceConfig)
  (h1 : config.wall_length = 1500)
  (h2 : config.start = { x := 0, y := 250 })
  (h3 : config.finish = { x := 800, y := 250 })
  (h4 : config.wall_start = { x := 0, y := 0 })
  (h5 : config.wall_end = { x := 0, y := 1500 }) :
  let reflected_end := reflect_point config.finish config.wall_start.x
  ∃ (touch_point : Point),
    touch_point.x = config.wall_start.x ∧
    0 ≤ touch_point.y ∧ touch_point.y ≤ config.wall_length ∧
    distance config.start reflected_end =
      distance config.start touch_point + distance touch_point config.finish :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_race_path_l670_67090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_is_three_l670_67003

def mySequence (n : ℕ) : ℝ :=
  if n % 2 = 1 then 3 else 6

theorem eleventh_term_is_three :
  let a := mySequence
  (∀ n : ℕ, n ≥ 2 → a n * a (n-1) = 18) →
  a 1 = 3 →
  a 2 = 6 →
  a 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_is_three_l670_67003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equality_l670_67031

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := Real.sqrt ((x + 5) / 5)

-- State the theorem
theorem h_equality (x : ℝ) : h (3 * x) = 3 * h x ↔ x = -20/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equality_l670_67031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_correct_l670_67056

/-- The area of the overlap region formed by two strips of width 2 intersecting at an angle β -/
noncomputable def overlap_area (β : ℝ) : ℝ :=
  2 / Real.sin β

/-- Theorem stating that the area of the overlap region is 2 / sin(β) -/
theorem overlap_area_correct (β : ℝ) (h : 0 < β ∧ β < π) :
  overlap_area β = 2 / Real.sin β :=
by
  -- Unfold the definition of overlap_area
  unfold overlap_area
  -- The equality follows directly from the definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_correct_l670_67056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surface_area_difference_angle_theorem_l670_67071

/-- The angle that maximizes the difference in surface areas of truncated conical surfaces -/
noncomputable def max_surface_area_difference_angle (R r d : ℝ) : ℝ :=
  Real.arcsin (Real.sqrt (((R - r)^2 + d^2 - Real.sqrt ((R - r)^4 - (R - r)^2 * d^2 + d^4)) / (3 * d^2)))

/-- Theorem stating the angle that maximizes the difference in surface areas -/
theorem max_surface_area_difference_angle_theorem (R r d : ℝ) 
  (h1 : R > r) 
  (h2 : R > 0) 
  (h3 : r > 0) 
  (h4 : d > R + r) : 
  let α := max_surface_area_difference_angle R r d
  ∃ (P : ℝ → ℝ), 
    (∀ θ, P θ ≥ 0) ∧ 
    (∀ θ, θ ≠ α → P α > P θ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surface_area_difference_angle_theorem_l670_67071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andrews_age_l670_67026

def Ages : Finset ℕ := {4, 6, 8, 10, 12, 14}

structure Person where
  age : ℕ
  activity : String

def andrew : Person := ⟨0, "home"⟩  -- Andrew's age is initially unknown

theorem andrews_age : 
  ∀ (people : Finset Person),
    (∀ p, p ∈ people → p.age ∈ Ages) →
    (Finset.card people = 6) →
    (∃ p1 p2, p1 ∈ people ∧ p2 ∈ people ∧ p1 ≠ p2 ∧ p1.age + p2.age = 18 ∧ p1.activity = "movies" ∧ p2.activity = "movies") →
    (∃ p1 p2, p1 ∈ people ∧ p2 ∈ people ∧ p1 ≠ p2 ∧ p1.age < 12 ∧ p2.age < 12 ∧ p1.age ≠ 8 ∧ p2.age ≠ 8 ∧ p1.activity = "baseball" ∧ p2.activity = "baseball") →
    (∃ p, p ∈ people ∧ p.age = 6 ∧ p.activity = "home") →
    (andrew ∈ people ∧ andrew.activity = "home") →
    andrew.age = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andrews_age_l670_67026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_unique_plants_l670_67055

-- Define the sets of plants in each bed
variable (A B C D : Finset ℕ)

-- Define the cardinalities of the sets
axiom card_A : Finset.card A = 600
axiom card_B : Finset.card B = 500
axiom card_C : Finset.card C = 400
axiom card_D : Finset.card D = 300

-- Define the intersections
axiom card_A_inter_B : Finset.card (A ∩ B) = 80
axiom card_A_inter_C : Finset.card (A ∩ C) = 70
axiom card_A_inter_B_inter_D : Finset.card (A ∩ B ∩ D) = 40

-- No other overlaps
axiom no_other_overlaps : 
  (A ∩ D) \ (A ∩ B ∩ D) = ∅ ∧ 
  (B ∩ C) = ∅ ∧ 
  (B ∩ D) \ (A ∩ B ∩ D) = ∅ ∧ 
  (C ∩ D) = ∅

-- Theorem to prove
theorem total_unique_plants : Finset.card (A ∪ B ∪ C ∪ D) = 1690 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_unique_plants_l670_67055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_of_extreme_values_l670_67017

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1/x + 2*a*Real.log x

theorem min_difference_of_extreme_values (a : ℝ) :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ ≥ Real.exp 1 →
  (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 → x = x₁ ∨ x = x₂) →
  f a x₁ - f a x₂ ≥ 4 / Real.exp 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_of_extreme_values_l670_67017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_is_minus_six_l670_67020

/-- The x-coordinate of a point on a line given two other points on the line and its y-coordinate -/
noncomputable def find_x_coordinate (x1 y1 x2 y2 y : ℝ) : ℝ :=
  let m := (y2 - y1) / (x2 - x1)  -- slope of the line
  x1 + (y - y1) / m               -- x-coordinate of the point

/-- Theorem stating that the x-coordinate of (x, -6) on the line through (2,10) and (-2,2) is -6 -/
theorem x_coordinate_is_minus_six :
  find_x_coordinate 2 10 (-2) 2 (-6) = -6 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval find_x_coordinate 2 10 (-2) 2 (-6)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_is_minus_six_l670_67020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_plums_correct_l670_67043

/-- The ratio of pears to plums by weight -/
def pear_plum_ratio : ℚ := 4 / 3

/-- The number of pears Sally has -/
def sally_pears : ℕ := 20

/-- The equivalent number of plums to Sally's pears -/
def equivalent_plums : ℕ := 15

/-- Theorem stating that the equivalent number of plums is correct -/
theorem equivalent_plums_correct : 
  (sally_pears : ℚ) * pear_plum_ratio⁻¹ = equivalent_plums := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_plums_correct_l670_67043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_running_speed_calculation_l670_67092

noncomputable def walking_speed : ℝ := 4
noncomputable def total_distance : ℝ := 12
noncomputable def total_time : ℝ := 2.25
noncomputable def half_distance : ℝ := total_distance / 2

theorem running_speed_calculation (R : ℝ) : 
  (half_distance / walking_speed) + (half_distance / R) = total_time →
  R = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_running_speed_calculation_l670_67092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l670_67044

noncomputable def projection (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (v.1 * u.1 + v.2 * u.2) / (u.1 * u.1 + u.2 * u.2)
  (scalar * u.1, scalar * u.2)

theorem projection_property :
  let v₁ : ℝ × ℝ := (2, -3)
  let p₁ : ℝ × ℝ := (1, -3/2)
  let v₂ : ℝ × ℝ := (3, -1)
  let p₂ : ℝ × ℝ := (18/13, -27/13)
  projection v₁ p₁ = p₁ → projection v₂ p₁ = p₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l670_67044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weather_forecast_probability_l670_67028

def accuracy_rate : ℝ := 0.8
def days : ℕ := 3
def min_accurate_days : ℕ := 2

theorem weather_forecast_probability :
  let p := (Finset.range (days - min_accurate_days + 1)).sum (λ k => 
    (Nat.choose days (days - k) : ℝ) * accuracy_rate^(days - k) * (1 - accuracy_rate)^k)
  p = 0.896 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weather_forecast_probability_l670_67028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l670_67083

/-- A pyramid with a square base -/
structure SquarePyramid where
  /-- The length of each side of the square base in cm -/
  baseSide : ℝ
  /-- The height from the vertex to the center of the base in cm -/
  height : ℝ

/-- Calculate the volume of a square pyramid -/
noncomputable def volumeSquarePyramid (p : SquarePyramid) : ℝ :=
  (1 / 3) * p.baseSide^2 * p.height

/-- The specific pyramid described in the problem -/
def specificPyramid : SquarePyramid where
  baseSide := 10
  height := 12

theorem volume_of_specific_pyramid :
  volumeSquarePyramid specificPyramid = 400 := by
  sorry

#eval specificPyramid.baseSide -- This will output 10
#eval specificPyramid.height   -- This will output 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l670_67083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_pattern_area_l670_67078

theorem semicircle_pattern_area (diameter : ℝ) (pattern_length : ℝ) : 
  diameter = 3 →
  pattern_length = 36 →
  let num_semicircles := Int.floor (pattern_length / diameter * 2)
  let semicircle_area := π * (diameter / 2)^2 / 2
  (↑num_semicircles : ℝ) * semicircle_area = 27 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_pattern_area_l670_67078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_R_l670_67065

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of an equilateral triangle ABC -/
def isEquilateral (A B C : Point) : Prop := sorry

/-- Definition of the incircle of a triangle -/
def isIncircle (ω : Circle) (A B C : Point) : Prop := sorry

/-- Definition of a line being tangent to a circle -/
def isTangent (ℓ : Line) (ω : Circle) : Prop := sorry

/-- Definition of a point lying on a line segment -/
def onSegment (P : Point) (A B : Point) : Prop := sorry

/-- Definition of the distance between two points -/
noncomputable def distance (P Q : Point) : ℝ := sorry

/-- Helper definition for circular arcs -/
def isCircularArc (arc : Set Point) : Prop := sorry

/-- Helper definition for arc on circumcircle -/
def isOnCircumcircle (arc : Set Point) (A B C : Point) : Prop := sorry

/-- Helper definition for arc centered at incenter -/
def isCenteredAtIncenter (arc : Set Point) (A B C : Point) : Prop := sorry

/-- Main theorem about the locus of point R -/
theorem locus_of_R (A B C : Point) (ω : Circle) :
  isEquilateral A B C →
  isIncircle ω A B C →
  ∀ (ℓ : Line) (P Q R : Point),
    isTangent ℓ ω →
    onSegment P B C →
    onSegment Q C A →
    distance P R = distance P A →
    distance Q R = distance Q B →
    (∃ (arc1 arc2 : Set Point),
      R ∈ arc1 ∪ arc2 ∧
      isCircularArc arc1 ∧
      isCircularArc arc2 ∧
      isOnCircumcircle arc1 A B C ∧
      isCenteredAtIncenter arc2 A B C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_R_l670_67065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l670_67041

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 + 4*x + 5) / (2*x + 6)

-- State the theorem
theorem f_minimum_value :
  ∀ x : ℝ, -3 < x ∧ x < 2 → f x ≥ 3/4 ∧ (f x = 3/4 ↔ x = -1) := by
  sorry

#check f_minimum_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l670_67041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_distance_l670_67086

/-- Represents a car with its starting time and speed -/
structure Car where
  startTime : ℝ
  speed : ℝ

/-- Proves that given the conditions of the problem, each car travels 273 miles from when Car Z starts until they all stop -/
theorem car_travel_distance (carX carY carZ : Car)
  (hX : carX.startTime = 0 ∧ carX.speed = 35)
  (hY : carY.startTime = 1.2 ∧ carY.speed = 46)
  (hZ : carZ.startTime = 3.6 ∧ carZ.speed = 65)
  : ∃ t : ℝ,
    carX.speed * (t + carZ.startTime - carX.startTime) =
    carY.speed * (t + carZ.startTime - carY.startTime) ∧
    carX.speed * (t + carZ.startTime - carX.startTime) = carZ.speed * t ∧
    carZ.speed * t = 273 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_distance_l670_67086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_and_x_axis_l670_67052

-- Define the centers of the circles
def center_C : ℝ × ℝ := (4, 6)
def center_D : ℝ × ℝ := (14, 6)

-- Define the radius of both circles
def radius : ℝ := 6

-- Define the area of the region
noncomputable def region_area : ℝ := 60 - 18 * Real.pi

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Define a placeholder for the area function
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Define a placeholder for the region bound by circles and x-axis
def region_bound_by (C D : Circle) (axis : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem area_between_circles_and_x_axis :
  let C := Circle.mk center_C radius
  let D := Circle.mk center_D radius
  area (region_bound_by C D x_axis) = region_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_and_x_axis_l670_67052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_volume_l670_67010

/-- Regular quadrilateral pyramid with inscribed sphere -/
structure RegularPyramid where
  /-- Slant height (apothema) of the pyramid -/
  a : ℝ
  /-- Ratio of height division by inscribed sphere (1:8) -/
  height_ratio : ℝ
  /-- The inscribed sphere touches the base and all lateral faces -/
  sphere_touches_all : Prop
  /-- The pyramid is regular (quadrilateral base) -/
  is_regular : Prop

/-- Volume of the regular quadrilateral pyramid -/
noncomputable def pyramid_volume (p : RegularPyramid) : ℝ :=
  64 * p.a^3 / 125

/-- Theorem: The volume of the regular quadrilateral pyramid with given properties -/
theorem regular_pyramid_volume (p : RegularPyramid) 
    (h_positive : p.a > 0)
    (h_ratio : p.height_ratio = 1/8) :
    pyramid_volume p = 64 * p.a^3 / 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_volume_l670_67010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locker_digits_and_sum_properties_l670_67039

def locker_number (n : ℕ) : ℕ := n^2

def digits_count (n : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log n 10).succ

def sum_digits_first_hundred : ℕ :=
  (Finset.range 100).sum (λ i => digits_count (locker_number (i + 1)))

def sum_squares (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => locker_number (i + 1))

theorem locker_digits_and_sum_properties :
  (sum_digits_first_hundred = 358) ∧
  (sum_squares 2019 % 10 = 5) := by
  sorry

#eval sum_digits_first_hundred
#eval sum_squares 2019 % 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locker_digits_and_sum_properties_l670_67039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l670_67094

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else -a * x

theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/8 : ℝ) (1/3 : ℝ) ∧ a ≠ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l670_67094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l670_67080

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | (2 : ℝ)^(2*x - 1) ≥ 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 5*x < 0}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 5}) ∧
  (A ∪ B = {x : ℝ | x > 0}) ∧
  (Set.compl (A ∪ B) = {x : ℝ | x ≤ 0}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l670_67080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_global_minimum_global_maximum_l670_67064

open Set
open Function

-- Define the function f on the set of real numbers
variable (f : ℝ → ℝ)

-- Statement ④
theorem global_minimum (x₀ : ℝ) (h : ∀ x, x ≠ x₀ → f x > f x₀) :
  IsMinOn f univ x₀ := by
  sorry

-- Statement ⑤
theorem global_maximum (x₀ : ℝ) (h₁ : ∀ x, x < x₀ → (deriv f x) > 0)
  (h₂ : ∀ x, x > x₀ → (deriv f x) < 0) :
  IsMaxOn f univ x₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_global_minimum_global_maximum_l670_67064
