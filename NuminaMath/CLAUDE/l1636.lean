import Mathlib

namespace NUMINAMATH_CALUDE_product_units_digit_of_first_five_composite_l1636_163625

def first_five_composite_numbers : List Nat := [4, 6, 8, 9, 10]

def units_digit (n : Nat) : Nat := n % 10

def product_units_digit (numbers : List Nat) : Nat :=
  units_digit (numbers.foldl (·*·) 1)

theorem product_units_digit_of_first_five_composite : 
  product_units_digit first_five_composite_numbers = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_units_digit_of_first_five_composite_l1636_163625


namespace NUMINAMATH_CALUDE_inscribe_square_in_circle_l1636_163656

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in the plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if four points form a square -/
def is_square (p1 p2 p3 p4 : Point) : Prop :=
  let d12 := (p1.x - p2.x)^2 + (p1.y - p2.y)^2
  let d23 := (p2.x - p3.x)^2 + (p2.y - p3.y)^2
  let d34 := (p3.x - p4.x)^2 + (p3.y - p4.y)^2
  let d41 := (p4.x - p1.x)^2 + (p4.y - p1.y)^2
  let d13 := (p1.x - p3.x)^2 + (p1.y - p3.y)^2
  let d24 := (p2.x - p4.x)^2 + (p2.y - p4.y)^2
  (d12 = d23) ∧ (d23 = d34) ∧ (d34 = d41) ∧ (d13 = d24)

/-- Check if a point lies on a circle -/
def on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Construct a line through two points -/
def line_through_points (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y,
    b := p1.x - p2.x,
    c := p2.x * p1.y - p1.x * p2.y }

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  let l := line_through_points p1 p2
  l.a * p3.x + l.b * p3.y + l.c = 0

/-- Theorem: Given a circle with a marked center, it is possible to construct
    four points on the circle that form the vertices of a square using only
    straightedge constructions -/
theorem inscribe_square_in_circle (c : Circle) :
  ∃ (p1 p2 p3 p4 : Point),
    on_circle p1 c ∧ on_circle p2 c ∧ on_circle p3 c ∧ on_circle p4 c ∧
    is_square p1 p2 p3 p4 :=
sorry

end NUMINAMATH_CALUDE_inscribe_square_in_circle_l1636_163656


namespace NUMINAMATH_CALUDE_m_intersect_n_equals_n_l1636_163693

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M as the domain of ln(1-x)
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -6 < x ∧ x < 1}

-- Theorem statement
theorem m_intersect_n_equals_n : M ∩ N = N := by
  sorry

end NUMINAMATH_CALUDE_m_intersect_n_equals_n_l1636_163693


namespace NUMINAMATH_CALUDE_student_number_problem_l1636_163684

theorem student_number_problem (x : ℝ) : 2 * x - 140 = 102 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1636_163684


namespace NUMINAMATH_CALUDE_binomial_coefficient_15_l1636_163660

theorem binomial_coefficient_15 (n : ℕ) (h1 : n > 0) 
  (h2 : Nat.choose n 2 = 15) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_15_l1636_163660


namespace NUMINAMATH_CALUDE_base5_division_l1636_163645

/-- Converts a base 5 number to base 10 -/
def toBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base 10 number to base 5 -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem base5_division (dividend : List Nat) (divisor : List Nat) :
  dividend = [2, 0, 1, 3] ∧ divisor = [3, 2] →
  toBase5 (toBase10 dividend / toBase10 divisor) = [0, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base5_division_l1636_163645


namespace NUMINAMATH_CALUDE_wire_length_around_square_field_l1636_163685

/-- The length of wire required to go 15 times round a square field with area 69696 m^2 is 15840 meters. -/
theorem wire_length_around_square_field : 
  let field_area : ℝ := 69696
  let side_length : ℝ := (field_area) ^ (1/2 : ℝ)
  let perimeter : ℝ := 4 * side_length
  let wire_length : ℝ := 15 * perimeter
  wire_length = 15840 := by sorry

end NUMINAMATH_CALUDE_wire_length_around_square_field_l1636_163685


namespace NUMINAMATH_CALUDE_parabola_point_distance_l1636_163631

theorem parabola_point_distance (x₀ y₀ : ℝ) : 
  y₀^2 = 8 * x₀ →  -- Point (x₀, y₀) is on the parabola y² = 8x
  (x₀ - 2)^2 + y₀^2 = 3^2 →  -- Distance from (x₀, y₀) to focus (2, 0) is 3
  |y₀| = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l1636_163631


namespace NUMINAMATH_CALUDE_log_inequality_solution_l1636_163654

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the solution set
def solution_set : Set ℝ := {x | log_half (2*x - 1) < log_half (-x + 5)}

-- Theorem statement
theorem log_inequality_solution :
  solution_set = Set.Ioo 2 5 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_solution_l1636_163654


namespace NUMINAMATH_CALUDE_tim_has_five_marbles_l1636_163617

/-- The number of blue marbles Fred has -/
def fred_marbles : ℕ := 110

/-- The ratio of Fred's marbles to Tim's marbles -/
def ratio : ℕ := 22

/-- The number of blue marbles Tim has -/
def tim_marbles : ℕ := fred_marbles / ratio

theorem tim_has_five_marbles : tim_marbles = 5 := by
  sorry

end NUMINAMATH_CALUDE_tim_has_five_marbles_l1636_163617


namespace NUMINAMATH_CALUDE_sequence_formula_l1636_163677

theorem sequence_formula (a : ℕ+ → ℚ) :
  (∀ n : ℕ+, a (n + 1) / a n = (n + 2 : ℚ) / n) →
  a 1 = 1 →
  ∀ n : ℕ+, a n = n * (n + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l1636_163677


namespace NUMINAMATH_CALUDE_negative_a_fifth_times_a_l1636_163665

theorem negative_a_fifth_times_a (a : ℝ) : (-a)^5 * a = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_fifth_times_a_l1636_163665


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1636_163632

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (2 - a * I) / (1 + I) = b * I) ↔ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1636_163632


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_600_degrees_l1636_163659

theorem angle_with_same_terminal_side_as_600_degrees :
  ∀ α : ℝ, (∃ k : ℤ, α = 600 + k * 360) → (∃ k : ℤ, α = k * 360 + 240) :=
by sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_600_degrees_l1636_163659


namespace NUMINAMATH_CALUDE_range_of_m_l1636_163638

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0) → 2*x^2 - 9*x + m < 0) → 
  m < 9 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1636_163638


namespace NUMINAMATH_CALUDE_properties_dependency_l1636_163661

-- Define a type for geometric figures
inductive GeometricFigure
| Square
| Rectangle

-- Define properties for geometric figures
def hasEqualSides (f : GeometricFigure) : Prop :=
  match f with
  | GeometricFigure.Square => true
  | GeometricFigure.Rectangle => true

def hasRightAngles (f : GeometricFigure) : Prop :=
  match f with
  | GeometricFigure.Square => true
  | GeometricFigure.Rectangle => true

-- Define dependency of properties
def arePropertiesDependent (f : GeometricFigure) : Prop :=
  match f with
  | GeometricFigure.Square => hasEqualSides f ↔ hasRightAngles f
  | GeometricFigure.Rectangle => ¬(hasEqualSides f ↔ hasRightAngles f)

-- Theorem statement
theorem properties_dependency :
  arePropertiesDependent GeometricFigure.Square ∧
  ¬(arePropertiesDependent GeometricFigure.Rectangle) :=
sorry

end NUMINAMATH_CALUDE_properties_dependency_l1636_163661


namespace NUMINAMATH_CALUDE_fraction_reduction_l1636_163650

theorem fraction_reduction (b y : ℝ) : 
  (Real.sqrt (b^2 + y^2) - (y^2 - b^2) / Real.sqrt (b^2 + y^2)) / (b^2 + y^2) = 
  2 * b^2 / (b^2 + y^2)^(3/2) := by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l1636_163650


namespace NUMINAMATH_CALUDE_first_field_rows_l1636_163678

/-- Represents a corn field with a certain number of rows -/
structure CornField where
  rows : ℕ

/-- Represents a farm with two corn fields -/
structure Farm where
  field1 : CornField
  field2 : CornField

def corn_cobs_per_row : ℕ := 4

def total_corn_cobs (f : Farm) : ℕ :=
  (f.field1.rows + f.field2.rows) * corn_cobs_per_row

theorem first_field_rows (f : Farm) :
  f.field2.rows = 16 → total_corn_cobs f = 116 → f.field1.rows = 13 := by
  sorry

end NUMINAMATH_CALUDE_first_field_rows_l1636_163678


namespace NUMINAMATH_CALUDE_tan_value_from_trig_ratio_l1636_163610

theorem tan_value_from_trig_ratio (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 2) :
  Real.tan α = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_trig_ratio_l1636_163610


namespace NUMINAMATH_CALUDE_initial_children_on_bus_l1636_163666

/-- Given that 14 more children got on a bus at a bus stop, 
    resulting in a total of 78 children, prove that there were 
    initially 64 children on the bus. -/
theorem initial_children_on_bus : 
  ∀ (initial : ℕ), initial + 14 = 78 → initial = 64 := by
  sorry

end NUMINAMATH_CALUDE_initial_children_on_bus_l1636_163666


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1636_163680

theorem necessary_not_sufficient_condition :
  (∃ a : ℝ, a > 0 ∧ a^2 - 2*a ≥ 0) ∧
  (∀ a : ℝ, a^2 - 2*a < 0 → a > 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1636_163680


namespace NUMINAMATH_CALUDE_triangle_side_length_l1636_163615

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : Real)
  (x y z : Real)

-- State the theorem
theorem triangle_side_length 
  (t : Triangle) 
  (h1 : t.y = 7) 
  (h2 : t.z = 6) 
  (h3 : Real.cos (t.Y - t.Z) = 15/16) : 
  t.x = Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1636_163615


namespace NUMINAMATH_CALUDE_kayak_production_sum_l1636_163605

/-- Calculates the sum of a geometric sequence -/
def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

/-- The number of kayaks built in the first month -/
def initial_kayaks : ℕ := 8

/-- The ratio of kayaks built between consecutive months -/
def kayak_ratio : ℕ := 3

/-- The number of months of kayak production -/
def num_months : ℕ := 6

theorem kayak_production_sum :
  geometric_sum initial_kayaks kayak_ratio num_months = 2912 := by
  sorry

end NUMINAMATH_CALUDE_kayak_production_sum_l1636_163605


namespace NUMINAMATH_CALUDE_find_d_l1636_163606

theorem find_d : ∃ d : ℝ, 
  (∃ n : ℤ, n = ⌊d⌋ ∧ 3 * n^2 + 19 * n - 84 = 0) ∧ 
  (5 * (d - ⌊d⌋)^2 - 26 * (d - ⌊d⌋) + 12 = 0) ∧ 
  (0 ≤ d - ⌊d⌋ ∧ d - ⌊d⌋ < 1) ∧
  d = 3.44 := by
  sorry

end NUMINAMATH_CALUDE_find_d_l1636_163606


namespace NUMINAMATH_CALUDE_amusement_park_distance_l1636_163689

-- Define the speeds and time
def speed_A : ℝ := 3
def speed_B : ℝ := 4
def total_time : ℝ := 4

-- Define the distance functions
def distance_A (t : ℝ) : ℝ := speed_A * t
def distance_B (t : ℝ) : ℝ := speed_B * t

-- Theorem statement
theorem amusement_park_distance :
  ∃ (t_A t_B : ℝ),
    t_A + t_B = total_time ∧
    distance_B t_B = distance_A t_A + 2 ∧
    distance_B t_B = 8 :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_distance_l1636_163689


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l1636_163690

def is_in_third_quadrant (α : Real) : Prop :=
  180 < α % 360 ∧ α % 360 ≤ 270

theorem angle_in_third_quadrant (k : Int) (α : Real) 
  (h1 : (4*k + 1 : Real) * 180 < α) 
  (h2 : α < (4*k + 1 : Real) * 180 + 60) :
  is_in_third_quadrant α :=
sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l1636_163690


namespace NUMINAMATH_CALUDE_red_chips_probability_l1636_163698

theorem red_chips_probability (total_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ) 
  (h1 : total_chips = red_chips + green_chips)
  (h2 : red_chips = 5)
  (h3 : green_chips = 3) :
  (Nat.choose (total_chips - 1) (green_chips - 1) : ℚ) / (Nat.choose total_chips green_chips) = 3/8 :=
sorry

end NUMINAMATH_CALUDE_red_chips_probability_l1636_163698


namespace NUMINAMATH_CALUDE_point_C_coordinates_l1636_163657

/-- Given points O, A, B, and C in a 2D plane, prove that C has specific coordinates. -/
theorem point_C_coordinates (O A B C : ℝ × ℝ) : 
  (A.1 - O.1 = -3 ∧ A.2 - O.2 = 1) →  -- OA = (-3, 1)
  (B.1 - O.1 = 0 ∧ B.2 - O.2 = 5) →   -- OB = (0, 5)
  (∃ k : ℝ, C.1 - A.1 = k * (B.1 - O.1) ∧ C.2 - A.2 = k * (B.2 - O.2)) →  -- AC parallel to OB
  ((C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2) = 0) →  -- BC perpendicular to AB
  C = (-3, 29/4) := by
sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l1636_163657


namespace NUMINAMATH_CALUDE_triangle_side_length_l1636_163611

/-- Given a triangle ABC with angle A = 60°, area = √3, and b + c = 6, prove that side length a = 2√6 -/
theorem triangle_side_length (b c : ℝ) (h1 : b + c = 6) (h2 : (1/2) * b * c * (Real.sqrt 3 / 2) = Real.sqrt 3) : 
  Real.sqrt (b^2 + c^2 - b * c) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1636_163611


namespace NUMINAMATH_CALUDE_decimal_multiplication_l1636_163641

/-- Given that 85 × 345 = 29325, prove that 0.085 × 3.45 = 0.29325 -/
theorem decimal_multiplication (h : 85 * 345 = 29325) : 0.085 * 3.45 = 0.29325 := by
  sorry

#check decimal_multiplication

end NUMINAMATH_CALUDE_decimal_multiplication_l1636_163641


namespace NUMINAMATH_CALUDE_unique_solution_l1636_163644

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  x^3 - 3*x = 4 - y ∧
  2*y^3 - 6*y = 6 - z ∧
  3*z^3 - 9*z = 8 - x

-- Theorem statement
theorem unique_solution :
  ∀ x y z : ℝ, system x y z → (x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l1636_163644


namespace NUMINAMATH_CALUDE_factorial_simplification_l1636_163608

theorem factorial_simplification (N : ℕ) :
  (Nat.factorial (N + 1)) / ((Nat.factorial (N - 1)) * (N + 2)) = N * (N + 1) / (N + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l1636_163608


namespace NUMINAMATH_CALUDE_power_sum_equals_two_l1636_163619

theorem power_sum_equals_two : (-1)^2 + (1/3)^0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_two_l1636_163619


namespace NUMINAMATH_CALUDE_negation_equivalence_l1636_163692

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1636_163692


namespace NUMINAMATH_CALUDE_arithmetic_progression_properties_l1636_163691

/-- An infinite arithmetic progression of natural numbers -/
structure ArithmeticProgression :=
  (a : ℕ)  -- First term
  (d : ℕ)  -- Common difference

/-- Predicate to check if a number is composite -/
def IsComposite (n : ℕ) : Prop := ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b

/-- Predicate to check if a number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k

/-- The nth term of an arithmetic progression -/
def nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℕ := ap.a + n * ap.d

theorem arithmetic_progression_properties (ap : ArithmeticProgression) :
  (∀ (N : ℕ), ∃ (n : ℕ), n > N ∧ IsComposite (nthTerm ap n)) ∧
  ((∀ (n : ℕ), ¬IsPerfectSquare (nthTerm ap n)) ∨
   (∀ (N : ℕ), ∃ (n : ℕ), n > N ∧ IsPerfectSquare (nthTerm ap n))) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_properties_l1636_163691


namespace NUMINAMATH_CALUDE_max_m_inequality_l1636_163674

theorem max_m_inequality (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (∃ m : ℝ, ∀ x y : ℝ, x > 1 → y > 1 → x^2 / (y - 1) + y^2 / (x - 1) ≥ 3 * m - 1) ∧
  (∀ m : ℝ, (∀ x y : ℝ, x > 1 → y > 1 → x^2 / (y - 1) + y^2 / (x - 1) ≥ 3 * m - 1) → m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_max_m_inequality_l1636_163674


namespace NUMINAMATH_CALUDE_larger_number_from_hcf_lcm_l1636_163672

theorem larger_number_from_hcf_lcm (a b : ℕ+) : 
  (Nat.gcd a b = 15) → 
  (Nat.lcm a b = 2475) → 
  (max a b = 225) :=
by sorry

end NUMINAMATH_CALUDE_larger_number_from_hcf_lcm_l1636_163672


namespace NUMINAMATH_CALUDE_triangular_coin_array_l1636_163646

theorem triangular_coin_array (N : ℕ) : (N * (N + 1)) / 2 = 3003 → N = 77 := by
  sorry

end NUMINAMATH_CALUDE_triangular_coin_array_l1636_163646


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1636_163635

/-- The side length of a square inscribed in a right triangle with sides 6, 8, and 10 -/
def inscribedSquareSideLength : ℚ := 60 / 31

/-- Right triangle ABC with square XYZW inscribed -/
structure InscribedSquare where
  -- Triangle side lengths
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Square side length
  s : ℝ
  -- Conditions
  right_triangle : AB^2 + BC^2 = AC^2
  AB_eq : AB = 6
  BC_eq : BC = 8
  AC_eq : AC = 10
  inscribed : s > 0 -- The square is inscribed (side length is positive)
  on_AC : s ≤ AC -- X and Y are on AC
  on_AB : s ≤ AB -- W is on AB
  on_BC : s ≤ BC -- Z is on BC

/-- The side length of the inscribed square is equal to 60/31 -/
theorem inscribed_square_side_length (square : InscribedSquare) :
  square.s = inscribedSquareSideLength := by sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l1636_163635


namespace NUMINAMATH_CALUDE_arccos_one_equals_zero_l1636_163639

theorem arccos_one_equals_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_equals_zero_l1636_163639


namespace NUMINAMATH_CALUDE_smallest_proportional_part_l1636_163616

theorem smallest_proportional_part (total : ℚ) (p1 p2 p3 : ℚ) (h1 : total = 130) 
  (h2 : p1 = 1) (h3 : p2 = 1/4) (h4 : p3 = 1/5) 
  (h5 : ∃ x : ℚ, x * p1 + x * p2 + x * p3 = total) : 
  min (x * p1) (min (x * p2) (x * p3)) = 2600/145 :=
sorry

end NUMINAMATH_CALUDE_smallest_proportional_part_l1636_163616


namespace NUMINAMATH_CALUDE_extended_triangle_properties_l1636_163671

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the circumcircle of a triangle
def circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Define the inradius of a triangle
def inradius (t : Triangle) : ℝ := sorry

-- Define the extended triangle DEF
def extendedTriangle (t : Triangle) : Triangle := sorry

-- Theorem statement
theorem extended_triangle_properties (t : Triangle) :
  let t' := extendedTriangle t
  perimeter t' ≥ perimeter t ∧ inradius t' > inradius t := by sorry

end NUMINAMATH_CALUDE_extended_triangle_properties_l1636_163671


namespace NUMINAMATH_CALUDE_flea_market_spending_l1636_163626

/-- Given that Jayda spent $400 and Aitana spent 2/5 times more than Jayda,
    prove that the total amount they spent together is $960. -/
theorem flea_market_spending (jayda_spent : ℝ) (aitana_ratio : ℝ) : 
  jayda_spent = 400 → 
  aitana_ratio = 2/5 → 
  jayda_spent + (jayda_spent + aitana_ratio * jayda_spent) = 960 := by
sorry

end NUMINAMATH_CALUDE_flea_market_spending_l1636_163626


namespace NUMINAMATH_CALUDE_marble_count_proof_l1636_163629

/-- The smallest positive integer greater than 1 that leaves a remainder of 1 when divided by 6, 7, and 8 -/
def smallest_marble_count : ℕ := 169

/-- Proves that the smallest_marble_count satisfies the given conditions -/
theorem marble_count_proof :
  smallest_marble_count > 1 ∧
  smallest_marble_count % 6 = 1 ∧
  smallest_marble_count % 7 = 1 ∧
  smallest_marble_count % 8 = 1 ∧
  ∀ n : ℕ, n > 1 →
    (n % 6 = 1 ∧ n % 7 = 1 ∧ n % 8 = 1) →
    n ≥ smallest_marble_count :=
by sorry

end NUMINAMATH_CALUDE_marble_count_proof_l1636_163629


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l1636_163670

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l1636_163670


namespace NUMINAMATH_CALUDE_side_to_hotdog_ratio_l1636_163699

def food_weights (chicken hamburger hotdog side : ℝ) : Prop :=
  chicken = 16 ∧
  hamburger = chicken / 2 ∧
  hotdog = hamburger + 2 ∧
  chicken + hamburger + hotdog + side = 39

theorem side_to_hotdog_ratio (chicken hamburger hotdog side : ℝ) :
  food_weights chicken hamburger hotdog side →
  side / hotdog = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_side_to_hotdog_ratio_l1636_163699


namespace NUMINAMATH_CALUDE_train_crossing_time_l1636_163695

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 750 ∧ 
  train_speed_kmh = 180 →
  crossing_time = 15 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1636_163695


namespace NUMINAMATH_CALUDE_unique_solution_product_l1636_163686

theorem unique_solution_product (r : ℝ) : 
  (∃! x : ℝ, (1 : ℝ) / (3 * x) = (r - 2 * x) / 10) → 
  (∃ r₁ r₂ : ℝ, r = r₁ ∨ r = r₂) ∧ r₁ * r₂ = -80 / 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_product_l1636_163686


namespace NUMINAMATH_CALUDE_analects_reasoning_is_common_sense_l1636_163614

-- Define the types of reasoning
inductive ReasoningType
  | CommonSense
  | Inductive
  | Analogical
  | Deductive

-- Define the structure of a statement in the reasoning chain
structure Statement :=
  (premise : String)
  (conclusion : String)

-- Define the passage from The Analects as a list of statements
def analectsPassage : List Statement :=
  [⟨"Names are not correct", "Language will not be in accordance with the truth of things"⟩,
   ⟨"Language is not in accordance with the truth of things", "Affairs cannot be carried out successfully"⟩,
   ⟨"Affairs cannot be carried out successfully", "Rites and music will not flourish"⟩,
   ⟨"Rites and music do not flourish", "Punishments will not be properly executed"⟩,
   ⟨"Punishments are not properly executed", "The people will have nowhere to put their hands and feet"⟩]

-- Define a function to determine the type of reasoning
def determineReasoningType (passage : List Statement) : ReasoningType := sorry

-- Theorem stating that the reasoning in the Analects passage is common sense reasoning
theorem analects_reasoning_is_common_sense :
  determineReasoningType analectsPassage = ReasoningType.CommonSense := by sorry

end NUMINAMATH_CALUDE_analects_reasoning_is_common_sense_l1636_163614


namespace NUMINAMATH_CALUDE_extreme_value_implies_n_eq_9_l1636_163613

/-- The function f(x) = x^3 + 6x^2 + nx + 4 -/
def f (n : ℝ) (x : ℝ) : ℝ := x^3 + 6*x^2 + n*x + 4

/-- f has an extreme value at x = -1 -/
def has_extreme_value_at_neg_one (n : ℝ) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 
    x ≠ -1 ∧ |x + 1| < ε → (f n x - f n (-1)) * (x + 1) ≤ 0

theorem extreme_value_implies_n_eq_9 :
  ∀ (n : ℝ), has_extreme_value_at_neg_one n → n = 9 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_n_eq_9_l1636_163613


namespace NUMINAMATH_CALUDE_divisibility_problem_l1636_163652

theorem divisibility_problem (n : ℕ) : 
  n > 2 → 
  (((1 + n + n * (n - 1) / 2 + n * (n - 1) * (n - 2) / 6) ∣ 2^2000) ↔ (n = 3 ∨ n = 7 ∨ n = 23)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1636_163652


namespace NUMINAMATH_CALUDE_line_range_l1636_163621

/-- A line that does not pass through the second quadrant -/
def line_not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (a + 1) * x + y + 2 - a = 0 → ¬(x < 0 ∧ y > 0)

/-- The theorem stating the range of 'a' for a line not passing through the second quadrant -/
theorem line_range (a : ℝ) : line_not_in_second_quadrant a → a ∈ Set.Iic (-1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_line_range_l1636_163621


namespace NUMINAMATH_CALUDE_system_solutions_l1636_163634

/-- System of equations -/
def system (x y z p : ℝ) : Prop :=
  x^2 - 3*y + p = z ∧ y^2 - 3*z + p = x ∧ z^2 - 3*x + p = y

theorem system_solutions :
  (∀ p : ℝ, p = 4 → ∀ x y z : ℝ, system x y z p → x = 2 ∧ y = 2 ∧ z = 2) ∧
  (∀ p : ℝ, 1 < p ∧ p < 4 → ∀ x y z : ℝ, system x y z p → x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l1636_163634


namespace NUMINAMATH_CALUDE_fraction_equality_l1636_163668

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 5) : a / (a + b) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1636_163668


namespace NUMINAMATH_CALUDE_final_price_calculation_l1636_163602

-- Define the original price
def original_price : ℝ := 120

-- Define the first discount rate
def first_discount_rate : ℝ := 0.20

-- Define the second discount rate
def second_discount_rate : ℝ := 0.15

-- Theorem to prove
theorem final_price_calculation :
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let final_price := price_after_first_discount * (1 - second_discount_rate)
  final_price = 81.60 := by
  sorry


end NUMINAMATH_CALUDE_final_price_calculation_l1636_163602


namespace NUMINAMATH_CALUDE_remainder_theorem_l1636_163662

theorem remainder_theorem (n : ℤ) : (7 - 2*n + (n + 5)) % 5 = (-n + 2) % 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1636_163662


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1636_163622

def A (a : ℝ) : Set ℝ := {x | x^2 < a^2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_implies_a_value (a : ℝ) 
  (h1 : Set.Nonempty (A a))
  (h2 : A a ∩ B = {x | 1 < x ∧ x < 2}) : 
  a = 2 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1636_163622


namespace NUMINAMATH_CALUDE_flour_needed_for_doubled_recipe_l1636_163630

theorem flour_needed_for_doubled_recipe 
  (original_recipe : ℕ) 
  (already_added : ℕ) 
  (h1 : original_recipe = 7)
  (h2 : already_added = 3) : 
  (2 * original_recipe) - already_added = 11 :=
by sorry

end NUMINAMATH_CALUDE_flour_needed_for_doubled_recipe_l1636_163630


namespace NUMINAMATH_CALUDE_evaluate_fraction_l1636_163603

theorem evaluate_fraction : (0.4 ^ 4) / (0.04 ^ 3) = 400 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_fraction_l1636_163603


namespace NUMINAMATH_CALUDE_yogurt_cases_l1636_163679

theorem yogurt_cases (total_cups : ℕ) (cups_per_box : ℕ) (boxes_per_case : ℕ) 
  (h1 : total_cups = 960) 
  (h2 : cups_per_box = 6) 
  (h3 : boxes_per_case = 8) : 
  (total_cups / cups_per_box) / boxes_per_case = 20 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_cases_l1636_163679


namespace NUMINAMATH_CALUDE_berries_and_coconut_cost_l1636_163651

/-- The cost of a bundle of berries and a coconut given the specified conditions -/
theorem berries_and_coconut_cost :
  ∀ (p b c d : ℚ),
  p + b + c + d = 30 →
  d = 3 * p →
  c = (p + b) / 2 →
  b + c = 65 / 9 := by
sorry

end NUMINAMATH_CALUDE_berries_and_coconut_cost_l1636_163651


namespace NUMINAMATH_CALUDE_sequence_properties_l1636_163609

/-- Given two sequences a and b with no equal items, and S_n as the sum of the first n terms of a. -/
def Sequence (a b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) * b n = S n + 1

theorem sequence_properties
  (a b : ℕ → ℝ) (S : ℕ → ℝ)
  (h_seq : Sequence a b S)
  (h_a1 : a 1 = 1)
  (h_bn : ∀ n, b n = n / 2)
  (h_geometric : ∃ q ≠ 1, ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : ∃ d, ∀ n, b (n + 1) = b n + d)
  (h_nonzero : ∀ n, a n ≠ 0) :
  (∃ q ≠ 1, ∀ n, (b n + 1 / (1 - q)) = (b 1 + 1 / (1 - q)) * q^(n - 1)) ∧
  (∀ n ≥ 2, a (n + 1) - a n = a n - a (n - 1) ↔ d = 1 / 2) :=
sorry

#check sequence_properties

end NUMINAMATH_CALUDE_sequence_properties_l1636_163609


namespace NUMINAMATH_CALUDE_fraction_irreducibility_l1636_163653

/-- The fraction (3n^2 + 2n + 4) / (n + 1) is irreducible if and only if n is not congruent to 4 modulo 5 -/
theorem fraction_irreducibility (n : ℤ) : 
  (Int.gcd (3*n^2 + 2*n + 4) (n + 1) = 1) ↔ (n % 5 ≠ 4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducibility_l1636_163653


namespace NUMINAMATH_CALUDE_insufficient_blue_points_l1636_163618

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A tetrahedron formed by four points in 3D space -/
structure Tetrahedron where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D

/-- Checks if a point is inside a tetrahedron -/
def isInside (t : Tetrahedron) (p : Point3D) : Prop := sorry

/-- The set of all tetrahedra formed by n red points -/
def allTetrahedra (redPoints : Finset Point3D) : Finset Tetrahedron := sorry

/-- Theorem: There exists a configuration of n red points such that 3n blue points
    are not sufficient to cover all tetrahedra formed by the red points -/
theorem insufficient_blue_points (n : ℕ) :
  ∃ (redPoints : Finset Point3D),
    redPoints.card = n ∧
    ∀ (bluePoints : Finset Point3D),
      bluePoints.card = 3 * n →
      ∃ (t : Tetrahedron),
        t ∈ allTetrahedra redPoints ∧
        ∀ (p : Point3D), p ∈ bluePoints → ¬isInside t p :=
sorry

end NUMINAMATH_CALUDE_insufficient_blue_points_l1636_163618


namespace NUMINAMATH_CALUDE_fraction_of_girls_at_event_l1636_163694

theorem fraction_of_girls_at_event (maplewood_total : ℕ) (brookside_total : ℕ)
  (maplewood_boy_ratio maplewood_girl_ratio : ℕ)
  (brookside_boy_ratio brookside_girl_ratio : ℕ)
  (h1 : maplewood_total = 300)
  (h2 : brookside_total = 240)
  (h3 : maplewood_boy_ratio = 3)
  (h4 : maplewood_girl_ratio = 2)
  (h5 : brookside_boy_ratio = 2)
  (h6 : brookside_girl_ratio = 3) :
  (maplewood_total * maplewood_girl_ratio / (maplewood_boy_ratio + maplewood_girl_ratio) +
   brookside_total * brookside_girl_ratio / (brookside_boy_ratio + brookside_girl_ratio)) /
  (maplewood_total + brookside_total) = 22 / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_girls_at_event_l1636_163694


namespace NUMINAMATH_CALUDE_periodicity_theorem_l1636_163688

/-- A polynomial with rational coefficients -/
def RationalPolynomial := Polynomial ℚ

/-- A sequence of rational numbers -/
def RationalSequence := ℕ → ℚ

/-- The statement of the periodicity theorem -/
theorem periodicity_theorem
  (p : RationalPolynomial)
  (q : RationalSequence)
  (h1 : p.degree ≥ 2)
  (h2 : ∀ n : ℕ, n ≥ 1 → q n = p.eval (q (n + 1))) :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n ≥ 1 → q (n + k) = q n :=
sorry

end NUMINAMATH_CALUDE_periodicity_theorem_l1636_163688


namespace NUMINAMATH_CALUDE_largest_integer_below_sqrt_sum_power_l1636_163636

theorem largest_integer_below_sqrt_sum_power : 
  ∃ n : ℕ, n = 7168 ∧ n < (Real.sqrt 5 + Real.sqrt (3/2))^8 ∧ 
  ∀ m : ℕ, m < (Real.sqrt 5 + Real.sqrt (3/2))^8 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_largest_integer_below_sqrt_sum_power_l1636_163636


namespace NUMINAMATH_CALUDE_base6_product_132_14_l1636_163600

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Multiplies two base 6 numbers --/
def multiplyBase6 (a b : List Nat) : List Nat :=
  base10ToBase6 (base6ToBase10 a * base6ToBase10 b)

theorem base6_product_132_14 :
  multiplyBase6 [2, 3, 1] [4, 1] = [2, 3, 3, 2] := by sorry

end NUMINAMATH_CALUDE_base6_product_132_14_l1636_163600


namespace NUMINAMATH_CALUDE_postman_speed_calculation_postman_speed_is_30_l1636_163667

/-- Calculates the downhill average speed of a postman's round trip, given the following conditions:
  * The route length is 5 miles each way
  * The uphill delivery takes 2 hours
  * The uphill average speed is 4 miles per hour
  * The overall average speed for the round trip is 6 miles per hour
  * There's an extra 15 minutes (0.25 hours) delay on the return trip due to rain
-/
theorem postman_speed_calculation (route_length : ℝ) (uphill_time : ℝ) (uphill_speed : ℝ) 
  (overall_speed : ℝ) (rain_delay : ℝ) : ℝ :=
  let downhill_speed := 
    route_length / (((2 * route_length) / overall_speed) - uphill_time - rain_delay)
  30

/-- The main theorem that proves the downhill speed is 30 mph given the specific conditions -/
theorem postman_speed_is_30 : 
  postman_speed_calculation 5 2 4 6 0.25 = 30 := by
  sorry

end NUMINAMATH_CALUDE_postman_speed_calculation_postman_speed_is_30_l1636_163667


namespace NUMINAMATH_CALUDE_x_n_root_bound_l1636_163620

theorem x_n_root_bound (n : ℕ) (a : ℝ) (x : ℕ → ℝ) (α : ℕ → ℝ)
  (hn : n > 1)
  (ha : a ≥ 1)
  (hx1 : x 1 = 1)
  (hxi : ∀ i ∈ Finset.range n, i ≥ 2 → x i / x (i-1) = a + α i)
  (hαi : ∀ i ∈ Finset.range n, i ≥ 2 → α i ≤ 1 / (i * (i + 1))) :
  (x n) ^ (1 / (n - 1 : ℝ)) < a + 1 / (n - 1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_x_n_root_bound_l1636_163620


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_m_range_l1636_163682

/-- An ellipse with equation x²/5 + y²/m = 1 -/
def isEllipse (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2/5 + y^2/m = 1 ∧ m ≠ 0 ∧ m ≠ 5

/-- A hyperbola with equation x²/5 + y²/(m-6) = 1 -/
def isHyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2/5 + y^2/(m-6) = 1 ∧ m ≠ 6

/-- The range of valid m values -/
def validRange (m : ℝ) : Prop :=
  (0 < m ∧ m < 5) ∨ (5 < m ∧ m < 6)

theorem ellipse_hyperbola_m_range :
  ∀ m : ℝ, (isEllipse m ∧ isHyperbola m) ↔ validRange m :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_m_range_l1636_163682


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1636_163683

def complex_power (z : ℂ) (n : ℕ) := z ^ n

theorem complex_equation_sum (a b : ℝ) :
  (↑a + ↑b * Complex.I : ℂ) = complex_power Complex.I 2019 →
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1636_163683


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1636_163612

theorem complex_equation_solution (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) : 
  z = -Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1636_163612


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l1636_163601

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l1636_163601


namespace NUMINAMATH_CALUDE_complex_arithmetic_proof_l1636_163647

theorem complex_arithmetic_proof :
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -5*I
  let R : ℂ := 1 + I
  let T : ℂ := 3 - 4*I
  B * R + Q + T = 4 + I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_proof_l1636_163647


namespace NUMINAMATH_CALUDE_coin_difference_l1636_163628

def coin_values : List ℕ := [5, 10, 25, 50]

def target_amount : ℕ := 75

def is_valid_combination (combination : List ℕ) : Prop :=
  combination.all (λ x => x ∈ coin_values) ∧
  combination.sum = target_amount

def num_coins (combination : List ℕ) : ℕ := combination.length

theorem coin_difference :
  ∃ (max_combination min_combination : List ℕ),
    is_valid_combination max_combination ∧
    is_valid_combination min_combination ∧
    (∀ c, is_valid_combination c →
      num_coins c ≤ num_coins max_combination ∧
      num_coins c ≥ num_coins min_combination) ∧
    num_coins max_combination - num_coins min_combination = 13 :=
  sorry

end NUMINAMATH_CALUDE_coin_difference_l1636_163628


namespace NUMINAMATH_CALUDE_savings_duration_l1636_163655

/-- Thomas and Joseph's savings problem -/
theorem savings_duration : 
  ∀ (thomas_monthly joseph_monthly total_savings : ℚ),
  thomas_monthly = 40 →
  joseph_monthly = (3/5) * thomas_monthly →
  total_savings = 4608 →
  ∃ (months : ℕ), 
    (thomas_monthly + joseph_monthly) * months = total_savings ∧ 
    months = 72 := by
  sorry

end NUMINAMATH_CALUDE_savings_duration_l1636_163655


namespace NUMINAMATH_CALUDE_ellipse_min_major_axis_l1636_163633

/-- Given an ellipse where the maximum area of a triangle formed by a point
    on the ellipse and its two foci is 1, the minimum value of its major axis is 2√2. -/
theorem ellipse_min_major_axis (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_ellipse : a^2 = b^2 + c^2) (h_area : b * c = 1) :
  2 * a ≥ 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_min_major_axis_l1636_163633


namespace NUMINAMATH_CALUDE_salesman_commission_percentage_l1636_163604

/-- Proves that the flat commission percentage in the previous scheme is 5% --/
theorem salesman_commission_percentage :
  ∀ (previous_commission_percentage : ℝ),
    -- New scheme: Rs. 1000 fixed salary + 2.5% commission on sales exceeding Rs. 4,000
    let new_scheme_fixed_salary : ℝ := 1000
    let new_scheme_commission_rate : ℝ := 2.5 / 100
    let sales_threshold : ℝ := 4000
    -- Total sales
    let total_sales : ℝ := 12000
    -- Calculate new scheme remuneration
    let new_scheme_commission : ℝ := new_scheme_commission_rate * (total_sales - sales_threshold)
    let new_scheme_remuneration : ℝ := new_scheme_fixed_salary + new_scheme_commission
    -- Previous scheme remuneration
    let previous_scheme_remuneration : ℝ := previous_commission_percentage / 100 * total_sales
    -- New scheme remuneration is Rs. 600 more than the previous scheme
    new_scheme_remuneration = previous_scheme_remuneration + 600
    →
    previous_commission_percentage = 5 := by
  sorry

end NUMINAMATH_CALUDE_salesman_commission_percentage_l1636_163604


namespace NUMINAMATH_CALUDE_red_hair_ratio_l1636_163624

theorem red_hair_ratio (red_hair_count : ℕ) (total_count : ℕ) 
  (h1 : red_hair_count = 9) 
  (h2 : total_count = 48) : 
  (red_hair_count : ℚ) / total_count = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_red_hair_ratio_l1636_163624


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l1636_163648

/-- The number of diagonals in a convex n-gon --/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Proof that a convex heptagon has 14 diagonals --/
theorem heptagon_diagonals : num_diagonals 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l1636_163648


namespace NUMINAMATH_CALUDE_martyrs_cemetery_distance_l1636_163607

/-- The distance from the school to the Martyrs' Cemetery in meters -/
def distance : ℝ := 180000

/-- The original speed of the car in meters per minute -/
def original_speed : ℝ := 500

/-- The scheduled travel time in minutes -/
def scheduled_time : ℝ := 120

theorem martyrs_cemetery_distance :
  ∃ (d : ℝ) (v : ℝ),
    d = distance ∧
    v = original_speed ∧
    -- Condition 1: Increased speed by 1/5 after 1 hour
    (60 / v + (d - 60 * v) / (6/5 * v) = scheduled_time - 10) ∧
    -- Condition 2: Increased speed by 1/3 after 60 km
    (60000 / v + (d - 60000) / (4/3 * v) = scheduled_time - 20) ∧
    -- Scheduled time is 2 hours
    scheduled_time = 120 :=
by sorry

end NUMINAMATH_CALUDE_martyrs_cemetery_distance_l1636_163607


namespace NUMINAMATH_CALUDE_function_inequality_l1636_163640

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the theorem
theorem function_inequality (h : ∀ x, f' x > f x) : 3 * f (log 2) < 2 * f (log 3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1636_163640


namespace NUMINAMATH_CALUDE_value_of_4x_l1636_163681

theorem value_of_4x (x : ℝ) (h : 2 * x - 3 = 10) : 4 * x = 26 := by
  sorry

end NUMINAMATH_CALUDE_value_of_4x_l1636_163681


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1636_163658

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxesEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxesEllipse) : ℝ := sorry

theorem ellipse_foci_distance 
  (e : ParallelAxesEllipse) 
  (h1 : e.x_tangent = (8, 0)) 
  (h2 : e.y_tangent = (0, 2)) : 
  foci_distance e = 4 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1636_163658


namespace NUMINAMATH_CALUDE_company_employees_l1636_163637

theorem company_employees (female_managers : ℕ) (male_female_ratio : ℚ) 
  (total_manager_ratio : ℚ) (male_manager_ratio : ℚ) :
  female_managers = 200 →
  male_female_ratio = 3 / 2 →
  total_manager_ratio = 2 / 5 →
  male_manager_ratio = 2 / 5 →
  ∃ (female_employees : ℕ) (total_employees : ℕ),
    female_employees = 500 ∧
    total_employees = 1250 := by
  sorry

end NUMINAMATH_CALUDE_company_employees_l1636_163637


namespace NUMINAMATH_CALUDE_k_range_l1636_163643

/-- The ellipse equation -/
def ellipse_eq (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0

/-- The origin (0,0) is inside the ellipse -/
def origin_inside (k : ℝ) : Prop :=
  ∃ ε > 0, ∀ x y : ℝ, x^2 + y^2 < ε^2 → ellipse_eq k x y

/-- The theorem stating the range of k -/
theorem k_range (k : ℝ) (h : origin_inside k) : 0 < |k| ∧ |k| < 1 :=
sorry

end NUMINAMATH_CALUDE_k_range_l1636_163643


namespace NUMINAMATH_CALUDE_trivia_team_size_l1636_163663

theorem trivia_team_size (absent_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) :
  absent_members = 2 →
  points_per_member = 6 →
  total_points = 18 →
  ∃ initial_members : ℕ, initial_members = 5 ∧ 
    points_per_member * (initial_members - absent_members) = total_points :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_size_l1636_163663


namespace NUMINAMATH_CALUDE_train_crossing_time_l1636_163627

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 300 →
  train_speed_kmh = 120 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 9 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1636_163627


namespace NUMINAMATH_CALUDE_polynomial_nonnegative_l1636_163687

theorem polynomial_nonnegative (a : ℝ) : a^2 * (a^2 - 1) - a^2 + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_nonnegative_l1636_163687


namespace NUMINAMATH_CALUDE_complex_fraction_power_l1636_163649

theorem complex_fraction_power (i : ℂ) (a b : ℝ) :
  i * i = -1 →
  (1 : ℂ) / (1 + i) = a + b * i →
  a ^ b = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l1636_163649


namespace NUMINAMATH_CALUDE_circle_covers_three_points_l1636_163664

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point is inside a unit square -/
def isInUnitSquare (p : Point) : Prop :=
  0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1

/-- Predicate to check if a point is inside a circle -/
def isInCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

theorem circle_covers_three_points (points : Finset Point) 
    (h1 : points.card = 51)
    (h2 : ∀ p ∈ points, isInUnitSquare p) :
    ∃ (c : Circle), c.radius = 1/7 ∧ (∃ (p1 p2 p3 : Point), 
      p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
      p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
      isInCircle p1 c ∧ isInCircle p2 c ∧ isInCircle p3 c) := by
  sorry

end NUMINAMATH_CALUDE_circle_covers_three_points_l1636_163664


namespace NUMINAMATH_CALUDE_angle_terminal_side_l1636_163623

open Real

theorem angle_terminal_side (α : Real) :
  (tan α < 0 ∧ cos α < 0) → 
  (π / 2 < α ∧ α < π) :=
by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l1636_163623


namespace NUMINAMATH_CALUDE_fraction_addition_l1636_163675

theorem fraction_addition (x y : ℚ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1636_163675


namespace NUMINAMATH_CALUDE_officers_on_duty_l1636_163673

theorem officers_on_duty (total_female_officers : ℕ) 
  (female_on_duty_ratio : ℚ) (female_ratio_on_duty : ℚ) :
  total_female_officers = 250 →
  female_on_duty_ratio = 1/5 →
  female_ratio_on_duty = 1/2 →
  (female_on_duty_ratio * total_female_officers : ℚ) / female_ratio_on_duty = 100 := by
  sorry

end NUMINAMATH_CALUDE_officers_on_duty_l1636_163673


namespace NUMINAMATH_CALUDE_diagonal_difference_l1636_163676

/-- The number of diagonals in a convex polygon with n sides -/
def f (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The difference between the number of diagonals in a convex polygon
    with n+1 sides and n sides is n-1, for n ≥ 4 -/
theorem diagonal_difference (n : ℕ) (h : n ≥ 4) : f (n + 1) - f n = n - 1 :=
  sorry

end NUMINAMATH_CALUDE_diagonal_difference_l1636_163676


namespace NUMINAMATH_CALUDE_partnership_investment_ratio_l1636_163697

/-- Partnership investment problem -/
theorem partnership_investment_ratio :
  ∀ (n : ℚ) (c : ℚ),
  c > 0 →  -- C's investment is positive
  (2 / 3 * c) / (n * (2 / 3 * c) + (2 / 3 * c) + c) = 800 / 4400 →
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_partnership_investment_ratio_l1636_163697


namespace NUMINAMATH_CALUDE_exists_three_adjacent_sum_exceeds_17_l1636_163696

-- Define a type for jersey numbers
def JerseyNumber := Fin 10

-- Define a type for the circular arrangement of players
def CircularArrangement := Fin 10 → JerseyNumber

-- Define a function to check if three consecutive numbers sum to more than 17
def SumExceeds17 (arrangement : CircularArrangement) (i : Fin 10) : Prop :=
  (arrangement i).val + (arrangement (i + 1)).val + (arrangement (i + 2)).val > 17

-- Theorem statement
theorem exists_three_adjacent_sum_exceeds_17 (arrangement : CircularArrangement) :
  (∀ i j : Fin 10, i ≠ j → arrangement i ≠ arrangement j) →
  ∃ i : Fin 10, SumExceeds17 arrangement i := by
  sorry

end NUMINAMATH_CALUDE_exists_three_adjacent_sum_exceeds_17_l1636_163696


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1636_163669

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 3 + a 9 = 20) :
  4 * a 5 - a 7 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1636_163669


namespace NUMINAMATH_CALUDE_square_sum_inequality_l1636_163642

theorem square_sum_inequality (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (2 + a) * (2 + b) ≥ c * d := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l1636_163642
