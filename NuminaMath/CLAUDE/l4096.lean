import Mathlib

namespace NUMINAMATH_CALUDE_number_division_problem_l4096_409622

theorem number_division_problem : ∃ N : ℕ, 
  (N / (555 + 445) = 2 * (555 - 445)) ∧ 
  (N % (555 + 445) = 30) ∧ 
  (N = 220030) := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l4096_409622


namespace NUMINAMATH_CALUDE_max_initial_pieces_is_285_l4096_409649

/-- Represents a Go board with dimensions n x n -/
structure GoBoard (n : ℕ) where
  size : n > 0

/-- Represents a rectangular arrangement of pieces on a Go board -/
structure Rectangle (n : ℕ) where
  width : ℕ
  height : ℕ
  pieces : ℕ
  width_valid : width ≤ n
  height_valid : height ≤ n
  area_eq_pieces : width * height = pieces

/-- The maximum number of pieces in the initial rectangle -/
def max_initial_pieces (board : GoBoard 19) : ℕ := 285

/-- Theorem stating the maximum number of pieces in the initial rectangle -/
theorem max_initial_pieces_is_285 (board : GoBoard 19) :
  ∃ (init final : Rectangle 19),
    init.pieces = max_initial_pieces board ∧
    final.pieces = init.pieces + 45 ∧
    final.width = init.width ∧
    final.height > init.height ∧
    ∀ (other : Rectangle 19),
      (∃ (other_final : Rectangle 19),
        other_final.pieces = other.pieces + 45 ∧
        other_final.width = other.width ∧
        other_final.height > other.height) →
      other.pieces ≤ init.pieces :=
by sorry

end NUMINAMATH_CALUDE_max_initial_pieces_is_285_l4096_409649


namespace NUMINAMATH_CALUDE_last_four_matches_average_l4096_409603

/-- Represents a cricket scoring scenario -/
structure CricketScoring where
  totalMatches : Nat
  firstMatchesCount : Nat
  totalAverage : ℚ
  firstMatchesAverage : ℚ

/-- Calculates the average score of the remaining matches -/
def remainingMatchesAverage (cs : CricketScoring) : ℚ :=
  let totalRuns := cs.totalAverage * cs.totalMatches
  let firstMatchesRuns := cs.firstMatchesAverage * cs.firstMatchesCount
  let remainingMatchesCount := cs.totalMatches - cs.firstMatchesCount
  (totalRuns - firstMatchesRuns) / remainingMatchesCount

/-- Theorem stating that under the given conditions, the average of the last 4 matches is 34.25 -/
theorem last_four_matches_average (cs : CricketScoring) 
  (h1 : cs.totalMatches = 10)
  (h2 : cs.firstMatchesCount = 6)
  (h3 : cs.totalAverage = 389/10)
  (h4 : cs.firstMatchesAverage = 42) :
  remainingMatchesAverage cs = 137/4 := by
  sorry

end NUMINAMATH_CALUDE_last_four_matches_average_l4096_409603


namespace NUMINAMATH_CALUDE_angle_bisector_implies_line_equation_l4096_409675

/-- Given points A and B, and a line representing the angle bisector of ∠ACB,
    prove that the line AC has the equation x - 2y - 1 = 0 -/
theorem angle_bisector_implies_line_equation 
  (A B : ℝ × ℝ)
  (angle_bisector : ℝ → ℝ)
  (h1 : A = (3, 1))
  (h2 : B = (-1, 2))
  (h3 : ∀ x y, y = angle_bisector x ↔ y = x + 1)
  (h4 : ∃ C : ℝ × ℝ, (angle_bisector (C.1) = C.2) ∧ 
       (∃ t : ℝ, C = (1 - t) • A + t • B) ∧
       (∃ s : ℝ, C = (1 - s) • A' + s • B)) 
  (A' : ℝ × ℝ)
  (h5 : A'.2 - 1 = -(A'.1 - 3))  -- Reflection condition
  (h6 : (A'.2 + 1) / 2 = (A'.1 + 3) / 2 + 1)  -- Reflection condition
  : ∀ x y, x - 2*y - 1 = 0 ↔ ∃ t : ℝ, (x, y) = (1 - t) • A + t • C :=
sorry


end NUMINAMATH_CALUDE_angle_bisector_implies_line_equation_l4096_409675


namespace NUMINAMATH_CALUDE_sequence_sum_l4096_409694

/-- Given a geometric sequence a, b, c and arithmetic sequences a, x, b and b, y, c, 
    prove that a/x + c/y = 2 -/
theorem sequence_sum (a b c x y : ℝ) 
  (h_geom : b^2 = a*c) 
  (h_arith1 : x = (a + b)/2) 
  (h_arith2 : y = (b + c)/2) : 
  a/x + c/y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l4096_409694


namespace NUMINAMATH_CALUDE_expand_expression_l4096_409668

theorem expand_expression (x y : ℝ) : 
  (5 * x^2 - 3/2 * y) * (-4 * x^3 * y^2) = -20 * x^5 * y^2 + 6 * x^3 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l4096_409668


namespace NUMINAMATH_CALUDE_smallest_divisible_by_million_l4096_409623

/-- Represents a geometric sequence with first term a and common ratio r -/
def GeometricSequence (a : ℚ) (r : ℚ) : ℕ → ℚ := λ n => a * r^(n - 1)

/-- The nth term of the specific geometric sequence in the problem -/
def SpecificSequence : ℕ → ℚ := GeometricSequence (1/2) 60

/-- Predicate to check if a rational number is divisible by one million -/
def DivisibleByMillion (q : ℚ) : Prop := ∃ (k : ℤ), q = (k : ℚ) * 1000000

theorem smallest_divisible_by_million :
  (∀ n < 7, ¬ DivisibleByMillion (SpecificSequence n)) ∧
  DivisibleByMillion (SpecificSequence 7) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_million_l4096_409623


namespace NUMINAMATH_CALUDE_max_area_special_quadrilateral_l4096_409698

/-- A quadrilateral with the property that the product of any two adjacent sides is 1 -/
structure SpecialQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  ab_eq_one : a * b = 1
  bc_eq_one : b * c = 1
  cd_eq_one : c * d = 1
  da_eq_one : d * a = 1

/-- The area of a quadrilateral -/
def area (q : SpecialQuadrilateral) : ℝ := sorry

/-- The maximum area of a SpecialQuadrilateral is 1 -/
theorem max_area_special_quadrilateral :
  ∀ q : SpecialQuadrilateral, area q ≤ 1 ∧ ∃ q' : SpecialQuadrilateral, area q' = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_area_special_quadrilateral_l4096_409698


namespace NUMINAMATH_CALUDE_tangent_circles_m_value_l4096_409686

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x + m^2 - 1 = 0

-- Define what it means for two circles to be externally tangent
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y m ∧
  ∀ (x' y' : ℝ), circle1 x' y' ∧ circle2 x' y' m → (x = x' ∧ y = y')

-- State the theorem
theorem tangent_circles_m_value (m : ℝ) :
  externally_tangent m → (m = 3 ∨ m = -3) :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_m_value_l4096_409686


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l4096_409659

/-- For positive real numbers a, b, c with abc = 1, the sum S = 1/(2a+1) + 1/(2b+1) + 1/(2c+1) is greater than or equal to 1 -/
theorem min_sum_reciprocals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  1 / (2 * a + 1) + 1 / (2 * b + 1) + 1 / (2 * c + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l4096_409659


namespace NUMINAMATH_CALUDE_quadratic_roots_and_k_value_l4096_409692

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := x^2 + (2*k + 1)*x + k^2 + 1

-- Theorem statement
theorem quadratic_roots_and_k_value (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0) ↔ k > 3/4 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 ∧ x₁ * x₂ = 5) → k = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_k_value_l4096_409692


namespace NUMINAMATH_CALUDE_otimes_nested_equality_l4096_409636

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^3 + 5*x*y - y

theorem otimes_nested_equality (a : ℝ) : otimes a (otimes a a) = 5*a^4 + 24*a^3 - 10*a^2 + a := by
  sorry

end NUMINAMATH_CALUDE_otimes_nested_equality_l4096_409636


namespace NUMINAMATH_CALUDE_red_bottle_caps_l4096_409631

theorem red_bottle_caps (total : ℕ) (green_percentage : ℚ) : 
  total = 125 → green_percentage = 60 / 100 → 
  (total : ℚ) * (1 - green_percentage) = 50 := by
sorry

end NUMINAMATH_CALUDE_red_bottle_caps_l4096_409631


namespace NUMINAMATH_CALUDE_new_apartment_rent_is_1400_l4096_409615

/-- Calculates the monthly rent of John's new apartment -/
def new_apartment_rent (former_rent_per_sqft : ℚ) (former_sqft : ℕ) (annual_savings : ℚ) : ℚ :=
  let former_monthly_rent := former_rent_per_sqft * former_sqft
  let former_annual_rent := former_monthly_rent * 12
  let new_annual_rent := former_annual_rent - annual_savings
  new_annual_rent / 12

/-- Proves that the monthly rent of John's new apartment is $1400 -/
theorem new_apartment_rent_is_1400 :
  new_apartment_rent 2 750 1200 = 1400 := by sorry

end NUMINAMATH_CALUDE_new_apartment_rent_is_1400_l4096_409615


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l4096_409671

theorem lcm_of_ratio_and_hcf (a b : ℕ) : 
  a ≠ 0 → b ≠ 0 → a * 4 = b * 3 → Nat.gcd a b = 8 → Nat.lcm a b = 96 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l4096_409671


namespace NUMINAMATH_CALUDE_max_knights_and_courtiers_l4096_409635

def king_table_size : ℕ := 7
def min_courtiers : ℕ := 12
def max_courtiers : ℕ := 18
def min_knights : ℕ := 10
def max_knights : ℕ := 20

def is_valid_solution (courtiers knights : ℕ) : Prop :=
  min_courtiers ≤ courtiers ∧ courtiers ≤ max_courtiers ∧
  min_knights ≤ knights ∧ knights ≤ max_knights ∧
  (1 : ℚ) / courtiers + (1 : ℚ) / knights = (1 : ℚ) / king_table_size

theorem max_knights_and_courtiers :
  ∃ (max_knights courtiers : ℕ),
    is_valid_solution courtiers max_knights ∧
    ∀ (k : ℕ), is_valid_solution courtiers k → k ≤ max_knights ∧
    max_knights = 14 ∧ courtiers = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_knights_and_courtiers_l4096_409635


namespace NUMINAMATH_CALUDE_prob_12th_roll_last_correct_l4096_409682

/-- The probability of the 12th roll being the last roll when rolling a standard six-sided die
    until getting the same number on consecutive rolls -/
def prob_12th_roll_last : ℚ := (5^10 : ℚ) / (6^11 : ℚ)

/-- Theorem stating that the probability of the 12th roll being the last roll is correct -/
theorem prob_12th_roll_last_correct :
  prob_12th_roll_last = (5^10 : ℚ) / (6^11 : ℚ) := by sorry

end NUMINAMATH_CALUDE_prob_12th_roll_last_correct_l4096_409682


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_regular_polygon_perimeter_proof_l4096_409664

/-- A regular polygon with side length 6 and exterior angle 90 degrees has a perimeter of 24 units. -/
theorem regular_polygon_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun (side_length : ℝ) (exterior_angle : ℝ) (perimeter : ℝ) =>
    side_length = 6 ∧
    exterior_angle = 90 ∧
    perimeter = 24

/-- The theorem statement -/
theorem regular_polygon_perimeter_proof :
  ∃ (side_length exterior_angle perimeter : ℝ),
    regular_polygon_perimeter side_length exterior_angle perimeter :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_regular_polygon_perimeter_proof_l4096_409664


namespace NUMINAMATH_CALUDE_x_greater_y_greater_z_l4096_409629

theorem x_greater_y_greater_z (α b x y z : Real) 
  (h_α : α ∈ Set.Ioo (π / 4) (π / 2))
  (h_b : b ∈ Set.Ioo 0 1)
  (h_x : Real.log x = (Real.log (Real.sin α))^2 / Real.log b)
  (h_y : Real.log y = (Real.log (Real.cos α))^2 / Real.log b)
  (h_z : Real.log z = (Real.log (Real.sin α * Real.cos α))^2 / Real.log b) :
  x > y ∧ y > z := by
  sorry

end NUMINAMATH_CALUDE_x_greater_y_greater_z_l4096_409629


namespace NUMINAMATH_CALUDE_value_of_y_l4096_409653

theorem value_of_y (y : ℚ) (h : 2/3 - 3/5 = 5/y) : y = 75 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l4096_409653


namespace NUMINAMATH_CALUDE_min_sum_of_square_roots_l4096_409612

theorem min_sum_of_square_roots (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ x : ℝ, Real.sqrt ((x - a)^2 + b^2) + Real.sqrt ((x - b)^2 + a^2) ≥ Real.sqrt (2 * (a^2 + b^2))) ∧
  (∃ x : ℝ, Real.sqrt ((x - a)^2 + b^2) + Real.sqrt ((x - b)^2 + a^2) = Real.sqrt (2 * (a^2 + b^2))) :=
by
  sorry

#check min_sum_of_square_roots

end NUMINAMATH_CALUDE_min_sum_of_square_roots_l4096_409612


namespace NUMINAMATH_CALUDE_x_value_in_set_l4096_409657

theorem x_value_in_set (x : ℝ) : x ∈ ({1, 2, x^2 - x} : Set ℝ) → x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_set_l4096_409657


namespace NUMINAMATH_CALUDE_closest_point_is_A_l4096_409632

-- Define the points as real numbers
variable (A B C D E : ℝ)

-- Define the conditions
axiom A_range : 0 < A ∧ A < 1
axiom B_range : 0 < B ∧ B < 1
axiom C_range : 0 < C ∧ C < 1
axiom D_range : 0 < D ∧ D < 1
axiom E_range : 1 < E ∧ E < 2

-- Define the order of points
axiom point_order : A < B ∧ B < C ∧ C < D

-- Define a function to calculate the distance between two real numbers
def distance (x y : ℝ) : ℝ := |x - y|

-- State the theorem
theorem closest_point_is_A :
  distance (B * C) A < distance (B * C) B ∧
  distance (B * C) A < distance (B * C) C ∧
  distance (B * C) A < distance (B * C) D ∧
  distance (B * C) A < distance (B * C) E :=
sorry

end NUMINAMATH_CALUDE_closest_point_is_A_l4096_409632


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l4096_409654

-- Define the quadratic function
def f (x : ℝ) : ℝ := x * (1 - 3 * x)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x > 0} = Set.Ioo 0 (1/3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l4096_409654


namespace NUMINAMATH_CALUDE_hemisphere_radius_from_cylinder_l4096_409660

/-- The radius of a hemisphere formed from a cylinder of equal volume --/
theorem hemisphere_radius_from_cylinder (r h R : ℝ) : 
  r = 2 * (2 : ℝ)^(1/3) → 
  h = 12 → 
  π * r^2 * h = (2/3) * π * R^3 → 
  R = 2 * 3^(1/3) := by
sorry

end NUMINAMATH_CALUDE_hemisphere_radius_from_cylinder_l4096_409660


namespace NUMINAMATH_CALUDE_monotone_cubic_implies_m_bound_l4096_409656

/-- A function f: ℝ → ℝ is monotonically increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The cubic function f(x) = x³ + 2x² + mx - 5 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x - 5

theorem monotone_cubic_implies_m_bound :
  ∀ m : ℝ, MonotonicallyIncreasing (f m) → m ≥ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_monotone_cubic_implies_m_bound_l4096_409656


namespace NUMINAMATH_CALUDE_parabola_equation_l4096_409699

/-- A parabola with vertex at the origin, axis of symmetry along the y-axis,
    and distance between vertex and focus equal to 6 -/
structure Parabola where
  vertex : ℝ × ℝ
  axis_of_symmetry : ℝ → ℝ
  focus_distance : ℝ
  h_vertex : vertex = (0, 0)
  h_axis : axis_of_symmetry = fun y => 0
  h_focus : focus_distance = 6

/-- The standard equation of the parabola -/
def standard_equation (p : Parabola) : Prop :=
  ∀ x y : ℝ, (x^2 = 24*y ∨ x^2 = -24*y) ↔ (x, y) ∈ {(x, y) | p.axis_of_symmetry x = y}

/-- Theorem stating that the standard equation holds for the given parabola -/
theorem parabola_equation (p : Parabola) : standard_equation p := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l4096_409699


namespace NUMINAMATH_CALUDE_probability_not_red_marble_l4096_409696

theorem probability_not_red_marble (total : ℕ) (red green yellow blue : ℕ) 
  (h1 : total = red + green + yellow + blue)
  (h2 : red = 8)
  (h3 : green = 10)
  (h4 : yellow = 12)
  (h5 : blue = 15) :
  (green + yellow + blue : ℚ) / total = 37 / 45 := by
sorry

end NUMINAMATH_CALUDE_probability_not_red_marble_l4096_409696


namespace NUMINAMATH_CALUDE_arithmetic_progression_theorem_l4096_409600

/-- Represents an arithmetic progression of five terms. -/
structure ArithmeticProgression :=
  (a : ℝ)  -- First term
  (d : ℝ)  -- Common difference

/-- Checks if the arithmetic progression is decreasing. -/
def ArithmeticProgression.isDecreasing (ap : ArithmeticProgression) : Prop :=
  ap.d > 0

/-- Calculates the sum of cubes of the terms in the arithmetic progression. -/
def ArithmeticProgression.sumOfCubes (ap : ArithmeticProgression) : ℝ :=
  ap.a^3 + (ap.a - ap.d)^3 + (ap.a - 2*ap.d)^3 + (ap.a - 3*ap.d)^3 + (ap.a - 4*ap.d)^3

/-- Calculates the sum of fourth powers of the terms in the arithmetic progression. -/
def ArithmeticProgression.sumOfFourthPowers (ap : ArithmeticProgression) : ℝ :=
  ap.a^4 + (ap.a - ap.d)^4 + (ap.a - 2*ap.d)^4 + (ap.a - 3*ap.d)^4 + (ap.a - 4*ap.d)^4

/-- The main theorem stating the properties of the required arithmetic progression. -/
theorem arithmetic_progression_theorem (ap : ArithmeticProgression) :
  ap.isDecreasing ∧
  ap.sumOfCubes = 0 ∧
  ap.sumOfFourthPowers = 306 →
  ap.a - 4*ap.d = -2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_theorem_l4096_409600


namespace NUMINAMATH_CALUDE_money_difference_l4096_409662

def derek_initial : ℕ := 40
def derek_expense1 : ℕ := 14
def derek_expense2 : ℕ := 11
def derek_expense3 : ℕ := 5
def dave_initial : ℕ := 50
def dave_expense : ℕ := 7

theorem money_difference :
  dave_initial - dave_expense - (derek_initial - derek_expense1 - derek_expense2 - derek_expense3) = 33 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l4096_409662


namespace NUMINAMATH_CALUDE_problem_statement_l4096_409621

theorem problem_statement (x : ℝ) (h : 1 - 5/x + 6/x^3 = 0) : 3/x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4096_409621


namespace NUMINAMATH_CALUDE_circle_C_equation_l4096_409626

-- Define the circles
def circle_C (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = r^2}
def circle_other : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 3*p.1 = 0}

-- Define the line passing through (5, -2)
def common_chord_line (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + 2*p.2 - 5 + r^2 = 0}

-- Theorem statement
theorem circle_C_equation :
  ∃ (r : ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ circle_C r ∩ circle_other → (x, y) ∈ common_chord_line r) ∧
    (5, -2) ∈ common_chord_line r →
    r = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_l4096_409626


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l4096_409688

theorem inequality_system_solution_range (m : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ x : ℤ, (x + 5 > 0 ∧ x - m ≤ 1) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) ↔
  (-3 ≤ m ∧ m < -2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l4096_409688


namespace NUMINAMATH_CALUDE_megan_carrots_count_l4096_409630

/-- Calculates the total number of carrots Megan has after picking, throwing out, and picking again. -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) : ℕ :=
  initial - thrown_out + new_picked

/-- Proves that Megan's total carrots is 61 given the specific numbers in the problem. -/
theorem megan_carrots_count :
  total_carrots 19 4 46 = 61 := by
  sorry

end NUMINAMATH_CALUDE_megan_carrots_count_l4096_409630


namespace NUMINAMATH_CALUDE_payment_for_remaining_worker_l4096_409687

/-- Given a total payment for a job and the fraction of work done by two workers,
    calculate the payment for the remaining worker. -/
theorem payment_for_remaining_worker
  (total_payment : ℚ)
  (work_fraction_two_workers : ℚ)
  (h1 : total_payment = 529)
  (h2 : work_fraction_two_workers = 19 / 23) :
  (1 - work_fraction_two_workers) * total_payment = 92 := by
sorry

end NUMINAMATH_CALUDE_payment_for_remaining_worker_l4096_409687


namespace NUMINAMATH_CALUDE_max_distance_between_sine_cosine_curves_l4096_409647

theorem max_distance_between_sine_cosine_curves : ∃ (C : ℝ),
  (∀ (m : ℝ), |2 * Real.sin m - 2 * Real.sqrt 3 * Real.cos m| ≤ C) ∧
  (∃ (m : ℝ), |2 * Real.sin m - 2 * Real.sqrt 3 * Real.cos m| = C) ∧
  C = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_between_sine_cosine_curves_l4096_409647


namespace NUMINAMATH_CALUDE_cube_surface_area_equals_volume_l4096_409642

theorem cube_surface_area_equals_volume (a : ℝ) (h : a > 0) :
  6 * a^2 = a^3 → a = 6 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_equals_volume_l4096_409642


namespace NUMINAMATH_CALUDE_competition_score_difference_l4096_409627

def score_60_percent : Real := 0.12
def score_85_percent : Real := 0.20
def score_95_percent : Real := 0.38
def score_105_percent : Real := 1 - (score_60_percent + score_85_percent + score_95_percent)

def mean_score : Real :=
  score_60_percent * 60 + score_85_percent * 85 + score_95_percent * 95 + score_105_percent * 105

def median_score : Real := 95

theorem competition_score_difference : median_score - mean_score = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_competition_score_difference_l4096_409627


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l4096_409620

/-- Given two positive integers a and b in ratio 4:5 with LCM 180, prove that a = 36 -/
theorem smaller_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / b = 4 / 5 → 
  Nat.lcm a b = 180 → 
  a = 36 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l4096_409620


namespace NUMINAMATH_CALUDE_raised_beds_planks_l4096_409633

/-- Calculates the number of planks needed for raised beds -/
def planks_needed (num_beds : ℕ) (bed_height : ℕ) (bed_width : ℕ) (bed_length : ℕ) (plank_width : ℕ) : ℕ :=
  let long_sides := 2 * bed_height
  let short_sides := 2 * bed_height
  let planks_per_bed := long_sides + short_sides
  num_beds * planks_per_bed

/-- Proves that 60 planks are needed for 10 raised beds with given dimensions -/
theorem raised_beds_planks :
  planks_needed 10 2 2 8 1 = 60 := by
  sorry

end NUMINAMATH_CALUDE_raised_beds_planks_l4096_409633


namespace NUMINAMATH_CALUDE_cosine_ratio_comparison_l4096_409689

theorem cosine_ratio_comparison : 
  (Real.cos (2016 * π / 180)) / (Real.cos (2017 * π / 180)) < 
  (Real.cos (2018 * π / 180)) / (Real.cos (2019 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_ratio_comparison_l4096_409689


namespace NUMINAMATH_CALUDE_negation_of_p_negation_of_q_l4096_409684

-- Define the statement p
def p : Prop := ∀ x : ℝ, x > 0 → x^2 - 5*x ≥ -25/4

-- Define the statement q
def q : Prop := ∃ n : ℕ, Even n ∧ n % 3 = 0

-- Theorem for the negation of p
theorem negation_of_p : (¬p) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 5*x < -25/4) :=
sorry

-- Theorem for the negation of q
theorem negation_of_q : (¬q) ↔ (∀ n : ℕ, Even n → n % 3 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_p_negation_of_q_l4096_409684


namespace NUMINAMATH_CALUDE_james_waiting_period_l4096_409607

/-- Represents the timeline of James' injury and recovery process -/
structure InjuryTimeline where
  pain_duration : ℕ
  healing_multiplier : ℕ
  additional_wait : ℕ
  total_time : ℕ

/-- Calculates the number of days James waited to start working out after full healing -/
def waiting_period (timeline : InjuryTimeline) : ℕ :=
  timeline.total_time - (timeline.pain_duration * timeline.healing_multiplier) - (timeline.additional_wait * 7)

/-- Theorem stating that James waited 3 days to start working out after full healing -/
theorem james_waiting_period :
  let timeline : InjuryTimeline := {
    pain_duration := 3,
    healing_multiplier := 5,
    additional_wait := 3,
    total_time := 39
  }
  waiting_period timeline = 3 := by sorry

end NUMINAMATH_CALUDE_james_waiting_period_l4096_409607


namespace NUMINAMATH_CALUDE_cube_occupation_percentage_l4096_409679

/-- Represents the dimensions of a rectangular box in inches -/
structure BoxDimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- Represents the side length of a cube in inches -/
def CubeSideLength : ℚ := 3

/-- The dimensions of the given box -/
def givenBox : BoxDimensions := ⟨6, 5, 10⟩

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (box : BoxDimensions) : ℚ :=
  box.length * box.width * box.height

/-- Calculates the largest dimensions that can be filled with cubes -/
def largestFillableDimensions (box : BoxDimensions) (cubeSize : ℚ) : BoxDimensions :=
  ⟨
    (box.length / cubeSize).floor * cubeSize,
    (box.width / cubeSize).floor * cubeSize,
    (box.height / cubeSize).floor * cubeSize
  ⟩

/-- Calculates the percentage of the box volume occupied by cubes -/
def percentageOccupied (box : BoxDimensions) (cubeSize : ℚ) : ℚ :=
  let fillableBox := largestFillableDimensions box cubeSize
  (boxVolume fillableBox) / (boxVolume box) * 100

theorem cube_occupation_percentage :
  percentageOccupied givenBox CubeSideLength = 54 := by
  sorry

end NUMINAMATH_CALUDE_cube_occupation_percentage_l4096_409679


namespace NUMINAMATH_CALUDE_income_calculation_l4096_409691

theorem income_calculation (income expenditure savings : ℕ) : 
  income = 5 * expenditure / 4 →
  income - expenditure = savings →
  savings = 3600 →
  income = 18000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l4096_409691


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l4096_409634

-- Define the curve C
def C (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition for a and b
def condition (a b : ℝ) : Prop := a = 2 ∧ b = Real.sqrt 2

-- Theorem stating that the condition is sufficient but not necessary
theorem condition_sufficient_not_necessary :
  (∀ a b : ℝ, a * b ≠ 0 →
    (condition a b → C a b (Real.sqrt 2) 1)) ∧
  (∃ a b : ℝ, a * b ≠ 0 ∧ C a b (Real.sqrt 2) 1 ∧ ¬ condition a b) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l4096_409634


namespace NUMINAMATH_CALUDE_basic_structures_correct_l4096_409640

/-- The set of basic structures of an algorithm -/
def BasicStructures : Set String :=
  {"Sequential structure", "Conditional structure", "Loop structure"}

/-- The correct answer option -/
def CorrectAnswer : Set String :=
  {"Sequential structure", "Conditional structure", "Loop structure"}

/-- Theorem stating that the basic structures of an algorithm are correctly defined -/
theorem basic_structures_correct : BasicStructures = CorrectAnswer := by
  sorry

end NUMINAMATH_CALUDE_basic_structures_correct_l4096_409640


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l4096_409610

theorem complex_equation_solutions :
  {z : ℂ | z^6 - 6*z^4 + 9*z^2 = 0} = {0, Complex.I * Real.sqrt 3, -Complex.I * Real.sqrt 3} := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l4096_409610


namespace NUMINAMATH_CALUDE_mistaken_division_l4096_409609

theorem mistaken_division (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 21 = 36) :
  D / 12 = 63 := by
sorry

end NUMINAMATH_CALUDE_mistaken_division_l4096_409609


namespace NUMINAMATH_CALUDE_parabola_focus_vertex_distance_l4096_409685

/-- A parabola with vertex V and focus F -/
structure Parabola where
  V : ℝ × ℝ  -- Vertex
  F : ℝ × ℝ  -- Focus

/-- A point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : sorry  -- Condition for the point to be on the parabola

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_focus_vertex_distance 
  (p : Parabola) 
  (A : ParabolaPoint p) 
  (h1 : distance A.point p.F = 18) 
  (h2 : distance A.point p.V = 19) : 
  distance p.F p.V = Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_vertex_distance_l4096_409685


namespace NUMINAMATH_CALUDE_root_of_series_fraction_l4096_409628

theorem root_of_series_fraction (f g : ℕ → ℝ) :
  (∀ k, f k = 8 * k^3) →
  (∀ k, g k = 27 * k^3) →
  (∑' k, f k) / (∑' k, g k) = 8 / 27 →
  ((∑' k, f k) / (∑' k, g k))^(1/3 : ℝ) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_root_of_series_fraction_l4096_409628


namespace NUMINAMATH_CALUDE_evaluate_expression_l4096_409697

theorem evaluate_expression : (122^2 - 115^2 + 7) / 14 = 119 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4096_409697


namespace NUMINAMATH_CALUDE_max_attendance_l4096_409625

-- Define the days of the week
inductive Day
| monday
| tuesday
| wednesday
| thursday
| friday

-- Define the team members
inductive Member
| alice
| bob
| charlie
| diana
| edward

-- Define a function that returns whether a member is available on a given day
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.alice, Day.monday => false
  | Member.alice, Day.thursday => false
  | Member.bob, Day.tuesday => false
  | Member.bob, Day.friday => false
  | Member.charlie, Day.monday => false
  | Member.charlie, Day.tuesday => false
  | Member.charlie, Day.thursday => false
  | Member.charlie, Day.friday => false
  | Member.diana, Day.wednesday => false
  | Member.diana, Day.thursday => false
  | Member.edward, Day.wednesday => false
  | _, _ => true

-- Define a function that counts the number of available members on a given day
def countAvailable (d : Day) : Nat :=
  List.length (List.filter (λ m => isAvailable m d) [Member.alice, Member.bob, Member.charlie, Member.diana, Member.edward])

-- State the theorem
theorem max_attendance :
  (∀ d : Day, countAvailable d ≤ 3) ∧
  (countAvailable Day.monday = 3) ∧
  (countAvailable Day.wednesday = 3) ∧
  (countAvailable Day.friday = 3) :=
sorry

end NUMINAMATH_CALUDE_max_attendance_l4096_409625


namespace NUMINAMATH_CALUDE_exam_score_difference_l4096_409605

def math_exam_proof (bryan_score jen_score sammy_score total_points sammy_mistakes : ℕ) : Prop :=
  bryan_score = 20 ∧
  jen_score > bryan_score ∧
  sammy_score = jen_score - 2 ∧
  total_points = 35 ∧
  sammy_mistakes = 7 ∧
  sammy_score = total_points - sammy_mistakes ∧
  jen_score - bryan_score = 10

theorem exam_score_difference :
  ∀ (bryan_score jen_score sammy_score total_points sammy_mistakes : ℕ),
    math_exam_proof bryan_score jen_score sammy_score total_points sammy_mistakes :=
by
  sorry

end NUMINAMATH_CALUDE_exam_score_difference_l4096_409605


namespace NUMINAMATH_CALUDE_recommendation_plans_count_l4096_409601

/-- Represents the number of recommendation spots for each language -/
structure RecommendationSpots :=
  (russian : Nat)
  (japanese : Nat)
  (spanish : Nat)

/-- Represents the gender distribution of selected candidates -/
structure SelectedCandidates :=
  (males : Nat)
  (females : Nat)

/-- Calculates the number of different recommendation plans -/
def countRecommendationPlans (spots : RecommendationSpots) (candidates : SelectedCandidates) : Nat :=
  sorry

theorem recommendation_plans_count :
  let spots : RecommendationSpots := ⟨2, 2, 1⟩
  let candidates : SelectedCandidates := ⟨3, 2⟩
  countRecommendationPlans spots candidates = 24 := by sorry

end NUMINAMATH_CALUDE_recommendation_plans_count_l4096_409601


namespace NUMINAMATH_CALUDE_water_bottles_remaining_l4096_409604

/-- Calculates the number of bottles remaining after two days of consumption --/
def bottlesRemaining (initialBottles : ℕ) : ℕ :=
  let firstDayRemaining := initialBottles - 
    (initialBottles / 4 + initialBottles / 6 + initialBottles / 8)
  let fatherSecondDay := firstDayRemaining / 5
  let motherSecondDay := (firstDayRemaining - fatherSecondDay) / 7
  let sonSecondDay := (firstDayRemaining - fatherSecondDay - motherSecondDay) / 9
  let daughterSecondDay := (firstDayRemaining - fatherSecondDay - motherSecondDay - sonSecondDay) / 9
  firstDayRemaining - (fatherSecondDay + motherSecondDay + sonSecondDay + daughterSecondDay)

theorem water_bottles_remaining (initialBottles : ℕ) :
  initialBottles = 48 → bottlesRemaining initialBottles = 14 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_remaining_l4096_409604


namespace NUMINAMATH_CALUDE_set_equalities_l4096_409673

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 < x ∧ x < 1}
def B : Set ℝ := {x | x ≤ -1}
def C : Set ℝ := {x | -2 < x ∧ x ≤ 2}

-- Theorem stating the equalities
theorem set_equalities :
  (A = A ∩ (B ∪ C)) ∧
  (A = A ∪ (B ∩ C)) ∧
  (A = (A ∩ B) ∪ (A ∩ C)) := by
  sorry

end NUMINAMATH_CALUDE_set_equalities_l4096_409673


namespace NUMINAMATH_CALUDE_log_3897_between_consecutive_integers_l4096_409681

theorem log_3897_between_consecutive_integers : 
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 3897 / Real.log 10 ∧ Real.log 3897 / Real.log 10 < b ∧ a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_3897_between_consecutive_integers_l4096_409681


namespace NUMINAMATH_CALUDE_min_value_of_squared_differences_l4096_409608

theorem min_value_of_squared_differences (a b c : ℝ) :
  ∃ (min : ℝ), min = ((a - b)^2 + (b - c)^2 + (a - c)^2) / 3 ∧
  ∀ (x : ℝ), (x - a)^2 + (x - b)^2 + (x - c)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_squared_differences_l4096_409608


namespace NUMINAMATH_CALUDE_quadratic_factorization_l4096_409638

theorem quadratic_factorization (x : ℝ) : 
  x^2 - 5*x + 3 = (x + 5/2 + Real.sqrt 13/2) * (x + 5/2 - Real.sqrt 13/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l4096_409638


namespace NUMINAMATH_CALUDE_parallelogram_base_given_triangle_l4096_409678

/-- Given a triangle and a parallelogram with equal areas and the same height,
    if the base of the triangle is 24 inches, then the base of the parallelogram is 12 inches. -/
theorem parallelogram_base_given_triangle (h : ℝ) (b_p : ℝ) : 
  (1/2 * 24 * h = b_p * h) → b_p = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_given_triangle_l4096_409678


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l4096_409676

theorem probability_at_least_one_correct (n m : ℕ) (h : n > 0 ∧ m > 0) :
  let p := 1 - (1 - 1 / n) ^ m
  n = 6 ∧ m = 6 → p = 31031 / 46656 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l4096_409676


namespace NUMINAMATH_CALUDE_product_of_differences_equals_seven_l4096_409637

theorem product_of_differences_equals_seven
  (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ)
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2016)
  (h₂ : y₁^3 - 3*x₁^2*y₁ = 2000)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2016)
  (h₄ : y₂^3 - 3*x₂^2*y₂ = 2000)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2016)
  (h₆ : y₃^3 - 3*x₃^2*y₃ = 2000)
  (h₇ : y₁ ≠ 0)
  (h₈ : y₂ ≠ 0)
  (h₉ : y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 7 := by
sorry

end NUMINAMATH_CALUDE_product_of_differences_equals_seven_l4096_409637


namespace NUMINAMATH_CALUDE_min_steps_equal_iff_path_without_leaves_l4096_409624

/-- Represents a tree of players and ropes -/
structure PlayerTree where
  players : ℕ
  ropes : ℕ
  is_tree : ropes = players - 1

/-- Minimum steps to form a path in unrestricted scenario -/
def min_steps_unrestricted (t : PlayerTree) : ℕ := sorry

/-- Minimum steps to form a path in neighbor-only scenario -/
def min_steps_neighbor_only (t : PlayerTree) : ℕ := sorry

/-- Checks if the tree without leaves is a path -/
def is_path_without_leaves (t : PlayerTree) : Prop := sorry

/-- Main theorem: equality of minimum steps iff tree without leaves is a path -/
theorem min_steps_equal_iff_path_without_leaves (t : PlayerTree) :
  min_steps_unrestricted t = min_steps_neighbor_only t ↔ is_path_without_leaves t := by
  sorry

end NUMINAMATH_CALUDE_min_steps_equal_iff_path_without_leaves_l4096_409624


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_400_l4096_409651

theorem largest_multiple_of_15_under_400 : ∃ (n : ℕ), n * 15 = 390 ∧ 
  390 < 400 ∧ 
  ∀ (m : ℕ), m * 15 < 400 → m * 15 ≤ 390 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_400_l4096_409651


namespace NUMINAMATH_CALUDE_probability_at_least_one_of_three_l4096_409617

theorem probability_at_least_one_of_three (p : ℝ) (h : p = 1 / 3) :
  1 - (1 - p)^3 = 19 / 27 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_of_three_l4096_409617


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l4096_409672

theorem bridge_length_calculation (train_length : ℝ) (signal_post_time : ℝ) (bridge_cross_time : ℝ) :
  train_length = 600 →
  signal_post_time = 40 →
  bridge_cross_time = 360 →
  let train_speed := train_length / signal_post_time
  let bridge_only_time := bridge_cross_time - signal_post_time
  let bridge_length := train_speed * bridge_only_time
  bridge_length = 4800 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l4096_409672


namespace NUMINAMATH_CALUDE_expression_classification_l4096_409669

/-- Represents an algebraic expression -/
inductive AlgebraicExpr
  | Constant (n : ℚ)
  | Variable (name : String)
  | Product (coef : ℚ) (terms : List (String × ℕ))

/-- Calculates the degree of an algebraic expression -/
def degree (expr : AlgebraicExpr) : ℕ :=
  match expr with
  | AlgebraicExpr.Constant _ => 0
  | AlgebraicExpr.Variable _ => 1
  | AlgebraicExpr.Product _ terms => terms.foldl (fun acc (_, power) => acc + power) 0

/-- Checks if an expression contains variables -/
def hasVariables (expr : AlgebraicExpr) : Bool :=
  match expr with
  | AlgebraicExpr.Constant _ => false
  | AlgebraicExpr.Variable _ => true
  | AlgebraicExpr.Product _ terms => terms.length > 0

def expressions : List AlgebraicExpr := [
  AlgebraicExpr.Product (-2) [("a", 1)],
  AlgebraicExpr.Product 3 [("a", 1), ("b", 2)],
  AlgebraicExpr.Constant (2/3),
  AlgebraicExpr.Product 3 [("a", 2), ("b", 1)],
  AlgebraicExpr.Product (-3) [("a", 3)],
  AlgebraicExpr.Constant 25,
  AlgebraicExpr.Product (-(3/4)) [("b", 1)]
]

theorem expression_classification :
  (expressions.filter hasVariables).length = 5 ∧
  (expressions.filter (fun e => ¬hasVariables e)).length = 2 ∧
  (expressions.filter (fun e => degree e = 0)).length = 2 ∧
  (expressions.filter (fun e => degree e = 1)).length = 2 ∧
  (expressions.filter (fun e => degree e = 3)).length = 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_classification_l4096_409669


namespace NUMINAMATH_CALUDE_chicken_admission_combinations_l4096_409655

theorem chicken_admission_combinations : Nat.choose 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_chicken_admission_combinations_l4096_409655


namespace NUMINAMATH_CALUDE_christmas_sale_pricing_l4096_409650

/-- Represents the discount rate as a fraction -/
def discount_rate : ℚ := 2/5

/-- Calculates the sale price given the original price and discount rate -/
def sale_price (original_price : ℚ) : ℚ := original_price * (1 - discount_rate)

/-- Calculates the original price given the sale price and discount rate -/
def original_price (sale_price : ℚ) : ℚ := sale_price / (1 - discount_rate)

theorem christmas_sale_pricing (a b : ℚ) :
  sale_price a = 3/5 * a ∧ original_price b = 5/3 * b := by
  sorry

end NUMINAMATH_CALUDE_christmas_sale_pricing_l4096_409650


namespace NUMINAMATH_CALUDE_correct_product_is_5810_l4096_409674

/-- Reverses the digits of a three-digit number -/
def reverse_digits (n : Nat) : Nat :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- Checks if a number is three-digit -/
def is_three_digit (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem correct_product_is_5810 (a b : Nat) :
  a > 0 ∧ b > 0 ∧ is_three_digit a ∧ (reverse_digits a - 3) * b = 245 →
  a * b = 5810 := by
  sorry

end NUMINAMATH_CALUDE_correct_product_is_5810_l4096_409674


namespace NUMINAMATH_CALUDE_urn_problem_l4096_409693

theorem urn_problem (total : ℕ) (red_percent : ℚ) (new_red_percent : ℚ) 
  (h1 : total = 120)
  (h2 : red_percent = 2/5)
  (h3 : new_red_percent = 4/5) :
  ∃ (removed : ℕ), 
    (red_percent * total : ℚ) / (total - removed : ℚ) = new_red_percent ∧ 
    removed = 60 := by
  sorry

end NUMINAMATH_CALUDE_urn_problem_l4096_409693


namespace NUMINAMATH_CALUDE_product_of_real_parts_complex_equation_l4096_409643

theorem product_of_real_parts_complex_equation : 
  let f : ℂ → ℂ := fun x ↦ x^2 - 4*x + 2 - 2*I
  let solutions := {x : ℂ | f x = 0}
  ∃ x₁ x₂ : ℂ, x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
  (x₁.re * x₂.re = 3 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_parts_complex_equation_l4096_409643


namespace NUMINAMATH_CALUDE_prop_1_prop_3_l4096_409602

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define non-coinciding lines and planes
variable (a b : Line)
variable (α β : Plane)
variable (h_lines_distinct : a ≠ b)
variable (h_planes_distinct : α ≠ β)

-- Proposition 1
theorem prop_1 : 
  parallel a b → perpendicular_line_plane a α → perpendicular_line_plane b α :=
sorry

-- Proposition 3
theorem prop_3 : 
  perpendicular_line_plane a α → perpendicular_line_plane a β → parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_prop_1_prop_3_l4096_409602


namespace NUMINAMATH_CALUDE_mirror_to_wall_area_ratio_l4096_409666

/-- The ratio of a square mirror's area to a rectangular wall's area --/
theorem mirror_to_wall_area_ratio
  (mirror_side : ℝ)
  (wall_width : ℝ)
  (wall_length : ℝ)
  (h1 : mirror_side = 24)
  (h2 : wall_width = 42)
  (h3 : wall_length = 27.428571428571427)
  : (mirror_side ^ 2) / (wall_width * wall_length) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_mirror_to_wall_area_ratio_l4096_409666


namespace NUMINAMATH_CALUDE_group_size_calculation_l4096_409606

theorem group_size_calculation (n : ℕ) : 
  (n : ℝ) * 14 = n * 14 →                   -- Initial average age
  ((n : ℝ) * 14 + 32) / (n + 1) = 16 →      -- New average age
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_group_size_calculation_l4096_409606


namespace NUMINAMATH_CALUDE_cards_distribution_l4096_409670

/-- Given a total number of cards and people, calculates how many people receive fewer than the ceiling of the average number of cards. -/
def people_with_fewer_cards (total_cards : ℕ) (total_people : ℕ) : ℕ :=
  let avg_cards := total_cards / total_people
  let remainder := total_cards % total_people
  let max_cards := avg_cards + 1
  total_people - remainder

theorem cards_distribution (total_cards : ℕ) (total_people : ℕ) 
  (h1 : total_cards = 60) (h2 : total_people = 9) :
  people_with_fewer_cards total_cards total_people = 3 := by
  sorry

#eval people_with_fewer_cards 60 9

end NUMINAMATH_CALUDE_cards_distribution_l4096_409670


namespace NUMINAMATH_CALUDE_brent_nerds_count_l4096_409677

/-- Represents the candy inventory of Brent --/
structure CandyInventory where
  kitKat : ℕ
  hersheyKisses : ℕ
  nerds : ℕ
  lollipops : ℕ
  babyRuths : ℕ
  reeseCups : ℕ

/-- Calculates the total number of candy pieces --/
def totalCandy (inventory : CandyInventory) : ℕ :=
  inventory.kitKat + inventory.hersheyKisses + inventory.nerds + 
  inventory.lollipops + inventory.babyRuths + inventory.reeseCups

/-- Theorem stating that Brent received 8 boxes of Nerds --/
theorem brent_nerds_count : ∃ (inventory : CandyInventory),
  inventory.kitKat = 5 ∧
  inventory.hersheyKisses = 3 * inventory.kitKat ∧
  inventory.lollipops = 11 ∧
  inventory.babyRuths = 10 ∧
  inventory.reeseCups = inventory.babyRuths / 2 ∧
  totalCandy inventory - 5 = 49 ∧
  inventory.nerds = 8 := by
  sorry

end NUMINAMATH_CALUDE_brent_nerds_count_l4096_409677


namespace NUMINAMATH_CALUDE_x_percent_of_x_squared_is_nine_l4096_409661

theorem x_percent_of_x_squared_is_nine (x : ℝ) (h1 : x > 0) (h2 : x / 100 * x^2 = 9) : x = 10 * Real.rpow 3 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_x_percent_of_x_squared_is_nine_l4096_409661


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l4096_409611

-- Define a function to convert a number from base b to base 10
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

-- Define the number in base 7
def num_base_7 : List Nat := [1, 4, 3, 2, 5]

-- Define the number in base 8
def num_base_8 : List Nat := [1, 2, 3, 4]

-- Theorem statement
theorem base_conversion_subtraction :
  to_base_10 num_base_7 7 - to_base_10 num_base_8 8 = 10610 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l4096_409611


namespace NUMINAMATH_CALUDE_equal_pairs_l4096_409644

theorem equal_pairs : 
  (-2^4 ≠ (-2)^4) ∧ 
  (5^3 ≠ 3^5) ∧ 
  (-(-3) ≠ -|(-3)|) ∧ 
  ((-1)^2 = (-1)^2008) := by
  sorry

end NUMINAMATH_CALUDE_equal_pairs_l4096_409644


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l4096_409648

theorem algebraic_expression_value (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = -1) :
  (2 * a + 3 * b - 2 * a * b) - (a + 4 * b + a * b) - (3 * a * b + 2 * b - 2 * a) = 21 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l4096_409648


namespace NUMINAMATH_CALUDE_three_elements_satisfy_l4096_409614

/-- The set M containing elements A, A₁, A₂, A₃, A₄, A₅ -/
inductive M
  | A
  | A1
  | A2
  | A3
  | A4
  | A5

/-- The operation ⊗ defined on M -/
def otimes : M → M → M
  | M.A, M.A => M.A
  | M.A, M.A1 => M.A1
  | M.A, M.A2 => M.A2
  | M.A, M.A3 => M.A3
  | M.A, M.A4 => M.A4
  | M.A, M.A5 => M.A1
  | M.A1, M.A => M.A1
  | M.A1, M.A1 => M.A2
  | M.A1, M.A2 => M.A3
  | M.A1, M.A3 => M.A4
  | M.A1, M.A4 => M.A1
  | M.A1, M.A5 => M.A2
  | M.A2, M.A => M.A2
  | M.A2, M.A1 => M.A3
  | M.A2, M.A2 => M.A4
  | M.A2, M.A3 => M.A1
  | M.A2, M.A4 => M.A2
  | M.A2, M.A5 => M.A3
  | M.A3, M.A => M.A3
  | M.A3, M.A1 => M.A4
  | M.A3, M.A2 => M.A1
  | M.A3, M.A3 => M.A2
  | M.A3, M.A4 => M.A3
  | M.A3, M.A5 => M.A4
  | M.A4, M.A => M.A4
  | M.A4, M.A1 => M.A1
  | M.A4, M.A2 => M.A2
  | M.A4, M.A3 => M.A3
  | M.A4, M.A4 => M.A4
  | M.A4, M.A5 => M.A1
  | M.A5, M.A => M.A1
  | M.A5, M.A1 => M.A2
  | M.A5, M.A2 => M.A3
  | M.A5, M.A3 => M.A4
  | M.A5, M.A4 => M.A1
  | M.A5, M.A5 => M.A2

/-- The theorem stating that exactly 3 elements in M satisfy (a ⊗ a) ⊗ A₂ = A -/
theorem three_elements_satisfy :
  (∃! (s : Finset M), s.card = 3 ∧ ∀ a ∈ s, otimes (otimes a a) M.A2 = M.A) :=
sorry

end NUMINAMATH_CALUDE_three_elements_satisfy_l4096_409614


namespace NUMINAMATH_CALUDE_pair_probability_after_removal_l4096_409652

def initial_deck_size : ℕ := 52
def cards_per_value : ℕ := 4
def values_count : ℕ := 13
def removed_pairs : ℕ := 2

def remaining_deck_size : ℕ := initial_deck_size - removed_pairs * cards_per_value / 2

def total_ways_to_select_two : ℕ := remaining_deck_size.choose 2

def full_value_count : ℕ := values_count - removed_pairs
def pair_forming_ways_full : ℕ := full_value_count * (cards_per_value.choose 2)
def pair_forming_ways_reduced : ℕ := removed_pairs * ((cards_per_value - 2).choose 2)
def total_pair_forming_ways : ℕ := pair_forming_ways_full + pair_forming_ways_reduced

theorem pair_probability_after_removal : 
  (total_pair_forming_ways : ℚ) / total_ways_to_select_two = 17 / 282 := by sorry

end NUMINAMATH_CALUDE_pair_probability_after_removal_l4096_409652


namespace NUMINAMATH_CALUDE_D_144_l4096_409680

/-- 
D(n) represents the number of ways to write a positive integer n as a product of 
integers greater than 1, where the order of factors matters.
-/
def D (n : ℕ+) : ℕ := sorry

/-- The main theorem stating that D(144) = 41 -/
theorem D_144 : D 144 = 41 := by sorry

end NUMINAMATH_CALUDE_D_144_l4096_409680


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l4096_409665

theorem triangle_angle_measure (a b c : ℝ) (A : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  S > 0 →
  (4 * Real.sqrt 3 / 3) * S = b^2 + c^2 - a^2 →
  S = (1/2) * b * c * Real.sin A →
  0 < A → A < π →
  A = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l4096_409665


namespace NUMINAMATH_CALUDE_percentage_increase_l4096_409658

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 110 → final = 165 → (final - initial) / initial * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l4096_409658


namespace NUMINAMATH_CALUDE_impossible_to_swap_folds_l4096_409613

/-- Represents the number of folds on one side of a rhinoceros -/
structure Folds :=
  (vertical : ℕ)
  (horizontal : ℕ)

/-- Represents the state of folds on both sides of a rhinoceros -/
structure RhinoState :=
  (left : Folds)
  (right : Folds)

/-- A scratch operation that can be performed on the rhinoceros -/
inductive ScratchOp
  | left_vertical
  | left_horizontal
  | right_vertical
  | right_horizontal

/-- Defines a valid initial state for the rhinoceros -/
def valid_initial_state (s : RhinoState) : Prop :=
  s.left.vertical + s.left.horizontal + s.right.vertical + s.right.horizontal = 17

/-- Defines the result of applying a scratch operation to a state -/
def apply_scratch (s : RhinoState) (op : ScratchOp) : RhinoState :=
  sorry

/-- Defines when a state is reachable from the initial state through scratching -/
def reachable (initial : RhinoState) (final : RhinoState) : Prop :=
  sorry

/-- The main theorem stating that it's impossible to swap vertical and horizontal folds -/
theorem impossible_to_swap_folds (initial : RhinoState) :
  valid_initial_state initial →
  ¬∃ (final : RhinoState),
    reachable initial final ∧
    final.left.vertical = initial.left.horizontal ∧
    final.left.horizontal = initial.left.vertical ∧
    final.right.vertical = initial.right.horizontal ∧
    final.right.horizontal = initial.right.vertical :=
  sorry

end NUMINAMATH_CALUDE_impossible_to_swap_folds_l4096_409613


namespace NUMINAMATH_CALUDE_circle_diameter_property_l4096_409618

theorem circle_diameter_property (BC BD DA : ℝ) (h1 : BC = Real.sqrt 901) (h2 : BD = 1) (h3 : DA = 16) : ∃ EC : ℝ, EC = 1 ∧ BC * EC = BD * (BC - BD) := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_property_l4096_409618


namespace NUMINAMATH_CALUDE_weight_loss_calculation_l4096_409646

/-- Represents the weight loss calculation problem --/
theorem weight_loss_calculation 
  (current_weight : ℕ) 
  (previous_weight : ℕ) 
  (h1 : current_weight = 27) 
  (h2 : previous_weight = 128) :
  previous_weight - current_weight = 101 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_calculation_l4096_409646


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l4096_409690

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l4096_409690


namespace NUMINAMATH_CALUDE_least_sum_of_four_primes_l4096_409683

theorem least_sum_of_four_primes (n : ℕ) : 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), 
    p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧ p₄.Prime ∧
    p₁ > 10 ∧ p₂ > 10 ∧ p₃ > 10 ∧ p₄ > 10 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n = p₁ + p₂ + p₃ + p₄) →
  n ≥ 60 :=
by
  sorry

#check least_sum_of_four_primes

end NUMINAMATH_CALUDE_least_sum_of_four_primes_l4096_409683


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l4096_409639

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = 0.7333333333333333 ∧ x = 11 / 15 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l4096_409639


namespace NUMINAMATH_CALUDE_linear_equation_property_l4096_409616

theorem linear_equation_property (x y : ℝ) (h : x + 6 * y = 17) :
  7 * x + 42 * y = 119 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_property_l4096_409616


namespace NUMINAMATH_CALUDE_inverse_cube_root_relation_l4096_409641

/-- Given that z varies inversely as ∛w, prove that w = 1 when z = 6, 
    given that z = 3 when w = 8. -/
theorem inverse_cube_root_relation (z w : ℝ) (k : ℝ) 
  (h1 : ∀ w z, z * (w ^ (1/3 : ℝ)) = k)
  (h2 : 3 * (8 ^ (1/3 : ℝ)) = k)
  (h3 : 6 * (w ^ (1/3 : ℝ)) = k) : 
  w = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_cube_root_relation_l4096_409641


namespace NUMINAMATH_CALUDE_hockey_team_boys_percentage_l4096_409695

theorem hockey_team_boys_percentage
  (total_players : ℕ)
  (junior_girls : ℕ)
  (h1 : total_players = 50)
  (h2 : junior_girls = 10)
  (h3 : junior_girls = total_players - junior_girls - (total_players - 2 * junior_girls)) :
  (total_players - 2 * junior_girls : ℚ) / total_players = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_hockey_team_boys_percentage_l4096_409695


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l4096_409663

/-- A geometric sequence with given third and tenth terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) ∧ 
  a 3 = 3 ∧ 
  a 10 = 384

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := 3 * 2^(n - 3)

/-- Theorem stating that the general term is correct for the given geometric sequence -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  geometric_sequence a → (∀ n : ℕ, a n = general_term n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l4096_409663


namespace NUMINAMATH_CALUDE_speed_against_stream_calculation_mans_speed_is_six_l4096_409645

/-- Calculate the speed against the stream given the rate in still water and speed with the stream -/
def speed_against_stream (rate_still : ℝ) (speed_with : ℝ) : ℝ :=
  |rate_still - 2 * (speed_with - rate_still)|

/-- Theorem: Given a man's rowing rate in still water and his speed with the stream,
    his speed against the stream is the absolute difference between his rate in still water
    and twice the difference of his speed with the stream and his rate in still water -/
theorem speed_against_stream_calculation (rate_still : ℝ) (speed_with : ℝ) :
  speed_against_stream rate_still speed_with = 
  |rate_still - 2 * (speed_with - rate_still)| := by
  sorry

/-- The man's speed against the stream given his rate in still water and speed with the stream -/
def mans_speed_against_stream : ℝ :=
  speed_against_stream 5 16

/-- Theorem: The man's speed against the stream is 6 km/h -/
theorem mans_speed_is_six :
  mans_speed_against_stream = 6 := by
  sorry

end NUMINAMATH_CALUDE_speed_against_stream_calculation_mans_speed_is_six_l4096_409645


namespace NUMINAMATH_CALUDE_invalid_vote_percentage_l4096_409667

/-- Proves that the percentage of invalid votes is 15% given the specified conditions --/
theorem invalid_vote_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_valid_votes : ℕ)
  (h_total : total_votes = 560000)
  (h_percentage : candidate_a_percentage = 75 / 100)
  (h_valid_votes : candidate_a_valid_votes = 357000) :
  (total_votes - (candidate_a_valid_votes / candidate_a_percentage : ℚ)) / total_votes = 15 / 100 :=
sorry

end NUMINAMATH_CALUDE_invalid_vote_percentage_l4096_409667


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l4096_409619

/-- In a class of students, where half of the number of girls equals one-third of the total number of students, 
    the ratio of boys to girls is 1:2. -/
theorem boys_to_girls_ratio (S : ℕ) (B G : ℕ) : 
  S > 0 → 
  S = B + G → 
  (G : ℚ) / 2 = (S : ℚ) / 3 → 
  (B : ℚ) / G = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l4096_409619
