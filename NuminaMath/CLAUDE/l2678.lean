import Mathlib

namespace NUMINAMATH_CALUDE_group_division_arrangements_l2678_267847

/-- The number of teachers --/
def num_teachers : ℕ := 2

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of groups --/
def num_groups : ℕ := 2

/-- The number of teachers per group --/
def teachers_per_group : ℕ := 1

/-- The number of students per group --/
def students_per_group : ℕ := 2

/-- The total number of arrangements --/
def total_arrangements : ℕ := 12

theorem group_division_arrangements :
  (Nat.choose num_teachers teachers_per_group) *
  (Nat.choose num_students students_per_group) =
  total_arrangements :=
sorry

end NUMINAMATH_CALUDE_group_division_arrangements_l2678_267847


namespace NUMINAMATH_CALUDE_reporters_coverage_l2678_267826

theorem reporters_coverage (total : ℕ) (h_total : total > 0) : 
  let local_politics := (28 : ℕ) * total / 100
  let not_politics := (60 : ℕ) * total / 100
  let politics := total - not_politics
  (politics - local_politics) * 100 / politics = 30 :=
by sorry

end NUMINAMATH_CALUDE_reporters_coverage_l2678_267826


namespace NUMINAMATH_CALUDE_quadratic_relationship_l2678_267818

theorem quadratic_relationship (x : ℕ) (z : ℕ) : 
  (x = 1 ∧ z = 5) ∨ 
  (x = 2 ∧ z = 12) ∨ 
  (x = 3 ∧ z = 23) ∨ 
  (x = 4 ∧ z = 38) ∨ 
  (x = 5 ∧ z = 57) → 
  z = 2 * x^2 + x + 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_relationship_l2678_267818


namespace NUMINAMATH_CALUDE_maintenance_check_increase_maintenance_check_theorem_l2678_267852

theorem maintenance_check_increase (initial_interval : ℝ) 
  (additive_a_percent : ℝ) (additive_b_percent : ℝ) : ℝ :=
  let interval_after_a := initial_interval * (1 + additive_a_percent)
  let interval_after_b := interval_after_a * (1 + additive_b_percent)
  let total_increase_percent := (interval_after_b - initial_interval) / initial_interval * 100
  total_increase_percent

theorem maintenance_check_theorem :
  maintenance_check_increase 45 0.35 0.20 = 62 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_maintenance_check_theorem_l2678_267852


namespace NUMINAMATH_CALUDE_secret_spreading_day_l2678_267863

/-- The number of students who know the secret on the nth day -/
def students_knowing_secret (n : ℕ) : ℕ := 3^(n+1) - 1

/-- The day when 3280 students know the secret -/
theorem secret_spreading_day : ∃ (n : ℕ), students_knowing_secret n = 3280 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_secret_spreading_day_l2678_267863


namespace NUMINAMATH_CALUDE_otimes_equation_solution_l2678_267821

-- Define the custom operation
noncomputable def otimes (a b : ℝ) : ℝ :=
  a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem otimes_equation_solution :
  ∃ z : ℝ, otimes 3 z = 27 ∧ z = 72 :=
by sorry

end NUMINAMATH_CALUDE_otimes_equation_solution_l2678_267821


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l2678_267806

theorem binomial_coefficient_congruence (p a b : ℕ) : 
  Nat.Prime p → p > 3 → a > b → b > 1 → 
  (Nat.choose (a * p) (b * p)) ≡ (Nat.choose a b) [MOD p^3] := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l2678_267806


namespace NUMINAMATH_CALUDE_complex_fraction_ratio_l2678_267860

theorem complex_fraction_ratio : 
  let z : ℂ := (2 + Complex.I) / Complex.I
  let a : ℝ := z.re
  let b : ℝ := z.im
  b / a = -2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_ratio_l2678_267860


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_three_l2678_267870

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Represents a point on the right branch of a hyperbola -/
structure RightBranchPoint (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1
  on_right_branch : x > 0

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The left focus of a hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Predicate to check if three points form an equilateral triangle -/
def is_equilateral_triangle (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

/-- Main theorem: If a line through the right focus intersects the right branch
    at two points forming an equilateral triangle with the left focus,
    then the eccentricity of the hyperbola is √3 -/
theorem hyperbola_eccentricity_sqrt_three (h : Hyperbola a b)
  (M N : RightBranchPoint h)
  (h_line : ∃ (t : ℝ), (M.x, M.y) = right_focus h + t • ((N.x, N.y) - right_focus h))
  (h_equilateral : is_equilateral_triangle (M.x, M.y) (N.x, N.y) (left_focus h)) :
  eccentricity h = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_three_l2678_267870


namespace NUMINAMATH_CALUDE_range_of_negative_values_l2678_267824

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is decreasing on (-∞, 0] if f(x) ≥ f(y) for all x, y ∈ (-∞, 0] with x ≤ y -/
def IsDecreasingOnNegative (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → x ≤ 0 → y ≤ 0 → f x ≥ f y

/-- The theorem stating the range of x for which f(x) < 0 -/
theorem range_of_negative_values (f : ℝ → ℝ) 
  (h_even : IsEven f) 
  (h_decreasing : IsDecreasingOnNegative f) 
  (h_zero : f 2 = 0) : 
  {x : ℝ | f x < 0} = Set.Ioo (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_negative_values_l2678_267824


namespace NUMINAMATH_CALUDE_min_balls_for_twenty_of_one_color_l2678_267810

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minBallsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The specific ball counts in our problem -/
def ourBox : BallCounts :=
  { red := 30, green := 22, yellow := 18, blue := 15, white := 10, black := 6 }

/-- The theorem to be proved -/
theorem min_balls_for_twenty_of_one_color :
  minBallsForColor ourBox 20 = 88 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_for_twenty_of_one_color_l2678_267810


namespace NUMINAMATH_CALUDE_triangle_area_l2678_267828

/-- The area of a triangle with base 2t and height 3t - 1 is t(3t - 1) -/
theorem triangle_area (t : ℝ) : 
  let base : ℝ := 2 * t
  let height : ℝ := 3 * t - 1
  (1 / 2 : ℝ) * base * height = t * (3 * t - 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2678_267828


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2678_267813

/-- A parabola is defined by its coefficients a, b, and c in the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a parabola -/
def lies_on (p : Point) (par : Parabola) : Prop :=
  p.y = par.a * p.x^2 + par.b * p.x + par.c

/-- The axis of symmetry of a parabola -/
def axis_of_symmetry (par : Parabola) : ℝ := 3

/-- Theorem: The axis of symmetry of a parabola y = ax^2 + bx + c is x = 3, 
    given that the points (2,5) and (4,5) lie on the parabola -/
theorem parabola_axis_of_symmetry (par : Parabola) 
  (h1 : lies_on ⟨2, 5⟩ par) 
  (h2 : lies_on ⟨4, 5⟩ par) : 
  axis_of_symmetry par = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2678_267813


namespace NUMINAMATH_CALUDE_arithmetic_properties_l2678_267869

variable (a : ℤ)

theorem arithmetic_properties :
  (216 + 35 + 84 = 35 + (216 + 84)) ∧
  (298 - 35 - 165 = 298 - (35 + 165)) ∧
  (400 / 25 / 4 = 400 / (25 * 4)) ∧
  (a * 6 + 6 * 15 = 6 * (a + 15)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_properties_l2678_267869


namespace NUMINAMATH_CALUDE_distribute_five_to_three_l2678_267874

/-- The number of ways to distribute n students among k universities,
    with each university receiving at least one student. -/
def distribute_students (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to partition n elements into k non-empty subsets. -/
def stirling2 (n k : ℕ) : ℕ :=
  sorry

theorem distribute_five_to_three :
  distribute_students 5 3 = 150 :=
sorry

end NUMINAMATH_CALUDE_distribute_five_to_three_l2678_267874


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l2678_267834

/-- Given a cubic equation x^3 - ax^2 + bx - c = 0 with roots r, s, and t,
    prove that r^2 + s^2 + t^2 = a^2 - 2b -/
theorem cubic_root_sum_squares (a b c r s t : ℝ) : 
  (r^3 - a*r^2 + b*r - c = 0) → 
  (s^3 - a*s^2 + b*s - c = 0) → 
  (t^3 - a*t^2 + b*t - c = 0) → 
  r^2 + s^2 + t^2 = a^2 - 2*b := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l2678_267834


namespace NUMINAMATH_CALUDE_function_is_identity_l2678_267861

open Real

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x^2 + y * f z + 1) = x * f x + z * f y + 1

theorem function_is_identity (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (∀ x : ℝ, f x = x) ∧ f 5 = 5 := by
  sorry


end NUMINAMATH_CALUDE_function_is_identity_l2678_267861


namespace NUMINAMATH_CALUDE_fraction_equality_l2678_267872

theorem fraction_equality : 
  (3 + 6 - 12 + 24 + 48 - 96 + 192 - 384) / (6 + 12 - 24 + 48 + 96 - 192 + 384 - 768) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2678_267872


namespace NUMINAMATH_CALUDE_coat_shirt_ratio_l2678_267894

theorem coat_shirt_ratio (pants shirt coat : ℕ) : 
  pants + shirt = 100 →
  pants + coat = 244 →
  coat = 180 →
  ∃ (k : ℕ), coat = k * shirt →
  coat / shirt = 5 := by
sorry

end NUMINAMATH_CALUDE_coat_shirt_ratio_l2678_267894


namespace NUMINAMATH_CALUDE_jeremys_beads_l2678_267836

theorem jeremys_beads (n : ℕ) : n > 1 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 2 ∧ 
  n % 9 = 2 → 
  (∀ m : ℕ, m > 1 ∧ m % 5 = 2 ∧ m % 7 = 2 ∧ m % 9 = 2 → m ≥ n) →
  n = 317 := by
sorry

end NUMINAMATH_CALUDE_jeremys_beads_l2678_267836


namespace NUMINAMATH_CALUDE_sum_of_digits_8_pow_2004_l2678_267890

/-- The sum of the tens digit and the units digit of 8^2004 in its decimal representation -/
def sum_of_digits : ℕ :=
  let n := 8^2004
  let tens_digit := (n / 10) % 10
  let units_digit := n % 10
  tens_digit + units_digit

/-- Theorem stating that the sum of the tens digit and the units digit of 8^2004 is 7 -/
theorem sum_of_digits_8_pow_2004 : sum_of_digits = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_8_pow_2004_l2678_267890


namespace NUMINAMATH_CALUDE_expression_equality_l2678_267864

theorem expression_equality : (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2678_267864


namespace NUMINAMATH_CALUDE_part_one_part_two_l2678_267835

open Real

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- Theorem for part (1)
theorem part_one (t : Triangle) 
  (h1 : t.a^2 = 4 * Real.sqrt 3 * t.S)
  (h2 : t.C = π/3)
  (h3 : t.b = 1) : 
  t.a = 3 := by sorry

-- Theorem for part (2)
theorem part_two (t : Triangle)
  (h1 : t.a^2 = 4 * Real.sqrt 3 * t.S)
  (h2 : t.c / t.b = 2 + Real.sqrt 3) :
  t.A = π/3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2678_267835


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2678_267888

/-- The complex number z = (-2-3i)/i is in the second quadrant of the complex plane -/
theorem complex_number_in_second_quadrant : 
  let z : ℂ := (-2 - 3*Complex.I) / Complex.I
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2678_267888


namespace NUMINAMATH_CALUDE_triangle_area_fraction_l2678_267886

/-- The size of the grid -/
def gridSize : ℕ := 6

/-- The coordinates of the triangle vertices -/
def triangleVertices : List (ℕ × ℕ) := [(3, 3), (3, 5), (5, 5)]

/-- The area of the triangle -/
def triangleArea : ℚ := 2

/-- The area of the entire grid -/
def gridArea : ℕ := gridSize * gridSize

/-- The fraction of the grid area occupied by the triangle -/
def areaFraction : ℚ := triangleArea / gridArea

theorem triangle_area_fraction :
  areaFraction = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_triangle_area_fraction_l2678_267886


namespace NUMINAMATH_CALUDE_power_three_mod_eleven_l2678_267883

theorem power_three_mod_eleven : 3^87 + 5 ≡ 3 [ZMOD 11] := by sorry

end NUMINAMATH_CALUDE_power_three_mod_eleven_l2678_267883


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2678_267829

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 2 * y^2 + 3 * y - 35 = (2 * y + a) * (y + b)) →
  a - b = -12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2678_267829


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2678_267876

def is_increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ q > 1

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : is_increasing_geometric_sequence a q)
  (h_sum : a 1 + a 5 = 17)
  (h_prod : a 2 * a 4 = 16) :
  q = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2678_267876


namespace NUMINAMATH_CALUDE_school_ratio_proof_l2678_267895

/-- Represents the ratio of two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Simplifies a ratio by dividing both numerator and denominator by their GCD -/
def simplifyRatio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.numerator r.denominator
  { numerator := r.numerator / gcd, denominator := r.denominator / gcd }

theorem school_ratio_proof (total_students : ℕ) (num_girls : ℕ) 
    (h1 : total_students = 300) (h2 : num_girls = 160) : 
    simplifyRatio { numerator := num_girls, denominator := total_students - num_girls } = 
    { numerator := 8, denominator := 7 } := by
  sorry

#check school_ratio_proof

end NUMINAMATH_CALUDE_school_ratio_proof_l2678_267895


namespace NUMINAMATH_CALUDE_certain_number_problem_l2678_267844

theorem certain_number_problem (h : 2994 / 14.5 = 173) : 
  ∃ x : ℝ, x / 1.45 = 17.3 ∧ x = 25.085 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2678_267844


namespace NUMINAMATH_CALUDE_derivative_f_at_2_l2678_267809

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 2

-- State the theorem
theorem derivative_f_at_2 : 
  (deriv f) 2 = 12 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_2_l2678_267809


namespace NUMINAMATH_CALUDE_basketball_time_l2678_267845

theorem basketball_time (n : ℕ) (last_activity_time : ℝ) : 
  n = 5 ∧ last_activity_time = 160 →
  (let seq := fun i => (2 ^ i) * (last_activity_time / (2 ^ (n - 1)))
   seq 0 = 10) := by
  sorry

end NUMINAMATH_CALUDE_basketball_time_l2678_267845


namespace NUMINAMATH_CALUDE_water_park_admission_l2678_267868

/-- The admission charge for a child in a water park. -/
def child_admission : ℚ :=
  3⁻¹ * (13 / 4 - 1)

/-- The total amount paid by an adult. -/
def total_paid : ℚ := 13 / 4

/-- The number of children accompanying the adult. -/
def num_children : ℕ := 3

/-- The admission charge for an adult. -/
def adult_admission : ℚ := 1

theorem water_park_admission :
  child_admission * num_children + adult_admission = total_paid :=
sorry

end NUMINAMATH_CALUDE_water_park_admission_l2678_267868


namespace NUMINAMATH_CALUDE_polynomial_inequality_solution_l2678_267830

theorem polynomial_inequality_solution (x : ℝ) : 
  x^4 - 15*x^3 + 80*x^2 - 200*x > 0 ↔ (0 < x ∧ x < 5) ∨ x > 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_solution_l2678_267830


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_equation_proof_l2678_267801

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (given_line : Line)
  (given_point : Point)
  (result_line : Line) : Prop :=
  (given_line.a = 2 ∧ given_line.b = -3 ∧ given_line.c = 5) →
  (given_point.x = -2 ∧ given_point.y = 1) →
  (result_line.a = 2 ∧ result_line.b = -3 ∧ result_line.c = 7) →
  (given_point.liesOn result_line ∧ result_line.isParallelTo given_line)

-- The proof of the theorem
theorem line_equation_proof : line_through_point_parallel_to_line 
  (Line.mk 2 (-3) 5) 
  (Point.mk (-2) 1) 
  (Line.mk 2 (-3) 7) := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_equation_proof_l2678_267801


namespace NUMINAMATH_CALUDE_fencing_requirement_l2678_267881

/-- Given a rectangular field with area 210 sq. feet and one side 20 feet,
    prove that the sum of the other three sides is 41 feet. -/
theorem fencing_requirement (area : ℝ) (length : ℝ) (width : ℝ) : 
  area = 210 →
  length = 20 →
  area = length * width →
  2 * width + length = 41 := by
  sorry

end NUMINAMATH_CALUDE_fencing_requirement_l2678_267881


namespace NUMINAMATH_CALUDE_min_product_of_geometric_sequence_l2678_267858

theorem min_product_of_geometric_sequence (x y : ℝ) 
  (hx : x > 1) (hy : y > 1) 
  (h_seq : (Real.log x) * (Real.log y) = (1/2)^2) : 
  x * y ≥ Real.exp 1 ∧ ∃ x y, x > 1 ∧ y > 1 ∧ (Real.log x) * (Real.log y) = (1/2)^2 ∧ x * y = Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_geometric_sequence_l2678_267858


namespace NUMINAMATH_CALUDE_tables_required_l2678_267857

-- Define the base-5 number
def base5_seating : ℕ := 3 * 5^2 + 2 * 5^1 + 1 * 5^0

-- Define the number of people per table
def people_per_table : ℕ := 3

-- Theorem to prove
theorem tables_required :
  (base5_seating + people_per_table - 1) / people_per_table = 29 := by
  sorry

end NUMINAMATH_CALUDE_tables_required_l2678_267857


namespace NUMINAMATH_CALUDE_diameters_intersect_l2678_267873

-- Define a convex set in a plane
def ConvexSet (S : Set (Real × Real)) : Prop :=
  ∀ x y : Real × Real, x ∈ S → y ∈ S → ∀ t : Real, 0 ≤ t ∧ t ≤ 1 →
    (t * x.1 + (1 - t) * y.1, t * x.2 + (1 - t) * y.2) ∈ S

-- Define a diameter of a convex set
def Diameter (S : Set (Real × Real)) (d : Set (Real × Real)) : Prop :=
  ConvexSet S ∧ d ⊆ S ∧ ∀ x y : Real × Real, x ∈ S → y ∈ S →
    ∃ a b : Real × Real, a ∈ d ∧ b ∈ d ∧ 
      (a.1 - b.1)^2 + (a.2 - b.2)^2 ≥ (x.1 - y.1)^2 + (x.2 - y.2)^2

-- Theorem statement
theorem diameters_intersect (S : Set (Real × Real)) (d1 d2 : Set (Real × Real)) :
  ConvexSet S → Diameter S d1 → Diameter S d2 → (d1 ∩ d2).Nonempty := by
  sorry

end NUMINAMATH_CALUDE_diameters_intersect_l2678_267873


namespace NUMINAMATH_CALUDE_specific_conference_handshakes_l2678_267838

/-- The number of distinct handshakes in a conference --/
def conference_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating the number of handshakes for the specific conference scenario --/
theorem specific_conference_handshakes :
  conference_handshakes 3 5 = 75 := by
  sorry

#eval conference_handshakes 3 5

end NUMINAMATH_CALUDE_specific_conference_handshakes_l2678_267838


namespace NUMINAMATH_CALUDE_expansion_contains_constant_term_l2678_267841

/-- The expansion of (√x - 2/x)^n contains a constant term for some positive integer n -/
theorem expansion_contains_constant_term : ∃ (n : ℕ+), 
  ∃ (r : ℕ), n = 3 * r := by
  sorry

end NUMINAMATH_CALUDE_expansion_contains_constant_term_l2678_267841


namespace NUMINAMATH_CALUDE_equation_solution_l2678_267855

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2678_267855


namespace NUMINAMATH_CALUDE_union_A_B_minus_three_intersection_A_B_equals_B_iff_l2678_267866

-- Define set A
def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 1}

-- Statement 1: A ∪ B when m = -3
theorem union_A_B_minus_three : 
  A ∪ B (-3) = {x : ℝ | -7 ≤ x ∧ x ≤ 4} := by sorry

-- Statement 2: A ∩ B = B iff m ≥ -1
theorem intersection_A_B_equals_B_iff (m : ℝ) : 
  A ∩ B m = B m ↔ m ≥ -1 := by sorry

end NUMINAMATH_CALUDE_union_A_B_minus_three_intersection_A_B_equals_B_iff_l2678_267866


namespace NUMINAMATH_CALUDE_race_time_proof_l2678_267875

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Represents a race between two runners -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner

/-- The conditions of the specific race described in the problem -/
def race_conditions (r : Race) : Prop :=
  r.distance = 1000 ∧
  r.runner_a.speed * r.runner_a.time = r.distance ∧
  r.runner_b.speed * r.runner_b.time = r.distance ∧
  (r.runner_a.speed * r.runner_a.time - r.runner_b.speed * r.runner_a.time = 50 ∨
   r.runner_b.time - r.runner_a.time = 20)

theorem race_time_proof (r : Race) (h : race_conditions r) : r.runner_a.time = 400 := by
  sorry

end NUMINAMATH_CALUDE_race_time_proof_l2678_267875


namespace NUMINAMATH_CALUDE_larger_number_proof_l2678_267816

theorem larger_number_proof (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  (Nat.gcd a b = 23) → 
  (Nat.lcm a b = 23 * 12 * 13) → 
  (max a b = 299) := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2678_267816


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2678_267846

theorem cubic_root_sum (a b : ℝ) : 
  (∃ r s t : ℝ, r > 0 ∧ s > 0 ∧ t > 0 ∧ r ≠ s ∧ s ≠ t ∧ r ≠ t ∧
   (∀ x : ℝ, 4*x^3 + 7*a*x^2 + 6*b*x + 2*a = 0 ↔ (x = r ∨ x = s ∨ x = t)) ∧
   (r + s + t)^3 = 125) →
  a = -20/7 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2678_267846


namespace NUMINAMATH_CALUDE_g_zero_at_seven_fifths_l2678_267827

def g (x : ℝ) : ℝ := 5 * x - 7

theorem g_zero_at_seven_fifths : g (7 / 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_at_seven_fifths_l2678_267827


namespace NUMINAMATH_CALUDE_grid_midpoint_theorem_l2678_267831

theorem grid_midpoint_theorem (points : Finset (ℤ × ℤ)) 
  (h : points.card = 5) :
  ∃ p1 p2 : ℤ × ℤ, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ 
  (∃ m : ℤ × ℤ, m.1 * 2 = p1.1 + p2.1 ∧ m.2 * 2 = p1.2 + p2.2) :=
sorry

end NUMINAMATH_CALUDE_grid_midpoint_theorem_l2678_267831


namespace NUMINAMATH_CALUDE_teachers_gathering_problem_l2678_267814

theorem teachers_gathering_problem (male_teachers female_teachers : ℕ) 
  (h1 : female_teachers = male_teachers + 12)
  (h2 : (male_teachers : ℚ) / (male_teachers + female_teachers) = 9 / 20) :
  male_teachers + female_teachers = 120 := by
sorry

end NUMINAMATH_CALUDE_teachers_gathering_problem_l2678_267814


namespace NUMINAMATH_CALUDE_range_of_a_l2678_267871

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) → a ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2678_267871


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2678_267803

theorem complex_equation_sum (a b : ℝ) : 
  (Complex.mk a 3 + Complex.mk 2 (-1) = Complex.mk 5 b) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2678_267803


namespace NUMINAMATH_CALUDE_min_beta_delta_sum_l2678_267822

open Complex

/-- The complex number i -/
def i : ℂ := Complex.I

/-- The function g as defined in the problem -/
def g (β δ : ℂ) (z : ℂ) : ℂ := (3 + 2*i)*z^2 + β*z + δ

/-- The theorem statement -/
theorem min_beta_delta_sum :
  ∀ β δ : ℂ, (g β δ 1).im = 0 → (g β δ (-i)).im = 0 → 
  ∃ (min : ℝ), min = 2 * Real.sqrt 2 ∧ 
  ∀ β' δ' : ℂ, (g β' δ' 1).im = 0 → (g β' δ' (-i)).im = 0 → 
  Complex.abs β' + Complex.abs δ' ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_beta_delta_sum_l2678_267822


namespace NUMINAMATH_CALUDE_parabola_translation_l2678_267865

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk (-1) 0 2
  let translated := translate original 2 (-3)
  y = -(x - 2)^2 - 1 ↔ y = translated.a * x^2 + translated.b * x + translated.c :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_l2678_267865


namespace NUMINAMATH_CALUDE_problem_solution_l2678_267807

theorem problem_solution (a b : ℝ) 
  (h1 : 1 < a) 
  (h2 : a < b) 
  (h3 : 1 / a + 1 / b = 1) 
  (h4 : a * b = 6) : 
  b = 3 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2678_267807


namespace NUMINAMATH_CALUDE_linear_function_properties_l2678_267899

def f (x : ℝ) : ℝ := -2 * x + 2

theorem linear_function_properties :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ f x = y) ∧  -- First quadrant
  (∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ f x = y) ∧  -- Second quadrant
  (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ f x = y) ∧  -- Fourth quadrant
  (f 1 = 0) ∧                               -- x-intercept
  (∀ x > 0, f x < 2) ∧                      -- y < 2 when x > 0
  (∀ x1 x2, x1 < x2 → f x1 > f x2)          -- y decreases as x increases
  := by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l2678_267899


namespace NUMINAMATH_CALUDE_vann_teeth_cleaning_l2678_267878

/-- The number of teeth a dog has -/
def dog_teeth : ℕ := 42

/-- The number of teeth a cat has -/
def cat_teeth : ℕ := 30

/-- The number of teeth a pig has -/
def pig_teeth : ℕ := 28

/-- The number of dogs Vann will clean -/
def num_dogs : ℕ := 5

/-- The number of cats Vann will clean -/
def num_cats : ℕ := 10

/-- The number of pigs Vann will clean -/
def num_pigs : ℕ := 7

/-- The total number of teeth Vann will clean -/
def total_teeth : ℕ := dog_teeth * num_dogs + cat_teeth * num_cats + pig_teeth * num_pigs

theorem vann_teeth_cleaning :
  total_teeth = 706 := by sorry

end NUMINAMATH_CALUDE_vann_teeth_cleaning_l2678_267878


namespace NUMINAMATH_CALUDE_quadratic_root_expression_l2678_267837

theorem quadratic_root_expression (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ + 2 = 0 →
  x₂^2 - 4*x₂ + 2 = 0 →
  x₁ + x₂ = 4 →
  x₁ * x₂ = 2 →
  x₁^2 - 4*x₁ + 2*x₁*x₂ = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_expression_l2678_267837


namespace NUMINAMATH_CALUDE_unique_base_for_good_number_l2678_267805

def is_good_number (m : ℕ) : Prop :=
  ∃ (p n : ℕ), n ≥ 2 ∧ Nat.Prime p ∧ m = p^n

theorem unique_base_for_good_number :
  ∀ b : ℕ, (is_good_number (b^2 - 2*b - 3)) ↔ b = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_base_for_good_number_l2678_267805


namespace NUMINAMATH_CALUDE_number_comparison_l2678_267843

theorem number_comparison : 0.6^7 < 0.7^6 ∧ 0.7^6 < 6^0.7 := by
  sorry

end NUMINAMATH_CALUDE_number_comparison_l2678_267843


namespace NUMINAMATH_CALUDE_triangle_cosine_double_angle_l2678_267889

theorem triangle_cosine_double_angle 
  (A B C : Real) (a b c : Real) (S : Real) :
  c = 5 →
  B = 2 * Real.pi / 3 →
  S = 15 * Real.sqrt 3 / 4 →
  S = 1/2 * a * c * Real.sin B →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  Real.sin A / a = Real.sin B / b →
  Real.cos (2*A) = 71/98 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_double_angle_l2678_267889


namespace NUMINAMATH_CALUDE_heartsuit_three_eight_l2678_267817

-- Define the ⬥ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x - 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = -36 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_eight_l2678_267817


namespace NUMINAMATH_CALUDE_expression_evaluation_l2678_267853

theorem expression_evaluation :
  (-1)^2008 + (-1)^2009 + 2^2006 * (-1)^2007 + 1^2010 = -2^2006 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2678_267853


namespace NUMINAMATH_CALUDE_gcd_2023_2048_l2678_267823

theorem gcd_2023_2048 : Nat.gcd 2023 2048 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2023_2048_l2678_267823


namespace NUMINAMATH_CALUDE_total_interest_is_330_l2678_267840

/-- Calculates the total interest for a stock over 5 years with increasing rates -/
def stockInterest (initialRate : ℚ) : ℚ :=
  let faceValue : ℚ := 100
  let yearlyIncrease : ℚ := 2 / 100
  (initialRate + yearlyIncrease) * faceValue +
  (initialRate + 2 * yearlyIncrease) * faceValue +
  (initialRate + 3 * yearlyIncrease) * faceValue +
  (initialRate + 4 * yearlyIncrease) * faceValue +
  (initialRate + 5 * yearlyIncrease) * faceValue

/-- Calculates the total interest for all three stocks over 5 years -/
def totalInterest : ℚ :=
  let stock1 : ℚ := 16 / 100
  let stock2 : ℚ := 12 / 100
  let stock3 : ℚ := 20 / 100
  stockInterest stock1 + stockInterest stock2 + stockInterest stock3

theorem total_interest_is_330 : totalInterest = 330 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_is_330_l2678_267840


namespace NUMINAMATH_CALUDE_sufficient_condition_for_reciprocal_inequality_l2678_267856

theorem sufficient_condition_for_reciprocal_inequality (a b : ℝ) :
  b < a ∧ a < 0 → (1 : ℝ) / a < (1 : ℝ) / b :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_reciprocal_inequality_l2678_267856


namespace NUMINAMATH_CALUDE_f_satisfies_data_points_l2678_267832

/-- The function f that we want to prove fits the data points -/
def f (x : ℝ) : ℝ := x^2 + 3*x + 1

/-- The set of data points given in the problem -/
def data_points : List (ℝ × ℝ) := [(1, 5), (2, 11), (3, 19), (4, 29), (5, 41)]

/-- Theorem stating that f satisfies all given data points -/
theorem f_satisfies_data_points : ∀ (point : ℝ × ℝ), point ∈ data_points → f point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_data_points_l2678_267832


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_reciprocal_sum_equality_condition_l2678_267800

theorem min_value_sqrt_sum_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a + b) * (1 / Real.sqrt a + 1 / Real.sqrt b) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a + b) * (1 / Real.sqrt a + 1 / Real.sqrt b) = 2 * Real.sqrt 2 ↔ a = b :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_reciprocal_sum_equality_condition_l2678_267800


namespace NUMINAMATH_CALUDE_quadratic_sum_l2678_267867

/-- A quadratic function with specific properties -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: For a quadratic function g(x) = ax^2 + bx + c with vertex at (-2, 6) 
    and passing through (0, 2), the value of a + 2b + c is -7 -/
theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, g a b c x = a * x^2 + b * x + c) →  -- Definition of g
  g a b c (-2) = 6 →                        -- Vertex at (-2, 6)
  (∀ x, g a b c x ≤ 6) →                   -- (-2, 6) is the maximum point
  g a b c 0 = 2 →                          -- Point (0, 2) on the graph
  a + 2*b + c = -7 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2678_267867


namespace NUMINAMATH_CALUDE_vector_properties_l2678_267839

def a : ℝ × ℝ := (-3, 2)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (3, -1)

theorem vector_properties :
  (∃ (t : ℝ), ∀ (s : ℝ), ‖a + s • b‖ ≥ ‖a + t • b‖ ∧ ‖a + t • b‖ = (7 * Real.sqrt 5) / 5) ∧
  (∃ (t : ℝ), ∃ (k : ℝ), a - t • b = k • c) :=
sorry

end NUMINAMATH_CALUDE_vector_properties_l2678_267839


namespace NUMINAMATH_CALUDE_birthday_stickers_l2678_267849

theorem birthday_stickers (initial_stickers total_stickers : ℕ) 
  (h1 : initial_stickers = 269)
  (h2 : total_stickers = 423) : 
  total_stickers - initial_stickers = 154 := by
  sorry

end NUMINAMATH_CALUDE_birthday_stickers_l2678_267849


namespace NUMINAMATH_CALUDE_smallest_block_size_block_with_336_cubes_exists_l2678_267880

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of invisible cubes when viewed from a corner -/
def invisibleCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Calculates the total number of cubes in the block -/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Theorem stating the smallest possible number of cubes in the block -/
theorem smallest_block_size (d : BlockDimensions) :
  invisibleCubes d = 143 → totalCubes d ≥ 336 := by
  sorry

/-- Theorem proving the existence of a block with 336 cubes and 143 invisible cubes -/
theorem block_with_336_cubes_exists :
  ∃ d : BlockDimensions, invisibleCubes d = 143 ∧ totalCubes d = 336 := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_size_block_with_336_cubes_exists_l2678_267880


namespace NUMINAMATH_CALUDE_three_element_subsets_of_eight_l2678_267820

theorem three_element_subsets_of_eight (S : Finset Nat) :
  S.card = 8 → (S.powerset.filter (fun s => s.card = 3)).card = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_element_subsets_of_eight_l2678_267820


namespace NUMINAMATH_CALUDE_decimal_point_problem_l2678_267815

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 9 * (1 / x)) : 
  x = 3 * Real.sqrt 10 / 100 := by
sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l2678_267815


namespace NUMINAMATH_CALUDE_shaded_angle_is_fifteen_degrees_l2678_267850

/-- A configuration of three identical isosceles triangles in a square -/
structure TrianglesInSquare where
  /-- The measure of the angle where three triangles meet at a corner of the square -/
  corner_angle : ℝ
  /-- The measure of each of the two equal angles in each isosceles triangle -/
  isosceles_angle : ℝ
  /-- Axiom: The corner angle is formed by three equal parts -/
  corner_angle_eq : corner_angle = 90 / 3
  /-- Axiom: The sum of angles in each isosceles triangle is 180° -/
  triangle_sum : corner_angle + 2 * isosceles_angle = 180

/-- The theorem to be proved -/
theorem shaded_angle_is_fifteen_degrees (t : TrianglesInSquare) :
  90 - t.isosceles_angle = 15 := by
  sorry

end NUMINAMATH_CALUDE_shaded_angle_is_fifteen_degrees_l2678_267850


namespace NUMINAMATH_CALUDE_initial_men_correct_l2678_267854

/-- The number of men initially working -/
def initial_men : ℕ := 72

/-- The depth dug by the initial group in meters -/
def initial_depth : ℕ := 30

/-- The hours worked by the initial group per day -/
def initial_hours : ℕ := 8

/-- The new depth to be dug in meters -/
def new_depth : ℕ := 50

/-- The new hours to be worked per day -/
def new_hours : ℕ := 6

/-- The number of extra men needed for the new task -/
def extra_men : ℕ := 88

/-- Theorem stating that the initial number of men is correct -/
theorem initial_men_correct : 
  (initial_depth : ℚ) / (initial_hours * initial_men) = 
  (new_depth : ℚ) / (new_hours * (initial_men + extra_men)) :=
sorry

end NUMINAMATH_CALUDE_initial_men_correct_l2678_267854


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2678_267819

theorem complex_equation_solution (z : ℂ) :
  (3 + 4 * Complex.I) * z = 1 - 2 * Complex.I →
  z = -1/5 - 2/5 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2678_267819


namespace NUMINAMATH_CALUDE_product_80641_9999_l2678_267896

theorem product_80641_9999 : 80641 * 9999 = 806329359 := by
  sorry

end NUMINAMATH_CALUDE_product_80641_9999_l2678_267896


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l2678_267877

/-- The length of the generatrix of a cone formed by a semi-circular iron sheet -/
def generatrix_length (base_radius : ℝ) : ℝ :=
  2 * base_radius

/-- Theorem: The length of the generatrix of the cone is 8 cm -/
theorem cone_generatrix_length :
  let base_radius : ℝ := 4
  generatrix_length base_radius = 8 := by
  sorry

#check cone_generatrix_length

end NUMINAMATH_CALUDE_cone_generatrix_length_l2678_267877


namespace NUMINAMATH_CALUDE_deceased_member_income_family_income_problem_l2678_267859

theorem deceased_member_income 
  (initial_members : ℕ) 
  (initial_avg_income : ℚ) 
  (final_members : ℕ) 
  (final_avg_income : ℚ) : ℚ :=
  let initial_total_income := initial_members * initial_avg_income
  let final_total_income := final_members * final_avg_income
  initial_total_income - final_total_income

theorem family_income_problem 
  (h1 : initial_members = 4)
  (h2 : initial_avg_income = 840)
  (h3 : final_members = 3)
  (h4 : final_avg_income = 650) : 
  deceased_member_income initial_members initial_avg_income final_members final_avg_income = 1410 := by
  sorry

end NUMINAMATH_CALUDE_deceased_member_income_family_income_problem_l2678_267859


namespace NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_271_l2678_267898

theorem first_nonzero_digit_after_decimal_1_271 :
  ∃ (n : ℕ) (r : ℚ), 1000 * (1 / 271) = n + r ∧ n = 3 ∧ 0 < r ∧ r < 1 := by
  sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_271_l2678_267898


namespace NUMINAMATH_CALUDE_no_three_numbers_with_special_property_l2678_267879

theorem no_three_numbers_with_special_property : 
  ¬ (∃ (a b c : ℕ), 
    a > 1 ∧ b > 1 ∧ c > 1 ∧
    b ∣ (a^2 - 1) ∧ c ∣ (a^2 - 1) ∧
    a ∣ (b^2 - 1) ∧ c ∣ (b^2 - 1) ∧
    a ∣ (c^2 - 1) ∧ b ∣ (c^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_three_numbers_with_special_property_l2678_267879


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2678_267884

theorem complex_equation_solution (z : ℂ) : (1 + z) * Complex.I = 1 - z → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2678_267884


namespace NUMINAMATH_CALUDE_total_amount_shared_l2678_267812

theorem total_amount_shared (T : ℝ) : 
  (0.4 * T = 0.3 * T + 5) → T = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_shared_l2678_267812


namespace NUMINAMATH_CALUDE_wall_length_is_800_l2678_267885

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  height : ℝ
  width : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.height * w.width

/-- Theorem: The length of the wall is 800 cm -/
theorem wall_length_is_800 (brick : BrickDimensions) (wall : WallDimensions) 
    (h1 : brick.length = 40)
    (h2 : brick.width = 11.25)
    (h3 : brick.height = 6)
    (h4 : wall.height = 600)
    (h5 : wall.width = 22.5)
    (h6 : wallVolume wall / brickVolume brick = 4000) :
    wall.length = 800 := by
  sorry

end NUMINAMATH_CALUDE_wall_length_is_800_l2678_267885


namespace NUMINAMATH_CALUDE_garden_width_to_perimeter_ratio_l2678_267808

/-- Given a rectangular garden with length 23 feet and width 15 feet, 
    the ratio of its width to its perimeter is 15:76. -/
theorem garden_width_to_perimeter_ratio :
  let garden_length : ℕ := 23
  let garden_width : ℕ := 15
  let perimeter : ℕ := 2 * (garden_length + garden_width)
  (garden_width : ℚ) / perimeter = 15 / 76 := by
  sorry

end NUMINAMATH_CALUDE_garden_width_to_perimeter_ratio_l2678_267808


namespace NUMINAMATH_CALUDE_min_weighings_to_find_fake_pearl_l2678_267882

/-- Represents the result of a weighing operation -/
inductive WeighResult
  | Equal : WeighResult
  | Left : WeighResult
  | Right : WeighResult

/-- Represents a strategy for finding the fake pearl -/
def Strategy := List WeighResult → Nat

/-- The number of pearls -/
def numPearls : Nat := 9

/-- The minimum number of weighings needed to find the fake pearl -/
def minWeighings : Nat := 2

/-- A theorem stating that the minimum number of weighings to find the fake pearl is 2 -/
theorem min_weighings_to_find_fake_pearl :
  ∃ (s : Strategy), ∀ (outcomes : List WeighResult),
    outcomes.length ≤ minWeighings →
    s outcomes < numPearls ∧
    (∀ (t : Strategy),
      (∀ (outcomes' : List WeighResult),
        outcomes'.length < outcomes.length →
        t outcomes' = numPearls) →
      s outcomes ≤ t outcomes) :=
sorry

end NUMINAMATH_CALUDE_min_weighings_to_find_fake_pearl_l2678_267882


namespace NUMINAMATH_CALUDE_cos_A_in_third_quadrant_l2678_267842

theorem cos_A_in_third_quadrant (A : Real) :
  (A > π ∧ A < 3*π/2) →  -- Angle A is in the third quadrant
  (Real.sin A = -1/3) →  -- sin A = -1/3
  (Real.cos A = -2*Real.sqrt 2/3) :=  -- cos A = -2√2/3
by sorry

end NUMINAMATH_CALUDE_cos_A_in_third_quadrant_l2678_267842


namespace NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l2678_267802

/-- The height of a right circular cylinder inscribed in a hemisphere -/
theorem cylinder_height_in_hemisphere (r_cylinder : ℝ) (r_hemisphere : ℝ) 
  (h_cylinder : r_cylinder = 3)
  (h_hemisphere : r_hemisphere = 7) :
  Real.sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l2678_267802


namespace NUMINAMATH_CALUDE_divide_l_shaped_ice_sheet_l2678_267862

/-- Represents an L-shaped ice sheet composed of three unit squares -/
structure LShapedIceSheet :=
  (area : ℝ := 3)

/-- Represents a part of the divided ice sheet -/
structure IceSheetPart :=
  (area : ℝ)

/-- Theorem stating that the L-shaped ice sheet can be divided into four equal parts -/
theorem divide_l_shaped_ice_sheet (sheet : LShapedIceSheet) :
  ∃ (part1 part2 part3 part4 : IceSheetPart),
    part1.area = 3/4 ∧
    part2.area = 3/4 ∧
    part3.area = 3/4 ∧
    part4.area = 3/4 ∧
    part1.area + part2.area + part3.area + part4.area = sheet.area :=
sorry

end NUMINAMATH_CALUDE_divide_l_shaped_ice_sheet_l2678_267862


namespace NUMINAMATH_CALUDE_octal_sum_equality_l2678_267892

/-- Converts a list of digits in base 8 to a natural number -/
def fromOctal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- The sum of 235₈, 647₈, and 54₈ is equal to 1160₈ -/
theorem octal_sum_equality :
  fromOctal [2, 3, 5] + fromOctal [6, 4, 7] + fromOctal [5, 4] = fromOctal [1, 1, 6, 0] := by
  sorry

#eval fromOctal [2, 3, 5] + fromOctal [6, 4, 7] + fromOctal [5, 4]
#eval fromOctal [1, 1, 6, 0]

end NUMINAMATH_CALUDE_octal_sum_equality_l2678_267892


namespace NUMINAMATH_CALUDE_parkway_fifth_grade_count_l2678_267848

/-- The number of students in the fifth grade at Parkway Elementary School -/
def total_students : ℕ := sorry

/-- The number of boys in the fifth grade -/
def num_boys : ℕ := 312

/-- The number of students playing soccer -/
def num_soccer : ℕ := 250

/-- The proportion of boys among students playing soccer -/
def prop_boys_soccer : ℚ := 78 / 100

/-- The number of girls not playing soccer -/
def num_girls_not_soccer : ℕ := 53

theorem parkway_fifth_grade_count :
  total_students = 420 :=
by sorry

end NUMINAMATH_CALUDE_parkway_fifth_grade_count_l2678_267848


namespace NUMINAMATH_CALUDE_gcd_459_357_l2678_267825

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2678_267825


namespace NUMINAMATH_CALUDE_total_broken_marbles_l2678_267891

def marble_set_1 : ℕ := 50
def marble_set_2 : ℕ := 60
def broken_percent_1 : ℚ := 10 / 100
def broken_percent_2 : ℚ := 20 / 100

theorem total_broken_marbles :
  ⌊marble_set_1 * broken_percent_1⌋ + ⌊marble_set_2 * broken_percent_2⌋ = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_broken_marbles_l2678_267891


namespace NUMINAMATH_CALUDE_broken_line_enclosing_circle_l2678_267851

/-- A closed broken line in a metric space -/
structure ClosedBrokenLine (α : Type*) [MetricSpace α] where
  points : Set α
  is_closed : IsClosed points
  is_connected : IsConnected points
  perimeter : ℝ

/-- Theorem: Any closed broken line can be enclosed in a circle with radius not exceeding its perimeter divided by 4 -/
theorem broken_line_enclosing_circle 
  {α : Type*} [MetricSpace α] (L : ClosedBrokenLine α) :
  ∃ (center : α), ∀ (p : α), p ∈ L.points → dist center p ≤ L.perimeter / 4 := by
  sorry

end NUMINAMATH_CALUDE_broken_line_enclosing_circle_l2678_267851


namespace NUMINAMATH_CALUDE_crayon_count_theorem_l2678_267833

/-- Represents the number of crayons in various states --/
structure CrayonCounts where
  initial : ℕ
  givenAway : ℕ
  lost : ℕ
  remaining : ℕ

/-- Theorem stating the relationship between crayons lost, given away, and the total --/
theorem crayon_count_theorem (c : CrayonCounts) 
  (h1 : c.givenAway = 52)
  (h2 : c.lost = 535)
  (h3 : c.remaining = 492) :
  c.givenAway + c.lost = 587 := by
  sorry

end NUMINAMATH_CALUDE_crayon_count_theorem_l2678_267833


namespace NUMINAMATH_CALUDE_cube_root_product_simplification_l2678_267897

theorem cube_root_product_simplification :
  (1 + 27) ^ (1/3) * (1 + 27 ^ (1/3)) ^ (1/3) = 112 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_product_simplification_l2678_267897


namespace NUMINAMATH_CALUDE_oranges_left_l2678_267887

theorem oranges_left (initial_oranges : ℕ) (taken_oranges : ℕ) : 
  initial_oranges = 60 → taken_oranges = 35 → initial_oranges - taken_oranges = 25 := by
  sorry

end NUMINAMATH_CALUDE_oranges_left_l2678_267887


namespace NUMINAMATH_CALUDE_hair_cut_length_l2678_267893

/-- The length of hair cut off is equal to the difference between the initial and final hair lengths. -/
theorem hair_cut_length (initial_length final_length cut_length : ℝ) 
  (h1 : initial_length = 11)
  (h2 : final_length = 7)
  (h3 : cut_length = initial_length - final_length) :
  cut_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_length_l2678_267893


namespace NUMINAMATH_CALUDE_fractional_equation_positive_root_l2678_267804

theorem fractional_equation_positive_root (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ m / (x - 2) = (1 - x) / (2 - x) - 3) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_positive_root_l2678_267804


namespace NUMINAMATH_CALUDE_quadratic_inequality_result_l2678_267811

theorem quadratic_inequality_result (x : ℝ) :
  x^2 - 5*x + 6 < 0 → x^2 - 5*x + 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_result_l2678_267811
