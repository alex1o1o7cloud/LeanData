import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_and_range_l998_99876

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x

theorem f_zeros_and_range (a : ℝ) :
  (a < 0 → ∃! x, x > 0 ∧ f a x = 0) ∧
  (0 < a ∧ a < Real.exp 1 → ∀ x, x > 0 → f a x ≠ 0) ∧
  (∀ x, x > 1 → f a x ≥ a * x^a * Real.log x - x * Real.exp x → a < Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_and_range_l998_99876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_g_6_l998_99806

-- Define the functions s and g
noncomputable def s (x : ℝ) : ℝ := Real.sqrt (4 * x + 2)
noncomputable def g (x : ℝ) : ℝ := 7 - s x

-- State the theorem
theorem s_of_g_6 : s (g 6) = Real.sqrt (30 - 4 * Real.sqrt 26) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_g_6_l998_99806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a1_value_l998_99818

def mySequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = 1 / (1 - a n)

theorem sequence_a1_value (a : ℕ → ℚ) (h : mySequence a) (h8 : a 8 = 2) : a 1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a1_value_l998_99818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_abs_equation_l998_99808

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x = -2 - Real.sqrt 7 ∧
  x * abs x = 4 * x - 3 ∧
  ∀ (y : ℝ), y * abs y = 4 * y - 3 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_abs_equation_l998_99808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l998_99852

theorem diophantine_equation_solutions :
  {(x, y, z) : ℕ × ℕ × ℕ | x^2 + y^2 - 15 = 2^z} =
  {(0, 4, 0), (4, 0, 0), (1, 4, 1), (4, 1, 1)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l998_99852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leading_coefficient_is_eight_l998_99850

def polynomial (x : ℝ) : ℝ := 2*(2*x^4 - 3*x^3 + 4*x) - 5*(x^4 + x^3 - 2) + 3*(3*x^4 - 2*x^2 + 1)

theorem leading_coefficient_is_eight :
  ∃ (a : ℝ) (p : ℝ → ℝ), (∀ x, polynomial x = 8 * x^4 + a * x^3 + p x) ∧ 
  (∀ ε > 0, ∃ M, ∀ x > M, |p x / x^3| < ε) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leading_coefficient_is_eight_l998_99850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l998_99863

-- Define the function f(x) = |sin x| + cos x
noncomputable def f (x : ℝ) := abs (Real.sin x) + Real.cos x

-- Theorem statement
theorem f_properties :
  -- 1. f is an even function
  (∀ x : ℝ, f (-x) = f x) ∧ 
  -- 2. The range of f is [-1, √2]
  (∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -1 ≤ y ∧ y ≤ Real.sqrt 2) ∧
  -- 3. The maximum length of the interval where f is monotonically increasing is 3π/4
  (∃ a b : ℝ, b - a = 3 * Real.pi / 4 ∧
    (∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∧
    (∀ c d : ℝ, d - c > 3 * Real.pi / 4 →
      ∃ x y : ℝ, c ≤ x ∧ x < y ∧ y ≤ d ∧ f y ≤ f x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l998_99863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_in_year_l998_99800

/-- A fair eight-sided die -/
def Die := Fin 8

/-- The probability of rolling each number on the die -/
noncomputable def prob (n : Die) : ℝ := 1 / 8

/-- Whether a number is divisible by 3 -/
def divisible_by_three (n : Die) : Prop := n.val % 3 = 0

/-- Whether a number is prime -/
def is_prime (n : Die) : Prop := n.val ∈ [2, 3, 5, 7]

/-- Whether Alice needs to reroll -/
def reroll (n : Die) : Prop := n.val = 1

/-- The expected number of rolls on a single day -/
noncomputable def expected_rolls_per_day : ℝ := 6 / 7

/-- The number of days in a non-leap year -/
def days_in_year : ℕ := 365

/-- The theorem to prove -/
theorem expected_rolls_in_year :
  (expected_rolls_per_day * days_in_year : ℝ) = 6 / 7 * 365 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_in_year_l998_99800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_face_value_l998_99885

/-- Represents a stock with its characteristics -/
structure Stock where
  percentage : ℚ  -- The stock's percentage (e.g., 20 for a 20% stock)
  quotedPrice : ℚ  -- The stock's quoted price
  percentageYield : ℚ  -- The stock's percentage yield

/-- Calculates the face value of a stock given its characteristics -/
noncomputable def faceValue (s : Stock) : ℚ :=
  s.quotedPrice * s.percentageYield / s.percentage

/-- Theorem stating that a 20% stock quoted at $200 with a 10% yield has a face value of $100 -/
theorem stock_face_value :
  let s : Stock := { percentage := 20, quotedPrice := 200, percentageYield := 10 }
  faceValue s = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_face_value_l998_99885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_z_forms_cylinder_l998_99883

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying z = c in cylindrical coordinates -/
def ConstantZSet (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.z = c ∧ p.r ≥ 0 ∧ 0 ≤ p.θ ∧ p.θ < 2 * Real.pi}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) : Prop :=
  ∃ c : ℝ, S = ConstantZSet c

theorem constant_z_forms_cylinder (c : ℝ) :
  IsCylinder (ConstantZSet c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_z_forms_cylinder_l998_99883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l998_99803

noncomputable def original_expression (x : ℝ) : ℝ :=
  ((x^2 - 4*x + 4) / (2*x - x^2)) / (2*x - (4 + x^2) / x)

noncomputable def simplified_expression (x : ℝ) : ℝ :=
  -1 / (x + 2)

theorem expression_simplification (x : ℝ) 
  (h1 : x ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ -2) : 
  original_expression x = simplified_expression x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l998_99803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_50_equals_102_l998_99884

open BigOperators

def series_sum (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, (2 + (k + 1) * 4) / 3^(n - k)

theorem series_sum_50_equals_102 :
  series_sum 50 = 102 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_50_equals_102_l998_99884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_not_through_center_l998_99872

-- Define the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 3)^2 = 4

-- Define the line
def line_eq (x y : ℝ) : Prop :=
  ∃ t : ℝ, x = 2*t - 1 ∧ y = 6*t - 1

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 3)

-- Define the distance between a point and a line
noncomputable def distance_point_line (px py : ℝ) : ℝ :=
  |3 * px - py + 2| / Real.sqrt 10

-- Theorem statement
theorem line_intersects_circle_not_through_center :
  (∃ x y : ℝ, circle_eq x y ∧ line_eq x y) ∧
  ¬(line_eq center.1 center.2) ∧
  distance_point_line center.1 center.2 < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_not_through_center_l998_99872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l998_99829

theorem complex_fraction_equality : 
  (Complex.I : ℂ) / (Real.sqrt 7 + 3 * Complex.I) = 
  Complex.ofReal (3/16) + Complex.I * Complex.ofReal (Real.sqrt 7 / 16) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l998_99829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_CDE_area_l998_99819

/-- Triangle CDE with base DE and height CE -/
structure TriangleCDE where
  DE : ℝ
  CE : ℝ
  DE_positive : 0 < DE
  CE_positive : 0 < CE

/-- The area of triangle CDE is 90 square cm -/
theorem triangle_CDE_area (t : TriangleCDE) (h1 : t.DE = 12) (h2 : t.CE = 15) : 
  (1 / 2 : ℝ) * t.DE * t.CE = 90 := by
  sorry

#check triangle_CDE_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_CDE_area_l998_99819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auxiliary_angle_formula_l998_99833

theorem auxiliary_angle_formula (φ : Real) :
  (∀ θ : Real, Real.sin θ - Real.sqrt 3 * Real.cos θ = 2 * Real.sin (θ + φ)) →
  -Real.pi < φ ∧ φ < Real.pi →
  φ = -Real.pi/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_auxiliary_angle_formula_l998_99833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_not_divisible_by_four_and_digit_product_l998_99866

def numbers : List Nat := [3544, 3554, 3564, 3572, 3576]

def is_divisible_by_four (n : Nat) : Prop := n % 4 = 0

def units_digit (n : Nat) : Nat := n % 10

def tens_digit (n : Nat) : Nat := (n / 10) % 10

theorem unique_not_divisible_by_four_and_digit_product : 
  ∃! n, n ∈ numbers ∧ ¬(is_divisible_by_four n) ∧ 
  (units_digit n * tens_digit n = 20) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_not_divisible_by_four_and_digit_product_l998_99866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_l998_99807

/-- Defines the equation we want to prove has infinitely many solutions -/
def equation (a b c : ℕ+) : Prop :=
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / (a * b * c) = 12 / (a + b + c)

/-- States that there are infinitely many solutions to the equation -/
theorem infinite_solutions : ∃ (f : ℕ → ℕ+), StrictMono f ∧ ∀ n, ∃ b c, equation (f n) b c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_l998_99807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l998_99856

/-- Given a circle C and a line l, prove that the length of the chord cut by l from C is 8/5 -/
theorem chord_length (C : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + y^2 = 1) →
  (∀ (x y t : ℝ), (x, y) ∈ l ↔ x = -1 + 4*t ∧ y = 3*t) →
  ∃ (a b : ℝ × ℝ), a ∈ C ∧ b ∈ C ∧ a ∈ l ∧ b ∈ l ∧ 
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 8/5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l998_99856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_problem_l998_99894

def goldfish_remaining_to_catch (total : ℕ) (allowed_fraction : ℚ) (caught_fraction : ℚ) : ℕ :=
  let allowed := (total : ℚ) * allowed_fraction
  let caught := allowed * caught_fraction
  let remaining := allowed - caught
  (remaining.num / remaining.den).natAbs

theorem goldfish_problem : goldfish_remaining_to_catch 100 (1/2) (3/5) = 20 := by
  -- Unfold the definition and simplify
  unfold goldfish_remaining_to_catch
  -- Perform the calculations
  simp [Rat.num, Rat.den]
  -- The proof is completed
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_problem_l998_99894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_implies_result_l998_99890

/-- A function satisfying the given property -/
def SatisfiesProperty (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x * g y - g (x * y) = 2 * x - y

/-- The set of possible values for g(3) -/
def PossibleValues (g : ℝ → ℝ) : Set ℝ :=
  {z : ℝ | SatisfiesProperty g → g 3 = z}

/-- The theorem to be proved -/
theorem property_implies_result :
  ∃ g : ℝ → ℝ, SatisfiesProperty g ∧
    (Finset.card {(3 + Real.sqrt 5) / 2, (3 - Real.sqrt 5) / 2} *
     Finset.sum {(3 + Real.sqrt 5) / 2, (3 - Real.sqrt 5) / 2} id) = 6 := by
  sorry

-- Helper lemma to show that the possible values are indeed the ones we expect
lemma possible_values_are_correct (g : ℝ → ℝ) (h : SatisfiesProperty g) :
  PossibleValues g = {(3 + Real.sqrt 5) / 2, (3 - Real.sqrt 5) / 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_implies_result_l998_99890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l998_99874

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the transformed function g
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 1) - 2

-- Theorem statement
theorem graph_shift (f : ℝ → ℝ) (x y : ℝ) :
  (y = g f x) ↔ (y + 2 = f (x + 1)) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l998_99874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_3_l998_99889

-- Define the line l: x + y - 1 = 0
def line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the circle C: (x - 3)² + (y + 4)² = 5
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 5

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ circle_eq A.1 A.2 ∧
  line B.1 B.2 ∧ circle_eq B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem chord_length_is_2_sqrt_3 (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_3_l998_99889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_interest_rate_l998_99891

/-- Calculates the simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem lending_interest_rate 
  (borrowed_amount : ℝ)
  (borrowed_rate : ℝ)
  (time : ℝ)
  (annual_gain : ℝ)
  (h1 : borrowed_amount = 5000)
  (h2 : borrowed_rate = 4)
  (h3 : time = 2)
  (h4 : annual_gain = 50)
  : ∃ (lending_rate : ℝ),
    lending_rate = 5 ∧
    simpleInterest borrowed_amount lending_rate 1 =
    simpleInterest borrowed_amount borrowed_rate 1 + annual_gain :=
by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_interest_rate_l998_99891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l998_99898

/-- The distance between the foci of an ellipse -/
noncomputable def foci_distance (a b c d e f : ℝ) : ℝ :=
  2 * Real.sqrt (462 / 25)

/-- Theorem: The distance between the foci of the ellipse 
    25x^2 - 100x + 4y^2 + 8y + 16 = 0 is 2√462/5 -/
theorem ellipse_foci_distance :
  foci_distance 25 (-100) 4 8 0 16 = 2 * Real.sqrt (462 / 25) := by
  -- Unfold the definition of foci_distance
  unfold foci_distance
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l998_99898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l998_99838

/-- Triangle ABC with given side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The area of a triangle given its side lengths -/
noncomputable def area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Angle A in triangle ABC -/
noncomputable def angle_A (t : Triangle) : ℝ :=
  Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))

/-- Theorem about the area and angle A of triangle ABC -/
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.a = 1) (h2 : t.b = 2) (h3 : t.c = 2 * Real.sqrt 2) :
  area t = Real.sqrt 7 / 4 ∧ 
  ((angle_A t + π / 3 = angle_A t + angle_A t) ∨ 
   (π - angle_A t - angle_A t = 2 * angle_A t)) → 
  angle_A t = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l998_99838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l998_99899

noncomputable def complex_equation (z : ℂ) : Prop :=
  (z - 2) * (z^2 + 3*z + 5) * (z^2 + 5*z + 10) = 0

noncomputable def solutions_on_ellipse (equation : ℂ → Prop) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  ∀ (z : ℂ), equation z → (z.re^2 / a^2 + z.im^2 / b^2 = 1)

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity :
  solutions_on_ellipse complex_equation →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ eccentricity a b = Real.sqrt 3 / 2 := by
  sorry

#check ellipse_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l998_99899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_gain_percent_l998_99843

noncomputable def purchase_price : ℝ := 4700
noncomputable def repair_costs : ℝ := 800
noncomputable def selling_price : ℝ := 5800

noncomputable def total_cost : ℝ := purchase_price + repair_costs
noncomputable def gain : ℝ := selling_price - total_cost
noncomputable def gain_percent : ℝ := (gain / total_cost) * 100

theorem scooter_gain_percent :
  ∃ ε > 0, |gain_percent - 5.45| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_gain_percent_l998_99843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l998_99879

/-- The coefficient of x^3 in the expansion of (√x - 2x)^5 -/
def coefficient_x_cubed : ℤ := -10

/-- The binomial expansion of (√x - 2x)^5 -/
noncomputable def expansion (x : ℝ) : ℝ := (Real.sqrt x - 2*x)^5

theorem coefficient_x_cubed_in_expansion :
  ∃ (f : ℝ → ℝ) (c : ℝ), (∀ x, expansion x = c * x^3 + f x) ∧ c = coefficient_x_cubed := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l998_99879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_cost_l998_99820

theorem chair_cost (num_chairs : ℕ) (table_cost plate_set_cost : ℚ) 
  (num_plate_sets : ℕ) (total_given change : ℚ) : 
  num_chairs = 3 ∧
  table_cost = 50 ∧
  plate_set_cost = 20 ∧
  num_plate_sets = 2 ∧
  total_given = 130 ∧
  change = 4 →
  (total_given - change - (table_cost + num_plate_sets * plate_set_cost)) / num_chairs = 12 := by
    intro h
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_cost_l998_99820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_form_all_rectangles_up_to_7_l998_99824

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents the set of nine rectangles cut from the 7x7 square -/
def nine_rectangles : List Rectangle := [
  ⟨1, 1⟩, ⟨2, 1⟩, ⟨4, 1⟩,
  ⟨1, 2⟩, ⟨2, 2⟩, ⟨4, 2⟩,
  ⟨1, 4⟩, ⟨2, 4⟩, ⟨4, 4⟩
]

/-- Checks if a rectangle can be formed from the given set of rectangles -/
def can_form_rectangle (r : Rectangle) (pieces : List Rectangle) : Prop :=
  ∃ (subset : List Rectangle), subset ⊆ pieces ∧
    (subset.map (λ rect => rect.width * rect.height)).sum = r.width * r.height

/-- Theorem stating that any rectangle with sides not exceeding 7 can be formed -/
theorem can_form_all_rectangles_up_to_7 :
  ∀ (w h : Nat), w ≤ 7 → h ≤ 7 →
    can_form_rectangle ⟨w, h⟩ nine_rectangles := by
  sorry

#check can_form_all_rectangles_up_to_7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_form_all_rectangles_up_to_7_l998_99824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scheme_A_highest_price_l998_99845

-- Define the original price
noncomputable def original_price : ℝ := 100

-- Define the pricing schemes
noncomputable def scheme_A (m n : ℝ) : ℝ := original_price * (1 + m / 100) * (1 - n / 100)
noncomputable def scheme_B (m n : ℝ) : ℝ := original_price * (1 + n / 100) * (1 - m / 100)
noncomputable def scheme_C (m n : ℝ) : ℝ := original_price * (1 + (m + n) / 200) * (1 - (m + n) / 200)
noncomputable def scheme_D (m n : ℝ) : ℝ := original_price * (1 + m * n / 10000) * (1 - m * n / 10000)

-- State the theorem
theorem scheme_A_highest_price (m n : ℝ) (h1 : 0 < n) (h2 : n < m) (h3 : m < 100) :
  scheme_A m n ≥ scheme_B m n ∧ 
  scheme_A m n ≥ scheme_C m n ∧ 
  scheme_A m n ≥ scheme_D m n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scheme_A_highest_price_l998_99845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l998_99809

/-- Given function f(x) = (1/3)x³ + ax² - x, where a ∈ ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 - x

theorem f_properties (a : ℝ) :
  (∃ (x : ℝ), f 0 x ≤ 2/3 ∧ ∀ (y : ℝ), f 0 y ≤ f 0 x) ∧ 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (∀ (y : ℝ), y ≠ x₁ → y ≠ x₂ → 
      (f a y - f a x₁) * (f a y - f a x₂) > 0)) ∧
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l998_99809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l998_99811

def A : Set ℝ := {x | x^2 ≥ 1}

def B : Set ℝ := {x | (x - 2) / x ≤ 0}

theorem set_intersection_theorem :
  A ∩ (Set.univ \ B) = Set.Iic (-1) ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l998_99811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_condition_l998_99873

open Real

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * (a+1) * x^2 + a * x

-- State the theorem
theorem max_value_condition (a : ℝ) (h1 : a > 1) :
  (∀ x ∈ Set.Icc 0 (a+1), f a x ≤ f a (a+1)) →
  a ∈ Set.Ioo 1 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_condition_l998_99873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l998_99854

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_properties 
  (A ω φ : ℝ) 
  (hA : A > 0) 
  (hω : ω > 0) 
  (hφ : |φ| ≤ π/2) 
  (h1 : f A ω φ (π/12) = 0)
  (h2 : f A ω φ (π/3) = 5)
  (h3 : ∀ x, x ∈ Set.Ioo ((π/12) - π/(2*ω)) ((π/12) + π/(2*ω)) → f A ω φ x ≤ 5) :
  (∀ x, f A ω φ x = 5 * Real.sin (2*x - π/6)) ∧
  (∀ k : ℤ, f A ω φ (k*π - π/6) = -5) ∧
  (∀ y ∈ Set.Icc (-5/2) 5, ∃ x ∈ Set.Icc 0 (π/2), f A ω φ x = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l998_99854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_return_l998_99877

theorem stock_price_return (initial_price : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) 
  (h1 : increase_percent = 30)
  (h2 : initial_price * (1 + increase_percent / 100) * (1 - decrease_percent / 100) = initial_price)
  : ∃ ε > 0, |decrease_percent - 23.08| < ε := by
  sorry

#check stock_price_return

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_return_l998_99877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polly_sandy_ratio_l998_99875

-- Define the marks for Polly, Sandy, and Willy
variable (P S W : ℝ)

-- Define the ratios given in the problem
def sandy_willy_ratio (p s w : ℝ) : Prop :=
  s / w = 5 / 2

def polly_willy_ratio (p s w : ℝ) : Prop :=
  p / w = 2 / 1

-- State the theorem to be proved
theorem polly_sandy_ratio (h1 : sandy_willy_ratio P S W) (h2 : polly_willy_ratio P S W) :
  P / S = 4 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polly_sandy_ratio_l998_99875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_velocity_specific_interval_l998_99860

/-- The position function of a particle -/
noncomputable def s (t : ℝ) : ℝ := t^2 + 3

/-- The average velocity of the particle in the interval (t₁, t₂) -/
noncomputable def avg_velocity (t₁ t₂ : ℝ) : ℝ := (s t₂ - s t₁) / (t₂ - t₁)

/-- Theorem: The average velocity of the particle in the interval (3, 3+Δt) is 6 + Δt -/
theorem avg_velocity_specific_interval (Δt : ℝ) (h : Δt ≠ 0) :
  avg_velocity 3 (3 + Δt) = 6 + Δt := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_velocity_specific_interval_l998_99860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relationship_l998_99827

/-- The line equation -/
def line (x y : ℝ) : Prop := 3 * x - 4 * y - 9 = 0

/-- The circle equation -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The center of the circle -/
def center : ℝ × ℝ := (0, 0)

/-- The radius of the circle -/
def radius : ℝ := 2

/-- The distance from a point to the line -/
noncomputable def distanceToLine (x y : ℝ) : ℝ :=
  |3 * x - 4 * y - 9| / Real.sqrt (3^2 + 4^2)

theorem line_circle_relationship :
  ∃ (x y : ℝ), line x y ∧ circleEq x y ∧
  distanceToLine (center.1) (center.2) < radius ∧
  ¬(line (center.1) (center.2)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relationship_l998_99827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_percentage_l998_99853

/-- Given a discount percentage and a gain percentage, calculates the ratio of cost price to marked price. -/
noncomputable def cost_price_ratio (discount : ℝ) (gain : ℝ) : ℝ :=
  (1 - discount / 100) / (1 + gain / 100)

/-- Theorem: Given a discount of 18% and a gain of 28.125%, the cost price is approximately 64% of the marked price. -/
theorem cost_price_percentage :
  let discount := (18 : ℝ)
  let gain := (28.125 : ℝ)
  abs (cost_price_ratio discount gain - 0.64) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_percentage_l998_99853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_pentagon_triangle_angles_eq_168_l998_99835

/-- The measure of an interior angle of a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ := 180 * (n - 2) / n

/-- The sum of the measure of an interior angle of a regular pentagon 
    and the measure of an interior angle of a regular triangle -/
noncomputable def sum_pentagon_triangle_angles : ℝ := interior_angle 5 + interior_angle 3

theorem sum_pentagon_triangle_angles_eq_168 : 
  sum_pentagon_triangle_angles = 168 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_pentagon_triangle_angles_eq_168_l998_99835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_area_for_R_l998_99861

-- Define the shapes
noncomputable def large_square_area : ℝ := 4^2
noncomputable def rectangle_area : ℝ := 2 * 4
noncomputable def small_square_area : ℝ := 2^2
noncomputable def circle_area : ℝ := Real.pi * 1^2

-- Define the theorem
theorem remaining_area_for_R :
  large_square_area - (rectangle_area + small_square_area + circle_area) = 4 - Real.pi := by
  -- Expand definitions
  unfold large_square_area rectangle_area small_square_area circle_area
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry

#check remaining_area_for_R

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_area_for_R_l998_99861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_is_one_number_of_zeros_l998_99851

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x + 1) + 2 * a * x - 4 * a * Real.exp x + 4

/-- Theorem stating the maximum value of f(x) when a = 1 -/
theorem max_value_when_a_is_one :
  ∃ (x_max : ℝ), ∀ (x : ℝ), f 1 x ≤ f 1 x_max ∧ f 1 x_max = 0 := by
  sorry

/-- Theorem stating the number of zeros of f(x) for different values of a -/
theorem number_of_zeros (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 1 → ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ∧
  (a = 1 → ∃! (x : ℝ), f a x = 0) ∧
  (a > 1 → ∀ (x : ℝ), f a x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_is_one_number_of_zeros_l998_99851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_quadratic_and_linear_l998_99844

theorem gcd_of_quadratic_and_linear (b : ℤ) (h : 2460 ∣ b) :
  Int.gcd (b^2 + 6*b + 30) (b + 5) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_quadratic_and_linear_l998_99844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_max_at_125_l998_99815

/-- B_k is defined as the binomial coefficient (500 choose k) multiplied by 0.3^k -/
def B (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.3 ^ k)

/-- The statement that B_k is largest when k = 125 -/
theorem B_max_at_125 : ∀ k : ℕ, k ≤ 500 → B 125 ≥ B k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_max_at_125_l998_99815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l998_99849

theorem min_abs_difference (x y : ℕ+) (h : x.val * y.val - 4 * x.val + 3 * y.val = 215) :
  ∃ (a b : ℕ+), a.val * b.val - 4 * a.val + 3 * b.val = 215 ∧
  ∀ (c d : ℕ+), c.val * d.val - 4 * c.val + 3 * d.val = 215 →
  |Int.ofNat a.val - Int.ofNat b.val| ≤ |Int.ofNat c.val - Int.ofNat d.val| ∧
  |Int.ofNat a.val - Int.ofNat b.val| = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l998_99849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l998_99893

variable {α : Type*} [LinearOrderedField α]

def is_constant_sequence (a : ℕ → α) : Prop :=
  ∀ n m : ℕ, a n = a m

def is_geometric_progression (a : ℕ → α) : Prop :=
  ∃ r : α, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

def is_arithmetic_progression (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

theorem sequence_properties (k b : α) (a : ℕ → α) 
    (hk : k ≠ 0) (hb : b ≠ 0) :
  (is_geometric_progression a ∧ is_geometric_progression (fun n ↦ k * a n + b)) ∨
  (is_arithmetic_progression a ∧ is_geometric_progression (fun n ↦ k * a n + b)) ∨
  (is_geometric_progression a ∧ is_arithmetic_progression (fun n ↦ k * a n + b))
  → is_constant_sequence a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l998_99893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_neg_eight_and_neg_seven_l998_99828

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def problem_conditions (x y : ℝ) : Prop :=
  y = 3 * (floor x) + 4 ∧
  y = 2 * (floor (x - 3)) + 7 ∧
  ¬(∃ n : ℤ, x = n)

theorem x_plus_y_between_neg_eight_and_neg_seven (x y : ℝ) 
  (h : problem_conditions x y) : 
  -8 < x + y ∧ x + y < -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_neg_eight_and_neg_seven_l998_99828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_food_consumption_l998_99887

/-- Calculates the daily food consumption in pounds for two dogs given specific feeding conditions. -/
theorem dog_food_consumption
  (cups_per_meal : ℝ)
  (meals_per_day : ℕ)
  (num_dogs : ℕ)
  (cups_per_pound : ℝ)
  (h1 : cups_per_meal = 1.5)
  (h2 : meals_per_day = 3)
  (h3 : num_dogs = 2)
  (h4 : cups_per_pound = 2.25) :
  (cups_per_meal * meals_per_day * num_dogs) / cups_per_pound = 4 := by
  sorry

#check dog_food_consumption

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_food_consumption_l998_99887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_rate_calculation_l998_99832

/-- The rate per kg for grapes -/
def grape_rate : ℕ → ℕ := sorry

/-- The quantity of grapes purchased in kg -/
def grape_quantity : ℕ := 9

/-- The quantity of mangoes purchased in kg -/
def mango_quantity : ℕ := 9

/-- The rate per kg for mangoes -/
def mango_rate : ℕ := 55

/-- The total amount paid -/
def total_paid : ℕ := 1125

theorem grape_rate_calculation :
  grape_rate grape_quantity = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_rate_calculation_l998_99832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_jay_and_paul_l998_99881

-- Define the speeds and time
noncomputable def jay_speed : ℝ := 1 / 20
noncomputable def paul_speed : ℝ := 3 / 40
noncomputable def time : ℝ := 2 * 60

-- Define the distances traveled
noncomputable def jay_distance : ℝ := jay_speed * time
noncomputable def paul_distance : ℝ := paul_speed * time

-- Theorem statement
theorem distance_between_jay_and_paul : 
  Real.sqrt (jay_distance^2 + paul_distance^2) = 3 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_jay_and_paul_l998_99881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_l998_99804

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 1 / (2^x + 1)

theorem odd_function_condition (a : ℝ) : 
  (∀ x : ℝ, f a (-x) = -(f a x)) ↔ a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_l998_99804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_value_area_value_l998_99862

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c ∧
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C

-- Define the area formula
noncomputable def area_formula (t : Triangle) : ℝ :=
  (t.a^2 + t.b^2 - t.c^2) / 4

-- Theorem 1
theorem sin_B_value (t : Triangle) 
  (h1 : triangle_conditions t)
  (h2 : area_formula t = t.a * t.b * Real.sin t.C / 2)
  (h3 : Real.sin t.A = 3/5) :
  Real.sin t.B = 7 * Real.sqrt 2 / 10 :=
by sorry

-- Theorem 2
theorem area_value (t : Triangle)
  (h1 : triangle_conditions t)
  (h2 : t.c = 5)
  (h3 : Real.sin t.A = 3/5)
  (h4 : Real.sin t.B = 7 * Real.sqrt 2 / 10) :
  area_formula t = 21/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_value_area_value_l998_99862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_monotonically_increasing_interval_range_on_interval_l998_99855

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.sin (x + Real.pi/6)

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 → (∀ x : ℝ, f (x + S) = f x) → T ≤ S :=
sorry

-- Theorem for the monotonically increasing interval
theorem monotonically_increasing_interval :
  ∀ k : ℤ, ∀ x y : ℝ, -Real.pi/12 + k * Real.pi ≤ x ∧ x < y ∧ y ≤ 5*Real.pi/12 + k * Real.pi → f x < f y :=
sorry

-- Theorem for the range when x ∈ [0, π/2]
theorem range_on_interval :
  ∀ y : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi/2 ∧ f x = y) ↔ 0 ≤ y ∧ y ≤ 1 + Real.sqrt 3 / 2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_monotonically_increasing_interval_range_on_interval_l998_99855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_inequality_l998_99831

theorem parallelepiped_inequality {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (a b c : V) :
  4 * (norm a + norm b + norm c) ≤ 
  2 * (norm (a + b + c) + norm (a - b + c) + norm (-a + b + c) + norm (a + b - c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_inequality_l998_99831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l998_99847

noncomputable def f (x : ℝ) := Real.cos (2 * x)

noncomputable def g (x : ℝ) := f (x - Real.pi / 4)

theorem g_properties :
  (∀ x, g x = Real.sin (2 * x)) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi / 4 → g x₁ < g x₂) ∧
  (∀ x, g (-x) = -g x) := by
  sorry

#check g_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l998_99847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_in_upper_half_probability_l998_99816

noncomputable section

-- Define the rectangle
def rectangle_length : ℝ := 10
def rectangle_width : ℝ := 6

-- Define the circle
def circle_radius : ℝ := 1

-- Define the valid range for the circle's center y-coordinate
def valid_y_range : ℝ := 1

-- Define the total vertical range in the upper half of the rectangle
def total_upper_half_range : ℝ := rectangle_width / 2 - circle_radius

-- Theorem statement
theorem circle_in_upper_half_probability :
  valid_y_range / total_upper_half_range = 1 / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_in_upper_half_probability_l998_99816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_equals_one_l998_99857

noncomputable section

/-- Given two curves f(x) and g(x) with a common tangent line at point P(s,t), prove that a = 1 -/
theorem common_tangent_implies_a_equals_one (e : ℝ) (a : ℝ) (s t : ℝ) :
  a > 0 →
  (∀ x, f x = (1 / (2 * Real.exp 1)) * x^2) →
  (∀ x, g x = a * Real.log x) →
  (∃ k : ℝ, (deriv f s = k ∧ deriv g s = k) ∧ f s = t ∧ g s = t) →
  a = 1 :=
by sorry

where
  f (x : ℝ) : ℝ := (1 / (2 * Real.exp 1)) * x^2
  g (x : ℝ) : ℝ := a * Real.log x

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_equals_one_l998_99857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_digit_integers_above_532_l998_99836

/-- A function that counts the number of 3-digit integers with distinct digits greater than a given number -/
def countDistinctDigitIntegers (n : Nat) : Nat :=
  (Finset.filter (fun x => x > n ∧ x < 1000 ∧ (Nat.digits 10 x).card = 3) (Finset.range 1000)).card

/-- The theorem stating that there are 216 3-digit integers with distinct digits greater than 532 -/
theorem distinct_digit_integers_above_532 : countDistinctDigitIntegers 532 = 216 := by
  sorry

#eval countDistinctDigitIntegers 532  -- Add this line to check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_digit_integers_above_532_l998_99836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_k_l998_99864

/-- Arithmetic sequence with first term 1 and common difference d -/
def arithmetic_seq (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => arithmetic_seq d n + d

/-- Geometric sequence with first term 1 and common ratio r -/
def geometric_seq (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => geometric_seq r n * r

/-- Sum of arithmetic and geometric sequences -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_seq d n + geometric_seq r n

/-- The main theorem -/
theorem find_c_k (d r k : ℕ) (h1 : c_seq d r (k - 1) = 80) (h2 : c_seq d r (k + 1) = 500) :
  c_seq d r k = 167 :=
by
  sorry

#check find_c_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_k_l998_99864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l998_99868

/-- Predicate for a set being an ellipse with given foci -/
def IsEllipse (E : Set (ℝ × ℝ)) (F₁ F₂ : ℝ × ℝ) : Prop :=
  ∃ (a : ℝ), a > 0 ∧ ∀ p ∈ E, dist p F₁ + dist p F₂ = 2 * a

/-- Predicate for a line being tangent to a set -/
def IsTangentLine (E : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) : Prop :=
  (E ∩ L).Nonempty ∧ ∀ p ∈ E ∩ L, ∃ ε > 0, ∀ q ∈ E, q ≠ p → dist q p > ε

/-- The length of the major axis of an ellipse -/
noncomputable def MajorAxisLength (E : Set (ℝ × ℝ)) : ℝ := sorry

/-- The length of the major axis of an ellipse with given foci and tangent line -/
theorem ellipse_major_axis_length : 
  ∀ (F₁ F₂ : ℝ × ℝ) (y₀ : ℝ),
  F₁ = (5, 8) →
  F₂ = (25, 28) →
  y₀ = 1 →
  ∃ (E : Set (ℝ × ℝ)),
    IsEllipse E F₁ F₂ ∧ 
    IsTangentLine E {p : ℝ × ℝ | p.2 = y₀} →
    MajorAxisLength E = 2 * Real.sqrt 389 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l998_99868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_is_36_l998_99897

noncomputable def a : Fin 2 → ℝ := ![1, -1]
noncomputable def b : Fin 2 → ℝ := ![1, 1]
noncomputable def c (α : ℝ) : Fin 2 → ℝ := ![Real.sqrt 2 * Real.cos α, Real.sqrt 2 * Real.sin α]

theorem max_value_is_36 (m n α : ℝ) :
  (∀ i : Fin 2, m * a i + n * b i = 2 * c α i) →
  (∃ m₀ n₀ α₀ : ℝ, (m₀ - 4)^2 + n₀^2 = 36 ∧
    ∀ m' n' α' : ℝ, (∀ i : Fin 2, m' * a i + n' * b i = 2 * c α' i) →
      (m' - 4)^2 + n'^2 ≤ 36) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_is_36_l998_99897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_propositions_l998_99822

-- Define what a proposition is
def is_proposition (s : String) : Prop := 
  (∃ (b : Bool), s = if b then "true" else "false") ∧ ¬(s.contains '?')

-- Define the statements
def statement1 : String := "x^2 - 3 = 0"
def statement2 : String := "Are two lines that intersect with a line parallel?"
def statement3 : String := "3 + 1 = 5"
def statement4 : String := "5x - 3 > 6"

-- Theorem to prove
theorem not_propositions : 
  ¬(is_proposition statement1) ∧ 
  ¬(is_proposition statement2) ∧ 
  ¬(is_proposition statement4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_propositions_l998_99822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_is_36_l998_99886

/-- Represents a number in base r as a list of digits -/
def BaseRepresentation (r : ℕ) := List ℕ

/-- Checks if a BaseRepresentation is a palindrome -/
def isPalindrome {r : ℕ} (digits : BaseRepresentation r) : Prop :=
  digits = digits.reverse

/-- Represents the condition that x is of the form ppqq in base r where 2q = 5p -/
def hasPPQQForm (x : ℕ) (r : ℕ) : Prop :=
  ∃ (p q : ℕ), x = p * r^3 + p * r^2 + q * r + q ∧ 2 * q = 5 * p

/-- Converts a number to its base r representation -/
noncomputable def toBaseR (x : ℕ) (r : ℕ) : BaseRepresentation r :=
  sorry -- implementation details omitted

/-- Sums the digits in a base r representation -/
def sumDigits {r : ℕ} (digits : BaseRepresentation r) : ℕ :=
  digits.sum

theorem sum_of_digits_is_36
  (x r : ℕ)
  (h_r : r ≤ 36)
  (h_ppqq : hasPPQQForm x r)
  (h_palindrome : isPalindrome (toBaseR (x^2) r))
  (h_seven_digits : (toBaseR (x^2) r).length = 7)
  (h_middle_zero : (toBaseR (x^2) r).get? 3 = some 0) :
  sumDigits (toBaseR (x^2) r) = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_is_36_l998_99886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l998_99882

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.sin x + Real.cos x

theorem function_properties (m : ℝ) :
  f m (π / 2) = 1 →
  f 1 (π / 12) = Real.sqrt 2 * Real.sin (π / 3) →
  (∃ (A B C : ℝ), 
    A > 0 ∧ A < π / 2 ∧
    1 / 2 * 2 * C * Real.sin A = 3 * Real.sqrt 3 / 2 ∧
    f 1 (π / 12) = Real.sqrt 2 * Real.sin A) →
  (m = 1 ∧
   (∀ x : ℝ, f 1 x = Real.sqrt 2 * Real.sin (x + π / 4)) ∧
   (∀ T : ℝ, T > 0 ∧ (∀ x : ℝ, f 1 (x + T) = f 1 x) → T ≥ 2 * π) ∧
   (∃ x : ℝ, f 1 x = Real.sqrt 2) ∧
   (∃ x : ℝ, f 1 x = -Real.sqrt 2) ∧
   (∀ x y : ℝ, f 1 x ≤ Real.sqrt 2 ∧ f 1 y ≥ -Real.sqrt 2) ∧
   (∃ (A B C : ℝ),
     A = π / 3 ∧
     C = 3 ∧
     B = Real.sqrt 7)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l998_99882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l998_99801

/-- Represents the distance to Kirov at a given time -/
structure DistanceAtTime where
  time : ℕ  -- Time in minutes past 12:00
  distance : ℝ  -- Actual distance in km

/-- Rounds a real number to the nearest integer -/
noncomputable def my_round (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem train_speed_proof 
  (d1 : DistanceAtTime) 
  (d2 : DistanceAtTime) 
  (d3 : DistanceAtTime)
  (h1 : d1.time = 0 ∧ my_round d1.distance = 73)
  (h2 : d2.time = 15 ∧ my_round d2.distance = 62)
  (h3 : d3.time = 45 ∧ my_round d3.distance = 37)
  (h_constant_speed : ∃ (v : ℝ), ∀ (t : ℕ), 
    (∃ (d : DistanceAtTime), d.time = t → d.distance = d1.distance - v * (t / 60))) :
  ∃! (v : ℝ), v = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l998_99801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_inequality_l998_99834

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := cos x

-- Define a, b, and c
noncomputable def a : ℝ := f (log 2)
noncomputable def b : ℝ := f (log π)
noncomputable def c : ℝ := f (log (1/3))

-- State the theorem
theorem cos_inequality : a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_inequality_l998_99834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l998_99837

def set_A : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 + 3*x + 4)}
def set_B : Set ℝ := {x | (2 : ℝ)^x > 4}

theorem union_of_A_and_B : set_A ∪ set_B = Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l998_99837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_min_value_l998_99839

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

-- Define the interval
def interval : Set ℝ := { x | -Real.pi/6 ≤ x ∧ x ≤ Real.pi/4 }

-- Theorem for the smallest positive period
theorem smallest_positive_period : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ 
  T = Real.pi :=
sorry

-- Theorem for the maximum value
theorem max_value : 
  ∃ (x : ℝ), x ∈ interval ∧ f x = 2 ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ 2 :=
sorry

-- Theorem for the minimum value
theorem min_value : 
  ∃ (x : ℝ), x ∈ interval ∧ f x = -1 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_min_value_l998_99839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l998_99867

open Real

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem vector_sum_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : vector_magnitude a = 1)
  (h2 : ∀ x : ℝ, vector_magnitude (a.1 - x * b.1, a.2 - x * b.2) ≥ Real.sqrt 3 / 2)
  (h3 : ∀ x : ℝ, vector_magnitude (a.1 - x * b.1, a.2 - x * b.2) = Real.sqrt 3 / 2 → 
    ∃ y : ℝ, vector_magnitude (a.1 - y * b.1, a.2 - y * b.2) < vector_magnitude (a.1 - x * b.1, a.2 - x * b.2))
  (h4 : ∀ y : ℝ, vector_magnitude (b.1 - y * a.1, b.2 - y * a.2) ≥ Real.sqrt 3)
  (h5 : ∀ y : ℝ, vector_magnitude (b.1 - y * a.1, b.2 - y * a.2) = Real.sqrt 3 → 
    ∃ x : ℝ, vector_magnitude (b.1 - x * a.1, b.2 - x * a.2) < vector_magnitude (b.1 - y * a.1, b.2 - y * a.2)) :
  vector_magnitude (a.1 + b.1, a.2 + b.2) = Real.sqrt 7 ∨ vector_magnitude (a.1 + b.1, a.2 + b.2) = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l998_99867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canal_bottom_width_approx_l998_99823

/-- Represents a trapezoidal canal cross-section -/
structure TrapezoidalCanal where
  topWidth : ℝ
  bottomWidth : ℝ
  depth : ℝ
  area : ℝ

/-- Calculates the area of a trapezoidal canal cross-section -/
noncomputable def calculateArea (canal : TrapezoidalCanal) : ℝ :=
  (canal.topWidth + canal.bottomWidth) * canal.depth / 2

/-- Theorem: Given the specified dimensions, the bottom width of the canal is approximately 74.02 m -/
theorem canal_bottom_width_approx (canal : TrapezoidalCanal) 
    (h1 : canal.topWidth = 6)
    (h2 : canal.depth = 257.25)
    (h3 : canal.area = 10290)
    (h4 : calculateArea canal = canal.area) :
    ∃ ε > 0, |canal.bottomWidth - 74.02| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_canal_bottom_width_approx_l998_99823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_subsets_disjoint_subsets_congruent_subsets_partition_natural_numbers_l998_99830

/-- A function that partitions natural numbers into infinite congruent subsets -/
noncomputable def partition_function : ℕ → ℕ := sorry

/-- Each subset defined by the partition function is infinite -/
theorem infinite_subsets (n : ℕ) : Set.Infinite {x : ℕ | partition_function x = n} := by
  sorry

/-- The subsets defined by the partition function are disjoint -/
theorem disjoint_subsets (m n : ℕ) (h : m ≠ n) :
  {x : ℕ | partition_function x = m} ∩ {x : ℕ | partition_function x = n} = ∅ := by
  sorry

/-- The subsets defined by the partition function are congruent -/
theorem congruent_subsets (n : ℕ) :
  ∃ k : ℤ, ∀ x : ℕ, partition_function (x + k.natAbs) = partition_function x + n := by
  sorry

/-- The main theorem: it's possible to partition ℕ into infinitely many infinite congruent subsets -/
theorem partition_natural_numbers :
  ∃ f : ℕ → ℕ, 
    (∀ n, Set.Infinite {x : ℕ | f x = n}) ∧ 
    (∀ m n, m ≠ n → {x : ℕ | f x = m} ∩ {x : ℕ | f x = n} = ∅) ∧
    (∀ n, ∃ k : ℤ, ∀ x : ℕ, f (x + k.natAbs) = f x + n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_subsets_disjoint_subsets_congruent_subsets_partition_natural_numbers_l998_99830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_value_l998_99821

theorem cos_2theta_value (θ : ℝ) (h : Real.cos θ + Real.sin θ = 3/2) : Real.cos (2*θ) = 7/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_value_l998_99821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_negative_one_l998_99846

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (-2 * x) + 3 * x

-- State the theorem
theorem f_derivative_at_negative_one :
  deriv f (-1) = 2 := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_negative_one_l998_99846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_curve_l998_99880

-- Define the curve (x-1)^2 + y^2 = 1
def on_curve (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_to_curve :
  ∃ (min_dist : ℝ), min_dist = 1 - Real.sqrt 3 ∧
  ∀ (x y qx qy : ℝ), on_curve qx qy →
    distance x y qx qy ≥ min_dist := by
  sorry

#check min_distance_to_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_curve_l998_99880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insect_growth_time_l998_99826

/-- An insect that doubles in size every day -/
structure Insect where
  growth : ℕ → ℝ
  growth_doubles : ∀ n, growth (n + 1) = 2 * growth n

theorem insect_growth_time (i : Insect) (h : i.growth 10 = 10) :
  i.growth 8 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insect_growth_time_l998_99826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_geometric_series_equality_l998_99805

/-- Given real numbers a and b satisfying an infinite alternating geometric series equation,
    prove that another related infinite alternating geometric series equals 6/7. -/
theorem alternating_geometric_series_equality (a b : ℝ) 
    (h : ∑' k, (-1)^k * a / b^(k+1) = 6) :
    ∑' k, (-1)^k * a / (a - b)^(k+1) = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_geometric_series_equality_l998_99805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_complex_power_diff_one_is_rational_l998_99878

/-- A complex number with rational real and imaginary parts -/
structure RationalComplex where
  re : ℚ
  im : ℚ

/-- The absolute value (modulus) of a RationalComplex number -/
noncomputable def rationalAbs (z : RationalComplex) : ℚ :=
  (z.re ^ 2 + z.im ^ 2).sqrt

/-- The nth power of a RationalComplex number -/
def pow (z : RationalComplex) (n : ℤ) : RationalComplex :=
  sorry

/-- The difference between two RationalComplex numbers -/
def sub (z w : RationalComplex) : RationalComplex :=
  ⟨z.re - w.re, z.im - w.im⟩

/-- The absolute value of the difference between two RationalComplex numbers -/
noncomputable def rationalAbsSub (z w : RationalComplex) : ℚ :=
  rationalAbs (sub z w)

/-- The main theorem -/
theorem rational_complex_power_diff_one_is_rational 
  (z : RationalComplex) (n : ℤ) 
  (h : rationalAbs z = 1) : 
  ∃ (q : ℚ), rationalAbsSub (pow z (2 * n)) ⟨1, 0⟩ = q :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_complex_power_diff_one_is_rational_l998_99878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_position_l998_99888

/-- Definition of the complex number Z in terms of real number m -/
def Z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)

/-- Theorem stating the conditions for Z to lie on the real axis, in the first quadrant, or on the line x + y + 5 = 0 -/
theorem Z_position (m : ℝ) :
  ((Z m).im = 0 ↔ m = -3 ∨ m = 5) ∧
  ((Z m).re > 0 ∧ (Z m).im > 0 ↔ m > 5) ∧
  ((Z m).re + (Z m).im + 5 = 0 ↔ m = (-5 + Real.sqrt 41) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_position_l998_99888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l998_99841

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the relation between points O, A, B, and M
def point_relation (x1 y1 x2 y2 x y θ : ℝ) : Prop :=
  x = x1 * Real.cos θ + x2 * Real.sin θ ∧ y = y1 * Real.cos θ + y2 * Real.sin θ

theorem ellipse_properties (x1 y1 x2 y2 x y θ : ℝ) 
  (h1 : is_on_ellipse x1 y1)
  (h2 : is_on_ellipse x2 y2)
  (h3 : is_on_ellipse x y)
  (h4 : point_relation x1 y1 x2 y2 x y θ)
  (h5 : x1 ≠ 0 ∧ x2 ≠ 0)
  (h6 : θ ≠ 0 ∧ θ ≠ π/2) :
  (y1 / x1) * (y2 / x2) = -1/2 ∧ x1^2 + y1^2 + x2^2 + y2^2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l998_99841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l998_99858

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The focal distance of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- The sum of distances from any point on the ellipse to the two foci -/
def sum_of_distances (e : Ellipse) : ℝ := 2 * e.a

/-- Theorem: If the distances from any point on the ellipse to the two foci and the focal distance
    form an arithmetic sequence, then the eccentricity of the ellipse is 1/2 -/
theorem ellipse_eccentricity_half (e : Ellipse) 
  (h_arithmetic : ∃ (d₁ d₂ : ℝ), d₁ + d₂ = sum_of_distances e ∧ 
    2 * focal_distance e = (d₁ + d₂) / 2) : 
  eccentricity e = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l998_99858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_k_range_l998_99871

-- Define the hyperbola equation
def hyperbola (x y k : ℝ) : Prop := x^2 / 4 + y^2 / k = 1

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity (k : ℝ) : ℝ := Real.sqrt (4 - k) / 2

-- Theorem statement
theorem hyperbola_k_range (k : ℝ) :
  (∀ x y, hyperbola x y k) →
  (1 < eccentricity k ∧ eccentricity k < 2) →
  -12 < k ∧ k < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_k_range_l998_99871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_regression_properties_l998_99870

/-- Linear regression model for worker's wage based on labor productivity -/
structure WageModel where
  /-- Intercept of the regression line -/
  intercept : ℝ
  /-- Slope of the regression line -/
  slope : ℝ

/-- Calculate the wage for a given labor productivity -/
def wage_function (model : WageModel) (x : ℝ) : ℝ :=
  model.intercept + model.slope * x

theorem wage_regression_properties (model : WageModel)
  (h_intercept : model.intercept = 50)
  (h_slope : model.slope = 80) :
  let f := wage_function model
  (f 1 = 130 ∧
   model.slope > 0 ∧
   (∀ x : ℝ, f (x + 1) - f x = 80) ∧
   f 2 = 210) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_regression_properties_l998_99870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_diagonal_l998_99814

/-- A quadrilateral with specific side lengths and an integer diagonal -/
structure SpecialQuadrilateral where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  ef_length : Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) = 6
  fg_length : Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2) = 19
  gh_length : Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2) = 6
  he_length : Real.sqrt ((H.1 - E.1)^2 + (H.2 - E.2)^2) = 10
  eg_integer : ∃ n : ℤ, Real.sqrt ((E.1 - G.1)^2 + (E.2 - G.2)^2) = n

/-- The diagonal EG in the special quadrilateral is 15 -/
theorem special_quadrilateral_diagonal (q : SpecialQuadrilateral) : 
  Real.sqrt ((q.E.1 - q.G.1)^2 + (q.E.2 - q.G.2)^2) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_diagonal_l998_99814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_black_edges_l998_99825

/-- A cube is a structure with edges that can be colored either red or black. -/
structure Cube where
  edges : Fin 12 → Bool
  -- True represents a black edge, False represents a red edge

/-- A face of a cube is represented by four edges. -/
def Face := Fin 4 → Fin 12

/-- The cube has exactly 6 faces. -/
def cube_faces : Fin 6 → Face := sorry

/-- Counts the number of black edges in a face. -/
def count_black_edges (c : Cube) (f : Face) : Nat :=
  (List.range 4).filter (λ i => c.edges (f i)) |>.length

/-- Counts the total number of black edges in a cube. -/
def total_black_edges (c : Cube) : Nat :=
  (List.range 12).filter (λ i => c.edges i) |>.length

/-- A cube configuration is valid if every face has exactly two black edges. -/
def is_valid_cube (c : Cube) : Prop :=
  ∀ f : Fin 6, count_black_edges c (cube_faces f) = 2

theorem min_black_edges :
  ∃ (c : Cube), is_valid_cube c ∧ total_black_edges c = 8 ∧
  (∀ (c' : Cube), is_valid_cube c' → total_black_edges c' ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_black_edges_l998_99825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_35_5_l998_99842

/-- The area of a triangle given by three points in a 2D coordinate system -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Theorem: The area of triangle ABC with given coordinates is 35.5 square units -/
theorem triangle_area_35_5 :
  triangleArea (-6, 2) (1, 7) (4, -3) = 35.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_35_5_l998_99842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l998_99848

/-- The shortest chord length passing through a point inside a circle --/
theorem shortest_chord_length (center : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) 
    (h_center : center = (1, 0))
    (h_radius : radius = 5)
    (h_p : p = (2, -1))
    (h_inside : (p.1 - center.1)^2 + (p.2 - center.2)^2 < radius^2) :
  2 * Real.sqrt 23 = 2 * Real.sqrt (radius^2 - ((p.1 - center.1)^2 + (p.2 - center.2)^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l998_99848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_surd_l998_99802

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

-- Define a function to check if a number is simplest quadratic surd
def isSimplestQuadraticSurd (n : ℚ) : Prop :=
  n > 0 ∧ ¬(isPerfectSquare (n.num * n.den)) ∧
  ∀ m : ℤ, m > 1 → ¬(isPerfectSquare (m * m * n.num * n.den))

-- Theorem statement
theorem simplest_quadratic_surd :
  isSimplestQuadraticSurd 7 ∧
  ¬(isSimplestQuadraticSurd 9) ∧
  ¬(isSimplestQuadraticSurd 20) ∧
  ¬(isSimplestQuadraticSurd (1/3)) :=
by
  sorry

#check simplest_quadratic_surd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_surd_l998_99802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l998_99869

/-- The time (in hours) it takes to fill the cistern without a leak -/
noncomputable def fill_time : ℝ := 4

/-- The time (in hours) it takes for the leak to empty a full cistern -/
noncomputable def empty_time : ℝ := 20 / 3

/-- The additional time needed to fill the cistern due to the leak -/
noncomputable def additional_time : ℝ := 6

theorem cistern_fill_time (fill_time empty_time additional_time : ℝ) :
  fill_time = 4 →
  empty_time = 20 / 3 →
  additional_time = (1 / ((1 / fill_time) - (1 / empty_time))) - fill_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l998_99869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l998_99810

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ → ℝ
  | 0, _ => 0  -- Define for n = 0 to avoid missing case
  | 1, θ => 2 * Real.cos θ
  | (n + 2), θ => Real.sqrt (2 + a (n + 1) θ)

-- State the theorem
theorem a_formula {θ : ℝ} (h : 0 < θ ∧ θ < Real.pi / 2) :
  ∀ n : ℕ, n ≥ 1 → a n θ = 2 * Real.cos (θ / 2^(n - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l998_99810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_specific_cone_l998_99817

/-- The radius of a sphere tangent to a truncated cone -/
noncomputable def sphere_radius (r₁ r₂ h : ℝ) : ℝ :=
  (Real.sqrt (h^2 + (r₁ - r₂)^2)) / 2

/-- Theorem: The radius of a sphere tangent to a truncated cone with specific dimensions -/
theorem sphere_radius_specific_cone : 
  sphere_radius 20 6 15 = Real.sqrt 421 / 2 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval sphere_radius 20 6 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_specific_cone_l998_99817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_fare_is_point_eight_l998_99892

/-- Represents the fare structure for a taxi ride -/
structure TaxiFare where
  initialFare : ℚ  -- Initial fare for the first 1/5 mile
  totalDistance : ℚ  -- Total distance of the ride in miles
  totalFare : ℚ  -- Total fare for the entire ride

/-- Calculates the fare for each 1/5 mile after the first 1/5 mile -/
def additionalFarePerFifthMile (tf : TaxiFare) : ℚ :=
  let totalFifthMiles := tf.totalDistance * 5
  let additionalFifthMiles := totalFifthMiles - 1
  let additionalFare := tf.totalFare - tf.initialFare
  additionalFare / additionalFifthMiles

/-- Theorem stating that given the specific conditions, the additional fare per 1/5 mile is $0.8 -/
theorem additional_fare_is_point_eight (tf : TaxiFare)
  (h1 : tf.initialFare = 8)
  (h2 : tf.totalDistance = 8)
  (h3 : tf.totalFare = 39.2) :
  additionalFarePerFifthMile tf = 4/5 := by
  sorry

#eval additionalFarePerFifthMile { initialFare := 8, totalDistance := 8, totalFare := 39.2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_fare_is_point_eight_l998_99892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_three_l998_99813

theorem three_digit_divisible_by_three :
  ∃! n : ℕ, n = (Finset.filter (fun C => (100 + 10 * C + 3) % 3 = 0) (Finset.range 10)).card ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_three_l998_99813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_bound_l998_99865

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := -1 / x
noncomputable def g (x : ℝ) : ℝ := Real.log x

-- Define the intersection point x₁
noncomputable def x₁ : ℝ := sorry

-- State the theorem
theorem smallest_integer_bound (h1 : x₁ < 0) (h2 : f x₁ = g x₁) (h3 : ∀ x < 0, f x = g x → x = x₁) :
  ∃ m : ℤ, m = -2 ∧ (∀ n : ℤ, n ≥ ⌈x₁⌉ → n ≥ m) ∧ m ≥ ⌈x₁⌉ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_bound_l998_99865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_l998_99812

/-- The parabola C₁ with equation y² = 8x -/
def C₁ (x y : ℝ) : Prop := y^2 = 8*x

/-- The circle C₂ with equation x² + y² = r² -/
def C₂ (x y r : ℝ) : Prop := x^2 + y^2 = r^2

/-- The focus of the parabola C₁ -/
def focus : ℝ × ℝ := (2, 0)

/-- The line with slope 1 passing through the focus -/
def tangent_line (x y : ℝ) : Prop := y = x - 2

theorem parabola_circle_tangent (r : ℝ) :
  (∃ x y : ℝ, C₁ x y ∧ C₂ x y r ∧ tangent_line x y) → r = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_l998_99812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y1_less_than_y2_l998_99840

/-- A linear function of the form y = x + b -/
def linear_function (b : ℝ) : ℝ → ℝ := λ x ↦ x + b

/-- Theorem: For a linear function y = x + b, if points A(2, y₁) and B(5, y₂) lie on its graph, then y₁ < y₂ -/
theorem y1_less_than_y2 (b : ℝ) (y₁ y₂ : ℝ) 
  (h1 : linear_function b 2 = y₁)
  (h2 : linear_function b 5 = y₂) : 
  y₁ < y₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y1_less_than_y2_l998_99840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_min_on_interval_l998_99896

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + 1)

-- Theorem for the decreasing property of f
theorem f_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ := by
  sorry

-- Theorem for the minimum value of f on [1, 5]
theorem f_min_on_interval : ∃ x₀ ∈ Set.Icc 1 5, ∀ x ∈ Set.Icc 1 5, f x₀ ≤ f x ∧ f x₀ = 1/33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_min_on_interval_l998_99896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l998_99859

theorem sufficient_not_necessary (θ : ℝ) :
  (|θ - π/12| < π/12 → Real.sin θ < 1/2) ∧
  ¬(Real.sin θ < 1/2 → |θ - π/12| < π/12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l998_99859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_2x_l998_99895

theorem definite_integral_exp_plus_2x : 
  ∫ x in (Set.Icc 0 1), (Real.exp x + 2 * x) = Real.exp 1 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_2x_l998_99895
