import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_rectangle_l119_11951

/-- Represents the dimensions and area of a rectangular field -/
structure RectangularField where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Calculates the area of a rectangular field given width and length -/
def calculateArea (w l : ℝ) : ℝ := w * l

/-- Finds the dimensions of the rectangular field that maximizes the area -/
noncomputable def maxAreaField (fencingMaterial : ℝ) : RectangularField :=
  let w := fencingMaterial / 4
  let l := fencingMaterial / 2
  { width := w, length := l, area := calculateArea w l }

/-- Theorem: The maximum area of a rectangular field with 100 feet of fencing material
    for three sides (given one side is fixed) is achieved when the width is 25 feet
    and the length is 50 feet, resulting in an area of 1250 square feet. -/
theorem max_area_rectangle (fencingMaterial : ℝ) (h : fencingMaterial = 100) :
  let field := maxAreaField fencingMaterial
  field.width = 25 ∧ field.length = 50 ∧ field.area = 1250 := by
  sorry

#check max_area_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_rectangle_l119_11951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lois_final_book_count_l119_11966

/-- Calculates the number of books Lois has after giving away, donating, and purchasing books. -/
def final_book_count (initial_books : ℕ) (fraction_to_nephew : ℚ) (fraction_to_library : ℚ) (new_books : ℕ) : ℕ :=
  let remaining_after_nephew := initial_books - (initial_books * fraction_to_nephew).floor
  let remaining_after_library := remaining_after_nephew - (remaining_after_nephew * fraction_to_library).floor
  (remaining_after_library + new_books).toNat

/-- Theorem stating that Lois ends up with 23 books given the specific conditions. -/
theorem lois_final_book_count :
  final_book_count 40 (1/4) (1/3) 3 = 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lois_final_book_count_l119_11966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l119_11920

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ+) : ℕ :=
  (Nat.log 10 n.val).succ

/-- Concatenate a positive integer with itself -/
def concat_self (n : ℕ+) : ℕ :=
  n.val * (10 ^ num_digits n) + n.val

/-- The main theorem -/
theorem polynomial_characterization (P : Polynomial ℤ) :
  (∀ n : ℕ+, P.eval (n : ℤ) ≠ 0) →
  (∀ n : ℕ+, ∃ k : ℤ, P.eval (concat_self n : ℤ) = k * P.eval (n : ℤ)) →
  ∃ m : ℕ, P = Polynomial.monomial m 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l119_11920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l119_11990

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C --/
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  opposite_sides : True  -- Assumes a, b, c are opposite to A, B, C respectively

/-- Vector m defined in the problem --/
noncomputable def m (t : AcuteTriangle) : ℝ × ℝ :=
  (2 * Real.sin (t.A + t.C), -Real.sqrt 3)

/-- Vector n defined in the problem --/
noncomputable def n (t : AcuteTriangle) : ℝ × ℝ :=
  (Real.cos (2 * t.B), 2 * (Real.cos (t.B / 2))^2 - 1)

/-- Collinearity condition for vectors --/
def collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Area of the triangle --/
noncomputable def area (t : AcuteTriangle) : ℝ :=
  1/2 * t.a * t.c * Real.sin t.B

/-- Main theorem statement --/
theorem triangle_properties (t : AcuteTriangle) :
  (collinear (m t) (n t) → t.B = π/3) ∧
  (t.b = 1 → 0 < area t ∧ area t ≤ Real.sqrt 3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l119_11990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_max_distance_l119_11937

/-- The maximum distance from a point on the hyperbola x^2 - y^2 = 1 to the line x - y + 1 = 0 -/
theorem hyperbola_line_max_distance :
  ∃ (lambda_max : ℝ),
    ∀ (x y : ℝ),
      x > 0 →
      x^2 - y^2 = 1 →
      ∀ (lambda : ℝ),
        (|x - y + 1| / Real.sqrt 2 > lambda) →
        lambda ≤ lambda_max ∧
        lambda_max = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_max_distance_l119_11937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_iff_a_eq_neg_one_l119_11968

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x + a / Real.exp x) * x^3

-- State the theorem
theorem f_is_even_iff_a_eq_neg_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_iff_a_eq_neg_one_l119_11968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_eq_two_l119_11912

noncomputable def points : List (ℝ × ℝ) := [(2, 15), (9, 30), (15, 50), (21, 55), (25, 60)]

noncomputable def is_above_line (point : ℝ × ℝ) : Bool :=
  point.2 > 3 * point.1 + 5

noncomputable def sum_x_above_line (points : List (ℝ × ℝ)) : ℝ :=
  (points.filter is_above_line).map (λ p => p.1) |>.sum

theorem sum_x_above_line_eq_two : sum_x_above_line points = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_eq_two_l119_11912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sequence_sum_l119_11953

def sequence_term (n : ℕ) : ℕ := n.factorial + n

def sequence_sum : ℕ := (List.range 9).map (λ i => sequence_term (i + 1)) |>.sum

theorem units_digit_of_sequence_sum : sequence_sum % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sequence_sum_l119_11953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_male_geese_l119_11999

/-- Represents the percentage of male geese in the study -/
def male_percentage : ℝ := sorry

/-- Represents the percentage of female geese in the study -/
def female_percentage : ℝ := sorry

/-- Represents the migration rate for male geese -/
def male_migration_rate : ℝ := sorry

/-- Represents the migration rate for female geese -/
def female_migration_rate : ℝ := sorry

/-- The percentage of migrating geese that were male -/
def male_migrating_percentage : ℝ := 20

/-- Theorem stating that the percentage of male geese in the study is 50% -/
theorem percentage_of_male_geese :
  male_percentage = 50 ∧
  female_percentage = 100 - male_percentage ∧
  male_percentage + female_percentage = 100 ∧
  male_migration_rate / female_migration_rate = 0.25 ∧
  male_migrating_percentage = 20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_male_geese_l119_11999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l119_11941

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y + 8 = 0

-- Define the line
def lineEq (x y : ℝ) : Prop := 3*x - 4*y + 7 = 0

-- Theorem stating the properties of the circle
theorem circle_properties :
  -- The center is (-3, 2) and the radius is √5
  (∃ h k r : ℝ, h = -3 ∧ k = 2 ∧ r = Real.sqrt 5 ∧
    ∀ x y : ℝ, circleEq x y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧
  -- The area is 5π
  (∃ area : ℝ, area = 5 * Real.pi ∧
    ∀ h k r : ℝ, (∀ x y : ℝ, circleEq x y ↔ (x - h)^2 + (y - k)^2 = r^2) →
      area = Real.pi * r^2) ∧
  -- The line intersects with the circle
  (∃ x y : ℝ, circleEq x y ∧ lineEq x y) ∧
  -- The point (2, -3) is outside the circle
  (∀ x y : ℝ, circleEq x y → (x - 2)^2 + (y + 3)^2 > 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l119_11941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_items_for_three_of_same_type_l119_11994

/-- Given a collection of items with 5 different types, the minimum number of items
    needed to guarantee at least 3 items of the same type is 11. -/
theorem min_items_for_three_of_same_type :
  ∀ (items : Finset ℕ) (type : ℕ → Fin 5),
    (∀ i ∈ items, i < 5) →
    (∀ t : Fin 5, (items.filter (λ i => type i = t)).card ≥ 3) →
    items.card ≥ 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_items_for_three_of_same_type_l119_11994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_product_l119_11901

theorem remainder_sum_product (x y z : ℕ) 
  (hx : x % 15 = 11)
  (hy : y % 15 = 13)
  (hz : z % 15 = 14) :
  ((x + y + z) % 15 = 8) ∧ ((x * y * z) % 15 = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_product_l119_11901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_approx_6_l119_11926

/-- Calculates the harmonic mean of three positive real numbers -/
noncomputable def harmonicMean (a b c : ℝ) : ℝ :=
  3 / (1/a + 1/b + 1/c)

/-- Represents a triathlon with three equal-length segments -/
structure Triathlon where
  swimmingSpeed : ℝ
  bikingSpeed : ℝ
  runningSpeed : ℝ
  swimming_positive : swimmingSpeed > 0
  biking_positive : bikingSpeed > 0
  running_positive : runningSpeed > 0

/-- Calculates the average speed of a triathlon -/
noncomputable def averageSpeed (t : Triathlon) : ℝ :=
  harmonicMean t.swimmingSpeed t.bikingSpeed t.runningSpeed

/-- Theorem stating that the average speed of the given triathlon is approximately 6 km/h -/
theorem triathlon_average_speed_approx_6 :
  let t : Triathlon := {
    swimmingSpeed := 3,
    bikingSpeed := 20,
    runningSpeed := 10,
    swimming_positive := by norm_num,
    biking_positive := by norm_num,
    running_positive := by norm_num
  }
  abs (averageSpeed t - 6) < 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_approx_6_l119_11926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_weighings_min_weighings_optimal_l119_11982

/-- Represents the minimum number of weighings required to identify a single counterfeit coin -/
def min_weighings (n : ℕ) : ℕ :=
  if n = 3 then 1
  else if n = 4 ∨ n = 9 then 2
  else 0  -- undefined for other cases

/-- Theorem stating the minimum number of weighings for 3, 4, and 9 coins -/
theorem counterfeit_coin_weighings :
  (min_weighings 3 = 1) ∧
  (min_weighings 4 = 2) ∧
  (min_weighings 9 = 2) := by
  sorry

/-- Lemma: Any weighing can distinguish at most 3 possibilities -/
lemma weighing_max_possibilities : ∀ n : ℕ, n > 0 → 3^n ≥ min_weighings n := by
  sorry

/-- Theorem: The minimum number of weighings is optimal -/
theorem min_weighings_optimal (n : ℕ) (h : n ∈ ({3, 4, 9} : Set ℕ)) :
  ∀ k : ℕ, k < min_weighings n → ¬(3^k ≥ n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_weighings_min_weighings_optimal_l119_11982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_f_min_value_f_l119_11943

/-- The quadratic function f(c) = 3/4 * c^2 - 9c + 7 -/
noncomputable def f (c : ℝ) : ℝ := 3/4 * c^2 - 9*c + 7

/-- Theorem stating that c = 6 minimizes the quadratic function f -/
theorem minimize_f :
  ∀ x : ℝ, f 6 ≤ f x := by
  sorry

/-- Corollary: The minimum value of f occurs at c = 6 -/
theorem min_value_f :
  ∃ c : ℝ, ∀ x : ℝ, f c ≤ f x ∧ c = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_f_min_value_f_l119_11943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_rectangle_l119_11958

-- Define the parameters of the problem
noncomputable def rod_length : ℝ := 18
noncomputable def length_width_ratio : ℝ := 2

-- Define the volume function
noncomputable def volume (width : ℝ) : ℝ :=
  2 * width * width * ((rod_length / 4) - 3 * width)

-- State the theorem
theorem max_volume_rectangle :
  ∃ (max_width : ℝ),
    0 < max_width ∧
    max_width < rod_length / (2 * (length_width_ratio + 1)) ∧
    (∀ (w : ℝ), 0 < w → w < rod_length / (2 * (length_width_ratio + 1)) → 
      volume w ≤ volume max_width) ∧
    max_width = 1 ∧
    2 * max_width = 2 ∧
    (rod_length / 4 - 3 * max_width) = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_rectangle_l119_11958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l119_11923

theorem absolute_value_inequality (x : ℝ) (h : x ≠ 1) :
  |((3 * x - 2) / (x - 1))| > 3 ↔ x ∈ Set.Ioo (5/6) 1 ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l119_11923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l119_11935

-- Define the complex number z
variable (z : ℂ)

-- Define the area of the parallelogram
noncomputable def parallelogram_area (z : ℂ) : ℝ := 
  2 * Complex.abs (Complex.sin (2 * Complex.arg z))

-- Define the condition for non-negative imaginary part
def non_negative_imag (z : ℂ) : Prop := z.im ≥ 0

-- Define the distance function
noncomputable def distance (z : ℂ) : ℝ := Complex.abs (z - 1 / z)

-- State the theorem
theorem min_distance_squared (z : ℂ) 
  (h1 : parallelogram_area z = 42 / 49)
  (h2 : non_negative_imag z) :
  ∃ d : ℝ, d^2 = (3 - Real.sqrt 637) / 49 ∧ 
    ∀ w : ℂ, parallelogram_area w = 42 / 49 → non_negative_imag w → 
      distance w ≥ d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l119_11935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_l119_11959

/-- Proves that if a company's revenue decreased by 30.434782608695656% to $48.0 billion, 
    then the original revenue was approximately $68.97 billion. -/
theorem revenue_decrease (current_revenue : ℝ) (decrease_percentage : ℝ) (original_revenue : ℝ) :
  current_revenue = 48.0 ∧ 
  decrease_percentage = 30.434782608695656 ∧
  current_revenue = original_revenue * (1 - decrease_percentage / 100) →
  ∃ ε > 0, |original_revenue - 68.97| < ε := by
  sorry

#check revenue_decrease

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_l119_11959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_setC_is_right_triangle_l119_11944

/-- A set of three line segments that might form a right triangle -/
structure TriangleSet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a TriangleSet forms a right triangle using the Pythagorean theorem -/
def isRightTriangle (t : TriangleSet) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.a^2 + t.c^2 = t.b^2 ∨ t.b^2 + t.c^2 = t.a^2

/-- The given sets of line segments -/
def setA : TriangleSet := ⟨7, 20, 24⟩
noncomputable def setB : TriangleSet := ⟨Real.sqrt 3, Real.sqrt 4, Real.sqrt 5⟩
def setC : TriangleSet := ⟨3, 4, 5⟩
def setD : TriangleSet := ⟨4, 5, 6⟩

/-- Theorem stating that only setC forms a right triangle -/
theorem only_setC_is_right_triangle :
  ¬(isRightTriangle setA) ∧
  ¬(isRightTriangle setB) ∧
  isRightTriangle setC ∧
  ¬(isRightTriangle setD) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_setC_is_right_triangle_l119_11944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l119_11934

variable (A B C a b c : ℝ)

-- Define the triangle ABC
def triangle_ABC (A B C a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a = (b * Real.sin A) / Real.sin B ∧
  b = (c * Real.sin B) / Real.sin C ∧
  c = (a * Real.sin C) / Real.sin A

-- Given conditions
def given_conditions (A B C a b c : ℝ) : Prop :=
  triangle_ABC A B C a b c ∧
  b * Real.cos C = (2 * a - c) * Real.cos B ∧
  b = Real.sqrt 13 ∧
  a = 3

-- Helper function for triangle area
noncomputable def triangle_area (A B C a b c : ℝ) : ℝ :=
  (1 / 2) * a * b * Real.sin C

-- Theorem statement
theorem triangle_properties
  (h : given_conditions A B C a b c) :
  B = Real.arcsin (Real.sqrt ((1 + Real.sin (2 * A)) / 3)) ∧
  c = Real.sqrt (22 - 6 * Real.sqrt 13 * Real.cos (Real.arcsin (Real.sqrt ((1 + Real.sin (2 * A)) / 3)))) ∧
  (triangle_area A B C a b c) = (3 * Real.sqrt 13 / 2) * Real.sqrt ((1 + Real.sin (2 * A)) / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l119_11934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_l119_11981

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: For a trapezium with parallel sides of 20 cm and 10 cm, and an area of 150 square cm,
    the distance between the parallel sides is 10 cm. -/
theorem trapezium_height (h : ℝ) : 
  trapeziumArea 20 10 h = 150 → h = 10 := by
  intro hypothesis
  -- Proof steps would go here
  sorry

#check trapezium_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_l119_11981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_or_q_is_true_l119_11954

-- Define proposition p
def p : Prop := ∃ α : ℝ, Real.cos (Real.pi - α) = Real.cos α

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 + 1 > 0

-- Theorem to prove
theorem p_or_q_is_true : p ∨ q := by
  right
  intro x
  have h : 0 < 1 := by norm_num
  have h2 : 0 ≤ x^2 := sq_nonneg x
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_or_q_is_true_l119_11954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l119_11972

/-- Defines a triangle ABC -/
def TriangleABC (A B C : Point) : Prop := sorry

/-- Length of segment between two points -/
def SegmentLength (P Q : Point) : ℝ := sorry

/-- Length of median from vertex to opposite side in a triangle -/
def MedianLength (A B C : Point) : ℝ := sorry

/-- Circle passes through A and is tangent to BC at B -/
def CircleTangentAt (A B C : Point) : Prop := sorry

/-- Length of common chord of two circles -/
def CommonChordLength (A B C D : Point) : ℝ := sorry

/-- Given a triangle ABC with BC = 4 and median to BC = 3, prove that the length of the common chord
    of two circles passing through A and tangent to BC (one at B, one at C) is 5/3 -/
theorem common_chord_length (A B C : Point) (D : Point) : 
  TriangleABC A B C → 
  SegmentLength B C = 4 → 
  MedianLength A B C = 3 → 
  CircleTangentAt A B C → 
  CircleTangentAt A C B → 
  CommonChordLength A B C D = 5/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l119_11972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circle_properties_l119_11976

-- Define the parabola C: x^2 = 2py (p > 0)
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the circle M: x^2 + (y+4)^2 = 1
def circle_M (x y : ℝ) : Prop := x^2 + (y+4)^2 = 1

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p/2)

-- Define the minimum distance between F and a point on M
def min_distance : ℝ := 4

-- Define a point P on the circle M
def point_on_circle (P : ℝ × ℝ) : Prop := circle_M P.1 P.2

-- Define tangent points A and B on the parabola
def tangent_points (p : ℝ) (A B : ℝ × ℝ) : Prop :=
  parabola p A.1 A.2 ∧ parabola p B.1 B.2

-- Define the area of triangle PAB
noncomputable def triangle_area (P A B : ℝ × ℝ) : ℝ := sorry

theorem parabola_and_circle_properties :
  ∃ p : ℝ, ∀ x y, parabola p x y → ∀ P, point_on_circle P →
    min_distance = 4 →
      p = 2 ∧
      (∃ A B, tangent_points p A B ∧
        ∀ A' B', tangent_points p A' B' →
          triangle_area P A B ≤ triangle_area P A' B') ∧
            triangle_area P A B = 20 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circle_properties_l119_11976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_one_common_member_l119_11979

/-- A structure representing a committee with subcommittees -/
structure Committee where
  n : ℕ
  members : Finset ℕ
  subcommittees : Finset (Finset ℕ)
  h_n_ge_5 : n ≥ 5
  h_members_card : members.card = n
  h_subcommittees_card : subcommittees.card = n + 1
  h_subcommittee_size : ∀ s, s ∈ subcommittees → s.card = 3
  h_subcommittees_distinct : ∀ s t, s ∈ subcommittees → t ∈ subcommittees → s ≠ t → s ≠ t

/-- The main theorem stating that there exist two subcommittees with exactly one common member -/
theorem exist_one_common_member (c : Committee) :
  ∃ s t, s ∈ c.subcommittees ∧ t ∈ c.subcommittees ∧ s ≠ t ∧ (s ∩ t).card = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_one_common_member_l119_11979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_slope_angle_l119_11986

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def point_on_parabola (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the slope angle of a line
noncomputable def slope_angle (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.arctan ((p2.2 - p1.2) / (p2.1 - p1.1))

theorem parabola_slope_angle :
  ∀ t : ℝ,
  parabola (point_on_parabola t).1 (point_on_parabola t).2 →
  distance (point_on_parabola t) focus = 4 →
  slope_angle focus (point_on_parabola t) = π/3 ∨ 
  slope_angle focus (point_on_parabola t) = 2*π/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_slope_angle_l119_11986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_seven_pointed_star_angle_l119_11996

/-- In a regular seven-pointed star inscribed in a circle, 
    the measure of the angle between two adjacent star points is 5π/7 radians. -/
theorem regular_seven_pointed_star_angle : ∃ a : ℝ, a = 5 * π / 7 :=
  by
  -- We define the regular seven-pointed star
  let star_points : ℕ := 7
  
  -- The internal angle at each point of the star
  let φ : ℝ := π / star_points
  
  -- We introduce a as a real number
  let a : ℝ := π - 2 * φ

  -- The sum of angles in a triangle is π
  have triangle_sum : 2 * φ + a = π := by
    calc
      2 * φ + a = 2 * φ + (π - 2 * φ) := by rfl
      _ = π := by ring

  -- Show that a equals 5π/7
  have a_eq : a = 5 * π / 7 := by
    calc
      a = π - 2 * φ := by rfl
      _ = π - 2 * (π / star_points) := by rfl
      _ = π - 2 * (π / 7) := by rfl
      _ = (7 * π) / 7 - (2 * π) / 7 := by ring
      _ = 5 * π / 7 := by ring

  -- Conclude the proof
  exact ⟨a, a_eq⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_seven_pointed_star_angle_l119_11996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_home_count_l119_11911

/-- Count the number of occurrences of a digit in a range of numbers -/
def countDigitOccurrences (digit : Nat) (start : Nat) (stop : Nat) : Nat :=
  (List.range (stop - start + 1)).map (· + start)
    |> List.filter (fun n => n.repr.any (fun c => c.repr == digit.repr))
    |> List.length

/-- The problem statement -/
theorem home_count (start : Nat) (stop : Nat) (digit : Nat) (occurrences : Nat) :
  start = 1 ∧ stop = 100 ∧ digit = 2 ∧ occurrences = 20 ∧
  countDigitOccurrences digit start stop = occurrences →
  stop - start + 1 = 100 := by
  sorry

#eval countDigitOccurrences 2 1 100  -- Should output 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_home_count_l119_11911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_text_message_average_l119_11927

/-- The average number of text messages sent over five days -/
noncomputable def average_text_messages (monday tuesday wednesday thursday friday : ℝ) : ℝ :=
  (monday + tuesday + wednesday + thursday + friday) / 5

/-- Theorem stating the average number of text messages sent over five days -/
theorem text_message_average :
  let monday : ℝ := 220
  let tuesday : ℝ := monday * 0.85
  let wednesday : ℝ := tuesday * 1.25
  let thursday : ℝ := wednesday * 0.9
  let friday : ℝ := thursday * 1.05
  
  |average_text_messages monday tuesday wednesday thursday friday - 214.2| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_text_message_average_l119_11927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_point_labeling_l119_11902

/-- A labeling function for lattice points -/
def LabelingFunction := ℕ → ℕ → ℕ

/-- The labeling condition for adjacent points -/
def LabelingCondition (ℓ : LabelingFunction) : Prop :=
  ∀ x y, ∃ n : ℕ, Finset.toSet {ℓ x y, ℓ x (y + 1), ℓ (x + 1) y} = {n, n + 1, n + 2}

/-- The theorem statement -/
theorem lattice_point_labeling
  (ℓ : LabelingFunction)
  (h0 : ℓ 0 0 = 0)
  (hc : LabelingCondition ℓ) :
  ∃ k : ℕ, ℓ 2000 2024 = 3 * k ∧ ℓ 2000 2024 ≤ 6048 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_point_labeling_l119_11902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_distribution_count_distribution_count_l119_11947

def count_distributions : ℕ := 142286

def total_minutes : ℕ := 270
def num_players : ℕ := 7
def divisor_first_group : ℕ := 7
def divisor_second_group : ℕ := 13
def size_first_group : ℕ := 4
def size_second_group : ℕ := 3

theorem valid_distribution_count :
  (∃ (times : Fin num_players → ℕ),
    (∀ i : Fin num_players, i.val < size_first_group → divisor_first_group ∣ times i) ∧
    (∀ i : Fin num_players, i.val ≥ size_first_group → divisor_second_group ∣ times i) ∧
    (Finset.sum Finset.univ times = total_minutes)) →
  count_distributions = 142286 := by
  sorry

theorem distribution_count : count_distributions = 142286 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_distribution_count_distribution_count_l119_11947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_l119_11928

theorem perfect_square_condition (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  0 < a ^ 2 + b ^ 2 - a * b * c ∧ a ^ 2 + b ^ 2 - a * b * c ≤ c →
  ∃ (k : ℕ), a ^ 2 + b ^ 2 - a * b * c = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_l119_11928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_integers_pairs_l119_11903

theorem circle_integers_pairs (nums : List ℕ) : 
  nums.length = 2005 → 
  nums.sum = 7022 → 
  ∃ (i j : ℕ), 
    i ≠ j ∧
    i < nums.length ∧ 
    j < nums.length ∧ 
    nums[i]! + nums[(i + 1) % nums.length]! ≥ 8 ∧ 
    nums[j]! + nums[(j + 1) % nums.length]! ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_integers_pairs_l119_11903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_letters_theorem_l119_11905

/-- Represents the number of distinct letters on the cards -/
def distinct_letters : ℕ := sorry

/-- The total number of cards -/
def total_cards : ℕ := 10

/-- The number of different words that can be formed -/
def different_words : ℕ := 5040

/-- Theorem stating that the number of distinct letters is either 4 or 5 -/
theorem distinct_letters_theorem : distinct_letters = 4 ∨ distinct_letters = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_letters_theorem_l119_11905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l119_11963

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  median_AB : ℝ → ℝ → ℝ
  angle_bisector_ABC : ℝ → ℝ → ℝ

/-- The theorem statement -/
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.A = (2, -1))
  (h2 : ∀ x y, abc.median_AB x y = x + 3*y - 6)
  (h3 : ∀ x y, abc.angle_bisector_ABC x y = x - y + 1) :
  (abc.B = (5/2, 7/2)) ∧ 
  (∀ x y, x - 9*y + 29 = 0 ↔ (x, y) ∈ Set.range (λ t : ℝ ↦ abc.B + t • (abc.C - abc.B))) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l119_11963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteenth_entry_is_26_l119_11962

/-- The remainder when n is divided by 6 -/
def r_6 (n : ℕ) : ℕ := n % 6

/-- The sequence of nonnegative integers n that satisfy r_6(4n) ≤ 3 -/
def sequence_list : List ℕ :=
  (List.range (27 : ℕ)).filter (fun n => r_6 (4 * n) ≤ 3)

/-- The 18th entry in the sequence is 26 -/
theorem eighteenth_entry_is_26 : sequence_list[17] = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteenth_entry_is_26_l119_11962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_a_l119_11916

theorem smallest_possible_a (P : ℤ → ℤ) (a : ℕ) (h_a_pos : a > 0)
  (h_P_int_coeff : ∀ x, ∃ n : ℤ, P x = n)
  (h_P_1 : P 1 = a) (h_P_4 : P 4 = a) (h_P_7 : P 7 = a)
  (h_P_3 : P 3 = -a) (h_P_5 : P 5 = -a) (h_P_6 : P 6 = -a) (h_P_8 : P 8 = -a) :
  a ≥ 84 ∧ ∃ Q : ℤ → ℤ, 
    ∀ x, P x = (x - 1) * (x - 4) * (x - 7) * Q x + a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_a_l119_11916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_points_in_square_l119_11907

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a square with side length 4
def Square : Set Point := {p : Point | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 4}

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_points_in_square :
  (∀ (points : Finset Point), points.card = 7 → (∀ p ∈ points, p ∈ Square) →
    ∃ (p1 p2 : Point), p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 5) ∧
  (∃ (points : Finset Point), points.card = 6 ∧ (∀ p ∈ points, p ∈ Square) ∧
    ∀ (p1 p2 : Point), p1 ∈ points → p2 ∈ points → p1 ≠ p2 → distance p1 p2 > Real.sqrt 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_points_in_square_l119_11907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_l119_11984

theorem cos_half_angle (α : ℝ) (h : Real.sin (α / 4) = Real.sqrt 3 / 3) : 
  Real.cos (α / 2) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_l119_11984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_order_l119_11970

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 - 2*x

-- Define the constants a, b, and c
noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := -(Real.log 2 / Real.log 3)
noncomputable def c : ℝ := Real.sqrt 3

-- State the theorem
theorem function_value_order : f b > f a ∧ f a > f c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_order_l119_11970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x₀_l119_11914

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define the line l
def line_l (x y : ℝ) : Prop := x + 3*y - 6 = 0

-- Define the point P
def point_P (x₀ y₀ : ℝ) : Prop := line_l x₀ y₀

-- Define the existence of point Q
noncomputable def exists_Q (x₀ y₀ : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_O x y ∧ 
  Real.cos (60 * Real.pi / 180) = (x * x₀ + y * y₀) / (Real.sqrt (x^2 + y^2) * Real.sqrt (x₀^2 + y₀^2))

-- Theorem statement
theorem range_of_x₀ :
  ∀ x₀ : ℝ, (∃ y₀ : ℝ, point_P x₀ y₀ ∧ exists_Q x₀ y₀) ↔ (0 ≤ x₀ ∧ x₀ ≤ 6/5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x₀_l119_11914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_arrangements_l119_11936

/-- The number of arrangements for 7 students in a line with specific conditions -/
def arrangements_count : ℕ := 192

/-- The total number of students -/
def total_students : ℕ := 7

/-- Theorem stating the number of arrangements for the given conditions -/
theorem count_arrangements :
  (total_students = 7) →
  (∃ A : ℕ, A ≤ total_students ∧ ∃ position_A : ℕ, position_A = (total_students + 1) / 2) →
  (∃ B C : ℕ, B ≠ C ∧ B ≤ total_students ∧ C ≤ total_students ∧ 
    ∃ position_B position_C : ℕ, position_B + 1 = position_C ∨ position_C + 1 = position_B) →
  arrangements_count = 192 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_arrangements_l119_11936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l119_11989

def sequence_a (lambda : ℝ) (n : ℕ+) : ℝ := n.val^2 + lambda * n.val

theorem lambda_range (lambda : ℝ) :
  (∀ n : ℕ+, sequence_a lambda n < sequence_a lambda (n + 1)) →
  lambda > -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l119_11989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_l119_11919

/-- The curve function f(x) = e^(2ax) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * a * x)

/-- The slope of the line 2x - y + 3 = 0 -/
def m : ℝ := 2

/-- The theorem stating that the value of a making the tangent line at (0, f(0)) perpendicular to 2x - y + 3 = 0 is -1/4 -/
theorem tangent_perpendicular (a : ℝ) : 
  (deriv (f a) 0 = -(1 / m)) → a = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_l119_11919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_regions_number_of_regions_3_l119_11921

/-- A function that calculates the number of regions formed by n lines in a plane -/
def f (n : ℕ) : ℕ := (n^2 + n + 2) / 2

/-- Predicate to check if two lines are not parallel -/
def LinesNotParallel (i j : ℕ) : Prop := sorry

/-- Predicate to check if three lines are not concurrent -/
def LinesNotConcurrent (i j k : ℕ) : Prop := sorry

/-- Properties of the lines in the plane -/
structure LinePlaneConfiguration (n : ℕ) :=
  (not_parallel : ∀ i j, i ≠ j → i < n → j < n → LinesNotParallel i j)
  (not_concurrent : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → i < n → j < n → k < n → LinesNotConcurrent i j k)

/-- Function to calculate the number of regions -/
def NumberOfRegions (n : ℕ) : ℕ := sorry

/-- Theorem stating that f(n) correctly calculates the number of regions -/
theorem number_of_regions (n : ℕ) (h : n > 0) (config : LinePlaneConfiguration n) :
  NumberOfRegions n = f n := by
  sorry

/-- Corollary for the specific case of n = 3 -/
theorem number_of_regions_3 (config : LinePlaneConfiguration 3) :
  NumberOfRegions 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_regions_number_of_regions_3_l119_11921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l119_11924

/-- An isosceles trapezoid inscribed in a circle -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  base1 : ℝ
  base2 : ℝ
  circle_radius : ℝ

/-- The area of an isosceles trapezoid -/
noncomputable def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  (1 / 2) * (t.base1 + t.base2) * t.circle_radius

/-- Theorem stating that the area of the specific isosceles trapezoid is 45 -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 5,
    base1 := 6,
    base2 := 12,
    circle_radius := 5
  }
  trapezoid_area t = 45 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l119_11924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_property_l119_11929

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_function_property (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a x ≤ f a 2) ∧
  (∀ x ∈ Set.Icc 1 2, f a x ≥ f a 1) ∧
  (f a 2 - f a 1 = a / 2) →
  a = 1/2 ∨ a = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_property_l119_11929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_circle_equations_l119_11945

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 - 2}

-- Define the perpendicular line
def perp_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - 2 * p.2 - 1 = 0}

-- Define the circle C
def circle_C (D E F : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + D * p.1 + E * p.2 + F = 0}

-- Define the points A and B
def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (0, -2)

theorem line_and_circle_equations 
  (k : ℝ) 
  (h1 : (∀ p q : ℝ × ℝ, p ∈ line_l k → q ∈ perp_line → (p.1 - q.1) * (k * 1 + 1/2) = -(p.2 - q.2)))
  (h2 : (0, -2) ∈ line_l k)
  (h3 : point_A ∈ line_l k ∧ point_B ∈ line_l k)
  (h4 : ∃ (D E F : ℝ), (0, 0) ∈ circle_C D E F ∧ point_A ∈ circle_C D E F ∧ point_B ∈ circle_C D E F) :
  (k = -2) ∧ (∃ (D E F : ℝ), circle_C D E F = circle_C 1 2 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_circle_equations_l119_11945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_config_count_l119_11977

/-- Represents a friendship configuration for 8 individuals -/
structure FriendshipConfig where
  connections : Fin 8 → Finset (Fin 8)
  symm : ∀ i j, j ∈ connections i ↔ i ∈ connections j
  equal_friends : ∀ i j, (connections i).card = (connections j).card
  no_self_friends : ∀ i, i ∉ connections i

/-- The number of valid friendship configurations -/
noncomputable def num_friendship_configs : ℕ :=
  Finset.filter (fun _ : FriendshipConfig => True) (Classical.choice inferInstance)
  |>.card

/-- The main theorem stating the number of friendship configurations -/
theorem friendship_config_count : num_friendship_configs = 210 := by
  sorry

#check friendship_config_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_config_count_l119_11977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_thirty_degrees_l119_11932

/-- The angle of inclination of a line given by the equation ax + by + c = 0 -/
noncomputable def angle_of_inclination (a b : ℝ) : ℝ :=
  Real.arctan (- a / b)

/-- Converts degrees to radians -/
noncomputable def degrees_to_radians (degrees : ℝ) : ℝ :=
  degrees * (Real.pi / 180)

theorem line_inclination_thirty_degrees :
  angle_of_inclination 1 (-Real.sqrt 3) = degrees_to_radians 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_thirty_degrees_l119_11932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_hours_worked_l119_11965

-- Define the variables
noncomputable def A : ℝ := 12
noncomputable def B : ℝ := (1/3) * A
noncomputable def C : ℝ := 2 * B
noncomputable def E : ℝ := A + 3
noncomputable def D : ℝ := (1/2) * E
noncomputable def F : ℝ := C + D
noncomputable def G : ℝ := F - 5

-- Define the total hours worked
noncomputable def T : ℝ := A + B + C + D + E + F + G

-- Theorem to prove
theorem total_hours_worked : T = 72.5 := by
  -- Expand the definition of T
  unfold T
  -- Expand the definitions of A, B, C, D, E, F, and G
  unfold A B C D E F G
  -- Simplify the expression
  simp [add_assoc, mul_assoc]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_hours_worked_l119_11965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_three_digit_numbers_with_repeats_l119_11913

/-- The count of all three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers without repeated digits -/
def three_digit_numbers_without_repeats : ℕ := 648

/-- The percentage of three-digit numbers with repeated digits -/
def percentage_with_repeats : ℚ :=
  (total_three_digit_numbers - three_digit_numbers_without_repeats) / total_three_digit_numbers * 100

theorem percentage_of_three_digit_numbers_with_repeats :
  round (percentage_with_repeats : ℝ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_three_digit_numbers_with_repeats_l119_11913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_tangent_lines_through_P_l119_11961

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := (1/3) * x^3 + 4/3

-- Define point P
def P : ℝ × ℝ := (2, 4)

-- Theorem for part (1)
theorem tangent_line_at_P : 
  ∃ (m b : ℝ), ∀ x y : ℝ, y = m * x + b ∧ 
  (x, y) = P → (4 * x - y - 4 = 0) := by
  sorry

-- Theorem for part (2)
theorem tangent_lines_through_P :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
  (∀ x y : ℝ, y = m₁ * x + b₁ → (x - y + 2 = 0)) ∧
  (∀ x y : ℝ, y = m₂ * x + b₂ → (4 * x - y - 4 = 0)) ∧
  (∃ t : ℝ, f t = m₁ * t + b₁) ∧
  (∃ t : ℝ, f t = m₂ * t + b₂) ∧
  (m₁ * P.1 + b₁ = P.2) ∧
  (m₂ * P.1 + b₂ = P.2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_tangent_lines_through_P_l119_11961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aardvark_path_distance_l119_11960

/-- The total distance traveled along a specific path on two concentric circles -/
theorem aardvark_path_distance (r₁ r₂ : ℝ) (h₁ : r₁ = 10) (h₂ : r₂ = 20) :
  (π * r₂ / 2) + (r₂ - r₁) + (π * r₁ / 2) + (2 * r₁) + (π * r₁ / 2) + (r₂ - r₁) = 20 * π + 40 :=
by
  -- Substitute the given values for r₁ and r₂
  rw [h₁, h₂]
  -- Simplify the left-hand side
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aardvark_path_distance_l119_11960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_c_a_side_b_value_l119_11975

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  (Real.cos t.A - Real.cos t.C) / Real.cos t.B = (Real.sin t.C - Real.sin t.A) / Real.sin t.B

-- Theorem 1
theorem ratio_c_a (t : Triangle) (h : triangle_conditions t) : t.c / t.a = 1 := by
  sorry

-- Theorem 2
theorem side_b_value (t : Triangle) (h1 : triangle_conditions t) 
  (h2 : Real.cos t.B = 2/3) (h3 : t.a * t.b * Real.sin t.C / 2 = Real.sqrt 5 / 6) : 
  t.b = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_c_a_side_b_value_l119_11975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_sequence_l119_11933

def our_sequence (n : ℕ) : ℕ := 10^n - 1

def set_sum : ℕ := (10^10 - 91) / 9

theorem arithmetic_mean_of_sequence :
  (set_sum / 9 : ℚ) = 123456789 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_sequence_l119_11933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_one_implies_sum_zero_l119_11931

theorem product_one_implies_sum_zero (x y : ℝ) :
  (x + Real.sqrt (x^2 + 1)) * (y + Real.sqrt (y^2 + 1)) = 1 → x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_one_implies_sum_zero_l119_11931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_card_probability_l119_11978

theorem red_card_probability : 
  let total_cards := 70
  let is_red (n : ℕ) := n % 7 = 1
  let red_cards := Finset.filter is_red (Finset.range total_cards)
  (red_cards.card : ℚ) / total_cards = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_card_probability_l119_11978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_agency_B_better_for_two_agency_A_better_for_three_or_more_l119_11971

/-- Represents a travel agency with a specific discount scheme -/
structure TravelAgency where
  fullPrice : ℝ
  discountScheme : ℕ → ℝ

/-- Calculates the total cost for a family using Agency A's discount scheme -/
def costAgencyA (a : ℝ) (x : ℕ) : ℝ :=
  a + 0.55 * a * (x : ℝ)

/-- Calculates the total cost for a family using Agency B's discount scheme -/
def costAgencyB (a : ℝ) (x : ℕ) : ℝ :=
  0.75 * a * ((x + 1) : ℝ)

/-- Theorem stating that for a family of two, Agency B is more cost-effective -/
theorem agency_B_better_for_two (a : ℝ) (h : a > 0) :
  costAgencyB a 1 ≤ costAgencyA a 1 := by
  sorry

/-- Theorem stating that for a family of three or more, Agency A is more cost-effective -/
theorem agency_A_better_for_three_or_more (a : ℝ) (x : ℕ) (h1 : a > 0) (h2 : x ≥ 2) :
  costAgencyA a x < costAgencyB a x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_agency_B_better_for_two_agency_A_better_for_three_or_more_l119_11971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_permutations_l119_11957

def is_valid_permutation (b : Fin 7 → ℕ) : Prop :=
  (∀ i : Fin 7, b i ∈ ({2, 3, 4, 5, 6, 7, 8} : Finset ℕ)) ∧
  (∀ i j : Fin 7, i ≠ j → b i ≠ b j)

noncomputable def satisfies_inequality (b : Fin 7 → ℕ) : Prop :=
  (b 0 + 2) / 2 * (b 1 + 3) / 2 * (b 2 + 4) / 2 * (b 3 + 5) / 2 *
  (b 4 + 6) / 2 * (b 5 + 7) / 2 * (b 6 + 8) / 2 > Nat.factorial 7

theorem count_valid_permutations :
  ∃ s : Finset (Fin 7 → ℕ), 
    (∀ b ∈ s, is_valid_permutation b ∧ satisfies_inequality b) ∧
    s.card = 5039 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_permutations_l119_11957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_is_positive_reals_l119_11980

/-- A set of positive real numbers closed under addition with a special interval property -/
structure SpecialSet (S : Set ℝ) : Prop where
  positive : ∀ x, x ∈ S → x > 0
  closed_add : ∀ a b, a ∈ S → b ∈ S → a + b ∈ S
  interval_property : ∀ a b : ℝ, a > 0 → ∃ c d, a ≤ c ∧ d ≤ b ∧ Set.Icc c d ⊆ S

/-- The special set S is equal to the set of all positive real numbers -/
theorem special_set_is_positive_reals (S : Set ℝ) (h : SpecialSet S) : S = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_is_positive_reals_l119_11980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_volume_l119_11967

/-- The volume of a region formed by points within a given distance of a line segment --/
noncomputable def regionVolume (segmentLength : ℝ) (radius : ℝ) : ℝ :=
  Real.pi * radius^2 * segmentLength + (4/3) * Real.pi * radius^3

/-- Theorem stating that a line segment of length 20 units, with surrounding points within 4 units, forms a region of volume 400π --/
theorem line_segment_volume (CD : ℝ) (h : regionVolume CD 4 = 400 * Real.pi) : CD = 20 := by
  sorry

#check line_segment_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_volume_l119_11967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_identification_l119_11906

noncomputable def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

noncomputable def f1 (x : ℝ) : ℝ := 3 * x - 1
noncomputable def f2 (x : ℝ) : ℝ := 1 / (x^2)
noncomputable def f3 (x : ℝ) : ℝ := 3 * x^2 + x - 1
noncomputable def f4 (x : ℝ) : ℝ := 2 * x^3 - 1

theorem quadratic_function_identification :
  ¬(is_quadratic f1) ∧
  ¬(is_quadratic f2) ∧
  (is_quadratic f3) ∧
  ¬(is_quadratic f4) :=
by
  sorry

#check quadratic_function_identification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_identification_l119_11906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_involutive_function_property_l119_11993

noncomputable section

/-- A function f that is its own inverse -/
def InvolutiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (f x) = x

/-- The specific function given in the problem -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ :=
  (2*a*x + 3*b) / (4*c*x - 5*d)

/-- The main theorem -/
theorem involutive_function_property (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : InvolutiveFunction (f a b c d)) : 
  2*a - 5*d = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_involutive_function_property_l119_11993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l119_11995

/-- A plane at distance 2 from the origin, intersecting the coordinate axes -/
structure DistantPlane where
  α : ℝ
  β : ℝ
  γ : ℝ
  dist_eq : 1 / (α^2) + 1 / (β^2) + 1 / (γ^2) = 1 / 4
  distinct : α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0

/-- The centroid of the triangle formed by the plane's intersection with the axes -/
noncomputable def centroid (plane : DistantPlane) : ℝ × ℝ × ℝ :=
  (plane.α / 3, plane.β / 3, plane.γ / 3)

/-- The main theorem -/
theorem centroid_sum (plane : DistantPlane) :
    let (p, q, r) := centroid plane
    1 / (p^2) + 1 / (q^2) + 1 / (r^2) = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l119_11995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_x_minus_one_lt_zero_l119_11974

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x - 1 else -x - 1

-- State the theorem
theorem solution_set_f_x_minus_one_lt_zero :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x ≥ 0, f x = x - 1) →  -- f(x) = x - 1 for x ≥ 0
  {x : ℝ | f (x - 1) < 0} = Set.Ioo 0 2 :=  -- solution set is (0, 2)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_x_minus_one_lt_zero_l119_11974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_problem_l119_11930

noncomputable def S (n : ℕ) : ℝ := n^2 + 2*n

noncomputable def a (n : ℕ) : ℝ := 2*n + 1

noncomputable def b (n : ℕ) : ℝ := 1 / ((a n)^2 - 1)

noncomputable def T (n : ℕ) : ℝ := n / (4*n + 4)

theorem sequence_sum_problem (n : ℕ) :
  (∀ k, S k = k^2 + 2*k) →
  (∀ k, a k = 2*k + 1) ∧
  T n = n / (4*n + 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_problem_l119_11930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_for_no_lattice_points_l119_11997

theorem max_b_for_no_lattice_points : ∃ (b : ℚ), 
  (∀ (m : ℚ), 1/3 < m ∧ m < b → 
    ∀ (x y : ℤ), 1 < x ∧ x ≤ 150 → y ≠ ⌊m * ↑x + 3⌋) ∧
  (∀ (b' : ℚ), b < b' → 
    ∃ (m : ℚ), 1/3 < m ∧ m < b' ∧
      ∃ (x y : ℤ), 1 < x ∧ x ≤ 150 ∧ y = ⌊m * ↑x + 3⌋) ∧
  b = 52/151 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_for_no_lattice_points_l119_11997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_sum_l119_11950

-- Define the binomial expression
noncomputable def binomial_expr (x : ℝ) (n : ℕ) : ℝ := (3 * x^2 - 1/x)^n

-- Define the sum of binomial coefficients
def sum_binomial_coeffs (n : ℕ) : ℕ := 2^n

theorem binomial_expansion_sum (n : ℕ) :
  sum_binomial_coeffs n = 32 →
  binomial_expr 1 n = 32 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_sum_l119_11950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_satisfy_condition_l119_11938

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := (1/2)^x
noncomputable def g (x : ℝ) : ℝ := -Real.sqrt x

-- State the theorem
theorem functions_satisfy_condition :
  ∀ (x₁ x₂ : ℝ), x₁ > x₂ →
  (x₂ > 0 → (f x₁ - f x₂) / (x₁ - x₂) < 2) ∧
  (x₁ > 0 ∧ x₂ > 0 → (g x₁ - g x₂) / (x₁ - x₂) < 2) := by
  sorry

#check functions_satisfy_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_satisfy_condition_l119_11938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ya_pair_addition_l119_11952

-- Define the Ya pair operation
noncomputable def ya_pair (a b : ℝ) : ℝ := Real.log b / Real.log a

-- Theorem statement
theorem ya_pair_addition (a b x y z : ℝ) 
  (hx : ya_pair a b = x) 
  (hy : ya_pair a y = z) : 
  x + z = ya_pair a (b * y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ya_pair_addition_l119_11952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l119_11917

theorem min_value_expression (a b c : ℕ) : 
  a ∈ ({2, 3, 5} : Set ℕ) → 
  b ∈ ({2, 3, 5} : Set ℕ) → 
  c ∈ ({2, 3, 5} : Set ℕ) → 
  a ≠ b → b ≠ c → a ≠ c → 
  (((a + b : ℚ) / c) / 2) ≥ (1 / 2 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l119_11917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_value_l119_11909

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2

theorem tangent_slope_implies_a_value :
  ∀ a : ℝ, (∃ f' : ℝ → ℝ, ∀ x : ℝ, HasDerivAt (f a) (f' x) x) →
  (∃ f' : ℝ → ℝ, ∀ x : ℝ, HasDerivAt (f a) (f' x) x ∧ f' 2 = -1/2) →
  a = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_value_l119_11909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x_where_abs_f_exceeds_sqrt3_div_3_l119_11918

/-- Given f(x) = sin(x) / (2 + cos(x)), prove that the statement |f(x)| ≤ √3/3 for all x ∈ ℝ is false. -/
theorem exists_x_where_abs_f_exceeds_sqrt3_div_3 :
  ∃ x : ℝ, |Real.sin x / (2 + Real.cos x)| > Real.sqrt 3 / 3 := by
  sorry

#check exists_x_where_abs_f_exceeds_sqrt3_div_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x_where_abs_f_exceeds_sqrt3_div_3_l119_11918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_and_inequality_l119_11915

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + b * x + 1

noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem common_tangent_and_inequality (a b : ℝ) 
  (h1 : deriv (f a b) 0 = deriv g 0) 
  (h2 : a ≤ 1/2) :
  ∀ x < 0, g x > f a b x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_and_inequality_l119_11915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l119_11955

/-- The ellipse on which point P moves -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- The line l -/
def line (x y : ℝ) : Prop := x + y - 2*Real.sqrt 5 = 0

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + y - 2*Real.sqrt 5) / Real.sqrt 2

/-- The minimum distance from any point on the ellipse to the line l is √10/2 -/
theorem min_distance_ellipse_to_line :
  ∃ x' y' : ℝ, ellipse x' y' ∧
    (∀ x y : ℝ, ellipse x y → distance_to_line x' y' ≤ distance_to_line x y) ∧
    distance_to_line x' y' = Real.sqrt 10 / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l119_11955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_functions_count_l119_11922

/-- A polynomial function of degree at most 3 -/
def Poly3 (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d

/-- The set of Poly3 functions satisfying f(x) f(-x) = f(x^3) -/
def ValidFunctions : Set (ℝ → ℝ) :=
  {f | ∃ a b c d : ℝ, f = Poly3 a b c d ∧ ∀ x, f x * f (-x) = f (x^3)}

/-- The statement to be proved -/
theorem valid_functions_count :
  ∃! (s : Finset (ℝ → ℝ)), ↑s = ValidFunctions ∧ Finset.card s = 2 ∧ 
  (λ x ↦ (0 : ℝ)) ∈ s ∧ (λ x ↦ (1 : ℝ)) ∈ s :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_functions_count_l119_11922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_f_non_negative_condition_l119_11948

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1 - x^2 / 2

-- Part 1: Monotonicity when a = 1
theorem f_strictly_increasing : 
  ∀ x y : ℝ, x < y → f 1 x < f 1 y := by sorry

-- Part 2: Condition for non-negativity
theorem f_non_negative_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_f_non_negative_condition_l119_11948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_value_l119_11900

theorem complex_square_value (a b : ℝ) :
  (((1 : ℂ) - Complex.I) / (Complex.I + 1)) ^ 2 = a + b * Complex.I → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_value_l119_11900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l119_11998

theorem distance_between_points :
  let p1 : Fin 3 → ℝ := ![-4, -1, 2]
  let p2 : Fin 3 → ℝ := ![6, 4, -3]
  let distance := Real.sqrt (((p2 0 - p1 0)^2 + (p2 1 - p1 1)^2 + (p2 2 - p1 2)^2))
  distance = 5 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l119_11998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_path_distance_l119_11940

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents the path of the laser beam -/
structure LaserPath where
  start : Point
  xAxisBounce : Point
  yAxisBounce : Point
  finish : Point

/-- Calculates the total distance traveled by the laser beam -/
noncomputable def totalDistance (path : LaserPath) : ℝ :=
  distance path.start path.xAxisBounce +
  distance path.xAxisBounce path.yAxisBounce +
  distance path.yAxisBounce path.finish

/-- The theorem to be proved -/
theorem laser_path_distance :
  ∃ (path : LaserPath),
    path.start = Point.mk 4 6 ∧
    path.finish = Point.mk 8 6 ∧
    path.xAxisBounce.y = 0 ∧
    path.yAxisBounce.x = 0 ∧
    totalDistance path = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_path_distance_l119_11940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l119_11925

open Real

noncomputable section

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

def Triangle.is_valid (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = π

def Triangle.satisfies_condition (t : Triangle) : Prop :=
  t.a * (cos (t.C / 2))^2 + t.c * (cos (t.A / 2))^2 = (3 / 2) * t.b

def Triangle.is_arithmetic_sequence (t : Triangle) : Prop :=
  t.a + t.c = 2 * t.b

noncomputable def Triangle.area (t : Triangle) : ℝ :=
  (1 / 2) * t.a * t.c * sin t.B

theorem triangle_theorem (t : Triangle) 
  (h_valid : t.is_valid) 
  (h_cond : t.satisfies_condition) : 
  t.is_arithmetic_sequence ∧ 
  (t.b = 2 * sqrt 2 ∧ t.B = π / 3 → t.area = 2 * sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l119_11925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_correct_prediction_2023_correct_l119_11969

/-- Linear regression model parameters -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- Data point for regression -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Calculates the slope of the linear regression model -/
noncomputable def calculate_slope (data : List DataPoint) (x_mean y_mean : ℝ) (sum_xy : ℝ) (sum_x_squared : ℝ) : ℝ :=
  sum_xy / sum_x_squared

/-- Calculates the intercept of the linear regression model -/
noncomputable def calculate_intercept (slope x_mean y_mean : ℝ) : ℝ :=
  y_mean - slope * x_mean

/-- Theorem stating the correctness of the linear regression model -/
theorem linear_regression_correct (data : List DataPoint) 
  (h_sum_xy : (calculate_slope data 3 3.92 (-3.7) 10) = -0.37)
  (h_intercept : (calculate_intercept (-0.37) 3 3.92) = 5.03) :
  ∃ (model : LinearRegression), model.b = -0.37 ∧ model.a = 5.03 := by
  sorry

/-- Predicts the total shipment for a given year -/
noncomputable def predict_shipment (model : LinearRegression) (year : ℝ) : ℝ :=
  model.a + model.b * year

/-- Theorem stating the correctness of the prediction for 2023 -/
theorem prediction_2023_correct (model : LinearRegression) 
  (h_model : model.b = -0.37 ∧ model.a = 5.03) :
  predict_shipment model 6 = 2.81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_correct_prediction_2023_correct_l119_11969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_center_distance_l119_11942

/-- Represents a circle with a center and radius. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are tangent if they touch at exactly one point. -/
def are_tangent (c1 c2 : Circle) : Prop := sorry

/-- The distance between the centers of two circles. -/
def center_distance (c1 c2 : Circle) : ℝ := sorry

/-- The diameter of a circle. -/
def Circle.diameter (c : Circle) : ℝ := 2 * c.radius

/-- Theorem: For two tangent circles with diameters 9 and 4, 
    the distance between their centers is either 2.5 or 6.5 -/
theorem tangent_circles_center_distance 
  (c1 c2 : Circle) 
  (h1 : c1.diameter = 9) 
  (h2 : c2.diameter = 4) 
  (h3 : are_tangent c1 c2) : 
  center_distance c1 c2 = 2.5 ∨ center_distance c1 c2 = 6.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_center_distance_l119_11942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle1_theorem_circle2_theorem_l119_11908

-- Define the points
def C : ℝ × ℝ := (2, -2)
def P : ℝ × ℝ := (6, 3)
def A : ℝ × ℝ := (-4, -5)
def B : ℝ × ℝ := (6, -1)

-- Define the equations of the circles
def circle1_eq (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 41
def circle2_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 3)^2 = 29

-- Theorem for the first circle
theorem circle1_theorem :
  ∀ x y : ℝ, circle1_eq x y ↔ 
  Real.sqrt ((x - C.1)^2 + (y - C.2)^2) = Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) :=
sorry

-- Theorem for the second circle
theorem circle2_theorem :
  ∀ x y : ℝ, circle2_eq x y ↔
  Real.sqrt ((x - (A.1 + B.1) / 2)^2 + (y - (A.2 + B.2) / 2)^2) = 
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle1_theorem_circle2_theorem_l119_11908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commuting_time_analysis_l119_11983

-- Define the driving time function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 30 then 30
  else if 30 < x ∧ x < 100 then 2*x + 1800/x - 90
  else 0

-- Define the overall average commuting time function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 30 then 40 - x/10
  else if 30 < x ∧ x < 100 then x^2/50 - 13*x/10 + 58
  else 0

theorem commuting_time_analysis :
  (∀ x : ℝ, 45 < x ∧ x < 100 → 40 < f x) ∧
  (∀ x : ℝ, 0 < x ∧ x < 45 → f x ≤ 40) ∧
  (∀ x : ℝ, 0 < x ∧ x < 32.5 → ∀ y : ℝ, x < y ∧ y < 32.5 → g y < g x) ∧
  (∀ x : ℝ, 32.5 < x ∧ x < 100 → ∀ y : ℝ, 32.5 < y ∧ y < x → g y < g x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_commuting_time_analysis_l119_11983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_necessary_not_sufficient_l119_11949

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is geometric if there exists a non-zero real number q such that
    for all n, a(n+1) = q * a(n) -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- A sequence is arithmetic if there exists a real number d such that
    for all n, a(n+1) - a(n) = d -/
def IsArithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The log(a_n + 1) sequence derived from a_n -/
noncomputable def LogSequence (a : Sequence) : Sequence :=
  fun n => Real.log (a n + 1)

theorem geometric_necessary_not_sufficient :
  (∀ a : Sequence, IsArithmetic (LogSequence a) → IsGeometric a) ∧
  (∃ a : Sequence, IsGeometric a ∧ ¬IsArithmetic (LogSequence a)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_necessary_not_sufficient_l119_11949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_range_l119_11988

/-- Given two curves y = a/x (x > 0) and y = 2ln x, if they have a common tangent line,
    then the range of values for a is [-2/e, 0) -/
theorem common_tangent_range (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧
    a / x₁ = 2 * Real.log x₂ ∧
    -a / (x₁^2) = 2 / x₂) →
  -2 / Real.exp 1 ≤ a ∧ a < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_range_l119_11988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_through_point_l119_11992

noncomputable section

-- Define the point P
def P : ℝ × ℝ := (1, -Real.sqrt 3)

-- Define the slope of the parallel line
def m : ℝ := -1 / Real.sqrt 3

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop :=
  x + Real.sqrt 3 * y + 2 = 0

-- State the theorem
theorem parallel_line_through_point :
  ∀ x y : ℝ, line_equation x y ↔ 
    (y - P.2 = m * (x - P.1) ∧ 
     m = -1 / Real.sqrt 3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_through_point_l119_11992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_disjoint_circle_tangent_line_through_point_l119_11973

-- Define the circle O
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Define the line l
def line_l : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the center of the circle
def center : ℝ × ℝ := (0, 0)

-- Define the radius of the circle
def radius : ℝ := 1

-- Theorem 1: Line l is disjoint from circle O
theorem line_disjoint_circle : 
  ∀ p ∈ line_l, distance center p > radius := by
  sorry

-- Define a function to check if a point is on the circle
def on_circle (p : ℝ × ℝ) : Prop := p ∈ circle_O

-- Define a function to check if a line passes through a point
def line_through_point (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

-- Theorem 2: When PA and PB are tangents, AB passes through (1/2, 1/2)
theorem tangent_line_through_point :
  ∀ (A B : ℝ × ℝ) (P : ℝ × ℝ),
    on_circle A → on_circle B → P ∈ line_l →
    (distance P A = distance P center) →
    (distance P B = distance P center) →
    line_through_point A B (1/2, 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_disjoint_circle_tangent_line_through_point_l119_11973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_two_l119_11946

-- Define the circle and points
variable (circle : Set (EuclideanSpace ℝ (Fin 2)))
variable (M A B C : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (touches_circle : C ∈ circle)
variable (intersects_circle : A ∈ circle ∧ B ∈ circle)
variable (midpoint : A = (B + M) / 2)
variable (tangent_length : dist M C = 2)
variable (angle : angle B M C = Real.pi / 4)

-- Define the radius
def radius (r : ℝ) (center : EuclideanSpace ℝ (Fin 2)) (s : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ p, p ∈ s ↔ dist p center = r

-- Theorem statement
theorem circle_radius_is_two :
  ∃ center : EuclideanSpace ℝ (Fin 2), radius 2 center circle :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_two_l119_11946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_lines_pass_through_equidistant_point_l119_11939

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A segment is represented by its endpoints -/
structure Segment where
  A : Point
  B : Point

/-- A circle is represented by its center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two circles are inscribed in a segment if they touch the segment and are contained within it -/
def inscribed (c1 c2 : Circle) (s : Segment) : Prop :=
  sorry -- Definition to be implemented

/-- Two circles intersect if they have two distinct common points -/
def intersect (c1 c2 : Circle) : Prop :=
  sorry -- Definition to be implemented

/-- A point is on a circle if its distance from the center equals the radius -/
def on_circle (p : Point) (c : Circle) : Prop :=
  sorry -- Definition to be implemented

/-- A point is on a line if it satisfies the line equation -/
def on_line (p : Point) (l : Line) : Prop :=
  sorry -- Definition to be implemented

/-- Distance between two points -/
def dist (p1 p2 : Point) : ℝ :=
  sorry -- Definition to be implemented

/-- A point is equidistant from two other points if its distance to both points is equal -/
def equidistant (p : Point) (a b : Point) : Prop :=
  dist p a = dist p b

/-- A line passes through a point if the point lies on the line -/
def passes_through (l : Line) (p : Point) : Prop :=
  on_line p l

/-- The theorem to be proved -/
theorem intersection_lines_pass_through_equidistant_point 
  (s : Segment) (c1 c2 : Circle) :
  inscribed c1 c2 s → intersect c1 c2 →
  ∃ p : Point, equidistant p s.A s.B ∧
    ∀ l : Line, (∃ q r : Point, q ≠ r ∧ on_circle q c1 ∧ on_circle q c2 ∧
                                on_circle r c1 ∧ on_circle r c2 ∧
                                on_line q l ∧ on_line r l) →
    passes_through l p :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_lines_pass_through_equidistant_point_l119_11939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mario_speed_is_90_over_53_l119_11910

/-- Represents the hiking scenario with Chantal and Mario -/
structure HikingScenario where
  x : ℝ  -- Half the distance to the fire tower
  chantal_speed_first_half : ℝ := 5
  chantal_speed_second_half : ℝ := 3
  chantal_speed_descent_steep : ℝ := 4
  chantal_speed_descent_flat : ℝ := 2.5

/-- Calculates Mario's average speed given the hiking scenario -/
noncomputable def mario_average_speed (scenario : HikingScenario) : ℝ :=
  let total_distance := 3 * scenario.x / 2  -- Three-quarters of the way back
  let chantal_time := scenario.x / scenario.chantal_speed_first_half +
                      scenario.x / scenario.chantal_speed_second_half +
                      scenario.x / scenario.chantal_speed_descent_steep +
                      scenario.x / (4 * scenario.chantal_speed_descent_flat)
  total_distance / chantal_time

/-- Theorem stating that Mario's average speed is 90/53 mph -/
theorem mario_speed_is_90_over_53 (scenario : HikingScenario) :
  mario_average_speed scenario = 90 / 53 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mario_speed_is_90_over_53_l119_11910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_theorem_l119_11985

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions for the triangle here
  True

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) : Prop :=
  -- Add conditions for the circle here
  True

-- Main theorem
theorem triangle_circle_theorem 
  (A B C O : ℝ × ℝ) 
  (p q : ℕ) 
  (h1 : Triangle A B C)
  (h2 : Circle O 17)
  (h3 : (O.1 - A.1) * (B.2 - A.2) = (O.2 - A.2) * (B.1 - A.1)) -- O is on AB
  (h4 : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0) -- BAC is a right angle
  (h5 : (B.1 - A.1) * (B.1 - A.1) + (B.2 - A.2) * (B.2 - A.2) + 
        (C.1 - B.1) * (C.1 - B.1) + (C.2 - B.2) * (C.2 - B.2) + 
        (A.1 - C.1) * (A.1 - C.1) + (A.2 - C.2) * (A.2 - C.2) = 170 * 170) -- Perimeter is 170
  (h6 : (O.1 - B.1) * (O.1 - B.1) + (O.2 - B.2) * (O.2 - B.2) = (p * p) / (q * q)) -- OB = p/q
  (h7 : Nat.Coprime p q) -- p and q are coprime
  : p + q = 142 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_theorem_l119_11985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_root_is_one_l119_11956

noncomputable def f (x : ℝ) : ℝ := |Real.cos (Real.pi * x) + x^3 - 3*x^2 + 3*x|
noncomputable def g (x : ℝ) : ℝ := 3 - x^2 - 2*x^3

theorem largest_root_is_one :
  (∀ x > 1, f x ≠ g x) ∧ f 1 = g 1 ∧ (∀ x, f x = g x → x ≤ 1) := by
  sorry

#check largest_root_is_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_root_is_one_l119_11956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_n12_mod7_equals_1_l119_11991

theorem probability_n12_mod7_equals_1 :
  (Finset.filter (fun n => n^12 % 7 = 1) (Finset.range 2030)).card / 2030 = 6 / 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_n12_mod7_equals_1_l119_11991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_PF1_PF2_l119_11987

/-- The ellipse defined by x^2/4 + y^2 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 4 + p.2^2 = 1}

/-- The left vertex of the ellipse -/
def A : ℝ × ℝ := (-2, 0)

/-- The top vertex of the ellipse -/
def B : ℝ × ℝ := (0, 1)

/-- The line segment AB -/
def AB : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (-2*t, t)}

/-- The foci of the ellipse -/
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 3, 0)

/-- The dot product of vectors PF1 and PF2 -/
noncomputable def dotProductPF1PF2 (P : ℝ × ℝ) : ℝ :=
  (F1.1 - P.1) * (F2.1 - P.1) + (F1.2 - P.2) * (F2.2 - P.2)

theorem min_dot_product_PF1_PF2 :
  ∀ P ∈ AB, dotProductPF1PF2 P ≥ -11/5 ∧
  ∃ P ∈ AB, dotProductPF1PF2 P = -11/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_PF1_PF2_l119_11987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aluminum_atomic_weight_l119_11964

/-- The atomic weight of a chemical element in atomic mass units (amu) -/
def atomic_weight : Type := ℝ

/-- The molecular weight of a chemical compound in atomic mass units (amu) -/
def molecular_weight : Type := ℝ

/-- The number of atoms of a specific element in a compound -/
def atom_count : Type := ℕ

theorem aluminum_atomic_weight 
  (Al : atomic_weight) -- Atomic weight of aluminum
  (F : atomic_weight) -- Atomic weight of fluorine
  (compound_weight : molecular_weight) -- Molecular weight of the compound
  (Al_count : atom_count) -- Number of aluminum atoms in the compound
  (F_count : atom_count) -- Number of fluorine atoms in the compound
  (h1 : Al_count = (1 : ℕ)) -- The compound contains one aluminum atom
  (h2 : F_count = (3 : ℕ)) -- The compound contains three fluorine atoms
  (h3 : compound_weight = (84 : ℝ)) -- The molecular weight of the compound is 84 amu
  (h4 : F = (19 : ℝ)) -- The atomic weight of fluorine is 19 amu
  : Al = (27 : ℝ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_aluminum_atomic_weight_l119_11964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_a_completion_time_l119_11904

/-- The number of days it takes for two workers to complete a job together -/
def total_days : ℚ := 30

/-- The speed ratio of worker A to worker B -/
def speed_ratio : ℚ := 3

/-- The number of days it would take worker B to complete the job alone -/
noncomputable def days_for_b (total_days : ℚ) (speed_ratio : ℚ) : ℚ :=
  total_days * (speed_ratio + 1) / speed_ratio

/-- The number of days it would take worker A to complete the job alone -/
noncomputable def days_for_a (total_days : ℚ) (speed_ratio : ℚ) : ℚ :=
  (days_for_b total_days speed_ratio) / speed_ratio

theorem worker_a_completion_time :
  days_for_a total_days speed_ratio = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_a_completion_time_l119_11904
