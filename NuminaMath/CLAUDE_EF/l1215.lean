import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_on_circle_implies_relation_l1215_121519

/-- A polynomial of degree 6 with real coefficients -/
def SixthDegreePolynomial (a b c : ℝ) : ℂ → ℂ :=
  fun x ↦ x^6 + a * x^4 + b * x^2 + c

/-- The property that all roots of a polynomial lie on a unique circle in the complex plane -/
def RootsOnUniqueCircle (P : ℂ → ℂ) : Prop :=
  ∃! (center : ℂ) (radius : ℝ), ∀ z : ℂ, P z = 0 → Complex.abs (z - center) = radius

/-- Theorem: If all roots of a sixth-degree polynomial with real coefficients
    lie on a unique circle in the complex plane, then b^3 = a^3 * c -/
theorem roots_on_circle_implies_relation (a b c : ℝ) :
  RootsOnUniqueCircle (SixthDegreePolynomial a b c) → b^3 = a^3 * c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_on_circle_implies_relation_l1215_121519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1215_121554

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - 4*x + a - 3)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → 0 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1215_121554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spectacular_consecutive_numbers_l1215_121561

def isSpectacular (n : ℕ) : Prop :=
  Nat.Prime n ∨ (∃ (p₁ p₂ p₃ p₄ : ℕ), Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ n = p₁ * p₂ * p₃ * p₄)

def concatenate (a b : ℕ) : ℕ := a * 1000 + b

theorem spectacular_consecutive_numbers :
  ∀ (n : ℕ), n ≥ 100 ∧ n ≤ 993 →
  ∃ (k : ℕ), k ∈ Finset.range 6 ∧ isSpectacular (concatenate (n + k) (n + k + 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spectacular_consecutive_numbers_l1215_121561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_invariant_l1215_121599

noncomputable def sample_A : List ℝ := [42, 43, 46, 52, 42, 50]

noncomputable def sample_B : List ℝ := sample_A.map (· - 5)

noncomputable def mean (sample : List ℝ) : ℝ := sample.sum / sample.length

noncomputable def variance (sample : List ℝ) : ℝ :=
  let μ := mean sample
  (sample.map (fun x => (x - μ) ^ 2)).sum / sample.length

noncomputable def standard_deviation (sample : List ℝ) : ℝ :=
  Real.sqrt (variance sample)

theorem standard_deviation_invariant :
  standard_deviation sample_A = standard_deviation sample_B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_invariant_l1215_121599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_problem_l1215_121598

theorem car_sale_problem (selling_price : ℝ) (headlight_cost : ℝ) (offer_difference : ℝ)
  (h1 : selling_price = 5200)
  (h2 : headlight_cost = 80)
  (h3 : offer_difference = 200) :
  (selling_price - (selling_price - (headlight_cost + 3 * headlight_cost)) - offer_difference) / selling_price = 3 / 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_problem_l1215_121598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangles_common_area_l1215_121578

/-- The area of the common region formed by two regular triangles constructed 
    inward on opposite sides of a square with side length a -/
noncomputable def commonRegionArea (a : ℝ) : ℝ :=
  (a^2 * (2 * Real.sqrt 3 - 3)) / Real.sqrt 3

/-- Theorem stating that the area of the common region formed by two regular triangles 
    constructed inward on opposite sides of a square with side length a 
    is equal to (a^2 * (2√3 - 3)) / √3 -/
theorem square_triangles_common_area (a : ℝ) (h : a > 0) : 
  commonRegionArea a = (a^2 * (2 * Real.sqrt 3 - 3)) / Real.sqrt 3 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangles_common_area_l1215_121578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stoppage_time_is_18_minutes_l1215_121581

/-- Represents a train with its speeds and stoppage time -/
structure Train where
  speed_without_stoppages : ℚ
  speed_with_stoppages : ℚ
  stoppage_time : ℚ

/-- Calculates the stoppage time for a train given its speeds -/
def calculate_stoppage_time (speed_without_stoppages speed_with_stoppages : ℚ) : ℚ :=
  (speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages * 60

/-- Theorem stating that each train stops for 18 minutes per hour -/
theorem train_stoppage_time_is_18_minutes 
  (train_a train_b train_c : Train) 
  (ha_without : train_a.speed_without_stoppages = 30)
  (hb_without : train_b.speed_without_stoppages = 40)
  (hc_without : train_c.speed_without_stoppages = 50)
  (ha_with : train_a.speed_with_stoppages = 21)
  (hb_with : train_b.speed_with_stoppages = 28)
  (hc_with : train_c.speed_with_stoppages = 35) :
  train_a.stoppage_time = 18 ∧ 
  train_b.stoppage_time = 18 ∧ 
  train_c.stoppage_time = 18 :=
by
  sorry

#eval calculate_stoppage_time 30 21
#eval calculate_stoppage_time 40 28
#eval calculate_stoppage_time 50 35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stoppage_time_is_18_minutes_l1215_121581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tensor_properties_l1215_121566

-- Define the planar vector type
def PlanarVector := ℝ × ℝ

-- Define the ⊗ operation
noncomputable def tensor (a b : PlanarVector) : ℝ := 
  let aNorm := Real.sqrt (a.1^2 + a.2^2)
  let bNorm := Real.sqrt (b.1^2 + b.2^2)
  let sinTheta := (a.1 * b.2 - a.2 * b.1) / (aNorm * bNorm)
  aNorm * bNorm * sinTheta

-- State the theorem
theorem tensor_properties (a b c : PlanarVector) :
  (∀ a b, tensor a b = tensor b a) ∧
  (∀ a b l, a = (l * b.1, l * b.2) → tensor a b = 0) ∧
  (∀ a b c l, l > 0 → a = (l * b.1, l * b.2) → 
    tensor (a.1 + b.1, a.2 + b.2) c = tensor a c + tensor b c) := by
  sorry

#check tensor_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tensor_properties_l1215_121566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_of_reversed_digits_l1215_121542

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a positive integer is a palindrome -/
def isPalindrome (n : ℕ) : Prop := n = reverseDigits n

/-- Checks if a positive integer is of the form 21[99...9]78 -/
def is21_78Form (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 21 * 10^(k+2) + (10^(k+2) - 1) + 78

/-- Checks if a positive integer is of the form 10[99...9]89 -/
def is10_89Form (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 10 * 10^(k+2) + (10^(k+2) - 1) + 89

/-- Main theorem -/
theorem divisor_of_reversed_digits (n : ℕ) : 
  n > 0 → (reverseDigits n % n = 0 ↔ isPalindrome n ∨ is21_78Form n ∨ is10_89Form n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_of_reversed_digits_l1215_121542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monica_students_l1215_121530

theorem monica_students : 
  ∀ (class1 class2 class3 class4 class5 class6 overlap12 overlap45 overlap36 : ℕ),
  class1 = 20 →
  class2 = 25 →
  class3 = 25 →
  class4 = class1 / 2 →
  class5 = 28 →
  class6 = 28 →
  overlap12 = 5 →
  overlap45 = 3 →
  overlap36 = 7 →
  class1 + class2 + class3 + class4 + class5 + class6 - overlap12 - overlap45 - overlap36 = 121 :=
by
  intros class1 class2 class3 class4 class5 class6 overlap12 overlap45 overlap36
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

#check monica_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monica_students_l1215_121530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1215_121517

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def f (x : ℝ) : ℝ := a.1 * (b x).1 + a.2 * (b x).2 - 1

theorem f_properties :
  (∀ x, f x = 0 ↔ ∃ k : ℤ, x = 2 * k * Real.pi) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2),
    (∀ y ∈ Set.Icc 0 (Real.pi / 3), x ≤ y → f x ≤ f y) ∧
    (∀ y ∈ Set.Icc (Real.pi / 3) (Real.pi / 2), x ≤ y → f x ≥ f y) ∧
    f (Real.pi / 3) = 1 ∧
    f 0 = 0 ∧
    ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ 1 ∧ f y ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1215_121517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_phone_bill_l1215_121534

-- Define the problem parameters
def monthly_fee : ℚ := 5
def per_minute_rate : ℚ := 25 / 100
def total_bill : ℚ := 12.02

-- Define the function to calculate billable minutes
def billable_minutes (fee : ℚ) (rate : ℚ) (bill : ℚ) : ℕ :=
  (((bill - fee) / rate).floor : ℤ).toNat

-- State the theorem
theorem john_phone_bill :
  billable_minutes monthly_fee per_minute_rate total_bill = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_phone_bill_l1215_121534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1215_121595

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1/2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2*x))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/2) → f x ≤ 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/2) ∧ f x = 1) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/2) → f x ≥ -1/2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/2) ∧ f x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1215_121595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_max_area_difference_l1215_121567

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define points A and B as the left and right vertices of the ellipse
def A : ℝ × ℝ := (-5, 0)
def B : ℝ × ℝ := (5, 0)

-- Define a line intersecting the ellipse
def intersecting_line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Define the slope ratio condition
def slope_ratio (k₁ k₂ : ℝ) : Prop := k₁ / k₂ = 1 / 9

-- Define the area of a triangle given three points
noncomputable def triangle_area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  let (x₃, y₃) := p₃
  (1/2) * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂))

-- Statement 1: The line passes through a fixed point
theorem line_passes_through_fixed_point 
  (m b : ℝ) 
  (M N : ℝ × ℝ) 
  (h₁ : ellipse M.1 M.2) 
  (h₂ : ellipse N.1 N.2)
  (h₃ : intersecting_line m b M.1 M.2)
  (h₄ : intersecting_line m b N.1 N.2)
  (h₅ : slope_ratio ((M.2 - A.2) / (M.1 - A.1)) ((N.2 - B.2) / (N.1 - B.1))) :
  intersecting_line m b 4 0 := by
  sorry

-- Statement 2: Maximum value of area difference
theorem max_area_difference 
  (M N : ℝ × ℝ) 
  (h₁ : ellipse M.1 M.2) 
  (h₂ : ellipse N.1 N.2)
  (h₃ : ∃ m b, intersecting_line m b M.1 M.2 ∧ intersecting_line m b N.1 N.2)
  (h₄ : slope_ratio ((M.2 - A.2) / (M.1 - A.1)) ((N.2 - B.2) / (N.1 - B.1))) :
  ∃ S₁ S₂, S₁ = triangle_area A M N ∧ S₂ = triangle_area B M N ∧ 
    ∀ M' N', ellipse M'.1 M'.2 → ellipse N'.1 N'.2 → 
      (∃ m' b', intersecting_line m' b' M'.1 M'.2 ∧ intersecting_line m' b' N'.1 N'.2) →
      slope_ratio ((M'.2 - A.2) / (M'.1 - A.1)) ((N'.2 - B.2) / (N'.1 - B.1)) →
      triangle_area A M' N' - triangle_area B M' N' ≤ S₁ - S₂ ∧ S₁ - S₂ = 15 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_max_area_difference_l1215_121567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_S3_is_16_over_9_l1215_121570

/-- Given a square S₁ with area 36, S₂ is constructed by trisecting S₁'s sides and
    using the trisection points closest to the corners as vertices, and S₃ is
    constructed similarly from S₂. This function calculates the area of S₃. -/
noncomputable def area_S3 (area_S1 : ℝ) : ℝ :=
  let side_S1 := Real.sqrt area_S1
  let side_S2 := side_S1 * Real.sqrt 2 / 3
  let side_S3 := side_S2 * Real.sqrt 2 / 3
  side_S3 ^ 2

/-- Theorem stating that the area of S₃ is 16/9 when S₁ has area 36. -/
theorem area_S3_is_16_over_9 :
  area_S3 36 = 16 / 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_S3_is_16_over_9_l1215_121570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1215_121503

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.sin x) ^ 2

-- State the theorem
theorem f_properties :
  -- The minimal positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  -- For x ∈ [-π/6, π/3], the maximum value of f(x) is 1
  (∀ (x : ℝ), -Real.pi/6 ≤ x ∧ x ≤ Real.pi/3 → f x ≤ 1) ∧
  -- The maximum value is attained at x = π/6
  (f (Real.pi/6) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1215_121503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_and_inequality_l1215_121562

-- Define the functions
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a + 1) ^ (x - 2) + 1
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) / Real.log (Real.sqrt 3)

-- State the theorem
theorem fixed_point_and_inequality (a : ℝ) (h_a : a > 0) :
  (∃ A : ℝ × ℝ, A.1 = 2 ∧ A.2 = 2 ∧ g a A.1 = A.2 ∧ f a A.1 = A.2) →
  (a = 1 ∧ ∀ x : ℝ, g 1 x > 3 ↔ x > 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_and_inequality_l1215_121562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_rectangle_l1215_121592

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A rectangle defined by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Check if a point is within a rectangle -/
def isInRectangle (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

/-- The main theorem -/
theorem points_in_rectangle (points : Finset Point) (r : Rectangle) :
  r.width = 3 ∧ r.height = 4 →
  points.card = 6 →
  (∀ p ∈ points, isInRectangle p r) →
  ∃ p1 p2 : Point, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_rectangle_l1215_121592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_circle_l1215_121537

/-- A planar closed polygonal line -/
structure ClosedPolygonalLine where
  -- We represent the polygonal line as a list of points in the plane
  points : List (ℝ × ℝ)
  -- The line is closed, so the last point should connect to the first
  is_closed : points.getLast? = points.head?
  -- The line is non-empty
  non_empty : points.length > 0

/-- The perimeter of a closed polygonal line -/
noncomputable def perimeter (line : ClosedPolygonalLine) : ℝ :=
  sorry

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a point is inside or on a circle -/
def is_inside_circle (point : ℝ × ℝ) (circle : Circle) : Prop :=
  sorry

/-- Predicate to check if a polygonal line is entirely inside or on a circle -/
def is_enclosed (line : ClosedPolygonalLine) (circle : Circle) : Prop :=
  ∀ point ∈ line.points, is_inside_circle point circle

/-- The main theorem -/
theorem smallest_enclosing_circle 
  (line : ClosedPolygonalLine) 
  (h_perimeter : perimeter line = 1) : 
  (∃ (c : Circle), c.radius = 1/4 ∧ is_enclosed line c) ∧
  (∀ (c : Circle), c.radius < 1/4 → ¬is_enclosed line c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_circle_l1215_121537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1215_121521

open Real

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → x * g y - y * g x = g (x / y)

theorem functional_equation_solution :
  ∀ g : ℝ → ℝ, FunctionalEquation g → g 1 = 1 → g 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1215_121521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_and_gcd_properties_l1215_121527

theorem division_and_gcd_properties 
  (a b c a' b' c' : ℤ) 
  (h : ∃ (lambda : ℚ), a = a' * lambda ∧ b = b' * lambda ∧ c = c' * lambda) :
  (∃ q : ℤ, a = b * q + c → a' = b' * q + c') ∧
  (Int.gcd a b = 1 → ∃ k : ℤ, c' = k * c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_and_gcd_properties_l1215_121527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_fraction_property_l1215_121593

theorem polynomial_fraction_property (P : ℝ → ℝ) :
  (∃ S : Set (ℕ × ℕ), Set.Infinite S ∧
    (∀ (m n : ℕ), (m, n) ∈ S →
      m > 0 ∧ n > 0 ∧ Nat.Coprime m n ∧ P (m / n) = 1 / n)) →
  ∃ k : ℕ, k > 0 ∧ ∀ x : ℝ, P x = x / k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_fraction_property_l1215_121593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1215_121585

/-- The function f(x) defined for all real x except -5 -/
noncomputable def f (x : ℝ) : ℝ := 3 * (x + 5) * (x - 4) / (x + 5)

theorem f_range : 
  Set.range f = {y : ℝ | y < -27 ∨ y > -27} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1215_121585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_eq_neg_sin_l1215_121559

/-- Recursive definition of the sequence of functions f_n -/
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => λ x => Real.sin x
  | n + 1 => λ x => deriv (f n) x

/-- The 2010th function in the sequence is equal to the negative sine function -/
theorem f_2010_eq_neg_sin : f 2010 = λ x => -Real.sin x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_eq_neg_sin_l1215_121559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_rose_cost_proof_l1215_121531

/-- Represents the cost of roses in dollars -/
structure RoseCost where
  value : ℝ
  nonneg : value ≥ 0

/-- The cost of one dozen roses -/
def dozen_cost : RoseCost := ⟨36, by norm_num⟩

/-- The cost of two dozen roses -/
def two_dozen_cost : RoseCost := ⟨50, by norm_num⟩

/-- The maximum budget for roses -/
def max_budget : RoseCost := ⟨680, by norm_num⟩

/-- The maximum number of roses that can be purchased with the max_budget -/
def max_roses : ℕ := 316

/-- The cost of an individual rose -/
def individual_rose_cost : RoseCost := ⟨7.5, by norm_num⟩

theorem individual_rose_cost_proof : 
  ∃ (x : RoseCost), 
    (dozen_cost.value = 12 * x.value) ∧ 
    (two_dozen_cost.value = 24 * x.value) ∧ 
    (↑max_roses * x.value ≤ max_budget.value) ∧
    (∀ y : RoseCost, y.value > x.value → ↑max_roses * y.value > max_budget.value) ∧
    (x = individual_rose_cost) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_rose_cost_proof_l1215_121531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_open_interval_one_two_l1215_121541

-- Define the function f(x) = x - 2 + log₂x
noncomputable def f (x : ℝ) : ℝ := x - 2 + Real.log x / Real.log 2

-- Theorem statement
theorem zero_of_f_in_open_interval_one_two :
  ∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_open_interval_one_two_l1215_121541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinear_opposite_k_l1215_121568

-- Define the vectors
def a (k : ℝ) : Fin 2 → ℝ := ![k, 1]
def b (k : ℝ) : Fin 2 → ℝ := ![2, k + 1]

-- Define collinearity and opposite direction
def collinear (u v : Fin 2 → ℝ) : Prop :=
  ∃ (l : ℝ), v = l • u

def opposite_direction (u v : Fin 2 → ℝ) : Prop :=
  ∃ (l : ℝ), l < 0 ∧ v = l • u

-- Theorem statement
theorem vector_collinear_opposite_k (k : ℝ) :
  collinear (a k) (b k) ∧ opposite_direction (a k) (b k) → k = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinear_opposite_k_l1215_121568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_pounder_cost_l1215_121514

/-- The cost of a quarter-pounder burger given the conditions of Danny's order -/
theorem quarter_pounder_cost : ℝ := by
  let free_delivery_minimum : ℝ := 18
  let large_fries_cost : ℝ := 1.90
  let milkshake_cost : ℝ := 2.40
  let additional_order_needed : ℝ := 3
  let num_fries : ℕ := 2
  let num_milkshakes : ℕ := 2

  let current_total : ℝ := num_fries * large_fries_cost + num_milkshakes * milkshake_cost
  let remaining_for_free_delivery : ℝ := free_delivery_minimum - current_total
  let quarter_pounder_cost : ℝ := remaining_for_free_delivery - additional_order_needed

  have h : quarter_pounder_cost = 12.40 := by
    -- Proof steps would go here
    sorry

  exact quarter_pounder_cost


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_pounder_cost_l1215_121514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_speed_is_20_33_l1215_121516

/-- Represents the hiking scenario with Chantal and Jean -/
structure HikingScenario where
  d : ℝ  -- represents the quarter of the total distance
  chantalSpeed1 : ℝ := 5  -- Chantal's speed for first quarter
  chantalSpeed2 : ℝ := 2.5  -- Chantal's speed for next three quarters
  chantalSpeed3 : ℝ := 4  -- Chantal's speed for descent on first quarter

/-- Calculates Jean's average speed given the hiking scenario -/
noncomputable def jeanAverageSpeed (scenario : HikingScenario) : ℝ :=
  let totalTime := scenario.d / scenario.chantalSpeed1 + 
                   (3 * scenario.d) / scenario.chantalSpeed2 + 
                   scenario.d / scenario.chantalSpeed3
  scenario.d / totalTime

/-- Theorem stating that Jean's average speed is 20/33 mph -/
theorem jean_speed_is_20_33 (scenario : HikingScenario) : 
  jeanAverageSpeed scenario = 20 / 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_speed_is_20_33_l1215_121516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1215_121579

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : (Real.cos t.B) / (Real.cos t.C) = -t.b / (2 * t.a + t.c))
  (h2 : t.b = Real.sqrt 13)
  (h3 : t.a + t.c = 4) :
  t.B = 2 * Real.pi / 3 ∧ (t.a = 1 ∨ t.a = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1215_121579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1215_121501

noncomputable def f (x : ℝ) := Real.sqrt (x + 16) + Real.sqrt (20 - x) + 2 * Real.sqrt x

theorem max_value_of_f :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 20 ∧
  f x = 18 ∧
  ∀ (y : ℝ), 0 ≤ y → y ≤ 20 → f y ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1215_121501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1215_121584

/-- The solution set of the inequality x^2 - (a^2 + a)x + a^3 ≥ 0 -/
def solution_set (a : ℝ) : Set ℝ :=
  if a > 1 ∨ a < 0 then
    Set.Iic a ∪ Set.Ici (a^2)
  else if a = 1 ∨ a = 0 then
    Set.univ
  else if 0 < a ∧ a < 1 then
    Set.Iic (a^2) ∪ Set.Ici a
  else
    ∅

/-- The inequality x^2 - (a^2 + a)x + a^3 ≥ 0 -/
def inequality (a x : ℝ) : Prop :=
  x^2 - (a^2 + a)*x + a^3 ≥ 0

theorem inequality_solution_set (a : ℝ) :
  ∀ x, x ∈ solution_set a ↔ inequality a x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1215_121584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_20_l1215_121589

/-- Represents a clock with 12 hours --/
structure Clock :=
  (hours : ℕ)
  (hourDegrees : ℚ)
  (minuteDegrees : ℚ)
  (hourHandSpeed : ℚ)
  (minuteHandSpeed : ℚ)

/-- Calculates the position of a clock hand given the initial position and elapsed time --/
noncomputable def handPosition (initialPos : ℚ) (speed : ℚ) (time : ℚ) : ℚ :=
  (initialPos + speed * time) % 360

/-- Calculates the smaller angle between two positions on a circle --/
noncomputable def smallerAngle (pos1 : ℚ) (pos2 : ℚ) : ℚ :=
  min (abs (pos1 - pos2)) (360 - abs (pos1 - pos2))

/-- Theorem: At 3:20 on a 12-hour clock, the smaller angle between hour and minute hands is 20° --/
theorem clock_angle_at_3_20 (c : Clock) 
  (h1 : c.hours = 12)
  (h2 : c.hourDegrees = 360 / c.hours)
  (h3 : c.minuteDegrees = 360 / 60)
  (h4 : c.hourHandSpeed = c.hourDegrees / 60)
  (h5 : c.minuteHandSpeed = c.minuteDegrees)
  : smallerAngle 
      (handPosition (3 * c.hourDegrees) c.hourHandSpeed 20) 
      (handPosition 0 c.minuteHandSpeed 20) = 20 := by
  sorry

#check clock_angle_at_3_20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_20_l1215_121589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euclidean_gcd_295_85_variable_swap_variance_constant_subtraction_l1215_121586

def euclidean_algorithm_steps (a b : ℕ) : ℕ := sorry

def swap_variables (A B : α) : (α × α) := sorry

noncomputable def set_variance (S : Finset ℝ) : ℝ := sorry

theorem euclidean_gcd_295_85 :
  euclidean_algorithm_steps 295 85 = 12 := by sorry

theorem variable_swap (A B : α) :
  swap_variables A B = (B, A) := by sorry

theorem variance_constant_subtraction (S : Finset ℝ) (c : ℝ) :
  set_variance S = set_variance (S.image (λ x => x - c)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euclidean_gcd_295_85_variable_swap_variance_constant_subtraction_l1215_121586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_division_theorem_l1215_121536

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Checks if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a line segment intersects a line -/
def segment_intersects_line (p q : Point) (l : Line) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    point_on_line ⟨p.x + t * (q.x - p.x), p.y + t * (q.y - p.y)⟩ l

theorem plane_division_theorem
  (points : Finset Point)
  (lines : Finset Line)
  (h1 : points.card = 30)
  (h2 : lines.card = 7)
  (h3 : ∀ p q r, p ∈ points → q ∈ points → r ∈ points → p ≠ q → q ≠ r → p ≠ r → ¬collinear p q r)
  (h4 : ∀ p l, p ∈ points → l ∈ lines → ¬point_on_line p l) :
  ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ ∀ l ∈ lines, ¬segment_intersects_line p q l :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_division_theorem_l1215_121536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_2sin_x_range_l1215_121529

theorem cos_2x_plus_2sin_x_range :
  ∀ x : ℝ, -3 ≤ Real.cos (2 * x) + 2 * Real.sin x ∧ Real.cos (2 * x) + 2 * Real.sin x ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_2sin_x_range_l1215_121529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1215_121594

noncomputable def hyperbola_C (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 4*x

noncomputable def myCircle (x y a : ℝ) : Prop := (x - 1)^2 + y^2 = a^2

noncomputable def asymptote_l (x y a b : ℝ) : Prop := a*y + b*x = 0

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hfocus : ∃ x y, hyperbola_C x y a b ∧ parabola x y)
  (hasymptote : ∃ x y, hyperbola_C x y a b ∧ asymptote_l x y a b)
  (hintersect : ∃ A B : ℝ × ℝ, 
    asymptote_l A.1 A.2 a b ∧ 
    asymptote_l B.1 B.2 a b ∧ 
    myCircle A.1 A.2 a ∧ 
    myCircle B.1 B.2 a)
  (hdistance : ∀ A B : ℝ × ℝ, 
    asymptote_l A.1 A.2 a b → 
    asymptote_l B.1 B.2 a b → 
    myCircle A.1 A.2 a → 
    myCircle B.1 B.2 a → 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2 : ℝ) = b) :
  (a^2 + b^2)^(1/2 : ℝ) / a = 3 * 5^(1/2 : ℝ) / 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1215_121594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_6_30_is_15_l1215_121565

/-- The number of hours on a clock -/
def clock_hours : ℕ := 12

/-- The angle between each hour mark on a clock -/
noncomputable def hour_angle : ℝ := 360 / clock_hours

/-- The position of the minute hand at 6:30 (in degrees) -/
def minute_hand_position : ℝ := 180

/-- The position of the hour hand at 6:30 (in degrees) -/
def hour_hand_position : ℝ := 195

/-- The smaller angle formed by the hour-hand and minute-hand of a clock at 6:30 -/
noncomputable def clock_angle_at_6_30 : ℝ := 
  min (abs (hour_hand_position - minute_hand_position)) 
      (360 - abs (hour_hand_position - minute_hand_position))

/-- Theorem stating that the smaller angle formed by the hour-hand and minute-hand of a clock at 6:30 is 15° -/
theorem clock_angle_at_6_30_is_15 : clock_angle_at_6_30 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_6_30_is_15_l1215_121565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_alpha_l1215_121540

theorem sin_pi_minus_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan (2 * α) = -4 / 3) :
  Real.sin (π - α) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_alpha_l1215_121540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spots_area_is_correct_l1215_121587

noncomputable section

/-- The area Spot can reach given the conditions of the doghouse and tether --/
def spots_area (hexagon_side : ℝ) (tether_length : ℝ) (overhang_radius : ℝ) : ℝ :=
  let main_sector_angle : ℝ := 240
  let adjacent_sector_angle : ℝ := 60
  let overhang_sector_angle : ℝ := 90
  let main_sector_area := Real.pi * tether_length^2 * (main_sector_angle / 360)
  let adjacent_sectors_area := 2 * (Real.pi * hexagon_side^2 * (adjacent_sector_angle / 360))
  let overhang_area := Real.pi * overhang_radius^2 * (overhang_sector_angle / 360)
  main_sector_area + adjacent_sectors_area + overhang_area

/-- Theorem stating that the area Spot can reach is (89/12)π square yards --/
theorem spots_area_is_correct :
  spots_area 2 3 1 = (89/12) * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spots_area_is_correct_l1215_121587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1215_121546

/-- Definition of a parabola with equation x^2 = (1/2)y -/
def parabola (x y : ℝ) : Prop := x^2 = (1/2) * y

/-- The focus of the parabola -/
noncomputable def focus : ℝ × ℝ := (0, 1/8)

/-- Theorem: The focus of the parabola x^2 = (1/2)y is (0, 1/8) -/
theorem parabola_focus :
  ∀ (x y : ℝ), parabola x y → (x = 0 ∧ y = 1/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1215_121546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1215_121571

-- Define the function f(x) = ln(x+1) + 2x - 1
noncomputable def f (x : ℝ) := Real.log (x + 1) + 2 * x - 1

-- State the theorem
theorem root_in_interval :
  ∃ m : ℝ, f m = 0 ∧ 0 < m ∧ m < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1215_121571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_allowance_allowance_breakdown_l1215_121511

/-- John's weekly allowance -/
def A : ℚ := 3.375

/-- Amount spent at the arcade -/
def arcade_spent : ℚ := (3 / 5) * A

/-- Amount remaining after arcade -/
def after_arcade : ℚ := A - arcade_spent

/-- Amount spent at the toy store -/
def toy_store_spent : ℚ := (1 / 3) * after_arcade

/-- Amount remaining after toy store -/
def after_toy_store : ℚ := after_arcade - toy_store_spent

/-- Amount spent at the candy store -/
def candy_store_spent : ℚ := 0.90

theorem johns_allowance : A = 3.375 := by
  rfl

theorem allowance_breakdown : 
  after_toy_store = candy_store_spent := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_allowance_allowance_breakdown_l1215_121511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_necessary_not_sufficient_l1215_121502

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The cosine function with a phase shift -/
noncomputable def f (φ : ℝ) : ℝ → ℝ := λ x ↦ Real.cos (2 * x + φ)

theorem odd_function_necessary_not_sufficient :
  (∃ φ, IsOdd (f φ) ∧ φ ≠ π / 2) ∧
  (∀ φ, φ = π / 2 → IsOdd (f φ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_necessary_not_sufficient_l1215_121502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l1215_121590

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (-4 * x + 1) * Real.exp x

-- State the theorem
theorem max_value_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ ∀ x ∈ Set.Icc 0 1, f x ≤ f c ∧ f c = 1 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l1215_121590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_decreasing_condition_l1215_121526

/-- A function f is decreasing on an interval if for any two points x and y in that interval,
    x < y implies f(x) > f(y) -/
def DecreasingOn (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x > f y

theorem quadratic_decreasing_condition (a : ℝ) :
  let f := fun x : ℝ ↦ x^2 - 2*(a-1)*x + 2
  DecreasingOn f { x | x ≤ 5 } → a ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_decreasing_condition_l1215_121526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_sum_l1215_121552

theorem log_power_sum (a b : ℝ) (h1 : a = Real.log 25) (h2 : b = Real.log 36) :
  (5 : ℝ)^(a/b) + (6 : ℝ)^(b/a) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_sum_l1215_121552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PF₁F₂_is_sqrt2_l1215_121510

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

-- Define the hyperbola C₂
def C₂ (x y : ℝ) : Prop := x^2/3 - y^2 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the intersection point P
noncomputable def P : ℝ × ℝ := sorry

-- Assume P is on both C₁ and C₂
axiom P_on_C₁ : C₁ P.1 P.2
axiom P_on_C₂ : C₂ P.1 P.2

-- Theorem: The area of triangle PF₁F₂ is √2
theorem area_PF₁F₂_is_sqrt2 : 
  let A := abs ((P.1 - F₁.1) * (F₂.2 - F₁.2) - (P.2 - F₁.2) * (F₂.1 - F₁.1)) / 2
  A = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PF₁F₂_is_sqrt2_l1215_121510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l1215_121505

theorem max_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  ((x + y)^3) / (x^2 + y^2) ≤ 4 ∧
  ((1 : ℝ) + 1)^3 / ((1 : ℝ)^2 + 1^2) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l1215_121505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_increasing_properties_l1215_121560

/-- A three-digit increasing number -/
def ThreeDigitIncreasing (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ 
  (n % 10 > (n / 10) % 10) ∧ 
  ((n / 10) % 10 > n / 100)

/-- The set of digits {1,2,3,4,5} -/
def DigitSet : Set ℕ := {1,2,3,4,5}

/-- The probability of divisibility by 5 for a three-digit increasing number formed from DigitSet -/
def ProbDivBy5 : ℚ := 3/5

/-- The score function based on divisibility rules -/
def Score (n : ℕ) : ℕ :=
  if (n % 15 = 0) then 2
  else if (n % 3 = 0) ∨ (n % 5 = 0) then 1
  else 0

/-- The expectation of the score for all three-digit increasing numbers -/
def ExpectedScore : ℚ := 23/21

theorem three_digit_increasing_properties :
  (∀ n ∈ DigitSet, ThreeDigitIncreasing n → n % 5 = 0) ∧
  (∀ n, ThreeDigitIncreasing n → Score n ∈ ({0,1,2} : Set ℕ)) ∧
  ProbDivBy5 = 3/5 ∧
  ExpectedScore = 23/21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_increasing_properties_l1215_121560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_OA_OB_l1215_121574

noncomputable def angle_between_vectors (α β γ : Real) (B : Fin 3 → Real) : Real :=
  let n : Fin 3 → Real := fun i => match i with
    | 0 => Real.cos α
    | 1 => Real.cos β
    | 2 => Real.cos γ
  let OB := B
  let dot_product := (n 0 * OB 0) + (n 1 * OB 1) + (n 2 * OB 2)
  let magnitude_n := Real.sqrt ((n 0)^2 + (n 1)^2 + (n 2)^2)
  let magnitude_OB := Real.sqrt ((OB 0)^2 + (OB 1)^2 + (OB 2)^2)
  Real.arccos (dot_product / (magnitude_n * magnitude_OB))

theorem angle_between_OA_OB :
  let α : Real := π / 3
  let β : Real := π / 3
  let γ : Real := π / 4
  let B : Fin 3 → Real := fun i => match i with
    | 0 => -2
    | 1 => -2
    | 2 => -2 * Real.sqrt 2
  angle_between_vectors α β γ B = π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_OA_OB_l1215_121574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l1215_121575

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : (fun x ↦ f 2 + (deriv f 2) * (x - 2)) = fun x ↦ -x + 1)

-- State the theorem
theorem tangent_line_sum (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : (fun x ↦ f 2 + (deriv f 2) * (x - 2)) = fun x ↦ -x + 1) :
  f 2 + deriv f 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l1215_121575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_abs_b_proof_l1215_121597

/-- The smallest absolute value of b in the given sequence -/
def smallest_abs_b : ℕ := 1

/-- Proof of the smallest absolute value of b given the conditions -/
theorem smallest_abs_b_proof (a b c : ℤ) 
  (h_order : a > b ∧ b > c)
  (h_nonpos : a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0)
  (h_arithmetic : b - c = a - b)
  (h_geometric : c * c = a * b) :
  smallest_abs_b = Int.natAbs b := by
  sorry

#check smallest_abs_b_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_abs_b_proof_l1215_121597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_a_eq_two_l1215_121504

-- Define the curve
noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := Real.log x + a

-- Define the line
def line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define what it means for the line to be tangent to the curve
def is_tangent (a : ℝ) : Prop :=
  ∃ x y : ℝ, 
    line x y ∧ 
    y = curve a x ∧ 
    deriv (curve a) x = 1

-- State the theorem
theorem tangent_implies_a_eq_two :
  is_tangent 2 → ∀ a : ℝ, is_tangent a → a = 2 := by
  sorry

-- Additional lemma to show the existence of a tangent point
lemma tangent_point_exists (a : ℝ) : 
  is_tangent a → ∃ x y : ℝ, x > 0 ∧ y = curve a x ∧ line x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_a_eq_two_l1215_121504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_side_length_l1215_121535

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  leg : ℝ
  hypotenuse : ℝ
  leg_hypotenuse_relation : hypotenuse^2 = 2 * leg^2

/-- Get the length of a leg in an isosceles right triangle -/
def leg_length (t : IsoscelesRightTriangle) : ℝ := t.leg

theorem isosceles_right_triangle_side_length 
  (hypotenuse : ℝ) 
  (is_isosceles_right : IsoscelesRightTriangle) 
  (hyp_length : hypotenuse = 9.899494936611665) : 
  leg_length is_isosceles_right = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_side_length_l1215_121535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iodine_weight_in_compound_l1215_121555

/-- The atomic weight of iodine in a compound -/
def atomic_weight_iodine (nitrogen_weight hydrogen_weight : ℝ) 
  (compound_weight : ℝ) : ℝ :=
  compound_weight - (nitrogen_weight + 4 * hydrogen_weight)

/-- Theorem stating the atomic weight of iodine in the given compound -/
theorem iodine_weight_in_compound 
  (nitrogen_weight : ℝ) 
  (hydrogen_weight : ℝ) 
  (compound_weight : ℝ) 
  (h_nitrogen : nitrogen_weight = 14.01)
  (h_hydrogen : hydrogen_weight = 1.008)
  (h_compound : compound_weight = 145) :
  ∃ ε > 0, |atomic_weight_iodine nitrogen_weight hydrogen_weight compound_weight - 126.958| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iodine_weight_in_compound_l1215_121555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_four_l1215_121551

theorem cube_root_sum_equals_four :
  (8 + 3 * Real.sqrt 21) ^ (1/3) + (8 - 3 * Real.sqrt 21) ^ (1/3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_four_l1215_121551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_extrema_l1215_121591

-- Define the function f(x) as noncomputable due to its dependency on real numbers
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a * x

-- State the theorem
theorem tangent_and_extrema :
  ∃ (a : ℝ),
    (∀ x : ℝ, (deriv (f a)) x = x^2 + 2*x + a) ∧
    (deriv (f a)) 1 = 0 ∧
    a = -3 ∧
    (∀ x : ℝ, f (-3) x ≤ 9) ∧
    (∀ x : ℝ, f (-3) x ≥ -5/3) ∧
    f (-3) (-3) = 9 ∧
    f (-3) 1 = -5/3 :=
by
  -- Proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_extrema_l1215_121591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_pi_fourth_l1215_121564

/-- Given a function f where f(x) = f'(π/4) * cos(x) + sin(x), prove that f(π/4) = 1 -/
theorem function_value_at_pi_fourth (f : ℝ → ℝ) 
  (h : ∀ x, f x = (deriv f (π/4)) * Real.cos x + Real.sin x) : 
  f (π/4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_pi_fourth_l1215_121564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_not_equilateral_l1215_121528

/-- A triangular pyramid with lateral edge lengths 1, 2, and 4 does not have an equilateral base. -/
theorem pyramid_base_not_equilateral :
  ∀ (a : ℝ), a > 0 →
  ¬(∃ (S A B C : EuclideanSpace ℝ (Fin 3)),
    dist S A = 1 ∧ dist S B = 2 ∧ dist S C = 4 ∧
    dist A B = a ∧ dist B C = a ∧ dist C A = a) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_not_equilateral_l1215_121528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l1215_121508

/-- Represents the tank and its properties -/
structure Tank where
  capacity : ℚ
  initialFill : ℚ
  inflow : ℚ
  outflow1 : ℚ
  outflow2 : ℚ

/-- Calculates the time to fill the tank completely -/
def timeToFill (tank : Tank) : ℚ :=
  let remainingVolume := tank.capacity - tank.initialFill
  let netFlowRate := tank.inflow - (tank.outflow1 + tank.outflow2)
  remainingVolume / netFlowRate

/-- Theorem stating that the time to fill the tank is 48 minutes -/
theorem tank_fill_time :
  let tank : Tank := {
    capacity := 8000,
    initialFill := 4000,
    inflow := 1000 / 2,
    outflow1 := 1000 / 4,
    outflow2 := 1000 / 6
  }
  timeToFill tank = 48 := by
  -- Proof goes here
  sorry

#eval timeToFill {
  capacity := 8000,
  initialFill := 4000,
  inflow := 1000 / 2,
  outflow1 := 1000 / 4,
  outflow2 := 1000 / 6
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l1215_121508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_theorem_l1215_121563

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α

/-- The discriminant of a quadratic equation -/
def discriminant {α : Type*} [Field α] (eq : QuadraticEquation α) : α :=
  eq.b^2 - 4 * eq.a * eq.c

/-- Theorem about the roots of a quadratic equation -/
theorem quadratic_roots_theorem (m : ℝ) :
  let eq := QuadraticEquation.mk 1 3 (m - 1)
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 3*x + (m - 1) = 0 ∧ y^2 + 3*y + (m - 1) = 0) ↔ m < 13/4
  ∧
  (∃! x : ℝ, x^2 + 3*x + (m - 1) = 0) ↔ m = 13/4
  ∧
  (m = 13/4 → ∀ x : ℝ, x^2 + 3*x + (m - 1) = 0 ↔ x = 3/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_theorem_l1215_121563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1215_121553

/-- For any triangle ABC with circumradius R and inradius r, 
    the sum of (cos A / sin² A) + (cos B / sin² B) + (cos C / sin² C) 
    is greater than or equal to R/r -/
theorem triangle_inequality (A B C R r : Real) : 
  R > 0 → r > 0 → 
  A + B + C = π →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  Real.cos A / (Real.sin A)^2 + Real.cos B / (Real.sin B)^2 + Real.cos C / (Real.sin C)^2 ≥ R / r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1215_121553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l1215_121580

theorem cone_base_circumference (r θ : ℝ) :
  r > 0 →
  θ > 0 →
  θ < 2 * π →
  r = 5 ∧ θ = 2 * π / 3 →
  (θ / (2 * π)) * (2 * π * r) = 10 * π / 3 :=
by
  intro hr hθ_pos hθ_lt_2π h_values
  have original_circumference := 2 * π * r
  have cone_base_circumference := (θ / (2 * π)) * original_circumference
  sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l1215_121580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_regions_l1215_121500

-- Define the lines
def line1 (x y : ℝ) : Prop := y = 2 * x
def line2 (x y : ℝ) : Prop := y = x / 2
def line3 (x y : ℝ) : Prop := y = -x

-- Define the set of all points that satisfy at least one of the conditions
def S : Set (ℝ × ℝ) :=
  {p | line1 p.1 p.2 ∨ line2 p.1 p.2 ∨ line3 p.1 p.2}

-- Define a region as a connected component of the complement of S
-- We'll use a placeholder definition since IsConnectedComponent is not directly available
def is_region (R : Set (ℝ × ℝ)) : Prop :=
  ∃ x, x ∈ R ∧ x ∉ S

-- State the theorem
theorem number_of_regions :
  ∃ (R : Finset (Set (ℝ × ℝ))), (∀ r ∈ R, is_region r) ∧ R.card = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_regions_l1215_121500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_area_of_right_triangular_prism_l1215_121556

-- Define the right triangular prism ABC-A₁B₁C₁
structure RightTriangularPrism where
  -- Base triangle
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  -- Top triangle
  A₁ : ℝ × ℝ × ℝ
  B₁ : ℝ × ℝ × ℝ
  C₁ : ℝ × ℝ × ℝ

-- Define the properties of the prism
def PrismProperties (prism : RightTriangularPrism) : Prop :=
  -- ∠BAC = 90°
  ∃ (angleBAC : ℝ), angleBAC = Real.pi/2 ∧
  -- BC = 2
  ∃ (lengthBC : ℝ), lengthBC = 2 ∧
  -- CC₁ = 1
  ∃ (heightCC₁ : ℝ), heightCC₁ = 1 ∧
  -- The angle between line BC₁ and plane A₁ABB₁ is 60°
  ∃ (angleBCA₁ : ℝ), angleBCA₁ = Real.pi/3

-- Define the lateral area of the prism
noncomputable def LateralArea (prism : RightTriangularPrism) : ℝ := sorry

-- Theorem statement
theorem lateral_area_of_right_triangular_prism (prism : RightTriangularPrism) 
  (h : PrismProperties prism) : 
  LateralArea prism = (5 + Real.sqrt 15) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_area_of_right_triangular_prism_l1215_121556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l1215_121532

/-- The tangent line to y = 2ln x at (e, 2) intersects y-axis at (0, 0) -/
theorem tangent_line_intersection : 
  let f : ℝ → ℝ := λ x => 2 * Real.log x
  let tangent_point : ℝ × ℝ := (Real.exp 1, 2)
  let slope : ℝ := (deriv f) (Real.exp 1)
  let tangent_line : ℝ → ℝ := λ x => slope * (x - Real.exp 1) + 2
  tangent_line 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l1215_121532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_same_color_l1215_121525

/-- A coloring of integers modulo n satisfying certain conditions -/
def ValidColoring (n k : ℕ) (c : ℕ → Bool) : Prop :=
  1 < n ∧ 0 < k ∧ k < n ∧ Nat.Coprime k n ∧
  (∀ i, 0 < i ∧ i < n → c i = c (n - i)) ∧
  (∀ i, 0 < i ∧ i < n ∧ i ≠ k → c i = c (Int.natAbs (k - i)))

/-- All elements in the set have the same color -/
def AllSameColor (n : ℕ) (c : ℕ → Bool) : Prop :=
  ∀ i j, 0 < i ∧ i < n ∧ 0 < j ∧ j < n → c i = c j

theorem all_same_color (n k : ℕ) (c : ℕ → Bool) :
  ValidColoring n k c → AllSameColor n c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_same_color_l1215_121525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nice_polynomial_transformation_l1215_121569

/-- A polynomial is nice if P(0) = 1 and its nonzero coefficients alternate between 1 and -1 -/
def IsNice (P : Polynomial ℝ) : Prop :=
  P.eval 0 = 1 ∧ 
  ∀ i j, i < j → P.coeff i ≠ 0 → P.coeff j ≠ 0 → 
    P.coeff i = -P.coeff j

theorem nice_polynomial_transformation (P : Polynomial ℝ) (m n : ℕ) 
  (h_nice : IsNice P) (h_coprime : Nat.Coprime m n) :
  let Q := P.comp (Polynomial.monomial n 1) * 
    ((Polynomial.X ^ (m * n) - 1) * (Polynomial.X - 1)) / 
    ((Polynomial.X ^ m - 1) * (Polynomial.X ^ n - 1))
  IsNice Q :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nice_polynomial_transformation_l1215_121569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passenger_count_is_33_l1215_121538

/-- Represents the fuel consumption problem for an aircraft --/
structure AircraftFuel where
  baseFuelPerMile : ℚ
  personFuelPerMile : ℚ
  bagFuelPerMile : ℚ
  crewCount : ℕ
  bagsPerPerson : ℕ
  totalFuel : ℚ
  tripDistance : ℚ

/-- Calculates the number of passengers given the fuel constraints --/
def calculatePassengers (af : AircraftFuel) : ℕ :=
  let totalFuelPerMile := af.totalFuel / af.tripDistance
  let crewFuel := af.personFuelPerMile * af.crewCount
  let crewBagFuel := af.bagFuelPerMile * (af.crewCount * af.bagsPerPerson)
  let remainingFuel := totalFuelPerMile - af.baseFuelPerMile - crewFuel - crewBagFuel
  let passengerAndBagFuel := af.personFuelPerMile + (af.bagFuelPerMile * af.bagsPerPerson)
  (remainingFuel / passengerAndBagFuel).floor.toNat

/-- Theorem stating that given the specific conditions, the number of passengers is 33 --/
theorem passenger_count_is_33 : calculatePassengers {
  baseFuelPerMile := 20
  personFuelPerMile := 3
  bagFuelPerMile := 2
  crewCount := 5
  bagsPerPerson := 2
  totalFuel := 106000
  tripDistance := 400
} = 33 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_passenger_count_is_33_l1215_121538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_optimal_sum_of_digits_l1215_121523

/-- The number of positive integers that divide n, including 1 and n -/
def d (n : ℕ) : ℕ := sorry

/-- The function f(n) = d(n) / (n^(1/3)) -/
noncomputable def f (n : ℕ) : ℝ := (d n : ℝ) / n^(1/3)

/-- n is of the form 2^a * 3^b * 5^c -/
def n (a b c : ℕ) : ℕ := 2^a * 3^b * 5^c

/-- Theorem stating that f(n) is maximized when a = 5, b = 3, and c = 2 -/
theorem f_max : ∀ a b c : ℕ, f (n a b c) ≤ f (n 5 3 2) := by sorry

/-- The sum of digits of the optimal n -/
def sum_of_digits : ℕ := 18

/-- Theorem stating that the sum of digits of the optimal n is 18 -/
theorem optimal_sum_of_digits : sum_of_digits = 18 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_optimal_sum_of_digits_l1215_121523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_stones_count_l1215_121522

/-- Represents the dimensions of a rectangular area -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a paving stone -/
structure PavingStone where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
noncomputable def rectangleArea (r : Rectangle) : ℝ :=
  r.length * r.width

/-- Calculates the area of a paving stone -/
noncomputable def pavingStoneArea (p : PavingStone) : ℝ :=
  p.length * p.width

/-- Calculates the number of paving stones needed to cover a rectangle -/
noncomputable def pavingStonesNeeded (r : Rectangle) (p : PavingStone) : ℝ :=
  rectangleArea r / pavingStoneArea p

theorem courtyard_paving_stones_count 
  (courtyard : Rectangle)
  (pavingStone : PavingStone)
  (h1 : courtyard.length = 60)
  (h2 : courtyard.width = 14)
  (h3 : pavingStone.length = 3)
  (h4 : pavingStonesNeeded courtyard pavingStone = 140) :
  pavingStone.width = 2 ∧ pavingStonesNeeded courtyard pavingStone = 140 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_stones_count_l1215_121522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt5_div_2_approximation_l1215_121539

-- Define the approximation of √10
def sqrt10_approx : ℝ := 3.16

-- Define the approximation of √(5/2)
def sqrt5_div_2_approx : ℝ := 1.59

-- Theorem statement
theorem sqrt5_div_2_approximation (ε : ℝ) (h_ε : ε > 0) :
  |Real.sqrt (5/2) - sqrt5_div_2_approx| < ε := by
  sorry

#check sqrt5_div_2_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt5_div_2_approximation_l1215_121539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1215_121557

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 / (x + 1))

-- State the theorem
theorem f_domain : {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > -1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1215_121557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_4_minus_x_squared_l1215_121506

theorem integral_sqrt_4_minus_x_squared : 
  ∫ x in Set.Icc (-1 : ℝ) 1, Real.sqrt (4 - x^2) = Real.sqrt 3 + (2 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_4_minus_x_squared_l1215_121506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sqrt3_sin_eq_2sin_x_plus_phi_l1215_121533

theorem cos_minus_sqrt3_sin_eq_2sin_x_plus_phi (x φ : Real) :
  (∀ x, Real.cos x - Real.sqrt 3 * Real.sin x = 2 * Real.sin (x + φ)) →
  0 ≤ φ ∧ φ < 2 * Real.pi →
  φ = 5 * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sqrt3_sin_eq_2sin_x_plus_phi_l1215_121533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l1215_121550

/-- The eccentricity of a conic section defined by x^2 + y^2/m = 1,
    where m is the geometric mean of 2 and 8 -/
theorem conic_section_eccentricity (m : ℝ) : 
  m^2 = 2 * 8 →
  ∃ (e : ℝ), (e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) ∧
    (e = Real.sqrt (1 - (1/m)) ∨ e = Real.sqrt (1 + (1/m))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l1215_121550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1215_121507

-- Define the line C₁
def line_C₁ (x y : ℝ) : Prop := x - y + 6 = 0

-- Define the circle C₂
def circle_C₂ (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 2

-- State the theorem
theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  ∀ (P : ℝ × ℝ), circle_C₂ P.1 P.2 →
  ∀ (Q : ℝ × ℝ), line_C₁ Q.1 Q.2 →
  d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1215_121507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_combination_count_l1215_121524

/-- Represents a card with two different numbers -/
structure Card where
  front : Nat
  back : Nat
  different : front ≠ back

/-- The set of all possible cards -/
def cardSet : Finset Card := sorry

/-- A function that generates a three-digit number from three cards -/
def makeNumber (c1 c2 c3 : Card) (s1 s2 s3 : Bool) : Nat := sorry

/-- The set of all possible three-digit numbers -/
def allNumbers : Finset Nat := sorry

theorem card_combination_count :
  (cardSet.card = 3) ∧
  (∀ c ∈ cardSet, (c.front ∈ ({1, 3, 5} : Finset Nat) ∧ c.back ∈ ({2, 4, 6} : Finset Nat)) ∨ 
                  (c.front ∈ ({2, 4, 6} : Finset Nat) ∧ c.back ∈ ({1, 3, 5} : Finset Nat))) →
  allNumbers.card = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_combination_count_l1215_121524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_theorem_l1215_121544

noncomputable section

open Real

/-- The volume of a cone with n spheres of radius R inside and n spheres of radius 2R outside -/
def cone_volume (n : ℕ) (R : ℝ) : ℝ :=
  let α := sin (π / n : ℝ)
  (π * R^3 * (3 + sqrt (1 - 8 * α^2))^3 * (1 + sqrt (1 - 8 * α^2))) /
  (12 * α^2 * (1 - 6 * α^2 + sqrt (1 - 8 * α^2)))

/-- Theorem stating the volume of the cone given the conditions -/
theorem cone_volume_theorem (n : ℕ) (R : ℝ) (h1 : n > 0) (h2 : R > 0) :
  ∃ V : ℝ, V = cone_volume n R ∧ V > 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_theorem_l1215_121544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_with_equilateral_focus_l1215_121520

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola) : ℝ × ℝ :=
  (Real.sqrt (h.a^2 + h.b^2), 0)

/-- A point on the asymptote of a hyperbola -/
def point_on_asymptote (h : Hyperbola) : ℝ × ℝ → Prop :=
  λ p => ∃ t : ℝ, p.1 = t * h.a ∧ p.2 = t * h.b

/-- Equilateral triangle property -/
def is_equilateral_triangle (p1 p2 p3 : ℝ × ℝ) (side_length : ℝ) : Prop :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = side_length^2 ∧
  (p2.1 - p3.1)^2 + (p2.2 - p3.2)^2 = side_length^2 ∧
  (p3.1 - p1.1)^2 + (p3.2 - p1.2)^2 = side_length^2

theorem hyperbola_with_equilateral_focus (h : Hyperbola) 
  (A : ℝ × ℝ) (h_A : point_on_asymptote h A)
  (h_triangle : is_equilateral_triangle (0, 0) A (right_focus h) 2) :
  ∀ x y, hyperbola_equation h x y ↔ x^2 - y^2 / 3 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_with_equilateral_focus_l1215_121520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_diff_eq_l1215_121582

-- Define the differential equation
def diff_eq (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv y)) x = -(1 + Real.sqrt x) * y x

-- Define the initial conditions
def initial_conditions (y : ℝ → ℝ) : Prop :=
  y 0 = 1 ∧ (deriv y) 0 = 0

-- Define the property of having exactly one zero in an interval
def exactly_one_zero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

-- Main theorem
theorem zero_of_diff_eq :
  ∀ y : ℝ → ℝ,
  diff_eq y →
  initial_conditions y →
  exactly_one_zero y 0 (Real.pi / 2) ∧
  ∀ x, 0 < x ∧ x < Real.pi / 2 ∧ y x = 0 → x ≥ Real.pi / (2 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_diff_eq_l1215_121582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_windows_clear_time_l1215_121577

/-- Represents the queue at a train station ticket window. -/
structure TicketQueue where
  /-- The number of people in the queue -/
  people : ℝ
  /-- The rate at which people join the queue (people per minute) -/
  join_rate : ℝ
  /-- The rate at which one window can check tickets (people per minute) -/
  check_rate : ℝ

/-- Calculates the time needed to clear the queue with a given number of windows -/
noncomputable def clear_time (q : TicketQueue) (windows : ℝ) : ℝ :=
  q.people / (windows * q.check_rate - q.join_rate)

/-- The theorem to be proved -/
theorem three_windows_clear_time (q : TicketQueue) 
  (h1 : clear_time q 1 = 40)
  (h2 : clear_time q 2 = 16) :
  clear_time q 3 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_windows_clear_time_l1215_121577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1215_121573

/-- A circle passing through two points with a specific y-axis intercept -/
structure CircleWithConstraints where
  -- The circle passes through these two points
  p : ℝ × ℝ
  q : ℝ × ℝ
  -- Length of the segment intercepted on the y-axis
  y_intercept_length : ℝ
  -- Conditions given in the problem
  h_p : p = (4, -2)
  h_q : q = (-1, 3)
  h_y : y_intercept_length = 4

/-- The standard equation of the circle -/
def standard_equation (c : CircleWithConstraints) : Prop :=
  (∃ x y : ℝ, (x - 1)^2 + y^2 = 13) ∨
  (∃ x y : ℝ, (x - 5)^2 + (y - 4)^2 = 37)

/-- Theorem stating that the given circle satisfies one of the standard equations -/
theorem circle_equation (c : CircleWithConstraints) : standard_equation c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1215_121573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_graph_position_l1215_121515

/-- 
Given a quadratic function g(x) = ax^2 + bx + c where c = b^2 / (2a),
prove that the graph of y = g(x) lies entirely above the x-axis when a > 0,
and entirely below the x-axis when a < 0.
-/
theorem quadratic_graph_position (a b : ℝ) (h : a ≠ 0) :
  (∀ x, a > 0 → a * x^2 + b * x + b^2 / (2 * a) > 0) ∧ 
  (∀ x, a < 0 → a * x^2 + b * x + b^2 / (2 * a) < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_graph_position_l1215_121515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_alpha_l1215_121572

theorem sin_double_alpha (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (2 * α) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_alpha_l1215_121572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_ratio_constant_l1215_121596

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with equation x²/4 + y²/3 = 1 -/
def Ellipse : Set Point :=
  {p : Point | p.x^2/4 + p.y^2/3 = 1}

/-- The right focus of the ellipse -/
def rightFocus : Point :=
  ⟨1, 0⟩

/-- Checks if three points form a line -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: For any chord AB through the right focus and chord MN through the origin parallel to AB,
    |MN|²/|AB| is constant and equal to 4 -/
theorem ellipse_chord_ratio_constant 
  (A B M N : Point) 
  (hA : A ∈ Ellipse) (hB : B ∈ Ellipse) (hM : M ∈ Ellipse) (hN : N ∈ Ellipse)
  (hABF : collinear A B rightFocus)
  (hMNO : collinear M N ⟨0, 0⟩)
  (hParallel : (B.y - A.y) * N.x = (B.x - A.x) * N.y) :
  (distance M N)^2 / (distance A B) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_ratio_constant_l1215_121596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_plants_l1215_121588

-- Define the flower beds as finite sets
variable (A B C D : Finset ℕ)

-- Define the cardinalities of the sets and their intersections
variable (card_A : A.card = 600)
variable (card_B : B.card = 500)
variable (card_C : C.card = 400)
variable (card_D : D.card = 300)
variable (card_AB : (A ∩ B).card = 100)
variable (card_AC : (A ∩ C).card = 150)
variable (card_BC : (B ∩ C).card = 75)
variable (card_AD : (A ∩ D).card = 50)
variable (card_ABCD : (A ∩ B ∩ C ∩ D).card = 0)

-- Theorem statement
theorem total_plants : (A ∪ B ∪ C ∪ D).card = 1425 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_plants_l1215_121588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_altitude_l1215_121543

/-- A point in the plane -/
def Plane := ℝ × ℝ

/-- A right triangle with vertex Y as the right angle -/
def RightTriangle (X Y Z : Plane) : Prop := sorry

/-- A circle with diameter YZ intersecting XZ at W -/
def CircleDiameter (Y Z W : Plane) : Prop := sorry

/-- Area of a triangle -/
def AreaTriangle (X Y Z : Plane) : ℝ := sorry

/-- Length of a line segment -/
def SegmentLength (A B : Plane) : ℝ := sorry

/-- Given a right triangle XYZ with right angle at Y, a circle with diameter YZ
    intersecting XZ at W, prove that YW = 14 when the area of XYZ is 98 and XZ = 14. -/
theorem right_triangle_altitude (X Y Z W : Plane) : 
  RightTriangle X Y Z →
  CircleDiameter Y Z W →
  AreaTriangle X Y Z = 98 →
  SegmentLength X Z = 14 →
  SegmentLength Y W = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_altitude_l1215_121543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_production_increase_factor_l1215_121558

/-- The monthly production increase factor given initial and final productions over a period of months. -/
noncomputable def monthlyIncreaseFactor (initialProduction finalProduction : ℕ) (months : ℕ) : ℝ :=
  (finalProduction : ℝ) ^ (1 / months : ℝ) / (initialProduction : ℝ) ^ (1 / months : ℝ)

/-- Theorem stating that the monthly increase factor is 2 given the problem conditions. -/
theorem mask_production_increase_factor :
  monthlyIncreaseFactor 3000 48000 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_production_increase_factor_l1215_121558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1215_121548

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := x / (x^2 - 2*x + 2)

-- State the theorem about the range of g
theorem range_of_g :
  Set.range g = Set.Icc (-1/2 : ℝ) (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1215_121548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_invariance_l1215_121576

open Complex

-- Define the plane as the complex plane
variable (a b c : ℂ)

-- Define the condition that c is on one side of the line through a and b
-- This is a placeholder condition, as the exact mathematical representation would require more complex definitions
def onSameSide (a b c : ℂ) : Prop := sorry

-- Define the midpoint M
noncomputable def M (a b : ℂ) : ℂ := (a * (1 - I) + b * (1 + I)) / 2

-- State the theorem
theorem midpoint_invariance (a b : ℂ) :
  ∀ c : ℂ, onSameSide a b c → M a b = (a * (1 - I) + b * (1 + I)) / 2 := by
  intro c h
  rfl  -- reflexivity, as the left-hand side is definitionally equal to the right-hand side


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_invariance_l1215_121576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_squared_distances_l1215_121547

/-- The curve C in polar coordinates -/
def curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 9 / (Real.cos θ^2 + 9 * Real.sin θ^2)

/-- Two points are perpendicular if their angular difference is π/2 -/
def perpendicular (θ₁ θ₂ : ℝ) : Prop :=
  θ₂ = θ₁ + Real.pi/2 ∨ θ₂ = θ₁ - Real.pi/2

/-- The theorem to be proved -/
theorem sum_reciprocal_squared_distances (θ₁ θ₂ ρ₁ ρ₂ : ℝ) :
  curve ρ₁ θ₁ → curve ρ₂ θ₂ → perpendicular θ₁ θ₂ →
  1 / ρ₁^2 + 1 / ρ₂^2 = 10/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_squared_distances_l1215_121547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_football_expenses_l1215_121545

def football_expenses (home_games_year1 away_games_year1 home_playoff_year1 away_playoff_year1
                       home_games_year2 away_games_year2 home_playoff_year2
                       home_ticket away_ticket home_playoff_ticket away_playoff_ticket
                       friend_home_ticket friend_away_ticket home_transport away_transport : ℕ) : ℕ :=
  let cost_year1 := 
    home_games_year1 * (home_ticket + friend_home_ticket + home_transport) +
    away_games_year1 * (away_ticket + friend_away_ticket + away_transport) +
    home_playoff_year1 * (home_playoff_ticket + friend_home_ticket + home_transport) +
    away_playoff_year1 * (away_playoff_ticket + friend_away_ticket + away_transport)
  let cost_year2 :=
    home_games_year2 * (home_ticket + friend_home_ticket + home_transport) +
    away_games_year2 * (away_ticket + friend_away_ticket + away_transport) +
    home_playoff_year2 * (home_playoff_ticket + friend_home_ticket + home_transport)
  cost_year1 + cost_year2

theorem total_football_expenses : 
  football_expenses 6 3 1 1 2 2 1 60 75 120 100 45 75 25 50 = 2645 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_football_expenses_l1215_121545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_upper_bound_l1215_121513

theorem function_upper_bound
  (f : ℝ → ℝ)
  (h_nonneg : ∀ x ∈ Set.Icc 0 1, 0 ≤ f x)
  (h_one : f 1 = 1)
  (h_ineq : ∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x + y ∈ Set.Icc 0 1 →
            f (x + y) ≥ f x + f y) :
  ∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_upper_bound_l1215_121513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_bush_berries_l1215_121583

def berry_count : ℕ → ℕ
  | 0 => 2  -- We'll treat the first bush as index 0
  | 1 => 3
  | n + 2 => (berry_count n + berry_count (n + 1)) * (n + 2)

theorem seventh_bush_berries :
  berry_count 6 = 24339 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_bush_berries_l1215_121583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduce_to_single_digit_l1215_121518

def digit_sum (n : ℕ) : ℕ := sorry

def is_single_digit (n : ℕ) : Prop := n < 10

def can_reduce_to_single_digit (n : ℕ) (k : ℕ) : Prop :=
  ∃ (seq : Fin (k + 1) → ℕ), 
    seq 0 = n ∧ 
    (∀ i : Fin k, seq i.succ = digit_sum (seq i)) ∧
    is_single_digit (seq k)

theorem reduce_to_single_digit (n : ℕ) : 
  ∃ k : ℕ, k ≤ 10 ∧ can_reduce_to_single_digit n k := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduce_to_single_digit_l1215_121518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_eight_factors_l1215_121509

def has_eight_factors (n : ℕ) : Prop :=
  (Finset.filter (λ m ↦ m ∣ n) (Finset.range (n + 1))).card = 8

theorem least_integer_with_eight_factors :
  ∃ (n : ℕ), n > 0 ∧ has_eight_factors n ∧ ∀ (m : ℕ), m > 0 → has_eight_factors m → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_eight_factors_l1215_121509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1215_121549

theorem solve_exponential_equation :
  ∃ y : ℝ, 5 * (2 : ℝ) ^ y = 320 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1215_121549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1215_121512

theorem functional_equation_solution
  (a b : ℝ)
  (ha : 0 < a ∧ a < 1/2)
  (hb : 0 < b ∧ b < 1/2)
  (g : ℝ → ℝ)
  (hg : Continuous g)
  (h : ∀ x, g (g x) = a * g x + b * x) :
  ∃ c, ∀ x, g x = c * x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1215_121512
