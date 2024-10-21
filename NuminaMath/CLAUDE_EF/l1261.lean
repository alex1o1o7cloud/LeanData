import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_is_one_fourth_l1261_126125

-- Define the point P
noncomputable def P : ℝ × ℝ := (1/2, 1)

-- Define the inclination angle of line l
noncomputable def α : ℝ := Real.pi/6

-- Define the polar equation of circle C
noncomputable def C (θ : ℝ) : ℝ := Real.sqrt 2 * Real.cos (θ - Real.pi/4)

-- Define the line l
noncomputable def l (t : ℝ) : ℝ × ℝ := (1/2 + t * Real.cos α, 1 + t * Real.sin α)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- State the theorem
theorem distance_product_is_one_fourth :
  (A.1 - P.1)^2 + (A.2 - P.2)^2 * ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_is_one_fourth_l1261_126125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trough_max_area_l1261_126188

/-- The cross-sectional area of a trough made from three identical boards --/
noncomputable def troughArea (a : ℝ) (α : ℝ) : ℝ := a^2 * (1 + Real.sin α) * Real.cos α

/-- The maximum cross-sectional area of a trough made from three identical boards --/
noncomputable def maxTroughArea (a : ℝ) : ℝ := (3 * Real.sqrt 3 / 4) * a^2

theorem trough_max_area (a : ℝ) (h : a > 0) :
  ∃ α : ℝ, α ∈ Set.Icc 0 (Real.pi / 2) ∧
    (∀ β ∈ Set.Icc 0 (Real.pi / 2), troughArea a α ≥ troughArea a β) ∧
    troughArea a α = maxTroughArea a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trough_max_area_l1261_126188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_successful_expected_pairs_greater_than_half_l1261_126128

/-- Represents the number of pairs of socks -/
def n : ℕ := sorry

/-- Probability that all pairs are successful -/
noncomputable def all_successful_probability : ℚ := (2^n * n.factorial) / ((2*n).factorial)

/-- Expected number of successful pairs -/
noncomputable def expected_successful_pairs : ℚ := n / (2*n - 1)

/-- Theorem stating the probability of all pairs being successful -/
theorem probability_all_successful :
  all_successful_probability = (2^n * n.factorial) / ((2*n).factorial) := by sorry

/-- Theorem stating that the expected number of successful pairs is greater than 0.5 -/
theorem expected_pairs_greater_than_half :
  expected_successful_pairs > 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_successful_expected_pairs_greater_than_half_l1261_126128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_abs_increasing_and_even_l1261_126176

-- Define the function f(x) = |sin x|
noncomputable def f (x : ℝ) : ℝ := |Real.sin x|

-- State the theorem
theorem sin_abs_increasing_and_even :
  (∀ x₁ x₂, x₁ ∈ Set.Ioo 0 (Real.pi / 2) → x₂ ∈ Set.Ioo 0 (Real.pi / 2) → x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x, x ∈ Set.Ioo 0 (Real.pi / 2) → f x = f (-x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_abs_increasing_and_even_l1261_126176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_eq_42_l1261_126178

/-- The number of pairs of positive integers (m,n) satisfying m^2 + 2n < 30 -/
def count_pairs : ℕ :=
  Finset.card (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + 2*p.2 < 30) (Finset.product (Finset.range 30) (Finset.range 30)))

/-- Theorem stating that there are exactly 42 pairs of positive integers (m,n) satisfying m^2 + 2n < 30 -/
theorem count_pairs_eq_42 : count_pairs = 42 := by
  sorry

#eval count_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_eq_42_l1261_126178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_120_180_l1261_126133

theorem common_divisors_120_180 : 
  Finset.card (Finset.filter (λ d => d ∣ 120 ∧ d ∣ 180) (Finset.range 181)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_120_180_l1261_126133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_balloon_radius_l1261_126193

/-- The radius of the original balloon in feet -/
noncomputable def original_radius : ℝ := 2

/-- The number of smaller balloons -/
def num_small_balloons : ℕ := 64

/-- The volume of a sphere given its radius -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

/-- The radius of each smaller balloon -/
noncomputable def small_radius : ℝ := 1/2

/-- Theorem stating that the volume of the original balloon equals the total volume of smaller balloons -/
theorem smaller_balloon_radius :
  sphere_volume original_radius = num_small_balloons * sphere_volume small_radius := by
  sorry

#eval num_small_balloons -- This will compile and run

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_balloon_radius_l1261_126193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_min_area_l1261_126158

/-- Predicate to check if a set of four points forms a square -/
def is_square (s : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c d : ℝ × ℝ, s = {a, b, c, d} ∧
  (∃ side : ℝ, 
    dist a b = side ∧ 
    dist b c = side ∧ 
    dist c d = side ∧ 
    dist d a = side ∧
    dist a c = side * Real.sqrt 2 ∧
    dist b d = side * Real.sqrt 2)

/-- Function to calculate the area of a set of points forming a shape -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- A square with two vertices on y = 2x + 1 and two on y = -2x + 9 has minimum area 2 -/
theorem square_min_area (s : Set (ℝ × ℝ)) : 
  (∃ a b c d : ℝ × ℝ, s = {a, b, c, d} ∧ 
    is_square s ∧
    (∃ x y, a = (x, 2*x + 1) ∧ b = (y, 2*y + 1)) ∧
    (∃ u v, c = (u, -2*u + 9) ∧ d = (v, -2*v + 9))) →
  (∀ t : Set (ℝ × ℝ), 
    (∃ a' b' c' d' : ℝ × ℝ, t = {a', b', c', d'} ∧ 
      is_square t ∧
      (∃ x' y', a' = (x', 2*x' + 1) ∧ b' = (y', 2*y' + 1)) ∧
      (∃ u' v', c' = (u', -2*u' + 9) ∧ d' = (v', -2*v' + 9))) →
    area s ≤ area t) ∧
  area s = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_min_area_l1261_126158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_count_l1261_126142

def original_expression : ℕ := 3^(3^(3^3))

def parenthesis_arrangements : List (ℕ → ℕ → ℕ → ℕ) := [
  (fun a b c => a^(b^c)),
  (fun a b c => a^((b^c))),
  (fun a b c => (a^b)^c),
  (fun a b c => (a^(b^c))),
  (fun a b c => (a^b)^(c))
]

theorem distinct_values_count : 
  (parenthesis_arrangements.map (fun f => f 3 3 3)).eraseDups.length = 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_count_l1261_126142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1261_126132

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1 / x

-- Define a, b, and c
noncomputable def a : ℝ := f (1 / 3)
noncomputable def b : ℝ := f Real.pi
noncomputable def c : ℝ := f 5

-- State the theorem
theorem function_inequality : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1261_126132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1261_126154

/-- A function that returns true if a number is a three-digit even number with the sum of its hundreds and tens digits equal to 9 -/
def isValidNumber (n : ℕ) : Bool :=
  100 ≤ n ∧ n < 1000 ∧  -- Three-digit number
  n % 2 = 0 ∧  -- Even number
  (n / 100 + (n / 10) % 10 = 9)  -- Sum of hundreds and tens digits is 9

/-- The count of valid numbers as defined above -/
def countValidNumbers : ℕ :=
  (List.range 1000).filter isValidNumber |>.length

/-- Theorem stating that the count of valid numbers is 45 -/
theorem count_valid_numbers : countValidNumbers = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1261_126154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l1261_126198

noncomputable section

/-- The parabola is defined by the equation y = 1/2 * (x - 1)^2 + 2 -/
def parabola (x : ℝ) : ℝ := 1/2 * (x - 1)^2 + 2

/-- The vertex of a parabola is the point where it reaches its minimum or maximum -/
def is_vertex (x y : ℝ) : Prop :=
  ∀ t, parabola t ≥ parabola x ∧ parabola x = y

/-- Theorem: The vertex of the parabola y = 1/2 * (x - 1)^2 + 2 is at the point (1, 2) -/
theorem parabola_vertex : is_vertex 1 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l1261_126198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1261_126115

/-- The line l: x + y - 1 = 0 -/
def line (x y : ℝ) : Prop := x + y - 1 = 0

/-- The parabola y = x^2 -/
def parabola (x y : ℝ) : Prop := y = x^2

/-- Point M -/
def M : ℝ × ℝ := (-1, 2)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The product of distances from M to intersection points is 2 -/
theorem intersection_distance_product : 
  ∃ (A B : ℝ × ℝ), 
    line A.1 A.2 ∧ parabola A.1 A.2 ∧
    line B.1 B.2 ∧ parabola B.1 B.2 ∧
    A ≠ B ∧
    distance M A * distance M B = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1261_126115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_product_cardinality_sum_l1261_126123

theorem set_product_cardinality_sum (A : Finset ℝ) : 
  A.card = 7 → 
  let B := (A.product A).filter (λ p : ℝ × ℝ ↦ p.1 ≠ p.2)
  (Finset.card (B.image (λ p : ℝ × ℝ ↦ p.1 * p.2))).max + 
  (Finset.card (B.image (λ p : ℝ × ℝ ↦ p.1 * p.2))).min = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_product_cardinality_sum_l1261_126123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_is_15_l1261_126190

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram PQRS -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if two line segments are parallel -/
def isParallel (p1 p2 p3 p4 : Point) : Prop :=
  (p2.y - p1.y) * (p4.x - p3.x) = (p2.x - p1.x) * (p4.y - p3.y)

/-- Checks if an angle is 90 degrees -/
def isRightAngle (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p2.x) + (p2.y - p1.y) * (p3.y - p2.y) = 0

/-- Calculates the area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ :=
  |((p.Q.x - p.P.x) * (p.R.y - p.P.y) - (p.Q.y - p.P.y) * (p.R.x - p.P.x))|

theorem parallelogram_area_is_15 
  (p : Parallelogram) 
  (W Z : Point)
  (h1 : isParallel p.P p.Q p.R p.S)
  (h2 : distance p.P p.S = distance p.Q p.R)
  (h3 : W.x < Z.x ∧ Z.x < p.R.x)
  (h4 : isRightAngle p.P W p.Q)
  (h5 : isRightAngle p.Q Z p.R)
  (h6 : distance p.P W = 5)
  (h7 : distance W Z = 1)
  (h8 : distance Z p.R = 4)
  (h9 : distance p.P p.R = distance p.Q p.S) :
  parallelogramArea p = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_is_15_l1261_126190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_products_l1261_126114

theorem remainder_sum_products (a b c d : ℕ) 
  (ha : a % 7 = 2)
  (hb : b % 7 = 3)
  (hc : c % 7 = 5)
  (hd : d % 7 = 6) :
  (a * b + c * d) % 7 = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_products_l1261_126114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_interval_l1261_126149

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x

-- Define the theorem
theorem min_value_interval (a : ℝ) :
  (∃ x ∈ Set.Ioo a (10 - a^2), ∀ y ∈ Set.Ioo a (10 - a^2), f x ≤ f y) →
  a ∈ Set.Icc (-2) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_interval_l1261_126149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l1261_126103

noncomputable def vector1 : ℝ × ℝ := (3, 4)
noncomputable def vector2 : ℝ × ℝ := (2, -1)
noncomputable def p : ℝ × ℝ := (55/26, 45/26)

theorem projection_equality (v : ℝ × ℝ) : v ≠ (0, 0) →
  (∃ (k1 k2 : ℝ), 
    vector1 - k1 • v = p ∧ 
    vector2 - k2 • v = p ∧
    (vector1 - p) • v = 0 ∧
    (vector2 - p) • v = 0) := by
  sorry

#check projection_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l1261_126103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_max_work_hours_l1261_126195

/-- Represents Mary's work schedule and payment structure --/
structure WorkSchedule where
  regularRate : ℚ
  overtimeRate : ℚ
  regularHours : ℚ
  maxEarnings : ℚ

/-- Calculates the maximum number of hours Mary can work in a week --/
def maxWorkHours (schedule : WorkSchedule) : ℚ :=
  let regularEarnings := schedule.regularRate * schedule.regularHours
  let maxOvertimeEarnings := schedule.maxEarnings - regularEarnings
  let maxOvertimeHours := maxOvertimeEarnings / schedule.overtimeRate
  schedule.regularHours + maxOvertimeHours

/-- Theorem stating that Mary's maximum work hours is 70 --/
theorem mary_max_work_hours :
  let schedule : WorkSchedule := {
    regularRate := 8,
    overtimeRate := 10,
    regularHours := 20,
    maxEarnings := 660
  }
  maxWorkHours schedule = 70 := by
  sorry

#eval maxWorkHours {
  regularRate := 8,
  overtimeRate := 10,
  regularHours := 20,
  maxEarnings := 660
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_max_work_hours_l1261_126195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l1261_126169

noncomputable section

/-- The tangent line to the graph of f at the point (a, f(a)) -/
def TangentLine (f : ℝ → ℝ) (a x : ℝ) : ℝ :=
  f a + (deriv f a) * (x - a)

theorem tangent_line_property (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, TangentLine f 1 x = -x + 2) → f 1 + deriv f 1 = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l1261_126169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l1261_126143

theorem sin_cos_identity (x : ℝ) (h : Real.tan x = -1/2) :
  (Real.sin x) ^ 2 + 3 * (Real.sin x) * (Real.cos x) - 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l1261_126143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_sqrt3_implies_sides_2_sinB_2sinA_implies_area_l1261_126164

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.c = 2 ∧ t.C = Real.pi/3

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ :=
  1/2 * t.a * t.b * Real.sin t.C

-- Theorem 1
theorem area_sqrt3_implies_sides_2 (t : Triangle) 
  (h : isValidTriangle t) (area_eq : area t = Real.sqrt 3) :
  t.a = 2 ∧ t.b = 2 := by sorry

-- Theorem 2
theorem sinB_2sinA_implies_area (t : Triangle) 
  (h : isValidTriangle t) (sin_rel : Real.sin t.B = 2 * Real.sin t.A) :
  area t = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_sqrt3_implies_sides_2_sinB_2sinA_implies_area_l1261_126164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_bananas_l1261_126187

theorem shopkeeper_bananas (oranges : ℕ) (bananas : ℕ) 
  (h1 : oranges = 600)
  (h2 : (85 * oranges + 94 * bananas : ℚ) / (100 * (oranges + bananas)) = 443/500) :
  bananas = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_bananas_l1261_126187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_passing_both_subjects_l1261_126136

theorem students_passing_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 0.25) 
  (h2 : failed_english = 0.35) 
  (h3 : failed_both = 0.40) : 
  1 - (failed_hindi + failed_english - failed_both) = 0.80 := by
  sorry

#check students_passing_both_subjects

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_passing_both_subjects_l1261_126136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_arrangement_trapezoid_area_l1261_126171

/-- Represents a square with a given side length -/
structure Square (α : Type*) [LinearOrderedField α] where
  side : α

/-- Represents a trapezoid formed between two squares in the arrangement -/
structure Trapezoid (α : Type*) [LinearOrderedField α] where
  lower_base : α
  upper_base : α
  height : α

/-- Calculates the area of a trapezoid -/
def trapezoid_area {α : Type*} [LinearOrderedField α] (t : Trapezoid α) : α :=
  (t.lower_base + t.upper_base) * t.height / 2

/-- The main theorem statement -/
theorem square_arrangement_trapezoid_area :
  let squares : List (Square ℝ) := [⟨1⟩, ⟨3⟩, ⟨5⟩, ⟨7⟩]
  let total_width : ℝ := (squares.map Square.side).sum
  let height_ratio : ℝ := 7 / total_width
  let lower_base : ℝ := height_ratio * (1 + 3)
  let upper_base : ℝ := height_ratio * (1 + 3 + 5)
  let trapezoid : Trapezoid ℝ := ⟨lower_base, upper_base, 5 - 3⟩
  trapezoid_area trapezoid = 11.375 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_arrangement_trapezoid_area_l1261_126171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_z_in_second_quadrant_l1261_126137

-- Define the complex number z
def z (m : ℝ) : ℂ := (m - 3) + (m + 1) * Complex.I

-- Define the condition for z to be in the second quadrant
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem range_of_m_for_z_in_second_quadrant :
  ∀ m : ℝ, in_second_quadrant (z m) ↔ -1 < m ∧ m < 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_z_in_second_quadrant_l1261_126137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_GHCD_eq_153_75_l1261_126119

/-- Represents a trapezoid with parallel sides AB and CD -/
structure Trapezoid where
  ab : ℝ  -- Length of side AB
  cd : ℝ  -- Length of side CD
  h : ℝ   -- Altitude of the trapezoid
  ab_positive : 0 < ab
  cd_positive : 0 < cd
  h_positive : 0 < h

/-- The area of a quadrilateral GHCD formed by connecting the midpoints of the legs of a trapezoid -/
noncomputable def area_GHCD (t : Trapezoid) : ℝ :=
  (t.h / 2) * ((t.ab + t.cd) / 2 + t.cd) / 2

/-- Theorem: The area of quadrilateral GHCD in the given trapezoid is 153.75 square units -/
theorem area_GHCD_eq_153_75 (t : Trapezoid) 
    (h_ab : t.ab = 10) 
    (h_cd : t.cd = 24) 
    (h_h : t.h = 15) : 
  area_GHCD t = 153.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_GHCD_eq_153_75_l1261_126119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactlyThreePropositionsTrue_l1261_126102

-- Define the complex number type
structure MyComplex : Type :=
  (re : ℝ) (im : ℝ)

-- Define the ordering relation for complex numbers
def complexGreater (z1 z2 : MyComplex) : Prop :=
  z1.re > z2.re ∨ (z1.re = z2.re ∧ z1.im > z2.im)

-- Define some specific complex numbers
def one : MyComplex := ⟨1, 0⟩
def i : MyComplex := ⟨0, 1⟩
def zero : MyComplex := ⟨0, 0⟩

-- Define the four propositions
def prop1 : Prop := complexGreater one i ∧ complexGreater i zero

def prop2 : Prop := ∀ z1 z2 z3 : MyComplex, 
  complexGreater z1 z2 → complexGreater z2 z3 → complexGreater z1 z3

def prop3 : Prop := ∀ z1 z2 z : MyComplex, 
  complexGreater z1 z2 → complexGreater ⟨z1.re + z.re, z1.im + z.im⟩ ⟨z2.re + z.re, z2.im + z.im⟩

def prop4 : Prop := ∀ z z1 z2 : MyComplex, 
  complexGreater zero z → complexGreater z1 z2 → 
  complexGreater ⟨z.re * z1.re - z.im * z1.im, z.re * z1.im + z.im * z1.re⟩ 
                 ⟨z.re * z2.re - z.im * z2.im, z.re * z2.im + z.im * z2.re⟩

-- The theorem to be proved
theorem exactlyThreePropositionsTrue : 
  (prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4) ∨
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4) ∨
  (prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4) ∨
  (¬prop1 ∧ prop2 ∧ prop3 ∧ prop4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactlyThreePropositionsTrue_l1261_126102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponents_product_l1261_126168

theorem fraction_exponents_product : 
  (3/5 : ℚ)^5 * (4/7 : ℚ)^(-2 : ℤ) * (1/3 : ℚ)^4 = 11877/4050000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponents_product_l1261_126168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1261_126106

theorem inequality_holds (m : ℝ) (h : m > -1/2) :
  ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → (Real.cos θ)^2 + 2*m*(Real.sin θ) - 2*m - 2 < 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1261_126106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_bonus_points_l1261_126194

/-- Calculates the bonus points for Martha's grocery shopping --/
def calculate_bonus_points (beef_price : ℚ) (beef_quantity : ℕ)
                           (fruit_veg_price : ℚ) (fruit_veg_quantity : ℕ)
                           (spice_price : ℚ) (spice_quantity : ℕ)
                           (other_groceries : ℚ)
                           (points_per_ten_dollars : ℕ)
                           (total_points : ℕ) : ℕ :=
  let total_spent := beef_price * beef_quantity +
                     fruit_veg_price * fruit_veg_quantity +
                     spice_price * spice_quantity +
                     other_groceries
  let base_points := (((total_spent / 10) * points_per_ten_dollars).floor).toNat
  total_points - base_points

theorem martha_bonus_points :
  calculate_bonus_points 11 3 4 8 6 3 37 50 850 = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_bonus_points_l1261_126194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_five_consecutive_even_integers_l1261_126159

theorem largest_of_five_consecutive_even_integers : ℕ :=
  let sum_first_25_even : ℕ := 2 * (25 * 26) / 2
  let five_consecutive_sum (n : ℕ) : ℕ := 5 * n - 20
  have h : ∃ n : ℕ, five_consecutive_sum n = sum_first_25_even := by sorry
  let largest : ℕ := 134
  have h_largest : five_consecutive_sum largest = sum_first_25_even := by sorry
  have h_max : ∀ m : ℕ, five_consecutive_sum m = sum_first_25_even → m ≤ largest := by sorry
  largest


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_five_consecutive_even_integers_l1261_126159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_speed_l1261_126130

/-- Calculates the average speed for Monday to Wednesday given the conditions of the bus driver problem -/
theorem bus_driver_speed 
  (hours_per_day : ℝ)
  (days_per_week : ℝ)
  (speed_thu_fri : ℝ)
  (total_distance : ℝ) :
  hours_per_day = 2 →
  days_per_week = 5 →
  speed_thu_fri = 9 →
  total_distance = 108 →
  (hours_per_day * 3 * (total_distance - speed_thu_fri * hours_per_day * 2)) / (hours_per_day * 3) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_speed_l1261_126130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_length_l1261_126173

/-- The area of a quadrilateral given its diagonal and offsets -/
noncomputable def quadrilateralArea (d h₁ h₂ : ℝ) : ℝ := (1 / 2) * d * (h₁ + h₂)

/-- The length of a quadrilateral's diagonal given its area and offsets -/
theorem quadrilateral_diagonal_length (area h₁ h₂ : ℝ) 
  (h_area : area = 75) 
  (h_offset1 : h₁ = 6) 
  (h_offset2 : h₂ = 4) : 
  ∃ d : ℝ, d = 15 ∧ quadrilateralArea d h₁ h₂ = area :=
by
  -- Proof goes here
  sorry

#check quadrilateral_diagonal_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_length_l1261_126173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toothpicks_10th_stage_l1261_126166

def toothpicks : ℕ → ℕ
  | 0 => 4  -- Adding the base case for 0
  | n+1 => toothpicks n + 3

theorem toothpicks_10th_stage : toothpicks 10 = 31 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toothpicks_10th_stage_l1261_126166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_l1261_126182

/-- Triangle PQR with given properties and rotation of PQ -/
theorem triangle_rotation (P Q R : ℝ × ℝ) (h1 : P = (0, 0)) (h2 : R = (8, 0))
  (h3 : Q.1 > 0 ∧ Q.2 > 0) -- Q in first quadrant
  (h4 : (Q.2 - R.2) * (R.1 - P.1) = (R.2 - P.2) * (Q.1 - R.1)) -- ∠QRP = 90°
  (h5 : (Q.2 - P.2) = (Q.1 - P.1)) -- ∠QPR = 45°
  : (
      -Q.1 / 2 - Q.2 * Real.sqrt 3 / 2,
      Q.1 * Real.sqrt 3 / 2 - Q.2 / 2
    ) = (-4 - 4 * Real.sqrt 3, 4 * Real.sqrt 3 - 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_l1261_126182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_greater_than_sum_l1261_126148

def S : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 6) (Finset.range 7)

def favorable_outcomes : Finset (ℕ × ℕ) :=
  Finset.filter (λ (a, b) => a * b > a + b) (S.product S)

theorem probability_product_greater_than_sum :
  (favorable_outcomes.card : ℚ) / (S.product S).card = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_greater_than_sum_l1261_126148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l1261_126131

/-- Given two plane vectors a and b, if a is perpendicular to (a + λb), then λ = -5 -/
theorem perpendicular_vectors_lambda (a b : ℝ × ℝ) (l : ℝ) : 
  a = (2, 1) → 
  b = (-1, 3) → 
  a.1 * (a.1 + l * b.1) + a.2 * (a.2 + l * b.2) = 0 → 
  l = -5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l1261_126131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_example_l1261_126124

theorem complex_magnitude_example : Complex.abs (7/4 - 3*Complex.I) = Real.sqrt 193 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_example_l1261_126124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_iff_l1261_126117

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

-- State the theorem
theorem f_equals_one_iff (x : ℝ) : f x = 1 ↔ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_iff_l1261_126117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_problem_solution_l1261_126167

/-- Calculates the downstream distance given the conditions of the swimming problem. -/
noncomputable def downstream_distance (upstream_distance : ℝ) (time : ℝ) (still_water_speed : ℝ) : ℝ :=
  let stream_speed := (still_water_speed * time - upstream_distance) / time
  (still_water_speed + stream_speed) * time

/-- Theorem stating that the downstream distance is 28 km given the problem conditions. -/
theorem swimming_problem_solution :
  downstream_distance 12 2 10 = 28 := by
  -- Unfold the definition of downstream_distance
  unfold downstream_distance
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_problem_solution_l1261_126167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_B_in_special_triangle_l1261_126161

open Real

/-- Given a triangle ABC where tan A, (1+√2) tan B, tan C form an arithmetic sequence,
    the minimum value of angle B is π/4 -/
theorem min_angle_B_in_special_triangle (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  2 * (1 + Real.sqrt 2) * Real.tan B = Real.tan A + Real.tan C →
  π/4 ≤ B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_B_in_special_triangle_l1261_126161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_pot_gallons_l1261_126177

/-- Calculates the number of gallons of soup in the pot given the serving conditions -/
def soupGallons (ounces_per_bowl : ℕ) (bowls_per_minute : ℕ) (serving_time_minutes : ℕ) (ounces_per_gallon : ℕ) : ℕ :=
  let total_ounces := ounces_per_bowl * bowls_per_minute * serving_time_minutes
  let gallons_exact := (total_ounces : ℚ) / ounces_per_gallon
  (gallons_exact + 1/2).floor.toNat

/-- The number of gallons of soup in the pot is 6 given the specified conditions -/
theorem soup_pot_gallons :
  soupGallons 10 5 15 128 = 6 := by
  sorry

#eval soupGallons 10 5 15 128

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_pot_gallons_l1261_126177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_theorem_l1261_126105

/-- The time taken for two people walking in opposite directions on a circular track to meet. -/
noncomputable def meetingTime (trackCircumference : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  trackCircumference / (speed1 + speed2)

/-- Theorem: The time taken for two people walking in opposite directions on a circular track
    to meet for the first time is equal to the circumference of the track divided by the sum
    of their speeds. -/
theorem meeting_time_theorem (trackCircumference : ℝ) (speed1 : ℝ) (speed2 : ℝ)
    (h1 : trackCircumference > 0)
    (h2 : speed1 > 0)
    (h3 : speed2 > 0) :
  meetingTime trackCircumference speed1 speed2 = trackCircumference / (speed1 + speed2) := by
  sorry

/-- Application of the theorem to the specific problem -/
noncomputable def problemSolution : ℝ :=
  let trackCircumference : ℝ := 726
  let deepakSpeed : ℝ := 4.5 * 1000 / 60  -- Convert km/hr to m/min
  let wifeSpeed : ℝ := 3.75 * 1000 / 60   -- Convert km/hr to m/min
  meetingTime trackCircumference deepakSpeed wifeSpeed

-- We can't use #eval with noncomputable definitions, so we'll use #check instead
#check problemSolution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_theorem_l1261_126105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_loss_percentage_l1261_126185

theorem shopkeeper_loss_percentage 
  (selling_price_gain : ℝ) 
  (gain_percentage : ℝ)
  (selling_price_loss : ℝ) :
  selling_price_gain = 264 →
  gain_percentage = 20 →
  selling_price_loss = 187 →
  let cost_price := selling_price_gain / (1 + gain_percentage / 100)
  let loss := cost_price - selling_price_loss
  let loss_percentage := (loss / cost_price) * 100
  loss_percentage = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_loss_percentage_l1261_126185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_represents_ten_meters_l1261_126191

/-- Represents a cloth sale with a given gain percentage -/
structure ClothSale where
  totalMeters : ℚ
  gainPercentage : ℚ

/-- Calculates the number of meters represented by the gain in a cloth sale -/
def metersRepresentedByGain (sale : ClothSale) : ℚ :=
  sale.totalMeters * (sale.gainPercentage / (100 + sale.gainPercentage))

/-- Theorem: For a sale of 50 meters of cloth with 25% gain, the gain represents 10 meters -/
theorem gain_represents_ten_meters :
  let sale : ClothSale := { totalMeters := 50, gainPercentage := 25 }
  metersRepresentedByGain sale = 10 := by
  sorry

#eval metersRepresentedByGain { totalMeters := 50, gainPercentage := 25 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_represents_ten_meters_l1261_126191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_l1261_126179

def sequenceN (n : ℕ) : ℕ :=
  if n = 1 then 47
  else 47 * (10^(2*n-2) + 10^(n-1) + 1)

theorem only_first_prime :
  ∀ n : ℕ, n > 1 → ¬(Nat.Prime (sequenceN n)) ∧ Nat.Prime 47 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_l1261_126179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survivor_quitters_probability_l1261_126186

/-- The probability that all three quitters are from the same tribe in a Survivor-like scenario -/
theorem survivor_quitters_probability (n k : ℕ) : 
  n = 18 → 
  k = 3 → 
  (n / 2).choose k * 2 / n.choose k = 7 / 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survivor_quitters_probability_l1261_126186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l1261_126118

theorem cos_double_angle_special_case (α : ℝ) (h : Real.sin α = 3/5) : Real.cos (2 * α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l1261_126118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_closed_hexagonal_path_hitting_all_faces_l1261_126155

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cubic box -/
structure CubicBox where
  sideLength : ℝ

/-- Represents a hexagonal path within a cubic box -/
structure HexagonalPath where
  vertices : List Point3D
  box : CubicBox

/-- Classical law of reflection: The angle of incidence equals the angle of reflection -/
def classicalReflectionLaw (incident : Point3D) (reflected : Point3D) (normal : Point3D) : Prop :=
  sorry  -- Definition of classical reflection law

/-- A hexagonal path is valid if it follows the classical law of reflection at each vertex -/
def isValidHexagonalPath (path : HexagonalPath) : Prop :=
  sorry  -- Definition of a valid hexagonal path

/-- A hexagonal path is closed if the last vertex connects back to the first vertex -/
def isClosedPath (path : HexagonalPath) : Prop :=
  sorry  -- Definition of a closed path

/-- A hexagonal path hits each face of the cube if it has exactly one vertex on each face -/
def hitsAllFaces (path : HexagonalPath) : Prop :=
  sorry  -- Definition of hitting all faces

/-- Main theorem: There exists a closed hexagonal path in a cubic box that hits all faces -/
theorem exists_closed_hexagonal_path_hitting_all_faces (box : CubicBox) :
  ∃ (path : HexagonalPath), path.box = box ∧ 
    isValidHexagonalPath path ∧ 
    isClosedPath path ∧ 
    hitsAllFaces path := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_closed_hexagonal_path_hitting_all_faces_l1261_126155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_a_win_probability_l1261_126110

def basketball_competition (p : ℝ) : ℝ :=
  let n := 6  -- total number of shots
  let k := 4  -- minimum number of successful shots to win
  let last_two_success : ℝ := p * p  -- probability of last two shots being successful
  (Finset.sum (Finset.range (n - k + 1)) (λ i ↦ 
    (n.choose (k + i)) * (p ^ (k + i)) * ((1 - p) ^ (n - k - i)))) * last_two_success

theorem teacher_a_win_probability : 
  let p : ℝ := 2/3
  basketball_competition p = 32/81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_a_win_probability_l1261_126110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_catches_tortoise_in_24_minutes_l1261_126162

/-- The time it takes for a hare to catch up with a tortoise in a race -/
noncomputable def hare_tortoise_race_time (tortoise_dist : ℝ) (tortoise_time : ℝ) 
  (hare_dist : ℝ) (hare_time : ℝ) (initial_gap : ℝ) : ℝ :=
  let tortoise_speed := tortoise_dist / tortoise_time
  let hare_speed := hare_dist / hare_time
  let relative_speed := hare_speed - tortoise_speed
  initial_gap / relative_speed

/-- Proof that the hare catches up to the tortoise in 24 minutes -/
theorem hare_catches_tortoise_in_24_minutes : 
  hare_tortoise_race_time 5 3 5 2 20 = 24 := by
  -- Unfold the definition of hare_tortoise_race_time
  unfold hare_tortoise_race_time
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check hare_tortoise_race_time 5 3 5 2 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_catches_tortoise_in_24_minutes_l1261_126162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_marbles_seventy_two_divisible_least_marbles_is_seventy_two_l1261_126174

theorem least_marbles (n : ℕ) : 
  (∀ k ∈ ({3, 4, 6, 8, 9} : Set ℕ), k ∣ n) → n ≥ 72 :=
by sorry

theorem seventy_two_divisible : 
  ∀ k ∈ ({3, 4, 6, 8, 9} : Set ℕ), k ∣ 72 :=
by sorry

theorem least_marbles_is_seventy_two : 
  (∀ k ∈ ({3, 4, 6, 8, 9} : Set ℕ), k ∣ 72) ∧ 
  (∀ n : ℕ, (∀ k ∈ ({3, 4, 6, 8, 9} : Set ℕ), k ∣ n) → n ≥ 72) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_marbles_seventy_two_divisible_least_marbles_is_seventy_two_l1261_126174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1261_126139

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3) + Real.sqrt 3 - 2 * Real.sqrt 3 * (Real.cos x) ^ 2 + 1

theorem f_properties :
  let T := Real.pi
  let center (k : ℤ) := (k * Real.pi / 2 + Real.pi / 6, 1)
  let monotonic_interval (k : ℤ) := Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)
  (∀ x, f (x + T) = f x) ∧ 
  (∀ k : ℤ, ∃ ε > 0, ∀ x, |x - (center k).1| < ε → f x = f (2 * (center k).1 - x)) ∧
  (∀ k : ℤ, ∀ x ∈ monotonic_interval k, ∀ y ∈ monotonic_interval k, x < y → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1261_126139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_theorem_l1261_126122

/-- A broken line in a square -/
structure BrokenLine where
  -- The set of points that make up the broken line
  points : Set (ℝ × ℝ)
  -- The broken line is contained within the square
  contained_in_square : ∀ p ∈ points, 0 ≤ p.1 ∧ p.1 ≤ 100 ∧ 0 ≤ p.2 ∧ p.2 ≤ 100
  -- The line is connected
  connected : True  -- This is a simplification; actual connectedness would be more complex

/-- The Euclidean distance between two points -/
noncomputable def euclidean_distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The distance along a broken line between two points -/
noncomputable def distance_along_line (L : BrokenLine) (p q : ℝ × ℝ) : ℝ :=
  sorry  -- This would require a more complex definition

/-- The main theorem -/
theorem broken_line_theorem (L : BrokenLine) :
  (∀ p : ℝ × ℝ, 0 ≤ p.1 ∧ p.1 ≤ 100 ∧ 0 ≤ p.2 ∧ p.2 ≤ 100 →
    ∃ q, q ∈ L.points ∧ euclidean_distance p q ≤ 0.5) →
  ∃ p q, p ∈ L.points ∧ q ∈ L.points ∧
    euclidean_distance p q ≤ 1 ∧
    distance_along_line L p q ≥ 198 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_theorem_l1261_126122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l1261_126126

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race conditions -/
structure RaceConditions where
  a : Runner
  b : Runner
  c : Runner
  race_length : ℝ
  a_to_b_start : ℝ
  b_to_c_start : ℝ

/-- Theorem stating the result of the race conditions -/
theorem race_result (conditions : RaceConditions) 
  (h1 : conditions.race_length = 1000)
  (h2 : conditions.a_to_b_start = 60)
  (h3 : conditions.b_to_c_start = 148.936170212766) :
  ∃ (a_to_c_start : ℝ), a_to_c_start = conditions.b_to_c_start := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l1261_126126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1261_126197

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x

def g (x : ℝ) : ℝ := -x^2 - 2

theorem problem_statement :
  ∀ (a : ℝ),
  (∀ (x : ℝ), x > 0 → f a x ≥ g x) ↔ a ≤ 3 ∧
  (a = -1 →
    ∀ (m : ℝ), m > 0 →
      (∃ (y : ℝ), y ∈ Set.Icc m (m + 3) ∧
        (∀ (z : ℝ), z ∈ Set.Icc m (m + 3) → f (-1) z ≥ f (-1) y) ∧
        (f (-1) y = m * (Real.log m + 1) ∨ f (-1) y = -Real.exp (-2))) ∧
      (∃ (y : ℝ), y ∈ Set.Icc m (m + 3) ∧
        (∀ (z : ℝ), z ∈ Set.Icc m (m + 3) → f (-1) z ≤ f (-1) y) ∧
        f (-1) y = (m + 3) * (Real.log (m + 3) + 1))) ∧
  (∀ (x : ℝ), x > 0 → Real.log x + 1 > Real.exp (-x) - 2 / (Real.exp 1 * x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1261_126197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_equations_l1261_126135

/-- The circle with center (2, -1) and radius 1 -/
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 1

/-- The point through which the tangent lines pass -/
def tangent_point : ℝ × ℝ := (3, 3)

/-- A line is tangent to the circle if it touches the circle at exactly one point -/
def is_tangent (a b c : ℝ) : Prop :=
  ∃! (x y : ℝ), circle_eq x y ∧ a * x + b * y + c = 0

/-- The theorem stating the equations of the tangent lines -/
theorem tangent_lines_equations :
  (is_tangent 1 0 (-3) ∧ is_tangent 15 (-8) (-21)) ∧
  ∀ (a b c : ℝ), is_tangent a b c →
    (a = 1 ∧ b = 0 ∧ c = -3) ∨ (a = 15 ∧ b = -8 ∧ c = -21) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_equations_l1261_126135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_two_walks_l1261_126127

-- Helper definitions
def total_distance (v₁ v₂ t₁ t₂ : ℝ) : ℝ := v₁ * t₁ + v₂ * t₂
def total_time (t₁ t₂ : ℝ) : ℝ := t₁ + t₂

theorem average_speed_two_walks 
  (v₁ v₂ t₁ t₂ : ℝ) 
  (hv₁ : v₁ > 0) 
  (hv₂ : v₂ > 0) 
  (ht₁ : t₁ > 0) 
  (ht₂ : t₂ > 0) : 
  (v₁ * t₁ + v₂ * t₂) / (t₁ + t₂) = 
    (total_distance v₁ v₂ t₁ t₂) / (total_time t₁ t₂) :=
by
  -- Unfold the definitions of total_distance and total_time
  unfold total_distance total_time
  -- The equation now holds by reflexivity
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_two_walks_l1261_126127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1261_126120

/-- The area of a right triangle with base 30 cm and height 15 cm is 225 cm². -/
theorem right_triangle_area : 
  ∀ (A B C : ℝ × ℝ) (area : ℝ),
  ‖B - A‖ = 30 →
  ‖C - A‖ = 15 →
  (B - A) • (C - A) = 0 →
  area = (1 / 2) * 30 * 15 →
  area = 225 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1261_126120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_usage_for_small_canvas_l1261_126147

theorem paint_usage_for_small_canvas
  (paint_large : ℝ)
  (paint_small : ℝ)
  (num_large : ℕ)
  (num_small : ℕ)
  (total_paint : ℝ)
  (h1 : paint_large = 3)
  (h2 : num_large = 3)
  (h3 : num_small = 4)
  (h4 : total_paint = 17)
  (h5 : paint_large * (num_large : ℝ) + paint_small * (num_small : ℝ) = total_paint) :
  paint_small = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_usage_for_small_canvas_l1261_126147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1261_126141

-- Define the function f(x) = a^(-|x|)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(-abs x)

-- Theorem statement
theorem range_of_f (a : ℝ) (h : a > 1) :
  Set.range (f a) = Set.Ioo 0 1 ∪ {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1261_126141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_sequence_l1261_126151

open BigOperators

def our_sequence (n : ℕ) : ℚ := (3 * n) / (3 * n + 3)

theorem product_of_sequence :
  (∏ n in Finset.range 835, our_sequence (n + 1)) = 1 / 836 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_sequence_l1261_126151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l1261_126153

-- Define the circle and its properties
def circle_area : ℝ := 20
def angle_AOB : ℝ := 60
def angle_COD : ℝ := 30

-- Define the theorem
theorem shaded_area_theorem :
  let total_angle : ℝ := angle_AOB + angle_COD
  let shaded_fraction : ℝ := total_angle / 360
  let shaded_area : ℝ := shaded_fraction * circle_area
  shaded_area = 5 := by
  -- Unfold the let bindings
  simp [circle_area, angle_AOB, angle_COD]
  -- Perform the calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l1261_126153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_for_vector_equation_l1261_126121

theorem no_real_solutions_for_vector_equation :
  ¬∃ k : ℝ, ‖k • (![3, -4] : Fin 2 → ℝ) - (![5, 8] : Fin 2 → ℝ)‖ = 3 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_for_vector_equation_l1261_126121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_product_value_l1261_126108

theorem trigonometric_product_value : 
  let f (x : ℝ) := x^4 - 126*x^3 + 441*x^2 - 490*x + 121
  (f (Real.cos (π/9)^2) = 0) ∧ 
  (f (Real.cos (2*π/9)^2) = 0) ∧ 
  (f (Real.cos (4*π/9)^2) = 0) →
  Real.sqrt ((3 - Real.cos (π/9)^2) * (3 - Real.cos (2*π/9)^2) * (3 - Real.cos (4*π/9)^2)) = 11/9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_product_value_l1261_126108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1261_126112

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1)

-- Define the function g(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a^(2*x) - a^(x - 2) + 8

theorem function_properties (a : ℝ) :
  a > 0 ∧ a ≠ 1 ∧ f a 2 = 1/2 →
  a = 1/2 ∧
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 1 → 4 ≤ g a x ∧ g a x ≤ 8) ∧
  (∃ x₁ x₂, x₁ ∈ Set.Icc (-2 : ℝ) 1 ∧ x₂ ∈ Set.Icc (-2 : ℝ) 1 ∧ g a x₁ = 4 ∧ g a x₂ = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1261_126112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_balls_l1261_126138

/-- Represents the weight of a ball in an arbitrary unit -/
structure BallWeight where
  weight : ℝ

/-- The weight of a blue ball, serving as the base unit -/
def blue : BallWeight := ⟨1⟩

/-- The weight of a green ball in terms of blue balls -/
def green : BallWeight := ⟨2 * blue.weight⟩

/-- The weight of a yellow ball in terms of blue balls -/
def yellow : BallWeight := ⟨2.5 * blue.weight⟩

/-- The weight of a white ball in terms of blue balls -/
def white : BallWeight := ⟨1.5 * blue.weight⟩

/-- Theorem stating the number of blue balls needed to balance the given balls -/
theorem balance_balls : 
  (5 * green.weight + 3 * yellow.weight + 3 * white.weight) = 22 * blue.weight := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_balls_l1261_126138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_of_F_l1261_126157

/-- Represents a 2D point --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the letter F --/
structure LetterF where
  base : Point
  stem : Point

/-- Rotates a point 180° around the origin --/
def rotate180 (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- Reflects a point in the x-axis --/
def reflectX (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Translates a point --/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

/-- Applies all transformations to the letter F --/
def transformF (f : LetterF) : LetterF :=
  let rotated : LetterF := { base := rotate180 f.base, stem := rotate180 f.stem }
  let reflected : LetterF := { base := reflectX rotated.base, stem := reflectX rotated.stem }
  { base := translate reflected.base (-3) (-2),
    stem := translate reflected.stem (-3) (-2) }

theorem final_position_of_F :
  let initialF : LetterF := { base := { x := 1, y := 0 }, stem := { x := 0, y := 1 } }
  let finalF := transformF initialF
  finalF.base.x < 0 ∧ finalF.base.y = -2 ∧ finalF.stem.x = finalF.base.x ∧ finalF.stem.y > finalF.base.y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_of_F_l1261_126157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_reachability_l1261_126100

/-- Represents a point in the Cartesian plane -/
structure Point where
  x : ℕ
  y : ℕ

/-- Defines the possible jumps a frog can make -/
inductive Jump : Point → Point → Prop where
  | double_x : ∀ (p : Point), Jump p ⟨2 * p.x, 6⟩
  | double_y : ∀ (p : Point), Jump p ⟨p.x, 2 * p.y⟩
  | subtract_x : ∀ (p : Point), p.x > p.y → Jump p ⟨p.x - p.y, p.y⟩
  | subtract_y : ∀ (p : Point), p.y > p.x → Jump p ⟨p.x, p.y - p.x⟩

/-- Defines reachability from the starting point to a target point -/
def Reachable (target : Point) : Prop :=
  ∃ (path : List Point), path.head? = some ⟨1, 1⟩ ∧
    path.getLast? = some target ∧
    ∀ (i : ℕ) (p q : Point), i + 1 < path.length →
      path.get? i = some p → path.get? (i + 1) = some q →
      Jump p q

/-- The main theorem stating which points are reachable and which are not -/
theorem frog_jump_reachability :
  Reachable ⟨24, 40⟩ ∧
  Reachable ⟨200, 4⟩ ∧
  ¬Reachable ⟨40, 60⟩ ∧
  ¬Reachable ⟨24, 60⟩ := by
  sorry

#check frog_jump_reachability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_reachability_l1261_126100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_principle_for_numbers_l1261_126189

/-- The number of distinct two-digit numbers that can be formed using digits 1, 2, and 3 -/
def two_digit_count : ℕ := 9

/-- The number of distinct three-digit numbers that can be formed using digits 1, 2, and 3 -/
def three_digit_count : ℕ := 27

/-- The total number of distinct numbers that can be formed -/
def total_distinct_numbers : ℕ := two_digit_count + three_digit_count

/-- The minimum number of students required to guarantee at least three form the same number -/
def min_students : ℕ := 2 * total_distinct_numbers + 1

theorem pigeonhole_principle_for_numbers :
  ∀ (students : Finset ℕ) (f : ℕ → ℕ),
    students.card = min_students →
    (∀ n ∈ students, f n ≤ total_distinct_numbers) →
    ∃ (m : ℕ), m ≤ total_distinct_numbers ∧ (students.filter (λ s ↦ f s = m)).card ≥ 3 :=
by sorry

#check pigeonhole_principle_for_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_principle_for_numbers_l1261_126189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charitable_association_raffle_prove_charitable_association_raffle_l1261_126107

theorem charitable_association_raffle (male_female_ratio : ℚ) 
  (avg_female_tickets : ℕ) (avg_male_tickets : ℕ) (avg_total_tickets : ℕ) : Prop :=
  male_female_ratio = 1 / 2 ∧
  avg_female_tickets = 70 ∧
  avg_male_tickets = 58 ∧
  avg_total_tickets = 66

theorem prove_charitable_association_raffle :
  ∃ (male_female_ratio : ℚ) (avg_female_tickets avg_male_tickets avg_total_tickets : ℕ),
    charitable_association_raffle male_female_ratio avg_female_tickets avg_male_tickets avg_total_tickets :=
by
  use 1/2, 70, 58, 66
  simp [charitable_association_raffle]
  sorry

#check charitable_association_raffle
#check prove_charitable_association_raffle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charitable_association_raffle_prove_charitable_association_raffle_l1261_126107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_g_minus_one_l1261_126160

-- Define f as a function
variable (f : ℝ → ℝ)

-- Define g as a function
variable (g : ℝ → ℝ)

-- State the theorem
theorem prove_g_minus_one (hOdd : ∀ x, f x = -f (-x))
                          (hDef : ∀ x, f x = g x + x^2)
                          (hg1 : g 1 = 1) :
  g (-1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_g_minus_one_l1261_126160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrow_overlap_area_l1261_126175

/-- Represents an arrow on a grid -/
structure Arrow where
  direction : String
  deriving Repr

/-- Represents a grid -/
structure Grid where
  size : ℕ
  deriving Repr

/-- Calculates the area of overlap between two arrows on a grid -/
noncomputable def overlapArea (grid : Grid) (arrow1 arrow2 : Arrow) : ℚ :=
  sorry

theorem arrow_overlap_area :
  ∀ (grid : Grid) (arrow1 arrow2 : Arrow),
    grid.size = 4 →
    arrow1.direction = "North" →
    arrow2.direction = "West" →
    overlapArea grid arrow1 arrow2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrow_overlap_area_l1261_126175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l1261_126152

/-- The length of the chord formed by the intersection of a line and a circle -/
noncomputable def chord_length (a b c : ℝ) (x₀ y₀ r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - (abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2))^2)

/-- Theorem: The length of the chord formed by the intersection of the line 3x - 4y = 0
    and the circle (x-1)^2 + (y-2)^2 = 2 is equal to 2 -/
theorem chord_length_specific_case :
  chord_length 3 (-4) 0 1 2 (Real.sqrt 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l1261_126152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_quad_iteration_l1261_126104

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := if x < 1 then 0 else 2 * x - 2

-- State the theorem
theorem two_solutions_for_quad_iteration :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, f (f (f (f x))) = x := by
  sorry

#check two_solutions_for_quad_iteration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_quad_iteration_l1261_126104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1261_126146

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x) - Real.sin (2 * x)

theorem f_monotone_decreasing :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1261_126146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1261_126184

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (15 - 12 * Real.cos x) +
  Real.sqrt (4 - 2 * Real.sqrt 3 * Real.sin x) +
  Real.sqrt (7 - 4 * Real.sqrt 3 * Real.sin x) +
  Real.sqrt (10 - 4 * Real.sqrt 3 * Real.sin x - 6 * Real.cos x)

theorem f_minimum_value : ∀ x : ℝ, f x ≥ (9/2) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1261_126184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cherry_earnings_l1261_126181

/-- Represents Cherry's delivery service --/
structure DeliveryService where
  small_cargo_price : ℚ  -- Price for 3-5 kg cargo
  large_cargo_price : ℚ  -- Price for 6-8 kg cargo
  small_cargo_daily : ℕ  -- Number of 3-5 kg cargos delivered daily
  large_cargo_daily : ℕ  -- Number of 6-8 kg cargos delivered daily

/-- Calculates the weekly earnings for Cherry's delivery service --/
def weekly_earnings (service : DeliveryService) : ℚ :=
  7 * (service.small_cargo_price * service.small_cargo_daily +
       service.large_cargo_price * service.large_cargo_daily)

/-- Theorem stating that Cherry's weekly earnings are $126 --/
theorem cherry_earnings :
  let service := DeliveryService.mk (5/2) 4 4 2
  weekly_earnings service = 126 := by
  sorry

#eval weekly_earnings (DeliveryService.mk (5/2) 4 4 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cherry_earnings_l1261_126181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_calculations_l1261_126163

theorem square_root_calculations :
  ((Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) = 1) ∧
  (Real.sqrt 8 + Real.sqrt 12 / Real.sqrt 6 = 3 * Real.sqrt 2) :=
by
  constructor
  · sorry  -- Proof for the first part
  · sorry  -- Proof for the second part

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_calculations_l1261_126163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_investment_amount_l1261_126134

/-- The amount of the first investment -/
def x : ℝ := sorry

/-- The combined yearly return rate -/
def combined_return_rate : ℝ := 0.085

/-- The yearly return rate of the first investment -/
def first_investment_rate : ℝ := 0.07

/-- The yearly return rate of the second investment -/
def second_investment_rate : ℝ := 0.09

/-- The amount of the second investment -/
def second_investment_amount : ℝ := 1500

theorem first_investment_amount :
  (first_investment_rate * x + second_investment_rate * second_investment_amount =
   combined_return_rate * (x + second_investment_amount)) →
  x = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_investment_amount_l1261_126134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_translated_l1261_126150

open Real

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := sqrt 3 * cos (2 * x) - sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * (x + π / 3))

-- State the theorem
theorem f_equiv_g_translated : ∀ x, f x = g x := by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_translated_l1261_126150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_common_points_imply_a_range_l1261_126129

/-- The function f(x) = a^x where a > 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

/-- The function g(x) = x^2 -/
def g (x : ℝ) : ℝ := x^2

/-- The number of distinct common points between f and g -/
def commonPoints (a : ℝ) : ℕ := sorry

theorem three_common_points_imply_a_range (a : ℝ) :
  a > 1 ∧ commonPoints a = 3 → 1 < a ∧ a < Real.exp (2 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_common_points_imply_a_range_l1261_126129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1261_126116

noncomputable section

/-- The time (in days) it takes for person A to complete the entire work -/
def a_complete_time : ℝ := 8

/-- The time (in days) person A works on the task -/
def a_work_time : ℝ := 4

/-- The time (in days) it takes for person B to complete the remaining work -/
def b_complete_time : ℝ := 6

/-- The fraction of work completed by person A -/
noncomputable def a_work_fraction : ℝ := a_work_time / a_complete_time

/-- The fraction of work completed by person B -/
noncomputable def b_work_fraction : ℝ := 1 - a_work_fraction

/-- The rate at which person A completes work (fraction per day) -/
noncomputable def a_rate : ℝ := 1 / a_complete_time

/-- The rate at which person B completes work (fraction per day) -/
noncomputable def b_rate : ℝ := b_work_fraction / b_complete_time

/-- The combined rate at which A and B complete work together (fraction per day) -/
noncomputable def combined_rate : ℝ := a_rate + b_rate

theorem work_completion_time :
  (1 / combined_rate) = 4.8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1261_126116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inner_triangle_l1261_126140

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Type where
  valid : Prop

-- Define the area of a triangle
noncomputable def area (P Q R : ℝ × ℝ) : ℝ := sorry

-- Define a point on a line segment
def on_segment (P Q M : ℝ × ℝ) : Prop := sorry

-- Define the midpoint of a line segment
def is_midpoint (P Q M : ℝ × ℝ) : Prop := sorry

-- Define a point dividing a line segment in a given ratio
def divides_in_ratio (P R N : ℝ × ℝ) (a b : ℝ) : Prop := sorry

theorem area_of_inner_triangle 
  (P Q R M N : ℝ × ℝ) 
  (h_triangle : Triangle P Q R) 
  (h_area : area P Q R = 36) 
  (h_M_on_PQ : on_segment P Q M) 
  (h_N_on_PR : on_segment P R N) 
  (h_M_midpoint : is_midpoint P Q M) 
  (h_N_divides : divides_in_ratio P R N 1 2) : 
  area P M N = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inner_triangle_l1261_126140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_reduction_l1261_126165

theorem trigonometric_equation_reduction :
  ∃ (a b c : ℕ+),
    (∀ x : ℝ, Real.sin x ^ 2 + Real.sin (3 * x) ^ 2 + Real.sin (5 * x) ^ 2 + Real.sin (6 * x) ^ 2 = 81 / 32 → 
      Real.cos (a * x) * Real.cos (b * x) * Real.cos (c * x) = 0) ∧
    a + b + c = 14 ∧
    (∀ a' b' c' : ℕ+, 
      (∀ x : ℝ, Real.sin x ^ 2 + Real.sin (3 * x) ^ 2 + Real.sin (5 * x) ^ 2 + Real.sin (6 * x) ^ 2 = 81 / 32 → 
        Real.cos (a' * x) * Real.cos (b' * x) * Real.cos (c' * x) = 0) →
      a' + b' + c' ≥ 14) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_reduction_l1261_126165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_intersection_is_union_of_complements_l1261_126183

theorem negation_of_intersection_is_union_of_complements 
  {U : Type} [Nonempty U] (A B : Set U) (x : U) :
  (¬ (x ∈ A ∩ B)) ↔ (x ∈ (Aᶜ ∪ Bᶜ)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_intersection_is_union_of_complements_l1261_126183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_of_equation_l1261_126101

-- Define the equation
noncomputable def f (x : ℝ) : ℝ := (4 : ℝ)^x + (10 : ℝ)^x - (25 : ℝ)^x

-- Define the unique solution
noncomputable def x_solution : ℝ := Real.log ((Real.sqrt 5 - 1) / 2) / Real.log (2/5)

-- Theorem statement
theorem unique_solution_of_equation :
  -- The equation has a unique solution
  ∃! x, f x = 0 ∧
  -- The solution is x_solution
  x = x_solution ∧
  -- The solution is between 0 and 1
  0 < x ∧ x < 1 :=
sorry

#check unique_solution_of_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_of_equation_l1261_126101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_are_identity_l1261_126144

def is_valid_sequence (a b : ℕ → ℕ) : Prop :=
  a 1 = b 1 ∧
  ∀ n : ℕ, n > 0 → (Finset.filter (fun k ↦ k > 0 ∧ k ≤ n ∧ a k ≤ n) (Finset.range (n + 1))).card = b n ∧
  ∀ n : ℕ, n > 0 → (Finset.filter (fun k ↦ k > 0 ∧ k ≤ n ∧ b k ≤ n) (Finset.range (n + 1))).card = a n

theorem sequences_are_identity {a b : ℕ → ℕ} (h : is_valid_sequence a b) :
  ∀ n : ℕ, n > 0 → a n = n ∧ b n = n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_are_identity_l1261_126144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_y_l1261_126196

-- Define the operation
noncomputable def circle_slash (a b : ℝ) : ℝ := (Real.sqrt (3 * a + 2 * b)) ^ 3

-- State the theorem
theorem solve_for_y (y : ℝ) : circle_slash 3 y = 125 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_y_l1261_126196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l1261_126199

theorem vector_decomposition (a e₁ e₂ : ℝ × ℝ) :
  a = (-3, 4) →
  e₁ = (-1, 2) →
  e₂ = (3, -1) →
  ∃! (x y : ℝ), a = x • e₁ + y • e₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l1261_126199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_right_triangle_hypotenuse_l1261_126156

/-- A right triangle with specific cone volume properties -/
structure SpecialRightTriangle where
  a : ℝ  -- Length of one leg
  b : ℝ  -- Length of the other leg
  h_positive : 0 < a ∧ 0 < b  -- Lengths are positive
  h_volume_ratio : (1/3 * Real.pi * b * a^2) = 2 * (1/3 * Real.pi * a * b^2)  -- Volume ratio condition
  h_volume_sum : (1/3 * Real.pi * b * a^2) + (1/3 * Real.pi * a * b^2) = 4480 * Real.pi  -- Volume sum condition

/-- The hypotenuse of a SpecialRightTriangle is approximately 22.89 cm -/
theorem special_right_triangle_hypotenuse (t : SpecialRightTriangle) :
  ∃ ε > 0, |Real.sqrt (t.a^2 + t.b^2) - 22.89| < ε := by
  sorry

#check special_right_triangle_hypotenuse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_right_triangle_hypotenuse_l1261_126156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_l1261_126111

-- Define complex numbers A, B, C, and real number D
def A : ℂ := 5 - 2 * Complex.I
def B : ℂ := -3 + 4 * Complex.I
def C : ℂ := 2 * Complex.I
def D : ℝ := 3

-- Theorem statement
theorem complex_arithmetic :
  A - B + C - (D : ℂ) = 5 - 4 * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_l1261_126111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xz_length_is_correct_l1261_126113

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  parallel_sides : (Y.2 - W.2) / (Y.1 - W.1) = (Z.2 - X.2) / (Z.1 - X.1)
  wy_length : Real.sqrt ((Y.1 - W.1)^2 + (Y.2 - W.2)^2) = 7
  xy_length : Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) = 5 * Real.sqrt 2
  angle_xyz : Real.cos (Real.arctan ((Z.2 - Y.2) / (Z.1 - Y.1)) - Real.arctan ((X.2 - Y.2) / (X.1 - Y.1))) = Real.cos (30 * π / 180)
  angle_wzx : Real.cos (Real.arctan ((X.2 - Z.2) / (X.1 - Z.1)) - Real.arctan ((W.2 - Z.2) / (W.1 - Z.1))) = Real.cos (60 * π / 180)

/-- The length of XZ in the special trapezoid -/
noncomputable def xz_length (t : SpecialTrapezoid) : ℝ :=
  Real.sqrt ((t.Z.1 - t.X.1)^2 + (t.Z.2 - t.X.2)^2)

/-- Theorem: The length of XZ in the special trapezoid is 7 + (7/4)√3 -/
theorem xz_length_is_correct (t : SpecialTrapezoid) : xz_length t = 7 + (7/4) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xz_length_is_correct_l1261_126113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_class_with_at_least_35_students_l1261_126180

theorem exists_class_with_at_least_35_students 
  (num_classes : ℕ) 
  (total_students : ℕ) 
  (h1 : num_classes = 33) 
  (h2 : total_students = 1150) : 
  ∃ (class_size : ℕ), class_size ≥ 35 ∧ class_size ∈ (Finset.range num_classes).image 
    (λ i ↦ (total_students / num_classes + if i < total_students % num_classes then 1 else 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_class_with_at_least_35_students_l1261_126180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1261_126170

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2)
  (h3 : a 1 < 0)
  (h4 : S 1999 = S 2023) :
  d > 0 ∧ S 4022 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1261_126170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_craig_first_day_pizzas_l1261_126145

/-- The number of pizzas Craig made on the first day -/
def C : ℕ := 40

/-- The total number of pizzas made over two days -/
def total_pizzas : ℕ := 380

/-- Heather made 4 times as many pizzas as Craig on the first day -/
def heather_day1 : ℕ := 4 * C

/-- Craig made 60 more pizzas on the second day than the first day -/
def craig_day2 : ℕ := C + 60

/-- Heather made 20 fewer pizzas than Craig on the second day -/
def heather_day2 : ℕ := craig_day2 - 20

theorem craig_first_day_pizzas : 
  C + heather_day1 + craig_day2 + heather_day2 = total_pizzas := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_craig_first_day_pizzas_l1261_126145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_side_lengths_l1261_126192

/-- Side length of a regular n-sided polygon inscribed in a circle of radius r -/
noncomputable def side_length (n : ℕ) (r : ℝ) : ℝ :=
  match n with
  | 3 => r * Real.sqrt 3
  | 4 => r * Real.sqrt 2
  | 5 => r / 2 * Real.sqrt (10 - 2 * Real.sqrt 5)
  | 6 => r
  | 10 => r * (Real.sqrt 5 - 1) / 2
  | _ => 0  -- Default case, not used in our theorem

/-- Theorem stating the equality of side lengths for regular polygons -/
theorem regular_polygon_side_lengths (r : ℝ) (h : r > 0) :
  (side_length 6 r)^2 + (side_length 4 r)^2 = (side_length 3 r)^2 ∧
  (side_length 6 r)^2 + (side_length 10 r)^2 = (side_length 5 r)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_side_lengths_l1261_126192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_modulus_for_real_roots_l1261_126172

theorem min_modulus_for_real_roots (m : ℂ) :
  (∃ x : ℝ, x^2 + m * x + 1 + (2 : ℂ) * Complex.I = 0) →
  Complex.abs m ≥ Real.sqrt (2 + 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_modulus_for_real_roots_l1261_126172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l1261_126109

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- Line l₁: 3x + 4y - 3 = 0 -/
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 3 = 0

/-- Line l₂: 3x + 4y + 7 = 0 -/
def l₂ (x y : ℝ) : Prop := 3 * x + 4 * y + 7 = 0

theorem distance_between_l₁_and_l₂ :
  distance_between_parallel_lines 3 4 (-3) 7 = 2 := by
  -- Unfold the definition of distance_between_parallel_lines
  unfold distance_between_parallel_lines
  -- Simplify the expression
  simp [abs_of_nonneg]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l1261_126109
