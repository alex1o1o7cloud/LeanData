import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_fixed_circle_l985_98548

structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

noncomputable def Ellipse.leftFocus (e : Ellipse) : ℝ × ℝ :=
  (-e.a * e.eccentricity, 0)

noncomputable def Ellipse.verticalChordLength (e : Ellipse) (x : ℝ) : ℝ :=
  2 * e.b * Real.sqrt (1 - x^2 / e.a^2)

theorem ellipse_equation_and_fixed_circle 
  (e : Ellipse) 
  (h_ecc : e.eccentricity = Real.sqrt 3 / 2) 
  (h_chord : e.verticalChordLength (e.leftFocus.1) = 1) :
  (∃ (a b : ℝ), a = 2 ∧ b = 1 ∧ ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 = 1)) ∧
  (∃ (r : ℝ), r^2 = 4/5 ∧
    ∀ (p q : ℝ × ℝ), 
      (p.1^2 / 4 + p.2^2 = 1) → 
      (q.1^2 / 4 + q.2^2 = 1) → 
      (p.1 * q.1 + p.2 * q.2 = 0) → 
      ∃ (t : ℝ), (t * p.1 + (1-t) * q.1)^2 + (t * p.2 + (1-t) * q.2)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_fixed_circle_l985_98548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_numbers_l985_98565

def numbers : Finset ℕ := {1234, 1567, 1890, 2023, 2147, 2255, 2401}

theorem mean_of_remaining_numbers (S : Finset ℕ) 
  (h1 : S ⊆ numbers)
  (h2 : S.card = 5)
  (h3 : (S.sum (fun x => (x : ℚ))) / 5 = 2020) :
  let T := numbers \ S
  (T.sum (fun x => (x : ℚ))) / 2 = 1708.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_numbers_l985_98565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l985_98599

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2 / 16 + y^2 / 7 = 1

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define the left focus F of the ellipse E
def F : ℝ × ℝ := (-3, 0)

-- Define a point P on the ellipse E
def P : Type := {p : ℝ × ℝ // E p.1 p.2}

-- Define a point M on the circle C
def M : Type := {m : ℝ × ℝ // C m.1 m.2}

-- Statement of the theorem
theorem min_distance_sum :
  ∃ (k : ℝ), (∀ (p : P) (m : M),
    Real.sqrt ((p.val.1 - F.1)^2 + (p.val.2 - F.2)^2) +
    Real.sqrt ((p.val.1 - m.val.1)^2 + (p.val.2 - m.val.2)^2) ≥ k) ∧
  k = 7 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l985_98599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barycentric_coordinates_existence_and_uniqueness_l985_98518

-- Define a triangle in 2D space
structure Triangle where
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ
  A₃ : ℝ × ℝ

-- Define barycentric coordinates as a structure
structure BarycentricCoords where
  m₁ : ℝ
  m₂ : ℝ
  m₃ : ℝ

-- Theorem statement
theorem barycentric_coordinates_existence_and_uniqueness 
  (t : Triangle) (X : ℝ × ℝ) :
  (∃ m : BarycentricCoords, 
    X = (m.m₁ * t.A₁.1 + m.m₂ * t.A₂.1 + m.m₃ * t.A₃.1,
         m.m₁ * t.A₁.2 + m.m₂ * t.A₂.2 + m.m₃ * t.A₃.2)) ∧
  (∀ m₁ m₂ : BarycentricCoords, 
    m₁.m₁ + m₁.m₂ + m₁.m₃ = 1 → 
    m₂.m₁ + m₂.m₂ + m₂.m₃ = 1 →
    (m₁.m₁ * t.A₁.1 + m₁.m₂ * t.A₂.1 + m₁.m₃ * t.A₃.1,
     m₁.m₁ * t.A₁.2 + m₁.m₂ * t.A₂.2 + m₁.m₃ * t.A₃.2) =
    (m₂.m₁ * t.A₁.1 + m₂.m₂ * t.A₂.1 + m₂.m₃ * t.A₃.1,
     m₂.m₁ * t.A₁.2 + m₂.m₂ * t.A₂.2 + m₂.m₃ * t.A₃.2) →
    m₁ = m₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_barycentric_coordinates_existence_and_uniqueness_l985_98518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l985_98590

def a : ℕ → ℚ
  | 0 => 1/2  -- Add this case for 0
  | 1 => 1/2
  | n + 1 => 2 * a n / (3 * a n + 2)

theorem a_2017_value : a 2017 = 1 / 3026 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l985_98590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_intersecting_line_l985_98541

/-- A line passing through (-√3, 0) intersecting the unit circle -/
structure IntersectingLine where
  /-- The slope of the line -/
  slope : ℝ

/-- The angle between the two intersection points from the origin -/
noncomputable def intersectionAngle (l : IntersectingLine) : ℝ := sorry

/-- The area of the triangle formed by the two intersection points and the origin -/
noncomputable def triangleArea (l : IntersectingLine) : ℝ := sorry

theorem slope_of_intersecting_line (l : IntersectingLine) 
  (angle_range : intersectionAngle l ∈ Set.Ioo 0 (π / 2))
  (area_condition : triangleArea l = Real.sqrt 3 / 4) :
  l.slope = Real.sqrt 3 / 3 ∨ l.slope = -(Real.sqrt 3 / 3) := by
  sorry

#check slope_of_intersecting_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_intersecting_line_l985_98541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convexity_and_means_l985_98500

open Real

-- Define the exponential function with base 10
noncomputable def exp10 (x : ℝ) : ℝ := (10 : ℝ) ^ x

-- Define the logarithm with base 10
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the weighted geometric mean
noncomputable def weightedGeometricMean (x₁ x₂ q₁ q₂ : ℝ) : ℝ := x₁ ^ q₁ * x₂ ^ q₂

-- Define the weighted arithmetic mean
def weightedArithmeticMean (x₁ x₂ q₁ q₂ : ℝ) : ℝ := q₁ * x₁ + q₂ * x₂

-- Define the weighted harmonic mean
noncomputable def weightedHarmonicMean (x₁ x₂ q₁ q₂ : ℝ) : ℝ := 1 / (q₁ / x₁ + q₂ / x₂)

theorem convexity_and_means (x₁ x₂ q₁ q₂ : ℝ) (hq : q₁ + q₂ = 1) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) :
  ConvexOn ℝ Set.univ exp10 ∧
  ConcaveOn ℝ Set.univ log10 ∧
  weightedGeometricMean x₁ x₂ q₁ q₂ ≤ weightedArithmeticMean x₁ x₂ q₁ q₂ ∧
  weightedGeometricMean x₁ x₂ q₁ q₂ ≥ weightedHarmonicMean x₁ x₂ q₁ q₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convexity_and_means_l985_98500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l985_98539

/-- An ellipse with the given properties has eccentricity √6/3 -/
theorem ellipse_eccentricity (a b c : ℝ) : 
  a > 0 → b > 0 → a > b →
  (∀ x y : ℝ, x^2/b^2 + y^2/a^2 = 1 ↔ (x, y) ∈ Set.range (λ t ↦ (a * Real.cos t, b * Real.sin t))) →
  (0, c) ∈ Set.range (λ t ↦ (a * Real.cos t, b * Real.sin t)) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁, 3*x₁ - 2) ∈ Set.range (λ t ↦ (a * Real.cos t, b * Real.sin t)) ∧
    (x₂, 3*x₂ - 2) ∈ Set.range (λ t ↦ (a * Real.cos t, b * Real.sin t)) ∧
    (x₁ + x₂)/2 = 1/2) →
  c/a = Real.sqrt 6/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l985_98539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_surface_area_ratio_l985_98523

/-- Represents a cylinder with base radius and height -/
structure Cylinder where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere with radius -/
structure Sphere where
  radius : ℝ

/-- Calculates the surface area of a cylinder -/
noncomputable def cylinderSurfaceArea (c : Cylinder) : ℝ :=
  2 * Real.pi * c.baseRadius * c.height + 2 * Real.pi * c.baseRadius^2

/-- Calculates the surface area of a sphere -/
noncomputable def sphereSurfaceArea (s : Sphere) : ℝ :=
  4 * Real.pi * s.radius^2

/-- Theorem: The ratio of surface areas of a cylinder to a sphere is 3:4
    when the cylinder's base diameter and height equal the sphere's diameter -/
theorem cylinder_sphere_surface_area_ratio
  (s : Sphere) (c : Cylinder)
  (h1 : c.baseRadius = s.radius)
  (h2 : c.height = 2 * s.radius) :
  cylinderSurfaceArea c / sphereSurfaceArea s = 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_surface_area_ratio_l985_98523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l985_98559

/-- The repeating decimal 0.63204̅ expressed as a fraction -/
theorem repeating_decimal_to_fraction :
  (∑' n : ℕ, (632 * 1000^n + 204) / (1000^(3*n+3) : ℚ)) = 2106598 / 3333000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l985_98559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l985_98561

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (2 : ℝ)^b < (2 : ℝ)^a ∧ (2 : ℝ)^a < (3 : ℝ)^a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l985_98561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f₁_not_even_nor_odd_f₂_min_value_l985_98501

-- Part 1
def f₁ (x : ℝ) : ℝ := x^2 + |x - 2| - 1

theorem f₁_not_even_nor_odd :
  ∀ x : ℝ, f₁ x ≠ f₁ (-x) ∧ f₁ x ≠ -f₁ (-x) :=
by sorry

-- Part 2
def f₂ (a x : ℝ) : ℝ := x^2 + |x - a| + 1

noncomputable def min_value (a : ℝ) : ℝ :=
  if a < -1/2 then 3/4 - a
  else if a ≤ 1/2 then a^2 + 1
  else 3/4 + a

theorem f₂_min_value :
  ∀ a x : ℝ, f₂ a x ≥ min_value a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f₁_not_even_nor_odd_f₂_min_value_l985_98501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_equals_two_l985_98534

-- Define the function g as noncomputable
noncomputable def g (a b x : ℝ) : ℝ := 1 / (2 * a * x + 3 * b)

-- State the theorem
theorem inverse_g_equals_two (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ x, g a b x = 1/2) → b = (1 - 4*a) / 3 :=
by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_equals_two_l985_98534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l985_98597

noncomputable def a : ℕ → ℝ := sorry
noncomputable def b : ℕ → ℝ := sorry

def is_geometric (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem sequence_formula :
  (∀ n : ℕ, n % 2 = 1 → b n = a ((n + 1) / 2)) →
  (∀ n : ℕ, n % 2 = 0 → b n = (a (n + 1))^(1/2)) →
  is_geometric b →
  a 2 + b 2 = 108 →
  ∀ n : ℕ, a n = 9^n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l985_98597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_isosceles_triangles_l985_98531

/-- A function that checks if three numbers can form an isosceles triangle -/
def is_isosceles (a b c : ℕ) : Bool :=
  (a = b ∧ a + b > c) ∨ (a = c ∧ a + c > b) ∨ (b = c ∧ b + c > a)

/-- The set of all three-digit numbers abc where a, b, and c can form an isosceles triangle -/
def isosceles_triangles : Finset ℕ :=
  Finset.filter (fun n => 
    let a := n / 100
    let b := (n / 10) % 10
    let c := n % 10
    a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 1 ∧ b ≤ 9 ∧ c ≥ 1 ∧ c ≤ 9 ∧ is_isosceles a b c
  ) (Finset.range 1000)

/-- Theorem stating that the number of three-digit numbers forming isosceles triangles is 165 -/
theorem count_isosceles_triangles : 
  Finset.card isosceles_triangles = 165 := by
  sorry

#eval Finset.card isosceles_triangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_isosceles_triangles_l985_98531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_ones_value_l985_98574

/-- The probability of rolling exactly two 1s when rolling twelve 6-sided dice -/
def prob_two_ones : ℚ :=
  (Nat.choose 12 2) * (1/6)^2 * (5/6)^10

/-- Theorem stating that the probability of rolling exactly two 1s
    when rolling twelve 6-sided dice is equal to 1031250/2184664 -/
theorem prob_two_ones_value : prob_two_ones = 1031250/2184664 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_ones_value_l985_98574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_condition_implies_b_bound_l985_98532

open Real

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (log x + (x - b)^2) / x

-- State the theorem
theorem function_condition_implies_b_bound (b : ℝ) :
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f b x + x * (deriv (f b)) x > 0) →
  b < 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_condition_implies_b_bound_l985_98532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l985_98589

-- Define the function f as noncomputable
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^(2*k)

-- State the theorem
theorem range_of_f (k : ℝ) (h : k ≥ 1) :
  Set.range (fun x => f k x) ∩ Set.Ici 2 = Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l985_98589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_largest_volume_l985_98521

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical pillar -/
structure CylindricalPillar where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylindrical pillar -/
noncomputable def cylinderVolume (p : CylindricalPillar) : ℝ :=
  Real.pi * p.radius^2 * p.height

/-- Checks if a cylindrical pillar fits in a crate -/
def fitsInCrate (c : CrateDimensions) (p : CylindricalPillar) : Prop :=
  2 * p.radius ≤ c.length ∧ 2 * p.radius ≤ c.width ∧ p.height ≤ c.height

/-- The main theorem stating that a cylinder is the shape with the largest volume -/
theorem cylinder_largest_volume 
  (c : CrateDimensions) 
  (h_dimensions : c.length = 2 ∧ c.width = 8 ∧ c.height = 12) 
  (p : CylindricalPillar) 
  (h_radius : p.radius = 2) 
  (h_fits : fitsInCrate c p) : 
  ∀ (q : CylindricalPillar), fitsInCrate c q → cylinderVolume p ≥ cylinderVolume q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_largest_volume_l985_98521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_max_value_of_g_l985_98507

-- Problem 1
noncomputable def f (x : ℝ) : ℝ := (1/2)^(-x^2 + 4*x + 1)

theorem range_of_f :
  ∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc (1/32) (1/2) :=
by sorry

-- Problem 2
noncomputable def g (a x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

theorem max_value_of_g (a : ℝ) :
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, g a x ≥ g a y) ∧
  (∃ x ∈ Set.Icc 0 1, g a x = 2) →
  a = -1 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_max_value_of_g_l985_98507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_data_properties_l985_98511

variable (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)

def is_ordered (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) : Prop :=
  x₁ ≤ x₂ ∧ x₂ ≤ x₃ ∧ x₃ ≤ x₄ ∧ x₄ ≤ x₅ ∧ x₅ ≤ x₆

noncomputable def median_subset (x₂ x₃ x₄ x₅ : ℝ) : ℝ := (x₃ + x₄) / 2

noncomputable def median_full (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) : ℝ := (x₃ + x₄) / 2

def range_subset (x₂ x₅ : ℝ) : ℝ := x₅ - x₂

def range_full (x₁ x₆ : ℝ) : ℝ := x₆ - x₁

theorem sample_data_properties
  (h : is_ordered x₁ x₂ x₃ x₄ x₅ x₆) :
  (median_subset x₂ x₃ x₄ x₅ = median_full x₁ x₂ x₃ x₄ x₅ x₆) ∧
  (range_subset x₂ x₅ ≤ range_full x₁ x₆) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_data_properties_l985_98511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_with_same_colored_multiples_l985_98528

/-- A set of 10 different prime numbers -/
def P : Finset ℕ := sorry

/-- The set of integers > 1 whose prime decomposition only contains primes from P -/
def A : Set ℕ := {n : ℕ | n > 1 ∧ ∀ p, Nat.Prime p → p ∣ n → p ∈ P}

/-- The coloring function for elements of A -/
noncomputable def color : A → ℕ := sorry

/-- Each element of P has a different color -/
axiom different_colors : ∀ p q : ℕ, p ∈ P → q ∈ P → p ≠ q → color ⟨p, sorry⟩ ≠ color ⟨q, sorry⟩

/-- If m, n ∈ A, then mn is the same color as m or n -/
axiom multiplicative_property : ∀ m n : A, color (⟨m * n, sorry⟩ : A) = color m ∨ color (⟨m * n, sorry⟩ : A) = color n

/-- The third coloring condition -/
axiom no_cross_divisibility : ∀ (j k m n : A) (c₁ c₂ : ℕ),
  c₁ ≠ c₂ →
  color j = c₁ → color k = c₁ → color m = c₂ → color n = c₂ →
  ¬(j.val ∣ m.val ∧ n.val ∣ k.val)

/-- The main theorem -/
theorem exists_prime_with_same_colored_multiples :
  ∃ p ∈ P, ∀ m : A, p ∣ m.val → color m = color ⟨p, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_with_same_colored_multiples_l985_98528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_tan_theta_l985_98564

/-- A triangle with side lengths 6, 8, and 10 -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 6
  hb : b = 8
  hc : c = 10

/-- A line that bisects both the perimeter and area of a triangle -/
structure BisectingLine (T : Triangle) where
  bisects_perimeter : ℝ → Prop
  bisects_area : ℝ → Prop

/-- The existence of exactly two bisecting lines for the given triangle -/
axiom exists_two_bisecting_lines (T : Triangle) :
  ∃! (L1 L2 : BisectingLine T), L1 ≠ L2

/-- The acute angle between the two bisecting lines -/
noncomputable def angle_between_lines (T : Triangle) (L1 L2 : BisectingLine T) : ℝ :=
  sorry

/-- The tangent of the acute angle between the two bisecting lines -/
noncomputable def tan_theta (T : Triangle) (L1 L2 : BisectingLine T) : ℝ :=
  Real.tan (angle_between_lines T L1 L2)

/-- The main theorem: the existence and uniqueness of tan θ -/
theorem exists_unique_tan_theta (T : Triangle) :
  ∃! θ, ∃ (L1 L2 : BisectingLine T), L1 ≠ L2 ∧ θ = tan_theta T L1 L2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_tan_theta_l985_98564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_sum_l985_98576

/-- Given that 5i/(2-i) = a + bi where a and b are real numbers and i is the imaginary unit, prove that a + b = 1 -/
theorem complex_fraction_sum (a b : ℝ) : (5 * Complex.I) / (2 - Complex.I) = Complex.ofReal a + Complex.I * Complex.ofReal b → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_sum_l985_98576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_solution_range_l985_98552

theorem cos_sin_equation_solution_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ioo 0 (Real.pi / 2) ∧ Real.cos x ^ 2 + Real.sin x + a = 0) ↔ 
  a ∈ Set.Icc (-5/4) (-1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_solution_range_l985_98552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_over_x_l985_98566

def complex_number (x y : ℝ) : ℂ := x + (y - 2) * Complex.I

theorem range_of_y_over_x (x y : ℝ) :
  Complex.abs (complex_number x y) = Real.sqrt 3 →
  (y / x < -Real.sqrt 3 / 3 ∨ y / x > Real.sqrt 3 / 3) ∨ 
  (y / x = -Real.sqrt 3 / 3 ∨ y / x = Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_over_x_l985_98566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_digits_2500_odd_integers_l985_98580

/-- The nth positive odd integer -/
def nthOddInteger (n : ℕ) : ℕ := 2 * n - 1

/-- Count of digits in a natural number -/
def digitCount (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

/-- Sum of digits for odd integers up to n -/
def sumDigitsOddIntegers (n : ℕ) : ℕ :=
  (List.range n).map (fun i => digitCount (nthOddInteger (i + 1))) |>.sum

theorem total_digits_2500_odd_integers :
  sumDigitsOddIntegers 2500 = 9445 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_digits_2500_odd_integers_l985_98580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_set_partition_l985_98563

theorem finite_set_partition (S : Finset ℝ) :
  ∃ (A B : Finset ℝ), A ∪ B = S ∧ A ∩ B = ∅ ∧
  (∀ x y, x ∈ A → y ∈ A → ∀ k : ℤ, |x - y| ≠ 3^k) ∧
  (∀ x y, x ∈ B → y ∈ B → ∀ k : ℤ, |x - y| ≠ 3^k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_set_partition_l985_98563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_integers_l985_98506

theorem no_special_integers : ¬∃ (a b c d : ℕ+), 
  (Nat.Coprime a.val b.val ∧ Nat.Coprime a.val c.val ∧ Nat.Coprime a.val d.val ∧
   Nat.Coprime b.val c.val ∧ Nat.Coprime b.val d.val ∧ Nat.Coprime c.val d.val) ∧
  (Odd (a.val * b.val + c.val * d.val) ∧
   Odd (a.val * c.val + b.val * d.val) ∧
   Odd (a.val * d.val + b.val * c.val)) ∧
  ((a.val * b.val + c.val * d.val) ∣ ((a.val + b.val - c.val - d.val) * (a.val - b.val + c.val - d.val) * (a.val - b.val - c.val + d.val))) ∧
  ((a.val * c.val + b.val * d.val) ∣ ((a.val + b.val - c.val - d.val) * (a.val - b.val + c.val - d.val) * (a.val - b.val - c.val + d.val))) ∧
  ((a.val * d.val + b.val * c.val) ∣ ((a.val + b.val - c.val - d.val) * (a.val - b.val + c.val - d.val) * (a.val - b.val - c.val + d.val))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_integers_l985_98506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_b_leq_2_l985_98542

-- Define the function f as noncomputable
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := -1/2 * x^2 + b * Real.log x

-- State the theorem
theorem decreasing_f_implies_b_leq_2 :
  ∀ b : ℝ, (∀ x y : ℝ, Real.sqrt 2 ≤ x ∧ x < y → f b y < f b x) → b ≤ 2 := by
  sorry

-- You can add more theorems or lemmas here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_b_leq_2_l985_98542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l985_98558

theorem largest_expression :
  let a := Real.sqrt (Real.rpow 6 (1/3) * Real.rpow 7 (1/3))
  let b := Real.sqrt (7 * Real.rpow 6 (1/3))
  let c := Real.sqrt (6 * Real.rpow 7 (1/3))
  let d := Real.rpow (7 * Real.sqrt 6) (1/3)
  let e := Real.rpow (6 * Real.sqrt 7) (1/3)
  b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l985_98558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_salary_increase_l985_98516

/-- Calculates the percentage increase between two values -/
noncomputable def percentageIncrease (original : ℝ) (new : ℝ) : ℝ :=
  ((new - original) / original) * 100

theorem john_salary_increase : 
  let original := (30 : ℝ)
  let new := (40 : ℝ)
  abs (percentageIncrease original new - 33.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_salary_increase_l985_98516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l985_98520

-- Define set M
def M : Set ℝ := {x | x^2 < 4}

-- Define set N
def N : Set ℝ := {x | ∃ α : ℝ, x = Real.sin α}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l985_98520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_value_l985_98502

theorem definite_integral_value : 
  ∫ x in (Set.Icc 0 1), (3 * x^2 + Real.exp x + 1) = Real.exp 1 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_value_l985_98502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ingrid_initial_flowers_l985_98560

theorem ingrid_initial_flowers (collins_initial_flowers petals_per_flower collins_total_petals : ℕ) :
  collins_initial_flowers = 25 →
  petals_per_flower = 4 →
  collins_total_petals = 144 →
  3 * (collins_total_petals - collins_initial_flowers * petals_per_flower) / petals_per_flower = 33 := by
  intros h1 h2 h3
  -- The proof goes here
  sorry

#eval (3 * (144 - 25 * 4) / 4) -- This should evaluate to 33

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ingrid_initial_flowers_l985_98560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_P_l985_98583

-- Define the points
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (-3, 2)
def C : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (4, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem y_coordinate_of_P (P : ℝ × ℝ) 
  (h1 : distance P A + distance P D = 10)
  (h2 : distance P B + distance P C = 10) :
  P.2 = 6/7 := by
  sorry

#check y_coordinate_of_P

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_P_l985_98583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_meeting_time_l985_98547

/-- Two people walking towards each other meet at the calculated time -/
theorem walking_meeting_time (start_time : ℕ) (distance : ℝ) (speed_a speed_b : ℝ) 
  (h1 : start_time = 13) -- 1 pm in 24-hour format
  (h2 : distance = 24)
  (h3 : speed_a = 5)
  (h4 : speed_b = 7) :
  ↑start_time + (distance / (speed_a + speed_b)) = 15 := by -- 3 pm in 24-hour format
  sorry

#check walking_meeting_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_meeting_time_l985_98547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l985_98525

-- Define the function g as noncomputable
noncomputable def g (A B C : ℤ) (x : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

-- State the theorem
theorem sum_of_coefficients 
  (A B C : ℤ) 
  (h1 : ∀ x > 3, g A B C x > (1/2 : ℝ)) : 
  A + B + C = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l985_98525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_expression_l985_98568

/-- 
Given that the terminal side of angle α passes through the point P(4a,3a) where a < 0, 
prove that 25sinα - 7tan2α = -1497/55.
-/
theorem angle_terminal_side_expression (a : ℝ) (α : ℝ) (h : a < 0) : 
  (4 * a : ℝ) * Real.sin α = 3 * a ∧ (3 * a : ℝ) * Real.cos α = 4 * a → 
  25 * Real.sin α - 7 * Real.tan (2 * α) = -1497 / 55 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_expression_l985_98568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_line_l985_98573

-- Define points
def P : ℝ × ℝ := (0, 1)
def A : ℝ × ℝ := (3, 3)
def B : ℝ × ℝ := (5, -1)

-- Define the distance function between a point and a line
noncomputable def distancePointLine (p : ℝ × ℝ) (m k : ℝ) : ℝ :=
  |m * p.1 - p.2 + k| / Real.sqrt (m^2 + 1)

-- Theorem statement
theorem equidistant_line :
  ∃ (m k : ℝ), 
    (m * P.1 + k = P.2) ∧ 
    (distancePointLine A m k = distancePointLine B m k) ∧
    ((m = 0 ∧ k = 1) ∨ (m = -2 ∧ k = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_line_l985_98573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l985_98513

-- Define the function as noncomputable
noncomputable def f (a : ℝ) : ℝ := 2 / a - 1 / (1 + a)

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), max = -3 - 2 * Real.sqrt 2 ∧
  ∀ (a : ℝ), -1 < a → a < 0 → f a ≤ max :=
by
  -- The proof is omitted using 'sorry'
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l985_98513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l985_98596

noncomputable section

variable (f : ℝ → ℝ)

axiom f_add (x y : ℝ) : f (x + y) = f x + f y - 2
axiom f_pos (x : ℝ) : x > 0 → f x > 2

theorem f_properties :
  (f 0 = 2) ∧
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ t : ℝ, f (2*t^2 - t - 3) - 2 < 0 ↔ -1 < t ∧ t < 3/2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l985_98596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_faucets_fill_time_l985_98567

noncomputable def faucet_rate (tub_size : ℝ) (num_faucets : ℝ) (fill_time : ℝ) : ℝ :=
  tub_size / (num_faucets * fill_time)

noncomputable def fill_time (tub_size : ℝ) (num_faucets : ℝ) (rate : ℝ) : ℝ :=
  tub_size / (num_faucets * rate)

theorem six_faucets_fill_time :
  let initial_tub_size : ℝ := 100
  let initial_num_faucets : ℝ := 3
  let initial_fill_time : ℝ := 6
  let new_tub_size : ℝ := 25
  let new_num_faucets : ℝ := 6
  let rate := faucet_rate initial_tub_size initial_num_faucets initial_fill_time
  fill_time new_tub_size new_num_faucets rate * 60 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_faucets_fill_time_l985_98567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_not_sum_increasing_sum_increasing_exponential_condition_g_tangent_at_zero_g_is_sum_increasing_l985_98529

/-- A function f is "∑ increasing" if for any real numbers s and t in (0, +∞),
    f(s+t) > f(s) + f(t) always holds. -/
def is_sum_increasing (f : ℝ → ℝ) : Prop :=
  ∀ (s t : ℝ), s > 0 → t > 0 → f (s + t) > f s + f t

/-- The sine function is not "∑ increasing". -/
theorem sin_not_sum_increasing : ¬ is_sum_increasing Real.sin := by sorry

/-- For the function f(x) = 2^(x-1) - x - a to be "∑ increasing",
    a must be greater than or equal to 1/2. -/
theorem sum_increasing_exponential_condition (a : ℝ) :
  is_sum_increasing (λ x ↦ 2^(x-1) - x - a) → a ≥ 1/2 := by sorry

/-- Define g(x) = e^x * ln(1+x) -/
noncomputable def g (x : ℝ) : ℝ := Real.exp x * Real.log (1 + x)

/-- The x-coordinate of the point where the tangent line to g(x) is y = x is 0. -/
theorem g_tangent_at_zero :
  ∃ (x₀ : ℝ), x₀ = 0 ∧ (deriv g) x₀ = 1 := by sorry

/-- The function g(x) = e^x * ln(1+x) is "∑ increasing". -/
theorem g_is_sum_increasing : is_sum_increasing g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_not_sum_increasing_sum_increasing_exponential_condition_g_tangent_at_zero_g_is_sum_increasing_l985_98529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_proof_l985_98512

/-- A cone with surface area 12π m² and lateral surface in the shape of a semicircle -/
structure Cone where
  surface_area : ℝ
  lateral_surface_shape : String
  surface_area_eq : surface_area = 12 * Real.pi
  lateral_surface_is_semicircle : lateral_surface_shape = "semicircle"

/-- The volume of the cone -/
noncomputable def cone_volume (c : Cone) : ℝ := (8 * Real.sqrt 3 * Real.pi) / 3

/-- Proof that the volume of the cone is (8√3π)/3 -/
theorem cone_volume_proof (c : Cone) : cone_volume c = (8 * Real.sqrt 3 * Real.pi) / 3 := by
  -- Unfold the definition of cone_volume
  unfold cone_volume
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_proof_l985_98512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_at_origin_l985_98556

open Real

-- Define the circle and point S
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
def S : ℝ × ℝ := (1, 0)

-- Define the positions of cockroaches as functions of time
noncomputable def posA (t : ℝ) : ℝ × ℝ := (cos t, sin t)
noncomputable def posB (t : ℝ) : ℝ × ℝ := (cos (2*t), sin (2*t))
noncomputable def posC (t : ℝ) : ℝ × ℝ := (cos (3*t), sin (3*t))

-- Define points X and Y on segment SC
noncomputable def X (t : ℝ) : ℝ × ℝ := ((2 + cos (3*t))/3, sin (3*t)/3)
noncomputable def Y (t : ℝ) : ℝ × ℝ := ((1 + 2*cos (3*t))/3, 2*sin (3*t)/3)

-- Define point Z as the intersection of AX and BY
noncomputable def Z (t : ℝ) : ℝ × ℝ := sorry

-- Define the centroid of triangle ZAB
noncomputable def centroid (t : ℝ) : ℝ × ℝ := 
  ((Z t).1 + (posA t).1 + (posB t).1, (Z t).2 + (posA t).2 + (posB t).2) 

theorem centroid_at_origin :
  ∀ t : ℝ, centroid t = (0, 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_at_origin_l985_98556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_graph_has_cycle_four_l985_98526

/-- A graph representing scientists at a conference -/
structure ConferenceGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  vertex_count : vertices.card = 50
  min_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 25

/-- Definition of a cycle of length 4 in the graph -/
def HasCycleFour (g : ConferenceGraph) : Prop :=
  ∃ a b c d, a ∈ g.vertices ∧ b ∈ g.vertices ∧ c ∈ g.vertices ∧ d ∈ g.vertices ∧
    (a, b) ∈ g.edges ∧ (b, c) ∈ g.edges ∧ (c, d) ∈ g.edges ∧ (d, a) ∈ g.edges ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a

/-- Theorem stating that a conference graph always contains a cycle of length 4 -/
theorem conference_graph_has_cycle_four (g : ConferenceGraph) : HasCycleFour g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_graph_has_cycle_four_l985_98526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_and_product_l985_98562

-- Define the variables a and b
variable (a b : Real)

-- State the theorem
theorem relationship_and_product (h1 : (2 : Real)^a = 3) (h2 : (3 : Real)^b = 2) : a > b ∧ a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_and_product_l985_98562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_numbers_stats_l985_98538

noncomputable def average (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let avg := average xs
  (xs.map (fun x => (x - avg)^2)).sum / xs.length

theorem eight_numbers_stats (original_seven : List ℝ) (s_squared : ℝ) :
  average original_seven = 3 →
  variance original_seven = s_squared →
  let new_eight := original_seven ++ [3]
  average new_eight = 3 ∧
  s_squared = 4 ∧
  variance new_eight = 7/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_numbers_stats_l985_98538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_accuracy_l985_98585

-- Define a type for population
def Population : Type := ℝ

-- Define a type for sample
def Sample (n : ℕ) : Type := Fin n → ℝ

-- Define a measure of estimation accuracy
noncomputable def EstimationAccuracy : ℝ → ℝ := sorry

-- Define a function that estimates a population parameter from a sample
noncomputable def Estimate (n : ℕ) (s : Sample n) : ℝ := sorry

-- Statement: Increasing sample size leads to more accurate estimation
theorem sample_size_accuracy (p : Population) (n m : ℕ) (h : n < m) :
  EstimationAccuracy (Estimate n (λ i => p)) ≤ EstimationAccuracy (Estimate m (λ i => p)) := by
  sorry

#check sample_size_accuracy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_accuracy_l985_98585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_triangle_area_l985_98503

/-- A circle with a given radius. -/
structure Circle where
  radius : ℝ

/-- A triangle. -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The area of a triangle. -/
def Triangle.area (T : Triangle) : ℝ := sorry

/-- A triangle is inscribed in a circle. -/
def Triangle.inscribed_in (T : Triangle) (C : Circle) : Prop := sorry

/-- A triangle has one side as a diameter of the circle. -/
def Triangle.has_diameter_side (T : Triangle) (C : Circle) : Prop := sorry

/-- The area of the largest inscribed triangle in a circle with radius 8 cm, 
    where one side of the triangle is a diameter of the circle, is 64 square centimeters. -/
theorem largest_inscribed_triangle_area (D : Circle) (h : D.radius = 8) :
  ∃ (T : Triangle), 
    T.inscribed_in D ∧ 
    T.has_diameter_side D ∧
    T.area = 64 ∧
    (∀ (T' : Triangle), T'.inscribed_in D ∧ T'.has_diameter_side D → T'.area ≤ T.area) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_triangle_area_l985_98503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_costs_l985_98554

/-- Represents the cost and profit information for a product type --/
structure ProductInfo where
  initialProfit : Float
  finalProfit : Float
  costReduction : Float
  priceReduction : Float

/-- Calculates the cost of a product given its information --/
def calculateCost (info : ProductInfo) : Float :=
  info.priceReduction / (info.initialProfit - info.finalProfit * (1 - info.costReduction))

/-- The main theorem stating the costs of products A and B --/
theorem product_costs :
  let productA : ProductInfo := {
    initialProfit := 0.40,
    finalProfit := 0.50,
    costReduction := 0.30,
    priceReduction := 14.50
  }
  let productB : ProductInfo := {
    initialProfit := 0.45,
    finalProfit := 0.55,
    costReduction := 0.35,
    priceReduction := 18.75
  }
  (Float.abs (calculateCost productA - 41.43) < 0.01) ∧ 
  (Float.abs (calculateCost productB - 42.37) < 0.01) := by
  sorry

#eval calculateCost {
  initialProfit := 0.40,
  finalProfit := 0.50,
  costReduction := 0.30,
  priceReduction := 14.50
}

#eval calculateCost {
  initialProfit := 0.45,
  finalProfit := 0.55,
  costReduction := 0.35,
  priceReduction := 18.75
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_costs_l985_98554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_perpendicular_slope_one_l985_98571

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the right focus of the hyperbola
noncomputable def right_focus (a b : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 + b^2), 0)

-- Define the asymptote in the first quadrant
def asymptote (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b / a) * x

-- Define a point on the hyperbola
def on_hyperbola (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  hyperbola a b p.1 p.2

-- Define perpendicularity
def perpendicular (p q r : ℝ × ℝ) : Prop :=
  (q.1 - p.1) * (r.1 - q.1) + (q.2 - p.2) * (r.2 - q.2) = 0

-- Define vector equality
def vector_eq (p q r s : ℝ × ℝ) : Prop :=
  q.1 - p.1 = s.1 - r.1 ∧ q.2 - p.2 = s.2 - r.2

-- Main theorem
theorem hyperbola_perpendicular_slope_one (a b : ℝ) (A B : ℝ × ℝ) :
  hyperbola a b A.1 A.2 →
  asymptote a b A.1 A.2 →
  on_hyperbola a b B →
  perpendicular (right_focus a b) A B →
  vector_eq (right_focus a b) B B A →
  (B.2 - A.2) / (B.1 - A.1) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_perpendicular_slope_one_l985_98571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_l985_98545

-- Define the region S
noncomputable def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |6 - p.1| + p.2 ≤ 8 ∧ 4 * p.2 - p.1 ≥ 20}

-- Define the axis of revolution
noncomputable def axis (x : ℝ) : ℝ := x / 4 + 5

-- Assume S is a triangle
axiom S_is_triangle : ∃ a b c : ℝ × ℝ, S = {a, b, c}

-- Assume a function for volume of revolution exists
noncomputable def VolumeOfRevolution (S : Set (ℝ × ℝ)) (axis : ℝ → ℝ) : ℝ := sorry

-- State the theorem
theorem volume_of_revolution :
  ∃ V : ℝ, V = VolumeOfRevolution S axis ∧ V = 48 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_l985_98545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mean_score_is_84_l985_98551

/-- Represents a class with a mean score and number of students -/
structure MyClass where
  meanScore : ℝ
  numStudents : ℝ

/-- Calculates the combined mean score of two classes -/
noncomputable def combinedMeanScore (class1 class2 : MyClass) : ℝ :=
  (class1.meanScore * class1.numStudents + class2.meanScore * class2.numStudents) /
  (class1.numStudents + class2.numStudents)

/-- Theorem stating that the combined mean score of the two classes is 84 -/
theorem combined_mean_score_is_84 :
  ∀ (x : ℝ), x > 0 →
  let class1 : MyClass := { meanScore := 90, numStudents := 2 * x }
  let class2 : MyClass := { meanScore := 80, numStudents := 3 * x }
  combinedMeanScore class1 class2 = 84 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mean_score_is_84_l985_98551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_school_time_l985_98504

/-- The boy's usual time to reach school -/
noncomputable def usual_time : ℝ := 28

/-- The boy's increased walking rate -/
noncomputable def increased_rate : ℝ := 7/6

/-- The time saved by walking at the increased rate -/
noncomputable def time_saved : ℝ := 4

theorem boy_school_time :
  usual_time * increased_rate = (usual_time - time_saved) * 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_school_time_l985_98504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_on_unit_circle_l985_98592

theorem cos_double_angle_on_unit_circle (y₀ : ℝ) (α : ℝ) :
  (1/3)^2 + y₀^2 = 1 →
  Real.cos α = 1/3 →
  Real.cos (2 * α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_on_unit_circle_l985_98592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_at_pi_over_4_l985_98517

noncomputable section

-- Define x as a function from ℝ to ℝ
def x : ℝ → ℝ := sorry

-- Define the differential equation
axiom diff_eq : ∀ t : ℝ, (x t + deriv x t)^2 + x t * deriv (deriv x) t = Real.cos t

-- Define initial conditions
axiom initial_cond_x : x 0 = Real.sqrt (2/5)
axiom initial_cond_dx : deriv x 0 = Real.sqrt (2/5)

-- State the theorem
theorem x_at_pi_over_4 : x (Real.pi/4) = Real.sqrt (Real.sqrt 2) / Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_at_pi_over_4_l985_98517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_set_area_bound_l985_98536

/-- A convex set in the plane -/
structure ConvexSet where
  set : Set (ℝ × ℝ)
  convex : Convex ℝ set

/-- Defines a lattice point in the plane -/
def LatticePoint : Set (ℝ × ℝ) :=
  {p | ∃ m n : ℤ, p = (↑m, ↑n)}

/-- The four quadrants of the plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- The intersection of a set with a quadrant -/
def quadrantIntersection (S : Set (ℝ × ℝ)) (q : Quadrant) : Set (ℝ × ℝ) :=
  match q with
  | Quadrant.I => {p ∈ S | p.1 ≥ 0 ∧ p.2 ≥ 0}
  | Quadrant.II => {p ∈ S | p.1 ≤ 0 ∧ p.2 ≥ 0}
  | Quadrant.III => {p ∈ S | p.1 ≤ 0 ∧ p.2 ≤ 0}
  | Quadrant.IV => {p ∈ S | p.1 ≥ 0 ∧ p.2 ≤ 0}

/-- The area of a set in the plane -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem -/
theorem convex_set_area_bound (S : ConvexSet) :
  (0, 0) ∈ S.set →
  (∀ p ∈ LatticePoint, p ≠ (0, 0) → p ∉ S.set) →
  (∀ q1 q2 : Quadrant, area (quadrantIntersection S.set q1) = area (quadrantIntersection S.set q2)) →
  area S.set ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_set_area_bound_l985_98536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l985_98557

-- Define the function representing the left side of the inequality
noncomputable def f (x : ℝ) := x * (x + 1) / ((x - 9) ^ 2)

-- Define the solution set
def S : Set ℝ := {x | (7.05 ≤ x ∧ x < 9) ∨ (9 < x ∧ x ≤ 12.30)}

-- Theorem statement
theorem inequality_solution : 
  ∀ x : ℝ, x ≠ 9 → (f x ≥ 15 ↔ x ∈ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l985_98557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l985_98577

/-- The present value of a machine given its future value, depletion rate, and time -/
noncomputable def present_value (future_value : ℝ) (depletion_rate : ℝ) (time : ℝ) : ℝ :=
  future_value / ((1 - depletion_rate) ^ time)

/-- Theorem: The present value of a machine with 10% annual depletion rate and $972 value after 2 years is $1200 -/
theorem machine_present_value :
  let future_value : ℝ := 972
  let depletion_rate : ℝ := 0.1
  let time : ℝ := 2
  present_value future_value depletion_rate time = 1200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l985_98577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_movement_distance_l985_98584

/-- A transformation consisting of a dilation followed by a translation -/
structure DilationTranslation (n : Type*) [NormedAddCommGroup n] [InnerProductSpace ℝ n] where
  dilation : n → n
  translation : n → n

/-- The theorem statement -/
theorem origin_movement_distance
  (initial_center : EuclideanSpace ℝ (Fin 2))
  (initial_radius : ℝ)
  (final_center : EuclideanSpace ℝ (Fin 2))
  (final_radius : ℝ)
  (transform : DilationTranslation (EuclideanSpace ℝ (Fin 2)))
  (h1 : initial_center = ![3, 3])
  (h2 : initial_radius = 3)
  (h3 : final_center = ![7, 10])
  (h4 : final_radius = 5)
  (h5 : transform.dilation initial_center = ![5, 5])
  (h6 : transform.translation (transform.dilation initial_center) = final_center) :
  ‖(0 : EuclideanSpace ℝ (Fin 2)) - transform.translation 0‖ = Real.sqrt 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_movement_distance_l985_98584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_to_n_value_l985_98595

theorem m_to_n_value (m n : ℤ) (h1 : (3 : ℝ)^m = 1/27) (h2 : (1/2 : ℝ)^n = 16) : 
  (m : ℝ)^n = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_to_n_value_l985_98595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doubleSeries_eq_four_thirds_l985_98524

/-- The sum of the double infinite series ∑∑ 4^(-k - 2j - (k + j)^2) for j, k from 0 to infinity -/
noncomputable def doubleSeries : ℝ :=
  ∑' (j : ℕ), ∑' (k : ℕ), (4 : ℝ) ^ (-(k : ℝ) - 2 * (j : ℝ) - (k + j : ℝ)^2)

/-- The double infinite series equals 4/3 -/
theorem doubleSeries_eq_four_thirds : doubleSeries = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doubleSeries_eq_four_thirds_l985_98524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_dilution_l985_98505

def initial_volume : ℝ := 11
def initial_alcohol_percentage : ℝ := 42
def added_water : ℝ := 3

theorem alcohol_dilution (initial_volume : ℝ) (initial_alcohol_percentage : ℝ) (added_water : ℝ) :
  initial_volume > 0 ∧ initial_alcohol_percentage > 0 ∧ initial_alcohol_percentage < 100 ∧ added_water > 0 →
  (initial_volume * (initial_alcohol_percentage / 100)) / (initial_volume + added_water) * 100 = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_dilution_l985_98505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_increase_approx_ten_percent_l985_98569

/-- The percentage increase from an initial value to a final value -/
noncomputable def percentageIncrease (initial : ℝ) (final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

theorem percentage_increase_approx_ten_percent :
  let initial : ℝ := 499.99999999999994
  let final : ℝ := 550
  abs (percentageIncrease initial final - 10) < 0.0000000000012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_increase_approx_ten_percent_l985_98569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_condition4_range_of_a_all_conditions_l985_98550

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) := -a * x + 1
def g (x : ℝ) := x^2

-- Define the domain
def domain := Set.Ici 2

-- Define the conditions
def condition1 (a : ℝ) := ∀ x ∈ domain, f a x + g x > 0
def condition3 (a : ℝ) := ∀ x ∈ domain, f a x * g x > 0
def condition4 (a : ℝ) := ∀ x ∈ domain, f a x / g x > 0

-- Theorem 1
theorem range_of_a_condition4 :
  {a : ℝ | condition4 a} = Set.Iic 0 :=
sorry

-- Theorem 2
theorem range_of_a_all_conditions :
  {a : ℝ | condition1 a ∧ condition3 a ∧ condition4 a} = Set.Ioo 0 (5/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_condition4_range_of_a_all_conditions_l985_98550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_is_correct_expected_not_purchasing_is_correct_l985_98586

-- Define the probabilities
noncomputable def prob_A : ℝ := 0.5
noncomputable def prob_B_not_A : ℝ := 0.3
def num_owners : ℕ := 100

-- Define the probability of purchasing at least one insurance
noncomputable def prob_at_least_one : ℝ := 1 - (1 - prob_A) * (1 - (prob_B_not_A / (1 - prob_A)))

-- Define the expected number of owners not purchasing either insurance
noncomputable def expected_not_purchasing : ℝ := num_owners * (1 - prob_at_least_one)

-- Theorem for the probability of purchasing at least one insurance
theorem prob_at_least_one_is_correct : prob_at_least_one = 0.8 := by sorry

-- Theorem for the expected number of owners not purchasing either insurance
theorem expected_not_purchasing_is_correct : expected_not_purchasing = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_is_correct_expected_not_purchasing_is_correct_l985_98586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_in_second_group_proportion_below_178_l985_98544

-- Define the frequency distribution
def frequency_distribution : List (Real × Real × Nat) :=
  [(8, 93, 50), (93, 178, 100), (178, 263, 34), (263, 348, 11),
   (348, 433, 1), (433, 518, 1), (518, 603, 2), (603, 688, 1)]

-- Total sample size
def sample_size : Nat := 200

-- Function to calculate cumulative frequency
def cumulative_frequency (n : Nat) : Nat :=
  (frequency_distribution.take n).foldl (fun acc (_, _, freq) => acc + freq) 0

-- Theorem for the median group
theorem median_in_second_group :
  ∃ i, i < frequency_distribution.length ∧
       cumulative_frequency i < sample_size / 2 ∧
       sample_size / 2 ≤ cumulative_frequency (i + 1) ∧
       (frequency_distribution.get! i).1 = 93 ∧
       (frequency_distribution.get! i).2 = 178 := by
  sorry

-- Theorem for the proportion of households below 178 kW•h
theorem proportion_below_178 :
  let below_178 := (frequency_distribution.take 2).foldl (fun acc (_, _, freq) => acc + freq) 0
  (below_178 : Real) / sample_size = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_in_second_group_proportion_below_178_l985_98544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_gcd_lcm_l985_98522

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem count_pairs_gcd_lcm : 
  let pairs := {(x, y) : ℕ × ℕ | x > 0 ∧ y > 0 ∧ Nat.gcd x y = factorial 5 ∧ Nat.lcm x y = factorial 50}
  Finset.card (Finset.filter (λ p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ Nat.gcd p.1 p.2 = factorial 5 ∧ Nat.lcm p.1 p.2 = factorial 50) (Finset.range (factorial 50 + 1) ×ˢ Finset.range (factorial 50 + 1))) = 2^14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_gcd_lcm_l985_98522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expressions_l985_98572

theorem simplify_expressions :
  (- (- (7/2 : ℚ)) = 7/2) ∧
  ((- (21/5 : ℚ)) = - (21/5)) ∧
  (- (- (- (- (3/5 : ℚ)))) = 3/5) ∧
  (- (- (- (3 : ℚ))) = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expressions_l985_98572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_ratio_specific_geometric_series_ratio_l985_98533

/-- For an infinite geometric series with first term a and sum S, 
    the common ratio r is given by r = 1 - (a / S) -/
theorem infinite_geometric_series_ratio 
  (a : ℝ) (S : ℝ) (h1 : a ≠ 0) (h2 : S ≠ 0) (h3 : S > a) :
  let r := 1 - (a / S)
  r * S = S - a ∧ 0 < r ∧ r < 1 := by sorry

/-- The common ratio of an infinite geometric series with first term 500 and sum 2500 is 4/5 -/
theorem specific_geometric_series_ratio :
  let a : ℝ := 500
  let S : ℝ := 2500
  let r := 1 - (a / S)
  r = 4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_ratio_specific_geometric_series_ratio_l985_98533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_second_derivative_at_three_is_zero_l985_98587

/-- Given a differentiable function f and conditions, prove g''(3) = 0 -/
theorem g_second_derivative_at_three_is_zero 
  (f : ℝ → ℝ) 
  (hf_diff : Differentiable ℝ f) 
  (hf_point : f 3 = 1) 
  (hf_tangent : ∃ k : ℝ, (λ x ↦ k * x + 2) = λ x ↦ f x + (deriv f 3) * (x - 3)) 
  (g : ℝ → ℝ) 
  (hg_def : g = λ x ↦ x * f x) : 
  (deriv (deriv g)) 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_second_derivative_at_three_is_zero_l985_98587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_polar_to_cartesian_O1_polar_to_cartesian_O2_l985_98578

-- Define the circles in polar form
noncomputable def circle_O1 (θ : ℝ) : ℝ := 4 * Real.cos θ
noncomputable def circle_O2 (θ : ℝ) : ℝ := -4 * Real.sin θ

-- Define the Cartesian equations of the circles
def cartesian_O1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0
def cartesian_O2 (x y : ℝ) : Prop := y = -x

-- Theorem statement
theorem intersection_line_equation :
  ∀ (x y : ℝ),
  (∃ θ : ℝ, x = circle_O1 θ * Real.cos θ ∧ y = circle_O1 θ * Real.sin θ) ∧
  (∃ θ : ℝ, x = circle_O2 θ * Real.cos θ ∧ y = circle_O2 θ * Real.sin θ) →
  y = -x :=
by sorry

-- Additional theorem to show the conversion from polar to Cartesian for circle_O1
theorem polar_to_cartesian_O1 :
  ∀ (x y : ℝ),
  (∃ θ : ℝ, x = circle_O1 θ * Real.cos θ ∧ y = circle_O1 θ * Real.sin θ) ↔
  cartesian_O1 x y :=
by sorry

-- Additional theorem to show the conversion from polar to Cartesian for circle_O2
theorem polar_to_cartesian_O2 :
  ∀ (x y : ℝ),
  (∃ θ : ℝ, x = circle_O2 θ * Real.cos θ ∧ y = circle_O2 θ * Real.sin θ) ↔
  cartesian_O2 x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_polar_to_cartesian_O1_polar_to_cartesian_O2_l985_98578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_steven_mowing_rate_l985_98591

/-- Represents the farming capabilities of Farmer Steven -/
structure FarmerCapabilities where
  plow_rate : ℚ  -- Acres of farmland plowed per day
  total_days : ℚ  -- Total days to complete both tasks
  farmland_area : ℚ  -- Total acres of farmland
  grassland_area : ℚ  -- Total acres of grassland

/-- Calculates the grassland mowing rate given farming capabilities -/
def mowing_rate (fc : FarmerCapabilities) : ℚ :=
  fc.grassland_area / (fc.total_days - fc.farmland_area / fc.plow_rate)

/-- Theorem stating that Farmer Steven can mow 12 acres of grassland per day -/
theorem farmer_steven_mowing_rate :
  let fc : FarmerCapabilities := {
    plow_rate := 10,
    total_days := 8,
    farmland_area := 55,
    grassland_area := 30
  }
  mowing_rate fc = 12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_steven_mowing_rate_l985_98591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l985_98582

-- Define the train's length in meters
noncomputable def train_length : ℝ := 120

-- Define the train's speed in km/hr
noncomputable def train_speed_km_hr : ℝ := 72

-- Define the conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℝ := 5 / 18

-- Define the time to cross in seconds
noncomputable def time_to_cross : ℝ := 6

-- Theorem statement
theorem train_crossing_time :
  time_to_cross = train_length / (train_speed_km_hr * km_hr_to_m_s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l985_98582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_triangle_area_l985_98540

-- Define the circle
noncomputable def circle_radius : ℝ := 8

-- Define the inscribed triangle
structure InscribedTriangle where
  -- One side is a diameter
  base : ℝ
  height : ℝ
  is_inscribed : base ≤ 2 * circle_radius ∧ height ≤ circle_radius

-- Define the area of a triangle
noncomputable def triangle_area (t : InscribedTriangle) : ℝ :=
  (1/2) * t.base * t.height

-- Theorem statement
theorem largest_inscribed_triangle_area :
  (∃ (t : InscribedTriangle), ∀ (t' : InscribedTriangle), triangle_area t' ≤ triangle_area t) →
  (∃ (t : InscribedTriangle), triangle_area t = 64) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_triangle_area_l985_98540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeans_sale_savings_l985_98575

/-- Represents the total savings from purchasing jeans during a sale. -/
noncomputable def total_savings (F P : ℝ) : ℝ :=
  45 * (F / 100) + 36 * (P / 100)

/-- Theorem stating the total savings formula for purchasing jeans during a sale. -/
theorem jeans_sale_savings (F P : ℝ) (h : F + P = 18) :
  total_savings F P = 45 * (F / 100) + 36 * (P / 100) := by
  -- Unfold the definition of total_savings
  unfold total_savings
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeans_sale_savings_l985_98575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_investment_is_3500_l985_98537

/-- Represents the investment and profit distribution in a business partnership --/
structure BusinessPartnership where
  a_investment : ℚ
  b_investment : ℚ
  total_profit : ℚ
  management_fee_rate : ℚ
  a_total_received : ℚ

/-- Calculates the amount received by partner A in a business partnership --/
noncomputable def amount_received_by_a (bp : BusinessPartnership) : ℚ :=
  let management_fee := bp.management_fee_rate * bp.total_profit
  let remaining_profit := bp.total_profit - management_fee
  let a_share_ratio := bp.a_investment / (bp.a_investment + bp.b_investment)
  management_fee + a_share_ratio * remaining_profit

/-- Theorem stating that given the problem conditions, A's investment is 3500 --/
theorem a_investment_is_3500 :
  ∃ bp : BusinessPartnership,
    bp.b_investment = 1500 ∧
    bp.total_profit = 9600 ∧
    bp.management_fee_rate = 1/10 ∧
    bp.a_total_received = 7008 ∧
    bp.a_investment = 3500 ∧
    amount_received_by_a bp = bp.a_total_received :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_investment_is_3500_l985_98537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_interesting_number_l985_98588

/-- A function that checks if a natural number is interesting --/
def is_interesting (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.length = digits.toFinset.card) ∧ 
  (∀ i : ℕ, i + 1 < digits.length → 
    ∃ k : ℕ, (digits[i]! + digits[i+1]! : ℕ) = k^2)

/-- The theorem stating that 6310972 is the largest interesting number --/
theorem largest_interesting_number : 
  is_interesting 6310972 ∧ 
  ∀ m : ℕ, m > 6310972 → ¬ is_interesting m := by
  sorry

#check largest_interesting_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_interesting_number_l985_98588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l985_98555

/-- An even function that is monotonically increasing on (-∞, 0) -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (|x - b|) / Real.log a

/-- The function is even -/
axiom f_even (a b : ℝ) : ∀ x, f a b x = f a b (-x)

/-- The function is monotonically increasing on (-∞, 0) -/
axiom f_increasing (a b : ℝ) : ∀ x y, x < y → x < 0 → y < 0 → f a b x < f a b y

/-- The main theorem to prove -/
theorem f_inequality (a b : ℝ) : f a b (a + 1) > f a b (b + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l985_98555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l985_98546

theorem inequality_proof (x : ℝ) : 
  -1/2 ≤ x → x < 45/8 → x ≠ 0 → (4 * x^2) / ((1 - Real.sqrt (1 + 2*x))^2) < 2*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l985_98546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l985_98509

-- Define x and y as noncomputable
noncomputable def x : ℝ := Real.log 7 / Real.log 0.1
noncomputable def y : ℝ := (1/2) * Real.log 7

-- State the theorem
theorem log_inequality : (x + y < x * y) ∧ (x + y ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l985_98509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_second_quadrant_l985_98530

theorem cos_alpha_second_quadrant (α : ℝ) :
  (π / 2 < α ∧ α < π) →  -- α is in the second quadrant
  Real.cos (π / 2 - α) = 4 / 5 →
  Real.cos α = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_second_quadrant_l985_98530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l985_98535

noncomputable section

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Definition of the line l₁ -/
def line_l₁ (x y : ℝ) : Prop := y = (1/2) * x

/-- Point P -/
def P : ℝ × ℝ := (2, 1)

/-- Point M -/
def M : ℝ × ℝ := (1, 3/2)

/-- Function to calculate the dot product of two vectors -/
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  (v₁.1 * v₂.1) + (v₁.2 * v₂.2)

/-- Main theorem -/
theorem ellipse_and_line_properties :
  (∀ x y, ellipse_C x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∃ A B : ℝ × ℝ,
    A ≠ B ∧
    ellipse_C A.1 A.2 ∧
    ellipse_C B.1 B.2 ∧
    line_l₁ A.1 A.2 ∧
    line_l₁ B.1 B.2 ∧
    dot_product (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2) =
      dot_product (M.1 - P.1, M.2 - P.2) (M.1 - P.1, M.2 - P.2)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l985_98535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weight_is_6300_l985_98519

/-- Represents the weight of an animal in grams -/
def Weight := ℕ

/-- Conversion factor from kilograms to grams -/
def kg_to_g : ℕ := 1000

/-- Weight of Jung-min's dog in kilograms -/
def dog_weight_kg : ℕ := 2

/-- Additional weight of Jung-min's dog in grams -/
def dog_weight_additional_g : ℕ := 600

/-- Weight of Jung-min's cat in grams -/
def cat_weight_g : ℕ := 3700

/-- Calculates the total weight of Jung-min's animals in grams -/
def total_weight : ℕ :=
  dog_weight_kg * kg_to_g + dog_weight_additional_g + cat_weight_g

/-- Proves that the total weight of Jung-min's animals is 6300 grams -/
theorem total_weight_is_6300 : total_weight = 6300 := by
  unfold total_weight
  unfold dog_weight_kg
  unfold kg_to_g
  unfold dog_weight_additional_g
  unfold cat_weight_g
  norm_num

#eval total_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weight_is_6300_l985_98519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_angle_is_90_degrees_l985_98581

def vector1 : ℝ × ℝ := (3, 4)
def vector2 : ℝ × ℝ := (4, -3)

theorem angle_between_vectors (v1 v2 : ℝ × ℝ) : 
  let dotProduct := v1.1 * v2.1 + v1.2 * v2.2
  let magnitude1 := Real.sqrt (v1.1^2 + v1.2^2)
  let magnitude2 := Real.sqrt (v2.1^2 + v2.2^2)
  let cosTheta := dotProduct / (magnitude1 * magnitude2)
  cosTheta = 0 → Real.arccos cosTheta = π / 2 := by sorry

theorem angle_is_90_degrees : 
  let dotProduct := vector1.1 * vector2.1 + vector1.2 * vector2.2
  let magnitude1 := Real.sqrt (vector1.1^2 + vector1.2^2)
  let magnitude2 := Real.sqrt (vector2.1^2 + vector2.2^2)
  let cosTheta := dotProduct / (magnitude1 * magnitude2)
  cosTheta = 0 ∧ Real.arccos cosTheta = π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_angle_is_90_degrees_l985_98581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_60_max_profit_is_21000_l985_98549

/-- Profit function for the travel agency -/
noncomputable def profit (x : ℝ) : ℝ :=
  if x ≤ 30 then 900 * x - 15000
  else (-10 * x + 1200) * x - 15000

/-- The charter fee for the travel agency -/
def charter_fee : ℝ := 15000

/-- The maximum number of people in the tour group -/
def max_people : ℝ := 75

/-- Theorem stating that the profit is maximized when there are 60 people -/
theorem profit_maximized_at_60 :
  ∀ x, 0 ≤ x ∧ x ≤ max_people → profit x ≤ profit 60 := by
  sorry

/-- The maximum profit achieved by the travel agency -/
noncomputable def max_profit : ℝ := profit 60

/-- Theorem stating that the maximum profit is 21000 yuan -/
theorem max_profit_is_21000 : max_profit = 21000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_60_max_profit_is_21000_l985_98549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outer_inner_squares_area_relation_l985_98553

-- Define a triangle by its side lengths
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_pos : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the medians of a triangle
noncomputable def median (t : Triangle) : ℝ × ℝ × ℝ :=
  let ma := Real.sqrt (2 * t.b^2 + 2 * t.c^2 - t.a^2) / 2
  let mb := Real.sqrt (2 * t.a^2 + 2 * t.c^2 - t.b^2) / 2
  let mc := Real.sqrt (2 * t.a^2 + 2 * t.b^2 - t.c^2) / 2
  (ma, mb, mc)

theorem outer_inner_squares_area_relation (t : Triangle) :
  let (ma, mb, mc) := median t
  t.a^2 + t.b^2 + t.c^2 = 3 * ((2*ma)^2 + (2*mb)^2 + (2*mc)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_outer_inner_squares_area_relation_l985_98553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_sum_l985_98527

-- Define the function f(x)
noncomputable def f (a n x : ℝ) : ℝ := a^(2*x - 4) + n

-- State the theorem
theorem fixed_point_sum (a n m : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  (∀ x, f a n x = 2 ↔ x = m) → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_sum_l985_98527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l985_98579

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := x^2 / (x + 2) - 3 / (x - 2) - 7 / 4

-- Define the solution set
def solution_set : Set ℝ := Set.Ioo (-2) 2 ∪ Set.Ici 3

-- Theorem statement
theorem inequality_solution :
  {x : ℝ | f x ≥ 0} = solution_set := by
  sorry

#check inequality_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l985_98579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stans_average_speed_l985_98543

/-- Calculates the average speed given two driving segments --/
noncomputable def average_speed (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ) : ℝ :=
  (distance1 + distance2) / (time1 + time2)

theorem stans_average_speed :
  let distance1 : ℝ := 350
  let time1 : ℝ := 6
  let distance2 : ℝ := 420
  let time2 : ℝ := 7
  average_speed distance1 time1 distance2 time2 = 770 / 13 := by
  sorry

-- Use Nat instead of ℝ for #eval to work
#eval (350 + 420) / (6 + 7)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stans_average_speed_l985_98543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_ratio_l985_98515

noncomputable section

-- Define the quadratic polynomials
def f₁ (x a : ℝ) : ℝ := x^2 - a*x + 2
def f₂ (x b : ℝ) : ℝ := x^2 + 3*x + b
def f₃ (x a b : ℝ) : ℝ := 3*x^2 + (3 - 2*a)*x + 4 + b
def f₄ (x a b : ℝ) : ℝ := 3*x^2 + (6 - a)*x + 2 + 2*b

-- Define the differences of roots
noncomputable def A (a : ℝ) : ℝ := Real.sqrt (a^2 - 8)
noncomputable def B (b : ℝ) : ℝ := Real.sqrt (9 - 4*b)
noncomputable def C (a b : ℝ) : ℝ := (1/3) * Real.sqrt (4*a^2 - 12*a - 39 - 12*b)
noncomputable def D (a b : ℝ) : ℝ := (1/3) * Real.sqrt (a^2 - 12*a + 12 - 24*b)

-- Theorem statement
theorem root_difference_ratio (a b : ℝ) (h : A a ≠ B b) :
  (C a b)^2 - (D a b)^2 = (1/3) * ((A a)^2 - (B b)^2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_ratio_l985_98515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_and_evaluation_l985_98510

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, 
  x > 0 →
  (3 : ℝ) - x ≥ 0 →
  x ≠ 1 →
  x ≠ 2 →
  (1 - (1 / (x - 1))) / ((x^2 - 4*x + 4) / (x - 1)) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_and_evaluation_l985_98510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_equality_implies_second_sum_l985_98593

/-- Proves that given a total sum of 2795, if it is split into two parts such that the interest on the first part for 8 years at 3% per annum equals the interest on the second part for 3 years at 5% per annum, then the second part is equal to 1720. -/
theorem interest_equality_implies_second_sum (total : ℕ) (first_part : ℕ) (second_part : ℕ) : 
  total = 2795 →
  first_part + second_part = total →
  (first_part * 3 * 8 : ℚ) / 100 = (second_part * 5 * 3 : ℚ) / 100 →
  second_part = 1720 := by
  intro h_total h_sum h_interest
  sorry

#check interest_equality_implies_second_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_equality_implies_second_sum_l985_98593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l985_98570

theorem trigonometric_identity (α : ℝ) : 
  (1 - (Real.tan (2 * α))^2) * 
  (Real.cos (2 * α))^2 * 
  (Real.tan (π / 4 - 2 * α)) + 
  Real.sin (4 * α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l985_98570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_properties_sum_l985_98594

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 3*x) / (x^4 + 2*x^3 - 3*x^2)

-- Define the properties of the function's graph
def num_holes (f : ℝ → ℝ) : ℕ := sorry
def num_vertical_asymptotes (f : ℝ → ℝ) : ℕ := sorry
def num_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := sorry
def num_oblique_asymptotes (f : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem graph_properties_sum :
  let p := num_holes f
  let q := num_vertical_asymptotes f
  let r := num_horizontal_asymptotes f
  let s := num_oblique_asymptotes f
  p + 2*q + 3*r + 4*s = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_properties_sum_l985_98594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l985_98598

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_d : d ≠ 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- Main theorem -/
theorem arithmetic_sequence_sum_ratio
  (seq : ArithmeticSequence)
  (h : S seq 9 = 3 * seq.a 8) :
  S seq 15 / (3 * seq.a 5) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l985_98598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_object_speed_theorem_l985_98508

/-- Calculates the speed in miles per hour given a distance in feet and time in seconds -/
noncomputable def speed_mph (distance_feet : ℝ) (time_seconds : ℝ) : ℝ :=
  let feet_per_second := distance_feet / time_seconds
  let feet_per_mile := 5280
  let seconds_per_hour := 3600
  feet_per_second * (seconds_per_hour / feet_per_mile)

/-- Theorem stating that an object traveling 90 feet in 3 seconds has a speed of approximately 20.45 mph -/
theorem object_speed_theorem :
  let calculated_speed := speed_mph 90 3
  |calculated_speed - 20.45| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_object_speed_theorem_l985_98508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l985_98514

/-- Given vectors a, b, and c in ℝ², prove that if a = (1, 2), 2a + b = (4, 2), 
    c = (1, lambda), and b is parallel to c, then lambda = -1 -/
theorem parallel_vectors_lambda (a b c : ℝ × ℝ) (lambda : ℝ) : 
  a = (1, 2) →
  2 • a + b = (4, 2) →
  c = (1, lambda) →
  ∃ (k : ℝ), k ≠ 0 ∧ b = k • c →
  lambda = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l985_98514
