import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_sine_inequality_l1211_121155

theorem acute_triangle_sine_inequality (α β γ : ℝ) 
  (h_acute : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_sum : α + β + γ = Real.pi) : 
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) ≤ 
  Real.sin (α + β) + Real.sin (β + γ) + Real.sin (γ + α) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_sine_inequality_l1211_121155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cable_length_l1211_121184

/-- The length of a curve defined by the intersection of a plane and a sphere --/
theorem cable_length : 
  ∃ (x y z : ℝ), 
    x + y + z = 8 ∧ 
    x * y + y * z + x * z = 14 → 
    (2 * Real.pi * Real.sqrt (44 / 3) : ℝ) = 4 * Real.pi * Real.sqrt (11 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cable_length_l1211_121184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_removal_possible_l1211_121120

/-- A sequence of 2000 natural numbers, each representing a power of 9 -/
def SandTrips := Fin 2000 → ℕ

/-- The property that each number in the sequence is a power of 9 -/
def IsPowerOf9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9^k

/-- The sum of all elements in a SandTrips sequence -/
def SumTrips (trips : SandTrips) : ℕ := (Finset.univ.sum fun i => trips i)

/-- The theorem stating that there exists a sequence of 2000 powers of 9 summing to 20160000 -/
theorem sand_removal_possible : 
  ∃ (trips : SandTrips), 
    (∀ i, IsPowerOf9 (trips i)) ∧ 
    (SumTrips trips = 20160000) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_removal_possible_l1211_121120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_l1211_121177

theorem triangle_count : ∃! n : ℕ, 
  n = (Finset.filter (λ x : ℕ ↦ 
    2003 < x ∧ x < 2013 ∧  -- Triangle inequality
    (5 + 2008 + x) % 2 = 0 ∧  -- Even perimeter
    5 + x > 2008 ∧ 2008 + x > 5 ∧ 5 + 2008 > x  -- Triangle inequality
  ) (Finset.range 2013)).card ∧ n = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_l1211_121177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_binomial_expansion_l1211_121106

theorem fourth_term_binomial_expansion 
  (a x : ℝ) (h : x ≠ 0) (h' : a ≠ 0) :
  let f := (a^2 / x - x^2 / a^3)^8
  (Finset.range 9).sum (λ k ↦ (Nat.choose 8 k) * (a^2 / x)^(8 - k) * (-x^2 / a^3)^k) =
  -56 * a * x + (Finset.range 9).sum (λ k ↦ if k = 3 then 0 else (Nat.choose 8 k) * (a^2 / x)^(8 - k) * (-x^2 / a^3)^k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_binomial_expansion_l1211_121106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_l1211_121125

theorem rectangle_division (w h : ℕ) (s₁ s₂ s₃ s₄ s₅ : ℕ) : 
  w = 6 → h = 7 → 
  s₁ * s₁ + s₂ * s₂ + s₃ * s₃ + s₄ * s₄ + s₅ * s₅ = w * h → 
  ∃ (a b c d e : ℕ), a = 3 ∧ b = 3 ∧ c = 3 ∧ d = 2 ∧ e = 1 ∧
  s₁ * s₁ = a * a ∧ s₂ * s₂ = b * b ∧ s₃ * s₃ = c * c ∧ s₄ * s₄ = d * d ∧ 
  s₅ * s₅ = e * e := by
  intros hw hh heq
  use 3, 3, 3, 2, 1
  repeat' apply And.intro
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · sorry
  · sorry
  · sorry
  · sorry
  · sorry

#check rectangle_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_l1211_121125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_family_properties_l1211_121149

/-- Family of lines L -/
def L (a b x₀ y₀ x y : ℝ) : Prop :=
  (x₀ * x / a^2) + (y₀ * y / b^2) = 1

/-- Ellipse equation -/
def ellipse (a b x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem line_family_properties
  (a b x₀ y₀ : ℝ)
  (h_ab : a > b)
  (h_b_pos : b > 0) :
  (((x₀^2 / a^2) + (y₀^2 / b^2) > 1 →
    ∃ (triangle_area : ℝ),
      triangle_area = (3 * Real.sqrt 3 / 4) * a * b ∧
      (∀ other_area : ℝ,
        (∃ x y : ℝ, L a b x₀ y₀ x y ∧ ellipse a b x y) →
        other_area ≤ triangle_area)) ∧
  ((x₀^2 / a^2) + (y₀^2 / b^2) = 1 →
    ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
      L a b x₀ y₀ x₁ y₁ ∧
      L a b x₀ y₀ x₂ y₂ ∧
      L a b x₀ y₀ x₃ y₃ ∧
      L a b x₀ y₀ x₄ y₄ ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₂ - x₃)^2 + (y₂ - y₃)^2 ∧
      (x₂ - x₃)^2 + (y₂ - y₃)^2 = (x₃ - x₄)^2 + (y₃ - y₄)^2 ∧
      (x₃ - x₄)^2 + (y₃ - y₄)^2 = (x₄ - x₁)^2 + (y₄ - y₁)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_family_properties_l1211_121149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1211_121172

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_increasing : ∀ n : ℕ, a n < a (n + 1))
  (h_sum_arith : arithmetic_sequence (λ n ↦ sum_arithmetic_sequence a n / a n)) :
  sum_arithmetic_sequence a 3 / a 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1211_121172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_side_range_l1211_121115

theorem acute_triangle_side_range :
  ∀ a : ℝ,
  (∃ A B C : ℝ, 
    0 < A ∧ A < π/2 ∧
    0 < B ∧ B < π/2 ∧
    0 < C ∧ C < π/2 ∧
    A + B + C = π ∧
    3^2 = 4^2 + a^2 - 2*4*a*(Real.cos C) ∧
    4^2 = 3^2 + a^2 - 2*3*a*(Real.cos B) ∧
    a^2 = 3^2 + 4^2 - 2*3*4*(Real.cos A)) ↔
  (Real.sqrt 7 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_side_range_l1211_121115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_three_polynomial_l1211_121196

/-- Polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 - 15*x + 4*x^2 - 5*x^3 + 6*x^4

/-- Polynomial g(x) -/
def g (x : ℝ) : ℝ := 4 - 3*x - 7*x^3 + 12*x^4

/-- The combined polynomial h(x) = f(x) + c g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- Theorem stating that c = -1/2 makes h(x) a polynomial of degree 3 -/
theorem degree_three_polynomial :
  ∃ a b d : ℝ, (∀ x : ℝ, h (-1/2) x = a*x^3 + b*x^2 + d*x + (2 + 4*(-1/2))) ∧ a ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_three_polynomial_l1211_121196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_circles_between_two_l1211_121126

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a function to check if two circles are tangent
def are_tangent (c1 c2 : Circle) : Prop :=
  let d := Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2)
  d = c1.radius + c2.radius

-- Define a function to check if a circle is inside another
def is_inside (inner outer : Circle) : Prop :=
  let d := Real.sqrt ((inner.center.1 - outer.center.1)^2 + (inner.center.2 - outer.center.2)^2)
  d + inner.radius < outer.radius

-- Main theorem
theorem eight_circles_between_two (k1 : Circle) :
  ∃ (k2 : Circle) (c1 c2 c3 c4 c5 c6 c7 c8 : Circle),
    k2.center ≠ k1.center ∧
    is_inside k2 k1 ∧
    (∀ i : Fin 8, are_tangent (match i with
      | ⟨0, _⟩ => c1 | ⟨1, _⟩ => c2 | ⟨2, _⟩ => c3 | ⟨3, _⟩ => c4
      | ⟨4, _⟩ => c5 | ⟨5, _⟩ => c6 | ⟨6, _⟩ => c7 | ⟨7, _⟩ => c8
      | _ => c1
    ) k1) ∧
    (∀ i : Fin 8, are_tangent (match i with
      | ⟨0, _⟩ => c1 | ⟨1, _⟩ => c2 | ⟨2, _⟩ => c3 | ⟨3, _⟩ => c4
      | ⟨4, _⟩ => c5 | ⟨5, _⟩ => c6 | ⟨6, _⟩ => c7 | ⟨7, _⟩ => c8
      | _ => c1
    ) k2) ∧
    (∀ i : Fin 8, are_tangent (match i with
      | ⟨0, _⟩ => c1 | ⟨1, _⟩ => c2 | ⟨2, _⟩ => c3 | ⟨3, _⟩ => c4
      | ⟨4, _⟩ => c5 | ⟨5, _⟩ => c6 | ⟨6, _⟩ => c7 | ⟨7, _⟩ => c8
      | _ => c1
    ) (match (i + 1) % 8 with
      | ⟨0, _⟩ => c1 | ⟨1, _⟩ => c2 | ⟨2, _⟩ => c3 | ⟨3, _⟩ => c4
      | ⟨4, _⟩ => c5 | ⟨5, _⟩ => c6 | ⟨6, _⟩ => c7 | ⟨7, _⟩ => c8
      | _ => c1
    )) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_circles_between_two_l1211_121126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1211_121140

/-- Given an ellipse with semi-minor axis b, where 0 < b < 1, 
    and P(m,n) is the circumcenter of the triangle formed by 
    the left focus, top vertex, and right vertex of the ellipse, 
    lying to the lower left of the line y = -x, 
    prove that the eccentricity e satisfies √2/2 < e < 1 -/
theorem ellipse_eccentricity_range (b : ℝ) (m n c : ℝ) 
  (h1 : 0 < b) (h2 : b < 1) 
  (h3 : m + n < 0) -- P is to the lower left of y = -x
  (h4 : m = (1 - c) / 2) -- m coordinate of circumcenter
  (h5 : n = (b^2 - c) / (2*b)) -- n coordinate of circumcenter
  (h6 : c^2 = 1 - b^2) -- relationship between c and b in an ellipse
  : ∃ (e : ℝ), Real.sqrt 2 / 2 < e ∧ e < 1 ∧ e^2 = 1 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1211_121140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_circumscribed_is_equiangular_l1211_121147

-- Define the Polygon and Circle types if they're not already defined in Mathlib
structure Polygon where
  -- Add necessary fields for a polygon

structure Circle where
  -- Add necessary fields for a circle

-- Define the properties as functions
def Polygon.isEquilateral (P : Polygon) : Prop := sorry
def Polygon.isCircumscribedAbout (P : Polygon) (C : Circle) : Prop := sorry
def Polygon.isEquiangular (P : Polygon) : Prop := sorry

theorem equilateral_circumscribed_is_equiangular 
  (P : Polygon) (C : Circle) (h1 : P.isEquilateral) (h2 : P.isCircumscribedAbout C) :
  P.isEquiangular := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_circumscribed_is_equiangular_l1211_121147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l1211_121180

-- Define the polygon
def Polygon : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 ≤ 4 ∧ p.1 + 2 * p.2 ≥ 4 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

-- Define the function to calculate the length of a side
noncomputable def sideLength (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- Theorem statement
theorem longest_side_length :
  ∃ (a b : ℝ × ℝ), a ∈ Polygon ∧ b ∈ Polygon ∧
  sideLength a b = 2 * Real.sqrt 5 ∧
  ∀ (c d : ℝ × ℝ), c ∈ Polygon → d ∈ Polygon →
  sideLength c d ≤ 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l1211_121180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1211_121176

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin (Real.pi / 6 * x) - 2 * a + 3
noncomputable def g (x : ℝ) : ℝ := (2 * x) / (x^2 + x + 2)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ f a x₁ = g x₂) →
  a ∈ Set.Icc (5/4) 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1211_121176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_time_difference_l1211_121183

-- Define the constants
noncomputable def distance : ℝ := 3
noncomputable def jill_speed : ℝ := 12
noncomputable def jack_speed : ℝ := 4

-- Define the arrival times in hours
noncomputable def jill_time : ℝ := distance / jill_speed
noncomputable def jack_time : ℝ := distance / jack_speed

-- Convert the time difference to minutes
noncomputable def time_difference_minutes : ℝ := (jack_time - jill_time) * 60

-- Theorem statement
theorem arrival_time_difference : time_difference_minutes = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_time_difference_l1211_121183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1211_121128

def OA : ℝ × ℝ := (1, -2)
def OB (a : ℝ) : ℝ × ℝ := (a, -1)
def OC (b : ℝ) : ℝ × ℝ := (-b, 0)

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_collinear : ∃ (t : ℝ), OB a - OA = t • (OC b - OA)) :
  (1 / a + 2 / b) ≥ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1211_121128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_length_l1211_121167

/-- The length of a broken line with n vertices above a line segment AB of length 1,
    where each segment forms a 45° angle with AB, is equal to √2. -/
theorem broken_line_length (n : ℕ) : Real.sqrt 2 = Real.sqrt 2 := by
  -- Define the given parameters
  let AB : ℝ := 1
  let angle : ℝ := Real.pi / 4  -- 45° in radians
  
  -- The actual proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_length_l1211_121167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_periodic_l1211_121179

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 2)

theorem f_is_odd_and_periodic : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + 4 * Real.pi) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_periodic_l1211_121179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_m_eq_1_m_value_when_max_is_neg_3_l1211_121119

-- Define the function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m / (x - 1)

-- Part 1: Minimum value when m = 1 and x > 1
theorem min_value_when_m_eq_1 :
  ∀ x > 1, f 1 x ≥ 3 ∧ ∃ x > 1, f 1 x = 3 := by sorry

-- Part 2: Value of m when maximum value is -3 and x < 1
theorem m_value_when_max_is_neg_3 :
  (∃ m > 0, ∀ x < 1, f m x ≤ -3 ∧ ∃ x < 1, f m x = -3) →
  (∃ m > 0, m = 4 ∧ ∀ x < 1, f m x ≤ -3 ∧ ∃ x < 1, f m x = -3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_m_eq_1_m_value_when_max_is_neg_3_l1211_121119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_half_l1211_121110

-- Define the curve
def f (x : ℝ) : ℝ := x^2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 2 * x

-- Define the tangent line slope at a point
def tangent_slope (x : ℝ) : ℝ := f' x

-- Define the inclination angle of the tangent line
noncomputable def inclination_angle (x : ℝ) : ℝ := Real.arctan (tangent_slope x)

-- Theorem statement
theorem tangent_at_half : inclination_angle (1/2) = π/4 := by
  -- Unfold the definitions
  unfold inclination_angle tangent_slope f'
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_half_l1211_121110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l1211_121164

noncomputable section

/-- Definition of an ellipse E with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Definition of eccentricity for an ellipse -/
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

/-- Area of the quadrilateral formed by the vertices of the ellipse -/
def quadrilateralArea (a b : ℝ) : ℝ := 4 * a * b

/-- Definition of a line passing through a point with a given slope -/
def Line (m : ℝ) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.2 - p.2 = m * (q.1 - p.1)}

/-- Condition for three lengths to form a geometric sequence -/
def isGeometricSequence (x y z : ℝ) : Prop := y^2 = x * z

theorem ellipse_and_line_properties
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : eccentricity a b = Real.sqrt 3 / 2)
  (h4 : quadrilateralArea a b = 16)
  (h5 : ∃ (m : ℝ) (M N : ℝ × ℝ),
    M ∈ Ellipse a b ∧
    N ∈ Line m (0, b) ∧
    N.2 = 0 ∧
    isGeometricSequence (Real.sqrt ((0 - N.1)^2 + b^2))
                        (Real.sqrt ((M.1 - 0)^2 + (M.2 - b)^2))
                        (Real.sqrt ((M.1 - N.1)^2 + M.2^2))) :
  (a = 4 ∧ b = 2) ∧ (∃ m : ℝ, m = Real.sqrt 5 / 20) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l1211_121164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_three_snow_days_approx_l1211_121108

noncomputable def probability_snow_day : ℝ := 1 / 5

def days_in_december : ℕ := 31

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

noncomputable def probability_at_most_three_snow_days : ℝ :=
  (Finset.range 4).sum (λ k => binomial_probability days_in_december k probability_snow_day)

theorem probability_at_most_three_snow_days_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |probability_at_most_three_snow_days - 0.809| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_three_snow_days_approx_l1211_121108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1211_121160

/-- The distance between two points (x₁, y₁) and (x₂, y₂) in a 2D plane. -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The theorem stating that the distance between (3, 20) and (12, 3) is √370. -/
theorem distance_between_points : distance 3 20 12 3 = Real.sqrt 370 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1211_121160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1211_121187

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ ∀ q, 0 < q ∧ q < p → ∃ y, f (y + q) ≠ f y) ∧
  (∀ k : ℤ, ∀ x : ℝ, f (k * Real.pi / 2 + Real.pi / 3 - x) = f (k * Real.pi / 2 + Real.pi / 3 + x)) ∧
  (∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ f x₀) ∧
  (∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), f x₀ = 3 / 2) ∧
  (∃ x₁ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x₁ ≤ f x) ∧
  (∃ x₁ ∈ Set.Icc 0 (Real.pi / 2), f x₁ = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1211_121187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequences_l1211_121193

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, b (n + 1) = b n * q

theorem arithmetic_and_geometric_sequences
  (a : ℕ → ℤ) (b : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a3 : a 3 = -6)
  (h_a6 : a 6 = 0)
  (h_geom : geometric_sequence b)
  (h_b1 : b 1 = 8)
  (h_b2 : b 2 = a 1 + a 2 + a 3) :
  (∀ n : ℕ, a n = 2 * n - 12) ∧
  (∀ n : ℕ, (Finset.range n).sum (fun i => b (i + 1)) = 2 - 2 * (-3)^n) :=
by
  sorry

#check arithmetic_and_geometric_sequences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequences_l1211_121193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_spherical_segment_eq_l1211_121105

/-- The half-angle of the cone associated with a floating spherical segment -/
noncomputable def half_angle_spherical_segment (s : ℝ) : ℝ :=
  Real.arccos ((-1 + Real.sqrt (1 + 8 * s)) / 2)

/-- Theorem: The half-angle of the cone associated with a floating spherical segment
    with specific gravity s satisfies the given equation. -/
theorem half_angle_spherical_segment_eq (s : ℝ) (h : 0 < s ∧ s < 1) :
  Real.cos (half_angle_spherical_segment s) = (-1 + Real.sqrt (1 + 8 * s)) / 2 := by
  sorry

#check half_angle_spherical_segment_eq

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_spherical_segment_eq_l1211_121105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_triangles_property_l1211_121168

-- Define the triangle and points
variable (A B C P Q R : ℂ)

-- Define the angles
noncomputable def angle_PBC : ℝ := Real.pi/4
noncomputable def angle_CAQ : ℝ := Real.pi/4
noncomputable def angle_BCP : ℝ := Real.pi/6
noncomputable def angle_QCA : ℝ := Real.pi/6
noncomputable def angle_ABR : ℝ := Real.pi/12
noncomputable def angle_RAB : ℝ := Real.pi/12

-- Define the construction of external triangles
def construct_external_triangles (A B C P Q R : ℂ) : Prop :=
  (P - B) = (C - B) * (Complex.exp (Complex.I * angle_PBC)) ∧
  (Q - C) = (A - C) * (Complex.exp (Complex.I * angle_QCA)) ∧
  (R - A) = (B - A) * (Complex.exp (Complex.I * angle_RAB))

-- State the theorem
theorem external_triangles_property 
  (h : construct_external_triangles A B C P Q R) :
  Complex.arg ((R - Q) / (R - P)) = Real.pi/2 ∧ Complex.abs (R - Q) = Complex.abs (R - P) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_triangles_property_l1211_121168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l1211_121174

/-- Given a right triangle GHI with angle G = 40°, angle H = 90°, and side IH = 12,
    prove that side GH is approximately 14.3 -/
theorem right_triangle_side_length (G H I : ℝ × ℝ) (angleG angleH : ℝ) (IH : ℝ) :
  angleG = 40 * π / 180 →
  angleH = π / 2 →
  IH = 12 →
  Real.sqrt ((H.1 - I.1)^2 + (H.2 - I.2)^2) = IH →
  (G.1 - I.1)^2 + (G.2 - I.2)^2 = (G.1 - H.1)^2 + (G.2 - H.2)^2 + (H.1 - I.1)^2 + (H.2 - I.2)^2 →
  |Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2) - 14.3| < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l1211_121174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1211_121131

variable (a b : ℝ)

def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) : ℝ := 4 * a^2 + 6 * a * b + 8 * a

theorem problem_solution :
  (∀ a b, 2 * A a b - B a b = -16 * a * b + 6 * b - 8 * a) ∧
  (A (-1) 2 - B (-1) 2 = 52) ∧
  (∀ a, 2 * A a (-1/2) - B a (-1/2) = 2 * A 0 (-1/2) - B 0 (-1/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1211_121131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_condition_l1211_121145

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sin x / x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m * Real.cos x - x

-- Define the theorem
theorem unique_root_condition (m : ℝ) : 
  (∃! x : ℝ, x ∈ Set.Ioo 0 (3 * Real.pi / 2) ∧ m * f x = g m x) ↔ m > 9 * Real.pi^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_condition_l1211_121145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leibniz_formula_l1211_121142

/-- Leibniz formula for triangles -/
theorem leibniz_formula (A B C P : EuclideanSpace ℝ (Fin 2)) :
  let G := (1/3 : ℝ) • (A + B + C)
  (dist P A)^2 + (dist P B)^2 + (dist P C)^2 =
    3 * (dist P G)^2 + (1/3) * ((dist A B)^2 + (dist B C)^2 + (dist C A)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leibniz_formula_l1211_121142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_minimum_l1211_121132

theorem vector_difference_minimum (a b : ℝ × ℝ) : 
  (a.1 * b.1 + a.2 * b.2 = -1) →
  (a.1 * b.1 + a.2 * b.2 = (Real.sqrt 3 / 2) * Real.sqrt ((a.1^2 + a.2^2) * (b.1^2 + b.2^2))) →
  (∀ c d : ℝ × ℝ, (c.1 * d.1 + c.2 * d.2 = -1) → 
    (c.1 * d.1 + c.2 * d.2 = (Real.sqrt 3 / 2) * Real.sqrt ((c.1^2 + c.2^2) * (d.1^2 + d.2^2))) →
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≤ Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2)) →
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_minimum_l1211_121132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_problem_l1211_121194

/-- The chord length cut by a line on a circle -/
noncomputable def chord_length (a b c : ℝ) (x₀ y₀ r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - (a * x₀ + b * y₀ + c)^2 / (a^2 + b^2))

/-- The problem statement -/
theorem chord_length_problem :
  chord_length 1 (-1) 3 (-2) 2 (Real.sqrt 2) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_problem_l1211_121194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_max_area_l1211_121171

theorem isosceles_trapezoid_max_area (d : ℝ) (α : ℝ) (h : 0 < α ∧ α < π / 2) :
  let x := d / (2 * (2 - Real.cos α))
  ∀ y : ℝ, y > 0 → 
    (d - 2*y) * y * Real.sin α + y^2 * Real.sin α * Real.cos α ≤ 
    (d - 2*x) * x * Real.sin α + x^2 * Real.sin α * Real.cos α :=
by
  intro x y hy
  sorry  -- The actual proof would go here

#check isosceles_trapezoid_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_max_area_l1211_121171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_ratio_theorem_l1211_121169

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Checks if a point is on a line segment -/
def isOnSegment (P Q R : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = Point.mk (Q.x + t * (R.x - Q.x)) (Q.y + t * (R.y - Q.y))

/-- Checks if two line segments intersect -/
def intersect (P Q R S : Point) : Prop :=
  ∃ I : Point, isOnSegment I P Q ∧ isOnSegment I R S

theorem parallelogram_ratio_theorem (EFGH : Parallelogram) (Q R S : Point) :
  isOnSegment Q EFGH.E EFGH.F →
  isOnSegment R EFGH.E EFGH.H →
  intersect EFGH.E EFGH.G Q R →
  (EFGH.E.x - Q.x) / (EFGH.F.x - EFGH.E.x) = 13 / 500 →
  (EFGH.E.y - Q.y) / (EFGH.F.y - EFGH.E.y) = 13 / 500 →
  (EFGH.E.x - R.x) / (EFGH.H.x - EFGH.E.x) = 13 / 1003 →
  (EFGH.E.y - R.y) / (EFGH.H.y - EFGH.E.y) = 13 / 1003 →
  (EFGH.G.x - EFGH.E.x) / (S.x - EFGH.E.x) = 1003000 / 16289037 ∧
  (EFGH.G.y - EFGH.E.y) / (S.y - EFGH.E.y) = 1003000 / 16289037 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_ratio_theorem_l1211_121169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_geometric_sum_l1211_121195

/-- The sum of the first k terms of a geometric sequence -/
noncomputable def geometricSum (a1 : ℝ) (q : ℝ) (k : ℕ) : ℝ :=
  (a1 * (1 - q^k)) / (1 - q)

/-- Theorem: The sum of the first k terms of a specific geometric sequence is 364 -/
theorem specific_geometric_sum :
  ∃ k : ℕ, k > 0 ∧ geometricSum 1 3 k = 364 ∧ 1 * 3^(k-1) = 243 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_geometric_sum_l1211_121195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1211_121173

noncomputable def F (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2
noncomputable def h (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

noncomputable def h_inv (x : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 1))

noncomputable def φ (x : ℝ) : ℝ := g (x - 1)

theorem problem_solution :
  (∀ x : ℝ, F x = g x + h x) →
  (∀ x : ℝ, g (-x) = g x) →
  (∀ x : ℝ, h (-x) = -h x) →
  (∀ x : ℝ, h (h_inv x) = x) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 3 → φ (2*a + 1) > φ (-a/2)) → 
    a ∈ Set.Icc (-1) (-2/5) ∪ Set.Ioc (2/3) 1) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Ioo 0 2 → g (2*x) - a * h x ≥ 0) → a ≤ 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1211_121173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_46_l1211_121199

theorem repeating_decimal_46 : ∃ (x : ℚ), x = 46 / 99 ∧ ∀ (n : ℕ), (x * 10^n - (x * 10^n).floor) * 100 = 46 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_46_l1211_121199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_sector_l1211_121163

/-- The volume of a cone formed by rolling up a half-sector of a circle -/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) : 
  (1/3) * π * (r/2)^2 * Real.sqrt (r^2 - (r/2)^2) = 9 * π * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_sector_l1211_121163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheburashka_eating_time_l1211_121190

/-- The time it takes Gena to eat the entire cake -/
noncomputable def gena_time : ℝ := 2

/-- Cheburashka's eating rate relative to Gena's -/
noncomputable def cheburashka_rate : ℝ := 1 / 2

/-- The time Cheburashka started eating before Gena -/
noncomputable def head_start : ℝ := 1

/-- The fraction of the cake each ate -/
noncomputable def cake_fraction : ℝ := 1 / 2

/-- Theorem stating that Cheburashka's eating time for the whole cake is 4 minutes -/
theorem cheburashka_eating_time :
  (1 / cheburashka_rate) * gena_time = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheburashka_eating_time_l1211_121190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_midpoint_negative_l1211_121151

open Real

-- Define the functions f and g
noncomputable def f (a b x : ℝ) : ℝ := a * exp x - x + b
noncomputable def g (x : ℝ) : ℝ := x - log (x + 1)

-- State the theorem
theorem derivative_at_midpoint_negative
  (a b x₁ x₂ : ℝ)
  (ha : a > 0)
  (hx : x₁ < x₂)
  (hf₁ : f a b x₁ = 0)
  (hf₂ : f a b x₂ = 0) :
  deriv (f a b) ((x₁ + x₂) / 2) < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_midpoint_negative_l1211_121151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_set_partition_impossible_nine_set_partition_possible_l1211_121134

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define a type for a partition of the plane
def Partition (n : ℕ) := Point → Fin n

-- Theorem for 3 disjoint sets
theorem three_set_partition_impossible :
  ¬∃ (p : Partition 3), ∀ (a b : Point),
    p a = p b → distance a b ≠ 1 := by
  sorry

-- Theorem for 9 disjoint sets
theorem nine_set_partition_possible :
  ∃ (p : Partition 9), ∀ (a b : Point),
    p a = p b → distance a b ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_set_partition_impossible_nine_set_partition_possible_l1211_121134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1211_121159

/-- The function f(x) defined as |2x + a| + |x - 1/a| -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |2*x + a| + |x - 1/a|

theorem f_properties (a : ℝ) (h : a < 0) :
  (f a 0 > 5/2 ↔ a < -2 ∨ -1/2 < a) ∧
  (∀ x : ℝ, f a x ≥ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1211_121159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_inequalities_l1211_121165

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_even : ∀ x, f (-x) = f x
axiom g_even : ∀ x, g (-x) = g x
axiom f_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y
axiom g_increasing : ∀ x y, 0 ≤ x → x ≤ y → g x ≤ g y

-- Define function F
def F (x : ℝ) : ℝ := f x + g (1 - x) - |f x - g (1 - x)|

-- State the theorem
theorem F_inequalities (a : ℝ) (ha : 0 < a) : F (-a) ≥ F a ∧ F (1 + a) ≥ F (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_inequalities_l1211_121165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l1211_121191

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add this case to cover Nat.zero
  | 1 => 1
  | n + 2 => -2 / (sequence_a (n + 1) + 3)

theorem sequence_a_general_term (n : ℕ) (h : n ≥ 1) : 
  sequence_a n = 2 / (3 * 2^(n-1) - 2) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l1211_121191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_positive_integer_l1211_121143

def sequence_a : ℕ → ℤ
  | 0 => 3
  | 1 => 2
  | 2 => 12
  | (n + 3) => (sequence_a (n + 2) + 8 * sequence_a (n + 1) - 4 * sequence_a n) / 2

theorem sequence_a_positive_integer : ∀ n : ℕ, sequence_a n > 0 ∧ Int.natAbs (sequence_a n) = sequence_a n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_positive_integer_l1211_121143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l1211_121162

/-- A sequence a is geometric if there exists a common ratio r such that a_(n+1) = r * a_n for all n -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence {a_n}, if a_4 * a_6 = 5, then a_2 * a_3 * a_7 * a_8 = 25 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : IsGeometricSequence a) 
  (h1 : a 4 * a 6 = 5) : a 2 * a 3 * a 7 * a 8 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l1211_121162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l1211_121166

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ :=
  abs (c₂ - c₁) / Real.sqrt (a₁^2 + b₁^2)

/-- Proof that the distance between 3x + y - 3 = 0 and 6x + 2y + 1 = 0 is (7/20) * √10 -/
theorem distance_specific_lines :
  distance_between_parallel_lines 3 1 (-3) 6 2 (-1) = (7/20) * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l1211_121166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insurance_allocation_l1211_121103

/-- Calculates the insurance allocation for a sales representative based on their earnings and budget allocation. -/
theorem insurance_allocation
  (basic_salary_per_hour : ℚ)
  (commission_rate : ℚ)
  (hours_worked : ℚ)
  (total_sales : ℚ)
  (budget_allocation_rate : ℚ)
  (h1 : basic_salary_per_hour = 7.5)
  (h2 : commission_rate = 0.16)
  (h3 : hours_worked = 160)
  (h4 : total_sales = 25000)
  (h5 : budget_allocation_rate = 0.95)
  : ℚ := by
  let total_earnings := basic_salary_per_hour * hours_worked + commission_rate * total_sales
  let insurance_allocation := total_earnings * (1 - budget_allocation_rate)
  have : insurance_allocation = 260 := by
    -- Proof steps would go here
    sorry
  exact 260


end NUMINAMATH_CALUDE_ERRORFEEDBACK_insurance_allocation_l1211_121103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_puzzle_solution_l1211_121118

def ArithmeticOperation : Type := ℕ → ℕ → ℕ

def Equation (op : ArithmeticOperation) (a b c : ℕ) : Prop :=
  op a b = c

theorem arithmetic_puzzle_solution :
  ∃ (A B C D E : ArithmeticOperation),
    (Equation A 4 2 2) ∧
    (Equation C 4 2 8) ∧
    (Equation D 2 3 5) ∧
    (Equation E 5 1 4) ∧
    (A = (λ x y ↦ x / y)) ∧
    (B = (λ x y ↦ x)) ∧
    (C = (λ x y ↦ x * y)) ∧
    (D = (λ x y ↦ x + y)) ∧
    (E = (λ x y ↦ x - y)) := by
  sorry

#check arithmetic_puzzle_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_puzzle_solution_l1211_121118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_cos_is_sin_l1211_121114

/-- The original cosine function -/
noncomputable def f (x : ℝ) : ℝ := Real.cos x

/-- The amount of right translation -/
noncomputable def translation : ℝ := Real.pi / 2

/-- The resulting function after translation -/
noncomputable def g (x : ℝ) : ℝ := f (x - translation)

/-- Theorem stating that the translated cosine function is equivalent to sine -/
theorem translated_cos_is_sin :
  ∀ x : ℝ, g x = Real.sin x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_cos_is_sin_l1211_121114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_combinations_l1211_121144

theorem flower_combinations : 
  let total_amount : ℕ := 60
  let tulip_cost : ℕ := 4
  let sunflower_cost : ℕ := 3
  (Finset.filter (fun (p : ℕ × ℕ) => 
    tulip_cost * p.1 + sunflower_cost * p.2 = total_amount ∧
    p.2 > p.1
  ) (Finset.product (Finset.range (total_amount + 1)) (Finset.range (total_amount + 1)))).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_combinations_l1211_121144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1211_121107

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (3 - a * x) / Real.log a

noncomputable def g (x : ℝ) : ℝ := f 3 x - f 3 (-x)

theorem problem_solution :
  (∀ x : ℝ, f 3 x ≠ 0 → x < 1) ∧
  (∀ x : ℝ, -1 < x ∧ x < 1 → g (-x) = -g x) ∧
  (∃ a : ℝ, 0 < a ∧ a < 1 ∧
    (∀ x y : ℝ, 2 ≤ x ∧ x < y ∧ y ≤ 3 → f a x < f a y) ∧
    (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f a x ≤ 1) ∧
    (∃ x : ℝ, 2 ≤ x ∧ x ≤ 3 ∧ f a x = 1) ∧
    a = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1211_121107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometria_schools_l1211_121100

/-- The number of schools in Geometria -/
def num_schools : ℕ := 20

/-- The total number of students in the contest -/
def total_students : ℕ := 4 * num_schools

/-- Andrea's rank from the top -/
def andrea_rank : ℕ := 3 * num_schools

/-- Beth's rank from the top -/
def beth_rank : ℕ := 20

/-- Carla's rank from the top -/
def carla_rank : ℕ := 47

/-- David's rank from the top -/
def david_rank : ℕ := 78

theorem geometria_schools :
  (∀ (school : ℕ), school ≤ num_schools → ∃! (team : Fin 4 → ℕ), ∀ (i j : Fin 4), i ≠ j → team i ≠ team j) ∧
  (andrea_rank ≤ total_students) ∧
  (andrea_rank = (3 * total_students + 1) / 4) ∧
  (∀ (teammate_rank : ℕ), teammate_rank ∈ ({beth_rank, carla_rank, david_rank} : Set ℕ) → andrea_rank < teammate_rank) ∧
  (beth_rank < carla_rank) ∧
  (carla_rank < david_rank) ∧
  (david_rank < total_students) →
  num_schools = 20 := by
  sorry

#check geometria_schools

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometria_schools_l1211_121100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1211_121170

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 / (x + 1)

-- Define the closed interval [0, 5]
def I : Set ℝ := Set.Icc 0 5

-- State the theorem
theorem f_max_min :
  (∃ (x : ℝ), x ∈ I ∧ ∀ (y : ℝ), y ∈ I → f y ≤ f x) ∧
  (∃ (x : ℝ), x ∈ I ∧ ∀ (y : ℝ), y ∈ I → f x ≤ f y) ∧
  (∀ (x : ℝ), x ∈ I → f x ≤ 3) ∧
  (∀ (x : ℝ), x ∈ I → 1/2 ≤ f x) := by
  sorry

#check f_max_min

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1211_121170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_solution_uniqueness_l1211_121127

/-- The differential equation y'' - 4y' + 4y = 0 -/
def diff_eq (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv y)) x - 4 * (deriv y x) + 4 * (y x) = 0

/-- The general solution of the differential equation -/
noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  (C₁ + C₂ * x) * Real.exp (2 * x)

/-- Theorem stating that the general solution satisfies the differential equation -/
theorem general_solution_satisfies_diff_eq (C₁ C₂ : ℝ) :
  diff_eq (general_solution C₁ C₂) := by
  sorry

/-- Theorem stating that any solution of the differential equation
    can be expressed in the form of the general solution -/
theorem solution_uniqueness (y : ℝ → ℝ) (h : diff_eq y) :
  ∃ C₁ C₂ : ℝ, ∀ x, y x = general_solution C₁ C₂ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_solution_uniqueness_l1211_121127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pipe_height_l1211_121141

/-- The height of a hollow cylindrical pipe without top or bottom faces. -/
noncomputable def pipe_height (SA : ℝ) (r_outer r_inner : ℝ) : ℝ :=
  SA / (2 * Real.pi * (r_outer + r_inner))

/-- Theorem: The height of a specific hollow cylindrical pipe is 25/8 feet. -/
theorem specific_pipe_height :
  pipe_height (50 * Real.pi) 5 3 = 25 / 8 := by
  -- Unfold the definition of pipe_height
  unfold pipe_height
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pipe_height_l1211_121141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abigail_initial_money_l1211_121178

noncomputable def initial_money (food_percentage : ℝ) (phone_percentage : ℝ) (entertainment_cost : ℝ) (remaining : ℝ) : ℝ :=
  let after_entertainment := remaining + entertainment_cost
  let after_phone := after_entertainment / (1 - phone_percentage)
  let after_food := after_phone / (1 - phone_percentage)
  after_food / (1 - food_percentage)

theorem abigail_initial_money :
  initial_money 0.6 0.25 20 40 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abigail_initial_money_l1211_121178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_roots_is_zero_l1211_121148

/-- The area of a triangle with sides equal to the roots of x³ - 6x² + 11x - 6 = 0 is zero -/
theorem area_of_triangle_with_roots_is_zero (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 0 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_roots_is_zero_l1211_121148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_range_l1211_121102

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (a - 1) * x + 1
noncomputable def g (x : ℝ) : ℝ := x * (Real.exp x - 1)

-- State the theorem
theorem extreme_value_and_range (a : ℝ) :
  (∀ x > 0, ¬∃ y, f a y = f a x ∧ ∀ z > 0, f a z ≤ f a x) ∨
  (∃ x > 0, f a x = -Real.log (a - 1) ∧ ∀ y > 0, f a y ≤ f a x ∧ ¬∃ z > 0, ∀ w > 0, f a w ≥ f a z) ∧
  (∀ x > 0, g x ≥ f a x ↔ a ≥ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_range_l1211_121102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_problem_l1211_121129

/-- The number of blue balls in the second urn -/
def N : ℕ := 5

/-- The probability of drawing a green ball from the first urn -/
noncomputable def p_green_1 : ℝ := 5 / 10

/-- The probability of drawing a blue ball from the first urn -/
noncomputable def p_blue_1 : ℝ := 5 / 10

/-- The probability of drawing a green ball from the second urn -/
noncomputable def p_green_2 : ℝ := 20 / (20 + N)

/-- The probability of drawing a blue ball from the second urn -/
noncomputable def p_blue_2 : ℝ := N / (20 + N)

/-- The probability of drawing the same color from both urns -/
noncomputable def p_same_color : ℝ := p_green_1 * p_green_2 + p_blue_1 * p_blue_2

theorem urn_problem : p_same_color = 0.60 → N = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_problem_l1211_121129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_initial_speed_time_l1211_121136

/-- Represents the marathon running scenario -/
structure MarathonRun where
  initialSpeed : ℚ
  slowerSpeed : ℚ
  totalDistance : ℚ
  totalTime : ℚ

/-- Calculates the time spent at the initial speed -/
def timeAtInitialSpeed (run : MarathonRun) : ℚ :=
  (run.totalDistance - run.slowerSpeed * run.totalTime) / (run.initialSpeed - run.slowerSpeed)

/-- Theorem stating that for the given conditions, the time at initial speed is 4 hours -/
theorem marathon_initial_speed_time (run : MarathonRun) 
  (h1 : run.initialSpeed = 6)
  (h2 : run.slowerSpeed = 4)
  (h3 : run.totalDistance = 52)
  (h4 : run.totalTime = 11) :
  timeAtInitialSpeed run = 4 := by
  sorry

def exampleRun : MarathonRun := {
  initialSpeed := 6,
  slowerSpeed := 4,
  totalDistance := 52,
  totalTime := 11
}

#eval timeAtInitialSpeed exampleRun

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_initial_speed_time_l1211_121136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_evenness_oddness_f_specific_range_l1211_121152

-- Define the function f(x) = x|x + a|
def f (a : ℝ) (x : ℝ) : ℝ := x * abs (x + a)

-- Theorem for the evenness/oddness of f(x)
theorem f_evenness_oddness (a : ℝ) :
  (a = 0 ∧ ∀ x, f a (-x) = -f a x) ∨
  (a ≠ 0 ∧ ∃ x, f a (-x) ≠ -f a x ∧ f a (-x) ≠ f a x) :=
sorry

-- Define the specific function f(x) = x|x + 4|
def f_specific (x : ℝ) : ℝ := f 4 x

-- Theorem for the range of f(x) in the interval [-4, 1]
theorem f_specific_range :
  Set.range (fun x ↦ f_specific x) ∩ Set.Icc (-4) 1 = Set.Icc (-4) 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_evenness_oddness_f_specific_range_l1211_121152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clearance_sale_discount_l1211_121153

noncomputable def original_price : ℝ := 30
noncomputable def original_gain_percent : ℝ := 30
noncomputable def sale_gain_percent : ℝ := 17

noncomputable def cost_price : ℝ := original_price / (1 + original_gain_percent / 100)

noncomputable def sale_price : ℝ := cost_price * (1 + sale_gain_percent / 100)

noncomputable def discount_percent : ℝ := (original_price - sale_price) / original_price * 100

theorem clearance_sale_discount :
  abs (discount_percent - 9.99) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clearance_sale_discount_l1211_121153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_at_135_chord_equation_when_bisected_l1211_121116

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define point P_0
def P_0 : ℝ × ℝ := (-1, 2)

-- Define a chord AB passing through P_0
def chord_AB (α : ℝ) (x y : ℝ) : Prop :=
  ∃ (t : ℝ), x = -1 + t * Real.cos α ∧ y = 2 + t * Real.sin α

-- Theorem 1: Length of chord AB when α = 135°
theorem chord_length_at_135 :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    chord_AB (135 * π / 180) x₁ y₁ ∧ chord_AB (135 * π / 180) x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 30 :=
sorry

-- Theorem 2: Equation of chord AB when bisected by P_0
theorem chord_equation_when_bisected :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    ∃ (α : ℝ), chord_AB α x₁ y₁ ∧ chord_AB α x₂ y₂ ∧
    (x₁ + x₂) / 2 = -1 ∧ (y₁ + y₂) / 2 = 2 ∧
    x₁ - 2*y₁ + 5 = 0 ∧ x₂ - 2*y₂ + 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_at_135_chord_equation_when_bisected_l1211_121116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1211_121111

/-- The speed of a train in kilometers per hour, given its length in meters and the time it takes to cross a pole in seconds. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: A train with a length of 500 meters that crosses a pole in 18 seconds has a speed of 100 kilometers per hour. -/
theorem train_speed_problem :
  train_speed 500 18 = 100 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the arithmetic expression
  simp [div_mul_eq_mul_div]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1211_121111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l1211_121192

noncomputable def enclosed_area : ℝ := 18 * Real.pi - 9

def square_side_length : ℝ := 3

noncomputable def segment_length : ℝ := Real.sqrt 18

theorem enclosed_area_theorem : 
  ∀ (s : ℝ) (l : ℝ),
  s = square_side_length →
  l = segment_length →
  enclosed_area = s^2 - 4 * (Real.pi * (s * Real.sqrt 2 / 2)^2 / 4) :=
by
  intros s l h1 h2
  sorry

#check enclosed_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l1211_121192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clerical_staff_percentage_l1211_121138

theorem clerical_staff_percentage (total_employees : ℕ) 
  (initial_clerical_fraction : ℚ) (clerical_reduction_fraction : ℚ) :
  total_employees = 3600 →
  initial_clerical_fraction = 1/4 →
  clerical_reduction_fraction = 1/4 →
  let initial_clerical := (initial_clerical_fraction * total_employees).floor
  let reduced_clerical := ((1 - clerical_reduction_fraction) * initial_clerical).floor
  let remaining_employees := total_employees - (initial_clerical - reduced_clerical)
  (reduced_clerical : ℚ) / remaining_employees * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clerical_staff_percentage_l1211_121138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_m_value_l1211_121185

/-- The line equation: 2mx - y - 8m - 3 = 0 -/
def line (m x y : ℝ) : Prop := 2*m*x - y - 8*m - 3 = 0

/-- The circle equation: (x - 3)^2 + (y + 6)^2 = 25 -/
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 6)^2 = 25

/-- The theorem stating that when the chord AB is the shortest, m = 1/6 -/
theorem shortest_chord_m_value :
  ∀ m : ℝ,
  (∃ A B : ℝ × ℝ, 
    line m A.1 A.2 ∧ 
    line m B.1 B.2 ∧
    circle_eq A.1 A.2 ∧ 
    circle_eq B.1 B.2 ∧
    A ≠ B ∧
    (∀ C D : ℝ × ℝ, 
      line m C.1 C.2 → 
      line m D.1 D.2 → 
      circle_eq C.1 C.2 → 
      circle_eq D.1 D.2 → 
      C ≠ D → 
      (A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ (C.1 - D.1)^2 + (C.2 - D.2)^2)) →
  m = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_m_value_l1211_121185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_implies_tan_x_eq_one_angle_pi_third_implies_x_eq_five_pi_twelfth_l1211_121101

noncomputable section

-- Define the vectors m and n
def m : ℝ × ℝ := (Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
def n (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the angle between two vectors
def angle (v w : ℝ × ℝ) : ℝ := Real.arccos (dot_product v w / (magnitude v * magnitude w))

-- Theorem 1: If m ⊥ n, then tan x = 1
theorem orthogonal_implies_tan_x_eq_one (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) :
  dot_product m (n x) = 0 → Real.tan x = 1 := by sorry

-- Theorem 2: If the angle between m and n is π/3, then x = 5π/12
theorem angle_pi_third_implies_x_eq_five_pi_twelfth (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) :
  angle m (n x) = Real.pi / 3 → x = 5 * Real.pi / 12 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_implies_tan_x_eq_one_angle_pi_third_implies_x_eq_five_pi_twelfth_l1211_121101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_seven_eighths_l1211_121157

/-- A rectangle with midpoints on adjacent sides -/
structure RectangleWithMidpoints where
  length : ℝ
  width : ℝ
  length_pos : 0 < length
  width_pos : 0 < width

/-- The fraction of the rectangle that is shaded -/
noncomputable def shaded_fraction (r : RectangleWithMidpoints) : ℝ :=
  1 - (r.length * r.width / 4) / (r.length * r.width)

/-- Theorem: The shaded fraction is always 7/8 -/
theorem shaded_fraction_is_seven_eighths (r : RectangleWithMidpoints) :
  shaded_fraction r = 7 / 8 := by
  sorry

#check shaded_fraction_is_seven_eighths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_seven_eighths_l1211_121157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_equality_l1211_121109

def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N : Set ℝ := {x | Real.exp ((x + 1) * Real.log 2) ≥ 1}

theorem set_intersection_equality : M ∩ N = {x | -1 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_equality_l1211_121109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_product_is_four_l1211_121135

/-- A point on a hyperbola -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  on_hyperbola : (2 * x^2 / 3) - (y^2 / 6) = 1

/-- The product of slopes of tangents from a point to a circle -/
noncomputable def tangent_slopes_product (p : HyperbolaPoint) : ℝ :=
  (p.y^2 - 2) / (p.x^2 - 2)

/-- Theorem: The product of slopes of tangents from a point on the given hyperbola
    to the given circle is always 4 -/
theorem tangent_slopes_product_is_four (p : HyperbolaPoint) 
  (outside_circle : p.x^2 + p.y^2 > 2) : tangent_slopes_product p = 4 := by
  sorry

#check tangent_slopes_product_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_product_is_four_l1211_121135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_signs_l1211_121113

theorem sum_of_signs (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a + b + c = 0) :
  Real.sign a + Real.sign b + Real.sign c + Real.sign (a * b) + 
  Real.sign (a * c) + Real.sign (b * c) + Real.sign (a * b * c) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_signs_l1211_121113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_divided_quadrilateral_l1211_121186

/-- A convex quadrilateral with two opposite sides divided into 100 equal parts -/
structure DividedQuadrilateral where
  /-- The list of 100 small quadrilaterals formed by connecting corresponding points -/
  small_quads : List Real
  /-- The number of divisions is 100 -/
  division_count : small_quads.length = 100
  /-- The area of the first small quadrilateral is 1 -/
  first_area : small_quads.head? = some 1
  /-- The area of the last small quadrilateral is 2 -/
  last_area : small_quads.getLast? = some 2
  /-- All areas are positive -/
  all_positive : ∀ a ∈ small_quads, a > 0

/-- The theorem stating that the area of the original quadrilateral is 150 -/
theorem area_of_divided_quadrilateral (q : DividedQuadrilateral) :
  (q.small_quads.sum) = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_divided_quadrilateral_l1211_121186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_point_not_monotone_decreasing_symmetry_about_line_max_value_max_value_attained_l1211_121139

open Real

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * sin (2 * x) - cos (2 * x) + 1

-- Statement 1: Symmetry about (π/12, 1)
theorem symmetry_about_point : ∀ (t : ℝ), f (π/12 + t) = f (π/12 - t) := by sorry

-- Statement 2: Not monotonically decreasing on (5π/12, 11π/12)
theorem not_monotone_decreasing : ¬ (∀ (x y : ℝ), 5*π/12 < x ∧ x < y ∧ y < 11*π/12 → f x > f y) := by sorry

-- Statement 3: Symmetry about x = π/3
theorem symmetry_about_line : ∀ (t : ℝ), f (π/3 + t) = f (π/3 - t) := by sorry

-- Statement 4: Maximum value is 3
theorem max_value : ∀ (x : ℝ), f x ≤ 3 := by sorry

-- Additional theorem to show that 3 is attained
theorem max_value_attained : ∃ (x : ℝ), f x = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_point_not_monotone_decreasing_symmetry_about_line_max_value_max_value_attained_l1211_121139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_equation_l1211_121112

/-- 
In a triangle ABC, if point M satisfies the given vector equations, 
then m = -3.
-/
theorem triangle_vector_equation (A B C M : EuclideanSpace ℝ (Fin 3)) 
  (m : ℝ) :
  (M - A) + (M - B) + (M - C) = 0 →
  (B - A) + (C - A) + m • (M - A) = 0 →
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_equation_l1211_121112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l1211_121154

theorem size_relationship : 
  let a := Real.log 0.7 / Real.log 2
  let b := (1/5: ℝ) ^ (2/3 : ℝ)
  let c := (1/2 : ℝ) ^ (-3 : ℝ)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l1211_121154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_tank_radius_l1211_121121

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

theorem stationary_tank_radius 
  (stationaryTank : Cylinder)
  (oilTruck : Cylinder)
  (h_stationary_height : stationaryTank.height = 25)
  (h_oil_drop : ℝ)
  (h_truck_radius : oilTruck.radius = 5)
  (h_truck_height : oilTruck.height = 12)
  (h_oil_drop_value : h_oil_drop = 0.03)
  (h_volume_equality : cylinderVolume oilTruck = Real.pi * stationaryTank.radius^2 * h_oil_drop) :
  stationaryTank.radius = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_tank_radius_l1211_121121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_one_and_neg_one_l1211_121161

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then (2 : ℝ)^x else x

theorem f_sum_one_and_neg_one : f 1 + f (-1) = 1 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the if-then-else expressions
  simp [if_pos (show 1 > 0 from by norm_num), if_neg (show ¬(-1 > 0) from by norm_num)]
  -- Evaluate 2^1 and perform the final arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_one_and_neg_one_l1211_121161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_radius_l1211_121156

/-- Two circles are tangent if the distance between their centers equals the sum of their radii -/
def circle_tangent_to_circle (r1 r2 : ℝ) : Prop :=
  ∃ (c1 c2 : ℝ × ℝ), (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

theorem inscribed_circles_radius (R : ℝ) (r : ℝ) : 
  R = 3 → 
  r > 0 →
  (∃ (n : ℕ), n = 6 ∧ 
    (∀ i : Fin n, 
      (circle_tangent_to_circle R r) ∧ 
      (circle_tangent_to_circle r r) ∧ 
      (circle_tangent_to_circle r r))) →
  r = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_radius_l1211_121156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_inscribable_iff_base_has_inscribed_circle_l1211_121123

/-- A right prism with a polygonal base -/
structure RightPrism where
  base : Set (ℝ × ℝ)
  height : ℝ
  is_polygonal : Prop  -- Changed from IsPolygon to Prop

/-- A cylinder -/
structure Cylinder where
  base : Set (ℝ × ℝ)
  height : ℝ
  is_circular : Prop  -- Changed from IsCircle to Prop

/-- Predicate to check if a circle can be inscribed in a polygon -/
def HasInscribedCircle (s : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate to check if a cylinder can be inscribed in a right prism -/
def CanInscribeCylinder (p : RightPrism) : Prop := sorry

/-- Theorem: A cylinder can be inscribed in a right prism if and only if 
    the base of the prism is a polygon into which a circle can be inscribed -/
theorem cylinder_inscribable_iff_base_has_inscribed_circle (p : RightPrism) : 
  CanInscribeCylinder p ↔ HasInscribedCircle p.base := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_inscribable_iff_base_has_inscribed_circle_l1211_121123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_plus_identity_l1211_121146

noncomputable def proj_vector : ℝ × ℝ := (4, 5)

noncomputable def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm_v := Real.sqrt (v.1^2 + v.2^2)
  let u := (v.1 / norm_v, v.2 / norm_v)
  ![![u.1 * u.1, u.1 * u.2], ![u.2 * u.1, u.2 * u.2]]

def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], ![0, 1]]

theorem projection_plus_identity :
  projection_matrix proj_vector + identity_matrix =
  ![![57/41, 20/41], ![20/41, 66/41]] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_plus_identity_l1211_121146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equivalence_l1211_121198

theorem probability_equivalence (event : Type) (P : event → ℝ) :
  (∃ e : event, P e = 1/2) → ∀ (experiment : event), P experiment = 50/100 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equivalence_l1211_121198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_iff_property_l1211_121137

/-- A discrete random variable with a geometric distribution -/
structure GeometricDistribution where
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- The probability mass function for a geometric distribution -/
def pmf (X : GeometricDistribution) (n : ℕ) : ℝ :=
  X.p * (1 - X.p)^(n - 1)

/-- The survival function for a geometric distribution -/
def survivorFunction (X : GeometricDistribution) (k : ℕ) : ℝ :=
  (1 - X.p)^k

/-- The conditional probability property for geometric distributions -/
def hasGeometricProperty (X : ℕ → ℝ) : Prop :=
  ∀ (m n : ℕ), X (n + m + 1) / X (n + 1) = X (m + 1)

/-- Theorem: A discrete random variable has a geometric distribution if and only if
    it satisfies the conditional probability property -/
theorem geometric_iff_property (X : ℕ → ℝ) :
  (∃ (g : GeometricDistribution), ∀ (n : ℕ), X n = pmf g n) ↔
  hasGeometricProperty X :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_iff_property_l1211_121137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l1211_121104

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + Real.log (Real.sqrt (x^2 - 2*x + 2) - x + 1)

-- State the theorem
theorem f_sum_equals_two : f (-12) + f 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l1211_121104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_f_2010_l1211_121117

def f (n : ℕ) : ℕ := 6^n

theorem divisors_of_f_2010 : 
  (Finset.filter (λ i => f 2010 % i = 0) (Finset.range (f 2010 + 1))).card = 4044121 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_f_2010_l1211_121117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_longer_lifespan_is_53_75_l1211_121122

/-- Represents the lifespan distribution in the tribe -/
structure TribeLifespan where
  prob_40 : ℝ
  prob_50 : ℝ
  prob_60 : ℝ
  sum_to_one : prob_40 + prob_50 + prob_60 = 1
  all_positive : prob_40 > 0 ∧ prob_50 > 0 ∧ prob_60 > 0

/-- Calculates the expected lifespan of the longer-living individual among two randomly selected individuals -/
def expectedLongerLifespan (t : TribeLifespan) : ℝ :=
  let p_both_40 := t.prob_40 * t.prob_40
  let p_longer_50 := t.prob_40 * t.prob_50 + t.prob_50 * t.prob_50 + t.prob_50 * t.prob_40
  let p_longer_60 := 1 - p_both_40 - p_longer_50
  p_both_40 * 40 + p_longer_50 * 50 + p_longer_60 * 60

/-- The specific tribe distribution given in the problem -/
def specificTribe : TribeLifespan where
  prob_40 := 0.25
  prob_50 := 0.50
  prob_60 := 0.25
  sum_to_one := by norm_num
  all_positive := by
    simp
    apply And.intro
    · norm_num
    apply And.intro
    · norm_num
    · norm_num

theorem expected_longer_lifespan_is_53_75 :
  expectedLongerLifespan specificTribe = 53.75 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_longer_lifespan_is_53_75_l1211_121122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_function_l1211_121158

theorem range_of_function (x : ℝ) : 
  -2 ≤ (Real.sin x - 1) / (Real.sin x + 2) ∧ (Real.sin x - 1) / (Real.sin x + 2) ≤ 0 :=
by
  sorry

#check range_of_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_function_l1211_121158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2y4_is_240_l1211_121181

/-- The coefficient of x²y⁴ in the expansion of (x-2y)⁶ is 240 -/
def coefficient_x2y4_in_expansion : ℤ :=
  let n : ℕ := 6
  let k : ℕ := 4
  let a : ℤ := 1
  let b : ℤ := -2
  (n.choose k) * (a^(n-k)) * (b^k)
  
#eval coefficient_x2y4_in_expansion -- This should evaluate to 240

theorem coefficient_x2y4_is_240 : coefficient_x2y4_in_expansion = 240 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2y4_is_240_l1211_121181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_op_example_l1211_121188

-- Define the @ operation
noncomputable def at_op (x y : ℝ) : ℝ := (2 * (x + y)) / 3

-- State the theorem
theorem at_op_example : at_op (at_op 4 7) 5 = 74 / 9 := by
  -- Unfold the definition of at_op
  unfold at_op
  -- Simplify the expression
  simp [mul_add, mul_div_assoc]
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_op_example_l1211_121188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l1211_121175

theorem angle_in_second_quadrant (θ : Real) 
  (h1 : Real.sin (Real.pi / 2 + θ) < 0) 
  (h2 : Real.tan (Real.pi - θ) > 0) : 
  Real.pi / 2 < θ ∧ θ < Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l1211_121175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_a3_a8_l1211_121130

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_positive : ∀ n, a n > 0
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

theorem max_product_a3_a8 (seq : ArithmeticSequence) 
  (h : sum_n seq 10 = 40) :
  (seq.a 3 * seq.a 8 ≤ 16) ∧ 
  (∃ seq : ArithmeticSequence, sum_n seq 10 = 40 ∧ seq.a 3 * seq.a 8 = 16) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_a3_a8_l1211_121130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_sum_proof_l1211_121124

/-- The line equation forming the triangle with coordinate axes -/
def line_equation (x y : ℝ) : Prop := 10 * x + 4 * y = 40

/-- The x-intercept of the line -/
def x_intercept : ℝ := 4

/-- The y-intercept of the line -/
def y_intercept : ℝ := 10

/-- The area of the triangle -/
def triangle_area : ℝ := 20

/-- The length of the hypotenuse of the triangle -/
noncomputable def hypotenuse_length : ℝ := 2 * Real.sqrt 29

/-- The sum of the lengths of the altitudes of the triangle -/
noncomputable def altitude_sum : ℝ := (406 + 20 * Real.sqrt 29) / 29

theorem altitude_sum_proof :
  line_equation x_intercept 0 ∧
  line_equation 0 y_intercept ∧
  triangle_area = (1/2) * x_intercept * y_intercept ∧
  hypotenuse_length^2 = x_intercept^2 + y_intercept^2 →
  altitude_sum = x_intercept + y_intercept + (2 * triangle_area / hypotenuse_length) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_sum_proof_l1211_121124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_approx_l1211_121189

-- Define the cost price, profit percentage, and list price
noncomputable def cost_price : ℝ := 51.50
noncomputable def profit_percentage : ℝ := 0.25
noncomputable def list_price : ℝ := 67.76

-- Define the selling price calculation
noncomputable def selling_price : ℝ := cost_price * (1 + profit_percentage)

-- Define the discount amount
noncomputable def discount_amount : ℝ := list_price - selling_price

-- Define the discount percentage calculation
noncomputable def discount_percentage : ℝ := (discount_amount / list_price) * 100

-- Theorem statement
theorem discount_percentage_approx :
  |discount_percentage - 4.995| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_approx_l1211_121189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_nina_digits_l1211_121133

theorem max_nina_digits (carlos : ℕ) (sam : ℕ) (mina : ℕ) (nina : ℕ) :
  sam = carlos + 6 →
  mina = 6 * carlos →
  mina = 24 →
  2 * nina ≤ 7 * carlos →
  carlos + sam + mina + nina ≤ 100 →
  ∃ (max_nina : ℕ), max_nina = 62 ∧ 
    ∀ n : ℕ, n > max_nina → carlos + sam + mina + n > 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_nina_digits_l1211_121133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_box_l1211_121182

-- Define the box dimensions
noncomputable def box_length : ℝ := 20
noncomputable def box_width : ℝ := 20
noncomputable def box_height : ℝ := 12

-- Define the total volume needed
noncomputable def total_volume : ℝ := 2400000

-- Define the minimum total amount spent
noncomputable def total_amount : ℝ := 200

-- Calculate the volume of one box
noncomputable def box_volume : ℝ := box_length * box_width * box_height

-- Calculate the number of boxes needed
noncomputable def num_boxes : ℝ := total_volume / box_volume

-- Define the theorem
theorem cost_per_box :
  total_amount / num_boxes = 0.40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_box_l1211_121182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1211_121197

noncomputable def f (x : ℝ) := Real.cos x ^ 2 - 1/2

theorem f_properties :
  (∀ x, f x = f (-x)) ∧ 
  (∀ t, t > 0 ∧ (∀ x, f (x + t) = f x) → t ≥ π) ∧
  (∀ x, f (x + π) = f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1211_121197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l1211_121150

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^5 + Real.tan x - 3

-- State the theorem
theorem f_symmetry (m : ℝ) (h : f (-m) = -2) : f m = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l1211_121150
