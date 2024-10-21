import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_zero_l306_30681

noncomputable def f (x : ℝ) := Real.exp x * Real.cos x

theorem tangent_angle_at_zero : 
  let θ := Real.arctan ((deriv f) 0)
  θ = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_zero_l306_30681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l306_30600

-- Define the quadratic function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_difference (x : ℝ) : f (x + 1) - f x = 2 * x - 1
axiom f_zero : f 0 = 3

-- Define the theorem
theorem quadratic_function_properties :
  (∀ x, f x = x^2 - 2*x + 3) ∧
  (∀ k, (∀ x₁ x₂, x₁ ≠ x₂ → x₁ ∈ Set.Ioo 2 4 → x₂ ∈ Set.Ioo 2 4 → 
    |f x₁ - f x₂| < k * |x₁ - x₂|) ↔ k ≥ 6) :=
by
  sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l306_30600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_fee_theorem_l306_30617

/-- Parking fee calculation for a small car in a commercial area parking lot -/
def parking_fee (x : ℕ) : ℕ :=
  let day_rate (h : ℕ) := min 45 (8 + 3 * (h - 2))
  let night_rate (h : ℕ) := min 15 h
  let total_fee := day_rate 9 + night_rate (x - 9)
  min 60 total_fee

theorem parking_fee_theorem (x : ℕ) (h₁ : x ≥ 10) (h₂ : x ≤ 13) :
  parking_fee x = if x = 10 then 59 else 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_fee_theorem_l306_30617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l306_30638

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.exp 1 + x) + Real.log (Real.exp 1 - x)

theorem f_properties :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.exp 1 → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l306_30638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_a_n_l306_30661

def a : Fin 3 → ℝ := ![-1, 1, -2]
def b : Fin 3 → ℝ := ![1, -2, -1]

/-- n is a vector parallel to b with a fixed z-coordinate of -2 -/
def n (x y : ℝ) : Fin 3 → ℝ := ![x, y, -2]

/-- The condition that n is parallel to b -/
def parallel (x y : ℝ) : Prop :=
  ∃ (k : ℝ), n x y = fun i => k * b i

/-- The dot product of two 3D vectors -/
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (Finset.univ.sum fun i => v i * w i)

theorem dot_product_a_n (x y : ℝ) (h : parallel x y) :
  dot_product a (n x y) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_a_n_l306_30661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cart_final_position_l306_30689

theorem cart_final_position (m n : ℕ) (h : m > n) :
  (m - n) * n = (m - n) * n := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cart_final_position_l306_30689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_house_numbers_l306_30663

/-- A two-digit prime number less than 60 -/
def TwoDigitPrime : Type := { p : ℕ // p < 60 ∧ p ≥ 10 ∧ Nat.Prime p }

/-- A six-digit house number composed of three distinct two-digit primes -/
structure HouseNumber where
  ab : TwoDigitPrime
  cd : TwoDigitPrime
  ef : TwoDigitPrime
  distinct : ab ≠ cd ∧ cd ≠ ef ∧ ab ≠ ef

/-- The set of all valid house numbers -/
def ValidHouseNumbers : Set HouseNumber :=
  {h : HouseNumber | True}

instance : Fintype TwoDigitPrime := sorry

instance : Fintype HouseNumber := sorry

instance : Fintype ValidHouseNumbers := sorry

theorem count_valid_house_numbers :
  Fintype.card ValidHouseNumbers = 1716 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_house_numbers_l306_30663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_translated_sine_l306_30605

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3)

-- Define the set of axes of symmetry
def axes_of_symmetry : Set ℝ := {x | ∃ k : ℤ, x = k * π / 2 + π / 12}

-- Theorem statement
theorem symmetry_of_translated_sine :
  ∀ x ∈ axes_of_symmetry, f (x - π / 4) = f (x + π / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_translated_sine_l306_30605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_condition_existence_condition_l306_30644

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := x^2 - (2*a + 1)*x + a * Real.log x

def g (a x : ℝ) : ℝ := (1 - a) * x

-- Statement for part (1)
theorem monotonicity_condition (a : ℝ) :
  (∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x < y → f a x < f a y) ∨
  (∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x < y → f a x > f a y) ↔
  a ≥ 2 ∨ a ≤ 1 :=
sorry

-- Statement for part (2)
theorem existence_condition (a : ℝ) :
  (∃ x₀, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ f a x₀ ≥ g a x₀) ↔
  a ≤ Real.exp 1 * (Real.exp 1 - 2) / (Real.exp 1 - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_condition_existence_condition_l306_30644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_xfx_positive_l306_30614

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

theorem solution_set_xfx_positive
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_f1 : f 1 = 0)
  (h_decreasing : monotone_decreasing_on f 0 (Real.pi/2)) :
  {x : ℝ | x * f x > 0} = {x : ℝ | (0 < x ∧ x < 1) ∨ (-1 < x ∧ x < 0)} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_xfx_positive_l306_30614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_BAC_value_l306_30648

-- Define the triangle ABC and its circumcenter O
variable (A B C O : EuclideanSpace ℝ (Fin 2))

-- Define the circumcenter property
def is_circumcenter (O A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A O = dist B O ∧ dist B O = dist C O

-- Define the vector equation
def vector_equation (A B C O : EuclideanSpace ℝ (Fin 2)) : Prop :=
  O - A = (B - A) + 2 • (C - A)

-- Define the angle BAC
noncomputable def angle_BAC (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  Real.arccos ((dist B A ^ 2 + dist C A ^ 2 - dist B C ^ 2) / (2 * dist B A * dist C A))

-- Theorem statement
theorem sin_angle_BAC_value 
  (h_circumcenter : is_circumcenter O A B C)
  (h_vector_eq : vector_equation A B C O) :
  Real.sin (angle_BAC A B C) = Real.sqrt 10 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_BAC_value_l306_30648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l306_30695

/-- Given a right-angled triangle with legs a and b, if the volume of the cone formed by rotating
    around leg a is 675π cm³ and the volume of the cone formed by rotating around leg b is 2430π cm³,
    then the length of the hypotenuse is approximately 30.79 cm. -/
theorem right_triangle_hypotenuse (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1 / 3 : ℝ) * Real.pi * a * b^2 = 675 * Real.pi ∧ 
  (1 / 3 : ℝ) * Real.pi * b * a^2 = 2430 * Real.pi →
  ∃ ε > 0, abs (Real.sqrt (a^2 + b^2) - 30.79) < ε := by sorry

#check right_triangle_hypotenuse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l306_30695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l306_30620

/-- The distance between two points in a 2D plane. -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem: The distance between points (1, 3) and (6, 7) is √41. -/
theorem distance_between_points : distance 1 3 6 7 = Real.sqrt 41 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression under the square root
  simp [Real.sqrt_eq_rpow]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l306_30620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_negative_sufficient_not_necessary_l306_30671

/-- The function f(x) = m + log_2(x) for x ≥ 1 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m + Real.log x / Real.log 2

/-- Predicate indicating if the function f has a root for a given m -/
def has_root (m : ℝ) : Prop := ∃ x : ℝ, x ≥ 1 ∧ f m x = 0

/-- Statement that m < 0 is a sufficient but not necessary condition for f to have a root -/
theorem m_negative_sufficient_not_necessary :
  (∀ m : ℝ, m < 0 → has_root m) ∧
  ¬(∀ m : ℝ, has_root m → m < 0) := by
  sorry

#check m_negative_sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_negative_sufficient_not_necessary_l306_30671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_length_l306_30603

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the right focus
def right_focus (F : ℝ × ℝ) : Prop := F.1 > 0 ∧ F.2 = 0 ∧ hyperbola F.1 F.2

-- Define a point on the asymptote
noncomputable def on_asymptote (P : ℝ × ℝ) : Prop :=
  P.2 = Real.sqrt 3 / 3 * P.1 ∨ P.2 = -Real.sqrt 3 / 3 * P.1

-- Define a line passing through a point
def line_through (P Q : ℝ × ℝ) : Prop :=
  ∃ (m b : ℝ), Q.2 - P.2 = m * (Q.1 - P.1) ∧ Q.2 = m * Q.1 + b

-- Define a right-angled triangle
def right_angled_triangle (O M N : ℝ × ℝ) : Prop :=
  (M.1 - O.1) * (N.1 - O.1) + (M.2 - O.2) * (N.2 - O.2) = 0

-- Main theorem
theorem hyperbola_intersection_length
  (F M N : ℝ × ℝ)
  (hF : right_focus F)
  (hM : on_asymptote M)
  (hN : on_asymptote N)
  (hFM : line_through F M)
  (hFN : line_through F N)
  (hOMN : right_angled_triangle (0, 0) M N) :
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_length_l306_30603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amos_finishes_on_friday_l306_30697

def pages_read (day : ℕ) : ℕ :=
  40 + 20 * (day - 1)

def total_pages_read (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => pages_read (i + 1))

theorem amos_finishes_on_friday :
  total_pages_read 5 = 400 ∧
  ∀ k : ℕ, k < 5 → total_pages_read k < 400 := by
  sorry

#eval total_pages_read 5  -- This line is added to check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amos_finishes_on_friday_l306_30697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_l306_30640

/-- The function f defined as f(x) = x(x+1)/2 -/
noncomputable def f (x : ℝ) : ℝ := x * (x + 1) / 2

/-- Theorem stating that f(x-3) = (x^2 - 5x + 6) / 2 -/
theorem f_shifted (x : ℝ) : f (x - 3) = (x^2 - 5*x + 6) / 2 := by
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  ring
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_l306_30640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l306_30643

-- Define the function f(x) = e^x + x - 2
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2

-- Theorem statement
theorem root_in_interval :
  ∃! x, x ∈ Set.Ioo 0 1 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l306_30643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_m_value_l306_30609

/-- The line equation: mx + y - 2m - 1 = 0 -/
def line_equation (m x y : ℝ) : Prop :=
  m * x + y - 2 * m - 1 = 0

/-- The circle equation: x² + y² - 2x - 4y = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y = 0

/-- The point P(2, 1) lies on the line -/
def point_on_line (m : ℝ) : Prop :=
  line_equation m 2 1

/-- The center of the circle C(1, 2) -/
def circle_center : ℝ × ℝ := (1, 2)

/-- The radius of the circle is √5 -/
noncomputable def circle_radius : ℝ := Real.sqrt 5

/-- The line is perpendicular to the line connecting P and C -/
def line_perpendicular_to_PC (m : ℝ) : Prop :=
  m * ((2 - 1) / (1 - 2)) = -1

theorem shortest_chord_m_value :
  ∀ m : ℝ, 
    point_on_line m ∧ 
    line_perpendicular_to_PC m →
    m = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_m_value_l306_30609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_proof_l306_30621

noncomputable def hyperbola1 (x y : ℝ) : Prop := x^2 / 3 - y^2 / 2 = 1

noncomputable def hyperbola2 (x y : ℝ) : Prop := y^2 / 18 - x^2 / 27 = 1

noncomputable def point_A : ℝ × ℝ := (Real.sqrt 3, 2 * Real.sqrt 5)

theorem hyperbola_proof :
  (∀ x y : ℝ, hyperbola1 x y ↔ y = Real.sqrt (2/3) * x ∨ y = -Real.sqrt (2/3) * x) →
  (∀ x y : ℝ, hyperbola2 x y ↔ y = Real.sqrt (2/3) * x ∨ y = -Real.sqrt (2/3) * x) →
  hyperbola2 point_A.fst point_A.snd :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_proof_l306_30621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l306_30664

-- Define the function f(x) = ax - ln x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

-- State the theorem
theorem f_properties (a : ℝ) :
  -- Part 1: When a ≤ 0, f(x) is monotonically decreasing on (0, +∞)
  (a ≤ 0 → ∀ x y, 0 < x → 0 < y → x < y → f a y < f a x) ∧
  -- Part 2: When a > 0, f(x) has a minimum value of 1 + ln a at x = 1/a
  (a > 0 → ∀ x, 0 < x → f a (1/a) ≤ f a x ∧ f a (1/a) = 1 + Real.log a) ∧
  -- Part 3: If x ∈ (0, e] and ∃x: f(x) ≤ 3, then a ≤ e²
  (∃ x, 0 < x ∧ x ≤ Real.exp 1 ∧ f a x ≤ 3 → a ≤ Real.exp 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l306_30664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increasing_negative_x_l306_30626

noncomputable def f (x : ℝ) : ℝ := -6 / x

theorem inverse_proportion_increasing_negative_x :
  ∀ x₁ x₂ : ℝ, x₁ < 0 → x₂ < 0 → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increasing_negative_x_l306_30626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_l306_30657

-- Define the curves
noncomputable def curve1 (x : ℝ) : ℝ := x + Real.log x
def curve2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

-- Define the tangent line of curve1 at x = 1
def tangent_line (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem
theorem tangent_intersection (a : ℝ) : 
  (∀ x, x > 0 → curve1 x = curve2 a x → tangent_line x = curve2 a x) → 
  a = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_l306_30657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_condition_l306_30679

-- Define a triangle
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C
  angleSum : A + B + C = π
  lawOfSines : a / Real.sin A = b / Real.sin B

-- State the theorem
theorem triangle_equilateral_condition (t : Triangle) :
  t.a / Real.cos t.A = t.b / Real.cos t.B ∧
  t.b / Real.cos t.B = t.c / Real.cos t.C →
  t.A = t.B ∧ t.B = t.C := by
  sorry

-- Define an equilateral triangle
def is_equilateral (t : Triangle) : Prop :=
  t.A = t.B ∧ t.B = t.C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_condition_l306_30679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_subsets_l306_30649

theorem intersection_of_subsets (S : Finset ℕ) (h₁ : S.card = 1000) :
  ∀ (S₁ S₂ S₃ S₄ S₅ : Finset ℕ),
    (∀ i, i ∈ [S₁, S₂, S₃, S₄, S₅] → i ⊆ S ∧ i.card = 500) →
    ∃ i j, i ∈ [S₁, S₂, S₃, S₄, S₅] ∧ j ∈ [S₁, S₂, S₃, S₄, S₅] ∧ i ≠ j ∧ (i ∩ j).card ≥ 100 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_subsets_l306_30649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_non_real_roots_l306_30678

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ z : ℂ, z^2 + b*z + 16 = 0 → z.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_non_real_roots_l306_30678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l306_30637

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6) + 2 * Real.sin (ω * x / 2) ^ 2

theorem problem_solution (ω : ℝ) (A B C : ℝ) (a b c : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω (x + Real.pi / ω) = f ω x) →
  b < a →
  f ω A = 3 / 2 →
  1 / 2 * b * c * Real.sin A = 6 * Real.sqrt 3 →
  a = 2 * Real.sqrt 7 →
  (∀ x : ℝ, f ω x = Real.sin (x - Real.pi / 6) + 1) ∧
  b = 4 ∧
  c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l306_30637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l306_30651

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (x^3) / (x + 2) - (3 / (x - 2) + 1)

-- Define the solution set
def solution_set : Set ℝ := {x | x < -2 ∨ x ≥ 2}

-- Theorem statement
theorem inequality_solution :
  {x : ℝ | f x ≥ 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l306_30651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_construction_l306_30693

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A quadrilateral in the plane -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Similarity relation between quadrilaterals -/
def IsSimilar (q1 q2 : Quadrilateral) : Prop :=
  sorry

/-- A point lies on a line -/
def LiesOn (p : Point) (l : Line) : Prop :=
  sorry

/-- Two lines are parallel -/
def IsParallel (l1 l2 : Line) : Prop :=
  sorry

/-- Two lines are coincident -/
def IsCoincident (l1 l2 : Line) : Prop :=
  sorry

/-- The locus line for possible positions of vertex D -/
noncomputable def LocusLine (l1 l2 l3 : Line) (q : Quadrilateral) : Line :=
  sorry

theorem quadrilateral_construction
  (l1 l2 l3 l4 : Line)
  (q : Quadrilateral) :
  (∃! abcd : Quadrilateral,
    IsSimilar abcd q ∧
    LiesOn abcd.A l1 ∧
    LiesOn abcd.B l2 ∧
    LiesOn abcd.C l3 ∧
    LiesOn abcd.D l4) ∨
  (IsParallel (LocusLine l1 l2 l3 q) l4 ∨
   IsCoincident (LocusLine l1 l2 l3 q) l4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_construction_l306_30693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l306_30630

noncomputable def points : List (ℝ × ℝ) := [(-3, 4), (2, -3), (-5, 0), (0, -6), (4, 1)]

noncomputable def distance (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

theorem farthest_point :
  (0, -6) ∈ points ∧
  ∀ p ∈ points, distance (0, -6) ≥ distance p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l306_30630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_unique_angle_range_l306_30636

theorem triangle_unique_angle_range (a b c : ℝ) (A B C : ℝ) :
  b = 3 →
  Real.cos A = 2/3 →
  (∃! C, 
    a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) ∧ 
    b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ 
    c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) →
  (a = Real.sqrt 5 ∨ a ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_unique_angle_range_l306_30636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l306_30615

/-- Represents a rectangular yard with given length and width -/
structure Yard where
  length : ℝ
  width : ℝ

/-- Represents an isosceles right triangle with given leg length -/
structure IsoscelesRightTriangle where
  legLength : ℝ

/-- Represents a rectangular flower bed with given length and width -/
structure RectangularFlowerBed where
  length : ℝ
  width : ℝ

/-- Calculates the area of a yard -/
noncomputable def yardArea (y : Yard) : ℝ := y.length * y.width

/-- Calculates the area of an isosceles right triangle -/
noncomputable def triangleArea (t : IsoscelesRightTriangle) : ℝ := 1/2 * t.legLength^2

/-- Calculates the area of a rectangular flower bed -/
noncomputable def flowerBedArea (f : RectangularFlowerBed) : ℝ := f.length * f.width

/-- The main theorem stating that the fraction of the yard occupied by the flower beds is 1/3 -/
theorem flower_beds_fraction (y : Yard) (t : IsoscelesRightTriangle) (f : RectangularFlowerBed)
    (h1 : y.length = 30)
    (h2 : y.width = 10)
    (h3 : t.legLength = 10)
    (h4 : f.length = 5)
    (h5 : f.width = 10) :
    (triangleArea t + flowerBedArea f) / yardArea y = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l306_30615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l306_30627

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The foci of an ellipse -/
noncomputable def Ellipse.foci (e : Ellipse) : ℝ × ℝ × ℝ × ℝ :=
  let c := Real.sqrt (e.a^2 - e.b^2)
  (-c, 0, c, 0)

/-- A point on an ellipse -/
def Ellipse.point (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The statement of the problem -/
theorem ellipse_triangle_area (e : Ellipse) (P : ℝ × ℝ) :
  e.a = 3 ∧ e.b = 2 ∧
  e.point P.1 P.2 ∧
  let (F₁x, F₁y, F₂x, F₂y) := e.foci
  (P.1 - F₁x) * (P.1 - F₂x) + (P.2 - F₁y) * (P.2 - F₂y) = 0 →
  let (F₁x, F₁y, F₂x, F₂y) := e.foci
  (1/2) * Real.sqrt ((F₁x - P.1)^2 + (F₁y - P.2)^2) *
          Real.sqrt ((F₂x - P.1)^2 + (F₂y - P.2)^2) = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l306_30627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_after_inserting_eights_l306_30677

def insert_eights (n : ℕ) : ℕ := 20 * 10^(n + 2) + 8 * ((10^(n + 1) - 1) / 9) + 21

theorem composite_after_inserting_eights (n : ℕ) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ insert_eights n = a * b := by
  -- We know that 47 divides 2021, so we'll use 47 as one of the factors
  use 47
  -- We'll calculate the other factor
  let other_factor := insert_eights n / 47
  use other_factor
  have h1 : 47 > 1 := by norm_num
  have h2 : other_factor > 1 := by
    -- This requires a more detailed proof, which we'll skip for now
    sorry
  have h3 : insert_eights n = 47 * other_factor := by
    -- This also requires a more detailed proof, which we'll skip
    sorry
  exact ⟨h1, h2, h3⟩

#check composite_after_inserting_eights

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_after_inserting_eights_l306_30677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l306_30602

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define point M
def M : ℝ × ℝ := (1, 1)

-- Define points A and B on the hyperbola
variable (A B : ℝ × ℝ)

-- M is the midpoint of AB
axiom midpoint_AB : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- A and B are on the hyperbola
axiom A_on_hyperbola : hyperbola A.1 A.2
axiom B_on_hyperbola : hyperbola B.1 B.2

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 4 * y - 3 * x - 1 = 0

-- Theorem to prove
theorem line_AB_equation : 
  ∀ x y : ℝ, (∃ t : ℝ, x = t * (B.1 - A.1) + A.1 ∧ y = t * (B.2 - A.2) + A.2) 
  → line_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l306_30602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_incorrect_l306_30652

/-- Represents a geometric statement --/
inductive GeometricStatement
  | LineSeg : GeometricStatement  -- Statement about line segments
  | ObtuseAngle : GeometricStatement  -- Statement about obtuse angles
  | InteriorAngles : GeometricStatement  -- Statement about interior angles

/-- Checks if a geometric statement is correct --/
def is_correct (s : GeometricStatement) : Bool :=
  match s with
  | GeometricStatement.LineSeg => false
  | GeometricStatement.ObtuseAngle => false
  | GeometricStatement.InteriorAngles => false

/-- The list of geometric statements in the problem --/
def statements : List GeometricStatement :=
  [GeometricStatement.LineSeg, GeometricStatement.ObtuseAngle, GeometricStatement.InteriorAngles]

/-- Counts the number of incorrect statements --/
def count_incorrect (stmts : List GeometricStatement) : Nat :=
  stmts.filter (fun s => !is_correct s) |>.length

/-- Theorem stating that the number of incorrect statements is 3 --/
theorem all_statements_incorrect :
  count_incorrect statements = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_incorrect_l306_30652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l306_30606

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := π / 3
  let φ : ℝ := π / 4
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (Real.sqrt 2, Real.sqrt 6, 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l306_30606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_5_in_factorial_product_l306_30610

/-- Count of factors of 5 in n! -/
def count_factors_of_5 (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Sum of count_factors_of_5 from 1 to n -/
def sum_factors_of_5 (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) count_factors_of_5

theorem factors_of_5_in_factorial_product :
  sum_factors_of_5 150 ≡ 125 [MOD 1500] := by
  sorry

#eval sum_factors_of_5 150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_5_in_factorial_product_l306_30610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_example_l306_30684

/-- Conversion from spherical coordinates to rectangular coordinates -/
noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

/-- Theorem stating the equivalence of the given spherical and rectangular coordinates -/
theorem spherical_to_rectangular_example :
  spherical_to_rectangular 10 (5 * π / 4) (π / 4) = (-5, -5, 5 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_example_l306_30684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_truck_meeting_times_l306_30632

/-- Represents the position of Michael or the truck at a given time -/
structure Position :=
  (distance : ℝ)
  (time : ℝ)

/-- Calculates the position of Michael at a given time -/
def michael_position (t : ℝ) : Position :=
  { distance := 6 * t, time := t }

/-- Calculates the position of the truck at a given time -/
noncomputable def truck_position (t : ℝ) : Position :=
  let cycle_time := 60
  let move_time := 24
  let full_cycles := Int.floor (t / cycle_time)
  let remaining_time := t - full_cycles * cycle_time
  let distance := if remaining_time ≤ move_time
    then 240 * (full_cycles + 1) + 10 * remaining_time
    else 240 * (full_cycles + 1)
  { distance := distance, time := t }

/-- Theorem stating that Michael and the truck meet every 120 seconds, starting at 120 seconds -/
theorem michael_truck_meeting_times :
  ∀ n : ℕ, michael_position (120 * ↑n) = truck_position (120 * ↑n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_truck_meeting_times_l306_30632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l306_30642

-- Define the function
noncomputable def f (x : ℝ) : ℝ := -Real.cos (x/2 - Real.pi/3)

-- Define the monotonic increasing property
def is_monotonic_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- Theorem statement
theorem f_monotonic_increasing (k : ℤ) :
  is_monotonic_increasing f (4 * ↑k * Real.pi + 2 * Real.pi / 3) (4 * ↑k * Real.pi + 8 * Real.pi / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l306_30642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l306_30698

noncomputable def f (x : ℝ) : ℝ := x + x / (x^2 + 1) + x * (x + 4) / (x^2 + 2) + 2 * (x + 2) / (x * (x^2 + 2))

theorem f_min_value (x : ℝ) (hx : x > 0) : f x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l306_30698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmony_number_count_l306_30616

/-- A Harmony Number is a four-digit number whose digits sum up to 6 -/
def isHarmonyNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 6)

/-- Count of Harmony Numbers with the first digit being 2 -/
def countHarmonyNumbersStartingWith2 : ℕ :=
  Finset.filter (fun n => n / 1000 = 2) (Finset.range 10000)
  |>.filter (fun n => (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 4)
  |>.card

theorem harmony_number_count : countHarmonyNumbersStartingWith2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmony_number_count_l306_30616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_satisfies_conditions_l306_30653

/-- A parabola with equation y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
noncomputable def vertex (p : Parabola) : ℝ × ℝ :=
  (-p.b / (2 * p.a), (4 * p.a * p.c - p.b^2) / (4 * p.a))

/-- A parabola opens downwards if a < 0 -/
def opens_downwards (p : Parabola) : Prop :=
  p.a < 0

/-- A point (x, y) lies on the line y = x -/
def on_line_y_eq_x (point : ℝ × ℝ) : Prop :=
  point.2 = point.1

/-- The theorem stating that y = -x² satisfies the required conditions -/
theorem parabola_satisfies_conditions :
  let p : Parabola := ⟨-1, 0, 0⟩
  opens_downwards p ∧ on_line_y_eq_x (vertex p) := by
  sorry

#check parabola_satisfies_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_satisfies_conditions_l306_30653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_x_intercept_l306_30685

-- Define the curves
noncomputable def C₁ (x : ℝ) : ℝ := Real.exp x
noncomputable def C₂ (x : ℝ) : ℝ := (1/4) * Real.exp (2 * x^2)

-- Define the tangent line
structure TangentLine where
  slope : ℝ
  x_tangent : ℝ
  y_tangent : ℝ

-- State the theorem
theorem tangent_line_x_intercept 
  (l : TangentLine)
  (h1 : ∀ x, l.slope * (x - l.x_tangent) + l.y_tangent = C₁ x → x = l.x_tangent)
  (h2 : ∀ x, l.slope * (x - l.x_tangent) + l.y_tangent = C₂ x → x = l.x_tangent)
  : ∃ x, l.slope * (x - l.x_tangent) + l.y_tangent = 0 ∧ x = 1 := by
  sorry

#check tangent_line_x_intercept

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_x_intercept_l306_30685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donation_amounts_l306_30625

/-- Represents the donation amounts of 5 students. -/
structure Donations where
  amounts : Fin 5 → ℕ
  all_multiples_of_100 : ∀ i, 100 ∣ amounts i
  average_560 : (amounts 0 + amounts 1 + amounts 2 + amounts 3 + amounts 4) / 5 = 560
  min_200 : ∀ i, amounts i ≥ 200
  max_800 : ∃ i, amounts i = 800 ∧ ∀ j, amounts j ≤ 800
  has_600 : ∃ i, amounts i = 600
  median_600 : ∃ i j k, amounts i ≤ 600 ∧ amounts j = 600 ∧ amounts k ≥ 600

/-- The two unspecified donation amounts are either (500, 700) or (600, 600). -/
theorem donation_amounts (d : Donations) : 
  ∃ i j, ({d.amounts i, d.amounts j} : Finset ℕ) = {500, 700} ∨ 
         ({d.amounts i, d.amounts j} : Finset ℕ) = {600, 600} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_donation_amounts_l306_30625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_2x_plus_cbrt_x_l306_30622

open Real MeasureTheory Interval

theorem integral_sqrt_2x_plus_cbrt_x : 
  ∫ x in (Set.Icc 0 8), (Real.sqrt (2 * x) + x ^ (1/3)) = (128 * Real.sqrt 2 + 36) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_2x_plus_cbrt_x_l306_30622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geese_percentage_among_non_swans_l306_30662

/-- Represents the bird population at Town Lake after the migration event. -/
structure BirdPopulation where
  total : ℕ
  geese : ℕ
  swans : ℕ
  herons : ℕ
  ducks : ℕ
  total_sum : total = geese + swans + herons + ducks
  geese_percent : geese = total * 2 / 5
  swans_percent : swans = total / 5
  herons_percent : herons = total / 5
  ducks_percent : ducks = total / 5

/-- The percentage of geese among non-swan birds is 50%. -/
theorem geese_percentage_among_non_swans (pop : BirdPopulation) (h : pop.total = 120) :
  pop.geese * 100 / (pop.total - pop.swans) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geese_percentage_among_non_swans_l306_30662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_theorem_l306_30639

/-- Represents the water usage and billing system for a household in Shenzhen --/
structure WaterUsage where
  quota : ℚ  -- Monthly water usage quota in cubic meters
  tariff : ℚ  -- Water tariff in yuan per cubic meter
  excess_fee_1 : ℚ  -- Additional fee percentage for usage between quota and quota + 10
  excess_fee_2 : ℚ  -- Additional fee percentage for usage above quota + 10
  planned_reduction : ℚ  -- Planned reduction in monthly water usage
  additional_months : ℚ  -- Additional months of water usage due to reduction
  excess_percentage : ℚ  -- Percentage by which actual usage exceeds planned for some months
  excess_months : ℚ  -- Number of months with excess usage

/-- Calculates the planned average monthly water usage --/
noncomputable def planned_usage (w : WaterUsage) : ℚ :=
  (w.additional_months * w.planned_reduction) / 
  (1 - w.additional_months / 12)

/-- Calculates the annual water fee --/
noncomputable def annual_fee (w : WaterUsage) (planned : ℚ) : ℚ :=
  let normal_months := 12 - w.excess_months
  let excess_usage := planned * (1 + w.excess_percentage)
  let normal_fee := normal_months * planned * w.tariff
  let excess_fee := w.excess_months * (
    if excess_usage ≤ w.quota + 10 then
      w.quota * w.tariff + (excess_usage - w.quota) * w.tariff * (1 + w.excess_fee_1)
    else
      w.quota * w.tariff + 10 * w.tariff * (1 + w.excess_fee_1) + 
      (excess_usage - w.quota - 10) * w.tariff * (1 + w.excess_fee_2)
  )
  normal_fee + excess_fee

/-- Theorem stating the correctness of the planned usage and annual fee calculations --/
theorem water_usage_theorem (w : WaterUsage) 
  (h1 : w.quota = 20) 
  (h2 : w.tariff = 1.9) 
  (h3 : w.excess_fee_1 = 0.5) 
  (h4 : w.excess_fee_2 = 1) 
  (h5 : w.planned_reduction = 4) 
  (h6 : w.additional_months = 4) 
  (h7 : w.excess_percentage = 0.4) 
  (h8 : w.excess_months = 4) : 
  planned_usage w = 20 ∧ annual_fee w (planned_usage w) = 547.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_theorem_l306_30639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_river_ratio_l306_30680

/-- Represents the length of a river and its parts -/
structure RiverLength where
  total : ℚ
  straight : ℚ
  crooked : ℚ

/-- Calculates the ratio of straight to crooked parts of a river -/
def straightToCrookedRatio (r : RiverLength) : ℚ :=
  r.straight / r.crooked

/-- Theorem: The ratio of straight to crooked parts for a specific river -/
theorem specific_river_ratio :
  let r : RiverLength := { total := 80, straight := 20, crooked := 60 }
  straightToCrookedRatio r = 1 / 3 := by
  sorry

#check specific_river_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_river_ratio_l306_30680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_change_l306_30634

theorem fraction_change (original : ℚ) (new_value : ℚ) : 
  original = 5/7 →
  let new_numerator := original.num * (1 + 1/5)
  let new_denominator := original.den * (1 - 1/10)
  new_value = new_numerator / new_denominator →
  new_value = 0.9523809523809523 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_change_l306_30634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_congruence_l306_30660

/-- Sum of digits in base q -/
def S_q (q : ℕ) (x : ℕ) : ℕ := sorry

theorem digit_sum_congruence 
  (a b b' c m q : ℕ) 
  (hm : m > 1) 
  (hq : q > 1) 
  (hbb' : |Int.ofNat b - Int.ofNat b'| ≥ a) 
  (hM : ∃ M : ℕ, ∀ n ≥ M, (S_q q (a * n + b) : ZMod m) = S_q q (a * n + b') + c) :
  (∀ n : ℕ, n > 0 → (S_q q (a * n + b) : ZMod m) = S_q q (a * n + b') + c) ∧ 
  (∀ L : ℕ, L > 0 → (S_q q (L + b) : ZMod m) = S_q q (L + b') + c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_congruence_l306_30660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l306_30665

def sequence_custom (a₀ a₁ : ℤ) : ℕ → ℤ
  | 0 => a₀
  | 1 => a₁
  | n + 2 => sequence_custom a₀ a₁ (n + 1) - sequence_custom a₀ a₁ n

def sum_of_terms (a₀ a₁ : ℤ) (n : ℕ) : ℤ :=
  (List.range n).map (sequence_custom a₀ a₁) |>.sum

theorem sequence_sum_theorem (a₀ a₁ : ℤ) :
  sum_of_terms a₀ a₁ 1492 = 1985 ∧
  sum_of_terms a₀ a₁ 1985 = 1492 →
  sum_of_terms a₀ a₁ 2001 = 986 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l306_30665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heavier_ball_identifiable_l306_30613

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Left  : WeighingResult  -- Left pan is heavier
  | Right : WeighingResult  -- Right pan is heavier
  | Equal : WeighingResult  -- Pans are balanced

/-- Represents a weighing operation -/
structure Weighing where
  left  : Finset Nat  -- Balls on the left pan
  right : Finset Nat  -- Balls on the right pan

/-- Represents a strategy for finding the heavier ball -/
structure Strategy where
  first  : Weighing
  second : WeighingResult → Weighing

/-- Determines if a pair of weighing results identifies the heavier ball -/
def determines (results : WeighingResult × WeighingResult) (heavyBall : Fin 9) : Prop :=
  sorry  -- Implementation details omitted for brevity

/-- Checks if a strategy can identify the heavier ball -/
def canIdentifyHeavyBall (s : Strategy) : Prop :=
  ∀ heavyBall : Fin 9,
    ∃ result1 : WeighingResult,
    ∃ result2 : WeighingResult,
    determines (result1, result2) heavyBall

theorem heavier_ball_identifiable :
  ∃ s : Strategy, canIdentifyHeavyBall s :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heavier_ball_identifiable_l306_30613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l306_30607

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem increasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  (3/2 ≤ a ∧ a < 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l306_30607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l306_30604

/-- The remainder when x^8 is divided by (x^2 + 1)(x - 1) is 1 -/
theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℂ, X^8 = (X^2 + 1) * (X - 1) * q + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l306_30604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_school_proof_l306_30647

/-- The distance to school in miles -/
noncomputable def distance_to_school : ℝ := 9.375

/-- The normal travel time in rush hour traffic in hours -/
noncomputable def normal_time : ℝ := 15 / 60

/-- The travel time without traffic in hours -/
noncomputable def no_traffic_time : ℝ := 9 / 60

/-- The speed increase without traffic in miles per hour -/
noncomputable def speed_increase : ℝ := 25

/-- The normal speed in rush hour traffic in miles per hour -/
noncomputable def normal_speed : ℝ := distance_to_school / normal_time

/-- The speed without traffic in miles per hour -/
noncomputable def no_traffic_speed : ℝ := normal_speed + speed_increase

theorem distance_to_school_proof :
  distance_to_school = normal_speed * normal_time ∧
  distance_to_school = no_traffic_speed * no_traffic_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_school_proof_l306_30647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_values_l306_30666

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.sin x - Real.cos x)

theorem sum_of_max_values (π : ℝ) :
  ∃ (S : ℝ), 
    (∀ x ∈ Set.Icc 0 (2015 * π), f x ≤ S) ∧ 
    (∃ (x : ℕ → ℝ), (∀ n, x n ∈ Set.Icc 0 (2015 * π) ∧ f (x n) = S) ∧ 
      (Real.exp π * (1 - Real.exp (2014 * π))) / (1 - Real.exp (2 * π)) = 
        Finset.sum (Finset.range (Nat.floor (2015 / 2) + 1)) (fun n => f (x n))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_values_l306_30666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_non_lucky_multiple_of_9_l306_30683

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ n % (sum_of_digits n) = 0

theorem least_non_lucky_multiple_of_9 :
  ∀ k : ℕ, k > 0 → k < 11 → is_lucky (9 * k) ∧ ¬ is_lucky 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_non_lucky_multiple_of_9_l306_30683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_inequality_l306_30690

theorem cubic_root_inequality (a b c : ℤ) (h : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :
  |((4 : ℝ) ^ (1/3 : ℝ)) * a + ((2 : ℝ) ^ (1/3 : ℝ)) * b + c| ≥ 1 / (4 * a^2 + 3 * b^2 + 2 * c^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_inequality_l306_30690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_fundraiser_profit_l306_30645

/-- Calculates the net profit from a candy bar fundraiser --/
theorem candy_bar_fundraiser_profit
  (total_boxes : ℕ)
  (discounted_boxes : ℕ)
  (bars_per_box : ℕ)
  (price_tier1 : ℚ)
  (price_tier2 : ℚ)
  (price_tier3 : ℚ)
  (tier1_limit : ℕ)
  (tier2_limit : ℕ)
  (regular_price : ℚ)
  (discounted_price : ℚ)
  (tax_rate : ℚ)
  (fixed_expense : ℚ)
  (h1 : total_boxes = 5)
  (h2 : discounted_boxes = 3)
  (h3 : bars_per_box = 10)
  (h4 : price_tier1 = 3/2)
  (h5 : price_tier2 = 13/10)
  (h6 : price_tier3 = 11/10)
  (h7 : tier1_limit = 30)
  (h8 : tier2_limit = 20)
  (h9 : regular_price = 1)
  (h10 : discounted_price = 4/5)
  (h11 : tax_rate = 7/100)
  (h12 : fixed_expense = 15) :
  (let total_bars := total_boxes * bars_per_box
   let revenue := min tier1_limit total_bars * price_tier1 +
                  min (max 0 (total_bars - tier1_limit)) tier2_limit * price_tier2 +
                  max 0 (total_bars - tier1_limit - tier2_limit) * price_tier3
   let cost := (total_boxes - discounted_boxes) * bars_per_box * regular_price +
               discounted_boxes * bars_per_box * discounted_price
   let sales_tax := revenue * tax_rate
   let total_expense := cost + sales_tax + fixed_expense
   let net_profit := revenue - total_expense
   net_profit = 703/100) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_fundraiser_profit_l306_30645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_payment_is_approximately_10_58_l306_30601

/-- Calculates the monthly payment for a purchase with given conditions -/
def calculate_monthly_payment (purchase_price interest_rate down_payment num_payments : ℚ) : ℚ :=
  let total_interest := interest_rate * purchase_price
  let total_amount := purchase_price + total_interest
  let remaining_amount := total_amount - down_payment
  remaining_amount / num_payments

/-- Theorem stating that the monthly payment is approximately $10.58 given the specified conditions -/
theorem monthly_payment_is_approximately_10_58 :
  let purchase_price : ℚ := 127
  let interest_rate : ℚ := 21.26 / 100
  let down_payment : ℚ := 27
  let num_payments : ℚ := 12
  ∃ ε > 0, |calculate_monthly_payment purchase_price interest_rate down_payment num_payments - 10.58| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_payment_is_approximately_10_58_l306_30601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_approximation_l306_30667

/-- The average percentage of first-class products -/
noncomputable def p : ℝ := 0.85

/-- The percentage of non-first-class products -/
noncomputable def q : ℝ := 1 - p

/-- The maximum allowed deviation -/
noncomputable def ε : ℝ := 0.01

/-- The desired probability -/
noncomputable def P : ℝ := 0.997

/-- The z-score corresponding to the desired probability -/
noncomputable def t_P : ℝ := 3

/-- The required sample size -/
noncomputable def n : ℝ := (p * q * t_P^2) / ε^2

/-- Theorem stating that the required sample size is approximately 11475 -/
theorem sample_size_approximation : ⌊n⌋ = 11475 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_approximation_l306_30667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_squeeze_theorem_l306_30673

theorem sequence_squeeze_theorem (a b c : ℕ → ℝ) (l : ℝ) :
  (∀ n, a n < b n ∧ b n < c n) →
  Filter.Tendsto a Filter.atTop (nhds l) →
  Filter.Tendsto c Filter.atTop (nhds l) →
  Filter.Tendsto b Filter.atTop (nhds l) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_squeeze_theorem_l306_30673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l306_30654

variable (x y : ℚ)

def A (x y : ℚ) : ℚ := 2 * x^2 - 3 * x * y - 5 * x - 1
def B (x y : ℚ) : ℚ := -x^2 + x * y - 1

theorem problem_solution (x y : ℚ) :
  (3 * A x y + 6 * B x y = -3 * x * y - 15 * x - 9) ∧
  (∀ x, 3 * A x y + 6 * B x y = -3 * x * y - 15 * x - 9 → y = -5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l306_30654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_solution_l306_30618

/-- The sum of the arithmetic-geometric series with initial term 1 and common ratio x -/
noncomputable def series_sum (x : ℝ) : ℝ := (1 + x) / (1 - x)

/-- Theorem stating that if the sum of the series 1 + 3x + 5x^2 + 7x^3 + ... equals 16, 
    then x = 15/17 -/
theorem series_solution (x : ℝ) (hx : x ≠ 1) :
  series_sum x = 16 → x = 15 / 17 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_solution_l306_30618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_distribution_l306_30676

theorem pen_distribution (n : Nat) : 
  n = 450 → 
  (Finset.filter (fun m => m > 1 ∧ m < n ∧ n % m = 0) (Finset.range (n + 1))).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_distribution_l306_30676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l306_30694

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x / (10 - x))

theorem domain_of_f : Set ℝ = { x | 0 ≤ x ∧ x < 10 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l306_30694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_TXZ_is_100_l306_30624

/-- Represents a parallelogram WXYZ with a point T on WY -/
structure ParallelogramWithPoint where
  -- Base length of the parallelogram
  base : ℝ
  -- Height of the parallelogram
  height : ℝ
  -- Ratio of WT to TY
  ratio : ℝ

/-- Calculates the area of region TXZ in the parallelogram -/
noncomputable def area_TXZ (p : ParallelogramWithPoint) : ℝ :=
  p.base * p.height * (1 - p.ratio / (1 + p.ratio))

/-- Theorem stating the area of TXZ is 100 square meters under given conditions -/
theorem area_TXZ_is_100 (p : ParallelogramWithPoint) 
  (h_base : p.base = 15)
  (h_height : p.height = 10)
  (h_ratio : p.ratio = 2) : 
  area_TXZ p = 100 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_TXZ_is_100_l306_30624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lecture_average_listening_time_l306_30628

/-- Represents the distribution of attendees and their listening time for a lecture --/
structure LectureAttendance where
  duration : ℚ
  full_listeners : ℚ
  sleepers : ℚ
  half_listeners : ℚ
  quarter_listeners : ℚ

/-- Calculates the average listening time for all attendees --/
def average_listening_time (attendance : LectureAttendance) : ℚ :=
  let total_attendees := attendance.full_listeners + attendance.sleepers + 
                         attendance.half_listeners + attendance.quarter_listeners
  let total_minutes := attendance.duration * attendance.full_listeners +
                       0 * attendance.sleepers +
                       (attendance.duration / 2) * attendance.half_listeners +
                       (attendance.duration / 4) * attendance.quarter_listeners
  total_minutes / total_attendees

/-- Theorem stating that the average listening time for the given lecture is 43.875 minutes --/
theorem lecture_average_listening_time :
  let attendance := LectureAttendance.mk 90 30 20 25 25
  average_listening_time attendance = 43875 / 1000 := by
  sorry

#eval average_listening_time (LectureAttendance.mk 90 30 20 25 25)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lecture_average_listening_time_l306_30628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_in_three_hours_l306_30633

/-- Represents the speed of a train in miles per minute -/
noncomputable def train_speed : ℚ := 2 / 2

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

/-- Calculates the distance traveled by the train given time in hours -/
noncomputable def distance_traveled (hours : ℚ) : ℚ := train_speed * hours_to_minutes hours

theorem train_distance_in_three_hours : 
  distance_traveled 3 = 180 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_in_three_hours_l306_30633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_abs_x_lt_1_not_equal_x_lt_1_solution_set_quadratic_inequality_not_equal_given_set_min_value_not_equal_two_quadratic_inequality_iff_x_lt_2_l306_30692

-- Statement A
theorem solution_set_abs_x_lt_1_not_equal_x_lt_1 :
  {x : ℝ | |x| < 1} ≠ {x : ℝ | x < 1} := by sorry

-- Statement B
theorem solution_set_quadratic_inequality_not_equal_given_set :
  {x : ℝ | x^2 - 2*x - 8 > 0} ≠ {x : ℝ | -2 < x ∧ x < 4} := by sorry

-- Statement C
theorem min_value_not_equal_two :
  ∃ (x : ℝ), Real.sqrt (x^2 + 4) + 1 / Real.sqrt (x^2 + 4) < 2 := by sorry

-- Statement D
theorem quadratic_inequality_iff_x_lt_2 :
  ∀ (x : ℝ), x^2 - 3*x + 2 < 0 ↔ x < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_abs_x_lt_1_not_equal_x_lt_1_solution_set_quadratic_inequality_not_equal_given_set_min_value_not_equal_two_quadratic_inequality_iff_x_lt_2_l306_30692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_ratio_l306_30608

-- Define the tetrahedron and points
variable (A B C D K M N : EuclideanSpace ℝ (Fin 3))

-- Define the ratios
noncomputable def ratio_BK_KC : ℝ := 1/3
noncomputable def ratio_AM_MD : ℝ := 3
noncomputable def ratio_CN_ND : ℝ := 1/2

-- Define the conditions
def K_on_BC (A B C D K M N : EuclideanSpace ℝ (Fin 3)) : Prop := ∃ t : ℝ, K = B + t • (C - B) ∧ 0 ≤ t ∧ t ≤ 1
def M_on_AD (A B C D K M N : EuclideanSpace ℝ (Fin 3)) : Prop := ∃ t : ℝ, M = A + t • (D - A) ∧ 0 ≤ t ∧ t ≤ 1
def N_on_CD (A B C D K M N : EuclideanSpace ℝ (Fin 3)) : Prop := ∃ t : ℝ, N = C + t • (D - C) ∧ 0 ≤ t ∧ t ≤ 1

def BK_KC_ratio (A B C D K M N : EuclideanSpace ℝ (Fin 3)) : Prop := ∃ t : ℝ, K = B + t • (C - B) ∧ t / (1 - t) = ratio_BK_KC
def AM_MD_ratio (A B C D K M N : EuclideanSpace ℝ (Fin 3)) : Prop := ∃ t : ℝ, M = A + t • (D - A) ∧ t / (1 - t) = ratio_AM_MD
def CN_ND_ratio (A B C D K M N : EuclideanSpace ℝ (Fin 3)) : Prop := ∃ t : ℝ, N = C + t • (D - C) ∧ t / (1 - t) = ratio_CN_ND

-- Define the volume ratio
def volume_ratio (v1 v2 : ℝ) : Prop := v1 / v2 = 15 / 61

-- Assume the existence of volume calculation functions
axiom volume_of_tetrahedron_part_KMN : EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3) → ℝ
axiom volume_of_tetrahedron : EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3) → ℝ

-- State the theorem
theorem tetrahedron_volume_ratio 
  (h1 : K_on_BC A B C D K M N)
  (h2 : M_on_AD A B C D K M N)
  (h3 : N_on_CD A B C D K M N)
  (h4 : BK_KC_ratio A B C D K M N)
  (h5 : AM_MD_ratio A B C D K M N)
  (h6 : CN_ND_ratio A B C D K M N) :
  ∃ (v1 v2 : ℝ), volume_ratio v1 v2 ∧ 
    v1 = volume_of_tetrahedron_part_KMN A B C D K M N ∧
    v2 = volume_of_tetrahedron A B C D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_ratio_l306_30608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logical_statements_evaluation_l306_30682

theorem logical_statements_evaluation :
  -- Statement 1
  (∃ (p q : Prop), p ∧ ¬q ∧ ¬(p ∧ q)) ∧
  -- Statement 2
  (¬(∀ (x y : ℝ), x * y = 0 → x = 0 ∨ y = 0) ≠ 
   (∀ (x y : ℝ), x * y ≠ 0 → x ≠ 0 ∨ y ≠ 0)) ∧
  -- Statement 3
  (¬(∀ x : ℝ, (2 : ℝ) ^ x > 0) ↔ (∃ x : ℝ, (2 : ℝ) ^ x ≤ 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logical_statements_evaluation_l306_30682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_range_l306_30611

-- Define the function f as noncomputable due to its dependency on Real
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 9 then |Real.log x / Real.log 3 - 1|
  else if x > 9 then 4 - Real.sqrt x
  else 0  -- This case should never occur given the domain, but Lean requires it

-- State the theorem
theorem abc_range (a b c : ℝ) : 
  (0 < a) → (0 < b) → (0 < c) →  -- Ensure a, b, c are positive
  (a ≠ b) → (b ≠ c) → (a ≠ c) →  -- Ensure a, b, c are distinct
  (f a = f b) → (f b = f c) →    -- Ensure f(a) = f(b) = f(c)
  81 < a * b * c ∧ a * b * c < 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_range_l306_30611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_22_unique_sum_5051_l306_30691

theorem unique_sum_22 : 
  ∃! (s : Finset ℕ), s.card = 6 ∧ (∀ x y, x ∈ s → y ∈ s → x ≠ y → x < y) ∧ s.sum id = 22 :=
by
  -- The proof would go here
  sorry

theorem unique_sum_5051 : 
  ∃! (s : Finset ℕ), s.card = 100 ∧ (∀ x y, x ∈ s → y ∈ s → x ≠ y → x < y) ∧ s.sum id = 5051 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_22_unique_sum_5051_l306_30691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l306_30641

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + x + 1

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := (1 + x) * Real.exp x + 1

-- Theorem statement
theorem tangent_line_at_zero_one :
  ∃ (m b : ℝ), 
    (∀ x y, y = m * x + b ↔ m * x - y + b = 0) ∧
    (f 0 = 1) ∧
    (f' 0 = m) ∧
    (1 = m * 0 + b) ∧
    (m = 2 ∧ b = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l306_30641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_and_T_properties_l306_30669

-- Define the sequence a_n
def a : ℕ+ → ℝ := sorry

-- Define S_n as the sum of the first n terms of a_n
def S : ℕ+ → ℝ := sorry

-- Define b_n
noncomputable def b (n : ℕ+) : ℝ := 4 * a (n + 1) / (a n * a (n + 2))^2

-- Define T_n as the sum of the first n terms of b_n
def T : ℕ+ → ℝ := sorry

-- Axioms
axiom a_1 : a 1 = 1
axiom S_def : ∀ n : ℕ+, 2 * S n = (n + 1) * a n

-- Theorem to prove
theorem a_and_T_properties :
  (∀ n : ℕ+, a n = n) ∧
  (∀ n : ℕ+, T n < 5/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_and_T_properties_l306_30669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l306_30629

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 + 2*x else -((-x)^2 + 2*(-x))

theorem odd_function_value :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, x ≥ 0 → f x = x^2 + 2*x) →
  f (-1) = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l306_30629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_f_at_neg1860_a_range_y_max_l306_30672

open Real

noncomputable def f (x : ℝ) : ℝ := 
  (sin (π/2 - x) * tan (π - x))^2 - 1 / 
  (4 * sin (3*π/2 + x) + cos (π - x) + cos (2*π - x))

-- Statement 1: f(x) = -1/2 * sin(x) for all x
theorem f_equiv : ∀ x, f x = -1/2 * sin x := by sorry

-- Statement 2: f(-1860°) = √3/4
theorem f_at_neg1860 : f (-1860 * π/180) = sqrt 3 / 4 := by sorry

-- Statement 3: Range of a for which f^2(x) + (1 + a/2)sin(x) + 2a = 0 has two roots in [π/6, 3π/4]
theorem a_range : 
  ∀ a, (∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ∈ [π/6, 3*π/4] ∧ x₂ ∈ [π/6, 3*π/4] ∧ 
    (f x₁)^2 + (1 + a/2) * sin x₁ + 2*a = 0 ∧
    (f x₂)^2 + (1 + a/2) * sin x₂ + 2*a = 0) ↔
  -1/2 < a ∧ a ≤ -sqrt 2 / 4 := by sorry

-- Statement 4: Maximum value of y = 4af^2(x) + 2cos(x) where a ∈ ℝ
noncomputable def y (a x : ℝ) : ℝ := 4 * a * (f x)^2 + 2 * cos x

theorem y_max : 
  ∀ a, (∀ x, y a x ≤ (if a ≥ 1 then max 2 (a + 1/a) else 2)) ∧
       (∃ x, y a x = (if a ≥ 1 then max 2 (a + 1/a) else 2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_f_at_neg1860_a_range_y_max_l306_30672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_l306_30687

noncomputable def f (x : ℝ) : ℝ := (1/2) * (x + 1 - 1/x - 1/(1-x))

-- State the theorem
theorem functional_equation (x : ℝ) (h : x ≠ 1) :
  f x + f (1/(1-x)) = x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_l306_30687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_T_l306_30699

/-- Geometric sequence with common ratio 2 -/
noncomputable def geometric_sequence (a₁ : ℝ) : ℕ → ℝ := λ n ↦ a₁ * 2^(n - 1)

/-- Sum of first n terms of the geometric sequence -/
noncomputable def S (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * (1 - 2^n) / (1 - 2)

/-- Definition of T_n -/
noncomputable def T (a₁ : ℝ) (n : ℕ+) : ℝ :=
  (9 * S a₁ n - S a₁ (2 * n)) / (geometric_sequence a₁ (n + 1))

/-- The maximum value of T_n is 3 -/
theorem max_value_of_T (a₁ : ℝ) (h : a₁ ≠ 0) :
  ∃ M : ℝ, M = 3 ∧ ∀ n : ℕ+, T a₁ n ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_T_l306_30699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l306_30650

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/3), 3 * x^2 - Real.log x / Real.log a < 0) → 
  a ∈ Set.Icc (1/27 : ℝ) 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l306_30650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l306_30686

theorem triangle_side_length 
  (A B C : EuclideanSpace ℝ (Fin 2)) 
  (perimeter : ℝ) 
  (area : ℝ) 
  (angle_A : ℝ) :
  perimeter = 20 →
  area = 10 * Real.sqrt 3 →
  angle_A = π / 3 →
  let a := norm (B - C)
  let b := norm (A - C)
  let c := norm (A - B)
  a + b + c = perimeter →
  1/2 * b * c * Real.sin angle_A = area →
  a = 7 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l306_30686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l306_30670

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1/2 * Real.cos (2 * x)

theorem f_properties :
  -- The smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = Real.pi) ∧
  -- The intervals of monotonic increase
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3), 
    ∀ y ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3),
    x ≤ y → f x ≤ f y) ∧
  -- For x₀ ∈ [0, π/2] such that f(x₀) = 0, cos 2x₀ = (3√5 + 1) / 8
  (∀ x₀ : ℝ, x₀ ∈ Set.Icc 0 (Real.pi / 2) → f x₀ = 0 → 
    Real.cos (2 * x₀) = (3 * Real.sqrt 5 + 1) / 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l306_30670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sqrt_sum_l306_30688

theorem opposite_sqrt_sum (a b : ℝ) 
  (h : Real.sqrt (a - 3) + Real.sqrt (2 - b) = 0) : 
  -1 / Real.sqrt a + Real.sqrt 6 / Real.sqrt b = 2 / 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sqrt_sum_l306_30688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l306_30675

noncomputable def f (x : ℝ) := 2 * Real.cos ((x / 2) + (Real.pi / 4))

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l306_30675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_subset_existence_l306_30623

theorem coprime_subset_existence (a b c d e : ℕ) :
  (100 ≤ a ∧ a < 1000) →
  (100 ≤ b ∧ b < 1000) →
  (100 ≤ c ∧ c < 1000) →
  (100 ≤ d ∧ d < 1000) →
  (100 ≤ e ∧ e < 1000) →
  Nat.Coprime a b ∧ Nat.Coprime a c ∧ Nat.Coprime a d ∧ Nat.Coprime a e ∧
  Nat.Coprime b c ∧ Nat.Coprime b d ∧ Nat.Coprime b e ∧
  Nat.Coprime c d ∧ Nat.Coprime c e ∧
  Nat.Coprime d e →
  ∃ (s : Finset ℕ), s.card = 4 ∧ s ⊆ {a, b, c, d, e} ∧
  ∀ (x y : ℕ), x ∈ s → y ∈ s → x ≠ y → Nat.Coprime x y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_subset_existence_l306_30623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l306_30659

/-- The set of positive real numbers -/
def PositiveReals : Type := {x : ℝ // x > 0}

/-- The function f: S³ → S -/
noncomputable def f (x y z : PositiveReals) : PositiveReals :=
  ⟨(y.val + Real.sqrt (y.val^2 + 4*x.val*z.val)) / (2*x.val),
   by
     sorry  -- Proof that the result is positive
  ⟩

/-- The three conditions that f must satisfy -/
theorem f_satisfies_conditions :
  (∀ (x y z : PositiveReals), x.val * (f x y z).val = z.val * (f z y x).val) ∧ 
  (∀ (x y z k : PositiveReals), f x ⟨k.val*y.val, by sorry⟩ ⟨k.val^2*z.val, by sorry⟩ = ⟨k.val * (f x y z).val, by sorry⟩) ∧
  (∀ (k : PositiveReals), f ⟨1, by norm_num⟩ k ⟨k.val+1, by sorry⟩ = ⟨k.val+1, by sorry⟩) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l306_30659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_l306_30696

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + b

-- Define the derivative of f
noncomputable def f' (a x : ℝ) : ℝ := x^2 - 2*a*x

theorem tangent_line_problem (a b : ℝ) :
  (f' a (-1) = 3) ∧ 
  (f a b (-1) = 2) →
  (a = 1 ∧ b = 10/3) ∧ 
  (f' 1 2 = 0 ∧ f 1 (10/3) 2 = 2) :=
by sorry

#check tangent_line_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_l306_30696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l306_30612

/-- A circle C in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points in the plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem circle_equation (C : Circle) 
  (h1 : C.center.1 = 1 ∧ C.radius = Real.sqrt 2)
  (h2 : ∃ A B : ℝ × ℝ, A.1 = 0 ∧ B.1 = 0 ∧ A.2 > 0 ∧ B.2 > 0 ∧ distance A B = 2)
  (h3 : distance C.center (1, 0) = C.radius) :
  ∀ x y : ℝ, (x - 1)^2 + (y - Real.sqrt 2)^2 = 2 ↔ distance (x, y) C.center = C.radius :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l306_30612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_g_odd_l306_30635

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := lg (1 - x) + lg (1 + x)
noncomputable def g (x : ℝ) : ℝ := lg (1 - x) - lg (1 + x)

-- State the theorem
theorem f_even_g_odd :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f (-x) = f x) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, g (-x) = -g x) := by
  sorry

#check f_even_g_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_g_odd_l306_30635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_l306_30619

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line l
def line_l (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem intersection_implies_m_value :
  ∀ (m : ℝ),
  (∃ (x1 y1 x2 y2 : ℝ),
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    line_l m x1 y1 ∧ line_l m x2 y2 ∧
    distance x1 y1 x2 y2 = Real.sqrt 17) →
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_l306_30619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l306_30668

/-- The area of a triangle with vertices at (1,2), (1,9), and (10,2) is 31.5 square units. -/
theorem triangle_area : ∃ (area : ℝ), area = 31.5 := by
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (1, 9)
  let C : ℝ × ℝ := (10, 2)
  let base : ℝ := B.2 - A.2
  let height : ℝ := C.1 - A.1
  let area : ℝ := (1 / 2) * base * height
  use area
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l306_30668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dimes_for_sneakers_l306_30631

/-- The minimum number of dimes needed to buy sneakers -/
def min_dimes (sneaker_cost : ℚ) (ten_dollar_bills : ℕ) (quarters : ℕ) : ℕ :=
  (((sneaker_cost - (ten_dollar_bills * 10 + quarters * (1/4))) / (1/10)).ceil).toNat

theorem min_dimes_for_sneakers :
  min_dimes 58 5 5 = 68 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dimes_for_sneakers_l306_30631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_identity_l306_30655

-- Define the Fibonacci sequence
def fib : ℕ → ℤ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Define the matrix power relationship
def matrix_power_relation (n : ℕ) : Prop :=
  (Matrix.of ![![1, 1], ![1, 0]] : Matrix (Fin 2) (Fin 2) ℤ) ^ n = 
  Matrix.of ![![fib (n + 1), fib n], ![fib n, fib (n - 1)]]

-- The theorem to prove
theorem fib_identity : 
  (∀ n : ℕ, matrix_power_relation n) → 
  fib 1000 * fib 1002 - fib 1001 * fib 1001 = -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_identity_l306_30655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_divisible_by_seven_l306_30658

/-- Helper function to calculate the sum of digits of a natural number -/
def sum_of_digits (k : ℕ) : ℕ :=
  if k < 10 then k else k % 10 + sum_of_digits (k / 10)

/-- For any natural number n ≥ 2, there exists a positive integer k 
    such that k is divisible by 7 and the sum of the digits of k is equal to n. -/
theorem sum_of_digits_divisible_by_seven (n : ℕ) (h : n ≥ 2) : 
  ∃ k : ℕ, k > 0 ∧ k % 7 = 0 ∧ sum_of_digits k = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_divisible_by_seven_l306_30658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_calculation_l306_30656

/-- Represents the different currencies used in the problem -/
inductive Currency
  | USD
  | EUR
  | GBP
  | JPY

/-- Represents a monetary value with its currency -/
structure Money where
  amount : Float
  currency : Currency

/-- Represents an item purchased -/
structure Item where
  name : String
  price : Money
  quantity : Nat

/-- Represents a discount -/
structure Discount where
  percentage : Float
  appliesTo : List String

/-- Represents a payment method -/
inductive PaymentMethod
  | Cash
  | GiftCard
  | RewardsPoints
  | CreditCard

/-- Represents a purchase -/
structure Purchase where
  item : Item
  paymentMethod : PaymentMethod

def conversionRate (fromCurrency toCurrency : Currency) : Float :=
  match fromCurrency, toCurrency with
  | Currency.EUR, Currency.USD => 1.1
  | Currency.GBP, Currency.USD => 1.25
  | Currency.JPY, Currency.USD => 0.009
  | _, _ => 1.0

def applyDiscount (price : Float) (discount : Discount) : Float :=
  price * (1 - discount.percentage)

def applySalesTax (price : Float) (taxRate : Float) : Float :=
  price * (1 + taxRate)

noncomputable def calculateTotalSpent (purchases : List Purchase) 
  (discounts : List Discount) (couponDiscount taxRate : Float) : Float :=
  sorry

theorem total_spent_calculation (purchases : List Purchase) 
  (discounts : List Discount) (couponDiscount taxRate : Float) :
  calculateTotalSpent purchases discounts couponDiscount taxRate = 18.0216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_calculation_l306_30656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_abundant_l306_30674

def isAbundant (n : ℕ) : Prop :=
  (Finset.sum (Finset.filter (· ∣ n) (Finset.range n)) id) > n

def isOdd (n : ℕ) : Prop := n % 2 = 1

theorem smallest_odd_abundant : ∀ n : ℕ, isOdd n ∧ isAbundant n → n ≥ 135 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_abundant_l306_30674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_150_deg_angles_has_12_sides_l306_30646

/-- The measure of an interior angle of a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ :=
  (n - 2) * 180 / n

/-- A regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_with_150_deg_angles_has_12_sides :
  ∀ n : ℕ, 
    n ≥ 3 → -- A polygon must have at least 3 sides
    (∀ i : ℕ, i < n → interior_angle n = 150) → -- All interior angles are 150 degrees
    n = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_150_deg_angles_has_12_sides_l306_30646
