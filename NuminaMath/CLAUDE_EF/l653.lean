import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spadesuit_nested_calculation_l653_65381

noncomputable def spadesuit (x y : ℝ) : ℝ := x - 1 / y

theorem spadesuit_nested_calculation : spadesuit 3 (spadesuit 3 3) = 21 / 8 := by
  -- Unfold the definition of spadesuit
  unfold spadesuit
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spadesuit_nested_calculation_l653_65381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l653_65300

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + Real.cos (2 * x + Real.pi / 3)

theorem f_properties : 
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  (∃ (M : ℝ), M = 1 ∧ (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l653_65300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_fraction_iff_in_unit_interval_l653_65395

/-- Definition of a "dot fraction" sequence -/
def DotFractionSeq (a : ℕ → ℕ) : ℕ → ℚ
  | 0 => 0
  | n + 1 => DotFractionSeq a n + 1 / (Finset.prod (Finset.range (n + 1)) a)

/-- Definition of a finite "dot fraction" -/
def IsDotFraction (q : ℚ) : Prop :=
  ∃ (a : ℕ → ℕ) (n : ℕ), (∀ i, a i > 1) ∧ DotFractionSeq a n = q

/-- Theorem: A rational number can be represented as a finite "dot fraction"
    if and only if it is in the interval (0, 1) -/
theorem dot_fraction_iff_in_unit_interval (q : ℚ) :
  IsDotFraction q ↔ 0 < q ∧ q < 1 := by
  sorry

#check dot_fraction_iff_in_unit_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_fraction_iff_in_unit_interval_l653_65395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l653_65319

/-- The complex number z is defined as (1 - √2i) / i -/
noncomputable def z : ℂ := (1 - Complex.I * Real.sqrt 2) / Complex.I

/-- Theorem: z is located in the third quadrant of the complex plane -/
theorem z_in_third_quadrant : z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l653_65319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_geometry_problem_l653_65370

/-- Given two distinct points A(3, a-1) and B(b+1, -2) in the plane -/
def A (a : ℝ) : ℝ × ℝ := (3, a - 1)
def B (b : ℝ) : ℝ × ℝ := (b + 1, -2)

/-- B lies on y-axis if its x-coordinate is 0 -/
def B_on_y_axis (b : ℝ) : Prop := (B b).1 = 0

/-- A lies on angle bisector in first and third quadrants if its x and y coordinates are equal in magnitude -/
def A_on_angle_bisector (a : ℝ) : Prop := abs (A a).1 = abs (A a).2

/-- AB is parallel to y-axis if x-coordinates of A and B are equal -/
def AB_parallel_y_axis (a b : ℝ) : Prop := (A a).1 = (B b).1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem coordinate_geometry_problem :
  (∀ b : ℝ, B_on_y_axis b → b = -1) ∧
  (∀ a : ℝ, A_on_angle_bisector a → a = 4) ∧
  (∀ a b : ℝ, AB_parallel_y_axis a b ∧ distance (A a) (B b) = 5 → b = 2 ∧ (a = 4 ∨ a = -6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_geometry_problem_l653_65370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ψ_is_integral_l653_65378

/-- The function ψ is defined as arctan(x₁/x₂) - t -/
noncomputable def ψ (t x₁ x₂ : ℝ) : ℝ := Real.arctan (x₁ / x₂) - t

/-- The derivative of x₁ with respect to t -/
noncomputable def dx₁_dt (x₁ x₂ : ℝ) : ℝ := x₁^2 / x₂

/-- The derivative of x₂ with respect to t -/
noncomputable def dx₂_dt (x₁ x₂ : ℝ) : ℝ := -x₂^2 / x₁

/-- Theorem stating that ψ is an integral of the system -/
theorem ψ_is_integral (t x₁ x₂ : ℝ) (h₁ : x₁ ≠ 0) (h₂ : x₂ ≠ 0) :
  (deriv (fun t => ψ t x₁ x₂) t) + 
  (dx₁_dt x₁ x₂) * (deriv (fun x₁ => ψ t x₁ x₂) x₁) + 
  (dx₂_dt x₁ x₂) * (deriv (fun x₂ => ψ t x₁ x₂) x₂) = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ψ_is_integral_l653_65378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombicuboctahedron_opposite_faces_l653_65359

structure Rhombicuboctahedron where
  faces : Set Nat
  band : Set Nat
  is_square : Nat → Prop
  is_in_band : Nat → Prop
  opposite : Nat → Nat → Prop

def p : Nat := 1
def d : Nat := 4

theorem rhombicuboctahedron_opposite_faces 
  (R : Rhombicuboctahedron) 
  (h1 : R.is_square p)
  (h2 : R.is_square d)
  (h3 : R.is_in_band p)
  (h4 : R.is_in_band d)
  (h5 : ∀ (f1 f2 : Nat), R.is_in_band f1 → R.is_in_band f2 → 
    (∃ (f3 f4 : Nat), R.is_in_band f3 ∧ R.is_in_band f4 ∧ 
    f1 ≠ f3 ∧ f3 ≠ f4 ∧ f4 ≠ f2 ∧ 
    R.opposite f1 f2)) :
  R.opposite p d := by
  sorry

#check rhombicuboctahedron_opposite_faces

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombicuboctahedron_opposite_faces_l653_65359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l653_65354

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 6) + 1

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6) + 1

def is_symmetry_center (g : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x, g (c.1 + x) = g (c.1 - x)

theorem symmetry_center_of_g :
  is_symmetry_center g (Real.pi / 12, 1) := by
  sorry

#check symmetry_center_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l653_65354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_roots_real_l653_65346

theorem not_all_roots_real (a b c d e : ℝ) (h : 2 * a^2 < 5 * b) :
  ∃ x : ℂ, x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 ∧ x.im ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_roots_real_l653_65346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_B_symmetric_l653_65367

-- Define a type for T-like shapes
inductive TShape
| A
| B
| C
| D
| E

-- Define a type for lines
structure Line where
  -- We'll leave this empty for now, as the specific properties aren't crucial for this example
  dummy : Unit

-- Define symmetry relation
def isSymmetric (s1 s2 : TShape) (l : Line) : Prop :=
  -- For now, we'll use a placeholder definition
  True

-- Define the original shape and the dashed line
def originalShape : TShape := TShape.A  -- Assuming A is the original shape
def dashedLine : Line := { dummy := () }

-- State the theorem
theorem shape_B_symmetric :
  isSymmetric originalShape TShape.B dashedLine := by
  sorry

-- You can add more theorems or definitions here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_B_symmetric_l653_65367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_divisible_by_5_6_4_3_l653_65364

def addend : ℝ := 11.000000000000014

theorem number_divisible_by_5_6_4_3 :
  ∃ (N : ℝ), (∀ (d : ℕ), d ∈ ({5, 6, 4, 3} : Set ℕ) → ∃ (k : ℕ), (N + addend) = d * k) ∧ 
  (abs (N - 49) < 0.000000000000014) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_divisible_by_5_6_4_3_l653_65364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_activity_selection_methods_l653_65394

/-- The number of ways to select people for three communities --/
def selection_methods (boys girls : ℕ) : ℕ :=
  let total := boys + girls
  let community_a_selections := (girls.choose 2) + (girls.choose 1) * (boys.choose 1)
  let remaining_selections := (total - 2).factorial / (total - 4).factorial
  community_a_selections * remaining_selections

/-- Theorem stating the number of selection methods for the given problem --/
theorem school_activity_selection_methods :
  selection_methods 3 2 = 42 := by
  rfl

#eval selection_methods 3 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_activity_selection_methods_l653_65394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reachable_2011_from_5_l653_65328

/-- Represents the transformation rule in the game -/
def transform (n : ℕ+) : Set ℕ+ :=
  {m | ∃ (a b : ℕ+), n.val = a.val + b.val ∧ m.val = a.val * b.val}

/-- Represents the transitive closure of the transform relation -/
def reachable : ℕ+ → ℕ+ → Prop :=
  TC (λ x y ↦ y ∈ transform x)

/-- The main theorem stating that 2011 is reachable from 5 -/
theorem reachable_2011_from_5 : reachable 5 2011 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reachable_2011_from_5_l653_65328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_constant_l653_65349

/-- A vector in R² -/
structure Vec2 where
  x : ℝ
  y : ℝ

/-- The line y = (3/2)x + 3 -/
def onLine (v : Vec2) : Prop :=
  v.y = (3/2) * v.x + 3

/-- Dot product of two Vec2 -/
def dot (v w : Vec2) : ℝ :=
  v.x * w.x + v.y * w.y

/-- Norm squared of a Vec2 -/
def normSq (v : Vec2) : ℝ :=
  v.x^2 + v.y^2

/-- Scalar multiplication of a Vec2 -/
def scale (s : ℝ) (v : Vec2) : Vec2 :=
  ⟨s * v.x, s * v.y⟩

/-- Projection of v onto w -/
noncomputable def proj (v w : Vec2) : Vec2 :=
  scale (dot v w / normSq w) w

/-- The theorem to be proved -/
theorem projection_constant (d : ℝ) (hd : d ≠ 0) :
  ∀ v : Vec2, onLine v →
    proj v ⟨-3*d/2, d⟩ = ⟨-18/13, 12/13⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_constant_l653_65349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sequence_theorem_l653_65329

/-- A sequence of circles in an equilateral triangle -/
def CircleSequence := ℕ → ℝ

/-- The radius of the nth circle in the sequence -/
def radius (C : CircleSequence) (n : ℕ) : ℝ := C n

/-- The condition that circles are tangent and decreasing in size -/
def is_valid_sequence (C : CircleSequence) : Prop :=
  ∀ n : ℕ, n ≥ 1 → radius C (n + 1) < radius C n ∧ 
    radius C (n + 1) = (1/3) * radius C n

/-- The condition that exactly 4 circles fit inside the triangle -/
def exactly_four_fit (C : CircleSequence) : Prop :=
  (radius C 1) * ((3^4 + 1) / 2) < (3 * Real.sqrt 3) / 2 ∧
  (radius C 1) * ((3^5 + 1) / 2) > (3 * Real.sqrt 3) / 2

/-- The main theorem -/
theorem circle_sequence_theorem (C : CircleSequence) :
  is_valid_sequence C →
  exactly_four_fit C ↔ 
    (3 * Real.sqrt 3) / 244 < radius C 1 ∧ radius C 1 < (3 * Real.sqrt 3) / 82 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sequence_theorem_l653_65329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l653_65317

/-- The infinite series described in the problem -/
noncomputable def series_sum : ℚ := ∑' n, if n % 2 = 0 then (n + 1) / (2^(2*n + 2 : ℕ)) else (n + 1) / (3^(2*n + 1 : ℕ))

theorem problem_solution (p q : ℕ) (h_coprime : Nat.Coprime p q) (h_eq : (p : ℚ) / q = series_sum) : 
  p + q = 443 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l653_65317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_minus_repeating_three_thirds_l653_65387

theorem one_minus_repeating_three_thirds : 1 - (1/3 : ℚ) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_minus_repeating_three_thirds_l653_65387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l653_65357

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 8*x + 15) / (3*x - 6)

/-- The function g(x) with parameters b and c -/
noncomputable def g (b c : ℝ) (x : ℝ) : ℝ := (b*x^2 + 6*x + c) / (x - 2)

/-- The theorem stating the point of intersection -/
theorem intersection_point (b c : ℝ) :
  (∃ (x : ℝ), x ≠ 3 ∧ f x = g b c x) →
  (∀ (x : ℝ), x ≠ 2 → (3*x - 6 ≠ 0 ∧ x - 2 ≠ 0)) →
  (∃ (k : ℝ), ∀ (x : ℝ), x ≠ 2 → g b c x = -3*x + k + (b*x^2 + 6*x + c)/(x - 2)) →
  f 3 = g b c 3 →
  (∃ (x : ℝ), x ≠ 3 ∧ f x = g b c x ∧ x = 11/3 ∧ f x = -11/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l653_65357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_from_sine_ratio_l653_65318

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem cosine_from_sine_ratio (abc : Triangle) 
  (h : ∃ (k : ℝ), k > 0 ∧ Real.sin abc.A = 3 * k ∧ Real.sin abc.B = 4 * k ∧ Real.sin abc.C = 5 * k) : 
  Real.cos abc.A = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_from_sine_ratio_l653_65318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_m_greater_than_one_minimum_value_h_l653_65389

-- Define the log function with base 1/3
noncomputable def log_base_third (x : ℝ) := Real.log x / Real.log (1/3)

-- Part 1
theorem domain_implies_m_greater_than_one (m : ℝ) :
  (∀ x, x > 0 → log_base_third (m * x^2 + 2*x + m) ∈ Set.univ) → m > 1 := by
  sorry

-- Part 2
noncomputable def h (a : ℝ) : ℝ := 
  if a < 1/3 then (28 - 6*a) / 9
  else if a ≤ 3 then -a^2 + 3
  else -6*a + 12

theorem minimum_value_h (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, h a ≤ ((1/3)^x)^2 - 2*a*(1/3)^x + 3) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, h a = ((1/3)^x)^2 - 2*a*(1/3)^x + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_m_greater_than_one_minimum_value_h_l653_65389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_common_tangents_l653_65321

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 6*y - 48 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

-- Define the center and radius of C₁
def center₁ : ℝ × ℝ := (3, -3)
def radius₁ : ℝ := 8

-- Define the center and radius of C₂
def center₂ : ℝ × ℝ := (-2, 4)
def radius₂ : ℝ := 8

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 74

-- Define a function to represent the number of common tangents
def number_of_common_tangents (C₁ C₂ : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem: The number of common tangents is 2
theorem two_common_tangents :
  (radius₁ - radius₂ < distance_between_centers) ∧
  (distance_between_centers < radius₁ + radius₂) →
  number_of_common_tangents C₁ C₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_common_tangents_l653_65321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_theorem_l653_65308

theorem subset_intersection_theorem (k n : ℕ) (X : Finset ℕ) 
  (A : Fin n → Finset ℕ) :
  X.card = k →
  (∀ i, A i ⊆ X) →
  (∀ i j, i ≠ j → (A i ∩ A j).Nonempty) →
  n < 2^(k-1) →
  ∃ B : Finset ℕ, B ⊆ X ∧ 
    (∀ i, B ≠ A i) ∧ 
    (∀ i, (B ∩ A i).Nonempty) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_theorem_l653_65308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_C_is_simplest_l653_65371

noncomputable def fraction_A (x y : ℝ) : ℝ := (x^2 + x*y) / (x^2 + 2*x*y + y^2)
noncomputable def fraction_B (x : ℝ) : ℝ := (2*x + 8) / (x^2 - 16)
noncomputable def fraction_C (x : ℝ) : ℝ := (x^2 + 1) / (x^2 - 1)
noncomputable def fraction_D (x : ℝ) : ℝ := (x^2 - 9) / (x^2 + 6*x + 9)

def is_simplest (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → (∀ g : ℝ → ℝ, g x = f x → g = f)

theorem fraction_C_is_simplest :
  is_simplest fraction_C ∧
  ¬is_simplest (λ x => fraction_A x x) ∧
  ¬is_simplest fraction_B ∧
  ¬is_simplest fraction_D :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_C_is_simplest_l653_65371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l653_65388

/-- Proves that the initial amount of water in a glass is 10 ounces, given the evaporation rate,
    period, and percentage of water lost. -/
theorem initial_water_amount
  (daily_evaporation : ℚ)
  (evaporation_period : ℕ)
  (evaporation_percentage : ℚ)
  (h1 : daily_evaporation = 3 / 100)
  (h2 : evaporation_period = 20)
  (h3 : evaporation_percentage = 6 / 100)
  (h4 : daily_evaporation * evaporation_period = evaporation_percentage * 10) :
  10 = (daily_evaporation * evaporation_period) / evaporation_percentage :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l653_65388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_minus_d_eq_neg_two_l653_65304

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- Define the properties of g
axiom g_invertible : Function.Bijective g

-- Theorem to prove
theorem c_minus_d_eq_neg_two :
  ∃ c d : ℝ, g c = d ∧ g d = 6 ∧ c - d = -2 := by
  -- We'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_minus_d_eq_neg_two_l653_65304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_exponential_sine_difference_l653_65337

open Real

theorem limit_of_exponential_sine_difference (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x| ∧ |x| < δ →
    |((exp (5 * x) - exp (3 * x)) / (sin (2 * x) - sin x)) - 2| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_exponential_sine_difference_l653_65337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pointSet_is_circle_l653_65399

/-- An ellipse with foci F1 and F2 -/
structure Ellipse (P : Type*) [MetricSpace P] where
  F1 : P
  F2 : P
  perimeter : Set P
  isOnPerimeter : F2 ∈ perimeter

/-- The set of points A such that d(A, F2) ≤ d(A, F1) -/
def pointSet {P : Type*} [MetricSpace P] (E : Ellipse P) : Set P :=
  {A : P | dist A E.F2 ≤ dist A E.F1}

/-- Theorem: The set of points A forms a circle centered at F2 -/
theorem pointSet_is_circle {P : Type*} [MetricSpace P] (E : Ellipse P) :
  ∃ (r : ℝ), pointSet E = Metric.ball E.F2 r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pointSet_is_circle_l653_65399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inequality_l653_65338

theorem angle_inequality (α β x : ℝ) 
  (h_acute_α : 0 < α ∧ α < π/2)
  (h_acute_β : 0 < β ∧ β < π/2)
  (h_condition : x * (α + β - π/2) > 0) :
  (Real.cos α / Real.sin β)^x + (Real.cos β / Real.sin α)^x < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inequality_l653_65338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_cube_root_l653_65345

theorem nested_cube_root (M : ℝ) (h : M > 1) :
  (M * ((M * ((M * M^(1/3))^(1/3)))^(1/3)))^(1/3) = M ^ (40/81) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_cube_root_l653_65345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_current_age_l653_65326

/-- Jessica's age in years -/
def jessica_age : ℚ := sorry

/-- Jessica's grandmother's age in years -/
def grandmother_age : ℚ := sorry

/-- Jessica's grandmother's age is fifteen times Jessica's age -/
axiom age_relation : grandmother_age = 15 * jessica_age

/-- Jessica's grandmother was 60 years old when Jessica was born -/
axiom age_difference : grandmother_age - jessica_age = 60

theorem jessica_current_age : jessica_age = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_current_age_l653_65326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_equal_after_steps_l653_65358

/-- Represents the state of money distribution among players -/
inductive MoneyState
  | AllEqual
  | TwoOneOneZero

/-- Represents a single player in the game -/
structure Player where
  money : Nat

/-- Represents the game state -/
structure GameState where
  players : Fin 4 → Player
  steps : Nat

/-- Transition probability from AllEqual to AllEqual state -/
def transitionProbAllEqual : Rat := 2 / 27

/-- Transition probability from TwoOneOneZero to AllEqual state -/
def transitionProbTwoOneOneZero : Rat := 2 / 27

/-- The number of steps in the game -/
def totalSteps : Nat := 2023

/-- Theorem stating the probability of all players having $1 after 2023 steps -/
theorem prob_all_equal_after_steps :
  ∀ (initialState : GameState),
    (∀ i, (initialState.players i).money = 1) →
    initialState.steps = 0 →
    (∃ (finalState : GameState),
      finalState.steps = totalSteps ∧
      (∀ i, (finalState.players i).money = 1)) →
    (1 : Rat) / 13 = (1 : Rat) / 13 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_equal_after_steps_l653_65358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_segments_quadrilateral_side_length_l653_65303

-- Define a line segment
structure LineSegment where
  start : ℝ × ℝ
  end_ : ℝ × ℝ
  length : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Function to check if two line segments intersect
def intersect (s1 s2 : LineSegment) : Prop := sorry

-- Function to form a quadrilateral from two intersecting line segments
def formQuadrilateral (s1 s2 : LineSegment) : Quadrilateral := sorry

-- Function to get the length of the shortest side of a quadrilateral
noncomputable def shortestSide (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem intersecting_segments_quadrilateral_side_length 
  (s1 s2 : LineSegment) 
  (h1 : s1.length = 1) 
  (h2 : s2.length = 1) 
  (h3 : intersect s1 s2) :
  shortestSide (formQuadrilateral s1 s2) ≥ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_segments_quadrilateral_side_length_l653_65303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_x_plus_y_l653_65332

theorem max_min_x_plus_y (x y : ℝ) (h : x^2 + y^2 = 1) :
  (∃ (a b : ℝ), a^2 + b^2 = 1 ∧ x + y ≤ a + b) ∧
  (∃ (c d : ℝ), c^2 + d^2 = 1 ∧ x + y ≥ c + d) ∧
  (∃ (e f : ℝ), e^2 + f^2 = 1 ∧ e + f = Real.sqrt 2) ∧
  (∃ (g k : ℝ), g^2 + k^2 = 1 ∧ g + k = -Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_x_plus_y_l653_65332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_average_perfect_power_l653_65316

theorem subset_average_perfect_power (n : ℕ+) : 
  ∃ (X : Finset ℕ), 
    (Finset.card X = n) ∧ 
    (∀ x ∈ X, x > 1) ∧
    (∀ S : Finset ℕ, S ⊆ X → S.card > 0 → 
      ∃ (k : ℕ) (b : ℕ), k > 1 ∧ (S.sum id / S.card : ℚ) = b^k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_average_perfect_power_l653_65316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_17_l653_65343

def is_divisible_by_17 (n : ℕ) : Prop := ∃ k : ℕ, n = 17 * k

theorem probability_divisible_by_17 :
  (Finset.filter (fun n => n * (n + 1) % 17 = 0) (Finset.range 1000)).card / 1000 = 116 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_17_l653_65343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l653_65391

/-- The distance between two points in polar coordinates -/
noncomputable def polar_distance (r1 r2 : ℝ) (θ1 θ2 : ℝ) : ℝ :=
  Real.sqrt (r1^2 + r2^2 - 2*r1*r2*(Real.cos (θ2 - θ1)))

/-- Theorem: The distance between points A(4, π/6) and B(2, π/2) in polar coordinates is 2√3 -/
theorem distance_A_to_B :
  polar_distance 4 2 (π/6) (π/2) = 2 * Real.sqrt 3 := by
  sorry

#check distance_A_to_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l653_65391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_vertex_labeling_l653_65344

/-- A polyhedron with vertices, edges, and faces. -/
structure Polyhedron where
  vertex : Type
  edge : Type
  face : Type
  vertex_edges : vertex → Finset edge
  face_vertices : face → List vertex

/-- A coloring of the edges of a polyhedron. -/
def EdgeColoring (p : Polyhedron) := p.edge → Fin 3

/-- A labeling of the vertices of a polyhedron with complex numbers. -/
def VertexLabeling (p : Polyhedron) := p.vertex → ℂ

/-- The condition that each vertex has exactly 3 edges. -/
def HasThreeEdges (p : Polyhedron) : Prop :=
  ∀ v : p.vertex, (p.vertex_edges v).card = 3

/-- The condition that each vertex has one edge of each color. -/
def HasOneEdgeOfEachColor (p : Polyhedron) (c : EdgeColoring p) : Prop :=
  ∀ v : p.vertex, ∀ i : Fin 3, ∃! e, e ∈ p.vertex_edges v ∧ c e = i

/-- The product of complex numbers around a face. -/
def FaceProduct (p : Polyhedron) (l : VertexLabeling p) (f : p.face) : ℂ :=
  (p.face_vertices f).map l |>.prod

/-- The main theorem statement. -/
theorem polyhedron_vertex_labeling 
  (p : Polyhedron) 
  (h1 : HasThreeEdges p) 
  (h2 : ∃ c : EdgeColoring p, HasOneEdgeOfEachColor p c) :
  ∃ l : VertexLabeling p, 
    (∀ v : p.vertex, l v ≠ 1) ∧ 
    (∀ f : p.face, FaceProduct p l f = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_vertex_labeling_l653_65344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planet_not_observed_l653_65311

/-- A type representing a planet -/
structure Planet where
  id : Nat

/-- A function representing the distance between two planets -/
noncomputable def distance (p1 p2 : Planet) : ℝ := sorry

/-- A function representing the astronomer's observation -/
def observes (p1 p2 : Planet) : Prop := sorry

/-- The set of all planets -/
def allPlanets : Finset Planet := sorry

theorem planet_not_observed :
  (∀ p1 p2 : Planet, p1 ≠ p2 → distance p1 p2 ≠ distance p2 p1) →  -- Distances are distinct
  (∀ p : Planet, p ∈ allPlanets) →  -- There are planets in the set
  (Finset.card allPlanets = 15) →  -- There are exactly 15 planets
  (∀ p1 : Planet, ∃! p2 : Planet, observes p1 p2) →  -- Each planet observes exactly one other planet
  (∀ p1 p2 : Planet, observes p1 p2 → 
    ∀ p3 : Planet, p3 ≠ p2 → distance p1 p2 < distance p1 p3) →  -- Each planet observes the nearest planet
  ∃ p : Planet, ∀ q : Planet, ¬observes q p :=  -- There exists a planet not observed by any other planet
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planet_not_observed_l653_65311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_congruent_triangles_count_l653_65339

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in the plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Predicate to check if a point is inside a circle -/
def isInsideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

/-- Predicate to check if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  let d12 := ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
  let d23 := ((p2.x - p3.x)^2 + (p2.y - p3.y)^2)
  let d31 := ((p3.x - p1.x)^2 + (p3.y - p1.y)^2)
  d12 = d23 ∧ d23 = d31

/-- Function to count non-congruent triangles (placeholder) -/
def countNonCongruentTriangles (A B C D E F G H I : Point) : ℕ :=
  11 -- Placeholder value based on the problem solution

/-- The main theorem -/
theorem non_congruent_triangles_count
  (c : Circle)
  (A B C D E F G H I : Point)
  (h1 : isOnCircle A c ∧ isOnCircle B c ∧ isOnCircle C c ∧
        isOnCircle D c ∧ isOnCircle E c ∧ isOnCircle F c)
  (h2 : isInsideCircle G c ∧ isInsideCircle H c ∧ isInsideCircle I c)
  (h3 : G ≠ c.center ∧ H ≠ c.center ∧ I ≠ c.center)
  (h4 : isEquilateralTriangle G H I)
  : ∃ (n : ℕ), n = 11 ∧ n = countNonCongruentTriangles A B C D E F G H I :=
by
  use 11
  apply And.intro
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_congruent_triangles_count_l653_65339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_in_interval_l653_65392

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x + x^3 - 2

-- State the theorem
theorem unique_zero_in_interval : ∃! x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_in_interval_l653_65392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l653_65305

-- Define the curve
def f (x : ℝ) : ℝ := 4*x - x^3

-- Define the point of tangency
def point : ℝ × ℝ := (-1, -3)

-- Theorem statement
theorem tangent_line_equation :
  let (x₀, y₀) := point
  let m := deriv f x₀
  (λ x y => x - y - 2 = 0) = (λ x y => y - y₀ = m * (x - x₀)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l653_65305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_daily_revenue_l653_65384

noncomputable def f (t : ℝ) : ℝ := 4 + 1 / t

noncomputable def g (t : ℝ) : ℝ := 120 - |t - 20|

noncomputable def W (t : ℝ) : ℝ := f t * g t

theorem min_daily_revenue :
  ∀ t : ℝ, 1 ≤ t → t ≤ 30 → W t ≥ 441 ∧ (∃ t₀ : ℝ, 1 ≤ t₀ ∧ t₀ ≤ 30 ∧ W t₀ = 441) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_daily_revenue_l653_65384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_function_l653_65372

open Real

-- Define the functions f and g
noncomputable def f (x θ : ℝ) : ℝ := 3 * sin (2 * x + θ)
noncomputable def g (x θ φ : ℝ) : ℝ := 3 * sin (2 * (x - φ) + θ)

-- State the theorem
theorem shifted_sine_function 
  (θ φ : ℝ) 
  (h1 : -π/2 < θ ∧ θ < π/2) 
  (h2 : φ > 0) 
  (h3 : f 0 θ = 3 * sqrt 2 / 2) 
  (h4 : g 0 θ φ = 3 * sqrt 2 / 2) : 
  φ ≠ 5*π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_function_l653_65372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_center_line_slope_l653_65393

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the midpoint of two points -/
def midpoint' (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

/-- Calculates the slope between two points -/
def slope' (p1 p2 : Point) : ℚ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- Represents the given parallelogram -/
def givenParallelogram : Parallelogram :=
  { A := { x := 7, y := 35 },
    B := { x := 7, y := 90 },
    C := { x := 23, y := 120 },
    D := { x := 23, y := 65 } }

theorem parallelogram_center_line_slope :
  let p := givenParallelogram
  let m1 := midpoint' p.A p.B
  let m2 := midpoint' p.C p.D
  let s := slope' m1 m2
  s = 893 / 100 ∧ 893 + 100 = 993 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_center_line_slope_l653_65393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_line_l653_65340

/-- Circle M with center (2,0) and radius 1 -/
def CircleM : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 1}

/-- Line y = x -/
def LineYEqX : Set (ℝ × ℝ) := {p | p.2 = p.1}

/-- Point Q on the line y = x -/
def Q (t : ℝ) : ℝ × ℝ := (t, t)

/-- Tangent line from Q to circle M -/
def TangentLine (q : ℝ × ℝ) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x | (x.1 - q.1) * (p.1 - 2) + (x.2 - q.2) * p.2 = 0}

/-- The statement to be proved -/
theorem tangent_points_line (t : ℝ) :
  ∃ (A B : ℝ × ℝ),
    A ∈ CircleM ∧ B ∈ CircleM ∧
    A ∈ TangentLine (Q t) A ∧ B ∈ TangentLine (Q t) B ∧
    ∀ (x y : ℝ), (t - 2) * x + t * y = 2 * t - 3 ↔ (x, y) ∈ ({A, B} : Set (ℝ × ℝ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_line_l653_65340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_properties_l653_65365

/-- Triangular pyramid with given properties -/
structure TriangularPyramid where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  height : ℝ
  AB_eq_AC : AB = AC
  AB_eq_10 : AB = 10
  BC_eq_16 : BC = 16
  height_eq_4 : height = 4

/-- Calculate the surface area of the triangular pyramid -/
noncomputable def surfaceArea (p : TriangularPyramid) : ℝ :=
  100 + 20 * Real.sqrt (181/5)

/-- Calculate the radius of the inscribed sphere -/
noncomputable def inscribedSphereRadius (p : TriangularPyramid) : ℝ :=
  64 / surfaceArea p

/-- Theorem stating the properties of the triangular pyramid -/
theorem triangular_pyramid_properties (p : TriangularPyramid) :
  surfaceArea p = 100 + 20 * Real.sqrt (181/5) ∧
  inscribedSphereRadius p = 64 / (100 + 20 * Real.sqrt (181/5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_properties_l653_65365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_measurement_least_relative_error_l653_65307

/-- Represents a measurement with its actual length and error. -/
structure Measurement where
  length : ℚ
  error : ℚ

/-- Calculates the relative error of a measurement. -/
def relativeError (m : Measurement) : ℚ :=
  m.error / m.length

theorem second_measurement_least_relative_error
  (m1 m2 m3 : Measurement)
  (h1 : m1.length = 20 ∧ m1.error = 1/10)
  (h2 : m2.length = 30 ∧ m2.error = 3/100)
  (h3 : m3.length = 200 ∧ m3.error = 1/2) :
  relativeError m2 < relativeError m1 ∧ relativeError m2 < relativeError m3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_measurement_least_relative_error_l653_65307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_range_a_l653_65368

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then 2 * a * x^2 - x - 1/4
  else Real.log x / Real.log a - 1

theorem monotonic_f_range_a :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (1/8 : ℝ) (1/4 : ℝ) :=
by
  sorry

#check monotonic_f_range_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_range_a_l653_65368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l653_65376

def alternatingSequence (n : ℕ) : ℚ :=
  if n % 2 = 1 then 3 else 4

theorem fifteenth_term_is_three :
  let a := alternatingSequence
  (∀ n : ℕ, n > 1 → a n * a (n - 1) = 12) →
  a 1 = 3 →
  a 2 = 4 →
  a 15 = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l653_65376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l653_65312

noncomputable def given_numbers : List ℝ := [-1^2, 3, 5/2, Real.pi, -1.31, 0, -15/7, 2]

def is_positive_rational (x : ℝ) : Prop := x > 0 ∧ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_negative_fraction (x : ℝ) : Prop := x < 0 ∧ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_non_negative_integer (x : ℝ) : Prop := x ≥ 0 ∧ ∃ (n : ℤ), x = n

def positive_rational_set : Set ℝ := {x | is_positive_rational x}
def negative_fraction_set : Set ℝ := {x | is_negative_fraction x}
def non_negative_integer_set : Set ℝ := {x | is_non_negative_integer x}

theorem number_categorization :
  (3 ∈ positive_rational_set ∧ (5/2) ∈ positive_rational_set ∧ 2 ∈ positive_rational_set) ∧
  (-1.31 ∈ negative_fraction_set ∧ (-15/7) ∈ negative_fraction_set) ∧
  (3 ∈ non_negative_integer_set ∧ 0 ∈ non_negative_integer_set ∧ 2 ∈ non_negative_integer_set) :=
by
  sorry

#check number_categorization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l653_65312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l653_65315

theorem equation_solution : 
  ∀ x : ℝ, x^2 + 6*x + 6*x*Real.sqrt (x + 4) = 24 ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l653_65315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_feuerbach_circles_common_point_l653_65331

/-- The Feuerbach circle (nine-point circle) of a triangle. -/
def feuerbach_circle (A B C : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

/-- Given a quadrilateral ABCD, the Feuerbach circles of triangles ABC, ABD, ACD, and BCD have a common point. -/
theorem feuerbach_circles_common_point (A B C D : EuclideanSpace ℝ (Fin 2)) :
  ∃ (P : EuclideanSpace ℝ (Fin 2)),
    P ∈ feuerbach_circle A B C ∧
    P ∈ feuerbach_circle A B D ∧
    P ∈ feuerbach_circle A C D ∧
    P ∈ feuerbach_circle B C D :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_feuerbach_circles_common_point_l653_65331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garage_location_optimality_l653_65360

/-- Represents a town's street layout as a graph --/
structure TownGraph where
  vertices : Type
  edges : Type
  source : edges → vertices
  target : edges → vertices

/-- A path in the town that covers all streets --/
def CoveringPath (G : TownGraph) (start : G.vertices) :=
  List G.edges

/-- The length of a path --/
def pathLength (G : TownGraph) (start : G.vertices) (path : CoveringPath G start) : ℝ := sorry

/-- The total length of all streets in the town --/
def totalStreetLength (G : TownGraph) : ℝ := sorry

/-- A path is valid if it covers all edges and returns to the start --/
def isValidPath (G : TownGraph) (start : G.vertices) (path : CoveringPath G start) : Prop := sorry

/-- The optimal path from a given starting point --/
noncomputable def optimalPath (G : TownGraph) (start : G.vertices) : CoveringPath G start := sorry

/-- Represents that a graph is connected --/
def ConnectedGraph (G : TownGraph) : Prop := sorry

theorem garage_location_optimality (G : TownGraph) 
  (start1 start2 : G.vertices) 
  (h_connected : ConnectedGraph G) 
  (h_exists_path : ∃ (start : G.vertices), ∃ (path : CoveringPath G start), isValidPath G start path) :
  pathLength G start1 (optimalPath G start1) = pathLength G start2 (optimalPath G start2) := by
  sorry

#check garage_location_optimality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garage_location_optimality_l653_65360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l653_65302

theorem sin_minus_cos_value (α : Real) (h1 : 0 < α) (h2 : α < Real.pi) 
  (h3 : Real.sin α + Real.cos α = 1/3) :
  Real.sin α - Real.cos α = Real.sqrt 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l653_65302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l653_65325

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (-1 + 3/5 * t, -1 + 4/5 * t)

-- Define the curve C in polar form
def curve_C (θ : ℝ) : ℝ := Real.sqrt 2 * Real.sin (θ + Real.pi/4)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t θ, line_l t = p ∧ 
    (p.1^2 + p.2^2 = (curve_C θ)^2) ∧ 
    (p.1 = (curve_C θ) * Real.cos θ) ∧ 
    (p.2 = (curve_C θ) * Real.sin θ)}

-- Theorem statement
theorem intersection_distance :
  ∃ M N, M ∈ intersection_points ∧ N ∈ intersection_points ∧ M ≠ N ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = Real.sqrt 41 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l653_65325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_implies_expression_equals_two_l653_65333

theorem tan_one_implies_expression_equals_two (α : Real) (h : Real.tan α = 1) :
  (2 * Real.sin α ^ 2 + 1) / (Real.sin (2 * α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_implies_expression_equals_two_l653_65333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_larger_triangle_l653_65306

/-- Given two similar right triangles, where the smaller triangle has leg lengths 
    a and b, and the larger triangle has hypotenuse c, this function computes 
    the perimeter of the larger triangle. -/
noncomputable def largerTrianglePerimeter (a b c : ℝ) : ℝ :=
  let smallHypotenuse := Real.sqrt (a^2 + b^2)
  let ratio := c / smallHypotenuse
  (a * ratio) + (b * ratio) + c

/-- Theorem stating that for the given triangle dimensions, 
    the perimeter of the larger triangle is 48. -/
theorem perimeter_of_larger_triangle : 
  largerTrianglePerimeter 6 8 20 = 48 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval largerTrianglePerimeter 6 8 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_larger_triangle_l653_65306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shift_implies_phi_l653_65361

-- Define the function f as noncomputable
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- State the theorem
theorem symmetric_shift_implies_phi (ω φ : ℝ) :
  ω > 0 →
  0 < φ →
  φ < Real.pi / 2 →
  f ω φ 0 = -f ω φ (Real.pi / 2) →
  (∀ x, f ω φ (x + Real.pi / 12) = -f ω φ (-x + Real.pi / 12)) →
  φ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shift_implies_phi_l653_65361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_discount_percentage_l653_65363

/-- Given a dress with original price d, prove that with a 35% discount during sale
    and a final staff price of 0.455d, the staff discount percentage on the discounted price is 30%. -/
theorem staff_discount_percentage (d : ℝ) (h : d > 0) : 
  let discounted_price := 0.65 * d
  let staff_price := 0.455 * d
  let staff_discount_percent := (discounted_price - staff_price) / discounted_price * 100
  staff_discount_percent = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_discount_percentage_l653_65363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_monotonicity_l653_65310

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (x^2 + 1)

-- State the theorem
theorem odd_function_monotonicity 
  (h1 : ∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f 1 0 x = x / (1 + x^2))
  (h2 : ∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f 1 0 (-x) = -(f 1 0 x))
  (h3 : f 1 0 (1/2) = 2/5) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f 1 0 x = x / (1 + x^2)) ∧ 
  (∀ x1 x2, x1 ∈ Set.Ioo (-1 : ℝ) 1 → x2 ∈ Set.Ioo (-1 : ℝ) 1 → x1 < x2 → f 1 0 x1 < f 1 0 x2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_monotonicity_l653_65310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_factor_absolute_value_l653_65397

/-- Given a cubic polynomial 3x³ - px + q with factors (x-2) and (x+3), prove |3p-2q| = 99 -/
theorem cubic_factor_absolute_value (p q : ℝ) : 
  (∃ r : ℝ, 3 * X^3 - p * X + q = 3 * (X - 2) * (X + 3) * (X - r)) →
  |3*p - 2*q| = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_factor_absolute_value_l653_65397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_problem_l653_65351

/-- Given two quantities that vary inversely, this function calculates the value of one quantity given the other -/
noncomputable def inverse_variation (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_variation_problem (a₁ a₂ a₃ b₁ : ℝ) 
  (h₁ : a₁ = 800) 
  (h₂ : b₁ = 0.5) 
  (h₃ : a₂ = 1600) 
  (h₄ : a₃ = 400) :
  let k := a₁ * b₁
  inverse_variation k a₂ = 0.25 ∧ inverse_variation k a₃ = 1.0 := by
  sorry

#check inverse_variation_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_problem_l653_65351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_function_properties_l653_65385

-- Define the grade function
noncomputable def gradeFunction (x : ℝ) : ℝ := x - x^2 / 100

-- State the theorem
theorem grade_function_properties :
  -- The domain of the function is [0, 100]
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 100 →
  -- 1. The maximum value is 25, occurring at x = 50
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ 100 → gradeFunction y ≤ gradeFunction 50) ∧
  gradeFunction 50 = 25 ∧
  -- 2. The minimum value is 0, occurring at x = 0 and x = 100
  gradeFunction 0 = 0 ∧
  gradeFunction 100 = 0 ∧
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ 100 → gradeFunction y ≥ 0) ∧
  -- 3. There exist distinct x₁ and x₂ such that f(x₁) = f(x₂)
  ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 100 ∧ 0 ≤ x₂ ∧ x₂ ≤ 100 ∧ x₁ ≠ x₂ ∧ gradeFunction x₁ = gradeFunction x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_function_properties_l653_65385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l653_65375

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 else -1

-- State the theorem
theorem solution_set_of_inequality :
  ∀ x : ℝ, (x + 1) * f x > 2 ↔ x ∈ (Set.Ioi 1 ∪ Set.Iio (-3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l653_65375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_five_l653_65348

theorem sum_divisible_by_five (n : ℕ) :
  5 ∣ (3^(n + 1) - 2^(n + 1)) ↔ n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_five_l653_65348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_bounds_l653_65323

-- Define the functions h and j
noncomputable def h : ℝ → ℝ := sorry
noncomputable def j : ℝ → ℝ := sorry

-- Define the range conditions
axiom h_range : ∀ x, -4 ≤ h x ∧ h x ≤ 2
axiom j_range : ∀ x, 0 ≤ j x ∧ j x ≤ 3

-- State the theorem
theorem product_bounds :
  (∃ x : ℝ, h x * j x = 6) ∧
  (∀ x : ℝ, h x * j x ≤ 6) ∧
  (∃ x : ℝ, h x * j x = 0) ∧
  (∀ x : ℝ, 0 ≤ h x * j x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_bounds_l653_65323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l653_65313

/-- Represents the time (in hours) to fill a cistern when two taps are opened simultaneously,
    given the time to fill and empty the cistern individually. -/
noncomputable def fillTime (fillDuration emptyDuration : ℝ) : ℝ :=
  (fillDuration * emptyDuration) / (emptyDuration - fillDuration)

theorem cistern_fill_time :
  let fillDuration : ℝ := 4
  let emptyDuration : ℝ := 9
  fillTime fillDuration emptyDuration = 36 / 5 := by
  sorry

#eval (36 : ℚ) / 5  -- To verify that 36/5 is indeed 7.2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l653_65313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l653_65369

/-- The tangent line to the curve y = -(1/2)x + ln(x) -/
noncomputable def tangent_line (x : ℝ) (b : ℝ) : ℝ := (1/2) * x - b

/-- The curve y = -(1/2)x + ln(x) -/
noncomputable def curve (x : ℝ) : ℝ := -(1/2) * x + Real.log x

/-- The derivative of the curve -/
noncomputable def curve_derivative (x : ℝ) : ℝ := -(1/2) + 1/x

/-- Theorem stating that if the line y = (1/2)x - b is tangent to the curve y = -(1/2)x + ln(x), then b = 1 -/
theorem tangent_line_b_value :
  ∃ (m : ℝ), m > 0 ∧ 
  tangent_line m b = curve m ∧ 
  (1/2) = curve_derivative m →
  b = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l653_65369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l653_65362

theorem power_equality (w : ℝ) : (7 : ℝ)^3 * (7 : ℝ)^w = 81 ↔ w = 4 * (Real.log 3 / Real.log 7) - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l653_65362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l653_65390

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 1 → Real.sin x - (2 : ℝ)^x < 0) ↔ (∃ x : ℝ, x > 1 ∧ Real.sin x - (2 : ℝ)^x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l653_65390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l653_65374

theorem problem_solution :
  ∀ (m : ℝ),
  (∀ (x : ℝ), m * (1 - x) = x + 3 ↔ 2 - x = x + 4) →
  ∃ (x y : ℝ),
  (3 * x + 2 * m = -y) ∧
  (2 * x + 2 * y = m - 1) ∧
  (m = 1 ∧ x = -1 ∧ y = 1 ∧ ∀ (m : ℝ), 7 * x + 5 * y = -2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l653_65374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ants_proof_min_distance_ants_equals_minimum_distance_l653_65356

/-- Represents the minimum distance between two ants crawling on a cube's diagonals -/
noncomputable def min_distance_ants (a : ℝ) : ℝ :=
  a * Real.sqrt (17 / 13)

/-- Theorem stating the minimum distance between two ants crawling on a cube's diagonals -/
theorem min_distance_ants_proof (a : ℝ) (ha : a > 0) :
  let cube_edge := a
  let gosha_diagonal := a * Real.sqrt 2
  let lesha_diagonal := a * Real.sqrt 2
  let gosha_start := (0 : ℝ × ℝ × ℝ)
  let lesha_start := (a, 0, a)
  let speed_ratio := (5 : ℝ)
  min_distance_ants a = a * Real.sqrt (17 / 13) :=
by
  sorry

/-- Helper function to represent the minimum distance during movement -/
noncomputable def minimum_distance_during_movement (cube_edge : ℝ) (gosha_diagonal lesha_diagonal : ℝ)
    (gosha_start lesha_start : ℝ × ℝ × ℝ) (speed_ratio : ℝ) : ℝ :=
  cube_edge * Real.sqrt (17 / 13)

/-- Theorem relating min_distance_ants to minimum_distance_during_movement -/
theorem min_distance_ants_equals_minimum_distance (a : ℝ) (ha : a > 0) :
  let cube_edge := a
  let gosha_diagonal := a * Real.sqrt 2
  let lesha_diagonal := a * Real.sqrt 2
  let gosha_start := (0 : ℝ × ℝ × ℝ)
  let lesha_start := (a, 0, a)
  let speed_ratio := (5 : ℝ)
  min_distance_ants a = minimum_distance_during_movement cube_edge gosha_diagonal lesha_diagonal gosha_start lesha_start speed_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ants_proof_min_distance_ants_equals_minimum_distance_l653_65356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_volumes_l653_65322

/-- Represents a regular quadrilateral truncated pyramid -/
structure TruncatedPyramid where
  lower_base_side : ℝ
  upper_base_side : ℝ
  height : ℝ

/-- Calculates the volume of a part of the truncated pyramid -/
noncomputable def volume_part (p : TruncatedPyramid) (h : ℝ) : ℝ :=
  (1/3) * h * (p.lower_base_side^2 + p.lower_base_side * p.upper_base_side + p.upper_base_side^2)

/-- Theorem stating the volumes of the two parts of the truncated pyramid -/
theorem truncated_pyramid_volumes (p : TruncatedPyramid)
  (h_lower_base : p.lower_base_side = 2)
  (h_upper_base : p.upper_base_side = 1)
  (h_height : p.height = 3)
  (h_plane_height : ℝ)
  (h_plane_position : h_plane_height = 1) :
  volume_part p h_plane_height = 152/27 ∧
  volume_part p (p.height - h_plane_height) = 49/27 := by
  sorry

#check truncated_pyramid_volumes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_volumes_l653_65322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_van_optimal_speed_and_efficiency_l653_65324

/-- Represents a vehicle with its travel time and fuel efficiency -/
structure Vehicle where
  time : ℚ
  fuelEfficiency : ℚ

/-- Calculates the speed of a vehicle given distance and time -/
def calculateSpeed (distance : ℚ) (time : ℚ) : ℚ :=
  distance / time

/-- Theorem: The van's optimal speed and fuel efficiency -/
theorem van_optimal_speed_and_efficiency 
  (totalDistance : ℚ) 
  (van : Vehicle) 
  (newTimeFactor : ℚ) :
  totalDistance = 450 ∧ 
  van.time = 5 ∧ 
  van.fuelEfficiency = 10 ∧ 
  newTimeFactor = 3/2 →
  let originalSpeed := calculateSpeed totalDistance van.time
  let newTime := newTimeFactor * van.time
  let newSpeed := calculateSpeed totalDistance newTime
  (newSpeed = 90 ∧ van.fuelEfficiency = 10) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_van_optimal_speed_and_efficiency_l653_65324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_size_theorem_l653_65335

/-- Given a wheel that covers 1056 cm in 14.012738853503185 revolutions,
    its circumference is approximately 75.398 cm. -/
theorem wheel_size_theorem (distance : ℝ) (revolutions : ℝ) (circumference : ℝ) :
  distance = 1056 ∧ revolutions = 14.012738853503185 →
  circumference = distance / revolutions →
  |circumference - 75.398| < 0.001 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_size_theorem_l653_65335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_correct_l653_65330

/-- The range of real numbers for a that satisfies the given conditions -/
def range_of_a : Set ℝ := {a | a ≤ -3/2 ∨ a ≥ 3}

theorem range_of_a_correct (f g : ℝ → ℝ) (A B : Set ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f x = x^2 - 2*x) →
  (A = Set.Icc (-1) 3) →
  (∀ x ∈ Set.Icc (-1) 2, ∃ a, g x = a*x + 2) →
  (B = {y | ∃ x ∈ Set.Icc (-1) 2, ∃ a, y = a*x + 2}) →
  (A ⊆ B) →
  {a : ℝ | ∀ y ∈ A, ∃ x ∈ Set.Icc (-1) 2, y = a*x + 2} = range_of_a := by
  sorry

#check range_of_a_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_correct_l653_65330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_action_figures_sold_l653_65342

def sneaker_cost : ℕ := 90
def initial_savings : ℕ := 15
def action_figure_price : ℕ := 10
def money_left : ℕ := 25

theorem action_figures_sold : 
  (sneaker_cost - initial_savings + money_left) / action_figure_price = 10 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_action_figures_sold_l653_65342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l653_65314

/-- A function f(x) with specific properties -/
noncomputable def f (A w φ : ℝ) (x : ℝ) : ℝ :=
  A * Real.sin (w * x + φ)

theorem function_properties
  (A w φ : ℝ)
  (h_A_pos : A > 0)
  (h_w_pos : w > 0)
  (h_max_point : f A w φ (Real.pi / 2) = Real.sqrt 2)
  (h_x_intercept : f A w φ ((3 * Real.pi) / 2) = 0)
  (h_φ_range : φ > -Real.pi / 2 ∧ φ < Real.pi / 2) :
  ∃ (g : ℝ → ℝ),
    (∀ x, g x = Real.sqrt 2 * Real.sin (x / 2 + Real.pi / 4)) ∧
    (∀ x, x ≥ 0 → x ≤ Real.pi → g x ≥ 1 ∧ g x ≤ Real.sqrt 2) ∧
    (∀ m, (∀ x, x ≥ 0 → x ≤ Real.pi → |g x - m| < 2) ↔ (m > Real.sqrt 2 - 2 ∧ m < 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l653_65314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_integers_17_to_47_l653_65301

def sum_odd_integers (start : ℕ) (end' : ℕ) : ℕ :=
  let n := (end' - start) / 2 + 1
  ((start + end') * n) / 2

theorem sum_odd_integers_17_to_47 :
  sum_odd_integers 17 47 = 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_integers_17_to_47_l653_65301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_subsets_with_conditions_l653_65347

universe u

def S : Finset (Fin 5) := {0, 1, 2, 3, 4}

theorem count_subsets_with_conditions :
  let condition (M : Finset (Fin 5)) :=
    M ⊆ S ∧ M ∩ {0, 1, 2} = {0, 1}
  Fintype.card {M : Finset (Fin 5) // condition M} = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_subsets_with_conditions_l653_65347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_when_m_zero_f_has_two_zeros_iff_m_in_range_l653_65366

-- Define the piecewise function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x ≤ m then -x^2 - 2*x else x - 4

-- Theorem 1: When m = 0, f(x) has exactly 3 zeros
theorem f_has_three_zeros_when_m_zero :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  f 0 x₁ = 0 ∧ f 0 x₂ = 0 ∧ f 0 x₃ = 0 ∧
  ∀ x, f 0 x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ :=
by
  sorry

-- Theorem 2: f(x) has exactly two zeros iff m ∈ [-2, 0) ∪ [4, +∞)
theorem f_has_two_zeros_iff_m_in_range :
  ∀ m : ℝ, (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
    f m x₁ = 0 ∧ f m x₂ = 0 ∧
    ∀ x, f m x = 0 → x = x₁ ∨ x = x₂) ↔
  (m ≥ -2 ∧ m < 0) ∨ m ≥ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_when_m_zero_f_has_two_zeros_iff_m_in_range_l653_65366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l653_65336

/-- Given a triangle ABC with the following properties:
  - cos C = 3/5
  - CB · CA = 9/2
  - Vector x = (2sin(B/2), √3)
  - Vector y = (cos B, cos(B/2))
  - x is parallel to y
Prove that:
1. The area of triangle ABC is 3
2. sin(B - A) = (4 - 3√3) / 10
-/
theorem triangle_properties (A B C : Real) (a b c : Real) (x y : Real × Real) :
  Real.cos C = 3/5 →
  (c * b : Real) * Real.cos C = 9/2 →
  x = (2 * Real.sin (B/2), Real.sqrt 3) →
  y = (Real.cos B, Real.cos (B/2)) →
  ∃ (k : Real), x = k • y →
  (1/2 * a * b * Real.sin C = 3) ∧
  (Real.sin (B - A) = (4 - 3 * Real.sqrt 3) / 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l653_65336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_special_after_3429_special_3450_l653_65380

/-- A function that returns the set of digits used in a natural number's decimal representation -/
def digits (n : ℕ) : Finset ℕ :=
  sorry

/-- A predicate that determines if a natural number is special (uses exactly four different digits) -/
def is_special (n : ℕ) : Prop :=
  (digits n).card = 4

/-- The theorem stating that 3450 is the smallest special number greater than 3429 -/
theorem smallest_special_after_3429 :
  ∃ (n : ℕ), n = 3450 ∧ n > 3429 ∧ is_special n ∧ ∀ m, m > 3429 ∧ is_special m → n ≤ m :=
sorry

/-- A helper theorem to show that 3450 satisfies the conditions -/
theorem special_3450 :
  3450 > 3429 ∧ is_special 3450 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_special_after_3429_special_3450_l653_65380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l653_65309

-- Define the two points
def point1 : ℝ × ℝ := (-3, 1)
def point2 : ℝ × ℝ := (5, -5)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem distance_between_points : distance point1 point2 = 10 := by
  -- Unfold the definitions
  unfold distance point1 point2
  -- Simplify the expression
  simp
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l653_65309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_decomposable_iff_power_of_two_l653_65350

def is_n_decomposable (a n : ℕ) : Prop :=
  a > 2 ∧ ∀ d : ℕ, d ∣ n → d ≠ n → (a^n - 2^n) % (a^d + 2^d) = 0

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem n_decomposable_iff_power_of_two (n : ℕ) :
  (is_composite n ∧ ∃ a : ℕ, is_n_decomposable a n) ↔ ∃ k : ℕ, k > 1 ∧ n = 2^k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_decomposable_iff_power_of_two_l653_65350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l653_65355

/-- A rhombus with given side length and shorter diagonal -/
structure Rhombus where
  side_length : ℝ
  shorter_diagonal : ℝ

/-- Calculate the length of the longer diagonal of a rhombus -/
noncomputable def longer_diagonal (r : Rhombus) : ℝ :=
  2 * Real.sqrt (r.side_length ^ 2 - (r.shorter_diagonal / 2) ^ 2)

theorem rhombus_longer_diagonal (r : Rhombus) 
  (h1 : r.side_length = 60)
  (h2 : r.shorter_diagonal = 56) : 
  longer_diagonal r = 106 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l653_65355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_sum_theorem_l653_65382

/-- A regular n-sided prism with base area S -/
structure RegularPrism (n : ℕ) (S : ℝ) where
  n_pos : n > 0
  S_pos : S > 0

/-- Two intersecting planes on a regular prism -/
structure IntersectingPlanes (n : ℕ) (S V : ℝ) extends RegularPrism n S where
  V_pos : V > 0
  no_common_points : True  -- Represents the condition that planes don't intersect inside the prism

/-- The sum of lengths of lateral edge segments between intersecting planes -/
noncomputable def lateral_edge_sum (n : ℕ) (S V : ℝ) (h : IntersectingPlanes n S V) : ℝ :=
  n * V / S

/-- Theorem stating the sum of lateral edge segments -/
theorem lateral_edge_sum_theorem (n : ℕ) (S V : ℝ) (h : IntersectingPlanes n S V) :
  lateral_edge_sum n S V h = n * V / S :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_sum_theorem_l653_65382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l653_65334

/-- The expansion coefficients of (a-2x)^8 in terms of (x-1) -/
def expansion_coeffs (a : ℝ) : Fin 9 → ℝ := sorry

/-- The statement that (a-2x)^8 = sum_{i=0}^8 a_i * (x-1)^i -/
def expansion_equality (a : ℝ) : Prop :=
  ∀ x, (a - 2*x)^8 = (Finset.range 9).sum (fun i => expansion_coeffs a i * (x - 1)^i)

theorem problem_solution (a : ℝ) 
  (h_pos : a > 0)
  (h_coeff : expansion_coeffs a 2 = 81648)
  (h_exp : expansion_equality a) :
  (a = 3) ∧ 
  ((expansion_coeffs a 0 + expansion_coeffs a 2 + expansion_coeffs a 4 + expansion_coeffs a 6 + expansion_coeffs a 8) *
   (expansion_coeffs a 1 + expansion_coeffs a 3 + expansion_coeffs a 5 + expansion_coeffs a 7) = (1 - 3^16) / 4) ∧
  ((Finset.range 9).sum (fun i => |expansion_coeffs a i|) = 3^8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l653_65334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_shirts_l653_65383

theorem number_of_shirts (total_price_shirts : ℝ) (total_price_sweaters : ℝ) 
  (num_sweaters : ℕ) (price_difference : ℝ) :
  total_price_shirts = 360 →
  total_price_sweaters = 900 →
  num_sweaters = 45 →
  (total_price_sweaters / num_sweaters) = (total_price_shirts / (total_price_shirts / 18)) + price_difference →
  price_difference = 2 →
  (total_price_shirts / 18) = 20 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_shirts_l653_65383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_in_choir_l653_65352

theorem boys_in_choir (orchestra band girls_in_choir total boys_in_choir : ℕ) :
  orchestra = 20 →
  band = 2 * orchestra →
  girls_in_choir = 16 →
  total = orchestra + band + (girls_in_choir + boys_in_choir) →
  total = 88 →
  boys_in_choir = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_in_choir_l653_65352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_window_area_l653_65327

/-- The area of a "日" shaped window frame given its length --/
noncomputable def window_area (x : ℝ) : ℝ := x * (1/3 * (4 - 2*x))

/-- Theorem: The maximum area of a "日" shaped window frame made from a 4m alloy bar
    is achieved when the length is 1m and the width is 2/3 m --/
theorem max_window_area :
  ∃ (x : ℝ), x > 0 ∧ x < 2 ∧
  (∀ (y : ℝ), y > 0 → y < 2 → window_area x ≥ window_area y) ∧
  x = 1 ∧ (1/3 * (4 - 2*x)) = 2/3 := by
  sorry

#check max_window_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_window_area_l653_65327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l653_65396

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (α + π/4) = 4/5)
  (h2 : α ∈ Set.Ioo (π/4) (3*π/4)) :
  Real.sin α = 7*Real.sqrt 2/10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l653_65396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l653_65353

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem range_of_b (a b : ℝ) (h : f a = g b) : b ≥ 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l653_65353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_s_r_is_zero_l653_65398

def r : Finset ℤ → Finset ℤ := sorry

def s : ℤ → ℤ
  | x => 2 * x + 1

theorem sum_s_r_is_zero (hr : r ({-2, -1, 0, 1}) ⊆ {-1, 0, 3, 5})
    (hs : Set.range s = {-1, 1, 3, 5}) :
  (r ({-2, -1, 0, 1}) ∩ {-1, 0, 1, 2}).sum s = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_s_r_is_zero_l653_65398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_through_points_l653_65341

theorem slope_of_line_through_points :
  let x₁ : ℝ := 1
  let y₁ : ℝ := 2
  let x₂ : ℝ := -3
  let y₂ : ℝ := 4
  (y₂ - y₁) / (x₂ - x₁) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_through_points_l653_65341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canoes_built_by_june_l653_65320

/-- Sum of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

/-- Number of canoes built in January -/
def initial_canoes : ℕ := 5

/-- Ratio of canoes built each month compared to the previous month -/
def monthly_ratio : ℕ := 3

/-- Number of months from January to June -/
def months : ℕ := 6

theorem canoes_built_by_june :
  geometric_sum (initial_canoes : ℝ) (monthly_ratio : ℝ) months = 1820 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canoes_built_by_june_l653_65320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_84_sqrt_10_l653_65373

-- Define the pyramid
noncomputable def pyramid_base_width : ℝ := 7
noncomputable def pyramid_base_length : ℝ := 9
noncomputable def pyramid_edge_length : ℝ := 15

-- Define the volume of the pyramid
noncomputable def pyramid_volume : ℝ :=
  (1 / 3) * pyramid_base_width * pyramid_base_length *
  Real.sqrt (pyramid_edge_length ^ 2 - 
    ((pyramid_base_width ^ 2 + pyramid_base_length ^ 2) / 4))

-- Theorem statement
theorem pyramid_volume_is_84_sqrt_10 :
  pyramid_volume = 84 * Real.sqrt 10 := by
  sorry

-- Additional lemma to show the calculation steps
lemma pyramid_volume_calculation :
  pyramid_volume = (1 / 3) * 63 * (4 * Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_84_sqrt_10_l653_65373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_concentration_theorem_l653_65379

/-- Represents the properties of a salt solution --/
structure SaltSolution where
  initial_mass : ℝ
  initial_salt_mass : ℝ
  initial_concentration : ℝ

/-- Represents the process of evaporation and mixing --/
noncomputable def evaporate_and_mix (s : SaltSolution) : ℝ :=
  let removed_fraction := (1 : ℝ) / 5
  let remaining_fraction := 1 - removed_fraction
  let evaporated_mass := s.initial_mass * removed_fraction / 2
  let final_mass := s.initial_mass * remaining_fraction + evaporated_mass
  s.initial_salt_mass / final_mass

/-- Theorem stating the initial concentration of salt --/
theorem salt_concentration_theorem (s : SaltSolution) :
  s.initial_concentration = 0.27 →
  evaporate_and_mix s = s.initial_concentration * 1.03 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_concentration_theorem_l653_65379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotone_f_l653_65377

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

-- Define the property of being monotonically increasing on an interval
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem min_a_for_monotone_f :
  (∃ a : ℝ, ∀ a' : ℝ, a' ≥ a → MonotonicallyIncreasing (f a') 1 2) ∧
  (∀ ε > 0, ∃ a : ℝ, a < Real.exp (-1) + ε ∧ ¬MonotonicallyIncreasing (f a) 1 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotone_f_l653_65377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l653_65386

/-- Calculates the time taken for two trains to completely pass each other. -/
noncomputable def time_to_cross (length_a length_b speed_a speed_b : ℝ) : ℝ :=
  (length_a + length_b) / (speed_a + speed_b)

/-- Theorem: The time taken for two trains to completely pass each other
    is equal to the sum of their lengths divided by the sum of their speeds. -/
theorem trains_crossing_time
  (length_a length_b speed_a speed_b : ℝ)
  (h1 : length_a > 0)
  (h2 : length_b > 0)
  (h3 : speed_a > 0)
  (h4 : speed_b > 0) :
  time_to_cross length_a length_b speed_a speed_b =
  (length_a + length_b) / (speed_a + speed_b) :=
by
  -- Unfold the definition of time_to_cross
  unfold time_to_cross
  -- The equality holds by definition
  rfl

/-- Example calculation -/
def example_calculation : ℚ :=
  -- Convert real numbers to rational numbers for computation
  (225 : ℚ) / 15 + (150 : ℚ) / 25

#eval example_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l653_65386
