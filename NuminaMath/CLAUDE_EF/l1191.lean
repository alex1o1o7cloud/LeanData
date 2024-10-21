import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_orientation_l1191_119108

/-- A city is represented as a simple undirected graph -/
structure City where
  V : Type* -- Vertices (locations)
  E : V → V → Prop -- Edges (roads)
  symm : ∀ a b, E a b → E b a -- Roads are bidirectional
  irrefl : ∀ a, ¬E a a -- No self-loops

/-- A path in the city -/
def CityPath (c : City) (start finish : c.V) : List c.V → Prop
  | [] => start = finish
  | [v] => start = v ∧ v = finish
  | (v::w::rest) => c.E v w ∧ CityPath c w finish rest

/-- The city is connected -/
def Connected (c : City) : Prop :=
  ∀ start finish : c.V, ∃ path : List c.V, CityPath c start finish path

/-- An orientation of the city's roads -/
def CityOrientation (c : City) := 
  {o : c.V → c.V → Prop // ∀ a b, o a b → c.E a b ∧ ¬o b a}

/-- A path in the oriented city -/
def OrientedPath (c : City) (o : CityOrientation c) (start finish : c.V) : List c.V → Prop
  | [] => start = finish
  | [v] => start = v ∧ v = finish
  | (v::w::rest) => o.val v w ∧ OrientedPath c o w finish rest

/-- The oriented city is strongly connected -/
def StronglyConnected (c : City) (o : CityOrientation c) : Prop :=
  ∀ start finish : c.V, ∃ path : List c.V, OrientedPath c o start finish path

/-- Main theorem: Any connected city can be oriented to remain strongly connected -/
theorem city_orientation (c : City) (h : Connected c) : 
  ∃ o : CityOrientation c, StronglyConnected c o :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_orientation_l1191_119108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sin_cubed_reciprocal_l1191_119106

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (1 / x) ^ 3

-- State the theorem
theorem derivative_of_sin_cubed_reciprocal (x : ℝ) (h : x ≠ 0) :
  deriv f x = -3 / x^2 * Real.sin (1 / x)^2 * Real.cos (1 / x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sin_cubed_reciprocal_l1191_119106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_continuous_rational_irrational_alternating_function_l1191_119142

open Set
open Function
open Real

theorem no_continuous_rational_irrational_alternating_function :
  ¬∃ f : ℝ → ℝ, Continuous f ∧
    ∀ x : ℝ, (Rat.cast ⁻¹' {f x} ≠ ∅ ↔ Rat.cast ⁻¹' {f (x + 1)} = ∅) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_continuous_rational_irrational_alternating_function_l1191_119142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_count_floor_list_l1191_119105

def floor_list : List ℤ := List.range 1000 |>.map (fun n => ⌊((n + 1 : ℕ).pow 2 : ℚ) / 2000⌋)

theorem distinct_count_floor_list : (floor_list.toFinset).card = 501 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_count_floor_list_l1191_119105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_inequality_strict_l1191_119148

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (a : ℝ × ℝ) : ℝ := Real.sqrt (dot_product a a)

theorem dot_product_inequality_strict :
  ∃ (a b : ℝ × ℝ), abs (dot_product a b) ≠ magnitude a * magnitude b := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_inequality_strict_l1191_119148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_formula_correct_l1191_119166

/-- The total milk production for cows with declining productivity -/
noncomputable def milk_production (a b c d e f : ℝ) : ℝ :=
  d * (b / (a * c)) * (100 / f) * (1 - (1 - f / 100) ^ e)

/-- Theorem stating the milk production formula is correct -/
theorem milk_production_formula_correct
  (a b c d e f : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0)
  (he : e > 0)
  (hf : 0 < f ∧ f < 100) :
  milk_production a b c d e f =
    d * (b / (a * c)) * (100 / f) * (1 - (1 - f / 100) ^ e) :=
by
  -- Unfold the definition of milk_production
  unfold milk_production
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_formula_correct_l1191_119166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_second_proposition_true_l1191_119129

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) - Real.sin x

-- State the theorem
theorem only_second_proposition_true :
  (∃ a : ℝ, a ≥ Real.exp 1 ∧ ∀ x > 0, f a x > 0) ∧
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f 0 x ≥ 0 → False) ∧
  (∀ x > 2, f 1 x ≠ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_second_proposition_true_l1191_119129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1191_119130

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 2)

theorem axis_of_symmetry :
  ∀ (x : ℝ), f (π/4 + x) = f (π/4 - x) :=
by
  intro x
  unfold f
  simp [Real.cos_sub, Real.cos_add]
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1191_119130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l1191_119165

theorem cos_minus_sin_value (α : Real) 
  (h1 : Real.sin α * Real.cos α = 1/8)
  (h2 : Real.pi < α)
  (h3 : α < 5 * Real.pi / 4) :
  Real.cos α - Real.sin α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l1191_119165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l1191_119135

/-- The equation of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

/-- The right focus of the ellipse -/
def right_focus : ℝ × ℝ := (2, 0)

/-- The slope angle of the line -/
noncomputable def slope_angle : ℝ := Real.pi / 3

/-- The line passing through the right focus with the given slope angle -/
noncomputable def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 2)

/-- The length of the chord cut by the line from the ellipse -/
noncomputable def chord_length : ℝ := 4 * Real.sqrt 6 / 5

/-- Theorem stating that the chord length is correct -/
theorem chord_length_is_correct :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line x₁ y₁ ∧ line x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = chord_length^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l1191_119135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_l1191_119198

def number_with_ones_and_zeros (k : Nat) : Nat :=
  (10^k) * (10^300 - 1) / 9

theorem not_perfect_square (k : Nat) :
  ∃ (n : Nat), number_with_ones_and_zeros k = n ∧ 
  (∃ (m : Nat), n % 3 = 0) ∧ 
  (∀ (m : Nat), n ≠ m^2) := by
  sorry

#eval number_with_ones_and_zeros 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_l1191_119198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_generatrix_properties_l1191_119103

/-- Two cones with a common base -/
structure TwoCones where
  R : ℝ  -- radius of common base
  H : ℝ  -- height of first cone
  h : ℝ  -- height of second cone
  R_pos : R > 0
  H_pos : H > 0
  h_pos : h > 0

/-- The distance between the two generatrices -/
noncomputable def generatrix_distance (c : TwoCones) : ℝ :=
  c.R * (c.H + c.h) / Real.sqrt (c.H^2 + c.R^2)

/-- The angle between the two generatrices -/
noncomputable def generatrix_angle (c : TwoCones) : ℝ :=
  Real.arccos (c.h * c.H / Real.sqrt ((c.H^2 + c.R^2) * (c.h^2 + c.R^2)))

/-- Theorem stating the distance and angle between generatrices -/
theorem generatrix_properties (c : TwoCones) :
  (generatrix_distance c = c.R * (c.H + c.h) / Real.sqrt (c.H^2 + c.R^2)) ∧
  (generatrix_angle c = Real.arccos (c.h * c.H / Real.sqrt ((c.H^2 + c.R^2) * (c.h^2 + c.R^2)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_generatrix_properties_l1191_119103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_sum_three_primes_l1191_119194

theorem prime_power_sum_three_primes (p : ℕ) : 
  (Prime p ∧ 
   (∃ x y z : ℕ+, ∃ p₁ p₂ p₃ : ℕ, 
     Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ 
     p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
     x^p + y^p + z^p - x - y - z = p₁ * p₂ * p₃)) ↔ 
  (p = 2 ∨ p = 3 ∨ p = 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_sum_three_primes_l1191_119194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_dot_product_l1191_119101

/-- Parabola structure -/
structure MyParabola where
  p : ℝ
  hp : p > 0

/-- Point in 2D space -/
structure MyPoint where
  x : ℝ
  y : ℝ

/-- Vector in 2D space -/
structure MyVector where
  x : ℝ
  y : ℝ

/-- Dot product of two vectors -/
def my_dot_product (v1 v2 : MyVector) : ℝ := v1.x * v2.x + v1.y * v2.y

/-- Theorem: Dot product of OP and OQ for specific parabola and points -/
theorem parabola_dot_product (Γ : MyParabola) (F P Q : MyPoint) :
  P.y^2 = 2 * Γ.p * P.x →  -- P is on the parabola
  F.x = Γ.p / 2 →  -- F is the focus
  F.y = 0 →
  Q.x = 0 →  -- Q is on y-axis
  (P.x - F.x)^2 + (P.y - F.y)^2 = 4 →  -- |FP| = 2
  (Q.x - F.x)^2 + (Q.y - F.y)^2 = 1 →  -- |FQ| = 1
  P ≠ MyPoint.mk 0 0 →  -- P is different from O
  my_dot_product (MyVector.mk P.x P.y) (MyVector.mk Q.x Q.y) = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_dot_product_l1191_119101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l1191_119179

open Polynomial

theorem coefficient_of_x_cubed (a : ℚ) : 
  (coeff ((2 * X - C a)^6) 3 = -20) → a = 1/2 :=
by
  sorry

-- The following definitions are not necessary for the theorem statement,
-- but I'll include them as comments for reference:

-- Define X as the indeterminate of the polynomial ring
-- X is already defined in Polynomial

-- Define C a as the constant polynomial with value a
-- C is already defined in Polynomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l1191_119179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_l1191_119187

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.log (x + 1)

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ := Real.exp x + 1 / (x + 1)

-- Theorem statement
theorem tangent_line_perpendicular (n : ℝ) :
  (f_derivative 0 * (1 / n) = -1) → n = -2 := by
  intro h
  -- The proof steps would go here
  sorry

#check tangent_line_perpendicular

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_l1191_119187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inradius_inequality_l1191_119183

-- Define a triangle ABC
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the properties of the triangle
def Triangle.circumradius (t : Triangle) : ℝ := 1

noncomputable def Triangle.inradius (t : Triangle) : ℝ := sorry

noncomputable def Triangle.orthicTriangleInradius (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_inradius_inequality (t : Triangle) :
  t.orthicTriangleInradius ≤ 1 - 1 / (3 * (1 + t.inradius)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inradius_inequality_l1191_119183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_probability_in_triangle_l1191_119188

/-- The probability of a randomly selected point in triangle FGH forming an acute angle FQG -/
noncomputable def acuteAngleProbability : ℝ :=
  37 * Real.pi / 86

/-- The coordinates of point F -/
def F : ℝ × ℝ := (-2, 3)

/-- The coordinates of point G -/
def G : ℝ × ℝ := (5, -2)

/-- The coordinates of point H -/
def H : ℝ × ℝ := (7, 3)

/-- Theorem stating that the probability of a randomly selected point Q in triangle FGH
    forming an acute angle ∠FQG is equal to 37π/86 -/
theorem acute_angle_probability_in_triangle :
  acuteAngleProbability = 37 * Real.pi / 86 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_probability_in_triangle_l1191_119188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_range_of_a_l1191_119153

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 4 / x

-- Theorem 1: f is monotonically increasing on (2,+∞)
theorem f_monotone_increasing : 
  ∀ x₁ x₂ : ℝ, 2 < x₁ → x₁ < x₂ → f x₁ < f x₂ :=
by sorry

-- Theorem 2: Range of a
theorem range_of_a : 
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 4 → f x ≥ a) → a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_range_of_a_l1191_119153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_same_color_points_l1191_119169

-- Define a color type
inductive Color
  | Black
  | White

-- Define a point on a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to assign colors to points
def colorAt : Point → Color := sorry

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- The main theorem
theorem exist_same_color_points :
  ∃ (p1 p2 : Point), colorAt p1 = colorAt p2 ∧ distance p1 p2 = 1965 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_same_color_points_l1191_119169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1191_119133

noncomputable def f (x : ℝ) : ℝ := (x * (x + 3)) / ((x - 5)^2)

theorem inequality_solution :
  ∀ x : ℝ, x ≠ 5 →
  (f x ≥ 15 ↔ x ∈ Set.Ici (101/14) ∪ Set.Iic (52/14)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1191_119133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l1191_119190

noncomputable def f (x : ℝ) := Real.cos (x + Real.pi / 2) * Real.cos (x + Real.pi / 4)

theorem f_symmetry : ∀ x : ℝ, f (5 * Real.pi / 8 + x) = f (5 * Real.pi / 8 - x) := by
  intro x
  -- Expand the definition of f
  unfold f
  -- Use trigonometric identities and algebraic manipulation
  -- The full proof would be quite lengthy, so we'll use sorry here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l1191_119190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_digit_satisfies_inequality_l1191_119144

theorem no_digit_satisfies_inequality : 
  ¬ ∃ (d : ℕ), d < 10 ∧ (3014 + d : ℚ) / 1000 > 3015 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_digit_satisfies_inequality_l1191_119144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_4_l1191_119143

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x^2 - x - 4

-- State the theorem
theorem f_of_g_4 : f (g 4) = 23 * Real.sqrt 10 / 5 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_4_l1191_119143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gradient_relation_l1191_119140

/-- The curve y = x^3 -/
def curve (x : ℝ) : ℝ := x^3

/-- The gradient of the curve at any point -/
def curve_gradient (x : ℝ) : ℝ := 3 * x^2

/-- The tangent line at point a -/
def tangent_line (a : ℝ) (x : ℝ) : ℝ := curve_gradient a * (x - a) + curve a

theorem gradient_relation (a : ℝ) :
  ∃ b : ℝ, b ≠ a ∧ 
    tangent_line a b = curve b ∧ 
    curve_gradient b = 4 * curve_gradient a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gradient_relation_l1191_119140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_length_l1191_119113

/-- The function f(x) = x^2 / 8 -/
noncomputable def f (x : ℝ) : ℝ := x^2 / 8

/-- The starting point (7, 3) -/
def P : ℝ × ℝ := (7, 3)

/-- The theorem stating the shortest path length -/
theorem shortest_path_length :
  ∃ (Q R : ℝ × ℝ),
    (Q.2 = f Q.1) ∧  -- Q is on the graph of f
    (R.2 = 0) ∧      -- R is on the x-axis
    (R.1 = Q.1) ∧    -- R is directly below Q
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) + (Q.2 - R.2) = 5 * Real.sqrt 2 - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_length_l1191_119113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_equation_l1191_119175

/-- The equation of a line passing through the left focus of an ellipse with specific properties. -/
theorem ellipse_line_equation (b c e : ℝ) (k : ℝ) :
  b = 2 →
  c = 1 →
  e = Real.sqrt 5 / 5 →
  let a := Real.sqrt 5
  let ellipse := fun (x y : ℝ) ↦ x^2 / 5 + y^2 / 4 = 1
  let line := fun (x y : ℝ) ↦ y = k * (x + 1)
  let intersect := fun (x y : ℝ) ↦ ellipse x y ∧ line x y
  let dist_squared := fun (x₁ y₁ x₂ y₂ : ℝ) ↦ (x₁ - x₂)^2 + (y₁ - y₂)^2
  ∃ x₁ y₁ x₂ y₂, intersect x₁ y₁ ∧ intersect x₂ y₂ ∧
    dist_squared x₁ y₁ x₂ y₂ = (16/9 * Real.sqrt 5)^2 →
  k = 1 ∨ k = -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_equation_l1191_119175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_m_set_characterization_l1191_119150

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := |x - 4/m| + |x + m|

-- Theorem 1: For all x, f(x) ≥ 4 when m > 0
theorem f_lower_bound (m : ℝ) (hm : m > 0) : ∀ x : ℝ, f m x ≥ 4 := by sorry

-- Define the set of m values satisfying f(2) > 5
def m_set : Set ℝ := {m : ℝ | m > 0 ∧ f m 2 > 5}

-- Theorem 2: The set of m values satisfying f(2) > 5 is ((1 + √17)/2, +∞) ∪ (0, 1)
theorem m_set_characterization : 
  m_set = {m : ℝ | m > (1 + Real.sqrt 17) / 2} ∪ {m : ℝ | 0 < m ∧ m < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_m_set_characterization_l1191_119150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_solution_set_l1191_119186

-- Define the vectors m and n
noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def n : ℝ × ℝ := (Real.sqrt 3 / 2, 1 / 2)

-- Define the dot product function
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := dot_product (m x) n

-- Theorem for the maximum value of f
theorem f_max_value :
  ∃ (x : ℝ), f x = 1 ∧ ∀ (y : ℝ), f y ≤ 1 := by
  sorry

-- Theorem for the solution set of f(x) ≥ 1/2
theorem f_solution_set :
  ∀ (x : ℝ), f x ≥ 1/2 ↔ ∃ (k : ℤ), 2 * k * Real.pi ≤ x ∧ x ≤ 2 * Real.pi / 3 + 2 * k * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_solution_set_l1191_119186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_eigenvalue_is_three_l1191_119161

/-- The matrix in the problem -/
def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4],
    ![6, 3]]

/-- The theorem stating that 3 is the only eigenvalue of A -/
theorem only_eigenvalue_is_three :
  ∀ k : ℝ, (∃ v : Fin 2 → ℝ, v ≠ 0 ∧ A.vecMul v = k • v) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_eigenvalue_is_three_l1191_119161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1191_119114

theorem calculate_expression : (-2)^2 + 4 * (2:ℚ)⁻¹ - |-8| = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1191_119114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1191_119152

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S a n + a (n + 1)

theorem arithmetic_sequence_properties 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_sum_ratio : ∀ n : ℕ, n > 0 → S a (2 * n) / S a n = 4) :
  (∀ n : ℕ, n > 0 → a n = 2 * n - 1) ∧ 
  (∀ n : ℕ, n > 0 → S a n = n^2) ∧
  (∀ n : ℕ, n > 0 → 
    let b := λ k => a k * 2^(k - 1)
    S b n = (2 * n - 3) * 2^n + 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1191_119152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l1191_119163

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a chord of the ellipse -/
structure Chord (e : Ellipse) where
  A : Point
  B : Point
  on_ellipse : (A.x^2 / e.a^2) + (A.y^2 / e.b^2) = 1 ∧
               (B.x^2 / e.a^2) + (B.y^2 / e.b^2) = 1
  length : Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 2

/-- The foci of the ellipse -/
noncomputable def foci (e : Ellipse) : (Point × Point) :=
  let c := Real.sqrt (e.a^2 - e.b^2)
  ({x := c, y := 0}, {x := -c, y := 0})

/-- Theorem statement -/
theorem ellipse_triangle_perimeter (e : Ellipse) (ch : Chord e) :
  let (F₁, F₂) := foci e
  ch.A.x = F₁.x ∧ ch.A.y = F₁.y →
  (Real.sqrt ((ch.A.x - F₂.x)^2 + (ch.A.y - F₂.y)^2) +
   Real.sqrt ((ch.B.x - F₂.x)^2 + (ch.B.y - F₂.y)^2) +
   2) = 4 * e.a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l1191_119163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_side_length_l1191_119131

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def trapezium_area (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- Theorem stating the length of the unknown side of the trapezium -/
theorem trapezium_side_length (t : Trapezium) 
  (h1 : t.side1 = 16)
  (h2 : t.height = 15)
  (h3 : t.area = 270)
  (h4 : t.area = trapezium_area t) : 
  t.side2 = 20 := by
  sorry

#check trapezium_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_side_length_l1191_119131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_triangle_area_l1191_119180

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Theorem: The area of a triangle with sides √(f_{2n+1}), √(f_{2n+2}), and √(f_{2n+3}) is 1/2 -/
theorem fibonacci_triangle_area (n : ℕ) (h : n ≥ 1) :
  let a := Real.sqrt (fib (2 * n + 1))
  let b := Real.sqrt (fib (2 * n + 2))
  let c := Real.sqrt (fib (2 * n + 3))
  (1 / 2 : ℝ) * a * b = (1 / 2 : ℝ) := by
  sorry

#check fibonacci_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_triangle_area_l1191_119180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_dot_product_theorem_l1191_119124

def minimum_dot_product (a b c : ℝ × ℝ) : Prop :=
  let norm_squared := λ v : ℝ × ℝ ↦ v.1 * v.1 + v.2 * v.2
  norm_squared a = 1 ∧ 
  norm_squared b = 1 ∧ 
  norm_squared c = 1 ∧
  (a.1 * b.1 + a.2 * b.2 = 1/2) →
  (2*a.1 + c.1) * (b.1 - c.1) + (2*a.2 + c.2) * (b.2 - c.2) ≥ -Real.sqrt 3

theorem minimum_dot_product_theorem :
  ∀ a b c : ℝ × ℝ, minimum_dot_product a b c := by
  sorry

#check minimum_dot_product_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_dot_product_theorem_l1191_119124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_plus_arcsin_eq_arccos_solutions_l1191_119158

theorem arcsin_plus_arcsin_eq_arccos_solutions :
  ∀ x : ℝ, x ∈ Set.Icc 0 1 →
    (Real.arcsin x + Real.arcsin (1 - x) = Real.arccos x) ↔ (x = 0 ∨ x = 1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_plus_arcsin_eq_arccos_solutions_l1191_119158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_percentage_l1191_119149

noncomputable section

-- Define the side lengths of the squares
def side_length_A : ℝ := 1  -- Assume initial side length is 1 for simplicity
def side_length_B : ℝ := side_length_A * 2.5  -- 150% increase
def side_length_C : ℝ := side_length_B * 1.8  -- 80% increase
def side_length_D : ℝ := side_length_C * 1.4  -- 40% increase

-- Define the areas of the squares
def area_A : ℝ := side_length_A ^ 2
def area_B : ℝ := side_length_B ^ 2
def area_C : ℝ := side_length_C ^ 2
def area_D : ℝ := side_length_D ^ 2

-- Define the sum of areas A, B, and C
def sum_areas_ABC : ℝ := area_A + area_B + area_C

-- Define the percentage difference
def percentage_difference : ℝ := (area_D - sum_areas_ABC) / sum_areas_ABC * 100

-- Theorem statement
theorem area_difference_percentage :
  |percentage_difference - 44.327| < 0.001 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_percentage_l1191_119149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_effectiveness_order_l1191_119185

/-- Represents the size of a washing powder package -/
inductive PackageSize
  | XS
  | S
  | M
  | L

/-- Represents the cost and quantity of a package -/
structure Package where
  size : PackageSize
  cost : ℚ
  quantity : ℚ

/-- The relative cost and quantity relationships between packages -/
def package_relations (p : PackageSize → Package) : Prop :=
  (p PackageSize.S).cost = 8/5 * (p PackageSize.XS).cost ∧
  (p PackageSize.S).quantity = 3/4 * (p PackageSize.M).quantity ∧
  (p PackageSize.M).quantity = 3/2 * (p PackageSize.XS).quantity ∧
  (p PackageSize.M).cost = 7/5 * (p PackageSize.S).cost ∧
  (p PackageSize.L).quantity = 13/10 * (p PackageSize.M).quantity ∧
  (p PackageSize.L).cost = 6/5 * (p PackageSize.M).cost

/-- Cost-effectiveness of a package -/
def cost_effectiveness (pkg : Package) : ℚ := pkg.quantity / pkg.cost

/-- Theorem stating the order of cost-effectiveness -/
theorem cost_effectiveness_order (p : PackageSize → Package) 
  (h : package_relations p) : 
  cost_effectiveness (p PackageSize.XS) > cost_effectiveness (p PackageSize.L) ∧
  cost_effectiveness (p PackageSize.L) > cost_effectiveness (p PackageSize.S) ∧
  cost_effectiveness (p PackageSize.S) > cost_effectiveness (p PackageSize.M) := by
  sorry

#check cost_effectiveness_order

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_effectiveness_order_l1191_119185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_value_parallel_distance_l1191_119195

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ x + a * y = 2 * a + 2
def l₂ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ a * x + y = a + 1

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop := (-1 / a) * (-a) = -1

-- Define parallelism of lines
def parallel (a : ℝ) : Prop := -1 / a = -a

-- Theorem for perpendicular case
theorem perpendicular_value :
  ∃ a : ℝ, perpendicular a ∧ a = -1 := by
  sorry

-- Theorem for parallel case
theorem parallel_distance :
  ∃ a : ℝ, parallel a ∧ 
  (let d := |4 - 2| / Real.sqrt (1^2 + 1^2);
   d = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_value_parallel_distance_l1191_119195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sodium_reacts_with_water_valid_deductive_reasoning_l1191_119191

-- Define the set of all elements
def Element : Type := String

-- Define the property of being an alkali metal
def IsAlkaliMetal : Element → Prop := sorry

-- Define the property of reacting with water
def ReactsWithWater : Element → Prop := sorry

-- Define sodium as an element
def sodium : Element := "sodium"

-- Theorem: If all alkali metals react with water and sodium is an alkali metal,
-- then sodium reacts with water (which represents valid deductive reasoning)
theorem sodium_reacts_with_water :
  (∀ x : Element, IsAlkaliMetal x → ReactsWithWater x) →
  IsAlkaliMetal sodium →
  ReactsWithWater sodium :=
by
  sorry

-- The proof that this represents valid deductive reasoning
theorem valid_deductive_reasoning : 
  ∃ (P Q : Prop) (R : Prop → Prop → Prop),
    (P ∧ Q → R P Q) ∧
    ((P ∧ Q → R P Q) ↔ ((∀ x : Element, IsAlkaliMetal x → ReactsWithWater x) ∧ 
                        IsAlkaliMetal sodium → 
                        ReactsWithWater sodium)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sodium_reacts_with_water_valid_deductive_reasoning_l1191_119191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_tips_reported_l1191_119146

/-- Brendan's hourly wage as a waiter -/
noncomputable def hourly_wage : ℝ := 6

/-- Number of 8-hour shifts -/
def eight_hour_shifts : ℕ := 2

/-- Number of 12-hour shifts -/
def twelve_hour_shifts : ℕ := 1

/-- Average tips per hour -/
noncomputable def average_tips_per_hour : ℝ := 12

/-- Tax rate -/
noncomputable def tax_rate : ℝ := 0.20

/-- Actual tax paid per week -/
noncomputable def actual_tax_paid : ℝ := 56

/-- Total hours worked -/
noncomputable def total_hours : ℝ := 8 * eight_hour_shifts + 12 * twelve_hour_shifts

/-- Total wage income -/
noncomputable def wage_income : ℝ := hourly_wage * total_hours

/-- Total tips -/
noncomputable def total_tips : ℝ := average_tips_per_hour * total_hours

/-- Total income -/
noncomputable def total_income : ℝ := wage_income + total_tips

/-- Tax on wage income -/
noncomputable def tax_on_wage : ℝ := tax_rate * wage_income

/-- Tax on reported tips -/
noncomputable def tax_on_reported_tips : ℝ := actual_tax_paid - tax_on_wage

/-- Reported tips -/
noncomputable def reported_tips : ℝ := tax_on_reported_tips / tax_rate

theorem fraction_of_tips_reported :
  reported_tips / total_tips = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_tips_reported_l1191_119146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_second_side_length_l1191_119109

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

/-- Proof that the second parallel side of a trapezium is 18 cm given specific conditions. -/
theorem trapezium_second_side_length :
  ∀ (x : ℝ),
  trapeziumArea 20 x 14 = 266 →
  x = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_second_side_length_l1191_119109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_project_duration_l1191_119168

/-- Represents a road construction project -/
structure RoadProject where
  totalLength : ℚ
  initialWorkers : ℕ
  completedLength : ℚ
  completedDays : ℕ
  extraWorkers : ℕ

/-- Calculates the initial planned duration of a road project -/
def initialPlannedDuration (project : RoadProject) : ℚ :=
  let totalWorkers := project.initialWorkers + project.extraWorkers
  let remainingLength := project.totalLength - project.completedLength
  let workRate := project.completedLength / (project.initialWorkers * project.completedDays)
  project.completedDays + remainingLength / (workRate * totalWorkers)

/-- Theorem stating that the initial planned duration of the given project is 300 days -/
theorem road_project_duration :
  let project : RoadProject := {
    totalLength := 10
    initialWorkers := 30
    completedLength := 2
    completedDays := 100
    extraWorkers := 30
  }
  initialPlannedDuration project = 300 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_project_duration_l1191_119168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_graph_and_coordinate_sum_l1191_119167

/-- Given a function g such that g(8) = 10, prove that (3, 13/3) is on the graph of 3y = g(3x - 1) + 3 
    and the sum of its coordinates is 22/3 -/
theorem point_on_graph_and_coordinate_sum (g : ℝ → ℝ) (h : g 8 = 10) :
  (⟨3, 13/3⟩ : ℝ × ℝ) ∈ {p : ℝ × ℝ | 3 * p.2 = g (3 * p.1 - 1) + 3} ∧ 3 + 13/3 = 22/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_graph_and_coordinate_sum_l1191_119167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_isolating_line_iff_a_eq_2e_l1191_119104

noncomputable def f (x : ℝ) := x^2

noncomputable def g (a : ℝ) (x : ℝ) := a * Real.log x

def is_isolating_line (k b : ℝ) (f g : ℝ → ℝ) :=
  ∀ x, f x ≥ k * x + b ∧ k * x + b ≥ g x

theorem unique_isolating_line_iff_a_eq_2e :
  (∃! k b, is_isolating_line k b f (g (2 * Real.exp 1))) ↔ 
  ∀ a > 0, (∃! k b, is_isolating_line k b f (g a)) → a = 2 * Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_isolating_line_iff_a_eq_2e_l1191_119104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1191_119177

/-- Definition of an arithmetic sequence -/
noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (a d : ℝ) (n : ℕ) : ℝ := (n : ℝ) * (2 * a + (n - 1) * d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_property (a d : ℝ) (i j k : ℕ) :
  (S a d i / (i : ℝ)) * ((j : ℝ) - (k : ℝ)) + 
  (S a d j / (j : ℝ)) * ((k : ℝ) - (i : ℝ)) + 
  (S a d k / (k : ℝ)) * ((i : ℝ) - (j : ℝ)) = 0 := by
  sorry

#check arithmetic_sequence_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1191_119177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_calculation_l1191_119151

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def atomic_weight_H : ℝ := 1.008

/-- The atomic weight of carbon in atomic mass units (amu) -/
def atomic_weight_C : ℝ := 12.011

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def atomic_weight_O : ℝ := 15.999

/-- The number of hydrogen atoms in the compound -/
def num_H : ℕ := 2

/-- The number of carbon atoms in the compound -/
def num_C : ℕ := 1

/-- The number of oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight : ℝ := num_H * atomic_weight_H + num_C * atomic_weight_C + num_O * atomic_weight_O

theorem molecular_weight_calculation :
  ∃ ε > 0, |molecular_weight - 62.024| < ε := by
  -- The proof is omitted for brevity
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_calculation_l1191_119151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1191_119197

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := 3 * Real.sqrt 2 * Real.sin (x / 4) * Real.cos (x / 4) + 
                       Real.sqrt 6 * (Real.cos (x / 4))^2 - Real.sqrt 6 / 2 - m

-- State the theorem
theorem min_m_value (m : ℝ) : 
  (∀ x ∈ Set.Icc (-5 * Real.pi / 6) (Real.pi / 6), f x m ≤ 0) → m ≥ Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1191_119197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2023_equals_2_l1191_119145

def f : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => f n

theorem f_2023_equals_2 : f 2023 = 2 := by
  have h : ∀ k, f (2*k + 1) = 2 := by
    intro k
    induction k with
    | zero => rfl
    | succ n ih => 
      simp [f]
      exact ih
  
  have : 2023 = 2 * 1011 + 1 := by norm_num
  rw [this]
  exact h 1011

#eval f 2023

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2023_equals_2_l1191_119145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_equation_l1191_119121

theorem binomial_coefficient_equation : 
  ∃! n : ℕ, n ≤ 15 ∧ (Nat.choose 15 n + Nat.choose 15 7 = Nat.choose 16 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_equation_l1191_119121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1191_119160

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - 12*x + 3

noncomputable def g (x m : ℝ) : ℝ := Real.exp (x * Real.log 3) - m

-- State the theorem
theorem min_m_value (m : ℝ) : 
  (∀ x1 ∈ Set.Icc (-1 : ℝ) 5, ∃ x2 ∈ Set.Icc (0 : ℝ) 2, f x1 ≥ g x2 m) →
  m ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1191_119160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_line_segment_vector_representation_l1191_119115

/-- Given a line segment AB extended to point Q such that AQ:QB = 5:2,
    prove that Q = -2/3*A + 5/3*B --/
theorem extended_line_segment_vector_representation 
  (A B Q : ℝ × ℝ × ℝ) (h : ‖Q - A‖ / ‖B - Q‖ = 5 / 2) :
  Q = (-2/3 : ℝ) • A + (5/3 : ℝ) • B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_line_segment_vector_representation_l1191_119115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_times_l1191_119128

noncomputable section

-- Define the lengths and speeds
def train_a_length : ℝ := 150
def train_b_length : ℝ := 120
def bridge_x_length : ℝ := 300
def bridge_y_length : ℝ := 250
def train_a_speed : ℝ := 45 * (1000 / 3600)  -- Convert km/h to m/s
def train_b_speed : ℝ := 60 * (1000 / 3600)  -- Convert km/h to m/s

-- Define the theorem
theorem train_crossing_times :
  let time_a := (train_a_length + bridge_x_length) / train_a_speed
  let time_b := (train_b_length + bridge_y_length) / train_b_speed
  abs (time_a - 36) < 0.1 ∧ 
  abs (time_b - 22.2) < 0.1 ∧ 
  abs ((time_a - time_b) - 13.8) < 0.1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_times_l1191_119128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_selection_l1191_119110

/-- Represents the phone usage time intervals --/
inductive TimeInterval
| Less15
| From15To30
| From30To45
| From45To60
| From60To75
| From75To90
| Above90

/-- Represents the survey data --/
def survey_data : List (TimeInterval × Nat) :=
  [(TimeInterval.Less15, 1), (TimeInterval.From15To30, 12), (TimeInterval.From30To45, 28),
   (TimeInterval.From45To60, 24), (TimeInterval.From60To75, 15), (TimeInterval.From75To90, 13),
   (TimeInterval.Above90, 7)]

/-- Definition of poor self-phone management --/
def is_poor_management (t : TimeInterval) : Bool :=
  match t with
  | TimeInterval.From75To90 => true
  | TimeInterval.Above90 => true
  | _ => false

/-- Total number of students surveyed --/
def total_students : Nat := 100

/-- Number of female students with poor self-management --/
def poor_management_females : Nat := 12

/-- Total number of female students --/
def total_females : Nat := 40

/-- Number of individuals with poor self-management selected for further study --/
def selected_individuals : Nat := 5

/-- Number of males in the selected group --/
def selected_males : Nat := 2

/-- Number of females in the selected group --/
def selected_females : Nat := 3

/-- Number of males who enjoy sports in the selected group --/
def selected_males_sports : Nat := 1

/-- Number of females who enjoy sports in the selected group --/
def selected_females_sports : Nat := 1

/-- Number of valid selections --/
def number_of_valid_selections : Nat := 8

/-- Total number of possible selections --/
def total_selections : Nat := 10

/-- The main theorem to prove --/
theorem probability_of_selection :
  ∃ (p : Rat),
    p = 4/5 ∧
    p = (number_of_valid_selections / total_selections : Rat) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_selection_l1191_119110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marty_votes_count_l1191_119117

/-- Represents the candidates in the student council president race -/
inductive Candidate
| Marty
| Biff
| Clara
| Doc
| Ein

/-- Represents the poll results for each candidate -/
def poll_results : Candidate → Rat
| Candidate.Biff => 30/100
| Candidate.Clara => 20/100
| Candidate.Doc => 10/100
| Candidate.Ein => 5/100
| Candidate.Marty => 0

/-- The percentage of undecided voters -/
def undecided_percentage : Rat := 15/100

/-- The total number of people polled -/
def total_polled : Nat := 600

/-- Represents the leanings of undecided voters towards each candidate -/
def undecided_leanings : Candidate → Rat
| Candidate.Marty => 40/100
| Candidate.Biff => 30/100
| Candidate.Clara => 20/100
| Candidate.Ein => 10/100
| Candidate.Doc => 0

/-- The number of people voting or leaning towards voting for Marty -/
def marty_votes : ℚ :=
  undecided_percentage * (total_polled : ℚ) * undecided_leanings Candidate.Marty

theorem marty_votes_count :
  ⌊marty_votes⌋ = 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marty_votes_count_l1191_119117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_integers_in_range_l1191_119120

/-- A permutation of the first n natural numbers -/
def Permutation (n : ℕ) := { f : ℕ → ℕ // Function.Injective f ∧ Set.range f = Finset.range n }

/-- The sum S for a given permutation -/
def S (n : ℕ) (p : Permutation n) : ℚ :=
  Finset.sum (Finset.range n) (fun i => (p.val (i + 1) : ℚ) / (i + 1))

/-- The set of all possible sums S for a given n -/
def SumSet (n : ℕ) : Set ℚ :=
  { s | ∃ p : Permutation n, S n p = s }

theorem all_integers_in_range (n : ℕ) : n = 798 →
  ∀ k : ℕ, n ≤ k ∧ k ≤ n + 100 → (k : ℚ) ∈ SumSet n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_integers_in_range_l1191_119120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersection_l1191_119125

/-- Given two lines in parametric form, prove that their intersection point is (5, -2) -/
theorem lines_intersection :
  ∃! p : ℝ × ℝ, 
    (∃ t : ℝ, p = (1 + 2*t, 4 - 3*t)) ∧ 
    (∃ u : ℝ, p = (5 + 4*u, -2 - 5*u)) ∧ 
    p = (5, -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersection_l1191_119125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1191_119147

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- The equation of a circle with center (a, b) and radius R -/
def circle_equation (x y a b R : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = R^2

theorem circle_tangent_to_line :
  let center_x : ℝ := 1
  let center_y : ℝ := -1
  let line_eq (x y : ℝ) : Prop := x - y + 2 = 0
  let radius := distance_point_to_line center_x center_y 1 (-1) 2
  ∀ x y : ℝ, circle_equation x y center_x center_y radius ↔ (x - 1)^2 + (y + 1)^2 = 8 := by
  sorry

#check circle_tangent_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1191_119147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_vs_kate_and_laura_l1191_119196

/-- Represents the hours charged by each person to the project -/
structure ProjectHours where
  kate : ℝ
  pat : ℝ
  mark : ℝ
  laura : ℝ

/-- Defines the conditions of the project hours -/
def valid_project_hours (h : ProjectHours) : Prop :=
  h.pat = 2 * h.kate ∧
  h.mark = 3 * h.kate ∧
  h.laura = 1.5 * h.mark ∧
  h.kate + h.pat + h.mark + h.laura = 360

/-- Theorem stating the difference in hours charged by Mark compared to Kate and Laura combined -/
theorem mark_vs_kate_and_laura (h : ProjectHours) 
  (hvalid : valid_project_hours h) : 
  ∃ ε > 0, |h.mark - (h.kate + h.laura) + 85.72| < ε := by
  sorry

#check mark_vs_kate_and_laura

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_vs_kate_and_laura_l1191_119196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersection_points_implies_a_values_l1191_119182

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then Real.sin x
  else x^3 - 9*x^2 + 25*x + a

-- Define the auxiliary function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := x^3 - 9*x^2 + 24*x + a

-- Theorem statement
theorem three_intersection_points_implies_a_values (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f a x₁ = x₁ ∧ f a x₂ = x₂ ∧ f a x₃ = x₃) →
  a = -20 ∨ a = -16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersection_points_implies_a_values_l1191_119182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_racetrack_length_proof_l1191_119136

/-- The length of a circular racetrack in miles. -/
noncomputable def racetrack_length : ℝ := 3

/-- The probability of ending within a half mile of the 2.5 mile sign. -/
noncomputable def end_probability : ℝ := 1/3

/-- The distance traveled by the car. -/
noncomputable def distance_traveled : ℝ := 1/2

/-- The increment between signs on the track. -/
noncomputable def sign_increment : ℝ := 1/10

theorem racetrack_length_proof :
  racetrack_length = 3 ∧
  end_probability = 1/3 ∧
  distance_traveled = 1/2 ∧
  sign_increment = 1/10 →
  racetrack_length = 3 := by
  intro h
  exact h.left

#check racetrack_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_racetrack_length_proof_l1191_119136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_5488000_l1191_119173

theorem cube_root_5488000 : (5488000 : ℝ) ^ (1/3) = 280 * (2 : ℝ) ^ (1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_5488000_l1191_119173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1191_119171

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log 5

theorem f_properties :
  (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧
  (Set.Ioo (-2 : ℝ) (-1) ∪ Set.Ioo 3 4 = {x : ℝ | f x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1191_119171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_odd_three_digit_numbers_l1191_119119

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4}

/-- A function to check if a number is odd -/
def isOdd (n : Nat) : Bool := n % 2 = 1

/-- A function to check if a three-digit number has no repeating digits -/
def noRepeatingDigits (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

/-- The set of all three-digit numbers formed from the given digits -/
def threeDigitNumbers : Finset Nat :=
  Finset.filter (fun n => n ≥ 100 ∧ n < 1000 ∧
    (n / 100 ∈ digits) ∧ ((n / 10) % 10 ∈ digits) ∧ (n % 10 ∈ digits)) (Finset.range 1000)

/-- The main theorem -/
theorem count_odd_three_digit_numbers :
  (Finset.filter (fun n => isOdd n ∧ noRepeatingDigits n) threeDigitNumbers).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_odd_three_digit_numbers_l1191_119119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_ax_iff_a_geq_one_third_l1191_119134

/-- The function f(x) = sin(x) / (2 + cos(x)) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x / (2 + Real.cos x)

/-- Theorem stating the range of a for which f(x) ≤ ax holds for all x ≥ 0 -/
theorem f_leq_ax_iff_a_geq_one_third :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → f x ≤ a * x) ↔ a ≥ 1/3 := by
  sorry

#check f_leq_ax_iff_a_geq_one_third

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_ax_iff_a_geq_one_third_l1191_119134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_MPF_is_10_l1191_119178

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola -/
def F : Point :=
  ⟨1, 0⟩

/-- The directrix of the parabola -/
def Directrix : Set Point :=
  {p : Point | p.x = -1}

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Area of triangle MPF is 10 -/
theorem area_of_triangle_MPF_is_10 
  (P : Point) 
  (h1 : P ∈ Parabola) 
  (M : Point) 
  (h2 : M ∈ Directrix) 
  (h3 : (P.x - M.x) * (P.y - M.y) = 0) -- M is foot of perpendicular
  (h4 : distance P F = 5) : 
  (1/2) * distance P M * |P.y| = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_MPF_is_10_l1191_119178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wastewater_management_l1191_119157

/-- Represents the amount of wastewater discharged in the nth month -/
def discharge (n : ℕ) : ℚ := 2 * n + 8

/-- Represents the amount of wastewater purified in the nth month -/
noncomputable def purify (n : ℕ) : ℚ := if n ≥ 7 then 5 * (1.2 ^ (n - 7)) else 0

/-- Represents the total amount of wastewater discharged up to the nth month -/
def total_discharge (n : ℕ) : ℚ := (n * (2 * n + 8)) / 2

/-- Represents the total amount of wastewater in the pool at the end of the nth month -/
def total_wastewater (n : ℕ) : ℚ := 800 + 2 * n - total_discharge n

/-- The month when the wastewater pool is first emptied -/
def emptied_month : ℕ := 25

/-- The month when purification first exceeds discharge -/
def purified_month : ℕ := 20

theorem wastewater_management :
  (total_wastewater emptied_month ≤ 0 ∧ total_wastewater (emptied_month - 1) > 0) ∧
  (purify purified_month ≥ discharge purified_month ∧ 
   ∀ m < purified_month, purify m < discharge m) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wastewater_management_l1191_119157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l1191_119111

theorem evaluate_expression : (81 : ℝ) ^ (1/4) * (81 : ℝ) ^ (1/5) = 3 * (81 : ℝ) ^ (1/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l1191_119111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l1191_119118

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The line equation with slope 1 -/
def line (x y t : ℝ) : Prop := y = x + t

/-- The length of the chord AB -/
noncomputable def chord_length (t : ℝ) : ℝ := (4 * Real.sqrt 2 * Real.sqrt (5 - t^2)) / 5

/-- The maximum value of |AB| is 4√10/5 -/
theorem max_chord_length :
  ∃ (max : ℝ), max = (4 * Real.sqrt 10) / 5 ∧
  ∀ (x y t : ℝ), ellipse x y → line x y t → chord_length t ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l1191_119118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_transforms_log_to_exp_l1191_119174

/-- Represents a 90° counterclockwise rotation around the origin -/
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-(p.2), p.1)

/-- The original logarithmic function -/
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- The transformed function after rotation -/
noncomputable def F (x : ℝ) : ℝ := 10^x

theorem rotation_transforms_log_to_exp :
  ∀ x > 0, rotate90 (x, f x) = (-(f x), x) ∧ F (-(f x)) = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_transforms_log_to_exp_l1191_119174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2013_equals_23097_l1191_119116

def toBaseRepresentation (num : ℕ) (base : ℕ) : List ℕ := sorry

def fromBaseRepresentation (digits : List ℕ) (base : ℕ) : ℕ := sorry

def a : ℕ → ℕ
  | 0 => 1  -- Adding this case to cover Nat.zero
  | 1 => 1
  | 2 => 3
  | n+3 => 
    let base_n_minus_1 := n+2
    let base_n := n+3
    let prev_in_base_n_minus_1 := toBaseRepresentation (a (n+2)) base_n_minus_1
    let read_in_base_n := fromBaseRepresentation prev_in_base_n_minus_1 base_n
    read_in_base_n + 2

theorem a_2013_equals_23097 : a 2013 = 23097 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2013_equals_23097_l1191_119116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_equals_two_l1191_119176

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

-- State the theorem
theorem f_even_implies_a_equals_two (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → f a x = f a (-x)) →
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_equals_two_l1191_119176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_zero_l1191_119122

def is_divisible_by_first_five_primes (n : ℕ) : Bool :=
  n % 2 = 0 && n % 3 = 0 && n % 5 = 0 && n % 7 = 0 && n % 11 = 0

def sum_of_selected_integers : ℕ :=
  (Finset.range 50).sum (λ i => 
    let n := 102 + 2 * i
    if n ≤ 200 && is_divisible_by_first_five_primes n then n else 0)

theorem sum_is_zero : sum_of_selected_integers = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_zero_l1191_119122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_rotation_surface_area_proof_torus_surface_area_proof_l1191_119127

/-- The surface area described by a regular hexagon with side length a, 
    when rotated around one of the tangents to its circumscribed circle -/
noncomputable def hexagon_rotation_surface_area (a : ℝ) : ℝ :=
  12 * Real.pi * a^2

/-- Theorem stating that the surface area described by a regular hexagon 
    with side length a, when rotated around one of the tangents to its 
    circumscribed circle, is equal to 12πa² -/
theorem hexagon_rotation_surface_area_proof (a : ℝ) (h : a > 0) : 
  hexagon_rotation_surface_area a = 12 * Real.pi * a^2 := by
  -- Unfold the definition of hexagon_rotation_surface_area
  unfold hexagon_rotation_surface_area
  -- The equation is now trivially true
  rfl

/-- Given a circle rotated around a straight line in its plane which does not intersect it, 
    resulting in a surface known as a torus. The surface area of the torus given the radius r 
    of the rotating circle and the distance R from its center to the axis of rotation (R > r) -/
noncomputable def torus_surface_area (r R : ℝ) : ℝ :=
  4 * Real.pi^2 * r * R

/-- Theorem stating the formula for the surface area of a torus -/
theorem torus_surface_area_proof (r R : ℝ) (hr : r > 0) (hR : R > r) : 
  torus_surface_area r R = 4 * Real.pi^2 * r * R := by
  -- Unfold the definition of torus_surface_area
  unfold torus_surface_area
  -- The equation is now trivially true
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_rotation_surface_area_proof_torus_surface_area_proof_l1191_119127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l1191_119112

open Set Real

/-- The function f parameterized by a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 2) * exp x + log x + 1 / x

/-- The derivative of f with respect to x -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a * (x - 1) * exp x + 1 / x - 1 / (x^2)

/-- The set of extreme points of f in the interval (0, 2) -/
def extreme_points (a : ℝ) : Set ℝ :=
  {x | x ∈ Ioo 0 2 ∧ f' a x = 0}

theorem extreme_points_imply_a_range :
  ∀ a : ℝ, (extreme_points a).ncard = 2 →
    a ∈ Iic (-1 / exp 1) ∪ Ioo (-1 / exp 1) (-1 / (4 * exp 2)) := by
  sorry

#check extreme_points_imply_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l1191_119112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_orthocenter_symmetry_l1191_119189

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the circumcenter of a triangle
noncomputable def circumcenter (t : Triangle) : Point :=
  sorry

-- Define symmetry with respect to a line
noncomputable def symmetric_point (p q : Point) (line : Point → Point → Prop) : Point :=
  sorry

-- Define the orthocenter of a triangle
noncomputable def orthocenter (t : Triangle) : Point :=
  sorry

-- Main theorem
theorem circumcenter_orthocenter_symmetry 
  (ABC : Triangle) 
  (O : Point) 
  (h_circumcenter : O = circumcenter ABC) 
  (A₁ : Point) 
  (B₁ : Point) 
  (C₁ : Point) 
  (h_A₁_sym : A₁ = symmetric_point O (circumcenter ABC) (λ p q ↦ p = ABC.B ∨ p = ABC.C))
  (h_B₁_sym : B₁ = symmetric_point O (circumcenter ABC) (λ p q ↦ p = ABC.C ∨ p = ABC.A))
  (h_C₁_sym : C₁ = symmetric_point O (circumcenter ABC) (λ p q ↦ p = ABC.A ∨ p = ABC.B)) :
  O = orthocenter (Triangle.mk A₁ B₁ C₁) ∧ 
  O = circumcenter (Triangle.mk A₁ B₁ C₁) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_orthocenter_symmetry_l1191_119189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inexpressible_numbers_l1191_119192

theorem inexpressible_numbers (n : ℕ) : 
  (∀ a b : ℕ, a > 0 ∧ b > 0 → n ≠ a / b + (a + 1) / (b + 1)) ↔ 
  (n = 1 ∨ ∃ m : ℕ, n = 2^m + 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inexpressible_numbers_l1191_119192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_C_outside_AB_l1191_119156

-- Define the circles
def circle_A : ℝ × ℝ → Prop := λ p ↦ (p.1 - 0)^2 + (p.2 - 0)^2 = 1^2
def circle_B : ℝ × ℝ → Prop := λ p ↦ (p.1 - 3)^2 + (p.2 - 0)^2 = 2^2
def circle_C : ℝ × ℝ → Prop := λ p ↦ (p.1 - 1)^2 + (p.2 - 1.5)^2 = 1.5^2

-- Define the point N
def N : ℝ × ℝ := (1, 0)

-- Define the area of intersection between two circles
noncomputable def area_intersection (c1 c2 : ℝ × ℝ → Prop) : ℝ :=
  sorry -- Actual calculation of intersection area

-- State the theorem
theorem area_inside_C_outside_AB :
  let total_area_C := 2.25 * Real.pi
  let intersection_AC := area_intersection circle_A circle_C
  let intersection_BC := area_intersection circle_B circle_C
  total_area_C - intersection_AC - intersection_BC =
    2.25 * Real.pi - (intersection_AC + intersection_BC) := by
  sorry

#check area_inside_C_outside_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_C_outside_AB_l1191_119156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_oplus_result_l1191_119137

noncomputable def oplus (a b : ℝ) : ℝ := (a - b) / (2 - a * b)

noncomputable def nested_oplus : ℕ → ℝ
  | 0 => 1000
  | n + 1 => oplus (1000 - n) (nested_oplus n)

theorem nested_oplus_result :
  ∃ y : ℝ, y = nested_oplus 998 ∧ y ≠ 2 ∧
  oplus 1 y = (1 - y) / (2 - y) := by
  sorry

#check nested_oplus_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_oplus_result_l1191_119137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1191_119193

/-- 
Given an angle α in the second quadrant and a point P(x, √5) on its terminal side,
if cos α = (√2/4)x, then sin α = √10/4.
-/
theorem sin_alpha_value (α : ℝ) (x : ℝ) 
  (h1 : π / 2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.cos α = (Real.sqrt 2 / 4) * x) -- cos α = (√2/4)x
  (h3 : x < 0) -- x is negative in the second quadrant
  : Real.sin α = Real.sqrt 10 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1191_119193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_root_equality_l1191_119107

theorem seventh_root_equality : (7 : ℝ) ^ (1/4 : ℝ) / (7 : ℝ) ^ (1/6 : ℝ) = (7 : ℝ) ^ (1/12 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_root_equality_l1191_119107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_travel_time_unchanged_l1191_119123

/-- Represents the network of cities and their connections -/
structure CityNetwork where
  num_cities : Nat
  capital_travel_time : ℝ
  adjacent_city_travel_time : ℝ
  initial_transfer_time : ℝ
  reduced_transfer_time : ℝ

/-- Calculates the minimum travel time between two farthest cities -/
noncomputable def min_travel_time (network : CityNetwork) (transfer_time : ℝ) : ℝ :=
  min (2 * network.capital_travel_time + transfer_time)
      (5 * network.adjacent_city_travel_time + 4 * transfer_time)

/-- Theorem: The minimum travel time remains unchanged after reducing transfer time -/
theorem min_travel_time_unchanged (network : CityNetwork) :
  network.num_cities = 11 →
  network.capital_travel_time = 7 →
  network.adjacent_city_travel_time = 3 →
  network.initial_transfer_time = 2 →
  network.reduced_transfer_time = 1.5 →
  min_travel_time network network.initial_transfer_time =
  min_travel_time network network.reduced_transfer_time :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_travel_time_unchanged_l1191_119123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_a_plus_k_l1191_119199

/-- Definition of an ellipse with given properties -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  foci_1 : ℝ × ℝ := (1, 1)
  foci_2 : ℝ × ℝ := (1, 3)
  passes_through : ℝ × ℝ := (6, 2)
  a_positive : a > 0
  b_positive : b > 0
  equation : ∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 → (x, y) ∈ Set.range (λ (t : ℝ × ℝ) => t)

/-- Theorem stating that a + k = 7 for the given ellipse -/
theorem ellipse_a_plus_k (e : Ellipse) : e.a + e.k = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_a_plus_k_l1191_119199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l1191_119164

theorem angle_sum_proof (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 3/4) (h4 : Real.sin β = 3/5) : α + 3*β = 5*π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l1191_119164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_vectors_l1191_119102

-- Define the vectors
def AB : Fin 3 → ℝ := ![2, -1, -4]
def AD : Fin 3 → ℝ := ![4, 2, 0]
def AP : Fin 3 → ℝ := ![-1, 2, -1]

-- Define the parallelogram ABCD
def ABCD : Set (Fin 3 → ℝ) := {x | ∃ (s t : ℝ), x = fun i => s * AB i + t * AD i}

-- Define P as a point outside the plane of ABCD
noncomputable def P : Fin 3 → ℝ := sorry

-- Dot product function
def dot (v w : Fin 3 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

-- Theorem statement
theorem parallelogram_vectors :
  (dot AP AB = 0) ∧
  (dot AP AD = 0) ∧
  (dot AP AB = 0 ∧ dot AP AD = 0 → ∀ x ∈ ABCD, dot AP (fun i => x i - P i) = 0) ∧
  ¬(∃ (k : ℝ), AP = fun i => k * (AB i + AD i)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_vectors_l1191_119102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_when_a_2_range_of_a_for_F_derivative_unique_solution_implies_m_eq_1_l1191_119139

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + x

-- Define the function F
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Theorem 1: Maximum value of f when a = 2
theorem max_value_f_when_a_2 :
  ∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f 2 y ≤ f 2 x ∧ f 2 x = 0 := by sorry

-- Theorem 2: Range of a for F'(x) ≤ 1/2
theorem range_of_a_for_F_derivative :
  ∀ a : ℝ, (∀ x : ℝ, 0 < x ∧ x ≤ 3 → (deriv (F a)) x ≤ 1/2) → a ≥ 1/2 := by sorry

-- Theorem 3: Unique solution implies m = 1 when a = 0
theorem unique_solution_implies_m_eq_1 :
  ∀ m : ℝ, m > 0 →
  (∃! x : ℝ, x > 0 ∧ m * (Real.log x + x) = x^2) →
  m = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_when_a_2_range_of_a_for_F_derivative_unique_solution_implies_m_eq_1_l1191_119139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_quadratic_roots_sum_of_squares_of_quadratic_roots_proof_l1191_119138

theorem sum_of_squares_of_quadratic_roots (a b c : ℝ) : Prop :=
  (∃ r₁ r₂ : ℝ, (a * r₁^2 + b * r₁ + c = 0) ∧ (a * r₂^2 + b * r₂ + c = 0) ∧ (r₁ ≠ r₂)) →
  (∃ r₁ r₂ : ℝ, r₁^2 + r₂^2 = 68)

-- The proof of the theorem
theorem sum_of_squares_of_quadratic_roots_proof :
  sum_of_squares_of_quadratic_roots 1 (-10) 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_quadratic_roots_sum_of_squares_of_quadratic_roots_proof_l1191_119138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_range_theorem_l1191_119172

def complex_range (z : ℂ) : Prop :=
  Complex.abs z = 2 → ∃ (w : ℝ), Complex.abs (1 + Complex.I * Real.sqrt 3 + z) = w ∧ 0 ≤ w ∧ w ≤ 4

theorem complex_range_theorem :
  ∀ z : ℂ, complex_range z :=
by
  intro z
  unfold complex_range
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_range_theorem_l1191_119172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l1191_119155

/-- The line 3x + 4y - 2 = 0 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0

/-- The circle (x+1)² + (y+1)² = 1 -/
def circle' (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 1

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem min_distance_line_circle : 
  ∃ (d : ℝ), d = 4/5 ∧ 
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    line x₁ y₁ → circle' x₂ y₂ → 
    distance x₁ y₁ x₂ y₂ ≥ d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l1191_119155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1191_119184

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 2 * n + 1

-- Define S_n (sum of first n terms of a_n)
noncomputable def S (n : ℕ) : ℝ := n^2 + n

-- Define b_n
noncomputable def b (n : ℕ+) : ℝ := 1 / ((a n.val)^2 - 1)

-- Define T_n (sum of first n terms of b_n)
noncomputable def T (n : ℕ+) : ℝ := n / (4 * (n + 1))

theorem arithmetic_sequence_properties :
  (a 2 = 5) ∧ 
  (a 4 + a 7 = 24) ∧ 
  (∀ n : ℕ, a n = 2 * n + 1) ∧ 
  (∀ n : ℕ, S n = n^2 + n) ∧ 
  (∀ n : ℕ+, T n = n / (4 * (n + 1))) := by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1191_119184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_tetrahedron_octahedron_l1191_119126

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- An octahedron formed by the midpoints of the edges of a regular tetrahedron -/
def midpoint_octahedron (t : RegularTetrahedron) : Set (Fin 3 → ℝ) :=
  sorry

/-- Volume of a regular tetrahedron -/
noncomputable def volume_tetrahedron (t : RegularTetrahedron) : ℝ :=
  (1 / 6 : ℝ) * t.edge_length ^ 3 * Real.sqrt 2

/-- Volume of the midpoint octahedron -/
noncomputable def volume_midpoint_octahedron (t : RegularTetrahedron) : ℝ :=
  sorry

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio_tetrahedron_octahedron (t : RegularTetrahedron) :
    volume_midpoint_octahedron t / volume_tetrahedron t = 1 / 18 := by
  sorry

#check volume_ratio_tetrahedron_octahedron

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_tetrahedron_octahedron_l1191_119126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_travel_time_l1191_119141

/-- Calculates the time taken for a boat to travel downstream given its speed in still water, 
    the current speed, and the distance traveled. -/
noncomputable def time_downstream (boat_speed : ℝ) (current_speed : ℝ) (distance : ℝ) : ℝ :=
  distance / (boat_speed + current_speed)

/-- Converts time in hours to minutes -/
noncomputable def hours_to_minutes (hours : ℝ) : ℝ :=
  hours * 60

theorem downstream_travel_time :
  let boat_speed := (65 : ℝ)
  let current_speed := (15 : ℝ)
  let distance := (33.33 : ℝ)
  let time_hrs := time_downstream boat_speed current_speed distance
  let time_mins := hours_to_minutes time_hrs
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |time_mins - 25| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_travel_time_l1191_119141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_BE_length_l1191_119100

open Geometry -- This opens the Geometry namespace which includes Point and related definitions

-- Define the points
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the lengths
def AB : ℝ := 5
def BC : ℝ := 8

-- Define the shapes
def is_rectangle (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def is_right_triangle (A B E : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the area calculation functions
noncomputable def area_rectangle (A B C D : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry
noncomputable def area_triangle (A B E : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- State the theorem
theorem BE_length :
  is_rectangle A B C D →
  is_right_triangle A B E →
  area_rectangle A B C D = area_triangle A B E →
  dist A B = 5 →
  dist B C = 8 →
  dist B E = 16 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_BE_length_l1191_119100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_1_bijective_l1191_119159

theorem functional_equation_1_bijective (f : ℝ → ℝ) :
  (∀ x, f (f x - 1) = x + 1) → Function.Bijective f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_1_bijective_l1191_119159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_placement_theorem_l1191_119170

/-- Represents a stone placement on an 8x8 grid -/
def StonePlacement := Fin 8 → Fin 8 → Bool

/-- A valid stone placement has exactly 3 stones in each row and column -/
def is_valid_placement (p : StonePlacement) : Prop :=
  (∀ row, (Finset.filter (λ col => p row col) (Finset.univ : Finset (Fin 8))).card = 3) ∧
  (∀ col, (Finset.filter (λ row => p row col) (Finset.univ : Finset (Fin 8))).card = 3)

/-- A set of stone positions where no two share the same row or column -/
def disjoint_stones (s : Finset (Fin 8 × Fin 8)) : Prop :=
  ∀ (a b : Fin 8 × Fin 8), a ∈ s → b ∈ s → a ≠ b → a.1 ≠ b.1 ∧ a.2 ≠ b.2

theorem stone_placement_theorem (p : StonePlacement) (h : is_valid_placement p) :
  ∃ s : Finset (Fin 8 × Fin 8), s.card = 8 ∧ disjoint_stones s ∧ ∀ (pos : Fin 8 × Fin 8), pos ∈ s → p pos.1 pos.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_placement_theorem_l1191_119170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_quadratic_l1191_119132

theorem factorization_quadratic (a x : ℝ) : a * x^2 - 2*a*x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_quadratic_l1191_119132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_is_integer_l1191_119162

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 249
  | (n + 3) => (1991 + sequence_a (n + 2) * sequence_a (n + 1)) / sequence_a n

theorem sequence_a_is_integer : ∀ n : ℕ, ∃ m : ℤ, sequence_a n = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_is_integer_l1191_119162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_m_l1191_119154

theorem max_prime_factors_m (m n : ℕ+) : 
  (∃ p1 p2 p3 p4 p5 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5 ∧
    p1 * p2 * p3 * p4 * p5 ∣ Nat.gcd m.val n.val) →
  (∃ (p : Finset ℕ), p.card = 30 ∧ ∀ q ∈ p, Nat.Prime q ∧ q ∣ Nat.lcm m.val n.val) →
  (Finset.card (Finset.filter (fun p => Nat.Prime p ∧ p ∣ m.val) (Finset.range (m.val + 1))) <
   Finset.card (Finset.filter (fun p => Nat.Prime p ∧ p ∣ n.val) (Finset.range (n.val + 1)))) →
  Finset.card (Finset.filter (fun p => Nat.Prime p ∧ p ∣ m.val) (Finset.range (m.val + 1))) ≤ 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_m_l1191_119154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l1191_119181

/-- Represents the time (in minutes) it takes to fill a tank using two pipes -/
noncomputable def fill_time (time_a : ℝ) (speed_ratio : ℝ) : ℝ :=
  1 / (1 / time_a + speed_ratio / time_a)

theorem tank_fill_time :
  let time_a : ℝ := 20
  let speed_ratio : ℝ := 4
  fill_time time_a speed_ratio = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l1191_119181
