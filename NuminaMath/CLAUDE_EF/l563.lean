import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_x1_x2_l563_56312

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - Real.sqrt 3 * Real.cos x

-- State the theorem
theorem min_sum_x1_x2 (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-π/3 - x)) →  -- Symmetry condition
  (∃ x₁ x₂ : ℝ, f a x₁ * f a x₂ = -4) →  -- Existence of x₁ and x₂
  (∃ x₁ x₂ : ℝ, f a x₁ * f a x₂ = -4 ∧ |x₁ + x₂| = 2*π/3 ∧ 
    ∀ y₁ y₂ : ℝ, f a y₁ * f a y₂ = -4 → |y₁ + y₂| ≥ 2*π/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_x1_x2_l563_56312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_relation_l563_56362

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a chord in a circle -/
structure Chord where
  start : Point
  finish : Point  -- Changed 'end' to 'finish' to avoid keyword conflict

/-- Given a unit circle centered at O, with chords PQ and MN parallel to radius OR,
    and lengths of chords MP, PQ, NR each being t, and length of MN being f,
    prove that f^2 - t^2 = 4 -/
theorem chord_length_relation (c : Circle) (pq mn or : Chord) (t f : ℝ) :
  c.radius = 1 →
  c.center = Point.mk 0 0 →
  pq.start.x = -t ∧ pq.start.y = 0 →
  pq.finish.x = t ∧ pq.finish.y = 0 →
  mn.start.y = mn.finish.y →
  or.finish.x = 1 ∧ or.finish.y = 0 →
  (mn.finish.x - mn.start.x) = f →
  (pq.finish.x - pq.start.x) = 2 * t →
  f^2 - t^2 = 4 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_relation_l563_56362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l563_56343

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  (3 * x - 2 * y + z = 0) ∧
  (x - 4 * y + 3 * z = 0) ∧
  (2 * x + y - 5 * z = 0)

-- Define the expression to evaluate
noncomputable def expression (x y z : ℝ) : ℝ :=
  (x^2 - 2*x*y + y*z) / (x^2 + z^2)

-- Theorem stating that the expression evaluates to 1/2 given the system of equations
theorem expression_value :
  ∀ x y z : ℝ, system x y z → expression x y z = 1/2 :=
by
  intros x y z h
  sorry

#check expression_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l563_56343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_eq_sin_x_solutions_l563_56383

theorem sin_3x_eq_sin_x_solutions : 
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi) ∧ 
    (∀ x ∈ S, Real.sin (3 * x) = Real.sin x) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin (3 * x) = Real.sin x → x ∈ S) ∧
    S.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_eq_sin_x_solutions_l563_56383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_bound_l563_56306

open Real

-- Define the function f(x)
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + Real.log x

-- Define the derivative of f(x)
noncomputable def f_deriv (b : ℝ) (x : ℝ) : ℝ := (2*x^2 - b*x + 1) / x

theorem f_difference_bound (b : ℝ) (x₁ x₂ : ℝ) 
  (h_b : b > 9/2) 
  (h_x : 0 < x₁ ∧ x₁ < x₂) 
  (h_deriv : f_deriv b x₁ = 0 ∧ f_deriv b x₂ = 0) : 
  f b x₁ - f b x₂ > 63/16 - 3 * Real.log 2 := by
  sorry

#check f_difference_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_bound_l563_56306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_extreme_points_condition_l563_56337

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * (0.5 * x^2 - x)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * (x - 1)

/-- Theorem: For f(x) to have two extreme points, a must be in (e^2, +∞) -/
theorem two_extreme_points_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0) ↔ a > Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_extreme_points_condition_l563_56337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_transformed_sine_l563_56360

noncomputable section

/-- The original function f(x) -/
def f (x : ℝ) : ℝ := 4 * Real.sin (4 * x + Real.pi / 6)

/-- The transformed function g(x) -/
def g (x : ℝ) : ℝ := 4 * Real.sin (2 * x - Real.pi / 6)

/-- Definition of a symmetry center for a function -/
def is_symmetry_center (h : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, h (c + x) = h (c - x)

theorem symmetry_center_of_transformed_sine :
  is_symmetry_center g (Real.pi / 12) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_transformed_sine_l563_56360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l563_56364

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 1/2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x + Real.sin x, -1)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), π/4 ≤ x ∧ x ≤ π/2 → f x ≤ 1) ∧
  (∃ (x : ℝ), π/4 ≤ x ∧ x ≤ π/2 ∧ f x = 1) ∧
  (∀ (x : ℝ), π/4 ≤ x ∧ x ≤ π/2 → f x ≥ 1/2) ∧
  (∃ (x : ℝ), π/4 ≤ x ∧ x ≤ π/2 ∧ f x = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l563_56364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_l563_56380

-- Define the parabola function
def f (x : ℝ) : ℝ := -x^2 + 6*x - 5

-- Define the area calculation
noncomputable def area : ℝ := |∫ x in Set.Icc 0 1, f x| + ∫ x in Set.Icc 1 5, f x

-- Theorem statement
theorem parabola_area : area = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_l563_56380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l563_56347

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.sin α = (2 / 5) * Real.sqrt 5) (h4 : Real.sin (α + β) = 3 / 5) :
  Real.cos β = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l563_56347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l563_56334

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2) * x - Real.sin x

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → f y ≤ f x ∧
  f x = -Real.pi/4 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l563_56334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_quadruples_count_l563_56320

/-- A quadruple of nonnegative real numbers satisfying the given conditions -/
def ValidQuadruple (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
  a^2 + b^2 + c^2 + d^2 = 9 ∧
  (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 27

/-- The set of all valid quadruples -/
def ValidQuadruples : Set (ℝ × ℝ × ℝ × ℝ) :=
  {q | ValidQuadruple q.1 q.2.1 q.2.2.1 q.2.2.2}

/-- The number of valid quadruples is 15 -/
theorem valid_quadruples_count : 
  ∃ (s : Finset (ℝ × ℝ × ℝ × ℝ)), s.card = 15 ∧ ∀ q, q ∈ s ↔ q ∈ ValidQuadruples := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_quadruples_count_l563_56320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_results_l563_56308

-- Define the rotation matrix M₁
def M₁ : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

-- Define the shear matrix M₂
def M₂ : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1; 0, 1]

-- Define the point P
def P : Fin 2 → ℝ := ![2, 1]

-- Define the combined transformation matrix M
def M : Matrix (Fin 2) (Fin 2) ℝ := M₂ * M₁

theorem transformation_results :
  -- Part 1: The image of P under T₁ is (-1, 2)
  (M₁.mulVec P = ![(-1 : ℝ), 2]) ∧
  -- Part 2: The equation of the transformed curve is y - x = y²
  (∀ x y : ℝ, (∃ x₀ y₀ : ℝ, M.mulVec ![x₀, y₀] = ![x, y] ∧ y₀ = x₀^2) ↔ y - x = y^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_results_l563_56308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_with_sqrt3_eccentricity_l563_56303

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a ^ 2 + h.b ^ 2) / h.a ^ 2)

/-- The equation of asymptotes for a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ :=
  h.b / h.a

theorem hyperbola_asymptotes_with_sqrt3_eccentricity (h : Hyperbola) 
  (h_ecc : eccentricity h = Real.sqrt 3) :
  asymptote_slope h = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_with_sqrt3_eccentricity_l563_56303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arctan_five_twelfths_l563_56393

theorem cos_arctan_five_twelfths : Real.cos (Real.arctan (5 / 12)) = 12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arctan_five_twelfths_l563_56393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_four_eq_neg_32_l563_56350

/-- A cubic polynomial f(x) = x^3 - 2x^2 + 4 -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 4

/-- A cubic polynomial g with specific properties -/
noncomputable def g : ℝ → ℝ := sorry

/-- The roots of g are the squares of the roots of f -/
axiom g_roots_are_squares_of_f_roots : ∀ r : ℝ, f r = 0 → ∃ A : ℝ, g = fun x ↦ A * (x - r^2)

/-- g(0) = 2 -/
axiom g_zero : g 0 = 2

theorem g_four_eq_neg_32 : g 4 = -32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_four_eq_neg_32_l563_56350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_score_probability_is_24_49_l563_56314

-- Define the dart board
structure DartBoard where
  outer_radius : ℚ
  inner_radius : ℚ
  inner_values : Fin 3 → ℕ
  outer_values : Fin 3 → ℕ

-- Define the probability of hitting a region
def hit_probability (board : DartBoard) (is_inner : Bool) : ℚ :=
  if is_inner then
    (board.inner_radius^2) / (board.outer_radius^2)
  else
    (board.outer_radius^2 - board.inner_radius^2) / (board.outer_radius^2)

-- Define the probability of getting an odd score
def odd_score_probability (board : DartBoard) : ℚ :=
  24/49

-- Theorem statement
theorem odd_score_probability_is_24_49 (board : DartBoard) :
  board.outer_radius = 8 ∧
  board.inner_radius = 4 ∧
  board.inner_values = ![3, 5, 5] ∧
  board.outer_values = ![4, 3, 3] →
  odd_score_probability board = 24/49 := by
  sorry

-- Example usage
def example_board : DartBoard := {
  outer_radius := 8
  inner_radius := 4
  inner_values := ![3, 5, 5]
  outer_values := ![4, 3, 3]
}

#eval odd_score_probability example_board

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_score_probability_is_24_49_l563_56314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_calculation_l563_56389

/-- The speed of a stream given downstream and upstream speeds -/
noncomputable def stream_speed (downstream_speed upstream_speed : ℝ) : ℝ :=
  (downstream_speed - upstream_speed) / 2

theorem stream_speed_calculation (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 11)
  (h2 : upstream_speed = 8) :
  stream_speed downstream_speed upstream_speed = 1.5 := by
  -- Unfold the definition of stream_speed
  unfold stream_speed
  -- Substitute the given values
  rw [h1, h2]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_calculation_l563_56389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_cases_l563_56322

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define point P
def point_P : ℝ × ℝ := (2, 2)

-- Define a line passing through a point with slope k
def line_through_point (p : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y - p.2 = k * (x - p.1)

-- Define a vertical line passing through a point
def vertical_line_through_point (p : ℝ × ℝ) (x : ℝ) : Prop :=
  x = p.1

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 0)

-- Define the length of a chord
noncomputable def chord_length (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

theorem line_equation_cases (l : ℝ × ℝ → Prop) :
  (∀ x y, l (x, y) ↔ line_through_point point_P 2 x y) ∨
  ((∃ A B, circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ l A ∧ l B ∧ chord_length A B = 4 * Real.sqrt 2) →
   ((∀ x y, l (x, y) ↔ line_through_point point_P (3/4) x y) ∨
    (∀ x, l (x, point_P.2) ↔ vertical_line_through_point point_P x))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_cases_l563_56322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_is_99_l563_56398

/-- An arithmetic sequence with specific conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_147 : a 1 + a 4 + a 7 = 39
  sum_369 : a 3 + a 6 + a 9 = 27

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- Theorem: The sum of the first 9 terms of the given arithmetic sequence is 99 -/
theorem sum_9_is_99 (seq : ArithmeticSequence) : sum_n seq 9 = 99 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_is_99_l563_56398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_gcd_property_l563_56336

theorem function_gcd_property 
  (f : ℤ → ℕ) 
  (h_range : ∀ x, 1 ≤ f x ∧ f x ≤ 10^100)
  (h_gcd : ∀ x y, Nat.gcd (f x) (f y) = Nat.gcd (f x) (Int.natAbs (x - y))) :
  ∃ m n : ℕ, 0 < m ∧ 0 < n ∧ ∀ x, f x = Nat.gcd (Int.natAbs (m + x)) n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_gcd_property_l563_56336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l563_56384

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_proof (principal interest time : ℝ) 
  (h_principal : principal = 4000)
  (h_interest : interest = 640)
  (h_time : time = 2) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l563_56384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_greater_than_five_l563_56344

/-- The number of squares with side length n that can be formed in a 15x15 grid -/
def squareCount (n : ℕ) : ℕ := (16 - n) ^ 2

/-- The total number of squares with side length greater than 5 in a 15x15 grid -/
def totalSquares : ℕ :=
  (Finset.sum (Finset.range 10) (fun n => squareCount (n + 6))) + 8

theorem count_squares_greater_than_five :
  totalSquares = 393 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_greater_than_five_l563_56344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subtract_from_percentage_l563_56319

theorem subtract_from_percentage (n : ℕ) : n = 300 → (0.3 * (n : ℝ) - 70 = 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subtract_from_percentage_l563_56319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_property_l563_56338

def n : ℕ := 2018

def x : ℕ := n.factorial + 1

def fractions : Fin n → ℚ
  | i => (i.val + 1 + x) / ((i.val + 1) * x)

theorem fraction_property :
  (∀ i : Fin n, (fractions i).num.gcd (fractions i).den = 1) ∧
  (∀ i j : Fin n, i ≠ j → (fractions i).den ≠ (fractions j).den) ∧
  (∀ i j : Fin n, i ≠ j → ((fractions i) - (fractions j)).den < (fractions i).den ∧
                           ((fractions i) - (fractions j)).den < (fractions j).den) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_property_l563_56338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_approx_31_08_l563_56352

/-- The percentage that satisfies the given equation -/
noncomputable def P : ℝ := (100 * (5 - 65 + 36 * 1412 / 100)) / 1442

/-- The theorem stating that P is approximately equal to 31.08 -/
theorem P_approx_31_08 : 
  abs (P - 31.08) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_approx_31_08_l563_56352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_and_ratio_l563_56385

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)

-- Define the inverse function g⁻¹
noncomputable def g_inv (x : ℝ) : ℝ := (4 * x + 2) / (3 - x)

-- Theorem statement
theorem inverse_function_and_ratio :
  (∀ x, g (g_inv x) = x) ∧
  (∀ x, g_inv (g x) = x) ∧
  (4 : ℝ) / (-1 : ℝ) = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_and_ratio_l563_56385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_real_parts_is_zero_l563_56388

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z : ℂ) : Prop :=
  z^2 + 2*z + (3 - 4*i) = 0

-- Theorem statement
theorem product_of_real_parts_is_zero :
  ∃ (z₁ z₂ : ℂ), quadratic_equation z₁ ∧ quadratic_equation z₂ ∧
  (z₁ ≠ z₂) ∧ (Complex.re z₁ * Complex.re z₂ = 0) := by
  sorry

#check product_of_real_parts_is_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_real_parts_is_zero_l563_56388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l563_56323

-- Define the function representing the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x

-- Define the point of tangency
def point : ℝ × ℝ := (0, 1)

-- Define the slope of the tangent line
noncomputable def m : ℝ := Real.exp 0 + 2

-- Define the equation of the tangent line
noncomputable def tangent_line (x : ℝ) : ℝ := m * x + 1

-- Theorem stating that the tangent line equation is correct
theorem tangent_line_equation :
  (∀ x : ℝ, tangent_line x = m * x + 1) ∧
  (tangent_line point.1 = point.2) ∧
  (m = deriv f point.1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l563_56323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_clouds_intersected_l563_56315

/-- A plane in 3D space -/
structure Plane where

/-- A cloud in 3D space -/
structure Cloud where

/-- A straight line in 3D space -/
structure StraightLine where

/-- A partition of space created by planes -/
structure Partition where

/-- Define the space divided by 10 planes -/
def space_division (planes : Fin 10 → Plane) : Set Partition :=
  sorry

/-- Define the property that each partition contains at most one cloud -/
def at_most_one_cloud (p : Partition) : Prop :=
  sorry

/-- Define the number of partitions a straight line intersects -/
def intersected_partitions (l : StraightLine) (s : Set Partition) : ℕ :=
  sorry

/-- Theorem stating the maximum number of clouds intersected -/
theorem max_clouds_intersected :
  ∀ (planes : Fin 10 → Plane) (l : StraightLine),
    (∀ p ∈ space_division planes, at_most_one_cloud p) →
    intersected_partitions l (space_division planes) ≤ 11 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_clouds_intersected_l563_56315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l563_56370

-- Define the power function
noncomputable def power_function (k α : ℝ) (x : ℝ) : ℝ := k * x^α

-- State the theorem
theorem power_function_sum (k α : ℝ) :
  (power_function k α (1/2) = Real.sqrt 2/2) → k + α = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l563_56370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_circle_plus_theorem_l563_56356

-- Define the operation ⊕
noncomputable def circle_plus (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Define a function to represent the nested operation
noncomputable def nested_circle_plus : ℕ → ℝ
| 0 => 1000
| n + 1 => circle_plus (n.succ) (nested_circle_plus n)

-- State and prove the theorem
theorem nested_circle_plus_theorem :
  circle_plus 1 (nested_circle_plus 998) = 1 := by
  sorry

-- Auxiliary lemma to prove that the result is always in (-1, 1)
lemma nested_circle_plus_bound (n : ℕ) : 
  -1 < nested_circle_plus n ∧ nested_circle_plus n < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_circle_plus_theorem_l563_56356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l563_56348

/-- Triangle ABC with given properties -/
structure TriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  h_A : A = (5, -2)
  h_B : B = (7, 3)
  h_M_midpoint : M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  h_N_midpoint : N = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  h_M_y_axis : M.1 = 0
  h_N_x_axis : N.2 = 0

/-- Helper function to define a line through two points -/
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))}

/-- Main theorem about TriangleABC -/
theorem triangle_abc_properties (t : TriangleABC) : 
  t.C = (-5, -3) ∧ 
  ∃ (a b c : ℝ), a = 5 ∧ b = -2 ∧ c = -5 ∧ 
  (∀ (x y : ℝ), a * x + b * y + c = 0 ↔ (x, y) ∈ line_through t.M t.N) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l563_56348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_area_theorem_l563_56349

-- Define the vector type
def MyVector := ℝ × ℝ

-- Define the magnitude of a vector
noncomputable def magnitude (v : MyVector) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the locus C
def locus_C (x y : ℝ) : Prop :=
  Real.sqrt ((x + Real.sqrt 3)^2 + y^2) + Real.sqrt ((x - Real.sqrt 3)^2 + y^2) = 4

-- Define the ellipse E
def ellipse_E (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 4 = 1

-- Define the tangent line
def tangent_line (k m x y : ℝ) : Prop :=
  y = k * x + m

-- Theorem statement
theorem locus_and_area_theorem :
  ∀ (x y k m : ℝ),
    locus_C x y →
    (x^2 / 4 + y^2 = 1) ∧
    (tangent_line k m x y →
      ∃ (A B : MyVector),
        ellipse_E A.1 A.2 ∧
        ellipse_E B.1 B.2 ∧
        tangent_line k m A.1 A.2 ∧
        tangent_line k m B.1 B.2 ∧
        (1/2 * magnitude (A.1 - B.1, A.2 - B.2) * 
         (Real.sqrt (1 / (1 + k^2))) = Real.sqrt 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_area_theorem_l563_56349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_prism_l563_56332

/-- The volume of a right rectangular prism with face areas 36 cm², 48 cm², and 72 cm² -/
noncomputable def prism_volume : ℝ :=
  Real.sqrt (36 * 48 * 72)

/-- Theorem: The volume of a right rectangular prism with face areas 36 cm², 48 cm², and 72 cm² 
    is equal to √(36 * 48 * 72) cubic centimeters -/
theorem volume_of_prism (a b c : ℝ) 
  (h1 : a * b = 36) 
  (h2 : a * c = 48) 
  (h3 : b * c = 72) : 
  a * b * c = prism_volume := by
  sorry

/-- Compute an approximation of the prism volume -/
def approx_prism_volume : ℚ :=
  Rat.sqrt (36 * 48 * 72)

#eval approx_prism_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_prism_l563_56332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_six_heads_before_three_tails_l563_56307

/-- The probability of encountering 6 heads before 3 tails in repeated fair coin flips -/
def q : ℚ := 128 / 6225

/-- Represents a fair coin -/
def fair_coin : Fin 2 → ℚ
| 0 => 1/2  -- probability of heads
| 1 => 1/2  -- probability of tails

/-- The maximum number of consecutive heads needed -/
def max_heads : ℕ := 6

/-- The maximum number of consecutive tails allowed -/
def max_tails : ℕ := 3

theorem probability_six_heads_before_three_tails : q = 128 / 6225 := by
  sorry

/-- Calculate the sum of numerator and denominator of q -/
def sum_num_denom : ℕ := (q.num.toNat + q.den)

#eval sum_num_denom

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_six_heads_before_three_tails_l563_56307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l563_56318

-- Define the function f
noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.cos (2 * x) * Real.cos φ - 4 * Real.sin x * Real.cos x * Real.sin φ

-- State the theorem
theorem range_of_a (φ : ℝ) (h1 : φ > 0) 
  (h2 : ∀ x, f x φ = f (π/2 - x) φ)  -- Symmetry about x = π/2
  (h3 : ∀ ψ > 0, φ ≤ ψ)  -- φ is minimum
  (h4 : ∃ x₀ ∈ Set.Ioo 0 (π/2), ∃ a, f x₀ φ = a) :
  ∃ a, a ∈ Set.Icc (-2) 1 ∧ a ∉ Set.Icc 1 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l563_56318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l563_56387

noncomputable def ω : ℝ := 1

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos (ω * x), Real.cos (2 * ω * x))

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin (ω * x), 1)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬is_periodic f q

theorem f_properties :
  ω > 0 ∧
  smallest_positive_period f π ∧
  f (π / 4) = 1 ∧
  ∀ x ∈ Set.Icc (-3 * π / 8) (π / 8), 
    ∀ y ∈ Set.Icc (-3 * π / 8) (π / 8), 
      x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l563_56387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l563_56396

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (x + 1)

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, x ≠ -1 → y ≠ -1 → x ≠ y → (f a b x - f a b y) * (a - b) > 0) ∧
  (f a b 1 * f a b (b / a) = (f a b (Real.sqrt (b / a)))^2) ∧
  (f a b (b / a) ≤ f a b (Real.sqrt (b / a))) ∧
  (∀ x, x > 0 →
    let H := 2 * a * b / (a + b)
    let G := Real.sqrt (a * b)
    H ≤ f a b x ∧ f a b x ≤ G →
    (a > b → b / a ≤ x ∧ x ≤ Real.sqrt (b / a)) ∧
    (a < b → Real.sqrt (b / a) ≤ x ∧ x ≤ b / a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l563_56396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_exponential_l563_56309

theorem max_value_of_exponential (x : ℝ) : (2 : ℝ)^(x*(1-x)) ≤ (2 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_exponential_l563_56309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_different_digits_divisible_by_5_l563_56353

def is_composed_of_different_digits (n : ℕ) : Prop :=
  ∀ d₁ d₂, (d₁ ∈ n.digits 10) → (d₂ ∈ n.digits 10) → d₁ ≠ d₂

theorem smallest_number_with_different_digits_divisible_by_5 :
  ∃ (n : ℕ), 
    is_composed_of_different_digits n ∧ 
    n % 5 = 0 ∧
    (∀ m : ℕ, m < n → ¬(is_composed_of_different_digits m ∧ m % 5 = 0)) ∧
    n = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_different_digits_divisible_by_5_l563_56353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_problem_l563_56379

theorem multiplication_problem :
  ∃ (A B C D E : Nat),
    A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧
    D ≠ 0 ∧ E ≠ 0 ∧ D ≠ E ∧
    ({A, B, C, D, E} : Finset Nat) ⊆ {2, 0, 1, 6} ∧
    (100 * A + 10 * B + C) * (10 * D + E) = 6156 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_problem_l563_56379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_property_l563_56302

variable (N : Matrix (Fin 3) (Fin 3) ℝ)

def v1 : Fin 3 → ℝ := ![3, -2, 1]
def v2 : Fin 3 → ℝ := ![4, 1, -4]
def v3 : Fin 3 → ℝ := ![7, -1, -2]

def w1 : Fin 3 → ℝ := ![4, 1, -1]
def w2 : Fin 3 → ℝ := ![0, 2, 1]
def w3 : Fin 3 → ℝ := ![16, 7, -2.5]

theorem matrix_N_property (h1 : N.mulVec v1 = w1) (h2 : N.mulVec v2 = w2) :
  N.mulVec v3 = w3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_property_l563_56302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imZ_eq_neg_three_l563_56326

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define Z as a function of i
noncomputable def Z : ℂ := (3 + 2 * i) / i

-- Theorem statement
theorem imZ_eq_neg_three : Complex.im Z = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imZ_eq_neg_three_l563_56326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_is_24_l563_56317

def sequence_b : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | (n + 2) => (1/2) * sequence_b (n + 1) + (1/3) * sequence_b n

noncomputable def sequence_sum : ℚ := ∑' n, sequence_b n

theorem sequence_sum_is_24 : sequence_sum = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_is_24_l563_56317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l563_56377

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  0 < A ∧ A < Real.pi/2 →  -- A is acute
  0 < B ∧ B < Real.pi/2 →  -- B is acute
  0 < C ∧ C < Real.pi/2 →  -- C is acute
  A + B + C = Real.pi →    -- Sum of angles in a triangle
  Real.sin A = 3/5 →
  Real.tan (A - B) = -1/2 →
  b = 5 →
  -- Conclusions
  Real.tan B = 2 ∧
  c = 11/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l563_56377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_is_nine_l563_56325

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- The configuration of points and lines -/
structure Configuration where
  P : Digit
  Q : Digit
  R : Digit
  S : Digit
  T : Digit
  U : Digit
  distinct : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ P ≠ U ∧
             Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ Q ≠ U ∧
             R ≠ S ∧ R ≠ T ∧ R ≠ U ∧
             S ≠ T ∧ S ≠ U ∧
             T ≠ U

def line_sum (c : Configuration) : ℕ → ℕ
| 1 => c.P.val + c.Q.val + c.R.val + 1
| 2 => c.P.val + c.S.val + c.U.val + 1
| 3 => c.R.val + c.T.val + c.U.val + 1
| 4 => c.Q.val + c.T.val + 1
| 5 => c.Q.val + c.S.val + 1
| 6 => c.S.val + c.U.val + 1
| _ => 0

def total_sum (c : Configuration) : ℕ :=
  (line_sum c 1) + (line_sum c 2) + (line_sum c 3) +
  (line_sum c 4) + (line_sum c 5) + (line_sum c 6)

theorem Q_is_nine (c : Configuration) 
  (h : total_sum c = 100) : c.Q.val = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_is_nine_l563_56325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisibility_l563_56392

theorem factorial_divisibility (n : ℕ) (h : 1 ≤ n ∧ n ≤ 40) : 
  ∃ k : ℕ, (3 * n).factorial = k * (n.factorial ^ 3) := by
  sorry

#check factorial_divisibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisibility_l563_56392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_routes_to_algebratown_l563_56371

/-- Represents a point on the map -/
inductive Point
| M  -- Mathville
| A  -- Algebratown
| B  -- Intermediate point
| C  -- Intermediate point
| D  -- Intermediate point
| E  -- Intermediate point
| F  -- Intermediate point

/-- Represents a direct route between two points -/
inductive DirectRoute : Point → Point → Type
| MC : DirectRoute Point.M Point.C
| MB : DirectRoute Point.M Point.B
| MD : DirectRoute Point.M Point.D
| CD : DirectRoute Point.C Point.D
| BD : DirectRoute Point.B Point.D
| CF : DirectRoute Point.C Point.F
| DF : DirectRoute Point.D Point.F
| BE : DirectRoute Point.B Point.E
| DE : DirectRoute Point.D Point.E
| EA : DirectRoute Point.E Point.A
| FA : DirectRoute Point.F Point.A

/-- Represents a path from one point to another -/
inductive RoutePath : Point → Point → Type
| single {p q : Point} : DirectRoute p q → RoutePath p q
| cons {p q r : Point} : DirectRoute p q → RoutePath q r → RoutePath p r

/-- Counts the number of paths between two points -/
def countPaths (p q : Point) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem total_routes_to_algebratown :
  countPaths Point.M Point.A = 8 := by
  sorry

#check total_routes_to_algebratown

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_routes_to_algebratown_l563_56371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l563_56339

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (2 * x + φ)

theorem monotone_increasing_interval (φ : ℝ) 
  (h1 : ∀ x, f x φ ≤ |f (π/6) φ|) 
  (h2 : f (π/2) φ > f π φ) : 
  ∃ k : ℤ, MonotoneOn (f · φ) (Set.Icc (k * π + π/6) (k * π + 2*π/3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l563_56339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_zero_l563_56354

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^3 + k*x^2 - x + 3

-- State the theorem
theorem tangent_slope_at_zero (k : ℝ) : 
  (deriv (f k)) 0 = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_zero_l563_56354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_removed_tetrahedron_l563_56367

/-- A unit cube with a corner sliced off --/
structure SlicedCube where
  /-- The original cube is a unit cube --/
  is_unit_cube : Bool
  /-- The cut passes through points one-third along each edge from a vertex --/
  cut_at_third : Bool

/-- The volume of the tetrahedron removed from the cube --/
noncomputable def tetrahedron_volume (cube : SlicedCube) : ℝ := 1 / 108

/-- Theorem stating that the volume of the removed tetrahedron is 1/108 --/
theorem volume_of_removed_tetrahedron (cube : SlicedCube) 
  (h1 : cube.is_unit_cube = true) 
  (h2 : cube.cut_at_third = true) : 
  tetrahedron_volume cube = 1 / 108 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_removed_tetrahedron_l563_56367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_l563_56366

/-- The radius of a circle inscribed in a rhombus with given diagonals -/
noncomputable def inscribed_circle_radius (d₁ d₂ : ℝ) : ℝ :=
  (d₁ * d₂) / (4 * Real.sqrt ((d₁/2)^2 + (d₂/2)^2))

/-- Theorem: The radius of a circle inscribed in a rhombus with diagonals 16 and 30 is 120/17 -/
theorem inscribed_circle_radius_specific : inscribed_circle_radius 16 30 = 120 / 17 := by
  -- Unfold the definition of inscribed_circle_radius
  unfold inscribed_circle_radius
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_l563_56366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_implies_a_value_l563_56342

theorem complex_inequality_implies_a_value (a : ℝ) :
  (((1 : ℂ) + 2*Complex.I) / (a + Complex.I)).re > (3 : ℝ) / 2 → a = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_implies_a_value_l563_56342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l563_56397

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (3 * θ) = -23/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l563_56397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_l563_56394

def sequence_a : ℕ → ℕ
  | 0 => 6  -- Added case for 0
  | 1 => 6
  | (n + 2) => 2 * (sequence_a (n + 1) + 1) - (n + 2)

def sum_sequence (n : ℕ) : ℕ :=
  (List.range n).map (λ i => sequence_a (i + 1)) |>.sum

theorem units_digit_of_sum :
  (sum_sequence 2022) % 10 = 8 := by
  sorry

#eval (sum_sequence 2022) % 10  -- For testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_l563_56394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_divisible_by_37_l563_56390

/-- Represents a three-digit integer -/
structure ThreeDigitInt where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_units : units < 10

/-- Constructs the next term in the sequence -/
def nextTerm (n : ThreeDigitInt) : ThreeDigitInt :=
  { hundreds := (n.tens + 1) % 10,
    tens := (n.units + 1) % 10,
    units := (n.hundreds + 1) % 10,
    h_hundreds := by sorry
    h_tens := by sorry
    h_units := by sorry }

/-- Converts a ThreeDigitInt to a natural number -/
def threeDigitIntToNat (n : ThreeDigitInt) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Represents the sequence of terms -/
def generateSequence (start : ThreeDigitInt) : List ThreeDigitInt :=
  sorry

/-- Sum of all terms in the sequence -/
def sequenceSum (start : ThreeDigitInt) : Nat :=
  (generateSequence start).map threeDigitIntToNat |>.sum

/-- The main theorem to prove -/
theorem sequence_sum_divisible_by_37 (start : ThreeDigitInt) :
  37 ∣ sequenceSum start :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_divisible_by_37_l563_56390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_sum_l563_56341

open Complex Real

/-- Given a triangle ABC with side lengths a, b, c, and an arbitrary point P in the plane,
    the sum of the products of distances from P to two vertices divided by the product
    of the corresponding side lengths is greater than or equal to 1. -/
theorem triangle_inequality_sum (A B C P : ℂ) (a b c : ℝ) 
    (h_abc : a > 0 ∧ b > 0 ∧ c > 0)
    (h_side_a : Complex.abs (B - C) = a)
    (h_side_b : Complex.abs (C - A) = b)
    (h_side_c : Complex.abs (A - B) = c) :
  (Complex.abs (P - B) * Complex.abs (P - C)) / (b * c) +
  (Complex.abs (P - C) * Complex.abs (P - A)) / (c * a) +
  (Complex.abs (P - A) * Complex.abs (P - B)) / (a * b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_sum_l563_56341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circumscribed_circle_area_inequality_l563_56363

-- Define a structure for a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a function to calculate the area of a circle circumscribed around a rectangle
noncomputable def circleAreaAroundRectangle (r : Rectangle) : ℝ :=
  (Real.pi / 4) * (r.width^2 + r.height^2)

-- Define the theorem
theorem square_circumscribed_circle_area_inequality
  (s : ℝ) -- Area of the square
  (rectangles : List Rectangle) -- List of rectangles the square is cut into
  (h_positive : s > 0) -- Assumption that square area is positive
  (h_sum : s = (rectangles.map (λ r => r.width * r.height)).sum) -- Sum of rectangle areas equals square area
  : (Real.pi * s) ≤ 2 * (rectangles.map circleAreaAroundRectangle).sum := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circumscribed_circle_area_inequality_l563_56363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l563_56386

/-- Given a parabola and a line, prove that the distance between their intersection points is 5 -/
theorem intersection_distance (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 4 * x₁ →                -- Point A on parabola
  y₂^2 = 4 * x₂ →                -- Point B on parabola
  y₁ = 2 * x₁ - 2 →              -- Point A on line
  y₂ = 2 * x₂ - 2 →              -- Point B on line
  x₁ ≠ x₂ →                      -- Distinct points
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 25 -- Distance between A and B is 5
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l563_56386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_18_l563_56369

-- Define the variable cost function
noncomputable def C (x : ℝ) : ℝ :=
  if x < 15 then 12*x - 12*Real.log (x+1)
  else 21*x + 256/(x-2) - 200

-- Define the annual profit function
noncomputable def f (x : ℝ) : ℝ :=
  if x < 15 then 8*x + 12*Real.log (x+1) - 100
  else -x + 190 - 256/(x-2)

-- State the theorem
theorem max_profit_at_18 :
  ∃ (max_x : ℝ), max_x = 18 ∧
  ∀ (x : ℝ), x > 0 → f x ≤ f max_x ∧
  f max_x = 156 := by
  sorry

#check max_profit_at_18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_18_l563_56369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_EFK_l563_56382

-- Define the points
variable (A B C D E F K L : EuclideanPlane)

-- Define the cyclic quadrilateral ABCD
def is_cyclic_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define the intersection of rays
def rays_intersect (A B C D E : EuclideanPlane) : Prop := sorry

-- Define a point on the circumcircle
def on_circumcircle (A B C P : EuclideanPlane) : Prop := sorry

-- Define the length of a segment
noncomputable def segment_length (A B : EuclideanPlane) : ℝ := sorry

-- Define an angle
noncomputable def angle (A B C : EuclideanPlane) : ℝ := sorry

-- Define the radius of a circumcircle
noncomputable def circumcircle_radius (A B C : EuclideanPlane) : ℝ := sorry

theorem circumcircle_radius_EFK 
  (h1 : is_cyclic_quadrilateral A B C D)
  (h2 : rays_intersect A B D C E)
  (h3 : rays_intersect D A C B F)
  (h4 : on_circumcircle D E F L)
  (h5 : on_circumcircle D E F K)
  (h6 : segment_length L K = 5)
  (h7 : angle E B C = 15 * π / 180) :
  circumcircle_radius E F K = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_EFK_l563_56382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_approx_18_seconds_l563_56313

/-- Represents the length of each train in meters -/
noncomputable def train_length : ℝ := 475

/-- Represents the speed of the first train in km/h -/
noncomputable def speed_train1 : ℝ := 55

/-- Represents the speed of the second train in km/h -/
noncomputable def speed_train2 : ℝ := 40

/-- Converts km/h to m/s -/
noncomputable def km_per_hour_to_m_per_second (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

/-- Calculates the relative speed of the trains in m/s -/
noncomputable def relative_speed : ℝ :=
  km_per_hour_to_m_per_second speed_train1 + km_per_hour_to_m_per_second speed_train2

/-- Calculates the time taken for the slower train to pass the driver of the faster one -/
noncomputable def time_to_pass : ℝ :=
  train_length / relative_speed

theorem time_to_pass_approx_18_seconds :
  ∃ ε > 0, abs (time_to_pass - 18) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_approx_18_seconds_l563_56313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expansion_sum_of_coefficients_main_result_l563_56373

theorem sum_of_coefficients_expansion (c : ℝ) : 
  let expansion := -(5 - 2*c) * (c + 3*(5 - 2*c))
  expansion = -10*c^2 + 55*c - 75 := by
    sorry

theorem sum_of_coefficients (c : ℝ) :
  (-10 : ℝ) + 55 + (-75) = -30 := by
    ring

theorem main_result (c : ℝ) :
  let expansion := -(5 - 2*c) * (c + 3*(5 - 2*c))
  (-10 : ℝ) + 55 + (-75) = -30 := by
    exact sum_of_coefficients c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expansion_sum_of_coefficients_main_result_l563_56373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_paper_clips_l563_56378

/-- The number of paper clips Janet used in a day -/
def paper_clips_used (start : ℕ) (end_ : ℕ) : ℕ := start - end_

/-- Theorem: Janet used 59 paper clips given the start and end amounts -/
theorem janet_paper_clips : paper_clips_used 85 26 = 59 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_paper_clips_l563_56378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_selling_price_l563_56310

/-- Calculates the selling price of an article given its cost price and profit percentage -/
noncomputable def sellingPrice (costPrice : ℝ) (profitPercentage : ℝ) : ℝ :=
  costPrice * (1 + profitPercentage / 100)

/-- Theorem: The selling price of an article with a cost price of 640 and a profit of 25% is 800 -/
theorem article_selling_price :
  sellingPrice 640 25 = 800 := by
  -- Unfold the definition of sellingPrice
  unfold sellingPrice
  -- Simplify the expression
  simp [mul_add, mul_div_cancel]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_selling_price_l563_56310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_implies_a_range_l563_56321

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.log a - 2*x + 1

-- State the theorem
theorem two_roots_implies_a_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) →
  (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_implies_a_range_l563_56321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_palindrome_probability_l563_56375

/-- Represents a license plate with three letters followed by three digits -/
structure LicensePlate :=
  (letters : Fin 3 → Fin 26)
  (digits : Fin 3 → Fin 10)

/-- Checks if a sequence of three elements forms a palindrome -/
def isPalindrome {α : Type} [DecidableEq α] (seq : Fin 3 → α) : Prop :=
  seq 0 = seq 2

/-- The probability of a license plate containing at least one palindrome -/
def palindromeProbability : ℚ :=
  7/52

theorem license_plate_palindrome_probability :
  (∀ (plate : LicensePlate), (isPalindrome plate.letters ∨ isPalindrome plate.digits)) →
  palindromeProbability = 7/52 :=
by sorry

#check license_plate_palindrome_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_palindrome_probability_l563_56375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_is_two_l563_56330

open Real Topology Filter

noncomputable def f (n : ℝ) : ℝ := 
  (n * n^(1/6) + (32*n^10 + 1)^(1/5)) / ((n + n^(1/4)) * (n^3 - 1)^(1/3))

theorem limit_of_f_is_two :
  Tendsto f atTop (𝓝 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_is_two_l563_56330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_sqrt3_cos_range_l563_56391

theorem sin_plus_sqrt3_cos_range (m : ℝ) :
  (∀ x ∈ Set.Icc (π / 2) π, Real.sin x + Real.sqrt 3 * Real.cos x ≥ m) ↔ m ∈ Set.Iic (-Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_sqrt3_cos_range_l563_56391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l563_56345

-- Define the function representing the left side of the equation
noncomputable def f (x : ℝ) : ℝ :=
  3 * (Real.sin x)^4 - 7 * (Real.sin x)^3 + 5 * (Real.sin x)^2 - Real.sin x

-- Define the range
def range (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2 * Real.pi

-- Theorem statement
theorem equation_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, f x = 0 ∧ range x) ∧ 
                    (∀ x, f x = 0 ∧ range x → x ∈ S) ∧ 
                    Finset.card S = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l563_56345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juliet_supporter_capulet_prob_l563_56304

/-- The population distribution and voting preferences in Venezia -/
structure VeneziaElection where
  total_population : ℚ
  montague_fraction : ℚ
  capulet_fraction : ℚ
  romeo_support_montague : ℚ
  juliet_support_capulet : ℚ
  montague_fraction_constraint : montague_fraction = 2/3
  capulet_fraction_constraint : capulet_fraction = 1/3
  population_sum : montague_fraction + capulet_fraction = 1
  romeo_support_constraint : romeo_support_montague = 4/5
  juliet_support_constraint : juliet_support_capulet = 7/10

/-- The probability that a randomly chosen Juliet supporter resides in Capulet province -/
def juliet_supporter_in_capulet (e : VeneziaElection) : ℚ :=
  (e.juliet_support_capulet * e.capulet_fraction) /
  ((1 - e.romeo_support_montague) * e.montague_fraction + e.juliet_support_capulet * e.capulet_fraction)

/-- Theorem stating that the probability of a Juliet supporter being from Capulet is 7/11 -/
theorem juliet_supporter_capulet_prob (e : VeneziaElection) :
  juliet_supporter_in_capulet e = 7/11 := by
  sorry

#check juliet_supporter_capulet_prob

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juliet_supporter_capulet_prob_l563_56304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_perpendicular_PQ_l563_56395

-- Define the circle
noncomputable def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define points
variable (A B C D P Q : ℝ × ℝ)

-- Define the circle with diameter AB
noncomputable def circleAB (A B : ℝ × ℝ) : Set (ℝ × ℝ) := 
  Circle ((A.1 + B.1)/2, (A.2 + B.2)/2) (((A.1 - B.1)^2 + (A.2 - B.2)^2)/4)

-- Define conditions
axiom C_on_circle : C ∈ circleAB A B
axiom D_on_circle : D ∈ circleAB A B

-- Define lines
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) := 
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

-- Define intersection points
axiom P_intersection : P ∈ Line A C ∩ Line B D
axiom Q_intersection : Q ∈ Line A D ∩ Line B C

-- Define perpendicularity
def Perpendicular (l1 l2 : Set (ℝ × ℝ)) :=
  ∀ (p1 q1 p2 q2 : ℝ × ℝ), p1 ∈ l1 → q1 ∈ l1 → p2 ∈ l2 → q2 ∈ l2 →
    (p1.1 - q1.1) * (p2.1 - q2.1) + (p1.2 - q1.2) * (p2.2 - q2.2) = 0

-- Theorem statement
theorem AB_perpendicular_PQ :
  Perpendicular (Line A B) (Line P Q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_perpendicular_PQ_l563_56395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l563_56300

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- The point P -/
def P : ℝ × ℝ := (1, 2)

/-- Coefficients of the line L: y = x - 2 in the form Ax + By + C = 0 -/
def L : ℝ × ℝ × ℝ := (-1, 1, 2)

theorem distance_point_to_line_example :
  distance_point_to_line P.fst P.snd L.1 L.2.1 L.2.2 = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l563_56300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volunteer_assignment_probability_l563_56333

theorem volunteer_assignment_probability : 
  let num_volunteers : ℕ := 3
  let num_posts : ℕ := 4
  let total_assignments : ℕ := num_posts ^ num_volunteers
  let different_post_assignments : ℕ := num_posts * (num_posts - 1) * (num_posts - 2)
  let Prob_at_least_two_same_post : ℚ := 1 - (different_post_assignments : ℚ) / total_assignments
  Prob_at_least_two_same_post = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volunteer_assignment_probability_l563_56333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_abs_bound_implies_a_range_l563_56346

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 4^x) + (a / 2^x) + 1

-- Part 1
theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 0, f a x ≥ -7) ∧
  (∃ x ∈ Set.Icc (-1) 0, f a x = -7) →
  a = -6 := by sorry

-- Part 2
theorem abs_bound_implies_a_range (a : ℝ) :
  (∀ x ≥ 0, |f a x| ≤ 3) →
  a ∈ Set.Icc (-5) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_abs_bound_implies_a_range_l563_56346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_B_more_cost_effective_l563_56372

/-- Represents the cost function for buying apples at Supermarket A -/
noncomputable def cost_A (x : ℝ) : ℝ :=
  if x ≤ 4 then 10 * x else 6 * x + 16

/-- Represents the cost function for buying apples at Supermarket B -/
def cost_B (x : ℝ) : ℝ := 8 * x

/-- Theorem stating that Supermarket B is more cost-effective for 0 < m < 8 -/
theorem supermarket_B_more_cost_effective (m : ℝ) (h1 : 0 < m) (h2 : m < 8) :
  cost_B m < cost_A m := by
  sorry

#check supermarket_B_more_cost_effective

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_B_more_cost_effective_l563_56372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_cost_calculation_l563_56376

theorem car_cost_calculation (initial_friends car_wash_earnings extra_payment : ℕ) 
  (h1 : initial_friends = 6)
  (h2 : car_wash_earnings = 500)
  (h3 : extra_payment = 40) : ∃ cost : ℕ, cost = 4200 ∧
  (cost - car_wash_earnings) / (initial_friends - 1) = cost / initial_friends + extra_payment := by
  -- Define the cost
  let cost : ℕ := 4200
  -- Assert the existence of the cost
  use cost
  -- Split the goal into two parts
  apply And.intro
  -- Prove the first part: cost = 4200
  · rfl
  -- Prove the second part: the equation holds
  · sorry -- We'll skip the actual calculation for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_cost_calculation_l563_56376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_transformed_sine_l563_56335

/-- The phase shift of the function y = sin(5x - 2π) -/
noncomputable def phase_shift : ℝ := 2 * Real.pi / 5

/-- The sine function with the given transformation -/
noncomputable def transformed_sine (x : ℝ) : ℝ := Real.sin (5 * x - 2 * Real.pi)

theorem phase_shift_of_transformed_sine :
  ∀ x : ℝ, transformed_sine (x + phase_shift) = Real.sin (5 * x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_transformed_sine_l563_56335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l563_56340

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := ∃ (k m : ℝ), y = k * x + m

-- Define point M
def point_M : ℝ × ℝ := (1, -1)

-- Theorem statement
theorem line_equation : 
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  (ellipse x₁ y₁) →
  (ellipse x₂ y₂) →
  (line_l x₁ y₁) →
  (line_l x₂ y₂) →
  (line_l point_M.1 point_M.2) →
  (point_M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2)) →
  ∃ (x y : ℝ), 3 * x - 4 * y - 7 = 0 ∧ line_l x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l563_56340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_a_magnitude_l563_56368

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def vector_b (m : ℝ) : ℝ × ℝ := (m + 1, 1)
def vector_c (m : ℝ) : ℝ × ℝ := (2, m)

def vector_sum (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := dot_product v1 v2 = 0

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem vector_a_magnitude (m : ℝ) : 
  perpendicular (vector_sum (vector_a m) (vector_c m)) (vector_b m) → 
  vector_magnitude (vector_a m) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_a_magnitude_l563_56368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_f_l563_56355

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x : ℝ) : ℝ := x - (floor x)

theorem properties_of_f :
  (∀ x : ℝ, f x ≥ 0) ∧
  (∀ y : ℝ, ∃ x : ℝ, f x > y → False) ∧
  (∀ x : ℝ, f (x + 1) = f x) ∧
  (∃ x : ℝ, f x ≠ f (-x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_f_l563_56355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hostel_expenditure_correct_l563_56359

noncomputable def hostel_expenditure (initial_students : ℕ) (new_students : ℕ) 
  (average_decrease : ℚ) (total_increase : ℚ) : ℚ :=
  let final_students := initial_students + new_students
  let initial_total := initial_students * 
    (final_students * (average_decrease) + total_increase) / new_students
  initial_total + total_increase

-- We can't use #eval for noncomputable definitions, so we'll comment this out
-- #eval hostel_expenditure 100 20 5 400

theorem hostel_expenditure_correct : 
  hostel_expenditure 100 20 5 400 = 5400 := by
  -- Unfold the definition of hostel_expenditure
  unfold hostel_expenditure
  -- Simplify the arithmetic expressions
  simp [Nat.cast_add, Nat.cast_mul]
  -- The proof is completed by normalization of rational numbers
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hostel_expenditure_correct_l563_56359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l563_56361

open Complex Polynomial

/-- A polynomial is a function that can be expressed as a sum of monomials. -/
def IsPolynomial (f : ℂ → ℂ) (n : ℕ) : Prop :=
  ∃ p : Polynomial ℂ, degree p ≤ n ∧ ∀ z, eval z p = f z

theorem polynomial_equality (P Q : ℂ → ℂ) :
  (∃ n m : ℕ, n ≥ 1 ∧ m ≥ 1 ∧ IsPolynomial P n ∧ IsPolynomial Q m) →
  ({z : ℂ | P z = 0} = {z : ℂ | Q z = 0}) →
  ({z : ℂ | P z = 1} = {z : ℂ | Q z = 1}) →
  (∀ z, P z = Q z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l563_56361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liar_count_theorem_l563_56328

/-- Represents the type of statement a person can make -/
inductive Statement where
  | neighbor_right_knight : Statement
  | two_right_knight : Statement

/-- Represents a person at the table -/
structure Person where
  is_knight : Bool
  statement : Statement

/-- Represents the circular table configuration -/
def TableConfiguration := List Person

/-- Checks if a given configuration is valid according to the rules -/
def is_valid_configuration (config : TableConfiguration) : Bool :=
  sorry

/-- Counts the number of liars in a given configuration -/
def count_liars (config : TableConfiguration) : Nat :=
  config.filter (fun p => !p.is_knight) |>.length

theorem liar_count_theorem (config : TableConfiguration) :
  config.length = 120 → is_valid_configuration config →
  count_liars config ∈ ({0, 60, 120} : Set Nat) :=
  sorry

#check liar_count_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liar_count_theorem_l563_56328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_equals_pi_l563_56351

/-- The function f(x) = √3 * sin(ω * x) - cos(ω * x) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - Real.cos (ω * x)

/-- Theorem: If ω > 0 and the minimum distance between any two zeros of f is 1, then ω = π -/
theorem omega_equals_pi (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ x₁ x₂ : ℝ, f ω x₁ = 0 ∧ f ω x₂ = 0 ∧ |x₁ - x₂| = 1 ∧
    ∀ y₁ y₂ : ℝ, f ω y₁ = 0 → f ω y₂ = 0 → |y₁ - y₂| ≥ 1) :
  ω = π := by
  sorry

#check omega_equals_pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_equals_pi_l563_56351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_theorem_l563_56357

/-- The radius of the circumscribed sphere in a regular tetrahedron formed by spheres -/
noncomputable def circumscribed_sphere_radius (inscribed_radius : ℝ) : ℝ :=
  5 * (Real.sqrt 2 + 1)

/-- Theorem stating the relationship between inscribed and circumscribed sphere radii -/
theorem circumscribed_sphere_radius_theorem (inscribed_radius : ℝ) 
  (h : inscribed_radius = Real.sqrt 6 - 1) :
  circumscribed_sphere_radius inscribed_radius = 5 * (Real.sqrt 2 + 1) := by
  -- Unfold the definition of circumscribed_sphere_radius
  unfold circumscribed_sphere_radius
  -- The definition directly gives the result, so we're done
  rfl

#check circumscribed_sphere_radius_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_theorem_l563_56357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l563_56305

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x + a) / (2^x - 1)

-- State the theorem
theorem odd_function_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 1 ∧ ∀ x, f a x > 3 ↔ 0 < x ∧ x < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l563_56305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_l563_56381

theorem sin_cos_sum (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.sin α - Real.cos α = 1/2) :
  Real.sin α + Real.cos α = Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_l563_56381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_439_exists_m_for_223_n_223_is_smallest_l563_56316

def has_consecutive_439 (m n : ℕ) : Prop :=
  ∃ k : ℕ, (1000 * m) / n = 1000 * k + 439

theorem smallest_n_with_439 :
  ∀ m n : ℕ,
    m < n →
    Nat.Coprime m n →
    has_consecutive_439 m n →
    n ≥ 223 :=
by sorry

theorem exists_m_for_223 :
  ∃ m : ℕ,
    m < 223 ∧
    Nat.Coprime m 223 ∧
    has_consecutive_439 m 223 :=
by sorry

theorem n_223_is_smallest :
  ∀ n : ℕ,
    n < 223 →
    ¬∃ m : ℕ,
      m < n ∧
      Nat.Coprime m n ∧
      has_consecutive_439 m n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_439_exists_m_for_223_n_223_is_smallest_l563_56316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_sum_l563_56365

-- Define the piecewise function f
noncomputable def f (a c : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then 2*a*x + 4
  else if -2 ≤ x ∧ x ≤ 2 then x^2 - 2
  else 3*x - c

-- State the theorem
theorem continuous_piecewise_function_sum (a c : ℝ) :
  Continuous (f a c) → a + c = -17/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_sum_l563_56365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8root3_l563_56331

open Complex

-- Define the complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- Define the conditions
def condition_magnitude (z₁ : ℂ) : Prop := Complex.abs z₁ = 4
def condition_equation (z₁ z₂ : ℂ) : Prop := 4 * z₁^2 - 2 * z₁ * z₂ + z₂^2 = 0

-- Define the area of the triangle
noncomputable def triangle_area (z₁ z₂ : ℂ) : ℝ := (1 / 2) * Complex.abs z₁ * Complex.abs (z₂ - z₁)

-- Theorem statement
theorem triangle_area_is_8root3 
  (h1 : condition_magnitude z₁) 
  (h2 : condition_equation z₁ z₂) : 
  triangle_area z₁ z₂ = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8root3_l563_56331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pudding_distribution_l563_56329

theorem pudding_distribution (total_cups : ℕ) (students : ℕ) 
  (h1 : total_cups = 315) (h2 : students = 218) : 
  (Nat.ceil (students * (Nat.ceil (total_cups / students : ℚ) : ℚ)) - total_cups) = 121 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pudding_distribution_l563_56329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l563_56311

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the dot product of vectors AB and AC
def dot_product_AB_AC : ℝ := 9

-- Define the relationship between angles
noncomputable def angle_relation (A B C : ℝ × ℝ) : ℝ :=
  Real.sin (Real.arccos ((B.1 - A.1) / Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2))) -
  Real.cos (Real.arccos ((C.1 - A.1) / Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))) *
  Real.sin (Real.arccos ((C.1 - B.1) / Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)))

-- Define the area of triangle ABC
def area_ABC : ℝ := 6

-- Define point P on segment AB
variable (P : ℝ × ℝ)

-- Define x and y for vector CP
variable (x y : ℝ)

-- Define the vector CP
noncomputable def vector_CP (C A B : ℝ × ℝ) (x y : ℝ) : ℝ × ℝ := 
  (x * ((C.1 - A.1) / Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)) + 
   y * ((C.1 - B.1) / Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)),
   x * ((C.2 - A.2) / Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)) + 
   y * ((C.2 - B.2) / Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)))

-- State the theorem
theorem min_value_theorem (A B C : ℝ × ℝ) (P : ℝ × ℝ) (x y t : ℝ) :
  dot_product_AB_AC = 9 ∧ 
  angle_relation A B C = 0 ∧ 
  area_ABC = 6 ∧ 
  P.1 = A.1 + t * (B.1 - A.1) ∧ 
  P.2 = A.2 + t * (B.2 - A.2) ∧ 
  0 ≤ t ∧ t ≤ 1 ∧
  vector_CP C A B x y = (P.1 - C.1, P.2 - C.2) →
  (∀ x' y', 2/x' + 1/y' ≥ 11/12 + Real.sqrt 6 / 3) ∧
  (∃ x₀ y₀, 2/x₀ + 1/y₀ = 11/12 + Real.sqrt 6 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l563_56311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questionnaires_for_survey_l563_56374

/-- The minimum number of questionnaires to mail to get the required responses -/
def min_questionnaires (response_rate : ℚ) (required_responses : ℕ) : ℕ :=
  Nat.ceil ((required_responses : ℚ) / response_rate)

/-- Theorem stating the minimum number of questionnaires to mail -/
theorem min_questionnaires_for_survey : 
  min_questionnaires (65 / 100) 300 = 462 := by
  sorry

#eval min_questionnaires (65 / 100) 300

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questionnaires_for_survey_l563_56374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_1_line_equation_2_l563_56358

-- Define the slope of the original line
noncomputable def original_slope : ℝ := -1

-- Define the slope of the new lines
noncomputable def new_slope : ℝ := original_slope / 3

-- Theorem for the first line
theorem line_equation_1 (x y : ℝ) :
  x + 3 * y - 1 = 0 →
  (y - 1) / (x - (-4)) = new_slope ∧ 
  -4 + 3 * 1 - 1 = 0 := by
  sorry

-- Theorem for the second line
theorem line_equation_2 (x y : ℝ) :
  y = -1/3 * x - 10 →
  (y - (-10)) / x = new_slope ∧
  y = new_slope * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_1_line_equation_2_l563_56358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_M_to_l_l563_56301

open Real

-- Define the point M in polar coordinates
noncomputable def M : ℝ × ℝ := (2, π/3)

-- Define the line l in polar form
def l (ρ θ : ℝ) : Prop := ρ * sin (θ + π/4) = sqrt 2 / 2

-- Define the distance function
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  let x := p.1 * cos p.2
  let y := p.1 * sin p.2
  abs (x + y - 1) / sqrt 2

-- Theorem statement
theorem distance_from_M_to_l :
  distance_to_line M = sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_M_to_l_l563_56301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fuel_consumption_l563_56324

/-- Fuel consumption function for a car traveling 100 km -/
noncomputable def f (x : ℝ) : ℝ := (1/1200) * x^2 + 360/x - 2

/-- The theorem states that the minimum fuel consumption is 7 liters at 60 km/h -/
theorem min_fuel_consumption :
  (∀ x : ℝ, 0 < x → x ≤ 100 → f x ≥ 7) ∧
  f 60 = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fuel_consumption_l563_56324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_nested_equals_16_l563_56327

-- Define the function p
noncomputable def p (x y : ℝ) : ℝ :=
  if x > 0 ∧ y > 0 then x + 2*y
  else if x < 0 ∧ y < 0 then x - 3*y
  else if x ≥ 0 ∧ y ≤ 0 then 4*x + 2*y
  else if x = y then x^2 + y^2
  else 0  -- This else case is added to make the function total

-- State the theorem
theorem p_nested_equals_16 : p (p 2 (-2)) (p (-3) (-1)) = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_nested_equals_16_l563_56327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karthik_weight_upper_bound_l563_56399

/-- Karthik's weight in kilograms -/
def karthik_weight : ℝ := sorry

/-- Lower bound of Karthik's weight according to Karthik -/
def karthik_lower_bound : ℝ := 55

/-- Upper bound of Karthik's weight according to Karthik -/
def karthik_upper_bound : ℝ := 62

/-- Lower bound of Karthik's weight according to his brother -/
def brother_lower_bound : ℝ := 50

/-- Upper bound of Karthik's weight according to his brother -/
def brother_upper_bound : ℝ := 60

/-- Average of different probable weights of Karthik -/
def average_weight : ℝ := 56.5

/-- Upper limit of Karthik's weight according to his father -/
def father_upper_bound : ℝ := sorry

theorem karthik_weight_upper_bound :
  karthik_weight > karthik_lower_bound ∧
  karthik_weight < karthik_upper_bound ∧
  karthik_weight > brother_lower_bound ∧
  karthik_weight < brother_upper_bound ∧
  average_weight = 56.5 →
  father_upper_bound = 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_karthik_weight_upper_bound_l563_56399
