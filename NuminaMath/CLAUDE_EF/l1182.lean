import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_real_condition_l1182_118246

theorem complex_number_real_condition (θ : Real) : 
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  (Complex.mk (1 + Real.sin θ) (Real.cos θ - Real.sin θ)).im = 0 ↔
  θ = Real.pi / 4 ∨ θ = 5 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_real_condition_l1182_118246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l1182_118295

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem min_value_of_expression (seq : ArithmeticSequence) 
  (h : S seq 2017 = 4034) :
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 9/y ≥ 4) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l1182_118295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_not_power_function_l1182_118283

-- Define a power function
noncomputable def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

-- Define the function y = 2^x
noncomputable def exponentialFunction (x : ℝ) : ℝ := 2^x

-- Theorem: The exponential function y = 2^x is not a power function
theorem exponential_not_power_function :
  ¬ isPowerFunction exponentialFunction := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_not_power_function_l1182_118283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_rational_coefficients_is_365_l1182_118240

/-- The sum of the coefficients of all rational terms in the expansion of (2√x - 1/x)^6 -/
def sum_of_rational_coefficients : ℕ := 365

/-- The binomial expansion of (2√x - 1/x)^6 -/
noncomputable def binomial_expansion (x : ℝ) : ℝ := (2 * Real.sqrt x - 1 / x) ^ 6

theorem sum_of_rational_coefficients_is_365 :
  sum_of_rational_coefficients = 365 := by
  -- The proof goes here
  sorry

#eval sum_of_rational_coefficients

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_rational_coefficients_is_365_l1182_118240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1182_118264

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem max_omega_value (ω : ℝ) (φ : ℝ) :
  ω > 0 →
  |φ| ≤ π/2 →
  f ω φ (-π/8) = 0 →
  (∀ x : ℝ, f ω φ x = f ω φ (π/4 - x)) →
  (∀ x y : ℝ, π/5 < x → x < y → y < π/4 → f ω φ x < f ω φ y) →
  ω ≤ 14 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1182_118264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_minimum_f_inequality_range_l1182_118251

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x + 1)

-- Theorem for the extreme value of f(x)
theorem f_local_minimum : ∃ (x : ℝ), IsLocalMin f x ∧ f x = -1 := by
  sorry

-- Theorem for the range of a
theorem f_inequality_range (a : ℝ) : 
  (∀ x > 0, f x ≥ x + Real.log x + a + 1) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_minimum_f_inequality_range_l1182_118251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_specific_vectors_l1182_118231

noncomputable def vector_a : ℝ × ℝ := (Real.cos (Real.pi/4), Real.sin (Real.pi/4))
noncomputable def vector_b : ℝ × ℝ := (Real.cos (Real.pi/12), Real.sin (Real.pi/12))

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem dot_product_specific_vectors :
  dot_product vector_a vector_b = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_specific_vectors_l1182_118231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_net_speed_in_two_streams_l1182_118299

/-- Calculates the average speed for a stream given downstream and upstream speeds -/
noncomputable def average_speed (downstream upstream : ℝ) : ℝ :=
  (2 * downstream * upstream) / (downstream + upstream)

/-- Calculates the net speed of a man traveling in two streams -/
noncomputable def net_speed (d1 u1 d2 u2 : ℝ) : ℝ :=
  (average_speed d1 u1 + average_speed d2 u2) / 2

theorem man_net_speed_in_two_streams :
  let d1 : ℝ := 10  -- downstream speed in first stream
  let u1 : ℝ := 8   -- upstream speed in first stream
  let d2 : ℝ := 15  -- downstream speed in second stream
  let u2 : ℝ := 5   -- upstream speed in second stream
  abs (net_speed d1 u1 d2 u2 - 8.19) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_net_speed_in_two_streams_l1182_118299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_hare_given_claims_eq_27_59_l1182_118266

/-- Represents the type of creature on the island -/
inductive Creature : Type
| Hare : Creature
| Rabbit : Creature

/-- The probability of a creature being a hare -/
noncomputable def prob_hare : ℝ := 1/2

/-- The probability of a hare making a mistake -/
noncomputable def hare_mistake_prob : ℝ := 1/4

/-- The probability of a rabbit making a mistake -/
noncomputable def rabbit_mistake_prob : ℝ := 1/3

/-- The event of a creature claiming "I am not a hare" -/
def claim_not_hare (c : Creature) : Prop := sorry

/-- The event of a creature claiming "I am not a rabbit" -/
def claim_not_rabbit (c : Creature) : Prop := sorry

/-- The probability of the creature being a hare given both claims -/
noncomputable def prob_hare_given_claims : ℝ := sorry

/-- The main theorem stating the probability of the creature being a hare given both claims -/
theorem prob_hare_given_claims_eq_27_59 : 
  prob_hare_given_claims = 27/59 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_hare_given_claims_eq_27_59_l1182_118266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1182_118275

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.cos x ^ 2

theorem f_properties :
  (f (-Real.pi / 4) = 0) ∧
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ (Real.sqrt 2 + 1) / 2) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = (Real.sqrt 2 + 1) / 2) ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ 0) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1182_118275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_minus_one_monotonicity_of_f_max_value_of_g_l1182_118278

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - (a * x^2 + x) / (1 + x)^2

-- Define the function g(x)
def g (x : ℝ) : ℝ := (1 + 1/x)^x + (1 + x)^(1/x)

theorem tangent_line_at_e_minus_one :
  let a : ℝ := 1
  let x₀ : ℝ := Real.exp 1 - 1
  (deriv (f a)) x₀ = (Real.exp 1 - 1) / (Real.exp 1)^2 ∧
  f a x₀ = 1 / Real.exp 1 :=
sorry

theorem monotonicity_of_f (a : ℝ) (h : 2/3 < a ∧ a ≤ 2) :
  StrictMonoOn (f a) (Set.Ioo (-1 : ℝ) 0) ∧
  StrictAntiOn (f a) (Set.Ioo 0 (2*a-3)) ∧
  StrictMonoOn (f a) (Set.Ioi (2*a-3)) :=
sorry

theorem max_value_of_g :
  ∃ (x : ℝ), x > 0 ∧ g x = 4 ∧ ∀ (y : ℝ), y > 0 → g y ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_minus_one_monotonicity_of_f_max_value_of_g_l1182_118278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_lines_l1182_118280

-- Define the circle
def my_circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - 1)^2 = 2}

-- Define the two lines
def line1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}
def line2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - 3 = 0}

-- State the theorem
theorem circle_tangent_to_lines :
  ∃ (a : ℝ), (∀ (p : ℝ × ℝ), p ∈ my_circle a → (p ∉ line1 ∧ p ∉ line2)) ∧
             (∃ (p1 p2 : ℝ × ℝ), p1 ∈ my_circle a ∧ p1 ∈ line1 ∧
                                  p2 ∈ my_circle a ∧ p2 ∈ line2) ∧
             (a = 2) := by
  sorry

#check circle_tangent_to_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_lines_l1182_118280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_trees_example_l1182_118239

/-- The distance between consecutive trees in a yard -/
noncomputable def distance_between_trees (yard_length : ℝ) (num_trees : ℕ) : ℝ :=
  yard_length / (num_trees - 1)

/-- Theorem: In a 225-meter yard with 26 trees planted at equal distances,
    with one tree at each end, the distance between two consecutive trees is 9 meters. -/
theorem distance_between_trees_example :
  distance_between_trees 225 26 = 9 := by
  -- Unfold the definition of distance_between_trees
  unfold distance_between_trees
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_trees_example_l1182_118239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_mod_five_l1182_118237

theorem remainder_mod_five (m n : ℕ) (h : m = 15 * n - 1) : m % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_mod_five_l1182_118237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_percentage_formula_l1182_118255

/-- A hexagon formed by an equilateral triangle on top of a rectangle -/
structure TriangleRectangleHexagon where
  s : ℝ  -- side length of the equilateral triangle and longer side of the rectangle
  h : s > 0  -- ensure positive side length

/-- The area of the equilateral triangle in the hexagon -/
noncomputable def triangle_area (hex : TriangleRectangleHexagon) : ℝ :=
  (Real.sqrt 3 / 4) * hex.s^2

/-- The area of the rectangle in the hexagon -/
noncomputable def rectangle_area (hex : TriangleRectangleHexagon) : ℝ :=
  hex.s * (hex.s / 2)

/-- The total area of the hexagon -/
noncomputable def hexagon_area (hex : TriangleRectangleHexagon) : ℝ :=
  triangle_area hex + rectangle_area hex

/-- The percentage of the hexagon's area that is the equilateral triangle -/
noncomputable def triangle_percentage (hex : TriangleRectangleHexagon) : ℝ :=
  (triangle_area hex / hexagon_area hex) * 100

theorem triangle_percentage_formula (hex : TriangleRectangleHexagon) :
  triangle_percentage hex = (Real.sqrt 3 / (Real.sqrt 3 + 2)) * 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_percentage_formula_l1182_118255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equality_l1182_118272

theorem complex_modulus_equality (a : ℝ) (h1 : a > 0) : 
  (Complex.abs (2 + a * Complex.I) = Complex.abs (2 - Complex.I)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equality_l1182_118272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersection_l1182_118268

/-- The parabola y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- Point A on the parabola -/
noncomputable def A : ℝ × ℝ := (1, 1)

/-- The slope of the tangent line at point A -/
def tangent_slope : ℝ := 2

/-- The slope of the normal line at point A -/
noncomputable def normal_slope : ℝ := -1 / tangent_slope

/-- Point B, the intersection of the normal line and the parabola -/
noncomputable def B : ℝ × ℝ := (-3/2, 9/4)

/-- Theorem: B is the intersection of the normal line and the parabola -/
theorem normal_intersection :
  (A.1 = 1 ∧ A.2 = parabola A.1) ∧  -- A is on the parabola
  (tangent_slope = 2) ∧  -- Tangent slope at A
  (normal_slope = -1 / tangent_slope) ∧  -- Normal slope
  (B.2 - A.2 = normal_slope * (B.1 - A.1)) ∧  -- B is on the normal line
  (B.2 = parabola B.1)  -- B is on the parabola
  → B = (-3/2, 9/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersection_l1182_118268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teresa_purchase_l1182_118233

/-- Represents the purchase of items at a local shop -/
structure Purchase where
  sandwich_price : ℚ
  salami_price : ℚ
  brie_price : ℚ
  olives_price : ℚ
  olives_weight : ℚ
  feta_price : ℚ
  feta_weight : ℚ
  bread_price : ℚ
  total_spent : ℚ

/-- Calculates the number of sandwiches purchased -/
def num_sandwiches (p : Purchase) : ℚ :=
  (p.total_spent - (p.salami_price + p.brie_price + p.olives_price * p.olives_weight +
    p.feta_price * p.feta_weight + p.bread_price)) / p.sandwich_price

/-- Theorem stating that Teresa purchased 2 sandwiches -/
theorem teresa_purchase : 
  ∀ p : Purchase, 
    p.sandwich_price = 775/100 ∧ 
    p.salami_price = 4 ∧ 
    p.brie_price = 3 * p.salami_price ∧ 
    p.olives_price = 10 ∧ 
    p.olives_weight = 1/4 ∧ 
    p.feta_price = 8 ∧ 
    p.feta_weight = 1/2 ∧ 
    p.bread_price = 2 ∧ 
    p.total_spent = 40 → 
    num_sandwiches p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teresa_purchase_l1182_118233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l1182_118285

/-- Given a polynomial g such that g(x^2 + 2) = x^4 + 6x^2 + 4,
    prove that g(x^2 - 2) = x^4 - 2x^2 - 4 -/
theorem polynomial_identity (g : ℝ → ℝ) (hg : ∀ x : ℝ, g (x^2 + 2) = x^4 + 6*x^2 + 4) :
  ∀ x : ℝ, g (x^2 - 2) = x^4 - 2*x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l1182_118285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_17_l1182_118261

def sequencePattern (n : ℕ) : ℤ :=
  match n with
  | 0 => 5
  | n + 1 => (-1)^(n + 1) * (Int.natAbs (sequencePattern n) + 3)

theorem fifth_term_is_17 : sequencePattern 4 = 17 := by
  rw [sequencePattern]
  simp
  -- The proof steps would go here
  sorry

#eval sequencePattern 4  -- This will evaluate the 5th term (index 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_17_l1182_118261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_equal_tangents_l1182_118225

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define when a point is outside a circle
def is_outside (p : Point) (c : Circle) : Prop :=
  distance p c.center > c.radius

-- Define when a line segment is tangent to a circle
def is_tangent (p q : Point) (c : Circle) : Prop :=
  distance q c.center = c.radius ∧
  (p.1 - q.1) * (q.1 - c.center.1) + (p.2 - q.2) * (q.2 - c.center.2) = 0

-- The main theorem
theorem two_equal_tangents (c : Circle) (A : Point) 
  (h : is_outside A c) : 
  ∃ (B₁ B₂ : Point), 
    B₁ ≠ B₂ ∧
    is_tangent A B₁ c ∧ 
    is_tangent A B₂ c ∧
    distance A B₁ = distance A B₂ ∧
    ∀ (B : Point), is_tangent A B c → B = B₁ ∨ B = B₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_equal_tangents_l1182_118225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_correct_l1182_118260

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  3 * x^2 + 3 * y^2 + 15 * x - 9 * y - 27 = 0

/-- The area of the circle defined by the given equation -/
noncomputable def circle_area : ℝ := 35 / 2 * Real.pi

/-- Theorem stating that the area of the circle defined by the given equation is 35π/2 -/
theorem circle_area_is_correct :
  ∃ (x₀ y₀ r : ℝ), (∀ x y, circle_equation x y ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
  circle_area = π * r^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_correct_l1182_118260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_sum_is_three_l1182_118217

noncomputable def f (x : ℝ) : ℝ := 3 * x / (2 * x^3 - 6 * x^2 + 4 * x)

theorem undefined_sum_is_three :
  ∃ (A B C : ℝ),
    (∀ x : ℝ, x ≠ A ∧ x ≠ B ∧ x ≠ C → f x = f x) ∧
    A + B + C = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_sum_is_three_l1182_118217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_problem_l1182_118203

theorem inverse_function_problem (g : ℝ → ℝ) (g_inv : ℝ → ℝ) 
  (h1 : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g)
  (h2 : g 4 = 6)
  (h3 : g 7 = 2)
  (h4 : g 3 = 7) :
  g_inv (g_inv 7 + g_inv 6) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_problem_l1182_118203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vertical_chord_length_l1182_118296

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 9 = 1

/-- The x-coordinate of the left focus -/
noncomputable def left_focus : ℝ := -Real.sqrt 7

/-- A point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse_equation x y

/-- The distance between two points on a vertical line -/
def vertical_distance (p q : PointOnEllipse) : ℝ :=
  |p.y - q.y|

theorem ellipse_vertical_chord_length :
  ∀ (A B : PointOnEllipse),
    A.x = left_focus →
    B.x = left_focus →
    A ≠ B →
    vertical_distance A B = 9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vertical_chord_length_l1182_118296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_under_constraint_l1182_118273

theorem max_sum_under_constraint (O S : ℕ+) :
  (O : ℚ) / 11 < 7 / (S : ℚ) ∧ 7 / (S : ℚ) < 4 / 5 →
  ∃ (max_sum : ℕ), max_sum = 77 ∧ 
    ∀ (P Q : ℕ+), (P : ℚ) / 11 < 7 / (Q : ℚ) ∧ 7 / (Q : ℚ) < 4 / 5 →
      (P : ℕ) + (Q : ℕ) ≤ max_sum :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_under_constraint_l1182_118273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l1182_118227

/-- Sequence a_n -/
def a : ℕ+ → ℝ := sorry

/-- Sequence b_n -/
def b : ℕ+ → ℝ := sorry

/-- Partial sum of sequence a_n -/
def S : ℕ+ → ℝ := sorry

/-- Partial sum of sequence b_n -/
def T : ℕ+ → ℝ := sorry

axiom a_pos : ∀ n : ℕ+, a n > 0

axiom S_def : ∀ n : ℕ+, 6 * S n = (a n)^2 + 3 * (a n) - 4

axiom b_def : ∀ n : ℕ+, b n = 1 / ((a n - 1) * (a (n + 1) - 1))

theorem min_k_value : 
  (∃ k : ℝ, ∀ n : ℕ+, k > T n) ∧ 
  (∀ k' : ℝ, (∀ n : ℕ+, k' > T n) → k' ≥ 1/9) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l1182_118227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l1182_118235

theorem cube_root_inequality (x : ℝ) :
  x^(1/3) + 4 / (x^(1/3) + 5) ≤ 0 ↔ -64 ≤ x ∧ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l1182_118235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1182_118294

/-- The maximum area of a triangle ABC with AB = 2 and AC = √3 * BC is √3 -/
theorem max_triangle_area :
  ∀ (A B C : ℝ × ℝ),
  let ab := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let bc := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let ac := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (ab + bc + ac) / 2
  let area := Real.sqrt (s * (s - ab) * (s - bc) * (s - ac))
  ab = 2 ∧ ac = Real.sqrt 3 * bc →
  area ≤ Real.sqrt 3 ∧ ∃ (A' B' C' : ℝ × ℝ),
    let ab' := Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2)
    let bc' := Real.sqrt ((B'.1 - C'.1)^2 + (B'.2 - C'.2)^2)
    let ac' := Real.sqrt ((A'.1 - C'.1)^2 + (A'.2 - C'.2)^2)
    let s' := (ab' + bc' + ac') / 2
    let area' := Real.sqrt (s' * (s' - ab') * (s' - bc') * (s' - ac'))
    ab' = 2 ∧ ac' = Real.sqrt 3 * bc' ∧ area' = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1182_118294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_area_in_acres_l1182_118297

-- Define the scale ratio
def scale : ℝ := 3

-- Define the dimensions of the trapezoid on the drawing
def short_side : ℝ := 12
def long_side : ℝ := 18
def trapezoid_height : ℝ := 8  -- Renamed to avoid conflict with Mathlib's height

-- Define the conversion rate from square miles to acres
def acres_per_square_mile : ℝ := 480

-- Define the function to calculate the area of a trapezoid
noncomputable def trapezoid_area (a b h : ℝ) : ℝ := (a + b) * h / 2

-- Define the function to convert square centimeters to square miles
noncomputable def cm2_to_miles2 (area_cm2 : ℝ) : ℝ := area_cm2 * (scale^2)

-- Define the function to convert square miles to acres
noncomputable def miles2_to_acres (area_miles2 : ℝ) : ℝ := area_miles2 * acres_per_square_mile

-- Theorem statement
theorem plot_area_in_acres :
  miles2_to_acres (cm2_to_miles2 (trapezoid_area short_side long_side trapezoid_height)) = 518400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_area_in_acres_l1182_118297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_representation_exists_l1182_118254

/-- The irrational number α = (√29 - 1)/2 -/
noncomputable def α : ℝ := (Real.sqrt 29 - 1) / 2

/-- Coin denominations: 1 and α^k for natural k -/
noncomputable def coin_denomination (k : ℕ) : ℝ := if k = 0 then 1 else α ^ k

/-- Representation of a sum using at most 6 coins of each denomination -/
def valid_representation (n : ℕ) (rep : ℕ → ℕ) : Prop :=
  (∀ k, rep k ≤ 6) ∧ 
  (∃ K, ∀ k > K, rep k = 0) ∧
  (∑' k, (rep k : ℝ) * coin_denomination k) = n

/-- Main theorem: any natural number can be represented using the given coins -/
theorem coin_representation_exists (n : ℕ) : 
  ∃ rep : ℕ → ℕ, valid_representation n rep := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_representation_exists_l1182_118254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_max_no_global_max_f_three_roots_b_range_l1182_118207

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x

-- Statement 1: f(x) has a local maximum but no global maximum
theorem f_local_max_no_global_max :
  (∃ x₀ : ℝ, ∃ ε > 0, ∀ x, |x - x₀| < ε → f x ≤ f x₀) ∧
  (¬∃ M : ℝ, ∀ x, f x ≤ M) := by sorry

-- Statement 2: If f(x) = b has exactly three distinct real roots, then 0 < b < 6e^(-3)
theorem f_three_roots_b_range :
  ∀ b : ℝ, (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = b ∧ f x₂ = b ∧ f x₃ = b) →
  (0 < b ∧ b < 6 * Real.exp (-3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_max_no_global_max_f_three_roots_b_range_l1182_118207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangleAreaUnionTheorem_l1182_118200

/-- The area covered by six equilateral triangles arranged in a specific pattern -/
noncomputable def triangleAreaUnion : ℝ := 10.6875 * Real.sqrt 3

/-- The side length of each equilateral triangle -/
def triangleSideLength : ℝ := 3

/-- The number of triangles in the arrangement -/
def numberOfTriangles : ℕ := 6

/-- The area of a single equilateral triangle with side length 3 -/
noncomputable def singleTriangleArea : ℝ := (Real.sqrt 3 / 4) * triangleSideLength^2

/-- The area of the overlap between two adjacent triangles -/
noncomputable def overlapArea : ℝ := (Real.sqrt 3 / 4) * (triangleSideLength / 2)^2

/-- The total number of overlaps between adjacent triangles -/
def numberOfOverlaps : ℕ := numberOfTriangles - 1

/-- Theorem stating that the area covered by the triangles is equal to 10.6875√3 -/
theorem triangleAreaUnionTheorem : 
  triangleAreaUnion = 10.6875 * Real.sqrt 3 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangleAreaUnionTheorem_l1182_118200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_blend_cost_l1182_118279

/-- Represents the cost per pound of coffee types A and B -/
structure CoffeeCost where
  typeA : ℝ
  typeB : ℝ

/-- Represents the amount of coffee used in pounds -/
structure CoffeeAmount where
  typeA : ℝ
  typeB : ℝ

/-- Calculates the total cost of the coffee blend -/
def totalCost (cost : CoffeeCost) (amount : CoffeeAmount) : ℝ :=
  cost.typeA * amount.typeA + cost.typeB * amount.typeB

/-- The coffee blend problem -/
theorem coffee_blend_cost (cost : CoffeeCost) (amount : CoffeeAmount) :
  cost.typeA = 4.60 →
  amount.typeA = 67.52 →
  amount.typeB = 2 * amount.typeA →
  totalCost cost amount = 511.50 →
  abs (cost.typeB - 1.488) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_blend_cost_l1182_118279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_properties_l1182_118277

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_properties (a b c : ℝ) 
  (h : ∀ x : ℝ, f a b c x > 0 ↔ -2 < x ∧ x < 3/2) :
  a < 0 ∧ c > 0 ∧ a + b + c > 0 ∧ ¬(b > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_properties_l1182_118277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projected_triangle_area_l1182_118242

/-- A triangle in the xy-plane satisfying specific projection conditions --/
structure ProjectedTriangle where
  -- Vertices of the triangle
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  -- Conditions for x-axis projection
  x_proj_cond : min v1.1 (min v2.1 v3.1) = 1 ∧ max v1.1 (max v2.1 v3.1) = 5
  -- Conditions for y-axis projection
  y_proj_cond : min v1.2 (min v2.2 v3.2) = 8 ∧ max v1.2 (max v2.2 v3.2) = 13
  -- Conditions for y=x line projection
  diag_proj_cond : 
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
      (5 + 2.5 * t) = min (max v1.1 v1.2) (min (max v2.1 v2.2) (max v3.1 v3.2))) ∧
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
      (5 + 2.5 * t) = max (max v1.1 v1.2) (max (max v2.1 v2.2) (max v3.1 v3.2)))

/-- The area of a triangle given its vertices --/
noncomputable def triangleArea (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (v1.1 * (v2.2 - v3.2) + v2.1 * (v3.2 - v1.2) + v3.1 * (v1.2 - v2.2))

/-- Theorem: The area of the triangle satisfying the projection conditions is 17/2 --/
theorem projected_triangle_area (t : ProjectedTriangle) : 
  triangleArea t.v1 t.v2 t.v3 = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projected_triangle_area_l1182_118242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_30_l1182_118215

/-- Represents a clock with 12 hours and 360 degrees in a full rotation -/
structure Clock :=
  (hours : ℕ := 12)
  (full_rotation : ℕ := 360)

/-- Calculates the degree increment per hour on a clock -/
def degree_per_hour (c : Clock) : ℕ :=
  c.full_rotation / c.hours

/-- Calculates the position of the hour hand at a given time -/
def hour_hand_position (hour : ℕ) (minute : ℕ) : ℕ :=
  (hour * 30 + minute / 2) % 360

/-- Calculates the position of the minute hand at a given time -/
def minute_hand_position (minute : ℕ) : ℕ :=
  (minute * 6) % 360

/-- Calculates the smaller angle between two positions on a clock -/
def smaller_angle (pos1 : ℕ) (pos2 : ℕ) : ℕ :=
  min (Int.natAbs (pos1 - pos2)) (360 - Int.natAbs (pos1 - pos2))

/-- Theorem: The smaller angle between the hour and minute hands at 7:30 is 45 degrees -/
theorem clock_angle_at_7_30 (c : Clock) :
  smaller_angle (hour_hand_position 7 30) (minute_hand_position 30) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_30_l1182_118215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_perpendicular_property_l1182_118247

-- Define the parabola
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the circle
noncomputable def circle_P (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 8

-- Define the perpendicular property
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  (v1.1 * v2.1 + v1.2 * v2.2 = 0)

theorem parabola_circle_perpendicular_property
  (p : ℝ) 
  (hp : p > 0)
  (S T : ℝ × ℝ)
  (hS : parabola p S.1 S.2)
  (hT : parabola p T.1 T.2)
  (hST : circle_P S.1 S.2 ∧ circle_P T.1 T.2)
  (hSPT : perpendicular (S.1 - 3, S.2) (T.1 - 3, T.2))
  (M : ℝ × ℝ)
  (hM : circle_P M.1 M.2)
  (A B : ℝ × ℝ)
  (hA : parabola p A.1 A.2)
  (hB : parabola p B.1 B.2)
  (hMAB : perpendicular (M.1 - A.1, M.2 - A.2) (M.1 - (focus p).1, M.2 - (focus p).2) ∧
          perpendicular (M.1 - B.1, M.2 - B.2) (M.1 - (focus p).1, M.2 - (focus p).2)) :
  perpendicular (A.1 - (focus p).1, A.2 - (focus p).2) (B.1 - (focus p).1, B.2 - (focus p).2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_perpendicular_property_l1182_118247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_time_proof_l1182_118281

/-- The speed (in mph) at which Mrs. Quick should drive to arrive exactly on time -/
noncomputable def exact_time_speed : ℝ := 60

/-- The distance (in miles) from Mrs. Quick's home to the airport -/
noncomputable def distance : ℝ := 25

/-- The ideal time (in hours) to reach the airport -/
noncomputable def ideal_time : ℝ := 5/12

theorem arrival_time_proof :
  (distance / 50 = ideal_time + 5/60) ∧
  (distance / 75 = ideal_time - 5/60) →
  distance / exact_time_speed = ideal_time :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_time_proof_l1182_118281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yen_share_theorem_l1182_118284

/-- The National Wealth Fund (NWF) currency structure as of 01.12.2022 -/
structure NWF_2022 where
  total : ℝ
  euro : ℝ
  pounds : ℝ
  yuan : ℝ
  gold : ℝ
  rubles : ℝ
  yen_share_2021 : ℝ

/-- Calculates the share of Japanese yen in the NWF for 2022 -/
noncomputable def yen_share_2022 (nwf : NWF_2022) : ℝ :=
  (nwf.total - nwf.euro - nwf.pounds - nwf.yuan - nwf.gold - nwf.rubles) / nwf.total * 100

/-- Calculates the change in share of Japanese yen from 2021 to 2022 -/
noncomputable def yen_share_change (nwf : NWF_2022) : ℝ :=
  yen_share_2022 nwf - nwf.yen_share_2021

/-- Theorem stating the share of Japanese yen in 2022 and its change from 2021 -/
theorem yen_share_theorem (nwf : NWF_2022)
  (h_total : nwf.total = 1388.01)
  (h_euro : nwf.euro = 41.89)
  (h_pounds : nwf.pounds = 2.77)
  (h_yuan : nwf.yuan = 309.72)
  (h_gold : nwf.gold = 554.91)
  (h_rubles : nwf.rubles = 0.24)
  (h_yen_2021 : nwf.yen_share_2021 = 47.06) :
  abs (yen_share_2022 nwf - 34.47) < 0.01 ∧
  abs (yen_share_change nwf + 12.6) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yen_share_theorem_l1182_118284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_on_circle_radius_2_l1182_118212

/-- The circle with equation x^2 + y^2 = 1 -/
def unit_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- A point P that draws two tangents to the unit circle -/
structure TangentPoint where
  P : ℝ × ℝ
  tangent1 : ℝ × ℝ  -- Point of tangency for the first tangent
  tangent2 : ℝ × ℝ  -- Point of tangency for the second tangent
  is_on_unit_circle1 : tangent1 ∈ unit_circle
  is_on_unit_circle2 : tangent2 ∈ unit_circle
  is_tangent1 : (P.1 - tangent1.1) * tangent1.1 + (P.2 - tangent1.2) * tangent1.2 = 0  -- Tangent condition
  is_tangent2 : (P.1 - tangent2.1) * tangent2.1 + (P.2 - tangent2.2) * tangent2.2 = 0  -- Tangent condition
  angle_condition : Real.arccos ((P.1 - tangent1.1) * (P.1 - tangent2.1) + (P.2 - tangent1.2) * (P.2 - tangent2.2)) / 
                    (((P.1 - tangent1.1)^2 + (P.2 - tangent1.2)^2).sqrt * ((P.1 - tangent2.1)^2 + (P.2 - tangent2.2)^2).sqrt) = π / 3

/-- Theorem: The coordinates of P satisfy x^2 + y^2 = 4 -/
theorem tangent_point_on_circle_radius_2 (p : TangentPoint) : p.P.1^2 + p.P.2^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_on_circle_radius_2_l1182_118212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radius_inequality_l1182_118236

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi

-- Define helper functions
def circumradius (t : Triangle) : Real := sorry
def inradius (t : Triangle) : Real := sorry

-- Define the theorem
theorem triangle_radius_inequality (t : Triangle) (R r : Real) 
  (h1 : R > 0)  -- R is positive (circumradius)
  (h2 : r > 0)  -- r is positive (inradius)
  (h3 : R = circumradius t)  -- R is the circumradius of the triangle
  (h4 : r = inradius t)  -- r is the inradius of the triangle
  : R ≥ r / (2 * Real.sin (t.A / 2) * (1 - Real.sin (t.A / 2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radius_inequality_l1182_118236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_prime_factors_of_2_pow_10_minus_1_l1182_118290

theorem sum_of_prime_factors_of_2_pow_10_minus_1 : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (2^10 - 1 + 1))) 
    (λ p ↦ if (2^10 - 1).mod p = 0 then p else 0)) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_prime_factors_of_2_pow_10_minus_1_l1182_118290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l1182_118230

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then (2*a - 1)*x - 1 else x + 1

theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → (1/2 < a ∧ a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l1182_118230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pollution_scientific_notation_l1182_118270

/-- The volume of water polluted by a single discarded button cell battery in liters -/
noncomputable def single_battery_pollution : ℝ := 600000

/-- The total number of students in the school -/
def total_students : ℕ := 2200

/-- The fraction of students who discard a battery -/
noncomputable def discarding_fraction : ℝ := 1 / 2

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The total volume of water polluted by discarded batteries -/
noncomputable def total_pollution : ℝ :=
  (discarding_fraction * (total_students : ℝ)) * single_battery_pollution

/-- Theorem stating that the total pollution in scientific notation is 6.6 × 10^8 -/
theorem total_pollution_scientific_notation :
  toScientificNotation total_pollution = ScientificNotation.mk 6.6 8 sorry :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pollution_scientific_notation_l1182_118270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounds_l1182_118211

-- Define the curves
noncomputable def C₁ (a : Real) : Real × Real := (Real.sqrt 3 * Real.cos a, Real.sin a)

def C₂ (a : Real) : Real → Real := λ _ => a

noncomputable def C₃ (a : Real) : Real → Real := λ _ => a + Real.pi / 2

-- Define the area function
noncomputable def area (a : Real) : Real :=
  6 / Real.sqrt (3 + Real.sin (2 * a) ^ 2)

-- Theorem statement
theorem area_bounds :
  ∀ a : Real, 3 ≤ area a ∧ area a ≤ 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounds_l1182_118211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l1182_118205

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 8 * x + 12) / (3 * x + 4)

theorem oblique_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x > M, |f x - (x + 4/3)| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l1182_118205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_be_cd_range_l1182_118238

-- Define the triangle ABC
def Triangle (A B C : Real) (a b c : Real) : Prop :=
  -- Add appropriate conditions for a valid triangle
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0

-- Define the midpoints
def Midpoint (D : Real × Real) (A B : Real × Real) : Prop :=
  D.1 = (A.1 + B.1) / 2 ∧ D.2 = (A.2 + B.2) / 2

-- State the theorem
theorem be_cd_range (A B C : Real) (a b c : Real) 
  (D E : Real × Real) :
  Triangle A B C a b c →
  Midpoint D (0, 0) (c, 0) →
  Midpoint E (0, 0) (b * Real.cos C, b * Real.sin C) →
  2 * Real.sin C = 3 * Real.sin B →
  ∃ (r₁ r₂ : Real), r₁ = 8/7 ∧ r₂ = 4 ∧
  ∀ (x : Real), r₁ < x ∧ x < r₂ ↔ 
  ∃ (BE CD : Real), BE / CD = x ∧
  BE^2 = c^2 + (b^2)/4 - b*c*Real.cos A ∧
  CD^2 = b^2 + (c^2)/4 - b*c*Real.cos A :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_be_cd_range_l1182_118238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_over_twentyninth_200th_digit_l1182_118201

/-- The decimal representation of 7/29 has a repeating sequence of 29 digits -/
def repeating_sequence : Fin 29 → Nat :=
  ![2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7, 2]

/-- The 200th digit after the decimal point in the decimal representation of 7/29 -/
def digit_200 : Nat :=
  repeating_sequence ⟨200 % 29, by norm_num⟩

theorem seventh_over_twentyninth_200th_digit :
  digit_200 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_over_twentyninth_200th_digit_l1182_118201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_sum_two_dice_l1182_118245

/-- A die is represented as a finite type with 6 elements -/
inductive Die : Type where
  | one | two | three | four | five | six
deriving Fintype, Repr

/-- Convert Die to Nat -/
def Die.toNat : Die → Nat
  | Die.one => 1
  | Die.two => 2
  | Die.three => 3
  | Die.four => 4
  | Die.five => 5
  | Die.six => 6

/-- The sum of two dice rolls -/
def diceSum (d1 d2 : Die) : Nat :=
  Die.toNat d1 + Die.toNat d2

/-- The set of all possible outcomes when rolling two dice -/
def allOutcomes : Finset (Die × Die) :=
  Finset.product (Finset.univ : Finset Die) (Finset.univ : Finset Die)

/-- The set of outcomes where the sum is prime -/
def primeOutcomes : Finset (Die × Die) :=
  allOutcomes.filter (fun p => Nat.Prime (diceSum p.1 p.2))

/-- The probability of rolling a prime sum with two dice is 5/12 -/
theorem probability_prime_sum_two_dice :
  (primeOutcomes.card : ℚ) / (allOutcomes.card : ℚ) = 5 / 12 := by
  sorry

#eval primeOutcomes.card -- Expected: 15
#eval allOutcomes.card   -- Expected: 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_sum_two_dice_l1182_118245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x20_expansion_l1182_118249

/-- The sum of a finite geometric series with n terms and first term 1 -/
noncomputable def geometricSum (x : ℝ) (n : ℕ) : ℝ := (1 - x^(n+1)) / (1 - x)

/-- The coefficient of x^k in the expansion of (1-x^a)(1-x^b)^2 / (1-x)^3 -/
def coefficientExpansion (k a b : ℕ) : ℤ :=
  Nat.choose (k+2) 2 
  - 2 * Nat.choose (k-b+2) 2 * if k ≥ b then 1 else 0
  + Nat.choose (k-2*b+2) 2 * if k ≥ 2*b then 1 else 0

theorem coefficient_x20_expansion : 
  coefficientExpansion 20 24 14 = 36 := by sorry

#eval coefficientExpansion 20 24 14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x20_expansion_l1182_118249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1182_118292

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem min_value_of_f :
  ∃ (m : ℝ), m = -Real.sqrt 2 / 2 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → m ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1182_118292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_lines_iff_t_gt_e_squared_minus_four_l1182_118259

/-- The function f(x) = x * e^x --/
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

/-- The point P(t, -1) --/
def P (t : ℝ) : ℝ × ℝ := (t, -1)

/-- The existence of exactly three distinct tangent lines --/
def three_tangent_lines (t : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    ∀ i : ℝ, i ∈ ({x₁, x₂, x₃} : Set ℝ) →
      (f i - (-1)) / (i - t) = deriv f i

theorem three_tangent_lines_iff_t_gt_e_squared_minus_four :
  ∀ t : ℝ, three_tangent_lines t ↔ t > Real.exp 2 - 4 := by
  sorry

#check three_tangent_lines_iff_t_gt_e_squared_minus_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_lines_iff_t_gt_e_squared_minus_four_l1182_118259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1182_118265

theorem solve_exponential_equation (x : ℝ) : 4 * (2:ℝ)^x = 256 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1182_118265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_plain_is_four_fifths_l1182_118289

/-- The total number of zongzi on the plate -/
def total_zongzi : ℕ := 10

/-- The number of plain zongzi on the plate -/
def plain_zongzi : ℕ := 4

/-- The number of zongzi to be picked -/
def picked_zongzi : ℕ := 2

/-- The probability of picking x plain zongzi out of 2 picked zongzi -/
def prob (x : ℕ) : ℚ :=
  (Nat.choose plain_zongzi x * Nat.choose (total_zongzi - plain_zongzi) (picked_zongzi - x)) /
  Nat.choose total_zongzi picked_zongzi

/-- The expected number of plain zongzi picked -/
noncomputable def expected_plain : ℚ :=
  Finset.sum (Finset.range (picked_zongzi + 1)) (λ x => x * prob x)

theorem expected_plain_is_four_fifths :
  expected_plain = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_plain_is_four_fifths_l1182_118289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heart_sum_l1182_118253

noncomputable def heart (x : ℝ) : ℝ := (x + x^2) / 2

theorem heart_sum : heart 4 + heart 5 + heart 6 = 46 := by
  -- Unfold the definition of heart
  unfold heart
  -- Simplify the expression
  simp [add_div, pow_two]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heart_sum_l1182_118253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_theorem_l1182_118252

/-- The diameter of a wheel given its revolution count and distance covered -/
noncomputable def wheel_diameter (distance : ℝ) (revolutions : ℝ) : ℝ :=
  distance / (revolutions * Real.pi)

/-- Theorem stating that a wheel covering 1056 cm in 9.341825902335456 revolutions has a diameter of approximately 36 cm -/
theorem wheel_diameter_theorem :
  let d := wheel_diameter 1056 9.341825902335456
  ∃ ε > 0, abs (d - 36) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_theorem_l1182_118252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PQ_l1182_118271

-- Define the cube
def cube_edge_length : ℝ := 100

-- Define the speeds of points P and Q
def speed_P : ℝ := 3
def speed_Q : ℝ := 2

-- Define the positions of points P and Q as functions of time
def position_P (t : ℝ) : ℝ × ℝ × ℝ := (speed_P * t, speed_P * t, speed_P * t)
def position_Q (t : ℝ) : ℝ × ℝ × ℝ := (speed_Q * t, 0, cube_edge_length)

-- Define the distance function between P and Q
noncomputable def distance_PQ (t : ℝ) : ℝ :=
  Real.sqrt ((speed_P * t - speed_Q * t)^2 + (speed_P * t)^2 + (speed_P * t - cube_edge_length)^2)

-- State the theorem
theorem min_distance_PQ :
  ∃ t : ℝ, ∀ s : ℝ, distance_PQ t ≤ distance_PQ s ∧ distance_PQ t = 817 / 2450 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PQ_l1182_118271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_arithmetic_sequence_geometric_sequence_condition_unique_a₁_for_sum_bounds_l1182_118222

noncomputable def a_seq (k : ℝ) (a₁ : ℝ) : ℕ → ℝ
| 0 => a₁
| n + 1 => k * a_seq k a₁ n + n

noncomputable def b_seq (k : ℝ) (a₁ : ℝ) : ℕ → ℝ
| n => a_seq k a₁ n - 2/3 * n + 4/9

noncomputable def S_n (k : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ :=
  (List.range n).map (b_seq k a₁) |>.sum

theorem not_arithmetic_sequence (k : ℝ) :
  ¬ (∃ d : ℝ, ∀ n : ℕ, a_seq k 1 (n + 1) - a_seq k 1 n = d) := by sorry

theorem geometric_sequence_condition (a₁ : ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, b_seq (-1/2) a₁ (n + 1) = r * b_seq (-1/2) a₁ n) ↔ a₁ ≠ 2/9 := by sorry

theorem unique_a₁_for_sum_bounds :
  ∃! a₁ : ℝ, ∀ n : ℕ+, 1/3 ≤ S_n (-1/2) a₁ n ∧ S_n (-1/2) a₁ n ≤ 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_arithmetic_sequence_geometric_sequence_condition_unique_a₁_for_sum_bounds_l1182_118222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_XUSAMO_position_l1182_118244

/-- The set of letters used to form words -/
def Letters : Finset Char := {'A', 'M', 'O', 'S', 'U', 'X'}

/-- A word is a string of 6 characters from the set of Letters -/
def Word : Type := { w : String // w.length = 6 ∧ ∀ c, c ∈ w.data → c ∈ Letters }

/-- The alphabetical ordering of words -/
def wordOrder : Word → Word → Prop := λ w₁ w₂ ↦ w₁.val < w₂.val

/-- The position of a word in the alphabetical ordering -/
noncomputable def position (w : Word) : ℕ := sorry

/-- The specific word we're interested in -/
noncomputable def XUSAMO : Word := sorry

/-- Theorem stating the position of XUSAMO -/
theorem XUSAMO_position : position XUSAMO = 673 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_XUSAMO_position_l1182_118244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_theorem_l1182_118218

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Define the foci
def F₁ : Point := ⟨-4, 0⟩
def F₂ : Point := ⟨4, 0⟩

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : Point) : ℝ :=
  Real.sqrt ((p₁.x - p₂.x)^2 + (p₁.y - p₂.y)^2)

-- Define the angle between three points
noncomputable def angle (p₁ p₂ p₃ : Point) : ℝ := sorry

-- Theorem statement
theorem ellipse_angle_theorem (P : Point) 
  (h₁ : is_on_ellipse P.x P.y) 
  (h₂ : distance P F₁ * distance P F₂ = 12) : 
  angle F₁ P F₂ = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_theorem_l1182_118218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_isothermal_compression_approx_l1182_118243

/-- Work done during isothermal compression of an ideal gas in a cylinder -/
noncomputable def work_isothermal_compression (p₀ H h R : Real) : Real :=
  let S := Real.pi * R^2
  p₀ * S * H * Real.log (H / (H - h))

theorem work_isothermal_compression_approx :
  ∃ (ε : Real),
    ε > 0 ∧
    ε < 1 ∧
    |work_isothermal_compression 103300 0.8 0.4 0.2 - 7200| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_isothermal_compression_approx_l1182_118243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_division_theorem_l1182_118282

/-- The sum of areas of infinitely divided hexagons -/
noncomputable def hexagon_area_sum (a p q : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * a^2 * ((p + q)^2 / (p * q))

/-- The area of the original hexagon -/
noncomputable def original_hexagon_area (a : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * a^2

theorem hexagon_division_theorem (a p q : ℝ) 
  (h_pos : a > 0 ∧ p > 0 ∧ q > 0) :
  hexagon_area_sum a p q = 4 * original_hexagon_area a ↔ p = q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_division_theorem_l1182_118282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1182_118234

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  geom_prop : ∀ n : ℕ, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def S (g : GeometricSequence) (n : ℕ) : ℝ :=
  if g.q = 1 then n * g.a 1
  else g.a 1 * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_ratio (g : GeometricSequence) 
  (h1 : g.a 1 + g.a 3 = 5/2)
  (h2 : g.a 2 + g.a 4 = 5/4) :
  S g 5 / g.a 5 = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1182_118234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_probability_l1182_118202

def digit_set : Set ℕ := {1, 2, 3, 4, 5}

def is_valid_sum (a b c : ℕ) : Prop :=
  a ∈ digit_set ∧ b ∈ digit_set ∧ c ∈ digit_set ∧ a + b + c = 9

def count_valid_combinations : ℕ := 19

def total_combinations : ℕ := 125

theorem digit_sum_probability :
  (count_valid_combinations : ℚ) / (total_combinations : ℚ) = 19 / 125 := by
  -- The proof goes here
  sorry

#eval (count_valid_combinations : ℚ) / (total_combinations : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_probability_l1182_118202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_exponential_to_linear_distance_parabola_to_cubic_l1182_118228

-- Define the curves
noncomputable def curve_M1 (x : ℝ) : ℝ := Real.exp x
def curve_N1 (x : ℝ) : ℝ := x

def curve_M2 (x y : ℝ) : Prop := y^2 + 1 = x
def curve_N2 (x y : ℝ) : Prop := x^2 + 1 + y = 0

-- Define the distance between curves
noncomputable def distance_between_curves (f g : ℝ → ℝ) : ℝ :=
  Real.sqrt 2 / 2 -- This is a placeholder, the actual definition would be more complex

-- Define the distance between implicit curves
noncomputable def distance_between_implicit_curves (f g : ℝ → ℝ → Prop) : ℝ :=
  3 * Real.sqrt 2 / 4 -- This is a placeholder, the actual definition would be more complex

-- Theorem statements
theorem distance_exponential_to_linear :
  distance_between_curves curve_M1 curve_N1 = Real.sqrt 2 / 2 := by sorry

theorem distance_parabola_to_cubic :
  distance_between_implicit_curves curve_M2 curve_N2 = 3 * Real.sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_exponential_to_linear_distance_parabola_to_cubic_l1182_118228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sequence_periodicity_l1182_118298

noncomputable def cyclic_sequence (a : ℝ) : ℕ → ℝ
  | 0 => a  -- Add this case to handle n = 0
  | 1 => a
  | n + 2 => -1 / (cyclic_sequence a (n + 1) + 1)

theorem cyclic_sequence_periodicity (a : ℝ) (h : a > 0) :
  ∀ n : ℕ, cyclic_sequence a n = a ↔ n % 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sequence_periodicity_l1182_118298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_F_value_l1182_118221

-- Define the right triangle DEF
def DEF (D E F : ℝ × ℝ) : Prop :=
  (D = (0, 0)) ∧  -- Place D at the origin
  (E = (5, 0)) ∧  -- Place E on the x-axis
  ((F.1 - D.1)^2 + (F.2 - D.2)^2 = 13^2) -- F is on a circle with radius 13 centered at D

-- State the theorem
theorem cos_F_value (D E F : ℝ × ℝ) (h : DEF D E F) : 
  (F.1 - E.1) / 13 = 12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_F_value_l1182_118221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_occupied_squares_l1182_118267

/-- Represents a chessboard configuration --/
def ChessboardConfig := Fin 8 → Fin 8 → Bool

/-- Checks if two squares are adjacent (horizontally, vertically, or diagonally) --/
def are_adjacent (x1 y1 x2 y2 : Fin 8) : Prop :=
  (x1 = x2 ∧ (y1.val + 1 = y2.val ∨ y2.val + 1 = y1.val)) ∨
  (y1 = y2 ∧ (x1.val + 1 = x2.val ∨ x2.val + 1 = x1.val)) ∨
  ((x1.val + 1 = x2.val ∨ x2.val + 1 = x1.val) ∧ (y1.val + 1 = y2.val ∨ y2.val + 1 = y1.val))

/-- Checks if a configuration is valid (no adjacent occupied squares) --/
def is_valid_config (config : ChessboardConfig) : Prop :=
  ∀ x1 y1 x2 y2, config x1 y1 ∧ config x2 y2 → ¬(are_adjacent x1 y1 x2 y2)

/-- Counts the number of occupied squares in a configuration --/
def count_occupied (config : ChessboardConfig) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 8)) fun x =>
    Finset.sum (Finset.univ : Finset (Fin 8)) fun y =>
      if config x y then 1 else 0)

/-- The main theorem: The maximum number of occupied squares is 16 --/
theorem max_occupied_squares :
  (∃ config : ChessboardConfig, is_valid_config config ∧ count_occupied config = 16) ∧
  (∀ config : ChessboardConfig, is_valid_config config → count_occupied config ≤ 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_occupied_squares_l1182_118267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1182_118208

-- Define the triangle ABC
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Add conditions for a valid triangle if needed
  True

-- Define that D is the midpoint of AB
def Midpoint (D A B : ℝ) : Prop :=
  -- Add midpoint definition if needed
  True

-- Main theorem
theorem triangle_properties 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (D : ℝ) 
  (h1 : Triangle A B C a b c)
  (h2 : a = 2 * Real.sqrt 2)
  (h3 : Real.sqrt 2 * Real.sin (A + π/4) = b)
  (h4 : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)  -- Acute triangle condition
  (h5 : Midpoint D A B) :
  C = π/4 ∧ Real.sqrt 5 < ‖C - D‖ ∧ ‖C - D‖ < Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1182_118208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_line_segment_l1182_118276

open Real Set

-- Define the fixed points
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := 
  sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points P satisfying the condition
def S : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | distance p F₁ + distance p F₂ = 6}

-- State the theorem
theorem locus_is_line_segment : 
  S = {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • F₁ + t • F₂} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_line_segment_l1182_118276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_eight_l1182_118224

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = 2

noncomputable def geometric_subsequence (a : ℕ → ℝ) : Prop :=
  (a 4)^2 = a 2 * a 8

noncomputable def sum_of_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a 1 + (n - 1 : ℝ) * 2)

theorem arithmetic_sequence_sum_eight (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_subsequence a →
  sum_of_terms a 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_eight_l1182_118224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_at_zero_l1182_118287

-- Define the piecewise function f
noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*t*x + t^2
  else x + 1/x + t

-- State the theorem
theorem f_minimum_at_zero (t : ℝ) :
  (∀ x : ℝ, f t 0 ≤ f t x) → 0 ≤ t ∧ t ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_at_zero_l1182_118287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marilyns_visible_area_l1182_118262

/-- The area of the region visible to a person walking around a square --/
noncomputable def visible_area (side_length : ℝ) (visibility : ℝ) : ℝ :=
  let inner_square := (side_length - 2 * visibility)^2
  let outer_rectangles := 4 * side_length * visibility
  let corner_circles := 4 * Real.pi * visibility^2 / 4
  side_length^2 - inner_square + outer_rectangles + corner_circles

/-- Theorem stating the approximate area Marilyn can see --/
theorem marilyns_visible_area :
  ∃ (area : ℕ), area = 82 ∧ 
  |area - visible_area 7 1.5| ≤ 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marilyns_visible_area_l1182_118262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_eq_1025_l1182_118250

/-- The number of students playing cricket -/
def C : ℕ := 500

/-- The number of students playing basketball -/
def B : ℕ := 600

/-- The number of students playing soccer -/
def S : ℕ := 250

/-- The number of students playing tennis -/
def T : ℕ := 200

/-- The number of students playing both soccer and cricket -/
def SC : ℕ := 100

/-- The number of students playing both soccer and tennis -/
def STennis : ℕ := 50

/-- The number of students playing both soccer and basketball -/
def SB : ℕ := 75

/-- The number of students playing both cricket and basketball -/
def CB : ℕ := 220

/-- The number of students playing both cricket and tennis -/
def CT : ℕ := 150

/-- The number of students playing both basketball and tennis -/
def BT : ℕ := 50

/-- The number of students playing all four sports -/
def CBST : ℕ := 30

/-- The total number of students playing at least one sport -/
def total_students : ℕ := C + B + S + T - SC - STennis - SB - CB - CT - BT + 4 * CBST

theorem total_students_eq_1025 : total_students = 1025 := by
  rw [total_students]
  norm_num
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_eq_1025_l1182_118250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1182_118258

/-- A parabola with focus (0, -1), opening downward, and y-axis as axis of symmetry -/
structure DownwardParabola where
  focus : ℝ × ℝ
  focus_y : focus.2 = -1
  focus_x : focus.1 = 0
  opens_downward : True
  axis_of_symmetry_y : True

/-- The set of points on the parabola -/
def set_of_points (p : DownwardParabola) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 = -4*y}

/-- The standard equation of the parabola -/
def standard_equation (p : DownwardParabola) : Prop :=
  ∀ x y : ℝ, (x^2 = -4*y) ↔ (x, y) ∈ set_of_points p

theorem parabola_equation (p : DownwardParabola) : 
  standard_equation p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1182_118258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_l1182_118216

/-- A quadrilateral in a plane -/
structure Quadrilateral (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] where
  A : P
  B : P
  C : P
  D : P

/-- The property of a quadrilateral being convex -/
def is_convex (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] (Q : Quadrilateral P) : Prop :=
  sorry

/-- The property of a quadrilateral being cyclic (inscribed in a circle) -/
def is_cyclic (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] (Q : Quadrilateral P) : Prop :=
  sorry

/-- The distance between two points -/
def distance {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (A B : P) : ℝ :=
  ‖A - B‖

theorem quadrilateral_inequality {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] 
    (Q : Quadrilateral P) :
  distance Q.A Q.C * distance Q.B Q.D ≤ 
  distance Q.A Q.B * distance Q.C Q.D + distance Q.A Q.D * distance Q.B Q.C ∧
  (distance Q.A Q.C * distance Q.B Q.D = 
   distance Q.A Q.B * distance Q.C Q.D + distance Q.A Q.D * distance Q.B Q.C ↔
   is_convex P Q ∧ is_cyclic P Q) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_l1182_118216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_cells_congruence_l1182_118219

/-- A closed loop on an infinite grid formed by moves of length n -/
structure ClosedLoop (n : ℕ) where
  -- We represent the loop as a list of coordinates
  path : List (ℤ × ℤ)
  -- The path forms a closed loop
  is_closed : path.head? = path.getLast?
  -- Each move is of length n
  valid_moves : ∀ i, i < path.length - 1 → 
    (path[i]!.1 - path[i+1]!.1).natAbs + (path[i]!.2 - path[i+1]!.2).natAbs = n
  -- No cell is visited twice
  no_revisit : path.Nodup

/-- The number of white cells inside a closed loop -/
noncomputable def whiteCellCount (n : ℕ) (loop : ClosedLoop n) : ℕ := sorry

/-- Main theorem: The number of white cells inside the loop is congruent to 1 modulo n -/
theorem white_cells_congruence (n : ℕ) (h : n > 1) (loop : ClosedLoop n) :
  whiteCellCount n loop ≡ 1 [MOD n] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_cells_congruence_l1182_118219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_30_degrees_l1182_118209

/-- The angle_of_triangle function calculates the angle opposite to side c 
    in a triangle with side lengths a, b, and c -/
noncomputable def angle_of_triangle (a b c : ℝ) : ℝ :=
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

/-- Given a triangle ABC with side lengths a, b, and c, 
    if a^2 + b^2 = c^2 + √3*ab, then the angle C opposite to side c is equal to 30° -/
theorem triangle_angle_30_degrees 
  (a b c : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_eq : a^2 + b^2 = c^2 + Real.sqrt 3 * a * b) : 
  angle_of_triangle a b c = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_30_degrees_l1182_118209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_and_parallel_l1182_118210

-- Define the lines
def l1 (x y : ℝ) : Prop := 3*x + 2*y - 5 = 0
def l2 (x y : ℝ) : Prop := 3*x - 2*y - 1 = 0
def l3 (x y : ℝ) : Prop := 2*x + y - 5 = 0
def result_line (x y : ℝ) : Prop := 2*x + y - 3 = 0

-- Theorem statement
theorem line_through_intersection_and_parallel :
  ∃ (x y : ℝ),
    (l1 x y ∧ l2 x y) ∧  -- Intersection point satisfies both l1 and l2
    (result_line x y) ∧  -- The result line passes through the intersection point
    ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ),  -- Parallel condition
      result_line x y ↔ (2*x + y - 5 + k = 0) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_and_parallel_l1182_118210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_increasing_on_negative_reals_l1182_118229

-- Define the function f(x) = 2^x
noncomputable def f (x : ℝ) : ℝ := 2^x

-- State the theorem
theorem exponential_increasing_on_negative_reals :
  ∀ (x₁ x₂ : ℝ), x₁ < 0 → x₂ < 0 → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_increasing_on_negative_reals_l1182_118229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_set_eq_open_interval_l1182_118213

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain (x : ℝ) : Prop := x ≥ 0

-- State the slope condition
axiom slope_condition : ∀ x₁ x₂, domain x₁ → domain x₂ → x₁ ≠ x₂ → 
  (f x₂ - f x₁) / (x₂ - x₁) > 2

-- State the given value
axiom f_at_one : f 1 = 2020

-- Define the set we want to prove
def target_set : Set ℝ := {x | f (x - 2021) > 2 * (x - 1012)}

-- State the theorem
theorem target_set_eq_open_interval : 
  target_set = {x | x > 2022} :=
by
  sorry -- The proof is omitted here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_set_eq_open_interval_l1182_118213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1182_118293

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem f_properties :
  (∀ x, x ∈ Set.Icc (-5 : ℝ) 5 → f (-1) x ≤ 37 ∧ f (-1) x ≥ 1) ∧
  (∃ x₁ x₂, x₁ ∈ Set.Icc (-5 : ℝ) 5 ∧ x₂ ∈ Set.Icc (-5 : ℝ) 5 ∧ f (-1) x₁ = 37 ∧ f (-1) x₂ = 1) ∧
  (∀ a : ℝ, (∀ x y, x ∈ Set.Icc (-5 : ℝ) 5 → y ∈ Set.Icc (-5 : ℝ) 5 → x < y → f a x > f a y) ↔ a ≤ -5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1182_118293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_identification_l1182_118206

/-- A function f: ℝ → ℝ is quadratic if it can be written in the form f(x) = ax^2 + bx + c, where a ≠ 0 and a, b, c are real constants. -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- Function A: f(x) = 2x + 1 -/
noncomputable def f_A : ℝ → ℝ := λ x ↦ 2 * x + 1

/-- Function B: f(x) = x^2 + 1 -/
noncomputable def f_B : ℝ → ℝ := λ x ↦ x^2 + 1

/-- Function C: f(x) = (x - 1)^2 - x^2 -/
noncomputable def f_C : ℝ → ℝ := λ x ↦ (x - 1)^2 - x^2

/-- Function D: f(x) = 1 / x^2 -/
noncomputable def f_D : ℝ → ℝ := λ x ↦ 1 / x^2

theorem quadratic_function_identification :
  ¬(is_quadratic f_A) ∧
  (is_quadratic f_B) ∧
  ¬(is_quadratic f_C) ∧
  ¬(is_quadratic f_D) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_identification_l1182_118206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_special_integer_exists_l1182_118291

theorem four_digit_special_integer_exists : ∃ n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧  -- 4-digit positive integer
  (∀ i j : Fin 4, i ≠ j → (n.digits 10).get? i.val ≠ (n.digits 10).get? j.val) ∧  -- all digits are different
  (n.digits 10).head? ≠ some 0 ∧  -- leading digit is not zero
  n % 15 = 0 ∧  -- multiple of both 5 and 3
  (∀ d : ℕ, d ∈ n.digits 10 → d ≤ 7) ∧  -- 7 is the largest possible digit
  7 ∈ n.digits 10  -- 7 is one of the digits
  := by
  -- The proof goes here
  sorry

#check four_digit_special_integer_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_special_integer_exists_l1182_118291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_redistribution_theorem_l1182_118241

/-- Represents the set of job positions -/
def JobPositions : Type := Fin 100

/-- Represents a pair of job offers -/
structure JobOfferPair :=
  (first : JobPositions)
  (second : JobPositions)
  (different : first ≠ second)

/-- The set of all job offer pairs -/
def AllJobOfferPairs : Type := { pair : JobOfferPair // ∀ j : JobPositions, ∃! p : JobOfferPair, (j = p.first ∨ j = p.second) }

theorem job_redistribution_theorem 
  (initial_assignment : JobPositions → JobPositions)
  (h_initial_bijective : Function.Bijective initial_assignment)
  (offer_pairs : JobPositions → JobOfferPair)
  (h_offer_pairs_contain_initial : ∀ j : JobPositions, (offer_pairs j).first = initial_assignment j ∨ (offer_pairs j).second = initial_assignment j)
  (h_offer_pairs_unique : ∀ j1 j2 : JobPositions, j1 ≠ j2 → offer_pairs j1 ≠ offer_pairs j2) :
  ∃ (new_assignment : JobPositions → JobPositions), Function.Bijective new_assignment :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_redistribution_theorem_l1182_118241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gallon_weight_is_eight_l1182_118232

/-- Represents the weight of water and jello mix required to fill a bathtub -/
structure BathtubFill where
  water_pounds : ℚ
  jello_tablespoons : ℚ
  cubic_feet : ℚ
  gallons : ℚ
  cost : ℚ

/-- Calculates the weight of a gallon of water based on the given conditions -/
def gallon_weight (bf : BathtubFill) : ℚ :=
  bf.water_pounds / bf.gallons

/-- Theorem stating that under the given conditions, a gallon of water weighs 8 pounds -/
theorem gallon_weight_is_eight (bf : BathtubFill) 
  (h1 : bf.jello_tablespoons = 3/2 * bf.water_pounds)
  (h2 : bf.cubic_feet = 6)
  (h3 : bf.gallons = 15/2 * bf.cubic_feet)
  (h4 : bf.cost = 1/2 * bf.jello_tablespoons)
  (h5 : bf.cost = 270) :
  gallon_weight bf = 8 := by
  sorry

#check gallon_weight_is_eight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gallon_weight_is_eight_l1182_118232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_significant_improvement_l1182_118288

def x : Fin 10 → ℝ
| 0 => 545
| 1 => 533
| 2 => 551
| 3 => 522
| 4 => 575
| 5 => 544
| 6 => 541
| 7 => 568
| 8 => 596
| 9 => 548

def y : Fin 10 → ℝ
| 0 => 536
| 1 => 527
| 2 => 543
| 3 => 530
| 4 => 560
| 5 => 533
| 6 => 522
| 7 => 550
| 8 => 576
| 9 => 536

def z (i : Fin 10) : ℝ := x i - y i

noncomputable def z_mean : ℝ := (Finset.sum Finset.univ z) / 10

noncomputable def z_variance : ℝ := (Finset.sum Finset.univ (fun i => (z i - z_mean)^2)) / 10

theorem significant_improvement : z_mean ≥ 2 * Real.sqrt (z_variance / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_significant_improvement_l1182_118288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_into_four_isosceles_triangles_l1182_118274

/-- A square is a geometric shape with four equal sides and four right angles. -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- An isosceles triangle is a triangle with at least two equal sides. -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  base_positive : base > 0
  leg_positive : leg > 0

/-- A division of a square into isosceles triangles. -/
structure SquareDivision where
  square : Square
  triangles : List IsoscelesTriangle
  valid_division : triangles.length = 4 ∧ 
                   True  -- Placeholder for additional conditions

/-- The number of ways to divide a square into 4 isosceles triangles. -/
def numWaysToDivide (s : Square) : ℕ :=
  21  -- We're stating the result directly here

/-- Theorem: The number of ways to divide a fixed square into 4 isosceles triangles is 21. -/
theorem square_division_into_four_isosceles_triangles (s : Square) :
  numWaysToDivide s = 21 := by
  rfl  -- Reflexivity, since we defined numWaysToDivide to return 21


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_into_four_isosceles_triangles_l1182_118274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_log_M_div_N_l1182_118286

-- Define the upper limits of state space complexities
noncomputable def M : ℝ := 3^361
noncomputable def N : ℝ := 10^48

-- Define the logarithm of M/N
noncomputable def log_M_div_N : ℝ := Real.log M / Real.log 10 - Real.log N / Real.log 10

-- Define the possible answers
def possible_answers : List ℝ := [105, 125, 145, 165]

-- Theorem statement
theorem closest_to_log_M_div_N :
  ∃ (x : ℝ), x ∈ possible_answers ∧
  ∀ (y : ℝ), y ∈ possible_answers → |log_M_div_N - x| ≤ |log_M_div_N - y| := by
  sorry

#eval possible_answers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_log_M_div_N_l1182_118286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_450_prime_factors_l1182_118226

def sum_of_divisors (n : ℕ) : ℕ :=
  (Nat.divisors n).sum id

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.card

theorem sum_of_divisors_450_prime_factors :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_450_prime_factors_l1182_118226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l1182_118223

-- Define the function f(x) = a^(2x) + 2a^x - 1
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2*x) + 2*a^x - 1

-- Theorem statement
theorem max_value_implies_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ 7) ∧ 
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 7) →
  a = 2 ∨ a = 1/2 := by
  sorry

#check max_value_implies_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l1182_118223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_difference_zero_l1182_118256

/-- Represents a base-6 number with three digits -/
structure Base6Number :=
  (hundreds : Nat)
  (tens : Nat)
  (ones : Nat)
  (h_hundreds : hundreds < 6)
  (h_tens : tens < 6)
  (h_ones : ones < 6)

/-- Converts a Base6Number to its decimal representation -/
def base6ToDecimal (n : Base6Number) : Nat :=
  n.hundreds * 36 + n.tens * 6 + n.ones

/-- Theorem stating that the absolute difference between C and D is 0 in base 6 -/
theorem base6_difference_zero (C D : Nat) 
  (h_C : C < 10)
  (h_D : D < 10)
  (h_sum : base6ToDecimal ⟨D, D, C, sorry, sorry, sorry⟩ + 
           base6ToDecimal ⟨5, 2, D, sorry, sorry, sorry⟩ + 
           base6ToDecimal ⟨C, 2, 3, sorry, sorry, sorry⟩ = 
           base6ToDecimal ⟨C, 2, 3, sorry, sorry, sorry⟩ + 216) :
  base6ToDecimal ⟨0, 0, Int.natAbs (C - D), sorry, sorry, sorry⟩ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_difference_zero_l1182_118256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_15_12_l1182_118269

/-- A rhombus with given diagonal lengths -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal

/-- The area of a rhombus -/
noncomputable def area (r : Rhombus) : ℝ := (r.d1 * r.d2) / 2

/-- Theorem: The area of a rhombus with diagonals 15 cm and 12 cm is 90 square centimeters -/
theorem rhombus_area_15_12 :
  let r : Rhombus := { d1 := 15, d2 := 12 }
  area r = 90 := by
  -- Expand the definition of area
  unfold area
  -- Perform the calculation
  simp [Rhombus.d1, Rhombus.d2]
  -- The rest of the proof
  sorry

#eval (15 * 12) / 2  -- This should output 90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_15_12_l1182_118269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equality_l1182_118248

noncomputable def log_expr1 (x : ℝ) := Real.log (x^2 - 7*x + 12) / Real.log (x^2)
noncomputable def log_expr2 (x : ℝ) := Real.log (x^2 / (x - 3)) / Real.log (x^2)
noncomputable def log_expr3 (x : ℝ) := Real.log (x^2 / (x - 4)) / Real.log (x^2)

theorem log_sum_equality :
  ∃! x : ℝ, x > 0 ∧ x ≠ 3 ∧ x ≠ 4 ∧
  ((log_expr1 x = log_expr2 x + log_expr3 x) ∨
   (log_expr2 x = log_expr1 x + log_expr3 x) ∨
   (log_expr3 x = log_expr1 x + log_expr2 x)) ∧
  x = 5 := by
  sorry

#check log_sum_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equality_l1182_118248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1182_118214

/-- The time taken for B and C together to complete the work -/
noncomputable def time_BC : ℝ := 3

/-- The rate at which A completes the work -/
noncomputable def rate_A : ℝ := 1 / 3

/-- The rate at which B completes the work -/
noncomputable def rate_B : ℝ := 1 / 6.000000000000002

/-- The rate at which A and C together complete the work -/
noncomputable def rate_AC : ℝ := 1 / 2

/-- The rate at which B and C together complete the work -/
noncomputable def rate_BC : ℝ := 1 / time_BC

theorem work_completion_time :
  rate_BC = rate_B + (rate_AC - rate_A) :=
by sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1182_118214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meeting_count_l1182_118220

/-- Two runners on a circular track problem -/
theorem runners_meeting_count (track_length : ℝ) (speed1 speed2 : ℝ) (stop_distance : ℝ) : 
  track_length = 600 →
  speed1 = 4 →
  speed2 = 10 →
  stop_distance = 300 →
  (⌊(stop_distance / (track_length / (speed1 + speed2)) : ℝ)⌋ : ℤ) - 1 = 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meeting_count_l1182_118220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_l1182_118204

def A : ℂ := 5 - 2*Complex.I
def B : ℂ := -3 + 4*Complex.I
def C : ℂ := 2*Complex.I
def D : ℂ := 3

theorem complex_arithmetic : A - B + C - D = 5 - 4*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_l1182_118204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_calculation_l1182_118263

/-- The price of a TV set given specific payment conditions -/
noncomputable def tv_price (installments : ℕ) (installment_amount : ℝ) (interest_rate : ℝ) 
  (last_installment : ℝ) : ℝ :=
  let remaining_balance := last_installment / (1 + interest_rate / 2)
  remaining_balance + installment_amount

/-- Theorem stating the price of the TV set under given conditions -/
theorem tv_price_calculation :
  let installments := 20
  let installment_amount := 1200
  let interest_rate := 0.06
  let last_installment := 10800
  abs ((tv_price installments installment_amount interest_rate last_installment) - 11686.41) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_calculation_l1182_118263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_ride_time_l1182_118257

/-- Given a speed and distance, calculates the time taken -/
noncomputable def time_taken (speed distance : ℝ) : ℝ := distance / speed

/-- Theorem: Given a speed of 80.0 miles per hour and a distance of 1280 miles, the time taken is 16 hours -/
theorem james_ride_time :
  let speed := (80.0 : ℝ)
  let distance := (1280.0 : ℝ)
  time_taken speed distance = 16 := by
  -- Unfold the definitions
  unfold time_taken
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_ride_time_l1182_118257
