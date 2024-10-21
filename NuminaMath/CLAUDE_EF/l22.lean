import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_simplification_l22_2225

theorem trigonometric_function_simplification (α : Real) :
  let f : Real → Real := λ β => (Real.sin (Real.pi + β) * Real.sin (β + Real.pi/2)) / Real.cos (β - Real.pi/2)
  (f α = -Real.cos α) ∧
  (α ∈ Set.Icc Real.pi (3*Real.pi/2) → Real.cos (α + Real.pi/2) = 1/5 → f α = 2*Real.sqrt 6/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_simplification_l22_2225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l22_2242

theorem calculate_expression : 
  abs (-Real.sqrt 3) + (1/5)⁻¹ - Real.sqrt 27 + 2 * Real.cos (30 * π / 180) = 5 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l22_2242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_roots_Q_l22_2245

-- Define a real polynomial
def RealPolynomial := ℝ → ℝ

-- Define the property of having no real roots
def NoRealRoots (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, P x ≠ 0

-- Define Q(x) in terms of P(x)
noncomputable def Q (P : RealPolynomial) : RealPolynomial :=
  λ x => P x + (deriv (deriv P) x) / 2 + (deriv (deriv (deriv (deriv P))) x) / 24 + 0

-- Theorem statement
theorem no_real_roots_Q (P : RealPolynomial) (h : NoRealRoots P) : NoRealRoots (Q P) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_roots_Q_l22_2245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solutions_l22_2240

theorem complex_equation_solutions :
  ∃ (S : Set ℂ), 
    (∀ z ∈ S, Complex.abs z < 20 ∧ Complex.exp z = (z - 2) / (z + 2)) ∧
    (∃ (f : S → Fin 8), Function.Bijective f) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solutions_l22_2240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crossRatio_invariance_l22_2223

-- Define a structure for a point in a projective plane
structure ProjectivePoint where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a structure for a line in a projective plane
structure ProjectiveLine where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a pencil of lines
def LinePencil := ℝ → ProjectiveLine

-- Define a function to calculate the cross-ratio of four points on a line
noncomputable def crossRatio (A B C D : ProjectivePoint) : ℝ := sorry

-- Define a membership relation between ProjectivePoint and ProjectiveLine
def pointOnLine (P : ProjectivePoint) (L : ProjectiveLine) : Prop :=
  L.a * P.x + L.b * P.y + L.c * P.z = 0

-- Instance for the membership relation
instance : Membership ProjectivePoint ProjectiveLine where
  mem := pointOnLine

-- State the theorem
theorem crossRatio_invariance 
  (pencil : LinePencil) 
  (transversal1 transversal2 : ProjectiveLine) 
  (A B C D A1 B1 C1 D1 : ProjectivePoint) :
  A ∈ transversal1 → 
  B ∈ transversal1 → 
  C ∈ transversal1 → 
  D ∈ transversal1 →
  A1 ∈ transversal2 → 
  B1 ∈ transversal2 → 
  C1 ∈ transversal2 → 
  D1 ∈ transversal2 →
  (∃ (t : ℝ), pencil t = transversal1) →
  (∃ (t : ℝ), pencil t = transversal2) →
  crossRatio A B C D = crossRatio A1 B1 C1 D1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crossRatio_invariance_l22_2223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_of_point_l22_2275

noncomputable def rotate_point (x y angle : ℝ) : ℝ × ℝ :=
  (x * Real.cos angle - y * Real.sin angle,
   x * Real.sin angle + y * Real.cos angle)

theorem rotation_of_point :
  let initial_x : ℝ := Real.sqrt 3 / 2
  let initial_y : ℝ := 1 / 2
  let rotation_angle : ℝ := π / 2
  let (final_x, final_y) := rotate_point initial_x initial_y rotation_angle
  final_x = -1 / 2 ∧ final_y = Real.sqrt 3 / 2 := by
  sorry

#check rotation_of_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_of_point_l22_2275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_max_concave_angles_theorem_l22_2296

/-- 
Represents the maximum number of interior angles greater than 180° in an n-gon 
with the given properties.
-/
def max_concave_angles (n : ℕ) : ℕ :=
  if n = 4 then 0 else n - 3

/-- 
Represents an n-gon in the plane.
-/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  not_self_intersecting : True  -- Placeholder for the property
  equal_side_lengths : True     -- Placeholder for the property

/-- 
Count of concave angles in a polygon.
-/
def num_concave_angles (p : Polygon n) : ℕ :=
  sorry  -- Implementation details omitted

/-- 
Theorem stating the maximum number of interior angles greater than 180° in an n-gon 
with the given properties.
-/
theorem max_concave_angles_theorem (n : ℕ) 
  (h1 : n ≥ 3) :
  (∀ (p : Polygon n), num_concave_angles p ≤ max_concave_angles n) ∧
  (∃ (p : Polygon n), num_concave_angles p = max_concave_angles n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_max_concave_angles_theorem_l22_2296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_numerical_form_463001_l22_2265

def numerical_form (n : String) : ℕ :=
  match n with
  | "four hundred sixty-three thousand and one" => 463001
  | _ => 0  -- Default case, you might want to handle other cases differently

theorem numerical_form_463001 :
  numerical_form "four hundred sixty-three thousand and one" = 463001 := by
  rfl  -- reflexivity, since it's true by definition

#eval numerical_form "four hundred sixty-three thousand and one"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_numerical_form_463001_l22_2265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmers_speed_l22_2237

/-- Represents the swimming scenario with a stream -/
structure SwimmingScenario where
  stream_speed : ℚ
  still_water_speed : ℚ

/-- Calculates the time ratio of upstream to downstream swimming -/
def time_ratio (s : SwimmingScenario) : ℚ :=
  (s.still_water_speed - s.stream_speed) / (s.still_water_speed + s.stream_speed)

/-- The main theorem stating the swimmer's speed in still water -/
theorem swimmers_speed (s : SwimmingScenario) 
  (h1 : s.stream_speed = 3/2)
  (h2 : time_ratio s = 2) : 
  s.still_water_speed = 9/2 := by
  sorry

#check swimmers_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmers_speed_l22_2237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_a_in_range_l22_2230

/-- The function f(x) = ln x - ax + 1 has a root in the interval [1/e, e] -/
def has_root_in_interval (a : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), Real.log x - a * x + 1 = 0

/-- Theorem: If f(x) = ln x - ax + 1 has a root in [1/e, e], then 0 ≤ a ≤ 1 -/
theorem root_implies_a_in_range (a : ℝ) (h : has_root_in_interval a) :
  0 ≤ a ∧ a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_a_in_range_l22_2230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_l22_2259

/-- Given a circle and two points P and Q on it, this function returns the locus of
    intersection points M of chords AQ and BP, where A and B are variable points on the circle. -/
def locusOfIntersection (circle : Set (ℝ × ℝ)) (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- The theorem states that the locus of intersection points M is a circle passing through P and Q. -/
theorem locus_is_circle (circle : Set (ℝ × ℝ)) (P Q : ℝ × ℝ) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    locusOfIntersection circle P Q = {M | dist M center = radius ∧ M ∈ circle ∧ P ∈ circle ∧ Q ∈ circle} :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_l22_2259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_graph_is_finite_distinct_l22_2215

/-- Represents the cost of buying goldfish under given conditions -/
def goldfish_cost (n : ℕ) : ℚ :=
  if n ≥ 3 then 18 * n else 0

/-- Represents the set of points on the graph for 3 to 15 goldfish -/
def goldfish_graph : Set (ℕ × ℚ) :=
  {p | ∃ n : ℕ, 3 ≤ n ∧ n ≤ 15 ∧ p = (n, goldfish_cost n)}

/-- Theorem stating that the graph is a finite set of distinct points -/
theorem goldfish_graph_is_finite_distinct : 
  Finite goldfish_graph ∧ ∀ p q, p ∈ goldfish_graph → q ∈ goldfish_graph → p ≠ q → p.1 ≠ q.1 ∧ p.2 ≠ q.2 := by
  sorry

#check goldfish_graph_is_finite_distinct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_graph_is_finite_distinct_l22_2215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_range_l22_2220

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x + 2

/-- The condition that a ≤ 2 -/
def a_condition (a : ℝ) : Prop := a ≤ 2

/-- The range of x for which f(x) ≥ 0 should hold -/
def x_range : Set ℝ := Set.Icc 1 2

/-- The range of a that satisfies the conditions -/
def a_range : Set ℝ := Set.Icc (1 - 2 * Real.log 2) (1/2) ∪ Set.Ici 1

theorem f_nonnegative_range (a : ℝ) (ha : a_condition a) :
  (∀ x ∈ x_range, f a x ≥ 0) ↔ a ∈ a_range := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_range_l22_2220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_shifted_polynomials_l22_2244

/-- A monic polynomial of degree 2018 -/
def MonicPoly2018 := {p : Polynomial ℝ // p.degree = 2018 ∧ p.leadingCoeff = 1}

/-- The theorem statement -/
theorem exists_equal_shifted_polynomials
  (P Q : MonicPoly2018)
  (h : ∀ x : ℝ, (P.val).eval x ≠ (Q.val).eval x) :
  ∃ x : ℝ, (P.val).eval (x - 1) = (Q.val).eval (x + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_shifted_polynomials_l22_2244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_l22_2231

theorem factorial_difference : (Nat.factorial 12 / Nat.factorial 11 : Int) - Nat.factorial 7 = -5028 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_l22_2231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_reduction_l22_2273

theorem trigonometric_equation_reduction (a b c : ℕ+) :
  (∀ x : ℝ, (Real.sin x)^2 + (Real.sin (3*x))^2 + (Real.sin (5*x))^2 + (Real.sin (7*x))^2 = 2 ↔ 
    Real.cos (a.val * x) * Real.cos (b.val * x) * Real.cos (c.val * x) = 0) →
  a.val + b.val + c.val = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_reduction_l22_2273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_swap_l22_2284

/-- Represents a time on a clock face in degrees -/
noncomputable def ClockPosition (hours : ℝ) (minutes : ℝ) : ℝ :=
  (hours * 30 + minutes / 2) % 360

/-- The morning time when the person crossed the bridge -/
noncomputable def MorningTime : ℝ × ℝ := (8, 23 + 71 / 143)

/-- The afternoon time when the person returned -/
noncomputable def AfternoonTime : ℝ × ℝ := (4, 41 + 137 / 143)

/-- Theorem stating that the clock hands swap positions between the morning and afternoon times -/
theorem clock_hands_swap :
  ClockPosition MorningTime.1 MorningTime.2 = (6 * AfternoonTime.2) % 360 ∧
  ClockPosition AfternoonTime.1 AfternoonTime.2 = (6 * MorningTime.2) % 360 := by
  sorry

-- Remove #eval statements as they are not computable
-- Instead, we can use the following to show the values:
#print MorningTime
#print AfternoonTime
#print ClockPosition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_swap_l22_2284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_k_6_k_value_when_max_8_l22_2249

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 4 - k * abs (x - 2)

-- Define the domain
def domain : Set ℝ := Set.Icc 0 4

-- Theorem 1: Maximum value when k = 6
theorem max_value_when_k_6 :
  ∃ (M : ℝ), M = 0 ∧ ∀ x ∈ domain, f 6 x ≤ M := by
  sorry

-- Theorem 2: Value of k when maximum is 8
theorem k_value_when_max_8 :
  (∃ (M : ℝ), M = 8 ∧ ∀ x ∈ domain, f 2 x ≤ M) ∧
  (∀ k : ℝ, (∃ (M : ℝ), M = 8 ∧ ∀ x ∈ domain, f k x ≤ M) → k = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_k_6_k_value_when_max_8_l22_2249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l22_2204

theorem solutions_count : 
  ∃! n : ℕ, n = (Finset.filter 
    (λ p : ℕ × ℕ ↦ Nat.gcd p.fst p.snd = Nat.factorial 20 ∧ 
                  Nat.lcm p.fst p.snd = Nat.factorial 30)
    (Finset.product (Finset.range (Nat.factorial 30 + 1)) 
                    (Finset.range (Nat.factorial 30 + 1)))).card
  ∧ n = 256 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l22_2204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l22_2297

noncomputable def f (x : ℝ) : ℝ := x^2 + x

theorem problem_statement (m n : ℕ+) (y z : ℝ) :
  y ≠ z →
  y ≠ 0 →
  z ≠ 0 →
  f y = m + Real.sqrt n →
  f z = m + Real.sqrt n →
  f (1/y) + f (1/z) = 1/10 →
  100 * m + n = 1735 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l22_2297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_market_demand_variance_l22_2229

noncomputable def strawberry_prices : List ℝ := [12, 16, 20, 24, 28]

def market_demand (x : ℝ) : ℝ := -0.5 * x + 20

noncomputable def variance (l : List ℝ) : ℝ :=
  let mean := (l.sum) / l.length
  (l.map (fun x => (x - mean)^2)).sum / l.length

theorem market_demand_variance :
  variance (strawberry_prices.map market_demand) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_market_demand_variance_l22_2229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ln_curve_l22_2292

theorem tangent_line_to_ln_curve (x : ℝ) (b : ℝ) :
  (HasDerivAt Real.log (1 / x) x) →
  ((1 / 3 : ℝ) = 1 / x) →
  (Real.log x = (1 / 3) * x + b) →
  (b = -1 + Real.log 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ln_curve_l22_2292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_minus_three_count_l22_2247

-- Define the number of pages in the book
def num_pages : ℕ := 530

-- Function to count occurrences of a digit in a number
def count_digit (d : ℕ) (n : ℕ) : ℕ := sorry

-- Function to count occurrences of a digit in a range of numbers
def count_digit_in_range (d : ℕ) (start : ℕ) (end_ : ℕ) : ℕ := sorry

-- Theorem statement
theorem five_minus_three_count : 
  count_digit_in_range 5 1 num_pages - count_digit_in_range 3 1 num_pages = 34 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_minus_three_count_l22_2247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l22_2210

-- Define the original line
def original_line (x y : ℝ) : Prop := x + Real.sqrt 3 * y = 0

-- Define the rotated line
def rotated_line (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 0

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 1 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 0)

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem line_tangent_to_circle :
  distance_point_to_line circle_center (Real.sqrt 3) 1 0 = 
  Real.sqrt (circle_center.1^2 + circle_center.2^2 - 4*circle_center.1 + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l22_2210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_side_l22_2299

-- Define the angle θ
variable (θ : Real)

-- Define the point M(x,1)
variable (x : Real)

-- Define the condition that M(x,1) is on the terminal side of θ
def on_terminal_side (x : Real) (θ : Real) : Prop :=
  x ≥ 0 ∨ (x < 0 ∧ θ > Real.pi)

-- State the theorem
theorem point_on_angle_side (h1 : on_terminal_side x θ) (h2 : Real.cos θ = (Real.sqrt 2 / 2) * x) :
  x = -1 ∨ x = 0 ∨ x = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_side_l22_2299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_equals_n_l22_2217

/-- Given an integer n > 1 and a positive integer m, let S_m be the set {1, 2, ..., mn}.
    F is a family of sets satisfying the following conditions:
    1. |F| = 2n
    2. Every set in F is an m-element subset of S_m
    3. Any two sets in F have at most one common element
    4. Each element of S_m appears in exactly two sets in F
    
    Then, the maximum value of m is n. -/
theorem max_m_equals_n (n : ℕ) (m : ℕ) (h_n : n > 1) (h_m : m > 0) 
  (S_m : Finset ℕ) (F : Finset (Finset ℕ))
  (h_S_m : S_m = Finset.range (m * n))
  (h_F_card : F.card = 2 * n)
  (h_F_subset : ∀ A, A ∈ F → A ⊆ S_m ∧ A.card = m)
  (h_F_intersection : ∀ A B, A ∈ F → B ∈ F → A ≠ B → (A ∩ B).card ≤ 1)
  (h_S_m_coverage : ∀ x, x ∈ S_m → (F.filter (λ A ↦ x ∈ A)).card = 2) :
  m ≤ n ∧ ∃ m_max : ℕ, m_max = n ∧ 
    ∀ m' : ℕ, m' > m_max → ¬∃ F' : Finset (Finset ℕ), 
      F'.card = 2 * n ∧
      (∀ A, A ∈ F' → A ⊆ Finset.range (m' * n) ∧ A.card = m') ∧
      (∀ A B, A ∈ F' → B ∈ F' → A ≠ B → (A ∩ B).card ≤ 1) ∧
      (∀ x, x ∈ Finset.range (m' * n) → (F'.filter (λ A ↦ x ∈ A)).card = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_equals_n_l22_2217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_discount_proof_l22_2221

/-- Calculates the discounted price of a shirt given the original price and discount percentage. -/
def discounted_price (original_price : ℚ) (discount_percent : ℚ) : ℚ :=
  original_price * (1 - discount_percent / 100)

/-- Rounds a rational number to the nearest hundredth (penny). -/
def round_to_penny (price : ℚ) : ℚ :=
  (price * 100).floor / 100

theorem shirt_discount_proof :
  let original_price : ℚ := 933.33
  let discount_percent : ℚ := 40
  round_to_penny (discounted_price original_price discount_percent) = 560 := by
  sorry

#eval round_to_penny (discounted_price 933.33 40)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_discount_proof_l22_2221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_jump_discontinuity_at_zero_l22_2276

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (5 ^ (1 / x)) / (1 + 5 ^ (1 / x))

-- Define the theorem
theorem f_jump_discontinuity_at_zero :
  (∀ ε > 0, ∃ δ₁ > 0, ∀ x < 0, |x| < δ₁ → |f x - 0| < ε) ∧
  (∀ ε > 0, ∃ δ₂ > 0, ∀ x > 0, |x| < δ₂ → |f x - 1| < ε) →
  ¬ ContinuousAt f 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_jump_discontinuity_at_zero_l22_2276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l22_2257

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  ∃ (k : ℤ),
    (∀ x ∈ Set.Icc (- Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi), Monotone f) ∧
    (∀ x, f x = f (2 * (Real.pi / 3 + k * Real.pi / 2) - x)) ∧
    (f 0 = -1) ∧
    (f (Real.pi / 3) = 2) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), -1 ≤ f x ∧ f x ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l22_2257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_calculation_l22_2219

def initial_population : ℕ := 4079
def death_rate : ℚ := 5 / 100
def leaving_rate : ℚ := 15 / 100

noncomputable def current_population : ℕ :=
  let survived := initial_population - (↑initial_population * death_rate).floor.toNat
  (↑survived - (↑survived * leaving_rate).floor).toNat

theorem village_population_calculation :
  current_population = 3295 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_calculation_l22_2219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l22_2214

-- Define the curve C
def curve_C (α : ℝ) (x y : ℝ) : Prop :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

-- Define the line x = π/4
def line_x (x : ℝ) : Prop := x = Real.pi / 4

-- Define the chord length d
noncomputable def chord_length (α : ℝ) : ℝ :=
  let p₁ : ℝ × ℝ := (Real.arcsin α, Real.arcsin α)
  let p₂ : ℝ × ℝ := (Real.arccos α, -Real.arccos α)
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Theorem statement
theorem min_chord_length :
  ∀ α : ℝ, -1 ≤ α ∧ α ≤ 1 →
  chord_length α ≥ Real.pi / 2 ∧
  ∃ α₀ : ℝ, -1 ≤ α₀ ∧ α₀ ≤ 1 ∧ chord_length α₀ = Real.pi / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l22_2214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_radius_bound_l22_2269

/-- A circle C1 with center (0, 0) and radius R > 0 -/
def C1 (R : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = R^2}

/-- A line C2 with equation x - y - 2 = 0 -/
def C2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 - 2 = 0}

/-- The distance from (0, 0) to the line x - y - 2 = 0 -/
noncomputable def distance_to_line : ℝ := 2 / Real.sqrt 2

theorem intersection_implies_radius_bound
  (R : ℝ) (h_pos : R > 0) (h_intersect : (C1 R) ∩ C2 ≠ ∅) :
  R ≥ Real.sqrt 2 := by
  sorry

#check intersection_implies_radius_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_radius_bound_l22_2269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_location_l22_2232

theorem roots_location (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ (x y : ℝ), x ∈ Set.Ioo a b ∧ y ∈ Set.Ioo b c ∧
  (∀ z : ℝ, (z - a) * (z - b) + (z - b) * (z - c) + (z - c) * (z - a) = 0 → z = x ∨ z = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_location_l22_2232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oakdale_earnings_approx_l22_2255

/-- Represents the number of students from each school --/
structure StudentCounts where
  maplewood : ℕ
  oakdale : ℕ
  pinecrest : ℕ

/-- Represents the number of days worked by students from each school --/
structure WorkDays where
  maplewood : ℕ
  oakdale : ℕ
  pinecrest : ℕ

/-- Calculates the total earnings for Oakdale students --/
def oakdaleEarnings (counts : StudentCounts) (days : WorkDays) (totalPaid : ℚ) : ℚ :=
  let totalStudentDays := 
    counts.maplewood * days.maplewood + 
    counts.oakdale * days.oakdale + 
    counts.pinecrest * days.pinecrest
  let dailyWage := totalPaid / totalStudentDays
  (counts.oakdale * days.oakdale : ℚ) * dailyWage

/-- Theorem stating that Oakdale students' earnings are approximately $270.55 --/
theorem oakdale_earnings_approx (counts : StudentCounts) (days : WorkDays) 
    (h1 : counts.maplewood = 5)
    (h2 : counts.oakdale = 6)
    (h3 : counts.pinecrest = 8)
    (h4 : days.maplewood = 6)
    (h5 : days.oakdale = 4)
    (h6 : days.pinecrest = 7)
    (h7 : abs (oakdaleEarnings counts days 1240 - 270.55) < 0.01) : 
  abs (oakdaleEarnings counts days 1240 - 270.55) < 0.01 := by
  sorry

#eval oakdaleEarnings ⟨5, 6, 8⟩ ⟨6, 4, 7⟩ 1240

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oakdale_earnings_approx_l22_2255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l22_2262

/-- The minimum value of m for a point P satisfying given conditions -/
theorem min_m_value (P : ℝ × ℝ) (m : ℝ) : 
  let (x, y) := P
  (x - y + 2 ≥ 0) → 
  (x + y - 2 ≤ 0) → 
  (2 * y ≥ x + 2) → 
  (x^2 / 4 + y^2 = m^2) →
  (m > 0) →
  (∀ m' > 0, (∃ P' : ℝ × ℝ, 
    let (x', y') := P'
    (x' - y' + 2 ≥ 0) ∧ 
    (x' + y' - 2 ≤ 0) ∧ 
    (2 * y' ≥ x' + 2) ∧ 
    (x'^2 / 4 + y'^2 = m'^2)) → 
  m' ≥ m) →
  m = Real.sqrt 2 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l22_2262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l22_2270

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^3 - 4*x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -2 ∨ (-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2) ∨ 2 < x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l22_2270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_with_divisibility_constraints_l22_2202

theorem min_sum_with_divisibility_constraints (a b : ℕ) 
  (ha : a > 0)
  (hb : b > 0)
  (h1 : (79 : ℤ) ∣ (a + 77 * b))
  (h2 : (77 : ℤ) ∣ (a + 79 * b)) :
  (a : ℤ) + b ≥ 193 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_with_divisibility_constraints_l22_2202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_count_l22_2291

theorem unique_solution_count : ∃ (S : Finset ℕ), 
  (S.card = 4) ∧ 
  (∀ a : ℕ, a ∈ S ↔ 
    (∀ x : ℕ, x > 0 → 
      ((4 * x > 5 * x - 5 ∧ 4 * x > a - 8) ↔ x = 3))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_count_l22_2291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l22_2200

/-- Parabola represented by y = x^2 + bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- Theorem about the properties of a parabola -/
theorem parabola_properties (p : Parabola) :
  (∀ x, x^2 + p.b*x + p.c = (x-2)^2 + p.c - 4) →  -- axis of symmetry is x = 2
  (∃ y_min y_max, ∀ x, 1 ≤ x ∧ x ≤ 4 → 
    x^2 + p.b*x + p.c ≥ y_min ∧ 
    x^2 + p.b*x + p.c ≤ y_max ∧ 
    y_min + y_max = 6) →  -- sum of max and min is 6 when 1 ≤ x ≤ 4
  (∃! x, 1 < x ∧ x < 4 ∧ x^2 + p.b*x + p.c = 0) →  -- intersects x-axis at exactly one point when 1 < x < 4
  (p.b = -4 ∧ p.c = 5 ∧ (0 < p.c ∧ p.c ≤ 3 ∨ p.c = 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l22_2200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l22_2283

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ 2 + 2 * (8 : ℝ) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l22_2283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l22_2294

-- Define the curve
noncomputable def curve (x a : ℝ) : ℝ := 3 * x - Real.log (x + a)

-- Define the derivative of the curve
noncomputable def curve_derivative (x a : ℝ) : ℝ := 3 - 1 / (x + a)

theorem tangent_line_implies_a_value :
  ∀ a : ℝ,
  (curve_derivative 0 a = 2) →
  a = 1 := by
  intro a h
  -- Proof steps would go here
  sorry

#check tangent_line_implies_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l22_2294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_increase_l22_2272

/-- Given an initial production scenario and a new scenario with increased productivity,
    calculate the number of articles produced in the new scenario. -/
theorem production_increase 
  (a b c d p q r : ℕ) 
  (h_positive : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_initial : a * b * c > 0) 
  (h_new : p * q * r > 0) : 
  (2 * d * p * q * r : ℚ) / (a * b * c) = 
    (d : ℚ) / (a * b * c) * 2 * p * q * r := by
  -- Proof steps would go here
  sorry

#check production_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_increase_l22_2272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f₁_domain_f₂_l22_2280

-- Function 1
noncomputable def f₁ (x : ℝ) : ℝ := Real.sqrt (x - 2) * Real.sqrt (x + 2)

-- Function 2
noncomputable def f₂ (x : ℝ) : ℝ := Real.log (x + 1) / Real.sqrt (-x^2 - 3*x + 4)

-- Theorem for the domain of f₁
theorem domain_f₁ : 
  {x : ℝ | ∃ y, f₁ x = y} = {x : ℝ | x ≥ 2} := by
  sorry

-- Theorem for the domain of f₂
theorem domain_f₂ : 
  {x : ℝ | ∃ y, f₂ x = y} = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f₁_domain_f₂_l22_2280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_C_intersection_point_l22_2213

-- Define the parametric equations of line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, Real.sqrt 3 + Real.sqrt 3 * t)

-- Define the polar equation of curve C
def curve_C (ρ θ : ℝ) : Prop := Real.sin θ - Real.sqrt 3 * ρ * (Real.cos θ)^2 = 0

-- Theorem for the Cartesian equation of curve C
theorem cartesian_equation_C (x y : ℝ) :
  (∃ ρ θ, curve_C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ y - Real.sqrt 3 * x^2 = 0 := by
  sorry

-- Theorem for the intersection point
theorem intersection_point :
  ∃ t, curve_C 2 (π/3) ∧ line_l t = (1, Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_C_intersection_point_l22_2213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l22_2228

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

-- Define the domain
def domain : Set ℝ := {x | -Real.pi ≤ x ∧ x ≤ 0}

-- State the theorem
theorem monotone_increasing_interval :
  ∃ (a b : ℝ), a = -Real.pi / 6 ∧ b = 0 ∧
  (∀ x ∈ domain, f x ≤ f b) ∧
  (∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l22_2228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_lower_bound_l22_2258

/-- Represents a sequence of positive integers with specific properties -/
structure SpecialSequence where
  seq : ℕ → ℕ
  strictly_increasing : ∀ i j, i < j → seq i < seq j
  no_arithmetic_progression : ∀ i j k, i < j → j < k → seq j - seq i ≠ seq k - seq j

/-- Counts the number of terms in the sequence not exceeding x -/
noncomputable def A (s : SpecialSequence) (x : ℝ) : ℕ :=
  Finset.card (Finset.filter (fun i => ↑(s.seq i) ≤ x) (Finset.range (Nat.floor x + 1)))

/-- The main theorem to be proved -/
theorem exists_lower_bound (s : SpecialSequence) :
  ∃ (c K : ℝ), c > 1 ∧ K > 0 ∧ ∀ x > K, (A s x : ℝ) ≥ c * Real.sqrt x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_lower_bound_l22_2258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_log_difference_zero_l22_2261

open Real MeasureTheory

theorem probability_log_difference_zero :
  let E := {x ∈ Set.Ioo 0 1 | ⌊log 3 * x⌋ - ⌊log x⌋ = 0}
  volume E / volume (Set.Ioo 0 1 : Set ℝ) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_log_difference_zero_l22_2261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edens_average_speed_l22_2268

/-- Calculates the average speed of a two-segment trip -/
noncomputable def average_speed (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) : ℝ :=
  (distance1 + distance2) / (distance1 / speed1 + distance2 / speed2)

/-- Theorem: Eden's average speed for the entire trip is 20 miles per hour -/
theorem edens_average_speed :
  let distance1 : ℝ := 20
  let speed1 : ℝ := 15
  let distance2 : ℝ := 20
  let speed2 : ℝ := 30
  average_speed distance1 speed1 distance2 speed2 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_edens_average_speed_l22_2268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_travels_270_feet_check_lindy_distance_l22_2279

/-- Calculates the distance traveled by Lindy when Jack and Christina meet --/
noncomputable def lindy_distance (initial_distance : ℝ) (jack_speed : ℝ) (christina_speed : ℝ) (lindy_speed : ℝ) : ℝ :=
  let relative_speed := jack_speed + christina_speed
  let meeting_time := initial_distance / relative_speed
  lindy_speed * meeting_time

/-- Theorem stating that Lindy travels 270 feet when Jack and Christina meet --/
theorem lindy_travels_270_feet :
  lindy_distance 240 5 3 9 = 270 := by
  unfold lindy_distance
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- We can't use #eval with noncomputable functions, so let's use a theorem to check the result
theorem check_lindy_distance :
  lindy_distance 240 5 3 9 = 270 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_travels_270_feet_check_lindy_distance_l22_2279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_miami_to_newyork_distance_l22_2205

/-- The distance between two complex numbers -/
noncomputable def distance (z w : ℂ) : ℝ := Complex.abs (z - w)

theorem miami_to_newyork_distance :
  let new_york : ℂ := 0
  let san_francisco : ℂ := 3400 * I
  let miami : ℂ := 1020 + 1360 * I
  distance miami new_york = 1700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_miami_to_newyork_distance_l22_2205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l22_2238

/-- The transformed sine function -/
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem g_properties :
  (∃ (a b : ℝ), a ∈ Set.Icc 0 Real.pi ∧ b ∈ Set.Icc 0 Real.pi ∧ a ≠ b ∧ g a = 0 ∧ g b = 0 ∧
    ∀ c ∈ Set.Icc 0 Real.pi, g c = 0 → c = a ∨ c = b) ∧
  (∀ x : ℝ, g (Real.pi / 12 + x) = g (Real.pi / 12 - x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l22_2238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l22_2248

noncomputable def f (x : ℝ) := Real.exp (x * Real.log 3) + 3*x - 8

theorem root_in_interval :
  Continuous f →
  f 1 < 0 →
  f 1.5 > 0 →
  f 1.25 > 0 →
  ∃ x, x ∈ Set.Ioo 1 1.25 ∧ f x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l22_2248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_miles_february_l22_2267

/-- Represents the days of the week -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a lady in the walking group -/
inductive Lady
| Jamie
| Laura
| Melissa
| Katie
| Sue

def group_walk_distance : ℝ := 3
def group_walk_days_per_week : ℕ := 6
def weeks_in_february : ℕ := 4

/-- Returns the additional distance walked by a lady on a specific day -/
def additional_distance (lady : Lady) (day : Day) : ℝ :=
  match lady, day with
  | Lady.Jamie, Day.Monday | Lady.Jamie, Day.Wednesday | Lady.Jamie, Day.Friday => 2
  | Lady.Sue, Day.Tuesday | Lady.Sue, Day.Thursday | Lady.Sue, Day.Saturday => 1.5
  | Lady.Laura, Day.Monday | Lady.Laura, Day.Tuesday => 1
  | Lady.Laura, Day.Wednesday | Lady.Laura, Day.Thursday | Lady.Laura, Day.Friday => 1.5
  | Lady.Melissa, Day.Tuesday | Lady.Melissa, Day.Friday => 2
  | Lady.Melissa, Day.Sunday => 4
  | Lady.Katie, Day.Sunday => 3
  | Lady.Katie, _ => 1
  | _, _ => 0

/-- Calculates the total miles walked by all ladies in February -/
def total_miles_walked : ℝ :=
  let group_miles := group_walk_distance * (group_walk_days_per_week : ℝ) * (weeks_in_february : ℝ)
  let individual_miles (lady : Lady) := 
    (List.sum (List.map (additional_distance lady) [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday, Day.Saturday, Day.Sunday])) * (weeks_in_february : ℝ)
  group_miles + (individual_miles Lady.Jamie + individual_miles Lady.Laura + individual_miles Lady.Melissa + individual_miles Lady.Katie + individual_miles Lady.Sue)

theorem total_miles_february :
  total_miles_walked = 208 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_miles_february_l22_2267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_one_l22_2278

-- Define the function f(x) = 2ln x - x^2
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2

-- State the theorem
theorem f_max_at_one :
  ∀ x > 0, f x ≤ f 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_one_l22_2278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_48_plus_a_49_ninth_occurrence_of_5_l22_2233

def a : ℕ → ℕ
  | 0 => 0  -- Define for 0 to avoid division by zero
  | n + 1 => if (n + 1) % 2 = 1 then n + 1 else a ((n + 1) / 2)

def positions_of_5 : ℕ → ℕ
  | 0 => 5  -- Define for 0 to match ℕ type
  | n + 1 => 5 * 2^n

theorem a_48_plus_a_49 : a 48 + a 49 = 52 := by sorry

theorem ninth_occurrence_of_5 : positions_of_5 9 = 1280 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_48_plus_a_49_ninth_occurrence_of_5_l22_2233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_360_l22_2250

/-- The sum of the positive whole number divisors of 360 is 1170. -/
theorem sum_of_divisors_360 : (Finset.sum (Nat.divisors 360) id) = 1170 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_360_l22_2250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_of_angle_through_point_one_two_l22_2274

/-- For an angle whose terminal side passes through the point (1, 2), its tangent is 2. -/
theorem tangent_of_angle_through_point_one_two (α : Real) :
  (∃ (t : Real), t > 0 ∧ (t * 1, t * 2) ∈ {(x, y) | x = Real.cos α ∧ y = Real.sin α}) →
  Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_of_angle_through_point_one_two_l22_2274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_l22_2234

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem max_omega (ω : ℝ) (φ : ℝ) : 
  ω > 0 →
  0 < φ ∧ φ < Real.pi / 2 →
  f ω φ (-Real.pi / 4) = 0 →
  (∀ x, f ω φ (Real.pi / 4 - x) = f ω φ (Real.pi / 4 + x)) →
  (∀ x y, Real.pi / 18 < x ∧ x < y ∧ y < 2 * Real.pi / 9 → 
    (f ω φ x < f ω φ y ∨ f ω φ x > f ω φ y)) →
  ω ≤ 5 ∧ ∃ ω', ω' > 0 ∧ ω' ≤ 5 ∧ 
    (∀ ω'', ω'' > 0 → 
      ω'' ≤ 5 → 
      ∃ φ', 0 < φ' ∧ φ' < Real.pi / 2 ∧
        f ω'' φ' (-Real.pi / 4) = 0 ∧
        (∀ x, f ω'' φ' (Real.pi / 4 - x) = f ω'' φ' (Real.pi / 4 + x)) ∧
        (∀ x y, Real.pi / 18 < x ∧ x < y ∧ y < 2 * Real.pi / 9 → 
          (f ω'' φ' x < f ω'' φ' y ∨ f ω'' φ' x > f ω'' φ' y))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_l22_2234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_reachable_l22_2203

/-- Represents a circular arrangement of 12 numbers -/
def CircularArrangement := Fin 12 → Fin 12

/-- The initial arrangement of numbers -/
def initial_arrangement : CircularArrangement := fun i => i + 1

/-- The target arrangement of numbers -/
def target_arrangement : CircularArrangement :=
  fun i => match i with
  | 0 => 9
  | 11 => 10
  | 10 => 11
  | 9 => 12
  | _ => i + 1

/-- Applies a transformation to four adjacent numbers in the arrangement -/
def transform (arr : CircularArrangement) (start : Fin 12) : CircularArrangement :=
  fun i => if i ∈ [start, start + 1, start + 2, start + 3] then
             arr (start + 3 - (i - start))
           else
             arr i

/-- Theorem stating that the target arrangement is reachable from the initial arrangement -/
theorem target_reachable : ∃ (n : ℕ) (transforms : List (Fin 12)),
  (transforms.foldl transform initial_arrangement) = target_arrangement := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_reachable_l22_2203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_good_number_le_2011_l22_2208

/-- A natural number is "good" if it can be expressed as both the sum of two consecutive non-zero natural numbers and the sum of three consecutive non-zero natural numbers. -/
def is_good (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ n = m + (m + 1) ∧
  ∃ k : ℕ, k > 0 ∧ n = k + (k + 1) + (k + 2)

/-- The largest "good" number less than or equal to 2011 is 2007. -/
theorem largest_good_number_le_2011 :
  ∃ n : ℕ, is_good n ∧ n ≤ 2011 ∧ n = 2007 ∧ ∀ m : ℕ, is_good m ∧ m ≤ 2011 → m ≤ n :=
by
  -- We claim that 2007 is the largest good number ≤ 2011
  use 2007
  constructor
  · -- Prove that 2007 is good
    sorry
  constructor
  · -- Prove that 2007 ≤ 2011
    sorry
  constructor
  · -- Trivial: 2007 = 2007
    rfl
  · -- Prove that 2007 is the largest
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_good_number_le_2011_l22_2208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_center_of_mass_distance_l22_2263

/-- The distance from the center of the largest disk to the center of mass of an infinite series of disks -/
noncomputable def center_of_mass_distance (largest_radius : ℝ) (ratio : ℝ) : ℝ :=
  let mass_sum := (1 - ratio^2)⁻¹
  let weighted_sum := largest_radius * (1 - ratio^3)⁻¹
  weighted_sum / mass_sum

/-- Theorem: The distance from the center of the largest disk to the center of mass is 6/7 meters -/
theorem disk_center_of_mass_distance :
  center_of_mass_distance 2 (1/2) = 6/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_center_of_mass_distance_l22_2263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_at_radius_3_optimal_radius_is_3_l22_2282

/-- The volume of the cylinder -/
noncomputable def volume : ℝ := 27 * Real.pi

/-- The surface area of a cylinder given its radius -/
noncomputable def surface_area (r : ℝ) : ℝ := Real.pi * r^2 + 2 * Real.pi * r * (volume / (Real.pi * r^2))

/-- Theorem stating that the surface area is minimized when the radius is 3 -/
theorem min_surface_area_at_radius_3 :
  ∀ r : ℝ, r > 0 → surface_area 3 ≤ surface_area r := by
  sorry

/-- The optimal radius that minimizes the surface area -/
def optimal_radius : ℝ := 3

/-- Theorem stating that the optimal radius is indeed 3 -/
theorem optimal_radius_is_3 : optimal_radius = 3 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_at_radius_3_optimal_radius_is_3_l22_2282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ninth_term_l22_2211

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  r : ℝ
  h : ∀ n : ℕ, a (n + 1) = a n * r

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def GeometricSum (g : GeometricSequence) (n : ℕ) : ℝ :=
  if g.r = 1 then n * g.a 1
  else g.a 1 * (1 - g.r^n) / (1 - g.r)

/-- Theorem statement for the geometric sequence problem -/
theorem geometric_sequence_ninth_term
  (g : GeometricSequence)
  (h1 : g.a 3 = 2)
  (h2 : GeometricSum g 12 = 4 * GeometricSum g 6) :
  g.a 9 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ninth_term_l22_2211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_unique_x_l22_2235

/-- A sequence of non-negative integers satisfying the given conditions -/
def SequenceA : ℕ → ℕ := sorry

/-- The condition that the sequence satisfies for all i, j ≥ 1 with i+j ≤ 1997 -/
axiom sequence_condition : ∀ i j : ℕ, i ≥ 1 → j ≥ 1 → i + j ≤ 1997 →
  SequenceA i + SequenceA j ≤ SequenceA (i + j) ∧
  SequenceA (i + j) ≤ SequenceA i + SequenceA j + 1

/-- The theorem to be proved -/
theorem existence_of_unique_x : ∃! x : ℝ, ∀ n : ℕ, 1 ≤ n → n ≤ 1997 →
  SequenceA n = ⌊n * x⌋ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_unique_x_l22_2235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_gt_5_range_a_no_solution_l22_2243

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Theorem for the solution set of f(x) > 5
theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -3 ∨ x > 2} := by sorry

-- Theorem for the range of a where f(x) < a has no solution
theorem range_a_no_solution :
  {a : ℝ | ∀ x, f x ≥ a} = Set.Iic 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_gt_5_range_a_no_solution_l22_2243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_approx_l22_2206

/-- The value of the expression ( x / √y ) - ( √a / b^2 ) + [ ( √c / √d ) / (3 * e) ] 
    given the specified conditions -/
noncomputable def expression_value : ℝ :=
  let x := (Real.sqrt 1.21) ^ 3
  let y := (Real.sqrt 0.81) ^ 2
  let a := 4 * (Real.sqrt 0.81)
  let b := 2 * (Real.sqrt 0.49)
  let c := 3 * (Real.sqrt 1.21)
  let d := 2 * (Real.sqrt 0.49)
  let e := (Real.sqrt 0.81) ^ 4
  (x / Real.sqrt y) - (Real.sqrt a / b^2) + ((Real.sqrt c / Real.sqrt d) / (3 * e))

/-- The expression value is approximately equal to 1.291343 -/
theorem expression_value_approx :
  abs (expression_value - 1.291343) < 0.000001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_approx_l22_2206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_distance_problem_l22_2293

/-- The distance traveled by a dog running between two people moving towards each other -/
noncomputable def dogDistance (initialDistance : ℝ) (speedA speedB speedDog : ℝ) : ℝ :=
  (initialDistance * speedDog) / (speedA + speedB)

theorem dog_distance_problem :
  let initialDistance : ℝ := 22.5
  let speedA : ℝ := 2.5
  let speedB : ℝ := 5
  let speedDog : ℝ := 7.5
  dogDistance initialDistance speedA speedB speedDog = 22.5 := by
  -- Unfold the definition of dogDistance
  unfold dogDistance
  -- Simplify the expression
  simp
  -- The proof is completed
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_distance_problem_l22_2293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_eight_primes_units_3_l22_2212

/-- A function that returns true if a number is prime and has a units digit of 3 -/
def isPrimeWithUnits3 (n : ℕ) : Bool :=
  Nat.Prime n ∧ n % 10 = 3

/-- The list of the first eight prime numbers with a units digit of 3 -/
def firstEightPrimesWithUnits3 : List ℕ :=
  (List.range 1000).filter (fun n => isPrimeWithUnits3 n) |>.take 8

theorem sum_first_eight_primes_units_3 :
  firstEightPrimesWithUnits3.sum = 404 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_eight_primes_units_3_l22_2212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hollow_sphere_thickness_l22_2260

/-- Represents a hollow sphere -/
structure HollowSphere where
  outer_radius : ℝ
  inner_radius : ℝ
  specific_gravity : ℝ

/-- Calculates the volume of a sphere given its radius -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

/-- Calculates the weight of a hollow sphere -/
noncomputable def sphere_weight (s : HollowSphere) : ℝ :=
  s.specific_gravity * (sphere_volume s.outer_radius - sphere_volume s.inner_radius)

/-- Calculates the submerged volume of a floating object -/
noncomputable def submerged_volume (weight : ℝ) (water_density : ℝ) : ℝ := weight / water_density

/-- Theorem about the thickness of a hollow iron sphere -/
theorem hollow_sphere_thickness 
  (s : HollowSphere) 
  (h_weight : sphere_weight s = 3012)
  (h_submerged : submerged_volume 3012 1 = (3/4) * sphere_volume s.outer_radius)
  (h_gravity : s.specific_gravity = 7.5) :
  ∃ (ε : ℝ), abs (s.outer_radius - s.inner_radius - 0.34) < ε ∧ ε > 0 := by
  sorry

#check hollow_sphere_thickness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hollow_sphere_thickness_l22_2260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_angle_l22_2287

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Creates a Line from a point and an angle -/
noncomputable def lineFromPointAndAngle (x y angle : ℝ) : Line :=
  { slope := Real.tan angle,
    yIntercept := y - (Real.tan angle) * x }

/-- Checks if a point lies on a line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- Converts a Line to its general form equation coefficients -/
def Line.toGeneralForm (l : Line) : (ℝ × ℝ × ℝ) :=
  (l.slope, -1, l.yIntercept)

theorem line_through_point_with_angle (x₀ y₀ angle : ℝ) :
  let l := lineFromPointAndAngle x₀ y₀ angle
  let (a, b, c) := l.toGeneralForm
  (a = 1 ∧ b = -1 ∧ c = 3) ∧ l.containsPoint (-1) 2 ∧ angle = π/4 := by
  sorry

#check line_through_point_with_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_angle_l22_2287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_imply_triangle_and_inequality_l22_2201

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 - 4 * b * x + 4 * c

-- State the theorem
theorem quadratic_roots_imply_triangle_and_inequality 
  (a b c : ℝ) 
  (ha : a > 0)
  (hf : ∃ x y, 2 ≤ x ∧ x < y ∧ y ≤ 3 ∧ f a b c x = 0 ∧ f a b c y = 0) :
  (∃ s1 s2 s3 : ℝ, s1 = a ∧ s2 = b ∧ s3 = c ∧ 
    s1 + s2 > s3 ∧ s2 + s3 > s1 ∧ s3 + s1 > s2) ∧
  (a / (a + c) + b / (b + a) > c / (b + c)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_imply_triangle_and_inequality_l22_2201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l22_2256

/-- The function f(x) = sin(2x - π/4) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

/-- The domain of the function -/
def domain : Set ℝ := Set.Icc 0 (Real.pi / 2)

/-- The proposition that the graph of f intersects y = a at two points in the domain -/
def intersects_twice (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ∈ domain ∧ x₂ ∈ domain ∧ x₁ ≠ x₂ ∧ f x₁ = a ∧ f x₂ = a

/-- The theorem stating that if the graph intersects y = a twice in the domain,
    then the sum of the x-coordinates of the intersection points is 3π/4 -/
theorem intersection_sum (a : ℝ) :
  intersects_twice a → ∃ x₁ x₂, x₁ + x₂ = 3 * Real.pi / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l22_2256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l22_2224

def mySequence (n : ℕ) : ℤ := 
  5 * n - 2

theorem sequence_formula (n : ℕ) : mySequence n = 5 * n - 2 := by
  rfl

#eval mySequence 1  -- Should output 3
#eval mySequence 2  -- Should output 8
#eval mySequence 3  -- Should output 13
#eval mySequence 4  -- Should output 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l22_2224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l22_2239

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral where
  /-- Point A -/
  A : ℝ × ℝ
  /-- Point B -/
  B : ℝ × ℝ
  /-- Point C -/
  C : ℝ × ℝ
  /-- Point D -/
  D : ℝ × ℝ
  /-- Point E (intersection of diagonals) -/
  E : ℝ × ℝ
  /-- Angle ABC is 90 degrees -/
  angle_ABC_right : (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0
  /-- Angle ACD is 90 degrees -/
  angle_ACD_right : (C.1 - A.1) * (C.1 - D.1) + (C.2 - A.2) * (C.2 - D.2) = 0
  /-- Length of AC is 25 -/
  AC_length : ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 25^2
  /-- Length of CD is 40 -/
  CD_length : ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 40^2
  /-- E is on AC -/
  E_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))
  /-- E is on BD -/
  E_on_BD : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (B.1 + s * (D.1 - B.1), B.2 + s * (D.2 - B.2))
  /-- Length of AE is 10 -/
  AE_length : ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 10^2

/-- The area of the quadrilateral ABCD is 750 -/
theorem quadrilateral_area (q : Quadrilateral) : 
  (1/2) * |((q.B.1 - q.D.1) * (q.A.2 - q.C.2) - (q.A.1 - q.C.1) * (q.B.2 - q.D.2))| = 750 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l22_2239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_third_expression_is_perfect_square_trinomial_l22_2241

/-- Represents a polynomial expression -/
structure PolynomialExpression where
  coeffs : List ℚ
  deriving Repr

/-- Checks if a polynomial expression is a perfect square trinomial -/
def isPerfectSquareTrinomial (p : PolynomialExpression) : Prop :=
  ∃ a b : ℚ, p.coeffs = [a^2, 2*a*b, b^2]

/-- The list of polynomial expressions to check -/
def expressions : List PolynomialExpression :=
  [
    ⟨[4, 4, 4]⟩,      -- 4x^2 + 4x + 4
    ⟨[-1, 4, 4]⟩,     -- -x^2 + 4x + 4
    ⟨[1, 0, -4, 0, 4]⟩, -- x^4 - 4x^2 + 4
    ⟨[-1, 0, -4]⟩     -- -x^2 - 4
  ]

theorem only_third_expression_is_perfect_square_trinomial :
  ∃! i : Fin 4, isPerfectSquareTrinomial (expressions[i]) ∧ i.val = 2 :=
by
  sorry

#eval expressions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_third_expression_is_perfect_square_trinomial_l22_2241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2022_less_than_b_2022_l22_2246

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => 2 / (2 + sequence_a n)

noncomputable def sequence_b : ℕ → ℝ
  | 0 => 1
  | n + 1 => 3 / (3 + sequence_b n)

theorem a_2022_less_than_b_2022 : sequence_a 2022 < sequence_b 2022 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2022_less_than_b_2022_l22_2246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l22_2285

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the derivative relationship
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Define f as an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- Define the inequality condition
axiom inequality_condition : ∀ x ∈ Set.Ioo (-Real.pi/2) 0, f x < f' x * Real.tan x

-- Theorem to prove
theorem f_inequality : f (-Real.pi/3) > -Real.sqrt 3 * f (Real.pi/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l22_2285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_pasture_capacity_l22_2252

/-- Represents a pasture with its characteristics -/
structure Pasture where
  acres : ℝ
  cows : ℝ
  days : ℝ

/-- Calculates the number of cows a pasture can feed for a given number of days -/
noncomputable def cowsSupported (p : Pasture) (grassPerAcre : ℝ) (grassGrowthRate : ℝ) (daysToFeed : ℝ) : ℝ :=
  (p.acres * grassPerAcre + p.acres * daysToFeed * grassGrowthRate) / daysToFeed

theorem third_pasture_capacity 
  (p1 p2 p3 : Pasture)
  (h1 : p1.acres = 33 ∧ p1.cows = 22 ∧ p1.days = 27)
  (h2 : p2.acres = 28 ∧ p2.cows = 17 ∧ p2.days = 42)
  (h3 : p3.acres = 10)
  (uniform_growth : ∃ (g : ℝ), ∀ p : Pasture, cowsSupported p g g p.days = p.cows)
  : ∃ (g : ℝ), cowsSupported p3 g g 3 = 20 := by
  sorry

#check third_pasture_capacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_pasture_capacity_l22_2252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_l22_2298

/-- Represents the number of towns in the kingdom -/
def num_towns : ℕ := 2021

/-- Represents the number of towns each road connects to clockwise -/
def road_length : ℕ := 101

/-- A type representing the towns in the kingdom -/
def Town := Fin num_towns

/-- A function representing the coloring of roads -/
def road_color := Town → Fin road_length → ℕ

/-- A path between two towns -/
def path (a b : Town) := List Town

/-- Predicate to check if a path is valid (follows the road rules) -/
def valid_path {a b : Town} (p : path a b) : Prop := sorry

/-- Predicate to check if a path has no repeated colors -/
def no_repeated_colors {a b : Town} (p : path a b) (coloring : road_color) : Prop := sorry

/-- The main theorem stating the minimal number of colors required -/
theorem min_colors : 
  (∃ (coloring : road_color), 
    (∀ (a b : Town), ∃ (p : path a b), valid_path p ∧ no_repeated_colors p coloring)) →
  (∀ (coloring : road_color),
    (∀ (a b : Town), ∃ (p : path a b), valid_path p ∧ no_repeated_colors p coloring) →
    (∃ (n : ℕ), n ≥ 21 ∧ ∀ (i : Town) (j : Fin road_length), coloring i j < n)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_l22_2298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_between_zero_and_two_l22_2271

/-- A random variable following a normal distribution with mean 2 and standard deviation σ -/
noncomputable def ξ : Real → Prop := sorry

/-- The probability density function of ξ -/
noncomputable def pdf_ξ : Real → Real := sorry

/-- The cumulative distribution function of ξ -/
noncomputable def cdf_ξ : Real → Real := sorry

/-- The condition that P(ξ < 0) = 0.08 -/
axiom prob_less_than_zero : cdf_ξ 0 = 0.08

/-- The symmetry of the normal distribution around its mean -/
axiom symmetry : ∀ x, pdf_ξ (2 - x) = pdf_ξ (2 + x)

theorem prob_between_zero_and_two : cdf_ξ 2 - cdf_ξ 0 = 0.42 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_between_zero_and_two_l22_2271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_pi_sixth_sufficient_not_necessary_l22_2226

theorem alpha_pi_sixth_sufficient_not_necessary :
  (∃ α : ℝ, α = π / 6 → Real.cos (2 * α) = 1 / 2) ∧
  (∃ α : ℝ, Real.cos (2 * α) = 1 / 2 ∧ α ≠ π / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_pi_sixth_sufficient_not_necessary_l22_2226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_sum_l22_2286

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem simple_interest_sum (x : ℝ) : 
  simple_interest x 8 5 = (1/2) * compound_interest 8000 15 2 → x = 3225 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_sum_l22_2286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_simplified_fraction_l22_2264

theorem decimal_to_simplified_fraction :
  ∃ (n d : ℤ), d ≠ 0 ∧ (3675 : ℚ) / 1000 = n / d ∧ Int.gcd n d = 1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_simplified_fraction_l22_2264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_fraction_sum_l22_2209

/-- The number of rectangles on a 9x9 checkerboard -/
def num_rectangles : ℕ := 2025

/-- The number of squares on a 9x9 checkerboard -/
def num_squares : ℕ := 285

/-- The fraction of squares to rectangles -/
def square_rectangle_fraction : ℚ := num_squares / num_rectangles

theorem checkerboard_fraction_sum :
  (square_rectangle_fraction.num + square_rectangle_fraction.den = 154) ∧
  (square_rectangle_fraction = 19 / 135) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_fraction_sum_l22_2209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l22_2266

/-- Definition of a hyperbola with given parameters -/
noncomputable def Hyperbola (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / 8) = 1}

/-- The right focus of the hyperbola -/
def RightFocus : ℝ × ℝ := (3, 0)

/-- The eccentricity of a hyperbola -/
noncomputable def Eccentricity (a : ℝ) : ℝ := 3 / a

/-- Theorem: The eccentricity of the given hyperbola is 3 -/
theorem hyperbola_eccentricity :
  ∃ a : ℝ, a > 0 ∧ RightFocus ∈ Hyperbola a ∧ Eccentricity a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l22_2266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_three_element_set_l22_2254

theorem number_of_subsets_of_three_element_set {α : Type*} [Fintype α] [DecidableEq α] (M : Finset α) (h : M.card = 3) :
  (Finset.powerset M).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_three_element_set_l22_2254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_touches_curve_at_two_points_l22_2290

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3

/-- The line function -/
noncomputable def g (x : ℝ) : ℝ := -8/9 * x - 4/27

/-- Theorem stating that the line touches the curve at two distinct points -/
theorem line_touches_curve_at_two_points :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ ∧
  ∀ x, f x ≥ g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_touches_curve_at_two_points_l22_2290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l22_2218

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 + 2*a else -x

theorem a_range (a : ℝ) (h1 : a < 0) (h2 : f a (1 - a) ≥ f a (1 + a)) :
  -2 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l22_2218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_direction_vector_l22_2295

-- Define the two points on the line
def point1 : Fin 3 → ℝ := ![(-5), 0, 1]
def point2 : Fin 3 → ℝ := ![1, 4, 3]

-- Define the direction vector form
def direction_vector (b c : ℝ) : Fin 3 → ℝ := ![2, b, c]

-- Theorem statement
theorem line_direction_vector : 
  ∃ (b c : ℝ), 
    (λ i => point2 i - point1 i) = direction_vector b c ∧ 
    b = 4/3 ∧ 
    c = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_direction_vector_l22_2295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base4_calculation_l22_2288

/-- Represents a number in base 4 --/
structure Base4 where
  value : ℕ
  isValid : value < 4^64 := by sorry

/-- Converts a base 4 number to base 10 --/
def to_base10 (n : Base4) : ℕ := n.value

/-- Converts a base 10 number to base 4 --/
def to_base4 (n : ℕ) : Base4 := ⟨n % (4^64), by sorry⟩

/-- Subtraction in base 4 --/
def sub_base4 (a b : Base4) : Base4 := 
  to_base4 (to_base10 a - to_base10 b)

/-- Multiplication in base 4 --/
def mul_base4 (a b : Base4) : Base4 := 
  to_base4 (to_base10 a * to_base10 b)

/-- Division in base 4 --/
def div_base4 (a b : Base4) : Base4 := 
  to_base4 (to_base10 a / to_base10 b)

/-- Convert a natural number to Base4 --/
def ofNat (n : ℕ) : Base4 := to_base4 n

instance : OfNat Base4 n where
  ofNat := ofNat n

/-- The main theorem to prove --/
theorem base4_calculation : 
  div_base4 (mul_base4 (sub_base4 120 21) 13) 3 = 203 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base4_calculation_l22_2288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l22_2236

theorem inequality_proof (a b c : ℝ) (m n : ℕ)
  (hab : a > b) (hbc : b > c) (hc : c > 0) (hmn : m > n) :
  a^m * b^n > c^n * a^m ∧ c^n * a^m > b^n * c^m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l22_2236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_count_l22_2207

/-- Represents a coloring of a 3x4 grid --/
def GridColoring := Fin 3 → Fin 4 → Bool

/-- Checks if a rectangle in the grid has all vertices of the same color --/
def hasMonochromaticRectangle (coloring : GridColoring) : Prop :=
  ∃ (r1 r2 : Fin 3) (c1 c2 : Fin 4),
    r1 < r2 ∧ c1 < c2 ∧
    coloring r1 c1 = coloring r1 c2 ∧
    coloring r1 c1 = coloring r2 c1 ∧
    coloring r1 c1 = coloring r2 c2

/-- The set of valid colorings --/
def ValidColorings : Set GridColoring :=
  { coloring | ¬hasMonochromaticRectangle coloring }

/-- Assume ValidColorings is finite --/
instance : Fintype ValidColorings := sorry

theorem valid_colorings_count :
  Fintype.card ValidColorings = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_count_l22_2207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_color_has_infinite_multiples_l22_2216

/-- A coloring of the integers using two colors -/
def Coloring := ℤ → Fin 2

/-- The property that a color has infinitely many multiples of every natural number -/
def HasInfiniteMultiples (c : Coloring) (color : Fin 2) :=
  ∀ k : ℕ, ∀ N : ℕ, ∃ n : ℤ, n > N ∧ (k : ℤ) ∣ n ∧ c n = color

/-- The main theorem: at least one color has infinite multiples of every natural number -/
theorem at_least_one_color_has_infinite_multiples (c : Coloring) :
  HasInfiniteMultiples c 0 ∨ HasInfiniteMultiples c 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_color_has_infinite_multiples_l22_2216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sin_minus_cos_l22_2253

open Real

theorem range_of_sin_minus_cos :
  ∀ x : ℝ, x ∈ Set.Icc (π / 2) (3 * π / 4) →
    (∃ y ∈ Set.Icc (π / 2) (3 * π / 4), Real.sin y - Real.cos y = 0) ∧
    (∃ z ∈ Set.Icc (π / 2) (3 * π / 4), Real.sin z - Real.cos z = Real.sqrt 2) ∧
    (∀ w ∈ Set.Icc (π / 2) (3 * π / 4), 0 ≤ Real.sin w - Real.cos w ∧ Real.sin w - Real.cos w ≤ Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sin_minus_cos_l22_2253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_sine_l22_2251

theorem area_bounded_by_sine : 
  ∫ x in (-π/2)..(-π/4), Real.sin x = π/4 - Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_sine_l22_2251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_total_amount_l22_2227

/-- Calculates the total amount received from book sales given the specified conditions. -/
theorem book_sale_total_amount : 
  let total_books : ℚ := 450
  let novel_percentage : ℚ := 40 / 100
  let nonfiction_percentage : ℚ := 35 / 100
  let childrens_percentage : ℚ := 25 / 100
  let novel_price : ℚ := 4
  let nonfiction_price : ℚ := 7/2
  let childrens_price : ℚ := 5/2
  let nonfiction_sold_ratio : ℚ := 4/5
  let childrens_sold_ratio : ℚ := 3/4

  let novels_count : ℚ := ⌊total_books * novel_percentage⌋
  let nonfiction_count : ℚ := ⌊total_books * nonfiction_percentage⌋
  let childrens_count : ℚ := ⌊total_books * childrens_percentage⌋

  let novels_sold : ℚ := novels_count
  let nonfiction_sold : ℚ := ⌊nonfiction_count * nonfiction_sold_ratio⌋
  let childrens_sold : ℚ := ⌊childrens_count * childrens_sold_ratio⌋

  let total_amount : ℚ := 
    novels_sold * novel_price + 
    nonfiction_sold * nonfiction_price + 
    childrens_sold * childrens_price

  total_amount = 2735 / 2 := by sorry

#eval (2735 : ℚ) / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_total_amount_l22_2227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_similar_triangles_l22_2289

open Geometry

-- Define the triangles ABC and DEF
variable (A B C D E F : EuclideanSpace ℝ (Fin 2))

-- Define the similarity of triangles
def similar_triangles (ABC DEF : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∃ k : ℝ, k > 0 ∧ ∀ (X Y : EuclideanSpace ℝ (Fin 2)), 
    (X ∈ ABC ∧ Y ∈ DEF) → 
    dist X Y = k * dist X Y

-- Define the largest angle in a triangle
noncomputable def largest_angle (t : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := 
  sorry -- Placeholder for the actual implementation

-- Theorem statement
theorem largest_angle_in_similar_triangles 
  (ABC DEF : Set (EuclideanSpace ℝ (Fin 2))) 
  (h_similar : similar_triangles ABC DEF) 
  (h_ratio : ∃ (k : ℝ), k = 3/2 ∧ ∀ (X Y : EuclideanSpace ℝ (Fin 2)), 
    (X ∈ ABC ∧ Y ∈ DEF) → 
    dist X Y = k * dist X Y)
  (h_largest_ABC : largest_angle ABC = 110 * π / 180) : 
  largest_angle DEF = 110 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_similar_triangles_l22_2289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_fixed_l22_2222

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line that intersects the ellipse
def intersecting_line (m x y : ℝ) : Prop := x = m*y + 1

-- Define a point on the ellipse
def point_on_ellipse (m : ℝ) (P : ℝ × ℝ) : Prop :=
  ellipse_C P.1 P.2 ∧ intersecting_line m P.1 P.2

-- Define the reflection of a point across the x-axis
def reflect_x (P : ℝ × ℝ) : ℝ × ℝ := (P.1, -P.2)

-- Define the line passing through two points
def line_through_points (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - A.2) * (B.1 - A.1) = (x - A.1) * (B.2 - A.2)

-- Theorem statement
theorem intersection_point_fixed (m : ℝ) (P Q : ℝ × ℝ) :
  point_on_ellipse m P ∧ point_on_ellipse m Q ∧ P ≠ Q ∧ reflect_x P ≠ Q →
  ∃ (x : ℝ), line_through_points (reflect_x P) Q x 0 ∧ x = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_fixed_l22_2222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_odd_and_period_l22_2277

-- Define the function f(x) = tan(x/2)
noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 2)

-- Theorem stating that f is an odd function with period 2π
theorem tan_half_odd_and_period : 
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2 * Real.pi) = f x) := by
  sorry

#check tan_half_odd_and_period

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_odd_and_period_l22_2277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_is_correct_l22_2281

/-- The cosine of the acute angle between two lines with direction vectors (4, 5) and (2, 6) -/
noncomputable def cos_angle_between_lines : ℝ :=
  let v1 : Fin 2 → ℝ := ![4, 5]
  let v2 : Fin 2 → ℝ := ![2, 6]
  let dot_product := (v1 0) * (v2 0) + (v1 1) * (v2 1)
  let magnitude1 := Real.sqrt ((v1 0) ^ 2 + (v1 1) ^ 2)
  let magnitude2 := Real.sqrt ((v2 0) ^ 2 + (v2 1) ^ 2)
  dot_product / (magnitude1 * magnitude2)

/-- The theorem stating that the cosine of the angle between the lines is 38 / sqrt(1640) -/
theorem cos_angle_is_correct : cos_angle_between_lines = 38 / Real.sqrt 1640 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_is_correct_l22_2281
