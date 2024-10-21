import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l3_335

noncomputable def f (x : ℝ) : ℝ := abs x / x

noncomputable def g (x : ℝ) : ℝ :=
  if x > 0 then 1 else -1

theorem f_eq_g : ∀ x : ℝ, x ≠ 0 → f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l3_335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l3_322

/-- Triangle ABC with internal angles A, B, C and corresponding sides a, b, c -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Vector in 2D space -/
structure Vector2D where
  x : Real
  y : Real

noncomputable def Triangle.area (t : Triangle) : Real :=
  1/2 * t.a * t.b * Real.sin t.C

theorem triangle_proof (t : Triangle) 
  (hm : Vector2D) (hn : Vector2D)
  (h_angles : t.A + t.B + t.C = Real.pi)
  (h_m : hm = ⟨-1, Real.sqrt 3⟩)
  (h_n : hn = ⟨Real.cos t.A, Real.sin t.A⟩)
  (h_dot : hm.x * hn.x + hm.y * hn.y = 1)
  (h_c : t.c = Real.sqrt 5)
  (h_cos_ratio : Real.cos t.B / Real.cos t.C = t.b / t.c) :
  t.A = Real.pi / 3 ∧ t.area = 5 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l3_322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_four_l3_337

-- Define the complex number type using the existing Complex type
open Complex

-- Define addition for complex numbers (already defined in Complex)
-- Define multiplication for complex numbers (already defined in Complex)

-- Define exponentiation for complex numbers
def complex_pow (z : ℂ) (n : ℕ) : ℂ :=
  z ^ n

-- State the theorem
theorem complex_power_four :
  complex_pow (2 + I) 4 = -7 + 24 * I :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_four_l3_337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l3_345

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ a + b + c) ∧
  (a + b + c = 1 → 2 * a * b / (a + b) + 2 * b * c / (b + c) + 2 * a * c / (a + c) ≤ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l3_345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equilateral_triangle_l3_321

/-- Given an ellipse with semi-major axis a and semi-minor axis b, where a > b > 0,
    if a perpendicular line from the left focus to the x-axis intersects the ellipse at points P and Q
    such that triangle PQF₂ is equilateral (where F₂ is the right focus),
    then the eccentricity of the ellipse is √3/3. -/
theorem ellipse_eccentricity_equilateral_triangle 
  (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0) : 
  let e := Real.sqrt (1 - b^2 / a^2)
  ∃ (P Q : ℝ × ℝ) (F₂ : ℝ × ℝ),
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧ 
    (Q.1^2 / a^2 + Q.2^2 / b^2 = 1) ∧
    (F₂.1 = e * a) ∧ (F₂.2 = 0) ∧
    (P.1 = Q.1) ∧ 
    (((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt = ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2).sqrt) ∧ 
    (((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2).sqrt = ((Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2).sqrt) →
    e = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equilateral_triangle_l3_321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lulu_saved_amount_l3_365

/-- Represents the savings of Lulu in dollars -/
def lulu_savings : ℚ := sorry

/-- Represents the savings of Nora in dollars -/
def nora_savings : ℚ := sorry

/-- Represents the savings of Tamara in dollars -/
def tamara_savings : ℚ := sorry

/-- Represents the total debt to be paid off -/
def debt : ℚ := 40

/-- Represents the amount each girl received after paying the debt -/
def individual_remainder : ℚ := 2

/-- The relation between Nora's and Lulu's savings -/
axiom nora_lulu_relation : 5 * lulu_savings = nora_savings

/-- The relation between Nora's and Tamara's savings -/
axiom nora_tamara_relation : 3 * tamara_savings = nora_savings

/-- The total savings equation -/
axiom total_savings_equation : 
  lulu_savings + nora_savings + tamara_savings = debt + 3 * individual_remainder

/-- Theorem stating Lulu's savings -/
theorem lulu_saved_amount : lulu_savings = 138 / 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lulu_saved_amount_l3_365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_80_l3_329

def scores : List ℤ := [71, 76, 80, 82, 91]

def is_integer_average (partial_scores : List ℤ) : Prop :=
  ∀ i : Fin partial_scores.length, (partial_scores.take (i + 1)).sum % (i + 1) = 0

def is_valid_order (order : List ℤ) : Prop :=
  order.length = 5 ∧ 
  order.toFinset = scores.toFinset ∧
  is_integer_average order

theorem last_score_is_80 :
  ∀ order : List ℤ, is_valid_order order → order.getLast? = some 80 := by
  sorry

#check last_score_is_80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_80_l3_329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_two_roots_l3_344

theorem equation_two_roots :
  ∃ (a b : ℝ), a ≠ b ∧ 
  (∀ x : ℝ, Real.sqrt (6 - x) = x * Real.sqrt (6 - x) ↔ x = a ∨ x = b) ∧
  (∀ x : ℝ, Real.sqrt (6 - x) = x * Real.sqrt (6 - x) → x = a ∨ x = b) :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_two_roots_l3_344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_flips_theorem_l3_386

/-- Represents the possible outcomes of a coin flip -/
inductive CoinOutcome
  | Heads
  | Tails
  | Middle

/-- The sequence to be observed -/
def target_sequence : List (List CoinOutcome) := List.replicate 2016 [CoinOutcome.Heads, CoinOutcome.Middle, CoinOutcome.Middle, CoinOutcome.Tails]

/-- The probability of each outcome -/
def coin_probability : CoinOutcome → ℚ 
  | _ => 1 / 3

/-- The expected number of flips to observe the target sequence -/
noncomputable def expected_flips : ℚ := (3^8068 - 81) / 80

/-- Theorem stating the expected number of flips -/
theorem expected_flips_theorem :
  expected_flips = (3^8068 - 81) / 80 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_flips_theorem_l3_386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l3_336

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a/x

-- State the theorem
theorem sufficient_not_necessary_condition (a : ℝ) :
  (0 < a ∧ a < 2) →
  (∀ x y, 1 < x ∧ x < y → f a x < f a y) ∧
  ¬ (∀ a, (∀ x y, 1 < x ∧ x < y → f a x < f a y) → (0 < a ∧ a < 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l3_336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l3_300

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the given conditions
def satisfies_conditions (t : Triangle) : Prop :=
  2 * t.c * Real.cos t.A = t.a * Real.cos t.B + t.b * Real.cos t.A ∧
  t.a + t.b + t.c = 3 * Real.sqrt 3 ∧
  2 * Real.sin t.A = t.a -- Radius of circumcircle is 1

-- Theorem to prove
theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_conditions t) : 
  t.A = Real.pi/3 ∧ 
  t.a = t.b ∧ t.b = t.c ∧ 
  (t.a * t.b * Real.sin t.A) / 4 = (3 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l3_300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_for_all_reals_l3_370

/-- The function f(x) defined for all real x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.rpow (x - 7) (1/3 : ℝ)) / (a * x^2 + 4 * a * x + 3)

/-- The theorem stating the range of a for which f is defined for all real x -/
theorem f_defined_for_all_reals (a : ℝ) : 
  (∀ x : ℝ, (a * x^2 + 4 * a * x + 3) ≠ 0) ↔ (0 < a ∧ a < 3/4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_for_all_reals_l3_370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parrot_count_l3_306

noncomputable def number_of_parrot_legs (p : ℕ) : ℕ := p * 2
noncomputable def number_of_parrot_heads (p : ℕ) : ℕ := p
noncomputable def number_of_rabbit_legs (r : ℕ) : ℕ := r * 4
noncomputable def number_of_rabbit_heads (r : ℕ) : ℕ := r

theorem parrot_count (total_legs total_heads : ℕ) 
  (h_legs : total_legs = 26) 
  (h_heads : total_heads = 10) 
  (h_parrot_legs : ∀ p : ℕ, p * 2 = number_of_parrot_legs p)
  (h_parrot_heads : ∀ p : ℕ, p = number_of_parrot_heads p)
  (h_rabbit_legs : ∀ r : ℕ, r * 4 = number_of_rabbit_legs r)
  (h_rabbit_heads : ∀ r : ℕ, r = number_of_rabbit_heads r) :
  ∃ p r : ℕ, 
    p + r = total_heads ∧ 
    number_of_parrot_legs p + number_of_rabbit_legs r = total_legs ∧
    p = 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parrot_count_l3_306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2010_equals_2_l3_318

def sequence_a : ℕ → ℚ
  | 0 => 1/2  -- Add this case for 0
  | 1 => 1/2
  | n + 2 => 1 - 1 / sequence_a (n + 1)

theorem a_2010_equals_2 : sequence_a 2010 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2010_equals_2_l3_318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_transformation_l3_315

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 + 4

-- Define the transformation operations
def rotate_90_ccw (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ -f x + 7

def shift_right (f : ℝ → ℝ) (n : ℝ) : ℝ → ℝ := λ x ↦ f (x - n)

def shift_down (f : ℝ → ℝ) (n : ℝ) : ℝ → ℝ := λ x ↦ f x - n

-- Define the final parabola after transformations
def final_parabola : ℝ → ℝ :=
  shift_down (shift_right (rotate_90_ccw initial_parabola) 4) 3

-- Theorem statement
theorem parabola_transformation :
  (∀ x, final_parabola x = -2 * (x - 8)^2) ∧
  (∃! z, final_parabola z = 0) ∧
  (∀ z, final_parabola z = 0 → z = 8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_transformation_l3_315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_angle_l3_313

/-- Given a triangle with sides a, b, c satisfying (a+b+c)(a+b-c) = 3ab, 
    the angle opposite side c is 60 degrees. -/
theorem triangle_special_angle (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b)
  (special_eq : (a + b + c) * (a + b - c) = 3 * a * b) :
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_angle_l3_313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_game_remaining_overs_l3_399

/-- Calculates the remaining overs in a cricket game given the target, initial run rate, required run rate, and game length -/
noncomputable def remaining_overs (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) (game_length : ℕ) : ℝ :=
  let initial_overs : ℕ := 10
  let initial_runs : ℝ := initial_rate * (initial_overs : ℝ)
  let remaining_runs : ℝ := target - initial_runs
  remaining_runs / required_rate

/-- Proves that the remaining overs in the given cricket game scenario is 40 -/
theorem cricket_game_remaining_overs :
  remaining_overs 282 4.8 5.85 50 = 40 := by
  sorry

-- Use #eval only for computable functions
def computable_remaining_overs (target : ℚ) (initial_rate : ℚ) (required_rate : ℚ) (game_length : ℕ) : ℚ :=
  let initial_overs : ℕ := 10
  let initial_runs : ℚ := initial_rate * initial_overs
  let remaining_runs : ℚ := target - initial_runs
  remaining_runs / required_rate

#eval computable_remaining_overs 282 (48/10) (585/100) 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_game_remaining_overs_l3_399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_l3_319

/-- A line y = kx + b is tangent to a curve y = f(x) at point x if the slope of the line equals the derivative of f at x -/
def is_tangent (k b : ℝ) (f : ℝ → ℝ) (x : ℝ) : Prop :=
  k * x + b = f x ∧ k = deriv f x

theorem tangent_line_value (k b : ℝ) :
  (∃ x₁ : ℝ, is_tangent k b (λ x ↦ Real.log x + 2) x₁) ∧
  (∃ x₂ : ℝ, is_tangent k b (λ x ↦ Real.log (x + 1)) x₂) →
  b = 1 + Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_l3_319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_similar_inflection_point_ln_l3_359

/-- A point P(x₀,y₀) is a "similar inflection point" for a curve C if:
    1) There exists a line m tangent to C at P
    2) C lies on both sides of m in a neighborhood of P -/
def is_similar_inflection_point (C : ℝ → ℝ) (x₀ y₀ : ℝ) : Prop :=
  ∃ (m : ℝ → ℝ), 
    (∀ x, m x = (deriv C x₀) * (x - x₀) + y₀) ∧ 
    (∃ δ > 0, ∀ x ∈ Set.Ioo (x₀ - δ) (x₀ + δ), 
      (C x - m x) * (C (2 * x₀ - x) - m (2 * x₀ - x)) < 0)

/-- The natural logarithm function -/
noncomputable def ln_curve (x : ℝ) : ℝ := Real.log x

theorem not_similar_inflection_point_ln :
  ¬ is_similar_inflection_point ln_curve 1 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_similar_inflection_point_ln_l3_359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euclidean_algorithm_and_bezout_identity_l3_358

theorem euclidean_algorithm_and_bezout_identity
  (m₀ m₁ : ℕ) (h : 0 < m₁ ∧ m₁ ≤ m₀) :
  ∃ (k : ℕ) (k_gt_1 : k > 1)
    (a : ℕ → ℕ) (m : ℕ → ℕ)
    (u v : ℤ),
    (∀ i, 1 < i ∧ i < k → m i < m (i-1)) ∧
    (m 1 = m₁) ∧
    (m 0 = m₀) ∧
    (0 < m k) ∧
    (a k > 1) ∧
    (m 0 = m 1 * a 0 + m 2) ∧
    (∀ i, 0 < i ∧ i < k-1 → m i = m (i+1) * a i + m (i+2)) ∧
    (m (k-1) = m k * a (k-1)) ∧
    (Nat.gcd m₀ m₁ = m k) ∧
    (m₀ * u + m₁ * v = Nat.gcd m₀ m₁) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euclidean_algorithm_and_bezout_identity_l3_358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ADE_radius_R_l3_348

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define membership for points in circles
variable [Membership Point Circle]

-- Define the parallelogram
variable (A B C D : Point)
variable (parallelogram : IsParallelogram A B C D)

-- Define the radius
variable (R : ℝ)

-- Define the circles
variable (circle1 circle2 : Circle)

-- Define the properties of the circles
variable (circle1_contains_A : A ∈ circle1)
variable (circle1_contains_B : B ∈ circle1)
variable (circle2_contains_B : B ∈ circle2)
variable (circle2_contains_C : C ∈ circle2)

-- Define HasRadius as a function
variable (HasRadius : Circle → ℝ → Prop)

variable (circle1_radius : HasRadius circle1 R)
variable (circle2_radius : HasRadius circle2 R)

-- Define point E
variable (E : Point)
variable (E_on_circle1 : E ∈ circle1)
variable (E_on_circle2 : E ∈ circle2)
variable (E_not_vertex : E ∉ {A, B, C, D})

-- Define the circle through A, D, and E
variable (circle_ADE : Circle)
variable (circle_ADE_contains_A : A ∈ circle_ADE)
variable (circle_ADE_contains_D : D ∈ circle_ADE)
variable (circle_ADE_contains_E : E ∈ circle_ADE)

-- State the theorem
theorem circle_ADE_radius_R : HasRadius circle_ADE R := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ADE_radius_R_l3_348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_on_nonparallel_planes_l3_360

-- Define a structure for a plane in 3D space
structure Plane3D where
  normal : Fin 3 → ℝ
  point : Fin 3 → ℝ

-- Define a structure for a line in 3D space
structure Line3D where
  direction : Fin 3 → ℝ
  point : Fin 3 → ℝ

-- Define two non-parallel planes
noncomputable def plane1 : Plane3D := ⟨λ _ => 1, λ _ => 0⟩
noncomputable def plane2 : Plane3D := ⟨λ _ => 2, λ _ => 0⟩

-- Define the line of intersection between the two planes
noncomputable def intersectionLine : Line3D := ⟨λ _ => 1, λ _ => 0⟩

-- Define membership for Line3D in Plane3D
def Line3D.mem (l : Line3D) (p : Plane3D) : Prop :=
  ∀ t, (p.normal 0) * (l.point 0 + t * l.direction 0) +
       (p.normal 1) * (l.point 1 + t * l.direction 1) +
       (p.normal 2) * (l.point 2 + t * l.direction 2) = 0

instance : Membership Line3D Plane3D where
  mem := Line3D.mem

-- Define parallelism for Line3D
def Line3D.parallel (l1 l2 : Line3D) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ i : Fin 3, l1.direction i = k * l2.direction i

infix:50 " ∥ " => Line3D.parallel

-- Axiom: The two planes are not parallel
axiom planes_not_parallel : plane1 ≠ plane2

-- Axiom: The intersection line lies on both planes
axiom intersection_on_planes : 
  (intersectionLine ∈ plane1) ∧ (intersectionLine ∈ plane2)

-- Theorem to prove
theorem parallel_lines_on_nonparallel_planes :
  ∃ (line1 line2 : Line3D), 
    (line1 ∈ plane1) ∧ 
    (line2 ∈ plane2) ∧ 
    (line1 ∥ line2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_on_nonparallel_planes_l3_360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_change_l3_377

/-- Represents a cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Theorem: Doubling the radius and halving the height of a cylinder with volume 16 results in a new volume of 32 -/
theorem cylinder_volume_change (c : Cylinder) 
  (h_orig_volume : volume c = 16) :
  volume { radius := 2 * c.radius, height := c.height / 2 } = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_change_l3_377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_is_60_l3_354

/-- The amount Ben spent at the pastry shop -/
def B : ℝ := sorry

/-- The amount David spent at the pastry shop -/
def D : ℝ := sorry

/-- For every dollar Ben spent, David spent 50 cents less -/
axiom david_spent_less : D = B / 2

/-- Ben paid $20 more than David -/
axiom ben_paid_more : B = D + 20

/-- The total amount spent by Ben and David -/
def total_spent : ℝ := B + D

/-- Theorem: The total amount spent by Ben and David is $60 -/
theorem total_spent_is_60 : total_spent = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_is_60_l3_354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_for_given_conditions_l3_332

/-- Calculate the difference between simple interest and compound interest -/
noncomputable def interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) (compounding_frequency : ℝ) : ℝ :=
  let simple_interest := principal * rate * time / 100
  let compound_interest := principal * ((1 + rate / (100 * compounding_frequency)) ^ (compounding_frequency * time) - 1)
  simple_interest - compound_interest

theorem interest_difference_for_given_conditions :
  let principal : ℝ := 1200
  let rate : ℝ := 10
  let time : ℝ := 1
  let compounding_frequency : ℝ := 2
  abs (interest_difference principal rate time compounding_frequency - 59.25) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_for_given_conditions_l3_332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_tv_show_payment_l3_384

/-- The total amount paid per episode for all characters in James' TV show -/
def total_payment_per_episode (num_main num_minor minor_pay main_factor : ℕ) : ℕ :=
  num_minor * minor_pay + num_main * (minor_pay * main_factor)

/-- Theorem stating the total payment per episode for James' TV show -/
theorem james_tv_show_payment :
  total_payment_per_episode 5 4 15000 3 = 285000 := by
  -- Unfold the definition of total_payment_per_episode
  unfold total_payment_per_episode
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_tv_show_payment_l3_384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l3_307

/-- The differential equation (1+x^2)dy - 2xy dx = 0 has the general solution y = C(1 + x^2) -/
theorem differential_equation_solution (x : ℝ) (y : ℝ → ℝ) (C : ℝ) :
  (∀ x, y x = C * (1 + x^2)) →
  ∀ x, (1 + x^2) * (deriv y x) - 2 * x * y x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l3_307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_impossible_triangle_division_l3_394

/-- A right-angled triangle in a square --/
structure RightTriangleInSquare where
  -- Define the properties of a right-angled triangle within a square
  area : ℝ
  isRightAngled : Bool

/-- Default instance for RightTriangleInSquare --/
instance : Inhabited RightTriangleInSquare where
  default := ⟨0, true⟩

/-- A square divided into right-angled triangles --/
structure DividedSquare where
  -- The side length of the square
  side : ℝ
  -- The list of right-angled triangles that the square is divided into
  triangles : List RightTriangleInSquare
  -- Ensure that the square is divided into exactly 12 triangles
  proper_division : triangles.length = 12
  -- Ensure that the sum of areas of all triangles equals the area of the square
  area_conservation : (triangles.map (·.area)).sum = side * side

/-- 
Theorem stating that it's impossible to divide a square into 12 right-angled triangles 
where 10 are congruent and 2 are different from the 10 and each other
-/
theorem impossible_triangle_division (s : DividedSquare) : 
  ¬ (∃ (t₁ t₂ : RightTriangleInSquare) (rest : List RightTriangleInSquare), 
      s.triangles = t₁ :: t₂ :: rest ∧ 
      rest.length = 10 ∧ 
      (∀ t ∈ rest, t = rest.head!) ∧
      t₁ ≠ t₂ ∧ 
      t₁ ∉ rest ∧ 
      t₂ ∉ rest) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_impossible_triangle_division_l3_394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_2014_in_third_quadrant_l3_304

-- Define the quadrant function
noncomputable def quadrant (angle : ℝ) : ℕ :=
  let normalized_angle := angle % 360
  if 0 ≤ normalized_angle ∧ normalized_angle < 90 then 1
  else if 90 ≤ normalized_angle ∧ normalized_angle < 180 then 2
  else if 180 ≤ normalized_angle ∧ normalized_angle < 270 then 3
  else 4

-- Theorem statement
theorem angle_2014_in_third_quadrant : quadrant 2014 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_2014_in_third_quadrant_l3_304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_logarithmic_functions_decreasing_exists_increasing_logarithmic_function_l3_383

-- Define a logarithmic function
noncomputable def logarithmic_function (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define natural logarithm
noncomputable def natural_log (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem not_all_logarithmic_functions_decreasing :
  ¬ (∀ (a : ℝ), a > 0 → ∀ (x₁ x₂ : ℝ), x₁ < x₂ → logarithmic_function a x₁ > logarithmic_function a x₂) :=
by sorry

-- Alternatively, we can state the theorem positively
theorem exists_increasing_logarithmic_function :
  ∃ (a : ℝ), a > 0 ∧ ∀ (x₁ x₂ : ℝ), x₁ < x₂ → logarithmic_function a x₁ < logarithmic_function a x₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_logarithmic_functions_decreasing_exists_increasing_logarithmic_function_l3_383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_decreasing_on_interval_l3_379

def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def DecreasingOn (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x > f y

theorem even_function_decreasing_on_interval
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_decreasing : ∀ x₁ x₂, x₁ ∈ Set.Iic (-1) → x₂ ∈ Set.Iic (-1) →
    (x₂ - x₁) * (f x₂ - f x₁) < 0) :
  f (-1) < f (-3/2) ∧ f (-3/2) < f 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_decreasing_on_interval_l3_379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l3_392

/-- Arithmetic sequence with first term 1 and common difference d -/
def arithmetic_seq (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => arithmetic_seq d n + d

/-- Geometric sequence with first term 1 and common ratio r -/
def geometric_seq (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => geometric_seq r n * r

/-- Sum of corresponding terms of arithmetic and geometric sequences -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r k : ℕ) :
  d > 0 ∧ r > 1 ∧ k > 2 ∧
  c_seq d r (k - 1) = 30 ∧
  c_seq d r (k + 1) = 300 →
  c_seq d r k = 83 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l3_392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_and_function_properties_l3_346

noncomputable def a (x : Real) : Real × Real := (Real.sqrt 3 * Real.sin x, Real.sin x)
noncomputable def b (x : Real) : Real × Real := (Real.cos x, Real.sin x)
noncomputable def f (x : Real) : Real := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem vector_equality_and_function_properties :
  (∃ x : Real, x ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    ((a x).1^2 + (a x).2^2 = (b x).1^2 + (b x).2^2) ∧ 
    x = Real.pi / 6) ∧
  (∀ x : Real, f (x + Real.pi) = f x) ∧
  (∀ k : Int, ∀ x : Real, 
    x ∈ Set.Icc (- Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi) →
    ∀ y : Real, y ∈ Set.Icc (- Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi) →
    x ≤ y → f x ≤ f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_and_function_properties_l3_346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_l3_382

theorem complex_inequality (a b c : ℂ) (m n : ℝ) 
  (h1 : Complex.abs (a + b) = m)
  (h2 : Complex.abs (a - b) = n)
  (h3 : m * n ≠ 0) :
  max (Complex.abs (a * c + b)) (Complex.abs (a + b * c)) ≥ m * n / Real.sqrt (m^2 + n^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_l3_382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_unique_l3_388

def is_valid_digit (d : ℕ) : Prop := d < 10

def distinct_digits (a b c d e f g h i j : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

def sum_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_digits (n / 10)

theorem puzzle_solution_unique :
  ∃! (e_dot e m a y u s l j n : ℕ),
    is_valid_digit e_dot ∧ is_valid_digit e ∧ is_valid_digit m ∧ 
    is_valid_digit a ∧ is_valid_digit y ∧ is_valid_digit u ∧ 
    is_valid_digit s ∧ is_valid_digit l ∧ is_valid_digit j ∧ 
    is_valid_digit n ∧
    distinct_digits e_dot e m a y u s l j n ∧
    e_dot * 10000 + l * 1000 + j * 100 + e * 10 + n +
    m * 10000 + a * 1000 + y * 100 + u * 10 + s =
    e * 100000 + l * 10000 + s * 1000 + e * 100 + j * 10 + e ∧
    sum_digits (e_dot * 10000 + l * 1000 + j * 100 + e * 10 + n) =
    sum_digits (m * 10000 + a * 1000 + y * 100 + u * 10 + s) ∧
    e_dot = 9 ∧ l = 3 ∧ j = 0 ∧ e = 1 ∧ n = 6 ∧
    m = 4 ∧ a = 2 ∧ y = 0 ∧ u = 8 ∧ s = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_unique_l3_388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vector_l3_334

/-- Given a line and a circle intersecting at two points, prove that (4, 2) is collinear with the sum of vectors from the origin to these points. -/
theorem collinear_vector (c R : ℝ) (A B : ℝ × ℝ) : 
  (2 * A.1 + A.2 = c) ∧ 
  (2 * B.1 + B.2 = c) ∧ 
  (A.1^2 + A.2^2 = R^2) ∧ 
  (B.1^2 + B.2^2 = R^2) → 
  ∃ (k : ℝ), k ≠ 0 ∧ (k * (A.1 + B.1) = 4 ∧ k * (A.2 + B.2) = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vector_l3_334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_l3_375

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a helper function to check if a real number is an integer
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

-- Define the theorem
theorem triangle_circle_intersection (ABC : Triangle) (circle : Circle) (X : ℝ × ℝ) :
  let AB := Real.sqrt ((ABC.B.1 - ABC.A.1)^2 + (ABC.B.2 - ABC.A.2)^2)
  let AC := Real.sqrt ((ABC.C.1 - ABC.A.1)^2 + (ABC.C.2 - ABC.A.2)^2)
  let AX := Real.sqrt ((X.1 - ABC.A.1)^2 + (X.2 - ABC.A.2)^2)
  let BX := Real.sqrt ((X.1 - ABC.B.1)^2 + (X.2 - ABC.B.2)^2)
  let CX := Real.sqrt ((X.1 - ABC.C.1)^2 + (X.2 - ABC.C.2)^2)
  let BC := BX + CX
  AB = 95 →
  AC = 115 →
  circle.center = ABC.A →
  circle.radius = AB →
  AX = 10 →
  isInteger BX →
  isInteger CX →
  BC = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_l3_375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_cubic_equation_l3_385

theorem solve_cubic_equation (x : ℝ) :
  (x - 6)^3 = (1/16)⁻¹ ↔ x = 6 + 2^(4/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_cubic_equation_l3_385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l3_373

open Real Set

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * sin x * cos x - 2 * (cos x)^2

/-- The smallest positive period of f(x) -/
noncomputable def smallest_period : ℝ := π

/-- The monotonically decreasing interval of f(x) -/
def decreasing_interval (k : ℤ) : Set ℝ := Icc (k * π + π / 3) (k * π + 5 * π / 6)

/-- The range of f(x) when x is in [0, π/2] -/
def range_in_interval : Set ℝ := Icc (-2) 1

theorem f_properties :
  (∀ x : ℝ, f (x + smallest_period) = f x) ∧
  (∀ k : ℤ, StrictMonoOn f (decreasing_interval k)) ∧
  (f '' Icc 0 (π / 2) = range_in_interval) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l3_373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l3_312

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ)

-- Define the line
def line (t : ℝ) : ℝ × ℝ :=
  (-1 + t, 2 * t)

-- Define the distance function between a point and the line
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p
  abs (2 * x - y + 2) / Real.sqrt 5

-- Theorem statement
theorem max_distance_curve_to_line :
  (∀ θ : ℝ, ∃ d : ℝ, distance_to_line (curve_C θ) ≤ d) ∧
  (∃ θ : ℝ, distance_to_line (curve_C θ) = (4 * Real.sqrt 5 + 5) / 5) := by
  sorry

#check max_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l3_312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_from_congruent_subtriangles_l3_301

/-- A triangle is isosceles if it has two congruent sides. -/
def IsIsosceles (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A B = dist A C ∨ dist B A = dist B C

/-- Two triangles are congruent if they have the same shape and size. -/
def Congruent (A B C D E F : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A B = dist D E ∧ dist B C = dist E F ∧ dist C A = dist F D

/-- A point lies on a line segment if it's between the endpoints. -/
def OnSegment (P A B : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A P + dist P B = dist A B

theorem isosceles_from_congruent_subtriangles 
  (A B C C₁ C₂ : EuclideanSpace ℝ (Fin 2)) 
  (h1 : OnSegment C₁ A C) 
  (h2 : OnSegment C₂ B C) 
  (h3 : Congruent A B C₁ B A C₂) : 
  IsIsosceles A B C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_from_congruent_subtriangles_l3_301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_4_l3_374

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x - 4

-- State the theorem
theorem f_of_g_4 : f (g 4) = 5 * Real.sqrt 6 := by
  -- Evaluate g(4)
  have h1 : g 4 = 24 := by
    simp [g]
    norm_num
  
  -- Apply f to g(4)
  have h2 : f (g 4) = f 24 := by
    rw [h1]
  
  -- Simplify f(24)
  have h3 : f 24 = 5 * Real.sqrt 6 := by
    simp [f]
    -- The rest of the proof steps would go here
    sorry
  
  -- Conclude
  rw [h2, h3]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_4_l3_374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_increase_l3_390

/-- Calculates the new wheel radius given the original and altered trip readings -/
noncomputable def new_wheel_radius (original_distance : ℝ) (altered_distance : ℝ) (original_radius : ℝ) : ℝ :=
  (original_distance * original_radius) / altered_distance

/-- Proves that the new wheel radius is approximately 17.57 inches -/
theorem wheel_radius_increase (original_distance altered_distance original_radius : ℝ)
  (h1 : original_distance = 600)
  (h2 : altered_distance = 580)
  (h3 : original_radius = 17) :
  ∃ ε > 0, |new_wheel_radius original_distance altered_distance original_radius - 17.57| < ε :=
by
  -- The proof is skipped using 'sorry'
  sorry

#eval Float.round ((600 * 17) / 580 - 17) * 100 / 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_increase_l3_390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_l3_393

/-- The trajectory of the center of a moving circle internally tangent to a fixed circle and passing through a point -/
theorem moving_circle_trajectory (C : Set (ℝ × ℝ)) (A : ℝ × ℝ) :
  C = {p : ℝ × ℝ | (p.1 + 4)^2 + p.2^2 = 100} →
  A = (4, 0) →
  ∃ (M : Set (ℝ × ℝ)), M = {p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1} ∧
    ∀ (c : ℝ × ℝ), c ∈ M ↔
      (∃ (r : ℝ), r > 0 ∧
        (∀ (p : ℝ × ℝ), p ∈ C → dist p c ≥ r) ∧
        (∃ (q : ℝ × ℝ), q ∈ C ∧ dist q c = r) ∧
        dist A c = r) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_l3_393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l3_387

noncomputable section

-- Define the lines
def line1 (x : ℝ) : ℝ := 3 * x + 6
def line2 (x : ℝ) : ℝ := -2 * x + 10
def y_axis : ℝ := 0

-- Define the intersection point of line1 and line2
def intersection_x : ℝ := 4 / 5
def intersection_y : ℝ := line1 intersection_x

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (y_axis, line1 y_axis)
def vertex2 : ℝ × ℝ := (y_axis, line2 y_axis)
def vertex3 : ℝ × ℝ := (intersection_x, intersection_y)

end noncomputable section

-- Theorem statement
theorem triangle_area : 
  let base := (line2 y_axis) - (line1 y_axis)
  let height := intersection_x
  (1 / 2) * base * height = 8 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l3_387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_range_l3_391

def a : ℝ × ℝ := (-2, -1)
def b (t : ℝ) : ℝ × ℝ := (t, 1)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def is_obtuse (v w : ℝ × ℝ) : Prop := dot_product v w < 0

def is_antiparallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ v = (k * w.1, k * w.2)

theorem vector_angle_range (t : ℝ) :
  (is_obtuse a (b t) ∧ ¬is_antiparallel a (b t)) ↔ t ∈ Set.Ioo (-1/2) 2 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_range_l3_391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l3_369

open Real Set

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m^2/2 + 2*m - 3

theorem problem_statement (m : ℝ) :
  -- Part 1
  (∃ a : ℝ, Ioo 1 a = {x | g m x < m^2/2 + 1}) →
  (∃ a : ℝ, Ioo 1 a = {x | g m x < m^2/2 + 1} ∧ a = 2) ∧
  -- Part 2
  (∀ x₁ ∈ Icc 0 1, ∃ x₂ ∈ Icc 1 2, f m x₁ > g m x₂) →
  -2 < m ∧ m < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l3_369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_l3_327

/-- The height of a right circular cone with given volume and base radius -/
noncomputable def cone_height (volume : ℝ) (radius : ℝ) : ℝ :=
  3 * volume / (Real.pi * radius^2)

/-- Theorem: The height of a right circular cone with volume 120 cubic inches
    and base radius 3 inches is 40/π inches -/
theorem funnel_height :
  cone_height 120 3 = 40 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_l3_327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_of_triangle_area_eight_l3_366

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

/-- The focal length of a hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := 2 * Real.sqrt (h.a^2 + h.b^2)

/-- The area of the triangle formed by the origin and the intersections of x = a with the asymptotes -/
def triangle_area (h : Hyperbola) : ℝ := h.a * h.b

theorem min_focal_length_of_triangle_area_eight (h : Hyperbola) 
    (h_area : triangle_area h = 8) :
    8 ≤ focal_length h ∧ ∃ (h' : Hyperbola), focal_length h' = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_of_triangle_area_eight_l3_366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_2015_l3_331

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def a : ℕ → ℤ
  | 0 => 1
  | n + 1 => floor (Real.sqrt ((n + 1 : ℝ) * a n))

theorem a_2017_equals_2015 : a 2017 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_2015_l3_331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishermanEarnings_l3_351

/-- Represents the daily catch and prices of fish -/
structure FishCatch where
  redSnapperCount : ℕ
  tunaCount : ℕ
  redSnapperPrice : ℕ
  tunaPrice : ℕ

/-- Calculates the daily earnings of a fisherman -/
def dailyEarnings (fc : FishCatch) : ℕ :=
  fc.redSnapperCount * fc.redSnapperPrice +
  fc.tunaCount * fc.tunaPrice

/-- Theorem: The fisherman's daily earnings are $52 -/
theorem fishermanEarnings :
  ∀ (fc : FishCatch),
    fc.redSnapperCount = 8 →
    fc.tunaCount = 14 →
    fc.redSnapperPrice = 3 →
    fc.tunaPrice = 2 →
    dailyEarnings fc = 52 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishermanEarnings_l3_351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_gas_mileage_round_trip_l3_308

/-- Calculates the average gas mileage for a round trip given the distances and fuel efficiencies of two different vehicles. -/
theorem average_gas_mileage_round_trip 
  (distance_to_city : ℝ) 
  (distance_from_city : ℝ) 
  (mpg_to_city : ℝ) 
  (mpg_from_city : ℝ) 
  (h1 : distance_to_city = 100) 
  (h2 : distance_from_city = 150) 
  (h3 : mpg_to_city = 25) 
  (h4 : mpg_from_city = 15) : 
  ∃ (avg_mpg : ℝ), 
    abs (avg_mpg - 17.86) < 0.01 ∧ 
    avg_mpg = (distance_to_city + distance_from_city) / (distance_to_city / mpg_to_city + distance_from_city / mpg_from_city) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_gas_mileage_round_trip_l3_308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomials_equal_l3_342

-- Define polynomials P and Q over real numbers
variable (P Q : ℝ[X])

-- Define the given equality condition
axiom equality_condition : ∀ x : ℝ, P.eval (P.eval (P.eval x)) = Q.eval (Q.eval (Q.eval x)) ∧ 
                                    Q.eval (Q.eval (Q.eval x)) = Q.eval (P.eval (P.eval x))

-- Theorem to prove
theorem polynomials_equal : P = Q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomials_equal_l3_342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_k_value_l3_316

/-- Given four points A, B, X, Y on a Cartesian plane, where AB is parallel to XY,
    prove that the y-coordinate of Y is -8. -/
theorem parallel_segments_k_value :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (0, -3)
  let X : ℝ × ℝ := (0, 10)
  let Y : ℝ × ℝ := (18, k)
  (((B.2 - A.2) / (B.1 - A.1)) = ((Y.2 - X.2) / (Y.1 - X.1))) →
  k = -8 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_k_value_l3_316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_difference_l3_356

/-- Given two real numbers, their arithmetic mean is their sum divided by 2 -/
noncomputable def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : arithmetic_mean p q = 10) 
  (h2 : arithmetic_mean q r = 20) : 
  r - p = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_difference_l3_356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radii_arrangements_l3_317

/-- The number of letters in the word "RADII" -/
def n : ℕ := 5

/-- The number of repeated letters (I's) in "RADII" -/
def r : ℕ := 2

/-- The factorial function -/
def factorial (m : ℕ) : ℕ :=
  match m with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

/-- The number of distinct arrangements of letters in "RADII" -/
def arrangements : ℕ := factorial n / factorial r

theorem radii_arrangements : arrangements = 60 := by
  rw [arrangements, n, r]
  norm_num
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radii_arrangements_l3_317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l3_353

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x)

theorem smallest_positive_period_of_f :
  ∀ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) → T ≥ π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l3_353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l3_372

/-- Reflection of a point across a line --/
def point_reflection (x y x' y' m b : ℝ) : Prop :=
  let midx := (x + x') / 2
  let midy := (y + y') / 2
  (midy = m * midx + b) ∧ (m = -(x' - x) / (y' - y))

/-- The main theorem --/
theorem reflection_sum (m b : ℝ) : 
  point_reflection 2 3 10 7 m b → m + b = 15 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l3_372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_girls_percentage_l3_396

def classroom_problem (initial_size : ℕ) (initial_girls_percent : ℚ) 
  (initial_boys_percent : ℚ) (new_boys : ℕ) (new_girls : ℕ) (boys_leaving : ℕ) : ℚ :=
  let initial_girls := (initial_girls_percent * initial_size).floor
  let initial_boys := (initial_boys_percent * initial_size).floor
  let new_total_girls := initial_girls + new_girls
  let new_total_boys := initial_boys + new_boys - boys_leaving
  let new_total := new_total_girls + new_total_boys
  (new_total_girls : ℚ) / (new_total : ℚ) * 100

theorem classroom_girls_percentage :
  classroom_problem 50 (38/100) (62/100) 8 6 4 = 25/60 * 100 := by
  sorry

#eval classroom_problem 50 (38/100) (62/100) 8 6 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_girls_percentage_l3_396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l3_343

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem hyperbola_eccentricity 
  (h : Hyperbola a b) 
  (p : Point) 
  (f1 f2 : Point) 
  (on_hyperbola : p.x^2 / a^2 - p.y^2 / b^2 = 1)
  (second_quadrant : p.x < 0 ∧ p.y > 0)
  (asymptote_bisects : ∃ (m : Point), 
    m.y = (b / a) * m.x ∧ 
    distance m f2 = distance m p ∧ 
    (m.y - f2.y) * (p.x - f2.x) = (f2.x - m.x) * (p.y - f2.y))
  (dist_to_f1 : distance p f1 = 2 * a) :
  eccentricity h = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l3_343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_with_sixteen_twos_l3_378

def countTwos (n : ℕ) : ℕ :=
  (n.repr.toList.filter (· = '2')).length

def isConsecutive (l : List ℕ) : Prop :=
  ∀ i, i + 1 < l.length → l[i + 1]! = l[i]! + 1

theorem consecutive_numbers_with_sixteen_twos : 
  ∃ (l : List ℕ), l.length = 7 ∧ isConsecutive l ∧ (l.map countTwos).sum = 16 := by
  -- Proof goes here
  sorry

#eval countTwos 2229
#eval countTwos 2230
#eval countTwos 2231
#eval countTwos 2232
#eval countTwos 2233
#eval countTwos 2234
#eval countTwos 2235

#eval [2229, 2230, 2231, 2232, 2233, 2234, 2235].map countTwos
#eval ([2229, 2230, 2231, 2232, 2233, 2234, 2235].map countTwos).sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_with_sixteen_twos_l3_378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_product_one_l3_320

theorem log_equality_implies_product_one 
  (M N : ℝ) 
  (h1 : Real.log (N^3) / Real.log M = Real.log (M^2) / Real.log N)
  (h2 : M ≠ N)
  (h3 : M * N > 0)
  (h4 : M ≠ 1)
  (h5 : N ≠ 1) :
  M * N = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_product_one_l3_320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_CDFE_l3_398

noncomputable section

/-- Square ABCD with side length 2 -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- Point E on side AB -/
def E (x : ℝ) : ℝ × ℝ := (2*x, 2)

/-- Point F on side AD -/
def F (x : ℝ) : ℝ × ℝ := (0, x)

/-- Area of quadrilateral CDFE -/
def area_CDFE (x : ℝ) : ℝ := 2 - x + (1/2) * x^2

/-- Theorem: Maximum area of CDFE is 1.5 when AF = 1 -/
theorem max_area_CDFE :
  ∃ (x : ℝ), x ∈ Set.Icc 0 2 ∧
  (∀ y ∈ Set.Icc 0 2, area_CDFE y ≤ area_CDFE x) ∧
  x = 1 ∧ area_CDFE x = 1.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_CDFE_l3_398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l3_380

def A : Set ℝ := {x | x^2 - 4*x - 5 ≤ 0}
def B : Set ℝ := {x | 1 < (2 : ℝ)^x ∧ (2 : ℝ)^x < 4}
def C (m : ℝ) : Set ℝ := {x | x < m}

theorem problem_solution :
  (A ∩ (Set.univ \ B) = {x | -1 ≤ x ∧ x ≤ 0 ∨ 2 ≤ x ∧ x ≤ 5}) ∧
  (∀ m : ℝ, (A ∩ C m ≠ A ∧ (B ∩ C m).Nonempty) → (0 < m ∧ m ≤ 5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l3_380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_passes_through_fixed_point_l3_363

/-- A point in a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A vector in a plane --/
structure Vec where
  x : ℝ
  y : ℝ

/-- A triangle in a plane --/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A line in a plane --/
structure Line where
  p : Point
  v : Vec

/-- The orthocenter of a triangle --/
noncomputable def orthocenter (t : Triangle) : Point := sorry

/-- The circumcenter of a triangle --/
noncomputable def circumcenter (t : Triangle) : Point := sorry

/-- A point lies on a line --/
def Point.on_line (p : Point) (l : Line) : Prop := sorry

/-- Two points are equidistant from a third point --/
def equidistant (p1 p2 center : Point) : Prop := sorry

theorem circumcenter_passes_through_fixed_point 
  (t : Triangle) (l : Line) (P : Point) : 
  Point.on_line t.B l → 
  Point.on_line t.C l → 
  orthocenter t = P → 
  let O := circumcenter t
  equidistant t.A t.B O ∧ 
  equidistant t.B t.C O ∧ 
  equidistant t.C t.A O ∧ 
  equidistant P t.A O := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_passes_through_fixed_point_l3_363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_line_l3_355

-- Define the ellipse C₁
def C₁ (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define a line
def Line (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m

-- Define tangency
def IsTangent (l : ℝ → ℝ → Prop) (c : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, l x y ∧ c x y ∧ ∀ x' y', l x' y' → c x' y' → (x', y') = (x, y)

theorem ellipse_and_tangent_line
  (a b : ℝ)
  (h1 : C₁ a b (-1) 0)  -- Left focus of C₁ is at (-1, 0)
  (h2 : C₁ a b 0 1)     -- Point (0, 1) lies on C₁
  (h3 : ∃ k m, IsTangent (Line k m) (C₁ a b) ∧ IsTangent (Line k m) C₂) :
  (∀ x y, C₁ a b x y ↔ x^2/2 + y^2 = 1) ∧
  (∃ k m, (k = Real.sqrt 2/2 ∧ m = Real.sqrt 2) ∨ (k = -Real.sqrt 2/2 ∧ m = -Real.sqrt 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_line_l3_355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_problem_l3_324

/-- The length of the chord intercepted by a circle on a line -/
noncomputable def chord_length (center : ℝ × ℝ) (radius : ℝ) (line : ℝ → ℝ → ℝ) : ℝ :=
  let d := abs (line center.1 center.2) / Real.sqrt ((line 1 0)^2 + (line 0 1)^2)
  2 * Real.sqrt (radius^2 - d^2)

/-- The problem statement -/
theorem chord_length_problem :
  let circle := fun (x y : ℝ) => (x - 2)^2 + (y + 1)^2 = 4
  let line := fun (x y : ℝ) => x + 2*y - 3 = 0
  chord_length (2, -1) 2 (fun x y => x + 2*y - 3) = 2 * Real.sqrt 55 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_problem_l3_324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_greater_than_sqrt_450_l3_311

theorem least_integer_greater_than_sqrt_450 : ∀ n : ℕ, n > Real.sqrt 450 → n ≥ 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_greater_than_sqrt_450_l3_311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_structure_l3_323

/-- Represents the structure of a company with knights and liars -/
structure Company where
  n : ℕ  -- Total number of people
  m : ℕ  -- Number of liars following each knight
  h_at_least_two : n ≥ 2  -- At least two people in the company

/-- Predicate to represent if a person is a knight -/
def is_knight (c : Company) (k : ℕ) : Prop := sorry

/-- Predicate to represent if a person is a liar -/
def is_liar (c : Company) (k : ℕ) : Prop := sorry

/-- The condition that not everyone can be a liar -/
axiom not_all_liars (c : Company) : ∃ (k : ℕ), k < c.n ∧ is_knight c k

/-- The condition that in a group of m people following a liar, there is at least one knight -/
axiom knight_after_liars (c : Company) (start : ℕ) (h_start : start < c.n) (h_liar : is_liar c start) :
  ∃ (k : ℕ), k < c.n ∧ k > start ∧ k ≤ start + c.m ∧ is_knight c k

/-- The main theorem: the number of people is divisible by m+1 -/
theorem company_structure (c : Company) : (c.n % (c.m + 1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_structure_l3_323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_l3_325

-- Define the quadrilateral
def quadrilateral (a b c d : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0)

-- Define the semiperimeter
noncomputable def semiperimeter (a b c d : ℝ) : ℝ := (a + b + c + d) / 2

-- Define the area using Brahmagupta's formula
noncomputable def area (a b c d : ℝ) : ℝ :=
  let s := semiperimeter a b c d
  Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d))

-- Theorem statement
theorem max_area_quadrilateral :
  ∀ (a b c d : ℝ), quadrilateral a b c d →
    a = 1 ∧ b = 4 ∧ c = 7 ∧ d = 8 →
    ∀ (x y z w : ℝ), quadrilateral x y z w →
      area x y z w ≤ area a b c d ∧
      area a b c d = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_l3_325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urban_green_spaces_support_l3_328

theorem urban_green_spaces_support (men_support_percent : ℝ) (women_support_percent : ℝ)
  (men_count : ℕ) (women_count : ℕ)
  (h1 : men_support_percent = 0.75)
  (h2 : women_support_percent = 0.70)
  (h3 : men_count = 150)
  (h4 : women_count = 650) :
  let total_count := men_count + women_count
  let men_support := ⌊men_support_percent * men_count⌋
  let women_support := ⌊women_support_percent * women_count⌋
  let total_support := men_support + women_support
  (total_support : ℝ) / total_count * 100 = 71 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_urban_green_spaces_support_l3_328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_positive_integer_solutions_l3_352

theorem inequality_solution (x : ℝ) :
  (2 * x + 1) / 3 - 1 ≤ 2 * x / 5 → x ≤ 5 / 2 :=
by sorry

def is_positive_integer_solution (x : ℝ) : Prop :=
  x > 0 ∧ x = ⌊x⌋ ∧ (2 * x + 1) / 3 - 1 ≤ 2 * x / 5

theorem positive_integer_solutions :
  ∀ x : ℝ, is_positive_integer_solution x ↔ (x = 1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_positive_integer_solutions_l3_352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounded_by_power_l3_395

noncomputable def a : ℕ → ℝ
| 0 => 1
| n + 1 => Real.sqrt (a n ^ 2 + 1 / a n)

theorem sequence_bounded_by_power : ∃ α : ℝ, α > 0 ∧ ∀ n : ℕ, n ≥ 1 → 1/2 ≤ a n / n^α ∧ a n / n^α ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounded_by_power_l3_395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_invertible_interval_l3_310

-- Define the quadratic function g
def g (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 9

-- Define the interval (-∞, -1]
def interval : Set ℝ := {x : ℝ | x ≤ -1}

-- State the theorem
theorem largest_invertible_interval :
  (∀ x ∈ interval, ∀ y ∈ interval, x ≠ y → g x ≠ g y) ∧
  (∀ I : Set ℝ, (-1 : ℝ) ∈ I → (∀ x ∈ I, ∀ y ∈ I, x ≠ y → g x ≠ g y) →
    ∃ x ∈ interval, x ∉ I ∨ ∃ y ∈ I, y ∉ interval) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_invertible_interval_l3_310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l3_367

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the distance from the point (-√2, 0) to one of its asymptotes is √5/5,
    then its eccentricity is √10/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (m : ℝ), |m * (-Real.sqrt 2) - 0 + 1| / Real.sqrt (m^2 + 1) = Real.sqrt 5 / 5) →
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 10 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l3_367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_expression_calculation_l3_389

-- Part 1: Equation solving
theorem equation_solution :
  ∃ (x₁ x₂ : ℝ), x₁ = -2/3 ∧ x₂ = 2 ∧
  (∀ x : ℝ, 3 * x * (x - 2) = 2 * (2 - x) ↔ x = x₁ ∨ x = x₂) := by sorry

-- Part 2: Expression calculation
theorem expression_calculation :
  |(-4)| - 2 * Real.cos (π / 3) + (Real.sqrt 3 - Real.sqrt 2)^0 - (-3)^2 = -5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_expression_calculation_l3_389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_100_similar_distinct_division_l3_361

/-- A rectangle is represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Two rectangles are similar if they have the same aspect ratio -/
def similar (r1 r2 : Rectangle) : Prop :=
  r1.width / r1.height = r2.width / r2.height

/-- A division of a rectangle into smaller rectangles -/
structure RectangleDivision where
  original : Rectangle
  parts : Finset Rectangle

/-- All parts in a division are similar to the original -/
def all_similar (d : RectangleDivision) : Prop :=
  ∀ r, r ∈ d.parts → similar d.original r

/-- All parts in a division are distinct -/
def all_distinct (d : RectangleDivision) : Prop :=
  ∀ r1 r2, r1 ∈ d.parts → r2 ∈ d.parts → r1 ≠ r2 → r1.width ≠ r2.width ∨ r1.height ≠ r2.height

/-- The main theorem: there exists a rectangle that can be divided into 100 similar but distinct rectangles -/
theorem exists_100_similar_distinct_division :
  ∃ d : RectangleDivision, d.parts.card = 100 ∧ all_similar d ∧ all_distinct d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_100_similar_distinct_division_l3_361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_of_f_l3_340

def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 5*x^3 + 3

theorem max_min_values_of_f :
  ∃ (x_max x_min : ℝ),
    x_max ∈ Set.Icc (-1 : ℝ) 2 ∧
    x_min ∈ Set.Icc (-1 : ℝ) 2 ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x_min ≤ f x) ∧
    f x_max = 4 ∧
    f x_min = -8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_of_f_l3_340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_roots_iff_m_in_range_l3_314

/-- The function f(x) = x(m + e^(-x)) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * (m + Real.exp (-x))

/-- The derivative of f with respect to x -/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := m + Real.exp (-x) - x * Real.exp (-x)

/-- The function g(x) = (x - 1) / e^x -/
noncomputable def g (x : ℝ) : ℝ := (x - 1) / Real.exp x

theorem f_two_roots_iff_m_in_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv m x₁ = 0 ∧ f_deriv m x₂ = 0) ↔ 
  (0 < m ∧ m < Real.exp (-2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_roots_iff_m_in_range_l3_314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_and_triangle_areas_l3_350

/-- Rectangle ABCD with points M on AB and N on AD -/
structure Rectangle :=
  (A B C D M N : ℝ × ℝ)
  (AN : ℝ)
  (NC : ℝ)
  (AM : ℝ)
  (MB : ℝ)

/-- The area of rectangle ABCD is 690 and the area of triangle MNC is 286.5 -/
theorem rectangle_and_triangle_areas (rect : Rectangle)
  (h1 : rect.AN = 7)
  (h2 : rect.NC = 39)
  (h3 : rect.AM = 12)
  (h4 : rect.MB = 3)
  (h5 : rect.A = (0, 0))
  (h6 : rect.B = (15, 0))
  (h7 : rect.C = (15, 46))
  (h8 : rect.D = (0, 46))
  (h9 : rect.M = (12, 0))
  (h10 : rect.N = (0, 7)) :
  (rect.B.1 - rect.A.1) * (rect.D.2 - rect.A.2) = 690 ∧
  (1/2) * abs (rect.M.1 * (rect.N.2 - rect.C.2) + 
               rect.N.1 * (rect.C.2 - rect.M.2) + 
               rect.C.1 * (rect.M.2 - rect.N.2)) = 286.5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_and_triangle_areas_l3_350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amy_final_time_l3_357

-- Define the given information
noncomputable def rachel_distance : ℚ := 9
noncomputable def rachel_time : ℚ := 36
noncomputable def amy_initial_distance : ℚ := 4
noncomputable def amy_final_distance : ℚ := 7

-- Define the relationship between Amy and Rachel's times
noncomputable def time_ratio : ℚ := 1 / 3

-- Theorem to prove
theorem amy_final_time (
  h1 : amy_initial_distance * rachel_time = time_ratio * rachel_distance * rachel_time
) : 
  (amy_final_distance * rachel_time * time_ratio * rachel_distance) / 
  (amy_initial_distance * rachel_distance) = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amy_final_time_l3_357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_series_equals_three_fourths_main_result_l3_333

/-- The sum of the even-indexed terms in the series -/
noncomputable def U : ℚ := ∑' n, (2*n + 1) / (2^(2*n + 2 : ℕ))

/-- The sum of the odd-indexed terms in the series -/
noncomputable def V : ℚ := ∑' n, (2*n + 2) / (3^(2*n + 3 : ℕ))

/-- The entire sum of the series -/
noncomputable def S : ℚ := U + V

theorem sum_series_equals_three_fourths : S = 3/4 := by sorry

theorem main_result (c d : ℕ) (h1 : Nat.Coprime c d) (h2 : (c : ℚ) / d = S) : c + d = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_series_equals_three_fourths_main_result_l3_333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_inequality_condition_l3_397

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + Real.log x)
def g (x : ℝ) : ℝ := x^2

-- Part 1: Tangent line condition
theorem tangent_line_condition (a : ℝ) (h : a ≠ 0) :
  (∃ m b : ℝ, ∀ x : ℝ, m * x + b = f a x ↔ m * x + b = g x) → a = 1 :=
by sorry

-- Part 2: Inequality condition
theorem inequality_condition (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x1 x2 : ℝ, x1 ∈ Set.Icc 1 2 → x2 ∈ Set.Icc 1 2 → x1 ≠ x2 →
    |f a x1 - f a x2| < |g x1 - g x2| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_inequality_condition_l3_397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_90000_units_l3_302

-- Define the production range
noncomputable def production_range : Set ℝ := {x | 0 ≤ x ∧ x ≤ 35}

-- Define the variable cost function
noncomputable def variable_cost (x : ℝ) : ℝ :=
  if x ≤ 14 then (2/3) * x^2 + 4*x else 17*x + 400/x - 80

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ :=
  if x ≤ 14 then -2/3 * x^2 + 12*x - 30 else 50 - x - 400/x

-- Theorem statement
theorem max_profit_at_90000_units :
  ∃ (max_profit : ℝ), ∀ x ∈ production_range,
    profit x ≤ max_profit ∧
    profit 9 = max_profit :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_90000_units_l3_302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_FOH_l3_349

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  EF : ℝ
  GH : ℝ
  area : ℝ

/-- The area of triangle FOH in an isosceles trapezoid -/
noncomputable def triangle_FOH_area (t : IsoscelesTrapezoid) : ℝ := t.area / 4

/-- Theorem: In the given isosceles trapezoid, the area of triangle FOH is 96 square units -/
theorem area_of_triangle_FOH (t : IsoscelesTrapezoid) 
  (h1 : t.EF = 24) 
  (h2 : t.GH = 40) 
  (h3 : t.area = 384) : 
  triangle_FOH_area t = 96 := by
  sorry

#check area_of_triangle_FOH

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_FOH_l3_349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinguishable_grids_l3_305

/-- Represents a 3x3 grid of black and white squares -/
def Grid := Fin 3 → Fin 3 → Bool

/-- The group of symmetries for a 3x3 grid -/
inductive Symmetry
| Rot0
| Rot90
| Rot180
| Rot270

/-- Applies a symmetry operation to a grid -/
def applySymmetry (s : Symmetry) (g : Grid) : Grid :=
  sorry

/-- Checks if a grid is fixed under a given symmetry -/
def isFixed (s : Symmetry) (g : Grid) : Bool :=
  sorry

/-- Counts the number of grids fixed under a given symmetry -/
def countFixed (s : Symmetry) : Nat :=
  sorry

/-- The number of distinguishable 3x3 grids -/
def numDistinguishableGrids : Nat :=
  ((countFixed Symmetry.Rot0 + countFixed Symmetry.Rot90 +
    countFixed Symmetry.Rot180 + countFixed Symmetry.Rot270) / 4) / 2

theorem num_distinguishable_grids :
  numDistinguishableGrids = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinguishable_grids_l3_305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_propositions_l3_339

theorem exactly_two_correct_propositions :
  let prop1 := ∀ a : ℝ, (a ≠ 0 → a^2 + a ≠ 0) ∧ ¬(a^2 + a ≠ 0 → a ≠ 0)
  let prop2 := (∀ a b : ℝ, a > b → (2 : ℝ)^a > (2 : ℝ)^b - 1) ↔ (∀ a b : ℝ, a < b → (2 : ℝ)^a ≤ (2 : ℝ)^b - 1)
  let prop3 := (∀ x : ℝ, x^2 + 1 ≥ 1) ↔ ¬(∃ x : ℝ, x^2 + 1 < 1)
  (prop1 ∧ ¬prop2 ∧ prop3) ∨
  (prop1 ∧ prop2 ∧ ¬prop3) ∨
  (¬prop1 ∧ prop2 ∧ prop3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_propositions_l3_339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l3_368

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2)^2 + 3 * (Real.log x / Real.log 2) + 2

-- State the theorem
theorem f_properties :
  ∃ (min_x max_x : ℝ),
    (∀ x, 1/4 ≤ x ∧ x ≤ 4 → -2 ≤ Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 ≤ 2) ∧
    (∀ x, 1/4 ≤ x ∧ x ≤ 4 → f x ≥ -1/4) ∧
    (f min_x = -1/4 ∧ 1/4 ≤ min_x ∧ min_x ≤ 4) ∧
    (∀ x, 1/4 ≤ x ∧ x ≤ 4 → f x ≤ 12) ∧
    (f max_x = 12 ∧ 1/4 ≤ max_x ∧ max_x ≤ 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l3_368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_H₂S_reaction_faster_l3_362

-- Define the gases and their properties
structure Gas where
  name : String
  mass : ℚ
  molar_mass : ℚ

-- Define the reaction rate based on the number of moles produced
noncomputable def reaction_rate (g : Gas) : ℚ := g.mass / g.molar_mass

-- Define the gases from the problem
def CO₂ : Gas := { name := "CO₂", mass := 23, molar_mass := 44 }
def H₂S : Gas := { name := "H₂S", mass := 20, molar_mass := 34 }

-- Theorem statement
theorem H₂S_reaction_faster : reaction_rate H₂S > reaction_rate CO₂ := by
  -- Unfold the definition of reaction_rate
  unfold reaction_rate
  -- Simplify the fraction
  simp
  -- The proof is completed using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_H₂S_reaction_faster_l3_362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_neg_x_plus_4_l3_326

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ :=
  -l.intercept / l.slope

/-- Check if a triangle formed by a line and the axes is isosceles -/
noncomputable def is_isosceles_triangle (l : Line) : Prop :=
  let origin : Point := ⟨0, 0⟩
  let y_intercept : Point := ⟨0, l.intercept⟩
  let x_intercept : Point := ⟨x_intercept l, 0⟩
  distance origin y_intercept = distance origin x_intercept

/-- The theorem stating that the line y = -x + 4 forms an isosceles triangle with the axes -/
theorem isosceles_triangle_neg_x_plus_4 :
  is_isosceles_triangle ⟨-1, 4⟩ := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_neg_x_plus_4_l3_326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_multiple_of_four_l3_371

/-- The probability of choosing at least one multiple of 4 when randomly selecting 3 integers
    between 1 and 60 (inclusive, with replacement) is 37/64. -/
theorem prob_at_least_one_multiple_of_four : 
  let range := Finset.range 60
  let multiples_of_four := range.filter (fun n => n % 4 = 0)
  let prob_not_multiple := 1 - (multiples_of_four.card : ℝ) / range.card
  1 - prob_not_multiple ^ 3 = 37 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_multiple_of_four_l3_371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_sum_existence_l3_338

theorem composite_sum_existence (A B C a b c : ℕ) (h : a * b * c > 1) :
  ∃ n : ℕ, ∃ k : ℕ, k > 1 ∧ k ∣ (A * a^n + B * b^n + C * c^n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_sum_existence_l3_338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coins_arbitrary_distribution_max_coins_equal_distribution_l3_347

/-- Represents the distribution of coins among warrior groups -/
structure CoinDistribution where
  groups : List Nat  -- List of group sizes
  coins_per_group : List Nat  -- List of coins per group

/-- Calculates the total coins Chernomor receives from a given distribution -/
def chernomor_coins (d : CoinDistribution) : Nat :=
  d.groups.zip d.coins_per_group |>.map (fun (g, c) => c % g) |>.sum

/-- Checks if a distribution is valid (total warriors = 33, total coins = 240) -/
def is_valid_distribution (d : CoinDistribution) : Prop :=
  d.groups.sum = 33 ∧ d.coins_per_group.sum = 240

/-- Checks if a distribution has equal coins per group -/
def is_equal_distribution (d : CoinDistribution) : Prop :=
  d.coins_per_group.all (· = d.coins_per_group.head!)

theorem max_coins_arbitrary_distribution :
  ∃ (d : CoinDistribution), is_valid_distribution d ∧
    (∀ (d' : CoinDistribution), is_valid_distribution d' →
      chernomor_coins d' ≤ chernomor_coins d) ∧
    chernomor_coins d = 31 := by sorry

theorem max_coins_equal_distribution :
  ∃ (d : CoinDistribution), is_valid_distribution d ∧ is_equal_distribution d ∧
    (∀ (d' : CoinDistribution), is_valid_distribution d' → is_equal_distribution d' →
      chernomor_coins d' ≤ chernomor_coins d) ∧
    chernomor_coins d = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coins_arbitrary_distribution_max_coins_equal_distribution_l3_347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_intersection_points_l3_303

/-- Parabola definition -/
noncomputable def parabola (t : ℝ) : ℝ × ℝ :=
  (3/2 + t^2, Real.sqrt 6 * t)

/-- Ellipse definition -/
noncomputable def ellipse (m : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (m + 2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

/-- Theorem stating that there is no m for which the parabola and ellipse intersect at four points -/
theorem no_four_intersection_points :
  ¬ ∃ (m : ℝ), ∃ (t₁ t₂ t₃ t₄ θ₁ θ₂ θ₃ θ₄ : ℝ),
    (t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₁ ≠ t₄ ∧ t₂ ≠ t₃ ∧ t₂ ≠ t₄ ∧ t₃ ≠ t₄) ∧
    (parabola t₁ = ellipse m θ₁) ∧
    (parabola t₂ = ellipse m θ₂) ∧
    (parabola t₃ = ellipse m θ₃) ∧
    (parabola t₄ = ellipse m θ₄) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_intersection_points_l3_303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_m_in_range_l3_381

/-- The inequality x^2 - log_m x < 0 holds for all x in (0, 1/2) if and only if m is in [1/16, 1) -/
theorem inequality_holds_iff_m_in_range (m : ℝ) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), x^2 - Real.log x / Real.log m < 0) ↔ m ∈ Set.Ico (1/16 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_m_in_range_l3_381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_properties_l3_341

/-- Represents a clock with hour and minute hands. -/
structure Clock where
  hour_hand_length : ℝ
  minute_hand_length : ℝ

/-- Calculates the area of the triangle formed by the clock hands at a given time. -/
noncomputable def triangle_area (clock : Clock) (t : ℝ) : ℝ :=
  (1/2) * clock.hour_hand_length * clock.minute_hand_length * |Real.sin ((11 * Real.pi / 6) * t)|

/-- Calculates the number of times the triangle area reaches its maximum value in a day. -/
noncomputable def max_area_count (clock : Clock) : ℕ :=
  Nat.floor (24 / ((2 * Real.pi) / (11 * Real.pi / 6)))

/-- Theorem about the properties of a specific clock. -/
theorem clock_properties :
  let c := Clock.mk 3 4
  (∀ t, triangle_area c t = 6 * |Real.sin ((11 * Real.pi / 6) * t)|) ∧
  max_area_count c = 44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_properties_l3_341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_triangle_side_length_l3_330

/-- Triangle ABC with side lengths AB, BC, and CA -/
structure Triangle where
  AB : ℝ
  BC : ℝ
  CA : ℝ

/-- Point P extends side BC of triangle ABC -/
structure ExtendedTriangle (t : Triangle) where
  PC : ℝ
  h : PC > t.BC

/-- Similarity of triangles PAB and PCA -/
def SimilarTriangles (t : Triangle) (e : ExtendedTriangle t) : Prop :=
  e.PC / (e.PC - t.BC) = t.AB / t.CA

theorem extended_triangle_side_length 
  (t : Triangle) 
  (h1 : t.AB = 10) 
  (h2 : t.BC = 8) 
  (h3 : t.CA = 7) 
  (e : ExtendedTriangle t) 
  (h4 : SimilarTriangles t e) : 
  e.PC = 56 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_triangle_side_length_l3_330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l3_376

noncomputable def a : ℕ+ → ℝ := fun n => Real.cos (Real.pi / (3 * 2^(n.val - 2)))

noncomputable def b : ℕ+ → ℝ := fun n => 1 - 2 / (n.val * n.val.factorial)

theorem sequence_properties :
  (∀ n : ℕ+, a (n + 1) = Real.sqrt ((a n + 1) / 2)) ∧
  a 1 > b 1 ∧
  a 2 = b 2 ∧
  (∀ n : ℕ+, n ≥ 3 → a n < b n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l3_376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_seven_l3_309

def modifiedFibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 2
  | n + 2 => (modifiedFibonacci (n + 1) + modifiedFibonacci n) % 10

def digitsAppeared (n : ℕ) : Finset ℕ :=
  Finset.filter (fun d => ∃ k, k ≤ n ∧ modifiedFibonacci k % 10 = d) (Finset.range 10)

theorem last_digit_is_seven :
  ∃ N, ∀ n ≥ N, digitsAppeared n = digitsAppeared N ∧
  7 ∉ digitsAppeared N ∧
  ∀ d < 10, d ≠ 7 → d ∈ digitsAppeared N := by
  sorry

#eval digitsAppeared 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_seven_l3_309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l3_364

-- Define the function f(x₁, x₂, a)
noncomputable def f (x₁ x₂ a : ℝ) : ℝ := 4 * x₁ * Real.log x₁ - x₁^2 + 3 + 4 * x₁ * x₂^2 + 8 * a * x₁ * x₂ - 16 * x₁

-- State the theorem
theorem range_of_a :
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ x₂ a ≥ 0) ↔ a ≥ -1/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l3_364
