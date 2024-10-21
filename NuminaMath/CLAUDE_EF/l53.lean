import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l53_5384

/-- Given a line y = 2x - 30 parameterized by (x, y) = (f(t), 20t - 10), prove that f(t) = 10t + 10 -/
theorem line_parameterization (t : ℝ) :
  let y : ℝ → ℝ := λ x ↦ 2 * x - 30
  let f : ℝ → ℝ := λ t ↦ 10 * t + 10
  (∀ t, y (f t) = 20 * t - 10) →
  f t = 10 * t + 10 := by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l53_5384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_I_is_positive_l53_5360

noncomputable def floor (x : ℝ) : ℤ := 
  Int.floor x

noncomputable def I (n : ℕ) : ℝ := 
  (n + 1)^2 + n - (floor (Real.sqrt ((n + 1)^2 + n + 1)))^2

-- Theorem statement
theorem I_is_positive (n : ℕ) : I n > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_I_is_positive_l53_5360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l53_5347

-- Define the radius of the larger circle C
def R : ℝ := 30

-- Define the number of smaller circles
def n : ℕ := 6

-- Define the area function K
noncomputable def K (r : ℝ) : ℝ := Real.pi * R^2 - n * Real.pi * r^2

-- State the theorem
theorem area_between_circles :
  ∃ r : ℝ,
    r > 0 ∧
    r < R ∧
    R - r = 2 * r ∧
    ⌊K r⌋ = 942 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l53_5347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l53_5358

-- Define the points
noncomputable def P : ℝ × ℝ := (2, 5)
noncomputable def Q : ℝ × ℝ := (-P.1, P.2)  -- Reflection of P over y-axis
noncomputable def R : ℝ × ℝ := (Q.2, Q.1)   -- Reflection of Q over y=x

-- Define the area of the triangle
noncomputable def area_PQR : ℝ := (1/2) * |P.1 - Q.1| * |R.2 - P.2|

-- Theorem statement
theorem area_of_triangle_PQR : area_PQR = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l53_5358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_probability_l53_5356

-- Define the chi-square statistic for a 2x2 contingency table
def chi_square_statistic : ℝ := 13.097

-- Define the probability of relationship
def probability_of_relationship (p : ℝ) : Prop := p ≥ 0.99

-- Theorem statement
theorem relationship_probability :
  probability_of_relationship (1 - Real.exp (-chi_square_statistic / 2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_probability_l53_5356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_squared_equals_27_l53_5389

noncomputable section

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- ABC is a right triangle
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- BC (hypotenuse) measures 4 cm
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 16

-- Define point D
def PointD (A B C D : ℝ × ℝ) : Prop :=
  -- D is on line BC
  ∃ t : ℝ, D = (B.1 + t * (C.1 - B.1), B.2 + t * (C.2 - B.2)) ∧
  -- Tangent at A to the circumcircle of ABC meets BC at D
  (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0

-- Define the equality BA = BD
def EqualDistances (A B D : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (D.1 - B.1)^2 + (D.2 - B.2)^2

-- Define the area of triangle ACD
def AreaACD (A C D : ℝ × ℝ) : ℝ :=
  abs ((C.1 - A.1) * (D.2 - A.2) - (D.1 - A.1) * (C.2 - A.2)) / 2

-- Theorem statement
theorem area_squared_equals_27 (A B C D : ℝ × ℝ) :
  Triangle A B C → PointD A B C D → EqualDistances A B D →
  (AreaACD A C D)^2 = 27 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_squared_equals_27_l53_5389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_condition_l53_5321

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x - k / 2 * x^2 + k * x

theorem local_minimum_condition (k : ℝ) (h : k > 0) :
  (∃ δ > 0, ∀ x, |x - 1| < δ → f k x ≥ f k 1) → 0 < k ∧ k < Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_condition_l53_5321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l53_5345

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * Real.sin (x + Real.pi/8) * (Real.sin (x + Real.pi/8) - Real.cos (x + Real.pi/8))

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/2) 0 → f (x + Real.pi/8) ≤ M) ∧
  (∃ (m : ℝ), m = -1 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/2) 0 → m ≤ f (x + Real.pi/8)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l53_5345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_l53_5394

theorem sin_shift (x : ℝ) : Real.sin (x - π / 12) = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_l53_5394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l53_5382

-- Define the equation
def satisfies_equation (x y : ℕ) (z : ℕ) : Prop :=
  x^2 + y^2 = 3 * 2016^z + 77

-- Define the set of solutions
def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(4, 8, 0), (8, 4, 0), (14, 77, 1), (77, 14, 1), (35, 70, 1), (70, 35, 1)}

-- Theorem statement
theorem equation_solutions :
  {(x, y, z) : ℕ × ℕ × ℕ | satisfies_equation x y z} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l53_5382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_square_theorem_l53_5319

def is_valid_permutation (n : ℕ) (perm : List ℕ) : Prop :=
  (perm.toFinset = Nat.divisors n) ∧
  (∀ i : Fin perm.length, ∃ m : ℕ, (perm.take (i.val + 1)).sum = m ^ 2)

theorem divisor_sum_square_theorem (n : ℕ) :
  (∃ perm : List ℕ, is_valid_permutation n perm) → n = 1 ∨ n = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_square_theorem_l53_5319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr1_simplification_expr2_simplification_expr2_value_at_3_expr2_value_at_4_l53_5334

-- Define the expressions
noncomputable def expr1 (a b : ℝ) : ℝ := (Real.sqrt a + Real.sqrt b - Real.sqrt (a - b)) / (Real.sqrt a + Real.sqrt b + Real.sqrt (a - b))

noncomputable def expr2 (a : ℝ) : ℝ := (Real.sqrt a - Real.sqrt (a - 1) - Real.sqrt (a^2 - a) + Real.sqrt (a + 2 - a - 1)) / (Real.sqrt (a^2 - a) + Real.sqrt (a^2 - a - 1) - Real.sqrt a - Real.sqrt (a - 1))

-- Theorem for the first expression
theorem expr1_simplification (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  expr1 a b = (Real.sqrt a - Real.sqrt (a - b)) / Real.sqrt b := by sorry

-- Theorem for the second expression
theorem expr2_simplification (a : ℝ) (h : a > 1) :
  expr2 a = (Real.sqrt a - Real.sqrt (a - 1)) * (Real.sqrt (a^2 - a) - Real.sqrt (a + 1)) := by sorry

-- Theorems for specific values of a
theorem expr2_value_at_3 :
  ∃ ε > 0, |expr2 3 - 0.0678| < ε := by sorry

theorem expr2_value_at_4 :
  ∃ ε > 0, |expr2 4 - 0.0395| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr1_simplification_expr2_simplification_expr2_value_at_3_expr2_value_at_4_l53_5334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_A_B_disjoint_iff_l53_5328

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | -1 < x - m ∧ x - m < 5}
def B : Set ℝ := {x | 1/2 < (2 : ℝ)^x ∧ (2 : ℝ)^x < 4}

-- Theorem for part (1)
theorem intersection_A_complement_B :
  A (-1) ∩ (Bᶜ) = {x | (-2 < x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x < 4)} := by sorry

-- Theorem for part (2)
theorem A_B_disjoint_iff (m : ℝ) :
  A m ∩ B = ∅ ↔ m ≤ -6 ∨ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_A_B_disjoint_iff_l53_5328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_correct_intersection_correct_l53_5355

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ (2 : ℝ)^(2-x) ∧ (2 : ℝ)^(2-x) < 8}
def B : Set ℝ := {x | x < 0}

-- Define the complement of A in R
def complement_A : Set ℝ := {x | x ≤ -1 ∨ x > 1}

-- Define the intersection of complement of B and A
def intersection_complement_B_A : Set ℝ := Set.Icc 0 1

-- Theorem statements
theorem complement_A_correct :
  Aᶜ = complement_A := by sorry

theorem intersection_correct :
  (Bᶜ ∩ A) = intersection_complement_B_A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_correct_intersection_correct_l53_5355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_coordinates_l53_5353

theorem rotation_coordinates (α : ℝ) (x₀ y₀ : ℝ) : 
  (Real.cos α - Real.sqrt 3 * Real.sin α = -22/13) →
  ((x₀ = 1/26) ∨ (x₀ = -23/26)) ∧ 
  (x₀ = Real.cos α) ∧ 
  (y₀ = Real.sin α) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_coordinates_l53_5353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_line_equation_with_condition_l53_5375

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line
def line_eq (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define the point P
def point_P : ℝ × ℝ := (1, 1)

-- Theorem 1: The line always intersects the circle at two points
theorem line_intersects_circle (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ line_eq m x₁ y₁ ∧ line_eq m x₂ y₂ :=
sorry

-- Theorem 2: If P divides AB such that |AP| = 1/2|PB|, then the line equation is x - y = 0 or x + y - 2 = 0
theorem line_equation_with_condition :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    ((x₁ - 1)^2 + (y₁ - 1)^2 = 1/4 * ((x₂ - 1)^2 + (y₂ - 1)^2)) →
    (∀ x y : ℝ, (x - y = 0 ∨ x + y - 2 = 0) ↔ (∃ m : ℝ, line_eq m x y)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_line_equation_with_condition_l53_5375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l53_5395

/-- A random variable following a normal distribution with mean 0 and variance σ² -/
def X (σ : ℝ) : Type := ℝ

/-- The probability measure for the normal distribution -/
noncomputable def P (σ : ℝ) : Set ℝ → ℝ := sorry

/-- Theorem: If P(-2 ≤ X ≤ 0) = 0.4 for a normal distribution with mean 0,
    then P(X > 2) = 0.1 -/
theorem normal_distribution_probability (σ : ℝ) (h : P σ {x : ℝ | -2 ≤ x ∧ x ≤ 0} = 0.4) :
  P σ {x : ℝ | x > 2} = 0.1 := by
  sorry

#check normal_distribution_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l53_5395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l53_5377

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.sin x) ^ 2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 3) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ 0) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 3) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l53_5377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_finite_sequences_l53_5357

/-- The function g(x) = 3x - x^2 -/
def g (x : ℝ) : ℝ := 3 * x - x^2

/-- The sequence x_n defined by x_n = g(x_{n-1}) for all n ≥ 1 -/
def sequenceG (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => g (sequenceG x₀ n)

/-- A set is finite if it has a bijection with a finite subset of ℕ -/
def IsFiniteSet (S : Set ℝ) : Prop := ∃ (n : ℕ) (f : S → Fin n), Function.Bijective f

/-- The set of values in the sequence starting from x₀ -/
def sequenceValues (x₀ : ℝ) : Set ℝ := {x | ∃ n, sequenceG x₀ n = x}

theorem infinitely_many_finite_sequences :
  ∃ S : Set ℝ, (Set.Infinite S) ∧ (∀ x₀ ∈ S, IsFiniteSet (sequenceValues x₀)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_finite_sequences_l53_5357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equidistant_from_points_l53_5340

-- Define the points
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (3, 5)
def P : ℝ × ℝ := (2, 1)

-- Define the line L
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 - 3 = 0 ∨ p.1 = 2}

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem line_equidistant_from_points :
  (P ∈ L) ∧ 
  (∀ p ∈ L, distance p A = distance p B) →
  (∀ p ∈ L, 2 * p.1 - p.2 - 3 = 0 ∨ p.1 - 2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equidistant_from_points_l53_5340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_z_is_zero_l53_5333

/-- The real part of the complex number z = ((i-1)^2 + 2) / (i+1) is equal to 0 -/
theorem real_part_of_z_is_zero : 
  let z : ℂ := ((Complex.I - 1)^2 + 2) / (Complex.I + 1)
  Complex.re z = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_z_is_zero_l53_5333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_digit_multiples_of_55_l53_5332

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000 ∧ n ≤ 9999999) ∧
  (∃ (a b c d e f g : ℕ),
    n = 1000000*a + 100000*b + 10000*c + 1000*d + 100*e + 10*f + g ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
    e ≠ f ∧ e ≠ g ∧
    f ≠ g ∧
    a ∈ ({1,2,3,4,5,6} : Set ℕ) ∧
    b ∈ ({0,1,2,3,4,5,6} : Set ℕ) ∧
    c ∈ ({0,1,2,3,4,5,6} : Set ℕ) ∧
    d ∈ ({0,1,2,3,4,5,6} : Set ℕ) ∧
    e ∈ ({0,1,2,3,4,5,6} : Set ℕ) ∧
    f ∈ ({0,1,2,3,4,5,6} : Set ℕ) ∧
    g ∈ ({0,1,2,3,4,5,6} : Set ℕ))

theorem seven_digit_multiples_of_55 :
  ∀ n : ℕ, is_valid_number n ∧ n % 55 = 0 →
    n ≥ 1042635 ∧ n ≤ 6431205 ∧
    (n = 1042635 ∨ n = 6431205 ∨ (1042635 < n ∧ n < 6431205)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_digit_multiples_of_55_l53_5332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_over_2_l53_5371

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x + Real.pi / 2

theorem derivative_f_at_pi_over_2 :
  deriv f (Real.pi / 2) = -Real.pi / 2 := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_over_2_l53_5371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_trig_value_for_obtuse_angle_l53_5339

theorem unique_trig_value_for_obtuse_angle (x : Real) 
  (h1 : 90 * Real.pi / 180 < x ∧ x < 180 * Real.pi / 180) 
  (h2 : ∃! v, v = Real.sin x ∨ v = Real.cos x ∨ v = 1 / Real.sin x) : 
  ∃ v, v = (-1 + Real.sqrt 5) / 2 ∧ (v = Real.sin x ∨ v = Real.cos x ∨ v = 1 / Real.sin x) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_trig_value_for_obtuse_angle_l53_5339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_kiosk_placement_l53_5305

/-- Minimum maximum distance for n kiosks in a unit circle -/
noncomputable def min_max_distance_circle (n : ℕ) : ℝ :=
  match n with
  | 1 | 2 => 1
  | 3 => Real.sqrt 3 / 2
  | 4 => Real.sqrt 2 / 2
  | 7 => 1 / 2
  | _ => 0  -- undefined for other values of n

/-- Minimum maximum distance for n kiosks in a unit square -/
noncomputable def min_max_distance_square (n : ℕ) : ℝ :=
  match n with
  | 1 => Real.sqrt 2 / 2
  | 2 | 4 => 1 / 2
  | 3 => Real.sqrt 3 / 3
  | _ => 0  -- undefined for other values of n

theorem optimal_kiosk_placement :
  ∀ (n : ℕ) (shape : String),
    (shape = "circle" ∧ n ∈ ({1, 2, 3, 4, 7} : Set ℕ)) ∨ (shape = "square" ∧ n ∈ ({1, 2, 3, 4} : Set ℕ)) →
    ∃ (r : ℝ),
      (shape = "circle" → r = min_max_distance_circle n) ∧
      (shape = "square" → r = min_max_distance_square n) ∧
      (r = min_max_distance_circle n ∨ r = min_max_distance_square n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_kiosk_placement_l53_5305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_from_perimeter_l53_5352

/-- The perimeter of a semi-circle with radius r -/
noncomputable def semicircle_perimeter (r : ℝ) : ℝ := Real.pi * r + 2 * r

/-- Theorem: Given a semi-circle with perimeter 26.736281798666923 cm, its radius is approximately 5.2 cm -/
theorem semicircle_radius_from_perimeter :
  ∃ r : ℝ, semicircle_perimeter r = 26.736281798666923 ∧ abs (r - 5.2) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_from_perimeter_l53_5352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log5_domain_l53_5391

-- Define the logarithm function with base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem about the domain of log5
theorem log5_domain : Set.range log5 = Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log5_domain_l53_5391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grady_initial_red_cubes_l53_5350

/-- The number of red cubes Grady had initially -/
def R : ℕ := 20

/-- The number of blue cubes Grady had initially -/
def B : ℕ := 15

/-- The fraction of red cubes Grady gave to Gage -/
def red_fraction : ℚ := 2/5

/-- The fraction of blue cubes Grady gave to Gage -/
def blue_fraction : ℚ := 1/3

/-- The number of red cubes Gage had initially -/
def gage_initial_red : ℕ := 10

/-- The number of blue cubes Gage had initially -/
def gage_initial_blue : ℕ := 12

/-- The total number of cubes Gage has after receiving cubes from Grady -/
def gage_total : ℕ := 35

theorem grady_initial_red_cubes : 
  R = 20 ∧ 
  gage_total = gage_initial_red + gage_initial_blue + 
    (red_fraction * R).floor + (blue_fraction * B).floor :=
by
  constructor
  · rfl
  · sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grady_initial_red_cubes_l53_5350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_iff_k_eq_neg_eight_l53_5362

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
variable (e₁ e₂ : V)

-- Define vectors a and b
def a : V := e₁ - 4 • e₂
def b (k : ℝ) : V := 2 • e₁ + k • e₂

-- State the theorem
theorem vectors_parallel_iff_k_eq_neg_eight
  (h_not_collinear : LinearIndependent ℝ ![e₁, e₂]) :
  (∃ (c : ℝ), c ≠ 0 ∧ a = c • b (-8)) ↔ 
  (∀ (k : ℝ), (∃ (c : ℝ), c ≠ 0 ∧ a = c • b k) → k = -8) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_iff_k_eq_neg_eight_l53_5362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l53_5329

-- Define the line l: x + y + 1 = 0
def line_l (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the circle: (x-1)² + (y-2)² = 4
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- State the theorem
theorem min_distance_circle_to_line :
  ∀ (x y : ℝ), circle_eq x y →
  (∃ (x' y' : ℝ), line_l x' y' ∧
    ∀ (x'' y'' : ℝ), line_l x'' y'' →
      ((x - x')^2 + (y - y')^2)^(1/2) ≤ ((x - x'')^2 + (y - y'')^2)^(1/2)) →
  (∃ (x' y' : ℝ), line_l x' y' ∧
    ((x - x')^2 + (y - y')^2)^(1/2) = 2 * Real.sqrt 2 - 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l53_5329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_two_l53_5368

-- Define the function f
noncomputable def f (a m x : ℝ) : ℝ := 1 + m / (a^x - 1)

-- State the theorem
theorem odd_function_implies_m_equals_two (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ m : ℝ, ∀ x : ℝ, f a m x = -f a m (-x)) → 
  ∃ m : ℝ, (∀ x : ℝ, f a m x = -f a m (-x)) ∧ m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_two_l53_5368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l53_5363

/-- Represents the three possible ball colors --/
inductive BallColor
  | Black
  | White
  | Green

/-- Represents the three boxes --/
inductive Box
  | One
  | Two
  | Three

/-- Represents the labels on the boxes --/
inductive Label
  | White
  | Black
  | WhiteOrGreen

/-- A function that assigns a ball color to each box --/
def assignment := Box → BallColor

/-- A function that assigns a label to each box --/
def labeling := Box → Label

/-- The condition that no label corresponds to the actual content of its box --/
def labelsAreIncorrect (a : assignment) (l : labeling) : Prop :=
  ∀ b : Box, 
    (l b = Label.White → a b ≠ BallColor.White) ∧
    (l b = Label.Black → a b ≠ BallColor.Black) ∧
    (l b = Label.WhiteOrGreen → (a b ≠ BallColor.White ∧ a b ≠ BallColor.Green))

/-- The theorem stating the unique solution to the problem --/
theorem unique_solution :
  ∀ a : assignment,
  (∀ c : BallColor, ∃! b : Box, a b = c) →
  labelsAreIncorrect a (λ b => match b with
    | Box.One => Label.White
    | Box.Two => Label.Black
    | Box.Three => Label.WhiteOrGreen) →
  a Box.One = BallColor.Green ∧
  a Box.Two = BallColor.White ∧
  a Box.Three = BallColor.Black :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l53_5363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l53_5304

/-- The area of a triangle given its vertices in the xy-coordinate system -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- Theorem: The area of the triangle with vertices at (-2, -3), (4, -3), and (28, 7) is 30 square units -/
theorem triangle_area_specific : triangleArea (-2) (-3) 4 (-3) 28 7 = 30 := by
  -- Expand the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l53_5304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l53_5380

open Real

theorem min_value_expression (x : ℝ) (h1 : 0 < x) (h2 : x < π) :
  (25 * x^2 * (sin x)^2 + 16) / (x * sin x) ≥ 40 ∧
  ∃ y, 0 < y ∧ y < π ∧ (25 * y^2 * (sin y)^2 + 16) / (y * sin y) = 40 := by
  sorry

#check min_value_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l53_5380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a2_plus_b2_l53_5379

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y + 1 = 0

-- Define the line
def line_eq (a b x y : ℝ) : Prop := a*x + 2*b*y + 4 = 0

-- Define the chord length
def chord_length (a b : ℝ) : ℝ := 4

-- Theorem statement
theorem min_a2_plus_b2 :
  ∃ min : ℝ, min = 2 ∧ 
  ∀ a b : ℝ, 
    (∃ x y : ℝ, circle_eq x y ∧ line_eq a b x y) →
    chord_length a b = 4 →
    a^2 + b^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a2_plus_b2_l53_5379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girl_speed_l53_5396

/-- The speed of a girl traveling a given distance in a given time -/
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem girl_speed : speed 128 32 = 4 := by
  -- Unfold the definition of speed
  unfold speed
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_girl_speed_l53_5396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_and_triangle_angle_l53_5398

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos (x - Real.pi/6)

theorem function_max_and_triangle_angle (A B C : ℝ) (a b : ℝ) :
  (∀ x, f x ≤ Real.sqrt 3) ∧
  (B = 2 * A) ∧
  (b = 2 * a * f (A - Real.pi/6)) →
  (∃ x, f x = Real.sqrt 3) ∧ C = Real.pi/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_and_triangle_angle_l53_5398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_lines_angle_l53_5337

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a function to calculate the angle between two lines
noncomputable def angle_between (l1 l2 : Line) : ℝ := sorry

-- Define the theorem
theorem twelve_lines_angle (O : Point) (lines : Finset Line) :
  lines.card = 12 → (∀ l ∈ lines, l O.fst O.snd) →
  ∃ l1 l2, l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 ∧ angle_between l1 l2 < 17 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_lines_angle_l53_5337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l53_5313

noncomputable def f (x : ℝ) : ℝ := Real.cos (4 * x - 5 * Real.pi / 6)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l53_5313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l53_5300

-- Define the universal set U
def U : Set ℝ := {x | x < 0}

-- Define set M
def M : Set ℝ := {x | x + 1 < 0}

-- Define set N
def N : Set ℝ := {x | (1/8 : ℝ) < Real.exp (x * Real.log 2) ∧ Real.exp (x * Real.log 2) < 1}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ M) ∩ N = {x : ℝ | -1 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l53_5300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_factorial_product_l53_5392

theorem smallest_n_factorial_product (a b c d : ℕ) (m n : ℕ) :
  a + b + c + d = 3000 →
  (Nat.factorial a * Nat.factorial b * Nat.factorial c * Nat.factorial d : ℕ) = m * 10^n →
  ¬(10 ∣ m) →
  ∀ k : ℕ, (∃ l : ℕ, (Nat.factorial a * Nat.factorial b * Nat.factorial c * Nat.factorial d : ℕ) = l * 10^k ∧ ¬(10 ∣ l)) → k ≥ n →
  n = 748 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_factorial_product_l53_5392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l53_5310

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 350 →
  train_speed_kmh = 180 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 7 := by
  intros h1 h2 h3
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l53_5310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l53_5388

noncomputable def v : ℝ × ℝ := (4, 7)
noncomputable def w : ℝ × ℝ := (2, -3)

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let scalar := dot_product / (v.1 * v.1 + v.2 * v.2)
  (scalar * v.1, scalar * v.2)

theorem projection_theorem :
  proj w v = (-2, 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l53_5388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sparrow_population_decline_l53_5349

/-- The smallest positive integer n such that 0.7^n < 0.2 is 5 -/
theorem sparrow_population_decline : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (0.7 : ℝ)^m < 0.2 → n ≤ m) ∧ (0.7 : ℝ)^n < 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sparrow_population_decline_l53_5349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_seating_with_constraints_l53_5311

def circular_seating : ℕ → ℕ
| n => n  -- Placeholder definition

def adults : ℕ := 3
def children : ℕ := 3
def total_seats : ℕ := 6

theorem circular_seating_with_constraints : 
  circular_seating total_seats = 72 := by
  sorry

#eval circular_seating total_seats

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_seating_with_constraints_l53_5311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_token_will_exit_l53_5318

/-- Represents the four possible directions of an arrow. -/
inductive Direction
  | Up
  | Right
  | Down
  | Left

/-- Represents a position in the maze. -/
structure Position where
  x : Fin 8
  y : Fin 8

/-- Represents the state of a cell in the maze. -/
structure Cell where
  arrow : Direction

/-- Represents the state of the entire maze. -/
def Maze := Fin 8 → Fin 8 → Cell

/-- Represents the state of the token in the maze. -/
structure TokenState where
  position : Position
  maze : Maze

/-- Rotates an arrow 90 degrees clockwise. -/
def rotateArrow (d : Direction) : Direction :=
  match d with
  | Direction.Up => Direction.Right
  | Direction.Right => Direction.Down
  | Direction.Down => Direction.Left
  | Direction.Left => Direction.Up

/-- Updates the maze state after a token move. -/
def updateMaze (m : Maze) (p : Position) : Maze :=
  fun x y => if x = p.x ∧ y = p.y then { arrow := rotateArrow (m p.x p.y).arrow } else m x y

/-- Moves the token according to the rules. -/
def moveToken (s : TokenState) : TokenState :=
  let newPos := 
    match (s.maze s.position.x s.position.y).arrow with
    | Direction.Up => ⟨s.position.x, s.position.y - 1⟩
    | Direction.Right => ⟨s.position.x + 1, s.position.y⟩
    | Direction.Down => ⟨s.position.x, s.position.y + 1⟩
    | Direction.Left => ⟨s.position.x - 1, s.position.y⟩
  { position := if newPos.x < 8 ∧ newPos.y < 8 then newPos else s.position,
    maze := updateMaze s.maze s.position }

/-- Checks if the token has reached the exit. -/
def isAtExit (s : TokenState) : Prop :=
  s.position.x = 7 ∧ s.position.y = 0

/-- The main theorem stating that the token will eventually exit the maze. -/
theorem token_will_exit (initialMaze : Maze) : 
  ∃ n : ℕ, isAtExit (Nat.iterate moveToken n { position := ⟨0, 7⟩, maze := initialMaze }) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_token_will_exit_l53_5318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_proof_l53_5366

-- Define the set of statement numbers
inductive StatementNumber
| one
| two
| three
| four

-- Define the properties of statements
noncomputable def fitting_effect (R_squared : ℝ) : ℝ := sorry

noncomputable def better_fitting_effect (R_squared : ℝ) : Prop := 
  ∀ R_squared' : ℝ, R_squared > R_squared' → fitting_effect R_squared > fitting_effect R_squared'

structure Property where
  dummy : Unit

def sphere_property (p : Property) : Prop := sorry
def circle_property (p : Property) : Prop := sorry

def sphere_circle_analogy : Prop :=
  ∃ (p : Property), sphere_property p ↔ circle_property p

-- Complex numbers are not naturally ordered, so we'll define a custom order
noncomputable def complex_order (z w : ℂ) : Prop := sorry

def some_complex_comparable : Prop :=
  ∃ (z w : ℂ), complex_order z w ∨ complex_order w z

structure Flowchart where
  dummy : Unit

def endpoints (f : Flowchart) : Finset Unit := sorry

def flowchart_multiple_endpoints : Prop :=
  ∃ (f : Flowchart), (endpoints f).card > 1

-- Define the set of correct statements
def correct_statements : Set StatementNumber :=
  {StatementNumber.one, StatementNumber.two}

-- Theorem statement
theorem correct_statements_proof :
  better_fitting_effect R_squared ∧
  sphere_circle_analogy ∧
  some_complex_comparable ∧
  flowchart_multiple_endpoints →
  correct_statements = {StatementNumber.one, StatementNumber.two} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_proof_l53_5366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_inverse_graph_and_sum_l53_5372

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the inverse function of f
noncomputable def f_inv (f : ℝ → ℝ) : ℝ → ℝ := Function.invFun f

-- Theorem statement
theorem point_on_inverse_graph_and_sum (h : 2 * f 2 = 3) :
  ∃ (x y : ℝ), 
    (x = 3/2 ∧ y = 2/3) ∧ 
    (y = f_inv f x / 3) ∧
    (x + y = 13/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_inverse_graph_and_sum_l53_5372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_deg_to_rad_l53_5324

-- Define the conversion factor from degrees to radians
noncomputable def deg_to_rad : ℝ := Real.pi / 180

-- Theorem statement
theorem eighteen_deg_to_rad : 
  18 * deg_to_rad = Real.pi / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_deg_to_rad_l53_5324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l53_5386

/-- The line y = 2x + 4 -/
def line (x : ℝ) : ℝ := 2 * x + 4

/-- The point we're measuring distance from -/
def point : ℝ × ℝ := (3, 1)

/-- The proposed closest point on the line -/
def closest_point : ℝ × ℝ := (0, 4)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem closest_point_on_line :
  ∀ x : ℝ, distance point (x, line x) ≥ distance point closest_point :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l53_5386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tadpole_fish_difference_l53_5322

/-- Represents the state of the pond -/
structure PondState where
  fish : ℕ
  tadpoles : ℕ
  snails : ℕ

/-- Calculates the new state of the pond after changes -/
def newPondState (initial : PondState) : PondState :=
  let fishCaught : ℕ := 12
  let tadpolesToFrogs : ℕ := (2 * initial.tadpoles) / 3
  let snailsCrawledAway : ℕ := 20
  { fish := initial.fish - fishCaught,
    tadpoles := initial.tadpoles - tadpolesToFrogs,
    snails := initial.snails - snailsCrawledAway }

/-- The main theorem to prove -/
theorem tadpole_fish_difference (initial : PondState)
  (h1 : initial.fish = 100)
  (h2 : initial.tadpoles = 4 * initial.fish)
  (h3 : initial.snails = 150) :
  (newPondState initial).tadpoles - (newPondState initial).fish = 46 := by
  sorry

-- Remove the #eval command as it's not allowed in this context

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tadpole_fish_difference_l53_5322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l53_5387

/-- The coefficient of x^2 in the expansion of (√x - 2/x)^7 -/
theorem coefficient_x_squared_in_expansion : ℕ := by
  -- Define the binomial expansion
  let binomial_expansion := fun x : ℝ => (Real.sqrt x - 2 / x) ^ 7

  -- Define the coefficient of x^2
  let coefficient_x_squared := 84

  -- State and prove the theorem
  have : coefficient_x_squared = 84 := by
    -- The actual proof would go here
    sorry

  -- Return the result
  exact coefficient_x_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l53_5387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l53_5397

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (ABC : Triangle) 
  (h1 : ABC.C = Real.pi / 4)
  (h2 : Real.cos ABC.A / Real.cos ABC.B = (Real.sqrt 5 * ABC.c - ABC.a) / ABC.b) :
  Real.cos ABC.A = Real.sqrt 10 / 10 ∧ 
  (ABC.b = Real.sqrt 5 → ABC.a * ABC.b * Real.sin ABC.C / 2 = 15 / 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l53_5397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomials_inequality_l53_5373

/-- Given quadratic polynomials P and Q, prove that b ≠ d when P(Q(x)) = Q(P(x)) has no real roots -/
theorem quadratic_polynomials_inequality (a b c d : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + b)^2 + c*(x^2 + a*x + b) + d ≠ (x^2 + c*x + d)^2 + a*(x^2 + c*x + d) + b) → 
  b ≠ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomials_inequality_l53_5373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_in_interval_l53_5346

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_decreasing_in_interval : 
  ∀ x y, π / 3 < x → x < y → y < 5 * π / 6 → f y < f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_in_interval_l53_5346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cash_amount_l53_5320

theorem cash_amount (gold_value silver_value : ℕ) 
                    (gold_count silver_count : ℕ) 
                    (total_value : ℕ) : ℕ :=
  let gold_value := 50
  let silver_value := 25
  let gold_count := 3
  let silver_count := 5
  let total_value := 305
  let cash_amount := total_value - (gold_value * gold_count + silver_value * silver_count)
  by
    have h1 : cash_amount = 30 := by
      -- Proof steps would go here
      sorry
    exact cash_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cash_amount_l53_5320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_a_equals_two_l53_5367

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≠ 4 then 2 / |x - 4| else a

-- Define the function y
noncomputable def y (a : ℝ) (x : ℝ) : ℝ := f a x - 2

-- Theorem statement
theorem three_zeros_implies_a_equals_two (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    y a x₁ = 0 ∧ y a x₂ = 0 ∧ y a x₃ = 0) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_a_equals_two_l53_5367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_king_game_strategy_l53_5312

/-- Represents a player in the game -/
inductive Player
| A  -- First player
| B  -- Second player

/-- Represents the game state -/
structure GameState where
  m : ℕ  -- Number of rows
  n : ℕ  -- Number of columns
  current_player : Player

/-- Determines the winner based on the dimensions of the board -/
def winner (state : GameState) : Player :=
  if state.m * state.n % 2 = 0 then Player.A else Player.B

/-- The main theorem stating the winning strategy -/
theorem king_game_strategy (state : GameState) :
  (winner state = Player.A → ∃ (strategy : GameState → ℕ × ℕ), 
    strategy state ∈ Set.univ ∧ 
    ∀ (next_move : ℕ × ℕ), next_move ∈ Set.univ → 
      ∃ (response : ℕ × ℕ), response ∈ Set.univ) ∧
  (winner state = Player.B → ∀ (first_move : ℕ × ℕ), first_move ∈ Set.univ → 
    ∃ (strategy : GameState → ℕ × ℕ), 
      strategy state ∈ Set.univ ∧ 
      ∀ (next_move : ℕ × ℕ), next_move ∈ Set.univ → 
        ∃ (response : ℕ × ℕ), response ∈ Set.univ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_king_game_strategy_l53_5312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_distribution_l53_5315

theorem flag_distribution (total_flags : ℕ) (total_children : ℕ) 
  (h_even : Even total_flags)
  (h_all_used : total_flags = 2 * total_children)
  (h_blue : (60 : ℚ) / 100 * total_children = (total_children : ℚ))
  (h_red : (70 : ℚ) / 100 * total_children = (total_children : ℚ)) :
  (30 : ℚ) / 100 * total_children = (total_children : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_distribution_l53_5315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_zero_l53_5301

/-- A line in 2D space -/
structure Line where
  slope : ℚ
  point : ℚ × ℚ

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- Theorem: The area of the triangle formed by the intersection of three lines is 0 -/
theorem triangle_area_zero 
  (l1 l2 : Line) 
  (h1 : l1.point = (2, 2))
  (h2 : l2.point = (2, 2))
  (h3 : l1.slope = -1/2)
  (h4 : l2.slope = 2)
  (l3 : ℚ × ℚ → Prop)
  (h5 : ∀ x y, l3 (x, y) ↔ x + y = 8) :
  ∃ p1 p2 p3, triangleArea p1 p2 p3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_zero_l53_5301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equals_eight_l53_5325

theorem ceiling_sum_equals_eight :
  ⌈Real.sqrt (16/9 : ℝ)⌉ + ⌈(16/9 : ℝ)⌉ + ⌈((16/9 : ℝ)^2)⌉ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equals_eight_l53_5325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l53_5341

/-- Given a geometric sequence {aₙ} with common ratio q and first term a,
    Sₙ represents the sum of the first n terms. -/
noncomputable def S (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio (a : ℝ) (q : ℝ) (hq : q ≠ 1) :
  (8 * (a * q) - (a * q^4) = 0) → (S a q 3 / S a q 2 = 7/3) := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l53_5341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michaels_initial_fish_l53_5323

/-- Given that Ben gave Michael 18.0 fish and Michael now has 67 fish,
    prove that Michael initially had 49 fish. -/
theorem michaels_initial_fish :
  ∀ (initial_fish : ℕ) (bens_gift : ℚ) (total_fish : ℕ),
  bens_gift = 18 →
  total_fish = 67 →
  total_fish = initial_fish + Int.floor bens_gift →
  initial_fish = 49 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_michaels_initial_fish_l53_5323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_equation_l53_5361

theorem power_difference_equation (x' y' z' : ℕ) : 
  720 = 2^x' * 3^y' * 5^z' →
  (∀ n : ℕ, n > x' → ¬(2^n ∣ 720)) →
  (∀ n : ℕ, n > y' → ¬(3^n ∣ 720)) →
  (∀ n : ℕ, n > z' → ¬(5^n ∣ 720)) →
  (1/6 : ℚ)^(z' - y') - (1/6 : ℚ)^(x' - z') = 1295/216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_equation_l53_5361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_odd_prime_factor_of_2023_8_plus_1_l53_5317

theorem least_odd_prime_factor_of_2023_8_plus_1 :
  ∃ p : Nat, Nat.Prime p ∧ p = 97 ∧ p ∣ (2023^8 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_odd_prime_factor_of_2023_8_plus_1_l53_5317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_ratio_l53_5326

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  a : ℝ  -- length of AB
  b : ℝ  -- length of CD
  d : ℝ  -- distance between lines AB and CD
  w : ℝ  -- angle between AB and CD
  k : ℝ  -- ratio of distances from plane π to AB and CD

/-- Calculates the ratio of volumes of two parts of a tetrahedron divided by a plane -/
noncomputable def volume_ratio (t : Tetrahedron) : ℝ :=
  (t.k^3 + 3 * t.k^2) / (3 * t.k + 1)

/-- Theorem stating the volume ratio of a divided tetrahedron -/
theorem tetrahedron_volume_ratio (t : Tetrahedron) :
  let v1 := volume_ratio t  -- volume of part containing AB
  let v2 := 1 - v1          -- volume of part containing CD
  v1 / v2 = (t.k^3 + 3 * t.k^2) / (3 * t.k + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_ratio_l53_5326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_max_height_projectile_initial_velocity_l53_5364

/-- The maximum height reached by a projectile launched vertically upward -/
noncomputable def max_height (v₀ : ℝ) (h₀ : ℝ) : ℝ := v₀^2 / (2 * 10) + h₀

/-- Theorem: The maximum height is 45 meters above the initial height when v₀ = 30 m/s -/
theorem projectile_max_height (h₀ : ℝ) :
  max_height 30 h₀ = h₀ + 45 := by
  sorry

/-- Theorem: The initial velocity that results in a maximum height 45 meters above the initial height is 30 m/s -/
theorem projectile_initial_velocity (h₀ : ℝ) (v₀ : ℝ) :
  max_height v₀ h₀ = h₀ + 45 → v₀ = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_max_height_projectile_initial_velocity_l53_5364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_win_sectors_area_calculation_l53_5309

/-- The radius of the circular spinner in centimeters -/
noncomputable def radius : ℝ := 15

/-- The probability of winning on one spin -/
noncomputable def win_probability : ℝ := 1/3

/-- The number of equal parts the winning section is divided into -/
def winning_section_parts : ℕ := 3

/-- The total area of the WIN sectors on the circular spinner -/
noncomputable def win_sectors_area : ℝ := 75 * Real.pi

/-- Theorem stating that the area of the WIN sectors is equal to the probability of winning
    multiplied by the area of the entire circle -/
theorem win_sectors_area_calculation :
  win_sectors_area = win_probability * Real.pi * radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_win_sectors_area_calculation_l53_5309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_zeroing_l53_5381

/-- Represents the state of numbers on a regular hexagon's vertices -/
structure HexagonState where
  vertices : Fin 6 → ℕ
  sum_is_2003 : (Finset.univ.sum vertices) = 2003

/-- Represents a single move on the hexagon -/
def move (s : HexagonState) (i : Fin 6) : HexagonState where
  vertices j := if j = i then 
    Int.natAbs (s.vertices ((i + 1) % 6) - s.vertices ((i + 5) % 6))
  else 
    s.vertices j
  sum_is_2003 := sorry  -- Proof that the sum is preserved omitted

/-- Represents a sequence of moves -/
def move_sequence : HexagonState → List (Fin 6) → HexagonState
  | s, [] => s
  | s, (m :: ms) => move_sequence (move s m) ms

/-- The main theorem to be proved -/
theorem hexagon_zeroing (s : HexagonState) : 
  ∃ (moves : List (Fin 6)), ∀ i, (move_sequence s moves).vertices i = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_zeroing_l53_5381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l53_5348

noncomputable def floor (z : ℝ) : ℤ := Int.floor z

theorem problem_statement (x y : ℝ) : 
  (y = 3 * (floor x) + 4) →
  (y = 2 * (floor (x - 3)) + 8 + x) →
  (x ≠ ↑(floor x)) →
  (x + y = 9) := by
  sorry

#check problem_statement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l53_5348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_x_equals_60_l53_5316

-- Define the star operation as noncomputable
noncomputable def star (a b : ℝ) : ℝ := Real.sqrt (a + b) / Real.sqrt (a - b)

-- Theorem statement
theorem solution_x_equals_60 : 
  ∀ x : ℝ, star x 48 = 3 → x = 60 := by
  intro x h
  -- Here we would normally prove the theorem
  -- For now, we'll use sorry to skip the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_x_equals_60_l53_5316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rich_walk_distance_l53_5344

/-- The distance Rich walks from his house to the sidewalk -/
def house_to_sidewalk : ℝ → ℝ := λ x => x

/-- The distance Rich walks down the sidewalk -/
def sidewalk_distance : ℝ := 200

/-- The distance Rich walks after making a left turn -/
def left_turn_distance : ℝ → ℝ := λ x => 2 * (x + sidewalk_distance)

/-- The distance Rich walks in the final leg before turning back -/
def final_leg_distance : ℝ → ℝ := λ x => 0.5 * (x + sidewalk_distance + left_turn_distance x)

/-- The total distance Rich walks one way -/
def one_way_distance : ℝ → ℝ := λ x => 
  house_to_sidewalk x + sidewalk_distance + left_turn_distance x + final_leg_distance x

/-- The total distance Rich walks (there and back) -/
def total_distance : ℝ → ℝ := λ x => 2 * one_way_distance x

/-- Theorem stating that if Rich walks 1980 feet in total, then his house is approximately 111 feet from the sidewalk -/
theorem rich_walk_distance : 
  ∃ x : ℝ, total_distance x = 1980 ∧ Int.floor x = 111 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rich_walk_distance_l53_5344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l53_5393

theorem sin_double_angle_special_case (A : ℝ) 
  (h1 : 0 < A) (h2 : A < π/2) (h3 : Real.cos A = 3/5) : 
  Real.sin (2*A) = 24/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l53_5393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transformation_l53_5354

/-- Represents a square table with filled and unfilled cells -/
def SquareTable (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a column is fully filled -/
def hasFullColumn {n : ℕ} (table : SquareTable n) : Prop :=
  ∃ j, ∀ i, table i j = true

/-- Swaps two rows in a table -/
def swapRows {n : ℕ} (table : SquareTable n) (i j : Fin n) : SquareTable n :=
  sorry

/-- Swaps two columns in a table -/
def swapColumns {n : ℕ} (table : SquareTable n) (i j : Fin n) : SquareTable n :=
  sorry

/-- Represents a sequence of row and column swaps -/
inductive TableOperation (n : ℕ)
  | swapRow (i j : Fin n)
  | swapColumn (i j : Fin n)

/-- Applies a sequence of operations to a table -/
def applyOperations {n : ℕ} (table : SquareTable n) (ops : List (TableOperation n)) : SquareTable n :=
  sorry

theorem impossible_transformation {n : ℕ} (initial final : SquareTable n) 
  (h_initial : hasFullColumn initial) 
  (h_final : ¬hasFullColumn final) :
  ¬∃ (ops : List (TableOperation n)), applyOperations initial ops = final :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transformation_l53_5354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_other_is_ten_percent_l53_5331

/-- Represents the tax rate on other items -/
def tax_rate_other : ℝ → Prop := sorry

/-- The total amount spent before taxes -/
def total_amount : ℝ := sorry

/-- The percentage spent on clothing -/
def clothing_percent : ℝ := sorry

/-- The percentage spent on food -/
def food_percent : ℝ := sorry

/-- The percentage spent on other items -/
def other_percent : ℝ := sorry

/-- The tax rate on clothing -/
def clothing_tax_rate : ℝ := sorry

/-- The total tax rate as a percentage of the total amount -/
def total_tax_rate : ℝ := sorry

theorem tax_rate_other_is_ten_percent :
  clothing_percent = 0.5 →
  food_percent = 0.2 →
  other_percent = 0.3 →
  clothing_tax_rate = 0.04 →
  total_tax_rate = 0.05 →
  clothing_percent + food_percent + other_percent = 1 →
  tax_rate_other 0.1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_other_is_ten_percent_l53_5331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skyscraper_workers_proof_skyscraper_workers_verify_l53_5336

/-- The number of workers needed to build a skyscraper in 70 days,
    given that 50 workers would take 42 days. -/
def skyscraper_workers : ℕ := 30

/-- Proof that the number of workers in the original scenario is correct. -/
theorem skyscraper_workers_proof :
  skyscraper_workers * 70 = 50 * 42 := by
  -- Unfold the definition of skyscraper_workers
  unfold skyscraper_workers
  -- Perform the calculation
  norm_num

/-- Verification that the solution satisfies the problem conditions -/
theorem skyscraper_workers_verify :
  ∃ (total_work : ℕ), 
    total_work = skyscraper_workers * 70 ∧
    total_work = 50 * 42 := by
  -- Use the result from skyscraper_workers
  use skyscraper_workers * 70
  constructor
  · rfl
  · exact skyscraper_workers_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skyscraper_workers_proof_skyscraper_workers_verify_l53_5336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pill_supply_duration_l53_5338

/-- Represents the number of days in a month -/
def days_per_month : ℕ := 30

/-- Represents the fraction of a pill taken daily -/
def daily_dose : ℚ := 1/4

/-- Represents the total number of pills in the supply -/
def total_pills : ℕ := 60

/-- Calculates the duration of the pill supply in months -/
noncomputable def duration_months : ℕ :=
  let days_supply := (total_pills : ℚ) / daily_dose
  Int.toNat ((days_supply / days_per_month).floor)

/-- Theorem stating that the pill supply lasts 8 months -/
theorem pill_supply_duration :
  duration_months = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pill_supply_duration_l53_5338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_count_l53_5343

theorem hyperbola_count : ∃! k : ℕ, 
  k = (Finset.filter (λ p : ℕ × ℕ ↦
    let (m, n) := p
    1 ≤ n ∧ n ≤ m ∧ m ≤ 5 ∧ Nat.choose m n > 1
  ) (Finset.product (Finset.range 6) (Finset.range 6))).card ∧ k = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_count_l53_5343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_periodic_l53_5327

noncomputable def f (x : ℝ) := 2 * Real.sin x ^ 2

theorem f_is_even_and_periodic :
  (∀ x, f x = f (-x)) ∧
  (∃ p > 0, ∀ x, f (x + p) = f x) ∧
  (∀ q > 0, (∀ x, f (x + q) = f x) → q ≥ π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_periodic_l53_5327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_combination_with_product_one_l53_5374

/-- There exists a unique combination of 6 distinct integers from 1 to 50 whose product is 1 -/
theorem unique_combination_with_product_one :
  ∃! (combination : Finset ℕ),
    combination.card = 6 ∧
    (∀ n ∈ combination, 1 ≤ n ∧ n ≤ 50) ∧
    combination.prod id = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_combination_with_product_one_l53_5374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonal_sum_is_20_root_3_l53_5359

/-- A rectangular box with given surface area and edge length sum -/
structure RectangularBox where
  a : ℝ
  b : ℝ
  c : ℝ
  surface_area : 2 * (a * b + b * c + c * a) = 150
  edge_sum : 4 * (a + b + c) = 60

/-- The sum of lengths of all interior diagonals of a rectangular box -/
noncomputable def interior_diagonal_sum (box : RectangularBox) : ℝ :=
  4 * Real.sqrt (box.a^2 + box.b^2 + box.c^2)

/-- Theorem: The sum of interior diagonal lengths for the given box is 20√3 -/
theorem interior_diagonal_sum_is_20_root_3 (box : RectangularBox) :
  interior_diagonal_sum box = 20 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonal_sum_is_20_root_3_l53_5359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l53_5399

/-- Given four points A(-6, 0), B(0, 6), C(0, -18), and D(18, m) on the Cartesian plane,
    if AC is parallel to BD, then m = -48. -/
theorem parallel_lines_m_value (m : ℝ) : 
  let A : ℝ × ℝ := (-6, 0)
  let B : ℝ × ℝ := (0, 6)
  let C : ℝ × ℝ := (0, -18)
  let D : ℝ × ℝ := (18, m)
  (C.2 - A.2) / (C.1 - A.1) = (D.2 - B.2) / (D.1 - B.1) → m = -48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l53_5399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l53_5302

-- Define the complex number type
variable (z : ℂ)

-- Define i as the imaginary unit
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  z * (1 + i) = 1 + 3 * i → z = 2 + i :=
by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l53_5302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l53_5365

/-- Represents a right triangle ABC with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of side BC -/
  x : ℝ
  /-- Center of the inscribed circle -/
  O : ℝ × ℝ
  /-- Assumption that AC is twice the length of BC -/
  ac_twice_bc : 2 * x = 2 * x
  /-- AB is the hypotenuse -/
  ab_is_hypotenuse : x * Real.sqrt 5 = Real.sqrt ((2 * x)^2 + x^2)
  /-- Area of the inscribed circle is 4π -/
  inscribed_circle_area : π * ((2 * x + x - x * Real.sqrt 5) / 2)^2 = 4 * π

/-- The area of the triangle ABC is 56 - 24√5 -/
theorem area_of_triangle (t : RightTriangleWithInscribedCircle) :
  (1/2) * t.x * (2 * t.x) = 56 - 24 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l53_5365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_artist_has_enough_paint_l53_5342

/-- Represents the artist's wing painting project -/
structure WingProject where
  feathers_per_wing : ℕ := 7
  sections_per_feather : ℕ := 3
  paint_layers : ℕ := 3
  paint_per_layer_section : ℝ := 0.15
  extra_paint_factor : ℝ := 0.1
  available_paint : ℝ := 157

/-- Calculates the total paint needed for the project -/
def total_paint_needed (project : WingProject) : ℝ :=
  let sections := 2 * project.feathers_per_wing * project.sections_per_feather
  let base_paint := sections * project.paint_layers * project.paint_per_layer_section * 2
  base_paint * (1 + project.extra_paint_factor)

/-- Theorem stating that the artist has enough paint for the project -/
theorem artist_has_enough_paint (project : WingProject) : 
  project.available_paint > total_paint_needed project := by
  sorry

#eval total_paint_needed ({ } : WingProject)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_artist_has_enough_paint_l53_5342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_surface_area_ratio_l53_5307

/-- Given a cylinder and a sphere where the cylinder's base diameter and height
    are equal to the sphere's diameter, the ratio of their surface areas is 3:2 -/
theorem cylinder_sphere_surface_area_ratio :
  ∀ (r : ℝ), r > 0 →
  (2 * Real.pi * r * r + 2 * Real.pi * r * (2 * r)) / (4 * Real.pi * r * r) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_surface_area_ratio_l53_5307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l53_5378

/-- A power function passing through (1/4, 1/2) -/
noncomputable def f : ℝ → ℝ := λ x => Real.sqrt x

/-- The point (1/4, 1/2) -/
noncomputable def point : ℝ × ℝ := (1/4, 1/2)

theorem power_function_through_point :
  f (point.1) = point.2 ∧
  ∀ g : ℝ → ℝ, (∃ α : ℝ, ∀ x : ℝ, g x = x ^ α) →
    g (point.1) = point.2 → g = f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l53_5378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_20_equals_3_pow_4181_l53_5303

/-- Sequence c_n defined recursively -/
def c : ℕ → ℕ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | 2 => 3^2
  | (n + 3) => c (n + 2) * c (n + 1)

/-- Auxiliary sequence d_n -/
def d : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => d (n + 2) + d (n + 1)

/-- Theorem stating that c_20 = 3^4181 -/
theorem c_20_equals_3_pow_4181 : c 20 = 3^4181 := by
  sorry

/-- Lemma: c_n = 3^(d_n) for all n -/
lemma c_equals_3_pow_d (n : ℕ) : c n = 3^(d n) := by
  sorry

/-- Lemma: d_20 = 4181 -/
lemma d_20_equals_4181 : d 20 = 4181 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_20_equals_3_pow_4181_l53_5303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factor_coefficient_bound_l53_5370

theorem quadratic_factor_coefficient_bound (r : ℕ+) (g : Polynomial ℤ) :
  (X ^ 2 - (r : ℤ) • X - 1 : Polynomial ℤ) ∣ g →
  ∃ (i : ℕ), |g.coeff i| ≥ (r : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factor_coefficient_bound_l53_5370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_sum_l53_5390

noncomputable def sequence_a : ℕ → ℝ := sorry
noncomputable def sequence_b : ℕ → ℝ := sorry

axiom recurrence_relation (n : ℕ) :
  (sequence_a (n + 1), sequence_b (n + 1)) = 
    (2 * sequence_a n - Real.sqrt 3 * sequence_b n, 
     2 * sequence_b n + Real.sqrt 3 * sequence_a n)

axiom fiftieth_term : 
  (sequence_a 50, sequence_b 50) = (3 * Real.sqrt 3, -3)

theorem first_term_sum :
  sequence_a 1 + sequence_b 1 = (3 * (Real.sqrt 3 - 1)) / (2^49) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_sum_l53_5390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_iff_sides_and_diagonals_equal_min_operations_for_square_check_l53_5335

/-- A quadrilateral represented by four points in a 2D plane -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Check if all sides of a quadrilateral are equal -/
def allSidesEqual (q : Quadrilateral) : Prop :=
  let AB := distance q.A q.B
  let BC := distance q.B q.C
  let CD := distance q.C q.D
  let DA := distance q.D q.A
  AB = BC ∧ BC = CD ∧ CD = DA

/-- Check if diagonals of a quadrilateral are equal -/
def diagonalsEqual (q : Quadrilateral) : Prop :=
  distance q.A q.C = distance q.B q.D

/-- Define what it means for a quadrilateral to be a square -/
def isSquare (q : Quadrilateral) : Prop :=
  allSidesEqual q ∧ diagonalsEqual q

/-- Theorem: A quadrilateral is a square if and only if all its sides are equal and its diagonals are equal -/
theorem square_iff_sides_and_diagonals_equal (q : Quadrilateral) :
  isSquare q ↔ allSidesEqual q ∧ diagonalsEqual q := by
  sorry

/-- The number of operations required to determine if a quadrilateral is a square -/
def numOperations : Nat := 10

/-- Theorem: The minimum number of operations required to determine if a quadrilateral is a square is 10 -/
theorem min_operations_for_square_check :
  ∀ (q : Quadrilateral), (isSquare q ↔ True) → numOperations = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_iff_sides_and_diagonals_equal_min_operations_for_square_check_l53_5335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relation_l53_5351

theorem order_relation (m : ℝ) (a b : ℝ) 
  (h1 : (9 : ℝ)^m = 10) 
  (h2 : a = (10 : ℝ)^m - 11) 
  (h3 : b = (8 : ℝ)^m - 9) : 
  a > 0 ∧ 0 > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relation_l53_5351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_inequality_l53_5306

/-- Predicate to represent a convex quadrilateral -/
def IsConvexQuadrilateral (A B C D : ℂ) : Prop :=
  ∃ (z₁ z₂ z₃ z₄ : ℂ), z₁ = A ∧ z₂ = B ∧ z₃ = C ∧ z₄ = D ∧
    (z₂ - z₁).arg < (z₃ - z₁).arg ∧
    (z₃ - z₁).arg < (z₄ - z₁).arg ∧
    (z₄ - z₁).arg < (z₂ - z₁).arg + 2 * Real.pi

/-- Given a convex quadrilateral ABCD, prove that AC · BD ≤ AB · CD + BC · AD -/
theorem convex_quadrilateral_inequality (A B C D : ℂ) (h : IsConvexQuadrilateral A B C D) :
  Complex.abs (A - C) * Complex.abs (B - D) ≤ 
  Complex.abs (A - B) * Complex.abs (C - D) + Complex.abs (B - C) * Complex.abs (A - D) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_inequality_l53_5306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_may_has_greatest_percentage_difference_l53_5314

/-- Represents the sales data for a single month -/
structure MonthSales where
  brass : ℕ
  woodwinds : ℕ
  percussionists : ℕ
deriving Inhabited

/-- Calculates the percentage difference for a given month's sales -/
def percentageDifference (sales : MonthSales) : ℚ :=
  let max := max sales.brass (max sales.woodwinds sales.percussionists)
  let min := min sales.brass (min sales.woodwinds sales.percussionists)
  (max - min : ℚ) / min * 100

/-- The sales data for each month -/
def salesData : List MonthSales := [
  ⟨6, 4, 5⟩,  -- January
  ⟨7, 5, 6⟩,  -- February
  ⟨7, 7, 7⟩,  -- March
  ⟨5, 6, 7⟩,  -- April
  ⟨3, 4, 2⟩   -- May
]

theorem may_has_greatest_percentage_difference :
  ∀ m ∈ salesData, percentageDifference (salesData.getLast!) ≥ percentageDifference m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_may_has_greatest_percentage_difference_l53_5314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_range_l53_5330

-- Define the line l
def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * x + 4

-- Define the circle O
def circle_O (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Define the range of r
def r_range (r : ℝ) : Prop := 1 < r ∧ r < 2

-- Define a rhombus
structure Rhombus where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  angle_60 : Prop  -- One interior angle is 60°

-- Define the condition that A and B are on line l
def A_B_on_l (rhombus : Rhombus) : Prop :=
  line_l rhombus.A.1 rhombus.A.2 ∧ line_l rhombus.B.1 rhombus.B.2

-- Define the condition that C and D are on circle O
def C_D_on_O (rhombus : Rhombus) (r : ℝ) : Prop :=
  circle_O rhombus.C.1 rhombus.C.2 r ∧ circle_O rhombus.D.1 rhombus.D.2 r

-- Define the area of a rhombus
noncomputable def rhombus_area (rhombus : Rhombus) : ℝ := sorry

-- Theorem statement
theorem rhombus_area_range (rhombus : Rhombus) (r : ℝ) :
  r_range r →
  A_B_on_l rhombus →
  C_D_on_O rhombus r →
  (0 < rhombus_area rhombus ∧ rhombus_area rhombus < (3 * Real.sqrt 3) / 2) ∨
  ((3 * Real.sqrt 3) / 2 < rhombus_area rhombus ∧ rhombus_area rhombus < 6 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_range_l53_5330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_member_age_l53_5308

/-- Represents a family with 7 members -/
structure Family :=
  (total_age : ℕ)
  (average_age : ℚ)
  (average_age_at_birth : ℚ)
  (youngest_age : ℕ)

/-- The conditions of the problem -/
def family_conditions (f : Family) : Prop :=
  f.total_age = 7 * 29 ∧
  f.average_age = 29 ∧
  f.average_age_at_birth = 28 ∧
  f.total_age = (6 * f.average_age_at_birth).floor + f.youngest_age * 7

/-- The theorem to be proved -/
theorem youngest_member_age (f : Family) :
  family_conditions f → f.youngest_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_member_age_l53_5308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_six_times_nineteen_l53_5383

theorem closest_to_six_times_nineteen : 
  let options : List ℤ := [600, 120, 100, 25]
  let actual_product : ℤ := 6 * 19
  ∀ x ∈ options, |120 - actual_product| ≤ |x - actual_product| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_six_times_nineteen_l53_5383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wax_sculpture_problem_l53_5385

/-- The number of sticks of wax used for all animals -/
def total_sticks (large_animals medium_animals small_animals : ℕ) : ℕ :=
  8 * large_animals + 5 * medium_animals + 3 * small_animals

/-- The problem statement -/
theorem wax_sculpture_problem :
  ∀ large_animals : ℕ,
    let medium_animals := 4 * large_animals
    let small_animals := 2 * large_animals
    3 * small_animals = 36 →
    total_sticks large_animals medium_animals small_animals = 204 :=
by
  intro large_animals
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wax_sculpture_problem_l53_5385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_profit_percentage_l53_5376

/-- Represents the cost price of an article -/
noncomputable def cost_price : ℝ → ℝ := sorry

/-- Represents the selling price of an article -/
noncomputable def selling_price : ℝ → ℝ := sorry

/-- Calculates the profit percentage given cost and selling prices -/
noncomputable def profit_percentage (c s : ℝ) : ℝ := (s - c) / c * 100

theorem merchant_profit_percentage :
  ∀ (c : ℝ), c > 0 →
  24 * (cost_price c) = 16 * (selling_price c) →
  profit_percentage (cost_price c) (selling_price c) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_profit_percentage_l53_5376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_third_implies_sin_alpha_plus_seven_pi_sixth_l53_5369

theorem cos_alpha_minus_pi_third_implies_sin_alpha_plus_seven_pi_sixth
  (α : ℝ)
  (h : Real.cos (α - π/3) = 3/4) :
  Real.sin (α + 7*π/6) = -3/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_third_implies_sin_alpha_plus_seven_pi_sixth_l53_5369
