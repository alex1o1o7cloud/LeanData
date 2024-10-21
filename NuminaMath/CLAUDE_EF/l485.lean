import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_equals_two_l485_48503

theorem tan_plus_cot_equals_two (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = Real.sqrt 2) 
  (h2 : α ∈ Set.Ioo 0 (Real.pi / 2)) : 
  Real.tan α + (1 / Real.tan α) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_equals_two_l485_48503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_sin_eq_x_div_3_solutions_l485_48579

theorem arcsin_sin_eq_x_div_3_solutions (x : ℝ) : 
  Real.arcsin (Real.sin x) = x / 3 ∧ -3 * π / 2 ≤ x ∧ x ≤ 3 * π / 2 →
  x = -3 * π ∨ x = -π ∨ x = 0 ∨ x = π ∨ x = 3 * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_sin_eq_x_div_3_solutions_l485_48579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_square_root_l485_48528

theorem factorial_square_root : Real.sqrt ((4 : ℕ).factorial * (4 : ℕ).factorial) = 24 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_square_root_l485_48528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_ratio_l485_48507

noncomputable section

/-- Parabola with equation y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- Focus of the parabola -/
def F : ℝ × ℝ := (2, 0)

/-- Line passing through F at 60° angle -/
noncomputable def Line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = Real.sqrt 3 * (p.1 - 2)}

/-- Point A: intersection of Line and Parabola in first quadrant -/
noncomputable def A : ℝ × ℝ := (6, 4 * Real.sqrt 3)

/-- Point B: other intersection of Line and Parabola -/
noncomputable def B : ℝ × ℝ := (2/3, 4 * Real.sqrt 3 / 3)

/-- Point C: intersection of Line and directrix -/
noncomputable def C : ℝ × ℝ := (-2, -4 * Real.sqrt 3)

/-- Origin -/
def O : ℝ × ℝ := (0, 0)

/-- Area of triangle given three points -/
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ := 
  (1/2) * abs ((p.1 - r.1) * (q.2 - r.2) - (q.1 - r.1) * (p.2 - r.2))

theorem parabola_triangle_area_ratio : 
  triangleArea A O C / triangleArea B O F = 6 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_ratio_l485_48507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interviewees_l485_48597

-- Define the number of people to be recruited
def num_recruited : ℕ := 3

-- Define the probability of two specific people being recruited
def prob_two_recruited : ℚ := 1 / 70

-- Theorem statement
theorem total_interviewees :
  ∀ n : ℕ,
  n ≥ num_recruited →
  (Nat.choose n num_recruited : ℚ) ≠ 0 →
  (Nat.choose (n - 2) 1 : ℚ) / (Nat.choose n num_recruited : ℚ) = prob_two_recruited →
  n = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interviewees_l485_48597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l485_48526

theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), x^2 - Real.log x / Real.log m < 0) →
  m > 0 →
  m ≠ 1 →
  m ∈ Set.Icc (1/16) 1 ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l485_48526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_of_coefficients_l485_48568

theorem polynomial_sum_of_coefficients 
  (g : ℂ → ℂ) 
  (p q r s : ℝ) :
  (∀ x : ℂ, g x = x^4 + p*x^3 + q*x^2 + r*x + s) →
  g (3*Complex.I) = 0 →
  g (1 + 2*Complex.I) = 0 →
  p + q + r + s = 39 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_of_coefficients_l485_48568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_correct_no_right_angle_intersection_l485_48565

/-- A circle with center (2, 2) that is tangent to the line 3x + 4y - 9 = 0 -/
def CircleC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 2)^2 = 1}

/-- The tangent line to CircleC -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 9 = 0}

/-- The intersecting line with parameter a -/
def IntersectingLine (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + a = 0}

/-- Theorem stating that CircleC is the correct equation for the given conditions -/
theorem circle_equation_correct :
  CircleC = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 2)^2 = 1} ∧
  ∃ p : ℝ × ℝ, p ∈ CircleC ∧ p ∈ TangentLine :=
sorry

/-- Theorem stating that no value of a satisfies the right angle condition -/
theorem no_right_angle_intersection :
  ∀ a : ℝ, ¬∃ (A B : ℝ × ℝ),
    A ∈ CircleC ∧ B ∈ CircleC ∧
    A ∈ IntersectingLine a ∧ B ∈ IntersectingLine a ∧
    (A.1 - 2) * (B.1 - 2) + (A.2 - 2) * (B.2 - 2) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_correct_no_right_angle_intersection_l485_48565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l485_48569

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := x / (2 * x^2 - 3 * x + 4)

-- State the theorem about the range of g
theorem range_of_g :
  ∀ y : ℝ, (∃ x : ℝ, g x = y) → -1 ≤ y ∧ y ≤ 1/23 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l485_48569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l485_48514

/-- A function f: (ℝ₊*)³ → ℝ₊* satisfying given conditions -/
noncomputable def f (x y z : ℝ) : ℝ := (z + Real.sqrt (y^2 + 4*x*z)) / (2*x)

/-- The main theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    x * f x y z = z * f z y x) ∧
  (∀ x y z t : ℝ, x > 0 → y > 0 → z > 0 → t > 0 →
    f x (t*y) (t^2*z) = t * f x y z) ∧
  (∀ k : ℝ, k > 0 → f 1 k (k+1) = k+1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l485_48514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_theorem_l485_48544

noncomputable section

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the circle
def Circle (x y x₀ y₀ r : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = r^2

-- Define the point M on the ellipse
def M (x₀ y₀ : ℝ) : Prop := Ellipse x₀ y₀

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the slopes of OP and OQ
def Slopes (k₁ k₂ : ℝ) (x₀ y₀ r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    Circle x₁ y₁ x₀ y₀ r ∧
    Circle x₂ y₂ x₀ y₀ r ∧
    k₁ = y₁ / x₁ ∧
    k₂ = y₂ / x₂

-- Define the condition |AF₁| - |BF₂| = 2r
def ConditionAB (x y x₀ y₀ r : ℝ) : Prop :=
  (Real.sqrt ((x + 2)^2 + y^2) - Real.sqrt ((x - 2)^2 + y^2)) = 2 * r

-- Main theorem
theorem ellipse_circle_theorem :
  ∃ (f : ℝ → ℝ → ℝ → Set (ℝ × ℝ)) (c : ℝ),
    (∀ x₀ y₀ r, M x₀ y₀ → (0 < r ∧ r < 1) →
      (∀ x y, (x, y) ∈ f x₀ y₀ r ↔ ConditionAB x y x₀ y₀ r)) ∧
    (∀ x₀ y₀ r k₁ k₂,
      M x₀ y₀ → (0 < r ∧ r < 1) → Slopes k₁ k₂ x₀ y₀ r →
      ∃ (OP OQ : ℝ),
        Ellipse OP (k₁ * OP) ∧
        Ellipse OQ (k₂ * OQ) ∧
        k₁ * k₂ = c →
        ∀ OP' OQ',
          Ellipse OP' (k₁ * OP') →
          Ellipse OQ' (k₂ * OQ') →
          OP * OQ ≥ OP' * OQ') :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_theorem_l485_48544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_wage_days_l485_48525

/-- The daily wage of person p -/
def P : ℝ := sorry

/-- The daily wage of person q -/
def Q : ℝ := sorry

/-- The total sum of money available -/
def S : ℝ := sorry

/-- The number of days the sum S is sufficient to pay q's wages -/
def D : ℝ := sorry

/-- The sum S is sufficient to pay p's wages for 24 days -/
axiom p_wages : S = 24 * P

/-- The sum S is sufficient to pay q's wages for D days -/
axiom q_wages : S = D * Q

/-- The sum S is sufficient to pay both p and q's wages for 15 days -/
axiom combined_wages : S = 15 * (P + Q)

/-- Theorem: The number of days the sum S is sufficient to pay q's wages is 40 -/
theorem q_wage_days : D = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_wage_days_l485_48525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l485_48542

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 3)

theorem parallel_vectors_lambda (l : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ (l • a + b) = k • (b - a)) → l = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l485_48542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l485_48520

/-- The length of a train given its speed, platform length, and crossing time --/
theorem train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  speed = 72 * (1000 / 3600) →
  platform_length = 210 →
  crossing_time = 26 →
  speed * crossing_time - platform_length = 310 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l485_48520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_tangent_circle_l485_48566

/-- The hyperbola equation -/
def hyperbola (m : ℝ) (x y : ℝ) : Prop := y^2 - x^2 / m^2 = 1

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 3 = 0

/-- Asymptote of the hyperbola -/
def asymptote (m : ℝ) (x y : ℝ) : Prop := x = m*y ∨ x = -m*y

/-- Tangency condition -/
def is_tangent (m : ℝ) : Prop :=
  ∀ x y, asymptote m x y → (∃ x' y', circle_eq x' y' ∧ (x - x')^2 + (y - y')^2 = 1)

theorem hyperbola_asymptote_tangent_circle (m : ℝ) (hm : m > 0) :
  is_tangent m → m = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_tangent_circle_l485_48566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_problem_l485_48552

theorem quadratic_factorization_problem :
  ∀ c d : ℕ,
  (∀ x : ℝ, x^2 - 18*x + 72 = (x - c : ℝ) * (x - d : ℝ)) →
  c > d →
  c > 0 →
  d > 0 →
  4 * d - c = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_problem_l485_48552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_octagon_diagonals_l485_48561

structure InscribedOctagon :=
  (radius : ℝ)
  (side_length : ℝ)
  (symmetry_axes : ℕ)
  (h_radius : radius = 6)
  (h_side : side_length = 5)
  (h_axes : symmetry_axes = 4)

noncomputable def center_to_vertex (o : InscribedOctagon) : ℝ :=
  o.side_length / Real.sqrt (2 - Real.sqrt 2)

noncomputable def longest_diagonal (o : InscribedOctagon) : ℝ :=
  2 * center_to_vertex o

noncomputable def other_diagonal (o : InscribedOctagon) : ℝ :=
  2 * Real.sqrt 45

theorem inscribed_octagon_diagonals (o : InscribedOctagon) :
  ∃ (a b : ℝ),
    a = longest_diagonal o ∧
    b = other_diagonal o ∧
    a = 2 * center_to_vertex o ∧
    b = 2 * Real.sqrt 45 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_octagon_diagonals_l485_48561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_digits_correct_l485_48523

/-- The numerator of the fraction -/
def numerator : ℕ := 987654321

/-- The denominator of the fraction -/
def denominator : ℕ := 2^30 * 5^3

/-- The fraction to be expressed as a decimal -/
def fraction : ℚ := numerator / denominator

/-- The minimum number of digits to the right of the decimal point -/
def min_digits : ℕ := 30

/-- Theorem statement -/
theorem min_digits_correct :
  (∀ n : ℕ, n < min_digits → ∃ m : ℕ, fraction * 10^n ≠ m) ∧
  (∃ m : ℕ, fraction * 10^min_digits = m) := by
  sorry

#eval min_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_digits_correct_l485_48523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S_T_l485_48506

-- Define the set T
def T : Set (ℝ × ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0 ∧ p.1 + p.2.1 + p.2.2 = 1}

-- Define the support relation
def supports (p q : ℝ × ℝ × ℝ) : Prop :=
  (p.1 ≥ q.1 ∧ p.2.1 ≥ q.2.1) ∨ (p.1 ≥ q.1 ∧ p.2.2 ≥ q.2.2) ∨ (p.2.1 ≥ q.2.1 ∧ p.2.2 ≥ q.2.2)

-- Define the set S
def S : Set (ℝ × ℝ × ℝ) :=
  {p ∈ T | supports p (1/3, 1/4, 1/5)}

-- Define the area function (this is just a placeholder)
noncomputable def area (X : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem area_ratio_S_T : area S / area T = 347 / 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S_T_l485_48506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_value_triangle_area_l485_48511

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
noncomputable def triangle_ABC : Triangle where
  a := 1
  c := Real.sqrt 2
  C := Real.arccos (3/4)
  -- Other fields are left undefined as they are not given in the problem
  b := sorry
  A := sorry
  B := sorry

-- Theorem 1: sin A = √14/8
theorem sin_A_value (t : Triangle) (h1 : t = triangle_ABC) : 
  Real.sin t.A = Real.sqrt 14 / 8 := by sorry

-- Theorem 2: Area of triangle ABC = √7/4
theorem triangle_area (t : Triangle) (h1 : t = triangle_ABC) :
  (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 7 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_value_triangle_area_l485_48511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l485_48508

-- Define the vectors a and b
def a : Fin 3 → ℝ := ![1, 1, 0]
def b : Fin 3 → ℝ := ![-1, 0, 1]

-- Define the dot product of two 3D vectors
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

-- Define vector addition and scalar multiplication
def vector_add (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => v i + w i

def scalar_mult (k : ℝ) (v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => k * (v i)

-- State the theorem
theorem perpendicular_vectors (k : ℝ) :
  dot_product (vector_add (scalar_mult k a) b) a = 0 → k = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l485_48508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l485_48550

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 3*x ≤ 4}

-- Define set B
def B : Set ℝ := {x : ℝ | Real.exp (x * Real.log 2) > 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioc 1 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l485_48550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l485_48593

noncomputable def sequence_a (n : ℕ) : ℝ := 3^(n-1)

noncomputable def S (n : ℕ) : ℝ := (3 * sequence_a n - 1) / 2

noncomputable def sequence_b (n : ℕ) : ℝ := n / sequence_a n

noncomputable def T (n : ℕ) : ℝ := 9/4 - (6*n + 9)/(4 * 3^n)

theorem sequence_properties (n : ℕ) :
  (∀ k, 2 * S k = 3 * sequence_a k - 1) →
  (sequence_a n = 3^(n-1) ∧
   T n = 9/4 - (6*n + 9)/(4 * 3^n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l485_48593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bargaining_agreed_price_l485_48567

/-- The agreed price function -/
noncomputable def agreed_price (a b : ℝ) : ℝ := (2 * a * b) / (a + b)

/-- Theorem: The agreed price is (2ab)/(a+b) given initial offers a and b -/
theorem bargaining_agreed_price (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a < b) :
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ a * (1 + x) = b * (1 - x) ∧ 
  a * (1 + x) = agreed_price a b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bargaining_agreed_price_l485_48567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_l485_48513

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Area of a triangle given its vertices -/
noncomputable def area_triangle (A B C : Point) : ℝ :=
  sorry

/-- Set of points C such that triangle ABC has area 2 -/
def S' (A B : Point) : Set Point :=
  {C | area_triangle A B C = 2}

/-- Two lines are parallel if they have the same slope -/
def parallel_lines (l1 l2 : Set Point) : Prop :=
  sorry

theorem S'_is_two_parallel_lines :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  ∃ (l1 l2 : Set Point), S' A B = l1 ∪ l2 ∧ parallel_lines l1 l2 :=
by
  sorry

#check S'_is_two_parallel_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_l485_48513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_specific_case_triangle_acute_case_l485_48522

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) (m : ℝ) : Prop :=
  Real.sin t.B + Real.sin t.C = m * Real.sin t.A ∧ t.a^2 - 4*t.b*t.c = 0

-- Theorem 1
theorem triangle_specific_case (t : Triangle) (m : ℝ) :
  satisfies_conditions t m → t.a = 2 → m = 5/4 →
  ((t.b = 2 ∧ t.c = 1/2) ∨ (t.b = 1/2 ∧ t.c = 2)) := by
  sorry

-- Theorem 2
theorem triangle_acute_case (t : Triangle) (m : ℝ) :
  satisfies_conditions t m → 0 < t.A → t.A < π/2 →
  Real.sqrt (3/2) < m ∧ m < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_specific_case_triangle_acute_case_l485_48522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l485_48571

-- Define the function f as noncomputable
noncomputable def f (x m : ℝ) : ℝ := Real.sqrt (|x + 2| + |x - 4| - m)

-- State the theorem
theorem function_properties :
  (∃ (m : ℝ), ∀ (x : ℝ), f x m ∈ Set.univ) ∧
  (∀ (m : ℝ), (∀ (x : ℝ), f x m ∈ Set.univ) → m ≤ 6) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → 4 / (a + 5*b) + 1 / (3*a + 2*b) = 6 → 4*a + 7*b ≥ 3/2) ∧
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 4 / (a + 5*b) + 1 / (3*a + 2*b) = 6 ∧ 4*a + 7*b = 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l485_48571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l485_48534

noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

theorem range_of_g :
  ∀ y ∈ Set.Icc (-Real.sqrt 3 / 2) 1,
    ∃ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 3),
      g x = y ∧
      ∀ z ∈ Set.Icc (-Real.pi / 12) (Real.pi / 3),
        g z ∈ Set.Icc (-Real.sqrt 3 / 2) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l485_48534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_symmetry_axis_l485_48543

/-- Proves that x = 11π/24 is one of the symmetry axes of the function y = sin(4x - π/3) -/
theorem sin_symmetry_axis : 
  ∃ (k : ℤ), (11 * Real.pi) / 24 = (k * Real.pi) / 4 + (5 * Real.pi) / 24 ∧ 
  ∀ (x : ℝ), Real.sin (4 * ((k * Real.pi) / 4 + (5 * Real.pi) / 24) - Real.pi / 3) = 
              Real.sin (4 * x - Real.pi / 3) → 
              x = (11 * Real.pi) / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_symmetry_axis_l485_48543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l485_48519

/-- A conic section with two foci -/
structure ConicSection where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Eccentricity of a conic section -/
noncomputable def eccentricity (c : ConicSection) : ℝ :=
  distance c.F₁ c.F₂ / (2 * Real.sqrt ((c.F₁.1 + c.F₂.1)^2 + (c.F₁.2 + c.F₂.2)^2) / 4)

theorem conic_section_eccentricity (Γ : ConicSection) :
  (∃ P : ℝ × ℝ, distance P Γ.F₁ / distance Γ.F₁ Γ.F₂ = 4/3 ∧
                distance P Γ.F₂ / distance Γ.F₁ Γ.F₂ = 2/3) →
  eccentricity Γ = 1/2 ∨ eccentricity Γ = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l485_48519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_triple_l485_48540

theorem pythagorean_triple : 
  ∃! x : Nat × Nat × Nat, x ∈ [(9, 16, 25), (2, 2, 22), (1, 2, 3), (9, 40, 41)] ∧
    let (a, b, c) := x
    a^2 + b^2 = c^2 ∧ a ≤ b ∧ b ≤ c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_triple_l485_48540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zhukovsky_transformation_image_l485_48501

/-- The Zhukovsky (Joukowsky) transformation -/
noncomputable def zhukovsky (z : ℂ) : ℂ := (z + z⁻¹) / 2

/-- The domain in the z-plane -/
def z_domain (z : ℂ) : Prop :=
  0 < Complex.abs z ∧ Complex.abs z < 1 ∧ 0 < Complex.arg z ∧ Complex.arg z < Real.pi/4

/-- The image in the w-plane -/
def w_image (w : ℂ) : Prop :=
  let u := w.re
  let v := w.im
  u^2 - v^2 > 1/2 ∧ u > Real.sqrt 2 / 2 ∧ v < 0

/-- Theorem: The image of z_domain under the Zhukovsky transformation is w_image -/
theorem zhukovsky_transformation_image :
  ∀ z w : ℂ, z_domain z → w = zhukovsky z → w_image w := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zhukovsky_transformation_image_l485_48501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_divisors_count_l485_48576

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (fun d => n % d = 0) (Finset.range (n + 1))

theorem distinct_divisors_count (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  (divisors a).card = 10 ∧ 
  (divisors b).card = 9 ∧ 
  6 ∈ divisors a ∧ 
  6 ∈ divisors b →
  ((divisors a) ∪ (divisors b)).card = 13 :=
by
  sorry

#eval divisors 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_divisors_count_l485_48576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_trig_ratio_l485_48551

theorem inverse_trig_ratio : 
  (Real.arcsin (Real.sqrt 3 / 2) + Real.arccos (-1 / 2)) / Real.arctan (-Real.sqrt 3) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_trig_ratio_l485_48551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerusha_earned_68_l485_48558

/-- Represents the earnings of Lottie in dollars -/
def lottie_earnings : ℝ := sorry

/-- Represents the earnings of Jerusha in dollars -/
def jerusha_earnings : ℝ := sorry

/-- Jerusha earned 4 times as much as Lottie -/
axiom jerusha_lottie_ratio : jerusha_earnings = 4 * lottie_earnings

/-- The combined earnings of Jerusha and Lottie were $85 -/
axiom total_earnings : jerusha_earnings + lottie_earnings = 85

/-- Theorem: Jerusha's earnings were $68 -/
theorem jerusha_earned_68 : jerusha_earnings = 68 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerusha_earned_68_l485_48558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_percent_increase_l485_48556

def question_value : Fin 15 → ℕ
  | 1 => 150
  | 2 => 250
  | 3 => 400
  | 4 => 600
  | 5 => 1100
  | 6 => 2300
  | 7 => 4700
  | 8 => 9500
  | 9 => 19000
  | 10 => 38000
  | 11 => 76000
  | 12 => 150000
  | 13 => 300000
  | 14 => 600000
  | 15 => 1200000

def percent_increase (start : Fin 15) (endQ : Fin 15) : ℚ :=
  (question_value endQ - question_value start) / question_value start * 100

theorem smallest_percent_increase :
  percent_increase 1 4 < percent_increase 2 6 ∧
  percent_increase 1 4 < percent_increase 5 10 ∧
  percent_increase 1 4 < percent_increase 9 15 := by
  sorry

#eval percent_increase 1 4
#eval percent_increase 2 6
#eval percent_increase 5 10
#eval percent_increase 9 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_percent_increase_l485_48556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l485_48533

theorem election_votes : ∃ (total_votes : ℕ),
  let candidate1_percent : ℚ := 55 / 100
  let invalid_percent : ℚ := 20 / 100
  let candidate2_votes : ℕ := 1980
  let valid_votes_percent : ℚ := 1 - invalid_percent
  let candidate2_percent : ℚ := 1 - candidate1_percent
  (candidate2_votes : ℚ) = candidate2_percent * valid_votes_percent * total_votes ∧
  total_votes = 5500 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l485_48533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_increase_calculation_l485_48549

noncomputable def grade_increase_per_hour (video_game_time : ℝ) (study_time_ratio : ℝ) (initial_grade : ℝ) (final_grade : ℝ) : ℝ :=
  let study_time := video_game_time * study_time_ratio
  (final_grade - initial_grade) / study_time

theorem grade_increase_calculation :
  grade_increase_per_hour 9 (1/3) 0 45 = 15 := by
  unfold grade_increase_per_hour
  simp
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_increase_calculation_l485_48549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_semigroup_is_commutative_and_associative_l485_48594

class BinaryOperation (S : Type) where
  op : S → S → S

infix:70 " * " => BinaryOperation.op

class SpecialSemigroup (S : Type) extends BinaryOperation S where
  idempotent : ∀ a : S, a * a = a
  rotational : ∀ a b c : S, (a * b) * c = (b * c) * a

theorem special_semigroup_is_commutative_and_associative 
  (S : Type) [SpecialSemigroup S] : 
  (∀ a b : S, a * b = b * a) ∧ 
  (∀ a b c : S, (a * b) * c = a * (b * c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_semigroup_is_commutative_and_associative_l485_48594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relationship_l485_48555

theorem number_relationship (a b c : ℝ) 
  (ha : a = Real.rpow 0.6 0.3)
  (hb : b = Real.log 3 / Real.log 0.6)
  (hc : c = Real.log Real.pi) : 
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relationship_l485_48555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_x_value_l485_48536

-- Define the points
def A : ℝ × ℝ × ℝ := (-1, 2, 3)
def B : ℝ × ℝ × ℝ := (2, -4, 1)
def C (x : ℝ) : ℝ × ℝ × ℝ := (x, -1, -3)

-- Define vectors
def AB : ℝ × ℝ × ℝ := (B.1 - A.1, B.2.1 - A.2.1, B.2.2 - A.2.2)
def AC (x : ℝ) : ℝ × ℝ × ℝ := ((C x).1 - A.1, (C x).2.1 - A.2.1, (C x).2.2 - A.2.2)

-- Define dot product
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2.1 * v2.2.1 + v1.2.2 * v2.2.2

-- Theorem statement
theorem right_triangle_x_value :
  dot_product AB (AC (-11)) = 0 :=
by
  -- Expand the definitions and calculate
  simp [dot_product, AB, AC, A, B, C]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_x_value_l485_48536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_last_two_digits_factorial_sum_l485_48587

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_last_two_digits (n : ℕ) : ℕ := 
  let digits := last_two_digits n
  (digits / 10) + (digits % 10)

theorem sum_last_two_digits_factorial_sum : 
  sum_last_two_digits (Finset.sum (Finset.range 2006) factorial) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_last_two_digits_factorial_sum_l485_48587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_theorem_l485_48521

/-- Represents a pizza with 16 slices and three types of toppings -/
structure Pizza :=
  (pepperoni : Finset Nat)
  (mushroom : Finset Nat)
  (extra_cheese : Finset Nat)

/-- The properties of our specific pizza -/
def special_pizza (p : Pizza) : Prop :=
  -- There are 16 slices in total
  Finset.card (p.pepperoni ∪ p.mushroom ∪ p.extra_cheese) = 16 ∧
  -- Each slice has at least one topping
  p.pepperoni ∪ p.mushroom ∪ p.extra_cheese = Finset.range 16 ∧
  -- 8 slices have pepperoni
  Finset.card p.pepperoni = 8 ∧
  -- 12 slices have mushrooms
  Finset.card p.mushroom = 12 ∧
  -- No slices have all three toppings
  (p.pepperoni ∩ p.mushroom ∩ p.extra_cheese).card = 0

theorem pizza_theorem (p : Pizza) (h : special_pizza p) :
  Finset.card (p.pepperoni ∩ p.mushroom) = 4 := by
  sorry

#check pizza_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_theorem_l485_48521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l485_48573

theorem imaginary_part_of_z (z : ℂ) : z = (2 - Complex.I) * Complex.I → z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l485_48573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_point_same_side_l485_48592

-- Define the line equation
def line_equation (x y : ℝ) : ℝ := 3 * y - 2 * x + 1

-- Define the reference point
def reference_point : ℝ × ℝ := (0, -1)

-- Define the points to check
def points_to_check : List (ℝ × ℝ) := [(1, 1), (2, 3), (4, 2)]

-- Define a function to check if a point is on the same side as the reference point
noncomputable def same_side (p : ℝ × ℝ) : Bool :=
  (line_equation p.1 p.2) * (line_equation reference_point.1 reference_point.2) < 0

theorem one_point_same_side :
  (points_to_check.filter same_side).length = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_point_same_side_l485_48592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_covering_l485_48560

/-- Predicate for an equilateral triangle in 2D space -/
def EquilateralTriangle (S : Set (Fin 2 → ℝ)) : Prop :=
  ∃ (A B C : Fin 2 → ℝ), S = {A, B, C} ∧ 
    ‖A - B‖ = ‖B - C‖ ∧ 
    ‖B - C‖ = ‖C - A‖ ∧
    ‖C - A‖ = ‖A - B‖

/-- Area of a set in 2D space -/
noncomputable def Area (S : Set (Fin 2 → ℝ)) : ℝ := sorry

/-- An equilateral triangle cannot be covered by two smaller equilateral triangles -/
theorem equilateral_triangle_covering (T S1 S2 : Set (Fin 2 → ℝ)) : 
  EquilateralTriangle T → 
  EquilateralTriangle S1 → 
  EquilateralTriangle S2 → 
  Area S1 < Area T → 
  Area S2 < Area T → 
  Area S1 + Area S2 < Area T :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_covering_l485_48560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_nineteen_l485_48545

theorem divisible_by_nineteen (n : ℕ) : ∃ k : ℤ, (3^(3*n+2) + 5 * 2^(3*n+1) : ℤ) = 19 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_nineteen_l485_48545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_substance_C_has_most_nitrogen_atoms_l485_48500

-- Define Avogadro's constant
noncomputable def avogadro_constant : ℝ := 6.02e23

-- Define the number of moles for each substance
noncomputable def moles_A : ℝ := 0.1
noncomputable def moles_B : ℝ := 0.1
noncomputable def moles_C : ℝ := 1.204e23 / avogadro_constant
noncomputable def moles_D : ℝ := 0.2

-- Define the number of nitrogen atoms per molecule for each substance
def N_per_molecule_A : ℕ := 1
def N_per_molecule_B : ℕ := 2
def N_per_molecule_C : ℕ := 2
def N_per_molecule_D : ℕ := 1

-- Calculate the number of nitrogen atoms for each substance
noncomputable def N_atoms_A : ℝ := moles_A * avogadro_constant * N_per_molecule_A
noncomputable def N_atoms_B : ℝ := moles_B * avogadro_constant * N_per_molecule_B
noncomputable def N_atoms_C : ℝ := moles_C * avogadro_constant * N_per_molecule_C
noncomputable def N_atoms_D : ℝ := moles_D * avogadro_constant * N_per_molecule_D

-- Theorem: Substance C contains the most nitrogen atoms
theorem substance_C_has_most_nitrogen_atoms :
  N_atoms_C > N_atoms_A ∧ N_atoms_C > N_atoms_B ∧ N_atoms_C > N_atoms_D :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_substance_C_has_most_nitrogen_atoms_l485_48500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nickys_pace_l485_48502

/-- Proves that Nicky's running pace is 3 meters per second given the race conditions -/
theorem nickys_pace (race_length : ℝ) (head_start : ℝ) (cristina_pace : ℝ) (catch_up_time : ℝ)
  (h1 : race_length = 400)
  (h2 : head_start = 12)
  (h3 : cristina_pace = 5)
  (h4 : catch_up_time = 30) :
  race_length / catch_up_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nickys_pace_l485_48502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l485_48535

noncomputable def complex_number : ℂ := (3 - Complex.I) / (1 - Complex.I)

theorem complex_number_in_first_quadrant :
  Complex.re complex_number > 0 ∧ Complex.im complex_number > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l485_48535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_intersections_and_origin_line_AC_equation_x_coordinate_range_l485_48580

-- Define the circles and line
def circle_M (x y : ℝ) : Prop := 2*x^2 + 2*y^2 - 8*x - 8*y - 1 = 0
def circle_N (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 6 = 0
def line_l (x y : ℝ) : Prop := x + y - 9 = 0

-- Define the angle condition
def angle_BAC_45 (A B C : ℝ × ℝ) : Prop := sorry

-- Define the condition that AB passes through the center of M
def AB_through_M_center (A B : ℝ × ℝ) : Prop := sorry

-- Define the condition that B and C are on circle M
def B_C_on_M (B C : ℝ × ℝ) : Prop := circle_M B.1 B.2 ∧ circle_M C.1 C.2

theorem circle_through_intersections_and_origin :
  ∀ x y : ℝ, x^2 + y^2 - (50/11)*x - (50/11)*y = 0 ↔
  (∃ lambda : ℝ, lambda ≠ -2 ∧ 2*x^2 + 2*y^2 - 8*x - 8*y - 1 + lambda*(x^2 + y^2 + 2*x + 2*y - 6) = 0) :=
by sorry

theorem line_AC_equation (A B C : ℝ × ℝ) :
  A.1 = 4 ∧ line_l A.1 A.2 ∧ angle_BAC_45 A B C ∧ AB_through_M_center A B ∧ B_C_on_M B C →
  (5*C.1 + C.2 - 25 = 0 ∨ C.1 - 5*C.2 + 21 = 0) :=
by sorry

theorem x_coordinate_range (A : ℝ × ℝ) :
  line_l A.1 A.2 ∧ (∃ B C : ℝ × ℝ, angle_BAC_45 A B C ∧ AB_through_M_center A B ∧ B_C_on_M B C) →
  3 ≤ A.1 ∧ A.1 ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_intersections_and_origin_line_AC_equation_x_coordinate_range_l485_48580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_rate_equation_l485_48588

/-- Represents the hourly production rate of worker B -/
def x : ℝ := sorry

/-- Worker A produces 8 more components per hour than worker B -/
def worker_a_rate (x : ℝ) : ℝ := x + 8

/-- Time taken by worker A to produce 600 components -/
noncomputable def time_a (x : ℝ) : ℝ := 600 / worker_a_rate x

/-- Time taken by worker B to produce 400 components -/
noncomputable def time_b (x : ℝ) : ℝ := 400 / x

/-- Theorem stating that the equation correctly represents the relationship 
    between the production rates of worker A and worker B -/
theorem production_rate_equation (x : ℝ) (h : x > 0) : 
  time_a x = time_b x ↔ 600 / (x + 8) = 400 / x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_rate_equation_l485_48588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_value_l485_48531

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the condition function
def satisfies_condition (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  (∃ (m n : ℤ), m > 0 ∧ n > 0 ∧
    (Real.sqrt (log10 a) : ℝ) = m ∧
    (Real.sqrt (log10 b) : ℝ) = n ∧
    (log10 (Real.sqrt a) : ℝ) ∈ Set.range (Int.cast : ℤ → ℝ) ∧
    (log10 (Real.sqrt b) : ℝ) ∈ Set.range (Int.cast : ℤ → ℝ) ∧
    m + n + (log10 (Real.sqrt a) : ℝ) + (log10 (Real.sqrt b) : ℝ) = 100)

-- State the theorem
theorem product_value {a b : ℝ} (h : satisfies_condition a b) :
  a * b = (10 : ℝ) ^ 164 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_value_l485_48531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thursday_friday_speed_is_nine_l485_48554

/-- Represents the bus driver's schedule and travel data -/
structure BusDriverData where
  hours_per_day : ℚ
  days_per_week : ℕ
  mon_wed_speed : ℚ
  total_distance : ℚ

/-- Calculates the average speed for Thursday and Friday -/
noncomputable def thursday_friday_speed (data : BusDriverData) : ℚ :=
  let mon_wed_days : ℕ := 3
  let mon_wed_hours : ℚ := data.hours_per_day * mon_wed_days
  let mon_wed_distance : ℚ := mon_wed_hours * data.mon_wed_speed
  let thur_fri_days : ℕ := data.days_per_week - mon_wed_days
  let thur_fri_hours : ℚ := data.hours_per_day * thur_fri_days
  let thur_fri_distance : ℚ := data.total_distance - mon_wed_distance
  thur_fri_distance / thur_fri_hours

/-- Theorem stating that the average speed from Thursday to Friday is 9 km/h -/
theorem thursday_friday_speed_is_nine (data : BusDriverData)
    (h1 : data.hours_per_day = 2)
    (h2 : data.days_per_week = 5)
    (h3 : data.mon_wed_speed = 12)
    (h4 : data.total_distance = 108) :
    thursday_friday_speed data = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thursday_friday_speed_is_nine_l485_48554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_a_eq_l485_48583

/-- Function 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

/-- Function 2 -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 3)*x + 1

/-- p: Function 1 has 1 zero point in the interval (1,2) -/
def p (a : ℝ) : Prop := ∃! x, 1 < x ∧ x < 2 ∧ f a x = 0

/-- q: Function 2 intersects the x-axis at two distinct points -/
def q (a : ℝ) : Prop := ∃ x y, x ≠ y ∧ g a x = 0 ∧ g a y = 0

/-- The range of a satisfying the given conditions -/
def range_a : Set ℝ := {a | (¬(p a ∧ q a)) ∧ (p a ∨ q a)}

theorem range_a_eq : range_a = Set.Iic 0 ∪ (Set.Icc (1/2) 1) ∪ Set.Ioi (5/2) := by
  sorry

#check range_a_eq

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_a_eq_l485_48583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_calculation_l485_48577

noncomputable def frustum_rainfall (D1 D2 H h : ℝ) : ℝ :=
  let k := h / H
  let Dw := D1 - k * (D1 - D2)
  let V_water := (1/3) * Real.pi * h * ((Dw/2)^2 + (Dw/2)*(D2/2) + (D2/2)^2)
  let A := Real.pi * (D1/2)^2
  V_water / A

theorem rainfall_calculation :
  frustum_rainfall 28 12 18 9 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_calculation_l485_48577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l485_48559

theorem expression_equality : 
  |(-Real.sqrt 3)| + 2 * Real.cos (60 * π / 180) - (Real.pi - 2020)^(0 : ℕ) + (1/3)^(-1 : ℤ) = Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l485_48559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_midpoint_area_fraction_l485_48505

theorem square_midpoint_area_fraction (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let p : ℝ × ℝ := (0, s/2)
  let q : ℝ × ℝ := (s, s/2)
  let above_pq_area := s * (s/2)
  above_pq_area / square_area = (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_midpoint_area_fraction_l485_48505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_similarity_theorem_l485_48585

-- Define the complex plane
variable (C : Type*) [NormedAddCommGroup C] [NormedSpace ℂ C]

-- Define points in the complex plane
variable (z₀ z₁ z₂ z₃ z₄ t₁ t₂ t₃ t₄ : C)

-- Define the similarity relation
def similar (a b c d e f : C) : Prop :=
  ∃ k : ℂ, k ≠ 0 ∧ (b - a) = k • (e - d) ∧ (c - a) = k • (f - d)

-- State the theorem
theorem quadrilateral_similarity_theorem :
  similar C z₀ z₂ z₃ t₄ t₁ t₃ →
  similar C z₀ z₄ z₁ t₃ t₂ t₁ →
  ∃ z₀' : C,
    similar C z₀' z₃ z₄ t₁ t₂ t₄ ∧
    similar C z₀' z₁ z₂ t₃ t₄ t₂ ∧
    ∃ r s : C,
      similar C r t₄ t₁ z₃ z₄ z₂ ∧
      similar C r t₂ t₃ z₁ z₂ z₄ ∧
      similar C s t₃ t₄ z₂ z₃ z₁ ∧
      similar C s t₁ t₂ z₄ z₁ z₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_similarity_theorem_l485_48585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_sum_l485_48584

open Real

theorem min_value_sin_sum (x y : ℝ) 
  (h1 : cos y + cos x = sin (3 * x))
  (h2 : sin (2 * y) - sin (2 * x) = cos (4 * x) - cos (2 * x)) :
  ∃ (m : ℝ), m = -1 - (sqrt (2 + sqrt 2)) / 2 ∧ 
  ∀ (z w : ℝ), cos z + cos w = sin (3 * w) → 
  sin (2 * z) - sin (2 * w) = cos (4 * w) - cos (2 * w) → 
  m ≤ sin z + sin w :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_sum_l485_48584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_formula_l485_48518

/-- The area of a parallelogram with sides a and b, and acute angle α between the diagonals. -/
noncomputable def parallelogram_area (a b α : ℝ) : ℝ :=
  (1/2) * abs (b^2 - a^2) * Real.tan α

/-- Theorem: The area of a parallelogram with sides a and b, and acute angle α between the diagonals,
    is equal to (1/2) |b^2 - a^2| tan α. -/
theorem parallelogram_area_formula (a b α : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
    (h_acute : 0 < α ∧ α < π/2) :
  parallelogram_area a b α = (1/2) * abs (b^2 - a^2) * Real.tan α := by
  -- Unfold the definition of parallelogram_area
  unfold parallelogram_area
  -- The equality holds by definition
  rfl

#check parallelogram_area_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_formula_l485_48518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_decreasing_interval_l485_48557

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * Real.log x

-- Theorem for the tangent line equation
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    y = m * (x - 1) + f 1 ↔ y = -x + 2 :=
by sorry

-- Theorem for the decreasing interval
theorem decreasing_interval :
  ∀ x ∈ Set.Ioo (0 : ℝ) (Real.sqrt 6 / 2),
    ∃ h > 0, f (x + h) < f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_decreasing_interval_l485_48557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_not_in_range_of_g_l485_48595

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then
    ⌈(2 : ℝ) / (x + 3)⌉
  else if x < -3 then
    ⌊(2 : ℝ) / (x + 3)⌋
  else
    0  -- This case is just to make the function total, it's not used in the proof

-- State the theorem
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_not_in_range_of_g_l485_48595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_expansion_side_area_l485_48541

/-- Represents a cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The side area of a cylinder --/
noncomputable def sideArea (c : Cylinder) : ℝ := 2 * Real.pi * c.radius * c.height

/-- Theorem stating that when a cylinder's base circumference is tripled (radius tripled) 
    and height unchanged, the side area is not nine times the original --/
theorem cylinder_expansion_side_area 
  (c : Cylinder) 
  (c_expanded : Cylinder) 
  (h_radius : c_expanded.radius = 3 * c.radius) 
  (h_height : c_expanded.height = c.height) : 
  sideArea c_expanded ≠ 9 * sideArea c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_expansion_side_area_l485_48541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l485_48524

/-- Given a circle with polar equation ρ²-4√2ρcos(θ-π/4)+6=0, this theorem proves:
    1. The ordinary equation
    2. The parametric equation
    3. The maximum and minimum values of xy for points on the circle -/
theorem circle_properties :
  ∃ (x y : ℝ → ℝ),
    (∀ t, (x t)^2 + (y t)^2 - 4*(x t) - 4*(y t) + 6 = 0) ∧
    (∀ t, x t = 2 + Real.sqrt 2 * Real.cos t) ∧
    (∀ t, y t = 2 + Real.sqrt 2 * Real.sin t) ∧
    (∀ t, (x t) * (y t) ≤ 9) ∧
    (∀ t, (x t) * (y t) ≥ 1) ∧
    (∃ t, (x t) * (y t) = 9) ∧
    (∃ t, (x t) * (y t) = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l485_48524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_weight_upper_limit_l485_48529

/-- Arun's weight in kilograms -/
def arun_weight : ℝ := sorry

/-- Upper limit of Arun's weight according to his brother's estimation -/
def brother_upper_limit : ℝ := sorry

/-- The average of different probable weights of Arun -/
def average_weight : ℝ := sorry

theorem arun_weight_upper_limit :
  (65 < arun_weight) ∧ 
  (arun_weight < 72) ∧ 
  (60 < arun_weight) ∧ 
  (arun_weight ≤ brother_upper_limit) ∧
  (arun_weight ≤ 68) ∧
  (average_weight = 67) →
  brother_upper_limit = 68 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_weight_upper_limit_l485_48529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l485_48517

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -2 * Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x, -Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), x ≥ 0 ∧ x ≤ Real.pi / 2 → f x ≤ 5/2) ∧
  (∀ (x : ℝ), x ≥ 0 ∧ x ≤ Real.pi / 2 → f x ≥ 1) ∧
  (∃ (x : ℝ), x ≥ 0 ∧ x ≤ Real.pi / 2 ∧ f x = 5/2) ∧
  (∃ (x : ℝ), x ≥ 0 ∧ x ≤ Real.pi / 2 ∧ f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l485_48517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_validation_l485_48591

noncomputable def line (x : ℝ) : ℝ := (7/4) * x - 11/2

def valid_parameterization (p₀ : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line (p₀.1 + t * v.1) = p₀.2 + t * v.2

theorem parameterization_validation :
  (valid_parameterization (-2, -9) (4, 7)) ∧
  (valid_parameterization (8, 3) (8, 14)) ∧
  (valid_parameterization (3, -1/4) (1/2, 1)) ∧
  ¬(valid_parameterization (2, 1) (2, 7/4)) ∧
  ¬(valid_parameterization (0, -11/2) (8, -14)) :=
by
  sorry

#check parameterization_validation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_validation_l485_48591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l485_48532

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) := Real.log (-x^2 + 2*x + 3)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

-- You can add more theorems or lemmas here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l485_48532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interval_length_at_a_equals_3_l485_48516

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((a^2 + a) * x - 1) / (a^2 * x)

/-- The theorem statement -/
theorem max_interval_length_at_a_equals_3 
  (a : ℝ) 
  (ha : a ≠ 0) 
  (m n : ℝ) 
  (hmn : m < n) 
  (hdom : Set.Icc m n ⊆ {x | x ≠ 0}) 
  (hrange : Set.range (f a) = Set.Icc m n) :
  (∀ b : ℝ, b ≠ 0 → ∃ p q : ℝ, p < q ∧ 
    Set.Icc p q ⊆ {x | x ≠ 0} ∧ 
    Set.range (f b) = Set.Icc p q → 
    n - m ≥ q - p) → 
  a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interval_length_at_a_equals_3_l485_48516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_to_equation_l485_48562

theorem solutions_to_equation (x : ℂ) : 
  x^4 - 16 = 0 ↔ x ∈ ({2, -2, 2*I, -2*I} : Set ℂ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_to_equation_l485_48562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_f_min_value_of_expression_min_value_achieved_l485_48589

noncomputable def f (x m : ℝ) : ℝ := Real.sqrt (|x + 1| + |x - 3| - m)

theorem domain_and_range_of_f (m : ℝ) : 
  (∀ x, f x m ∈ Set.univ) → m ≤ 4 := by sorry

theorem min_value_of_expression (a b : ℝ) :
  a > 0 → b > 0 → 2 / (3 * a + b) + 1 / (a + 2 * b) = 4 → 7 * a + 4 * b ≥ 9 / 4 := by sorry

theorem min_value_achieved (a b : ℝ) :
  a > 0 → b > 0 → 2 / (3 * a + b) + 1 / (a + 2 * b) = 4 → 
  (7 * a + 4 * b = 9 / 4 ↔ b = 2 * a ∧ a = 3 / 20) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_f_min_value_of_expression_min_value_achieved_l485_48589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_movement_under_dilation_l485_48538

/-- A dilation transformation that maps one circle to another --/
structure MyDilation where
  original_center : ℝ × ℝ
  original_radius : ℝ
  dilated_center : ℝ × ℝ
  dilated_radius : ℝ

/-- The distance a point moves under a dilation --/
def distance_moved (d : MyDilation) (p : ℝ × ℝ) : ℝ :=
  sorry

/-- The specific dilation described in the problem --/
def problem_dilation : MyDilation where
  original_center := (3, 3)
  original_radius := 4
  dilated_center := (7, 9)
  dilated_radius := 6

/-- Theorem stating the distance the origin moves under the problem dilation --/
theorem origin_movement_under_dilation :
  distance_moved problem_dilation (0, 0) = 0.5 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_movement_under_dilation_l485_48538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transposition_parity_l485_48512

open Function

/-- A permutation of {1, ..., n} -/
def Permutation (n : ℕ) := {σ : ℕ → ℕ // Function.Bijective σ ∧ ∀ i, i ≤ n ↔ σ i ≤ n}

/-- The number of transpositions needed to obtain a permutation -/
def numTranspositions (n : ℕ) (σ : Permutation n) : ℕ → Prop := sorry

/-- Two ways to obtain the same permutation have the same parity of transpositions -/
theorem transposition_parity (n : ℕ) (σ : Permutation n) (m₁ m₂ : ℕ) 
  (h₁ : numTranspositions n σ m₁) (h₂ : numTranspositions n σ m₂) : 
  Even (m₁ - m₂) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transposition_parity_l485_48512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_2_l485_48570

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 2 else 2 * Real.sin (Real.pi * x / 12) - 1

theorem f_composition_at_2 : f (f 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_2_l485_48570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_lower_bound_l485_48575

/-- Represents a cell in the grid -/
structure Cell where
  value : ℕ

/-- Represents the m × m grid -/
def Grid (m : ℕ) := Fin m → Fin m → Cell

/-- The sum of a row in the grid -/
def rowSum (grid : Grid m) (i : Fin m) : ℕ :=
  (Finset.univ : Finset (Fin m)).sum fun j => (grid i j).value

/-- The sum of a column in the grid -/
def colSum (grid : Grid m) (j : Fin m) : ℕ :=
  (Finset.univ : Finset (Fin m)).sum fun i => (grid i j).value

/-- The sum of all numbers in the grid -/
def totalSum (grid : Grid m) : ℕ :=
  (Finset.univ : Finset (Fin m)).sum fun i =>
    (Finset.univ : Finset (Fin m)).sum fun j => (grid i j).value

/-- The theorem to be proved -/
theorem grid_sum_lower_bound (m : ℕ) (grid : Grid m)
  (h : ∀ i j, (grid i j).value = 0 → rowSum grid i + colSum grid j ≥ m) :
  totalSum grid ≥ (m * m) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_lower_bound_l485_48575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l485_48590

/-- An isosceles triangle with sides of length 7 and 4 has a perimeter of either 18 or 15. -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Triangle inequality
  (a = b ∨ b = c ∨ a = c) →  -- Isosceles condition
  ({a, b, c} : Set ℝ) = {7, 7, 4} ∨ ({a, b, c} : Set ℝ) = {4, 4, 7} →  -- Given side lengths
  a + b + c = 18 ∨ a + b + c = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l485_48590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l485_48564

/-- A function f with given properties -/
noncomputable def f (A ω θ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + θ)

/-- The theorem stating the properties of the function and its monotonically increasing interval -/
theorem function_properties 
  (A ω θ : ℝ) 
  (h_A : A > 0) 
  (h_ω : ω > 0) 
  (h_θ : |θ| < π/2) 
  (h_period : ∀ x, f A ω θ (x + π) = f A ω θ x) 
  (h_min_point : f A ω θ (7*π/12) = -3) :
  (∀ x, f A ω θ x = 3 * Real.sin (2*x + π/3)) ∧ 
  (∃ S₁ S₂ : Set ℝ, S₁ = Set.Icc 0 (π/12) ∧ S₂ = Set.Icc (7*π/12) π ∧
    StrictMonoOn (f A ω θ) S₁ ∧ StrictMonoOn (f A ω θ) S₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l485_48564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_payment_difference_l485_48574

/-- Pizza sharing problem -/
theorem pizza_payment_difference :
  ∀ (total_slices : ℕ) 
    (plain_cost pepperoni_cost : ℚ) 
    (pepperoni_slices jill_plain_slices : ℕ),
  total_slices = 12 →
  plain_cost = 12 →
  pepperoni_cost = 3 →
  pepperoni_slices = 4 →
  jill_plain_slices = 3 →
  let total_cost := plain_cost + pepperoni_cost;
  let jack_slices := total_slices - (pepperoni_slices + jill_plain_slices);
  let jack_payment := (jack_slices : ℚ) * (plain_cost / total_slices);
  let jill_payment := total_cost - jack_payment;
  jill_payment - jack_payment = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_payment_difference_l485_48574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_for_x_less_than_one_exists_x_not_forming_triangle_zero_exists_for_obtuse_triangle_l485_48504

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := a^x + b^x - c^x

-- Define the conditions
variable (a b c : ℝ)
axiom h1 : c > a
axiom h2 : a > 0
axiom h3 : c > b
axiom h4 : b > 0
axiom h5 : a + b > c  -- Triangle inequality

-- Theorem 1
theorem f_positive_for_x_less_than_one :
  ∀ x < 1, f a b c x > 0 := by sorry

-- Theorem 2
theorem exists_x_not_forming_triangle :
  ∃ x : ℝ, ¬(a^x + b^x > c^x ∧ b^x + c^x > a^x ∧ c^x + a^x > b^x) := by sorry

-- Define obtuse triangle
def is_obtuse_triangle (a b c : ℝ) : Prop := a^2 + b^2 < c^2

-- Theorem 3
theorem zero_exists_for_obtuse_triangle :
  is_obtuse_triangle a b c → ∃ x ∈ Set.Ioo 1 2, f a b c x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_for_x_less_than_one_exists_x_not_forming_triangle_zero_exists_for_obtuse_triangle_l485_48504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_ratio_l485_48548

/-- For a right triangle with legs a and b, hypotenuse c, and angle θ between legs a and b,
    the maximum value of (a + b + c) / (a cos θ + b) is 3/2. -/
theorem right_triangle_max_ratio (a b c θ : ℝ) (h_right : a^2 + b^2 = c^2)
    (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_angle : 0 < θ ∧ θ < π/2)
    (h_cos : Real.cos θ = b / c) : 
    (∀ x y z α, x^2 + y^2 = z^2 → x > 0 → y > 0 → z > 0 → 0 < α → α < π/2 → Real.cos α = y / z →
      (x + y + z) / (x * Real.cos α + y) ≤ (a + b + c) / (a * Real.cos θ + b)) ∧
    (a + b + c) / (a * Real.cos θ + b) = 3/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_ratio_l485_48548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinguishable_colorings_l485_48527

/-- A coloring of a cube is a function from the set of faces to the set of colors. -/
def Coloring := Fin 6 → Fin 4

/-- Two colorings are equivalent if they can be transformed into each other by a rotation. -/
def Equivalent (c1 c2 : Coloring) : Prop := sorry

/-- A coloring is distinguishable if it's not equivalent to any other coloring. -/
def Distinguishable (c : Coloring) : Prop := ∀ c', c ≠ c' → ¬Equivalent c c'

/-- The set of all distinguishable colorings. -/
def DistinguishableColorings : Set Coloring :=
  {c | Distinguishable c}

/-- Instance to show that DistinguishableColorings is finite -/
instance : Fintype DistinguishableColorings := sorry

/-- The number of distinguishable colorings of a cube with 4 colors is 37. -/
theorem num_distinguishable_colorings :
  Fintype.card DistinguishableColorings = 37 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinguishable_colorings_l485_48527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_point_exists_l485_48578

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The theorem stating that no point exists satisfying the given condition -/
theorem no_point_exists : ¬ ∃ (x y : ℝ), 
  distance x y (-1) 0 + distance x y 1 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_point_exists_l485_48578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_g_range_l485_48537

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * (f x)^2 - f (x + Real.pi / 4) - 1

-- Theorem for the interval of monotonic increase of f
theorem f_monotonic_increase :
  ∃ (a b : ℝ), a = 0 ∧ b = Real.pi / 6 ∧
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), 
  (∀ y ∈ Set.Icc a b, x < y → f x < f y) ∧
  (∀ y ∈ Set.Ioo 0 (Real.pi / 2), y < a ∨ b < y → ¬(∀ z ∈ Set.Icc y (y + Real.pi / 6), f y < f z)) :=
by
  sorry

-- Theorem for the range of g
theorem g_range :
  Set.range (fun x => g x) = Set.Icc (-3) (3 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_g_range_l485_48537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_fraction_equality_l485_48546

theorem trig_fraction_equality : 
  (Real.sin (40 * π / 180) - Real.cos (10 * π / 180)) / 
  (Real.sin (10 * π / 180) - Real.cos (40 * π / 180)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_fraction_equality_l485_48546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_twelve_l485_48510

/-- Represents the properties of a car and its journey -/
structure CarJourney where
  speed : ℚ  -- Speed in miles per hour
  fuelEfficiency : ℚ  -- Miles per gallon
  travelTime : ℚ  -- Travel time in hours
  fractionUsed : ℚ  -- Fraction of tank used during the journey

/-- Calculates the tank capacity given a car journey -/
def tankCapacity (journey : CarJourney) : ℚ :=
  (journey.speed * journey.travelTime / journey.fuelEfficiency) / journey.fractionUsed

/-- Theorem stating that the tank capacity is 12 gallons for the given journey -/
theorem tank_capacity_is_twelve :
  let journey : CarJourney := {
    speed := 40,
    fuelEfficiency := 40,
    travelTime := 5,
    fractionUsed := 5/12
  }
  tankCapacity journey = 12 := by
  -- Proof goes here
  sorry

#eval tankCapacity {
  speed := 40,
  fuelEfficiency := 40,
  travelTime := 5,
  fractionUsed := 5/12
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_twelve_l485_48510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l485_48530

/-- The length of the rectangular garden in feet -/
def length : ℝ := 60

/-- The width of the rectangular garden in feet -/
def width : ℝ := 12

/-- The perimeter of the rectangular garden in feet -/
def perimeter : ℝ := 2 * (length + width)

/-- The area of the rectangular garden in square feet -/
def rectangular_area : ℝ := length * width

/-- The radius of the circular garden in feet -/
noncomputable def radius : ℝ := perimeter / (2 * Real.pi)

/-- The area of the circular garden in square feet -/
noncomputable def circular_area : ℝ := Real.pi * radius^2

/-- The difference in area between the circular and rectangular gardens -/
noncomputable def area_difference : ℝ := circular_area - rectangular_area

theorem garden_area_increase :
  ∃ ε > 0, |area_difference - 929| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l485_48530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enchiladas_and_tacos_price_l485_48586

/-- The price of an enchilada in dollars -/
def e : ℚ := sorry

/-- The price of a taco in dollars -/
def t : ℚ := sorry

/-- The first pricing condition -/
axiom condition1 : 4 * e + 5 * t = 4

/-- The second pricing condition -/
axiom condition2 : 5 * e + 3 * t = 38/10

/-- The third pricing condition -/
axiom condition3 : 7 * e + 6 * t = 61/10

/-- The theorem to prove -/
theorem enchiladas_and_tacos_price : 4 * e + 7 * t = 475/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enchiladas_and_tacos_price_l485_48586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_not_always_greater_than_all_data_l485_48563

theorem mean_not_always_greater_than_all_data : ¬ ∀ (data : List ℝ), data.length > 0 → 
  ∀ x ∈ data, (data.sum / (data.length : ℝ)) > x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_not_always_greater_than_all_data_l485_48563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l485_48553

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l485_48553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_proof_l485_48547

-- Define a structure to represent the angle properties
structure AngleProperties (θ : Real) :=
  (vertex_at_origin : Prop)
  (initial_side_on_positive_x_axis : Prop)
  (terminal_side_on_line : (Real → Real → Prop) → Prop)

-- Theorem statement
theorem trigonometric_ratio_proof (θ : Real) 
  (h : AngleProperties θ)
  (h_tan : Real.tan θ = -4/3) :
  (Real.cos (π / 2 + θ) - Real.sin (-π - θ)) / 
  (Real.cos (11 * π / 2 - θ) + Real.sin (9 * π / 2 + θ)) = 8 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_proof_l485_48547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_percentage_l485_48572

-- Define the time it takes John to complete the job alone
noncomputable def john_time : ℝ := 2

-- Define the time it takes John and David to complete the job together
noncomputable def combined_time : ℝ := 1

-- Define John's work rate
noncomputable def john_rate : ℝ := 1 / john_time

-- Define the combined work rate of John and David
noncomputable def combined_rate : ℝ := 1 / combined_time

-- Define David's work rate
noncomputable def david_rate : ℝ := combined_rate - john_rate

-- Theorem to prove
theorem david_percentage : (david_rate / combined_rate) * 100 = 50 := by
  -- Expand the definitions
  unfold david_rate combined_rate john_rate john_time combined_time
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_percentage_l485_48572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trigonometric_expression_l485_48515

theorem min_value_trigonometric_expression :
  ∀ x : ℝ, (Real.sin x)^4 + (Real.cos x)^4 + 3 ≥ 3/8 * ((Real.sin x)^2 + (Real.cos x)^2 + 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trigonometric_expression_l485_48515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l485_48581

noncomputable section

/-- A function f is odd if f(-x) = -f(x) for all x in the domain of f -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = ln|a + 1/(1-x)| + b -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  IsOdd (f a b) → a = -1/2 ∧ b = Real.log 2 := by
  sorry

#check odd_function_values

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l485_48581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l485_48582

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 4 * a
  else Real.log x / Real.log a

theorem f_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (6/5 ≤ a ∧ a < 6) := by
  sorry

#check f_increasing_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l485_48582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_translation_reflection_l485_48509

-- Define a function that represents the logarithm
noncomputable def log : ℝ → ℝ := Real.log

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x => log (1 - x)

-- Theorem statement
theorem log_translation_reflection (x : ℝ) : 
  f x = (log ∘ (λ y => -y) ∘ (λ z => z + 1)) x :=
by
  -- Expand the definition of f
  unfold f
  -- Expand the composition of functions
  simp [Function.comp]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_translation_reflection_l485_48509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l485_48598

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 : ℝ)^x * (8 : ℝ)^y = 16) : 
  (2 : ℝ)^(-1 + Real.log (2*x)) + (Real.log 9)^(27*y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l485_48598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l485_48599

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem: The time taken for a train of length 100 m, traveling at 60 kmph, 
    to cross a bridge of length 80 m is approximately 10.8 seconds -/
theorem train_crossing_bridge_time :
  ∃ ε > 0, |train_crossing_time 100 60 80 - 10.8| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l485_48599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_ratio_l485_48539

/-- The ratio of areas between a small equilateral triangle cut from a larger one -/
theorem equilateral_triangle_area_ratio : 
  ∀ (large_side small_side : ℝ),
  large_side = 12 →
  small_side = 3 →
  let large_area := (Real.sqrt 3 / 4) * large_side^2
  let small_area := (Real.sqrt 3 / 4) * small_side^2
  let remaining_area := large_area - small_area
  (small_area / remaining_area) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_ratio_l485_48539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abs_z_plus_two_l485_48596

/-- Given a complex number z = (x-2) + yi where x and y are real numbers,
    and |z| = 2, the maximum value of |z+2| is 4. -/
theorem max_abs_z_plus_two :
  (⨆ (z : ℂ) (h : Complex.abs z = 2), Complex.abs (z + 2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abs_z_plus_two_l485_48596
