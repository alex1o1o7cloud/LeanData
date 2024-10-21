import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_between_tangents_l245_24552

/-- The parabola C₁ -/
def C₁ (x y : ℝ) : Prop := y^2 = 4*x

/-- The circle A -/
def circleA (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 2

/-- The angle between two tangents from a point to a circle -/
noncomputable def angle_between_tangents (x y : ℝ) : ℝ :=
  2 * Real.arcsin (1 / (2 * Real.sqrt ((x - 3)^2 + y^2)))

/-- The maximum angle between tangents theorem -/
theorem max_angle_between_tangents :
  ∃ (x y : ℝ), C₁ x y ∧ (∀ (x' y' : ℝ), C₁ x' y' →
    angle_between_tangents x y ≥ angle_between_tangents x' y') ∧
    angle_between_tangents x y = π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_between_tangents_l245_24552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_set_size_l245_24539

def is_valid_set (S : Set ℕ) : Prop :=
  ∀ x, x ∈ S → x ≤ 2023 ∧
  ∀ a b, a ∈ S → b ∈ S → a ≠ b → ¬(((a + b) : ℤ) ∣ ((a - b) : ℤ))

theorem max_valid_set_size :
  ∃ (S : Finset ℕ), (∀ x ∈ S, is_valid_set {y | y ∈ S}) ∧ 
  S.card = 675 ∧
  ∀ (T : Finset ℕ), (∀ x ∈ T, is_valid_set {y | y ∈ T}) → T.card ≤ 675 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_set_size_l245_24539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l245_24550

def binomial_sum (n : ℕ) : ℕ := 2^n

noncomputable def general_term (n r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r : ℝ) * (-1)^r * x^((n : ℝ)/2 - (5*r : ℝ)/6)

theorem expansion_properties (n : ℕ) (h : binomial_sum n = 128) :
  n = 7 ∧
  (∀ k, k ≠ 3 ∧ k ≠ 4 → Nat.choose n k ≤ Nat.choose n 3) ∧
  (∀ k, k ≠ 3 ∧ k ≠ 4 → Nat.choose n k ≤ Nat.choose n 4) ∧
  (∀ k, k ≠ 4 → |general_term n k 1| ≤ |general_term n 4 1|) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l245_24550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_number_appearance_l245_24579

/-- Represents a 9x9 grid of numbers -/
def Grid := Fin 9 → Fin 9 → ℕ

/-- Checks if a row has at most three different numbers -/
def row_valid (g : Grid) (i : Fin 9) : Prop :=
  (Finset.image (λ j ↦ g i j) Finset.univ).card ≤ 3

/-- Checks if a column has at most three different numbers -/
def col_valid (g : Grid) (j : Fin 9) : Prop :=
  (Finset.image (λ i ↦ g i j) Finset.univ).card ≤ 3

/-- Checks if a number appears at least 3 times in a row -/
def appears_thrice_in_row (g : Grid) (n : ℕ) (i : Fin 9) : Prop :=
  (Finset.filter (λ j ↦ g i j = n) Finset.univ).card ≥ 3

/-- Checks if a number appears at least 3 times in a column -/
def appears_thrice_in_col (g : Grid) (n : ℕ) (j : Fin 9) : Prop :=
  (Finset.filter (λ i ↦ g i j = n) Finset.univ).card ≥ 3

theorem grid_number_appearance (g : Grid) 
  (h_row : ∀ i, row_valid g i) 
  (h_col : ∀ j, col_valid g j) :
  ∃ n, ∃ i j, appears_thrice_in_row g n i ∧ appears_thrice_in_col g n j :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_number_appearance_l245_24579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_task_time_l245_24536

/-- The time taken by Alpha alone to complete the job -/
noncomputable def A : ℝ := sorry

/-- The time taken by Beta alone to complete the job -/
noncomputable def B : ℝ := sorry

/-- The time taken by Gamma alone to complete the job -/
noncomputable def C : ℝ := sorry

/-- The time taken by Alpha and Beta together to complete the main task -/
noncomputable def h : ℝ := sorry

/-- The theorem stating the conditions and the result to be proved -/
theorem main_task_time :
  (1/A + 1/B + 1/C = 1/(A - 7)) →
  (1/A + 1/B + 1/C = 1/(B - 2)) →
  (1/A + 1/B + 1/C = 2/C) →
  (1/h = 1/A + 1/B) →
  (h = 35/12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_task_time_l245_24536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_alpha_equals_negative_two_implies_result_l245_24512

theorem tan_pi_minus_alpha_equals_negative_two_implies_result 
  (α : ℝ) 
  (h : Real.tan (π - α) = -2) : 
  1 / (Real.cos (2 * α) + Real.cos α ^ 2) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_alpha_equals_negative_two_implies_result_l245_24512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_and_g_l245_24573

-- Function 1
noncomputable def f (x : ℝ) : ℝ := x^3 * Real.log x

-- Function 2
noncomputable def g (x : ℝ) : ℝ := (1 - x^2) / Real.exp x

theorem derivative_f_and_g :
  (∀ x, x > 0 → deriv f x = 3 * x^2 * Real.log x + x^2) ∧
  (∀ x, deriv g x = (x^2 - 2*x - 1) / Real.exp x) := by
  sorry

#check derivative_f_and_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_and_g_l245_24573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l245_24587

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and a point (3, 4) on its terminal side, prove that cos(2α) = -7/25 -/
theorem cos_double_angle_special_case (α : ℝ) 
  (h1 : ∃ (x y : ℝ), x = 3 ∧ y = 4 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) : 
  Real.cos (2 * α) = -7/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l245_24587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_speed_200m_24s_l245_24575

/-- The speed of an athlete running a race -/
noncomputable def athlete_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: An athlete running 200 metres in 24 seconds has a speed of approximately 8.33 m/s -/
theorem athlete_speed_200m_24s :
  ∃ (speed : ℝ), abs (athlete_speed 200 24 - speed) < 0.01 ∧ abs (speed - 8.33) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_speed_200m_24s_l245_24575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l245_24568

noncomputable def f (x : ℝ) : ℝ := x + 1/x

theorem f_properties :
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → f x ≤ 17/4) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → f x ≥ 2) ∧
  f 4 = 17/4 ∧ f 1 = 2 := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l245_24568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equation_solutions_l245_24576

open Real

theorem tan_equation_solutions :
  ∃ (s : Finset ℝ), s.toSet ⊆ { x : ℝ | 0 ≤ x ∧ x ≤ Real.arctan 1000 } ∧
    (∀ x ∈ s, tan x = tan (tan x)) ∧
    s.card = 320 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equation_solutions_l245_24576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_with_internal_point_l245_24551

/-- A triangle in a 2D plane. -/
structure Triangle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] where
  A : V
  B : V
  C : V

/-- A point is inside a triangle. -/
def IsInside {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (D : V) (t : Triangle V) : Prop := sorry

theorem triangle_inequality_with_internal_point
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (t : Triangle V) (D : V)
  (h_inside : IsInside D t)
  (h_equal : ‖t.A - D‖ = ‖t.A - t.B‖) :
  ‖t.A - t.B‖ < ‖t.A - t.C‖ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_with_internal_point_l245_24551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_positive_power_of_two_l245_24523

theorem negation_of_forall_positive_power_of_two :
  (¬ ∀ x : ℝ, (2 : ℝ)^(x - 1) > 0) ↔ (∃ x : ℝ, (2 : ℝ)^(x - 1) ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_positive_power_of_two_l245_24523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition1_is_true_proposition2_not_always_true_correct_answer_is_B_l245_24545

-- Define the propositions using real parts of complex numbers
def proposition1 (a b c : ℂ) : Prop :=
  (Real.sqrt ((a.re^2 - a.im^2) + (b.re^2 - b.im^2)) > Real.sqrt (c.re^2 - c.im^2)) →
  ((a.re^2 - a.im^2) + (b.re^2 - b.im^2) - (c.re^2 - c.im^2) > 0)

def proposition2 (a b c : ℂ) : Prop :=
  ((a.re^2 - a.im^2) + (b.re^2 - b.im^2) - (c.re^2 - c.im^2) > 0) →
  (Real.sqrt ((a.re^2 - a.im^2) + (b.re^2 - b.im^2)) > Real.sqrt (c.re^2 - c.im^2))

-- Theorem stating that proposition1 is true for all complex numbers
theorem proposition1_is_true : ∀ (a b c : ℂ), proposition1 a b c := by
  sorry

-- Theorem stating that proposition2 is not always true for complex numbers
theorem proposition2_not_always_true : ¬(∀ (a b c : ℂ), proposition2 a b c) := by
  sorry

-- The correct answer is B: Proposition1 is correct, but Proposition2 is incorrect
theorem correct_answer_is_B : 
  (∀ (a b c : ℂ), proposition1 a b c) ∧ ¬(∀ (a b c : ℂ), proposition2 a b c) := by
  exact ⟨proposition1_is_true, proposition2_not_always_true⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition1_is_true_proposition2_not_always_true_correct_answer_is_B_l245_24545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_range_inverse_tangent_sum_l245_24558

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Part 1
theorem cosine_sum_range (t : Triangle) (h : t.B = (t.A + t.C) / 2) :
  (1/2 : Real) < Real.cos t.A + Real.cos t.C ∧ Real.cos t.A + Real.cos t.C ≤ 1 := by
  sorry

-- Part 2
theorem inverse_tangent_sum (t : Triangle) 
  (h1 : t.b^2 = t.a * t.c) 
  (h2 : Real.cos t.B = 4/5) :
  1 / Real.tan t.A + 1 / Real.tan t.C = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_range_inverse_tangent_sum_l245_24558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_g_definition_l245_24537

-- Define the function g first
noncomputable def g (x : ℝ) : ℝ := -x^2 + Real.exp (x * Real.log 3) - 1

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 - Real.exp (-x * Real.log 3) else g x + 1

-- State the theorem
theorem odd_function_g_definition (x : ℝ) :
  (∀ y : ℝ, f (-y) = -f y) →  -- f is an odd function
  (∀ z : ℝ, z > 0 → f z = g z + 1) →  -- Definition of f for positive x
  (x > 0 → g x = -x^2 + Real.exp (x * Real.log 3) - 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_g_definition_l245_24537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_comparison_l245_24598

/-- The area of a triangle given its side lengths using Heron's formula -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The theorem stating that the area of the triangle with sides 17, 17, and 12
    is less than the area of the triangle with sides 17, 17, and 16 -/
theorem triangle_area_comparison : 
  triangleArea 17 17 12 < triangleArea 17 17 16 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_comparison_l245_24598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rolles_theorem_l245_24513

theorem rolles_theorem {f : ℝ → ℝ} {a b : ℝ} (hab : a < b)
  (hf : DifferentiableOn ℝ f (Set.Icc a b))
  (hfab : f a = f b) :
  ∃ x ∈ Set.Ioo a b, deriv f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rolles_theorem_l245_24513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l245_24553

/-- Represents an ellipse with equation 3x^2 + 4y^2 = 12 -/
structure Ellipse where
  equation : ∀ (x y : ℝ), 3 * x^2 + 4 * y^2 = 12

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 1/2

/-- The foci coordinates of the ellipse -/
def foci_coordinates (e : Ellipse) : (ℝ × ℝ) × (ℝ × ℝ) := ((-1, 0), (1, 0))

/-- Theorem stating properties of the ellipse -/
theorem ellipse_properties (e : Ellipse) : 
  eccentricity e = 1/2 ∧ foci_coordinates e = ((-1, 0), (1, 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l245_24553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_bounds_l245_24504

theorem function_bounds (a b c : ℝ) 
  (f g : ℝ → ℝ)
  (hf : f = λ x ↦ a * x^2 + b * x + c)
  (hg : g = λ x ↦ a * x + b)
  (h_bound : ∀ x ∈ Set.Icc (-1) 1, |f x| ≤ 1) :
  (|c| ≤ 1) ∧ (∀ x ∈ Set.Icc (-1) 1, |g x| ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_bounds_l245_24504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_of_twelve_l245_24502

theorem sum_of_divisors_of_twelve : 
  (Finset.filter (λ n : ℕ => n > 0 ∧ 12 % n = 0) (Finset.range 13)).sum id = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_of_twelve_l245_24502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l245_24564

theorem division_remainder_problem (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : (x : ℝ) / (y : ℝ) = 86.1)
  (h2 : (y : ℝ) = 70.00000000000398) : 
  x % y = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l245_24564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_resilience_maximizer_l245_24532

noncomputable def probability (x : ℝ) (s : ℕ) : ℝ := x^s / (1 + x + x^2)

def K : List ℕ := [1, 2, 2, 1, 0, 2, 1, 0, 1, 2]

noncomputable def probability_of_K (x : ℝ) : ℝ :=
  (List.map (probability x) K).prod

theorem dragon_resilience_maximizer :
  ∃ (x : ℝ), x > 0 ∧ 
    ∀ (y : ℝ), y > 0 → probability_of_K x ≥ probability_of_K y ∧
    x = (Real.sqrt 97 + 1) / 8 := by
  sorry

#check dragon_resilience_maximizer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_resilience_maximizer_l245_24532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_concurrent_lines_l245_24546

/-- A regular heptagon in the plane -/
structure RegularHeptagon :=
  (center : ℝ × ℝ)
  (vertices : Fin 7 → ℝ × ℝ)

/-- The orientation of a heptagon -/
inductive HeptagonOrientation
  | Positive
  | Negative

/-- Two regular heptagons with the same orientation -/
structure TwoHeptagons :=
  (heptagon1 : RegularHeptagon)
  (heptagon2 : RegularHeptagon)
  (same_orientation : HeptagonOrientation)
  (common_vertex : heptagon1.vertices 0 = heptagon2.vertices 0)

/-- A line in the plane defined by two points -/
def Line (p q : ℝ × ℝ) := {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

/-- The theorem statement -/
theorem heptagon_concurrent_lines (h : TwoHeptagons) :
  ∃ (p : ℝ × ℝ), ∀ i : Fin 6, p ∈ Line (h.heptagon1.vertices (i + 1)) (h.heptagon2.vertices (i + 1)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_concurrent_lines_l245_24546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_squared_sqrt_one_minus_t_squared_l245_24519

-- Define the cubic polynomial f
def f (t : ℝ) : ℝ := 4 * t^3 - 3 * t

-- State the theorem
theorem integral_f_squared_sqrt_one_minus_t_squared :
  (∀ x : ℝ, Real.cos (3 * x) = f (Real.cos x)) →
  ∫ t in Set.Icc 0 1, (f t)^2 * Real.sqrt (1 - t^2) = π / 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_squared_sqrt_one_minus_t_squared_l245_24519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theresa_basketball_scores_l245_24524

def first_ten_scores : List Nat := [9, 5, 4, 7, 6, 2, 4, 8, 3, 7]

def is_valid_score (score : Nat) : Prop :=
  score < 10

theorem theresa_basketball_scores :
  ∀ (score11 score12 : Nat),
  is_valid_score score11 →
  is_valid_score score12 →
  (first_ten_scores.sum + score11) % 11 = 0 →
  (first_ten_scores.sum + score11 + score12) % 12 = 0 →
  score11 * score12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theresa_basketball_scores_l245_24524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l245_24563

-- Define the triangle ABC
def Triangle (a b c A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  0 < A ∧ A < Real.pi/2 ∧ 
  0 < B ∧ B < Real.pi/2 ∧ 
  0 < C ∧ C < Real.pi/2

-- State the theorem
theorem triangle_problem (a b c A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_relation : Real.sqrt 3 * a = 2 * b * Real.sin A)
  (h_sum : a + c = 5)
  (h_b : b = Real.sqrt 7) :
  B = Real.pi/3 ∧ 
  (1/2) * a * b * Real.sin B = (9 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l245_24563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_two_n_over_three_n_minus_one_l245_24556

theorem limit_fraction_two_n_over_three_n_minus_one :
  ∀ ε > 0, ∃ N : ℝ, ∀ n : ℝ, n ≥ N → |2*n/(3*n-1) - 2/3| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_two_n_over_three_n_minus_one_l245_24556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l245_24525

theorem count_integers_in_range : 
  (Finset.filter (fun x : ℕ => 144 ≤ x^2 ∧ x^2 ≤ 256) (Finset.range 17)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l245_24525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_good_pair_l245_24538

-- Define the slopes of the lines
def m1 : ℚ := 4
def m2 : ℚ := 3
def m3 : ℚ := 4
def m4 : ℚ := 3/2
def m5 : ℚ := 1/2

-- Define a function to check if two lines are parallel
def are_parallel (m1 m2 : ℚ) : Bool := m1 = m2

-- Define a function to check if two lines are perpendicular
def are_perpendicular (m1 m2 : ℚ) : Bool := m1 * m2 = -1

-- Define a function to count the number of good pairs
def count_good_pairs (m1 m2 m3 m4 m5 : ℚ) : ℕ :=
  let pairs := [
    (m1, m2), (m1, m3), (m1, m4), (m1, m5),
    (m2, m3), (m2, m4), (m2, m5),
    (m3, m4), (m3, m5),
    (m4, m5)
  ]
  (pairs.filter (fun (a, b) => are_parallel a b || are_perpendicular a b)).length

-- Theorem statement
theorem one_good_pair :
  count_good_pairs m1 m2 m3 m4 m5 = 1 := by
  -- Unfold the definitions
  unfold count_good_pairs
  unfold are_parallel
  unfold are_perpendicular
  -- Evaluate the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_good_pair_l245_24538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_calculation_l245_24560

/-- Given a large equilateral triangle with side length 4 units and a smaller equilateral triangle
    with area one-third of the larger triangle, the median of a trapezoid formed by the sides of
    these triangles is equal to 2 + (2√3)/3 units. -/
theorem trapezoid_median_calculation (large_side : ℝ) (small_area_ratio : ℝ) :
  large_side = 4 →
  small_area_ratio = 1/3 →
  let large_area := (Real.sqrt 3 / 4) * large_side^2
  let small_area := small_area_ratio * large_area
  let small_side := Real.sqrt ((4 * small_area) / Real.sqrt 3)
  (large_side + small_side) / 2 = 2 + (2 * Real.sqrt 3) / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_calculation_l245_24560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_angle_sine_inequality_l245_24521

theorem acute_triangle_angle_sine_inequality (α β γ : ℝ) : 
  0 < α → 0 < β → 0 < γ →  -- angles are positive
  α < π/2 → β < π/2 → γ < π/2 →  -- angles are acute
  α + β + γ = π →  -- sum of angles in a triangle
  α < β → β < γ →  -- given order of angles
  Real.sin (2*α) > Real.sin (2*β) ∧ Real.sin (2*β) > Real.sin (2*γ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_angle_sine_inequality_l245_24521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_in_range_l245_24590

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a^2*Real.log x

-- State the theorem
theorem f_nonnegative_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → f a x ≥ 0) ↔ 
  (a ≥ -2 * Real.exp (3/4) ∧ a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_in_range_l245_24590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_k_range_l245_24592

-- Define the function f as noncomputable
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.sqrt (k * x^2 - 6 * x + k + 8)

-- State the theorem
theorem domain_implies_k_range :
  (∀ k : ℝ, (∀ x : ℝ, ∃ y : ℝ, f k x = y) → k ≥ 1) ∧
  (∀ k : ℝ, k ≥ 1 → ∀ x : ℝ, ∃ y : ℝ, f k x = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_k_range_l245_24592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_from_tax_l245_24509

/-- Represents the tax system of Country X -/
structure TaxSystem where
  base_income : ℝ
  base_rate : ℝ
  excess_rate : ℝ

/-- Calculates the tax for a given income under a specific tax system -/
noncomputable def calculate_tax (ts : TaxSystem) (income : ℝ) : ℝ :=
  if income ≤ ts.base_income then
    income * ts.base_rate
  else
    ts.base_income * ts.base_rate + (income - ts.base_income) * ts.excess_rate

/-- Theorem: Given the tax system and total tax paid, prove the income -/
theorem income_from_tax (ts : TaxSystem) (total_tax : ℝ) (income : ℝ) :
  ts.base_income = 40000 ∧
  ts.base_rate = 0.1 ∧
  ts.excess_rate = 0.2 ∧
  total_tax = 8000 ∧
  calculate_tax ts income = total_tax →
  income = 60000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_from_tax_l245_24509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l245_24534

theorem train_journey_distance 
  (normal_speed : ℝ) 
  (speed_increase : ℝ) 
  (time_difference : ℝ) :
  normal_speed = 25 →
  speed_increase = 5 →
  time_difference = 2 →
  normal_speed * time_difference * (normal_speed + speed_increase) / speed_increase = 300 :=
by
  intro h1 h2 h3
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l245_24534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_550_optimal_order_maximizes_profit_l245_24565

-- Define the cost and price functions
def cost (x : ℝ) : ℝ := 40 * x

noncomputable def price (x : ℝ) : ℝ :=
  if x ≤ 100 then 60
  else 62 - 0.02 * x

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ := price x * x - cost x

-- State the theorem
theorem max_profit_at_550 :
  ∀ x, 0 < x ∧ x ≤ 600 → profit x ≤ profit 550 ∧ profit 550 = 6050 := by
  sorry

-- Define the optimal order quantity
def optimal_order : ℝ := 550

-- State that the optimal order quantity maximizes profit
theorem optimal_order_maximizes_profit :
  ∀ x, 0 < x ∧ x ≤ 600 → profit x ≤ profit optimal_order := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_550_optimal_order_maximizes_profit_l245_24565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_cos_cos_l245_24510

noncomputable def f (x : ℝ) := Real.cos (Real.cos x)

theorem period_of_cos_cos :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S) ∧
  T = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_cos_cos_l245_24510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_slope_bound_l245_24580

/-- The function f(x) = 1/x - x + a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/x - x + a * Real.log x

/-- Theorem: If f(x) has two extreme points, then the slope between them is less than a - 2 -/
theorem extreme_points_slope_bound (a : ℝ) :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
  (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 → x = x₁ ∨ x = x₂) →
  (f a x₁ - f a x₂) / (x₁ - x₂) < a - 2 := by
  sorry

#check extreme_points_slope_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_slope_bound_l245_24580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_l245_24581

theorem det_B_squared {n : Type*} [Fintype n] [DecidableEq n] 
  (B : Matrix n n ℝ) (h : Matrix.det B = 8) : 
  Matrix.det (B ^ 2) = 64 := by
  have h1 : Matrix.det (B ^ 2) = Matrix.det B ^ 2 := by
    exact Matrix.det_pow B 2
  rw [h1, h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_l245_24581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_b_arrives_first_l245_24530

/-- Represents a person's journey from dormitory to classroom -/
structure Journey where
  walkSpeed : ℝ
  runSpeed : ℝ
  distance : ℝ

/-- Calculates the time taken for Person A's journey -/
noncomputable def timeA (j : Journey) : ℝ :=
  (j.distance / 2) * (1 / j.walkSpeed + 1 / j.runSpeed)

/-- Calculates the time taken for Person B's journey -/
noncomputable def timeB (j : Journey) : ℝ :=
  2 * j.distance / (j.walkSpeed + j.runSpeed)

/-- Theorem stating that Person B arrives before Person A -/
theorem person_b_arrives_first (j : Journey) (h1 : j.walkSpeed > 0) (h2 : j.runSpeed > 0) 
    (h3 : j.walkSpeed ≠ j.runSpeed) (h4 : j.distance > 0) : timeB j < timeA j := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_b_arrives_first_l245_24530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_l245_24554

noncomputable def f (x : ℝ) := 2 * (Real.cos (x / 2 + Real.pi / 3))^2 - 1

theorem symmetry_axis (x : ℝ) : 
  f (Real.pi / 3 + (Real.pi / 3 - x)) = f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_l245_24554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_l245_24588

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x > 4}

-- State the theorem
theorem intersection_of_A_and_complement_of_B :
  A ∩ (Bᶜ) = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_l245_24588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_task_assignment_count_l245_24584

theorem task_assignment_count : ℕ := by
  -- Define the number of people and tasks
  let total_people : ℕ := 10
  let selected_people : ℕ := 4
  let task_A_people : ℕ := 2
  let task_B_people : ℕ := 1
  let task_C_people : ℕ := 1

  -- Define the function to calculate the number of ways
  let calculate_ways : ℕ := 
    Nat.choose total_people selected_people * 
    Nat.choose selected_people task_A_people * 
    Nat.factorial task_B_people

  -- Prove that the number of ways is equal to 2520
  have h : calculate_ways = 2520 := by sorry

  -- Return the result
  exact 2520

end NUMINAMATH_CALUDE_ERRORFEEDBACK_task_assignment_count_l245_24584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_addition_and_rounding_l245_24562

theorem addition_and_rounding :
  let sum := 47.32 + 28.659
  let rounded := (sum * 10).round / 10
  rounded = 76.0 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_addition_and_rounding_l245_24562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_current_speed_l245_24529

/-- Calculates the speed of a river's current given a paddler's speed on still water,
    the length of the river, and the time taken to paddle up the river. -/
theorem river_current_speed (still_water_speed : ℝ) (river_length : ℝ) (time_taken : ℝ) 
    (h1 : still_water_speed > 0)
    (h2 : river_length > 0)
    (h3 : time_taken > 0) :
  let effective_speed := river_length / time_taken
  effective_speed > 0 ∧ effective_speed < still_water_speed →
  still_water_speed - effective_speed = 
    (still_water_speed - (river_length / time_taken)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_current_speed_l245_24529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_difference_l245_24520

def geometric_sequence (a₁ : ℤ) (r : ℤ) (n : ℕ) : ℤ := a₁ * r^(n - 1)

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sequence_A (n : ℕ) : ℤ := geometric_sequence 3 3 n

def sequence_B (n : ℕ) : ℤ := arithmetic_sequence 15 15 n

def last_term_A : ℤ := 243
def last_term_B : ℤ := 390

theorem least_positive_difference :
  (∃ (m n : ℕ), 
    sequence_A m ≤ last_term_A ∧ 
    sequence_B n ≤ last_term_B ∧ 
    |sequence_A m - sequence_B n| = 3) ∧ 
  (∀ (m n : ℕ), 
    sequence_A m ≤ last_term_A → 
    sequence_B n ≤ last_term_B → 
    |sequence_A m - sequence_B n| ≥ 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_difference_l245_24520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l245_24527

theorem power_equation (n b : ℝ) : n = 2^(1/10) → n^b = 16 → b = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l245_24527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_property_l245_24585

/-- A positive geometric sequence with first term a₁ and common ratio q -/
noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

/-- Theorem: For a positive geometric sequence satisfying certain conditions,
    the sum of the nth term and the sum of the first n terms is always 4 -/
theorem geometric_sequence_sum_property
  (a₁ : ℝ) (q : ℝ) (n : ℕ) 
  (h_pos : a₁ > 0 ∧ q > 0)
  (h_cond1 : a₁ * q + 2 * a₁ * q^2 = a₁)
  (h_cond2 : a₁ * (a₁ * q^3) = a₁ * q^5) :
  geometric_sequence a₁ q n + geometric_sum a₁ q n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_property_l245_24585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l245_24574

/-- Calculates the future value of an investment with compound interest -/
noncomputable def future_value (principal : ℝ) (rate : ℝ) (compounds_per_year : ℕ) (years : ℕ) : ℝ :=
  principal * (1 + rate / (compounds_per_year : ℝ)) ^ ((compounds_per_year : ℝ) * (years : ℝ))

/-- Proves that an investment of $45,045 grows to approximately $100,000 in 10 years
    with 8% annual interest compounded monthly -/
theorem investment_growth :
  let principal : ℝ := 45045
  let rate : ℝ := 0.08
  let compounds_per_year : ℕ := 12
  let years : ℕ := 10
  let target : ℝ := 100000
  abs (future_value principal rate compounds_per_year years - target) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l245_24574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_change_percentage_closest_to_five_l245_24577

noncomputable def item_prices : List ℝ := [7.99, 4.99, 2.99, 1.99, 0.99]
def paid_amount : ℝ := 20.00

noncomputable def total_price : ℝ := item_prices.sum
noncomputable def change : ℝ := paid_amount - total_price
noncomputable def change_percentage : ℝ := (change / paid_amount) * 100

def options : List ℝ := [5, 10, 15, 20, 25]

-- Define a function to find the closest value in the list
noncomputable def find_closest (target : ℝ) (list : List ℝ) : ℝ :=
  list.foldl (fun acc x => if |x - target| < |acc - target| then x else acc) (list.head!)

theorem change_percentage_closest_to_five :
  find_closest change_percentage options = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_change_percentage_closest_to_five_l245_24577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_is_half_l245_24570

/-- Represents the race track and participants' performances -/
structure RaceData where
  track_length : ℝ
  polly_laps : ℕ
  polly_time : ℝ
  gerald_speed : ℝ

/-- Calculates the ratio of Gerald's speed to Polly's speed -/
noncomputable def speed_ratio (data : RaceData) : ℝ :=
  let polly_distance := data.track_length * data.polly_laps
  let polly_speed := polly_distance / data.polly_time
  data.gerald_speed / polly_speed

/-- Theorem stating that the speed ratio is 1/2 for the given conditions -/
theorem speed_ratio_is_half :
  let data : RaceData := {
    track_length := 1/4,
    polly_laps := 12,
    polly_time := 1/2,
    gerald_speed := 3
  }
  speed_ratio data = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_is_half_l245_24570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_internal_point_distances_l245_24526

/-- Given a circle with center O and a point P inside the circle,
    if the distance from P to the nearest point on the circle is 1
    and the distance to the farthest point is 7,
    then the radius of the circle is 4. -/
theorem circle_radius_from_internal_point_distances
  {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  (O P : E) (r : ℝ) :
  (∃ (N F : E), ‖N - O‖ = r ∧ ‖F - O‖ = r ∧
    ‖P - N‖ = 1 ∧
    ‖P - F‖ = 7 ∧
    (∀ X : E, ‖X - O‖ = r → ‖P - X‖ ≥ 1) ∧
    (∀ Y : E, ‖Y - O‖ = r → ‖P - Y‖ ≤ 7)) →
  r = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_internal_point_distances_l245_24526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_second_derivative_at_one_is_zero_l245_24594

-- Define the function f
noncomputable def f (f''1 : ℝ) : ℝ → ℝ := λ x ↦ (1/3) * x^3 - f''1 * x^2 - x

-- State the theorem
theorem f_second_derivative_at_one_is_zero :
  ∃ f''1 : ℝ, (∀ x, (deriv (deriv (f f''1))) x = f''1) ∧ f''1 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_second_derivative_at_one_is_zero_l245_24594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l245_24542

def P : Set ℤ := {-1, 1}
def Q : Set ℤ := {x : ℤ | 0 ≤ x ∧ x < 3}

theorem union_of_P_and_Q : P ∪ Q = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l245_24542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_employees_needed_l245_24557

/-- Represents the set of employees monitoring water pollution -/
def W : Finset Nat := sorry

/-- Represents the set of employees monitoring air pollution -/
def A : Finset Nat := sorry

/-- Represents the set of employees monitoring soil pollution -/
def S : Finset Nat := sorry

theorem minimum_employees_needed :
  (Finset.card W = 90) →
  (Finset.card A = 80) →
  (Finset.card S = 50) →
  (Finset.card (W ∩ A) = 30) →
  (Finset.card (A ∩ S) = 20) →
  (Finset.card (W ∩ S) = 15) →
  (Finset.card (W ∩ A ∩ S) = 10) →
  Finset.card (W ∪ A ∪ S) = 165 := by
  sorry

#check minimum_employees_needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_employees_needed_l245_24557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_day_of_week_theorem_l245_24561

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure DayInYear where
  year : Int
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- Given conditions of the problem -/
def condition1 (n : Int) : DayInYear :=
  { year := n, dayNumber := 250, dayOfWeek := DayOfWeek.Friday }

def condition2 (n : Int) : DayInYear :=
  { year := n + 1, dayNumber := 160, dayOfWeek := DayOfWeek.Friday }

/-- The day we want to prove about -/
def targetDay (n : Int) : DayInYear :=
  { year := n - 1, dayNumber := 110, dayOfWeek := DayOfWeek.Thursday }

/-- Function to determine the day of the week given the conditions -/
def determineDayOfWeek (n : Int) : Prop :=
  (targetDay n).dayOfWeek = DayOfWeek.Thursday

/-- Theorem statement -/
theorem day_of_week_theorem (n : Int) :
  determineDayOfWeek n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_day_of_week_theorem_l245_24561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_third_l245_24533

theorem sin_alpha_plus_pi_third (α : ℝ) 
  (h1 : Real.cos (α - π/6) + Real.sin α = 4 * Real.sqrt 3 / 5)
  (h2 : α ∈ Set.Ioo (π/2) π) :
  Real.sin (α + π/3) = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_third_l245_24533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_rate_proof_l245_24589

/-- Proof of investment rate for remaining amount --/
theorem investment_rate_proof 
  (total_investment : ℝ) 
  (first_investment : ℝ) 
  (second_investment : ℝ) 
  (first_rate : ℝ) 
  (second_rate : ℝ) 
  (desired_income : ℝ) : 
  total_investment = 12000 ∧ 
  first_investment = 5000 ∧ 
  second_investment = 4000 ∧ 
  first_rate = 0.03 ∧ 
  second_rate = 0.045 ∧ 
  desired_income = 600 → 
  let remaining_investment := total_investment - first_investment - second_investment
  let remaining_income := desired_income - (first_investment * first_rate) - (second_investment * second_rate)
  (remaining_income / remaining_investment) = 0.09 := by
  intro h
  -- The proof steps would go here
  sorry

#check investment_rate_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_rate_proof_l245_24589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_graph_shift_l245_24578

-- Define the original and target functions
noncomputable def original_func (x : ℝ) : ℝ := 3 * Real.cos (2 * x)
noncomputable def target_func (x : ℝ) : ℝ := 3 * Real.cos (2 * x + Real.pi/4)

-- Define the shift amount
noncomputable def shift_amount : ℝ := Real.pi/8

-- Theorem statement
theorem cosine_graph_shift :
  ∀ x : ℝ, target_func x = original_func (x + shift_amount) :=
by
  intro x
  simp [target_func, original_func, shift_amount]
  -- The actual proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_graph_shift_l245_24578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_geometric_sequence_l245_24597

noncomputable def geometric_sequence (n : ℕ) : ℝ := 3 * (-1/2)^(n-1)

theorem sum_of_geometric_sequence :
  (∑' n, geometric_sequence n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_geometric_sequence_l245_24597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l245_24507

def solution_set : Set (ℂ × ℂ × ℂ) :=
  {(0, 0, 0), (2, 2, 2), 
   (-1 + Complex.I, -Complex.I, -Complex.I), 
   (-Complex.I, -1 + Complex.I, -Complex.I), 
   (-Complex.I, -Complex.I, -1 + Complex.I),
   (-1 - Complex.I, Complex.I, Complex.I), 
   (Complex.I, -1 - Complex.I, Complex.I), 
   (Complex.I, Complex.I, -1 - Complex.I)}

theorem system_solutions :
  ∀ (x y z : ℂ), (x^2 = y + z ∧ y^2 = x + z ∧ z^2 = x + y) ↔ (x, y, z) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l245_24507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_5_range_of_m_min_value_of_f_l245_24531

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Part 1: Solution set of f(x) ≤ 5
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = Set.Icc (-7/2) (3/2) := by sorry

-- Part 2: Range of m satisfying m^2 - 3m < f(x) for all x
theorem range_of_m (m : ℝ) :
  (∀ x, m^2 - 3*m < f x) ↔ m ∈ Set.Ioo (-1) 4 := by sorry

-- Additional theorem to show the minimum value of f(x)
theorem min_value_of_f :
  ∀ x, f x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_5_range_of_m_min_value_of_f_l245_24531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_on_x_axis_implies_sum_y_zero_l245_24505

/-- A triangle in a 2D plane -/
structure Triangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  (((t.v1.1 + t.v2.1 + t.v3.1) / 3), ((t.v1.2 + t.v2.2 + t.v3.2) / 3))

/-- Theorem: If the centroid of a triangle lies on the x-axis, then the sum of y-coordinates of its vertices is zero -/
theorem centroid_on_x_axis_implies_sum_y_zero (t : Triangle) :
  (centroid t).2 = 0 → t.v1.2 + t.v2.2 + t.v3.2 = 0 := by
  sorry

#check centroid_on_x_axis_implies_sum_y_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_on_x_axis_implies_sum_y_zero_l245_24505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_of_f_l245_24522

noncomputable def f (x : ℝ) : ℝ := (x^2 + 3*x + 9) / (x^2 - 5*x + 6)

theorem vertical_asymptotes_of_f :
  ∀ x : ℝ, (x = 3 ∨ x = 2) ↔ 
    (∃ ε > 0, ∀ δ > 0, ∃ y : ℝ, 
      |y - x| < δ ∧ |f y| > 1/ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_of_f_l245_24522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_chord_theorem_l245_24514

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the line structure
structure Line where
  point1 : Point
  point2 : Point

-- Define helper functions
def are_tangent_internally (c1 c2 : Circle) (p : Point) : Prop := sorry
def is_chord_of (c : Circle) (l : Line) : Prop := sorry
def is_tangent_to (l : Line) (c : Circle) (p : Point) : Prop := sorry
def lies_on (l : Line) (p : Point) : Prop := sorry
def intersects_at (l : Line) (c : Circle) (p : Point) : Prop := sorry
def distance (p1 p2 : Point) : ℝ := sorry

-- Define the theorem
theorem tangent_circles_chord_theorem 
  (circle1 circle2 : Circle) 
  (A B C D M : Point) 
  (BC AD : Line) 
  (a b : ℝ) 
  (h1 : are_tangent_internally circle1 circle2 A)
  (h2 : is_chord_of circle1 BC)
  (h3 : is_tangent_to BC circle2 D)
  (h4 : lies_on AD A)
  (h5 : lies_on AD D)
  (h6 : intersects_at AD circle1 M)
  (h7 : distance A M = a)
  (h8 : distance M D = b) :
  distance M B = Real.sqrt (a * b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_chord_theorem_l245_24514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_l245_24535

theorem log_equality (x : ℝ) (h : Real.log (x + 10) / Real.log 7 = 2) : 
  Real.log x / Real.log 13 = Real.log 39 / Real.log 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_l245_24535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cola_price_l245_24567

/-- The cost of a hot dog in cents -/
def hot_dog_cost (n : ℕ) : Prop := true

/-- The cost of a cola in cents -/
def cola_cost (n : ℕ) : Prop := true

/-- Three hot dogs and two colas cost 360 cents -/
axiom purchase1 : ∃ h c, hot_dog_cost h ∧ cola_cost c ∧ 3 * h + 2 * c = 360

/-- Two hot dogs and three colas cost 390 cents -/
axiom purchase2 : ∃ h c, hot_dog_cost h ∧ cola_cost c ∧ 2 * h + 3 * c = 390

/-- The cost of a cola is 90 cents -/
theorem cola_price : cola_cost 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cola_price_l245_24567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_inventory_calculation_l245_24541

/-- Calculates the final number of bottles in storage for each size --/
def final_bottle_count (initial : ℕ) (sold_percent : ℚ) (new_production expired shipped : ℕ) : ℕ :=
  initial - (↑initial * sold_percent).floor.toNat + new_production - expired - shipped

theorem bottle_inventory_calculation :
  let small_initial : ℕ := 5000
  let medium_initial : ℕ := 12000
  let big_initial : ℕ := 8000
  let xl_initial : ℕ := 2500

  let small_sold_percent : ℚ := 15 / 100
  let medium_sold_percent : ℚ := 18 / 100
  let big_sold_percent : ℚ := 23 / 100
  let xl_sold_percent : ℚ := 10 / 100

  let small_new : ℕ := 550
  let medium_new : ℕ := 425
  let big_new : ℕ := 210
  let xl_new : ℕ := 0

  let small_expired : ℕ := 200
  let medium_expired : ℕ := 350
  let big_expired : ℕ := 0
  let xl_expired : ℕ := 150

  let small_shipped : ℕ := 525
  let medium_shipped : ℕ := 320
  let big_shipped : ℕ := 255
  let xl_shipped : ℕ := 150

  final_bottle_count small_initial small_sold_percent small_new small_expired small_shipped = 4075 ∧
  final_bottle_count medium_initial medium_sold_percent medium_new medium_expired medium_shipped = 9595 ∧
  final_bottle_count big_initial big_sold_percent big_new big_expired big_shipped = 6115 ∧
  final_bottle_count xl_initial xl_sold_percent xl_new xl_expired xl_shipped = 1950 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_inventory_calculation_l245_24541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_bouquet_theorem_l245_24548

/-- Represents the number of flowers in a bouquet -/
structure Bouquet :=
  (roses tulips daisies lilies sunflowers : ℕ)

/-- Calculates the remaining flowers after giving away a fraction -/
def giveAway (b : Bouquet) (fraction : ℚ) : Bouquet :=
  { roses := (b.roses - Int.floor (fraction * b.roses : ℚ)).natAbs,
    tulips := (b.tulips - Int.floor (fraction * b.tulips : ℚ)).natAbs,
    daisies := (b.daisies - Int.floor (fraction * b.daisies : ℚ)).natAbs,
    lilies := (b.lilies - Int.floor (fraction * b.lilies : ℚ)).natAbs,
    sunflowers := (b.sunflowers - Int.floor (fraction * b.sunflowers : ℚ)).natAbs }

/-- Adds new flowers to the bouquet -/
def addFlowers (b : Bouquet) (newRoses newDaisies newSunflowers : ℕ) : Bouquet :=
  { roses := b.roses + newRoses,
    tulips := b.tulips,
    daisies := b.daisies + newDaisies,
    lilies := b.lilies,
    sunflowers := b.sunflowers + newSunflowers }

/-- Removes wilted flowers from the bouquet -/
def removeWilted (b : Bouquet) (rosesFraction tulipsFraction daisiesFraction liliesFraction sunflowersFraction : ℚ) : Bouquet :=
  { roses := (b.roses - Int.floor (rosesFraction * b.roses : ℚ)).natAbs,
    tulips := (b.tulips - Int.floor (tulipsFraction * b.tulips : ℚ)).natAbs,
    daisies := (b.daisies - Int.floor (daisiesFraction * b.daisies : ℚ)).natAbs,
    lilies := (b.lilies - Int.floor (liliesFraction * b.lilies : ℚ)).natAbs,
    sunflowers := (b.sunflowers - Int.floor (sunflowersFraction * b.sunflowers : ℚ)).natAbs }

theorem susan_bouquet_theorem (initialBouquet : Bouquet) :
  initialBouquet.roses = 36 →
  initialBouquet.tulips = 24 →
  initialBouquet.daisies = 48 →
  initialBouquet.lilies = 60 →
  initialBouquet.sunflowers = 36 →
  let remainingAfterGiving := giveAway initialBouquet (3/4)
  let addedFlowers := addFlowers remainingAfterGiving 12 24 6
  let finalBouquet := removeWilted addedFlowers (1/3) (1/4) (1/2) (2/5) (1/6)
  finalBouquet.roses = 14 ∧
  finalBouquet.tulips = 5 ∧
  finalBouquet.daisies = 18 ∧
  finalBouquet.lilies = 9 ∧
  finalBouquet.sunflowers = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_bouquet_theorem_l245_24548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l245_24599

/-- The line equation -/
def line (x y : ℝ) : Prop := 4 * x - 3 * y - 2 = 0

/-- The circle equation -/
def circle_eq (x y a : ℝ) : Prop := x^2 + y^2 - 2*a*x + 4*y + a^2 - 12 = 0

/-- The theorem stating the condition for two distinct intersections -/
theorem line_circle_intersection (a : ℝ) :
  (∀ x y : ℝ, line x y → circle_eq x y a → ∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ line x' y' ∧ circle_eq x' y' a) ↔
  -6 < a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l245_24599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_remaining_l245_24500

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint = 1 → 
  (initial_paint - (1/4 * initial_paint) - 
   (1/2 * (initial_paint - (1/4 * initial_paint))) - 
   (1/3 * (initial_paint - (1/4 * initial_paint) - 
           (1/2 * (initial_paint - (1/4 * initial_paint))))))
  = 1/4 * initial_paint := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_remaining_l245_24500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arccos_equality_l245_24549

theorem sin_arccos_equality (y : ℝ) (h1 : y > 0) (h2 : Real.sin (Real.arccos y) = y) : y^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arccos_equality_l245_24549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_grid_with_area_11_l245_24501

/-- A grid with dimensions m × n --/
structure Grid (m n : ℕ) where
  rows : Fin m → Fin n → Bool

/-- The area of a shaded region in a grid --/
def shadedArea (m n : ℕ) (g : Grid m n) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin m)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin n)) fun j =>
      if g.rows i j then 1 else 0)

/-- There exists a 5 × 4 grid with a shaded region of area 11 --/
theorem exists_grid_with_area_11 : ∃ (g : Grid 5 4), shadedArea 5 4 g = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_grid_with_area_11_l245_24501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_car_charging_cost_l245_24518

/-- The average charging cost per kilometer for the electric car -/
noncomputable def x : ℝ := 0.2

/-- The average refueling cost per kilometer for the fuel car -/
noncomputable def fuel_cost : ℝ := x + 0.6

/-- The total charging cost for the electric car -/
def total_charge_cost : ℝ := 200

/-- The total refueling cost for the fuel car -/
def total_fuel_cost : ℝ := 200

/-- The distance the electric car can travel -/
noncomputable def electric_distance : ℝ := total_charge_cost / x

/-- The distance the fuel car can travel -/
noncomputable def fuel_distance : ℝ := total_fuel_cost / fuel_cost

theorem electric_car_charging_cost :
  (electric_distance = 4 * fuel_distance) → x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_car_charging_cost_l245_24518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_line_l245_24506

/-- A parabola is tangent to a line if and only if their equations have exactly one solution -/
def is_tangent (a b : ℝ) : Prop :=
  ∃! x, a * x^2 + b * x + 2 = 2 * x + 3

/-- The value of 'a' for which the parabola y = ax^2 + bx + 2 is tangent to the line y = 2x + 3 -/
noncomputable def tangent_a (b : ℝ) : ℝ := -(b - 2)^2 / 4

/-- Theorem stating the condition for tangency between the parabola and the line -/
theorem parabola_tangent_line (b : ℝ) :
  is_tangent (tangent_a b) b ∧ ∀ a : ℝ, is_tangent a b → a = tangent_a b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_line_l245_24506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l245_24516

noncomputable def f (x : Real) : Real := x + 3 * Real.sin (Real.pi / 6 - 2 * x)

theorem max_value_of_f :
  let a := Real.pi / 3
  let b := 5 * Real.pi / 6
  ∃ (c : Real), c ∈ Set.Icc a b ∧ 
    (∀ (x : Real), x ∈ Set.Icc a b → f x ≤ f c) ∧
    f c = 3 + 5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l245_24516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l245_24555

theorem tan_phi_value (φ : ℝ) 
  (h1 : Real.cos (π / 2 + φ) = 2 / 3)
  (h2 : |φ| < π / 2) :
  Real.tan φ = -2 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l245_24555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_10_diamonds_l245_24517

/-- Triangular number -/
def T (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Number of diamonds in figure F_n -/
def F : ℕ → ℕ
  | 0 => 1  -- Add this case for n = 0
  | 1 => 1
  | 2 => 5
  | n + 3 => F (n + 2) + 4 * T (n + 2)

theorem F_10_diamonds : F 10 = 365 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_10_diamonds_l245_24517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lcm_with_18_l245_24586

def S : Finset Nat := {3, 9, 12, 16, 21, 18}

theorem largest_lcm_with_18 : Finset.sup S (fun n => Nat.lcm 18 n) = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lcm_with_18_l245_24586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flu_epidemic_transmission_rate_l245_24547

/-- Represents the average number of people infected by one person in each round of transmission -/
def average_infection_rate : ℝ → Prop := λ _ => True

/-- The total number of infected people after two rounds of transmission -/
def total_infected (x : ℝ) : ℝ := 1 + x + x * (1 + x)

theorem flu_epidemic_transmission_rate :
  ∃ x : ℝ, average_infection_rate x ∧ total_infected x = 100 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flu_epidemic_transmission_rate_l245_24547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_around_cone_l245_24591

theorem sphere_radius_around_cone (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (Real.sqrt (r^2 + h^2)) / 2 = (Real.sqrt (r^2 + h^2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_around_cone_l245_24591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l245_24569

/-- Parabola structure -/
structure MyParabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := λ x y => x^2 = 2 * p * y

/-- Line structure -/
structure MyLine where
  k : ℝ
  equation : ℝ → ℝ → Prop := λ x y => y = k * x + 1

/-- Point structure -/
structure MyPoint where
  x : ℝ
  y : ℝ

/-- Vector structure -/
structure MyVector where
  x : ℝ
  y : ℝ

/-- Main theorem -/
theorem parabola_line_intersection
  (C : MyParabola)
  (l : MyLine)
  (F A B E : MyPoint)
  (h1 : l.k > 0)
  (h2 : C.equation F.x F.y)  -- F is on the parabola
  (h3 : l.equation F.x F.y)  -- F is on the line
  (h4 : C.equation A.x A.y)  -- A is on the parabola
  (h5 : l.equation A.x A.y)  -- A is on the line
  (h6 : C.equation B.x B.y)  -- B is on the parabola
  (h7 : l.equation B.x B.y)  -- B is on the line
  (h8 : l.equation E.x E.y)  -- E is on the line
  (h9 : E.y = -C.p)  -- E is on the directrix
  (h10 : MyVector.mk (E.x - F.x) (E.y - F.y) = MyVector.mk (F.x - B.x) (F.y - B.y))  -- EF = FB
  : C.p = 2 ∧ l.k = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l245_24569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_sin_C_l245_24583

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def SpecialTriangle (t : Triangle) : Prop :=
  t.a = Real.sqrt 2 ∧
  t.b = Real.sqrt 3 ∧
  t.A + t.B + t.C = Real.pi ∧
  2 * t.B = t.A + t.C ∧
  t.a / Real.sin t.A = t.b / Real.sin t.B

-- State the theorem
theorem special_triangle_sin_C (t : Triangle) (h : SpecialTriangle t) :
  Real.sin t.C = (Real.sqrt 2 + Real.sqrt 6) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_sin_C_l245_24583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_percentage_l245_24540

noncomputable def original_price : ℝ := 345
noncomputable def first_discount : ℝ := 0.12
noncomputable def final_price : ℝ := 227.70

noncomputable def price_after_first_discount : ℝ := original_price * (1 - first_discount)

noncomputable def second_discount : ℝ := 1 - (final_price / price_after_first_discount)

theorem second_discount_percentage :
  second_discount = 0.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_percentage_l245_24540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l245_24515

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3*x - 8)*(x - 4)/(x - 1)

-- Define the solution set
def solution_set : Set ℝ := Set.Iic 1 ∪ Set.Ici 4

-- Theorem statement
theorem inequality_solution :
  {x : ℝ | g x ≥ 0 ∧ x ≠ 1} = solution_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l245_24515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_1020_1022_minus_1021_squared_l245_24566

open Matrix

/-- The Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The Fibonacci matrix -/
def fibMatrix : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; 1, 0]

/-- The matrix identity for Fibonacci numbers -/
axiom fib_matrix_power (n : ℕ) : 
  fibMatrix ^ n = !![fib (n + 1), fib n; fib n, fib (n - 1)]

theorem fib_1020_1022_minus_1021_squared : 
  fib 1020 * fib 1022 - fib 1021 * fib 1021 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_1020_1022_minus_1021_squared_l245_24566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_theorem_l245_24571

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def parallel_lines (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : Prop :=
  (y₂ - y₁) * (x₄ - x₃) = (y₄ - y₃) * (x₂ - x₁)

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem ellipse_ratio_theorem (a b : ℝ) (hpos : 0 < a ∧ 0 < b) (hgt : a > b)
  (P Q : ℝ × ℝ) (hP : ellipse a b P.1 P.2) (hQ : ellipse a b Q.1 Q.2)
  (hpar : parallel_lines (-a) 0 Q.1 Q.2 0 0 P.1 P.2)
  (R : ℝ × ℝ) (hR : R.1 = 0 ∧ parallel_lines (-a) 0 Q.1 Q.2 0 R.2 Q.1 Q.2) :
  (distance (-a) 0 Q.1 Q.2 * distance (-a) 0 0 R.2) / (distance 0 0 P.1 P.2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_theorem_l245_24571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_touching_circles_l245_24508

theorem area_of_triangle_with_touching_circles (r : ℝ) (hr : r > 0) :
  let a := 2 * r * (Real.sqrt 3 + 1)
  let S := a^2 * Real.sqrt 3 / 4
  S = 2 * r^2 * (2 * Real.sqrt 3 + 3) := by
  sorry

#check area_of_triangle_with_touching_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_touching_circles_l245_24508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_more_cost_effective_l245_24572

/-- Represents the cost-effectiveness of grain purchasing methods -/
inductive PurchaseMethod
  | A
  | B

/-- Determines which purchase method is more cost-effective -/
noncomputable def more_cost_effective (x y : ℝ) : PurchaseMethod :=
  if (x + y) / 2 > 2 * x * y / (x + y) then PurchaseMethod.B else PurchaseMethod.A

theorem b_more_cost_effective (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  more_cost_effective x y = PurchaseMethod.B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_more_cost_effective_l245_24572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_equilateral_triangle_with_cutoffs_l245_24582

/-- The ratio of areas in an equilateral triangle with cut-off corners -/
theorem area_ratio_equilateral_triangle_with_cutoffs :
  let large_side : ℝ := 10
  let small_side : ℝ := 3
  let large_area := (Real.sqrt 3 / 4) * large_side^2
  let small_area := (Real.sqrt 3 / 4) * small_side^2
  let central_area := large_area - 3 * small_area
  small_area / central_area = 9 / 73 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_equilateral_triangle_with_cutoffs_l245_24582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_perimeter_l245_24543

/-- Represents a triangle with given side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Creates a new triangle by connecting the midpoints of the given triangle -/
noncomputable def midpointTriangle (t : Triangle) : Triangle :=
  { a := t.a / 2
  , b := t.b / 2
  , c := t.c / 2 }

/-- Calculates the perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- The main theorem to prove -/
theorem midpoint_triangle_perimeter :
  let t₁ : Triangle := { a := 101, b := 102, c := 103 }
  let t₂ : Triangle := midpointTriangle t₁
  let t₃ : Triangle := midpointTriangle t₂
  perimeter t₃ = 76.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_perimeter_l245_24543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_sum_of_squares_l245_24528

theorem cubic_roots_sum_of_squares (a b c : ℝ) (s : ℝ) : 
  (∀ x : ℝ, x^3 - 9*x^2 + 11*x - 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  s = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  s^4 - 18*s^2 - 8*s = -37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_sum_of_squares_l245_24528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_change_l245_24559

noncomputable def initial_investment : ℝ := 200

noncomputable def first_year_rate : ℝ := 0.9  -- 10% decrease
noncomputable def second_year_rate : ℝ := 1.3  -- 30% gain
noncomputable def third_year_rate : ℝ := 0.85  -- 15% loss

noncomputable def final_investment : ℝ := initial_investment * first_year_rate * second_year_rate * third_year_rate

noncomputable def percentage_change : ℝ := (final_investment - initial_investment) / initial_investment * 100

theorem investment_change :
  ∃ ε > 0, abs (percentage_change + 0.55) < ε ∧ ε < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_change_l245_24559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pqrs_ratio_is_eight_ninths_l245_24544

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℚ
  height : ℚ

/-- The large rectangle PQRS formed by smaller rectangles -/
structure PQRS where
  smallRectangle : Rectangle
  horizontalCount : ℕ
  verticalCount : ℕ

/-- The ratio of width to height of PQRS -/
def widthHeightRatio (pqrs : PQRS) : ℚ :=
  let totalWidth := pqrs.horizontalCount * pqrs.smallRectangle.width
  let totalHeight := pqrs.verticalCount * pqrs.smallRectangle.height
  totalWidth / totalHeight

/-- Theorem stating that the ratio of PQ to QR is 8:9 for the given configuration -/
theorem pqrs_ratio_is_eight_ninths (pqrs : PQRS) 
    (h1 : pqrs.horizontalCount = 3)
    (h2 : pqrs.verticalCount = 4)
    (h3 : pqrs.smallRectangle.height * 3 = pqrs.smallRectangle.width * 2) :
    widthHeightRatio pqrs = 8 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pqrs_ratio_is_eight_ninths_l245_24544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_principal_proof_l245_24596

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time / 100

theorem loan_principal_proof (rate time interest : ℝ) (h1 : rate = 12) (h2 : time = 3) (h3 : interest = 9000) :
  ∃ principal : ℝ, simple_interest principal rate time = interest ∧ principal = 25000 := by
  -- We'll use 25000 as our witness for the existential quantifier
  use 25000
  -- Split the goal into two parts
  constructor
  -- Prove that simple_interest 25000 rate time = interest
  · simp [simple_interest, h1, h2, h3]
    norm_num
  -- Prove that 25000 = 25000 (trivial)
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_principal_proof_l245_24596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_at_unit_ratio_l245_24593

-- Define the circles and points
structure CircleConfig where
  k : ℝ
  h_k_range : 1 < k ∧ k < 3

-- Define the line segment XY
structure LineSegment (config : CircleConfig) where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  B : ℝ × ℝ
  h_X_on_C2 : (X.1 - 1/2)^2 + X.2^2 = (config.k/2)^2
  h_Y_on_C3 : Y.1^2 + Y.2^2 = config.k^2
  h_B_on_line : ∃ t : ℝ, B = (1 - t) • X + t • Y

-- Define the length of XY
noncomputable def length_XY (config : CircleConfig) (seg : LineSegment config) : ℝ :=
  Real.sqrt ((seg.X.1 - seg.Y.1)^2 + (seg.X.2 - seg.Y.2)^2)

-- Define the ratio XB/BY
noncomputable def ratio_XB_BY (config : CircleConfig) (seg : LineSegment config) : ℝ :=
  Real.sqrt ((seg.X.1 - seg.B.1)^2 + (seg.X.2 - seg.B.2)^2) /
  Real.sqrt ((seg.B.1 - seg.Y.1)^2 + (seg.B.2 - seg.Y.2)^2)

-- State the theorem
theorem min_length_at_unit_ratio (config : CircleConfig) :
  ∀ seg : LineSegment config,
    length_XY config seg ≥ Real.sqrt (2 * (config.k^2 - 1)) ∧
    (length_XY config seg = Real.sqrt (2 * (config.k^2 - 1)) ↔ ratio_XB_BY config seg = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_at_unit_ratio_l245_24593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gravitational_force_at_500_miles_l245_24595

/-- Represents the gravitational force equation -/
noncomputable def gravitational_force (k : ℝ) (d : ℝ) : ℝ := k / (d^2)

/-- Theorem stating the gravitational force at 500 miles -/
theorem gravitational_force_at_500_miles 
  (initial_force : ℝ) 
  (initial_distance : ℝ) 
  (new_distance : ℝ) 
  (h1 : initial_force = 200)
  (h2 : initial_distance = 100)
  (h3 : new_distance = 500)
  (h4 : gravitational_force (2 * initial_force * initial_distance^2) new_distance = 16) :
  gravitational_force (2 * initial_force * initial_distance^2) new_distance = 16 :=
by
  sorry

#check gravitational_force_at_500_miles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gravitational_force_at_500_miles_l245_24595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_l245_24503

/-- The area of a regular octagon inscribed in a circle with radius 5 units -/
noncomputable def octagonArea : ℝ := 100 * Real.sqrt (2 - Real.sqrt 2)

/-- The radius of the circle -/
def circleRadius : ℝ := 5

theorem regular_octagon_area :
  let r := circleRadius
  let s := 2 * r * Real.sin (π / 8)
  let triangleArea := (1 / 2) * s * r
  8 * triangleArea = octagonArea := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_l245_24503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_B_completion_time_l245_24511

/-- The number of days it takes for two workers to complete a job together -/
def days_together : ℝ := 14

/-- The rate at which worker B completes the job -/
noncomputable def rate_B : ℝ := 1 / (days_together * (3/2))

/-- The rate at which worker A completes the job -/
noncomputable def rate_A : ℝ := rate_B / 2

/-- The number of days it takes for worker B to complete the job alone -/
noncomputable def days_B_alone : ℝ := 1 / rate_B

theorem worker_B_completion_time :
  days_B_alone = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_B_completion_time_l245_24511
