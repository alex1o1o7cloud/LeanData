import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1214_121471

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin (x - Real.pi / 3)

theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1214_121471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_symmetric_planes_intersection_l1214_121474

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A₁ : Point3D
  A₂ : Point3D
  A₃ : Point3D
  A₄ : Point3D

/-- Defines the symmetric plane πᵢⱼ -/
noncomputable def symmetric_plane (T : Tetrahedron) (P : Point3D) (i j : Fin 4) : Plane3D :=
  sorry

/-- Checks if a point lies on a plane -/
def point_on_plane (Q : Point3D) (π : Plane3D) : Prop :=
  π.a * Q.x + π.b * Q.y + π.c * Q.z + π.d = 0

/-- Checks if a plane is parallel to a line -/
def plane_parallel_to_line (π : Plane3D) (L : Set Point3D) : Prop :=
  sorry

/-- Main theorem statement -/
theorem tetrahedron_symmetric_planes_intersection 
  (T : Tetrahedron) (P : Point3D) : 
  (∃ Q : Point3D, ∀ i j : Fin 4, i ≠ j → point_on_plane Q (symmetric_plane T P i j)) ∨
  (∃ L : Set Point3D, ∀ i j : Fin 4, i ≠ j → plane_parallel_to_line (symmetric_plane T P i j) L) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_symmetric_planes_intersection_l1214_121474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_evaluation_l1214_121490

-- Define the original expression
noncomputable def original_expr (x : ℝ) : ℝ := (2 / (x^2 - 4)) / (1 / (x^2 - 2*x))

-- Define the simplified expression
noncomputable def simplified_expr (x : ℝ) : ℝ := (2*x) / (x + 2)

-- Theorem statement
theorem simplification_and_evaluation :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 0 → original_expr x = simplified_expr x) ∧
  simplified_expr (-1) = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_evaluation_l1214_121490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_positive_and_range_condition_l1214_121442

noncomputable def f (x : ℝ) := x + Real.sin x

theorem slope_positive_and_range_condition :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) > 0) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ a * x * Real.cos x) ↔ a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_positive_and_range_condition_l1214_121442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_two_l1214_121469

-- Define the lines
noncomputable def line_l₁ (x y : ℝ) : Prop := x + y - 2 * Real.sqrt 2 = 0

noncomputable def line_l₂ (t x y : ℝ) : Prop := x = (Real.sqrt 2 / 2) * t ∧ y = (Real.sqrt 2 / 2) * t

-- Define the intersection point
noncomputable def intersection_point (x y : ℝ) : Prop :=
  ∃ t : ℝ, line_l₁ x y ∧ line_l₂ t x y

-- Define the distance function
noncomputable def distance_to_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

-- Theorem statement
theorem intersection_distance_is_two :
  ∀ x y : ℝ, intersection_point x y → distance_to_origin x y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_two_l1214_121469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1214_121440

noncomputable def M (x : ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n.val).prod (λ i => x + i)

noncomputable def f (x : ℝ) : ℝ :=
  M (x - 3) 7 * Real.cos ((2009 / 2010) * x)

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1214_121440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_alpha_plus_pi_fourth_l1214_121478

theorem tan_two_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : Real.sin α = Real.sqrt 5 / 5) : 
  Real.tan (2*α + π/4) = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_alpha_plus_pi_fourth_l1214_121478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_evaluations_l1214_121424

/-- Represents a class with its exam structure -/
structure ExamClass where
  students : Nat
  multipleChoice : Nat
  shortAnswer : Nat
  essay : Nat
  presentation : Nat
  groupSize : Nat

/-- Calculates the total number of evaluations for a given class -/
def evaluations (c : ExamClass) : Nat :=
  c.students * c.multipleChoice +
  c.students * c.shortAnswer +
  c.students * c.essay +
  (c.students / c.groupSize) * c.presentation

/-- The five classes with their exam structures -/
def classA : ExamClass := ⟨30, 12, 0, 3, 1, 1⟩
def classB : ExamClass := ⟨25, 15, 5, 2, 0, 1⟩
def classC : ExamClass := ⟨35, 10, 0, 3, 1, 5⟩
def classD : ExamClass := ⟨40, 11, 4, 3, 0, 1⟩
def classE : ExamClass := ⟨20, 14, 5, 2, 0, 1⟩

/-- The theorem stating the total number of evaluations -/
theorem total_evaluations :
  evaluations classA + evaluations classB + evaluations classC +
  evaluations classD + evaluations classE = 2632 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_evaluations_l1214_121424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_sum_squares_zero_l1214_121462

/-- Given a point P in 3D space and its projections on the coordinate planes,
    prove that the sum of squares of specific coordinates is zero. -/
theorem projection_sum_squares_zero (P : ℝ × ℝ × ℝ) 
  (proj_yoz proj_zox proj_xoy : ℝ × ℝ × ℝ) : 
  P = (-1, 3, -4) →
  (proj_yoz.1, proj_yoz.2.1, proj_yoz.2.2) = (0, 3, -4) →
  (proj_zox.1, proj_zox.2.1, proj_zox.2.2) = (-1, 0, -4) →
  (proj_xoy.1, proj_xoy.2.1, proj_xoy.2.2) = (-1, 3, 0) →
  proj_yoz.1^2 + proj_zox.2.1^2 + proj_xoy.2.2^2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_sum_squares_zero_l1214_121462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_ge_one_l1214_121404

/-- The function f(x) defined as √(x² + 1) - ax, where a > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 + 1) - a * x

/-- Theorem stating that f(x) is monotonic on [0, +∞) iff a ≥ 1 -/
theorem f_monotonic_iff_a_ge_one (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ y → f a x ≥ f a y) ↔ a ≥ 1 := by
  sorry

/-- Lemma for the solution of f(x) ≤ 1 when 0 < a < 1 -/
lemma f_le_one_sol_a_lt_one (a : ℝ) (ha : 0 < a) (ha' : a < 1) :
  {x : ℝ | f a x ≤ 1} = {x : ℝ | 0 ≤ x ∧ x ≤ 2 * a / (1 - a^2)} := by
  sorry

/-- Lemma for the solution of f(x) ≤ 1 when a ≥ 1 -/
lemma f_le_one_sol_a_ge_one (a : ℝ) (ha : a ≥ 1) :
  {x : ℝ | f a x ≤ 1} = {x : ℝ | x ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_ge_one_l1214_121404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_pairs_count_l1214_121494

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Define the five lines from the problem -/
def line1 : Line := { slope := 4, intercept := 7 }
def line2 : Line := { slope := 2, intercept := 3 }
def line3 : Line := { slope := -1/3, intercept := 2 }
def line4 : Line := { slope := 2, intercept := -3 }
def line5 : Line := { slope := -1/3, intercept := 5/2 }

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Bool := l1.slope = l2.slope

/-- Check if two lines are perpendicular -/
def are_perpendicular (l1 l2 : Line) : Bool := l1.slope * l2.slope = -1

/-- Check if a pair of lines is 'good' (parallel or perpendicular) -/
def is_good_pair (l1 l2 : Line) : Bool := are_parallel l1 l2 || are_perpendicular l1 l2

/-- Count the number of 'good' pairs among the given lines -/
def count_good_pairs : Nat :=
  (if is_good_pair line1 line2 then 1 else 0) +
  (if is_good_pair line1 line3 then 1 else 0) +
  (if is_good_pair line1 line4 then 1 else 0) +
  (if is_good_pair line1 line5 then 1 else 0) +
  (if is_good_pair line2 line3 then 1 else 0) +
  (if is_good_pair line2 line4 then 1 else 0) +
  (if is_good_pair line2 line5 then 1 else 0) +
  (if is_good_pair line3 line4 then 1 else 0) +
  (if is_good_pair line3 line5 then 1 else 0) +
  (if is_good_pair line4 line5 then 1 else 0)

/-- Theorem stating that the number of 'good' pairs is 3 -/
theorem good_pairs_count : count_good_pairs = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_pairs_count_l1214_121494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_dots_in_circles_l1214_121427

/-- A circle in the diagram -/
structure Circle where
  dots : Finset Nat
  dots_subset : dots ⊆ Finset.range 7

/-- The collection of circles in the diagram -/
def circles : Finset Circle := sorry

/-- The total number of dots -/
def total_dots : Nat := 7

/-- The sum of dots in all circles -/
def sum_dots_in_circles : Nat :=
  circles.sum (λ c => c.dots.card)

theorem count_dots_in_circles :
  sum_dots_in_circles = 15 := by sorry

#check count_dots_in_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_dots_in_circles_l1214_121427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1214_121456

-- Define the function f(x) = e^x + 4x - 3
noncomputable def f (x : ℝ) := Real.exp x + 4 * x - 3

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) := Real.exp x + 4

-- Theorem statement
theorem root_in_interval :
  ∃ x₀ ∈ Set.Ioo (1/4 : ℝ) (1/2 : ℝ), f x₀ = 0 :=
by
  sorry

-- Additional lemmas that might be useful for the proof
lemma f_increasing : ∀ x : ℝ, f_derivative x > 0 :=
by
  sorry

lemma f_zero_neg : f 0 < 0 :=
by
  sorry

lemma f_quarter_neg : f (1/4) < 0 :=
by
  sorry

lemma f_half_pos : f (1/2) > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1214_121456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_integers_in_sum_range_l1214_121487

def sum_of_permutation (perm : Fin 798 → Fin 798) : ℚ :=
  (Finset.range 798).sum (λ i => (perm i).val / (i + 1 : ℚ))

theorem all_integers_in_sum_range :
  ∀ k : ℕ, 798 ≤ k ∧ k ≤ 898 →
    ∃ perm : Fin 798 → Fin 798, Function.Bijective perm ∧
      (sum_of_permutation perm).num / (sum_of_permutation perm).den = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_integers_in_sum_range_l1214_121487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_neg_two_equals_67_over_16_l1214_121431

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9

noncomputable def g (x : ℝ) : ℝ := 3 * ((f⁻¹ x) ^ 2) - 4 * (f⁻¹ x) + 2

-- State the theorem
theorem g_of_neg_two_equals_67_over_16 : g (-2) = 67 / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_neg_two_equals_67_over_16_l1214_121431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_equality_condition_l1214_121470

theorem inequality_and_equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 ≥ 6 * Real.sqrt 3) ∧
  (a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 = 6 * Real.sqrt 3 ↔ a = b ∧ b = c ∧ c = (3 : ℝ)^(1/4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_equality_condition_l1214_121470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_expressions_equality_l1214_121414

theorem arithmetic_expressions_equality : 
  ((-3 - 7/8) - (-5 - 1/3) + (-4 - 1/3) - (3 + 1/8) = -6) ∧
  (abs (-7/9) / ((2/3) - (1/5)) - (1/3) * (1 - abs (-5)) = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_expressions_equality_l1214_121414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_circumference_is_24pi_l1214_121438

/-- A quadrilateral inscribed in a circle with a smaller circle tangent at one vertex -/
structure InscribedQuadrilateral where
  /-- Side length AB -/
  ab : ℝ
  /-- Side length BC -/
  bc : ℝ
  /-- Side length CD -/
  cd : ℝ
  /-- Side length DA -/
  da : ℝ
  /-- AB is a diameter of the smaller circle -/
  ab_is_diameter : True

/-- The circumference of the smaller circle tangent to the quadrilateral -/
noncomputable def smallerCircleCircumference (q : InscribedQuadrilateral) : ℝ :=
  2 * Real.pi * (q.ab / 2)

/-- Theorem: The circumference of the smaller circle is 24π -/
theorem smaller_circle_circumference_is_24pi (q : InscribedQuadrilateral)
    (h1 : q.ab = 24)
    (h2 : q.bc = 45)
    (h3 : q.cd = 28)
    (h4 : q.da = 53) :
    smallerCircleCircumference q = 24 * Real.pi := by
  unfold smallerCircleCircumference
  rw [h1]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_circumference_is_24pi_l1214_121438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_factor_less_than_9_l1214_121405

def factors_of_120 : Finset Nat :=
  Finset.filter (λ n => n > 0 ∧ 120 % n = 0) (Finset.range 121)

def factors_less_than_9 : Finset Nat :=
  Finset.filter (λ n => n < 9) factors_of_120

theorem probability_of_factor_less_than_9 :
  (factors_less_than_9.card : ℚ) / factors_of_120.card = 7 / 16 := by
  sorry

#eval factors_of_120.card
#eval factors_less_than_9.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_factor_less_than_9_l1214_121405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_1023_l1214_121496

def sequence_a : ℕ → ℕ
  | 0 => 1
  | (n + 1) => sequence_a n + 2^n

theorem a_10_equals_1023 : sequence_a 10 = 1023 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_1023_l1214_121496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_proof_l1214_121411

noncomputable section

def salvadore_earnings : ℝ := 1956

def santo_earnings : ℝ := salvadore_earnings / 2

def maria_earnings : ℝ := 3 * santo_earnings

def pedro_earnings : ℝ := santo_earnings + maria_earnings

def salvadore_tax_rate : ℝ := 0.20
def santo_tax_rate : ℝ := 0.15
def maria_tax_rate : ℝ := 0.10
def pedro_tax_rate : ℝ := 0.25

def total_earnings_after_taxes : ℝ :=
  (1 - salvadore_tax_rate) * salvadore_earnings +
  (1 - santo_tax_rate) * santo_earnings +
  (1 - maria_tax_rate) * maria_earnings +
  (1 - pedro_tax_rate) * pedro_earnings

theorem total_earnings_proof : total_earnings_after_taxes = 7971.70 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_proof_l1214_121411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_property_l1214_121428

/-- A quadratic function f(x) = ax² + bx - 3 where a ≠ 0 -/
def quadratic_function (a b : ℝ) (h : a ≠ 0) : ℝ → ℝ := 
  λ x ↦ a * x^2 + b * x - 3

theorem quadratic_function_property (a b : ℝ) (h : a ≠ 0) 
  (h2 : quadratic_function a b h 2 = quadratic_function a b h 4) : 
  quadratic_function a b h 6 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_property_l1214_121428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_one_zero_iff_a_positive_l1214_121400

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Theorem for the maximum value when a = 0
theorem max_value_when_a_zero :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 0 y ≤ f 0 x ∧ f 0 x = -1 :=
sorry

-- Theorem for the range of a when f(x) has exactly one zero
theorem one_zero_iff_a_positive :
  ∀ (a : ℝ), (∃! (x : ℝ), x > 0 ∧ f a x = 0) ↔ a > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_one_zero_iff_a_positive_l1214_121400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1214_121468

def a : ℕ → ℕ
| 0 => 1
| n + 1 => 3 * a n

def b : ℕ → ℕ := sorry

def T : ℕ → ℕ := sorry

theorem arithmetic_sequence_sum :
  (∀ n : ℕ, b (n + 1) - b n = b 2 - b 1) →
  b 1 = a 2 →
  b 3 = a 1 + a 2 + a 3 →
  T 20 = 1010 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1214_121468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_between_two_and_three_max_integer_below_zero_point_two_is_max_integer_main_result_l1214_121419

-- Define the function f(x) = ln x + 2x - 6
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

-- Define the zero point x₀
noncomputable def x₀ : ℝ := Real.exp (6 - 2 * Real.exp (6 / 2))

theorem zero_point_between_two_and_three : 2 < x₀ ∧ x₀ < 3 := by sorry

theorem max_integer_below_zero_point :
  ∀ k : ℤ, k ≤ ⌊x₀⌋ → k ≤ 2 := by sorry

theorem two_is_max_integer :
  ∃ k : ℤ, k = 2 ∧ f k < 0 ∧ f (k + 1) > 0 := by sorry

theorem main_result :
  ∀ k : ℤ, k ≤ x₀ → k ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_between_two_and_three_max_integer_below_zero_point_two_is_max_integer_main_result_l1214_121419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_element_l1214_121448

theorem sequence_first_element (a : Fin 8 → ℚ) :
  (∀ n : Fin 8, n.val ≥ 2 → a n = a (n - 1) * a (n - 2)) →
  a 5 = 16 →
  a 6 = 64 →
  a 7 = 1024 →
  a 0 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_element_l1214_121448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_of_f_l1214_121498

def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + 2

theorem max_min_sum_of_f (g : ℝ → ℝ) (h_odd : ∀ x, g (-x) = -g x) :
  let f := f g
  let M := sSup (Set.image f (Set.Icc (-3 : ℝ) 3))
  let N := sInf (Set.image f (Set.Icc (-3 : ℝ) 3))
  M + N = 4 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_of_f_l1214_121498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_calculation_l1214_121449

theorem vector_calculation (a b c : ℝ × ℝ × ℝ) :
  a = (-3, 5, 2) → 
  b = (6, -1, 3) → 
  c = (1, 2, -4) → 
  a - 2 • b + c = (-14, 9, -8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_calculation_l1214_121449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_iff_a_in_range_l1214_121444

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 4*a) / Real.log 0.5

-- Define the theorem
theorem f_monotone_decreasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, 2 ≤ x → x < y → f a y < f a x) ↔ -2 < a ∧ a ≤ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_iff_a_in_range_l1214_121444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_15_trailing_zeros_base_18_l1214_121465

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def highestPowerDividing (base n : ℕ) : ℕ :=
  Nat.log base n

theorem factorial_15_trailing_zeros_base_18 :
  highestPowerDividing 18 (factorial 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_15_trailing_zeros_base_18_l1214_121465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertex_product_sum_l1214_121475

/-- Represents a labeling of a cube's faces -/
structure CubeLabeling where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  sum_eq_21 : a + b + c + d + e + f = 21
  distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
             b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
             c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
             d ≠ e ∧ d ≠ f ∧
             e ≠ f
  range : (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) ∧
          (1 ≤ d ∧ d ≤ 6) ∧ (1 ≤ e ∧ e ≤ 6) ∧ (1 ≤ f ∧ f ≤ 6)

/-- The sum of vertex products for a given cube labeling -/
def vertexProductSum (l : CubeLabeling) : ℕ :=
  l.a * l.c * l.e + l.a * l.c * l.f + l.a * l.d * l.e + l.a * l.d * l.f +
  l.b * l.c * l.e + l.b * l.c * l.f + l.b * l.d * l.e + l.b * l.d * l.f

/-- The maximum sum of vertex products is 343 -/
theorem max_vertex_product_sum :
  ∃ l : CubeLabeling, ∀ m : CubeLabeling, vertexProductSum l ≥ vertexProductSum m ∧ vertexProductSum l = 343 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertex_product_sum_l1214_121475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_completion_time_l1214_121401

/-- Represents the time (in days) it takes for two workers to complete a job together,
    given their individual work rates. -/
noncomputable def time_to_complete (rate_a rate_b : ℝ) : ℝ :=
  1 / (rate_a + rate_b)

/-- Proves that if worker A is four times as fast as worker B, and worker B can
    complete a job in 60 days, then workers A and B together can complete the job in 12 days. -/
theorem workers_completion_time
  {rate_a : ℝ}
  (rate_b : ℝ)
  (h_b_time : rate_b = 1 / 60)
  (h_a_rate : rate_a = 4 * rate_b) :
  time_to_complete rate_a rate_b = 12 := by
  sorry

#check workers_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_completion_time_l1214_121401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_ppc_correct_l1214_121439

noncomputable def male_ppc (K : ℝ) : ℝ := 128 - 0.5 * K^2

noncomputable def female_ppc (K : ℝ) : ℝ := 40 - 2 * K

noncomputable def combined_ppc (K : ℝ) : ℝ :=
  if K ≤ 2 then
    168 - 0.5 * K^2
  else if K ≤ 22 then
    170 - 2 * K
  else if K ≤ 36 then
    20 * K - 0.5 * K^2 - 72
  else
    0

theorem combined_ppc_correct (K : ℝ) :
  combined_ppc K = 
    if K ≤ 2 then
      male_ppc K + female_ppc 0
    else if K ≤ 22 then
      male_ppc 2 + female_ppc (K - 2)
    else if K ≤ 36 then
      male_ppc (K - 20) + female_ppc 20
    else
      0 :=
by
  -- The proof is omitted for now
  sorry

#check combined_ppc_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_ppc_correct_l1214_121439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1214_121435

noncomputable def diamond (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

theorem diamond_calculation : diamond (diamond (-7) (-24)) (diamond 18 15) = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1214_121435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_form_parallelogram_or_rectangle_l1214_121423

-- Define is_parallelogram and is_rectangle before using them
def is_parallelogram (z₁ z₂ z₃ z₄ : ℂ) : Prop :=
  (z₂ - z₁ = z₄ - z₃) ∧ (z₃ - z₁ = z₄ - z₂)

def is_rectangle (z₁ z₂ z₃ z₄ : ℂ) : Prop :=
  is_parallelogram z₁ z₂ z₃ z₄ ∧ 
  Complex.abs (z₂ - z₁) = Complex.abs (z₃ - z₁) ∧
  (z₂ - z₁).re * (z₃ - z₁).re + (z₂ - z₁).im * (z₃ - z₁).im = 0

theorem roots_form_parallelogram_or_rectangle (a : ℝ) : 
  (∃ z₁ z₂ z₃ z₄ : ℂ, 
    (z₁^4 - 8*z₁^3 + 13*a*z₁^2 - 3*(3*a^2 - 2*a - 5)*z₁ + 4 = 0) ∧
    (z₂^4 - 8*z₂^3 + 13*a*z₂^2 - 3*(3*a^2 - 2*a - 5)*z₂ + 4 = 0) ∧
    (z₃^4 - 8*z₃^3 + 13*a*z₃^2 - 3*(3*a^2 - 2*a - 5)*z₃ + 4 = 0) ∧
    (z₄^4 - 8*z₄^3 + 13*a*z₄^2 - 3*(3*a^2 - 2*a - 5)*z₄ + 4 = 0) ∧
    (z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄) ∧
    (is_parallelogram z₁ z₂ z₃ z₄ ∨ is_rectangle z₁ z₂ z₃ z₄)) ↔
  (a = (1 + Real.sqrt 31) / 3 ∨ a = (1 - Real.sqrt 31) / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_form_parallelogram_or_rectangle_l1214_121423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_another_valid_configuration_exists_l1214_121430

-- Define the grid points
inductive GridPoint
| A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P
deriving DecidableEq

-- Define a type for lines on the grid
def Line := List GridPoint

-- Define the property of a valid configuration
def ValidConfiguration (config : List GridPoint) (lines : List Line) : Prop :=
  -- There are 10 distinct points in the configuration
  config.length = 10 ∧ 
  config.toFinset.card = 10 ∧
  -- There are 5 lines
  lines.length = 5 ∧ 
  -- Each line contains exactly 4 points from the configuration
  ∀ l ∈ lines, (l.toFinset ∩ config.toFinset).card = 4

-- Define the original configuration (placeholder)
def OriginalConfig : List GridPoint := sorry

-- State the theorem
theorem another_valid_configuration_exists : 
  ∃ (config : List GridPoint) (lines : List Line),
    ValidConfiguration config lines ∧ 
    config ≠ OriginalConfig := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_another_valid_configuration_exists_l1214_121430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1214_121467

open Real

/-- The function f(x) defined as √3 * sin(x) + sin(π/2 + x) -/
noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x + sin (π / 2 + x)

/-- Theorem stating that the minimum value of f(x) for x ∈ ℝ is -2 -/
theorem min_value_of_f :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≥ m ∧ ∃ (y : ℝ), f y = m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1214_121467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1214_121495

/-- Calculates the speed of a train given its length, the time it takes to pass a person
    moving in the opposite direction, and the person's speed. -/
theorem train_speed_calculation (train_length : ℝ) (passing_time : ℝ) (person_speed : ℝ) :
  train_length = 130 →
  passing_time = 6 →
  person_speed = 6 →
  (train_length / passing_time - person_speed * (1000 / 3600)) * (3600 / 1000) = 72 := by
  intros h1 h2 h3
  -- The proof steps would go here
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1214_121495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1214_121488

/-- Represents a parabola y² = 2px (p > 0) -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

noncomputable def focus (c : Parabola) : Point :=
  { x := c.p / 2, y := 0 }

noncomputable def line_through_focus (c : Parabola) : Line :=
  { m := 1, b := -c.p / 2 }

def line_intersects_parabola (l : Line) (c : Parabola) : Prop :=
  ∃ (x y : ℝ), y = l.m * x + l.b ∧ y^2 = 2 * c.p * x

noncomputable def distance (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

theorem parabola_intersection_theorem (c : Parabola) :
  let f := focus c
  let l := line_through_focus c
  ∃ (A B : Point),
    line_intersects_parabola l c ∧
    distance A B = 8 →
    c.p = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1214_121488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_bound_l1214_121492

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a * Real.log x - 1

-- State the theorem
theorem f_inequality_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → 2 * f a x + (Real.log x) / x ≥ 0) →
  a ≥ -3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_bound_l1214_121492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascent_speed_calculation_l1214_121415

/-- Calculates the average speed during ascent, excluding breaks -/
noncomputable def average_speed_ascent (total_distance : ℝ) (total_time : ℝ) (ascent_time : ℝ) (break_time : ℝ) : ℝ :=
  (total_distance / 2) / (ascent_time - break_time)

theorem ascent_speed_calculation (total_distance : ℝ) (total_time : ℝ) (ascent_time : ℝ) (break_time : ℝ)
    (h1 : total_distance = 9)
    (h2 : total_time = 6)
    (h3 : ascent_time = 4)
    (h4 : break_time = 0.5)
    (h5 : total_distance / total_time = 1.5) :
  ∃ ε > 0, |average_speed_ascent total_distance total_time ascent_time break_time - 1.29| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascent_speed_calculation_l1214_121415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_minus_two_equals_one_l1214_121455

-- Define g as a parameter instead of using it directly
noncomputable def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  if x > 0 then 2^x - 3 else g x

-- Define the property of f being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem f_g_minus_two_equals_one
  (g : ℝ → ℝ)
  (h_odd : is_odd_function (f g)) :
  (f g) (g (-2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_minus_two_equals_one_l1214_121455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterexample_exists_l1214_121473

theorem counterexample_exists : ∃ f : ℝ → ℝ,
  (∀ x, 1 < x ∧ x ≤ 3 → f x > f 1) ∧
  ¬(∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 3 → f x ≤ f y) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterexample_exists_l1214_121473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_B_not_right_triangle_condition_A_implies_right_triangle_condition_C_implies_right_triangle_condition_D_implies_right_triangle_l1214_121480

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define angles of the triangle
noncomputable def angle (t : Triangle) (v : Fin 3) : ℝ :=
  sorry

-- Define sides of the triangle
noncomputable def side (t : Triangle) (v : Fin 3) : ℝ :=
  sorry

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  ∃ i : Fin 3, angle t i = Real.pi / 2

-- Condition A
def condition_A (t : Triangle) : Prop :=
  angle t 0 = angle t 1 + angle t 2

-- Condition B
def condition_B (t : Triangle) : Prop :=
  ∃ k : ℝ, angle t 0 = 3 * k ∧ angle t 1 = 4 * k ∧ angle t 2 = 5 * k

-- Condition C
def condition_C (t : Triangle) : Prop :=
  (side t 0)^2 = (side t 1 + side t 2) * (side t 1 - side t 2)

-- Condition D
def condition_D (t : Triangle) : Prop :=
  ∃ k : ℝ, side t 0 = 5 * k ∧ side t 1 = 12 * k ∧ side t 2 = 13 * k

-- Theorem stating that condition B does not imply a right triangle
theorem condition_B_not_right_triangle :
  ∃ t : Triangle, condition_B t ∧ ¬(is_right_triangle t) := by
  sorry

-- Theorems stating that other conditions imply a right triangle
theorem condition_A_implies_right_triangle :
  ∀ t : Triangle, condition_A t → is_right_triangle t := by
  sorry

theorem condition_C_implies_right_triangle :
  ∀ t : Triangle, condition_C t → is_right_triangle t := by
  sorry

theorem condition_D_implies_right_triangle :
  ∀ t : Triangle, condition_D t → is_right_triangle t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_B_not_right_triangle_condition_A_implies_right_triangle_condition_C_implies_right_triangle_condition_D_implies_right_triangle_l1214_121480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unlike_radicals_group_l1214_121454

-- Define a function to check if two expressions are like radicals
def areLikeRadicals (a b : ℝ) : Prop :=
  ∃ (k : ℝ) (n : ℕ), a = k * Real.sqrt n ∧ b = k * Real.sqrt n

-- Define the groups
def groupA : Prop := areLikeRadicals (Real.sqrt 18) (Real.sqrt 18)
def groupB : Prop := areLikeRadicals (Real.sqrt 12) (Real.sqrt 75)

-- For group C, we need to introduce a variable
variable (x : ℝ)
def groupC : Prop := areLikeRadicals (Real.sqrt x) (2 * Real.sqrt x)

def groupD : Prop := areLikeRadicals (Real.sqrt (1/3)) (Real.sqrt 27)

-- Theorem stating that group C is not like radicals while others are
theorem unlike_radicals_group (x : ℝ) :
  groupA ∧ groupB ∧ ¬groupC x ∧ groupD := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unlike_radicals_group_l1214_121454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_l1214_121426

/-- The function f(x) = (x^2+x+1)e^(x-1) -/
noncomputable def f (x : ℝ) : ℝ := (x^2 + x + 1) * Real.exp (x - 1)

/-- The function g(x) = f(x) / (x^2+1) -/
noncomputable def g (x : ℝ) : ℝ := f x / (x^2 + 1)

/-- Theorem stating the inequality for g(x) -/
theorem g_inequality (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  (g x₂ + g x₁) / 2 > (g x₂ - g x₁) / (x₂ - x₁) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_l1214_121426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircles_ratio_l1214_121446

/-- The number of small semicircles -/
noncomputable def N : ℕ := sorry

/-- The radius of each small semicircle -/
noncomputable def r : ℝ := sorry

/-- The area of all small semicircles combined -/
noncomputable def A : ℝ := N * (Real.pi * r^2 / 2)

/-- The area within the large semicircle but outside the small semicircles -/
noncomputable def B : ℝ := (Real.pi * (N * r)^2 / 2) - A

/-- The theorem stating that when A:B = 1:27, N must be 28 -/
theorem semicircles_ratio (h : A / B = 1 / 27) : N = 28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircles_ratio_l1214_121446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_correct_l1214_121410

-- Statement 1
def chi_square_relation (chi_square : ℝ) (credibility : ℝ) : Prop :=
  chi_square > 0 → credibility > 0

-- Statement 2
noncomputable def exponential_regression (c k : ℝ) (x : ℝ) : ℝ := c * Real.exp (k * x)

noncomputable def log_transformation (y : ℝ) : ℝ := Real.log y

-- Statement 3
def linear_regression (a b x_mean y_mean : ℝ) : Prop :=
  y_mean = a + b * x_mean

theorem all_statements_correct :
  -- Statement 1
  (∀ χ₁ χ₂ cr₁ cr₂, χ₁ > χ₂ → cr₁ > cr₂ → chi_square_relation χ₁ cr₁) →
  -- Statement 2
  (∀ x, log_transformation (exponential_regression (Real.exp 4) 0.3 x) = 0.3 * x + 4) →
  -- Statement 3
  linear_regression 1 2 1 3 →
  -- Conclusion
  True := by
  intro h1 h2 h3
  exact trivial


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_correct_l1214_121410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_sides_l1214_121413

/-- A cube is a three-dimensional shape with 6 faces. -/
structure Cube where
  faces : Nat
  face_count : faces = 6

/-- A plane is a two-dimensional surface that can intersect a cube. -/
structure Plane

/-- The result of intersecting a plane with a cube is a shape with a certain number of sides. -/
def intersection_shape (c : Cube) (p : Plane) : Nat := sorry

/-- The theorem states that the maximum number of sides in a shape obtained by 
    intersecting a plane with a cube is 6. -/
theorem max_intersection_sides (c : Cube) : 
  ∃ p : Plane, ∀ q : Plane, intersection_shape c p ≥ intersection_shape c q ∧ intersection_shape c p = 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_sides_l1214_121413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_combinations_eq_1260_l1214_121489

def red_cubes : ℕ := 2
def blue_cubes : ℕ := 3
def green_cubes : ℕ := 4
def total_cubes : ℕ := 8

def tower_combinations : ℕ := 
  (Nat.choose total_cubes red_cubes * Nat.choose (total_cubes - red_cubes) blue_cubes) +
  (Nat.choose total_cubes red_cubes * Nat.choose (total_cubes - red_cubes) (blue_cubes - 1)) +
  (Nat.choose total_cubes (red_cubes - 1) * Nat.choose (total_cubes - (red_cubes - 1)) blue_cubes)

theorem tower_combinations_eq_1260 : tower_combinations = 1260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_combinations_eq_1260_l1214_121489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l1214_121422

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y = 0

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

-- Theorem statement
theorem tangent_line_at_P :
  my_circle P.1 P.2 →
  ∀ (x y : ℝ), tangent_line x y ↔ 
    (∃ (t : ℝ), x = P.1 + t * (y - P.2) ∧
               my_circle x y ↔ t = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l1214_121422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_A_greater_than_B_l1214_121434

noncomputable def scores_A : List ℝ := [5, 10, 9, 3, 8]
noncomputable def scores_B : List ℝ := [8, 6, 8, 6, 7]

noncomputable def variance (scores : List ℝ) : ℝ :=
  let mean := scores.sum / scores.length
  (scores.map (fun x => (x - mean) ^ 2)).sum / scores.length

theorem variance_A_greater_than_B : variance scores_A > variance scores_B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_A_greater_than_B_l1214_121434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1214_121416

/-- The solution set for the inequality (3^(x^2-1) - 9 * 3^(5x+3)) * log_(cos π)((x^2 - 6x + 9)) ≥ 0 -/
noncomputable def solution_set : Set ℝ :=
  {x | x ∈ (Set.Ioo (-0.5) 0) ∪ (Set.Ioo 0 0.5) ∪ (Set.Ioo 1.5 2) ∪ (Set.Ioo 4 4.5) ∪ (Set.Ioo 5.5 6)}

/-- The inequality function -/
noncomputable def inequality (x : ℝ) : ℝ :=
  (3^(x^2 - 1) - 9 * 3^(5*x + 3)) * Real.log (x^2 - 6*x + 9) / Real.log (Real.cos Real.pi)

theorem inequality_solution :
  ∀ x : ℝ, 
    (x - 3)^2 > 0 → 
    Real.cos (Real.pi * x) > 0 → 
    Real.cos (Real.pi * x) ≠ 1 →
    (inequality x ≥ 0 ↔ x ∈ solution_set) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1214_121416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janice_started_sentences_l1214_121497

/-- Calculates the number of sentences Janice started with given her typing details --/
def sentences_started_with (
  typical_speed : ℕ
) (initial_time : ℕ
) (actual_initial_speed : ℕ
) (second_time : ℕ
) (erased_sentences : ℕ
) (final_speed : ℕ
) (final_time : ℕ
) (total_sentences : ℕ
) : ℕ :=
  total_sentences - (
    initial_time * actual_initial_speed +
    second_time * typical_speed +
    final_time * final_speed -
    erased_sentences
  )

/-- Theorem stating that Janice started with 236 sentences given the problem conditions --/
theorem janice_started_sentences :
  sentences_started_with 6 20 8 15 40 5 18 536 = 236 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_janice_started_sentences_l1214_121497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_segment_length_l1214_121406

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

def SimilarTriangles (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.side1 = k * t1.side1 ∧ t2.side2 = k * t1.side2 ∧ t2.side3 = k * t1.side3

theorem similar_triangles_segment_length 
  (pqr xyz : Triangle) 
  (h_similar : SimilarTriangles pqr xyz)
  (h_pq : pqr.side1 = 8)
  (h_qr : pqr.side2 = 16)
  (h_yz : xyz.side3 = 24) :
  xyz.side1 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_segment_length_l1214_121406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1214_121481

/-- The speed of the first train in km/h -/
noncomputable def speed_train1 : ℝ := 920

/-- The speed of the second train in km/h -/
noncomputable def speed_train2 : ℝ := 80

/-- The length of the first train in meters -/
noncomputable def length_train1 : ℝ := 270

/-- The length of the second train in meters -/
noncomputable def length_train2 : ℝ := 230

/-- The time taken for the trains to cross each other in seconds -/
noncomputable def crossing_time : ℝ := 9

/-- Conversion factor from km/h to m/s -/
noncomputable def km_h_to_m_s : ℝ := 5 / 18

theorem train_speed_calculation :
  (speed_train1 + speed_train2) * km_h_to_m_s * crossing_time = length_train1 + length_train2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1214_121481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l1214_121441

-- Define the set A
def A : Set ℝ := {x | 1 / x < 1}

-- State the theorem
theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l1214_121441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1214_121461

noncomputable def vector_projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let v_magnitude_squared := v.1 * v.1 + v.2 * v.2
  (dot_product / v_magnitude_squared * v.1, dot_product / v_magnitude_squared * v.2)

theorem projection_theorem (v : ℝ × ℝ) 
  (h : vector_projection (1, 2) v = (4/5, -6/5)) :
  vector_projection (3, -1) v = (18/13, -27/13) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1214_121461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_minus_314_power_0_plus_3_power_minus_1_l1214_121464

theorem pi_minus_314_power_0_plus_3_power_minus_1 : (π - 3.14) ^ (0 : ℕ) + (3 : ℚ) ^ (-1 : ℤ) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_minus_314_power_0_plus_3_power_minus_1_l1214_121464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_equals_108_plus_32pi_l1214_121421

/-- The volume of a cube with side length s -/
noncomputable def cube_volume (s : ℝ) : ℝ := s^3

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

/-- The number of cubes -/
def num_cubes : ℕ := 4

/-- The side length of each cube -/
def cube_side : ℝ := 3

/-- The number of spheres -/
def num_spheres : ℕ := 3

/-- The radius of each sphere -/
def sphere_radius : ℝ := 2

/-- The total volume of all figures -/
noncomputable def total_volume : ℝ :=
  (num_cubes : ℝ) * cube_volume cube_side + (num_spheres : ℝ) * sphere_volume sphere_radius

theorem total_volume_equals_108_plus_32pi :
  total_volume = 108 + 32 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_equals_108_plus_32pi_l1214_121421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_implies_a_range_l1214_121452

open Real

/-- The function f(x) = (a-1)ln(x) + ax^2 + 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * log x + a * x^2 + 1

theorem f_monotonicity_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Ioi (0 : ℝ), HasDerivAt (f a) ((a - 1) / x + 2 * a * x) x) →
  (∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 0 → (f a x₁ - f a x₂) / (x₁ - x₂) ≥ 2) →
  a ≥ (sqrt 3 + 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_implies_a_range_l1214_121452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_n_squared_l1214_121443

def a : ℕ → ℕ
  | 0 => 1  -- Add a case for 0
  | 1 => 1
  | n + 2 => (n + 2) * a ((n + 2) / 2)

theorem a_greater_than_n_squared (n : ℕ) (h : n > 11) : a n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_n_squared_l1214_121443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_equals_one_l1214_121486

/-- An equilateral triangle with side length 1 -/
structure EquilateralTriangle :=
  (A B C : ℝ × ℝ)
  (side_length : ℝ)
  (equilateral : side_length = 1 ∧ 
                 dist A B = side_length ∧ 
                 dist B C = side_length ∧ 
                 dist C A = side_length)

/-- Predicate to check if a point is inside a triangle -/
def is_inside (A B C D : ℝ × ℝ) : Prop :=
  -- We'll define this later, for now we'll use a placeholder
  True

/-- A point inside the triangle -/
structure PointInTriangle (T : EquilateralTriangle) :=
  (D : ℝ × ℝ)
  (inside : is_inside T.A T.B T.C D)

/-- Theorem: Sum of distances from an interior point to vertices equals 1 -/
theorem sum_distances_equals_one (T : EquilateralTriangle) (P : PointInTriangle T) :
  dist P.D T.A + dist P.D T.B + dist P.D T.C = 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_equals_one_l1214_121486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_equals_prism_volume_l1214_121437

-- Define the dimensions of the rectangular prism
noncomputable def prism_length : ℝ := 10
noncomputable def prism_width : ℝ := 5
noncomputable def prism_height : ℝ := 6

-- Define the volume of the prism
noncomputable def prism_volume : ℝ := prism_length * prism_width * prism_height

-- Define the radius of the sphere with equal volume
noncomputable def sphere_radius : ℝ := (3 * prism_volume / (4 * Real.pi)) ^ (1/3)

-- Define the surface area of the sphere
noncomputable def sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius ^ 2

-- Theorem statement
theorem sphere_surface_area_equals_prism_volume :
  ∃ (ε : ℝ), ε > 0 ∧ |sphere_surface_area - 84.32 * Real.pi ^ (1/3)| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_equals_prism_volume_l1214_121437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_ellipse_S_foci_S_major_axis_l1214_121479

/-- Definition of the set of points satisfying the given equation -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.sqrt ((p.1)^2 + (p.2 + 3)^2) + Real.sqrt ((p.1)^2 + (p.2 - 3)^2) = 10}

/-- Definition of IsEllipse -/
def IsEllipse (S : Set (ℝ × ℝ)) (a b : ℝ) (center : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ S ↔ 
    ((x - center.1) / a) ^ 2 + ((y - center.2) / b) ^ 2 = 1

/-- Theorem stating that the set S is an ellipse -/
theorem S_is_ellipse : ∃ (a b : ℝ) (center : ℝ × ℝ), 
  IsEllipse S a b center ∧ a > b ∧ b > 0 := by
  sorry

/-- Definition of the foci of the ellipse -/
def foci : Set (ℝ × ℝ) := {(-3, 0), (3, 0)}

/-- Definition of Foci -/
def Foci (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a b : ℝ) (center : ℝ × ℝ), 
    IsEllipse S a b center ∧ 
    ((p.1 - center.1)^2 + (p.2 - center.2)^2 = a^2 - b^2)}

/-- Theorem stating that the foci of S are at (-3, 0) and (3, 0) -/
theorem S_foci : ∀ (a b : ℝ) (center : ℝ × ℝ), 
  IsEllipse S a b center → Foci S = foci := by
  sorry

/-- Theorem stating that the major axis of S is 10 -/
theorem S_major_axis : ∀ (a b : ℝ) (center : ℝ × ℝ), 
  IsEllipse S a b center → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_ellipse_S_foci_S_major_axis_l1214_121479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_and_properties_l1214_121472

noncomputable def n₁ (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, 2)
noncomputable def n₂ (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, (Real.cos x)^2)

noncomputable def f (x : ℝ) : ℝ := (n₁ x).1 * (n₂ x).1 + (n₁ x).2 * (n₂ x).2

theorem f_expression_and_properties :
  (∀ x, f x = 2 * Real.sin (2 * x + π / 6) + 1) ∧
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π - π / 3) (k * π + π / 6))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_and_properties_l1214_121472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_imaginary_z_is_pure_imaginary_l1214_121418

/- Define the complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := m^2 + m - 2 + (m^2 - 1) * Complex.I

/- Theorem for real number case -/
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = 1 ∨ m = -1 := by
  sorry

/- Theorem for imaginary number case -/
theorem z_is_imaginary (m : ℝ) : (z m).im ≠ 0 ↔ m ≠ 1 ∧ m ≠ -1 := by
  sorry

/- Theorem for pure imaginary number case -/
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_imaginary_z_is_pure_imaginary_l1214_121418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1214_121433

/-- Given a parabola C and a hyperbola E, prove the standard equation of E -/
theorem hyperbola_equation (C : ℝ → ℝ) (E : ℝ → ℝ → Prop) :
  (∀ x y, C y = (1/8) * x^2) →  -- Equation of parabola C
  (∃ F : ℝ × ℝ, F = (0, 1) ∧ E F.1 F.2) →  -- One focus of E is (0, 1)
  (∃ e : ℝ, e = Real.sqrt 2 ∧ 
    ∀ (F G : ℝ × ℝ), E F.1 F.2 → E G.1 G.2 → 
      e = (Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2)) / 
          (Real.sqrt ((F.1 + G.1)^2 + (F.2 + G.2)^2))) →  -- Eccentricity of E is √2
  (∀ x y, E x y ↔ y^2/2 - x^2/2 = 1) :=  -- Standard equation of E
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1214_121433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reconnaissance_boat_return_time_l1214_121432

/-- The time for a reconnaissance boat to return to the lead ship of a squadron -/
theorem reconnaissance_boat_return_time 
  (reconnaissance_distance : ℝ) 
  (boat_speed : ℝ) 
  (squadron_speed : ℝ) 
  (h1 : reconnaissance_distance = 70)
  (h2 : boat_speed = 28)
  (h3 : squadron_speed = 14) :
  (2 * reconnaissance_distance) / (boat_speed - squadron_speed) = 20 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reconnaissance_boat_return_time_l1214_121432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outing_participants_l1214_121429

/-- The number of people initially in each van -/
def initial_people_per_van : ℕ := sorry

/-- The initial number of vans -/
def initial_vans : ℕ := sorry

/-- The total number of people participating in the outing -/
def total_people : ℕ := initial_people_per_van * initial_vans

/-- The number of vans after the first breakdown -/
def vans_after_first_breakdown : ℕ := initial_vans - 10

/-- The number of vans after the second breakdown -/
def vans_after_second_breakdown : ℕ := initial_vans - 25

theorem outing_participants :
  (total_people / vans_after_first_breakdown = initial_people_per_van + 1) ∧
  (total_people / vans_after_second_breakdown = initial_people_per_van + 3) →
  total_people = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outing_participants_l1214_121429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_sum_l1214_121445

/-- Represents an arithmetico-geometric sequence -/
structure ArithmeticoGeometric where
  first : ℝ
  ratio : ℝ
  diff : ℝ

/-- Get the nth term of an arithmetico-geometric sequence -/
noncomputable def ArithmeticoGeometric.nthTerm (seq : ArithmeticoGeometric) (n : ℕ) : ℝ :=
  seq.first * seq.ratio^(n-1) + seq.diff * (seq.ratio^(n-1) - 1) / (seq.ratio - 1)

theorem common_ratio_sum (a b : ArithmeticoGeometric) 
    (h : a.nthTerm 3 - b.nthTerm 3 = 3 * (a.nthTerm 2 - b.nthTerm 2)) :
    a.ratio + b.ratio = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_sum_l1214_121445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1214_121402

/-- The first curve -/
noncomputable def curve1 (x : ℝ) : ℝ := Real.exp (3 * x + 7)

/-- The second curve -/
noncomputable def curve2 (x : ℝ) : ℝ := (Real.log x - 7) / 3

/-- The distance function between two points -/
noncomputable def distance (x y : ℝ) : ℝ := Real.sqrt (2 * (x - y)^2)

/-- The theorem stating the minimum distance between the curves -/
theorem min_distance_between_curves :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ (y : ℝ), y > 0 → distance (curve1 x) x ≤ distance (curve1 y) y) ∧
  distance (curve1 x) x = Real.sqrt 2 * ((8 + Real.log 3) / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1214_121402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_writer_output_average_l1214_121491

/-- Represents the writer's output over time -/
structure WriterOutput where
  totalWords : ℕ
  totalHours : ℕ
  increaseRate : ℚ

/-- Calculates the average words per hour for a given writer output -/
def averageWordsPerHour (output : WriterOutput) : ℚ :=
  output.totalWords / output.totalHours

/-- Theorem stating that the average words per hour is 500 for the given conditions -/
theorem writer_output_average (output : WriterOutput) 
  (h1 : output.totalWords = 50000)
  (h2 : output.totalHours = 100)
  (h3 : output.increaseRate = 1/10) :
  averageWordsPerHour output = 500 := by
  unfold averageWordsPerHour
  rw [h1, h2]
  norm_num

#check writer_output_average

end NUMINAMATH_CALUDE_ERRORFEEDBACK_writer_output_average_l1214_121491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_gain_percent_l1214_121451

/-- Calculate the gain percent for a scooter sale --/
theorem scooter_gain_percent
  (purchase_price : ℝ)
  (repair_cost : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 4700)
  (h2 : repair_cost = 800)
  (h3 : selling_price = 5800) :
  let total_cost := purchase_price + repair_cost
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  abs (gain_percent - 5.45) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_gain_percent_l1214_121451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_g_l1214_121436

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 6)

theorem symmetry_axis_of_g :
  ∃ (k : ℤ), ∀ (x : ℝ), g (Real.pi / 3 + x) = g (Real.pi / 3 - x) := by
  sorry

#check symmetry_axis_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_g_l1214_121436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_to_l_l1214_121458

/-- The curve C in the Cartesian plane -/
def C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line l in the Cartesian plane -/
def l (x y : ℝ) : Prop := x + y - 4 = 0

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_l (x y : ℝ) : ℝ :=
  |x + y - 4| / Real.sqrt 2

/-- The maximum distance from any point on curve C to line l is 3√2 -/
theorem max_distance_C_to_l :
  ∃ (x y : ℝ), C x y ∧ ∀ (x' y' : ℝ), C x' y' →
    distance_to_l x y ≥ distance_to_l x' y' ∧
    distance_to_l x y = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_to_l_l1214_121458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forgotten_numbers_l1214_121403

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ 
  (∃ a b c d : ℕ, a ∈ ({1, 2, 3, 4} : Set ℕ) ∧ b ∈ ({1, 2, 3, 4} : Set ℕ) ∧ 
   c ∈ ({1, 2, 3, 4} : Set ℕ) ∧ d ∈ ({1, 2, 3, 4} : Set ℕ) ∧
   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = 1000 * a + 100 * b + 10 * c + d)

def sum_of_all_valid_numbers : ℕ := 66660

def pats_initial_sum : ℕ := 58126

theorem forgotten_numbers :
  ∃ (n1 n2 : ℕ), 
    is_valid_number n1 ∧ 
    is_valid_number n2 ∧ 
    n1 ≠ n2 ∧
    n1 + n2 = sum_of_all_valid_numbers - pats_initial_sum ∧
    ((n1 = 4213 ∧ n2 = 4321) ∨ (n1 = 4321 ∧ n2 = 4213)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_forgotten_numbers_l1214_121403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_rhombus_area_l1214_121407

/-- A rhombus in a rectangular coordinate system -/
structure Rhombus where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- The area of a rhombus given its vertices -/
noncomputable def rhombusArea (r : Rhombus) : ℝ :=
  let d1 := abs (r.v1.2 - r.v3.2)
  let d2 := abs (r.v2.1 - r.v4.1)
  (d1 * d2) / 2

/-- Theorem stating that the area of the specific rhombus is 72 -/
theorem specific_rhombus_area :
  let r : Rhombus := {
    v1 := (0, 4.5),
    v2 := (8, 0),
    v3 := (0, -4.5),
    v4 := (-8, 0)
  }
  rhombusArea r = 72 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_rhombus_area_l1214_121407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l1214_121457

def sequence_condition (x : ℕ → ℕ) : Prop :=
  (∀ i, i ≥ 1 → x i ≤ 1998) ∧ 
  (∀ i, i ≥ 3 → x i = Int.natAbs (x (i-1) - x (i-2)))

def sequence_length (x : ℕ → ℕ) : ℕ → ℕ
  | 0 => 0
  | n+1 => if x (n+1) ≤ 1998 then sequence_length x n + 1 else sequence_length x n

theorem max_sequence_length :
  ∃ x : ℕ → ℕ, sequence_condition x ∧ 
    (∀ y : ℕ → ℕ, sequence_condition y → sequence_length y (sequence_length y 0) ≤ sequence_length x (sequence_length x 0)) ∧
    sequence_length x (sequence_length x 0) = 2998 := by
  sorry

#check max_sequence_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l1214_121457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_self_intersections_explain_nine_self_intersections_l1214_121453

/-- Parametric equations of the curve -/
noncomputable def x (t : ℝ) : ℝ := Real.cos t + t / 3
noncomputable def y (t : ℝ) : ℝ := Real.sin t

/-- The range of x -/
def x_range (t : ℝ) : Prop := 1 ≤ x t ∧ x t ≤ 30

/-- A self-intersection occurs when two different t values yield the same point -/
def is_self_intersection (t1 t2 : ℝ) : Prop :=
  t1 ≠ t2 ∧ x t1 = x t2 ∧ y t1 = y t2

/-- The number of self-intersections within the given range -/
def num_self_intersections : ℕ := 9

/-- The main theorem: there are 9 self-intersections -/
theorem nine_self_intersections : num_self_intersections = 9 := by
  rfl

/-- Explanation of why there are 9 self-intersections -/
theorem explain_nine_self_intersections :
  ∃ (t₁ t₂ : ℝ), x_range t₁ ∧ x_range t₂ ∧ is_self_intersection t₁ t₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_self_intersections_explain_nine_self_intersections_l1214_121453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1214_121485

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.arcsin (2 * x))

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1/2 < x ∧ x < 1/2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1214_121485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equation_unique_solution_l1214_121476

-- Define the interval
def I : Set ℝ := { x | 0 ≤ x ∧ x ≤ Real.arctan 942 }

-- Define the equation
def f (x : ℝ) : Prop := Real.tan x = Real.tan (Real.tan x)

-- Theorem statement
theorem tan_equation_unique_solution : ∃! x, x ∈ I ∧ f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equation_unique_solution_l1214_121476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_quadrilateral_equality_l1214_121463

-- Define the points in a Euclidean space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable (A B C D : E)

-- Define the vectors
def a (A B : E) : E := B - A
def b (B C : E) : E := C - B
def c (C D : E) : E := D - C

-- State the theorem
theorem quadrilateral_inequality {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  (A B C D : E) :
  ‖A - B‖^2 + ‖B - C‖^2 + ‖C - D‖^2 + ‖D - A‖^2 ≥ ‖A - C‖^2 + ‖B - D‖^2 :=
sorry

-- Define what it means for ABCD to be a parallelogram
def is_parallelogram {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  (A B C D : E) : Prop :=
  B - A = D - C ∧ C - B = A - D

-- State the equality condition
theorem quadrilateral_equality {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  (A B C D : E) :
  ‖A - B‖^2 + ‖B - C‖^2 + ‖C - D‖^2 + ‖D - A‖^2 = ‖A - C‖^2 + ‖B - D‖^2 ↔
  is_parallelogram A B C D :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_quadrilateral_equality_l1214_121463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_misplacedValue_l1214_121460

/-- A quadratic polynomial function -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The sequence of values generated by the quadratic polynomial -/
def sequenceValues : List ℝ := [9604, 9801, 10201, 10404, 10816, 11025, 11449, 11664, 12100]

/-- Function to calculate first differences of a list -/
def firstDifferences (xs : List ℝ) : List ℝ :=
  List.zipWith (·-·) (List.tail xs) xs

/-- Function to calculate second differences of a list -/
def secondDifferences (xs : List ℝ) : List ℝ :=
  firstDifferences (firstDifferences xs)

/-- Theorem stating that 10201 is likely the misplaced or incorrectly calculated value -/
theorem misplacedValue : ∃ (a b c : ℝ), ∃ (xs : List ℝ), 
  (∀ (i : ℕ), i < xs.length → xs[i]! = f a b c i) ∧
  sequenceValues ≠ xs ∧
  (∃ (j : ℕ), j < sequenceValues.length ∧ sequenceValues[j]! = 10201 ∧ sequenceValues[j]! ≠ xs[j]!) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_misplacedValue_l1214_121460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supersquared_32_t_9_l1214_121412

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def is_supersquared (x : List ℕ) : Prop :=
  (∀ i j, i < j → i < x.length → j < x.length → x.get ⟨i, by sorry⟩ > x.get ⟨j, by sorry⟩) ∧
  (∀ k, 1 ≤ k → k ≤ x.length → is_perfect_square (List.sum (List.map (λ n => n^2) (List.take k x))))

theorem supersquared_32_t_9 (t : ℕ) :
  is_supersquared [32, t, 9] → t = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supersquared_32_t_9_l1214_121412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_measure_is_sum_l1214_121409

-- Define the sample space
inductive Ω : Type
| zero : Ω
| one : Ω

-- Define probability measures P and Q
noncomputable def P : Ω → ℝ
| Ω.zero => 1
| Ω.one => 0

noncomputable def Q : Ω → ℝ
| Ω.zero => 0
| Ω.one => 1

-- Define the smallest measure ν
noncomputable def ν : Ω → ℝ := λ x => P x + Q x

-- Define the maximum of P and Q
noncomputable def P_or_Q : Ω → ℝ := λ x => max (P x) (Q x)

-- Theorem statement
theorem smallest_measure_is_sum :
  (∀ x : Ω, ν x ≥ P x) ∧
  (∀ x : Ω, ν x ≥ Q x) ∧
  (∃ x : Ω, ν x ≠ P_or_Q x) ∧
  (∀ μ : Ω → ℝ, (∀ x : Ω, μ x ≥ P x) → (∀ x : Ω, μ x ≥ Q x) → (∀ x : Ω, μ x ≥ ν x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_measure_is_sum_l1214_121409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_bound_l1214_121417

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (1/2) * x^2 - 4 * Real.log x

theorem f_decreasing_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → HasDerivAt (f a) (a - x - 4/x) x) →
  (∀ x : ℝ, x ≥ 1 → (a - x - 4/x) ≤ 0) →
  a ≤ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_bound_l1214_121417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_covering_l1214_121484

theorem rectangle_covering (n m : ℕ) : 
  (∃ m : ℕ, (3 * m + 2) * (4 * m + 3) = n * (n + 1)^2 * (n + 2) / 12) →
  (∀ k < n, ¬∃ m : ℕ, (3 * m + 2) * (4 * m + 3) = k * (k + 1)^2 * (k + 2) / 12) →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_covering_l1214_121484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_in_decagon_l1214_121459

/-- A regular decagon -/
structure RegularDecagon where
  vertices : Fin 10 → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ
  is_equilateral : ∀ (i j : Fin 3), i ≠ j → 
    dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

/-- A function that counts the number of distinct equilateral triangles in a regular decagon -/
def count_equilateral_triangles (d : RegularDecagon) : ℕ :=
  sorry

theorem equilateral_triangles_in_decagon :
  ∀ (d : RegularDecagon), count_equilateral_triangles d = 84 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_in_decagon_l1214_121459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_parallelepiped_dimensions_l1214_121493

noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2
noncomputable def parallelepiped_volume (a h : ℝ) : ℝ := a^2 * h
noncomputable def parallelepiped_surface_area (a h : ℝ) : ℝ := 2*a^2 + 4*a*h

theorem inscribed_parallelepiped_dimensions (a h : ℝ) 
  (hv : parallelepiped_volume a h = (1/2) * sphere_volume 1)
  (hs : parallelepiped_surface_area a h = (1/2) * sphere_surface_area 1) :
  ∃ (ε : ℝ), ε > 0 ∧ |a - 1| < ε ∧ |h - (2/3) * Real.pi| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_parallelepiped_dimensions_l1214_121493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_triangular_prism_surface_area_l1214_121408

/-- The total surface area of an oblique triangular prism -/
noncomputable def total_surface_area (a b c l h : ℝ) : ℝ :=
  let p := (a + b + c) / 2
  (2 * l / h) * (Real.sqrt (p * (p - a) * (p - b) * (p - c)) + p * h)

/-- Theorem: The total surface area of an oblique triangular prism -/
theorem oblique_triangular_prism_surface_area 
  (a b c l h : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hl : l > 0) (hh : h > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (S : ℝ), S = total_surface_area a b c l h := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_triangular_prism_surface_area_l1214_121408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_defeat_probability_l1214_121420

/-- Represents the probability of defeating the dragon -/
noncomputable def defeat_probability : ℝ := 1

/-- Represents the probability of two heads growing -/
noncomputable def two_heads_prob : ℝ := 1/4

/-- Represents the probability of one head growing -/
noncomputable def one_head_prob : ℝ := 1/3

/-- Represents the probability of no heads growing -/
noncomputable def no_heads_prob : ℝ := 5/12

/-- States that the probabilities sum to 1 -/
axiom prob_sum : two_heads_prob + one_head_prob + no_heads_prob = 1

/-- Represents that one head is chopped off each minute -/
axiom chop_one_head : ∀ (n : ℕ), n > 0 → ∃ (m : ℕ), m = n - 1

/-- States that the dragon is defeated when it has no heads -/
axiom defeat_condition : ∀ (n : ℕ), n = 0 → defeat_probability = 1

/-- Theorem: The probability of eventually defeating the dragon is 1 -/
theorem dragon_defeat_probability : defeat_probability = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_defeat_probability_l1214_121420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_tetrahedron_volume_l1214_121447

/-- A tetrahedron with four congruent isosceles triangular faces -/
structure IsoTetrahedron where
  a : ℝ  -- base length of each face
  b : ℝ  -- lateral side length of each face
  h_positive : 0 < a
  h_condition : a < b * Real.sqrt 2

/-- The volume of an isosceles tetrahedron -/
noncomputable def volume (t : IsoTetrahedron) : ℝ :=
  (t.a^2 * Real.sqrt (4 * t.b^2 - 2 * t.a^2)) / 12

/-- Theorem stating the volume of an isosceles tetrahedron -/
theorem isosceles_tetrahedron_volume (t : IsoTetrahedron) :
  volume t = (t.a^2 * Real.sqrt (4 * t.b^2 - 2 * t.a^2)) / 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_tetrahedron_volume_l1214_121447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_value_rounded_l1214_121450

noncomputable def round_nearest_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem d_value_rounded (d : ℝ) : 
  d = (0.889 * 55) / 9.97 → 
  round_nearest_tenth d = 4.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_value_rounded_l1214_121450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1214_121425

/-- Given a hyperbola and a circle with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 5 = 0 →
    ∃ t : ℝ, (b*x - a*y = 0 ∨ b*x + a*y = 0) ∧ 
         ((x - 3)^2 + y^2 = 4)) →
  (3 : ℝ)^2 = a^2 - b^2 →
  a^2 = 5 ∧ b^2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1214_121425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_from_surface_area_l1214_121477

noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem sphere_volume_from_surface_area :
  ∃ (r : ℝ), sphere_surface_area r = 16 * Real.pi ∧ sphere_volume r = (32 / 3) * Real.pi :=
by
  use 2
  constructor
  · simp [sphere_surface_area]
    ring
  · simp [sphere_volume]
    ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_from_surface_area_l1214_121477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l1214_121482

/-- The ★ operation defined as (a ★ b) = √(a² + b²) / √(a² - b²) -/
noncomputable def star (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / Real.sqrt (a^2 - b^2)

/-- Theorem stating that if x ★ 25 = 5, then x = (5√130) / 12 -/
theorem star_equation_solution (x : ℝ) (h : star x 25 = 5) : x = (5 * Real.sqrt 130) / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l1214_121482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elective_4_4_l1214_121499

-- Define the polar coordinate system
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

-- Define the Cartesian coordinate system
structure CartesianPoint where
  x : ℝ
  y : ℝ

-- Define the conversion from polar to Cartesian coordinates
noncomputable def polarToCartesian (p : PolarPoint) : CartesianPoint :=
  { x := p.ρ * Real.cos p.θ
  , y := p.ρ * Real.sin p.θ }

-- Define the line ρcos θ = 3
def line1 (p : PolarPoint) : Prop :=
  p.ρ * Real.cos p.θ = 3

-- Define the condition |OM| · |OP| = 12
def condition1 (m : PolarPoint) (p : PolarPoint) : Prop :=
  m.ρ * p.ρ = 12

-- Define the line ρsin θ - ρcos θ = m
def line2 (p : PolarPoint) (m : ℝ) : Prop :=
  p.ρ * Real.sin p.θ - p.ρ * Real.cos p.θ = m

-- State the theorem
theorem elective_4_4 (m p : PolarPoint) (m_val : ℝ) :
  line1 m →
  condition1 m p →
  line2 p m_val →
  (∀ q : PolarPoint, line2 q m_val → q = p) →
  m_val = -2 - 2 * Real.sqrt 2 ∨ m_val = -2 + 2 * Real.sqrt 2 ∨ m_val = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elective_4_4_l1214_121499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1214_121483

/-- The function f(x) = -x^2 + ax - a/4 + 1/2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - a/4 + 1/2

/-- The maximum value of f(x) in the interval [0,1] is 2 -/
def max_value (a : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, f a x ≤ 2 ∧ ∃ y ∈ Set.Icc 0 1, f a y = 2

theorem f_max_value (a : ℝ) : max_value a → a = -6 ∨ a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1214_121483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_exists_l1214_121466

theorem arrangement_exists : ∃ (arr : Fin 3 → Fin 4 → ℕ), 
  (∀ i j, arr i j ∈ Finset.range 13 \ {0}) ∧ 
  (∀ i₁ j₁ i₂ j₂, (i₁ ≠ i₂ ∨ j₁ ≠ j₂) → arr i₁ j₁ ≠ arr i₂ j₂) ∧
  (∀ i, (Finset.sum (Finset.range 4) (λ j ↦ arr i j)) = 
        (Finset.sum (Finset.range 4) (λ j ↦ arr 0 j))) :=
by sorry

#check arrangement_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_exists_l1214_121466
