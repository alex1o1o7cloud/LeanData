import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_change_approx_net_salary_change_example_l174_17418

/-- Calculates the net change in salary after a series of percentage changes -/
noncomputable def net_salary_change (initial_salary : ℝ) : ℝ :=
  let after_first_increase := initial_salary * 1.20
  let after_first_decrease := after_first_increase * 0.90
  let after_second_increase := after_first_decrease * 1.15
  let final_salary := after_second_increase * 0.95
  (final_salary - initial_salary) / initial_salary

/-- The net change in salary is approximately 17.99% -/
theorem salary_change_approx (initial_salary : ℝ) (h : initial_salary > 0) :
  ∃ ε > 0, abs (net_salary_change initial_salary - 0.1799) < ε := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use a theorem instead
theorem net_salary_change_example :
  ∃ ε > 0, abs (net_salary_change 1000 - 0.1799) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_change_approx_net_salary_change_example_l174_17418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonal_AC_l174_17498

/-- A cyclic quadrilateral with integer side lengths -/
structure CyclicQuadrilateral where
  x : ℕ
  y : ℕ
  z : ℕ
  w : ℕ
  distinct : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w
  less_than_12 : x < 12 ∧ y < 12 ∧ z < 12 ∧ w < 12
  xy_eq_wz : x * y = w * z

/-- The length of the diagonal AC in a cyclic quadrilateral -/
noncomputable def diagonal_AC (q : CyclicQuadrilateral) : ℝ :=
  Real.sqrt (q.x^2 + q.y^2 + q.z^2 + q.w^2)

theorem max_diagonal_AC :
  ∃ q : CyclicQuadrilateral, ∀ q' : CyclicQuadrilateral,
    diagonal_AC q ≥ diagonal_AC q' ∧ diagonal_AC q = Real.sqrt 272 := by
  sorry

#check max_diagonal_AC

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonal_AC_l174_17498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l174_17487

noncomputable def power_function (m : ℝ) : ℝ → ℝ := fun x ↦ x ^ m

theorem power_function_value (m : ℝ) :
  (∀ x : ℝ, power_function m x = power_function m (-x)) →  -- even function
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → power_function m x₂ < power_function m x₁) →  -- decreasing for x > 0
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l174_17487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l174_17457

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b * Real.cos (t.A / 2) = t.a * Real.sin t.B ∧
  t.a = 6 ∧
  ∃ (O : Real × Real), 
    (O.1 = (t.A + t.B + t.C) / 3) ∧ 
    (O.2 = (t.a + t.b + t.c) / 3) ∧
    Real.sqrt ((O.1 - t.A)^2 + (O.2 - t.a)^2) = 2 * Real.sqrt 3

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.A = Real.pi / 3 ∧ 
  (1/2 * t.a * t.b * Real.sin t.C) = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l174_17457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_specific_triangle_l174_17499

/-- The sum of distances from a point to the vertices of a triangle --/
noncomputable def sum_distances (D E F P : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) +
  Real.sqrt ((P.1 - E.1)^2 + (P.2 - E.2)^2) +
  Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

/-- Theorem about the sum of distances for a specific triangle and point --/
theorem sum_distances_specific_triangle :
  let D : ℝ × ℝ := (0, 0)
  let E : ℝ × ℝ := (10, -2)
  let F : ℝ × ℝ := (6, 6)
  let P : ℝ × ℝ := (5, 3)
  sum_distances D E F P = 7 * Real.sqrt 46 ∧ 0 + 7 + 46 = 53 := by
  sorry

#check sum_distances_specific_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_specific_triangle_l174_17499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l174_17491

noncomputable def f (a : ℝ) (x : ℝ) := Real.log (a * x^2 - x + 1 / (16 * a))

def p (a : ℝ) := ∀ x, ∃ y, f a x = f a y

def q (a : ℝ) := ∀ x, 3^x - 9^x < a

theorem problem_statement :
  (∀ a, p a → a > 2) ∧
  (∀ a, (p a ∨ q a) ∧ ¬(p a ∧ q a) → 1/4 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l174_17491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_iff_not_neg_four_fifths_l174_17440

def line1 (a t : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 2 + 3*t
  | 1 => 1 + 4*t
  | 2 => a + 5*t

def line2 (u : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 5 + 6*u
  | 1 => 3 + 3*u
  | 2 => 1 + 2*u

def are_skew (a : ℝ) : Prop :=
  ∀ t u : ℝ, line1 a t ≠ line2 u

theorem skew_iff_not_neg_four_fifths (a : ℝ) :
  are_skew a ↔ a ≠ -4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_iff_not_neg_four_fifths_l174_17440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_neg_two_sqrt_two_l174_17486

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -2 * x else
  if 0 ≤ x ∧ x < Real.pi / 2 then 4 * Real.cos (13 * x) else 0

theorem f_composition_equals_neg_two_sqrt_two :
  f (f (-Real.pi / 8)) = -2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_neg_two_sqrt_two_l174_17486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_sum_theorem_l174_17404

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define point F
def F : ℝ × ℝ := (3, 0)

-- Define vector from F to a point (x, y)
def vector_F_to (x y : ℝ) : ℝ × ℝ := (x - F.1, y - F.2)

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Theorem statement
theorem ellipse_vector_sum_theorem (A B C : ℝ × ℝ) 
  (hA : ellipse A.1 A.2) 
  (hB : ellipse B.1 B.2) 
  (hC : ellipse C.1 C.2) 
  (h_sum : vector_F_to A.1 A.2 + vector_F_to B.1 B.2 + vector_F_to C.1 C.2 = (0, 0)) :
  magnitude (vector_F_to A.1 A.2) + magnitude (vector_F_to B.1 B.2) + magnitude (vector_F_to C.1 C.2) = 48/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_sum_theorem_l174_17404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_cosine_fraction_l174_17413

theorem max_value_cosine_fraction (x : ℝ) :
  (2 + Real.cos x) / (2 - Real.cos x) ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_cosine_fraction_l174_17413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_on_interval_l174_17410

noncomputable def f (x : ℝ) : ℝ := (1/4) * x^4 + (1/3) * x^3 + (1/2) * x^2

theorem min_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-1) 1 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-1) 1 → f y ≥ f x) ∧
  f x = 0 := by
  -- The proof goes here
  sorry

#check min_value_of_f_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_on_interval_l174_17410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_chip_value_l174_17428

def chip_values (x : ℕ) : List ℕ := [1, 5, x, 11]

theorem purple_chip_value :
  ∀ x : ℕ,
  5 < x →
  x < 11 →
  (∃ (subset : List ℕ), subset.all (λ y => y ∈ chip_values x) ∧ subset.prod = 140800) →
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_chip_value_l174_17428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l174_17450

-- Define the functions f and h
noncomputable def f (x : ℝ) : ℝ := 18 / (x + 2)
noncomputable def h (x : ℝ) : ℝ := 2 * (Function.invFun f x)

-- State the theorem
theorem solution_exists : ∃ x : ℝ, h x = 12 ∧ x = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l174_17450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l174_17485

-- Define the function f(x) = (x-3)e^x
noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x y : ℝ, x < y → y < 2 → f y < f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l174_17485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_value_l174_17401

-- Define the radius of the larger circle
def R : ℝ := 10

-- Define the distance from the center of the larger circle to the tangent point of smaller circles
def d : ℝ := 4

-- Define the radius of the smaller circles
noncomputable def r : ℝ := Real.sqrt (R^2 - d^2)

-- Define the area of the shaded region
noncomputable def shaded_area : ℝ := Real.pi * R^2 - 2 * Real.pi * r^2

theorem shaded_area_value : shaded_area = -68 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_value_l174_17401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_in_fourth_quadrant_l174_17482

/-- The complex number Z -/
noncomputable def Z : ℂ := 2 / (3 - Complex.I) + Complex.I ^ 2015

/-- Theorem: Z is in the fourth quadrant -/
theorem Z_in_fourth_quadrant :
  (Z.re > 0) ∧ (Z.im < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_in_fourth_quadrant_l174_17482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_axonometric_area_ratio_l174_17430

/-- Represents a triangle with base on the x-axis -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- The area of a triangle -/
noncomputable def triangle_area (t : Triangle) : ℝ := (1/2) * t.base * t.height

/-- The height of the intuitive diagram after oblique axonometric drawing -/
noncomputable def intuitive_height (t : Triangle) : ℝ := (Real.sqrt 2/4) * t.base

/-- The area of the intuitive diagram after oblique axonometric drawing -/
noncomputable def intuitive_area (t : Triangle) : ℝ := (1/2) * t.base * (intuitive_height t)

/-- The ratio of the intuitive diagram area to the original triangle area -/
noncomputable def area_ratio (t : Triangle) : ℝ := (intuitive_area t) / (triangle_area t)

theorem oblique_axonometric_area_ratio (t : Triangle) :
  area_ratio t = Real.sqrt 2/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_axonometric_area_ratio_l174_17430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l174_17451

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | 1 => 4/9
  | k + 2 => (mySequence k * mySequence (k + 1)) / (3 * mySequence k - 2 * mySequence (k + 1))

theorem mySequence_formula (n : ℕ) : n ≥ 1 → mySequence (n - 1) = 8 / (4 * n + 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l174_17451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_bound_trigonometric_sum_bound_tight_l174_17437

open Real

theorem trigonometric_sum_bound (θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ) :
  Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + Real.cos θ₃ * Real.sin θ₄ + 
  Real.cos θ₄ * Real.sin θ₅ + Real.cos θ₅ * Real.sin θ₆ + Real.cos θ₆ * Real.sin θ₁ ≤ 3 := by
  sorry

theorem trigonometric_sum_bound_tight :
  ∃ θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ,
    Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + Real.cos θ₃ * Real.sin θ₄ + 
    Real.cos θ₄ * Real.sin θ₅ + Real.cos θ₅ * Real.sin θ₆ + Real.cos θ₆ * Real.sin θ₁ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_bound_trigonometric_sum_bound_tight_l174_17437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_quality_related_to_production_line_probability_selecting_line_a_l174_17415

-- Define the contingency table data
def line_a_fair : ℕ := 40
def line_a_good : ℕ := 80
def line_b_fair : ℕ := 80
def line_b_good : ℕ := 100
def total_sample : ℕ := 300

-- Define the K^2 statistic formula
noncomputable def k_squared (a b c d n : ℕ) : ℝ :=
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 90% confidence
def critical_value : ℝ := 2.706

-- Define the probability calculation for the second part
def prob_at_least_one_line_a : ℚ := 3 / 5

-- Theorem statements
theorem product_quality_related_to_production_line :
  k_squared line_a_fair line_a_good line_b_fair line_b_good total_sample > critical_value := by
  sorry

theorem probability_selecting_line_a :
  prob_at_least_one_line_a = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_quality_related_to_production_line_probability_selecting_line_a_l174_17415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l174_17411

/-- The distance between a point (x₀, y₀) and a line ax + by + c = 0 --/
noncomputable def distance_point_line (x₀ y₀ a b c : ℝ) : ℝ :=
  (|a * x₀ + b * y₀ + c|) / Real.sqrt (a^2 + b^2)

/-- The theorem stating that the line 4x - 3y - 2 = 0 intersects the circle (x-3)² + (y+5)² = 36 --/
theorem line_intersects_circle :
  let line_a : ℝ := 4
  let line_b : ℝ := -3
  let line_c : ℝ := -2
  let circle_center_x : ℝ := 3
  let circle_center_y : ℝ := -5
  let circle_radius : ℝ := 6
  distance_point_line circle_center_x circle_center_y line_a line_b line_c < circle_radius :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l174_17411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equiv_range_l174_17494

/-- The function f(x) = ln x + 2^x -/
noncomputable def f (x : ℝ) : ℝ := Real.log x + (2 : ℝ)^x

/-- Theorem stating that f(x^2 + 2) < f(3x) if and only if 1 < x < 2 -/
theorem inequality_equiv_range (x : ℝ) :
  f (x^2 + 2) < f (3*x) ↔ 1 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equiv_range_l174_17494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l174_17408

theorem trigonometric_simplification (α : ℝ) : 
  Real.cos (270 * π / 180 - 2 * α) * 
  (Real.cos (30 * π / 180 - 2 * α) / Real.sin (30 * π / 180 - 2 * α)) * 
  Real.tan (240 * π / 180 - 2 * α) * 
  (2 * Real.cos (4 * α) - 1) = 
  -Real.sin (6 * α) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l174_17408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l174_17456

-- Problem 1
theorem simplify_expression_1 : 
  (1 : ℝ) + 2^(-2 : ℝ) * (2 + 1/4)^(-1/2 : ℝ) - (1/100 : ℝ)^(1/2 : ℝ) = 16/15 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (5/6 : ℝ) * a^(1/3 : ℝ) * b^(-2 : ℝ) * (-3 : ℝ) * a^(-1/2 : ℝ) * b^(-1 : ℝ) / (4 * a^(2/3 : ℝ) * b^(-3 : ℝ))^(1/2 : ℝ) = 
  -5 * (a * b)^(1/2 : ℝ) / (4 * a * b^2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l174_17456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l174_17493

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*x + a^2 + 3*a - 3

def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

noncomputable def M (a : ℝ) : ℝ × ℝ := (a^2/4, a)

def focus_F : ℝ × ℝ := (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def p (a : ℝ) : Prop := ∃ x, f a x < 0

def q (a : ℝ) : Prop := distance (M a) focus_F > 2

theorem a_range (a : ℝ) : 
  (¬(¬p a) ∧ ¬(p a ∧ q a)) → a ∈ Set.Icc (-2) 1 ∧ a ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l174_17493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_is_real_l174_17454

noncomputable def h (x : ℝ) : ℝ := (x^4 - 4*x^2 + 4) / (|x - 4| + |x + 2|)

theorem h_domain_is_real : 
  ∀ x : ℝ, |x - 4| + |x + 2| ≠ 0 → h x ∈ Set.univ :=
by
  intro x h
  simp [h, Set.mem_univ]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_is_real_l174_17454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_single_meal_cost_l174_17471

/-- The cost of a meal with given numbers of sandwiches, coffees, and pies -/
def meal_cost (s c p : ℚ) : ℚ → Prop := λ cost => cost = s + c + p

/-- The cost of the first meal -/
axiom meal1 : meal_cost 2 5 1 5

/-- The cost of the second meal -/
axiom meal2 : meal_cost 3 8 1 7

/-- Theorem: The cost of a meal with 1 sandwich, 1 coffee, and 1 pie is $3.00 -/
theorem single_meal_cost : meal_cost 1 1 1 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_single_meal_cost_l174_17471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_minus_2b_l174_17462

-- Define the vectors a and b
noncomputable def a : ℝ × ℝ := sorry
noncomputable def b : ℝ × ℝ := sorry

-- Define the angle between a and b
noncomputable def angle : ℝ := 2 * Real.pi / 3

-- Define the magnitudes of a and b
axiom mag_a : Real.sqrt (a.1^2 + a.2^2) = 4
axiom mag_b : Real.sqrt (b.1^2 + b.2^2) = 2

-- Define the angle between a and b using dot product
axiom angle_def : Real.cos angle = (a.1 * b.1 + a.2 * b.2) / (4 * 2)

-- Theorem to prove
theorem magnitude_a_minus_2b : 
  Real.sqrt ((a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_minus_2b_l174_17462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l174_17442

noncomputable section

def A : ℝ × ℝ := (1, 4)
def B : ℝ × ℝ := (-2, 3)
def C : ℝ × ℝ := (2, -1)

def vec_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vec_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def vec_OC : ℝ × ℝ := C

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem problem_solution :
  (dot_product vec_AB vec_AC = 2) ∧
  (vector_length (vec_AB.1 + vec_AC.1, vec_AB.2 + vec_AC.2) = 2 * Real.sqrt 10) ∧
  (∃ t : ℝ, t = -1 ∧ dot_product (vec_AB.1 - t * vec_OC.1, vec_AB.2 - t * vec_OC.2) vec_OC = 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l174_17442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_result_l174_17431

open Real

/-- A function satisfying the given conditions -/
noncomputable def f : ℝ → ℝ := sorry

/-- The derivative of f -/
noncomputable def f' : ℝ → ℝ := sorry

/-- f satisfies f(x) + f'(x) > 2 for all x -/
axiom f_condition (x : ℝ) : f x + f' x > 2

/-- f satisfies e^(f(1)) = 2e + 4 -/
axiom f_value : exp (f 1) = 2 * exp 1 + 4

/-- Main theorem: e^x * f(x) > 4 + 2e^x if and only if x > 1 -/
theorem main_result (x : ℝ) : exp x * f x > 4 + 2 * exp x ↔ x > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_result_l174_17431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_p_employee_increase_l174_17406

/-- The percentage increase in employees for Company P from January to December --/
noncomputable def percentage_increase (jan_employees : ℝ) (dec_employees : ℝ) : ℝ :=
  (dec_employees - jan_employees) / jan_employees * 100

/-- Theorem stating that the percentage increase is approximately 15% --/
theorem company_p_employee_increase : 
  let jan_employees : ℝ := 408.7
  let dec_employees : ℝ := 470
  abs (percentage_increase jan_employees dec_employees - 15) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_p_employee_increase_l174_17406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bc_length_l174_17425

/-- A parabola defined by y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- Triangle ABC with vertices on the parabola y = x^2 -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  on_parabola : 
    A.2 = parabola A.1 ∧
    B.2 = parabola B.1 ∧
    C.2 = parabola C.1

/-- The area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

theorem triangle_bc_length 
  (abc : Triangle)
  (h1 : abc.A = (2, 4))
  (h2 : abc.B.2 = 16 ∧ abc.C.2 = 16)
  (h3 : triangle_area (abc.C.1 - abc.B.1) (16 - 4) = 144) :
  abc.C.1 - abc.B.1 = 24 := by
  sorry

#check triangle_bc_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bc_length_l174_17425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l174_17407

/-- The time taken for two trains moving in opposite directions to cross each other -/
noncomputable def train_crossing_time (speed1 speed2 : ℝ) (length1 length2 : ℝ) : ℝ :=
  (length1 + length2) / ((speed1 + speed2) * 1000 / 3600)

/-- Theorem stating that the time taken for the given trains to cross each other is approximately 19.63 seconds -/
theorem train_crossing_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_crossing_time 100 120 500 700 - 19.63| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l174_17407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l174_17452

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

/-- Sequence a_n defined as f(n) for positive integers n -/
noncomputable def a_n (a : ℝ) (n : ℕ+) : ℝ := f a n

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ+, a_n a n < a_n a (n + 1)) → 
  (2 < a ∧ a < 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l174_17452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l174_17429

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (x^2 + 3*x + k) / (x^2 - x - 12)

theorem one_vertical_asymptote (k : ℝ) :
  (∃! x, ¬ ∃ y, g k x = y) ↔ k = -28 ∨ k = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l174_17429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_costs_l174_17444

/-- Represents a store with a pricing strategy for paddles and table tennis balls. -/
structure Store where
  paddle_price : ℝ
  ball_price : ℝ
  strategy : ℝ → ℝ → ℝ

/-- The cost calculation for Store A -/
def store_A : Store :=
  { paddle_price := 48
    ball_price := 12
    strategy := fun paddles balls => 48 * paddles + 12 * (balls - paddles) }

/-- The cost calculation for Store B -/
def store_B : Store :=
  { paddle_price := 48
    ball_price := 12
    strategy := fun paddles balls => (48 * paddles + 12 * balls) * 0.9 }

/-- The theorem stating the cost formulas for both stores -/
theorem store_costs (x : ℝ) (h : x ≥ 5) :
  store_A.strategy 5 x = 12 * x + 180 ∧
  store_B.strategy 5 x = 10.8 * x + 216 := by
  sorry

#check store_costs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_costs_l174_17444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marys_number_l174_17423

theorem marys_number (m : ℕ) 
  (h1 : 225 ∣ m) 
  (h2 : 45 ∣ m) 
  (h3 : 1000 < m ∧ m < 3000) : 
  m ∈ ({1125, 1350, 1575, 1800, 2025, 2250, 2475, 2700} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marys_number_l174_17423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l174_17432

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_10 (a d : ℝ) :
  arithmetic_sequence a d 3 = 8 →
  arithmetic_sequence a d 6 = 14 →
  sum_arithmetic_sequence a d 10 = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l174_17432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l174_17435

noncomputable section

def f : ℝ → ℝ := sorry

def domain_f : Set ℝ := Set.Icc 0 2

noncomputable def g (x : ℝ) : ℝ := f (x^2) / (x - 1)

def domain_g : Set ℝ := {x : ℝ | -Real.sqrt 2 ≤ x ∧ x < 1 ∨ 1 < x ∧ x ≤ Real.sqrt 2}

theorem domain_of_g (x : ℝ) : 
  x ∈ domain_g ↔ (x^2 ∈ domain_f ∧ x ≠ 1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l174_17435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_pattern_area_l174_17473

/-- The area of alternating semicircles in a given length -/
theorem semicircle_pattern_area 
  (diameter : ℝ) 
  (pattern_length : ℝ) 
  (h1 : diameter > 0) 
  (h2 : pattern_length > 0) :
  (pattern_length / diameter) * π * (diameter / 2)^2 = (11.25 : ℝ) * π := by
  sorry

#check semicircle_pattern_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_pattern_area_l174_17473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_equals_four_fifths_l174_17417

/-- Given that the terminal side of angle α passes through point P(4, -3), prove that cos α = 4/5 -/
theorem cos_alpha_equals_four_fifths (α : ℝ) :
  (∃ (P : ℝ × ℝ), P = (4, -3) ∧ P.1 = 4 * Real.cos α ∧ P.2 = 4 * Real.sin α) →
  Real.cos α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_equals_four_fifths_l174_17417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_nine_factorial_greater_than_eight_factorial_l174_17477

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem divisors_of_nine_factorial_greater_than_eight_factorial :
  (Finset.filter (fun d => d ∣ factorial 9 ∧ d > factorial 8) (Finset.range (factorial 9 + 1))).card = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_nine_factorial_greater_than_eight_factorial_l174_17477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grazing_area_corner_tether_grazing_area_approx_l174_17467

/-- The area over which a horse can graze when tethered to a corner of a rectangular field -/
noncomputable def grazing_area (field_length field_width rope_length : ℝ) : ℝ :=
  (1 / 4) * Real.pi * rope_length ^ 2

/-- Theorem stating the conditions and the grazing area formula -/
theorem grazing_area_corner_tether 
  (field_length field_width rope_length : ℝ) 
  (h1 : field_length > 0) 
  (h2 : field_width > 0) 
  (h3 : rope_length > 0) 
  (h4 : rope_length < field_length) 
  (h5 : rope_length < field_width) :
  grazing_area field_length field_width rope_length = (1 / 4) * Real.pi * rope_length ^ 2 :=
by
  -- Unfold the definition of grazing_area
  unfold grazing_area
  -- The equality holds by definition
  rfl

/-- Numerical approximation of the grazing area for the given problem -/
theorem grazing_area_approx :
  ∃ (area : ℝ), abs (area - grazing_area 46 20 17) < 0.00001 ∧ abs (area - 227.02225) < 0.00001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grazing_area_corner_tether_grazing_area_approx_l174_17467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_60_and_cs_value_l174_17479

/-- Prove that sin(60°) = √3/2 and that given b = (1/3)a, cs = 17/27 -/
theorem sin_60_and_cs_value :
  (Real.sin (60 * π / 180) = Real.sqrt 3 / 2) ∧
  (∀ a b : ℝ, b = (1/3) * a → (17:ℝ)/(27:ℝ) = 17/27) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_60_and_cs_value_l174_17479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_coordinates_l174_17496

/-- The point symmetric to A(x, y, z) with respect to the x-axis in 3D space. -/
def symmetric_point (A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (A.1, -A.2.1, -A.2.2)

/-- Theorem: The point symmetric to A(1, 1, 2) with respect to the x-axis has coordinates (1, -1, -2). -/
theorem symmetric_point_coordinates :
  let A : ℝ × ℝ × ℝ := (1, 1, 2)
  symmetric_point A = (1, -1, -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_coordinates_l174_17496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_segment_length_bound_l174_17416

/-- A point on a face of a unit cube -/
structure FacePoint where
  x : ℝ
  y : ℝ
  z : ℝ
  is_on_face : (x = 0 ∨ x = 1) ∨ (y = 0 ∨ y = 1) ∨ (z = 0 ∨ z = 1)

/-- The set of points on the faces of a unit cube -/
def CubeFacePoints := Fin 6 → FacePoint

/-- The length of a line segment between two points -/
noncomputable def segmentLength (p1 p2 : FacePoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- The sum of lengths of line segments connecting adjacent face points -/
noncomputable def totalLength (points : CubeFacePoints) : ℝ :=
  (segmentLength (points 0) (points 1)) +
  (segmentLength (points 0) (points 2)) +
  (segmentLength (points 0) (points 4)) +
  (segmentLength (points 1) (points 3)) +
  (segmentLength (points 1) (points 5)) +
  (segmentLength (points 2) (points 3)) +
  (segmentLength (points 2) (points 5)) +
  (segmentLength (points 3) (points 4)) +
  (segmentLength (points 4) (points 5))

/-- Theorem: The sum of lengths of line segments connecting adjacent face points is at least 6√2 -/
theorem cube_segment_length_bound (points : CubeFacePoints) :
  totalLength points ≥ 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_segment_length_bound_l174_17416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_k_equals_one_f_is_increasing_l174_17460

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k - 2^(-x)) / (2^(-x+1) + 2)

theorem odd_function_implies_k_equals_one (k : ℝ) :
  (∀ x : ℝ, f k x = -f k (-x)) → k = 1 := by sorry

theorem f_is_increasing :
  ∀ x y : ℝ, x < y → f 1 x < f 1 y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_k_equals_one_f_is_increasing_l174_17460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_millet_majority_on_day_three_millet_minority_on_day_two_l174_17461

/-- Represents the amount of millet in the feeder on a given day -/
noncomputable def milletAmount (day : ℕ) : ℝ :=
  0.6 * (1 - (1/2)^day)

/-- Represents the total amount of seeds in the feeder on a given day -/
def totalSeeds : ℕ → ℝ
  | _ => 1

/-- Theorem stating that on the third day, more than half the seeds are millet -/
theorem millet_majority_on_day_three :
  milletAmount 3 > (1/2) * totalSeeds 3 := by
  sorry

/-- Theorem stating that on the second day, millet is not yet more than half -/
theorem millet_minority_on_day_two :
  milletAmount 2 ≤ (1/2) * totalSeeds 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_millet_majority_on_day_three_millet_minority_on_day_two_l174_17461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_minus_reciprocal_l174_17472

open Real MeasureTheory

theorem integral_exp_minus_reciprocal : 
  ∫ x in Set.Icc 1 2, (exp x - 1 / x) = exp 2 - exp 1 - log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_minus_reciprocal_l174_17472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l174_17488

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem f_properties :
  (∃! (a b : ℝ), a ≠ b ∧ (∀ x : ℝ, (deriv f x = 0) ↔ (x = a ∨ x = b))) ∧
  (∃! (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0) ∧
  f (2 * Real.sin (10 * π / 180)) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l174_17488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_mistake_fraction_l174_17480

theorem student_mistake_fraction (original_number : ℚ) 
  (correct_fraction : ℚ) (extra_amount : ℚ) :
  original_number = 480 →
  correct_fraction = 5 / 16 →
  extra_amount = 250 →
  let correct_answer := correct_fraction * original_number
  let student_answer := correct_answer + extra_amount
  let mistake_fraction := student_answer / original_number
  mistake_fraction = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_mistake_fraction_l174_17480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pineapple_cost_is_14_l174_17438

/-- The cost of a single pineapple -/
def pineapple_cost : ℕ := sorry

/-- The number of pineapples purchased -/
def pineapples_bought : ℕ := 2

/-- The cost of a single watermelon -/
def watermelon_cost : ℕ := 5

/-- The number of watermelons purchased -/
def watermelons_bought : ℕ := sorry

/-- The total amount spent -/
def total_spent : ℕ := 38

/-- Theorem stating that the cost of each pineapple is 14 -/
theorem pineapple_cost_is_14 :
  pineapple_cost = 14 ∧
  pineapples_bought * pineapple_cost + watermelons_bought * watermelon_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pineapple_cost_is_14_l174_17438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_overlaps_inverse_l174_17495

/-- A function f(x) = (2x + 1) / (x + a) where a ≠ 1/2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * x + 1) / (x + a)

/-- The inverse function of f -/
noncomputable def f_inverse (a : ℝ) (x : ℝ) : ℝ := (-a * x + 1) / (x - 2)

theorem function_overlaps_inverse (a : ℝ) (h : a ≠ 1/2) :
  (∀ x : ℝ, f a x = f_inverse a x) → a = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_overlaps_inverse_l174_17495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_example_l174_17497

-- Define the factorization property
def is_factorization (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x ∧ ∃ (a b : ℝ → ℝ), g x = a x * b x

-- State the theorem
theorem factorization_example :
  is_factorization (λ x => x^2 + 2*x + 1) (λ x => (x + 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_example_l174_17497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_plus_alpha_l174_17445

theorem tan_pi_fourth_plus_alpha (α : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = Real.sqrt 5 / 5) :
  Real.tan (π / 4 + α) = -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_plus_alpha_l174_17445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_in_interval_l174_17436

/-- The function f(x) = e^(14|x|) - 1/(1+x^4) -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (14 * abs x) - 1 / (1 + x^4)

/-- Theorem stating that f(2x) < f(1-x) if and only if x ∈ (-1, 1/3) -/
theorem f_inequality_iff_in_interval (x : ℝ) : 
  f (2 * x) < f (1 - x) ↔ x > -1 ∧ x < 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_in_interval_l174_17436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_crosses_4x10_impossible_5x10_l174_17489

/-- Represents a rectangular table with crosses placed in its cells -/
structure CrossTable (m n : ℕ) where
  crosses : Fin m → Fin n → Bool

/-- Checks if a row has an odd number of crosses -/
def has_odd_row_crosses (t : CrossTable m n) (row : Fin m) : Prop :=
  Odd (Finset.card (Finset.filter (λ j => t.crosses row j) (Finset.univ : Finset (Fin n))))

/-- Checks if a column has an odd number of crosses -/
def has_odd_col_crosses (t : CrossTable m n) (col : Fin n) : Prop :=
  Odd (Finset.card (Finset.filter (λ i => t.crosses i col) (Finset.univ : Finset (Fin m))))

/-- Checks if all rows and columns have an odd number of crosses -/
def valid_cross_placement (t : CrossTable m n) : Prop :=
  (∀ row, has_odd_row_crosses t row) ∧ (∀ col, has_odd_col_crosses t col)

/-- Counts the total number of crosses in the table -/
def count_crosses (t : CrossTable m n) : ℕ :=
  Finset.card (Finset.filter (λ (i, j) => t.crosses i j) (Finset.univ : Finset (Fin m × Fin n)))

theorem max_crosses_4x10 :
  (∃ (t : CrossTable 4 10), valid_cross_placement t ∧ 
    (∀ (t' : CrossTable 4 10), valid_cross_placement t' → 
      count_crosses t' ≤ count_crosses t)) ∧
  (∀ (t : CrossTable 4 10), valid_cross_placement t → 
    count_crosses t ≤ 30) :=
sorry

theorem impossible_5x10 :
  ¬∃ (t : CrossTable 5 10), valid_cross_placement t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_crosses_4x10_impossible_5x10_l174_17489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_g_value_l174_17478

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

-- Define the function g
noncomputable def g (x m : ℝ) : ℝ := f x + m / x + 3

-- Theorem statement
theorem f_odd_and_g_value (m : ℝ) 
  (h : g (1/4) m = 5) : 
  (∀ x, f (-x) = -f x) ∧ g (-1/4) m = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_g_value_l174_17478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_cover_unit_square_l174_17453

-- Define a finite collection of squares
def SquareCollection := List Float

-- Define the total area of the square collection
def totalArea (squares : SquareCollection) : Float :=
  squares.map (λ s => s * s) |>.sum

-- Define the property of covering a unit square
def coversUnitSquare (squares : SquareCollection) : Prop :=
  ∃ (arrangement : List (Float × Float × Float)), 
    arrangement.length = squares.length ∧
    (∀ (x y : Float), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → 
      ∃ (i : Nat), i < arrangement.length ∧
        let (side, xPos, yPos) := arrangement[i]!
        x ≥ xPos ∧ x ≤ xPos + side ∧ y ≥ yPos ∧ y ≤ yPos + side)

-- State the theorem
theorem squares_cover_unit_square (squares : SquareCollection) :
  totalArea squares = 4 → coversUnitSquare squares := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_cover_unit_square_l174_17453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l174_17492

theorem power_equation_solution : ∃ x : ℝ, (3 : ℝ)^4 * (3 : ℝ)^x = 81 ∧ x = 0 := by
  use 0
  constructor
  · simp [Real.rpow_add, Real.rpow_nat_cast]
    norm_num
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l174_17492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l174_17466

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  Real.cos A = -3/5 →
  Real.sin C = 1/2 →
  c = 1 →
  let S := (1/2) * a * c * Real.sin B
  S = (8 * Real.sqrt 3 + 6) / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l174_17466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_120_value_l174_17402

def a : ℕ → ℚ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | 2 => 1/2
  | n+3 => (1 - a (n+2)) / (2 * a (n+1))

theorem a_120_value : a 120 = 20/41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_120_value_l174_17402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l174_17412

theorem sin_double_angle_special_case (x : ℝ) :
  Real.sin (x - π / 4) = 3 / 5 → Real.sin (2 * x) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l174_17412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_proof_l174_17421

noncomputable def a : Fin 3 → ℝ := ![4, 3, 1]
noncomputable def b : Fin 3 → ℝ := ![2, -1, 2]
noncomputable def u : Fin 3 → ℝ := ![-8/Real.sqrt 26, -11/Real.sqrt 26, 1/Real.sqrt 26]

theorem bisector_proof :
  (‖u‖ = 1) ∧ 
  (∃ (k : ℝ), k > 0 ∧ k • (a + u) = 2 • b) := by
  sorry

#check bisector_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_proof_l174_17421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_people_round_table_l174_17455

def circular_permutations (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem seven_people_round_table : circular_permutations 7 = 720 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_people_round_table_l174_17455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tap_b_fill_time_l174_17459

-- Define the pool capacity as 1 unit
noncomputable def pool_capacity : ℝ := 1

-- Define the combined rate of taps A and B
noncomputable def combined_rate : ℝ := 1 / 30

-- Define the time taken when both taps are used
noncomputable def both_taps_time : ℝ := 30

-- Define the time for which both taps are on in the second scenario
noncomputable def both_taps_partial_time : ℝ := 10

-- Define the additional time tap B runs alone in the second scenario
noncomputable def tap_b_additional_time : ℝ := 40

-- Theorem to prove
theorem tap_b_fill_time (rate_b : ℝ) : 
  (combined_rate * both_taps_partial_time + rate_b * tap_b_additional_time = pool_capacity) →
  (1 / rate_b = 60) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tap_b_fill_time_l174_17459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_proof_l174_17475

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def start : point := (-3, 6)
def origin : point := (0, 0)
def mid : point := (2, 3)
def destination : point := (7, -2)

theorem total_distance_proof :
  distance start origin + distance origin mid + distance mid destination =
  Real.sqrt 45 + Real.sqrt 13 + 5 * Real.sqrt 2 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_proof_l174_17475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_min_range_l174_17465

def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m+6)*x + 1

theorem function_max_min_range (m : ℝ) : 
  (∃ (x y : ℝ), (∀ z : ℝ, f m z ≤ f m x) ∧ (∀ w : ℝ, f m y ≤ f m w)) ↔ 
  (m < -3 ∨ m > 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_min_range_l174_17465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l174_17483

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the hyperbola -/
def onHyperbola (h : Hyperbola) (p : Point) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Minimum value of |AF_2| + |BF_2| for the given hyperbola -/
theorem min_sum_distances (h : Hyperbola) (f1 f2 a b : Point) :
  h.a = 3 →
  h.b = Real.sqrt 6 →
  onHyperbola h f1 →
  onHyperbola h f2 →
  onHyperbola h a →
  onHyperbola h b →
  f1.x < 0 →
  f2.x > 0 →
  a.x < 0 →
  b.x < 0 →
  ∃ (l : Set Point), f1 ∈ l ∧ a ∈ l ∧ b ∈ l →
  ∀ (a' b' : Point), a' ∈ l → b' ∈ l → onHyperbola h a' → onHyperbola h b' →
    distance a' f2 + distance b' f2 ≥ 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l174_17483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_directional_derivative_u_l174_17468

/-- The function u(x, y, z) = xy + yz + 1 -/
noncomputable def u (x y z : ℝ) : ℝ := x * y + y * z + 1

/-- The direction vector l = (12, -3, -4) -/
def l : Fin 3 → ℝ
| 0 => 12
| 1 => -3
| 2 => -4

/-- The directional derivative of u in the direction of l -/
noncomputable def u_l (x y z : ℝ) : ℝ := (8 * y - 3 * (x + z)) / 13

theorem directional_derivative_u :
  ∀ x y z : ℝ,
  (u_l x y z) = (8 * y - 3 * (x + z)) / 13 ∧
  (u_l 0 (-2) (-1)) = -1 ∧
  (u_l 3 3 5) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_directional_derivative_u_l174_17468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erased_digit_theorem_l174_17458

def is_valid_number (N : ℕ) : Prop :=
  ∃ (a : ℕ), a ∈ ({1, 2, 3} : Set ℕ) ∧ N = 125 * a * (10 ^ 197)

theorem erased_digit_theorem (N : ℕ) :
  (∃ (k : ℕ) (m n : ℕ) (a : ℕ), 
    k = 199 ∧
    a < 10 ∧
    m < 10^k ∧
    N = m + 10^k * a + 10^(k+1) * n ∧
    N = 5 * (m + 10^k * n)) →
  is_valid_number N :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_erased_digit_theorem_l174_17458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_distance_l174_17449

/-- Represents Linda's car and journey --/
structure CarJourney where
  consumption : ℚ  -- gallons per mile (using rational numbers)
  tankCapacity : ℚ -- gallons
  initialFuel : ℚ  -- gallons
  firstLegDistance : ℚ -- miles
  refuel : ℚ -- gallons
  finalFuelRatio : ℚ -- ratio of full tank

/-- Calculates the total distance driven given a car journey --/
def totalDistance (j : CarJourney) : ℚ :=
  let firstLegFuelUsed := j.firstLegDistance * j.consumption
  let secondLegFuel := j.refuel - (j.tankCapacity * j.finalFuelRatio)
  let secondLegDistance := secondLegFuel / j.consumption
  j.firstLegDistance + secondLegDistance

/-- Theorem stating that Linda's total distance driven is 637.5 miles --/
theorem linda_distance : 
  let j : CarJourney := {
    consumption := 1/30,
    tankCapacity := 15,
    initialFuel := 15,
    firstLegDistance := 450,
    refuel := 10,
    finalFuelRatio := 1/4
  }
  totalDistance j = 637.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_distance_l174_17449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_gen_tail_length_l174_17424

/-- The growth factor between generations -/
noncomputable def growth_factor : ℝ := 1.25

/-- The tail length of the third generation in cm -/
noncomputable def third_gen_length : ℝ := 25

/-- The tail length of the first generation in cm -/
noncomputable def first_gen_length : ℝ := third_gen_length / (growth_factor ^ 2)

theorem first_gen_tail_length :
  first_gen_length = 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_gen_tail_length_l174_17424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_lower_bound_l174_17439

theorem sin_lower_bound (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) : Real.sin x ≥ (2 / Real.pi) * x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_lower_bound_l174_17439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_l174_17427

/-- Given a circle and a line of symmetry, proves that the symmetric circle has a specific equation. -/
theorem symmetric_circle_equation :
  ∀ (x y : ℝ),
  let original_circle := x^2 + y^2 - 4*x + 3 = 0
  let symmetry_line := y = (Real.sqrt 3 / 3) * x
  let symmetric_circle := (x - 1)^2 + (y - Real.sqrt 3)^2 = 1
  original_circle → ∃ (x' y' : ℝ), symmetry_line ∧ symmetric_circle :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_l174_17427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l174_17490

/-- The circle C with equation x² + y² = 2x -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 2*x

/-- The line L with equation x = √3y + m -/
def Line (x y m : ℝ) : Prop := x = Real.sqrt 3 * y + m

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The theorem statement -/
theorem intersection_property (m : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    Circle x1 y1 ∧ Circle x2 y2 ∧
    Line x1 y1 m ∧ Line x2 y2 m ∧
    (distance x1 y1 m 0) * (distance x2 y2 m 0) = 1) →
  m = 1 + Real.sqrt 2 ∨ m = 1 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l174_17490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_l174_17470

theorem angle_difference (α β : Real) : 
  0 < α ∧ α < π/2 → 
  0 < β ∧ β < π/2 → 
  Real.sin α = Real.sqrt 5 / 5 → 
  Real.cos β = Real.sqrt 10 / 10 → 
  α - β = -π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_l174_17470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_car_motorcycle_l174_17474

/-- The maximum distance between a car and a motorcycle under specific conditions --/
theorem max_distance_car_motorcycle :
  let initial_distance : ℝ := 9  -- Distance between A and B in km
  let car_speed : ℝ := 40        -- Car speed in km/h
  let motorcycle_accel : ℝ := 32 -- Motorcycle acceleration in km/h²
  let time_limit : ℝ := 2        -- Time limit in hours
  
  -- Car position function
  let car_pos (t : ℝ) := car_speed * t
  
  -- Motorcycle position function
  let motorcycle_pos (t : ℝ) := initial_distance + (1/2) * motorcycle_accel * t^2
  
  -- Distance between car and motorcycle at time t
  let distance (t : ℝ) := |motorcycle_pos t - car_pos t|
  
  -- Maximum distance within the time limit
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ time_limit ∧ distance t = 16 ∧
    ∀ s : ℝ, 0 ≤ s ∧ s ≤ time_limit → distance s ≤ 16 := by
  sorry

#check max_distance_car_motorcycle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_car_motorcycle_l174_17474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_expression_l174_17447

theorem golden_ratio_expression (R : ℝ) (h : R^2 + R = 1) :
  R^(R^(R^2 + R⁻¹) + R⁻¹) + R⁻¹ = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_expression_l174_17447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_profit_percentage_l174_17463

/-- Represents the profit percentage when an article is sold -/
noncomputable def ProfitPercentage (costPrice sellingPrice : ℝ) : ℝ :=
  (sellingPrice - costPrice) / costPrice * 100

theorem initial_profit_percentage
  (costPrice initialSellingPrice : ℝ)
  (initialSellingPrice_positive : 0 < initialSellingPrice)
  (costPrice_positive : 0 < costPrice)
  (initialSellingPrice_greater : costPrice < initialSellingPrice)
  (double_profit_condition : ProfitPercentage costPrice (2 * initialSellingPrice) = 140) :
  ProfitPercentage costPrice initialSellingPrice = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_profit_percentage_l174_17463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_circle_radius_common_tangent_line_l174_17469

-- Define the circle C
structure Circle where
  a : ℝ
  r : ℝ

-- Define the conditions
def tangent_to_line (C : Circle) : Prop :=
  C.r = |C.a - 2| * Real.sqrt 2

def chord_length (C : Circle) : Prop :=
  2 * Real.sqrt (2 * (C.a - 2)^2 - 2 * C.a^2) = 2 * Real.sqrt 6

def externally_tangent (C : Circle) : Prop :=
  Real.sqrt 5 * C.r = C.r + 4 * Real.sqrt 2

-- Theorem statements
theorem circle_equation (C : Circle) 
  (h1 : tangent_to_line C) (h2 : chord_length C) :
  ∃ x y : ℝ, (x - 1/4)^2 + (y - 3/4)^2 = 49/8 := by
  sorry

theorem circle_radius (C : Circle) 
  (h : externally_tangent C) :
  C.r = Real.sqrt 10 + Real.sqrt 2 := by
  sorry

theorem common_tangent_line (C : Circle) 
  (h1 : tangent_to_line C) (h2 : chord_length C) :
  ∃ x y : ℝ, 7*x + y - 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_circle_radius_common_tangent_line_l174_17469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l174_17433

-- Define the centers of the circles
structure Point where
  x : ℝ
  y : ℝ

-- Define the circles
structure Circle where
  center : Point
  radius : ℝ

-- Define the line m
def line_m : Set Point := sorry

-- Define the circles
def circle_A : Circle := { center := { x := -7, y := 3 }, radius := 3 }
def circle_B : Circle := { center := { x := 0, y := 4 }, radius := 4 }
def circle_C : Circle := { center := { x := 9, y := 5 }, radius := 5 }

-- Define the tangent points
def A' : Point := sorry
def B' : Point := sorry
def C' : Point := sorry

-- Helper functions (not proved)
def is_tangent (c1 c2 : Circle) : Prop := sorry
def area_triangle (p1 p2 p3 : Point) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_ABC :
  -- Conditions
  (A' ∈ line_m) →
  (B' ∈ line_m) →
  (C' ∈ line_m) →
  (circle_A.center.x < circle_B.center.x) →
  (circle_B.center.x < circle_C.center.x) →
  (is_tangent circle_A circle_B) →
  (is_tangent circle_B circle_C) →
  -- Conclusion
  area_triangle circle_A.center circle_B.center circle_C.center = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l174_17433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l174_17476

-- Define CirclePosition
inductive CirclePosition
| Intersecting
| InternallyTangent
| ExternallyTangent
| Disjoint

-- Define CirclesTangent
def CirclesTangent (position : CirclePosition) (r₁ r₂ d : ℝ) : Prop :=
  match position with
  | CirclePosition.ExternallyTangent => r₁ + r₂ = d
  | _ => False  -- We only define ExternallyTangent for this problem

-- Define ExternallyTangent
def ExternallyTangent : CirclePosition := CirclePosition.ExternallyTangent

-- The main theorem
theorem circles_externally_tangent (r₁ r₂ : ℝ) : 
  r₁^2 - 7*r₁ + 10 = 0 →
  r₂^2 - 7*r₂ + 10 = 0 →
  r₁ ≠ r₂ →
  (r₁ + r₂ = 7) →
  CirclesTangent ExternallyTangent r₁ r₂ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l174_17476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l174_17420

/-- An ellipse with semi-major axis 3 and semi-minor axis √5 -/
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ
  h_a : a = 3
  h_b : b^2 = 5

/-- The foci of the ellipse -/
noncomputable def Ellipse.foci (e : Ellipse) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (e.a^2 - e.b^2)
  ((e.center.1 - c, e.center.2), (e.center.1 + c, e.center.2))

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  point : ℝ × ℝ
  h_on_ellipse : (point.1 - e.center.1)^2 / e.a^2 + (point.2 - e.center.2)^2 / e.b^2 = 1

/-- The theorem stating that the perimeter of the triangle formed by a point on the ellipse and its foci is 10 -/
theorem ellipse_triangle_perimeter (e : Ellipse) (p : PointOnEllipse e) :
  let (f₁, f₂) := e.foci
  Real.sqrt ((p.point.1 - f₁.1)^2 + (p.point.2 - f₁.2)^2) +
  Real.sqrt ((p.point.1 - f₂.1)^2 + (p.point.2 - f₂.2)^2) +
  Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l174_17420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_connected_components_inequality_l174_17426

/-- A simple graph -/
structure SimpleGraph' (V : Type*) where
  edge : V → V → Prop
  symm : ∀ u v, edge u v → edge v u
  irrefl : ∀ v, ¬edge v v

variable {V : Type*} [Fintype V]
variable (G : SimpleGraph' V)

/-- A subset of edges -/
def EdgeSet (G : SimpleGraph' V) := {e : V × V | G.edge e.fst e.snd}

/-- Number of connected components in a subgraph -/
noncomputable def numConnectedComponents (G : SimpleGraph' V) (A : Set (V × V)) : ℕ := sorry

theorem connected_components_inequality (G : SimpleGraph' V) (A B : Set (V × V)) :
  let a := numConnectedComponents G A
  let b := numConnectedComponents G B
  let c := numConnectedComponents G (A ∪ B)
  let d := numConnectedComponents G (A ∩ B)
  a + b ≤ c + d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_connected_components_inequality_l174_17426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_intersection_l174_17414

noncomputable section

-- Define the curve C₁
def C₁ (α : ℝ) : ℝ × ℝ :=
  (-2 + 2 * Real.cos α, 2 * Real.sin α)

-- Define the rotation function
def rotate (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

-- Define curve C₂ as the rotated version of C₁
def C₂ (θ : ℝ) : ℝ × ℝ :=
  rotate (C₁ (θ + Real.pi/2))

-- Define point F
def F : ℝ × ℝ := (0, -1)

-- Define line L
def L (t : ℝ) : ℝ × ℝ :=
  (t/2, -1 + (Real.sqrt 3)/2 * t)

-- State the theorem
theorem curve_and_intersection :
  (∀ θ, θ ∈ Set.Icc 0 (Real.pi/2) → (C₂ θ).1^2 + (C₂ θ).2^2 = (4 * Real.sin θ)^2) ∧
  (∃ A B, A ∈ C₂ '' Set.Icc 0 (Real.pi/2) ∧ B ∈ C₂ '' Set.Icc 0 (Real.pi/2) ∧
          A ∈ L '' Set.univ ∧ B ∈ L '' Set.univ ∧
          Real.sqrt ((F.1 - A.1)^2 + (F.2 - A.2)^2) +
          Real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2) = 3 * Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_intersection_l174_17414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_and_fraction_value_l174_17443

theorem largest_x_and_fraction_value : ∃ (a b c d : ℤ) (x : ℝ),
  (7 * x / 8 + 1 = 4 / x) ∧
  (x = (a + b * Real.sqrt c) / d) ∧
  (∀ (a' b' c' d' : ℤ) (x' : ℝ), (7 * x' / 8 + 1 = 4 / x') ∧ (x' = (a' + b' * Real.sqrt c') / d') → x' ≤ x) ∧
  (a = -4 ∧ b = 8 ∧ c = 15 ∧ d = 7) ∧
  (a * c * d / b : ℚ) = -105/2 := by
  sorry

#eval ((-4 * 15 * 7) / 8 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_and_fraction_value_l174_17443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_specific_line_l174_17419

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ × ℝ :=
  (0, (l.y₁ * l.x₂ - l.y₂ * l.x₁) / (l.x₂ - l.x₁))

/-- The theorem stating that the y-intercept of the given line is (0, 12) -/
theorem y_intercept_of_specific_line :
  let l : Line := { x₁ := 3, y₁ := 18, x₂ := -9, y₂ := -6 }
  y_intercept l = (0, 12) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_specific_line_l174_17419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l174_17400

/-- Given a positive real number a, and a function f(x) = ax^2 - a^2x - 1/a
    whose graph intersects the x-axis at points A and B (A left of B),
    this theorem states properties about the distance AB and the length OA. -/
theorem quadratic_function_properties (a : ℝ) (h_a_pos : a > 0) :
  let f := λ x : ℝ => a * x^2 - a^2 * x - 1/a
  let A := (a^2 - Real.sqrt (a^4 + 4)) / (2*a)
  let B := (a^2 + Real.sqrt (a^4 + 4)) / (2*a)
  let AB := B - A
  let OA := -A
  -- The minimum value of AB is 2
  (∀ a : ℝ, a > 0 → AB ≥ 2) ∧
  (∃ a : ℝ, a > 0 ∧ AB = 2) ∧
  -- When a ∈ [1, 2√2], the range of OA is [(4√2 - √6)/52, (√5 - 1)/2]
  (∀ a : ℝ, a > 0 → 1 ≤ a → a ≤ 2 * Real.sqrt 2 →
    (4 * Real.sqrt 2 - Real.sqrt 6) / 52 ≤ OA ∧ OA ≤ (Real.sqrt 5 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l174_17400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_four_thirds_pi_minus_alpha_l174_17409

theorem tan_four_thirds_pi_minus_alpha (α : ℝ) 
  (h1 : Real.sin (α + π/6) = -3/5) 
  (h2 : α ∈ Set.Ioo (-2*π/3) (-π/6)) : 
  Real.tan (4*π/3 - α) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_four_thirds_pi_minus_alpha_l174_17409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_l174_17464

theorem solution_pairs (a b : ℕ) :
  (∃ p : ℕ, Nat.Prime p ∧ ∃ k : ℕ, a^2 + b - 1 = p^k) →
  (a^2 + b + 1 ∣ b^2 - a^3 - 1) →
  ¬(a^2 + b + 1 ∣ (a + b - 1)^2) →
  ∃ n : ℕ, n > 1 ∧ a = 2^n ∧ b = 2^(2*n) - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_l174_17464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l174_17448

def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

def symmetry_line (x y : ℝ) : Prop := y = -x + 2

-- Define reflect_point as a sorry function
def reflect_point (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

def is_symmetrical (C : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (center_x center_y : ℝ),
    (∀ x y, C x y ↔ (x - center_x)^2 + (y - center_y)^2 = 1) ∧
    (center_x, center_y) = reflect_point (1, 0) symmetry_line

theorem circle_symmetry :
  ∀ C : ℝ → ℝ → Prop,
  is_symmetrical C →
  (∀ x y, C x y ↔ (x - 2)^2 + (y - 1)^2 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l174_17448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l174_17484

open Real

noncomputable def f (x : ℝ) : ℝ := 4 * sin x * cos (x + π/3) + 4 * Real.sqrt 3 * sin x ^ 2 - Real.sqrt 3

theorem f_properties :
  ∃ (k : ℤ),
    f (π/3) = Real.sqrt 3 ∧
    (∀ x, f x = f (k * π/2 + 5*π/12 - x)) ∧
    (∀ x ∈ Set.Icc (-π/4) (π/3), f x ≤ Real.sqrt 3) ∧
    (∀ x ∈ Set.Icc (-π/4) (π/3), f x ≥ -2) ∧
    (∃ x ∈ Set.Icc (-π/4) (π/3), f x = Real.sqrt 3) ∧
    (∃ x ∈ Set.Icc (-π/4) (π/3), f x = -2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l174_17484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l174_17441

/-- Given vectors a, b, c, and a real number lambda, prove that if (a + lambda*b) is parallel to c, then lambda = 3 -/
theorem parallel_vectors_lambda (a b c : ℝ × ℝ) (lambda : ℝ) :
  a = (2, 1) →
  b = (0, 1) →
  c = (3, 6) →
  (∃ (k : ℝ), a + lambda • b = k • c) →
  lambda = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l174_17441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_side_range_l174_17403

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  t.a + 2 * t.a * Real.cos t.B = t.c

-- Define an acute triangle
def is_acute_triangle (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

-- Theorem 1
theorem angle_relation (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_condition t) : 
  t.B = 2 * t.A := by
  sorry

-- Theorem 2
theorem side_range (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_condition t)
  (h3 : is_acute_triangle t)
  (h4 : t.c = 2) : 
  1 < t.a ∧ t.a < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_side_range_l174_17403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_two_zeros_l174_17434

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 2 then 2 / x else (x - 1) ^ 3

-- State the theorem
theorem k_range_for_two_zeros :
  ∀ k : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = k ∧ f x₂ = k) → k ∈ Set.Ioo 0 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_two_zeros_l174_17434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandwich_cost_theorem_l174_17405

/-- Represents the cost of ingredients for sandwiches -/
structure SandwichCost where
  breadCost : ℚ
  meatCost : ℚ
  cheeseCost : ℚ

/-- Represents the number of ingredients needed for sandwiches -/
structure SandwichIngredients where
  breadLoaves : ℕ
  meatPacks : ℕ
  cheesePacks : ℕ

/-- Calculates the total cost of ingredients -/
def totalCost (cost : SandwichCost) (ingredients : SandwichIngredients) : ℚ :=
  cost.breadCost * (ingredients.breadLoaves : ℚ) +
  cost.meatCost * (ingredients.meatPacks : ℚ) +
  cost.cheeseCost * (ingredients.cheesePacks : ℚ)

/-- Theorem: The cost per sandwich is $2.16 -/
theorem sandwich_cost_theorem (cost : SandwichCost) (ingredients : SandwichIngredients) :
  cost.breadCost = 4 ∧
  cost.meatCost = 5 ∧
  cost.cheeseCost = 4 ∧
  ingredients.breadLoaves = 5 ∧
  ingredients.meatPacks = 10 ∧
  ingredients.cheesePacks = 10 →
  (totalCost cost ingredients - 2) / 50 = 216 / 100 := by
  sorry

#eval (108 : ℚ) / 50  -- Expected output: 27/25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandwich_cost_theorem_l174_17405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_f_smallest_positive_period_g_l174_17422

noncomputable def f (x : ℝ) : ℝ := (Real.sin (3/2 * x))^(1/3) - (Real.cos (2/3 * x))^(1/5)

theorem smallest_positive_period_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = 12 * Real.pi := by sorry

-- Part 2
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (3 * x) + 3 * abs (Real.sin (4 * x))

theorem smallest_positive_period_g :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, g (x + T) = g x) ∧
  (∀ T' > 0, (∀ x, g (x + T') = g x) → T ≤ T') ∧
  T = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_f_smallest_positive_period_g_l174_17422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_energy_ratio_8_6_l174_17481

/-- The energy released by an earthquake given its magnitude -/
noncomputable def earthquake_energy (k : ℝ) (n : ℝ) : ℝ := k * (10 : ℝ) ^ (1.5 * n)

/-- The ratio of energy released by earthquakes of different magnitudes -/
noncomputable def energy_ratio (k : ℝ) (n₁ n₂ : ℝ) : ℝ :=
  earthquake_energy k n₁ / earthquake_energy k n₂

theorem earthquake_energy_ratio_8_6 (k : ℝ) (h : k > 0) :
  energy_ratio k 8 6 = 1000 := by
  -- Expand the definition of energy_ratio
  unfold energy_ratio
  -- Expand the definition of earthquake_energy
  unfold earthquake_energy
  -- Simplify the expression
  simp [pow_sub, pow_mul]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_energy_ratio_8_6_l174_17481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_terms_sum_l174_17446

noncomputable def sequence_sum (n : ℕ) (first : ℝ) (diff : ℝ) : ℝ :=
  n * (2 * first + (n - 1) * diff) / 2

theorem odd_terms_sum (first : ℝ) : 
  sequence_sum 2500 first 1 = 7000 →
  sequence_sum 1250 first 2 = 2875 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_terms_sum_l174_17446
