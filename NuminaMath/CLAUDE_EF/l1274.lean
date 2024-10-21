import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1274_127479

noncomputable section

-- Define the piecewise function
def f (a b c : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x^3 + x^2 + b*x + c
  else a * Real.log x

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  let f := f a 0 0
  (f (-1) = 2) →
  (∃ k, k * (Real.sqrt 26) = 1 ∧ k * ((deriv f) (-1)) = -5) →
  (∀ x ∈ Set.Icc (-1) (Real.exp 1), f x ≤ max a 2) ∧
  (∃ x ∈ Set.Icc (-1) (Real.exp 1), f x = max a 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1274_127479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_plus_commutative_circle_plus_not_associative_l1274_127444

/-- The custom operation ⊕ -/
noncomputable def circle_plus (x y : ℝ) : ℝ := (2 * x * y) / (x + y)

/-- Theorem stating that circle_plus is commutative for positive real numbers -/
theorem circle_plus_commutative (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  circle_plus x y = circle_plus y x := by
  -- Proof goes here
  sorry

/-- Theorem stating that circle_plus is not associative for positive real numbers -/
theorem circle_plus_not_associative :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  circle_plus (circle_plus x y) z ≠ circle_plus x (circle_plus y z) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_plus_commutative_circle_plus_not_associative_l1274_127444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1274_127441

/-- The time (in seconds) for a train to pass a man moving in the opposite direction -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let relative_speed := train_speed + man_speed
  let relative_speed_ms := relative_speed * (5 / 18)
  train_length / relative_speed_ms

/-- Theorem stating that the time for a 165-meter long train moving at 60 kmph to pass
    a man moving at 6 kmph in the opposite direction is approximately 9 seconds -/
theorem train_passing_time_approx :
  ∃ ε > 0, |train_passing_time 165 60 6 - 9| < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_passing_time 165 60 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1274_127441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1274_127465

/-- A circle C with radius 1 and center symmetric to (1,0) with respect to y=x has equation x^2 + (y-1)^2 = 1 -/
theorem circle_equation (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) :
  (∀ p ∈ C, dist p center = 1) →  -- radius is 1
  center = (0, 1) →  -- center is symmetric to (1,0) w.r.t. y=x
  (∀ p : ℝ × ℝ, p ∈ C ↔ p.1^2 + (p.2 - 1)^2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1274_127465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_example_monomial_properties_l1274_127449

/-- Definition of a monomial -/
structure Monomial where
  coefficient : ℚ
  vars : List (Char × ℕ)

/-- The given monomial -/
def example_monomial : Monomial :=
  { coefficient := -1/3,
    vars := [('a', 3), ('b', 1), ('c', 2)] }

/-- The coefficient of a monomial -/
def coefficient (m : Monomial) : ℚ := m.coefficient

/-- The degree of a monomial -/
def degree (m : Monomial) : ℕ := m.vars.foldl (fun acc (_, exp) => acc + exp) 0

/-- Theorem: The coefficient and degree of the example monomial -/
theorem example_monomial_properties :
  (coefficient example_monomial = -1/3) ∧ (degree example_monomial = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_example_monomial_properties_l1274_127449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_decomposition_l1274_127499

theorem tan_22_5_decomposition :
  ∃ (e f g h : ℕ),
    (Real.tan (22.5 * Real.pi / 180) = Real.sqrt (e : ℝ) + (f : ℝ) - Real.sqrt (g : ℝ) - (h : ℝ)) ∧
    (e ≥ f) ∧ (f ≥ g) ∧ (g ≥ h) ∧
    (e > 0) ∧ (f > 0) ∧ (g > 0) ∧ (h > 0) ∧
    (e + f + g + h = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_decomposition_l1274_127499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_result_l1274_127496

/-- The continued fraction representation of x --/
noncomputable def x : ℝ := Real.sqrt 3 + 1

/-- The function that calculates 1 / ((x+2)(x-3)) --/
noncomputable def f (x : ℝ) : ℝ := 1 / ((x + 2) * (x - 3))

/-- The theorem stating the result of the calculation --/
theorem calculation_result : ∃ (A B C : ℝ), 
  f x = (A + Real.sqrt B) / C ∧ 
  |A| + |B| + |C| = 72 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_result_l1274_127496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_iff_integer_or_fraction_l1274_127448

/-- A number is rational if and only if it is either an integer or a fraction (non-integer rational number) -/
theorem rational_iff_integer_or_fraction : 
  ∀ x : ℚ, (∃ n : ℤ, (x : ℚ) = n) ∨ (∃ p q : ℤ, q ≠ 0 ∧ x = ↑p / ↑q ∧ ¬∃ m : ℤ, (x : ℚ) = m) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_iff_integer_or_fraction_l1274_127448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_y_intercept_range_eq_l1274_127458

/-- The y-intercept of the tangent line to y = e^x at point x₀ -/
noncomputable def tangent_y_intercept (x₀ : ℝ) : ℝ := Real.exp x₀ * (1 - x₀)

/-- The range of y-intercepts of tangent lines to y = e^x -/
def tangent_y_intercept_range : Set ℝ := Set.range tangent_y_intercept

theorem tangent_y_intercept_range_eq : tangent_y_intercept_range = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_y_intercept_range_eq_l1274_127458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drops_equality_drinking_pattern_result_l1274_127429

/-- Represents the days of the month with sunny days -/
def sunny_days : Fin 15 → ℕ := sorry

/-- The total number of days in the month -/
def total_days : ℕ := 30

/-- Calculates the total drops drunk by Andrei Stepanovich -/
def andrei_drops (a : Fin 15 → ℕ) : ℕ :=
  15 * 31 - (Finset.sum Finset.univ (λ i => a i))

/-- Calculates the total drops drunk by Ivan Petrovich -/
def ivan_drops (a : Fin 15 → ℕ) : ℕ :=
  15 * 31 - (Finset.sum Finset.univ (λ i => a i))

/-- The main theorem stating that both Andrei and Ivan drink the same amount -/
theorem drops_equality (a : Fin 15 → ℕ) : 
  andrei_drops a = ivan_drops a := by
  -- The proof is trivial since the definitions are identical
  rfl

/-- Theorem stating that the drinking pattern results in the given formula -/
theorem drinking_pattern_result (a : Fin 15 → ℕ) :
  (Finset.sum Finset.univ (λ i : Fin 14 => (i.val + 1) * (a (i.succ) - a i))) +
  15 * (total_days - a 14 + 1) =
  15 * 31 - (Finset.sum Finset.univ (λ i => a i)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_drops_equality_drinking_pattern_result_l1274_127429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_a_eq_neg_one_l1274_127418

-- Define the function f(x, a)
noncomputable def f (x a : ℝ) : ℝ := (2019 : ℝ)^(|x - 1|) + a * Real.sin (x - 1) + a

-- State the theorem
theorem unique_solution_iff_a_eq_neg_one :
  (∃! x, f x a = 0) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_a_eq_neg_one_l1274_127418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_comparison_l1274_127489

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  directrix : ℝ

/-- Chord of a parabola passing through its focus -/
structure Chord (para : Parabola) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ
  passes_through_focus : endpoint1.1 < para.focus.1 ∧ para.focus.1 < endpoint2.1

/-- Surface area obtained by rotating a chord around the directrix -/
noncomputable def surface_area_rotation (para : Parabola) (chord : Chord para) : ℝ :=
  Real.pi * (chord.endpoint1.1 - para.directrix + chord.endpoint2.1 - para.directrix)^2

/-- Surface area of a sphere with diameter equal to the projection of the chord on the directrix -/
noncomputable def surface_area_sphere (para : Parabola) (chord : Chord para) : ℝ :=
  4 * Real.pi * (chord.endpoint1.1 - para.directrix) * (chord.endpoint2.1 - para.directrix)

/-- Theorem stating that the surface area of rotation is always greater than or equal to 
    the surface area of the sphere for any chord passing through the focus of a parabola -/
theorem surface_area_comparison (para : Parabola) (chord : Chord para) :
  surface_area_rotation para chord ≥ surface_area_sphere para chord := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_comparison_l1274_127489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_eight_factors_l1274_127485

def has_exactly_eight_factors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 8

theorem smallest_with_eight_factors :
  ∀ m : ℕ, m < 24 → ¬(has_exactly_eight_factors m) ∧ has_exactly_eight_factors 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_eight_factors_l1274_127485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_expression_simplification_l1274_127445

theorem root_expression_simplification :
  (5 : ℝ) ^ (1/6 : ℝ) * (5 : ℝ) ^ (1/2 : ℝ) / (5 : ℝ) ^ (1/3 : ℝ) = (5 : ℝ) ^ (1/6 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_expression_simplification_l1274_127445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_time_l1274_127428

/-- Represents the time it takes for a person to complete the work alone -/
structure WorkTime where
  days : ℚ
  days_positive : days > 0

/-- Represents the portion of work completed by a person in one day -/
def work_rate (wt : WorkTime) : ℚ := 1 / wt.days

/-- The problem setup -/
structure WorkProblem where
  a_time : WorkTime
  c_time : WorkTime
  total_time : ℚ
  a_work_days : ℚ
  c_work_days : ℚ
  a_time_is_20 : a_time.days = 20
  c_time_is_10 : c_time.days = 10
  total_time_is_15 : total_time = 15
  a_work_days_is_2 : a_work_days = 2
  c_work_days_is_4 : c_work_days = 4

/-- The theorem to prove -/
theorem b_work_time (wp : WorkProblem) : 
  ∃ (b_time : WorkTime), 
    work_rate wp.a_time * wp.a_work_days + 
    work_rate wp.c_time * wp.c_work_days + 
    work_rate b_time * wp.total_time = 1 ∧ 
    b_time.days = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_time_l1274_127428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BEC_l1274_127487

-- Define the rectangle ABCD
def Rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ A.2 = B.2 ∧ B.1 = C.1 ∧ D.2 = C.2

-- Define point E on line AB
def PointE (A B E : ℝ × ℝ) : Prop :=
  E.2 = A.2 ∧ E.1 - A.1 = 20

-- Calculate the area of a triangle given three points
noncomputable def TriangleArea (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

theorem area_of_triangle_BEC (A B C D E : ℝ × ℝ) :
  Rectangle A B C D →
  A = (0, 0) →
  B = (30, 0) →
  C = (30, 50) →
  D = (0, 50) →
  PointE A B E →
  TriangleArea B E C = 1000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BEC_l1274_127487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1274_127416

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

/-- Theorem: The area of a trapezium with parallel sides of lengths 20 cm and 18 cm, 
    and a distance of 15 cm between them, is equal to 285 square centimeters. -/
theorem trapezium_area_example : trapeziumArea 20 18 15 = 285 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the expression
  simp [mul_add, mul_assoc]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1274_127416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_eight_l1274_127468

/-- The production volume in ten thousand pieces. -/
def x : Type := ℝ

/-- The total cost function in ten thousand yuan. -/
noncomputable def c (x : ℝ) : ℝ := 100 + 13 * x

/-- The selling price function in yuan per piece. -/
noncomputable def p (x : ℝ) : ℝ := 800 / (x + 2) - 3

/-- The total profit function in ten thousand yuan. -/
noncomputable def f (x : ℝ) : ℝ := x * (p x) / 10000 - c x

/-- The theorem stating that the production volume maximizing total profit is 8 ten thousand pieces. -/
theorem max_profit_at_eight :
  ∃ (x_max : ℝ), x_max = 8 ∧ ∀ (x : ℝ), f x ≤ f x_max := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_eight_l1274_127468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_technicians_l1274_127422

/-- Prove that the number of technicians is 7 in a workshop with the following conditions:
  * There are 14 workers in total
  * The average salary of all workers is 9000
  * The average salary of technicians is 12000
  * The average salary of non-technicians is 6000
-/
theorem number_of_technicians (total_workers : ℕ) (avg_salary : ℕ) 
  (technician_salary : ℕ) (other_salary : ℕ) : ℕ :=
by
  have h1 : total_workers = 14 := by sorry
  have h2 : avg_salary = 9000 := by sorry
  have h3 : technician_salary = 12000 := by sorry
  have h4 : other_salary = 6000 := by sorry
  
  -- The actual proof steps would go here
  sorry

-- Remove the #eval statement as it's causing issues with compilation
-- #eval number_of_technicians 14 9000 12000 6000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_technicians_l1274_127422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1274_127490

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 + m

-- Define the theorem
theorem f_properties (m : ℝ) (A B C : ℝ) (a b c : ℝ) :
  f m (π / 12) = 0 →
  c * Real.cos B + b * Real.cos C = 2 * a * Real.cos B →
  0 < A ∧ A < 2 * π / 3 →
  m = 1 / 2 ∧ -1 / 2 < f m A ∧ f m A ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1274_127490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_horizontal_line_slope_angle_y_eq_1_l1274_127461

/-- Slope angle of a line -/
def SlopeAngle (f : ℝ → ℝ) : ℝ := sorry

/-- The slope angle of a horizontal line is 0 degrees. -/
theorem slope_angle_horizontal_line :
  ∀ (c : ℝ), SlopeAngle (λ x : ℝ => c) = 0 := by
  sorry

/-- The line y = 1 is a horizontal line. -/
def line_y_eq_1 : ℝ → ℝ := λ _ => 1

/-- The slope angle of the line y = 1 is 0 degrees. -/
theorem slope_angle_y_eq_1 : SlopeAngle line_y_eq_1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_horizontal_line_slope_angle_y_eq_1_l1274_127461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_min_value_is_six_exact_min_value_l1274_127420

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 3 * y = 2) :
  ∀ a b : ℝ, a + 3 * b = 2 → (3 : ℝ)^x + (27 : ℝ)^y ≤ (3 : ℝ)^a + (27 : ℝ)^b :=
by sorry

theorem min_value_is_six (x y : ℝ) (h : x + 3 * y = 2) :
  (3 : ℝ)^x + (27 : ℝ)^y ≥ 6 :=
by sorry

theorem exact_min_value (x y : ℝ) (h : x + 3 * y = 2) :
  ∃ a b : ℝ, a + 3 * b = 2 ∧ (3 : ℝ)^a + (27 : ℝ)^b = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_min_value_is_six_exact_min_value_l1274_127420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l1274_127440

-- Define the function f
def f : Set ℝ → Prop := fun x ↦ x ⊆ Set.Icc (-1) 1

-- Define the composite function g
def g (f : Set ℝ → Prop) : Set ℝ := {x : ℝ | f {x^2 - 1}}

-- Theorem stating the domain of the composite function
theorem domain_of_composite_function :
  g f = Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l1274_127440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_condition_l1274_127469

/-- A linear equation in two variables x and y has the form ax + by + c = 0,
    where a, b, and c are constants and at least one of a or b is non-zero. -/
def IsLinearInTwoVars (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y = a * x + b * y + c

/-- The equation x^(m-1) - 2y^(3+n) = 5 -/
noncomputable def EquationF (m n : ℝ) (x y : ℝ) : ℝ := 
  x^(m-1) - 2*(y^(3+n)) - 5

theorem linear_equation_condition (m n : ℝ) :
  IsLinearInTwoVars (EquationF m n) → m = 2 ∧ n = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_condition_l1274_127469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonia_moles_formed_l1274_127409

/-- Represents the number of moles of a chemical substance -/
structure Moles where
  value : ℝ

/-- Represents the molar ratio between two substances -/
structure MolarRatio where
  value : ℝ

/-- The number of moles of Ammonium chloride used -/
def nh4cl_moles : Moles := ⟨3⟩

/-- The molar ratio of Ammonium chloride to Ammonia -/
def nh4cl_nh3_ratio : MolarRatio := ⟨1⟩

/-- The total moles of Ammonia formed -/
def total_nh3_moles : Moles := ⟨3⟩

/-- Multiplication of Moles and MolarRatio -/
def mul_moles_ratio (m : Moles) (r : MolarRatio) : Moles := ⟨m.value * r.value⟩

/-- Theorem stating that the number of moles of Ammonia formed is equal to 3 -/
theorem ammonia_moles_formed (m : Moles) 
  (h1 : m = mul_moles_ratio nh4cl_moles nh4cl_nh3_ratio) 
  (h2 : m = total_nh3_moles) : 
  m.value = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonia_moles_formed_l1274_127409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l1274_127483

theorem fraction_equality (a : ℕ) (h1 : a > 0) : (a : ℚ) / ((a : ℚ) + 36) = 88 / 100 → a = 264 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l1274_127483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_paving_cost_approx_l1274_127412

-- Define constants
def rectangle_length : ℝ := 6.5
def rectangle_width : ℝ := 4.75
def semicircle_radius : ℝ := 3.25
def marble_cost_per_sqm : ℝ := 800
def limestone_cost_per_sqm : ℝ := 950

-- Define functions for area calculations
def rectangle_area (l w : ℝ) : ℝ := l * w

noncomputable def semicircle_area (r : ℝ) : ℝ := 0.5 * Real.pi * r^2

-- Define function for total cost calculation
def total_cost (rect_area semi_area : ℝ) (marble_cost limestone_cost : ℝ) : ℝ :=
  rect_area * marble_cost + semi_area * limestone_cost

-- Theorem statement
theorem total_paving_cost_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧
  abs (total_cost
    (rectangle_area rectangle_length rectangle_width)
    (semicircle_area semicircle_radius)
    marble_cost_per_sqm
    limestone_cost_per_sqm - 40488.05) < ε := by
  sorry

#eval rectangle_area rectangle_length rectangle_width

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_paving_cost_approx_l1274_127412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1274_127413

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x y : ℝ, x^2/2 + y^2/m = 1 → m > 2

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (4/3*x^3 - 2*m*x^2 + (4*m - 3)*x - m) < (4/3*y^3 - 2*m*y^2 + (4*m - 3)*y - m)

-- Define the range of m
def m_range (m : ℝ) : Prop := 1 ≤ m ∧ m ≤ 2

-- State the theorem
theorem range_of_m (m : ℝ) : (¬p m) ∧ q m → m_range m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1274_127413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_fraction_l1274_127401

theorem modulus_of_complex_fraction :
  Complex.abs ((1 + Complex.I)^2 / (1 - Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_fraction_l1274_127401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_finish_time_l1274_127436

-- Define the cities
inductive City : Type
| Appleminster : City
| Boniham : City

-- Define a person
structure Person where
  startCity : City
  startTime : ℕ  -- in minutes since midnight
  speed : ℚ      -- in km/hr

-- Define the problem setup
def problem : Prop := ∃ (distance : ℚ) (personA personB : Person),
  personA.startCity = City.Appleminster ∧
  personA.startTime = 12 * 60 ∧  -- 12:00 PM
  personB.startCity = City.Boniham ∧
  personB.startTime = 14 * 60 ∧  -- 2:00 PM
  (personA.startTime : ℚ) + (distance / personA.speed) * 60 = 
    (personB.startTime : ℚ) + (distance / personB.speed) * 60 ∧
  (personA.startTime : ℚ) + (distance / personA.speed) * 60 = 16 * 60 + 55 ∧  -- 4:55 PM
  ∃ (finishTime : ℕ),
    (finishTime : ℚ) = (personA.startTime : ℚ) + (distance / personA.speed * 60) ∧
    (finishTime : ℚ) = (personB.startTime : ℚ) + (distance / personB.speed * 60) ∧
    finishTime = 19 * 60  -- 7:00 PM

-- Theorem statement
theorem journey_finish_time : problem := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_finish_time_l1274_127436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_eight_l1274_127470

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line
def line (x y : ℝ) : Prop := y = -x + 1

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  line A.1 A.2 ∧ line B.1 B.2

-- Define the chord length
noncomputable def chord_length (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem chord_length_is_eight :
  ∀ A B : ℝ × ℝ,
  intersection_points A B →
  chord_length A B = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_eight_l1274_127470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parameter_l1274_127417

-- Define the line equation
def line (x y : ℝ) (a : ℝ) : Prop := 3 * x + 4 * y + a = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 - 2*x + y^2 = 0

-- Define the tangent condition
def is_tangent (a : ℝ) : Prop :=
  ∃ (x y : ℝ), line x y a ∧ circle_eq x y ∧
  ∀ (x' y' : ℝ), line x' y' a ∧ circle_eq x' y' → (x = x' ∧ y = y')

-- Theorem statement
theorem tangent_line_parameter :
  ∀ a : ℝ, is_tangent a → (a = 2 ∨ a = -8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parameter_l1274_127417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_to_sixth_l1274_127459

theorem cube_root_of_eight_to_sixth : (8 : ℝ) ^ ((1/3 : ℝ) * 6) = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_to_sixth_l1274_127459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_arcs_area_theorem_l1274_127476

/-- The area of the region bounded by 8 reflected arcs of a circle, where each arc is determined by a side of a regular octagon with side length 1 inscribed in the circle. -/
noncomputable def reflectedArcsArea : ℝ := 4 + 2 * Real.sqrt 2 - Real.pi * (6 + 4 * Real.sqrt 2)

/-- A regular octagon with side length 1 inscribed in a circle. -/
structure InscribedOctagon where
  sideLength : ℝ
  isSideLength1 : sideLength = 1

/-- The theorem stating that the area of the region bounded by 8 reflected arcs of a circle,
    where each arc is determined by a side of a regular octagon with side length 1 inscribed
    in the circle, is equal to 4 + 2√2 - π(6 + 4√2). -/
theorem reflected_arcs_area_theorem (octagon : InscribedOctagon) :
  reflectedArcsArea = 4 + 2 * Real.sqrt 2 - Real.pi * (6 + 4 * Real.sqrt 2) := by
  sorry

#check reflected_arcs_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_arcs_area_theorem_l1274_127476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_needed_for_journey_l1274_127455

/-- Represents the fuel efficiency of a car in different road conditions -/
structure FuelEfficiency where
  highway : ℚ
  city : ℚ
  uphill : ℚ

/-- Represents the distances traveled in different road conditions -/
structure TravelDistance where
  total : ℚ
  highway : ℚ
  uphill : ℚ

/-- Calculates the total gasoline needed for a journey -/
def calculateGasolineNeeded (efficiency : FuelEfficiency) (distance : TravelDistance) : ℚ :=
  let cityDistance := distance.total - (distance.highway + distance.uphill)
  (distance.highway / efficiency.highway) +
  (distance.uphill / efficiency.uphill) +
  (cityDistance / efficiency.city)

theorem gasoline_needed_for_journey
  (efficiency : FuelEfficiency)
  (distance : TravelDistance)
  (h1 : efficiency.highway = 40)
  (h2 : efficiency.city = 30)
  (h3 : efficiency.uphill = 20)
  (h4 : distance.total = 160)
  (h5 : distance.highway = 70)
  (h6 : distance.uphill = 60) :
  calculateGasolineNeeded efficiency distance = 23/4 := by
  sorry

#eval (23 : ℚ) / 4  -- To show that 23/4 is indeed equal to 5.75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_needed_for_journey_l1274_127455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l1274_127447

open Real

theorem max_value_of_expression (x y : ℝ) (hx : x ∈ Set.Ioo 0 (π/2)) (hy : y ∈ Set.Ioo 0 (π/2)) :
  (Real.sqrt (cos x * cos y)) / (Real.sqrt (tan x⁻¹) + Real.sqrt (tan y⁻¹)) ≤ Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l1274_127447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_equals_e_l1274_127405

open Real

/-- The function f(x) = ln x + 2a/x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + 2 * a / x

theorem min_value_implies_a_equals_e (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (exp 1), f a x ≥ 3) ∧
  (∃ x ∈ Set.Icc 1 (exp 1), f a x = 3) →
  a = exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_equals_e_l1274_127405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_rectangles_l1274_127472

/-- Represents a rectangle on a chessboard --/
structure Rectangle where
  width : Nat
  height : Nat
  deriving Repr

/-- Represents an 8x8 chessboard --/
def Chessboard : Nat := 8

/-- Checks if a rectangle fits within the chessboard --/
def fits_on_board (r : Rectangle) : Prop :=
  r.width ≤ Chessboard ∧ r.height ≤ Chessboard

/-- Defines when two rectangles are considered distinct --/
def distinct (r1 r2 : Rectangle) : Prop :=
  (r1.width ≠ r2.width ∨ r1.height ≠ r2.height) ∧
  (r1.width ≠ r2.height ∨ r1.height ≠ r2.width)

/-- The theorem to be proved --/
theorem max_distinct_rectangles :
  ∃ (rectangles : List Rectangle),
    (∀ r, r ∈ rectangles → fits_on_board r) ∧
    (∀ r1 r2, r1 ∈ rectangles → r2 ∈ rectangles → r1 ≠ r2 → distinct r1 r2) ∧
    rectangles.length = 12 ∧
    (∀ rectangles' : List Rectangle,
      (∀ r, r ∈ rectangles' → fits_on_board r) →
      (∀ r1 r2, r1 ∈ rectangles' → r2 ∈ rectangles' → r1 ≠ r2 → distinct r1 r2) →
      rectangles'.length ≤ 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_rectangles_l1274_127472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_36_equals_4_11_l1274_127463

noncomputable def repeating_decimal_36 : ℚ := 0.3636363636363636

theorem repeating_decimal_36_equals_4_11 : repeating_decimal_36 = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_36_equals_4_11_l1274_127463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_and_solution_set_l1274_127495

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x / 2 else x^2

theorem f_composition_and_solution_set :
  (f (f (-1)) = 1/2) ∧
  (∀ x : ℝ, f (f x) ≥ 1 ↔ x ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ici 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_and_solution_set_l1274_127495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_congruent_triangle_exists_l1274_127462

/-- A triangle represented by three points in a plane -/
structure Triangle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] where
  A : α
  B : α
  C : α

/-- Predicate to check if a triangle is inscribed in another triangle -/
def IsInscribed {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (inner outer : Triangle α) : Prop :=
  sorry

/-- Congruence relation between two triangles -/
def Congruent {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t1 t2 : Triangle α) : Prop :=
  sorry

/-- Theorem stating that for any two triangles, there exists an inscribed triangle congruent to the second -/
theorem inscribed_congruent_triangle_exists {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (ABC GHI : Triangle α) : 
  ∃ (DEF : Triangle α), IsInscribed DEF ABC ∧ Congruent DEF GHI :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_congruent_triangle_exists_l1274_127462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1274_127493

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => λ x => deriv f x
  | n + 1 => λ x => deriv (f_n n) x

theorem f_properties :
  (∃ x : ℝ, deriv f x = 0) ∧
  (∀ n : ℕ, ∀ x : ℝ, f_n n x = x * Real.exp x + (n + 1) * Real.exp x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1274_127493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ABO_min_dist_product_l1274_127456

-- Define the line l: kx - y + 1 + 2k = 0
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0

-- Define points A and B
noncomputable def point_A (k : ℝ) : ℝ × ℝ := (-2 - 1/k, 0)
noncomputable def point_B (k : ℝ) : ℝ × ℝ := (0, 1 + 2*k)

-- Define the area of triangle ABO
noncomputable def area_ABO (k : ℝ) : ℝ := (1/2) * (1 + 2*k) * (2 + 1/k)

-- Define the fixed point M
def point_M : ℝ × ℝ := (-2, 1)

-- Define the product of distances MA and MB
noncomputable def dist_product (k : ℝ) : ℝ := 2/k + 2*k

-- Theorem 1: Minimum area of triangle ABO
theorem min_area_ABO :
  ∃ (k : ℝ), k > 0 ∧ 
  (∀ (k' : ℝ), k' > 0 → area_ABO k ≤ area_ABO k') ∧
  area_ABO k = 4 ∧
  line_l k 1 2 := by sorry

-- Theorem 2: Minimum product of distances MA and MB
theorem min_dist_product :
  ∃ (k : ℝ), k > 0 ∧
  (∀ (k' : ℝ), k' > 0 → dist_product k ≤ dist_product k') ∧
  dist_product k = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ABO_min_dist_product_l1274_127456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equivalence_transitivity_l1274_127454

theorem set_equivalence_transitivity 
  {α : Type*} 
  (A B A₁ B₁ : Set α) 
  (h1 : A ⊇ A₁) 
  (h2 : B ⊇ B₁) 
  (h3 : A ≃ B₁) 
  (h4 : B ≃ A₁) : 
  A ≃ B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equivalence_transitivity_l1274_127454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_d_value_l1274_127411

theorem factor_implies_d_value (d : ℤ) : 
  (∃ q : Polynomial ℚ, (8 * X^3 + 23 * X^2 + d * X + 45 : Polynomial ℚ) = (2 * X + 5) * q) → d = 163 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_d_value_l1274_127411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_octagon_hexagon_is_105_l1274_127442

/-- The measure of an exterior angle between a regular octagon and a regular hexagon sharing a side -/
def exterior_angle_octagon_hexagon : ℚ :=
  let octagon_interior_angle : ℚ := 135
  let hexagon_interior_angle : ℚ := 120
  360 - octagon_interior_angle - hexagon_interior_angle

/-- Proof that the exterior angle between a regular octagon and a regular hexagon sharing a side is 105° -/
theorem exterior_angle_octagon_hexagon_is_105 :
  exterior_angle_octagon_hexagon = 105 := by
  unfold exterior_angle_octagon_hexagon
  norm_num

#eval exterior_angle_octagon_hexagon -- Should output 105

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_octagon_hexagon_is_105_l1274_127442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1274_127484

noncomputable def f (x : ℝ) := x^3 - (1/2)^(x-2)

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1274_127484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_reassembly_l1274_127494

-- Define a rectangle
structure Rectangle where
  width : ℕ
  height : ℕ

-- Define a colored area within a rectangle
structure ColoredArea where
  x : ℕ
  y : ℕ
  width : ℕ
  height : ℕ
  color : ℕ

-- Define a cut
structure Cut where
  direction : Bool  -- True for vertical, False for horizontal
  position : ℕ

-- Define the problem
def reassemblable_rectangle (r : Rectangle) (areas : List ColoredArea) (cuts : List Cut) : Prop :=
  ∃ (new_areas : List ColoredArea),
    (new_areas.length = areas.length) ∧
    ((new_areas.map (λ a ↦ a.width * a.height)).sum = (areas.map (λ a ↦ a.width * a.height)).sum) ∧
    (∀ color, (new_areas.filter (λ a ↦ a.color = color)).length = (areas.filter (λ a ↦ a.color = color)).length) ∧
    (∀ a ∈ new_areas, a.x + a.width ≤ r.width ∧ a.y + a.height ≤ r.height)

-- Theorem statement
theorem rectangle_reassembly :
  ∃ (areas : List ColoredArea) (cuts : List Cut),
    let r : Rectangle := { width := 12, height := 9 }
    cuts.length = 3 ∧
    reassemblable_rectangle r areas cuts := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_reassembly_l1274_127494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_l1274_127488

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_composition (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  (f (1 / x) = x / (1 - x)) → (f x = 1 / (x - 1)) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_l1274_127488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_to_plane_implies_perpendicular_to_line_in_plane_l1274_127478

/-- A line in 3D space -/
structure Line where

/-- A plane in 3D space -/
structure Plane where

/-- Predicate for a line being perpendicular to a plane -/
def perpendicular_line_plane (l : Line) (α : Plane) : Prop := sorry

/-- Predicate for a line being contained in a plane -/
def line_in_plane (m : Line) (α : Plane) : Prop := sorry

/-- Predicate for two lines being perpendicular -/
def perpendicular_lines (l m : Line) : Prop := sorry

/-- Theorem: If a line is perpendicular to a plane, and another line is contained in that plane,
    then the two lines are perpendicular -/
theorem perpendicular_line_to_plane_implies_perpendicular_to_line_in_plane
  (l m : Line) (α : Plane)
  (h1 : perpendicular_line_plane l α)
  (h2 : line_in_plane m α) :
  perpendicular_lines l m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_to_plane_implies_perpendicular_to_line_in_plane_l1274_127478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_b_cost_l1274_127474

-- Define the cylinders and their properties
def Cylinder (radius height : ℝ) := { r : ℝ // r = radius } × { h : ℝ // h = height }

-- Define the volumes of the cylinders
noncomputable def volume (c : Cylinder r h) : ℝ := Real.pi * r^2 * h

-- Define the cost to fill a cylinder
def cost_to_fill (c : Cylinder r h) (full_cost : ℝ) : ℝ := full_cost

-- Theorem statement
theorem half_b_cost 
  (b : Cylinder r h) 
  (v : Cylinder (2*r) (h/2)) 
  (hv : cost_to_fill v 16 = 16) : 
  cost_to_fill b 16 / 4 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_b_cost_l1274_127474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_properties_l1274_127423

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  hσ_pos : σ > 0

/-- The probability density function for a normal distribution -/
noncomputable def normalPDF (X : NormalRV) (x : ℝ) : ℝ :=
  (1 / (X.σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - X.μ) / X.σ)^2)

/-- The cumulative distribution function for a normal distribution -/
noncomputable def normalCDF (X : NormalRV) (x : ℝ) : ℝ :=
  ∫ y in Set.Iio x, normalPDF X y

/-- The probability of a normal random variable falling within an interval -/
noncomputable def normalProb (X : NormalRV) (a b : ℝ) : ℝ :=
  normalCDF X b - normalCDF X a

theorem normal_distribution_properties (ξ : NormalRV) 
  (h1 : ξ.μ = 4)
  (h2 : normalProb ξ 2 6 = 0.6826) :
  ξ.σ = 2 ∧ normalProb ξ (-2) 6 = 0.8400 := by
  sorry

#check normal_distribution_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_properties_l1274_127423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_area_l1274_127475

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a square pyramid -/
structure SquarePyramid where
  baseSideLength : ℝ
  height : ℝ

/-- Function to calculate the area of the intersection triangle -/
noncomputable def area_of_intersection_triangle (pyramid : SquarePyramid) (v p q r : Point3D) : ℝ :=
  sorry -- Placeholder for the actual area calculation

/-- Theorem stating the area of the intersection triangle -/
theorem intersection_triangle_area 
  (pyramid : SquarePyramid)
  (v : Point3D)
  (p : Point3D)
  (q : Point3D)
  (r : Point3D)
  (h_base : pyramid.baseSideLength = 30)
  (h_height : pyramid.height = 40)
  (h_v : v = ⟨0, 0, pyramid.height⟩)
  (h_p : p.z = 10 ∧ (p.x - v.x)^2 + (p.y - v.y)^2 + (p.z - v.z)^2 = 10^2)
  (h_q : q.z = 20 ∧ (q.x - v.x)^2 + (q.y - v.y)^2 + (q.z - v.z)^2 = 20^2)
  (h_r : r.z = 15 ∧ (r.x - v.x)^2 + (r.y - v.y)^2 + (r.z - v.z)^2 = 15^2)
  : ∃ (area : ℝ), area_of_intersection_triangle pyramid v p q r = area :=
by
  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_area_l1274_127475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_card_spending_ratio_l1274_127408

theorem gift_card_spending_ratio :
  ∀ (initial_value : ℕ) (remaining : ℕ),
    initial_value = 200 →
    remaining = 75 →
    let monday_spending := initial_value / 2
    let after_monday := initial_value - monday_spending
    let tuesday_spending := after_monday - remaining
    (tuesday_spending : ℚ) / after_monday = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_card_spending_ratio_l1274_127408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_AF_BF_eq_nine_l1274_127482

/-- Parabola C defined by y^2 = 2x -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 2 * p.1}

/-- Focus F of parabola C -/
noncomputable def F : ℝ × ℝ := (1/2, 0)

/-- Point P -/
def P : ℝ × ℝ := (4, 1)

/-- Line l passing through P -/
def l : Set (ℝ × ℝ) := sorry

/-- A and B are intersection points of l and C -/
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

/-- P is the midpoint of AB -/
axiom P_midpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- Theorem: The sum of distances |AF| + |BF| is 9 -/
theorem sum_distances_AF_BF_eq_nine :
  Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) +
  Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_AF_BF_eq_nine_l1274_127482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_function_satisfying_equation_l1274_127424

-- Define the positive rational numbers
def PositiveRationals := {q : ℚ // q > 0}

-- Define the function type
def FunctionType := PositiveRationals → PositiveRationals

-- Define multiplication for PositiveRationals
instance : Mul PositiveRationals where
  mul a b := ⟨a.val * b.val, by
    apply mul_pos
    exact a.property
    exact b.property⟩

-- Define division for PositiveRationals
instance : Div PositiveRationals where
  div a b := ⟨a.val / b.val, by
    apply div_pos
    exact a.property
    exact b.property⟩

-- State the theorem
theorem exists_function_satisfying_equation :
  ∃ (f : FunctionType), ∀ (x y : PositiveRationals), 
    f (x * (f y)) = (f x) / y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_function_satisfying_equation_l1274_127424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_border_length_calculation_l1274_127477

-- Define constants
def π : ℚ := 22 / 7
def circleArea : ℚ := 100
def extraBorder : ℚ := 3

-- Define the theorem
theorem border_length_calculation :
  let r : ℚ := (circleArea / π).sqrt
  let circumference : ℚ := 2 * π * r
  let borderLength : ℚ := circumference + extraBorder
  ∃ (ε : ℚ), abs (borderLength - 38.4) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_border_length_calculation_l1274_127477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_value_l1274_127407

/-- Sequence definition for a_n -/
def a (k : ℤ) : ℕ → ℤ
| 0 => 0
| 1 => 1
| (n + 2) => k * a k (n + 1) + a k n

/-- Main theorem statement -/
theorem unique_k_value :
  ∀ k : ℤ, k > 1 →
  (∃ (l m p q : ℕ), l ≠ m ∧ p > 0 ∧ q > 0 ∧ a k l + k * a k p = a k m + k * a k q) ↔ k = 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_value_l1274_127407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_probability_l1274_127437

/-- The probability of obtaining exactly n points in a coin toss game -/
noncomputable def probability_n_points (n : ℕ) : ℝ :=
  2/3 + 1/3 * (-1/2)^n

/-- The coin toss game where heads awards 1 point and tails awards 2 points -/
theorem coin_toss_probability (n : ℕ) :
  let p := probability_n_points
  -- Initial probabilities
  (p 1 = 1/2) ∧
  (p 2 = 3/4) ∧
  -- Recurrence relation for n ≥ 3
  (∀ k ≥ 3, p k = 1/2 * p (k-1) + 1/2 * p (k-2)) →
  -- The probability of obtaining exactly n points
  p n = 2/3 + 1/3 * (-1/2)^n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_probability_l1274_127437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_sums_l1274_127453

/-- Represents the value of a coin in cents -/
inductive Coin : Type
| Penny : Coin
| Nickel : Coin
| Dime : Coin
| Quarter : Coin
| HalfDollar : Coin

/-- Returns the value of a coin in cents -/
def coinValue : Coin → Nat
| Coin.Penny => 1
| Coin.Nickel => 5
| Coin.Dime => 10
| Coin.Quarter => 25
| Coin.HalfDollar => 50

/-- Represents the collection of coins in the wallet -/
def wallet : List Coin := [
  Coin.Penny, Coin.Penny, Coin.Penny,
  Coin.Nickel, Coin.Nickel,
  Coin.Dime, Coin.Dime,
  Coin.Quarter,
  Coin.HalfDollar
]

/-- Calculates the sum of two coins -/
def sumOfCoins (c1 c2 : Coin) : Nat :=
  coinValue c1 + coinValue c2

/-- Theorem: The maximum number of distinct sums is 15 -/
theorem max_distinct_sums :
  (List.join (List.map (λ c1 => List.map (λ c2 => sumOfCoins c1 c2) wallet) wallet)).toFinset.card = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_sums_l1274_127453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_validColoringsCount_l1274_127467

/-- A color represented as a natural number -/
def Color := Fin 3

/-- The set of integers from 2 to 10 -/
def IntegerSet : Finset Nat := Finset.range 9

/-- Function type for a coloring of the integer set -/
def Coloring := Nat → Color

/-- Predicate to check if a number is prime -/
def isPrime (n : Nat) : Prop := Nat.Prime n

/-- Function to get the proper divisors of a number -/
def properDivisors (n : Nat) : Finset Nat :=
  (Finset.range (n - 1)).filter (fun d => d > 1 ∧ n % d = 0)

/-- Predicate to check if a coloring is valid -/
def isValidColoring (c : Coloring) : Prop :=
  ∀ n ∈ IntegerSet,
    (∀ d ∈ properDivisors (n + 2), c (n + 2) ≠ c d) ∧
    (n > 0 → c (n + 2) ≠ c (n + 1))

/-- The set of all valid colorings -/
def ValidColorings : Finset Coloring :=
  sorry

/-- The main theorem statement -/
theorem validColoringsCount :
  Finset.card ValidColorings = 96 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_validColoringsCount_l1274_127467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l1274_127457

theorem negation_of_sin_inequality :
  (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x : ℝ, Real.sin x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l1274_127457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l1274_127431

-- Define the circle C
def Circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 12 = 0

-- Define the points P and Q
def P : ℝ × ℝ := (4, -2)
def Q : ℝ × ℝ := (-1, 3)

-- Define the line l
def Line (m : ℝ) (x y : ℝ) : Prop := x + y + m = 0

-- Define the theorem
theorem circle_and_line_equations :
  -- Circle C passes through P and Q
  Circle P.1 P.2 ∧ Circle Q.1 Q.2 ∧
  -- Radius of C is smaller than 5
  ∀ x y, Circle x y → (x - (-1))^2 + (y - (-6))^2 < 25 ∧
  -- Length of segment intercepted by y-axis is 4√3
  ∃ y₁ y₂, Circle 0 y₁ ∧ Circle 0 y₂ ∧ (y₁ - y₂)^2 = 48 ∧
  -- Line l is parallel to PQ
  ∃ m, ∀ x y, Line m x y → y - Q.2 = -1 * (x - Q.1) ∧
  -- Line l intersects C at A and B
  ∃ (A B : ℝ × ℝ), Circle A.1 A.2 ∧ Circle B.1 B.2 ∧ Line m A.1 A.2 ∧ Line m B.1 B.2 ∧
  -- Circle with diameter AB passes through origin
  (A.1 * B.1 + A.2 * B.2 = 0) →
  -- Conclusion: Equation of circle C and possible equations of line l
  (∀ x y, Circle x y ↔ x^2 + y^2 - 2*x - 12 = 0) ∧
  (m = 3 ∨ m = -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l1274_127431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_triangle_is_equilateral_l1274_127464

/-- A triangle is equilateral if all its sides are equal -/
def IsEquilateralTriangle (a b c : ℝ) : Prop := a = b ∧ b = c

/-- Predicate to check if three side lengths form a valid triangle -/
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_equilateral (a b c : ℝ) (h_triangle : IsTriangle a b c) 
  (h_condition : a^2 + b^2 + c^2 = a*b + a*c + b*c) : a = b ∧ b = c := by
  sorry

theorem triangle_is_equilateral (a b c : ℝ) (h_triangle : IsTriangle a b c) 
  (h_condition : a^2 + b^2 + c^2 = a*b + a*c + b*c) : IsEquilateralTriangle a b c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_triangle_is_equilateral_l1274_127464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_equals_i_i_squared_eq_neg_one_l1274_127425

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the theorem
theorem complex_division_equals_i : (3 + i) / (1 - 3 * i) = i := by
  -- The proof is omitted
  sorry

-- Verify that i * i = -1
theorem i_squared_eq_neg_one : i * i = -1 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_equals_i_i_squared_eq_neg_one_l1274_127425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l1274_127473

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

-- Define the line of symmetry
def symmetry_line (a b x y : ℝ) : Prop := 2*a*x + b*y + 6 = 0

-- Define the condition that the circle is symmetric with respect to the line
def is_symmetric (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), circle_C x y ∧ symmetry_line a b x y

-- Define the tangent length function
noncomputable def tangent_length (a b : ℝ) : ℝ := 
  Real.sqrt ((a+1)^2 + (b-2)^2 - 2)

-- Statement to prove
theorem min_tangent_length (a b : ℝ) :
  is_symmetric a b → ∃ (min_length : ℝ), 
    (∀ a' b', is_symmetric a' b' → tangent_length a' b' ≥ min_length) ∧
    (∃ a₀ b₀, is_symmetric a₀ b₀ ∧ tangent_length a₀ b₀ = min_length) ∧
    min_length = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l1274_127473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thaiangulation_difference_l1274_127403

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields
  vertices : List (Real × Real)
  is_convex : Bool -- This should be a proper condition in a real implementation

/-- A triangulation of a convex polygon -/
structure Triangulation (π : ConvexPolygon) where
  -- Add necessary fields
  triangles : List (Nat × Nat × Nat)
  valid : Bool -- This should be a proper condition in a real implementation

/-- A Thaiangulation is a triangulation where all triangles have the same area -/
structure Thaiangulation (π : ConvexPolygon) extends Triangulation π where
  equal_area : Bool -- This should be a proper condition in a real implementation

/-- The number of triangles that differ between two Thaiangulations -/
def differing_triangles (π : ConvexPolygon) (T1 T2 : Thaiangulation π) : Nat :=
  sorry

/-- Any two different Thaiangulations of a convex polygon differ by exactly two triangles -/
theorem thaiangulation_difference (π : ConvexPolygon) (T1 T2 : Thaiangulation π) 
  (h : T1 ≠ T2) : differing_triangles π T1 T2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thaiangulation_difference_l1274_127403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_not_contradictory_l1274_127471

-- Define the bag contents
def total_balls : ℕ := 4
def white_balls : ℕ := 2
def black_balls : ℕ := 2

-- Define the events
def exactly_one_black (drawn : Finset ℕ) : Prop :=
  drawn.card = 2 ∧ (drawn.filter (λ x ↦ x > white_balls)).card = 1

def exactly_two_black (drawn : Finset ℕ) : Prop :=
  drawn.card = 2 ∧ (drawn.filter (λ x ↦ x > white_balls)).card = 2

-- Theorem statement
theorem mutually_exclusive_not_contradictory :
  (∀ drawn : Finset ℕ, ¬(exactly_one_black drawn ∧ exactly_two_black drawn)) ∧
  (∃ drawn : Finset ℕ, exactly_one_black drawn) ∧
  (∃ drawn : Finset ℕ, exactly_two_black drawn) := by
  sorry

#check mutually_exclusive_not_contradictory

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_not_contradictory_l1274_127471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_and_beta_values_l1274_127480

theorem cosine_and_beta_values (α β : Real) 
  (h1 : Real.cos α = Real.sqrt 5 / 5)
  (h2 : Real.sin (α - β) = Real.sqrt 10 / 10)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : 0 < β ∧ β < Real.pi / 2) :
  Real.cos (2 * α - β) = Real.sqrt 2 / 10 ∧ β = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_and_beta_values_l1274_127480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1274_127451

def x : ℕ → ℝ
  | 0 => 115  -- Add this case for 0
  | 1 => 115
  | k + 2 => x (k + 1) ^ 2 + x (k + 1)

theorem series_sum : ∑' k, 1 / (x k + 1) = 1 / 115 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1274_127451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_ones_three_dice_l1274_127460

/-- The probability of rolling a 1 on a single standard die -/
def prob_one : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a single standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expected_ones : ℚ := 1 / 2

/-- Theorem stating that the expected number of 1's when rolling three standard dice is 1/2 -/
theorem expected_ones_three_dice : 
  (prob_one * num_dice) = expected_ones := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_ones_three_dice_l1274_127460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probability_of_units_digits_l1274_127481

/-- Represents the set of integers from 0 to 14 -/
def IntegerSet : Set ℕ := {n : ℕ | n ≤ 14}

/-- Jack's first pick -/
noncomputable def JackPick1 : IntegerSet → ℕ := sorry

/-- Jack's second pick -/
noncomputable def JackPick2 : IntegerSet → ℕ := sorry

/-- Jill's pick -/
noncomputable def JillPick : IntegerSet → ℕ := sorry

/-- The sum of the three picks -/
def TotalSum (j1 j2 k : ℕ) : ℕ := j1 + j2 + k

/-- The units digit of a number -/
def UnitsDigit (n : ℕ) : ℕ := n % 10

/-- The probability of a specific units digit occurring -/
noncomputable def ProbabilityOfUnitsDigit (d : ℕ) : ℚ := sorry

theorem equal_probability_of_units_digits :
  ∀ d₁ d₂ : ℕ, d₁ < 10 → d₂ < 10 →
  ProbabilityOfUnitsDigit d₁ = ProbabilityOfUnitsDigit d₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probability_of_units_digits_l1274_127481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_of_a_l1274_127419

noncomputable def a : Fin 2 → ℝ
  | 0 => 2
  | 1 => Real.sqrt 5

theorem unit_vector_of_a : 
  let norm_a := Real.sqrt (a 0 ^ 2 + a 1 ^ 2)
  let a₀ := fun i => a i / norm_a
  (a₀ 0 = 2/3) ∧ (a₀ 1 = Real.sqrt 5 / 3) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_of_a_l1274_127419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1274_127433

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3 * x^2

/-- The point of tangency -/
def point : ℝ × ℝ := (3, 27)

/-- The slope of the tangent line at the point of tangency -/
def tangent_slope : ℝ := f' point.1

/-- The y-intercept of the tangent line -/
def y_intercept : ℝ := point.2 - tangent_slope * point.1

/-- The x-intercept of the tangent line -/
noncomputable def x_intercept : ℝ := -y_intercept / tangent_slope

/-- The area of the triangle formed by the tangent line and the coordinate axes -/
noncomputable def triangle_area : ℝ := (1/2) * x_intercept * (-y_intercept)

theorem tangent_triangle_area : triangle_area = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1274_127433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_m_value_l1274_127434

/-- Given a hyperbola with equation x^2 - y^2/m = 1 (where m > 0) 
    and asymptotes y = ±√3x, the value of m is 3. -/
theorem hyperbola_asymptote_m_value (m : ℝ) 
  (h1 : m > 0) 
  (h2 : ∀ (x y : ℝ), x^2 - y^2/m = 1 → (y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x)) : 
  m = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_m_value_l1274_127434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maddox_camera_price_l1274_127491

/-- The price at which Maddox sold each camera -/
def maddox_price : ℕ → ℕ := sorry

/-- The number of cameras each person bought -/
def num_cameras : ℕ := 3

/-- The cost price of each camera -/
def cost_price : ℕ := 20

/-- The price at which Theo sold each camera -/
def theo_price : ℕ := 23

/-- The additional profit Maddox made compared to Theo -/
def additional_profit : ℕ := 15

theorem maddox_camera_price :
  ∀ n : ℕ, 
  (n * maddox_price n - n * cost_price) = 
  (n * theo_price - n * cost_price + additional_profit) →
  n = num_cameras →
  maddox_price n = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maddox_camera_price_l1274_127491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_of_neg_i_l1274_127404

/-- Dilation of a complex number -/
noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

theorem dilation_of_neg_i :
  dilation (1 + 3*I) (-3 : ℝ) (-I) = 4 + 15*I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_of_neg_i_l1274_127404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_rectangle_circle_specific_common_area_l1274_127426

/-- The area of the region common to a rectangle and a circle with shared center -/
noncomputable def area_common_region (a b r : ℝ) : ℝ := sorry

/-- The area of the region common to a rectangle and a circle with shared center -/
theorem common_area_rectangle_circle (a b r : ℝ) (ha : a > 0) (hb : b > 0) (hr : r > 0) :
  a ≤ 2 * r ∧ b ≤ 2 * r → area_common_region a b r = a * b :=
by sorry

/-- The specific case for a 10 by 4 rectangle and a circle of radius 5 -/
theorem specific_common_area : 
  area_common_region 10 4 5 = 40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_rectangle_circle_specific_common_area_l1274_127426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_member_count_l1274_127452

/-- Represents a club with committees and members -/
structure Club where
  num_committees : ℕ
  num_members : ℕ

/-- A club satisfies the given conditions -/
def satisfies_conditions (c : Club) : Prop :=
  c.num_committees = 5 ∧
  ∀ m : Fin c.num_members, ∃! pair : Fin 2 → Fin c.num_committees, True ∧
  ∀ i j : Fin c.num_committees, i < j →
    ∃! m : Fin c.num_members, ∃ pair : Fin 2 → Fin c.num_committees, pair 0 = i ∧ pair 1 = j

/-- Theorem stating that a club satisfying the conditions has 10 members -/
theorem club_member_count (c : Club) (h : satisfies_conditions c) : c.num_members = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_member_count_l1274_127452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perp_to_plane_are_parallel_l1274_127432

-- Define a 3D space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Fact (finrank ℝ V = 3)]

-- Define a plane in 3D space
def Plane (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := Submodule ℝ V

-- Define a line in 3D space
def Line (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := Submodule ℝ V

-- Define perpendicularity between a line and a plane
def perpendicular (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] (l : Line V) (p : Plane V) : Prop := sorry

-- Define parallelism between two lines
def parallel (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] (l1 l2 : Line V) : Prop := sorry

-- Theorem statement
theorem lines_perp_to_plane_are_parallel {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (p : Plane V) (l1 l2 : Line V) 
  (h1 : perpendicular V l1 p) (h2 : perpendicular V l2 p) : parallel V l1 l2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perp_to_plane_are_parallel_l1274_127432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_water_ratio_after_mixing_l1274_127427

/-- Given two jars A and B with alcohol-to-water ratios r:1 and s:1 respectively,
    where jar A has volume 2V₁ and jar B has volume 3V₁, and V₁ of alcohol is added to jar A before mixing,
    prove that the ratio of total alcohol to total water in the final mixture is as given. -/
theorem alcohol_water_ratio_after_mixing (r s V₁ : ℝ) (hr : r > 0) (hs : s > 0) (hV₁ : V₁ > 0) :
  (2 * r * V₁ / (r + 1) + V₁ + 3 * s * V₁ / (s + 1)) / (2 * V₁ / (r + 1) + 3 * V₁ / (s + 1)) =
  (2 * r / (r + 1) + 1 + 3 * s / (s + 1)) / (2 / (r + 1) + 3 / (s + 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_water_ratio_after_mixing_l1274_127427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_properties_l1274_127438

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (e₁ e₂ : V)

-- Define the plane as the span of two vectors
def plane (e₁ e₂ : V) := Submodule.span ℝ {e₁, e₂}

-- Non-collinearity condition
def non_collinear (e₁ e₂ : V) : Prop := ∀ (c : ℝ), c • e₁ ≠ e₂

theorem plane_properties (h : non_collinear e₁ e₂) :
  (∀ v ∈ plane e₁ e₂, ∃ (a b : ℝ), v = a • e₁ + b • e₂) ∧
  (∀ (a b : ℝ), a • e₁ + b • e₂ = 0 → a = 0 ∧ b = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_properties_l1274_127438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_cells_inequality_l1274_127497

theorem black_cells_inequality (n k : ℕ) (h : k > 0) (hn : n > 0) :
  (2 : ℝ) * k / n ≤ Real.sqrt (8 * n - 7) + 1 := by
  sorry

#check black_cells_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_cells_inequality_l1274_127497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_harmonic_sets_union_reals_l1274_127466

-- Definition of a harmonic set
def is_harmonic_set (S : Set ℝ) : Prop :=
  S.Nonempty ∧ ∀ a b, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a - b) ∈ S

-- Theorem statement
theorem exists_harmonic_sets_union_reals :
  ∃ (S₁ S₂ : Set ℝ), is_harmonic_set S₁ ∧ is_harmonic_set S₂ ∧ 
  S₁ ≠ Set.univ ∧ S₂ ≠ Set.univ ∧ S₁ ∪ S₂ = Set.univ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_harmonic_sets_union_reals_l1274_127466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_A_line_intersects_circle_shortest_chord_l1274_127400

/-- Circle C -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

/-- Line l -/
def line_l (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y = 7*m + 4

/-- Point A -/
def point_A : ℝ × ℝ := (3, 1)

/-- Theorem stating that line l passes through point A for all m ∈ ℝ -/
theorem line_passes_through_A : ∀ m : ℝ, line_l m (point_A.1) (point_A.2) := by sorry

/-- Theorem stating that line l intersects circle C for all m ∈ ℝ -/
theorem line_intersects_circle : ∀ m : ℝ, ∃ x y : ℝ, line_l m x y ∧ circle_C x y := by sorry

/-- The equation of line l when the chord length is shortest -/
def shortest_chord_line (x y : ℝ) : Prop := 2*x - y - 5 = 0

/-- Theorem stating that the shortest_chord_line produces the shortest chord when intersecting circle C -/
theorem shortest_chord :
  ∀ m : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ,
    line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ →
    ∀ x₃ y₃ x₄ y₄ : ℝ,
      shortest_chord_line x₃ y₃ ∧ shortest_chord_line x₄ y₄ ∧ circle_C x₃ y₃ ∧ circle_C x₄ y₄ →
      (x₁ - x₂)^2 + (y₁ - y₂)^2 ≥ (x₃ - x₄)^2 + (y₃ - y₄)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_A_line_intersects_circle_shortest_chord_l1274_127400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1274_127498

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (ω * x + φ) + 2 * (Real.sin ((ω * x + φ) / 2))^2 - 1

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 3)

theorem function_properties (ω φ : ℝ) 
    (h_ω : ω > 0) 
    (h_φ : 0 < φ ∧ φ < Real.pi) 
    (h_symmetry : ∀ x : ℝ, f ω φ (x + Real.pi / (2 * ω)) = f ω φ x) 
    (h_f0 : f ω φ 0 = 0) :
  (∀ x, f ω φ x = 2 * Real.sin (2 * x)) ∧
  (∀ x, g x = 2 * Real.sin (4 * x - Real.pi / 3)) ∧
  (Set.Icc (-Real.pi/12) (Real.pi/6) ⊆ g ⁻¹' (Set.Icc (-2) (Real.sqrt 3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1274_127498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_Q_points_l1274_127486

-- Define the cyclic 100-gon
def cyclic_100gon (P : ℕ → ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i, dist center (P i) = radius

-- Define the periodic condition
def periodic_condition (P : ℕ → ℝ × ℝ) : Prop :=
  ∀ i, P i = P (i + 100)

-- Define the Q points
noncomputable def Q (P : ℕ → ℝ × ℝ) (i : ℕ) : ℝ × ℝ :=
  sorry -- Placeholder for the intersection point calculation

-- Define the perpendicular condition
def perpendicular_condition (P : ℕ → ℝ × ℝ) : Prop :=
  ∃ point : ℝ × ℝ, ∀ i, (point.1 - (P i).1) * ((P (i - 1)).2 - (P (i + 1)).2) + 
                        (point.2 - (P i).2) * ((P (i + 1)).1 - (P (i - 1)).1) = 0

-- Define concyclicity
def concyclic (points : List (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ point ∈ points, dist center point = radius

-- The main theorem
theorem concyclic_Q_points (P : ℕ → ℝ × ℝ) :
  cyclic_100gon P →
  periodic_condition P →
  perpendicular_condition P →
  concyclic (List.map (Q P) (List.range 100)) :=
by
  sorry -- Proof to be implemented


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_Q_points_l1274_127486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1274_127406

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_property (a₁ q : ℝ) (m : ℕ) :
  (a₁ ≠ 0) →
  (q ≠ 0) →
  (q ≠ 1) →
  (geometric_sum a₁ q 3 + geometric_sum a₁ q 6 = 2 * geometric_sum a₁ q 9) →
  (geometric_sequence a₁ q 2 + geometric_sequence a₁ q 5 = 2 * geometric_sequence a₁ q m) →
  m = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1274_127406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1274_127414

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (|x + 1|) / Real.log a

-- State the theorem
theorem solution_set_of_inequality (a : ℝ) :
  (∀ x ∈ Set.Ioo (-2 : ℝ) (-1), f a x > 0) →
  (Set.Ioo 0 (1/2) = {a | f a (4^a - 1) > f a 1}) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1274_127414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_midpoint_locus_l1274_127415

-- Define the parabola E
def E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle F
def F (x y r : ℝ) : Prop := (x-1)^2 + y^2 = r^2

-- Define the line l₀
def l₀ (t : ℝ) (x y : ℝ) : Prop := x = t*y + 1

-- Define the focus point
def focus : ℝ × ℝ := (1, 0)

-- Define point T
def T : ℝ × ℝ := (0, 1)

-- Theorem for part A
theorem intersection_property (t : ℝ) (x₁ y₁ x₂ y₂ y₃ : ℝ) :
  E x₁ y₁ → E x₂ y₂ → l₀ t x₁ y₁ → l₀ t x₂ y₂ → y₃ = -1/t →
  1/y₁ + 1/y₂ = 1/y₃ := by sorry

-- Theorem for part B
theorem midpoint_locus (t : ℝ) (x₁ y₁ x₂ y₂ x y : ℝ) :
  E x₁ y₁ → E x₂ y₂ → l₀ t x₁ y₁ → l₀ t x₂ y₂ →
  x = (x₁ + x₂)/2 → y = (y₁ + y₂)/2 →
  ∃ (a b c : ℝ), y^2 = a*x + b*y + c := by sorry

-- Additional constraints
axiom r_bounds : ∃ (r : ℝ), 0 < r ∧ r < 1

-- Modified circle_def to avoid redeclaration
axiom circle_constraint : ∀ (x y : ℝ), ∃ (r : ℝ), F x y r → r_bounds.choose = r

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_midpoint_locus_l1274_127415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1274_127430

theorem parallel_vectors_lambda (a b : ℝ × ℝ) (lambda : ℝ) :
  a = (2, 5) →
  b = (lambda, 4) →
  (∃ (k : ℝ), a = k • b) →
  lambda = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1274_127430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_problem_l1274_127492

structure Restaurant :=
  (name : String)

def A : Restaurant := ⟨"A"⟩
def B : Restaurant := ⟨"B"⟩

noncomputable def probability_A_second_day (p_A_given_A p_A_given_B : ℝ) : ℝ :=
  0.5 * p_A_given_A + 0.5 * p_A_given_B

def choose (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def probability_at_least_one_western (total western chosen : ℕ) : ℝ :=
  1 - (choose (total - western) chosen : ℝ) / (choose total chosen : ℝ)

theorem restaurant_problem :
  (probability_A_second_day 0.4 0.8 = 0.6) ∧
  (probability_at_least_one_western 10 4 3 = 5/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_problem_l1274_127492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_plus_alpha_sin_geq_one_l1274_127450

theorem cos_squared_plus_alpha_sin_geq_one (α : ℝ) (h : 0 ≤ α ∧ α ≤ π/2) :
  (Real.cos α) ^ 2 + α * Real.sin α ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_plus_alpha_sin_geq_one_l1274_127450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_from_tan_l1274_127439

theorem sin_cos_from_tan (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_from_tan_l1274_127439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_answer_choices_l1274_127421

theorem test_answer_choices (num_questions num_ways num_choices : ℕ) :
  num_questions = 4 ∧ 
  num_ways = 625 ∧
  num_ways = (num_choices + 1) ^ num_questions →
  num_choices = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_answer_choices_l1274_127421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_change_l1274_127443

theorem salary_change (S : ℝ) (h : S > 0) : 
  (S + S * (10 / 100)) * (1 - 10 / 100) = S * (1 - 1 / 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_change_l1274_127443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_length_l1274_127435

/-- A vector in the 0 × y plane with nonnegative coordinates -/
structure NonNegVector :=
  (x : ℝ)
  (y : ℝ)
  (x_nonneg : x ≥ 0)
  (y_nonneg : y ≥ 0)

/-- A unit vector in the 0 × y plane with nonnegative coordinates -/
def UnitVector (v : NonNegVector) : Prop :=
  v.x^2 + v.y^2 = 1

/-- The sum of a list of vectors -/
def VectorSum (vs : List NonNegVector) : NonNegVector :=
  ⟨vs.foldl (· + ·.x) 0, vs.foldl (· + ·.y) 0, 
   by sorry, by sorry⟩

/-- The length of a vector -/
noncomputable def VectorLength (v : NonNegVector) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

theorem smallest_sum_length (vs : List NonNegVector) 
  (h1 : vs.length = 7) 
  (h2 : ∀ v ∈ vs, UnitVector v) : 
  VectorLength (VectorSum vs) ≥ 5 ∧ 
  ∃ vs', vs'.length = 7 ∧ (∀ v ∈ vs', UnitVector v) ∧ 
         VectorLength (VectorSum vs') = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_length_l1274_127435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1274_127410

theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = 1) 
  (h2 : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 2) 
  (h3 : (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2)) = 3) :
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) * Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)))) = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1274_127410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_plane_through_parallel_lines_l1274_127446

-- Define the concept of a line in 3D space
def Line : Type := ℝ → ℝ × ℝ × ℝ

-- Define the concept of a plane in 3D space
def Plane : Type := ℝ × ℝ × ℝ → Prop

-- Define what it means for two lines to be parallel
def parallel (a b : Line) : Prop :=
  ∃ (P : Plane), (∀ t : ℝ, P (a t)) ∧ (∀ t : ℝ, P (b t)) ∧ (∀ t s : ℝ, a t ≠ b s)

-- State the theorem
theorem unique_plane_through_parallel_lines (a b : Line) (h : parallel a b) :
  ∃! P : Plane, (∀ t : ℝ, P (a t)) ∧ (∀ t : ℝ, P (b t)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_plane_through_parallel_lines_l1274_127446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1274_127402

theorem trigonometric_identities (α : Real) 
  (h : Real.sin α / (Real.sin α - Real.cos α) = -1) : 
  Real.tan α = 1/2 ∧ Real.sin α^4 + Real.cos α^4 = 17/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1274_127402
