import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_diff_l687_68714

theorem min_cos_diff (x y z : ℝ) 
  (eq1 : Real.sqrt 3 * Real.sin x = Real.tan y)
  (eq2 : 2 * Real.sin y = 1 / Real.tan z)
  (eq3 : Real.sin z = 2 * Real.tan x) :
  ∃ (m : ℝ), ∀ (x' y' z' : ℝ),
    Real.sqrt 3 * Real.sin x' = Real.tan y' →
    2 * Real.sin y' = 1 / Real.tan z' →
    Real.sin z' = 2 * Real.tan x' →
    Real.cos x' - Real.cos z' ≥ m ∧
    m = -7 * Real.sqrt 2 / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_diff_l687_68714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l687_68733

-- Define the points P and Q
noncomputable def P (m : ℝ) : ℝ × ℝ := (-2, m)
noncomputable def Q (m : ℝ) : ℝ × ℝ := (m, 4)

-- Define the inclination angle
noncomputable def inclination_angle : ℝ := Real.arctan (1/2)

-- Theorem statement
theorem line_inclination (m : ℝ) :
  let slope := (Q m).2 - (P m).2 / ((Q m).1 - (P m).1)
  slope = Real.tan inclination_angle → m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l687_68733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_transformation_l687_68742

-- Define the points
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (4, 6)
def R : ℝ × ℝ := (7, 2)

-- Define the rotation function
def rotate180AroundP (point : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 - (point.1 - P.1), P.2 - (point.2 - P.2))

-- Define the translation function
def translate (point : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (point.1 + dx, point.2 + dy)

-- Theorem statement
theorem midpoint_after_transformation :
  let Q' := translate (rotate180AroundP Q) 3 (-4)
  let R' := translate (rotate180AroundP R) 3 (-4)
  (Q'.1 + R'.1) / 2 = -1/2 ∧ (Q'.2 + R'.2) / 2 = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_transformation_l687_68742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l687_68758

/-- The minimum distance from any point on the line 3x + 4y = 15 to the origin is 3 -/
theorem min_distance_to_origin : 
  ∃ (d : ℝ), d = 3 ∧ 
  ∀ (a b : ℝ), (3 * a + 4 * b = 15) → Real.sqrt (a^2 + b^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l687_68758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l687_68774

open Real

theorem equation_solution (x y z : ℝ) (hx : Real.sin x ≠ 0) (hy : Real.cos y ≠ 0) :
  (Real.sin x ^ 2 + 1 / (Real.sin x ^ 2)) ^ 3 + (Real.cos y ^ 2 + 1 / (Real.cos y ^ 2)) ^ 3 = 16 * Real.cos z ↔
  ∃ (n k m : ℤ), x = π / 2 + n * π ∧ y = k * π ∧ z = 2 * m * π :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l687_68774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_P_l687_68737

/-- The point A in 3D space -/
def A : Fin 3 → ℝ := ![8, 0, 0]

/-- The point B in 3D space -/
def B : Fin 3 → ℝ := ![0, -4, 0]

/-- The point C in 3D space -/
def C : Fin 3 → ℝ := ![0, 0, 6]

/-- The point D in 3D space -/
def D : Fin 3 → ℝ := ![0, 0, 0]

/-- The point P in 3D space -/
def P : Fin 3 → ℝ := ![4, -2, 3]

/-- Calculate the squared distance between two points in 3D space -/
def squaredDistance (p1 p2 : Fin 3 → ℝ) : ℝ :=
  (p1 0 - p2 0)^2 + (p1 1 - p2 1)^2 + (p1 2 - p2 2)^2

/-- Theorem stating that P is equidistant from A, B, C, and D -/
theorem equidistant_P : 
  squaredDistance P A = squaredDistance P B ∧
  squaredDistance P B = squaredDistance P C ∧
  squaredDistance P C = squaredDistance P D := by
  sorry

#eval squaredDistance P A
#eval squaredDistance P B
#eval squaredDistance P C
#eval squaredDistance P D

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_P_l687_68737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_unique_factorization_l687_68728

theorem polynomial_unique_factorization (k : Type) [Field k] :
  ∀ (f : Polynomial k), 
    ∃! (factors : List (Polynomial k)), 
      (∀ p ∈ factors, Irreducible p) ∧ 
      (∃ (c : k), f = c • (factors.prod)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_unique_factorization_l687_68728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l687_68716

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)
variable (x y : ℝ)

def p (a b : V) : V := 2 • a - 3 • b
def q (a b : V) : V := -1 • a + 5 • b

theorem vector_equation_solution 
  (h_not_collinear : ¬ ∃ (k : ℝ), a = k • b)
  (h_eq : x • p a b + y • q a b = 2 • a - b) :
  x = 9/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l687_68716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_fourth_power_l687_68740

theorem det_B_fourth_power {n : Type*} [Fintype n] [DecidableEq n] 
  (B : Matrix n n ℝ) (h : Matrix.det B = -3) : Matrix.det (B^4) = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_fourth_power_l687_68740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_words_on_each_page_l687_68707

/-- Represents the number of words on each page of a book -/
def words_per_page : ℕ := sorry

/-- The total number of pages in the book -/
def total_pages : ℕ := 150

/-- The maximum number of words allowed on a page -/
def max_words_per_page : ℕ := 200

/-- Theorem stating the conditions and the result to be proved -/
theorem words_on_each_page :
  words_per_page ≤ max_words_per_page ∧
  (total_pages * words_per_page) % 250 = 137 →
  words_per_page = 198 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_words_on_each_page_l687_68707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l687_68738

noncomputable def A (n : ℕ) (a : ℝ) : ℝ :=
  match n with
  | 0 => 0
  | k + 1 => Real.sqrt (1 + Real.sqrt (a + A k a))

theorem inequality_proof (n : ℕ) (a : ℝ) (hn : n ≥ 2) (ha : a ≥ 2) :
  (Finset.range n).sum (λ k => A (k + 1) a) < n * a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l687_68738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l687_68757

theorem function_inequality (m : ℝ) (h_m : m ∈ Set.Ioo (-1) 0) :
  let f (x : ℝ) := (x^2 + m*x + 1) / Real.exp x
  ∀ x₁ x₂, x₁ ∈ Set.Icc 1 (1 - m) → x₂ ∈ Set.Icc 1 (1 - m) → 4 * f x₁ + x₂ < 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l687_68757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_team_average_weight_l687_68710

theorem bowling_team_average_weight 
  (original_players : ℕ) 
  (original_average : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) :
  original_players = 7 →
  original_average = 112 →
  new_player1_weight = 110 →
  new_player2_weight = 60 →
  (original_players * original_average + new_player1_weight + new_player2_weight) / (original_players + 2) = 106 :=
by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_team_average_weight_l687_68710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_ratio_l687_68724

theorem garden_area_ratio (L W : ℝ) (h1 : L > 0) (h2 : W > 0) : 
  (1.4 * L * 0.8 * W) / (L * W) = 1.12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_ratio_l687_68724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l687_68796

/-- Given a function f such that f(2x-1) = x^2 + x for all x, 
    prove that the minimum value of f is -1/4 -/
theorem min_value_of_f (f : ℝ → ℝ) 
    (h : ∀ x, f (2*x - 1) = x^2 + x) : 
    ∃ x₀, f x₀ = -1/4 ∧ ∀ x, f x ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l687_68796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_point_l687_68784

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the distance function between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem min_distance_ellipse_to_point : 
  ∃ (min_dist : ℝ), min_dist = 2 * Real.sqrt 6 / 3 ∧
  ∀ (x y : ℝ), ellipse x y → 
    distance x y 2 0 ≥ min_dist := by
  sorry

#check min_distance_ellipse_to_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_point_l687_68784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_outside_angles_is_720_l687_68794

/-- A pentagon inscribed in a circle -/
structure InscribedPentagon where
  /-- The circle in which the pentagon is inscribed -/
  circle : Set ℝ
  /-- The five vertices of the pentagon -/
  vertices : Fin 5 → ℝ
  /-- Assertion that the vertices lie on the circle -/
  vertices_on_circle : ∀ i, (vertices i) ∈ circle

/-- An angle inscribed in a segment outside the pentagon -/
noncomputable def OutsideAngle (p : InscribedPentagon) (i : Fin 5) : ℝ :=
  -- Definition of the angle (implementation details omitted)
  sorry

/-- The sum of the five outside angles -/
noncomputable def SumOutsideAngles (p : InscribedPentagon) : ℝ :=
  (Finset.sum Finset.univ fun i => OutsideAngle p i)

/-- Theorem stating that the sum of outside angles is 720° -/
theorem sum_outside_angles_is_720 (p : InscribedPentagon) :
  SumOutsideAngles p = 720 := by
  sorry

#check sum_outside_angles_is_720

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_outside_angles_is_720_l687_68794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l687_68734

/-- Calculate the present value given future value, interest rate, and time period. -/
noncomputable def present_value (future_value : ℝ) (interest_rate : ℝ) (time : ℕ) : ℝ :=
  future_value / (1 + interest_rate) ^ time

/-- The problem statement -/
theorem investment_problem :
  let future_value : ℝ := 400000
  let interest_rate : ℝ := 0.06
  let time : ℕ := 10
  let calculated_present_value := present_value future_value interest_rate time
  ∃ ε > 0, |calculated_present_value - 223387.15| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l687_68734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l687_68748

-- Define the set of x satisfying the inequality
def solution_set : Set ℝ := {x : ℝ | x^2 + 4*x > 45}

-- Define the expected result set
def expected_set : Set ℝ := Set.Iic (-9) ∪ Set.Ioi 5

-- Theorem statement
theorem quadratic_inequality_solution : solution_set = expected_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l687_68748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_over_cos_l687_68741

theorem sin_double_over_cos (α : ℝ) : 
  Real.sin (π + α) = -(1/3) → (Real.sin (2*α)) / (Real.cos α) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_over_cos_l687_68741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l687_68711

/-- Represents a parallelogram with given perimeter and distances between opposite sides -/
structure Parallelogram where
  perimeter : ℝ
  distance1 : ℝ
  distance2 : ℝ

/-- Calculates the area of a parallelogram -/
noncomputable def area (p : Parallelogram) : ℝ :=
  (p.perimeter / 4) * p.distance1

/-- Theorem: A parallelogram with perimeter 30 and distances 2 and 3 between opposite sides has an area of 18 -/
theorem parallelogram_area (p : Parallelogram) 
  (h1 : p.perimeter = 30) 
  (h2 : p.distance1 = 2) 
  (h3 : p.distance2 = 3) : 
  area p = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l687_68711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l687_68744

noncomputable def f (x : ℝ) := 2 * Real.sin x * Real.cos x - Real.cos x ^ 2 + Real.sin x ^ 2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ k : ℤ, ∀ x y : ℝ, k * Real.pi + 3 * Real.pi / 8 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 7 * Real.pi / 8 → f y < f x) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ Real.sqrt 2) ∧
  (f (3 * Real.pi / 8) = Real.sqrt 2) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ -1) ∧
  (f 0 = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l687_68744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_profit_and_decision_l687_68788

/-- Represents the quality grade of a product -/
inductive Grade
| FirstClass
| SecondClass
| Unqualified

/-- Represents the company's production characteristics -/
structure Company where
  dailyOutput : ℕ
  firstClassProb : ℝ
  secondClassProb : ℝ
  unqualifiedProb : ℝ
  firstClassProfit : ℝ
  secondClassProfit : ℝ
  unqualifiedProfit : ℝ

/-- Calculates the expected daily profit for the company -/
def expectedDailyProfit (c : Company) : ℝ :=
  c.dailyOutput * (c.firstClassProb * c.firstClassProfit +
                   c.secondClassProb * c.secondClassProfit +
                   c.unqualifiedProb * c.unqualifiedProfit)

/-- Calculates the net profit increase for additional units -/
noncomputable def netProfitIncrease (n : ℕ) (avgProfit : ℝ) : ℝ :=
  Real.log (n : ℝ) - 0.39 * (n : ℝ)

/-- Theorem stating the expected daily profit and production increase decision -/
theorem company_profit_and_decision (c : Company) 
  (h1 : c.dailyOutput = 2)
  (h2 : c.firstClassProb = 0.5)
  (h3 : c.secondClassProb = 0.4)
  (h4 : c.unqualifiedProb = 0.1)
  (h5 : c.firstClassProfit = 0.8)
  (h6 : c.secondClassProfit = 0.6)
  (h7 : c.unqualifiedProfit = -0.3) :
  expectedDailyProfit c = 1.22 ∧ 
  ∀ n : ℕ, n > 0 → netProfitIncrease n (expectedDailyProfit c / c.dailyOutput) < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_profit_and_decision_l687_68788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l687_68754

noncomputable def A : ℝ × ℝ := (-1, 0)
noncomputable def B : ℝ × ℝ := (3, 8)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def C : ℝ × ℝ := (7/3, 20/3)

theorem point_C_coordinates :
  distance A C = 2 * distance B C ∧
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l687_68754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_sixth_l687_68778

theorem sin_alpha_plus_pi_sixth (α : Real) 
  (h1 : Real.cos α = -3/5) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.sin (α + π/6) = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_sixth_l687_68778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l687_68799

/-- A line y = kx + 3 intersects a circle (x-3)^2 + (y-2)^2 = 4 at points M and N. -/
def intersects (k : ℝ) (M N : ℝ × ℝ) : Prop :=
  (M.2 = k * M.1 + 3) ∧
  (N.2 = k * N.1 + 3) ∧
  ((M.1 - 3)^2 + (M.2 - 2)^2 = 4) ∧
  ((N.1 - 3)^2 + (N.2 - 2)^2 = 4)

/-- The distance between two points in ℝ² -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- Theorem: If a line y = kx + 3 intersects a circle (x-3)^2 + (y-2)^2 = 4 at points M and N
    such that |MN| = 2√3, then k = -3/4 or k = 0. -/
theorem line_circle_intersection (k : ℝ) (M N : ℝ × ℝ) 
  (h : intersects k M N) (d : distance M N = 2 * Real.sqrt 3) :
  k = -3/4 ∨ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l687_68799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l687_68735

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then 2*x - x^2
  else if -2 ≤ x ∧ x ≤ 0 then x^2 + 6*x
  else 0  -- This else case is added to make the function total

theorem f_range : Set.range f = Set.Icc (-8 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l687_68735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_workout_distance_l687_68725

/-- Calculates the total distance walked by Sarah given her workout parameters. -/
noncomputable def total_distance_walked (uphill_speed : ℝ) (downhill_speed : ℝ) (rest_time : ℝ) (total_time : ℝ) : ℝ :=
  let walking_time := total_time - rest_time
  let d := (walking_time * uphill_speed * downhill_speed) / (uphill_speed + downhill_speed)
  2 * d

/-- Theorem stating that Sarah's total distance walked is approximately 10.858 miles. -/
theorem sarah_workout_distance :
  let uphill_speed := (3 : ℝ) -- mph
  let downhill_speed := (4 : ℝ) -- mph
  let rest_time := (1/3 : ℝ) -- hours (20 minutes)
  let total_time := (3.5 : ℝ) -- hours
  abs (total_distance_walked uphill_speed downhill_speed rest_time total_time - 10.858) < 0.001 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval total_distance_walked 3 4 (1/3) 3.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_workout_distance_l687_68725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_cos_2theta_l687_68722

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.cos (x + Real.pi/3)

theorem f_range_and_cos_2theta :
  (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ∈ Set.Icc (-1/4) (1/2)) ∧
  (∀ θ : ℝ, -Real.pi/6 < θ ∧ θ < Real.pi/6 → f θ = 13/20 → Real.cos (2*θ) = (4 - 3*Real.sqrt 3)/10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_cos_2theta_l687_68722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_of_right_triangle_with_given_medians_hypotenuse_calculation_correct_l687_68793

/-- A right triangle with given medians -/
structure RightTriangle where
  -- Length of the first median
  median1 : ℝ
  -- Length of the second median
  median2 : ℝ
  -- Condition: The first median is 6
  h1 : median1 = 6
  -- Condition: The second median is √72
  h2 : median2 = Real.sqrt 72

/-- The length of the hypotenuse of a right triangle with given medians -/
noncomputable def hypotenuse (t : RightTriangle) : ℝ :=
  2 * Real.sqrt 86.4

/-- Theorem: The hypotenuse of a right triangle with medians 6 and √72 is 2√86.4 -/
theorem hypotenuse_of_right_triangle_with_given_medians (t : RightTriangle) :
  hypotenuse t = 2 * Real.sqrt 86.4 := by
  -- Unfold the definition of hypotenuse
  unfold hypotenuse
  -- The equality holds by definition
  rfl

/-- Proof that the hypotenuse calculation is correct -/
theorem hypotenuse_calculation_correct (t : RightTriangle) :
  (hypotenuse t) ^ 2 = 4 * (t.median1 ^ 2 + t.median2 ^ 2) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_of_right_triangle_with_given_medians_hypotenuse_calculation_correct_l687_68793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_digit_divisibility_l687_68755

def is_divisible_by (n m : ℕ) : Bool :=
  n % m = 0

def count_divisors (n : ℕ) : ℕ :=
  (List.range 9).filter (λ i => is_divisible_by n (i + 1)) |>.length

def eleven_digit_number (a b : ℕ) : ℕ :=
  a * 10000000000 + 123456789 * 10 + b

theorem eleven_digit_divisibility :
  ∀ a b : ℕ, 
    a < 10 → b < 10 →
    count_divisors (eleven_digit_number a b) = 8 →
    a = 3 ∧ b = 6 := by
  sorry

#eval count_divisors (eleven_digit_number 3 6)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_digit_divisibility_l687_68755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_xyz_equals_34_l687_68700

-- Define c and d as noncomputable
noncomputable def c : ℝ := -5/9
noncomputable def d : ℝ := (Real.sqrt 3 - 1) / 3

-- State the theorem
theorem sum_of_xyz_equals_34 :
  c^2 = 25/81 ∧ 
  d^2 = (Real.sqrt 3 - 1)^2 / 9 ∧ 
  c < 0 ∧ 
  d > 0 ∧ 
  ∃ (x y z : ℕ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    (c - d)^2 = (x : ℝ) * Real.sqrt y / z →
  x + y + z = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_xyz_equals_34_l687_68700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_ratio_range_l687_68795

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given vectors a and b satisfying ‖a + b‖ = 3 and ‖a - b‖ = 2,
    the range of values for ‖a‖ / (a • b) is [2/5, 2]. -/
theorem vector_ratio_range (a b : E) 
    (h1 : ‖a + b‖ = 3)
    (h2 : ‖a - b‖ = 2) :
    ∃ (x : ℝ), x ∈ Set.Icc (2/5 : ℝ) 2 ∧ ‖a‖ / (inner a b) = x :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_ratio_range_l687_68795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_l687_68797

-- Define the angles
variable (angle_A angle_B angle_C : ℝ)

-- State the conditions
axiom supplementary_angles : angle_A + angle_B = 180
axiom angle_C_relation : angle_C = angle_B + 15

-- State the theorem to be proved
theorem angle_C_value : angle_C = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_l687_68797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approximately_5_39_minutes_l687_68721

/-- Represents the jogging track and joggers' characteristics -/
structure TrackAndJoggers where
  trackCircumference : ℝ
  sureshFlatSpeed : ℝ
  wifeFlatSpeed : ℝ
  sureshDownhillIncrease : ℝ
  sureshUphillDecrease : ℝ
  wifeDownhillIncrease : ℝ
  wifeUphillDecrease : ℝ

/-- Calculates the meeting time for the joggers -/
noncomputable def calculateMeetingTime (tj : TrackAndJoggers) : ℝ :=
  let sureshAvgSpeed := (tj.sureshFlatSpeed * (2 + tj.sureshDownhillIncrease - tj.sureshUphillDecrease)) / 2
  let wifeAvgSpeed := (tj.wifeFlatSpeed * (2 + tj.wifeDownhillIncrease - tj.wifeUphillDecrease)) / 2
  let combinedSpeed := sureshAvgSpeed + wifeAvgSpeed
  tj.trackCircumference / combinedSpeed

/-- Theorem stating that the meeting time is approximately 5.39 minutes -/
theorem meeting_time_approximately_5_39_minutes :
  let tj : TrackAndJoggers := {
    trackCircumference := 726,
    sureshFlatSpeed := 75,  -- 4.5 km/hr converted to m/min
    wifeFlatSpeed := 62.5,  -- 3.75 km/hr converted to m/min
    sureshDownhillIncrease := 0.1,
    sureshUphillDecrease := 0.15,
    wifeDownhillIncrease := 0.07,
    wifeUphillDecrease := 0.1
  }
  ∃ ε > 0, |calculateMeetingTime tj - 5.39| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approximately_5_39_minutes_l687_68721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_five_l687_68760

theorem reciprocal_of_negative_five :
  (λ x : ℚ => -5 * x = 1) (-1/5) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_five_l687_68760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l687_68736

noncomputable def f (x : ℝ) : ℝ := 2 / Real.sqrt (x - 1)

theorem f_domain : Set ℝ = {x : ℝ | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l687_68736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_1500_terms_l687_68705

/-- Defines the sequence as described in the problem -/
def my_sequence : ℕ → ℕ
| 0 => 1
| n + 1 => 
  let block := n / 3
  let pos := n % 3
  if pos = 0 then 1
  else if pos = 1 then 
    if block % 2 = 0 then 2 else block / 2 + 3
  else 2

/-- Calculates the sum of the first n terms of the sequence -/
def sum_sequence (n : ℕ) : ℕ :=
  (List.range n).map my_sequence |>.sum

/-- The theorem to be proved -/
theorem sum_1500_terms : 
  ∃ (result : ℕ), sum_sequence 1500 = result := by
  -- The proof goes here
  sorry

#eval sum_sequence 1500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_1500_terms_l687_68705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_theorem_l687_68704

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 90 degrees counter-clockwise about the origin -/
def rotate90 (p : Point) : Point :=
  { x := -p.y, y := p.x }

/-- The original function y = e^x -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x

/-- The rotated function y = -ln(-x) -/
noncomputable def g (x : ℝ) : ℝ := -Real.log (-x)

theorem rotation_theorem :
  ∀ x : ℝ, x < 0 →
    let original_point : Point := { x := x, y := f x }
    let rotated_point : Point := rotate90 original_point
    rotated_point.y = g rotated_point.x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_theorem_l687_68704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l687_68782

noncomputable def f (x : ℝ) := x^3 - 1/2 * x^2 - 2*x

theorem f_properties :
  (f 1 = -3/2) ∧
  (deriv f 1 = 0) ∧
  (∀ x, x < -2/3 → StrictMono (f ∘ (fun y ↦ min y x))) ∧
  (∀ x, x > 1 → StrictMono (f ∘ (fun y ↦ max y x))) ∧
  (StrictAnti (f ∘ (fun y ↦ max (-2/3) (min y 1)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l687_68782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medical_research_support_percentage_l687_68773

theorem medical_research_support_percentage 
  (men_support_rate : ℝ) 
  (women_support_rate : ℝ)
  (men_surveyed : ℕ) 
  (women_surveyed : ℕ) 
  (h1 : men_support_rate = 0.6)
  (h2 : women_support_rate = 0.8)
  (h3 : men_surveyed = 100)
  (h4 : women_surveyed = 900) :
  (men_support_rate * (men_surveyed : ℝ) + women_support_rate * (women_surveyed : ℝ)) / 
  ((men_surveyed : ℝ) + (women_surveyed : ℝ)) = 0.78 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_medical_research_support_percentage_l687_68773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l687_68777

-- Define OddFunction
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x > 0 then 2017^x + Real.log x / Real.log 2017
  else if x < 0 then -(2017^(-x) + Real.log (-x) / Real.log 2017)
  else 0

-- State the theorem
theorem f_has_three_zeros :
  OddFunction f ∧ (∃! (z₁ z₂ z₃ : ℝ), z₁ < 0 ∧ z₂ = 0 ∧ z₃ > 0 ∧ f z₁ = 0 ∧ f z₂ = 0 ∧ f z₃ = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l687_68777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_P_and_parallel_to_tangent_l687_68701

-- Define the curve
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the point M
def M : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at M
noncomputable def m : ℝ := deriv f M.1

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := 2 * x - y + 4 = 0

-- Theorem statement
theorem line_passes_through_P_and_parallel_to_tangent :
  line_equation P.1 P.2 ∧
  (∀ x y : ℝ, line_equation x y → (y - P.2) = m * (x - P.1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_P_and_parallel_to_tangent_l687_68701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_trig_value_in_obtuse_angle_l687_68768

-- Define the range for obtuse angles
def ObtuseAngle (y : ℝ) : Prop := Real.pi / 2 < y ∧ y < Real.pi

-- Theorem statement
theorem unique_trig_value_in_obtuse_angle (y : ℝ) (h : ObtuseAngle y) :
  (∃! f : ℝ → ℝ, f ∈ ({Real.sin, Real.cos, Real.tan} : Set (ℝ → ℝ)) ∧ 
    (∀ g : ℝ → ℝ, g ∈ ({Real.sin, Real.cos, Real.tan} : Set (ℝ → ℝ)) → g ≠ f → 
      ∃ z, ObtuseAngle z ∧ g z = f y)) →
  f = Real.sin ∧ f y ≤ 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_trig_value_in_obtuse_angle_l687_68768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_2011_expression_l687_68715

/-- Represents an expression using only the digit 1, sums, subtractions, and multiplications -/
inductive OneDigitExpr : Type
| one : OneDigitExpr
| add : OneDigitExpr → OneDigitExpr → OneDigitExpr
| sub : OneDigitExpr → OneDigitExpr → OneDigitExpr
| mul : OneDigitExpr → OneDigitExpr → OneDigitExpr

/-- Evaluates a OneDigitExpr to an integer -/
def eval : OneDigitExpr → Int
| OneDigitExpr.one => 1
| OneDigitExpr.add e1 e2 => eval e1 + eval e2
| OneDigitExpr.sub e1 e2 => eval e1 - eval e2
| OneDigitExpr.mul e1 e2 => eval e1 * eval e2

/-- Checks if an expression has any repeated equal addends -/
def hasRepeatedAddends : OneDigitExpr → Bool
| OneDigitExpr.one => false
| OneDigitExpr.add e1 e2 => hasRepeatedAddends e1 || hasRepeatedAddends e2
| OneDigitExpr.sub e1 e2 => hasRepeatedAddends e1 || hasRepeatedAddends e2
| OneDigitExpr.mul e1 e2 => hasRepeatedAddends e1 || hasRepeatedAddends e2

/-- Theorem: There exists an expression for 2011 using only the digit 1, 
    sums, subtractions, and multiplications, without repeating equal addends -/
theorem exists_2011_expression : 
  ∃ e : OneDigitExpr, eval e = 2011 ∧ ¬hasRepeatedAddends e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_2011_expression_l687_68715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_contains_all_naturals_l687_68739

def is_club (S : Set ℕ) : Prop :=
  (∃ x, x ∈ S) ∧
  (∀ y ∈ S, (4 * y) ∈ S) ∧
  (∀ y ∈ S, (Nat.sqrt y) ∈ S)

theorem club_contains_all_naturals (S : Set ℕ) (h : is_club S) : S = Set.univ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_contains_all_naturals_l687_68739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_odd_condition_l687_68785

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.cos (2 * x + φ)

def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem cos_odd_condition (φ : ℝ) :
  (φ = π / 2 → is_odd (f φ)) ∧
  (∃ ψ : ℝ, ψ ≠ π / 2 ∧ is_odd (f ψ)) :=
by
  sorry

#check cos_odd_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_odd_condition_l687_68785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_not_equal_48_l687_68772

theorem product_not_equal_48 : ∃! pair : ℚ × ℚ, 
  pair ∈ [(-6, -8), (-4, -12), (1/3, -144), (2, 24), (4/3, 36)] ∧ 
  pair.1 * pair.2 ≠ 48 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_not_equal_48_l687_68772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_acuteness_acute_triangle_l687_68776

theorem triangle_acuteness (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_cube_sum : a^3 + b^3 = c^3) :
  a^2 + b^2 > c^2 := by
  sorry

theorem acute_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_cube_sum : a^3 + b^3 = c^3) :
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  cos_C > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_acuteness_acute_triangle_l687_68776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_coefficient_sum_l687_68781

noncomputable def f (x : ℝ) : ℝ := (3 * x^3 - x^2 + 4 * x - 8) / (x - 2)

theorem slant_asymptote_coefficient_sum :
  ∃ (m b : ℝ), (∀ ε > 0, ∃ N, ∀ x > N, |f x - (m * x + b)| < ε) ∧ m + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_coefficient_sum_l687_68781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l687_68780

/-- A function f defined piecewise on the real line. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else a^x

/-- The theorem stating that if f is decreasing, then a is in the open interval (1/6, 1/3). -/
theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a > 1/6 ∧ a < 1/3 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l687_68780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_heads_probability_l687_68792

/-- Probability of getting heads for the unfair coin -/
noncomputable def p : ℝ := 3/4

/-- Number of coin tosses -/
def n : ℕ := 40

/-- Probability of getting an odd number of heads after k tosses -/
noncomputable def Q (k : ℕ) : ℝ :=
  1/2 * (1 - (1/2)^k)

/-- Main theorem: The probability of getting an odd number of heads
    after n tosses of an unfair coin with probability p of heads -/
theorem odd_heads_probability :
  Q n = 1/2 * (1 - (1/2)^n) := by
  -- The proof is omitted here
  sorry

/-- Lemma: The probability of getting an odd number of heads
    is independent of the probability of getting heads on a single toss -/
lemma odd_heads_independent_of_p :
  ∀ k : ℕ, Q k = 1/2 * (1 - (1/2)^k) := by
  -- The proof is omitted here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_heads_probability_l687_68792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_minus_alpha_l687_68798

theorem tan_pi_fourth_minus_alpha (α : Real) (h1 : α > 0) (h2 : α < π) (h3 : Real.sin α = 3/5) :
  Real.tan (π/4 - α) = 1/7 ∨ Real.tan (π/4 - α) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_minus_alpha_l687_68798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_victor_shrimp_count_l687_68751

/-- The number of shrimp Victor's trap caught -/
def V : ℕ := 26

/-- The number of shrimp Austin's trap caught -/
def austin_shrimp : ℕ := V - 8

/-- The number of shrimp Brian's trap caught -/
def brian_shrimp : ℕ := (V + austin_shrimp) / 2

/-- The total number of shrimp caught by all three boys -/
def total_shrimp : ℕ := V + austin_shrimp + brian_shrimp

/-- The price of shrimp in dollars per 11 tails -/
def price_per_11_tails : ℕ := 7

/-- The earnings of each boy in dollars -/
def earnings_per_boy : ℕ := 14

/-- The total earnings from selling the shrimp in dollars -/
def total_earnings : ℕ := 3 * earnings_per_boy

theorem victor_shrimp_count : V = 26 := by
  rfl

#eval V
#eval austin_shrimp
#eval brian_shrimp
#eval total_shrimp
#eval total_earnings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_victor_shrimp_count_l687_68751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_face_area_specific_l687_68745

/-- The total area of the four triangular faces of a right, square-based pyramid -/
noncomputable def pyramid_face_area (base_edge : ℝ) (lateral_edge : ℝ) : ℝ :=
  4 * (1/2 * base_edge * Real.sqrt (lateral_edge^2 - (base_edge/2)^2))

/-- Theorem: The total area of the four triangular faces of a right, square-based pyramid
    with base edges of 8 units and lateral edges of 10 units is equal to 32√21 square units -/
theorem pyramid_face_area_specific :
  pyramid_face_area 8 10 = 32 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_face_area_specific_l687_68745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_rectangles_in_square_l687_68762

/-- A rectangle type -/
structure Rectangle where
  -- Add necessary fields and properties for a rectangle
  width : ℝ
  height : ℝ
  area : ℝ := width * height

/-- A square type -/
structure Square where
  -- Add necessary fields and properties for a square
  side : ℝ
  area : ℝ := side * side

/-- Predicate for a rectangle being inscribed in a square -/
def Rectangle.inscribed_in (R : Rectangle) (S : Square) : Prop :=
  R.width ≤ S.side ∧ R.height ≤ S.side

/-- Predicate for similarity between two rectangles -/
def Rectangle.similar_to (R₁ R₂ : Rectangle) : Prop :=
  R₁.width / R₁.height = R₂.width / R₂.height

/-- Given a rectangle and a square, there exist two similar rectangles inscribed in the square -/
theorem similar_rectangles_in_square (R : Rectangle) (S : Square) :
  ∃ (ABCD A₁B₁C₁D₁ : Rectangle),
    ABCD.inscribed_in S ∧
    A₁B₁C₁D₁.inscribed_in S ∧
    ABCD.similar_to R ∧
    A₁B₁C₁D₁.similar_to R :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_rectangles_in_square_l687_68762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_optimal_choice_l687_68706

-- Define the probability function for Carol's win
noncomputable def carol_win_prob (c : ℝ) : ℝ :=
  if c < 1/2 then c
  else if c ≤ 2/3 then -12*c^2 + 12*c - 3
  else 1 - c

-- Define Alice's and Bob's intervals
def alice_interval : Set ℝ := Set.Icc 0 1
def bob_interval : Set ℝ := Set.Icc (1/2) (2/3)

-- Theorem statement
theorem carol_optimal_choice :
  ∃ (max_c : ℝ), max_c = 13/24 ∧
  ∀ (c : ℝ), c ∈ Set.Icc 0 1 → carol_win_prob c ≤ carol_win_prob max_c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_optimal_choice_l687_68706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_sum_seven_l687_68753

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def ceil (x : ℝ) : ℤ := ⌈x⌉

theorem floor_ceil_sum_seven (x : ℝ) : floor x + ceil x = 7 ↔ 3 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_sum_seven_l687_68753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l687_68761

-- Define the function f
noncomputable def f (x : Real) : Real :=
  let m : Real × Real := (1, Real.sin (2 * x))
  let n : Real × Real := (Real.cos (2 * x), Real.sqrt 3)
  m.1 * n.1 + m.2 * n.2

-- Define the triangle ABC
structure Triangle where
  a : Real
  b : Real
  c : Real
  A : Real
  B : Real
  C : Real

-- State the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : f abc.A = 1)
  (h2 : abc.a = Real.sqrt 3)
  (h3 : abc.b + abc.c = 3) :
  abc.A = Real.pi / 3 ∧ 
  (1/2 * abc.b * abc.c * Real.sin abc.A) = Real.sqrt 3 / 2 := by
  sorry

#check triangle_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l687_68761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_approximately_0_085_l687_68713

-- Define the regular decagon
def regular_decagon_side_length : ℝ := 1

-- Define the probability calculation function
noncomputable def probability_dart_in_square (decagon_side : ℝ) : ℝ :=
  let decagon_radius := decagon_side / (2 * Real.sin (Real.pi / 10))
  let decagon_area := (10 / 4) * Real.tan (Real.pi / 5) * decagon_radius ^ 2
  let square_side := 0.8090  -- Approximation based on geometry
  let square_area := square_side ^ 2
  square_area / decagon_area

-- Theorem statement
theorem dart_probability_approximately_0_085 :
  ∃ ε > 0, abs (probability_dart_in_square regular_decagon_side_length - 0.085) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_approximately_0_085_l687_68713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_line_l687_68764

-- Define the set of points
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ u : ℝ, p.1 = (Real.cos (2*u))^2 ∧ p.2 = (Real.sin (2*u))^2}

-- Theorem statement
theorem points_form_line :
  ∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ ∀ p ∈ S, a * p.1 + b * p.2 = c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_line_l687_68764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l687_68775

noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 6 + Real.cos x ^ 4

theorem g_range : ∀ x : ℝ, 1 ≤ g x ∧ g x ≤ g ((Real.sqrt 7 - 1) / 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l687_68775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_coefficient_of_friction_l687_68766

/-- Coefficient of friction between a rod and a surface -/
noncomputable def coefficient_of_friction (tilt_angle : ℝ) (normal_force_factor : ℝ) : ℝ :=
  let sin_tilt := Real.sin (tilt_angle * Real.pi / 180)
  let cos_tilt := Real.cos (tilt_angle * Real.pi / 180)
  (sin_tilt * cos_tilt - normal_force_factor * (sin_tilt^2 - cos_tilt^2)) /
  (normal_force_factor * (sin_tilt + cos_tilt))

/-- Theorem stating the coefficient of friction for the given conditions -/
theorem rod_coefficient_of_friction :
  ∃ ε > 0, |coefficient_of_friction 85 6 - 0.08| < ε := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_coefficient_of_friction_l687_68766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fractional_zeta_even_l687_68720

-- Define the zeta function
noncomputable def zeta (x : ℝ) : ℝ := ∑' n, (1 : ℝ) / n^x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- Theorem statement
theorem sum_fractional_zeta_even :
  (∑' k : ℕ, frac (zeta (2 * ↑k))) = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fractional_zeta_even_l687_68720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_roots_of_composite_l687_68779

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_roots_of_composite (h1 : a ≠ 0) (h2 : ∀ x, f a b c x ≠ 2 * x) :
  ∀ x, f a b c (f a b c x) ≠ 4 * x :=
by
  intro x
  by_cases h : a > 0
  · -- Case: a > 0
    have h3 : ∀ y, f a b c y > 2 * y := by sorry
    have h4 : f a b c (f a b c x) > 2 * (f a b c x) := h3 (f a b c x)
    have h5 : 2 * (f a b c x) > 4 * x := by sorry
    have h6 : f a b c (f a b c x) > 4 * x := by sorry
    exact ne_of_gt h6
  · -- Case: a < 0
    have h7 : a < 0 := by sorry
    have h8 : ∀ y, f a b c y < 2 * y := by sorry
    have h9 : f a b c (f a b c x) < 2 * (f a b c x) := h8 (f a b c x)
    have h10 : 2 * (f a b c x) < 4 * x := by sorry
    have h11 : f a b c (f a b c x) < 4 * x := by sorry
    exact ne_of_lt h11


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_roots_of_composite_l687_68779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_integer_product_exceeds_2000_l687_68783

theorem smallest_odd_integer_product_exceeds_2000 :
  ∃ (n : ℕ), 
    Odd n ∧ 
    (∀ m : ℕ, Odd m → m < n → (2 : ℝ)^((m + 1)^2 / 9) ≤ 2000) ∧
    (2 : ℝ)^((n + 1)^2 / 9) > 2000 ∧
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_integer_product_exceeds_2000_l687_68783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_at_y_equals_180_l687_68746

-- Define the @ operation
def atOp (a b : ℝ) : ℝ := a * (b ^ (1/2))

-- Define x and y
noncomputable def x : ℝ := (2 * 3) ^ 2
noncomputable def y : ℝ := (3 * 5) ^ 2 / 9

-- Theorem statement
theorem x_at_y_equals_180 : atOp x y = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_at_y_equals_180_l687_68746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l687_68791

/-- Calculates the length of a bridge given the train's length, speed, and time to cross. -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_cross : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * time_to_cross
  total_distance - train_length

/-- Theorem stating that under the given conditions, the bridge length is approximately 170.31 meters. -/
theorem bridge_length_calculation :
  let train_length : ℝ := 120
  let train_speed_kmph : ℝ := 60
  let time_to_cross : ℝ := 17.39860811135109
  let calculated_length := bridge_length train_length train_speed_kmph time_to_cross
  ∃ ε > 0, |calculated_length - 170.31| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l687_68791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_negative_in_fourth_quadrant_l687_68756

/-- An angle in the fourth quadrant -/
def fourth_quadrant (a : ℝ) : Prop := 3 * Real.pi / 2 < a ∧ a < 2 * Real.pi

theorem tan_half_negative_in_fourth_quadrant (a : ℝ) 
  (h : fourth_quadrant a) : Real.tan (a / 2) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_negative_in_fourth_quadrant_l687_68756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johnson_family_seating_theorem_l687_68752

/-- The number of ways to arrange 6 boys and 4 girls in a row of 10 chairs
    such that at least 3 boys are next to each other -/
def johnson_family_seating_arrangements : ℕ :=
  Nat.factorial 10 - (Nat.factorial 9 / (Nat.factorial 4 * Nat.factorial 5) * 2^3)

/-- Theorem stating that the number of seating arrangements for the Johnson family
    with at least 3 boys next to each other is 3627792 -/
theorem johnson_family_seating_theorem :
  johnson_family_seating_arrangements = 3627792 := by
  sorry

#eval johnson_family_seating_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johnson_family_seating_theorem_l687_68752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_l687_68749

theorem exponential_equation (x : ℝ) :
  (2:ℝ)^x - (4:ℝ)^x + (2:ℝ)^(-x) - (4:ℝ)^(-x) = 3 →
  (8:ℝ)^x + 3 * (2:ℝ)^x + (8:ℝ)^(-x) + 3 * (2:ℝ)^(-x) = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_l687_68749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_temperature_l687_68763

/-- Temperatures for each day of the week -/
structure WeekTemperatures where
  monday : ℚ
  tuesday : ℚ
  wednesday : ℚ
  thursday : ℚ
  friday : ℚ

/-- The average temperature of four consecutive days -/
def average4Days (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem friday_temperature (temps : WeekTemperatures) 
  (h1 : average4Days temps.monday temps.tuesday temps.wednesday temps.thursday = 48)
  (h2 : average4Days temps.tuesday temps.wednesday temps.thursday temps.friday = 46)
  (h3 : temps.monday = 43) :
  temps.friday = 35 := by
  sorry

#eval average4Days 1 2 3 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_temperature_l687_68763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tetrahedra_l687_68702

/-- Given n points in space (n ≥ 5) with no four points coplanar, 
    connected by several lines forming m triangles, 
    the number of tetrahedra is at least 1/2[m(n-3) - C(n,4)] -/
theorem min_tetrahedra (n m : ℕ) (h1 : n ≥ 5) (h2 : m > 0) : 
  ∃ x : ℕ, x ≥ (m * (n - 3) - Nat.choose n 4) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tetrahedra_l687_68702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_equals_cos_l687_68708

open Real

noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => cos
  | n + 1 => deriv (f n)

theorem f_2012_equals_cos : f 2012 = cos := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_equals_cos_l687_68708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_OP_l687_68718

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let ab := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let bc := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let ac := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  ab = 13 ∧ bc = 5 ∧ ac = 12

-- Define a circle
def Circle (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ x => (x.1 - center.1)^2 + (x.2 - center.2)^2 = (point.1 - center.1)^2 + (point.2 - center.2)^2

-- Define a line
def Line (p q : ℝ × ℝ) : ℝ → ℝ × ℝ :=
  λ t => (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))

-- Define tangency
def Tangent (line : ℝ → ℝ × ℝ) (circle : ℝ × ℝ → Prop) (point : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), circle (line t) ∧ ∀ (s : ℝ), s ≠ t → ¬ circle (line s)

-- Theorem statement
theorem distance_OP (A B C O P : ℝ × ℝ) :
  Triangle A B C →
  Tangent (Line A B) (Circle O C) A →
  Tangent (Line B C) (Circle P C) B →
  (O.1 - P.1)^2 + (O.2 - P.2)^2 = 48.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_OP_l687_68718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_sum_24_with_3_l687_68765

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem factor_sum_24_with_3 (x : ℕ) (h1 : x > 0) (h2 : sum_of_factors x = 24) (h3 : 3 ∣ x) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_sum_24_with_3_l687_68765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_rate_is_six_l687_68727

/-- The rate of a man rowing in still water, given his speeds with and against the stream. -/
noncomputable def mans_rate (speed_with_stream speed_against_stream : ℝ) : ℝ :=
  (speed_with_stream + speed_against_stream) / 2

/-- Theorem stating that the man's rate in still water is 6 km/h. -/
theorem mans_rate_is_six :
  mans_rate 8 4 = 6 := by
  unfold mans_rate
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_rate_is_six_l687_68727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l687_68723

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The left focus of an ellipse -/
noncomputable def left_focus (e : Ellipse) : ℝ × ℝ := 
  (-Real.sqrt (e.a^2 - e.b^2), 0)

/-- The symmetric point of a point with respect to the line √3x + y = 0 -/
noncomputable def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  let m := (p.1 + Real.sqrt 3 * p.2) / 2
  let n := (Real.sqrt 3 * p.1 - p.2) / 2
  (m, n)

/-- A point lies on the ellipse -/
def on_ellipse (e : Ellipse) (p : ℝ × ℝ) : Prop :=
  p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1

theorem ellipse_eccentricity (e : Ellipse) : 
  on_ellipse e (symmetric_point (left_focus e)) → 
  eccentricity e = Real.sqrt 3 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l687_68723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_l687_68769

/-- Given a triangle ABC where:
  - Sides opposite to angles A, B, C are a, b, c respectively
  - sin B = 2 sin A
  - The area of triangle ABC is a^2 * sin B
  Prove that cos B = 1/4 -/
theorem triangle_cosine (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  A > 0 → B > 0 → C > 0 → 
  A + B + C = Real.pi →
  Real.sin B = 2 * Real.sin A →
  (1/2) * a * c * Real.sin B = a^2 * Real.sin B →
  Real.cos B = 1/4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_l687_68769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_of_circle_and_sphere_l687_68759

-- Define the sphere
def sphere (h k l R : ℝ) (x y z : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 + (z - l)^2 = R^2

-- Define the circle (renamed to avoid conflict)
def parallelCircle (x₀ y₀ z₀ r : ℝ) (x y z : ℝ) : Prop :=
  x = x₀ ∧ (y - y₀)^2 + (z - z₀)^2 = r^2

-- Theorem statement
theorem intersection_points_of_circle_and_sphere
  (h k l R c x₀ y₀ z₀ r : ℝ) :
  ∀ y z : ℝ,
  (sphere h k l R c y z ∧ parallelCircle x₀ y₀ z₀ r c y z) ↔
  ((y - k)^2 + (z - l)^2 = R^2 - (c - h)^2 ∧
   (y - y₀)^2 + (z - z₀)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_of_circle_and_sphere_l687_68759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_game_cost_is_830_l687_68747

/-- Jitka's daily wage in Kč -/
def daily_wage : ℝ := sorry

/-- Cost of one board game in Kč -/
def board_game_cost : ℝ := sorry

/-- Condition: In 3 days, Jitka can buy 1 board game and have 490 Kč left -/
axiom three_day_condition : 3 * daily_wage = board_game_cost + 490

/-- Condition: In 5 days, Jitka can buy 2 board games and have 540 Kč left -/
axiom five_day_condition : 5 * daily_wage = 2 * board_game_cost + 540

/-- Theorem: The board game costs 830 Kč -/
theorem board_game_cost_is_830 : board_game_cost = 830 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_game_cost_is_830_l687_68747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_condition_l687_68730

/-- The curve y = ax^2 + ln x has a tangent line at (1, a) parallel to y = 2x iff a = 1/2 -/
theorem tangent_parallel_condition (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x, f x = a * x^2 + Real.log x) →
  (∃ g : ℝ → ℝ, ∀ x, g x = 2 * x) →
  (∃ h : ℝ → ℝ, ∀ x, h x = 2 * a * x + 1 / x) →
  (((λ x ↦ 2 * a * x + 1 / x) 1) = ((λ x ↦ 2 * x) 1)) ↔
  a = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_condition_l687_68730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_growth_l687_68726

theorem sequence_growth (a : ℕ → ℕ+) 
  (h : ∀ i : ℕ, (Nat.gcd (a (i + 1)).val (a (i + 2)).val : ℕ) > (a i).val) :
  ∀ n : ℕ, (a n).val ≥ 2^n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_growth_l687_68726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l687_68712

-- Define the quadratic function
noncomputable def f (m x : ℝ) : ℝ := -1/2 * (x - 2*m)^2 + 3 - m

-- Define the line y = -1/2x + 3
noncomputable def line (x : ℝ) : ℝ := -1/2 * x + 3

-- Theorem statement
theorem quadratic_properties (m a c : ℝ) 
  (h1 : f m (a+1) = c) 
  (h2 : f m (4*m-5+a) = c) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f 2 x1 = 0 ∧ f 2 x2 = 0) ∧ 
  (∃ x : ℝ, f m x = line x ∧ ∀ y : ℝ, f m y ≤ f m x) ∧
  c ≤ 13/8 := by
  sorry

#check quadratic_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l687_68712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_overall_gain_l687_68703

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * years)

def loan_AB : ℝ := 10000
def rate_AB : ℝ := 0.08
def years_AB : ℝ := 4

def loan_BC : ℝ := 4000
def rate_BC : ℝ := 0.10
def compounds_BC : ℝ := 2
def years_BC : ℝ := 3

def loan_BD : ℝ := 3000
def rate_BD : ℝ := 0.12
def compounds_BD : ℝ := 4
def years_BD : ℝ := 2

def loan_BE : ℝ := 3000
def rate_BE : ℝ := 0.115
def years_BE : ℝ := 4

theorem B_overall_gain :
  let amount_AB := compound_interest loan_AB rate_AB 1 years_AB
  let amount_BC := compound_interest loan_BC rate_BC compounds_BC years_BC
  let amount_BD := compound_interest loan_BD rate_BD compounds_BD years_BD
  let amount_BE := compound_interest loan_BE rate_BE 1 years_BE
  ∃ ε > 0, |amount_BC + amount_BD + amount_BE - amount_AB - 280.46| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_overall_gain_l687_68703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l687_68731

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (-(π/3) + k * π) ((π/6) + k * π))) ∧
  (∃ xmax xmin : ℝ, xmax ∈ Set.Icc 0 (π/2) ∧ xmin ∈ Set.Icc 0 (π/2) ∧
    (∀ x : ℝ, x ∈ Set.Icc 0 (π/2) → f x ≤ f xmax) ∧
    (∀ x : ℝ, x ∈ Set.Icc 0 (π/2) → f xmin ≤ f x) ∧
    f xmax = 1/2 ∧ f xmin = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l687_68731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_oslash_equation_l687_68771

-- Define the oslash operation
noncomputable def oslash (a b : ℝ) : ℝ := (Real.sqrt (3 * a + b)) ^ 4

-- Theorem statement
theorem solve_oslash_equation (x : ℝ) :
  oslash 5 x = 256 → x = 1 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_oslash_equation_l687_68771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_150_deg_l687_68719

/-- The area of a triangle given two sides and the included angle -/
noncomputable def triangleArea (a b : ℝ) (θ : ℝ) : ℝ :=
  1/2 * a * b * Real.sin θ

/-- Theorem: The area of a triangle with sides 8 and 12, and included angle 150°, is 24 -/
theorem triangle_area_150_deg : 
  triangleArea 8 12 (150 * π / 180) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_150_deg_l687_68719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hunter_cannot_always_catch_rabbit_l687_68790

/-- Represents a point in the Euclidean plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The game state after n rounds -/
structure GameState where
  n : ℕ
  rabbit_pos : Point
  hunter_pos : Point

/-- A strategy for the rabbit -/
def RabbitStrategy := GameState → Point

/-- A strategy for the hunter -/
def HunterStrategy := GameState → Point → Point

/-- The tracking device's report -/
def TrackingDevice := Point → Point

/-- Simulate one round of the game -/
def playRound (state : GameState) (rabbit_strategy : RabbitStrategy) 
              (hunter_strategy : HunterStrategy) (tracking_device : TrackingDevice) : GameState :=
  let new_rabbit_pos := rabbit_strategy state
  let tracked_pos := tracking_device new_rabbit_pos
  let new_hunter_pos := hunter_strategy state tracked_pos
  { n := state.n + 1, rabbit_pos := new_rabbit_pos, hunter_pos := new_hunter_pos }

/-- Play the game for a given number of rounds -/
def playGame (rounds : ℕ) (rabbit_strategy : RabbitStrategy) 
             (hunter_strategy : HunterStrategy) (tracking_device : TrackingDevice) : GameState :=
  (List.range rounds).foldl
    (λ state _ => playRound state rabbit_strategy hunter_strategy tracking_device)
    { n := 0, rabbit_pos := ⟨0, 0⟩, hunter_pos := ⟨0, 0⟩ }

theorem hunter_cannot_always_catch_rabbit :
  ∃ (rabbit_strategy : RabbitStrategy),
    ∀ (hunter_strategy : HunterStrategy) (tracking_device : TrackingDevice),
      let final_state := playGame (10^9) rabbit_strategy hunter_strategy tracking_device
      distance final_state.rabbit_pos final_state.hunter_pos > 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hunter_cannot_always_catch_rabbit_l687_68790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_acid_in_solution_l687_68717

/-- Given a solution with a total volume and an acid percentage, 
    calculate the amount of pure acid in the solution. -/
noncomputable def pure_acid_amount (total_volume : ℝ) (acid_percentage : ℝ) : ℝ :=
  total_volume * (acid_percentage / 100)

/-- Theorem: The amount of pure acid in 12 litres of a 40% solution is 4.8 litres -/
theorem pure_acid_in_solution : pure_acid_amount 12 40 = 4.8 := by
  -- Unfold the definition of pure_acid_amount
  unfold pure_acid_amount
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- The result should now be obvious to Lean
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_acid_in_solution_l687_68717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l687_68789

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 3 - 2 / 5 * x)

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = 5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l687_68789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersect_centers_radii_l687_68770

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem: Circles O₁ and O₂ intersect -/
theorem circles_intersect : ∃ (x y : ℝ),
  ((x - 1)^2 + y^2 = 1) ∧ (x^2 + (y - 3)^2 = 9) := by
  sorry

/-- Proof that the circles intersect based on their centers and radii -/
theorem circles_intersect_centers_radii :
  let d := distance 1 0 0 3  -- Distance between centers
  let r1 := 1  -- Radius of O₁
  let r2 := 3  -- Radius of O₂
  d > r2 - r1 ∧ d < r1 + r2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersect_centers_radii_l687_68770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_and_zeros_l687_68732

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x - a) / x - a * Real.log x

theorem tangent_line_and_monotonicity_and_zeros (a : ℝ) :
  let f := f a
  (∀ x, x > 0 → x ≠ 1 → (f x - (Real.exp 1 - a)) / (x - 1) ≠ 0) ∧
  (∀ x h, 0 < x → x < 1 → a ≤ 1 → h ≠ 0 → (f (x + h) - f x) / h < 0) ∧
  (∀ x h, x > 1 → h ≠ 0 → (f (x + h) - f x) / h > 0) ∧
  (a ≥ Real.exp 1 → ∃! x, x > 0 ∧ f x = 0) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_and_zeros_l687_68732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_and_line_check_l687_68767

-- Define the function h(x) as noncomputable
noncomputable def h (x : ℝ) : ℝ := 4.125 - (x + 0.5)^2 / 2

-- State the theorem
theorem intersection_point_and_line_check :
  let a : ℝ := -1.5
  let b : ℝ := 3.875
  (h a = h (a - 4) + 1) ∧ 
  (h a = b) ∧
  (b ≠ -a) := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_and_line_check_l687_68767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l687_68729

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)

theorem phi_value (φ : ℝ) (h1 : |φ| < π/2) 
  (h2 : f φ (π/12 - π/6) = -Real.sqrt 2) : φ = -π/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l687_68729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_diameter_l687_68743

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the endpoints of the diameter
def endpoint1 : ℝ × ℝ := (2, 0)
def endpoint2 : ℝ × ℝ := (2, -2)

-- Define the circle based on the diameter endpoints
noncomputable def circleFromDiameter (p1 p2 : ℝ × ℝ) : Circle :=
  { center := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2),
    radius := Real.sqrt (((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) / 4) }

-- Define the circle equation
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_equation_from_diameter :
  let c := circleFromDiameter endpoint1 endpoint2
  ∀ x y : ℝ, circleEquation c x y ↔ (x - 2)^2 + (y + 1)^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_diameter_l687_68743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_example_l687_68750

/-- The sum of an infinite geometric series with first term a and common ratio r where |r| < 1 -/
noncomputable def geometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Proof that the sum of the geometric series 2 + (1/3) + (1/3)² + (1/3)³ + ... is equal to 3 -/
theorem geometric_series_sum_example : geometricSeriesSum 2 (1/3) = 3 := by
  -- Unfold the definition of geometricSeriesSum
  unfold geometricSeriesSum
  -- Simplify the expression
  simp
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_example_l687_68750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l687_68786

theorem circular_table_seating (seating_ways : ℕ) :
  (seating_ways = 144) → (∃ n : ℕ, n > 5 ∧ Nat.factorial (n - 1) = seating_ways) →
  (∃ group_size : ℕ, group_size = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l687_68786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_distance_equality_l687_68787

/-- A rectangle in a 2D plane --/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Distance between two points in a 2D plane --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: For any rectangle and any point on the plane, the sum of squares of distances
    from the point to two opposite vertices equals the sum of squares of distances
    from the point to the other two vertices --/
theorem rectangle_distance_equality (rect : Rectangle) (M : ℝ × ℝ) :
  (distance M rect.A)^2 + (distance M rect.C)^2 =
  (distance M rect.B)^2 + (distance M rect.D)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_distance_equality_l687_68787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_points_count_l687_68709

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the height of a tetrahedron from point D to the base triangle ABC -/
noncomputable def tetrahedronHeight (t : Tetrahedron) : ℝ := sorry

/-- Calculates the area of a face of a tetrahedron -/
noncomputable def faceArea (t : Tetrahedron) (face : Fin 4) : ℝ := sorry

/-- Counts the number of points D that satisfy the given conditions -/
noncomputable def countValidPoints (triangle : Triangle3D) (h s₁ s₂ : ℝ) : Finset ℕ :=
  let validPoints := { D : Point3D |
    let tetra := Tetrahedron.mk triangle.A triangle.B triangle.C D
    tetrahedronHeight tetra = h ∧
    faceArea tetra 2 = s₁ ∧
    faceArea tetra 3 = s₂ }
  sorry -- This should return a Finset ℕ containing the possible counts

/-- Theorem stating that the number of valid points D is 0, 2, 4, or 8 -/
theorem valid_points_count (triangle : Triangle3D) (h s₁ s₂ : ℝ) 
    (h_pos : h > 0) (s₁_pos : s₁ > 0) (s₂_pos : s₂ > 0) :
    (countValidPoints triangle h s₁ s₂) ⊆ {0, 2, 4, 8} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_points_count_l687_68709
