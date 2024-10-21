import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_paycheck_is_674_l36_3625

-- Define the number of paychecks and their amounts
def total_paychecks : ℕ := 26
noncomputable def initial_paycheck : ℚ := 750
def first_raise : ℚ := 5 / 100
def second_raise : ℚ := 3 / 100
def bonus : ℚ := 250
def deductible : ℚ := 120
def tax_rate : ℚ := 15 / 100

-- Define the function to calculate the average paycheck
noncomputable def average_paycheck : ℚ :=
  let first_period := 6 * initial_paycheck
  let second_period := 10 * (initial_paycheck * (1 + first_raise))
  let third_period := 10 * (initial_paycheck * (1 + first_raise) * (1 + second_raise))
  let total_before_adjustments := first_period + second_period + third_period
  let total_with_adjustments := total_before_adjustments + bonus - deductible
  let total_after_tax := total_with_adjustments * (1 - tax_rate)
  total_after_tax / total_paychecks

-- Theorem statement
theorem average_paycheck_is_674 :
  Int.floor average_paycheck = 674 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_paycheck_is_674_l36_3625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_hyperbola_line_slope_l36_3613

-- Define the hyperbola
def hyperbola (b : ℝ) (x y : ℝ) : Prop := x^2 - y^2/b^2 = 1 ∧ b > 0

-- Define the foci
noncomputable def foci (b : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let c := Real.sqrt (1 + b^2)
  (-c, 0, c, 0)

-- Define a line passing through a point with a given slope
def line_through_point (x0 y0 k : ℝ) (x y : ℝ) : Prop :=
  y - y0 = k * (x - x0)

-- Define an equilateral triangle
def is_equilateral_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (x1 - x2)^2 + (y1 - y2)^2 = (x2 - x3)^2 + (y2 - y3)^2 ∧
  (x2 - x3)^2 + (y2 - y3)^2 = (x3 - x1)^2 + (y3 - y1)^2

-- Part 1
theorem hyperbola_asymptotes (b : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    hyperbola b x1 y1 ∧ hyperbola b x2 y2 ∧
    let (x_f1, y_f1, x_f2, y_f2) := foci b
    line_through_point x_f2 y_f2 (Real.tan (π/2)) x1 y1 ∧
    line_through_point x_f2 y_f2 (Real.tan (π/2)) x2 y2 ∧
    is_equilateral_triangle x_f1 y_f1 x1 y1 x2 y2) →
  (∀ x y : ℝ, y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x) := by
  sorry

-- Part 2
theorem hyperbola_line_slope (x1 y1 x2 y2 : ℝ) :
  (hyperbola (Real.sqrt 3) x1 y1 ∧ hyperbola (Real.sqrt 3) x2 y2 ∧
   let (x_f1, y_f1, x_f2, y_f2) := foci (Real.sqrt 3)
   let v1 := (x1 - x_f1, y1 - y_f1)
   let v2 := (x2 - x_f1, y2 - y_f1)
   let v3 := (x2 - x1, y2 - y1)
   (v1.1 + v2.1) * v3.1 + (v1.2 + v2.2) * v3.2 = 0) →
  (∃ k : ℝ, k = Real.sqrt 15 / 5 ∨ k = -Real.sqrt 15 / 5 ∧
   line_through_point x_f2 y_f2 k x1 y1 ∧
   line_through_point x_f2 y_f2 k x2 y2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_hyperbola_line_slope_l36_3613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l36_3601

-- Define the curve
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 6 * x + 2

-- Theorem statement
theorem tangent_line_parallel (a : ℝ) :
  f 1 = 5 →  -- Point (1, 5) is on the curve
  (λ x ↦ f' 1 * (x - 1) + f 1) = (λ x ↦ 2 * a * x - 2017) →  -- Tangent line is parallel to 2ax - y - 2017 = 0
  a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l36_3601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angle_mean_l36_3692

/-- A quadrilateral is a polygon with four sides and four interior angles. -/
structure Quadrilateral where
  angles : Fin 4 → ℝ

/-- The sum of interior angles in a quadrilateral is always 360°. -/
def sum_of_angles (q : Quadrilateral) : ℝ :=
  (q.angles 0) + (q.angles 1) + (q.angles 2) + (q.angles 3)

/-- Axiom: The sum of interior angles in a quadrilateral is always 360°. -/
axiom quadrilateral_angle_sum (q : Quadrilateral) : sum_of_angles q = 360

/-- The mean value of the measures of the four interior angles of any quadrilateral is 90°. -/
theorem quadrilateral_angle_mean :
  ∀ (q : Quadrilateral), (sum_of_angles q) / 4 = 90 :=
by
  intro q
  have h : sum_of_angles q = 360 := quadrilateral_angle_sum q
  rw [h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angle_mean_l36_3692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_grid_area_l36_3616

/-- The side length of the square -/
noncomputable def square_side : ℝ := 5

/-- The side length of an equilateral triangle in the grid -/
noncomputable def triangle_side : ℝ := square_side / 2

/-- The height of an equilateral triangle in the grid -/
noncomputable def triangle_height : ℝ := (Real.sqrt 3 / 2) * triangle_side

/-- The area of one equilateral triangle in the grid -/
noncomputable def triangle_area : ℝ := (1 / 2) * triangle_side * triangle_height

/-- The number of equilateral triangles that fit in the square -/
def triangle_count : ℕ := 8

/-- The total area of all equilateral triangles in the grid -/
noncomputable def total_area : ℝ := triangle_count * triangle_area

theorem equilateral_triangle_grid_area :
  total_area = 12.5 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_grid_area_l36_3616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_expressions_l36_3664

theorem odd_expressions (m n : ℕ) (hm : Odd m) (hn : Even n) :
  Odd (m + n) ∧ Odd (m^2 - 3*n) ∧ Odd (5*m^2 + 7*n^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_expressions_l36_3664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l36_3696

/-- The focus of the parabola x² = 4y has coordinates (0, 1) -/
theorem parabola_focus_coordinates :
  let parabola := {(x, y) : ℝ × ℝ | x^2 = 4*y}
  ∃ (f : ℝ × ℝ), f = (0, 1) ∧ f ∈ parabola :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l36_3696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l36_3606

/-- The time taken for a train to cross a bridge -/
noncomputable def time_to_cross_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_length + bridge_length
  total_distance / train_speed_mps

/-- Theorem: A train 165 meters long running at 54 kmph takes approximately 52.67 seconds to cross a bridge 625 meters long -/
theorem train_crossing_bridge :
  let train_length : ℝ := 165
  let train_speed_kmph : ℝ := 54
  let bridge_length : ℝ := 625
  let crossing_time := time_to_cross_bridge train_length train_speed_kmph bridge_length
  (crossing_time ≥ 52.66) ∧ (crossing_time ≤ 52.68) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l36_3606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l36_3628

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.tan (ω * x + Real.pi / 3)

theorem omega_value (ω : ℝ) (h1 : 2 < ω) (h2 : ω < 10) :
  (∀ x, f ω x = f ω (x - Real.pi / 6)) → ω = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l36_3628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equals_one_fourth_l36_3638

theorem trig_expression_equals_one_fourth :
  (Real.sin (10 * π / 180)) / (1 - Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equals_one_fourth_l36_3638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_positive_integers_reducible_to_one_l36_3604

-- Define the allowed operations
def divideByTwo (n : ℕ) : ℕ := n / 2

def multiplyByThreePowerPlusOne (n k : ℕ) : ℕ := n * (3^k) + 1

-- Define a predicate that checks if a number can be reduced to 1
def canReduceToOne (n : ℕ) : Prop :=
  ∃ (seq : List ℕ), 
    seq.head? = some n ∧ 
    seq.getLast? = some 1 ∧
    ∀ (i : ℕ) (a b : ℕ), i < seq.length - 1 → 
      seq.get? i = some a → seq.get? (i+1) = some b →
      (a % 2 = 0 ∧ b = divideByTwo a) ∨
      (∃ k : ℕ, k > 0 ∧ b = multiplyByThreePowerPlusOne a k)

-- The main theorem
theorem all_positive_integers_reducible_to_one :
  ∀ (n : ℕ), n > 0 → canReduceToOne n := by
  sorry

#check all_positive_integers_reducible_to_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_positive_integers_reducible_to_one_l36_3604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_length_l36_3619

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * (x - 1)

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem mn_length (k : ℝ) (x1 y1 x2 y2 : ℝ) :
  k > 4/3 →
  circle_C x1 y1 →
  circle_C x2 y2 →
  line_l x1 y1 k →
  line_l x2 y2 k →
  dot_product x1 y1 x2 y2 = 12 →
  (x1 - x2)^2 + (y1 - y2)^2 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_length_l36_3619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l36_3674

theorem angle_in_third_quadrant (α : ℝ) :
  (Real.tan α > 0 ∧ Real.cos α < 0) → 
  (Real.sin α < 0 ∧ Real.cos α < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l36_3674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mollys_current_age_l36_3672

/-- Proves that Molly's current age is 45 years given the conditions -/
theorem mollys_current_age (sandy_age molly_age : ℕ) : 
  (sandy_age : ℚ) / (molly_age : ℚ) = 4 / 3 →
  sandy_age + 6 = 66 →
  molly_age = 45 := by
  intro h1 h2
  -- The proof steps would go here
  sorry

#check mollys_current_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mollys_current_age_l36_3672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fencing_cost_is_correct_l36_3658

/-- Represents the irregular field with its dimensions and fencing costs. -/
structure IrregularField where
  circular_radius : ℝ
  semicircular_diameter : ℝ
  circular_cost_per_meter : ℝ
  semicircular_flat_cost_per_meter : ℝ
  semicircular_curved_cost_per_meter : ℝ

/-- Calculates the total fencing cost for the irregular field. -/
noncomputable def total_fencing_cost (field : IrregularField) : ℝ :=
  let circular_circumference := 2 * Real.pi * field.circular_radius
  let semicircular_radius := field.semicircular_diameter / 2
  let semicircular_curved_length := Real.pi * semicircular_radius
  let semicircular_flat_length := field.semicircular_diameter
  (circular_circumference * field.circular_cost_per_meter) +
  (semicircular_flat_length * field.semicircular_flat_cost_per_meter) +
  (semicircular_curved_length * field.semicircular_curved_cost_per_meter)

/-- Theorem stating that the total fencing cost for the given irregular field is (2300 * π) + 450. -/
theorem total_fencing_cost_is_correct (field : IrregularField)
  (h1 : field.circular_radius = 200)
  (h2 : field.semicircular_diameter = 150)
  (h3 : field.circular_cost_per_meter = 5)
  (h4 : field.semicircular_flat_cost_per_meter = 3)
  (h5 : field.semicircular_curved_cost_per_meter = 4) :
  total_fencing_cost field = 2300 * Real.pi + 450 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fencing_cost_is_correct_l36_3658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l36_3686

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Check if a point is on the ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The foci of an ellipse -/
noncomputable def foci (e : Ellipse) : (Point × Point) :=
  let c := Real.sqrt (e.a^2 - e.b^2)
  (Point.mk (-c) 0, Point.mk c 0)

/-- Dot product of two vectors -/
def dot_product (v w : Point) : ℝ :=
  v.x * w.x + v.y * w.y

theorem ellipse_properties (e : Ellipse) 
  (h_point : on_ellipse e (Point.mk 1 (Real.sqrt 3 / 2)))
  (h_foci : distance (foci e).1 (foci e).2 = 2 * Real.sqrt 3) :
  (∃ (p : Point), on_ellipse e p ∧ 
    p.x > 0 ∧ 
    dot_product (Point.mk (p.x - (foci e).1.x) (p.y - (foci e).1.y))
                (Point.mk (p.x - (foci e).2.x) (p.y - (foci e).2.y)) ≤ 1/4) →
  (∃ (x : ℝ), 0 < x ∧ x ≤ 2 * Real.sqrt 6 / 3 ∧
    on_ellipse e (Point.mk x (Real.sqrt (1 - x^2/4)))) ∧
  (∀ (x : ℝ), on_ellipse e (Point.mk x (Real.sqrt (1 - x^2/4))) →
    e.a = 2 ∧ e.b = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l36_3686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expected_value_binomial_expected_value_6_1_4_l36_3691

/-- A random variable X follows a binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ

/-- The expected value of a random variable -/
noncomputable def expectedValue (X : ℝ → ℝ) : ℝ := sorry

/-- Theorem stating the expected value of a binomial distribution -/
theorem binomial_expected_value (n : ℕ) (p : ℝ) (X : BinomialDistribution n p) :
  expectedValue X.X = n * p := by sorry

/-- Specific case for X ~ B(6, 1/4) -/
theorem binomial_expected_value_6_1_4 (X : BinomialDistribution 6 (1/4)) :
  expectedValue X.X = 3/2 := by
  have h := binomial_expected_value 6 (1/4) X
  rw [h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expected_value_binomial_expected_value_6_1_4_l36_3691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_in_terms_of_b_min_triangle_area_l36_3615

/-- The linear function y = kx + b intersecting positive x and y axes --/
structure LinearFunction where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0
  b_positive : b > 0
  k_negative : k < 0

/-- The area of triangle OAB formed by the linear function --/
noncomputable def triangleArea (f : LinearFunction) : ℝ := (-f.b^2) / (2 * f.k)

/-- The sum of lengths OA and OB --/
noncomputable def lengthSum (f : LinearFunction) : ℝ := -f.b / f.k + f.b

/-- Theorem stating the relationship between k and b --/
theorem k_in_terms_of_b (f : LinearFunction) 
  (h : triangleArea f = lengthSum f + 3) :
  f.k = (2 * f.b - f.b^2) / (2 * (f.b + 3)) ∧ f.b > 2 := by
  sorry

/-- Theorem stating the minimum area of triangle OAB --/
theorem min_triangle_area :
  ∃ (f : LinearFunction), 
    (∀ (g : LinearFunction), triangleArea f ≤ triangleArea g) ∧
    triangleArea f = 7 + 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_in_terms_of_b_min_triangle_area_l36_3615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_form_l36_3660

noncomputable def cbrt (x : ℝ) : ℝ := Real.rpow x (1/3)

theorem cubic_root_form : ∃ (x : ℝ), 
  16 * x^3 - 4 * x^2 - 4 * x - 1 = 0 ∧ 
  x = (cbrt 2 + cbrt 8 + 1) / 16 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_form_l36_3660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_only_odd_l36_3687

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.cos x
def h (x : ℝ) : ℝ := abs x
noncomputable def k (x : ℝ) : ℝ := Real.sin x

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statement
theorem sin_only_odd :
  ¬ is_odd f ∧ ¬ is_odd g ∧ ¬ is_odd h ∧ is_odd k := by
  sorry

#check sin_only_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_only_odd_l36_3687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rahuls_share_l36_3656

/-- Represents the number of days it takes for a person to complete the work alone -/
structure WorkRate where
  days : ℚ

/-- Calculates the fraction of work done by a person in one day -/
def work_per_day (rate : WorkRate) : ℚ :=
  1 / rate.days

/-- Calculates the total work done by two people working together in one day -/
def total_work_per_day (rate1 rate2 : WorkRate) : ℚ :=
  work_per_day rate1 + work_per_day rate2

/-- Calculates the fraction of total work done by a person when working together -/
def work_fraction (rate1 rate2 : WorkRate) : ℚ :=
  work_per_day rate1 / total_work_per_day rate1 rate2

theorem rahuls_share 
  (rahul_rate : WorkRate) 
  (rajesh_rate : WorkRate) 
  (total_payment : ℚ) 
  (h1 : rahul_rate.days = 3)
  (h2 : rajesh_rate.days = 2)
  (h3 : total_payment = 355)
  : work_fraction rahul_rate rajesh_rate * total_payment = 142 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rahuls_share_l36_3656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l36_3641

noncomputable section

/-- Curve parameter -/
def t₀ : ℝ := Real.pi / 3

/-- x-coordinate of the curve -/
def x (a t : ℝ) : ℝ := a * (t - Real.sin t)

/-- y-coordinate of the curve -/
def y (a t : ℝ) : ℝ := a * (1 - Real.cos t)

/-- Slope of the tangent line at t₀ -/
def tangent_slope : ℝ := Real.sqrt 3

/-- Slope of the normal line at t₀ -/
def normal_slope : ℝ := -1 / Real.sqrt 3

/-- Theorem: Tangent and Normal Line Equations -/
theorem tangent_and_normal_equations (a : ℝ) :
  let x₀ := x a t₀
  let y₀ := y a t₀
  (∀ x y : ℝ, y - y₀ = tangent_slope * (x - x₀) ↔ 
    y = Real.sqrt 3 * x - (Real.sqrt 3 * Real.pi / 3 - 2) * a) ∧
  (∀ x y : ℝ, y - y₀ = normal_slope * (x - x₀) ↔ 
    y = -x / Real.sqrt 3 + a * Real.pi / (3 * Real.sqrt 3)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l36_3641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_diagonal_approx_l36_3640

noncomputable section

-- Define the dimensions of the shapes
def rectangular_length : ℝ := 45
def rectangular_width : ℝ := 40
def triangular_base : ℝ := 30
def triangular_height : ℝ := 20
def triangular_perpendicular : ℝ := 50

-- Define the volumes of the shapes
noncomputable def rectangular_volume : ℝ := rectangular_length * rectangular_width * 1
noncomputable def square_volume : ℝ := rectangular_volume
noncomputable def triangular_volume : ℝ := (triangular_base * triangular_height / 2) * triangular_perpendicular

-- Define the total volume
noncomputable def total_volume : ℝ := rectangular_volume + square_volume + triangular_volume

-- Define the side length of the cube
noncomputable def cube_side : ℝ := Real.rpow total_volume (1/3)

-- Define the diagonal length of the cube
noncomputable def cube_diagonal : ℝ := cube_side * Real.sqrt 3

-- Theorem statement
theorem cube_diagonal_approx :
  ∃ ε > 0, |cube_diagonal - 45.89| < ε := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_diagonal_approx_l36_3640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l36_3634

open Real

/-- The function f(x) = (x-2)e^x -/
noncomputable def f (x : ℝ) : ℝ := (x - 2) * exp x

/-- The function g(x) = f(x) + 2e^x - ax^2 -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f x + 2 * exp x - a * x^2

/-- The function h(x) = x -/
def h (x : ℝ) : ℝ := x

/-- The theorem statement -/
theorem range_of_a (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → (g a x₁ - h x₁) * (g a x₂ - h x₂) > 0) ↔ 
  a ≤ 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l36_3634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l36_3695

/-- The rational function g(x) with parameter c -/
noncomputable def g (c : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + c) / (x^2 - x - 6)

/-- Theorem: g(x) has exactly one vertical asymptote iff c = -3 or c = -8 -/
theorem one_vertical_asymptote (c : ℝ) : 
  (∃! x, (x^2 - x - 6 = 0 ∧ x^2 - 2*x + c ≠ 0)) ↔ (c = -3 ∨ c = -8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l36_3695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_proof_l36_3649

theorem sequence_limit_proof : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((3 * (n : ℝ)^2 - 5 * n) / (3 * (n : ℝ)^2 - 5 * n + 7))^(n + 1) - 1| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_proof_l36_3649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bc_length_l36_3600

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 1

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ := sorry

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.A = (0, 1) ∧
  (∃ a : ℝ, t.B = (-a, parabola (-a)) ∧ t.C = (a, parabola a)) ∧
  (t.C.2 - t.B.2 = t.C.1 - t.B.1) ∧  -- BC parallel to y = x
  (area t = 32)

-- The theorem to prove
theorem triangle_bc_length (t : Triangle) 
  (h : triangle_conditions t) : 
  Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bc_length_l36_3600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_divisible_by_396_l36_3679

def special_number (p : Fin 10 → Fin 10) : ℕ :=
  -- Define the number using the permutation p
  3 * 10^20 + (p 0).val * 10^19 + 4 * 10^18 + (p 1).val * 10^17 + 1 * 10^16 + (p 2).val * 10^15 +
  0 * 10^14 + (p 3).val * 10^13 + 8 * 10^12 + (p 4).val * 10^11 + 2 * 10^10 + (p 5).val * 10^9 +
  40923 * 10^8 + (p 6).val * 10^7 + 0 * 10^6 + (p 7).val * 10^5 + 320 * 10^4 + (p 8).val * 10^3 +
  2 * 10^2 + (p 9).val * 10 + 56

theorem special_number_divisible_by_396 (p : Fin 10 → Fin 10) (h : Function.Bijective p) :
  396 ∣ special_number p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_divisible_by_396_l36_3679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l36_3626

/-- The distance between Alice and Bob in miles -/
def distance_AB : ℝ := 15

/-- The angle of elevation from Alice's position in radians -/
noncomputable def angle_Alice : ℝ := Real.pi / 4

/-- The angle of elevation from Bob's position in radians -/
noncomputable def angle_Bob : ℝ := Real.pi / 4

/-- The altitude of the airplane in miles -/
def altitude : ℝ := distance_AB

theorem airplane_altitude :
  distance_AB * (Real.tan angle_Alice) = altitude ∧
  distance_AB * (Real.tan angle_Bob) = altitude :=
by
  sorry

#eval distance_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l36_3626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_properties_l36_3683

/-- Represents an isosceles triangle with given side length and perimeter -/
structure IsoscelesTriangle where
  side_length : ℝ
  perimeter : ℝ

/-- Calculates the base length of an isosceles triangle -/
def base_length (t : IsoscelesTriangle) : ℝ := t.perimeter - 2 * t.side_length

/-- Calculates the height of an isosceles triangle -/
noncomputable def triangle_height (t : IsoscelesTriangle) : ℝ := 
  Real.sqrt (t.side_length^2 - (base_length t / 2)^2)

/-- Calculates the area of an isosceles triangle -/
noncomputable def triangle_area (t : IsoscelesTriangle) : ℝ := 
  (1 / 2) * base_length t * triangle_height t

/-- Theorem about the base length and area of a specific isosceles triangle -/
theorem isosceles_triangle_properties :
  let t : IsoscelesTriangle := { side_length := 8, perimeter := 30 }
  base_length t = 14 ∧ triangle_area t = 7 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_properties_l36_3683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_90_moves_l36_3644

/-- Represents a complex number -/
structure ComplexNumber where
  re : ℝ
  im : ℝ

/-- The complex number ω = cis(π/3) -/
noncomputable def ω : ComplexNumber where
  re := 1/2
  im := Real.sqrt 3/2

/-- The position of the particle after n moves -/
def position (n : ℕ) : ComplexNumber :=
  sorry

/-- The theorem stating that after 90 moves, the particle returns to its initial position -/
theorem particle_position_after_90_moves :
  position 90 = { re := 8, im := 0 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_90_moves_l36_3644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l36_3667

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  area : ℝ

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the vector from one point to another
def vector (p q : ℝ × ℝ) : ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2)

-- Main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 3 * (dot_product (vector t.A t.B) (vector t.A t.C)) = 2 * t.area) :
  ∃ (sin_A : ℝ), sin_A = (3 * Real.sqrt 10) / 10 ∧
  (∀ (h2 : Real.cos (Real.pi / 4) = (t.C.1 - t.A.1) / Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2) ∧ 
           Real.sin (Real.pi / 4) = (t.C.2 - t.A.2) / Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)) 
      (h3 : dot_product (vector t.A t.B) (vector t.A t.C) = 16),
    Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) = 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l36_3667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_black_probability_l36_3676

/-- Represents a square on the grid -/
structure Square where
  x : Fin 4
  y : Fin 4

/-- The type of colors a square can have -/
inductive Color where
  | White
  | Black

/-- Represents the state of the grid -/
def Grid := Square → Color

/-- The probability of a square being black initially -/
noncomputable def initial_black_prob : ℝ := 1 / 2

/-- Rotates a square 180 degrees -/
def rotate (s : Square) : Square where
  x := 3 - s.x
  y := 3 - s.y

/-- The probability of the grid being all black after rotation -/
noncomputable def prob_all_black_after_rotation (g : Grid) : ℝ :=
  sorry

theorem grid_black_probability :
  ∀ g : Grid, prob_all_black_after_rotation g = 1 / 65536 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_black_probability_l36_3676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_system_problem_l36_3629

-- Define the polar coordinates of point P
noncomputable def P_polar : ℝ × ℝ := (4 * Real.sqrt 3, Real.pi / 6)

-- Define the polar equation of curve C
def C_polar (ρ θ : ℝ) : Prop := ρ^2 + 4 * Real.sqrt 3 * ρ * Real.sin θ = 4

-- Define the line l
def l (t : ℝ) : ℝ × ℝ := (3 + 2*t, -2 + 2*t)

-- Theorem statement
theorem coordinate_system_problem :
  -- Part 1: Cartesian coordinates of P
  let P_cartesian := (6, 2 * Real.sqrt 3)
  -- Part 2: Standard equation of curve C
  let C_cartesian (x y : ℝ) := x^2 + (y + 2 * Real.sqrt 3)^2 = 16
  -- Part 3: Maximum distance from midpoint M to line l
  ∃ (d_max : ℝ), d_max = 2 + Real.sqrt 2 ∧
    ∀ (θ : ℝ), 
      let Q := (4 * Real.cos θ, 4 * Real.sin θ - 2 * Real.sqrt 3)
      let M := ((P_cartesian.1 + Q.1) / 2, (P_cartesian.2 + Q.2) / 2)
      let d := |M.1 - M.2 - 5| / Real.sqrt 2
      d ≤ d_max :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_system_problem_l36_3629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l36_3665

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / (x - 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | x > -1 ∧ x ≠ 2}

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l36_3665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l36_3663

theorem polynomial_remainder : ∃ q : Polynomial ℝ, (X : Polynomial ℝ)^14 + 1 = (X + 1) * q + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l36_3663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_equiv_sin_shift_l36_3645

theorem cos_squared_equiv_sin_shift :
  ∀ x : ℝ, 2 * (Real.cos (x + π/4))^2 = -Real.sin (2*x) + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_equiv_sin_shift_l36_3645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l36_3637

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line ax+(2-a)y=0 -/
noncomputable def slope1 (a : ℝ) : ℝ := -a / (2 - a)

/-- The slope of the line x-ay=1 -/
noncomputable def slope2 (a : ℝ) : ℝ := 1 / a

/-- The condition that a=1 is sufficient but not necessary for perpendicularity -/
theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → perpendicular (slope1 a) (slope2 a)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ perpendicular (slope1 a) (slope2 a)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l36_3637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_nested_squares_l36_3694

/-- A set is a square with side length a -/
def is_square (S : Set (ℝ × ℝ)) (a : ℝ) : Prop := sorry

/-- The area of a set -/
def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- A square is inscribed in another square and rotated 45 degrees -/
def inscribed_rotated (S₁ S₂ : Set (ℝ × ℝ)) : Prop := sorry

/-- Given a square S₁ with area 25 cm², S₂ constructed inside S₁ with one vertex at 
    the center of S₁ and rotated 45 degrees, and S₃ constructed inside S₂ in the same manner, 
    prove that the area of S₃ is equal to the area of S₁. -/
theorem area_of_nested_squares (S₁ S₂ S₃ : Set (ℝ × ℝ)) : 
  (∃ (a : ℝ), area S₁ = 25 ∧ is_square S₁ a) →
  (∃ (b : ℝ), is_square S₂ b ∧ inscribed_rotated S₂ S₁) →
  (∃ (c : ℝ), is_square S₃ c ∧ inscribed_rotated S₃ S₂) →
  area S₃ = area S₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_nested_squares_l36_3694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l36_3605

/-- An ellipse with semi-major axis a, semi-minor axis b, and linear eccentricity c -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : 2 * b = a + c
  h4 : a^2 = b^2 + c^2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- Theorem stating that the eccentricity of the given ellipse is 3/5 -/
theorem ellipse_eccentricity (e : Ellipse) : eccentricity e = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l36_3605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_focus_product_of_distances_to_focus_lower_bound_l36_3608

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line passing through E(-1, 0)
def line (m : ℝ) (x y : ℝ) : Prop := x = m*y - 1

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem 1
theorem sum_of_distances_to_focus (A B : ℝ × ℝ) (m : ℝ) :
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  line m A.1 A.2 →
  line m B.1 B.2 →
  (A.1 + B.1) / 2 = 3 →
  distance A focus + distance B focus = 8 := by sorry

-- Theorem 2
theorem product_of_distances_to_focus_lower_bound (A B : ℝ × ℝ) (m : ℝ) :
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  line m A.1 A.2 →
  line m B.1 B.2 →
  distance A focus * distance B focus > 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_focus_product_of_distances_to_focus_lower_bound_l36_3608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_odd_l36_3622

-- Define the function f(x) = x^(1/3)
noncomputable def f (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem cube_root_odd : ∀ x : ℝ, f x + f (-x) = 0 := by
  intro x
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_odd_l36_3622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_sine_cosine_values_l36_3627

open Real

theorem tangent_and_sine_cosine_values (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : tan α = -2) : 
  tan (α + π/4) = -1/3 ∧ sin (2*α) * cos 2 = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_sine_cosine_values_l36_3627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_area_proof_l36_3678

/-- Calculates the area of a trapezoid-shaped envelope --/
noncomputable def envelope_area (bottom_width top_width height : ℝ) : ℝ :=
  (1 / 2) * (bottom_width + top_width) * height

/-- Proves that the area of a specific trapezoid-shaped envelope is 25 square inches --/
theorem envelope_area_proof :
  envelope_area 4 6 5 = 25 := by
  -- Unfold the definition of envelope_area
  unfold envelope_area
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_area_proof_l36_3678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twentieth_term_is_six_sevenths_l36_3612

/-- Defines the sequence as described in the problem -/
def our_sequence : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => 
    let m := Nat.sqrt n
    let k := n - m * (m + 1) / 2
    (k + 1 : ℚ) / (m + 2 : ℚ)

/-- The 20th term of the sequence is 6/7 -/
theorem twentieth_term_is_six_sevenths : our_sequence 19 = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twentieth_term_is_six_sevenths_l36_3612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_vertex_l36_3655

/-- Represents a face of a die --/
inductive Face
| one | two | three | four | five | six

/-- A function that returns the opposite face of a given face --/
def opposite : Face → Face
| Face.one => Face.six
| Face.two => Face.five
| Face.three => Face.four
| Face.four => Face.three
| Face.five => Face.two
| Face.six => Face.one

/-- A function that returns the numeric value of a face --/
def face_value : Face → Nat
| Face.one => 1
| Face.two => 2
| Face.three => 3
| Face.four => 4
| Face.five => 5
| Face.six => 6

/-- A predicate that checks if three faces can meet at a vertex --/
def can_meet_at_vertex (f1 f2 f3 : Face) : Prop :=
  f1 ≠ f2 ∧ f2 ≠ f3 ∧ f3 ≠ f1 ∧
  f1 ≠ opposite f2 ∧ f2 ≠ opposite f3 ∧ f3 ≠ opposite f1

/-- The theorem stating that the maximum sum of three faces meeting at a vertex is 14 --/
theorem max_sum_at_vertex :
  ∀ f1 f2 f3 : Face, can_meet_at_vertex f1 f2 f3 →
    face_value f1 + face_value f2 + face_value f3 ≤ 14 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_vertex_l36_3655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_half_l36_3631

theorem cos_alpha_plus_pi_half (α : ℝ) (h : Real.sin (α + π) = 3/4) :
  Real.cos (α + π/2) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_half_l36_3631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l36_3666

noncomputable def f1 (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f2 (x : ℝ) : ℝ := Real.log (abs x)
noncomputable def f3 (x : ℝ) : ℝ := 1 / (x - 1)
noncomputable def f4 (x : ℝ) : ℝ := x * Real.cos x

theorem odd_function_property :
  (∃ x, f1 x + f1 (-x) ≠ 0) ∧
  (∃ x, f2 x + f2 (-x) ≠ 0) ∧
  (∃ x, f3 x + f3 (-x) ≠ 0) ∧
  (∀ x, f4 x + f4 (-x) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l36_3666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_a_square_b_correct_l36_3661

/-- The algebraic expression for "the difference between a and the square of b" -/
def difference_a_square_b (a b : ℝ) : ℝ := a - b^2

/-- The verbal description of the expression -/
def verbal_description (a b : ℝ) : Prop :=
  difference_a_square_b a b = (fun x y => x - y^2) a b

theorem difference_a_square_b_correct (a b : ℝ) :
  verbal_description a b :=
by
  -- Unfold the definitions
  unfold verbal_description
  unfold difference_a_square_b
  -- The equality is true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_a_square_b_correct_l36_3661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l36_3675

noncomputable def f (ω a x : ℝ) : ℝ := (1/2) * (Real.sin (ω * x) + a * Real.cos (ω * x))

theorem function_properties (a : ℝ) (ω : ℝ) (h_ω : 0 < ω ∧ ω ≤ 1) :
  (∀ x, f ω a x = f ω a (π/3 - x)) ∧
  (∀ x, f ω a (x - π) = f ω a (x + π)) →
  (∀ x, f ω a x = Real.sin (x + π/3)) ∧
  ∀ x₁ x₂, x₁ ∈ Set.Ioo (-π/3) (5*π/3) → x₂ ∈ Set.Ioo (-π/3) (5*π/3) →
    f ω a x₁ = -1/2 → f ω a x₂ = -1/2 →
    x₁ + x₂ = 7*π/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l36_3675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_solution_exists_iff_l36_3652

-- Define the function f(x) = 4^x - 2^(x+1)
noncomputable def f (x : ℝ) : ℝ := (4 : ℝ)^x - 2^(x+1)

-- Theorem 1: The range of f is [-1, +∞)
theorem range_of_f : Set.range f = Set.Ici (-1) := by sorry

-- Theorem 2: The equation f(x) + a = 0 has solutions iff a ≤ 1
theorem solution_exists_iff (a : ℝ) : (∃ x, f x + a = 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_solution_exists_iff_l36_3652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_simplest_quadratic_surd_l36_3693

-- Definition of a quadratic surd
def is_quadratic_surd (x : ℝ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ ¬ (∃ m : ℕ, m * m = n) ∧ x = Real.sqrt n

-- Definition of simplest quadratic surd
def is_simplest_quadratic_surd (x : ℝ) : Prop :=
  is_quadratic_surd x ∧ ∀ y : ℝ, is_quadratic_surd y → x ≤ y

-- Theorem statement
theorem sqrt_3_simplest_quadratic_surd :
  is_simplest_quadratic_surd (Real.sqrt 3) ∧
  ¬ is_simplest_quadratic_surd (Real.sqrt (1/2)) ∧
  ¬ is_quadratic_surd 2 ∧
  ¬ is_quadratic_surd 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_simplest_quadratic_surd_l36_3693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l36_3642

open Real

-- Define the parametric equations of the line
noncomputable def x (t : ℝ) : ℝ := -2 + t * cos (π / 6)
noncomputable def y (t : ℝ) : ℝ := 3 - t * sin (π / 3)

-- Define the inclination angle
noncomputable def inclination_angle : ℝ := 3 * π / 4  -- 135°

-- Theorem statement
theorem line_inclination_angle :
  ∀ t : ℝ, ∃ m : ℝ,
    (y t - y 0) = m * (x t - x 0) ∧
    tan inclination_angle = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l36_3642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l36_3630

-- Define the interval [0, 1]
def I : Set ℝ := Set.Icc 0 1

-- Define the functions K, f, and g
variable (K : ℝ → ℝ → ℝ)
variable (f g : ℝ → ℝ)

-- State the conditions
variable (hK : ∀ x ∈ I, ∀ y ∈ I, K x y > 0 ∧ Continuous (λ p : ℝ × ℝ ↦ K p.1 p.2))
variable (hf : ∀ x ∈ I, f x > 0 ∧ Continuous f)
variable (hg : ∀ x ∈ I, g x > 0 ∧ Continuous g)

variable (h1 : ∀ x ∈ I, ∫ y in I, f y * K x y = g x)
variable (h2 : ∀ x ∈ I, ∫ y in I, g y * K x y = f x)

-- State the theorem
theorem f_equals_g : f = g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l36_3630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_reroll_two_dice_l36_3618

/-- Represents a standard six-sided die --/
def Die := Fin 6

/-- Represents the result of rolling three dice --/
def ThreeDiceRoll := Die × Die × Die

/-- Calculates the sum of a three dice roll --/
def sum_roll (roll : ThreeDiceRoll) : Nat :=
  roll.1.val + 1 + roll.2.1.val + 1 + roll.2.2.val + 1

/-- Determines if a given roll is a winning roll (sum is 9) --/
def is_winning_roll (roll : ThreeDiceRoll) : Prop :=
  sum_roll roll = 9

/-- Represents Jason's strategy for rerolling --/
def optimal_reroll_strategy (roll : ThreeDiceRoll) : Fin 4 := sorry

/-- Represents the event of Jason choosing to reroll exactly two dice --/
def reroll_two_dice (roll : ThreeDiceRoll) : Prop :=
  optimal_reroll_strategy roll = 2

/-- The probability of an event occurring when rolling three fair dice --/
noncomputable def prob (event : ThreeDiceRoll → Prop) : ℚ := sorry

/-- The main theorem: probability of rerolling two dice is 5/9 --/
theorem prob_reroll_two_dice :
  prob reroll_two_dice = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_reroll_two_dice_l36_3618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_integer_pairs_sqrt_2009_l36_3624

theorem distinct_integer_pairs_sqrt_2009 :
  ∃! (pairs : List (ℕ × ℕ)), 
    (∀ (pair : ℕ × ℕ), pair ∈ pairs → 
      let (x, y) := pair
      0 < x ∧ x < y ∧ Real.sqrt 2009 = Real.sqrt x + Real.sqrt y) ∧
    pairs.length = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_integer_pairs_sqrt_2009_l36_3624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l36_3643

/-- Given a circle intersected by three equally spaced parallel lines creating chords of lengths 40, 40, and 36, the distance between two adjacent parallel lines is 2. -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (∃ (chord1 chord2 chord3 : ℝ), 
    chord1 = 40 ∧ 
    chord2 = 40 ∧ 
    chord3 = 36 ∧ 
    r^2 * 20 + r^2 * 20 = 40 * ((d/2)^2 + 20 * 20) ∧
    r^2 * 18 + r^2 * 18 = 36 * ((d/2)^2 + 18 * 18)) →
  d = 2 := by
  intro h
  sorry

#check parallel_lines_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l36_3643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l36_3685

/-- The parabola is defined by y = (1/4)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

/-- The focus of the parabola is at (0, 1) -/
def focus : ℝ × ℝ := (0, 1)

/-- P is a point on the parabola -/
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  parabola x y

/-- Q is the midpoint of PF -/
def midpoint_of_PF (P Q : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (fx, fy) := focus
  qx = (px + fx) / 2 ∧ qy = (py + fy) / 2

/-- The theorem to be proved -/
theorem midpoint_trajectory :
  ∀ P Q : ℝ × ℝ,
  point_on_parabola P →
  midpoint_of_PF P Q →
  let (x, y) := Q
  x^2 = 2*y - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l36_3685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_theorem_l36_3659

-- Define the curve C
def curve_C (a : ℝ) (x y : ℝ) : Prop := y^2 = a * x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define point P
def point_P : ℝ × ℝ := (-2, -4)

-- Define the intersection points A and B
def intersection_points (a : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ),
    curve_C a A.1 A.2 ∧
    curve_C a B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2

-- Define the condition |PA| · |PB| = |AB|²
def distance_condition (A B : ℝ × ℝ) : Prop :=
  let PA := ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2);
  let PB := ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2);
  let AB := ((B.1 - A.1)^2 + (B.2 - A.2)^2);
  PA * PB = AB^2

theorem curve_line_intersection_theorem :
  ∀ a : ℝ, a > 0 →
    intersection_points a →
    (∀ A B : ℝ × ℝ, curve_C a A.1 A.2 → curve_C a B.1 B.2 →
      line_l A.1 A.2 → line_l B.1 B.2 → distance_condition A B) →
    a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_theorem_l36_3659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_parallel_l36_3617

/-- The curve C defined by y = ax³ + bx² + d -/
def C (a b d : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + b * x^2 + d

/-- The derivative of curve C -/
def C' (a b : ℝ) : ℝ → ℝ := fun x ↦ 3 * a * x^2 + 2 * b * x

theorem curve_tangent_parallel (a b d : ℝ) :
  C a b d 1 = 1 ∧ 
  C a b d (-1) = -3 ∧ 
  C' a b 1 = C' a b (-1) →
  a^3 + b^2 + d = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_parallel_l36_3617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daisy_cooking_percentage_l36_3657

/-- The percentage of remaining milk Daisy uses for cooking -/
noncomputable def cooking_percentage (total_milk : ℝ) (kids_consumption_rate : ℝ) (leftover_milk : ℝ) : ℝ :=
  let remaining_milk := total_milk * (1 - kids_consumption_rate)
  let cooking_milk := remaining_milk - leftover_milk
  (cooking_milk / remaining_milk) * 100

/-- Proof that Daisy uses 50% of the remaining milk for cooking -/
theorem daisy_cooking_percentage :
  cooking_percentage 16 0.75 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daisy_cooking_percentage_l36_3657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_and_perpendicular_chord_l36_3682

-- Define the curve
def f (x : ℝ) : ℝ := x^2 - 6*x + 1

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 9

-- Define the line
def line_eq (x y a : ℝ) : Prop := x - y + a = 0

theorem intersection_circle_and_perpendicular_chord (a : ℝ) :
  -- The curve intersects the axes at points on the circle
  (∃ x, f x = 0 ∧ circle_eq x 0) ∧
  (circle_eq 0 (f 0)) ∧
  -- The circle intersects the line at two points
  (∃ A B : ℝ × ℝ, 
    A.1 ≠ B.1 ∧
    circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
    line_eq A.1 A.2 a ∧ line_eq B.1 B.2 a ∧
    -- CA is perpendicular to CB
    (A.1 - 3) * (B.1 - 3) + (A.2 - 1) * (B.2 - 1) = 0) →
  -- Then a = 1 or a = -5
  a = 1 ∨ a = -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_and_perpendicular_chord_l36_3682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_day_theorem_l36_3684

/-- The day of the week, represented as an integer from 0 (Sunday) to 6 (Saturday) -/
inductive DayOfWeek : Type
| sunday : DayOfWeek
| monday : DayOfWeek
| tuesday : DayOfWeek
| wednesday : DayOfWeek
| thursday : DayOfWeek
| friday : DayOfWeek
| saturday : DayOfWeek

/-- Returns true if the given year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Calculates the number of days between two dates -/
def daysBetween (startYear startMonth startDay endYear endMonth endDay : ℕ) : ℕ :=
  sorry -- Implementation details omitted

/-- Calculates the day of the week given a number of days from a known day -/
def dayOfWeekAfter (start : DayOfWeek) (days : ℕ) : DayOfWeek :=
  sorry -- Implementation details omitted

theorem olympic_day_theorem :
  let startDay : DayOfWeek := DayOfWeek.saturday
  let daysBetweenDates := daysBetween 2016 3 12 2022 2 4
  dayOfWeekAfter startDay daysBetweenDates = startDay :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_day_theorem_l36_3684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_sum_l36_3647

theorem trigonometric_ratio_sum (x y : ℝ) 
  (h1 : Real.tan x / Real.tan y = 2)
  (h2 : Real.sin x / Real.sin y = 4) :
  Real.sin (2 * x) / Real.sin (2 * y) + Real.cos (2 * x) / Real.cos (2 * y) = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_sum_l36_3647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_from_parametric_l36_3690

def plane_equation (v : ℝ × ℝ × ℝ) (s t : ℝ) : Prop :=
  v.1 = 2 + 2*s - 3*t ∧ v.2.1 = 1 - 2*s ∧ v.2.2 = 4 + s + 3*t

theorem plane_equation_from_parametric :
  ∃ (A B C D : ℤ), 
    (∀ (x y z : ℝ), (∃ (s t : ℝ), plane_equation (x, y, z) s t) ↔ A * x + B * y + C * z + D = 0) ∧
    A > 0 ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 ∧
    A = 2 ∧ B = 3 ∧ C = 2 ∧ D = -15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_from_parametric_l36_3690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_gt_arccos_iff_l36_3648

theorem arctan_gt_arccos_iff (x : ℝ) : Real.arctan x > Real.arccos x ↔ x ∈ Set.Icc (-1 : ℝ) 0 ∧ x ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_gt_arccos_iff_l36_3648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_sibling_age_l36_3639

/-- Represents the ages of four siblings --/
structure SiblingAges where
  oldest : ℝ
  middle : ℝ
  secondYoungest : ℝ
  youngest : ℝ

/-- The conditions of the sibling ages problem --/
def SiblingAgesProblem (ages : SiblingAges) : Prop :=
  -- The sum of the ages is 100
  ages.oldest + ages.middle + ages.secondYoungest + ages.youngest = 100 ∧
  -- The youngest sibling's age condition
  ages.youngest = (1/3) * ages.middle - (1/5) * ages.oldest ∧
  -- The middle sibling's age condition
  ages.middle = ages.oldest - 8 ∧
  -- The second youngest sibling's age condition
  ages.secondYoungest = (1/2) * ages.youngest + 6.5

/-- The theorem stating the age of the youngest sibling --/
theorem youngest_sibling_age (ages : SiblingAges) 
  (h : SiblingAgesProblem ages) : 
  ∃ ε > 0, |ages.youngest - 3.83| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_sibling_age_l36_3639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_natural_l36_3670

noncomputable def f (n : ℕ) : ℝ := Real.tan (Real.pi / 7) ^ (2 * n) + Real.tan (2 * Real.pi / 7) ^ (2 * n) + Real.tan (3 * Real.pi / 7) ^ (2 * n)

theorem f_is_natural : ∀ n : ℕ, ∃ m : ℕ, f n = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_natural_l36_3670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l36_3697

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := (4 * x - 3) / (2 * x - 5)

-- State the theorem about the domain of f(x)
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 5/2 ∨ x > 5/2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l36_3697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_two_thirds_minus_four_fifths_i_l36_3636

theorem complex_magnitude_two_thirds_minus_four_fifths_i :
  Complex.abs (2/3 - (4/5)*Complex.I) = 2*Real.sqrt 61/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_two_thirds_minus_four_fifths_i_l36_3636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l36_3620

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α - Real.cos α = Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l36_3620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l36_3650

/-- Given a geometric sequence with first term a and common ratio q,
    S_n represents the sum of its first n terms. -/
noncomputable def S (a q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with S_3 = 6 and S_6 = 54, the common ratio q is 2. -/
theorem geometric_sequence_common_ratio 
  (a q : ℝ) (h1 : S a q 3 = 6) (h2 : S a q 6 = 54) : q = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l36_3650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l36_3681

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b ∨ b = k • a

/-- Given vectors a = (λ, 3) and b = (-2, 4), if a and b are collinear, then λ = -3/2 -/
theorem collinear_vectors_lambda (l : ℝ) :
  collinear (l, 3) (-2, 4) → l = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l36_3681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sicos_properties_l36_3646

noncomputable def sicos (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (x + Real.pi / 4)

theorem sicos_properties :
  (∀ x : ℝ, sicos (3 * Real.pi / 2 - x) = -sicos x) ∧
  (∀ x : ℝ, sicos x ≠ 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sicos_properties_l36_3646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_norm_achievers_is_24_l36_3635

/-- The number of players in the tournament -/
def num_players : ℕ := 30

/-- The number of games each player plays -/
def games_per_player : ℕ := num_players - 1

/-- The total number of games in the tournament -/
def total_games : ℕ := (num_players * games_per_player) / 2

/-- The total points available in the tournament -/
def total_points : ℕ := total_games

/-- The percentage of points needed for the 4th category norm -/
def norm_percentage : ℚ := 3/5

/-- The points needed for a player to achieve the norm -/
def points_for_norm : ℚ := (games_per_player : ℚ) * norm_percentage

/-- The maximum number of players who can achieve the norm -/
def max_norm_achievers : ℕ := (((total_points : ℚ) / points_for_norm).floor : ℤ).toNat

theorem max_norm_achievers_is_24 : max_norm_achievers = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_norm_achievers_is_24_l36_3635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_route_designs_l36_3623

/-- Represents the number of cities in each province -/
def Province := ℕ

/-- The total number of cities -/
def total_cities (provinces : List ℕ) : ℕ :=
  provinces.sum

/-- The number of different flight route designs -/
def num_designs (provinces : List ℕ) : ℕ :=
  let n := total_cities provinces
  let k := provinces.length
  n^(k-2) * (provinces.map (fun a => (n - a)^(a - 1))).prod

theorem flight_route_designs (provinces : List ℕ) :
  num_designs provinces =
  let n := total_cities provinces
  let k := provinces.length
  n^(k-2) * (provinces.map (fun a => (n - a)^(a - 1))).prod :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_route_designs_l36_3623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_no_equation_l36_3651

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + x⁻¹

-- Theorem stating that f doesn't satisfy any of the four equations
theorem f_satisfies_no_equation :
  (∀ x y : ℝ, f (x + y) ≠ f x + f y) ∧
  (∀ x y : ℝ, f (x * y) ≠ f x + f y) ∧
  (∀ x y : ℝ, f (x + y) ≠ f x * f y) ∧
  (∀ x y : ℝ, f (x * y) ≠ f x * f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_no_equation_l36_3651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_spheres_l36_3653

def sphere1_center : ℝ × ℝ × ℝ := (3, -4, 7)
def sphere2_center : ℝ × ℝ × ℝ := (-8, 9, -10)
def sphere1_radius : ℝ := 23
def sphere2_radius : ℝ := 76

noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2.1 - q.2.1)^2 + (p.2.2 - q.2.2)^2)

theorem max_distance_between_spheres :
  ∃ (p1 p2 : ℝ × ℝ × ℝ),
    distance p1 sphere1_center = sphere1_radius ∧
    distance p2 sphere2_center = sphere2_radius ∧
    ∀ (q1 q2 : ℝ × ℝ × ℝ),
      distance q1 sphere1_center = sphere1_radius →
      distance q2 sphere2_center = sphere2_radius →
      distance q1 q2 ≤ distance p1 p2 ∧
      distance p1 p2 = 99 + Real.sqrt 579 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_spheres_l36_3653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_sixth_minus_two_alpha_l36_3662

theorem sin_pi_sixth_minus_two_alpha (α : ℝ) 
  (h1 : Real.cos (α + π/6) = 1/3) 
  (h2 : 0 < α ∧ α < π/2) : 
  Real.sin (π/6 - 2*α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_sixth_minus_two_alpha_l36_3662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_real_implies_a_equals_one_l36_3654

theorem complex_product_real_implies_a_equals_one :
  ∀ (a : ℝ), (Complex.im ((1 - Complex.I) * (Complex.ofReal a + Complex.I)) = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_real_implies_a_equals_one_l36_3654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_is_936_l36_3677

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℚ
  deriving Repr

/-- Calculates the volume of a cube given its dimensions -/
def cubeVolume (c : CubeDimensions) : ℚ := c.side ^ 3

/-- Represents the container and its contents -/
structure Container where
  dimensions : CubeDimensions
  waterFillRatio : ℚ
  iceCubes : CubeDimensions
  numIceCubes : ℕ
  deriving Repr

/-- Calculates the unoccupied volume in the container -/
def unoccupiedVolume (c : Container) : ℚ :=
  let containerVolume := cubeVolume c.dimensions
  let waterVolume := c.waterFillRatio * containerVolume
  let iceCubeVolume := cubeVolume c.iceCubes
  let totalIceVolume := (c.numIceCubes : ℚ) * iceCubeVolume
  containerVolume - (waterVolume + totalIceVolume)

/-- The main theorem to be proved -/
theorem unoccupied_volume_is_936 (c : Container) 
  (h1 : c.dimensions.side = 12)
  (h2 : c.waterFillRatio = 1/3)
  (h3 : c.iceCubes.side = 3)
  (h4 : c.numIceCubes = 8) :
  unoccupiedVolume c = 936 := by
  sorry

def main : IO Unit := do
  let result := unoccupiedVolume {
    dimensions := { side := 12 },
    waterFillRatio := 1/3,
    iceCubes := { side := 3 },
    numIceCubes := 8
  }
  IO.println s!"Unoccupied volume: {result}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_is_936_l36_3677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l36_3680

/-- An inverse proportion function with parameter k -/
noncomputable def inverse_proportion (k : ℝ) : ℝ → ℝ := fun x ↦ (k - 3) / x

/-- Predicate to check if a point (x, y) is in the first or third quadrant -/
def in_first_or_third_quadrant (x y : ℝ) : Prop := x * y > 0

/-- Theorem: If the graph of y = (k-3)/x is in the first and third quadrants, then k > 3 -/
theorem inverse_proportion_quadrants (k : ℝ) :
  (∀ x ≠ 0, in_first_or_third_quadrant x (inverse_proportion k x)) → k > 3 := by
  sorry

#check inverse_proportion_quadrants

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l36_3680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_polynomial_root_l36_3698

theorem complex_polynomial_root (a b c : ℤ) : 
  (a * (3 + Complex.I)^4 + b * (3 + Complex.I)^3 + c * (3 + Complex.I)^2 + b * (3 + Complex.I) + a = 0) →
  (∀ (p : ℕ), p.Prime → (p ∣ a.natAbs ∧ p ∣ b.natAbs ∧ p ∣ c.natAbs) → p = 1) →
  |c| = 97 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_polynomial_root_l36_3698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_frog_expected_returns_l36_3609

/-- Represents the probability of moving left, right, or being removed. -/
def step_probability : ℚ := 1 / 3

/-- Represents the expected number of returns to the starting point. -/
noncomputable def expected_returns : ℝ := (3 * Real.sqrt 5 - 5) / 5

/-- 
Theorem stating that the expected number of returns to the starting point
for a frog on an infinite number line is (3√5 - 5) / 5, given that in each step,
the frog has an equal probability (1/3) of moving left one unit, moving right one unit,
or being removed from the line.
-/
theorem frog_expected_returns :
  let p := step_probability
  ∀ (n : ℕ), 
    (p = 1 / 3) →
    (expected_returns = ∑' (k : ℕ), (Nat.choose (2 * k) k : ℝ) * ((p : ℝ)^2)^k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_frog_expected_returns_l36_3609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_increase_l36_3669

/-- The increase in area of a circle when its radius is increased from 5 cm to 10 cm is 75π cm². -/
theorem circle_area_increase : 
  let original_radius : ℝ := 5
  let new_radius : ℝ := 10
  π * new_radius^2 - π * original_radius^2 = 75 * π := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_increase_l36_3669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l36_3668

/-- A quadratic radical is considered simple if it has a non-negative integer radicand
    and is in its simplest form. -/
def is_simple_quadratic_radical (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≥ 0 ∧ x = a * Real.sqrt (b : ℝ) ∧ ¬∃ (c d : ℤ), d > 1 ∧ b = c * d * d

/-- The given options for quadratic radicals -/
noncomputable def options : List ℝ := [Real.sqrt 0.1, Real.sqrt (-2), 3 * Real.sqrt 2, -Real.sqrt 20]

theorem simplest_quadratic_radical :
  ∀ x ∈ options, x ≠ 3 * Real.sqrt 2 → ¬(is_simple_quadratic_radical x) ∧ is_simple_quadratic_radical (3 * Real.sqrt 2) :=
by sorry

#check simplest_quadratic_radical

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l36_3668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l36_3610

theorem solution_range (k : ℝ) : 
  (∃ x : ℝ, Real.sin x ^ 2 + Real.cos x + k = 0) ↔ -2 ≤ k ∧ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l36_3610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_ratio_specific_boat_problem_l36_3611

/-- Calculates the ratio of average speed for a round trip to the boat's speed in still water -/
noncomputable def speedRatio (s c : ℝ) : ℝ :=
  (s^2 - c^2) / s^2

theorem boat_speed_ratio (s c : ℝ) (hs : s > 0) (hc : c > 0) (hsc : s > c) :
  let downstreamSpeed := s + c
  let upstreamSpeed := s - c
  let averageSpeed := 2 / (1 / downstreamSpeed + 1 / upstreamSpeed)
  averageSpeed / s = speedRatio s c := by
  sorry

theorem specific_boat_problem :
  speedRatio 12 3 = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_ratio_specific_boat_problem_l36_3611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_theorem_l36_3621

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance between the center and the focus of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2)

theorem hyperbola_eccentricity_theorem (h : Hyperbola) 
  (h_ab_cd : ∃ (A B C D : ℝ × ℝ), 
    A.1 = focal_distance h ∧ 
    B.1 = focal_distance h ∧
    C.1 = focal_distance h ∧ 
    D.1 = focal_distance h ∧
    |A.2 - B.2| = (3/5) * |C.2 - D.2|) :
  eccentricity h = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_theorem_l36_3621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l36_3614

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f φ (x + Real.pi / 3)

theorem phi_value (φ : ℝ) (h1 : |φ| < Real.pi / 2) (h2 : ∀ x, g φ x = g φ (-x)) : φ = -Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l36_3614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l36_3673

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem f_properties :
  (∃ x : ℝ, f x > Real.sqrt 2) ∧
  (∀ x : ℝ, f (x - 2 * Real.pi) = f x) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → f (x + Real.pi) > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l36_3673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_D_l36_3632

noncomputable section

-- Define the polynomial and its factorization
def p (x : ℝ) := x^3 - 2*x^2 - 13*x + 10
def p_factored (x : ℝ) := (x - 2) * (x - 1) * (x + 5)

-- Define the partial fraction decomposition
def partial_fraction (D E F x : ℝ) := D / (x - 2) + E / (x - 1) + F / (x + 5)^2

-- State the theorem
theorem find_D : 
  ∃ (D E F : ℝ), 
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 1 ∧ x ≠ -5 → 
      1 / p_factored x = partial_fraction D E F x) → 
    D = 1/7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_D_l36_3632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l36_3603

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (Real.sin x)^2 - Real.tan x
  else Real.exp (-2 * x)

-- State the theorem
theorem f_composition_value : f (f (-25 * Real.pi / 4)) = Real.exp (-3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l36_3603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_variance_with_zero_and_one_l36_3699

-- Define a set of n real numbers containing 0 and 1
def NumberSet (n : ℕ) := {s : Finset ℝ | s.card = n ∧ (0 : ℝ) ∈ s ∧ (1 : ℝ) ∈ s}

-- Define the variance of a set
noncomputable def variance (s : Finset ℝ) : ℝ :=
  let mean := s.sum id / s.card
  (s.sum (λ x => (x - mean)^2)) / s.card

-- Theorem statement
theorem min_variance_with_zero_and_one (n : ℕ) (hn : n ≥ 2) :
  ∃ (s : Finset ℝ), s ∈ NumberSet n ∧
    (∀ (t : Finset ℝ), t ∈ NumberSet n → variance s ≤ variance t) ∧
    variance s = 1 / (2 * n) ∧
    (∀ x ∈ s, x = 0 ∨ x = 1 ∨ x = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_variance_with_zero_and_one_l36_3699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l36_3633

-- Define the function as noncomputable due to Real.log
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4*x + 5)

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l36_3633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_B_and_A_minus_B_l36_3689

-- Define the polynomial type
def MyPolynomial (α : Type) := ℕ → α

-- Define the polynomial A
def A : MyPolynomial ℚ := fun n ↦ 
  match n with
  | 0 => 6
  | 1 => -5
  | 2 => 2
  | _ => 0

-- Define the sum of A and B
def A_plus_B : MyPolynomial ℚ := fun n ↦
  match n with
  | 0 => 6
  | 1 => -4
  | 2 => 4
  | _ => 0

-- Theorem to prove B and A - B
theorem find_B_and_A_minus_B :
  ∃ (B : MyPolynomial ℚ),
    (∀ n, A_plus_B n = A n + B n) ∧
    (B 0 = 0 ∧ B 1 = 1 ∧ B 2 = 2 ∧ ∀ n > 2, B n = 0) ∧
    (∀ n, A n - B n = 
      match n with
      | 0 => 6
      | 1 => -6
      | _ => 0
    ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_B_and_A_minus_B_l36_3689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_is_87_l36_3671

def numbers : List Nat := [46, 69, 87, 121, 143]

/-- The largest prime factor of a natural number -/
def largestPrimeFactor (n : Nat) : Nat :=
  (Nat.factors n).maximum?.getD 1

theorem largest_prime_factor_is_87 :
  ∀ n ∈ numbers, n ≠ 87 → largestPrimeFactor n < largestPrimeFactor 87 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_is_87_l36_3671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_properties_l36_3607

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x + Real.cos x, -1)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem vector_dot_product_properties :
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x ≤ 2 ∧ f x ≥ 1) ∧
  (∀ x₀ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f x₀ = 6/5 → Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10) ∧
  (∀ ω > 0, (∀ x ∈ Set.Ioo (Real.pi / 3) ((2 * Real.pi) / 3), StrictMono (fun x => f (ω * x))) →
    ω ≤ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_properties_l36_3607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l36_3688

/-- The parabola defined by y = (1/4)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, 1)

/-- The directrix of the parabola -/
def directrix (y : ℝ) : Prop := y = -1

/-- The distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_focus_directrix_distance :
  ∀ (x : ℝ), distance focus (x, -1) = 2 :=
by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l36_3688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_for_given_sine_l36_3602

theorem trig_values_for_given_sine (x : ℝ) 
  (h1 : 0 < x) (h2 : x < 3 * Real.pi / 2) (h3 : Real.sin x = -3/5) :
  Real.cos x = -4/5 ∧ Real.tan x = 3/4 ∧ 1 / Real.tan x = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_for_given_sine_l36_3602
