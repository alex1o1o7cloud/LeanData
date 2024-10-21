import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ten_mod_thirteen_l238_23860

theorem factorial_ten_mod_thirteen : ∃ k, Nat.factorial 10 = 13 * k + 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ten_mod_thirteen_l238_23860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_q_l238_23897

noncomputable def M (x y : ℝ) : ℝ := max x y

noncomputable def m (x y : ℝ) : ℝ := min x y

theorem expression_equals_q (p q r s t : ℝ) 
  (h1 : p < q) (h2 : q < r) (h3 : r < s) (h4 : s < t) :
  M (M p (m q r)) (m s (M p t)) = q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_q_l238_23897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l238_23874

open Real

/-- The area of the triangle formed by the tangent line to y = e^x at (2, e^2) and the axes --/
theorem tangent_triangle_area : 
  let f (x : ℝ) := exp x
  let point : ℝ × ℝ := (2, exp 2)
  let slope := (deriv f) point.1
  let tangent_line (x : ℝ) := slope * (x - point.1) + point.2
  let x_intercept := (point.2 - slope * point.1) / slope
  let y_intercept := tangent_line 0
  (1/2) * x_intercept * abs y_intercept = exp 2 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l238_23874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_eq_one_over_512_l238_23876

open Real

/-- The nth term of the series -/
noncomputable def a (n : ℕ) : ℝ := Real.sqrt (4 * n + 2) / ((4 * n)^2 * (4 * n + 4)^2)

/-- The sum of the infinite series -/
noncomputable def series_sum : ℝ := ∑' n, a n

/-- Theorem: The sum of the infinite series is 1/512 -/
theorem series_sum_eq_one_over_512 : series_sum = 1 / 512 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_eq_one_over_512_l238_23876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_trigonometry_l238_23802

noncomputable section

open Real

theorem angle_trigonometry (θ : ℝ) :
  (∃ (x y r : ℝ), x = 3 ∧ y = -4 ∧ x = r * (cos θ) ∧ y = r * (sin θ) ∧ r > 0) →
  sin θ = -4/5 ∧ cos θ = 3/5 ∧ tan θ = -4/3 ∧
  (cos (3*π - θ) + cos (3*π/2 + θ)) / (sin (π/2 - θ) + tan (π + θ)) = 21/11 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_trigonometry_l238_23802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_result_l238_23808

/-- Election Result Theorem -/
theorem election_result (total_votes : ℕ) (invalid_percent : ℚ) (difference_percent : ℚ) 
  (h_total : total_votes = 8720)
  (h_invalid : invalid_percent = 1/5)
  (h_difference : difference_percent = 3/20) :
  let valid_votes := total_votes - (invalid_percent * ↑total_votes).floor
  let b_votes := (valid_votes - (difference_percent * ↑total_votes).floor) / 2
  b_votes = 2834 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_result_l238_23808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_norm_of_v_l238_23851

open Real NormedSpace

noncomputable def v : ℝ × ℝ := sorry

theorem smallest_norm_of_v (h : ‖v + (4, 2)‖ = 10) :
  ∀ w : ℝ × ℝ, ‖w + (4, 2)‖ = 10 → ‖v‖ ≤ ‖w‖ ∧ ‖v‖ = 10 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_norm_of_v_l238_23851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l238_23834

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the foci of the ellipse
def foci (F1 F2 : ℝ × ℝ) : Prop := ∃ c : ℝ, c^2 = 1 ∧ 
  F1 = (c, 0) ∧ F2 = (-c, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  ellipse P.1 P.2

-- Define a right angle
def right_angle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0

-- Define the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Theorem statement
theorem ellipse_triangle_area 
  (P F1 F2 : ℝ × ℝ) 
  (h1 : point_on_ellipse P) 
  (h2 : foci F1 F2) 
  (h3 : right_angle F1 P F2) : 
  triangle_area F1 P F2 = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l238_23834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l238_23883

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ p = Real.pi ∧ ∀ (x : ℝ), (f x)^2 = (f (x + p))^2) ∧
  (∀ (x : ℝ), f (2*x - Real.pi/2) = Real.sqrt 2 * Real.sin (x/2)) ∧
  (∃ (M : ℝ), M = 1 + Real.sqrt 3 / 2 ∧
    ∀ (x : ℝ), (f x + Real.cos x) * (Real.sqrt 3 * Real.sin x + Real.cos x) ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l238_23883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l238_23861

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + 2 * x

-- State the theorem
theorem f_minimum_value :
  ∃ (min : ℝ), min = -2 ∧ ∀ (x : ℝ), f x ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l238_23861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l238_23882

-- Define the coefficients of the quadratic equation
noncomputable def a : ℝ := 9 + 6 * Real.sqrt 2
noncomputable def b : ℝ := 3 + 2 * Real.sqrt 2
def c : ℝ := -3

-- Define the quadratic equation
def quad_equation (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the difference between roots
noncomputable def root_difference : ℝ := Real.sqrt (97 - 72 * Real.sqrt 2) / 3

-- Theorem statement
theorem quadratic_root_difference :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quad_equation x₁ ∧ quad_equation x₂ ∧
  (x₁ > x₂ → x₁ - x₂ = root_difference) ∧
  (x₂ > x₁ → x₂ - x₁ = root_difference) := by
  sorry

#check quadratic_root_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l238_23882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_four_l238_23871

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2*x^2 + x - 2 + Real.sin x) / (x^2 - 1)

-- Define the domain of the function
def domain (x : ℝ) : Prop := x ≠ 1 ∧ x ≠ -1

-- State the theorem
theorem max_min_sum_equals_four :
  ∃ (M m : ℝ), (∀ x, domain x → f x ≤ M) ∧
               (∀ x, domain x → m ≤ f x) ∧
               (∃ x, domain x ∧ f x = M) ∧
               (∃ x, domain x ∧ f x = m) ∧
               (M + m = 4) := by
  sorry

#check max_min_sum_equals_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_four_l238_23871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_practice_time_approximation_l238_23864

/-- Average number of weeks in a month -/
noncomputable def avg_weeks_per_month : ℝ := 52 / 12

/-- Number of months in the practice period -/
def practice_months : ℕ := 5

/-- Hours of practice per week -/
def practice_hours_per_week : ℝ := 4

/-- Calculate the total practice hours over a given number of months -/
noncomputable def total_practice_hours (months : ℕ) : ℝ :=
  (months : ℝ) * avg_weeks_per_month * practice_hours_per_week

/-- Theorem stating that practicing 4 hours per week for 5 months results in approximately 86.9 hours -/
theorem practice_time_approximation :
  ∃ ε > 0, |total_practice_hours practice_months - 86.9| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_practice_time_approximation_l238_23864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l238_23800

theorem sequence_inequality (n : ℕ) (hn : n > 0) : 
  ∃ (a : ℕ → ℚ), 
    a 0 = 1/2 ∧
    (∀ k, a (k + 1) = a k + (1/n) * (a k)^2) ∧
    1 - 1/n < a n ∧ a n < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l238_23800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_relationship_l238_23887

theorem proposition_relationship (x : ℝ) :
  (abs (x - 1) < 5 → abs (abs x - 1) < 5) ∧
  ¬(abs (abs x - 1) < 5 → abs (x - 1) < 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_relationship_l238_23887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sean_jogging_time_l238_23838

/-- Sean's jogging pace in meters per minute -/
noncomputable def sean_pace : ℝ := 100 / 4

/-- Length of the running track in meters -/
def track_length : ℝ := 400

/-- Number of laps Sean needs to complete -/
def num_laps : ℝ := 2

/-- Time taken to complete the given number of laps -/
noncomputable def time_taken (pace : ℝ) (track_length : ℝ) (num_laps : ℝ) : ℝ :=
  (track_length * num_laps) / pace

theorem sean_jogging_time :
  time_taken sean_pace track_length num_laps = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sean_jogging_time_l238_23838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_a_l238_23836

theorem existence_of_a (f : ℝ → ℝ) 
  (h_pos : ∀ x, f x > 0)
  (h_mono : ∀ x y, x ≤ y → f x ≤ f y) :
  ∃ a : ℝ, f (a + (1 / f a)) < 2 * f a := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_a_l238_23836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l238_23869

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 2 then Real.log x - a * x else 0

-- State the theorem
theorem odd_function_a_value (a : ℝ) :
  (a > 1/2) →
  (∀ x, -2 < x ∧ x < 0 → f a x ≥ 1) →
  (∀ x y, f a x = -f a (-y)) →
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l238_23869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_is_floor_l238_23855

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧
  f (-1) = -1 ∧
  (∀ x, 0 < x → x < 1 → f x ≤ f 0) ∧
  (∀ x y, f (x + y) ≥ f x + f y) ∧
  (∀ x y, f (x + y) ≤ f x + f y + 1)

theorem unique_function_is_floor (f : ℝ → ℝ) (h : satisfies_conditions f) :
  ∀ x, f x = ↑(floor x) := by
  sorry

#check unique_function_is_floor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_is_floor_l238_23855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_die_roll_sum_bounds_l238_23862

/-- Represents a die with the property that opposite faces sum to 7 -/
structure Die where
  faces : Fin 6 → ℕ
  opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents a path on the chessboard -/
def ChessPath := List (Fin 50 × Fin 50)

/-- The sum of numbers recorded on the cells for a given path and die -/
def pathSum (d : Die) (p : ChessPath) : ℕ := sorry

/-- The set of all valid paths from (0,0) to (49,49) moving only right or up -/
def validPaths : Set ChessPath := sorry

theorem die_roll_sum_bounds :
  ∀ (d : Die) (p : ChessPath),
    p ∈ validPaths → 8745 ≤ pathSum d p ∧ pathSum d p ≤ 8754 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_die_roll_sum_bounds_l238_23862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l238_23824

-- Define the points and the line
def A : ℝ × ℝ := (0, 2)
def line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem min_distance_sum :
  ∃ (P Q : ℝ × ℝ),
    line P.1 P.2 ∧ 
    circleEq Q.1 Q.2 ∧ 
    ∀ (P' Q' : ℝ × ℝ), 
      line P'.1 P'.2 → 
      circleEq Q'.1 Q'.2 → 
      distance A P + distance P Q ≤ distance A P' + distance P' Q' ∧
      distance A P + distance P Q = Real.sqrt 61 - Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l238_23824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_plus_e_minus_six_floor_l238_23823

-- Define the greatest integer function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem pi_plus_e_minus_six_floor : floor (Real.pi + Real.exp 1 - 6) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_plus_e_minus_six_floor_l238_23823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l238_23817

-- Define the polar coordinate system
def PolarCoordinate := ℝ × ℝ

-- Define the curves
noncomputable def curve1 (θ : ℝ) : PolarCoordinate := (2 * Real.sin θ, θ)
noncomputable def curve2 (θ : ℝ) : PolarCoordinate := (-1 / Real.cos θ, θ)

-- Define the intersection point
noncomputable def intersection_point : PolarCoordinate := (Real.sqrt 2, 3 * Real.pi / 4)

-- Theorem statement
theorem intersection_point_correct :
  ∃ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  curve1 θ = intersection_point ∧
  curve2 θ = intersection_point := by
  sorry

#check intersection_point_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l238_23817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suspension_days_per_instance_l238_23853

/-- The number of fingers and toes of a typical person -/
def typical_person_digits : ℕ := 20

/-- The number of bullying instances -/
def bullying_instances : ℕ := 20

/-- The total number of suspension days -/
def total_suspension_days : ℕ := 3 * typical_person_digits

/-- The number of days suspended for each instance of bullying -/
def days_per_instance : ℚ := (total_suspension_days : ℚ) / bullying_instances

theorem suspension_days_per_instance :
  days_per_instance = 3 := by
  unfold days_per_instance total_suspension_days typical_person_digits bullying_instances
  norm_num

#eval days_per_instance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suspension_days_per_instance_l238_23853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platinum_earrings_theorem_l238_23875

/-- Represents the composition of ornaments in a basket -/
structure OrnamentBasket where
  total : ℝ
  rings : ℝ
  earrings : ℝ
  platinum_earrings : ℝ

/-- The percentage of ornaments that are platinum earrings -/
noncomputable def platinum_earrings_percentage (basket : OrnamentBasket) : ℝ :=
  basket.platinum_earrings / basket.total * 100

/-- Theorem: In a basket where 30% are rings and 70% of earrings are platinum,
    49% of all ornaments are platinum earrings -/
theorem platinum_earrings_theorem (basket : OrnamentBasket) 
  (h1 : basket.rings / basket.total = 0.3)
  (h2 : basket.earrings / basket.total = 0.7)
  (h3 : basket.platinum_earrings / basket.earrings = 0.7) :
  platinum_earrings_percentage basket = 49 := by
  sorry

#check platinum_earrings_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platinum_earrings_theorem_l238_23875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_l238_23831

theorem triangle_obtuse (A B C : ℝ) (h1 : 0 < B) (h2 : B < π / 2) (h3 : 0 < C) (h4 : C < π / 2) 
  (h5 : Real.sin B < Real.cos C) : π / 2 < A :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_l238_23831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equation_nested_radical_positive_nested_radical_fixed_point_l238_23892

/-- The value of the infinite nested radical sqrt(3 - sqrt(3 - sqrt(3 - ...))) -/
noncomputable def nestedRadical : ℝ :=
  (-1 + Real.sqrt 13) / 2

/-- Theorem stating that the nested radical satisfies the equation x^2 + x - 3 = 0 -/
theorem nested_radical_equation : nestedRadical^2 + nestedRadical - 3 = 0 := by
  sorry

/-- Theorem stating that the nested radical is the positive solution of x^2 + x - 3 = 0 -/
theorem nested_radical_positive : nestedRadical > 0 := by
  sorry

/-- Theorem stating that the nested radical equals sqrt(3 - nestedRadical) -/
theorem nested_radical_fixed_point : nestedRadical = Real.sqrt (3 - nestedRadical) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equation_nested_radical_positive_nested_radical_fixed_point_l238_23892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l238_23879

-- Define the normal distribution
noncomputable def normal_distribution (μ : ℝ) (σ : ℝ) : ℝ → ℝ := 
  λ x => (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

-- Define the cumulative distribution function (CDF) for the normal distribution
noncomputable def normal_cdf (μ : ℝ) (σ : ℝ) : ℝ → ℝ := sorry

-- State the theorem
theorem normal_distribution_probability (μ σ : ℝ) 
  (h1 : σ > 0)
  (h2 : normal_cdf μ σ (μ + σ) - normal_cdf μ σ (μ - σ) = 0.6826)
  (h3 : normal_cdf μ σ (μ + 2*σ) - normal_cdf μ σ (μ - 2*σ) = 0.9544)
  (h4 : normal_cdf μ σ (μ + 3*σ) - normal_cdf μ σ (μ - 3*σ) = 0.9974)
  (h5 : ∀ x, normal_distribution μ σ x ≤ 1 / (2 * Real.sqrt (2 * Real.pi)))
  (h6 : ∀ x, normal_distribution μ σ x = normal_distribution μ σ (-x)) :
  normal_cdf 0 2 (-2) - normal_cdf 0 2 (-4) = 0.1359 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l238_23879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_center_of_gravity_l238_23818

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents the center of gravity of a uniform plate -/
def centerOfGravity (shape : Quadrilateral) : Point := sorry

/-- Represents the centroid of a triangle -/
def centroid (A B C : Point) : Point := sorry

/-- Divides a line segment into three equal parts -/
def divideIntoThirds (A B : Point) : (Point × Point) := sorry

/-- Constructs the parallelogram from the divided sides of the quadrilateral -/
def constructParallelogram (q : Quadrilateral) : Quadrilateral := sorry

/-- The center of a parallelogram -/
def parallelogramCenter (p : Quadrilateral) : Point := sorry

/-- Predicate to check if a quadrilateral is convex -/
def IsConvex (q : Quadrilateral) : Prop := sorry

theorem quadrilateral_center_of_gravity 
  (q : Quadrilateral) 
  (h_convex : IsConvex q) 
  (h_triangle : ∀ A B C : Point, centerOfGravity {A := A, B := B, C := C, D := C} = centroid A B C) :
  centerOfGravity q = parallelogramCenter (constructParallelogram q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_center_of_gravity_l238_23818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_when_floor_sqrt_n_is_5_l238_23835

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the theorem
theorem max_n_when_floor_sqrt_n_is_5 :
  ∀ n : ℕ, (floor (Real.sqrt n) = 5) → n ≤ 35 ∧ ∀ m : ℕ, m > 35 → floor (Real.sqrt m) ≠ 5 :=
by
  sorry

#check max_n_when_floor_sqrt_n_is_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_when_floor_sqrt_n_is_5_l238_23835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l238_23877

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x+1) + 3

theorem f_range :
  ∀ y ∈ Set.range f,
  (∃ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f x = y) →
  y ∈ Set.Icc (3/2 : ℝ) (1 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l238_23877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cos_y_l238_23828

/-- In a right triangle XYZ, if sin Y = 3/5, then cos Y = 4/5 -/
theorem right_triangle_cos_y (X Y Z : ℝ) (h1 : 0 < X) (h2 : 0 < Y) (h3 : 0 < Z) :
  X^2 + Y^2 = Z^2 → Real.sin Y = 3/5 → Real.cos Y = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cos_y_l238_23828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_transformation_l238_23801

open Real

-- Define vectors in ℝ³
variable (a b c : ℝ × ℝ × ℝ)

-- Define the cross product for ℝ³ vectors
def cross (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1,
   v.2.2 * w.1 - v.1 * w.2.2,
   v.1 * w.2.1 - v.2.1 * w.1)

-- Define the dot product for ℝ³ vectors
def dot (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2.1 * w.2.1 + v.2.2 * w.2.2

-- Define scalar multiplication for ℝ³ vectors
def smul (r : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (r * v.1, r * v.2.1, r * v.2.2)

-- Define vector addition for ℝ³ vectors
def vadd (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.1 + w.1, v.2.1 + w.2.1, v.2.2 + w.2.2)

-- Define the volume of the original parallelepiped
def original_volume (a b c : ℝ × ℝ × ℝ) : ℝ := 
  abs (dot a (cross b c))

-- Define the volume of the new parallelepiped
def new_volume (a b c : ℝ × ℝ × ℝ) : ℝ := 
  abs (dot (vadd a b) (cross (vadd b (smul 2 c)) (vadd c (smul (-5) a))))

-- Theorem statement
theorem parallelepiped_volume_transformation 
  (h : original_volume a b c = 4) : 
  new_volume a b c = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_transformation_l238_23801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_inequality_l238_23807

open BigOperators

theorem remainder_sum_inequality (n : ℕ) (hn : n > 0) :
  let S_n := ∑ a in Finset.range n, ∑ b in Finset.range n, (a * b) % n
  (1 / 2 : ℝ) - 1 / Real.sqrt n ≤ (S_n : ℝ) / (n ^ 3 : ℝ) ∧ (S_n : ℝ) / (n ^ 3 : ℝ) ≤ 1 / 2 := by
  sorry

#check remainder_sum_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_inequality_l238_23807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_run_time_l238_23814

/-- The time it takes Carlos to run around the entire block -/
def carlos_time : ℝ := 3

/-- The time it takes Diego to run around half the block -/
def diego_half_time : ℝ := 2.5

/-- The average time for both racers in seconds -/
def average_time : ℝ := 240

theorem carlos_run_time : carlos_time = 3 :=
  by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_run_time_l238_23814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_eight_squared_l238_23821

theorem cube_root_negative_eight_squared : ((-8 : ℝ) ^ (1/3 : ℝ)) ^ 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_eight_squared_l238_23821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_theorem_l238_23870

-- Define the angles a and β
variable (a β : Real)

-- Define the conditions
axiom sin_a : Real.sin a = 1/3
axiom sin_β : Real.sin β = 1/3
axiom cos_opposite : Real.cos a = -Real.cos β

-- State the theorem
theorem cos_difference_theorem : Real.cos (a - β) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_theorem_l238_23870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_KDL_is_right_angle_l238_23890

-- Define the points
variable (A B C D E F P Q S K L : EuclideanPlane)

-- Define the hexagon
def is_cyclic_hexagon (A B C D E F : EuclideanPlane) : Prop := sorry

-- Define perpendicularity
def perpendicular (a b c d : EuclideanPlane) : Prop := sorry

-- Define equality of segments
def seg_eq (a b c d : EuclideanPlane) : Prop := sorry

-- Define intersection of lines
def intersection (a b c d p : EuclideanPlane) : Prop := sorry

-- Define the concept of "same side"
def same_side (P Q D : EuclideanPlane) : Prop := sorry

-- Define the concept of "opposite side"
def opposite_side (A D : EuclideanPlane) : Prop := sorry

-- Define midpoint
def is_midpoint (S A D : EuclideanPlane) : Prop := sorry

-- Define incenter
def incenter (K B P S : EuclideanPlane) : Prop := sorry

-- Define angle measure
noncomputable def angle_measure (K D L : EuclideanPlane) : ℝ := sorry

-- State the theorem
theorem angle_KDL_is_right_angle 
  (h1 : is_cyclic_hexagon A B C D E F)
  (h2 : perpendicular A B B D)
  (h3 : seg_eq B C E F)
  (h4 : intersection B C A D P)
  (h5 : intersection E F A D Q)
  (h6 : same_side P Q D)
  (h7 : opposite_side A D)
  (h8 : is_midpoint S A D)
  (h9 : incenter K B P S)
  (h10 : incenter L E Q S) :
  angle_measure K D L = 90 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_KDL_is_right_angle_l238_23890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l238_23813

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem range_of_g (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∃ p : ℝ, p > 0 ∧ is_periodic f p) →
  (g = λ x ↦ f x - 2 * x) →
  (∃ a b : ℝ, a ∈ Set.Icc 2 3 ∧ b ∈ Set.Icc 2 3 ∧ 
    g a = -2 ∧ g b = 6 ∧ 
    ∀ x ∈ Set.Icc 2 3, -2 ≤ g x ∧ g x ≤ 6) →
  ∃ x y : ℝ, x ∈ Set.Icc (-2017) 2017 ∧ y ∈ Set.Icc (-2017) 2017 ∧ 
    g x = -4030 ∧ g y = 4044 ∧
    ∀ z ∈ Set.Icc (-2017) 2017, -4030 ≤ g z ∧ g z ≤ 4044 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l238_23813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l238_23888

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2 + Real.sin x

theorem max_a_value (a : ℝ) :
  (∀ x > 0, f (a - x * Real.exp x) + f (Real.log x + x + 1) ≤ 0) →
  a ≤ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l238_23888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_length_eight_is_valid_l238_23833

/-- A predicate stating that three real numbers can form a triangle. -/
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (a b c : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) → (a + b > c ∧ b + c > a ∧ c + a > b) ↔ IsTriangle a b c :=
sorry

theorem third_side_length (c : ℝ) : 
  (c > 0) → 
  (10 + 5 > c) → 
  (|10 - 5| < c) → 
  (5 < c ∧ c < 15) ∧ 
  IsTriangle 10 5 c :=
sorry

theorem eight_is_valid : IsTriangle 10 5 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_length_eight_is_valid_l238_23833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l238_23822

/-- Line l in polar coordinates -/
def line_l (ρ θ : ℝ) : Prop := ρ * Real.cos (θ + Real.pi / 3) = 1

/-- Circle C in parametric form -/
def circle_C (r θ x y : ℝ) : Prop := x = r * Real.cos θ ∧ y = r * Real.sin θ

/-- Tangency condition (distance from origin to line equals radius) -/
def tangent_condition (r : ℝ) : Prop := r = 2 / Real.sqrt (1^2 + (-Real.sqrt 3)^2)

theorem line_tangent_to_circle (r : ℝ) :
  (∃ ρ θ x y : ℝ, line_l ρ θ ∧ circle_C r θ x y ∧ tangent_condition r) → r = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l238_23822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_on_absolute_value_graph_l238_23805

/-- The cosine of an angle whose terminal side is on y = -|x| is ± √2/2 -/
theorem cosine_on_absolute_value_graph (α : ℝ) :
  (∃ x : ℝ, x * Real.cos α = |x| ∧ x * Real.sin α = -|x|) →
  Real.cos α = Real.sqrt 2 / 2 ∨ Real.cos α = -(Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_on_absolute_value_graph_l238_23805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l238_23866

def M : Set ℝ := {x | x^2 + x - 6 ≤ 0}
def N : Set ℝ := {x | |2*x + 1| > 3}

theorem intersection_M_N : M ∩ N = Set.Ioc (-3) (-2) ∪ Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l238_23866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_dates_2008_l238_23873

/-- A prime date is a date where both the month and day are prime numbers. -/
def isPrimeDate (month : ℕ) (day : ℕ) : Bool :=
  Nat.Prime month && Nat.Prime day

/-- The number of days in each month of 2008 (a leap year) -/
def daysInMonth2008 : ℕ → ℕ
| 2 => 29  -- February (leap year)
| 4 | 6 | 9 | 11 => 30  -- April, June, September, November
| _ => 31  -- All other months

/-- Counts the number of prime dates in a given month of 2008 -/
def primeDatesInMonth (month : ℕ) : ℕ :=
  (List.range (daysInMonth2008 month)).filter (fun day => isPrimeDate month (day + 1)) |>.length

/-- The total number of prime dates in 2008 -/
def totalPrimeDates2008 : ℕ :=
  (List.range 12).map (fun month => primeDatesInMonth (month + 1)) |>.sum

theorem prime_dates_2008 : totalPrimeDates2008 = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_dates_2008_l238_23873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_plane_angle_theorem_l238_23803

/-- The angle between an intersecting plane and the base of a cube -/
noncomputable def intersecting_plane_angle (m n : ℝ) : ℝ :=
  Real.arctan (2 * m / (m + n))

/-- Theorem stating the relationship between the intersecting plane angle and volume ratio -/
theorem intersecting_plane_angle_theorem (m n : ℝ) (h : m ≤ n) :
  ∃ (α : ℝ), α = intersecting_plane_angle m n ∧
  α = Real.arctan (2 * m / (m + n)) ∧
  (∃ (cube_volume plane_volume : ℝ),
    cube_volume > 0 ∧
    plane_volume / (cube_volume - plane_volume) = m / n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_plane_angle_theorem_l238_23803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_l238_23858

def m : ℕ := 2^20 * 3^15 * 5^6

theorem divisor_count : 
  (Finset.filter (fun d => d ∣ m^2 ∧ d < m ∧ ¬(d ∣ m)) (Finset.range (m + 1))).card = 5924 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_l238_23858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_is_one_l238_23867

/-- The radius of the base circle of a cone formed by wrapping a sector -/
noncomputable def base_circle_radius (sector_radius : ℝ) (central_angle : ℝ) : ℝ :=
  sector_radius * (central_angle / (2 * Real.pi))

/-- Theorem: The radius of the base circle of a cone formed by wrapping a sector
    with radius 4 and central angle 90° is 1 -/
theorem cone_base_radius_is_one :
  base_circle_radius 4 (Real.pi / 2) = 1 := by
  unfold base_circle_radius
  simp [Real.pi]
  -- The rest of the proof would go here
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_is_one_l238_23867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_eleven_l238_23840

def octahedral_die : Finset ℕ := {2, 3, 4, 5, 6, 7, 8, 9}

def sum_is_eleven (roll : ℕ × ℕ) : Bool := roll.1 + roll.2 = 11

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (octahedral_die.product octahedral_die).filter (fun x => sum_is_eleven x)

theorem probability_sum_eleven :
  (favorable_outcomes.card : ℚ) / ((octahedral_die.card * octahedral_die.card) : ℚ) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_eleven_l238_23840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l238_23899

/-- A three-digit number with digits a, b, and c -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_range : a ≥ 1 ∧ a ≤ 9
  b_range : b ≥ 1 ∧ b ≤ 9
  c_range : c ≥ 1 ∧ c ≤ 9

/-- The property that the difference between consecutive digits is 3 -/
def ConsecutiveDiffThree (n : ThreeDigitNumber) : Prop :=
  n.b = n.a + 3 ∧ n.c = n.b + 3

/-- The set of three-digit numbers with consecutive digit difference of 3 -/
def ValidNumbers : Set ThreeDigitNumber :=
  {n : ThreeDigitNumber | ConsecutiveDiffThree n}

/-- Proof that ValidNumbers is finite -/
instance : Fintype ValidNumbers := by
  sorry

theorem count_valid_numbers : Fintype.card ValidNumbers = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l238_23899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_orthocenter_property_l238_23872

open Set
open Real
open InnerProductSpace

/-- Definition of a circle in R² with center at the origin -/
def Circle (radius : ℝ) : Set (Fin 2 → ℝ) :=
  {v | ‖v‖ = radius}

/-- Definition of the interior of a circle -/
def CircleInterior (radius : ℝ) : Set (Fin 2 → ℝ) :=
  {v | ‖v‖ < radius}

/-- Definition of an inscribed triangle in a circle -/
def InscribedTriangle (c : Set (Fin 2 → ℝ)) : Set (Fin 3 → Fin 2 → ℝ) :=
  {t | ∀ i, t i ∈ c}

/-- Definition of the orthocenter of a triangle -/
def Orthocenter (t : Fin 3 → Fin 2 → ℝ) : Fin 2 → ℝ :=
  t 0 + t 1 + t 2

theorem concentric_circles_orthocenter_property (R : ℝ) (hR : R > 0) :
  (∀ t ∈ InscribedTriangle (Circle R), Orthocenter t ∈ CircleInterior (3 * R)) ∧
  (∀ p ∈ CircleInterior (3 * R), ∃ t ∈ InscribedTriangle (Circle R), Orthocenter t = p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_orthocenter_property_l238_23872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l238_23847

def U : Set ℕ := {x | x < 7 ∧ x > 0}
def A : Set ℕ := {1, 2, 5}
def B : Set ℕ := {2, 3, 4, 5}

theorem set_operations :
  (A ∩ B = {2, 5}) ∧
  ((U \ A) = {3, 4, 6}) ∧
  (A ∪ (U \ B) = {1}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l238_23847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_semicircle_integral_piecewise_function_l238_23816

noncomputable section

-- Part I
theorem integral_semicircle : ∫ x in (0)..(3), Real.sqrt (9 - x^2) = (9 * Real.pi) / 4 := by sorry

-- Part II
def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x^2
  else if 1 ≤ x ∧ x ≤ Real.exp 1 then 1/x
  else 0

theorem integral_piecewise_function : ∫ x in (0)..(Real.exp 1), f x = 4/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_semicircle_integral_piecewise_function_l238_23816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_binomial_congruence_l238_23885

/-- Binomial coefficient definition -/
def binomial (n : ℕ) (k : ℕ) : ℚ :=
  if k > n then 0
  else (Finset.range k).prod (fun i => (n - i : ℚ)) / (Finset.range k).prod (fun i => (i + 1 : ℚ))

/-- The theorem to be proved -/
theorem smallest_k_for_binomial_congruence :
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (b x : ℕ), b > 0 → x > 0 → 
    (binomial (x + k * b) 12 - binomial x 12).num % b = 0) ∧
  (∀ (k' : ℕ), 0 < k' → k' < k → 
    ∃ (b x : ℕ), b > 0 ∧ x > 0 ∧ 
      (binomial (x + k' * b) 12 - binomial x 12).num % b ≠ 0) ∧
  k = 27720 := by
  sorry

#check smallest_k_for_binomial_congruence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_binomial_congruence_l238_23885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_minus_3_l238_23863

-- Define a function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+2)
def domain_f_plus_2 : Set ℝ := Set.Ioo (-2) 2

-- State the theorem
theorem domain_f_minus_3 (h : ∀ x, f (x + 2) ∈ domain_f_plus_2 ↔ x ∈ domain_f_plus_2) :
  ∀ x, f (x - 3) ∈ Set.Ioo 3 7 ↔ x ∈ Set.Ioo 3 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_minus_3_l238_23863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_not_arithmetic_sequence_l238_23842

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log a - a * Real.log x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) (λ i ↦ f a (x^(i+1)))

-- Part 1: Range of a
theorem range_of_a (a : ℝ) :
  (a > 1) →
  (∀ x ≥ 4, f a x ≥ 0) →
  (2 ≤ a ∧ a ≤ 4) := by
  sorry

-- Part 2: g(x,n), g(x,2n), g(x,3n) cannot form an arithmetic sequence
theorem not_arithmetic_sequence (a : ℝ) (x : ℝ) (n : ℕ) :
  (a > 1) →
  (0 < x ∧ x < 1) →
  (n > 0) →
  (g a x n + g a x (3*n) - 2 * g a x (2*n) > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_not_arithmetic_sequence_l238_23842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_products_composite_l238_23868

-- Define the set of natural numbers from 2 to 101
def numbers : Finset ℕ := Finset.filter (fun n => 2 ≤ n ∧ n ≤ 101) (Finset.range 102)

-- Define a partition of the set into two equal subsets
def partition (S T : Finset ℕ) : Prop :=
  S ∪ T = numbers ∧ S ∩ T = ∅ ∧ S.card = 50

-- Define the product of numbers in a set
def product (S : Finset ℕ) : ℕ := S.prod id

-- Define a composite number
def isComposite (n : ℕ) : Prop := n > 1 ∧ ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b

-- The theorem to prove
theorem sum_of_products_composite (S T : Finset ℕ) :
  partition S T → isComposite (product S + product T) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_products_composite_l238_23868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_plus_cos_l238_23845

theorem tan_value_from_sin_plus_cos (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin θ + Real.cos θ = 7/13) : 
  Real.tan θ = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_plus_cos_l238_23845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_rotation_120_l238_23826

noncomputable def rotation_matrix (angle : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos angle, -Real.sin angle],
    ![Real.sin angle,  Real.cos angle]]

theorem smallest_power_rotation_120 :
  ∃ k : ℕ+, k = 3 ∧
  (∀ m : ℕ+, m < k → rotation_matrix (2 * Real.pi / 3) ^ m.val ≠ 1) ∧
  rotation_matrix (2 * Real.pi / 3) ^ k.val = 1 := by
  sorry

#check smallest_power_rotation_120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_rotation_120_l238_23826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_l238_23825

theorem limit_of_sequence (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N →
    |((n + 2 : ℝ)^2 - (n - 2 : ℝ)^2) / (n + 3 : ℝ)^2 - 0| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_l238_23825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_points_l238_23844

/-- Given two points A and B, and two lines passing through them, prove that the line AB has the equation 3x - y = 0 -/
theorem line_equation_through_points (m : ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 3)
  let line_through_A : ℝ → ℝ → Prop := λ x y ↦ x + m * y = 0
  let line_through_B : ℝ → ℝ → Prop := λ x y ↦ m * x - y - m + 3 = 0
  (line_through_A A.1 A.2) →
  (line_through_B B.1 B.2) →
  ∀ x y, (y - A.2) * (B.1 - A.1) = (B.2 - A.2) * (x - A.1) ↔ 3 * x - y = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_points_l238_23844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_pairs_l238_23865

def satisfies_inequalities (a b : ℤ) : Prop :=
  (a^2 + b^2 < 15) ∧ (a^2 + b^2 < 8*a) ∧ (a^2 + b^2 < 8*b + 8)

theorem count_satisfying_pairs :
  ∃! (S : Finset (ℤ × ℤ)), 
    (∀ (a b : ℤ), (a, b) ∈ S ↔ satisfies_inequalities a b) ∧
    S.card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_pairs_l238_23865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l238_23819

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem f_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l238_23819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_coordinates_l238_23827

noncomputable def angle_between_vectors : ℝ := 2 * Real.pi / 3

noncomputable def vector_a : Fin 2 → ℝ
  | 0 => Real.sqrt 3
  | 1 => 1
  | _ => 0

def norm_b : ℝ := 1

theorem vector_b_coordinates :
  ∃ (x y : ℝ),
    (x = 0 ∧ y = -1) ∨
    (x = -Real.sqrt 3 / 2 ∧ y = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_coordinates_l238_23827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeta_frac_sum_eq_one_sixth_l238_23829

/-- The Riemann zeta function -/
noncomputable def zeta (x : ℝ) : ℝ := ∑' n, (n : ℝ)^(-x)

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

/-- The sum of fractional parts of zeta(2k) for k from 2 to ∞ -/
noncomputable def zeta_frac_sum : ℝ := ∑' k : ℕ, frac (zeta (2 * ↑k))

/-- Theorem stating that the sum of fractional parts of zeta(2k) for k from 2 to ∞ equals 1/6 -/
theorem zeta_frac_sum_eq_one_sixth :
  zeta_frac_sum = 1/6 := by
  sorry

#check zeta_frac_sum_eq_one_sixth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeta_frac_sum_eq_one_sixth_l238_23829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_not_three_elements_l238_23820

theorem solution_set_not_three_elements
  (a b m n p : ℝ)
  (ha : a > 0)
  (ha_neq : a ≠ 1)
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (hp : p ≠ 0) :
  ¬ (∃ (S : Finset ℝ), S.card = 3 ∧
    (∀ x, x ∈ S ↔ m * (a ^ |x - b|)^2 + n * (a ^ |x - b|) + p = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_not_three_elements_l238_23820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_and_set_size_l238_23884

theorem polynomial_property_and_set_size 
  (A : Finset ℤ) 
  (h_size : 3 ≤ A.card)
  (M : ℤ) 
  (h_M : ∀ a ∈ A, a ≤ M)
  (m : ℤ) 
  (h_m : ∀ a ∈ A, m ≤ a)
  (P : ℤ → ℤ)
  (h_P_bounds : ∀ a ∈ A, m < P a ∧ P a < M)
  (h_P_strict : ∀ a ∈ A, a ≠ m → a ≠ M → P m < P a) :
  (A.card < 6) ∧ 
  (∃ b c : ℤ, ∀ x ∈ A, P x + x^2 + b*x + c = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_and_set_size_l238_23884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l238_23878

-- Define the sets A and B
def A : Set ℝ := {x | |x - 3| < 2}
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Set.Ici (-1) ∩ Set.Iio 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l238_23878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_a_eq_plus_minus_one_l238_23859

/-- A function f is odd if f(-x) = -f(x) for all x in the domain of f -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = (2^x - a) / (2^x + a) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (2^x - a) / (2^x + a)

/-- Theorem: The function f(x) = (2^x - a) / (2^x + a) is odd if and only if a = 1 or a = -1 -/
theorem f_is_odd_iff_a_eq_plus_minus_one (a : ℝ) :
  IsOdd (f a) ↔ a = 1 ∨ a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_a_eq_plus_minus_one_l238_23859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_PQRS_approx_l238_23880

-- Define the tetrahedron KLNM
structure Tetrahedron :=
  (KL MN KM LN KN ML : ℝ)

-- Define the inscribed circle centers
structure InscribedCenters :=
  (P Q R S : ℝ × ℝ × ℝ)

-- Define the volume function for PQRS
noncomputable def volume_PQRS (t : Tetrahedron) (c : InscribedCenters) : ℝ := sorry

-- Theorem statement
theorem volume_of_PQRS_approx (t : Tetrahedron) (c : InscribedCenters) :
  t.KL = 4 ∧ t.MN = 4 ∧ t.KM = 5 ∧ t.LN = 5 ∧ t.KN = 6 ∧ t.ML = 6 →
  |volume_PQRS t c - 0.29| < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_PQRS_approx_l238_23880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sensitivity_l238_23846

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the larger root of a quadratic equation -/
noncomputable def largerRoot (eq : QuadraticEquation) : ℝ :=
  (-eq.b + Real.sqrt (eq.b^2 - 4*eq.a*eq.c)) / (2*eq.a)

/-- Checks if two real numbers are within 0.001 of each other -/
def withinTolerance (x y : ℝ) : Prop :=
  |x - y| ≤ 0.001

theorem quadratic_root_sensitivity :
  ∃ (original perturbed : QuadraticEquation),
    withinTolerance original.a perturbed.a ∧
    withinTolerance original.b perturbed.b ∧
    withinTolerance original.c perturbed.c ∧
    |largerRoot original - largerRoot perturbed| > 1000 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sensitivity_l238_23846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l238_23837

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a + c = 4 and sin A * (1 + cos B) = (2 - cos A) * sin B,
    then the maximum area of the triangle is √3. -/
theorem triangle_max_area (a b c A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a + c = 4 →
  Real.sin A * (1 + Real.cos B) = (2 - Real.cos A) * Real.sin B →
  (∃ (S : ℝ), S = (1/2) * a * c * Real.sin B ∧ 
    ∀ (S' : ℝ), S' = (1/2) * a * c * Real.sin B → S' ≤ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l238_23837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eight_equals_87_l238_23891

/-- A function satisfying the given property for all positive real numbers -/
noncomputable def f : ℝ → ℝ := sorry

/-- The property that f satisfies for all positive real numbers -/
axiom f_property : ∀ x : ℝ, x > 0 → 3 * f x + 7 * f (2016 / x) = 2 * x

/-- The theorem to be proved -/
theorem f_eight_equals_87 : f 8 = 87 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eight_equals_87_l238_23891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_odd_vertices_split_l238_23898

/-- Represents a convex polygon with a given number of vertices -/
def convex_polygon (num_vertices : ℕ) : Type := sorry

/-- Predicate that checks if a diagonal splits a polygon into two polygons with odd vertices -/
def diagonal_splits_polygon_odd_vertices 
  (polygon : Type) (n : ℕ) (diagonal : Fin (n^2)) : Prop := sorry

theorem diagonal_odd_vertices_split (n : ℕ) : 
  ∃ (diagonal : Fin (n^2)), 
    diagonal_splits_polygon_odd_vertices (convex_polygon (2*n+2)) n diagonal :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_odd_vertices_split_l238_23898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_below_iff_a_geq_one_l238_23886

open Real

/-- The function f(x) defined as ax - a + ln x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - a + log x

/-- The theorem stating the equivalence between the curve property and the range of a -/
theorem curve_below_iff_a_geq_one (a : ℝ) :
  (∀ x > 1, f a x < a * (x^2 - 1)) ↔ a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_below_iff_a_geq_one_l238_23886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l238_23843

noncomputable def z : ℂ := (5 * Complex.I) / (2 * Complex.I - 1)

theorem z_in_fourth_quadrant : 
  Real.sign z.re = 1 ∧ Real.sign z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l238_23843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_counterexample_l238_23839

def next_quadruple (q : Fin 4 → ℤ) : Fin 4 → ℤ := fun i =>
  match i with
  | 0 => |q 0 - q 1|
  | 1 => |q 1 - q 2|
  | 2 => |q 2 - q 3|
  | 3 => |q 3 - q 0|

def iterate_quadruple (q : Fin 4 → ℤ) (n : ℕ) : Fin 4 → ℤ :=
  match n with
  | 0 => q
  | n+1 => next_quadruple (iterate_quadruple q n)

theorem exists_counterexample : 
  ∃ (q₀ : Fin 4 → ℤ), 
    (∀ i, Even (iterate_quadruple q₀ 4 i)) ∧ 
    (∀ i, ¬(4 ∣ iterate_quadruple q₀ 4 i)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_counterexample_l238_23839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_in_set_l238_23896

/-- Given a set of 100 consecutive multiples of a number, 
    where the smallest number is 108 and the greatest is 900,
    prove that the number whose multiples are in the set is 8. -/
theorem multiples_in_set (s : Set ℕ) (n : ℕ) :
  (∃ k : ℕ, s = (Finset.range 100).image (fun i => n * (k + i))) ∧
  108 ∈ s ∧
  900 ∈ s ∧
  (∀ m ∈ s, 108 ≤ m ∧ m ≤ 900) →
  n = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_in_set_l238_23896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_X_properties_l238_23881

structure DiscreteRandomVariable where
  values : List ℝ
  probabilities : List ℝ
  sum_to_one : probabilities.sum = 1

def X : DiscreteRandomVariable := {
  values := [1, 2, 4, 6],
  probabilities := [0.2, 0.3, 0.4, 0.1],
  sum_to_one := by sorry
}

def expected_value (X : DiscreteRandomVariable) : ℝ :=
  List.sum (List.zipWith (· * ·) X.values X.probabilities)

def variance (X : DiscreteRandomVariable) : ℝ :=
  let μ := expected_value X
  List.sum (List.zipWith (· * ·) (List.map (fun x => (x - μ)^2) X.values) X.probabilities)

noncomputable def standard_deviation (X : DiscreteRandomVariable) : ℝ :=
  Real.sqrt (variance X)

theorem X_properties :
  X.probabilities[2] = 0.4 ∧
  expected_value X = 3 ∧
  standard_deviation X = 2 * Real.sqrt 15 / 5 := by
  sorry

#eval X.probabilities[2]
#eval expected_value X
#eval variance X

end NUMINAMATH_CALUDE_ERRORFEEDBACK_X_properties_l238_23881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_to_curve_l238_23895

noncomputable section

/-- The curve y² = x³ + 39x - 35 -/
def curve (x y : ℝ) : Prop := y^2 = x^3 + 39*x - 35

/-- A line passing through the origin with slope m -/
def line_through_origin (m x y : ℝ) : Prop := y = m * x

/-- The slope of the tangent line to the curve at point (x, y) -/
noncomputable def tangent_slope (x y : ℝ) : ℝ := (3*x^2 + 39) / (2*y)

/-- A point (x, y) is on the curve and the line through the origin with slope m is tangent to the curve at this point -/
def tangent_point (x y m : ℝ) : Prop :=
  curve x y ∧ line_through_origin m x y ∧ m = tangent_slope x y

/-- The main theorem stating the slopes of tangent lines passing through the origin -/
theorem tangent_slopes_to_curve :
  ∀ m : ℝ, (∃ x y : ℝ, tangent_point x y m) ↔ 
    m = Real.sqrt 51 / 2 ∨ m = -Real.sqrt 51 / 2 ∨ m = Real.sqrt 285 / 5 ∨ m = -Real.sqrt 285 / 5 :=
by
  sorry  -- The proof is omitted

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_to_curve_l238_23895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l238_23856

/-- The time taken for a person to cover the length of an escalator -/
noncomputable def escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) : ℝ :=
  escalator_length / (escalator_speed + person_speed)

/-- Theorem: The time taken to cover the escalator length is 20 seconds -/
theorem escalator_problem :
  let escalator_speed : ℝ := 7
  let person_speed : ℝ := 2
  let escalator_length : ℝ := 180
  escalator_time escalator_speed person_speed escalator_length = 20 := by
  -- Unfold the definition and simplify
  unfold escalator_time
  -- Perform the calculation
  norm_num
  -- QED
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l238_23856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_slope_neg_third_tangent_line_through_point_one_zero_l238_23889

noncomputable def curve (x : ℝ) : ℝ := 1 / x

noncomputable def curve_derivative (x : ℝ) : ℝ := -1 / (x^2)

theorem tangent_lines_slope_neg_third :
  ∃ (a b : ℝ), 
    (curve_derivative a = -1/3 ∧ a > 0 → x + 3*y - 2*Real.sqrt 3 = 0) ∧
    (curve_derivative b = -1/3 ∧ b < 0 → x + 3*y + 2*Real.sqrt 3 = 0) := by
  sorry

theorem tangent_line_through_point_one_zero :
  ∃ (b : ℝ), b ≠ 0 ∧ 
    (curve b = 1/b ∧ 
     curve_derivative b = -1/(b^2) ∧ 
     0 - 1/b = -1/(b^2) * (1 - b)) → 
    4*x + y - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_slope_neg_third_tangent_line_through_point_one_zero_l238_23889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l238_23849

/-- Represents a rectangle in 2D space. -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a triangle in 2D space. -/
structure Triangle where
  -- Define the triangle structure (e.g., using vertices or side lengths)
  -- This is a placeholder and should be properly defined based on the problem requirements
  dummy : Unit

/-- Checks if a triangle is equilateral. -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  sorry -- Define the condition for an equilateral triangle

/-- Checks if a triangle is inscribed in a rectangle. -/
def Triangle.isInscribedIn (t : Triangle) (r : Rectangle) : Prop :=
  sorry -- Define the condition for a triangle to be inscribed in a rectangle

/-- Calculates the area of a triangle. -/
noncomputable def Triangle.area (t : Triangle) : ℝ :=
  sorry -- Define the area calculation for a triangle

/-- Given a rectangle with side lengths 12 and 14, the maximum area of an
    equilateral triangle that can be inscribed within the rectangle is 36√3. -/
theorem max_equilateral_triangle_area_in_rectangle :
  ∃ (rect : Rectangle) (tri : Triangle),
    rect.length = 12 ∧
    rect.width = 14 ∧
    tri.isEquilateral ∧
    tri.isInscribedIn rect ∧
    tri.area = 36 * Real.sqrt 3 ∧
    ∀ (other_tri : Triangle),
      other_tri.isEquilateral →
      other_tri.isInscribedIn rect →
      other_tri.area ≤ tri.area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l238_23849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_third_derivative_correct_l238_23806

/-- The function y(x) = (2x+3) ln^2 x -/
noncomputable def y (x : ℝ) : ℝ := (2*x + 3) * (Real.log x)^2

/-- The third derivative of y -/
noncomputable def y_third_derivative (x : ℝ) : ℝ := (4 * Real.log x * (3 - x) - 18) / x^3

/-- Theorem stating that the third derivative of y is equal to y_third_derivative -/
theorem y_third_derivative_correct (x : ℝ) (h : x > 0) : 
  (deriv^[3] y) x = y_third_derivative x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_third_derivative_correct_l238_23806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l238_23893

/-- A polynomial of degree 3 with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluate the polynomial at a given value -/
def evaluate (p : CubicPolynomial) (x : ℚ) : ℚ :=
  x^3 + p.a * x^2 + p.b * x + p.c

/-- Condition that one root is the product of the other two -/
def has_product_root (p : CubicPolynomial) : Prop :=
  ∃ (r s : ℚ), evaluate p r = 0 ∧ evaluate p s = 0 ∧ evaluate p (r*s) = 0

/-- The main theorem -/
theorem divisibility_theorem (p : CubicPolynomial) 
  (h : has_product_root p) : 
  (2 * (evaluate p (-1))) ∣ 
  (evaluate p 1 + evaluate p (-1) - 2 * (1 + evaluate p 0)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l238_23893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_above_line_is_half_l238_23857

/-- A square in the 2D plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A line in the 2D plane defined by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculate the area of a square -/
noncomputable def squareArea (s : Square) : ℝ :=
  let (x1, y1) := s.bottomLeft
  let (x2, y2) := s.topRight
  (x2 - x1) * (y2 - y1)

/-- Calculate the area of the triangle formed by the line and the square -/
noncomputable def triangleArea (l : Line) (s : Square) : ℝ :=
  let (x1, y1) := l.point1
  let (x2, y2) := l.point2
  let base := x2 - x1
  let height := s.topRight.2 - s.bottomLeft.2
  (1/2) * base * height

/-- Calculate the fraction of the square's area above the line -/
noncomputable def fractionAboveLine (s : Square) (l : Line) : ℝ :=
  let squareArea := squareArea s
  let triangleArea := triangleArea l s
  (squareArea - triangleArea) / squareArea

/-- Theorem stating that the fraction of the square's area above the line is 1/2 -/
theorem fraction_above_line_is_half :
  let s : Square := ⟨(1, 0), (4, 3)⟩
  let l : Line := ⟨(1, 3), (4, 0)⟩
  fractionAboveLine s l = 1/2 := by
  sorry

#eval "Theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_above_line_is_half_l238_23857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_time_together_l238_23804

-- Define the painting rates for each person
noncomputable def taylor_rate : ℝ := 1 / 12
noncomputable def jennifer_rate : ℝ := 1 / 10
noncomputable def alex_rate : ℝ := 1 / 15

-- Define the combined rate
noncomputable def combined_rate : ℝ := taylor_rate + jennifer_rate + alex_rate

-- Theorem: The time to paint the room together is 4 hours
theorem paint_time_together : (1 / combined_rate) = 4 := by
  -- Expand the definition of combined_rate
  unfold combined_rate
  -- Expand the definitions of individual rates
  unfold taylor_rate jennifer_rate alex_rate
  -- Simplify the fractions
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_time_together_l238_23804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_250_l238_23850

-- Define the triangle XYZ
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  (t.X.1 - t.Z.1) * (t.Y.1 - t.Z.1) + (t.X.2 - t.Z.2) * (t.Y.2 - t.Z.2) = 0

noncomputable def hypotenuse_length (t : Triangle) : ℝ :=
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2)

def median_through_X : ℝ → ℝ :=
  λ x => x + 1

def median_through_Y : ℝ → ℝ :=
  λ x => 2*x + 2

noncomputable def triangle_area (t : Triangle) : ℝ :=
  abs ((t.X.1 - t.Z.1) * (t.Y.2 - t.Z.2) - (t.Y.1 - t.Z.1) * (t.X.2 - t.Z.2)) / 2

-- Theorem statement
theorem triangle_area_is_250 (t : Triangle) :
  is_right_triangle t ∧
  hypotenuse_length t = 50 ∧
  (∃ m : ℝ, t.X.2 = median_through_X m ∧ t.X.1 = m) ∧
  (∃ n : ℝ, t.Y.2 = median_through_Y n ∧ t.Y.1 = n) →
  triangle_area t = 250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_250_l238_23850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l238_23812

-- Define R as a parameter
variable (R : ℝ)

-- Define the sequence a_n
def a : ℕ → ℝ
  | 0 => R  -- Add case for 0
  | 1 => R
  | n + 1 => a n + 2 * n

-- State the theorem
theorem a_100_value : a R 100 = R + 9900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l238_23812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_price_is_two_l238_23810

/-- Represents the daily production and pricing of bakery items -/
structure BakeryData where
  cupcake_price : ℚ
  biscuit_price : ℚ
  daily_cupcakes : ℕ
  daily_cookies : ℕ
  daily_biscuits : ℕ
  total_earnings : ℚ
  days : ℕ

/-- Calculates the price of cookies given bakery data -/
noncomputable def cookie_price (data : BakeryData) : ℚ :=
  let daily_cupcake_earnings := data.cupcake_price * data.daily_cupcakes
  let daily_biscuit_earnings := data.biscuit_price * data.daily_biscuits
  let total_cupcake_biscuit_earnings := (daily_cupcake_earnings + daily_biscuit_earnings) * data.days
  let cookie_earnings := data.total_earnings - total_cupcake_biscuit_earnings
  let total_cookies := data.daily_cookies * data.days
  cookie_earnings / total_cookies

/-- Theorem stating that the cookie price is $2 given the specific bakery data -/
theorem cookie_price_is_two :
  let data : BakeryData := {
    cupcake_price := 3/2,
    biscuit_price := 1,
    daily_cupcakes := 20,
    daily_cookies := 10,
    daily_biscuits := 20,
    total_earnings := 350,
    days := 5
  }
  cookie_price data = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_price_is_two_l238_23810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinear_and_dot_product_l238_23815

theorem vector_collinear_and_dot_product (a x : Fin 3 → ℝ) : 
  a = ![2, -1, 2] → 
  x = ![-4, 2, -4] → 
  ∃ (k : ℝ), (∀ i, x i = k * a i) ∧ 
  (Finset.sum Finset.univ (λ i => a i * x i) = -18) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinear_and_dot_product_l238_23815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_integral_l238_23832

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Define the conditions
def conditions (a b c d : ℝ) : Prop :=
  f a b c d (-1) = 0 ∧
  f a b c d 1 = 0 ∧
  ∀ x, abs x ≤ 1 → f a b c d x ≥ 1 - abs x

-- Define the integral to be minimized
noncomputable def integral_to_minimize (a b c d : ℝ) : ℝ :=
  ∫ x in (-1)..1, (3 * a * x^2 + 2 * b * x + c - x)^2

theorem minimize_integral (a b c d : ℝ) :
  conditions a b c d →
  integral_to_minimize a b c d ≥ integral_to_minimize 0 (-1) 0 1 := by
  sorry

#check minimize_integral

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_integral_l238_23832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l238_23894

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A point on the hyperbola -/
noncomputable def point_on_hyperbola (h : Hyperbola) : ℝ × ℝ := (4, Real.sqrt 2)

/-- The product of slopes from endpoints of real axis to the point -/
noncomputable def slope_product (h : Hyperbola) : ℝ := 1/4

/-- The standard equation of the hyperbola -/
def standard_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2/8 - y^2/2 = 1 ↔ x^2/h.a^2 - y^2/h.b^2 = 1

/-- A line passing through Q(2,2) that intersects the hyperbola at exactly one point -/
noncomputable def tangent_line (h : Hyperbola) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2*p.2 + 2 = 0) ∨ (p.1 + 2*p.2 - 6 = 0) ∨ 
    (∃ k : ℝ, (k = (-1 + Real.sqrt 10)/2 ∨ k = (-1 - Real.sqrt 10)/2) ∧ 
      p.2 - 2 = k*(p.1 - 2))}

/-- The main theorem -/
theorem hyperbola_properties (h : Hyperbola) : 
  (point_on_hyperbola h).1^2/h.a^2 - (point_on_hyperbola h).2^2/h.b^2 = 1 ∧
  slope_product h = 1/4 →
  standard_equation h ∧
  ∃ l : Set (ℝ × ℝ), l = tangent_line h ∧ 
    (∀ p ∈ l, p.1^2/8 - p.2^2/2 = 1 → 
      ∀ q ∈ l, q.1^2/8 - q.2^2/2 = 1 → p = q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l238_23894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_on_circle_l238_23841

/-- The circle with equation x^2 + y^2 = 4 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The line with equation 4x + 3y - 12 = 0 -/
def line_equation (x y : ℝ) : Prop := 4*x + 3*y - 12 = 0

/-- The distance from a point (x, y) to the line 4x + 3y - 12 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (4*x + 3*y - 12) / Real.sqrt (4^2 + 3^2)

theorem farthest_point_on_circle :
  ∀ x y : ℝ, circle_equation x y →
    distance_to_line x y ≤ distance_to_line (-8/5) (-6/5) := by
  sorry

#check farthest_point_on_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_on_circle_l238_23841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l238_23852

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_problem (principal interest time : ℝ) 
  (h_principal : principal = 800)
  (h_interest : interest = 128)
  (h_time : time = 4) :
  ∃ rate : ℝ, simple_interest principal rate time = interest ∧ rate = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l238_23852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_shift_for_symmetry_l238_23848

/-- Definition of the function f(x) --/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x - Real.sin x

/-- Theorem stating the smallest positive shift for y-axis symmetry --/
theorem smallest_shift_for_symmetry :
  ∃ (n : ℝ), n > 0 ∧
  (∀ (x : ℝ), f (x + n) = f (-x + n)) ∧
  (∀ (m : ℝ), m > 0 ∧ (∀ (x : ℝ), f (x + m) = f (-x + m)) → m ≥ n) ∧
  n = 5 * Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_shift_for_symmetry_l238_23848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_mass_is_15k_l238_23854

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

/-- Calculates the mass of a material segment given its endpoints and density function -/
noncomputable def segmentMass (A B : Point3D) (k : ℝ) : ℝ :=
  let densityFunction (t : ℝ) := k * distance A { x := A.x + t * (B.x - A.x),
                                                  y := A.y + t * (B.y - A.y),
                                                  z := A.z + t * (B.z - A.z) }
  let integrand (t : ℝ) := densityFunction t * distance A B
  ∫ t in Set.Icc 0 1, integrand t

/-- Theorem: The mass of the material segment AB is 15k -/
theorem segment_mass_is_15k (k : ℝ) :
  let A : Point3D := { x := -2, y := 1, z := 0 }
  let B : Point3D := { x := -1, y := 3, z := 5 }
  segmentMass A B k = 15 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_mass_is_15k_l238_23854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l238_23811

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

-- Define the line l
def lineL (x y k : ℝ) : Prop := k * x - y - 2 * k = 0

-- Define the condition that the line always intersects the circle
def always_intersects (k : ℝ) : Prop :=
  ∀ x y, circleC x y → lineL x y k → (∃ x' y', circleC x' y' ∧ lineL x' y' k)

-- State the theorem
theorem min_k_value :
  ∀ k : ℝ, always_intersects k → k ≥ -Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l238_23811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lines_through_single_point_l238_23809

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for lines
structure Line where
  color : Color

-- Define a type for points
structure Point where

-- Define the plane
structure Plane where
  lines : Finset Line
  points : Set Point

-- Function to check if two lines are parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop := sorry

-- Function to check if a point is an intersection of two lines
def intersection (p : Point) (l1 l2 : Line) : Prop := sorry

-- Main theorem
theorem all_lines_through_single_point (plane : Plane) :
  (∀ l1 l2 : Line, l1 ∈ plane.lines → l2 ∈ plane.lines → l1 ≠ l2 → ¬ parallel l1 l2) →
  (∀ p : Point, p ∈ plane.points → ∀ l1 l2 : Line, l1 ∈ plane.lines → l2 ∈ plane.lines →
    l1.color = l2.color → intersection p l1 l2 →
    ∃ l3 : Line, l3 ∈ plane.lines ∧ l3.color ≠ l1.color ∧ pointOnLine p l3) →
  ∃ p : Point, p ∈ plane.points ∧ ∀ l : Line, l ∈ plane.lines → pointOnLine p l :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lines_through_single_point_l238_23809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_area_ratio_l238_23830

/-- SimilarTriangles represents that two triangles are similar -/
def SimilarTriangles (triangle1 triangle2 : Set ℝ × Set ℝ × Set ℝ) : Prop := sorry

/-- SimilarityRatio represents the ratio of corresponding sides of similar triangles -/
def SimilarityRatio (triangle1 triangle2 : Set ℝ × Set ℝ × Set ℝ) : ℝ := sorry

/-- AreaRatio represents the ratio of areas of two triangles -/
def AreaRatio (triangle1 triangle2 : Set ℝ × Set ℝ × Set ℝ) : ℝ := sorry

theorem similar_triangles_area_ratio (triangle1 triangle2 : Set ℝ × Set ℝ × Set ℝ) 
  (h_similar : SimilarTriangles triangle1 triangle2) 
  (h_ratio : SimilarityRatio triangle1 triangle2 = 1 / 3) : 
  AreaRatio triangle1 triangle2 = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_area_ratio_l238_23830
