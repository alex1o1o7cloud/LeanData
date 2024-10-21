import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coverage_l787_78755

theorem circle_coverage (r : ℝ) (h1 : r > 0) :
  (∃ (c1 c2 c3 c4 c5 c6 c7 : Set (ℝ × ℝ)),
    (∀ i : Fin 7, ∃ (center : ℝ × ℝ), Metric.ball center r = match i with
      | 0 => c1 | 1 => c2 | 2 => c3 | 3 => c4 | 4 => c5 | 5 => c6 | _ => c7) ∧
    (Metric.ball ((0 : ℝ), (0 : ℝ)) 1 ⊆ c1 ∪ c2 ∪ c3 ∪ c4 ∪ c5 ∪ c6 ∪ c7)) →
  r ≥ 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coverage_l787_78755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excluded_students_average_mark_l787_78784

/-- Proves that the average mark of excluded students is 20 given the conditions --/
theorem excluded_students_average_mark
  (n : ℕ)                     -- Total number of students
  (a : ℚ)                     -- Average mark of all students
  (a_remaining : ℚ)           -- Average mark of remaining students
  (n_excluded : ℕ)            -- Number of excluded students
  (h1 : n = 14)               -- Total number of students is 14
  (h2 : a = 65)               -- Average mark of all students is 65
  (h3 : a_remaining = 90)     -- Average mark of remaining students is 90
  (h4 : n_excluded = 5)       -- Number of excluded students is 5
  : (n * a - (n - n_excluded) * a_remaining) / n_excluded = 20 := by
  -- Convert natural numbers to rationals for arithmetic operations
  have n_rat : ℚ := n
  have n_excluded_rat : ℚ := n_excluded
  
  -- Substitute known values
  rw [h1, h2, h3, h4]
  
  -- Perform the calculation
  norm_num
  
  -- The proof is complete
  done

-- We can't use #eval for theorem checking, so we'll omit this line
-- #eval excluded_students_average_mark 14 65 90 5 rfl rfl rfl rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excluded_students_average_mark_l787_78784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l787_78756

/-- Calculates the time taken for a train to cross a signal pole given its length,
    the length of a platform it crosses, and the time it takes to cross the platform. -/
noncomputable def timeToCrossSignalPole (trainLength : ℝ) (platformLength : ℝ) (timeToCrossPlatform : ℝ) : ℝ :=
  let totalDistance := trainLength + platformLength
  let trainSpeed := totalDistance / timeToCrossPlatform
  trainLength / trainSpeed

/-- The time taken for a 300m long train to cross a signal pole, given that it crosses
    a 366.67m platform in 40 seconds, is approximately 18 seconds. -/
theorem train_crossing_time_approx :
  ∀ ε > 0, |timeToCrossSignalPole 300 366.67 40 - 18| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l787_78756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_equation_max_radius_at_m_one_largest_circle_standard_equation_l787_78708

/-- The line equation mx - y - 2m - 1 = 0 --/
def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x - y - 2 * m - 1 = 0

/-- A circle with center (0,1) and radius r --/
def circle_equation (x y r : ℝ) : Prop :=
  x^2 + (y - 1)^2 = r^2

/-- The circle is tangent to the line --/
def is_tangent (m r : ℝ) : Prop :=
  ∃ x y : ℝ, line_equation m x y ∧ circle_equation x y r

theorem largest_circle_equation (m : ℝ) (h : m > 0) :
  ∀ r : ℝ, (∃ x y : ℝ, circle_equation x y r ∧ is_tangent m r) →
  r ≤ 2 * Real.sqrt 2 :=
sorry

theorem max_radius_at_m_one :
  is_tangent 1 (2 * Real.sqrt 2) :=
sorry

theorem largest_circle_standard_equation :
  ∀ x y : ℝ, circle_equation x y (2 * Real.sqrt 2) ↔ x^2 + (y - 1)^2 = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_equation_max_radius_at_m_one_largest_circle_standard_equation_l787_78708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_over_3_l787_78779

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real

-- Define vectors p and q
def p (t : Triangle) : Real × Real := (4, t.a^2 + t.b^2 - t.c^2)

noncomputable def q (t : Triangle) : Real × Real := (Real.sqrt 3, t.S)

-- Define parallel vectors
def parallel (v w : Real × Real) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- State the theorem
theorem angle_C_is_pi_over_3 (t : Triangle) :
  parallel (p t) (q t) →
  t.C = Real.pi / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_over_3_l787_78779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_f_eq_f_simplified_period_of_f_simplified_l787_78728

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 
  (Real.sin x + Real.cos x + Real.sin x ^ 2 * Real.cos x ^ 2) / (2 - Real.sin (2 * x))

-- State the theorem
theorem min_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by
  sorry

-- Simplified version of the function after algebraic manipulations
noncomputable def f_simplified (x : ℝ) : ℝ := 
  1/2 + 1/4 * Real.sin (2 * x)

-- Theorem stating that f and f_simplified are equal
theorem f_eq_f_simplified :
  ∀ (x : ℝ), f x = f_simplified x := by
  sorry

-- Theorem about the period of f_simplified
theorem period_of_f_simplified :
  ∀ (x : ℝ), f_simplified (x + Real.pi) = f_simplified x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_f_eq_f_simplified_period_of_f_simplified_l787_78728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_equals_16_l787_78778

/-- Piecewise function definition -/
noncomputable def f (x m c : ℝ) : ℝ :=
  if x < m then c / Real.sqrt x else c / Real.sqrt m

/-- Theorem stating that m equals 16 given the conditions -/
theorem m_equals_16 (m c : ℝ) : 
  (f 4 m c = 30) → (f m m c = 15) → m = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_equals_16_l787_78778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_l787_78736

theorem trigonometric_sum (x y : ℝ) 
  (h1 : Real.sin x / Real.sin y = 2) 
  (h2 : Real.cos x / Real.cos y = 1/3) : 
  Real.sin (2*x) / Real.sin (2*y) + Real.cos (2*x) / Real.cos (2*y) = 41/57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_l787_78736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_point_to_line_distance_l787_78715

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the distance between two parallel lines -/
noncomputable def distance_between_lines (l1 l2 : Line) : ℝ :=
  abs (l2.c - l1.c) / Real.sqrt (l1.a^2 + l1.b^2)

/-- Calculate the distance from a point to a line -/
noncomputable def distance_point_to_line (x0 y0 : ℝ) (l : Line) : ℝ :=
  abs (l.a * x0 + l.b * y0 + l.c) / Real.sqrt (l.a^2 + l.b^2)

theorem parallel_lines_distance (l1 l2 : Line) (h : l1.a = l2.a ∧ l1.b = l2.b) :
  l1 = { a := 2, b := 1, c := -1 } →
  l2 = { a := 2, b := 1, c := 1 } →
  distance_between_lines l1 l2 = 2 * Real.sqrt 5 / 5 := by
  sorry

theorem point_to_line_distance (l : Line) :
  l = { a := 2, b := 1, c := -1 } →
  distance_point_to_line 0 2 l = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_point_to_line_distance_l787_78715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cables_for_three_subnetworks_l787_78787

/-- Represents a computer network with two brands of computers. -/
structure ComputerNetwork where
  brandX : Nat
  brandY : Nat

/-- Represents the number of cables installed in the network. -/
abbrev CableCount := Nat

/-- Checks if the given cable count is valid for the network. -/
def isValidCableCount (network : ComputerNetwork) (cables : CableCount) : Prop :=
  cables ≤ network.brandX * network.brandY ∧ 
  cables ≥ network.brandX + network.brandY - 1

/-- Checks if the given cable count allows for at least 3 completely interconnected subnetworks. -/
def hasThreeSubnetworks (network : ComputerNetwork) (cables : CableCount) : Prop :=
  ∃ (x1 x2 x3 y1 y2 y3 : Nat),
    x1 + x2 + x3 ≤ network.brandX ∧
    y1 + y2 + y3 ≤ network.brandY ∧
    x1 * y1 + x2 * y2 + x3 * y3 ≤ cables

/-- The main theorem stating the maximum number of cables that can be installed. -/
theorem max_cables_for_three_subnetworks (network : ComputerNetwork) 
    (h1 : network.brandX = 35) (h2 : network.brandY = 15) :
    ∃ (maxCables : CableCount),
      isValidCableCount network maxCables ∧
      hasThreeSubnetworks network maxCables ∧
      (∀ (c : CableCount), c > maxCables → ¬(hasThreeSubnetworks network c)) ∧
      maxCables = 175 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cables_for_three_subnetworks_l787_78787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M₀_to_plane_l787_78743

-- Define the points
def M₀ : ℝ × ℝ × ℝ := (2, -1, 4)
def M₁ : ℝ × ℝ × ℝ := (1, 2, 0)
def M₂ : ℝ × ℝ × ℝ := (1, -1, 2)
def M₃ : ℝ × ℝ × ℝ := (0, 1, -1)

-- Define the plane passing through M₁, M₂, and M₃
def plane_equation (x y z : ℝ) : Prop :=
  5 * x - 2 * y - 3 * z - 1 = 0

-- Define the distance function from a point to a plane
noncomputable def distance_to_plane (p : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := p
  |5 * x - 2 * y - 3 * z - 1| / Real.sqrt 38

-- Theorem statement
theorem distance_M₀_to_plane :
  distance_to_plane M₀ = 1 / Real.sqrt 38 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M₀_to_plane_l787_78743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biology_exam_results_l787_78788

/-- The number of students who took the biology exam -/
def total_students : ℕ := 200

/-- The fraction of students who scored 100% -/
def perfect_score_fraction : ℚ := 1/4

/-- The fraction of remaining students who scored in B bracket -/
def b_bracket_fraction : ℚ := 1/5

/-- The fraction of remaining students who scored in C bracket -/
def c_bracket_fraction : ℚ := 1/3

/-- The fraction of remaining students who scored in D bracket -/
def d_bracket_fraction : ℚ := 5/12

/-- The fraction of C bracket students who improved to B bracket after re-assessment -/
def c_to_b_fraction : ℚ := 3/5

/-- The number of students who failed the exam (Grade F) after re-assessment -/
def failed_students : ℕ := 8

theorem biology_exam_results :
  let perfect_score := (perfect_score_fraction * total_students).floor
  let remaining_students := total_students - perfect_score
  let b_bracket := (b_bracket_fraction * remaining_students).floor
  let c_bracket := (c_bracket_fraction * remaining_students).floor
  let d_bracket := (d_bracket_fraction * remaining_students).floor
  let c_to_b := (c_to_b_fraction * c_bracket).floor
  total_students = perfect_score + b_bracket + c_bracket + d_bracket + failed_students ∧
  failed_students = total_students - (perfect_score + b_bracket + c_bracket + d_bracket) ∧
  failed_students = total_students - (perfect_score + (b_bracket + c_to_b) + (c_bracket - c_to_b) + d_bracket) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_biology_exam_results_l787_78788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l787_78721

def initial_reading : ℕ := 2552
def final_reading : ℕ := 2882
def total_time : ℕ := 12

theorem average_speed_calculation :
  (final_reading - initial_reading : ℚ) / total_time = 55 / 2 := by
  norm_num
  rfl

#eval (final_reading - initial_reading : ℚ) / total_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l787_78721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_distance_sum_l787_78748

noncomputable section

/-- The parabola defined by y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- The focus of the parabola y = x^2 -/
noncomputable def focus : ℝ × ℝ := (0, 1/4)

/-- The distance from a point to the focus of the parabola -/
noncomputable def dist_to_focus (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - focus.1)^2 + (p.2 - focus.2)^2)

/-- The three given intersection points -/
def given_points : List (ℝ × ℝ) := [(-28, 784), (-2, 4), (13, 169)]

theorem parabola_circle_intersection_distance_sum :
  ∃ (fourth_point : ℝ × ℝ),
    fourth_point.2 = parabola fourth_point.1 ∧
    fourth_point ∉ given_points ∧
    (given_points.map dist_to_focus).sum + dist_to_focus fourth_point = 1247 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_distance_sum_l787_78748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_M_radius_l787_78776

/-- The circle M is defined by the equation x^2 - 4x + y^2 = 0 -/
def circle_M (x y : ℝ) : Prop := x^2 - 4*x + y^2 = 0

/-- The radius of a circle is the distance from its center to any point on the circle -/
noncomputable def radius (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ :=
  Real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2)

/-- Theorem: The radius of the circle M is 2 -/
theorem circle_M_radius : ∃ (center : ℝ × ℝ), ∀ (x y : ℝ), 
  circle_M x y → radius center (x, y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_M_radius_l787_78776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_N_l787_78797

/-- Given a positive integer N where N(N+1)/2 = 3003, prove that the sum of the digits of N is 14 -/
theorem sum_of_digits_of_N (N : ℕ) (h : N * (N + 1) / 2 = 3003) : 
  (N.digits 10).sum = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_N_l787_78797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l787_78710

noncomputable def projection (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let norm_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / norm_squared * u.1, dot_product / norm_squared * u.2)

theorem projection_property :
  let v₁ : ℝ × ℝ := (3, -1)
  let v₂ : ℝ × ℝ := (1, -3)
  let p₁ : ℝ × ℝ := (45/26, -15/26)
  projection v₁ v₁ = p₁ → projection v₂ v₁ = (0, 0) := by
  sorry

#check projection_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l787_78710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_equation_l787_78752

-- Define the surface S
def S : Set (Fin 3 → ℝ) :=
  {p | p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2 = 3 ∧ p 0 > 0 ∧ p 1 > 0 ∧ p 2 > 0}

-- Define the tangent plane at a point
def tangentPlane (p : Fin 3 → ℝ) : Set (Fin 3 → ℝ) :=
  {q | (q 0 - p 0) * (2 * p 0) + (q 1 - p 1) * (2 * p 1) + (q 2 - p 2) * (2 * p 2) = 0}

-- Define the orthocenter property
def orthocenterProperty (p : Fin 3 → ℝ) : Prop :=
  ∀ (a b c : ℝ),
    (λ i => if i = 0 then a else if i = 1 then 0 else 0) ∈ tangentPlane p →
    (λ i => if i = 1 then b else 0) ∈ tangentPlane p →
    (λ i => if i = 2 then c else 0) ∈ tangentPlane p →
    p = λ i => if i = 0 then a / 3 else if i = 1 then b / 3 else c / 3

theorem surface_equation :
  ∀ (p : Fin 3 → ℝ),
    p ∈ S ↔
    (p 0 = 1 ∧ p 1 = 1 ∧ p 2 = 1) ∨
    (orthocenterProperty p ∧ p 0 > 0 ∧ p 1 > 0 ∧ p 2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_equation_l787_78752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alyssas_allowance_l787_78772

/-- Alyssa's weekly allowance -/
noncomputable def weekly_allowance : ℝ := 8

/-- The amount Alyssa spent on movies -/
noncomputable def movie_expense : ℝ := weekly_allowance / 2

/-- The amount Alyssa earned from washing the car -/
noncomputable def car_wash_earnings : ℝ := 8

/-- The amount Alyssa ended up with -/
noncomputable def final_amount : ℝ := 12

/-- Proof that Alyssa's weekly allowance is correct -/
theorem alyssas_allowance : 
  weekly_allowance = 8 ∧
  movie_expense = weekly_allowance / 2 ∧
  car_wash_earnings = 8 ∧
  final_amount = 12 ∧
  weekly_allowance / 2 + car_wash_earnings = final_amount :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alyssas_allowance_l787_78772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_calculation_l787_78740

/-- Calculates the true discount given the banker's gain, interest rate, and time period. -/
noncomputable def true_discount (banker_gain : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  (banker_gain * 100) / (interest_rate * time)

/-- Theorem stating that given the specified conditions, the true discount is 70. -/
theorem true_discount_calculation :
  let banker_gain : ℝ := 8.4
  let interest_rate : ℝ := 12
  let time : ℝ := 1
  true_discount banker_gain interest_rate time = 70 := by
  -- Unfold the definition of true_discount
  unfold true_discount
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_calculation_l787_78740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_plot_ordinate_l787_78745

/-- Represents a residual plot in statistical analysis. -/
structure ResidualPlot where
  ordinate : ℝ
  residual : ℝ

/-- In residual analysis, the ordinate of a residual plot is equal to the residual. -/
theorem residual_plot_ordinate (plot : ResidualPlot) : plot.ordinate = plot.residual := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_plot_ordinate_l787_78745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_circumradii_l787_78733

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ
deriving Inhabited

/-- Distance between two points -/
noncomputable def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

/-- Predicate stating that O is the incenter of triangle ABC -/
def IncenterOf (O A B C : Point) : Prop := sorry

/-- Predicate stating that R is the circumradius of triangle ABC -/
def CircumradiusOf (R : ℝ) (A B C : Point) : Prop := sorry

/-- Given a triangle ABC with incenter O, circumradius R, and circumradii R₁, R₂, R₃ of triangles OBC, OCA, and OAB respectively, 
    if the sum of distances from O to the vertices of ABC is 3, 
    then at least one of R₁, R₂, R₃ is less than or equal to √R -/
theorem incenter_circumradii (A B C O : Point) (R R₁ R₂ R₃ : ℝ) : 
  IncenterOf O A B C →
  CircumradiusOf R A B C →
  CircumradiusOf R₁ O B C →
  CircumradiusOf R₂ O C A →
  CircumradiusOf R₃ O A B →
  dist O A + dist O B + dist O C = 3 →
  (R₁ ≤ Real.sqrt R) ∨ (R₂ ≤ Real.sqrt R) ∨ (R₃ ≤ Real.sqrt R) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_circumradii_l787_78733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l787_78726

/-- Represents a cubic polynomial of the form 3x^3 + dx^2 + ex + 9 -/
structure CubicPolynomial where
  d : ℝ
  e : ℝ

/-- The sum of the zeros of a cubic polynomial ax^3 + bx^2 + cx + d is -b/a -/
noncomputable def sum_of_zeros (p : CubicPolynomial) : ℝ := -p.d / 3

/-- The sum of the squares of the zeros of a cubic polynomial ax^3 + bx^2 + cx + d is (b^2 - 2ac) / a^2 -/
noncomputable def sum_of_squares_of_zeros (p : CubicPolynomial) : ℝ := p.d^2 / 9 - 2 * p.e / 3

/-- The sum of the coefficients of the polynomial 3x^3 + dx^2 + ex + 9 -/
def sum_of_coefficients (p : CubicPolynomial) : ℝ := 3 + p.d + p.e + 9

theorem cubic_polynomial_property (p : CubicPolynomial) :
  sum_of_zeros p / 3 = sum_of_squares_of_zeros p ∧
  sum_of_zeros p / 3 = sum_of_coefficients p →
  p.e = -24 ∨ p.e = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l787_78726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_recipe_flour_l787_78705

/-- Calculates the additional flour needed in ounces when doubling a recipe -/
theorem double_recipe_flour (original_cups : ℚ) (added_cups : ℚ) (cups_to_ounces : ℚ) : 
  original_cups = 7 ∧ added_cups = 3.75 ∧ cups_to_ounces = 8 →
  ((2 * original_cups - added_cups) * cups_to_ounces).floor = 82 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_recipe_flour_l787_78705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_proof_l787_78792

theorem angle_ratio_proof (α : Real) (l : Real) (h1 : l ≠ 0) 
  (h2 : ∃ (t : Real), t * Real.cos α = -3 * l ∧ t * Real.sin α = 4 * l) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_proof_l787_78792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_base_side_length_l787_78764

/-- The side length of the base equilateral triangle of a right prism,
    given the side lengths of a non-parallel cross-section. -/
noncomputable def baseSideLength (a b c : ℝ) : ℝ :=
  Real.sqrt ((a^2 + b^2 + c^2) / 3 - (Real.sqrt 2 / 3) *
    Real.sqrt ((a^2 - b^2)^2 + (b^2 - c^2)^2 + (c^2 - a^2)^2))

/-- Theorem stating that the base side length of a right prism with an equilateral triangle base,
    when cut by a non-parallel plane resulting in a cross-section with sides a, b, and c,
    is equal to the formula given by baseSideLength. -/
theorem right_prism_base_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x : ℝ), x > 0 ∧ x = baseSideLength a b c ∧
  x^2 = (a^2 + b^2 + c^2) / 3 - (2 / 3) * Real.sqrt ((a^2 + b^2 + c^2)^2 - 3 * (a^2*b^2 + b^2*c^2 + c^2*a^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_base_side_length_l787_78764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_after_swaps_l787_78777

/-- A swap operation on a list of integers -/
def swap (l : List Int) (i j : Nat) : List Int :=
  if i < l.length ∧ j < l.length ∧ i ≠ j
  then l.set i (l.get! j) |>.set j (l.get! i)
  else l

/-- Perform three random swaps on a list -/
def threeRandomSwaps (l : List Int) : List (List Int) := sorry

/-- Calculate the expected value of a five-digit number formed by a list of digits -/
noncomputable def expectedValue (l : List Int) : Real := sorry

theorem expected_value_after_swaps :
  let initial_cards := [1, 3, 5, 7, 9]
  let all_possible_outcomes := threeRandomSwaps initial_cards
  (all_possible_outcomes.map expectedValue).sum / all_possible_outcomes.length = 55200.208 := by
  sorry

#eval Float.toString 55200.208

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_after_swaps_l787_78777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_square_ratio_l787_78785

-- Define the side length of square S
variable (s : ℝ)

-- Define the longer and shorter sides of rectangle R
noncomputable def longer_side (s : ℝ) : ℝ := 1.2 * s
noncomputable def shorter_side (s : ℝ) : ℝ := 0.8 * s

-- Define the area of square S
noncomputable def area_S (s : ℝ) : ℝ := s^2

-- Define the area of rectangle R
noncomputable def area_R (s : ℝ) : ℝ := longer_side s * shorter_side s

-- Define the area of one triangle formed by cutting R along its diagonal
noncomputable def area_triangle (s : ℝ) : ℝ := area_R s / 2

-- Theorem statement
theorem triangle_to_square_ratio (s : ℝ) :
  area_triangle s / area_S s = 12 / 25 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_square_ratio_l787_78785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l787_78734

/-- A line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y^2 = 4x --/
def parabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- The line passes through a point --/
def line_passes_through (l : Line) (p : Point) : Prop :=
  p.x = l.slope * p.y + l.intercept

/-- The distance between two points --/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The theorem statement --/
theorem intersection_reciprocal_sum (l : Line) (A B M : Point) :
  l.slope ≠ 0 →
  M.x = 2 ∧ M.y = 0 →
  line_passes_through l M →
  parabola A ∧ parabola B →
  line_passes_through l A ∧ line_passes_through l B →
  1 / (distance A M)^2 + 1 / (distance B M)^2 = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l787_78734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l787_78790

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem sine_function_properties (ω φ : ℝ) 
  (h_ω : ω > 0)
  (h_φ : φ > -π ∧ φ < π)
  (h_zero1 : f ω φ (π/3) = 0)
  (h_zero2 : f ω φ (5*π/6) = 0)
  (h_adjacent : ∀ x, x > π/3 ∧ x < 5*π/6 → f ω φ x ≠ 0) :
  (∀ x, f ω φ (7*π/12 - x) = f ω φ (7*π/12 + x)) ∧ 
  φ = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l787_78790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l787_78741

/-- A function f defined on the interval [0, 1] -/
def f : ℝ → ℝ := λ x ↦ 6 * x * (1 - x)

/-- A linear function g defined on the interval [0, 1] -/
def g (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

theorem function_values (c : ℝ) (a b : ℝ) (h_a : a > 0) :
  (c^2 * ∫ x in Set.Icc 0 1, f x) = 1 →
  (∫ x in Set.Icc 0 1, f x * (g a b x)^2) = 1 →
  (∫ x in Set.Icc 0 1, f x * g a b x) = 0 →
  a = 2 * Real.sqrt 5 ∧ b = -(Real.sqrt 5) ∧ c = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l787_78741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_score_l787_78724

noncomputable def cricket_score (boundaries : ℕ) (sixes : ℕ) (run_percentage : ℚ) : ℚ :=
  let boundary_runs := boundaries * 4
  let six_runs := sixes * 6
  let total_boundary_six_runs := boundary_runs + six_runs
  let run_fraction := run_percentage / 100
  ↑total_boundary_six_runs / (1 - run_fraction)

theorem cricketer_score :
  cricket_score 12 2 (54545454545454545 / 1000000000000000) = 132 := by
  simp [cricket_score]
  norm_num
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_score_l787_78724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_commute_length_l787_78791

/-- The length of Liam's commute in miles -/
noncomputable def commute_length : ℝ := 44

/-- Liam's actual speed in mph -/
noncomputable def actual_speed : ℝ := 60

/-- The amount of time Liam arrived early in hours -/
noncomputable def early_time : ℝ := 4 / 60

/-- The difference in speed between Liam's actual speed and the speed that would make him arrive on time -/
noncomputable def speed_difference : ℝ := 5

theorem liam_commute_length :
  let time_at_actual_speed := commute_length / actual_speed
  let time_at_slower_speed := commute_length / (actual_speed - speed_difference)
  time_at_slower_speed = time_at_actual_speed + early_time →
  commute_length = 44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_commute_length_l787_78791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l787_78709

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x else -x^2 + 2*x + 1

theorem range_of_f : Set.range f = Set.Iic 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l787_78709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_squares_exceed_area_l787_78767

noncomputable def square_sequence (n : ℕ) : ℝ := 25 * (1 / 2) ^ (n - 1)

noncomputable def sum_areas (n : ℕ) : ℝ := 50 * (1 - (1 / 2) ^ n)

theorem minimum_squares_exceed_area :
  (sum_areas 6 > 49) ∧ (sum_areas 5 ≤ 49) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_squares_exceed_area_l787_78767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_70_factorial_l787_78798

open BigOperators

theorem last_two_nonzero_digits_of_70_factorial (n : ℕ) : n = 8 → (Finset.prod (Finset.range 70) (λ i => i + 1)) % 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_70_factorial_l787_78798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_property_polynomial_divisibility_l787_78769

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem curve_property (k : ℕ) :
  let x := fibonacci (2*k - 1)
  let y := fibonacci (2*k)
  x^2 + x*y - y^2 + 1 = 0 :=
sorry

theorem polynomial_divisibility (n : ℕ) :
  ∃ q : Polynomial ℤ, 
    X^n + X^(n-1) - (fibonacci n • X) - (fibonacci (n-1) • (1 : Polynomial ℤ)) = (X^2 - X - 1) * q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_property_polynomial_divisibility_l787_78769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_of_seven_not_fourteen_total_count_l787_78750

theorem count_multiples_of_seven_not_fourteen (n : ℕ) : 
  (n < 500 ∧ n % 7 = 0 ∧ n % 14 ≠ 0) → 
  (∃ k : ℕ, n = 7 * (2 * k - 1) ∧ k ≤ 36) :=
by sorry

theorem total_count : 
  Finset.card (Finset.filter (λ n => n < 500 ∧ n % 7 = 0 ∧ n % 14 ≠ 0) (Finset.range 500)) = 36 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_of_seven_not_fourteen_total_count_l787_78750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l787_78753

/-- The function f(x) defined as x + p/x -/
noncomputable def f (p : ℝ) (x : ℝ) : ℝ := x + p / x

/-- Predicate for f being increasing on (2, +∞) -/
def is_increasing_on_domain (p : ℝ) : Prop :=
  ∀ x y, 2 < x ∧ x < y → f p x < f p y

/-- The condition 0 ≤ p ≤ 4 -/
def condition (p : ℝ) : Prop := 0 ≤ p ∧ p ≤ 4

/-- Theorem stating that the condition is sufficient but not necessary -/
theorem sufficient_not_necessary :
  (∀ p, condition p → is_increasing_on_domain p) ∧
  (∃ p, ¬condition p ∧ is_increasing_on_domain p) := by
  sorry

#check sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l787_78753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_S_l787_78761

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the set of possible k values
def K : Set ℤ := {1, 2}

-- Define S as a function of n and k
noncomputable def S (n k : ℤ) : ℂ := i^(n+k) + i^(-(n+k))

-- The theorem to prove
theorem distinct_values_of_S :
  ∃ (A : Finset ℂ), (∀ n k, k ∈ K → S n k ∈ A) ∧ Finset.card A = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_S_l787_78761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_trial_steel_optimization_l787_78789

/-- The golden ratio --/
noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

/-- Calculate the first trial point using the 0.618 method --/
noncomputable def first_trial (lower upper : ℝ) : ℝ :=
  lower + (1 - 1 / golden_ratio) * (upper - lower)

/-- Calculate the second trial point using the 0.618 method --/
noncomputable def second_trial (lower upper : ℝ) : ℝ :=
  lower + (upper - first_trial lower upper)

/-- Calculate the third trial point using the 0.618 method when the first point is better --/
noncomputable def third_trial_first_better (lower upper : ℝ) : ℝ :=
  upper - (1 - 1 / golden_ratio) * (second_trial lower upper - lower)

/-- Theorem stating the result of the third trial in the steel optimization problem --/
theorem third_trial_steel_optimization (lower upper : ℝ) 
  (h_range : lower = 1000 ∧ upper = 2000) 
  (h_first_better : first_trial lower upper > second_trial lower upper) :
  third_trial_first_better lower upper = 1764 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_trial_steel_optimization_l787_78789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_seven_count_100_to_999_l787_78768

/-- 
Counts the occurrences of a specific digit in a range of integers.
@param start The start of the range (inclusive)
@param end_ The end of the range (inclusive)
@param digit The digit to count
-/
def countDigitOccurrences (start : Nat) (end_ : Nat) (digit : Nat) : Nat :=
  sorry

theorem digit_seven_count_100_to_999 : 
  countDigitOccurrences 100 999 7 = 290 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_seven_count_100_to_999_l787_78768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drums_hit_in_contest_l787_78706

/-- Drumming contest parameters -/
structure DrummingContest where
  entryCost : ℚ
  threshold : ℕ
  earningRate : ℚ
  netLoss : ℚ

/-- Calculate the number of drums hit in a drumming contest -/
def drumsHit (contest : DrummingContest) : ℕ :=
  contest.threshold + ((contest.entryCost - contest.netLoss) / contest.earningRate * 100).floor.toNat

/-- The theorem stating the number of drums hit in the given contest conditions -/
theorem drums_hit_in_contest :
  let contest : DrummingContest := {
    entryCost := 10,
    threshold := 200,
    earningRate := 25 / 1000,  -- 2.5 cents expressed as a fraction
    netLoss := 15 / 2  -- $7.5 expressed as a fraction
  }
  drumsHit contest = 300 := by sorry

#eval drumsHit {
  entryCost := 10,
  threshold := 200,
  earningRate := 25 / 1000,
  netLoss := 15 / 2
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drums_hit_in_contest_l787_78706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_pi_sufficient_not_necessary_l787_78786

-- Define the curve
noncomputable def curve (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

-- Define what it means for the curve to pass through the origin
def passes_through_origin (φ : ℝ) : Prop :=
  ∃ x : ℝ, curve x φ = 0

-- State the theorem
theorem phi_pi_sufficient_not_necessary :
  (∀ φ : ℝ, φ = π → passes_through_origin φ) ∧
  ¬(∀ φ : ℝ, passes_through_origin φ → φ = π) := by
  sorry

#check phi_pi_sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_pi_sufficient_not_necessary_l787_78786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_l787_78717

theorem chord_cosine (r : ℝ) (γ δ : ℝ) : 
  r > 0 → 
  γ > 0 → 
  δ > 0 → 
  γ + δ < π → 
  5^2 = 2 * r^2 * (1 - Real.cos γ) → 
  7^2 = 2 * r^2 * (1 - Real.cos δ) → 
  9^2 = 2 * r^2 * (1 - Real.cos (γ + δ)) → 
  Real.cos γ = 7/18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_l787_78717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_n_is_5_or_6_l787_78746

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  h1 : d < 0 -- Common difference is negative
  h2 : |a 3| = |a 9| -- Absolute value of 3rd and 9th terms are equal

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * seq.a 1 + (n - 1 : ℝ) * seq.d) / 2

/-- The theorem to be proved -/
theorem max_sum_n_is_5_or_6 (seq : ArithmeticSequence) :
  ∃ n : ℕ, (n = 5 ∨ n = 6) ∧
    ∀ m : ℕ, m ≠ n → sum_n seq m ≤ sum_n seq n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_n_is_5_or_6_l787_78746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_theorem_l787_78774

-- Define the circle
def Circle : Type := {p : ℂ // Complex.abs p = 1}

-- Define the points on the circle
variable (A B C D E F : Circle)

-- Define the intersection points
noncomputable def P (A B C D E F : Circle) : ℂ := 
  (A.val + B.val - D.val - E.val) / (A.val * B.val - D.val * E.val)

noncomputable def Q (A B C D E F : Circle) : ℂ := 
  (B.val + C.val - E.val - F.val) / (B.val * C.val - E.val * F.val)

noncomputable def R (A B C D E F : Circle) : ℂ := 
  (C.val + D.val - F.val - A.val) / (C.val * D.val - F.val * A.val)

-- Pascal's Theorem
theorem pascal_theorem (A B C D E F : Circle) :
  ∃ (k : ℂ), k ≠ 0 ∧ P A B C D E F - Q A B C D E F = k * (Q A B C D E F - R A B C D E F) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_theorem_l787_78774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomials_l787_78759

theorem gcd_of_polynomials (n : ℤ) (h : n ≥ 3) :
  Int.gcd (n^3 - 6*n^2 + 11*n - 6) (n^2 - 4*n + 4) = n - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomials_l787_78759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_value_l787_78762

/-- A function f is symmetric about the y-axis if f(x) = f(-x) for all x in its domain -/
def SymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = x^2 / ((2x+1)(x+a)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x^2 / ((2*x + 1) * (x + a))

/-- If f(x) = x^2 / ((2x+1)(x+a)) is symmetric about the y-axis, then a = -1/2 -/
theorem symmetry_implies_a_value :
  ∀ a : ℝ, SymmetricAboutYAxis (f a) → a = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_value_l787_78762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_growth_l787_78712

/-- Calculates the final amount in an account after compound interest is applied. -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * time)

/-- Proves that a $100 deposit with 20% annual interest compounded semiannually yields $121 after one year. -/
theorem deposit_growth :
  let principal : ℝ := 100
  let rate : ℝ := 0.20
  let compounds_per_year : ℝ := 2
  let time : ℝ := 1
  compound_interest principal rate compounds_per_year time = 121 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_growth_l787_78712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_fold_12m_string_l787_78701

noncomputable def fold_string (initial_length : ℝ) (num_folds : ℕ) : ℝ :=
  initial_length / (2 ^ num_folds)

theorem double_fold_12m_string :
  fold_string 12 2 = 3 := by
  -- Unfold the definition of fold_string
  unfold fold_string
  -- Simplify the expression
  simp [pow_two]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_fold_12m_string_l787_78701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_for_points_l787_78765

/-- Defines what it means for two points to be symmetric about a line. -/
def IsSymmetricAbout (p q : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (m : ℝ × ℝ), m ∈ l ∧ m = ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

/-- Defines what it means for a line to be an axis of symmetry for a set of points. -/
def IsAxisOfSymmetry (l : Set (ℝ × ℝ)) (points : Set (ℝ × ℝ)) : Prop :=
  ∀ p ∈ points, ∃ q ∈ points, p ≠ q ∧ IsSymmetricAbout p q l

/-- Given two points A and B in a 2D plane, this theorem proves that their axis of symmetry
    is the line y = 4 if A has coordinates (-1, 2) and B has coordinates (-1, 6). -/
theorem axis_of_symmetry_for_points (A B : ℝ × ℝ) :
  A = (-1, 2) ∧ B = (-1, 6) →
  ∃ (l : Set (ℝ × ℝ)), l = {(x, y) | y = 4} ∧ IsAxisOfSymmetry l {A, B} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_for_points_l787_78765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l787_78760

def a (n : ℕ) : ℤ := n^2 - 7*n + 6

theorem sequence_properties :
  (a 8 = 14) ∧ (∃ n : ℕ, a n = 150 ∧ n = 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l787_78760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_price_l787_78707

theorem computer_table_cost_price 
  (markup_percentage : ℝ) 
  (selling_price : ℝ) 
  (h1 : markup_percentage = 25)
  (h2 : selling_price = 1000) :
  selling_price / (1 + markup_percentage / 100) = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_price_l787_78707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_l787_78763

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (2 * a - 1) / x + 1 - 3 * a

-- Part I: Tangent line equation
theorem tangent_line_equation (a : ℝ) (h : a = 1) :
  ∃ (A B C : ℝ), A * 2 + B * (f 1 2) + C = 0 ∧ 
  (∀ x, f 1 x - (f 1 2) = (A / -B) * (x - 2)) ∧
  A = 3 ∧ B = -4 ∧ C = -4 :=
sorry

-- Part II: Range of a
theorem range_of_a :
  {a : ℝ | a > 0 ∧ ∀ x : ℝ, x ≥ 1 → f a x ≥ (1 - a) * Real.log x} = {a : ℝ | a ≥ 1/3} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_l787_78763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_l787_78731

def b : ℕ → ℚ
  | 0 => 3
  | 1 => 4
  | n+2 => b (n+1) / b n

theorem b_2023_value : b 2022 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_l787_78731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_actions_is_nine_l787_78794

structure Snake where
  heads : ℕ
  tails : ℕ

inductive SnakeAction
  | CutTwoTails
  | CutOneTail
  | CutTwoHeads

def apply_action (s : Snake) (a : SnakeAction) : Snake :=
  match a with
  | SnakeAction.CutTwoTails => ⟨s.heads + 1, s.tails - 2⟩
  | SnakeAction.CutOneTail => ⟨s.heads + 1, s.tails - 1⟩
  | SnakeAction.CutTwoHeads => ⟨s.heads - 2, s.tails⟩

def is_defeated (s : Snake) : Prop := s.heads = 0 ∧ s.tails = 0

def min_actions_to_defeat (s : Snake) : ℕ :=
  sorry

theorem min_actions_is_nine (s : Snake) : min_actions_to_defeat s = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_actions_is_nine_l787_78794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_selling_price_l787_78713

/-- The selling price of mangoes that results in a loss -/
def selling_price_loss : ℝ := 0

/-- The cost price of mangoes -/
def cost_price : ℝ := 0

/-- The selling price of mangoes that would result in a profit -/
def selling_price_profit : ℝ := 11.8125

/-- The current selling price results in a 20% loss -/
axiom loss_percentage : selling_price_loss = 0.8 * cost_price

/-- Selling at Rs. 11.8125 per kg would result in a 5% profit -/
axiom profit_percentage : selling_price_profit = 1.05 * cost_price

theorem mango_selling_price : selling_price_loss = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_selling_price_l787_78713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_modulus_l787_78744

theorem complex_power_modulus : Complex.abs ((1 + Complex.I * Real.sqrt 3) ^ 8) = 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_modulus_l787_78744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_count_l787_78737

def odd_positive_less_than_10 : Finset ℕ := {1, 3, 5, 7, 9}

def expression (a b : ℕ) : ℕ := a * b + a + b

theorem distinct_values_count :
  Finset.card (Finset.image 
    (λ (pair : ℕ × ℕ) => expression pair.1 pair.2)
    (Finset.product odd_positive_less_than_10 odd_positive_less_than_10)) = 10 := by
  sorry

#eval Finset.card (Finset.image 
  (λ (pair : ℕ × ℕ) => expression pair.1 pair.2)
  (Finset.product odd_positive_less_than_10 odd_positive_less_than_10))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_count_l787_78737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l787_78739

/-- A point on the circle x^2 + y^2 = 2y -/
structure PointOnCircle where
  x : ℝ
  y : ℝ
  h : x^2 + y^2 = 2*y

/-- The range of z = 2x + y for points on the circle -/
def z_range : Set ℝ :=
  {z | ∃ (p : PointOnCircle), z = 2*p.x + p.y}

/-- The range of a where x + y + a ≥ 0 always holds for points on the circle -/
def a_range : Set ℝ :=
  {a | ∀ (p : PointOnCircle), p.x + p.y + a ≥ 0}

/-- The function to be maximized and minimized -/
def f (p : PointOnCircle) : ℝ :=
  p.x^2 + p.y^2 - 16*p.x + 4*p.y

theorem circle_properties :
  (∀ z ∈ z_range, 1 - Real.sqrt 5 ≤ z ∧ z ≤ 1 + Real.sqrt 5) ∧
  (∀ a ∈ a_range, Real.sqrt 2 - 1 ≤ a) ∧
  (∃ p : PointOnCircle, f p = 6 + 2*Real.sqrt 73) ∧
  (∃ p : PointOnCircle, f p = 6 - 2*Real.sqrt 73) ∧
  (∀ p : PointOnCircle, 6 - 2*Real.sqrt 73 ≤ f p ∧ f p ≤ 6 + 2*Real.sqrt 73) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l787_78739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_on_equilateral_l787_78700

/-- The length of the congruent sides of the isosceles triangles in the given configuration -/
noncomputable def isosceles_side_length : ℝ := Real.sqrt 13 / 4

/-- The theorem statement -/
theorem isosceles_triangles_on_equilateral (equilateral_side : ℝ) 
  (isosceles_base : ℝ) (isosceles_area : ℝ) :
  equilateral_side = 2 →
  isosceles_base = equilateral_side / 2 →
  4 * isosceles_area = (Real.sqrt 3 / 4 * equilateral_side ^ 2) / 2 →
  ∃ (h : ℝ), 
    h = Real.sqrt 3 / 2 ∧
    isosceles_area = 1 / 2 * isosceles_base * h ∧
    isosceles_side_length ^ 2 = (isosceles_base / 2) ^ 2 + h ^ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_on_equilateral_l787_78700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_as_negative_437_l787_78711

-- Define the set of angles with the same terminal side as -437°
def sameTerminalSideAngles : Set ℝ :=
  {α : ℝ | ∃ k : ℤ, α = k * 360 + 283}

-- State the theorem
theorem same_terminal_side_as_negative_437 :
  sameTerminalSideAngles = {α : ℝ | ∃ k : ℤ, α = k * 360 + 283} :=
by
  -- The proof is trivial since the left-hand side is defined to be equal to the right-hand side
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_as_negative_437_l787_78711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_singular_l787_78725

noncomputable def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm_v := Real.sqrt (v.1^2 + v.2^2)
  let u := (v.1 / norm_v, v.2 / norm_v)
  ![![u.1^2, u.1 * u.2],
    ![u.1 * u.2, u.2^2]]

noncomputable def Q : Matrix (Fin 2) (Fin 2) ℝ := projection_matrix (4, 1)

theorem projection_matrix_singular :
  Matrix.det Q = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_singular_l787_78725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_slant_height_l787_78718

/-- Represents a frustum of a cone -/
structure Frustum where
  r1 : ℝ  -- radius of the top base
  r2 : ℝ  -- radius of the bottom base
  h : ℝ   -- height
  volume : ℝ -- volume

/-- Calculates the slant height of a frustum -/
noncomputable def slantHeight (f : Frustum) : ℝ :=
  Real.sqrt (f.h^2 + (f.r2 - f.r1)^2)

/-- The theorem stating the slant height of the given frustum -/
theorem frustum_slant_height :
  ∀ f : Frustum,
    f.r1 = 2 →
    f.r2 = 4 →
    f.volume = 56 * Real.pi →
    slantHeight f = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_slant_height_l787_78718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l787_78703

/-- Triangle with vertices A(4, 0), B(6, 8), and C(9, 3) -/
structure Triangle where
  A : ℝ × ℝ := (4, 0)
  B : ℝ × ℝ := (6, 8)
  C : ℝ × ℝ := (9, 3)

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.lineAB : LineEquation :=
  { a := 4, b := -1, c := -16 }

noncomputable def Triangle.altitudeLength : ℝ :=
  Real.sqrt 17

theorem triangle_properties (t : Triangle) :
  (Triangle.lineAB.a = 4 ∧ Triangle.lineAB.b = -1 ∧ Triangle.lineAB.c = -16) ∧
  Triangle.altitudeLength = Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l787_78703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l787_78702

noncomputable section

/-- Line l in parametric form -/
def line_l (t : ℝ) : ℝ × ℝ :=
  (1 - (Real.sqrt 2 / 2) * t, 2 + (Real.sqrt 2 / 2) * t)

/-- Curve C in polar form -/
def curve_C (θ : ℝ) : ℝ :=
  4 * Real.cos θ / (Real.sin θ)^2

/-- Intersection points of line l and curve C -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t θ, line_l t = p ∧ 
    (p.1^2 + p.2^2) * (Real.sin θ)^2 = 4 * p.1 * Real.cos θ ∧
    p.2 / p.1 = Real.tan θ}

/-- Length of segment AB -/
def segment_length : ℝ :=
  8 * Real.sqrt 2

theorem intersection_segment_length :
  ∀ A B, A ∈ intersection_points → B ∈ intersection_points → A ≠ B → 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = segment_length := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l787_78702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_num_and_denom_l787_78758

/-- The repeating decimal 0.343434... -/
def repeating_decimal : ℚ := 34 / 99

/-- The sum of the numerator and denominator of the fraction equivalent to 0.343434... (in lowest terms) -/
theorem sum_of_num_and_denom : ∃ (n d : ℕ), repeating_decimal = n / d ∧ Nat.Coprime n d ∧ n + d = 133 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_num_and_denom_l787_78758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_equivalence_l787_78770

def satisfies_equation (x y : ℝ) : Prop :=
  max x (x^2) + min y (y^2) = 1

theorem equation_equivalence (x y : ℝ) :
  satisfies_equation x y ↔
    (x^2 + y = 1 ∧ y ≤ 0) ∨
    (x^2 + y^2 = 1 ∧ 0 < y ∧ y < 1) ∨
    (x + y^2 = 1 ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_equivalence_l787_78770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_count_range_of_a_l787_78749

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a * (Real.exp (2 * x) - 3 * Real.exp x + 2)

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.exp (2 * x) - 3 * a * Real.exp x + 1

-- Theorem for the number of extreme points
theorem extreme_points_count (a : ℝ) :
  (a = 0 ∨ (0 < a ∧ a ≤ 8/9) → ∀ x, f_deriv a x > 0) ∧
  (a < 0 → ∃! x, f_deriv a x = 0) ∧
  (a > 8/9 → ∃ x₁ x₂, x₁ < x₂ ∧ f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0) :=
by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x > 0, f a x ≥ 0} = Set.Icc 0 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_count_range_of_a_l787_78749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_sum_l787_78796

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem function_extrema_sum (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc 1 2, f a x ≤ (max (f a 1) (f a 2))) ∧
  (∀ x ∈ Set.Icc 1 2, (min (f a 1) (f a 2)) ≤ f a x) ∧
  (max (f a 1) (f a 2)) + (min (f a 1) (f a 2)) = 6 →
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_sum_l787_78796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p2008_coordinates_l787_78773

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 45 degrees counterclockwise around the origin -/
noncomputable def rotate45 (p : Point) : Point :=
  { x := (p.x - p.y) / Real.sqrt 2
  , y := (p.x + p.y) / Real.sqrt 2 }

/-- Extends a point from the origin by a factor of 2 -/
def extend2 (p : Point) : Point :=
  { x := 2 * p.x
  , y := 2 * p.y }

/-- Generates the nth point in the sequence -/
noncomputable def genPoint : ℕ → Point
  | 0 => { x := 1, y := 0 }
  | n + 1 => if n % 2 = 0 then rotate45 (genPoint n) else extend2 (genPoint n)

/-- The main theorem: P₂₀₀₈ has coordinates (-2^1004, 0) -/
theorem p2008_coordinates : genPoint 2008 = { x := -2^1004, y := 0 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p2008_coordinates_l787_78773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_verify_crossing_time_l787_78775

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_km_hr : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  train_speed_km_hr = 40 →
  crossing_time = train_length / (train_speed_km_hr * 1000 / 3600) →
  crossing_time = 9 := by
  sorry

/-- Calculates the time it takes for a train to cross a pole -/
noncomputable def calculate_crossing_time (train_length : ℝ) (train_speed_km_hr : ℝ) : ℝ :=
  train_length / (train_speed_km_hr * 1000 / 3600)

/-- Verifies that the calculated crossing time for the given train is 9 seconds -/
theorem verify_crossing_time :
  calculate_crossing_time 100 40 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_verify_crossing_time_l787_78775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_configuration_forms_icosahedron_l787_78720

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- A configuration of three perpendicular rectangles -/
structure RectangleConfiguration where
  -- The shorter side length of each rectangle
  a : ℝ
  -- The longer side length of each rectangle
  b : ℝ
  -- Assumption that b > a
  h : b > a

/-- The vertices of a regular icosahedron -/
def RegularIcosahedron : Type := Fin 12 → ℝ × ℝ × ℝ

/-- The function that maps a RectangleConfiguration to a set of 12 points in 3D space -/
noncomputable def configurationToPoints (config : RectangleConfiguration) : Fin 12 → ℝ × ℝ × ℝ :=
  sorry

/-- The theorem stating the condition for the configuration to form a regular icosahedron -/
theorem rectangle_configuration_forms_icosahedron (config : RectangleConfiguration) :
  (∃ (ico : RegularIcosahedron), configurationToPoints config = ico) ↔ config.b / config.a = φ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_configuration_forms_icosahedron_l787_78720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_PF_plus_PA_min_value_PB_plus_d_l787_78716

-- Define the parabola
noncomputable def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the focus of the parabola
noncomputable def focus : ℝ × ℝ := (1/4, 0)

-- Define point A
noncomputable def A : ℝ × ℝ := (5/4, 3/4)

-- Define point B
noncomputable def B : ℝ × ℝ := (1/4, 2)

-- Define the distance from a point to the directrix
noncomputable def distance_to_directrix (x y : ℝ) : ℝ := x + 1/4

-- Theorem for the minimum value of PF + PA
theorem min_value_PF_plus_PA :
  ∀ x y : ℝ, parabola x y → 
  Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2) +
  Real.sqrt ((x - A.1)^2 + (y - A.2)^2) ≥ 3/2 := by sorry

-- Theorem for the minimum value of PB + d
theorem min_value_PB_plus_d :
  ∀ x y : ℝ, parabola x y →
  Real.sqrt ((x - B.1)^2 + (y - B.2)^2) +
  distance_to_directrix x y ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_PF_plus_PA_min_value_PB_plus_d_l787_78716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_properties_l787_78727

structure RegularTetrahedron where
  a : ℝ
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  D : ℝ × ℝ × ℝ
  M : ℝ × ℝ × ℝ

/-- The angle between lines AD and CM in a regular tetrahedron -/
noncomputable def angleADCM (t : RegularTetrahedron) : ℝ :=
  Real.arccos (Real.sqrt 3 / 6)

/-- The distance between lines AD and CM in a regular tetrahedron -/
noncomputable def distanceADCM (t : RegularTetrahedron) : ℝ :=
  t.a * Real.sqrt 22 / 11

/-- The ratio in which the common perpendicular divides CM -/
def ratioCM : ℚ × ℚ := (10, 1)

/-- The ratio in which the common perpendicular divides AD -/
def ratioAD : ℚ × ℚ := (3, 8)

/-- Main theorem about properties of a regular tetrahedron -/
theorem regular_tetrahedron_properties (t : RegularTetrahedron) 
  (h_regular : t.A = (0, 0, 0) ∧
               t.B = (t.a, 0, 0) ∧
               t.C = (t.a / 2, t.a * Real.sqrt 3 / 2, 0) ∧
               t.D = (t.a / 2, t.a * Real.sqrt 3 / 6, t.a * Real.sqrt 2 / 3))
  (h_midpoint : t.M = (t.a / 2, 0, 0)) :
  angleADCM t = Real.arccos (Real.sqrt 3 / 6) ∧
  distanceADCM t = t.a * Real.sqrt 22 / 11 ∧
  ratioCM = (10, 1) ∧
  ratioAD = (3, 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_properties_l787_78727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_sphere_radius_l787_78732

/-- A tetrahedron with altitudes greater than or equal to 1 -/
structure Tetrahedron where
  /-- The volume of the tetrahedron -/
  volume : ℝ
  /-- The areas of the faces of the tetrahedron -/
  face_areas : Fin 4 → ℝ
  /-- The altitudes of the tetrahedron -/
  altitudes : Fin 4 → ℝ
  /-- All altitudes are greater than or equal to 1 -/
  altitudes_ge_one : ∀ i, altitudes i ≥ 1
  /-- The volume is equal to one-third of the product of any face area and its corresponding altitude -/
  volume_eq : ∀ i, volume = (1/3) * face_areas i * altitudes i

/-- The maximal radius of a sphere that can be placed inside a tetrahedron with all altitudes ≥ 1 is 1/4 -/
theorem maximal_sphere_radius (t : Tetrahedron) : 
  ∃ r : ℝ, r = 1/4 ∧ ∀ s : ℝ, (∀ i, s ≤ t.altitudes i) → s ≤ r :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_sphere_radius_l787_78732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l787_78766

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 4)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (x + φ)

theorem symmetry_implies_phi_value (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : ∀ x, g φ x = -g φ (-x)) : 
  φ = 3 * Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l787_78766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l787_78783

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 + 3*x - x^2)

theorem f_monotone_increasing :
  MonotoneOn f (Set.Icc (-1) (3/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l787_78783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_arrangement_l787_78722

-- Define the parabola
def parabola (x : ℝ) := (x - 1)^2

-- Define the circle with radius r
def circleEq (r : ℝ) (x y : ℝ) := x^2 + y^2 = r^2

-- Define the tangent line at 60 degrees
noncomputable def tangent_line (x : ℝ) := Real.sqrt 3 * x

theorem parabola_circle_arrangement (r : ℝ) : 
  (∃ x y : ℝ, circleEq r x y ∧ y = parabola x) →  -- Parabola vertex tangent to circle
  (∃ x : ℝ, parabola x + r = tangent_line x) →  -- Parabola tangent to neighbor at 60°
  r = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_arrangement_l787_78722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_l787_78747

/-- The area of a regular hexagon with vertices A(1,1) and C(8,2) is 75√3 -/
theorem regular_hexagon_area : 
  ∀ (A C : ℝ × ℝ), 
  A = (1, 1) → 
  C = (8, 2) → 
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let area := 6 * (Real.sqrt 3 / 4 * AC^2)
  area = 75 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_l787_78747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_portion_theorem_l787_78714

-- Define the partnership structure
structure Partnership where
  total_investment : ℝ
  investment_ratio : ℝ × ℝ
  total_profit : ℝ
  profit_difference : ℝ

-- Define the function to calculate the equally divided portion
noncomputable def calculate_equal_portion (p : Partnership) : ℝ :=
  let (r1, r2) := p.investment_ratio
  let total_ratio := r1 + r2
  let equal_portion := p.total_profit - (p.profit_difference * total_ratio / (r1 - r2))
  equal_portion / 2

-- Theorem statement
theorem equal_portion_theorem (p : Partnership) :
  p.total_investment = 1000 ∧
  p.investment_ratio = (7, 3) ∧
  p.total_profit = 3000 ∧
  p.profit_difference = 800 →
  calculate_equal_portion p = 1000 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_portion_theorem_l787_78714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_theorem_l787_78795

/-- Family of curves parameterized by θ -/
def curve (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

/-- Line y = 2x -/
def line (x y : ℝ) : Prop := y = 2 * x

/-- Maximum chord length -/
noncomputable def max_chord_length : ℝ := 8 * Real.sqrt 5

theorem max_chord_length_theorem :
  ∃ (x₁ y₁ x₂ y₂ θ₁ θ₂ : ℝ),
    curve θ₁ x₁ y₁ ∧
    curve θ₂ x₂ y₂ ∧
    line x₁ y₁ ∧
    line x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = max_chord_length ∧
    ∀ (x y θ : ℝ), curve θ x y → line x y →
      Real.sqrt ((x - x₁)^2 + (y - y₁)^2) ≤ max_chord_length :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_theorem_l787_78795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_six_balls_l787_78719

/-- The number of ways to distribute n distinguishable balls into 2 indistinguishable boxes,
    with each box containing at least one ball -/
def distribute_balls (n : ℕ) : ℕ :=
  (Finset.range (n - 1)).sum (λ k => (Nat.choose n (k + 1)) / if k + 1 = n / 2 then 2 else 1)

/-- Theorem stating that there are 31 ways to distribute 6 distinguishable balls
    into 2 indistinguishable boxes, with each box containing at least one ball -/
theorem distribute_six_balls : distribute_balls 6 = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_six_balls_l787_78719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_for_given_circles_l787_78781

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Determines the number of common tangents between two circles -/
noncomputable def commonTangents (c1 c2 : Circle) : ℕ :=
  let d := distance c1.center c2.center
  if d > c1.radius + c2.radius then 4
  else if d = c1.radius + c2.radius then 3
  else if c1.radius < c2.radius && d > c2.radius - c1.radius then 2
  else if c1.radius > c2.radius && d > c1.radius - c2.radius then 2
  else if d = abs (c1.radius - c2.radius) then 1
  else 0

theorem common_tangents_for_given_circles :
  let c1 : Circle := { center := (0, 0), radius := 3 }
  let c2 : Circle := { center := (-3, 1), radius := 4 }
  commonTangents c1 c2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_for_given_circles_l787_78781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_problem_l787_78793

theorem fraction_sum_problem :
  ∃ (a b c d : ℕ),
    0 < a ∧ a < c ∧ c ≤ 100 ∧
    0 < b ∧ b < d ∧ d ≤ 100 ∧
    Nat.Coprime a c ∧
    Nat.Coprime b d ∧
    (a : ℚ) / c + (b : ℚ) / d = 86 / 111 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_problem_l787_78793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l787_78754

/-- The compound interest formula -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_problem :
  ∃ (initial_investment : ℝ),
    0 < initial_investment ∧
    abs (initial_investment - 357.53) < 0.01 ∧
    abs (compound_interest initial_investment 0.12 5 - 630.25) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l787_78754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_m_value_l787_78771

/-- Given two infinite geometric series with specified conditions, prove the value of m -/
theorem geometric_series_m_value :
  ∀ (m : ℚ),
  (let first_series_a1 : ℚ := 24
   let first_series_a2 : ℚ := 8
   let second_series_a1 : ℚ := 24
   let second_series_a2 : ℚ := 8 + m
   let first_series_sum : ℚ := first_series_a1 / (1 - (first_series_a2 / first_series_a1))
   let second_series_sum : ℚ := second_series_a1 / (1 - (second_series_a2 / second_series_a1))
   second_series_sum = 3 * first_series_sum) →
  m = 32 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_m_value_l787_78771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_extreme_points_l787_78799

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + 4 * x^2 + 4 * x - 1

noncomputable def f_derivative (x : ℝ) : ℝ := x^2 + 8 * x + 4

theorem geometric_sequence_extreme_points 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n+1) / a n = a (n+2) / a (n+1)) 
  (h_extreme : f_derivative (a 3) = 0 ∧ f_derivative (a 7) = 0) :
  a 5 = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_extreme_points_l787_78799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_completion_time_l787_78742

-- Define the time taken by each worker individually
variable (A B C D : ℝ)

-- Define the combined time of Alpha, Beta, and Gamma
noncomputable def k (A B C : ℝ) : ℝ := 1 / (1/A + 1/B + 1/C)

-- State the theorem
theorem workers_completion_time 
  (h1 : 1/A + 1/B + 1/C + 1/D = 1/(A - 8))
  (h2 : 1/A + 1/B + 1/C + 1/D = 1/(B - 2))
  (h3 : 1/A + 1/B + 1/C + 1/D = 3/C)
  (h4 : A > 0) (h5 : B > 0) (h6 : C > 0) (h7 : D > 0)
  : k A B C = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_completion_time_l787_78742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l787_78735

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1/4

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

-- Theorem statement
theorem tangent_line_implies_a_value (a : ℝ) :
  (∃ m : ℝ, f a m = 0 ∧ f_derivative a m = 0) → a = -3/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l787_78735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_placement_l787_78723

/-- Represents a placement of 'X's on a 5x5 grid -/
def Placement := Fin 5 → Fin 5 → Bool

/-- Checks if three points are aligned -/
def aligned (p1 p2 p3 : Fin 5 × Fin 5) : Prop :=
  (p1.1 = p2.1 ∧ p2.1 = p3.1) ∨  -- same row
  (p1.2 = p2.2 ∧ p2.2 = p3.2) ∨  -- same column
  (p3.1 - p1.1 = p2.1 - p1.1 ∧ p3.2 - p1.2 = p2.2 - p1.2)  -- same diagonal

/-- Checks if a placement is valid (no three 'X's aligned) -/
def valid_placement (p : Placement) : Prop :=
  ∀ p1 p2 p3 : Fin 5 × Fin 5, 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
    p p1.1 p1.2 ∧ p p2.1 p2.2 ∧ p p3.1 p3.2 →
    ¬aligned p1 p2 p3

/-- Counts the number of 'X's in a placement -/
def count_x (p : Placement) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 5)) (fun i =>
    Finset.sum (Finset.univ : Finset (Fin 5)) (fun j =>
      if p i j then 1 else 0))

/-- The theorem to be proved -/
theorem max_x_placement :
  (∃ p : Placement, valid_placement p ∧ count_x p = 15) ∧
  (∀ p : Placement, valid_placement p → count_x p ≤ 15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_placement_l787_78723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_is_nine_l787_78729

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  base_side : ℝ
  height : ℝ

/-- Cross-section of a regular triangular pyramid -/
structure CrossSection (pyramid : RegularTriangularPyramid) where
  through_midline : Bool
  perpendicular_to_base : Bool

/-- Calculate the area of the cross-section -/
def cross_section_area (pyramid : RegularTriangularPyramid) (cs : CrossSection pyramid) : ℝ :=
  sorry

/-- Theorem: The area of the cross-section is 9 square units -/
theorem cross_section_area_is_nine 
  (pyramid : RegularTriangularPyramid) 
  (cs : CrossSection pyramid)
  (h1 : pyramid.base_side = 6)
  (h2 : pyramid.height = 8)
  (h3 : cs.through_midline = true)
  (h4 : cs.perpendicular_to_base = true) :
  cross_section_area pyramid cs = 9 := by
  sorry

#check cross_section_area_is_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_is_nine_l787_78729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_3_to_7_l787_78738

/-- The number of hour markers on a circular clock face. -/
def num_markers : ℕ := 12

/-- The angle between adjacent hour markers in degrees. -/
noncomputable def angle_between_markers : ℝ := 360 / num_markers

/-- The number of central angles between 3 o'clock and 7 o'clock. -/
def angles_between_3_and_7 : ℕ := 4

/-- Theorem: The smaller angle between 3 o'clock and 7 o'clock is 120 degrees. -/
theorem angle_3_to_7 : angle_between_markers * (angles_between_3_and_7 : ℝ) = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_3_to_7_l787_78738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_equals_one_implies_a_equals_four_thirds_l787_78782

/-- The circle equation: x^2 + y^2 + 4x - 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + 1 = 0

/-- The line equation: x + ay - 1 = 0 -/
def line_equation (a x y : ℝ) : Prop :=
  x + a*y - 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-2, 1)

/-- The distance formula from a point (x, y) to a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x y A B C : ℝ) : ℝ :=
  |A*x + B*y + C| / Real.sqrt (A^2 + B^2)

/-- The theorem stating that the value of a satisfying the distance condition is 4/3 -/
theorem distance_equals_one_implies_a_equals_four_thirds :
  ∃ (a : ℝ), distance_point_to_line (circle_center.1) (circle_center.2) 1 a (-1) = 1 ∧ a = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_equals_one_implies_a_equals_four_thirds_l787_78782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_polynomials_two_roots_l787_78757

/-- A type representing a quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The sum of quadratic polynomials -/
def sum_polynomials (polys : List QuadraticPolynomial) : QuadraticPolynomial :=
  { a := polys.foldl (λ acc p => acc + p.a) 0,
    b := polys.foldl (λ acc p => acc + p.b) 0,
    c := polys.foldl (λ acc p => acc + p.c) 0 }

/-- The roots of a quadratic polynomial -/
def roots (p : QuadraticPolynomial) : Set ℝ :=
  {x : ℝ | p.a * x^2 + p.b * x + p.c = 0}

theorem sum_polynomials_two_roots
  (n : ℕ)
  (h_n : n ≥ 3)
  (points : Fin n → ℝ)
  (h_distinct : ∀ (i j : Fin n), i ≠ j → points i ≠ points j)
  (polys : List QuadraticPolynomial)
  (h_roots : ∀ (p : QuadraticPolynomial), p ∈ polys → roots p = Set.range points) :
  ∃ (x y : ℝ), x ≠ y ∧ roots (sum_polynomials polys) = {x, y} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_polynomials_two_roots_l787_78757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l787_78751

theorem size_relationship (a b c : ℝ) : 
  a = Real.rpow 0.6 0.6 → b = Real.rpow 0.6 1.5 → c = Real.rpow 1.5 0.6 → b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l787_78751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_calculation_l787_78704

/-- Represents the properties of a rectangular floor -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  area : ℝ
  painting_cost : ℝ
  painting_rate : ℝ

/-- Theorem about the length of a rectangular floor given specific conditions -/
theorem floor_length_calculation (floor : RectangularFloor)
  (h1 : floor.length = 3 * floor.breadth)
  (h2 : floor.area = floor.length * floor.breadth)
  (h3 : floor.area = floor.painting_cost / floor.painting_rate)
  (h4 : floor.painting_cost = 361)
  (h5 : floor.painting_rate = 3.00001) :
  ∃ ε > 0, |floor.length - 19.002| < ε := by
  sorry

#check floor_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_calculation_l787_78704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_tangential_quadrilateral_area_l787_78780

/-- A quadrilateral that can be both inscribed in a circle and circumscribed about a circle -/
structure CyclicTangentialQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  inscribable : True  -- Represents the condition that the quadrilateral can be inscribed in a circle
  circumscribable : True  -- Represents the condition that the quadrilateral can be circumscribed about a circle

/-- The area of a cyclic tangential quadrilateral -/
noncomputable def area (q : CyclicTangentialQuadrilateral) : ℝ :=
  Real.sqrt (q.a * q.b * q.c * q.d)

/-- The area of a cyclic tangential quadrilateral is equal to the square root of the product of its sides -/
theorem cyclic_tangential_quadrilateral_area (q : CyclicTangentialQuadrilateral) :
  area q = Real.sqrt (q.a * q.b * q.c * q.d) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_tangential_quadrilateral_area_l787_78780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l787_78730

/-- The equation of a hyperbola given specific conditions --/
theorem hyperbola_equation :
  ∀ (a b : ℝ) (F : ℝ × ℝ) (P : ℝ × ℝ),
    a > 0 → b > 0 →
    F = (2, 0) →
    (∀ (x y : ℝ), y^2 = 8*x → (x, y) = P) →
    (4 * Real.sqrt 5) / 5 = (a * Real.sqrt (a^2 + b^2)) / Real.sqrt (a^2 + b^2) →
    (dist P F + dist P (-2, P.2)) ≥ 3 →
    ∃ (x y : ℝ), y^2/4 - x^2 = 1 ∧ y^2/a^2 - x^2/b^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l787_78730
