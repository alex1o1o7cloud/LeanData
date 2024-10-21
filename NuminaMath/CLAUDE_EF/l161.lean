import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_info_for_average_age_l161_16164

theorem insufficient_info_for_average_age 
  (total_members : Nat) 
  (members_above_certain_age : Nat) 
  (h1 : total_members = 22) 
  (h2 : members_above_certain_age = 21) :
  ¬ ∃ (average_age : ℝ), ∀ (ages : Fin total_members → ℝ), 
    (∃ (threshold : ℝ), (Finset.filter (fun i => ages i > threshold) (Finset.univ : Finset (Fin total_members))).card = members_above_certain_age) →
    (Finset.sum Finset.univ ages) / total_members = average_age :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_info_for_average_age_l161_16164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_i_l161_16139

open Complex

/-- The complex number i -/
noncomputable def i : ℂ := Complex.I

/-- The function f(x) -/
noncomputable def f (x : ℂ) : ℂ := (cos x + x^2) / (x + 1)

/-- Theorem stating the value of f(i) -/
theorem f_of_i : f i = (Real.exp 1 + Real.exp (-1) - 2) / (2 * (i + 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_i_l161_16139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l161_16105

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with vertex and focus -/
structure Parabola where
  vertex : Point
  focus : Point

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is in the first quadrant -/
def isFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Check if a point lies on a parabola -/
def isOnParabola (p : Point) (para : Parabola) : Prop :=
  distance p para.focus = p.y - para.vertex.y + distance para.vertex para.focus

theorem parabola_point_coordinates :
  ∀ (para : Parabola) (p : Point),
    para.vertex = ⟨0, 0⟩ →
    para.focus = ⟨0, 2⟩ →
    isFirstQuadrant p →
    isOnParabola p para →
    distance p para.focus = 150 →
    p = ⟨2 * Real.sqrt 296, 148⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l161_16105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_b_values_l161_16128

noncomputable def f (n : ℝ) : ℝ :=
  if n < 0 then n^2 - 4 else 2*n - 24

theorem difference_of_b_values : ∃ b₁ b₂ : ℝ,
  f (-3) + f 3 + f b₁ = 2 ∧
  f (-3) + f 3 + f b₂ = 2 ∧
  b₁ ≠ b₂ ∧
  (max b₁ b₂ - min b₁ b₂ = 39/2 + Real.sqrt 19) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_b_values_l161_16128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_ellipse_is_parametric_curve_l161_16179

-- Define the original ellipse
def originalEllipse (x y : ℝ) : Prop :=
  x^2/16 + y^2/4 = 1

-- Define the transformation function
noncomputable def transform (x y : ℝ) : ℝ × ℝ :=
  (x/2, y)

-- Define the parametric equations of the new curve
noncomputable def newCurve (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, 2 * Real.sin θ)

-- Theorem statement
theorem transformed_ellipse_is_parametric_curve :
  ∀ x y : ℝ, originalEllipse x y →
  ∃ θ : ℝ, transform x y = newCurve θ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_ellipse_is_parametric_curve_l161_16179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l161_16156

/-- The function g(x) defined on the interval [0, 10] -/
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (100 - x)) + Real.sqrt (x * (10 - x))

/-- The domain of the function -/
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 10 }

theorem g_max_value :
  ∃ (M : ℝ) (x₁ : ℝ), x₁ ∈ domain ∧ 
    (∀ x ∈ domain, g x ≤ M) ∧
    g x₁ = M ∧
    M = 30 ∧ x₁ = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l161_16156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_angle_l161_16184

/-- Given two intersecting circles with equal radii, if the shaded area
    (region inside one circle but outside the triangle formed by the
    intersection points and the center) is 1/3 of the total circle area,
    then the angle formed by the intersection points and the center of
    one circle is approximately 2.6053 radians. -/
theorem intersecting_circles_angle (r : ℝ) (α : ℝ) (h : α > 0) :
  (α * r^2 / 2 - r^2 * Real.sin α / 2) = π * r^2 / 3 →
  ‖α - 2.6053‖ < 0.0001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_angle_l161_16184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l161_16133

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (1/2) * x + Real.log x

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a + 1/2 + 1/x

-- State the theorem
theorem tangent_line_parallel (a : ℝ) : 
  (f_deriv a 1 = 7/2) → a = 2 := by
  intro h
  -- The proof steps would go here
  sorry

#check tangent_line_parallel

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l161_16133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_problem_l161_16158

noncomputable section

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the cross product operation
variable (cross : V → V → V)

-- Define the transformation T
variable (T : V → V)

-- Define specific vectors
def v1 : V := sorry
def v2 : V := sorry
def v3 : V := sorry
def result : V := sorry

-- State the theorem
theorem transformation_problem 
  (h1 : ∀ (a b : ℝ) (v w : V), T (a • v + b • w) = a • T v + b • T w)
  (h2 : ∀ (v w : V), T (cross v w) = cross (T v) (T w))
  (h3 : T v1 = sorry)
  (h4 : T v2 = sorry)
  (h5 : T v3 = sorry) :
  T (sorry : V) = result := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_problem_l161_16158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_theorem_l161_16167

-- Define the number of minutes run per day for each grade
noncomputable def sixth_grade_minutes : ℚ := 18
noncomputable def seventh_grade_minutes : ℚ := 16
noncomputable def eighth_grade_minutes : ℚ := 12

-- Define the ratio of students in each grade compared to eighth grade
noncomputable def sixth_grade_ratio : ℚ := 3
noncomputable def seventh_grade_ratio : ℚ := 3
noncomputable def eighth_grade_ratio : ℚ := 1

-- Define the function to calculate the average minutes run
noncomputable def average_minutes_run (e : ℚ) : ℚ :=
  let total_minutes := sixth_grade_minutes * sixth_grade_ratio * e +
                       seventh_grade_minutes * seventh_grade_ratio * e +
                       eighth_grade_minutes * eighth_grade_ratio * e
  let total_students := sixth_grade_ratio * e + seventh_grade_ratio * e + eighth_grade_ratio * e
  total_minutes / total_students

-- Theorem statement
theorem average_minutes_theorem :
  ∀ e : ℚ, e > 0 → average_minutes_run e = 114 / 7 :=
by
  intro e he
  unfold average_minutes_run
  simp [sixth_grade_minutes, seventh_grade_minutes, eighth_grade_minutes,
        sixth_grade_ratio, seventh_grade_ratio, eighth_grade_ratio]
  -- The actual proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_theorem_l161_16167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_theorem_l161_16141

def vector_problem (a b : ℝ × ℝ) : Prop :=
  let magnitude (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
  let dot_product (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2
  let angle (v w : ℝ × ℝ) := Real.arccos ((dot_product v w) / (magnitude v * magnitude w))
  magnitude a = 4 ∧ 
  magnitude b = 3 ∧ 
  dot_product a b = 6 →
  angle a b = Real.pi / 3 ∧
  magnitude (3 • a - 4 • b) = 12

theorem vector_theorem : ∀ a b : ℝ × ℝ, vector_problem a b := by
  sorry

#check vector_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_theorem_l161_16141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_l161_16196

/-- Given n points on a circle (n ≥ 6), with every two points connected by a line segment,
    and any three line segments within the circle not intersecting at a single point,
    the number of triangles formed is ₃C + 4 * ₄C + 5 * ₅C + ₆C --/
theorem triangle_count (n : ℕ) (h : n ≥ 6) : 
  (Nat.choose n 3) + 4 * (Nat.choose n 4) + 5 * (Nat.choose n 5) + (Nat.choose n 6) = 
  (Nat.choose n 3) + 4 * (Nat.choose n 4) + 5 * (Nat.choose n 5) + (Nat.choose n 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_l161_16196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wheels_after_changes_l161_16122

/-- Represents the number of wheels for each vehicle type -/
structure VehicleWheels where
  bicycle : Nat
  car : Nat
  motorcycle : Nat
  tricycle : Nat
  quad : Nat

/-- Represents the count of vehicles -/
structure VehicleCount where
  bicycles : Nat
  cars : Nat
  motorcycles : Nat
  tricycles : Nat
  quads : Nat

/-- Represents the changes in vehicle counts -/
structure VehicleChanges where
  bicycles : Int
  cars : Int
  motorcycles : Int

/-- Represents the number of vehicles with wheel issues -/
structure WheelIssues where
  bicycles : Nat
  cars : Nat
  motorcycles : Nat

def calculateTotalWheels (
  initialCount : VehicleCount)
  (changes : VehicleChanges)
  (issues : WheelIssues)
  (normalWheels : VehicleWheels) : Nat :=
  sorry

theorem total_wheels_after_changes (
  initialCount : VehicleCount)
  (changes : VehicleChanges)
  (issues : WheelIssues)
  (normalWheels : VehicleWheels)
  (h1 : initialCount.bicycles = 20)
  (h2 : initialCount.cars = 10)
  (h3 : initialCount.motorcycles = 5)
  (h4 : initialCount.tricycles = 3)
  (h5 : initialCount.quads = 2)
  (h6 : changes.bicycles = -7)
  (h7 : changes.cars = 4)
  (h8 : changes.motorcycles = 1)
  (h9 : issues.bicycles = 5)
  (h10 : issues.cars = 2)
  (h11 : issues.motorcycles = 1)
  (h12 : normalWheels.bicycle = 2)
  (h13 : normalWheels.car = 4)
  (h14 : normalWheels.motorcycle = 2)
  (h15 : normalWheels.tricycle = 3)
  (h16 : normalWheels.quad = 4) :
  calculateTotalWheels initialCount changes issues normalWheels = 102 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wheels_after_changes_l161_16122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l161_16163

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the set M
def M (a b c : ℝ) : Set ℝ := {x | f a b c x > 0}

-- Part 1
theorem part1 (a b c : ℝ) (ha : a ≠ 0) :
  M a b c = {x | -1/2 < x ∧ x < 2} →
  f a b c 0 = 2 →
  ∀ x, f a b c x = -2 * x^2 + 3 * x + 2 :=
sorry

-- Part 2
theorem part2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ m, m < 0 ∧ M a b c = Set.Ioi m ∪ Set.Iic m) →
  (⨅ m, (b / c - 2 * m)) = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l161_16163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_parameters_determination_l161_16175

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem function_parameters_determination (ω φ : ℝ) 
  (h1 : ω > 0)
  (h2 : abs φ < π)
  (h3 : f ω φ (5 * π / 8) = 2)
  (h4 : f ω φ (11 * π / 8) = 0)
  (h5 : ∀ T > 0, (∀ x, f ω φ (x + T) = f ω φ x) → T ≥ 3 * π) :
  ω = 2 / 3 ∧ φ = π / 12 := by
  sorry

#check function_parameters_determination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_parameters_determination_l161_16175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_george_movie_cost_l161_16171

/-- Calculates the total cost of a movie theater visit given specific prices and discounts -/
def movieTheaterCost (ticketPrice : ℚ) (nachosFraction : ℚ) (popcornDiscount : ℚ) 
  (sodaFraction : ℚ) (foodDiscount : ℚ) (salesTax : ℚ) : ℚ :=
  let nachosPrice := ticketPrice * nachosFraction
  let popcornPrice := nachosPrice * (1 - popcornDiscount)
  let sodaPrice := popcornPrice * sodaFraction
  let totalFoodBeforeDiscount := nachosPrice + popcornPrice + sodaPrice
  let totalFoodAfterDiscount := totalFoodBeforeDiscount * (1 - foodDiscount)
  let totalBeforeTax := ticketPrice + totalFoodAfterDiscount
  let totalWithTax := totalBeforeTax * (1 + salesTax)
  (totalWithTax * 100).floor / 100  -- Rounding to nearest cent

/-- Theorem stating that the total cost of George's movie theater visit is $34.28 -/
theorem george_movie_cost : 
  movieTheaterCost 16 (1/2) (1/4) (3/4) (1/10) (1/20) = 3428/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_george_movie_cost_l161_16171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sine_l161_16119

theorem geometric_sequence_sine (α β γ : ℝ) : 
  α ∈ Set.Icc 0 (2 * Real.pi) →
  β = 2 * α →
  γ = 4 * α →
  (∃ r : ℝ, Real.sin β = r * Real.sin α ∧ Real.sin γ = r * Real.sin β) →
  ((α = 2 * Real.pi / 3 ∧ β = 4 * Real.pi / 3 ∧ γ = 8 * Real.pi / 3) ∨
   (α = 4 * Real.pi / 3 ∧ β = 8 * Real.pi / 3 ∧ γ = 16 * Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sine_l161_16119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_growth_l161_16157

/-- Represents the remaining funds after dividends at the end of the nth year in thousands of dollars -/
noncomputable def a : ℕ → ℝ
  | 0 => 200  -- Initial capital
  | n + 1 => (3/2) * a n - 60  -- Recursive relationship

/-- The number of years it takes for the remaining funds to reach or exceed 1200 thousand dollars -/
def years_to_reach_1200 : ℕ := 7

theorem company_growth :
  (∀ n : ℕ, a (n + 1) = (3/2) * a n - 60) ∧
  a 1 = 240 ∧
  a 2 = 300 ∧
  years_to_reach_1200 = 7 ∧
  a years_to_reach_1200 ≥ 1200 ∧
  ∀ m : ℕ, m < years_to_reach_1200 → a m < 1200 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_growth_l161_16157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_cosine_extrema_l161_16152

/-- The minimum distance between adjacent extrema of a cosine function -/
theorem min_distance_cosine_extrema (a θ : ℝ) (ha : a > 0) :
  let f := λ x => a * Real.cos (a * x + θ)
  let d := λ a => Real.sqrt ((2 * a) ^ 2 + (Real.pi / a) ^ 2)
  (∀ a > 0, d a ≥ 2 * Real.sqrt Real.pi) ∧
  d (Real.sqrt (2 * Real.pi) / 2) = 2 * Real.sqrt Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_cosine_extrema_l161_16152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_postcard_collection_average_l161_16180

/-- Arithmetic sequence with first term 12, common difference 10, and 7 terms -/
def postcard_sequence : Fin 7 → ℕ := fun n => 12 + 10 * n.val

/-- The average of the postcard sequence -/
def postcard_average : ℚ := (postcard_sequence 3 : ℚ)

theorem postcard_collection_average :
  postcard_average = 42 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_postcard_collection_average_l161_16180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_functional_equation_l161_16109

theorem polynomial_functional_equation (k : ℝ) (hk : k > 1) :
  ∀ P : Polynomial ℝ, (∀ x : ℝ, P.eval (x^k) = (P.eval x)^k) →
  (P = 0) ∨ (P = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_functional_equation_l161_16109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_for_inequality_l161_16182

theorem smallest_c_for_inequality :
  ∃ c : ℝ, c > 0 ∧
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → (x * y) ^ (1/3) + c * |x - y| ≥ (x + y) / 2) ∧
  (∀ c' : ℝ, c' > 0 →
    (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → (x * y) ^ (1/3) + c' * |x - y| ≥ (x + y) / 2) →
    c' ≥ c) ∧
  c = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_for_inequality_l161_16182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_shift_sin_shift_equiv_cos_sin_equiv_shift_l161_16108

theorem cos_sin_shift (x : ℝ) : 
  Real.cos (2 * x + π / 3) = Real.sin (2 * x + 5 * π / 6) :=
by sorry

theorem sin_shift_equiv (x : ℝ) :
  Real.sin (2 * x + 5 * π / 6) = Real.sin (2 * (x + 5 * π / 12)) :=
by sorry

theorem cos_sin_equiv_shift (x : ℝ) :
  Real.cos (2 * x + π / 3) = Real.sin (2 * (x + 5 * π / 12)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_shift_sin_shift_equiv_cos_sin_equiv_shift_l161_16108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_numbers_necessary_and_sufficient_l161_16147

/-- Represents an infinite chessboard filling the upper half-plane -/
structure Chessboard where
  /-- The function mapping a position (row, column) to its value -/
  value : ℤ → ℤ → ℚ

/-- The property that for each black cell, the sum of the numbers in the two adjacent cells
    to the right and left equals the sum of the other two numbers in the adjacent cells above and below -/
def valid_chessboard (c : Chessboard) : Prop :=
  ∀ (row col : ℤ), 
    c.value (row + 1) col + c.value (row - 1) col = c.value row (col - 1) + c.value row (col + 1)

/-- The theorem stating that 4 additional numbers are necessary and sufficient -/
theorem four_numbers_necessary_and_sufficient (c : Chessboard) (n : ℤ) (col : ℤ) :
  valid_chessboard c →
  ∃! (a b d e : ℚ),
    (∀ (x y z w : ℚ),
      (c.value 0 (col - 1) = x ∧
       c.value 0 (col + 1) = y ∧
       c.value 1 (col - 1) = z ∧
       c.value 1 (col + 1) = w) →
      c.value (n + 2) col = c.value n col + (z - x) + (w - y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_numbers_necessary_and_sufficient_l161_16147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l161_16169

/-- Represents a cone with given base radius and slant height -/
structure Cone where
  base_radius : ℝ
  slant_height : ℝ

/-- Calculates the central angle of the unfolded side of a cone -/
noncomputable def central_angle (c : Cone) : ℝ :=
  (2 * c.base_radius * 360) / c.slant_height

/-- Theorem stating that for a cone with base radius 4 and slant height 9,
    the central angle of the unfolded side is 160° -/
theorem cone_central_angle :
  let c : Cone := { base_radius := 4, slant_height := 9 }
  central_angle c = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l161_16169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_other_class_time_approx_l161_16187

/-- Represents the duration of a school day in hours -/
noncomputable def school_day_hours : ℚ := 9

/-- Represents the total number of classes -/
def total_classes : ℕ := 10

/-- Represents the combined duration of history and chemistry classes in hours -/
noncomputable def history_chemistry_hours : ℚ := 2

/-- Represents the duration of the mathematics class in hours -/
noncomputable def math_class_hours : ℚ := 3/2

/-- Calculates the average time spent in other classes in minutes -/
noncomputable def average_other_class_time : ℚ :=
  let total_minutes := school_day_hours * 60
  let history_chemistry_minutes := history_chemistry_hours * 60
  let math_minutes := math_class_hours * 60
  let remaining_minutes := total_minutes - (history_chemistry_minutes + math_minutes)
  let other_classes := total_classes - 3
  remaining_minutes / other_classes

/-- Theorem stating that the average time spent in other classes is approximately 47.14 minutes -/
theorem average_other_class_time_approx :
  ∃ ε > 0, |average_other_class_time - 47.14| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_other_class_time_approx_l161_16187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residuals_assess_effectiveness_l161_16160

/-- Residual in a statistical model -/
structure Residual where
  value : ℝ
  model_fit : ℝ → ℝ

/-- A measure of model effectiveness -/
def model_effectiveness : ℝ → ℝ := sorry

/-- Theorem stating that residuals can be used to assess model effectiveness -/
theorem residuals_assess_effectiveness (r : Residual) :
  ∃ (f : Residual → ℝ), f r = model_effectiveness (r.model_fit r.value) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_residuals_assess_effectiveness_l161_16160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_power_equals_eightyone_l161_16198

theorem nine_power_equals_eightyone (x : ℝ) : (9 : ℝ)^x = 81 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_power_equals_eightyone_l161_16198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l161_16115

noncomputable def f (x : ℝ) : ℝ := Real.rpow x (3⁻¹)

def g (x : ℝ) : ℝ := x

theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l161_16115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clerical_staff_percentage_specific_l161_16148

/-- Calculates the percentage of clerical staff after reduction -/
def clerical_staff_percentage (total_employees : ℕ) 
  (initial_clerical_fraction : ℚ) (reduction_fraction : ℚ) : ℚ :=
  let initial_clerical := (initial_clerical_fraction * total_employees : ℚ)
  let reduced_clerical := initial_clerical * (1 - reduction_fraction)
  let remaining_employees := (total_employees : ℚ) - (initial_clerical * reduction_fraction)
  (reduced_clerical / remaining_employees) * 100

/-- Proves that the specific case results in approximately 11.76% -/
theorem clerical_staff_percentage_specific :
  ∃ ε > 0, |clerical_staff_percentage 3600 (1/6) (1/3) - 11.76| < ε := by
  -- The proof is omitted for brevity
  sorry

#eval clerical_staff_percentage 3600 (1/6) (1/3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clerical_staff_percentage_specific_l161_16148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_in_range_g_two_zeros_iff_a_in_range_l161_16103

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a^2 * Real.log x - a * x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a^2 * Real.log x

-- Statement for part 1
theorem f_nonnegative_iff_a_in_range (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ a ∈ Set.Icc (-2 * Real.exp (3/4)) 1 := by
  sorry

-- Statement for part 2
theorem g_two_zeros_iff_a_in_range (a : ℝ) :
  (∃! x y, x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) ∧
           y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) ∧
           x ≠ y ∧ g a x = 0 ∧ g a y = 0) ↔
  a ∈ Set.Icc (-Real.exp 1) (-Real.sqrt (2 * Real.exp 1)) ∪
      Set.Ioo (Real.sqrt (2 * Real.exp 1)) (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_in_range_g_two_zeros_iff_a_in_range_l161_16103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l161_16125

-- Define the circle C
noncomputable def C (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 10*y + 21 = 0

-- Define the midpoint M of OP
noncomputable def M (xp yp : ℝ) : ℝ × ℝ := (xp/2, yp/2)

-- Theorem statement
theorem midpoint_trajectory :
  ∀ (x y : ℝ), 
    (∃ (xp yp : ℝ), C xp yp ∧ M xp yp = (x, y)) →
    (x - 2)^2 + (y - 5/2)^2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l161_16125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_in_range_l161_16100

-- Define the function f(x)
noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 - (a + 2) * x + 2 * a * Real.log x + 1

-- Define what it means for f to have an extremum point in an interval
def has_extremum_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ (∀ y, a < y ∧ y < b → f y ≤ f x) ∨ (∀ y, a < y ∧ y < b → f y ≥ f x)

-- State the theorem
theorem extremum_implies_a_in_range (a : ℝ) :
  has_extremum_in (f a) 4 6 → 4 < a ∧ a < 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_in_range_l161_16100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_condition_l161_16190

/-- A function f(x) = x³ - ax² - ax - 1 has both a maximum and a minimum value if and only if a < -3 or a > 0 -/
theorem function_extrema_condition (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, x^3 - a*x^2 - a*x - 1 ≤ max ∧ min ≤ x^3 - a*x^2 - a*x - 1) ↔ 
  (a < -3 ∨ a > 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_condition_l161_16190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_outfit_cost_l161_16154

/-- The cost of John's outfit given the cost of his pants and the percentage increase for his shirt -/
noncomputable def outfit_cost (pants_cost : ℚ) (shirt_percentage_increase : ℚ) : ℚ :=
  pants_cost + pants_cost * (1 + shirt_percentage_increase / 100)

/-- Theorem stating the total cost of John's outfit -/
theorem johns_outfit_cost :
  outfit_cost 50 60 = 130 := by
  -- Unfold the definition of outfit_cost
  unfold outfit_cost
  -- Simplify the arithmetic
  simp [Rat.add_def, Rat.mul_def]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_outfit_cost_l161_16154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_condition_l161_16194

theorem log_inequality_condition (a b : ℝ) :
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log a > Real.log b → a > b) ∧
  ¬(∀ a b : ℝ, a > b → Real.log a > Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_condition_l161_16194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l161_16192

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

-- Define the distance from a point to a focus
noncomputable def distance_to_focus (x y : ℝ) (fx fy : ℝ) : ℝ := 
  Real.sqrt ((x - fx)^2 + (y - fy)^2)

-- Theorem statement
theorem ellipse_focus_distance 
  (x y : ℝ) 
  (h_on_ellipse : is_on_ellipse x y) 
  (f1x f1y f2x f2y : ℝ) 
  (h_focus1 : distance_to_focus x y f1x f1y = 5) :
  distance_to_focus x y f2x f2y = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l161_16192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_phi_divisibility_l161_16153

theorem euler_phi_divisibility (n : ℕ) : 
  (2 ^ (n * (n + 1))) ∣ (32 * Nat.totient (2 ^ (2 ^ n) - 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_phi_divisibility_l161_16153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rect_equivalence_max_x_plus_y_min_x_plus_y_max_min_achievable_l161_16102

-- Define the polar coordinate equation of the circle
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 - 4*ρ*(Real.cos θ) + 2 = 0

-- Define the rectangular coordinate equation of the circle
def rect_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 2

-- Theorem stating the equivalence of polar and rectangular equations
theorem polar_to_rect_equivalence :
  ∀ (x y ρ θ : ℝ), 
    x = ρ * (Real.cos θ) ∧ 
    y = ρ * (Real.sin θ) ∧ 
    polar_equation ρ θ → 
    rect_equation x y :=
by sorry

-- Theorem for the maximum value of x + y
theorem max_x_plus_y :
  ∀ (x y : ℝ), 
    rect_equation x y → 
    x + y ≤ 4 :=
by sorry

-- Theorem for the minimum value of x + y
theorem min_x_plus_y :
  ∀ (x y : ℝ), 
    rect_equation x y → 
    x + y ≥ 0 :=
by sorry

-- Theorem stating that the maximum and minimum are achievable
theorem max_min_achievable :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    rect_equation x₁ y₁ ∧
    rect_equation x₂ y₂ ∧
    x₁ + y₁ = 4 ∧
    x₂ + y₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rect_equivalence_max_x_plus_y_min_x_plus_y_max_min_achievable_l161_16102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l161_16150

noncomputable def complex_number : ℂ := (3 - 5*Complex.I) / (1 - Complex.I)

theorem complex_number_in_fourth_quadrant :
  let z := complex_number
  (z.re > 0) ∧ (z.im < 0) :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l161_16150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_sum_l161_16193

/-- Parametric representation of an ellipse --/
noncomputable def ellipse (t : ℝ) : ℝ × ℝ :=
  ((3 * (Real.sin t - 2)) / (3 - Real.cos t), (4 * (Real.cos t - 4)) / (3 - Real.cos t))

/-- Coefficients of the ellipse equation --/
def ellipse_coeffs : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ := (12, 4, 12, 16, 84, 304)

theorem ellipse_equation_sum (A B C D E F : ℤ) 
  (h1 : ∀ t, ∃ x y, ellipse t = (x, y) ∧ 
    A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0)
  (h2 : Int.gcd (abs A) (Int.gcd (abs B) (Int.gcd (abs C) 
    (Int.gcd (abs D) (Int.gcd (abs E) (abs F))))) = 1)
  (h3 : (A, B, C, D, E, F) = ellipse_coeffs) :
  abs A + abs B + abs C + abs D + abs E + abs F = 432 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_sum_l161_16193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_non_square_formula_l161_16155

/-- The n-th non-square positive integer -/
def nth_non_square (n : ℕ+) : ℕ+ :=
  sorry

/-- The integer closest to a positive real number -/
noncomputable def closest_integer (x : ℝ) : ℤ :=
  sorry

/-- For positive real numbers ending in .5, round down -/
axiom closest_integer_half (m : ℤ) :
  closest_integer (m + 0.5) = m

/-- The main theorem: x_n = n + ⟨√n⟩ -/
theorem nth_non_square_formula (n : ℕ+) :
  (nth_non_square n : ℝ) = n + closest_integer (Real.sqrt n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_non_square_formula_l161_16155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l161_16146

/-- The function f(x) = 3x² - 2ln(x) -/
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2 * Real.log x

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := 6 * x - 2 / x

theorem monotonic_increasing_interval :
  ∀ x : ℝ, x > 0 → (f' x > 0 ↔ x > Real.sqrt 3 / 3) :=
by
  intro x hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l161_16146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_three_l161_16170

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x) else Real.log x / Real.log 81

-- State the theorem
theorem unique_solution_is_three :
  ∃! x, f x = 1/4 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_three_l161_16170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducible_fraction_to_repeating_decimal_l161_16140

def is_repeating_decimal (p q : ℕ+) : Prop :=
  (p : ℚ) / q = 18 / 99

theorem irreducible_fraction_to_repeating_decimal (p q : ℕ+) :
  is_repeating_decimal p q →
  Nat.Coprime p q →
  (∀ q' : ℕ+, q' < q → ¬ is_repeating_decimal p q') →
  p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducible_fraction_to_repeating_decimal_l161_16140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_l161_16176

theorem circle_points (P Q R S : ℝ × ℝ) (h : ℝ) :
  P = (9, 12) →
  S = (h, 0) →
  ‖P‖ = ‖Q‖ →
  ‖S‖ = ‖R‖ →
  ‖Q - R‖ = 5 →
  h = 10 ∨ h = -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_l161_16176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_significant_improvement_l161_16185

def old_device_data : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def sample_mean (data : List Float) : Float :=
  (data.sum) / (data.length.toFloat)

def sample_variance (data : List Float) : Float :=
  let mean := sample_mean data
  (data.map (fun x => (x - mean) ^ 2)).sum / (data.length.toFloat)

def x_bar : Float := sample_mean old_device_data
def y_bar : Float := sample_mean new_device_data
def s1_squared : Float := sample_variance old_device_data
def s2_squared : Float := sample_variance new_device_data

theorem significant_improvement : 
  y_bar - x_bar ≥ 2 * (((s1_squared + s2_squared) / 10).sqrt) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_significant_improvement_l161_16185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_A_to_l_l161_16159

/-- The line l: x + y + 3 = 0 -/
def line_l (x y : ℝ) : Prop := x + y + 3 = 0

/-- Point A with coordinates (2, 1) -/
def point_A : ℝ × ℝ := (2, 1)

/-- A point P on the line l -/
def point_P (x y : ℝ) : Prop := line_l x y

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The minimum distance between point A and any point on line l is 3√2 -/
theorem min_distance_A_to_l :
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 ∧
  ∀ (x y : ℝ), point_P x y →
  distance point_A (x, y) ≥ d := by
  sorry

#check min_distance_A_to_l

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_A_to_l_l161_16159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_double_radius_l161_16129

/-- The volume of a sphere with radius r -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- Theorem: When a sphere's radius is doubled, its volume becomes 8 times the original volume -/
theorem sphere_volume_double_radius (r : ℝ) (hr : r > 0) :
  sphereVolume (2 * r) = 8 * sphereVolume r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_double_radius_l161_16129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_decreasing_l161_16188

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := -x + 1
noncomputable def g (x : ℝ) : ℝ := -3 / x
noncomputable def h (x : ℝ) : ℝ := x^2 + x - 2

-- State the theorem
theorem only_f_decreasing :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
  (¬∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ > g x₂) ∧
  (¬∀ x₁ x₂ : ℝ, x₁ < x₂ → h x₁ > h x₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_decreasing_l161_16188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_trebled_after_five_years_l161_16151

/-- Represents the number of years after which the principal is trebled -/
def years_to_treble : ℕ := sorry

/-- The original principal amount -/
def principal : ℚ := sorry

/-- The annual interest rate as a percentage -/
def rate : ℚ := sorry

/-- Calculates simple interest -/
noncomputable def simple_interest (p : ℚ) (r : ℚ) (t : ℕ) : ℚ :=
  (p * r * t) / 100

theorem principal_trebled_after_five_years :
  simple_interest principal rate 10 = 800 ∧
  simple_interest principal rate years_to_treble +
    simple_interest (3 * principal) rate (10 - years_to_treble) = 1600 →
  years_to_treble = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_trebled_after_five_years_l161_16151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_ellipse_foci_distance_l161_16137

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxesEllipse where
  /-- The x-coordinate of the point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ
  /-- The y-coordinate of the point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ

/-- The distance between the foci of a parallel axes ellipse -/
noncomputable def foci_distance (e : ParallelAxesEllipse) : ℝ :=
  2 * Real.sqrt ((2 * e.x_tangent) ^ 2 - (2 * e.y_tangent) ^ 2)

/-- Theorem: The distance between the foci of the specific ellipse is 6√3 -/
theorem specific_ellipse_foci_distance :
  let e : ParallelAxesEllipse := { x_tangent := 6, y_tangent := 3 }
  foci_distance e = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_ellipse_foci_distance_l161_16137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l161_16142

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x + Real.pi / 3)

theorem function_symmetry (ω : ℝ) (h : ω > 0) :
  ∀ x : ℝ, f ω (x + Real.pi / 6) = -f ω (-x - Real.pi / 6) := by
  intro x
  simp [f]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l161_16142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_probability_l161_16191

/-- Represents the days of the week for the activity -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- Helper function to get the next day -/
def nextDay : Day → Option Day
| Day.Monday => some Day.Tuesday
| Day.Tuesday => some Day.Wednesday
| Day.Wednesday => some Day.Thursday
| Day.Thursday => some Day.Friday
| Day.Friday => none

/-- Represents a pair of consecutive days -/
structure ConsecutiveDayPair :=
  (first : Day)
  (second : Day)
  (consecutive : (nextDay first = some second) ∨ (first = Day.Friday ∧ second = Day.Monday))

/-- The set of all valid consecutive day pairs -/
def allPairs : Finset ConsecutiveDayPair := sorry

/-- The set of consecutive day pairs that include Tuesday -/
def tuesdayPairs : Finset ConsecutiveDayPair := sorry

/-- Theorem stating the probability of choosing a pair including Tuesday -/
theorem tuesday_probability :
  (Finset.card tuesdayPairs : ℚ) / (Finset.card allPairs : ℚ) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_probability_l161_16191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l161_16131

/-- Curve C in parametric form -/
noncomputable def curve_C (φ : ℝ) : ℝ × ℝ := (1 + 2 * Real.cos φ, 1 + 2 * Real.sin φ)

/-- Line l in polar form -/
noncomputable def line_l (θ : ℝ) : ℝ := 4 / (Real.cos θ - Real.sin θ)

/-- Distance function from a point on C to line l -/
noncomputable def distance (φ : ℝ) : ℝ :=
  |2 * Real.cos φ - 2 * Real.sin φ - 4| / Real.sqrt 2

theorem min_distance_curve_to_line :
  ∃ (φ : ℝ), distance φ = Real.sqrt 2 ∧
  curve_C φ = (1 + Real.sqrt 2, 1 - Real.sqrt 2) ∧
  ∀ (ψ : ℝ), distance ψ ≥ Real.sqrt 2 := by
  sorry

#check min_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l161_16131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l161_16149

open Real

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- State the conditions
axiom f_plus_f'_gt_1 : ∀ x, f x + f' x > 1
axiom f_0_eq_2016 : f 0 = 2016

-- State the theorem
theorem solution_set (x : ℝ) : exp x * f x > exp x + 2015 ↔ x > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l161_16149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l161_16136

noncomputable def f (x : ℝ) := (1/2) * x^2 - Real.log x - 1

theorem zero_in_interval :
  ∃ x : ℝ, x > 1 ∧ x < 2 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l161_16136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_prime_divisors_of_2520_l161_16138

def n : ℕ := 2520

theorem sum_distinct_prime_divisors_of_2520 :
  (Finset.filter (fun p => Nat.Prime p ∧ n % p = 0) (Finset.range (n + 1))).sum id = 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_prime_divisors_of_2520_l161_16138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_perpendicular_l161_16106

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The focal length of an ellipse -/
noncomputable def Ellipse.focalLength (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- Represents a point on the ellipse -/
structure EllipsePoint (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The line y = -x + m intersecting the ellipse -/
def intersectionLine (m : ℝ) (x : ℝ) : ℝ := -x + m

theorem ellipse_intersection_perpendicular (e : Ellipse) (m : ℝ) :
  e.eccentricity = Real.sqrt 3 / 3 →
  e.focalLength = 2 →
  ∃ (A B : EllipsePoint e),
    A.y = intersectionLine m A.x ∧
    B.y = intersectionLine m B.x ∧
    A.x * B.x + A.y * B.y = 0 →
  m = 2 * Real.sqrt 15 / 5 ∨ m = -2 * Real.sqrt 15 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_perpendicular_l161_16106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heart_probability_is_one_fourth_l161_16135

/-- Represents a standard deck of cards -/
structure Deck where
  cards : Finset (Nat × Nat)
  card_count : cards.card = 52
  rank_count : (cards.image Prod.fst).card = 13
  suit_count : (cards.image Prod.snd).card = 4
  unique_cards : ∀ r s, (r, s) ∈ cards → (∃! c, c ∈ cards ∧ c.1 = r ∧ c.2 = s)

/-- Represents the ♥ suit -/
def heart_suit : Nat := 1

/-- The probability of drawing a ♥ card from a standard deck -/
def heart_probability (d : Deck) : ℚ :=
  (d.cards.filter (fun c => c.2 = heart_suit)).card / d.cards.card

/-- Theorem stating that the probability of drawing a ♥ card is 1/4 -/
theorem heart_probability_is_one_fourth (d : Deck) :
  heart_probability d = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heart_probability_is_one_fourth_l161_16135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_in_A_P_l161_16104

/-- A polynomial of degree 8 -/
def Polynomial8 := Polynomial ℝ

/-- The set A_P for a polynomial P -/
def A_P (P : Polynomial8) (c : ℝ) : Set ℝ :=
  {x : ℝ | P.eval x = c}

/-- The theorem statement -/
theorem min_odd_in_A_P (P : Polynomial8) (c : ℝ) (h : 8 ∈ A_P P c) :
  ∃ (x : ℝ), x ∈ A_P P c ∧ Odd x := by
  sorry

#check min_odd_in_A_P

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_in_A_P_l161_16104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_minus_losses_is_integer_l161_16145

-- Define the possible game outcomes
inductive GameOutcome
| Win
| Draw
| Loss
deriving BEq, Repr

-- Define the scoring function
def score (outcome : GameOutcome) : Rat :=
  match outcome with
  | GameOutcome.Win => 1
  | GameOutcome.Draw => 1/2
  | GameOutcome.Loss => 0

-- Define a tournament as a list of game outcomes
def Tournament := List GameOutcome

-- Calculate the total score of a tournament
def totalScore (t : Tournament) : Rat :=
  t.map score |>.sum

-- Count the number of losses in a tournament
def lossCount (t : Tournament) : Nat :=
  t.filter (· == GameOutcome.Loss) |>.length

-- Theorem: The difference between total score and loss count is always an integer
theorem score_minus_losses_is_integer (t : Tournament) :
  ∃ n : Int, totalScore t - lossCount t = n := by
  sorry

#eval totalScore [GameOutcome.Win, GameOutcome.Draw, GameOutcome.Loss]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_minus_losses_is_integer_l161_16145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_derivative_implies_a_equals_one_l161_16177

/-- Given a real number a, define f(x) = e^x + e^(-ax). If xf'(x) is an even function, then a = 1. -/
theorem even_derivative_implies_a_equals_one (a : ℝ) : 
  (let f := fun x => Real.exp x + Real.exp (-a * x)
   let f' := fun x => Real.exp x - a * Real.exp (-a * x)
   (∀ x, x * (f' x) = (x * f' (-x)))) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_derivative_implies_a_equals_one_l161_16177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cathy_silver_tokens_l161_16107

/-- Represents the number of tokens Cathy has -/
structure TokenCount where
  red : Int
  blue : Int
  silver : Int

/-- Represents the exchange rates at the booths -/
structure ExchangeRates where
  first_booth : TokenCount
  second_booth : TokenCount

/-- Checks if further exchanges are possible -/
def can_exchange (tokens : TokenCount) (rates : ExchangeRates) : Prop :=
  tokens.red ≥ rates.first_booth.red ∨ tokens.blue ≥ rates.second_booth.blue

/-- The final state after all possible exchanges -/
noncomputable def final_state (initial : TokenCount) (rates : ExchangeRates) : TokenCount :=
  sorry

/-- Theorem stating that Cathy ends up with 100 silver tokens -/
theorem cathy_silver_tokens :
  let initial := TokenCount.mk 100 100 0
  let rates := ExchangeRates.mk
    (TokenCount.mk 3 (-1) (-2))  -- First booth: -3 red, +1 blue, +2 silver
    (TokenCount.mk 1 4 (-2))  -- Second booth: +1 red, -4 blue, +2 silver
  (final_state initial rates).silver = 100 := by
  sorry

#check cathy_silver_tokens

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cathy_silver_tokens_l161_16107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_proof_l161_16114

/-- The sequence a_n defined as (4n^2 + 1) / (3n^2 + 2) converges to 4/3 as n approaches infinity. -/
theorem sequence_limit_proof (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N →
    |((4 * (n : ℝ)^2 + 1) / (3 * (n : ℝ)^2 + 2)) - 4/3| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_proof_l161_16114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_age_problem_l161_16173

theorem cricket_team_age_problem (team_size : ℕ) (team_avg_age : ℝ) 
  (wicket_keeper_age : ℝ) (captain_age : ℝ) (vice_captain_age : ℝ) :
  team_size = 11 →
  team_avg_age = 27 →
  wicket_keeper_age = team_avg_age + 3 →
  captain_age = team_avg_age - 2 →
  vice_captain_age = team_avg_age →
  (team_avg_age * team_size - (team_avg_age + wicket_keeper_age + captain_age + vice_captain_age)) / (team_size - 4 : ℝ) = team_avg_age - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_age_problem_l161_16173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_double_angle_l161_16118

/-- Given that vectors (sin α, 2) and (cos α, 1) are parallel, prove that tan(2α) = -4/3 -/
theorem parallel_vectors_tan_double_angle (α : ℝ) :
  let a : Fin 2 → ℝ := ![Real.sin α, 2]
  let b : Fin 2 → ℝ := ![Real.cos α, 1]
  (∃ (k : ℝ), a = k • b) →
  Real.tan (2 * α) = -4/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_double_angle_l161_16118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_arithmetic_equation_l161_16116

theorem binary_arithmetic_equation : 
  (0b1101 : Nat) + (0b1011 : Nat) - (0b101 : Nat) + (0b111 : Nat) = (0b11010 : Nat) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_arithmetic_equation_l161_16116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_divisible_by_24_l161_16111

theorem power_divisible_by_24 (m : ℕ) (hm : m > 0) :
  (∃ k : ℕ, 24 ∣ m ^ k) →
  (∀ d : ℕ, d > 0 → d ∣ m → d ≤ 8) →
  (∃ k : ℕ, k > 0 ∧ 24 ∣ m ^ k ∧ ∀ j : ℕ, j > 0 → 24 ∣ m ^ j → k ≤ j) →
  (∃ k : ℕ, k > 0 ∧ 24 ∣ m ^ k ∧ k = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_divisible_by_24_l161_16111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_special_angles_l161_16126

-- Define the tangent function for degrees
noncomputable def tan_deg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- State the theorem
theorem tan_sum_special_angles :
  tan_deg 10 * tan_deg 20 + Real.sqrt 3 * (tan_deg 10 + tan_deg 20) = 1 :=
by
  -- Assuming the following:
  have tan_30 : tan_deg 30 = Real.sqrt 3 / 3 := sorry
  have tan_sum (a b : ℝ) : tan_deg (a + b) = (tan_deg a + tan_deg b) / (1 - tan_deg a * tan_deg b) := sorry
  
  sorry -- The proof goes here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_special_angles_l161_16126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_council_fundraiser_l161_16110

theorem student_council_fundraiser 
  (boxes : ℕ) 
  (erasers_per_box : ℕ) 
  (price_per_eraser : ℚ) 
  (h1 : boxes = 48)
  (h2 : erasers_per_box = 24)
  (h3 : price_per_eraser = 3/4)
  : boxes * erasers_per_box * price_per_eraser = 864 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_council_fundraiser_l161_16110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_from_face_perimeter_l161_16165

/-- Given a cube with a face perimeter of 20 cm, its volume is 125 cubic centimeters. -/
theorem cube_volume_from_face_perimeter : 
  ∀ (cube : Real → Real → Real → Real), 
  (∃ (side : Real), 4 * side = 20 ∧ cube = λ x y z ↦ x * y * z) →
  (∀ x y z, 0 ≤ x ∧ x ≤ side ∧ 0 ≤ y ∧ y ≤ side ∧ 0 ≤ z ∧ z ≤ side → cube x y z ≤ 125) ∧
  (∃ x y z, cube x y z = 125) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_from_face_perimeter_l161_16165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l161_16123

theorem zero_point_interval (a b : ℝ) (n : ℤ) : 
  (2019 : ℝ) ^ a = 2020 →
  (2020 : ℝ) ^ b = 2019 →
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioo (n : ℝ) ((n + 1) : ℝ) ∧ a + x₀ - b ^ x₀ = 0) →
  n = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l161_16123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_shaded_region_l161_16166

noncomputable section

-- Define the radii of the three circles
def r₁ : ℝ := 4
def r₂ : ℝ := 3
def r₃ : ℝ := 2

-- Define the total area of the circles
noncomputable def total_area : ℝ := Real.pi * (r₁^2 + r₂^2 + r₃^2)

-- Define the ratio of shaded area to total area
def shaded_ratio : ℝ := 2 / 7

-- Define the acute angle formed by the two lines
noncomputable def θ : ℝ := 5 * Real.pi / 77

-- Theorem statement
theorem angle_of_shaded_region :
  let shaded_area := shaded_ratio * total_area
  shaded_area = (r₁^2 + r₃^2) * θ + r₂^2 * (Real.pi - θ) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_shaded_region_l161_16166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_decreasing_function_m_value_l161_16120

/-- A function y is linear in x if it can be written as y = ax + b for some constants a and b -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

/-- A function y decreases as x increases if for any x1 < x2, y(x1) > y(x2) -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2

/-- The function y = (m+1)x^(3-|m|) + 2 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  (m + 1) * x^(3 - abs m) + 2

theorem linear_decreasing_function_m_value :
  (∃ m : ℝ, IsLinearFunction (f m) ∧ IsDecreasing (f m)) →
  (∃ m : ℝ, m = -2 ∧ IsLinearFunction (f m) ∧ IsDecreasing (f m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_decreasing_function_m_value_l161_16120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stephen_round_trips_l161_16178

def mountain_height : ℚ := 40000
def fraction_climbed : ℚ := 3/4
def total_distance : ℚ := 600000

theorem stephen_round_trips :
  let distance_per_trip := fraction_climbed * mountain_height
  let round_trip_distance := 2 * distance_per_trip
  total_distance / round_trip_distance = 10 := by
  -- Unfold the definitions
  unfold mountain_height fraction_climbed total_distance
  -- Simplify the expressions
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stephen_round_trips_l161_16178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_identity_l161_16199

theorem log_product_identity (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x / Real.log (y^3)) * (Real.log (y^3) / Real.log (x^4)) * (Real.log (x^2) / Real.log (y^5)) *
  (Real.log (y^5) / Real.log (x^2)) * (Real.log (x^4) / Real.log (y^3)) = (1/3) * (Real.log x / Real.log y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_identity_l161_16199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_periodic_word_l161_16181

/-- Represents a word in the sequence --/
inductive Word
| A : Word
| B : Word
| Concat : Word → Word → Word

/-- Generates the k-th word in the sequence --/
def generate_word : ℕ → Word
| 0 => Word.A  -- Added case for 0
| 1 => Word.A
| 2 => Word.B
| (n + 3) => Word.Concat (generate_word (n + 2)) (generate_word (n + 1))

/-- Counts the number of 'A's in a word --/
def count_A : Word → ℕ
| Word.A => 1
| Word.B => 0
| (Word.Concat w1 w2) => count_A w1 + count_A w2

/-- Counts the number of 'B's in a word --/
def count_B : Word → ℕ
| Word.A => 0
| Word.B => 1
| (Word.Concat w1 w2) => count_B w1 + count_B w2

/-- Checks if a word is periodic --/
def is_periodic (w : Word) : Prop :=
  ∃ (subword : Word), w = Word.Concat subword subword ∨ 
    ∃ (subword' : Word), w = Word.Concat (Word.Concat subword subword') subword

/-- The main theorem: no word in the sequence is periodic --/
theorem no_periodic_word (n : ℕ) : ¬ (is_periodic (generate_word n)) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_periodic_word_l161_16181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_inequality_l161_16186

theorem largest_lambda_inequality :
  ∃ (lambda_max : ℝ), ∀ (lambda : ℝ),
    (∀ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 ≥ a*b + lambda*b*c + c*d) →
    lambda ≤ lambda_max ∧
    lambda_max = (3 : ℝ) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_inequality_l161_16186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log9_3_sqrt3_equals_3_4_l161_16144

-- Define the logarithm function for base 9
noncomputable def log9 (x : ℝ) : ℝ := Real.log x / Real.log 9

-- State the theorem
theorem log9_3_sqrt3_equals_3_4 : log9 (3 * Real.sqrt 3) = 3 / 4 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log9_3_sqrt3_equals_3_4_l161_16144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_equals_e_l161_16112

/-- The work done by a force F(x) = 1 + e^x on a particle moving from x = 0 to x = 1 along the x-axis -/
noncomputable def work : ℝ := ∫ x in (0 : ℝ)..(1 : ℝ), (1 + Real.exp x)

/-- Theorem stating that the work done is equal to e -/
theorem work_equals_e : work = Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_equals_e_l161_16112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_wins_four_rounds_prob_C_wins_three_rounds_l161_16121

-- Define the probabilities
def p_A_beats_B : ℝ := 0.4
def p_B_beats_C : ℝ := 0.5
def p_C_beats_A : ℝ := 0.6

-- Define the game rules (we'll use these in the theorem statements)
def game_rules : Prop := True  -- Placeholder for the game rules

-- Theorem for B winning four consecutive rounds
theorem prob_B_wins_four_rounds (h : game_rules) : 
  (1 - p_A_beats_B)^2 * p_B_beats_C^2 = 0.09 := by
  sorry

-- Theorem for C winning three consecutive rounds
theorem prob_C_wins_three_rounds (h : game_rules) : 
  p_A_beats_B * p_C_beats_A^2 * p_B_beats_C + 
  (1 - p_A_beats_B) * p_B_beats_C^2 * p_C_beats_A = 0.162 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_wins_four_rounds_prob_C_wins_three_rounds_l161_16121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_equals_106_l161_16168

theorem fraction_sum_equals_106 (a b : ℕ+) : 
  (∃ (S : ℕ → ℚ), 
    (∀ n : ℕ, S n = if n % 2 = 0 then (2 * n + 1) / (4 ^ (n / 2 + 1)) else (2 * n + 1) / (2 ^ (n / 2 + 1))) ∧
    (a.val : ℚ) / b.val = ∑' n, S n) →
  Nat.Coprime a.val b.val →
  a.val + b.val = 106 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_equals_106_l161_16168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l161_16162

-- Define the hyperbola C
def hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}

-- Define the foci
noncomputable def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (a^2 + b^2)
  ((-c, 0), (c, 0))

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

-- State the theorem
theorem hyperbola_eccentricity
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (A B : ℝ × ℝ)
  (hA : A ∈ hyperbola a b h1 h2)
  (hB : B ∈ hyperbola a b h1 h2)
  (F₁ F₂ : ℝ × ℝ)
  (hF : (F₁, F₂) = foci a b)
  (h_orthogonal : (A.1 - F₁.1, A.2 - F₁.2) • (A.1 - F₂.1, A.2 - F₂.2) = 0)
  (h_relation : (F₂.1 - B.1, F₂.2 - B.2) + 2 * (F₂.1 - A.1, F₂.2 - A.2) = (0, 0)) :
  eccentricity a b = Real.sqrt 17 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l161_16162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_T_is_765_l161_16101

/-- A function g satisfying the given functional equation -/
noncomputable def g : ℝ → ℝ := sorry

/-- The functional equation that g satisfies for all non-zero x -/
axiom g_eq (x : ℝ) (hx : x ≠ 0) : 3 * g x + g (1 / x) = 7 * x + 6

/-- The sum of roots of the quadratic equation derived from g(x) = 2010 -/
noncomputable def T : ℝ := 16068 / 21

/-- Main theorem: The integer nearest to T is 765 -/
theorem nearest_integer_to_T_is_765 : ⌊T + 1/2⌋ = 765 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_T_is_765_l161_16101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_savings_l161_16197

/-- Calculates Mr. Kishore's savings given his expenses and savings rate -/
def calculate_savings (rent milk groceries education petrol misc : ℕ) (savings_rate : ℚ) : ℕ :=
  let total_expenses := rent + milk + groceries + education + petrol + misc
  let salary := (total_expenses : ℚ) / (1 - savings_rate)
  (salary * savings_rate).floor.toNat

/-- Theorem stating that Mr. Kishore's savings are 1800 Rs. -/
theorem kishore_savings :
  calculate_savings 5000 1500 4500 2500 2000 700 (1/10) = 1800 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_savings_l161_16197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_result_l161_16127

def matrix_product (n : ℕ) : Matrix (Fin 2) (Fin 2) ℚ :=
  (List.range n).foldl
    (fun acc i => acc * !![1, (2*(i+1) : ℚ); 0, 1])
    !![1, 0; 0, 1]

theorem matrix_product_result :
  matrix_product 50 = !![1, 2550; 0, 1] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_result_l161_16127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_x_minus_pi_half_l161_16132

theorem cos_neg_x_minus_pi_half (x : ℝ) (h1 : x ∈ Set.Ioo (π/2) π) (h2 : Real.tan x = -4/3) :
  Real.cos (-x - π/2) = -3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_x_minus_pi_half_l161_16132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_primes_count_l161_16183

def digit_set : Finset Nat := {3, 5, 8, 9}

def is_two_digit (n : Nat) : Prop :=
  10 ≤ n ∧ n ≤ 99

def digits_from_set (n : Nat) : Prop :=
  (n / 10) ∈ digit_set ∧ (n % 10) ∈ digit_set

def different_digits (n : Nat) : Prop :=
  (n / 10) ≠ (n % 10)

theorem two_digit_primes_count :
  ∃ (S : Finset Nat),
    (∀ n ∈ S, is_two_digit n ∧ digits_from_set n ∧ different_digits n ∧ Nat.Prime n) ∧
    S.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_primes_count_l161_16183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_between_zero_and_one_l161_16134

/-- Given a quadratic equation f(z) = 0 with specific coefficients and constraints,
    prove that its roots lie between 0 and 1. -/
theorem roots_between_zero_and_one
  (g h c d : ℝ)
  (g_pos : 0 < g)
  (h_pos : 0 < h)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  (d_bound : d ≤ (c / g) * Real.sqrt (c^2 + 2*g*h))
  (z : ℝ)
  (z_def : ∃ α, z = Real.cos α ^ 2)
  (f : ℝ → ℝ)
  (f_def : f = λ z ↦ 4*c^4*(h^2+d^2)*z^2 - 4*c^2*d^2*(g*h+c^2)*z + g^2*d^4)
  (f_root : f z = 0) :
  ∃ z₁ z₂, z₁ ≠ z₂ ∧ f z₁ = 0 ∧ f z₂ = 0 ∧ 0 < z₁ ∧ z₁ < 1 ∧ 0 < z₂ ∧ z₂ < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_between_zero_and_one_l161_16134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l161_16117

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the circle (renamed to avoid conflict)
def hyperbola_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

-- Define the focus
def focus (x y : ℝ) : Prop :=
  x = 2 ∧ y = 0

-- Define the eccentricity
noncomputable def eccentricity (c a : ℝ) : ℝ :=
  c / a

-- Theorem statement
theorem hyperbola_eccentricity :
  ∃ (a b : ℝ),
    (∃ (x y : ℝ), hyperbola a b x y) ∧
    (∃ (x y : ℝ), focus x y) ∧
    (∃ (x y : ℝ), hyperbola a b x y ∧ hyperbola_circle x y) →
    eccentricity 2 1 = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l161_16117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_change_l161_16161

theorem arithmetic_progression_sum_change 
  (n : ℕ) 
  (a₁ d : ℝ) 
  (h₁ : a₁ > 0) 
  (h₂ : d > 0) 
  (h₃ : ∀ (S : ℝ), 
    S = (n / 2) * (2 * a₁ + (n - 1) * d) → 
    3 * S = (n / 2) * (2 * a₁ + (n - 1) * (4 * d))) :
  (n / 2) * (2 * a₁ + (n - 1) * (2 * d)) / ((n / 2) * (2 * a₁ + (n - 1) * d)) = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_change_l161_16161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_is_12pi_l161_16189

-- Define the radius of the unfolded semicircle
def semicircle_radius : ℝ := 4

-- Define the surface area of the cone
noncomputable def cone_surface_area (semicircle_radius : ℝ) : ℝ := 
  let base_radius := semicircle_radius / 2
  Real.pi * base_radius^2 + Real.pi * base_radius * semicircle_radius

-- Theorem statement
theorem cone_surface_area_is_12pi : 
  cone_surface_area semicircle_radius = 12 * Real.pi := by
  -- Unfold the definition of cone_surface_area
  unfold cone_surface_area
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_is_12pi_l161_16189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_income_l161_16130

/-- Represents the tax regulations and company's income --/
structure TaxSystem where
  p : ℝ  -- Base tax rate in percentage
  income : ℝ  -- Company's annual income in millions of yuan

/-- The tax amount for a given income under the specified tax system --/
noncomputable def taxAmount (ts : TaxSystem) : ℝ :=
  if ts.income ≤ 280 then
    ts.income * (ts.p / 100)
  else
    280 * (ts.p / 100) + (ts.income - 280) * ((ts.p + 2) / 100)

/-- The actual tax rate for a given income --/
noncomputable def actualTaxRate (ts : TaxSystem) : ℝ :=
  (taxAmount ts / ts.income) * 100

/-- Theorem stating that given the tax regulations and the company's actual tax rate,
    the company's annual income is 320 million yuan --/
theorem company_income (ts : TaxSystem) :
  actualTaxRate ts = ts.p + 0.25 → ts.income = 320 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_income_l161_16130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_darrels_coin_count_l161_16174

/-- Calculate the total amount Darrel receives after fees and conversions -/
theorem darrels_coin_count (
  quarters : ℕ
) (dimes : ℕ)
  (nickels : ℕ)
  (pennies : ℕ)
  (half_dollars : ℕ)
  (euro_coins : ℕ)
  (pound_coins : ℕ)
  (quarter_value : ℚ)
  (dime_value : ℚ)
  (nickel_value : ℚ)
  (penny_value : ℚ)
  (half_dollar_value : ℚ)
  (euro_to_usd : ℚ)
  (pound_to_usd : ℚ)
  (quarter_fee : ℚ)
  (dime_fee : ℚ)
  (nickel_fee : ℚ)
  (penny_fee : ℚ)
  (half_dollar_fee : ℚ)
  (euro_fee : ℚ)
  (pound_fee : ℚ)
  (h1 : quarters = 127)
  (h2 : dimes = 183)
  (h3 : nickels = 47)
  (h4 : pennies = 237)
  (h5 : half_dollars = 64)
  (h6 : euro_coins = 32)
  (h7 : pound_coins = 55)
  (h8 : quarter_value = 25 / 100)
  (h9 : dime_value = 10 / 100)
  (h10 : nickel_value = 5 / 100)
  (h11 : penny_value = 1 / 100)
  (h12 : half_dollar_value = 50 / 100)
  (h13 : euro_to_usd = 118 / 100)
  (h14 : pound_to_usd = 139 / 100)
  (h15 : quarter_fee = 12 / 100)
  (h16 : dime_fee = 7 / 100)
  (h17 : nickel_fee = 15 / 100)
  (h18 : penny_fee = 10 / 100)
  (h19 : half_dollar_fee = 5 / 100)
  (h20 : euro_fee = 3 / 100)
  (h21 : pound_fee = 4 / 100) :
  ℚ := by
  let total_after_fees :=
    (quarters * quarter_value * (1 - quarter_fee)) +
    (dimes * dime_value * (1 - dime_fee)) +
    (nickels * nickel_value * (1 - nickel_fee)) +
    (pennies * penny_value * (1 - penny_fee)) +
    (half_dollars * half_dollar_value * (1 - half_dollar_fee)) +
    (euro_coins * euro_to_usd * (1 - euro_fee)) +
    (pound_coins * pound_to_usd * (1 - pound_fee))
  have : total_after_fees = 18951 / 100 := by sorry
  exact total_after_fees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_darrels_coin_count_l161_16174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_parallel_line_in_plane_l161_16172

-- Define the basic structures
structure Plane where

structure Line where

structure Point where

-- Define the properties and relations
axiom parallel : Line → Line → Prop

axiom parallelToPlane : Line → Plane → Prop

axiom pointOnPlane : Point → Plane → Prop

axiom linePassesThrough : Line → Point → Prop

axiom lineInPlane : Line → Plane → Prop

-- State the theorem
theorem unique_parallel_line_in_plane 
  (l : Line) (α : Plane) (P : Point)
  (h1 : parallelToPlane l α)
  (h2 : pointOnPlane P α) :
  ∃! m : Line, 
    linePassesThrough m P ∧ 
    parallel m l ∧ 
    lineInPlane m α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_parallel_line_in_plane_l161_16172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_n_plus_one_l161_16113

theorem polynomial_value_at_n_plus_one (n : ℕ) (P : ℝ → ℝ) :
  (∀ (x : ℝ), ∃ (p : Polynomial ℝ), P x = p.eval x) →
  (∀ (k : ℕ), k ≤ n → P k = k / (k + 1)) →
  P (n + 1) = (n + 1 + (-1)^(n + 1)) / (n + 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_n_plus_one_l161_16113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l161_16143

theorem sufficient_not_necessary : 
  (∀ x : ℝ, abs (2*x - 1) ≤ x → x^2 + x - 2 ≤ 0) ∧ 
  (∃ x : ℝ, x^2 + x - 2 ≤ 0 ∧ ¬(abs (2*x - 1) ≤ x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l161_16143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ACD_is_22_l161_16124

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the angles in the quadrilateral
def angle_ABD (q : Quadrilateral) : ℝ := 70
def angle_CAD (q : Quadrilateral) : ℝ := 20
def angle_BAC (q : Quadrilateral) : ℝ := 48
def angle_CBD (q : Quadrilateral) : ℝ := 40

-- Define angle_ACD
def angle_ACD (q : Quadrilateral) : ℝ := 22

-- Define the theorem
theorem angle_ACD_is_22 (q : Quadrilateral) :
  angle_ABD q = 70 ∧
  angle_CAD q = 20 ∧
  angle_BAC q = 48 ∧
  angle_CBD q = 40 →
  angle_ACD q = 22 :=
by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ACD_is_22_l161_16124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_level_transform_inverse_prop_l161_16195

-- Define the k-level transformation
noncomputable def k_level_transform (k : ℝ) (a b : ℝ) : ℝ × ℝ := (k * a, -k * b)

-- Define the inverse proportion function
noncomputable def inverse_prop (x : ℝ) : ℝ := -18 / x

-- Theorem statement
theorem k_level_transform_inverse_prop :
  ∃ (k₁ k₂ : ℝ), k₁ ≠ 0 ∧ k₂ ≠ 0 ∧
  k_level_transform k₁ 1 2 = (3, -6) ∧
  k_level_transform k₂ 1 2 = (-3, 6) ∧
  inverse_prop 3 = -6 ∧
  inverse_prop (-3) = 6 := by
  -- Proof goes here
  sorry

#check k_level_transform_inverse_prop

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_level_transform_inverse_prop_l161_16195
