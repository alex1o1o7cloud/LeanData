import Mathlib

namespace stratified_sampling_proportion_l1242_124231

-- Define the total number of purchases and samples for the first category
def purchases_category1 : ℕ := 116000
def samples_category1 : ℕ := 116

-- Define the number of purchases for the second category
def purchases_category2 : ℕ := 94000

-- Define the function to calculate the number of samples for the second category
def samples_category2 : ℚ := (samples_category1 : ℚ) * (purchases_category2 : ℚ) / (purchases_category1 : ℚ)

-- Theorem statement
theorem stratified_sampling_proportion :
  samples_category2 = 94 := by
  sorry

end stratified_sampling_proportion_l1242_124231


namespace agent_commission_l1242_124213

def commission_rate : ℝ := 0.025
def sales : ℝ := 840

theorem agent_commission :
  sales * commission_rate = 21 := by sorry

end agent_commission_l1242_124213


namespace equation_solution_l1242_124296

theorem equation_solution (x : ℝ) : 3*x - 5 = 10*x + 9 → 4*(x + 7) = 20 := by
  sorry

end equation_solution_l1242_124296


namespace mans_walking_speed_l1242_124265

theorem mans_walking_speed (woman_speed : ℝ) (passing_wait_time : ℝ) (catch_up_time : ℝ) :
  woman_speed = 25 →
  passing_wait_time = 5 / 60 →
  catch_up_time = 20 / 60 →
  ∃ (man_speed : ℝ),
    woman_speed * passing_wait_time = man_speed * (passing_wait_time + catch_up_time) ∧
    man_speed = 25 / 4 := by
  sorry

end mans_walking_speed_l1242_124265


namespace right_triangle_sides_from_median_perimeters_l1242_124269

/-- Given a right triangle with a median to the hypotenuse dividing it into two triangles
    with perimeters m and n, this theorem states the sides of the original triangle. -/
theorem right_triangle_sides_from_median_perimeters (m n : ℝ) 
  (h₁ : m > 0) (h₂ : n > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    ∃ (x : ℝ), x > 0 ∧
      x^2 = (a/2)^2 + (b/2)^2 ∧
      m = x + (c/2 - x) + b ∧
      n = x + (c/2 - x) + a ∧
      a = Real.sqrt (2*m*n) - m ∧
      b = Real.sqrt (2*m*n) - n ∧
      c = n + m - Real.sqrt (2*m*n) :=
by
  sorry


end right_triangle_sides_from_median_perimeters_l1242_124269


namespace max_sum_given_quadratic_l1242_124229

theorem max_sum_given_quadratic (a b : ℝ) (h : a^2 - a*b + b^2 = 1) : a + b ≤ 2 := by
  sorry

end max_sum_given_quadratic_l1242_124229


namespace sufficient_condition_for_equation_l1242_124263

theorem sufficient_condition_for_equation (a : ℝ) (f g h : ℝ → ℝ) 
  (ha : a > 1)
  (h_sum_nonneg : ∀ x, f x + g x + h x ≥ 0)
  (h_common_root : ∃ x₀, f x₀ = 0 ∧ g x₀ = 0 ∧ h x₀ = 0) :
  ∃ x, a^(f x) + a^(g x) + a^(h x) = 3 := by
  sorry

end sufficient_condition_for_equation_l1242_124263


namespace cubic_function_properties_l1242_124289

/-- A cubic function with parameters m and n -/
def f (m n x : ℝ) : ℝ := x^3 + m*x^2 + n*x

/-- The derivative of f with respect to x -/
def f' (m n x : ℝ) : ℝ := 3*x^2 + 2*m*x + n

theorem cubic_function_properties (m n : ℝ) :
  (∀ x, f' m n x ≤ f' m n 1) →
  (f' m n 1 = 0 ∧ ∃! (a b : ℝ), a ≠ b ∧ 
    ∃ (t : ℝ), f m n t = a*t + (1 - a) ∧
    f m n t = b*t + (1 - b)) →
  (m < -3 ∧ m = -3) :=
sorry

end cubic_function_properties_l1242_124289


namespace tangent_point_condition_tangent_lines_equations_l1242_124204

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point M
def point_M (a : ℝ) : ℝ × ℝ := (1, a)

-- Theorem 1: M lies on O iff a = ±√3
theorem tangent_point_condition (a : ℝ) :
  circle_O 1 a ↔ a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
sorry

-- Theorem 2: Tangent lines when a = 2
theorem tangent_lines_equations :
  let M := point_M 2
  ∃ (l₁ l₂ : ℝ → ℝ → Prop),
    (∀ x y, l₁ x y ↔ y = 2) ∧
    (∀ x y, l₂ x y ↔ 4*x + 3*y = 10) ∧
    (∀ x y, l₁ x y → circle_O x y → x = 1 ∧ y = 2) ∧
    (∀ x y, l₂ x y → circle_O x y → x = 1 ∧ y = 2) :=
sorry

end tangent_point_condition_tangent_lines_equations_l1242_124204


namespace number_equation_l1242_124236

theorem number_equation (x : ℝ) : 3 * x - 4 = 5 ↔ x = 3 := by
  sorry

end number_equation_l1242_124236


namespace mary_work_hours_l1242_124281

/-- Mary's weekly work schedule and earnings -/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ
  tue_thu_hours : ℕ
  weekly_earnings : ℕ
  hourly_rate : ℕ

/-- Theorem stating Mary's work hours on Monday, Wednesday, and Friday -/
theorem mary_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.tue_thu_hours = 5)
  (h2 : schedule.weekly_earnings = 407)
  (h3 : schedule.hourly_rate = 11)
  (h4 : schedule.hourly_rate * (3 * schedule.mon_wed_fri_hours + 2 * schedule.tue_thu_hours) = schedule.weekly_earnings) :
  schedule.mon_wed_fri_hours = 9 := by
  sorry


end mary_work_hours_l1242_124281


namespace biased_coin_probability_l1242_124216

theorem biased_coin_probability (h : ℝ) : 
  0 < h ∧ h < 1 → 
  (Nat.choose 6 2 : ℝ) * h^2 * (1 - h)^4 = (Nat.choose 6 3 : ℝ) * h^3 * (1 - h)^3 → 
  (Nat.choose 6 4 : ℝ) * h^4 * (1 - h)^2 = 19440 / 117649 := by
sorry

end biased_coin_probability_l1242_124216


namespace x_eighth_equals_one_l1242_124206

theorem x_eighth_equals_one (x : ℝ) (h : x + 1/x = Real.sqrt 2) : x^8 = 1 := by
  sorry

end x_eighth_equals_one_l1242_124206


namespace binomial_200_200_l1242_124247

theorem binomial_200_200 : Nat.choose 200 200 = 1 := by
  sorry

end binomial_200_200_l1242_124247


namespace square_perimeter_from_rectangle_area_l1242_124248

/-- Given a rectangle with dimensions 32 cm * 10 cm, if the area of a square is five times
    the area of this rectangle, then the perimeter of the square is 160 cm. -/
theorem square_perimeter_from_rectangle_area : 
  let rectangle_length : ℝ := 32
  let rectangle_width : ℝ := 10
  let rectangle_area := rectangle_length * rectangle_width
  let square_area := 5 * rectangle_area
  let square_side := Real.sqrt square_area
  square_side * 4 = 160 := by
  sorry

end square_perimeter_from_rectangle_area_l1242_124248


namespace cube_sum_implies_sum_l1242_124222

theorem cube_sum_implies_sum (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end cube_sum_implies_sum_l1242_124222


namespace vectors_collinear_l1242_124253

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, 0)
def c : ℝ × ℝ := (2, 4)

theorem vectors_collinear : ∃ k : ℝ, k • a = b + c := by sorry

end vectors_collinear_l1242_124253


namespace zero_subset_X_l1242_124261

def X : Set ℝ := {x | x > -1}

theorem zero_subset_X : {0} ⊆ X := by sorry

end zero_subset_X_l1242_124261


namespace sum_digits_inequality_l1242_124223

/-- S(n) represents the sum of digits of a natural number n -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n, S(8n) ≥ (1/8) * S(n) -/
theorem sum_digits_inequality (n : ℕ) : S (8 * n) ≥ (1 / 8) * S n := by sorry

end sum_digits_inequality_l1242_124223


namespace fraction_operations_l1242_124210

theorem fraction_operations : (3 / 7 : ℚ) / 4 * (1 / 2) = 3 / 56 := by sorry

end fraction_operations_l1242_124210


namespace trapezoid_area_l1242_124264

/-- The area of a trapezoid with height 2a, one base 5a, and the other base 4a, is 9a² -/
theorem trapezoid_area (a : ℝ) : 
  let height : ℝ := 2 * a
  let base1 : ℝ := 5 * a
  let base2 : ℝ := 4 * a
  let area : ℝ := (height * (base1 + base2)) / 2
  area = 9 * a^2 := by sorry

end trapezoid_area_l1242_124264


namespace quadratic_form_equivalence_l1242_124260

theorem quadratic_form_equivalence :
  ∀ x y : ℝ, y = x^2 - 4*x + 5 ↔ y = (x - 2)^2 + 1 :=
by
  sorry

end quadratic_form_equivalence_l1242_124260


namespace area_inequality_special_quadrilateral_l1242_124276

/-- A point in a 2D plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral := (A B C D : Point)

/-- Check if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Calculate the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Check if a point is inside a quadrilateral -/
def isInside (p : Point) (q : Quadrilateral) : Prop := sorry

/-- Check if a point is on a line segment between two other points -/
def isOnSegment (p : Point) (a b : Point) : Prop := sorry

/-- Check if four points form a parallelogram -/
def isParallelogram (a b c d : Point) : Prop := sorry

/-- Theorem: Area inequality for quadrilaterals with special interior point -/
theorem area_inequality_special_quadrilateral 
  (ABCD : Quadrilateral) 
  (O K L M N : Point) 
  (h_convex : isConvex ABCD)
  (h_inside : isInside O ABCD)
  (h_K : isOnSegment K ABCD.A ABCD.B)
  (h_L : isOnSegment L ABCD.B ABCD.C)
  (h_M : isOnSegment M ABCD.C ABCD.D)
  (h_N : isOnSegment N ABCD.D ABCD.A)
  (h_OKBL : isParallelogram O K ABCD.B L)
  (h_OMDN : isParallelogram O M ABCD.D N)
  (S := area ABCD)
  (S1 := area (Quadrilateral.mk O N ABCD.A K))
  (S2 := area (Quadrilateral.mk O L ABCD.C M)) :
  Real.sqrt S ≥ Real.sqrt S1 + Real.sqrt S2 := by
  sorry

end area_inequality_special_quadrilateral_l1242_124276


namespace savings_difference_l1242_124259

def original_value : ℝ := 20000

def discount_scheme_1 (x : ℝ) : ℝ :=
  x * (1 - 0.3) * (1 - 0.1) - 800

def discount_scheme_2 (x : ℝ) : ℝ :=
  x * (1 - 0.25) * (1 - 0.2) - 1000

theorem savings_difference :
  discount_scheme_1 original_value - discount_scheme_2 original_value = 800 := by
  sorry

end savings_difference_l1242_124259


namespace tangent_ellipse_d_value_l1242_124219

/-- An ellipse in the first quadrant tangent to both x-axis and y-axis with foci at (3,7) and (d,7) -/
structure TangentEllipse where
  d : ℝ
  focus1 : ℝ × ℝ := (3, 7)
  focus2 : ℝ × ℝ := (d, 7)
  in_first_quadrant : d > 3
  tangent_to_axes : True  -- This is a simplification, as we can't directly represent tangency in this structure

/-- The value of d for the given ellipse is 49/3 -/
theorem tangent_ellipse_d_value (e : TangentEllipse) : e.d = 49/3 := by
  sorry

end tangent_ellipse_d_value_l1242_124219


namespace min_operations_to_check_square_l1242_124274

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define an operation (either measurement or comparison)
inductive Operation
  | Measure : Point → Point → Operation
  | Compare : ℝ → ℝ → Operation

-- Define a function to check if a quadrilateral is a square
def isSquare (q : Quadrilateral) : Prop := sorry

-- Define a function that returns the list of operations needed to check if a quadrilateral is a square
def operationsToCheckSquare (q : Quadrilateral) : List Operation := sorry

-- Theorem statement
theorem min_operations_to_check_square (q : Quadrilateral) :
  (isSquare q ↔ operationsToCheckSquare q = [
    Operation.Measure q.A q.B,
    Operation.Measure q.B q.C,
    Operation.Measure q.C q.D,
    Operation.Measure q.D q.A,
    Operation.Measure q.A q.C,
    Operation.Measure q.B q.D,
    Operation.Compare (q.A.x - q.B.x) (q.B.x - q.C.x),
    Operation.Compare (q.B.x - q.C.x) (q.C.x - q.D.x),
    Operation.Compare (q.C.x - q.D.x) (q.D.x - q.A.x),
    Operation.Compare (q.A.x - q.C.x) (q.B.x - q.D.x)
  ]) :=
sorry

end min_operations_to_check_square_l1242_124274


namespace expression_evaluation_l1242_124215

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := 1
  3 * x^2 + 2 * x * y - 4 * y^2 - 2 * (3 * y^2 + x * y - x^2) = -35/4 :=
by sorry

end expression_evaluation_l1242_124215


namespace equation_roots_l1242_124224

theorem equation_roots : 
  {x : ℝ | (x + 1) * (x - 2) = x + 1} = {-1, 3} := by sorry

end equation_roots_l1242_124224


namespace f_has_unique_zero_a_lower_bound_l1242_124275

noncomputable section

def f (x : ℝ) : ℝ := -1/2 * Real.log x + 2/(x+1)

theorem f_has_unique_zero :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

theorem a_lower_bound (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (1/Real.exp 1) 1 →
    ∀ t : ℝ, t ∈ Set.Icc (1/2) 2 →
      f x ≥ t^3 - t^2 - 2*a*t + 2) →
  a ≥ 5/4 :=
sorry

end f_has_unique_zero_a_lower_bound_l1242_124275


namespace f_3_equals_130_l1242_124290

def f (x : ℝ) : ℝ := 3 * x^4 - 5 * x^3 + 2 * x^2 - x + 7

theorem f_3_equals_130 : f 3 = 130 := by
  sorry

end f_3_equals_130_l1242_124290


namespace complex_fraction_sum_l1242_124295

theorem complex_fraction_sum (A B : ℝ) : 
  (Complex.I : ℂ) * (3 + Complex.I) = (1 + 2 * Complex.I) * (A + B * Complex.I) → 
  A + B = 0 := by
  sorry

end complex_fraction_sum_l1242_124295


namespace equation_solution_l1242_124271

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  x^2 + 36 / x^2 = 13 ↔ x = 2 ∨ x = -2 ∨ x = 3 ∨ x = -3 := by sorry

end equation_solution_l1242_124271


namespace connie_markers_count_l1242_124298

/-- The number of red markers Connie has -/
def red_markers : ℕ := 41

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 64

/-- The total number of markers Connie has -/
def total_markers : ℕ := red_markers + blue_markers

theorem connie_markers_count : total_markers = 105 := by
  sorry

end connie_markers_count_l1242_124298


namespace inscribed_circle_radius_l1242_124257

/-- The radius of the inscribed circle of a triangle with side lengths 8, 10, and 12 is √7 -/
theorem inscribed_circle_radius (a b c : ℝ) (h_a : a = 8) (h_b : b = 10) (h_c : c = 12) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = Real.sqrt 7 := by sorry

end inscribed_circle_radius_l1242_124257


namespace arithmetic_mean_of_fractions_l1242_124252

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((x + a) / x + (x - a) / x) = 1 := by
  sorry

end arithmetic_mean_of_fractions_l1242_124252


namespace bob_shucking_rate_is_two_bob_shucking_rate_consistent_with_two_hours_l1242_124220

/-- Represents the rate at which Bob shucks oysters in oysters per minute -/
def bob_shucking_rate : ℚ :=
  10 / 5

theorem bob_shucking_rate_is_two :
  bob_shucking_rate = 2 :=
by
  -- Proof goes here
  sorry

theorem bob_shucking_rate_consistent_with_two_hours :
  bob_shucking_rate * 120 = 240 :=
by
  -- Proof goes here
  sorry

end bob_shucking_rate_is_two_bob_shucking_rate_consistent_with_two_hours_l1242_124220


namespace ellipse_foci_l1242_124234

/-- The foci of the ellipse x^2/6 + y^2/9 = 1 are at (0, √3) and (0, -√3) -/
theorem ellipse_foci (x y : ℝ) : 
  (x^2 / 6 + y^2 / 9 = 1) → 
  (∃ (f₁ f₂ : ℝ × ℝ), 
    f₁ = (0, Real.sqrt 3) ∧ 
    f₂ = (0, -Real.sqrt 3) ∧ 
    (∀ (p : ℝ × ℝ), p.1^2 / 6 + p.2^2 / 9 = 1 → 
      (Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
       Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 2 * 3))) :=
by sorry


end ellipse_foci_l1242_124234


namespace decreasing_linear_function_condition_l1242_124226

/-- A linear function y = (m-3)x + 5 where y decreases as x increases -/
def decreasingLinearFunction (m : ℝ) : ℝ → ℝ := fun x ↦ (m - 3) * x + 5

/-- Theorem: If y decreases as x increases for the linear function y = (m-3)x + 5, then m < 3 -/
theorem decreasing_linear_function_condition (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → decreasingLinearFunction m x₁ > decreasingLinearFunction m x₂) →
  m < 3 := by
  sorry

end decreasing_linear_function_condition_l1242_124226


namespace quadratic_equation_equal_roots_l1242_124280

theorem quadratic_equation_equal_roots (a : ℝ) :
  (∃ x : ℝ, (3 * a - 1) * x^2 - a * x + 1/4 = 0 ∧
   ∀ y : ℝ, (3 * a - 1) * y^2 - a * y + 1/4 = 0 → y = x) →
  a^2 - 2 * a + 2021 + 1/a = 2023 := by
sorry

end quadratic_equation_equal_roots_l1242_124280


namespace min_distinct_values_l1242_124272

theorem min_distinct_values (total : ℕ) (mode_count : ℕ) (h1 : total = 2023) (h2 : mode_count = 15) :
  ∃ (distinct : ℕ), distinct = 145 ∧ 
  (∀ d : ℕ, d < 145 → 
    ¬∃ (l : List ℕ), l.length = total ∧ 
    (∃! x : ℕ, x ∈ l ∧ l.count x = mode_count) ∧
    l.toFinset.card = d) :=
by sorry

end min_distinct_values_l1242_124272


namespace graph_passes_through_quadrants_l1242_124212

-- Define the function
def f (x : ℝ) : ℝ := -3 * x + 1

-- Theorem statement
theorem graph_passes_through_quadrants :
  (∃ x y, x > 0 ∧ y > 0 ∧ f x = y) ∧  -- First quadrant
  (∃ x y, x < 0 ∧ y > 0 ∧ f x = y) ∧  -- Second quadrant
  (∃ x y, x > 0 ∧ y < 0 ∧ f x = y) :=  -- Fourth quadrant
by sorry

end graph_passes_through_quadrants_l1242_124212


namespace combined_annual_income_l1242_124201

-- Define the monthly incomes as real numbers
variable (A_income B_income C_income D_income : ℝ)

-- Define the conditions
def income_ratio : Prop :=
  A_income / C_income = 5 / 3 ∧ B_income / C_income = 2 / 3 ∧ D_income / C_income = 4 / 3

def B_income_relation : Prop :=
  B_income = 1.12 * C_income

def D_income_relation : Prop :=
  D_income = 0.85 * A_income

def C_income_value : Prop :=
  C_income = 15000

-- Define the theorem
theorem combined_annual_income
  (h1 : income_ratio A_income B_income C_income D_income)
  (h2 : B_income_relation B_income C_income)
  (h3 : D_income_relation A_income D_income)
  (h4 : C_income_value C_income) :
  (A_income + B_income + C_income + D_income) * 12 = 936600 :=
by sorry

end combined_annual_income_l1242_124201


namespace series_sum_equals_half_l1242_124221

/-- The sum of the series Σ(3^(2^k) / (9^(2^k) - 1)) for k from 0 to infinity is equal to 1/2 -/
theorem series_sum_equals_half :
  (∑' k : ℕ, (3 ^ (2 ^ k)) / ((9 ^ (2 ^ k)) - 1)) = 1 / 2 := by
  sorry

end series_sum_equals_half_l1242_124221


namespace power_mod_six_l1242_124233

theorem power_mod_six : 5^2013 % 6 = 5 := by
  sorry

end power_mod_six_l1242_124233


namespace trajectory_and_angle_property_l1242_124218

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ -2

-- Define the condition for angle equality
def angle_equality (t x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ / (x₁ - t)) + (y₂ / (x₂ - t)) = 0

-- Theorem statement
theorem trajectory_and_angle_property :
  ∃ (C : ℝ → ℝ → Prop),
    (∀ x y, C x y ↔ trajectory_C x y) ∧
    (∃ t : ℝ, t = 4 ∧
      ∀ k x₁ y₁ x₂ y₂,
        C x₁ y₁ ∧ C x₂ y₂ ∧
        y₁ = k * (x₁ - 1) ∧ y₂ = k * (x₂ - 1) →
        angle_equality t x₁ y₁ x₂ y₂) :=
sorry

end trajectory_and_angle_property_l1242_124218


namespace compound_nitrogen_percentage_l1242_124209

/-- Mass percentage of nitrogen in a compound -/
def mass_percentage_N : ℝ := 26.42

/-- Theorem stating the mass percentage of nitrogen in the compound -/
theorem compound_nitrogen_percentage : mass_percentage_N = 26.42 := by
  sorry

end compound_nitrogen_percentage_l1242_124209


namespace not_recurring_decimal_example_l1242_124211

def is_recurring_decimal (x : ℝ) : Prop :=
  ∃ (a b : ℕ) (c : ℕ+), x = (a : ℝ) / b + (c : ℝ) / (10^b * 9)

theorem not_recurring_decimal_example : ¬ is_recurring_decimal 0.89898989 := by
  sorry

end not_recurring_decimal_example_l1242_124211


namespace claire_profit_is_60_l1242_124225

def claire_profit (total_loaves : ℕ) (morning_price afternoon_price late_price cost_per_loaf fixed_cost : ℚ) : ℚ :=
  let morning_sales := total_loaves / 3
  let afternoon_sales := (total_loaves - morning_sales) / 2
  let late_sales := total_loaves - morning_sales - afternoon_sales
  let total_revenue := morning_sales * morning_price + afternoon_sales * afternoon_price + late_sales * late_price
  let total_cost := total_loaves * cost_per_loaf + fixed_cost
  total_revenue - total_cost

theorem claire_profit_is_60 :
  claire_profit 60 3 2 (3/2) 1 10 = 60 := by
  sorry

end claire_profit_is_60_l1242_124225


namespace geometric_sequence_first_term_l1242_124266

theorem geometric_sequence_first_term (a r : ℝ) : 
  (a * r^2 = 3) → (a * r^4 = 27) → (a = Real.sqrt 9 ∨ a = -Real.sqrt 9) := by
  sorry

end geometric_sequence_first_term_l1242_124266


namespace lemonade_price_ratio_l1242_124243

theorem lemonade_price_ratio :
  -- Define the ratio of small cups sold
  let small_ratio : ℚ := 3/5
  -- Define the ratio of large cups sold
  let large_ratio : ℚ := 1 - small_ratio
  -- Define the fraction of revenue from large cups
  let large_revenue_fraction : ℚ := 357142857142857150 / 1000000000000000000
  -- Define the price ratio of large to small cups
  let price_ratio : ℚ := large_revenue_fraction * (1 / large_ratio)
  -- The theorem
  price_ratio = 892857142857143 / 1000000000000000 :=
by sorry

end lemonade_price_ratio_l1242_124243


namespace tangent_line_implies_sum_l1242_124207

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that the tangent line at (1, f(1)) has equation x - 2y + 1 = 0
def has_tangent_line (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), (∀ x, m * x + b = f x) ∧ (m * 1 + b = f 1) ∧ (m = 1 / 2) ∧ (b = 1 / 2)

-- Theorem statement
theorem tangent_line_implies_sum (f : ℝ → ℝ) (h : has_tangent_line f) :
  f 1 + 2 * (deriv f 1) = 2 :=
sorry

end tangent_line_implies_sum_l1242_124207


namespace points_four_units_from_negative_two_l1242_124239

theorem points_four_units_from_negative_two :
  ∀ x : ℝ, |x - (-2)| = 4 ↔ x = 2 ∨ x = -6 := by sorry

end points_four_units_from_negative_two_l1242_124239


namespace f_properties_l1242_124262

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 3|

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f x < 7 ↔ -2 < x ∧ x < 5) ∧
  (∀ x : ℝ, f x - |2*x - 7| < x^2 - 2*x + Real.sqrt 26) := by
sorry

end f_properties_l1242_124262


namespace original_recipe_yield_l1242_124293

/-- Represents a cookie recipe -/
structure Recipe where
  butter : ℝ
  cookies : ℝ

/-- Proves that given a recipe that uses 4 pounds of butter, 
    if 1 pound of butter makes 4 dozen cookies, 
    then the original recipe makes 16 dozen cookies. -/
theorem original_recipe_yield 
  (original : Recipe) 
  (h1 : original.butter = 4) 
  (h2 : ∃ (scaled : Recipe), scaled.butter = 1 ∧ scaled.cookies = 4) : 
  original.cookies = 16 := by
sorry

end original_recipe_yield_l1242_124293


namespace middle_school_students_l1242_124238

theorem middle_school_students (band_percentage : ℝ) (band_students : ℕ) 
  (h1 : band_percentage = 0.20)
  (h2 : band_students = 168) : 
  ℕ := by
  sorry

end middle_school_students_l1242_124238


namespace range_of_p_l1242_124284

def h (x : ℝ) : ℝ := 2 * x + 1

def p (x : ℝ) : ℝ := h (h (h x))

theorem range_of_p :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → -1 ≤ p x ∧ p x ≤ 31 :=
by
  sorry

end range_of_p_l1242_124284


namespace ninth_grade_class_problem_l1242_124297

theorem ninth_grade_class_problem (total : ℕ) (math : ℕ) (foreign : ℕ) (science_only : ℕ) (math_and_foreign : ℕ) :
  total = 120 →
  math = 85 →
  foreign = 75 →
  science_only = 20 →
  math_and_foreign = 40 →
  ∃ (math_only : ℕ), math_only = 45 ∧ math_only = math - math_and_foreign :=
by sorry

end ninth_grade_class_problem_l1242_124297


namespace friday_return_count_l1242_124202

/-- The number of books returned on Friday -/
def books_returned_friday (initial_books : ℕ) (wed_checkout : ℕ) (thur_return : ℕ) (thur_checkout : ℕ) (final_books : ℕ) : ℕ :=
  final_books - (initial_books - wed_checkout + thur_return - thur_checkout)

/-- Proof that 7 books were returned on Friday given the conditions -/
theorem friday_return_count :
  books_returned_friday 98 43 23 5 80 = 7 := by
  sorry

#eval books_returned_friday 98 43 23 5 80

end friday_return_count_l1242_124202


namespace tangent_count_depends_on_position_l1242_124286

/-- Represents the position of a point relative to a circle -/
inductive PointPosition
  | OnCircle
  | OutsideCircle
  | InsideCircle

/-- Represents the number of tangents that can be drawn -/
inductive TangentCount
  | Zero
  | One
  | Two

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines the position of a point relative to a circle -/
def pointPosition (c : Circle) (p : ℝ × ℝ) : PointPosition :=
  sorry

/-- Counts the number of tangents that can be drawn from a point to a circle -/
def tangentCount (c : Circle) (p : ℝ × ℝ) : TangentCount :=
  sorry

/-- Theorem: The number of tangents depends on the point's position relative to the circle -/
theorem tangent_count_depends_on_position (c : Circle) (p : ℝ × ℝ) :
  (pointPosition c p = PointPosition.OnCircle → tangentCount c p = TangentCount.One) ∧
  (pointPosition c p = PointPosition.OutsideCircle → tangentCount c p = TangentCount.Two) ∧
  (pointPosition c p = PointPosition.InsideCircle → tangentCount c p = TangentCount.Zero) :=
  sorry

end tangent_count_depends_on_position_l1242_124286


namespace quadratic_real_root_range_l1242_124254

theorem quadratic_real_root_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 4 = 0) ∨ 
  (∃ x : ℝ, x^2 + (a-2)*x + 4 = 0) ∨ 
  (∃ x : ℝ, x^2 + 2*a*x + a^2 + 1 = 0) ↔ 
  a ≥ 4 ∨ a ≤ -2 := by sorry

end quadratic_real_root_range_l1242_124254


namespace initial_number_exists_l1242_124240

theorem initial_number_exists : ∃ N : ℝ, ∃ k : ℤ, N + 69.00000000008731 = 330 * (k : ℝ) := by
  sorry

end initial_number_exists_l1242_124240


namespace cube_sum_reciprocal_l1242_124205

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 := by
  sorry

end cube_sum_reciprocal_l1242_124205


namespace only_minute_hand_rotates_l1242_124241

-- Define the set of objects
inductive Object
  | MinuteHand
  | Boat
  | Car

-- Define the motion types
inductive Motion
  | Rotation
  | Translation
  | Combined

-- Function to determine the motion type of an object
def motionType (obj : Object) : Motion :=
  match obj with
  | Object.MinuteHand => Motion.Rotation
  | Object.Boat => Motion.Combined
  | Object.Car => Motion.Combined

-- Theorem statement
theorem only_minute_hand_rotates :
  ∀ (obj : Object), motionType obj = Motion.Rotation ↔ obj = Object.MinuteHand :=
by sorry

end only_minute_hand_rotates_l1242_124241


namespace molecular_weight_AlPO4_l1242_124214

/-- The atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- The atomic weight of Phosphorus in g/mol -/
def P_weight : ℝ := 30.97

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The number of Oxygen atoms in AlPO4 -/
def O_count : ℕ := 4

/-- The molecular weight of AlPO4 in g/mol -/
def AlPO4_weight : ℝ := Al_weight + P_weight + O_count * O_weight

/-- The number of moles of AlPO4 -/
def moles : ℕ := 4

/-- Theorem stating the molecular weight of 4 moles of AlPO4 -/
theorem molecular_weight_AlPO4 : moles * AlPO4_weight = 487.80 := by
  sorry

end molecular_weight_AlPO4_l1242_124214


namespace lcm_hcf_problem_l1242_124292

theorem lcm_hcf_problem (A B : ℕ) (h1 : A = 330) (h2 : Nat.lcm A B = 2310) (h3 : Nat.gcd A B = 30) :
  B = 210 := by
  sorry

end lcm_hcf_problem_l1242_124292


namespace ten_people_handshakes_l1242_124217

/-- The number of handshakes in a group where each person shakes hands only with lighter people -/
def handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- Proof that in a group of 10 people with distinct weights, where each person shakes hands
    only with those lighter than themselves, the total number of handshakes is 45 -/
theorem ten_people_handshakes :
  handshakes 9 = 45 := by
  sorry

#eval handshakes 9  -- Should output 45

end ten_people_handshakes_l1242_124217


namespace parenthesized_results_l1242_124299

def original_expression : ℚ := 72 / 9 - 3 * 2

def parenthesized_expressions : List ℚ := [
  (72 / 9 - 3) * 2,
  72 / (9 - 3) * 2,
  72 / ((9 - 3) * 2)
]

theorem parenthesized_results :
  original_expression = 2 →
  (parenthesized_expressions.toFinset = {6, 10, 24}) ∧
  (parenthesized_expressions.length = 3) :=
by sorry

end parenthesized_results_l1242_124299


namespace min_value_reciprocal_sum_l1242_124242

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (1/a + 1/b) ≥ 3 + 2 * Real.sqrt 2 := by
sorry

end min_value_reciprocal_sum_l1242_124242


namespace cityF_greatest_increase_l1242_124250

/-- Represents a city with population data for 1970 and 1980 --/
structure City where
  name : String
  pop1970 : Nat
  pop1980 : Nat

/-- Calculates the percentage increase in population from 1970 to 1980 --/
def percentageIncrease (city : City) : Rat :=
  (city.pop1980 - city.pop1970 : Rat) / city.pop1970 * 100

/-- The set of cities in the region --/
def cities : Finset City := sorry

/-- City F with its population data --/
def cityF : City := { name := "F", pop1970 := 30000, pop1980 := 45000 }

/-- City G with its population data --/
def cityG : City := { name := "G", pop1970 := 60000, pop1980 := 75000 }

/-- Combined City H (including I) with its population data --/
def cityH : City := { name := "H", pop1970 := 60000, pop1980 := 70000 }

/-- City J with its population data --/
def cityJ : City := { name := "J", pop1970 := 90000, pop1980 := 120000 }

/-- Theorem stating that City F had the greatest percentage increase --/
theorem cityF_greatest_increase : 
  ∀ city ∈ cities, percentageIncrease cityF ≥ percentageIncrease city :=
sorry

end cityF_greatest_increase_l1242_124250


namespace train_speed_l1242_124291

/-- Given a train of length 125 metres that takes 7.5 seconds to pass a pole, 
    its speed is 60 km/hr. -/
theorem train_speed (train_length : Real) (time_to_pass : Real) 
  (h1 : train_length = 125) 
  (h2 : time_to_pass = 7.5) : 
  (train_length / time_to_pass) * 3.6 = 60 := by
  sorry

end train_speed_l1242_124291


namespace royal_family_children_l1242_124230

/-- Represents the number of years that have passed -/
def n : ℕ := sorry

/-- Represents the number of daughters -/
def d : ℕ := sorry

/-- The initial age of the king and queen -/
def initial_parent_age : ℕ := 35

/-- The initial total age of the children -/
def initial_children_age : ℕ := 35

/-- The number of sons -/
def num_sons : ℕ := 3

/-- The maximum allowed number of children -/
def max_children : ℕ := 20

theorem royal_family_children :
  (initial_parent_age * 2 + 2 * n = initial_children_age + (d + num_sons) * n) ∧
  (d + num_sons ≤ max_children) →
  (d + num_sons = 7) ∨ (d + num_sons = 9) := by
  sorry

end royal_family_children_l1242_124230


namespace slower_ball_speed_l1242_124232

/-- Two balls moving on a circular path with the following properties:
    - When moving in the same direction, they meet every 20 seconds
    - When moving in opposite directions, they meet every 4 seconds
    - When moving towards each other, the distance between them decreases by 75 cm every 3 seconds
    Prove that the speed of the slower ball is 10 cm/s -/
theorem slower_ball_speed (v u : ℝ) (C : ℝ) : 
  (20 * (v - u) = C) →  -- Same direction meeting condition
  (4 * (v + u) = C) →   -- Opposite direction meeting condition
  ((v + u) * 3 = 75) →  -- Approaching speed condition
  (u = 10) :=           -- Speed of slower ball
by sorry

end slower_ball_speed_l1242_124232


namespace fraction_sum_equality_l1242_124246

theorem fraction_sum_equality (a b c : ℝ) (hc : c ≠ 0) :
  (a + b) / c = a / c + b / c := by
  sorry

end fraction_sum_equality_l1242_124246


namespace combined_final_selling_price_is_630_45_l1242_124237

/-- Calculate the final selling price for an item given its cost price, profit percentage, and tax or discount percentage -/
def finalSellingPrice (costPrice : ℝ) (profitPercentage : ℝ) (taxOrDiscountPercentage : ℝ) (isTax : Bool) : ℝ :=
  let sellingPriceBeforeTaxOrDiscount := costPrice * (1 + profitPercentage)
  if isTax then
    sellingPriceBeforeTaxOrDiscount * (1 + taxOrDiscountPercentage)
  else
    sellingPriceBeforeTaxOrDiscount * (1 - taxOrDiscountPercentage)

/-- The combined final selling price for all three items -/
def combinedFinalSellingPrice : ℝ :=
  finalSellingPrice 180 0.15 0.05 true +
  finalSellingPrice 220 0.20 0.10 false +
  finalSellingPrice 130 0.25 0.08 true

theorem combined_final_selling_price_is_630_45 :
  combinedFinalSellingPrice = 630.45 := by sorry

end combined_final_selling_price_is_630_45_l1242_124237


namespace quadratic_root_problem_l1242_124278

theorem quadratic_root_problem (a : ℝ) (k : ℝ) :
  (∃ x : ℂ, x^2 + 4*x + k = 0 ∧ x = a + 3*Complex.I) →
  k = 13 := by
  sorry

end quadratic_root_problem_l1242_124278


namespace axis_of_symmetry_parabola_axis_of_symmetry_specific_parabola_l1242_124235

/-- The axis of symmetry of a parabola y = ax² + bx + c is x = -b / (2a) -/
theorem axis_of_symmetry_parabola (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∀ x, f ((-b / (2 * a)) + x) = f ((-b / (2 * a)) - x)) := by sorry

/-- The axis of symmetry of the parabola y = -3x² + 6x - 1 is the line x = 1 -/
theorem axis_of_symmetry_specific_parabola :
  let f : ℝ → ℝ := λ x ↦ -3 * x^2 + 6 * x - 1
  (∀ x, f (1 + x) = f (1 - x)) := by sorry

end axis_of_symmetry_parabola_axis_of_symmetry_specific_parabola_l1242_124235


namespace five_cuts_sixteen_pieces_l1242_124200

/-- The number of pieces obtained by cutting a cake n times -/
def cakePieces (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem: The number of pieces obtained by cutting a cake 5 times is 16 -/
theorem five_cuts_sixteen_pieces : cakePieces 5 = 16 := by
  sorry

#eval cakePieces 5  -- This will evaluate to 16

end five_cuts_sixteen_pieces_l1242_124200


namespace gcd_153_119_l1242_124203

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end gcd_153_119_l1242_124203


namespace largest_divisor_of_expression_l1242_124267

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (10*x + 2) * (10*x + 6)^2 * (5*x + 1) = 24 * k) ∧
  (∀ (m : ℤ), m > 24 → ∃ (y : ℤ), Odd y ∧ ¬(∃ (l : ℤ), (10*y + 2) * (10*y + 6)^2 * (5*y + 1) = m * l)) :=
sorry

end largest_divisor_of_expression_l1242_124267


namespace factorial_fraction_equals_one_l1242_124256

theorem factorial_fraction_equals_one : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end factorial_fraction_equals_one_l1242_124256


namespace total_votes_l1242_124244

/-- Given that Ben and Matt received votes in the ratio 2:3 and Ben got 24 votes,
    prove that the total number of votes cast is 60. -/
theorem total_votes (ben_votes : ℕ) (matt_votes : ℕ) : 
  ben_votes = 24 → 
  ben_votes * 3 = matt_votes * 2 → 
  ben_votes + matt_votes = 60 := by
sorry

end total_votes_l1242_124244


namespace paper_distribution_l1242_124294

theorem paper_distribution (total_sheets : ℕ) (num_printers : ℕ) 
  (h1 : total_sheets = 221) (h2 : num_printers = 31) :
  (total_sheets / num_printers : ℕ) = 7 := by
  sorry

end paper_distribution_l1242_124294


namespace parabola_p_value_l1242_124283

/-- Represents a parabola with equation y^2 = 2px and directrix x = -2 -/
structure Parabola where
  p : ℝ
  eq : ∀ x y : ℝ, y^2 = 2 * p * x
  directrix : ∀ x : ℝ, x = -2

/-- The value of p for the given parabola is 4 -/
theorem parabola_p_value (par : Parabola) : par.p = 4 := by
  sorry

end parabola_p_value_l1242_124283


namespace sum_integer_chord_lengths_equals_40_l1242_124268

/-- A circle with center O and a point P inside it. -/
structure CircleWithPoint where
  O : Point    -- Center of the circle
  P : Point    -- Point inside the circle
  radius : ℝ   -- Radius of the circle
  OP : ℝ       -- Distance between O and P

/-- The sum of all possible integer chord lengths passing through P -/
def sumIntegerChordLengths (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem -/
theorem sum_integer_chord_lengths_equals_40 (c : CircleWithPoint) 
  (h_radius : c.radius = 5)
  (h_OP : c.OP = 4) :
  sumIntegerChordLengths c = 40 := by
  sorry

end sum_integer_chord_lengths_equals_40_l1242_124268


namespace repaired_shoes_duration_is_one_year_l1242_124255

/-- The duration for which the repaired shoes last, in years -/
def repaired_shoes_duration : ℝ := 1

/-- The cost to repair the used shoes, in dollars -/
def repair_cost : ℝ := 10.50

/-- The cost of new shoes, in dollars -/
def new_shoes_cost : ℝ := 30.00

/-- The duration for which new shoes last, in years -/
def new_shoes_duration : ℝ := 2

/-- The percentage by which the average cost per year of new shoes 
    is greater than the cost of repairing used shoes -/
def cost_difference_percentage : ℝ := 42.857142857142854

theorem repaired_shoes_duration_is_one_year :
  repaired_shoes_duration = 
    (repair_cost * (1 + cost_difference_percentage / 100)) / 
    (new_shoes_cost / new_shoes_duration) :=
by sorry

end repaired_shoes_duration_is_one_year_l1242_124255


namespace valid_outfit_count_l1242_124279

/-- Represents the number of shirts, pants, and hats -/
def total_items : ℕ := 8

/-- Represents the number of colors for each item -/
def total_colors : ℕ := 8

/-- Represents the number of colors with matching sets -/
def matching_colors : ℕ := 6

/-- Calculates the total number of outfit combinations -/
def total_combinations : ℕ := total_items * total_items * total_items

/-- Calculates the number of restricted combinations for one pair of matching items -/
def restricted_per_pair : ℕ := matching_colors * total_items

/-- Calculates the total number of restricted combinations -/
def total_restricted : ℕ := 3 * restricted_per_pair

/-- Represents the number of valid outfit choices -/
def valid_outfits : ℕ := total_combinations - total_restricted

theorem valid_outfit_count : valid_outfits = 368 := by
  sorry

end valid_outfit_count_l1242_124279


namespace system_solution_l1242_124249

theorem system_solution (a : ℝ) (h : a > 0) :
  ∃ (x y : ℝ), 
    (a^(7*x) * a^(15*y) = (a^19)^(1/2)) ∧ 
    ((a^(25*y))^(1/3) / (a^(13*x))^(1/2) = a^(1/12)) ∧
    x = 1/2 ∧ y = 2/5 := by
  sorry

end system_solution_l1242_124249


namespace two_integer_tangent_lengths_l1242_124285

def circle_circumference : ℝ := 10

def is_valid_arc_length (x : ℝ) : Prop :=
  0 < x ∧ x < circle_circumference

theorem two_integer_tangent_lengths :
  ∃ (t₁ t₂ : ℕ), t₁ ≠ t₂ ∧
  (∀ m : ℕ, is_valid_arc_length m →
    (∃ n : ℝ, is_valid_arc_length n ∧
      m + n = circle_circumference ∧
      (t₁ : ℝ)^2 = m * n ∨ (t₂ : ℝ)^2 = m * n)) ∧
  (∀ t : ℕ, (∃ m : ℕ, is_valid_arc_length m ∧
    (∃ n : ℝ, is_valid_arc_length n ∧
      m + n = circle_circumference ∧
      (t : ℝ)^2 = m * n)) →
    t = t₁ ∨ t = t₂) :=
by sorry

end two_integer_tangent_lengths_l1242_124285


namespace bicycle_discount_proof_l1242_124273

theorem bicycle_discount_proof (original_price : ℝ) 
  (discount1 discount2 discount3 : ℝ) : 
  original_price = 200 →
  discount1 = 0.60 →
  discount2 = 0.20 →
  discount3 = 0.10 →
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 57.60 := by
  sorry

end bicycle_discount_proof_l1242_124273


namespace smaller_circles_radius_l1242_124277

/-- Given a central circle of radius 2 and 4 identical smaller circles
    touching the central circle and each other, the radius of each smaller circle is 6. -/
theorem smaller_circles_radius (r : ℝ) : r = 6 :=
  by
  -- Define the relationship between the radii
  have h1 : (2 + r)^2 + (2 + r)^2 = (2*r)^2 :=
    sorry
  -- Solve the resulting equation
  have h2 : r^2 - 4*r - 4 = 0 :=
    sorry
  -- Apply the quadratic formula and choose the positive solution
  sorry

end smaller_circles_radius_l1242_124277


namespace additional_cars_needed_min_additional_cars_l1242_124251

def current_cars : ℕ := 35
def cars_per_row : ℕ := 8

theorem additional_cars_needed : 
  ∃ (n : ℕ), n > 0 ∧ (current_cars + n) % cars_per_row = 0 ∧
  ∀ (m : ℕ), m < n → (current_cars + m) % cars_per_row ≠ 0 := by
  sorry

theorem min_additional_cars : 
  ∃ (n : ℕ), n = 5 ∧ (current_cars + n) % cars_per_row = 0 ∧
  ∀ (m : ℕ), m < n → (current_cars + m) % cars_per_row ≠ 0 := by
  sorry

end additional_cars_needed_min_additional_cars_l1242_124251


namespace vector_operation_l1242_124245

/-- Given plane vectors a and b, prove that -2a - b equals (-3, -1) -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (1, -1)) :
  (-2 : ℝ) • a - b = (-3, -1) := by sorry

end vector_operation_l1242_124245


namespace cereal_eating_time_l1242_124208

theorem cereal_eating_time 
  (fat_rate : ℚ) 
  (thin_rate : ℚ) 
  (total_cereal : ℚ) 
  (h1 : fat_rate = 1 / 15) 
  (h2 : thin_rate = 1 / 40) 
  (h3 : total_cereal = 5) : 
  total_cereal / (fat_rate + thin_rate) = 600 / 11 := by
  sorry

end cereal_eating_time_l1242_124208


namespace geometric_sequence_problem_l1242_124282

theorem geometric_sequence_problem (x : ℝ) : 
  x > 0 → 
  (∃ r : ℝ, r > 0 ∧ x = 40 * r ∧ (10/3) = x * r) → 
  x = (20 * Real.sqrt 3) / 3 := by
  sorry

end geometric_sequence_problem_l1242_124282


namespace sams_book_count_l1242_124227

/-- The total number of books Sam bought --/
def total_books (a m c f s : ℝ) : ℝ := a + m + c + f + s

/-- Theorem stating the total number of books Sam bought --/
theorem sams_book_count :
  ∀ (a m c f s : ℝ),
    a = 13.0 →
    m = 17.0 →
    c = 15.0 →
    f = 10.0 →
    s = 2 * a →
    total_books a m c f s = 81.0 := by
  sorry

end sams_book_count_l1242_124227


namespace sin_increasing_interval_l1242_124288

/-- The function f with given properties has (-π/12, 5π/12) as its strictly increasing interval -/
theorem sin_increasing_interval (ω : ℝ) (h_pos : ω > 0) :
  let f : ℝ → ℝ := λ x => Real.sin (ω * x + π / 6)
  (∀ x, f x > 0) →
  (∀ p, p > 0 → (∀ x, f (x + p) = f x) → p ≥ π) →
  (∃ p, p > 0 ∧ (∀ x, f (x + p) = f x) ∧ p = π) →
  (∀ x ∈ Set.Ioo (-π/12 : ℝ) (5*π/12), StrictMono f) :=
by
  sorry

end sin_increasing_interval_l1242_124288


namespace odot_inequality_equivalence_l1242_124258

-- Define the operation ⊙
def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

-- State the theorem
theorem odot_inequality_equivalence :
  ∀ x : ℝ, odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end odot_inequality_equivalence_l1242_124258


namespace certain_number_equation_l1242_124270

theorem certain_number_equation (x : ℝ) : 300 + 5 * x = 340 ↔ x = 8 := by sorry

end certain_number_equation_l1242_124270


namespace triangle_arctangent_sum_l1242_124228

/-- In a triangle ABC with sides a, b, c, arbitrary angle C, and positive real number k,
    under certain conditions, the sum of two specific arctangents equals π/4. -/
theorem triangle_arctangent_sum (a b c k : ℝ) (h1 : k > 0) : 
  ∃ (h : Set ℝ), h.Nonempty ∧ ∀ (x : ℝ), x ∈ h → 
    Real.arctan (a / (b + c + k)) + Real.arctan (b / (a + c + k)) = π / 4 := by
  sorry

end triangle_arctangent_sum_l1242_124228


namespace hyperbola_transformation_l1242_124287

/-- Given a hyperbola with equation x^2/4 - y^2/5 = 1, 
    prove that the standard equation of a hyperbola with the same foci as vertices 
    and perpendicular asymptotes is x^2/9 - y^2/9 = 1 -/
theorem hyperbola_transformation (x y : ℝ) : 
  (∃ (a b : ℝ), x^2/a - y^2/b = 1 ∧ a = 4 ∧ b = 5) →
  (∃ (c : ℝ), x^2/c - y^2/c = 1 ∧ c = 9) :=
by sorry

end hyperbola_transformation_l1242_124287
