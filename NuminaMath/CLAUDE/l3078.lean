import Mathlib

namespace NUMINAMATH_CALUDE_smallest_gcd_value_l3078_307887

theorem smallest_gcd_value (a b c d : ℕ) : 
  (∃ (gcd_list : List ℕ), 
    gcd_list.length = 6 ∧ 
    1 ∈ gcd_list ∧ 
    2 ∈ gcd_list ∧ 
    3 ∈ gcd_list ∧ 
    4 ∈ gcd_list ∧ 
    5 ∈ gcd_list ∧
    (∃ (N : ℕ), N > 5 ∧ N ∈ gcd_list) ∧
    (∀ (x : ℕ), x ∈ gcd_list → 
      x = Nat.gcd a b ∨ 
      x = Nat.gcd a c ∨ 
      x = Nat.gcd a d ∨ 
      x = Nat.gcd b c ∨ 
      x = Nat.gcd b d ∨ 
      x = Nat.gcd c d)) →
  (∀ (M : ℕ), M > 5 ∧ 
    (∃ (gcd_list : List ℕ), 
      gcd_list.length = 6 ∧ 
      1 ∈ gcd_list ∧ 
      2 ∈ gcd_list ∧ 
      3 ∈ gcd_list ∧ 
      4 ∈ gcd_list ∧ 
      5 ∈ gcd_list ∧
      M ∈ gcd_list ∧
      (∀ (x : ℕ), x ∈ gcd_list → 
        x = Nat.gcd a b ∨ 
        x = Nat.gcd a c ∨ 
        x = Nat.gcd a d ∨ 
        x = Nat.gcd b c ∨ 
        x = Nat.gcd b d ∨ 
        x = Nat.gcd c d)) →
    M ≥ 14) :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_value_l3078_307887


namespace NUMINAMATH_CALUDE_derivative_sin_plus_cos_at_pi_l3078_307859

/-- Given f(x) = sin(x) + cos(x), prove that f'(π) = -1 -/
theorem derivative_sin_plus_cos_at_pi :
  let f := λ x : ℝ => Real.sin x + Real.cos x
  (deriv f) π = -1 := by sorry

end NUMINAMATH_CALUDE_derivative_sin_plus_cos_at_pi_l3078_307859


namespace NUMINAMATH_CALUDE_cheryl_same_color_probability_l3078_307817

def total_marbles : ℕ := 9
def marbles_per_color : ℕ := 3
def colors : ℕ := 3
def marbles_taken_each : ℕ := 3

theorem cheryl_same_color_probability :
  let total_outcomes := (total_marbles.choose marbles_taken_each) *
                        ((total_marbles - marbles_taken_each).choose marbles_taken_each) *
                        ((total_marbles - 2 * marbles_taken_each).choose marbles_taken_each)
  let favorable_outcomes := colors * ((total_marbles - 2 * marbles_taken_each).choose marbles_taken_each)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 28 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_same_color_probability_l3078_307817


namespace NUMINAMATH_CALUDE_todd_snow_cone_profit_l3078_307888

/-- Calculates Todd's profit from his snow-cone stand business --/
theorem todd_snow_cone_profit :
  let loan := 300
  let repayment := 330
  let equipment_cost := 120
  let initial_ingredient_cost := 60
  let marketing_cost := 40
  let misc_cost := 10
  let snow_cone_price := 1.75
  let snow_cone_sales := 500
  let custom_cup_price := 2
  let custom_cup_sales := 250
  let ingredient_cost_increase_rate := 0.2
  let snow_cones_before_increase := 300

  let total_initial_expenses := equipment_cost + initial_ingredient_cost + marketing_cost + misc_cost + repayment
  let total_revenue := snow_cone_price * snow_cone_sales + custom_cup_price * custom_cup_sales
  let increased_ingredient_cost := initial_ingredient_cost * ingredient_cost_increase_rate
  let snow_cones_after_increase := snow_cone_sales - snow_cones_before_increase
  let total_expenses := total_initial_expenses + increased_ingredient_cost

  let profit := total_revenue - total_expenses

  profit = 803 := by sorry

end NUMINAMATH_CALUDE_todd_snow_cone_profit_l3078_307888


namespace NUMINAMATH_CALUDE_parallel_lines_D_eq_18_l3078_307823

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The first line equation -/
def line1 (x y : ℝ) : Prop := x + 2 * y + 1 = 0

/-- The second line equation -/
def line2 (D : ℝ) (x y : ℝ) : Prop := 9 * x + D * y + 1 = 0

/-- The main theorem: if the two lines are parallel, then D = 18 -/
theorem parallel_lines_D_eq_18 :
  (∃ D : ℝ, ∀ x y : ℝ, (line1 x y ↔ line2 D x y) → D = 18) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_D_eq_18_l3078_307823


namespace NUMINAMATH_CALUDE_min_teachers_cover_all_subjects_l3078_307866

/-- Represents the number of teachers for each subject -/
structure TeacherCounts where
  maths : Nat
  physics : Nat
  chemistry : Nat

/-- The maximum number of subjects a teacher can teach -/
def maxSubjectsPerTeacher : Nat := 3

/-- The total number of subjects -/
def totalSubjects : Nat := 3

/-- Given the number of teachers for each subject, calculates the minimum number
    of teachers required to cover all subjects -/
def minTeachersRequired (counts : TeacherCounts) : Nat :=
  sorry

theorem min_teachers_cover_all_subjects (counts : TeacherCounts) :
  counts.maths = 7 →
  counts.physics = 6 →
  counts.chemistry = 5 →
  minTeachersRequired counts = 7 :=
sorry

end NUMINAMATH_CALUDE_min_teachers_cover_all_subjects_l3078_307866


namespace NUMINAMATH_CALUDE_expression_evaluation_l3078_307851

theorem expression_evaluation : (20 ^ 40) / (40 ^ 20) = 10 ^ 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3078_307851


namespace NUMINAMATH_CALUDE_a_plus_b_value_l3078_307874

theorem a_plus_b_value (a b : ℝ) : 
  (∀ x, (3 * (a * x + b) - 8) = 4 * x + 5) → 
  a + b = 17/3 := by sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l3078_307874


namespace NUMINAMATH_CALUDE_orange_picking_theorem_l3078_307824

/-- The total number of oranges picked over three days -/
def total_oranges (day1 day2 day3 : ℕ) : ℕ := day1 + day2 + day3

/-- Theorem stating the total number of oranges picked over three days -/
theorem orange_picking_theorem (day1 day2 day3 : ℕ) 
  (h1 : day1 = 100)
  (h2 : day2 = 3 * day1)
  (h3 : day3 = 70) :
  total_oranges day1 day2 day3 = 470 := by
  sorry


end NUMINAMATH_CALUDE_orange_picking_theorem_l3078_307824


namespace NUMINAMATH_CALUDE_yield_increase_correct_l3078_307850

/-- The percentage increase in rice yield after each harvest -/
def yield_increase_percentage : ℝ := 20

/-- The initial harvest yield in sacks of rice -/
def initial_harvest : ℝ := 20

/-- The total yield after two harvests in sacks of rice -/
def total_yield_two_harvests : ℝ := 44

/-- Theorem stating that the given yield increase percentage is correct -/
theorem yield_increase_correct : 
  initial_harvest + initial_harvest * (1 + yield_increase_percentage / 100) = total_yield_two_harvests :=
by sorry

end NUMINAMATH_CALUDE_yield_increase_correct_l3078_307850


namespace NUMINAMATH_CALUDE_ngon_existence_uniqueness_l3078_307856

/-- Represents a line in a plane -/
structure Line where
  -- Add necessary fields

/-- Represents a point in a plane -/
structure Point where
  -- Add necessary fields

/-- Represents an n-gon -/
structure Polygon (n : ℕ) where
  vertices : Fin n → Point

/-- Checks if a line is perpendicular to a side of a polygon at its midpoint -/
def is_perpendicular_at_midpoint (l : Line) (p : Polygon n) (i : Fin n) : Prop :=
  sorry

/-- Checks if a line is a bisector of an internal or external angle of a polygon -/
def is_angle_bisector (l : Line) (p : Polygon n) (i : Fin n) : Prop :=
  sorry

/-- Represents the solution status of the problem -/
inductive SolutionStatus
| Unique
| Indeterminate
| NoSolution

/-- The main theorem stating the existence and uniqueness of the n-gon -/
theorem ngon_existence_uniqueness 
  (n : ℕ) 
  (lines : Fin n → Line) 
  (condition : (l : Line) → (p : Polygon n) → (i : Fin n) → Prop) : 
  SolutionStatus :=
sorry

end NUMINAMATH_CALUDE_ngon_existence_uniqueness_l3078_307856


namespace NUMINAMATH_CALUDE_acute_triangle_tangent_inequality_l3078_307886

theorem acute_triangle_tangent_inequality (A B C : Real) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  (1 / (1 + Real.tan A) + 1 / (1 + Real.tan B)) < (Real.tan A / (1 + Real.tan A) + Real.tan B / (1 + Real.tan B)) := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_tangent_inequality_l3078_307886


namespace NUMINAMATH_CALUDE_manny_money_left_l3078_307879

/-- The cost of a plastic chair in dollars -/
def chair_cost : ℚ := 55 / 5

/-- The cost of a portable table in dollars -/
def table_cost : ℚ := 3 * chair_cost

/-- Manny's initial amount of money in dollars -/
def initial_money : ℚ := 100

/-- The cost of Manny's purchase (one table and two chairs) in dollars -/
def purchase_cost : ℚ := table_cost + 2 * chair_cost

/-- The amount of money left after Manny's purchase -/
def money_left : ℚ := initial_money - purchase_cost

theorem manny_money_left : money_left = 45 := by sorry

end NUMINAMATH_CALUDE_manny_money_left_l3078_307879


namespace NUMINAMATH_CALUDE_abc_area_is_sqrt3_over_12_l3078_307815

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the points M, P, and O
def M (t : Triangle) : ℝ × ℝ := sorry
def P (t : Triangle) : ℝ × ℝ := sorry
def O (t : Triangle) : ℝ × ℝ := sorry

-- Define the similarity of triangles BOM and AOP
def triangles_similar (t : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    dist (t.B) (O t) / dist (t.A) (O t) = k ∧
    dist (O t) (M t) / dist (O t) (P t) = k ∧
    dist (t.B) (M t) / dist (t.A) (P t) = k

-- Define the condition BO = (1 + √3) OP
def bo_op_relation (t : Triangle) : Prop :=
  dist (t.B) (O t) = (1 + Real.sqrt 3) * dist (O t) (P t)

-- Define the condition BC = 1
def bc_length (t : Triangle) : Prop :=
  dist (t.B) (t.C) = 1

-- Define the area of the triangle
def triangle_area (t : Triangle) : ℝ := sorry

-- State the theorem
theorem abc_area_is_sqrt3_over_12 (t : Triangle) 
  (h1 : triangles_similar t) 
  (h2 : bo_op_relation t) 
  (h3 : bc_length t) : 
  triangle_area t = Real.sqrt 3 / 12 := by sorry

end NUMINAMATH_CALUDE_abc_area_is_sqrt3_over_12_l3078_307815


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3078_307812

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3078_307812


namespace NUMINAMATH_CALUDE_coefficient_x4_equals_180_l3078_307800

/-- The coefficient of x^4 in the expansion of (2 + √x - 1/x^2016)^10 -/
def coefficient_x4 (x : ℝ) : ℕ :=
  -- We define this as a natural number since coefficients in polynomial expansions are typically integers
  -- The actual computation is not implemented here
  sorry

/-- The main theorem stating that the coefficient of x^4 is 180 -/
theorem coefficient_x4_equals_180 :
  ∀ x : ℝ, coefficient_x4 x = 180 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_equals_180_l3078_307800


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3078_307865

theorem quadratic_roots_sum (a b : ℝ) : 
  (∃ (p q r : ℝ), p * (Complex.I ^ 2) + q * Complex.I + r = 0 ∧ 
   (3 + a * Complex.I) * ((3 + a * Complex.I) - (b - 2 * Complex.I)) = 0) → 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3078_307865


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l3078_307827

theorem no_function_satisfies_conditions :
  ¬∃ (f : ℝ → ℝ) (a b : ℝ),
    (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ ≠ f x₂) ∧
    (a > 0 ∧ b > 0) ∧
    (∀ x : ℝ, f (x^2) - (f (a * x + b))^2 ≥ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l3078_307827


namespace NUMINAMATH_CALUDE_b_range_l3078_307869

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x

-- Define the derivative of f(x)
def f_derivative (b : ℝ) (x : ℝ) : ℝ := 3*x^2 + b

-- Theorem statement
theorem b_range (b : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, (f_derivative b x) ≤ 0) →
  b ∈ Set.Iic (-3) :=
by sorry

end NUMINAMATH_CALUDE_b_range_l3078_307869


namespace NUMINAMATH_CALUDE_english_spanish_difference_l3078_307898

/-- The number of hours Ryan spends learning English -/
def hours_english : ℕ := 7

/-- The number of hours Ryan spends learning Chinese -/
def hours_chinese : ℕ := 2

/-- The number of hours Ryan spends learning Spanish -/
def hours_spanish : ℕ := 4

/-- Theorem: Ryan spends 3 more hours on learning English than Spanish -/
theorem english_spanish_difference : hours_english - hours_spanish = 3 := by
  sorry

end NUMINAMATH_CALUDE_english_spanish_difference_l3078_307898


namespace NUMINAMATH_CALUDE_solution_set_when_m_zero_solution_set_all_reals_l3078_307801

/-- The quadratic inequality in question -/
def quadratic_inequality (m : ℝ) (x : ℝ) : Prop :=
  (m - 1) * x^2 + (m - 1) * x + 2 > 0

/-- The solution set when m = 0 -/
theorem solution_set_when_m_zero :
  {x : ℝ | quadratic_inequality 0 x} = Set.Ioo (-2) 1 := by sorry

/-- The condition for the solution set to be all real numbers -/
theorem solution_set_all_reals (m : ℝ) :
  ({x : ℝ | quadratic_inequality m x} = Set.univ) ↔ (m ∈ Set.Icc 1 9) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_zero_solution_set_all_reals_l3078_307801


namespace NUMINAMATH_CALUDE_circular_garden_radius_l3078_307809

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1 / 5) * π * r^2 → r = 10 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l3078_307809


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l3078_307826

theorem sum_reciprocals_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  2 ≤ (1 / a + 1 / b) ∧ ∀ x ≥ 2, ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 2 ∧ 1 / a + 1 / b = x :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l3078_307826


namespace NUMINAMATH_CALUDE_car_wash_rate_l3078_307802

def babysitting_families : ℕ := 4
def babysitting_rate : ℕ := 30
def cars_washed : ℕ := 5
def total_raised : ℕ := 180

theorem car_wash_rate :
  (total_raised - babysitting_families * babysitting_rate) / cars_washed = 12 := by
  sorry

end NUMINAMATH_CALUDE_car_wash_rate_l3078_307802


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3078_307811

/-- Given a rhombus whose diagonal lengths are the roots of x^2 - 14x + 48 = 0, its perimeter is 20 -/
theorem rhombus_perimeter (x₁ x₂ : ℝ) : 
  x₁^2 - 14*x₁ + 48 = 0 → 
  x₂^2 - 14*x₂ + 48 = 0 → 
  x₁ ≠ x₂ →
  let s := Real.sqrt ((x₁^2 + x₂^2) / 4)
  4 * s = 20 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3078_307811


namespace NUMINAMATH_CALUDE_sqrt_190_44_sqrt_176_9_and_18769_integer_n_between_sqrt_l3078_307828

-- Define the table as a function
def f (x : ℝ) : ℝ := x^2

-- Theorem 1
theorem sqrt_190_44 : Real.sqrt 190.44 = 13.8 ∨ Real.sqrt 190.44 = -13.8 := by sorry

-- Theorem 2
theorem sqrt_176_9_and_18769 :
  (abs (Real.sqrt 176.9 - 13.3) < 0.1) ∧ (Real.sqrt 18769 = 137) := by sorry

-- Theorem 3
theorem integer_n_between_sqrt :
  ∀ n : ℕ, (13.5 < Real.sqrt n) ∧ (Real.sqrt n < 13.6) → (n = 183 ∨ n = 184) := by sorry

end NUMINAMATH_CALUDE_sqrt_190_44_sqrt_176_9_and_18769_integer_n_between_sqrt_l3078_307828


namespace NUMINAMATH_CALUDE_f_two_equals_negative_eight_l3078_307842

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem f_two_equals_negative_eight
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_neg : ∀ x < 0, f x = x^3) :
  f 2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_f_two_equals_negative_eight_l3078_307842


namespace NUMINAMATH_CALUDE_min_distance_to_2i_l3078_307804

theorem min_distance_to_2i (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z - 2*Complex.I) ≥ 1 ∧ ∃ w : ℂ, Complex.abs w = 1 ∧ Complex.abs (w - 2*Complex.I) = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_2i_l3078_307804


namespace NUMINAMATH_CALUDE_andy_rahim_age_difference_l3078_307838

/-- The age difference between Andy and Rahim -/
def ageDifference (rahimAge : ℕ) (andyFutureAge : ℕ) : ℕ :=
  andyFutureAge - 5 - rahimAge

theorem andy_rahim_age_difference :
  ∀ (rahimAge : ℕ) (andyFutureAge : ℕ),
    rahimAge = 6 →
    andyFutureAge = 2 * rahimAge →
    ageDifference rahimAge andyFutureAge = 1 := by
  sorry

end NUMINAMATH_CALUDE_andy_rahim_age_difference_l3078_307838


namespace NUMINAMATH_CALUDE_tangent_circles_m_value_l3078_307885

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x + m^2 - 1 = 0

-- Define the condition of external tangency
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y m

-- State the theorem
theorem tangent_circles_m_value (m : ℝ) :
  externally_tangent m → m = 3 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_m_value_l3078_307885


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3078_307837

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) : 
  B = 60 * π / 180 →
  b = 7 * Real.sqrt 6 →
  a = 14 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  A = 45 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3078_307837


namespace NUMINAMATH_CALUDE_rock_cd_price_l3078_307870

/-- The price of a rock and roll CD -/
def rock_price : ℝ := sorry

/-- The price of a pop CD -/
def pop_price : ℝ := 10

/-- The price of a dance CD -/
def dance_price : ℝ := 3

/-- The price of a country CD -/
def country_price : ℝ := 7

/-- The number of each type of CD Julia wants to buy -/
def quantity : ℕ := 4

/-- The amount of money Julia has -/
def julia_money : ℝ := 75

/-- The amount Julia is short by -/
def short_amount : ℝ := 25

theorem rock_cd_price : rock_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_rock_cd_price_l3078_307870


namespace NUMINAMATH_CALUDE_decimal_sum_and_multiply_l3078_307896

theorem decimal_sum_and_multiply : 
  let a : ℚ := 0.0034
  let b : ℚ := 0.125
  let c : ℚ := 0.00678
  2 * (a + b + c) = 0.27036 := by
sorry

end NUMINAMATH_CALUDE_decimal_sum_and_multiply_l3078_307896


namespace NUMINAMATH_CALUDE_bagel_bakery_bound_l3078_307829

/-- Definition of a bagel -/
def Bagel (a b : ℕ) : ℕ := 2 * a + 2 * b + 4

/-- Definition of a bakery of order n -/
def Bakery (n : ℕ) : Set (ℕ × ℕ) := sorry

/-- The smallest possible number of cells in a bakery of order n -/
def f (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem bagel_bakery_bound :
  ∃ (α : ℝ), ∀ (n : ℕ), n ≥ 8 → Even n →
    ∃ (N : ℕ), ∀ (m : ℕ), m ≥ N →
      (1 / 100 : ℝ) < (f m : ℝ) / m ^ α ∧ (f m : ℝ) / m ^ α < 100 :=
by sorry

end NUMINAMATH_CALUDE_bagel_bakery_bound_l3078_307829


namespace NUMINAMATH_CALUDE_function_property_l3078_307891

theorem function_property (f : ℤ → ℤ) 
  (h : ∀ (x y : ℤ), f x + f y = f (x + 1) + f (y - 1))
  (h1 : f 2016 = 6102)
  (h2 : f 6102 = 2016) :
  f 1 = 8117 := by
sorry

end NUMINAMATH_CALUDE_function_property_l3078_307891


namespace NUMINAMATH_CALUDE_ambers_age_l3078_307858

theorem ambers_age :
  ∀ (a g : ℕ),
  g = 15 * a →
  g - a = 70 →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_ambers_age_l3078_307858


namespace NUMINAMATH_CALUDE_negation_of_unique_solution_l3078_307854

theorem negation_of_unique_solution (a b : ℝ) (h : a ≠ 0) :
  ¬(∃! x : ℝ, a * x = b) ↔ (¬∃ x : ℝ, a * x = b) ∨ (∃ x y : ℝ, x ≠ y ∧ a * x = b ∧ a * y = b) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_unique_solution_l3078_307854


namespace NUMINAMATH_CALUDE_cannot_determine_percentage_increase_l3078_307852

/-- Represents a manufacturing machine -/
structure Machine where
  name : String
  production_rate : ℝ

/-- The problem setup -/
def sprocket_problem (time_q : ℝ) : Prop :=
  let machine_a : Machine := ⟨"A", 4⟩
  let machine_q : Machine := ⟨"Q", 440 / time_q⟩
  let machine_p : Machine := ⟨"P", 440 / (time_q + 10)⟩
  let percentage_increase := (machine_q.production_rate - machine_a.production_rate) / machine_a.production_rate * 100

  -- Conditions
  440 > 0 ∧
  time_q > 0 ∧
  machine_p.production_rate < machine_q.production_rate ∧
  -- Question: Can we determine the percentage increase?
  ∃ (x : ℝ), percentage_increase = x

/-- The theorem stating that we cannot determine the percentage increase without knowing time_q -/
theorem cannot_determine_percentage_increase :
  ¬∃ (x : ℝ), ∀ (time_q : ℝ), sprocket_problem time_q → 
    (440 / time_q - 4) / 4 * 100 = x :=
sorry

end NUMINAMATH_CALUDE_cannot_determine_percentage_increase_l3078_307852


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3078_307895

def A : Set ℕ := {2, 4, 6, 16, 29}
def B : Set ℕ := {4, 16, 20, 27, 29, 32}

theorem intersection_of_A_and_B : A ∩ B = {4, 16, 29} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3078_307895


namespace NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l3078_307849

/-- Given an arithmetic sequence where a₇ = 10 and a₂₁ = 34, prove that a₅₀ = 682/7 -/
theorem arithmetic_sequence_50th_term :
  ∀ (a : ℕ → ℚ), 
    (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
    a 7 = 10 →                                        -- 7th term is 10
    a 21 = 34 →                                       -- 21st term is 34
    a 50 = 682 / 7 :=                                 -- 50th term is 682/7
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l3078_307849


namespace NUMINAMATH_CALUDE_cos_2alpha_is_zero_l3078_307881

theorem cos_2alpha_is_zero (α : Real) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.sin (2*α) = Real.cos (π/4 - α)) : Real.cos (2*α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_is_zero_l3078_307881


namespace NUMINAMATH_CALUDE_equation_represents_parabola_l3078_307890

/-- The equation |y-3| = √((x+4)² + (y-1)²) represents a parabola -/
theorem equation_represents_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (∀ x y : ℝ, |y - 3| = Real.sqrt ((x + 4)^2 + (y - 1)^2) ↔ y = a * x^2 + b * x + c) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_parabola_l3078_307890


namespace NUMINAMATH_CALUDE_det_scaled_matrices_l3078_307853

-- Define a 2x2 matrix type
def Matrix2x2 := Fin 2 → Fin 2 → ℝ

-- Define the determinant function for 2x2 matrices
def det (A : Matrix2x2) : ℝ :=
  A 0 0 * A 1 1 - A 0 1 * A 1 0

-- Define a function to scale all elements of a matrix by a factor
def scaleMatrix (A : Matrix2x2) (k : ℝ) : Matrix2x2 :=
  λ i j ↦ k * A i j

-- Define a function to scale columns of a matrix by different factors
def scaleColumns (A : Matrix2x2) (k1 k2 : ℝ) : Matrix2x2 :=
  λ i j ↦ if j = 0 then k1 * A i j else k2 * A i j

-- State the theorem
theorem det_scaled_matrices (A : Matrix2x2) (h : det A = 3) :
  det (scaleMatrix A 3) = 27 ∧ det (scaleColumns A 4 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_det_scaled_matrices_l3078_307853


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l3078_307893

-- Define the complex number z
def z : ℂ := 1 - 2 * Complex.I

-- Theorem statement
theorem imaginary_part_of_reciprocal (z : ℂ) (h : z = 1 - 2 * Complex.I) :
  Complex.im (z⁻¹) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l3078_307893


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l3078_307857

theorem parallelogram_side_sum (x y : ℝ) : 
  (5 : ℝ) = 10 * y - 3 ∧ (11 : ℝ) = 4 * x + 1 → x + y = 3.3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l3078_307857


namespace NUMINAMATH_CALUDE_factorial_sum_not_end_1990_l3078_307808

theorem factorial_sum_not_end_1990 (m n : ℕ) : (m.factorial + n.factorial) % 10000 ≠ 1990 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_not_end_1990_l3078_307808


namespace NUMINAMATH_CALUDE_prime_product_sum_difference_l3078_307899

theorem prime_product_sum_difference : ∃ x y : ℕ, 
  x.Prime ∧ y.Prime ∧ 
  x ≠ y ∧ 
  20 < x ∧ x < 40 ∧ 
  20 < y ∧ y < 40 ∧ 
  x * y - (x + y) = 899 := by
  sorry

end NUMINAMATH_CALUDE_prime_product_sum_difference_l3078_307899


namespace NUMINAMATH_CALUDE_vehicle_value_depreciation_l3078_307848

theorem vehicle_value_depreciation (last_year_value : ℝ) (depreciation_factor : ℝ) (this_year_value : ℝ) :
  last_year_value = 20000 →
  depreciation_factor = 0.8 →
  this_year_value = last_year_value * depreciation_factor →
  this_year_value = 16000 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_value_depreciation_l3078_307848


namespace NUMINAMATH_CALUDE_isabel_birthday_money_l3078_307876

/-- The amount of money Isabel received for her birthday -/
def birthday_money : ℕ := sorry

/-- The cost of each toy -/
def toy_cost : ℕ := 2

/-- The number of toys Isabel could buy -/
def toys_bought : ℕ := 7

/-- Theorem stating that Isabel's birthday money is equal to the total cost of the toys she could buy -/
theorem isabel_birthday_money :
  birthday_money = toy_cost * toys_bought :=
sorry

end NUMINAMATH_CALUDE_isabel_birthday_money_l3078_307876


namespace NUMINAMATH_CALUDE_cubic_expression_value_l3078_307806

theorem cubic_expression_value (p q : ℝ) : 
  3 * p^2 - 7 * p + 1 = 0 →
  3 * q^2 - 7 * q + 1 = 0 →
  (9 * p^3 - 9 * q^3) / (p - q) = 46 := by
sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l3078_307806


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3078_307878

theorem quadratic_one_root (m : ℝ) : 
  (∀ x : ℝ, x^2 + 6*m*x + 4*m = 0 → (∀ y : ℝ, y^2 + 6*m*y + 4*m = 0 → x = y)) →
  m = 4/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3078_307878


namespace NUMINAMATH_CALUDE_initial_pink_hats_l3078_307832

/-- The number of pink hard hats initially in the truck -/
def initial_pink : ℕ := sorry

/-- The number of green hard hats initially in the truck -/
def initial_green : ℕ := 15

/-- The number of yellow hard hats initially in the truck -/
def initial_yellow : ℕ := 24

/-- The number of pink hard hats Carl takes away -/
def carl_pink : ℕ := 4

/-- The number of pink hard hats John takes away -/
def john_pink : ℕ := 6

/-- The number of green hard hats John takes away -/
def john_green : ℕ := 2 * john_pink

/-- The total number of hard hats remaining in the truck after Carl and John take some away -/
def remaining_hats : ℕ := 43

theorem initial_pink_hats : initial_pink = 26 := by sorry

end NUMINAMATH_CALUDE_initial_pink_hats_l3078_307832


namespace NUMINAMATH_CALUDE_marbles_problem_l3078_307845

theorem marbles_problem (total : ℕ) (bags : ℕ) (remaining : ℕ) : 
  bags = 4 →
  remaining = 21 →
  (total / bags) * (bags - 1) = remaining →
  total = 28 :=
by sorry

end NUMINAMATH_CALUDE_marbles_problem_l3078_307845


namespace NUMINAMATH_CALUDE_cuboid_area_example_l3078_307860

/-- The surface area of a cuboid -/
def cuboid_surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a cuboid with length 8 cm, breadth 6 cm, and height 9 cm is 348 square centimeters -/
theorem cuboid_area_example : cuboid_surface_area 8 6 9 = 348 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_area_example_l3078_307860


namespace NUMINAMATH_CALUDE_circle_equation_l3078_307814

theorem circle_equation (x y : ℝ) :
  (x^2 + 8*x + y^2 + 4*y - 36 = 0) ↔
  ((x + 4)^2 + (y + 2)^2 = 4^2) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3078_307814


namespace NUMINAMATH_CALUDE_circle_equations_l3078_307889

-- Define the points
def M : ℝ × ℝ := (-1, 1)
def N : ℝ × ℝ := (0, 2)
def Q : ℝ × ℝ := (2, 0)

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 1/2)^2 + (y - 1/2)^2 = 5/2
def C₂ (x y : ℝ) : Prop := (x + 3/2)^2 + (y - 5/2)^2 = 5/2

-- Define the line MN
def MN (x y : ℝ) : Prop := x - y + 2 = 0

-- Theorem statement
theorem circle_equations :
  (C₁ M.1 M.2 ∧ C₁ N.1 N.2 ∧ C₁ Q.1 Q.2) ∧
  (∀ x y x' y', C₁ x y ∧ C₂ x' y' → 
    MN ((x + x')/2) ((y + y')/2) ∧
    (x' - x)^2 + (y' - y)^2 = 4 * ((x - 1/2)^2 + (y - 1/2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equations_l3078_307889


namespace NUMINAMATH_CALUDE_equation_solution_l3078_307883

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (7 * x)^14 = (14 * x)^7 ↔ x = 2/7 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3078_307883


namespace NUMINAMATH_CALUDE_select_volunteers_l3078_307813

theorem select_volunteers (boys girls volunteers : ℕ) 
  (h1 : boys = 6)
  (h2 : girls = 2)
  (h3 : volunteers = 3) :
  (Nat.choose (boys + girls) volunteers) - (Nat.choose boys volunteers) = 36 := by
  sorry

end NUMINAMATH_CALUDE_select_volunteers_l3078_307813


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3078_307807

theorem linear_equation_solution (a b m : ℝ) : 
  (∀ y, (a + b) * y^2 - y^((1/3)*a + 2) + 5 = 0 → (a + b = 0 ∧ (1/3)*a + 2 = 1)) →
  ((a + 2)/6 - (a - 1)/2 + 3 = a - (2*a - m)/6) →
  |a - b| - |b - m| = -32 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3078_307807


namespace NUMINAMATH_CALUDE_cubic_inequality_l3078_307822

theorem cubic_inequality (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  a^3 + b^3 ≤ a*b^2 + a^2*b := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3078_307822


namespace NUMINAMATH_CALUDE_chicken_pasta_orders_count_l3078_307835

def chicken_pasta_pieces : ℕ := 2
def barbecue_chicken_pieces : ℕ := 3
def fried_chicken_dinner_pieces : ℕ := 8
def fried_chicken_dinner_orders : ℕ := 2
def barbecue_chicken_orders : ℕ := 3
def total_chicken_pieces : ℕ := 37

theorem chicken_pasta_orders_count : 
  ∃ (chicken_pasta_orders : ℕ), 
    chicken_pasta_orders * chicken_pasta_pieces + 
    barbecue_chicken_orders * barbecue_chicken_pieces + 
    fried_chicken_dinner_orders * fried_chicken_dinner_pieces = 
    total_chicken_pieces ∧ 
    chicken_pasta_orders = 6 := by
  sorry

end NUMINAMATH_CALUDE_chicken_pasta_orders_count_l3078_307835


namespace NUMINAMATH_CALUDE_reflection_equivalence_l3078_307821

-- Define the shape type
inductive Shape
  | OriginalL
  | InvertedL
  | UpsideDownRotatedL
  | VerticallyFlippedL
  | HorizontallyMirroredL
  | UnalteredL

-- Define the reflection operation
def reflectAcrossDiagonal (s : Shape) : Shape :=
  match s with
  | Shape.OriginalL => Shape.HorizontallyMirroredL
  | _ => s  -- For completeness, though we only care about OriginalL

-- State the theorem
theorem reflection_equivalence :
  reflectAcrossDiagonal Shape.OriginalL = Shape.HorizontallyMirroredL :=
by sorry

end NUMINAMATH_CALUDE_reflection_equivalence_l3078_307821


namespace NUMINAMATH_CALUDE_expression_simplification_l3078_307841

theorem expression_simplification (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 6) * 5 + (5 - 2 / 4) * (8 * p - 12) = -19 * p - 39 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3078_307841


namespace NUMINAMATH_CALUDE_additional_track_length_l3078_307894

/-- Calculate the additional track length required when reducing grade -/
theorem additional_track_length
  (rise : ℝ)
  (initial_grade : ℝ)
  (reduced_grade : ℝ)
  (h1 : rise = 800)
  (h2 : initial_grade = 0.04)
  (h3 : reduced_grade = 0.025) :
  (rise / reduced_grade) - (rise / initial_grade) = 12000 :=
by sorry

end NUMINAMATH_CALUDE_additional_track_length_l3078_307894


namespace NUMINAMATH_CALUDE_chemistry_alone_count_l3078_307816

/-- Represents the number of students in a school with chemistry and biology classes -/
structure School where
  total : ℕ
  chemistry : ℕ
  biology : ℕ
  both : ℕ

/-- The conditions of the school -/
def school_conditions (s : School) : Prop :=
  s.total = 100 ∧
  s.chemistry + s.biology - s.both = s.total ∧
  s.chemistry = 4 * s.biology ∧
  s.both = 10

/-- The theorem stating that under the given conditions, 
    the number of students in chemistry class alone is 80 -/
theorem chemistry_alone_count (s : School) 
  (h : school_conditions s) : s.chemistry - s.both = 80 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_alone_count_l3078_307816


namespace NUMINAMATH_CALUDE_five_digit_palindrome_digits_l3078_307803

/-- A function that calculates the number of 5-digit palindromes that can be formed
    using n distinct digits -/
def palindrome_count (n : ℕ) : ℕ := n * n * n

/-- The theorem stating that if there are 125 possible 5-digit palindromes formed
    using some distinct digits, then the number of distinct digits is 5 -/
theorem five_digit_palindrome_digits :
  (∃ (n : ℕ), n > 0 ∧ palindrome_count n = 125) →
  (∃ (n : ℕ), n > 0 ∧ palindrome_count n = 125 ∧ n = 5) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_palindrome_digits_l3078_307803


namespace NUMINAMATH_CALUDE_irreducible_fraction_l3078_307880

theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l3078_307880


namespace NUMINAMATH_CALUDE_inequality_proof_l3078_307863

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : n ≥ 2) :
  (1 - x)^n + (1 + x)^n < 2^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3078_307863


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_even_l3078_307825

/-- Sum of first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of first n positive even integers -/
def sum_first_n_even (n : ℕ) : ℕ := 2 * sum_first_n n

/-- Sum of five consecutive even integers -/
def sum_five_consecutive_even (n : ℕ) : ℕ := 5 * n - 20

theorem largest_of_five_consecutive_even :
  ∃ n : ℕ, sum_first_n_even 30 = sum_five_consecutive_even n ∧ n = 190 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_even_l3078_307825


namespace NUMINAMATH_CALUDE_mask_quality_most_suitable_l3078_307862

-- Define the survey types
inductive SurveyType
| SecurityCheck
| TeacherRecruitment
| MaskQuality
| StudentVision

-- Define a function to determine if a survey is suitable for sampling
def isSuitableForSampling (survey : SurveyType) : Prop :=
  match survey with
  | SurveyType.MaskQuality => True
  | _ => False

-- Theorem statement
theorem mask_quality_most_suitable :
  ∀ (survey : SurveyType), isSuitableForSampling survey → survey = SurveyType.MaskQuality :=
by sorry

end NUMINAMATH_CALUDE_mask_quality_most_suitable_l3078_307862


namespace NUMINAMATH_CALUDE_wicket_keeper_age_l3078_307884

theorem wicket_keeper_age (team_size : ℕ) (team_avg_age : ℝ) (remaining_avg_age : ℝ) : 
  team_size = 11 →
  team_avg_age = 24 →
  remaining_avg_age = team_avg_age - 1 →
  ∃ (wicket_keeper_age : ℝ),
    wicket_keeper_age = team_avg_age + 9 ∧
    (team_size - 2) * remaining_avg_age + wicket_keeper_age + team_avg_age = team_size * team_avg_age :=
by sorry

end NUMINAMATH_CALUDE_wicket_keeper_age_l3078_307884


namespace NUMINAMATH_CALUDE_distance_at_16_00_l3078_307855

/-- Represents the distance to Moscow at a given time -/
structure DistanceAtTime where
  time : ℕ  -- Time in hours since 12:00
  lowerBound : ℚ
  upperBound : ℚ

/-- The problem statement -/
theorem distance_at_16_00 
  (d12 : DistanceAtTime) 
  (d13 : DistanceAtTime)
  (d15 : DistanceAtTime)
  (h_constant_speed : ∀ t₁ t₂, d12.time ≤ t₁ → t₁ < t₂ → t₂ ≤ d15.time → 
    (d12.lowerBound - d15.upperBound) / (d15.time - d12.time) ≤ 
    (d12.upperBound - d15.lowerBound) / (d15.time - d12.time))
  (h_d12 : d12.time = 0 ∧ d12.lowerBound = 81.5 ∧ d12.upperBound = 82.5)
  (h_d13 : d13.time = 1 ∧ d13.lowerBound = 70.5 ∧ d13.upperBound = 71.5)
  (h_d15 : d15.time = 3 ∧ d15.lowerBound = 45.5 ∧ d15.upperBound = 46.5) :
  ∃ (d : ℚ), d = 34 ∧ 
    (d12.lowerBound - d) / 4 = (d12.upperBound - d) / 4 ∧
    (d13.lowerBound - d) / 3 = (d13.upperBound - d) / 3 ∧
    (d15.lowerBound - d) / 1 = (d15.upperBound - d) / 1 :=
sorry

end NUMINAMATH_CALUDE_distance_at_16_00_l3078_307855


namespace NUMINAMATH_CALUDE_seven_telephones_wires_l3078_307836

/-- The number of wires needed to connect n telephone sets, where each pair is connected. -/
def wiresNeeded (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 7 telephone sets, the number of wires needed is 21. -/
theorem seven_telephones_wires : wiresNeeded 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_seven_telephones_wires_l3078_307836


namespace NUMINAMATH_CALUDE_annes_total_distance_l3078_307819

/-- Anne's hiking journey -/
def annes_hike (flat_speed flat_time uphill_speed uphill_time downhill_speed downhill_time : ℝ) : ℝ :=
  flat_speed * flat_time + uphill_speed * uphill_time + downhill_speed * downhill_time

/-- Theorem: Anne's total distance traveled is 14 miles -/
theorem annes_total_distance :
  annes_hike 3 2 2 2 4 1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_annes_total_distance_l3078_307819


namespace NUMINAMATH_CALUDE_expand_expression_l3078_307818

theorem expand_expression (y : ℝ) : 12 * (3 * y - 4) = 36 * y - 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3078_307818


namespace NUMINAMATH_CALUDE_sum_34_47_in_base4_l3078_307830

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Adds two numbers in base 10 and returns the result in base 4 -/
def addAndConvertToBase4 (a b : ℕ) : List ℕ :=
  toBase4 (a + b)

theorem sum_34_47_in_base4 :
  addAndConvertToBase4 34 47 = [1, 1, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_34_47_in_base4_l3078_307830


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3078_307846

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7*I
  let z₂ : ℂ := 4 - 7*I
  (z₁ / z₂) + (z₂ / z₁) = -66/65 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3078_307846


namespace NUMINAMATH_CALUDE_remainder_sum_l3078_307805

theorem remainder_sum (c d : ℤ) 
  (hc : c % 60 = 47)
  (hd : d % 45 = 28) :
  (c + d) % 30 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3078_307805


namespace NUMINAMATH_CALUDE_cross_pollinated_percentage_l3078_307871

theorem cross_pollinated_percentage
  (total : ℕ)  -- Total number of trees
  (fuji : ℕ)   -- Number of pure Fuji trees
  (gala : ℕ)   -- Number of pure Gala trees
  (cross : ℕ)  -- Number of cross-pollinated trees
  (h1 : total = fuji + gala + cross)  -- Total trees equation
  (h2 : fuji + cross = 221)           -- Pure Fuji + Cross-pollinated
  (h3 : fuji = (3 * total) / 4)       -- 3/4 of all trees are pure Fuji
  (h4 : gala = 39)                    -- Number of pure Gala trees
  : (cross : ℚ) / total * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cross_pollinated_percentage_l3078_307871


namespace NUMINAMATH_CALUDE_convex_polygon_covered_by_three_similar_l3078_307882

/-- A planar convex polygon. -/
structure PlanarConvexPolygon where
  -- Add necessary fields and properties here
  -- This is a placeholder definition

/-- Similarity between two planar convex polygons. -/
def IsSimilar (P Q : PlanarConvexPolygon) : Prop :=
  -- Define similarity condition here
  sorry

/-- One polygon covers another. -/
def Covers (P Q : PlanarConvexPolygon) : Prop :=
  -- Define covering condition here
  sorry

/-- Union of three polygons. -/
def Union3 (P Q R : PlanarConvexPolygon) : PlanarConvexPolygon :=
  -- Define union operation here
  sorry

/-- A polygon is smaller than another. -/
def IsSmaller (P Q : PlanarConvexPolygon) : Prop :=
  -- Define size comparison here
  sorry

/-- Theorem: Every planar convex polygon can be covered by three smaller similar polygons. -/
theorem convex_polygon_covered_by_three_similar :
  ∀ (M : PlanarConvexPolygon),
  ∃ (N₁ N₂ N₃ : PlanarConvexPolygon),
    IsSimilar N₁ M ∧ IsSimilar N₂ M ∧ IsSimilar N₃ M ∧
    IsSmaller N₁ M ∧ IsSmaller N₂ M ∧ IsSmaller N₃ M ∧
    Covers (Union3 N₁ N₂ N₃) M :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_covered_by_three_similar_l3078_307882


namespace NUMINAMATH_CALUDE_adjacent_sum_9_is_30_l3078_307892

def divisors_of_216 : List ℕ := [2, 3, 4, 6, 8, 9, 12, 18, 24, 27, 36, 54, 72, 108, 216]

def valid_arrangement (arr : List ℕ) : Prop :=
  ∀ i j, i ≠ j → (arr.get! i).gcd (arr.get! j) > 1

def adjacent_sum_9 (arr : List ℕ) : ℕ :=
  let idx := arr.indexOf 9
  (arr.get! ((idx - 1 + arr.length) % arr.length)) + (arr.get! ((idx + 1) % arr.length))

theorem adjacent_sum_9_is_30 :
  ∃ arr : List ℕ, arr.Perm divisors_of_216 ∧ valid_arrangement arr ∧ adjacent_sum_9 arr = 30 :=
sorry

end NUMINAMATH_CALUDE_adjacent_sum_9_is_30_l3078_307892


namespace NUMINAMATH_CALUDE_factorial_multiple_of_eight_l3078_307872

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem factorial_multiple_of_eight (n : ℕ) :
  (∃ k : ℕ, factorial n = 8 * k) → n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_multiple_of_eight_l3078_307872


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3078_307877

theorem polynomial_division_theorem :
  ∃ (α β r : ℝ), ∀ z : ℝ,
    4 * z^4 - 3 * z^3 + 5 * z^2 - 7 * z + 6 =
    (4 * z + 7) * (z^3 - 2.5 * z^2 + α * z + β) + r :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3078_307877


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3078_307831

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) :
  InverselyProportional x y →
  (x + y = 30 → x = 3 * y) →
  x = -12 →
  y = -14.0625 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3078_307831


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l3078_307840

theorem partial_fraction_decomposition_product (N₁ N₂ : ℝ) : 
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ 2 → (42 * x - 36) / (x^2 - 3*x + 2) = N₁ / (x - 1) + N₂ / (x - 2)) →
  N₁ * N₂ = -288 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l3078_307840


namespace NUMINAMATH_CALUDE_sum_fraction_bounds_l3078_307868

theorem sum_fraction_bounds (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let S := a / (a + b + d) + b / (b + c + a) + c / (c + d + b) + d / (d + a + c)
  1 < S ∧ S < 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_fraction_bounds_l3078_307868


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3078_307820

/-- The constant term in the binomial expansion of (x - 2/x)^8 -/
def constant_term : ℤ := 1120

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The general term of the binomial expansion (x - 2/x)^8 -/
def general_term (r : ℕ) : ℤ := (-2)^r * binomial 8 r

theorem constant_term_binomial_expansion :
  constant_term = general_term 4 := by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3078_307820


namespace NUMINAMATH_CALUDE_expand_binomial_product_simplify_algebraic_expression_expand_cubic_difference_l3078_307897

-- 1. Expansion of (2m-3)(5-3m)
theorem expand_binomial_product (m : ℝ) : 
  (2*m - 3) * (5 - 3*m) = -6*m^2 + 19*m - 15 := by sorry

-- 2. Simplification of (3a^3)^2⋅(2b^2)^3÷(6ab)^2
theorem simplify_algebraic_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (3*a^3)^2 * (2*b^2)^3 / (6*a*b)^2 = 2*a^4*b^4 := by sorry

-- 3. Expansion of (a-b)(a^2+ab+b^2)
theorem expand_cubic_difference (a b : ℝ) : 
  (a - b) * (a^2 + a*b + b^2) = a^3 - b^3 := by sorry

end NUMINAMATH_CALUDE_expand_binomial_product_simplify_algebraic_expression_expand_cubic_difference_l3078_307897


namespace NUMINAMATH_CALUDE_fifa_world_cup_players_l3078_307810

/-- The number of teams in the 17th FIFA World Cup -/
def num_teams : ℕ := 35

/-- The number of players in each team -/
def players_per_team : ℕ := 23

/-- The total number of players in the 17th FIFA World Cup -/
def total_players : ℕ := num_teams * players_per_team

theorem fifa_world_cup_players :
  total_players = 805 := by sorry

end NUMINAMATH_CALUDE_fifa_world_cup_players_l3078_307810


namespace NUMINAMATH_CALUDE_division_multiplication_equivalence_l3078_307875

theorem division_multiplication_equivalence : 
  (5.8 / 0.001) = (5.8 * 1000) := by sorry

end NUMINAMATH_CALUDE_division_multiplication_equivalence_l3078_307875


namespace NUMINAMATH_CALUDE_unique_point_for_equal_angles_l3078_307867

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the point P
def P : ℝ × ℝ := (4, 0)

-- Define a chord passing through the focus
def chord (a b : ℝ × ℝ) : Prop :=
  a.1 ≠ b.1 ∨ a.2 ≠ b.2  -- Ensure A and B are distinct points
  ∧ ellipse a.1 a.2      -- A is on the ellipse
  ∧ ellipse b.1 b.2      -- B is on the ellipse
  ∧ (b.2 - a.2) * (a.1 - 1) = (b.1 - a.1) * (a.2 - 0)  -- AB passes through F(1,0)

-- Define the equality of angles APF and BPF
def equal_angles (a b : ℝ × ℝ) : Prop :=
  (a.2 - 0) * (b.1 - 4) = (b.2 - 0) * (a.1 - 4)

theorem unique_point_for_equal_angles :
  ∀ a b : ℝ × ℝ, chord a b → equal_angles a b ∧
  ∀ p : ℝ, p > 0 ∧ p ≠ 4 →
    ∃ c d : ℝ × ℝ, chord c d ∧ ¬(c.2 - 0) * (d.1 - p) = (d.2 - 0) * (c.1 - p) :=
sorry

end NUMINAMATH_CALUDE_unique_point_for_equal_angles_l3078_307867


namespace NUMINAMATH_CALUDE_xyz_inequality_l3078_307843

theorem xyz_inequality (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l3078_307843


namespace NUMINAMATH_CALUDE_system_solution_l3078_307861

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ = 6 ∧ y₁ = 3 ∧ x₂ = 3 ∧ y₂ = 3/2) ∧
    (∀ x y : ℝ,
      3*x - 2*y > 0 ∧ x > 0 →
      (Real.sqrt ((3*x - 2*y)/(2*x)) + Real.sqrt ((2*x)/(3*x - 2*y)) = 2 ∧
       x^2 - 18 = 2*y*(4*y - 9)) →
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3078_307861


namespace NUMINAMATH_CALUDE_smallest_n_mod_30_l3078_307847

theorem smallest_n_mod_30 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(435 * m ≡ 867 * m [ZMOD 30])) ∧ 
  (435 * n ≡ 867 * n [ZMOD 30]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_mod_30_l3078_307847


namespace NUMINAMATH_CALUDE_stock_price_calculation_l3078_307864

/-- Given a closing price and percent increase, calculate the opening price of a stock. -/
theorem stock_price_calculation (closing_price : ℝ) (percent_increase : ℝ) (opening_price : ℝ) :
  closing_price = 29 ∧ 
  percent_increase = 3.571428571428581 ∧
  (closing_price - opening_price) / opening_price * 100 = percent_increase →
  opening_price = 28 := by
sorry


end NUMINAMATH_CALUDE_stock_price_calculation_l3078_307864


namespace NUMINAMATH_CALUDE_faculty_size_l3078_307833

/-- The number of students studying numeric methods -/
def nm : ℕ := 240

/-- The number of students studying automatic control of airborne vehicles -/
def acav : ℕ := 423

/-- The number of students studying both numeric methods and automatic control -/
def nm_acav : ℕ := 134

/-- The number of students studying advanced robotics -/
def ar : ℕ := 365

/-- The number of students studying both numeric methods and advanced robotics -/
def nm_ar : ℕ := 75

/-- The number of students studying both automatic control and advanced robotics -/
def acav_ar : ℕ := 95

/-- The number of students studying all three subjects -/
def all_three : ℕ := 45

/-- The proportion of second year students to total students -/
def second_year_ratio : ℚ := 4/5

/-- The total number of students in the faculty -/
def total_students : ℕ := 905

theorem faculty_size :
  (nm + acav + ar - nm_acav - nm_ar - acav_ar + all_three : ℚ) / second_year_ratio = total_students := by
  sorry

end NUMINAMATH_CALUDE_faculty_size_l3078_307833


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3078_307873

def M : Set ℕ := {1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2*a - 1}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3078_307873


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l3078_307844

def total_cost (adult_meal_cost adult_drink_cost kid_drink_cost dessert_cost : ℚ)
               (num_adults num_kids num_exclusive_dishes : ℕ)
               (discount_rate sales_tax_rate service_charge_rate : ℚ)
               (exclusive_dish_charge : ℚ) : ℚ :=
  let subtotal := adult_meal_cost * num_adults +
                  adult_drink_cost * num_adults +
                  kid_drink_cost * num_kids +
                  dessert_cost * (num_adults + num_kids) +
                  exclusive_dish_charge * num_exclusive_dishes
  let discounted_subtotal := subtotal * (1 - discount_rate)
  let with_tax := discounted_subtotal * (1 + sales_tax_rate)
  let final_total := with_tax * (1 + service_charge_rate)
  final_total

theorem restaurant_bill_calculation :
  let adult_meal_cost : ℚ := 12
  let adult_drink_cost : ℚ := 2.5
  let kid_drink_cost : ℚ := 1.5
  let dessert_cost : ℚ := 4
  let num_adults : ℕ := 7
  let num_kids : ℕ := 4
  let num_exclusive_dishes : ℕ := 3
  let discount_rate : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.075
  let service_charge_rate : ℚ := 0.15
  let exclusive_dish_charge : ℚ := 3
  total_cost adult_meal_cost adult_drink_cost kid_drink_cost dessert_cost
             num_adults num_kids num_exclusive_dishes
             discount_rate sales_tax_rate service_charge_rate
             exclusive_dish_charge = 178.57 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l3078_307844


namespace NUMINAMATH_CALUDE_particle_position_after_2023_minutes_l3078_307834

/-- Represents the position of a particle as a pair of integers -/
def Position := ℤ × ℤ

/-- Represents the movement pattern of the particle -/
inductive MovementPattern
| OddSequence
| EvenSequence

/-- Calculates the next position based on the current position, movement pattern, and side length -/
def nextPosition (pos : Position) (pattern : MovementPattern) (side : ℕ) : Position :=
  match pattern with
  | MovementPattern.OddSequence => (pos.1 - side, pos.2 - side)
  | MovementPattern.EvenSequence => (pos.1 + side, pos.2 + side)

/-- Calculates the position of the particle after a given number of minutes -/
def particlePosition (minutes : ℕ) : Position :=
  sorry

/-- Theorem stating that the particle's position after 2023 minutes is (-43, -43) -/
theorem particle_position_after_2023_minutes :
  particlePosition 2023 = (-43, -43) :=
  sorry

end NUMINAMATH_CALUDE_particle_position_after_2023_minutes_l3078_307834


namespace NUMINAMATH_CALUDE_inequality_chain_l3078_307839

theorem inequality_chain (a : ℝ) (h : a - 1 > 0) : -a < -1 ∧ -1 < 1 ∧ 1 < a := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l3078_307839
