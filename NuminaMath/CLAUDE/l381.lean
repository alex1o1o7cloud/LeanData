import Mathlib

namespace NUMINAMATH_CALUDE_quadrilateral_similarity_l381_38199

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  -- Add necessary fields and properties to define a convex quadrilateral
  -- This is a placeholder and should be expanded based on specific requirements

/-- Construct a new quadrilateral from the given one using perpendicular bisectors -/
def constructNextQuadrilateral (Q : ConvexQuadrilateral) : ConvexQuadrilateral :=
  sorry  -- Definition of the construction process

/-- Two quadrilaterals are similar -/
def isSimilar (Q1 Q2 : ConvexQuadrilateral) : Prop :=
  sorry  -- Definition of similarity for quadrilaterals

theorem quadrilateral_similarity (Q1 : ConvexQuadrilateral) :
  let Q2 := constructNextQuadrilateral Q1
  let Q3 := constructNextQuadrilateral Q2
  isSimilar Q3 Q1 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_similarity_l381_38199


namespace NUMINAMATH_CALUDE_expression_evaluation_l381_38173

theorem expression_evaluation : 
  (1.2 : ℝ)^3 - (0.9 : ℝ)^3 / (1.2 : ℝ)^2 + 1.08 + (0.9 : ℝ)^2 = 3.11175 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l381_38173


namespace NUMINAMATH_CALUDE_solve_equations_l381_38172

/-- Solutions to the quadratic equation x^2 - 6x + 3 = 0 -/
def solutions_eq1 : Set ℝ := {3 + Real.sqrt 6, 3 - Real.sqrt 6}

/-- Solutions to the equation x(x-2) = x-2 -/
def solutions_eq2 : Set ℝ := {2, 1}

theorem solve_equations :
  (∀ x ∈ solutions_eq1, x^2 - 6*x + 3 = 0) ∧
  (∀ x ∈ solutions_eq2, x*(x-2) = x-2) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l381_38172


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l381_38129

theorem opposite_of_negative_five : 
  -((-5 : ℤ)) = (5 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l381_38129


namespace NUMINAMATH_CALUDE_exists_x_y_sequences_l381_38157

/-- The sequence a_n satisfying the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 4
  | 1 => 22
  | (n + 2) => 6 * a (n + 1) - a n

/-- Theorem stating the existence of sequences x_n and y_n satisfying the given property -/
theorem exists_x_y_sequences :
  ∃ (x y : ℕ → ℕ), ∀ n, a n = (y n ^ 2 + 7 : ℚ) / ((x n : ℚ) - y n) :=
sorry

end NUMINAMATH_CALUDE_exists_x_y_sequences_l381_38157


namespace NUMINAMATH_CALUDE_number_of_groups_l381_38153

def lunch_times : List ℕ := [10, 12, 15, 8, 16, 18, 19, 18, 20, 18, 18, 20, 28, 22, 25, 20, 15, 16, 21, 16]

def class_interval : ℕ := 4

theorem number_of_groups : 
  let min_time := lunch_times.minimum?
  let max_time := lunch_times.maximum?
  match min_time, max_time with
  | some min, some max => 
    (max - min) / class_interval + 1 = 6
  | _, _ => False
  := by sorry

end NUMINAMATH_CALUDE_number_of_groups_l381_38153


namespace NUMINAMATH_CALUDE_factorial_340_trailing_zeros_l381_38194

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

theorem factorial_340_trailing_zeros :
  trailingZeros 340 = 83 := by sorry

end NUMINAMATH_CALUDE_factorial_340_trailing_zeros_l381_38194


namespace NUMINAMATH_CALUDE_inequality_on_unit_circle_l381_38106

theorem inequality_on_unit_circle (a b c d : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a * b + c * d = 1)
  (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1) 
  (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) :
  (a * y₁ + b * y₂ + c * y₃ + d * y₄)^2 + (a * x₄ + b * x₃ + c * x₂ + d * x₁)^2 ≤ 
  2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := by
sorry

end NUMINAMATH_CALUDE_inequality_on_unit_circle_l381_38106


namespace NUMINAMATH_CALUDE_parabola_focus_and_directrix_l381_38108

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := x - 2 = (y - 3)^2

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2.25, 3)

-- Define the directrix of the parabola
def directrix : ℝ → Prop := λ x => x = 1.75

theorem parabola_focus_and_directrix :
  ∀ x y : ℝ, parabola_equation x y →
  (∃ p : ℝ × ℝ, p = focus ∧ 
   ∀ q : ℝ × ℝ, parabola_equation q.1 q.2 → 
   (q.1 - p.1)^2 + (q.2 - p.2)^2 = (q.1 - 1.75)^2) ∧
  (∀ q : ℝ × ℝ, parabola_equation q.1 q.2 → 
   ∃ r : ℝ, directrix r ∧ 
   (q.1 - focus.1)^2 + (q.2 - focus.2)^2 = (q.1 - r)^2) :=
by sorry


end NUMINAMATH_CALUDE_parabola_focus_and_directrix_l381_38108


namespace NUMINAMATH_CALUDE_sales_increase_price_reduction_for_target_profit_l381_38125

/-- Represents the Asian Games mascot badge sales scenario -/
structure BadgeSales where
  originalProfit : ℝ  -- Original profit per set
  originalSold : ℝ    -- Original number of sets sold per day
  profitReduction : ℝ -- Reduction in profit per set
  saleIncrease : ℝ    -- Increase in sales per $1 reduction

/-- Calculates the increase in sets sold given a profit reduction -/
def increasedSales (s : BadgeSales) : ℝ :=
  s.profitReduction * s.saleIncrease

/-- Calculates the daily profit given a price reduction -/
def dailyProfit (s : BadgeSales) (priceReduction : ℝ) : ℝ :=
  (s.originalProfit - priceReduction) * (s.originalSold + priceReduction * s.saleIncrease)

/-- Theorem stating the increase in sales when profit is reduced by $2 -/
theorem sales_increase (s : BadgeSales) (h : s.originalProfit = 40 ∧ s.originalSold = 20 ∧ s.profitReduction = 2 ∧ s.saleIncrease = 2) :
  increasedSales s = 4 := by sorry

/-- Theorem stating the price reduction needed for a daily profit of $1200 -/
theorem price_reduction_for_target_profit (s : BadgeSales) (h : s.originalProfit = 40 ∧ s.originalSold = 20 ∧ s.saleIncrease = 2) :
  ∃ x : ℝ, x = 20 ∧ dailyProfit s x = 1200 := by sorry

end NUMINAMATH_CALUDE_sales_increase_price_reduction_for_target_profit_l381_38125


namespace NUMINAMATH_CALUDE_solution_inequality_minimum_value_l381_38179

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - |x - 4|

-- Theorem for the solution of f(x) > 3
theorem solution_inequality (x : ℝ) : f x > 3 ↔ x > 2 := by sorry

-- Theorem for the minimum value of f(x)
theorem minimum_value : ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = 0 := by sorry

end NUMINAMATH_CALUDE_solution_inequality_minimum_value_l381_38179


namespace NUMINAMATH_CALUDE_bookshelf_problem_l381_38169

theorem bookshelf_problem (initial_books : ℕ) 
  (day1_borrow day1_return day2_borrow day2_return : ℤ) : 
  initial_books = 20 →
  day1_borrow = -3 →
  day1_return = 1 →
  day2_borrow = -1 →
  day2_return = 2 →
  (initial_books : ℤ) + day1_borrow + day1_return + day2_borrow + day2_return = 19 :=
by sorry

end NUMINAMATH_CALUDE_bookshelf_problem_l381_38169


namespace NUMINAMATH_CALUDE_difference_of_squares_403_397_l381_38174

theorem difference_of_squares_403_397 : 403^2 - 397^2 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_403_397_l381_38174


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l381_38100

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x < -1 ∨ x > 3/2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l381_38100


namespace NUMINAMATH_CALUDE_value_of_b_l381_38163

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 3) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l381_38163


namespace NUMINAMATH_CALUDE_arithmetic_equality_l381_38180

theorem arithmetic_equality : (30 - (3010 - 310)) + (3010 - (310 - 30)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l381_38180


namespace NUMINAMATH_CALUDE_range_f_a_2_range_a_two_zeros_l381_38175

-- Define the function f(x) = x^2 - ax - a + 3
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a + 3

-- Part 1: Range of f(x) when a = 2 and x ∈ [0, 3]
theorem range_f_a_2 :
  ∀ y ∈ Set.Icc 0 4, ∃ x ∈ Set.Icc 0 3, f 2 x = y :=
sorry

-- Part 2: Range of a when f(x) has two zeros x₁ and x₂ with x₁x₂ > 0
theorem range_a_two_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ * x₂ > 0) →
  a ∈ Set.Ioi (-6) ∪ Set.Ioo 2 3 :=
sorry

end NUMINAMATH_CALUDE_range_f_a_2_range_a_two_zeros_l381_38175


namespace NUMINAMATH_CALUDE_consecutive_integer_average_l381_38122

theorem consecutive_integer_average (c d : ℤ) : 
  (∀ i : ℕ, i < 7 → c + i > 0) →
  d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7 →
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 : ℚ) = c + 6 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integer_average_l381_38122


namespace NUMINAMATH_CALUDE_complex_number_problem_l381_38107

theorem complex_number_problem (a b c : ℂ) 
  (h_real : a.im = 0)
  (h_sum : a + b + c = 4)
  (h_prod_sum : a * b + b * c + c * a = 6)
  (h_prod : a * b * c = 8) :
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l381_38107


namespace NUMINAMATH_CALUDE_f_properties_l381_38140

def f (x : ℝ) := |2*x + 3| + |2*x - 1|

theorem f_properties :
  (∀ x : ℝ, f x < 10 ↔ x ∈ Set.Ioo (-3) 2) ∧
  (∀ a : ℝ, (∀ x : ℝ, f x ≥ |a - 1|) ↔ a ∈ Set.Icc (-2) 5) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l381_38140


namespace NUMINAMATH_CALUDE_retail_price_calculation_l381_38148

/-- Proves that the retail price of a machine is $120 given specific conditions --/
theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  wholesale_price = 90 →
  discount_rate = 0.1 →
  profit_rate = 0.2 →
  ∃ (retail_price : ℝ),
    retail_price * (1 - discount_rate) = wholesale_price * (1 + profit_rate) ∧
    retail_price = 120 := by
  sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l381_38148


namespace NUMINAMATH_CALUDE_K_is_perfect_square_l381_38120

def K (n : ℕ) : ℚ :=
  (4 * (10^(2*n) - 1) / 9) - (8 * (10^n - 1) / 9)

theorem K_is_perfect_square (n : ℕ) :
  ∃ (m : ℚ), K n = m^2 := by
sorry

end NUMINAMATH_CALUDE_K_is_perfect_square_l381_38120


namespace NUMINAMATH_CALUDE_sum_product_range_l381_38187

theorem sum_product_range (x y z : ℝ) (h : x + y + z = 3) :
  ∃ S : Set ℝ, S = Set.Iic (9/4) ∧
  ∀ t : ℝ, (∃ a b c : ℝ, a + b + c = 3 ∧ t = a*b + a*c + b*c) ↔ t ∈ S :=
sorry

end NUMINAMATH_CALUDE_sum_product_range_l381_38187


namespace NUMINAMATH_CALUDE_congruence_addition_l381_38188

theorem congruence_addition (a b c d m : ℤ) : 
  a ≡ b [ZMOD m] → c ≡ d [ZMOD m] → (a + c) ≡ (b + d) [ZMOD m] := by
  sorry

end NUMINAMATH_CALUDE_congruence_addition_l381_38188


namespace NUMINAMATH_CALUDE_more_science_than_math_books_l381_38160

def total_budget : ℕ := 500
def math_books : ℕ := 4
def math_book_price : ℕ := 20
def science_book_price : ℕ := 10
def art_book_price : ℕ := 20
def music_book_cost : ℕ := 160

theorem more_science_than_math_books :
  ∃ (science_books : ℕ) (art_books : ℕ),
    science_books > math_books ∧
    art_books = 2 * math_books ∧
    total_budget = math_books * math_book_price + science_books * science_book_price + 
                   art_books * art_book_price + music_book_cost ∧
    science_books - math_books = 6 :=
by sorry

end NUMINAMATH_CALUDE_more_science_than_math_books_l381_38160


namespace NUMINAMATH_CALUDE_smallest_square_multiplier_l381_38116

def y : ℕ := 2^4 * 3^2 * 4^3 * 5^3 * 6^2 * 7^3 * 8^3 * 9^2

theorem smallest_square_multiplier :
  (∀ k : ℕ, k > 0 ∧ k < 350 → ¬ ∃ m : ℕ, k * y = m^2) ∧
  ∃ m : ℕ, 350 * y = m^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_square_multiplier_l381_38116


namespace NUMINAMATH_CALUDE_lavinias_son_older_than_daughter_l381_38158

def katies_daughter_age : ℕ := 12

def lavinias_daughter_age (k : ℕ) : ℕ :=
  k / 3

def lavinias_son_age (k : ℕ) : ℕ :=
  2 * k

theorem lavinias_son_older_than_daughter :
  lavinias_son_age katies_daughter_age - lavinias_daughter_age katies_daughter_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_lavinias_son_older_than_daughter_l381_38158


namespace NUMINAMATH_CALUDE_parkway_soccer_boys_percentage_l381_38170

/-- Given the student population data for the fifth grade at Parkway Elementary School,
    prove that 86% of the students playing soccer are boys. -/
theorem parkway_soccer_boys_percentage
  (total_students : ℕ)
  (boys : ℕ)
  (soccer_players : ℕ)
  (girls_not_playing : ℕ)
  (h1 : total_students = 470)
  (h2 : boys = 300)
  (h3 : soccer_players = 250)
  (h4 : girls_not_playing = 135)
  : (boys_playing_soccer : ℚ) / soccer_players * 100 = 86 :=
by sorry

end NUMINAMATH_CALUDE_parkway_soccer_boys_percentage_l381_38170


namespace NUMINAMATH_CALUDE_servant_leaving_time_l381_38154

/-- The number of months a servant works before leaving -/
def months_worked : ℕ := 9

/-- The total yearly salary in rupees -/
def yearly_salary : ℕ := 200

/-- The amount received by the servant upon leaving in rupees -/
def leaving_amount : ℕ := 150

theorem servant_leaving_time :
  (months_worked : ℚ) / 12 * yearly_salary = leaving_amount := by
  sorry

#check servant_leaving_time

end NUMINAMATH_CALUDE_servant_leaving_time_l381_38154


namespace NUMINAMATH_CALUDE_isosceles_triangle_with_80_degree_angle_l381_38134

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We represent angles in degrees as natural numbers
  angle1 : ℕ
  angle2 : ℕ
  angle3 : ℕ
  -- Sum of angles is 180°
  sum_180 : angle1 + angle2 + angle3 = 180
  -- Two angles are equal (property of isosceles triangle)
  two_equal : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3)

-- Theorem statement
theorem isosceles_triangle_with_80_degree_angle 
  (t : IsoscelesTriangle) 
  (h : t.angle1 = 80 ∨ t.angle2 = 80 ∨ t.angle3 = 80) :
  (t.angle1 = 80 ∧ t.angle2 = 80 ∧ t.angle3 = 20) ∨
  (t.angle1 = 80 ∧ t.angle2 = 20 ∧ t.angle3 = 80) ∨
  (t.angle1 = 20 ∧ t.angle2 = 80 ∧ t.angle3 = 80) ∨
  (t.angle1 = 50 ∧ t.angle2 = 50 ∧ t.angle3 = 80) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_with_80_degree_angle_l381_38134


namespace NUMINAMATH_CALUDE_wedding_catering_budget_l381_38135

/-- Calculates the total catering budget for a wedding given the specified conditions. -/
theorem wedding_catering_budget 
  (total_guests : ℕ) 
  (steak_to_chicken_ratio : ℕ) 
  (steak_cost chicken_cost : ℕ) : 
  total_guests = 80 → 
  steak_to_chicken_ratio = 3 → 
  steak_cost = 25 → 
  chicken_cost = 18 → 
  (total_guests * steak_cost * steak_to_chicken_ratio + total_guests * chicken_cost) / (steak_to_chicken_ratio + 1) = 1860 := by
  sorry

#eval (80 * 25 * 3 + 80 * 18) / (3 + 1)

end NUMINAMATH_CALUDE_wedding_catering_budget_l381_38135


namespace NUMINAMATH_CALUDE_right_triangle_ratio_minimum_right_triangle_ratio_minimum_achievable_l381_38117

/-- For a right triangle with legs a and b and hypotenuse c, 
    (a² + b) / c² is always greater than or equal to 1 -/
theorem right_triangle_ratio_minimum (a b c : ℝ) 
  (h_positive : a ≥ 0 ∧ b > 0 ∧ c > 0) 
  (h_pythagoras : a^2 + b^2 = c^2) : 
  (a^2 + b) / c^2 ≥ 1 := by
  sorry

/-- The minimum value of (a² + b) / c² can be achieved -/
theorem right_triangle_ratio_minimum_achievable (ε : ℝ) (h_ε : ε > 0) :
  ∃ (a b c : ℝ), a ≥ 0 ∧ b > 0 ∧ c > 0 ∧ 
  a^2 + b^2 = c^2 ∧ 
  (a^2 + b) / c^2 < 1 + ε := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_minimum_right_triangle_ratio_minimum_achievable_l381_38117


namespace NUMINAMATH_CALUDE_production_and_salary_optimization_l381_38167

-- Define the variables and constants
def workday_minutes : ℕ := 8 * 60
def base_salary : ℕ := 100
def type_b_wage : ℚ := 2.5
def total_products : ℕ := 28

-- Define the production time equations
def production_equation_1 (x y : ℚ) : Prop := 6 * x + 4 * y = 170
def production_equation_2 (x y : ℚ) : Prop := 10 * x + 10 * y = 350

-- Define the time constraint
def time_constraint (x y : ℚ) (m : ℕ) : Prop :=
  x * m + y * (total_products - m) ≤ workday_minutes

-- Define the salary function
def salary (a : ℚ) (m : ℕ) : ℚ :=
  a * m + type_b_wage * (total_products - m) + base_salary

-- Theorem statement
theorem production_and_salary_optimization
  (x y : ℚ) (a : ℚ) (h_a : 2 < a ∧ a < 3) :
  (production_equation_1 x y ∧ production_equation_2 x y) →
  (x = 15 ∧ y = 20) ∧
  (∀ m : ℕ, m ≤ total_products →
    time_constraint x y m →
    (2 < a ∧ a < 2.5 → salary a 16 ≥ salary a m) ∧
    (a = 2.5 → salary a m = salary a 16) ∧
    (2.5 < a ∧ a < 3 → salary a 28 ≥ salary a m)) :=
sorry

end NUMINAMATH_CALUDE_production_and_salary_optimization_l381_38167


namespace NUMINAMATH_CALUDE_lucy_fish_count_l381_38144

/-- The number of fish Lucy wants to buy -/
def fish_to_buy : ℕ := 68

/-- The total number of fish Lucy would have after buying more -/
def total_fish_after : ℕ := 280

/-- The current number of fish in Lucy's aquarium -/
def current_fish : ℕ := total_fish_after - fish_to_buy

theorem lucy_fish_count : current_fish = 212 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l381_38144


namespace NUMINAMATH_CALUDE_smallest_power_of_ten_minus_one_divisible_by_37_l381_38137

theorem smallest_power_of_ten_minus_one_divisible_by_37 :
  (∃ n : ℕ, 10^n - 1 ≡ 0 [MOD 37]) ∧
  (∀ k : ℕ, k < 3 → ¬(10^k - 1 ≡ 0 [MOD 37])) ∧
  (10^3 - 1 ≡ 0 [MOD 37]) :=
sorry

end NUMINAMATH_CALUDE_smallest_power_of_ten_minus_one_divisible_by_37_l381_38137


namespace NUMINAMATH_CALUDE_triangle_vector_sum_l381_38111

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the triangle ABC and vectors a and b
variable (A B C : V) (a b : V)

-- State the theorem
theorem triangle_vector_sum (h1 : B - C = a) (h2 : C - A = b) : 
  A - B = -a - b := by sorry

end NUMINAMATH_CALUDE_triangle_vector_sum_l381_38111


namespace NUMINAMATH_CALUDE_cupcakes_for_classes_l381_38149

/-- The number of fourth-grade classes for which Jessa needs to make cupcakes -/
def num_fourth_grade_classes : ℕ := 3

/-- The number of students in each fourth-grade class -/
def students_per_fourth_grade : ℕ := 30

/-- The number of students in the P.E. class -/
def students_in_pe : ℕ := 50

/-- The total number of cupcakes Jessa needs to make -/
def total_cupcakes : ℕ := 140

theorem cupcakes_for_classes :
  num_fourth_grade_classes * students_per_fourth_grade + students_in_pe = total_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_cupcakes_for_classes_l381_38149


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l381_38105

/-- Proof of initial mixture volume given ratio changes after water addition -/
theorem initial_mixture_volume
  (initial_milk : ℝ)
  (initial_water : ℝ)
  (added_water : ℝ)
  (h1 : initial_milk / initial_water = 4)
  (h2 : added_water = 23)
  (h3 : initial_milk / (initial_water + added_water) = 1.125)
  : initial_milk + initial_water = 45 := by
  sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l381_38105


namespace NUMINAMATH_CALUDE_not_always_externally_tangent_l381_38138

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the positional relationship between two circles
inductive CircleRelationship
  | Disjoint
  | ExternallyTangent
  | Intersecting
  | InternallyTangent
  | Contained

-- Define a function to determine if two circles have no intersection
def noIntersection (c1 c2 : Circle) : Prop :=
  sorry

-- Define a function to determine the relationship between two circles
def circleRelationship (c1 c2 : Circle) : CircleRelationship :=
  sorry

-- Theorem statement
theorem not_always_externally_tangent (c1 c2 : Circle) :
  ¬(noIntersection c1 c2 → circleRelationship c1 c2 = CircleRelationship.ExternallyTangent) :=
sorry

end NUMINAMATH_CALUDE_not_always_externally_tangent_l381_38138


namespace NUMINAMATH_CALUDE_parabola_c_value_l381_38121

/-- A parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value :
  ∀ p : Parabola,
    p.x_coord (-3) = 2 →  -- vertex at (2, -3)
    p.x_coord (-1) = 7 →  -- passes through (7, -1)
    p.c = 53/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l381_38121


namespace NUMINAMATH_CALUDE_picnic_blanket_side_length_l381_38115

theorem picnic_blanket_side_length 
  (number_of_blankets : ℕ) 
  (folds : ℕ) 
  (total_folded_area : ℝ) 
  (L : ℝ) :
  number_of_blankets = 3 →
  folds = 4 →
  total_folded_area = 48 →
  (number_of_blankets : ℝ) * (L^2 / 2^folds) = total_folded_area →
  L = 16 :=
by sorry

end NUMINAMATH_CALUDE_picnic_blanket_side_length_l381_38115


namespace NUMINAMATH_CALUDE_triangle_side_length_l381_38161

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 1 → c = Real.sqrt 3 → A = π / 6 →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →
  b = 1 ∨ b = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l381_38161


namespace NUMINAMATH_CALUDE_lower_right_is_five_l381_38104

/-- Represents a 5x5 grid where each cell contains a digit from 1 to 5 -/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Checks if a given grid satisfies the Latin square property -/
def is_latin_square (g : Grid) : Prop :=
  ∀ i j k : Fin 5, 
    (∀ l : Fin 5, l ≠ j → g i l ≠ g i j) ∧ 
    (∀ l : Fin 5, l ≠ i → g l k ≠ g i k)

/-- The initial configuration of the grid -/
def initial_grid : Grid :=
  λ i j => 
    if i = 0 ∧ j = 0 then 1
    else if i = 0 ∧ j = 2 then 2
    else if i = 0 ∧ j = 4 then 3
    else if i = 1 ∧ j = 0 then 2
    else if i = 1 ∧ j = 1 then 3
    else if i = 1 ∧ j = 3 then 4
    else if i = 2 ∧ j = 1 then 4
    else if i = 2 ∧ j = 4 then 1
    else if i = 3 ∧ j = 0 then 3
    else if i = 3 ∧ j = 2 then 5
    else 0

/-- The theorem to be proved -/
theorem lower_right_is_five : 
  ∀ g : Grid, is_latin_square g → 
  (∀ i j, initial_grid i j ≠ 0 → g i j = initial_grid i j) → 
  g 4 4 = 5 := by sorry

end NUMINAMATH_CALUDE_lower_right_is_five_l381_38104


namespace NUMINAMATH_CALUDE_seating_probability_l381_38193

/-- The number of chairs in the row -/
def total_chairs : ℕ := 10

/-- The number of usable chairs -/
def usable_chairs : ℕ := total_chairs - 1

/-- The probability that Mary and James do not sit next to each other -/
def probability_not_adjacent : ℚ := 7/9

theorem seating_probability :
  (total_chairs : ℕ) = 10 →
  (usable_chairs : ℕ) = total_chairs - 1 →
  probability_not_adjacent = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_seating_probability_l381_38193


namespace NUMINAMATH_CALUDE_radian_measure_60_degrees_l381_38186

/-- The radian measure of a 60° angle is π/3. -/
theorem radian_measure_60_degrees :
  (60 * Real.pi / 180 : ℝ) = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_radian_measure_60_degrees_l381_38186


namespace NUMINAMATH_CALUDE_train_crossing_time_l381_38198

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 180 →
  train_speed_kmh = 72 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l381_38198


namespace NUMINAMATH_CALUDE_surfers_problem_l381_38195

/-- The number of surfers on the beach with fewer surfers -/
def x : ℕ := sorry

/-- The number of surfers on Malibu beach -/
def y : ℕ := sorry

/-- The total number of surfers on both beaches -/
def total : ℕ := 60

theorem surfers_problem :
  (y = 2 * x) ∧ (x + y = total) → x = 20 := by sorry

end NUMINAMATH_CALUDE_surfers_problem_l381_38195


namespace NUMINAMATH_CALUDE_expression_simplification_l381_38197

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2 - x) + 4*(2 + 3*x) - 5*(1 - 2*x) = 27*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l381_38197


namespace NUMINAMATH_CALUDE_worst_player_is_daughter_l381_38183

-- Define the set of players
inductive Player
| Father
| Sister
| Daughter
| Son

-- Define the gender type
inductive Gender
| Male
| Female

-- Define the generation type
inductive Generation
| Older
| Younger

-- Function to get the gender of a player
def gender : Player → Gender
| Player.Father => Gender.Male
| Player.Sister => Gender.Female
| Player.Daughter => Gender.Female
| Player.Son => Gender.Male

-- Function to get the generation of a player
def generation : Player → Generation
| Player.Father => Generation.Older
| Player.Sister => Generation.Older
| Player.Daughter => Generation.Younger
| Player.Son => Generation.Younger

-- Function to determine if two players could be twins
def couldBeTwins : Player → Player → Prop
| Player.Daughter, Player.Son => True
| Player.Son, Player.Daughter => True
| _, _ => False

-- Theorem statement
theorem worst_player_is_daughter :
  ∀ (worst best : Player),
    (∃ twin : Player, couldBeTwins worst twin ∧ gender twin = gender best) →
    generation worst ≠ generation best →
    worst = Player.Daughter :=
sorry

end NUMINAMATH_CALUDE_worst_player_is_daughter_l381_38183


namespace NUMINAMATH_CALUDE_triangle_inequality_squared_l381_38139

theorem triangle_inequality_squared (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b) : 
  a^2 + b^2 + c^2 < 2*(a*b + b*c + a*c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_squared_l381_38139


namespace NUMINAMATH_CALUDE_equations_have_same_solutions_l381_38130

def daniels_equation (x : ℝ) : Prop := |x - 8| = 3

def emmas_equation (x : ℝ) : Prop := x^2 - 16*x + 55 = 0

theorem equations_have_same_solutions :
  (∀ x : ℝ, daniels_equation x ↔ emmas_equation x) :=
sorry

end NUMINAMATH_CALUDE_equations_have_same_solutions_l381_38130


namespace NUMINAMATH_CALUDE_quadratic_root_two_l381_38126

theorem quadratic_root_two (a b c : ℝ) : 
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = 2) ↔ 4 * a + 2 * b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_two_l381_38126


namespace NUMINAMATH_CALUDE_eight_digit_divisible_by_11_l381_38156

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def digit (d : ℕ) : Prop :=
  d ≥ 0 ∧ d < 10

theorem eight_digit_divisible_by_11 (m : ℕ) :
  digit m →
  is_divisible_by_11 (73400000 + m * 100000 + 8527) →
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_eight_digit_divisible_by_11_l381_38156


namespace NUMINAMATH_CALUDE_faster_train_speed_faster_train_speed_is_10_l381_38162

theorem faster_train_speed 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (speed_ratio : ℝ) : ℝ :=
  let slower_speed := (2 * train_length) / (crossing_time * (1 + speed_ratio))
  let faster_speed := speed_ratio * slower_speed
  faster_speed

theorem faster_train_speed_is_10 :
  faster_train_speed 200 30 3 = 10 := by sorry

end NUMINAMATH_CALUDE_faster_train_speed_faster_train_speed_is_10_l381_38162


namespace NUMINAMATH_CALUDE_positive_real_properties_l381_38145

theorem positive_real_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 - b^2 = 1 → a - b < 1) ∧
  (a > b + 1 → a^2 > b^2 + 1) := by
sorry

end NUMINAMATH_CALUDE_positive_real_properties_l381_38145


namespace NUMINAMATH_CALUDE_percentage_relation_l381_38151

theorem percentage_relation (x y : ℝ) (h : 0.6 * (x - y) = 0.2 * (x + y)) : y = 0.5 * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l381_38151


namespace NUMINAMATH_CALUDE_circle_center_l381_38164

/-- The center of the circle defined by x^2 + y^2 + 2y = 1 is (0, -1) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 + 2*y = 1) → (0, -1) = (0, -1) := by sorry

end NUMINAMATH_CALUDE_circle_center_l381_38164


namespace NUMINAMATH_CALUDE_equal_share_theorem_l381_38168

def money_sharing (emani_money : ℕ) (difference : ℕ) : ℕ :=
  let howard_money := emani_money - difference
  let total_money := emani_money + howard_money
  total_money / 2

theorem equal_share_theorem (emani_money : ℕ) (difference : ℕ) 
  (h1 : emani_money = 150) 
  (h2 : difference = 30) : 
  money_sharing emani_money difference = 135 := by
sorry

end NUMINAMATH_CALUDE_equal_share_theorem_l381_38168


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l381_38196

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x > 0, x^2 - 2*x + 1 < 0) ↔ (∀ x > 0, x^2 - 2*x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l381_38196


namespace NUMINAMATH_CALUDE_circle_circumference_with_inscribed_rectangle_l381_38177

theorem circle_circumference_with_inscribed_rectangle :
  let rectangle_width : ℝ := 9
  let rectangle_height : ℝ := 12
  let diagonal : ℝ := (rectangle_width^2 + rectangle_height^2).sqrt
  let diameter : ℝ := diagonal
  let circumference : ℝ := π * diameter
  circumference = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_with_inscribed_rectangle_l381_38177


namespace NUMINAMATH_CALUDE_no_real_solutions_l381_38181

theorem no_real_solutions :
  ∀ x : ℝ, (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 20 * x^9 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l381_38181


namespace NUMINAMATH_CALUDE_cookies_per_bag_l381_38102

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : total_cookies = 75)
  (h2 : num_bags = 25)
  (h3 : total_cookies = num_bags * cookies_per_bag) :
  cookies_per_bag = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l381_38102


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l381_38142

theorem rectangular_prism_volume (l w h : ℝ) 
  (face1 : l * w = 10)
  (face2 : w * h = 14)
  (face3 : l * h = 35) :
  l * w * h = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l381_38142


namespace NUMINAMATH_CALUDE_spelling_bee_contestants_l381_38189

theorem spelling_bee_contestants (total : ℕ) 
  (h1 : (total : ℝ) * (1 - 0.6) * 0.25 = 30) : total = 300 := by
  sorry

end NUMINAMATH_CALUDE_spelling_bee_contestants_l381_38189


namespace NUMINAMATH_CALUDE_distance_inequality_l381_38131

theorem distance_inequality (x y : ℝ) :
  Real.sqrt ((x + 4)^2 + (y + 2)^2) + Real.sqrt ((x - 5)^2 + (y + 4)^2) ≤
  Real.sqrt ((x - 2)^2 + (y - 6)^2) + Real.sqrt ((x - 5)^2 + (y - 6)^2) + 20 := by
  sorry

end NUMINAMATH_CALUDE_distance_inequality_l381_38131


namespace NUMINAMATH_CALUDE_gmat_exam_correct_answers_l381_38185

theorem gmat_exam_correct_answers 
  (total : ℕ) 
  (first_correct : ℕ) 
  (second_correct : ℕ) 
  (neither_correct : ℕ) 
  (h1 : first_correct = (85 * total) / 100)
  (h2 : second_correct = (80 * total) / 100)
  (h3 : neither_correct = (5 * total) / 100)
  : ((first_correct + second_correct - (total - neither_correct)) * 100) / total = 70 :=
by sorry

end NUMINAMATH_CALUDE_gmat_exam_correct_answers_l381_38185


namespace NUMINAMATH_CALUDE_bucket_volume_proof_l381_38165

/-- The volume of water (in liters) that Tap A runs per minute -/
def tap_a_rate : ℝ := 3

/-- The time (in minutes) it takes Tap B to fill 1/3 of the bucket -/
def tap_b_third_time : ℝ := 20

/-- The time (in minutes) it takes both taps working together to fill the bucket -/
def combined_time : ℝ := 10

/-- The total volume of the bucket in liters -/
def bucket_volume : ℝ := 36

theorem bucket_volume_proof :
  let tap_b_rate := bucket_volume / (3 * tap_b_third_time)
  tap_a_rate + tap_b_rate = bucket_volume / combined_time := by
  sorry

end NUMINAMATH_CALUDE_bucket_volume_proof_l381_38165


namespace NUMINAMATH_CALUDE_water_displacement_l381_38113

theorem water_displacement (tank_length tank_width : ℝ) 
  (water_level_rise : ℝ) (num_men : ℕ) :
  tank_length = 40 ∧ 
  tank_width = 20 ∧ 
  water_level_rise = 0.25 ∧ 
  num_men = 50 → 
  (tank_length * tank_width * water_level_rise) / num_men = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_displacement_l381_38113


namespace NUMINAMATH_CALUDE_arrangements_with_restriction_l381_38184

theorem arrangements_with_restriction (n : ℕ) (h : n = 6) :
  (n - 1) * Nat.factorial (n - 1) = 600 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_with_restriction_l381_38184


namespace NUMINAMATH_CALUDE_train_distance_l381_38152

/-- Proves that a train traveling at 10 m/s for 8 seconds covers 80 meters. -/
theorem train_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 10)
  (h2 : time = 8)
  (h3 : distance = speed * time) : 
  distance = 80 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l381_38152


namespace NUMINAMATH_CALUDE_eighth_term_matchsticks_l381_38146

/-- The number of matchsticks in the nth term of the sequence -/
def matchsticks (n : ℕ) : ℕ := (n + 1) * 3

/-- Theorem: The number of matchsticks in the eighth term is 27 -/
theorem eighth_term_matchsticks : matchsticks 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_matchsticks_l381_38146


namespace NUMINAMATH_CALUDE_cube_difference_152_l381_38178

theorem cube_difference_152 : ∃! n : ℤ, 
  (∃ a : ℤ, a > 0 ∧ n - 76 = a^3) ∧ 
  (∃ b : ℤ, b > 0 ∧ n + 76 = b^3) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_difference_152_l381_38178


namespace NUMINAMATH_CALUDE_painting_area_is_5400_l381_38143

/-- The area of a painting inside a uniform frame -/
def painting_area (outer_width : ℝ) (outer_height : ℝ) (frame_width : ℝ) : ℝ :=
  (outer_width - 2 * frame_width) * (outer_height - 2 * frame_width)

/-- Theorem: The area of the painting inside the frame is 5400 cm² -/
theorem painting_area_is_5400 :
  painting_area 90 120 15 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_painting_area_is_5400_l381_38143


namespace NUMINAMATH_CALUDE_parallel_angles_theorem_l381_38103

/-- Given two angles A and B where the sides of A are parallel to the sides of B, 
    prove that if B = 3A - 60°, then B is either 30° or 120° -/
theorem parallel_angles_theorem (A B : ℝ) : 
  (B = 3 * A - 60) → (B = 30 ∨ B = 120) := by
  sorry

end NUMINAMATH_CALUDE_parallel_angles_theorem_l381_38103


namespace NUMINAMATH_CALUDE_two_points_imply_line_in_plane_l381_38141

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define what it means for a point to be on a line
variable (on_line : Point → Line → Prop)

-- Define what it means for a point to be within a plane
variable (in_plane : Point → Plane → Prop)

-- Define what it means for a line to be within a plane
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem two_points_imply_line_in_plane 
  (a : Line) (α : Plane) (A B : Point) 
  (h1 : A ≠ B) 
  (h2 : on_line A a) 
  (h3 : on_line B a) 
  (h4 : in_plane A α) 
  (h5 : in_plane B α) : 
  line_in_plane a α :=
sorry

end NUMINAMATH_CALUDE_two_points_imply_line_in_plane_l381_38141


namespace NUMINAMATH_CALUDE_muscovy_duck_count_muscovy_duck_count_proof_l381_38123

theorem muscovy_duck_count : ℕ → ℕ → ℕ → Prop :=
  fun muscovy cayuga khaki =>
    muscovy = cayuga + 4 ∧
    muscovy = 2 * cayuga + khaki + 3 ∧
    muscovy + cayuga + khaki = 90 →
    muscovy = 89

-- The proof is omitted
theorem muscovy_duck_count_proof : muscovy_duck_count 89 85 6 :=
  sorry

end NUMINAMATH_CALUDE_muscovy_duck_count_muscovy_duck_count_proof_l381_38123


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l381_38109

theorem smallest_number_of_students (grade12 grade11 grade10 : ℕ) : 
  grade12 > 0 ∧ grade11 > 0 ∧ grade10 > 0 →
  grade12 * 3 = grade10 * 4 →
  grade12 * 5 = grade11 * 7 →
  grade11 * 9 = grade10 * 10 →
  grade12 + grade11 + grade10 ≥ 66 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l381_38109


namespace NUMINAMATH_CALUDE_calculation_proof_l381_38132

theorem calculation_proof : (-1)^2023 + 6 * Real.cos (π / 3) + (Real.pi - 3.14)^0 - Real.sqrt 16 = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l381_38132


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l381_38192

/-- Represents a square pattern of tiles -/
structure TilePattern :=
  (size : ℕ)
  (black_tiles : ℕ)
  (white_tiles : ℕ)

/-- Adds a border of white tiles to a given pattern -/
def add_border (pattern : TilePattern) : TilePattern :=
  { size := pattern.size + 2,
    black_tiles := pattern.black_tiles,
    white_tiles := pattern.white_tiles + (pattern.size + 2)^2 - pattern.size^2 }

/-- The ratio of black tiles to white tiles -/
def tile_ratio (pattern : TilePattern) : ℚ :=
  pattern.black_tiles / (pattern.black_tiles + pattern.white_tiles)

theorem extended_pattern_ratio :
  let initial_pattern : TilePattern := ⟨6, 12, 24⟩
  let extended_pattern := add_border initial_pattern
  tile_ratio extended_pattern = 3/13 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l381_38192


namespace NUMINAMATH_CALUDE_bees_on_first_day_l381_38176

/-- Given that Mrs. Hilt saw some bees on the first day and 3 times as many on the second day,
    counting 432 bees on the second day, prove that she saw 144 bees on the first day. -/
theorem bees_on_first_day (first_day : ℕ) (second_day : ℕ) : 
  second_day = 3 * first_day → second_day = 432 → first_day = 144 := by
  sorry

end NUMINAMATH_CALUDE_bees_on_first_day_l381_38176


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_not_equal_l381_38190

def A : Set ℝ := {x : ℝ | |x - 2| ≤ 2}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

theorem complement_A_intersect_B_not_equal :
  (Aᶜ ∪ Bᶜ) ≠ Set.univ ∧
  (Aᶜ ∪ Bᶜ) ≠ {x : ℝ | x ≠ 0} ∧
  (Aᶜ ∪ Bᶜ) ≠ {0} :=
by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_not_equal_l381_38190


namespace NUMINAMATH_CALUDE_parallelogram_center_not_axis_symmetric_l381_38124

-- Define the shape types
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | Rectangle
  | Rhombus

-- Define the symmetry properties
def isAxisSymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.Parallelogram => false
  | Shape.Rectangle => true
  | Shape.Rhombus => true

def isCenterSymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => false
  | Shape.Parallelogram => true
  | Shape.Rectangle => true
  | Shape.Rhombus => true

-- Theorem statement
theorem parallelogram_center_not_axis_symmetric :
  ∃ (s : Shape), isCenterSymmetric s ∧ ¬isAxisSymmetric s ∧
  ∀ (t : Shape), t ≠ s → ¬(isCenterSymmetric t ∧ ¬isAxisSymmetric t) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_center_not_axis_symmetric_l381_38124


namespace NUMINAMATH_CALUDE_equation_is_linear_and_has_solution_l381_38150

-- Define the equation
def equation (x : ℝ) : Prop := 1 - x = -3

-- State the theorem
theorem equation_is_linear_and_has_solution :
  (∃ a b : ℝ, ∀ x, equation x ↔ a * x + b = 0) ∧ 
  equation 4 := by sorry

end NUMINAMATH_CALUDE_equation_is_linear_and_has_solution_l381_38150


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l381_38119

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 1 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l381_38119


namespace NUMINAMATH_CALUDE_largest_size_percentage_longer_than_smallest_l381_38114

-- Define the shoe size range
def min_size : ℕ := 8
def max_size : ℕ := 17

-- Define the length increase per size
def length_increase_per_size : ℚ := 1 / 5

-- Define the length of size 15 shoe
def size_15_length : ℚ := 21 / 2  -- 10.4 as a rational number

-- Function to calculate shoe length given size
def shoe_length (size : ℕ) : ℚ :=
  size_15_length + (size - 15 : ℚ) * length_increase_per_size

-- Theorem statement
theorem largest_size_percentage_longer_than_smallest :
  (shoe_length max_size - shoe_length min_size) / shoe_length min_size = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_largest_size_percentage_longer_than_smallest_l381_38114


namespace NUMINAMATH_CALUDE_arrange_four_math_four_history_l381_38191

/-- The number of ways to arrange books on a shelf --/
def arrange_books (n_math : ℕ) (n_history : ℕ) : ℕ :=
  if n_math ≥ 2 then
    n_math * (n_math - 1) * (n_math + n_history - 2).factorial
  else
    0

/-- Theorem: Arranging 4 math books and 4 history books with math books on both ends --/
theorem arrange_four_math_four_history :
  arrange_books 4 4 = 8640 := by
  sorry

end NUMINAMATH_CALUDE_arrange_four_math_four_history_l381_38191


namespace NUMINAMATH_CALUDE_largest_satisfying_number_l381_38112

/-- Returns the leading digit of a positive integer -/
def leadingDigit (n : ℕ) : ℕ :=
  if n < 10 then n else leadingDigit (n / 10)

/-- Returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Checks if a number satisfies the condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  n > 0 ∧ n = leadingDigit n * sumOfDigits n

theorem largest_satisfying_number :
  satisfiesCondition 48 ∧ ∀ m : ℕ, m > 48 → ¬satisfiesCondition m :=
sorry

end NUMINAMATH_CALUDE_largest_satisfying_number_l381_38112


namespace NUMINAMATH_CALUDE_car_distance_theorem_l381_38147

/-- Calculates the total distance travelled by a car with increasing speed over a given number of hours -/
def totalDistance (initialDistance : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  let distanceList := List.range hours |>.map (fun h => initialDistance + h * speedIncrease)
  distanceList.sum

/-- Theorem stating that a car with given initial speed and speed increase travels 546 km in 12 hours -/
theorem car_distance_theorem :
  totalDistance 35 2 12 = 546 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l381_38147


namespace NUMINAMATH_CALUDE_x_fourth_gt_x_minus_half_l381_38136

theorem x_fourth_gt_x_minus_half (x : ℝ) : x^4 - x + (1/2 : ℝ) > 0 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_gt_x_minus_half_l381_38136


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l381_38133

/-- The line x = k intersects the parabola x = -3y^2 + 2y + 7 at exactly one point if and only if k = 22/3 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! y : ℝ, k = -3 * y^2 + 2 * y + 7) ↔ k = 22/3 := by
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l381_38133


namespace NUMINAMATH_CALUDE_floor_of_expression_equals_32_l381_38101

theorem floor_of_expression_equals_32 :
  ⌊(1 + (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 4) / 
     (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 8 + 4))^10⌋ = 32 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_expression_equals_32_l381_38101


namespace NUMINAMATH_CALUDE_root_difference_zero_l381_38182

/-- The nonnegative difference between the roots of x^2 + 40x + 300 = -100 is 0 -/
theorem root_difference_zero : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 40*x + 300 + 100
  let roots := {x : ℝ | f x = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_zero_l381_38182


namespace NUMINAMATH_CALUDE_original_average_l381_38159

theorem original_average (n : ℕ) (a : ℝ) (h1 : n = 15) (h2 : (n * (a + 12)) / n = 52) : a = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_average_l381_38159


namespace NUMINAMATH_CALUDE_digit_sum_of_special_number_l381_38128

theorem digit_sum_of_special_number : 
  ∀ (x : ℕ) (x' : ℕ) (y : ℕ),
  10000 ≤ x ∧ x < 100000 →  -- x is a five-digit number
  1000 ≤ x' ∧ x' < 10000 →  -- x' is a four-digit number
  0 ≤ y ∧ y < 10 →          -- y is a single digit
  x = 10 * x' + y →         -- x' is x with the ones digit removed
  x + x' = 52713 →          -- given condition
  (x / 10000) + ((x / 1000) % 10) + ((x / 100) % 10) + ((x / 10) % 10) + (x % 10) = 23 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_of_special_number_l381_38128


namespace NUMINAMATH_CALUDE_figurine_cost_l381_38155

/-- The cost of a single figurine given Annie's purchases -/
theorem figurine_cost (num_tvs : ℕ) (tv_cost : ℕ) (num_figurines : ℕ) (total_spent : ℕ) : 
  num_tvs = 5 → 
  tv_cost = 50 → 
  num_figurines = 10 → 
  total_spent = 260 → 
  (total_spent - num_tvs * tv_cost) / num_figurines = 1 := by
sorry

end NUMINAMATH_CALUDE_figurine_cost_l381_38155


namespace NUMINAMATH_CALUDE_f_equals_g_l381_38127

theorem f_equals_g (f g : Nat → Nat)
  (h1 : ∀ n : Nat, n > 0 → f (g n) = f n + 1)
  (h2 : ∀ n : Nat, n > 0 → g (f n) = g n + 1) :
  ∀ n : Nat, n > 0 → f n = g n :=
by sorry

end NUMINAMATH_CALUDE_f_equals_g_l381_38127


namespace NUMINAMATH_CALUDE_floor_length_l381_38110

/-- Represents the properties of a rectangular floor -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  paintCost : ℝ
  paintRate : ℝ

/-- The length of the floor is 200% more than its breadth -/
def lengthCondition (floor : RectangularFloor) : Prop :=
  floor.length = 3 * floor.breadth

/-- The cost to paint the floor is Rs. 300 -/
def costCondition (floor : RectangularFloor) : Prop :=
  floor.paintCost = 300

/-- The painting rate is Rs. 5 per sq m -/
def rateCondition (floor : RectangularFloor) : Prop :=
  floor.paintRate = 5

/-- Theorem stating the length of the floor -/
theorem floor_length (floor : RectangularFloor) 
  (h1 : lengthCondition floor) 
  (h2 : costCondition floor) 
  (h3 : rateCondition floor) : 
  floor.length = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_length_l381_38110


namespace NUMINAMATH_CALUDE_positive_sum_from_absolute_difference_l381_38166

theorem positive_sum_from_absolute_difference (a b : ℝ) : 
  b - |a| > 0 → a + b > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_from_absolute_difference_l381_38166


namespace NUMINAMATH_CALUDE_triangle_area_proof_l381_38171

theorem triangle_area_proof (a b : ℝ × ℝ) : 
  a = (2, -3) → b = (4, -1) → 
  abs (a.1 * b.2 - a.2 * b.1) / 2 = 5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l381_38171


namespace NUMINAMATH_CALUDE_smallest_value_fraction_achievable_value_l381_38118

theorem smallest_value_fraction (a b : ℕ) (h1 : a > b) (h2 : b > 0) : 
  (a + 2*b : ℚ)/(a - b) + (a - b : ℚ)/(a + 2*b) ≥ 10/3 :=
sorry

theorem achievable_value (a b : ℕ) (h1 : a > b) (h2 : b > 0) : 
  ∃ (a b : ℕ), a > b ∧ b > 0 ∧ (a + 2*b : ℚ)/(a - b) + (a - b : ℚ)/(a + 2*b) = 10/3 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_fraction_achievable_value_l381_38118
