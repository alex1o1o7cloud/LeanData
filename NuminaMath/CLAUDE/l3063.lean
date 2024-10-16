import Mathlib

namespace NUMINAMATH_CALUDE_adams_friends_strawberries_l3063_306348

/-- The number of strawberries Adam's friends ate -/
def friends_strawberries (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Proof that Adam's friends ate 2 strawberries -/
theorem adams_friends_strawberries :
  friends_strawberries 35 33 = 2 := by
  sorry

end NUMINAMATH_CALUDE_adams_friends_strawberries_l3063_306348


namespace NUMINAMATH_CALUDE_range_of_expression_l3063_306372

theorem range_of_expression (x y : ℝ) 
  (h1 : x ≥ 0) 
  (h2 : y ≥ x) 
  (h3 : 4 * x + 3 * y ≤ 12) : 
  3 ≤ (x + 2 * y + 3) / (x + 1) ∧ (x + 2 * y + 3) / (x + 1) ≤ 11 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l3063_306372


namespace NUMINAMATH_CALUDE_problem_statement_l3063_306314

theorem problem_statement (x y : ℝ) (h : |x - 1/2| + Real.sqrt (y^2 - 1) = 0) : 
  |x| + |y| = 3/2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3063_306314


namespace NUMINAMATH_CALUDE_complex_number_line_l3063_306312

theorem complex_number_line (z : ℂ) (h : 2 * (1 + Complex.I) * z = 1 - Complex.I) :
  z.im = -1/2 * z.re := by
  sorry

end NUMINAMATH_CALUDE_complex_number_line_l3063_306312


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3063_306373

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation x^2 - 3x + 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 1

/-- Theorem stating that f is a quadratic equation in one variable -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l3063_306373


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3063_306345

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3063_306345


namespace NUMINAMATH_CALUDE_percent_of_percent_l3063_306385

theorem percent_of_percent (x : ℝ) :
  (20 / 100) * (x / 100) = 80 / 100 → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l3063_306385


namespace NUMINAMATH_CALUDE_relative_error_comparison_l3063_306320

/-- Given two measurements and their respective errors, this theorem states that
    the relative error of the second measurement is less than that of the first. -/
theorem relative_error_comparison
  (measurement1 : ℝ) (error1 : ℝ) (measurement2 : ℝ) (error2 : ℝ)
  (h1 : measurement1 = 0.15)
  (h2 : error1 = 0.03)
  (h3 : measurement2 = 125)
  (h4 : error2 = 0.25)
  : error2 / measurement2 < error1 / measurement1 := by
  sorry

end NUMINAMATH_CALUDE_relative_error_comparison_l3063_306320


namespace NUMINAMATH_CALUDE_fibonacci_triangle_isosceles_l3063_306357

def fibonacci_set : Set ℕ := {2, 3, 5, 8, 13, 21, 34, 55, 89, 144}

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem fibonacci_triangle_isosceles :
  ∀ a b c : ℕ,
    a ∈ fibonacci_set →
    b ∈ fibonacci_set →
    c ∈ fibonacci_set →
    is_triangle a b c →
    is_isosceles a b c :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_triangle_isosceles_l3063_306357


namespace NUMINAMATH_CALUDE_car_insurance_cost_l3063_306361

theorem car_insurance_cost (nancy_percentage : ℝ) (nancy_annual_payment : ℝ) :
  nancy_percentage = 0.40 →
  nancy_annual_payment = 384 →
  (nancy_annual_payment / nancy_percentage) / 12 = 80 := by
sorry

end NUMINAMATH_CALUDE_car_insurance_cost_l3063_306361


namespace NUMINAMATH_CALUDE_vertex_locus_is_parabola_l3063_306369

theorem vertex_locus_is_parabola (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  let vertex (t : ℝ) := (-(t / (2 * a)), a * (-(t / (2 * a)))^2 + t * (-(t / (2 * a))) + c)
  ∃ f : ℝ → ℝ, (∀ x, f x = -a * x^2 + c) ∧
    (∀ t, (vertex t).2 = f (vertex t).1) :=
by sorry

end NUMINAMATH_CALUDE_vertex_locus_is_parabola_l3063_306369


namespace NUMINAMATH_CALUDE_mens_wages_l3063_306328

/-- Given that 5 men are equal to W women, W women are equal to 8 boys,
    and the total earnings of all (5 men + W women + 8 boys) is 180 Rs,
    prove that each man's wage is 36 Rs. -/
theorem mens_wages (W : ℕ) : 
  (5 : ℕ) = W → -- 5 men are equal to W women
  W = 8 → -- W women are equal to 8 boys
  (5 : ℕ) * x + W * x + 8 * x = 180 → -- total earnings equation
  x = 36 := by sorry

end NUMINAMATH_CALUDE_mens_wages_l3063_306328


namespace NUMINAMATH_CALUDE_tangent_line_circle_l3063_306319

theorem tangent_line_circle (r : ℝ) (hr : r > 0) :
  (∀ x y : ℝ, x + y = r → x^2 + y^2 = r → (∀ x' y' : ℝ, x' + y' = r → x'^2 + y'^2 ≤ r)) →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_circle_l3063_306319


namespace NUMINAMATH_CALUDE_area_of_absolute_value_graph_l3063_306331

/-- The area enclosed by the graph of |x| + |3y| = 9 is 54 square units -/
theorem area_of_absolute_value_graph : 
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ |x| + |3 * y|
  ∃ S : Set (ℝ × ℝ), S = {p : ℝ × ℝ | f p = 9} ∧ MeasureTheory.volume S = 54 := by
  sorry

end NUMINAMATH_CALUDE_area_of_absolute_value_graph_l3063_306331


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l3063_306378

/-- Given a paint mixture with a ratio of blue:green:yellow as 4:3:5,
    if 15 quarts of yellow paint are used, then 9 quarts of green paint should be used. -/
theorem paint_mixture_ratio (blue green yellow : ℚ) :
  blue / green = 4 / 3 →
  green / yellow = 3 / 5 →
  yellow = 15 →
  green = 9 := by
sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l3063_306378


namespace NUMINAMATH_CALUDE_pie_chart_shows_percentage_relation_l3063_306346

/-- Represents different types of statistical graphs -/
inductive StatGraph
  | PieChart
  | BarGraph
  | LineGraph
  | Histogram

/-- Defines the property of showing percentage of a part in relation to the whole -/
def shows_percentage_relation (g : StatGraph) : Prop :=
  match g with
  | StatGraph.PieChart => true
  | _ => false

/-- Theorem stating that the Pie chart is the graph that shows percentage relation -/
theorem pie_chart_shows_percentage_relation :
  ∀ (g : StatGraph), shows_percentage_relation g ↔ g = StatGraph.PieChart :=
by
  sorry

end NUMINAMATH_CALUDE_pie_chart_shows_percentage_relation_l3063_306346


namespace NUMINAMATH_CALUDE_rate_of_change_f_l3063_306343

/-- The function f(x) = 2x^2 + 5 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 5

/-- Theorem: For the function f(x) = 2x^2 + 5, given the points (1, 7) and (1 + Δx, 7 + Δy) 
    on the graph of f, the rate of change Δy / Δx is equal to 2Δx + 4 -/
theorem rate_of_change_f (Δx : ℝ) (Δy : ℝ) 
  (h1 : f 1 = 7)
  (h2 : f (1 + Δx) = 7 + Δy)
  (h3 : Δx ≠ 0) :
  Δy / Δx = 2 * Δx + 4 :=
sorry

end NUMINAMATH_CALUDE_rate_of_change_f_l3063_306343


namespace NUMINAMATH_CALUDE_table_length_proof_l3063_306396

theorem table_length_proof (table_width : ℝ) (sheet_width sheet_height : ℝ) 
  (h1 : table_width = 80)
  (h2 : sheet_width = 8)
  (h3 : sheet_height = 5)
  (h4 : ∃ n : ℕ, n * 1 = table_width - sheet_width ∧ n * 1 = table_width - sheet_height) :
  ∃ x : ℝ, x = 77 ∧ x = table_width - (sheet_width - sheet_height) := by
sorry

end NUMINAMATH_CALUDE_table_length_proof_l3063_306396


namespace NUMINAMATH_CALUDE_ellipse_tangent_perpendicular_l3063_306392

/-- Two ellipses with equations x²/a² + y²/b² = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

def is_on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def is_tangent_to_ellipse (l : Line) (e : Ellipse) (p : Point) : Prop :=
  is_on_ellipse p e ∧ l.m = -p.x * e.b^2 / (p.y * e.a^2)

def intersect_line_ellipse (l : Line) (e : Ellipse) : Set Point :=
  {p : Point | is_on_ellipse p e ∧ p.y = l.m * p.x + l.c}

def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.m * l2.m = -1

theorem ellipse_tangent_perpendicular 
  (e1 e2 : Ellipse) 
  (p : Point) 
  (l : Line) 
  (a b q : Point) :
  e1.a^2 - e1.b^2 = e2.a^2 - e2.b^2 →  -- shared foci condition
  is_tangent_to_ellipse l e1 p →
  a ∈ intersect_line_ellipse l e2 →
  b ∈ intersect_line_ellipse l e2 →
  is_tangent_to_ellipse (Line.mk ((q.y - a.y) / (q.x - a.x)) (q.y - (q.y - a.y) / (q.x - a.x) * q.x)) e2 a →
  is_tangent_to_ellipse (Line.mk ((q.y - b.y) / (q.x - b.x)) (q.y - (q.y - b.y) / (q.x - b.x) * q.x)) e2 b →
  are_perpendicular 
    (Line.mk ((q.y - p.y) / (q.x - p.x)) (q.y - (q.y - p.y) / (q.x - p.x) * q.x))
    (Line.mk ((b.y - a.y) / (b.x - a.x)) (b.y - (b.y - a.y) / (b.x - a.x) * b.x)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_perpendicular_l3063_306392


namespace NUMINAMATH_CALUDE_intersection_M_N_l3063_306356

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3063_306356


namespace NUMINAMATH_CALUDE_water_evaporation_period_l3063_306317

theorem water_evaporation_period (initial_amount : Real) (daily_rate : Real) (evaporation_percentage : Real) : 
  initial_amount > 0 → 
  daily_rate > 0 → 
  evaporation_percentage > 0 → 
  evaporation_percentage < 100 →
  initial_amount = 40 →
  daily_rate = 0.01 →
  evaporation_percentage = 0.5 →
  (initial_amount * evaporation_percentage / 100) / daily_rate = 20 := by
sorry

end NUMINAMATH_CALUDE_water_evaporation_period_l3063_306317


namespace NUMINAMATH_CALUDE_math_test_problems_left_l3063_306374

/-- Calculates the number of problems left to solve in a math test -/
def problems_left_to_solve (total_problems : ℕ) (first_20min : ℕ) (second_20min : ℕ) : ℕ :=
  total_problems - (first_20min + second_20min)

/-- Proves that given the conditions, the number of problems left to solve is 45 -/
theorem math_test_problems_left : 
  let total_problems : ℕ := 75
  let first_20min : ℕ := 10
  let second_20min : ℕ := first_20min * 2
  problems_left_to_solve total_problems first_20min second_20min = 45 := by
  sorry

#eval problems_left_to_solve 75 10 20

end NUMINAMATH_CALUDE_math_test_problems_left_l3063_306374


namespace NUMINAMATH_CALUDE_choir_theorem_l3063_306365

def choir_problem (original_size absent first_fraction second_fraction third_fraction fourth_fraction late_arrivals : ℕ) : Prop :=
  let present := original_size - absent
  let first_verse := present / 2
  let second_verse := (present - first_verse) / 3
  let third_verse := (present - first_verse - second_verse) / 4
  let fourth_verse := (present - first_verse - second_verse - third_verse) / 5
  let total_before_fifth := first_verse + second_verse + third_verse + fourth_verse + late_arrivals
  total_before_fifth + (present - total_before_fifth) = present

theorem choir_theorem :
  choir_problem 70 10 2 3 4 5 5 :=
sorry

end NUMINAMATH_CALUDE_choir_theorem_l3063_306365


namespace NUMINAMATH_CALUDE_ladder_problem_l3063_306352

theorem ladder_problem (ladder_length height base : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12)
  (h3 : ladder_length ^ 2 = height ^ 2 + base ^ 2) : 
  base = 5 := by
sorry

end NUMINAMATH_CALUDE_ladder_problem_l3063_306352


namespace NUMINAMATH_CALUDE_batch_not_qualified_l3063_306311

-- Define the parameters of the normal distribution
def mean : ℝ := 4
def std_dev : ℝ := 0.5  -- sqrt(0.25)

-- Define the measured diameter
def measured_diameter : ℝ := 5.7

-- Define a function to determine if a batch is qualified
def is_qualified (x : ℝ) : Prop :=
  (x - mean) / std_dev ≤ 3 ∧ (x - mean) / std_dev ≥ -3

-- Theorem statement
theorem batch_not_qualified : ¬(is_qualified measured_diameter) :=
sorry

end NUMINAMATH_CALUDE_batch_not_qualified_l3063_306311


namespace NUMINAMATH_CALUDE_truck_tunnel_time_l3063_306376

theorem truck_tunnel_time (truck_length : ℝ) (tunnel_length : ℝ) (speed_mph : ℝ) :
  truck_length = 66 →
  tunnel_length = 330 →
  speed_mph = 45 →
  let speed_fps := speed_mph * 5280 / 3600
  let total_distance := tunnel_length + truck_length
  let time := total_distance / speed_fps
  time = 6 := by sorry

end NUMINAMATH_CALUDE_truck_tunnel_time_l3063_306376


namespace NUMINAMATH_CALUDE_two_medians_not_unique_l3063_306302

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the concept of a median
def Median (t : Triangle) : ℝ → Prop := sorry

-- Define the concept of uniquely determining a triangle's shape
def UniquelyDeterminesShape (data : Set (Triangle → Prop)) : Prop := sorry

-- Define the five sets of data
def TwoSidesIncludedAngle : Set (Triangle → Prop) := sorry
def ThreeSides : Set (Triangle → Prop) := sorry
def TwoMedians : Set (Triangle → Prop) := sorry
def OneAltitudeAndBase : Set (Triangle → Prop) := sorry
def TwoAngles : Set (Triangle → Prop) := sorry

-- Theorem statement
theorem two_medians_not_unique :
  UniquelyDeterminesShape TwoSidesIncludedAngle ∧
  UniquelyDeterminesShape ThreeSides ∧
  ¬UniquelyDeterminesShape TwoMedians ∧
  UniquelyDeterminesShape OneAltitudeAndBase ∧
  UniquelyDeterminesShape TwoAngles :=
sorry

end NUMINAMATH_CALUDE_two_medians_not_unique_l3063_306302


namespace NUMINAMATH_CALUDE_erik_money_left_l3063_306389

-- Define the problem parameters
def initial_money : ℚ := 86
def bread_price : ℚ := 3
def juice_price : ℚ := 6
def eggs_price : ℚ := 4
def chocolate_price : ℚ := 2
def apples_price : ℚ := 1.25
def grapes_price : ℚ := 2.50

def bread_quantity : ℕ := 3
def juice_quantity : ℕ := 3
def eggs_quantity : ℕ := 2
def chocolate_quantity : ℕ := 5
def apples_quantity : ℚ := 4
def grapes_quantity : ℚ := 1.5

def bread_eggs_discount : ℚ := 0.10
def other_items_discount : ℚ := 0.05
def sales_tax_rate : ℚ := 0.06

-- Define the theorem
theorem erik_money_left : 
  let total_cost := bread_price * bread_quantity + juice_price * juice_quantity + 
                    eggs_price * eggs_quantity + chocolate_price * chocolate_quantity + 
                    apples_price * apples_quantity + grapes_price * grapes_quantity
  let bread_eggs_cost := bread_price * bread_quantity + eggs_price * eggs_quantity
  let other_items_cost := total_cost - bread_eggs_cost
  let discounted_cost := total_cost - (bread_eggs_cost * bread_eggs_discount) - 
                         (other_items_cost * other_items_discount)
  let final_cost := discounted_cost * (1 + sales_tax_rate)
  initial_money - final_cost = 32.78 := by
  sorry


end NUMINAMATH_CALUDE_erik_money_left_l3063_306389


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3063_306360

/-- Proves that the repeating decimal 0.3̄03 is equal to the fraction 109/330 -/
theorem repeating_decimal_to_fraction : 
  (0.3 : ℚ) + (3 : ℚ) / 100 / (1 - 1 / 100) = 109 / 330 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3063_306360


namespace NUMINAMATH_CALUDE_largest_number_proof_l3063_306390

theorem largest_number_proof (a b c : ℕ) 
  (h1 : c - a = 6) 
  (h2 : b = (a + c) / 2) 
  (h3 : a * b * c = 46332) : 
  c = 39 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_proof_l3063_306390


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l3063_306359

def number_to_convert : ℝ := 280000

theorem scientific_notation_proof :
  number_to_convert = 2.8 * (10 : ℝ)^5 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l3063_306359


namespace NUMINAMATH_CALUDE_pyramid_height_equal_volume_l3063_306308

theorem pyramid_height_equal_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) :
  cube_edge = 6 →
  pyramid_base = 10 →
  cube_edge ^ 3 = (1 / 3) * pyramid_base ^ 2 * pyramid_height →
  pyramid_height = 6.48 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equal_volume_l3063_306308


namespace NUMINAMATH_CALUDE_pies_sold_per_day_l3063_306318

/-- Given a restaurant that sells pies every day for a week and sells 56 pies in total,
    prove that the number of pies sold each day is 8. -/
theorem pies_sold_per_day (total_pies : ℕ) (days_in_week : ℕ) 
  (h1 : total_pies = 56) 
  (h2 : days_in_week = 7) :
  total_pies / days_in_week = 8 := by
  sorry

end NUMINAMATH_CALUDE_pies_sold_per_day_l3063_306318


namespace NUMINAMATH_CALUDE_median_salary_is_manager_salary_l3063_306305

/-- Represents a job position with its title, number of employees, and salary -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- Calculates the median salary given a list of positions -/
def medianSalary (positions : List Position) : Nat :=
  sorry

/-- The list of positions in the company -/
def companyPositions : List Position :=
  [{ title := "CEO", count := 1, salary := 140000 },
   { title := "Senior Manager", count := 4, salary := 95000 },
   { title := "Manager", count := 13, salary := 78000 },
   { title := "Assistant Manager", count := 7, salary := 55000 },
   { title := "Clerk", count := 38, salary := 25000 }]

/-- The total number of employees in the company -/
def totalEmployees : Nat :=
  companyPositions.foldl (fun acc pos => acc + pos.count) 0

theorem median_salary_is_manager_salary :
  medianSalary companyPositions = 78000 ∧ totalEmployees = 63 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_manager_salary_l3063_306305


namespace NUMINAMATH_CALUDE_airplane_distance_difference_l3063_306398

/-- The difference in distance traveled by an airplane flying without wind for 4 hours
    and against a 20 km/h wind for 3 hours, given that the airplane's windless speed is a km/h. -/
theorem airplane_distance_difference (a : ℝ) : 
  4 * a - (3 * (a - 20)) = a + 60 := by
  sorry

end NUMINAMATH_CALUDE_airplane_distance_difference_l3063_306398


namespace NUMINAMATH_CALUDE_f_properties_l3063_306326

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - 8^x)

theorem f_properties :
  (∀ x, f x ≠ 0 → x ≤ 2/3) ∧
  (∀ y, (∃ x, f x = y) → 0 ≤ y ∧ y < 2) ∧
  (∀ x, f x ≤ 1 → Real.log 3 / Real.log 8 ≤ x ∧ x ≤ 2/3) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3063_306326


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3063_306340

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (3*x - 1)^6 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a) →
  a₆ + a₅ + a₄ + a₃ + a₂ + a₁ + a = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3063_306340


namespace NUMINAMATH_CALUDE_solve_grocery_store_problem_l3063_306322

def grocery_store_problem (regular_soda : ℕ) (diet_soda : ℕ) (total_bottles : ℕ) : Prop :=
  let lite_soda : ℕ := total_bottles - (regular_soda + diet_soda)
  lite_soda = 27

theorem solve_grocery_store_problem :
  grocery_store_problem 57 26 110 := by
  sorry

end NUMINAMATH_CALUDE_solve_grocery_store_problem_l3063_306322


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3063_306353

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 7*x - 20) → (∃ y : ℝ, y^2 = 7*y - 20 ∧ x + y = 7) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3063_306353


namespace NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l3063_306362

theorem product_from_lcm_and_gcd (a b : ℤ) : 
  lcm a b = 36 → gcd a b = 6 → a * b = 216 := by sorry

end NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l3063_306362


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3063_306383

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 7
  let b := (6 : ℚ) / 11
  (a + b) / 2 = 75 / 154 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3063_306383


namespace NUMINAMATH_CALUDE_remainder_2015_div_28_l3063_306329

theorem remainder_2015_div_28 : 2015 % 28 = 17 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2015_div_28_l3063_306329


namespace NUMINAMATH_CALUDE_pop_spent_15_l3063_306306

def cereal_spending (pop crackle snap : ℝ) : Prop :=
  pop + crackle + snap = 150 ∧
  snap = 2 * crackle ∧
  crackle = 3 * pop

theorem pop_spent_15 :
  ∃ (pop crackle snap : ℝ), cereal_spending pop crackle snap ∧ pop = 15 := by
  sorry

end NUMINAMATH_CALUDE_pop_spent_15_l3063_306306


namespace NUMINAMATH_CALUDE_spelling_contest_total_questions_l3063_306367

/-- Represents a participant in the spelling contest -/
structure Participant where
  name : String
  round1_correct : ℕ
  round1_wrong : ℕ
  round2_correct : ℕ
  round2_wrong : ℕ
  round3_correct : ℕ
  round3_wrong : ℕ

/-- Calculates the total number of questions for a participant -/
def totalQuestions (p : Participant) : ℕ :=
  p.round1_correct + p.round1_wrong +
  p.round2_correct + p.round2_wrong +
  p.round3_correct + p.round3_wrong

/-- The spelling contest -/
def spellingContest : Prop :=
  let drew : Participant := {
    name := "Drew"
    round1_correct := 20
    round1_wrong := 6
    round2_correct := 24
    round2_wrong := 9
    round3_correct := 28
    round3_wrong := 14
  }
  let carla : Participant := {
    name := "Carla"
    round1_correct := 14
    round1_wrong := 2 * drew.round1_wrong
    round2_correct := 21
    round2_wrong := 8
    round3_correct := 22
    round3_wrong := 10
  }
  let blake : Participant := {
    name := "Blake"
    round1_correct := 0
    round1_wrong := 0
    round2_correct := 18
    round2_wrong := 11
    round3_correct := 15
    round3_wrong := 16
  }
  
  -- Conditions
  (∀ p : Participant, (p.round1_correct : ℚ) / (p.round1_correct + p.round1_wrong) ≥ 0.7) ∧
  (∀ p : Participant, (p.round2_correct : ℚ) / (p.round2_correct + p.round2_wrong) ≥ 0.7) ∧
  (∀ p : Participant, (p.round3_correct : ℚ) / (p.round3_correct + p.round3_wrong) ≥ 0.7) ∧
  (∀ p : Participant, ((p.round1_correct + p.round2_correct) : ℚ) / (p.round1_correct + p.round1_wrong + p.round2_correct + p.round2_wrong) ≥ 0.75) ∧
  
  -- Theorem to prove
  (totalQuestions drew + totalQuestions carla + totalQuestions blake = 248)

theorem spelling_contest_total_questions : spellingContest := by sorry

end NUMINAMATH_CALUDE_spelling_contest_total_questions_l3063_306367


namespace NUMINAMATH_CALUDE_tip_percentage_is_15_percent_l3063_306337

def lunch_cost : ℝ := 50.50
def total_spent : ℝ := 58.075

theorem tip_percentage_is_15_percent :
  (total_spent - lunch_cost) / lunch_cost * 100 = 15 := by sorry

end NUMINAMATH_CALUDE_tip_percentage_is_15_percent_l3063_306337


namespace NUMINAMATH_CALUDE_carlton_outfit_combinations_l3063_306366

theorem carlton_outfit_combinations 
  (button_up_shirts : ℕ) 
  (sweater_vests : ℕ) 
  (ties : ℕ) 
  (h1 : button_up_shirts = 4)
  (h2 : sweater_vests = 3 * button_up_shirts)
  (h3 : ties = 2 * sweater_vests) : 
  button_up_shirts * sweater_vests * ties = 1152 := by
  sorry

end NUMINAMATH_CALUDE_carlton_outfit_combinations_l3063_306366


namespace NUMINAMATH_CALUDE_horner_operations_for_f_l3063_306382

def f (x : ℝ) := 6 * x^6 + 5

def horner_operations (p : ℝ → ℝ) (degree : ℕ) : ℕ × ℕ :=
  (degree, degree)

theorem horner_operations_for_f :
  horner_operations f 6 = (6, 6) := by sorry

end NUMINAMATH_CALUDE_horner_operations_for_f_l3063_306382


namespace NUMINAMATH_CALUDE_justin_flower_gathering_time_l3063_306351

/-- Calculates the additional time needed for Justin to gather flowers for his classmates -/
def additional_time_needed (
  classmates : ℕ)
  (average_time_per_flower : ℕ)
  (gathering_time_hours : ℕ)
  (lost_flowers : ℕ) : ℕ :=
  let gathering_time_minutes := gathering_time_hours * 60
  let flowers_gathered := gathering_time_minutes / average_time_per_flower
  let flowers_remaining := flowers_gathered - lost_flowers
  let additional_flowers_needed := classmates - flowers_remaining
  additional_flowers_needed * average_time_per_flower

theorem justin_flower_gathering_time :
  additional_time_needed 30 10 2 3 = 210 := by
  sorry

end NUMINAMATH_CALUDE_justin_flower_gathering_time_l3063_306351


namespace NUMINAMATH_CALUDE_banking_problem_l3063_306384

/-- Calculates the final amount after deposit growth and withdrawal fee --/
def finalAmount (initialDeposit : ℝ) (growthRate : ℝ) (feeRate : ℝ) : ℝ :=
  initialDeposit * (1 + growthRate) * (1 - feeRate)

/-- Represents the banking problem with Vlad and Dima's deposits --/
theorem banking_problem (initialDeposit : ℝ) 
  (h_initial : initialDeposit = 3000) 
  (vladGrowthRate dimaGrowthRate vladFeeRate dimaFeeRate : ℝ)
  (h_vlad_growth : vladGrowthRate = 0.2)
  (h_vlad_fee : vladFeeRate = 0.1)
  (h_dima_growth : dimaGrowthRate = 0.4)
  (h_dima_fee : dimaFeeRate = 0.2) :
  finalAmount initialDeposit dimaGrowthRate dimaFeeRate - 
  finalAmount initialDeposit vladGrowthRate vladFeeRate = 120 := by
  sorry


end NUMINAMATH_CALUDE_banking_problem_l3063_306384


namespace NUMINAMATH_CALUDE_white_balls_count_l3063_306327

theorem white_balls_count (total : ℕ) (red : ℕ) (white : ℕ) : 
  red = 8 →
  red + white = total →
  (5 : ℚ) / 6 * total = white →
  white = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l3063_306327


namespace NUMINAMATH_CALUDE_emily_unanswered_questions_l3063_306313

def total_questions : ℕ := 50
def new_score : ℕ := 120
def old_score : ℕ := 95

def scoring_systems (c w u : ℕ) : Prop :=
  (6 * c + u = new_score) ∧
  (50 + 3 * c - 2 * w = old_score) ∧
  (c + w + u = total_questions)

theorem emily_unanswered_questions :
  ∃ (c w u : ℕ), scoring_systems c w u ∧ u = 37 :=
by sorry

end NUMINAMATH_CALUDE_emily_unanswered_questions_l3063_306313


namespace NUMINAMATH_CALUDE_certain_multiple_remainder_l3063_306354

theorem certain_multiple_remainder (m : ℤ) (h : m % 5 = 2) :
  (∃ k : ℕ+, k * m % 5 = 1) ∧ (∀ k : ℕ+, k * m % 5 = 1 → k ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_certain_multiple_remainder_l3063_306354


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3063_306399

/-- The line equation as a function of m, x, and y -/
def line_equation (m x y : ℝ) : Prop :=
  (m - 1) * x - y + 2 * m + 1 = 0

/-- The theorem stating that the line passes through (-2, 3) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m (-2) 3 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3063_306399


namespace NUMINAMATH_CALUDE_beach_relaxation_l3063_306358

/-- The number of people left relaxing on the beach -/
def people_left_relaxing (row1_initial : ℕ) (row1_left : ℕ) (row2_initial : ℕ) (row2_left : ℕ) (row3 : ℕ) : ℕ :=
  (row1_initial - row1_left) + (row2_initial - row2_left) + row3

/-- Theorem stating the number of people left relaxing on the beach -/
theorem beach_relaxation : 
  people_left_relaxing 24 3 20 5 18 = 54 := by
  sorry

end NUMINAMATH_CALUDE_beach_relaxation_l3063_306358


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l3063_306307

/-- Two parabolas with different vertices, where each parabola's vertex lies on the other parabola -/
structure TwoParabolas where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  p : ℝ
  q : ℝ
  h_diff_vertices : x₁ ≠ x₂
  h_vertex_on_other₁ : y₂ = p * (x₂ - x₁)^2 + y₁
  h_vertex_on_other₂ : y₁ = q * (x₁ - x₂)^2 + y₂

/-- The sum of the leading coefficients of two parabolas with the described properties is zero -/
theorem sum_of_coefficients_is_zero (tp : TwoParabolas) : tp.p + tp.q = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l3063_306307


namespace NUMINAMATH_CALUDE_symmetry_axis_of_sine_l3063_306321

theorem symmetry_axis_of_sine (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (1/2 * x + π/3)
  f (π/3 + (x - π/3)) = f (π/3 - (x - π/3)) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axis_of_sine_l3063_306321


namespace NUMINAMATH_CALUDE_average_equals_5x_minus_9_l3063_306381

theorem average_equals_5x_minus_9 (x : ℚ) : 
  (1/3 : ℚ) * ((x + 8) + (8*x + 3) + (3*x + 9)) = 5*x - 9 → x = 47/3 := by
sorry

end NUMINAMATH_CALUDE_average_equals_5x_minus_9_l3063_306381


namespace NUMINAMATH_CALUDE_expression_equals_ten_to_twelve_l3063_306386

theorem expression_equals_ten_to_twelve : (2 * 5 * 10^5) * 10^6 = 10^12 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_ten_to_twelve_l3063_306386


namespace NUMINAMATH_CALUDE_fireworks_display_total_l3063_306379

/-- The number of fireworks used in a New Year's Eve display -/
def fireworks_display (fireworks_per_number : ℕ) (fireworks_per_letter : ℕ) 
  (year_digits : ℕ) (phrase_letters : ℕ) (additional_boxes : ℕ) (fireworks_per_box : ℕ) : ℕ :=
  (fireworks_per_number * year_digits) + 
  (fireworks_per_letter * phrase_letters) + 
  (additional_boxes * fireworks_per_box)

/-- Theorem stating the total number of fireworks used in the display -/
theorem fireworks_display_total : 
  fireworks_display 6 5 4 12 50 8 = 484 := by
  sorry

end NUMINAMATH_CALUDE_fireworks_display_total_l3063_306379


namespace NUMINAMATH_CALUDE_remainder_of_binary_number_div_4_l3063_306332

def binary_number : ℕ := 3789 -- 111001001101₂ in decimal

theorem remainder_of_binary_number_div_4 :
  binary_number % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_number_div_4_l3063_306332


namespace NUMINAMATH_CALUDE_solve_race_problem_l3063_306309

def race_problem (patrick_time manu_extra_time : ℕ) (amy_speed_ratio : ℚ) : Prop :=
  let manu_time := patrick_time + manu_extra_time
  let amy_time := manu_time / amy_speed_ratio
  amy_time = 36

theorem solve_race_problem :
  race_problem 60 12 2 := by sorry

end NUMINAMATH_CALUDE_solve_race_problem_l3063_306309


namespace NUMINAMATH_CALUDE_square_sum_identity_l3063_306377

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l3063_306377


namespace NUMINAMATH_CALUDE_beach_house_rental_l3063_306341

theorem beach_house_rental (individual_payment : ℕ) (total_payment : ℕ) 
  (h1 : individual_payment = 70)
  (h2 : total_payment = 490) :
  total_payment / individual_payment = 7 :=
by sorry

end NUMINAMATH_CALUDE_beach_house_rental_l3063_306341


namespace NUMINAMATH_CALUDE_three_good_pairs_l3063_306324

-- Define a structure for a line in slope-intercept form
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the lines
def L1 : Line := { slope := 2, intercept := 3 }
def L2 : Line := { slope := 2, intercept := 3 }
def L3 : Line := { slope := 4, intercept := -2 }
def L4 : Line := { slope := -4, intercept := 3 }
def L5 : Line := { slope := -4, intercept := 3 }

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

-- Define what it means for two lines to be perpendicular
def perpendicular (l1 l2 : Line) : Prop := l1.slope * l2.slope = -1

-- Define what it means for two lines to be "good"
def good (l1 l2 : Line) : Prop := parallel l1 l2 ∨ perpendicular l1 l2

-- The main theorem
theorem three_good_pairs :
  ∃ (pairs : List (Line × Line)),
    pairs.length = 3 ∧
    (∀ p ∈ pairs, good p.1 p.2) ∧
    (∀ l1 l2 : Line, l1 ≠ l2 → good l1 l2 → (l1, l2) ∈ pairs ∨ (l2, l1) ∈ pairs) :=
by
  sorry

end NUMINAMATH_CALUDE_three_good_pairs_l3063_306324


namespace NUMINAMATH_CALUDE_f_max_value_l3063_306342

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x - 2 * Real.sin (3 * x)

theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = (16 * Real.sqrt 3) / 9 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l3063_306342


namespace NUMINAMATH_CALUDE_value_of_a_l3063_306364

theorem value_of_a (a b : ℚ) (h1 : b/a = 4) (h2 : b = 15 - 4*a) : a = 15/8 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3063_306364


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l3063_306330

theorem quadratic_root_implies_a (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x - a = 0) ∧ ((-1)^2 - 3*(-1) - a = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l3063_306330


namespace NUMINAMATH_CALUDE_work_completion_time_l3063_306304

/-- Given that:
  - A can do a work in 4 days
  - A and B together can finish the work in 3 days
  Prove that B can do the work alone in 12 days -/
theorem work_completion_time (a_time b_time combined_time : ℝ) 
  (ha : a_time = 4)
  (hc : combined_time = 3)
  (h_combined : 1 / a_time + 1 / b_time = 1 / combined_time) :
  b_time = 12 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3063_306304


namespace NUMINAMATH_CALUDE_quadratic_properties_l3063_306339

def f (x : ℝ) := -2 * x^2 + 4 * x + 3

theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f x > f y) ∧
  (∀ x : ℝ, f (x + 1) = f (1 - x)) ∧
  (f 1 = 5) ∧
  (∀ x : ℝ, x > 1 → ∀ y : ℝ, y > x → f y < f x) ∧
  (∀ x : ℝ, x < 1 → ∀ y : ℝ, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3063_306339


namespace NUMINAMATH_CALUDE_man_speed_opposite_train_man_speed_specific_case_l3063_306301

/-- Calculates the speed of a man running opposite to a train, given the train's length, speed, and time to pass the man. -/
theorem man_speed_opposite_train (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let man_speed_mps := train_length / passing_time - train_speed_mps
  let man_speed_kmph := man_speed_mps * (3600 / 1000)
  man_speed_kmph

/-- The speed of a man running opposite to a train is approximately 5.99 kmph, given:
    - The train is 550 meters long
    - The train's speed is 60 kmph
    - The train passes the man in 30 seconds -/
theorem man_speed_specific_case : 
  abs (man_speed_opposite_train 550 60 30 - 5.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_opposite_train_man_speed_specific_case_l3063_306301


namespace NUMINAMATH_CALUDE_no_real_roots_l3063_306300

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (2 * x + 8) - Real.sqrt (x - 1) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3063_306300


namespace NUMINAMATH_CALUDE_polygon_sides_l3063_306334

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : sum_interior_angles = 1260 → (n - 2) * 180 = sum_interior_angles → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3063_306334


namespace NUMINAMATH_CALUDE_robin_female_fraction_l3063_306336

theorem robin_female_fraction (total_birds : ℝ) (h1 : total_birds > 0) : 
  let robins : ℝ := (2/5) * total_birds
  let bluejays : ℝ := (3/5) * total_birds
  let female_bluejays : ℝ := (2/3) * bluejays
  let male_birds : ℝ := (7/15) * total_birds
  let female_robins : ℝ := (1/3) * robins
  female_robins + female_bluejays = total_birds - male_birds :=
by
  sorry

#check robin_female_fraction

end NUMINAMATH_CALUDE_robin_female_fraction_l3063_306336


namespace NUMINAMATH_CALUDE_stating_price_reduction_achieves_target_profit_l3063_306316

/-- Represents the price reduction problem for a product in a shopping mall. -/
structure PriceReductionProblem where
  initialSales : ℕ        -- Initial average daily sales
  initialProfit : ℕ       -- Initial profit per unit
  salesIncrease : ℕ       -- Sales increase per yuan of price reduction
  targetProfit : ℕ        -- Target daily profit
  priceReduction : ℕ      -- Price reduction per unit

/-- 
Theorem stating that the given price reduction achieves the target profit 
for the specified problem parameters.
-/
theorem price_reduction_achieves_target_profit 
  (p : PriceReductionProblem)
  (h1 : p.initialSales = 30)
  (h2 : p.initialProfit = 50)
  (h3 : p.salesIncrease = 2)
  (h4 : p.targetProfit = 2000)
  (h5 : p.priceReduction = 25) :
  (p.initialProfit - p.priceReduction) * (p.initialSales + p.salesIncrease * p.priceReduction) = p.targetProfit :=
by sorry

end NUMINAMATH_CALUDE_stating_price_reduction_achieves_target_profit_l3063_306316


namespace NUMINAMATH_CALUDE_adjacent_diff_at_least_16_l3063_306397

/-- Represents a 6x6 grid with integers from 1 to 36 -/
def Grid := Fin 6 → Fin 6 → Fin 36

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p1 p2 : Fin 6 × Fin 6) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- A valid grid satisfies the problem conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ≤ 36) ∧
  (∃ i1 j1 i2 j2 i3 j3 i4 j4,
    g i1 j1 = 1 ∧ g i2 j2 = 2 ∧ g i3 j3 = 3 ∧ g i4 j4 = 4) ∧
  (∀ i j, g i j ≤ 4 → g i j ≥ 1) ∧
  (∀ i j, g i j > 4 → g i j ≤ 36)

/-- The main theorem -/
theorem adjacent_diff_at_least_16 (g : Grid) (h : valid_grid g) :
  ∃ p1 p2 : Fin 6 × Fin 6, adjacent p1 p2 ∧ |g p1.1 p1.2 - g p2.1 p2.2| ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_diff_at_least_16_l3063_306397


namespace NUMINAMATH_CALUDE_neznaika_claims_l3063_306380

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divisibility_claims (n : ℕ) : List Bool :=
  [n % 3 = 0, n % 4 = 0, n % 5 = 0, n % 9 = 0, n % 10 = 0, n % 15 = 0, n % 18 = 0, n % 30 = 0]

theorem neznaika_claims (n : ℕ) : 
  is_two_digit n → (divisibility_claims n).count false = 4 → n = 36 ∨ n = 45 ∨ n = 72 := by
  sorry

end NUMINAMATH_CALUDE_neznaika_claims_l3063_306380


namespace NUMINAMATH_CALUDE_river_distance_l3063_306310

theorem river_distance (d : ℝ) : 
  (¬ (d ≥ 8)) → (¬ (d ≤ 7)) → (¬ (d ≤ 6)) → (7 < d ∧ d < 8) := by
  sorry

end NUMINAMATH_CALUDE_river_distance_l3063_306310


namespace NUMINAMATH_CALUDE_meeting_time_calculation_l3063_306395

-- Define the speeds of the two people
def v₁ : ℝ := 6
def v₂ : ℝ := 4

-- Define the time difference in reaching the final destination
def time_difference : ℝ := 10

-- Define the theorem to prove
theorem meeting_time_calculation (t₁ : ℝ) :
  v₂ * t₁ = v₁ * (t₁ - time_difference) → t₁ = 30 :=
by sorry

end NUMINAMATH_CALUDE_meeting_time_calculation_l3063_306395


namespace NUMINAMATH_CALUDE_earthquake_victims_scientific_notation_l3063_306323

/-- Definition of scientific notation -/
def is_scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  x = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10

/-- The problem statement -/
theorem earthquake_victims_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation 153000 a n ∧ a = 1.53 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_earthquake_victims_scientific_notation_l3063_306323


namespace NUMINAMATH_CALUDE_unique_configuration_l3063_306394

/-- A configuration of n points in the plane with associated real numbers. -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ
  r : Fin n → ℝ

/-- The area of a triangle given by three points in the plane. -/
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

/-- Predicate stating that three points are not collinear. -/
def nonCollinear (p₁ p₂ p₃ : ℝ × ℝ) : Prop := sorry

/-- The configuration satisfies the area condition for all triples of points. -/
def satisfiesAreaCondition (config : PointConfiguration n) : Prop :=
  ∀ (i j k : Fin n), i < j → j < k →
    triangleArea (config.points i) (config.points j) (config.points k) =
    config.r i + config.r j + config.r k

/-- The configuration satisfies the non-collinearity condition for all triples of points. -/
def satisfiesNonCollinearityCondition (config : PointConfiguration n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    nonCollinear (config.points i) (config.points j) (config.points k)

/-- The main theorem stating that 4 is the only integer greater than 3 satisfying the conditions. -/
theorem unique_configuration :
  ∀ (n : ℕ), n > 3 →
  (∃ (config : PointConfiguration n),
    satisfiesAreaCondition config ∧
    satisfiesNonCollinearityCondition config) →
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_configuration_l3063_306394


namespace NUMINAMATH_CALUDE_parallel_vectors_condition_l3063_306338

/-- Given two vectors a and b in R², prove that if they are parallel and a = (-1, 3) and b = (1, t), then t = -3. -/
theorem parallel_vectors_condition (a b : ℝ × ℝ) (t : ℝ) : 
  a = (-1, 3) → b = (1, t) → (∃ (k : ℝ), a = k • b) → t = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_condition_l3063_306338


namespace NUMINAMATH_CALUDE_meeting_probability_l3063_306315

/-- Xiaocong's arrival time at Wuhan Station -/
def xiaocong_arrival : ℝ := 13.5

/-- Duration of Xiaocong's rest at Wuhan Station -/
def xiaocong_rest : ℝ := 1

/-- Earliest possible arrival time for Xiaoming -/
def xiaoming_earliest : ℝ := 14

/-- Latest possible arrival time for Xiaoming -/
def xiaoming_latest : ℝ := 15

/-- Xiaoming's train departure time -/
def xiaoming_departure : ℝ := 15.5

/-- The probability of Xiaocong and Xiaoming meeting at Wuhan Station -/
theorem meeting_probability : ℝ := by sorry

end NUMINAMATH_CALUDE_meeting_probability_l3063_306315


namespace NUMINAMATH_CALUDE_binomial_half_variance_l3063_306393

/-- A random variable following a binomial distribution -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial variable -/
def variance (X : BinomialVariable) : ℝ := X.n * X.p * (1 - X.p)

/-- The main theorem -/
theorem binomial_half_variance (X : BinomialVariable) 
  (h2 : X.n = 8) (h3 : X.p = 3/5) : 
  variance X * (1/2)^2 = 12/25 := by sorry

end NUMINAMATH_CALUDE_binomial_half_variance_l3063_306393


namespace NUMINAMATH_CALUDE_no_integer_tangent_length_l3063_306303

/-- A circle with a point P outside it, from which a tangent and a secant are drawn -/
structure CircleWithExternalPoint where
  /-- The circumference of the circle -/
  circumference : ℝ
  /-- The length of one arc created by the secant -/
  m : ℕ
  /-- The length of the tangent from P to the circle -/
  t₁ : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (c : CircleWithExternalPoint) : Prop :=
  c.circumference = 15 * Real.pi ∧
  c.t₁ * c.t₁ = c.m * (c.circumference - c.m)

/-- The theorem stating that no integer values of t₁ satisfy the conditions -/
theorem no_integer_tangent_length :
  ¬∃ c : CircleWithExternalPoint, satisfiesConditions c :=
sorry

end NUMINAMATH_CALUDE_no_integer_tangent_length_l3063_306303


namespace NUMINAMATH_CALUDE_two_discount_equation_l3063_306347

/-- Proves the equation for a product's price after two consecutive discounts -/
theorem two_discount_equation (original_price final_price x : ℝ) :
  original_price = 400 →
  final_price = 225 →
  0 < x →
  x < 1 →
  original_price * (1 - x)^2 = final_price :=
by sorry

end NUMINAMATH_CALUDE_two_discount_equation_l3063_306347


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3063_306350

/-- The sum of the coordinates of the center of the circle defined by x^2 + y^2 = -4x - 6y + 5 is -5 -/
theorem circle_center_coordinate_sum :
  ∃ (x y : ℝ), x^2 + y^2 = -4*x - 6*y + 5 ∧ x + y = -5 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3063_306350


namespace NUMINAMATH_CALUDE_equation_solution_l3063_306363

theorem equation_solution : ∃! x : ℝ, 3 * x - 4 = -2 * x + 11 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3063_306363


namespace NUMINAMATH_CALUDE_shooting_training_probabilities_l3063_306375

/-- Shooting training probabilities -/
structure ShootingProbabilities where
  nine_or_above : ℝ
  eight_to_nine : ℝ
  seven_to_eight : ℝ
  six_to_seven : ℝ

/-- Theorem for shooting training probabilities -/
theorem shooting_training_probabilities
  (probs : ShootingProbabilities)
  (h1 : probs.nine_or_above = 0.18)
  (h2 : probs.eight_to_nine = 0.51)
  (h3 : probs.seven_to_eight = 0.15)
  (h4 : probs.six_to_seven = 0.09) :
  (probs.nine_or_above + probs.eight_to_nine = 0.69) ∧
  (probs.nine_or_above + probs.eight_to_nine + probs.seven_to_eight + probs.six_to_seven = 0.93) :=
by sorry

end NUMINAMATH_CALUDE_shooting_training_probabilities_l3063_306375


namespace NUMINAMATH_CALUDE_optimal_newspaper_sales_l3063_306349

/-- Represents the newsstand's daily newspaper sales and profit calculation. -/
structure NewspaperSales where
  buyPrice : ℚ
  sellPrice : ℚ
  returnPrice : ℚ
  highDemandDays : ℕ
  lowDemandDays : ℕ
  highDemandAmount : ℕ
  lowDemandAmount : ℕ

/-- Calculates the monthly profit for a given number of daily purchases. -/
def monthlyProfit (sales : NewspaperSales) (dailyPurchase : ℕ) : ℚ :=
  sorry

/-- Theorem stating the optimal daily purchase and maximum monthly profit. -/
theorem optimal_newspaper_sales :
  ∃ (sales : NewspaperSales),
    sales.buyPrice = 24/100 ∧
    sales.sellPrice = 40/100 ∧
    sales.returnPrice = 8/100 ∧
    sales.highDemandDays = 20 ∧
    sales.lowDemandDays = 10 ∧
    sales.highDemandAmount = 300 ∧
    sales.lowDemandAmount = 200 ∧
    (∀ x : ℕ, monthlyProfit sales x ≤ monthlyProfit sales 300) ∧
    monthlyProfit sales 300 = 1120 := by
  sorry

end NUMINAMATH_CALUDE_optimal_newspaper_sales_l3063_306349


namespace NUMINAMATH_CALUDE_additional_curtain_material_l3063_306387

-- Define the room height in feet
def room_height_feet : ℕ := 8

-- Define the desired curtain length in inches
def desired_curtain_length : ℕ := 101

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Theorem to prove the additional material needed
theorem additional_curtain_material :
  desired_curtain_length - (room_height_feet * feet_to_inches) = 5 := by
  sorry

end NUMINAMATH_CALUDE_additional_curtain_material_l3063_306387


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l3063_306325

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/2, 1/3 + 1/9, 1/3 + 1/6]
  (∀ s ∈ sums, s ≤ (1/3 + 1/2)) ∧ (1/3 + 1/2 = 5/6) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l3063_306325


namespace NUMINAMATH_CALUDE_derivative_sin_pi_sixth_l3063_306335

theorem derivative_sin_pi_sixth (h : Real.sin (π / 6) = (1 : ℝ) / 2) : 
  deriv (λ _ : ℝ => Real.sin (π / 6)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_pi_sixth_l3063_306335


namespace NUMINAMATH_CALUDE_intersection_theorem_l3063_306370

open Set Real

-- Define sets A and B
def A : Set ℝ := {x | x^2 + x - 6 < 0}
def B : Set ℝ := {x | x + 1 > 0}

-- Define the intersection of A and B
def A_inter_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_theorem : A_inter_B = Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l3063_306370


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fifteen_fourths_l3063_306355

theorem greatest_integer_less_than_negative_fifteen_fourths :
  ⌊-15/4⌋ = -4 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fifteen_fourths_l3063_306355


namespace NUMINAMATH_CALUDE_unique_congruence_solution_l3063_306368

theorem unique_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 4 ∧ n ≡ -998 [ZMOD 5] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_solution_l3063_306368


namespace NUMINAMATH_CALUDE_min_x_plus_y_l3063_306371

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = x*y) :
  x + y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 4*y₀ = x₀*y₀ ∧ x₀ + y₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_x_plus_y_l3063_306371


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l3063_306388

/-- In a right-angled triangle ABC, prove that arctan(a/(b+c)) + arctan(c/(a+b)) = π/4 -/
theorem right_triangle_arctan_sum (a b c : ℝ) (h : a^2 + c^2 = b^2) :
  Real.arctan (a / (b + c)) + Real.arctan (c / (a + b)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l3063_306388


namespace NUMINAMATH_CALUDE_count_cubic_functions_l3063_306391

-- Define the structure of our cubic function
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the property we're interested in
def satisfiesProperty (f : CubicFunction) : Prop :=
  ∀ x : ℝ, (f.a * x^3 + f.b * x^2 + f.c * x + f.d) *
            ((-f.a) * x^3 + f.b * x^2 + (-f.c) * x + f.d) =
            f.a * x^6 + f.b * x^4 + f.c * x^2 + f.d

-- State the theorem
theorem count_cubic_functions :
  ∃! (s : Finset CubicFunction),
    (∀ f ∈ s, satisfiesProperty f) ∧ s.card = 16 := by
  sorry

end NUMINAMATH_CALUDE_count_cubic_functions_l3063_306391


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3063_306333

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x + 3 / (x + 1) ≥ 2 * Real.sqrt 3 - 1 :=
by sorry

theorem min_value_achievable :
  ∃ x > 0, x + 3 / (x + 1) = 2 * Real.sqrt 3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3063_306333


namespace NUMINAMATH_CALUDE_reader_collection_pages_l3063_306344

def book1_chapters : List Nat := [24, 32, 40, 20]
def book2_chapters : List Nat := [48, 52, 36]
def book3_chapters : List Nat := [16, 28, 44, 22, 34]

def total_pages (chapters : List Nat) : Nat :=
  chapters.sum

theorem reader_collection_pages :
  (total_pages book1_chapters) +
  (total_pages book2_chapters) +
  (total_pages book3_chapters) = 396 := by
  sorry

end NUMINAMATH_CALUDE_reader_collection_pages_l3063_306344
