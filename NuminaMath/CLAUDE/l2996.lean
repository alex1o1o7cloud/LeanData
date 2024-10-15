import Mathlib

namespace NUMINAMATH_CALUDE_triangle_third_sides_l2996_299652

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t1.a / t2.a = k ∧ 
    t1.b / t2.b = k ∧ 
    t1.c / t2.c = k

theorem triangle_third_sides 
  (t1 t2 : Triangle) 
  (h_similar : similar t1 t2) 
  (h_not_congruent : t1 ≠ t2) 
  (h_t1_sides : t1.a = 12 ∧ t1.b = 18) 
  (h_t2_sides : t2.a = 12 ∧ t2.b = 18) : 
  (t1.c = 27/2 ∧ t2.c = 8) ∨ (t1.c = 8 ∧ t2.c = 27/2) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_sides_l2996_299652


namespace NUMINAMATH_CALUDE_bills_difference_l2996_299699

/-- The number of bills each person had at the beginning -/
structure Bills where
  geric : ℕ
  kyla : ℕ
  jessa : ℕ

/-- The conditions of the problem -/
def problem_conditions (b : Bills) : Prop :=
  b.geric = 2 * b.kyla ∧
  b.geric = 16 ∧
  b.jessa - 3 = 7

/-- The theorem to prove -/
theorem bills_difference (b : Bills) 
  (h : problem_conditions b) : b.jessa - b.kyla = 2 := by
  sorry

end NUMINAMATH_CALUDE_bills_difference_l2996_299699


namespace NUMINAMATH_CALUDE_trig_identity_l2996_299647

theorem trig_identity : 
  Real.cos (π / 3) * Real.tan (π / 4) + 3 / 4 * (Real.tan (π / 6))^2 - Real.sin (π / 6) + (Real.cos (π / 6))^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2996_299647


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2996_299628

theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) 
  (h1 : principal = 1200)
  (h2 : interest_paid = 432) :
  ∃ (rate : ℝ), 
    rate > 0 ∧ 
    rate = (interest_paid * 100) / (principal * rate) ∧ 
    rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2996_299628


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_18_27_36_l2996_299681

theorem gcf_lcm_sum_18_27_36 : 
  let X := Nat.gcd 18 (Nat.gcd 27 36)
  let Y := Nat.lcm 18 (Nat.lcm 27 36)
  X + Y = 117 := by
sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_18_27_36_l2996_299681


namespace NUMINAMATH_CALUDE_april_price_achieves_profit_l2996_299654

/-- Represents the sales and pricing data for a desk lamp over several months -/
structure LampSalesData where
  cost_price : ℝ
  selling_price_jan_mar : ℝ
  sales_jan : ℕ
  sales_mar : ℕ
  price_reduction_sales_increase : ℝ
  desired_profit_apr : ℝ

/-- Calculates the selling price in April that achieves the desired profit -/
def calculate_april_price (data : LampSalesData) : ℝ :=
  sorry

/-- Theorem stating that the calculated April price achieves the desired profit -/
theorem april_price_achieves_profit (data : LampSalesData) 
  (h1 : data.cost_price = 25)
  (h2 : data.selling_price_jan_mar = 40)
  (h3 : data.sales_jan = 256)
  (h4 : data.sales_mar = 400)
  (h5 : data.price_reduction_sales_increase = 4)
  (h6 : data.desired_profit_apr = 4200) :
  let april_price := calculate_april_price data
  let april_sales := data.sales_mar + data.price_reduction_sales_increase * (data.selling_price_jan_mar - april_price)
  (april_price - data.cost_price) * april_sales = data.desired_profit_apr ∧ april_price = 35 :=
sorry

end NUMINAMATH_CALUDE_april_price_achieves_profit_l2996_299654


namespace NUMINAMATH_CALUDE_sum_three_squares_to_four_fractions_l2996_299625

theorem sum_three_squares_to_four_fractions (A B C : ℤ) :
  ∃ (x y z : ℝ), 
    (A : ℝ)^2 + (B : ℝ)^2 + (C : ℝ)^2 = 
      ((A * (x^2 + y^2 - z^2) + B * (2*x*z) + C * (2*y*z)) / (x^2 + y^2 + z^2))^2 +
      ((A * (2*x*z) - B * (x^2 + y^2 - z^2)) / (x^2 + y^2 + z^2))^2 +
      ((B * (2*y*z) - C * (2*x*z)) / (x^2 + y^2 + z^2))^2 +
      ((C * (x^2 + y^2 - z^2) - A * (2*y*z)) / (x^2 + y^2 + z^2))^2 :=
by sorry

end NUMINAMATH_CALUDE_sum_three_squares_to_four_fractions_l2996_299625


namespace NUMINAMATH_CALUDE_min_chord_length_proof_l2996_299661

/-- The circle equation x^2 + y^2 - 6x = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

/-- The point through which the chord passes -/
def point : ℝ × ℝ := (1, 2)

/-- The minimum length of the chord intercepted by the circle passing through the point -/
def min_chord_length : ℝ := 2

theorem min_chord_length_proof :
  ∀ (x y : ℝ), circle_equation x y →
  min_chord_length = (2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_min_chord_length_proof_l2996_299661


namespace NUMINAMATH_CALUDE_root_implies_a_range_l2996_299672

theorem root_implies_a_range :
  ∀ a : ℝ, (∃ x : ℝ, 9^(-|x - 2|) - 4 * 3^(-|x - 2|) - a = 0) → -3 ≤ a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_range_l2996_299672


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_384_l2996_299617

/-- Counts 4-digit numbers beginning with 2 that have exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits := 10 -- Total number of digits (0-9)
  let first_digit := 2 -- First digit is always 2
  let remaining_digits := digits - 1 -- Excluding 2
  let configurations := 2 -- Two main configurations: 2 is repeated or not

  -- When 2 is one of the repeated digits
  let case1 := 3 * remaining_digits * remaining_digits

  -- When 2 is not one of the repeated digits
  let case2 := 3 * remaining_digits * remaining_digits

  case1 + case2

theorem count_special_numbers_eq_384 : count_special_numbers = 384 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_384_l2996_299617


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l2996_299663

theorem common_number_in_overlapping_lists (l : List ℚ) : 
  l.length = 7 ∧ 
  (l.take 4).sum / 4 = 7 ∧ 
  (l.drop 3).sum / 4 = 11 ∧ 
  l.sum / 7 = 66 / 7 → 
  ∃ x ∈ l.take 4 ∩ l.drop 3, x = 6 := by
sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l2996_299663


namespace NUMINAMATH_CALUDE_weightlifter_total_weight_l2996_299653

/-- The weight a weightlifter can lift in each hand, in pounds. -/
def weight_per_hand : ℕ := 10

/-- The total weight a weightlifter can lift at once, in pounds. -/
def total_weight : ℕ := weight_per_hand * 2

/-- Theorem stating that the total weight a weightlifter can lift at once is 20 pounds. -/
theorem weightlifter_total_weight : total_weight = 20 := by sorry

end NUMINAMATH_CALUDE_weightlifter_total_weight_l2996_299653


namespace NUMINAMATH_CALUDE_larger_number_problem_l2996_299611

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : 
  max x y = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2996_299611


namespace NUMINAMATH_CALUDE_largest_t_value_l2996_299678

theorem largest_t_value : 
  let f (t : ℝ) := (15 * t^2 - 38 * t + 14) / (4 * t - 3) + 6 * t
  ∃ (t_max : ℝ), t_max = 1 ∧ 
    (∀ (t : ℝ), f t = 7 * t - 2 → t ≤ t_max) ∧
    (f t_max = 7 * t_max - 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_t_value_l2996_299678


namespace NUMINAMATH_CALUDE_larger_integer_value_l2996_299659

theorem larger_integer_value (a b : ℕ+) (h1 : (a : ℝ) / (b : ℝ) = 7 / 3) (h2 : (a : ℕ) * b = 441) :
  max a b = ⌊7 * Real.sqrt 21⌋ := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l2996_299659


namespace NUMINAMATH_CALUDE_ratio_problem_l2996_299613

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2996_299613


namespace NUMINAMATH_CALUDE_major_premise_incorrect_l2996_299648

theorem major_premise_incorrect : ¬(∀ (a : ℝ) (n : ℕ), n > 0 → (a^(1/n : ℝ))^n = a) := by
  sorry

end NUMINAMATH_CALUDE_major_premise_incorrect_l2996_299648


namespace NUMINAMATH_CALUDE_sum_even_integers_402_to_500_l2996_299674

/-- Sum of first n positive even integers -/
def sumFirstEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of even integers from a to b inclusive -/
def sumEvenIntegers (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  n * (a + b) / 2

theorem sum_even_integers_402_to_500 :
  sumFirstEvenIntegers 50 = 2550 →
  sumEvenIntegers 402 500 = 22550 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_integers_402_to_500_l2996_299674


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2996_299606

/-- A line in the xy-plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line (b : Line) :
  parallel b { slope := 3/2, point := (0, 6) } →
  b.point = (4, 2) →
  y_intercept b = -4 := by
sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2996_299606


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2996_299698

/-- Three points in 3D space are collinear if they all lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop := sorry

/-- The main theorem: if the given points are collinear, then 2a + b = 8. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → 2 * a + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l2996_299698


namespace NUMINAMATH_CALUDE_ab_neg_necessary_not_sufficient_for_hyperbola_l2996_299638

/-- Represents a conic section in the form ax^2 + by^2 = c -/
structure ConicSection where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to determine if a conic section is a hyperbola -/
def IsHyperbola (conic : ConicSection) : Prop :=
  sorry  -- The actual definition would depend on the formal definition of a hyperbola

/-- The main theorem stating that ab < 0 is necessary but not sufficient for a hyperbola -/
theorem ab_neg_necessary_not_sufficient_for_hyperbola :
  (∀ conic : ConicSection, IsHyperbola conic → conic.a * conic.b < 0) ∧
  (∃ conic : ConicSection, conic.a * conic.b < 0 ∧ ¬IsHyperbola conic) :=
sorry

end NUMINAMATH_CALUDE_ab_neg_necessary_not_sufficient_for_hyperbola_l2996_299638


namespace NUMINAMATH_CALUDE_units_digit_sum_octal_l2996_299600

/-- Converts a number from base 8 to base 10 -/
def octalToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 8 -/
def decimalToOctal (n : ℕ) : ℕ := sorry

/-- Returns the units digit of a number in base 8 -/
def unitsDigitOctal (n : ℕ) : ℕ := sorry

theorem units_digit_sum_octal :
  unitsDigitOctal (decimalToOctal (octalToDecimal 45 + octalToDecimal 67)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_octal_l2996_299600


namespace NUMINAMATH_CALUDE_trivia_team_tryouts_l2996_299691

/-- 
Given 8 schools, where 17 students didn't get picked for each team,
and 384 total students make the teams, prove that 65 students tried out
for the trivia teams in each school.
-/
theorem trivia_team_tryouts (
  num_schools : ℕ) 
  (students_not_picked : ℕ) 
  (total_students_picked : ℕ) 
  (h1 : num_schools = 8)
  (h2 : students_not_picked = 17)
  (h3 : total_students_picked = 384) :
  num_schools * (65 - students_not_picked) = total_students_picked := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_tryouts_l2996_299691


namespace NUMINAMATH_CALUDE_equation_properties_l2996_299658

variable (a : ℝ)
variable (z : ℂ)

def has_real_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - (a + Complex.I)*x - (Complex.I + 2) = 0

def has_imaginary_solution (a : ℝ) : Prop :=
  ∃ y : ℝ, y ≠ 0 ∧ (Complex.I*y)^2 - (a + Complex.I)*(Complex.I*y) - (Complex.I + 2) = 0

theorem equation_properties :
  (has_real_solution a ↔ a = 1) ∧
  ¬(has_imaginary_solution a) := by sorry

end NUMINAMATH_CALUDE_equation_properties_l2996_299658


namespace NUMINAMATH_CALUDE_triangle_condition_l2996_299642

theorem triangle_condition (x y z : ℝ) : 
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 →
  (x + y > z ∧ x + z > y ∧ y + z > x) ↔ (x < 1 ∧ y < 1 ∧ z < 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_condition_l2996_299642


namespace NUMINAMATH_CALUDE_patricks_age_to_roberts_age_ratio_l2996_299620

/-- Given that Robert will turn 30 after 2 years and Patrick is 14 years old now,
    prove that the ratio of Patrick's age to Robert's age is 1:2 -/
theorem patricks_age_to_roberts_age_ratio :
  ∀ (roberts_age patricks_age : ℕ),
  roberts_age + 2 = 30 →
  patricks_age = 14 →
  patricks_age / roberts_age = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_patricks_age_to_roberts_age_ratio_l2996_299620


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l2996_299608

theorem units_digit_of_fraction (n : ℕ) : n = 30 * 31 * 32 * 33 * 34 / 120 → n % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l2996_299608


namespace NUMINAMATH_CALUDE_road_length_proof_l2996_299684

/-- The total length of the road in meters -/
def total_length : ℝ := 1000

/-- The length repaired in the first week in meters -/
def first_week_repair : ℝ := 0.2 * total_length

/-- The length repaired in the second week in meters -/
def second_week_repair : ℝ := 0.25 * total_length

/-- The length repaired in the third week in meters -/
def third_week_repair : ℝ := 480

/-- The length remaining unrepaired in meters -/
def remaining_length : ℝ := 70

theorem road_length_proof :
  first_week_repair + second_week_repair + third_week_repair + remaining_length = total_length := by
  sorry

end NUMINAMATH_CALUDE_road_length_proof_l2996_299684


namespace NUMINAMATH_CALUDE_quadratic_properties_l2996_299603

-- Define the quadratic function
def f (x : ℝ) := -x^2 + 9*x - 20

-- Theorem statement
theorem quadratic_properties :
  (∃ (max : ℝ), ∀ (x : ℝ), f x ≥ 0 → x ≤ max) ∧
  (∃ (max : ℝ), f max ≥ 0 ∧ max = 5) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x^2 - 9*x + 20 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2996_299603


namespace NUMINAMATH_CALUDE_lambda_n_lower_bound_l2996_299602

/-- The ratio of the longest to the shortest distance between any two of n points in the plane -/
def lambda_n (n : ℕ) : ℝ := sorry

/-- Theorem: For n ≥ 4, λ_n ≥ 2 * sin((n-2)π/(2n)) -/
theorem lambda_n_lower_bound (n : ℕ) (h : n ≥ 4) : 
  lambda_n n ≥ 2 * Real.sin ((n - 2) * Real.pi / (2 * n)) := by sorry

end NUMINAMATH_CALUDE_lambda_n_lower_bound_l2996_299602


namespace NUMINAMATH_CALUDE_one_fourth_between_fractions_l2996_299670

theorem one_fourth_between_fractions :
  let start := (1 : ℚ) / 5
  let finish := (4 : ℚ) / 5
  let one_fourth_way := (3 * start + 1 * finish) / (3 + 1)
  one_fourth_way = (7 : ℚ) / 20 := by sorry

end NUMINAMATH_CALUDE_one_fourth_between_fractions_l2996_299670


namespace NUMINAMATH_CALUDE_soccer_league_teams_l2996_299623

/-- The number of teams in a soccer league where each team plays every other team once 
    and the total number of games is 105. -/
def num_teams : ℕ := 15

/-- The total number of games played in the league. -/
def total_games : ℕ := 105

/-- Formula for the number of games in a round-robin tournament. -/
def games_formula (n : ℕ) : ℕ := n * (n - 1) / 2

theorem soccer_league_teams : 
  games_formula num_teams = total_games ∧ num_teams > 0 :=
sorry

end NUMINAMATH_CALUDE_soccer_league_teams_l2996_299623


namespace NUMINAMATH_CALUDE_bisection_next_point_l2996_299619

theorem bisection_next_point 
  (f : ℝ → ℝ) 
  (h_continuous : ContinuousOn f (Set.Icc 1 2))
  (h_f1 : f 1 < 0)
  (h_f1_5 : f 1.5 > 0) :
  (1 + 1.5) / 2 = 1.25 := by sorry

end NUMINAMATH_CALUDE_bisection_next_point_l2996_299619


namespace NUMINAMATH_CALUDE_jessica_red_marbles_l2996_299695

theorem jessica_red_marbles (sandy_marbles : ℕ) (sandy_multiple : ℕ) :
  sandy_marbles = 144 →
  sandy_multiple = 4 →
  (sandy_marbles / sandy_multiple) / 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jessica_red_marbles_l2996_299695


namespace NUMINAMATH_CALUDE_highway_project_employees_l2996_299639

/-- Represents the highway construction project -/
structure HighwayProject where
  initial_workforce : ℕ
  total_length : ℕ
  initial_days : ℕ
  initial_hours_per_day : ℕ
  days_worked : ℕ
  work_completed : ℚ
  remaining_days : ℕ
  new_hours_per_day : ℕ

/-- Calculates the number of additional employees needed to complete the project on time -/
def additional_employees_needed (project : HighwayProject) : ℕ :=
  sorry

/-- Theorem stating that 60 additional employees are needed for the given project -/
theorem highway_project_employees (project : HighwayProject) 
  (h1 : project.initial_workforce = 100)
  (h2 : project.total_length = 2)
  (h3 : project.initial_days = 50)
  (h4 : project.initial_hours_per_day = 8)
  (h5 : project.days_worked = 25)
  (h6 : project.work_completed = 1/3)
  (h7 : project.remaining_days = 25)
  (h8 : project.new_hours_per_day = 10) :
  additional_employees_needed project = 60 :=
sorry

end NUMINAMATH_CALUDE_highway_project_employees_l2996_299639


namespace NUMINAMATH_CALUDE_star_equation_solution_l2996_299697

def star (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

theorem star_equation_solution :
  ∀ A : ℝ, star A 6 = 31 → A = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l2996_299697


namespace NUMINAMATH_CALUDE_least_k_value_l2996_299631

theorem least_k_value (p q k : ℕ) : 
  p > 1 → 
  q > 1 → 
  p + q = 36 → 
  17 * (p + 1) = k * (q + 1) → 
  k ≥ 2 ∧ (∃ (p' q' : ℕ), p' > 1 ∧ q' > 1 ∧ p' + q' = 36 ∧ 17 * (p' + 1) = 2 * (q' + 1)) :=
by sorry

end NUMINAMATH_CALUDE_least_k_value_l2996_299631


namespace NUMINAMATH_CALUDE_age_problem_l2996_299668

/-- Mr. Li's current age -/
def mr_li_age : ℕ := 23

/-- Xiao Ming's current age -/
def xiao_ming_age : ℕ := 10

/-- The age difference between Mr. Li and Xiao Ming -/
def age_difference : ℕ := 13

theorem age_problem :
  (mr_li_age - 6 = xiao_ming_age + 7) ∧
  (mr_li_age + 4 + xiao_ming_age - 5 = 32) ∧
  (mr_li_age = xiao_ming_age + age_difference) :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l2996_299668


namespace NUMINAMATH_CALUDE_cost_price_example_l2996_299644

/-- Given a selling price and a profit percentage, calculate the cost price -/
def cost_price (selling_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  selling_price / (1 + profit_percentage / 100)

/-- Theorem: Given a selling price of 500 and a profit of 25%, the cost price is 400 -/
theorem cost_price_example : cost_price 500 25 = 400 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_example_l2996_299644


namespace NUMINAMATH_CALUDE_max_product_of_three_primes_l2996_299643

theorem max_product_of_three_primes (x y z : ℕ) : 
  Prime x → Prime y → Prime z →
  x ≠ y → x ≠ z → y ≠ z →
  x + y + z = 49 →
  x * y * z ≤ 4199 := by
sorry

end NUMINAMATH_CALUDE_max_product_of_three_primes_l2996_299643


namespace NUMINAMATH_CALUDE_triangle_problem_l2996_299607

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
def Triangle (a b c A B C : ℝ) : Prop :=
  -- Add necessary conditions for a valid triangle here
  True

theorem triangle_problem (a b c A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : Real.sqrt 3 * c * Real.cos A + a * Real.sin C = Real.sqrt 3 * c)
  (h_sum : b + c = 5)
  (h_area : (1/2) * b * c * Real.sin A = Real.sqrt 3) :
  A = π/3 ∧ a = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2996_299607


namespace NUMINAMATH_CALUDE_product_selection_probabilities_l2996_299605

def totalProducts : ℕ := 5
def authenticProducts : ℕ := 3
def defectiveProducts : ℕ := 2

theorem product_selection_probabilities :
  let totalSelections := totalProducts.choose 2
  let bothAuthenticSelections := authenticProducts.choose 2
  let mixedSelections := authenticProducts * defectiveProducts
  (bothAuthenticSelections : ℚ) / totalSelections = 3 / 10 ∧
  (mixedSelections : ℚ) / totalSelections = 3 / 5 ∧
  1 - (bothAuthenticSelections : ℚ) / totalSelections = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_product_selection_probabilities_l2996_299605


namespace NUMINAMATH_CALUDE_distinct_triangles_count_l2996_299675

/-- The number of vertices in our geometric solid -/
def n : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- Combination function -/
def combination (n k : ℕ) : ℕ := 
  Nat.choose n k

theorem distinct_triangles_count : combination n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_distinct_triangles_count_l2996_299675


namespace NUMINAMATH_CALUDE_parallelogram_area_l2996_299667

/-- Represents a parallelogram ABCD with given properties -/
structure Parallelogram where
  perimeter : ℝ
  height_BC : ℝ
  height_CD : ℝ
  perimeter_positive : perimeter > 0
  height_BC_positive : height_BC > 0
  height_CD_positive : height_CD > 0

/-- The area of the parallelogram ABCD is 280 cm² -/
theorem parallelogram_area (ABCD : Parallelogram)
  (h_perimeter : ABCD.perimeter = 75)
  (h_height_BC : ABCD.height_BC = 14)
  (h_height_CD : ABCD.height_CD = 16) :
  ∃ (area : ℝ), area = 280 ∧ (∃ (base : ℝ), base * ABCD.height_BC = area ∧ base * ABCD.height_CD = area) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2996_299667


namespace NUMINAMATH_CALUDE_sum_of_squares_l2996_299662

theorem sum_of_squares (a b : ℝ) : (a^2 + b^2) * (a^2 + b^2 + 4) = 12 → a^2 + b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2996_299662


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2996_299650

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x+8)*(x-6) = -55 + k*x) ↔ (k = -10 + 2*Real.sqrt 21 ∨ k = -10 - 2*Real.sqrt 21) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2996_299650


namespace NUMINAMATH_CALUDE_two_cars_speed_l2996_299685

/-- Two cars traveling in the same direction with given conditions -/
theorem two_cars_speed (t v₁ S₁ S₂ : ℝ) (h_t : t = 30) (h_v₁ : v₁ = 25)
  (h_S₁ : S₁ = 100) (h_S₂ : S₂ = 400) :
  ∃ v₂ : ℝ, (v₂ = 35 ∨ v₂ = 15) ∧
  (S₂ - S₁) / t = |v₂ - v₁| :=
by sorry

end NUMINAMATH_CALUDE_two_cars_speed_l2996_299685


namespace NUMINAMATH_CALUDE_erwans_shopping_trip_l2996_299622

/-- Proves that the price of each shirt is $80 given the conditions of Erwan's shopping trip -/
theorem erwans_shopping_trip (shoe_price : ℝ) (shirt_price : ℝ) :
  shoe_price = 200 →
  (shoe_price * 0.7 + 2 * shirt_price) * 0.95 = 285 →
  shirt_price = 80 :=
by sorry

end NUMINAMATH_CALUDE_erwans_shopping_trip_l2996_299622


namespace NUMINAMATH_CALUDE_concert_attendance_l2996_299676

theorem concert_attendance (total_students : ℕ) (total_attendees : ℕ)
  (h_total : total_students = 1500)
  (h_attendees : total_attendees = 900) :
  ∃ (girls boys girls_attended : ℕ),
    girls + boys = total_students ∧
    (3 * girls + 2 * boys = 5 * total_attendees) ∧
    girls_attended = 643 ∧
    4 * girls_attended = 3 * girls :=
by sorry

end NUMINAMATH_CALUDE_concert_attendance_l2996_299676


namespace NUMINAMATH_CALUDE_f_condition_equivalent_to_a_range_l2996_299679

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x + (1/2) * a * x^2 + a * x

theorem f_condition_equivalent_to_a_range :
  ∀ a : ℝ, (∀ x : ℝ, 2 * Real.exp 1 * f a x + Real.exp 1 + 2 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_condition_equivalent_to_a_range_l2996_299679


namespace NUMINAMATH_CALUDE_candy_bar_price_is_correct_l2996_299604

/-- The selling price of a candy bar that results in a $25 profit when selling 5 boxes of 10 candy bars, each bought for $1. -/
def candy_bar_price : ℚ :=
  let boxes : ℕ := 5
  let bars_per_box : ℕ := 10
  let cost_price : ℚ := 1
  let total_profit : ℚ := 25
  let total_bars : ℕ := boxes * bars_per_box
  (total_profit / total_bars + cost_price)

theorem candy_bar_price_is_correct : candy_bar_price = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_price_is_correct_l2996_299604


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2996_299635

/-- Atomic weight in atomic mass units (amu) -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "H" => 1.008
  | "Br" => 79.904
  | "O" => 15.999
  | "C" => 12.011
  | "N" => 14.007
  | "S" => 32.065
  | _ => 0  -- Default case for unknown elements

/-- Number of atoms of each element in the compound -/
def atom_count (element : String) : ℕ :=
  match element with
  | "H" => 2
  | "Br" => 1
  | "O" => 3
  | "C" => 1
  | "N" => 1
  | "S" => 2
  | _ => 0  -- Default case for elements not in the compound

/-- Calculate the molecular weight of the compound -/
def molecular_weight : ℝ :=
  (atomic_weight "H" * atom_count "H") +
  (atomic_weight "Br" * atom_count "Br") +
  (atomic_weight "O" * atom_count "O") +
  (atomic_weight "C" * atom_count "C") +
  (atomic_weight "N" * atom_count "N") +
  (atomic_weight "S" * atom_count "S")

/-- Theorem stating that the molecular weight of the compound is 220.065 amu -/
theorem compound_molecular_weight : molecular_weight = 220.065 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2996_299635


namespace NUMINAMATH_CALUDE_particle_motion_l2996_299645

/-- Height of the particle in meters after t seconds -/
def s (t : ℝ) : ℝ := 180 * t - 18 * t^2

/-- Time at which the particle reaches its highest point -/
def t_max : ℝ := 5

/-- The highest elevation reached by the particle -/
def h_max : ℝ := 450

theorem particle_motion :
  (∀ t : ℝ, s t ≤ h_max) ∧
  s t_max = h_max :=
sorry

end NUMINAMATH_CALUDE_particle_motion_l2996_299645


namespace NUMINAMATH_CALUDE_extended_tile_ratio_l2996_299637

/-- The ratio of black tiles to white tiles in an extended rectangular pattern -/
theorem extended_tile_ratio (orig_width orig_height : ℕ) 
  (orig_black orig_white : ℕ) : 
  orig_width = 5 → 
  orig_height = 6 → 
  orig_black = 12 → 
  orig_white = 18 → 
  (orig_black : ℚ) / ((orig_white : ℚ) + 2 * (orig_width + orig_height + 2)) = 3 / 11 := by
  sorry

end NUMINAMATH_CALUDE_extended_tile_ratio_l2996_299637


namespace NUMINAMATH_CALUDE_ellipse_line_slope_l2996_299657

/-- The slope of a line passing through the right focus of an ellipse -/
theorem ellipse_line_slope (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := Real.sqrt 3 / 2
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let F := (Real.sqrt (a^2 - b^2), 0)
  ∀ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧ 
    A.2 > 0 ∧ 
    B.2 < 0 ∧
    (A.1 - F.1, A.2 - F.2) = 3 • (F.1 - B.1, F.2 - B.2) →
    (A.2 - B.2) / (A.1 - B.1) = -Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_line_slope_l2996_299657


namespace NUMINAMATH_CALUDE_point_on_angle_negative_pi_third_l2996_299616

/-- Given a point P(2,y) on the terminal side of angle -π/3, prove that y = -2√3 -/
theorem point_on_angle_negative_pi_third (y : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = 2 ∧ P.2 = y ∧ P.2 / P.1 = Real.tan (-π/3)) → 
  y = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_angle_negative_pi_third_l2996_299616


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2996_299671

theorem inequality_solution_set : 
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2996_299671


namespace NUMINAMATH_CALUDE_cara_family_age_difference_l2996_299666

/-- The age difference between Cara's grandmother and Cara's mom -/
def age_difference (cara_age mom_age grandma_age : ℕ) : ℕ :=
  grandma_age - mom_age

theorem cara_family_age_difference :
  ∀ (cara_age mom_age grandma_age : ℕ),
    cara_age = 40 →
    mom_age = cara_age + 20 →
    grandma_age = 75 →
    age_difference cara_age mom_age grandma_age = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_cara_family_age_difference_l2996_299666


namespace NUMINAMATH_CALUDE_overlapping_strips_area_l2996_299612

/-- Represents a rectangular strip with given length and width -/
structure Strip where
  length : ℕ
  width : ℕ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℕ := s.length * s.width

/-- Calculates the number of overlaps between n strips -/
def numOverlaps (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The total area covered by 5 overlapping strips -/
theorem overlapping_strips_area :
  let strips : List Strip := List.replicate 5 ⟨12, 1⟩
  let totalStripArea := (strips.map stripArea).sum
  let overlapArea := numOverlaps 5
  totalStripArea - overlapArea = 50 := by
  sorry


end NUMINAMATH_CALUDE_overlapping_strips_area_l2996_299612


namespace NUMINAMATH_CALUDE_sum_of_squares_l2996_299614

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x * y = 100) : x^2 + y^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2996_299614


namespace NUMINAMATH_CALUDE_tony_school_years_l2996_299630

/-- The total number of years Tony went to school to become an astronaut -/
def total_school_years (first_degree_years : ℕ) (additional_degrees : ℕ) (graduate_degree_years : ℕ) : ℕ :=
  first_degree_years + additional_degrees * first_degree_years + graduate_degree_years

/-- Theorem stating that Tony went to school for 14 years -/
theorem tony_school_years :
  total_school_years 4 2 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_tony_school_years_l2996_299630


namespace NUMINAMATH_CALUDE_three_digit_number_operations_l2996_299621

theorem three_digit_number_operations (a b c : Nat) 
  (h1 : a > 0) 
  (h2 : a < 10) 
  (h3 : b < 10) 
  (h4 : c < 10) : 
  ((2 * a + 3) * 5 + b) * 10 + c - 150 = 100 * a + 10 * b + c := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_operations_l2996_299621


namespace NUMINAMATH_CALUDE_A_eq_union_l2996_299634

/-- The set of real numbers a > 0 such that either y = a^x is not monotonically increasing on R
    or ax^2 - ax + 1 > 0 does not hold for all x ∈ R, but at least one of these conditions is true. -/
def A : Set ℝ :=
  {a : ℝ | a > 0 ∧
    (¬(∀ x y : ℝ, x < y → a^x < a^y) ∨ ¬(∀ x : ℝ, a*x^2 - a*x + 1 > 0)) ∧
    ((∀ x y : ℝ, x < y → a^x < a^y) ∨ (∀ x : ℝ, a*x^2 - a*x + 1 > 0))}

/-- The theorem stating that A is equal to the interval (0,1] union [4,+∞) -/
theorem A_eq_union : A = Set.Ioo 0 1 ∪ Set.Ici 4 := by sorry

end NUMINAMATH_CALUDE_A_eq_union_l2996_299634


namespace NUMINAMATH_CALUDE_relationship_between_x_and_y_l2996_299673

theorem relationship_between_x_and_y (x y : ℝ) 
  (h1 : 2 * x - y > x + 1) 
  (h2 : x + 2 * y < 2 * y - 3) : 
  x < -3 ∧ y < -4 ∧ x > y + 1 := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_y_l2996_299673


namespace NUMINAMATH_CALUDE_truck_fill_rate_l2996_299627

/-- The rate at which a person can fill a truck with stone blocks per hour -/
def fill_rate : ℕ → Prop :=
  λ r => 
    -- Truck capacity
    let capacity : ℕ := 6000
    -- Number of people working initially
    let initial_workers : ℕ := 2
    -- Number of hours initial workers work
    let initial_hours : ℕ := 4
    -- Total number of workers after more join
    let total_workers : ℕ := 8
    -- Number of hours all workers work together
    let final_hours : ℕ := 2
    -- Total time to fill the truck
    let total_time : ℕ := 6

    -- The truck is filled when the sum of blocks filled in both phases equals the capacity
    (initial_workers * initial_hours * r) + (total_workers * final_hours * r) = capacity

theorem truck_fill_rate : fill_rate 250 := by
  sorry

end NUMINAMATH_CALUDE_truck_fill_rate_l2996_299627


namespace NUMINAMATH_CALUDE_square_side_length_l2996_299680

theorem square_side_length (s AF DH BG AE : ℝ) (area_EFGH : ℝ) 
  (h1 : AF = 7)
  (h2 : DH = 4)
  (h3 : BG = 5)
  (h4 : AE = 1)
  (h5 : area_EFGH = 78)
  (h6 : s > 0)
  (h7 : s * s = ((area_EFGH - (AF - DH) * (BG - AE)) * 2) + area_EFGH) :
  s = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2996_299680


namespace NUMINAMATH_CALUDE_max_value_y_plus_one_squared_l2996_299609

theorem max_value_y_plus_one_squared (y : ℝ) : 
  (4 * y^2 + 4 * y + 3 = 1) → ((y + 1)^2 ≤ (1/4 : ℝ)) ∧ (∃ y : ℝ, 4 * y^2 + 4 * y + 3 = 1 ∧ (y + 1)^2 = (1/4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_max_value_y_plus_one_squared_l2996_299609


namespace NUMINAMATH_CALUDE_abs_eq_zero_iff_eq_seven_fifths_l2996_299664

theorem abs_eq_zero_iff_eq_seven_fifths (x : ℝ) : |5*x - 7| = 0 ↔ x = 7/5 := by sorry

end NUMINAMATH_CALUDE_abs_eq_zero_iff_eq_seven_fifths_l2996_299664


namespace NUMINAMATH_CALUDE_negation_equivalence_l2996_299629

theorem negation_equivalence (x y : ℝ) : 
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2996_299629


namespace NUMINAMATH_CALUDE_teds_age_l2996_299694

theorem teds_age (t s : ℕ) : t = 3 * s - 20 → t + s = 76 → t = 52 := by
  sorry

end NUMINAMATH_CALUDE_teds_age_l2996_299694


namespace NUMINAMATH_CALUDE_apple_bag_weight_l2996_299693

theorem apple_bag_weight (empty_weight loaded_weight : ℕ) (num_bags : ℕ) : 
  empty_weight = 500 →
  loaded_weight = 1700 →
  num_bags = 20 →
  (loaded_weight - empty_weight) / num_bags = 60 :=
by sorry

end NUMINAMATH_CALUDE_apple_bag_weight_l2996_299693


namespace NUMINAMATH_CALUDE_badminton_team_combinations_l2996_299632

theorem badminton_team_combinations : 
  ∀ (male_players female_players : ℕ), 
    male_players = 6 → 
    female_players = 7 → 
    (male_players.choose 1) * (female_players.choose 1) = 42 := by
sorry

end NUMINAMATH_CALUDE_badminton_team_combinations_l2996_299632


namespace NUMINAMATH_CALUDE_problem_statement_l2996_299646

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a^2*Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a^2*x - a*Real.log x

theorem problem_statement :
  (∀ a : ℝ, (∃ x_min : ℝ, x_min > 0 ∧ f a x_min = 0 ∧ ∀ x : ℝ, x > 0 → f a x ≥ 0) →
    a = 1 ∨ a = -2 * Real.exp (3/4)) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → g a x ≥ 0) →
    0 ≤ a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2996_299646


namespace NUMINAMATH_CALUDE_building_heights_sum_l2996_299641

theorem building_heights_sum (h1 h2 h3 h4 : ℝ) : 
  h1 = 100 →
  h2 = h1 / 2 →
  h3 = h2 / 2 →
  h4 = h3 / 5 →
  h1 + h2 + h3 + h4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_building_heights_sum_l2996_299641


namespace NUMINAMATH_CALUDE_increase_decrease_percentage_l2996_299636

theorem increase_decrease_percentage (initial : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) : 
  let increased := initial * (1 + increase_percent / 100)
  let final := increased * (1 - decrease_percent / 100)
  initial = 80 ∧ increase_percent = 150 ∧ decrease_percent = 25 → final = 150 := by
  sorry

end NUMINAMATH_CALUDE_increase_decrease_percentage_l2996_299636


namespace NUMINAMATH_CALUDE_square_difference_quotient_l2996_299626

theorem square_difference_quotient : (347^2 - 333^2) / 14 = 680 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_quotient_l2996_299626


namespace NUMINAMATH_CALUDE_definite_integral_exp_abs_x_l2996_299692

theorem definite_integral_exp_abs_x : 
  ∫ x in (-2)..4, Real.exp (|x|) = Real.exp 2 - Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_definite_integral_exp_abs_x_l2996_299692


namespace NUMINAMATH_CALUDE_ordering_of_expressions_l2996_299682

/-- Given a = e^0.1 - 1, b = sin 0.1, and c = ln 1.1, prove that c < b < a -/
theorem ordering_of_expressions :
  let a : ℝ := Real.exp 0.1 - 1
  let b : ℝ := Real.sin 0.1
  let c : ℝ := Real.log 1.1
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ordering_of_expressions_l2996_299682


namespace NUMINAMATH_CALUDE_bread_slices_proof_l2996_299640

theorem bread_slices_proof (S : ℕ) : S ≥ 20 → (∃ T : ℕ, S = 2 * T + 10 ∧ S - 7 = 2 * T + 3) → S ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_bread_slices_proof_l2996_299640


namespace NUMINAMATH_CALUDE_joan_initial_balloons_l2996_299683

/-- The number of balloons Joan lost -/
def lost_balloons : ℕ := 2

/-- The number of balloons Joan currently has -/
def current_balloons : ℕ := 7

/-- The initial number of balloons Joan had -/
def initial_balloons : ℕ := current_balloons + lost_balloons

theorem joan_initial_balloons : initial_balloons = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_initial_balloons_l2996_299683


namespace NUMINAMATH_CALUDE_unique_x_with_square_conditions_l2996_299665

theorem unique_x_with_square_conditions : ∃! (x : ℕ), 
  x > 0 ∧ 
  (∃ (n : ℕ), 2 * x + 1 = n^2) ∧ 
  (∀ (k : ℕ), (2 * x + 2 ≤ k) ∧ (k ≤ 3 * x + 2) → ¬∃ (m : ℕ), k = m^2) ∧
  x = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_with_square_conditions_l2996_299665


namespace NUMINAMATH_CALUDE_sum_inequality_l2996_299610

theorem sum_inequality {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2996_299610


namespace NUMINAMATH_CALUDE_jake_fewer_peaches_l2996_299677

theorem jake_fewer_peaches (steven_peaches jill_peaches : ℕ) 
  (h1 : steven_peaches = 14)
  (h2 : jill_peaches = 5)
  (jake_peaches : ℕ)
  (h3 : jake_peaches = jill_peaches + 3)
  (h4 : jake_peaches < steven_peaches) :
  steven_peaches - jake_peaches = 6 := by
  sorry

end NUMINAMATH_CALUDE_jake_fewer_peaches_l2996_299677


namespace NUMINAMATH_CALUDE_no_integer_solution_l2996_299669

theorem no_integer_solution : ¬∃ (x y : ℤ), 
  (x + 2019) * (x + 2020) + (x + 2020) * (x + 2021) + (x + 2019) * (x + 2021) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2996_299669


namespace NUMINAMATH_CALUDE_race_time_difference_l2996_299656

/-- Proves the time difference between Malcolm and Joshua finishing a race --/
theorem race_time_difference 
  (malcolm_speed : ℝ) 
  (joshua_speed : ℝ) 
  (race_distance : ℝ) 
  (h1 : malcolm_speed = 5)
  (h2 : joshua_speed = 7)
  (h3 : race_distance = 12) :
  joshua_speed * race_distance - malcolm_speed * race_distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l2996_299656


namespace NUMINAMATH_CALUDE_fifth_term_is_nine_l2996_299655

/-- An arithmetic sequence with first term 1 and common difference 2 -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  1 + (n - 1) * 2

/-- The fifth term of the arithmetic sequence is 9 -/
theorem fifth_term_is_nine : arithmetic_sequence 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_nine_l2996_299655


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_l2996_299687

theorem right_triangle_arithmetic_progression (a b c : ℝ) : 
  -- The triangle is right-angled
  a^2 + b^2 = c^2 →
  -- The lengths form an arithmetic progression
  b - a = c - b →
  -- The common difference is 1
  b - a = 1 →
  -- The hypotenuse is 5
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_l2996_299687


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2996_299689

/-- The focal length of the hyperbola x²/10 - y²/2 = 1 is 4√3 -/
theorem hyperbola_focal_length : 
  ∃ (f : ℝ), f = 4 * Real.sqrt 3 ∧ 
  f = 2 * Real.sqrt ((10 : ℝ) + 2) ∧
  ∀ (x y : ℝ), x^2 / 10 - y^2 / 2 = 1 → 
    ∃ (c : ℝ), c = Real.sqrt ((10 : ℝ) + 2) ∧ 
    f = 2 * c :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2996_299689


namespace NUMINAMATH_CALUDE_max_food_per_guest_l2996_299633

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ) (max_food : ℚ) : 
  total_food = 337 → min_guests = 169 → max_food = 2 → 
  max_food = (total_food : ℚ) / min_guests :=
by sorry

end NUMINAMATH_CALUDE_max_food_per_guest_l2996_299633


namespace NUMINAMATH_CALUDE_dimitri_burger_calories_l2996_299615

/-- Calculates the number of calories per burger given the daily burger consumption and total calories over two days. -/
def calories_per_burger (burgers_per_day : ℕ) (total_calories : ℕ) : ℕ :=
  total_calories / (burgers_per_day * 2)

/-- Theorem stating that given Dimitri's burger consumption and calorie intake, each burger contains 20 calories. -/
theorem dimitri_burger_calories :
  calories_per_burger 3 120 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dimitri_burger_calories_l2996_299615


namespace NUMINAMATH_CALUDE_fixed_point_on_graph_l2996_299686

theorem fixed_point_on_graph (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ a^(x + 2) - 3
  f (-2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_graph_l2996_299686


namespace NUMINAMATH_CALUDE_car_discount_proof_l2996_299649

/-- Proves that the discount on a car is 20% of the original price given the specified conditions -/
theorem car_discount_proof (P : ℝ) (D : ℝ) : 
  P > 0 →  -- Assuming positive original price
  D > 0 →  -- Assuming positive discount
  D < P →  -- Discount is less than original price
  (P - D + 0.45 * (P - D)) = (P + 0.16 * P) →  -- Selling price equation
  D = 0.2 * P :=  -- Conclusion: discount is 20% of original price
by sorry  -- Proof is omitted

end NUMINAMATH_CALUDE_car_discount_proof_l2996_299649


namespace NUMINAMATH_CALUDE_number_guessing_game_l2996_299688

theorem number_guessing_game (a b c : ℕ) : 
  a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 ∧ c > 0 ∧ c < 10 →
  ((2 * a + 2) * 5 + b) * 10 + c = 567 →
  a = 4 ∧ b = 6 ∧ c = 7 :=
by sorry

end NUMINAMATH_CALUDE_number_guessing_game_l2996_299688


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2996_299618

/-- The sum of an infinite geometric series with first term 1 and common ratio 1/3 is 3/2 -/
theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/3
  let S : ℝ := ∑' n, a * r^n
  S = 3/2 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2996_299618


namespace NUMINAMATH_CALUDE_evaluate_expression_l2996_299660

theorem evaluate_expression : 6 - 8 * (9 - 4^2) / 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2996_299660


namespace NUMINAMATH_CALUDE_solve_system_l2996_299696

theorem solve_system (p q : ℚ) (eq1 : 5 * p + 3 * q = 7) (eq2 : 2 * p + 5 * q = 8) : p = 11 / 19 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2996_299696


namespace NUMINAMATH_CALUDE_ones_digit_factorial_sum_10_l2996_299651

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def ones_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => factorial (n + 1) + factorial_sum n

theorem ones_digit_factorial_sum_10 :
  ones_digit (factorial_sum 10) = 3 := by sorry

end NUMINAMATH_CALUDE_ones_digit_factorial_sum_10_l2996_299651


namespace NUMINAMATH_CALUDE_fishing_competition_l2996_299690

/-- Fishing Competition Problem -/
theorem fishing_competition (days : ℕ) (jackson_per_day : ℕ) (george_per_day : ℕ) (total_catch : ℕ) :
  days = 5 →
  jackson_per_day = 6 →
  george_per_day = 8 →
  total_catch = 90 →
  ∃ (jonah_per_day : ℕ),
    jonah_per_day = 4 ∧
    total_catch = days * (jackson_per_day + george_per_day + jonah_per_day) :=
by
  sorry


end NUMINAMATH_CALUDE_fishing_competition_l2996_299690


namespace NUMINAMATH_CALUDE_points_below_line_l2996_299601

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d ∧ a₄ = a₃ + d

-- Define the geometric sequence
def geometric_sequence (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∃ q : ℝ, a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q

theorem points_below_line (x₁ x₂ y₁ y₂ : ℝ) :
  arithmetic_sequence 1 x₁ x₂ 2 →
  geometric_sequence 1 y₁ y₂ 2 →
  x₁ > y₁ ∧ x₂ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_points_below_line_l2996_299601


namespace NUMINAMATH_CALUDE_puppy_food_consumption_l2996_299624

/-- Given the cost of a puppy, the duration of food supply, the amount and cost of food per bag,
    and the total cost, this theorem proves the daily food consumption of the puppy. -/
theorem puppy_food_consumption
  (puppy_cost : ℚ)
  (food_duration_weeks : ℕ)
  (food_per_bag : ℚ)
  (cost_per_bag : ℚ)
  (total_cost : ℚ)
  (h1 : puppy_cost = 10)
  (h2 : food_duration_weeks = 3)
  (h3 : food_per_bag = 7/2)
  (h4 : cost_per_bag = 2)
  (h5 : total_cost = 14) :
  (total_cost - puppy_cost) / cost_per_bag * food_per_bag / (food_duration_weeks * 7 : ℚ) = 1/3 :=
sorry

end NUMINAMATH_CALUDE_puppy_food_consumption_l2996_299624
