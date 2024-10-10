import Mathlib

namespace batsman_overall_average_l2949_294991

def total_matches : ℕ := 30
def first_set_matches : ℕ := 20
def second_set_matches : ℕ := 10
def first_set_average : ℕ := 30
def second_set_average : ℕ := 15

theorem batsman_overall_average :
  let first_set_total := first_set_matches * first_set_average
  let second_set_total := second_set_matches * second_set_average
  let total_runs := first_set_total + second_set_total
  total_runs / total_matches = 25 := by sorry

end batsman_overall_average_l2949_294991


namespace negation_equivalence_l2949_294915

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ 
  (∃ x : ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end negation_equivalence_l2949_294915


namespace square_sum_equals_five_l2949_294944

theorem square_sum_equals_five (a b : ℝ) 
  (h1 : a^3 - 3*a*b^2 = 11) 
  (h2 : b^3 - 3*a^2*b = 2) : 
  a^2 + b^2 = 5 := by sorry

end square_sum_equals_five_l2949_294944


namespace total_caps_production_l2949_294921

/-- The total number of caps produced over four weeks, given the production
    of the first three weeks and the fourth week being the average of the first three. -/
theorem total_caps_production
  (week1 : ℕ)
  (week2 : ℕ)
  (week3 : ℕ)
  (h1 : week1 = 320)
  (h2 : week2 = 400)
  (h3 : week3 = 300) :
  week1 + week2 + week3 + (week1 + week2 + week3) / 3 = 1360 := by
  sorry

#eval 320 + 400 + 300 + (320 + 400 + 300) / 3

end total_caps_production_l2949_294921


namespace sally_age_proof_l2949_294914

theorem sally_age_proof (sally_age_five_years_ago : ℕ) : 
  sally_age_five_years_ago = 7 → 
  (sally_age_five_years_ago + 5 + 2 : ℕ) = 14 := by
  sorry

end sally_age_proof_l2949_294914


namespace A_initial_investment_l2949_294998

/-- Represents the initial investment of A in rupees -/
def A_investment : ℝ := sorry

/-- Represents B's contribution to the capital in rupees -/
def B_contribution : ℝ := 15750

/-- Represents the number of months A was in the business -/
def A_months : ℝ := 12

/-- Represents the number of months B was in the business -/
def B_months : ℝ := 4

/-- Represents the ratio of profit division for A -/
def A_profit_ratio : ℝ := 2

/-- Represents the ratio of profit division for B -/
def B_profit_ratio : ℝ := 3

/-- Theorem stating that A's initial investment is 1750 rupees -/
theorem A_initial_investment : 
  A_investment = 1750 :=
by
  sorry

end A_initial_investment_l2949_294998


namespace largest_even_not_sum_of_composite_odds_l2949_294945

/-- A function that checks if a natural number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that checks if a natural number is odd -/
def isOdd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

/-- A function that checks if a natural number can be expressed as the sum of two composite odd positive integers -/
def isSumOfTwoCompositeOdds (n : ℕ) : Prop :=
  ∃ a b, isComposite a ∧ isComposite b ∧ isOdd a ∧ isOdd b ∧ n = a + b

/-- Theorem stating that 38 is the largest even positive integer that cannot be expressed as the sum of two composite odd positive integers -/
theorem largest_even_not_sum_of_composite_odds :
  (∀ n : ℕ, n > 38 → isSumOfTwoCompositeOdds n) ∧
  ¬isSumOfTwoCompositeOdds 38 ∧
  (∀ n : ℕ, n < 38 → n % 2 = 0 → isSumOfTwoCompositeOdds n ∨ n < 38) :=
sorry

end largest_even_not_sum_of_composite_odds_l2949_294945


namespace simplify_expression_l2949_294904

theorem simplify_expression (x : ℝ) : (2*x + 20) + (150*x + 20) = 152*x + 40 := by
  sorry

end simplify_expression_l2949_294904


namespace linear_regression_estimate_l2949_294939

/-- Given a linear regression equation y = 0.50x - 0.81, prove that when x = 25, y = 11.69 -/
theorem linear_regression_estimate (x y : ℝ) : 
  y = 0.50 * x - 0.81 → x = 25 → y = 11.69 := by
  sorry

end linear_regression_estimate_l2949_294939


namespace percentage_of_male_employees_l2949_294930

theorem percentage_of_male_employees 
  (total_employees : ℕ)
  (males_below_50 : ℕ)
  (h1 : total_employees = 800)
  (h2 : males_below_50 = 120)
  (h3 : (males_below_50 : ℝ) = 0.6 * (total_employees * (percentage_males / 100))) :
  percentage_males = 25 := by
  sorry

#check percentage_of_male_employees

end percentage_of_male_employees_l2949_294930


namespace blue_hat_cost_l2949_294959

/-- Proves that the cost of each blue hat is $6 given the conditions of the hat purchase problem. -/
theorem blue_hat_cost (total_hats : ℕ) (green_hat_cost : ℕ) (total_price : ℕ) (green_hats : ℕ) :
  total_hats = 85 →
  green_hat_cost = 7 →
  total_price = 530 →
  green_hats = 20 →
  (total_price - green_hat_cost * green_hats) / (total_hats - green_hats) = 6 := by
sorry

end blue_hat_cost_l2949_294959


namespace min_value_of_c_l2949_294942

theorem min_value_of_c (a b c d e : ℕ) : 
  a > 0 → 
  b = a + 1 → 
  c = b + 1 → 
  d = c + 1 → 
  e = d + 1 → 
  ∃ n : ℕ, a + b + c + d + e = n^3 → 
  ∃ m : ℕ, b + c + d = m^2 → 
  ∀ c' : ℕ, (∃ a' b' d' e' : ℕ, 
    a' > 0 ∧ 
    b' = a' + 1 ∧ 
    d' = c' + 1 ∧ 
    e' = d' + 1 ∧ 
    (∃ n' : ℕ, a' + b' + c' + d' + e' = n'^3) ∧ 
    (∃ m' : ℕ, b' + c' + d' = m'^2)) → 
  c' ≥ c → 
  c = 675 :=
sorry

end min_value_of_c_l2949_294942


namespace sqrt_equation_solution_l2949_294980

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (2 * x + 15) = 12) ∧ (x = 64.5) := by sorry

end sqrt_equation_solution_l2949_294980


namespace employment_agency_payroll_l2949_294913

/-- Calculates the total payroll for an employment agency --/
theorem employment_agency_payroll
  (total_hired : ℕ)
  (num_laborers : ℕ)
  (operator_pay : ℕ)
  (laborer_pay : ℕ)
  (h_total : total_hired = 35)
  (h_laborers : num_laborers = 19)
  (h_operator_pay : operator_pay = 140)
  (h_laborer_pay : laborer_pay = 90) :
  let num_operators := total_hired - num_laborers
  let operator_total := num_operators * operator_pay
  let laborer_total := num_laborers * laborer_pay
  operator_total + laborer_total = 3950 := by
  sorry

#check employment_agency_payroll

end employment_agency_payroll_l2949_294913


namespace expand_product_l2949_294979

theorem expand_product (x : ℝ) : (3*x - 4) * (2*x + 9) = 6*x^2 + 19*x - 36 := by
  sorry

end expand_product_l2949_294979


namespace cube_sum_and_reciprocal_l2949_294932

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -5) : x^3 + 1/x^3 = -110 := by
  sorry

end cube_sum_and_reciprocal_l2949_294932


namespace color_assignment_theorem_l2949_294935

def numbers : List ℕ := List.range 13 |>.map (· + 13)

structure ColorAssignment where
  black : ℕ
  red : List ℕ
  blue : List ℕ
  yellow : List ℕ
  green : List ℕ

def isValidAssignment (ca : ColorAssignment) : Prop :=
  ca.black ∈ numbers ∧
  ca.red.length = 3 ∧ ca.red.all (· ∈ numbers) ∧
  ca.blue.length = 3 ∧ ca.blue.all (· ∈ numbers) ∧
  ca.yellow.length = 3 ∧ ca.yellow.all (· ∈ numbers) ∧
  ca.green.length = 3 ∧ ca.green.all (· ∈ numbers) ∧
  ca.red.sum = ca.blue.sum ∧ ca.blue.sum = ca.yellow.sum ∧ ca.yellow.sum = ca.green.sum ∧
  13 ∈ ca.red ∧ 15 ∈ ca.yellow ∧ 23 ∈ ca.blue ∧
  (ca.black :: ca.red ++ ca.blue ++ ca.yellow ++ ca.green).toFinset = numbers.toFinset

theorem color_assignment_theorem (ca : ColorAssignment) 
  (h : isValidAssignment ca) : 
  ca.black = 19 ∧ ca.green = [14, 21, 22] := by
  sorry

#check color_assignment_theorem

end color_assignment_theorem_l2949_294935


namespace count_numbers_with_three_in_range_l2949_294967

def count_numbers_with_three (lower_bound upper_bound : ℕ) : ℕ :=
  sorry

theorem count_numbers_with_three_in_range : 
  count_numbers_with_three 200 499 = 138 := by
  sorry

end count_numbers_with_three_in_range_l2949_294967


namespace line_properties_l2949_294992

/-- The line l₁ with equation (m + 1)x - (m - 3)y - 8 = 0 where m ∈ ℝ --/
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 1) * x - (m - 3) * y - 8 = 0

/-- The line l₂ parallel to l₁ passing through the origin --/
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m + 1) * x - (m - 3) * y = 0

theorem line_properties :
  (∀ m : ℝ, l₁ m 2 2) ∧ 
  (∀ x y : ℝ, x + y = 0 → (∀ m : ℝ, l₂ m x y) ∧ 
    ∀ a b : ℝ, l₂ m a b → (a^2 + b^2 ≤ x^2 + y^2)) :=
sorry

end line_properties_l2949_294992


namespace locus_of_tangent_circles_l2949_294929

/-- The locus of centers of circles externally tangent to a given circle and a line -/
theorem locus_of_tangent_circles (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    ((x - 0)^2 + (y - 3)^2)^(1/2) = r + 1 ∧
    y = r) → 
  ∃ (a b c : ℝ), a ≠ 0 ∧ (y - b)^2 = 4 * a * (x - c) :=
sorry

end locus_of_tangent_circles_l2949_294929


namespace product_in_base7_l2949_294903

/-- Converts a base 10 number to base 7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Converts a base 7 number to base 10 --/
def fromBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base 7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := sorry

theorem product_in_base7 : 
  multiplyBase7 (toBase7 231) (toBase7 452) = 613260 := by sorry

end product_in_base7_l2949_294903


namespace absolute_value_inequality_l2949_294958

theorem absolute_value_inequality (x : ℝ) : 
  |((7 - x) / 4)| < 3 ↔ 2 < x ∧ x < 19 := by
  sorry

end absolute_value_inequality_l2949_294958


namespace mans_age_twice_sons_l2949_294912

theorem mans_age_twice_sons (son_age : ℕ) (age_difference : ℕ) : 
  son_age = 26 → age_difference = 28 → 
  ∃ (years : ℕ), (son_age + years + age_difference) = 2 * (son_age + years) ∧ years = 2 :=
by sorry

end mans_age_twice_sons_l2949_294912


namespace radius_of_third_circle_l2949_294918

structure Triangle :=
  (a b c : ℝ)

structure Circle :=
  (radius : ℝ)

def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b

def isInscribed (c : Circle) (t : Triangle) : Prop :=
  sorry

def isTangent (c1 c2 : Circle) (t : Triangle) : Prop :=
  sorry

theorem radius_of_third_circle (t : Triangle) (q1 q2 q3 : Circle) :
  t.a = 78 →
  t.b = 78 →
  t.c = 60 →
  isIsosceles t →
  isInscribed q1 t →
  isTangent q2 q1 t →
  isTangent q3 q2 t →
  q3.radius = 320 / 81 :=
sorry

end radius_of_third_circle_l2949_294918


namespace pencil_eraser_cost_problem_l2949_294986

theorem pencil_eraser_cost_problem : ∃ (p e : ℕ), 
  13 * p + 3 * e = 100 ∧ 
  p > e ∧ 
  p + e = 10 := by
sorry

end pencil_eraser_cost_problem_l2949_294986


namespace min_value_expression_min_value_achievable_l2949_294962

theorem min_value_expression (a b : ℤ) (h : a > b) :
  (a + 2*b) / (a - b) + (a - b) / (a + 2*b) ≥ 2 :=
sorry

theorem min_value_achievable :
  ∃ (a b : ℤ), a > b ∧ (a + 2*b) / (a - b) + (a - b) / (a + 2*b) = 2 :=
sorry

end min_value_expression_min_value_achievable_l2949_294962


namespace quadratic_equation_sum_l2949_294911

theorem quadratic_equation_sum (x r s : ℝ) : 
  (15 * x^2 + 30 * x - 450 = 0) →
  ((x + r)^2 = s) →
  (r + s = 32) := by
  sorry

end quadratic_equation_sum_l2949_294911


namespace cylinder_painted_area_l2949_294936

/-- The total painted area of a cylinder with given dimensions and painting conditions -/
theorem cylinder_painted_area (h r : ℝ) (paint_percent : ℝ) : 
  h = 15 → r = 5 → paint_percent = 0.75 → 
  (2 * π * r^2) + (paint_percent * 2 * π * r * h) = 162.5 * π := by
  sorry

end cylinder_painted_area_l2949_294936


namespace jose_bottle_caps_l2949_294987

theorem jose_bottle_caps (initial : Real) (given_away : Real) (remaining : Real) : 
  initial = 7.0 → given_away = 2.0 → remaining = initial - given_away → remaining = 5.0 := by
  sorry

end jose_bottle_caps_l2949_294987


namespace red_highest_probability_l2949_294907

def num_red : ℕ := 5
def num_yellow : ℕ := 4
def num_white : ℕ := 1
def num_blue : ℕ := 3

def total_balls : ℕ := num_red + num_yellow + num_white + num_blue

theorem red_highest_probability :
  (num_red : ℚ) / total_balls > max ((num_yellow : ℚ) / total_balls)
                                    (max ((num_white : ℚ) / total_balls)
                                         ((num_blue : ℚ) / total_balls)) :=
by sorry

end red_highest_probability_l2949_294907


namespace largest_integral_x_l2949_294955

theorem largest_integral_x : ∃ x : ℤ, x = 4 ∧ 
  (∀ y : ℤ, (1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 7/9 → y ≤ x) ∧
  (1/4 : ℚ) < (x : ℚ)/6 ∧ (x : ℚ)/6 < 7/9 := by
  sorry

end largest_integral_x_l2949_294955


namespace next_coincidence_exact_next_coincidence_l2949_294977

def factory_whistle := 18
def train_bell := 24
def fire_alarm := 30

theorem next_coincidence (t : ℕ) : t > 0 ∧ t % factory_whistle = 0 ∧ t % train_bell = 0 ∧ t % fire_alarm = 0 → t ≥ 360 :=
sorry

theorem exact_next_coincidence : ∃ (t : ℕ), t = 360 ∧ t % factory_whistle = 0 ∧ t % train_bell = 0 ∧ t % fire_alarm = 0 :=
sorry

end next_coincidence_exact_next_coincidence_l2949_294977


namespace cube_root_equation_solution_l2949_294919

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  -- Proof goes here
  sorry

end cube_root_equation_solution_l2949_294919


namespace profit_per_meter_l2949_294950

/-- Profit per meter calculation -/
theorem profit_per_meter
  (length : ℕ)
  (selling_price : ℕ)
  (total_profit : ℕ)
  (h1 : length = 40)
  (h2 : selling_price = 8200)
  (h3 : total_profit = 1000) :
  total_profit / length = 25 :=
by
  sorry

end profit_per_meter_l2949_294950


namespace difference_greater_than_twice_l2949_294978

theorem difference_greater_than_twice (a : ℝ) : 
  (∀ x, x - 5 > 2*x ↔ x = a) ↔ a - 5 > 2*a := by sorry

end difference_greater_than_twice_l2949_294978


namespace circle_tangent_to_parallel_lines_l2949_294900

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_tangent_to_parallel_lines (x y : ℚ) :
  -- The circle is tangent to these two lines
  (6 * x - 5 * y = 40 ∨ 6 * x - 5 * y = -20) →
  -- The center lies on this line
  (3 * x + 2 * y = 0) →
  -- The point (20/27, -10/9) is the center of the circle
  x = 20/27 ∧ y = -10/9 := by
  sorry

end circle_tangent_to_parallel_lines_l2949_294900


namespace sin_cos_fourth_power_difference_l2949_294974

theorem sin_cos_fourth_power_difference (θ : ℝ) (h : Real.cos (2 * θ) = Real.sqrt 2 / 3) :
  Real.sin θ ^ 4 - Real.cos θ ^ 4 = -(Real.sqrt 2 / 3) := by
  sorry

end sin_cos_fourth_power_difference_l2949_294974


namespace laura_cycling_distance_l2949_294940

def base_7_to_10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

theorem laura_cycling_distance : base_7_to_10 3 5 1 6 = 1287 := by
  sorry

end laura_cycling_distance_l2949_294940


namespace smallest_valid_distribution_l2949_294924

/-- Represents a distribution of candy pieces to children in a circle. -/
def CandyDistribution := List Nat

/-- Checks if all elements in the list are distinct. -/
def all_distinct (l : List Nat) : Prop :=
  l.Nodup

/-- Checks if all elements in the list are at least 1. -/
def all_at_least_one (l : List Nat) : Prop :=
  ∀ x ∈ l, x ≥ 1

/-- Checks if adjacent elements (including the first and last) have a common factor other than 1. -/
def adjacent_common_factor (l : List Nat) : Prop :=
  ∀ i, ∃ k > 1, k ∣ (l.get! i) ∧ k ∣ (l.get! ((i + 1) % l.length))

/-- Checks if there is no prime that divides all elements in the list. -/
def no_common_prime_divisor (l : List Nat) : Prop :=
  ¬∃ p, Nat.Prime p ∧ ∀ x ∈ l, p ∣ x

/-- Checks if a candy distribution satisfies all conditions. -/
def valid_distribution (d : CandyDistribution) : Prop :=
  d.length = 7 ∧
  all_distinct d ∧
  all_at_least_one d ∧
  adjacent_common_factor d ∧
  no_common_prime_divisor d

/-- The main theorem stating that 44 is the smallest number of candy pieces
    that satisfies all conditions for seven children. -/
theorem smallest_valid_distribution :
  (∃ d : CandyDistribution, valid_distribution d ∧ d.sum = 44) ∧
  (∀ d : CandyDistribution, valid_distribution d → d.sum ≥ 44) :=
sorry

end smallest_valid_distribution_l2949_294924


namespace arithmetic_and_geometric_sequences_l2949_294949

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℤ := 2*n - 12

-- Define the sum of the geometric sequence b_n
def S (n : ℕ) : ℤ := 4*(1 - 3^n)

theorem arithmetic_and_geometric_sequences :
  -- Conditions for a_n
  (a 3 = -6) ∧ (a 6 = 0) ∧
  -- Arithmetic sequence property
  (∀ n : ℕ, a (n+1) - a n = a (n+2) - a (n+1)) ∧
  -- Conditions for b_n
  (∃ b : ℕ → ℤ, b 1 = -8 ∧ b 2 = a 1 + a 2 + a 3 ∧
  -- Geometric sequence property
  (∀ n : ℕ, n ≥ 1 → b (n+1) / b n = b 2 / b 1) ∧
  -- S_n is the sum of the first n terms of b_n
  (∀ n : ℕ, n ≥ 1 → S n = (b 1) * (1 - (b 2 / b 1)^n) / (1 - b 2 / b 1))) := by
  sorry

#check arithmetic_and_geometric_sequences

end arithmetic_and_geometric_sequences_l2949_294949


namespace arc_length_from_central_angle_l2949_294993

theorem arc_length_from_central_angle (D : Real) (EF : Real) (DEF : Real) : 
  D = 80 → DEF = 45 → EF = 10 := by
  sorry

end arc_length_from_central_angle_l2949_294993


namespace unique_quadratic_solution_l2949_294956

theorem unique_quadratic_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 + a * x + 1 = 0) → a = 4 := by
  sorry

end unique_quadratic_solution_l2949_294956


namespace inequality_solution_l2949_294916

theorem inequality_solution (x : ℝ) : (x - 1) / (x - 3) ≥ 3 ↔ x ∈ Set.Ioo 3 4 ∪ {4} :=
by sorry

end inequality_solution_l2949_294916


namespace camel_cost_l2949_294934

/-- Represents the cost of different animals in Rupees -/
structure AnimalCosts where
  camel : ℝ
  horse : ℝ
  ox : ℝ
  elephant : ℝ
  giraffe : ℝ
  zebra : ℝ
  llama : ℝ

/-- The conditions given in the problem -/
def problem_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  16 * costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  3 * costs.elephant = 5 * costs.giraffe ∧
  8 * costs.giraffe = 12 * costs.zebra ∧
  20 * costs.zebra = 7 * costs.llama ∧
  10 * costs.elephant = 120000

theorem camel_cost (costs : AnimalCosts) :
  problem_conditions costs → costs.camel = 4800 := by
  sorry

end camel_cost_l2949_294934


namespace vector_simplification_l2949_294971

variable (V : Type*) [AddCommGroup V]
variable (A B C D : V)

theorem vector_simplification (h : A + (B - A) + (C - B) = C) :
  (B - A) + (C - B) - (C - A) - (D - C) = C - D :=
sorry

end vector_simplification_l2949_294971


namespace unique_seating_arrangement_l2949_294946

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  rows_with_8 : ℕ
  rows_with_7 : ℕ

/-- Checks if a seating arrangement is valid --/
def is_valid (s : SeatingArrangement) : Prop :=
  s.rows_with_8 * 8 + s.rows_with_7 * 7 = 55

/-- Theorem stating the unique valid seating arrangement --/
theorem unique_seating_arrangement :
  ∃! s : SeatingArrangement, is_valid s ∧ s.rows_with_8 = 6 := by sorry

end unique_seating_arrangement_l2949_294946


namespace c4h1o_molecular_weight_l2949_294947

/-- The atomic weight of carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The number of carbon atoms in the compound -/
def carbon_count : ℕ := 4

/-- The number of hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 1

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 1

/-- The molecular weight of C4H1O -/
def molecular_weight : ℝ :=
  carbon_weight * carbon_count + hydrogen_weight * hydrogen_count + oxygen_weight * oxygen_count

theorem c4h1o_molecular_weight :
  molecular_weight = 65.048 := by sorry

end c4h1o_molecular_weight_l2949_294947


namespace division_into_proportional_parts_l2949_294970

theorem division_into_proportional_parts :
  let total : ℚ := 156
  let proportions : List ℚ := [2, 1/2, 1/4, 1/8]
  let parts := proportions.map (λ p => p * (total / proportions.sum))
  parts[2] = 13 + 15/23 := by sorry

end division_into_proportional_parts_l2949_294970


namespace three_numbers_problem_l2949_294996

theorem three_numbers_problem (x y z : ℝ) : 
  (x / y = y / z) ∧ 
  (x - (y + z) = 2) ∧ 
  (x + (y - z) / 2 = 9) →
  ((x = 8 ∧ y = 4 ∧ z = 2) ∨ (x = -6.4 ∧ y = 11.2 ∧ z = -19.6)) :=
by sorry

end three_numbers_problem_l2949_294996


namespace smallest_sum_of_roots_l2949_294975

theorem smallest_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + 3*a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 4*b*x + 2*a = 0) :
  a + b ≥ (10/9)^(1/3) := by
sorry

end smallest_sum_of_roots_l2949_294975


namespace grade_assignment_count_l2949_294952

/-- The number of different grades that can be assigned to each student. -/
def numGrades : ℕ := 4

/-- The number of students in the class. -/
def numStudents : ℕ := 15

/-- The theorem stating the number of ways to assign grades to students. -/
theorem grade_assignment_count : numGrades ^ numStudents = 1073741824 := by
  sorry

end grade_assignment_count_l2949_294952


namespace smallest_value_3a_plus_1_l2949_294964

theorem smallest_value_3a_plus_1 (a : ℂ) (h : 8 * a^2 + 6 * a + 2 = 0) :
  ∃ (z : ℂ), 3 * z + 1 = -1/8 ∧ ∀ (w : ℂ), 8 * w^2 + 6 * w + 2 = 0 → Complex.re (3 * w + 1) ≥ -1/8 :=
sorry

end smallest_value_3a_plus_1_l2949_294964


namespace range_of_m_l2949_294938

def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 9 := by
  sorry

end range_of_m_l2949_294938


namespace duck_flight_days_l2949_294922

/-- The number of days it takes for a duck to fly south in winter. -/
def days_south : ℕ := sorry

/-- The number of days it takes for a duck to fly north in summer. -/
def days_north : ℕ := 2 * days_south

/-- The number of days it takes for a duck to fly east in spring. -/
def days_east : ℕ := 60

/-- The total number of days the duck flies during winter, summer, and spring. -/
def total_days : ℕ := 180

/-- Theorem stating that the number of days it takes for the duck to fly south in winter is 40. -/
theorem duck_flight_days : days_south = 40 := by
  sorry

end duck_flight_days_l2949_294922


namespace smallest_prime_divisor_of_quadratic_l2949_294905

theorem smallest_prime_divisor_of_quadratic : 
  ∃ (p : ℕ), Prime p ∧ 
  (∃ (n : ℤ), (n^2 + n + 11).natAbs % p = 0) ∧
  (∀ (q : ℕ), Prime q → q < p → 
    ∀ (m : ℤ), (m^2 + m + 11).natAbs % q ≠ 0) ∧
  p = 11 := by
sorry

end smallest_prime_divisor_of_quadratic_l2949_294905


namespace last_two_nonzero_digits_70_factorial_l2949_294902

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- Function to get the last two nonzero digits of a number -/
def lastTwoNonzeroDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the last two nonzero digits of 70! are 44 -/
theorem last_two_nonzero_digits_70_factorial :
  lastTwoNonzeroDigits (factorial 70) = 44 := by sorry

end last_two_nonzero_digits_70_factorial_l2949_294902


namespace other_number_proof_l2949_294927

theorem other_number_proof (a b : ℕ+) : 
  Nat.gcd a b = 12 → 
  Nat.lcm a b = 396 → 
  a = 36 → 
  b = 132 := by
sorry

end other_number_proof_l2949_294927


namespace linda_basketball_scores_l2949_294973

theorem linda_basketball_scores (first_seven : List Nat) 
  (h1 : first_seven = [5, 6, 4, 7, 3, 2, 6])
  (h2 : first_seven.length = 7)
  (eighth_game : Nat) (ninth_game : Nat)
  (h3 : eighth_game < 10)
  (h4 : ninth_game < 10)
  (h5 : (first_seven.sum + eighth_game) % 8 = 0)
  (h6 : (first_seven.sum + eighth_game + ninth_game) % 9 = 0) :
  eighth_game * ninth_game = 35 := by
sorry

end linda_basketball_scores_l2949_294973


namespace oplus_three_one_l2949_294999

-- Define the operation ⊕ for real numbers
def oplus (a b : ℝ) : ℝ := 3 * a + 4 * b

-- State the theorem
theorem oplus_three_one : oplus 3 1 = 13 := by
  sorry

end oplus_three_one_l2949_294999


namespace luke_spent_3_dollars_per_week_l2949_294906

def luke_problem (lawn_income weed_income total_weeks : ℕ) : Prop :=
  let total_income := lawn_income + weed_income
  total_income / total_weeks = 3

theorem luke_spent_3_dollars_per_week :
  luke_problem 9 18 9 := by
  sorry

end luke_spent_3_dollars_per_week_l2949_294906


namespace inequality_system_solution_l2949_294937

theorem inequality_system_solution : 
  {x : ℤ | x > 0 ∧ 
           (1 + 2*x : ℚ)/4 - (1 - 3*x)/10 > -1/5 ∧ 
           (3*x - 1 : ℚ) < 2*(x + 1)} = {1, 2} := by
  sorry

end inequality_system_solution_l2949_294937


namespace simplify_polynomial_expression_l2949_294920

theorem simplify_polynomial_expression (x : ℝ) :
  6 * x^2 + 4 * x + 9 - (7 - 5 * x - 9 * x^3 + 8 * x^2) = 9 * x^3 - 2 * x^2 + 9 * x + 2 := by
  sorry

end simplify_polynomial_expression_l2949_294920


namespace minimal_sum_for_equal_last_digits_l2949_294994

theorem minimal_sum_for_equal_last_digits (m n : ℕ) : 
  n > m ∧ m ≥ 1 ∧ 
  (1978^m : ℕ) % 1000 = (1978^n : ℕ) % 1000 ∧
  (∀ m' n' : ℕ, n' > m' ∧ m' ≥ 1 ∧ 
    (1978^m' : ℕ) % 1000 = (1978^n' : ℕ) % 1000 → 
    m + n ≤ m' + n') →
  m = 3 ∧ n = 103 := by
sorry

end minimal_sum_for_equal_last_digits_l2949_294994


namespace triangle_theorem_l2949_294928

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle_at_A : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the theorem
theorem triangle_theorem (A B C D E : ℝ × ℝ) :
  Triangle A B C →
  D ∈ Circle A (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) →
  E ∈ Circle A (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) →
  D.1 = C.1 ∧ D.2 = C.2 →
  E.1 = C.1 ∧ E.2 = C.2 →
  Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 20 →
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 16 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 936 := by
sorry


end triangle_theorem_l2949_294928


namespace janelle_blue_marble_bags_l2949_294901

/-- Calculates the number of bags of blue marbles Janelle bought -/
def blue_marble_bags (initial_green : ℕ) (marbles_per_bag : ℕ) (green_given : ℕ) (blue_given : ℕ) (total_remaining : ℕ) : ℕ :=
  ((total_remaining + green_given + blue_given) - initial_green) / marbles_per_bag

/-- Proves that Janelle bought 6 bags of blue marbles -/
theorem janelle_blue_marble_bags :
  blue_marble_bags 26 10 6 8 72 = 6 := by
  sorry

#eval blue_marble_bags 26 10 6 8 72

end janelle_blue_marble_bags_l2949_294901


namespace amy_math_problems_l2949_294909

/-- The number of math problems Amy had to solve -/
def math_problems : ℕ := sorry

/-- The number of spelling problems Amy had to solve -/
def spelling_problems : ℕ := 6

/-- The number of problems Amy can finish in an hour -/
def problems_per_hour : ℕ := 4

/-- The number of hours it took Amy to finish all problems -/
def total_hours : ℕ := 6

/-- Theorem stating that Amy had 18 math problems -/
theorem amy_math_problems : 
  math_problems = 18 := by sorry

end amy_math_problems_l2949_294909


namespace variable_value_l2949_294917

theorem variable_value (x : ℝ) : 5 / (4 + 1 / x) = 1 → x = 1 := by
  sorry

end variable_value_l2949_294917


namespace first_term_of_sequence_l2949_294910

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first 30 terms
  sum_30 : ℚ
  -- The sum of terms from 31st to 80th
  sum_31_to_80 : ℚ
  -- Property: The sum of the first 30 terms is 300
  sum_30_eq : sum_30 = 300
  -- Property: The sum of terms from 31st to 80th is 3750
  sum_31_to_80_eq : sum_31_to_80 = 3750

/-- The first term of the arithmetic sequence is -217/16 -/
theorem first_term_of_sequence (seq : ArithmeticSequence) : 
  ∃ (a d : ℚ), a = -217/16 ∧ 
  (∀ n : ℕ, n > 0 → n ≤ 30 → seq.sum_30 = (n/2) * (2*a + (n-1)*d)) ∧
  (seq.sum_31_to_80 = 25 * (2*a + 109*d)) :=
sorry

end first_term_of_sequence_l2949_294910


namespace line_properties_l2949_294963

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + a^2 * y + 1 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a^2 + 1) * x - b * y + 3 = 0

-- Define parallel lines
def parallel (a b : ℝ) : Prop := ∀ x y, l₁ a x y → l₂ a b x y

-- Define perpendicular lines
def perpendicular (a b : ℝ) : Prop := ∀ x y, l₁ a x y → l₂ a b x y

theorem line_properties (a b : ℝ) :
  (b = -2 ∧ parallel a b → a = 1 ∨ a = -1) ∧
  (perpendicular a b → ∀ c d : ℝ, perpendicular c d → |a * b| ≤ |c * d| ∧ |a * b| = 2) :=
sorry

end line_properties_l2949_294963


namespace ab_positive_necessary_not_sufficient_l2949_294961

/-- Predicate to check if the equation ax^2 + by^2 = 1 represents an ellipse -/
def is_ellipse (a b : ℝ) : Prop := sorry

/-- Theorem stating that ab > 0 is a necessary but not sufficient condition for ax^2 + by^2 = 1 to represent an ellipse -/
theorem ab_positive_necessary_not_sufficient :
  (∀ a b : ℝ, is_ellipse a b → a * b > 0) ∧
  ¬(∀ a b : ℝ, a * b > 0 → is_ellipse a b) := by sorry

end ab_positive_necessary_not_sufficient_l2949_294961


namespace lowest_divisible_by_one_and_two_l2949_294954

theorem lowest_divisible_by_one_and_two : 
  ∃ n : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2 → k ∣ n) ∧ 
  (∀ m : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2 → k ∣ m) → n ≤ m) :=
by sorry

end lowest_divisible_by_one_and_two_l2949_294954


namespace separation_of_homologous_chromosomes_unique_l2949_294985

-- Define the cell division processes
inductive CellDivisionProcess
  | ChromosomeReplication
  | SeparationOfHomologousChromosomes
  | SeparationOfChromatids
  | Cytokinesis

-- Define the types of cell division
inductive CellDivision
  | Mitosis
  | Meiosis

-- Define a function that determines if a process occurs in a given cell division
def occursIn (process : CellDivisionProcess) (division : CellDivision) : Prop :=
  match division with
  | CellDivision.Mitosis =>
    process ≠ CellDivisionProcess.SeparationOfHomologousChromosomes
  | CellDivision.Meiosis => True

-- Theorem statement
theorem separation_of_homologous_chromosomes_unique :
  ∀ (process : CellDivisionProcess),
    (occursIn process CellDivision.Meiosis ∧ ¬occursIn process CellDivision.Mitosis) →
    process = CellDivisionProcess.SeparationOfHomologousChromosomes :=
by sorry

end separation_of_homologous_chromosomes_unique_l2949_294985


namespace megan_folders_l2949_294982

/-- Given the initial number of files, number of deleted files, and files per folder,
    calculate the number of folders needed to store all remaining files. -/
def folders_needed (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : ℕ :=
  let remaining_files := initial_files - deleted_files
  (remaining_files + files_per_folder - 1) / files_per_folder

/-- Prove that given 237 initial files, 53 deleted files, and 12 files per folder,
    the number of folders needed is 16. -/
theorem megan_folders : folders_needed 237 53 12 = 16 := by
  sorry

end megan_folders_l2949_294982


namespace y_value_proof_l2949_294983

/-- Proves that y = 8 on an equally spaced number line from 0 to 32 with 8 steps,
    where y is 2 steps before the midpoint -/
theorem y_value_proof (total_distance : ℝ) (num_steps : ℕ) (y : ℝ) :
  total_distance = 32 →
  num_steps = 8 →
  y = (total_distance / 2) - 2 * (total_distance / num_steps) →
  y = 8 := by
  sorry

end y_value_proof_l2949_294983


namespace molecular_weight_h2o_is_18_l2949_294925

/-- The molecular weight of dihydrogen monoxide in grams per mole -/
def molecular_weight_h2o : ℝ := 18

/-- The number of moles of dihydrogen monoxide -/
def moles_h2o : ℝ := 7

/-- The total weight of dihydrogen monoxide in grams -/
def total_weight_h2o : ℝ := 126

/-- Theorem: The molecular weight of dihydrogen monoxide is 18 grams per mole -/
theorem molecular_weight_h2o_is_18 :
  molecular_weight_h2o = total_weight_h2o / moles_h2o :=
by sorry

end molecular_weight_h2o_is_18_l2949_294925


namespace forest_logging_time_l2949_294926

/-- Represents a logging team with its characteristics -/
structure LoggingTeam where
  loggers : ℕ
  daysPerWeek : ℕ
  treesPerLoggerPerDay : ℕ

/-- Calculates the number of months needed to cut down all trees in the forest -/
def monthsToLogForest (forestWidth : ℕ) (forestLength : ℕ) (treesPerSquareMile : ℕ) 
  (teams : List LoggingTeam) (daysPerMonth : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that it takes 5 months to log the entire forest -/
theorem forest_logging_time : 
  let forestWidth := 4
  let forestLength := 6
  let treesPerSquareMile := 600
  let teamA := LoggingTeam.mk 6 5 5
  let teamB := LoggingTeam.mk 8 4 6
  let teamC := LoggingTeam.mk 10 3 8
  let teamD := LoggingTeam.mk 12 2 10
  let teams := [teamA, teamB, teamC, teamD]
  let daysPerMonth := 30
  monthsToLogForest forestWidth forestLength treesPerSquareMile teams daysPerMonth = 5 :=
  sorry

end forest_logging_time_l2949_294926


namespace newspaper_cost_difference_l2949_294976

/-- Calculates the annual cost difference between Juanita's newspaper purchases and Grant's subscription --/
theorem newspaper_cost_difference : 
  let grant_base_cost : ℝ := 200
  let grant_loyalty_discount : ℝ := 0.1
  let grant_summer_discount : ℝ := 0.05
  let juanita_mon_wed_price : ℝ := 0.5
  let juanita_thu_fri_price : ℝ := 0.6
  let juanita_sat_price : ℝ := 0.8
  let juanita_sun_price : ℝ := 3
  let juanita_monthly_coupon : ℝ := 0.25
  let juanita_holiday_surcharge : ℝ := 0.5
  let weeks_per_year : ℕ := 52
  let months_per_year : ℕ := 12
  let summer_months : ℕ := 2

  let grant_annual_cost := grant_base_cost * (1 - grant_loyalty_discount) - 
    (grant_base_cost / months_per_year) * summer_months * grant_summer_discount

  let juanita_weekly_cost := 3 * juanita_mon_wed_price + 2 * juanita_thu_fri_price + 
    juanita_sat_price + juanita_sun_price

  let juanita_annual_cost := juanita_weekly_cost * weeks_per_year - 
    juanita_monthly_coupon * months_per_year + juanita_holiday_surcharge * months_per_year

  juanita_annual_cost - grant_annual_cost = 162.5 := by sorry

end newspaper_cost_difference_l2949_294976


namespace function_zeros_bound_l2949_294966

open Real

theorem function_zeros_bound (m : ℝ) (x₁ x₂ : ℝ) :
  let f := fun (x : ℝ) => (sin x) / (exp x) - x^2 + π * x
  (0 ≤ x₁ ∧ x₁ ≤ π) →
  (0 ≤ x₂ ∧ x₂ ≤ π) →
  f x₁ = m →
  f x₂ = m →
  x₁ ≠ x₂ →
  |x₂ - x₁| ≤ π - (2 * m) / (π + 1) :=
by sorry

end function_zeros_bound_l2949_294966


namespace puppy_price_is_five_l2949_294957

/-- The price of a kitten in dollars -/
def kitten_price : ℕ := 6

/-- The number of kittens sold -/
def kittens_sold : ℕ := 2

/-- The total earnings from all pets sold in dollars -/
def total_earnings : ℕ := 17

/-- The price of the puppy in dollars -/
def puppy_price : ℕ := total_earnings - (kitten_price * kittens_sold)

theorem puppy_price_is_five : puppy_price = 5 := by
  sorry

end puppy_price_is_five_l2949_294957


namespace inequality_solution_l2949_294981

theorem inequality_solution (x : ℝ) : 
  (x - 1) / ((x - 3)^2) < 0 ↔ x < 1 ∧ x ≠ 3 :=
by sorry

end inequality_solution_l2949_294981


namespace round_4995000_to_million_l2949_294923

/-- Round a natural number to the nearest million -/
def round_to_million (n : ℕ) : ℕ :=
  if n % 1000000 ≥ 500000 then
    ((n + 500000) / 1000000) * 1000000
  else
    (n / 1000000) * 1000000

/-- Theorem: Rounding 4995000 to the nearest million equals 5000000 -/
theorem round_4995000_to_million :
  round_to_million 4995000 = 5000000 := by
  sorry

end round_4995000_to_million_l2949_294923


namespace polynomial_irreducibility_l2949_294943

theorem polynomial_irreducibility (n : ℕ) (hn : n > 1) :
  ¬∃ (g h : Polynomial ℤ), (Polynomial.degree g ≥ 1) ∧ (Polynomial.degree h ≥ 1) ∧
  (x^n + 5 * x^(n-1) + 3 : Polynomial ℤ) = g * h :=
by sorry

end polynomial_irreducibility_l2949_294943


namespace simplify_fraction_l2949_294995

theorem simplify_fraction : (5^6 + 5^3) / (5^5 - 5^2) = 315 / 62 := by
  sorry

end simplify_fraction_l2949_294995


namespace pete_backward_speed_l2949_294968

/-- Represents the speeds of various activities in miles per hour -/
structure Speeds where
  susan_forward : ℝ
  pete_backward : ℝ
  tracy_cartwheel : ℝ
  pete_hands : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : Speeds) : Prop :=
  s.pete_backward = 3 * s.susan_forward ∧
  s.tracy_cartwheel = 2 * s.susan_forward ∧
  s.pete_hands = 1/4 * s.tracy_cartwheel ∧
  s.pete_hands = 2

/-- The theorem stating Pete's backward walking speed -/
theorem pete_backward_speed (s : Speeds) 
  (h : problem_conditions s) : s.pete_backward = 12 := by
  sorry


end pete_backward_speed_l2949_294968


namespace quadratic_function_points_range_l2949_294931

theorem quadratic_function_points_range (m n y₁ y₂ : ℝ) : 
  y₁ = (m - 2)^2 + n → 
  y₂ = (m - 1)^2 + n → 
  y₁ < y₂ → 
  m > 3/2 := by
sorry

end quadratic_function_points_range_l2949_294931


namespace hyperbola_properties_l2949_294988

/-- Given a hyperbola with specific properties, prove its equation and a property of its intersection with a line --/
theorem hyperbola_properties (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let e : ℝ := Real.sqrt 3
  let vertex : ℝ × ℝ := (Real.sqrt 3, 0)
  ∀ x y, C x y →
    (∃ c, c > 0 ∧ c^2 = a^2 + b^2 ∧ c / a = e) →
    (C (Real.sqrt 3) 0) →
    (∃ F₂ : ℝ × ℝ, F₂.1 > 0 ∧
      (∀ x y, (y - F₂.2) = Real.sqrt 3 / 3 * (x - F₂.1) →
        C x y →
        ∃ A B : ℝ × ℝ, A ≠ B ∧ C A.1 A.2 ∧ C B.1 B.2 ∧
          Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 * Real.sqrt 3 / 5)) →
  C x y ↔ x^2 / 3 - y^2 / 6 = 1 := by
sorry

end hyperbola_properties_l2949_294988


namespace isosceles_triangle_exists_l2949_294990

/-- A regular polygon with (2n-1) sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n-1) → ℝ × ℝ

/-- A subset of n vertices from a (2n-1)-gon -/
def VertexSubset (n : ℕ) := Fin n → Fin (2*n-1)

/-- Predicate to check if three vertices form an isosceles triangle -/
def IsIsosceles (p : RegularPolygon n) (a b c : Fin (2*n-1)) : Prop :=
  let va := p.vertices a
  let vb := p.vertices b
  let vc := p.vertices c
  (va.1 - vc.1)^2 + (va.2 - vc.2)^2 = (vb.1 - vc.1)^2 + (vb.2 - vc.2)^2

/-- Main theorem: In any subset of n vertices of a (2n-1)-gon, there exists an isosceles triangle -/
theorem isosceles_triangle_exists (n : ℕ) (h : n ≥ 3) (p : RegularPolygon n) (s : VertexSubset n) :
  ∃ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ IsIsosceles p (s a) (s b) (s c) :=
sorry

end isosceles_triangle_exists_l2949_294990


namespace distinct_tetrahedra_count_l2949_294960

/-- A type representing a thin rod with a length -/
structure Rod where
  length : ℝ
  positive : length > 0

/-- A type representing a set of 6 rods -/
structure SixRods where
  rods : Fin 6 → Rod
  distinct : ∀ i j, i ≠ j → rods i ≠ rods j

/-- A predicate stating that any three rods can form a triangle -/
def can_form_triangle (sr : SixRods) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    (sr.rods i).length + (sr.rods j).length > (sr.rods k).length

/-- A type representing a tetrahedral edge framework -/
structure Tetrahedron where
  edges : Fin 6 → Rod

/-- A function to count distinct tetrahedral edge frameworks -/
noncomputable def count_distinct_tetrahedra (sr : SixRods) : ℕ := sorry

/-- The main theorem -/
theorem distinct_tetrahedra_count (sr : SixRods) 
  (h : can_form_triangle sr) : 
  count_distinct_tetrahedra sr = 30 := by sorry

end distinct_tetrahedra_count_l2949_294960


namespace sequence_a_formula_l2949_294948

def sequence_a (n : ℕ) : ℚ :=
  if n = 1 then -1
  else 2 / (n * (n + 1))

def S (n : ℕ) : ℚ := -2 / (n + 1)

theorem sequence_a_formula (n : ℕ) :
  (n = 1 ∧ sequence_a n = -1) ∨
  (n ≥ 2 ∧ sequence_a n = 2 / (n * (n + 1))) ∧
  (∀ k ≥ 2, (S k)^2 - (sequence_a k) * (S k) = 2 * (sequence_a k)) :=
sorry

end sequence_a_formula_l2949_294948


namespace nonnegative_integer_solution_l2949_294997

theorem nonnegative_integer_solution (x y z : ℕ) :
  (16 / 3 : ℝ)^x * (27 / 25 : ℝ)^y * (5 / 4 : ℝ)^z = 256 →
  x + y + z = 6 := by
sorry

end nonnegative_integer_solution_l2949_294997


namespace odd_function_sum_l2949_294984

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) (h_odd : IsOdd f) (h_f_neg_one : f (-1) = 2) :
  f 0 + f 1 = -2 := by
  sorry

end odd_function_sum_l2949_294984


namespace courtyard_length_l2949_294953

/-- The length of a rectangular courtyard given specific conditions -/
theorem courtyard_length (width : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) : 
  width = 20 → 
  num_stones = 100 → 
  stone_length = 4 → 
  stone_width = 2 → 
  (width * (num_stones * stone_length * stone_width / width) : ℝ) = 40 := by
  sorry

end courtyard_length_l2949_294953


namespace count_ones_in_500_pages_l2949_294969

/-- Count the occurrences of digit 1 in a number -/
def countOnesInNumber (n : ℕ) : ℕ := sorry

/-- Count the occurrences of digit 1 in page numbers from 1 to n -/
def countOnesInPages (n : ℕ) : ℕ :=
  (List.range n).map countOnesInNumber |>.sum

theorem count_ones_in_500_pages :
  countOnesInPages 500 = 200 := by sorry

end count_ones_in_500_pages_l2949_294969


namespace jack_christina_lindy_meeting_l2949_294908

/-- The problem of Jack, Christina, and Lindy meeting --/
theorem jack_christina_lindy_meeting 
  (jack_speed : ℝ) 
  (christina_speed : ℝ) 
  (lindy_speed : ℝ) 
  (lindy_distance : ℝ) : 
  jack_speed = 3 → 
  christina_speed = 3 → 
  lindy_speed = 10 → 
  lindy_distance = 400 → 
  ∃ (initial_distance : ℝ), 
    initial_distance = 240 ∧ 
    (initial_distance / 2) / jack_speed = lindy_distance / lindy_speed :=
sorry

end jack_christina_lindy_meeting_l2949_294908


namespace transformed_quadratic_equation_l2949_294951

theorem transformed_quadratic_equation 
  (p q : ℝ) 
  (x₁ x₂ : ℝ) 
  (h1 : x₁^2 + p*x₁ + q = 0) 
  (h2 : x₂^2 + p*x₂ + q = 0) 
  (h3 : x₁ ≠ x₂) :
  ∃ (t : ℝ), q*t^2 + (q+1)*t + 1 = 0 ↔ 
    (t = x₁ + 1/x₁ ∨ t = x₂ + 1/x₂) :=
by sorry

end transformed_quadratic_equation_l2949_294951


namespace minimum_a_value_l2949_294933

theorem minimum_a_value (a : ℝ) : 
  (∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ↔ 
  a ≥ Real.sqrt 2 := by
sorry

end minimum_a_value_l2949_294933


namespace unique_function_satisfying_equation_l2949_294989

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = x + y :=
by
  sorry

end unique_function_satisfying_equation_l2949_294989


namespace intersection_point_condition_l2949_294941

theorem intersection_point_condition (α β : ℝ) : 
  (∃ x y : ℝ, 
    (x / (Real.sin α + Real.sin β) + y / (Real.sin α + Real.cos β) = 1) ∧
    (x / (Real.cos α + Real.sin β) + y / (Real.cos α + Real.cos β) = 1) ∧
    y = -x) →
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 := by
sorry

end intersection_point_condition_l2949_294941


namespace polynomial_factorization_l2949_294965

theorem polynomial_factorization (a b : ℝ) :
  -3 * a^3 * b + 6 * a^2 * b^2 - 3 * a * b^3 = -3 * a * b * (a - b)^2 := by
  sorry

end polynomial_factorization_l2949_294965


namespace mike_taller_than_mark_l2949_294972

/-- Converts feet and inches to total inches -/
def heightToInches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- The height difference between two people in inches -/
def heightDifference (height1 : ℕ) (height2 : ℕ) : ℕ :=
  max height1 height2 - min height1 height2

theorem mike_taller_than_mark : 
  let markHeight := heightToInches 5 3
  let mikeHeight := heightToInches 6 1
  heightDifference markHeight mikeHeight = 10 := by
sorry

end mike_taller_than_mark_l2949_294972
