import Mathlib

namespace NUMINAMATH_CALUDE_biased_coin_probability_l200_20029

theorem biased_coin_probability (h : ℝ) : 
  h > 0 ∧ h < 1 ∧                                           -- h is a valid probability
  (Nat.choose 6 2 * h^2 * (1-h)^4 = Nat.choose 6 3 * h^3 * (1-h)^3) ∧  -- P(2 heads) = P(3 heads)
  (Nat.choose 6 2 * h^2 * (1-h)^4 ≠ 0) →                    -- P(2 heads) ≠ 0
  Nat.choose 6 4 * h^4 * (1-h)^2 = 240 / 1453 :=             -- P(4 heads) = 240/1453
by sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l200_20029


namespace NUMINAMATH_CALUDE_hair_cut_length_l200_20039

/-- Given Isabella's original and current hair lengths, prove the length of hair cut off. -/
theorem hair_cut_length (original_length current_length cut_length : ℕ) : 
  original_length = 18 → current_length = 9 → cut_length = original_length - current_length :=
by sorry

end NUMINAMATH_CALUDE_hair_cut_length_l200_20039


namespace NUMINAMATH_CALUDE_age_difference_l200_20088

/-- The difference in ages between two people given a ratio and one person's age -/
theorem age_difference (sachin_age rahul_age : ℝ) : 
  sachin_age = 24.5 → 
  sachin_age / rahul_age = 7 / 9 → 
  rahul_age - sachin_age = 7 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l200_20088


namespace NUMINAMATH_CALUDE_range_of_a_l200_20013

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a^2 * x - 2 ≤ 0) → -2 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l200_20013


namespace NUMINAMATH_CALUDE_alligators_in_pond_l200_20072

/-- The number of snakes in the pond -/
def num_snakes : ℕ := 18

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := 56

/-- The number of eyes each snake has -/
def snake_eyes : ℕ := 2

/-- The number of eyes each alligator has -/
def alligator_eyes : ℕ := 2

/-- The number of alligators in the pond -/
def num_alligators : ℕ := 10

theorem alligators_in_pond :
  num_snakes * snake_eyes + num_alligators * alligator_eyes = total_eyes :=
by sorry

end NUMINAMATH_CALUDE_alligators_in_pond_l200_20072


namespace NUMINAMATH_CALUDE_vector_magnitude_l200_20061

def a : ℝ × ℝ := (-2, -1)

theorem vector_magnitude (b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = 10) 
  (h2 : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 5) : 
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l200_20061


namespace NUMINAMATH_CALUDE_probability_not_green_l200_20033

def total_balls : ℕ := 6 + 3 + 4 + 5
def non_green_balls : ℕ := 6 + 3 + 4

theorem probability_not_green (red_balls : ℕ) (yellow_balls : ℕ) (black_balls : ℕ) (green_balls : ℕ)
  (h_red : red_balls = 6)
  (h_yellow : yellow_balls = 3)
  (h_black : black_balls = 4)
  (h_green : green_balls = 5) :
  (red_balls + yellow_balls + black_balls : ℚ) / (red_balls + yellow_balls + black_balls + green_balls) = 13 / 18 :=
by sorry

end NUMINAMATH_CALUDE_probability_not_green_l200_20033


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_six_l200_20010

theorem at_least_one_not_less_than_six (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬(a + 4/b < 6 ∧ b + 9/c < 6 ∧ c + 16/a < 6) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_six_l200_20010


namespace NUMINAMATH_CALUDE_litter_patrol_pickup_l200_20032

/-- The number of glass bottles picked up by the Litter Patrol -/
def glass_bottles : ℕ := 10

/-- The number of aluminum cans picked up by the Litter Patrol -/
def aluminum_cans : ℕ := 8

/-- The total number of pieces of litter picked up by the Litter Patrol -/
def total_litter : ℕ := glass_bottles + aluminum_cans

theorem litter_patrol_pickup :
  total_litter = 18 := by sorry

end NUMINAMATH_CALUDE_litter_patrol_pickup_l200_20032


namespace NUMINAMATH_CALUDE_log_x_16_eq_0_8_implies_x_eq_32_l200_20092

-- Define the logarithm function for our specific base
noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_x_16_eq_0_8_implies_x_eq_32 :
  ∀ x : ℝ, x > 0 → log_base x 16 = 0.8 → x = 32 := by
  sorry

end NUMINAMATH_CALUDE_log_x_16_eq_0_8_implies_x_eq_32_l200_20092


namespace NUMINAMATH_CALUDE_parallelogram_sides_l200_20000

theorem parallelogram_sides (a b : ℝ) (h1 : a = 3 * b) (h2 : 2 * a + 2 * b = 24) :
  (a = 9 ∧ b = 3) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_sides_l200_20000


namespace NUMINAMATH_CALUDE_problem_solution_l200_20091

-- Define the solution set
def SolutionSet (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

-- Define the inequality
def Inequality (x m n : ℝ) : Prop := |x - m| ≤ n

theorem problem_solution :
  -- Conditions
  (∀ x, Inequality x m n ↔ SolutionSet x) →
  -- Part 1: Prove m = 2 and n = 2
  (m = 2 ∧ n = 2) ∧
  -- Part 2: Prove minimum value of a + b
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = m/a + n/b → a + b ≥ 2 * Real.sqrt 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = m/a + n/b ∧ a + b = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l200_20091


namespace NUMINAMATH_CALUDE_polynomial_constant_term_product_l200_20077

variable (p q r : ℝ[X])

theorem polynomial_constant_term_product 
  (h1 : r = p * q)
  (h2 : p.coeff 0 = 6)
  (h3 : r.coeff 0 = -18) :
  q.eval 0 = -3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_constant_term_product_l200_20077


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l200_20058

theorem subtraction_of_decimals : 2.5 - 0.32 = 2.18 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l200_20058


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l200_20095

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^4 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^4 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l200_20095


namespace NUMINAMATH_CALUDE_remainder_3_1000_mod_7_l200_20024

theorem remainder_3_1000_mod_7 : 3^1000 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_1000_mod_7_l200_20024


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_20122012_l200_20089

theorem no_integer_solutions_for_20122012 :
  ¬∃ (a b c : ℤ), a^2 + b^2 + c^2 = 20122012 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_20122012_l200_20089


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l200_20025

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- State the theorem
theorem quadratic_function_minimum (a b : ℝ) :
  (∀ x : ℝ, f a b x ≥ f a b (-1)) ∧ (f a b (-1) = 0) →
  ∀ x : ℝ, f a b x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l200_20025


namespace NUMINAMATH_CALUDE_largest_perimeter_is_164_l200_20007

/-- Represents a rectangle with integer side lengths -/
structure IntRectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of an IntRectangle -/
def perimeter (r : IntRectangle) : ℕ :=
  2 * (r.length + r.width)

/-- Calculates the area of an IntRectangle -/
def area (r : IntRectangle) : ℕ :=
  r.length * r.width

/-- Checks if a rectangle satisfies the given condition -/
def satisfiesCondition (r : IntRectangle) : Prop :=
  4 * perimeter r = area r - 1

/-- Theorem stating that the largest possible perimeter of a rectangle satisfying the condition is 164 -/
theorem largest_perimeter_is_164 :
  ∀ r : IntRectangle, satisfiesCondition r → perimeter r ≤ 164 :=
by sorry

end NUMINAMATH_CALUDE_largest_perimeter_is_164_l200_20007


namespace NUMINAMATH_CALUDE_students_without_A_l200_20038

theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (both : ℕ) 
  (h_total : total = 50)
  (h_history : history = 12)
  (h_math : math = 25)
  (h_both : both = 6) : 
  total - (history + math - both) = 19 := by
  sorry

end NUMINAMATH_CALUDE_students_without_A_l200_20038


namespace NUMINAMATH_CALUDE_regression_line_equation_l200_20047

/-- Given a regression line with slope 1.2 passing through the point (4, 5),
    prove that its equation is ŷ = 1.2x + 0.2 -/
theorem regression_line_equation 
  (slope : ℝ) 
  (center_x : ℝ) 
  (center_y : ℝ) 
  (h1 : slope = 1.2) 
  (h2 : center_x = 4) 
  (h3 : center_y = 5) : 
  ∃ (a : ℝ), ∀ (x y : ℝ), y = slope * x + a ↔ (x = center_x ∧ y = center_y) ∨ y = 1.2 * x + 0.2 :=
sorry

end NUMINAMATH_CALUDE_regression_line_equation_l200_20047


namespace NUMINAMATH_CALUDE_therapy_pricing_theorem_l200_20026

/-- Represents the pricing structure of a psychologist's therapy sessions. -/
structure TherapyPricing where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  firstHourPremium : ℕ
  fiveHourTotal : ℕ

/-- Calculates the total charge for a given number of therapy hours. -/
def totalCharge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  if hours = 0 then 0
  else pricing.firstHourCharge + (hours - 1) * pricing.additionalHourCharge

/-- Theorem stating the conditions and the result to be proved. -/
theorem therapy_pricing_theorem (pricing : TherapyPricing) 
  (h1 : pricing.firstHourCharge = pricing.additionalHourCharge + pricing.firstHourPremium)
  (h2 : pricing.firstHourPremium = 35)
  (h3 : pricing.fiveHourTotal = 350)
  (h4 : totalCharge pricing 5 = pricing.fiveHourTotal) :
  totalCharge pricing 2 = 161 := by
  sorry


end NUMINAMATH_CALUDE_therapy_pricing_theorem_l200_20026


namespace NUMINAMATH_CALUDE_range_of_a_l200_20063

theorem range_of_a (x a : ℝ) :
  (∀ x, (-4 < x - a ∧ x - a < 4) ↔ (1 < x ∧ x < 2)) →
  -2 ≤ a ∧ a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l200_20063


namespace NUMINAMATH_CALUDE_rectangle_width_l200_20078

/-- Given a rectangle where the length is 3 times the width and the area is 108 square inches,
    prove that the width is 6 inches. -/
theorem rectangle_width (w : ℝ) (h1 : w > 0) (h2 : 3 * w * w = 108) : w = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l200_20078


namespace NUMINAMATH_CALUDE_ball_probabilities_solution_l200_20082

/-- Represents the color of a ball -/
inductive Color
  | Red
  | Black
  | Yellow
  | Green

/-- Represents the probabilities of drawing balls of different colors -/
structure BallProbabilities where
  red : ℚ
  black : ℚ
  yellow : ℚ
  green : ℚ

/-- The conditions of the problem -/
def problem_conditions (p : BallProbabilities) : Prop :=
  p.red + p.black + p.yellow + p.green = 1 ∧
  p.red = 1/3 ∧
  p.black + p.yellow = 5/12 ∧
  p.yellow + p.green = 5/12

/-- The theorem stating the solution -/
theorem ball_probabilities_solution :
  ∃ (p : BallProbabilities), problem_conditions p ∧ 
    p.black = 1/4 ∧ p.yellow = 1/6 ∧ p.green = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_solution_l200_20082


namespace NUMINAMATH_CALUDE_log_equality_implies_relation_l200_20093

theorem log_equality_implies_relation (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  Real.log p + Real.log q + Real.log r = Real.log (p * q * r + p + q) → p = -q := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_relation_l200_20093


namespace NUMINAMATH_CALUDE_complex_equation_sum_l200_20094

theorem complex_equation_sum (x y : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : x - 3 * i = (8 * x - y) * i) : x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l200_20094


namespace NUMINAMATH_CALUDE_share_calculation_l200_20035

/-- The amount y gets for each rupee x gets -/
def a : ℝ := 0.45

/-- The share of y in rupees -/
def y : ℝ := 63

/-- The total amount in rupees -/
def total : ℝ := 273

theorem share_calculation (x : ℝ) :
  x > 0 →
  x + a * x + 0.5 * x = total ∧
  a * x = y →
  a = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_share_calculation_l200_20035


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l200_20008

theorem lcm_hcf_problem (A B : ℕ+) (h1 : B = 671) (h2 : Nat.lcm A B = 2310) (h3 : Nat.gcd A B = 61) : A = 210 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l200_20008


namespace NUMINAMATH_CALUDE_simplify_square_roots_l200_20048

theorem simplify_square_roots : Real.sqrt 81 - Real.sqrt 144 = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l200_20048


namespace NUMINAMATH_CALUDE_g_2009_divisors_l200_20097

/-- g(n) returns the smallest positive integer k such that 1/k has exactly n+1 digits after the decimal point -/
def g (n : ℕ+) : ℕ+ := sorry

/-- The number of positive integer divisors of g(2009) -/
def num_divisors_g_2009 : ℕ := sorry

theorem g_2009_divisors : num_divisors_g_2009 = 2011 := by sorry

end NUMINAMATH_CALUDE_g_2009_divisors_l200_20097


namespace NUMINAMATH_CALUDE_david_weighted_average_l200_20044

def david_marks : List ℕ := [76, 65, 82, 67, 85, 93, 71]

def english_weight : ℕ := 2
def math_weight : ℕ := 3
def science_weight : ℕ := 1

def weighted_sum : ℕ := 
  david_marks[0] * english_weight + 
  david_marks[1] * math_weight + 
  david_marks[2] * science_weight + 
  david_marks[3] * science_weight + 
  david_marks[4] * science_weight

def total_weight : ℕ := english_weight + math_weight + 3 * science_weight

theorem david_weighted_average :
  (weighted_sum : ℚ) / total_weight = 581 / 8 := by sorry

end NUMINAMATH_CALUDE_david_weighted_average_l200_20044


namespace NUMINAMATH_CALUDE_pizza_fraction_eaten_l200_20098

/-- Calculates the fraction of pizza eaten given the calorie information and consumption --/
theorem pizza_fraction_eaten 
  (lettuce_cal : ℕ) 
  (dressing_cal : ℕ) 
  (crust_cal : ℕ) 
  (cheese_cal : ℕ) 
  (total_consumed : ℕ) 
  (h1 : lettuce_cal = 50)
  (h2 : dressing_cal = 210)
  (h3 : crust_cal = 600)
  (h4 : cheese_cal = 400)
  (h5 : total_consumed = 330) :
  (total_consumed - (lettuce_cal + 2 * lettuce_cal + dressing_cal) / 4) / 
  (crust_cal + crust_cal / 3 + cheese_cal) = 1 / 5 := by
  sorry

#check pizza_fraction_eaten

end NUMINAMATH_CALUDE_pizza_fraction_eaten_l200_20098


namespace NUMINAMATH_CALUDE_circle_on_parabola_circle_standard_equation_l200_20084

def parabola (x y : ℝ) : Prop := y^2 = 16 * x

def circle_equation (h k r x y : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem circle_on_parabola (h k : ℝ) :
  parabola h k →
  first_quadrant h k →
  circle_equation h k 6 0 0 →
  circle_equation h k 6 4 0 →
  h = 2 ∧ k = 4 * Real.sqrt 2 :=
sorry

theorem circle_standard_equation (h k : ℝ) :
  h = 2 →
  k = 4 * Real.sqrt 2 →
  ∀ x y : ℝ, circle_equation h k 6 x y ↔ circle_equation 2 (4 * Real.sqrt 2) 6 x y :=
sorry

end NUMINAMATH_CALUDE_circle_on_parabola_circle_standard_equation_l200_20084


namespace NUMINAMATH_CALUDE_jebbs_take_home_pay_l200_20042

/-- Calculates the take-home pay given a gross salary and various tax rates and deductions. -/
def calculateTakeHomePay (grossSalary : ℚ) : ℚ :=
  let federalTaxRate1 := 0.10
  let federalTaxRate2 := 0.15
  let federalTaxRate3 := 0.25
  let federalTaxThreshold1 := 2500
  let federalTaxThreshold2 := 5000
  let stateTaxRate1 := 0.05
  let stateTaxRate2 := 0.07
  let stateTaxThreshold := 3000
  let socialSecurityTaxRate := 0.062
  let socialSecurityTaxCap := 4800
  let medicareTaxRate := 0.0145
  let healthInsurance := 300
  let retirementContributionRate := 0.07

  let federalTax := 
    federalTaxRate1 * federalTaxThreshold1 +
    federalTaxRate2 * (federalTaxThreshold2 - federalTaxThreshold1) +
    federalTaxRate3 * (grossSalary - federalTaxThreshold2)

  let stateTax := 
    stateTaxRate1 * stateTaxThreshold +
    stateTaxRate2 * (grossSalary - stateTaxThreshold)

  let socialSecurityTax := socialSecurityTaxRate * (min grossSalary socialSecurityTaxCap)

  let medicareTax := medicareTaxRate * grossSalary

  let retirementContribution := retirementContributionRate * grossSalary

  let totalDeductions := 
    federalTax + stateTax + socialSecurityTax + medicareTax + healthInsurance + retirementContribution

  grossSalary - totalDeductions

/-- Theorem stating that Jebb's take-home pay is $3,958.15 given his gross salary and deductions. -/
theorem jebbs_take_home_pay :
  calculateTakeHomePay 6500 = 3958.15 := by
  sorry


end NUMINAMATH_CALUDE_jebbs_take_home_pay_l200_20042


namespace NUMINAMATH_CALUDE_max_points_in_tournament_l200_20060

/-- Represents a tournament with the given conditions --/
structure Tournament :=
  (num_teams : Nat)
  (points_for_win : Nat)
  (points_for_draw : Nat)
  (points_for_loss : Nat)

/-- Calculates the total number of games in the tournament --/
def total_games (t : Tournament) : Nat :=
  (t.num_teams * (t.num_teams - 1)) / 2 * 2

/-- Represents the maximum points achievable by top teams --/
def max_points_for_top_teams (t : Tournament) : Nat :=
  let games_with_other_top_teams := 4
  let games_with_lower_teams := 6
  games_with_other_top_teams * t.points_for_win / 2 +
  games_with_lower_teams * t.points_for_win

/-- The main theorem to be proved --/
theorem max_points_in_tournament (t : Tournament) 
  (h1 : t.num_teams = 6)
  (h2 : t.points_for_win = 3)
  (h3 : t.points_for_draw = 1)
  (h4 : t.points_for_loss = 0) :
  max_points_for_top_teams t = 24 := by
  sorry

#eval max_points_for_top_teams ⟨6, 3, 1, 0⟩

end NUMINAMATH_CALUDE_max_points_in_tournament_l200_20060


namespace NUMINAMATH_CALUDE_inequality_theorem_l200_20009

theorem inequality_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z * (x + y + z + Real.sqrt (x^2 + y^2 + z^2))) / 
  ((x^2 + y^2 + z^2) * (y*z + z*x + x*y)) ≤ (3 + Real.sqrt 3) / 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l200_20009


namespace NUMINAMATH_CALUDE_base_8_units_digit_l200_20018

theorem base_8_units_digit : (((324 + 73) * 27) % 8 = 7) := by
  sorry

end NUMINAMATH_CALUDE_base_8_units_digit_l200_20018


namespace NUMINAMATH_CALUDE_circle_center_l200_20023

theorem circle_center (x y : ℝ) : 
  4 * x^2 - 16 * x + 4 * y^2 + 8 * y - 12 = 0 → 
  ∃ (h k : ℝ), h = 2 ∧ k = -1 ∧ (x - h)^2 + (y - k)^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l200_20023


namespace NUMINAMATH_CALUDE_salary_calculation_l200_20099

/-- Given a series of salary changes and a final salary, calculate the original salary --/
theorem salary_calculation (S : ℝ) : 
  S * 1.12 * 0.93 * 1.15 * 0.90 = 5204.21 → S = 5504.00 := by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l200_20099


namespace NUMINAMATH_CALUDE_problem_statement_l200_20037

theorem problem_statement (a b x y : ℝ) 
  (eq1 : a * x + b * y = 3)
  (eq2 : a * x^2 + b * y^2 = 7)
  (eq3 : a * x^3 + b * y^3 = 16)
  (eq4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l200_20037


namespace NUMINAMATH_CALUDE_city_population_problem_l200_20075

theorem city_population_problem (p : ℝ) : 
  (0.84 * (p + 2500) + 500 = p + 2680) → p = 500 := by
  sorry

end NUMINAMATH_CALUDE_city_population_problem_l200_20075


namespace NUMINAMATH_CALUDE_lawn_mowing_problem_lawn_mowing_solution_l200_20053

theorem lawn_mowing_problem (original_people : ℕ) (original_time : ℝ) 
  (new_time : ℝ) (efficiency : ℝ) (new_people : ℕ) : Prop :=
  original_people = 8 →
  original_time = 3 →
  new_time = 2 →
  efficiency = 0.9 →
  (original_people : ℝ) * original_time = (new_people : ℝ) * new_time * efficiency →
  new_people = 14

-- The proof of the theorem
theorem lawn_mowing_solution : lawn_mowing_problem 8 3 2 0.9 14 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_problem_lawn_mowing_solution_l200_20053


namespace NUMINAMATH_CALUDE_seeds_in_first_plot_l200_20046

/-- The number of seeds planted in the first plot -/
def seeds_plot1 : ℕ := sorry

/-- The number of seeds planted in the second plot -/
def seeds_plot2 : ℕ := 200

/-- The percentage of seeds that germinated in the first plot -/
def germination_rate_plot1 : ℚ := 30 / 100

/-- The percentage of seeds that germinated in the second plot -/
def germination_rate_plot2 : ℚ := 35 / 100

/-- The percentage of total seeds that germinated -/
def total_germination_rate : ℚ := 32 / 100

/-- Theorem stating that the number of seeds planted in the first plot is 300 -/
theorem seeds_in_first_plot : 
  (germination_rate_plot1 * seeds_plot1 + germination_rate_plot2 * seeds_plot2 : ℚ) = 
  total_germination_rate * (seeds_plot1 + seeds_plot2) ∧ 
  seeds_plot1 = 300 := by sorry

end NUMINAMATH_CALUDE_seeds_in_first_plot_l200_20046


namespace NUMINAMATH_CALUDE_quadratic_polynomial_value_l200_20031

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℚ) : ℚ → ℚ := fun x ↦ a * x^2 + b * x + c

/-- The condition that [q(x)]^3 - x is divisible by (x - 2)(x + 2)(x - 5) -/
def DivisibilityCondition (q : ℚ → ℚ) : Prop :=
  ∀ x, x = 2 ∨ x = -2 ∨ x = 5 → q x ^ 3 = x

theorem quadratic_polynomial_value (a b c : ℚ) :
  (∃ q : ℚ → ℚ, q = QuadraticPolynomial a b c ∧ DivisibilityCondition q) →
  QuadraticPolynomial a b c 10 = -58/7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_value_l200_20031


namespace NUMINAMATH_CALUDE_function_value_at_five_l200_20056

def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

theorem function_value_at_five (a b : ℝ) (h1 : f a b 1 = 3) (h2 : f a b 8 = 10) : f a b 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_five_l200_20056


namespace NUMINAMATH_CALUDE_power_of_two_preserves_order_l200_20049

theorem power_of_two_preserves_order (a b : ℝ) : a > b → (2 : ℝ) ^ a > (2 : ℝ) ^ b := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_preserves_order_l200_20049


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l200_20070

theorem ratio_equation_solution (a b : ℚ) 
  (h1 : b / a = 4)
  (h2 : b = 20 - 7 * a) : 
  a = 20 / 11 := by
sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l200_20070


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l200_20043

/-- The surface area of a sphere that circumscribes a rectangular solid -/
theorem sphere_surface_area_from_rectangular_solid 
  (length width height : ℝ) 
  (h_length : length = 4) 
  (h_width : width = 3) 
  (h_height : height = 2) : 
  ∃ (radius : ℝ), 4 * Real.pi * radius^2 = 29 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l200_20043


namespace NUMINAMATH_CALUDE_savings_percentage_l200_20066

-- Define the original prices and discount rates
def coat_price : ℝ := 120
def hat_price : ℝ := 30
def gloves_price : ℝ := 50

def coat_discount : ℝ := 0.20
def hat_discount : ℝ := 0.40
def gloves_discount : ℝ := 0.30

-- Define the total original cost
def total_original_cost : ℝ := coat_price + hat_price + gloves_price

-- Define the savings for each item
def coat_savings : ℝ := coat_price * coat_discount
def hat_savings : ℝ := hat_price * hat_discount
def gloves_savings : ℝ := gloves_price * gloves_discount

-- Define the total savings
def total_savings : ℝ := coat_savings + hat_savings + gloves_savings

-- Theorem to prove
theorem savings_percentage :
  (total_savings / total_original_cost) * 100 = 25.5 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_l200_20066


namespace NUMINAMATH_CALUDE_root_sum_absolute_value_l200_20090

theorem root_sum_absolute_value (m : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + m = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 106 := by
sorry

end NUMINAMATH_CALUDE_root_sum_absolute_value_l200_20090


namespace NUMINAMATH_CALUDE_ice_cream_cost_l200_20041

theorem ice_cream_cost (people : ℕ) (meal_cost : ℚ) (total_amount : ℚ) 
  (h1 : people = 3)
  (h2 : meal_cost = 10)
  (h3 : total_amount = 45)
  (h4 : total_amount ≥ people * meal_cost) :
  (total_amount - people * meal_cost) / people = 5 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l200_20041


namespace NUMINAMATH_CALUDE_selection_with_at_least_one_girl_l200_20083

def total_students : ℕ := 6
def boys : ℕ := 4
def girls : ℕ := 2
def students_to_select : ℕ := 4

theorem selection_with_at_least_one_girl :
  (Nat.choose total_students students_to_select) - (Nat.choose boys students_to_select) = 14 :=
by sorry

end NUMINAMATH_CALUDE_selection_with_at_least_one_girl_l200_20083


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l200_20027

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest two-digit positive integer that satisfies the given condition -/
theorem smallest_fourth_number : 
  ∃ x : ℕ, 
    x ≥ 10 ∧ x < 100 ∧ 
    (∀ y : ℕ, y ≥ 10 ∧ y < 100 → 
      sumOfDigits 28 + sumOfDigits 46 + sumOfDigits 59 + sumOfDigits y = (28 + 46 + 59 + y) / 4 →
      x ≤ y) ∧
    sumOfDigits 28 + sumOfDigits 46 + sumOfDigits 59 + sumOfDigits x = (28 + 46 + 59 + x) / 4 ∧
    x = 11 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fourth_number_l200_20027


namespace NUMINAMATH_CALUDE_tangent_circles_m_values_l200_20074

/-- Definition of circle C1 -/
def C1 (m x y : ℝ) : Prop := (x - m)^2 + (y + 2)^2 = 9

/-- Definition of circle C2 -/
def C2 (m x y : ℝ) : Prop := (x + 1)^2 + (y - m)^2 = 4

/-- C1 is tangent to C2 from the inside -/
def is_tangent_inside (m : ℝ) : Prop :=
  ∃ x y : ℝ, C1 m x y ∧ C2 m x y ∧
  ∀ x' y' : ℝ, C1 m x' y' → C2 m x' y' → (x = x' ∧ y = y')

/-- The theorem to be proved -/
theorem tangent_circles_m_values :
  ∀ m : ℝ, is_tangent_inside m ↔ (m = -2 ∨ m = -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_m_values_l200_20074


namespace NUMINAMATH_CALUDE_jellybean_difference_l200_20028

/-- Proves that the difference between green and orange jellybeans is 1 -/
theorem jellybean_difference (total : ℕ) (black : ℕ) (green : ℕ) (orange : ℕ) : 
  total = 27 → 
  black = 8 → 
  green = black + 2 → 
  total = black + green + orange →
  green - orange = 1 := by
sorry

end NUMINAMATH_CALUDE_jellybean_difference_l200_20028


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l200_20065

theorem arctan_equation_solution (y : ℝ) :
  2 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4 →
  y = -121/60 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l200_20065


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l200_20003

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l200_20003


namespace NUMINAMATH_CALUDE_inequality_direction_change_l200_20050

theorem inequality_direction_change (a b x : ℝ) (h : x < 0) :
  (a < b) ↔ (a * x > b * x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_direction_change_l200_20050


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_one_five_l200_20051

theorem sqrt_equality_implies_one_five (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) :
  (Real.sqrt (1 + Real.sqrt (45 + 20 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b) →
  (a = 1 ∧ b = 5) := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_one_five_l200_20051


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l200_20086

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℝ := 3 * n^2 + 4 * n + 5

/-- The r-th term of the arithmetic progression -/
def a (r : ℕ) : ℝ := 6 * r + 1

theorem arithmetic_progression_rth_term (r : ℕ) : 
  a r = S r - S (r - 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l200_20086


namespace NUMINAMATH_CALUDE_steve_initial_berries_l200_20081

/-- Proves that Steve started with 21 berries given the conditions of the problem -/
theorem steve_initial_berries :
  ∀ (stacy_initial steve_initial : ℕ),
    stacy_initial = 32 →
    steve_initial + 4 = stacy_initial - 7 →
    steve_initial = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_steve_initial_berries_l200_20081


namespace NUMINAMATH_CALUDE_P_less_than_Q_l200_20030

theorem P_less_than_Q (a : ℝ) (ha : a ≥ 0) :
  Real.sqrt (a + 3) + Real.sqrt (a + 5) < Real.sqrt (a + 1) + Real.sqrt (a + 7) := by
  sorry

end NUMINAMATH_CALUDE_P_less_than_Q_l200_20030


namespace NUMINAMATH_CALUDE_divisible_by_240_l200_20015

-- Define a prime number p that is greater than or equal to 7
def p : ℕ := sorry

-- Axiom: p is prime
axiom p_prime : Nat.Prime p

-- Axiom: p is greater than or equal to 7
axiom p_ge_7 : p ≥ 7

-- Theorem to prove
theorem divisible_by_240 : 240 ∣ p^4 - 1 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_240_l200_20015


namespace NUMINAMATH_CALUDE_expand_product_l200_20067

theorem expand_product (x : ℝ) : (x + 3) * (x - 8) = x^2 - 5*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l200_20067


namespace NUMINAMATH_CALUDE_min_route_length_5x5_city_l200_20017

/-- Represents a square grid city -/
structure SquareGridCity where
  size : Nat

/-- Represents a route in the city -/
structure CityRoute where
  length : Nat
  covers_all_streets : Bool
  returns_to_start : Bool

/-- The minimum length of a route that covers all streets and returns to the starting point -/
def min_route_length (city : SquareGridCity) : Nat :=
  sorry

theorem min_route_length_5x5_city :
  ∀ (city : SquareGridCity) (route : CityRoute),
    city.size = 5 →
    route.covers_all_streets = true →
    route.returns_to_start = true →
    route.length ≥ min_route_length city →
    min_route_length city = 68 :=
by sorry

end NUMINAMATH_CALUDE_min_route_length_5x5_city_l200_20017


namespace NUMINAMATH_CALUDE_carries_work_hours_l200_20055

/-- Proves that Carrie worked 2 hours each day to earn a profit of $122 -/
theorem carries_work_hours 
  (days : ℕ) 
  (hourly_rate : ℚ) 
  (supply_cost : ℚ) 
  (profit : ℚ) 
  (h : ℚ)
  (h_days : days = 4)
  (h_rate : hourly_rate = 22)
  (h_cost : supply_cost = 54)
  (h_profit : profit = 122)
  (h_equation : profit = days * hourly_rate * h - supply_cost) : 
  h = 2 := by
  sorry

end NUMINAMATH_CALUDE_carries_work_hours_l200_20055


namespace NUMINAMATH_CALUDE_right_triangle_roots_l200_20040

/-- Given complex numbers a and b, and complex roots z₁ and z₂ of z² + az + b = 0
    such that 0, z₁, and z₂ form a right triangle with z₂ opposite the right angle,
    prove that a²/b = 2 -/
theorem right_triangle_roots (a b z₁ z₂ : ℂ) 
    (h_root : z₁^2 + a*z₁ + b = 0 ∧ z₂^2 + a*z₂ + b = 0)
    (h_right_triangle : z₂ = z₁ * Complex.I) : 
    a^2 / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_roots_l200_20040


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l200_20064

/-- 
Given a triangle with two sides of 27 units and 34 units, and the third side having an integral length,
the least possible perimeter is 69 units.
-/
theorem least_perimeter_triangle : 
  ∀ z : ℕ, 
  z > 0 → 
  z + 27 > 34 → 
  34 + 27 > z → 
  27 + z > 34 → 
  ∀ w : ℕ, 
  w > 0 → 
  w + 27 > 34 → 
  34 + 27 > w → 
  27 + w > 34 → 
  w ≥ z → 
  27 + 34 + w ≥ 69 :=
by sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l200_20064


namespace NUMINAMATH_CALUDE_sum_of_roots_l200_20085

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x : ℝ, x^2 - 14*a*x + 15*b = 0 ↔ x = c ∨ x = d) →
  (∀ x : ℝ, x^2 - 14*c*x - 15*d = 0 ↔ x = a ∨ x = b) →
  a + b + c + d = 3150 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l200_20085


namespace NUMINAMATH_CALUDE_inverse_composition_problem_l200_20052

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- State the theorem
theorem inverse_composition_problem
  (h : ∀ x, f_inv (g x) = 3 * x + 5)
  (h_inv_f : ∀ x, f_inv (f x) = x)
  (h_inv_g : ∀ x, g_inv (g x) = x)
  (h_f_inv : ∀ x, f (f_inv x) = x)
  (h_g_inv : ∀ x, g (g_inv x) = x) :
  g_inv (f (-8)) = -13/3 :=
sorry

end NUMINAMATH_CALUDE_inverse_composition_problem_l200_20052


namespace NUMINAMATH_CALUDE_expression_simplification_l200_20057

theorem expression_simplification (m : ℝ) (h : m = 2) : 
  (m^2 / (1 - m^2)) * (1 - 1/m) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l200_20057


namespace NUMINAMATH_CALUDE_odd_function_derivative_l200_20076

theorem odd_function_derivative (f : ℝ → ℝ) (x₀ : ℝ) (k : ℝ) :
  (∀ x, f (-x) = -f x) →
  Differentiable ℝ f →
  deriv f (-x₀) = k →
  k ≠ 0 →
  deriv f x₀ = k :=
by sorry

end NUMINAMATH_CALUDE_odd_function_derivative_l200_20076


namespace NUMINAMATH_CALUDE_roses_in_vase_l200_20080

-- Define the initial number of roses and orchids
def initial_roses : ℕ := 5
def initial_orchids : ℕ := 3

-- Define the current number of orchids
def current_orchids : ℕ := 2

-- Define the difference between roses and orchids
def rose_orchid_difference : ℕ := 10

-- Theorem to prove
theorem roses_in_vase :
  ∃ (current_roses : ℕ),
    current_roses = current_orchids + rose_orchid_difference ∧
    current_roses > initial_roses ∧
    current_roses = 12 :=
by sorry

end NUMINAMATH_CALUDE_roses_in_vase_l200_20080


namespace NUMINAMATH_CALUDE_expression_value_l200_20004

theorem expression_value (a b c d : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) : 
  (a + b + c * d) + (a + b) / (c * d) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l200_20004


namespace NUMINAMATH_CALUDE_two_special_numbers_exist_l200_20071

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def has_no_single_digit_prime_factors (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p > 9

theorem two_special_numbers_exist : ∃ x y : ℕ,
  x + y = 173717 ∧
  is_four_digit (x - y) ∧
  has_no_single_digit_prime_factors (x - y) ∧
  (1558 ∣ x ∨ 1558 ∣ y) ∧
  x = 91143 ∧ y = 82574 := by
  sorry

end NUMINAMATH_CALUDE_two_special_numbers_exist_l200_20071


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l200_20087

theorem geometric_series_first_term 
  (r : ℚ) (S : ℚ) (h1 : r = -3/7) (h2 : S = 20) :
  S = a / (1 - r) → a = 200/7 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l200_20087


namespace NUMINAMATH_CALUDE_upper_bound_expression_l200_20045

theorem upper_bound_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  -1/(2*a) - 2/b ≤ -9/2 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 1 ∧ -1/(2*a₀) - 2/b₀ = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_upper_bound_expression_l200_20045


namespace NUMINAMATH_CALUDE_equal_circles_radius_l200_20059

/-- The radius of two equal circles that satisfy the given conditions -/
def radius_of_equal_circles : ℝ := 16

/-- The radius of the third circle that touches the line -/
def radius_of_third_circle : ℝ := 4

/-- Theorem stating that the radius of the two equal circles is 16 -/
theorem equal_circles_radius :
  let r₁ := radius_of_equal_circles
  let r₂ := radius_of_third_circle
  (r₁ : ℝ) > 0 ∧ r₂ > 0 ∧
  r₁^2 + (r₁ - r₂)^2 = (r₁ + r₂)^2 →
  r₁ = 16 := by sorry


end NUMINAMATH_CALUDE_equal_circles_radius_l200_20059


namespace NUMINAMATH_CALUDE_factor_implies_absolute_value_l200_20016

/-- Given a polynomial 3x^4 - mx^2 + nx + p with factors (x-3), (x+1), and (x-2), 
    prove that |3m - 2n| = 25 -/
theorem factor_implies_absolute_value (m n p : ℝ) : 
  (∀ x : ℝ, (x - 3) * (x + 1) * (x - 2) ∣ (3 * x^4 - m * x^2 + n * x + p)) → 
  |3 * m - 2 * n| = 25 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_absolute_value_l200_20016


namespace NUMINAMATH_CALUDE_picture_area_is_6600_l200_20012

/-- Calculates the area of a picture within a rectangular frame. -/
def picture_area (outer_height outer_width short_frame_width long_frame_width : ℕ) : ℕ :=
  (outer_height - 2 * short_frame_width) * (outer_width - 2 * long_frame_width)

/-- Theorem stating that for a frame with given dimensions, the enclosed picture has an area of 6600 cm². -/
theorem picture_area_is_6600 :
  picture_area 100 140 20 15 = 6600 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_is_6600_l200_20012


namespace NUMINAMATH_CALUDE_sin_120_cos_1290_l200_20096

theorem sin_120_cos_1290 : Real.sin (-120 * π / 180) * Real.cos (1290 * π / 180) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_cos_1290_l200_20096


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_for_necessary_not_sufficient_l200_20021

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3)*(2 - x) ≥ 0

-- Theorem 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 ≤ x ∧ x < 3) :=
sorry

-- Theorem 2
theorem range_of_a_for_necessary_not_sufficient :
  ∀ a : ℝ, (∀ x : ℝ, ¬(q x) → ¬(p x a)) ∧ (∃ x : ℝ, ¬(p x a) ∧ q x) ↔ (1 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_for_necessary_not_sufficient_l200_20021


namespace NUMINAMATH_CALUDE_f_at_three_l200_20054

/-- Horner's method representation of the polynomial f(x) = 2x^4 + 3x^3 + 5x - 4 -/
def f (x : ℝ) : ℝ := (((2 * x + 3) * x + 0) * x + 5) * x - 4

/-- Theorem stating that f(3) = 254 -/
theorem f_at_three : f 3 = 254 := by sorry

end NUMINAMATH_CALUDE_f_at_three_l200_20054


namespace NUMINAMATH_CALUDE_distance_to_SFL_is_81_l200_20006

/-- The distance to Super Fun-tastic Land -/
def distance_to_SFL (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proof that the distance to Super Fun-tastic Land is 81 miles -/
theorem distance_to_SFL_is_81 :
  distance_to_SFL 27 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_SFL_is_81_l200_20006


namespace NUMINAMATH_CALUDE_rook_removal_theorem_l200_20068

/-- Represents a chessboard -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Checks if a rook at position (x, y) attacks a square (i, j) -/
def attacks (x y i j : Fin 8) : Bool :=
  x = i ∨ y = j

/-- A configuration of rooks on a chessboard -/
def RookConfiguration := Fin 20 → Fin 8 × Fin 8

/-- Checks if a configuration of rooks attacks the entire board -/
def attacks_all_squares (config : RookConfiguration) : Prop :=
  ∀ i j, ∃ k, attacks (config k).1 (config k).2 i j

/-- Represents a subset of 8 rooks from the original 20 -/
def Subset := Fin 8 → Fin 20

theorem rook_removal_theorem (initial_config : RookConfiguration) 
  (h : attacks_all_squares initial_config) :
  ∃ (subset : Subset), attacks_all_squares (λ i => initial_config (subset i)) :=
sorry

end NUMINAMATH_CALUDE_rook_removal_theorem_l200_20068


namespace NUMINAMATH_CALUDE_line_through_point_l200_20019

/-- Theorem: If the line ax + 3y - 2 = 0 passes through point (1, 0), then a = 2. -/
theorem line_through_point (a : ℝ) : 
  (∀ x y, a * x + 3 * y - 2 = 0 → (x = 1 ∧ y = 0)) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_l200_20019


namespace NUMINAMATH_CALUDE_marbles_distribution_l200_20034

theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) :
  total_marbles = 35 →
  num_boys = 5 →
  marbles_per_boy = total_marbles / num_boys →
  marbles_per_boy = 7 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l200_20034


namespace NUMINAMATH_CALUDE_tangent_ellipse_d_value_l200_20005

/-- An ellipse in the first quadrant tangent to the x-axis and y-axis with foci at (5,9) and (d,9) -/
structure TangentEllipse where
  d : ℝ
  focus1 : ℝ × ℝ := (5, 9)
  focus2 : ℝ × ℝ := (d, 9)
  first_quadrant : d > 5
  tangent_to_axes : True  -- We assume this property without formally defining it

/-- The value of d for the given ellipse is 29.9 -/
theorem tangent_ellipse_d_value (e : TangentEllipse) : e.d = 29.9 := by
  sorry

#check tangent_ellipse_d_value

end NUMINAMATH_CALUDE_tangent_ellipse_d_value_l200_20005


namespace NUMINAMATH_CALUDE_clock_angle_at_6_30_l200_20011

/-- The smaller angle between the hour and minute hands of a clock at 6:30 -/
def clock_angle : ℝ :=
  let hour_hand_rate : ℝ := 0.5  -- degrees per minute
  let minute_hand_rate : ℝ := 6  -- degrees per minute
  let time_passed : ℝ := 30      -- minutes since 6:00
  let hour_hand_position : ℝ := hour_hand_rate * time_passed
  let minute_hand_position : ℝ := minute_hand_rate * time_passed
  minute_hand_position - hour_hand_position

theorem clock_angle_at_6_30 : clock_angle = 15 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_6_30_l200_20011


namespace NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l200_20062

theorem largest_n_for_trig_inequality : 
  (∃ (n : ℕ), n > 0 ∧ (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 2/n)) ∧
  (∀ (m : ℕ), m > 6 → ∃ (x : ℝ), (Real.sin x)^m + (Real.cos x)^m < 2/m) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l200_20062


namespace NUMINAMATH_CALUDE_subtraction_from_percentage_l200_20022

theorem subtraction_from_percentage (x : ℝ) : x = 100 → (0.7 * x - 40 = 30) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_from_percentage_l200_20022


namespace NUMINAMATH_CALUDE_fraction_17_39_415th_digit_l200_20073

def decimal_expansion (n d : ℕ) : List ℕ :=
  sorry

def nth_digit (n d k : ℕ) : ℕ :=
  sorry

theorem fraction_17_39_415th_digit :
  nth_digit 17 39 415 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_17_39_415th_digit_l200_20073


namespace NUMINAMATH_CALUDE_immediate_sale_more_profitable_l200_20001

/-- Proves that selling flowers immediately is more profitable than selling after dehydration --/
theorem immediate_sale_more_profitable (initial_weight : ℝ) (initial_price : ℝ) (price_increase : ℝ) 
  (weight_loss_fraction : ℝ) (hw : initial_weight = 49) (hp : initial_price = 1.25) 
  (hpi : price_increase = 2) (hwl : weight_loss_fraction = 5/7) :
  initial_weight * initial_price > 
  (initial_weight * (1 - weight_loss_fraction)) * (initial_price + price_increase) :=
by sorry

end NUMINAMATH_CALUDE_immediate_sale_more_profitable_l200_20001


namespace NUMINAMATH_CALUDE_larger_integer_value_l200_20079

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  max a b = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l200_20079


namespace NUMINAMATH_CALUDE_matrix_transformation_l200_20069

/-- Given a 2nd-order matrix M satisfying the condition, prove M and the transformed curve equation -/
theorem matrix_transformation (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M * !![1, 2; 3, 4] = !![7, 10; 4, 6]) : 
  (M = !![1, 2; 1, 1]) ∧ 
  (∀ x' y' : ℝ, (∃ x y : ℝ, 3*x^2 + 8*x*y + 6*y^2 = 1 ∧ 
                            x' = x + 2*y ∧ 
                            y' = x + y) ↔ 
                x'^2 + 2*y'^2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_matrix_transformation_l200_20069


namespace NUMINAMATH_CALUDE_periodic_sequence_sum_l200_20002

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is periodic with period T if a_{n+T} = a_n for all n -/
def IsPeriodic (a : Sequence) (T : ℕ) : Prop :=
  ∀ n, a (n + T) = a n

/-- The sum of the first n terms of a sequence -/
def SequenceSum (a : Sequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

/-- Theorem: For a periodic sequence with period T, 
    the sum of m terms can be expressed in terms of T and r -/
theorem periodic_sequence_sum 
  (a : Sequence) (T m q r : ℕ) 
  (h_periodic : IsPeriodic a T) 
  (h_smallest : ∀ k, 0 < k → k < T → ¬IsPeriodic a k)
  (h_pos : 0 < T ∧ 0 < m ∧ 0 < q ∧ 0 < r)
  (h_decomp : m = q * T + r) :
  SequenceSum a m = q * SequenceSum a T + SequenceSum a r := by
  sorry

end NUMINAMATH_CALUDE_periodic_sequence_sum_l200_20002


namespace NUMINAMATH_CALUDE_fraction_simplification_l200_20014

theorem fraction_simplification : (5 * 7) / 10 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l200_20014


namespace NUMINAMATH_CALUDE_arithmetic_geometric_properties_l200_20036

-- Define the arithmetic-geometric sequence
def arithmetic_geometric (a b : ℝ) (u : ℕ → ℝ) : Prop :=
  ∀ n, u (n + 1) = a * u n + b

-- Define another sequence satisfying the same recurrence relation
def same_recurrence (a b : ℝ) (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 1) = a * v n + b

-- Define the sequence w as the difference of u and v
def w (u v : ℕ → ℝ) : ℕ → ℝ :=
  λ n => u n - v n

-- State the theorem
theorem arithmetic_geometric_properties
  (a b : ℝ)
  (u v : ℕ → ℝ)
  (hu : arithmetic_geometric a b u)
  (hv : same_recurrence a b v)
  (ha : a ≠ 1) :
  (∀ n, w u v (n + 1) = a * w u v n) ∧
  (∃ c : ℝ, ∀ n, v n = c ∧ c = b / (1 - a)) ∧
  (∀ n, u n = a^n * (u 0 - b/(1-a)) + b/(1-a)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_properties_l200_20036


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l200_20020

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_complement_equality :
  A ∩ (Set.univ \ B) = {1, 5, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l200_20020
