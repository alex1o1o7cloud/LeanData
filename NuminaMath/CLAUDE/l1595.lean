import Mathlib

namespace NUMINAMATH_CALUDE_sock_selection_problem_l1595_159549

theorem sock_selection_problem :
  let total_socks : ℕ := 7
  let socks_to_choose : ℕ := 4
  let number_of_ways : ℕ := Nat.choose total_socks socks_to_choose
  number_of_ways = 35 := by
sorry

end NUMINAMATH_CALUDE_sock_selection_problem_l1595_159549


namespace NUMINAMATH_CALUDE_estimate_wildlife_population_l1595_159521

/-- Estimate the total number of animals in a wildlife reserve using the mark-recapture method. -/
theorem estimate_wildlife_population
  (initial_catch : ℕ)
  (second_catch : ℕ)
  (marked_in_second : ℕ)
  (h1 : initial_catch = 1200)
  (h2 : second_catch = 1000)
  (h3 : marked_in_second = 100) :
  (initial_catch * second_catch) / marked_in_second = 12000 :=
by sorry

end NUMINAMATH_CALUDE_estimate_wildlife_population_l1595_159521


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1595_159575

theorem expand_and_simplify (y : ℝ) (h : y ≠ 0) :
  (3 / 4) * (8 / y + 6 * y^3) = 6 / y + (9 * y^3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1595_159575


namespace NUMINAMATH_CALUDE_tangent_line_problem_l1595_159584

-- Define the curve
def curve (x a b : ℝ) : ℝ := x^3 + a*x + b

-- Define the tangent line
def tangent_line (x k : ℝ) : ℝ := k*x + 1

-- Define the derivative of the curve
def curve_derivative (x a : ℝ) : ℝ := 3*x^2 + a

theorem tangent_line_problem (a b k : ℝ) : 
  curve 1 a b = 2 →
  tangent_line 1 k = 2 →
  curve_derivative 1 a = k →
  b - a = 5 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_problem_l1595_159584


namespace NUMINAMATH_CALUDE_angela_is_157_cm_tall_l1595_159531

def amy_height : ℕ := 150

def helen_height (amy : ℕ) : ℕ := amy + 3

def angela_height (helen : ℕ) : ℕ := helen + 4

theorem angela_is_157_cm_tall :
  angela_height (helen_height amy_height) = 157 :=
by sorry

end NUMINAMATH_CALUDE_angela_is_157_cm_tall_l1595_159531


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1595_159573

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k = 0 → ∃ y : ℝ, y ≠ x ∧ y^2 - 2*y + k = 0) ↔ k < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1595_159573


namespace NUMINAMATH_CALUDE_johns_order_cost_l1595_159571

/-- The total cost of John's food order for a massive restaurant. -/
def total_cost (beef_amount : ℕ) (beef_price : ℕ) (chicken_amount_multiplier : ℕ) (chicken_price : ℕ) : ℕ :=
  beef_amount * beef_price + (beef_amount * chicken_amount_multiplier) * chicken_price

/-- Proof that John's total food order cost is $14000. -/
theorem johns_order_cost :
  total_cost 1000 8 2 3 = 14000 :=
by sorry

end NUMINAMATH_CALUDE_johns_order_cost_l1595_159571


namespace NUMINAMATH_CALUDE_andrew_runs_two_miles_l1595_159592

/-- Andrew's daily run in miles -/
def andrew_daily_run : ℝ := 2

/-- Peter's daily run in miles -/
def peter_daily_run : ℝ := andrew_daily_run + 3

/-- Total number of days they run -/
def days : ℕ := 5

/-- Total miles run by both Peter and Andrew -/
def total_miles : ℝ := 35

theorem andrew_runs_two_miles :
  andrew_daily_run = 2 ∧
  peter_daily_run = andrew_daily_run + 3 ∧
  days * (andrew_daily_run + peter_daily_run) = total_miles :=
by sorry

end NUMINAMATH_CALUDE_andrew_runs_two_miles_l1595_159592


namespace NUMINAMATH_CALUDE_tangent_range_l1595_159506

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the circle C
def C (k : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y + 2*k - 1 = 0

-- Define the condition for two tangents
def has_two_tangents (P : ℝ × ℝ) (C : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ k, (P.1^2 + P.2^2 + 2*P.1 + P.2 + 2*k - 1 > 0) ∧
       (4 + 1 - 4*(2*k - 1) > 0)

-- Theorem statement
theorem tangent_range :
  has_two_tangents P C → ∃ k, -4 < k ∧ k < 9/8 :=
sorry

end NUMINAMATH_CALUDE_tangent_range_l1595_159506


namespace NUMINAMATH_CALUDE_exactlyOneBlack_exactlyTwoBlack_mutually_exclusive_not_complementary_l1595_159576

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The set of all possible outcomes when drawing two balls -/
def sampleSpace : Set DrawOutcome := sorry

/-- The event of drawing exactly one black ball -/
def exactlyOneBlack : Set DrawOutcome := sorry

/-- The event of drawing exactly two black balls -/
def exactlyTwoBlack : Set DrawOutcome := sorry

/-- Definition of mutually exclusive events -/
def mutuallyExclusive (A B : Set DrawOutcome) : Prop :=
  A ∩ B = ∅

/-- Definition of complementary events -/
def complementary (A B : Set DrawOutcome) : Prop :=
  A ∪ B = sampleSpace ∧ A ∩ B = ∅

/-- Main theorem: exactlyOneBlack and exactlyTwoBlack are mutually exclusive but not complementary -/
theorem exactlyOneBlack_exactlyTwoBlack_mutually_exclusive_not_complementary :
  mutuallyExclusive exactlyOneBlack exactlyTwoBlack ∧
  ¬complementary exactlyOneBlack exactlyTwoBlack := by
  sorry

end NUMINAMATH_CALUDE_exactlyOneBlack_exactlyTwoBlack_mutually_exclusive_not_complementary_l1595_159576


namespace NUMINAMATH_CALUDE_expression_evaluation_l1595_159594

theorem expression_evaluation :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 2
  f 2 = 6 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1595_159594


namespace NUMINAMATH_CALUDE_average_problem_l1595_159557

theorem average_problem (x : ℝ) : (15 + 25 + x + 30) / 4 = 23 → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l1595_159557


namespace NUMINAMATH_CALUDE_triangle_area_l1595_159586

/-- The area of a triangle with vertices at (2,1,0), (3,3,2), and (5,8,1) is √170/2 -/
theorem triangle_area : ℝ := by
  -- Define the vertices of the triangle
  let a : Fin 3 → ℝ := ![2, 1, 0]
  let b : Fin 3 → ℝ := ![3, 3, 2]
  let c : Fin 3 → ℝ := ![5, 8, 1]

  -- Calculate the area using the cross product method
  let area := (1/2 : ℝ) * Real.sqrt ((b 0 - a 0) * (c 1 - a 1) - (b 1 - a 1) * (c 0 - a 0))^2 +
                                    ((b 1 - a 1) * (c 2 - a 2) - (b 2 - a 2) * (c 1 - a 1))^2 +
                                    ((b 2 - a 2) * (c 0 - a 0) - (b 0 - a 0) * (c 2 - a 2))^2

  -- Prove that the calculated area equals √170/2
  have : area = Real.sqrt 170 / 2 := by sorry

  exact area

end NUMINAMATH_CALUDE_triangle_area_l1595_159586


namespace NUMINAMATH_CALUDE_equation_holds_l1595_159587

theorem equation_holds (a b c : ℝ) (h : a^2 + c^2 = 2*b^2) : 
  (a+b)*(a+c) + (c+a)*(c+b) = 2*(b+a)*(b+c) := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l1595_159587


namespace NUMINAMATH_CALUDE_chips_juice_weight_difference_l1595_159552

/-- Given that 2 bags of chips weigh 800 g and 5 bags of chips and 4 bottles of juice
    together weigh 2200 g, prove that a bag of chips is 350 g heavier than a bottle of juice. -/
theorem chips_juice_weight_difference :
  (∀ (chips_weight bottle_weight : ℕ),
    2 * chips_weight = 800 →
    5 * chips_weight + 4 * bottle_weight = 2200 →
    chips_weight - bottle_weight = 350) :=
by sorry

end NUMINAMATH_CALUDE_chips_juice_weight_difference_l1595_159552


namespace NUMINAMATH_CALUDE_high_school_nine_games_l1595_159502

/-- The number of teams in the league -/
def num_teams : ℕ := 9

/-- The number of games each team plays against non-league opponents -/
def non_league_games : ℕ := 6

/-- The total number of games played in a season -/
def total_games : ℕ := 126

/-- Theorem stating the total number of games in a season -/
theorem high_school_nine_games :
  (num_teams * (num_teams - 1)) + (num_teams * non_league_games) = total_games :=
sorry

end NUMINAMATH_CALUDE_high_school_nine_games_l1595_159502


namespace NUMINAMATH_CALUDE_complex_number_opposite_parts_l1595_159574

theorem complex_number_opposite_parts (a : ℝ) : 
  (∃ z : ℂ, z = (2 + a * Complex.I) * Complex.I ∧ 
   z.re = -z.im) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_opposite_parts_l1595_159574


namespace NUMINAMATH_CALUDE_ball_sampling_theorem_l1595_159529

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Black

/-- Represents the bag with balls -/
structure Bag :=
  (white : ℕ)
  (black : ℕ)

/-- Represents the sampling method -/
inductive SamplingMethod
  | WithReplacement
  | WithoutReplacement

/-- The probability of drawing two balls of different colors with replacement -/
def prob_diff_colors (bag : Bag) (method : SamplingMethod) : ℚ :=
  sorry

/-- The expectation of the number of white balls drawn without replacement -/
def expectation_white (bag : Bag) : ℚ :=
  sorry

/-- The variance of the number of white balls drawn without replacement -/
def variance_white (bag : Bag) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem ball_sampling_theorem (bag : Bag) :
  bag.white = 2 ∧ bag.black = 3 →
  prob_diff_colors bag SamplingMethod.WithReplacement = 12/25 ∧
  expectation_white bag = 4/5 ∧
  variance_white bag = 9/25 :=
sorry

end NUMINAMATH_CALUDE_ball_sampling_theorem_l1595_159529


namespace NUMINAMATH_CALUDE_prime_squared_plus_41_composite_l1595_159530

theorem prime_squared_plus_41_composite (p : ℕ) (hp : Prime p) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ p^2 + 41 = a * b :=
sorry

end NUMINAMATH_CALUDE_prime_squared_plus_41_composite_l1595_159530


namespace NUMINAMATH_CALUDE_crayons_difference_l1595_159505

/-- Given the initial number of crayons, the number of crayons given away, and the number of crayons lost,
    prove that the difference between the number of crayons lost and the number of crayons given away is 322. -/
theorem crayons_difference (initial : ℕ) (given_away : ℕ) (lost : ℕ)
    (h1 : initial = 110)
    (h2 : given_away = 90)
    (h3 : lost = 412) :
    lost - given_away = 322 := by
  sorry

end NUMINAMATH_CALUDE_crayons_difference_l1595_159505


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1595_159590

/-- A right triangle with the given cone volume properties has a hypotenuse of approximately 21.3 cm. -/
theorem right_triangle_hypotenuse (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (1 / 3 * π * y^2 * x = 1250 * π) →
  (1 / 3 * π * x^2 * y = 2700 * π) →
  abs (Real.sqrt (x^2 + y^2) - 21.3) < 0.1 := by
  sorry

#check right_triangle_hypotenuse

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1595_159590


namespace NUMINAMATH_CALUDE_age_problem_l1595_159523

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 32 → 
  b = 12 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l1595_159523


namespace NUMINAMATH_CALUDE_initial_number_proof_l1595_159591

theorem initial_number_proof : ∃ x : ℝ, (x / 34) * 15 + 270 = 405 ∧ x = 306 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1595_159591


namespace NUMINAMATH_CALUDE_waiter_earnings_correct_l1595_159544

/-- Calculates the waiter's total earnings during a shift given the specified conditions -/
def waiterEarnings (noTipCustomers tipThreeCustomers tipFourCustomers tipFiveCustomers : ℕ)
  (tipPoolPercentage mealCost employeeDiscountPercentage : ℚ) : ℚ :=
  let totalCustomers := noTipCustomers + tipThreeCustomers + tipFourCustomers + tipFiveCustomers
  let totalTips := 3 * tipThreeCustomers + 4 * tipFourCustomers + 5 * tipFiveCustomers
  let tipPoolContribution := tipPoolPercentage * totalTips
  let netTips := totalTips - tipPoolContribution
  let discountedMealCost := mealCost * (1 - employeeDiscountPercentage)
  netTips - discountedMealCost

/-- Theorem stating that the waiter's earnings during the shift equal $64.20 -/
theorem waiter_earnings_correct :
  waiterEarnings 5 8 6 6 (1/10) 7.5 (1/5) = 64.2 := by
  sorry

end NUMINAMATH_CALUDE_waiter_earnings_correct_l1595_159544


namespace NUMINAMATH_CALUDE_roses_cut_l1595_159514

theorem roses_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) 
  (h1 : initial_roses = 13)
  (h2 : initial_orchids = 84)
  (h3 : final_roses = 14)
  (h4 : final_orchids = 91) :
  final_roses - initial_roses = 1 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l1595_159514


namespace NUMINAMATH_CALUDE_planet_coloring_theorem_specific_planet_coloring_case_l1595_159525

/-- The number of colors needed for planet coloring -/
def colors_needed (num_planets : ℕ) (num_people : ℕ) : ℕ :=
  num_planets * num_people

/-- Theorem: In the planet coloring scenario, the number of colors needed
    is equal to the number of planets multiplied by the number of people coloring. -/
theorem planet_coloring_theorem (num_planets : ℕ) (num_people : ℕ) :
  colors_needed num_planets num_people = num_planets * num_people :=
by
  sorry

/-- The specific case mentioned in the problem -/
theorem specific_planet_coloring_case :
  colors_needed 8 3 = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_planet_coloring_theorem_specific_planet_coloring_case_l1595_159525


namespace NUMINAMATH_CALUDE_expression_equality_l1595_159516

theorem expression_equality : 
  abs (-3) - Real.sqrt 8 - (1/2)⁻¹ + 2 * Real.cos (π/4) = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1595_159516


namespace NUMINAMATH_CALUDE_debby_total_texts_l1595_159511

def texts_before_noon : ℕ := 21
def initial_texts_after_noon : ℕ := 2
def hours_after_noon : ℕ := 12

def texts_after_noon (n : ℕ) : ℕ := initial_texts_after_noon * 2^n

def total_texts : ℕ := texts_before_noon + (Finset.sum (Finset.range hours_after_noon) texts_after_noon)

theorem debby_total_texts : total_texts = 8211 := by sorry

end NUMINAMATH_CALUDE_debby_total_texts_l1595_159511


namespace NUMINAMATH_CALUDE_rain_amount_l1595_159564

theorem rain_amount (malina_initial : ℕ) (jahoda_initial : ℕ) (rain_amount : ℕ) : 
  malina_initial = 48 →
  malina_initial = jahoda_initial + 32 →
  (malina_initial + rain_amount) - (jahoda_initial + rain_amount) = 32 →
  malina_initial + rain_amount = 2 * (jahoda_initial + rain_amount) →
  rain_amount = 16 := by
sorry

end NUMINAMATH_CALUDE_rain_amount_l1595_159564


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l1595_159532

theorem smallest_five_digit_divisible_by_first_five_primes :
  ∃ (n : ℕ), 
    (n ≥ 10000 ∧ n < 100000) ∧ 
    (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 → 
      (2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 11 ∣ m) → m ≥ n) ∧
    (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n) ∧
    n = 11550 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l1595_159532


namespace NUMINAMATH_CALUDE_distance_A_to_y_axis_l1595_159568

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate -/
def distanceToYAxis (x : ℝ) (y : ℝ) : ℝ := |x|

/-- Point A has coordinates (2, -3) -/
def pointA : ℝ × ℝ := (2, -3)

/-- Theorem: The distance from point A(2, -3) to the y-axis is 2 -/
theorem distance_A_to_y_axis :
  distanceToYAxis pointA.1 pointA.2 = 2 := by sorry

end NUMINAMATH_CALUDE_distance_A_to_y_axis_l1595_159568


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_75_plus_6_l1595_159565

-- Define the function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the function to get the units digit of 7^n
def unitsDigitOf7Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case should never occur

-- Theorem statement
theorem units_digit_of_7_power_75_plus_6 :
  unitsDigit (unitsDigitOf7Power 75 + 6) = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_75_plus_6_l1595_159565


namespace NUMINAMATH_CALUDE_line_equation_through_points_l1595_159593

/-- The equation 3y - 5x = 1 represents the line passing through points (-2, -3) and (4, 7) -/
theorem line_equation_through_points :
  let point1 : ℝ × ℝ := (-2, -3)
  let point2 : ℝ × ℝ := (4, 7)
  let line_eq (x y : ℝ) := 3 * y - 5 * x = 1
  (line_eq point1.1 point1.2 ∧ line_eq point2.1 point2.2) := by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l1595_159593


namespace NUMINAMATH_CALUDE_log_inequality_l1595_159546

theorem log_inequality : 
  let m := Real.log 0.6 / Real.log 0.3
  let n := (1/2) * (Real.log 0.6 / Real.log 2)
  m + n > m * n := by sorry

end NUMINAMATH_CALUDE_log_inequality_l1595_159546


namespace NUMINAMATH_CALUDE_min_value_theorem_l1595_159583

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hm : m > 0) (hn : n > 0) (h_intersection : m + 4*n = 1) : 
  (1/m + 4/n) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1595_159583


namespace NUMINAMATH_CALUDE_f_inequality_solutions_l1595_159504

/-- The function f(x) = (x-a)(x-2) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * (x - 2)

theorem f_inequality_solutions :
  (∀ x, f 1 x > 0 ↔ x ∈ Set.Ioi 2 ∪ Set.Iic 1) ∧
  (∀ x, f 2 x < 0 → False) ∧
  (∀ a, a > 2 → ∀ x, f a x < 0 ↔ x ∈ Set.Ioo 2 a) ∧
  (∀ a, a < 2 → ∀ x, f a x < 0 ↔ x ∈ Set.Ioo a 2) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_solutions_l1595_159504


namespace NUMINAMATH_CALUDE_super_soup_revenue_theorem_l1595_159538

def super_soup_revenue (
  initial_stores : ℕ)
  (initial_avg_revenue : ℝ)
  (new_stores_2019 : ℕ)
  (new_revenue_2019 : ℝ)
  (closed_stores_2019 : ℕ)
  (closed_revenue_2019 : ℝ)
  (closed_expense_2019 : ℝ)
  (new_stores_2020 : ℕ)
  (new_revenue_2020 : ℝ)
  (closed_stores_2020 : ℕ)
  (closed_revenue_2020 : ℝ)
  (closed_expense_2020 : ℝ)
  (avg_expense : ℝ) : ℝ :=
  let initial_revenue := initial_stores * initial_avg_revenue
  let revenue_2019 := initial_revenue + new_stores_2019 * new_revenue_2019 - closed_stores_2019 * closed_revenue_2019
  let net_revenue_2019 := revenue_2019 + closed_stores_2019 * (closed_revenue_2019 - closed_expense_2019)
  let stores_2019 := initial_stores + new_stores_2019 - closed_stores_2019
  let revenue_2020 := net_revenue_2019 + new_stores_2020 * new_revenue_2020 - closed_stores_2020 * closed_revenue_2020
  let net_revenue_2020 := revenue_2020 + closed_stores_2020 * (closed_expense_2020 - closed_revenue_2020)
  let final_stores := stores_2019 + new_stores_2020 - closed_stores_2020
  net_revenue_2020 - final_stores * avg_expense

theorem super_soup_revenue_theorem :
  super_soup_revenue 23 500000 5 450000 2 300000 350000 10 600000 6 350000 380000 400000 = 5130000 := by
  sorry

end NUMINAMATH_CALUDE_super_soup_revenue_theorem_l1595_159538


namespace NUMINAMATH_CALUDE_friends_total_score_l1595_159585

/-- Given three friends' scores in a table football game, prove their total score. -/
theorem friends_total_score (darius_score matt_score marius_score : ℕ) : 
  marius_score = darius_score + 3 →
  matt_score = darius_score + 5 →
  darius_score = 10 →
  darius_score + matt_score + marius_score = 38 := by
sorry


end NUMINAMATH_CALUDE_friends_total_score_l1595_159585


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1595_159598

/-- Given a line y = mx + b, if the reflection of point (1, 2) across this line is (7, 6), then m + b = 8.5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    (x - 1)^2 + (y - 2)^2 = (7 - x)^2 + (6 - y)^2 ∧ 
    (x + 7) / 2 = (y + 6) / 2 / m + b ∧
    (y + 6) / 2 = m * (x + 7) / 2 + b) → 
  m + b = 8.5 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l1595_159598


namespace NUMINAMATH_CALUDE_chef_potato_count_l1595_159518

/-- The number of potatoes a chef needs to cook -/
def total_potatoes (cooked : ℕ) (cooking_time_per_potato : ℕ) (remaining_cooking_time : ℕ) : ℕ :=
  cooked + remaining_cooking_time / cooking_time_per_potato

/-- Proof that the chef needs to cook 13 potatoes in total -/
theorem chef_potato_count : total_potatoes 5 6 48 = 13 := by
  sorry

#eval total_potatoes 5 6 48

end NUMINAMATH_CALUDE_chef_potato_count_l1595_159518


namespace NUMINAMATH_CALUDE_gcd_problem_l1595_159509

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 17 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 65 * b + 143) (5 * b + 22) = 33 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1595_159509


namespace NUMINAMATH_CALUDE_triangle_isosceles_from_quadratic_equation_l1595_159595

/-- A triangle with sides a, b, and c is isosceles if the quadratic equation
    (c-b)x^2 + 2(b-a)x + (a-b) = 0 has two equal real roots. -/
theorem triangle_isosceles_from_quadratic_equation (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_eq : ∃ x : ℝ, (c - b) * x^2 + 2*(b - a)*x + (a - b) = 0 ∧ 
    ∀ y : ℝ, (c - b) * y^2 + 2*(b - a)*y + (a - b) = 0 → y = x) :
  (a = b ∧ c ≠ b) ∨ (a = c ∧ b ≠ c) ∨ (b = c ∧ a ≠ b) :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_from_quadratic_equation_l1595_159595


namespace NUMINAMATH_CALUDE_remaining_files_indeterminate_l1595_159570

/-- Represents the state of Dave's phone -/
structure PhoneState where
  apps : ℕ
  files : ℕ

/-- Represents the change in Dave's phone state -/
structure PhoneStateChange where
  initialState : PhoneState
  finalState : PhoneState
  appsDeleted : ℕ

/-- Predicate to check if a PhoneStateChange is valid according to the problem conditions -/
def isValidPhoneStateChange (change : PhoneStateChange) : Prop :=
  change.initialState.apps = 16 ∧
  change.initialState.files = 77 ∧
  change.finalState.apps = 5 ∧
  change.appsDeleted = 11 ∧
  change.initialState.apps - change.appsDeleted = change.finalState.apps ∧
  change.finalState.files ≤ change.initialState.files

/-- Theorem stating that the number of remaining files cannot be uniquely determined -/
theorem remaining_files_indeterminate (change : PhoneStateChange) 
  (h : isValidPhoneStateChange change) :
  ∃ (x y : ℕ), x ≠ y ∧ 
    isValidPhoneStateChange { change with finalState := { change.finalState with files := x } } ∧
    isValidPhoneStateChange { change with finalState := { change.finalState with files := y } } :=
  sorry

end NUMINAMATH_CALUDE_remaining_files_indeterminate_l1595_159570


namespace NUMINAMATH_CALUDE_hiker_count_l1595_159540

theorem hiker_count : ∃ (n m : ℕ), n > 13 ∧ n = 23 ∧ m > 0 ∧ 
  2 * m ≡ 1 [MOD n] ∧ 3 * m ≡ 13 [MOD n] := by
  sorry

end NUMINAMATH_CALUDE_hiker_count_l1595_159540


namespace NUMINAMATH_CALUDE_fish_population_estimate_l1595_159522

/-- Estimate the fish population in a pond given tagging and sampling data --/
theorem fish_population_estimate
  (initial_tagged : ℕ)
  (august_sample : ℕ)
  (august_tagged : ℕ)
  (left_pond_ratio : ℚ)
  (new_fish_ratio : ℚ)
  (h_initial_tagged : initial_tagged = 50)
  (h_august_sample : august_sample = 80)
  (h_august_tagged : august_tagged = 4)
  (h_left_pond : left_pond_ratio = 3/10)
  (h_new_fish : new_fish_ratio = 45/100)
  (h_representative_sample : True)  -- Assuming the sample is representative
  (h_negligible_tag_loss : True)    -- Assuming tag loss is negligible
  : ↑initial_tagged * (august_sample * (1 - new_fish_ratio)) / august_tagged = 550 := by
  sorry


end NUMINAMATH_CALUDE_fish_population_estimate_l1595_159522


namespace NUMINAMATH_CALUDE_first_to_light_is_match_l1595_159566

/-- Represents items that can be lit --/
inductive LightableItem
  | Match
  | Candle
  | KeroseneLamp
  | Stove

/-- Represents the state of a room --/
structure Room where
  isDark : Bool
  hasMatch : Bool
  items : List LightableItem

/-- Determines the first item that must be lit in a given room --/
def firstItemToLight (room : Room) : LightableItem := by sorry

/-- Theorem: The first item to light in a dark room with a match and other lightable items is the match itself --/
theorem first_to_light_is_match (room : Room) 
  (h1 : room.isDark = true) 
  (h2 : room.hasMatch = true) 
  (h3 : LightableItem.Candle ∈ room.items) 
  (h4 : LightableItem.KeroseneLamp ∈ room.items) 
  (h5 : LightableItem.Stove ∈ room.items) : 
  firstItemToLight room = LightableItem.Match := by sorry

end NUMINAMATH_CALUDE_first_to_light_is_match_l1595_159566


namespace NUMINAMATH_CALUDE_room_population_lower_limit_l1595_159580

theorem room_population_lower_limit (total : ℕ) (under_21 : ℕ) (over_65 : ℕ) : 
  under_21 = 30 →
  under_21 = (3 : ℚ) / 7 * total →
  over_65 = (5 : ℚ) / 10 * total →
  ∃ (upper : ℕ), total ∈ Set.Icc total upper →
  70 ≤ total :=
by sorry

end NUMINAMATH_CALUDE_room_population_lower_limit_l1595_159580


namespace NUMINAMATH_CALUDE_ducks_joined_l1595_159527

theorem ducks_joined (initial_ducks final_ducks : ℕ) (h : final_ducks ≥ initial_ducks) :
  final_ducks - initial_ducks = final_ducks - initial_ducks :=
by sorry

end NUMINAMATH_CALUDE_ducks_joined_l1595_159527


namespace NUMINAMATH_CALUDE_ten_power_plus_eight_div_nine_is_integer_l1595_159589

theorem ten_power_plus_eight_div_nine_is_integer (n : ℕ) : ∃ k : ℤ, (10^n : ℤ) + 8 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_ten_power_plus_eight_div_nine_is_integer_l1595_159589


namespace NUMINAMATH_CALUDE_tournament_theorem_l1595_159572

/-- Represents a team in the tournament -/
inductive Team : Type
| A
| B
| C

/-- Represents the state of a player (active or eliminated) -/
inductive PlayerState : Type
| Active
| Eliminated

/-- Represents the state of the tournament -/
structure TournamentState :=
  (team_players : Team → Fin 9 → PlayerState)
  (matches_played : ℕ)
  (champion_wins : ℕ)

/-- The rules of the tournament -/
def tournament_rules (initial_state : TournamentState) : Prop :=
  ∀ (t : Team), ∃ (i : Fin 9), initial_state.team_players t i = PlayerState.Active

/-- The condition for a team to be eliminated -/
def team_eliminated (state : TournamentState) (t : Team) : Prop :=
  ∀ (i : Fin 9), state.team_players t i = PlayerState.Eliminated

/-- The condition for the tournament to end -/
def tournament_ended (state : TournamentState) : Prop :=
  ∃ (t1 t2 : Team), t1 ≠ t2 ∧ team_eliminated state t1 ∧ team_eliminated state t2

/-- The main theorem to prove -/
theorem tournament_theorem 
  (initial_state : TournamentState) 
  (h_rules : tournament_rules initial_state) :
  (∃ (final_state : TournamentState), 
    tournament_ended final_state ∧ 
    final_state.champion_wins ≥ 9) ∧
  (∀ (final_state : TournamentState),
    tournament_ended final_state → 
    final_state.champion_wins = 11 → 
    final_state.matches_played ≥ 24) :=
sorry

end NUMINAMATH_CALUDE_tournament_theorem_l1595_159572


namespace NUMINAMATH_CALUDE_range_of_a_l1595_159559

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 4*x + a ≥ -2 * x^2 + 1) ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1595_159559


namespace NUMINAMATH_CALUDE_min_dot_product_l1595_159567

/-- A line with direction vector (4, -4) passing through (0, -4) -/
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (t, -t - 4)}

/-- Two points on line_l -/
def point_on_line (M N : ℝ × ℝ) : Prop :=
  M ∈ line_l ∧ N ∈ line_l

/-- Distance between two points is 4 -/
def distance_is_4 (M N : ℝ × ℝ) : Prop :=
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 16

/-- Dot product of OM and ON -/
def dot_product (M N : ℝ × ℝ) : ℝ :=
  M.1 * N.1 + M.2 * N.2

theorem min_dot_product (M N : ℝ × ℝ) 
  (h1 : point_on_line M N) 
  (h2 : distance_is_4 M N) : 
  ∃ min_val : ℝ, min_val = 4 ∧ ∀ M' N' : ℝ × ℝ, 
    point_on_line M' N' → distance_is_4 M' N' → 
    dot_product M' N' ≥ min_val :=
  sorry

end NUMINAMATH_CALUDE_min_dot_product_l1595_159567


namespace NUMINAMATH_CALUDE_nine_points_chords_l1595_159543

/-- The number of different chords that can be drawn by connecting two points
    out of n points on the circumference of a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: There are 36 different chords that can be drawn by connecting two
    points out of nine points on the circumference of a circle -/
theorem nine_points_chords : num_chords 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nine_points_chords_l1595_159543


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1595_159560

theorem perfect_square_condition (n : ℕ) : 
  (∃ k : ℕ, n * 2^(n + 1) + 1 = k^2) ↔ n = 0 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1595_159560


namespace NUMINAMATH_CALUDE_men_absent_l1595_159524

/-- Proves that 15 men became absent given the original group size, planned completion time, and actual completion time. -/
theorem men_absent (total_men : ℕ) (planned_days : ℕ) (actual_days : ℕ) 
  (h1 : total_men = 180) 
  (h2 : planned_days = 55)
  (h3 : actual_days = 60) :
  ∃ (absent_men : ℕ), 
    absent_men = 15 ∧ 
    (total_men * planned_days = (total_men - absent_men) * actual_days) :=
by
  sorry

#check men_absent

end NUMINAMATH_CALUDE_men_absent_l1595_159524


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1595_159577

/-- Given two real numbers x and y that are inversely proportional,
    prove that if x + y = 30 and x = 3y, then when x = -6, y = -28.125 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) 
  (h1 : x * y = k)  -- x and y are inversely proportional
  (h2 : x + y = 30) -- sum condition
  (h3 : x = 3 * y)  -- x is three times y
  : x = -6 → y = -28.125 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1595_159577


namespace NUMINAMATH_CALUDE_combined_sum_equals_3751_l1595_159510

/-- The first element of the nth set in the pattern -/
def first_element (n : ℕ) : ℕ := 1 + n * (n - 1) / 2

/-- The last element of the nth set in the pattern -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set -/
def set_sum (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- The combined sum of elements in the 15th and 16th sets -/
def combined_sum : ℕ := set_sum 15 + set_sum 16

theorem combined_sum_equals_3751 : combined_sum = 3751 := by
  sorry

end NUMINAMATH_CALUDE_combined_sum_equals_3751_l1595_159510


namespace NUMINAMATH_CALUDE_cost_price_percentage_l1595_159534

/-- Proves that the cost price is 25% of the marked price given the conditions -/
theorem cost_price_percentage (MP : ℝ) (CP : ℝ) : 
  (∃ x : ℝ, CP = (x / 100) * MP) →  -- Cost price is some percentage of marked price
  (MP / 2 = 2 * CP) →               -- After 50% discount, gain is 100%
  CP = 0.25 * MP :=                 -- Cost price is 25% of marked price
by sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l1595_159534


namespace NUMINAMATH_CALUDE_finite_primes_imply_equal_bases_l1595_159541

def divides_set (a b c d : ℕ+) : Set ℕ :=
  {p : ℕ | ∃ n : ℕ, n > 0 ∧ p.Prime ∧ p ∣ (a * b^n + c * d^n)}

theorem finite_primes_imply_equal_bases (a b c d : ℕ+) :
  (Set.Finite (divides_set a b c d)) → b = d := by
  sorry

end NUMINAMATH_CALUDE_finite_primes_imply_equal_bases_l1595_159541


namespace NUMINAMATH_CALUDE_roots_product_l1595_159539

theorem roots_product (a b c d : ℝ) (h1 : 36 * a^3 - 66 * a^2 + 31 * a - 4 = 0)
  (h2 : 36 * b^3 - 66 * b^2 + 31 * b - 4 = 0)
  (h3 : 36 * c^3 - 66 * c^2 + 31 * c - 4 = 0)
  (h4 : b - a = c - b) -- arithmetic progression
  (h5 : a < b ∧ b < c) -- ordering of roots
  : a * c = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_l1595_159539


namespace NUMINAMATH_CALUDE_wall_width_l1595_159550

/-- The width of a wall given its dimensions, brick dimensions, and number of bricks required. -/
theorem wall_width
  (wall_length : ℝ)
  (wall_height : ℝ)
  (brick_length : ℝ)
  (brick_width : ℝ)
  (brick_height : ℝ)
  (num_bricks : ℕ)
  (h1 : wall_length = 7)
  (h2 : wall_height = 6)
  (h3 : brick_length = 0.25)
  (h4 : brick_width = 0.1125)
  (h5 : brick_height = 0.06)
  (h6 : num_bricks = 5600) :
  ∃ (wall_width : ℝ), wall_width = 0.225 ∧
    wall_length * wall_height * wall_width = ↑num_bricks * brick_length * brick_width * brick_height :=
by sorry

end NUMINAMATH_CALUDE_wall_width_l1595_159550


namespace NUMINAMATH_CALUDE_means_inequality_l1595_159526

theorem means_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (a + b + c) / 3 > (a * b * c) ^ (1/3) ∧ 
  (a * b * c) ^ (1/3) > 3 / ((1/a) + (1/b) + (1/c)) := by
  sorry

#check means_inequality

end NUMINAMATH_CALUDE_means_inequality_l1595_159526


namespace NUMINAMATH_CALUDE_workout_solution_correct_l1595_159533

/-- Laura's workout parameters -/
structure WorkoutParams where
  bike_distance : ℝ
  bike_rate : ℝ → ℝ
  transition_time : ℝ
  run_distance : ℝ
  total_time : ℝ

/-- The solution to Laura's workout problem -/
def workout_solution (p : WorkoutParams) : ℝ :=
  8

/-- Theorem stating that the workout_solution is correct -/
theorem workout_solution_correct (p : WorkoutParams) 
  (h1 : p.bike_distance = 25)
  (h2 : p.bike_rate = fun x => 3 * x + 1)
  (h3 : p.transition_time = 1/6)  -- 10 minutes in hours
  (h4 : p.run_distance = 8)
  (h5 : p.total_time = 13/6)  -- 130 minutes in hours
  : ∃ (x : ℝ), 
    x = workout_solution p ∧ 
    p.bike_distance / (p.bike_rate x) + p.transition_time + p.run_distance / x = p.total_time :=
  sorry

#check workout_solution_correct

end NUMINAMATH_CALUDE_workout_solution_correct_l1595_159533


namespace NUMINAMATH_CALUDE_perfect_power_multiple_l1595_159578

theorem perfect_power_multiple : ∃ (n : ℕ), 
  n > 0 ∧ 
  ∃ (a b c : ℕ), 
    2 * n = a^2 ∧ 
    3 * n = b^3 ∧ 
    5 * n = c^5 := by
  sorry

end NUMINAMATH_CALUDE_perfect_power_multiple_l1595_159578


namespace NUMINAMATH_CALUDE_pirate_coins_l1595_159545

theorem pirate_coins (x : ℕ) : 
  let round1 := x / 2
  let round2 := (x - round1 + round1) / 2
  let round3 := (x - round1 + round2 - round2) / 2
  (round3 = 15 ∧ (x - round1 + round2 - round3) = 33) → x = 24 :=
by sorry

end NUMINAMATH_CALUDE_pirate_coins_l1595_159545


namespace NUMINAMATH_CALUDE_f_2023_equals_1_l1595_159513

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def period_4 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x

theorem f_2023_equals_1 (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 2) = f (2 - x))
  (h3 : ∀ x ∈ Set.Icc 0 2, f x = x^2) : 
  f 2023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2023_equals_1_l1595_159513


namespace NUMINAMATH_CALUDE_perpendicular_bisector_and_parallel_line_l1595_159597

-- Define points A, B, and P
def A : ℝ × ℝ := (8, -6)
def B : ℝ × ℝ := (2, 2)
def P : ℝ × ℝ := (2, -3)

-- Define the perpendicular bisector equation
def perpendicular_bisector (x y : ℝ) : Prop :=
  3 * x - 4 * y - 23 = 0

-- Define the parallel line equation
def parallel_line (x y : ℝ) : Prop :=
  4 * x + 3 * y + 1 = 0

-- Theorem statement
theorem perpendicular_bisector_and_parallel_line :
  (∀ x y : ℝ, perpendicular_bisector x y ↔ 
    (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1) ∧
    (x - (A.1 + B.1) / 2) ^ 2 + (y - (A.2 + B.2) / 2) ^ 2 = 
    ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) / 4) ∧
  (∀ x y : ℝ, parallel_line x y ↔
    (y - P.2) * (B.1 - A.1) = (x - P.1) * (B.2 - A.2)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_and_parallel_line_l1595_159597


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l1595_159553

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents a T-shape formed by two rectangles -/
structure TShape where
  top : Rectangle
  bottom : Rectangle

/-- Calculates the perimeter of a T-shape -/
def TShape.perimeter (t : TShape) : ℝ :=
  t.top.perimeter + t.bottom.perimeter - 2 * t.top.width

theorem t_shape_perimeter : 
  let t : TShape := {
    top := { width := 1, height := 4 },
    bottom := { width := 5, height := 2 }
  }
  TShape.perimeter t = 20 := by sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l1595_159553


namespace NUMINAMATH_CALUDE_expression_evaluation_l1595_159579

theorem expression_evaluation : (-7)^3 / 7^2 + 4^3 - 5 * 2^2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1595_159579


namespace NUMINAMATH_CALUDE_specific_shape_perimeter_l1595_159515

/-- A shape consisting of a regular hexagon, six triangles, and six squares -/
structure Shape where
  hexagon_side : ℝ
  num_triangles : ℕ
  num_squares : ℕ

/-- The outer perimeter of the shape -/
def outer_perimeter (s : Shape) : ℝ :=
  12 * s.hexagon_side

/-- Theorem stating that the outer perimeter of the specific shape is 216 cm -/
theorem specific_shape_perimeter :
  ∃ (s : Shape), s.hexagon_side = 18 ∧ s.num_triangles = 6 ∧ s.num_squares = 6 ∧ outer_perimeter s = 216 :=
by sorry

end NUMINAMATH_CALUDE_specific_shape_perimeter_l1595_159515


namespace NUMINAMATH_CALUDE_tetrahedron_centroid_intersection_sum_l1595_159542

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron defined by four points -/
structure Tetrahedron where
  P : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Centroid of a tetrahedron -/
def centroid (t : Tetrahedron) : Point3D := sorry

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- Distance between two points in 3D space -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Intersection point of a line and a face of the tetrahedron -/
def intersectionPoint (l : Line3D) (t : Tetrahedron) (face : Fin 4) : Point3D := sorry

theorem tetrahedron_centroid_intersection_sum (t : Tetrahedron) (l : Line3D) : 
  let G := centroid t
  let M := intersectionPoint l t 0
  let N := intersectionPoint l t 1
  let S := intersectionPoint l t 2
  let T := intersectionPoint l t 3
  1 / distance G M + 1 / distance G N + 1 / distance G S + 1 / distance G T = 0 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_centroid_intersection_sum_l1595_159542


namespace NUMINAMATH_CALUDE_interesting_numbers_200_to_400_l1595_159563

/-- A natural number is interesting if there exists another natural number that satisfies certain conditions. -/
def IsInteresting (A : ℕ) : Prop :=
  ∃ B : ℕ, A > B ∧ Nat.Prime (A - B) ∧ ∃ n : ℕ, A * B = n * n

/-- The theorem stating the interesting numbers between 200 and 400. -/
theorem interesting_numbers_200_to_400 :
  ∀ A : ℕ, 200 < A → A < 400 → (IsInteresting A ↔ A = 225 ∨ A = 256 ∨ A = 361) := by
  sorry


end NUMINAMATH_CALUDE_interesting_numbers_200_to_400_l1595_159563


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1595_159596

theorem arithmetic_sequence_length (a₁ aₙ d : ℤ) (n : ℕ) : 
  a₁ = 128 → aₙ = 14 → d = -3 → aₙ = a₁ + (n - 1) * d → n = 39 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1595_159596


namespace NUMINAMATH_CALUDE_farm_problem_l1595_159569

/-- The farm problem -/
theorem farm_problem (H C : ℕ) : 
  (H - 15) / (C + 15) = 3 →  -- After transaction, ratio is 3:1
  H - 15 = C + 15 + 70 →    -- After transaction, 70 more horses than cows
  H / C = 6                  -- Initial ratio is 6:1
:= by sorry

end NUMINAMATH_CALUDE_farm_problem_l1595_159569


namespace NUMINAMATH_CALUDE_interior_triangles_count_l1595_159548

/-- The number of points on the circle -/
def n : ℕ := 9

/-- The number of triangles formed inside the circle -/
def num_triangles : ℕ := Nat.choose n 6

/-- Theorem stating the number of triangles formed inside the circle -/
theorem interior_triangles_count : num_triangles = 84 := by
  sorry

end NUMINAMATH_CALUDE_interior_triangles_count_l1595_159548


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l1595_159562

theorem complement_intersection_problem (U A B : Set ℕ) : 
  U = {1, 2, 3, 4, 5} →
  A = {1, 2, 3} →
  B = {3, 4, 5} →
  (U \ A) ∩ B = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l1595_159562


namespace NUMINAMATH_CALUDE_people_on_stairs_l1595_159558

/-- The number of ways to arrange people on steps. -/
def arrange_people (num_people : ℕ) (num_steps : ℕ) (max_per_step : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements for the given problem. -/
theorem people_on_stairs :
  arrange_people 4 7 3 = 2394 := by
  sorry

end NUMINAMATH_CALUDE_people_on_stairs_l1595_159558


namespace NUMINAMATH_CALUDE_smallest_missing_number_is_22_l1595_159512

/-- Represents a problem in HMMT November 2023 -/
structure HMMTProblem where
  round : String
  number : Nat

/-- The set of all problems in HMMT November 2023 -/
def HMMTProblems : Set HMMTProblem := sorry

/-- A number appears in HMMT November 2023 if it's used in at least one problem -/
def appears_in_HMMT (n : Nat) : Prop :=
  ∃ (p : HMMTProblem), p ∈ HMMTProblems ∧ p.number = n

theorem smallest_missing_number_is_22 :
  (∀ n : Nat, n > 0 ∧ n ≤ 21 → appears_in_HMMT n) →
  (¬ appears_in_HMMT 22) →
  ∀ m : Nat, m > 0 ∧ ¬ appears_in_HMMT m → m ≥ 22 :=
sorry

end NUMINAMATH_CALUDE_smallest_missing_number_is_22_l1595_159512


namespace NUMINAMATH_CALUDE_rectangle_ratio_square_l1595_159536

theorem rectangle_ratio_square (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a < b) :
  let d := Real.sqrt (a^2 + b^2)
  (a / b = b / d) → (a / b)^2 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_square_l1595_159536


namespace NUMINAMATH_CALUDE_accuracy_of_0_598_l1595_159528

/-- Represents the place value of a digit in a decimal number. -/
inductive PlaceValue
  | Ones
  | Tenths
  | Hundredths
  | Thousandths
  | TenThousandths
  deriving Repr

/-- Determines the place value of accuracy for a given decimal number. -/
def placeOfAccuracy (n : Float) : PlaceValue :=
  match n.toString.split (· = '.') with
  | [_, fractional] =>
    match fractional.length with
    | 1 => PlaceValue.Tenths
    | 2 => PlaceValue.Hundredths
    | 3 => PlaceValue.Thousandths
    | _ => PlaceValue.TenThousandths
  | _ => PlaceValue.Ones

/-- Theorem: The approximate number 0.598 is accurate to the thousandths place. -/
theorem accuracy_of_0_598 :
  placeOfAccuracy 0.598 = PlaceValue.Thousandths := by
  sorry

end NUMINAMATH_CALUDE_accuracy_of_0_598_l1595_159528


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l1595_159556

/-- Given two rectangles with equal areas, where one has length 5 and width 24,
    and the other has width 10, prove that the length of the second rectangle is 12. -/
theorem equal_area_rectangles (l₁ w₁ w₂ : ℝ) (h₁ : l₁ = 5) (h₂ : w₁ = 24) (h₃ : w₂ = 10) :
  let a₁ := l₁ * w₁
  let l₂ := a₁ / w₂
  l₂ = 12 := by sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l1595_159556


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l1595_159547

/-- Given two circles in the xy-plane, this theorem states that
    their common chord lies on a specific line. -/
theorem common_chord_of_circles (x y : ℝ) : 
  (x^2 + y^2 + 2*x = 0) ∧ (x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) := by
  sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l1595_159547


namespace NUMINAMATH_CALUDE_cow_count_is_24_l1595_159517

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (a : AnimalCount) : ℕ := 2 * a.ducks + 4 * a.cows

/-- The total number of heads in the group -/
def totalHeads (a : AnimalCount) : ℕ := a.ducks + a.cows

/-- The condition given in the problem -/
def satisfiesCondition (a : AnimalCount) : Prop :=
  totalLegs a = 2 * totalHeads a + 48

theorem cow_count_is_24 (a : AnimalCount) (h : satisfiesCondition a) : a.cows = 24 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_is_24_l1595_159517


namespace NUMINAMATH_CALUDE_degree_of_P_l1595_159508

/-- The polynomial in question -/
def P (a b : ℚ) : ℚ := 2/3 * a * b^2 + 4/3 * a^3 * b + 1/3

/-- The degree of a polynomial -/
def polynomial_degree (p : ℚ → ℚ → ℚ) : ℕ :=
  sorry  -- Definition of polynomial degree

theorem degree_of_P : polynomial_degree P = 4 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_P_l1595_159508


namespace NUMINAMATH_CALUDE_west_ten_meters_representation_l1595_159520

/-- Represents the direction of walking -/
inductive Direction
  | East
  | West

/-- Represents a walk with a distance and direction -/
structure Walk where
  distance : ℝ
  direction : Direction

/-- Converts a walk to its numerical representation -/
def Walk.toNumber (w : Walk) : ℝ :=
  match w.direction with
  | Direction.East => w.distance
  | Direction.West => -w.distance

theorem west_ten_meters_representation :
  let w : Walk := { distance := 10, direction := Direction.West }
  w.toNumber = -10 := by sorry

end NUMINAMATH_CALUDE_west_ten_meters_representation_l1595_159520


namespace NUMINAMATH_CALUDE_palace_rotation_l1595_159551

theorem palace_rotation (x : ℕ) : 
  (x % 30 = 15 ∧ x % 50 = 25 ∧ x % 70 = 35) → x ≥ 525 :=
by sorry

end NUMINAMATH_CALUDE_palace_rotation_l1595_159551


namespace NUMINAMATH_CALUDE_simplify_radical_fraction_l1595_159599

theorem simplify_radical_fraction :
  (3 * Real.sqrt 10) / (Real.sqrt 5 + 2) = 15 * Real.sqrt 2 - 6 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_fraction_l1595_159599


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1595_159554

theorem circle_line_intersection (k : ℝ) : 
  k ≤ -2 * Real.sqrt 2 → 
  ∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1595_159554


namespace NUMINAMATH_CALUDE_binomial_16_13_l1595_159501

theorem binomial_16_13 : Nat.choose 16 13 = 560 := by
  sorry

end NUMINAMATH_CALUDE_binomial_16_13_l1595_159501


namespace NUMINAMATH_CALUDE_factor_cubic_l1595_159507

theorem factor_cubic (a b c : ℝ) : 
  (∀ x, x^3 - 12*x + 16 = (x + 4)*(a*x^2 + b*x + c)) → 
  a*x^2 + b*x + c = (x - 2)^2 := by
sorry

end NUMINAMATH_CALUDE_factor_cubic_l1595_159507


namespace NUMINAMATH_CALUDE_ceasar_pages_read_l1595_159503

/-- The number of pages Ceasar has already read -/
def pages_read (total_pages remaining_pages : ℕ) : ℕ :=
  total_pages - remaining_pages

theorem ceasar_pages_read :
  pages_read 563 416 = 147 := by
  sorry

end NUMINAMATH_CALUDE_ceasar_pages_read_l1595_159503


namespace NUMINAMATH_CALUDE_min_fraction_sum_l1595_159535

def ValidDigits : Finset Nat := {1, 3, 4, 5, 6, 8, 9}

theorem min_fraction_sum (A B C D : Nat) 
  (hA : A ∈ ValidDigits) (hB : B ∈ ValidDigits) 
  (hC : C ∈ ValidDigits) (hD : D ∈ ValidDigits)
  (hAB : A ≠ B) (hAC : A ≠ C) (hAD : A ≠ D) 
  (hBC : B ≠ C) (hBD : B ≠ D) (hCD : C ≠ D)
  (hB_pos : B > 0) (hD_pos : D > 0) :
  (A : ℚ) / B + (C : ℚ) / D ≥ 11 / 24 := by
  sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l1595_159535


namespace NUMINAMATH_CALUDE_choose_3_from_15_l1595_159537

theorem choose_3_from_15 : Nat.choose 15 3 = 455 := by sorry

end NUMINAMATH_CALUDE_choose_3_from_15_l1595_159537


namespace NUMINAMATH_CALUDE_ski_camp_directions_l1595_159582

-- Define the four cardinal directions
inductive Direction
| North
| South
| East
| West

-- Define the four friends
inductive Friend
| Karel
| Mojmir
| Pepa
| Zdenda

-- Define a function that assigns a direction to each friend
def came_from : Friend → Direction := sorry

-- Define the statements made by each friend
def karel_statement : Prop :=
  came_from Friend.Karel ≠ Direction.North ∧ came_from Friend.Karel ≠ Direction.South

def mojmir_statement : Prop :=
  came_from Friend.Mojmir = Direction.South

def pepa_statement : Prop :=
  came_from Friend.Pepa = Direction.North

def zdenda_statement : Prop :=
  came_from Friend.Zdenda ≠ Direction.South

-- Define a function that checks if a statement is true
def is_true_statement : Friend → Prop
| Friend.Karel => karel_statement
| Friend.Mojmir => mojmir_statement
| Friend.Pepa => pepa_statement
| Friend.Zdenda => zdenda_statement

-- Theorem to prove
theorem ski_camp_directions :
  (∃! f : Friend, ¬is_true_statement f) ∧
  (came_from Friend.Zdenda = Direction.North) ∧
  (came_from Friend.Mojmir = Direction.South) ∧
  (¬is_true_statement Friend.Pepa) :=
by sorry

end NUMINAMATH_CALUDE_ski_camp_directions_l1595_159582


namespace NUMINAMATH_CALUDE_equation_solution_l1595_159555

theorem equation_solution : ∃! x : ℚ, x + 2/3 = 7/15 + 1/5 - x/2 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1595_159555


namespace NUMINAMATH_CALUDE_percentage_difference_l1595_159519

theorem percentage_difference : 
  (0.5 * 56 : ℝ) - (0.3 * 50 : ℝ) = 13 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l1595_159519


namespace NUMINAMATH_CALUDE_square_of_binomial_l1595_159581

theorem square_of_binomial (a : ℚ) : 
  (∃ (r s : ℚ), ∀ (x : ℚ), a * x^2 + 15 * x + 16 = (r * x + s)^2) → 
  a = 225 / 64 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l1595_159581


namespace NUMINAMATH_CALUDE_joe_hvac_cost_l1595_159561

/-- The total cost of an HVAC system with given parameters. -/
def hvac_total_cost (num_zones : ℕ) (vents_per_zone : ℕ) (cost_per_vent : ℕ) : ℕ :=
  num_zones * vents_per_zone * cost_per_vent

/-- Theorem stating that the total cost of Joe's HVAC system is $20,000. -/
theorem joe_hvac_cost :
  hvac_total_cost 2 5 2000 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_joe_hvac_cost_l1595_159561


namespace NUMINAMATH_CALUDE_similar_canister_capacity_l1595_159588

/-- Given that a small canister with volume 24 cm³ can hold 100 nails,
    prove that a similar canister with volume 72 cm³ can hold 300 nails,
    assuming the nails are packed in the same manner. -/
theorem similar_canister_capacity
  (small_volume : ℝ)
  (small_nails : ℕ)
  (large_volume : ℝ)
  (h1 : small_volume = 24)
  (h2 : small_nails = 100)
  (h3 : large_volume = 72)
  (h4 : small_volume > 0)
  (h5 : large_volume > 0) :
  (large_volume / small_volume) * small_nails = 300 := by
  sorry

#check similar_canister_capacity

end NUMINAMATH_CALUDE_similar_canister_capacity_l1595_159588


namespace NUMINAMATH_CALUDE_percent_greater_average_l1595_159500

theorem percent_greater_average (M N : ℝ) (h : M > N) :
  (M - N) / ((M + N) / 2) * 100 = 200 * (M - N) / (M + N) := by
  sorry

end NUMINAMATH_CALUDE_percent_greater_average_l1595_159500
