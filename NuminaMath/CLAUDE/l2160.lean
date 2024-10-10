import Mathlib

namespace square_sum_equals_two_l2160_216050

theorem square_sum_equals_two (a b : ℝ) (h1 : a * b = -1) (h2 : a - b = 2) : a^2 + b^2 = 2 := by
  sorry

end square_sum_equals_two_l2160_216050


namespace product_equals_zero_l2160_216011

def product_sequence (a : ℤ) : ℤ := (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a

theorem product_equals_zero : product_sequence 3 = 0 := by
  sorry

end product_equals_zero_l2160_216011


namespace square_equation_solution_l2160_216092

theorem square_equation_solution (b c x : ℝ) : 
  x^2 + c^2 = (b - x)^2 → x = (b^2 - c^2) / (2 * b) :=
by sorry

end square_equation_solution_l2160_216092


namespace adults_trekking_l2160_216086

/-- Represents the trekking group and meal information -/
structure TrekkingGroup where
  total_children : ℕ
  meal_adults : ℕ
  meal_children : ℕ
  adults_eaten : ℕ
  remaining_children : ℕ

/-- Theorem stating the number of adults who went trekking -/
theorem adults_trekking (group : TrekkingGroup) 
  (h1 : group.total_children = 70)
  (h2 : group.meal_adults = 70)
  (h3 : group.meal_children = 90)
  (h4 : group.adults_eaten = 42)
  (h5 : group.remaining_children = 36) :
  ∃ (adults_trekking : ℕ), adults_trekking = 70 := by
  sorry


end adults_trekking_l2160_216086


namespace compare_sqrt_expressions_l2160_216026

theorem compare_sqrt_expressions : 3 * Real.sqrt 5 > 2 * Real.sqrt 11 := by sorry

end compare_sqrt_expressions_l2160_216026


namespace solve_average_weight_l2160_216082

def average_weight_problem (weight_16 : ℝ) (weight_all : ℝ) (num_16 : ℕ) (num_8 : ℕ) : Prop :=
  let num_total : ℕ := num_16 + num_8
  let weight_8 : ℝ := (num_total * weight_all - num_16 * weight_16) / num_8
  weight_16 = 50.25 ∧ 
  weight_all = 48.55 ∧ 
  num_16 = 16 ∧ 
  num_8 = 8 ∧ 
  weight_8 = 45.15

theorem solve_average_weight : 
  ∃ (weight_16 weight_all : ℝ) (num_16 num_8 : ℕ), 
    average_weight_problem weight_16 weight_all num_16 num_8 :=
by
  sorry

end solve_average_weight_l2160_216082


namespace midsize_to_fullsize_ratio_l2160_216080

/-- Proves that the ratio of the mid-size model's length to the full-size mustang's length is 1:10 -/
theorem midsize_to_fullsize_ratio :
  let full_size : ℝ := 240
  let smallest_size : ℝ := 12
  let mid_size : ℝ := 2 * smallest_size
  (mid_size / full_size) = (1 / 10 : ℝ) :=
by sorry

end midsize_to_fullsize_ratio_l2160_216080


namespace nth_equation_l2160_216048

/-- The product of consecutive integers from n+1 to n+n -/
def leftSide (n : ℕ) : ℕ := (n + 1).factorial / n.factorial

/-- The product of odd numbers from 1 to 2n-1 -/
def oddProduct (n : ℕ) : ℕ := 
  Finset.prod (Finset.range n) (fun i => 2 * i + 1)

/-- The statement of the equality to be proved -/
theorem nth_equation (n : ℕ) : 
  leftSide n = 2^n * oddProduct n := by
  sorry

end nth_equation_l2160_216048


namespace some_number_value_l2160_216095

theorem some_number_value (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 49 * some_number * 25) :
  some_number = 45 := by
  sorry

end some_number_value_l2160_216095


namespace certain_value_problem_l2160_216065

theorem certain_value_problem (n x : ℝ) : n = 5 ∧ n = 5 * (n - x) → x = 4 := by
  sorry

end certain_value_problem_l2160_216065


namespace max_quarters_sasha_l2160_216068

def quarter_value : ℚ := 25 / 100
def nickel_value : ℚ := 5 / 100
def dime_value : ℚ := 10 / 100

def total_value : ℚ := 380 / 100

theorem max_quarters_sasha :
  ∃ (q : ℕ), 
    (q * quarter_value + q * nickel_value + 2 * q * dime_value ≤ total_value) ∧
    (∀ (n : ℕ), n > q → 
      n * quarter_value + n * nickel_value + 2 * n * dime_value > total_value) ∧
    q = 7 :=
by sorry

end max_quarters_sasha_l2160_216068


namespace rectangular_plot_dimensions_l2160_216097

theorem rectangular_plot_dimensions (length width : ℝ) : 
  length = 58 →
  (4 * width + 2 * length) * 26.5 = 5300 →
  length - width = 37 := by
  sorry

end rectangular_plot_dimensions_l2160_216097


namespace correct_meal_probability_l2160_216076

def number_of_people : ℕ := 12
def pasta_orders : ℕ := 5
def salad_orders : ℕ := 7

theorem correct_meal_probability : 
  let total_arrangements := Nat.factorial number_of_people
  let favorable_outcomes := 157410
  (favorable_outcomes : ℚ) / total_arrangements = 157410 / 479001600 := by
  sorry

end correct_meal_probability_l2160_216076


namespace dartboard_region_angle_l2160_216021

def circular_dartboard (total_area : ℝ) : Prop := total_area > 0

def region_probability (prob : ℝ) : Prop := prob = 1 / 8

def central_angle (angle : ℝ) : Prop := 
  0 ≤ angle ∧ angle ≤ 360

theorem dartboard_region_angle 
  (total_area : ℝ) 
  (prob : ℝ) 
  (angle : ℝ) :
  circular_dartboard total_area →
  region_probability prob →
  central_angle angle →
  prob = angle / 360 →
  angle = 45 := by sorry

end dartboard_region_angle_l2160_216021


namespace greatest_x_with_lcm_l2160_216018

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (y : ℕ), y > 0 ∧ Nat.lcm x (Nat.lcm 15 21) = 210) → x ≤ 70 :=
by sorry

end greatest_x_with_lcm_l2160_216018


namespace discount_calculation_l2160_216060

theorem discount_calculation (original_price discount_percentage : ℝ) 
  (h1 : original_price = 10)
  (h2 : discount_percentage = 10) :
  original_price * (1 - discount_percentage / 100) = 9 := by
  sorry

end discount_calculation_l2160_216060


namespace rectangleEnclosures_eq_100_l2160_216035

/-- The number of ways to choose 4 lines (2 horizontal and 2 vertical) from 5 horizontal and 5 vertical lines to enclose a rectangular region. -/
def rectangleEnclosures : ℕ :=
  let horizontalLines := 5
  let verticalLines := 5
  let horizontalChoices := Nat.choose horizontalLines 2
  let verticalChoices := Nat.choose verticalLines 2
  horizontalChoices * verticalChoices

/-- Theorem stating that the number of ways to choose 4 lines to enclose a rectangular region is 100. -/
theorem rectangleEnclosures_eq_100 : rectangleEnclosures = 100 := by
  sorry

end rectangleEnclosures_eq_100_l2160_216035


namespace addition_proof_l2160_216008

theorem addition_proof : 9873 + 3927 = 13800 := by
  sorry

end addition_proof_l2160_216008


namespace arcade_play_time_l2160_216010

def weekly_pay : ℕ := 100
def arcade_budget : ℕ := weekly_pay / 2
def food_cost : ℕ := 10
def token_budget : ℕ := arcade_budget - food_cost
def play_cost : ℕ := 8
def total_play_time : ℕ := 300

theorem arcade_play_time : 
  (token_budget / play_cost) * (total_play_time / (token_budget / play_cost)) = total_play_time :=
by sorry

end arcade_play_time_l2160_216010


namespace max_value_theorem_l2160_216072

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x*y - 1)^2 = (5*y + 2)*(y - 2)) : 
  x + 1/(2*y) ≤ -1 + (3*Real.sqrt 2)/2 := by
sorry

end max_value_theorem_l2160_216072


namespace largest_number_with_digit_sum_13_l2160_216003

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def valid_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 1 ∨ d = 2 ∨ d = 3

theorem largest_number_with_digit_sum_13 :
  ∀ n : ℕ, 
    valid_digits n → 
    digit_sum n = 13 → 
    n ≤ 222211111 :=
sorry

end largest_number_with_digit_sum_13_l2160_216003


namespace arithmetic_mean_of_first_four_prime_reciprocals_l2160_216024

-- Define the first four prime numbers
def first_four_primes : List Nat := [2, 3, 5, 7]

-- Define a function to calculate the arithmetic mean of reciprocals
def arithmetic_mean_of_reciprocals (numbers : List Nat) : ℚ :=
  let reciprocals := numbers.map (λ x => (1 : ℚ) / x)
  reciprocals.sum / numbers.length

-- Theorem statement
theorem arithmetic_mean_of_first_four_prime_reciprocals :
  arithmetic_mean_of_reciprocals first_four_primes = 247 / 840 := by
  sorry

end arithmetic_mean_of_first_four_prime_reciprocals_l2160_216024


namespace smallest_m_divisible_by_15_l2160_216014

-- Define q as the largest prime with 2011 digits
def q : ℕ := sorry

-- Axiom: q is prime
axiom q_prime : Nat.Prime q

-- Axiom: q has 2011 digits
axiom q_digits : 10^2010 ≤ q ∧ q < 10^2011

-- Define the property we want to prove
def is_divisible_by_15 (m : ℕ) : Prop :=
  ∃ k : ℤ, (q^2 - m : ℤ) = 15 * k

-- Theorem statement
theorem smallest_m_divisible_by_15 :
  ∃ m : ℕ, m > 0 ∧ is_divisible_by_15 m ∧
  ∀ n : ℕ, 0 < n ∧ n < m → ¬is_divisible_by_15 n :=
sorry

end smallest_m_divisible_by_15_l2160_216014


namespace slope_angle_45_implies_a_equals_1_l2160_216085

theorem slope_angle_45_implies_a_equals_1 (a : ℝ) : 
  (∃ (x y : ℝ), a * x + (2 * a - 3) * y = 0 ∧ 
   Real.tan (45 * π / 180) = -(a / (2 * a - 3))) → a = 1 := by
  sorry

end slope_angle_45_implies_a_equals_1_l2160_216085


namespace vector_perpendicular_l2160_216066

/-- Given two vectors a and b in R², prove that if a + b is perpendicular to a,
    then the second component of b is -7/2. -/
theorem vector_perpendicular (a b : ℝ × ℝ) (h : a = (1, 2)) (h' : b.1 = 2) :
  (a + b) • a = 0 → b.2 = -7/2 := by sorry

end vector_perpendicular_l2160_216066


namespace quadratic_equal_roots_l2160_216002

theorem quadratic_equal_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0 ∧ 
   ∀ y : ℝ, y^2 + 2*a*y + a + 2 = 0 → y = x) → 
  a = -1 ∨ a = 2 := by
sorry

end quadratic_equal_roots_l2160_216002


namespace joanie_wants_three_cups_l2160_216028

-- Define the relationship between tablespoons of kernels and cups of popcorn
def kernels_to_popcorn (tablespoons : ℕ) : ℕ := 2 * tablespoons

-- Define the amount of popcorn each person wants
def mitchell_popcorn : ℕ := 4
def miles_davis_popcorn : ℕ := 6
def cliff_popcorn : ℕ := 3

-- Define the total amount of kernels needed
def total_kernels : ℕ := 8

-- Define Joanie's popcorn amount
def joanie_popcorn : ℕ := kernels_to_popcorn total_kernels - (mitchell_popcorn + miles_davis_popcorn + cliff_popcorn)

-- Theorem statement
theorem joanie_wants_three_cups :
  joanie_popcorn = 3 := by sorry

end joanie_wants_three_cups_l2160_216028


namespace min_value_expression_l2160_216063

theorem min_value_expression (x y : ℝ) : 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) + y^2 ≥ -208.25 := by
  sorry

end min_value_expression_l2160_216063


namespace sqrt_450_equals_15_sqrt_2_l2160_216005

theorem sqrt_450_equals_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_equals_15_sqrt_2_l2160_216005


namespace cube_root_equation_solution_l2160_216023

theorem cube_root_equation_solution :
  ∃ x : ℝ, x ≠ 0 ∧ (5 - 1/x)^(1/3 : ℝ) = -6 ↔ x = 1/221 :=
by sorry

end cube_root_equation_solution_l2160_216023


namespace course_selection_theorem_l2160_216045

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def total_courses : ℕ := physical_education_courses + art_courses

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def two_course_selections : ℕ := (choose physical_education_courses 1) * (choose art_courses 1)

def three_course_selections : ℕ := 
  (choose physical_education_courses 2) * (choose art_courses 1) +
  (choose physical_education_courses 1) * (choose art_courses 2)

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_theorem : total_selections = 64 := by
  sorry

end course_selection_theorem_l2160_216045


namespace percentage_decrease_l2160_216000

theorem percentage_decrease (original : ℝ) (increase_percent : ℝ) (difference : ℝ) : 
  original = 80 →
  increase_percent = 12.5 →
  difference = 30 →
  let increased_value := original * (1 + increase_percent / 100)
  let decreased_value := original * (1 - (25 : ℝ) / 100)
  increased_value - decreased_value = difference :=
by
  sorry

#check percentage_decrease

end percentage_decrease_l2160_216000


namespace jessica_cut_two_roses_l2160_216088

/-- The number of roses Jessica cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that Jessica cut 2 roses -/
theorem jessica_cut_two_roses : roses_cut 15 17 = 2 := by
  sorry

end jessica_cut_two_roses_l2160_216088


namespace factor_implies_m_value_l2160_216078

theorem factor_implies_m_value (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - m*x - 40 = (x + 5) * k) → m = 3 := by
  sorry

end factor_implies_m_value_l2160_216078


namespace parabola_tangent_intersection_l2160_216020

/-- Parabola defined by x^2 = 2y -/
def parabola (x y : ℝ) : Prop := x^2 = 2*y

/-- Tangent line to the parabola at a given point (a, a^2/2) -/
def tangent_line (a x y : ℝ) : Prop := y - (a^2/2) = a*(x - a)

/-- Point of intersection of two lines -/
def intersection (m₁ b₁ m₂ b₂ x y : ℝ) : Prop :=
  y = m₁*x + b₁ ∧ y = m₂*x + b₂

theorem parabola_tangent_intersection :
  ∃ (x y : ℝ),
    intersection 4 (-8) (-2) (-2) x y ∧
    y = -4 :=
sorry

end parabola_tangent_intersection_l2160_216020


namespace rearrangeable_natural_segments_l2160_216017

theorem rearrangeable_natural_segments (A B : Fin 1961 → ℕ) : 
  ∃ (σ τ : Equiv.Perm (Fin 1961)) (m : ℕ),
    ∀ (i : Fin 1961), A (σ i) + B (τ i) = m + i.val :=
sorry

end rearrangeable_natural_segments_l2160_216017


namespace vector_b_value_l2160_216037

/-- Given a vector a and conditions on vector b, prove b equals (-3, 6) -/
theorem vector_b_value (a b : ℝ × ℝ) : 
  a = (1, -2) → 
  (∃ k : ℝ, k < 0 ∧ b = k • a) → 
  ‖b‖ = 3 * Real.sqrt 5 → 
  b = (-3, 6) := by
  sorry

end vector_b_value_l2160_216037


namespace quadratic_roots_expression_l2160_216084

theorem quadratic_roots_expression (m n : ℝ) : 
  m^2 + m - 2023 = 0 → n^2 + n - 2023 = 0 → m^2 + 2*m + n = 2022 := by
  sorry

end quadratic_roots_expression_l2160_216084


namespace raffle_donation_calculation_l2160_216059

theorem raffle_donation_calculation (num_tickets : ℕ) (ticket_price : ℚ) 
  (total_raised : ℚ) (fixed_donation : ℚ) :
  num_tickets = 25 →
  ticket_price = 2 →
  total_raised = 100 →
  fixed_donation = 20 →
  ∃ (equal_donation : ℚ),
    equal_donation * 2 + fixed_donation = total_raised - (num_tickets : ℚ) * ticket_price ∧
    equal_donation = 15 := by
  sorry

end raffle_donation_calculation_l2160_216059


namespace average_value_function_2x_squared_average_value_function_exponential_l2160_216067

/-- Definition of average value function on [a,b] -/
def is_average_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ : ℝ, a < x₀ ∧ x₀ < b ∧ f x₀ = (f b - f a) / (b - a)

/-- The function f(x) = 2x^2 is an average value function on [-1,1] with average value point 0 -/
theorem average_value_function_2x_squared :
  is_average_value_function (fun x => 2 * x^2) (-1) 1 ∧
  (fun x => 2 * x^2) 0 = ((fun x => 2 * x^2) 1 - (fun x => 2 * x^2) (-1)) / (1 - (-1)) :=
sorry

/-- The function g(x) = -2^(2x+1) + m⋅2^(x+1) + 1 is an average value function on [-1,1]
    if and only if m ∈ (-∞, 13/10) ∪ (17/2, +∞) -/
theorem average_value_function_exponential (m : ℝ) :
  is_average_value_function (fun x => -2^(2*x+1) + m * 2^(x+1) + 1) (-1) 1 ↔
  m < 13/10 ∨ m > 17/2 :=
sorry

end average_value_function_2x_squared_average_value_function_exponential_l2160_216067


namespace ring_arrangement_count_l2160_216070

def number_of_rings : ℕ := 10
def rings_to_arrange : ℕ := 6
def number_of_fingers : ℕ := 5

def ring_arrangements (n k : ℕ) : ℕ := (n.choose k) * k.factorial

def finger_distributions (m n : ℕ) : ℕ := (m + n - 1).choose n

theorem ring_arrangement_count :
  ring_arrangements number_of_rings rings_to_arrange * 
  finger_distributions (rings_to_arrange + number_of_fingers - 1) number_of_fingers = 31752000 :=
by sorry

end ring_arrangement_count_l2160_216070


namespace kite_profit_theorem_l2160_216032

/-- Cost price of type B kite -/
def cost_B : ℝ := 80

/-- Cost price of type A kite -/
def cost_A : ℝ := cost_B + 20

/-- Selling price of type B kite -/
def sell_B : ℝ := 120

/-- Selling price of type A kite -/
def sell_A (m : ℝ) : ℝ := 2 * (130 - m)

/-- Total number of kites -/
def total_kites : ℕ := 300

/-- Profit function -/
def profit (m : ℝ) : ℝ := (sell_A m - cost_A) * m + (sell_B - cost_B) * (total_kites - m)

/-- Theorem stating the cost prices and maximum profit -/
theorem kite_profit_theorem :
  (∀ m : ℝ, 50 ≤ m → m ≤ 150 → profit m ≤ 13000) ∧
  (20000 / cost_A = 2 * (8000 / cost_B)) ∧
  (profit 50 = 13000) := by sorry

end kite_profit_theorem_l2160_216032


namespace polynomial_simplification_l2160_216029

theorem polynomial_simplification (s : ℝ) : 
  (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 7) = -4 * s + 4 := by
  sorry

end polynomial_simplification_l2160_216029


namespace opposite_hands_count_l2160_216057

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hour : ℝ) -- Hour hand position (0 ≤ hour < 12)
  (minute : ℝ) -- Minute hand position (0 ≤ minute < 60)

/-- The speed ratio between minute and hour hands -/
def minute_hour_speed_ratio : ℝ := 12

/-- The angle difference when hands are opposite -/
def opposite_angle_diff : ℝ := 30

/-- Counts the number of times the clock hands are opposite in a 24-hour period -/
def count_opposite_hands (c : Clock) : ℕ := sorry

/-- Theorem stating that the hands are opposite 22 times in a day -/
theorem opposite_hands_count :
  ∀ c : Clock, count_opposite_hands c = 22 := by sorry

end opposite_hands_count_l2160_216057


namespace partnership_investment_time_l2160_216058

/-- A partnership problem with three partners A, B, and C --/
theorem partnership_investment_time (x : ℝ) : 
  let total_investment := x * 12 + 2 * x * 6 + 3 * x * (12 - m)
  let m := 12 - (36 * x - 24 * x) / (3 * x)
  x > 0 → m = 8 := by
  sorry

end partnership_investment_time_l2160_216058


namespace circles_tangent_line_slope_l2160_216098

-- Define the circles and their properties
def Circle := ℝ × ℝ → Prop

-- Define the conditions
def intersect_at_4_9 (C₁ C₂ : Circle) : Prop := 
  C₁ (4, 9) ∧ C₂ (4, 9)

def product_of_radii_85 (C₁ C₂ : Circle) : Prop := 
  ∃ r₁ r₂ : ℝ, r₁ * r₂ = 85

def tangent_to_y_axis (C : Circle) : Prop := 
  ∃ x : ℝ, C (0, x)

def tangent_line (n : ℝ) (C : Circle) : Prop := 
  ∃ x y : ℝ, C (x, y) ∧ y = n * x

-- Main theorem
theorem circles_tangent_line_slope (C₁ C₂ : Circle) (n : ℝ) :
  intersect_at_4_9 C₁ C₂ →
  product_of_radii_85 C₁ C₂ →
  tangent_to_y_axis C₁ →
  tangent_to_y_axis C₂ →
  tangent_line n C₁ →
  tangent_line n C₂ →
  n > 0 →
  ∃ d e f : ℕ,
    d > 0 ∧ e > 0 ∧ f > 0 ∧
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ e)) ∧
    Nat.Coprime d f ∧
    n = (d : ℝ) * Real.sqrt e / f ∧
    d + e + f = 243 :=
sorry

end circles_tangent_line_slope_l2160_216098


namespace atMostOneHead_atLeastTwoHeads_mutually_exclusive_l2160_216009

/-- Represents the outcome of throwing a coin -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the outcome of throwing 3 coins simultaneously -/
def ThreeCoinsOutcome := (CoinOutcome × CoinOutcome × CoinOutcome)

/-- Counts the number of heads in a ThreeCoinsOutcome -/
def countHeads : ThreeCoinsOutcome → Nat
  | (CoinOutcome.Heads, CoinOutcome.Heads, CoinOutcome.Heads) => 3
  | (CoinOutcome.Heads, CoinOutcome.Heads, CoinOutcome.Tails) => 2
  | (CoinOutcome.Heads, CoinOutcome.Tails, CoinOutcome.Heads) => 2
  | (CoinOutcome.Tails, CoinOutcome.Heads, CoinOutcome.Heads) => 2
  | (CoinOutcome.Heads, CoinOutcome.Tails, CoinOutcome.Tails) => 1
  | (CoinOutcome.Tails, CoinOutcome.Heads, CoinOutcome.Tails) => 1
  | (CoinOutcome.Tails, CoinOutcome.Tails, CoinOutcome.Heads) => 1
  | (CoinOutcome.Tails, CoinOutcome.Tails, CoinOutcome.Tails) => 0

/-- Event: At most one head facing up -/
def atMostOneHead (outcome : ThreeCoinsOutcome) : Prop :=
  countHeads outcome ≤ 1

/-- Event: At least two heads facing up -/
def atLeastTwoHeads (outcome : ThreeCoinsOutcome) : Prop :=
  countHeads outcome ≥ 2

/-- Theorem: The events "at most one head facing up" and "at least two heads facing up" 
    are mutually exclusive when throwing 3 coins simultaneously -/
theorem atMostOneHead_atLeastTwoHeads_mutually_exclusive :
  ∀ (outcome : ThreeCoinsOutcome), ¬(atMostOneHead outcome ∧ atLeastTwoHeads outcome) :=
by sorry

end atMostOneHead_atLeastTwoHeads_mutually_exclusive_l2160_216009


namespace modulus_of_complex_l2160_216042

/-- Given that i is the imaginary unit and z is defined as z = (2+i)/i, prove that |z| = √5 -/
theorem modulus_of_complex (i : ℂ) (z : ℂ) :
  i * i = -1 →
  z = (2 + i) / i →
  Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_complex_l2160_216042


namespace unique_solution_for_equation_l2160_216096

theorem unique_solution_for_equation : 
  ∀ (n k : ℕ), 2023 + 2^n = k^2 ↔ n = 1 ∧ k = 45 := by sorry

end unique_solution_for_equation_l2160_216096


namespace base4_odd_digits_317_l2160_216039

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

theorem base4_odd_digits_317 :
  countOddDigits (toBase4 317) = 4 := by
  sorry

end base4_odd_digits_317_l2160_216039


namespace dice_probability_relationship_l2160_216094

/-- The probability that the sum of two fair dice does not exceed 5 -/
def p₁ : ℚ := 5/18

/-- The probability that the sum of two fair dice is greater than 5 -/
def p₂ : ℚ := 11/18

/-- The probability that the sum of two fair dice is an even number -/
def p₃ : ℚ := 1/2

/-- Theorem stating the relationship between p₁, p₂, and p₃ -/
theorem dice_probability_relationship : p₁ < p₃ ∧ p₃ < p₂ := by
  sorry

end dice_probability_relationship_l2160_216094


namespace arithmetic_sequence_general_term_l2160_216061

/-- 
Given an arithmetic sequence {a_n} with sum of first n terms S_n = n^2 - 3n,
prove that the general term formula is a_n = 2n - 4.
-/
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = n^2 - 3*n) : 
  ∀ n, a n = 2*n - 4 := by
sorry

end arithmetic_sequence_general_term_l2160_216061


namespace washing_machine_price_difference_l2160_216056

def total_price : ℕ := 7060
def refrigerator_price : ℕ := 4275

theorem washing_machine_price_difference : 
  refrigerator_price - (total_price - refrigerator_price) = 1490 := by
  sorry

end washing_machine_price_difference_l2160_216056


namespace sum_of_defined_values_l2160_216033

theorem sum_of_defined_values : 
  let x : ℝ := -2 + 3
  let y : ℝ := |(-5)|
  let z : ℝ := 4 * (-1/4)
  x + y + z = 5 := by sorry

end sum_of_defined_values_l2160_216033


namespace logarithm_calculation_l2160_216077

theorem logarithm_calculation : 
  (Real.log 3 / Real.log (1/9) - (-8)^(2/3)) * (0.125^(1/3)) = -9/4 := by
  sorry

end logarithm_calculation_l2160_216077


namespace smallest_m_for_probability_l2160_216081

def probability_condition (m : ℕ) : Prop :=
  (m - 1)^4 > (3/4) * m^4

theorem smallest_m_for_probability : 
  probability_condition 17 ∧ 
  ∀ k : ℕ, k < 17 → ¬ probability_condition k :=
sorry

end smallest_m_for_probability_l2160_216081


namespace tan_x_minus_pi_sixth_l2160_216031

theorem tan_x_minus_pi_sixth (x : ℝ) 
  (h : Real.sin (π / 3 - x) = (1 / 2) * Real.cos (x - π / 2)) : 
  Real.tan (x - π / 6) = Real.sqrt 3 / 9 := by
  sorry

end tan_x_minus_pi_sixth_l2160_216031


namespace incorrect_inequality_transformation_l2160_216051

theorem incorrect_inequality_transformation (x y : ℝ) (h : x < y) : ¬(-2*x < -2*y) := by
  sorry

end incorrect_inequality_transformation_l2160_216051


namespace right_triangle_trig_l2160_216038

theorem right_triangle_trig (A B C : ℝ) (h_right : A^2 + B^2 = C^2) 
  (h_hypotenuse : C = 15) (h_leg : A = 7) :
  Real.sqrt ((C^2 - A^2) / C^2) = 4 * Real.sqrt 11 / 15 ∧ A / C = 7 / 15 := by
  sorry

end right_triangle_trig_l2160_216038


namespace adjacent_product_geometric_sequence_l2160_216090

/-- Given a geometric sequence with common ratio q, prove that the sequence formed by
    the product of adjacent terms is a geometric sequence with common ratio q² -/
theorem adjacent_product_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (hq : q ≠ 0) :
  (∀ n : ℕ, a (n + 1) = q * a n) →
  ∀ n : ℕ, (a (n + 1) * a (n + 2)) = q^2 * (a n * a (n + 1)) :=
by
  sorry

end adjacent_product_geometric_sequence_l2160_216090


namespace square_sum_l2160_216036

theorem square_sum (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = -1) : x^2 + y^2 = 11 := by
  sorry

end square_sum_l2160_216036


namespace three_digit_numbers_theorem_l2160_216055

def digits : List Nat := [3, 4, 5, 7, 9]

def isValidNumber (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧
  d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits

def validNumbers : List Nat :=
  (List.range 900).filter (fun n => n ≥ 300 ∧ isValidNumber n)

theorem three_digit_numbers_theorem :
  validNumbers.length = 60 ∧ validNumbers.sum = 37296 := by
  sorry

end three_digit_numbers_theorem_l2160_216055


namespace triangle_perimeter_l2160_216025

/-- Given a triangle with side lengths x-1, x+1, and 7, where x = 10, the perimeter of the triangle is 27. -/
theorem triangle_perimeter (x : ℝ) : x = 10 → (x - 1) + (x + 1) + 7 = 27 := by
  sorry

end triangle_perimeter_l2160_216025


namespace ellipse_properties_l2160_216099

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of the triangle AF₁F₂ -/
def triangle_AF1F2 (A F1 F2 : ℝ × ℝ) : Prop :=
  let d12 := Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2)
  let d1A := Real.sqrt ((F1.1 - A.1)^2 + (F1.2 - A.2)^2)
  let d2A := Real.sqrt ((F2.1 - A.1)^2 + (F2.2 - A.2)^2)
  d12 = 2 * Real.sqrt 2 ∧ d1A = d2A ∧ d1A^2 + d2A^2 = d12^2

/-- Main theorem -/
theorem ellipse_properties
  (a b : ℝ)
  (A F1 F2 : ℝ × ℝ)
  (h_ellipse : ∀ x y, ellipse_C x y a b)
  (h_triangle : triangle_AF1F2 A F1 F2) :
  (∀ x y, x^2 / 4 + y^2 / 2 = 1 ↔ ellipse_C x y a b) ∧
  (∀ P Q : ℝ × ℝ, P.2 = P.1 + 1 → Q.2 = Q.1 + 1 →
    ellipse_C P.1 P.2 a b → ellipse_C Q.1 Q.2 a b →
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (4/3)^2 * 5) ∧
  (¬ ∃ m : ℝ, ∀ P Q : ℝ × ℝ,
    P.2 = P.1 + m → Q.2 = Q.1 + m →
    ellipse_C P.1 P.2 a b → ellipse_C Q.1 Q.2 a b →
    (1/2) * Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) * (|m| / Real.sqrt 2) = 4/3) :=
sorry

end ellipse_properties_l2160_216099


namespace milk_water_ratio_l2160_216013

/-- 
Given a mixture of milk and water with total volume 145 liters,
if adding 58 liters of water changes the ratio of milk to water to 3:4,
then the initial ratio of milk to water was 3:2.
-/
theorem milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (milk : ℝ) 
  (water : ℝ) : 
  total_volume = 145 →
  added_water = 58 →
  milk + water = total_volume →
  milk / (water + added_water) = 3 / 4 →
  milk / water = 3 / 2 := by
sorry

end milk_water_ratio_l2160_216013


namespace triangle_area_l2160_216043

theorem triangle_area (A B C : ℝ × ℝ) : 
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let angle_B := Real.arccos ((AC^2 + BC^2 - (A.1 - B.1)^2 - (A.2 - B.2)^2) / (2 * AC * BC))
  AC = Real.sqrt 7 ∧ BC = 2 ∧ angle_B = π/3 →
  (1/2) * AC * BC * Real.sin angle_B = (3 * Real.sqrt 3) / 2 :=
by sorry

end triangle_area_l2160_216043


namespace square_difference_division_problem_solution_l2160_216075

theorem square_difference_division (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution : (144^2 - 121^2) / 23 = 265 :=
by sorry

end square_difference_division_problem_solution_l2160_216075


namespace water_drinkers_l2160_216079

theorem water_drinkers (total : ℕ) (fruit_juice : ℕ) (h1 : fruit_juice = 140) 
  (h2 : (fruit_juice : ℚ) / total = 7 / 10) : 
  (total - fruit_juice : ℚ) = 60 := by
  sorry

end water_drinkers_l2160_216079


namespace area_above_line_is_two_thirds_l2160_216049

/-- A square in a 2D plane -/
structure Square where
  bottom_left : ℝ × ℝ
  top_right : ℝ × ℝ

/-- A line in a 2D plane defined by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculate the area of a square -/
def square_area (s : Square) : ℝ :=
  let (x1, y1) := s.bottom_left
  let (x2, y2) := s.top_right
  (x2 - x1) * (y2 - y1)

/-- Calculate the area of the region above a line in a square -/
noncomputable def area_above_line (s : Square) (l : Line) : ℝ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the area above the specified line is 2/3 of the square's area -/
theorem area_above_line_is_two_thirds (s : Square) (l : Line) : 
  s.bottom_left = (2, 1) ∧ 
  s.top_right = (5, 4) ∧ 
  l.point1 = (2, 1) ∧ 
  l.point2 = (5, 3) → 
  area_above_line s l = (2/3) * square_area s := by
  sorry


end area_above_line_is_two_thirds_l2160_216049


namespace time_sum_after_advance_l2160_216054

/-- Represents time on a 12-hour digital clock -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  h_valid : hours < 12
  m_valid : minutes < 60
  s_valid : seconds < 60

/-- Calculates the time after a given number of hours, minutes, and seconds -/
def advanceTime (start : Time) (hours minutes seconds : Nat) : Time :=
  sorry

/-- Theorem: After 122 hours, 39 minutes, and 44 seconds from midnight, 
    the sum of resulting hours, minutes, and seconds is 85 -/
theorem time_sum_after_advance : 
  let midnight : Time := ⟨0, 0, 0, by simp, by simp, by simp⟩
  let result := advanceTime midnight 122 39 44
  result.hours + result.minutes + result.seconds = 85 := by
  sorry

end time_sum_after_advance_l2160_216054


namespace largest_mersenne_prime_under_500_l2160_216046

/-- A Mersenne number is of the form 2^n - 1 --/
def mersenne_number (n : ℕ) : ℕ := 2^n - 1

/-- A Mersenne prime is a Mersenne number that is prime --/
def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n, Prime n ∧ p = mersenne_number n ∧ Prime p

/-- The largest Mersenne prime less than 500 is 127 --/
theorem largest_mersenne_prime_under_500 :
  (∀ p, is_mersenne_prime p → p < 500 → p ≤ 127) ∧
  is_mersenne_prime 127 :=
sorry

end largest_mersenne_prime_under_500_l2160_216046


namespace shoe_price_calculation_l2160_216062

theorem shoe_price_calculation (initial_price : ℝ) (friday_increase : ℝ) (monday_decrease : ℝ) : 
  initial_price = 50 → 
  friday_increase = 0.20 → 
  monday_decrease = 0.15 → 
  initial_price * (1 + friday_increase) * (1 - monday_decrease) = 51 := by
sorry

end shoe_price_calculation_l2160_216062


namespace function_inequality_l2160_216034

open Set
open Function
open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, (x - 1) * (deriv f x) ≤ 0)
variable (h2 : ∀ x, f (-x) = f (2 + x))

-- Define the theorem
theorem function_inequality (x₁ x₂ : ℝ) 
  (h3 : |x₁ - 1| < |x₂ - 1|) : 
  f (2 - x₁) > f (2 - x₂) := by
  sorry

end function_inequality_l2160_216034


namespace arithmetic_geometric_mean_difference_bound_l2160_216069

theorem arithmetic_geometric_mean_difference_bound 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hab : a < b) : 
  (a + b) / 2 - Real.sqrt (a * b) < (b - a)^2 / (8 * a) := by
  sorry

end arithmetic_geometric_mean_difference_bound_l2160_216069


namespace graph_shift_l2160_216012

theorem graph_shift (x : ℝ) : (10 : ℝ) ^ (x + 3) = (10 : ℝ) ^ ((x + 4) - 1) := by
  sorry

end graph_shift_l2160_216012


namespace correct_sum_calculation_l2160_216015

theorem correct_sum_calculation (n : ℕ) (h1 : n ≥ 1000 ∧ n < 10000) 
  (h2 : n % 10 = 9) (h3 : (n - 3 + 57) = 1823) : n + 57 = 1826 :=
by
  sorry

end correct_sum_calculation_l2160_216015


namespace root_product_equals_27_l2160_216001

theorem root_product_equals_27 : (81 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end root_product_equals_27_l2160_216001


namespace range_of_expression_l2160_216027

theorem range_of_expression (x y a b : ℝ) : 
  x ≥ 1 →
  y ≥ 2 →
  x + y ≤ 4 →
  2*a + b ≥ 1 →
  3*a - b ≥ 2 →
  5*a ≤ 4 →
  (b + 2) / (a - 1) ≥ -12 ∧ (b + 2) / (a - 1) ≤ -9/2 :=
by sorry

end range_of_expression_l2160_216027


namespace common_tangents_theorem_l2160_216053

/-- Represents the relative position of two circles -/
inductive CirclePosition
  | Outside
  | TouchingExternally
  | Intersecting
  | TouchingInternally
  | Inside
  | Identical
  | OnePoint
  | TwoDistinctPoints
  | TwoCoincidingPoints

/-- Represents the number of common tangents -/
inductive TangentCount
  | Zero
  | One
  | Two
  | Three
  | Four
  | Infinite

/-- Function to determine the number of common tangents based on circle position -/
def commonTangents (position : CirclePosition) : TangentCount :=
  match position with
  | CirclePosition.Outside => TangentCount.Four
  | CirclePosition.TouchingExternally => TangentCount.Three
  | CirclePosition.Intersecting => TangentCount.Two
  | CirclePosition.TouchingInternally => TangentCount.One
  | CirclePosition.Inside => TangentCount.Zero
  | CirclePosition.Identical => TangentCount.Infinite
  | CirclePosition.OnePoint => TangentCount.Two  -- Assuming the point is outside the circle
  | CirclePosition.TwoDistinctPoints => TangentCount.One
  | CirclePosition.TwoCoincidingPoints => TangentCount.Infinite

/-- Theorem stating that the number of common tangents depends on the relative position of circles -/
theorem common_tangents_theorem (position : CirclePosition) :
  (commonTangents position = TangentCount.Zero) ∨
  (commonTangents position = TangentCount.One) ∨
  (commonTangents position = TangentCount.Two) ∨
  (commonTangents position = TangentCount.Three) ∨
  (commonTangents position = TangentCount.Four) ∨
  (commonTangents position = TangentCount.Infinite) :=
by sorry

end common_tangents_theorem_l2160_216053


namespace hospital_patient_distribution_l2160_216022

/-- Represents the number of patients each doctor takes care of in a hospital -/
def patients_per_doctor (total_patients : ℕ) (total_doctors : ℕ) : ℕ :=
  total_patients / total_doctors

/-- Theorem stating that given 400 patients and 16 doctors, each doctor takes care of 25 patients -/
theorem hospital_patient_distribution :
  patients_per_doctor 400 16 = 25 := by
  sorry

end hospital_patient_distribution_l2160_216022


namespace final_output_is_218_l2160_216052

def machine_transform (a : ℕ) : ℕ :=
  if a % 2 = 1 then a + 3 else a + 5

def repeated_transform (a : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => a
  | n + 1 => machine_transform (repeated_transform a n)

theorem final_output_is_218 :
  repeated_transform 15 51 = 218 := by
  sorry

end final_output_is_218_l2160_216052


namespace exists_roots_when_discriminant_negative_not_always_three_roots_when_p_neg_q_pos_l2160_216093

/-- Represents the equation x|x| + px + q = 0 --/
def abs_equation (x p q : ℝ) : Prop :=
  x * abs x + p * x + q = 0

/-- There exists a case where p^2 - 4q < 0 and the equation has real roots --/
theorem exists_roots_when_discriminant_negative :
  ∃ (p q : ℝ), p^2 - 4*q < 0 ∧ (∃ x : ℝ, abs_equation x p q) :=
sorry

/-- There exists a case where p < 0, q > 0, and the equation does not have exactly three real roots --/
theorem not_always_three_roots_when_p_neg_q_pos :
  ∃ (p q : ℝ), p < 0 ∧ q > 0 ∧ ¬(∃! (x y z : ℝ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    abs_equation x p q ∧ abs_equation y p q ∧ abs_equation z p q) :=
sorry

end exists_roots_when_discriminant_negative_not_always_three_roots_when_p_neg_q_pos_l2160_216093


namespace cyclist_distance_l2160_216006

/-- The distance between two points A and B for two cyclists with given conditions -/
theorem cyclist_distance (a k : ℝ) (ha : a > 0) (hk : k > 0) : ∃ (z x y : ℝ),
  z > 0 ∧ x > y ∧ y > 0 ∧
  (z + a) / (z - a) = x / y ∧
  (2 * k + 1) * z / ((2 * k - 1) * z) = x / y ∧
  z = 2 * a * k :=
by sorry

end cyclist_distance_l2160_216006


namespace decimal_to_binary_53_l2160_216091

theorem decimal_to_binary_53 : 
  (53 : ℕ) = 
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 : ℕ) :=
by sorry

end decimal_to_binary_53_l2160_216091


namespace problem_1_l2160_216089

theorem problem_1 (x y : ℝ) : (2*x + y)^2 - 8*(2*x + y) - 9 = 0 → 2*x + y = 9 ∨ 2*x + y = -1 := by
  sorry

end problem_1_l2160_216089


namespace xy_value_l2160_216073

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x^2 + y^2 = 2) (h2 : x^4 + y^4 = 15/8) :
  x * y = Real.sqrt 17 / 4 := by
sorry

end xy_value_l2160_216073


namespace student_groups_l2160_216019

theorem student_groups (group_size : ℕ) (left_early : ℕ) (remaining : ℕ) : 
  group_size = 8 → left_early = 2 → remaining = 22 → 
  (remaining + left_early) / group_size = 3 :=
by
  sorry

end student_groups_l2160_216019


namespace kants_clock_problem_l2160_216030

/-- Kant's Clock Problem -/
theorem kants_clock_problem (T_F T_2 T_S : ℝ) :
  ∃ T : ℝ, T = T_F + (T_2 - T_S) / 2 :=
by
  sorry

end kants_clock_problem_l2160_216030


namespace harolds_utilities_car_ratio_l2160_216071

/-- Harold's financial situation --/
structure HaroldFinances where
  income : ℕ
  rent : ℕ
  car_payment : ℕ
  groceries : ℕ
  retirement_savings : ℕ
  remaining : ℕ

/-- Calculate the ratio of utilities cost to car payment --/
def utilities_to_car_ratio (h : HaroldFinances) : ℚ :=
  let total_expenses := h.rent + h.car_payment + h.groceries
  let money_before_retirement := h.income - total_expenses
  let utilities := money_before_retirement - h.retirement_savings - h.remaining
  utilities / h.car_payment

/-- Theorem stating the ratio of Harold's utilities cost to his car payment --/
theorem harolds_utilities_car_ratio :
  ∃ h : HaroldFinances,
    h.income = 2500 ∧
    h.rent = 700 ∧
    h.car_payment = 300 ∧
    h.groceries = 50 ∧
    h.retirement_savings = (h.income - h.rent - h.car_payment - h.groceries) / 2 ∧
    h.remaining = 650 ∧
    utilities_to_car_ratio h = 1 / 4 :=
  sorry

end harolds_utilities_car_ratio_l2160_216071


namespace T_properties_l2160_216083

/-- T(N) is the number of arrangements of integers 1 to N satisfying specific conditions. -/
def T (N : ℕ) : ℕ := sorry

/-- v₂(n) is the 2-adic valuation of n. -/
def v₂ (n : ℕ) : ℕ := sorry

theorem T_properties :
  (T 7 = 80) ∧
  (∀ n : ℕ, n ≥ 1 → v₂ (T (2^n - 1)) = 2^n - n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → v₂ (T (2^n + 1)) = 2^n - 1) :=
by sorry

end T_properties_l2160_216083


namespace mary_baseball_cards_l2160_216004

def baseball_cards_problem (initial_cards : ℝ) (promised_cards : ℝ) (bought_cards : ℝ) : ℝ :=
  initial_cards + bought_cards - promised_cards

theorem mary_baseball_cards :
  baseball_cards_problem 18.0 26.0 40.0 = 32.0 := by
  sorry

end mary_baseball_cards_l2160_216004


namespace rationalize_denominator_l2160_216007

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) =
    (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -12 ∧
    B = 7 ∧
    C = 9 ∧
    D = 13 ∧
    E = 5 :=
by sorry

end rationalize_denominator_l2160_216007


namespace area_of_specific_figure_l2160_216040

/-- A figure composed of squares and triangles -/
structure Figure where
  num_squares : ℕ
  num_triangles : ℕ

/-- The area of a figure in square centimeters -/
def area (f : Figure) : ℝ :=
  f.num_squares + (f.num_triangles * 0.5)

/-- Theorem: The area of a specific figure is 10.5 cm² -/
theorem area_of_specific_figure :
  ∃ (f : Figure), f.num_squares = 8 ∧ f.num_triangles = 5 ∧ area f = 10.5 :=
sorry

end area_of_specific_figure_l2160_216040


namespace reflection_across_x_axis_l2160_216087

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  ⟨p.x, -p.y⟩

theorem reflection_across_x_axis :
  let P : Point := ⟨-3, 2⟩
  reflect_x P = ⟨-3, -2⟩ := by
  sorry

end reflection_across_x_axis_l2160_216087


namespace survivor_quitters_probability_l2160_216016

-- Define the total number of contestants
def total_contestants : ℕ := 20

-- Define the number of tribes
def num_tribes : ℕ := 4

-- Define the number of contestants per tribe
def contestants_per_tribe : ℕ := 5

-- Define the number of quitters
def num_quitters : ℕ := 3

-- Theorem statement
theorem survivor_quitters_probability :
  let total_ways := Nat.choose total_contestants num_quitters
  let same_tribe_ways := num_tribes * Nat.choose contestants_per_tribe num_quitters
  (same_tribe_ways : ℚ) / total_ways = 2 / 57 := by
  sorry


end survivor_quitters_probability_l2160_216016


namespace perpendicular_line_equation_l2160_216044

/-- Given a line L1 with equation 2x - 5y + 3 = 0 and a point P(2, -1),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 5x + 2y - 8 = 0 -/
theorem perpendicular_line_equation 
  (L1 : Set (ℝ × ℝ)) 
  (P : ℝ × ℝ) :
  (L1 = {(x, y) | 2 * x - 5 * y + 3 = 0}) →
  (P = (2, -1)) →
  (∃ L2 : Set (ℝ × ℝ), 
    (P ∈ L2) ∧ 
    (∀ (Q R : ℝ × ℝ), Q ∈ L1 → R ∈ L1 → Q ≠ R → 
      ∀ (S T : ℝ × ℝ), S ∈ L2 → T ∈ L2 → S ≠ T →
        ((Q.1 - R.1) * (S.1 - T.1) + (Q.2 - R.2) * (S.2 - T.2) = 0)) ∧
    (L2 = {(x, y) | 5 * x + 2 * y - 8 = 0})) :=
by sorry

end perpendicular_line_equation_l2160_216044


namespace distribute_five_balls_four_boxes_l2160_216064

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 7 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 7 := by
  sorry

end distribute_five_balls_four_boxes_l2160_216064


namespace total_age_is_23_l2160_216074

/-- Proves that the total combined age of Ryanne, Hezekiah, and Jamison is 23 years -/
theorem total_age_is_23 (hezekiah_age : ℕ) 
  (ryanne_older : ryanne_age = hezekiah_age + 7)
  (sum_ryanne_hezekiah : ryanne_age + hezekiah_age = 15)
  (jamison_twice : jamison_age = 2 * hezekiah_age) :
  hezekiah_age + ryanne_age + jamison_age = 23 :=
by
  sorry

#check total_age_is_23

end total_age_is_23_l2160_216074


namespace boat_purchase_problem_l2160_216047

theorem boat_purchase_problem (a b c d e : ℝ) : 
  a + b + c + d + e = 120 ∧
  a = (1/3) * (b + c + d + e) ∧
  b = (1/4) * (a + c + d + e) ∧
  c = (1/5) * (a + b + d + e) ∧
  d = (1/6) * (a + b + c + e) →
  e = 40 := by sorry

end boat_purchase_problem_l2160_216047


namespace output_for_input_8_l2160_216041

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 22 then
    step1 - 7
  else
    step1 + 10

theorem output_for_input_8 : function_machine 8 = 17 := by
  sorry

end output_for_input_8_l2160_216041
