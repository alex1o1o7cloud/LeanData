import Mathlib

namespace ping_pong_tournament_participants_ping_pong_tournament_solution_l2612_261211

/-- Represents a ping-pong tournament with elimination rules -/
structure PingPongTournament where
  participants : ℕ
  games_played : ℕ
  remaining_players : ℕ

/-- Conditions for our specific tournament -/
def tournament_conditions (t : PingPongTournament) : Prop :=
  t.games_played = 29 ∧ t.remaining_players = 2

/-- Theorem stating the number of participants in the tournament -/
theorem ping_pong_tournament_participants 
  (t : PingPongTournament) 
  (h : tournament_conditions t) : 
  t.participants = 16 := by
  sorry

/-- Main theorem combining the structure and the proof -/
theorem ping_pong_tournament_solution : 
  ∃ t : PingPongTournament, tournament_conditions t ∧ t.participants = 16 := by
  sorry

end ping_pong_tournament_participants_ping_pong_tournament_solution_l2612_261211


namespace number_of_boys_l2612_261270

/-- Given a school with girls and boys, prove the number of boys. -/
theorem number_of_boys (girls boys : ℕ) 
  (h1 : girls = 635)
  (h2 : boys = girls + 510) : 
  boys = 1145 := by
  sorry

end number_of_boys_l2612_261270


namespace salary_comparison_l2612_261222

/-- Represents the salary ratios of employees A, B, C, D, and E -/
def salary_ratios : Fin 5 → ℚ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5

/-- The combined salary of employees B, C, and D in rupees -/
def combined_salary_bcd : ℚ := 15000

/-- Calculates the base salary unit given the combined salary of B, C, and D -/
def base_salary : ℚ := combined_salary_bcd / (salary_ratios 1 + salary_ratios 2 + salary_ratios 3)

theorem salary_comparison :
  /- The salary of C is 200% more than that of A -/
  (salary_ratios 2 * base_salary - salary_ratios 0 * base_salary) / (salary_ratios 0 * base_salary) * 100 = 200 ∧
  /- The ratio of the salary of E to the combined salary of A and B is 5:3 -/
  (salary_ratios 4 * base_salary) / ((salary_ratios 0 + salary_ratios 1) * base_salary) = 5 / 3 := by
  sorry

end salary_comparison_l2612_261222


namespace imaginary_part_of_i_times_one_minus_i_l2612_261214

theorem imaginary_part_of_i_times_one_minus_i (i : ℂ) : 
  i * i = -1 → Complex.im (i * (1 - i)) = 1 := by
  sorry

end imaginary_part_of_i_times_one_minus_i_l2612_261214


namespace number_of_small_boxes_correct_number_of_small_boxes_l2612_261286

theorem number_of_small_boxes 
  (dolls_per_big_box : ℕ) 
  (dolls_per_small_box : ℕ) 
  (num_big_boxes : ℕ) 
  (total_dolls : ℕ) : ℕ :=
  let remaining_dolls := total_dolls - dolls_per_big_box * num_big_boxes
  remaining_dolls / dolls_per_small_box

theorem correct_number_of_small_boxes :
  number_of_small_boxes 7 4 5 71 = 9 := by
  sorry

end number_of_small_boxes_correct_number_of_small_boxes_l2612_261286


namespace quadratic_equation_range_l2612_261226

theorem quadratic_equation_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - x + a = 0) → a ≥ -1/4 := by
  sorry

end quadratic_equation_range_l2612_261226


namespace base8_plus_15_l2612_261283

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (x : ℕ) : ℕ :=
  (x / 100) * 64 + ((x / 10) % 10) * 8 + (x % 10)

/-- The problem statement --/
theorem base8_plus_15 : base8_to_base10 123 + 15 = 98 := by
  sorry

end base8_plus_15_l2612_261283


namespace gcd_8_factorial_10_factorial_l2612_261213

/-- Definition of factorial for positive integers -/
def factorial (n : ℕ+) : ℕ := (Finset.range n.val.succ).prod (fun i => i + 1)

/-- Theorem stating that the greatest common divisor of 8! and 10! is equal to 8! -/
theorem gcd_8_factorial_10_factorial :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end gcd_8_factorial_10_factorial_l2612_261213


namespace bathing_suits_for_men_l2612_261238

theorem bathing_suits_for_men (total : ℕ) (women : ℕ) (men : ℕ) : 
  total = 19766 → women = 4969 → men = total - women → men = 14797 := by
  sorry

end bathing_suits_for_men_l2612_261238


namespace pet_sitting_charge_per_night_l2612_261248

def num_cats : ℕ := 2
def num_dogs : ℕ := 3
def total_payment : ℕ := 65

theorem pet_sitting_charge_per_night :
  (total_payment : ℚ) / (num_cats + num_dogs : ℚ) = 13 := by
  sorry

end pet_sitting_charge_per_night_l2612_261248


namespace min_value_of_sum_of_fourth_powers_equality_condition_l2612_261280

theorem min_value_of_sum_of_fourth_powers (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ((a + b) / c) ^ 4 + ((b + c) / d) ^ 4 + ((c + d) / a) ^ 4 + ((d + a) / b) ^ 4 ≥ 64 :=
by sorry

theorem equality_condition (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ((a + b) / c) ^ 4 + ((b + c) / d) ^ 4 + ((c + d) / a) ^ 4 + ((d + a) / b) ^ 4 = 64 ↔ 
  a = b ∧ b = c ∧ c = d :=
by sorry

end min_value_of_sum_of_fourth_powers_equality_condition_l2612_261280


namespace free_trade_superior_l2612_261282

/-- Represents a country with its production capacity for zucchinis and cauliflower -/
structure Country where
  zucchini_capacity : ℝ
  cauliflower_capacity : ℝ

/-- Represents the market conditions and consumption preferences -/
structure MarketConditions where
  price_ratio : ℝ  -- Price of zucchini / Price of cauliflower
  consumption_ratio : ℝ  -- Ratio of zucchini to cauliflower consumption

/-- Calculates the total vegetable consumption under free trade conditions -/
def free_trade_consumption (a b : Country) (market : MarketConditions) : ℝ :=
  sorry

/-- Calculates the total vegetable consumption for the unified country without trade -/
def unified_consumption (a b : Country) : ℝ :=
  sorry

/-- Theorem stating that free trade leads to higher total consumption -/
theorem free_trade_superior (a b : Country) (market : MarketConditions) :
  free_trade_consumption a b market > unified_consumption a b :=
sorry

end free_trade_superior_l2612_261282


namespace abs_value_inequality_solution_set_l2612_261246

theorem abs_value_inequality_solution_set :
  {x : ℝ | |x| > 1} = {x : ℝ | x > 1 ∨ x < -1} := by sorry

end abs_value_inequality_solution_set_l2612_261246


namespace circle_ratio_l2612_261210

theorem circle_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  π * b^2 - π * a^2 = 4 * (π * a^2) → a / b = 1 / Real.sqrt 5 := by
sorry

end circle_ratio_l2612_261210


namespace product_of_sum_and_cube_sum_l2612_261259

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 8) 
  (h2 : a^3 + b^3 = 152) : 
  a * b = 15 := by sorry

end product_of_sum_and_cube_sum_l2612_261259


namespace circle_C_theorem_l2612_261212

/-- Definition of the circle C with parameter t -/
def circle_C (t : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*t*x - 2*t^2*y + 4*t - 4 = 0

/-- Definition of the line on which the center of C lies -/
def center_line (x y : ℝ) : Prop :=
  x - y + 2 = 0

/-- First possible equation of circle C -/
def circle_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y - 8 = 0

/-- Second possible equation of circle C -/
def circle_C2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 8*y + 4 = 0

/-- The fixed point that C passes through -/
def fixed_point : ℝ × ℝ := (2, 0)

theorem circle_C_theorem :
  ∀ t : ℝ,
  (∃ x y : ℝ, circle_C t x y ∧ center_line x y) →
  ((∀ x y : ℝ, circle_C t x y ↔ circle_C1 x y) ∨
   (∀ x y : ℝ, circle_C t x y ↔ circle_C2 x y)) ∧
  circle_C t fixed_point.1 fixed_point.2 :=
sorry

end circle_C_theorem_l2612_261212


namespace geometric_series_sum_2_to_2048_l2612_261229

def geometric_series_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

def last_term (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * r^(n - 1)

theorem geometric_series_sum_2_to_2048 :
  ∃ n : ℕ, 
    last_term 2 2 n = 2048 ∧ 
    geometric_series_sum 2 2 n = 4094 :=
by sorry

end geometric_series_sum_2_to_2048_l2612_261229


namespace part_one_part_two_l2612_261288

-- Define the function f
def f (a x : ℝ) : ℝ := (a + 1) * x^2 + 4 * a * x - 3

-- Part I
theorem part_one (a : ℝ) :
  a > 0 →
  (∃ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ > 1 ∧ x₂ < 1) ↔
  (0 < a ∧ a < 2/5) :=
sorry

-- Part II
theorem part_two (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x ≤ f a 2) ↔
  a ≥ -1/3 :=
sorry

end part_one_part_two_l2612_261288


namespace lucy_cookie_packs_l2612_261200

/-- The number of cookie packs Lucy bought at the grocery store. -/
def cookie_packs : ℕ := 28 - 16

/-- The total number of grocery packs Lucy bought. -/
def total_packs : ℕ := 28

/-- The number of noodle packs Lucy bought. -/
def noodle_packs : ℕ := 16

theorem lucy_cookie_packs : 
  cookie_packs = 12 ∧ 
  total_packs = cookie_packs + noodle_packs :=
by sorry

end lucy_cookie_packs_l2612_261200


namespace obrienHatsAfterLoss_l2612_261277

/-- The number of hats Policeman O'Brien has after losing one -/
def obrienHats (simpsonHats : ℕ) : ℕ :=
  2 * simpsonHats + 5 - 1

theorem obrienHatsAfterLoss (simpsonHats : ℕ) (h : simpsonHats = 15) : 
  obrienHats simpsonHats = 34 := by
  sorry

end obrienHatsAfterLoss_l2612_261277


namespace rental_cost_difference_l2612_261260

/-- Calculates the total cost of renting a boat for two days with a discount on the second day -/
def total_cost (daily_rental : ℝ) (hourly_rental : ℝ) (hourly_fuel : ℝ) (hours_per_day : ℝ) (discount_rate : ℝ) : ℝ :=
  let first_day := daily_rental + hourly_fuel * hours_per_day
  let second_day := (daily_rental + hourly_fuel * hours_per_day) * (1 - discount_rate)
  first_day + second_day + hourly_rental * hours_per_day * 2

/-- The difference in cost between renting a ski boat and a sailboat -/
theorem rental_cost_difference : 
  let sailboat_cost := total_cost 60 0 10 3 0.1
  let ski_boat_cost := total_cost 0 80 20 3 0.1
  ski_boat_cost - sailboat_cost = 402 := by sorry

end rental_cost_difference_l2612_261260


namespace emily_furniture_time_l2612_261279

def chairs : ℕ := 4
def tables : ℕ := 2
def time_per_piece : ℕ := 8

theorem emily_furniture_time : chairs + tables * time_per_piece = 48 := by
  sorry

end emily_furniture_time_l2612_261279


namespace square_sum_given_conditions_l2612_261220

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -6) : 
  x^2 + y^2 = 16 := by
sorry

end square_sum_given_conditions_l2612_261220


namespace mean_problem_l2612_261298

theorem mean_problem (x : ℝ) :
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 :=
by
  sorry

end mean_problem_l2612_261298


namespace distribute_five_items_four_bags_l2612_261224

/-- The number of ways to distribute n different items into k identical bags, allowing empty bags. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 36 ways to distribute 5 different items into 4 identical bags, allowing empty bags. -/
theorem distribute_five_items_four_bags : distribute 5 4 = 36 := by sorry

end distribute_five_items_four_bags_l2612_261224


namespace four_step_staircase_l2612_261236

/-- The number of ways to climb a staircase with n steps -/
def climbStairs (n : ℕ) : ℕ := sorry

/-- Theorem: There are exactly 8 ways to climb a staircase with 4 steps -/
theorem four_step_staircase : climbStairs 4 = 8 := by sorry

end four_step_staircase_l2612_261236


namespace product_difference_l2612_261256

theorem product_difference (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : ∃ k, b = 10 * k) :
  let correct_product := a * b
  let incorrect_product := (a * b) / 10
  correct_product = 10 * incorrect_product :=
by
  sorry

end product_difference_l2612_261256


namespace cylinder_cross_section_area_l2612_261262

/-- The area of a cross-section created by cutting a right circular cylinder -/
theorem cylinder_cross_section_area
  (r : ℝ) -- radius of the cylinder
  (h : ℝ) -- height of the cylinder
  (θ : ℝ) -- angle of the arc in radians
  (hr : r = 8) -- given radius
  (hh : h = 10) -- given height
  (hθ : θ = π / 2) -- 90° in radians
  : r^2 * θ * h = 320 * π :=
by sorry

end cylinder_cross_section_area_l2612_261262


namespace sharon_coffee_cost_l2612_261207

/-- Calculates the total cost of coffee pods for a vacation. -/
def coffee_cost (days : ℕ) (pods_per_day : ℕ) (pods_per_box : ℕ) (cost_per_box : ℚ) : ℚ :=
  let total_pods := days * pods_per_day
  let boxes_needed := (total_pods + pods_per_box - 1) / pods_per_box  -- Ceiling division
  boxes_needed * cost_per_box

/-- Proves that Sharon's coffee cost for her vacation is $32.00 -/
theorem sharon_coffee_cost :
  coffee_cost 40 3 30 8 = 32 :=
by
  sorry

#eval coffee_cost 40 3 30 8

end sharon_coffee_cost_l2612_261207


namespace problem_solution_l2612_261204

def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x > -3 ∧ x ≤ 3}

theorem problem_solution :
  (A = {x | -2 < x ∧ x < 3}) ∧
  (Set.compl (A ∩ B) = {x | x ≤ -2 ∨ x ≥ 3}) ∧
  ((Set.compl A) ∩ B = {x | -3 < x ∧ x ≤ -2 ∨ x = 3}) := by
  sorry

end problem_solution_l2612_261204


namespace petes_number_l2612_261234

theorem petes_number : ∃ x : ℝ, 3 * (2 * x + 15) = 141 ∧ x = 16 := by sorry

end petes_number_l2612_261234


namespace negation_of_forall_product_nonzero_l2612_261216

theorem negation_of_forall_product_nonzero (f g : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x * g x ≠ 0) ↔ (∃ x : ℝ, f x = 0 ∨ g x = 0) := by
  sorry

end negation_of_forall_product_nonzero_l2612_261216


namespace complex_equation_solution_l2612_261292

theorem complex_equation_solution (x y : ℝ) : 
  (Complex.I * (x + y) = x - 1) → (x = 1 ∧ y = -1) := by
  sorry

end complex_equation_solution_l2612_261292


namespace problem_solution_l2612_261290

theorem problem_solution (x y : ℕ+) :
  (x : ℚ) / (Nat.gcd x.val y.val : ℚ) + (y : ℚ) / (Nat.gcd x.val y.val : ℚ) = 18 ∧
  Nat.lcm x.val y.val = 975 →
  x = 75 ∧ y = 195 := by
  sorry

end problem_solution_l2612_261290


namespace imaginary_part_of_z_l2612_261275

def complex_equation (z : ℂ) : Prop :=
  z * ((1 + Complex.I) ^ 2) / 2 = 1 + 2 * Complex.I

theorem imaginary_part_of_z (z : ℂ) (h : complex_equation z) :
  z.im = -1 := by
  sorry

end imaginary_part_of_z_l2612_261275


namespace weighted_average_percentage_l2612_261272

def bag1_popped : ℕ := 60
def bag1_total : ℕ := 75

def bag2_popped : ℕ := 42
def bag2_total : ℕ := 50

def bag3_popped : ℕ := 112
def bag3_total : ℕ := 130

def bag4_popped : ℕ := 68
def bag4_total : ℕ := 90

def bag5_popped : ℕ := 82
def bag5_total : ℕ := 100

def total_kernels : ℕ := bag1_total + bag2_total + bag3_total + bag4_total + bag5_total

def weighted_sum : ℚ :=
  (bag1_popped : ℚ) / (bag1_total : ℚ) * (bag1_total : ℚ) +
  (bag2_popped : ℚ) / (bag2_total : ℚ) * (bag2_total : ℚ) +
  (bag3_popped : ℚ) / (bag3_total : ℚ) * (bag3_total : ℚ) +
  (bag4_popped : ℚ) / (bag4_total : ℚ) * (bag4_total : ℚ) +
  (bag5_popped : ℚ) / (bag5_total : ℚ) * (bag5_total : ℚ)

theorem weighted_average_percentage (ε : ℚ) (hε : ε = 1 / 10000) :
  ∃ (x : ℚ), abs (x - (weighted_sum / (total_kernels : ℚ))) < ε ∧ x = 7503 / 10000 := by
  sorry

end weighted_average_percentage_l2612_261272


namespace check_max_value_l2612_261296

theorem check_max_value (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →  -- x is a two-digit number
  (10 ≤ y ∧ y ≤ 99) →  -- y is a two-digit number
  (100 * x + y) - (100 * y + x) = 2061 →  -- difference between correct and incorrect amounts
  x ≤ 78 :=
by sorry

end check_max_value_l2612_261296


namespace sum_of_squares_of_coefficients_l2612_261247

/-- The polynomial we're working with -/
def p (x : ℝ) : ℝ := 5 * (2 * x^5 - x^3 + 2 * x^2 - 3)

/-- The coefficients of the expanded polynomial -/
def coefficients : List ℝ := [10, 0, -5, 10, 0, -15]

/-- Sum of squares of coefficients -/
def sum_of_squares (l : List ℝ) : ℝ := (l.map (λ x => x^2)).sum

theorem sum_of_squares_of_coefficients :
  sum_of_squares coefficients = 450 := by sorry

end sum_of_squares_of_coefficients_l2612_261247


namespace library_visitors_month_length_l2612_261235

theorem library_visitors_month_length :
  ∀ (s : ℕ) (d : ℕ),
    s > 0 →  -- At least one Sunday
    s + d > 0 →  -- Total days in month is positive
    150 * s + 120 * d = 125 * (s + d) →  -- Equation balancing total visitors
    s = 5 ∧ d = 25 :=
by
  sorry

end library_visitors_month_length_l2612_261235


namespace prob_adjacent_vertices_octagon_l2612_261223

/-- An octagon is a polygon with 8 vertices -/
def Octagon : Type := Unit

/-- The number of vertices in an octagon -/
def num_vertices (o : Octagon) : ℕ := 8

/-- The number of adjacent vertices for any vertex in an octagon -/
def num_adjacent (o : Octagon) : ℕ := 2

/-- The probability of choosing two distinct adjacent vertices in an octagon -/
def prob_adjacent_vertices (o : Octagon) : ℚ :=
  (num_adjacent o : ℚ) / ((num_vertices o - 1) : ℚ)

theorem prob_adjacent_vertices_octagon :
  ∀ o : Octagon, prob_adjacent_vertices o = 2 / 7 := by
  sorry

end prob_adjacent_vertices_octagon_l2612_261223


namespace greatest_common_divisor_and_digit_sum_l2612_261206

def number1 : ℕ := 1305
def number2 : ℕ := 4665
def number3 : ℕ := 6905

def difference1 : ℕ := number2 - number1
def difference2 : ℕ := number3 - number2
def difference3 : ℕ := number3 - number1

def n : ℕ := Nat.gcd difference1 (Nat.gcd difference2 difference3)

def sum_of_digits (num : ℕ) : ℕ :=
  if num < 10 then num
  else (num % 10) + sum_of_digits (num / 10)

theorem greatest_common_divisor_and_digit_sum :
  n = 1120 ∧ sum_of_digits n = 4 := by sorry

end greatest_common_divisor_and_digit_sum_l2612_261206


namespace euler_line_intersection_l2612_261218

structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

def isAcute (t : Triangle) : Prop := sorry

def isObtuse (t : Triangle) : Prop := sorry

def eulerLine (t : Triangle) : Set (ℝ × ℝ) := sorry

def sideLength (t : Triangle) (i : Fin 3) : ℝ := sorry

def largestSide (t : Triangle) : Fin 3 := sorry

def smallestSide (t : Triangle) : Fin 3 := sorry

def medianSide (t : Triangle) : Fin 3 := sorry

def intersects (line : Set (ℝ × ℝ)) (side : Fin 3) (t : Triangle) : Prop := sorry

theorem euler_line_intersection (t : Triangle) :
  (isAcute t → intersects (eulerLine t) (largestSide t) t ∧ intersects (eulerLine t) (smallestSide t) t) ∧
  (isObtuse t → intersects (eulerLine t) (largestSide t) t ∧ intersects (eulerLine t) (medianSide t) t) := by
  sorry

end euler_line_intersection_l2612_261218


namespace base6_265_equals_base10_113_l2612_261291

/-- Converts a base-6 number to base 10 --/
def base6ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 6^2 + tens * 6^1 + ones * 6^0

/-- Theorem: The base-6 number 265₆ is equal to 113 in base 10 --/
theorem base6_265_equals_base10_113 : base6ToBase10 2 6 5 = 113 := by
  sorry

end base6_265_equals_base10_113_l2612_261291


namespace intersection_point_polar_curves_l2612_261278

theorem intersection_point_polar_curves (θ : Real) (ρ : Real) 
  (h1 : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (h2 : ρ = 2 * Real.sin θ)
  (h3 : ρ * Real.cos θ = -1) :
  ∃ (ρ_intersect θ_intersect : Real),
    ρ_intersect = Real.sqrt (8 + 4 * Real.sqrt 3) ∧
    θ_intersect = 3 * Real.pi / 4 ∧
    ρ_intersect = 2 * Real.sin θ_intersect ∧
    ρ_intersect * Real.cos θ_intersect = -1 :=
by sorry

end intersection_point_polar_curves_l2612_261278


namespace equiangular_rational_sides_prime_is_regular_l2612_261221

/-- An equiangular polygon with p sides -/
structure EquiangularPolygon (p : ℕ) where
  sides : Fin p → ℚ
  is_equiangular : True  -- We assume this property is satisfied

/-- A regular polygon is an equiangular polygon with all sides equal -/
def is_regular (poly : EquiangularPolygon p) : Prop :=
  ∀ i j : Fin p, poly.sides i = poly.sides j

theorem equiangular_rational_sides_prime_is_regular
  (p : ℕ) (hp : p.Prime) (hp2 : p > 2) (poly : EquiangularPolygon p) :
  is_regular poly :=
sorry

end equiangular_rational_sides_prime_is_regular_l2612_261221


namespace quadratic_integer_root_set_characterization_l2612_261202

/-- The set of positive integers a for which the quadratic equation
    ax^2 + 2(2a-1)x + 4(a-3) = 0 has at least one integer root -/
def QuadraticIntegerRootSet : Set ℕ+ :=
  {a | ∃ x : ℤ, a * x^2 + 2*(2*a-1)*x + 4*(a-3) = 0}

/-- Theorem stating that the QuadraticIntegerRootSet contains exactly 1, 3, 6, and 10 -/
theorem quadratic_integer_root_set_characterization :
  QuadraticIntegerRootSet = {1, 3, 6, 10} := by
  sorry

#check quadratic_integer_root_set_characterization

end quadratic_integer_root_set_characterization_l2612_261202


namespace breads_after_five_thieves_l2612_261261

/-- The number of breads remaining after a thief takes their share. -/
def remaining_breads (initial : ℕ) (thief : ℕ) : ℚ :=
  if thief = 0 then initial
  else (remaining_breads initial (thief - 1) / 2) - 1/2

/-- The theorem stating that after 5 thieves, 3 breads remain from an initial count of 127. -/
theorem breads_after_five_thieves :
  ⌊remaining_breads 127 5⌋ = 3 := by sorry

end breads_after_five_thieves_l2612_261261


namespace quadratic_roots_condition_l2612_261266

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 6*x₁ + 9*k = 0 ∧ x₂^2 - 6*x₂ + 9*k = 0) → k < 1 :=
by sorry

end quadratic_roots_condition_l2612_261266


namespace seating_arrangements_l2612_261219

/-- The number of ways to arrange n people in a row --/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row with k specific people in consecutive seats --/
def consecutiveArrangements (n k : ℕ) : ℕ := 
  Nat.factorial (n - k + 1) * Nat.factorial k

/-- The number of people to be seated --/
def totalPeople : ℕ := 10

/-- The number of people who refuse to sit consecutively --/
def refusingPeople : ℕ := 4

theorem seating_arrangements :
  totalArrangements totalPeople - consecutiveArrangements totalPeople refusingPeople = 3507840 := by
  sorry

end seating_arrangements_l2612_261219


namespace problem_solution_l2612_261289

-- Define A and B as functions of a and b
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) : ℝ := 4 * a^2 + 6 * a * b + 8 * a

-- Theorem for the three parts of the problem
theorem problem_solution :
  (∀ a b : ℝ, 2 * A a b - B a b = -16 * a * b + 6 * b - 8 * a) ∧
  (2 * A (-2) 1 - B (-2) 1 = 54) ∧
  (∀ a b : ℝ, (∀ a' : ℝ, 2 * A a' b - B a' b = 2 * A a b - B a b) ↔ b = -1/2) := by
  sorry


end problem_solution_l2612_261289


namespace curve_C_properties_l2612_261274

-- Define the curve C in polar coordinates
def C (ρ θ : ℝ) : Prop :=
  ρ^2 - 4*ρ*(Real.cos θ) - 6*ρ*(Real.sin θ) + 12 = 0

-- Define the rectangular coordinates of a point on C
def point_on_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 1

-- Define the distance |PM| + |PN|
def PM_PN (x y : ℝ) : ℝ :=
  y + (x + 1)

-- Theorem statement
theorem curve_C_properties :
  (∀ ρ θ : ℝ, C ρ θ ↔ point_on_C (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  (∃ max : ℝ, max = 6 + Real.sqrt 2 ∧
    ∀ x y : ℝ, point_on_C x y → PM_PN x y ≤ max) :=
sorry

end curve_C_properties_l2612_261274


namespace sufficient_but_not_necessary_l2612_261265

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) :=
by sorry

end sufficient_but_not_necessary_l2612_261265


namespace largest_n_for_equation_l2612_261240

theorem largest_n_for_equation : 
  ∃ (x y z : ℕ+), 8^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 10 ∧ 
  ∀ (n : ℕ+), n > 8 → ¬∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 10 :=
by sorry

end largest_n_for_equation_l2612_261240


namespace std_dev_of_scaled_data_l2612_261271

-- Define the type for our data set
def DataSet := Fin 100 → ℝ

-- Define the variance of a data set
noncomputable def variance (data : DataSet) : ℝ := sorry

-- Define the standard deviation of a data set
noncomputable def std_dev (data : DataSet) : ℝ := Real.sqrt (variance data)

-- Define a function that multiplies each element of a data set by 3
def scale_by_3 (data : DataSet) : DataSet := λ i => 3 * data i

-- Our theorem
theorem std_dev_of_scaled_data (original_data : DataSet) 
  (h : variance original_data = 2) : 
  std_dev (scale_by_3 original_data) = 3 * Real.sqrt 2 := by sorry

end std_dev_of_scaled_data_l2612_261271


namespace packs_per_carton_is_five_l2612_261257

/-- The number of sticks of gum in each pack -/
def sticks_per_pack : ℕ := 3

/-- The number of cartons in each brown box -/
def cartons_per_box : ℕ := 4

/-- The total number of sticks of gum in all brown boxes -/
def total_sticks : ℕ := 480

/-- The number of brown boxes -/
def num_boxes : ℕ := 8

/-- The number of packs of gum in each carton -/
def packs_per_carton : ℕ := total_sticks / (num_boxes * cartons_per_box * sticks_per_pack)

theorem packs_per_carton_is_five : packs_per_carton = 5 := by sorry

end packs_per_carton_is_five_l2612_261257


namespace point_on_curve_with_slope_l2612_261281

def curve (x : ℝ) : ℝ := x^2 + x - 2

def tangent_slope (x : ℝ) : ℝ := 2*x + 1

theorem point_on_curve_with_slope : 
  ∃ (x y : ℝ), curve x = y ∧ tangent_slope x = 3 ∧ x = 1 ∧ y = 0 := by
  sorry

end point_on_curve_with_slope_l2612_261281


namespace solution_set_inequality_holds_l2612_261299

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1| - 1

-- Theorem 1: Solution set of f(x) ≤ x + 1
theorem solution_set (x : ℝ) : f x ≤ x + 1 ↔ 0 ≤ x ∧ x ≤ 2 := by
  sorry

-- Theorem 2: 3f(x) ≥ f(2x) for all x
theorem inequality_holds (x : ℝ) : 3 * f x ≥ f (2 * x) := by
  sorry

end solution_set_inequality_holds_l2612_261299


namespace max_ab_value_l2612_261237

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 4) :
  ab ≤ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 4 ∧ a₀*b₀ = 2 :=
sorry

end max_ab_value_l2612_261237


namespace homework_students_l2612_261250

theorem homework_students (total : ℕ) (silent_reading : ℕ) (board_games : ℕ) : 
  total = 24 →
  silent_reading = total / 2 →
  board_games = total / 3 →
  total - (silent_reading + board_games) = 4 := by
sorry

end homework_students_l2612_261250


namespace jan_water_collection_l2612_261227

theorem jan_water_collection :
  ∀ (initial_water : ℕ) 
    (car_water : ℕ) 
    (plant_water : ℕ) 
    (plates_clothes_water : ℕ),
  car_water = 7 * 2 →
  plant_water = car_water - 11 →
  plates_clothes_water = 24 →
  plates_clothes_water * 2 = initial_water - (car_water + plant_water) →
  initial_water = 65 :=
by
  sorry


end jan_water_collection_l2612_261227


namespace no_refuel_needed_l2612_261276

-- Define the parameters
def total_distance : ℕ := 156
def distance_driven : ℕ := 48
def gas_added : ℕ := 12
def fuel_consumption : ℕ := 24

-- Define the remaining distance
def remaining_distance : ℕ := total_distance - distance_driven

-- Define the range with added gas
def range_with_added_gas : ℕ := gas_added * fuel_consumption

-- Theorem statement
theorem no_refuel_needed : range_with_added_gas ≥ remaining_distance := by
  sorry

end no_refuel_needed_l2612_261276


namespace star_chain_evaluation_l2612_261249

def star (a b : ℤ) : ℤ := a * b + a + b

theorem star_chain_evaluation :
  ∃ f : ℕ → ℤ, f 1 = star 1 2 ∧ 
  (∀ n : ℕ, n ≥ 2 → f n = star (f (n-1)) (n+1)) ∧
  f 99 = Nat.factorial 101 - 1 := by sorry

end star_chain_evaluation_l2612_261249


namespace first_valid_year_is_1980_l2612_261205

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 1950 ∧ sum_of_digits year = 18

theorem first_valid_year_is_1980 :
  (∀ y : ℕ, y < 1980 → ¬(is_valid_year y)) ∧ is_valid_year 1980 := by
  sorry

end first_valid_year_is_1980_l2612_261205


namespace store_discount_l2612_261239

theorem store_discount (original_price : ℝ) (original_price_pos : 0 < original_price) : 
  let sale_price := 0.5 * original_price
  let coupon_discount := 0.2
  let promotion_discount := 0.1
  let final_price := (1 - promotion_discount) * ((1 - coupon_discount) * sale_price)
  (original_price - final_price) / original_price = 0.64 :=
by sorry

end store_discount_l2612_261239


namespace part_one_part_two_l2612_261287

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - (a - 1)) * (x - (a + 1)) < 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Part 1: Prove that when a = 2, A ∪ B = B
theorem part_one : A 2 ∪ B = B := by sorry

-- Part 2: Prove that x ∈ A ⇔ x ∈ B holds if and only if 0 ≤ a ≤ 2
theorem part_two : (∀ x, x ∈ A a ↔ x ∈ B) ↔ 0 ≤ a ∧ a ≤ 2 := by sorry

end part_one_part_two_l2612_261287


namespace max_term_at_k_max_l2612_261263

/-- The value of k that maximizes the term C_{209}^k (√5)^k in the expansion of (1+√5)^209 -/
def k_max : ℕ := 145

/-- The binomial coefficient C(n,k) -/
def binomial_coeff (n k : ℕ) : ℕ := sorry

/-- The term C_{209}^k (√5)^k in the expansion of (1+√5)^209 -/
noncomputable def term (k : ℕ) : ℝ := (binomial_coeff 209 k) * (Real.sqrt 5) ^ k

theorem max_term_at_k_max :
  ∀ k : ℕ, k ≠ k_max → term k ≤ term k_max :=
sorry

end max_term_at_k_max_l2612_261263


namespace eunji_class_size_l2612_261297

/-- The number of students who can play instrument (a) -/
def students_a : ℕ := 24

/-- The number of students who can play instrument (b) -/
def students_b : ℕ := 17

/-- The number of students who can play both instruments -/
def students_both : ℕ := 8

/-- The total number of students in Eunji's class -/
def total_students : ℕ := students_a + students_b - students_both

theorem eunji_class_size :
  total_students = 33 ∧
  students_a = 24 ∧
  students_b = 17 ∧
  students_both = 8 ∧
  total_students = students_a + students_b - students_both :=
by sorry

end eunji_class_size_l2612_261297


namespace divisors_of_8_factorial_l2612_261293

/-- The number of positive divisors of n! -/
def num_divisors_factorial (n : ℕ) : ℕ :=
  sorry

/-- 8! has 96 positive divisors -/
theorem divisors_of_8_factorial :
  num_divisors_factorial 8 = 96 := by
  sorry

end divisors_of_8_factorial_l2612_261293


namespace quadratic_coefficient_l2612_261294

/-- A quadratic function f(x) = ax² + bx + c with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The value of the quadratic function at a given x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_coefficient (f : QuadraticFunction) 
  (vertex_x : f.eval 2 = 5)
  (point : f.eval 1 = 2) : 
  f.a = -3 := by sorry

end quadratic_coefficient_l2612_261294


namespace crayon_count_is_44_l2612_261230

/-- The number of crayons in the drawer after a series of additions and removals. -/
def final_crayon_count (initial : ℝ) (benny_add : ℝ) (lucy_remove : ℝ) (sam_add : ℝ) : ℝ :=
  initial + benny_add - lucy_remove + sam_add

/-- Theorem stating that the final number of crayons is 44 given the initial count and actions. -/
theorem crayon_count_is_44 :
  final_crayon_count 25 15.5 8.75 12.25 = 44 := by
  sorry

end crayon_count_is_44_l2612_261230


namespace marbleSelectionWays_l2612_261225

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 marbles from 3 pairs of different colored marbles -/
def twoFromThreePairs : ℕ := sorry

/-- The total number of marbles -/
def totalMarbles : ℕ := 15

/-- The number of special colored marbles (red, green, blue) -/
def specialColoredMarbles : ℕ := 6

/-- The number of marbles to be chosen -/
def marblesToChoose : ℕ := 5

/-- The number of special colored marbles that must be chosen -/
def specialMarblesToChoose : ℕ := 2

theorem marbleSelectionWays : 
  twoFromThreePairs * choose (totalMarbles - specialColoredMarbles) (marblesToChoose - specialMarblesToChoose) = 1008 :=
sorry

end marbleSelectionWays_l2612_261225


namespace sum_reciprocals_l2612_261253

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) :
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^3 = 1 →
  ω ≠ 1 →
  (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω →
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 3 := by
  sorry

end sum_reciprocals_l2612_261253


namespace max_value_of_fraction_l2612_261243

theorem max_value_of_fraction (a b : ℝ) (h1 : a * b = 1) (h2 : a > b) (h3 : b ≥ 2/3) :
  (∀ x y : ℝ, x * y = 1 → x > y → y ≥ 2/3 → (x - y) / (x^2 + y^2) ≤ (a - b) / (a^2 + b^2)) →
  (a - b) / (a^2 + b^2) = 30/97 :=
by sorry

end max_value_of_fraction_l2612_261243


namespace elaine_rent_percentage_l2612_261258

/-- The percentage of Elaine's annual earnings spent on rent last year -/
def rent_percentage_last_year : ℝ := 20

/-- Elaine's earnings this year as a percentage of last year's earnings -/
def earnings_ratio : ℝ := 115

/-- The percentage of Elaine's earnings spent on rent this year -/
def rent_percentage_this_year : ℝ := 25

/-- The ratio of this year's rent expenditure to last year's rent expenditure -/
def rent_expenditure_ratio : ℝ := 143.75

theorem elaine_rent_percentage :
  rent_percentage_this_year * earnings_ratio / 100 = 
  rent_expenditure_ratio * rent_percentage_last_year / 100 :=
sorry

end elaine_rent_percentage_l2612_261258


namespace min_value_of_f_range_of_a_l2612_261268

-- Define the quadratic function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- Theorem 1
theorem min_value_of_f (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) 
  (h3 : ∀ x : ℝ, f a (Real.sin x) ≤ 5/4) :
  ∃ x : ℝ, f a x = -1 ∧ ∀ y : ℝ, f a y ≥ -1 := by sorry

-- Theorem 2
theorem range_of_a (a : ℝ) 
  (h : ∀ x : ℝ, x ∈ Set.Icc (-π/2) 0 → 
    a/2 * Real.sin x * Real.cos x + 1/2 * Real.sin x + 1/2 * Real.cos x + a/4 ≤ 1) :
  a ≤ 2 := by sorry

end min_value_of_f_range_of_a_l2612_261268


namespace only_optionA_is_valid_l2612_261285

-- Define a type for programming statements
inductive ProgramStatement
  | Print (expr : String)
  | Input
  | InputAssign (var : String) (value : Nat)
  | PrintAssign (var : String) (expr : String)

-- Define a function to check if a statement is valid
def isValidStatement (stmt : ProgramStatement) : Prop :=
  match stmt with
  | ProgramStatement.Print expr => True
  | ProgramStatement.Input => False
  | ProgramStatement.InputAssign _ _ => False
  | ProgramStatement.PrintAssign _ _ => False

-- Define the given options
def optionA := ProgramStatement.Print "4*x"
def optionB := ProgramStatement.Input
def optionC := ProgramStatement.InputAssign "B" 3
def optionD := ProgramStatement.PrintAssign "y" "2*x+1"

-- Theorem to prove
theorem only_optionA_is_valid :
  isValidStatement optionA ∧
  ¬isValidStatement optionB ∧
  ¬isValidStatement optionC ∧
  ¬isValidStatement optionD :=
sorry

end only_optionA_is_valid_l2612_261285


namespace nina_total_spending_l2612_261203

/-- The total cost of Nina's purchases --/
def total_cost (toy_price toy_quantity card_price card_quantity shirt_price shirt_quantity : ℕ) : ℕ :=
  toy_price * toy_quantity + card_price * card_quantity + shirt_price * shirt_quantity

/-- Theorem stating that Nina's total spending is $70 --/
theorem nina_total_spending :
  total_cost 10 3 5 2 6 5 = 70 := by
  sorry

end nina_total_spending_l2612_261203


namespace parallel_planes_sufficient_not_necessary_l2612_261232

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallelPlanes : Plane → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define a predicate for a line being in a plane
variable (lineInPlane : Line → Plane → Prop)

variable (α β : Plane)
variable (l : Line)

-- State the theorem
theorem parallel_planes_sufficient_not_necessary :
  (α ≠ β) →
  (lineInPlane l α) →
  (parallelPlanes α β → parallelLinePlane l β) ∧
  ∃ γ : Plane, (parallelLinePlane l γ ∧ ¬parallelPlanes α γ) :=
by sorry

end parallel_planes_sufficient_not_necessary_l2612_261232


namespace problem_statement_l2612_261254

theorem problem_statement : 
  (Real.sin (15 * π / 180)) / (Real.cos (75 * π / 180)) + 
  1 / (Real.sin (75 * π / 180))^2 - 
  (Real.tan (15 * π / 180))^2 = 2 := by sorry

end problem_statement_l2612_261254


namespace win_rate_problem_l2612_261242

/-- Represents the win rate problem for a sports team -/
theorem win_rate_problem (first_third_win_rate : ℚ) (total_matches : ℕ) :
  first_third_win_rate = 55 / 100 →
  (∃ (remaining_win_rate : ℚ),
    remaining_win_rate = 85 / 100 ∧
    first_third_win_rate * (1 / 3) + remaining_win_rate * (2 / 3) = 3 / 4) ∧
  (first_third_win_rate * (1 / 3) + 1 * (2 / 3) = 85 / 100) :=
by sorry

end win_rate_problem_l2612_261242


namespace tree_planting_solution_l2612_261284

/-- Represents the configuration of trees in a circle. -/
structure TreeCircle where
  total : ℕ
  birches : ℕ
  lindens : ℕ
  all_lindens_between_birches : Bool
  one_birch_same_neighbors : Bool

/-- The theorem stating the unique solution for the tree planting problem. -/
theorem tree_planting_solution (circle : TreeCircle) : 
  circle.total = 130 ∧ 
  circle.total = circle.birches + circle.lindens ∧ 
  circle.birches > 0 ∧ 
  circle.lindens > 0 ∧
  circle.all_lindens_between_birches = true ∧
  circle.one_birch_same_neighbors = true →
  circle.birches = 87 := by
  sorry

#check tree_planting_solution

end tree_planting_solution_l2612_261284


namespace point_on_line_l2612_261231

theorem point_on_line (m n p : ℝ) : 
  (m = n / 3 - 2 / 5) → 
  (m + p = (n + 9) / 3 - 2 / 5) → 
  p = 3 := by sorry

end point_on_line_l2612_261231


namespace exponential_decreasing_for_base_less_than_one_l2612_261201

theorem exponential_decreasing_for_base_less_than_one 
  (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : 
  a^((-0.1) : ℝ) > a^(0.1 : ℝ) := by
sorry

end exponential_decreasing_for_base_less_than_one_l2612_261201


namespace complex_fraction_equality_l2612_261273

theorem complex_fraction_equality : 
  let i : ℂ := Complex.I
  (7 + i) / (3 + 4*i) = 1 - i :=
by sorry

end complex_fraction_equality_l2612_261273


namespace non_tipping_customers_l2612_261217

/-- Calculates the number of non-tipping customers given the total number of customers,
    the tip amount per tipping customer, and the total tips earned. -/
theorem non_tipping_customers
  (total_customers : ℕ)
  (tip_amount : ℕ)
  (total_tips : ℕ)
  (h1 : total_customers > 0)
  (h2 : tip_amount > 0)
  (h3 : total_tips % tip_amount = 0)
  (h4 : total_tips / tip_amount ≤ total_customers) :
  total_customers - (total_tips / tip_amount) =
    total_customers - (total_tips / tip_amount) :=
by sorry

end non_tipping_customers_l2612_261217


namespace men_per_table_l2612_261228

theorem men_per_table (num_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ)
  (h1 : num_tables = 7)
  (h2 : women_per_table = 7)
  (h3 : total_customers = 63) :
  (total_customers - num_tables * women_per_table) / num_tables = 2 :=
by sorry

end men_per_table_l2612_261228


namespace expression_equality_l2612_261209

theorem expression_equality : 
  500 * 987 * 0.0987 * 50 = 2.5 * 987^2 := by sorry

end expression_equality_l2612_261209


namespace intersection_and_perpendicular_line_l2612_261245

/-- Given three lines in the 2D plane:
    l₁: 3x + 4y - 2 = 0
    l₂: 2x + y + 2 = 0
    l₃: 3x - 2y + 4 = 0
    Prove that the line l: 2x - 3y - 22 = 0 passes through the intersection of l₁ and l₂,
    and is perpendicular to l₃. -/
theorem intersection_and_perpendicular_line 
  (l₁ : Real → Real → Prop) 
  (l₂ : Real → Real → Prop)
  (l₃ : Real → Real → Prop)
  (l : Real → Real → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ 3*x + 4*y - 2 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 2*x + y + 2 = 0)
  (h₃ : ∀ x y, l₃ x y ↔ 3*x - 2*y + 4 = 0)
  (h : ∀ x y, l x y ↔ 2*x - 3*y - 22 = 0) :
  (∃ x y, l₁ x y ∧ l₂ x y ∧ l x y) ∧ 
  (∀ x₁ y₁ x₂ y₂, l x₁ y₁ → l x₂ y₂ → l₃ x₁ y₁ → l₃ x₂ y₂ → 
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) = 0) :=
by sorry

end intersection_and_perpendicular_line_l2612_261245


namespace oblique_square_area_l2612_261295

/-- Represents a square in an oblique projection drawing as a parallelogram -/
structure ObliqueSquare where
  parallelogram_side : ℝ

/-- The possible areas of the original square given its oblique projection -/
def possible_areas (os : ObliqueSquare) : Set ℝ :=
  {16, 64}

/-- 
Given a square represented as a parallelogram in an oblique projection drawing,
if one side of the parallelogram is 4, then the area of the original square
is either 16 or 64.
-/
theorem oblique_square_area (os : ObliqueSquare) 
  (h : os.parallelogram_side = 4) : 
  ∀ a ∈ possible_areas os, a = 16 ∨ a = 64 := by
  sorry

end oblique_square_area_l2612_261295


namespace bingo_paths_l2612_261264

/-- Represents the number of paths to spell BINGO on a grid --/
def num_bingo_paths (b_to_i : Nat) (i_to_n : Nat) (n_to_g : Nat) (g_to_o : Nat) : Nat :=
  b_to_i * i_to_n * n_to_g * g_to_o

/-- Theorem stating the number of paths to spell BINGO --/
theorem bingo_paths :
  ∀ (b_to_i i_to_n n_to_g g_to_o : Nat),
    b_to_i = 3 →
    i_to_n = 3 →
    n_to_g = 2 →
    g_to_o = 2 →
    num_bingo_paths b_to_i i_to_n n_to_g g_to_o = 36 :=
by
  sorry

#eval num_bingo_paths 3 3 2 2

end bingo_paths_l2612_261264


namespace solve_system_l2612_261267

theorem solve_system (x y : ℝ) 
  (eq1 : 3 * x - y = 7)
  (eq2 : x + 3 * y = 6) : 
  x = 2.7 := by sorry

end solve_system_l2612_261267


namespace min_distance_at_median_l2612_261233

/-- Represents a point on a line -/
structure Point :=
  (x : ℝ)

/-- Represents the distance between two points -/
def distance (p q : Point) : ℝ :=
  |p.x - q.x|

/-- Given 9 points on a line, the sum of distances from an arbitrary point to all 9 points
    is minimized when the arbitrary point coincides with the 5th point -/
theorem min_distance_at_median (p₁ p₂ p₃ p₄ p₅ p₆ p₇ p₈ p₉ : Point) 
    (h : p₁.x < p₂.x ∧ p₂.x < p₃.x ∧ p₃.x < p₄.x ∧ p₄.x < p₅.x ∧ 
         p₅.x < p₆.x ∧ p₆.x < p₇.x ∧ p₇.x < p₈.x ∧ p₈.x < p₉.x) :
  ∀ p : Point, 
    distance p p₁ + distance p p₂ + distance p p₃ + distance p p₄ + 
    distance p p₅ + distance p p₆ + distance p p₇ + distance p p₈ + distance p p₉ ≥
    distance p₅ p₁ + distance p₅ p₂ + distance p₅ p₃ + distance p₅ p₄ + 
    distance p₅ p₅ + distance p₅ p₆ + distance p₅ p₇ + distance p₅ p₈ + distance p₅ p₉ :=
by sorry

end min_distance_at_median_l2612_261233


namespace p_plus_q_equals_31_l2612_261241

theorem p_plus_q_equals_31 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-5 * x^2 + 18 * x + 27) / (x - 3)) →
  P + Q = 31 := by
sorry

end p_plus_q_equals_31_l2612_261241


namespace perpendicular_parallel_implies_perpendicular_l2612_261269

-- Define a type for lines in 3D space
structure Line3D where
  -- We don't need to specify the exact representation of a line here
  -- as we're only interested in their relationships

-- Define the relationships between lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry
def parallel (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem perpendicular_parallel_implies_perpendicular 
  (l1 l2 l3 : Line3D) 
  (h1 : perpendicular l1 l2) 
  (h2 : parallel l2 l3) : 
  perpendicular l1 l3 := by sorry

end perpendicular_parallel_implies_perpendicular_l2612_261269


namespace chessboard_coverage_l2612_261251

/-- Represents a chessboard square --/
inductive Square
| Black
| White

/-- Represents a 2x1 tile --/
structure Tile :=
  (first : Square)
  (second : Square)

/-- Represents a chessboard --/
def Chessboard := List (List Square)

/-- Creates a standard 8x8 chessboard --/
def standardChessboard : Chessboard :=
  sorry

/-- Removes two squares of different colors from the chessboard --/
def removeSquares (board : Chessboard) (pos1 pos2 : Nat × Nat) : Chessboard :=
  sorry

/-- Checks if a given tile placement is valid on the board --/
def isValidPlacement (board : Chessboard) (tile : Tile) (pos : Nat × Nat) : Bool :=
  sorry

/-- Theorem: A chessboard with two squares of different colors removed can always be covered with 2x1 tiles --/
theorem chessboard_coverage (board : Chessboard) (pos1 pos2 : Nat × Nat) :
  let removedBoard := removeSquares standardChessboard pos1 pos2
  ∃ (tilePlacements : List (Tile × (Nat × Nat))),
    (∀ (placement : Tile × (Nat × Nat)), placement ∈ tilePlacements →
      isValidPlacement removedBoard placement.fst placement.snd) ∧
    (∀ (square : Nat × Nat), square ≠ pos1 ∧ square ≠ pos2 →
      ∃ (placement : Tile × (Nat × Nat)), placement ∈ tilePlacements ∧
        (placement.snd = square ∨ (placement.snd.1 + 1, placement.snd.2) = square)) :=
  sorry


end chessboard_coverage_l2612_261251


namespace work_completion_time_l2612_261255

theorem work_completion_time (P : ℕ) (D : ℕ) : 
  (P * D = 2 * (2 * P * 3)) → D = 12 := by
  sorry

end work_completion_time_l2612_261255


namespace minimum_value_a2_plus_b2_l2612_261252

theorem minimum_value_a2_plus_b2 (a b : ℝ) : 
  (∃ k : ℕ, (20 : ℝ) = k * a^3 * b^3 ∧ Nat.choose 6 k * a^(6-k) * b^k = (20 : ℝ) * a^(6-k) * b^k) → 
  a^2 + b^2 ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀^2 + b₀^2 = 2 := by
sorry

end minimum_value_a2_plus_b2_l2612_261252


namespace only_solutions_all_negative_one_or_all_one_l2612_261208

/-- A sequence of 2016 real numbers satisfying the given equation -/
def SequenceSatisfyingEquation (x : Fin 2016 → ℝ) : Prop :=
  ∀ i : Fin 2016, x i ^ 2 + x i - 1 = x (i.succ)

/-- The theorem stating the only solutions are all -1 or all 1 -/
theorem only_solutions_all_negative_one_or_all_one
  (x : Fin 2016 → ℝ) (h : SequenceSatisfyingEquation x) :
  (∀ i, x i = -1) ∨ (∀ i, x i = 1) := by
  sorry

#check only_solutions_all_negative_one_or_all_one

end only_solutions_all_negative_one_or_all_one_l2612_261208


namespace tim_income_percentage_forty_percent_less_l2612_261215

/-- Proves that Tim's income is 40% less than Juan's income given the conditions -/
theorem tim_income_percentage (tim mary juan : ℝ) 
  (h1 : mary = 1.4 * tim) 
  (h2 : mary = 0.84 * juan) : 
  tim = 0.6 * juan := by
  sorry

/-- Proves that 40% less is equivalent to 60% of the original amount -/
theorem forty_percent_less (x y : ℝ) (h : x = 0.6 * y) : 
  x = y - 0.4 * y := by
  sorry

end tim_income_percentage_forty_percent_less_l2612_261215


namespace rice_profit_l2612_261244

/-- Calculates the profit from selling a sack of rice -/
theorem rice_profit (weight : ℝ) (cost : ℝ) (price_per_kg : ℝ) :
  weight = 50 ∧ cost = 50 ∧ price_per_kg = 1.20 →
  weight * price_per_kg - cost = 10 := by
  sorry

end rice_profit_l2612_261244
