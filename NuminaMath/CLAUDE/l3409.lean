import Mathlib

namespace thomas_money_left_l3409_340907

/-- Calculates the money left over after selling books and buying records. -/
def money_left_over (num_books : ℕ) (book_price : ℚ) (num_records : ℕ) (record_price : ℚ) : ℚ :=
  num_books * book_price - num_records * record_price

/-- Proves that Thomas has $75 left over after selling his books and buying records. -/
theorem thomas_money_left : money_left_over 200 (3/2) 75 3 = 75 := by
  sorry

end thomas_money_left_l3409_340907


namespace michaels_pets_percentage_l3409_340994

theorem michaels_pets_percentage (total_pets : ℕ) (cat_percentage : ℚ) (num_bunnies : ℕ) :
  total_pets = 36 →
  cat_percentage = 1/2 →
  num_bunnies = 9 →
  (total_pets : ℚ) * (1 - cat_percentage) - num_bunnies = (total_pets : ℚ) / 4 :=
by sorry

end michaels_pets_percentage_l3409_340994


namespace sufficient_not_necessary_condition_l3409_340978

theorem sufficient_not_necessary_condition :
  ∃ (q : ℝ → Prop), 
    (∀ x, q x → x^2 - x - 6 < 0) ∧ 
    (∃ x, x^2 - x - 6 < 0 ∧ ¬(q x)) := by
  sorry

end sufficient_not_necessary_condition_l3409_340978


namespace complex_product_real_l3409_340983

theorem complex_product_real (b : ℝ) : 
  let z₁ : ℂ := 1 + Complex.I
  let z₂ : ℂ := 2 + b * Complex.I
  (z₁ * z₂).im = 0 → b = -2 := by
sorry

end complex_product_real_l3409_340983


namespace max_value_z_l3409_340915

/-- The maximum value of z given the constraints -/
theorem max_value_z (x y : ℝ) (h1 : x - y ≤ 0) (h2 : 4 * x - y ≥ 0) (h3 : x + y ≤ 3) :
  ∃ (z : ℝ), z = x + 2 * y - 1 / x ∧ z ≤ 4 ∧ ∀ (w : ℝ), w = x + 2 * y - 1 / x → w ≤ z :=
by sorry

end max_value_z_l3409_340915


namespace apple_count_theorem_l3409_340948

def is_valid_apple_count (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ (n % 6 = 0)

theorem apple_count_theorem :
  ∀ n : ℕ, is_valid_apple_count n ↔ (n = 72 ∨ n = 78) :=
by sorry

end apple_count_theorem_l3409_340948


namespace pizza_varieties_count_l3409_340996

/-- The number of base pizza flavors -/
def base_flavors : ℕ := 4

/-- The number of extra topping options -/
def extra_toppings : ℕ := 3

/-- The number of topping combinations (including no extra toppings) -/
def topping_combinations : ℕ := 2^extra_toppings

/-- The total number of pizza varieties -/
def total_varieties : ℕ := base_flavors * topping_combinations

theorem pizza_varieties_count : total_varieties = 16 := by
  sorry

end pizza_varieties_count_l3409_340996


namespace simplify_expression_l3409_340988

theorem simplify_expression (w : ℝ) : 3*w + 6*w + 9*w + 12*w + 15*w + 18 + 24 = 45*w + 42 := by
  sorry

end simplify_expression_l3409_340988


namespace x_value_l3409_340980

theorem x_value : ∃ x : ℝ, 0.25 * x = 0.20 * 1000 - 30 ∧ x = 680 := by
  sorry

end x_value_l3409_340980


namespace cheerleaders_who_quit_l3409_340991

theorem cheerleaders_who_quit 
  (initial_football_players : Nat) 
  (initial_cheerleaders : Nat)
  (football_players_quit : Nat)
  (total_left : Nat)
  (h1 : initial_football_players = 13)
  (h2 : initial_cheerleaders = 16)
  (h3 : football_players_quit = 10)
  (h4 : total_left = 15)
  (h5 : initial_football_players - football_players_quit + initial_cheerleaders - cheerleaders_quit = total_left)
  : cheerleaders_quit = 4 :=
by
  sorry

#check cheerleaders_who_quit

end cheerleaders_who_quit_l3409_340991


namespace binary_to_base4_conversion_l3409_340952

/-- Converts a binary (base 2) number to its base 4 representation -/
def binary_to_base4 (b : ℕ) : ℕ := sorry

/-- The binary representation of the number -/
def binary_num : ℕ := 11011001

/-- The base 4 representation of the number -/
def base4_num : ℕ := 3121

theorem binary_to_base4_conversion :
  binary_to_base4 binary_num = base4_num := by sorry

end binary_to_base4_conversion_l3409_340952


namespace gcf_of_lcms_eq_210_l3409_340943

theorem gcf_of_lcms_eq_210 : Nat.gcd (Nat.lcm 10 21) (Nat.lcm 14 15) = 210 := by
  sorry

end gcf_of_lcms_eq_210_l3409_340943


namespace budget_supplies_percentage_l3409_340982

theorem budget_supplies_percentage (transportation research_development utilities equipment salaries supplies : ℝ)
  (h1 : transportation = 15)
  (h2 : research_development = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : salaries = 234 / 360 * 100)
  (h6 : transportation + research_development + utilities + equipment + salaries + supplies = 100) :
  supplies = 2 := by
  sorry

end budget_supplies_percentage_l3409_340982


namespace angle_sum_is_pi_over_two_l3409_340904

theorem angle_sum_is_pi_over_two (a b : Real) : 
  0 < a ∧ a < π/2 →
  0 < b ∧ b < π/2 →
  5 * (Real.sin a)^2 + 3 * (Real.sin b)^2 = 2 →
  4 * Real.sin (2*a) + 3 * Real.sin (2*b) = 3 →
  2*a + b = π/2 := by
  sorry

end angle_sum_is_pi_over_two_l3409_340904


namespace largest_integer_with_mean_seven_l3409_340916

theorem largest_integer_with_mean_seven (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 7 →
  ∀ x : ℕ, (x = a ∨ x = b ∨ x = c) → x ≤ 18 :=
by sorry

end largest_integer_with_mean_seven_l3409_340916


namespace race_finish_orders_l3409_340989

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of racers -/
def num_racers : ℕ := 3

/-- Theorem: The number of different possible orders for three distinct individuals 
    to finish a race without ties is equal to 6 -/
theorem race_finish_orders : permutations num_racers = 6 := by
  sorry

end race_finish_orders_l3409_340989


namespace art_fair_sales_l3409_340911

theorem art_fair_sales (total_customers : ℕ) (two_painting_buyers : ℕ) 
  (one_painting_buyers : ℕ) (four_painting_buyers : ℕ) (total_paintings_sold : ℕ) :
  total_customers = 20 →
  one_painting_buyers = 12 →
  four_painting_buyers = 4 →
  total_paintings_sold = 36 →
  two_painting_buyers + one_painting_buyers + four_painting_buyers = total_customers →
  2 * two_painting_buyers + one_painting_buyers + 4 * four_painting_buyers = total_paintings_sold →
  two_painting_buyers = 4 := by
sorry

end art_fair_sales_l3409_340911


namespace restaurant_hamburgers_l3409_340960

/-- 
Given a restaurant that:
- Made some hamburgers and 4 hot dogs
- Served 3 hamburgers
- Had 6 hamburgers left over

Prove that the initial number of hamburgers was 9.
-/
theorem restaurant_hamburgers (served : ℕ) (leftover : ℕ) : 
  served = 3 → leftover = 6 → served + leftover = 9 :=
by sorry

end restaurant_hamburgers_l3409_340960


namespace equal_shaded_unshaded_probability_l3409_340957

/-- Represents a grid of squares -/
structure Grid :=
  (size : ℕ)
  (square_size : ℝ)

/-- Represents a circle -/
structure Circle :=
  (diameter : ℝ)

/-- Represents a position on the grid -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Counts the number of favorable positions -/
def count_favorable_positions (g : Grid) (c : Circle) : ℕ := sorry

/-- Counts the total number of possible positions -/
def count_total_positions (g : Grid) : ℕ := sorry

/-- Calculates the probability of placing the circle in a favorable position -/
def probability_favorable_position (g : Grid) (c : Circle) : ℚ :=
  (count_favorable_positions g c : ℚ) / (count_total_positions g : ℚ)

theorem equal_shaded_unshaded_probability 
  (g : Grid) 
  (c : Circle) 
  (h1 : g.square_size = 2)
  (h2 : c.diameter = 3)
  (h3 : g.size = 5) :
  probability_favorable_position g c = 3/5 := by
  sorry

end equal_shaded_unshaded_probability_l3409_340957


namespace day_of_week_n_minus_one_l3409_340908

-- Define a type for days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to add days to a given day
def addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (addDays d n)

-- Define the theorem
theorem day_of_week_n_minus_one (n : Nat) :
  -- Given conditions
  (addDays DayOfWeek.Friday (150 % 7) = DayOfWeek.Friday) →
  (addDays DayOfWeek.Wednesday (210 % 7) = DayOfWeek.Wednesday) →
  -- Conclusion
  (addDays DayOfWeek.Monday 50 = DayOfWeek.Tuesday) :=
by
  sorry


end day_of_week_n_minus_one_l3409_340908


namespace linear_system_solution_ratio_l3409_340900

/-- Given a system of linear equations with parameter k:
    x + ky + 3z = 0
    3x + ky - 2z = 0
    x + 6y - 5z = 0
    which has a nontrivial solution where x, y, z are all non-zero,
    prove that yz/x^2 = 2/3 -/
theorem linear_system_solution_ratio (k : ℝ) (x y z : ℝ) :
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + k*y + 3*z = 0 →
  3*x + k*y - 2*z = 0 →
  x + 6*y - 5*z = 0 →
  y*z / (x^2) = 2/3 :=
by sorry

end linear_system_solution_ratio_l3409_340900


namespace bennys_savings_l3409_340945

/-- Proves that Benny's savings in January (and February) must be $19 given the conditions -/
theorem bennys_savings (x : ℕ) : 2 * x + 8 = 46 → x = 19 := by
  sorry

end bennys_savings_l3409_340945


namespace system_one_solution_system_two_solution_inequality_three_solution_inequality_four_solution_system_five_solution_l3409_340937

-- System 1
theorem system_one_solution (x y : ℝ) : 
  x + y = 10 ∧ 2*x + y = 16 → x = 6 ∧ y = 4 := by sorry

-- System 2
theorem system_two_solution (x y : ℝ) : 
  4*(x - y - 1) = 3*(1 - y) - 2 ∧ x/2 + y/3 = 2 → x = 2 ∧ y = 3 := by sorry

-- Inequality 3
theorem inequality_three_solution (x : ℝ) : 
  10 - 4*(x - 4) ≤ 2*(x + 1) ↔ x ≥ 4 := by sorry

-- Inequality 4
theorem inequality_four_solution (y : ℝ) : 
  (y + 1)/6 - (2*y - 5)/4 ≥ 1 ↔ y ≤ 5/4 := by sorry

-- System 5
theorem system_five_solution (x : ℝ) : 
  x - 3*(x - 2) ≥ 4 ∧ (2*x - 1)/5 ≥ (x + 1)/2 → x ≤ -7 := by sorry

end system_one_solution_system_two_solution_inequality_three_solution_inequality_four_solution_system_five_solution_l3409_340937


namespace B_current_age_l3409_340986

-- Define variables for A's and B's current ages
variable (A B : ℕ)

-- Define the conditions
def condition1 : Prop := A + 10 = 2 * (B - 10)
def condition2 : Prop := A = B + 6

-- Theorem statement
theorem B_current_age (h1 : condition1 A B) (h2 : condition2 A B) : B = 36 := by
  sorry

end B_current_age_l3409_340986


namespace soybean_oil_conversion_l3409_340918

/-- Represents the problem of determining the amount of soybeans converted to soybean oil --/
theorem soybean_oil_conversion (total_soybeans : ℝ) (total_revenue : ℝ) 
  (tofu_conversion : ℝ) (oil_conversion : ℝ) (tofu_price : ℝ) (oil_price : ℝ) :
  total_soybeans = 460 ∧ 
  total_revenue = 1800 ∧
  tofu_conversion = 3 ∧
  oil_conversion = 1 / 6 ∧
  tofu_price = 3 ∧
  oil_price = 15 →
  ∃ (x : ℝ), 
    x = 360 ∧ 
    tofu_price * tofu_conversion * (total_soybeans - x) + oil_price * oil_conversion * x = total_revenue :=
by sorry

end soybean_oil_conversion_l3409_340918


namespace complement_of_range_l3409_340940

def f (x : ℝ) : ℝ := x^2 - 2*x - 3

def domain : Set ℝ := Set.univ

def range : Set ℝ := {y | ∃ x, f x = y}

theorem complement_of_range :
  (domain \ range) = {x | x < -4} :=
sorry

end complement_of_range_l3409_340940


namespace total_tickets_l3409_340913

def tate_initial_tickets : ℕ := 32
def additional_tickets : ℕ := 2

def tate_total_tickets : ℕ := tate_initial_tickets + additional_tickets

def peyton_tickets : ℕ := tate_total_tickets / 2

theorem total_tickets : tate_total_tickets + peyton_tickets = 51 := by
  sorry

end total_tickets_l3409_340913


namespace minimum_balls_drawn_minimum_balls_drawn_correct_minimum_balls_drawn_minimal_l3409_340906

theorem minimum_balls_drawn (blue_balls red_balls : ℕ) 
  (h_blue : blue_balls = 7) (h_red : red_balls = 5) : ℕ :=
  let total_balls := blue_balls + red_balls
  let min_blue := 2
  let min_red := 1
  8

theorem minimum_balls_drawn_correct (blue_balls red_balls : ℕ) 
  (h_blue : blue_balls = 7) (h_red : red_balls = 5) :
  ∀ n : ℕ, n ≥ minimum_balls_drawn blue_balls red_balls h_blue h_red →
  (∃ b r : ℕ, b ≥ 2 ∧ r ≥ 1 ∧ b + r ≤ n ∧ b ≤ blue_balls ∧ r ≤ red_balls) :=
by
  sorry

theorem minimum_balls_drawn_minimal (blue_balls red_balls : ℕ) 
  (h_blue : blue_balls = 7) (h_red : red_balls = 5) :
  ¬∃ m : ℕ, m < minimum_balls_drawn blue_balls red_balls h_blue h_red ∧
  (∀ n : ℕ, n ≥ m →
  (∃ b r : ℕ, b ≥ 2 ∧ r ≥ 1 ∧ b + r ≤ n ∧ b ≤ blue_balls ∧ r ≤ red_balls)) :=
by
  sorry

end minimum_balls_drawn_minimum_balls_drawn_correct_minimum_balls_drawn_minimal_l3409_340906


namespace first_number_in_ratio_l3409_340954

/-- Given two positive integers with a ratio of 3:4 and an LCM of 84, prove that the first number is 21 -/
theorem first_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 3 / 4 → 
  Nat.lcm a b = 84 → 
  a = 21 := by
  sorry

end first_number_in_ratio_l3409_340954


namespace equation_solution_l3409_340987

theorem equation_solution (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x ≠ 1/16) 
  (h3 : x ≠ 1/2) 
  (h4 : x ≠ 1) : 
  (Real.log 2 / Real.log (4 * Real.sqrt x)) / (Real.log 2 / Real.log (2 * x)) + 
  (Real.log 2 / Real.log (2 * x)) * (Real.log (2 * x) / Real.log (1/2)) = 0 ↔ 
  x = 4 := by
sorry

end equation_solution_l3409_340987


namespace value_range_of_f_l3409_340932

def f (x : Int) : Int := x + 1

theorem value_range_of_f :
  {y | ∃ x ∈ ({-1, 1} : Set Int), f x = y} = {0, 2} := by sorry

end value_range_of_f_l3409_340932


namespace scientific_notation_equality_l3409_340993

theorem scientific_notation_equality : 2912000 = 2.912 * (10 ^ 6) := by
  sorry

end scientific_notation_equality_l3409_340993


namespace students_without_scholarships_l3409_340990

def total_students : ℕ := 300

def full_merit_percent : ℚ := 5 / 100
def half_merit_percent : ℚ := 10 / 100
def sports_percent : ℚ := 3 / 100
def need_based_percent : ℚ := 7 / 100

def full_merit_and_sports_percent : ℚ := 1 / 100
def half_merit_and_need_based_percent : ℚ := 2 / 100
def sports_and_need_based_percent : ℚ := 1 / 200

theorem students_without_scholarships :
  (total_students : ℚ) - 
  (((full_merit_percent + half_merit_percent + sports_percent + need_based_percent) * total_students) -
   ((full_merit_and_sports_percent + half_merit_and_need_based_percent + sports_and_need_based_percent) * total_students)) = 236 := by
  sorry

end students_without_scholarships_l3409_340990


namespace max_profit_at_max_price_l3409_340928

/-- Represents the relationship between price and sales --/
def sales_function (x : ℝ) : ℝ := -3 * x + 240

/-- Represents the profit function --/
def profit_function (x : ℝ) : ℝ := -3 * x^2 + 360 * x - 9600

/-- The cost price of apples --/
def cost_price : ℝ := 40

/-- The maximum allowed selling price --/
def max_price : ℝ := 55

/-- Theorem stating that the maximum profit is achieved at the maximum allowed price --/
theorem max_profit_at_max_price : 
  ∀ x, x ≥ cost_price → x ≤ max_price → profit_function x ≤ profit_function max_price :=
sorry

end max_profit_at_max_price_l3409_340928


namespace distribute_6_balls_4_boxes_l3409_340972

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 84 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_6_balls_4_boxes :
  distribute_balls 6 4 = 84 := by
  sorry

end distribute_6_balls_4_boxes_l3409_340972


namespace lindseys_money_is_36_l3409_340905

/-- Calculates the remaining money for Lindsey given her savings and spending. -/
def lindseys_remaining_money (sept_savings oct_savings nov_savings mom_bonus_threshold mom_bonus video_game_cost : ℕ) : ℕ :=
  let total_savings := sept_savings + oct_savings + nov_savings
  let with_bonus := total_savings + if total_savings > mom_bonus_threshold then mom_bonus else 0
  with_bonus - video_game_cost

/-- Proves that Lindsey's remaining money is $36 given her savings and spending. -/
theorem lindseys_money_is_36 :
  lindseys_remaining_money 50 37 11 75 25 87 = 36 := by
  sorry

#eval lindseys_remaining_money 50 37 11 75 25 87

end lindseys_money_is_36_l3409_340905


namespace expression_equivalence_l3409_340942

theorem expression_equivalence (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x) * ((y^3 + 2) / y) + ((x^3 - 2) / y) * ((y^3 - 2) / x) = 2 * x^2 * y^2 + 8 / (x * y) := by
  sorry

end expression_equivalence_l3409_340942


namespace correct_calculation_l3409_340925

theorem correct_calculation (x : ℚ) (h : x + 7/5 = 81/20) : (x - 7/5) * 5 = 25/4 := by
  sorry

end correct_calculation_l3409_340925


namespace largest_consecutive_sum_of_digits_divisible_l3409_340975

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def satisfies_condition (start : ℕ) (N : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ 1 → k ≤ N → (sum_of_digits (start + k - 1)) % k = 0

theorem largest_consecutive_sum_of_digits_divisible :
  ∃ start : ℕ, satisfies_condition start 21 ∧
  ∀ N : ℕ, N > 21 → ¬∃ start : ℕ, satisfies_condition start N :=
by sorry

end largest_consecutive_sum_of_digits_divisible_l3409_340975


namespace triangle_area_is_six_l3409_340926

-- Define the triangle vertices
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, 3)

-- Define the line on which C lies
def line_C (x y : ℝ) : Prop := x + y = 7

-- Define the area of the triangle
def triangle_area (C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_is_six :
  ∀ C : ℝ × ℝ, line_C C.1 C.2 → triangle_area C = 6 := by sorry

end triangle_area_is_six_l3409_340926


namespace hyperbola_vertices_distance_l3409_340976

theorem hyperbola_vertices_distance (x y : ℝ) :
  x^2 / 144 - y^2 / 64 = 1 → 
  ∃ (a : ℝ), a > 0 ∧ x^2 / a^2 - y^2 / (64 : ℝ) = 1 ∧ 2 * a = 24 :=
by
  sorry

end hyperbola_vertices_distance_l3409_340976


namespace bakery_sales_projection_l3409_340967

theorem bakery_sales_projection (white_bread_ratio : ℕ) (wheat_bread_ratio : ℕ) 
  (projected_white_bread : ℕ) (expected_wheat_bread : ℕ) : 
  white_bread_ratio = 5 → 
  wheat_bread_ratio = 8 → 
  projected_white_bread = 45 →
  expected_wheat_bread = wheat_bread_ratio * projected_white_bread / white_bread_ratio →
  expected_wheat_bread = 72 := by
  sorry

end bakery_sales_projection_l3409_340967


namespace smallest_prime_after_seven_nonprimes_l3409_340977

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_nonprime (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

def consecutive_nonprimes (start : ℕ) : Prop :=
  ∀ i : ℕ, i < 7 → is_nonprime (start + i)

theorem smallest_prime_after_seven_nonprimes :
  ∃ start : ℕ, consecutive_nonprimes start ∧ 
    is_prime 97 ∧
    (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ p > start + 6)) :=
by sorry

end smallest_prime_after_seven_nonprimes_l3409_340977


namespace local_minimum_implies_a_in_open_unit_interval_l3409_340973

/-- The function f(x) = x³ - 3ax + 1 has a local minimum in the interval (0,1) -/
def has_local_minimum (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x, x ∈ Set.Ioo 0 1 ∧ ∀ y ∈ Set.Ioo 0 1, f y ≥ f x

/-- The main theorem stating that if f(x) = x³ - 3ax + 1 has a local minimum 
    in the interval (0,1), then 0 < a < 1 -/
theorem local_minimum_implies_a_in_open_unit_interval (a : ℝ) :
  has_local_minimum (fun x => x^3 - 3*a*x + 1) a → 0 < a ∧ a < 1 :=
sorry

end local_minimum_implies_a_in_open_unit_interval_l3409_340973


namespace smallest_intersection_percentage_l3409_340984

theorem smallest_intersection_percentage (S J : ℝ) : 
  S = 90 → J = 80 → 
  ∃ (I : ℝ), I ≥ 70 ∧ I ≤ S ∧ I ≤ J ∧ 
  ∀ (I' : ℝ), I' ≤ S ∧ I' ≤ J → I' ≤ I := by
  sorry

end smallest_intersection_percentage_l3409_340984


namespace geometric_sum_456_l3409_340962

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_456 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 9 →
  (a 4 + a 5 + a 6 = 9 ∨ a 4 + a 5 + a 6 = -72) :=
by sorry

end geometric_sum_456_l3409_340962


namespace gcd_problem_l3409_340963

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, k % 2 = 1 ∧ a = k * 7771) : 
  Nat.gcd (Int.natAbs (8 * a^2 + 57 * a + 132)) (Int.natAbs (2 * a + 9)) = 9 := by
sorry

end gcd_problem_l3409_340963


namespace stating_arrangements_count_l3409_340934

/-- 
Given a positive integer n, this function returns the number of arrangements
of integers 1 to n, where each number (except the leftmost) differs by 1
from some number to its left.
-/
def countArrangements (n : ℕ) : ℕ :=
  2^(n-1)

/-- 
Theorem stating that the number of arrangements of integers 1 to n,
where each number (except the leftmost) differs by 1 from some number to its left,
is equal to 2^(n-1).
-/
theorem arrangements_count (n : ℕ) (h : n > 0) :
  countArrangements n = 2^(n-1) := by
  sorry

end stating_arrangements_count_l3409_340934


namespace duty_arrangements_eq_180_l3409_340995

/-- The number of different duty arrangements for 3 staff members over 5 days -/
def duty_arrangements (num_staff : ℕ) (num_days : ℕ) (max_days_per_staff : ℕ) : ℕ :=
  -- Number of ways to choose the person working only one day
  num_staff *
  -- Number of ways to permute the duties
  (Nat.factorial num_days / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)) *
  -- Number of ways to assign the two-day duties to the remaining two staff members
  Nat.factorial 2

/-- Theorem stating that the number of duty arrangements for the given conditions is 180 -/
theorem duty_arrangements_eq_180 :
  duty_arrangements 3 5 2 = 180 := by
  sorry

end duty_arrangements_eq_180_l3409_340995


namespace angle_ABC_measure_l3409_340919

/-- Given three angles around a point B, prove that ∠ABC = 60° -/
theorem angle_ABC_measure (ABC ABD CBD : ℝ) : 
  CBD = 90 → ABD = 30 → ABC + ABD + CBD = 180 → ABC = 60 := by
  sorry

end angle_ABC_measure_l3409_340919


namespace cubic_root_equation_solution_l3409_340961

theorem cubic_root_equation_solution (y : ℝ) : 
  (y + 16) ^ (1/3) - (y - 4) ^ (1/3) = 2 → y = 12 ∨ y = -8 := by
  sorry

end cubic_root_equation_solution_l3409_340961


namespace august_electricity_bill_l3409_340917

-- Define electricity prices for different seasons
def electricity_price (month : Nat) : Real :=
  if month ≤ 3 then 0.12
  else if month ≤ 6 then 0.10
  else if month ≤ 9 then 0.09
  else 0.11

-- Define appliance consumption rates
def oven_consumption : Real := 2.4
def ac_consumption : Real := 1.6
def fridge_consumption : Real := 0.15
def washer_consumption : Real := 0.5

-- Define appliance usage durations
def oven_usage : Nat := 25
def ac_usage : Nat := 150
def fridge_usage : Nat := 720
def washer_usage : Nat := 20

-- Define the month of August
def august : Nat := 8

-- Theorem: Coco's total electricity bill for August is $37.62
theorem august_electricity_bill :
  let price := electricity_price august
  let oven_cost := oven_consumption * oven_usage * price
  let ac_cost := ac_consumption * ac_usage * price
  let fridge_cost := fridge_consumption * fridge_usage * price
  let washer_cost := washer_consumption * washer_usage * price
  oven_cost + ac_cost + fridge_cost + washer_cost = 37.62 := by
  sorry


end august_electricity_bill_l3409_340917


namespace roots_sum_of_cubes_reciprocal_l3409_340935

theorem roots_sum_of_cubes_reciprocal (a b c : ℝ) (r s : ℂ) 
  (hr : a * r^2 + b * r - c = 0) 
  (hs : a * s^2 + b * s - c = 0) 
  (ha : a ≠ 0) 
  (hc : c ≠ 0) : 
  1 / r^3 + 1 / s^3 = (b^3 + 3*a*b*c) / c^3 := by
  sorry

end roots_sum_of_cubes_reciprocal_l3409_340935


namespace contradiction_assumption_for_greater_than_l3409_340912

theorem contradiction_assumption_for_greater_than (a b : ℝ) : 
  (¬(a > b) ↔ (a ≤ b)) := by sorry

end contradiction_assumption_for_greater_than_l3409_340912


namespace prime_sum_difference_l3409_340938

theorem prime_sum_difference (p q : Nat) : 
  Nat.Prime p → Nat.Prime q → p > 0 → q > 0 →
  p + p^2 + p^4 - q - q^2 - q^4 = 83805 →
  p = 17 ∧ q = 2 := by
  sorry

end prime_sum_difference_l3409_340938


namespace strange_clock_time_l3409_340971

/-- Represents a hand on the strange clock -/
inductive ClockHand
| A
| B
| C

/-- Represents the position of a clock hand -/
structure HandPosition where
  exactHourMark : Bool
  slightlyBeforeHourMark : Bool

/-- Represents the strange clock -/
structure StrangeClock where
  hands : ClockHand → HandPosition
  sameLength : Bool
  noNumbers : Bool
  unclearTop : Bool

/-- Determines if a given time matches the strange clock configuration -/
def matchesClockConfiguration (clock : StrangeClock) (hours : Nat) (minutes : Nat) : Prop :=
  hours = 16 ∧ minutes = 50 ∧
  clock.hands ClockHand.A = { exactHourMark := true, slightlyBeforeHourMark := false } ∧
  clock.hands ClockHand.B = { exactHourMark := false, slightlyBeforeHourMark := true } ∧
  clock.hands ClockHand.C = { exactHourMark := false, slightlyBeforeHourMark := true } ∧
  clock.sameLength ∧ clock.noNumbers ∧ clock.unclearTop

theorem strange_clock_time (clock : StrangeClock) 
  (h1 : clock.hands ClockHand.A = { exactHourMark := true, slightlyBeforeHourMark := false })
  (h2 : clock.hands ClockHand.B = { exactHourMark := false, slightlyBeforeHourMark := true })
  (h3 : clock.hands ClockHand.C = { exactHourMark := false, slightlyBeforeHourMark := true })
  (h4 : clock.sameLength)
  (h5 : clock.noNumbers)
  (h6 : clock.unclearTop) :
  ∃ (hours minutes : Nat), matchesClockConfiguration clock hours minutes :=
by
  sorry

end strange_clock_time_l3409_340971


namespace course_selection_theorem_l3409_340901

def category_A_courses : ℕ := 3
def category_B_courses : ℕ := 4
def total_courses_to_choose : ℕ := 3

/-- The number of ways to choose courses from two categories with the given constraints -/
def number_of_ways_to_choose : ℕ :=
  (Nat.choose category_A_courses 1 * Nat.choose category_B_courses 2) +
  (Nat.choose category_A_courses 2 * Nat.choose category_B_courses 1)

theorem course_selection_theorem :
  number_of_ways_to_choose = 30 :=
by sorry

end course_selection_theorem_l3409_340901


namespace reciprocal_of_sum_l3409_340939

theorem reciprocal_of_sum : (1 / (1/3 + 1/4) : ℚ) = 12/7 := by
  sorry

end reciprocal_of_sum_l3409_340939


namespace inequality_preservation_l3409_340914

theorem inequality_preservation (x y : ℝ) (h : x > y) : x/2 > y/2 := by
  sorry

end inequality_preservation_l3409_340914


namespace last_number_is_30_l3409_340923

theorem last_number_is_30 (numbers : Fin 8 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5 + numbers 6 + numbers 7) / 8 = 25)
  (h2 : (numbers 0 + numbers 1) / 2 = 20)
  (h3 : (numbers 2 + numbers 3 + numbers 4) / 3 = 26)
  (h4 : numbers 5 = numbers 6 - 4)
  (h5 : numbers 5 = numbers 7 - 6) :
  numbers 7 = 30 := by
sorry

end last_number_is_30_l3409_340923


namespace sunway_taihulight_performance_l3409_340930

theorem sunway_taihulight_performance :
  (12.5 * (10^12 : ℝ)) = (1.25 * (10^13 : ℝ)) := by
  sorry

end sunway_taihulight_performance_l3409_340930


namespace quadratic_roots_and_triangle_l3409_340951

theorem quadratic_roots_and_triangle (α β : ℝ) (p k : ℝ) : 
  α^2 - 10*α + 20 = 0 →
  β^2 - 10*β + 20 = 0 →
  p = α^2 + β^2 →
  k * Real.sqrt 3 = (p^2 / 36) * Real.sqrt 3 →
  p = 60 ∧ k = p^2 / 36 := by
sorry

end quadratic_roots_and_triangle_l3409_340951


namespace middle_term_coefficient_l3409_340936

/-- Given a natural number n, returns the binomial expansion of (1-2x)^n -/
def binomialExpansion (n : ℕ) : List ℤ := sorry

/-- Returns the sum of coefficients of even-numbered terms in a list -/
def sumEvenTerms (coeffs : List ℤ) : ℤ := sorry

/-- Returns the middle coefficient of a list -/
def middleCoefficient (coeffs : List ℤ) : ℤ := sorry

theorem middle_term_coefficient (n : ℕ) :
  sumEvenTerms (binomialExpansion n) = 128 →
  middleCoefficient (binomialExpansion n) = 1120 := by
  sorry

end middle_term_coefficient_l3409_340936


namespace sin_inequality_l3409_340969

theorem sin_inequality (α : Real) (h : 0 < α ∧ α < π / 2) :
  Real.sin (2 * α) + 2 / Real.sin (2 * α) ≥ 3 := by
  sorry

end sin_inequality_l3409_340969


namespace characterization_of_good_numbers_l3409_340979

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → (d + 1) ∣ (n + 1)

theorem characterization_of_good_numbers (n : ℕ) :
  is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1) :=
sorry

end characterization_of_good_numbers_l3409_340979


namespace rearranged_balls_theorem_l3409_340974

/-- Represents a ball with its initial and final pile sizes -/
structure Ball where
  initialPileSize : ℕ+
  finalPileSize : ℕ+

/-- The problem statement -/
theorem rearranged_balls_theorem (n k : ℕ+) (balls : Finset Ball) 
    (h_initial_piles : (balls.sum fun b => (1 : ℚ) / b.initialPileSize) = n)
    (h_final_piles : (balls.sum fun b => (1 : ℚ) / b.finalPileSize) = n + k) :
    ∃ (subset : Finset Ball), subset.card = k + 1 ∧ 
    ∀ b ∈ subset, b.initialPileSize > b.finalPileSize :=
  sorry

end rearranged_balls_theorem_l3409_340974


namespace five_circles_arrangement_exists_four_circles_arrangement_not_exists_l3409_340922

-- Define a circle on a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a ray starting from a point
structure Ray where
  start : ℝ × ℝ
  direction : ℝ × ℝ
  direction_nonzero : direction ≠ (0, 0)

-- Function to check if a ray intersects a circle
def ray_intersects_circle (r : Ray) (c : Circle) : Prop :=
  sorry

-- Function to check if a ray intersects at least two circles from a list
def ray_intersects_at_least_two (r : Ray) (circles : List Circle) : Prop :=
  sorry

-- Function to check if a circle covers a point
def circle_covers_point (c : Circle) (p : ℝ × ℝ) : Prop :=
  sorry

-- Theorem for part (a)
theorem five_circles_arrangement_exists :
  ∃ (circles : List Circle), circles.length = 5 ∧
  ∀ (r : Ray), r.start = (0, 0) → ray_intersects_at_least_two r circles :=
sorry

-- Theorem for part (b)
theorem four_circles_arrangement_not_exists :
  ¬ ∃ (circles : List Circle), circles.length = 4 ∧
  (∀ c ∈ circles, ¬ circle_covers_point c (0, 0)) ∧
  (∀ (r : Ray), r.start = (0, 0) → ray_intersects_at_least_two r circles) :=
sorry

end five_circles_arrangement_exists_four_circles_arrangement_not_exists_l3409_340922


namespace library_books_difference_l3409_340920

theorem library_books_difference (initial_books borrowed_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : borrowed_books = 18) :
  initial_books - borrowed_books = 57 := by
sorry

end library_books_difference_l3409_340920


namespace geometric_sequence_common_ratio_l3409_340968

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n,
    if a_2 * a_3 = 2 * a_1 and 5/4 is the arithmetic mean of a_4 and 2 * a_7,
    then q = 1/2 -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q))
  (h3 : a 2 * a 3 = 2 * a 1)
  (h4 : (a 4 + 2 * a 7) / 2 = 5 / 4)
  : q = 1 / 2 := by
  sorry

end geometric_sequence_common_ratio_l3409_340968


namespace lyras_remaining_budget_l3409_340944

/-- Calculates the remaining budget after food purchases -/
def remaining_budget (weekly_budget : ℕ) (chicken_cost : ℕ) (beef_price_per_pound : ℕ) (beef_pounds : ℕ) : ℕ :=
  weekly_budget - (chicken_cost + beef_price_per_pound * beef_pounds)

/-- Proves that Lyra's remaining budget is $53 -/
theorem lyras_remaining_budget :
  remaining_budget 80 12 3 5 = 53 := by
  sorry

end lyras_remaining_budget_l3409_340944


namespace modulus_of_squared_complex_l3409_340929

theorem modulus_of_squared_complex (z : ℂ) (h : z^2 = 15 - 20*I) : Complex.abs z = 5 := by
  sorry

end modulus_of_squared_complex_l3409_340929


namespace unique_satisfying_function_l3409_340921

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- The condition that the function must satisfy for all positive integers x and y -/
def SatisfiesCondition (f : PositiveIntFunction) : Prop :=
  ∀ x y : ℕ+, ∃ k : ℕ, (x : ℤ)^2 - (y : ℤ)^2 + 2*(y : ℤ)*((f x : ℤ) + (f y : ℤ)) = (k : ℤ)^2

/-- The theorem stating that the identity function is the only function satisfying the condition -/
theorem unique_satisfying_function :
  ∃! f : PositiveIntFunction, SatisfiesCondition f ∧ ∀ n : ℕ+, f n = n :=
sorry

end unique_satisfying_function_l3409_340921


namespace difference_in_circumferences_l3409_340985

/-- The difference in circumferences of two concentric circular paths -/
theorem difference_in_circumferences 
  (inner_radius : ℝ) 
  (width_difference : ℝ) 
  (h1 : inner_radius = 25) 
  (h2 : width_difference = 15) : 
  2 * π * (inner_radius + width_difference) - 2 * π * inner_radius = 30 * π := by
sorry

end difference_in_circumferences_l3409_340985


namespace count_three_painted_faces_4x4x4_l3409_340941

/-- Represents a cube with painted faces -/
structure PaintedCube :=
  (size : ℕ)
  (painted_faces : Fin 6 → Bool)

/-- Counts the number of subcubes with at least three painted faces -/
def count_subcubes_with_three_painted_faces (cube : PaintedCube) : ℕ := sorry

/-- Theorem: In a 4x4x4 cube with all outer faces painted, 
    the number of 1x1x1 subcubes with at least three painted faces is 8 -/
theorem count_three_painted_faces_4x4x4 : 
  ∀ (cube : PaintedCube), 
  cube.size = 4 → 
  (∀ (f : Fin 6), cube.painted_faces f = true) →
  count_subcubes_with_three_painted_faces cube = 8 := by sorry

end count_three_painted_faces_4x4x4_l3409_340941


namespace bernoulli_inequality_l3409_340903

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (h : x ≥ -1) :
  (1 + x)^n ≥ 1 + n*x := by
  sorry

end bernoulli_inequality_l3409_340903


namespace contingency_fund_allocation_l3409_340992

def total_donation : ℚ := 240

def community_pantry_ratio : ℚ := 1/3
def local_crisis_ratio : ℚ := 1/2
def livelihood_ratio : ℚ := 1/4

def community_pantry : ℚ := total_donation * community_pantry_ratio
def local_crisis : ℚ := total_donation * local_crisis_ratio

def remaining_after_main : ℚ := total_donation - (community_pantry + local_crisis)
def livelihood : ℚ := remaining_after_main * livelihood_ratio

def contingency : ℚ := remaining_after_main - livelihood

theorem contingency_fund_allocation :
  contingency = 30 := by sorry

end contingency_fund_allocation_l3409_340992


namespace difference_of_fractions_l3409_340910

theorem difference_of_fractions (n : ℕ) : 
  (n / 10 : ℚ) - (n / 1000 : ℚ) = 693 ↔ n = 7000 := by sorry

end difference_of_fractions_l3409_340910


namespace basketball_success_rate_increase_success_rate_increase_approx_17_l3409_340931

/-- Calculates the increase in success rate percentage for basketball free throws -/
theorem basketball_success_rate_increase 
  (initial_success : Nat) 
  (initial_attempts : Nat) 
  (subsequent_success_rate : Rat) 
  (subsequent_attempts : Nat) : ℝ :=
  let total_success := initial_success + ⌊subsequent_success_rate * subsequent_attempts⌋
  let total_attempts := initial_attempts + subsequent_attempts
  let new_rate := (total_success : ℝ) / total_attempts
  let initial_rate := (initial_success : ℝ) / initial_attempts
  let increase := (new_rate - initial_rate) * 100
  ⌊increase + 0.5⌋

/-- The increase in success rate percentage is approximately 17 percentage points -/
theorem success_rate_increase_approx_17 :
  ⌊basketball_success_rate_increase 7 15 (3/4) 18 + 0.5⌋ = 17 := by
  sorry

end basketball_success_rate_increase_success_rate_increase_approx_17_l3409_340931


namespace triangle_problem_l3409_340999

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : (Real.sin t.B) / (2 * Real.sin t.A - Real.sin t.C) = 1 / (2 * Real.cos t.C))
  (h2 : t.a = 1)
  (h3 : t.b = Real.sqrt 7) :
  t.B = π / 3 ∧ (1/2 * t.a * t.c * Real.sin t.B) = (3 * Real.sqrt 3) / 4 := by
  sorry

end triangle_problem_l3409_340999


namespace professor_coffee_meeting_l3409_340946

theorem professor_coffee_meeting (n p q r : ℕ) : 
  (∀ (x : ℕ), x > 1 → x.Prime → r % (x ^ 2) ≠ 0) →  -- r is not divisible by the square of any prime
  (n : ℝ) = p - q * Real.sqrt r →  -- n = p - q√r
  (((120 : ℝ) - n) ^ 2 / 14400 = 1 / 2) →  -- probability of meeting is 50%
  p + q + r = 182 := by
  sorry

end professor_coffee_meeting_l3409_340946


namespace six_balls_four_boxes_l3409_340964

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 157 := by sorry

end six_balls_four_boxes_l3409_340964


namespace complex_fraction_equality_l3409_340981

theorem complex_fraction_equality : (2 - I) / (1 + 2*I) = -I := by
  sorry

end complex_fraction_equality_l3409_340981


namespace sin_cos_graph_shift_l3409_340959

theorem sin_cos_graph_shift (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x - Real.pi / 4)
  let g : ℝ → ℝ := λ x => Real.cos (2 * x)
  ∃ (shift : ℝ), shift = 3 * Real.pi / 8 ∧
    f x = g (x - shift) :=
by sorry

end sin_cos_graph_shift_l3409_340959


namespace divides_n_squared_plus_one_l3409_340902

theorem divides_n_squared_plus_one (n : ℕ) : 
  (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 := by
  sorry

end divides_n_squared_plus_one_l3409_340902


namespace line_param_solution_l3409_340933

/-- Represents a 2D vector -/
structure Vec2 where
  x : ℝ
  y : ℝ

/-- Represents the parameterization of a line -/
def lineParam (s h : ℝ) (t : ℝ) : Vec2 :=
  { x := s + 5 * t
    y := -2 + h * t }

/-- The equation of the line y = 3x - 11 -/
def lineEq (v : Vec2) : Prop :=
  v.y = 3 * v.x - 11

theorem line_param_solution :
  ∃ (s h : ℝ), ∀ (t : ℝ), lineEq (lineParam s h t) ∧ s = 3 ∧ h = 15 := by
  sorry

end line_param_solution_l3409_340933


namespace perfect_square_trinomial_l3409_340998

/-- A trinomial x^2 - kx + 9 is a perfect square if and only if k = 6 or k = -6 -/
theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x, x^2 - k*x + 9 = (a*x + b)^2) ↔ (k = 6 ∨ k = -6) :=
by sorry

end perfect_square_trinomial_l3409_340998


namespace simplify_expression_l3409_340927

theorem simplify_expression (w : ℝ) : 4*w + 6*w + 8*w + 10*w + 12*w + 24 = 40*w + 24 := by
  sorry

end simplify_expression_l3409_340927


namespace rectangular_prism_volume_l3409_340955

/-- Given a rectangular prism with side face areas of √2, √3, and √6, its volume is √6 -/
theorem rectangular_prism_volume (a b c : ℝ) 
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : a * c = Real.sqrt 6) : 
  a * b * c = Real.sqrt 6 := by
sorry

end rectangular_prism_volume_l3409_340955


namespace complex_sum_equals_polar_form_l3409_340953

theorem complex_sum_equals_polar_form : 
  5 * Complex.exp (Complex.I * (3 * Real.pi / 7)) + 
  15 * Complex.exp (Complex.I * (23 * Real.pi / 14)) = 
  20 * Real.sqrt ((3 + Real.cos (13 * Real.pi / 14)) / 4) * 
  Complex.exp (Complex.I * (29 * Real.pi / 28)) := by sorry

end complex_sum_equals_polar_form_l3409_340953


namespace least_subtraction_for_divisibility_problem_solution_l3409_340956

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (k : ℕ), k < 8 ∧ (964807 - k) % 8 = 0 ∧ ∀ (m : ℕ), m < k → (964807 - m) % 8 ≠ 0 ∧ k = 7 :=
by
  sorry

end least_subtraction_for_divisibility_problem_solution_l3409_340956


namespace sufficient_not_necessary_condition_l3409_340947

theorem sufficient_not_necessary_condition : 
  (∃ x : ℝ, x ≠ 1 ∧ x^2 - 1 = 0) ∧ 
  (∀ x : ℝ, x = 1 → x^2 - 1 = 0) := by
  sorry

end sufficient_not_necessary_condition_l3409_340947


namespace trig_identity_simplification_l3409_340958

theorem trig_identity_simplification (θ : Real) :
  ((1 + Real.sin θ ^ 2) ^ 2 - Real.cos θ ^ 4) * ((1 + Real.cos θ ^ 2) ^ 2 - Real.sin θ ^ 4) =
  4 * (Real.sin (2 * θ)) ^ 2 := by
  sorry

end trig_identity_simplification_l3409_340958


namespace integer_power_sum_l3409_340924

theorem integer_power_sum (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m :=
sorry

end integer_power_sum_l3409_340924


namespace average_height_calculation_l3409_340949

theorem average_height_calculation (north_count : ℕ) (south_count : ℕ) 
  (north_avg : ℝ) (south_avg : ℝ) :
  north_count = 300 →
  south_count = 200 →
  north_avg = 1.60 →
  south_avg = 1.50 →
  (north_count * north_avg + south_count * south_avg) / (north_count + south_count) = 1.56 := by
  sorry

end average_height_calculation_l3409_340949


namespace average_cookies_l3409_340970

def cookie_counts : List ℕ := [9, 11, 13, 15, 15, 17, 19, 21, 5]

theorem average_cookies : 
  (List.sum cookie_counts) / (List.length cookie_counts) = 125 / 9 := by
  sorry

end average_cookies_l3409_340970


namespace line_equations_l3409_340966

-- Define a line passing through (-1, 3) with equal absolute intercepts
def line_through_point_with_equal_intercepts (a b c : ℝ) : Prop :=
  -- The line passes through (-1, 3)
  a * (-1) + b * 3 + c = 0 ∧
  -- The line has intercepts of equal absolute values on x and y axes
  ∃ k : ℝ, k ≠ 0 ∧ (a * k + c = 0 ∨ b * k + c = 0) ∧ (a * (-k) + c = 0 ∨ b * (-k) + c = 0)

-- Theorem stating the possible equations of the line
theorem line_equations :
  ∃ (a b c : ℝ),
    line_through_point_with_equal_intercepts a b c ∧
    ((a = 3 ∧ b = 1 ∧ c = 0) ∨
     (a = 1 ∧ b = -1 ∧ c = -4) ∨
     (a = 1 ∧ b = 1 ∧ c = -2)) :=
by sorry

end line_equations_l3409_340966


namespace complex_subtraction_l3409_340997

theorem complex_subtraction (z₁ z₂ : ℂ) (h1 : z₁ = -2 - I) (h2 : z₂ = I) :
  z₁ - 2 * z₂ = -2 - 3 * I := by
  sorry

end complex_subtraction_l3409_340997


namespace higher_profit_percentage_l3409_340909

/-- The profit percentage that results in $72 more profit than 9% on a cost price of $800 is 18% -/
theorem higher_profit_percentage (cost_price : ℝ) (additional_profit : ℝ) :
  cost_price = 800 →
  additional_profit = 72 →
  ∃ (P : ℝ), P * cost_price / 100 = (9 * cost_price / 100) + additional_profit ∧ P = 18 :=
by sorry

end higher_profit_percentage_l3409_340909


namespace coefficient_a3_equals_84_l3409_340950

theorem coefficient_a3_equals_84 (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (a * x - 1)^9 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + 
                        a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 + a₉ * x^9) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 0) →
  a₃ = 84 := by
sorry

end coefficient_a3_equals_84_l3409_340950


namespace equation_solutions_l3409_340965

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => 1/((x - 2)*(x - 3)) + 1/((x - 3)*(x - 4)) + 1/((x - 4)*(x - 5))
  ∀ x : ℝ, f x = 1/8 ↔ x = 13 ∨ x = -2 := by
  sorry

end equation_solutions_l3409_340965
