import Mathlib

namespace NUMINAMATH_CALUDE_circle_diameter_l2356_235652

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l2356_235652


namespace NUMINAMATH_CALUDE_square_properties_l2356_235659

theorem square_properties (a : ℝ) (h : a^2 = 10) :
  a = Real.sqrt 10 ∧ a^2 - 10 = 0 ∧ 3 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_square_properties_l2356_235659


namespace NUMINAMATH_CALUDE_bicycle_price_reduction_l2356_235689

theorem bicycle_price_reduction (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 200 → 
  discount1 = 0.4 → 
  discount2 = 0.25 → 
  (original_price * (1 - discount1) * (1 - discount2)) = 90 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_reduction_l2356_235689


namespace NUMINAMATH_CALUDE_revenue_decrease_l2356_235676

theorem revenue_decrease (projected_increase : ℝ) (actual_vs_projected : ℝ) : 
  projected_increase = 0.30 →
  actual_vs_projected = 0.57692307692307686 →
  1 - actual_vs_projected * (1 + projected_increase) = 0.25 := by
sorry

end NUMINAMATH_CALUDE_revenue_decrease_l2356_235676


namespace NUMINAMATH_CALUDE_count_prime_digit_even_sum_integers_l2356_235613

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is a three-digit integer
def isThreeDigit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

-- Define a function to get the digits of a three-digit number
def getDigits (n : ℕ) : ℕ × ℕ × ℕ :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  (hundreds, tens, ones)

-- Define the main theorem
theorem count_prime_digit_even_sum_integers :
  (∃ S : Finset ℕ, 
    (∀ n ∈ S, isThreeDigit n ∧ 
              let (d1, d2, d3) := getDigits n
              isPrime d1 ∧ isPrime d2 ∧ isPrime d3 ∧
              (d1 + d2 + d3) % 2 = 0) ∧
    S.card = 18) := by sorry

end NUMINAMATH_CALUDE_count_prime_digit_even_sum_integers_l2356_235613


namespace NUMINAMATH_CALUDE_pens_paid_equals_pens_bought_l2356_235698

/-- Represents a retail transaction -/
structure RetailTransaction where
  pens_bought : ℕ
  discount_percent : ℝ
  profit_percent : ℝ

/-- Theorem: The number of pens paid for at market price equals the number of pens bought -/
theorem pens_paid_equals_pens_bought (transaction : RetailTransaction) :
  transaction.pens_bought = transaction.pens_bought := by
  sorry

/-- Example transaction matching the problem -/
def example_transaction : RetailTransaction :=
  { pens_bought := 40
  , discount_percent := 1
  , profit_percent := 9.999999999999996 }

#check pens_paid_equals_pens_bought example_transaction

end NUMINAMATH_CALUDE_pens_paid_equals_pens_bought_l2356_235698


namespace NUMINAMATH_CALUDE_radhika_total_games_l2356_235660

def christmas_games : ℕ := 12
def birthday_games : ℕ := 8
def original_games_ratio : ℚ := 1 / 2

theorem radhika_total_games :
  let total_gift_games := christmas_games + birthday_games
  let original_games := (total_gift_games : ℚ) * original_games_ratio
  (original_games + total_gift_games : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_radhika_total_games_l2356_235660


namespace NUMINAMATH_CALUDE_unique_cube_ending_in_nine_l2356_235684

theorem unique_cube_ending_in_nine :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ 1000 ≤ n^3 ∧ n^3 < 10000 ∧ n^3 % 10 = 9 ∧ n = 19 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_ending_in_nine_l2356_235684


namespace NUMINAMATH_CALUDE_lcm_of_1428_and_924_l2356_235665

theorem lcm_of_1428_and_924 : Nat.lcm 1428 924 = 15708 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_1428_and_924_l2356_235665


namespace NUMINAMATH_CALUDE_seven_mile_taxi_cost_l2356_235604

/-- The cost of a taxi ride given the distance traveled -/
def taxi_cost (fixed_cost : ℚ) (per_mile_cost : ℚ) (miles : ℚ) : ℚ :=
  fixed_cost + per_mile_cost * miles

/-- Theorem: The cost of a 7-mile taxi ride is $4.10 -/
theorem seven_mile_taxi_cost :
  taxi_cost 2 0.3 7 = 4.1 := by
  sorry

end NUMINAMATH_CALUDE_seven_mile_taxi_cost_l2356_235604


namespace NUMINAMATH_CALUDE_pauls_homework_average_l2356_235617

/-- Represents the homework schedule for Paul --/
structure HomeworkSchedule where
  weeknight_hours : ℕ
  weekend_hours : ℕ
  practice_nights : ℕ
  total_nights : ℕ

/-- Calculates the average homework hours per available night --/
def average_homework_hours (schedule : HomeworkSchedule) : ℚ :=
  let total_homework := schedule.weeknight_hours * (schedule.total_nights - 2) + schedule.weekend_hours
  let available_nights := schedule.total_nights - schedule.practice_nights
  (total_homework : ℚ) / available_nights

/-- Theorem stating that Paul's average homework hours per available night is 3 --/
theorem pauls_homework_average (pauls_schedule : HomeworkSchedule) 
  (h1 : pauls_schedule.weeknight_hours = 2)
  (h2 : pauls_schedule.weekend_hours = 5)
  (h3 : pauls_schedule.practice_nights = 2)
  (h4 : pauls_schedule.total_nights = 7) :
  average_homework_hours pauls_schedule = 3 := by
  sorry


end NUMINAMATH_CALUDE_pauls_homework_average_l2356_235617


namespace NUMINAMATH_CALUDE_percent_calculation_l2356_235656

theorem percent_calculation (x : ℝ) (h : 0.20 * x = 200) : 1.20 * x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_percent_calculation_l2356_235656


namespace NUMINAMATH_CALUDE_time_ratio_l2356_235677

def minutes_to_seconds (m : ℕ) : ℕ := m * 60

def hours_to_seconds (h : ℕ) : ℕ := h * 3600

def time_period_1 : ℕ := minutes_to_seconds 37 + 48

def time_period_2 : ℕ := hours_to_seconds 2 + minutes_to_seconds 13 + 15

theorem time_ratio : 
  time_period_1 * 7995 = time_period_2 * 2268 := by sorry

end NUMINAMATH_CALUDE_time_ratio_l2356_235677


namespace NUMINAMATH_CALUDE_line_equation_correct_l2356_235609

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfies_equation (p : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * p.1 + eq.b * p.2 + eq.c = 0

/-- Check if a line equation represents a line with a given slope -/
def has_slope (eq : LineEquation) (m : ℝ) : Prop :=
  eq.a ≠ 0 ∧ eq.b ≠ 0 ∧ m = -eq.a / eq.b

theorem line_equation_correct (L : Line) (eq : LineEquation) : 
  L.point = (-2, 5) →
  L.slope = -3/4 →
  eq = ⟨3, 4, -14⟩ →
  satisfies_equation L.point eq ∧ has_slope eq L.slope :=
sorry

end NUMINAMATH_CALUDE_line_equation_correct_l2356_235609


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l2356_235690

theorem intersection_sum_zero (x₁ x₂ : ℝ) (y : ℝ) :
  y = 8 →
  x₁^2 + y^2 = 145 →
  x₂^2 + y^2 = 145 →
  x₁ + x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l2356_235690


namespace NUMINAMATH_CALUDE_lottery_expected_profit_l2356_235622

/-- The expected profit for buying one lottery ticket -/
theorem lottery_expected_profit :
  let ticket_cost : ℝ := 10
  let win_probability : ℝ := 0.02
  let prize : ℝ := 300
  let expected_profit := (prize - ticket_cost) * win_probability + (-ticket_cost) * (1 - win_probability)
  expected_profit = -4 := by sorry

end NUMINAMATH_CALUDE_lottery_expected_profit_l2356_235622


namespace NUMINAMATH_CALUDE_expression_value_l2356_235616

theorem expression_value (x : ℝ) (h : x^2 + 3*x - 1 = 0) : 
  (x - 3)^2 - (2*x + 1)*(2*x - 1) - 3*x = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2356_235616


namespace NUMINAMATH_CALUDE_solve_star_equation_l2356_235648

-- Define the operation ★
def star (a b : ℝ) : ℝ := a * b + 2 * b - a

-- Theorem statement
theorem solve_star_equation :
  ∀ x : ℝ, star 5 x = 37 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l2356_235648


namespace NUMINAMATH_CALUDE_min_value_theorem_l2356_235661

theorem min_value_theorem (x : ℝ) (h : x > 0) : x + 16 / (x + 1) ≥ 7 ∧ ∃ y > 0, y + 16 / (y + 1) = 7 :=
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2356_235661


namespace NUMINAMATH_CALUDE_prob_all_red_when_n_3_n_value_when_prob_at_least_2_red_is_3_4_l2356_235625

-- Define the contents of the bags
def bag_A : ℕ × ℕ := (2, 2)  -- (red balls, white balls)
def bag_B (n : ℕ) : ℕ × ℕ := (2, n)  -- (red balls, white balls)

-- Define the probability of drawing all red balls
def prob_all_red (n : ℕ) : ℚ :=
  (Nat.choose 2 2 * Nat.choose 2 2) / (Nat.choose 4 2 * Nat.choose (n + 2) 2)

-- Define the probability of drawing at least 2 red balls
def prob_at_least_2_red (n : ℕ) : ℚ :=
  1 - (Nat.choose 2 2 * Nat.choose n 2 + Nat.choose 2 1 * Nat.choose 2 1 * Nat.choose n 2 + Nat.choose 2 2 * Nat.choose 2 1 * Nat.choose n 1) / (Nat.choose 4 2 * Nat.choose (n + 2) 2)

theorem prob_all_red_when_n_3 :
  prob_all_red 3 = 1 / 60 := by sorry

theorem n_value_when_prob_at_least_2_red_is_3_4 :
  ∃ n : ℕ, prob_at_least_2_red n = 3 / 4 ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_prob_all_red_when_n_3_n_value_when_prob_at_least_2_red_is_3_4_l2356_235625


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2356_235623

theorem quadratic_roots_sum (p : ℝ) : 
  (∃ x y : ℝ, x * y = 9 ∧ 2 * x^2 + p * x - p + 4 = 0 ∧ 2 * y^2 + p * y - p + 4 = 0) →
  (∃ x y : ℝ, x + y = 7 ∧ 2 * x^2 + p * x - p + 4 = 0 ∧ 2 * y^2 + p * y - p + 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2356_235623


namespace NUMINAMATH_CALUDE_print_time_rounded_l2356_235655

/-- Represents a printer with fast and normal modes -/
structure Printer :=
  (fast_speed : ℕ)
  (normal_speed : ℕ)

/-- Calculates the total printing time in minutes -/
def total_print_time (p : Printer) (fast_pages normal_pages : ℕ) : ℚ :=
  (fast_pages : ℚ) / p.fast_speed + (normal_pages : ℚ) / p.normal_speed

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem print_time_rounded (p : Printer) (h1 : p.fast_speed = 23) (h2 : p.normal_speed = 15) :
  round_to_nearest (total_print_time p 150 130) = 15 := by
  sorry

end NUMINAMATH_CALUDE_print_time_rounded_l2356_235655


namespace NUMINAMATH_CALUDE_hair_cut_calculation_l2356_235680

/-- Calculates the amount of hair cut off given initial length, growth rate, time, and final length --/
theorem hair_cut_calculation (initial_length growth_rate weeks final_length : ℝ) :
  initial_length = 11 ∧ 
  growth_rate = 0.5 ∧ 
  weeks = 4 ∧ 
  final_length = 7 →
  initial_length + growth_rate * weeks - final_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_calculation_l2356_235680


namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l2356_235643

/-- Represents a tile with a diagonal --/
inductive Tile
| TopLeftToBottomRight
| TopRightToBottomLeft

/-- Represents a position in the 6×6 grid --/
structure Position :=
  (x : Fin 6)
  (y : Fin 6)

/-- Represents an arrangement of tiles in the 6×6 grid --/
def Arrangement := Position → Option Tile

/-- Checks if two positions are adjacent --/
def adjacent (p1 p2 : Position) : Bool :=
  (p1.x = p2.x ∧ (p1.y.val + 1 = p2.y.val ∨ p2.y.val + 1 = p1.y.val)) ∨
  (p1.y = p2.y ∧ (p1.x.val + 1 = p2.x.val ∨ p2.x.val + 1 = p1.x.val))

/-- Checks if the arrangement is valid --/
def validArrangement (arr : Arrangement) : Prop :=
  (∀ p : Position, ∃ t : Tile, arr p = some t) ∧
  (∀ p1 p2 : Position, adjacent p1 p2 → arr p1 ≠ arr p2)

/-- The main theorem stating that a valid arrangement exists --/
theorem valid_arrangement_exists : ∃ arr : Arrangement, validArrangement arr :=
sorry


end NUMINAMATH_CALUDE_valid_arrangement_exists_l2356_235643


namespace NUMINAMATH_CALUDE_leftover_value_is_zero_l2356_235696

/-- Represents the number of coins in a roll -/
def roll_size : ℕ := 40

/-- Represents Michael's coin counts -/
def michael_quarters : ℕ := 75
def michael_nickels : ℕ := 123

/-- Represents Sarah's coin counts -/
def sarah_quarters : ℕ := 85
def sarah_nickels : ℕ := 157

/-- Calculates the total number of quarters -/
def total_quarters : ℕ := michael_quarters + sarah_quarters

/-- Calculates the total number of nickels -/
def total_nickels : ℕ := michael_nickels + sarah_nickels

/-- Calculates the number of leftover quarters -/
def leftover_quarters : ℕ := total_quarters % roll_size

/-- Calculates the number of leftover nickels -/
def leftover_nickels : ℕ := total_nickels % roll_size

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Calculates the total value of leftover coins in cents -/
def leftover_value : ℕ := leftover_quarters * quarter_value + leftover_nickels * nickel_value

/-- Theorem stating that the value of leftover coins is $0.00 -/
theorem leftover_value_is_zero : leftover_value = 0 := by sorry

end NUMINAMATH_CALUDE_leftover_value_is_zero_l2356_235696


namespace NUMINAMATH_CALUDE_pencil_pen_choices_l2356_235682

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (m n : ℕ) : ℕ := m * n

/-- Theorem: Choosing one item from a set of 4 and one from a set of 6 results in 24 possibilities -/
theorem pencil_pen_choices : choose_one_from_each 4 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_choices_l2356_235682


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l2356_235654

theorem pentagon_area_sum (u v : ℤ) 
  (h1 : 0 < v) (h2 : v < u) 
  (h3 : u^2 + 3*u*v = 451) : u + v = 21 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l2356_235654


namespace NUMINAMATH_CALUDE_circle_area_diameter_increase_l2356_235626

theorem circle_area_diameter_increase : 
  ∀ (A D A' D' : ℝ), 
  A > 0 → D > 0 → 
  A = (Real.pi / 4) * D^2 →
  A' = 4 * A →
  A' = (Real.pi / 4) * D'^2 →
  D' / D - 1 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_diameter_increase_l2356_235626


namespace NUMINAMATH_CALUDE_least_positive_integer_divisibility_l2356_235615

theorem least_positive_integer_divisibility (n : ℕ) : 
  (n % 2 = 1) → (∃ (a : ℕ), a > 0 ∧ (55^n + a * 32^n) % 2001 = 0) → 
  (∃ (a : ℕ), a > 0 ∧ a ≤ 436 ∧ (55^n + a * 32^n) % 2001 = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisibility_l2356_235615


namespace NUMINAMATH_CALUDE_fraction_sum_zero_implies_one_zero_l2356_235678

theorem fraction_sum_zero_implies_one_zero (a b c : ℝ) :
  (a - b) / (1 + a * b) + (b - c) / (1 + b * c) + (c - a) / (1 + c * a) = 0 →
  (a - b) / (1 + a * b) = 0 ∨ (b - c) / (1 + b * c) = 0 ∨ (c - a) / (1 + c * a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_implies_one_zero_l2356_235678


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l2356_235693

def binary_to_nat (b : List Bool) : Nat :=
  b.foldr (λ x acc => 2 * acc + if x then 1 else 0) 0

theorem binary_addition_subtraction :
  let a := [true, true, false, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, false, false, true] -- 1001₂
  let d := [true, false, true, false] -- 1010₂
  let result := [true, false, true, true, true] -- 10111₂
  binary_to_nat a + binary_to_nat b - binary_to_nat c + binary_to_nat d = binary_to_nat result := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l2356_235693


namespace NUMINAMATH_CALUDE_pau_total_chicken_l2356_235631

def kobe_order : ℕ := 5

def pau_order (kobe : ℕ) : ℕ := 2 * kobe

def total_pau_order (kobe : ℕ) : ℕ := 2 * pau_order kobe

theorem pau_total_chicken :
  total_pau_order kobe_order = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_pau_total_chicken_l2356_235631


namespace NUMINAMATH_CALUDE_cube_sum_over_product_equals_33_l2356_235644

theorem cube_sum_over_product_equals_33 
  (a b c : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_eq : a + b + c = 30)
  (sq_diff_eq : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) :
  (a^3 + b^3 + c^3) / (a*b*c) = 33 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_equals_33_l2356_235644


namespace NUMINAMATH_CALUDE_det_matrix_eq_one_l2356_235664

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![5, 7; 2, 3]

theorem det_matrix_eq_one : Matrix.det matrix = 1 := by sorry

end NUMINAMATH_CALUDE_det_matrix_eq_one_l2356_235664


namespace NUMINAMATH_CALUDE_t_range_for_inequality_l2356_235647

theorem t_range_for_inequality (t : ℝ) : 
  (∀ x : ℝ, abs x ≤ 1 → t + 1 > (t^2 - 4) * x) ↔ 
  (t > (Real.sqrt 13 - 1) / 2 ∧ t < (Real.sqrt 21 + 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_t_range_for_inequality_l2356_235647


namespace NUMINAMATH_CALUDE_triangle_inequality_l2356_235624

/-- Triangle inequality theorem -/
theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0)
  (h_triangle : A + B + C = π)
  (h_sine_law : a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C)) :
  A * a + B * b + C * c ≥ (1/2) * (A * b + B * a + A * c + C * a + B * c + C * b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2356_235624


namespace NUMINAMATH_CALUDE_tricubic_properties_l2356_235687

def tricubic (k : ℕ) : Prop :=
  ∃ a b c : ℕ, k = a^3 + b^3 + c^3

def exactly_one_tricubic (n : ℕ) : Prop :=
  (tricubic n ∧ ¬tricubic (n+2) ∧ ¬tricubic (n+28)) ∨
  (¬tricubic n ∧ tricubic (n+2) ∧ ¬tricubic (n+28)) ∨
  (¬tricubic n ∧ ¬tricubic (n+2) ∧ tricubic (n+28))

def exactly_two_tricubic (n : ℕ) : Prop :=
  (tricubic n ∧ tricubic (n+2) ∧ ¬tricubic (n+28)) ∨
  (tricubic n ∧ ¬tricubic (n+2) ∧ tricubic (n+28)) ∨
  (¬tricubic n ∧ tricubic (n+2) ∧ tricubic (n+28))

def all_three_tricubic (n : ℕ) : Prop :=
  tricubic n ∧ tricubic (n+2) ∧ tricubic (n+28)

theorem tricubic_properties :
  (∃ f : ℕ → ℕ, ∀ k, k < f k ∧ exactly_one_tricubic (f k)) ∧
  (∃ g : ℕ → ℕ, ∀ k, k < g k ∧ exactly_two_tricubic (g k)) ∧
  (∃ h : ℕ → ℕ, ∀ k, k < h k ∧ all_three_tricubic (h k)) := by
  sorry

end NUMINAMATH_CALUDE_tricubic_properties_l2356_235687


namespace NUMINAMATH_CALUDE_min_w_for_max_sin_l2356_235686

theorem min_w_for_max_sin (y : ℝ → ℝ) (w : ℝ) : 
  (∀ x, y x = Real.sin (w * x)) →  -- Condition 1
  w > 0 →  -- Condition 2
  (∃ n : ℕ, n ≥ 50 ∧ ∀ i : ℕ, i < n → ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ y x = 1) →  -- Condition 3
  w ≥ Real.pi * 100 :=  -- Conclusion
by sorry

end NUMINAMATH_CALUDE_min_w_for_max_sin_l2356_235686


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_zero_l2356_235629

theorem subset_implies_a_equals_zero (a : ℝ) : 
  let A : Set ℝ := {1, a - 1}
  let B : Set ℝ := {-1, 2*a - 3, 1 - 2*a}
  A ⊆ B → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_zero_l2356_235629


namespace NUMINAMATH_CALUDE_stratified_sampling_appropriate_l2356_235620

/-- Represents a sampling method -/
inductive SamplingMethod
  | DrawingLots
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a student population -/
structure Population where
  male_count : Nat
  female_count : Nat

/-- Represents a survey -/
structure Survey where
  sample_size : Nat
  method : SamplingMethod

/-- Determines if a sampling method is appropriate for a given population and survey -/
def is_appropriate_method (pop : Population) (survey : Survey) : Prop :=
  pop.male_count = pop.female_count ∧ 
  pop.male_count + pop.female_count > survey.sample_size ∧
  survey.method = SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the appropriate method for the given scenario -/
theorem stratified_sampling_appropriate (pop : Population) (survey : Survey) :
  pop.male_count = 500 ∧ pop.female_count = 500 ∧ survey.sample_size = 100 →
  is_appropriate_method pop survey :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_appropriate_l2356_235620


namespace NUMINAMATH_CALUDE_last_erased_numbers_l2356_235669

-- Define a function to count prime factors
def count_prime_factors (n : Nat) : Nat :=
  sorry

-- Theorem statement
theorem last_erased_numbers :
  ∀ n : Nat, 1 ≤ n ∧ n ≤ 100 →
    (count_prime_factors n = 6 ↔ n = 64 ∨ n = 96) ∧
    (count_prime_factors n ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_last_erased_numbers_l2356_235669


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_l2356_235634

theorem min_value_sqrt_sum (x : ℝ) :
  Real.sqrt (x^2 + 3*x + 3) + Real.sqrt (x^2 - 3*x + 3) ≥ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_l2356_235634


namespace NUMINAMATH_CALUDE_exists_all_accessible_l2356_235640

-- Define the type for cities
variable {City : Type}

-- Define the accessibility relation
variable (accessible : City → City → Prop)

-- Axioms based on the problem conditions
axiom self_accessible (c : City) : accessible c c

axiom exists_common_accessible (p q : City) : 
  ∃ r, accessible p r ∧ accessible q r

-- The main theorem to prove
theorem exists_all_accessible :
  (∀ a b : City, (accessible a b → accessible a b) → (accessible b a → accessible b a)) →
  ∃ a : City, ∀ b : City, accessible b a :=
sorry

end NUMINAMATH_CALUDE_exists_all_accessible_l2356_235640


namespace NUMINAMATH_CALUDE_remainder_of_sum_product_l2356_235673

theorem remainder_of_sum_product (p q r s : ℕ) : 
  p < 12 → q < 12 → r < 12 → s < 12 →
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s →
  Nat.gcd p 12 = 1 → Nat.gcd q 12 = 1 → Nat.gcd r 12 = 1 → Nat.gcd s 12 = 1 →
  (p * q + q * r + r * s + s * p) % 12 = 8 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_sum_product_l2356_235673


namespace NUMINAMATH_CALUDE_range_of_f_l2356_235657

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 6*x - 9

-- Define the open interval (1, 4)
def open_interval : Set ℝ := {x | 1 < x ∧ x < 4}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ open_interval, f x = y} = {y | -18 ≤ y ∧ y < -14} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2356_235657


namespace NUMINAMATH_CALUDE_sum_of_squares_l2356_235639

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0)
  (cubic_eq_quintic_eq_septic : a^3 + b^3 + c^3 = a^5 + b^5 + c^5 ∧ a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6/5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2356_235639


namespace NUMINAMATH_CALUDE_arctan_sum_one_third_three_eighths_l2356_235605

theorem arctan_sum_one_third_three_eighths (x y : ℝ) :
  x = 1 / 3 →
  y = 3 / 8 →
  x + y ≠ -π / 2 →
  (∀ a b : ℝ, a + b ≠ -π / 2 → Real.arctan a + Real.arctan b = Real.arctan ((a + b) / (1 - a * b))) →
  Real.arctan x + Real.arctan y = Real.arctan (17 / 21) := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_one_third_three_eighths_l2356_235605


namespace NUMINAMATH_CALUDE_club_size_after_four_years_l2356_235667

-- Define the club structure
structure Club where
  leaders : ℕ
  regular_members : ℕ

-- Define the initial state and yearly update function
def initial_club : Club := { leaders := 4, regular_members := 16 }

def update_club (c : Club) : Club :=
  { leaders := 4, regular_members := 4 * c.regular_members }

-- Define the club state after n years
def club_after_years (n : ℕ) : Club :=
  match n with
  | 0 => initial_club
  | n+1 => update_club (club_after_years n)

-- Theorem statement
theorem club_size_after_four_years :
  (club_after_years 4).leaders + (club_after_years 4).regular_members = 4100 := by
  sorry


end NUMINAMATH_CALUDE_club_size_after_four_years_l2356_235667


namespace NUMINAMATH_CALUDE_unique_integers_for_odd_prime_l2356_235606

theorem unique_integers_for_odd_prime (p : ℕ) (h_prime : Prime p) (h_odd : Odd p) :
  ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ m^2 = n * (n + p) ∧ m = (p^2 - 1) / 2 ∧ n = (p - 1)^2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_integers_for_odd_prime_l2356_235606


namespace NUMINAMATH_CALUDE_apts_on_fewer_floors_eq_30_total_apts_on_fewer_floors_l2356_235635

/-- Represents a block of flats with given specifications -/
structure BlockOfFlats where
  total_floors : ℕ
  floors_with_more_apts : ℕ
  apts_on_more_floors : ℕ
  max_residents_per_apt : ℕ
  max_total_residents : ℕ

/-- The number of apartments on floors with fewer apartments -/
def apts_on_fewer_floors (b : BlockOfFlats) : ℕ :=
  (b.max_total_residents - b.max_residents_per_apt * b.floors_with_more_apts * b.apts_on_more_floors) /
  (b.max_residents_per_apt * (b.total_floors - b.floors_with_more_apts))

/-- Theorem stating the number of apartments on floors with fewer apartments -/
theorem apts_on_fewer_floors_eq_30 (b : BlockOfFlats) 
  (h1 : b.total_floors = 12)
  (h2 : b.floors_with_more_apts = 6)
  (h3 : b.apts_on_more_floors = 6)
  (h4 : b.max_residents_per_apt = 4)
  (h5 : b.max_total_residents = 264) :
  apts_on_fewer_floors b = 5 := by
  sorry

/-- Corollary for the total number of apartments on floors with fewer apartments -/
theorem total_apts_on_fewer_floors (b : BlockOfFlats) 
  (h1 : b.total_floors = 12)
  (h2 : b.floors_with_more_apts = 6)
  (h3 : b.apts_on_more_floors = 6)
  (h4 : b.max_residents_per_apt = 4)
  (h5 : b.max_total_residents = 264) :
  (b.total_floors - b.floors_with_more_apts) * apts_on_fewer_floors b = 30 := by
  sorry

end NUMINAMATH_CALUDE_apts_on_fewer_floors_eq_30_total_apts_on_fewer_floors_l2356_235635


namespace NUMINAMATH_CALUDE_triangle_area_implies_p_value_l2356_235649

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, p),
    if the area of the triangle is 36, then p = 12.75 -/
theorem triangle_area_implies_p_value :
  ∀ (p : ℝ),
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, p)
  let triangle_area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  triangle_area = 36 → p = 12.75 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_implies_p_value_l2356_235649


namespace NUMINAMATH_CALUDE_base_eight_distinct_digits_l2356_235688

/-- The number of four-digit numbers with distinct digits in base b -/
def distinctDigitCount (b : ℕ) : ℕ := (b - 1) * (b - 2) * (b - 3)

/-- Theorem stating that there are exactly 168 four-digit numbers with distinct digits in base 8 -/
theorem base_eight_distinct_digits :
  ∃ (b : ℕ), b > 4 ∧ distinctDigitCount b = 168 ↔ distinctDigitCount 8 = 168 :=
sorry

end NUMINAMATH_CALUDE_base_eight_distinct_digits_l2356_235688


namespace NUMINAMATH_CALUDE_white_balls_count_l2356_235630

theorem white_balls_count (n : ℕ) : 
  n = 27 ∧ 
  (∃ (total : ℕ), 
    total = n + 3 ∧ 
    (3 : ℚ) / total = 1 / 10) := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l2356_235630


namespace NUMINAMATH_CALUDE_roger_earnings_l2356_235646

theorem roger_earnings : ∀ (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ),
  rate = 9 →
  total_lawns = 14 →
  forgotten_lawns = 8 →
  (total_lawns - forgotten_lawns) * rate = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_roger_earnings_l2356_235646


namespace NUMINAMATH_CALUDE_expression_simplification_l2356_235638

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 - b^2) / (a^2 + 2*a*b + b^2) + 
  (2 / (a*b)) / ((1/a + 1/b)^2) * (2 / (a^2 - b^2 + 2*a*b)) = 
  2 / (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2356_235638


namespace NUMINAMATH_CALUDE_f_properties_l2356_235645

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / a - Real.exp x

def hasExactlyTwoZeroPoints (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂

theorem f_properties (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Icc 1 2, f a x ≤ (
    if a ≤ Real.exp (-2) then 2 / a - Real.exp 2
    else if a < Real.exp (-1) then (Real.log (1 / a)) / a - 1 / a
    else 1 / a - Real.exp 1
  )) ∧
  (hasExactlyTwoZeroPoints (f a) ↔ 0 < a ∧ a < Real.exp (-1)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2356_235645


namespace NUMINAMATH_CALUDE_twenty_photos_needed_l2356_235668

/-- The minimum number of non-overlapping rectangular photos required to form a square -/
def min_photos_for_square (width : ℕ) (length : ℕ) : ℕ :=
  let square_side := Nat.lcm width length
  (square_side * square_side) / (width * length)

/-- Theorem stating that 20 photos of 12cm x 15cm are needed for the smallest square -/
theorem twenty_photos_needed : min_photos_for_square 12 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_photos_needed_l2356_235668


namespace NUMINAMATH_CALUDE_line_contains_point_l2356_235633

/-- The value of k that makes the line 3 - ky = -4x contain the point (2, -1) -/
def k : ℝ := -11

/-- The equation of the line -/
def line_equation (x y : ℝ) (k : ℝ) : Prop :=
  3 - k * y = -4 * x

/-- The point that should lie on the line -/
def point : ℝ × ℝ := (2, -1)

/-- Theorem stating that k makes the line contain the given point -/
theorem line_contains_point : line_equation point.1 point.2 k := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l2356_235633


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l2356_235607

/-- Given a train of length 1200 m that crosses a tree in 80 seconds,
    prove that the time it takes to pass a platform of length 1000 m is 146.67 seconds. -/
theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (tree_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1200)
  (h2 : tree_crossing_time = 80)
  (h3 : platform_length = 1000) :
  (train_length + platform_length) / (train_length / tree_crossing_time) = 146.67 := by
sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l2356_235607


namespace NUMINAMATH_CALUDE_original_profit_margin_l2356_235601

theorem original_profit_margin (original_price selling_price : ℝ) : 
  original_price > 0 →
  selling_price > original_price →
  let new_price := 0.9 * original_price
  let original_margin := (selling_price - original_price) / original_price
  let new_margin := (selling_price - new_price) / new_price
  new_margin - original_margin = 0.12 →
  original_margin = 0.08 := by
sorry

end NUMINAMATH_CALUDE_original_profit_margin_l2356_235601


namespace NUMINAMATH_CALUDE_fraction_repeating_block_length_l2356_235618

/-- The length of the repeating block in the decimal expansion of 7/13 -/
def repeatingBlockLength : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 7/13

theorem fraction_repeating_block_length :
  ∃ (d : ℕ+) (n : ℕ), 
    fraction * d.val = n ∧ 
    d = 10^repeatingBlockLength - 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_repeating_block_length_l2356_235618


namespace NUMINAMATH_CALUDE_multiply_powers_of_y_l2356_235672

theorem multiply_powers_of_y (y : ℝ) : 5 * y^3 * (3 * y^2) = 15 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_y_l2356_235672


namespace NUMINAMATH_CALUDE_q_invest_time_is_20_l2356_235674

/-- Represents a business partnership between two partners -/
structure Partnership where
  investment_ratio : ℚ × ℚ
  profit_ratio : ℚ × ℚ
  p_invest_time : ℕ

/-- Calculates the investment time for partner q given a Partnership -/
def q_invest_time (p : Partnership) : ℚ :=
  (p.profit_ratio.2 * p.investment_ratio.1 * p.p_invest_time : ℚ) / (p.profit_ratio.1 * p.investment_ratio.2)

theorem q_invest_time_is_20 (p : Partnership) 
  (h1 : p.investment_ratio = (7, 5))
  (h2 : p.profit_ratio = (7, 10))
  (h3 : p.p_invest_time = 10) :
  q_invest_time p = 20 := by
  sorry

end NUMINAMATH_CALUDE_q_invest_time_is_20_l2356_235674


namespace NUMINAMATH_CALUDE_square_field_area_l2356_235666

/-- Calculates the area of a square field given the cost of barbed wire around it -/
theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) : 
  wire_cost_per_meter = 3 →
  total_cost = 1998 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ), 
    wire_cost_per_meter * (4 * side_length - num_gates * gate_width) = total_cost ∧
    side_length ^ 2 = 27889 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l2356_235666


namespace NUMINAMATH_CALUDE_composite_divisor_bound_l2356_235600

/-- A number is composite if it's a natural number greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

/-- Theorem: Every composite number has a divisor greater than 1 but not greater than its square root -/
theorem composite_divisor_bound {n : ℕ} (h : IsComposite n) :
  ∃ d : ℕ, d ∣ n ∧ 1 < d ∧ d ≤ Real.sqrt (n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_composite_divisor_bound_l2356_235600


namespace NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_l2356_235612

theorem a_gt_abs_b_sufficient_not_necessary :
  (∃ a b : ℝ, a > |b| ∧ a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > |b|)) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_l2356_235612


namespace NUMINAMATH_CALUDE_betty_order_total_payment_l2356_235636

/-- Calculates the total payment for Betty's order including shipping -/
def totalPayment (
  slipperPrice : Float) (slipperWeight : Float) (slipperCount : Nat)
  (lipstickPrice : Float) (lipstickWeight : Float) (lipstickCount : Nat)
  (hairColorPrice : Float) (hairColorWeight : Float) (hairColorCount : Nat)
  (sunglassesPrice : Float) (sunglassesWeight : Float) (sunglassesCount : Nat)
  (tshirtPrice : Float) (tshirtWeight : Float) (tshirtCount : Nat)
  : Float :=
  let totalCost := 
    slipperPrice * slipperCount.toFloat +
    lipstickPrice * lipstickCount.toFloat +
    hairColorPrice * hairColorCount.toFloat +
    sunglassesPrice * sunglassesCount.toFloat +
    tshirtPrice * tshirtCount.toFloat
  let totalWeight :=
    slipperWeight * slipperCount.toFloat +
    lipstickWeight * lipstickCount.toFloat +
    hairColorWeight * hairColorCount.toFloat +
    sunglassesWeight * sunglassesCount.toFloat +
    tshirtWeight * tshirtCount.toFloat
  let shippingCost :=
    if totalWeight ≤ 5 then 2
    else if totalWeight ≤ 10 then 4
    else 6
  totalCost + shippingCost

theorem betty_order_total_payment :
  totalPayment 2.5 0.3 6 1.25 0.05 4 3 0.2 8 5.75 0.1 3 12.25 0.5 4 = 114.25 := by
  sorry

end NUMINAMATH_CALUDE_betty_order_total_payment_l2356_235636


namespace NUMINAMATH_CALUDE_decagon_adjacent_vertices_probability_l2356_235651

/-- A decagon is a polygon with 10 sides -/
def Decagon : Type := Unit

/-- The number of vertices in a decagon -/
def num_vertices : Nat := 10

/-- The number of ways to choose 3 distinct vertices from a decagon -/
def total_choices : Nat := Nat.choose num_vertices 3

/-- The number of ways to choose 3 adjacent vertices in a decagon -/
def adjacent_choices : Nat := num_vertices

/-- The probability of choosing 3 adjacent vertices in a decagon -/
def prob_adjacent_vertices : Rat := adjacent_choices / total_choices

theorem decagon_adjacent_vertices_probability :
  prob_adjacent_vertices = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_decagon_adjacent_vertices_probability_l2356_235651


namespace NUMINAMATH_CALUDE_bills_toilet_paper_duration_l2356_235662

/-- The number of days Bill's toilet paper supply will last -/
def toilet_paper_duration (bathroom_visits_per_day : ℕ) (squares_per_visit : ℕ) 
  (total_rolls : ℕ) (squares_per_roll : ℕ) : ℕ :=
  (total_rolls * squares_per_roll) / (bathroom_visits_per_day * squares_per_visit)

/-- Theorem stating that Bill's toilet paper supply will last 20,000 days -/
theorem bills_toilet_paper_duration :
  toilet_paper_duration 3 5 1000 300 = 20000 := by
  sorry

#eval toilet_paper_duration 3 5 1000 300

end NUMINAMATH_CALUDE_bills_toilet_paper_duration_l2356_235662


namespace NUMINAMATH_CALUDE_magnitude_of_one_plus_two_i_to_eighth_l2356_235663

theorem magnitude_of_one_plus_two_i_to_eighth : Complex.abs ((1 + 2*Complex.I)^8) = 625 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_one_plus_two_i_to_eighth_l2356_235663


namespace NUMINAMATH_CALUDE_convention_handshakes_eq_680_l2356_235641

/-- Represents the number of handshakes at a twins and quadruplets convention --/
def convention_handshakes : ℕ := by
  -- Define the number of twin sets and quadruplet sets
  let twin_sets : ℕ := 8
  let quad_sets : ℕ := 5

  -- Calculate total number of twins and quadruplets
  let total_twins : ℕ := twin_sets * 2
  let total_quads : ℕ := quad_sets * 4

  -- Calculate handshakes among twins
  let twin_handshakes : ℕ := (total_twins * (total_twins - 2)) / 2

  -- Calculate handshakes among quadruplets
  let quad_handshakes : ℕ := (total_quads * (total_quads - 4)) / 2

  -- Calculate cross handshakes between twins and quadruplets
  let cross_handshakes : ℕ := total_twins * (2 * total_quads / 3)

  -- Sum all handshakes
  exact twin_handshakes + quad_handshakes + cross_handshakes

/-- Theorem stating that the total number of handshakes is 680 --/
theorem convention_handshakes_eq_680 : convention_handshakes = 680 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_eq_680_l2356_235641


namespace NUMINAMATH_CALUDE_power_inequality_l2356_235627

theorem power_inequality : 81^31 > 27^41 ∧ 27^41 > 9^61 := by sorry

end NUMINAMATH_CALUDE_power_inequality_l2356_235627


namespace NUMINAMATH_CALUDE_larger_divided_by_smaller_l2356_235603

theorem larger_divided_by_smaller (L S : ℕ) (hL : L > S) : 
  L - S = 1365 → S = 270 → ∃ Q : ℕ, L = S * Q + 15 → Q = 6 := by
  sorry

end NUMINAMATH_CALUDE_larger_divided_by_smaller_l2356_235603


namespace NUMINAMATH_CALUDE_odd_expressions_l2356_235685

-- Define positive odd integers
def is_positive_odd (n : ℤ) : Prop := n > 0 ∧ ∃ k : ℤ, n = 2*k + 1

-- Theorem statement
theorem odd_expressions (p q : ℤ) 
  (hp : is_positive_odd p) (hq : is_positive_odd q) : 
  ∃ m n : ℤ, p * q + 2 = 2*m + 1 ∧ p^3 * q + q^2 = 2*n + 1 :=
sorry

end NUMINAMATH_CALUDE_odd_expressions_l2356_235685


namespace NUMINAMATH_CALUDE_mrs_lee_class_boys_without_glasses_l2356_235611

/-- Represents Mrs. Lee's biology class -/
structure BiologyClass where
  total_boys : ℕ
  students_with_glasses : ℕ
  girls_with_glasses : ℕ

/-- Calculates the number of boys not wearing glasses in the class -/
def boys_without_glasses (c : BiologyClass) : ℕ :=
  c.total_boys - (c.students_with_glasses - c.girls_with_glasses)

/-- Theorem stating that in Mrs. Lee's class, 15 boys do not wear glasses -/
theorem mrs_lee_class_boys_without_glasses :
  ∃ (c : BiologyClass),
    c.total_boys = 30 ∧
    c.students_with_glasses = 36 ∧
    c.girls_with_glasses = 21 ∧
    boys_without_glasses c = 15 := by
  sorry

end NUMINAMATH_CALUDE_mrs_lee_class_boys_without_glasses_l2356_235611


namespace NUMINAMATH_CALUDE_homework_difference_is_two_l2356_235650

/-- The number of pages of reading homework Rachel had to complete -/
def reading_pages : ℕ := 2

/-- The number of pages of math homework Rachel had to complete -/
def math_pages : ℕ := 4

/-- The difference between math homework pages and reading homework pages -/
def homework_difference : ℕ := math_pages - reading_pages

theorem homework_difference_is_two : homework_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_homework_difference_is_two_l2356_235650


namespace NUMINAMATH_CALUDE_apple_seedling_survival_probability_l2356_235695

/-- Survival rate data for apple seedlings -/
def survival_data : List (ℕ × ℝ) := [
  (100, 0.81),
  (200, 0.78),
  (500, 0.79),
  (1000, 0.8),
  (2000, 0.8)
]

/-- The estimated probability of survival for apple seedlings after transplantation -/
def estimated_survival_probability : ℝ := 0.8

/-- Theorem stating that the estimated probability of survival is 0.8 -/
theorem apple_seedling_survival_probability :
  estimated_survival_probability = 0.8 :=
sorry

end NUMINAMATH_CALUDE_apple_seedling_survival_probability_l2356_235695


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2356_235619

theorem container_volume_ratio (V₁ V₂ V₃ : ℝ) 
  (h₁ : V₁ > 0) (h₂ : V₂ > 0) (h₃ : V₃ > 0)
  (h₄ : (3/4) * V₁ = (5/8) * V₂)
  (h₅ : (5/8) * V₂ = (1/2) * V₃) :
  V₁ / V₃ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2356_235619


namespace NUMINAMATH_CALUDE_sum_of_squares_even_2_to_14_l2356_235699

def evenSquareSum : ℕ → ℕ
| 0 => 0
| n + 1 => if n + 1 ≤ 7 ∧ 2 * (n + 1) ≤ 14 then (2 * (n + 1))^2 + evenSquareSum n else evenSquareSum n

theorem sum_of_squares_even_2_to_14 : evenSquareSum 7 = 560 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_even_2_to_14_l2356_235699


namespace NUMINAMATH_CALUDE_inverse_sum_modulo_13_l2356_235681

theorem inverse_sum_modulo_13 :
  (((2⁻¹ : ZMod 13) + (4⁻¹ : ZMod 13) + (5⁻¹ : ZMod 13) + (7⁻¹ : ZMod 13))⁻¹ : ZMod 13) = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_modulo_13_l2356_235681


namespace NUMINAMATH_CALUDE_complement_of_union_l2356_235691

def U : Set Nat := {1,2,3,4,5,6,7,8}
def S : Set Nat := {1,3,5}
def T : Set Nat := {3,6}

theorem complement_of_union : 
  (U \ (S ∪ T)) = {2,4,7,8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2356_235691


namespace NUMINAMATH_CALUDE_sandra_coffee_cups_l2356_235670

/-- Given that Sandra and Marcie took a total of 8 cups of coffee, 
    and Marcie took 2 cups, prove that Sandra took 6 cups of coffee. -/
theorem sandra_coffee_cups (total : ℕ) (marcie : ℕ) (sandra : ℕ) 
  (h1 : total = 8) 
  (h2 : marcie = 2) 
  (h3 : sandra + marcie = total) : 
  sandra = 6 := by
  sorry

end NUMINAMATH_CALUDE_sandra_coffee_cups_l2356_235670


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l2356_235610

theorem negation_of_existence_proposition :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l2356_235610


namespace NUMINAMATH_CALUDE_all_propositions_true_l2356_235694

-- Define the original proposition
def original_proposition (x : ℝ) : Prop :=
  (x = 2 ∨ x = -3) → (x - 2) * (x + 3) = 0

-- Define the converse
def converse (x : ℝ) : Prop :=
  (x - 2) * (x + 3) = 0 → (x = 2 ∨ x = -3)

-- Define the inverse
def inverse (x : ℝ) : Prop :=
  (x ≠ 2 ∧ x ≠ -3) → (x - 2) * (x + 3) ≠ 0

-- Define the contrapositive
def contrapositive (x : ℝ) : Prop :=
  (x - 2) * (x + 3) ≠ 0 → (x ≠ 2 ∧ x ≠ -3)

-- Theorem stating that all propositions are true for all real numbers
theorem all_propositions_true :
  ∀ x : ℝ, original_proposition x ∧ converse x ∧ inverse x ∧ contrapositive x :=
by sorry


end NUMINAMATH_CALUDE_all_propositions_true_l2356_235694


namespace NUMINAMATH_CALUDE_ab_value_l2356_235642

theorem ab_value (a b : ℤ) (h1 : |a| = 5) (h2 : b = -3) (h3 : a < b) : a * b = 15 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2356_235642


namespace NUMINAMATH_CALUDE_symmetric_points_coordinate_sum_l2356_235628

/-- Given two points P and Q symmetric with respect to the origin O in a Cartesian coordinate system, 
    prove that the sum of their x and y coordinates is -4. -/
theorem symmetric_points_coordinate_sum (p q : ℝ) : 
  (∃ (P Q : ℝ × ℝ), P = (p, -2) ∧ Q = (6, q) ∧ P.1 = -Q.1 ∧ P.2 = -Q.2) →
  p + q = -4 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_coordinate_sum_l2356_235628


namespace NUMINAMATH_CALUDE_wins_to_losses_ratio_l2356_235632

/-- Represents the statistics of a baseball team's season. -/
structure BaseballSeason where
  total_games : ℕ
  wins : ℕ
  losses : ℕ

/-- Defines the conditions for the baseball season. -/
def validSeason (s : BaseballSeason) : Prop :=
  s.total_games = 130 ∧
  s.wins = s.losses + 14 ∧
  s.wins = 101

/-- Theorem stating the ratio of wins to losses for the given conditions. -/
theorem wins_to_losses_ratio (s : BaseballSeason) (h : validSeason s) :
  s.wins = 101 ∧ s.losses = 87 := by
  sorry

#check wins_to_losses_ratio

end NUMINAMATH_CALUDE_wins_to_losses_ratio_l2356_235632


namespace NUMINAMATH_CALUDE_product_of_numbers_l2356_235683

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 220) : x * y = 56 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2356_235683


namespace NUMINAMATH_CALUDE_fruit_salad_oranges_l2356_235671

theorem fruit_salad_oranges :
  ∀ (s k a o : ℕ),
    s + k + a + o = 360 →
    s = k / 2 →
    a = 2 * o →
    o = 3 * s →
    o = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_oranges_l2356_235671


namespace NUMINAMATH_CALUDE_jane_tom_sum_difference_l2356_235608

/-- The sum of numbers from 1 to 50 -/
def janeSum : ℕ := (List.range 50).map (· + 1) |>.sum

/-- Function to replace 3 with 2 in a number -/
def replace3With2 (n : ℕ) : ℕ :=
  let s := toString n
  (s.replace "3" "2").toNat!

/-- The sum of numbers from 1 to 50 with 3 replaced by 2 -/
def tomSum : ℕ := (List.range 50).map (· + 1) |>.map replace3With2 |>.sum

/-- Theorem stating the difference between Jane's and Tom's sums -/
theorem jane_tom_sum_difference : janeSum - tomSum = 105 := by
  sorry

end NUMINAMATH_CALUDE_jane_tom_sum_difference_l2356_235608


namespace NUMINAMATH_CALUDE_segment_length_parallel_to_x_axis_l2356_235692

/-- Given two points M and N, where M's coordinates depend on parameter a,
    and MN is parallel to the x-axis, prove that the length of MN is 6. -/
theorem segment_length_parallel_to_x_axis 
  (a : ℝ) 
  (M : ℝ × ℝ := (a + 3, a - 4))
  (N : ℝ × ℝ := (-1, -2))
  (h_parallel : M.2 = N.2) : 
  abs (M.1 - N.1) = 6 := by
sorry

end NUMINAMATH_CALUDE_segment_length_parallel_to_x_axis_l2356_235692


namespace NUMINAMATH_CALUDE_tank_full_time_l2356_235621

/-- Represents the time it takes to fill a tank with given parameters -/
def fill_time (tank_capacity : ℕ) (pipe_a_rate : ℕ) (pipe_b_rate : ℕ) (pipe_c_rate : ℕ) : ℕ :=
  let cycle_net_fill := pipe_a_rate + pipe_b_rate - pipe_c_rate
  let cycles := tank_capacity / cycle_net_fill
  let total_minutes := cycles * 3
  total_minutes - 1

/-- Theorem stating that the tank will be full after 50 minutes -/
theorem tank_full_time :
  fill_time 850 40 30 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_tank_full_time_l2356_235621


namespace NUMINAMATH_CALUDE_new_range_theorem_l2356_235658

/-- Represents the number of mutual funds -/
def num_funds : ℕ := 150

/-- Represents the range of annual yield last year -/
def last_year_range : ℝ := 12500

/-- Represents the percentage increase for the first group of funds -/
def increase_group1 : ℝ := 0.12

/-- Represents the percentage increase for the second group of funds -/
def increase_group2 : ℝ := 0.17

/-- Represents the percentage increase for the third group of funds -/
def increase_group3 : ℝ := 0.22

/-- Represents the size of each group of funds -/
def group_size : ℕ := 50

/-- Theorem stating that the range of annual yield this year is $27,750 -/
theorem new_range_theorem : 
  ∃ (L H : ℝ), 
    H - L = last_year_range ∧ 
    (H * (1 + increase_group3)) - (L * (1 + increase_group1)) = 27750 :=
sorry

end NUMINAMATH_CALUDE_new_range_theorem_l2356_235658


namespace NUMINAMATH_CALUDE_haley_money_received_l2356_235697

/-- Proves that Haley received 13 dollars from doing chores and her birthday -/
theorem haley_money_received (initial_amount : ℕ) (difference : ℕ) : 
  initial_amount = 2 → difference = 11 → initial_amount + difference = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_haley_money_received_l2356_235697


namespace NUMINAMATH_CALUDE_magician_decks_left_l2356_235653

/-- A magician sells magic card decks. -/
structure Magician where
  initial_decks : ℕ  -- Number of decks at the start
  price_per_deck : ℕ  -- Price of each deck in dollars
  earnings : ℕ  -- Total earnings in dollars

/-- Calculate the number of decks left for a magician. -/
def decks_left (m : Magician) : ℕ :=
  m.initial_decks - m.earnings / m.price_per_deck

/-- Theorem: The magician has 3 decks left at the end of the day. -/
theorem magician_decks_left :
  ∀ (m : Magician),
    m.initial_decks = 5 →
    m.price_per_deck = 2 →
    m.earnings = 4 →
    decks_left m = 3 := by
  sorry

end NUMINAMATH_CALUDE_magician_decks_left_l2356_235653


namespace NUMINAMATH_CALUDE_two_solutions_exist_sum_of_solutions_l2356_235602

/-- Sum of digits of a positive integer in base 10 -/
def sum_of_digits (n : ℕ+) : ℕ :=
  sorry

/-- The equation n - 3 * sum_of_digits n = 2022 has exactly two solutions -/
theorem two_solutions_exist :
  ∃ (n1 n2 : ℕ+),
    n1 - 3 * sum_of_digits n1 = 2022 ∧
    n2 - 3 * sum_of_digits n2 = 2022 ∧
    n1 ≠ n2 ∧
    ∀ (n : ℕ+), n - 3 * sum_of_digits n = 2022 → n = n1 ∨ n = n2 :=
  sorry

/-- The sum of the two solutions is 4107 -/
theorem sum_of_solutions :
  ∃ (n1 n2 : ℕ+),
    n1 - 3 * sum_of_digits n1 = 2022 ∧
    n2 - 3 * sum_of_digits n2 = 2022 ∧
    n1 ≠ n2 ∧
    n1 + n2 = 4107 :=
  sorry

end NUMINAMATH_CALUDE_two_solutions_exist_sum_of_solutions_l2356_235602


namespace NUMINAMATH_CALUDE_factors_of_72_l2356_235679

theorem factors_of_72 : Nat.card (Nat.divisors 72) = 12 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_72_l2356_235679


namespace NUMINAMATH_CALUDE_sector_arc_length_l2356_235675

/-- Given a circular sector with circumference 4 and central angle 2 radians, 
    the arc length of the sector is 2. -/
theorem sector_arc_length (r : ℝ) (l : ℝ) : 
  l + 2 * r = 4 →  -- circumference of the sector
  l = 2 * r →      -- relationship between arc length and radius
  l = 2 :=         -- arc length is 2
by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2356_235675


namespace NUMINAMATH_CALUDE_extra_flowers_l2356_235614

theorem extra_flowers (tulips roses used : ℕ) : 
  tulips = 4 → roses = 11 → used = 11 → tulips + roses - used = 4 := by
  sorry

end NUMINAMATH_CALUDE_extra_flowers_l2356_235614


namespace NUMINAMATH_CALUDE_triangle_property_l2356_235637

-- Define the necessary types and structures
structure Point := (x y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Line :=
  (p1 p2 : Point)

-- Define the given conditions
def isAcute (t : Triangle) : Prop := sorry

def isOrthocenter (H : Point) (t : Triangle) : Prop := sorry

def lieOnSide (P : Point) (l : Line) : Prop := sorry

def angleEquals (A B C : Point) (angle : ℝ) : Prop := sorry

def intersectsAt (l1 l2 : Line) (P : Point) : Prop := sorry

def isCircumcenter (O : Point) (t : Triangle) : Prop := sorry

def sameSideAs (P Q : Point) (l : Line) : Prop := sorry

def collinear (P Q R : Point) : Prop := sorry

-- Define the theorem
theorem triangle_property 
  (ABC : Triangle) 
  (H M N P Q O E : Point) :
  isAcute ABC →
  isOrthocenter H ABC →
  lieOnSide M (Line.mk ABC.A ABC.B) →
  lieOnSide N (Line.mk ABC.A ABC.C) →
  angleEquals H M ABC.B (60 : ℝ) →
  angleEquals H N ABC.C (60 : ℝ) →
  intersectsAt (Line.mk H M) (Line.mk ABC.C ABC.A) P →
  intersectsAt (Line.mk H N) (Line.mk ABC.B ABC.A) Q →
  isCircumcenter O (Triangle.mk H M N) →
  angleEquals E ABC.B ABC.C (60 : ℝ) →
  sameSideAs E ABC.A (Line.mk ABC.B ABC.C) →
  collinear E O H →
  (Line.mk O H).p1 = (Line.mk P Q).p1 ∧ -- OH ⊥ PQ
  (Triangle.mk E ABC.B ABC.C).A = (Triangle.mk E ABC.B ABC.C).B ∧ 
  (Triangle.mk E ABC.B ABC.C).B = (Triangle.mk E ABC.B ABC.C).C -- Triangle EBC is equilateral
  := by sorry

end NUMINAMATH_CALUDE_triangle_property_l2356_235637
