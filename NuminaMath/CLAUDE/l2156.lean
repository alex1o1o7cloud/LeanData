import Mathlib

namespace hex_F2E1_equals_62177_l2156_215676

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : Nat :=
  match c with
  | '0' => 0 | '1' => 1 | '2' => 2 | '3' => 3 | '4' => 4 | '5' => 5 | '6' => 6 | '7' => 7
  | '8' => 8 | '9' => 9 | 'A' => 10 | 'B' => 11 | 'C' => 12 | 'D' => 13 | 'E' => 14 | 'F' => 15
  | _ => 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_to_decimal (s : String) : Nat :=
  s.foldr (fun c acc => hex_to_dec c + 16 * acc) 0

/-- The hexadecimal number F2E1 -/
def hex_number : String := "F2E1"

/-- Theorem stating that F2E1 in hexadecimal is equal to 62177 in decimal -/
theorem hex_F2E1_equals_62177 : hex_to_decimal hex_number = 62177 := by
  sorry

end hex_F2E1_equals_62177_l2156_215676


namespace magnitude_of_complex_number_l2156_215695

theorem magnitude_of_complex_number (z : ℂ) : z = (2 * Complex.I) / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end magnitude_of_complex_number_l2156_215695


namespace symmetric_point_x_axis_l2156_215694

/-- Given a point M(3, -4), its symmetric point with respect to the x-axis has coordinates (3, 4) -/
theorem symmetric_point_x_axis : 
  let M : ℝ × ℝ := (3, -4)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point M = (3, 4) := by sorry

end symmetric_point_x_axis_l2156_215694


namespace binomial_inequality_l2156_215648

theorem binomial_inequality (x : ℝ) (n : ℕ) (h1 : x > -1) (h2 : x ≠ 0) (h3 : n ≥ 2) :
  (1 + x)^n > 1 + n * x := by
  sorry

end binomial_inequality_l2156_215648


namespace weight_of_HClO2_l2156_215653

/-- The molar mass of HClO2 in g/mol -/
def molar_mass_HClO2 : ℝ := 68.46

/-- The number of moles of HClO2 -/
def moles_HClO2 : ℝ := 6

/-- The weight of HClO2 in grams -/
def weight_HClO2 : ℝ := molar_mass_HClO2 * moles_HClO2

theorem weight_of_HClO2 :
  weight_HClO2 = 410.76 := by sorry

end weight_of_HClO2_l2156_215653


namespace symmetric_angles_sum_l2156_215617

theorem symmetric_angles_sum (α β : Real) : 
  0 < α ∧ α < 2 * Real.pi ∧ 
  0 < β ∧ β < 2 * Real.pi ∧ 
  α = 2 * Real.pi - β → 
  α + β = 2 * Real.pi :=
by sorry

end symmetric_angles_sum_l2156_215617


namespace zoo_badge_problem_l2156_215651

/-- Represents the commemorative badges sold by the zoo -/
inductive Badge
| A
| B

/-- Represents the cost and selling prices of badges -/
structure BadgePrices where
  cost_A : ℝ
  cost_B : ℝ
  sell_A : ℝ
  sell_B : ℝ

/-- Represents the purchasing plan for badges -/
structure PurchasePlan where
  num_A : ℕ
  num_B : ℕ

/-- Calculates the total cost of a purchasing plan -/
def total_cost (prices : BadgePrices) (plan : PurchasePlan) : ℝ :=
  prices.cost_A * plan.num_A + prices.cost_B * plan.num_B

/-- Calculates the total profit of a purchasing plan -/
def total_profit (prices : BadgePrices) (plan : PurchasePlan) : ℝ :=
  (prices.sell_A - prices.cost_A) * plan.num_A + (prices.sell_B - prices.cost_B) * plan.num_B

/-- Theorem representing the zoo's badge problem -/
theorem zoo_badge_problem (prices : BadgePrices) 
  (h1 : prices.cost_A = prices.cost_B + 4)
  (h2 : 6 * prices.cost_A = 10 * prices.cost_B)
  (h3 : prices.sell_A = 13)
  (h4 : prices.sell_B = 8)
  : 
  prices.cost_A = 10 ∧ 
  prices.cost_B = 6 ∧
  ∃ (optimal_plan : PurchasePlan),
    optimal_plan.num_A + optimal_plan.num_B = 400 ∧
    total_cost prices optimal_plan ≤ 2800 ∧
    total_profit prices optimal_plan = 900 ∧
    ∀ (plan : PurchasePlan),
      plan.num_A + plan.num_B = 400 →
      total_cost prices plan ≤ 2800 →
      total_profit prices plan ≤ total_profit prices optimal_plan :=
by sorry


end zoo_badge_problem_l2156_215651


namespace total_songs_in_june_l2156_215657

def june_days : ℕ := 30
def weekend_days : ℕ := 8
def holiday_days : ℕ := 1
def vivian_songs_per_day : ℕ := 10
def clara_songs_per_day : ℕ := vivian_songs_per_day - 2
def lucas_songs_per_day : ℕ := vivian_songs_per_day + 5

theorem total_songs_in_june :
  let playing_days : ℕ := june_days - weekend_days - holiday_days
  let vivian_total : ℕ := playing_days * vivian_songs_per_day
  let clara_total : ℕ := playing_days * clara_songs_per_day
  let lucas_total : ℕ := playing_days * lucas_songs_per_day
  vivian_total + clara_total + lucas_total = 693 := by
  sorry

end total_songs_in_june_l2156_215657


namespace volleyball_league_female_fraction_l2156_215658

theorem volleyball_league_female_fraction 
  (last_year_male : ℕ)
  (total_increase : ℝ)
  (male_increase : ℝ)
  (female_increase : ℝ)
  (h1 : last_year_male = 30)
  (h2 : total_increase = 0.15)
  (h3 : male_increase = 0.10)
  (h4 : female_increase = 0.25) :
  let this_year_male : ℝ := last_year_male * (1 + male_increase)
  let last_year_female : ℝ := last_year_male * (1 + total_increase) / (2 + male_increase + female_increase) - last_year_male
  let this_year_female : ℝ := last_year_female * (1 + female_increase)
  let total_this_year : ℝ := this_year_male + this_year_female
  (this_year_female / total_this_year) = 25 / 47 := by
sorry

end volleyball_league_female_fraction_l2156_215658


namespace multiple_problem_l2156_215611

theorem multiple_problem (x : ℝ) (m : ℝ) (h1 : x = -4.5) (h2 : 10 * x = m * x - 36) : m = 2 := by
  sorry

end multiple_problem_l2156_215611


namespace smallest_positive_integer_ending_in_3_divisible_by_11_l2156_215615

theorem smallest_positive_integer_ending_in_3_divisible_by_11 : 
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 10 = 3 → m % 11 = 0 → m ≥ n :=
by
  use 33
  sorry

end smallest_positive_integer_ending_in_3_divisible_by_11_l2156_215615


namespace cubic_roots_sum_l2156_215683

theorem cubic_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 94 :=
by sorry

end cubic_roots_sum_l2156_215683


namespace min_buses_for_given_route_l2156_215647

/-- Represents the bus route configuration -/
structure BusRoute where
  one_way_time : ℕ
  stop_time : ℕ
  departure_interval : ℕ

/-- Calculates the minimum number of buses required for a given bus route -/
def min_buses_required (route : BusRoute) : ℕ :=
  let round_trip_time := 2 * (route.one_way_time + route.stop_time)
  (round_trip_time / route.departure_interval)

/-- Theorem stating that the minimum number of buses required for the given conditions is 20 -/
theorem min_buses_for_given_route :
  let route := BusRoute.mk 50 10 6
  min_buses_required route = 20 := by
  sorry

#eval min_buses_required (BusRoute.mk 50 10 6)

end min_buses_for_given_route_l2156_215647


namespace smallest_cube_ending_544_l2156_215601

theorem smallest_cube_ending_544 :
  ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 544 ∧ ∀ (m : ℕ), m > 0 ∧ m^3 % 1000 = 544 → n ≤ m :=
by sorry

end smallest_cube_ending_544_l2156_215601


namespace seeds_planted_wednesday_l2156_215607

/-- The number of seeds planted on Wednesday -/
def seeds_wednesday : ℕ := sorry

/-- The number of seeds planted on Thursday -/
def seeds_thursday : ℕ := 2

/-- The total number of seeds planted -/
def total_seeds : ℕ := 22

/-- Theorem stating that the number of seeds planted on Wednesday is 20 -/
theorem seeds_planted_wednesday :
  seeds_wednesday = total_seeds - seeds_thursday ∧ seeds_wednesday = 20 := by sorry

end seeds_planted_wednesday_l2156_215607


namespace remaining_pages_l2156_215622

/-- Calculates the remaining pages in a pad after various projects --/
theorem remaining_pages (initial_pages : ℕ) : 
  initial_pages = 120 → 
  (initial_pages / 2 - 
   (initial_pages / 4 + 10 + initial_pages * 15 / 100) / 2) = 31 := by
  sorry

end remaining_pages_l2156_215622


namespace exists_prime_not_cube_root_l2156_215663

theorem exists_prime_not_cube_root (p q : ℕ) : 
  ∃ q : ℕ, Prime q ∧ ∀ p : ℕ, Prime p → ¬∃ n : ℕ, n^3 = p^2 + q :=
sorry

end exists_prime_not_cube_root_l2156_215663


namespace sasha_can_buy_everything_l2156_215636

-- Define the store's discount policy and item prices
def discount_threshold : ℝ := 1500
def discount_rate : ℝ := 0.26
def shashlik_price : ℝ := 350
def sauce_price : ℝ := 70

-- Define Sasha's budget and desired quantities
def budget : ℝ := 1800
def shashlik_quantity : ℝ := 5
def sauce_quantity : ℝ := 1

-- Define a function to calculate the discounted price
def discounted_price (price : ℝ) : ℝ := price * (1 - discount_rate)

-- Theorem: Sasha can buy everything he planned within his budget
theorem sasha_can_buy_everything :
  ∃ (first_shashlik second_shashlik first_sauce : ℝ),
    first_shashlik + second_shashlik = shashlik_quantity ∧
    first_sauce = sauce_quantity ∧
    first_shashlik * shashlik_price + first_sauce * sauce_price ≥ discount_threshold ∧
    (first_shashlik * shashlik_price + first_sauce * sauce_price) +
    (second_shashlik * (discounted_price shashlik_price)) ≤ budget :=
  sorry

end sasha_can_buy_everything_l2156_215636


namespace original_number_l2156_215680

theorem original_number (x : ℝ) : x * 1.1 = 660 ↔ x = 600 := by
  sorry

end original_number_l2156_215680


namespace greatest_int_prime_abs_quadratic_l2156_215696

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def f (x : ℤ) : ℤ := |4*x^2 - 39*x + 21|

theorem greatest_int_prime_abs_quadratic : 
  ∀ x : ℤ, x > 8 → ¬(is_prime (f x).toNat) ∧ is_prime (f 8).toNat :=
sorry

end greatest_int_prime_abs_quadratic_l2156_215696


namespace even_function_implies_A_equals_one_l2156_215631

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = (x+1)(x-A) -/
def f (A : ℝ) (x : ℝ) : ℝ :=
  (x + 1) * (x - A)

/-- If f(x) = (x+1)(x-A) is an even function, then A = 1 -/
theorem even_function_implies_A_equals_one :
  ∀ A : ℝ, IsEven (f A) → A = 1 := by
  sorry

end even_function_implies_A_equals_one_l2156_215631


namespace third_shape_symmetric_l2156_215640

-- Define a type for F-like shapes
inductive FLikeShape
| first
| second
| third
| fourth
| fifth

-- Define a function to check if a shape has reflection symmetry
def has_reflection_symmetry (shape : FLikeShape) : Prop :=
  match shape with
  | FLikeShape.third => True
  | _ => False

-- Theorem statement
theorem third_shape_symmetric :
  ∃ (shape : FLikeShape), has_reflection_symmetry shape ∧ shape = FLikeShape.third :=
by
  sorry

#check third_shape_symmetric

end third_shape_symmetric_l2156_215640


namespace negative_expression_l2156_215638

theorem negative_expression (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 1) : 
  b + 3 * b^2 < 0 :=
by sorry

end negative_expression_l2156_215638


namespace initial_oak_trees_l2156_215616

theorem initial_oak_trees (final_trees : ℕ) (cut_trees : ℕ) : final_trees = 7 → cut_trees = 2 → final_trees + cut_trees = 9 := by
  sorry

end initial_oak_trees_l2156_215616


namespace factorization_theorem_l2156_215603

theorem factorization_theorem (a b c : ℝ) :
  ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = (a + b) * (b + c) * (c + a) :=
by sorry

end factorization_theorem_l2156_215603


namespace price_quantity_change_l2156_215608

theorem price_quantity_change (original_price original_quantity : ℝ) :
  let price_increase_factor := 1.20
  let quantity_decrease_factor := 0.70
  let new_cost := original_price * price_increase_factor * original_quantity * quantity_decrease_factor
  let original_cost := original_price * original_quantity
  new_cost / original_cost = 0.84 :=
by sorry

end price_quantity_change_l2156_215608


namespace complex_modulus_sum_l2156_215682

theorem complex_modulus_sum : Complex.abs (3 - 3*I) + Complex.abs (3 + 3*I) = 6 * Real.sqrt 2 := by
  sorry

end complex_modulus_sum_l2156_215682


namespace only_one_solves_l2156_215655

def prob_A : ℚ := 1/2
def prob_B : ℚ := 1/3
def prob_C : ℚ := 1/4

theorem only_one_solves : 
  let prob_only_one := 
    (prob_A * (1 - prob_B) * (1 - prob_C)) + 
    ((1 - prob_A) * prob_B * (1 - prob_C)) + 
    ((1 - prob_A) * (1 - prob_B) * prob_C)
  prob_only_one = 11/24 := by
  sorry

end only_one_solves_l2156_215655


namespace candied_apples_count_l2156_215600

/-- The number of candied apples that were made -/
def num_apples : ℕ := 15

/-- The price of each candied apple in dollars -/
def apple_price : ℚ := 2

/-- The number of candied grapes -/
def num_grapes : ℕ := 12

/-- The price of each candied grape in dollars -/
def grape_price : ℚ := 3/2

/-- The total earnings from selling all items in dollars -/
def total_earnings : ℚ := 48

theorem candied_apples_count :
  num_apples * apple_price + num_grapes * grape_price = total_earnings :=
sorry

end candied_apples_count_l2156_215600


namespace two_colors_probability_l2156_215619

/-- The number of black balls in the bin -/
def black_balls : ℕ := 10

/-- The number of white balls in the bin -/
def white_balls : ℕ := 8

/-- The number of red balls in the bin -/
def red_balls : ℕ := 6

/-- The total number of balls in the bin -/
def total_balls : ℕ := black_balls + white_balls + red_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 4

/-- The probability of drawing 2 balls of one color and 2 balls of another color -/
theorem two_colors_probability : 
  (Nat.choose black_balls 2 * Nat.choose white_balls 2 + 
   Nat.choose black_balls 2 * Nat.choose red_balls 2 + 
   Nat.choose white_balls 2 * Nat.choose red_balls 2) / 
  Nat.choose total_balls drawn_balls = 157 / 845 := by sorry

end two_colors_probability_l2156_215619


namespace no_six_if_mean_and_median_two_l2156_215681

/-- Represents the result of 5 dice rolls -/
def DiceRolls := Fin 5 → Nat

/-- The mean of the dice rolls is 2 -/
def mean_is_2 (rolls : DiceRolls) : Prop :=
  (rolls 0 + rolls 1 + rolls 2 + rolls 3 + rolls 4) / 5 = 2

/-- The median of the dice rolls is 2 -/
def median_is_2 (rolls : DiceRolls) : Prop :=
  ∃ (p : Equiv (Fin 5) (Fin 5)), 
    rolls (p 2) = 2 ∧ 
    (∀ i < 2, rolls (p i) ≤ 2) ∧ 
    (∀ i > 2, rolls (p i) ≥ 2)

/-- The theorem stating that if the mean and median are 2, then 6 cannot appear in the rolls -/
theorem no_six_if_mean_and_median_two (rolls : DiceRolls) 
  (h_mean : mean_is_2 rolls) (h_median : median_is_2 rolls) : 
  ∀ i, rolls i ≠ 6 := by
  sorry

end no_six_if_mean_and_median_two_l2156_215681


namespace expression_evaluation_l2156_215675

theorem expression_evaluation : 4 * (-3) + 60 / (-15) = -16 := by
  sorry

end expression_evaluation_l2156_215675


namespace power_calculation_l2156_215692

theorem power_calculation : (16^6 * 8^3) / 4^11 = 2048 := by
  sorry

end power_calculation_l2156_215692


namespace dividend_divisor_properties_l2156_215684

theorem dividend_divisor_properties : ∃ (dividend : Nat) (divisor : Nat),
  dividend = 957 ∧ divisor = 75 ∧
  (dividend / divisor = (divisor / 10 + divisor % 10)) ∧
  (dividend % divisor = 57) ∧
  ((dividend % divisor) * (dividend / divisor) + divisor = 759) := by
  sorry

end dividend_divisor_properties_l2156_215684


namespace unique_perfect_cube_l2156_215637

theorem unique_perfect_cube (Z K : ℤ) : 
  (1000 < Z) → (Z < 1500) → (K > 1) → (Z = K^3) → 
  (∃! k : ℤ, k > 1 ∧ 1000 < k^3 ∧ k^3 < 1500 ∧ Z = k^3) ∧ (K = 11) := by
sorry

end unique_perfect_cube_l2156_215637


namespace meal_sales_tax_percentage_l2156_215625

/-- The maximum total spending allowed for the meal -/
def total_limit : ℝ := 50

/-- The maximum cost of food allowed -/
def max_food_cost : ℝ := 40.98

/-- The tip percentage as a decimal -/
def tip_percentage : ℝ := 0.15

/-- The maximum sales tax percentage that satisfies the conditions -/
def max_sales_tax_percentage : ℝ := 6.1

/-- Theorem stating that the maximum sales tax percentage is approximately 6.1% -/
theorem meal_sales_tax_percentage :
  ∀ (sales_tax_percentage : ℝ),
    sales_tax_percentage ≤ max_sales_tax_percentage →
    max_food_cost + (sales_tax_percentage / 100 * max_food_cost) +
    (tip_percentage * (max_food_cost + (sales_tax_percentage / 100 * max_food_cost))) ≤ total_limit :=
by sorry

end meal_sales_tax_percentage_l2156_215625


namespace base_conversion_1357_to_base_5_l2156_215641

theorem base_conversion_1357_to_base_5 :
  (2 * 5^4 + 0 * 5^3 + 4 * 5^2 + 1 * 5^1 + 2 * 5^0 : ℕ) = 1357 := by
  sorry

#eval 2 * 5^4 + 0 * 5^3 + 4 * 5^2 + 1 * 5^1 + 2 * 5^0

end base_conversion_1357_to_base_5_l2156_215641


namespace isosceles_triangle_side_lengths_l2156_215623

/-- Represents a triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

/-- Represents an isosceles triangle. -/
structure IsoscelesTriangle extends Triangle where
  isosceles : (a = b) ∨ (b = c) ∨ (c = a)

/-- 
Given an isosceles triangle with perimeter 17 and one side length 4,
prove that the other two sides must both be 6.5.
-/
theorem isosceles_triangle_side_lengths :
  ∀ t : IsoscelesTriangle,
    t.a + t.b + t.c = 17 →
    (t.a = 4 ∨ t.b = 4 ∨ t.c = 4) →
    ((t.a = 6.5 ∧ t.b = 6.5 ∧ t.c = 4) ∨
     (t.a = 6.5 ∧ t.b = 4 ∧ t.c = 6.5) ∨
     (t.a = 4 ∧ t.b = 6.5 ∧ t.c = 6.5)) := by
  sorry

end isosceles_triangle_side_lengths_l2156_215623


namespace bird_nest_problem_l2156_215628

theorem bird_nest_problem (birds : ℕ) (nests : ℕ) 
  (h1 : birds = 6) 
  (h2 : birds = nests + 3) : 
  nests = 3 := by
  sorry

end bird_nest_problem_l2156_215628


namespace students_taking_neither_music_nor_art_l2156_215660

theorem students_taking_neither_music_nor_art 
  (total_students : ℕ) 
  (music_students : ℕ) 
  (art_students : ℕ) 
  (both_students : ℕ) 
  (h1 : total_students = 500)
  (h2 : music_students = 40)
  (h3 : art_students = 20)
  (h4 : both_students = 10)
  : total_students - (music_students + art_students - both_students) = 450 :=
by
  sorry

#check students_taking_neither_music_nor_art

end students_taking_neither_music_nor_art_l2156_215660


namespace sum_of_coefficients_l2156_215687

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (3*x - 2)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = -63 := by
sorry

end sum_of_coefficients_l2156_215687


namespace average_age_decrease_l2156_215685

theorem average_age_decrease (original_strength : ℕ) (original_avg_age : ℝ) 
  (new_students : ℕ) (new_avg_age : ℝ) : 
  original_strength = 17 →
  original_avg_age = 40 →
  new_students = 17 →
  new_avg_age = 32 →
  let new_total_strength := original_strength + new_students
  let new_avg_age := (original_strength * original_avg_age + new_students * new_avg_age) / new_total_strength
  original_avg_age - new_avg_age = 4 := by
sorry

end average_age_decrease_l2156_215685


namespace expression_evaluation_l2156_215639

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The product of powers of x from 1 to n -/
def prod_powers (x : ℕ) (n : ℕ) : ℕ := x ^ sum_first_n n

/-- The product of powers of x for multiples of 3 up to 3n -/
def prod_powers_mult3 (x : ℕ) (n : ℕ) : ℕ := x ^ (3 * sum_first_n n)

theorem expression_evaluation (x : ℕ) (hx : x = 3) :
  prod_powers x 20 / prod_powers_mult3 x 10 = x ^ 45 := by
  sorry

end expression_evaluation_l2156_215639


namespace isosceles_triangle_perimeter_l2156_215613

/-- An isosceles triangle with two sides of length 6 and one side of length 2 has a perimeter of 14. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 6 ∧ b = 6 ∧ c = 2 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  a + b + c = 14 :=
by
  sorry

#check isosceles_triangle_perimeter

end isosceles_triangle_perimeter_l2156_215613


namespace danny_in_position_three_l2156_215620

-- Define the people
inductive Person : Type
| Amelia : Person
| Blake : Person
| Claire : Person
| Danny : Person

-- Define the positions
inductive Position : Type
| One : Position
| Two : Position
| Three : Position
| Four : Position

-- Define the seating arrangement
def Seating := Person → Position

-- Define opposite positions
def opposite (p : Position) : Position :=
  match p with
  | Position.One => Position.Three
  | Position.Two => Position.Four
  | Position.Three => Position.One
  | Position.Four => Position.Two

-- Define adjacent positions
def adjacent (p1 p2 : Position) : Prop :=
  (p1 = Position.One ∧ p2 = Position.Two) ∨
  (p1 = Position.Two ∧ p2 = Position.Three) ∨
  (p1 = Position.Three ∧ p2 = Position.Four) ∨
  (p1 = Position.Four ∧ p2 = Position.One) ∨
  (p2 = Position.One ∧ p1 = Position.Two) ∨
  (p2 = Position.Two ∧ p1 = Position.Three) ∨
  (p2 = Position.Three ∧ p1 = Position.Four) ∨
  (p2 = Position.Four ∧ p1 = Position.One)

-- Define between positions
def between (p1 p2 p3 : Position) : Prop :=
  (adjacent p1 p2 ∧ adjacent p2 p3) ∨
  (adjacent p3 p1 ∧ adjacent p1 p2)

-- Theorem statement
theorem danny_in_position_three 
  (s : Seating)
  (claire_in_one : s Person.Claire = Position.One)
  (not_blake_opposite_claire : s Person.Blake ≠ opposite (s Person.Claire))
  (not_amelia_between_blake_claire : ¬ between (s Person.Blake) (s Person.Amelia) (s Person.Claire)) :
  s Person.Danny = Position.Three :=
by sorry

end danny_in_position_three_l2156_215620


namespace semicircle_pattern_area_l2156_215630

/-- The area of the shaded region formed by semicircles in a pattern -/
theorem semicircle_pattern_area (pattern_length : ℝ) (semicircle_diameter : ℝ) :
  pattern_length = 18 →
  semicircle_diameter = 3 →
  let num_semicircles : ℝ := pattern_length / semicircle_diameter
  let num_full_circles : ℝ := num_semicircles / 2
  let circle_radius : ℝ := semicircle_diameter / 2
  pattern_length > 0 →
  semicircle_diameter > 0 →
  (num_full_circles * π * circle_radius^2) = (27 / 4) * π :=
by sorry

end semicircle_pattern_area_l2156_215630


namespace bead_arrangement_probability_l2156_215632

def total_beads : ℕ := 6
def red_beads : ℕ := 3
def white_beads : ℕ := 2
def blue_beads : ℕ := 1

def total_arrangements : ℕ := (Nat.factorial total_beads) / ((Nat.factorial red_beads) * (Nat.factorial white_beads) * (Nat.factorial blue_beads))

def valid_arrangements : ℕ := 10

theorem bead_arrangement_probability :
  (valid_arrangements : ℚ) / total_arrangements = 1 / 6 := by
  sorry

end bead_arrangement_probability_l2156_215632


namespace compare_fractions_l2156_215699

theorem compare_fractions :
  (-7/2 : ℚ) < (-7/3 : ℚ) ∧ (-3/4 : ℚ) > (-4/5 : ℚ) := by sorry

end compare_fractions_l2156_215699


namespace sin_pi_over_six_l2156_215626

theorem sin_pi_over_six : Real.sin (π / 6) = 1 / 2 := by sorry

end sin_pi_over_six_l2156_215626


namespace weekly_rental_cost_l2156_215674

/-- The weekly rental cost of a parking space, given the monthly cost,
    yearly savings, and number of months and weeks in a year. -/
theorem weekly_rental_cost (monthly_cost : ℕ) (yearly_savings : ℕ) 
                            (months_per_year : ℕ) (weeks_per_year : ℕ) :
  monthly_cost = 42 →
  yearly_savings = 16 →
  months_per_year = 12 →
  weeks_per_year = 52 →
  (monthly_cost * months_per_year + yearly_savings) / weeks_per_year = 10 :=
by
  sorry

end weekly_rental_cost_l2156_215674


namespace function_property_implies_zero_l2156_215652

theorem function_property_implies_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f (y^2) = f (x^2 + y)) : 
  f (-2017) = 0 := by
  sorry

end function_property_implies_zero_l2156_215652


namespace max_area_difference_rectangles_l2156_215646

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Theorem: The maximum difference between areas of two rectangles with perimeter 144 is 1225 -/
theorem max_area_difference_rectangles :
  ∃ (r1 r2 : Rectangle),
    perimeter r1 = 144 ∧
    perimeter r2 = 144 ∧
    ∀ (r3 r4 : Rectangle),
      perimeter r3 = 144 →
      perimeter r4 = 144 →
      area r1 - area r2 ≥ area r3 - area r4 ∧
      area r1 - area r2 = 1225 :=
sorry

end max_area_difference_rectangles_l2156_215646


namespace total_apples_l2156_215661

theorem total_apples (cecile_apples diane_apples : ℕ) : 
  cecile_apples = 15 → 
  diane_apples = cecile_apples + 20 → 
  cecile_apples + diane_apples = 50 := by
sorry

end total_apples_l2156_215661


namespace hash_example_l2156_215665

def hash (a b c d : ℝ) : ℝ := d * b^2 - 5 * a * c

theorem hash_example : hash 2 3 1 4 = 26 := by
  sorry

end hash_example_l2156_215665


namespace vector_equality_implies_coordinates_l2156_215671

/-- Given four points A, B, C, D in a plane, where vector AB equals vector CD,
    prove that the coordinates of C and D satisfy specific values. -/
theorem vector_equality_implies_coordinates (A B C D : ℝ × ℝ) :
  A = (1, 2) →
  B = (5, 4) →
  C.2 = 3 →
  D.1 = -3 →
  B.1 - A.1 = D.1 - C.1 →
  B.2 - A.2 = D.2 - C.2 →
  C.1 = -7 ∧ D.2 = 5 := by
sorry

end vector_equality_implies_coordinates_l2156_215671


namespace student_selection_properties_l2156_215693

/-- Represents the selection of students from different grades -/
structure StudentSelection where
  total : Nat
  first_year : Nat
  second_year : Nat
  third_year : Nat
  selected : Nat

/-- Calculate the probability of selecting students from different grades -/
def prob_different_grades (s : StudentSelection) : Rat :=
  (s.first_year.choose 1 * s.second_year.choose 1 * s.third_year.choose 1) /
  (s.total.choose s.selected)

/-- Calculate the mathematical expectation of the number of first-year students selected -/
def expectation_first_year (s : StudentSelection) : Rat :=
  (0 * (s.total - s.first_year).choose s.selected +
   1 * (s.first_year.choose 1 * (s.total - s.first_year).choose (s.selected - 1)) +
   2 * (s.first_year.choose 2 * (s.total - s.first_year).choose (s.selected - 2))) /
  (s.total.choose s.selected)

/-- The main theorem stating the properties of the student selection problem -/
theorem student_selection_properties (s : StudentSelection) 
  (h1 : s.total = 5)
  (h2 : s.first_year = 2)
  (h3 : s.second_year = 2)
  (h4 : s.third_year = 1)
  (h5 : s.selected = 3) :
  prob_different_grades s = 2/5 ∧ expectation_first_year s = 6/5 := by
  sorry

#eval prob_different_grades ⟨5, 2, 2, 1, 3⟩
#eval expectation_first_year ⟨5, 2, 2, 1, 3⟩

end student_selection_properties_l2156_215693


namespace equilateral_condition_obtuse_condition_two_triangles_condition_l2156_215678

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the properties we need to prove
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

def isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi / 2 ∨ t.B > Real.pi / 2 ∨ t.C > Real.pi / 2

def hasTwoConfigurations (a b : ℝ) (B : ℝ) : Prop :=
  ∃ (C₁ C₂ : ℝ), C₁ ≠ C₂ ∧
    (∃ (t₁ t₂ : Triangle), 
      t₁.a = a ∧ t₁.b = b ∧ t₁.B = B ∧ t₁.C = C₁ ∧
      t₂.a = a ∧ t₂.b = b ∧ t₂.B = B ∧ t₂.C = C₂)

-- State the theorems
theorem equilateral_condition (t : Triangle) 
  (h1 : t.b^2 = t.a * t.c) (h2 : t.B = Real.pi / 3) : 
  isEquilateral t := by sorry

theorem obtuse_condition (t : Triangle) 
  (h : Real.cos t.A^2 + Real.sin t.B^2 + Real.sin t.C^2 < 1) : 
  isObtuse t := by sorry

theorem two_triangles_condition :
  hasTwoConfigurations 4 2 (25 * Real.pi / 180) := by sorry

end equilateral_condition_obtuse_condition_two_triangles_condition_l2156_215678


namespace inequality_preservation_l2156_215624

theorem inequality_preservation (a b : ℝ) (h : a < b) : 2 - a > 2 - b := by
  sorry

end inequality_preservation_l2156_215624


namespace process_output_for_4_l2156_215670

/-- A function representing the process described in the flowchart --/
def process (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the process outputs 3 when given input 4 --/
theorem process_output_for_4 : process 4 = 3 := by
  sorry

end process_output_for_4_l2156_215670


namespace sum_in_range_l2156_215612

theorem sum_in_range : ∃ (s : ℚ), 
  s = 3 + 1/8 + 4 + 1/3 + 6 + 1/21 ∧ 13 < s ∧ s < 14.5 := by
  sorry

end sum_in_range_l2156_215612


namespace arithmetic_progression_sum_l2156_215686

/-- 
Given an arithmetic progression where the sum of the 4th and 12th terms is 8,
prove that the sum of the first 15 terms is 60.
-/
theorem arithmetic_progression_sum (a d : ℝ) : 
  (a + 3*d) + (a + 11*d) = 8 → 
  (15 : ℝ) / 2 * (2*a + 14*d) = 60 := by
sorry

end arithmetic_progression_sum_l2156_215686


namespace lcm_problem_l2156_215654

theorem lcm_problem (d n : ℕ) : 
  d > 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  Nat.lcm d n = 690 ∧ 
  ¬(3 ∣ n) ∧ 
  ¬(2 ∣ d) →
  n = 230 := by sorry

end lcm_problem_l2156_215654


namespace inverse_217_mod_397_l2156_215633

theorem inverse_217_mod_397 : ∃ a : ℤ, 0 ≤ a ∧ a < 397 ∧ (217 * a) % 397 = 1 :=
by
  use 161
  sorry

end inverse_217_mod_397_l2156_215633


namespace bus_empty_seats_l2156_215679

/-- Calculates the number of empty seats on a bus after a series of boarding and disembarking events -/
def empty_seats_after_events (rows : ℕ) (seats_per_row : ℕ) 
  (initial_boarding : ℕ) 
  (stop1_board : ℕ) (stop1_disembark : ℕ)
  (stop2_board : ℕ) (stop2_disembark : ℕ)
  (stop3_board : ℕ) (stop3_disembark : ℕ) : ℕ :=
  let total_seats := rows * seats_per_row
  let after_initial := total_seats - initial_boarding
  let after_stop1 := after_initial - (stop1_board - stop1_disembark)
  let after_stop2 := after_stop1 - (stop2_board - stop2_disembark)
  let after_stop3 := after_stop2 - (stop3_board - stop3_disembark)
  after_stop3

theorem bus_empty_seats : 
  empty_seats_after_events 23 4 16 15 3 17 10 12 8 = 53 := by
  sorry

end bus_empty_seats_l2156_215679


namespace distance_between_squares_l2156_215643

/-- Given two squares where:
  * The smaller square has a perimeter of 8 cm
  * The larger square has an area of 64 cm²
  * The bottom left corner of the larger square is 2 cm to the right of the top right corner of the smaller square
  Prove that the distance between the top right corner of the larger square (A) and 
  the top left corner of the smaller square (B) is √136 cm -/
theorem distance_between_squares (small_perimeter : ℝ) (large_area : ℝ) (horizontal_shift : ℝ) :
  small_perimeter = 8 →
  large_area = 64 →
  horizontal_shift = 2 →
  let small_side := small_perimeter / 4
  let large_side := Real.sqrt large_area
  let horizontal_distance := horizontal_shift + large_side
  let vertical_distance := large_side - small_side
  Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2) = Real.sqrt 136 :=
by sorry

end distance_between_squares_l2156_215643


namespace quadratic_monotonicity_l2156_215650

theorem quadratic_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Ioo 2 3, Monotone (fun x => x^2 - 2*a*x + 1) ∨ StrictMono (fun x => x^2 - 2*a*x + 1)) ↔
  (a ≤ 2 ∨ a ≥ 3) :=
by sorry

end quadratic_monotonicity_l2156_215650


namespace complex_equation_solution_l2156_215672

theorem complex_equation_solution (z : ℂ) : (1 - 2*I)*z = Complex.abs (3 + 4*I) → z = 1 + 2*I := by
  sorry

end complex_equation_solution_l2156_215672


namespace tangent_at_P_tangent_through_P_l2156_215688

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Theorem for the tangent line at P
theorem tangent_at_P :
  let (x₀, y₀) := P
  let slope := (3 * x₀^2 - 3 : ℝ)
  (∀ x, f x₀ + slope * (x - x₀) = -2) :=
sorry

-- Theorem for the tangent lines passing through P
theorem tangent_through_P :
  let (x₀, y₀) := P
  (∃ x₁ : ℝ, 
    (∀ x, f x₁ + (3 * x₁^2 - 3) * (x - x₁) = -2) ∨
    (∀ x, f x₁ + (3 * x₁^2 - 3) * (x - x₁) = -9/4*x + 1/4)) :=
sorry

end tangent_at_P_tangent_through_P_l2156_215688


namespace consecutive_integers_sqrt_eight_l2156_215627

theorem consecutive_integers_sqrt_eight (a b : ℤ) : 
  (a < Real.sqrt 8 ∧ Real.sqrt 8 < b) → 
  (b = a + 1) → 
  (b ^ a : ℝ) = 9 := by
sorry

end consecutive_integers_sqrt_eight_l2156_215627


namespace stationary_rigid_body_l2156_215642

/-- A point in a two-dimensional plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A rigid body in a two-dimensional plane -/
structure RigidBody2D where
  points : Set Point2D

/-- Three points are non-collinear if they do not lie on the same straight line -/
def NonCollinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) ≠ (p3.x - p1.x) * (p2.y - p1.y)

/-- A rigid body is stationary if it has no translational or rotational motion -/
def IsStationary (body : RigidBody2D) : Prop :=
  ∃ (p1 p2 p3 : Point2D), p1 ∈ body.points ∧ p2 ∈ body.points ∧ p3 ∈ body.points ∧
    NonCollinear p1 p2 p3

theorem stationary_rigid_body (body : RigidBody2D) :
  IsStationary body ↔ ∃ (p1 p2 p3 : Point2D), p1 ∈ body.points ∧ p2 ∈ body.points ∧ p3 ∈ body.points ∧
    NonCollinear p1 p2 p3 :=
  sorry

end stationary_rigid_body_l2156_215642


namespace construct_square_and_dodecagon_l2156_215689

/-- A point in a 2D plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A circle in a 2D plane --/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a compass --/
structure Compass :=
  (create_circle : Point → ℝ → Circle)

/-- Represents a square --/
structure Square :=
  (vertices : Fin 4 → Point)

/-- Represents a regular dodecagon --/
structure RegularDodecagon :=
  (vertices : Fin 12 → Point)

/-- Theorem stating that a square and a regular dodecagon can be constructed using only a compass --/
theorem construct_square_and_dodecagon 
  (A B : Point) 
  (compass : Compass) : 
  ∃ (square : Square) (dodecagon : RegularDodecagon),
    (square.vertices 0 = A ∧ square.vertices 1 = B) ∧
    (dodecagon.vertices 0 = A ∧ dodecagon.vertices 1 = B) :=
sorry

end construct_square_and_dodecagon_l2156_215689


namespace range_of_m_l2156_215621

/-- Given a function f with derivative f', we define g and prove a property about m. -/
theorem range_of_m (f : ℝ → ℝ) (f' : ℝ → ℝ) (g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, HasDerivAt f (f' x) x) →  -- f has derivative f' for all x
  (∀ x, g x = f x - (1/2) * x^2) →  -- definition of g
  (∀ x, f' x < x) →  -- condition on f'
  (f (4 - m) - f m ≥ 8 - 4*m) →  -- given inequality
  m ≥ 2 :=  -- conclusion: m is in [2, +∞)
sorry

end range_of_m_l2156_215621


namespace tan_2alpha_values_l2156_215698

theorem tan_2alpha_values (α : Real) (h : 2 * Real.sin (2 * α) = 1 + Real.cos (2 * α)) :
  Real.tan (2 * α) = 4 / 3 ∨ Real.tan (2 * α) = 0 := by
  sorry

end tan_2alpha_values_l2156_215698


namespace fish_tank_balls_l2156_215677

/-- The total number of balls in a fish tank with goldfish, platyfish, and angelfish -/
def total_balls (goldfish platyfish angelfish : ℕ) 
                (goldfish_balls platyfish_balls angelfish_balls : ℚ) : ℚ :=
  (goldfish : ℚ) * goldfish_balls + 
  (platyfish : ℚ) * platyfish_balls + 
  (angelfish : ℚ) * angelfish_balls

/-- Theorem stating the total number of balls in the fish tank -/
theorem fish_tank_balls : 
  total_balls 5 8 4 12.5 7.5 4.5 = 140.5 := by
  sorry

end fish_tank_balls_l2156_215677


namespace quadratic_minimum_l2156_215669

/-- The quadratic function f(x) = x^2 - 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

theorem quadratic_minimum :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ x = 4 :=
by sorry

end quadratic_minimum_l2156_215669


namespace james_money_calculation_l2156_215659

theorem james_money_calculation (num_bills : ℕ) (bill_value : ℕ) (existing_amount : ℕ) : 
  num_bills = 3 → bill_value = 20 → existing_amount = 75 → 
  num_bills * bill_value + existing_amount = 135 := by
  sorry

end james_money_calculation_l2156_215659


namespace tuesday_flower_sales_l2156_215618

/-- The number of flowers sold by Ginger on Tuesday -/
def total_flowers (lilacs roses gardenias : ℕ) : ℕ :=
  lilacs + roses + gardenias

/-- Theorem stating the total number of flowers sold on Tuesday -/
theorem tuesday_flower_sales :
  ∀ (lilacs roses gardenias : ℕ),
    lilacs = 10 →
    roses = 3 * lilacs →
    gardenias = lilacs / 2 →
    total_flowers lilacs roses gardenias = 45 :=
by
  sorry


end tuesday_flower_sales_l2156_215618


namespace min_value_inequality_l2156_215614

theorem min_value_inequality (r s t : ℝ) (h1 : 1 ≤ r) (h2 : r ≤ s) (h3 : s ≤ t) (h4 : t ≤ 4) :
  (r - 1)^2 + (s/r - 1)^2 + (t/s - 1)^2 + (4/t - 1)^2 ≥ 4 * (Real.sqrt 2 - 1)^2 := by
  sorry

end min_value_inequality_l2156_215614


namespace polynomial_coefficient_equality_l2156_215666

theorem polynomial_coefficient_equality : 
  ∃! (a b c : ℝ), ∀ (x : ℝ), 
    2*x^4 + x^3 - 41*x^2 + 83*x - 45 = (a*x^2 + b*x + c)*(x^2 + 4*x + 9) ∧ 
    a = 2 ∧ b = -7 ∧ c = -5 := by sorry

end polynomial_coefficient_equality_l2156_215666


namespace pie_price_is_seven_l2156_215609

def number_of_cakes : ℕ := 453
def price_per_cake : ℕ := 12
def number_of_pies : ℕ := 126
def total_earnings : ℕ := 6318

theorem pie_price_is_seven :
  ∃ (price_per_pie : ℕ),
    price_per_pie = 7 ∧
    price_per_pie * number_of_pies + price_per_cake * number_of_cakes = total_earnings :=
by sorry

end pie_price_is_seven_l2156_215609


namespace f_range_l2156_215634

noncomputable def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 5*x + 2) / (x + 1)

theorem f_range :
  Set.range f = {y : ℝ | y > 0} := by sorry

end f_range_l2156_215634


namespace park_area_l2156_215629

/-- Proves that a rectangular park with sides in ratio 3:2 and fencing cost of $225 at 90 ps per meter has an area of 3750 square meters -/
theorem park_area (length width perimeter cost_per_meter total_cost : ℝ) : 
  length / width = 3 / 2 →
  perimeter = 2 * (length + width) →
  cost_per_meter = 0.9 →
  total_cost = 225 →
  total_cost = perimeter * cost_per_meter →
  length * width = 3750 :=
by sorry

end park_area_l2156_215629


namespace bus_journey_distance_l2156_215690

/-- Represents the distance traveled by a bus after k hours, given a total journey of 100 km -/
def distance_traveled (k : ℕ) : ℚ :=
  (100 * k) / (k + 1)

/-- Theorem stating that after 6 hours, the distance traveled is 600/7 km -/
theorem bus_journey_distance :
  distance_traveled 6 = 600 / 7 := by
  sorry

end bus_journey_distance_l2156_215690


namespace min_abs_phi_l2156_215604

/-- Given a function f(x) = 2sin(ωx + φ) with ω > 0, prove that the minimum value of |φ| is π/2 
    under the following conditions:
    1. Three consecutive intersection points with y = b (0 < b < 2) are at x = π/6, 5π/6, 7π/6
    2. f(x) reaches its minimum value at x = 3π/2 -/
theorem min_abs_phi (ω : ℝ) (φ : ℝ) (b : ℝ) (h_ω : ω > 0) (h_b : 0 < b ∧ b < 2) : 
  (∃ (k : ℤ), φ = 2 * π * k - 3 * π / 2) →
  (∀ (x : ℝ), 2 * Real.sin (ω * x + φ) = b → 
    (x = π / 6 ∨ x = 5 * π / 6 ∨ x = 7 * π / 6)) →
  (∀ (x : ℝ), 2 * Real.sin (ω * 3 * π / 2 + φ) ≤ 2 * Real.sin (ω * x + φ)) →
  ω = 2 ∧ (∀ (ψ : ℝ), |ψ| ≥ π / 2) := by
  sorry

end min_abs_phi_l2156_215604


namespace remaining_value_probability_theorem_l2156_215645

/-- Represents a bag of bills -/
structure Bag where
  tens : ℕ
  fives : ℕ
  ones : ℕ

/-- Calculates the total value of bills in a bag -/
def bagValue (b : Bag) : ℕ := 10 * b.tens + 5 * b.fives + b.ones

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the probability of the remaining value in Bag A being greater than Bag B -/
def remainingValueProbability (bagA bagB : Bag) : ℚ :=
  let totalA := choose (bagA.tens + bagA.fives + bagA.ones) 2
  let totalB := choose (bagB.tens + bagB.fives + bagB.ones) 2
  let favorableA := choose bagA.ones 2
  let favorableB := totalB - choose bagB.ones 2
  (favorableA * favorableB : ℚ) / (totalA * totalB : ℚ)

theorem remaining_value_probability_theorem :
  let bagA : Bag := { tens := 2, fives := 0, ones := 3 }
  let bagB : Bag := { tens := 0, fives := 4, ones := 3 }
  remainingValueProbability bagA bagB = 9 / 35 := by
  sorry

end remaining_value_probability_theorem_l2156_215645


namespace karen_wall_paint_area_l2156_215662

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℝ
  width : ℝ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℝ := d.height * d.width

/-- Represents Karen's living room wall with its components -/
structure Wall where
  dimensions : Dimensions
  window : Dimensions
  door : Dimensions

/-- Calculates the area to be painted on the wall -/
def areaToPaint (w : Wall) : ℝ :=
  area w.dimensions - area w.window - area w.door

theorem karen_wall_paint_area :
  let wall : Wall := {
    dimensions := { height := 10, width := 15 },
    window := { height := 3, width := 5 },
    door := { height := 2, width := 6 }
  }
  areaToPaint wall = 123 := by sorry

end karen_wall_paint_area_l2156_215662


namespace jerry_books_count_l2156_215664

/-- The number of books each shelf can hold -/
def books_per_shelf : ℕ := 4

/-- The number of shelves Jerry needs -/
def shelves_needed : ℕ := 3

/-- The total number of books Jerry has to put away -/
def total_books : ℕ := books_per_shelf * shelves_needed

theorem jerry_books_count : total_books = 12 := by
  sorry

end jerry_books_count_l2156_215664


namespace total_birds_in_store_l2156_215610

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 6

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 7

/-- Theorem: The total number of birds in the pet store is 54 -/
theorem total_birds_in_store : 
  num_cages * (parrots_per_cage + parakeets_per_cage) = 54 := by
  sorry

end total_birds_in_store_l2156_215610


namespace line_circle_intersection_distance_l2156_215605

/-- The line y = kx + 1 intersects the circle (x - 2)² + y² = 9 at two points with distance 4 apart -/
theorem line_circle_intersection_distance (k : ℝ) : ∃ (A B : ℝ × ℝ),
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  (y₁ = k * x₁ + 1) ∧
  (y₂ = k * x₂ + 1) ∧
  ((x₁ - 2)^2 + y₁^2 = 9) ∧
  ((x₂ - 2)^2 + y₂^2 = 9) ∧
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) := by
  sorry

#check line_circle_intersection_distance

end line_circle_intersection_distance_l2156_215605


namespace flour_for_one_loaf_l2156_215602

/-- Given that 5 cups of flour are needed for 2 loaves of bread,
    prove that 2.5 cups of flour are needed for 1 loaf of bread. -/
theorem flour_for_one_loaf (total_flour : ℝ) (total_loaves : ℝ) 
  (h1 : total_flour = 5)
  (h2 : total_loaves = 2) :
  total_flour / total_loaves = 2.5 := by
  sorry

end flour_for_one_loaf_l2156_215602


namespace max_quotient_value_l2156_215656

theorem max_quotient_value (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) :
  (∀ x y, 300 ≤ x ∧ x ≤ 500 → 800 ≤ y ∧ y ≤ 1600 → 2 * y / x ≤ 32 / 3) ∧
  (∃ x y, 300 ≤ x ∧ x ≤ 500 ∧ 800 ≤ y ∧ y ≤ 1600 ∧ 2 * y / x = 32 / 3) :=
by sorry

end max_quotient_value_l2156_215656


namespace sin_240_degrees_l2156_215673

theorem sin_240_degrees : Real.sin (240 * π / 180) = -(1/2) := by
  sorry

end sin_240_degrees_l2156_215673


namespace weight_fluctuation_l2156_215649

theorem weight_fluctuation (initial_weight : ℕ) (initial_loss : ℕ) (final_gain : ℕ) : 
  initial_weight = 99 →
  initial_loss = 12 →
  final_gain = 6 →
  initial_weight - initial_loss + 2 * initial_loss - 3 * initial_loss + final_gain = 81 := by
  sorry

end weight_fluctuation_l2156_215649


namespace b_completion_time_l2156_215691

/-- The number of days A needs to complete the work alone -/
def a_days : ℝ := 12

/-- The number of days A works before B joins -/
def a_solo_days : ℝ := 2

/-- The total number of days A and B work together to complete the job -/
def total_days : ℝ := 8

/-- The number of days B needs to complete the work alone -/
def b_days : ℝ := 18

/-- The theorem stating that given the conditions, B can complete the work alone in 18 days -/
theorem b_completion_time :
  (a_days = 12) →
  (a_solo_days = 2) →
  (total_days = 8) →
  (b_days = 18) →
  (1 / a_days * a_solo_days + (total_days - a_solo_days) * (1 / a_days + 1 / b_days) = 1) :=
by sorry

end b_completion_time_l2156_215691


namespace quadrilateral_S_l2156_215667

/-- Given a quadrilateral with sides a, b, c, d and an angle A (where A ≠ 90°),
    S is equal to (a^2 + d^2 - b^2 - c^2) / (4 * tan(A)) -/
theorem quadrilateral_S (a b c d : ℝ) (A : ℝ) (h : A ≠ π / 2) :
  let S := (a^2 + d^2 - b^2 - c^2) / (4 * Real.tan A)
  ∃ (S : ℝ), S = (a^2 + d^2 - b^2 - c^2) / (4 * Real.tan A) :=
by
  sorry

end quadrilateral_S_l2156_215667


namespace large_cups_sold_is_five_l2156_215697

/-- Represents the number of cups sold for each size --/
structure CupsSold where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total revenue based on the number of cups sold --/
def totalRevenue (cups : CupsSold) : ℕ :=
  cups.small + 2 * cups.medium + 3 * cups.large

theorem large_cups_sold_is_five :
  ∃ (cups : CupsSold),
    totalRevenue cups = 50 ∧
    cups.small = 11 ∧
    2 * cups.medium = 24 ∧
    cups.large = 5 := by
  sorry

end large_cups_sold_is_five_l2156_215697


namespace valid_purchase_plans_l2156_215606

/-- Represents a purchasing plan for basketballs and footballs -/
structure PurchasePlan where
  basketballs : ℕ
  footballs : ℕ

/-- Checks if a purchase plan is valid according to the given conditions -/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.basketballs + p.footballs = 20 ∧
  p.basketballs > p.footballs ∧
  80 * p.basketballs + 50 * p.footballs ≤ 1400

theorem valid_purchase_plans :
  ∀ (p : PurchasePlan), isValidPlan p ↔ 
    (p = ⟨11, 9⟩ ∨ p = ⟨12, 8⟩ ∨ p = ⟨13, 7⟩) :=
by sorry

end valid_purchase_plans_l2156_215606


namespace work_completion_time_l2156_215668

-- Define work rates as fractions of work completed per hour
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 12
def work_rate_BC : ℚ := 1 / 3

-- Define the time taken for A and C together
def time_AC : ℚ := 2

theorem work_completion_time :
  let work_rate_C : ℚ := work_rate_BC - work_rate_B
  let work_rate_AC : ℚ := work_rate_A + work_rate_C
  time_AC = 1 / work_rate_AC :=
by sorry

end work_completion_time_l2156_215668


namespace f_increasing_and_even_l2156_215635

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_increasing_and_even :
  -- f is monotonically increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) ∧
  -- f is an even function
  (∀ x : ℝ, f (-x) = f x) := by
  sorry

end f_increasing_and_even_l2156_215635


namespace weight_replacement_l2156_215644

theorem weight_replacement (n : ℕ) (old_avg new_avg new_weight : ℝ) : 
  n = 8 → 
  new_avg - old_avg = 2.5 →
  new_weight = 70 →
  (n * new_avg - new_weight + (n * old_avg - n * new_avg)) / (n - 1) = 50 :=
by sorry

end weight_replacement_l2156_215644
