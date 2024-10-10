import Mathlib

namespace sufficient_but_not_necessary_l2960_296079

/-- Given a > 0 and a ≠ 1, if f(x) = ax is decreasing on ℝ, then g(x) = (2-a)x³ is increasing on ℝ, 
    but the converse is not always true. -/
theorem sufficient_but_not_necessary (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a * x < a * y) →
  (∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3) ∧
  ¬(∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3 → a * x < a * y) :=
by sorry

end sufficient_but_not_necessary_l2960_296079


namespace abs_k_less_than_abs_b_l2960_296038

/-- Given a linear function y = kx + b, prove that |k| < |b| under certain conditions --/
theorem abs_k_less_than_abs_b (k b : ℝ) : 
  (∀ x y, y = k * x + b) →  -- The function is of the form y = kx + b
  (b > 0) →  -- The y-intercept is positive
  (0 < k + b) →  -- The point (1, k+b) is above the x-axis
  (k + b < b) →  -- The point (1, k+b) is below b
  |k| < |b| := by
sorry


end abs_k_less_than_abs_b_l2960_296038


namespace largest_divisor_of_five_consecutive_even_integers_l2960_296095

theorem largest_divisor_of_five_consecutive_even_integers (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (d : ℕ), d = 96 ∧
  (∀ (k : ℕ), k > 96 → ¬(k ∣ n * (n + 2) * (n + 4) * (n + 6) * (n + 8))) ∧
  (96 ∣ n * (n + 2) * (n + 4) * (n + 6) * (n + 8)) :=
sorry

end largest_divisor_of_five_consecutive_even_integers_l2960_296095


namespace pencil_distribution_l2960_296042

theorem pencil_distribution (total : ℕ) (h1 : total = 8 * 6 + 4) : 
  total / 4 = 13 := by
sorry

end pencil_distribution_l2960_296042


namespace slope_of_line_l2960_296077

/-- The slope of a line given by the equation 4y = 5x - 8 is 5/4 -/
theorem slope_of_line (x y : ℝ) : 4 * y = 5 * x - 8 → (∃ m b : ℝ, y = m * x + b ∧ m = 5 / 4) := by
  sorry

end slope_of_line_l2960_296077


namespace opposite_of_negative_2023_l2960_296013

theorem opposite_of_negative_2023 : 
  ∀ x : ℤ, (x + (-2023) = 0) → x = 2023 := by
  sorry

end opposite_of_negative_2023_l2960_296013


namespace log_product_equals_ten_l2960_296029

theorem log_product_equals_ten (n : ℕ) (h : n = 2) : 
  7.63 * (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) = 10 := by
  sorry

end log_product_equals_ten_l2960_296029


namespace problem_1_l2960_296059

theorem problem_1 (a : ℝ) : 
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) ↔ a ∈ Set.Iic (-1 : ℝ) := by sorry

end problem_1_l2960_296059


namespace james_investment_l2960_296070

theorem james_investment (initial_balance : ℝ) (weekly_investment : ℝ) (weeks : ℕ) (windfall_percentage : ℝ) : 
  initial_balance = 250000 ∧ 
  weekly_investment = 2000 ∧ 
  weeks = 52 ∧ 
  windfall_percentage = 0.5 →
  let final_balance := initial_balance + weekly_investment * weeks
  let windfall := windfall_percentage * final_balance
  final_balance + windfall = 531000 := by
sorry

end james_investment_l2960_296070


namespace field_length_is_96_l2960_296036

/-- Proves that the length of a rectangular field is 96 meters given specific conditions -/
theorem field_length_is_96 (w : ℝ) (l : ℝ) : 
  l = 2 * w →                   -- length is double the width
  64 = (1 / 72) * (l * w) →     -- area of pond (8^2) is 1/72 of field area
  l = 96 := by
sorry

end field_length_is_96_l2960_296036


namespace derek_remaining_money_l2960_296045

theorem derek_remaining_money (initial_amount : ℕ) : 
  initial_amount = 960 →
  let textbook_expense := initial_amount / 2
  let remaining_after_textbooks := initial_amount - textbook_expense
  let supply_expense := remaining_after_textbooks / 4
  let final_remaining := remaining_after_textbooks - supply_expense
  final_remaining = 360 := by
sorry

end derek_remaining_money_l2960_296045


namespace count_numbers_satisfying_conditions_l2960_296017

/-- A function that returns true if n can be expressed as the sum of k consecutive positive integers starting from a -/
def is_sum_of_consecutive_integers (n k a : ℕ) : Prop :=
  n = k * a + k * (k - 1) / 2

/-- A function that checks if n satisfies all the conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  n ≤ 2000 ∧ ∃ k a : ℕ, k ≥ 60 ∧ is_sum_of_consecutive_integers n k a

/-- The main theorem stating that there are exactly 6 numbers satisfying the conditions -/
theorem count_numbers_satisfying_conditions :
  ∃! (S : Finset ℕ), (∀ n ∈ S, satisfies_conditions n) ∧ S.card = 6 :=
sorry

end count_numbers_satisfying_conditions_l2960_296017


namespace sine_shift_l2960_296044

theorem sine_shift (x : ℝ) : 3 * Real.sin (2 * x + π / 5) = 3 * Real.sin (2 * (x + π / 10)) := by
  sorry

end sine_shift_l2960_296044


namespace tommy_calculation_l2960_296096

theorem tommy_calculation (x : ℚ) : (x - 7) / 5 = 23 → (x - 5) / 7 = 16 := by
  sorry

end tommy_calculation_l2960_296096


namespace solve_for_k_l2960_296055

theorem solve_for_k : ∃ k : ℚ, (4 * k - 3 * (-1) = 2) ∧ (k = -1/4) := by
  sorry

end solve_for_k_l2960_296055


namespace inequality_proof_l2960_296098

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  1 / (x^2 + y + z) + 1 / (x + y^2 + z) + 1 / (x + y + z^2) ≤ 1 := by
  sorry

end inequality_proof_l2960_296098


namespace water_reservoir_ratio_l2960_296023

/-- The ratio of the amount of water in the reservoir at the end of the month to the normal level -/
theorem water_reservoir_ratio : 
  ∀ (total_capacity normal_level end_month_level : ℝ),
  end_month_level = 30 →
  end_month_level = 0.75 * total_capacity →
  normal_level = total_capacity - 20 →
  end_month_level / normal_level = 1.5 := by
sorry

end water_reservoir_ratio_l2960_296023


namespace probability_in_standard_deck_l2960_296063

/-- Represents a standard deck of 52 cards -/
structure Deck :=
  (total_cards : Nat)
  (diamonds : Nat)
  (spades : Nat)
  (hearts : Nat)

/-- The probability of drawing a diamond, then a spade, then a heart from a standard deck -/
def probability_diamond_spade_heart (d : Deck) : Rat :=
  (d.diamonds : Rat) / d.total_cards *
  (d.spades : Rat) / (d.total_cards - 1) *
  (d.hearts : Rat) / (d.total_cards - 2)

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52
  , diamonds := 13
  , spades := 13
  , hearts := 13 }

theorem probability_in_standard_deck :
  probability_diamond_spade_heart standard_deck = 13 / 780 := by
  sorry

end probability_in_standard_deck_l2960_296063


namespace investment_rate_problem_l2960_296053

theorem investment_rate_problem (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (desired_income : ℝ) :
  total_investment = 12000 →
  first_investment = 5000 →
  second_investment = 4000 →
  first_rate = 0.03 →
  second_rate = 0.035 →
  desired_income = 430 →
  let remainder := total_investment - first_investment - second_investment
  let income_from_first := first_investment * first_rate
  let income_from_second := second_investment * second_rate
  let required_additional_income := desired_income - income_from_first - income_from_second
  let required_rate := required_additional_income / remainder
  required_rate = 0.047 := by
  sorry

end investment_rate_problem_l2960_296053


namespace water_usage_calculation_l2960_296061

/-- Calculates the weekly water usage for baths given the specified parameters. -/
def weekly_water_usage (bucket_capacity : ℕ) (buckets_to_fill : ℕ) (buckets_removed : ℕ) (baths_per_week : ℕ) : ℕ :=
  let total_capacity := bucket_capacity * buckets_to_fill
  let water_removed := bucket_capacity * buckets_removed
  let water_per_bath := total_capacity - water_removed
  water_per_bath * baths_per_week

/-- Theorem stating that the weekly water usage is 9240 ounces given the specified parameters. -/
theorem water_usage_calculation :
  weekly_water_usage 120 14 3 7 = 9240 := by
  sorry

end water_usage_calculation_l2960_296061


namespace corn_farmer_profit_l2960_296062

/-- Calculates the profit for a corn farmer given specific conditions. -/
theorem corn_farmer_profit : 
  let seeds_per_ear : ℕ := 4
  let price_per_ear : ℚ := 1/10
  let seeds_per_bag : ℕ := 100
  let price_per_bag : ℚ := 1/2
  let ears_sold : ℕ := 500
  let total_seeds : ℕ := seeds_per_ear * ears_sold
  let bags_needed : ℕ := (total_seeds + seeds_per_bag - 1) / seeds_per_bag
  let total_cost : ℚ := bags_needed * price_per_bag
  let total_revenue : ℚ := ears_sold * price_per_ear
  let profit : ℚ := total_revenue - total_cost
  profit = 40 := by sorry

end corn_farmer_profit_l2960_296062


namespace perpendicular_vectors_l2960_296094

/-- Given vectors a and b, if ka + b is perpendicular to a, then k = 2/5 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : b = (-2, 0)) 
  (h3 : (k • a.1 + b.1) * a.1 + (k • a.2 + b.2) * a.2 = 0) : 
  k = 2/5 := by sorry

end perpendicular_vectors_l2960_296094


namespace range_of_p_l2960_296076

open Set

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - 10*x

def A : Set ℝ := {x | (deriv f) x ≤ 0}

def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

theorem range_of_p (p : ℝ) (h : A ∪ B p = A) : p ≤ 3 := by
  sorry

end range_of_p_l2960_296076


namespace probability_prime_product_l2960_296019

/-- A standard 6-sided die -/
def Die : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of prime numbers on a 6-sided die -/
def PrimesOnDie : Finset ℕ := {2, 3, 5}

/-- The number of favorable outcomes -/
def FavorableOutcomes : ℕ := 9

/-- The total number of possible outcomes when rolling three dice -/
def TotalOutcomes : ℕ := 216

/-- The probability of rolling three 6-sided dice and getting a prime number as the product of their face values -/
theorem probability_prime_product (d : Finset ℕ) (p : Finset ℕ) (f : ℕ) (t : ℕ) 
  (h1 : d = Die) 
  (h2 : p = PrimesOnDie) 
  (h3 : f = FavorableOutcomes) 
  (h4 : t = TotalOutcomes) :
  (f : ℚ) / t = 1 / 24 := by
  sorry

end probability_prime_product_l2960_296019


namespace oranges_left_l2960_296001

theorem oranges_left (total : ℕ) (percentage : ℚ) (remaining : ℕ) : 
  total = 96 → 
  percentage = 48/100 →
  remaining = total - Int.floor (percentage * total) →
  remaining = 50 := by
sorry

end oranges_left_l2960_296001


namespace remainder_problem_l2960_296009

theorem remainder_problem : (7 * 10^24 + 2^24) % 13 = 8 := by
  sorry

end remainder_problem_l2960_296009


namespace minimal_difference_factors_l2960_296090

theorem minimal_difference_factors : ∃ (a b : ℤ),
  a * b = 1234567890 ∧
  ∀ (x y : ℤ), x * y = 1234567890 → |x - y| ≥ |a - b| ∧
  a = 36070 ∧ b = 34227 := by sorry

end minimal_difference_factors_l2960_296090


namespace second_class_average_l2960_296043

/-- Proves that given two classes with specified student counts and averages,
    the average mark of the second class is 90. -/
theorem second_class_average (students1 students2 : ℕ) (avg1 avg_combined : ℚ) :
  students1 = 30 →
  students2 = 50 →
  avg1 = 40 →
  avg_combined = 71.25 →
  (students1 * avg1 + students2 * (90 : ℚ)) / (students1 + students2 : ℚ) = avg_combined :=
by sorry

end second_class_average_l2960_296043


namespace equation_arrangements_l2960_296012

def word : String := "equation"

def letter_count : Nat := word.length

theorem equation_arrangements :
  let distinct_letters : Nat := 8
  let qu_as_unit : Nat := 1
  let remaining_letters : Nat := distinct_letters - 2
  let units_to_arrange : Nat := qu_as_unit + remaining_letters
  let letters_to_select : Nat := 5 - 2
  let ways_to_select : Nat := Nat.choose remaining_letters letters_to_select
  let ways_to_arrange : Nat := Nat.factorial (letters_to_select + 1)
  ways_to_select * ways_to_arrange = 480 := by
  sorry

end equation_arrangements_l2960_296012


namespace inscribed_square_area_l2960_296040

/-- A square inscribed in a semicircle with radius 1 -/
structure InscribedSquare where
  /-- The center of the semicircle -/
  center : ℝ × ℝ
  /-- The vertices of the square -/
  vertices : Fin 4 → ℝ × ℝ
  /-- Two vertices are on the semicircle -/
  on_semicircle : ∃ (i j : Fin 4), i ≠ j ∧
    (vertices i).1^2 + (vertices i).2^2 = 1 ∧
    (vertices j).1^2 + (vertices j).2^2 = 1 ∧
    (vertices i).2 ≥ 0 ∧ (vertices j).2 ≥ 0
  /-- Two vertices are on the diameter -/
  on_diameter : ∃ (i j : Fin 4), i ≠ j ∧
    (vertices i).2 = 0 ∧ (vertices j).2 = 0 ∧
    abs ((vertices i).1 - (vertices j).1) = 2
  /-- The vertices form a square -/
  is_square : ∀ (i j : Fin 4), i ≠ j →
    (vertices i).1^2 + (vertices i).2^2 =
    (vertices j).1^2 + (vertices j).2^2

/-- The area of an inscribed square is 4/5 -/
theorem inscribed_square_area (s : InscribedSquare) :
  let side_length := abs ((s.vertices 0).1 - (s.vertices 1).1)
  side_length^2 = 4/5 := by
  sorry

end inscribed_square_area_l2960_296040


namespace driving_equation_correct_l2960_296065

/-- Represents a driving trip with a stop -/
structure DrivingTrip where
  speed_before_stop : ℝ
  speed_after_stop : ℝ
  stop_duration : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The equation for calculating the total distance is correct -/
theorem driving_equation_correct (trip : DrivingTrip) 
  (h1 : trip.speed_before_stop = 60)
  (h2 : trip.speed_after_stop = 90)
  (h3 : trip.stop_duration = 1/2)
  (h4 : trip.total_distance = 270)
  (h5 : trip.total_time = 4) :
  ∃ t : ℝ, 60 * t + 90 * (7/2 - t) = 270 ∧ 
           0 ≤ t ∧ t ≤ trip.total_time - trip.stop_duration :=
by sorry

end driving_equation_correct_l2960_296065


namespace tree_initial_height_l2960_296058

/-- Given a tree with constant yearly growth for 6 years, prove its initial height. -/
theorem tree_initial_height (growth_rate : ℝ) (h1 : growth_rate = 0.4) : ∃ (initial_height : ℝ),
  initial_height + 6 * growth_rate = (initial_height + 4 * growth_rate) * (1 + 1/7) ∧
  initial_height = 4 := by
  sorry

end tree_initial_height_l2960_296058


namespace g_zeros_count_l2960_296041

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x + a)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (x - a) - x^2

theorem g_zeros_count (a : ℝ) :
  (∀ x, g a x ≠ 0) ∨
  (∃! x, g a x = 0) ∨
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧ ∀ x, g a x = 0 → x = x₁ ∨ x = x₂) ∨
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧ g a x₃ = 0 ∧
    ∀ x, g a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) :=
by sorry

end g_zeros_count_l2960_296041


namespace binomial_probability_problem_l2960_296018

theorem binomial_probability_problem (p : ℝ) (X : ℕ → ℝ) :
  (∀ k, X k = Nat.choose 4 k * p^k * (1 - p)^(4 - k)) →
  X 2 = 8/27 →
  p = 1/3 ∨ p = 2/3 :=
by sorry

end binomial_probability_problem_l2960_296018


namespace complex_equation_solution_l2960_296026

theorem complex_equation_solution (x y : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (2 * x + i) * (1 - i) = y) : y = 2 := by
  sorry

end complex_equation_solution_l2960_296026


namespace simplify_trig_expression_l2960_296030

theorem simplify_trig_expression :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) =
  Real.tan (60 * π / 180) :=
by sorry

end simplify_trig_expression_l2960_296030


namespace distance_ratio_l2960_296075

-- Define the speeds and time for both cars
def speed_A : ℝ := 70
def speed_B : ℝ := 35
def time : ℝ := 10

-- Define the distances traveled by each car
def distance_A : ℝ := speed_A * time
def distance_B : ℝ := speed_B * time

-- Theorem to prove the ratio of distances
theorem distance_ratio :
  distance_A / distance_B = 2 := by sorry

end distance_ratio_l2960_296075


namespace graph_is_parabola_l2960_296064

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * (x - 2)^2 + 6

-- Theorem stating that the graph of f is a parabola
theorem graph_is_parabola : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c :=
sorry

end graph_is_parabola_l2960_296064


namespace berry_ratio_l2960_296006

theorem berry_ratio (total berries stacy steve skylar : ℕ) : 
  total = 1100 →
  stacy = 800 →
  stacy = 4 * steve →
  total = stacy + steve + skylar →
  steve = 2 * skylar := by
sorry

end berry_ratio_l2960_296006


namespace koch_snowflake_area_l2960_296011

/-- Given a sequence of curves P₀, P₁, P₂, ..., where:
    1. P₀ is an equilateral triangle with area 1
    2. Pₖ₊₁ is obtained from Pₖ by trisecting each side, constructing an equilateral 
       triangle on the middle segment, and removing the middle segment
    3. Sₙ is the area enclosed by curve Pₙ
    
    This theorem states the formula for Sₙ and its limit as n approaches infinity. -/
theorem koch_snowflake_area (n : ℕ) : 
  ∃ (S : ℕ → ℝ), 
    (∀ k, S k = (47/20) * (1 - (4/9)^k)) ∧ 
    (∀ ε > 0, ∃ N, ∀ n ≥ N, |S n - 47/20| < ε) := by
  sorry

end koch_snowflake_area_l2960_296011


namespace rectangular_plot_length_l2960_296033

/-- Proves that the length of a rectangular plot is 62 meters given the specified conditions -/
theorem rectangular_plot_length : ∀ (breadth length perimeter : ℝ),
  length = breadth + 24 →
  perimeter = 2 * (length + breadth) →
  perimeter * 26.5 = 5300 →
  length = 62 := by
  sorry

end rectangular_plot_length_l2960_296033


namespace investment_solution_l2960_296071

def compound_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate)

def investment_problem (principal : ℝ) : Prop :=
  let year1_amount := compound_interest principal 0.05
  let year2_amount := compound_interest year1_amount 0.07
  let year3_amount := compound_interest year2_amount 0.04
  year3_amount = 1232

theorem investment_solution :
  ∃ (principal : ℝ), investment_problem principal ∧ 
    (principal ≥ 1054.75 ∧ principal ≤ 1054.77) :=
by
  sorry

#check investment_solution

end investment_solution_l2960_296071


namespace lines_perpendicular_iff_product_slopes_neg_one_l2960_296091

/-- Two lines y = k₁x + l₁ and y = k₂x + l₂, where k₁ ≠ 0 and k₂ ≠ 0, are perpendicular if and only if k₁k₂ = -1 -/
theorem lines_perpendicular_iff_product_slopes_neg_one
  (k₁ k₂ l₁ l₂ : ℝ) (hk₁ : k₁ ≠ 0) (hk₂ : k₂ ≠ 0) :
  (∃ (x y : ℝ), y = k₁ * x + l₁ ∧ y = k₂ * x + l₂) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ),
    y₁ = k₁ * x₁ + l₁ →
    y₂ = k₂ * x₂ + l₂ →
    (x₂ - x₁) * (y₂ - y₁) = 0) ↔
  k₁ * k₂ = -1 := by
sorry

end lines_perpendicular_iff_product_slopes_neg_one_l2960_296091


namespace area_increase_6_to_7_l2960_296097

/-- Calculates the increase in area of a square when its side length is increased by 1 unit -/
def area_increase (side_length : ℝ) : ℝ :=
  (side_length + 1)^2 - side_length^2

/-- Theorem: The increase in area of a square with side length 6 units, 
    when increased by 1 unit, is 13 square units -/
theorem area_increase_6_to_7 : area_increase 6 = 13 := by
  sorry

end area_increase_6_to_7_l2960_296097


namespace necessary_not_sufficient_condition_for_x_gt_e_l2960_296046

theorem necessary_not_sufficient_condition_for_x_gt_e (x : ℝ) :
  (x > Real.exp 1 → x > 1) ∧ ¬(x > 1 → x > Real.exp 1) :=
sorry

end necessary_not_sufficient_condition_for_x_gt_e_l2960_296046


namespace equation_solution_l2960_296027

theorem equation_solution (x : ℝ) :
  3 / (x - 3) + 5 / (2 * x - 6) = 11 / 2 →
  2 * x - 6 = 2 :=
by sorry

end equation_solution_l2960_296027


namespace perfect_square_product_divisible_by_12_l2960_296010

theorem perfect_square_product_divisible_by_12 (n : ℤ) : 
  12 ∣ (n^2 * (n^2 - 1)) := by
  sorry

end perfect_square_product_divisible_by_12_l2960_296010


namespace completing_square_equivalence_l2960_296039

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_square_equivalence_l2960_296039


namespace root_equation_value_l2960_296056

theorem root_equation_value (m : ℝ) (h : m^2 - 3*m + 1 = 0) : 
  (m - 3)^2 + (m + 2)*(m - 2) = 3 := by
  sorry

end root_equation_value_l2960_296056


namespace unique_sequence_l2960_296000

def sequence_condition (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n > 0 → a n < a (n + 1)) ∧
  (∀ n : ℕ, n > 0 → a (2 * n) = a n + n) ∧
  (∀ n : ℕ, n > 0 → Prime (a n) → Prime n)

theorem unique_sequence :
  ∀ a : ℕ → ℕ, sequence_condition a → ∀ n : ℕ, n > 0 → a n = n :=
sorry

end unique_sequence_l2960_296000


namespace jose_profit_share_l2960_296078

/-- Calculates the share of profit for an investor in a partnership --/
def calculate_profit_share (investment1 : ℕ) (months1 : ℕ) (investment2 : ℕ) (months2 : ℕ) (total_profit : ℕ) : ℕ :=
  let total_investment := investment1 * months1 + investment2 * months2
  let share_ratio := investment2 * months2 * total_profit / total_investment
  share_ratio

/-- Proves that Jose's share of the profit is 3500 --/
theorem jose_profit_share :
  calculate_profit_share 3000 12 4500 10 6300 = 3500 := by
  sorry

end jose_profit_share_l2960_296078


namespace quadrilateral_angle_measure_l2960_296072

/-- Given two angles x and y in a quadrilateral satisfying certain conditions,
    prove that x equals (1 + √13) / 6 degrees. -/
theorem quadrilateral_angle_measure (x y : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x = a ∧ y = b) →  -- x and y are positive real numbers (representing angles)
  (3 * x^2 - x + 4 = 5) →                        -- First condition
  (x^2 + y^2 = 9) →                              -- Second condition
  x = (1 + Real.sqrt 13) / 6 :=                  -- Conclusion
by sorry

end quadrilateral_angle_measure_l2960_296072


namespace subtract_fractions_l2960_296054

theorem subtract_fractions (p q : ℚ) (h1 : 3 / p = 4) (h2 : 3 / q = 18) : p - q = 7/12 := by
  sorry

end subtract_fractions_l2960_296054


namespace compare_expressions_l2960_296002

theorem compare_expressions (x y : ℝ) (h1 : x < y) (h2 : y < 0) :
  (x^2 + y^2) * (x - y) > (x^2 - y^2) * (x + y) := by
  sorry

end compare_expressions_l2960_296002


namespace total_power_cost_l2960_296086

/-- Represents the cost of power for each appliance in Joseph's house --/
structure ApplianceCosts where
  waterHeater : ℝ
  refrigerator : ℝ
  electricOven : ℝ
  airConditioner : ℝ
  washingMachine : ℝ

/-- Calculates the total cost of power for all appliances --/
def totalCost (costs : ApplianceCosts) : ℝ :=
  costs.waterHeater + costs.refrigerator + costs.electricOven + costs.airConditioner + costs.washingMachine

/-- Theorem stating the total cost of power for all appliances --/
theorem total_power_cost (costs : ApplianceCosts) 
  (h1 : costs.refrigerator = 3 * costs.waterHeater)
  (h2 : costs.electricOven = 500)
  (h3 : costs.electricOven = 2.5 * costs.waterHeater)
  (h4 : costs.airConditioner = 300)
  (h5 : costs.washingMachine = 100) :
  totalCost costs = 1700 := by
  sorry


end total_power_cost_l2960_296086


namespace power_equation_solution_l2960_296089

theorem power_equation_solution (n : ℕ) : 
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^42 → n = 42 := by
sorry

end power_equation_solution_l2960_296089


namespace fifth_largest_divisor_of_n_l2960_296007

def n : ℕ := 1209600000

/-- The fifth-largest divisor of n -/
def fifth_largest_divisor : ℕ := 75600000

/-- A function that returns the kth largest divisor of a number -/
def kth_largest_divisor (m k : ℕ) : ℕ := sorry

theorem fifth_largest_divisor_of_n :
  kth_largest_divisor n 5 = fifth_largest_divisor := by sorry

end fifth_largest_divisor_of_n_l2960_296007


namespace max_value_sqrt_sum_l2960_296093

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 5 ∧ Real.sqrt (x + 1) + Real.sqrt (y + 3) ≥ Real.sqrt (a + 1) + Real.sqrt (b + 3)) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y = 5 → Real.sqrt (x + 1) + Real.sqrt (y + 3) ≤ 3 * Real.sqrt 2) :=
by sorry

end max_value_sqrt_sum_l2960_296093


namespace kaylaScoreEighthLevel_l2960_296084

/-- Fibonacci sequence starting with 1 and 1 -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Kayla's score at a given level -/
def kaylaScore : ℕ → ℤ
  | 0 => 2
  | n + 1 => if n % 2 = 0 then kaylaScore n - fib n else kaylaScore n + fib n

theorem kaylaScoreEighthLevel : kaylaScore 7 = -7 := by
  sorry

end kaylaScoreEighthLevel_l2960_296084


namespace extrema_and_tangent_line_l2960_296080

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem extrema_and_tangent_line :
  -- Local extrema conditions
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f x ≤ f (-1)) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  -- Tangent line condition
  (∃ x₀ : ℝ, 9*x₀ - f x₀ + 16 = 0 ∧
    ∀ x : ℝ, 9*x - f x + 16 = 0 → x = x₀) :=
by sorry

end extrema_and_tangent_line_l2960_296080


namespace sum_of_squares_of_roots_l2960_296048

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 13*x + 4 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = 13 ∧ r₁ * r₂ = 4 ∧ r₁^2 + r₂^2 = 161 := by
  sorry

end sum_of_squares_of_roots_l2960_296048


namespace computer_literate_female_employees_l2960_296021

/-- Proves the number of computer literate female employees in an office -/
theorem computer_literate_female_employees
  (total_employees : ℕ)
  (female_percentage : ℚ)
  (male_computer_literate_percentage : ℚ)
  (total_computer_literate_percentage : ℚ)
  (h_total : total_employees = 1200)
  (h_female : female_percentage = 60 / 100)
  (h_male_cl : male_computer_literate_percentage = 50 / 100)
  (h_total_cl : total_computer_literate_percentage = 62 / 100) :
  ↑(total_employees : ℚ) * female_percentage * total_computer_literate_percentage -
  (↑(total_employees : ℚ) * (1 - female_percentage) * male_computer_literate_percentage) = 504 :=
sorry

end computer_literate_female_employees_l2960_296021


namespace tangent_slope_at_one_l2960_296052

-- Define the function f(x) = x³ - x² + x + 1
def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 2 := by sorry

end tangent_slope_at_one_l2960_296052


namespace coin_representation_l2960_296035

def is_representable (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 3 * a + 5 * b

theorem coin_representation :
  ∀ n : ℕ, n > 0 → (is_representable n ↔ n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 4 ∧ n ≠ 7) :=
by sorry

end coin_representation_l2960_296035


namespace pond_water_after_50_days_l2960_296087

/-- Calculates the remaining water in a pond after a given number of days, considering evaporation. -/
def remaining_water (initial_water : ℕ) (evaporation_rate : ℕ) (days : ℕ) : ℕ :=
  initial_water - evaporation_rate * days

/-- Theorem stating that a pond with 500 gallons of water, losing 1 gallon per day, will have 450 gallons after 50 days. -/
theorem pond_water_after_50_days :
  remaining_water 500 1 50 = 450 := by
  sorry

#eval remaining_water 500 1 50

end pond_water_after_50_days_l2960_296087


namespace x_values_l2960_296020

def A (x : ℝ) : Set ℝ := {x, x^2}

theorem x_values (x : ℝ) (h : 1 ∈ A x) : x = 1 ∨ x = -1 := by
  sorry

end x_values_l2960_296020


namespace quadratic_function_sum_l2960_296022

theorem quadratic_function_sum (a b : ℝ) (h1 : a ≠ 0) : 
  (∀ x y : ℝ, y = a * x^2 + b * x - 1) → 
  (1 = a * 1^2 + b * 1 - 1) →
  a + b + 1 = 3 := by
sorry

end quadratic_function_sum_l2960_296022


namespace smartphone_sales_l2960_296004

theorem smartphone_sales (units_at_400 price_400 price_800 : ℝ) 
  (h1 : units_at_400 = 20)
  (h2 : price_400 = 400)
  (h3 : price_800 = 800)
  (h4 : ∀ (p c : ℝ), p * c = units_at_400 * price_400) :
  (units_at_400 * price_400) / price_800 = 10 := by
  sorry

end smartphone_sales_l2960_296004


namespace parallelogram_base_l2960_296047

theorem parallelogram_base (area height : ℝ) (h1 : area = 231) (h2 : height = 11) :
  area / height = 21 := by
  sorry

end parallelogram_base_l2960_296047


namespace gcd_special_numbers_l2960_296015

theorem gcd_special_numbers :
  let m : ℕ := 55555555
  let n : ℕ := 111111111
  Nat.gcd m n = 1 := by sorry

end gcd_special_numbers_l2960_296015


namespace solution_set_of_inequality_l2960_296067

theorem solution_set_of_inequality (x : ℝ) :
  (2 / (x - 1) ≥ 1) ↔ (1 < x ∧ x ≤ 3) :=
by sorry

end solution_set_of_inequality_l2960_296067


namespace no_solution_for_floor_sum_l2960_296057

theorem no_solution_for_floor_sum (x : ℝ) : 
  ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ ≠ 12345 := by
  sorry

end no_solution_for_floor_sum_l2960_296057


namespace matrix_identity_sum_l2960_296049

open Matrix

variable {n : Type*} [DecidableEq n] [Fintype n]

theorem matrix_identity_sum (B : Matrix n n ℝ) :
  Invertible B →
  (B - 3 • 1) * (B - 5 • 1) = 0 →
  B + 15 • B⁻¹ = 8 • 1 := by
  sorry

end matrix_identity_sum_l2960_296049


namespace tangency_points_form_cyclic_quadrilateral_l2960_296099

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles being externally tangent
def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define a point of tangency between two circles
def tangency_point (c1 c2 : Circle) : ℝ × ℝ :=
  sorry

-- Define the property of a quadrilateral being cyclic
def is_cyclic_quadrilateral (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  sorry

-- Theorem statement
theorem tangency_points_form_cyclic_quadrilateral 
  (S1 S2 S3 S4 : Circle)
  (h12 : externally_tangent S1 S2)
  (h23 : externally_tangent S2 S3)
  (h34 : externally_tangent S3 S4)
  (h41 : externally_tangent S4 S1) :
  let p1 := tangency_point S1 S2
  let p2 := tangency_point S2 S3
  let p3 := tangency_point S3 S4
  let p4 := tangency_point S4 S1
  is_cyclic_quadrilateral p1 p2 p3 p4 :=
by
  sorry

end tangency_points_form_cyclic_quadrilateral_l2960_296099


namespace C_equiv_C_param_l2960_296005

/-- A semicircular curve C in the polar coordinate system -/
def C : Set (ℝ × ℝ) := {(p, θ) | p = 2 * Real.cos θ ∧ 0 ≤ θ ∧ θ ≤ Real.pi / 2}

/-- The parametric representation of curve C -/
def C_param : Set (ℝ × ℝ) := {(x, y) | ∃ α, 0 ≤ α ∧ α ≤ Real.pi ∧ x = 1 + Real.cos α ∧ y = Real.sin α}

/-- Theorem stating that the parametric representation is equivalent to the polar representation -/
theorem C_equiv_C_param : C = C_param := by sorry

end C_equiv_C_param_l2960_296005


namespace sum_of_reciprocals_negative_l2960_296008

theorem sum_of_reciprocals_negative (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_pos : a * b * c > 0) : 
  1 / a + 1 / b + 1 / c < 0 :=
by sorry

end sum_of_reciprocals_negative_l2960_296008


namespace complement_of_A_in_U_l2960_296085

-- Define the universal set U
def U : Set ℕ := {0, 1, 2}

-- Define set A
def A : Set ℕ := {x ∈ U | x^2 - x = 0}

-- State the theorem
theorem complement_of_A_in_U : 
  {x ∈ U | x ∉ A} = {2} := by sorry

end complement_of_A_in_U_l2960_296085


namespace time_is_one_point_two_hours_l2960_296081

/-- The number of letters in the name -/
def name_length : ℕ := 6

/-- The number of rearrangements that can be written per minute -/
def rearrangements_per_minute : ℕ := 10

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the time in hours to write all rearrangements of a name -/
def time_to_write_all_rearrangements : ℚ :=
  (name_length.factorial / rearrangements_per_minute : ℚ) / minutes_per_hour

/-- Theorem stating that the time to write all rearrangements is 1.2 hours -/
theorem time_is_one_point_two_hours :
  time_to_write_all_rearrangements = 6/5 := by sorry

end time_is_one_point_two_hours_l2960_296081


namespace order_of_expressions_l2960_296073

-- Define the base of the logarithm
def b : Real := 0.2

-- State the theorem
theorem order_of_expressions (a : Real) (h : a > 1) :
  Real.log a / Real.log b < b * a ∧ b * a < a ^ b :=
sorry

end order_of_expressions_l2960_296073


namespace third_generation_tail_length_l2960_296031

/-- The tail length growth factor between generations -/
def growth_factor : ℝ := 1.25

/-- The initial tail length of the first generation in centimeters -/
def initial_length : ℝ := 16

/-- The tail length of the nth generation -/
def tail_length (n : ℕ) : ℝ := initial_length * growth_factor ^ n

theorem third_generation_tail_length :
  tail_length 2 = 25 := by sorry

end third_generation_tail_length_l2960_296031


namespace sphere_radius_in_cone_l2960_296074

/-- A right circular cone with a base radius of 7 and height of 15 -/
structure Cone where
  baseRadius : ℝ := 7
  height : ℝ := 15

/-- A sphere with radius r -/
structure Sphere (r : ℝ) where

/-- Configuration of four spheres in the cone -/
structure SphereConfiguration (r : ℝ) where
  cone : Cone
  spheres : Fin 4 → Sphere r
  bottomThreeTangent : Bool
  bottomThreeTouchBase : Bool
  bottomThreeTouchSide : Bool
  topSphereTouchesOthers : Bool
  topSphereTouchesSide : Bool
  topSphereNotTouchBase : Bool

/-- The theorem stating the radius of the spheres in the given configuration -/
theorem sphere_radius_in_cone (config : SphereConfiguration r) :
  r = (162 - 108 * Real.sqrt 3) / 3 :=
sorry

end sphere_radius_in_cone_l2960_296074


namespace fixed_distance_vector_l2960_296034

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem fixed_distance_vector (a b : E) :
  ∃ t u : ℝ, ∀ p : E,
    (‖p - b‖ = 3 * ‖p - a‖) →
    (∃ c : ℝ, ∀ q : E, (‖p - b‖ = 3 * ‖p - a‖) → ‖q - (t • a + u • b)‖ = c) →
    t = 9/8 ∧ u = -1/8 :=
by sorry

end fixed_distance_vector_l2960_296034


namespace geometric_sequence_properties_l2960_296024

/-- Given a geometric sequence with common ratio q > 0 and T_n as the product of the first n terms,
    if T_7 > T_6 > T_8, then 0 < q < 1 and T_13 > 1 > T_14 -/
theorem geometric_sequence_properties (q : ℝ) (T : ℕ → ℝ) 
    (h_q_pos : q > 0)
    (h_T : ∀ n, T n = (T 1) * q^(n-1))
    (h_ineq : T 7 > T 6 ∧ T 6 > T 8) :
    (0 < q ∧ q < 1) ∧ (T 13 > 1 ∧ 1 > T 14) := by
  sorry


end geometric_sequence_properties_l2960_296024


namespace arithmetic_sequence_sum_l2960_296083

/-- An arithmetic sequence with 2036 terms -/
def ArithmeticSequence (a d : ℝ) : ℕ → ℝ :=
  fun n => a + (n - 1) * d

theorem arithmetic_sequence_sum (a d : ℝ) :
  let t := ArithmeticSequence a d
  t 2018 = 100 →
  t 2000 + 5 * t 2015 + 5 * t 2021 + t 2036 = 1200 := by
  sorry

end arithmetic_sequence_sum_l2960_296083


namespace probability_of_specific_match_l2960_296050

/-- The number of teams in the tournament -/
def num_teams : ℕ := 128

/-- The probability of two specific teams playing each other in a single elimination tournament -/
def probability_of_match (n : ℕ) : ℚ :=
  (n - 1) / (n * (n - 1) / 2)

/-- Theorem: In a single elimination tournament with 128 equally strong teams,
    the probability of two specific teams playing each other is 1/64 -/
theorem probability_of_specific_match :
  probability_of_match num_teams = 1 / 64 := by sorry

end probability_of_specific_match_l2960_296050


namespace ned_games_problem_l2960_296066

theorem ned_games_problem (initial_games : ℕ) : 
  (3/4 : ℚ) * (2/3 : ℚ) * initial_games = 6 → initial_games = 12 := by
  sorry

end ned_games_problem_l2960_296066


namespace special_polygon_properties_l2960_296037

/-- A polygon where the sum of interior angles is more than three times the sum of exterior angles by 180° --/
structure SpecialPolygon where
  n : ℕ
  interior_sum : ℝ
  exterior_sum : ℝ
  h : interior_sum = 3 * exterior_sum + 180

theorem special_polygon_properties (p : SpecialPolygon) :
  p.n = 9 ∧ p.n - 3 = 6 := by sorry


end special_polygon_properties_l2960_296037


namespace arithmetic_calculation_l2960_296032

theorem arithmetic_calculation : 5 + 15 / 3 - 2^3 = 2 := by
  sorry

end arithmetic_calculation_l2960_296032


namespace computer_price_increase_l2960_296003

theorem computer_price_increase (y : ℝ) (h1 : 2 * y = 540) : 
  y * (1 + 0.3) = 351 := by sorry

end computer_price_increase_l2960_296003


namespace monotonicity_indeterminate_l2960_296014

theorem monotonicity_indeterminate 
  (f : ℝ → ℝ) 
  (h_domain : ∀ x, x ∈ Set.Icc (-1) 2 → f x ≠ 0) 
  (h_inequality : f (-1/2) < f 1) : 
  ¬ (∀ x y, x ∈ Set.Icc (-1) 2 → y ∈ Set.Icc (-1) 2 → x < y → f x < f y) ∧ 
  ¬ (∀ x y, x ∈ Set.Icc (-1) 2 → y ∈ Set.Icc (-1) 2 → x < y → f x > f y) :=
sorry

end monotonicity_indeterminate_l2960_296014


namespace expression_evaluation_l2960_296051

theorem expression_evaluation (a b c : ℚ) : 
  a = 6 → 
  b = 2 * a - 1 → 
  c = 2 * b - 30 → 
  a + 2 ≠ 0 → 
  b - 3 ≠ 0 → 
  c + 7 ≠ 0 → 
  (a + 3) / (a + 2) * (b + 5) / (b - 3) * (c + 10) / (c + 7) = 9 / 2 := by
  sorry

end expression_evaluation_l2960_296051


namespace resulting_temperature_correct_l2960_296088

/-- The resulting temperature when rising from 5°C to t°C -/
def resulting_temperature (t : ℝ) : ℝ := 5 + t

/-- Theorem stating that the resulting temperature is correct -/
theorem resulting_temperature_correct (t : ℝ) : 
  resulting_temperature t = 5 + t := by sorry

end resulting_temperature_correct_l2960_296088


namespace average_weight_increase_l2960_296068

/-- Proves that replacing a person weighing 58 kg with a person weighing 106 kg
    in a group of 12 people increases the average weight by 4 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total_weight := 12 * initial_average
  let new_total_weight := initial_total_weight - 58 + 106
  let new_average := new_total_weight / 12
  new_average - initial_average = 4 := by
  sorry

end average_weight_increase_l2960_296068


namespace min_sum_squares_l2960_296060

theorem min_sum_squares (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → 
  (∃ m : ℝ, m = a^2 + b^2 ∧ ∀ c d : ℝ, (∃ y : ℝ, y^4 + c*y^3 + d*y^2 + c*y + 1 = 0) → m ≤ c^2 + d^2) ∧ 
  (∃ n : ℝ, n = 4/5 ∧ n = a^2 + b^2) :=
by sorry

end min_sum_squares_l2960_296060


namespace sequence_property_l2960_296025

-- Define the sequence type
def Sequence := ℕ+ → ℝ

-- Define the property of the sequence
def HasProperty (a : Sequence) : Prop :=
  ∀ n : ℕ+, a n * a (n + 2) = (a (n + 1))^2

-- State the theorem
theorem sequence_property (a : Sequence) 
  (h1 : HasProperty a) 
  (h2 : a 7 = 16) 
  (h3 : a 3 * a 5 = 4) : 
  a 3 = 1 := by
  sorry

end sequence_property_l2960_296025


namespace quadratic_solution_average_l2960_296092

/-- Given a quadratic equation 2x^2 - 6x + c = 0 with two real solutions and discriminant 12,
    prove that the average of the solutions is 1.5 -/
theorem quadratic_solution_average (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 2 * x₁^2 - 6 * x₁ + c = 0 ∧ 2 * x₂^2 - 6 * x₂ + c = 0) →
  ((-6)^2 - 4 * 2 * c = 12) →
  (∃ x₁ x₂ : ℝ, 2 * x₁^2 - 6 * x₁ + c = 0 ∧ 2 * x₂^2 - 6 * x₂ + c = 0 ∧ (x₁ + x₂) / 2 = 1.5) :=
by sorry

end quadratic_solution_average_l2960_296092


namespace max_vertex_product_sum_l2960_296016

/-- Represents the assignment of numbers to the faces of a cube -/
structure CubeAssignment where
  faces : Fin 6 → ℕ
  valid : ∀ i, faces i ∈ ({3, 4, 5, 6, 7, 8} : Set ℕ)
  distinct : ∀ i j, i ≠ j → faces i ≠ faces j

/-- Calculates the sum of products at vertices for a given cube assignment -/
def vertexProductSum (assignment : CubeAssignment) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem: The maximum sum of vertex products is 1331 -/
theorem max_vertex_product_sum :
  ∃ (assignment : CubeAssignment), 
    vertexProductSum assignment = 1331 ∧
    ∀ (other : CubeAssignment), vertexProductSum other ≤ 1331 :=
  sorry

end max_vertex_product_sum_l2960_296016


namespace sqrt_equation_solution_l2960_296028

theorem sqrt_equation_solution :
  ∃ x : ℝ, x > 0 ∧ Real.sqrt 289 - Real.sqrt 625 / Real.sqrt x = 12 ∧ x = 25 :=
by sorry

end sqrt_equation_solution_l2960_296028


namespace complex_subtraction_l2960_296069

theorem complex_subtraction (i : ℂ) (h : i^2 = -1) :
  (5 - 3*i) - (7 - 7*i) = -2 + 4*i :=
sorry

end complex_subtraction_l2960_296069


namespace potato_problem_result_l2960_296082

/-- Represents the potato problem --/
structure PotatoProblem where
  totalPotatoes : Nat
  potatoesForWedges : Nat
  wedgesPerPotato : Nat
  chipsPerPotato : Nat

/-- Calculates the difference between potato chips and wedges --/
def chipWedgeDifference (p : PotatoProblem) : Nat :=
  let remainingPotatoes := p.totalPotatoes - p.potatoesForWedges
  let potatoesForChips := remainingPotatoes / 2
  let totalChips := potatoesForChips * p.chipsPerPotato
  let totalWedges := p.potatoesForWedges * p.wedgesPerPotato
  totalChips - totalWedges

/-- Theorem stating the result of the potato problem --/
theorem potato_problem_result :
  let p : PotatoProblem := {
    totalPotatoes := 67,
    potatoesForWedges := 13,
    wedgesPerPotato := 8,
    chipsPerPotato := 20
  }
  chipWedgeDifference p = 436 := by
  sorry

end potato_problem_result_l2960_296082
