import Mathlib

namespace different_color_probability_l1517_151797

/-- The probability of drawing two balls of different colors from a box -/
theorem different_color_probability (total : ℕ) (red : ℕ) (yellow : ℕ) : 
  total = red + yellow →
  red = 3 →
  yellow = 2 →
  (red.choose 1 * yellow.choose 1 : ℚ) / total.choose 2 = 3/5 :=
sorry

end different_color_probability_l1517_151797


namespace students_who_just_passed_l1517_151761

theorem students_who_just_passed
  (total_students : ℕ)
  (first_division_percent : ℚ)
  (second_division_percent : ℚ)
  (h1 : total_students = 300)
  (h2 : first_division_percent = 27 / 100)
  (h3 : second_division_percent = 54 / 100)
  (h4 : first_division_percent + second_division_percent < 1) :
  total_students - (total_students * (first_division_percent + second_division_percent)).floor = 57 :=
by sorry

end students_who_just_passed_l1517_151761


namespace equals_2022_l1517_151770

theorem equals_2022 : 1 - (-2021) = 2022 := by
  sorry

#check equals_2022

end equals_2022_l1517_151770


namespace base4_to_base10_conversion_l1517_151717

/-- Converts a base 4 number represented as a list of digits to base 10 -/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The base 4 representation of the number -/
def base4Number : List Nat := [2, 1, 0, 1, 2]

theorem base4_to_base10_conversion :
  base4ToBase10 base4Number = 582 := by
  sorry

end base4_to_base10_conversion_l1517_151717


namespace apple_cost_l1517_151705

/-- The cost of apples given specific pricing rules and total costs for certain weights. -/
theorem apple_cost (l q : ℝ) : 
  (∀ x, x ≤ 30 → x * l = x * 0.362) →  -- Cost for first 30 kgs
  (∀ x, x > 30 → x * l + (x - 30) * q = 30 * l + (x - 30) * q) →  -- Cost for additional kgs
  (33 * l + 3 * q = 11.67) →  -- Price for 33 kgs
  (36 * l + 6 * q = 12.48) →  -- Price for 36 kgs
  (10 * l = 3.62) :=  -- Cost of first 10 kgs
by sorry

end apple_cost_l1517_151705


namespace meaningful_expression_l1517_151707

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 1)) ↔ x > 1 := by
sorry

end meaningful_expression_l1517_151707


namespace harmonic_progression_solutions_l1517_151765

def is_harmonic_progression (a b c : ℕ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (1 / a + 1 / c = 2 / b)

def valid_harmonic_progression (a b c : ℕ) : Prop :=
  is_harmonic_progression a b c ∧ a < b ∧ b < c ∧ a = 20 ∧ c % b = 0

theorem harmonic_progression_solutions :
  {(b, c) : ℕ × ℕ | valid_harmonic_progression 20 b c} =
    {(30, 60), (35, 140), (36, 180), (38, 380), (39, 780)} := by
  sorry

end harmonic_progression_solutions_l1517_151765


namespace product_of_sums_inequality_l1517_151719

theorem product_of_sums_inequality {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  (x + 1) * (y + 1) ≥ 4 ∧ ((x + 1) * (y + 1) = 4 ↔ x = 1 ∧ y = 1) :=
sorry

end product_of_sums_inequality_l1517_151719


namespace power_of_two_equation_solution_l1517_151788

theorem power_of_two_equation_solution : ∃ k : ℕ, 
  2^2004 - 2^2003 - 2^2002 + 2^2001 = k * 2^2001 ∧ k = 3 := by
  sorry

end power_of_two_equation_solution_l1517_151788


namespace set_union_problem_l1517_151794

-- Define the sets A and B as functions of x
def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

-- State the theorem
theorem set_union_problem (x : ℝ) :
  (A x ∩ B x = {9}) →
  (∃ y, A y ∪ B y = {-8, -7, -4, 4, 9}) :=
by sorry

end set_union_problem_l1517_151794


namespace solution_set_length_l1517_151799

theorem solution_set_length (a : ℝ) (h1 : a > 0) : 
  (∃ x1 x2 : ℝ, x1 < x2 ∧ 
    (∀ x : ℝ, x1 ≤ x ∧ x ≤ x2 ↔ Real.sqrt (x + a) + Real.sqrt (x - a) ≤ Real.sqrt (2 * (x + 1))) ∧
    x2 - x1 = 1/2) →
  a = 3/4 := by sorry

end solution_set_length_l1517_151799


namespace stones_for_hall_l1517_151736

-- Define the hall dimensions in decimeters
def hall_length : ℕ := 360
def hall_width : ℕ := 150

-- Define the stone dimensions in decimeters
def stone_length : ℕ := 3
def stone_width : ℕ := 5

-- Define the function to calculate the number of stones required
def stones_required (hall_l hall_w stone_l stone_w : ℕ) : ℕ :=
  (hall_l * hall_w) / (stone_l * stone_w)

-- Theorem statement
theorem stones_for_hall :
  stones_required hall_length hall_width stone_length stone_width = 3600 := by
  sorry

end stones_for_hall_l1517_151736


namespace coffee_decaf_percentage_l1517_151762

theorem coffee_decaf_percentage 
  (initial_stock : ℝ) 
  (initial_decaf_percent : ℝ) 
  (additional_purchase : ℝ) 
  (additional_decaf_percent : ℝ) 
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : additional_purchase = 100)
  (h4 : additional_decaf_percent = 60) :
  let initial_decaf := initial_stock * (initial_decaf_percent / 100)
  let additional_decaf := additional_purchase * (additional_decaf_percent / 100)
  let total_decaf := initial_decaf + additional_decaf
  let total_stock := initial_stock + additional_purchase
  (total_decaf / total_stock) * 100 = 28 := by
sorry

end coffee_decaf_percentage_l1517_151762


namespace football_equipment_cost_l1517_151740

/-- Given the cost of football equipment, prove the total cost relation -/
theorem football_equipment_cost (x : ℝ) 
  (h1 : x + x = 2 * x)           -- Shorts + T-shirt = 2x
  (h2 : x + 4 * x = 5 * x)       -- Shorts + boots = 5x
  (h3 : x + 2 * x = 3 * x)       -- Shorts + shin guards = 3x
  : x + x + 4 * x + 2 * x = 8 * x := by
  sorry


end football_equipment_cost_l1517_151740


namespace max_value_expression_l1517_151714

theorem max_value_expression (a b c d : ℝ) 
  (ha : -5.5 ≤ a ∧ a ≤ 5.5)
  (hb : -5.5 ≤ b ∧ b ≤ 5.5)
  (hc : -5.5 ≤ c ∧ c ≤ 5.5)
  (hd : -5.5 ≤ d ∧ d ≤ 5.5) :
  (∀ a' b' c' d' : ℝ, 
    -5.5 ≤ a' ∧ a' ≤ 5.5 →
    -5.5 ≤ b' ∧ b' ≤ 5.5 →
    -5.5 ≤ c' ∧ c' ≤ 5.5 →
    -5.5 ≤ d' ∧ d' ≤ 5.5 →
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' ≤ 132) ∧
  (∃ a' b' c' d' : ℝ, 
    -5.5 ≤ a' ∧ a' ≤ 5.5 ∧
    -5.5 ≤ b' ∧ b' ≤ 5.5 ∧
    -5.5 ≤ c' ∧ c' ≤ 5.5 ∧
    -5.5 ≤ d' ∧ d' ≤ 5.5 ∧
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' = 132) :=
by sorry

end max_value_expression_l1517_151714


namespace smallest_positive_solution_is_18_l1517_151713

theorem smallest_positive_solution_is_18 : 
  let f : ℝ → ℝ := fun t => -t^2 + 14*t + 40
  ∃ t : ℝ, t > 0 ∧ f t = 94 ∧ ∀ s : ℝ, s > 0 ∧ f s = 94 → t ≤ s → t = 18 :=
by sorry

end smallest_positive_solution_is_18_l1517_151713


namespace clockHandsOpposite_eq_48_l1517_151741

/-- The number of times clock hands are in a straight line but opposite in direction in a day -/
def clockHandsOpposite : ℕ :=
  let hoursOnClockFace : ℕ := 12
  let hoursInDay : ℕ := 24
  let occurrencesPerHour : ℕ := 2
  hoursInDay * occurrencesPerHour

/-- Theorem stating that clock hands are in a straight line but opposite in direction 48 times a day -/
theorem clockHandsOpposite_eq_48 : clockHandsOpposite = 48 := by
  sorry

end clockHandsOpposite_eq_48_l1517_151741


namespace total_lives_in_game_game_lives_proof_l1517_151759

theorem total_lives_in_game (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) : ℕ :=
  (initial_players + additional_players) * lives_per_player

theorem game_lives_proof :
  total_lives_in_game 4 5 3 = 27 := by
  sorry

end total_lives_in_game_game_lives_proof_l1517_151759


namespace street_trees_count_l1517_151738

theorem street_trees_count (road_length : ℕ) (interval : ℕ) : 
  road_length = 2575 → interval = 25 → (road_length / interval + 1 : ℕ) = 104 := by
  sorry

end street_trees_count_l1517_151738


namespace unique_solution_l1517_151760

/-- Represents the work information for a worker -/
structure WorkerInfo where
  days : ℕ
  totalPay : ℕ
  dailyWage : ℚ

/-- Verifies if the given work information satisfies the problem conditions -/
def satisfiesConditions (a b : WorkerInfo) : Prop :=
  b.days = a.days - 3 ∧
  a.totalPay = 30 ∧
  b.totalPay = 14 ∧
  a.dailyWage = a.totalPay / a.days ∧
  b.dailyWage = b.totalPay / b.days ∧
  (a.days - 2) * a.dailyWage = (b.days + 5) * b.dailyWage

/-- The main theorem stating the unique solution to the problem -/
theorem unique_solution :
  ∃! (a b : WorkerInfo),
    satisfiesConditions a b ∧
    a.days = 10 ∧
    b.days = 7 ∧
    a.dailyWage = 3 ∧
    b.dailyWage = 2 :=
by
  sorry

end unique_solution_l1517_151760


namespace grid_midpoint_theorem_l1517_151743

theorem grid_midpoint_theorem (points : Finset (ℤ × ℤ)) :
  points.card = 5 →
  ∃ p q : ℤ × ℤ, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧
    Even (p.1 + q.1) ∧ Even (p.2 + q.2) :=
by sorry

end grid_midpoint_theorem_l1517_151743


namespace right_triangle_ratio_range_l1517_151753

theorem right_triangle_ratio_range (a b c h : ℝ) :
  a > 0 → b > 0 →
  c = (a^2 + b^2).sqrt →
  h = (a * b) / c →
  1 < (c + 2 * h) / (a + b) ∧ (c + 2 * h) / (a + b) ≤ Real.sqrt 2 := by
  sorry

end right_triangle_ratio_range_l1517_151753


namespace last_flip_heads_prob_2010_l1517_151712

/-- A coin that comes up the same as the last flip 2/3 of the time and opposite 1/3 of the time -/
structure BiasedCoin where
  same_prob : ℚ
  diff_prob : ℚ
  prob_sum_one : same_prob + diff_prob = 1
  same_prob_val : same_prob = 2/3
  diff_prob_val : diff_prob = 1/3

/-- The probability of the last flip being heads after n flips, given the first flip was heads -/
def last_flip_heads_prob (coin : BiasedCoin) (n : ℕ) : ℚ :=
  (3^n + 1) / (2 * 3^n)

/-- The theorem statement -/
theorem last_flip_heads_prob_2010 (coin : BiasedCoin) :
  last_flip_heads_prob coin 2010 = (3^2010 + 1) / (2 * 3^2010) := by
  sorry

end last_flip_heads_prob_2010_l1517_151712


namespace quadratic_form_ratio_l1517_151778

/-- For the quadratic x^2 + 2200x + 4200, when written in the form (x+b)^2 + c, c/b = -1096 -/
theorem quadratic_form_ratio : ∃ (b c : ℝ), 
  (∀ x, x^2 + 2200*x + 4200 = (x + b)^2 + c) ∧ 
  c / b = -1096 := by
  sorry

end quadratic_form_ratio_l1517_151778


namespace train_bridge_crossing_time_l1517_151789

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : Real) 
  (train_speed_kmh : Real) 
  (bridge_length : Real) : 
  train_length = 110 → 
  train_speed_kmh = 72 → 
  bridge_length = 142 → 
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let crossing_time := total_distance / train_speed_ms
  crossing_time = 12.6 := by
  sorry

end train_bridge_crossing_time_l1517_151789


namespace linda_cookies_theorem_l1517_151791

/-- The number of batches Linda needs to bake to have enough cookies for her classmates -/
def batches_needed (num_classmates : ℕ) (cookies_per_student : ℕ) (cookies_per_batch : ℕ) 
  (choc_chip_batches : ℕ) (oatmeal_raisin_batches : ℕ) : ℕ :=
  let total_cookies_needed := num_classmates * cookies_per_student
  let cookies_made := (choc_chip_batches + oatmeal_raisin_batches) * cookies_per_batch
  let cookies_left_to_make := total_cookies_needed - cookies_made
  (cookies_left_to_make + cookies_per_batch - 1) / cookies_per_batch

theorem linda_cookies_theorem : 
  batches_needed 24 10 48 2 1 = 2 := by
  sorry

end linda_cookies_theorem_l1517_151791


namespace non_degenerate_ellipse_condition_l1517_151744

/-- The equation of the curve -/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 + 9*y^2 - 6*x + 18*y = k

/-- Definition of a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ (a b h₁ h₂ : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), curve_equation x y k ↔ (x - h₁)^2 / a^2 + (y - h₂)^2 / b^2 = 1

/-- Theorem stating the condition for the curve to be a non-degenerate ellipse -/
theorem non_degenerate_ellipse_condition :
  ∀ k : ℝ, is_non_degenerate_ellipse k ↔ k > -9 := by sorry

end non_degenerate_ellipse_condition_l1517_151744


namespace potato_cost_for_group_l1517_151747

/-- The cost of potatoes for a group, given the number of people, amount each person eats,
    bag size, and cost per bag. -/
def potatoCost (people : ℕ) (poundsPerPerson : ℚ) (bagSize : ℕ) (costPerBag : ℚ) : ℚ :=
  let totalPounds : ℚ := people * poundsPerPerson
  let bagsNeeded : ℕ := (totalPounds / bagSize).ceil.toNat
  bagsNeeded * costPerBag

/-- Theorem stating that the cost of potatoes for 40 people, where each person eats 1.5 pounds,
    and a 20-pound bag costs $5, is $15. -/
theorem potato_cost_for_group : potatoCost 40 (3/2) 20 5 = 15 := by
  sorry

end potato_cost_for_group_l1517_151747


namespace absolute_value_condition_l1517_151755

theorem absolute_value_condition (a : ℝ) : 
  (a ≤ 0 → |a - 2| ≥ 1) ∧ 
  ¬(|a - 2| ≥ 1 → a ≤ 0) :=
sorry

end absolute_value_condition_l1517_151755


namespace rare_card_cost_proof_l1517_151758

/-- The cost of each rare card in Tom's deck -/
def rare_card_cost : ℝ := 1

/-- The number of rare cards in Tom's deck -/
def num_rare_cards : ℕ := 19

/-- The number of uncommon cards in Tom's deck -/
def num_uncommon_cards : ℕ := 11

/-- The number of common cards in Tom's deck -/
def num_common_cards : ℕ := 30

/-- The cost of each uncommon card -/
def uncommon_card_cost : ℝ := 0.50

/-- The cost of each common card -/
def common_card_cost : ℝ := 0.25

/-- The total cost of Tom's deck -/
def total_deck_cost : ℝ := 32

theorem rare_card_cost_proof :
  rare_card_cost * num_rare_cards +
  uncommon_card_cost * num_uncommon_cards +
  common_card_cost * num_common_cards = total_deck_cost :=
by sorry

end rare_card_cost_proof_l1517_151758


namespace system_solution_unique_l1517_151790

theorem system_solution_unique (x y : ℝ) : 
  x + y = 5 ∧ 3 * x + y = 7 ↔ x = 1 ∧ y = 4 := by sorry

end system_solution_unique_l1517_151790


namespace ordinary_time_rate_l1517_151768

/-- Calculates the ordinary time rate given total hours, overtime hours, overtime rate, and total earnings -/
theorem ordinary_time_rate 
  (total_hours : ℕ) 
  (overtime_hours : ℕ) 
  (overtime_rate : ℚ) 
  (total_earnings : ℚ) 
  (h1 : total_hours = 50)
  (h2 : overtime_hours = 8)
  (h3 : overtime_rate = 9/10)
  (h4 : total_earnings = 1620/50)
  (h5 : overtime_hours ≤ total_hours) :
  let ordinary_hours := total_hours - overtime_hours
  let ordinary_rate := (total_earnings - overtime_rate * overtime_hours) / ordinary_hours
  ordinary_rate = 3/5 := by
  sorry

end ordinary_time_rate_l1517_151768


namespace average_weight_whole_class_l1517_151784

theorem average_weight_whole_class 
  (students_a : ℕ) (students_b : ℕ) 
  (avg_weight_a : ℚ) (avg_weight_b : ℚ) :
  students_a = 40 →
  students_b = 20 →
  avg_weight_a = 50 →
  avg_weight_b = 40 →
  (students_a * avg_weight_a + students_b * avg_weight_b) / (students_a + students_b) = 140 / 3 :=
by sorry

end average_weight_whole_class_l1517_151784


namespace units_digit_of_m_cubed_plus_three_to_m_l1517_151752

def m : ℕ := 2011^2 + 2^2011

theorem units_digit_of_m_cubed_plus_three_to_m (m : ℕ := 2011^2 + 2^2011) : 
  (m^3 + 3^m) % 10 = 2 := by sorry

end units_digit_of_m_cubed_plus_three_to_m_l1517_151752


namespace find_wrong_height_l1517_151725

/-- Given a class of boys with an initially miscalculated average height and the correct average height after fixing one boy's height, find the wrongly written height of that boy. -/
theorem find_wrong_height (n : ℕ) (initial_avg : ℝ) (actual_height : ℝ) (correct_avg : ℝ) 
    (hn : n = 35)
    (hi : initial_avg = 181)
    (ha : actual_height = 106)
    (hc : correct_avg = 179) :
    ∃ wrong_height : ℝ,
      wrong_height = n * initial_avg - (n * correct_avg - actual_height) :=
by sorry

end find_wrong_height_l1517_151725


namespace james_coffee_consumption_l1517_151776

/-- Proves that James bought 2 coffees per day before buying a coffee machine -/
theorem james_coffee_consumption
  (machine_cost : ℕ)
  (daily_making_cost : ℕ)
  (previous_coffee_cost : ℕ)
  (payoff_days : ℕ)
  (h1 : machine_cost = 180)
  (h2 : daily_making_cost = 3)
  (h3 : previous_coffee_cost = 4)
  (h4 : payoff_days = 36) :
  ∃ x : ℕ, x = 2 ∧ payoff_days * (previous_coffee_cost * x - daily_making_cost) = machine_cost :=
by sorry

end james_coffee_consumption_l1517_151776


namespace total_age_is_22_l1517_151775

/-- Given three people A, B, and C with the following age relationships:
    - A is two years older than B
    - B is twice as old as C
    - B is 8 years old
    This theorem proves that the sum of their ages is 22 years. -/
theorem total_age_is_22 (a b c : ℕ) : 
  b = 8 → a = b + 2 → b = 2 * c → a + b + c = 22 := by
  sorry

end total_age_is_22_l1517_151775


namespace salary_change_percentage_l1517_151708

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let increased_salary := initial_salary * 1.5
  let final_salary := increased_salary * 0.9
  (final_salary - initial_salary) / initial_salary * 100 = 35 := by
  sorry

end salary_change_percentage_l1517_151708


namespace intersection_implies_a_range_l1517_151772

/-- Given two functions f and g, where f(x) = ax and g(x) = ln x, 
    if their graphs intersect at two different points in (0, +∞),
    then 0 < a < 1/e. -/
theorem intersection_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ 
   a * x₁ = Real.log x₁ ∧ a * x₂ = Real.log x₂) →
  0 < a ∧ a < 1 / Real.exp 1 := by
  sorry

end intersection_implies_a_range_l1517_151772


namespace goldfish_problem_l1517_151739

theorem goldfish_problem (initial : ℕ) (final : ℕ) (birds : ℕ) (disease : ℕ) 
  (h1 : initial = 240)
  (h2 : final = 45)
  (h3 : birds = 15)
  (h4 : disease = 30) :
  let vanished := initial - final
  let heat := (vanished * 20) / 100
  let eaten := vanished - heat - disease - birds
  let raccoons := eaten / 3
  let cats := raccoons * 2
  cats = 64 ∧ raccoons = 32 ∧ heat = 39 := by
  sorry

end goldfish_problem_l1517_151739


namespace product_remainder_mod_five_l1517_151754

theorem product_remainder_mod_five :
  (1024 * 1455 * 1776 * 2018 * 2222) % 5 = 0 := by
  sorry

end product_remainder_mod_five_l1517_151754


namespace min_attacking_pairs_8x8_16rooks_l1517_151786

/-- Represents a chessboard configuration -/
structure ChessBoard where
  size : Nat
  rooks : Nat

/-- Calculates the minimum number of attacking rook pairs -/
def minAttackingPairs (board : ChessBoard) : Nat :=
  sorry

/-- Theorem stating the minimum number of attacking rook pairs for a specific configuration -/
theorem min_attacking_pairs_8x8_16rooks :
  let board : ChessBoard := { size := 8, rooks := 16 }
  minAttackingPairs board = 16 := by
  sorry

end min_attacking_pairs_8x8_16rooks_l1517_151786


namespace utility_bill_total_l1517_151711

def fifty_bill_value : ℕ := 50
def ten_bill_value : ℕ := 10
def fifty_bill_count : ℕ := 3
def ten_bill_count : ℕ := 2

theorem utility_bill_total : 
  fifty_bill_value * fifty_bill_count + ten_bill_value * ten_bill_count = 170 := by
  sorry

end utility_bill_total_l1517_151711


namespace det_2CD_l1517_151702

theorem det_2CD (C D : Matrix (Fin 3) (Fin 3) ℝ) 
  (hC : Matrix.det C = 3)
  (hD : Matrix.det D = 8) :
  Matrix.det (2 • (C * D)) = 192 := by
  sorry

end det_2CD_l1517_151702


namespace repeating_decimal_fraction_sum_l1517_151792

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / (d : ℚ) = 3.17171717 ∧ 
  n + d = 413 := by
  sorry

end repeating_decimal_fraction_sum_l1517_151792


namespace egg_price_calculation_l1517_151735

/-- The price of a dozen eggs given the number of chickens, eggs laid per day, and total earnings --/
def price_per_dozen (num_chickens : ℕ) (eggs_per_chicken_per_day : ℕ) (total_earnings : ℕ) : ℚ :=
  let total_days : ℕ := 28  -- 4 weeks
  let total_eggs : ℕ := num_chickens * eggs_per_chicken_per_day * total_days
  let total_dozens : ℕ := total_eggs / 12
  (total_earnings : ℚ) / total_dozens

theorem egg_price_calculation :
  price_per_dozen 8 3 280 = 5 := by
  sorry

end egg_price_calculation_l1517_151735


namespace function_properties_l1517_151793

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + Real.log (2 - x) + a * x

-- State the theorem
theorem function_properties (a : ℝ) (h : a > 0) :
  -- 1. Domain of f(x) is (0, 2)
  (∀ x, f a x ≠ Real.log 0 → 0 < x ∧ x < 2) ∧
  -- 2. When a = 1, f(x) is increasing on (0, √2) and decreasing on (√2, 2)
  (a = 1 →
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.sqrt 2 → f 1 x₁ < f 1 x₂) ∧
    (∀ x₁ x₂, Real.sqrt 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f 1 x₁ > f 1 x₂)) ∧
  -- 3. If the maximum value of f(x) on (0, 1] is 1/2, then a = 1/2
  ((∃ x, 0 < x ∧ x ≤ 1 ∧ f a x = 1/2 ∧ ∀ y, 0 < y ∧ y ≤ 1 → f a y ≤ 1/2) → a = 1/2) :=
by sorry

end function_properties_l1517_151793


namespace smaller_solution_quadratic_equation_l1517_151722

theorem smaller_solution_quadratic_equation :
  ∃ (x y : ℝ), x < y ∧ 
  x^2 - 9*x - 22 = 0 ∧ 
  y^2 - 9*y - 22 = 0 ∧
  ∀ z : ℝ, z^2 - 9*z - 22 = 0 → z = x ∨ z = y ∧
  x = -2 := by
sorry

end smaller_solution_quadratic_equation_l1517_151722


namespace root_implies_k_value_l1517_151766

theorem root_implies_k_value (k : ℝ) : ((-3 : ℝ)^2 + (-3) - k = 0) → k = 6 := by
  sorry

end root_implies_k_value_l1517_151766


namespace events_mutually_exclusive_l1517_151728

-- Define the sample space
def Ω : Type := Fin 3 → Bool

-- Define the events
def A (ω : Ω) : Prop := ∀ i, ω i = true
def B (ω : Ω) : Prop := ∀ i, ω i = false
def C (ω : Ω) : Prop := ∃ i j, ω i ≠ ω j

-- Theorem statement
theorem events_mutually_exclusive :
  (∀ ω, ¬(A ω ∧ B ω)) ∧
  (∀ ω, ¬(A ω ∧ C ω)) ∧
  (∀ ω, ¬(B ω ∧ C ω)) :=
sorry

end events_mutually_exclusive_l1517_151728


namespace sequence_characterization_l1517_151769

theorem sequence_characterization (a : ℕ+ → ℝ) :
  (∀ m n : ℕ+, a (m + n) = a m + a n - (m * n : ℝ)) ∧
  (∀ m n : ℕ+, a (m * n) = (m ^ 2 : ℝ) * a n + (n ^ 2 : ℝ) * a m + 2 * a m * a n) →
  (∀ n : ℕ+, a n = -(n * (n - 1) : ℝ) / 2) ∨
  (∀ n : ℕ+, a n = -(n ^ 2 : ℝ) / 2) := by
sorry

end sequence_characterization_l1517_151769


namespace stream_rate_calculation_l1517_151701

/-- The speed of the man rowing in still water (in kmph) -/
def still_water_speed : ℝ := 36

/-- The ratio of time taken to row upstream vs downstream -/
def upstream_downstream_ratio : ℝ := 3

/-- The rate of the stream (in kmph) -/
def stream_rate : ℝ := 18

theorem stream_rate_calculation :
  let d : ℝ := 1  -- Arbitrary distance
  let downstream_time := d / (still_water_speed + stream_rate)
  let upstream_time := d / (still_water_speed - stream_rate)
  upstream_time = upstream_downstream_ratio * downstream_time →
  stream_rate = 18 := by
sorry

end stream_rate_calculation_l1517_151701


namespace remainder_problem_l1517_151756

theorem remainder_problem (x : ℤ) : 
  (4 * x) % 7 = 6 → x % 7 = 5 := by
sorry

end remainder_problem_l1517_151756


namespace gcd_of_squares_l1517_151704

theorem gcd_of_squares : Nat.gcd (130^2 + 251^2 + 372^2) (129^2 + 250^2 + 373^2) = 15 := by
  sorry

end gcd_of_squares_l1517_151704


namespace apple_loss_fraction_l1517_151796

/-- Calculates the fraction of loss given the cost price and selling price -/
def fractionOfLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice

theorem apple_loss_fraction :
  fractionOfLoss 19 18 = 1 / 19 := by
  sorry

end apple_loss_fraction_l1517_151796


namespace unique_number_property_l1517_151715

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 3 := by sorry

end unique_number_property_l1517_151715


namespace ceiling_sqrt_225_l1517_151734

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by sorry

end ceiling_sqrt_225_l1517_151734


namespace cell_phone_providers_l1517_151764

theorem cell_phone_providers (n : ℕ) (k : ℕ) : n = 25 ∧ k = 4 → (n - 0) * (n - 1) * (n - 2) * (n - 3) = 303600 := by
  sorry

end cell_phone_providers_l1517_151764


namespace total_distance_triangle_l1517_151780

theorem total_distance_triangle (XZ XY : ℝ) (h1 : XZ = 5000) (h2 : XY = 5200) :
  let YZ := Real.sqrt (XY ^ 2 - XZ ^ 2)
  XZ + XY + YZ = 11628 := by
  sorry

end total_distance_triangle_l1517_151780


namespace sin_270_degrees_l1517_151726

theorem sin_270_degrees : Real.sin (270 * π / 180) = -1 := by
  sorry

end sin_270_degrees_l1517_151726


namespace x_squared_minus_2x_minus_3_is_quadratic_l1517_151795

/-- Definition of a quadratic equation -/
def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∀ x, a * x^2 + b * x + c = 0 → true

/-- The equation x² - 2x - 3 = 0 is a quadratic equation -/
theorem x_squared_minus_2x_minus_3_is_quadratic :
  is_quadratic_equation 1 (-2) (-3) := by
  sorry

end x_squared_minus_2x_minus_3_is_quadratic_l1517_151795


namespace s_upper_bound_l1517_151777

/-- Represents a triangle with side lengths p, q, r -/
structure Triangle where
  p : ℝ
  q : ℝ
  r : ℝ
  h_positive : 0 < p ∧ 0 < q ∧ 0 < r
  h_inequality : p ≤ r ∧ r ≤ q
  h_triangle : p + r > q ∧ q + r > p ∧ p + q > r
  h_ratio : p / (q + r) = r / (p + q)

/-- Represents a point inside the triangle -/
structure InnerPoint (t : Triangle) where
  x : ℝ
  y : ℝ
  z : ℝ
  h_inside : x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z < t.p + t.q + t.r

/-- The sum of distances from inner point to sides -/
def s (t : Triangle) (p : InnerPoint t) : ℝ := p.x + p.y + p.z

/-- The theorem to be proved -/
theorem s_upper_bound (t : Triangle) (p : InnerPoint t) : s t p ≤ 3 * t.p := by sorry

end s_upper_bound_l1517_151777


namespace series_sum_equals_half_l1517_151771

/-- The sum of the series Σ(k=1 to ∞) 3^k / (9^k - 1) is equal to 1/2 -/
theorem series_sum_equals_half :
  ∑' k, (3 : ℝ)^k / (9^k - 1) = 1/2 := by
  sorry

end series_sum_equals_half_l1517_151771


namespace solve_bones_problem_l1517_151749

def bones_problem (initial_bones final_bones : ℕ) : Prop :=
  let doubled_bones := 2 * initial_bones
  let stolen_bones := doubled_bones - final_bones
  stolen_bones = 2

theorem solve_bones_problem :
  bones_problem 4 6 := by sorry

end solve_bones_problem_l1517_151749


namespace parabola_transformation_transformed_vertex_l1517_151781

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := 2 * x^2

/-- The transformed parabola function -/
def transformed_parabola (x : ℝ) : ℝ := 2 * (x - 4)^2 - 1

/-- Theorem stating that the transformed parabola is a shift of the original parabola -/
theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 4) - 1 := by
  sorry

/-- Corollary showing the vertex of the transformed parabola -/
theorem transformed_vertex :
  ∃ x y : ℝ, x = 4 ∧ y = -1 ∧ ∀ t : ℝ, transformed_parabola t ≥ transformed_parabola x := by
  sorry

end parabola_transformation_transformed_vertex_l1517_151781


namespace unique_solution_l1517_151773

theorem unique_solution (a b c : ℝ) 
  (ha : a > 4) (hb : b > 4) (hc : c > 4)
  (heq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  a = 12 ∧ b = 10 ∧ c = 8 := by
  sorry

end unique_solution_l1517_151773


namespace max_d_value_l1517_151751

def a (n : ℕ+) : ℕ := 101 + n.val ^ 2 + 3 * n.val

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∀ n : ℕ+, d n ≤ 4 ∧ ∃ m : ℕ+, d m = 4 :=
sorry

end max_d_value_l1517_151751


namespace volume_P_5_l1517_151779

/-- Represents the volume of the dodecahedron after i iterations -/
def P (i : ℕ) : ℚ :=
  sorry

/-- The height of the tetrahedra at step i -/
def r (i : ℕ) : ℚ :=
  (1 / 2) ^ i

/-- The initial dodecahedron has volume 1 -/
axiom P_0 : P 0 = 1

/-- The recursive definition of P(i+1) based on P(i) and r(i) -/
axiom P_step (i : ℕ) : P (i + 1) = P i + 6 * (1 / 3) * (r i)^3

/-- The main theorem: the volume of P₅ is 8929/4096 -/
theorem volume_P_5 : P 5 = 8929 / 4096 :=
  sorry

end volume_P_5_l1517_151779


namespace absolute_value_five_l1517_151737

theorem absolute_value_five (x : ℝ) : |x| = 5 → x = 5 ∨ x = -5 := by
  sorry

end absolute_value_five_l1517_151737


namespace imaginary_part_of_complex_fraction_l1517_151729

theorem imaginary_part_of_complex_fraction (i : Complex) :
  i * i = -1 →
  Complex.im ((1 + 2*i) / (1 + i)) = 1/2 := by
  sorry

end imaginary_part_of_complex_fraction_l1517_151729


namespace pure_imaginary_implies_m_eq_neg_two_fourth_quadrant_implies_m_lt_neg_two_m_eq_two_implies_sum_of_parts_l1517_151730

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (m - 1) * (m + 2) + (m - 1) * Complex.I

-- Theorem 1: If z is a pure imaginary number, then m = -2
theorem pure_imaginary_implies_m_eq_neg_two (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 → m = -2 := by sorry

-- Theorem 2: If z is in the fourth quadrant, then m < -2
theorem fourth_quadrant_implies_m_lt_neg_two (m : ℝ) :
  (z m).re > 0 ∧ (z m).im < 0 → m < -2 := by sorry

-- Theorem 3: If m = 2, then (z+i)/(z-1) = a + bi where a + b = 8/5
theorem m_eq_two_implies_sum_of_parts (m : ℝ) :
  m = 2 →
  ∃ a b : ℝ, (z m + Complex.I) / (z m - 1) = a + b * Complex.I ∧ a + b = 8/5 := by sorry

end pure_imaginary_implies_m_eq_neg_two_fourth_quadrant_implies_m_lt_neg_two_m_eq_two_implies_sum_of_parts_l1517_151730


namespace probability_even_sum_two_wheels_l1517_151724

theorem probability_even_sum_two_wheels : 
  let wheel1_total := 6
  let wheel1_even := 3
  let wheel2_total := 4
  let wheel2_even := 3
  let prob_wheel1_even := wheel1_even / wheel1_total
  let prob_wheel1_odd := 1 - prob_wheel1_even
  let prob_wheel2_even := wheel2_even / wheel2_total
  let prob_wheel2_odd := 1 - prob_wheel2_even
  let prob_both_even := prob_wheel1_even * prob_wheel2_even
  let prob_both_odd := prob_wheel1_odd * prob_wheel2_odd
  prob_both_even + prob_both_odd = 1/2
:= by sorry

end probability_even_sum_two_wheels_l1517_151724


namespace system_solution_l1517_151798

theorem system_solution (x y : ℚ) (h1 : x + 2*y = -1) (h2 : 2*x + y = 3) : x + y = 2/3 := by
  sorry

end system_solution_l1517_151798


namespace smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11_l1517_151785

theorem smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11 :
  ∃ w : ℕ, w > 0 ∧ w % 13 = 0 ∧ (w + 3) % 11 = 0 ∧
  ∀ x : ℕ, x > 0 ∧ x % 13 = 0 ∧ (x + 3) % 11 = 0 → w ≤ x :=
by
  use 52
  sorry

end smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11_l1517_151785


namespace hyperbola_I_equation_equilateral_hyperbola_equation_l1517_151709

-- Part I
def hyperbola_I (x y : ℝ) : Prop :=
  y^2 / 36 - x^2 / 28 = 1

theorem hyperbola_I_equation
  (foci_on_y_axis : True)
  (focal_distance : ℝ)
  (h_focal_distance : focal_distance = 16)
  (eccentricity : ℝ)
  (h_eccentricity : eccentricity = 4/3) :
  ∃ (x y : ℝ), hyperbola_I x y :=
sorry

-- Part II
def equilateral_hyperbola (x y : ℝ) : Prop :=
  x^2 / 18 - y^2 / 18 = 1

theorem equilateral_hyperbola_equation
  (is_equilateral : True)
  (focus : ℝ × ℝ)
  (h_focus : focus = (-6, 0)) :
  ∃ (x y : ℝ), equilateral_hyperbola x y :=
sorry

end hyperbola_I_equation_equilateral_hyperbola_equation_l1517_151709


namespace equation_solution_l1517_151783

theorem equation_solution : 
  ∃! x : ℚ, (x + 7) / (x - 4) = (x - 5) / (x + 3) ∧ x = -1/19 := by sorry

end equation_solution_l1517_151783


namespace impossible_to_reach_all_plus_l1517_151706

/- Define the sign type -/
inductive Sign : Type
| Plus : Sign
| Minus : Sign

/- Define the 4x4 grid type -/
def Grid := Matrix (Fin 4) (Fin 4) Sign

/- Define the initial grid -/
def initial_grid : Grid :=
  λ i j => match i, j with
  | 0, 1 => Sign.Minus
  | 3, 1 => Sign.Minus
  | _, _ => Sign.Plus

/- Define a move (flipping a row or column) -/
def flip_row (g : Grid) (row : Fin 4) : Grid :=
  λ i j => if i = row then
    match g i j with
    | Sign.Plus => Sign.Minus
    | Sign.Minus => Sign.Plus
    else g i j

def flip_column (g : Grid) (col : Fin 4) : Grid :=
  λ i j => if j = col then
    match g i j with
    | Sign.Plus => Sign.Minus
    | Sign.Minus => Sign.Plus
    else g i j

/- Define the goal state (all Plus signs) -/
def all_plus (g : Grid) : Prop :=
  ∀ i j, g i j = Sign.Plus

/- The main theorem -/
theorem impossible_to_reach_all_plus :
  ¬ ∃ (moves : List (Sum (Fin 4) (Fin 4))),
    all_plus (moves.foldl (λ g move => match move with
      | Sum.inl row => flip_row g row
      | Sum.inr col => flip_column g col) initial_grid) :=
sorry

end impossible_to_reach_all_plus_l1517_151706


namespace quiz_bowl_points_per_answer_l1517_151720

/-- Represents the quiz bowl game structure and James' performance --/
structure QuizBowl where
  total_rounds : Nat
  questions_per_round : Nat
  bonus_points : Nat
  james_total_points : Nat
  james_missed_questions : Nat

/-- Calculates the points per correct answer in the quiz bowl --/
def points_per_correct_answer (qb : QuizBowl) : Nat :=
  let total_questions := qb.total_rounds * qb.questions_per_round
  let james_correct_answers := total_questions - qb.james_missed_questions
  let perfect_rounds := (james_correct_answers / qb.questions_per_round)
  let bonus_total := perfect_rounds * qb.bonus_points
  let points_from_answers := qb.james_total_points - bonus_total
  points_from_answers / james_correct_answers

/-- Theorem stating that given the specific conditions, the points per correct answer is 2 --/
theorem quiz_bowl_points_per_answer :
  let qb : QuizBowl := {
    total_rounds := 5,
    questions_per_round := 5,
    bonus_points := 4,
    james_total_points := 66,
    james_missed_questions := 1
  }
  points_per_correct_answer qb = 2 := by
  sorry

end quiz_bowl_points_per_answer_l1517_151720


namespace sequence_properties_l1517_151732

-- Define the sequence S_n
def S (n : ℕ+) : ℚ := n^2 + n

-- Define the sequence a_n
def a (n : ℕ+) : ℚ := 2 * n

-- Define the sequence T_n
def T (n : ℕ+) : ℚ := (1 / 2) * (1 - 1 / (n + 1))

theorem sequence_properties :
  (∀ n : ℕ+, S n = n^2 + n) →
  (∀ n : ℕ+, a n = 2 * n) ∧
  (∀ n : ℕ+, T n = (1 / 2) * (1 - 1 / (n + 1))) :=
by
  sorry

end sequence_properties_l1517_151732


namespace pattern_properties_l1517_151723

/-- Represents a figure in the pattern -/
structure Figure where
  n : ℕ

/-- Number of squares in a figure -/
def num_squares (f : Figure) : ℕ :=
  3 + 2 * (f.n - 1)

/-- Perimeter of a figure in cm -/
def perimeter (f : Figure) : ℕ :=
  8 + 2 * (f.n - 1)

theorem pattern_properties :
  ∀ (f : Figure),
    (num_squares f = 3 + 2 * (f.n - 1)) ∧
    (perimeter f = 8 + 2 * (f.n - 1)) ∧
    (perimeter ⟨16⟩ = 38) ∧
    ((perimeter ⟨29⟩ : ℚ) / (perimeter ⟨85⟩ : ℚ) = 4 / 11) :=
by sorry

end pattern_properties_l1517_151723


namespace compound_interest_principal_l1517_151742

def simple_interest_rate : ℝ := 0.14
def compound_interest_rate : ℝ := 0.07
def simple_interest_time : ℝ := 6
def compound_interest_time : ℝ := 2
def simple_interest_amount : ℝ := 603.75

theorem compound_interest_principal (P_SI : ℝ) (P_CI : ℝ) 
  (h1 : P_SI * simple_interest_rate * simple_interest_time = simple_interest_amount)
  (h2 : P_SI * simple_interest_rate * simple_interest_time = 
        1/2 * (P_CI * ((1 + compound_interest_rate) ^ compound_interest_time - 1)))
  (h3 : P_SI = 603.75 / (simple_interest_rate * simple_interest_time)) :
  P_CI = 8333.33 := by
sorry

end compound_interest_principal_l1517_151742


namespace specific_trapezoid_area_l1517_151787

/-- Represents a trapezoid ABCD with given properties -/
structure IsoscelesTrapezoid where
  AB : ℝ
  CD : ℝ
  AD_eq_BC : AD = BC
  O_interior : O_in_interior
  OT : ℝ

/-- The area of the isosceles trapezoid with the given properties -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  ∃ (t : IsoscelesTrapezoid),
    t.AB = 6 ∧ t.CD = 12 ∧ t.OT = 18 ∧
    trapezoid_area t = 54 + 27 * Real.sqrt 3 := by
  sorry

end specific_trapezoid_area_l1517_151787


namespace product_abc_equals_195_l1517_151748

/-- Given the conditions on products of variables a, b, c, d, e, f,
    prove that a * b * c equals 195. -/
theorem product_abc_equals_195
  (h1 : b * c * d = 65)
  (h2 : c * d * e = 1000)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 3/4)
  : a * b * c = 195 := by
  sorry

end product_abc_equals_195_l1517_151748


namespace five_athletes_three_events_l1517_151757

/-- The number of different ways athletes can win championships in events -/
def championship_ways (num_athletes : ℕ) (num_events : ℕ) : ℕ :=
  num_athletes ^ num_events

/-- Theorem: 5 athletes winning 3 events results in 5^3 different ways -/
theorem five_athletes_three_events : 
  championship_ways 5 3 = 5^3 := by
  sorry

end five_athletes_three_events_l1517_151757


namespace problem_statement_l1517_151774

theorem problem_statement (a b c : ℝ) :
  (∀ c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) := by
  sorry

end problem_statement_l1517_151774


namespace factorial_fraction_l1517_151718

theorem factorial_fraction (N : ℕ) : 
  (Nat.factorial (N - 1) * (N^2 + N)) / Nat.factorial (N + 2) = 1 / (N + 2) :=
sorry

end factorial_fraction_l1517_151718


namespace plain_pancakes_count_l1517_151700

/-- Given a total of 67 pancakes, with 20 having blueberries and 24 having bananas,
    prove that there are 23 plain pancakes. -/
theorem plain_pancakes_count (total : ℕ) (blueberry : ℕ) (banana : ℕ) 
  (h1 : total = 67) 
  (h2 : blueberry = 20) 
  (h3 : banana = 24) :
  total - (blueberry + banana) = 23 := by
  sorry

#check plain_pancakes_count

end plain_pancakes_count_l1517_151700


namespace equation_linearity_l1517_151782

/-- The equation (k^2 - 1)x^2 + (k + 1)x + (k - 7)y = k + 2 -/
def equation (k x y : ℝ) : Prop :=
  (k^2 - 1) * x^2 + (k + 1) * x + (k - 7) * y = k + 2

/-- The equation is linear in one variable -/
def is_linear_one_var (k : ℝ) : Prop :=
  k^2 - 1 = 0 ∧ k + 1 = 0

/-- The equation is linear in two variables -/
def is_linear_two_var (k : ℝ) : Prop :=
  k^2 - 1 = 0 ∧ k + 1 ≠ 0

theorem equation_linearity :
  (is_linear_one_var (-1) ∧ is_linear_two_var 1) :=
by sorry

end equation_linearity_l1517_151782


namespace whitewashing_cost_is_1812_l1517_151746

/-- Calculates the cost of white washing a room with given dimensions and openings. -/
def whitewashingCost (length width height : ℝ) (doorHeight doorWidth : ℝ) 
  (windowHeight windowWidth : ℝ) (numWindows : ℕ) (ratePerSqFt : ℝ) : ℝ :=
  let wallArea := 2 * (length + width) * height
  let doorArea := doorHeight * doorWidth
  let windowArea := windowHeight * windowWidth * (numWindows : ℝ)
  let adjustedArea := wallArea - doorArea - windowArea
  adjustedArea * ratePerSqFt

/-- The cost of white washing the room is Rs. 1812. -/
theorem whitewashing_cost_is_1812 :
  whitewashingCost 25 15 12 6 3 4 3 3 2 = 1812 := by
  sorry

end whitewashing_cost_is_1812_l1517_151746


namespace cube_root_equation_solutions_l1517_151716

theorem cube_root_equation_solutions :
  ∀ x : ℝ, (x^(1/3) = 15 / (10 - x^(1/3))) ↔ (x = 125 ∨ x = 27) := by
  sorry

end cube_root_equation_solutions_l1517_151716


namespace tan_theta_value_l1517_151731

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_theta_value (θ : Real) (h : ∃ (t : Real), t > 0 ∧ t * (-Real.sqrt 3 / 2) = Real.cos θ ∧ t * (1 / 2) = Real.sin θ) : 
  Real.tan θ = -Real.sqrt 3 / 3 := by
sorry

end tan_theta_value_l1517_151731


namespace jordan_rectangle_length_l1517_151703

theorem jordan_rectangle_length (carol_length carol_width jordan_width : ℝ) 
  (h1 : carol_length = 5)
  (h2 : carol_width = 24)
  (h3 : jordan_width = 30) :
  carol_length * carol_width = jordan_width * (120 / jordan_width) := by
  sorry

#check jordan_rectangle_length

end jordan_rectangle_length_l1517_151703


namespace min_marked_cells_l1517_151763

/-- Represents a board with dimensions m × n -/
structure Board (m n : ℕ) where
  cells : Fin m → Fin n → Bool

/-- Represents an L-shaped piece on the board -/
inductive LShape
  | makeL : Fin 2 → Fin 2 → LShape

/-- Checks if an L-shape touches a marked cell on the board -/
def touchesMarked (b : Board m n) (l : LShape) : Bool :=
  sorry

/-- Checks if a marking strategy satisfies the condition for all L-shape placements -/
def validMarking (b : Board m n) : Prop :=
  ∀ l : LShape, touchesMarked b l = true

/-- Counts the number of marked cells on the board -/
def countMarked (b : Board m n) : ℕ :=
  sorry

/-- The main theorem stating that 50 is the smallest number of cells to be marked -/
theorem min_marked_cells :
  ∃ (b : Board 10 11), validMarking b ∧ countMarked b = 50 ∧
  ∀ (b' : Board 10 11), validMarking b' → countMarked b' ≥ 50 :=
sorry

end min_marked_cells_l1517_151763


namespace angle_terminal_side_l1517_151727

/-- Given that the terminal side of angle α passes through point (a, 1) and tan α = -1/2, prove that a = -2 -/
theorem angle_terminal_side (α : Real) (a : Real) : 
  (∃ (x y : Real), x = a ∧ y = 1 ∧ Real.tan α = y / x) → 
  Real.tan α = -1/2 → 
  a = -2 := by
sorry

end angle_terminal_side_l1517_151727


namespace complement_M_in_U_l1517_151721

def U : Set ℕ := {x | x < 5 ∧ x > 0}
def M : Set ℕ := {x | x ^ 2 - 5 * x + 6 = 0}

theorem complement_M_in_U : U \ M = {1, 4} := by sorry

end complement_M_in_U_l1517_151721


namespace triangle_area_from_perimeter_and_inradius_l1517_151745

/-- Theorem: Area of a triangle with given perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) 
  (inradius : ℝ) 
  (h_perimeter : perimeter = 42) 
  (h_inradius : inradius = 5) : 
  inradius * (perimeter / 2) = 105 := by
sorry

end triangle_area_from_perimeter_and_inradius_l1517_151745


namespace inequality_of_positive_reals_l1517_151710

theorem inequality_of_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end inequality_of_positive_reals_l1517_151710


namespace strawberry_weight_sum_l1517_151750

/-- The total weight of Marco's and his dad's strawberries is 23 pounds. -/
theorem strawberry_weight_sum : 
  let marco_weight : ℕ := 14
  let dad_weight : ℕ := 9
  marco_weight + dad_weight = 23 := by sorry

end strawberry_weight_sum_l1517_151750


namespace prob_n_minus_one_matches_is_zero_l1517_151767

/-- Represents the number of pairs in the matching problem -/
def n : ℕ := 10

/-- Represents a function that returns the probability of correctly matching
    exactly k pairs out of n pairs when choosing randomly -/
noncomputable def probability_exact_matches (k : ℕ) : ℝ := sorry

/-- Theorem stating that the probability of correctly matching exactly n-1 pairs
    out of n pairs is 0 when choosing randomly -/
theorem prob_n_minus_one_matches_is_zero :
  probability_exact_matches (n - 1) = 0 := by sorry

end prob_n_minus_one_matches_is_zero_l1517_151767


namespace certain_number_proof_l1517_151733

theorem certain_number_proof : ∃! x : ℚ, x / 4 + 3 = 5 ∧ x = 8 := by
  sorry

end certain_number_proof_l1517_151733
