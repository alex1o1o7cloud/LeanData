import Mathlib

namespace NUMINAMATH_CALUDE_common_number_in_overlapping_sets_l3600_360054

theorem common_number_in_overlapping_sets (numbers : List ℝ) : 
  numbers.length = 9 ∧ 
  (numbers.take 5).sum / 5 = 7 ∧ 
  (numbers.drop 4).sum / 5 = 9 ∧ 
  numbers.sum / 9 = 25 / 3 →
  ∃ x ∈ numbers.take 5 ∩ numbers.drop 4, x = 5 := by
sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_sets_l3600_360054


namespace NUMINAMATH_CALUDE_john_vowel_learning_days_l3600_360012

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of days John takes to learn one alphabet -/
def days_per_alphabet : ℕ := 3

/-- The total number of days John needs to finish learning all vowels -/
def total_days : ℕ := num_vowels * days_per_alphabet

theorem john_vowel_learning_days : total_days = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_vowel_learning_days_l3600_360012


namespace NUMINAMATH_CALUDE_exists_permutation_divisible_by_seven_l3600_360080

def digits : List Nat := [1, 3, 7, 9]

def is_permutation (l1 l2 : List Nat) : Prop :=
  l1.length = l2.length ∧ ∀ x, l1.count x = l2.count x

def list_to_number (l : List Nat) : Nat :=
  l.foldl (fun acc d => acc * 10 + d) 0

theorem exists_permutation_divisible_by_seven :
  ∃ perm : List Nat, is_permutation digits perm ∧ 
    (list_to_number perm) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_permutation_divisible_by_seven_l3600_360080


namespace NUMINAMATH_CALUDE_total_cars_in_week_is_450_l3600_360072

/-- The number of cars passing through a toll booth in a week -/
def totalCarsInWeek (mondayCars : ℕ) : ℕ :=
  -- Monday and Tuesday
  2 * mondayCars +
  -- Wednesday and Thursday
  2 * (2 * mondayCars) +
  -- Friday, Saturday, and Sunday
  3 * mondayCars

/-- Theorem stating that the total number of cars in a week is 450 -/
theorem total_cars_in_week_is_450 : totalCarsInWeek 50 = 450 := by
  sorry

#eval totalCarsInWeek 50

end NUMINAMATH_CALUDE_total_cars_in_week_is_450_l3600_360072


namespace NUMINAMATH_CALUDE_beth_crayons_l3600_360040

/-- The number of crayon packs Beth has -/
def num_packs : ℕ := 4

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 10

/-- The number of extra crayons Beth has -/
def extra_crayons : ℕ := 6

/-- The total number of crayons Beth has -/
def total_crayons : ℕ := num_packs * crayons_per_pack + extra_crayons

theorem beth_crayons : total_crayons = 46 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayons_l3600_360040


namespace NUMINAMATH_CALUDE_lcm_of_48_and_14_l3600_360066

theorem lcm_of_48_and_14 (n : ℕ) (h1 : n = 48) (h2 : Nat.gcd n 14 = 12) :
  Nat.lcm n 14 = 56 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_48_and_14_l3600_360066


namespace NUMINAMATH_CALUDE_one_pepperoni_fell_off_l3600_360034

/-- Represents a pizza with pepperoni slices -/
structure Pizza :=
  (total_pepperoni : ℕ)
  (num_slices : ℕ)
  (pepperoni_on_given_slice : ℕ)

/-- Calculates the number of pepperoni slices that fell off -/
def pepperoni_fell_off (p : Pizza) : ℕ :=
  (p.total_pepperoni / p.num_slices) - p.pepperoni_on_given_slice

/-- Theorem stating that one pepperoni slice fell off -/
theorem one_pepperoni_fell_off (p : Pizza) 
    (h1 : p.total_pepperoni = 40)
    (h2 : p.num_slices = 4)
    (h3 : p.pepperoni_on_given_slice = 9) : 
  pepperoni_fell_off p = 1 := by
  sorry

#eval pepperoni_fell_off { total_pepperoni := 40, num_slices := 4, pepperoni_on_given_slice := 9 }

end NUMINAMATH_CALUDE_one_pepperoni_fell_off_l3600_360034


namespace NUMINAMATH_CALUDE_bushes_needed_for_perfume_l3600_360064

/-- The number of rose petals needed to make an ounce of perfume -/
def petals_per_ounce : ℕ := 320

/-- The number of petals produced by each rose -/
def petals_per_rose : ℕ := 8

/-- The number of roses per bush -/
def roses_per_bush : ℕ := 12

/-- The number of bottles of perfume to be made -/
def num_bottles : ℕ := 20

/-- The number of ounces in each bottle of perfume -/
def ounces_per_bottle : ℕ := 12

/-- The theorem stating the number of bushes needed to make the required perfume -/
theorem bushes_needed_for_perfume : 
  (petals_per_ounce * num_bottles * ounces_per_bottle) / (petals_per_rose * roses_per_bush) = 800 := by
  sorry

end NUMINAMATH_CALUDE_bushes_needed_for_perfume_l3600_360064


namespace NUMINAMATH_CALUDE_marcel_total_cost_l3600_360090

/-- The cost of Marcel's purchases -/
def total_cost (pen_price briefcase_price : ℝ) : ℝ :=
  pen_price + briefcase_price

/-- Theorem: Marcel's total cost for a pen and briefcase is $24 -/
theorem marcel_total_cost :
  ∃ (pen_price briefcase_price : ℝ),
    pen_price = 4 ∧
    briefcase_price = 5 * pen_price ∧
    total_cost pen_price briefcase_price = 24 := by
  sorry

end NUMINAMATH_CALUDE_marcel_total_cost_l3600_360090


namespace NUMINAMATH_CALUDE_laticia_socks_l3600_360059

/-- Proves that Laticia knitted 13 pairs of socks in the first week -/
theorem laticia_socks (x : ℕ) : x + (x + 4) + (x + 2) + (x - 1) = 57 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_laticia_socks_l3600_360059


namespace NUMINAMATH_CALUDE_tank_empty_time_l3600_360041

/- Define the tank capacity in liters -/
def tank_capacity : ℝ := 5760

/- Define the time it takes for the leak to empty the tank in hours -/
def leak_empty_time : ℝ := 6

/- Define the inlet pipe fill rate in liters per minute -/
def inlet_fill_rate : ℝ := 4

/- Define the time it takes to empty the tank with inlet open in hours -/
def empty_time_with_inlet : ℝ := 8

/- Theorem statement -/
theorem tank_empty_time :
  let leak_rate := tank_capacity / leak_empty_time
  let inlet_rate := inlet_fill_rate * 60
  let net_empty_rate := leak_rate - inlet_rate
  tank_capacity / net_empty_rate = empty_time_with_inlet :=
by sorry

end NUMINAMATH_CALUDE_tank_empty_time_l3600_360041


namespace NUMINAMATH_CALUDE_integral_inequality_l3600_360047

theorem integral_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 1) :
  ∫ x in (0 : ℝ)..1, ((1 - a*x)^3 + (1 - b*x)^3 + (1 - c*x)^3 - 3*x) ≥ 
    a*b + b*c + c*a - 3/2*(a + b + c) - 3/4*a*b*c := by
  sorry

end NUMINAMATH_CALUDE_integral_inequality_l3600_360047


namespace NUMINAMATH_CALUDE_ali_baba_strategy_l3600_360056

/-- A game with diamonds where players split piles. -/
structure DiamondGame where
  total_diamonds : ℕ
  
/-- The number of moves required to end the game. -/
def moves_to_end (game : DiamondGame) : ℕ :=
  game.total_diamonds - 1

/-- Determines if the second player wins the game. -/
def second_player_wins (game : DiamondGame) : Prop :=
  Even (moves_to_end game)

/-- Theorem: In a game with 2017 diamonds, the second player wins. -/
theorem ali_baba_strategy (game : DiamondGame) (h : game.total_diamonds = 2017) :
  second_player_wins game := by
  sorry

#eval moves_to_end { total_diamonds := 2017 }

end NUMINAMATH_CALUDE_ali_baba_strategy_l3600_360056


namespace NUMINAMATH_CALUDE_prime_pair_equation_solution_l3600_360042

theorem prime_pair_equation_solution :
  ∀ p q : ℕ,
  Prime p → Prime q →
  (∃ m : ℕ+, (p * q : ℚ) / (p + q) = (m.val^2 + 6 : ℚ) / (m.val + 1)) →
  (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := by
sorry

end NUMINAMATH_CALUDE_prime_pair_equation_solution_l3600_360042


namespace NUMINAMATH_CALUDE_maddy_chocolate_eggs_l3600_360016

/-- The number of chocolate eggs Maddy eats per day -/
def eggs_per_day : ℕ := 2

/-- The number of weeks the chocolate eggs last -/
def weeks_lasting : ℕ := 4

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Maddy was given 56 chocolate eggs -/
theorem maddy_chocolate_eggs :
  eggs_per_day * weeks_lasting * days_in_week = 56 := by
  sorry

end NUMINAMATH_CALUDE_maddy_chocolate_eggs_l3600_360016


namespace NUMINAMATH_CALUDE_systematic_sampling_size_l3600_360063

/-- Proves that the sample size for systematic sampling is 6 given the conditions of the problem -/
theorem systematic_sampling_size (total_population : Nat) (n : Nat) 
  (h1 : total_population = 36)
  (h2 : total_population % n = 0)
  (h3 : (total_population - 1) % (n + 1) = 0) : 
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_size_l3600_360063


namespace NUMINAMATH_CALUDE_resistor_value_l3600_360077

/-- The resistance of a single resistor in a circuit where three identical resistors are initially 
    in series, and then connected in parallel, such that the change in total resistance is 10 Ω. -/
theorem resistor_value (R : ℝ) : 
  (3 * R - R / 3 = 10) → R = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_resistor_value_l3600_360077


namespace NUMINAMATH_CALUDE_ones_digit_of_triple_4567_l3600_360033

def triple_number (n : ℕ) : ℕ := 3 * n

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_triple_4567 :
  ones_digit (triple_number 4567) = 1 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_triple_4567_l3600_360033


namespace NUMINAMATH_CALUDE_notebook_word_count_l3600_360051

theorem notebook_word_count (total_pages : Nat) (max_words_per_page : Nat) 
  (h1 : total_pages = 150)
  (h2 : max_words_per_page = 90)
  (h3 : ∃ (words_per_page : Nat), words_per_page ≤ max_words_per_page ∧ 
        (total_pages * words_per_page) % 221 = 210) :
  ∃ (words_per_page : Nat), words_per_page = 90 ∧ 
    words_per_page ≤ max_words_per_page ∧ 
    (total_pages * words_per_page) % 221 = 210 := by
  sorry

end NUMINAMATH_CALUDE_notebook_word_count_l3600_360051


namespace NUMINAMATH_CALUDE_tree_planting_equation_correct_l3600_360083

/-- Represents the tree planting scenario -/
structure TreePlanting where
  totalTrees : ℕ
  originalRate : ℝ
  actualRateFactor : ℝ
  daysAhead : ℕ

/-- The equation representing the tree planting scenario is correct -/
theorem tree_planting_equation_correct (tp : TreePlanting)
  (h1 : tp.totalTrees = 960)
  (h2 : tp.originalRate > 0)
  (h3 : tp.actualRateFactor = 4/3)
  (h4 : tp.daysAhead = 4) :
  (tp.totalTrees : ℝ) / tp.originalRate - (tp.totalTrees : ℝ) / (tp.actualRateFactor * tp.originalRate) = tp.daysAhead :=
sorry

end NUMINAMATH_CALUDE_tree_planting_equation_correct_l3600_360083


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l3600_360082

/-- The volume ratio of two right circular cylinders -/
theorem cylinder_volume_ratio :
  let r1 := 4 / Real.pi
  let h1 := 10
  let r2 := 5 / Real.pi
  let h2 := 8
  let v1 := Real.pi * r1^2 * h1
  let v2 := Real.pi * r2^2 * h2
  v1 / v2 = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l3600_360082


namespace NUMINAMATH_CALUDE_train_length_proof_l3600_360004

/-- Proves that a train with the given conditions has a length of 1800 meters -/
theorem train_length_proof (train_speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) : 
  train_speed = 216 →
  crossing_time = 1 →
  train_length = 1800 :=
by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l3600_360004


namespace NUMINAMATH_CALUDE_cloth_sold_calculation_l3600_360085

/-- The number of meters of cloth sold by a trader -/
def meters_of_cloth : ℕ := 40

/-- The profit per meter of cloth in rupees -/
def profit_per_meter : ℕ := 30

/-- The total profit earned by the trader in rupees -/
def total_profit : ℕ := 1200

/-- Theorem stating that the number of meters of cloth sold is 40 -/
theorem cloth_sold_calculation :
  meters_of_cloth * profit_per_meter = total_profit :=
by sorry

end NUMINAMATH_CALUDE_cloth_sold_calculation_l3600_360085


namespace NUMINAMATH_CALUDE_expression_evaluation_l3600_360095

/-- Proves that the given expression evaluates to -8 when a = 2 and b = -1 -/
theorem expression_evaluation :
  let a : ℤ := 2
  let b : ℤ := -1
  3 * (2 * a^2 * b - 3 * a * b^2 - 1) - 2 * (3 * a^2 * b - 4 * a * b^2 + 1) - 1 = -8 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3600_360095


namespace NUMINAMATH_CALUDE_smallest_x_cosine_equality_l3600_360000

theorem smallest_x_cosine_equality : ∃ x : ℝ, 
  (x > 0 ∧ x < 30) ∧ 
  (∀ y : ℝ, y > 0 ∧ y < 30 ∧ Real.cos (3 * y * π / 180) = Real.cos ((2 * y^2 - y) * π / 180) → x ≤ y) ∧
  Real.cos (3 * x * π / 180) = Real.cos ((2 * x^2 - x) * π / 180) ∧
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_cosine_equality_l3600_360000


namespace NUMINAMATH_CALUDE_persons_age_puzzle_l3600_360098

theorem persons_age_puzzle : ∃ (x : ℕ), 5 * (x + 7) - 3 * (x - 7) = x ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_persons_age_puzzle_l3600_360098


namespace NUMINAMATH_CALUDE_sin_cos_difference_special_angle_l3600_360026

theorem sin_cos_difference_special_angle : 
  Real.sin (80 * π / 180) * Real.cos (20 * π / 180) - 
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_special_angle_l3600_360026


namespace NUMINAMATH_CALUDE_a_2016_value_l3600_360032

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 3
  | n + 1 => ⌊a n⌋ + 1 / (a n - ⌊a n⌋)

theorem a_2016_value : a 2016 = 3024 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_a_2016_value_l3600_360032


namespace NUMINAMATH_CALUDE_range_x_when_m_is_one_range_m_for_not_p_necessary_but_not_sufficient_l3600_360031

-- Define propositions p and q
def p (x m : ℝ) : Prop := |2 * x - m| ≥ 1
def q (x : ℝ) : Prop := (1 - 3 * x) / (x + 2) > 0

-- Theorem for part 1
theorem range_x_when_m_is_one :
  {x : ℝ | p x 1 ∧ q x} = {x : ℝ | -2 < x ∧ x ≤ 0} := by sorry

-- Theorem for part 2
theorem range_m_for_not_p_necessary_but_not_sufficient :
  {m : ℝ | ∀ x, q x → ¬(p x m) ∧ ∃ y, ¬(p y m) ∧ ¬(q y)} = {m : ℝ | -3 ≤ m ∧ m ≤ -1/3} := by sorry

end NUMINAMATH_CALUDE_range_x_when_m_is_one_range_m_for_not_p_necessary_but_not_sufficient_l3600_360031


namespace NUMINAMATH_CALUDE_ribbon_lengths_after_cutting_l3600_360029

def initial_lengths : List ℝ := [15, 20, 24, 26, 30]

def median (l : List ℝ) : ℝ := sorry
def range (l : List ℝ) : ℝ := sorry
def average (l : List ℝ) : ℝ := sorry

theorem ribbon_lengths_after_cutting (new_lengths : List ℝ) :
  (average new_lengths = average initial_lengths - 5) →
  (median new_lengths = median initial_lengths) →
  (range new_lengths = range initial_lengths) →
  new_lengths.length = initial_lengths.length →
  (∀ x ∈ new_lengths, x > 0) →
  new_lengths = [9, 9, 24, 24, 24] :=
by sorry

end NUMINAMATH_CALUDE_ribbon_lengths_after_cutting_l3600_360029


namespace NUMINAMATH_CALUDE_ad_duration_l3600_360045

theorem ad_duration (num_ads : ℕ) (cost_per_minute : ℕ) (total_cost : ℕ) 
  (h1 : num_ads = 5)
  (h2 : cost_per_minute = 4000)
  (h3 : total_cost = 60000) :
  (total_cost / cost_per_minute) / num_ads = 3 := by
  sorry

end NUMINAMATH_CALUDE_ad_duration_l3600_360045


namespace NUMINAMATH_CALUDE_max_type_a_workers_l3600_360005

theorem max_type_a_workers (total : ℕ) (x : ℕ) : 
  total = 150 → 
  total - x ≥ 3 * x → 
  x ≤ 37 ∧ ∃ y : ℕ, y > 37 → total - y < 3 * y :=
sorry

end NUMINAMATH_CALUDE_max_type_a_workers_l3600_360005


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3600_360093

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8/15) (h2 : x - y = 1/35) : x^2 - y^2 = 8/525 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3600_360093


namespace NUMINAMATH_CALUDE_proposition_validity_l3600_360087

theorem proposition_validity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (((a^2 - b^2 = 1) → (a - b < 1)) ∧
   ¬((1/b - 1/a = 1) → (a - b < 1)) ∧
   ((Real.exp a - Real.exp b = 1) → (a - b < 1)) ∧
   ¬((Real.log a - Real.log b = 1) → (a - b < 1))) := by
sorry

end NUMINAMATH_CALUDE_proposition_validity_l3600_360087


namespace NUMINAMATH_CALUDE_ratio_adjustment_l3600_360071

theorem ratio_adjustment (x : ℚ) : x = 29 ↔ (4 + x) / (15 + x) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_adjustment_l3600_360071


namespace NUMINAMATH_CALUDE_franklin_valentines_l3600_360009

/-- The number of Valentines Mrs. Franklin gave away -/
def valentines_given : ℕ := 42

/-- The number of Valentines Mrs. Franklin has left -/
def valentines_left : ℕ := 16

/-- The initial number of Valentines Mrs. Franklin had -/
def initial_valentines : ℕ := valentines_given + valentines_left

theorem franklin_valentines : initial_valentines = 58 := by
  sorry

end NUMINAMATH_CALUDE_franklin_valentines_l3600_360009


namespace NUMINAMATH_CALUDE_ship_passengers_l3600_360078

theorem ship_passengers : ∀ (P : ℕ),
  (P / 12 : ℚ) + (P / 8 : ℚ) + (P / 3 : ℚ) + (P / 6 : ℚ) + 35 = P →
  P = 120 := by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_l3600_360078


namespace NUMINAMATH_CALUDE_diamonds_G15_l3600_360017

/-- The number of diamonds in the nth figure of the sequence -/
def diamonds (n : ℕ) : ℕ :=
  3 * n^2 - 3 * n + 1

/-- The sequence G is constructed such that for n ≥ 3, 
    Gₙ is surrounded by a hexagon with n-1 diamonds on each of its 6 sides -/
axiom sequence_construction (n : ℕ) (h : n ≥ 3) :
  diamonds n = diamonds (n-1) + 6 * (n-1)

/-- G₁ has 1 diamond -/
axiom G1_diamonds : diamonds 1 = 1

/-- The number of diamonds in G₁₅ is 631 -/
theorem diamonds_G15 : diamonds 15 = 631 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_G15_l3600_360017


namespace NUMINAMATH_CALUDE_coupon1_best_for_given_prices_coupon1_not_best_for_lower_prices_l3600_360046

-- Define the discount functions for each coupon
def coupon1_discount (price : ℝ) : ℝ := 0.12 * price

def coupon2_discount (price : ℝ) : ℝ := 30

def coupon3_discount (price : ℝ) : ℝ := 0.15 * (price - 150)

def coupon4_discount (price : ℝ) : ℝ := 25 + 0.05 * (price - 25)

-- Define a function to check if Coupon 1 gives the best discount
def coupon1_is_best (price : ℝ) : Prop :=
  coupon1_discount price > coupon2_discount price ∧
  coupon1_discount price > coupon3_discount price ∧
  coupon1_discount price > coupon4_discount price

-- Theorem stating that Coupon 1 is best for $300, $350, and $400
theorem coupon1_best_for_given_prices :
  coupon1_is_best 300 ∧ coupon1_is_best 350 ∧ coupon1_is_best 400 :=
by sorry

-- Additional theorem to show Coupon 1 is not best for $200 and $250
theorem coupon1_not_best_for_lower_prices :
  ¬(coupon1_is_best 200) ∧ ¬(coupon1_is_best 250) :=
by sorry

end NUMINAMATH_CALUDE_coupon1_best_for_given_prices_coupon1_not_best_for_lower_prices_l3600_360046


namespace NUMINAMATH_CALUDE_unique_numbers_problem_l3600_360049

theorem unique_numbers_problem (a b : ℕ) : 
  a ≠ b → 
  a > 11 → 
  b > 11 → 
  (∃ (s : ℕ), s = a + b) → 
  (a % 2 = 0 ∨ b % 2 = 0) →
  (∀ (x y : ℕ), x ≠ y → x > 11 → y > 11 → x + y = a + b → 
    (x % 2 = 0 ∨ y % 2 = 0) → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) →
  ((a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12)) :=
by sorry

end NUMINAMATH_CALUDE_unique_numbers_problem_l3600_360049


namespace NUMINAMATH_CALUDE_same_first_last_digit_exists_l3600_360043

-- Define a function to get the first digit of a natural number
def firstDigit (n : ℕ) : ℕ :=
  if n < 10 then n else firstDigit (n / 10)

-- Define a function to get the last digit of a natural number
def lastDigit (n : ℕ) : ℕ :=
  n % 10

-- Theorem statement
theorem same_first_last_digit_exists (n : ℕ) (h : n > 0 ∧ n % 10 ≠ 0) :
  ∃ k : ℕ, k > 0 ∧ firstDigit (n^k) = lastDigit (n^k) :=
sorry

end NUMINAMATH_CALUDE_same_first_last_digit_exists_l3600_360043


namespace NUMINAMATH_CALUDE_product_pqr_equals_864_l3600_360060

theorem product_pqr_equals_864 (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 36)
  (h3 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 540 / (p * q * r) = 1) :
  p * q * r = 864 := by
  sorry

end NUMINAMATH_CALUDE_product_pqr_equals_864_l3600_360060


namespace NUMINAMATH_CALUDE_gmat_test_probabilities_l3600_360061

theorem gmat_test_probabilities
  (p_first : ℝ)
  (p_second : ℝ)
  (p_both : ℝ)
  (h1 : p_first = 0.85)
  (h2 : p_second = 0.80)
  (h3 : p_both = 0.70)
  : 1 - (p_first + p_second - p_both) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_gmat_test_probabilities_l3600_360061


namespace NUMINAMATH_CALUDE_correct_proposition_l3600_360084

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a > b → a * c^2 > b * c^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 - Real.log x₀ = 0

-- Theorem to prove
theorem correct_proposition : ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_correct_proposition_l3600_360084


namespace NUMINAMATH_CALUDE_modular_congruence_unique_solution_l3600_360007

theorem modular_congruence_unique_solution : ∃! m : ℤ, 0 ≤ m ∧ m < 31 ∧ 79453 ≡ m [ZMOD 31] := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_unique_solution_l3600_360007


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_x_plus_y_to_8_l3600_360088

theorem coefficient_x3y5_in_x_plus_y_to_8 :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k) * X^k * Y^(8 - k)) =
  56 * X^3 * Y^5 + (Finset.range 9).sum (fun k => if k ≠ 3 then (Nat.choose 8 k) * X^k * Y^(8 - k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_x_plus_y_to_8_l3600_360088


namespace NUMINAMATH_CALUDE_agrey_caught_more_l3600_360044

def fishing_problem (leo_fish agrey_fish total_fish : ℕ) : Prop :=
  leo_fish + agrey_fish = total_fish ∧ agrey_fish > leo_fish

theorem agrey_caught_more (leo_fish total_fish : ℕ) 
  (h : fishing_problem leo_fish (total_fish - leo_fish) total_fish) 
  (h_leo : leo_fish = 40) 
  (h_total : total_fish = 100) : 
  (total_fish - leo_fish) - leo_fish = 20 := by
  sorry

end NUMINAMATH_CALUDE_agrey_caught_more_l3600_360044


namespace NUMINAMATH_CALUDE_det_dilation_matrix_5_l3600_360024

/-- A dilation matrix with scale factor k -/
def dilationMatrix (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ => k)

/-- Theorem: The determinant of a 3x3 dilation matrix with scale factor 5 is 125 -/
theorem det_dilation_matrix_5 :
  Matrix.det (dilationMatrix 5) = 125 := by
  sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_5_l3600_360024


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_inequality_three_l3600_360050

-- 1. 3x + 1 ≥ -2 if and only if x ≥ -1
theorem inequality_one (x : ℝ) : 3 * x + 1 ≥ -2 ↔ x ≥ -1 := by sorry

-- 2. (y ≥ 1 and -2y ≥ -2) if and only if y = 1
theorem inequality_two (y : ℝ) : (y ≥ 1 ∧ -2 * y ≥ -2) ↔ y = 1 := by sorry

-- 3. y²(x² + 1) - 1 ≤ x² if and only if -1 ≤ y ≤ 1
theorem inequality_three (x y : ℝ) : y^2 * (x^2 + 1) - 1 ≤ x^2 ↔ -1 ≤ y ∧ y ≤ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_inequality_three_l3600_360050


namespace NUMINAMATH_CALUDE_compare_log_and_sqrt_l3600_360021

theorem compare_log_and_sqrt : 2 + Real.log 6 / Real.log 2 > 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_compare_log_and_sqrt_l3600_360021


namespace NUMINAMATH_CALUDE_evaluate_expression_l3600_360002

theorem evaluate_expression : 4 - (-3)^(-1/2 : ℂ) = 4 + (Complex.I * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3600_360002


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3600_360036

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x

-- State the theorem
theorem f_derivative_at_zero : 
  (deriv f) 0 = -3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3600_360036


namespace NUMINAMATH_CALUDE_complementary_event_a_l3600_360094

/-- A batch of products containing genuine and defective items. -/
structure Batch where
  genuine : ℕ
  defective : ℕ
  defective_gt_four : defective > 4

/-- A selection of 4 products from a batch. -/
structure Selection where
  batch : Batch
  items : Finset (Fin 4)
  items_card : items.card = 4

/-- Event A: At least one defective product in the selection. -/
def eventA (s : Selection) : Prop :=
  ∃ i ∈ s.items, i < s.batch.defective

/-- Complementary event of A: No defective products in the selection. -/
def complementEventA (s : Selection) : Prop :=
  ∀ i ∈ s.items, i ≥ s.batch.defective

/-- Theorem stating that the complementary event of A is "no defective products". -/
theorem complementary_event_a (s : Selection) :
  ¬(eventA s) ↔ complementEventA s :=
sorry

end NUMINAMATH_CALUDE_complementary_event_a_l3600_360094


namespace NUMINAMATH_CALUDE_latte_price_calculation_l3600_360008

-- Define the prices and quantities
def total_cost : ℚ := 25
def drip_coffee_price : ℚ := 2.25
def drip_coffee_quantity : ℕ := 2
def espresso_price : ℚ := 3.50
def espresso_quantity : ℕ := 1
def latte_quantity : ℕ := 2
def vanilla_syrup_price : ℚ := 0.50
def vanilla_syrup_quantity : ℕ := 1
def cold_brew_price : ℚ := 2.50
def cold_brew_quantity : ℕ := 2
def cappuccino_price : ℚ := 3.50
def cappuccino_quantity : ℕ := 1

-- Define the theorem
theorem latte_price_calculation :
  ∃ (latte_price : ℚ),
    latte_price * latte_quantity +
    drip_coffee_price * drip_coffee_quantity +
    espresso_price * espresso_quantity +
    vanilla_syrup_price * vanilla_syrup_quantity +
    cold_brew_price * cold_brew_quantity +
    cappuccino_price * cappuccino_quantity = total_cost ∧
    latte_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_latte_price_calculation_l3600_360008


namespace NUMINAMATH_CALUDE_polynomial_no_real_roots_l3600_360091

theorem polynomial_no_real_roots :
  ∀ x : ℝ, 4 * x^8 - 2 * x^7 + x^6 - 3 * x^4 + x^2 - x + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_no_real_roots_l3600_360091


namespace NUMINAMATH_CALUDE_janet_investment_interest_l3600_360019

/-- Calculates the total interest earned from an investment --/
def total_interest (total_investment : ℝ) (high_rate_investment : ℝ) (high_rate : ℝ) (low_rate : ℝ) : ℝ :=
  let low_rate_investment := total_investment - high_rate_investment
  let high_rate_interest := high_rate_investment * high_rate
  let low_rate_interest := low_rate_investment * low_rate
  high_rate_interest + low_rate_interest

/-- Proves that Janet's investment yields $1,390 in interest --/
theorem janet_investment_interest :
  total_interest 31000 12000 0.10 0.01 = 1390 := by
  sorry

end NUMINAMATH_CALUDE_janet_investment_interest_l3600_360019


namespace NUMINAMATH_CALUDE_hexagon_side_count_l3600_360023

-- Define a convex hexagon with two distinct side lengths
structure ConvexHexagon where
  side_length1 : ℕ
  side_length2 : ℕ
  side_count1 : ℕ
  side_count2 : ℕ
  distinct_lengths : side_length1 ≠ side_length2
  total_sides : side_count1 + side_count2 = 6

-- Theorem statement
theorem hexagon_side_count (h : ConvexHexagon) 
  (side_ab : h.side_length1 = 7)
  (side_bc : h.side_length2 = 8)
  (perimeter : h.side_length1 * h.side_count1 + h.side_length2 * h.side_count2 = 46) :
  h.side_count2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_count_l3600_360023


namespace NUMINAMATH_CALUDE_proportion_solution_l3600_360065

theorem proportion_solution (x : ℝ) : (0.6 / x = 5 / 8) → x = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3600_360065


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3600_360086

-- Define the set A
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

-- Define the set B
def B : Set ℝ := {x : ℝ | x^2 < 4}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x : ℝ | ¬ (x ∈ B)}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ complement_B = {x : ℝ | 2 ≤ x ∧ x < 6} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3600_360086


namespace NUMINAMATH_CALUDE_cake_flour_requirement_l3600_360037

theorem cake_flour_requirement (total_flour : ℝ) (cake_flour : ℝ) (cupcake_flour : ℝ)
  (cupcake_requirement : ℝ) (cake_price : ℝ) (cupcake_price : ℝ) (total_earnings : ℝ)
  (h1 : total_flour = 6)
  (h2 : cake_flour = 4)
  (h3 : cupcake_flour = 2)
  (h4 : cupcake_requirement = 1/5)
  (h5 : cake_price = 2.5)
  (h6 : cupcake_price = 1)
  (h7 : total_earnings = 30) :
  cake_flour / (cake_flour / (total_earnings - cupcake_flour / cupcake_requirement * cupcake_price) * cake_price) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cake_flour_requirement_l3600_360037


namespace NUMINAMATH_CALUDE_tan_alpha_plus_20_l3600_360099

theorem tan_alpha_plus_20 (α : ℝ) (h : Real.tan (α + 80 * π / 180) = 4 * Real.sin (420 * π / 180)) :
  Real.tan (α + 20 * π / 180) = Real.sqrt 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_20_l3600_360099


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3600_360011

open Set

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4, 5}
  A ∩ B = {3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3600_360011


namespace NUMINAMATH_CALUDE_invalid_prism_diagonals_l3600_360022

theorem invalid_prism_diagonals : ¬∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a^2 + b^2 = 5^2 ∨ a^2 + b^2 = 12^2 ∨ a^2 + b^2 = 13^2) ∧
  (b^2 + c^2 = 5^2 ∨ b^2 + c^2 = 12^2 ∨ b^2 + c^2 = 13^2) ∧
  (a^2 + c^2 = 5^2 ∨ a^2 + c^2 = 12^2 ∨ a^2 + c^2 = 13^2) ∧
  (a^2 + b^2 + c^2 = 14^2) :=
by sorry

end NUMINAMATH_CALUDE_invalid_prism_diagonals_l3600_360022


namespace NUMINAMATH_CALUDE_factorization_equality_l3600_360089

theorem factorization_equality (a b : ℝ) : 3 * a^2 * b - 12 * b = 3 * b * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3600_360089


namespace NUMINAMATH_CALUDE_equation_solution_l3600_360039

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 3) * x = 14 ∧ x = 147 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3600_360039


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3600_360055

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ x ≥ 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 4}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (Bᶜ) = {x | x < -1 ∨ x ≥ 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3600_360055


namespace NUMINAMATH_CALUDE_classroom_gpa_l3600_360058

/-- Given a classroom where one-third of the students have a GPA of 54 and the remaining two-thirds have a GPA of 45, the GPA of the whole class is 48. -/
theorem classroom_gpa : 
  ∀ (n : ℕ) (total_gpa : ℝ),
  n > 0 →
  total_gpa = (n / 3 : ℝ) * 54 + (2 * n / 3 : ℝ) * 45 →
  total_gpa / n = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_gpa_l3600_360058


namespace NUMINAMATH_CALUDE_square_even_implies_even_l3600_360067

theorem square_even_implies_even (a : ℤ) (h : Even (a^2)) : Even a := by
  sorry

end NUMINAMATH_CALUDE_square_even_implies_even_l3600_360067


namespace NUMINAMATH_CALUDE_trip_time_at_new_speed_l3600_360096

/-- Calculate the time taken for a trip at a new speed, given the original time and speeds -/
theorem trip_time_at_new_speed (original_time original_speed new_speed : ℝ) 
  (h1 : original_time > 0)
  (h2 : original_speed > 0)
  (h3 : new_speed > 0) :
  let new_time := original_time * (original_speed / new_speed)
  ∃ ε > 0, |new_time - 3.94| < ε ∧ original_time = 4.5 ∧ original_speed = 70 ∧ new_speed = 80 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_at_new_speed_l3600_360096


namespace NUMINAMATH_CALUDE_beta_highest_success_ratio_l3600_360079

/-- Represents a participant's scores in a two-day challenge -/
structure ParticipantScores where
  day1_score : ℕ
  day1_attempted : ℕ
  day2_score : ℕ
  day2_attempted : ℕ

def ParticipantScores.total_score (p : ParticipantScores) : ℕ :=
  p.day1_score + p.day2_score

def ParticipantScores.total_attempted (p : ParticipantScores) : ℕ :=
  p.day1_attempted + p.day2_attempted

def ParticipantScores.success_ratio (p : ParticipantScores) : ℚ :=
  (p.total_score : ℚ) / p.total_attempted

def ParticipantScores.daily_success_ratio (p : ParticipantScores) (day : Fin 2) : ℚ :=
  match day with
  | 0 => (p.day1_score : ℚ) / p.day1_attempted
  | 1 => (p.day2_score : ℚ) / p.day2_attempted

theorem beta_highest_success_ratio
  (alpha : ParticipantScores)
  (beta : ParticipantScores)
  (h_total_points : alpha.total_attempted = 500)
  (h_alpha_scores : alpha.day1_score = 200 ∧ alpha.day1_attempted = 300 ∧
                    alpha.day2_score = 100 ∧ alpha.day2_attempted = 200)
  (h_beta_fewer : beta.day1_attempted < alpha.day1_attempted ∧
                  beta.day2_attempted < alpha.day2_attempted)
  (h_beta_nonzero : beta.day1_score > 0 ∧ beta.day2_score > 0)
  (h_beta_lower_ratio : ∀ day, beta.daily_success_ratio day < alpha.daily_success_ratio day)
  (h_alpha_ratio : alpha.success_ratio = 3/5)
  (h_beta_day1 : beta.day1_attempted = 220) :
  beta.success_ratio ≤ 248/500 :=
sorry

end NUMINAMATH_CALUDE_beta_highest_success_ratio_l3600_360079


namespace NUMINAMATH_CALUDE_max_value_F_H_surjective_implies_s_value_l3600_360006

noncomputable section

def f (x : ℝ) : ℝ := (Real.log x) / x

def F (x : ℝ) : ℝ := x^2 - x * f x

def H (s : ℝ) (x : ℝ) : ℝ :=
  if x ≥ s then x / (2 * Real.exp 1) else f x

theorem max_value_F :
  ∃ (x : ℝ), x ∈ Set.Icc (1/2) 2 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (1/2) 2 → F x ≥ F y ∧
  F x = 4 - Real.log 2 := by sorry

theorem H_surjective_implies_s_value (s : ℝ) :
  (∀ (k : ℝ), ∃ (x : ℝ), H s x = k) →
  s = Real.sqrt (Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_max_value_F_H_surjective_implies_s_value_l3600_360006


namespace NUMINAMATH_CALUDE_min_probability_alex_dylan_same_team_l3600_360014

/-- The probability that Alex and Dylan are on the same team given that Alex picks one of the cards a or a+7, and Dylan picks the other. -/
def p (a : ℕ) : ℚ :=
  (Nat.choose (32 - a) 2 + Nat.choose (a - 1) 2) / 703

/-- The statement to be proved -/
theorem min_probability_alex_dylan_same_team :
  (∃ a : ℕ, a ≤ 40 ∧ a + 7 ≤ 40 ∧ p a ≥ 1/2) ∧
  (∀ a : ℕ, a ≤ 40 ∧ a + 7 ≤ 40 ∧ p a ≥ 1/2 → p a ≥ 497/703) ∧
  (∃ a : ℕ, a ≤ 40 ∧ a + 7 ≤ 40 ∧ p a = 497/703) :=
sorry

end NUMINAMATH_CALUDE_min_probability_alex_dylan_same_team_l3600_360014


namespace NUMINAMATH_CALUDE_f_even_implies_a_zero_f_not_odd_l3600_360074

/-- Definition of the function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + |x - a| + 1

/-- Theorem 1: If f is even, then a = 0 -/
theorem f_even_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by sorry

/-- Theorem 2: f is not odd for any real a -/
theorem f_not_odd (a : ℝ) :
  ¬(∀ x : ℝ, f a (-x) = -(f a x)) := by sorry

end NUMINAMATH_CALUDE_f_even_implies_a_zero_f_not_odd_l3600_360074


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3600_360001

/-- A geometric sequence with first term x, second term 3x+3, and third term 6x+6 has fourth term -24 -/
theorem geometric_sequence_fourth_term :
  ∀ x : ℝ,
  let a₁ : ℝ := x
  let a₂ : ℝ := 3*x + 3
  let a₃ : ℝ := 6*x + 6
  let r : ℝ := a₂ / a₁
  (a₂ = r * a₁) ∧ (a₃ = r * a₂) →
  r * a₃ = -24 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3600_360001


namespace NUMINAMATH_CALUDE_percentage_problem_l3600_360028

theorem percentage_problem (y : ℝ) (h1 : y > 0) (h2 : y * (y / 100) = 9) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3600_360028


namespace NUMINAMATH_CALUDE_average_weight_increase_l3600_360018

theorem average_weight_increase (initial_count : ℕ) (replaced_weight new_weight : ℝ) : 
  initial_count = 8 →
  replaced_weight = 65 →
  new_weight = 93 →
  (new_weight - replaced_weight) / initial_count = 3.5 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3600_360018


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3600_360068

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = 1) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 41/20 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3600_360068


namespace NUMINAMATH_CALUDE_twentieth_fisherman_catch_l3600_360048

theorem twentieth_fisherman_catch (total_fishermen : Nat) (total_fish : Nat) 
  (each_fish : Nat) (n : Nat) (h1 : total_fishermen = 20) 
  (h2 : total_fish = 10000) (h3 : each_fish = 400) (h4 : n = 19) : 
  total_fish - n * each_fish = 2400 := by
  sorry

#check twentieth_fisherman_catch

end NUMINAMATH_CALUDE_twentieth_fisherman_catch_l3600_360048


namespace NUMINAMATH_CALUDE_average_weight_of_children_l3600_360070

theorem average_weight_of_children (num_boys num_girls : ℕ) 
  (avg_weight_boys avg_weight_girls : ℚ) :
  num_boys = 8 →
  num_girls = 5 →
  avg_weight_boys = 160 →
  avg_weight_girls = 130 →
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 148 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_of_children_l3600_360070


namespace NUMINAMATH_CALUDE_rachel_apple_picking_l3600_360097

/-- Rachel's apple picking problem -/
theorem rachel_apple_picking (total_trees : ℕ) (initial_apples : ℕ) (remaining_apples : ℕ) 
  (h1 : total_trees = 52)
  (h2 : initial_apples = 9)
  (h3 : remaining_apples = 7) :
  initial_apples - remaining_apples = 2 := by
  sorry

#check rachel_apple_picking

end NUMINAMATH_CALUDE_rachel_apple_picking_l3600_360097


namespace NUMINAMATH_CALUDE_function_properties_l3600_360057

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Define the theorem
theorem function_properties (a m n : ℝ) 
  (h_m_pos : m > 0) (h_n_pos : n > 0) (h_m_neq_n : m ≠ n)
  (h_fm : f a m = 3) (h_fn : f a n = 3) :
  0 < a ∧ a < Real.exp 2 ∧ a^2 < m * n ∧ m * n < a * Real.exp 2 := by
  sorry

end

end NUMINAMATH_CALUDE_function_properties_l3600_360057


namespace NUMINAMATH_CALUDE_polynomial_equality_l3600_360069

theorem polynomial_equality (x : ℝ) (h : 3 * x^3 - x = 1) :
  9 * x^4 + 12 * x^3 - 3 * x^2 - 7 * x + 2001 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3600_360069


namespace NUMINAMATH_CALUDE_cherry_tomatoes_count_l3600_360053

theorem cherry_tomatoes_count (tomatoes_per_jar : ℕ) (jars_needed : ℕ) 
  (h1 : tomatoes_per_jar = 8) 
  (h2 : jars_needed = 7) : 
  tomatoes_per_jar * jars_needed = 56 := by
  sorry

end NUMINAMATH_CALUDE_cherry_tomatoes_count_l3600_360053


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l3600_360010

def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_standard_equation 
  (major_axis_length : ℝ) 
  (eccentricity : ℝ) :
  major_axis_length = 12 →
  eccentricity = 2/3 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    ∀ x y : ℝ, ellipse_equation a b x y ↔ 
      x^2 / 36 + y^2 / 20 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l3600_360010


namespace NUMINAMATH_CALUDE_dad_second_half_speed_l3600_360003

-- Define the given conditions
def total_time : Real := 0.5  -- 30 minutes in hours
def first_half_speed : Real := 28
def jake_bike_speed : Real := 11
def jake_bike_time : Real := 2

-- Define the theorem
theorem dad_second_half_speed :
  let total_distance := jake_bike_speed * jake_bike_time
  let first_half_distance := first_half_speed * (total_time / 2)
  let second_half_distance := total_distance - first_half_distance
  let second_half_speed := second_half_distance / (total_time / 2)
  second_half_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_dad_second_half_speed_l3600_360003


namespace NUMINAMATH_CALUDE_smallest_max_sum_l3600_360073

theorem smallest_max_sum (p q r s t : ℕ+) (h_sum : p + q + r + s + t = 4020) :
  let N := max (p + q) (max (q + r) (max (r + s) (s + t)))
  ∀ m : ℕ, (∀ a b c d e : ℕ+, a + b + c + d + e = 4020 →
    m ≥ max (a + b) (max (b + c) (max (c + d) (d + e)))) →
  m ≥ 1342 :=
by sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l3600_360073


namespace NUMINAMATH_CALUDE_larger_triangle_equilateral_iff_l3600_360075

/-- Two identical right-angled triangles with angles α and β form a larger triangle when placed together with identical legs adjacent. -/
structure TrianglePair where
  α : Real
  β : Real
  right_angled : α + β = 90
  non_negative : 0 ≤ α ∧ 0 ≤ β

/-- The larger triangle formed by combining two identical right-angled triangles. -/
structure LargerTriangle where
  pair : TrianglePair
  side_a : Real
  side_b : Real
  side_c : Real
  angle_A : Real
  angle_B : Real
  angle_C : Real

/-- The larger triangle is equilateral if and only if the original right-angled triangles have α = 60° and β = 30°. -/
theorem larger_triangle_equilateral_iff (t : LargerTriangle) :
  (t.side_a = t.side_b ∧ t.side_b = t.side_c) ↔ (t.pair.α = 60 ∧ t.pair.β = 30) :=
sorry

end NUMINAMATH_CALUDE_larger_triangle_equilateral_iff_l3600_360075


namespace NUMINAMATH_CALUDE_angle_QNR_is_165_l3600_360020

/-- An isosceles triangle PQR with a point N inside -/
structure IsoscelesTriangleWithPoint where
  /-- The measure of angle PRQ in degrees -/
  angle_PRQ : ℝ
  /-- The measure of angle PNR in degrees -/
  angle_PNR : ℝ
  /-- The measure of angle PRN in degrees -/
  angle_PRN : ℝ
  /-- PR = QR (isosceles condition) -/
  isosceles : True
  /-- N is in the interior of the triangle -/
  N_interior : True
  /-- Angle PRQ is 108 degrees -/
  h_PRQ : angle_PRQ = 108
  /-- Angle PNR is 9 degrees -/
  h_PNR : angle_PNR = 9
  /-- Angle PRN is 21 degrees -/
  h_PRN : angle_PRN = 21

/-- Theorem: In the given isosceles triangle with point N, angle QNR is 165 degrees -/
theorem angle_QNR_is_165 (t : IsoscelesTriangleWithPoint) : ∃ angle_QNR : ℝ, angle_QNR = 165 := by
  sorry

end NUMINAMATH_CALUDE_angle_QNR_is_165_l3600_360020


namespace NUMINAMATH_CALUDE_charity_fundraising_contribution_l3600_360038

theorem charity_fundraising_contribution 
  (total_goal : ℝ) 
  (collected : ℝ) 
  (num_people : ℕ) 
  (h1 : total_goal = 2400)
  (h2 : collected = 300)
  (h3 : num_people = 8) :
  (total_goal - collected) / num_people = 262.5 := by
sorry

end NUMINAMATH_CALUDE_charity_fundraising_contribution_l3600_360038


namespace NUMINAMATH_CALUDE_periodic_function_value_l3600_360035

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function_value (f : ℝ → ℝ) :
  is_periodic f 4 →
  (∀ x ∈ Set.Icc (-2) 2, f x = x) →
  f 7.6 = -0.4 := by
sorry

end NUMINAMATH_CALUDE_periodic_function_value_l3600_360035


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3600_360030

def sandwich_cost : ℝ := 4
def soda_cost : ℝ := 3
def tax_rate : ℝ := 0.1
def num_sandwiches : ℕ := 4
def num_sodas : ℕ := 6

theorem total_cost_calculation :
  let subtotal := sandwich_cost * num_sandwiches + soda_cost * num_sodas
  let tax := subtotal * tax_rate
  let total_cost := subtotal + tax
  total_cost = 37.4 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3600_360030


namespace NUMINAMATH_CALUDE_cos_sin_sum_implies_cos_double_sum_zero_l3600_360052

theorem cos_sin_sum_implies_cos_double_sum_zero 
  (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z = 1)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 1) :
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_implies_cos_double_sum_zero_l3600_360052


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3600_360013

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  swimmer : ℝ
  stream : ℝ

/-- Calculates the effective speed of the swimmer given the direction. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 7 km/h. -/
theorem swimmer_speed_in_still_water
  (s : SwimmerSpeeds)
  (h_downstream : effectiveSpeed s true * 4 = 32)
  (h_upstream : effectiveSpeed s false * 4 = 24) :
  s.swimmer = 7 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3600_360013


namespace NUMINAMATH_CALUDE_exp_ge_e_l3600_360081

theorem exp_ge_e (x : ℝ) (h : x > 0) : Real.exp x ≥ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_exp_ge_e_l3600_360081


namespace NUMINAMATH_CALUDE_flag_arrangement_count_remainder_mod_1000_l3600_360025

/-- The number of red flags -/
def red_flags : ℕ := 11

/-- The number of white flags -/
def white_flags : ℕ := 6

/-- The total number of flags -/
def total_flags : ℕ := red_flags + white_flags

/-- The number of distinguishable flagpoles -/
def flagpoles : ℕ := 2

/-- Represents a valid flag arrangement -/
structure FlagArrangement where
  arrangement : List Bool
  red_count : ℕ
  white_count : ℕ
  no_adjacent_white : Bool
  at_least_one_per_pole : Bool

/-- The number of valid distinguishable arrangements -/
def valid_arrangements : ℕ := 10164

theorem flag_arrangement_count :
  (∃ (arrangements : List FlagArrangement),
    (∀ a ∈ arrangements,
      a.red_count = red_flags ∧
      a.white_count = white_flags ∧
      a.no_adjacent_white = true ∧
      a.at_least_one_per_pole = true) ∧
    arrangements.length = valid_arrangements) :=
sorry

theorem remainder_mod_1000 :
  valid_arrangements % 1000 = 164 :=
sorry

end NUMINAMATH_CALUDE_flag_arrangement_count_remainder_mod_1000_l3600_360025


namespace NUMINAMATH_CALUDE_grid_equal_sums_l3600_360092

/-- Given a, b, c, prove that there exist x, y, z, t, u, v such that all rows, columns, and diagonals in a 3x3 grid sum to the same value -/
theorem grid_equal_sums (a b c : ℚ) : ∃ (x y z t u v : ℚ),
  (x + a + b = x + y + c) ∧
  (y + z + t = b + z + c) ∧
  (u + t + v = a + t + c) ∧
  (x + y + c = y + z + t) ∧
  (x + a + b = a + z + v) ∧
  (x + y + c = u + t + v) ∧
  (x + a + b = b + z + c) :=
by sorry

end NUMINAMATH_CALUDE_grid_equal_sums_l3600_360092


namespace NUMINAMATH_CALUDE_three_numbers_problem_l3600_360062

theorem three_numbers_problem (x y z : ℝ) : 
  x = 0.8 * y ∧ 
  y / z = 0.5 / (9/20) ∧ 
  x + z = y + 70 →
  x = 80 ∧ y = 100 ∧ z = 90 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l3600_360062


namespace NUMINAMATH_CALUDE_no_real_roots_l3600_360027

theorem no_real_roots (a b c : ℕ) (ha : a < 1000000) (hb : b < 1000000) (hc : c < 1000000) :
  ¬∃ x : ℝ, (a * x^2)^(1/21) + (b * x)^(1/21) + c^(1/21) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3600_360027


namespace NUMINAMATH_CALUDE_prime_odd_sum_l3600_360015

theorem prime_odd_sum (a b : ℕ) : 
  Nat.Prime a → 
  Odd b → 
  a^2 + b = 2001 → 
  a + b = 1999 := by sorry

end NUMINAMATH_CALUDE_prime_odd_sum_l3600_360015


namespace NUMINAMATH_CALUDE_special_number_value_l3600_360076

/-- Represents a positive integer with specific properties in different bases -/
def SpecialNumber (n : ℕ+) : Prop :=
  ∃ (X Y : ℕ),
    X < 8 ∧ Y < 9 ∧
    n = 8 * X + Y ∧
    n = 9 * Y + X

/-- The unique value of the special number in base 10 -/
theorem special_number_value :
  ∀ n : ℕ+, SpecialNumber n → n = 71 := by
  sorry

end NUMINAMATH_CALUDE_special_number_value_l3600_360076
