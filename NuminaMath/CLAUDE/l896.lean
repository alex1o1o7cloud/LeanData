import Mathlib

namespace y_intercept_after_transformation_l896_89607

/-- A linear function f(x) = -2x + 3 -/
def f (x : ℝ) : ℝ := -2 * x + 3

/-- The transformed function g(x) after moving f(x) up by 2 units -/
def g (x : ℝ) : ℝ := f x + 2

/-- Theorem: The y-intercept of g(x) is at the point (0, 5) -/
theorem y_intercept_after_transformation :
  g 0 = 5 := by sorry

end y_intercept_after_transformation_l896_89607


namespace gcd_lcm_888_1147_l896_89678

theorem gcd_lcm_888_1147 : 
  (Nat.gcd 888 1147 = 37) ∧ (Nat.lcm 888 1147 = 27528) := by sorry

end gcd_lcm_888_1147_l896_89678


namespace paint_needed_paint_problem_l896_89640

theorem paint_needed (initial_paint : ℚ) (day1_fraction : ℚ) (day2_fraction : ℚ) (additional_needed : ℚ) : ℚ :=
  let remaining_after_day1 := initial_paint - day1_fraction * initial_paint
  let remaining_after_day2 := remaining_after_day1 - day2_fraction * remaining_after_day1
  let total_needed := remaining_after_day2 + additional_needed
  total_needed - remaining_after_day2

theorem paint_problem : paint_needed 2 (1/4) (1/2) (1/2) = 1/2 := by
  sorry

end paint_needed_paint_problem_l896_89640


namespace girls_in_school_l896_89633

theorem girls_in_school (total_students sample_size girls_boys_diff : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : girls_boys_diff = 20) : 
  ∃ (girls : ℕ), girls = 860 ∧ 
  girls + (total_students - girls) = total_students ∧
  (girls : ℚ) / total_students * sample_size = 
    (total_students - girls : ℚ) / total_students * sample_size - girls_boys_diff := by
  sorry

end girls_in_school_l896_89633


namespace hugo_first_roll_four_given_win_l896_89694

-- Define the number of players
def num_players : ℕ := 5

-- Define the number of sides on the die
def die_sides : ℕ := 6

-- Define Hugo's winning probability
def hugo_win_prob : ℚ := 1 / num_players

-- Define the probability of rolling a 4
def roll_four_prob : ℚ := 1 / die_sides

-- Define the probability of Hugo winning given he rolled a 4
def hugo_win_given_four_prob : ℚ := 256 / 1296

-- Theorem to prove
theorem hugo_first_roll_four_given_win (
  num_players : ℕ) (die_sides : ℕ) (hugo_win_prob : ℚ) 
  (roll_four_prob : ℚ) (hugo_win_given_four_prob : ℚ) :
  num_players = 5 ∧ die_sides = 6 ∧ 
  hugo_win_prob = 1 / num_players ∧
  roll_four_prob = 1 / die_sides ∧
  hugo_win_given_four_prob = 256 / 1296 →
  (roll_four_prob * hugo_win_given_four_prob) / hugo_win_prob = 40 / 243 :=
by sorry

end hugo_first_roll_four_given_win_l896_89694


namespace oil_bottles_total_volume_l896_89612

theorem oil_bottles_total_volume (total_bottles : ℕ) (small_bottles : ℕ) 
  (small_volume : ℚ) (large_volume : ℚ) :
  total_bottles = 35 →
  small_bottles = 17 →
  small_volume = 250 / 1000 →
  large_volume = 300 / 1000 →
  (small_bottles * small_volume + (total_bottles - small_bottles) * large_volume) = 9.65 := by
sorry

end oil_bottles_total_volume_l896_89612


namespace work_completion_time_l896_89650

/-- 
Given that Paul completes a piece of work in 80 days and Rose completes the same work in 120 days,
prove that they will complete the work together in 48 days.
-/
theorem work_completion_time 
  (paul_time : ℕ) 
  (rose_time : ℕ) 
  (h_paul : paul_time = 80) 
  (h_rose : rose_time = 120) : 
  (paul_time * rose_time) / (paul_time + rose_time) = 48 := by
  sorry

end work_completion_time_l896_89650


namespace smallest_n_trailing_zeros_l896_89645

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- The smallest integer n ≥ 48 for which the number of trailing zeros in n! is exactly n - 48 -/
theorem smallest_n_trailing_zeros : ∀ n : ℕ, n ≥ 48 → (trailingZeros n = n - 48 → n ≥ 62) ∧ trailingZeros 62 = 62 - 48 := by
  sorry

end smallest_n_trailing_zeros_l896_89645


namespace sequence_termination_l896_89642

def b : ℕ → ℚ
  | 0 => 41
  | 1 => 68
  | (k+2) => b k - 5 / b (k+1)

theorem sequence_termination :
  ∃ n : ℕ, n > 0 ∧ b n = 0 ∧ ∀ k < n, b k ≠ 0 ∧ b (k+1) = b (k-1) - 5 / b k :=
by
  use 559
  sorry

#eval b 559

end sequence_termination_l896_89642


namespace fuel_mixture_problem_l896_89638

theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 212 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (fuel_a : ℝ), 
    fuel_a = 98 ∧
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end fuel_mixture_problem_l896_89638


namespace lateral_surface_area_is_4S_l896_89616

/-- A regular quadrilateral pyramid with specific properties -/
structure RegularQuadPyramid where
  -- The dihedral angle at the lateral edge
  dihedral_angle : ℝ
  -- The area of the diagonal section
  diagonal_section_area : ℝ
  -- Condition that the dihedral angle is 120°
  angle_is_120 : dihedral_angle = 120 * π / 180

/-- The lateral surface area of a regular quadrilateral pyramid -/
def lateral_surface_area (p : RegularQuadPyramid) : ℝ := 4 * p.diagonal_section_area

/-- Theorem: The lateral surface area of a regular quadrilateral pyramid with a 120° dihedral angle
    at the lateral edge is 4 times the area of its diagonal section -/
theorem lateral_surface_area_is_4S (p : RegularQuadPyramid) :
  lateral_surface_area p = 4 * p.diagonal_section_area := by
  sorry

end lateral_surface_area_is_4S_l896_89616


namespace probability_of_three_hits_is_one_fifth_l896_89643

/-- A set of random numbers -/
structure RandomSet :=
  (numbers : List Nat)

/-- Predicate to check if a number is a hit (1 to 6) -/
def isHit (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 6

/-- Count the number of hits in a set -/
def countHits (s : RandomSet) : Nat :=
  s.numbers.filter isHit |>.length

/-- The experiment data -/
def experimentData : List RandomSet := sorry

/-- The number of sets with exactly three hits -/
def setsWithThreeHits : Nat :=
  experimentData.filter (fun s => countHits s = 3) |>.length

/-- Total number of sets in the experiment -/
def totalSets : Nat := 20

theorem probability_of_three_hits_is_one_fifth :
  (setsWithThreeHits : ℚ) / totalSets = 1 / 5 := by sorry

end probability_of_three_hits_is_one_fifth_l896_89643


namespace discounted_milk_price_is_correct_l896_89695

/-- The discounted price of a gallon of whole milk -/
def discounted_milk_price : ℝ := 2

/-- The normal price of a gallon of whole milk -/
def normal_milk_price : ℝ := 3

/-- The discount on a box of cereal -/
def cereal_discount : ℝ := 1

/-- The total savings when buying 3 gallons of milk and 5 boxes of cereal -/
def total_savings : ℝ := 8

/-- The number of gallons of milk bought -/
def milk_quantity : ℕ := 3

/-- The number of boxes of cereal bought -/
def cereal_quantity : ℕ := 5

theorem discounted_milk_price_is_correct :
  (milk_quantity : ℝ) * (normal_milk_price - discounted_milk_price) + 
  (cereal_quantity : ℝ) * cereal_discount = total_savings := by
  sorry

end discounted_milk_price_is_correct_l896_89695


namespace largest_unexpressible_l896_89662

/-- The set An for a given n -/
def An (n : ℕ) : Set ℕ :=
  {x | ∃ k : ℕ, k < n ∧ x = 2^n - 2^k}

/-- The property of being expressible as a sum of elements from An -/
def isExpressible (n : ℕ) (m : ℕ) : Prop :=
  ∃ (s : Multiset ℕ), (∀ x ∈ s, x ∈ An n) ∧ (s.sum = m)

/-- The main theorem -/
theorem largest_unexpressible (n : ℕ) (h : n ≥ 2) :
  ∀ m : ℕ, m > (n - 2) * 2^n + 1 → isExpressible n m :=
sorry

end largest_unexpressible_l896_89662


namespace f_range_l896_89614

noncomputable def f (x : ℝ) : ℝ := 1 - 2 / (Real.log x + 1)

theorem f_range (m n : ℝ) (hm : m > Real.exp 1) (hn : n > Real.exp 1)
  (h : f m = 2 * Real.log (Real.sqrt (Real.exp 1)) - f n) :
  5/7 ≤ f (m * n) ∧ f (m * n) < 1 := by
  sorry

end f_range_l896_89614


namespace total_medals_1996_l896_89622

def gold_medals : ℕ := 16
def silver_medals : ℕ := 22
def bronze_medals : ℕ := 12

theorem total_medals_1996 : gold_medals + silver_medals + bronze_medals = 50 := by
  sorry

end total_medals_1996_l896_89622


namespace max_n_sin_cos_inequality_l896_89689

theorem max_n_sin_cos_inequality : 
  (∃ (n : ℕ), n > 0 ∧ ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / n) ∧ 
  (∀ (m : ℕ), m > 8 → ∃ (x : ℝ), (Real.sin x)^m + (Real.cos x)^m < 1 / m) :=
by sorry

end max_n_sin_cos_inequality_l896_89689


namespace smallest_number_with_conditions_l896_89671

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem smallest_number_with_conditions : ℕ := by
  let n : ℕ := 2979942
  have h1 : tens_digit n = 4 := by sorry
  have h2 : units_digit n = 2 := by sorry
  have h3 : sum_of_digits n = 42 := by sorry
  have h4 : n % 42 = 0 := by sorry
  have h5 : ∀ m : ℕ, m < n →
    ¬(tens_digit m = 4 ∧ units_digit m = 2 ∧ sum_of_digits m = 42 ∧ m % 42 = 0) := by sorry
  exact n

end smallest_number_with_conditions_l896_89671


namespace prob_both_heads_is_one_fourth_l896_89605

/-- A coin of uniform density -/
structure Coin :=
  (side : Bool)

/-- The sample space of tossing two coins -/
def TwoCoins := Coin × Coin

/-- The event where both coins land heads up -/
def BothHeads (outcome : TwoCoins) : Prop :=
  outcome.1.side ∧ outcome.2.side

/-- The probability measure on the sample space -/
axiom prob : Set TwoCoins → ℝ

/-- The probability measure satisfies basic properties -/
axiom prob_nonneg : ∀ A : Set TwoCoins, 0 ≤ prob A
axiom prob_le_one : ∀ A : Set TwoCoins, prob A ≤ 1
axiom prob_additive : ∀ A B : Set TwoCoins, A ∩ B = ∅ → prob (A ∪ B) = prob A + prob B

/-- The probability of each outcome is equal due to uniform density -/
axiom prob_uniform : ∀ x y : TwoCoins, prob {x} = prob {y}

theorem prob_both_heads_is_one_fourth :
  prob {x : TwoCoins | BothHeads x} = 1/4 := by
  sorry

#check prob_both_heads_is_one_fourth

end prob_both_heads_is_one_fourth_l896_89605


namespace expression_range_l896_89661

theorem expression_range (x y : ℝ) (h : x^2 + (y - 2)^2 ≤ 1) :
  1 ≤ (x + Real.sqrt 3 * y) / Real.sqrt (x^2 + y^2) ∧
  (x + Real.sqrt 3 * y) / Real.sqrt (x^2 + y^2) ≤ 2 :=
by sorry

end expression_range_l896_89661


namespace max_sum_with_digit_constraints_l896_89699

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is within a specific digit range -/
def is_n_digit (n : ℕ) (lower : ℕ) (upper : ℕ) : Prop :=
  lower ≤ n ∧ n ≤ upper

theorem max_sum_with_digit_constraints :
  ∃ (a b c : ℕ),
    is_n_digit a 10 99 ∧
    is_n_digit b 100 999 ∧
    is_n_digit c 1000 9999 ∧
    sum_of_digits (a + b) = 2 ∧
    sum_of_digits (b + c) = 2 ∧
    ∀ (x y z : ℕ),
      is_n_digit x 10 99 →
      is_n_digit y 100 999 →
      is_n_digit z 1000 9999 →
      sum_of_digits (x + y) = 2 →
      sum_of_digits (y + z) = 2 →
      x + y + z ≤ a + b + c ∧
      a + b + c = 10199 :=
sorry

end max_sum_with_digit_constraints_l896_89699


namespace percentage_of_percentage_l896_89679

theorem percentage_of_percentage (total : ℝ) (percentage1 : ℝ) (amount : ℝ) (percentage2 : ℝ) :
  total = 500 →
  percentage1 = 50 →
  amount = 25 →
  percentage2 = 10 →
  (amount / (percentage1 / 100 * total)) * 100 = percentage2 := by
  sorry

end percentage_of_percentage_l896_89679


namespace car_distance_theorem_l896_89667

/-- Calculates the total distance traveled by a car with increasing speed over a given number of hours -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  (hours * (2 * initialSpeed + (hours - 1) * speedIncrease)) / 2

/-- Theorem stating that a car with given initial speed and speed increase travels a specific distance in 12 hours -/
theorem car_distance_theorem (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) :
  initialSpeed = 40 ∧ speedIncrease = 2 ∧ hours = 12 →
  totalDistance initialSpeed speedIncrease hours = 606 := by
  sorry

end car_distance_theorem_l896_89667


namespace family_gathering_handshakes_l896_89666

theorem family_gathering_handshakes :
  let num_twin_sets : ℕ := 10
  let num_quadruplet_sets : ℕ := 5
  let num_twins : ℕ := num_twin_sets * 2
  let num_quadruplets : ℕ := num_quadruplet_sets * 4
  let twin_handshakes : ℕ := num_twins * (num_twins - 2)
  let quadruplet_handshakes : ℕ := num_quadruplets * (num_quadruplets - 4)
  let twin_to_quadruplet : ℕ := num_twins * (2 * num_quadruplets / 3)
  let quadruplet_to_twin : ℕ := num_quadruplets * (3 * num_twins / 4)
  let total_handshakes : ℕ := (twin_handshakes + quadruplet_handshakes + twin_to_quadruplet + quadruplet_to_twin) / 2
  total_handshakes = 620 :=
by sorry

end family_gathering_handshakes_l896_89666


namespace sallys_purchase_l896_89610

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents50 : ℕ
  dollars5 : ℕ
  dollars10 : ℕ

/-- The problem statement -/
theorem sallys_purchase (counts : ItemCounts) : 
  counts.cents50 + counts.dollars5 + counts.dollars10 = 30 →
  50 * counts.cents50 + 500 * counts.dollars5 + 1000 * counts.dollars10 = 10000 →
  counts.cents50 = 20 := by
  sorry


end sallys_purchase_l896_89610


namespace fruit_group_sizes_l896_89681

theorem fruit_group_sizes (total_bananas total_oranges total_apples : ℕ)
                          (banana_groups orange_groups apple_groups : ℕ)
                          (h1 : total_bananas = 142)
                          (h2 : total_oranges = 356)
                          (h3 : total_apples = 245)
                          (h4 : banana_groups = 47)
                          (h5 : orange_groups = 178)
                          (h6 : apple_groups = 35) :
  ∃ (B O A : ℕ),
    banana_groups * B = total_bananas ∧
    orange_groups * O = total_oranges ∧
    apple_groups * A = total_apples ∧
    B = 3 ∧ O = 2 ∧ A = 7 := by
  sorry

end fruit_group_sizes_l896_89681


namespace boys_playing_both_sports_l896_89628

theorem boys_playing_both_sports (total : ℕ) (basketball : ℕ) (football : ℕ) (neither : ℕ) :
  total = 22 →
  basketball = 13 →
  football = 15 →
  neither = 3 →
  ∃ (both : ℕ), both = 9 ∧ total = basketball + football - both + neither :=
by sorry

end boys_playing_both_sports_l896_89628


namespace equal_after_adjustments_l896_89621

/-- The number of adjustments needed to equalize the number of boys and girls -/
def num_adjustments : ℕ := 8

/-- The initial number of boys -/
def initial_boys : ℕ := 40

/-- The initial number of girls -/
def initial_girls : ℕ := 0

/-- The number of boys reduced in each adjustment -/
def boys_reduction : ℕ := 3

/-- The number of girls increased in each adjustment -/
def girls_increase : ℕ := 2

/-- Calculates the number of boys after a given number of adjustments -/
def boys_after (n : ℕ) : ℤ :=
  initial_boys - n * boys_reduction

/-- Calculates the number of girls after a given number of adjustments -/
def girls_after (n : ℕ) : ℤ :=
  initial_girls + n * girls_increase

theorem equal_after_adjustments :
  boys_after num_adjustments = girls_after num_adjustments := by
  sorry

end equal_after_adjustments_l896_89621


namespace cubic_function_symmetry_l896_89684

/-- Given a cubic function f(x) = ax³ + bx + 4 where a and b are non-zero real numbers,
    if f(5) = 10, then f(-5) = -2 -/
theorem cubic_function_symmetry (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f := λ x : ℝ => a * x^3 + b * x + 4
  f 5 = 10 → f (-5) = -2 := by
  sorry

end cubic_function_symmetry_l896_89684


namespace system_solution_l896_89673

theorem system_solution (x y : ℝ) (h1 : 2 * x + y = 5) (h2 : x + 2 * y = 6) : x - y = -1 := by
  sorry

end system_solution_l896_89673


namespace polynomial_division_remainder_l896_89698

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^3 - 4 * X^2 - 23 * X + 60 = (X - 3) * q + 36 := by
  sorry

end polynomial_division_remainder_l896_89698


namespace correlation_significance_l896_89611

/-- The critical value for a 5% significance level -/
def r_0_05 : ℝ := sorry

/-- The observed correlation coefficient -/
def r : ℝ := sorry

/-- An event with a probability of less than 5% -/
def low_probability_event : Prop := sorry

theorem correlation_significance :
  |r| > r_0_05 → low_probability_event := by sorry

end correlation_significance_l896_89611


namespace inequality_solution_set_l896_89685

-- Define the base-10 logarithm function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the solution set
def solution_set : Set ℝ := {x | -4 < x ∧ x < 2}

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | log10 (x^2 + 2*x + 2) < 1} = solution_set :=
sorry

end inequality_solution_set_l896_89685


namespace markup_percentages_correct_l896_89635

/-- Represents an item with its purchase price, overhead percentage, and desired net profit. -/
structure Item where
  purchase_price : ℕ
  overhead_percent : ℕ
  net_profit : ℕ

/-- Calculates the selling price of an item, rounded up to the nearest whole dollar. -/
def selling_price (item : Item) : ℕ :=
  let total_cost := item.purchase_price + (item.purchase_price * item.overhead_percent / 100) + item.net_profit
  (total_cost + 99) / 100 * 100

/-- Calculates the markup percentage for an item, rounded up to the nearest whole percent. -/
def markup_percentage (item : Item) : ℕ :=
  let markup := selling_price item - item.purchase_price
  ((markup * 100 + item.purchase_price - 1) / item.purchase_price)

theorem markup_percentages_correct (item_a item_b item_c : Item) : 
  item_a.purchase_price = 48 ∧ 
  item_a.overhead_percent = 20 ∧ 
  item_a.net_profit = 12 ∧
  item_b.purchase_price = 36 ∧ 
  item_b.overhead_percent = 15 ∧ 
  item_b.net_profit = 8 ∧
  item_c.purchase_price = 60 ∧ 
  item_c.overhead_percent = 25 ∧ 
  item_c.net_profit = 16 →
  markup_percentage item_a = 46 ∧
  markup_percentage item_b = 39 ∧
  markup_percentage item_c = 52 := by
  sorry

end markup_percentages_correct_l896_89635


namespace square_fraction_count_l896_89688

theorem square_fraction_count : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, ∃ m : ℤ, n / (25 - n) = m^2) ∧ 
    (∀ n : ℤ, n ∉ S → ¬∃ m : ℤ, n / (25 - n) = m^2) ∧ 
    S.card = 2 :=
by sorry

end square_fraction_count_l896_89688


namespace max_identifiable_bulbs_max_identifiable_bulbs_and_switches_l896_89609

/-- Represents the state of a bulb -/
inductive BulbState
  | On
  | OffWarm
  | OffCold

/-- Represents a trip to the basement -/
def Trip := Nat → BulbState

/-- The number of trips allowed to the basement -/
def numTrips : Nat := 2

/-- The number of possible states for each bulb in a single trip -/
def statesPerTrip : Nat := 3

/-- Theorem: The maximum number of unique bulb configurations identifiable in two trips -/
theorem max_identifiable_bulbs :
  (statesPerTrip ^ numTrips : Nat) = 9 := by
  sorry

/-- Corollary: The maximum number of bulbs and switches that can be identified with each other in two trips -/
theorem max_identifiable_bulbs_and_switches :
  ∃ (n : Nat), n = 9 ∧ n = (statesPerTrip ^ numTrips : Nat) := by
  sorry

end max_identifiable_bulbs_max_identifiable_bulbs_and_switches_l896_89609


namespace wrapping_paper_usage_l896_89648

theorem wrapping_paper_usage 
  (total_used : ℚ) 
  (num_presents : ℕ) 
  (h1 : total_used = 4 / 15) 
  (h2 : num_presents = 5) :
  total_used / num_presents = 4 / 75 :=
by sorry

end wrapping_paper_usage_l896_89648


namespace imaginary_part_of_complex_product_l896_89675

theorem imaginary_part_of_complex_product : Complex.im ((4 - 8 * Complex.I) * Complex.I) = 4 := by
  sorry

end imaginary_part_of_complex_product_l896_89675


namespace unique_integer_solution_l896_89651

theorem unique_integer_solution (a b c : ℤ) : 
  a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end unique_integer_solution_l896_89651


namespace area_of_triangle_PQR_l896_89657

/-- Given two lines intersecting at P(2,5) with slopes 3 and 1 respectively,
    and Q and R as the intersections of these lines with the x-axis,
    prove that the area of triangle PQR is 25/3 -/
theorem area_of_triangle_PQR (P Q R : ℝ × ℝ) : 
  P = (2, 5) →
  (∃ m₁ m₂ : ℝ, m₁ = 3 ∧ m₂ = 1 ∧ 
    (∀ x y : ℝ, y - 5 = m₁ * (x - 2) ∨ y - 5 = m₂ * (x - 2))) →
  Q.2 = 0 ∧ R.2 = 0 →
  (∃ x₁ x₂ : ℝ, Q = (x₁, 0) ∧ R = (x₂, 0) ∧ 
    (5 - 0) = 3 * (2 - x₁) ∧ (5 - 0) = 1 * (2 - x₂)) →
  (1/2 : ℝ) * |Q.1 - R.1| * 5 = 25/3 :=
by sorry

end area_of_triangle_PQR_l896_89657


namespace second_chapter_pages_l896_89692

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  chapter1_pages : ℕ
  chapter2_pages : ℕ
  two_chapters : chapter1_pages + chapter2_pages = total_pages

/-- The specific book in the problem -/
def problem_book : Book where
  total_pages := 93
  chapter1_pages := 60
  chapter2_pages := 33
  two_chapters := by sorry

theorem second_chapter_pages (b : Book) 
  (h1 : b.total_pages = 93) 
  (h2 : b.chapter1_pages = 60) : 
  b.chapter2_pages = 33 := by
  sorry

end second_chapter_pages_l896_89692


namespace cube_volume_from_space_diagonal_l896_89664

theorem cube_volume_from_space_diagonal :
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 10 * Real.sqrt 3 → s^3 = 1000 := by
  sorry

end cube_volume_from_space_diagonal_l896_89664


namespace ball_bounce_distance_l896_89682

/-- The sum of an infinite geometric series with first term h and common ratio 0.8 is equal to 5h -/
theorem ball_bounce_distance (h : ℝ) (h_pos : h > 0) : 
  (∑' n, h * (0.8 ^ n)) = 5 * h := by sorry

end ball_bounce_distance_l896_89682


namespace right_triangle_three_four_five_l896_89632

theorem right_triangle_three_four_five :
  ∀ (a b c : ℝ),
    a = 3 ∧ b = 4 ∧ c = 5 →
    a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_three_four_five_l896_89632


namespace tiffany_treasures_l896_89608

/-- The number of points each treasure is worth -/
def points_per_treasure : ℕ := 6

/-- The number of treasures Tiffany found on the second level -/
def treasures_second_level : ℕ := 5

/-- Tiffany's total score -/
def total_score : ℕ := 48

/-- The number of treasures Tiffany found on the first level -/
def treasures_first_level : ℕ := (total_score - points_per_treasure * treasures_second_level) / points_per_treasure

theorem tiffany_treasures : treasures_first_level = 3 := by
  sorry

end tiffany_treasures_l896_89608


namespace triangle_abc_properties_l896_89626

noncomputable section

/-- Triangle ABC with given properties -/
structure TriangleABC where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Given conditions
  a_eq : a = Real.sqrt 5
  b_eq : b = 3
  sin_C_eq : Real.sin C = 2 * Real.sin A
  -- Triangle inequality and angle sum
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b
  angle_sum : A + B + C = π

theorem triangle_abc_properties (t : TriangleABC) : 
  t.c = 2 * Real.sqrt 5 ∧ 
  Real.sin (2 * t.A - π/4) = Real.sqrt 2 / 10 := by
  sorry

end

end triangle_abc_properties_l896_89626


namespace used_car_selection_l896_89600

theorem used_car_selection (num_cars : ℕ) (num_clients : ℕ) (selections_per_car : ℕ) 
  (h1 : num_cars = 16)
  (h2 : num_clients = 24)
  (h3 : selections_per_car = 3) :
  (num_cars * selections_per_car) / num_clients = 2 := by
  sorry

end used_car_selection_l896_89600


namespace value_of_a_minus_b_l896_89620

theorem value_of_a_minus_b (a b : ℝ) : (a - 5)^2 + |b^3 - 27| = 0 → a - b = 2 := by
  sorry

end value_of_a_minus_b_l896_89620


namespace min_speed_to_arrive_earlier_l896_89623

/-- Proves the minimum speed required for the second person to arrive earlier -/
theorem min_speed_to_arrive_earlier
  (distance : ℝ)
  (speed_A : ℝ)
  (delay : ℝ)
  (h_distance : distance = 180)
  (h_speed_A : speed_A = 30)
  (h_delay : delay = 2) :
  ∀ speed_B : ℝ, speed_B > 45 →
    distance / speed_B + delay < distance / speed_A :=
by sorry

end min_speed_to_arrive_earlier_l896_89623


namespace sum_of_roots_l896_89619

theorem sum_of_roots (y₁ y₂ k m : ℝ) (h1 : y₁ ≠ y₂) 
  (h2 : 5 * y₁^2 - k * y₁ = m) (h3 : 5 * y₂^2 - k * y₂ = m) : 
  y₁ + y₂ = k / 5 := by
sorry

end sum_of_roots_l896_89619


namespace solution_set_theorem_l896_89617

/-- A function f: ℝ → ℝ is increasing -/
def IsIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

/-- The set of x where |f(x)| ≥ 2 -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ := {x : ℝ | |f x| ≥ 2}

theorem solution_set_theorem (f : ℝ → ℝ) 
  (h_increasing : IsIncreasing f) 
  (h_f1 : f 1 = -2) 
  (h_f3 : f 3 = 2) : 
  SolutionSet f = Set.Ici 3 ∪ Set.Iic 1 := by
  sorry

end solution_set_theorem_l896_89617


namespace income_comparison_l896_89644

theorem income_comparison (juan tim other : ℝ) 
  (h1 : tim = juan * (1 - 0.5))
  (h2 : other = juan * 0.8) :
  other = tim * 1.6 := by
  sorry

end income_comparison_l896_89644


namespace fixed_points_for_specific_values_condition_for_two_fixed_points_minimum_b_value_l896_89639

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define what it means for x to be a fixed point of f
def is_fixed_point (a b x : ℝ) : Prop := f a b x = x

-- Statement 1
theorem fixed_points_for_specific_values :
  is_fixed_point 1 3 (-2) ∧ is_fixed_point 1 3 (-1) :=
sorry

-- Statement 2
theorem condition_for_two_fixed_points :
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point a b x ∧ is_fixed_point a b y) ↔
  (0 < a ∧ a < 1) :=
sorry

-- Statement 3
theorem minimum_b_value (a : ℝ) (h : 0 < a ∧ a < 1) :
  let g (x : ℝ) := -x + (2 * a) / (5 * a^2 - 4 * a + 1)
  ∃ b x y : ℝ, x ≠ y ∧ 
    is_fixed_point a b x ∧ 
    is_fixed_point a b y ∧ 
    g ((x + y) / 2) = (x + y) / 2 ∧
    (∀ b' : ℝ, b' ≥ b) ∧
    b = -2 :=
sorry

end fixed_points_for_specific_values_condition_for_two_fixed_points_minimum_b_value_l896_89639


namespace existence_of_special_integer_l896_89670

theorem existence_of_special_integer :
  ∃ (A : ℕ), 
    (∃ (n : ℕ), A = n * (n + 1) * (n + 2)) ∧
    (∃ (k : ℕ), (A / 10^k) % 10^99 = 10^99 - 1) := by
  sorry

end existence_of_special_integer_l896_89670


namespace tank_capacity_is_2000_liters_l896_89663

-- Define the flow rates and time
def inflow_rate : ℚ := 1 / 2 -- kiloliters per minute
def outflow_rate1 : ℚ := 1 / 4 -- kiloliters per minute
def outflow_rate2 : ℚ := 1 / 6 -- kiloliters per minute
def fill_time : ℚ := 12 -- minutes

-- Define the net flow rate
def net_flow_rate : ℚ := inflow_rate - outflow_rate1 - outflow_rate2

-- Define the theorem
theorem tank_capacity_is_2000_liters :
  let volume_added : ℚ := net_flow_rate * fill_time
  let full_capacity_kl : ℚ := 2 * volume_added
  let full_capacity_l : ℚ := 1000 * full_capacity_kl
  full_capacity_l = 2000 := by sorry

end tank_capacity_is_2000_liters_l896_89663


namespace quiz_competition_participants_l896_89697

theorem quiz_competition_participants :
  let initial_participants : ℕ := 300
  let first_round_ratio : ℚ := 2/5
  let second_round_ratio : ℚ := 1/4
  let final_participants : ℕ := 30
  (initial_participants : ℚ) * first_round_ratio * second_round_ratio = final_participants := by
sorry

end quiz_competition_participants_l896_89697


namespace not_in_range_iff_b_in_interval_l896_89665

/-- The function g(x) defined as x^2 + bx + 1 -/
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

/-- Theorem stating that -3 is not in the range of g(x) if and only if b is in the open interval (-4, 4) -/
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, g b x ≠ -3) ↔ b ∈ Set.Ioo (-4 : ℝ) 4 :=
sorry

end not_in_range_iff_b_in_interval_l896_89665


namespace probability_log_integer_l896_89631

def S : Set ℕ := {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 20 ∧ n = 3^k}

def is_valid_pair (a b : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ∃ k : ℕ, b = a^k

def total_pairs : ℕ := Nat.choose 20 2

def valid_pairs : ℕ := 48

theorem probability_log_integer :
  (valid_pairs : ℚ) / total_pairs = 24 / 95 := by sorry

end probability_log_integer_l896_89631


namespace board_number_game_l896_89646

theorem board_number_game (n : ℕ) (h : n = 2009) : 
  let initial_sum := n * (n + 1) / 2
  let initial_remainder := initial_sum % 13
  ∃ (a : ℕ), a ≤ n ∧ (a + 9 + 999) % 13 = initial_remainder ∧ a = 8 :=
sorry

end board_number_game_l896_89646


namespace smallest_positive_multiple_of_45_l896_89618

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end smallest_positive_multiple_of_45_l896_89618


namespace waitress_tips_fraction_l896_89654

theorem waitress_tips_fraction (salary : ℝ) (tips : ℝ) (h : tips = 2/4 * salary) :
  tips / (salary + tips) = 1/3 := by
  sorry

end waitress_tips_fraction_l896_89654


namespace sin_2x_value_l896_89641

theorem sin_2x_value (x : ℝ) (h : Real.tan (π - x) = 3) : Real.sin (2 * x) = -3/5 := by
  sorry

end sin_2x_value_l896_89641


namespace base7_25_to_binary_l896_89674

/-- Converts a number from base 7 to base 10 -/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 2 -/
def decimalToBinary (n : ℕ) : List ℕ := sorry

theorem base7_25_to_binary :
  decimalToBinary (base7ToDecimal 25) = [1, 0, 0, 1, 1] := by sorry

end base7_25_to_binary_l896_89674


namespace even_perfect_square_ablab_l896_89658

theorem even_perfect_square_ablab : 
  ∃! n : ℕ, 
    (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ 
      n = 10000 * a + 1000 * b + 100 + 10 * a + b) ∧ 
    (∃ m : ℕ, n = m^2) ∧ 
    (∃ k : ℕ, n = 2 * k) ∧
    n = 76176 :=
by
  sorry

end even_perfect_square_ablab_l896_89658


namespace unique_valid_assignment_l896_89686

/-- Represents the possible arithmetic operations --/
inductive Operation
  | Plus
  | Minus
  | Multiply
  | Divide
  | Equal

/-- Represents the assignment of operations to letters --/
structure Assignment :=
  (A B C D E : Operation)

/-- Checks if an assignment is valid according to the problem conditions --/
def is_valid_assignment (a : Assignment) : Prop :=
  a.A ≠ a.B ∧ a.A ≠ a.C ∧ a.A ≠ a.D ∧ a.A ≠ a.E ∧
  a.B ≠ a.C ∧ a.B ≠ a.D ∧ a.B ≠ a.E ∧
  a.C ≠ a.D ∧ a.C ≠ a.E ∧
  a.D ≠ a.E ∧
  (a.A = Operation.Plus ∨ a.B = Operation.Plus ∨ a.C = Operation.Plus ∨ a.D = Operation.Plus ∨ a.E = Operation.Plus) ∧
  (a.A = Operation.Minus ∨ a.B = Operation.Minus ∨ a.C = Operation.Minus ∨ a.D = Operation.Minus ∨ a.E = Operation.Minus) ∧
  (a.A = Operation.Multiply ∨ a.B = Operation.Multiply ∨ a.C = Operation.Multiply ∨ a.D = Operation.Multiply ∨ a.E = Operation.Multiply) ∧
  (a.A = Operation.Divide ∨ a.B = Operation.Divide ∨ a.C = Operation.Divide ∨ a.D = Operation.Divide ∨ a.E = Operation.Divide) ∧
  (a.A = Operation.Equal ∨ a.B = Operation.Equal ∨ a.C = Operation.Equal ∨ a.D = Operation.Equal ∨ a.E = Operation.Equal)

/-- Checks if an assignment satisfies the equations --/
def satisfies_equations (a : Assignment) : Prop :=
  (a.A = Operation.Divide ∧ 4 / 2 = 2) ∧
  (a.B = Operation.Equal) ∧
  (a.C = Operation.Multiply ∧ 4 * 2 = 8) ∧
  (a.D = Operation.Plus ∧ 2 + 3 = 5) ∧
  (a.E = Operation.Minus ∧ 5 - 1 = 4)

/-- The main theorem: there is a unique valid assignment that satisfies the equations --/
theorem unique_valid_assignment :
  ∃! (a : Assignment), is_valid_assignment a ∧ satisfies_equations a :=
sorry

end unique_valid_assignment_l896_89686


namespace pats_stickers_l896_89624

/-- Pat's sticker problem -/
theorem pats_stickers (initial_stickers earned_stickers : ℕ) 
  (h1 : initial_stickers = 39)
  (h2 : earned_stickers = 22) :
  initial_stickers + earned_stickers = 61 :=
by sorry

end pats_stickers_l896_89624


namespace cube_volume_and_surface_area_l896_89615

/-- Represents a cube with edge length in centimeters -/
structure Cube where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

/-- The sum of all edge lengths of the cube -/
def Cube.sumEdgeLength (c : Cube) : ℝ := 12 * c.edgeLength

/-- The volume of the cube -/
def Cube.volume (c : Cube) : ℝ := c.edgeLength ^ 3

/-- The surface area of the cube -/
def Cube.surfaceArea (c : Cube) : ℝ := 6 * c.edgeLength ^ 2

theorem cube_volume_and_surface_area 
  (c : Cube) 
  (h : c.sumEdgeLength = 72) : 
  c.volume = 216 ∧ c.surfaceArea = 216 := by
  sorry

end cube_volume_and_surface_area_l896_89615


namespace min_bailing_rate_is_14_l896_89630

/-- Represents the scenario of Amy and Boris in the leaking boat --/
structure BoatScenario where
  distance_to_shore : Real
  water_intake_rate : Real
  sinking_threshold : Real
  initial_speed : Real
  speed_increase : Real
  speed_increase_interval : Real

/-- Calculates the time taken to reach the shore --/
def time_to_shore (scenario : BoatScenario) : Real :=
  sorry

/-- Calculates the total potential water intake --/
def total_water_intake (scenario : BoatScenario) (time : Real) : Real :=
  scenario.water_intake_rate * time

/-- Calculates the minimum bailing rate required --/
def min_bailing_rate (scenario : BoatScenario) : Real :=
  sorry

/-- The main theorem stating the minimum bailing rate for the given scenario --/
theorem min_bailing_rate_is_14 (scenario : BoatScenario) 
  (h1 : scenario.distance_to_shore = 2)
  (h2 : scenario.water_intake_rate = 15)
  (h3 : scenario.sinking_threshold = 50)
  (h4 : scenario.initial_speed = 2)
  (h5 : scenario.speed_increase = 1)
  (h6 : scenario.speed_increase_interval = 0.5) :
  min_bailing_rate scenario = 14 := by
  sorry

end min_bailing_rate_is_14_l896_89630


namespace alternating_7x7_grid_difference_l896_89603

/-- Represents a square on the grid -/
inductive Square
| Dark
| Light

/-- Represents a row in the grid -/
def Row := List Square

/-- Represents the entire grid -/
def Grid := List Row

/-- Generates an alternating row starting with the given square type -/
def alternatingRow (start : Square) (length : Nat) : Row :=
  sorry

/-- Counts the number of dark squares in a row -/
def countDarkInRow (row : Row) : Nat :=
  sorry

/-- Counts the number of light squares in a row -/
def countLightInRow (row : Row) : Nat :=
  sorry

/-- Generates a 7x7 grid with alternating squares, starting with a dark square -/
def generateGrid : Grid :=
  sorry

/-- Counts the total number of dark squares in the grid -/
def countTotalDark (grid : Grid) : Nat :=
  sorry

/-- Counts the total number of light squares in the grid -/
def countTotalLight (grid : Grid) : Nat :=
  sorry

theorem alternating_7x7_grid_difference :
  let grid := generateGrid
  countTotalDark grid = countTotalLight grid + 1 := by
  sorry

end alternating_7x7_grid_difference_l896_89603


namespace subset_empty_range_superset_range_l896_89696

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a - 1}

-- Theorem for part I
theorem subset_empty_range : ¬∃ a : ℝ, M ⊆ N a := by sorry

-- Theorem for part II
theorem superset_range : {a : ℝ | M ⊇ N a} = {a : ℝ | a ≤ 3} := by sorry

end subset_empty_range_superset_range_l896_89696


namespace f_two_range_l896_89676

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the theorem
theorem f_two_range (a b c : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ 
   f a b c 1 = 0 ∧ f a b c x₁ = 0 ∧ f a b c x₂ = 0) →
  (∀ x : ℝ, f a b c (x + 1) = -f a b c (-x + 1)) →
  0 < f a b c 2 ∧ f a b c 2 < 1 :=
by sorry

end f_two_range_l896_89676


namespace square_expression_l896_89606

theorem square_expression (a b : ℝ) (square : ℝ) :
  square * (2 * a * b) = 4 * a^2 * b → square = 2 * a :=
by
  sorry

end square_expression_l896_89606


namespace first_nonzero_digit_of_1_over_143_l896_89690

theorem first_nonzero_digit_of_1_over_143 : ∃ (n : ℕ) (d : ℕ), 
  (1 : ℚ) / 143 = (n : ℚ) / 10^d ∧ 
  n % 10 = 7 ∧ 
  ∀ (m : ℕ), m < d → (1 : ℚ) / 143 * 10^m < 1 :=
sorry

end first_nonzero_digit_of_1_over_143_l896_89690


namespace ceva_triangle_ratio_product_l896_89604

/-- Given a triangle ABC with points A', B', C' on sides BC, AC, AB respectively,
    and lines AA', BB', CC' intersecting at point O, if the sum of the ratios
    AO/OA', BO/OB', and CO/OC' is 56, then the square of their product is 2916. -/
theorem ceva_triangle_ratio_product (A B C A' B' C' O : ℝ × ℝ) : 
  let ratio (P Q R : ℝ × ℝ) := dist P Q / dist Q R
  (ratio O A A' + ratio O B B' + ratio O C C' = 56) →
  (ratio O A A' * ratio O B B' * ratio O C C')^2 = 2916 := by
  sorry

end ceva_triangle_ratio_product_l896_89604


namespace digit_property_characterization_l896_89693

def has_property (z : Nat) : Prop :=
  z < 10 ∧ 
  ∀ k : Nat, k ≥ 1 → 
    ∃ n : Nat, n ≥ 1 ∧ 
      ∃ m : Nat, n^9 = m * 10^k + z * ((10^k - 1) / 9)

theorem digit_property_characterization :
  ∀ z : Nat, has_property z ↔ z ∈ ({0, 1, 3, 7, 9} : Set Nat) :=
sorry

end digit_property_characterization_l896_89693


namespace natural_number_divisibility_l896_89669

theorem natural_number_divisibility (a b : ℕ) : 
  (∃ k : ℕ, a = k * (b + 1)) → 
  (∃ m : ℕ, 43 = m * (a + b)) → 
  ((a = 22 ∧ b = 21) ∨ 
   (a = 33 ∧ b = 10) ∨ 
   (a = 40 ∧ b = 3) ∨ 
   (a = 42 ∧ b = 1)) := by
sorry

end natural_number_divisibility_l896_89669


namespace fill_measuring_cup_l896_89602

/-- The capacity of a spoon in milliliters -/
def spoon_capacity : ℝ := 5

/-- The volume of a measuring cup in liters -/
def cup_volume : ℝ := 1

/-- The conversion factor from liters to milliliters -/
def liter_to_ml : ℝ := 1000

/-- The number of spoons needed to fill the measuring cup -/
def spoons_needed : ℕ := 200

theorem fill_measuring_cup : 
  ⌊(cup_volume * liter_to_ml) / spoon_capacity⌋ = spoons_needed := by
  sorry

end fill_measuring_cup_l896_89602


namespace cos_alpha_minus_beta_l896_89636

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : Real.sin α + Real.sin β = 1/2) 
  (h2 : Real.cos α + Real.cos β = 1/3) : 
  Real.cos (α - β) = -(59/72) := by sorry

end cos_alpha_minus_beta_l896_89636


namespace ben_is_25_l896_89677

/-- Ben's age -/
def ben_age : ℕ := sorry

/-- Dan's age -/
def dan_age : ℕ := sorry

/-- Ben is 3 years younger than Dan -/
axiom age_difference : ben_age = dan_age - 3

/-- The sum of their ages is 53 -/
axiom age_sum : ben_age + dan_age = 53

theorem ben_is_25 : ben_age = 25 := by sorry

end ben_is_25_l896_89677


namespace second_month_sale_l896_89655

/-- Represents the sales data for a grocery shop over 6 months -/
structure GrocerySales where
  month1 : ℕ
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ

/-- Calculates the average sale over 6 months -/
def average_sale (sales : GrocerySales) : ℚ :=
  (sales.month1 + sales.month2 + sales.month3 + sales.month4 + sales.month5 + sales.month6) / 6

/-- Theorem stating the conditions and the result to be proved -/
theorem second_month_sale 
  (sales : GrocerySales)
  (h1 : sales.month1 = 6435)
  (h2 : sales.month3 = 7230)
  (h3 : sales.month4 = 6562)
  (h4 : sales.month6 = 4991)
  (h5 : average_sale sales = 6500) :
  sales.month2 = 13782 := by
  sorry

end second_month_sale_l896_89655


namespace remaining_game_price_l896_89625

def total_games : ℕ := 346
def expensive_games : ℕ := 80
def expensive_price : ℕ := 12
def mid_price : ℕ := 7
def total_spent : ℕ := 2290

theorem remaining_game_price :
  let remaining_games := total_games - expensive_games
  let mid_games := remaining_games / 2
  let cheap_games := remaining_games - mid_games
  let spent_on_expensive := expensive_games * expensive_price
  let spent_on_mid := mid_games * mid_price
  let spent_on_cheap := total_spent - spent_on_expensive - spent_on_mid
  spent_on_cheap / cheap_games = 3 := by
sorry

end remaining_game_price_l896_89625


namespace sum_of_digits_properties_l896_89680

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem sum_of_digits_properties :
  (∀ n : ℕ, sum_of_digits (2 * n) ≤ 2 * sum_of_digits n) ∧
  (∀ n : ℕ, 2 * sum_of_digits n ≤ 10 * sum_of_digits (2 * n)) ∧
  (∃ k : ℕ, sum_of_digits k = 1996 * sum_of_digits (3 * k)) := by
  sorry

end sum_of_digits_properties_l896_89680


namespace max_profit_appliance_business_l896_89637

/-- Represents the cost and profit structure for small electrical appliances --/
structure ApplianceBusiness where
  cost_a : ℝ  -- Cost of one unit of type A
  cost_b : ℝ  -- Cost of one unit of type B
  profit_a : ℝ  -- Profit from selling one unit of type A
  profit_b : ℝ  -- Profit from selling one unit of type B

/-- Theorem stating the maximum profit for the given business scenario --/
theorem max_profit_appliance_business 
  (business : ApplianceBusiness)
  (h1 : 2 * business.cost_a + 3 * business.cost_b = 90)
  (h2 : 3 * business.cost_a + business.cost_b = 65)
  (h3 : business.profit_a = 3)
  (h4 : business.profit_b = 4)
  (h5 : ∀ a : ℕ, 30 ≤ a ∧ a ≤ 50 → 
    2750 ≤ a * business.cost_a + (150 - a) * business.cost_b ∧
    a * business.cost_a + (150 - a) * business.cost_b ≤ 2850)
  (h6 : ∀ a : ℕ, 30 ≤ a ∧ a ≤ 35 → 
    565 ≤ a * business.profit_a + (150 - a) * business.profit_b) :
  ∃ (max_profit : ℝ), 
    max_profit = 30 * business.profit_a + 120 * business.profit_b ∧
    max_profit = 570 ∧
    ∀ (a : ℕ), 30 ≤ a ∧ a ≤ 35 → 
      a * business.profit_a + (150 - a) * business.profit_b ≤ max_profit :=
by sorry


end max_profit_appliance_business_l896_89637


namespace q_join_time_l896_89656

/-- Represents the number of months after which Q joined the business --/
def x : ℕ := sorry

/-- P's initial investment --/
def p_investment : ℕ := 4000

/-- Q's investment --/
def q_investment : ℕ := 9000

/-- Total number of months in a year --/
def total_months : ℕ := 12

/-- Ratio of P's profit share to Q's profit share --/
def profit_ratio : ℚ := 2 / 3

theorem q_join_time :
  (p_investment * total_months) / (q_investment * (total_months - x)) = profit_ratio →
  x = 4 := by sorry

end q_join_time_l896_89656


namespace unclaimed_books_fraction_l896_89659

/-- Represents the fraction of books each person takes -/
def take_books (total : ℚ) (fraction : ℚ) (remaining : ℚ) : ℚ :=
  fraction * remaining

/-- The fraction of books that goes unclaimed after all four people take their share -/
def unclaimed_fraction : ℚ :=
  let total := 1
  let al_takes := take_books total (2/5) total
  let bert_takes := take_books total (3/10) (total - al_takes)
  let carl_takes := take_books total (1/5) (total - al_takes - bert_takes)
  let dan_takes := take_books total (1/10) (total - al_takes - bert_takes - carl_takes)
  total - (al_takes + bert_takes + carl_takes + dan_takes)

theorem unclaimed_books_fraction :
  unclaimed_fraction = 1701 / 2500 :=
sorry

end unclaimed_books_fraction_l896_89659


namespace distance_focus_to_asymptotes_l896_89653

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the asymptotes of the hyperbola
def asymptote_pos (x y : ℝ) : Prop := y = Real.sqrt 2 * x
def asymptote_neg (x y : ℝ) : Prop := y = -Real.sqrt 2 * x

-- State the theorem
theorem distance_focus_to_asymptotes :
  ∃ (d : ℝ), d = Real.sqrt 6 / 3 ∧
  (∀ (x y : ℝ), asymptote_pos x y →
    d = abs (Real.sqrt 2 * focus.1 - focus.2) / Real.sqrt (1 + 2)) ∧
  (∀ (x y : ℝ), asymptote_neg x y →
    d = abs (-Real.sqrt 2 * focus.1 - focus.2) / Real.sqrt (1 + 2)) :=
sorry

end distance_focus_to_asymptotes_l896_89653


namespace radium_decay_heat_equivalence_l896_89629

/-- The amount of radium in the Earth's crust in kilograms -/
def radium_in_crust : ℝ := 10000000000

/-- The amount of coal in kilograms that releases equivalent heat to 1 kg of radium decay -/
def coal_equivalent : ℝ := 375000

/-- The amount of coal in kilograms that releases equivalent heat to the complete decay of radium in Earth's crust -/
def total_coal_equivalent : ℝ := radium_in_crust * coal_equivalent

theorem radium_decay_heat_equivalence :
  total_coal_equivalent = 3.75 * (10 ^ 15) := by
  sorry

end radium_decay_heat_equivalence_l896_89629


namespace not_square_sum_ceiling_l896_89601

theorem not_square_sum_ceiling (a b : ℕ+) : ¬∃ (n : ℕ), (n : ℝ)^2 = (a : ℝ)^2 + ⌈(4 * (a : ℝ)^2) / (b : ℝ)⌉ := by
  sorry

end not_square_sum_ceiling_l896_89601


namespace total_votes_is_82_l896_89627

/-- Represents the number of votes for each cake type -/
structure CakeVotes where
  unicorn : ℕ
  witch : ℕ
  dragon : ℕ
  mermaid : ℕ
  fairy : ℕ

/-- Conditions for the baking contest votes -/
def contestConditions (votes : CakeVotes) : Prop :=
  votes.witch = 12 ∧
  votes.unicorn = 3 * votes.witch ∧
  votes.dragon = votes.witch + (2 * votes.witch / 5) ∧
  votes.mermaid = votes.dragon - 7 ∧
  votes.mermaid = 2 * votes.fairy ∧
  votes.fairy = votes.witch - 5

/-- Theorem stating that the total number of votes is 82 -/
theorem total_votes_is_82 (votes : CakeVotes) 
  (h : contestConditions votes) : 
  votes.unicorn + votes.witch + votes.dragon + votes.mermaid + votes.fairy = 82 := by
  sorry

end total_votes_is_82_l896_89627


namespace cost_price_calculation_l896_89613

/-- Given an article sold at $1800 with a 20% profit, prove that the cost price is $1500. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 1800)
  (h2 : profit_percentage = 20) :
  selling_price / (1 + profit_percentage / 100) = 1500 := by
sorry

end cost_price_calculation_l896_89613


namespace sample_size_example_l896_89647

/-- Definition of a sample size in a statistical context -/
def sample_size (population : ℕ) (selected : ℕ) : ℕ := selected

/-- Theorem: The sample size for 100 items selected from a population of 5000 is 100 -/
theorem sample_size_example : sample_size 5000 100 = 100 := by
  sorry

end sample_size_example_l896_89647


namespace factorial_difference_not_seven_l896_89652

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem factorial_difference_not_seven (a b : ℕ) (h : b > a) :
  ∃ k : ℕ, (factorial b - factorial a) % 10 ≠ 7 :=
sorry

end factorial_difference_not_seven_l896_89652


namespace power_of_three_equality_l896_89660

theorem power_of_three_equality (x : ℕ) :
  3^x = 3^20 * 3^20 * 3^18 + 3^19 * 3^20 * 3^19 + 3^18 * 3^21 * 3^19 → x = 59 := by
  sorry

end power_of_three_equality_l896_89660


namespace product_of_roots_undefined_expression_l896_89649

theorem product_of_roots_undefined_expression : ∃ (x y : ℝ),
  x^2 + 4*x - 5 = 0 ∧ 
  y^2 + 4*y - 5 = 0 ∧ 
  x * y = -5 :=
by sorry

end product_of_roots_undefined_expression_l896_89649


namespace domino_distribution_l896_89683

theorem domino_distribution (total_dominoes : ℕ) (num_players : ℕ) 
  (h1 : total_dominoes = 28) (h2 : num_players = 4) :
  total_dominoes / num_players = 7 := by
  sorry

end domino_distribution_l896_89683


namespace existence_of_special_n_l896_89687

theorem existence_of_special_n (t : ℕ) : ∃ n : ℕ, n > 1 ∧ 
  (Nat.gcd n t = 1) ∧ 
  (∀ k x m : ℕ, k ≥ 1 → m > 1 → n^k + t ≠ x^m) := by
  sorry

end existence_of_special_n_l896_89687


namespace circle_intersection_theorem_l896_89672

def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem circle_intersection_theorem :
  ∃ (x₁ y₁ x₂ y₂ m : ℝ),
    circle_equation x₁ y₁ m ∧
    circle_equation x₂ y₂ m ∧
    line_equation x₁ y₁ ∧
    line_equation x₂ y₂ ∧
    perpendicular x₁ y₁ x₂ y₂ →
    m = 8/5 ∧
    ∀ (x y : ℝ), (x - 4/5)^2 + (y - 8/5)^2 = 16/5 ↔
      circle_equation x y (8/5) :=
by sorry

end circle_intersection_theorem_l896_89672


namespace inscribed_rectangle_area_l896_89691

/-- A rectangle inscribed in a semicircle -/
structure InscribedRectangle where
  /-- The length of side PR of the rectangle -/
  pr : ℝ
  /-- The length of PG and SH, which are equal -/
  pg : ℝ
  /-- Assumption that PR is positive -/
  pr_pos : pr > 0
  /-- Assumption that PG is positive -/
  pg_pos : pg > 0

/-- The theorem stating that the area of the inscribed rectangle is 160√6 -/
theorem inscribed_rectangle_area (rect : InscribedRectangle) 
  (h1 : rect.pr = 20) (h2 : rect.pg = 12) : 
  ∃ (area : ℝ), area = rect.pr * Real.sqrt (rect.pg * (rect.pr + 2 * rect.pg - rect.pg)) ∧ 
  area = 160 * Real.sqrt 6 := by
  sorry

end inscribed_rectangle_area_l896_89691


namespace correct_seating_arrangements_l896_89634

/-- The number of ways to arrange n people in a row -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of people to be seated -/
def total_people : ℕ := 8

/-- The number of ways to arrange people with restrictions -/
def seating_arrangements : ℕ :=
  2 * factorial (total_people - 1) - 2 * factorial (total_people - 2) * factorial 2

theorem correct_seating_arrangements :
  seating_arrangements = 7200 := by
  sorry

#eval seating_arrangements

end correct_seating_arrangements_l896_89634


namespace unique_arrangement_l896_89668

/-- Represents a 4x4 grid with letters A, B, and C --/
def Grid := Fin 4 → Fin 4 → Char

/-- Checks if a given grid satisfies the arrangement conditions --/
def valid_arrangement (g : Grid) : Prop :=
  -- A is in the upper left corner
  g 0 0 = 'A' ∧
  -- Each row contains one of each letter
  (∀ i : Fin 4, ∃ j₁ j₂ j₃ : Fin 4, g i j₁ = 'A' ∧ g i j₂ = 'B' ∧ g i j₃ = 'C') ∧
  -- Each column contains one of each letter
  (∀ j : Fin 4, ∃ i₁ i₂ i₃ : Fin 4, g i₁ j = 'A' ∧ g i₂ j = 'B' ∧ g i₃ j = 'C') ∧
  -- Main diagonal (top-left to bottom-right) contains one of each letter
  (∃ i₁ i₂ i₃ : Fin 4, g i₁ i₁ = 'A' ∧ g i₂ i₂ = 'B' ∧ g i₃ i₃ = 'C') ∧
  -- Anti-diagonal (top-right to bottom-left) contains one of each letter
  (∃ i₁ i₂ i₃ : Fin 4, g i₁ (3 - i₁) = 'A' ∧ g i₂ (3 - i₂) = 'B' ∧ g i₃ (3 - i₃) = 'C')

/-- The main theorem stating there is only one valid arrangement --/
theorem unique_arrangement : ∃! g : Grid, valid_arrangement g :=
  sorry

end unique_arrangement_l896_89668
