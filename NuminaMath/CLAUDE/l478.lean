import Mathlib

namespace paper_towel_cost_l478_47880

theorem paper_towel_cost (case_price : ℝ) (savings_percent : ℝ) (rolls_per_case : ℕ) : 
  case_price = 9 ∧ savings_percent = 25 ∧ rolls_per_case = 12 →
  (case_price / (1 - savings_percent / 100)) / rolls_per_case = 0.9375 := by
sorry

end paper_towel_cost_l478_47880


namespace chris_leftover_money_l478_47875

theorem chris_leftover_money (
  video_game_cost : ℕ)
  (candy_cost : ℕ)
  (babysitting_rate : ℕ)
  (hours_worked : ℕ)
  (h1 : video_game_cost = 60)
  (h2 : candy_cost = 5)
  (h3 : babysitting_rate = 8)
  (h4 : hours_worked = 9) :
  babysitting_rate * hours_worked - (video_game_cost + candy_cost) = 7 := by
  sorry

end chris_leftover_money_l478_47875


namespace min_value_theorem_l478_47803

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) : 
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 1/a + 1/b + 1/c = 9 → 
  x^4 * y^3 * z^2 ≤ a^4 * b^3 * c^2 ∧ 
  x^4 * y^3 * z^2 = (1/3456 : ℝ) := by sorry

end min_value_theorem_l478_47803


namespace snack_sales_averages_l478_47833

/-- Represents the snack sales data for a special weekend event -/
structure EventData where
  tickets : ℕ
  crackers : ℕ
  crackerPrice : ℚ
  beverages : ℕ
  beveragePrice : ℚ
  chocolates : ℕ
  chocolatePrice : ℚ

/-- Calculates the total snack sales for an event -/
def totalSales (e : EventData) : ℚ :=
  e.crackers * e.crackerPrice + e.beverages * e.beveragePrice + e.chocolates * e.chocolatePrice

/-- Calculates the average snack sales per ticket for an event -/
def averageSales (e : EventData) : ℚ :=
  totalSales e / e.tickets

/-- Theorem stating the average snack sales for each event and the combined average -/
theorem snack_sales_averages 
  (valentines : EventData)
  (stPatricks : EventData)
  (christmas : EventData)
  (h1 : valentines = ⟨10, 4, 11/5, 6, 3/2, 7, 6/5⟩)
  (h2 : stPatricks = ⟨8, 3, 2, 5, 25/20, 8, 1⟩)
  (h3 : christmas = ⟨9, 6, 43/20, 4, 17/12, 9, 11/10⟩) :
  averageSales valentines = 131/50 ∧
  averageSales stPatricks = 253/100 ∧
  averageSales christmas = 79/25 ∧
  (totalSales valentines + totalSales stPatricks + totalSales christmas) / 
  (valentines.tickets + stPatricks.tickets + christmas.tickets) = 139/50 := by
  sorry

end snack_sales_averages_l478_47833


namespace same_color_shoe_probability_l478_47883

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 9

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The probability of selecting two shoes of the same color -/
def prob_same_color : ℚ := 9 / 2601

theorem same_color_shoe_probability :
  (num_pairs : ℚ) / (total_shoes - 1) / (total_shoes.choose 2) = prob_same_color := by
  sorry

end same_color_shoe_probability_l478_47883


namespace systematic_sampling_interval_example_l478_47813

/-- The interval for systematic sampling -/
def systematic_sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The systematic sampling interval for a population of 1200 and sample size of 30 is 40 -/
theorem systematic_sampling_interval_example :
  systematic_sampling_interval 1200 30 = 40 := by
  sorry

end systematic_sampling_interval_example_l478_47813


namespace choose_from_two_bags_l478_47848

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (m n : ℕ) : ℕ := m * n

/-- The number of balls in the red bag -/
def red_balls : ℕ := 3

/-- The number of balls in the blue bag -/
def blue_balls : ℕ := 5

/-- Theorem: The number of ways to choose one ball from the red bag and one from the blue bag is 15 -/
theorem choose_from_two_bags : choose_one_from_each red_balls blue_balls = 15 := by
  sorry

end choose_from_two_bags_l478_47848


namespace discount_ticket_price_l478_47834

theorem discount_ticket_price (discount_rate : ℝ) (discounted_price : ℝ) (original_price : ℝ) :
  discount_rate = 0.3 →
  discounted_price = 1400 →
  discounted_price = (1 - discount_rate) * original_price →
  original_price = 2000 := by
  sorry

end discount_ticket_price_l478_47834


namespace cube_difference_l478_47886

theorem cube_difference (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 40) : 
  a^3 - b^3 = 208 := by
sorry

end cube_difference_l478_47886


namespace replacement_process_terminates_l478_47801

/-- Represents a finite sequence of binary digits (0 or 1) -/
def BinarySequence := List Nat

/-- The operation that replaces "01" with "1000" in a binary sequence -/
def replace_operation (seq : BinarySequence) : BinarySequence :=
  sorry

/-- Predicate to check if a sequence contains the subsequence "01" -/
def has_replaceable_subsequence (seq : BinarySequence) : Prop :=
  sorry

/-- The number of ones in a binary sequence -/
def count_ones (seq : BinarySequence) : Nat :=
  sorry

theorem replacement_process_terminates (initial_seq : BinarySequence) :
  ∃ (n : Nat), ∀ (m : Nat), m ≥ n →
    ¬(has_replaceable_subsequence ((replace_operation^[m]) initial_seq)) :=
  sorry

end replacement_process_terminates_l478_47801


namespace max_min_values_on_interval_l478_47867

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values_on_interval :
  ∃ (a b : ℝ), a ∈ Set.Icc 0 3 ∧ b ∈ Set.Icc 0 3 ∧
  (∀ x, x ∈ Set.Icc 0 3 → f x ≤ f a) ∧
  (∀ x, x ∈ Set.Icc 0 3 → f x ≥ f b) ∧
  f a = 5 ∧ f b = -15 :=
sorry

end max_min_values_on_interval_l478_47867


namespace cost_of_jeans_and_shirts_l478_47868

/-- The cost of a pair of jeans -/
def jeans_cost : ℝ := sorry

/-- The cost of a shirt -/
def shirt_cost : ℝ := 9

theorem cost_of_jeans_and_shirts :
  3 * jeans_cost + 2 * shirt_cost = 69 :=
by
  have h1 : 2 * jeans_cost + 3 * shirt_cost = 61 := sorry
  sorry

end cost_of_jeans_and_shirts_l478_47868


namespace total_lunch_cost_l478_47874

/-- The total cost of lunches for a field trip --/
theorem total_lunch_cost : 
  let num_children : ℕ := 35
  let num_chaperones : ℕ := 5
  let num_teacher : ℕ := 1
  let num_additional : ℕ := 3
  let cost_per_lunch : ℕ := 7
  let total_lunches : ℕ := num_children + num_chaperones + num_teacher + num_additional
  total_lunches * cost_per_lunch = 308 :=
by sorry

end total_lunch_cost_l478_47874


namespace rectangle_perimeter_l478_47806

/-- The perimeter of a rectangle with length 15 inches and width 8 inches is 46 inches. -/
theorem rectangle_perimeter : 
  ∀ (length width perimeter : ℕ), 
  length = 15 → 
  width = 8 → 
  perimeter = 2 * (length + width) → 
  perimeter = 46 := by
  sorry

end rectangle_perimeter_l478_47806


namespace count_nines_to_hundred_l478_47896

/-- Count of digit 9 in a single number -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Sum of count_nines for numbers from 1 to n -/
def sum_nines (n : ℕ) : ℕ := sorry

/-- The theorem stating that the count of 9s in numbers from 1 to 100 is 19 -/
theorem count_nines_to_hundred : sum_nines 100 = 19 := by sorry

end count_nines_to_hundred_l478_47896


namespace anns_number_l478_47858

theorem anns_number (y : ℚ) : 5 * (3 * y + 15) = 200 → y = 25 / 3 := by
  sorry

end anns_number_l478_47858


namespace jennifer_remaining_money_l478_47811

def initial_amount : ℚ := 180

def sandwich_fraction : ℚ := 1 / 5
def museum_fraction : ℚ := 1 / 6
def book_fraction : ℚ := 1 / 2

def remaining_amount : ℚ := initial_amount * (1 - sandwich_fraction - museum_fraction - book_fraction)

theorem jennifer_remaining_money : remaining_amount = 24 := by
  sorry

end jennifer_remaining_money_l478_47811


namespace floor_sqrt_equality_l478_47817

theorem floor_sqrt_equality (n : ℕ) : ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ := by
  sorry

end floor_sqrt_equality_l478_47817


namespace A_work_days_l478_47822

/-- Represents the total work to be done -/
def W : ℝ := 1

/-- The number of days B alone can finish the work -/
def B_days : ℝ := 6

/-- The number of days A worked alone before B joined -/
def A_solo_days : ℝ := 3

/-- The number of days A and B worked together -/
def AB_days : ℝ := 3

/-- The number of days A can finish the work alone -/
def A_days : ℝ := 12

theorem A_work_days : 
  W = A_solo_days * (W / A_days) + AB_days * (W / A_days + W / B_days) → A_days = 12 := by
  sorry

end A_work_days_l478_47822


namespace sandy_clothes_cost_l478_47804

def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def jacket_cost : ℚ := 7.43

theorem sandy_clothes_cost :
  shorts_cost + shirt_cost + jacket_cost = 33.56 := by
  sorry

end sandy_clothes_cost_l478_47804


namespace point_on_line_l478_47800

/-- Given six points on a line and a point P satisfying certain conditions, prove OP -/
theorem point_on_line (a b c d e : ℝ) : 
  ∀ (O A B C D E P : ℝ), 
    O < A ∧ A < B ∧ B < C ∧ C < D ∧ D < E ∧   -- Points in order
    A - O = a ∧                               -- Distance OA
    B - O = b ∧                               -- Distance OB
    C - O = c ∧                               -- Distance OC
    D - O = d ∧                               -- Distance OE
    E - O = e ∧                               -- Distance OE
    C ≤ P ∧ P ≤ D ∧                           -- P between C and D
    (A - P) * (P - D) = (C - P) * (P - E) →   -- AP:PE = CP:PD
    P - O = (c * e - a * d) / (a - c + e - d) :=
by sorry

end point_on_line_l478_47800


namespace circles_externally_tangent_l478_47824

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- First circle: x^2 + y^2 = 4 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Second circle: (x-3)^2 + (y-4)^2 = 9 -/
def circle2 (x y : ℝ) : Prop := (x-3)^2 + (y-4)^2 = 9

theorem circles_externally_tangent :
  externally_tangent (0, 0) (3, 4) 2 3 := by sorry

end circles_externally_tangent_l478_47824


namespace correct_average_weight_l478_47891

theorem correct_average_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (misread_weight : ℝ) 
  (actual_weight : ℝ) : 
  n = 20 → 
  initial_average = 58.4 → 
  misread_weight = 56 → 
  actual_weight = 68 → 
  (n * initial_average + actual_weight - misread_weight) / n = 59 := by
  sorry

end correct_average_weight_l478_47891


namespace largest_number_bound_l478_47832

theorem largest_number_bound (a b : ℕ+) 
  (hcf_condition : Nat.gcd a b = 143)
  (lcm_condition : ∃ k : ℕ+, Nat.lcm a b = 143 * 17 * 23 * 31 * k) :
  max a b ≤ 143 * 17 * 23 * 31 := by
  sorry

end largest_number_bound_l478_47832


namespace ramu_car_profit_percent_l478_47827

/-- Calculates the profit percent from a car sale -/
def profit_percent (purchase_price repair_cost selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem: The profit percent for Ramu's car sale is approximately 41.30% -/
theorem ramu_car_profit_percent :
  let purchase_price : ℚ := 34000
  let repair_cost : ℚ := 12000
  let selling_price : ℚ := 65000
  abs (profit_percent purchase_price repair_cost selling_price - 41.30) < 0.01 := by
  sorry

#eval profit_percent 34000 12000 65000

end ramu_car_profit_percent_l478_47827


namespace unique_two_digit_number_l478_47808

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def ones_digit (n : ℕ) : ℕ := n % 10

def swap_digits (n : ℕ) : ℕ := 10 * (ones_digit n) + tens_digit n

theorem unique_two_digit_number : 
  ∃! n : ℕ, is_two_digit n ∧ 
    (tens_digit n * ones_digit n = 2 * (tens_digit n + ones_digit n)) ∧
    (n + 9 = 2 * swap_digits n) ∧
    n = 63 := by
  sorry

end unique_two_digit_number_l478_47808


namespace fourth_group_trees_l478_47842

theorem fourth_group_trees (total_groups : Nat) (average_trees : Nat)
  (group1_trees group2_trees group3_trees group5_trees : Nat) :
  total_groups = 5 →
  average_trees = 13 →
  group1_trees = 12 →
  group2_trees = 15 →
  group3_trees = 12 →
  group5_trees = 11 →
  (group1_trees + group2_trees + group3_trees + 15 + group5_trees) / total_groups = average_trees :=
by
  sorry

end fourth_group_trees_l478_47842


namespace cosine_symmetry_center_l478_47840

/-- Given a cosine function with a phase shift, prove that under certain symmetry conditions,
    the symmetric center closest to the origin is at a specific point when the period is maximized. -/
theorem cosine_symmetry_center (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x => Real.cos (ω * x + 3 * Real.pi / 4)
  (∀ x : ℝ, f (π / 3 - x) = f (π / 3 + x)) →  -- Symmetry about x = π/6
  (∀ k : ℤ, ω ≠ 6 * k - 9 / 2 → ω > 6 * k - 9 / 2) →  -- ω is the smallest positive value
  (π / 6 : ℝ) ∈ { x : ℝ | ∃ k : ℤ, x = 2 / 3 * k * π - π / 6 } →  -- Symmetric center formula
  (-π / 6 : ℝ) ∈ { x : ℝ | ∃ k : ℤ, x = 2 / 3 * k * π - π / 6 } ∧  -- Closest symmetric center
  (∀ x : ℝ, x ∈ { y : ℝ | ∃ k : ℤ, y = 2 / 3 * k * π - π / 6 } → |x| ≥ |(-π / 6 : ℝ)|) :=
by
  sorry

end cosine_symmetry_center_l478_47840


namespace monkey_climbing_time_l478_47846

/-- Monkey climbing problem -/
theorem monkey_climbing_time (tree_height : ℕ) (climb_rate : ℕ) (slip_rate : ℕ) : 
  tree_height = 22 ∧ climb_rate = 3 ∧ slip_rate = 2 → 
  (tree_height - 1) / (climb_rate - slip_rate) + 1 = 22 := by
sorry

end monkey_climbing_time_l478_47846


namespace arithmetic_sequence_max_product_l478_47861

/-- Given an arithmetic sequence {a_n} where a_4 = 2, the maximum value of a_2 * a_6 is 4. -/
theorem arithmetic_sequence_max_product (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  a 4 = 2 →                                        -- given condition
  ∃ (x : ℝ), x = a 2 * a 6 ∧ x ≤ 4 ∧ 
  ∀ (y : ℝ), y = a 2 * a 6 → y ≤ x :=
by sorry

end arithmetic_sequence_max_product_l478_47861


namespace pirate_treasure_probability_l478_47871

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 5

def prob_treasure : ℚ := 1/4
def prob_traps : ℚ := 1/12
def prob_neither : ℚ := 2/3

theorem pirate_treasure_probability :
  (num_islands.choose num_treasure_islands) * 
  (prob_treasure ^ num_treasure_islands) * 
  (prob_neither ^ (num_islands - num_treasure_islands)) = 7/432 := by
  sorry

end pirate_treasure_probability_l478_47871


namespace distinct_roots_sum_squares_l478_47821

theorem distinct_roots_sum_squares (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ → 
  x₁^2 + 2*x₁ - k = 0 → 
  x₂^2 + 2*x₂ - k = 0 → 
  x₁^2 + x₂^2 - 2 > 0 :=
by sorry

end distinct_roots_sum_squares_l478_47821


namespace problem_solution_l478_47872

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = -6) : 
  y = 33 := by
sorry

end problem_solution_l478_47872


namespace square_difference_given_product_and_sum_l478_47870

theorem square_difference_given_product_and_sum (p q : ℝ) 
  (h1 : p * q = 16) (h2 : p + q = 10) : (p - q)^2 = 36 := by
  sorry

end square_difference_given_product_and_sum_l478_47870


namespace toothpicks_stage_20_l478_47835

/-- The number of toothpicks in stage n of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  4 + 3 * (n - 1)

theorem toothpicks_stage_20 :
  toothpicks 20 = 61 := by
  sorry

end toothpicks_stage_20_l478_47835


namespace smallest_two_digit_multiple_of_17_l478_47812

theorem smallest_two_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 17 ∧ 
  n % 17 = 0 ∧ 
  10 ≤ n ∧ 
  n < 100 ∧ 
  ∀ m : ℕ, (m % 17 = 0 ∧ 10 ≤ m ∧ m < 100) → n ≤ m :=
by sorry

end smallest_two_digit_multiple_of_17_l478_47812


namespace intersection_A_B_l478_47890

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x - 2| ≥ 1}
def B : Set ℝ := {x : ℝ | 1 / x < 1}

-- State the theorem
theorem intersection_A_B : 
  ∀ x : ℝ, x ∈ A ∩ B ↔ x < 0 ∨ x ≥ 3 := by
  sorry

end intersection_A_B_l478_47890


namespace distinct_cube_configurations_l478_47847

-- Define the cube configuration
structure CubeConfig where
  white : Fin 8 → Fin 3
  blue : Fin 8 → Fin 3
  red : Fin 8 → Fin 2

-- Define the rotation group
def RotationGroup : Type := Unit

-- Define the action of the rotation group on cube configurations
def rotate : RotationGroup → CubeConfig → CubeConfig := sorry

-- Define the orbit of a cube configuration under rotations
def orbit (c : CubeConfig) : Set CubeConfig := sorry

-- Define the set of all valid cube configurations
def AllConfigs : Set CubeConfig := sorry

-- Count the number of distinct orbits
def countDistinctOrbits : ℕ := sorry

-- The main theorem
theorem distinct_cube_configurations :
  countDistinctOrbits = 25 := by sorry

end distinct_cube_configurations_l478_47847


namespace power_division_simplification_l478_47838

theorem power_division_simplification (a : ℝ) : (2 * a) ^ 7 / (2 * a) ^ 4 = 8 * a ^ 3 := by
  sorry

end power_division_simplification_l478_47838


namespace bird_lake_swans_l478_47845

theorem bird_lake_swans (total_birds : ℕ) (duck_fraction : ℚ) : 
  total_birds = 108 →
  duck_fraction = 5/6 →
  (1 - duck_fraction) * total_birds = 18 :=
by sorry

end bird_lake_swans_l478_47845


namespace max_value_of_a_l478_47893

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- State the theorem
theorem max_value_of_a :
  ∀ a : ℝ, (A a ∪ B a = Set.univ) → (∀ b : ℝ, (A b ∪ B b = Set.univ) → b ≤ a) → a = 2 :=
by sorry

end max_value_of_a_l478_47893


namespace pop_expenditure_l478_47869

theorem pop_expenditure (total : ℝ) (snap crackle pop : ℝ) : 
  total = 150 ∧ 
  snap = 2 * crackle ∧ 
  crackle = 3 * pop ∧ 
  total = snap + crackle + pop →
  pop = 15 := by
sorry

end pop_expenditure_l478_47869


namespace distance_is_60_l478_47866

/-- The distance between a boy's house and school. -/
def distance : ℝ := sorry

/-- The time it takes for the boy to reach school when arriving on time. -/
def on_time : ℝ := sorry

/-- Assertion that when traveling at 10 km/hr, the boy arrives 2 hours late. -/
axiom late_arrival : on_time + 2 = distance / 10

/-- Assertion that when traveling at 20 km/hr, the boy arrives 1 hour early. -/
axiom early_arrival : on_time - 1 = distance / 20

/-- Theorem stating that the distance between the boy's house and school is 60 kilometers. -/
theorem distance_is_60 : distance = 60 := by sorry

end distance_is_60_l478_47866


namespace remainder_problem_l478_47830

theorem remainder_problem : (7 * 10^20 + 2^20 + 5) % 9 = 7 := by sorry

end remainder_problem_l478_47830


namespace sequence_sum_l478_47876

theorem sequence_sum : ∀ (a b c d : ℕ), 
  (b - a = c - b) →  -- arithmetic progression
  (c * c = b * d) →  -- geometric progression
  (d = a + 50) →     -- difference between first and fourth terms
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) →  -- positive integers
  a + b + c + d = 215 := by sorry

end sequence_sum_l478_47876


namespace molecular_weight_CaOH2_is_74_10_l478_47839

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The number of calcium atoms in Ca(OH)2 -/
def num_Ca : ℕ := 1

/-- The number of oxygen atoms in Ca(OH)2 -/
def num_O : ℕ := 2

/-- The number of hydrogen atoms in Ca(OH)2 -/
def num_H : ℕ := 2

/-- The molecular weight of Ca(OH)2 in g/mol -/
def molecular_weight_CaOH2 : ℝ :=
  num_Ca * atomic_weight_Ca + num_O * atomic_weight_O + num_H * atomic_weight_H

theorem molecular_weight_CaOH2_is_74_10 :
  molecular_weight_CaOH2 = 74.10 := by sorry

end molecular_weight_CaOH2_is_74_10_l478_47839


namespace row_1007_sum_equals_2013_squared_l478_47877

/-- The sum of numbers in the nth row of the given pattern -/
def row_sum (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- The theorem stating that the 1007th row sum equals 2013² -/
theorem row_1007_sum_equals_2013_squared :
  row_sum 1007 = 2013 ^ 2 := by sorry

end row_1007_sum_equals_2013_squared_l478_47877


namespace calculate_expression_l478_47897

theorem calculate_expression : 
  Real.sqrt 5 * (Real.sqrt 10 + 2) - 1 / (Real.sqrt 5 - 2) - Real.sqrt (1/2) = 
  (9 * Real.sqrt 2) / 2 + Real.sqrt 5 - 2 := by sorry

end calculate_expression_l478_47897


namespace arithmetic_sequence_length_l478_47809

/-- Given an arithmetic sequence starting at 200, ending at 0, with a common difference of -5,
    the number of terms in the sequence is 41. -/
theorem arithmetic_sequence_length : 
  let start : ℤ := 200
  let end_val : ℤ := 0
  let diff : ℤ := -5
  let n : ℤ := (start - end_val) / (-diff) + 1
  n = 41 := by sorry

end arithmetic_sequence_length_l478_47809


namespace parallel_transitivity_l478_47826

-- Define the type for planes
variable {Plane : Type}

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_transitivity 
  (α β γ : Plane) 
  (h1 : parallel γ α) 
  (h2 : parallel γ β) : 
  parallel α β :=
sorry

end parallel_transitivity_l478_47826


namespace problem_solution_l478_47852

-- Define the functions f and g
def f (a b x : ℝ) := |x - a| - |x + b|
def g (a b x : ℝ) := -x^2 - a*x - b

-- State the theorem
theorem problem_solution (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmax : ∀ x, f a b x ≤ 3) 
  (hf_max : ∃ x, f a b x = 3) : 
  (a + b = 3) ∧ 
  (∀ x ≥ a, g a b x < f a b x) → 
  (1/2 < a ∧ a < 3) := by
sorry

end problem_solution_l478_47852


namespace age_problem_l478_47859

/-- Given three people whose total present age is 90 years and whose ages were in the ratio 1:2:3 ten years ago, 
    the present age of the person who was in the middle of the ratio is 30 years. -/
theorem age_problem (a b c : ℕ) : 
  a + b + c = 90 →  -- Total present age is 90
  (a - 10) = (b - 10) / 2 →  -- Ratio condition for a and b
  (c - 10) = 3 * ((b - 10) / 2) →  -- Ratio condition for b and c
  b = 30 :=
by sorry

end age_problem_l478_47859


namespace sector_radius_l478_47865

/-- The radius of a circle given the area of a sector and its central angle -/
theorem sector_radius (area : ℝ) (angle : ℝ) (pi : ℝ) (h1 : area = 52.8) (h2 : angle = 42) 
  (h3 : pi = Real.pi) (h4 : area = (angle / 360) * pi * (radius ^ 2)) : radius = 12 :=
by
  sorry

end sector_radius_l478_47865


namespace function_supremum_m_range_l478_47805

/-- The supremum of the given function for positive real x and y is 25/4 -/
theorem function_supremum : 
  (∀ x y : ℝ, x > 0 → y > 0 → 
    (4*x^4 + 17*x^2*y + 4*y^2) / (x^4 + 2*x^2*y + y^2) ≤ 25/4) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    (4*x^4 + 17*x^2*y + 4*y^2) / (x^4 + 2*x^2*y + y^2) = 25/4) :=
by sorry

/-- The range of m for which the inequality always holds is (25, +∞) -/
theorem m_range (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 
    (4*x^4 + 17*x^2*y + 4*y^2) / (x^4 + 2*x^2*y + y^2) < m/4) ↔ 
  m > 25 :=
by sorry

end function_supremum_m_range_l478_47805


namespace certain_number_is_six_l478_47850

theorem certain_number_is_six (a b n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) 
  (h4 : a % n = 2) (h5 : b % n = 3) (h6 : (a - b) % n = 5) : n = 6 := by
  sorry

end certain_number_is_six_l478_47850


namespace littering_citations_l478_47898

/-- Represents the number of citations for each category --/
structure Citations where
  littering : ℕ
  offLeash : ℕ
  smoking : ℕ
  parking : ℕ
  camping : ℕ

/-- Conditions for the park warden's citations --/
def citationConditions (c : Citations) : Prop :=
  c.littering = c.offLeash ∧
  c.littering = c.smoking + 5 ∧
  c.parking = 5 * (c.littering + c.offLeash + c.smoking) ∧
  c.camping = 10 ∧
  c.littering + c.offLeash + c.smoking + c.parking + c.camping = 150

/-- Theorem stating that under the given conditions, the number of littering citations is 9 --/
theorem littering_citations (c : Citations) (h : citationConditions c) : c.littering = 9 := by
  sorry


end littering_citations_l478_47898


namespace inequalities_for_negative_numbers_l478_47899

theorem inequalities_for_negative_numbers (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧ (Real.sqrt (-a) > Real.sqrt (-b)) ∧ (abs a > -b) := by
  sorry

end inequalities_for_negative_numbers_l478_47899


namespace supplementary_angles_ratio_l478_47892

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary
  a / b = 5 / 4 →  -- The ratio of the angles is 5:4
  b = 80 :=  -- The smaller angle is 80°
by sorry

end supplementary_angles_ratio_l478_47892


namespace f_minus_five_l478_47889

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_minus_five (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 6)
  (h_odd : is_odd f)
  (h_f_minus_one : f (-1) = 1) :
  f (-5) = -1 := by
sorry

end f_minus_five_l478_47889


namespace courtyard_length_l478_47831

/-- Prove that the length of a rectangular courtyard is 70 meters -/
theorem courtyard_length (width : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) :
  width = 16.5 ∧ 
  num_stones = 231 ∧ 
  stone_length = 2.5 ∧ 
  stone_width = 2 →
  (num_stones * stone_length * stone_width) / width = 70 := by
sorry

end courtyard_length_l478_47831


namespace glazed_doughnut_cost_l478_47820

/-- Proves that the cost of each glazed doughnut is $1 given the conditions of the problem -/
theorem glazed_doughnut_cost :
  let total_students : ℕ := 25
  let chocolate_lovers : ℕ := 10
  let glazed_lovers : ℕ := 15
  let chocolate_cost : ℚ := 2
  let total_cost : ℚ := 35
  chocolate_lovers + glazed_lovers = total_students →
  chocolate_lovers * chocolate_cost + glazed_lovers * (total_cost - chocolate_lovers * chocolate_cost) / glazed_lovers = total_cost →
  (total_cost - chocolate_lovers * chocolate_cost) / glazed_lovers = 1 := by
sorry

end glazed_doughnut_cost_l478_47820


namespace sequence_product_l478_47849

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem sequence_product (a b m n : ℝ) :
  is_arithmetic_sequence (-9) a (-1) →
  is_geometric_sequence (-9) m b n (-1) →
  a * b = 15 := by
  sorry

end sequence_product_l478_47849


namespace median_sufficiency_for_top_half_l478_47857

theorem median_sufficiency_for_top_half (scores : Finset ℝ) (xiaofen_score : ℝ) :
  Finset.card scores = 12 →
  Finset.card (Finset.filter (λ x => x = xiaofen_score) scores) ≤ 1 →
  (∃ median : ℝ, Finset.card (Finset.filter (λ x => x ≤ median) scores) = 6 ∧
                 Finset.card (Finset.filter (λ x => x ≥ median) scores) = 6) →
  (xiaofen_score > median ↔ Finset.card (Finset.filter (λ x => x > xiaofen_score) scores) < 6) :=
by sorry

end median_sufficiency_for_top_half_l478_47857


namespace curves_with_property_P_l478_47863

-- Define the line equation
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y + 1 = 0

-- Define property P
def property_P (curve : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∃ A B : ℝ × ℝ, 
    curve A.1 A.2 ∧ curve B.1 B.2 ∧
    line_equation k A.1 A.2 ∧ line_equation k B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = k^2

-- Define the three curves
def curve1 (x y : ℝ) : Prop := y = -abs x

def curve2 (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

def curve3 (x y : ℝ) : Prop := y = (x + 1)^2

-- Theorem statement
theorem curves_with_property_P :
  ¬(property_P curve1) ∧ 
  property_P curve2 ∧ 
  property_P curve3 :=
sorry

end curves_with_property_P_l478_47863


namespace stream_speed_l478_47888

/-- Given a man's downstream and upstream speeds, calculate the speed of the stream --/
theorem stream_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 10)
  (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 1 := by
  sorry

end stream_speed_l478_47888


namespace g_difference_l478_47853

/-- The function g(n) as defined in the problem -/
def g (n : ℤ) : ℚ := (1 / 4 : ℚ) * n^2 * (n + 1) * (n + 3) + 1

/-- Theorem stating the difference between g(m) and g(m-1) -/
theorem g_difference (m : ℤ) : g m - g (m - 1) = (3 / 4 : ℚ) * m^2 * (m + 5 / 3) := by
  sorry

end g_difference_l478_47853


namespace initial_water_fraction_in_larger_jar_l478_47864

theorem initial_water_fraction_in_larger_jar 
  (small_capacity large_capacity : ℝ) 
  (h1 : small_capacity > 0) 
  (h2 : large_capacity > 0) 
  (h3 : small_capacity ≠ large_capacity) :
  let water_amount := (1/5) * small_capacity
  let initial_fraction := water_amount / large_capacity
  let combined_fraction := (water_amount + water_amount) / large_capacity
  (combined_fraction = 0.4) → (initial_fraction = 1/10) := by
  sorry

end initial_water_fraction_in_larger_jar_l478_47864


namespace profit_distribution_l478_47878

/-- Represents the profit distribution problem -/
theorem profit_distribution 
  (john_investment : ℕ) (john_months : ℕ)
  (rose_investment : ℕ) (rose_months : ℕ)
  (tom_investment : ℕ) (tom_months : ℕ)
  (profit_share_diff : ℕ) :
  john_investment = 18000 →
  john_months = 12 →
  rose_investment = 12000 →
  rose_months = 9 →
  tom_investment = 9000 →
  tom_months = 8 →
  profit_share_diff = 370 →
  ∃ (total_profit : ℕ),
    total_profit = 4070 ∧
    (rose_investment * rose_months * total_profit) / 
      (john_investment * john_months + rose_investment * rose_months + tom_investment * tom_months) -
    (tom_investment * tom_months * total_profit) / 
      (john_investment * john_months + rose_investment * rose_months + tom_investment * tom_months) = 
    profit_share_diff :=
by sorry

end profit_distribution_l478_47878


namespace composition_of_even_is_even_l478_47810

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem composition_of_even_is_even (f : ℝ → ℝ) (hf : IsEven f) : IsEven (f ∘ f) := by
  sorry

end composition_of_even_is_even_l478_47810


namespace shifted_parabola_l478_47873

/-- The equation of a parabola shifted 1 unit to the left -/
theorem shifted_parabola (x y : ℝ) :
  (y = -(x^2) + 1) → 
  (∃ x' y', y' = -(x'^2) + 1 ∧ x' = x + 1) →
  y = -((x + 1)^2) + 1 := by
sorry

end shifted_parabola_l478_47873


namespace general_rule_l478_47818

theorem general_rule (n : ℕ+) :
  (n + 1 : ℚ) / n + (n + 1 : ℚ) = (n + 2 : ℚ) + 1 / n := by sorry

end general_rule_l478_47818


namespace largest_number_from_hcf_lcm_factors_l478_47844

theorem largest_number_from_hcf_lcm_factors (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 40)
  (lcm_eq : Nat.lcm a b = 40 * 11 * 12) :
  max a b = 480 := by
sorry

end largest_number_from_hcf_lcm_factors_l478_47844


namespace women_fair_hair_percentage_l478_47884

-- Define the percentage of fair-haired employees who are women
def fair_haired_women_percentage : ℝ := 0.40

-- Define the percentage of employees who have fair hair
def fair_haired_percentage : ℝ := 0.70

-- Theorem statement
theorem women_fair_hair_percentage :
  fair_haired_women_percentage * fair_haired_percentage = 0.28 := by
  sorry

end women_fair_hair_percentage_l478_47884


namespace rectangular_plot_perimeter_l478_47815

/-- A rectangular plot with given conditions --/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencing_rate : ℝ
  total_fencing_cost : ℝ
  length_width_relation : length = width + 10
  fencing_cost_relation : fencing_rate * (2 * (length + width)) = total_fencing_cost

/-- The perimeter of the rectangular plot is 180 meters --/
theorem rectangular_plot_perimeter (plot : RectangularPlot) 
  (h_rate : plot.fencing_rate = 6.5)
  (h_cost : plot.total_fencing_cost = 1170) : 
  2 * (plot.length + plot.width) = 180 := by
  sorry


end rectangular_plot_perimeter_l478_47815


namespace geometric_sequence_sum_l478_47816

theorem geometric_sequence_sum (a₁ : ℝ) (r : ℝ) :
  a₁ = 3125 →
  r = 1/5 →
  (a₁ * r^5 = 1) →
  (a₁ * r^3 + a₁ * r^4 = 30) :=
by
  sorry

end geometric_sequence_sum_l478_47816


namespace binomial_fraction_value_l478_47807

theorem binomial_fraction_value : 
  (Nat.choose 1 2023 * 3^2023) / Nat.choose 4046 2023 = 0 := by
  sorry

end binomial_fraction_value_l478_47807


namespace quadratic_equation_solution_l478_47894

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 1 ∧ x₂ = 3 ∧ 
  (x₁^2 - 4*x₁ + 3 = 0) ∧ 
  (x₂^2 - 4*x₂ + 3 = 0) ∧
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 → x = x₁ ∨ x = x₂) :=
by sorry

end quadratic_equation_solution_l478_47894


namespace circle_center_and_radius_l478_47887

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop := x^2 + y^2 + 4*x = 0

/-- The center of a circle -/
def Center : ℝ × ℝ := (-2, 0)

/-- The radius of a circle -/
def Radius : ℝ := 2

/-- Theorem: The circle described by x^2 + y^2 + 4x = 0 has center (-2, 0) and radius 2 -/
theorem circle_center_and_radius :
  (∀ x y : ℝ, CircleEquation x y ↔ (x + 2)^2 + y^2 = 4) ∧
  Center = (-2, 0) ∧
  Radius = 2 := by sorry

end circle_center_and_radius_l478_47887


namespace rice_cost_problem_l478_47851

/-- Proves that the cost of the first type of rice is 16 rupees per kg -/
theorem rice_cost_problem (rice1_weight : ℝ) (rice2_weight : ℝ) (rice2_cost : ℝ) (avg_cost : ℝ) 
  (h1 : rice1_weight = 8)
  (h2 : rice2_weight = 4)
  (h3 : rice2_cost = 22)
  (h4 : avg_cost = 18)
  (h5 : (rice1_weight * rice1_cost + rice2_weight * rice2_cost) / (rice1_weight + rice2_weight) = avg_cost) :
  rice1_cost = 16 :=
by
  sorry

end rice_cost_problem_l478_47851


namespace five_distinct_roots_l478_47825

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

-- Define the composition f(f(x))
def f_comp_f (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

-- State the theorem
theorem five_distinct_roots (c : ℝ) : 
  (∃! (roots : Finset ℝ), roots.card = 5 ∧ ∀ x ∈ roots, f_comp_f c x = 0) ↔ (c = 0 ∨ c = 3) :=
sorry

end five_distinct_roots_l478_47825


namespace rectangular_field_width_l478_47895

/-- Proves that for a rectangular field with length 7/5 of its width and perimeter 336 meters, the width is 70 meters -/
theorem rectangular_field_width (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = (7/5) * width →
  perimeter = 336 →
  perimeter = 2 * length + 2 * width →
  width = 70 :=
by sorry

end rectangular_field_width_l478_47895


namespace range_of_f_range_of_g_l478_47855

-- Define the functions f and g
def f (a x : ℝ) : ℝ := x^2 + 4*a*x + 2*a + 6
def g (a : ℝ) : ℝ := 2 - a * |a + 3|

-- Theorem for part (1)
theorem range_of_f (a : ℝ) :
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f a x = y) ∧ (∀ x : ℝ, f a x ≥ 0) ↔ a = -1 ∨ a = 3/2 :=
sorry

-- Theorem for part (2)
theorem range_of_g :
  (∀ a x : ℝ, f a x ≥ 0) →
  ∀ y : ℝ, -19/4 ≤ y ∧ y ≤ 4 ↔ ∃ a : ℝ, g a = y :=
sorry

end range_of_f_range_of_g_l478_47855


namespace quadratic_inequality_l478_47802

/-- The quadratic function f(x) = -x^2 + 2x + 3 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

/-- y₁ is the value of f at x = -2 -/
def y₁ : ℝ := f (-2)

/-- y₂ is the value of f at x = 2 -/
def y₂ : ℝ := f 2

/-- y₃ is the value of f at x = -4 -/
def y₃ : ℝ := f (-4)

/-- Theorem: For the quadratic function f(x) = -x^2 + 2x + 3,
    if f(-2) = y₁, f(2) = y₂, and f(-4) = y₃, then y₂ > y₁ > y₃ -/
theorem quadratic_inequality : y₂ > y₁ ∧ y₁ > y₃ := by
  sorry

end quadratic_inequality_l478_47802


namespace quadratic_unique_solution_l478_47823

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 12 * x + c = 0) →  -- exactly one solution
  (a + c = 14) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 7 - Real.sqrt 13 ∧ c = 7 + Real.sqrt 13) :=
by sorry

end quadratic_unique_solution_l478_47823


namespace solutions_of_equation_l478_47860

theorem solutions_of_equation (x : ℝ) : 
  (3 * x^2 = Real.sqrt 3 * x) ↔ (x = 0 ∨ x = Real.sqrt 3 / 3) := by
sorry

end solutions_of_equation_l478_47860


namespace scout_troop_profit_l478_47879

/-- The profit of a scout troop selling candy bars -/
theorem scout_troop_profit :
  let num_bars : ℕ := 1200
  let buy_price : ℚ := 1 / 3  -- price per bar when buying
  let sell_price : ℚ := 3 / 5 -- price per bar when selling
  let cost : ℚ := num_bars * buy_price
  let revenue : ℚ := num_bars * sell_price
  let profit : ℚ := revenue - cost
  profit = 320 := by sorry

end scout_troop_profit_l478_47879


namespace exists_society_with_subgroup_l478_47829

/-- Definition of a society with n girls and m boys -/
structure Society :=
  (n : ℕ) -- number of girls
  (m : ℕ) -- number of boys

/-- Definition of a relationship between boys and girls in a society -/
def Knows (s : Society) := 
  Fin s.m → Fin s.n → Prop

/-- Definition of a subgroup with the required property -/
def HasSubgroup (s : Society) (knows : Knows s) : Prop :=
  ∃ (girls : Fin 5 → Fin s.n) (boys : Fin 5 → Fin s.m),
    (∀ i j, knows (boys i) (girls j)) ∨ 
    (∀ i j, ¬knows (boys i) (girls j))

/-- Main theorem: Existence of n₀ and m₀ satisfying the property -/
theorem exists_society_with_subgroup :
  ∃ (n₀ m₀ : ℕ), ∀ (s : Society),
    s.n = n₀ → s.m = m₀ → 
    ∀ (knows : Knows s), HasSubgroup s knows :=
sorry

end exists_society_with_subgroup_l478_47829


namespace final_state_values_l478_47885

/-- Represents the state of variables a, b, and c -/
structure State :=
  (a : Int) (b : Int) (c : Int)

/-- Applies the sequence of operations to the initial state -/
def applyOperations (initial : State) : State :=
  let step1 := State.mk initial.b initial.b initial.c
  let step2 := State.mk step1.a step1.c step1.b
  State.mk step2.a step2.b step2.a

/-- The theorem stating the final values after operations -/
theorem final_state_values (initial : State := State.mk 3 (-5) 8) :
  let final := applyOperations initial
  final.a = -5 ∧ final.b = 8 ∧ final.c = -5 := by
  sorry

end final_state_values_l478_47885


namespace sum_is_positive_difference_is_negative_four_l478_47856

variables (a b : ℝ)

def A : ℝ := a^2 - 2*a*b + b^2
def B : ℝ := a^2 + 2*a*b + b^2

theorem sum_is_positive (h : a ≠ b) : A a b + B a b > 0 := by
  sorry

theorem difference_is_negative_four (h : a * b = 1) : A a b - B a b = -4 := by
  sorry

end sum_is_positive_difference_is_negative_four_l478_47856


namespace abc_inequality_l478_47828

theorem abc_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end abc_inequality_l478_47828


namespace tan_product_equals_fifteen_l478_47862

theorem tan_product_equals_fifteen : 
  15 * Real.tan (44 * π / 180) * Real.tan (45 * π / 180) * Real.tan (46 * π / 180) = 15 := by
  sorry

end tan_product_equals_fifteen_l478_47862


namespace ratio_of_balls_l478_47843

def red_balls : ℕ := 16
def white_balls : ℕ := 20

theorem ratio_of_balls : 
  (red_balls : ℚ) / white_balls = 4 / 5 := by
  sorry

end ratio_of_balls_l478_47843


namespace remainder_fraction_l478_47854

theorem remainder_fraction (x : ℝ) (h : x = 62.5) : 
  ((x + 5) * 2 / 5 - 5) / 44 = 1 / 2 := by sorry

end remainder_fraction_l478_47854


namespace system_unique_solution_l478_47881

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  Real.sqrt ((x - 6)^2 + (y - 13)^2) + Real.sqrt ((x - 18)^2 + (y - 4)^2) = 15 ∧
  (x - 2*a)^2 + (y - 4*a)^2 = 1/4

-- Define the set of a values for which the system has a unique solution
def unique_solution_set : Set ℝ :=
  {a | a = 145/44 ∨ a = 135/44 ∨ (63/20 < a ∧ a < 13/4)}

-- Theorem statement
theorem system_unique_solution (a : ℝ) :
  (∃! p : ℝ × ℝ, system p.1 p.2 a) ↔ a ∈ unique_solution_set :=
sorry

end system_unique_solution_l478_47881


namespace factorization_proof_l478_47836

theorem factorization_proof (b : ℝ) : 65 * b^2 + 195 * b = 65 * b * (b + 3) := by
  sorry

end factorization_proof_l478_47836


namespace f_prime_at_two_l478_47814

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_prime_at_two (a b : ℝ) :
  (f a b 1 = -2) →
  ((deriv (f a b)) 1 = 0) →
  ((deriv (f a b)) 2 = -1/2) :=
sorry

end f_prime_at_two_l478_47814


namespace exists_polynomial_for_cosine_multiple_l478_47841

-- Define Chebyshev polynomials of the first kind
def chebyshev (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => λ _ => 1
  | 1 => λ x => x
  | n + 2 => λ x => 2 * x * chebyshev (n + 1) x - chebyshev n x

-- State the theorem
theorem exists_polynomial_for_cosine_multiple (n : ℕ) (hn : n > 0) :
  ∃ (p : ℝ → ℝ), ∀ x, p (2 * Real.cos x) = 2 * Real.cos (n * x) :=
by
  -- The proof would go here
  sorry

end exists_polynomial_for_cosine_multiple_l478_47841


namespace sum_in_base6_l478_47882

/-- Converts a number from base 6 to base 10 -/
def base6To10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a number from base 10 to base 6 -/
def base10To6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The sum of the given numbers in base 6 equals 1214 in base 6 -/
theorem sum_in_base6 :
  let n1 := [5, 5, 5]
  let n2 := [5, 5]
  let n3 := [5]
  let n4 := [1, 1, 1]
  let sum := base6To10 n1 + base6To10 n2 + base6To10 n3 + base6To10 n4
  base10To6 sum = [1, 2, 1, 4] :=
by sorry

end sum_in_base6_l478_47882


namespace expression_values_l478_47819

theorem expression_values (x y : ℝ) (h1 : x + y = 2) (h2 : y > 0) (h3 : x ≠ 0) :
  (1 / |x| + |x| / (y + 2) = 3/4) ∨ (1 / |x| + |x| / (y + 2) = 5/4) :=
by sorry

end expression_values_l478_47819


namespace smallest_d_for_10000_l478_47837

theorem smallest_d_for_10000 : 
  ∃ (p q r : Nat), 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    (∀ d : Nat, d > 0 → 
      (∃ (p' q' r' : Nat), 
        Prime p' ∧ Prime q' ∧ Prime r' ∧ 
        p' ≠ q' ∧ p' ≠ r' ∧ q' ≠ r' ∧
        10000 * d = (p' * q' * r')^2) → 
      d ≥ 53361) ∧
    10000 * 53361 = (p * q * r)^2 :=
sorry

end smallest_d_for_10000_l478_47837
