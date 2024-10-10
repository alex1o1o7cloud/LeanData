import Mathlib

namespace a_4_equals_zero_l3027_302784

def sequence_a (n : ℕ) : ℤ := n^2 - 2*n - 8

theorem a_4_equals_zero : sequence_a 4 = 0 := by
  sorry

end a_4_equals_zero_l3027_302784


namespace matrix_power_identity_l3027_302751

/-- Given a 2x2 matrix B, prove that B^4 = 51*B + 52*I --/
theorem matrix_power_identity (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B = !![1, 2; 3, 1]) : 
  B^4 = 51 • B + 52 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end matrix_power_identity_l3027_302751


namespace parallel_line_distance_in_circle_l3027_302775

/-- Given a circle intersected by four equally spaced parallel lines creating chords of lengths 44, 44, 40, and 40, the distance between two adjacent parallel lines is 8/√23. -/
theorem parallel_line_distance_in_circle : ∀ (r : ℝ) (d : ℝ),
  (44 + (1/4) * d^2 = r^2) →
  (40 + (27/16) * d^2 = r^2) →
  d = 8 / Real.sqrt 23 := by
sorry

end parallel_line_distance_in_circle_l3027_302775


namespace smallest_lcm_with_gcd_five_l3027_302734

theorem smallest_lcm_with_gcd_five (k l : ℕ) : 
  k ≥ 10000 ∧ k < 100000 ∧ 
  l ≥ 10000 ∧ l < 100000 ∧ 
  Nat.gcd k l = 5 → 
  Nat.lcm k l ≥ 20010000 := by
sorry

end smallest_lcm_with_gcd_five_l3027_302734


namespace angle_relation_l3027_302703

theorem angle_relation (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.tan (α - β) = 1/2) (h4 : Real.tan β = -1/7) :
  2*α - β = -3*π/4 := by sorry

end angle_relation_l3027_302703


namespace steve_final_marbles_l3027_302716

/-- Represents the initial and final marble counts for each person --/
structure MarbleCounts where
  steve_initial : ℕ
  sam_initial : ℕ
  sally_initial : ℕ
  sarah_initial : ℕ
  steve_final : ℕ

/-- Defines the marble distribution scenario --/
def marble_distribution (counts : MarbleCounts) : Prop :=
  counts.sam_initial = 2 * counts.steve_initial ∧
  counts.sally_initial = counts.sam_initial - 5 ∧
  counts.sarah_initial = counts.steve_initial + 3 ∧
  counts.sam_initial - (3 + 3 + 4) = 6 ∧
  counts.steve_final = counts.steve_initial + 3

theorem steve_final_marbles (counts : MarbleCounts) :
  marble_distribution counts → counts.steve_final = 11 := by
  sorry

end steve_final_marbles_l3027_302716


namespace thermometer_to_bottle_ratio_l3027_302771

/-- Proves that the ratio of thermometers sold to hot-water bottles sold is 7:1 given the problem conditions -/
theorem thermometer_to_bottle_ratio :
  ∀ (T H : ℕ), 
    (2 * T + 6 * H = 1200) →  -- Total sales equation
    (H = 60) →                -- Number of hot-water bottles sold
    (T : ℚ) / H = 7 / 1 :=    -- Ratio of thermometers to hot-water bottles
by
  sorry

#check thermometer_to_bottle_ratio

end thermometer_to_bottle_ratio_l3027_302771


namespace percentage_theorem_l3027_302738

theorem percentage_theorem (x y : ℝ) (P : ℝ) 
  (h1 : 0.6 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.5 * x) : 
  P = 20 := by
sorry

end percentage_theorem_l3027_302738


namespace cricket_team_average_age_l3027_302700

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age_diff : ℕ) 
  (remaining_players_age_diff : ℕ) : 
  team_size = 11 → 
  captain_age = 28 → 
  wicket_keeper_age_diff = 3 → 
  remaining_players_age_diff = 1 → 
  ∃ (team_avg_age : ℚ), 
    team_avg_age = 25 ∧ 
    team_size * team_avg_age = 
      captain_age + (captain_age + wicket_keeper_age_diff) + 
      (team_size - 2) * (team_avg_age - remaining_players_age_diff) :=
by sorry

end cricket_team_average_age_l3027_302700


namespace beyonce_songs_count_l3027_302753

/-- The number of songs Beyonce has released in total -/
def total_songs : ℕ :=
  let singles := 12
  let albums := 4
  let songs_first_cd := 18
  let songs_second_cd := 14
  let songs_per_album := songs_first_cd + songs_second_cd
  singles + albums * songs_per_album

/-- Theorem stating that Beyonce has released 140 songs in total -/
theorem beyonce_songs_count : total_songs = 140 := by
  sorry

end beyonce_songs_count_l3027_302753


namespace batsman_average_after_17th_innings_l3027_302702

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the new average after an additional innings -/
def newAverage (bp : BatsmanPerformance) (newScore : ℕ) : ℚ :=
  (bp.totalScore + newScore) / (bp.innings + 1)

theorem batsman_average_after_17th_innings 
  (bp : BatsmanPerformance) 
  (h1 : bp.innings = 16) 
  (h2 : newAverage bp 85 = bp.average + 3) :
  newAverage bp 85 = 37 := by
sorry

end batsman_average_after_17th_innings_l3027_302702


namespace james_course_cost_l3027_302796

/-- Represents the cost per unit for James's community college courses. -/
def cost_per_unit (units_per_semester : ℕ) (total_cost : ℕ) (num_semesters : ℕ) : ℚ :=
  total_cost / (units_per_semester * num_semesters)

/-- Theorem stating that the cost per unit is $50 given the conditions. -/
theorem james_course_cost : 
  cost_per_unit 20 2000 2 = 50 := by
  sorry

end james_course_cost_l3027_302796


namespace function_inequality_l3027_302779

-- Define a real-valued function f on ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State that f' is the derivative of f
variable (hf' : ∀ x, HasDerivAt f (f' x) x)

-- State that f'(x) < f(x) for all x ∈ ℝ
variable (h : ∀ x, f' x < f x)

-- Theorem statement
theorem function_inequality (f f' : ℝ → ℝ) (hf' : ∀ x, HasDerivAt f (f' x) x) (h : ∀ x, f' x < f x) :
  f 2 < Real.exp 2 * f 0 ∧ f 2001 < Real.exp 2001 * f 0 := by
  sorry

end function_inequality_l3027_302779


namespace three_cakes_cooking_time_l3027_302709

/-- Represents the cooking process for cakes -/
structure CookingProcess where
  pot_capacity : ℕ
  cooking_time_per_cake : ℕ
  num_cakes : ℕ

/-- Calculates the minimum time needed to cook all cakes -/
def min_cooking_time (process : CookingProcess) : ℕ :=
  sorry

/-- Theorem stating the minimum time to cook 3 cakes -/
theorem three_cakes_cooking_time :
  ∀ (process : CookingProcess),
    process.pot_capacity = 2 →
    process.cooking_time_per_cake = 5 →
    process.num_cakes = 3 →
    min_cooking_time process = 15 :=
by sorry

end three_cakes_cooking_time_l3027_302709


namespace third_consecutive_odd_integer_l3027_302778

/-- Given three consecutive odd integers where 3 times the first is 3 more than twice the third, 
    prove that the third integer is 15. -/
theorem third_consecutive_odd_integer (x : ℤ) : 
  (∃ y z : ℤ, 
    y = x + 2 ∧ 
    z = x + 4 ∧ 
    Odd x ∧ 
    Odd y ∧ 
    Odd z ∧ 
    3 * x = 2 * z + 3) →
  x + 4 = 15 := by
sorry

end third_consecutive_odd_integer_l3027_302778


namespace number_of_balls_in_box_l3027_302791

theorem number_of_balls_in_box : ∃ x : ℕ, x > 20 ∧ x < 30 ∧ (x - 20 = 30 - x) ∧ x = 25 := by
  sorry

end number_of_balls_in_box_l3027_302791


namespace simplified_fraction_ratio_l3027_302714

theorem simplified_fraction_ratio (k : ℤ) : 
  ∃ (a b : ℤ), (6 * k + 18) / 3 = a * k + b ∧ a / b = 1 / 3 := by
  sorry

end simplified_fraction_ratio_l3027_302714


namespace no_integer_solutions_l3027_302732

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 4*m^2 + 3*m = 8*n^3 + 12*n^2 + 6*n + 1 := by
  sorry

end no_integer_solutions_l3027_302732


namespace pie_chart_highlights_part_whole_l3027_302757

/-- Enumeration of statistical graph types --/
inductive StatisticalGraph
  | BarGraph
  | PieChart
  | LineGraph
  | FrequencyDistributionHistogram

/-- Function to determine if a graph type highlights part-whole relationships --/
def highlights_part_whole_relationship (graph : StatisticalGraph) : Prop :=
  match graph with
  | StatisticalGraph.PieChart => True
  | _ => False

/-- Theorem stating that the Pie chart is the graph that highlights part-whole relationships --/
theorem pie_chart_highlights_part_whole :
  ∀ (graph : StatisticalGraph),
    highlights_part_whole_relationship graph ↔ graph = StatisticalGraph.PieChart :=
by sorry

end pie_chart_highlights_part_whole_l3027_302757


namespace means_inequality_l3027_302733

theorem means_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  Real.sqrt (a * b) > Real.rpow (a * b * c) (1/3) ∧
  Real.rpow (a * b * c) (1/3) > (2 * b * c) / (b + c) := by
sorry

end means_inequality_l3027_302733


namespace sum_of_a_values_l3027_302797

theorem sum_of_a_values : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, (∃! (sol : Finset ℤ), 
    (∀ x ∈ sol, (4 * x - a ≥ 1 ∧ (x + 13) / 2 ≥ x + 2)) ∧ 
    sol.card = 6)) ∧ 
  S.sum id = 54 := by
  sorry

end sum_of_a_values_l3027_302797


namespace floor_ceiling_calculation_l3027_302740

theorem floor_ceiling_calculation : 
  ⌊(15 : ℝ) / 8 * (-34 : ℝ) / 4⌋ - ⌈(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌉ = 0 := by
  sorry

end floor_ceiling_calculation_l3027_302740


namespace least_common_period_is_30_l3027_302725

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least positive period of a function -/
def is_least_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_period f p ∧ ∀ q : ℝ, 0 < q ∧ q < p → ¬ is_period f q

/-- The main theorem -/
theorem least_common_period_is_30 :
  ∃ p : ℝ, p = 30 ∧
    (∀ f : ℝ → ℝ, satisfies_condition f → is_least_positive_period f p) ∧
    (∀ q : ℝ, q ≠ p →
      ∃ f : ℝ → ℝ, satisfies_condition f ∧ ¬ is_least_positive_period f q) :=
sorry

end least_common_period_is_30_l3027_302725


namespace percentage_of_300_is_66_l3027_302706

theorem percentage_of_300_is_66 : 
  (66 : ℝ) / 300 * 100 = 22 := by sorry

end percentage_of_300_is_66_l3027_302706


namespace gcd_lcm_product_24_36_l3027_302708

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end gcd_lcm_product_24_36_l3027_302708


namespace remaining_money_l3027_302772

-- Define the initial amount, amount spent on sweets, and amount given to each friend
def initial_amount : ℚ := 20.10
def sweets_cost : ℚ := 1.05
def friend_gift : ℚ := 1.00
def num_friends : ℕ := 2

-- Define the theorem
theorem remaining_money :
  initial_amount - sweets_cost - (friend_gift * num_friends) = 17.05 := by
  sorry

end remaining_money_l3027_302772


namespace rose_bundle_price_l3027_302761

theorem rose_bundle_price (rose_price : ℕ) (total_roses : ℕ) (num_bundles : ℕ) 
  (h1 : rose_price = 500)
  (h2 : total_roses = 200)
  (h3 : num_bundles = 25) :
  (rose_price * total_roses) / num_bundles = 4000 :=
by sorry

end rose_bundle_price_l3027_302761


namespace geometric_sequence_ratio_l3027_302799

/-- Given a geometric sequence {a_n} with positive common ratio q,
    if a_3 · a_9 = (a_5)^2, then q = 1. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  q > 0 →
  (∀ n, a (n + 1) = a n * q) →
  a 3 * a 9 = (a 5)^2 →
  q = 1 := by
sorry

end geometric_sequence_ratio_l3027_302799


namespace triangle_area_with_base_12_height_15_l3027_302724

/-- The area of a triangle with base 12 and height 15 is 90 -/
theorem triangle_area_with_base_12_height_15 :
  let base : ℝ := 12
  let height : ℝ := 15
  let area : ℝ := (1 / 2) * base * height
  area = 90 := by
  sorry

end triangle_area_with_base_12_height_15_l3027_302724


namespace proportion_solution_l3027_302787

theorem proportion_solution (x : ℝ) : (0.75 / x = 5 / 11) → x = 1.65 := by
  sorry

end proportion_solution_l3027_302787


namespace meghan_coffee_order_cost_l3027_302745

/-- Represents the cost of a coffee order with given quantities and prices --/
def coffee_order_cost (drip_coffee_price : ℚ) (drip_coffee_qty : ℕ)
                      (espresso_price : ℚ) (espresso_qty : ℕ)
                      (latte_price : ℚ) (latte_qty : ℕ)
                      (vanilla_syrup_price : ℚ) (vanilla_syrup_qty : ℕ)
                      (cold_brew_price : ℚ) (cold_brew_qty : ℕ)
                      (cappuccino_price : ℚ) (cappuccino_qty : ℕ) : ℚ :=
  drip_coffee_price * drip_coffee_qty +
  espresso_price * espresso_qty +
  latte_price * latte_qty +
  vanilla_syrup_price * vanilla_syrup_qty +
  cold_brew_price * cold_brew_qty +
  cappuccino_price * cappuccino_qty

/-- The total cost of Meghan's coffee order is $25.00 --/
theorem meghan_coffee_order_cost :
  coffee_order_cost (25/10) 2
                    (35/10) 1
                    4 2
                    (1/2) 1
                    (25/10) 2
                    (35/10) 1 = 25 := by
  sorry

end meghan_coffee_order_cost_l3027_302745


namespace cat_and_mouse_positions_after_2023_seconds_l3027_302766

/-- Represents the number of positions in the cat's path -/
def cat_path_length : ℕ := 8

/-- Represents the number of positions in the mouse's path -/
def mouse_path_length : ℕ := 12

/-- Calculates the position of an object after a given number of seconds,
    given the length of its path -/
def position_after_time (path_length : ℕ) (time : ℕ) : ℕ :=
  time % path_length

/-- The main theorem stating the positions of the cat and mouse after 2023 seconds -/
theorem cat_and_mouse_positions_after_2023_seconds :
  position_after_time cat_path_length 2023 = 7 ∧
  position_after_time mouse_path_length 2023 = 7 := by
  sorry

end cat_and_mouse_positions_after_2023_seconds_l3027_302766


namespace four_letter_words_with_a_l3027_302798

theorem four_letter_words_with_a (n : ℕ) (total_letters : ℕ) (letters_without_a : ℕ) : 
  n = 4 → 
  total_letters = 5 → 
  letters_without_a = 4 → 
  (total_letters ^ n) - (letters_without_a ^ n) = 369 :=
by sorry

end four_letter_words_with_a_l3027_302798


namespace cans_difference_l3027_302715

/-- The number of cans Sarah collected yesterday -/
def sarah_yesterday : ℕ := 50

/-- The number of additional cans Lara collected compared to Sarah yesterday -/
def lara_extra_yesterday : ℕ := 30

/-- The number of cans Sarah collected today -/
def sarah_today : ℕ := 40

/-- The number of cans Lara collected today -/
def lara_today : ℕ := 70

/-- Theorem stating the difference in total cans collected between yesterday and today -/
theorem cans_difference : 
  (sarah_yesterday + (sarah_yesterday + lara_extra_yesterday)) - (sarah_today + lara_today) = 20 :=
by sorry

end cans_difference_l3027_302715


namespace linear_function_monotonicity_and_inequality_l3027_302701

variables (a b c : ℝ)

def f (x : ℝ) := a * x + b

theorem linear_function_monotonicity_and_inequality (a b c : ℝ) :
  (a > 0 → Monotone (f a b)) ∧
  (b^2 - 4*a*c < 0 → a^3 + a*b + c ≠ 0) :=
sorry

end linear_function_monotonicity_and_inequality_l3027_302701


namespace sin_230_minus_sqrt3_tan_170_l3027_302758

theorem sin_230_minus_sqrt3_tan_170 : 
  Real.sin (230 * π / 180) * (1 - Real.sqrt 3 * Real.tan (170 * π / 180)) = -1 := by
  sorry

end sin_230_minus_sqrt3_tan_170_l3027_302758


namespace ab_value_l3027_302711

/-- Given that p and q are integers satisfying the equation and other conditions, prove that ab = 10^324 -/
theorem ab_value (p q : ℤ) (a b : ℝ) 
  (hp : p = Real.sqrt (Real.log a))
  (hq : q = Real.sqrt (Real.log b))
  (ha : a = 10^(p^2))
  (hb : b = 10^(q^2))
  (heq : 2*p + 2*q + (Real.log a)/2 + (Real.log b)/2 + p * q = 200) :
  a * b = 10^324 := by
  sorry

end ab_value_l3027_302711


namespace ellipse_focal_distance_l3027_302750

/-- For an ellipse with equation x²/m + y²/4 = 1 and focal distance 2, m = 5 -/
theorem ellipse_focal_distance (m : ℝ) : 
  (∀ x y : ℝ, x^2/m + y^2/4 = 1) →  -- Ellipse equation
  2 = 2 * 1 →                      -- Focal distance is 2
  m = 5 := by
sorry

end ellipse_focal_distance_l3027_302750


namespace right_triangle_hypotenuse_segment_ratio_l3027_302769

theorem right_triangle_hypotenuse_segment_ratio 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_ratio : a / b = 3 / 4) 
  (d : ℝ) 
  (h_d : d * c = a * b) : 
  (c - d) / d = 3 / 4 := by
sorry

end right_triangle_hypotenuse_segment_ratio_l3027_302769


namespace flowchart_connection_is_flow_line_l3027_302705

-- Define the basic elements of a flowchart
inductive FlowchartElement
  | ConnectionPoint
  | DecisionBox
  | FlowLine
  | ProcessBox

-- Define a property for connecting steps in a flowchart
def connects_steps (element : FlowchartElement) : Prop :=
  element = FlowchartElement.FlowLine

-- Theorem statement
theorem flowchart_connection_is_flow_line :
  ∃ (element : FlowchartElement), connects_steps element :=
sorry

end flowchart_connection_is_flow_line_l3027_302705


namespace smallest_class_size_l3027_302723

/-- Represents a class of students who took a test -/
structure TestClass where
  n : ℕ                -- number of students
  scores : Fin n → ℕ   -- scores of each student
  test_max : ℕ         -- maximum possible score on the test

/-- Conditions for our specific test class -/
def SatisfiesConditions (c : TestClass) : Prop :=
  c.test_max = 100 ∧
  (∃ (i₁ i₂ i₃ i₄ : Fin c.n), i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₃ ≠ i₄ ∧
    c.scores i₁ = 90 ∧ c.scores i₂ = 90 ∧ c.scores i₃ = 90 ∧ c.scores i₄ = 90) ∧
  (∀ i, c.scores i ≥ 70) ∧
  (Finset.sum (Finset.univ : Finset (Fin c.n)) c.scores / c.n = 80)

/-- The main theorem stating that the smallest possible class size is 8 -/
theorem smallest_class_size (c : TestClass) (h : SatisfiesConditions c) :
  c.n ≥ 8 ∧ ∃ (c' : TestClass), SatisfiesConditions c' ∧ c'.n = 8 := by
  sorry

end smallest_class_size_l3027_302723


namespace second_cat_brown_kittens_count_l3027_302721

/-- The number of brown-eyed kittens the second cat has -/
def second_cat_brown_kittens : ℕ := sorry

/-- The total number of kittens from both cats -/
def total_kittens : ℕ := 14 + second_cat_brown_kittens

/-- The total number of blue-eyed kittens from both cats -/
def blue_eyed_kittens : ℕ := 7

/-- The percentage of blue-eyed kittens -/
def blue_eyed_percentage : ℚ := 35 / 100

theorem second_cat_brown_kittens_count : second_cat_brown_kittens = 6 := by
  sorry

end second_cat_brown_kittens_count_l3027_302721


namespace min_price_theorem_l3027_302789

/-- Represents the manufacturing scenario with two components -/
structure ManufacturingScenario where
  prod_cost_A : ℝ  -- Production cost for component A
  ship_cost_A : ℝ  -- Shipping cost for component A
  prod_cost_B : ℝ  -- Production cost for component B
  ship_cost_B : ℝ  -- Shipping cost for component B
  fixed_costs : ℝ  -- Fixed costs per month
  units_A : ℕ      -- Number of units of component A produced and sold
  units_B : ℕ      -- Number of units of component B produced and sold

/-- Calculates the total cost for the given manufacturing scenario -/
def total_cost (s : ManufacturingScenario) : ℝ :=
  s.fixed_costs +
  s.units_A * (s.prod_cost_A + s.ship_cost_A) +
  s.units_B * (s.prod_cost_B + s.ship_cost_B)

/-- Theorem: The minimum price per unit that ensures total revenue is at least equal to total costs is $103 -/
theorem min_price_theorem (s : ManufacturingScenario)
  (h1 : s.prod_cost_A = 80)
  (h2 : s.ship_cost_A = 2)
  (h3 : s.prod_cost_B = 60)
  (h4 : s.ship_cost_B = 3)
  (h5 : s.fixed_costs = 16200)
  (h6 : s.units_A = 200)
  (h7 : s.units_B = 300) :
  ∃ (P : ℝ), P ≥ 103 ∧ P * (s.units_A + s.units_B) ≥ total_cost s ∧
  ∀ (Q : ℝ), Q * (s.units_A + s.units_B) ≥ total_cost s → Q ≥ P :=
sorry


end min_price_theorem_l3027_302789


namespace q_min_at_two_l3027_302768

/-- The function q(x) defined as (x - 5)^2 + (x + 1)^2 - 6 -/
def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 6

/-- Theorem stating that q(x) has a minimum value when x = 2 -/
theorem q_min_at_two : 
  ∀ x : ℝ, q 2 ≤ q x := by sorry

end q_min_at_two_l3027_302768


namespace zeros_before_first_nonzero_digit_in_fraction_l3027_302744

theorem zeros_before_first_nonzero_digit_in_fraction :
  ∃ (n : ℕ) (d : ℚ), 
    (d = 7 / 800) ∧ 
    (∃ (m : ℕ), d * (10 ^ n) = m ∧ m ≥ 100 ∧ m < 1000) ∧
    n = 3 :=
by sorry

end zeros_before_first_nonzero_digit_in_fraction_l3027_302744


namespace simplify_first_expression_simplify_second_expression_l3027_302731

-- First expression
theorem simplify_first_expression (a b : ℝ) :
  6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1/2) * a * b) = -a * b := by
  sorry

-- Second expression
theorem simplify_second_expression (t : ℝ) :
  -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 := by
  sorry

end simplify_first_expression_simplify_second_expression_l3027_302731


namespace lemonade_proportion_l3027_302773

theorem lemonade_proportion (lemons_small : ℕ) (gallons_small : ℕ) (gallons_large : ℕ) :
  lemons_small = 36 →
  gallons_small = 48 →
  gallons_large = 100 →
  (lemons_small * gallons_large) / gallons_small = 75 :=
by
  sorry

end lemonade_proportion_l3027_302773


namespace parabola_intersection_points_l3027_302720

/-- The parabola y = x^2 + 2x + a - 2 has exactly two intersection points with the coordinate axes if and only if a = 2 or a = 3 -/
theorem parabola_intersection_points (a : ℝ) : 
  (∃! (x y : ℝ), y = x^2 + 2*x + a - 2 ∧ (x = 0 ∨ y = 0)) ∧ 
  (∃ (x1 x2 y1 y2 : ℝ), (x1 ≠ x2 ∨ y1 ≠ y2) ∧ 
    (y1 = x1^2 + 2*x1 + a - 2) ∧ (y2 = x2^2 + 2*x2 + a - 2) ∧ 
    ((x1 = 0 ∨ y1 = 0) ∧ (x2 = 0 ∨ y2 = 0))) ↔ 
  (a = 2 ∨ a = 3) :=
sorry

end parabola_intersection_points_l3027_302720


namespace train_speed_l3027_302759

/-- The speed of a train given its length, the speed of a man running in the opposite direction,
    and the time it takes for the train to pass the man. -/
theorem train_speed (train_length : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_length = 140 →
  man_speed = 6 →
  passing_time = 6 →
  (train_length / passing_time) * 3.6 - man_speed = 78 := by
  sorry

end train_speed_l3027_302759


namespace chess_tournament_participants_l3027_302774

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- Number of participants excluding the 12 lowest-scoring players
  total_points : ℕ → ℕ → ℚ  -- Function to calculate total points between two groups of players
  lowest_twelve_points : ℚ  -- Points earned by the 12 lowest-scoring players among themselves

/-- The theorem stating the total number of participants in the tournament -/
theorem chess_tournament_participants (t : ChessTournament) : 
  (t.n + 12 = 24) ∧ 
  (t.total_points t.n 12 = t.total_points t.n t.n / 2) ∧
  (t.lowest_twelve_points = 66) ∧
  (t.total_points (t.n + 12) (t.n + 12) / 2 = t.total_points t.n t.n + 2 * t.lowest_twelve_points) :=
by sorry

end chess_tournament_participants_l3027_302774


namespace ellipse_eccentricity_l3027_302783

/-- Given an ellipse C with equation x²/a² + y²/b² = 1 where a > b > 0,
    and points A(-a,0), B(a,0), M(x,y), N(x,-y) on C,
    prove that if the product of slopes of AM and BN is 4/9,
    then the eccentricity of C is √5/3 -/
theorem ellipse_eccentricity (a b : ℝ) (x y : ℝ) :
  a > b → b > 0 →
  x^2 / a^2 + y^2 / b^2 = 1 →
  (y / (x + a)) * (-y / (x - a)) = 4/9 →
  Real.sqrt (1 - b^2 / a^2) = Real.sqrt 5 / 3 := by
  sorry

end ellipse_eccentricity_l3027_302783


namespace complex_fraction_equality_l3027_302712

theorem complex_fraction_equality : (5 * Complex.I) / (1 - 2 * Complex.I) = -2 + Complex.I := by
  sorry

end complex_fraction_equality_l3027_302712


namespace gymnastics_students_count_l3027_302754

/-- The position of a student in a rectangular formation. -/
structure Position where
  column_from_right : ℕ
  column_from_left : ℕ
  row_from_back : ℕ
  row_from_front : ℕ

/-- The gymnastics formation. -/
structure GymnasticsFormation where
  eunji_position : Position
  equal_students_per_row : Bool

/-- Calculate the total number of students in the gymnastics formation. -/
def total_students (formation : GymnasticsFormation) : ℕ :=
  let total_columns := formation.eunji_position.column_from_right +
                       formation.eunji_position.column_from_left - 1
  let total_rows := formation.eunji_position.row_from_back +
                    formation.eunji_position.row_from_front - 1
  total_columns * total_rows

/-- The main theorem stating the total number of students in the given formation. -/
theorem gymnastics_students_count :
  ∀ (formation : GymnasticsFormation),
    formation.eunji_position.column_from_right = 8 →
    formation.eunji_position.column_from_left = 14 →
    formation.eunji_position.row_from_back = 7 →
    formation.eunji_position.row_from_front = 15 →
    formation.equal_students_per_row = true →
    total_students formation = 441 := by
  sorry

end gymnastics_students_count_l3027_302754


namespace total_shirts_produced_l3027_302729

/-- Represents the number of shirts produced per minute -/
def shirts_per_minute : ℕ := 6

/-- Represents the number of minutes the machine operates -/
def operation_time : ℕ := 6

/-- Theorem stating that the total number of shirts produced is 36 -/
theorem total_shirts_produced :
  shirts_per_minute * operation_time = 36 := by
  sorry

end total_shirts_produced_l3027_302729


namespace range_of_difference_l3027_302782

theorem range_of_difference (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) :
  -2 < a - b ∧ a - b < 0 := by
  sorry

end range_of_difference_l3027_302782


namespace ellipse_line_intersection_l3027_302719

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 2 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2

-- Define the midpoint condition
def midpoint_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1/2

-- Main theorem
theorem ellipse_line_intersection :
  ∀ A B : ℝ × ℝ,
  intersection_points A B →
  midpoint_condition A B →
  (∀ x y : ℝ, line x y ↔ x + 2*y - 2 = 0) ∧
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 5) :=
sorry

end ellipse_line_intersection_l3027_302719


namespace candy_distribution_l3027_302764

theorem candy_distribution (total_candy : ℕ) (candy_per_bag : ℕ) (h1 : total_candy = 648) (h2 : candy_per_bag = 81) :
  total_candy / candy_per_bag = 8 := by
  sorry

end candy_distribution_l3027_302764


namespace least_subtraction_for_divisibility_l3027_302785

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 11 ∧ 
  (∀ (y : ℕ), (2000 - y : ℤ) % 17 = 0 → y ≥ x) ∧ 
  (2000 - x : ℤ) % 17 = 0 := by
  sorry

end least_subtraction_for_divisibility_l3027_302785


namespace circle_inequality_l3027_302726

theorem circle_inequality (a b c d : ℝ) (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
    (hab : a * b + c * d = 1)
    (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1) 
    (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) : 
  (a * y₁ + b * y₂ + c * y₃ + d * y₄)^2 + (a * x₁ + b * x₃ + c * x₂ + d * x₁)^2 
    ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := by
  sorry

end circle_inequality_l3027_302726


namespace sum_of_possible_values_l3027_302704

theorem sum_of_possible_values (p q r ℓ : ℂ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 → p ≠ q → q ≠ r → r ≠ p →
  p / (1 - q^2) = ℓ → q / (1 - r^2) = ℓ → r / (1 - p^2) = ℓ →
  ∃ (ℓ₁ ℓ₂ : ℂ), (∀ x : ℂ, x = ℓ → x = ℓ₁ ∨ x = ℓ₂) ∧ ℓ₁ + ℓ₂ = 1 :=
sorry

end sum_of_possible_values_l3027_302704


namespace teena_speed_calculation_l3027_302776

/-- Teena's speed in miles per hour -/
def teena_speed : ℝ := 55

/-- Roe's speed in miles per hour -/
def roe_speed : ℝ := 40

/-- Initial distance Teena is behind Roe in miles -/
def initial_distance_behind : ℝ := 7.5

/-- Time elapsed in hours -/
def time_elapsed : ℝ := 1.5

/-- Final distance Teena is ahead of Roe in miles -/
def final_distance_ahead : ℝ := 15

theorem teena_speed_calculation :
  teena_speed * time_elapsed = 
    roe_speed * time_elapsed + initial_distance_behind + final_distance_ahead := by
  sorry

#check teena_speed_calculation

end teena_speed_calculation_l3027_302776


namespace ten_factorial_mod_thirteen_l3027_302777

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem ten_factorial_mod_thirteen : factorial 10 % 13 = 6 := by sorry

end ten_factorial_mod_thirteen_l3027_302777


namespace min_value_inequality_l3027_302717

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 9) : 
  (x^2 + y^2 + 1) / (x + y) + (x^2 + z^2 + 1) / (x + z) + (y^2 + z^2 + 1) / (y + z) ≥ 4.833 := by
sorry

end min_value_inequality_l3027_302717


namespace tan_double_angle_l3027_302792

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan (2 * α) = 3/4 := by
  sorry

end tan_double_angle_l3027_302792


namespace jaylens_vegetables_l3027_302727

theorem jaylens_vegetables (x y z g : ℚ) : 
  x = 5/3 * y → 
  z = 2 * (1/2 * y) → 
  g = (1/2 * (x/4)) - 3 → 
  20 = x/4 → 
  x + y + z + g = 183 := by
  sorry

end jaylens_vegetables_l3027_302727


namespace circle_reflection_minimum_l3027_302752

/-- Given a circle and a line, if reflection about the line keeps points on the circle,
    then there's a minimum value for a certain expression involving the line's parameters. -/
theorem circle_reflection_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0 → 
   ∃ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' + 1 = 0 ∧ 
              ((x + x')/2, (y + y')/2) ∈ {(x, y) | 2*a*x - b*y + 2 = 0}) →
  (∃ m : ℝ, m = 1/a + 2/b ∧ ∀ k : ℝ, k = 1/a + 2/b → m ≤ k) →
  1/a + 2/b = 3 + 2 * Real.sqrt 2 :=
sorry

end circle_reflection_minimum_l3027_302752


namespace minimum_value_theorem_l3027_302767

theorem minimum_value_theorem (x : ℝ) (h : x > 3) :
  (x + 18) / Real.sqrt (x - 3) ≥ 2 * Real.sqrt 21 ∧
  (∃ x₀ : ℝ, x₀ > 3 ∧ (x₀ + 18) / Real.sqrt (x₀ - 3) = 2 * Real.sqrt 21 ∧ x₀ = 24) :=
by sorry

end minimum_value_theorem_l3027_302767


namespace triangle_perimeter_l3027_302790

theorem triangle_perimeter (a b c : ℝ) : 
  (a - 2)^2 + |b - 4| = 0 → 
  c > 0 →
  c < a + b →
  c > |a - b| →
  ∃ (n : ℕ), c = 2 * n →
  a + b + c = 10 := by
sorry

end triangle_perimeter_l3027_302790


namespace cos_double_alpha_l3027_302743

theorem cos_double_alpha (α : ℝ) : 
  (Real.cos α)^2 + (Real.sqrt 2 / 2)^2 = (Real.sqrt 3 / 2)^2 → 
  Real.cos (2 * α) = -1/2 := by
  sorry

end cos_double_alpha_l3027_302743


namespace triangle_area_l3027_302756

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the area of the triangle is 3 under the given conditions. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  a = Real.sqrt 5 →
  b = 3 →
  Real.sin C = 2 * Real.sin A →
  (1 / 2) * a * c * Real.sin B = 3 :=
by sorry

end triangle_area_l3027_302756


namespace legoland_animals_l3027_302786

theorem legoland_animals (kangaroos koalas pandas : ℕ) : 
  kangaroos = 567 →
  kangaroos = 9 * koalas →
  koalas = 7 * pandas →
  kangaroos + koalas + pandas = 639 := by
sorry

end legoland_animals_l3027_302786


namespace three_hundred_percent_of_forty_l3027_302780

-- Define 300 percent as 3 in decimal form
def three_hundred_percent : ℝ := 3

-- Define the operation of taking a percentage of a number
def percentage_of (percent : ℝ) (number : ℝ) : ℝ := percent * number

-- Theorem statement
theorem three_hundred_percent_of_forty :
  percentage_of three_hundred_percent 40 = 120 := by
  sorry

end three_hundred_percent_of_forty_l3027_302780


namespace colored_plane_theorem_l3027_302746

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point in the plane
def colorAssignment : Point → Color := sorry

-- Define what it means for three points to form an equilateral triangle
def isEquilateralTriangle (A B C : Point) : Prop := sorry

-- Define what it means for a point to be the midpoint of two other points
def isMidpoint (M A C : Point) : Prop := sorry

theorem colored_plane_theorem :
  -- Part (a)
  (¬ ∃ A B C : Point, isEquilateralTriangle A B C ∧ colorAssignment A = colorAssignment B ∧ colorAssignment B = colorAssignment C →
   ∃ A B C : Point, colorAssignment A = colorAssignment B ∧ colorAssignment B = colorAssignment C ∧ isMidpoint B A C) ∧
  -- Part (b)
  ∃ A B C : Point, isEquilateralTriangle A B C ∧ colorAssignment A = colorAssignment B ∧ colorAssignment B = colorAssignment C :=
by sorry

end colored_plane_theorem_l3027_302746


namespace two_digit_sum_problem_l3027_302710

theorem two_digit_sum_problem :
  ∃! (x y z : ℕ), 
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    11 * x + 11 * y + 11 * z = 100 * x + 10 * y + z :=
by
  sorry

end two_digit_sum_problem_l3027_302710


namespace cupcakes_eaten_correct_l3027_302718

/-- Calculates the number of cupcakes Todd ate given the initial number of cupcakes,
    the number of packages, and the number of cupcakes per package. -/
def cupcakes_eaten (initial : ℕ) (packages : ℕ) (per_package : ℕ) : ℕ :=
  initial - (packages * per_package)

/-- Proves that the number of cupcakes Todd ate is correct -/
theorem cupcakes_eaten_correct (initial : ℕ) (packages : ℕ) (per_package : ℕ) :
  cupcakes_eaten initial packages per_package = initial - (packages * per_package) :=
by
  sorry

#eval cupcakes_eaten 39 6 3  -- Should evaluate to 21

end cupcakes_eaten_correct_l3027_302718


namespace remainder_98_102_div_11_l3027_302793

theorem remainder_98_102_div_11 : (98 * 102) % 11 = 6 := by
  sorry

end remainder_98_102_div_11_l3027_302793


namespace campaign_funds_proof_l3027_302794

/-- The total campaign funds raised by the 40th president -/
def total_funds : ℝ := 10000

/-- The amount raised by friends -/
def friends_contribution (total : ℝ) : ℝ := 0.4 * total

/-- The amount raised by family -/
def family_contribution (total : ℝ) : ℝ := 0.3 * (total - friends_contribution total)

/-- The amount contributed by the president himself -/
def president_contribution : ℝ := 4200

theorem campaign_funds_proof :
  friends_contribution total_funds +
  family_contribution total_funds +
  president_contribution = total_funds :=
by sorry

end campaign_funds_proof_l3027_302794


namespace abs_value_properties_l3027_302762

-- Define the absolute value function
def f (x : ℝ) := abs x

-- State the theorem
theorem abs_value_properties :
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end abs_value_properties_l3027_302762


namespace ben_win_probability_l3027_302788

/-- The probability of Ben winning a game, given the probability of losing and that tying is impossible -/
theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 5/7) (h2 : lose_prob + win_prob = 1) : win_prob = 2/7 := by
  sorry

end ben_win_probability_l3027_302788


namespace sufficient_conditions_for_x_squared_less_than_one_l3027_302728

theorem sufficient_conditions_for_x_squared_less_than_one :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 < 1) ∧
  (∀ x : ℝ, -1 < x ∧ x < 0 → x^2 < 1) ∧
  (∀ x : ℝ, -1 < x ∧ x < 1 → x^2 < 1) ∧
  (∃ x : ℝ, x < 1 ∧ x^2 ≥ 1) :=
by sorry

end sufficient_conditions_for_x_squared_less_than_one_l3027_302728


namespace bell_size_ratio_l3027_302737

theorem bell_size_ratio (first_bell : ℝ) (second_bell : ℝ) (third_bell : ℝ) 
  (h1 : first_bell = 50)
  (h2 : third_bell = 4 * second_bell)
  (h3 : first_bell + second_bell + third_bell = 550) :
  second_bell / first_bell = 2 := by
sorry

end bell_size_ratio_l3027_302737


namespace star_sum_minus_emilio_sum_l3027_302735

def star_list : List Nat := List.range 50

def replace_three_with_two (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list : List Nat :=
  star_list.map replace_three_with_two

theorem star_sum_minus_emilio_sum : 
  star_list.sum - emilio_list.sum = 105 := by
  sorry

end star_sum_minus_emilio_sum_l3027_302735


namespace product_pattern_l3027_302707

theorem product_pattern (n : ℕ) : (10 * n + 3) * (10 * n + 7) = 100 * n * (n + 1) + 21 := by
  sorry

end product_pattern_l3027_302707


namespace chives_count_l3027_302795

/-- Given a garden with the following properties:
  - The garden has 20 rows with 10 plants in each row.
  - Parsley is planted in the first 3 rows.
  - Rosemary is planted in the last 2 rows.
  - The remaining rows are planted with chives.
  This theorem proves that the number of chives planted is 150. -/
theorem chives_count (total_rows : ℕ) (plants_per_row : ℕ) 
  (parsley_rows : ℕ) (rosemary_rows : ℕ) : 
  total_rows = 20 → 
  plants_per_row = 10 → 
  parsley_rows = 3 → 
  rosemary_rows = 2 → 
  (total_rows - (parsley_rows + rosemary_rows)) * plants_per_row = 150 := by
  sorry

end chives_count_l3027_302795


namespace bodies_of_revolution_l3027_302748

-- Define the type for geometric solids
inductive GeometricSolid
  | Cylinder
  | HexagonalPyramid
  | Cube
  | Sphere
  | Tetrahedron

-- Define what it means to be a body of revolution
def isBodyOfRevolution : GeometricSolid → Prop :=
  fun solid => match solid with
    | GeometricSolid.Cylinder => True
    | GeometricSolid.Sphere => True
    | _ => False

-- Theorem statement
theorem bodies_of_revolution :
  ∀ (solid : GeometricSolid),
    isBodyOfRevolution solid ↔ (solid = GeometricSolid.Cylinder ∨ solid = GeometricSolid.Sphere) :=
by sorry

end bodies_of_revolution_l3027_302748


namespace line_segment_intersection_range_l3027_302741

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := a * x - y - 2 * a - 1 = 0

-- Define the endpoints of the line segment
def point_A : ℝ × ℝ := (-2, 3)
def point_B : ℝ × ℝ := (5, 2)

-- Define the intersection condition
def intersects_segment (a : ℝ) : Prop :=
  ∃ (x y : ℝ), line_equation a x y ∧
    ((x - point_A.1) / (point_B.1 - point_A.1) =
     (y - point_A.2) / (point_B.2 - point_A.2)) ∧
    0 ≤ (x - point_A.1) / (point_B.1 - point_A.1) ∧
    (x - point_A.1) / (point_B.1 - point_A.1) ≤ 1

-- State the theorem
theorem line_segment_intersection_range :
  ∀ a : ℝ, intersects_segment a ↔ a ≤ -1 ∨ a ≥ 1 :=
sorry

end line_segment_intersection_range_l3027_302741


namespace max_consecutive_interesting_numbers_l3027_302722

/-- A function that checks if a number is interesting (has at least one digit divisible by 3) -/
def is_interesting (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d % 3 = 0

/-- The theorem stating the maximum number of consecutive interesting three-digit numbers -/
theorem max_consecutive_interesting_numbers :
  ∃ start : ℕ,
    start ≥ 100 ∧
    start + 121 ≤ 999 ∧
    (∀ k : ℕ, k ∈ Finset.range 122 → is_interesting (start + k)) ∧
    (∀ m : ℕ, m > 122 →
      ¬∃ s : ℕ, s ≥ 100 ∧ s + m - 1 ≤ 999 ∧
        ∀ j : ℕ, j ∈ Finset.range m → is_interesting (s + j)) :=
by
  sorry


end max_consecutive_interesting_numbers_l3027_302722


namespace water_bottles_fourth_game_l3027_302781

/-- Represents the number of bottles in a case -/
structure CaseSize where
  water : ℕ
  sports_drink : ℕ

/-- Represents the number of cases purchased -/
structure CasesPurchased where
  water : ℕ
  sports_drink : ℕ

/-- Represents the consumption of bottles in a game -/
structure GameConsumption where
  water : ℕ
  sports_drink : ℕ

/-- Calculates the total number of bottles initially available -/
def totalInitialBottles (caseSize : CaseSize) (casesPurchased : CasesPurchased) : ℕ × ℕ :=
  (caseSize.water * casesPurchased.water, caseSize.sports_drink * casesPurchased.sports_drink)

/-- Calculates the total consumption for the first three games -/
def totalConsumptionFirstThreeGames (game1 game2 game3 : GameConsumption) : ℕ × ℕ :=
  (game1.water + game2.water + game3.water, game1.sports_drink + game2.sports_drink + game3.sports_drink)

/-- Theorem: The number of water bottles used in the fourth game is 20 -/
theorem water_bottles_fourth_game 
  (caseSize : CaseSize)
  (casesPurchased : CasesPurchased)
  (game1 game2 game3 : GameConsumption)
  (remainingBottles : ℕ × ℕ) :
  caseSize.water = 20 →
  caseSize.sports_drink = 15 →
  casesPurchased.water = 10 →
  casesPurchased.sports_drink = 5 →
  game1 = { water := 70, sports_drink := 30 } →
  game2 = { water := 40, sports_drink := 20 } →
  game3 = { water := 50, sports_drink := 25 } →
  remainingBottles = (20, 10) →
  let (initialWater, _) := totalInitialBottles caseSize casesPurchased
  let (consumedWater, _) := totalConsumptionFirstThreeGames game1 game2 game3
  initialWater - consumedWater - remainingBottles.1 = 20 := by
  sorry

end water_bottles_fourth_game_l3027_302781


namespace difference_X_Y_cost_per_capsule_l3027_302749

/-- Represents a bottle of capsules -/
structure Bottle where
  capsules : ℕ
  cost : ℚ

/-- Calculates the cost per capsule for a given bottle -/
def costPerCapsule (b : Bottle) : ℚ :=
  b.cost / b.capsules

/-- Theorem stating the difference in cost per capsule between bottles X and Y -/
theorem difference_X_Y_cost_per_capsule :
  let R : Bottle := { capsules := 250, cost := 25/4 }
  let T : Bottle := { capsules := 100, cost := 3 }
  let X : Bottle := { capsules := 300, cost := 15/2 }
  let Y : Bottle := { capsules := 120, cost := 4 }
  abs (costPerCapsule X - costPerCapsule Y) = 83/10000 := by
  sorry

end difference_X_Y_cost_per_capsule_l3027_302749


namespace flower_shop_optimal_strategy_l3027_302760

/-- Represents the flower shop's sales and profit model -/
structure FlowerShop where
  cost : ℝ := 50
  max_margin : ℝ := 0.52
  sales : ℝ → ℝ
  profit : ℝ → ℝ
  profit_after_donation : ℝ → ℝ → ℝ

/-- The main theorem about the flower shop's optimal pricing and donation strategy -/
theorem flower_shop_optimal_strategy (shop : FlowerShop) 
  (h_sales : ∀ x, shop.sales x = -6 * x + 600) 
  (h_profit : ∀ x, shop.profit x = (x - shop.cost) * shop.sales x) 
  (h_profit_donation : ∀ x n, shop.profit_after_donation x n = shop.profit x - n * shop.sales x) 
  (h_price_range : ∀ x, x ≥ shop.cost ∧ x ≤ shop.cost * (1 + shop.max_margin)) :
  (∃ max_profit : ℝ, max_profit = 3750 ∧ 
    ∀ x, shop.profit x ≤ max_profit ∧ 
    (shop.profit 75 = max_profit)) ∧
  (∀ n, (∀ x₁ x₂, x₁ < x₂ → shop.profit_after_donation x₁ n < shop.profit_after_donation x₂ n) 
    ↔ (1 < n ∧ n < 2)) :=
sorry

end flower_shop_optimal_strategy_l3027_302760


namespace class_selection_theorem_l3027_302713

theorem class_selection_theorem (n m k a : ℕ) (h1 : n = 10) (h2 : m = 4) (h3 : k = 4) (h4 : a = 2) :
  (Nat.choose m a) * (Nat.choose (n - m) (k - a)) = 90 := by
  sorry

end class_selection_theorem_l3027_302713


namespace expression_evaluation_l3027_302739

theorem expression_evaluation :
  let x : ℚ := 2
  (x^2 + 2*x + 1) / (x^2 - 1) / ((x / (x - 1)) - 1) = 3 := by
  sorry

end expression_evaluation_l3027_302739


namespace hyperbola_vertex_distance_l3027_302747

/-- The distance between the vertices of a hyperbola with equation x^2/16 - y^2/25 = 1 is 8 -/
theorem hyperbola_vertex_distance : 
  ∀ (x y : ℝ), x^2/16 - y^2/25 = 1 → ∃ (v₁ v₂ : ℝ × ℝ), 
    (v₁.1^2/16 - v₁.2^2/25 = 1) ∧ 
    (v₂.1^2/16 - v₂.2^2/25 = 1) ∧ 
    (v₁.2 = 0) ∧ (v₂.2 = 0) ∧
    (v₁.1 + v₂.1 = 0) ∧
    (|v₁.1 - v₂.1| = 8) :=
by sorry

end hyperbola_vertex_distance_l3027_302747


namespace roller_coaster_cost_l3027_302765

/-- The cost of a roller coaster ride in tickets, given the total number of tickets needed,
    the cost of a Ferris wheel ride, and the cost of a log ride. -/
theorem roller_coaster_cost
  (total_tickets : ℕ)
  (ferris_wheel_cost : ℕ)
  (log_ride_cost : ℕ)
  (h1 : total_tickets = 10)
  (h2 : ferris_wheel_cost = 2)
  (h3 : log_ride_cost = 1)
  : total_tickets - (ferris_wheel_cost + log_ride_cost) = 7 := by
  sorry

#check roller_coaster_cost

end roller_coaster_cost_l3027_302765


namespace justin_and_tim_games_l3027_302755

theorem justin_and_tim_games (total_players : ℕ) (h1 : total_players = 8) :
  Nat.choose (total_players - 2) 2 = 15 := by
  sorry

end justin_and_tim_games_l3027_302755


namespace lesser_number_problem_l3027_302730

theorem lesser_number_problem (x y : ℝ) 
  (sum_eq : x + y = 60) 
  (diff_eq : 4 * y - x = 10) : 
  y = 14 := by sorry

end lesser_number_problem_l3027_302730


namespace discount_apple_price_l3027_302742

/-- Given a discount percentage and the discounted total price for a certain quantity of apples,
    calculate the original price per kilogram. -/
def original_price_per_kg (discount_percent : ℚ) (discounted_total_price : ℚ) (quantity_kg : ℚ) : ℚ :=
  (discounted_total_price / quantity_kg) / (1 - discount_percent)

/-- Theorem stating that if a 40% discount results in $30 for 10 kg of apples, 
    then the original price was $5 per kg. -/
theorem discount_apple_price : 
  original_price_per_kg (40/100) 30 10 = 5 := by
  sorry

#eval original_price_per_kg (40/100) 30 10

end discount_apple_price_l3027_302742


namespace increasing_interval_implies_a_l3027_302763

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + a|

-- State the theorem
theorem increasing_interval_implies_a (a : ℝ) :
  (∀ x ≥ 3, ∀ y > x, f a y > f a x) ∧
  (∀ x < 3, ∃ y > x, f a y ≤ f a x) →
  a = -6 := by
  sorry

end increasing_interval_implies_a_l3027_302763


namespace spherical_to_rectangular_l3027_302736

/-- Conversion from spherical coordinates to rectangular coordinates -/
theorem spherical_to_rectangular (ρ θ φ : ℝ) :
  ρ = 5 ∧ θ = π/4 ∧ φ = π/3 →
  (ρ * Real.sin φ * Real.cos θ = 5 * Real.sqrt 6 / 4) ∧
  (ρ * Real.sin φ * Real.sin θ = 5 * Real.sqrt 6 / 4) ∧
  (ρ * Real.cos φ = 5 / 2) :=
by sorry

end spherical_to_rectangular_l3027_302736


namespace minimum_final_percentage_is_60_percent_l3027_302770

def total_points : ℕ := 700
def passing_threshold : ℚ := 70 / 100
def problem_set_score : ℕ := 100
def midterm1_score : ℚ := 60 / 100
def midterm2_score : ℚ := 70 / 100
def midterm3_score : ℚ := 80 / 100
def final_exam_points : ℕ := 300

def minimum_final_percentage (total : ℕ) (threshold : ℚ) (problem_set : ℕ) 
  (mid1 mid2 mid3 : ℚ) (final_points : ℕ) : ℚ :=
  -- Definition of the function to calculate the minimum final percentage
  sorry

theorem minimum_final_percentage_is_60_percent :
  minimum_final_percentage total_points passing_threshold problem_set_score
    midterm1_score midterm2_score midterm3_score final_exam_points = 60 / 100 :=
by sorry

end minimum_final_percentage_is_60_percent_l3027_302770
