import Mathlib

namespace specific_trade_profit_l2897_289757

/-- Represents a trading scenario for baseball cards -/
structure CardTrade where
  card_given_value : ℝ
  cards_given_count : ℕ
  card_received_value : ℝ

/-- Calculates the profit from a card trade -/
def trade_profit (trade : CardTrade) : ℝ :=
  trade.card_received_value - (trade.card_given_value * trade.cards_given_count)

/-- Theorem stating that the specific trade results in a $5 profit -/
theorem specific_trade_profit :
  let trade : CardTrade := {
    card_given_value := 8,
    cards_given_count := 2,
    card_received_value := 21
  }
  trade_profit trade = 5 := by sorry

end specific_trade_profit_l2897_289757


namespace units_digit_13_times_41_l2897_289708

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The theorem stating that the units digit of 13 · 41 is 3 -/
theorem units_digit_13_times_41 : unitsDigit (13 * 41) = 3 := by
  sorry

end units_digit_13_times_41_l2897_289708


namespace shooting_scenario_outcomes_l2897_289784

/-- Represents the number of shots fired -/
def total_shots : ℕ := 8

/-- Represents the number of successful hits -/
def total_hits : ℕ := 4

/-- Represents the number of consecutive hits required -/
def consecutive_hits : ℕ := 3

/-- Calculates the number of different outcomes for the shooting scenario -/
def shooting_outcomes : ℕ := total_shots + 1 - total_hits

/-- Theorem stating that the number of different outcomes is 20 -/
theorem shooting_scenario_outcomes : 
  shooting_outcomes = 20 := by sorry

end shooting_scenario_outcomes_l2897_289784


namespace quadratic_function_properties_l2897_289736

def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_properties (f : ℝ → ℝ) 
  (h_quadratic : quadratic_function f)
  (h_f_0 : f 0 = 1)
  (h_f_diff : ∀ x, f (x + 1) - f x = 2 * x) :
  (∀ x, f x = x^2 - x + 1) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≤ 3) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≥ 3/4) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = 3) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = 3/4) := by
  sorry

end quadratic_function_properties_l2897_289736


namespace equation_solution_l2897_289786

theorem equation_solution : ∃ x : ℝ, (225 - 4209520 / ((1000795 + (250 + x) * 50) / 27) = 113) ∧ x = 40 := by
  sorry

end equation_solution_l2897_289786


namespace nails_to_buy_l2897_289794

theorem nails_to_buy (tom_nails : ℝ) (toolshed_nails : ℝ) (drawer_nail : ℝ) (neighbor_nails : ℝ) (total_needed : ℝ) :
  tom_nails = 247 →
  toolshed_nails = 144 →
  drawer_nail = 0.5 →
  neighbor_nails = 58.75 →
  total_needed = 625.25 →
  total_needed - (tom_nails + toolshed_nails + drawer_nail + neighbor_nails) = 175 := by
  sorry

end nails_to_buy_l2897_289794


namespace curve_cartesian_to_polar_l2897_289751

/-- Given a curve C in the Cartesian coordinate system described by the parametric equations
    x = cos α and y = sin α + 1, prove that its polar equation is ρ = 2 sin θ. -/
theorem curve_cartesian_to_polar (α θ : Real) (ρ : Real) (x y : Real) :
  (x = Real.cos α ∧ y = Real.sin α + 1) →
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  ρ = 2 * Real.sin θ := by
  sorry

end curve_cartesian_to_polar_l2897_289751


namespace set_operations_l2897_289714

def A : Set ℕ := {x | x > 0 ∧ x < 11}
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6, 7}

theorem set_operations :
  (A ∩ C = {3, 4, 5, 6, 7}) ∧
  ((A \ B) = {5, 6, 7, 8, 9, 10}) ∧
  ((A \ (B ∪ C)) = {8, 9, 10}) ∧
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) := by
sorry

end set_operations_l2897_289714


namespace paper_folding_perimeter_ratio_l2897_289772

/-- Given a square piece of paper with side length 4 inches that is folded in half vertically
    and then cut in half parallel to the fold, the ratio of the perimeter of one of the resulting
    small rectangles to the perimeter of the large rectangle is 5/6. -/
theorem paper_folding_perimeter_ratio :
  let initial_side_length : ℝ := 4
  let small_rectangle_length : ℝ := initial_side_length
  let small_rectangle_width : ℝ := initial_side_length / 4
  let large_rectangle_length : ℝ := initial_side_length
  let large_rectangle_width : ℝ := initial_side_length / 2
  let small_perimeter : ℝ := 2 * (small_rectangle_length + small_rectangle_width)
  let large_perimeter : ℝ := 2 * (large_rectangle_length + large_rectangle_width)
  small_perimeter / large_perimeter = 5 / 6 := by
sorry


end paper_folding_perimeter_ratio_l2897_289772


namespace triangle_inequality_l2897_289729

theorem triangle_inequality (R r a b c p : ℝ) 
  (h_R : R > 0) 
  (h_r : r > 0) 
  (h_a : a > 0) 
  (h_b : b > 0) 
  (h_c : c > 0) 
  (h_p : p = (a + b + c) / 2) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_circumradius : R = (a * b * c) / (4 * p * r)) 
  (h_inradius : r = p * (p - a) * (p - b) * (p - c) / (a * b * c)) :
  20 * R * r - 4 * r^2 ≤ a * b + b * c + c * a ∧ 
  a * b + b * c + c * a ≤ 4 * (R + r)^2 :=
by sorry

end triangle_inequality_l2897_289729


namespace distance_to_market_is_40_l2897_289797

/-- The distance between Andy's house and the market -/
def distance_to_market (distance_to_school : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance - 2 * distance_to_school

/-- Theorem: The distance between Andy's house and the market is 40 meters -/
theorem distance_to_market_is_40 :
  distance_to_market 50 140 = 40 := by
  sorry

end distance_to_market_is_40_l2897_289797


namespace intersection_point_x_coordinate_l2897_289782

theorem intersection_point_x_coordinate :
  let line1 : ℝ → ℝ := λ x => 3 * x - 22
  let line2 : ℝ → ℝ := λ x => 100 - 3 * x
  ∃ x : ℝ, line1 x = line2 x ∧ x = 61 / 3 :=
by sorry

end intersection_point_x_coordinate_l2897_289782


namespace max_points_top_three_l2897_289740

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (games_per_pair : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculate the total number of games in the tournament -/
def total_games (t : Tournament) : ℕ :=
  (t.num_teams.choose 2) * t.games_per_pair

/-- Calculate the maximum points a team can achieve -/
def max_points_per_team (t : Tournament) : ℕ :=
  (t.num_teams - 1) * t.games_per_pair * t.points_for_win

/-- The main theorem to prove -/
theorem max_points_top_three (t : Tournament) 
  (h1 : t.num_teams = 9)
  (h2 : t.games_per_pair = 2)
  (h3 : t.points_for_win = 3)
  (h4 : t.points_for_draw = 1)
  (h5 : t.points_for_loss = 0) :
  ∃ (max_points : ℕ), max_points = 42 ∧ 
  (∀ (top_three_points : ℕ), top_three_points ≤ max_points) ∧
  (∃ (strategy : Tournament → ℕ), strategy t = max_points) :=
sorry

end max_points_top_three_l2897_289740


namespace root_in_interval_l2897_289720

-- Define the function f(x) = x³ - x - 3
def f (x : ℝ) : ℝ := x^3 - x - 3

-- State the theorem
theorem root_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 1 2 ∧ f c = 0 := by
  sorry

end root_in_interval_l2897_289720


namespace runner_time_difference_l2897_289798

theorem runner_time_difference (danny_time steve_time : ℝ) (h1 : danny_time = 27) 
  (h2 : danny_time = steve_time / 2) : steve_time / 4 - danny_time / 2 = 13.5 := by
  sorry

end runner_time_difference_l2897_289798


namespace jefferson_high_school_groups_l2897_289737

/-- Represents the number of students in exactly two groups -/
def students_in_two_groups (total_students : ℕ) (orchestra : ℕ) (band : ℕ) (chorus : ℕ) (in_any_group : ℕ) : ℕ :=
  orchestra + band + chorus - in_any_group

/-- Theorem: Given the conditions from Jefferson High School, 
    the number of students in exactly two groups is 130 -/
theorem jefferson_high_school_groups : 
  students_in_two_groups 500 120 190 220 400 = 130 := by
  sorry

end jefferson_high_school_groups_l2897_289737


namespace f_derivative_at_zero_l2897_289778

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x) / Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 := by sorry

end f_derivative_at_zero_l2897_289778


namespace white_balls_count_l2897_289758

/-- The number of balls in the bag -/
def total_balls : ℕ := 7

/-- The expected number of white balls when drawing 2 balls -/
def expected_white : ℚ := 6/7

/-- Calculates the expected number of white balls drawn -/
def calculate_expected (white_balls : ℕ) : ℚ :=
  (Nat.choose white_balls 2 * 2 + Nat.choose white_balls 1 * Nat.choose (total_balls - white_balls) 1) / Nat.choose total_balls 2

/-- Theorem stating that the number of white balls is 3 -/
theorem white_balls_count :
  ∃ (n : ℕ), n < total_balls ∧ calculate_expected n = expected_white ∧ n = 3 := by
  sorry

end white_balls_count_l2897_289758


namespace no_roots_in_interval_l2897_289776

-- Define the function f(x) = x^3 + x^2 - 2x - 1
def f (x : ℝ) : ℝ := x^3 + x^2 - 2*x - 1

-- State the theorem
theorem no_roots_in_interval :
  (Continuous f) →
  (f 0 < 0) →
  (f 1 < 0) →
  ∀ x ∈ Set.Ioo 0 1, f x ≠ 0 :=
by sorry

end no_roots_in_interval_l2897_289776


namespace factorial_ratio_l2897_289724

theorem factorial_ratio : (Nat.factorial 15) / ((Nat.factorial 6) * (Nat.factorial 9)) = 5005 := by
  sorry

end factorial_ratio_l2897_289724


namespace nonagon_diagonals_l2897_289766

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- A regular nine-sided polygon contains 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end nonagon_diagonals_l2897_289766


namespace plan_y_cheaper_at_min_mb_l2897_289733

/-- Represents the cost of a data plan in cents -/
def PlanCost (initialFee : ℕ) (ratePerMB : ℕ) (dataUsage : ℕ) : ℕ :=
  initialFee * 100 + ratePerMB * dataUsage

/-- The minimum whole number of MBs for Plan Y to be cheaper than Plan X -/
def minMBForPlanYCheaper : ℕ := 501

theorem plan_y_cheaper_at_min_mb :
  PlanCost 25 10 minMBForPlanYCheaper < PlanCost 0 15 minMBForPlanYCheaper ∧
  ∀ m : ℕ, m < minMBForPlanYCheaper →
    PlanCost 0 15 m ≤ PlanCost 25 10 m :=
by sorry

end plan_y_cheaper_at_min_mb_l2897_289733


namespace sum_of_xyz_l2897_289701

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 40) (h2 : x * z = 80) (h3 : y * z = 160) :
  x + y + z = 14 * Real.sqrt 5 := by
  sorry

end sum_of_xyz_l2897_289701


namespace coal_relationship_warehouse_b_coal_amount_l2897_289787

/-- The amount of coal in warehouse A in tons -/
def warehouse_a_coal : ℝ := 130

/-- The amount of coal in warehouse B in tons -/
def warehouse_b_coal : ℝ := 150

/-- Theorem stating the relationship between coal in warehouses A and B -/
theorem coal_relationship : warehouse_a_coal = 0.8 * warehouse_b_coal + 10 := by
  sorry

/-- Theorem proving the amount of coal in warehouse B -/
theorem warehouse_b_coal_amount : warehouse_b_coal = 150 := by
  sorry

end coal_relationship_warehouse_b_coal_amount_l2897_289787


namespace jane_sunflower_seeds_l2897_289710

/-- Calculates the total number of sunflower seeds given the number of cans and seeds per can. -/
def total_seeds (num_cans : ℕ) (seeds_per_can : ℕ) : ℕ :=
  num_cans * seeds_per_can

/-- Theorem stating that 9 cans with 6 seeds each results in 54 total seeds. -/
theorem jane_sunflower_seeds :
  total_seeds 9 6 = 54 := by
  sorry

end jane_sunflower_seeds_l2897_289710


namespace relay_race_arrangements_eq_12_l2897_289723

/-- The number of ways to arrange 5 runners in a relay race with specific constraints -/
def relay_race_arrangements : ℕ :=
  let total_runners : ℕ := 5
  let specific_runners : ℕ := 2
  let other_runners : ℕ := total_runners - specific_runners
  let ways_to_arrange_specific_runners : ℕ := 2
  let ways_to_arrange_other_runners : ℕ := Nat.factorial other_runners
  ways_to_arrange_specific_runners * ways_to_arrange_other_runners

/-- Theorem stating that the number of arrangements is 12 -/
theorem relay_race_arrangements_eq_12 : relay_race_arrangements = 12 := by
  sorry

end relay_race_arrangements_eq_12_l2897_289723


namespace fill_time_three_pipes_l2897_289741

/-- Represents a pipe that can fill or empty a tank -/
structure Pipe where
  rate : ℚ  -- Rate at which the pipe fills (positive) or empties (negative) the tank per hour

/-- Represents a system of pipes filling a tank -/
def PipeSystem (pipes : List Pipe) : ℚ :=
  pipes.map (·.rate) |> List.sum

theorem fill_time_three_pipes (a b c : Pipe) 
  (ha : a.rate = 1/3)
  (hb : b.rate = 1/4)
  (hc : c.rate = -1/4) :
  (PipeSystem [a, b, c])⁻¹ = 3 := by
  sorry

#check fill_time_three_pipes

end fill_time_three_pipes_l2897_289741


namespace profit_difference_example_l2897_289789

/-- Given a total profit and a ratio of division between two parties,
    calculates the difference in their profit shares. -/
def profit_difference (total_profit : ℚ) (ratio_x : ℚ) (ratio_y : ℚ) : ℚ :=
  let total_ratio := ratio_x + ratio_y
  let share_x := (ratio_x / total_ratio) * total_profit
  let share_y := (ratio_y / total_ratio) * total_profit
  share_x - share_y

/-- Theorem stating that for a total profit of 1000 and a ratio of 1/2 : 1/3,
    the difference in profit shares is 200. -/
theorem profit_difference_example :
  profit_difference 1000 (1/2) (1/3) = 200 := by
  sorry

end profit_difference_example_l2897_289789


namespace infinitely_many_common_terms_l2897_289709

/-- Sequence a_n defined by the recurrence relation -/
def a : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 14 * a (n + 1) + a n

/-- Sequence b_n defined by the recurrence relation -/
def b : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 6 * b (n + 1) - b n

/-- There exist infinitely many pairs of natural numbers (n, m) such that a_n = b_m -/
theorem infinitely_many_common_terms : ∀ k : ℕ, ∃ n m : ℕ, n > k ∧ m > k ∧ a n = b m := by
  sorry


end infinitely_many_common_terms_l2897_289709


namespace consecutive_non_prime_powers_l2897_289783

theorem consecutive_non_prime_powers (k : ℕ+) :
  ∃ (n : ℕ), ∀ (i : ℕ), i ∈ Finset.range k →
    ¬∃ (p : ℕ) (e : ℕ), Nat.Prime p ∧ (n + i = p ^ e) := by
  sorry

end consecutive_non_prime_powers_l2897_289783


namespace team_score_proof_l2897_289731

def team_size : ℕ := 15
def absent_members : ℕ := 5
def present_members : ℕ := team_size - absent_members
def scores : List ℕ := [4, 6, 2, 8, 3, 5, 10, 3, 7]

theorem team_score_proof :
  present_members = scores.length ∧ scores.sum = 48 := by sorry

end team_score_proof_l2897_289731


namespace triangle_side_length_l2897_289726

theorem triangle_side_length (A B C : ℝ) (AB BC AC : ℝ) : 
  BC = 1 → 
  A = π / 3 → 
  Real.sin B = 2 * Real.sin C → 
  AB = Real.sqrt 3 := by sorry

end triangle_side_length_l2897_289726


namespace train_length_calculation_l2897_289760

/-- Calculate the length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 60 →
  person_speed = 6 →
  passing_time = 6 →
  (train_speed + person_speed) * passing_time * (1000 / 3600) = 110.04 := by
  sorry

#check train_length_calculation

end train_length_calculation_l2897_289760


namespace range_of_e_l2897_289732

theorem range_of_e (a b c d e : ℝ) 
  (sum_eq : a + b + c + d + e = 8) 
  (sum_squares_eq : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  0 ≤ e ∧ e ≤ 16/5 := by
  sorry

end range_of_e_l2897_289732


namespace circles_internally_tangent_l2897_289744

/-- Represents a circle in 2D space --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if two circles are internally tangent --/
def are_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let distance := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance = abs (c2.radius - c1.radius)

/-- The first circle: x^2 + y^2 - 2x = 0 --/
def circle1 : Circle :=
  { center := (1, 0), radius := 1 }

/-- The second circle: x^2 + y^2 - 2x - 6y - 6 = 0 --/
def circle2 : Circle :=
  { center := (1, 3), radius := 4 }

/-- Theorem stating that the two given circles are internally tangent --/
theorem circles_internally_tangent : are_internally_tangent circle1 circle2 := by
  sorry

end circles_internally_tangent_l2897_289744


namespace unique_solution_l2897_289715

def U : Set ℤ := {-2, 3, 4, 5}

def M (p q : ℝ) : Set ℤ := {x ∈ U | (x : ℝ)^2 + p * x + q = 0}

theorem unique_solution :
  ∃! (p q : ℝ), (U \ M p q : Set ℤ) = {3, 5} := by sorry

end unique_solution_l2897_289715


namespace largest_five_digit_with_product_7_l2897_289748

/-- The product of the first 7 positive integers -/
def product_7 : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

/-- A function to check if a number is a five-digit integer -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- A function to calculate the product of digits of a number -/
def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

/-- The theorem stating that 98752 is the largest five-digit integer
    whose digits have a product equal to (7)(6)(5)(4)(3)(2)(1) -/
theorem largest_five_digit_with_product_7 :
  (is_five_digit 98752) ∧ 
  (digit_product 98752 = product_7) ∧ 
  (∀ n : ℕ, is_five_digit n → digit_product n = product_7 → n ≤ 98752) :=
by sorry

end largest_five_digit_with_product_7_l2897_289748


namespace f_value_plus_derivative_at_pi_half_l2897_289777

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

theorem f_value_plus_derivative_at_pi_half (π : ℝ) (h : π > 0) :
  f π + (deriv f) (π / 2) = -3 / π :=
sorry

end f_value_plus_derivative_at_pi_half_l2897_289777


namespace anna_candy_per_house_proof_l2897_289771

/-- The number of candy pieces Anna gets per house -/
def anna_candy_per_house : ℕ := 14

/-- The number of candy pieces Billy gets per house -/
def billy_candy_per_house : ℕ := 11

/-- The number of houses Anna visits -/
def anna_houses : ℕ := 60

/-- The number of houses Billy visits -/
def billy_houses : ℕ := 75

/-- The difference in total candy pieces between Anna and Billy -/
def candy_difference : ℕ := 15

theorem anna_candy_per_house_proof :
  anna_candy_per_house * anna_houses = billy_candy_per_house * billy_houses + candy_difference :=
by
  sorry

#eval anna_candy_per_house

end anna_candy_per_house_proof_l2897_289771


namespace sine_equality_l2897_289739

theorem sine_equality (n : ℤ) : 0 ≤ n ∧ n ≤ 180 ∧ n = 55 → Real.sin (n * π / 180) = Real.sin (845 * π / 180) := by
  sorry

end sine_equality_l2897_289739


namespace number_thought_of_l2897_289702

theorem number_thought_of (x : ℝ) : (x / 4) + 9 = 15 → x = 24 := by
  sorry

end number_thought_of_l2897_289702


namespace unit_intervals_have_continuum_cardinality_l2897_289703

-- Define the cardinality of the continuum
def continuum_cardinality := Cardinal.mk ℝ

-- Define the open interval (0,1)
def open_unit_interval := Set.Ioo (0 : ℝ) 1

-- Define the closed interval [0,1]
def closed_unit_interval := Set.Icc (0 : ℝ) 1

-- Theorem statement
theorem unit_intervals_have_continuum_cardinality :
  (Cardinal.mk open_unit_interval = continuum_cardinality) ∧
  (Cardinal.mk closed_unit_interval = continuum_cardinality) := by
  sorry

end unit_intervals_have_continuum_cardinality_l2897_289703


namespace paving_rate_per_sq_meter_l2897_289747

/-- Given a rectangular room with length 5.5 m and width 4 m, 
    and a total paving cost of Rs. 16500, 
    prove that the paving rate per square meter is Rs. 750. -/
theorem paving_rate_per_sq_meter 
  (length : ℝ) 
  (width : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 4)
  (h3 : total_cost = 16500) :
  total_cost / (length * width) = 750 := by
  sorry

end paving_rate_per_sq_meter_l2897_289747


namespace largest_non_sum_36_composite_l2897_289788

def is_composite (n : ℕ) : Prop := ∃ m k, 1 < m ∧ 1 < k ∧ n = m * k

def is_sum_of_multiple_36_and_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 0 < a ∧ is_composite b ∧ n = 36 * a + b

theorem largest_non_sum_36_composite : 
  (∀ n > 209, is_sum_of_multiple_36_and_composite n) ∧
  ¬is_sum_of_multiple_36_and_composite 209 :=
sorry

end largest_non_sum_36_composite_l2897_289788


namespace linear_function_composition_l2897_289764

theorem linear_function_composition (a b : ℝ) : 
  (∀ x : ℝ, (3 * (a * x + b) - 6) = 4 * x + 7) → 
  a + b = 17 / 3 := by
sorry

end linear_function_composition_l2897_289764


namespace solution_sets_l2897_289756

def f (a x : ℝ) := x^2 - (a - 1) * x - a

theorem solution_sets (a : ℝ) :
  (a = 2 → {x : ℝ | f 2 x < 0} = {x : ℝ | -1 < x ∧ x < 2}) ∧
  (a > -1 → {x : ℝ | f a x > 0} = {x : ℝ | x < -1 ∨ x > a}) ∧
  (a = -1 → {x : ℝ | f (-1) x > 0} = {x : ℝ | x < -1 ∨ x > -1}) ∧
  (a < -1 → {x : ℝ | f a x > 0} = {x : ℝ | x < a ∨ x > -1}) :=
by sorry

end solution_sets_l2897_289756


namespace percentage_problem_l2897_289742

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 600 = (40 / 100) * 1050 → P = 70 := by
  sorry

end percentage_problem_l2897_289742


namespace grid_whitening_l2897_289775

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 9x9 grid of cells -/
def Grid := Fin 9 → Fin 9 → Color

/-- Represents a corner shape operation -/
structure CornerOperation where
  row : Fin 9
  col : Fin 9
  orientation : Fin 4

/-- Applies a corner operation to a grid -/
def applyOperation (g : Grid) (op : CornerOperation) : Grid :=
  sorry

/-- Checks if all cells in the grid are white -/
def allWhite (g : Grid) : Prop :=
  ∀ (i j : Fin 9), g i j = Color.White

/-- Main theorem: Any grid can be made all white with finite operations -/
theorem grid_whitening (g : Grid) :
  ∃ (ops : List CornerOperation), allWhite (ops.foldl applyOperation g) :=
  sorry

end grid_whitening_l2897_289775


namespace coins_per_pile_l2897_289770

theorem coins_per_pile (total_piles : ℕ) (total_coins : ℕ) (h1 : total_piles = 10) (h2 : total_coins = 30) :
  ∃ (coins_per_pile : ℕ), coins_per_pile * total_piles = total_coins ∧ coins_per_pile = 3 :=
sorry

end coins_per_pile_l2897_289770


namespace left_handed_rock_lovers_l2897_289705

theorem left_handed_rock_lovers (total : ℕ) (left_handed : ℕ) (rock_lovers : ℕ) (right_handed_rock_dislikers : ℕ) :
  total = 30 →
  left_handed = 14 →
  rock_lovers = 20 →
  right_handed_rock_dislikers = 5 →
  ∃ (x : ℕ),
    x = left_handed + rock_lovers - total + right_handed_rock_dislikers ∧
    x = 9 :=
by sorry

end left_handed_rock_lovers_l2897_289705


namespace mr_li_age_is_25_l2897_289734

-- Define Xiaofang's age this year
def xiaofang_age : ℕ := 5

-- Define the number of years in the future
def years_in_future : ℕ := 3

-- Define the age difference between Mr. Li and Xiaofang in the future
def future_age_difference : ℕ := 20

-- Define Mr. Li's age this year
def mr_li_age : ℕ := xiaofang_age + future_age_difference

-- Theorem to prove
theorem mr_li_age_is_25 : mr_li_age = 25 := by
  sorry

end mr_li_age_is_25_l2897_289734


namespace feathers_needed_for_wings_l2897_289790

theorem feathers_needed_for_wings 
  (feathers_per_set : ℕ) 
  (num_sets : ℕ) 
  (charlie_feathers : ℕ) 
  (susan_feathers : ℕ) :
  feathers_per_set = 900 →
  num_sets = 2 →
  charlie_feathers = 387 →
  susan_feathers = 250 →
  feathers_per_set * num_sets - (charlie_feathers + susan_feathers) = 1163 :=
by sorry

end feathers_needed_for_wings_l2897_289790


namespace line_slope_intercept_product_l2897_289763

/-- For a line y = mx + b with negative slope m and positive y-intercept b, 
    the product mb satisfies -1 < mb < 0 -/
theorem line_slope_intercept_product (m b : ℝ) (h1 : m < 0) (h2 : b > 0) : 
  -1 < m * b ∧ m * b < 0 := by
  sorry

end line_slope_intercept_product_l2897_289763


namespace arithmetic_sequence_length_l2897_289792

theorem arithmetic_sequence_length
  (a₁ : ℤ)
  (aₙ : ℤ)
  (d : ℤ)
  (h1 : a₁ = -3)
  (h2 : aₙ = 45)
  (h3 : d = 4)
  (h4 : aₙ = a₁ + (n - 1) * d) :
  n = 13 := by
  sorry

end arithmetic_sequence_length_l2897_289792


namespace sqrt_equation_solution_l2897_289745

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 8 := by sorry

end sqrt_equation_solution_l2897_289745


namespace quadratic_sum_of_b_and_c_l2897_289730

/-- For the quadratic x^2 - 20x + 49, when written as (x+b)^2+c, b+c equals -61 -/
theorem quadratic_sum_of_b_and_c : ∃ b c : ℝ, 
  (∀ x : ℝ, x^2 - 20*x + 49 = (x+b)^2 + c) ∧ b + c = -61 := by
  sorry

end quadratic_sum_of_b_and_c_l2897_289730


namespace total_dolls_count_l2897_289749

/-- The number of dolls owned by grandmother -/
def grandmother_dolls : ℕ := 50

/-- The number of dolls owned by Rene's sister -/
def sister_dolls : ℕ := grandmother_dolls + 2

/-- The number of dolls owned by Rene -/
def rene_dolls : ℕ := 3 * sister_dolls

/-- The total number of dolls owned by Rene, her sister, and their grandmother -/
def total_dolls : ℕ := rene_dolls + sister_dolls + grandmother_dolls

theorem total_dolls_count : total_dolls = 258 := by
  sorry

end total_dolls_count_l2897_289749


namespace pastries_sum_is_147_l2897_289721

/-- The total number of pastries made by Lola, Lulu, and Lila -/
def total_pastries (lola_cupcakes lola_poptarts lola_pies lola_eclairs
                    lulu_cupcakes lulu_poptarts lulu_pies lulu_eclairs
                    lila_cupcakes lila_poptarts lila_pies lila_eclairs : ℕ) : ℕ :=
  lola_cupcakes + lola_poptarts + lola_pies + lola_eclairs +
  lulu_cupcakes + lulu_poptarts + lulu_pies + lulu_eclairs +
  lila_cupcakes + lila_poptarts + lila_pies + lila_eclairs

/-- Theorem stating that the total number of pastries is 147 -/
theorem pastries_sum_is_147 :
  total_pastries 13 10 8 6 16 12 14 9 22 15 10 12 = 147 := by
  sorry

end pastries_sum_is_147_l2897_289721


namespace max_min_product_l2897_289785

theorem max_min_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y + z = 12) (hprod : x * y + y * z + z * x = 30) :
  ∃ (n : ℝ), n = min (x * y) (min (y * z) (z * x)) ∧ n ≤ 2 ∧
  ∀ (m : ℝ), m = min (x * y) (min (y * z) (z * x)) → m ≤ n :=
sorry

end max_min_product_l2897_289785


namespace correct_sticker_count_l2897_289755

/-- Represents the number of stickers per page for each type -/
def stickers_per_page : Fin 4 → ℕ
  | 0 => 5  -- Type A
  | 1 => 3  -- Type B
  | 2 => 2  -- Type C
  | 3 => 1  -- Type D

/-- The total number of pages -/
def total_pages : ℕ := 22

/-- Calculates the total number of stickers for a given type -/
def total_stickers (type : Fin 4) : ℕ :=
  (stickers_per_page type) * total_pages

/-- Theorem stating the correct total number of stickers for each type -/
theorem correct_sticker_count :
  (total_stickers 0 = 110) ∧
  (total_stickers 1 = 66) ∧
  (total_stickers 2 = 44) ∧
  (total_stickers 3 = 22) := by
  sorry

end correct_sticker_count_l2897_289755


namespace f_increasing_on_interval_l2897_289718

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.log x / Real.log (1/2))^2 - 2 * (Real.log x / Real.log (1/2)) + 1

theorem f_increasing_on_interval :
  StrictMonoOn f { x : ℝ | x ≥ Real.sqrt 2 / 2 } :=
sorry

end f_increasing_on_interval_l2897_289718


namespace solve_system_l2897_289754

theorem solve_system (x y : ℤ) (h1 : x + y = 14) (h2 : x - y = 60) : x = 37 := by
  sorry

end solve_system_l2897_289754


namespace distance_between_intersection_points_max_sum_on_C₂_l2897_289750

-- Define the curves C₁ and C₂ in polar coordinates
def C₁ (ρ θ : ℝ) : Prop := ρ * (Real.sin θ + Real.cos θ) = 1
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ - 4 * Real.sin θ

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ (ρ₁ θ₁ ρ₂ θ₂ : ℝ),
    C₁ ρ₁ θ₁ ∧ C₂ ρ₁ θ₁ ∧
    C₁ ρ₂ θ₂ ∧ C₂ ρ₂ θ₂ ∧
    A = (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁) ∧
    B = (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂) ∧
    A ≠ B

-- Theorem 1: Distance between intersection points
theorem distance_between_intersection_points
  (A B : ℝ × ℝ) (h : intersection_points A B) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
sorry

-- Define a point on C₂ in Cartesian coordinates
def point_on_C₂ (x y : ℝ) : Prop :=
  ∃ (ρ θ : ℝ), C₂ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem 2: Maximum value of x + y for points on C₂
theorem max_sum_on_C₂ :
  ∃ (M : ℝ), M = Real.sqrt 10 - 1 ∧
  (∀ x y, point_on_C₂ x y → x + y ≤ M) ∧
  (∃ x y, point_on_C₂ x y ∧ x + y = M) :=
sorry

end distance_between_intersection_points_max_sum_on_C₂_l2897_289750


namespace remaining_balloons_l2897_289713

theorem remaining_balloons (fred_balloons sam_balloons destroyed_balloons : ℝ)
  (h1 : fred_balloons = 10.0)
  (h2 : sam_balloons = 46.0)
  (h3 : destroyed_balloons = 16.0) :
  fred_balloons + sam_balloons - destroyed_balloons = 40.0 :=
by
  sorry

end remaining_balloons_l2897_289713


namespace ceiling_equation_solution_l2897_289765

theorem ceiling_equation_solution :
  ∃! b : ℝ, b + ⌈b⌉ = 21.5 ∧ b = 10.5 := by
  sorry

end ceiling_equation_solution_l2897_289765


namespace family_ages_l2897_289779

/-- Problem statement about the ages of family members -/
theorem family_ages (mark_age john_age emma_age parents_current_age : ℕ)
  (h1 : mark_age = 18)
  (h2 : john_age = mark_age - 10)
  (h3 : emma_age = mark_age - 4)
  (h4 : parents_current_age = 7 * john_age)
  (h5 : parents_current_age = 25 + emma_age) :
  parents_current_age - mark_age = 38 := by
  sorry

end family_ages_l2897_289779


namespace johns_number_l2897_289769

def is_two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem johns_number 
  (t m j d : ℕ) 
  (h1 : is_two_digit_prime t)
  (h2 : is_two_digit_prime m)
  (h3 : is_two_digit_prime j)
  (h4 : is_two_digit_prime d)
  (h5 : t ≠ m ∧ t ≠ j ∧ t ≠ d ∧ m ≠ j ∧ m ≠ d ∧ j ≠ d)
  (h6 : t + j = 26)
  (h7 : m + d = 32)
  (h8 : j + d = 34)
  (h9 : t + d = 36) : 
  j = 13 := by sorry

end johns_number_l2897_289769


namespace inequality_range_proof_l2897_289753

theorem inequality_range_proof (a : ℝ) : 
  (∀ x : ℝ, x > -2/a → a * Real.exp (a * x) - Real.log (x + 2/a) - 2 ≥ 0) ↔ 
  (a ≥ Real.exp 1) :=
sorry

end inequality_range_proof_l2897_289753


namespace solution_set_l2897_289795

def system_solution (x y : ℝ) : Prop :=
  5 * x^2 + 14 * x * y + 10 * y^2 = 17 ∧ 4 * x^2 + 10 * x * y + 6 * y^2 = 8

theorem solution_set : 
  {p : ℝ × ℝ | system_solution p.1 p.2} = {(-1, 2), (11, -7), (-11, 7), (1, -2)} := by
sorry

end solution_set_l2897_289795


namespace reciprocal_of_sum_l2897_289796

theorem reciprocal_of_sum : (1 / (1/3 + 1/4) : ℚ) = 12/7 := by
  sorry

end reciprocal_of_sum_l2897_289796


namespace problem_solution_l2897_289781

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -3) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 5 + (2829/27) := by
sorry

end problem_solution_l2897_289781


namespace reciprocal_sum_inequality_quadratic_inequality_range_l2897_289759

variable (a b c : ℝ)

-- Define the conditions
def sum_condition (a b c : ℝ) : Prop := a + b + c = 3
def positive_condition (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0

-- Theorem 1
theorem reciprocal_sum_inequality (h1 : sum_condition a b c) (h2 : positive_condition a b c) :
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2 := by sorry

-- Theorem 2
theorem quadratic_inequality_range (h1 : sum_condition a b c) (h2 : positive_condition a b c) :
  ∀ m : ℝ, (∀ x : ℝ, -x^2 + m*x + 2 ≤ a^2 + b^2 + c^2) ↔ -2 ≤ m ∧ m ≤ 2 := by sorry

end reciprocal_sum_inequality_quadratic_inequality_range_l2897_289759


namespace complex_magnitude_l2897_289780

theorem complex_magnitude (z : ℂ) (h : (2 + Complex.I) * z = 4 - (1 + Complex.I)^2) : 
  Complex.abs z = 2 := by
  sorry

end complex_magnitude_l2897_289780


namespace range_of_x_l2897_289722

theorem range_of_x (x : ℝ) : Real.sqrt ((x - 3)^2) = x - 3 → x ≥ 3 := by sorry

end range_of_x_l2897_289722


namespace fly_ceiling_distance_l2897_289773

theorem fly_ceiling_distance (z : ℝ) :
  (3 : ℝ)^2 + 2^2 + z^2 = 6^2 → z = Real.sqrt 23 := by
  sorry

end fly_ceiling_distance_l2897_289773


namespace overlapping_sectors_area_l2897_289711

theorem overlapping_sectors_area (r : ℝ) (h : r = 12) :
  let sector_angle : ℝ := 60
  let sector_area := (sector_angle / 360) * Real.pi * r^2
  let triangle_area := (Real.sqrt 3 / 4) * r^2
  let shaded_area := 2 * (sector_area - triangle_area)
  shaded_area = 48 * Real.pi - 72 * Real.sqrt 3 := by
  sorry

end overlapping_sectors_area_l2897_289711


namespace water_leaked_calculation_l2897_289716

/-- The amount of water that leaked out of a bucket -/
def water_leaked (initial : ℝ) (remaining : ℝ) : ℝ :=
  initial - remaining

theorem water_leaked_calculation (initial : ℝ) (remaining : ℝ) 
  (h1 : initial = 0.75)
  (h2 : remaining = 0.5) : 
  water_leaked initial remaining = 0.25 := by
  sorry

end water_leaked_calculation_l2897_289716


namespace is_hyperbola_center_l2897_289761

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 4)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem is_hyperbola_center : 
  ∀ (x y : ℝ), hyperbola_equation x y ↔ 
    ((x - hyperbola_center.1)^2 / 9 - (y - hyperbola_center.2)^2 / 4 = 1) :=
by sorry

end is_hyperbola_center_l2897_289761


namespace number_problem_l2897_289774

theorem number_problem : ∃ x : ℝ, (0.3 * x = 0.6 * 50 + 30) ∧ (x = 200) := by
  sorry

end number_problem_l2897_289774


namespace circle_and_tangents_l2897_289762

-- Define the circles and points
def circle_C (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = (a + 1)^2 + 1}

def circle_D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = 4}

-- Define the conditions
def passes_through (C : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  p ∈ C

def tangent_line (P A : ℝ × ℝ) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (y - A.2 = m * (x - A.1)) →
    ((x, y) ∈ C → (x, y) = A ∨ (x, y) = P)

-- State the theorem
theorem circle_and_tangents :
  ∀ (a : ℝ),
    passes_through (circle_C a) (0, 0) →
    passes_through (circle_C a) (-1, 1) →
    (∀ (P : ℝ × ℝ), P ∈ circle_D →
      ∃ (A B : ℝ × ℝ),
        A.1 = 0 ∧ B.1 = 0 ∧
        tangent_line P A (circle_C a) ∧
        tangent_line P B (circle_C a)) →
  (∀ (x y : ℝ), (x, y) ∈ circle_C a ↔ (x + 1)^2 + y^2 = 1) ∧
  (∃ (min max : ℝ),
    min = 5 * Real.sqrt 2 / 4 ∧
    max = Real.sqrt 2 ∧
    (∀ (P : ℝ × ℝ), P ∈ circle_D →
      ∃ (A B : ℝ × ℝ),
        A.1 = 0 ∧ B.1 = 0 ∧
        tangent_line P A (circle_C a) ∧
        tangent_line P B (circle_C a) ∧
        min ≤ |A.2 - B.2| ∧ |A.2 - B.2| ≤ max)) :=
by sorry

end circle_and_tangents_l2897_289762


namespace correct_quadratic_equation_l2897_289706

theorem correct_quadratic_equation :
  ∀ (b c : ℝ),
  (∃ (b' c' : ℝ), (5 : ℝ) * (1 : ℝ) = c' ∧ 5 + 1 = -b) →
  (∃ (b'' : ℝ), (-7 : ℝ) * (-2 : ℝ) = c) →
  (x^2 + b*x + c = 0) = (x^2 - 6*x + 14 = 0) :=
by sorry

end correct_quadratic_equation_l2897_289706


namespace rainfall_problem_l2897_289746

/-- Rainfall problem -/
theorem rainfall_problem (total_rainfall : ℝ) (ratio : ℝ) :
  total_rainfall = 35 →
  ratio = 1.5 →
  ∃ (first_week : ℝ),
    first_week + ratio * first_week = total_rainfall ∧
    ratio * first_week = 21 := by
  sorry


end rainfall_problem_l2897_289746


namespace train_length_l2897_289768

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (speed_kmph : ℝ) (time_seconds : ℝ) (bridge_length : ℝ) : 
  speed_kmph = 36 → 
  time_seconds = 23.998080153587715 → 
  bridge_length = 140 → 
  (speed_kmph * 1000 / 3600) * time_seconds - bridge_length = 99.98080153587715 :=
by sorry

end train_length_l2897_289768


namespace log_power_sum_l2897_289725

theorem log_power_sum (c d : ℝ) (hc : c = Real.log 16) (hd : d = Real.log 25) :
  (9 : ℝ) ^ (c / d) + (4 : ℝ) ^ (d / c) = 4421 / 625 := by
  sorry

end log_power_sum_l2897_289725


namespace arithmetic_expression_result_l2897_289735

theorem arithmetic_expression_result : 3 + 15 / 3 - 2^3 = 0 := by
  sorry

end arithmetic_expression_result_l2897_289735


namespace calculate_expression_l2897_289700

theorem calculate_expression : 3 * 301 + 4 * 301 + 5 * 301 + 300 = 3912 := by
  sorry

end calculate_expression_l2897_289700


namespace probability_two_twos_in_five_rolls_probability_two_twos_in_five_rolls_proof_l2897_289752

/-- The probability of rolling a 2 exactly two times in five rolls of a fair eight-sided die -/
theorem probability_two_twos_in_five_rolls : ℝ :=
let p : ℝ := 1 / 8  -- probability of rolling a 2
let q : ℝ := 1 - p  -- probability of not rolling a 2
let n : ℕ := 5      -- number of rolls
let k : ℕ := 2      -- number of desired successes
3430 / 32768

/-- Proof that the probability is correct -/
theorem probability_two_twos_in_five_rolls_proof :
  probability_two_twos_in_five_rolls = 3430 / 32768 := by
  sorry

end probability_two_twos_in_five_rolls_probability_two_twos_in_five_rolls_proof_l2897_289752


namespace journey_speed_calculation_l2897_289767

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) :
  total_distance = 560 →
  total_time = 25 →
  first_half_speed = 21 →
  ∃ second_half_speed : ℝ,
    second_half_speed = 24 ∧
    (total_distance / 2) / first_half_speed + (total_distance / 2) / second_half_speed = total_time :=
by sorry

end journey_speed_calculation_l2897_289767


namespace twenty_fifth_in_base5_l2897_289738

/-- Converts a natural number to its representation in base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Checks if a list of digits represents a valid number in base 5 --/
def isValidBase5 (l : List ℕ) : Prop :=
  l.all (· < 5)

/-- Converts a list of base 5 digits to a natural number --/
def fromBase5 (l : List ℕ) : ℕ :=
  l.foldl (fun acc d => 5 * acc + d) 0

theorem twenty_fifth_in_base5 :
  ∃ (l : List ℕ), isValidBase5 l ∧ fromBase5 l = 25 ∧ l = [1, 0, 0] :=
sorry

end twenty_fifth_in_base5_l2897_289738


namespace total_spent_is_12_30_l2897_289719

/-- The cost of the football Alyssa bought -/
def football_cost : ℚ := 571/100

/-- The cost of the marbles Alyssa bought -/
def marbles_cost : ℚ := 659/100

/-- The total amount Alyssa spent on toys -/
def total_spent : ℚ := football_cost + marbles_cost

/-- Theorem stating that the total amount Alyssa spent on toys is $12.30 -/
theorem total_spent_is_12_30 : total_spent = 1230/100 := by
  sorry

end total_spent_is_12_30_l2897_289719


namespace fabian_shopping_cost_l2897_289793

/-- The cost of Fabian's shopping trip -/
def shopping_cost (apple_price : ℝ) (walnut_price : ℝ) (apple_quantity : ℝ) (sugar_quantity : ℝ) (walnut_quantity : ℝ) : ℝ :=
  let sugar_price := apple_price - 1
  apple_price * apple_quantity + sugar_price * sugar_quantity + walnut_price * walnut_quantity

/-- Theorem: The total cost of Fabian's shopping is $16 -/
theorem fabian_shopping_cost : 
  shopping_cost 2 6 5 3 0.5 = 16 := by
  sorry

end fabian_shopping_cost_l2897_289793


namespace average_work_difference_l2897_289712

def daily_differences : List ℤ := [2, -1, 3, 1, -2, 2, 1]

def days_in_week : ℕ := 7

theorem average_work_difference :
  (daily_differences.sum : ℚ) / days_in_week = 0.857 := by
  sorry

end average_work_difference_l2897_289712


namespace cone_surface_area_l2897_289799

/-- The surface area of a cone with slant height 2 and base radius 1 is 3π -/
theorem cone_surface_area :
  let slant_height : ℝ := 2
  let base_radius : ℝ := 1
  let lateral_area := π * base_radius * slant_height
  let base_area := π * base_radius^2
  lateral_area + base_area = 3 * π :=
by sorry

end cone_surface_area_l2897_289799


namespace triangle_problem_l2897_289743

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (b-2a)cos C + c cos B = 0, c = √7, and b = 3a, then the measure of angle C is π/3
    and the area of the triangle is 3√3/4. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (b - 2*a) * Real.cos C + c * Real.cos B = 0 →
  c = Real.sqrt 7 →
  b = 3*a →
  C = π/3 ∧ (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 4 := by
  sorry

end triangle_problem_l2897_289743


namespace rectangle_tiling_tiling_count_l2897_289717

/-- A piece is a shape that can be used to tile a rectangle -/
structure Piece where
  shape : Set (ℕ × ℕ)

/-- A tiling is a way to cover a rectangle with pieces -/
def Tiling (m n : ℕ) (pieces : Finset Piece) :=
  Set (ℕ × ℕ × Piece)

/-- The number of ways to tile a 5 × 2k rectangle with 2k pieces -/
def TilingCount (k : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem rectangle_tiling (n : ℕ) (pieces : Finset Piece) :
  (∃ (t : Tiling 5 n pieces), pieces.card = n) → Even n :=
sorry

/-- The counting theorem -/
theorem tiling_count (k : ℕ) :
  k ≥ 3 → TilingCount k > 2 * 3^(k - 1) :=
sorry

end rectangle_tiling_tiling_count_l2897_289717


namespace four_thirds_of_nine_halves_l2897_289704

theorem four_thirds_of_nine_halves : (4 / 3 : ℚ) * (9 / 2 : ℚ) = 6 := by
  sorry

end four_thirds_of_nine_halves_l2897_289704


namespace coin_flip_probability_l2897_289791

/-- The probability of a coin landing heads. -/
def p_heads : ℚ := 3/5

/-- The probability of a coin landing tails. -/
def p_tails : ℚ := 1 - p_heads

/-- The number of times the coin is flipped. -/
def num_flips : ℕ := 8

/-- The number of initial flips that should be heads. -/
def num_heads : ℕ := 3

/-- The number of final flips that should be tails. -/
def num_tails : ℕ := num_flips - num_heads

/-- The probability of getting heads on the first 3 flips and tails on the last 5 flips. -/
def prob_specific_sequence : ℚ := p_heads^num_heads * p_tails^num_tails

theorem coin_flip_probability : prob_specific_sequence = 864/390625 := by
  sorry

end coin_flip_probability_l2897_289791


namespace num_pentagons_from_circle_points_l2897_289727

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct points on the circle -/
def num_points : ℕ := 15

/-- The number of vertices in a pentagon -/
def pentagon_vertices : ℕ := 5

/-- Theorem: The number of different convex pentagons formed by selecting 5 points
    from 15 distinct points on the circumference of a circle is 3003 -/
theorem num_pentagons_from_circle_points :
  choose num_points pentagon_vertices = 3003 := by sorry

end num_pentagons_from_circle_points_l2897_289727


namespace tea_mixture_price_l2897_289728

/-- Given three varieties of tea with prices and mixing ratios, calculate the price of the mixture --/
theorem tea_mixture_price (price1 price2 price3 : ℚ) (ratio1 ratio2 ratio3 : ℚ) :
  price1 = 126 →
  price2 = 135 →
  price3 = 175.5 →
  ratio1 = 1 →
  ratio2 = 1 →
  ratio3 = 2 →
  (price1 * ratio1 + price2 * ratio2 + price3 * ratio3) / (ratio1 + ratio2 + ratio3) = 153 := by
  sorry

end tea_mixture_price_l2897_289728


namespace polynomial_real_root_exists_l2897_289707

theorem polynomial_real_root_exists (b : ℝ) : ∃ x : ℝ, x^3 + b*x^2 - 4*x + b = 0 := by
  sorry

end polynomial_real_root_exists_l2897_289707
