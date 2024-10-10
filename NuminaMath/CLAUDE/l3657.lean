import Mathlib

namespace range_of_a_for_false_proposition_l3657_365727

theorem range_of_a_for_false_proposition :
  ∀ (a : ℝ),
    (¬ ∃ (x₀ : ℝ), x₀^2 + a*x₀ - 4*a < 0) ↔
    (a ∈ Set.Icc (-16 : ℝ) 0) :=
by sorry

end range_of_a_for_false_proposition_l3657_365727


namespace midpoint_trajectory_l3657_365703

/-- The trajectory of the midpoint of a line segment connecting a point on a unit circle and a fixed point -/
theorem midpoint_trajectory (x y x₀ y₀ : ℝ) : 
  (x₀^2 + y₀^2 = 1) →  -- P is on the unit circle
  (x = (x₀ + 3) / 2) →  -- x-coordinate of midpoint M
  (y = y₀ / 2) →  -- y-coordinate of midpoint M
  ((2*x - 3)^2 + 4*y^2 = 1) :=
by sorry

end midpoint_trajectory_l3657_365703


namespace last_twelve_average_l3657_365741

theorem last_twelve_average (total_count : Nat) (total_average : ℝ) 
  (first_twelve_average : ℝ) (thirteenth_result : ℝ) :
  total_count = 25 →
  total_average = 18 →
  first_twelve_average = 10 →
  thirteenth_result = 90 →
  (total_count * total_average - 12 * first_twelve_average - thirteenth_result) / 12 = 20 := by
sorry

end last_twelve_average_l3657_365741


namespace traffic_light_color_change_probability_l3657_365705

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle time of the traffic light -/
def totalCycleTime (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def colorChangeInterval (cycle : TrafficLightCycle) (observationTime : ℕ) : ℕ :=
  3 * observationTime

/-- Theorem: The probability of observing a color change in a randomly selected 
    4-second interval of a traffic light cycle is 12/85 -/
theorem traffic_light_color_change_probability 
  (cycle : TrafficLightCycle) 
  (h1 : cycle.green = 40)
  (h2 : cycle.yellow = 5)
  (h3 : cycle.red = 40)
  (observationTime : ℕ)
  (h4 : observationTime = 4) :
  (colorChangeInterval cycle observationTime : ℚ) / (totalCycleTime cycle) = 12 / 85 := by
  sorry

end traffic_light_color_change_probability_l3657_365705


namespace parabola_cubic_intersection_l3657_365798

def parabola (x y : ℝ) : Prop := y = 3 * x^2 - 12 * x - 15

def cubic (x y : ℝ) : Prop := y = x^3 - 6 * x^2 + 11 * x - 6

def intersection_points : Set (ℝ × ℝ) := {(-1, 0), (1, -24), (9, 162)}

theorem parabola_cubic_intersection :
  ∀ x y : ℝ, (parabola x y ∧ cubic x y) ↔ (x, y) ∈ intersection_points :=
sorry

end parabola_cubic_intersection_l3657_365798


namespace angle_sum_bounds_l3657_365749

theorem angle_sum_bounds (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_acute_γ : 0 < γ ∧ γ < π / 2)
  (h_sum_sin_sq : Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1) :
  π / 2 < α + β + γ ∧ α + β + γ < 3 * π / 4 := by
sorry

end angle_sum_bounds_l3657_365749


namespace trapezoid_fg_squared_l3657_365711

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  /-- Length of EF -/
  ef : ℝ
  /-- Length of EH -/
  eh : ℝ
  /-- FG is perpendicular to EF and GH -/
  fg_perpendicular : Bool
  /-- Diagonals EG and FH are perpendicular -/
  diagonals_perpendicular : Bool

/-- Theorem about the length of FG in a specific trapezoid -/
theorem trapezoid_fg_squared (t : Trapezoid) 
  (h1 : t.ef = 3)
  (h2 : t.eh = Real.sqrt 2001)
  (h3 : t.fg_perpendicular = true)
  (h4 : t.diagonals_perpendicular = true) :
  ∃ (fg : ℝ), fg^2 = (9 + 3 * Real.sqrt 7977) / 2 := by
  sorry

end trapezoid_fg_squared_l3657_365711


namespace amys_remaining_money_is_56_04_l3657_365724

/-- Calculates the amount of money Amy has left after her purchases --/
def amys_remaining_money (initial_amount : ℚ) (doll_price : ℚ) (doll_count : ℕ)
  (board_game_price : ℚ) (board_game_count : ℕ) (comic_book_price : ℚ) (comic_book_count : ℕ)
  (board_game_discount : ℚ) (sales_tax : ℚ) : ℚ :=
  let doll_cost := doll_price * doll_count
  let board_game_cost := board_game_price * board_game_count
  let comic_book_cost := comic_book_price * comic_book_count
  let discounted_board_game_cost := board_game_cost * (1 - board_game_discount)
  let total_before_tax := doll_cost + discounted_board_game_cost + comic_book_cost
  let total_after_tax := total_before_tax * (1 + sales_tax)
  initial_amount - total_after_tax

/-- Theorem stating that Amy has $56.04 left after her purchases --/
theorem amys_remaining_money_is_56_04 :
  amys_remaining_money 100 1.25 3 12.75 2 3.50 4 0.10 0.08 = 56.04 := by
  sorry

end amys_remaining_money_is_56_04_l3657_365724


namespace lines_symmetric_about_y_axis_l3657_365790

/-- Two lines are symmetric about the y-axis if and only if their coefficients satisfy a specific relation -/
theorem lines_symmetric_about_y_axis 
  (a b c p q m : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hm : m ≠ 0) :
  (∃ (k : ℝ), k ≠ 0 ∧ -a = k*p ∧ b = k*q ∧ c = k*m) ↔ 
  (∀ (x y : ℝ), a*x + b*y + c = 0 ↔ p*(-x) + q*y + m = 0) :=
sorry

end lines_symmetric_about_y_axis_l3657_365790


namespace count_leftmost_seven_eq_diff_l3657_365764

/-- The set of powers of 7 from 0 to 3000 -/
def U : Set ℕ := {n : ℕ | ∃ k : ℕ, 0 ≤ k ∧ k ≤ 3000 ∧ n = 7^k}

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The leftmost digit of a natural number -/
def leftmost_digit (n : ℕ) : ℕ := sorry

/-- The number of elements in U with 7 as the leftmost digit -/
def count_leftmost_seven (U : Set ℕ) : ℕ := sorry

theorem count_leftmost_seven_eq_diff :
  num_digits (7^3000) = 2510 →
  leftmost_digit (7^3000) = 7 →
  count_leftmost_seven U = 3000 - 2509 := by sorry

end count_leftmost_seven_eq_diff_l3657_365764


namespace hyperbola_foci_distance_l3657_365759

/-- The distance between the foci of a hyperbola defined by x^2 - 2xy + y^2 = 2 is 4 -/
theorem hyperbola_foci_distance :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 - 2*x*y + y^2 = 2}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ hyperbola ∧ f₂ ∈ hyperbola ∧
    (∀ p ∈ hyperbola, dist p f₁ - dist p f₂ = 2 ∨ dist p f₂ - dist p f₁ = 2) ∧
    dist f₁ f₂ = 4 :=
by sorry


end hyperbola_foci_distance_l3657_365759


namespace cubic_equation_one_real_root_l3657_365762

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, 2007 * x^3 + 2006 * x^2 + 2005 * x = 0 :=
by sorry

end cubic_equation_one_real_root_l3657_365762


namespace semicircle_perimeter_l3657_365726

/-- The perimeter of a semicircle with radius r is equal to 2r + πr -/
theorem semicircle_perimeter (r : ℝ) (h : r = 35) :
  ∃ P : ℝ, P = 2 * r + π * r := by
  sorry

end semicircle_perimeter_l3657_365726


namespace absolute_value_equation_solution_difference_l3657_365716

theorem absolute_value_equation_solution_difference : ∃ (x y : ℝ), 
  (x ≠ y ∧ 
   (|x^2 + 3*x + 3| = 15 ∧ |y^2 + 3*y + 3| = 15) ∧
   ∀ z : ℝ, |z^2 + 3*z + 3| = 15 → (z = x ∨ z = y)) →
  |x - y| = 7 :=
sorry

end absolute_value_equation_solution_difference_l3657_365716


namespace inequality_implication_l3657_365747

theorem inequality_implication (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^19 / b^19 + b^19 / c^19 + c^19 / a^19 ≤ a^19 / c^19 + b^19 / a^19 + c^19 / b^19) :
  a^20 / b^20 + b^20 / c^20 + c^20 / a^20 ≤ a^20 / c^20 + b^20 / a^20 + c^20 / b^20 :=
by sorry

end inequality_implication_l3657_365747


namespace connie_marbles_l3657_365715

/-- Given that Juan has 25 more marbles than Connie and Juan has 64 marbles, 
    prove that Connie has 39 marbles. -/
theorem connie_marbles (connie juan : ℕ) 
  (h1 : juan = connie + 25) 
  (h2 : juan = 64) : 
  connie = 39 := by
  sorry

end connie_marbles_l3657_365715


namespace football_team_practice_hours_l3657_365786

/-- Given a football team's practice schedule, calculate the total practice hours in a week with one missed day. -/
theorem football_team_practice_hours (practice_hours_per_day : ℕ) (days_in_week : ℕ) (missed_days : ℕ) : 
  practice_hours_per_day = 5 → days_in_week = 7 → missed_days = 1 →
  (days_in_week - missed_days) * practice_hours_per_day = 30 := by
sorry

end football_team_practice_hours_l3657_365786


namespace luke_game_points_l3657_365718

theorem luke_game_points (points_per_round : ℕ) (num_rounds : ℕ) (total_points : ℕ) : 
  points_per_round = 327 → num_rounds = 193 → total_points = points_per_round * num_rounds → total_points = 63111 := by
  sorry

end luke_game_points_l3657_365718


namespace smallest_integer_with_remainder_one_l3657_365788

theorem smallest_integer_with_remainder_one : ∃ n : ℕ, 
  n > 1 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 1 ∧ 
  n % 6 = 1 ∧ 
  (∀ m : ℕ, m > 1 → m % 4 = 1 → m % 5 = 1 → m % 6 = 1 → n ≤ m) ∧
  n = 61 := by
  sorry

end smallest_integer_with_remainder_one_l3657_365788


namespace total_pay_for_given_scenario_l3657_365793

/-- The total amount paid to two employees, where one is paid 120% of the other's pay -/
def total_pay (y_pay : ℝ) : ℝ :=
  y_pay + 1.2 * y_pay

theorem total_pay_for_given_scenario :
  total_pay 260 = 572 := by
  sorry

end total_pay_for_given_scenario_l3657_365793


namespace chessboard_ratio_sum_l3657_365766

/-- The number of rectangles formed on an 8x8 chessboard with 9 horizontal and 9 vertical lines -/
def total_rectangles : ℕ := 1296

/-- The number of squares formed on an 8x8 chessboard with 9 horizontal and 9 vertical lines -/
def total_squares : ℕ := 204

/-- The ratio of squares to rectangles as a simplified fraction -/
def square_rectangle_ratio : ℚ := total_squares / total_rectangles

theorem chessboard_ratio_sum :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ square_rectangle_ratio = m / n ∧ m + n = 125 := by
  sorry

end chessboard_ratio_sum_l3657_365766


namespace area_of_shaded_region_l3657_365783

/-- Given a square composed of 25 congruent smaller squares with a diagonal of 10 cm,
    the total area of all 25 squares is 50 square cm. -/
theorem area_of_shaded_region (diagonal : ℝ) (num_squares : ℕ) : 
  diagonal = 10 → num_squares = 25 → (diagonal^2 / 2) = 50 := by sorry

end area_of_shaded_region_l3657_365783


namespace max_stores_visited_is_four_l3657_365787

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  num_stores : ℕ
  total_visits : ℕ
  num_shoppers : ℕ
  double_visitors : ℕ
  max_stores_visited : ℕ

/-- The specific shopping scenario described in the problem -/
def town_scenario : ShoppingScenario :=
  { num_stores := 7
  , total_visits := 21
  , num_shoppers := 11
  , double_visitors := 7
  , max_stores_visited := 4 }

/-- Theorem stating that the maximum number of stores visited by any single person is 4 -/
theorem max_stores_visited_is_four (s : ShoppingScenario) 
  (h1 : s.num_stores = town_scenario.num_stores)
  (h2 : s.total_visits = town_scenario.total_visits)
  (h3 : s.num_shoppers = town_scenario.num_shoppers)
  (h4 : s.double_visitors = town_scenario.double_visitors)
  (h5 : s.double_visitors * 2 + (s.num_shoppers - s.double_visitors) ≤ s.total_visits) :
  s.max_stores_visited = town_scenario.max_stores_visited :=
by sorry


end max_stores_visited_is_four_l3657_365787


namespace fraction_division_result_l3657_365775

theorem fraction_division_result : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end fraction_division_result_l3657_365775


namespace P_inter_Q_equiv_l3657_365706

-- Define the sets P and Q
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x^2 < 4}

-- State the theorem
theorem P_inter_Q_equiv : P ∩ Q = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end P_inter_Q_equiv_l3657_365706


namespace complement_of_A_l3657_365746

def U : Set ℕ := {x | 0 < x ∧ x < 8}
def A : Set ℕ := {2, 4, 5}

theorem complement_of_A : (U \ A) = {1, 3, 6, 7} := by sorry

end complement_of_A_l3657_365746


namespace A_intersect_B_eq_open_interval_l3657_365728

-- Define set A
def A : Set ℝ := {x | |x - 1| < 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.log (x^2 + 1)}

-- Theorem statement
theorem A_intersect_B_eq_open_interval :
  A ∩ B = Set.Ioo 0 2 := by sorry

end A_intersect_B_eq_open_interval_l3657_365728


namespace farmer_horses_count_l3657_365761

/-- Calculates the number of horses a farmer owns based on hay production and consumption --/
def farmer_horses (last_year_bales : ℕ) (last_year_acres : ℕ) (additional_acres : ℕ) 
                  (bales_per_horse_per_day : ℕ) (remaining_bales : ℕ) : ℕ :=
  let total_acres := last_year_acres + additional_acres
  let bales_per_month := (last_year_bales / last_year_acres) * total_acres
  let feeding_months := 4  -- September to December
  let total_bales := bales_per_month * feeding_months + remaining_bales
  let feeding_days := 122  -- Total days from September 1st to December 31st
  let bales_per_horse := bales_per_horse_per_day * feeding_days
  total_bales / bales_per_horse

/-- Theorem stating the number of horses owned by the farmer --/
theorem farmer_horses_count : 
  farmer_horses 560 5 7 3 12834 = 49 := by
  sorry

end farmer_horses_count_l3657_365761


namespace S_infinite_l3657_365731

/-- The number of distinct odd prime divisors of a natural number -/
def num_odd_prime_divisors (n : ℕ) : ℕ := sorry

/-- The set of natural numbers n for which the number of distinct odd prime divisors of n(n+3) is divisible by 3 -/
def S : Set ℕ := {n : ℕ | 3 ∣ num_odd_prime_divisors (n * (n + 3))}

/-- The main theorem stating that S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end S_infinite_l3657_365731


namespace owner_away_time_l3657_365732

/-- Calculates the time an owner was away based on cat's kibble consumption --/
def time_away (initial_kibble : ℝ) (remaining_kibble : ℝ) (consumption_rate : ℝ) : ℝ :=
  (initial_kibble - remaining_kibble) * consumption_rate

/-- Theorem stating that given the conditions, the owner was away for 8 hours --/
theorem owner_away_time (cat_consumption_rate : ℝ) (initial_kibble : ℝ) (remaining_kibble : ℝ)
  (h1 : cat_consumption_rate = 4) -- Cat eats 1 pound every 4 hours
  (h2 : initial_kibble = 3) -- Bowl filled with 3 pounds initially
  (h3 : remaining_kibble = 1) -- 1 pound remains when owner returns
  : time_away initial_kibble remaining_kibble cat_consumption_rate = 8 := by
  sorry

end owner_away_time_l3657_365732


namespace triangle_sides_proof_l3657_365755

theorem triangle_sides_proof (a b c : ℝ) (h : ℝ) (x : ℝ) :
  b - c = 3 →
  h = 10 →
  (a / 2 + 6) - (a / 2 - 6) = 12 →
  a^2 = 427 / 3 ∧
  b = Real.sqrt (427 / 3) + 3 / 2 ∧
  c = Real.sqrt (427 / 3) - 3 / 2 :=
by sorry

end triangle_sides_proof_l3657_365755


namespace network_connections_l3657_365748

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  (n * k) / 2 = 30 := by
  sorry

end network_connections_l3657_365748


namespace not_perfect_square_different_parity_l3657_365709

theorem not_perfect_square_different_parity (a b : ℤ) 
  (h : a % 2 ≠ b % 2) : 
  ¬∃ (k : ℤ), (a + 3*b) * (5*a + 7*b) = k^2 := by
  sorry

end not_perfect_square_different_parity_l3657_365709


namespace expression_evaluation_l3657_365796

theorem expression_evaluation : (-7)^3 / 7^2 + 4^3 - 5 * 2^2 = 37 := by
  sorry

end expression_evaluation_l3657_365796


namespace average_age_across_rooms_l3657_365729

theorem average_age_across_rooms (room_a_people room_b_people room_c_people : ℕ)
                                 (room_a_avg room_b_avg room_c_avg : ℚ)
                                 (h1 : room_a_people = 8)
                                 (h2 : room_b_people = 5)
                                 (h3 : room_c_people = 7)
                                 (h4 : room_a_avg = 35)
                                 (h5 : room_b_avg = 30)
                                 (h6 : room_c_avg = 25) :
  (room_a_people * room_a_avg + room_b_people * room_b_avg + room_c_people * room_c_avg) /
  (room_a_people + room_b_people + room_c_people : ℚ) = 30.25 := by
  sorry

end average_age_across_rooms_l3657_365729


namespace simplify_fraction_l3657_365739

theorem simplify_fraction : (180 : ℚ) / 1260 = 1 / 7 := by
  sorry

end simplify_fraction_l3657_365739


namespace solution_implies_m_equals_three_l3657_365765

theorem solution_implies_m_equals_three (x y m : ℝ) : 
  x = -2 → y = 1 → m * x + 5 * y = -1 → m = 3 := by
  sorry

end solution_implies_m_equals_three_l3657_365765


namespace sqrt_six_times_sqrt_two_l3657_365784

theorem sqrt_six_times_sqrt_two : Real.sqrt 6 * Real.sqrt 2 = 2 * Real.sqrt 3 := by
  sorry

end sqrt_six_times_sqrt_two_l3657_365784


namespace prob_two_spades_is_one_seventeenth_l3657_365773

/-- A standard deck of cards --/
structure Deck :=
  (total_cards : Nat)
  (spade_cards : Nat)
  (h_total : total_cards = 52)
  (h_spades : spade_cards = 13)

/-- The probability of drawing two spades as the first two cards --/
def prob_two_spades (d : Deck) : ℚ :=
  (d.spade_cards : ℚ) / d.total_cards * (d.spade_cards - 1) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing two spades as the first two cards is 1/17 --/
theorem prob_two_spades_is_one_seventeenth (d : Deck) : prob_two_spades d = 1 / 17 := by
  sorry


end prob_two_spades_is_one_seventeenth_l3657_365773


namespace complex_magnitude_proof_l3657_365754

theorem complex_magnitude_proof : Complex.abs (3/5 - 5/4 * Complex.I) = Real.sqrt 769 / 20 := by
  sorry

end complex_magnitude_proof_l3657_365754


namespace tourists_eq_scientific_l3657_365767

/-- Represents the number of domestic tourists during the "May Day" holiday in 2023 (in millions) -/
def tourists : ℝ := 274

/-- Represents the scientific notation of the number of tourists -/
def tourists_scientific : ℝ := 2.74 * (10 ^ 8)

/-- Theorem stating that the number of tourists in millions is equal to its scientific notation representation -/
theorem tourists_eq_scientific : tourists * (10 ^ 6) = tourists_scientific := by sorry

end tourists_eq_scientific_l3657_365767


namespace total_baseball_cards_l3657_365763

/-- The number of people with baseball cards -/
def num_people : ℕ := 6

/-- The number of baseball cards each person has -/
def cards_per_person : ℕ := 52

/-- The total number of baseball cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem total_baseball_cards : total_cards = 312 := by
  sorry

end total_baseball_cards_l3657_365763


namespace rabbit_storage_l3657_365737

/-- Represents the number of items stored per hole for each animal -/
structure StorageRate where
  rabbit : ℕ
  deer : ℕ
  fox : ℕ

/-- Represents the number of holes dug by each animal -/
structure Holes where
  rabbit : ℕ
  deer : ℕ
  fox : ℕ

/-- The main theorem stating that given the conditions, the rabbit stored 60 items -/
theorem rabbit_storage (rate : StorageRate) (holes : Holes) : 
  rate.rabbit = 4 →
  rate.deer = 5 →
  rate.fox = 7 →
  rate.rabbit * holes.rabbit = rate.deer * holes.deer →
  rate.rabbit * holes.rabbit = rate.fox * holes.fox →
  holes.deer = holes.rabbit - 3 →
  holes.fox = holes.deer + 2 →
  rate.rabbit * holes.rabbit = 60 := by
  sorry

end rabbit_storage_l3657_365737


namespace no_geometric_progression_2_3_5_l3657_365751

theorem no_geometric_progression_2_3_5 : 
  ¬ (∃ (a r : ℝ) (k n : ℕ), 
    a > 0 ∧ r > 0 ∧ 
    a * r^0 = 2 ∧
    a * r^k = 3 ∧
    a * r^n = 5 ∧
    0 < k ∧ k < n) :=
by sorry

end no_geometric_progression_2_3_5_l3657_365751


namespace candy_distribution_l3657_365712

theorem candy_distribution (bags : ℝ) (total_candy : ℕ) (h1 : bags = 15.0) (h2 : total_candy = 75) :
  (total_candy : ℝ) / bags = 5 := by
  sorry

end candy_distribution_l3657_365712


namespace multiplication_difference_l3657_365740

theorem multiplication_difference (number : ℕ) (correct_multiplier : ℕ) (mistaken_multiplier : ℕ) :
  number = 135 →
  correct_multiplier = 43 →
  mistaken_multiplier = 34 →
  (number * correct_multiplier) - (number * mistaken_multiplier) = 1215 := by
sorry

end multiplication_difference_l3657_365740


namespace ted_banana_purchase_l3657_365710

/-- The number of oranges Ted needs to purchase -/
def num_oranges : ℕ := 10

/-- The cost of one banana in dollars -/
def banana_cost : ℚ := 2

/-- The cost of one orange in dollars -/
def orange_cost : ℚ := 3/2

/-- The total cost of the fruits in dollars -/
def total_cost : ℚ := 25

/-- The number of bananas Ted needs to purchase -/
def num_bananas : ℕ := 5

theorem ted_banana_purchase :
  num_bananas * banana_cost + num_oranges * orange_cost = total_cost :=
sorry

end ted_banana_purchase_l3657_365710


namespace three_m_plus_n_equals_46_l3657_365723

theorem three_m_plus_n_equals_46 (m n : ℕ) 
  (h1 : m > n) 
  (h2 : 3 * (3 * m * n - 2)^2 - 2 * (3 * m - 3 * n)^2 = 2019) : 
  3 * m + n = 46 := by
sorry

end three_m_plus_n_equals_46_l3657_365723


namespace expression_evaluation_l3657_365722

theorem expression_evaluation (x y z : ℝ) : 
  (x - (y + z)) - ((x + y) - 2*z) = -2*y - 3*z := by sorry

end expression_evaluation_l3657_365722


namespace largest_coefficient_in_expansion_l3657_365756

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the absolute value of the coefficient for a given r
def coeff (r : ℕ) : ℕ := binomial 7 r

-- State the theorem
theorem largest_coefficient_in_expansion :
  ∀ r : ℕ, r ≤ 7 → coeff r ≤ coeff 4 :=
sorry

end largest_coefficient_in_expansion_l3657_365756


namespace race_finish_orders_l3657_365735

theorem race_finish_orders (n : ℕ) : n = 4 → Nat.factorial n = 24 := by
  sorry

end race_finish_orders_l3657_365735


namespace abc_inequality_l3657_365778

theorem abc_inequality (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a + b + c = 6) 
  (h4 : a * b + b * c + a * c = 9) : 
  0 < a ∧ a < 1 ∧ 1 < b ∧ b < 3 ∧ 3 < c ∧ c < 4 :=
by sorry

end abc_inequality_l3657_365778


namespace solution_pairs_l3657_365725

theorem solution_pairs (x y : ℝ) (hxy : x ≠ y) 
  (eq1 : x^100 - y^100 = 2^99 * (x - y))
  (eq2 : x^200 - y^200 = 2^199 * (x - y)) :
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) := by
sorry

end solution_pairs_l3657_365725


namespace magnitude_of_sum_l3657_365780

/-- Given real x, vectors a and b, with a parallel to b, 
    prove that the magnitude of their sum is √5 -/
theorem magnitude_of_sum (x : ℝ) (a b : ℝ × ℝ) :
  a = (x, 1) →
  b = (4, -2) →
  ∃ (k : ℝ), a = k • b →
  ‖a + b‖ = Real.sqrt 5 := by
  sorry

end magnitude_of_sum_l3657_365780


namespace olympic_numbers_l3657_365750

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

def all_digits_different (x y : ℕ) : Prop :=
  ∀ d, is_valid_digit d → (d ∈ x.digits 10 ↔ d ∉ y.digits 10)

theorem olympic_numbers :
  ∀ x y : ℕ,
    x < 1000 ∧ x ≥ 100 ∧  -- x is a three-digit number
    y < 10000 ∧ y ≥ 1000 ∧  -- y is a four-digit number
    (∀ d, d ∈ x.digits 10 → is_valid_digit d) ∧
    (∀ d, d ∈ y.digits 10 → is_valid_digit d) ∧
    all_digits_different x y ∧
    1 ∉ x.digits 10 ∧
    9 ∉ x.digits 10 ∧
    x / y = 1 / 9  -- Rational division
    →
    x = 163 ∨ x = 318 ∨ x = 729 ∨ x = 1638 ∨ x = 1647 :=
by sorry

end olympic_numbers_l3657_365750


namespace triangle_ratio_specific_l3657_365785

noncomputable def triangle_ratio (BC AC : ℝ) (angle_C : ℝ) : ℝ :=
  let AB := Real.sqrt (BC^2 + AC^2 - 2*BC*AC*(Real.cos angle_C))
  let s := (AB + BC + AC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  let AD := 2 * area / BC
  let BD := Real.sqrt (BC^2 - AD^2)
  let AH := AD - BD / 2
  let HD := BD / 2
  AH / HD

theorem triangle_ratio_specific : 
  triangle_ratio 6 (3 * Real.sqrt 3) (π / 4) = (2 * Real.sqrt 6 - 4) / 5 := by
  sorry

end triangle_ratio_specific_l3657_365785


namespace certain_number_value_l3657_365745

theorem certain_number_value : ∃ x : ℝ, 25 * x = 675 ∧ x = 27 := by
  sorry

end certain_number_value_l3657_365745


namespace third_quiz_score_l3657_365758

theorem third_quiz_score (score1 score2 score3 : ℕ) : 
  score1 = 91 → 
  score2 = 92 → 
  (score1 + score2 + score3) / 3 = 91 → 
  score3 = 90 := by
sorry

end third_quiz_score_l3657_365758


namespace quadratic_max_value_l3657_365700

-- Define the quadratic function
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 1

-- Define the interval
def I : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- State the theorem
theorem quadratic_max_value (t : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), x ∈ I → f t x ≤ m) ∧
  (t < 0 → (∀ (x : ℝ), x ∈ I → f t x ≤ -2*t + 2) ∧ (∃ (x : ℝ), x ∈ I ∧ f t x = -2*t + 2)) ∧
  (t = 0 → (∀ (x : ℝ), x ∈ I → f t x ≤ 2) ∧ (∃ (x : ℝ), x ∈ I ∧ f t x = 2)) ∧
  (t > 0 → (∀ (x : ℝ), x ∈ I → f t x ≤ 2*t + 2) ∧ (∃ (x : ℝ), x ∈ I ∧ f t x = 2*t + 2)) :=
by sorry

end quadratic_max_value_l3657_365700


namespace river_width_l3657_365734

/-- Given a river with the following properties:
  * The river is 4 meters deep
  * The river flows at a rate of 6 kilometers per hour
  * The volume of water flowing into the sea is 26000 cubic meters per minute
  Prove that the width of the river is 65 meters. -/
theorem river_width (depth : ℝ) (flow_rate : ℝ) (volume : ℝ) :
  depth = 4 →
  flow_rate = 6 →
  volume = 26000 →
  (volume / (depth * (flow_rate * 1000 / 60))) = 65 := by
  sorry

end river_width_l3657_365734


namespace machine_work_time_l3657_365720

/-- The number of shirts made by the machine today -/
def shirts_today : ℕ := 8

/-- The number of shirts the machine can make per minute -/
def shirts_per_minute : ℕ := 2

/-- The number of minutes the machine worked today -/
def minutes_worked : ℕ := shirts_today / shirts_per_minute

theorem machine_work_time : minutes_worked = 4 := by
  sorry

end machine_work_time_l3657_365720


namespace problem_solution_l3657_365760

theorem problem_solution (x y : ℝ) : 
  x + y = 150 ∧ 1.20 * y - 0.80 * x = 0.75 * (x + y) → x = 33.75 ∧ y = 116.25 := by
  sorry

end problem_solution_l3657_365760


namespace ai_chip_pass_rate_below_threshold_l3657_365738

-- Define the probabilities for intelligent testing indicators
def p_safety : ℚ := 49/50
def p_energy : ℚ := 48/49
def p_performance : ℚ := 47/48

-- Define the probability of passing manual testing
def p_manual : ℚ := 49/50

-- Define the number of chips selected for manual testing
def n_chips : ℕ := 50

-- Theorem statement
theorem ai_chip_pass_rate_below_threshold :
  let p_intelligent := p_safety * p_energy * p_performance
  let p_overall := p_intelligent * p_manual
  p_overall < 93/100 := by
  sorry

end ai_chip_pass_rate_below_threshold_l3657_365738


namespace vector_equation_solution_l3657_365736

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a : V)

theorem vector_equation_solution (x : V) (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a := by
  sorry

end vector_equation_solution_l3657_365736


namespace remainder_double_n_l3657_365701

theorem remainder_double_n (n : ℤ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end remainder_double_n_l3657_365701


namespace alcohol_water_ratio_l3657_365789

/-- Given a container with alcohol and water, prove the ratio after adding water. -/
theorem alcohol_water_ratio 
  (initial_alcohol : ℚ) 
  (initial_water : ℚ) 
  (added_water : ℚ) 
  (h1 : initial_alcohol = 4) 
  (h2 : initial_water = 4) 
  (h3 : added_water = 2666666666666667 / 1000000000000000) : 
  (initial_alcohol / (initial_water + added_water)) = 3 / 5 := by
sorry

end alcohol_water_ratio_l3657_365789


namespace early_arrival_equals_walking_time_l3657_365743

/-- Represents the scenario of a man meeting his wife while walking home from the train station. -/
structure Scenario where
  /-- The time (in minutes) saved by meeting on the way compared to usual arrival time. -/
  time_saved : ℕ
  /-- The time (in minutes) the man spent walking before meeting his wife. -/
  walking_time : ℕ
  /-- The time (in minutes) the wife would normally drive to the station. -/
  normal_driving_time : ℕ
  /-- Assumption that the normal driving time is the difference between walking time and time saved. -/
  h_normal_driving : normal_driving_time = walking_time - time_saved

/-- Theorem stating that the time the man arrived early at the station equals his walking time. -/
theorem early_arrival_equals_walking_time (s : Scenario) :
  s.walking_time = s.walking_time :=
by sorry

#check early_arrival_equals_walking_time

end early_arrival_equals_walking_time_l3657_365743


namespace sarah_friends_count_l3657_365702

/-- The number of friends Sarah brought into the bedroom -/
def friends_with_sarah (total_people bedroom_people living_room_people : ℕ) : ℕ :=
  total_people - (bedroom_people + living_room_people)

theorem sarah_friends_count :
  ∀ (total_people bedroom_people living_room_people : ℕ),
  total_people = 15 →
  bedroom_people = 3 →
  living_room_people = 8 →
  friends_with_sarah total_people bedroom_people living_room_people = 4 := by
sorry

end sarah_friends_count_l3657_365702


namespace marys_green_beans_weight_l3657_365733

/-- Proves that the weight of green beans is 4 pounds given the conditions of Mary's grocery shopping. -/
theorem marys_green_beans_weight (bag_capacity : ℝ) (milk_weight : ℝ) (remaining_space : ℝ) :
  bag_capacity = 20 →
  milk_weight = 6 →
  remaining_space = 2 →
  ∃ (green_beans_weight : ℝ),
    green_beans_weight + milk_weight + 2 * green_beans_weight = bag_capacity - remaining_space ∧
    green_beans_weight = 4 := by
  sorry

end marys_green_beans_weight_l3657_365733


namespace stone_piles_theorem_l3657_365768

/-- Represents the state of stone piles after operations -/
structure StonePiles :=
  (num_piles : Nat)
  (initial_stones : Nat)
  (operations : Nat)
  (pile_a_stones : Nat)
  (pile_b_stones : Nat)

/-- The theorem to prove -/
theorem stone_piles_theorem (sp : StonePiles) : 
  sp.num_piles = 20 →
  sp.initial_stones = 2006 →
  sp.operations < 20 →
  sp.pile_a_stones = 1990 →
  2080 ≤ sp.pile_b_stones →
  sp.pile_b_stones ≤ 2100 →
  sp.pile_b_stones = 2090 := by
sorry

end stone_piles_theorem_l3657_365768


namespace cube_split_contains_2015_l3657_365753

def split_sum (m : ℕ) : ℕ := (m + 2) * (m - 1) / 2

theorem cube_split_contains_2015 (m : ℕ) (h1 : m > 1) :
  (split_sum m ≥ 1007) ∧ (split_sum (m - 1) < 1007) → m = 45 :=
sorry

end cube_split_contains_2015_l3657_365753


namespace perfect_power_multiple_l3657_365795

theorem perfect_power_multiple : ∃ (n : ℕ), 
  n > 0 ∧ 
  ∃ (a b c : ℕ), 
    2 * n = a^2 ∧ 
    3 * n = b^3 ∧ 
    5 * n = c^5 := by
  sorry

end perfect_power_multiple_l3657_365795


namespace solution_set_equivalence_l3657_365781

theorem solution_set_equivalence (x : ℝ) :
  (x + 1) * (x - 1) < 0 ↔ -1 < x ∧ x < 1 := by sorry

end solution_set_equivalence_l3657_365781


namespace wire_length_proof_l3657_365757

theorem wire_length_proof (part1 part2 total : ℕ) : 
  part1 = 106 →
  part2 = 74 →
  part1 = part2 + 32 →
  total = part1 + part2 →
  total = 180 := by
  sorry

end wire_length_proof_l3657_365757


namespace gcd_of_three_numbers_l3657_365721

theorem gcd_of_three_numbers : Nat.gcd 17934 (Nat.gcd 23526 51774) = 2 := by
  sorry

end gcd_of_three_numbers_l3657_365721


namespace john_memory_card_cost_l3657_365714

/-- The number of pictures John takes per day -/
def pictures_per_day : ℕ := 10

/-- The number of years John has been taking pictures -/
def years : ℕ := 3

/-- The number of images a memory card can store -/
def images_per_card : ℕ := 50

/-- The cost of each memory card in dollars -/
def cost_per_card : ℕ := 60

/-- The number of days in a year (assuming no leap years) -/
def days_per_year : ℕ := 365

theorem john_memory_card_cost :
  (years * days_per_year * pictures_per_day / images_per_card) * cost_per_card = 13140 :=
sorry

end john_memory_card_cost_l3657_365714


namespace meditation_time_per_week_l3657_365744

/-- Calculates the total hours spent meditating in a week given the daily meditation time in minutes -/
def weekly_meditation_hours (daily_minutes : ℕ) : ℚ :=
  (daily_minutes : ℚ) * 7 / 60

theorem meditation_time_per_week :
  weekly_meditation_hours (30 * 2) = 7 := by
  sorry

end meditation_time_per_week_l3657_365744


namespace num_possible_lists_eq_1728_l3657_365719

/-- The number of balls in the bin -/
def num_balls : ℕ := 12

/-- The number of draws -/
def num_draws : ℕ := 3

/-- The number of possible lists when drawing with replacement -/
def num_possible_lists : ℕ := num_balls ^ num_draws

/-- Theorem stating that the number of possible lists is 1728 -/
theorem num_possible_lists_eq_1728 : num_possible_lists = 1728 := by
  sorry

end num_possible_lists_eq_1728_l3657_365719


namespace math_competition_scores_l3657_365777

/-- Represents the scoring system for a math competition. -/
structure ScoringSystem where
  num_questions : ℕ
  correct_points : ℕ
  no_answer_points : ℕ
  wrong_answer_deduction : ℕ

/-- Calculates the number of different possible scores for a given scoring system. -/
def num_different_scores (s : ScoringSystem) : ℕ :=
  sorry

/-- Theorem stating that for the given scoring system, there are 35 different possible scores. -/
theorem math_competition_scores :
  let s : ScoringSystem := {
    num_questions := 10,
    correct_points := 4,
    no_answer_points := 0,
    wrong_answer_deduction := 1
  }
  num_different_scores s = 35 := by
  sorry

end math_competition_scores_l3657_365777


namespace interval_sum_l3657_365770

/-- The theorem states that for an interval [a, b] satisfying the given inequality,
    the sum of its endpoints is 12. -/
theorem interval_sum (a b : ℝ) : 
  (∀ x ∈ Set.Icc a b, |3*x - 80| ≤ |2*x - 105|) → a + b = 12 := by
  sorry

#check interval_sum

end interval_sum_l3657_365770


namespace transport_probabilities_theorem_l3657_365730

structure TransportProbabilities where
  plane : ℝ
  ship : ℝ
  train : ℝ
  car : ℝ
  sum_to_one : plane + ship + train + car = 1
  all_nonnegative : plane ≥ 0 ∧ ship ≥ 0 ∧ train ≥ 0 ∧ car ≥ 0

def prob_train_or_plane (p : TransportProbabilities) : ℝ :=
  p.train + p.plane

def prob_not_ship (p : TransportProbabilities) : ℝ :=
  1 - p.ship

theorem transport_probabilities_theorem (p : TransportProbabilities)
    (h1 : p.plane = 0.2)
    (h2 : p.ship = 0.3)
    (h3 : p.train = 0.4)
    (h4 : p.car = 0.1) :
    prob_train_or_plane p = 0.6 ∧ prob_not_ship p = 0.7 := by
  sorry

end transport_probabilities_theorem_l3657_365730


namespace prob_different_topics_is_five_sixths_l3657_365717

/-- The number of essay topics -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating that the probability of two students selecting different topics
    from 6 available topics is 5/6 -/
theorem prob_different_topics_is_five_sixths :
  prob_different_topics = 5/6 := by sorry

end prob_different_topics_is_five_sixths_l3657_365717


namespace pet_ownership_percentages_l3657_365774

def total_students : ℕ := 500
def dog_owners : ℕ := 125
def cat_owners : ℕ := 100
def rabbit_owners : ℕ := 50

def percent_dog_owners : ℚ := dog_owners / total_students * 100
def percent_cat_owners : ℚ := cat_owners / total_students * 100
def percent_rabbit_owners : ℚ := rabbit_owners / total_students * 100

theorem pet_ownership_percentages :
  percent_dog_owners = 25 ∧
  percent_cat_owners = 20 ∧
  percent_rabbit_owners = 10 :=
by sorry

end pet_ownership_percentages_l3657_365774


namespace parallel_lines_distance_l3657_365772

def a : ℝ × ℝ := (3, -2)
def b : ℝ × ℝ := (4, -1)
def d : ℝ × ℝ := (2, -5)

theorem parallel_lines_distance : 
  let line1 := fun (t : ℝ) => a + t • d
  let line2 := fun (s : ℝ) => b + s • d
  (∃ (p q : ℝ × ℝ), p ∈ Set.range line1 ∧ q ∈ Set.range line2 ∧ 
    ∀ (x y : ℝ × ℝ), x ∈ Set.range line1 → y ∈ Set.range line2 → 
      ‖p - q‖ ≤ ‖x - y‖) →
  ∃ (p q : ℝ × ℝ), p ∈ Set.range line1 ∧ q ∈ Set.range line2 ∧ 
    ‖p - q‖ = (5 * Real.sqrt 29) / 29 :=
by sorry

end parallel_lines_distance_l3657_365772


namespace inverse_proportion_problem_l3657_365794

/-- Given two real numbers x and y that are inversely proportional,
    prove that if x + y = 30 and x = 3y, then when x = -6, y = -28.125 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) 
  (h1 : x * y = k)  -- x and y are inversely proportional
  (h2 : x + y = 30) -- sum condition
  (h3 : x = 3 * y)  -- x is three times y
  : x = -6 → y = -28.125 := by
  sorry

end inverse_proportion_problem_l3657_365794


namespace problem_solving_probability_l3657_365704

theorem problem_solving_probability 
  (prob_A prob_B : ℚ) 
  (h_A : prob_A = 2/3) 
  (h_B : prob_B = 3/4) : 
  1 - (1 - prob_A) * (1 - prob_B) = 11/12 := by
sorry

end problem_solving_probability_l3657_365704


namespace coefficient_of_x4_l3657_365771

theorem coefficient_of_x4 (x : ℝ) : 
  let expr := 5*(x^4 - 2*x^5) + 3*(2*x^2 - x^6 + x^3) - (2*x^6 - 3*x^4 + x^2)
  ∃ (a b c d e f : ℝ), expr = 8*x^4 + a*x^6 + b*x^5 + c*x^3 + d*x^2 + e*x + f :=
by sorry

end coefficient_of_x4_l3657_365771


namespace quadratic_roots_nature_l3657_365752

theorem quadratic_roots_nature (a : ℝ) (h : a < -1) :
  ∃ (x₁ x₂ : ℝ), 
    (a^3 + 1) * x₁^2 + (a^2 + 1) * x₁ - (a + 1) = 0 ∧
    (a^3 + 1) * x₂^2 + (a^2 + 1) * x₂ - (a + 1) = 0 ∧
    x₁ > 0 ∧ x₂ < 0 ∧ |x₂| < x₁ :=
by sorry

end quadratic_roots_nature_l3657_365752


namespace point_on_x_axis_l3657_365708

theorem point_on_x_axis (m : ℝ) : 
  let P : ℝ × ℝ := (m + 3, m + 1)
  (P.2 = 0) → P = (2, 0) := by
sorry

end point_on_x_axis_l3657_365708


namespace exists_more_than_20_components_l3657_365797

/-- A diagonal in a cell can be either left-to-right or right-to-left -/
inductive Diagonal
| LeftToRight
| RightToLeft

/-- A grid is represented as a function from coordinates to diagonals -/
def Grid := Fin 8 → Fin 8 → Diagonal

/-- A point in the grid -/
structure Point where
  x : Fin 8
  y : Fin 8

/-- Two points are connected if there's a path of adjacent diagonals between them -/
def Connected (g : Grid) (p q : Point) : Prop := sorry

/-- A connected component is a maximal set of connected points -/
def ConnectedComponent (g : Grid) (s : Set Point) : Prop := sorry

/-- The number of connected components in a grid -/
def NumComponents (g : Grid) : ℕ := sorry

/-- There exists a configuration with more than 20 connected components -/
theorem exists_more_than_20_components : ∃ g : Grid, NumComponents g > 20 := by sorry

end exists_more_than_20_components_l3657_365797


namespace jean_jane_money_total_jean_jane_money_total_proof_l3657_365782

/-- Given that Jean has three times as much money as Jane, and Jean has $57,
    prove that their combined total is $76. -/
theorem jean_jane_money_total : ℕ → ℕ → Prop :=
  fun jean_money jane_money =>
    (jean_money = 3 * jane_money) →
    (jean_money = 57) →
    (jean_money + jane_money = 76)

/-- The actual theorem instance -/
theorem jean_jane_money_total_proof : jean_jane_money_total 57 19 := by
  sorry

end jean_jane_money_total_jean_jane_money_total_proof_l3657_365782


namespace subtract_largest_3digit_from_smallest_5digit_l3657_365792

def largest_3digit : ℕ := 999
def smallest_5digit : ℕ := 10000

theorem subtract_largest_3digit_from_smallest_5digit :
  smallest_5digit - largest_3digit = 9001 := by
  sorry

end subtract_largest_3digit_from_smallest_5digit_l3657_365792


namespace unique_x_exists_l3657_365742

theorem unique_x_exists : ∃! x : ℝ, x > 0 ∧ x * ↑(⌊x⌋) = 50 ∧ |x - 7.142857| < 0.000001 := by sorry

end unique_x_exists_l3657_365742


namespace merger_proportion_l3657_365713

/-- Represents the proportion of managers in a company -/
def ManagerProportion := Fin 101 → ℚ

/-- Represents the proportion of employees from one company in a merged company -/
def MergedProportion := Fin 101 → ℚ

theorem merger_proportion 
  (company_a_managers : ManagerProportion)
  (company_b_managers : ManagerProportion)
  (merged_managers : ManagerProportion)
  (h1 : company_a_managers 10 = 1)
  (h2 : company_b_managers 30 = 1)
  (h3 : merged_managers 25 = 1) :
  ∃ (result : MergedProportion), result 25 = 1 :=
sorry

end merger_proportion_l3657_365713


namespace complex_power_modulus_l3657_365776

theorem complex_power_modulus : Complex.abs ((2 + 2 * Complex.I) ^ 6 + 3) = 515 := by
  sorry

end complex_power_modulus_l3657_365776


namespace min_time_30_seconds_l3657_365791

/-- Represents a person moving along the perimeter of a square -/
structure Person where
  start_position : ℕ  -- Starting vertex (0 = A, 1 = B, 2 = C, 3 = D)
  speed : ℕ           -- Speed in meters per second

/-- Calculates the minimum time for two people to be on the same side of a square -/
def min_time_same_side (side_length : ℕ) (person_a : Person) (person_b : Person) : ℕ :=
  sorry

/-- Theorem stating that the minimum time for the given scenario is 30 seconds -/
theorem min_time_30_seconds (side_length : ℕ) (person_a person_b : Person) :
  side_length = 50 ∧ 
  person_a = { start_position := 0, speed := 5 } ∧
  person_b = { start_position := 2, speed := 3 } →
  min_time_same_side side_length person_a person_b = 30 :=
sorry

end min_time_30_seconds_l3657_365791


namespace triangle_area_sines_l3657_365779

theorem triangle_area_sines (a b c : ℝ) (h_a : a = 5) (h_b : b = 4 * Real.sqrt 2) (h_c : c = 7) :
  let R := (a * b * c) / (4 * Real.sqrt (((a + b + c)/2) * (((a + b + c)/2) - a) * (((a + b + c)/2) - b) * (((a + b + c)/2) - c)));
  let sin_A := a / (2 * R);
  let sin_B := b / (2 * R);
  let sin_C := c / (2 * R);
  let s := (sin_A + sin_B + sin_C) / 2;
  Real.sqrt (s * (s - sin_A) * (s - sin_B) * (s - sin_C)) = 7 / 25 := by
sorry

end triangle_area_sines_l3657_365779


namespace pentagon_h_coordinate_l3657_365707

structure Pentagon where
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  I : ℝ × ℝ
  J : ℝ × ℝ

def has_vertical_symmetry (p : Pentagon) : Prop := sorry

def area (p : Pentagon) : ℝ := sorry

theorem pentagon_h_coordinate (p : Pentagon) 
  (sym : has_vertical_symmetry p)
  (coords : p.F = (0, 0) ∧ p.G = (0, 6) ∧ p.H.1 = 3 ∧ p.J = (6, 0))
  (total_area : area p = 60) :
  p.H.2 = 14 := by sorry

end pentagon_h_coordinate_l3657_365707


namespace min_discount_factor_proof_l3657_365769

/-- Proves the minimum discount factor for a product with given cost and marked prices, ensuring a minimum profit margin. -/
theorem min_discount_factor_proof (cost_price marked_price : ℝ) (min_profit_margin : ℝ) 
  (h_cost : cost_price = 800)
  (h_marked : marked_price = 1200)
  (h_margin : min_profit_margin = 0.2) :
  ∃ x : ℝ, x = 0.8 ∧ 
    ∀ y : ℝ, (marked_price * y - cost_price ≥ cost_price * min_profit_margin) → y ≥ x :=
by sorry

end min_discount_factor_proof_l3657_365769


namespace nested_fraction_equality_l3657_365799

theorem nested_fraction_equality : 
  1 + 1 / (1 + 1 / (1 + 1 / 2)) = 8 / 5 := by sorry

end nested_fraction_equality_l3657_365799
