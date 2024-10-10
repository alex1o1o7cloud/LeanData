import Mathlib

namespace tournament_games_l2034_203494

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 30 players, 435 games are played -/
theorem tournament_games :
  num_games 30 = 435 := by
  sorry

end tournament_games_l2034_203494


namespace amy_school_year_hours_l2034_203437

/-- Represents Amy's work schedule and earnings --/
structure WorkSchedule where
  summer_hours_per_week : ℕ
  summer_weeks : ℕ
  summer_earnings : ℕ
  school_year_weeks : ℕ
  school_year_earnings : ℕ

/-- Calculates the required hours per week for the school year --/
def required_school_year_hours (schedule : WorkSchedule) : ℚ :=
  (schedule.summer_hours_per_week : ℚ) * (schedule.summer_weeks : ℚ) * (schedule.school_year_earnings : ℚ) /
  ((schedule.summer_earnings : ℚ) * (schedule.school_year_weeks : ℚ))

/-- Theorem stating that Amy needs to work 12 hours per week during the school year --/
theorem amy_school_year_hours (schedule : WorkSchedule)
  (h1 : schedule.summer_hours_per_week = 36)
  (h2 : schedule.summer_weeks = 10)
  (h3 : schedule.summer_earnings = 3000)
  (h4 : schedule.school_year_weeks = 30)
  (h5 : schedule.school_year_earnings = 3000) :
  required_school_year_hours schedule = 12 := by
  sorry


end amy_school_year_hours_l2034_203437


namespace solve_for_y_l2034_203453

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 6 = y + 2) (h2 : x = -5) : y = 44 := by
  sorry

end solve_for_y_l2034_203453


namespace set_equality_implies_a_equals_two_l2034_203477

theorem set_equality_implies_a_equals_two (A B : Set ℕ) (a : ℕ) 
  (h1 : A = {1, 2})
  (h2 : B = {1, a})
  (h3 : A = B) : 
  a = 2 := by
  sorry

end set_equality_implies_a_equals_two_l2034_203477


namespace bus_journey_time_l2034_203450

/-- Represents the journey of Xiao Ming to school -/
structure Journey where
  subway_time : ℕ -- Time taken by subway in minutes
  bus_time : ℕ -- Time taken by bus in minutes
  transfer_time : ℕ -- Time taken for transfer in minutes
  total_time : ℕ -- Total time of the journey in minutes

/-- Theorem stating the correct time spent on the bus -/
theorem bus_journey_time (j : Journey) 
  (h1 : j.subway_time = 30)
  (h2 : j.bus_time = 50)
  (h3 : j.transfer_time = 6)
  (h4 : j.total_time = 40)
  (h5 : j.total_time = j.subway_time + j.bus_time + j.transfer_time) :
  ∃ (actual_bus_time : ℕ), actual_bus_time = 10 ∧ 
    j.total_time = (j.subway_time - (j.subway_time - actual_bus_time)) + actual_bus_time + j.transfer_time :=
by sorry

end bus_journey_time_l2034_203450


namespace at_least_one_less_than_or_equal_to_one_l2034_203462

theorem at_least_one_less_than_or_equal_to_one
  (x y z : ℝ)
  (positive_x : 0 < x)
  (positive_y : 0 < y)
  (positive_z : 0 < z)
  (sum_eq_three : x + y + z = 3) :
  min (x * (x + y - z)) (min (y * (y + z - x)) (z * (z + x - y))) ≤ 1 := by
  sorry

end at_least_one_less_than_or_equal_to_one_l2034_203462


namespace normal_distribution_probability_theorem_l2034_203474

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  δ : ℝ
  hδ_pos : δ > 0

/-- The probability that a random variable is less than a given value -/
noncomputable def prob_lt (ξ : NormalRandomVariable) (x : ℝ) : ℝ :=
  sorry

/-- The probability that a random variable is greater than a given value -/
noncomputable def prob_gt (ξ : NormalRandomVariable) (x : ℝ) : ℝ :=
  sorry

/-- The probability that a random variable is between two given values -/
noncomputable def prob_between (ξ : NormalRandomVariable) (a b : ℝ) : ℝ :=
  sorry

theorem normal_distribution_probability_theorem (ξ : NormalRandomVariable) (p : ℝ) 
    (h1 : ξ.μ = 1)
    (h2 : prob_lt ξ 1 = 1/2)
    (h3 : prob_gt ξ 2 = p) :
  prob_between ξ 0 1 = 1/2 - p := by
  sorry

end normal_distribution_probability_theorem_l2034_203474


namespace min_distance_MN_l2034_203493

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define a line tangent to the unit circle
def tangent_line (P A B : ℝ × ℝ) : Prop :=
  unit_circle A.1 A.2 ∧ unit_circle B.1 B.2 ∧
  ∃ (t : ℝ), (1 - t) • A + t • B = P

-- Define the intersection points M and N
def intersection_points (A B : ℝ × ℝ) (M N : ℝ × ℝ) : Prop :=
  M.2 = 0 ∧ N.1 = 0 ∧ ∃ (t s : ℝ), (1 - t) • A + t • B = M ∧ (1 - s) • A + s • B = N

-- State the theorem
theorem min_distance_MN (P A B M N : ℝ × ℝ) :
  point_on_ellipse P →
  tangent_line P A B →
  intersection_points A B M N →
  ∃ (min_dist : ℝ), min_dist = 3/4 ∧ 
    ∀ (P' A' B' M' N' : ℝ × ℝ), 
      point_on_ellipse P' →
      tangent_line P' A' B' →
      intersection_points A' B' M' N' →
      Real.sqrt ((M'.1 - N'.1)^2 + (M'.2 - N'.2)^2) ≥ min_dist :=
sorry

end min_distance_MN_l2034_203493


namespace ordering_of_exponentials_l2034_203425

theorem ordering_of_exponentials (x a b : ℝ) :
  x > 0 → 1 < b^x → b^x < a^x → 1 < b ∧ b < a := by sorry

end ordering_of_exponentials_l2034_203425


namespace square_of_linear_expression_l2034_203472

theorem square_of_linear_expression (x : ℝ) :
  x = -2 → (3 * x + 4)^2 = 4 := by
  sorry

end square_of_linear_expression_l2034_203472


namespace perpendicular_line_angle_l2034_203429

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x - Real.sqrt 3 * y + 2 = 0

-- Define the angle of inclination of a line perpendicular to l
def perpendicular_angle (θ : ℝ) : Prop :=
  Real.tan θ = -(Real.sqrt 3 / 3)

-- Theorem statement
theorem perpendicular_line_angle :
  ∃ θ, perpendicular_angle θ ∧ θ = 150 * (π / 180) :=
sorry

end perpendicular_line_angle_l2034_203429


namespace tax_free_items_cost_l2034_203406

theorem tax_free_items_cost 
  (total_paid : ℝ)
  (sales_tax : ℝ)
  (tax_rate : ℝ)
  (h1 : total_paid = 30)
  (h2 : sales_tax = 1.28)
  (h3 : tax_rate = 0.08)
  : ∃ (tax_free_cost : ℝ), tax_free_cost = 14 :=
by
  sorry

end tax_free_items_cost_l2034_203406


namespace sin_2x_minus_pi_6_l2034_203492

theorem sin_2x_minus_pi_6 (x : ℝ) (h : Real.cos (x + π / 6) + Real.sin (2 * π / 3 + x) = 1 / 2) :
  Real.sin (2 * x - π / 6) = 7 / 8 := by
  sorry

end sin_2x_minus_pi_6_l2034_203492


namespace candy_bar_payment_l2034_203442

/-- Calculates the number of dimes used to pay for a candy bar -/
def dimes_used (quarter_value : ℕ) (nickel_value : ℕ) (dime_value : ℕ) 
  (num_quarters : ℕ) (num_nickels : ℕ) (change : ℕ) (candy_cost : ℕ) : ℕ :=
  let total_paid := candy_cost + change
  let paid_without_dimes := num_quarters * quarter_value + num_nickels * nickel_value
  (total_paid - paid_without_dimes) / dime_value

theorem candy_bar_payment :
  dimes_used 25 5 10 4 1 4 131 = 3 := by
  sorry

end candy_bar_payment_l2034_203442


namespace absolute_value_inequality_l2034_203409

theorem absolute_value_inequality (m : ℝ) :
  (∀ x : ℝ, |x - 3| + |x + 4| ≥ |2*m - 1|) ↔ -3 ≤ m ∧ m ≤ 4 := by
  sorry

end absolute_value_inequality_l2034_203409


namespace smallest_x_for_fifth_power_l2034_203418

theorem smallest_x_for_fifth_power (x : ℕ) (K : ℤ) : 
  (x = 135000 ∧ 
   180 * x = K^5 ∧ 
   ∀ y : ℕ, y < x → ¬∃ L : ℤ, 180 * y = L^5) :=
sorry

end smallest_x_for_fifth_power_l2034_203418


namespace existence_of_small_triangle_l2034_203473

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of points -/
def PointSet := Set Point

/-- Definition of a square with side length 20 -/
def is_square_20 (A B C D : Point) : Prop := sorry

/-- Check if three points are collinear -/
def are_collinear (P Q R : Point) : Prop := sorry

/-- Check if a point is inside a square -/
def is_inside_square (P : Point) (A B C D : Point) : Prop := sorry

/-- Calculate the area of a triangle -/
def triangle_area (P Q R : Point) : ℝ := sorry

theorem existence_of_small_triangle 
  (A B C D : Point) 
  (T : Fin 2000 → Point)
  (h_square : is_square_20 A B C D)
  (h_inside : ∀ i, is_inside_square (T i) A B C D)
  (h_not_collinear : ∀ P Q R, P ≠ Q → Q ≠ R → P ≠ R → 
    P ∈ {A, B, C, D} ∪ (Set.range T) → 
    Q ∈ {A, B, C, D} ∪ (Set.range T) → 
    R ∈ {A, B, C, D} ∪ (Set.range T) → 
    ¬(are_collinear P Q R)) :
  ∃ P Q R, P ∈ {A, B, C, D} ∪ (Set.range T) ∧ 
           Q ∈ {A, B, C, D} ∪ (Set.range T) ∧ 
           R ∈ {A, B, C, D} ∪ (Set.range T) ∧ 
           triangle_area P Q R < 1/10 :=
sorry

end existence_of_small_triangle_l2034_203473


namespace sum_34_47_in_base5_l2034_203414

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_34_47_in_base5 :
  toBase5 (34 + 47) = [3, 1, 1] :=
sorry

end sum_34_47_in_base5_l2034_203414


namespace ampersand_composition_l2034_203452

def ampersand_right (x : ℝ) : ℝ := 9 - x

def ampersand_left (x : ℝ) : ℝ := x - 9

theorem ampersand_composition : ampersand_left (ampersand_right 10) = -10 := by
  sorry

end ampersand_composition_l2034_203452


namespace interest_groups_intersection_difference_l2034_203468

theorem interest_groups_intersection_difference (total : ℕ) (math : ℕ) (english : ℕ)
  (h_total : total = 200)
  (h_math : math = 80)
  (h_english : english = 155) :
  (min math english) - (math + english - total) = 45 :=
sorry

end interest_groups_intersection_difference_l2034_203468


namespace scout_troop_profit_scout_troop_profit_is_200_l2034_203488

/-- Calculate the profit for a scout troop selling candy bars -/
theorem scout_troop_profit (num_bars : ℕ) (buy_price : ℚ) (sell_price : ℚ) : ℚ :=
  let cost := (num_bars : ℚ) * buy_price / 6
  let revenue := (num_bars : ℚ) * sell_price / 3
  revenue - cost

/-- Prove that the scout troop's profit is $200 -/
theorem scout_troop_profit_is_200 :
  scout_troop_profit 1200 (3 : ℚ) (2 : ℚ) = 200 := by
  sorry

end scout_troop_profit_scout_troop_profit_is_200_l2034_203488


namespace red_balls_count_l2034_203405

/-- The number of red balls in a box with specific conditions -/
def num_red_balls (total : ℕ) (blue : ℕ) : ℕ :=
  let green := 3 * blue
  let red := (total - blue - green) / 3
  red

/-- Theorem stating that the number of red balls is 4 under given conditions -/
theorem red_balls_count :
  num_red_balls 36 6 = 4 :=
by sorry

end red_balls_count_l2034_203405


namespace dog_distance_theorem_l2034_203461

/-- The problem of calculating the distance run by a dog between two people --/
theorem dog_distance_theorem 
  (anderson_speed baxter_speed dog_speed : ℝ)
  (head_start : ℝ)
  (h_anderson_speed : anderson_speed = 2)
  (h_baxter_speed : baxter_speed = 4)
  (h_dog_speed : dog_speed = 10)
  (h_head_start : head_start = 1) :
  let initial_distance := anderson_speed * head_start
  let relative_speed := baxter_speed - anderson_speed
  let catch_up_time := initial_distance / relative_speed
  dog_speed * catch_up_time = 10 := by sorry

end dog_distance_theorem_l2034_203461


namespace forum_posts_theorem_l2034_203432

/-- Calculates the total number of questions and answers posted on a forum in a day. -/
def forum_posts (members : ℕ) (questions_per_hour : ℕ) (answer_ratio : ℕ) : ℕ :=
  let questions_per_day := questions_per_hour * 24
  let answers_per_day := questions_per_day * answer_ratio
  members * (questions_per_day + answers_per_day)

/-- Theorem: Given the forum conditions, the total posts in a day is 1,008,000. -/
theorem forum_posts_theorem :
  forum_posts 1000 7 5 = 1008000 := by
  sorry

end forum_posts_theorem_l2034_203432


namespace opponent_total_score_l2034_203470

def hockey_team_goals : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

structure GameResult where
  team_score : Nat
  opponent_score : Nat

def is_lost_by_two (game : GameResult) : Bool :=
  game.opponent_score = game.team_score + 2

def is_half_or_double (game : GameResult) : Bool :=
  game.team_score = 2 * game.opponent_score ∨ 2 * game.team_score = game.opponent_score

theorem opponent_total_score (games : List GameResult) : 
  (games.length = 8) →
  (games.map (λ g => g.team_score) = hockey_team_goals) →
  (games.filter is_lost_by_two).length = 3 →
  (games.filter (λ g => ¬(is_lost_by_two g))).all is_half_or_double →
  (games.map (λ g => g.opponent_score)).sum = 56 := by
  sorry

end opponent_total_score_l2034_203470


namespace vegetable_planting_methods_l2034_203404

/-- The number of vegetable types available --/
def total_vegetables : ℕ := 4

/-- The number of vegetable types to be chosen --/
def chosen_vegetables : ℕ := 3

/-- The number of soil types --/
def soil_types : ℕ := 3

/-- The number of vegetables to be chosen excluding cucumber --/
def vegetables_to_choose : ℕ := chosen_vegetables - 1

/-- The number of remaining vegetables to choose from --/
def remaining_vegetables : ℕ := total_vegetables - 1

theorem vegetable_planting_methods :
  (Nat.choose remaining_vegetables vegetables_to_choose) * (Nat.factorial chosen_vegetables) = 18 :=
sorry

end vegetable_planting_methods_l2034_203404


namespace arcade_time_calculation_l2034_203479

/-- The number of hours spent at the arcade given the rate and total spend -/
def arcade_time (rate : ℚ) (interval : ℚ) (total_spend : ℚ) : ℚ :=
  (total_spend / rate * interval) / 60

/-- Theorem stating that given a rate of $0.50 per 6 minutes and a total spend of $15, 
    the time spent at the arcade is 3 hours -/
theorem arcade_time_calculation :
  arcade_time (1/2) 6 15 = 3 := by
  sorry

end arcade_time_calculation_l2034_203479


namespace probability_graduate_degree_l2034_203451

/-- Represents the number of college graduates with a graduate degree -/
def G : ℕ := 3

/-- Represents the number of college graduates without a graduate degree -/
def C : ℕ := 16

/-- Represents the number of non-college graduates -/
def N : ℕ := 24

/-- The ratio of college graduates with a graduate degree to non-college graduates is 1:8 -/
axiom ratio_G_N : G * 8 = N * 1

/-- The ratio of college graduates without a graduate degree to non-college graduates is 2:3 -/
axiom ratio_C_N : C * 3 = N * 2

/-- The probability that a randomly picked college graduate has a graduate degree -/
def prob_graduate_degree : ℚ := G / (G + C)

/-- Theorem: The probability that a randomly picked college graduate has a graduate degree is 3/19 -/
theorem probability_graduate_degree : prob_graduate_degree = 3 / 19 := by sorry

end probability_graduate_degree_l2034_203451


namespace machine_working_time_l2034_203486

theorem machine_working_time : ∃ y : ℝ, y > 0 ∧ 1 / (y + 4) + 1 / (y + 2) + 1 / y^2 = 1 / y ∧ y = (-1 + Real.sqrt 5) / 2 := by
  sorry

end machine_working_time_l2034_203486


namespace rain_gauge_calculation_l2034_203407

theorem rain_gauge_calculation : 
  let initial_water : ℝ := 2
  let rate_2pm_to_4pm : ℝ := 4
  let rate_4pm_to_7pm : ℝ := 3
  let rate_7pm_to_9pm : ℝ := 0.5
  let duration_2pm_to_4pm : ℝ := 2
  let duration_4pm_to_7pm : ℝ := 3
  let duration_7pm_to_9pm : ℝ := 2
  
  initial_water + 
  (rate_2pm_to_4pm * duration_2pm_to_4pm) + 
  (rate_4pm_to_7pm * duration_4pm_to_7pm) + 
  (rate_7pm_to_9pm * duration_7pm_to_9pm) = 20 := by
sorry

end rain_gauge_calculation_l2034_203407


namespace additional_oil_amount_l2034_203416

-- Define the original price, reduced price, and additional amount
def original_price : ℝ := 42.75
def reduced_price : ℝ := 34.2
def additional_amount : ℝ := 684

-- Define the price reduction percentage
def price_reduction : ℝ := 0.2

-- Theorem statement
theorem additional_oil_amount :
  reduced_price = original_price * (1 - price_reduction) →
  additional_amount / reduced_price = 20 := by
sorry

end additional_oil_amount_l2034_203416


namespace min_value_xyz_l2034_203484

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 27) :
  x + 3 * y + 6 * z ≥ 27 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧ x₀ + 3 * y₀ + 6 * z₀ = 27 :=
by sorry

end min_value_xyz_l2034_203484


namespace fence_cost_l2034_203419

theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 59) :
  4 * Real.sqrt area * price_per_foot = 4012 := by
  sorry

end fence_cost_l2034_203419


namespace july_birth_percentage_l2034_203476

theorem july_birth_percentage (total : ℕ) (july_births : ℕ) 
  (h1 : total = 150) (h2 : july_births = 18) : 
  (july_births : ℚ) / total * 100 = 12 := by
  sorry

end july_birth_percentage_l2034_203476


namespace max_first_day_volume_l2034_203475

def container_volumes : List Nat := [9, 13, 17, 19, 20, 38]

def is_valid_first_day_selection (selection : List Nat) : Prop :=
  selection.length = 3 ∧ selection.all (λ x => x ∈ container_volumes)

def is_valid_second_day_selection (first_day : List Nat) (second_day : List Nat) : Prop :=
  second_day.length = 2 ∧ 
  second_day.all (λ x => x ∈ container_volumes) ∧
  (∀ x ∈ second_day, x ∉ first_day)

def satisfies_volume_constraint (first_day : List Nat) (second_day : List Nat) : Prop :=
  first_day.sum = 2 * second_day.sum

theorem max_first_day_volume : 
  ∃ (first_day second_day : List Nat),
    is_valid_first_day_selection first_day ∧
    is_valid_second_day_selection first_day second_day ∧
    satisfies_volume_constraint first_day second_day ∧
    first_day.sum = 66 ∧
    (∀ (other_first_day : List Nat),
      is_valid_first_day_selection other_first_day →
      other_first_day.sum ≤ 66) :=
by sorry

end max_first_day_volume_l2034_203475


namespace hiking_team_gloves_l2034_203433

/-- The minimum number of gloves needed for a hiking team -/
theorem hiking_team_gloves (participants : ℕ) (gloves_per_pair : ℕ) : 
  participants = 43 → gloves_per_pair = 2 → participants * gloves_per_pair = 86 := by
  sorry

end hiking_team_gloves_l2034_203433


namespace expression_simplification_l2034_203483

theorem expression_simplification (a : ℝ) (h : a^2 - 4*a + 3 = 0) :
  (a - 4) / a / ((a + 2) / (a^2 - 2*a) - (a - 1) / (a^2 - 4*a + 4)) = 1 := by
  sorry

end expression_simplification_l2034_203483


namespace trapezium_side_length_l2034_203460

/-- Given a trapezium with the following properties:
  * One parallel side is 18 cm long
  * The distance between parallel sides is 11 cm
  * The area is 209 cm²
  Then the length of the other parallel side is 20 cm -/
theorem trapezium_side_length (a b h : ℝ) (hb : b = 18) (hh : h = 11) (harea : (a + b) * h / 2 = 209) :
  a = 20 := by
  sorry

end trapezium_side_length_l2034_203460


namespace vector_parallel_value_l2034_203411

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_parallel_value : 
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ → ℝ × ℝ := λ m ↦ (m, -4)
  ∀ m : ℝ, parallel a (b m) → m = 6 := by
sorry

end vector_parallel_value_l2034_203411


namespace combined_final_price_theorem_l2034_203438

def calculate_final_price (cost_price repairs discount_rate tax_rate : ℝ) : ℝ :=
  let total_cost := cost_price + repairs
  let discounted_price := total_cost * (1 - discount_rate)
  discounted_price * (1 + tax_rate)

def cycle_a_price := calculate_final_price 1800 200 0.10 0.05
def cycle_b_price := calculate_final_price 2400 300 0.12 0.06
def cycle_c_price := calculate_final_price 3200 400 0.15 0.07

theorem combined_final_price_theorem :
  cycle_a_price + cycle_b_price + cycle_c_price = 7682.76 := by
  sorry

end combined_final_price_theorem_l2034_203438


namespace geometric_sequence_product_l2034_203417

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 3 * a 11 = 8 →
  a 2 * a 8 = 4 := by
  sorry

end geometric_sequence_product_l2034_203417


namespace pr_qs_ratio_l2034_203469

-- Define the points and distances
def P : ℝ := 0
def Q : ℝ := 3
def R : ℝ := 9
def S : ℝ := 20

-- State the theorem
theorem pr_qs_ratio :
  (R - P) / (S - Q) = 9 / 17 := by
  sorry

end pr_qs_ratio_l2034_203469


namespace inequality_proof_equality_condition_l2034_203448

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z ≥ 3) :
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 :=
sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z ≥ 3) :
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) :=
sorry

end inequality_proof_equality_condition_l2034_203448


namespace initial_amount_is_21_l2034_203428

/-- Represents the money transactions between three people A, B, and C. -/
structure MoneyTransaction where
  a_initial : ℚ
  b_initial : ℚ := 5
  c_initial : ℚ := 9

/-- Calculates the final amounts after all transactions. -/
def final_amounts (mt : MoneyTransaction) : ℚ × ℚ × ℚ :=
  let a1 := mt.a_initial - (mt.b_initial + mt.c_initial)
  let b1 := 2 * mt.b_initial
  let c1 := 2 * mt.c_initial
  
  let a2 := a1 + (a1 / 2)
  let b2 := b1 - ((a1 / 2) + (c1 / 2))
  let c2 := c1 + (c1 / 2)
  
  let a3 := a2 + 3 * a2 + 3 * b2
  let b3 := b2 + 3 * b2 + 3 * c2
  let c3 := c2 - (3 * a2 + 3 * b2)
  
  (a3, b3, c3)

/-- Theorem stating that if the final amounts are (24, 16, 8), then A started with 21 cents. -/
theorem initial_amount_is_21 (mt : MoneyTransaction) : 
  final_amounts mt = (24, 16, 8) → mt.a_initial = 21 := by
  sorry

end initial_amount_is_21_l2034_203428


namespace percent_y_of_x_l2034_203446

theorem percent_y_of_x (x y : ℝ) (h : 0.2 * (x - y) = 0.14 * (x + y)) :
  y = (300 / 17) / 100 * x := by
  sorry

end percent_y_of_x_l2034_203446


namespace simplify_complex_fraction_l2034_203456

theorem simplify_complex_fraction (a b : ℝ) 
  (h1 : a + b ≠ 0) 
  (h2 : a - 2*b ≠ 0) 
  (h3 : a^2 - b^2 ≠ 0) 
  (h4 : a^2 - 4*a*b + 4*b^2 ≠ 0) : 
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end simplify_complex_fraction_l2034_203456


namespace calculation_difference_l2034_203408

def correct_calculation : ℤ := 12 - (3 + 2) * 2

def incorrect_calculation : ℤ := 12 - 3 + 2 * 2

theorem calculation_difference :
  correct_calculation - incorrect_calculation = -11 := by
  sorry

end calculation_difference_l2034_203408


namespace squirrel_problem_l2034_203466

/-- Theorem: Given the conditions of the squirrel problem, prove the original number of squirrels on each tree. -/
theorem squirrel_problem (s b j : ℕ) : 
  s + b + j = 34 ∧ 
  b + 7 = j + s - 7 ∧ 
  b + 12 = 2 * j → 
  s = 13 ∧ b = 10 ∧ j = 11 := by
sorry

end squirrel_problem_l2034_203466


namespace line_parallel_to_plane_no_intersection_l2034_203457

-- Define a structure for a 3D space
structure Space3D where
  -- Add any necessary fields

-- Define a line in 3D space
structure Line where
  -- Add any necessary fields

-- Define a plane in 3D space
structure Plane where
  -- Add any necessary fields

-- Define parallelism between a line and a plane
def parallel (l : Line) (p : Plane) : Prop := sorry

-- Define intersection between two lines
def intersect (l1 l2 : Line) : Prop := sorry

-- Define a function to get a line in a plane
def line_in_plane (p : Plane) : Line := sorry

-- Theorem statement
theorem line_parallel_to_plane_no_intersection 
  (a : Line) (α : Plane) : 
  parallel a α → ∀ l : Line, (∃ p : Plane, l = line_in_plane p) → ¬(intersect a l) := by
  sorry

end line_parallel_to_plane_no_intersection_l2034_203457


namespace no_integers_satisfy_conditions_l2034_203478

def f (i : ℕ) : ℕ := 1 + i^(1/3) + i

theorem no_integers_satisfy_conditions :
  ¬∃ i : ℕ, 1 ≤ i ∧ i ≤ 3000 ∧ (∃ m : ℕ, i = m^3) ∧ f i = 1 + i^(1/3) + i :=
by sorry

end no_integers_satisfy_conditions_l2034_203478


namespace sin_240_degrees_l2034_203436

theorem sin_240_degrees : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l2034_203436


namespace santa_gift_combinations_l2034_203455

theorem santa_gift_combinations (n : ℤ) : 
  ∃ k : ℤ, n^5 - n = 30 * k := by sorry

end santa_gift_combinations_l2034_203455


namespace mean_of_fractions_l2034_203463

theorem mean_of_fractions (a b : ℚ) (ha : a = 2/3) (hb : b = 4/9) :
  (a + b) / 2 = 5/9 := by
  sorry

end mean_of_fractions_l2034_203463


namespace words_per_page_l2034_203444

theorem words_per_page (total_pages : ℕ) (max_words_per_page : ℕ) (total_words_mod : ℕ) :
  total_pages = 154 →
  max_words_per_page = 120 →
  total_words_mod = 250 →
  ∃ (words_per_page : ℕ),
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % 227 = total_words_mod % 227 ∧
    words_per_page = 49 := by
  sorry

end words_per_page_l2034_203444


namespace beads_taken_out_l2034_203415

theorem beads_taken_out (green brown red left : ℕ) : 
  green = 1 → brown = 2 → red = 3 → left = 4 → 
  (green + brown + red) - left = 2 := by
  sorry

end beads_taken_out_l2034_203415


namespace root_implies_q_equals_four_l2034_203496

theorem root_implies_q_equals_four (p q : ℝ) : 
  (Complex.I * Real.sqrt 3 + 1) ^ 2 + p * (Complex.I * Real.sqrt 3 + 1) + q = 0 → q = 4 := by
sorry

end root_implies_q_equals_four_l2034_203496


namespace sum_of_square_and_triangular_l2034_203441

theorem sum_of_square_and_triangular (k : ℕ) :
  let Sₖ := (6 * 10^k - 1) * 10^(k+2) + 5 * 10^(k+1) + 1
  let n := 2 * 10^(k+1) - 1
  Sₖ = n^2 + n * (n + 1) / 2 := by
  sorry

end sum_of_square_and_triangular_l2034_203441


namespace cylinder_volume_ratio_l2034_203401

/-- The ratio of the larger volume to the smaller volume of cylinders formed by rolling a 6 × 10 rectangle -/
theorem cylinder_volume_ratio : 
  let rectangle_width : ℝ := 6
  let rectangle_length : ℝ := 10
  let cylinder1_volume := π * (rectangle_width / (2 * π))^2 * rectangle_length
  let cylinder2_volume := π * (rectangle_length / (2 * π))^2 * rectangle_width
  max cylinder1_volume cylinder2_volume / min cylinder1_volume cylinder2_volume = 5 / 3 := by
sorry

end cylinder_volume_ratio_l2034_203401


namespace restaurant_outdoor_section_area_l2034_203413

/-- The area of a rectangular section with width 7 feet and length 5 feet is 35 square feet. -/
theorem restaurant_outdoor_section_area :
  let width : ℝ := 7
  let length : ℝ := 5
  width * length = 35 := by sorry

end restaurant_outdoor_section_area_l2034_203413


namespace packaging_waste_exceeds_target_l2034_203464

/-- The year when the packaging waste exceeds 40 million tons -/
def exceed_year : ℕ := 2021

/-- The initial packaging waste in 2015 (in million tons) -/
def initial_waste : ℝ := 4

/-- The annual growth rate of packaging waste -/
def growth_rate : ℝ := 0.5

/-- The target waste amount to exceed (in million tons) -/
def target_waste : ℝ := 40

/-- Function to calculate the waste amount after n years -/
def waste_after_years (n : ℕ) : ℝ :=
  initial_waste * (1 + growth_rate) ^ n

theorem packaging_waste_exceeds_target :
  waste_after_years (exceed_year - 2015) > target_waste ∧
  ∀ y : ℕ, y < exceed_year - 2015 → waste_after_years y ≤ target_waste :=
by sorry

end packaging_waste_exceeds_target_l2034_203464


namespace prom_expenses_james_prom_expenses_l2034_203499

theorem prom_expenses (num_people : ℕ) (ticket_cost dinner_cost : ℚ) 
  (tip_percentage : ℚ) (limo_hours : ℕ) (limo_cost_per_hour : ℚ) 
  (tuxedo_rental : ℚ) : ℚ :=
  let total_ticket_cost := num_people * ticket_cost
  let total_dinner_cost := num_people * dinner_cost
  let dinner_tip := total_dinner_cost * tip_percentage
  let total_limo_cost := limo_hours * limo_cost_per_hour
  total_ticket_cost + total_dinner_cost + dinner_tip + total_limo_cost + tuxedo_rental

theorem james_prom_expenses : 
  prom_expenses 4 100 120 0.3 8 80 150 = 1814 := by
  sorry

end prom_expenses_james_prom_expenses_l2034_203499


namespace smallest_root_of_g_l2034_203439

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 14 * x^2 + 4

-- State the theorem
theorem smallest_root_of_g :
  ∃ (r : ℝ), g r = 0 ∧ r = -1 ∧ ∀ (x : ℝ), g x = 0 → x ≥ -1 := by
  sorry

end smallest_root_of_g_l2034_203439


namespace p_sufficient_not_necessary_for_q_l2034_203482

/-- Proposition p: am² < bm² -/
def p (a b m : ℝ) : Prop := a * m^2 < b * m^2

/-- Proposition q: a < b -/
def q (a b : ℝ) : Prop := a < b

/-- p is sufficient but not necessary for q -/
theorem p_sufficient_not_necessary_for_q :
  (∀ a b m : ℝ, p a b m → q a b) ∧
  ¬(∀ a b : ℝ, q a b → ∀ m : ℝ, p a b m) :=
sorry

end p_sufficient_not_necessary_for_q_l2034_203482


namespace balanced_quadruple_inequality_l2034_203498

/-- A quadruple of real numbers is balanced if the sum of its elements
    equals the sum of their squares. -/
def IsBalanced (a b c d : ℝ) : Prop :=
  a + b + c + d = a^2 + b^2 + c^2 + d^2

/-- For any positive real number x greater than or equal to 3/2,
    the product (x - a)(x - b)(x - c)(x - d) is non-negative
    for all balanced quadruples (a, b, c, d). -/
theorem balanced_quadruple_inequality (x : ℝ) (hx : x > 0) (hx_ge : x ≥ 3/2) :
  ∀ a b c d : ℝ, IsBalanced a b c d →
  (x - a) * (x - b) * (x - c) * (x - d) ≥ 0 := by
  sorry

end balanced_quadruple_inequality_l2034_203498


namespace length_of_GH_l2034_203471

-- Define the lengths of the segments
def AB : ℝ := 11
def CD : ℝ := 5
def FE : ℝ := 13

-- Define the length of GH as the sum of AB, CD, and FE
def GH : ℝ := AB + CD + FE

-- Theorem statement
theorem length_of_GH : GH = 29 := by sorry

end length_of_GH_l2034_203471


namespace min_shading_for_symmetry_l2034_203443

/-- Represents a triangular figure with some shaded triangles -/
structure TriangularFigure where
  total_triangles : Nat
  shaded_triangles : Nat
  h_shaded_le_total : shaded_triangles ≤ total_triangles

/-- Calculates the minimum number of additional triangles to shade for axial symmetry -/
def min_additional_shading (figure : TriangularFigure) : Nat :=
  sorry

/-- Theorem stating the minimum additional shading for the given problem -/
theorem min_shading_for_symmetry (figure : TriangularFigure) 
  (h_total : figure.total_triangles = 54)
  (h_some_shaded : figure.shaded_triangles > 0)
  (h_not_all_shaded : figure.shaded_triangles < 54) :
  min_additional_shading figure = 6 :=
sorry

end min_shading_for_symmetry_l2034_203443


namespace range_of_m_l2034_203481

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - m| < 5) ↔ -2 < m ∧ m < 8 := by
sorry

end range_of_m_l2034_203481


namespace probability_multiple_6_or_8_l2034_203412

def is_multiple_of_6_or_8 (n : ℕ) : Bool :=
  n % 6 = 0 || n % 8 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_6_or_8 |>.length

theorem probability_multiple_6_or_8 :
  count_multiples 100 / 100 = 6 / 25 := by
  sorry

end probability_multiple_6_or_8_l2034_203412


namespace fourteenth_root_of_unity_l2034_203440

theorem fourteenth_root_of_unity (n : ℕ) (hn : n ≤ 13) : 
  (Complex.tan (Real.pi / 7) + Complex.I) / (Complex.tan (Real.pi / 7) - Complex.I) = 
  Complex.exp (Complex.I * (5 * Real.pi / 7)) :=
sorry

end fourteenth_root_of_unity_l2034_203440


namespace lunch_spending_l2034_203449

theorem lunch_spending (total : ℝ) (difference : ℝ) (friend_spent : ℝ) : 
  total = 72 → difference = 11 → friend_spent = total / 2 + difference / 2 → friend_spent = 41.5 := by
  sorry

end lunch_spending_l2034_203449


namespace complement_of_union_is_four_l2034_203447

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3}
def N : Set ℕ := {2, 5}

theorem complement_of_union_is_four :
  (M ∪ N)ᶜ = {4} :=
by sorry

end complement_of_union_is_four_l2034_203447


namespace min_value_sum_reciprocals_l2034_203487

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 4 / b) ≥ 9 :=
sorry

end min_value_sum_reciprocals_l2034_203487


namespace conic_is_parabola_l2034_203421

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  |x - 3| = Real.sqrt ((y + 4)^2 + x^2)

-- Define what it means for an equation to describe a parabola
def describes_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x y, f x y ↔ y = a * x^2 + b * x + c

-- Theorem statement
theorem conic_is_parabola : describes_parabola conic_equation := by
  sorry

end conic_is_parabola_l2034_203421


namespace c_d_not_dine_city_center_l2034_203426

-- Define the participants
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define locations
inductive Location : Type
| CityCenter : Location
| NearAHome : Location

-- Define the dining relation
def dines_together (p1 p2 : Person) (l : Location) : Prop := sorry

-- Define participation in dining
def participates (p : Person) : Prop := sorry

-- Condition 1: Only if A participates, B and C will dine together
axiom cond1 : ∀ (l : Location), dines_together Person.B Person.C l → participates Person.A

-- Condition 2: A only dines at restaurants near their home
axiom cond2 : ∀ (p : Person) (l : Location), 
  dines_together Person.A p l → l = Location.NearAHome

-- Condition 3: Only if B participates, D will go to the restaurant to dine
axiom cond3 : ∀ (p : Person) (l : Location), 
  dines_together Person.D p l → participates Person.B

-- Theorem to prove
theorem c_d_not_dine_city_center : 
  ¬(dines_together Person.C Person.D Location.CityCenter) :=
sorry

end c_d_not_dine_city_center_l2034_203426


namespace carnival_wait_time_l2034_203427

/-- Carnival Ride Wait Time Problem -/
theorem carnival_wait_time (total_time roller_coaster_wait giant_slide_wait : ℕ)
  (roller_coaster_rides tilt_a_whirl_rides giant_slide_rides : ℕ)
  (h1 : total_time = 4 * 60)
  (h2 : roller_coaster_wait = 30)
  (h3 : giant_slide_wait = 15)
  (h4 : roller_coaster_rides = 4)
  (h5 : tilt_a_whirl_rides = 1)
  (h6 : giant_slide_rides = 4) :
  ∃ tilt_a_whirl_wait : ℕ,
    total_time = roller_coaster_wait * roller_coaster_rides +
                 tilt_a_whirl_wait * tilt_a_whirl_rides +
                 giant_slide_wait * giant_slide_rides ∧
    tilt_a_whirl_wait = 60 :=
by sorry

end carnival_wait_time_l2034_203427


namespace max_value_condition_l2034_203459

/-- 
Given that x and y are real numbers, prove that when 2005 - (x + y)^2 takes its maximum value, x = -y.
-/
theorem max_value_condition (x y : ℝ) : 
  (∀ a b : ℝ, 2005 - (x + y)^2 ≥ 2005 - (a + b)^2) → x = -y := by
sorry

end max_value_condition_l2034_203459


namespace distributor_profit_percentage_l2034_203402

/-- Proves that the distributor's profit percentage is 65% given the specified conditions -/
theorem distributor_profit_percentage
  (commission_rate : Real)
  (producer_price : Real)
  (final_price : Real)
  (h1 : commission_rate = 0.2)
  (h2 : producer_price = 15)
  (h3 : final_price = 19.8) :
  (((final_price / (1 - commission_rate)) - producer_price) / producer_price) * 100 = 65 := by
  sorry

end distributor_profit_percentage_l2034_203402


namespace b_investment_l2034_203490

/-- Proves that B's investment is Rs. 12000 given the conditions of the problem -/
theorem b_investment (a_investment b_investment c_investment : ℝ)
  (b_profit : ℝ) (profit_difference : ℝ) :
  a_investment = 8000 →
  c_investment = 12000 →
  b_profit = 3000 →
  profit_difference = 1199.9999999999998 →
  (a_investment / b_investment) * b_profit =
    (c_investment / b_investment) * b_profit - profit_difference →
  b_investment = 12000 := by
  sorry

end b_investment_l2034_203490


namespace yeongju_shortest_wire_l2034_203454

-- Define the wire lengths in centimeters
def suzy_length : ℝ := 9.8
def yeongju_length : ℝ := 8.9
def youngho_length : ℝ := 9.3

-- Define the conversion factor from cm to mm
def cm_to_mm : ℝ := 10

-- Theorem to prove Yeongju has the shortest wire
theorem yeongju_shortest_wire :
  let suzy_mm := suzy_length * cm_to_mm
  let yeongju_mm := yeongju_length * cm_to_mm
  let youngho_mm := youngho_length * cm_to_mm
  yeongju_mm < suzy_mm ∧ yeongju_mm < youngho_mm :=
by sorry

end yeongju_shortest_wire_l2034_203454


namespace complex_magnitude_product_l2034_203422

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 2 - 5 * Complex.I) * (2 * Real.sqrt 3 + 2 * Complex.I)) = 4 * Real.sqrt 43 := by
sorry

end complex_magnitude_product_l2034_203422


namespace composite_sum_of_powers_l2034_203489

theorem composite_sum_of_powers (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ (a^1984 + b^1984 + c^1984 + d^1984 = m * n) := by
  sorry

end composite_sum_of_powers_l2034_203489


namespace cost_of_700_pieces_l2034_203458

/-- The cost function for gum pieces -/
def gum_cost (pieces : ℕ) : ℚ :=
  if pieces ≤ 500 then
    pieces / 100
  else
    5 + (pieces - 500) * 8 / 1000

/-- Theorem stating the cost of 700 pieces of gum -/
theorem cost_of_700_pieces : gum_cost 700 = 33/5 := by
  sorry

end cost_of_700_pieces_l2034_203458


namespace moles_of_CH3COOH_l2034_203410

-- Define the chemical reaction
structure Reaction where
  reactant1 : ℝ  -- moles of CH3COOH
  reactant2 : ℝ  -- moles of NaOH
  product1  : ℝ  -- moles of NaCH3COO
  product2  : ℝ  -- moles of H2O

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.reactant1 = r.reactant2 ∧ r.reactant1 = r.product1 ∧ r.reactant1 = r.product2

-- Theorem statement
theorem moles_of_CH3COOH (r : Reaction) 
  (h1 : r.reactant2 = 1)  -- 1 mole of NaOH is used
  (h2 : r.product1 = 1)   -- 1 mole of NaCH3COO is formed
  (h3 : balanced_equation r)  -- The reaction follows the balanced equation
  : r.reactant1 = 1 :=  -- The number of moles of CH3COOH combined is 1
by sorry

end moles_of_CH3COOH_l2034_203410


namespace no_solution_exists_l2034_203480

theorem no_solution_exists : ¬∃ (k t : ℕ), 
  (1 ≤ k ∧ k ≤ 9) ∧ 
  (1 ≤ t ∧ t ≤ 9) ∧ 
  (808 + 10 * k) - (800 + 88 * k) = 1606 + 10 * t :=
by sorry

end no_solution_exists_l2034_203480


namespace intersection_value_l2034_203424

theorem intersection_value (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 2, a^2 + 4}
  (A ∩ B = {3}) → a = 1 :=
by sorry

end intersection_value_l2034_203424


namespace cube_surface_area_from_volume_l2034_203497

theorem cube_surface_area_from_volume (V : ℝ) (s : ℝ) (SA : ℝ) : 
  V = 729 → V = s^3 → SA = 6 * s^2 → SA = 486 := by
  sorry

end cube_surface_area_from_volume_l2034_203497


namespace sibling_ages_sum_l2034_203431

theorem sibling_ages_sum (a b c : ℕ+) : 
  a < b → b < c → a * b * c = 72 → a + b + c = 13 := by sorry

end sibling_ages_sum_l2034_203431


namespace five_toppings_from_eight_l2034_203445

theorem five_toppings_from_eight (n m : ℕ) (hn : n = 8) (hm : m = 5) :
  Nat.choose n m = 56 := by
  sorry

end five_toppings_from_eight_l2034_203445


namespace eulers_formula_two_power_inequality_l2034_203434

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Statement 1: Euler's formula
theorem eulers_formula (x : ℝ) : Complex.exp (i * x) = Complex.cos x + i * Complex.sin x := by sorry

-- Statement 2: Inequality for 2^x
theorem two_power_inequality (x : ℝ) (h : x ≥ 0) : 
  (2 : ℝ) ^ x ≥ 1 + x * Real.log 2 + (x * Real.log 2)^2 / 2 := by sorry

end eulers_formula_two_power_inequality_l2034_203434


namespace blue_ball_probability_l2034_203403

/-- The probability of selecting 3 blue balls from a jar containing 6 red and 4 blue balls -/
theorem blue_ball_probability (red_balls blue_balls selected : ℕ) 
  (h1 : red_balls = 6)
  (h2 : blue_balls = 4)
  (h3 : selected = 3) :
  (Nat.choose blue_balls selected) / (Nat.choose (red_balls + blue_balls) selected) = 1 / 30 := by
  sorry

end blue_ball_probability_l2034_203403


namespace no_integer_with_five_divisors_sum_square_l2034_203495

theorem no_integer_with_five_divisors_sum_square : ¬ ∃ (n : ℕ+), 
  (∃ (d₁ d₂ d₃ d₄ d₅ : ℕ+), 
    (d₁ < d₂) ∧ (d₂ < d₃) ∧ (d₃ < d₄) ∧ (d₄ < d₅) ∧
    (d₁ ∣ n) ∧ (d₂ ∣ n) ∧ (d₃ ∣ n) ∧ (d₄ ∣ n) ∧ (d₅ ∣ n) ∧
    (∀ (d : ℕ+), d ∣ n → d ≥ d₅ ∨ d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄)) ∧
  (∃ (x : ℕ), (d₁ : ℕ)^2 + (d₂ : ℕ)^2 + (d₃ : ℕ)^2 + (d₄ : ℕ)^2 + (d₅ : ℕ)^2 = x^2) :=
by sorry

end no_integer_with_five_divisors_sum_square_l2034_203495


namespace arithmetic_calculation_l2034_203491

theorem arithmetic_calculation : 2011 - (9 * 11 * 11 + 9 * 9 * 11 - 9 * 11) = 130 := by
  sorry

end arithmetic_calculation_l2034_203491


namespace hourly_wage_calculation_l2034_203430

/-- The hourly wage in dollars -/
def hourly_wage : ℝ := 12.5

/-- The number of hours worked per week -/
def hours_worked : ℝ := 40

/-- The pay per widget in dollars -/
def pay_per_widget : ℝ := 0.16

/-- The number of widgets produced per week -/
def widgets_produced : ℝ := 1250

/-- The total earnings for the week in dollars -/
def total_earnings : ℝ := 700

theorem hourly_wage_calculation :
  hourly_wage * hours_worked + pay_per_widget * widgets_produced = total_earnings :=
by sorry

end hourly_wage_calculation_l2034_203430


namespace zacks_marbles_l2034_203467

theorem zacks_marbles (n : ℕ) : 
  (∃ k : ℕ, n = 3 * k + 5) → 
  (n = 3 * 20 + 5) → 
  n = 65 := by
sorry

end zacks_marbles_l2034_203467


namespace hotel_room_charge_comparison_l2034_203465

theorem hotel_room_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.25 * R) 
  (h2 : P = G - 0.10 * G) : 
  R = 1.2 * G := by
  sorry

end hotel_room_charge_comparison_l2034_203465


namespace unique_quadratic_solution_l2034_203435

theorem unique_quadratic_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_unique : ∃! x, (5*a + 2*b)*x^2 + a*x + b = 0) : 
  ∃ x, (5*a + 2*b)*x^2 + a*x + b = 0 ∧ x = 5/2 := by
  sorry

end unique_quadratic_solution_l2034_203435


namespace fencing_cost_per_meter_l2034_203400

/-- Proves that for a rectangular plot with given dimensions and total fencing cost,
    the cost per meter of fencing is as calculated. -/
theorem fencing_cost_per_meter
  (length : ℝ) (breadth : ℝ) (total_cost : ℝ)
  (h1 : length = 56)
  (h2 : breadth = 44)
  (h3 : total_cost = 5300) :
  total_cost / (2 * (length + breadth)) = 26.5 := by
  sorry

end fencing_cost_per_meter_l2034_203400


namespace tangent_points_x_coordinate_sum_l2034_203485

/-- Parabola struct representing x^2 = 2py -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem stating the relationship between x-coordinates of tangent points and the point on y = -2p -/
theorem tangent_points_x_coordinate_sum (para : Parabola) (M A B : Point) :
  A.y = A.x^2 / (2 * para.p) →  -- A is on the parabola
  B.y = B.x^2 / (2 * para.p) →  -- B is on the parabola
  M.y = -2 * para.p →  -- M is on the line y = -2p
  (A.y - M.y) / (A.x - M.x) = A.x / para.p →  -- MA is tangent to the parabola
  (B.y - M.y) / (B.x - M.x) = B.x / para.p →  -- MB is tangent to the parabola
  A.x + B.x = 2 * M.x := by
  sorry

end tangent_points_x_coordinate_sum_l2034_203485


namespace molly_sunday_swim_l2034_203420

/-- Represents the distance Molly swam on Sunday -/
def sunday_swim (saturday_swim total_swim : ℕ) : ℕ :=
  total_swim - saturday_swim

/-- Proves that Molly swam 28 meters on Sunday -/
theorem molly_sunday_swim :
  let saturday_swim : ℕ := 45
  let total_swim : ℕ := 73
  let pool_length : ℕ := 25
  sunday_swim saturday_swim total_swim = 28 := by
  sorry

end molly_sunday_swim_l2034_203420


namespace arithmetic_mean_after_removal_l2034_203423

theorem arithmetic_mean_after_removal (s : Finset ℕ) (a : ℕ → ℝ) :
  Finset.card s = 75 →
  (Finset.sum s a) / 75 = 60 →
  72 ∈ s →
  48 ∈ s →
  let s' := s.erase 72 ∩ s.erase 48
  (Finset.sum s' a) / 73 = 60 := by
sorry

end arithmetic_mean_after_removal_l2034_203423
