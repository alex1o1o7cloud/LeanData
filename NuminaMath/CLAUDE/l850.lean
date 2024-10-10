import Mathlib

namespace line_intercept_theorem_l850_85099

/-- Given a line ax - 6y - 12a = 0 where a ≠ 0, if its x-intercept is three times its y-intercept, then a = -2 -/
theorem line_intercept_theorem (a : ℝ) (h1 : a ≠ 0) : 
  (∃ x y : ℝ, a * x - 6 * y - 12 * a = 0 ∧ 
   x = 3 * y ∧ 
   (∀ x' y' : ℝ, a * x' - 6 * y' - 12 * a = 0 → (x' = 0 ∨ y' = 0) → (x' = x ∨ y' = y))) → 
  a = -2 := by
sorry

end line_intercept_theorem_l850_85099


namespace hosing_time_is_10_minutes_l850_85086

def dog_cleaning_time (num_shampoos : ℕ) (time_per_shampoo : ℕ) (total_cleaning_time : ℕ) : ℕ :=
  total_cleaning_time - (num_shampoos * time_per_shampoo)

theorem hosing_time_is_10_minutes :
  dog_cleaning_time 3 15 55 = 10 := by
  sorry

end hosing_time_is_10_minutes_l850_85086


namespace perpendicular_bisector_of_intersecting_circles_l850_85036

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*y - 27 = 0

-- Define the line
def L (x y : ℝ) : Prop := x + y - 3 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersecting_circles :
  ∃ (A B : ℝ × ℝ), 
    C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ 
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧ 
    A ≠ B ∧
    L ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧
    (B.2 - A.2) * ((A.1 + B.1) / 2 - A.1) = (A.1 - B.1) * ((A.2 + B.2) / 2 - A.2) :=
by sorry


end perpendicular_bisector_of_intersecting_circles_l850_85036


namespace marias_test_scores_l850_85060

/-- Maria's test scores problem -/
theorem marias_test_scores 
  (score1 score2 score3 : ℝ) 
  (h1 : (score1 + score2 + score3 + 100) / 4 = 85) :
  score1 + score2 + score3 = 240 := by
  sorry

end marias_test_scores_l850_85060


namespace clock_shows_four_fifty_l850_85098

/-- Represents a clock hand --/
inductive ClockHand
| A
| B
| C

/-- Represents the position of a clock hand --/
structure HandPosition where
  hand : ClockHand
  exactHourMarker : Bool

/-- Represents a clock with three hands --/
structure Clock where
  hands : List HandPosition
  handsEqualLength : Bool

/-- Theorem stating that given the specific clock configuration, the time shown is 4:50 --/
theorem clock_shows_four_fifty (c : Clock) 
  (h1 : c.handsEqualLength = true)
  (h2 : c.hands.length = 3)
  (h3 : ∃ h ∈ c.hands, h.hand = ClockHand.A ∧ h.exactHourMarker = true)
  (h4 : ∃ h ∈ c.hands, h.hand = ClockHand.B ∧ h.exactHourMarker = true)
  (h5 : ∃ h ∈ c.hands, h.hand = ClockHand.C ∧ h.exactHourMarker = false) :
  ∃ (hour : Nat) (minute : Nat), hour = 4 ∧ minute = 50 := by
  sorry


end clock_shows_four_fifty_l850_85098


namespace intersection_of_planes_l850_85046

-- Define the two planes
def plane1 (x y z : ℝ) : Prop := 3*x + 4*y - 2*z = 5
def plane2 (x y z : ℝ) : Prop := 2*x + 3*y - z = 3

-- Define the line of intersection
def intersection_line (x y z : ℝ) : Prop :=
  (x - 3) / 2 = (y + 1) / (-1) ∧ (y + 1) / (-1) = z / 1

-- Theorem statement
theorem intersection_of_planes :
  ∀ x y z : ℝ, plane1 x y z ∧ plane2 x y z → intersection_line x y z :=
by sorry

end intersection_of_planes_l850_85046


namespace yellow_given_popped_prob_l850_85017

-- Define the probabilities of kernel colors in the bag
def white_prob : ℚ := 1/2
def yellow_prob : ℚ := 1/3
def blue_prob : ℚ := 1/6

-- Define the probabilities of popping for each color
def white_pop_prob : ℚ := 2/3
def yellow_pop_prob : ℚ := 1/2
def blue_pop_prob : ℚ := 3/4

-- State the theorem
theorem yellow_given_popped_prob :
  let total_pop_prob := white_prob * white_pop_prob + yellow_prob * yellow_pop_prob + blue_prob * blue_pop_prob
  (yellow_prob * yellow_pop_prob) / total_pop_prob = 4/23 := by
  sorry

end yellow_given_popped_prob_l850_85017


namespace expression_simplification_expression_evaluation_l850_85008

theorem expression_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) (h3 : x ≠ 0) :
  ((x - 2) / (x + 2) + 4 * x / (x^2 - 4)) / (4 * x / (x^2 - 4)) = (x^2 + 4) / (4 * x) :=
by sorry

theorem expression_evaluation :
  let x : ℝ := 1
  ((x - 2) / (x + 2) + 4 * x / (x^2 - 4)) / (4 * x / (x^2 - 4)) = 5 / 4 :=
by sorry

end expression_simplification_expression_evaluation_l850_85008


namespace set_equality_implies_m_value_l850_85089

theorem set_equality_implies_m_value (m : ℝ) :
  let A : Set ℝ := {1, 3, m^2}
  let B : Set ℝ := {1, m}
  A ∪ B = A →
  m = 0 ∨ m = 3 := by
sorry

end set_equality_implies_m_value_l850_85089


namespace ratio_a_to_b_l850_85009

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℝ  -- first term
  y : ℝ  -- second term
  b : ℝ  -- third term
  d : ℝ  -- common difference
  h1 : y = a + d  -- relation between y, a, and d
  h2 : b = a + 3 * d  -- relation between b, a, and d
  h3 : y / 2 = a + 3 * d  -- fourth term equals y/2

/-- The ratio of a to b in the given arithmetic sequence is 3/4 -/
theorem ratio_a_to_b (seq : ArithmeticSequence) : seq.a / seq.b = 3 / 4 := by
  sorry

end ratio_a_to_b_l850_85009


namespace seventh_term_ratio_l850_85042

/-- Two arithmetic sequences with sums S and T for their first n terms. -/
def ArithmeticSequences (S T : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n / T n = (5 * n + 10) / (2 * n - 1)

/-- The 7th term of an arithmetic sequence. -/
def seventhTerm (seq : ℕ → ℚ) : ℚ := seq 7

theorem seventh_term_ratio (S T : ℕ → ℚ) (h : ArithmeticSequences S T) :
  seventhTerm S / seventhTerm T = 3 / 1 := by
  sorry

end seventh_term_ratio_l850_85042


namespace quadratic_equation_roots_l850_85051

theorem quadratic_equation_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + 4*x₁ - 4 = 0) ∧ (x₂^2 + 4*x₂ - 4 = 0) :=
by
  sorry

end quadratic_equation_roots_l850_85051


namespace product_prs_is_54_l850_85018

theorem product_prs_is_54 (p r s : ℕ) : 
  3^p + 3^5 = 270 → 
  2^r + 58 = 122 → 
  7^2 + 5^s = 2504 → 
  p * r * s = 54 := by
sorry

end product_prs_is_54_l850_85018


namespace single_tool_users_count_l850_85045

/-- The number of attendants who used a pencil -/
def pencil_users : ℕ := 25

/-- The number of attendants who used a pen -/
def pen_users : ℕ := 15

/-- The number of attendants who used both pencil and pen -/
def both_users : ℕ := 10

/-- The number of attendants who used only one type of writing tool -/
def single_tool_users : ℕ := (pencil_users - both_users) + (pen_users - both_users)

theorem single_tool_users_count : single_tool_users = 20 := by
  sorry

end single_tool_users_count_l850_85045


namespace remainder_of_3_power_100_plus_5_mod_11_l850_85020

theorem remainder_of_3_power_100_plus_5_mod_11 : (3^100 + 5) % 11 = 6 := by
  sorry

end remainder_of_3_power_100_plus_5_mod_11_l850_85020


namespace parallel_lines_a_value_l850_85073

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The value of a for which the given lines are parallel -/
theorem parallel_lines_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, 2 * x + a * y + 2 = 0 ↔ a * x + (a + 4) * y - 1 = 0) ↔ (a = 4 ∨ a = -2) :=
sorry

end parallel_lines_a_value_l850_85073


namespace cubic_roots_sum_of_squares_reciprocal_l850_85097

theorem cubic_roots_sum_of_squares_reciprocal (a b c : ℝ) 
  (sum_eq : a + b + c = 12)
  (sum_prod_eq : a * b + b * c + c * a = 20)
  (prod_eq : a * b * c = -5) :
  1 / a^2 + 1 / b^2 + 1 / c^2 = 20.8 :=
by sorry

end cubic_roots_sum_of_squares_reciprocal_l850_85097


namespace regular_polygon_with_45_degree_exterior_angles_l850_85082

theorem regular_polygon_with_45_degree_exterior_angles (n : ℕ) 
  (h1 : n > 2) 
  (h2 : (360 : ℝ) / n = 45) : n = 8 := by
  sorry

end regular_polygon_with_45_degree_exterior_angles_l850_85082


namespace solve_for_m_l850_85095

/-- Given that (x, y) = (2, -3) is a solution of the equation mx + 3y = 1, prove that m = 5 -/
theorem solve_for_m (x y m : ℝ) (h1 : x = 2) (h2 : y = -3) (h3 : m * x + 3 * y = 1) : m = 5 := by
  sorry

end solve_for_m_l850_85095


namespace lunch_to_reading_time_ratio_l850_85091

theorem lunch_to_reading_time_ratio
  (book_pages : ℕ)
  (pages_per_hour : ℕ)
  (lunch_time : ℕ)
  (h1 : book_pages = 4000)
  (h2 : pages_per_hour = 250)
  (h3 : lunch_time = 4) :
  (lunch_time : ℚ) / ((book_pages : ℚ) / (pages_per_hour : ℚ)) = 1 / 4 := by
  sorry

end lunch_to_reading_time_ratio_l850_85091


namespace max_value_condition_l850_85032

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the theorem
theorem max_value_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (a + 2), f x ≤ 3) ∧ (∃ x ∈ Set.Icc 0 (a + 2), f x = 3) ↔ -2 < a ∧ a ≤ 0 :=
by sorry

end max_value_condition_l850_85032


namespace cot_thirty_degrees_l850_85039

theorem cot_thirty_degrees : 
  let cos_thirty : ℝ := Real.sqrt 3 / 2
  let sin_thirty : ℝ := 1 / 2
  let cot (θ : ℝ) : ℝ := (Real.cos θ) / (Real.sin θ)
  cot (30 * π / 180) = Real.sqrt 3 := by sorry

end cot_thirty_degrees_l850_85039


namespace twin_brothers_age_l850_85062

/-- Theorem: Age of twin brothers
  Given that the product of their ages today is 13 less than the product of their ages a year from today,
  prove that the age of twin brothers today is 6 years old.
-/
theorem twin_brothers_age (x : ℕ) : x * x + 13 = (x + 1) * (x + 1) → x = 6 := by
  sorry

end twin_brothers_age_l850_85062


namespace number_ordering_l850_85029

theorem number_ordering : (5 : ℝ) / 2 < 3 ∧ 3 < Real.sqrt 10 := by
  sorry

end number_ordering_l850_85029


namespace prob_white_ball_l850_85010

/-- Represents an urn with a certain number of black and white balls -/
structure Urn :=
  (black : ℕ)
  (white : ℕ)

/-- The probability of choosing each urn -/
def urn_choice_prob : ℚ := 1/2

/-- The two urns in the problem -/
def urn1 : Urn := ⟨2, 3⟩
def urn2 : Urn := ⟨2, 1⟩

/-- The probability of drawing a white ball from a given urn -/
def prob_white (u : Urn) : ℚ :=
  u.white / (u.black + u.white)

/-- The theorem stating the probability of drawing a white ball -/
theorem prob_white_ball : 
  urn_choice_prob * prob_white urn1 + urn_choice_prob * prob_white urn2 = 7/15 := by
  sorry

end prob_white_ball_l850_85010


namespace race_result_l850_85069

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ  -- Speed in meters per second
  time : ℝ   -- Time to complete the race in seconds

/-- The race scenario -/
def Race : Prop :=
  ∃ (A B : Runner),
    -- Total race distance is 200 meters
    A.speed * A.time = 200 ∧
    -- A's time is 33 seconds
    A.time = 33 ∧
    -- A is 35 meters ahead of B at the finish line
    A.speed * A.time - B.speed * A.time = 35 ∧
    -- B's total race time
    B.time * B.speed = 200 ∧
    -- A beats B by 7 seconds
    B.time - A.time = 7

/-- Theorem stating that given the race conditions, A beats B by 7 seconds -/
theorem race_result : Race := by sorry

end race_result_l850_85069


namespace sqrt_sum_inequality_l850_85001

theorem sqrt_sum_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
  sorry

end sqrt_sum_inequality_l850_85001


namespace city_rentals_rate_proof_l850_85080

/-- The daily rate for Safety Rent-a-Car in dollars -/
def safety_daily_rate : ℝ := 21.95

/-- The per-mile rate for Safety Rent-a-Car in dollars -/
def safety_mile_rate : ℝ := 0.19

/-- The daily rate for City Rentals in dollars -/
def city_daily_rate : ℝ := 18.95

/-- The number of miles driven -/
def miles_driven : ℝ := 150

/-- The per-mile rate for City Rentals in dollars -/
def city_mile_rate : ℝ := 0.21

theorem city_rentals_rate_proof :
  safety_daily_rate + safety_mile_rate * miles_driven =
  city_daily_rate + city_mile_rate * miles_driven :=
by sorry

end city_rentals_rate_proof_l850_85080


namespace complement_intersection_theorem_l850_85063

def U : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {2,5,8}
def B : Set Nat := {1,3,5,7}

theorem complement_intersection_theorem : 
  (U \ A) ∩ B = {1,3,7} := by sorry

end complement_intersection_theorem_l850_85063


namespace derivative_sin_at_pi_half_l850_85000

noncomputable def f (x : ℝ) : ℝ := Real.sin x

theorem derivative_sin_at_pi_half :
  deriv f (π / 2) = 0 := by sorry

end derivative_sin_at_pi_half_l850_85000


namespace inverse_mod_101_l850_85033

theorem inverse_mod_101 (h : (7⁻¹ : ZMod 101) = 55) : (49⁻¹ : ZMod 101) = 96 := by
  sorry

end inverse_mod_101_l850_85033


namespace polly_hungry_tweet_rate_l850_85026

def happy_tweets_per_minute : ℕ := 18
def mirror_tweets_per_minute : ℕ := 45
def duration_per_state : ℕ := 20
def total_tweets : ℕ := 1340

def hungry_tweets_per_minute : ℕ := 4

theorem polly_hungry_tweet_rate :
  happy_tweets_per_minute * duration_per_state +
  hungry_tweets_per_minute * duration_per_state +
  mirror_tweets_per_minute * duration_per_state = total_tweets :=
by sorry

end polly_hungry_tweet_rate_l850_85026


namespace knitting_productivity_comparison_l850_85021

/-- Represents a knitter with their working time and break time -/
structure Knitter where
  workTime : ℕ
  breakTime : ℕ

/-- Calculates the total cycle time for a knitter -/
def cycleTime (k : Knitter) : ℕ := k.workTime + k.breakTime

/-- Calculates the number of complete cycles in a given time -/
def completeCycles (k : Knitter) (totalTime : ℕ) : ℕ :=
  totalTime / cycleTime k

/-- Calculates the total working time within a given time period -/
def totalWorkTime (k : Knitter) (totalTime : ℕ) : ℕ :=
  completeCycles k totalTime * k.workTime

theorem knitting_productivity_comparison : 
  let girl1 : Knitter := ⟨5, 1⟩
  let girl2 : Knitter := ⟨7, 1⟩
  let commonBreakTime := lcm (cycleTime girl1) (cycleTime girl2)
  totalWorkTime girl1 commonBreakTime * 21 = totalWorkTime girl2 commonBreakTime * 20 := by
  sorry

end knitting_productivity_comparison_l850_85021


namespace total_donation_l850_85043

def cassandra_pennies : ℕ := 5000
def james_pennies : ℕ := cassandra_pennies - 276
def stephanie_pennies : ℕ := 2 * james_pennies

theorem total_donation :
  cassandra_pennies + james_pennies + stephanie_pennies = 19172 :=
by sorry

end total_donation_l850_85043


namespace isosceles_triangle_l850_85081

theorem isosceles_triangle (A B C : ℝ) (hsum : A + B + C = π) 
  (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) :
  A = B :=
sorry

end isosceles_triangle_l850_85081


namespace peter_calories_l850_85041

/-- Represents the number of calories Peter wants to eat -/
def calories_wanted (chip_calories : ℕ) (chips_per_bag : ℕ) (bag_cost : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent / bag_cost) * chips_per_bag * chip_calories

/-- Proves that Peter wants to eat 480 calories worth of chips -/
theorem peter_calories : calories_wanted 10 24 2 4 = 480 := by
  sorry

end peter_calories_l850_85041


namespace hadley_walk_back_home_l850_85055

/-- The distance Hadley walked back home -/
def distance_back_home (distance_to_grocery : ℝ) (distance_to_pet : ℝ) (total_distance : ℝ) : ℝ :=
  total_distance - (distance_to_grocery + distance_to_pet)

/-- Theorem: Hadley walked 3 miles back home -/
theorem hadley_walk_back_home :
  let distance_to_grocery : ℝ := 2
  let distance_to_pet : ℝ := 2 - 1
  let total_distance : ℝ := 6
  distance_back_home distance_to_grocery distance_to_pet total_distance = 3 := by
sorry

end hadley_walk_back_home_l850_85055


namespace perpendicular_lines_a_value_l850_85019

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + (3^a - 3) * y = 0 → 
    ∃ k : ℝ, y = (-3 / (3^a - 3)) * x + k) →
  (∀ x y : ℝ, 2 * x - y - 3 = 0 → 
    ∃ k : ℝ, y = 2 * x + k) →
  (∀ m₁ m₂ : ℝ, m₁ * m₂ = -1 → 
    m₁ = -3 / (3^a - 3) ∧ m₂ = 2) →
  a = 2 := by
sorry

end perpendicular_lines_a_value_l850_85019


namespace lemonade_sales_profit_difference_l850_85077

/-- Lemonade sales problem -/
theorem lemonade_sales_profit_difference : 
  let katya_glasses : ℕ := 8
  let katya_price : ℚ := 3/2
  let katya_cost : ℚ := 1/2
  let ricky_glasses : ℕ := 9
  let ricky_price : ℚ := 2
  let ricky_cost : ℚ := 3/4
  let tina_price : ℚ := 3
  let tina_cost : ℚ := 1
  
  let katya_revenue := katya_glasses * katya_price
  let ricky_revenue := ricky_glasses * ricky_price
  let combined_revenue := katya_revenue + ricky_revenue
  let tina_target := 2 * combined_revenue
  
  let katya_profit := katya_revenue - (katya_glasses : ℚ) * katya_cost
  let tina_glasses := tina_target / tina_price
  let tina_profit := tina_target - tina_glasses * tina_cost
  
  tina_profit - katya_profit = 32
  := by sorry

end lemonade_sales_profit_difference_l850_85077


namespace pythagorean_triple_divisibility_l850_85022

theorem pythagorean_triple_divisibility (x y z : ℕ) (h : x^2 + y^2 = z^2) :
  3 ∣ x ∨ 3 ∣ y ∨ 3 ∣ z := by
  sorry

end pythagorean_triple_divisibility_l850_85022


namespace average_of_first_three_l850_85075

theorem average_of_first_three (A B C D : ℝ) : 
  (B + C + D) / 3 = 5 → 
  A + D = 11 → 
  D = 4 → 
  (A + B + C) / 3 = 6 := by
sorry

end average_of_first_three_l850_85075


namespace robin_initial_distance_l850_85049

/-- The distance Robin walked before realizing he forgot his bag -/
def initial_distance : ℝ := sorry

/-- The distance between Robin's house and the city center -/
def house_to_center : ℝ := 500

/-- The total distance Robin walked -/
def total_distance : ℝ := 900

theorem robin_initial_distance :
  initial_distance = 200 :=
by
  have journey_equation : 2 * initial_distance + house_to_center = total_distance := by sorry
  sorry

end robin_initial_distance_l850_85049


namespace mitch_spare_candy_bars_l850_85048

/-- Proves that Mitch wants to have 10 spare candy bars --/
theorem mitch_spare_candy_bars : 
  let bars_per_friend : ℕ := 2
  let total_bars : ℕ := 24
  let num_friends : ℕ := 7
  let spare_bars : ℕ := total_bars - (bars_per_friend * num_friends)
  spare_bars = 10 := by sorry

end mitch_spare_candy_bars_l850_85048


namespace coin_denominations_exist_l850_85083

theorem coin_denominations_exist : ∃ (coins : Finset ℕ), 
  (Finset.card coins = 12) ∧ 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6543 → 
    ∃ (representation : Multiset ℕ), 
      (Multiset.toFinset representation ⊆ coins) ∧
      (Multiset.card representation ≤ 8) ∧
      (Multiset.sum representation = n)) :=
by sorry

end coin_denominations_exist_l850_85083


namespace carmen_pets_difference_l850_85002

/-- Proves that Carmen has 14 fewer cats than dogs after giving up some cats for adoption -/
theorem carmen_pets_difference (initial_cats initial_dogs : ℕ) 
  (cats_given_up_round1 cats_given_up_round2 cats_given_up_round3 : ℕ) : 
  initial_cats = 48 →
  initial_dogs = 36 →
  cats_given_up_round1 = 6 →
  cats_given_up_round2 = 12 →
  cats_given_up_round3 = 8 →
  initial_cats - (cats_given_up_round1 + cats_given_up_round2 + cats_given_up_round3) = initial_dogs - 14 :=
by
  sorry

end carmen_pets_difference_l850_85002


namespace basketball_season_games_l850_85067

/-- The number of teams in the basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team -/
def intra_conference_games : ℕ := 2

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season -/
def total_games : ℕ := (num_teams.choose 2 * intra_conference_games) + (num_teams * non_conference_games)

theorem basketball_season_games :
  total_games = 150 := by
sorry

end basketball_season_games_l850_85067


namespace geometric_sequence_a11_l850_85076

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- Define the theorem
theorem geometric_sequence_a11
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a2a5 : a 2 * a 5 = 20)
  (h_a1a6 : a 1 + a 6 = 9) :
  a 11 = 25 / 4 :=
sorry

end geometric_sequence_a11_l850_85076


namespace gcd_abc_plus_cba_l850_85096

def is_consecutive (a b c : ℕ) : Prop := b = a + 1 ∧ c = a + 2

def abc_plus_cba (a b c : ℕ) : ℕ := 100 * a + 10 * b + c + 100 * c + 10 * b + a

theorem gcd_abc_plus_cba :
  ∀ a b c : ℕ,
  0 ≤ a ∧ a ≤ 7 →
  is_consecutive a b c →
  (∃ k : ℕ, abc_plus_cba a b c = 2 * k) ∧
  (∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℕ,
    0 ≤ a₁ ∧ a₁ ≤ 7 ∧
    0 ≤ a₂ ∧ a₂ ≤ 7 ∧
    is_consecutive a₁ b₁ c₁ ∧
    is_consecutive a₂ b₂ c₂ ∧
    Nat.gcd (abc_plus_cba a₁ b₁ c₁) (abc_plus_cba a₂ b₂ c₂) = 2) :=
by sorry

end gcd_abc_plus_cba_l850_85096


namespace project_time_allocation_l850_85084

theorem project_time_allocation (total_time research_time proposal_time : ℕ) 
  (h1 : total_time = 20)
  (h2 : research_time = 10)
  (h3 : proposal_time = 2) :
  total_time - (research_time + proposal_time) = 8 := by
  sorry

end project_time_allocation_l850_85084


namespace punch_mixture_difference_l850_85064

/-- Proves that in a mixture with a 3:5 ratio of two components, 
    where the total volume is 72 cups, the difference between 
    the volumes of the two components is 18 cups. -/
theorem punch_mixture_difference (total_volume : ℕ) 
    (ratio_a : ℕ) (ratio_b : ℕ) (difference : ℕ) : 
    total_volume = 72 → 
    ratio_a = 3 → 
    ratio_b = 5 → 
    difference = ratio_b * (total_volume / (ratio_a + ratio_b)) - 
                 ratio_a * (total_volume / (ratio_a + ratio_b)) → 
    difference = 18 := by
  sorry

end punch_mixture_difference_l850_85064


namespace vertically_opposite_angles_equal_l850_85074

/-- Two angles are vertically opposite if they are formed by two intersecting lines
    and are not adjacent to each other. -/
def vertically_opposite (α β : Real) : Prop := sorry

theorem vertically_opposite_angles_equal {α β : Real} (h : vertically_opposite α β) : α = β := by
  sorry

end vertically_opposite_angles_equal_l850_85074


namespace non_square_sequence_250th_term_l850_85088

/-- The sequence of positive integers omitting perfect squares -/
def non_square_sequence : ℕ → ℕ := sorry

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- The 250th term of the non-square sequence -/
def term_250 : ℕ := non_square_sequence 250

theorem non_square_sequence_250th_term :
  term_250 = 265 := by sorry

end non_square_sequence_250th_term_l850_85088


namespace greatest_two_digit_with_digit_product_16_l850_85093

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_16 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 16 → n ≤ 82 :=
sorry

end greatest_two_digit_with_digit_product_16_l850_85093


namespace handshakes_count_l850_85044

/-- Represents a social gathering with specific group interactions -/
structure SocialGathering where
  total_people : ℕ
  group_a : ℕ  -- People who all know each other
  group_b : ℕ  -- People who know no one
  group_c : ℕ  -- People who know exactly 15 from group_a
  h_total : total_people = group_a + group_b + group_c
  h_group_a : group_a = 25
  h_group_b : group_b = 10
  h_group_c : group_c = 5

/-- Calculates the number of handshakes in the social gathering -/
def handshakes (sg : SocialGathering) : ℕ :=
  let ab_handshakes := sg.group_b * (sg.group_a + sg.group_c)
  let b_internal_handshakes := sg.group_b * (sg.group_b - 1) / 2
  let c_handshakes := sg.group_c * (sg.group_a - 15 + sg.group_c)
  ab_handshakes + b_internal_handshakes + c_handshakes

/-- Theorem stating that the number of handshakes in the given social gathering is 420 -/
theorem handshakes_count (sg : SocialGathering) : handshakes sg = 420 := by
  sorry

#eval handshakes { total_people := 40, group_a := 25, group_b := 10, group_c := 5,
                   h_total := rfl, h_group_a := rfl, h_group_b := rfl, h_group_c := rfl }

end handshakes_count_l850_85044


namespace largest_y_coordinate_l850_85006

theorem largest_y_coordinate (x y : ℝ) : 
  (x - 3)^2 / 25 + (y - 2)^2 / 9 = 0 → y ≤ 2 := by
  sorry

end largest_y_coordinate_l850_85006


namespace sequence_general_term_l850_85030

theorem sequence_general_term (a : ℕ → ℤ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 3) :
  ∀ n : ℕ, a n = 3 * n - 1 := by
  sorry

end sequence_general_term_l850_85030


namespace expression_evaluation_l850_85066

-- Define the expression as a function
def f (x : ℚ) : ℚ := (4 + x * (4 + x) - 4^2) / (x - 4 + x^2 + 2*x)

-- State the theorem
theorem expression_evaluation :
  f (-3) = 15 / 4 := by
  sorry

end expression_evaluation_l850_85066


namespace cone_lateral_area_l850_85070

/-- The lateral area of a cone with base radius 6 cm and height 8 cm is 60π cm². -/
theorem cone_lateral_area : 
  let r : ℝ := 6  -- base radius in cm
  let h : ℝ := 8  -- height in cm
  let l : ℝ := (r^2 + h^2).sqrt  -- slant height
  let lateral_area : ℝ := π * r * l  -- formula for lateral area
  lateral_area = 60 * π :=
by sorry

end cone_lateral_area_l850_85070


namespace simplify_expression_1_simplify_expression_2_l850_85092

-- First expression
theorem simplify_expression_1 (a b : ℝ) : (1 : ℝ) * (4 * a - 2 * b) - (5 * a - 3 * b) = -a + b := by
  sorry

-- Second expression
theorem simplify_expression_2 (x : ℝ) : 2 * (2 * x^2 + 3 * x - 1) - (4 * x^2 + 2 * x - 2) = 4 * x := by
  sorry

end simplify_expression_1_simplify_expression_2_l850_85092


namespace closest_integer_to_cube_root_500_l850_85003

theorem closest_integer_to_cube_root_500 : 
  ∀ n : ℤ, |n - ⌊(500 : ℝ)^(1/3)⌋| ≥ |8 - ⌊(500 : ℝ)^(1/3)⌋| := by
  sorry

end closest_integer_to_cube_root_500_l850_85003


namespace derivative_at_x0_l850_85079

theorem derivative_at_x0 (f : ℝ → ℝ) (x₀ : ℝ) (h : Differentiable ℝ f) :
  (∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |(f (x₀ - 2*Δx) - f x₀) / Δx - 2| < ε) →
  deriv f x₀ = -1 := by
sorry

end derivative_at_x0_l850_85079


namespace log_equality_implies_ln_a_l850_85040

theorem log_equality_implies_ln_a (a : ℝ) (h : a > 0) :
  (Real.log (8 * a) / Real.log (9 * a) = Real.log (2 * a) / Real.log (3 * a)) →
  (Real.log a = (Real.log 2 * Real.log 3) / (Real.log 3 - 2 * Real.log 2)) := by
sorry

end log_equality_implies_ln_a_l850_85040


namespace swimming_club_members_l850_85023

theorem swimming_club_members :
  ∃ (j s v : ℕ),
    j > 0 ∧ s > 0 ∧ v > 0 ∧
    3 * s = 2 * j ∧
    5 * v = 2 * s ∧
    j + s + v = 58 :=
by sorry

end swimming_club_members_l850_85023


namespace impossible_arrangement_l850_85061

theorem impossible_arrangement : ¬ ∃ (a b : Fin 2005 → Fin 4010),
  (∀ i : Fin 2005, a i < b i) ∧
  (∀ i : Fin 2005, b i - a i = i.val + 1) ∧
  (∀ k : Fin 4010, ∃! i : Fin 2005, a i = k ∨ b i = k) :=
by sorry

end impossible_arrangement_l850_85061


namespace simplify_roots_l850_85034

theorem simplify_roots : (625 : ℝ)^(1/4) * (125 : ℝ)^(1/3) = 25 := by
  sorry

end simplify_roots_l850_85034


namespace dropped_student_score_l850_85015

theorem dropped_student_score
  (total_students : ℕ)
  (remaining_students : ℕ)
  (initial_average : ℚ)
  (final_average : ℚ)
  (h1 : total_students = 16)
  (h2 : remaining_students = 15)
  (h3 : initial_average = 61.5)
  (h4 : final_average = 64)
  : (total_students : ℚ) * initial_average - (remaining_students : ℚ) * final_average = 24 := by
  sorry

end dropped_student_score_l850_85015


namespace arithmetic_sequence_formula_l850_85014

/-- An arithmetic sequence with positive terms where a_1 and a_3 are roots of x^2 - 8x + 7 = 0 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  (a 1)^2 - 8*(a 1) + 7 = 0 ∧
  (a 3)^2 - 8*(a 3) + 7 = 0

/-- The general formula for the arithmetic sequence -/
def GeneralFormula (n : ℕ) : ℝ := 3 * n - 2

/-- Theorem stating that the general formula is correct for the given arithmetic sequence -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  ∀ n, a n = GeneralFormula n :=
sorry

end arithmetic_sequence_formula_l850_85014


namespace range_of_a_l850_85059

theorem range_of_a (p q : Prop) 
  (hp : ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0)
  (hq : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) :
  a ≤ -2 ∨ a = 1 := by
  sorry

end range_of_a_l850_85059


namespace video_game_pricing_l850_85016

theorem video_game_pricing (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ) :
  total_games = 16 →
  non_working_games = 8 →
  total_earnings = 56 →
  (total_earnings : ℚ) / (total_games - non_working_games : ℚ) = 7 := by
  sorry

end video_game_pricing_l850_85016


namespace not_divisible_by_2019_l850_85052

theorem not_divisible_by_2019 (n : ℕ) : ¬(2019 ∣ (n^2 + n + 2)) := by
  sorry

end not_divisible_by_2019_l850_85052


namespace s_bounds_l850_85065

theorem s_bounds (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  let s := Real.sqrt (a * b / ((b + c) * (c + a))) +
           Real.sqrt (b * c / ((c + a) * (a + b))) +
           Real.sqrt (c * a / ((a + b) * (b + c)))
  1 ≤ s ∧ s ≤ 3/2 := by
  sorry

end s_bounds_l850_85065


namespace tangent_intersection_x_coordinate_l850_85047

-- Define the circles
def circle1 : Real × Real × Real := (0, 0, 3)  -- (center_x, center_y, radius)
def circle2 : Real × Real × Real := (12, 0, 5)  -- (center_x, center_y, radius)

-- Define the theorem
theorem tangent_intersection_x_coordinate :
  ∃ (x : Real),
    x > 0 ∧  -- Intersection to the right of origin
    (let (x1, y1, r1) := circle1
     let (x2, y2, r2) := circle2
     (x - x1) / (x - x2) = r1 / r2) ∧
    x = 18 := by
  sorry


end tangent_intersection_x_coordinate_l850_85047


namespace systematic_sampling_method_l850_85054

theorem systematic_sampling_method (population_size : ℕ) (sample_size : ℕ) 
  (h1 : population_size = 102) (h2 : sample_size = 9) : 
  ∃ (excluded : ℕ) (interval : ℕ), 
    excluded = 3 ∧ 
    interval = 11 ∧ 
    (population_size - excluded) % sample_size = 0 ∧
    (population_size - excluded) / sample_size = interval :=
by sorry

end systematic_sampling_method_l850_85054


namespace winston_gas_refill_l850_85038

/-- Calculates the amount of gas needed to refill a car's tank -/
def gas_needed_to_refill (initial_gas tank_capacity gas_used_store gas_used_doctor : ℚ) : ℚ :=
  tank_capacity - (initial_gas - gas_used_store - gas_used_doctor)

/-- Proves that given the initial conditions, the amount of gas needed to refill the tank is 10 gallons -/
theorem winston_gas_refill :
  let initial_gas : ℚ := 10
  let tank_capacity : ℚ := 12
  let gas_used_store : ℚ := 6
  let gas_used_doctor : ℚ := 2
  gas_needed_to_refill initial_gas tank_capacity gas_used_store gas_used_doctor = 10 := by
  sorry


end winston_gas_refill_l850_85038


namespace square_ratio_sum_l850_85056

theorem square_ratio_sum (area_ratio : ℚ) (a b c : ℕ) : 
  area_ratio = 300 / 75 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt area_ratio →
  a + b + c = 4 := by
sorry

end square_ratio_sum_l850_85056


namespace equiv_mod_seven_l850_85031

theorem equiv_mod_seven (n : ℤ) : 0 ≤ n ∧ n ≤ 10 ∧ n ≡ -3137 [ZMOD 7] → n = 1 ∨ n = 8 := by
  sorry

end equiv_mod_seven_l850_85031


namespace divisibility_congruence_l850_85005

theorem divisibility_congruence (n : ℤ) :
  (6 ∣ (n - 4)) → (10 ∣ (n - 8)) → n ≡ -2 [ZMOD 30] := by
  sorry

end divisibility_congruence_l850_85005


namespace right_triangle_median_ratio_bound_l850_85071

theorem right_triangle_median_ratio_bound (a b c s_a s_b s_c : ℝ) 
  (h_right : c^2 = a^2 + b^2)
  (h_s_a : s_a^2 = a^2/4 + b^2)
  (h_s_b : s_b^2 = b^2/4 + a^2)
  (h_s_c : s_c = c/2)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  (s_a + s_b) / s_c ≤ Real.sqrt 10 := by
sorry

end right_triangle_median_ratio_bound_l850_85071


namespace expression_equals_20_times_10_pow_1500_l850_85012

theorem expression_equals_20_times_10_pow_1500 :
  (2^1500 + 5^1501)^2 - (2^1500 - 5^1501)^2 = 20 * 10^1500 := by
  sorry

end expression_equals_20_times_10_pow_1500_l850_85012


namespace quadratic_equation_roots_l850_85027

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 6 = 0 ∧ x = 2) → 
  (∃ x : ℝ, x^2 + k*x + 6 = 0 ∧ x = 3 ∧ k = -5) :=
by sorry

end quadratic_equation_roots_l850_85027


namespace square_difference_1001_999_l850_85035

theorem square_difference_1001_999 : 1001^2 - 999^2 = 4000 := by
  sorry

end square_difference_1001_999_l850_85035


namespace factor_y6_minus_64_l850_85057

theorem factor_y6_minus_64 (y : ℝ) : 
  y^6 - 64 = (y - 2) * (y + 2) * (y^2 + 2*y + 4) * (y^2 - 2*y + 4) := by
  sorry

end factor_y6_minus_64_l850_85057


namespace odd_periodic_function_value_l850_85085

-- Define an odd function f on ℝ
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the periodic property of f
def hasPeriod (f : ℝ → ℝ) : Prop := ∀ x, f (x + 3/2) = -f x

-- Theorem statement
theorem odd_periodic_function_value (f : ℝ → ℝ) 
  (h_odd : isOdd f) (h_period : hasPeriod f) : f (-3/2) = 0 := by
  sorry

end odd_periodic_function_value_l850_85085


namespace allan_initial_balloons_l850_85090

def balloons_problem (initial_balloons : ℕ) : Prop :=
  let total_balloons := initial_balloons + 3
  6 = total_balloons + 1

theorem allan_initial_balloons : 
  ∃ (initial_balloons : ℕ), balloons_problem initial_balloons ∧ initial_balloons = 2 := by
  sorry

end allan_initial_balloons_l850_85090


namespace parallel_lines_b_value_l850_85094

/-- Given two lines in the xy-plane, this theorem proves that if they are parallel,
    then the value of b must be 6. -/
theorem parallel_lines_b_value (b : ℝ) :
  (∀ x y, 3 * y - 3 * b = 9 * x) →
  (∀ x y, y - 2 = (b - 3) * x) →
  (∃ k : ℝ, ∀ x y, 3 * y - 3 * b = 9 * x ↔ y - 2 = k * (x - 0)) →
  b = 6 := by
  sorry

end parallel_lines_b_value_l850_85094


namespace nine_chapters_equal_distribution_l850_85025

theorem nine_chapters_equal_distribution :
  ∀ (a : ℚ) (d : ℚ),
    (5 * a + 10 * d = 5) →  -- Sum of 5 terms is 5
    (2 * a + d = 3 * a + 9 * d) →  -- Sum of first two terms equals sum of last three terms
    a = 4 / 3 := by
  sorry

end nine_chapters_equal_distribution_l850_85025


namespace jamie_peeled_24_l850_85087

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  sylvia_rate : ℕ
  jamie_rate : ℕ
  sylvia_solo_time : ℕ

/-- Calculates the number of potatoes Jamie peeled -/
def jamie_peeled (scenario : PotatoPeeling) : ℕ :=
  let sylvia_solo := scenario.sylvia_rate * scenario.sylvia_solo_time
  let remaining := scenario.total_potatoes - sylvia_solo
  let combined_rate := scenario.sylvia_rate + scenario.jamie_rate
  let combined_time := remaining / combined_rate
  scenario.jamie_rate * combined_time

/-- Theorem stating that Jamie peeled 24 potatoes -/
theorem jamie_peeled_24 (scenario : PotatoPeeling) 
    (h1 : scenario.total_potatoes = 60)
    (h2 : scenario.sylvia_rate = 4)
    (h3 : scenario.jamie_rate = 6)
    (h4 : scenario.sylvia_solo_time = 5) : 
  jamie_peeled scenario = 24 := by
  sorry

end jamie_peeled_24_l850_85087


namespace circus_revenue_l850_85024

/-- Calculates the total revenue from circus ticket sales -/
theorem circus_revenue (lower_price upper_price : ℕ) (total_tickets lower_tickets : ℕ) :
  lower_price = 30 →
  upper_price = 20 →
  total_tickets = 80 →
  lower_tickets = 50 →
  lower_price * lower_tickets + upper_price * (total_tickets - lower_tickets) = 2100 := by
sorry

end circus_revenue_l850_85024


namespace ac_length_is_18_l850_85053

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  /-- Point A -/
  A : ℝ × ℝ
  /-- Point B -/
  B : ℝ × ℝ
  /-- Point C -/
  C : ℝ × ℝ
  /-- Point D -/
  D : ℝ × ℝ
  /-- AB length is 12 -/
  ab_length : dist A B = 12
  /-- AD length is 8 -/
  ad_length : dist A D = 8
  /-- DC length is 18 -/
  dc_length : dist D C = 18
  /-- AD is perpendicular to AB -/
  ad_perp_ab : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0
  /-- ABCD is symmetric about AC -/
  symmetric_about_ac : ∃ (m : ℝ) (b : ℝ), 
    (C.2 - A.2) = m * (C.1 - A.1) ∧
    B.2 - A.2 = m * (B.1 - A.1) + b ∧
    D.2 - A.2 = -(m * (D.1 - A.1) + b)

/-- The length of AC in a SpecialQuadrilateral is 18 -/
theorem ac_length_is_18 (q : SpecialQuadrilateral) : dist q.A q.C = 18 := by
  sorry

end ac_length_is_18_l850_85053


namespace cylinder_dimensions_l850_85068

/-- Represents a cylinder formed by rotating a rectangle around one of its sides. -/
structure Cylinder where
  height : ℝ
  radius : ℝ

/-- Theorem: Given a cylinder formed by rotating a rectangle with a diagonal of 26 cm
    around one of its sides, if a perpendicular plane equidistant from the bases has
    a total surface area of 2720 cm², then the height of the cylinder is 24 cm and
    its base radius is 10 cm. -/
theorem cylinder_dimensions (c : Cylinder) :
  c.height ^ 2 + c.radius ^ 2 = 26 ^ 2 →
  8 * c.radius ^ 2 + 8 * c.radius * c.height = 2720 →
  c.height = 24 ∧ c.radius = 10 := by
  sorry

#check cylinder_dimensions

end cylinder_dimensions_l850_85068


namespace problem_statement_l850_85028

-- Define the function f(x) = ax^2 + 1
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 1

-- Define what it means for a function to pass through a point
def passes_through (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y

-- Define parallel relation for lines and planes
def parallel (α β : Set (ℝ × ℝ × ℝ)) : Prop := sorry

theorem problem_statement :
  (∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ¬(passes_through (f a) (-1) 2)) ∧
  (∀ α β m : Set (ℝ × ℝ × ℝ), 
    parallel α β → (parallel m α ↔ parallel m β)) := by sorry

end problem_statement_l850_85028


namespace equation_is_ellipse_l850_85013

-- Define the equation
def equation (x y : ℝ) : Prop :=
  4 * x^2 + y^2 - 12 * x - 2 * y + 4 = 0

-- Define what it means for the equation to represent an ellipse
def is_ellipse (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b h k : ℝ) (A B : ℝ), 
    A > 0 ∧ B > 0 ∧
    ∀ (x y : ℝ), eq x y ↔ ((x - h)^2 / A + (y - k)^2 / B = 1)

-- Theorem statement
theorem equation_is_ellipse : is_ellipse equation := by
  sorry

end equation_is_ellipse_l850_85013


namespace find_divisor_l850_85004

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 163 + remainder) :
  ∃ (divisor : ℕ), dividend = quotient * divisor + remainder ∧ divisor = 163 := by
  sorry

end find_divisor_l850_85004


namespace min_doors_for_safety_l850_85011

/-- Represents a spaceship with a given number of corridors -/
structure Spaceship :=
  (corridors : ℕ)

/-- Represents the state of doors in the spaceship -/
def DoorState := Fin 23 → Bool

/-- Checks if there exists a path from reactor to lounge -/
def hasPath (s : Spaceship) (state : DoorState) : Prop :=
  sorry -- Definition of path existence

/-- Counts the number of closed doors -/
def closedDoors (state : DoorState) : ℕ :=
  sorry -- Count of closed doors

/-- Theorem stating the minimum number of doors to close for safety -/
theorem min_doors_for_safety (s : Spaceship) :
  (s.corridors = 23) →
  (∀ (state : DoorState), closedDoors state ≥ 22 → ¬hasPath s state) ∧
  (∃ (state : DoorState), closedDoors state = 21 ∧ hasPath s state) :=
sorry

#check min_doors_for_safety

end min_doors_for_safety_l850_85011


namespace A_intersect_B_equals_open_interval_l850_85078

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x * (x - 2) < 0}
def B : Set ℝ := {x : ℝ | x - 1 > 0}

-- State the theorem
theorem A_intersect_B_equals_open_interval : A ∩ B = Set.Ioo 1 2 := by sorry

end A_intersect_B_equals_open_interval_l850_85078


namespace benjamins_speed_l850_85072

/-- Given a distance of 800 kilometers and a time of 10 hours, prove that the speed is 80 kilometers per hour. -/
theorem benjamins_speed (distance : ℝ) (time : ℝ) (h1 : distance = 800) (h2 : time = 10) :
  distance / time = 80 := by
  sorry

end benjamins_speed_l850_85072


namespace average_weight_problem_l850_85037

/-- Given the weights of three people with specific relationships, prove their average weight. -/
theorem average_weight_problem (jalen_weight ponce_weight ishmael_weight : ℕ) : 
  jalen_weight = 160 ∧ 
  ponce_weight = jalen_weight - 10 ∧ 
  ishmael_weight = ponce_weight + 20 → 
  (jalen_weight + ponce_weight + ishmael_weight) / 3 = 160 := by
  sorry


end average_weight_problem_l850_85037


namespace min_pizzas_cover_expenses_l850_85058

/-- Represents the minimum number of pizzas John must deliver to cover his expenses -/
def min_pizzas : ℕ := 1063

/-- Represents the cost of the used car -/
def car_cost : ℕ := 8000

/-- Represents the upfront maintenance cost -/
def maintenance_cost : ℕ := 500

/-- Represents the earnings per pizza delivered -/
def earnings_per_pizza : ℕ := 12

/-- Represents the gas cost per delivery -/
def gas_cost_per_delivery : ℕ := 4

/-- Represents the net earnings per pizza (earnings minus gas cost) -/
def net_earnings_per_pizza : ℕ := earnings_per_pizza - gas_cost_per_delivery

theorem min_pizzas_cover_expenses :
  (min_pizzas : ℝ) * net_earnings_per_pizza ≥ car_cost + maintenance_cost :=
sorry

end min_pizzas_cover_expenses_l850_85058


namespace cos_product_eighth_and_five_eighths_pi_l850_85007

theorem cos_product_eighth_and_five_eighths_pi :
  Real.cos (π / 8) * Real.cos (5 * π / 8) = -Real.sqrt 2 / 4 := by
  sorry

end cos_product_eighth_and_five_eighths_pi_l850_85007


namespace sine_inequality_l850_85050

theorem sine_inequality : 
  (∀ x y, x ∈ Set.Icc 0 (π/2) → y ∈ Set.Icc 0 (π/2) → x < y → Real.sin x < Real.sin y) →
  3*π/7 > 2*π/5 →
  3*π/7 ∈ Set.Icc 0 (π/2) →
  2*π/5 ∈ Set.Icc 0 (π/2) →
  Real.sin (3*π/7) > Real.sin (2*π/5) := by
  sorry

end sine_inequality_l850_85050
