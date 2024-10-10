import Mathlib

namespace similar_triangles_leg_length_l3377_337712

/-- Two similar right triangles, one with legs 12 and 9, the other with legs y and 7 -/
def similar_triangles (y : ℝ) : Prop :=
  12 / y = 9 / 7

theorem similar_triangles_leg_length :
  ∃ y : ℝ, similar_triangles y ∧ y = 84 / 9 := by
  sorry

end similar_triangles_leg_length_l3377_337712


namespace equation_solutions_count_l3377_337705

theorem equation_solutions_count :
  let f : ℝ → ℝ := λ θ => 1 - 4 * Real.sin θ + 5 * Real.cos (2 * θ)
  ∃! (solutions : Finset ℝ), 
    (∀ θ ∈ solutions, 0 < θ ∧ θ ≤ 2 * Real.pi ∧ f θ = 0) ∧
    solutions.card = 4 :=
sorry

end equation_solutions_count_l3377_337705


namespace coeff_x2y2_in_expansion_l3377_337784

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^a * y^b in (1+x)^m * (1+y)^n
def coeff (m n a b : ℕ) : ℕ := binomial m a * binomial n b

-- Theorem statement
theorem coeff_x2y2_in_expansion : coeff 3 4 2 2 = 18 := by sorry

end coeff_x2y2_in_expansion_l3377_337784


namespace hourly_wage_calculation_l3377_337723

/-- Calculates the hourly wage given the total earnings, hours worked, widgets produced, and widget bonus rate. -/
def calculate_hourly_wage (total_earnings : ℚ) (hours_worked : ℚ) (widgets_produced : ℚ) (widget_bonus_rate : ℚ) : ℚ :=
  (total_earnings - widgets_produced * widget_bonus_rate) / hours_worked

theorem hourly_wage_calculation :
  let total_earnings : ℚ := 620
  let hours_worked : ℚ := 40
  let widgets_produced : ℚ := 750
  let widget_bonus_rate : ℚ := 0.16
  calculate_hourly_wage total_earnings hours_worked widgets_produced widget_bonus_rate = 12.5 := by
sorry

#eval calculate_hourly_wage 620 40 750 0.16

end hourly_wage_calculation_l3377_337723


namespace charity_event_volunteers_l3377_337715

theorem charity_event_volunteers (n : ℕ) : 
  (n : ℚ) / 2 = (((n : ℚ) / 2 - 3) / n) * n → n / 2 = 15 :=
by
  sorry

end charity_event_volunteers_l3377_337715


namespace nursery_school_count_nursery_school_count_proof_l3377_337772

theorem nursery_school_count : ℕ → Prop :=
  fun total_students =>
    let students_4_and_older := total_students / 10
    let students_under_3 := 20
    let students_not_between_3_and_4 := 50
    students_4_and_older = students_not_between_3_and_4 - students_under_3 ∧
    total_students = 300

-- The proof of the theorem
theorem nursery_school_count_proof : ∃ n : ℕ, nursery_school_count n :=
  sorry

end nursery_school_count_nursery_school_count_proof_l3377_337772


namespace clinton_shoes_count_l3377_337727

theorem clinton_shoes_count (hats belts shoes : ℕ) : 
  hats = 5 →
  belts = hats + 2 →
  shoes = 2 * belts →
  shoes = 14 := by
sorry

end clinton_shoes_count_l3377_337727


namespace or_necessary_not_sufficient_for_and_l3377_337708

theorem or_necessary_not_sufficient_for_and (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) := by
  sorry

end or_necessary_not_sufficient_for_and_l3377_337708


namespace combined_friends_list_l3377_337798

theorem combined_friends_list (james_friends : ℕ) (susan_friends : ℕ) (maria_friends : ℕ)
  (james_john_shared : ℕ) (james_john_maria_shared : ℕ)
  (h1 : james_friends = 90)
  (h2 : susan_friends = 50)
  (h3 : maria_friends = 80)
  (h4 : james_john_shared = 35)
  (h5 : james_john_maria_shared = 10) :
  james_friends + 4 * susan_friends - james_john_shared + maria_friends - james_john_maria_shared = 325 := by
  sorry

end combined_friends_list_l3377_337798


namespace reflection_across_origin_l3377_337779

/-- Reflects a point across the origin -/
def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-2, -3)

/-- The reflected point Q -/
def Q : ℝ × ℝ := (2, 3)

theorem reflection_across_origin :
  reflect_origin P = Q := by sorry

end reflection_across_origin_l3377_337779


namespace three_balls_four_boxes_l3377_337795

theorem three_balls_four_boxes :
  (∀ n : ℕ, n ≤ 3 → n > 0 → 4 ^ n = (Fintype.card (Fin 4)) ^ n) →
  4 ^ 3 = 64 :=
by sorry

end three_balls_four_boxes_l3377_337795


namespace x_intercept_of_line_l3377_337770

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end x_intercept_of_line_l3377_337770


namespace keith_receives_144_messages_l3377_337710

/-- Represents the number of messages sent between people in a day -/
structure MessageCount where
  juan_to_laurence : ℕ
  juan_to_keith : ℕ
  laurence_to_missy : ℕ

/-- The conditions of the messaging problem -/
def messaging_problem (m : MessageCount) : Prop :=
  m.juan_to_keith = 8 * m.juan_to_laurence ∧
  m.laurence_to_missy = m.juan_to_laurence ∧
  m.laurence_to_missy = 18

/-- The theorem stating that Keith receives 144 messages from Juan -/
theorem keith_receives_144_messages (m : MessageCount) 
  (h : messaging_problem m) : m.juan_to_keith = 144 := by
  sorry

end keith_receives_144_messages_l3377_337710


namespace team_a_games_won_lost_team_b_minimum_wins_l3377_337769

/-- Represents the number of games a team plays in the tournament -/
def total_games : ℕ := 10

/-- Represents the points earned for a win -/
def win_points : ℕ := 2

/-- Represents the points earned for a loss -/
def loss_points : ℕ := 1

/-- Represents the minimum points needed to qualify for the next round -/
def qualification_points : ℕ := 15

theorem team_a_games_won_lost (points : ℕ) (h : points = 18) :
  ∃ (wins losses : ℕ), wins + losses = total_games ∧
                        wins * win_points + losses * loss_points = points ∧
                        wins = 8 ∧ losses = 2 := by sorry

theorem team_b_minimum_wins :
  ∃ (min_wins : ℕ), ∀ (wins : ℕ),
    wins * win_points + (total_games - wins) * loss_points > qualification_points →
    wins ≥ min_wins ∧
    min_wins = 6 := by sorry

end team_a_games_won_lost_team_b_minimum_wins_l3377_337769


namespace even_function_implies_a_equals_one_l3377_337792

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x^2 + (a - 1) * x + 3

-- State the theorem
theorem even_function_implies_a_equals_one :
  (∀ x : ℝ, f a x = f a (-x)) → a = 1 := by
  sorry

end even_function_implies_a_equals_one_l3377_337792


namespace hyperbola_k_range_l3377_337713

-- Define the hyperbola equation
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (2 - k) + y^2 / (k - 1) = 1

-- Define the condition for the real axis to be on the x-axis
def real_axis_on_x (k : ℝ) : Prop :=
  (2 - k > 0) ∧ (k - 1 < 0)

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, hyperbola_equation x y k ∧ real_axis_on_x k) ↔ k ∈ Set.Iio 1 :=
sorry

end hyperbola_k_range_l3377_337713


namespace september_electricity_usage_l3377_337781

theorem september_electricity_usage
  (october_usage : ℕ)
  (savings_percentage : ℚ)
  (h1 : october_usage = 1400)
  (h2 : savings_percentage = 30 / 100)
  (h3 : october_usage = (1 - savings_percentage) * september_usage) :
  september_usage = 2000 :=
sorry

end september_electricity_usage_l3377_337781


namespace a_minus_b_value_l3377_337745

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 5) (h2 : b^2 = 64) (h3 : a * b > 0) :
  a - b = 3 ∨ a - b = -3 := by
sorry

end a_minus_b_value_l3377_337745


namespace rationalize_denominator_l3377_337763

theorem rationalize_denominator :
  7 / (2 * Real.sqrt 50) = (7 * Real.sqrt 2) / 20 := by
  sorry

end rationalize_denominator_l3377_337763


namespace subtraction_result_l3377_337778

def largest_3digit_number : ℕ := 999
def smallest_5digit_number : ℕ := 10000

theorem subtraction_result : 
  smallest_5digit_number - largest_3digit_number = 9001 := by sorry

end subtraction_result_l3377_337778


namespace marble_selection_ways_l3377_337771

def total_marbles : ℕ := 9
def marbles_to_choose : ℕ := 4
def red_marbles : ℕ := 1

def choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem marble_selection_ways : 
  choose_marbles (total_marbles - red_marbles) (marbles_to_choose - red_marbles) = 56 := by
  sorry

end marble_selection_ways_l3377_337771


namespace son_work_time_l3377_337787

theorem son_work_time (man_time son_father_time : ℝ) 
  (h1 : man_time = 5)
  (h2 : son_father_time = 3) : 
  let man_rate := 1 / man_time
  let combined_rate := 1 / son_father_time
  let son_rate := combined_rate - man_rate
  1 / son_rate = 7.5 := by sorry

end son_work_time_l3377_337787


namespace largest_divisor_of_n_l3377_337706

theorem largest_divisor_of_n (n : ℕ+) (h : 50 ∣ n^2) : 5 ∣ n := by
  sorry

end largest_divisor_of_n_l3377_337706


namespace arithmetic_sequence_sum_remainder_l3377_337725

/-- Arithmetic sequence sum and remainder theorem -/
theorem arithmetic_sequence_sum_remainder
  (a : ℕ) -- First term
  (d : ℕ) -- Common difference
  (l : ℕ) -- Last term
  (h1 : a = 2)
  (h2 : d = 5)
  (h3 : l = 142)
  : (((l - a) / d + 1) * (a + l) / 2) % 20 = 8 := by
  sorry

end arithmetic_sequence_sum_remainder_l3377_337725


namespace mrs_awesome_class_size_l3377_337744

theorem mrs_awesome_class_size :
  ∀ (total_jelly_beans : ℕ) (leftover_jelly_beans : ℕ) (boy_girl_difference : ℕ),
    total_jelly_beans = 480 →
    leftover_jelly_beans = 5 →
    boy_girl_difference = 3 →
    ∃ (girls : ℕ) (boys : ℕ),
      girls + boys = 31 ∧
      boys = girls + boy_girl_difference ∧
      girls * girls + boys * boys = total_jelly_beans - leftover_jelly_beans :=
by sorry

end mrs_awesome_class_size_l3377_337744


namespace body_temperature_survey_most_suitable_for_census_l3377_337739

/-- Represents a survey option -/
inductive SurveyOption
| HeightSurvey
| TrafficRegulationsSurvey
| BodyTemperatureSurvey
| MovieViewershipSurvey

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  requiresCompleteData : Bool
  impactsSafety : Bool
  populationSize : Nat

/-- Defines what makes a survey suitable for a census -/
def suitableForCensus (c : SurveyCharacteristics) : Prop :=
  c.requiresCompleteData ∧ c.impactsSafety ∧ c.populationSize > 0

/-- Assigns characteristics to each survey option -/
def getSurveyCharacteristics : SurveyOption → SurveyCharacteristics
| SurveyOption.HeightSurvey => ⟨false, false, 1000⟩
| SurveyOption.TrafficRegulationsSurvey => ⟨false, false, 10000⟩
| SurveyOption.BodyTemperatureSurvey => ⟨true, true, 500⟩
| SurveyOption.MovieViewershipSurvey => ⟨false, false, 2000⟩

theorem body_temperature_survey_most_suitable_for_census :
  suitableForCensus (getSurveyCharacteristics SurveyOption.BodyTemperatureSurvey) ∧
  ∀ (s : SurveyOption), s ≠ SurveyOption.BodyTemperatureSurvey →
    ¬(suitableForCensus (getSurveyCharacteristics s)) :=
  sorry

end body_temperature_survey_most_suitable_for_census_l3377_337739


namespace arithmetic_sequence_sum_l3377_337707

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  isArithmeticSequence a → a 6 + a 9 + a 12 = 48 → a 8 + a 10 = 32 := by
  sorry

end arithmetic_sequence_sum_l3377_337707


namespace roof_dimensions_difference_l3377_337726

/-- Represents the dimensions of a rectangular roof side -/
structure RoofSide where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular roof side -/
def area (side : RoofSide) : ℝ := side.width * side.length

theorem roof_dimensions_difference (roof : RoofSide) 
  (h1 : roof.length = 4 * roof.width)  -- Length is 3 times longer than width
  (h2 : 2 * area roof = 588)  -- Combined area of two sides is 588
  : roof.length - roof.width = 3 * Real.sqrt (588 / 8) := by
  sorry

end roof_dimensions_difference_l3377_337726


namespace train_speed_l3377_337773

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 140)
  (h2 : bridge_length = 235)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end train_speed_l3377_337773


namespace average_of_remaining_two_l3377_337752

theorem average_of_remaining_two (total_avg : ℝ) (avg1 : ℝ) (avg2 : ℝ) :
  total_avg = 3.95 →
  avg1 = 4.2 →
  avg2 = 3.8000000000000007 →
  (6 * total_avg - 2 * avg1 - 2 * avg2) / 2 = 3.85 := by
sorry


end average_of_remaining_two_l3377_337752


namespace mary_flour_calculation_l3377_337780

/-- The number of cups of flour Mary put in -/
def flour_put_in : ℕ := 2

/-- The total number of cups of flour required by the recipe -/
def total_flour : ℕ := 10

/-- The number of cups of sugar required by the recipe -/
def sugar : ℕ := 3

/-- The additional cups of flour needed compared to sugar -/
def extra_flour : ℕ := 5

theorem mary_flour_calculation :
  flour_put_in = total_flour - (sugar + extra_flour) :=
by sorry

end mary_flour_calculation_l3377_337780


namespace min_ab_in_triangle_l3377_337756

theorem min_ab_in_triangle (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  2 * c * Real.cos B = 2 * a + b →
  (1/2) * a * b * Real.sin C = (Real.sqrt 3 / 2) * c →
  a * b ≥ 12 := by
sorry

end min_ab_in_triangle_l3377_337756


namespace book_distribution_l3377_337766

/-- The number of books -/
def num_books : ℕ := 15

/-- The number of exercise books -/
def num_exercise_books : ℕ := 26

/-- The number of students in the first scenario -/
def students_scenario1 : ℕ := (num_exercise_books / 2)

/-- The number of students in the second scenario -/
def students_scenario2 : ℕ := (num_books / 3)

theorem book_distribution :
  (students_scenario1 + 2 = num_books) ∧
  (2 * students_scenario1 = num_exercise_books) ∧
  (3 * students_scenario2 = num_books) ∧
  (5 * students_scenario2 + 1 = num_exercise_books) :=
by sorry

end book_distribution_l3377_337766


namespace third_number_proof_l3377_337765

theorem third_number_proof (sum : ℝ) (a b c : ℝ) (h : sum = a + b + c + 0.217) :
  sum - a - b - c = 0.217 :=
by sorry

end third_number_proof_l3377_337765


namespace cylinder_volume_equality_l3377_337796

theorem cylinder_volume_equality (x : ℚ) : x > 0 →
  (5 + x)^2 * 4 = 25 * (4 + x) → x = 35/4 := by
  sorry

end cylinder_volume_equality_l3377_337796


namespace cows_bought_calculation_l3377_337767

def cows_bought (initial : ℕ) (died : ℕ) (sold : ℕ) (increase : ℕ) (gift : ℕ) (final : ℕ) : ℕ :=
  final - (initial - died - sold + increase + gift)

theorem cows_bought_calculation :
  cows_bought 39 25 6 24 8 83 = 43 := by
  sorry

end cows_bought_calculation_l3377_337767


namespace fence_cost_calculation_l3377_337764

/-- The cost of building a fence around a rectangular plot -/
def fence_cost (length width length_price width_price : ℕ) : ℕ :=
  2 * (length * length_price + width * width_price)

/-- Theorem stating the total cost of building the fence -/
theorem fence_cost_calculation :
  fence_cost 35 25 60 50 = 6700 := by
  sorry

end fence_cost_calculation_l3377_337764


namespace yellow_balls_count_l3377_337741

def total_balls (a : ℕ) : ℕ := 2 + 3 + a

def probability_red (a : ℕ) : ℚ := 2 / total_balls a

theorem yellow_balls_count : ∃ a : ℕ, probability_red a = 1/3 ∧ a = 1 := by
  sorry

end yellow_balls_count_l3377_337741


namespace diagonal_cut_square_dimensions_l3377_337783

/-- Given a square with side length 10 units that is cut diagonally,
    prove that the resulting triangles have dimensions 10, 10, and 10√2 units. -/
theorem diagonal_cut_square_dimensions :
  let square_side : ℝ := 10
  let diagonal : ℝ := square_side * Real.sqrt 2
  ∀ triangle : Set (ℝ × ℝ × ℝ),
    (∃ (a b c : ℝ), triangle = {(a, b, c)} ∧
      a = square_side ∧
      b = square_side ∧
      c = diagonal) →
    triangle = {(10, 10, 10 * Real.sqrt 2)} :=
by sorry

end diagonal_cut_square_dimensions_l3377_337783


namespace potato_fries_price_l3377_337704

/-- The price of a pack of potato fries given Einstein's fundraising scenario -/
theorem potato_fries_price (total_goal : ℚ) (pizza_price : ℚ) (soda_price : ℚ)
  (pizzas_sold : ℕ) (fries_sold : ℕ) (sodas_sold : ℕ) (remaining : ℚ)
  (h1 : total_goal = 500)
  (h2 : pizza_price = 12)
  (h3 : soda_price = 2)
  (h4 : pizzas_sold = 15)
  (h5 : fries_sold = 40)
  (h6 : sodas_sold = 25)
  (h7 : remaining = 258)
  : (total_goal - remaining - (pizza_price * pizzas_sold + soda_price * sodas_sold)) / fries_sold = (3 / 10) :=
sorry

end potato_fries_price_l3377_337704


namespace zoe_water_bottles_l3377_337700

/-- The initial number of water bottles Zoe had in her fridge -/
def initial_bottles : ℕ := 42

/-- The number of bottles Zoe drank -/
def bottles_drank : ℕ := 25

/-- The number of bottles Zoe bought -/
def bottles_bought : ℕ := 30

/-- The final number of bottles Zoe has -/
def final_bottles : ℕ := 47

theorem zoe_water_bottles :
  initial_bottles - bottles_drank + bottles_bought = final_bottles :=
sorry

end zoe_water_bottles_l3377_337700


namespace mod_pow_98_50_100_l3377_337743

theorem mod_pow_98_50_100 : 98^50 % 100 = 24 := by
  sorry

end mod_pow_98_50_100_l3377_337743


namespace max_product_sum_l3377_337701

theorem max_product_sum (f g h j : ℕ) : 
  f ∈ ({5, 7, 9, 11} : Set ℕ) →
  g ∈ ({5, 7, 9, 11} : Set ℕ) →
  h ∈ ({5, 7, 9, 11} : Set ℕ) →
  j ∈ ({5, 7, 9, 11} : Set ℕ) →
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j →
  (f * g + g * h + h * j + f * j : ℕ) ≤ 240 :=
by sorry

end max_product_sum_l3377_337701


namespace power_division_rule_l3377_337732

theorem power_division_rule (a : ℝ) (h : a ≠ 0) : a^4 / a = a^3 := by
  sorry

end power_division_rule_l3377_337732


namespace arithmetic_sequence_prime_divisibility_l3377_337775

theorem arithmetic_sequence_prime_divisibility 
  (n : ℕ) 
  (a : ℕ → ℕ) 
  (h_n : n ≥ 2021) 
  (h_arith : ∀ i j, i < j → j ≤ n → a j - a i = (j - i) * (a 2 - a 1))
  (h_inc : ∀ i j, i < j → j ≤ n → a i < a j)
  (h_first : a 1 > 2021)
  (h_prime : ∀ i, 1 ≤ i → i ≤ n → Nat.Prime (a i)) :
  ∀ p, p < 2021 → Nat.Prime p → (a 2 - a 1) % p = 0 :=
by sorry

end arithmetic_sequence_prime_divisibility_l3377_337775


namespace distinct_positive_solutions_l3377_337736

theorem distinct_positive_solutions (a b : ℝ) :
  (∃ (x y z : ℝ), x + y + z = a ∧ x^2 + y^2 + z^2 = b^2 ∧ x*y = z^2 ∧
    x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔
  (abs b < a ∧ a < Real.sqrt 3 * abs b) :=
sorry

end distinct_positive_solutions_l3377_337736


namespace problem_l3377_337703

def f (m : ℕ) (x : ℝ) : ℝ := |x - m| + |x|

theorem problem (m : ℕ) (h1 : m > 0) (h2 : ∃ x : ℝ, f m x < 2) :
  m = 1 ∧
  ∀ α β : ℝ, α > 1 → β > 1 → f m α + f m β = 6 → 4/α + 1/β ≥ 9/4 :=
by sorry

end problem_l3377_337703


namespace first_rope_longer_l3377_337789

-- Define the initial length of the ropes
variable (initial_length : ℝ)

-- Define the lengths cut from each rope
def cut_length_1 : ℝ := 0.3
def cut_length_2 : ℝ := 3

-- Define the remaining lengths of each rope
def remaining_length_1 : ℝ := initial_length - cut_length_1
def remaining_length_2 : ℝ := initial_length - cut_length_2

-- Theorem statement
theorem first_rope_longer :
  remaining_length_1 initial_length > remaining_length_2 initial_length :=
by sorry

end first_rope_longer_l3377_337789


namespace x_plus_2y_squared_equals_half_l3377_337746

theorem x_plus_2y_squared_equals_half (x y : ℝ) 
  (h : 8*y^4 + 4*x^2*y^2 + 4*x*y^2 + 2*x^3 + 2*y^2 + 2*x = x^2 + 1) : 
  x + 2*y^2 = 1/2 := by
  sorry

end x_plus_2y_squared_equals_half_l3377_337746


namespace michaels_estimate_greater_l3377_337729

theorem michaels_estimate_greater (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxy : x > y) : 
  3 * ((x + z) - (y - 2 * z)) > 3 * (x - y) := by
  sorry

end michaels_estimate_greater_l3377_337729


namespace min_value_theorem_l3377_337730

theorem min_value_theorem (x : ℝ) (h1 : x > 0) (h2 : Real.log x + 1 ≤ x) :
  (x^2 - Real.log x + x) / x ≥ 2 ∧
  ((x^2 - Real.log x + x) / x = 2 ↔ x = 1) :=
sorry

end min_value_theorem_l3377_337730


namespace range_of_a_l3377_337718

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0) → (a > 3 ∨ a < -1) :=
by sorry

end range_of_a_l3377_337718


namespace train_journey_time_l3377_337702

/-- Represents the train journey from A to B -/
structure TrainJourney where
  d : ℝ  -- Total distance
  v : ℝ  -- Initial speed
  t : ℝ  -- Total scheduled time

/-- The conditions of the train journey -/
def journey_conditions (j : TrainJourney) : Prop :=
  j.d > 0 ∧ j.v > 0 ∧
  (j.d / (2 * j.v)) + 15 + (j.d / (8 * j.v)) = j.t

/-- The theorem stating that the total journey time is 40 minutes -/
theorem train_journey_time (j : TrainJourney) 
  (h : journey_conditions j) : j.t = 40 := by
  sorry

#check train_journey_time

end train_journey_time_l3377_337702


namespace complex_division_result_l3377_337719

theorem complex_division_result : ((-2 : ℂ) - I) / I = -1 + 2*I := by sorry

end complex_division_result_l3377_337719


namespace quadratic_equal_roots_l3377_337720

theorem quadratic_equal_roots :
  ∃ (x : ℝ), x^2 + 2*x + 1 = 0 ∧
  (∀ (y : ℝ), y^2 + 2*y + 1 = 0 → y = x) :=
by sorry

end quadratic_equal_roots_l3377_337720


namespace ellipse_properties_l3377_337737

/-- Properties of an ellipse with given parameters -/
theorem ellipse_properties :
  let e : ℝ := 1/2  -- eccentricity
  let c : ℝ := 1    -- half the distance between foci
  let a : ℝ := 2    -- semi-major axis
  let b : ℝ := Real.sqrt 3  -- semi-minor axis
  let F₁ : ℝ × ℝ := (-1, 0)  -- left focus
  let A : ℝ × ℝ := (-2, 0)  -- left vertex
  ∀ x y : ℝ,
    (x^2 / 4 + y^2 / 3 = 1) →  -- point (x,y) is on the ellipse
    (0 ≤ (x + 1) * (x + 2) + y^2) ∧
    ((x + 1) * (x + 2) + y^2 ≤ 12) := by
  sorry

end ellipse_properties_l3377_337737


namespace product_closure_l3377_337777

def A : Set ℤ := {z | ∃ a b : ℤ, z = a^2 + 4*a*b + b^2}

theorem product_closure (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := by
  sorry

end product_closure_l3377_337777


namespace jason_pokemon_cards_l3377_337716

theorem jason_pokemon_cards (initial_cards : ℕ) (given_away : ℕ) : 
  initial_cards = 9 → given_away = 4 → initial_cards - given_away = 5 := by
sorry

end jason_pokemon_cards_l3377_337716


namespace volume_of_region_l3377_337717

-- Define the function f
def f (x y z : ℝ) : ℝ := |x - y + z| + |x - y - z| + |x + y - z| + |-x + y - z|

-- Define the region R
def R : Set (ℝ × ℝ × ℝ) := {(x, y, z) | f x y z ≤ 6}

-- Theorem statement
theorem volume_of_region : MeasureTheory.volume R = 36 := by
  sorry

end volume_of_region_l3377_337717


namespace claire_balloons_count_l3377_337731

/-- The number of balloons Claire has at the end of the fair --/
def claire_balloons : ℕ :=
  let initial := 50
  let given_to_girl := 1
  let floated_away := 12
  let given_away_later := 9
  let grabbed_from_coworker := 11
  initial - given_to_girl - floated_away - given_away_later + grabbed_from_coworker

theorem claire_balloons_count : claire_balloons = 39 := by
  sorry

end claire_balloons_count_l3377_337731


namespace subtraction_result_l3377_337793

theorem subtraction_result : 888888888888 - 111111111111 = 777777777777 := by
  sorry

end subtraction_result_l3377_337793


namespace car_speed_second_hour_l3377_337750

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 90)
  (h2 : average_speed = 66) : 
  (2 * average_speed - speed_first_hour) = 42 :=
by sorry

end car_speed_second_hour_l3377_337750


namespace triangle_property_l3377_337774

theorem triangle_property (A B C : Real) (a b c R : Real) :
  0 < B → B < π / 2 →
  2 * R - b = 2 * b * Real.sin B →
  a = Real.sqrt 3 →
  c = 3 →
  B = π / 6 ∧ Real.sin C = Real.sqrt 3 / 2 := by
  sorry

end triangle_property_l3377_337774


namespace ball_hitting_ground_time_l3377_337709

theorem ball_hitting_ground_time :
  ∃ t : ℝ, t > 0 ∧ -10 * t^2 - 20 * t + 180 = 0 ∧ t = 3 := by
  sorry

end ball_hitting_ground_time_l3377_337709


namespace car_cost_calculation_l3377_337797

/-- The cost of a car shared between two people, where one pays $900 for 3/7 of the usage -/
theorem car_cost_calculation (sue_payment : ℝ) (sue_usage : ℚ) (total_cost : ℝ) : 
  sue_payment = 900 → 
  sue_usage = 3/7 → 
  sue_payment / total_cost = sue_usage →
  total_cost = 2100 := by
  sorry

#check car_cost_calculation

end car_cost_calculation_l3377_337797


namespace sine_graph_shift_l3377_337762

theorem sine_graph_shift (x : ℝ) :
  2 * Real.sin (2 * (x - 2 * π / 3)) = 2 * Real.sin (2 * ((x + 2 * π / 3) - 2 * π / 3)) :=
by sorry

end sine_graph_shift_l3377_337762


namespace toll_constant_is_half_dollar_l3377_337782

/-- The number of axles on an 18-wheel truck with 2 wheels on its front axle and 4 wheels on each other axle -/
def truck_axles : ℕ := 5

/-- The toll formula for a truck -/
def toll (constant : ℝ) (x : ℕ) : ℝ := 2.50 + constant * (x - 2)

/-- The theorem stating that the constant in the toll formula is 0.50 -/
theorem toll_constant_is_half_dollar :
  ∃ (constant : ℝ), toll constant truck_axles = 4 ∧ constant = 0.50 := by
  sorry

end toll_constant_is_half_dollar_l3377_337782


namespace tangent_line_proof_l3377_337738

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + x + 1

/-- The proposed tangent line function -/
def g (x : ℝ) : ℝ := x + 1

theorem tangent_line_proof :
  (∃ x₀ : ℝ, f x₀ = g x₀ ∧ 
    (∀ x : ℝ, x ≠ x₀ → f x < g x ∨ f x > g x)) ∧
  g (-1) = 0 :=
sorry

end tangent_line_proof_l3377_337738


namespace initial_money_amount_initial_money_amount_proof_l3377_337751

/-- Proves that given the conditions in the problem, the initial amount of money is 160 dollars --/
theorem initial_money_amount : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun your_weekly_savings friend_initial_money friend_weekly_savings weeks initial_money =>
    your_weekly_savings = 7 →
    friend_initial_money = 210 →
    friend_weekly_savings = 5 →
    weeks = 25 →
    initial_money + (your_weekly_savings * weeks) = friend_initial_money + (friend_weekly_savings * weeks) →
    initial_money = 160

/-- The proof of the theorem --/
theorem initial_money_amount_proof :
  initial_money_amount 7 210 5 25 160 := by
  sorry

end initial_money_amount_initial_money_amount_proof_l3377_337751


namespace marys_blueberries_l3377_337776

theorem marys_blueberries (apples oranges total_left : ℕ) (h1 : apples = 14) (h2 : oranges = 9) (h3 : total_left = 26) :
  ∃ blueberries : ℕ, blueberries = 5 ∧ total_left = (apples - 1) + (oranges - 1) + (blueberries - 1) :=
by
  sorry

end marys_blueberries_l3377_337776


namespace initial_percent_problem_l3377_337757

theorem initial_percent_problem (x : ℝ) : 
  (x / 100) * (5 / 100) = 60 / 100 → x = 1200 := by
  sorry

end initial_percent_problem_l3377_337757


namespace employees_in_all_three_proof_l3377_337749

/-- The number of employees trained to work in all 3 restaurants -/
def employees_in_all_three : ℕ := 2

theorem employees_in_all_three_proof :
  let total_employees : ℕ := 39
  let min_restaurants : ℕ := 1
  let max_restaurants : ℕ := 3
  let family_buffet : ℕ := 15
  let dining_room : ℕ := 18
  let snack_bar : ℕ := 12
  let in_two_restaurants : ℕ := 4
  employees_in_all_three = 
    total_employees + employees_in_all_three - in_two_restaurants - 
    (family_buffet + dining_room + snack_bar) := by
  sorry

#check employees_in_all_three_proof

end employees_in_all_three_proof_l3377_337749


namespace prob_product_odd_eight_rolls_l3377_337761

-- Define a standard die
def StandardDie : Type := Fin 6

-- Define the property of being an odd number
def isOdd (n : Nat) : Prop := n % 2 = 1

-- Define the probability of rolling an odd number on a standard die
def probOddRoll : ℚ := 1 / 2

-- Define the number of rolls
def numRolls : Nat := 8

-- Theorem statement
theorem prob_product_odd_eight_rolls :
  (probOddRoll ^ numRolls : ℚ) = 1 / 256 := by
  sorry

end prob_product_odd_eight_rolls_l3377_337761


namespace product_increase_2016_l3377_337722

theorem product_increase_2016 : ∃ (a b c : ℕ), 
  ((a - 3) * (b - 3) * (c - 3)) - (a * b * c) = 2016 := by
  sorry

end product_increase_2016_l3377_337722


namespace imaginary_part_of_z_l3377_337758

theorem imaginary_part_of_z (z : ℂ) : z = (3 - I) / (1 + I) → z.im = -2 := by
  sorry

end imaginary_part_of_z_l3377_337758


namespace cube_of_complex_number_l3377_337760

theorem cube_of_complex_number :
  let z : ℂ := 2 + 5*I
  z^3 = -142 - 65*I := by sorry

end cube_of_complex_number_l3377_337760


namespace function_value_at_two_l3377_337768

/-- Given a function f(x) = ax^5 + bx^3 - x + 2 where a and b are constants,
    and f(-2) = 5, prove that f(2) = -1 -/
theorem function_value_at_two (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^5 + b * x^3 - x + 2)
  (h2 : f (-2) = 5) : f 2 = -1 := by
  sorry

end function_value_at_two_l3377_337768


namespace roots_are_irrational_l3377_337759

theorem roots_are_irrational (j : ℝ) : 
  (∃ x y : ℝ, x^2 - 5*j*x + 3*j^2 - 2 = 0 ∧ y^2 - 5*j*y + 3*j^2 - 2 = 0 ∧ x * y = 11) →
  (∃ x y : ℝ, x^2 - 5*j*x + 3*j^2 - 2 = 0 ∧ y^2 - 5*j*y + 3*j^2 - 2 = 0 ∧ ¬(∃ m n : ℤ, x = m / n ∨ y = m / n)) :=
by sorry

end roots_are_irrational_l3377_337759


namespace quadratic_function_properties_l3377_337711

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- State the theorem
theorem quadratic_function_properties
  (a b : ℝ) (h_a : a ≠ 0)
  (h_min : ∀ x, f a b x ≥ f a b 1)
  (h_zero : f a b 1 = 0) :
  -- 1. f(x) = x² - 2x + 1
  (∀ x, f a b x = x^2 - 2*x + 1) ∧
  -- 2. f(x) is decreasing on (-∞, 1] and increasing on [1, +∞)
  (∀ x y, x ≤ 1 → y ≤ 1 → x ≤ y → f a b x ≥ f a b y) ∧
  (∀ x y, 1 ≤ x → 1 ≤ y → x ≤ y → f a b x ≤ f a b y) ∧
  -- 3. If f(x) > x + k for all x ∈ [1, 3], then k < -5/4
  (∀ k, (∀ x, 1 ≤ x → x ≤ 3 → f a b x > x + k) → k < -5/4) :=
by sorry

end quadratic_function_properties_l3377_337711


namespace average_PQR_l3377_337721

theorem average_PQR (P Q R : ℚ) 
  (eq1 : 1001 * R - 3003 * P = 6006)
  (eq2 : 2002 * Q + 4004 * P = 8008) :
  (P + Q + R) / 3 = 2 * (P + 5) / 3 := by
  sorry

end average_PQR_l3377_337721


namespace golf_ball_difference_l3377_337724

theorem golf_ball_difference (bin_f bin_g : ℕ) : 
  bin_f = (2 * bin_g) / 3 →
  bin_f + bin_g = 150 →
  bin_g - bin_f = 30 := by
sorry

end golf_ball_difference_l3377_337724


namespace complement_determines_set_l3377_337799

def U : Set Nat := {1, 2, 3, 4}

theorem complement_determines_set (B : Set Nat) (h : Set.compl B = {2, 3}) : B = {1, 4} := by
  sorry

end complement_determines_set_l3377_337799


namespace simplify_expression_l3377_337734

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 := by
  sorry

end simplify_expression_l3377_337734


namespace find_number_l3377_337747

theorem find_number (x : ℝ) : ((x * 14) / 100) = 0.045374000000000005 → x = 0.3241 := by
  sorry

end find_number_l3377_337747


namespace ring_toss_earnings_l3377_337754

/-- The ring toss game earnings problem -/
theorem ring_toss_earnings (total_earnings : ℝ) (num_days : ℕ) (h1 : total_earnings = 120) (h2 : num_days = 20) :
  total_earnings / num_days = 6 := by
  sorry

end ring_toss_earnings_l3377_337754


namespace unique_solution_mod_125_l3377_337728

theorem unique_solution_mod_125 :
  ∃! x : ℕ, x < 125 ∧ (x^3 - 2*x + 6) % 125 = 0 :=
by sorry

end unique_solution_mod_125_l3377_337728


namespace square_root_fraction_equality_l3377_337755

theorem square_root_fraction_equality : 
  Real.sqrt (8^2 + 15^2) / Real.sqrt (25 + 16) = (17 * Real.sqrt 41) / 41 := by
  sorry

end square_root_fraction_equality_l3377_337755


namespace arithmetic_sequence_sum_l3377_337791

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The statement of the problem -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3)^2 + 3 * (a 3) - 18 = 0 →
  (a 8)^2 + 3 * (a 8) - 18 = 0 →
  a 5 + a 6 = 3 :=
by sorry

end arithmetic_sequence_sum_l3377_337791


namespace square_root_3adic_l3377_337735

/-- Checks if 201 is the square root of 112101 in 3-adic arithmetic up to 3 digits of precision -/
theorem square_root_3adic (n : Nat) : n = 201 → n * n ≡ 112101 [ZMOD 27] := by
  sorry

end square_root_3adic_l3377_337735


namespace sum_of_50th_row_l3377_337714

/-- Represents the sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℕ :=
  2^n - 2 * (n * (n + 1) / 2)

/-- The triangular array property -/
axiom triangular_array_property (n : ℕ) :
  f n = 2 * f (n - 1) + n * (n + 1)

/-- Theorem: The sum of numbers in the 50th row is 2^50 - 2550 -/
theorem sum_of_50th_row :
  f 50 = 2^50 - 2550 := by sorry

end sum_of_50th_row_l3377_337714


namespace strawberry_ratio_l3377_337740

def strawberry_problem (betty_strawberries matthew_strawberries natalie_strawberries : ℕ)
  (strawberries_per_jar jar_price total_revenue : ℕ) : Prop :=
  betty_strawberries = 16 ∧
  matthew_strawberries = betty_strawberries + 20 ∧
  matthew_strawberries = natalie_strawberries ∧
  strawberries_per_jar = 7 ∧
  jar_price = 4 ∧
  total_revenue = 40 ∧
  (matthew_strawberries : ℚ) / natalie_strawberries = 1

theorem strawberry_ratio :
  ∀ (betty_strawberries matthew_strawberries natalie_strawberries : ℕ)
    (strawberries_per_jar jar_price total_revenue : ℕ),
  strawberry_problem betty_strawberries matthew_strawberries natalie_strawberries
    strawberries_per_jar jar_price total_revenue →
  (matthew_strawberries : ℚ) / natalie_strawberries = 1 :=
by
  sorry

end strawberry_ratio_l3377_337740


namespace expansion_properties_l3377_337788

/-- The expansion of (x^(1/4) + x^(3/2))^n where the third-to-last term's coefficient is 45 -/
def expansion (x : ℝ) (n : ℕ) := (x^(1/4) + x^(3/2))^n

/-- The coefficient of the third-to-last term in the expansion -/
def third_to_last_coeff (n : ℕ) := Nat.choose n (n - 2)

theorem expansion_properties (x : ℝ) (n : ℕ) 
  (h : third_to_last_coeff n = 45) : 
  ∃ (k : ℕ), 
    (Nat.choose n k * x^5 = 45 * x^5) ∧ 
    (∀ (j : ℕ), j ≤ n → Nat.choose n j ≤ 252) ∧
    (Nat.choose n 5 * x^(35/4) = 252 * x^(35/4)) := by
  sorry

end expansion_properties_l3377_337788


namespace max_areas_theorem_l3377_337790

/-- Represents the number of non-overlapping areas in a circular disk -/
def max_areas (n : ℕ) : ℕ := 3 * n + 1

/-- 
Theorem: Given a circular disk divided by 2n equally spaced radii (n > 0) and one secant line, 
the maximum number of non-overlapping areas is 3n + 1.
-/
theorem max_areas_theorem (n : ℕ) (h : n > 0) : 
  max_areas n = 3 * n + 1 := by
  sorry

#check max_areas_theorem

end max_areas_theorem_l3377_337790


namespace trigonometric_product_equals_one_l3377_337753

theorem trigonometric_product_equals_one : 
  (1 - 1 / Real.cos (30 * π / 180)) * 
  (1 + 1 / Real.sin (60 * π / 180)) * 
  (1 - 1 / Real.sin (30 * π / 180)) * 
  (1 + 1 / Real.cos (60 * π / 180)) = 1 := by
  sorry

end trigonometric_product_equals_one_l3377_337753


namespace centers_regular_iff_original_affinely_regular_l3377_337785

open Complex

/-- Definition of an n-gon as a list of complex numbers -/
def NGon (n : ℕ) := List ℂ

/-- A convex n-gon -/
def ConvexNGon (n : ℕ) (A : NGon n) : Prop := sorry

/-- Centers of regular n-gons constructed on sides of an n-gon -/
def CentersOfExternalNGons (n : ℕ) (A : NGon n) : NGon n := sorry

/-- Check if an n-gon is regular -/
def IsRegularNGon (n : ℕ) (B : NGon n) : Prop := sorry

/-- Check if an n-gon is affinely regular -/
def IsAffinelyRegularNGon (n : ℕ) (A : NGon n) : Prop := sorry

/-- Main theorem: The centers form a regular n-gon iff the original n-gon is affinely regular -/
theorem centers_regular_iff_original_affinely_regular 
  (n : ℕ) (A : NGon n) (h : ConvexNGon n A) :
  IsRegularNGon n (CentersOfExternalNGons n A) ↔ IsAffinelyRegularNGon n A :=
sorry

end centers_regular_iff_original_affinely_regular_l3377_337785


namespace student_selection_methods_l3377_337733

/-- Represents the number of ways to select students by gender from a group -/
def select_students (total : ℕ) (boys : ℕ) (girls : ℕ) (to_select : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of ways to select 4 students by gender from a group of 8 students (6 boys and 2 girls) is 40 -/
theorem student_selection_methods :
  select_students 8 6 2 4 = 40 :=
sorry

end student_selection_methods_l3377_337733


namespace total_guests_served_l3377_337742

theorem total_guests_served (adults : ℕ) (children : ℕ) (seniors : ℕ) : 
  adults = 58 →
  children = adults - 35 →
  seniors = 2 * children →
  adults + children + seniors = 127 := by
  sorry

end total_guests_served_l3377_337742


namespace prob_second_draw_l3377_337794

structure Bag where
  red : ℕ
  blue : ℕ

def initial_bag : Bag := ⟨5, 4⟩

def P_A2 (b : Bag) : ℚ :=
  (b.red : ℚ) / (b.red + b.blue)

def P_B2 (b : Bag) : ℚ :=
  (b.blue : ℚ) / (b.red + b.blue)

def P_A2_given_A1 (b : Bag) : ℚ :=
  ((b.red - 1) : ℚ) / (b.red + b.blue - 1)

def P_B2_given_A1 (b : Bag) : ℚ :=
  (b.blue : ℚ) / (b.red + b.blue - 1)

theorem prob_second_draw (b : Bag) :
  P_A2 b = 5/9 ∧
  P_A2 b + P_B2 b = 1 ∧
  P_A2_given_A1 b + P_B2_given_A1 b = 1 :=
by sorry

end prob_second_draw_l3377_337794


namespace smallest_three_digit_congruence_l3377_337786

theorem smallest_three_digit_congruence :
  ∃ n : ℕ, 
    (100 ≤ n ∧ n < 1000) ∧ 
    (60 * n ≡ 180 [MOD 300]) ∧ 
    (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ (60 * m ≡ 180 [MOD 300]) → n ≤ m) ∧
    n = 103 := by
  sorry

end smallest_three_digit_congruence_l3377_337786


namespace smallest_number_with_2020_divisors_l3377_337748

def n : ℕ := 2^100 * 3^4 * 5 * 7

theorem smallest_number_with_2020_divisors :
  (∀ m : ℕ, m < n → (Nat.divisors m).card ≠ 2020) ∧
  (Nat.divisors n).card = 2020 := by
  sorry

end smallest_number_with_2020_divisors_l3377_337748
