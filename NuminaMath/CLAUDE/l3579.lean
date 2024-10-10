import Mathlib

namespace equal_selection_probability_l3579_357980

def TwoStepSelection (n : ℕ) (m : ℕ) (k : ℕ) :=
  (n > m) ∧ (m > k) ∧ (k > 0)

theorem equal_selection_probability
  (n m k : ℕ)
  (h : TwoStepSelection n m k)
  (eliminate_one : ℕ → ℕ)
  (systematic_sample : ℕ → Finset ℕ)
  (h_eliminate : ∀ i, i ∈ Finset.range n → eliminate_one i ∈ Finset.range (n - 1))
  (h_sample : ∀ i, i ∈ Finset.range (n - 1) → systematic_sample i ⊆ Finset.range (n - 1) ∧ (systematic_sample i).card = k) :
  ∀ j ∈ Finset.range n, (∃ i ∈ Finset.range (n - 1), j ∈ systematic_sample (eliminate_one i)) ↔ true :=
sorry

#check equal_selection_probability

end equal_selection_probability_l3579_357980


namespace sum_of_possible_A_values_l3579_357924

/-- The sum of digits of a number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by 9 -/
def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

/-- The given number with A as a parameter -/
def given_number (A : ℕ) : ℕ := 7456291 * 10 + A * 10 + 2

theorem sum_of_possible_A_values : 
  (∀ A : ℕ, A < 10 → is_divisible_by_9 (given_number A) → 
    sum_of_digits (given_number A) = sum_of_digits 7456291 + A + 2) →
  (∃ A₁ A₂ : ℕ, A₁ < 10 ∧ A₂ < 10 ∧ 
    is_divisible_by_9 (given_number A₁) ∧ 
    is_divisible_by_9 (given_number A₂) ∧
    A₁ + A₂ = 9) :=
sorry

end sum_of_possible_A_values_l3579_357924


namespace tangent_line_at_x_1_l3579_357982

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4*x^3 - 2*x

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 2*x - 1 :=
by sorry

end tangent_line_at_x_1_l3579_357982


namespace granddaughter_mother_age_ratio_l3579_357905

/-- The ratio of a granddaughter's age to her mother's age, given the ages of three generations. -/
theorem granddaughter_mother_age_ratio
  (betty_age : ℕ)
  (daughter_age : ℕ)
  (granddaughter_age : ℕ)
  (h1 : betty_age = 60)
  (h2 : daughter_age = betty_age - (40 * betty_age / 100))
  (h3 : granddaughter_age = 12) :
  granddaughter_age / daughter_age = 1 / 3 :=
by sorry

end granddaughter_mother_age_ratio_l3579_357905


namespace simplify_expression_l3579_357902

theorem simplify_expression (x y : ℝ) : 3*y - 5*x + 2*y + 4*x = 5*y - x := by
  sorry

end simplify_expression_l3579_357902


namespace company_gender_distribution_l3579_357926

theorem company_gender_distribution (total : ℕ) 
  (h1 : total / 3 = total - 2 * total / 3)  -- One-third of workers don't have a retirement plan
  (h2 : (3 * total / 5) / 3 = total / 3 - 2 * total / 5 / 3)  -- 60% of workers without a retirement plan are women
  (h3 : (2 * total / 5) / 3 = total / 3 - 3 * total / 5 / 3)  -- 40% of workers without a retirement plan are men
  (h4 : 4 * (2 * total / 3) / 10 = 2 * total / 3 - 6 * (2 * total / 3) / 10)  -- 40% of workers with a retirement plan are men
  (h5 : 6 * (2 * total / 3) / 10 = 2 * total / 3 - 4 * (2 * total / 3) / 10)  -- 60% of workers with a retirement plan are women
  (h6 : (2 * total / 5) / 3 + 4 * (2 * total / 3) / 10 = 120)  -- There are 120 men in total
  : total - 120 = 180 := by
  sorry

end company_gender_distribution_l3579_357926


namespace label_difference_less_than_distance_l3579_357908

open Set

theorem label_difference_less_than_distance :
  ∀ f : ℝ × ℝ → ℝ, ∃ P Q : ℝ × ℝ, P ≠ Q ∧ |f P - f Q| < ‖P - Q‖ :=
by sorry

end label_difference_less_than_distance_l3579_357908


namespace compare_with_one_twentieth_l3579_357938

theorem compare_with_one_twentieth : 
  (1 / 15 : ℚ) > 1 / 20 ∧ 
  (1 / 25 : ℚ) < 1 / 20 ∧ 
  (1 / 2 : ℚ) > 1 / 20 ∧ 
  (55 / 1000 : ℚ) > 1 / 20 ∧ 
  (1 / 10 : ℚ) > 1 / 20 := by
  sorry

#check compare_with_one_twentieth

end compare_with_one_twentieth_l3579_357938


namespace simplify_sqrt_500_l3579_357919

theorem simplify_sqrt_500 : Real.sqrt 500 = 10 * Real.sqrt 5 := by
  sorry

end simplify_sqrt_500_l3579_357919


namespace cow_sheep_value_l3579_357955

/-- The value of cows and sheep in taels of gold -/
theorem cow_sheep_value (x y : ℚ) 
  (h1 : 5 * x + 2 * y = 10) 
  (h2 : 2 * x + 5 * y = 8) : 
  x + y = 18 / 7 := by
  sorry

end cow_sheep_value_l3579_357955


namespace steven_owes_jeremy_l3579_357911

/-- The amount Steven owes Jeremy for cleaning rooms --/
def amount_owed (base_rate : ℚ) (rooms_cleaned : ℚ) (bonus_threshold : ℚ) (bonus_rate : ℚ) : ℚ :=
  let base_payment := base_rate * rooms_cleaned
  let bonus_payment := if rooms_cleaned > bonus_threshold then rooms_cleaned * bonus_rate else 0
  base_payment + bonus_payment

/-- Theorem: Steven owes Jeremy 145/12 dollars --/
theorem steven_owes_jeremy :
  let base_rate : ℚ := 13/3
  let rooms_cleaned : ℚ := 5/2
  let bonus_threshold : ℚ := 2
  let bonus_rate : ℚ := 1/2
  amount_owed base_rate rooms_cleaned bonus_threshold bonus_rate = 145/12 := by
  sorry


end steven_owes_jeremy_l3579_357911


namespace orange_price_is_60_l3579_357909

/- Define the problem parameters -/
def apple_price : ℚ := 40
def initial_total_fruits : ℕ := 10
def initial_avg_price : ℚ := 48
def oranges_removed : ℕ := 2
def final_avg_price : ℚ := 45

/- Define the function to calculate the orange price -/
def calculate_orange_price : ℚ :=
  let initial_total_cost : ℚ := initial_total_fruits * initial_avg_price
  let final_total_fruits : ℕ := initial_total_fruits - oranges_removed
  let final_total_cost : ℚ := final_total_fruits * final_avg_price
  60  -- The calculated price of each orange

/- Theorem statement -/
theorem orange_price_is_60 :
  calculate_orange_price = 60 :=
sorry

end orange_price_is_60_l3579_357909


namespace complex_fraction_equality_l3579_357935

theorem complex_fraction_equality : ∃ z : ℂ, z = (2 - I) / (1 - I) ∧ z = (3/2 : ℂ) + (1/2 : ℂ) * I :=
sorry

end complex_fraction_equality_l3579_357935


namespace negation_of_existence_square_positive_negation_l3579_357953

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, x < 0 ∧ p x) ↔ (∀ x, x < 0 → ¬ p x) := by sorry

theorem square_positive_negation :
  (¬ ∃ x : ℝ, x < 0 ∧ x^2 > 0) ↔ (∀ x : ℝ, x < 0 → x^2 ≤ 0) := by sorry

end negation_of_existence_square_positive_negation_l3579_357953


namespace point_on_x_axis_l3579_357996

/-- 
If a point P(a+2, a-3) lies on the x-axis, then its coordinates are (5, 0).
-/
theorem point_on_x_axis (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = a + 2 ∧ P.2 = a - 3 ∧ P.2 = 0) → 
  (∃ P : ℝ × ℝ, P = (5, 0)) :=
by sorry

end point_on_x_axis_l3579_357996


namespace percentage_of_employees_6_years_or_more_l3579_357929

/-- Represents the distribution of employees' duration of service at the Fermat Company -/
structure EmployeeDistribution :=
  (less_than_1_year : ℕ)
  (from_1_to_1_5_years : ℕ)
  (from_1_5_to_2_5_years : ℕ)
  (from_2_5_to_3_5_years : ℕ)
  (from_3_5_to_4_5_years : ℕ)
  (from_4_5_to_5_5_years : ℕ)
  (from_5_5_to_6_5_years : ℕ)
  (from_6_5_to_7_5_years : ℕ)
  (from_7_5_to_8_5_years : ℕ)
  (from_8_5_to_10_years : ℕ)

/-- Calculates the total number of employees -/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_1_year + d.from_1_to_1_5_years + d.from_1_5_to_2_5_years +
  d.from_2_5_to_3_5_years + d.from_3_5_to_4_5_years + d.from_4_5_to_5_5_years +
  d.from_5_5_to_6_5_years + d.from_6_5_to_7_5_years + d.from_7_5_to_8_5_years +
  d.from_8_5_to_10_years

/-- Calculates the number of employees who have worked for 6 years or more -/
def employees_6_years_or_more (d : EmployeeDistribution) : ℕ :=
  d.from_5_5_to_6_5_years + d.from_6_5_to_7_5_years + d.from_7_5_to_8_5_years +
  d.from_8_5_to_10_years

/-- The theorem to be proved -/
theorem percentage_of_employees_6_years_or_more
  (d : EmployeeDistribution)
  (h1 : d.less_than_1_year = 4)
  (h2 : d.from_1_to_1_5_years = 6)
  (h3 : d.from_1_5_to_2_5_years = 7)
  (h4 : d.from_2_5_to_3_5_years = 4)
  (h5 : d.from_3_5_to_4_5_years = 3)
  (h6 : d.from_4_5_to_5_5_years = 3)
  (h7 : d.from_5_5_to_6_5_years = 2)
  (h8 : d.from_6_5_to_7_5_years = 1)
  (h9 : d.from_7_5_to_8_5_years = 1)
  (h10 : d.from_8_5_to_10_years = 1) :
  (employees_6_years_or_more d : ℚ) / (total_employees d : ℚ) = 5 / 32 :=
sorry

end percentage_of_employees_6_years_or_more_l3579_357929


namespace sum_of_four_numbers_l3579_357912

theorem sum_of_four_numbers : 2345 + 3452 + 4523 + 5234 = 15554 := by
  sorry

end sum_of_four_numbers_l3579_357912


namespace total_mission_time_is_11_days_l3579_357937

/-- Calculates the total time spent on missions given the planned duration of the first mission,
    the percentage increase in duration, and the duration of the second mission. -/
def total_mission_time (planned_duration : ℝ) (percentage_increase : ℝ) (second_mission_duration : ℝ) : ℝ :=
  (planned_duration * (1 + percentage_increase)) + second_mission_duration

/-- Proves that the total time spent on missions is 11 days. -/
theorem total_mission_time_is_11_days : 
  total_mission_time 5 0.6 3 = 11 := by
  sorry

end total_mission_time_is_11_days_l3579_357937


namespace runner_ends_at_start_l3579_357972

/-- A runner on a circular track -/
structure Runner where
  start_position : ℝ  -- Position on the track (0 ≤ position < track_length)
  distance_run : ℝ    -- Total distance run
  track_length : ℝ    -- Length of the circular track

/-- Theorem: A runner who completes an integer number of laps ends at the starting position -/
theorem runner_ends_at_start (runner : Runner) (h : runner.track_length > 0) :
  runner.distance_run % runner.track_length = 0 →
  (runner.start_position + runner.distance_run) % runner.track_length = runner.start_position :=
by sorry

end runner_ends_at_start_l3579_357972


namespace number_divided_by_6_multiplied_by_12_equals_9_l3579_357947

theorem number_divided_by_6_multiplied_by_12_equals_9 (x : ℝ) : (x / 6) * 12 = 9 → x = 4.5 := by
  sorry

end number_divided_by_6_multiplied_by_12_equals_9_l3579_357947


namespace lime_score_difference_l3579_357906

/-- Given a ratio of white to black scores and a total number of lime scores,
    calculate 2/3 of the difference between the number of white and black scores. -/
theorem lime_score_difference (white_ratio black_ratio total_lime_scores : ℕ) : 
  white_ratio = 13 → 
  black_ratio = 8 → 
  total_lime_scores = 270 → 
  (2 : ℚ) / 3 * (white_ratio * (total_lime_scores / (white_ratio + black_ratio)) - 
                 black_ratio * (total_lime_scores / (white_ratio + black_ratio))) = 43 := by
  sorry

#eval (2 : ℚ) / 3 * (13 * (270 / (13 + 8)) - 8 * (270 / (13 + 8)))

end lime_score_difference_l3579_357906


namespace total_students_l3579_357963

theorem total_students (students_per_group : ℕ) (groups_per_class : ℕ) (classes : ℕ)
  (h1 : students_per_group = 7)
  (h2 : groups_per_class = 9)
  (h3 : classes = 13) :
  students_per_group * groups_per_class * classes = 819 := by
  sorry

end total_students_l3579_357963


namespace system_solution_l3579_357961

theorem system_solution : ∃! (x y z : ℝ),
  (3 * x - 2 * y + z = 7) ∧
  (9 * y - 6 * x - 3 * z = -21) ∧
  (x + y + z = 5) ∧
  (x = 1 ∧ y = 0 ∧ z = 4) :=
by sorry

end system_solution_l3579_357961


namespace trig_expression_equals_one_l3579_357966

theorem trig_expression_equals_one :
  let α : Real := 37 * π / 180
  let β : Real := 53 * π / 180
  (1 - 1 / Real.cos α) * (1 + 1 / Real.sin β) * (1 - 1 / Real.sin α) * (1 + 1 / Real.cos β) = 1 := by
  sorry

end trig_expression_equals_one_l3579_357966


namespace marbles_selection_count_l3579_357921

def total_marbles : ℕ := 15
def special_marbles : ℕ := 4
def marbles_to_choose : ℕ := 5
def special_marbles_to_choose : ℕ := 2

theorem marbles_selection_count :
  (Nat.choose special_marbles special_marbles_to_choose) *
  (Nat.choose (total_marbles - special_marbles) (marbles_to_choose - special_marbles_to_choose)) =
  990 := by sorry

end marbles_selection_count_l3579_357921


namespace consecutive_integers_sum_l3579_357979

theorem consecutive_integers_sum (a b c : ℤ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c = 13) → a + b + c = 36 := by
  sorry

end consecutive_integers_sum_l3579_357979


namespace at_least_one_irrational_l3579_357986

theorem at_least_one_irrational (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) :
  ¬(∃ (q r : ℚ), (↑q : ℝ) = a ∧ (↑r : ℝ) = b) :=
sorry

end at_least_one_irrational_l3579_357986


namespace envelope_printing_equation_l3579_357901

/-- The equation for two envelope-printing machines to print 500 envelopes in 2 minutes -/
theorem envelope_printing_equation (x : ℝ) : x > 0 → 500 / 8 + 500 / x = 500 / 2 := by
  sorry

end envelope_printing_equation_l3579_357901


namespace count_tricycles_l3579_357933

/-- The number of tricycles in a bike shop, given the number of bicycles,
    the number of wheels per bicycle and tricycle, and the total number of wheels. -/
theorem count_tricycles (num_bicycles : ℕ) (wheels_per_bicycle : ℕ) (wheels_per_tricycle : ℕ) 
    (total_wheels : ℕ) (h1 : num_bicycles = 50) (h2 : wheels_per_bicycle = 2) 
    (h3 : wheels_per_tricycle = 3) (h4 : total_wheels = 160) : 
    (total_wheels - num_bicycles * wheels_per_bicycle) / wheels_per_tricycle = 20 := by
  sorry

end count_tricycles_l3579_357933


namespace total_fruit_weight_l3579_357928

/-- The total weight of fruit sold by an orchard -/
theorem total_fruit_weight (frozen_fruit fresh_fruit : ℕ) 
  (h1 : frozen_fruit = 3513)
  (h2 : fresh_fruit = 6279) :
  frozen_fruit + fresh_fruit = 9792 := by
  sorry

end total_fruit_weight_l3579_357928


namespace factorization_eq_l3579_357946

theorem factorization_eq (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end factorization_eq_l3579_357946


namespace ashley_cocktail_calories_l3579_357999

/-- Represents the ingredients of Ashley's cocktail -/
structure Cocktail :=
  (mango_juice : ℝ)
  (honey : ℝ)
  (water : ℝ)
  (vodka : ℝ)

/-- Calculates the total calories in the cocktail -/
def total_calories (c : Cocktail) : ℝ :=
  c.mango_juice * 0.6 + c.honey * 6.4 + c.vodka * 0.7

/-- Calculates the total weight of the cocktail -/
def total_weight (c : Cocktail) : ℝ :=
  c.mango_juice + c.honey + c.water + c.vodka

/-- Ashley's cocktail recipe -/
def ashley_cocktail : Cocktail :=
  { mango_juice := 150
  , honey := 200
  , water := 300
  , vodka := 100 }

/-- Theorem stating that 300g of Ashley's cocktail contains 576 calories -/
theorem ashley_cocktail_calories :
  (300 / total_weight ashley_cocktail) * total_calories ashley_cocktail = 576 := by
  sorry


end ashley_cocktail_calories_l3579_357999


namespace sequence_a_odd_l3579_357984

def sequence_a : ℕ → ℤ
  | 0 => 2
  | 1 => 7
  | (n + 2) => sequence_a (n + 1)

axiom sequence_a_positive (n : ℕ) : 0 < sequence_a n

axiom sequence_a_inequality (n : ℕ) (h : n ≥ 2) :
  -1/2 < (sequence_a n - (sequence_a (n-1))^2 / sequence_a (n-2)) ∧
  (sequence_a n - (sequence_a (n-1))^2 / sequence_a (n-2)) ≤ 1/2

theorem sequence_a_odd (n : ℕ) (h : n > 1) : Odd (sequence_a n) := by
  sorry

end sequence_a_odd_l3579_357984


namespace M_equals_P_l3579_357944

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}
def P : Set ℝ := {a | ∃ x : ℝ, a = x^2 - 1}

theorem M_equals_P : M = P := by sorry

end M_equals_P_l3579_357944


namespace eight_bead_bracelet_arrangements_l3579_357967

/-- The number of distinct arrangements of n beads on a bracelet, 
    considering rotational symmetry but not reflection -/
def bracelet_arrangements (n : ℕ) : ℕ := Nat.factorial n / n

/-- Theorem: The number of distinct arrangements of 8 beads on a bracelet, 
    considering rotational symmetry but not reflection, is 5040 -/
theorem eight_bead_bracelet_arrangements :
  bracelet_arrangements 8 = 5040 := by
  sorry

end eight_bead_bracelet_arrangements_l3579_357967


namespace third_stack_difference_l3579_357975

/-- Represents the heights of five stacks of blocks -/
structure BlockStacks where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- The properties of the block stacks as described in the problem -/
def validBlockStacks (s : BlockStacks) : Prop :=
  s.first = 7 ∧
  s.second = s.first + 3 ∧
  s.third < s.second ∧
  s.fourth = s.third + 10 ∧
  s.fifth = 2 * s.second ∧
  s.first + s.second + s.third + s.fourth + s.fifth = 55

theorem third_stack_difference (s : BlockStacks) 
  (h : validBlockStacks s) : s.second - s.third = 1 := by
  sorry

end third_stack_difference_l3579_357975


namespace largest_common_divisor_17_30_l3579_357971

theorem largest_common_divisor_17_30 : 
  ∃ (n : ℕ), n > 0 ∧ n = 13 ∧ 
  (∀ (m : ℕ), m > 0 → 17 % m = 30 % m → m ≤ n) :=
sorry

end largest_common_divisor_17_30_l3579_357971


namespace dolphin_edge_probability_l3579_357978

/-- The probability of a point being within 2 m of the edge in a 30 m by 20 m rectangle is 23/75. -/
theorem dolphin_edge_probability : 
  let pool_length : ℝ := 30
  let pool_width : ℝ := 20
  let edge_distance : ℝ := 2
  let total_area := pool_length * pool_width
  let inner_length := pool_length - 2 * edge_distance
  let inner_width := pool_width - 2 * edge_distance
  let inner_area := inner_length * inner_width
  let edge_area := total_area - inner_area
  edge_area / total_area = 23 / 75 := by sorry

end dolphin_edge_probability_l3579_357978


namespace probability_of_winning_pair_l3579_357930

/-- Represents the color of a card -/
inductive Color
| Red
| Green

/-- Represents the label of a card -/
inductive Label
| A | B | C | D | E

/-- Represents a card in the deck -/
structure Card where
  color : Color
  label : Label

/-- The deck of cards -/
def deck : Finset Card := sorry

/-- Predicate for a winning pair of cards -/
def is_winning_pair (c1 c2 : Card) : Prop :=
  c1.color = c2.color ∨ c1.label = c2.label

/-- The number of cards in the deck -/
def deck_size : ℕ := sorry

/-- The number of winning pairs -/
def winning_pairs : ℕ := sorry

theorem probability_of_winning_pair :
  (winning_pairs : ℚ) / (deck_size.choose 2 : ℚ) = 51 / 91 := by sorry

end probability_of_winning_pair_l3579_357930


namespace bobby_candy_total_l3579_357991

/-- The total number of pieces of candy Bobby ate over two days -/
def total_candy (initial : ℕ) (next_day : ℕ) : ℕ :=
  initial + next_day

/-- Theorem stating that Bobby ate 241 pieces of candy in total -/
theorem bobby_candy_total : total_candy 89 152 = 241 := by
  sorry

end bobby_candy_total_l3579_357991


namespace teaching_competition_score_l3579_357995

theorem teaching_competition_score (teaching_design_weight : ℝ) 
                                   (on_site_demo_weight : ℝ) 
                                   (teaching_design_score : ℝ) 
                                   (on_site_demo_score : ℝ) 
                                   (h1 : teaching_design_weight = 0.2)
                                   (h2 : on_site_demo_weight = 0.8)
                                   (h3 : teaching_design_score = 90)
                                   (h4 : on_site_demo_score = 95) :
  teaching_design_weight * teaching_design_score + 
  on_site_demo_weight * on_site_demo_score = 94 := by
sorry

end teaching_competition_score_l3579_357995


namespace digit_equation_solution_l3579_357949

/-- Represents a four-digit number ABBD --/
def ABBD (A B D : Nat) : Nat := A * 1000 + B * 100 + B * 10 + D

/-- Represents a four-digit number BCAC --/
def BCAC (B C A : Nat) : Nat := B * 1000 + C * 100 + A * 10 + C

/-- Represents a five-digit number DDBBD --/
def DDBBD (D B : Nat) : Nat := D * 10000 + D * 1000 + B * 100 + B * 10 + D

theorem digit_equation_solution 
  (A B C D : Nat) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_digits : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) 
  (h_equation : ABBD A B D + BCAC B C A = DDBBD D B) : 
  D = 0 := by
  sorry

end digit_equation_solution_l3579_357949


namespace lcm_24_90_l3579_357973

theorem lcm_24_90 : Nat.lcm 24 90 = 360 := by
  sorry

end lcm_24_90_l3579_357973


namespace no_real_solutions_l3579_357987

theorem no_real_solutions (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≠ 0) :
  (∀ x : ℝ, a * x^2 + a * x + a ≠ b) ↔ (a = 0 ∨ (a ≠ 0 ∧ 4 * a * b - 3 * a^2 > 0)) :=
by sorry

end no_real_solutions_l3579_357987


namespace simplify_sqrt_expression_l3579_357936

theorem simplify_sqrt_expression : 
  Real.sqrt 5 - Real.sqrt 40 + Real.sqrt 45 = 4 * Real.sqrt 5 - 2 * Real.sqrt 10 := by
  sorry

end simplify_sqrt_expression_l3579_357936


namespace initial_money_calculation_l3579_357968

theorem initial_money_calculation (initial_money : ℚ) : 
  (2 / 5 : ℚ) * initial_money = 300 →
  initial_money = 750 := by
sorry

end initial_money_calculation_l3579_357968


namespace science_club_neither_subject_l3579_357969

theorem science_club_neither_subject (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ)
  (h_total : total = 75)
  (h_biology : biology = 40)
  (h_chemistry : chemistry = 30)
  (h_both : both = 18) :
  total - (biology + chemistry - both) = 23 := by
  sorry

end science_club_neither_subject_l3579_357969


namespace john_task_completion_time_l3579_357927

-- Define a custom type for time
structure Time where
  hours : Nat
  minutes : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

-- Define the problem statement
theorem john_task_completion_time 
  (start_time : Time)
  (two_tasks_end_time : Time)
  (h1 : start_time = { hours := 8, minutes := 30 })
  (h2 : two_tasks_end_time = { hours := 11, minutes := 10 })
  (h3 : ∃ (task_duration : Nat), 
        addMinutes start_time (2 * task_duration) = two_tasks_end_time) :
  addMinutes two_tasks_end_time 
    ((two_tasks_end_time.hours * 60 + two_tasks_end_time.minutes - 
      start_time.hours * 60 - start_time.minutes) / 2) = 
    { hours := 12, minutes := 30 } :=
by sorry

end john_task_completion_time_l3579_357927


namespace quadratic_distinct_roots_condition_l3579_357957

/-- 
Given a quadratic equation (m-2)x^2 + 2x + 1 = 0, this theorem states that 
for the equation to have two distinct real roots, m must be less than 3 and not equal to 2.
-/
theorem quadratic_distinct_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (m - 2) * x^2 + 2 * x + 1 = 0 ∧ 
   (m - 2) * y^2 + 2 * y + 1 = 0) ↔ 
  (m < 3 ∧ m ≠ 2) :=
sorry

end quadratic_distinct_roots_condition_l3579_357957


namespace prob_one_from_each_jurisdiction_prob_at_least_one_from_xiaogan_l3579_357993

/-- Represents the total number of intermediate stations -/
def total_stations : ℕ := 7

/-- Represents the number of stations in Wuhan's jurisdiction -/
def wuhan_stations : ℕ := 4

/-- Represents the number of stations in Xiaogan's jurisdiction -/
def xiaogan_stations : ℕ := 3

/-- Represents the number of stations to be selected for research -/
def selected_stations : ℕ := 2

/-- Theorem for the probability of selecting one station from each jurisdiction -/
theorem prob_one_from_each_jurisdiction :
  (total_stations.choose selected_stations : ℚ) / (total_stations.choose selected_stations) =
  (wuhan_stations * xiaogan_stations : ℚ) / (total_stations.choose selected_stations) := by sorry

/-- Theorem for the probability of selecting at least one station within Xiaogan's jurisdiction -/
theorem prob_at_least_one_from_xiaogan :
  1 - (wuhan_stations.choose selected_stations : ℚ) / (total_stations.choose selected_stations) =
  5 / 7 := by sorry

end prob_one_from_each_jurisdiction_prob_at_least_one_from_xiaogan_l3579_357993


namespace distinct_permutations_with_repetition_l3579_357914

theorem distinct_permutations_with_repetition : 
  let total_elements : ℕ := 5
  let repeated_elements : ℕ := 3
  let factorial (n : ℕ) := Nat.factorial n
  factorial total_elements / (factorial repeated_elements * factorial 1 * factorial 1) = 20 := by
  sorry

end distinct_permutations_with_repetition_l3579_357914


namespace next_simultaneous_ring_l3579_357950

def library_period : ℕ := 18
def fire_station_period : ℕ := 24
def hospital_period : ℕ := 30

def minutes_in_hour : ℕ := 60

theorem next_simultaneous_ring (start_time : ℕ) :
  ∃ (t : ℕ), t > 0 ∧ 
    t % library_period = 0 ∧ 
    t % fire_station_period = 0 ∧ 
    t % hospital_period = 0 ∧
    t / minutes_in_hour = 6 := by
  sorry

end next_simultaneous_ring_l3579_357950


namespace parabola_directrix_l3579_357990

/-- A parabola C with equation y² = mx passing through the point (-2, √3) has directrix x = 3/8 -/
theorem parabola_directrix (m : ℝ) : 
  (3 : ℝ) = m * (-2) → -- Condition: parabola passes through (-2, √3)
  (∀ x y : ℝ, y^2 = m*x → -- Definition of parabola C
    (x = 3/8 ↔ -- Equation of directrix
      ∃ p : ℝ × ℝ, 
        p.2^2 = m*p.1 ∧ -- Point on parabola
        (x - p.1)^2 = (y - p.2)^2 + (3/8 - x)^2)) -- Distance condition for directrix
  := by sorry

end parabola_directrix_l3579_357990


namespace initial_apples_count_l3579_357920

/-- The number of apples in a package -/
def apples_per_package : ℕ := 11

/-- The number of apples added to the pile -/
def apples_added : ℕ := 5

/-- The final number of apples in the pile -/
def final_apples : ℕ := 13

/-- The initial number of apples in the pile -/
def initial_apples : ℕ := final_apples - apples_added

theorem initial_apples_count : initial_apples = 8 := by
  sorry

end initial_apples_count_l3579_357920


namespace bankers_discount_l3579_357997

/-- Banker's discount calculation -/
theorem bankers_discount (bankers_gain : ℚ) (time : ℕ) (rate : ℚ) : 
  bankers_gain = 360 ∧ time = 3 ∧ rate = 12/100 → 
  ∃ (bankers_discount : ℚ), bankers_discount = 5625/10 :=
by
  sorry

end bankers_discount_l3579_357997


namespace polynomial_equality_l3579_357976

theorem polynomial_equality (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 9 * p^8 * q = 36 * p^7 * q^2 → p = 4/5 := by
  sorry

end polynomial_equality_l3579_357976


namespace hyperbola_asymptote_angle_l3579_357958

theorem hyperbola_asymptote_angle (c d : ℝ) (h1 : c > d) (h2 : c > 0) (h3 : d > 0) :
  (∀ x y : ℝ, x^2 / c^2 - y^2 / d^2 = 1) →
  (Real.arctan (d / c) - Real.arctan (-d / c) = π / 4) →
  c / d = 1 := by
sorry

end hyperbola_asymptote_angle_l3579_357958


namespace arithmetic_expression_equals_one_l3579_357951

theorem arithmetic_expression_equals_one : 3 * (7 - 5) - 5 = 1 := by
  sorry

end arithmetic_expression_equals_one_l3579_357951


namespace coupon1_best_discount_l3579_357923

def coupon1_discount (x : ℝ) : ℝ := 0.1 * x

def coupon2_discount : ℝ := 20

def coupon3_discount (x : ℝ) : ℝ := 0.18 * (x - 100)

theorem coupon1_best_discount (x : ℝ) : 
  (coupon1_discount x > coupon2_discount ∧ 
   coupon1_discount x > coupon3_discount x) ↔ 
  (200 < x ∧ x < 225) :=
sorry

end coupon1_best_discount_l3579_357923


namespace f_properties_l3579_357952

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_properties :
  (∃ (x_min : ℝ), f 1 x_min = 2 ∧ ∀ x, f 1 x ≥ f 1 x_min) ∧
  (∀ a ≤ 0, ∀ x y, x < y → f a x > f a y) ∧
  (∀ a > 0, ∀ x y, x < y → 
    ((x < -Real.log a ∧ y < -Real.log a) → f a x > f a y) ∧
    ((x > -Real.log a ∧ y > -Real.log a) → f a x < f a y)) :=
by sorry

end f_properties_l3579_357952


namespace quadratic_distinct_roots_l3579_357965

theorem quadratic_distinct_roots (a₁ a₂ a₃ a₄ : ℝ) (h : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄) :
  let discriminant := (a₁ + a₂ + a₃ + a₄)^2 - 4*(a₁*a₃ + a₂*a₄)
  discriminant > 0 := by
sorry

end quadratic_distinct_roots_l3579_357965


namespace glenn_spends_35_dollars_l3579_357988

/-- The cost of a movie ticket on Monday -/
def monday_price : ℕ := 5

/-- The cost of a movie ticket on Wednesday -/
def wednesday_price : ℕ := 2 * monday_price

/-- The cost of a movie ticket on Saturday -/
def saturday_price : ℕ := 5 * monday_price

/-- The total amount Glenn spends on movie tickets -/
def glenn_total_spent : ℕ := wednesday_price + saturday_price

/-- Theorem stating that Glenn spends $35 on movie tickets -/
theorem glenn_spends_35_dollars : glenn_total_spent = 35 := by
  sorry

end glenn_spends_35_dollars_l3579_357988


namespace division_and_addition_l3579_357916

theorem division_and_addition : (10 / (1/5)) + 6 = 56 := by
  sorry

end division_and_addition_l3579_357916


namespace bernardo_win_smallest_number_l3579_357998

def game_winner (N : ℕ) : Prop :=
  N ≤ 999 ∧ 8 * N + 600 < 1000 ∧ 8 * N + 700 ≥ 1000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem bernardo_win_smallest_number :
  ∃ (N : ℕ), N = 38 ∧ game_winner N ∧
  (∀ (M : ℕ), M < N → ¬game_winner M) ∧
  sum_of_digits N = 11 :=
sorry

end bernardo_win_smallest_number_l3579_357998


namespace expression_evaluation_l3579_357970

theorem expression_evaluation (a b : ℚ) (h1 : a = 2) (h2 : b = 1/2) :
  (a^3 + b^2)^2 - (a^3 - b^2)^2 = 8 := by sorry

end expression_evaluation_l3579_357970


namespace range_of_a_l3579_357960

def complex_number (a : ℝ) : ℂ := (1 - a * Complex.I) * (a + 2 * Complex.I)

def in_first_quadrant (z : ℂ) : Prop := Complex.re z > 0 ∧ Complex.im z > 0

theorem range_of_a (a : ℝ) :
  in_first_quadrant (complex_number a) → 0 < a ∧ a < Real.sqrt 2 := by sorry

end range_of_a_l3579_357960


namespace sqrt_of_sqrt_sixteen_l3579_357925

theorem sqrt_of_sqrt_sixteen : Real.sqrt (Real.sqrt 16) = 2 ∨ Real.sqrt (Real.sqrt 16) = -2 := by
  sorry

end sqrt_of_sqrt_sixteen_l3579_357925


namespace faster_train_length_l3579_357900

/-- The length of a train given its speed relative to another train and the time it takes to pass --/
def train_length (relative_speed : ℝ) (passing_time : ℝ) : ℝ :=
  relative_speed * passing_time

theorem faster_train_length :
  let faster_speed : ℝ := 108 * (1000 / 3600)  -- Convert km/h to m/s
  let slower_speed : ℝ := 36 * (1000 / 3600)   -- Convert km/h to m/s
  let relative_speed : ℝ := faster_speed - slower_speed
  let passing_time : ℝ := 17
  train_length relative_speed passing_time = 340 := by
  sorry

#check faster_train_length

end faster_train_length_l3579_357900


namespace additional_trays_is_ten_l3579_357989

/-- Represents the number of eggs in a tray -/
def eggs_per_tray : ℕ := 30

/-- Represents the initial number of trays to be collected -/
def initial_trays : ℕ := 10

/-- Represents the number of trays dropped -/
def dropped_trays : ℕ := 2

/-- Represents the total number of eggs sold -/
def total_eggs_sold : ℕ := 540

/-- Calculates the number of additional trays needed -/
def additional_trays : ℕ :=
  (total_eggs_sold - (initial_trays - dropped_trays) * eggs_per_tray) / eggs_per_tray

/-- Theorem stating that the number of additional trays is 10 -/
theorem additional_trays_is_ten : additional_trays = 10 := by
  sorry

end additional_trays_is_ten_l3579_357989


namespace sales_tax_difference_l3579_357942

/-- The price of the item before tax -/
def item_price : ℝ := 50

/-- The first sales tax rate -/
def tax_rate_1 : ℝ := 0.075

/-- The second sales tax rate -/
def tax_rate_2 : ℝ := 0.0625

/-- Theorem: The difference between the sales taxes is $0.625 -/
theorem sales_tax_difference : 
  item_price * tax_rate_1 - item_price * tax_rate_2 = 0.625 := by
  sorry

end sales_tax_difference_l3579_357942


namespace rachel_video_game_score_l3579_357904

/-- Rachel's video game scoring problem -/
theorem rachel_video_game_score :
  let level1_treasures : ℕ := 5
  let level1_points : ℕ := 9
  let level2_treasures : ℕ := 2
  let level2_points : ℕ := 12
  let level3_treasures : ℕ := 8
  let level3_points : ℕ := 15
  let total_score := 
    level1_treasures * level1_points +
    level2_treasures * level2_points +
    level3_treasures * level3_points
  total_score = 189 := by sorry

end rachel_video_game_score_l3579_357904


namespace reflection_y_transforms_points_l3579_357910

/-- Reflection in the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-(p.1), p.2)

theorem reflection_y_transforms_points :
  let C : ℝ × ℝ := (-3, 2)
  let D : ℝ × ℝ := (-4, -2)
  let C' : ℝ × ℝ := (3, 2)
  let D' : ℝ × ℝ := (4, -2)
  (reflect_y C = C') ∧ (reflect_y D = D') :=
by sorry

end reflection_y_transforms_points_l3579_357910


namespace highest_power_of_three_in_N_l3579_357954

def N : ℕ := sorry

-- Define the property that N is formed by writing down two-digit integers from 19 to 92 continuously
def is_valid_N (n : ℕ) : Prop := sorry

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem highest_power_of_three_in_N :
  is_valid_N N →
  ∃ m : ℕ, (sum_of_digits N = 3^2 * m) ∧ (m % 3 ≠ 0) := by
  sorry

end highest_power_of_three_in_N_l3579_357954


namespace player_a_wins_l3579_357964

/-- Represents a game state --/
structure GameState :=
  (current : ℕ)

/-- Defines a valid move in the game --/
def validMove (s : GameState) (next : ℕ) : Prop :=
  next > s.current ∧ next ≤ 2 * s.current - 1

/-- Defines the winning condition --/
def isWinningState (s : GameState) : Prop :=
  s.current = 2004

/-- Defines a winning strategy for Player A --/
def hasWinningStrategy (player : ℕ → GameState → Prop) : Prop :=
  ∀ s : GameState, s.current = 2 → 
    ∃ (strategy : GameState → ℕ),
      (∀ s, validMove s (strategy s)) ∧
      (∀ s, player 0 s → isWinningState (GameState.mk (strategy s)) ∨
        (∀ next, validMove (GameState.mk (strategy s)) next → 
          player 1 (GameState.mk next) → 
          player 0 (GameState.mk (strategy (GameState.mk next)))))

/-- The main theorem stating that Player A has a winning strategy --/
theorem player_a_wins : 
  ∃ (player : ℕ → GameState → Prop), hasWinningStrategy player :=
sorry

end player_a_wins_l3579_357964


namespace sector_central_angle_l3579_357992

theorem sector_central_angle (area : Real) (radius : Real) (h1 : area = 3 * Real.pi / 8) (h2 : radius = 1) :
  let central_angle := 2 * area / (radius ^ 2)
  central_angle = 3 * Real.pi / 4 := by
  sorry

end sector_central_angle_l3579_357992


namespace triangle_angle_calculation_l3579_357962

theorem triangle_angle_calculation (α β γ δ : ℝ) 
  (h1 : α = 120)
  (h2 : β = 30)
  (h3 : γ = 21)
  (h4 : α + (180 - α) = 180) : 
  180 - ((180 - α) + β + γ) = 69 := by
  sorry

end triangle_angle_calculation_l3579_357962


namespace jar_water_problem_l3579_357932

theorem jar_water_problem (S L : ℝ) (hS : S > 0) (hL : L > 0) (h_capacities : S ≠ L) : 
  let water := (1/5) * S
  (water = (1/4) * L) → ((2 * water) / L = 1/2) := by sorry

end jar_water_problem_l3579_357932


namespace final_output_is_127_l3579_357907

def flowchart_output : ℕ → ℕ
| 0 => 0
| (n + 1) => let a := flowchart_output n; if a < 100 then 2 * a + 1 else a

theorem final_output_is_127 : flowchart_output 7 = 127 := by
  sorry

end final_output_is_127_l3579_357907


namespace points_collinearity_l3579_357918

/-- Checks if three points are collinear -/
def are_collinear (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

theorem points_collinearity :
  (are_collinear 1 2 2 4 3 6) ∧
  ¬(are_collinear 2 3 (-2) 1 3 4) := by
  sorry

end points_collinearity_l3579_357918


namespace inequality_solutions_l3579_357977

theorem inequality_solutions (a : ℝ) (h1 : a < 0) (h2 : a ≤ -Real.rpow 2 (1/3)) :
  ∃ (w x y z : ℤ), w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
  (a^2 * |a + (w : ℝ)/a^2| + |1 + w| ≤ 1 - a^3) ∧
  (a^2 * |a + (x : ℝ)/a^2| + |1 + x| ≤ 1 - a^3) ∧
  (a^2 * |a + (y : ℝ)/a^2| + |1 + y| ≤ 1 - a^3) ∧
  (a^2 * |a + (z : ℝ)/a^2| + |1 + z| ≤ 1 - a^3) :=
sorry

end inequality_solutions_l3579_357977


namespace cos_six_arccos_two_fifths_l3579_357934

theorem cos_six_arccos_two_fifths :
  Real.cos (6 * Real.arccos (2/5)) = 12223/15625 := by
  sorry

end cos_six_arccos_two_fifths_l3579_357934


namespace compound_has_one_hydrogen_l3579_357974

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  hydrogen : ℕ
  bromine : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (h_weight o_weight br_weight : ℚ) : ℚ :=
  c.hydrogen * h_weight + c.bromine * br_weight + c.oxygen * o_weight

/-- The theorem stating that a compound with 1 Br, 3 O, and molecular weight 129 has 1 H atom -/
theorem compound_has_one_hydrogen :
  ∃ (c : Compound),
    c.bromine = 1 ∧
    c.oxygen = 3 ∧
    molecularWeight c 1 16 79.9 = 129 ∧
    c.hydrogen = 1 := by
  sorry


end compound_has_one_hydrogen_l3579_357974


namespace spencer_journey_distance_l3579_357903

def walking_distances : List Float := [1.2, 0.4, 0.6, 1.5]
def biking_distances : List Float := [1.8, 2]
def bus_distance : Float := 3

def biking_to_walking_factor : Float := 0.5
def bus_to_walking_factor : Float := 0.8

def total_walking_equivalent (walking : List Float) (biking : List Float) (bus : Float) 
  (bike_factor : Float) (bus_factor : Float) : Float :=
  (walking.sum) + 
  (biking.sum * bike_factor) + 
  (bus * bus_factor)

theorem spencer_journey_distance :
  total_walking_equivalent walking_distances biking_distances bus_distance
    biking_to_walking_factor bus_to_walking_factor = 8 := by
  sorry

end spencer_journey_distance_l3579_357903


namespace addison_raffle_tickets_l3579_357940

/-- The number of raffle tickets Addison sold on Friday -/
def friday_tickets : ℕ := 181

/-- The number of raffle tickets Addison sold on Saturday -/
def saturday_tickets : ℕ := 2 * friday_tickets

/-- The number of raffle tickets Addison sold on Sunday -/
def sunday_tickets : ℕ := 78

theorem addison_raffle_tickets :
  friday_tickets = 181 ∧
  saturday_tickets = 2 * friday_tickets ∧
  sunday_tickets = 78 ∧
  saturday_tickets = sunday_tickets + 284 :=
by sorry

end addison_raffle_tickets_l3579_357940


namespace tetrahedron_sum_l3579_357948

/-- A tetrahedron is a three-dimensional geometric shape with four faces, four vertices, and six edges. --/
structure Tetrahedron where
  edges : Nat
  corners : Nat
  faces : Nat

/-- The sum of edges, corners, and faces of a tetrahedron is 14. --/
theorem tetrahedron_sum (t : Tetrahedron) : t.edges + t.corners + t.faces = 14 := by
  sorry

#check tetrahedron_sum

end tetrahedron_sum_l3579_357948


namespace add_12345_seconds_to_10am_l3579_357915

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time -/
def initialTime : Time :=
  { hours := 10, minutes := 0, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 12345

/-- The expected final time -/
def expectedFinalTime : Time :=
  { hours := 13, minutes := 25, seconds := 45 }

theorem add_12345_seconds_to_10am :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end add_12345_seconds_to_10am_l3579_357915


namespace remaining_money_is_48_6_l3579_357917

/-- Calculates the remaining money in Country B's currency after shopping in Country A -/
def remaining_money_country_b (initial_amount : ℝ) (grocery_ratio : ℝ) (household_ratio : ℝ) 
  (personal_ratio : ℝ) (household_tax : ℝ) (personal_discount : ℝ) (exchange_rate : ℝ) : ℝ :=
  let groceries := initial_amount * grocery_ratio
  let household := initial_amount * household_ratio * (1 + household_tax)
  let personal := initial_amount * personal_ratio * (1 - personal_discount)
  let total_spent := groceries + household + personal
  let remaining_a := initial_amount - total_spent
  remaining_a * exchange_rate

/-- Theorem stating that the remaining money in Country B's currency is 48.6 units -/
theorem remaining_money_is_48_6 : 
  remaining_money_country_b 450 (3/5) (1/6) (1/10) 0.05 0.1 0.8 = 48.6 := by
  sorry

end remaining_money_is_48_6_l3579_357917


namespace symmetric_point_y_axis_l3579_357941

/-- Given a point P(4, -5) and its symmetric point P1 with respect to the y-axis, 
    prove that P1 has coordinates (-4, -5) -/
theorem symmetric_point_y_axis : 
  let P : ℝ × ℝ := (4, -5)
  let P1 : ℝ × ℝ := (-P.1, P.2)  -- Definition of symmetry with respect to y-axis
  P1 = (-4, -5) := by sorry

end symmetric_point_y_axis_l3579_357941


namespace train_speed_problem_l3579_357945

/-- Prove that given two trains on a 200 km track, where one starts at 7 am and the other at 8 am
    traveling towards each other, meeting at 12 pm, and the second train travels at 25 km/h,
    the speed of the first train is 20 km/h. -/
theorem train_speed_problem (total_distance : ℝ) (second_train_speed : ℝ) 
  (first_train_start_time : ℝ) (second_train_start_time : ℝ) (meeting_time : ℝ) :
  total_distance = 200 →
  second_train_speed = 25 →
  first_train_start_time = 7 →
  second_train_start_time = 8 →
  meeting_time = 12 →
  ∃ (first_train_speed : ℝ), first_train_speed = 20 :=
by sorry

end train_speed_problem_l3579_357945


namespace work_completion_time_l3579_357939

/-- Given that A can do a work in 9 days and A and B together can do the work in 6 days,
    prove that B can do the work alone in 18 days. -/
theorem work_completion_time (a_time b_time ab_time : ℝ) 
    (ha : a_time = 9)
    (hab : ab_time = 6)
    (h_work_rate : 1 / a_time + 1 / b_time = 1 / ab_time) : 
  b_time = 18 := by
sorry


end work_completion_time_l3579_357939


namespace ancient_chinese_math_problem_l3579_357983

theorem ancient_chinese_math_problem (x y : ℕ) : 
  (8 * x = y + 3) → (7 * x = y - 4) → ((y + 3) / 8 : ℚ) = ((y - 4) / 7 : ℚ) := by
  sorry

end ancient_chinese_math_problem_l3579_357983


namespace floor_plus_self_eq_nineteen_fourths_l3579_357943

theorem floor_plus_self_eq_nineteen_fourths :
  ∃! (x : ℚ), ⌊x⌋ + x = 19/4 :=
by
  -- The proof would go here
  sorry

end floor_plus_self_eq_nineteen_fourths_l3579_357943


namespace gretchen_walking_time_l3579_357931

/-- The number of minutes Gretchen should walk for every 90 minutes of sitting -/
def walking_time_per_90_min : ℕ := 10

/-- The number of minutes in 90 minutes -/
def sitting_time_per_break : ℕ := 90

/-- The number of hours Gretchen spends working at her desk -/
def work_hours : ℕ := 6

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the total walking time for Gretchen based on her work hours -/
def total_walking_time : ℕ :=
  (work_hours * minutes_per_hour / sitting_time_per_break) * walking_time_per_90_min

theorem gretchen_walking_time :
  total_walking_time = 40 := by
  sorry

end gretchen_walking_time_l3579_357931


namespace function_no_zeros_implies_a_less_than_neg_one_l3579_357981

theorem function_no_zeros_implies_a_less_than_neg_one (a : ℝ) : 
  (∀ x : ℝ, 4^x - 2^(x+1) - a ≠ 0) → a < -1 := by
  sorry

end function_no_zeros_implies_a_less_than_neg_one_l3579_357981


namespace polynomial_factorization_l3579_357959

theorem polynomial_factorization (a b c : ℝ) :
  (a - 2*b) * (a - 2*b - 4) + 4 - c^2 = ((a - 2*b) - 2 + c) * ((a - 2*b) - 2 - c) := by
  sorry

end polynomial_factorization_l3579_357959


namespace final_water_fraction_l3579_357956

def container_volume : ℚ := 20
def replacement_volume : ℚ := 5
def num_replacements : ℕ := 5

def water_fraction_after_replacements : ℚ := (3/4) ^ num_replacements

theorem final_water_fraction :
  water_fraction_after_replacements = 243/1024 :=
by sorry

end final_water_fraction_l3579_357956


namespace union_M_N_intersection_M_complement_N_l3579_357913

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 2)}
def N : Set ℝ := {x | x < 1 ∨ x > 3}

-- Theorem for M ∪ N
theorem union_M_N : M ∪ N = {x | x < 1 ∨ x ≥ 2} := by sorry

-- Theorem for M ∩ (U \ N)
theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = {x | 2 ≤ x ∧ x ≤ 3} := by sorry

end union_M_N_intersection_M_complement_N_l3579_357913


namespace rectangle_area_l3579_357922

/-- Given a rectangle with perimeter 50 cm and length 13 cm, its area is 156 cm² -/
theorem rectangle_area (perimeter width length : ℝ) : 
  perimeter = 50 → 
  length = 13 → 
  width = (perimeter - 2 * length) / 2 → 
  length * width = 156 := by
  sorry

end rectangle_area_l3579_357922


namespace derivative_of_y_l3579_357985

noncomputable def y (x : ℝ) : ℝ := Real.sin x - 2^x

theorem derivative_of_y (x : ℝ) :
  deriv y x = Real.cos x - 2^x * Real.log 2 := by sorry

end derivative_of_y_l3579_357985


namespace expression_equals_zero_l3579_357994

theorem expression_equals_zero : 2 * 2^5 - 8^58 / 8^56 = 0 := by sorry

end expression_equals_zero_l3579_357994
