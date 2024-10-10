import Mathlib

namespace max_sides_convex_polygon_is_maximum_max_sides_convex_polygon_is_convex_l2606_260602

/-- The maximum number of sides for a convex polygon with interior angles
    forming an arithmetic sequence with a common difference of 1°. -/
def max_sides_convex_polygon : ℕ := 27

/-- The common difference of the arithmetic sequence formed by the interior angles. -/
def common_difference : ℝ := 1

/-- Predicate to check if a polygon is convex based on its number of sides. -/
def is_convex (n : ℕ) : Prop :=
  let α : ℝ := (n - 2) * 180 / n - (n - 1) / 2
  α > 0 ∧ α + (n - 1) * common_difference < 180

/-- Theorem stating that max_sides_convex_polygon is the maximum number of sides
    for a convex polygon with interior angles forming an arithmetic sequence
    with a common difference of 1°. -/
theorem max_sides_convex_polygon_is_maximum :
  ∀ n : ℕ, n > max_sides_convex_polygon → ¬(is_convex n) :=
sorry

/-- Theorem stating that max_sides_convex_polygon satisfies the convexity condition. -/
theorem max_sides_convex_polygon_is_convex :
  is_convex max_sides_convex_polygon :=
sorry

end max_sides_convex_polygon_is_maximum_max_sides_convex_polygon_is_convex_l2606_260602


namespace complex_number_in_first_quadrant_l2606_260657

theorem complex_number_in_first_quadrant :
  let z : ℂ := (1 : ℂ) / (1 + Complex.I) + Complex.I
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_first_quadrant_l2606_260657


namespace election_win_percentage_l2606_260614

theorem election_win_percentage 
  (total_votes : ℕ) 
  (geoff_percentage : ℚ) 
  (additional_votes_needed : ℕ) 
  (h1 : total_votes = 6000)
  (h2 : geoff_percentage = 1/200)  -- 0.5% as a rational number
  (h3 : additional_votes_needed = 3000) :
  (((geoff_percentage * total_votes + additional_votes_needed) / total_votes) : ℚ) = 101/200 := by
sorry

end election_win_percentage_l2606_260614


namespace circle_ratio_after_increase_l2606_260616

theorem circle_ratio_after_increase (r : ℝ) : 
  let new_radius := r + 2
  let new_diameter := 2 * new_radius
  let new_circumference := 2 * Real.pi * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end circle_ratio_after_increase_l2606_260616


namespace h_eq_f_reflected_and_shifted_l2606_260644

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the function h obtained from f by reflection and shift
def h (x : ℝ) : ℝ := f (6 - x)

-- Theorem stating the relationship between h and f
theorem h_eq_f_reflected_and_shifted :
  ∀ x : ℝ, h f x = f (6 - x) := by sorry

end h_eq_f_reflected_and_shifted_l2606_260644


namespace weight_of_replaced_person_l2606_260679

/-- Given a group of 9 persons where the average weight increases by 1.5 kg
    after replacing one person with a new person weighing 78.5 kg,
    prove that the weight of the replaced person was 65 kg. -/
theorem weight_of_replaced_person
  (n : ℕ) -- number of persons in the group
  (avg_increase : ℝ) -- increase in average weight
  (new_weight : ℝ) -- weight of the new person
  (h1 : n = 9) -- there are 9 persons in the group
  (h2 : avg_increase = 1.5) -- average weight increases by 1.5 kg
  (h3 : new_weight = 78.5) -- new person weighs 78.5 kg
  : ℝ :=
by
  sorry

#check weight_of_replaced_person

end weight_of_replaced_person_l2606_260679


namespace geometric_sequence_a3_l2606_260648

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 5 = 82 →
  a 2 * a 4 = 81 →
  a 3 = 9 := by
  sorry

end geometric_sequence_a3_l2606_260648


namespace geometric_series_ratio_l2606_260623

/-- Given a geometric series with first term a and common ratio r,
    if the sum of the series is 20 and the sum of terms involving odd powers of r is 8,
    then r = 1/4 -/
theorem geometric_series_ratio (a r : ℝ) 
  (h1 : a / (1 - r) = 20)
  (h2 : a * r / (1 - r^2) = 8) :
  r = 1/4 := by
  sorry

end geometric_series_ratio_l2606_260623


namespace dan_minimum_speed_l2606_260669

/-- Proves the minimum speed Dan must exceed to arrive before Cara -/
theorem dan_minimum_speed (distance : ℝ) (cara_speed : ℝ) (dan_delay : ℝ) :
  distance = 180 →
  cara_speed = 30 →
  dan_delay = 1 →
  ∃ (min_speed : ℝ), min_speed > 36 ∧
    ∀ (dan_speed : ℝ), dan_speed > min_speed →
      distance / dan_speed < distance / cara_speed - dan_delay := by
  sorry

#check dan_minimum_speed

end dan_minimum_speed_l2606_260669


namespace multiple_identification_l2606_260621

/-- Given two integers a and b that are multiples of n, and q is the set of consecutive integers
    between a and b (inclusive), prove that if q contains 11 multiples of n and 21 multiples of 7,
    then n = 14. -/
theorem multiple_identification (a b n : ℕ) (q : Finset ℕ) (h1 : a ∣ n) (h2 : b ∣ n)
    (h3 : q = Finset.Icc a b) (h4 : (q.filter (· ∣ n)).card = 11)
    (h5 : (q.filter (· ∣ 7)).card = 21) : n = 14 := by
  sorry

end multiple_identification_l2606_260621


namespace f_properties_l2606_260639

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + Real.sin (x + Real.pi / 2)

theorem f_properties :
  (∀ x, f (-x) = f x) ∧
  (∃ x, f x = 55 / 8) ∧
  (∃ x, f x = -9 / 8) ∧
  (∀ x, f x ≤ 55 / 8) ∧
  (∀ x, f x ≥ -9 / 8) :=
by sorry

end f_properties_l2606_260639


namespace all_four_digit_numbers_generated_l2606_260634

/-- Represents the operations that can be performed on a number -/
inductive Operation
  | mul2sub2 : Operation
  | mul3add4 : Operation
  | add7 : Operation

/-- Applies an operation to a number -/
def applyOperation (op : Operation) (x : ℕ) : ℕ :=
  match op with
  | Operation.mul2sub2 => 2 * x - 2
  | Operation.mul3add4 => 3 * x + 4
  | Operation.add7 => x + 7

/-- Returns true if the number is four digits -/
def isFourDigits (n : ℕ) : Bool :=
  1000 ≤ n ∧ n ≤ 9999

/-- The set of all four-digit numbers -/
def fourDigitNumbers : Set ℕ :=
  {n : ℕ | isFourDigits n}

/-- The set of numbers that can be generated from 1 using the given operations -/
def generatedNumbers : Set ℕ :=
  {n : ℕ | ∃ (ops : List Operation), n = ops.foldl (fun acc op => applyOperation op acc) 1}

/-- Theorem stating that all four-digit numbers can be generated -/
theorem all_four_digit_numbers_generated :
  fourDigitNumbers ⊆ generatedNumbers :=
sorry

end all_four_digit_numbers_generated_l2606_260634


namespace final_breath_holding_time_l2606_260673

def breath_holding_progress (initial_time : ℝ) : ℝ :=
  let week1 := initial_time * 2
  let week2 := week1 * 2
  let week3 := week2 * 1.5
  week3

theorem final_breath_holding_time :
  breath_holding_progress 10 = 60 := by
  sorry

end final_breath_holding_time_l2606_260673


namespace least_three_digit_multiple_of_nine_l2606_260620

theorem least_three_digit_multiple_of_nine : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 9 ∣ n → n ≥ 108 :=
by sorry

end least_three_digit_multiple_of_nine_l2606_260620


namespace carousel_horse_ratio_l2606_260607

theorem carousel_horse_ratio : 
  ∀ (purple green gold : ℕ),
  purple > 0 →
  green = 2 * purple →
  gold = green / 6 →
  3 + purple + green + gold = 33 →
  (purple : ℚ) / 3 = 3 / 1 :=
by
  sorry

end carousel_horse_ratio_l2606_260607


namespace snail_reaches_top_in_ten_days_l2606_260691

/-- Represents the snail's climbing problem -/
structure SnailClimb where
  treeHeight : ℕ
  climbUp : ℕ
  slideDown : ℕ

/-- Calculates the number of days needed for the snail to reach the top of the tree -/
def daysToReachTop (s : SnailClimb) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the snail reaches the top in 10 days -/
theorem snail_reaches_top_in_ten_days :
  let s : SnailClimb := ⟨24, 6, 4⟩
  daysToReachTop s = 10 := by
  sorry

end snail_reaches_top_in_ten_days_l2606_260691


namespace flower_arrangement_count_l2606_260677

/-- The number of different pots of flowers --/
def total_pots : ℕ := 7

/-- The number of pots to be selected --/
def selected_pots : ℕ := 5

/-- The number of pots not allowed in the center --/
def restricted_pots : ℕ := 2

/-- The function to calculate the number of arrangements --/
def flower_arrangements (n m k : ℕ) : ℕ := sorry

theorem flower_arrangement_count :
  flower_arrangements total_pots selected_pots restricted_pots = 1800 := by sorry

end flower_arrangement_count_l2606_260677


namespace acute_angle_probability_l2606_260617

/-- Represents a clock with hour and minute hands. -/
structure Clock :=
  (hour_hand : ℝ)
  (minute_hand : ℝ)

/-- The angle between the hour and minute hands is acute. -/
def is_acute_angle (c : Clock) : Prop :=
  let angle := (c.minute_hand - c.hour_hand + 12) % 12
  angle < 3 ∨ angle > 9

/-- A random clock stop event. -/
def random_clock_stop : Clock → Prop :=
  sorry

/-- The probability of an event occurring. -/
def probability (event : Clock → Prop) : ℝ :=
  sorry

/-- The main theorem: The probability of an acute angle between clock hands is 1/2. -/
theorem acute_angle_probability :
  probability is_acute_angle = 1/2 :=
sorry

end acute_angle_probability_l2606_260617


namespace max_students_above_median_l2606_260684

theorem max_students_above_median (n : ℕ) (h : n = 81) :
  (n + 1) / 2 = (n + 1) / 2 ∧ (n - (n + 1) / 2) = 40 := by
  sorry

end max_students_above_median_l2606_260684


namespace junk_mail_distribution_l2606_260622

-- Define the number of houses on the block
def num_houses : ℕ := 6

-- Define the total number of junk mail pieces
def total_junk_mail : ℕ := 24

-- Define the function to calculate junk mail per house
def junk_mail_per_house (houses : ℕ) (total_mail : ℕ) : ℕ :=
  total_mail / houses

-- Theorem statement
theorem junk_mail_distribution :
  junk_mail_per_house num_houses total_junk_mail = 4 := by
  sorry

end junk_mail_distribution_l2606_260622


namespace average_carnations_value_l2606_260619

/-- The average number of carnations in Trevor's bouquets -/
def average_carnations : ℚ :=
  let bouquets : List ℕ := [9, 23, 13, 36, 28, 45]
  (bouquets.sum : ℚ) / bouquets.length

/-- Proof that the average number of carnations is 25.67 -/
theorem average_carnations_value :
  average_carnations = 25.67 := by
  sorry

end average_carnations_value_l2606_260619


namespace expression_evaluation_l2606_260693

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 - 1
  (1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6)) = Real.sqrt 2 := by
  sorry

end expression_evaluation_l2606_260693


namespace maxim_birth_probability_l2606_260678

/-- The year Maxim starts first grade -/
def start_year : ℕ := 2014

/-- Maxim's age when starting first grade -/
def start_age : ℕ := 6

/-- The day of the year when Maxim starts first grade (1st September) -/
def start_day : ℕ := 244

/-- The number of days in a year (assuming non-leap year) -/
def days_in_year : ℕ := 365

/-- The year we're interested in for Maxim's birth -/
def birth_year_of_interest : ℕ := 2008

/-- The number of days from 1st January to 31st August in 2008 (leap year) -/
def days_in_2008_until_august : ℕ := 244

theorem maxim_birth_probability :
  let total_possible_days := days_in_year
  let favorable_days := days_in_2008_until_august
  (favorable_days : ℚ) / total_possible_days = 244 / 365 := by
  sorry

end maxim_birth_probability_l2606_260678


namespace binomial_coefficient_equality_l2606_260611

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 28 x = Nat.choose 28 (3 * x - 8)) → (x = 4 ∨ x = 9) := by
  sorry

end binomial_coefficient_equality_l2606_260611


namespace combined_future_age_l2606_260668

-- Define the current age of Hurley
def hurley_current_age : ℕ := 14

-- Define the age difference between Richard and Hurley
def age_difference : ℕ := 20

-- Define the number of years into the future
def years_future : ℕ := 40

-- Theorem to prove
theorem combined_future_age :
  (hurley_current_age + years_future) + (hurley_current_age + age_difference + years_future) = 128 := by
  sorry

end combined_future_age_l2606_260668


namespace hexagon_interior_angles_sum_l2606_260636

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon is a polygon with 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The sum of the interior angles of a hexagon is 720 degrees -/
theorem hexagon_interior_angles_sum :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end hexagon_interior_angles_sum_l2606_260636


namespace six_digit_number_theorem_l2606_260689

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ (a b c d e f : ℕ),
    n = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧
    a ≠ 0 ∧ 
    10000 * b + 1000 * c + 100 * d + 10 * e + f + 100000 * a = 3 * n

theorem six_digit_number_theorem :
  ∀ n : ℕ, is_valid_number n → (n = 142857 ∨ n = 285714) :=
sorry

end six_digit_number_theorem_l2606_260689


namespace opposite_numbers_expression_l2606_260685

theorem opposite_numbers_expression (a b : ℝ) (h1 : a + b = 0) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  (a + b - 1) * (a / b + 1) = 0 := by
  sorry

end opposite_numbers_expression_l2606_260685


namespace decimal_point_problem_l2606_260606

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 100 * x = 9 / x) : x = 3 / 10 := by
  sorry

end decimal_point_problem_l2606_260606


namespace sin_b_in_arithmetic_sequence_triangle_l2606_260613

/-- In a triangle ABC where the interior angles form an arithmetic sequence, sin B = √3/2 -/
theorem sin_b_in_arithmetic_sequence_triangle (A B C : Real) : 
  A + B + C = Real.pi →  -- Sum of angles in radians
  A + C = 2 * B →        -- Arithmetic sequence property
  Real.sin B = Real.sqrt 3 / 2 := by
sorry

end sin_b_in_arithmetic_sequence_triangle_l2606_260613


namespace balls_after_5000_steps_l2606_260624

/-- Converts a natural number to its base-6 representation --/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Sums the digits in a list of natural numbers --/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.foldl (· + ·) 0

/-- Represents the ball placement process --/
def ballPlacement (steps : ℕ) : ℕ :=
  sumDigits (toBase6 steps)

/-- The main theorem stating that after 5000 steps, there are 13 balls in the boxes --/
theorem balls_after_5000_steps :
  ballPlacement 5000 = 13 := by
  sorry

end balls_after_5000_steps_l2606_260624


namespace base16_to_base4_C2A_l2606_260649

/-- Represents a digit in base 16 --/
inductive Base16Digit
| C | Two | A

/-- Represents a number in base 16 --/
def Base16Number := List Base16Digit

/-- Represents a digit in base 4 --/
inductive Base4Digit
| Zero | One | Two | Three

/-- Represents a number in base 4 --/
def Base4Number := List Base4Digit

/-- Converts a Base16Number to a Base4Number --/
def convertBase16ToBase4 (n : Base16Number) : Base4Number := sorry

/-- The main theorem --/
theorem base16_to_base4_C2A :
  convertBase16ToBase4 [Base16Digit.C, Base16Digit.Two, Base16Digit.A] =
  [Base4Digit.Three, Base4Digit.Zero, Base4Digit.Zero,
   Base4Digit.Two, Base4Digit.Two, Base4Digit.Two] :=
by sorry

end base16_to_base4_C2A_l2606_260649


namespace repeating_decimal_equals_fraction_l2606_260663

/-- The repeating decimal 0.565656... -/
def repeating_decimal : ℚ := 0.56565656

/-- The fraction 56/99 -/
def target_fraction : ℚ := 56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to 56/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end repeating_decimal_equals_fraction_l2606_260663


namespace chips_cost_split_l2606_260605

theorem chips_cost_split (num_friends : ℕ) (num_bags : ℕ) (cost_per_bag : ℕ) :
  num_friends = 3 →
  num_bags = 5 →
  cost_per_bag = 3 →
  (num_bags * cost_per_bag) / num_friends = 5 :=
by
  sorry

end chips_cost_split_l2606_260605


namespace min_value_expression_l2606_260610

theorem min_value_expression (a b : ℤ) (h : a > b) :
  (2 : ℝ) ≤ ((2*a + 3*b : ℝ) / (a - 2*b : ℝ)) + ((a - 2*b : ℝ) / (2*a + 3*b : ℝ)) ∧
  ∃ (a' b' : ℤ), a' > b' ∧ ((2*a' + 3*b' : ℝ) / (a' - 2*b' : ℝ)) + ((a' - 2*b' : ℝ) / (2*a' + 3*b' : ℝ)) = 2 :=
by sorry

end min_value_expression_l2606_260610


namespace lucas_avocados_l2606_260637

/-- Calculates the number of avocados bought given initial money, cost per avocado, and change --/
def avocados_bought (initial_money change cost_per_avocado : ℚ) : ℚ :=
  (initial_money - change) / cost_per_avocado

/-- Proves that Lucas bought 3 avocados --/
theorem lucas_avocados :
  let initial_money : ℚ := 20
  let change : ℚ := 14
  let cost_per_avocado : ℚ := 2
  avocados_bought initial_money change cost_per_avocado = 3 := by
  sorry

end lucas_avocados_l2606_260637


namespace skateboard_cost_l2606_260655

theorem skateboard_cost (toy_cars_cost toy_trucks_cost total_toys_cost : ℚ)
  (h1 : toy_cars_cost = 14.88)
  (h2 : toy_trucks_cost = 5.86)
  (h3 : total_toys_cost = 25.62) :
  total_toys_cost - (toy_cars_cost + toy_trucks_cost) = 4.88 := by
sorry

end skateboard_cost_l2606_260655


namespace negation_equivalence_l2606_260671

variable (a : ℝ)

def original_proposition : Prop :=
  ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0

theorem negation_equivalence :
  (¬ original_proposition a) ↔ (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) :=
by sorry

end negation_equivalence_l2606_260671


namespace original_equals_scientific_l2606_260697

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The original number we want to express in scientific notation -/
def original_number : ℝ := 0.00000164

/-- The scientific notation representation we want to prove is correct -/
def scientific_rep : ScientificNotation := {
  coefficient := 1.64
  exponent := -6
  coefficient_range := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : original_number = scientific_rep.coefficient * (10 : ℝ) ^ scientific_rep.exponent := by sorry

end original_equals_scientific_l2606_260697


namespace floor_painting_possibilities_l2606_260675

theorem floor_painting_possibilities :
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      (p.2 > p.1 ∧ 
       (p.1 - 4) * (p.2 - 4) = 2 * p.1 * p.2 / 3 ∧
       p.1 > 0 ∧ p.2 > 0)) ∧
    s.card = 2 := by
  sorry

end floor_painting_possibilities_l2606_260675


namespace bus_speed_relation_l2606_260653

/-- Represents the speed and stoppage characteristics of a bus -/
structure Bus where
  speed_with_stops : ℝ
  stop_time : ℝ
  speed_without_stops : ℝ

/-- Theorem stating the relationship between bus speeds and stop time -/
theorem bus_speed_relation (b : Bus) 
  (h1 : b.speed_with_stops = 12)
  (h2 : b.stop_time = 45)
  : b.speed_without_stops = 48 := by
  sorry

#check bus_speed_relation

end bus_speed_relation_l2606_260653


namespace towels_used_theorem_l2606_260651

/-- Calculates the number of towels used in a gym based on guest distribution and staff usage. -/
def towels_used (first_hour_guests : ℕ) : ℕ :=
  let second_hour_guests := first_hour_guests + first_hour_guests / 5
  let third_hour_guests := second_hour_guests + second_hour_guests / 4
  let fourth_hour_guests := third_hour_guests + third_hour_guests / 3
  let fifth_hour_guests := fourth_hour_guests - fourth_hour_guests * 3 / 20
  let sixth_hour_guests := fifth_hour_guests
  let seventh_hour_guests := sixth_hour_guests - sixth_hour_guests * 3 / 10
  let eighth_hour_guests := seventh_hour_guests / 2
  let total_guests := first_hour_guests + second_hour_guests + third_hour_guests + 
                      fourth_hour_guests + fifth_hour_guests + sixth_hour_guests + 
                      seventh_hour_guests + eighth_hour_guests
  let three_towel_guests := total_guests / 10
  let two_towel_guests := total_guests * 6 / 10
  let one_towel_guests := total_guests * 3 / 10
  let guest_towels := three_towel_guests * 3 + two_towel_guests * 2 + one_towel_guests
  guest_towels + 20

/-- The theorem stating that given 40 guests in the first hour, the total towels used is 807. -/
theorem towels_used_theorem : towels_used 40 = 807 := by
  sorry

#eval towels_used 40

end towels_used_theorem_l2606_260651


namespace expand_equals_difference_of_squares_l2606_260626

theorem expand_equals_difference_of_squares (x y : ℝ) :
  (-x + 2*y) * (-x - 2*y) = x^2 - 4*y^2 := by
  sorry

end expand_equals_difference_of_squares_l2606_260626


namespace two_cars_problem_l2606_260670

/-- Two cars problem -/
theorem two_cars_problem 
  (distance_between_villages : ℝ) 
  (speed_car_A speed_car_B : ℝ) 
  (target_distance : ℝ) :
  distance_between_villages = 18 →
  speed_car_A = 54 →
  speed_car_B = 36 →
  target_distance = 45 →
  -- Case 1: Cars driving towards each other
  (distance_between_villages + target_distance) / (speed_car_A + speed_car_B) = 0.7 ∧
  -- Case 2a: Cars driving in same direction, faster car behind
  (target_distance + distance_between_villages) / (speed_car_A - speed_car_B) = 3.5 ∧
  -- Case 2b: Cars driving in same direction, faster car ahead
  (target_distance - distance_between_villages) / (speed_car_A - speed_car_B) = 1.5 :=
by sorry

end two_cars_problem_l2606_260670


namespace square_side_length_l2606_260695

/-- Given a square ABCD with side length x, prove that x = 12 under the given conditions --/
theorem square_side_length (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive
  (h2 : x^2 - (1/2) * ((x-5) * (x-4)) - (7/2) * (x-7) - 2*(x-1) - 3.5 = 78) : x = 12 := by
  sorry

end square_side_length_l2606_260695


namespace principal_calculation_l2606_260686

theorem principal_calculation (P R : ℝ) 
  (h1 : P + (P * R * 2) / 100 = 660)
  (h2 : P + (P * R * 7) / 100 = 1020) : 
  P = 516 := by
  sorry

end principal_calculation_l2606_260686


namespace infiniteSeries_eq_three_halves_l2606_260665

/-- The sum of the infinite series Σ(k/(3^k)) for k from 1 to ∞ -/
noncomputable def infiniteSeries : ℝ := ∑' k, k / (3 ^ k)

/-- Theorem stating that the sum of the infinite series Σ(k/(3^k)) for k from 1 to ∞ is equal to 3/2 -/
theorem infiniteSeries_eq_three_halves : infiniteSeries = 3/2 := by sorry

end infiniteSeries_eq_three_halves_l2606_260665


namespace expression_simplification_l2606_260603

theorem expression_simplification (x y : ℚ) 
  (hx : x = -1/3) (hy : y = -2) : 
  2 * (x^2 - 2*x^2*y) - (3*(x^2 - x*y^2) - (x^2*y - 2*x*y^2 + x^2)) = -2/3 := by
  sorry

end expression_simplification_l2606_260603


namespace mrs_taylor_purchase_cost_l2606_260627

/-- Calculates the total cost of smart televisions and soundbars with discounts -/
def total_cost (tv_count : ℕ) (tv_price : ℚ) (tv_discount : ℚ)
                (soundbar_count : ℕ) (soundbar_price : ℚ) (soundbar_discount : ℚ) : ℚ :=
  let tv_total := tv_count * tv_price * (1 - tv_discount)
  let soundbar_total := soundbar_count * soundbar_price * (1 - soundbar_discount)
  tv_total + soundbar_total

/-- Theorem stating that Mrs. Taylor's purchase totals $2085 -/
theorem mrs_taylor_purchase_cost :
  total_cost 2 750 0.15 3 300 0.10 = 2085 := by
  sorry

end mrs_taylor_purchase_cost_l2606_260627


namespace sum_of_decimals_l2606_260628

/-- The sum of 5.47 and 2.359 is equal to 7.829 -/
theorem sum_of_decimals : (5.47 : ℚ) + (2.359 : ℚ) = (7.829 : ℚ) := by
  sorry

end sum_of_decimals_l2606_260628


namespace solution_set_f_less_g_range_of_a_l2606_260659

-- Define the functions f and g
def f (x : ℝ) := abs (x - 4)
def g (x : ℝ) := abs (2 * x + 1)

-- Statement 1
theorem solution_set_f_less_g :
  {x : ℝ | f x < g x} = {x : ℝ | x < -5 ∨ x > 1} := by sorry

-- Statement 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * f x + g x > a * x) ↔ a ∈ Set.Icc (-4) (9/4) := by sorry

end solution_set_f_less_g_range_of_a_l2606_260659


namespace factorial_of_factorial_divided_by_factorial_l2606_260654

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end factorial_of_factorial_divided_by_factorial_l2606_260654


namespace quadratic_monotonic_condition_l2606_260629

-- Define the quadratic function
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 1

-- Define monotonicity in an interval
def monotonic_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ f y < f x)

-- Theorem statement
theorem quadratic_monotonic_condition (t : ℝ) :
  monotonic_in_interval (f t) 1 3 → t ≤ 1 ∨ t ≥ 3 := by
  sorry

end quadratic_monotonic_condition_l2606_260629


namespace second_polygon_sides_l2606_260672

theorem second_polygon_sides (perimeter : ℝ) (side_length_second : ℝ) : 
  perimeter > 0 → side_length_second > 0 →
  perimeter = 50 * (3 * side_length_second) →
  perimeter = 150 * side_length_second := by
  sorry

end second_polygon_sides_l2606_260672


namespace faster_speed_calculation_l2606_260642

theorem faster_speed_calculation (total_distance : ℝ) (initial_speed : ℝ) (initial_time : ℝ) 
  (second_time : ℝ) (remaining_distance : ℝ) 
  (h1 : total_distance = 600)
  (h2 : initial_speed = 50)
  (h3 : initial_time = 3)
  (h4 : second_time = 4)
  (h5 : remaining_distance = 130) : 
  ∃ faster_speed : ℝ, faster_speed = 80 ∧ 
  total_distance = initial_speed * initial_time + faster_speed * second_time + remaining_distance :=
by
  sorry


end faster_speed_calculation_l2606_260642


namespace max_square_side_length_56_24_l2606_260692

/-- The maximum side length of squares that can be cut from a rectangular paper -/
def max_square_side_length (length width : ℕ) : ℕ := Nat.gcd length width

theorem max_square_side_length_56_24 :
  max_square_side_length 56 24 = 8 := by sorry

end max_square_side_length_56_24_l2606_260692


namespace three_four_five_pythagorean_one_two_five_not_pythagorean_two_three_four_not_pythagorean_four_five_six_not_pythagorean_only_three_four_five_pythagorean_l2606_260609

/-- A function that checks if three numbers form a Pythagorean triple --/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- Theorem stating that (3, 4, 5) is a Pythagorean triple --/
theorem three_four_five_pythagorean : isPythagoreanTriple 3 4 5 := by
  sorry

/-- Theorem stating that (1, 2, 5) is not a Pythagorean triple --/
theorem one_two_five_not_pythagorean : ¬ isPythagoreanTriple 1 2 5 := by
  sorry

/-- Theorem stating that (2, 3, 4) is not a Pythagorean triple --/
theorem two_three_four_not_pythagorean : ¬ isPythagoreanTriple 2 3 4 := by
  sorry

/-- Theorem stating that (4, 5, 6) is not a Pythagorean triple --/
theorem four_five_six_not_pythagorean : ¬ isPythagoreanTriple 4 5 6 := by
  sorry

/-- Main theorem stating that among the given sets, only (3, 4, 5) is a Pythagorean triple --/
theorem only_three_four_five_pythagorean :
  (isPythagoreanTriple 3 4 5) ∧
  (¬ isPythagoreanTriple 1 2 5) ∧
  (¬ isPythagoreanTriple 2 3 4) ∧
  (¬ isPythagoreanTriple 4 5 6) := by
  sorry

end three_four_five_pythagorean_one_two_five_not_pythagorean_two_three_four_not_pythagorean_four_five_six_not_pythagorean_only_three_four_five_pythagorean_l2606_260609


namespace two_zeros_sum_less_than_neg_two_l2606_260688

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x * Real.exp x
def g (x : ℝ) : ℝ := (x + 1)^2

-- Define the function G
def G (a : ℝ) (x : ℝ) : ℝ := a * f x + g x

-- Theorem statement
theorem two_zeros_sum_less_than_neg_two (a : ℝ) (x₁ x₂ : ℝ) :
  a > 0 →
  G a x₁ = 0 →
  G a x₂ = 0 →
  x₁ ≠ x₂ →
  x₁ + x₂ + 2 < 0 :=
by sorry

end

end two_zeros_sum_less_than_neg_two_l2606_260688


namespace probability_different_topics_l2606_260600

/-- The number of topics in the essay competition -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating the probability of two students selecting different topics -/
theorem probability_different_topics :
  (num_topics : ℚ) * (num_topics - 1) / (num_topics * num_topics) = prob_different_topics :=
sorry

end probability_different_topics_l2606_260600


namespace todd_spending_proof_l2606_260667

/-- Calculates the total amount Todd spent given the prices of items, discount rate, and tax rate -/
def todd_spending (candy_price cookies_price soda_price : ℚ) (discount_rate tax_rate : ℚ) : ℚ :=
  let discounted_candy := candy_price * (1 - discount_rate)
  let subtotal := discounted_candy + cookies_price + soda_price
  let total := subtotal * (1 + tax_rate)
  total

/-- Proves that Todd's total spending is $5.53 given the problem conditions -/
theorem todd_spending_proof :
  todd_spending 1.14 2.39 1.75 0.1 0.07 = 5.53 := by
  sorry

end todd_spending_proof_l2606_260667


namespace linear_equation_equivalence_l2606_260666

theorem linear_equation_equivalence (x y : ℝ) :
  (3 * x - y + 5 = 0) ↔ (y = 3 * x + 5) := by
  sorry

end linear_equation_equivalence_l2606_260666


namespace equation_solution_range_l2606_260699

theorem equation_solution_range (x k : ℝ) : 2 * x + 3 * k = 1 → x < 0 → k > 1/3 := by
  sorry

end equation_solution_range_l2606_260699


namespace monthly_salary_is_6250_l2606_260645

/-- Calculates the monthly salary given savings rate, expense increase, and new savings amount -/
def calculate_salary (savings_rate : ℚ) (expense_increase : ℚ) (new_savings : ℚ) : ℚ :=
  new_savings / (savings_rate - (1 - savings_rate) * expense_increase)

/-- Theorem stating that under the given conditions, the monthly salary is 6250 -/
theorem monthly_salary_is_6250 :
  let savings_rate : ℚ := 1/5
  let expense_increase : ℚ := 1/5
  let new_savings : ℚ := 250
  calculate_salary savings_rate expense_increase new_savings = 6250 := by
sorry

#eval calculate_salary (1/5) (1/5) 250

end monthly_salary_is_6250_l2606_260645


namespace first_month_sale_is_7435_l2606_260647

/-- Calculates the sale in the first month given the sales for months 2-6 and the average sale --/
def first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale2 + sale3 + sale4 + sale5 + sale6)

/-- Proves that the first month's sale is 7435 given the problem conditions --/
theorem first_month_sale_is_7435 :
  first_month_sale 7920 7855 8230 7560 6000 7500 = 7435 := by
  sorry

end first_month_sale_is_7435_l2606_260647


namespace circle_intersection_problem_l2606_260612

/-- Given two circles C₁ and C₂ with equations as defined below, prove that the value of a is 4 -/
theorem circle_intersection_problem (a b x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 + y₁^2 - 2*x₁ + 4*y₁ - b^2 + 5 = 0) →  -- C₁ equation for point A
  (x₂^2 + y₂^2 - 2*x₂ + 4*y₂ - b^2 + 5 = 0) →  -- C₁ equation for point B
  (x₁^2 + y₁^2 - 2*(a-6)*x₁ - 2*a*y₁ + 2*a^2 - 12*a + 27 = 0) →  -- C₂ equation for point A
  (x₂^2 + y₂^2 - 2*(a-6)*x₂ - 2*a*y₂ + 2*a^2 - 12*a + 27 = 0) →  -- C₂ equation for point B
  ((y₁ + y₂)/(x₁ + x₂) + (x₁ - x₂)/(y₁ - y₂) = 0) →  -- Given condition
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →  -- Distinct points condition
  a = 4 := by
sorry

end circle_intersection_problem_l2606_260612


namespace decimal_to_fraction_l2606_260694

theorem decimal_to_fraction :
  (0.32 : ℚ) = 8 / 25 := by
sorry

end decimal_to_fraction_l2606_260694


namespace square_area_ratio_when_tripled_l2606_260633

theorem square_area_ratio_when_tripled (s : ℝ) (h : s > 0) :
  (3 * s)^2 / s^2 = 9 := by
  sorry

end square_area_ratio_when_tripled_l2606_260633


namespace hoseok_number_subtraction_l2606_260630

theorem hoseok_number_subtraction (n : ℕ) : n / 10 = 6 → n - 15 = 45 := by
  sorry

end hoseok_number_subtraction_l2606_260630


namespace strawberry_weight_sum_l2606_260681

theorem strawberry_weight_sum (marco_weight dad_weight : ℕ) 
  (h1 : marco_weight = 15) 
  (h2 : dad_weight = 22) : 
  marco_weight + dad_weight = 37 := by
sorry

end strawberry_weight_sum_l2606_260681


namespace power_zero_eq_one_three_power_zero_l2606_260664

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

theorem three_power_zero : (3 : ℝ)^0 = 1 := by sorry

end power_zero_eq_one_three_power_zero_l2606_260664


namespace oliver_stickers_l2606_260683

theorem oliver_stickers (initial_stickers : ℕ) (h1 : initial_stickers = 135) :
  let remaining_after_use := initial_stickers - (initial_stickers / 3)
  let given_away := remaining_after_use * 2 / 5
  let kept := remaining_after_use - given_away
  kept = 54 := by sorry

end oliver_stickers_l2606_260683


namespace total_memory_space_l2606_260656

def morning_songs : ℕ := 10
def afternoon_songs : ℕ := 15
def night_songs : ℕ := 3
def song_size : ℕ := 5

theorem total_memory_space : 
  (morning_songs + afternoon_songs + night_songs) * song_size = 140 :=
by sorry

end total_memory_space_l2606_260656


namespace sector_area_l2606_260638

/-- Given a circular sector with arc length 3π and central angle 3/4π, its area is 6π. -/
theorem sector_area (r : ℝ) (h1 : (3/4) * π * r = 3 * π) : (1/2) * (3/4 * π) * r^2 = 6 * π := by
  sorry

end sector_area_l2606_260638


namespace smallest_non_factor_product_of_100_l2606_260646

def is_factor (n m : ℕ) : Prop := m % n = 0

theorem smallest_non_factor_product_of_100 :
  ∃ (a b : ℕ),
    a ≠ b ∧
    a > 0 ∧
    b > 0 ∧
    is_factor a 100 ∧
    is_factor b 100 ∧
    ¬(is_factor (a * b) 100) ∧
    a * b = 8 ∧
    ∀ (c d : ℕ),
      c ≠ d →
      c > 0 →
      d > 0 →
      is_factor c 100 →
      is_factor d 100 →
      ¬(is_factor (c * d) 100) →
      c * d ≥ 8 :=
sorry

end smallest_non_factor_product_of_100_l2606_260646


namespace average_rst_l2606_260696

theorem average_rst (r s t : ℝ) (h : (5 / 4) * (r + s + t) = 20) :
  (r + s + t) / 3 = 16 / 3 := by
sorry

end average_rst_l2606_260696


namespace cubic_quadratic_equation_solution_l2606_260635

theorem cubic_quadratic_equation_solution :
  ∃! (y : ℝ), y ≠ 0 ∧ (8 * y)^3 = (16 * y)^2 ∧ y = 1/2 := by sorry

end cubic_quadratic_equation_solution_l2606_260635


namespace seven_balls_three_boxes_l2606_260618

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 36 := by
  sorry

end seven_balls_three_boxes_l2606_260618


namespace handshake_arrangement_count_l2606_260652

/-- Represents a handshake arrangement for a group of people --/
def HandshakeArrangement := Fin 12 → Finset (Fin 12)

/-- A valid handshake arrangement satisfies the problem conditions --/
def is_valid_arrangement (h : HandshakeArrangement) : Prop :=
  ∀ i : Fin 12, (h i).card = 3 ∧ ∀ j ∈ h i, i ∈ h j

/-- The number of distinct valid handshake arrangements --/
def num_arrangements : ℕ := sorry

theorem handshake_arrangement_count :
  num_arrangements = 13296960 ∧ num_arrangements % 1000 = 960 := by sorry

end handshake_arrangement_count_l2606_260652


namespace coyote_speed_calculation_l2606_260608

/-- The speed of the coyote in miles per hour -/
def coyote_speed : ℝ := 15

/-- The time elapsed since the coyote left its prints, in hours -/
def time_elapsed : ℝ := 1

/-- Darrel's speed on his motorbike in miles per hour -/
def darrel_speed : ℝ := 30

/-- The time it takes Darrel to catch up to the coyote, in hours -/
def catch_up_time : ℝ := 1

theorem coyote_speed_calculation :
  coyote_speed * time_elapsed + coyote_speed * catch_up_time = darrel_speed * catch_up_time := by
  sorry

#check coyote_speed_calculation

end coyote_speed_calculation_l2606_260608


namespace acetone_nine_moles_weight_l2606_260660

/-- The molecular weight of a single molecule of Acetone in g/mol -/
def acetone_molecular_weight : ℝ :=
  3 * 12.01 + 6 * 1.008 + 1 * 16.00

/-- The molecular weight of n moles of Acetone in grams -/
def acetone_weight (n : ℝ) : ℝ :=
  n * acetone_molecular_weight

/-- Theorem: The molecular weight of 9 moles of Acetone is 522.702 grams -/
theorem acetone_nine_moles_weight :
  acetone_weight 9 = 522.702 := by
  sorry

end acetone_nine_moles_weight_l2606_260660


namespace linear_system_fraction_sum_l2606_260640

theorem linear_system_fraction_sum (a b c x y z : ℝ) 
  (eq1 : 11 * x + b * y + c * z = 0)
  (eq2 : a * x + 24 * y + c * z = 0)
  (eq3 : a * x + b * y + 41 * z = 0)
  (ha : a ≠ 11)
  (hx : x ≠ 0) :
  a / (a - 11) + b / (b - 24) + c / (c - 41) = 1 := by
  sorry

end linear_system_fraction_sum_l2606_260640


namespace gabrielle_blue_jays_eq_three_l2606_260674

/-- The number of blue jays Gabrielle saw -/
def gabrielle_blue_jays : ℕ :=
  let gabrielle_robins : ℕ := 5
  let gabrielle_cardinals : ℕ := 4
  let chase_robins : ℕ := 2
  let chase_blue_jays : ℕ := 3
  let chase_cardinals : ℕ := 5
  let chase_total : ℕ := chase_robins + chase_blue_jays + chase_cardinals
  let gabrielle_total : ℕ := chase_total + chase_total / 5
  gabrielle_total - gabrielle_robins - gabrielle_cardinals

theorem gabrielle_blue_jays_eq_three : gabrielle_blue_jays = 3 := by
  sorry

end gabrielle_blue_jays_eq_three_l2606_260674


namespace more_students_than_pets_l2606_260687

theorem more_students_than_pets : 
  let num_classrooms : ℕ := 5
  let students_per_classroom : ℕ := 20
  let rabbits_per_classroom : ℕ := 2
  let goldfish_per_classroom : ℕ := 3
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_pets : ℕ := num_classrooms * (rabbits_per_classroom + goldfish_per_classroom)
  total_students - total_pets = 75 := by
sorry

end more_students_than_pets_l2606_260687


namespace percentage_change_xyz_l2606_260680

theorem percentage_change_xyz (x y z : ℝ) (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0) :
  let x' := 0.8 * x
  let y' := 0.8 * y
  let z' := 1.1 * z
  (x' * y' * z' - x * y * z) / (x * y * z) = -0.296 :=
by sorry

end percentage_change_xyz_l2606_260680


namespace flower_count_proof_l2606_260631

/-- The number of daisy seeds planted -/
def daisy_seeds : ℕ := 25

/-- The number of sunflower seeds planted -/
def sunflower_seeds : ℕ := 25

/-- The percentage of daisy seeds that germinate -/
def daisy_germination_rate : ℚ := 60 / 100

/-- The percentage of sunflower seeds that germinate -/
def sunflower_germination_rate : ℚ := 80 / 100

/-- The percentage of germinated plants that produce flowers -/
def flower_production_rate : ℚ := 80 / 100

/-- The total number of plants that produce flowers -/
def plants_with_flowers : ℕ := 28

theorem flower_count_proof :
  (daisy_seeds * daisy_germination_rate * flower_production_rate +
   sunflower_seeds * sunflower_germination_rate * flower_production_rate).floor = plants_with_flowers := by
  sorry

end flower_count_proof_l2606_260631


namespace f_is_even_and_increasing_on_negative_l2606_260643

def f (x : ℝ) := -x^2

theorem f_is_even_and_increasing_on_negative : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f x < f y) :=
by sorry

end f_is_even_and_increasing_on_negative_l2606_260643


namespace quadratic_solution_l2606_260682

theorem quadratic_solution (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9 : ℝ) - 45 = 0) → b = 4 := by
  sorry

end quadratic_solution_l2606_260682


namespace objective_function_minimum_range_l2606_260661

-- Define the objective function
def objective_function (k x y : ℝ) : ℝ := k * x + 2 * y

-- Define the constraints
def constraint1 (x y : ℝ) : Prop := 2 * x - y ≤ 1
def constraint2 (x y : ℝ) : Prop := x + y ≥ 2
def constraint3 (x y : ℝ) : Prop := y - x ≤ 2

-- Define the feasible region
def feasible_region (x y : ℝ) : Prop :=
  constraint1 x y ∧ constraint2 x y ∧ constraint3 x y

-- Define the minimum point
def is_minimum_point (k : ℝ) (x y : ℝ) : Prop :=
  feasible_region x y ∧
  ∀ x' y', feasible_region x' y' →
    objective_function k x y ≤ objective_function k x' y'

-- Theorem statement
theorem objective_function_minimum_range :
  ∀ k : ℝ, (is_minimum_point k 1 1 ∧
    ∀ x y, x ≠ 1 ∨ y ≠ 1 → ¬(is_minimum_point k x y)) →
  -4 < k ∧ k < 2 := by
  sorry

end objective_function_minimum_range_l2606_260661


namespace max_subway_riders_l2606_260676

theorem max_subway_riders (total : ℕ) (part_time full_time : ℕ → ℕ) : 
  total = 251 →
  (∀ p f, part_time p + full_time f = total) →
  (∀ p, part_time p ≤ total) →
  (∀ f, full_time f ≤ total) →
  (∀ p, (part_time p) % 11 = 0) →
  (∀ f, (full_time f) % 13 = 0) →
  (∃ max : ℕ, ∀ p f, 
    part_time p + full_time f = total → 
    (part_time p) / 11 + (full_time f) / 13 ≤ max ∧
    (∃ p' f', part_time p' + full_time f' = total ∧ 
              (part_time p') / 11 + (full_time f') / 13 = max)) →
  (∃ p f, part_time p + full_time f = total ∧ 
          (part_time p) / 11 + (full_time f) / 13 = 22) :=
sorry

end max_subway_riders_l2606_260676


namespace parallel_vectors_angle_l2606_260604

theorem parallel_vectors_angle (α : Real) : 
  α > 0 → 
  α < π / 2 → 
  let a : Fin 2 → Real := ![3/4, Real.sin α]
  let b : Fin 2 → Real := ![Real.cos α, 1/3]
  (∃ (k : Real), k ≠ 0 ∧ a = k • b) → 
  α = π / 12 ∨ α = 5 * π / 12 := by
sorry

end parallel_vectors_angle_l2606_260604


namespace cupcake_price_l2606_260632

theorem cupcake_price (cupcake_count : ℕ) (cookie_count : ℕ) (cookie_price : ℚ)
  (basketball_count : ℕ) (basketball_price : ℚ) (drink_count : ℕ) (drink_price : ℚ) :
  cupcake_count = 50 →
  cookie_count = 40 →
  cookie_price = 1/2 →
  basketball_count = 2 →
  basketball_price = 40 →
  drink_count = 20 →
  drink_price = 2 →
  ∃ (cupcake_price : ℚ),
    cupcake_count * cupcake_price + cookie_count * cookie_price =
    basketball_count * basketball_price + drink_count * drink_price ∧
    cupcake_price = 2 :=
by
  sorry


end cupcake_price_l2606_260632


namespace square_digit_sum_100_bound_l2606_260658

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem square_digit_sum_100_bound (n : ℕ) :
  sum_of_digits (n^2) = 100 → n ≤ 100 := by sorry

end square_digit_sum_100_bound_l2606_260658


namespace triangle_properties_l2606_260698

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

theorem triangle_properties (t : Triangle) 
  (m : Vector2D) (n : Vector2D) (angle_mn : ℝ) (area : ℝ) :
  m.x = Real.cos (t.C / 2) ∧ 
  m.y = Real.sin (t.C / 2) ∧
  n.x = Real.cos (t.C / 2) ∧ 
  n.y = -Real.sin (t.C / 2) ∧
  angle_mn = π / 3 ∧
  t.c = 7 / 2 ∧
  area = 3 * Real.sqrt 3 / 2 →
  t.C = π / 3 ∧ t.a + t.b = 11 / 2 := by
  sorry

end triangle_properties_l2606_260698


namespace smallest_number_of_cubes_for_given_box_l2606_260690

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes needed to fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem stating that the smallest number of identical cubes needed to fill the given box is 90 -/
theorem smallest_number_of_cubes_for_given_box :
  smallestNumberOfCubes ⟨27, 15, 6⟩ = 90 := by
  sorry

#eval smallestNumberOfCubes ⟨27, 15, 6⟩

end smallest_number_of_cubes_for_given_box_l2606_260690


namespace quadrilateral_area_inequality_l2606_260601

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  convex : Bool
  area : ℝ

-- State the theorem
theorem quadrilateral_area_inequality (q : ConvexQuadrilateral) (h : q.convex = true) :
  q.area ≤ (q.a^2 + q.b^2 + q.c^2 + q.d^2) / 4 := by
  sorry

end quadrilateral_area_inequality_l2606_260601


namespace initial_investment_calculation_l2606_260641

def initial_rate : ℚ := 5 / 100
def additional_rate : ℚ := 8 / 100
def total_rate : ℚ := 6 / 100
def additional_investment : ℚ := 4000

theorem initial_investment_calculation (x : ℚ) :
  initial_rate * x + additional_rate * additional_investment = 
  total_rate * (x + additional_investment) →
  x = 8000 := by
sorry

end initial_investment_calculation_l2606_260641


namespace max_distinct_pairs_l2606_260615

theorem max_distinct_pairs (n : ℕ) (h : n = 3010) :
  let S := Finset.range n
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 1201 ∧
    pairs.card = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 3005) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (m : ℕ), m > k →
      ¬∃ (pairs' : Finset (ℕ × ℕ)),
        pairs'.card = m ∧
        (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
        (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
        (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 + p.2 ≤ 3005) ∧
        (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 + p.2 ≠ q.1 + q.2)) :=
by
  sorry

end max_distinct_pairs_l2606_260615


namespace line_l_equation_l2606_260662

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Returns true if the point (x, y) lies on the given line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The reference line y = x + 1 -/
def referenceLine : Line :=
  { slope := 1, yIntercept := 1 }

/-- The line we're trying to prove -/
def lineL : Line :=
  { slope := 2, yIntercept := -3 }

theorem line_l_equation :
  (lineL.slope = 2 * referenceLine.slope) ∧
  (lineL.containsPoint 3 3) →
  ∀ x y : ℝ, lineL.containsPoint x y ↔ y = 2*x - 3 := by
  sorry

end line_l_equation_l2606_260662


namespace cube_property_l2606_260625

-- Define a cube type
structure Cube where
  side : ℝ
  volume_eq : volume = 8 * x
  area_eq : surfaceArea = x / 2

-- Define volume and surface area functions
def volume (c : Cube) : ℝ := c.side ^ 3
def surfaceArea (c : Cube) : ℝ := 6 * c.side ^ 2

-- State the theorem
theorem cube_property (x : ℝ) (c : Cube) : x = 110592 := by
  sorry

end cube_property_l2606_260625


namespace positive_integer_solutions_independent_of_m_compare_M_N_l2606_260650

def oplus (a b : ℝ) : ℝ := a * (a - b)

theorem positive_integer_solutions :
  ∀ a b : ℕ+, (oplus 3 a = b) ↔ ((a = 2 ∧ b = 3) ∨ (a = 1 ∧ b = 6)) :=
sorry

theorem independent_of_m (a b m : ℝ) :
  (oplus 2 a = 5*b - 2*m ∧ oplus 3 b = 5*a + m) → 12*a + 11*b = 22 :=
sorry

theorem compare_M_N (a b : ℝ) (h : a > 1) :
  oplus (a*b) b ≥ oplus b (a*b) :=
sorry

end positive_integer_solutions_independent_of_m_compare_M_N_l2606_260650
