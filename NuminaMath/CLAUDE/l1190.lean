import Mathlib

namespace quadratic_even_deductive_reasoning_l1190_119046

-- Definition of an even function
def IsEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Definition of a quadratic function
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c

-- Definition of deductive reasoning process
structure DeductiveReasoning :=
  (majorPremise : Prop)
  (minorPremise : Prop)
  (conclusion : Prop)

-- Theorem stating that the reasoning process for proving x^2 is even is deductive
theorem quadratic_even_deductive_reasoning :
  ∃ (reasoning : DeductiveReasoning),
    reasoning.majorPremise = (∀ f : ℝ → ℝ, IsEvenFunction f → ∀ x : ℝ, f (-x) = f x) ∧
    reasoning.minorPremise = (∃ f : ℝ → ℝ, QuadraticFunction f ∧ ∀ x : ℝ, f (-x) = f x) ∧
    reasoning.conclusion = (∃ f : ℝ → ℝ, QuadraticFunction f ∧ IsEvenFunction f) :=
  sorry


end quadratic_even_deductive_reasoning_l1190_119046


namespace certain_value_proof_l1190_119016

theorem certain_value_proof (n : ℤ) (v : ℤ) : 
  (∀ m : ℤ, 101 * m^2 ≤ v → m ≤ 8) → 
  (101 * 8^2 ≤ v) →
  v = 6464 := by
sorry

end certain_value_proof_l1190_119016


namespace dislike_both_count_l1190_119021

/-- The number of people who don't like both radio and music in a poll -/
def people_dislike_both (total : ℕ) (radio_dislike_percent : ℚ) (music_dislike_percent : ℚ) : ℕ :=
  ⌊(radio_dislike_percent * music_dislike_percent * total : ℚ)⌋₊

/-- Theorem about the number of people who don't like both radio and music -/
theorem dislike_both_count :
  people_dislike_both 1500 (35/100) (15/100) = 79 := by
  sorry

#eval people_dislike_both 1500 (35/100) (15/100)

end dislike_both_count_l1190_119021


namespace mitya_travel_schedule_unique_l1190_119095

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the months of the year -/
inductive Month
  | February
  | March

/-- Represents a date within a month -/
structure Date where
  month : Month
  day : Nat

/-- Represents Mitya's travel schedule -/
structure TravelSchedule where
  smolensk : Date
  vologda : Date
  pskov : Date
  vladimir : Date

/-- Returns the day of the week for a given date -/
def dayOfWeek (date : Date) : DayOfWeek :=
  sorry

/-- Returns true if the given year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  sorry

/-- Returns the number of days in a given month -/
def daysInMonth (month : Month) (isLeap : Bool) : Nat :=
  sorry

/-- Theorem: Given the conditions of Mitya's travel and the calendar structure,
    there exists a unique travel schedule that satisfies all constraints -/
theorem mitya_travel_schedule_unique :
  ∃! (schedule : TravelSchedule),
    (dayOfWeek schedule.smolensk = DayOfWeek.Tuesday) ∧
    (dayOfWeek schedule.vologda = DayOfWeek.Tuesday) ∧
    (dayOfWeek schedule.pskov = DayOfWeek.Tuesday) ∧
    (dayOfWeek schedule.vladimir = DayOfWeek.Tuesday) ∧
    (schedule.smolensk.month = Month.February) ∧
    (schedule.vologda.month = Month.February) ∧
    (schedule.pskov.month = Month.March) ∧
    (schedule.vladimir.month = Month.March) ∧
    (schedule.smolensk.day = 1) ∧
    (schedule.vologda.day > schedule.smolensk.day) ∧
    (schedule.pskov.day = 1) ∧
    (schedule.vladimir.day > schedule.pskov.day) ∧
    (¬isLeapYear 0) ∧
    (daysInMonth Month.February false = 28) ∧
    (daysInMonth Month.March false = 31) :=
  sorry

end mitya_travel_schedule_unique_l1190_119095


namespace joe_trip_expenses_l1190_119030

/-- Calculates the remaining money after expenses -/
def remaining_money (initial_savings flight_cost hotel_cost food_cost : ℕ) : ℕ :=
  initial_savings - (flight_cost + hotel_cost + food_cost)

/-- Proves that Joe has $1,000 left after his trip expenses -/
theorem joe_trip_expenses :
  remaining_money 6000 1200 800 3000 = 1000 := by
  sorry

end joe_trip_expenses_l1190_119030


namespace dining_bill_share_l1190_119001

/-- Given a total bill, number of people, and tip percentage, calculate the amount each person should pay. -/
def calculate_share (total_bill : ℚ) (num_people : ℕ) (tip_percentage : ℚ) : ℚ :=
  let total_with_tip := total_bill * (1 + tip_percentage)
  total_with_tip / num_people

/-- Prove that for a bill of $139.00 split among 5 people with a 10% tip, each person should pay $30.58. -/
theorem dining_bill_share :
  calculate_share 139 5 (1/10) = 3058/100 := by
  sorry

end dining_bill_share_l1190_119001


namespace f_at_2023_half_l1190_119002

/-- A function that is odd and symmetric about x = 1 -/
def f (x : ℝ) : ℝ :=
  sorry

/-- The function f is odd -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- The function f is symmetric about x = 1 -/
axiom f_sym (x : ℝ) : f x = f (2 - x)

/-- The function f is defined as 2^x + b for x ∈ [0,1] -/
axiom f_def (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = 2^x + Real.pi

/-- The main theorem -/
theorem f_at_2023_half : f (2023/2) = 1 - Real.sqrt 2 :=
  sorry

end f_at_2023_half_l1190_119002


namespace product_digits_l1190_119012

def a : ℕ := 8476235982145327
def b : ℕ := 2983674531

theorem product_digits : (String.length (toString (a * b))) = 28 := by
  sorry

end product_digits_l1190_119012


namespace mari_buttons_l1190_119081

theorem mari_buttons (kendra_buttons : ℕ) (mari_buttons : ℕ) : 
  kendra_buttons = 15 →
  mari_buttons = 5 * kendra_buttons + 4 →
  mari_buttons = 79 := by
  sorry

end mari_buttons_l1190_119081


namespace bakers_friend_cakes_prove_bakers_friend_cakes_l1190_119051

/-- Given that Baker initially made 169 cakes and has 32 cakes left,
    prove that the number of cakes bought by Baker's friend is 137. -/
theorem bakers_friend_cakes : ℕ → ℕ → ℕ → Prop :=
  fun initial_cakes remaining_cakes cakes_bought =>
    initial_cakes = 169 →
    remaining_cakes = 32 →
    cakes_bought = initial_cakes - remaining_cakes →
    cakes_bought = 137

/-- Proof of the theorem -/
theorem prove_bakers_friend_cakes :
  bakers_friend_cakes 169 32 137 := by
  sorry

end bakers_friend_cakes_prove_bakers_friend_cakes_l1190_119051


namespace last_segment_speed_prove_last_segment_speed_l1190_119064

theorem last_segment_speed (total_distance : ℝ) (total_time : ℝ) 
  (speed1 : ℝ) (speed2 : ℝ) (speed3 : ℝ) : ℝ :=
  let total_segments : ℝ := 4
  let segment_time : ℝ := total_time / total_segments
  let overall_avg_speed : ℝ := total_distance / total_time
  let last_segment_speed : ℝ := 
    total_segments * overall_avg_speed - (speed1 + speed2 + speed3)
  last_segment_speed

theorem prove_last_segment_speed : 
  last_segment_speed 160 2 55 75 60 = 130 := by
  sorry

end last_segment_speed_prove_last_segment_speed_l1190_119064


namespace special_subset_count_l1190_119041

def subset_count (n : ℕ) : ℕ :=
  (Finset.range 11).sum (fun k => Nat.choose (n - k + 1) k)

theorem special_subset_count : subset_count 20 = 3164 := by
  sorry

end special_subset_count_l1190_119041


namespace compute_expression_l1190_119075

theorem compute_expression : 18 * (250 / 3 + 36 / 9 + 16 / 32 + 2) = 1617 := by
  sorry

end compute_expression_l1190_119075


namespace regression_properties_l1190_119077

/-- Regression line equation -/
def regression_line (x : ℝ) : ℝ := 6 * x + 8

/-- Data points -/
def data_points : List (ℝ × ℝ) := [(2, 19), (3, 25), (4, 0), (5, 38), (6, 44)]

/-- The value of the unclear data point -/
def unclear_data : ℝ := 34

/-- Theorem stating the properties of the regression line and data points -/
theorem regression_properties :
  let third_point := (4, unclear_data)
  let residual := (third_point.2 - regression_line third_point.1)
  (unclear_data = 34) ∧
  (residual = 2) ∧
  (regression_line 7 = 50) := by sorry

end regression_properties_l1190_119077


namespace smallest_with_eight_factors_l1190_119083

/-- The number of distinct positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Prove that 16 is the smallest positive integer with exactly eight distinct positive factors -/
theorem smallest_with_eight_factors : 
  (∀ m : ℕ+, m < 16 → num_factors m ≠ 8) ∧ num_factors 16 = 8 := by sorry

end smallest_with_eight_factors_l1190_119083


namespace pentagonal_sum_theorem_l1190_119084

def pentagonal_layer_sum (n : ℕ) : ℕ := 4 * (3^(n-1) - 1)

theorem pentagonal_sum_theorem (n : ℕ) :
  n ≥ 1 →
  (pentagonal_layer_sum 1 = 0) →
  (∀ k : ℕ, k ≥ 1 → pentagonal_layer_sum (k+1) = 3 * pentagonal_layer_sum k + 4) →
  pentagonal_layer_sum n = 4 * (3^(n-1) - 1) :=
by sorry

end pentagonal_sum_theorem_l1190_119084


namespace modulus_of_2_plus_i_l1190_119003

/-- The modulus of the complex number 2 + i is √5 -/
theorem modulus_of_2_plus_i : Complex.abs (2 + Complex.I) = Real.sqrt 5 := by
  sorry

end modulus_of_2_plus_i_l1190_119003


namespace expression_proof_l1190_119059

theorem expression_proof (x : ℝ) (E : ℝ) : 
  ((x + 3)^2 / E = 3) ∧ 
  (∃ x₁ x₂ : ℝ, x₁ - x₂ = 12 ∧ 
    ((x₁ + 3)^2 / E = 3) ∧ 
    ((x₂ + 3)^2 / E = 3)) → 
  (E = (x + 3)^2 / 3 ∧ E = 12) := by
sorry

end expression_proof_l1190_119059


namespace second_workshop_production_l1190_119068

/-- Given three workshops producing boots with samples forming an arithmetic sequence,
    prove that the second workshop's production is 1200 pairs. -/
theorem second_workshop_production
  (total_production : ℕ)
  (a b c : ℕ)
  (h1 : total_production = 3600)
  (h2 : a + c = 2 * b)  -- arithmetic sequence property
  (h3 : a + b + c > 0)  -- ensure division is valid
  : (b : ℚ) / (a + b + c : ℚ) * total_production = 1200 :=
by sorry

end second_workshop_production_l1190_119068


namespace relay_team_members_l1190_119006

/-- Represents a cross-country relay team -/
structure RelayTeam where
  totalDistance : ℝ
  standardMemberDistance : ℝ
  ralphDistance : ℝ
  otherMembersCount : ℕ

/-- Conditions for the relay team -/
def validRelayTeam (team : RelayTeam) : Prop :=
  team.totalDistance = 18 ∧
  team.standardMemberDistance = 3 ∧
  team.ralphDistance = 2 * team.standardMemberDistance ∧
  team.totalDistance = team.ralphDistance + team.otherMembersCount * team.standardMemberDistance

/-- Theorem: The number of other team members is 4 -/
theorem relay_team_members (team : RelayTeam) (h : validRelayTeam team) : 
  team.otherMembersCount = 4 := by
  sorry

end relay_team_members_l1190_119006


namespace school_comparison_l1190_119067

theorem school_comparison (students_A : ℝ) (qualified_A : ℝ) (students_B : ℝ) (qualified_B : ℝ)
  (h1 : qualified_A = 0.7 * students_A)
  (h2 : qualified_B = 1.5 * qualified_A)
  (h3 : qualified_B = 0.875 * students_B) :
  (students_B - students_A) / students_A = 0.2 := by
  sorry

end school_comparison_l1190_119067


namespace inequality_proof_l1190_119031

-- Define the set M
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- Theorem statement
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2 * |a - b|) := by
  sorry

end inequality_proof_l1190_119031


namespace right_triangle_perimeter_l1190_119092

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) :
  area = (1 / 2) * leg1 * leg2 →
  leg1 = 30 →
  area = 150 →
  leg2 * leg2 + leg1 * leg1 = hypotenuse * hypotenuse →
  leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10 := by
  sorry

end right_triangle_perimeter_l1190_119092


namespace equation_solutions_l1190_119050

theorem equation_solutions (x : ℝ) (y : ℝ) : 
  x^2 + 6 * (x / (x - 3))^2 = 81 →
  y = ((x - 3)^2 * (x + 4)) / (3*x - 4) →
  (y = -9 ∨ y = 225/176) :=
by sorry

end equation_solutions_l1190_119050


namespace expected_heads_value_l1190_119044

/-- The number of coins -/
def n : ℕ := 64

/-- The probability of getting heads on a single fair coin toss -/
def p : ℚ := 1/2

/-- The probability of getting heads after up to three tosses -/
def prob_heads : ℚ := p + (1 - p) * p + (1 - p) * (1 - p) * p

/-- The expected number of coins showing heads after the process -/
def expected_heads : ℚ := n * prob_heads

theorem expected_heads_value : expected_heads = 56 := by sorry

end expected_heads_value_l1190_119044


namespace min_reciprocal_sum_l1190_119043

/-- The problem statement -/
theorem min_reciprocal_sum (x y a b : ℝ) : 
  8 * x - y - 4 ≤ 0 →
  x + y + 1 ≥ 0 →
  y - 4 * x ≤ 0 →
  a > 0 →
  b > 0 →
  (∀ x' y', 8 * x' - y' - 4 ≤ 0 → x' + y' + 1 ≥ 0 → y' - 4 * x' ≤ 0 → a * x' + b * y' ≤ 2) →
  a * x + b * y = 2 →
  (∀ a' b', a' > 0 → b' > 0 → 
    (∀ x' y', 8 * x' - y' - 4 ≤ 0 → x' + y' + 1 ≥ 0 → y' - 4 * x' ≤ 0 → a' * x' + b' * y' ≤ 2) →
    1 / a + 1 / b ≤ 1 / a' + 1 / b') →
  1 / a + 1 / b = 9 / 2 :=
by sorry

end min_reciprocal_sum_l1190_119043


namespace blood_cell_count_l1190_119017

theorem blood_cell_count (total : ℕ) (second : ℕ) (first : ℕ) : 
  total = 7341 → second = 3120 → first = total - second → first = 4221 := by
  sorry

end blood_cell_count_l1190_119017


namespace inequality_and_equality_condition_l1190_119099

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧ 
  ((1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b)) ↔ a = b) := by
  sorry

end inequality_and_equality_condition_l1190_119099


namespace sum_at_thirteenth_position_l1190_119082

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℕ
  is_permutation : Function.Bijective vertices

/-- The sum of numbers in a specific position across all orientations of a regular polygon -/
def sum_at_position (p : RegularPolygon 100) (pos : ℕ) : ℕ :=
  sorry

/-- Theorem: The sum of numbers in the 13th position from the left across all orientations of a regular 100-gon is 10100 -/
theorem sum_at_thirteenth_position (p : RegularPolygon 100) :
  sum_at_position p 13 = 10100 := by
  sorry

end sum_at_thirteenth_position_l1190_119082


namespace puppies_sold_l1190_119047

theorem puppies_sold (initial_puppies cages puppies_per_cage : ℕ) :
  initial_puppies = 78 →
  puppies_per_cage = 8 →
  cages = 6 →
  initial_puppies - (cages * puppies_per_cage) = 30 :=
by sorry

end puppies_sold_l1190_119047


namespace lawn_width_proof_l1190_119028

/-- Proves that the width of a rectangular lawn is 60 meters given specific conditions --/
theorem lawn_width_proof (W : ℝ) : 
  W > 0 →  -- Width is positive
  (10 * W + 10 * 70 - 10 * 10) * 3 = 3600 →  -- Cost equation
  W = 60 := by
  sorry

end lawn_width_proof_l1190_119028


namespace margo_round_trip_distance_l1190_119069

/-- Calculates the total distance covered in a round trip given the time for each leg and the average speed -/
def total_distance (outward_time return_time avg_speed : ℚ) : ℚ :=
  avg_speed * (outward_time + return_time) / 60

/-- Proves that the total distance covered in the given scenario is 4 miles -/
theorem margo_round_trip_distance :
  total_distance (15 : ℚ) (25 : ℚ) (6 : ℚ) = 4 := by
  sorry

end margo_round_trip_distance_l1190_119069


namespace quadratic_roots_l1190_119042

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  point_neg_two : a * (-2)^2 + b * (-2) + c = 12
  point_zero : c = -8
  point_one : a + b + c = -12
  point_three : a * 3^2 + b * 3 + c = -8

/-- The theorem statement -/
theorem quadratic_roots (f : QuadraticFunction) :
  let roots := {x : ℝ | f.a * x^2 + f.b * x + f.c + 8 = 0}
  roots = {0, 3} := by sorry

end quadratic_roots_l1190_119042


namespace election_votes_l1190_119020

/-- In an election with 3 candidates, where two candidates received 5000 and 15000 votes
    respectively, and the winning candidate got 66.66666666666666% of the total votes,
    the winning candidate (third candidate) received 40000 votes. -/
theorem election_votes :
  let total_votes : ℕ := 60000
  let first_candidate_votes : ℕ := 5000
  let second_candidate_votes : ℕ := 15000
  let winning_percentage : ℚ := 200 / 3
  ∀ third_candidate_votes : ℕ,
    first_candidate_votes + second_candidate_votes + third_candidate_votes = total_votes →
    (third_candidate_votes : ℚ) / total_votes * 100 = winning_percentage →
    third_candidate_votes = 40000 :=
by sorry

end election_votes_l1190_119020


namespace fractional_equation_solution_l1190_119008

theorem fractional_equation_solution :
  ∃ x : ℝ, (1 / x = 2 / (x + 1)) ∧ (x = 1) :=
by sorry

end fractional_equation_solution_l1190_119008


namespace tangent_parallel_points_l1190_119080

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y →
    (∃ k : ℝ, (3 * x^2 + 1) * k = 1 ∧ 4 * k = 1) ↔ 
    ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

#check tangent_parallel_points

end tangent_parallel_points_l1190_119080


namespace aspirations_necessary_for_reaching_l1190_119034

-- Define the universe of discourse
variable (Person : Type)

-- Define predicates
variable (has_aspirations : Person → Prop)
variable (can_reach_extraordinary : Person → Prop)
variable (is_remote_dangerous : Person → Prop)
variable (few_venture : Person → Prop)

-- State the theorem
theorem aspirations_necessary_for_reaching :
  (∀ p : Person, is_remote_dangerous p → few_venture p) →
  (∀ p : Person, can_reach_extraordinary p → has_aspirations p) →
  (∀ p : Person, can_reach_extraordinary p → has_aspirations p) :=
by
  sorry


end aspirations_necessary_for_reaching_l1190_119034


namespace max_xy_value_l1190_119033

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 200) : x * y ≤ 348 := by
  sorry

end max_xy_value_l1190_119033


namespace geometric_sequence_sum_l1190_119086

/-- Given a geometric sequence {a_n} where the sum of the first n terms S_n
    is defined as S_n = x · 3^n + 1, this theorem states that x = -1. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (x : ℝ) :
  (∀ n, S n = x * 3^n + 1) →
  (∀ n, a (n+1) / a n = a (n+2) / a (n+1)) →
  x = -1 :=
by sorry

end geometric_sequence_sum_l1190_119086


namespace perpendicular_vectors_magnitude_l1190_119036

def a : ℝ × ℝ := (2, 3)
def b (t : ℝ) : ℝ × ℝ := (t, -1)

theorem perpendicular_vectors_magnitude (t : ℝ) :
  (a.1 * (b t).1 + a.2 * (b t).2 = 0) →
  Real.sqrt ((a.1 - 2 * (b t).1)^2 + (a.2 - 2 * (b t).2)^2) = Real.sqrt 26 :=
by sorry

end perpendicular_vectors_magnitude_l1190_119036


namespace pyramid_height_theorem_l1190_119058

/-- Properties of the Great Pyramid of Giza --/
structure Pyramid where
  h : ℝ  -- The certain height
  height : ℝ := h + 20  -- The actual height of the pyramid
  width : ℝ := height + 234  -- The width of the pyramid

/-- Theorem about the height of the Great Pyramid of Giza --/
theorem pyramid_height_theorem (p : Pyramid) 
    (sum_condition : p.height + p.width = 1274) : 
    p.h = 1000 / 3 := by
  sorry

end pyramid_height_theorem_l1190_119058


namespace triple_sum_squares_and_fourth_powers_l1190_119076

theorem triple_sum_squares_and_fourth_powers (t : ℤ) : 
  (4*t)^2 + (3 - 2*t - t^2)^2 + (3 + 2*t - t^2)^2 = 2*(3 + t^2)^2 ∧
  (4*t)^4 + (3 - 2*t - t^2)^4 + (3 + 2*t - t^2)^4 = 2*(3 + t^2)^4 := by
  sorry

end triple_sum_squares_and_fourth_powers_l1190_119076


namespace simplify_sqrt_expression_l1190_119018

theorem simplify_sqrt_expression :
  (Real.sqrt 450 / Real.sqrt 200) - (Real.sqrt 175 / Real.sqrt 75) = (9 - 2 * Real.sqrt 21) / 6 := by
  sorry

end simplify_sqrt_expression_l1190_119018


namespace instructor_schedule_lcm_l1190_119090

theorem instructor_schedule_lcm : Nat.lcm 9 (Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 10 12))) = 360 := by
  sorry

end instructor_schedule_lcm_l1190_119090


namespace one_intersection_implies_a_range_l1190_119089

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a^2*x + 1

-- State the theorem
theorem one_intersection_implies_a_range (a : ℝ) :
  (∃! x : ℝ, f a x = 3) → -1 < a ∧ a < 1 :=
by sorry

end one_intersection_implies_a_range_l1190_119089


namespace prime_sequence_ones_digit_l1190_119013

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def ones_digit (n : ℕ) : ℕ := n % 10

theorem prime_sequence_ones_digit (p q r s : ℕ) :
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧
  p > 10 ∧
  q = p + 10 ∧ r = q + 10 ∧ s = r + 10 →
  ones_digit p = 1 :=
sorry

end prime_sequence_ones_digit_l1190_119013


namespace factorization_difference_of_squares_l1190_119096

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorization_difference_of_squares_l1190_119096


namespace wizard_elixir_combinations_l1190_119026

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of mystical crystals available. -/
def num_crystals : ℕ := 6

/-- The number of herbs that react negatively with one crystal. -/
def num_incompatible_herbs : ℕ := 3

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - num_incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 21 :=
sorry

end wizard_elixir_combinations_l1190_119026


namespace triangle_inequality_l1190_119091

theorem triangle_inequality (a b c α β γ : ℝ) (n : ℕ) : 
  a > 0 → b > 0 → c > 0 → 
  α > 0 → β > 0 → γ > 0 → 
  α + β + γ = π → 
  (π/3)^n ≤ (a*α^n + b*β^n + c*γ^n) / (a + b + c) ∧ 
  (a*α^n + b*β^n + c*γ^n) / (a + b + c) < π^n/2 := by
  sorry

end triangle_inequality_l1190_119091


namespace parabola_equation_is_correct_coefficient_x2_positive_gcd_of_coefficients_is_one_l1190_119062

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in general form -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The focus of the parabola -/
def focus : Point := { x := 2, y := -1 }

/-- The directrix of the parabola -/
def directrix : Line := { a := 1, b := 2, c := -4 }

/-- The equation of the parabola -/
def parabola_equation : Parabola := { a := 4, b := -4, c := 1, d := -12, e := -6, f := 9 }

/-- Theorem stating that the given equation represents the parabola with the given focus and directrix -/
theorem parabola_equation_is_correct (p : Point) : 
  (parabola_equation.a * p.x^2 + parabola_equation.b * p.x * p.y + parabola_equation.c * p.y^2 + 
   parabola_equation.d * p.x + parabola_equation.e * p.y + parabola_equation.f = 0) ↔ 
  ((p.x - focus.x)^2 + (p.y - focus.y)^2 = 
   ((directrix.a * p.x + directrix.b * p.y + directrix.c)^2) / (directrix.a^2 + directrix.b^2)) :=
sorry

/-- Theorem stating that the coefficient of x^2 is positive -/
theorem coefficient_x2_positive : parabola_equation.a > 0 :=
sorry

/-- Theorem stating that the GCD of absolute values of coefficients is 1 -/
theorem gcd_of_coefficients_is_one : 
  Nat.gcd (Int.natAbs parabola_equation.a) 
    (Nat.gcd (Int.natAbs parabola_equation.b) 
      (Nat.gcd (Int.natAbs parabola_equation.c) 
        (Nat.gcd (Int.natAbs parabola_equation.d) 
          (Nat.gcd (Int.natAbs parabola_equation.e) 
            (Int.natAbs parabola_equation.f))))) = 1 :=
sorry

end parabola_equation_is_correct_coefficient_x2_positive_gcd_of_coefficients_is_one_l1190_119062


namespace angle_of_inclination_slope_one_l1190_119078

/-- The angle of inclination of a line with slope 1 is π/4 --/
theorem angle_of_inclination_slope_one :
  let line : ℝ → ℝ := λ x ↦ x + 1
  let slope : ℝ := 1
  let angle_of_inclination : ℝ := Real.arctan slope
  angle_of_inclination = π / 4 := by
  sorry

end angle_of_inclination_slope_one_l1190_119078


namespace sodium_thiosulfate_properties_l1190_119097

/-- Represents the sodium thiosulfate anion -/
structure SodiumThiosulfateAnion where
  has_s_s_bond : Bool
  has_s_o_s_bond : Bool
  has_o_o_bond : Bool

/-- Represents the formation method of sodium thiosulfate -/
inductive FormationMethod
  | ThermalDecomposition
  | SulfiteWithSulfur
  | AnodicOxidation

/-- Properties of sodium thiosulfate -/
structure SodiumThiosulfate where
  anion : SodiumThiosulfateAnion
  formation : FormationMethod

/-- Theorem stating the correct structure and formation of sodium thiosulfate -/
theorem sodium_thiosulfate_properties :
  ∃ (st : SodiumThiosulfate),
    st.anion.has_s_s_bond = true ∧
    st.formation = FormationMethod.SulfiteWithSulfur :=
  sorry

end sodium_thiosulfate_properties_l1190_119097


namespace seven_possible_D_values_l1190_119039

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the addition of two 5-digit numbers ABBCB + BCAIA = DBDDD -/
def ValidAddition (A B C D : Digit) : Prop :=
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D) ∧
  (10000 * A.val + 1000 * B.val + 100 * B.val + 10 * C.val + B.val) +
  (10000 * B.val + 1000 * C.val + 100 * A.val + 10 * 1 + A.val) =
  (10000 * D.val + 1000 * B.val + 100 * D.val + 10 * D.val + D.val)

/-- The theorem stating that there are exactly 7 possible values for D -/
theorem seven_possible_D_values :
  ∃ (S : Finset Digit), S.card = 7 ∧
  (∀ D, D ∈ S ↔ ∃ A B C, ValidAddition A B C D) :=
sorry

end seven_possible_D_values_l1190_119039


namespace baker_pastries_sold_l1190_119005

/-- Given information about a baker's production and sales of cakes and pastries, 
    prove that the number of pastries sold equals the number of cakes made. -/
theorem baker_pastries_sold (cakes_made pastries_made : ℕ) 
    (h1 : cakes_made = 19)
    (h2 : pastries_made = 131)
    (h3 : pastries_made - cakes_made = 112) :
    pastries_made - (pastries_made - cakes_made) = cakes_made := by
  sorry

end baker_pastries_sold_l1190_119005


namespace probability_theorem_l1190_119029

def num_forks : ℕ := 8
def num_spoons : ℕ := 5
def num_knives : ℕ := 7
def total_pieces : ℕ := num_forks + num_spoons + num_knives
def num_selected : ℕ := 4

def probability_two_forks_one_spoon_one_knife : ℚ :=
  (Nat.choose num_forks 2 * Nat.choose num_spoons 1 * Nat.choose num_knives 1 : ℚ) /
  Nat.choose total_pieces num_selected

theorem probability_theorem :
  probability_two_forks_one_spoon_one_knife = 196 / 969 := by
  sorry

end probability_theorem_l1190_119029


namespace blue_balls_count_l1190_119011

def total_balls : ℕ := 12

def prob_two_blue : ℚ := 1/22

theorem blue_balls_count :
  ∃ b : ℕ, 
    b ≤ total_balls ∧ 
    (b : ℚ) / total_balls * ((b - 1) : ℚ) / (total_balls - 1) = prob_two_blue ∧
    b = 3 :=
sorry

end blue_balls_count_l1190_119011


namespace smallest_cube_root_with_small_remainder_l1190_119007

theorem smallest_cube_root_with_small_remainder (m n : ℕ) (r : ℝ) : 
  (∀ k < m, ¬∃ (j : ℕ) (s : ℝ), k^(1/3 : ℝ) = j + s ∧ 0 < s ∧ s < 1/2000) →
  (m : ℝ)^(1/3 : ℝ) = n + r →
  0 < r →
  r < 1/2000 →
  n = 26 := by
sorry

end smallest_cube_root_with_small_remainder_l1190_119007


namespace triangle_area_calculation_l1190_119072

theorem triangle_area_calculation (base : ℝ) (height_factor : ℝ) :
  base = 3.6 →
  height_factor = 2.5 →
  (base * (height_factor * base)) / 2 = 16.2 := by
  sorry

end triangle_area_calculation_l1190_119072


namespace problem_statement_l1190_119060

theorem problem_statement (a b : ℝ) : 
  |a + 2| + (b - 1)^2 = 0 → (a + b)^2005 = -1 := by
  sorry

end problem_statement_l1190_119060


namespace flower_bed_total_l1190_119045

theorem flower_bed_total (tulips carnations : ℕ) : 
  tulips = 3 → carnations = 4 → tulips + carnations = 7 := by
  sorry

end flower_bed_total_l1190_119045


namespace gcd_triple_existence_l1190_119022

theorem gcd_triple_existence (S : Set ℕ+) (hS_infinite : Set.Infinite S)
  (a b c d : ℕ+) (hab : a ∈ S) (hbc : b ∈ S) (hcd : c ∈ S) (hda : d ∈ S)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (hgcd : Nat.gcd a.val b.val ≠ Nat.gcd c.val d.val) :
  ∃ x y z : ℕ+, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    Nat.gcd x.val y.val = Nat.gcd y.val z.val ∧
    Nat.gcd y.val z.val ≠ Nat.gcd z.val x.val :=
by sorry

end gcd_triple_existence_l1190_119022


namespace triangle_properties_l1190_119073

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a ≠ t.b ∧
  2 * Real.sin (t.A - t.B) = t.a * Real.sin t.A - t.b * Real.sin t.B ∧
  (1/2) * t.a * t.b * Real.sin t.C = 1 ∧
  Real.tan t.C = 2

-- State the theorem
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.c = 2 ∧ t.a + t.b = 1 + Real.sqrt 5 := by
  sorry

end triangle_properties_l1190_119073


namespace alexandre_winning_strategy_l1190_119027

/-- A game on an n-gon where players alternately mark vertices with 0 or 1 -/
def Game (n : ℕ) := Unit

/-- A strategy for the second player (Alexandre) -/
def Strategy (n : ℕ) := Game n → ℕ → ℕ

/-- Predicate to check if three consecutive vertices have a sum divisible by 3 -/
def HasWinningTriple (g : Game n) : Prop := sorry

/-- Predicate to check if a strategy is winning for the second player -/
def IsWinningStrategy (s : Strategy n) : Prop := sorry

theorem alexandre_winning_strategy 
  (n : ℕ) 
  (h1 : n > 3) 
  (h2 : Even n) : 
  ∃ (s : Strategy n), IsWinningStrategy s := by sorry

end alexandre_winning_strategy_l1190_119027


namespace probability_both_preferred_is_one_fourth_l1190_119015

/-- Represents the colors of the balls -/
inductive Color
| Red
| Yellow
| Blue
| Green
| Purple

/-- Represents a person -/
structure Person where
  name : String
  preferredColors : List Color

/-- Represents the bag of balls -/
def bag : List Color := [Color.Red, Color.Yellow, Color.Blue, Color.Green, Color.Purple]

/-- Person A's preferred colors -/
def personA : Person := { name := "A", preferredColors := [Color.Red, Color.Yellow] }

/-- Person B's preferred colors -/
def personB : Person := { name := "B", preferredColors := [Color.Yellow, Color.Green, Color.Purple] }

/-- Calculates the probability of both persons drawing their preferred colors -/
def probabilityBothPreferred (bag : List Color) (personA personB : Person) : ℚ :=
  sorry

/-- Theorem stating the probability of both persons drawing their preferred colors is 1/4 -/
theorem probability_both_preferred_is_one_fourth :
  probabilityBothPreferred bag personA personB = 1/4 :=
sorry

end probability_both_preferred_is_one_fourth_l1190_119015


namespace problem_solution_l1190_119071

theorem problem_solution (x k : ℕ) (h1 : (2^x) - (2^(x-2)) = k * (2^10)) (h2 : x = 12) : k = 3 := by
  sorry

end problem_solution_l1190_119071


namespace imaginary_part_of_complex_fraction_l1190_119010

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (3 + I) / (2 - I) → z.im = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l1190_119010


namespace rectangle_max_area_max_area_achievable_l1190_119057

/-- Given a rectangle with perimeter 40 inches, its maximum area is 100 square inches. -/
theorem rectangle_max_area :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * x + 2 * y = 40 →
  ∀ a : ℝ,
  (0 < a ∧ ∃ w h : ℝ, w > 0 ∧ h > 0 ∧ 2 * w + 2 * h = 40 ∧ a = w * h) →
  x * y ≥ a :=
by sorry

/-- The maximum area of 100 square inches is achievable. -/
theorem max_area_achievable :
  ∃ x y : ℝ,
  x > 0 ∧ y > 0 ∧
  2 * x + 2 * y = 40 ∧
  x * y = 100 :=
by sorry

end rectangle_max_area_max_area_achievable_l1190_119057


namespace systematic_sampling_l1190_119074

theorem systematic_sampling 
  (population_size : ℕ) 
  (num_groups : ℕ) 
  (sample_size : ℕ) 
  (first_draw : ℕ) :
  population_size = 60 →
  num_groups = 6 →
  sample_size = 6 →
  first_draw = 3 →
  let interval := population_size / num_groups
  let fifth_group_draw := first_draw + interval * 4
  fifth_group_draw = 43 := by
sorry


end systematic_sampling_l1190_119074


namespace kim_cookie_boxes_l1190_119000

theorem kim_cookie_boxes (jennifer_boxes : ℕ) (difference : ℕ) (h1 : jennifer_boxes = 71) (h2 : difference = 17) :
  jennifer_boxes - difference = 54 :=
by sorry

end kim_cookie_boxes_l1190_119000


namespace snake_head_fraction_l1190_119079

theorem snake_head_fraction (total_length body_length : ℝ) 
  (h1 : total_length = 10)
  (h2 : body_length = 9)
  (h3 : body_length < total_length) :
  (total_length - body_length) / total_length = 1 / 10 := by
sorry

end snake_head_fraction_l1190_119079


namespace sqrt_square_negative_l1190_119098

theorem sqrt_square_negative : Real.sqrt ((-2023)^2) = 2023 := by
  sorry

end sqrt_square_negative_l1190_119098


namespace shaded_area_in_circle_l1190_119025

/-- The area of a specific shaded region in a circle -/
theorem shaded_area_in_circle (r : ℝ) (h : r = 5) :
  let circle_area := π * r^2
  let triangle_area := r^2 / 2
  let sector_area := circle_area / 4
  2 * triangle_area + 2 * sector_area = 25 + 25 * π / 2 := by
  sorry

end shaded_area_in_circle_l1190_119025


namespace central_angle_of_chord_l1190_119040

theorem central_angle_of_chord (α : Real) (chord_length : Real) :
  (∀ R, R = 1 → chord_length = Real.sqrt 3 → 2 * Real.sin (α / 2) = chord_length) →
  α = 2 * Real.pi / 3 := by
  sorry

end central_angle_of_chord_l1190_119040


namespace f_prime_at_two_l1190_119070

-- Define f as a real-valued function
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_prime_at_two
  (h1 : (1 - 0) / (2 - 0) = 1 / 2)  -- Slope of line through (0,0) and (2,1) is 1/2
  (h2 : f 0 = 0)                    -- f(0) = 0
  (h3 : f 2 = 2)                    -- f(2) = 2
  (h4 : (2 * (deriv f 2) - (f 2)) / (2^2) = 1 / 2)  -- Derivative of f(x)/x at x=2 equals slope
  : deriv f 2 = 2 := by
sorry

end f_prime_at_two_l1190_119070


namespace second_car_speed_l1190_119063

/-- Given two cars traveling in opposite directions for 2.5 hours,
    with one car traveling at 60 mph and the total distance between them
    being 310 miles after 2.5 hours, prove that the speed of the second car is 64 mph. -/
theorem second_car_speed (car1_speed : ℝ) (car2_speed : ℝ) (time : ℝ) (total_distance : ℝ) :
  car1_speed = 60 →
  time = 2.5 →
  total_distance = 310 →
  car1_speed * time + car2_speed * time = total_distance →
  car2_speed = 64 := by
  sorry

end second_car_speed_l1190_119063


namespace lcm_of_20_45_75_l1190_119035

theorem lcm_of_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by sorry

end lcm_of_20_45_75_l1190_119035


namespace problem_3_l1190_119037

theorem problem_3 (a : ℝ) : a = 1 / (Real.sqrt 5 - 2) → 2 * a^2 - 8 * a + 1 = 3 := by
  sorry

end problem_3_l1190_119037


namespace residue_negative_811_mod_24_l1190_119014

theorem residue_negative_811_mod_24 : Int.mod (-811) 24 = 5 := by
  sorry

end residue_negative_811_mod_24_l1190_119014


namespace sandy_clothes_cost_l1190_119052

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℚ := 13.99

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℚ := 12.14

/-- The amount Sandy spent on a jacket -/
def jacket_cost : ℚ := 7.43

/-- The total amount Sandy spent on clothes -/
def total_cost : ℚ := shorts_cost + shirt_cost + jacket_cost

theorem sandy_clothes_cost : total_cost = 33.56 := by sorry

end sandy_clothes_cost_l1190_119052


namespace horners_rule_for_specific_polynomial_v3_value_at_3_l1190_119048

def horner_step (a : ℕ) (x v : ℕ) : ℕ := v * x + a

def horners_rule (coeffs : List ℕ) (x : ℕ) : ℕ :=
  coeffs.foldl (horner_step x) 0

theorem horners_rule_for_specific_polynomial (x : ℕ) :
  horners_rule [1, 1, 3, 2, 0, 1] x = x^5 + 2*x^3 + 3*x^2 + x + 1 := by sorry

theorem v3_value_at_3 :
  let coeffs := [1, 1, 3, 2, 0, 1]
  let x := 3
  let v0 := 1
  let v1 := horner_step 0 x v0
  let v2 := horner_step 2 x v1
  let v3 := horner_step 3 x v2
  v3 = 36 := by sorry

end horners_rule_for_specific_polynomial_v3_value_at_3_l1190_119048


namespace cheese_division_theorem_l1190_119023

/-- Represents the weight of cheese pieces -/
structure CheesePair :=
  (larger : ℕ)
  (smaller : ℕ)

/-- Simulates taking a bite from the larger piece -/
def takeBite (pair : CheesePair) : CheesePair :=
  ⟨pair.larger - pair.smaller, pair.smaller⟩

/-- Theorem: If after three bites, the cheese pieces are equal and weigh 20 grams each,
    then the original cheese weight was 680 grams -/
theorem cheese_division_theorem (initial : CheesePair) :
  (takeBite (takeBite (takeBite initial))) = ⟨20, 20⟩ →
  initial.larger + initial.smaller = 680 :=
by
  sorry

#check cheese_division_theorem

end cheese_division_theorem_l1190_119023


namespace juvys_garden_rows_l1190_119004

/-- Represents Juvy's garden -/
structure Garden where
  rows : ℕ
  plants_per_row : ℕ
  parsley_rows : ℕ
  rosemary_rows : ℕ
  chive_plants : ℕ

/-- Theorem: The number of rows in Juvy's garden is 20 -/
theorem juvys_garden_rows (g : Garden) 
  (h1 : g.plants_per_row = 10)
  (h2 : g.parsley_rows = 3)
  (h3 : g.rosemary_rows = 2)
  (h4 : g.chive_plants = 150)
  (h5 : g.chive_plants = g.plants_per_row * (g.rows - g.parsley_rows - g.rosemary_rows)) :
  g.rows = 20 := by
  sorry

end juvys_garden_rows_l1190_119004


namespace equation_system_implies_third_equation_l1190_119093

theorem equation_system_implies_third_equation (a b : ℝ) :
  a^2 - 3*a*b + 2*b^2 + a - b = 0 →
  a^2 - 2*a*b + b^2 - 5*a + 7*b = 0 →
  a*b - 12*a + 15*b = 0 := by
sorry

end equation_system_implies_third_equation_l1190_119093


namespace isosceles_triangle_is_convex_l1190_119094

-- Define an isosceles triangle
structure IsoscelesTriangle where
  sides : Fin 3 → ℝ
  is_isosceles : ∃ (i j : Fin 3), i ≠ j ∧ sides i = sides j

-- Define a convex polygon
def is_convex (polygon : Fin n → ℝ × ℝ) : Prop :=
  ∀ i j : Fin n, ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 →
    ∃ k : Fin n, polygon k = (1 - t) • (polygon i) + t • (polygon j)

-- Theorem statement
theorem isosceles_triangle_is_convex (T : IsoscelesTriangle) :
  is_convex (λ i : Fin 3 => sorry) :=
sorry

end isosceles_triangle_is_convex_l1190_119094


namespace sum_of_segments_l1190_119032

/-- Given a number line with points P at 3 and V at 33, and the line between them
    divided into six equal parts, the sum of the lengths of PS and TV is 25. -/
theorem sum_of_segments (P V Q R S T U : ℝ) : 
  P = 3 → V = 33 → 
  Q - P = R - Q → R - Q = S - R → S - R = T - S → T - S = U - T → U - T = V - U →
  (S - P) + (V - T) = 25 := by sorry

end sum_of_segments_l1190_119032


namespace problem_solution_l1190_119066

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) : (x^4 * y^2) / 8 = 162 := by
  sorry

end problem_solution_l1190_119066


namespace major_premise_false_correct_answer_is_major_premise_wrong_l1190_119061

-- Define the properties of a rhombus
structure Rhombus where
  diagonals_perpendicular : Bool
  diagonals_bisect : Bool
  diagonals_equal : Bool

-- Define a square as a special case of rhombus
def Square : Rhombus where
  diagonals_perpendicular := true
  diagonals_bisect := true
  diagonals_equal := true

-- Define the syllogism
def syllogism : Prop :=
  ∀ (r : Rhombus), r.diagonals_equal = true

-- Theorem stating that the major premise of the syllogism is false
theorem major_premise_false : ¬syllogism := by
  sorry

-- Theorem stating that the correct answer is that the major premise is wrong
theorem correct_answer_is_major_premise_wrong : 
  (¬syllogism) ∧ (Square.diagonals_equal = true) := by
  sorry

end major_premise_false_correct_answer_is_major_premise_wrong_l1190_119061


namespace num_arrangements_equals_5040_l1190_119085

/-- The number of candidates --/
def n : ℕ := 8

/-- The number of volunteers to be selected --/
def k : ℕ := 5

/-- The number of days --/
def days : ℕ := 5

/-- Function to calculate the number of arrangements --/
def num_arrangements (n k : ℕ) : ℕ :=
  let only_one := 2 * (n - 2).choose (k - 1) * k.factorial
  let both := (n - 2).choose (k - 2) * (k - 2).factorial * 2 * (k - 1)
  only_one + both

/-- Theorem stating the number of arrangements --/
theorem num_arrangements_equals_5040 :
  num_arrangements n k = 5040 := by sorry

end num_arrangements_equals_5040_l1190_119085


namespace original_number_before_increase_l1190_119087

theorem original_number_before_increase (x : ℝ) : x * 1.3 = 650 → x = 500 := by
  sorry

end original_number_before_increase_l1190_119087


namespace appropriate_sampling_methods_l1190_119054

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a region -/
inductive Region
  | A
  | B
  | C
  | D

/-- Represents the company's sales point distribution -/
structure SalesPointDistribution where
  total_points : Nat
  region_points : Region → Nat
  large_points_in_C : Nat

/-- Represents an investigation -/
structure Investigation where
  sample_size : Nat
  population_size : Nat

/-- Determines the appropriate sampling method for an investigation -/
def appropriate_sampling_method (dist : SalesPointDistribution) (inv : Investigation) : SamplingMethod :=
  sorry

/-- The company's actual sales point distribution -/
def company_distribution : SalesPointDistribution :=
  { total_points := 600,
    region_points := fun r => match r with
      | Region.A => 150
      | Region.B => 120
      | Region.C => 180
      | Region.D => 150,
    large_points_in_C := 20 }

/-- Investigation ① -/
def investigation_1 : Investigation :=
  { sample_size := 100,
    population_size := 600 }

/-- Investigation ② -/
def investigation_2 : Investigation :=
  { sample_size := 7,
    population_size := 20 }

/-- Theorem stating the appropriate sampling methods for the given investigations -/
theorem appropriate_sampling_methods :
  appropriate_sampling_method company_distribution investigation_1 = SamplingMethod.StratifiedSampling ∧
  appropriate_sampling_method company_distribution investigation_2 = SamplingMethod.SimpleRandomSampling :=
sorry

end appropriate_sampling_methods_l1190_119054


namespace rectangles_in_4x5_grid_l1190_119065

/-- The number of rectangles in a grid with sides along the grid lines -/
def count_rectangles (m n : ℕ) : ℕ :=
  (m * (m + 1) * n * (n + 1)) / 4

/-- Theorem: In a 4 × 5 grid, the total number of rectangles with sides along the grid lines is 24 -/
theorem rectangles_in_4x5_grid :
  count_rectangles 4 5 = 24 := by
  sorry

end rectangles_in_4x5_grid_l1190_119065


namespace rect_to_cylindrical_7_neg7_4_l1190_119009

/-- Converts rectangular coordinates to cylindrical coordinates -/
def rect_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ := sorry

theorem rect_to_cylindrical_7_neg7_4 :
  let (r, θ, z) := rect_to_cylindrical 7 (-7) 4
  r = 7 * Real.sqrt 2 ∧
  θ = 7 * Real.pi / 4 ∧
  z = 4 ∧
  r > 0 ∧
  0 ≤ θ ∧ θ < 2 * Real.pi := by sorry

end rect_to_cylindrical_7_neg7_4_l1190_119009


namespace hexagon_triangle_count_l1190_119038

/-- Represents a regular hexagon -/
structure RegularHexagon :=
  (area : ℝ)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (area : ℝ)

/-- Counts the number of equilateral triangles with a given area that can be formed from the vertices of a set of regular hexagons -/
def countEquilateralTriangles (hexagons : List RegularHexagon) (targetTriangle : EquilateralTriangle) : ℕ :=
  sorry

/-- The main theorem stating that 4 regular hexagons with area 6 can form 8 equilateral triangles with area 4 -/
theorem hexagon_triangle_count :
  let hexagons := List.replicate 4 { area := 6 : RegularHexagon }
  let targetTriangle := { area := 4 : EquilateralTriangle }
  countEquilateralTriangles hexagons targetTriangle = 8 := by sorry

end hexagon_triangle_count_l1190_119038


namespace x_twelve_equals_one_l1190_119049

theorem x_twelve_equals_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 := by
  sorry

end x_twelve_equals_one_l1190_119049


namespace arithmetic_sequence_common_difference_l1190_119019

/-- For an arithmetic sequence {a_n} with a_2 = 3 and a_5 = 12, the common difference d is 3. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  (h_a2 : a 2 = 3)  -- Given: a_2 = 3
  (h_a5 : a 5 = 12)  -- Given: a_5 = 12
  : ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
sorry

end arithmetic_sequence_common_difference_l1190_119019


namespace katie_game_difference_l1190_119088

def katie_new_games : ℕ := 57
def katie_old_games : ℕ := 39
def friends_new_games : ℕ := 34

theorem katie_game_difference :
  (katie_new_games + katie_old_games) - friends_new_games = 62 := by
  sorry

end katie_game_difference_l1190_119088


namespace objects_meeting_probability_l1190_119053

/-- The probability of two objects meeting on a coordinate plane -/
theorem objects_meeting_probability :
  let start_C : ℕ × ℕ := (0, 0)
  let start_D : ℕ × ℕ := (4, 6)
  let step_length : ℕ := 1
  let prob_C_right : ℚ := 1/2
  let prob_C_up : ℚ := 1/2
  let prob_D_left : ℚ := 1/2
  let prob_D_down : ℚ := 1/2
  ∃ (meeting_prob : ℚ), meeting_prob = 55/1024 :=
by sorry

end objects_meeting_probability_l1190_119053


namespace non_square_difference_characterization_l1190_119024

/-- A natural number that cannot be represented as the difference of squares of any two natural numbers. -/
def NonSquareDifference (n : ℕ) : Prop :=
  ∀ x y : ℕ, n ≠ x^2 - y^2

/-- Characterization of numbers that cannot be represented as the difference of squares. -/
theorem non_square_difference_characterization (n : ℕ) :
  NonSquareDifference n ↔ n = 1 ∨ n = 4 ∨ ∃ k : ℕ, n = 4*k + 2 :=
sorry

end non_square_difference_characterization_l1190_119024


namespace inequality_solution_l1190_119055

theorem inequality_solution (x : ℝ) : (x + 4) / (x^2 + 4*x + 13) ≥ 0 ↔ x ≥ -4 := by
  sorry

end inequality_solution_l1190_119055


namespace parabola_focus_distance_l1190_119056

theorem parabola_focus_distance (p : ℝ) (h1 : p > 0) : 
  let focus : ℝ × ℝ := (p / 2, 0)
  let distance_to_line (point : ℝ × ℝ) : ℝ := 
    |-(point.1) + point.2 - 1| / Real.sqrt 2
  distance_to_line focus = Real.sqrt 2 → p = 2 := by
sorry

end parabola_focus_distance_l1190_119056
