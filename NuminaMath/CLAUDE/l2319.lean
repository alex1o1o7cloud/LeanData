import Mathlib

namespace senate_subcommittee_seating_l2319_231924

/-- The number of ways to arrange senators around a circular table -/
def arrange_senators (num_democrats : ℕ) (num_republicans : ℕ) : ℕ :=
  -- Arrangements of 2 blocks (Democrats and Republicans) in a circle
  1 *
  -- Permutations of Democrats within their block
  (Nat.factorial num_democrats) *
  -- Permutations of Republicans within their block
  (Nat.factorial num_republicans)

/-- Theorem stating the number of arrangements for 6 Democrats and 6 Republicans -/
theorem senate_subcommittee_seating :
  arrange_senators 6 6 = 518400 :=
by sorry

end senate_subcommittee_seating_l2319_231924


namespace buffet_meal_combinations_l2319_231975

def meat_options : ℕ := 4
def vegetable_options : ℕ := 5
def dessert_options : ℕ := 5
def vegetables_to_choose : ℕ := 3

theorem buffet_meal_combinations :
  meat_options * Nat.choose vegetable_options vegetables_to_choose * dessert_options = 200 := by
  sorry

end buffet_meal_combinations_l2319_231975


namespace ball_attendees_l2319_231997

theorem ball_attendees :
  ∀ (ladies gentlemen : ℕ),
  ladies + gentlemen < 50 →
  (3 * ladies) / 4 = (5 * gentlemen) / 7 →
  ladies + gentlemen = 41 :=
by
  sorry

end ball_attendees_l2319_231997


namespace parametric_to_standard_equation_l2319_231921

theorem parametric_to_standard_equation (x y α : ℝ) :
  x = Real.sqrt 3 * Real.cos α + 2 ∧ 
  y = Real.sqrt 3 * Real.sin α - 3 →
  (x - 2)^2 + (y + 3)^2 = 3 := by
sorry

end parametric_to_standard_equation_l2319_231921


namespace least_number_divisibility_l2319_231907

theorem least_number_divisibility (n : ℕ) (h : n = 59789) : 
  let m := 16142
  (∀ k : ℕ, k < m → ¬((n + k) % 7 = 0 ∧ (n + k) % 11 = 0 ∧ (n + k) % 13 = 0 ∧ (n + k) % 17 = 0)) ∧ 
  ((n + m) % 7 = 0 ∧ (n + m) % 11 = 0 ∧ (n + m) % 13 = 0 ∧ (n + m) % 17 = 0) :=
by sorry

end least_number_divisibility_l2319_231907


namespace expression_evaluation_l2319_231956

theorem expression_evaluation : 5 + 7 * (2 - 9)^2 = 348 := by sorry

end expression_evaluation_l2319_231956


namespace cubic_expansion_sum_l2319_231913

theorem cubic_expansion_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (2 * x + 1)^3 = a₀ * x^3 + a₁ * x^2 + a₂ * x + a₃) →
  a₁ + a₃ = 13 := by
sorry

end cubic_expansion_sum_l2319_231913


namespace intersection_of_A_and_B_l2319_231929

def A : Set ℝ := {x | x^2 ≤ 1}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by
  sorry

end intersection_of_A_and_B_l2319_231929


namespace quadratic_linear_system_solution_l2319_231974

theorem quadratic_linear_system_solution : 
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    (x₁^2 - 6*x₁ + 8 = 0) ∧
    (x₂^2 - 6*x₂ + 8 = 0) ∧
    (2*x₁ - y₁ = 6) ∧
    (2*x₂ - y₂ = 6) ∧
    (y₁ = 2) ∧
    (y₂ = -2) :=
by
  sorry

end quadratic_linear_system_solution_l2319_231974


namespace swimmer_speed_in_still_water_l2319_231966

/-- Represents the speed of a swimmer in various conditions -/
structure SwimmerSpeed where
  stillWater : ℝ
  stream : ℝ

/-- Calculates the effective speed of the swimmer -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.stillWater + s.stream else s.stillWater - s.stream

/-- Theorem: Given the conditions, the swimmer's speed in still water is 5 km/h -/
theorem swimmer_speed_in_still_water
  (s : SwimmerSpeed)
  (h1 : effectiveSpeed s true * 3 = 18)  -- Downstream condition
  (h2 : effectiveSpeed s false * 3 = 12) -- Upstream condition
  : s.stillWater = 5 := by
  sorry


end swimmer_speed_in_still_water_l2319_231966


namespace pizza_bill_theorem_l2319_231937

/-- The total bill amount for a group of people dividing equally -/
def total_bill (num_people : ℕ) (amount_per_person : ℕ) : ℕ :=
  num_people * amount_per_person

/-- Theorem: For a group of 5 people paying $8 each, the total bill is $40 -/
theorem pizza_bill_theorem :
  total_bill 5 8 = 40 := by
  sorry

end pizza_bill_theorem_l2319_231937


namespace part_one_part_two_l2319_231954

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

-- Part 1
theorem part_one (m n : ℝ) :
  (∀ x, f m x < 0 ↔ -2 < x ∧ x < n) →
  m = 5/2 ∧ n = 1/2 := by sorry

-- Part 2
theorem part_two (m : ℝ) :
  (∀ x ∈ Set.Icc m (m+1), f m x < 0) →
  m > -Real.sqrt 2 / 2 ∧ m < 0 := by sorry

end part_one_part_two_l2319_231954


namespace isosceles_triangle_perimeter_l2319_231936

/-- An isosceles triangle with two sides of 6cm and 13cm has a perimeter of 32cm. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 6 ∧ b = 13 ∧ c = 13 →  -- Two sides are 13cm (base) and one side is 6cm
  a + b > c ∧ a + c > b ∧ b + c > a →  -- Triangle inequality
  a + b + c = 32 :=  -- Perimeter is 32cm
by
  sorry


end isosceles_triangle_perimeter_l2319_231936


namespace exists_special_polynomial_l2319_231903

/-- A fifth-degree polynomial with specific properties on [-1,1] -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ (x₁ x₂ : ℝ), -1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 ∧
    p x₁ = 1 ∧ p (-x₂) = 1 ∧ p (-x₁) = -1 ∧ p x₂ = -1) ∧
  p (-1) = 0 ∧ p 1 = 0 ∧
  ∀ x, x ∈ Set.Icc (-1) 1 → -1 ≤ p x ∧ p x ≤ 1

/-- There exists a fifth-degree polynomial with the special properties -/
theorem exists_special_polynomial :
  ∃ (p : ℝ → ℝ), ∃ (a b c d e f : ℝ),
    (∀ x, p x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) ∧
    special_polynomial p :=
sorry

end exists_special_polynomial_l2319_231903


namespace sum_of_cubes_l2319_231928

theorem sum_of_cubes (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a * b + a * c + b * c = 2) 
  (h3 : a * b * c = 5) : 
  a^3 + b^3 + c^3 = 14 := by sorry

end sum_of_cubes_l2319_231928


namespace james_beat_record_by_116_l2319_231981

/-- Represents James's scoring statistics for the football season -/
structure JamesStats where
  touchdownsPerGame : ℕ
  gamesInSeason : ℕ
  twoPointConversions : ℕ
  fieldGoals : ℕ
  extraPointAttempts : ℕ

/-- Calculates the total points scored by James -/
def totalPoints (stats : JamesStats) : ℕ :=
  stats.touchdownsPerGame * 6 * stats.gamesInSeason +
  stats.twoPointConversions * 2 +
  stats.fieldGoals * 3 +
  stats.extraPointAttempts

/-- The old record for points scored in a season -/
def oldRecord : ℕ := 300

/-- Theorem stating that James beat the old record by 116 points -/
theorem james_beat_record_by_116 (stats : JamesStats)
  (h1 : stats.touchdownsPerGame = 4)
  (h2 : stats.gamesInSeason = 15)
  (h3 : stats.twoPointConversions = 6)
  (h4 : stats.fieldGoals = 8)
  (h5 : stats.extraPointAttempts = 20) :
  totalPoints stats - oldRecord = 116 := by
  sorry

#eval totalPoints { touchdownsPerGame := 4, gamesInSeason := 15, twoPointConversions := 6, fieldGoals := 8, extraPointAttempts := 20 } - oldRecord

end james_beat_record_by_116_l2319_231981


namespace equation_solution_l2319_231990

theorem equation_solution (x a b : ℝ) (h : b / x^3 + 1 / x^2 + 1 / x + 1 = 0) :
  x^3 + x^2 + x + a = a - b := by
  sorry

end equation_solution_l2319_231990


namespace machine_time_difference_l2319_231941

/- Define the variables -/
variable (W : ℝ) -- Number of widgets
variable (X : ℝ) -- Rate of machine X in widgets per day
variable (Y : ℝ) -- Rate of machine Y in widgets per day

/- Define the conditions -/
axiom machine_X_rate : X = W / 6
axiom combined_rate : X + Y = 5 * W / 12
axiom machine_X_alone : 30 * X = 5 * W

/- State the theorem -/
theorem machine_time_difference : 
  W / X - W / Y = 2 := by sorry

end machine_time_difference_l2319_231941


namespace largest_unrepresentable_number_l2319_231977

theorem largest_unrepresentable_number (a b : ℤ) (ha : a > 1) (hb : b > 1) :
  ∃ n : ℤ, n = 47 ∧ (∀ m : ℤ, m > n → ∃ x y : ℤ, m = 7*a + 5*b + 7*x + 5*y) ∧
  (¬∃ x y : ℤ, n = 7*a + 5*b + 7*x + 5*y) := by
sorry

end largest_unrepresentable_number_l2319_231977


namespace hyperbola_eccentricity_l2319_231916

theorem hyperbola_eccentricity (a : ℝ) : 
  a > 0 → 
  (∃ x y : ℝ, x^2/a^2 - y^2/3 = 1) → 
  (∃ c : ℝ, c/a = 2) → 
  a = 1 := by
  sorry

end hyperbola_eccentricity_l2319_231916


namespace sum_distinct_prime_factors_of_420_l2319_231940

theorem sum_distinct_prime_factors_of_420 : 
  (Finset.sum (Nat.factors 420).toFinset id) = 17 := by
  sorry

end sum_distinct_prime_factors_of_420_l2319_231940


namespace sum_of_numbers_with_lcm_and_ratio_l2319_231979

theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ+) : 
  Nat.lcm a b = 108 → 
  (a : ℚ) / (b : ℚ) = 3 / 7 → 
  (a : ℕ) + b = 10 := by
sorry

end sum_of_numbers_with_lcm_and_ratio_l2319_231979


namespace port_distance_l2319_231922

/-- Represents a ship traveling between two ports -/
structure Ship where
  speed : ℝ
  trips : ℕ

/-- Represents the problem setup -/
structure PortProblem where
  blue : Ship
  green : Ship
  first_meeting_distance : ℝ
  total_distance : ℝ

/-- The theorem stating the distance between ports -/
theorem port_distance (p : PortProblem) 
  (h1 : p.blue.trips = 4)
  (h2 : p.green.trips = 3)
  (h3 : p.first_meeting_distance = 20)
  (h4 : p.blue.speed / p.green.speed = p.first_meeting_distance / (p.total_distance - p.first_meeting_distance))
  (h5 : p.blue.speed * p.blue.trips = p.green.speed * p.green.trips) :
  p.total_distance = 35 := by
  sorry

end port_distance_l2319_231922


namespace fraction_property_l2319_231978

theorem fraction_property (n : ℕ+) : 
  (∃ (a b c d e f : ℕ), 
    (1 : ℚ) / (2*n + 1) = (a*100000 + b*10000 + c*1000 + d*100 + e*10 + f) / 999999 ∧ 
    a + b + c + d + e + f = 999) ↔ 
  (2*n + 1 = 7 ∨ 2*n + 1 = 13) :=
sorry

end fraction_property_l2319_231978


namespace inequality_problem_l2319_231955

theorem inequality_problem (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (∀ x y z : ℝ, x < y ∧ y < z ∧ x * z < 0 → x * y^2 < x * z^2 → False) ∧
  (a * b > a * c) ∧
  (c * (b - a) > 0) ∧
  (a * c * (a - c) < 0) :=
sorry

end inequality_problem_l2319_231955


namespace probability_non_expired_bottle_l2319_231950

theorem probability_non_expired_bottle (total_bottles : ℕ) (expired_bottles : ℕ) 
  (h1 : total_bottles = 5) (h2 : expired_bottles = 1) : 
  (total_bottles - expired_bottles : ℚ) / total_bottles = 4 / 5 := by
  sorry

end probability_non_expired_bottle_l2319_231950


namespace point_on_line_for_all_k_l2319_231965

/-- The point P lies on the line (k+2)x + (1-k)y - 4k - 5 = 0 for all values of k. -/
theorem point_on_line_for_all_k :
  ∀ (k : ℝ), (k + 2) * 3 + (1 - k) * (-1) - 4 * k - 5 = 0 := by
sorry

end point_on_line_for_all_k_l2319_231965


namespace choose_three_from_nine_l2319_231910

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end choose_three_from_nine_l2319_231910


namespace factory_production_rate_solve_factory_production_rate_l2319_231933

/-- Proves that the daily production rate in the first year was 3650 televisions,
    given a 10% reduction in the second year and a total production of 3285 televisions
    in the second year. -/
theorem factory_production_rate : ℝ → Prop :=
  fun daily_rate =>
    let reduction_factor : ℝ := 0.9
    let second_year_production : ℝ := 3285
    daily_rate * reduction_factor * 365 = second_year_production →
    daily_rate = 3650

/-- The actual theorem statement -/
theorem solve_factory_production_rate : 
  ∃ (rate : ℝ), factory_production_rate rate :=
sorry

end factory_production_rate_solve_factory_production_rate_l2319_231933


namespace power_of_six_tens_digit_one_l2319_231917

theorem power_of_six_tens_digit_one : ∃ n : ℕ, (6^n) % 100 ≥ 10 ∧ (6^n) % 100 < 20 := by
  sorry

end power_of_six_tens_digit_one_l2319_231917


namespace sum_reciprocal_squares_cubic_l2319_231931

/-- Given a cubic equation x^3 - 12x^2 + 17x + 4 = 0 with real roots a, b, and c,
    prove that the sum of reciprocals of squares of roots equals 385/16 -/
theorem sum_reciprocal_squares_cubic (a b c : ℝ) : 
  a^3 - 12*a^2 + 17*a + 4 = 0 → 
  b^3 - 12*b^2 + 17*b + 4 = 0 → 
  c^3 - 12*c^2 + 17*c + 4 = 0 → 
  (1/a^2) + (1/b^2) + (1/c^2) = 385/16 :=
by sorry

end sum_reciprocal_squares_cubic_l2319_231931


namespace soccer_team_losses_l2319_231953

theorem soccer_team_losses (total_games : ℕ) (games_won : ℕ) (points_for_win : ℕ) 
  (points_for_draw : ℕ) (points_for_loss : ℕ) (total_points : ℕ) :
  total_games = 20 →
  games_won = 14 →
  points_for_win = 3 →
  points_for_draw = 1 →
  points_for_loss = 0 →
  total_points = 46 →
  ∃ (games_lost : ℕ) (games_drawn : ℕ),
    games_lost = 2 ∧
    games_won + games_drawn + games_lost = total_games ∧
    games_won * points_for_win + games_drawn * points_for_draw + games_lost * points_for_loss = total_points :=
by
  sorry

end soccer_team_losses_l2319_231953


namespace stadium_attendance_l2319_231912

/-- Given a stadium with initial attendees and girls, calculate remaining attendees after some leave --/
def remaining_attendees (total : ℕ) (girls : ℕ) : ℕ :=
  let boys := total - girls
  let boys_left := boys / 4
  let girls_left := girls / 8
  total - (boys_left + girls_left)

/-- Theorem stating that 480 people remain given the initial conditions --/
theorem stadium_attendance : remaining_attendees 600 240 = 480 := by
  sorry

end stadium_attendance_l2319_231912


namespace average_of_sequence_l2319_231993

theorem average_of_sequence (z : ℝ) : (0 + 3*z + 6*z + 12*z + 24*z) / 5 = 9*z := by
  sorry

end average_of_sequence_l2319_231993


namespace punch_bowl_ratio_l2319_231980

/-- Proves that the ratio of punch the cousin drank to the initial amount is 1:1 -/
theorem punch_bowl_ratio : 
  ∀ (initial_amount cousin_drink : ℚ),
  initial_amount > 0 →
  cousin_drink > 0 →
  initial_amount - cousin_drink + 4 - 2 + 12 = 16 →
  initial_amount + 14 = 16 →
  cousin_drink / initial_amount = 1 := by
sorry

end punch_bowl_ratio_l2319_231980


namespace john_reading_speed_l2319_231923

/-- Calculates the number of pages read per hour given the total pages, reading duration in weeks, and daily reading hours. -/
def pages_per_hour (total_pages : ℕ) (weeks : ℕ) (hours_per_day : ℕ) : ℚ :=
  (total_pages : ℚ) / ((weeks * 7 : ℕ) * hours_per_day)

/-- Theorem stating that under the given conditions, John reads 50 pages per hour. -/
theorem john_reading_speed :
  let total_pages : ℕ := 2800
  let weeks : ℕ := 4
  let hours_per_day : ℕ := 2
  pages_per_hour total_pages weeks hours_per_day = 50 := by
  sorry

end john_reading_speed_l2319_231923


namespace prob_select_copresidents_from_random_club_l2319_231973

/-- Represents a math club with a given number of students and two co-presidents -/
structure MathClub where
  students : Nat
  has_two_copresidents : Bool

/-- Calculates the probability of selecting two co-presidents when choosing three members from a club -/
def prob_select_copresidents (club : MathClub) : Rat :=
  if club.has_two_copresidents then
    (Nat.choose (club.students - 2) 1 : Rat) / (Nat.choose club.students 3 : Rat)
  else
    0

/-- The list of math clubs in the school district -/
def math_clubs : List MathClub := [
  { students := 5, has_two_copresidents := true },
  { students := 7, has_two_copresidents := true },
  { students := 8, has_two_copresidents := true }
]

/-- Theorem stating the probability of selecting two co-presidents when randomly choosing
    three members from a randomly selected club among the given math clubs -/
theorem prob_select_copresidents_from_random_club : 
  (1 / (math_clubs.length : Rat)) * (math_clubs.map prob_select_copresidents).sum = 11 / 60 := by
  sorry

end prob_select_copresidents_from_random_club_l2319_231973


namespace min_angle_function_l2319_231949

/-- For any triangle with internal angles α, β, and γ in radians, 
    the minimum value of 4/α + 1/(β + γ) is 9/π. -/
theorem min_angle_function (α β γ : ℝ) (h1 : 0 < α) (h2 : 0 < β) (h3 : 0 < γ) 
    (h4 : α + β + γ = π) : 
  (∀ α' β' γ' : ℝ, 0 < α' ∧ 0 < β' ∧ 0 < γ' ∧ α' + β' + γ' = π → 
    4 / α + 1 / (β + γ) ≤ 4 / α' + 1 / (β' + γ')) → 
  4 / α + 1 / (β + γ) = 9 / π := by
sorry

end min_angle_function_l2319_231949


namespace log_216_equals_3_log_2_plus_3_log_3_l2319_231905

theorem log_216_equals_3_log_2_plus_3_log_3 :
  Real.log 216 = 3 * (Real.log 2 + Real.log 3) := by
  sorry

end log_216_equals_3_log_2_plus_3_log_3_l2319_231905


namespace actual_height_of_boy_l2319_231971

/-- Proves that the actual height of a boy in a class of 35 boys is 226 cm, given the conditions of the problem. -/
theorem actual_height_of_boy (n : ℕ) (initial_avg : ℝ) (wrong_height : ℝ) (actual_avg : ℝ)
  (h1 : n = 35)
  (h2 : initial_avg = 181)
  (h3 : wrong_height = 166)
  (h4 : actual_avg = 179) :
  ∃ (actual_height : ℝ), actual_height = 226 ∧
    n * actual_avg = n * initial_avg - wrong_height + actual_height :=
by sorry

end actual_height_of_boy_l2319_231971


namespace min_value_of_f_l2319_231900

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x + m

-- State the theorem
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) ∧ 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -1) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≥ -1) :=
by sorry


end min_value_of_f_l2319_231900


namespace product_is_112015_l2319_231961

/-- Represents a three-digit number with distinct non-zero digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  distinct : hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones
  non_zero : hundreds ≠ 0 ∧ tens ≠ 0 ∧ ones ≠ 0
  valid_range : hundreds < 10 ∧ tens < 10 ∧ ones < 10

def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem product_is_112015 (iks ksi : ThreeDigitNumber) 
  (h1 : iks.hundreds = ksi.ones ∧ iks.tens = ksi.hundreds ∧ iks.ones = ksi.tens)
  (h2 : ∃ (c i k : Nat), c ≠ i ∧ c ≠ k ∧ i ≠ k ∧ 
    c = max iks.hundreds (max iks.tens iks.ones) ∧
    c = max ksi.hundreds (max ksi.tens ksi.ones))
  (h3 : ∃ (p : Nat), p = to_nat iks * to_nat ksi ∧ 
    (∃ (d1 d2 d3 d4 d5 d6 : Nat),
      p = 100000 * d1 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6 ∧
      d1 = c ∧ d2 = c ∧ d3 = c ∧
      ((d4 = i ∧ d5 = k ∧ d6 = 0) ∨ 
       (d4 = i ∧ d5 = 0 ∧ d6 = k) ∨ 
       (d4 = k ∧ d5 = i ∧ d6 = 0) ∨ 
       (d4 = k ∧ d5 = 0 ∧ d6 = i) ∨ 
       (d4 = 0 ∧ d5 = i ∧ d6 = k) ∨ 
       (d4 = 0 ∧ d5 = k ∧ d6 = i))))
  : to_nat iks * to_nat ksi = 112015 := by
  sorry

end product_is_112015_l2319_231961


namespace farm_sale_earnings_l2319_231925

/-- Calculates the total money earned from selling farm animals -/
def total_money_earned (num_cows : ℕ) (pig_cow_ratio : ℕ) (price_per_pig : ℕ) (price_per_cow : ℕ) : ℕ :=
  let num_pigs := num_cows * pig_cow_ratio
  let money_from_pigs := num_pigs * price_per_pig
  let money_from_cows := num_cows * price_per_cow
  money_from_pigs + money_from_cows

/-- Theorem stating that given the specific conditions, the total money earned is $48,000 -/
theorem farm_sale_earnings : total_money_earned 20 4 400 800 = 48000 := by
  sorry

end farm_sale_earnings_l2319_231925


namespace product_quotient_l2319_231943

theorem product_quotient (a b c d e f : ℚ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 750)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 2/3 := by
  sorry

end product_quotient_l2319_231943


namespace quadratic_inequality_l2319_231946

theorem quadratic_inequality (x : ℝ) : x^2 - 9*x + 14 ≤ 0 ↔ x ∈ Set.Icc 2 7 := by
  sorry

end quadratic_inequality_l2319_231946


namespace magnitude_AB_is_5_l2319_231911

def A : ℝ × ℝ := (-1, -6)
def B : ℝ × ℝ := (2, -2)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem magnitude_AB_is_5 : 
  Real.sqrt ((vector_AB.1)^2 + (vector_AB.2)^2) = 5 := by sorry

end magnitude_AB_is_5_l2319_231911


namespace intersection_of_M_and_N_l2319_231942

def M : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}
def N : Set ℝ := {x | x ≤ -3 ∨ x ≥ 6}

theorem intersection_of_M_and_N :
  M ∩ N = {x | -5 ≤ x ∧ x ≤ -3} := by sorry

end intersection_of_M_and_N_l2319_231942


namespace least_number_with_remainder_l2319_231976

theorem least_number_with_remainder (n : Nat) : n = 125 ↔ 
  (n % 12 = 5 ∧ ∀ m : Nat, m % 12 = 5 → m ≥ n) := by
  sorry

end least_number_with_remainder_l2319_231976


namespace xw_value_l2319_231952

theorem xw_value (x w : ℝ) (h1 : 7 * x = 28) (h2 : x + w = 9) : x * w = 20 := by
  sorry

end xw_value_l2319_231952


namespace card_area_theorem_l2319_231999

/-- Represents the dimensions of a rectangular card -/
structure CardDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a card given its dimensions -/
def cardArea (d : CardDimensions) : ℝ := d.length * d.width

/-- Theorem: If shortening one side of a 5x7 card by 2 inches results in an area of 15 square inches,
    then shortening the other side by 2 inches results in an area of 21 square inches -/
theorem card_area_theorem (original : CardDimensions) 
    (h1 : original.length = 5 ∧ original.width = 7)
    (h2 : ∃ (shortened : CardDimensions), 
      (shortened.length = original.length ∧ shortened.width = original.width - 2) ∨
      (shortened.length = original.length - 2 ∧ shortened.width = original.width) ∧
      cardArea shortened = 15) :
  ∃ (other_shortened : CardDimensions),
    ((other_shortened.length = original.length - 2 ∧ other_shortened.width = original.width) ∨
     (other_shortened.length = original.length ∧ other_shortened.width = original.width - 2)) ∧
    cardArea other_shortened = 21 := by
  sorry

end card_area_theorem_l2319_231999


namespace valid_combinations_count_l2319_231934

def digits : List Nat := [1, 1, 2, 2, 3, 3, 3, 3]

def is_valid_price (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 9999

def count_valid_combinations (digits : List Nat) : Nat :=
  sorry

theorem valid_combinations_count :
  count_valid_combinations digits = 14700 := by sorry

end valid_combinations_count_l2319_231934


namespace sum_of_consecutive_terms_l2319_231963

theorem sum_of_consecutive_terms (n : ℝ) : n + (n + 1) + (n + 2) + (n + 3) = 20 → n = 3.5 := by
  sorry

end sum_of_consecutive_terms_l2319_231963


namespace probability_three_same_color_l2319_231984

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def total_balls : ℕ := white_balls + black_balls
def drawn_balls : ℕ := 3

def probability_same_color : ℚ :=
  (Nat.choose white_balls drawn_balls + Nat.choose black_balls drawn_balls) /
  Nat.choose total_balls drawn_balls

theorem probability_three_same_color :
  probability_same_color = 1 / 5 := by
  sorry

end probability_three_same_color_l2319_231984


namespace binomial_26_6_l2319_231989

theorem binomial_26_6 (h1 : Nat.choose 25 5 = 53130) (h2 : Nat.choose 25 6 = 177100) :
  Nat.choose 26 6 = 230230 := by
  sorry

end binomial_26_6_l2319_231989


namespace circles_intersection_product_of_coordinates_l2319_231901

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 25 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 10*y + 37 = 0

-- Theorem stating that (2, 5) is the intersection point of the two circles
theorem circles_intersection :
  ∃! (x y : ℝ), circle1 x y ∧ circle2 x y ∧ x = 2 ∧ y = 5 := by
  sorry

-- Theorem stating that the product of the coordinates of the intersection point is 10
theorem product_of_coordinates :
  ∀ (x y : ℝ), circle1 x y ∧ circle2 x y → x * y = 10 := by
  sorry

end circles_intersection_product_of_coordinates_l2319_231901


namespace calculation_result_l2319_231992

theorem calculation_result : (377 / 13) / 29 * (1 / 4) / 2 = 0.125 := by
  sorry

end calculation_result_l2319_231992


namespace trailing_zeros_bound_l2319_231983

/-- The number of trailing zeros in the base-b representation of n! -/
def trailing_zeros (n : ℕ) (b : ℕ) : ℕ := sorry

/-- The largest prime factor of a natural number -/
def largest_prime_factor (b : ℕ) : ℕ := sorry

theorem trailing_zeros_bound {b : ℕ} (hb : b ≥ 2) :
  ∀ n : ℕ, trailing_zeros n b < n / (largest_prime_factor b - 1) := by sorry

end trailing_zeros_bound_l2319_231983


namespace number_problem_l2319_231996

theorem number_problem : ∃ x : ℚ, (x / 6) * 12 = 11 ∧ x = 11 / 2 := by
  sorry

end number_problem_l2319_231996


namespace circle_radius_from_polar_equation_l2319_231909

/-- The radius of a circle given by the polar equation ρ = 2cosθ is 1 -/
theorem circle_radius_from_polar_equation : 
  ∃ (center : ℝ × ℝ) (r : ℝ), 
    (∀ θ : ℝ, (2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ) ∈ 
      {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2}) ∧ 
    r = 1 :=
sorry

end circle_radius_from_polar_equation_l2319_231909


namespace julieta_total_spend_l2319_231958

/-- Calculates the total amount spent by Julieta at the store -/
def total_amount_spent (
  backpack_original_price : ℕ)
  (ringbinder_original_price : ℕ)
  (backpack_price_increase : ℕ)
  (ringbinder_price_reduction : ℕ)
  (num_ringbinders : ℕ) : ℕ :=
  (backpack_original_price + backpack_price_increase) +
  num_ringbinders * (ringbinder_original_price - ringbinder_price_reduction)

/-- Theorem stating that Julieta's total spend is $109 -/
theorem julieta_total_spend :
  total_amount_spent 50 20 5 2 3 = 109 := by
  sorry

end julieta_total_spend_l2319_231958


namespace only_B_is_quadratic_l2319_231944

-- Define the structure of a general function
structure GeneralFunction where
  f : ℝ → ℝ

-- Define what it means for a function to be quadratic
def is_quadratic (f : GeneralFunction) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f.f x = a * x^2 + b * x + c

-- Define the given functions
def function_A : GeneralFunction :=
  { f := λ x => 2 * x + 1 }

def function_B : GeneralFunction :=
  { f := λ x => -5 * x^2 - 3 }

def function_C (a b c : ℝ) : GeneralFunction :=
  { f := λ x => a * x^2 + b * x + c }

def function_D : GeneralFunction :=
  { f := λ x => x^3 + x + 1 }

-- State the theorem
theorem only_B_is_quadratic :
  ¬ is_quadratic function_A ∧
  is_quadratic function_B ∧
  (∃ a b c, ¬ is_quadratic (function_C a b c)) ∧
  ¬ is_quadratic function_D :=
sorry

end only_B_is_quadratic_l2319_231944


namespace range_of_a_l2319_231915

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a + Real.cos (2 * x) < 5 - 4 * Real.sin x + Real.sqrt (5 * a - 4)) →
  (4/5 ≤ a ∧ a < 8) :=
by sorry

end range_of_a_l2319_231915


namespace quadratic_complete_square_l2319_231986

theorem quadratic_complete_square (a b c : ℝ) (h : 4 * a^2 - 8 * a - 320 = 0) :
  ∃ s : ℝ, s = 81 ∧ ∃ k : ℝ, (a - k)^2 = s :=
sorry

end quadratic_complete_square_l2319_231986


namespace lydia_apple_tree_age_l2319_231967

theorem lydia_apple_tree_age (tree_fruit_time : ℕ) (planting_age : ℕ) : 
  tree_fruit_time = 10 → planting_age = 6 → planting_age + tree_fruit_time = 16 := by
  sorry

end lydia_apple_tree_age_l2319_231967


namespace arrangements_theorem_l2319_231987

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The total number of students in the group -/
def total_students : ℕ := num_boys + num_girls

/-- Function to calculate the number of arrangements where girls are not next to each other -/
def arrangements_girls_not_adjacent : ℕ := sorry

/-- Function to calculate the number of arrangements with girl A not at left end and girl B not at right end -/
def arrangements_girl_A_B_restricted : ℕ := sorry

/-- Function to calculate the number of arrangements where all boys stand next to each other -/
def arrangements_boys_together : ℕ := sorry

/-- Function to calculate the number of arrangements where A, B, C stand in height order -/
def arrangements_ABC_height_order : ℕ := sorry

theorem arrangements_theorem :
  arrangements_girls_not_adjacent = 480 ∧
  arrangements_girl_A_B_restricted = 504 ∧
  arrangements_boys_together = 144 ∧
  arrangements_ABC_height_order = 120 := by sorry

end arrangements_theorem_l2319_231987


namespace green_face_probability_l2319_231969

/-- Probability of rolling a green face on an octahedron with 5 green faces out of 8 total faces -/
theorem green_face_probability (total_faces : ℕ) (green_faces : ℕ) 
  (h1 : total_faces = 8) 
  (h2 : green_faces = 5) : 
  (green_faces : ℚ) / total_faces = 5 / 8 := by
  sorry

end green_face_probability_l2319_231969


namespace solution_set_quadratic_inequality_l2319_231972

theorem solution_set_quadratic_inequality :
  let S : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}
  S = {x | -1 ≤ x ∧ x ≤ 2} := by
  sorry

end solution_set_quadratic_inequality_l2319_231972


namespace fifth_month_sale_l2319_231904

def sales_first_four : List ℕ := [6435, 6927, 6855, 7230]
def sale_sixth : ℕ := 6191
def average_sale : ℕ := 6700
def num_months : ℕ := 6

theorem fifth_month_sale :
  let total_sales := average_sale * num_months
  let sum_known_sales := sales_first_four.sum + sale_sixth
  total_sales - sum_known_sales = 6562 := by
  sorry

end fifth_month_sale_l2319_231904


namespace garden_flowers_l2319_231927

/-- Represents a rectangular garden with a rose planted at a specific position -/
structure Garden where
  rows_front : Nat  -- Number of rows in front of the rose
  rows_back : Nat   -- Number of rows behind the rose
  cols_left : Nat   -- Number of columns to the left of the rose
  cols_right : Nat  -- Number of columns to the right of the rose

/-- Calculates the total number of flowers in the garden -/
def total_flowers (g : Garden) : Nat :=
  (g.rows_front + 1 + g.rows_back) * (g.cols_left + 1 + g.cols_right)

/-- Theorem stating the total number of flowers in the specified garden -/
theorem garden_flowers :
  let g : Garden := {
    rows_front := 6,
    rows_back := 15,
    cols_left := 8,
    cols_right := 12
  }
  total_flowers g = 462 := by
  sorry

#eval total_flowers { rows_front := 6, rows_back := 15, cols_left := 8, cols_right := 12 }

end garden_flowers_l2319_231927


namespace quadratic_is_square_of_binomial_l2319_231947

theorem quadratic_is_square_of_binomial :
  ∃ (r s : ℚ), (r * X + s)^2 = (81/16 : ℚ) * X^2 + 18 * X + 16 :=
by sorry

end quadratic_is_square_of_binomial_l2319_231947


namespace equation_solutions_l2319_231930

theorem equation_solutions :
  (∃ x : ℚ, 2 * x - 1 = 5 * x + 2 ↔ x = -1) ∧
  (∃ x : ℚ, (5 * x + 1) / 2 - (2 * x - 1) / 4 = 1 ↔ x = 1 / 8) :=
by sorry

end equation_solutions_l2319_231930


namespace suit_price_calculation_l2319_231919

theorem suit_price_calculation (original_price : ℝ) (increase_rate : ℝ) (discount_rate : ℝ) : 
  original_price = 150 →
  increase_rate = 0.2 →
  discount_rate = 0.2 →
  let increased_price := original_price * (1 + increase_rate)
  let final_price := increased_price * (1 - discount_rate)
  final_price = 144 := by
sorry

end suit_price_calculation_l2319_231919


namespace total_blankets_is_243_l2319_231964

/-- Represents the number of blankets collected over three days --/
def total_blankets : ℕ := 
  let day1_team := 15 * 2
  let day1_online := 5 * 4
  let day1_total := day1_team + day1_online

  let day2_new_members := 5 * 4
  let day2_original_members := 15 * 2 * 3
  let day2_online := 3 * 5
  let day2_total := day2_new_members + day2_original_members + day2_online

  let day3_schools := 22
  let day3_online := 7 * 3
  let day3_business := day2_total / 5
  let day3_total := day3_schools + day3_online + day3_business

  day1_total + day2_total + day3_total

/-- Theorem stating that the total number of blankets collected is 243 --/
theorem total_blankets_is_243 : total_blankets = 243 := by
  sorry

end total_blankets_is_243_l2319_231964


namespace pencil_length_l2319_231920

/-- The length of one pencil when two equal-length pencils together measure 24 cubes -/
theorem pencil_length (total_length : ℕ) (pencil_length : ℕ) : 
  total_length = 24 → 2 * pencil_length = total_length → pencil_length = 12 := by
  sorry

end pencil_length_l2319_231920


namespace lamp_price_after_discounts_l2319_231935

/-- Calculates the final price of a lamp after applying two discounts -/
theorem lamp_price_after_discounts (original_price : ℝ) 
  (first_discount_rate : ℝ) (second_discount_rate : ℝ) : 
  original_price = 120 → 
  first_discount_rate = 0.20 → 
  second_discount_rate = 0.15 → 
  original_price * (1 - first_discount_rate) * (1 - second_discount_rate) = 81.60 := by
sorry

end lamp_price_after_discounts_l2319_231935


namespace cos_15_degrees_l2319_231932

theorem cos_15_degrees : Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end cos_15_degrees_l2319_231932


namespace f_neg_three_equals_six_l2319_231962

-- Define the function f with the given property
def f : ℝ → ℝ := sorry

-- State the main theorem
theorem f_neg_three_equals_six :
  (∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y) →
  f 1 = 2 →
  f (-3) = 6 := by sorry

end f_neg_three_equals_six_l2319_231962


namespace max_abs_z_quadratic_equation_l2319_231982

/-- Given complex numbers a, b, c, z and a real number k satisfying certain conditions,
    the maximum value of |z| is (k^3 + √(k^6 + 4k^3)) / 2. -/
theorem max_abs_z_quadratic_equation (a b c z d : ℂ) (k : ℝ) 
    (h1 : Complex.abs a = Complex.abs d)
    (h2 : Complex.abs d > 0)
    (h3 : b = k • d)
    (h4 : c = k^2 • d)
    (h5 : a * z^2 + b * z + c = 0) :
    Complex.abs z ≤ (k^3 + Real.sqrt (k^6 + 4 * k^3)) / 2 :=
sorry

end max_abs_z_quadratic_equation_l2319_231982


namespace f_max_value_inequality_proof_l2319_231948

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| - |2*x + 4|

-- Statement 1: The maximum value of f(x) is 3
theorem f_max_value : ∃ (M : ℝ), M = 3 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

-- Statement 2: For positive real numbers x, y, z such that x + y + z = 3, y²/x + z²/y + x²/z ≥ 3
theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 3) :
  y^2 / x + z^2 / y + x^2 / z ≥ 3 :=
sorry

end f_max_value_inequality_proof_l2319_231948


namespace cos_330_degrees_l2319_231994

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l2319_231994


namespace intersection_of_A_and_B_l2319_231914

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end intersection_of_A_and_B_l2319_231914


namespace prob_qualified_volleyball_expected_net_profit_l2319_231988

-- Define the supply percentages and qualification rates
def supply_A : ℝ := 0.4
def supply_B : ℝ := 0.3
def supply_C : ℝ := 0.3
def qual_rate_A : ℝ := 0.95
def qual_rate_B : ℝ := 0.92
def qual_rate_C : ℝ := 0.96

-- Define profit and loss for each factory
def profit_A : ℝ := 10
def loss_A : ℝ := 5
def profit_C : ℝ := 8
def loss_C : ℝ := 6

-- Theorem 1: Probability of purchasing a qualified volleyball
theorem prob_qualified_volleyball :
  supply_A * qual_rate_A + supply_B * qual_rate_B + supply_C * qual_rate_C = 0.944 :=
sorry

-- Theorem 2: Expected net profit from purchasing one volleyball from Factory A and one from Factory C
theorem expected_net_profit :
  qual_rate_A * qual_rate_C * (profit_A + profit_C) +
  qual_rate_A * (1 - qual_rate_C) * (profit_A - loss_C) +
  (1 - qual_rate_A) * qual_rate_C * (profit_C - loss_A) +
  (1 - qual_rate_A) * (1 - qual_rate_C) * (-loss_A - loss_C) = 16.69 :=
sorry

end prob_qualified_volleyball_expected_net_profit_l2319_231988


namespace prob_C_correct_prob_C_given_A_correct_l2319_231959

/-- Represents a box containing red and white balls -/
structure Box where
  red : ℕ
  white : ℕ

/-- The probability of drawing a red ball from a box -/
def prob_red (b : Box) : ℚ :=
  b.red / (b.red + b.white)

/-- The probability of drawing a white ball from a box -/
def prob_white (b : Box) : ℚ :=
  b.white / (b.red + b.white)

/-- Initial state of box A -/
def box_A : Box := ⟨3, 2⟩

/-- Initial state of box B -/
def box_B : Box := ⟨2, 3⟩

/-- State of box B after transferring a ball from box A -/
def box_B_after (red_transferred : Bool) : Box :=
  if red_transferred then ⟨box_B.red + 1, box_B.white⟩
  else ⟨box_B.red, box_B.white + 1⟩

/-- Probability of event C given the initial conditions -/
def prob_C : ℚ :=
  (prob_red box_A * prob_red (box_B_after true)) +
  (prob_white box_A * prob_red (box_B_after false))

/-- Conditional probability of event C given event A -/
def prob_C_given_A : ℚ :=
  prob_red (box_B_after true)

theorem prob_C_correct : prob_C = 13 / 30 := by sorry

theorem prob_C_given_A_correct : prob_C_given_A = 1 / 2 := by sorry

end prob_C_correct_prob_C_given_A_correct_l2319_231959


namespace strips_intersection_angle_l2319_231951

/-- A strip is defined as the region between two parallel lines. -/
structure Strip where
  width : ℝ

/-- The intersection of two strips forms a parallelogram. -/
structure StripIntersection where
  strip1 : Strip
  strip2 : Strip
  area : ℝ

/-- The angle between two strips is the angle between their defining lines. -/
def angleBetweenStrips (intersection : StripIntersection) : ℝ := sorry

theorem strips_intersection_angle (intersection : StripIntersection) :
  intersection.strip1.width = 1 →
  intersection.strip2.width = 1 →
  intersection.area = 2 →
  angleBetweenStrips intersection = 30 * π / 180 := by
  sorry

end strips_intersection_angle_l2319_231951


namespace arithmetic_geometric_sequence_l2319_231902

/-- Given an arithmetic sequence with common difference 2 where a₁, a₃, a₄ form a geometric sequence, a₆ = 2 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →         -- a₁, a₃, a₄ form a geometric sequence
  a 6 = 2 := by
sorry

end arithmetic_geometric_sequence_l2319_231902


namespace simplify_expression_l2319_231945

theorem simplify_expression (y : ℝ) : (3 - Real.sqrt (y^2 - 9))^2 = y^2 - 6 * Real.sqrt (y^2 - 9) := by
  sorry

end simplify_expression_l2319_231945


namespace isosceles_triangle_perimeter_l2319_231998

/-- An isosceles triangle with two sides of lengths 1 and 2 has a perimeter of 5 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 1 ∧ b = 2 ∧ c = 2 →  -- Two sides are 1 and 2, the third side must be 2 to form an isosceles triangle
  a + b + c = 5 :=         -- The perimeter is the sum of all sides
by
  sorry


end isosceles_triangle_perimeter_l2319_231998


namespace games_won_l2319_231939

/-- Proves that the number of games won is 8, given the total games and lost games. -/
theorem games_won (total_games lost_games : ℕ) 
  (h1 : total_games = 12) 
  (h2 : lost_games = 4) : 
  total_games - lost_games = 8 := by
  sorry

#check games_won

end games_won_l2319_231939


namespace cat_path_tiles_l2319_231957

def garden_width : ℕ := 12
def garden_length : ℕ := 20
def tile_size : ℕ := 2
def tiles_width : ℕ := garden_width / tile_size
def tiles_length : ℕ := garden_length / tile_size

theorem cat_path_tiles : 
  tiles_width + tiles_length - Nat.gcd tiles_width tiles_length - 1 = 13 := by
  sorry

end cat_path_tiles_l2319_231957


namespace tina_pens_count_l2319_231906

/-- Calculates the total number of pens Tina has given the number of pink pens and the relationships between different colored pens. -/
def total_pens (pink : ℕ) (green_diff : ℕ) (blue_diff : ℕ) : ℕ :=
  pink + (pink - green_diff) + ((pink - green_diff) + blue_diff)

/-- Proves that given the conditions, Tina has 21 pens in total. -/
theorem tina_pens_count : total_pens 12 9 3 = 21 := by
  sorry

end tina_pens_count_l2319_231906


namespace maria_average_balance_l2319_231938

def maria_balance : List ℝ := [50, 250, 100, 200, 150, 250]

theorem maria_average_balance :
  (maria_balance.sum / maria_balance.length : ℝ) = 1000 / 6 := by sorry

end maria_average_balance_l2319_231938


namespace equal_probability_for_claudia_and_adela_l2319_231991

/-- The probability that a single die roll is not a multiple of 3 -/
def p_not_multiple_of_3 : ℚ := 2/3

/-- The probability that a single die roll is a multiple of 3 -/
def p_multiple_of_3 : ℚ := 1/3

/-- The number of dice rolled -/
def n : ℕ := 2

theorem equal_probability_for_claudia_and_adela :
  p_not_multiple_of_3 ^ n = n * p_multiple_of_3 * p_not_multiple_of_3 ^ (n - 1) :=
sorry

end equal_probability_for_claudia_and_adela_l2319_231991


namespace point_reflection_x_axis_l2319_231908

/-- Given a point P(-1,2) in the Cartesian coordinate system, 
    its coordinates with respect to the x-axis are (-1,-2). -/
theorem point_reflection_x_axis : 
  let P : ℝ × ℝ := (-1, 2)
  let reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  reflect_x P = (-1, -2) := by sorry

end point_reflection_x_axis_l2319_231908


namespace cube_root_square_l2319_231970

theorem cube_root_square (x : ℝ) : (x - 1)^(1/3) = 3 → (x - 1)^2 = 729 := by
  sorry

end cube_root_square_l2319_231970


namespace expand_polynomial_simplify_expression_l2319_231926

-- Problem 1
theorem expand_polynomial (x : ℝ) : x * (x + 3) * (x + 5) = x^3 + 8*x^2 + 15*x := by
  sorry

-- Problem 2
theorem simplify_expression (x y : ℝ) : (5*x + 2*y) * (5*x - 2*y) - 5*x * (5*x - 3*y) = -4*y^2 + 15*x*y := by
  sorry

end expand_polynomial_simplify_expression_l2319_231926


namespace expression_equality_l2319_231918

theorem expression_equality : (2^1006 + 5^1007)^2 - (2^1006 - 5^1007)^2 = 40 * 10^1006 := by
  sorry

end expression_equality_l2319_231918


namespace intersection_of_M_and_N_l2319_231968

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x ≥ 0}
def N : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} := by sorry

end intersection_of_M_and_N_l2319_231968


namespace inequality_range_l2319_231995

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, |x + 1| - |x - 2| > k) → k < -3 := by
  sorry

end inequality_range_l2319_231995


namespace circular_ring_area_l2319_231960

/-- Given a regular n-gon with area t, the area of the circular ring formed by
    its inscribed and circumscribed circles is (π * t * tan(180°/n)) / n. -/
theorem circular_ring_area (n : ℕ) (t : ℝ) (h1 : n ≥ 3) (h2 : t > 0) :
  let T := (Real.pi * t * Real.tan (Real.pi / n)) / n
  ∃ (r R : ℝ), r > 0 ∧ R > r ∧
    t = n * r^2 * Real.sin (Real.pi / n) * Real.cos (Real.pi / n) ∧
    R = r / Real.cos (Real.pi / n) ∧
    T = Real.pi * (R^2 - r^2) :=
by sorry

end circular_ring_area_l2319_231960


namespace function_value_2010_l2319_231985

theorem function_value_2010 (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 1) = -f x) 
  (h2 : f 1 = 4) : 
  f 2010 = -4 := by sorry

end function_value_2010_l2319_231985
