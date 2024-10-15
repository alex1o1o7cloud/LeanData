import Mathlib

namespace NUMINAMATH_CALUDE_pascal_leibniz_relation_l3277_327742

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Element of the Leibniz triangle -/
def leibniz (n k : ℕ) : ℚ := 1 / ((n + 1 : ℚ) * (binomial n k))

/-- Theorem stating the relationship between Pascal's and Leibniz's triangles -/
theorem pascal_leibniz_relation (n k : ℕ) (h : k ≤ n) :
  leibniz n k = 1 / ((n + 1 : ℚ) * (binomial n k)) := by
  sorry

end NUMINAMATH_CALUDE_pascal_leibniz_relation_l3277_327742


namespace NUMINAMATH_CALUDE_star_example_l3277_327732

-- Define the star operation
def star (x y : ℚ) : ℚ := (x + y) / 4

-- Theorem statement
theorem star_example : star (star 3 8) 6 = 35 / 16 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l3277_327732


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3277_327797

theorem interest_rate_calculation (P : ℝ) (t : ℝ) (diff : ℝ) (r : ℝ) : 
  P = 5100 → 
  t = 2 → 
  P * ((1 + r) ^ t - 1) - P * r * t = diff → 
  diff = 51 → 
  r = 0.1 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3277_327797


namespace NUMINAMATH_CALUDE_real_return_calculation_l3277_327724

theorem real_return_calculation (nominal_rate inflation_rate : ℝ) 
  (h1 : nominal_rate = 0.21)
  (h2 : inflation_rate = 0.10) :
  (1 + nominal_rate) / (1 + inflation_rate) - 1 = 0.10 := by
  sorry

end NUMINAMATH_CALUDE_real_return_calculation_l3277_327724


namespace NUMINAMATH_CALUDE_function_ordering_l3277_327754

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the property of being monotonically decreasing on an interval
def is_monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

-- State the theorem
theorem function_ordering (h1 : is_even f) (h2 : is_monotone_decreasing_on (fun x => f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 :=
sorry

end NUMINAMATH_CALUDE_function_ordering_l3277_327754


namespace NUMINAMATH_CALUDE_K_3_15_10_l3277_327766

noncomputable def K (x y z : ℝ) : ℝ := x / y + y / z + z / x

theorem K_3_15_10 : K 3 15 10 = 151 / 30 := by sorry

end NUMINAMATH_CALUDE_K_3_15_10_l3277_327766


namespace NUMINAMATH_CALUDE_divisor_sum_theorem_l3277_327798

def sum_of_geometric_series (a r : ℕ) (n : ℕ) : ℕ := (a * (r^(n+1) - 1)) / (r - 1)

theorem divisor_sum_theorem (i j k : ℕ) : 
  (sum_of_geometric_series 1 2 i) * (sum_of_geometric_series 1 3 j) * (sum_of_geometric_series 1 5 k) = 3600 → 
  i + j + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_theorem_l3277_327798


namespace NUMINAMATH_CALUDE_max_value_fraction_l3277_327730

theorem max_value_fraction (x : ℝ) : 
  (3 * x^2 + 9 * x + 17) / (3 * x^2 + 9 * x + 7) ≤ 41 ∧ 
  ∃ y : ℝ, (3 * y^2 + 9 * y + 17) / (3 * y^2 + 9 * y + 7) = 41 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3277_327730


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l3277_327707

theorem quadratic_coefficient_sum (p q a b : ℝ) : 
  (∀ x, -x^2 + p*x + q = 0 ↔ x = a ∨ x = b) →
  b < 1 →
  1 < a →
  p + q > 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l3277_327707


namespace NUMINAMATH_CALUDE_juan_oranges_picked_l3277_327750

theorem juan_oranges_picked (total : ℕ) (del_per_day : ℕ) (del_days : ℕ) : 
  total = 107 → del_per_day = 23 → del_days = 2 → 
  total - (del_per_day * del_days) = 61 := by
  sorry

end NUMINAMATH_CALUDE_juan_oranges_picked_l3277_327750


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3277_327721

/-- A line passing through point A(1,0) and tangent to the circle (x-3)^2 + (y-4)^2 = 4 -/
def TangentLine (l : Set (ℝ × ℝ)) : Prop :=
  ∃ k : ℝ, 
    (∀ x y, (x, y) ∈ l ↔ y = k * (x - 1)) ∧
    (abs (2 * k - 4) / Real.sqrt (k^2 + 1) = 2)

theorem tangent_line_equation :
  ∀ l : Set (ℝ × ℝ), TangentLine l →
    (∀ x y, (x, y) ∈ l ↔ x = 1 ∨ 3 * x - 4 * y - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3277_327721


namespace NUMINAMATH_CALUDE_brick_width_is_11_l3277_327744

-- Define the dimensions and quantities
def wall_length : ℝ := 200 -- in cm
def wall_width : ℝ := 300  -- in cm
def wall_height : ℝ := 2   -- in cm
def brick_length : ℝ := 25 -- in cm
def brick_height : ℝ := 6  -- in cm
def num_bricks : ℝ := 72.72727272727273

-- Define the theorem
theorem brick_width_is_11 :
  ∃ (brick_width : ℝ),
    brick_width = 11 ∧
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by sorry

end NUMINAMATH_CALUDE_brick_width_is_11_l3277_327744


namespace NUMINAMATH_CALUDE_cos_four_pi_thirds_minus_alpha_l3277_327795

theorem cos_four_pi_thirds_minus_alpha (α : ℝ) 
  (h : Real.sin (π / 6 + α) = 3 / 5) : 
  Real.cos (4 * π / 3 - α) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_pi_thirds_minus_alpha_l3277_327795


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3277_327782

/-- The distance between the vertices of a hyperbola with equation x²/144 - y²/64 = 1 is 24. -/
theorem hyperbola_vertices_distance : 
  ∃ (x y : ℝ), x^2/144 - y^2/64 = 1 → 
  ∃ (v₁ v₂ : ℝ × ℝ), (v₁.1^2/144 - v₁.2^2/64 = 1) ∧ 
                     (v₂.1^2/144 - v₂.2^2/64 = 1) ∧ 
                     (v₁.2 = 0) ∧ (v₂.2 = 0) ∧
                     (v₁.1 = -v₂.1) ∧
                     (Real.sqrt ((v₁.1 - v₂.1)^2 + (v₁.2 - v₂.2)^2) = 24) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3277_327782


namespace NUMINAMATH_CALUDE_james_argument_l3277_327708

theorem james_argument (initial_friends : ℕ) (new_friends : ℕ) (current_friends : ℕ) :
  initial_friends = 20 →
  new_friends = 1 →
  current_friends = 19 →
  initial_friends - (current_friends - new_friends) = 1 :=
by sorry

end NUMINAMATH_CALUDE_james_argument_l3277_327708


namespace NUMINAMATH_CALUDE_student_arrangement_theorem_l3277_327731

/-- The number of ways to arrange 6 students in two rows of three, 
    with the taller student in each column in the back row -/
def arrangement_count : ℕ := 90

/-- The number of students -/
def num_students : ℕ := 6

/-- The number of students in each row -/
def students_per_row : ℕ := 3

theorem student_arrangement_theorem :
  (num_students = 6) →
  (students_per_row = 3) →
  (∀ n : ℕ, n ≤ num_students → n > 0 → ∃! h : ℕ, h = n) →  -- All students have different heights
  arrangement_count = 90 :=
by sorry

end NUMINAMATH_CALUDE_student_arrangement_theorem_l3277_327731


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3277_327791

theorem cubic_root_sum (a b c : ℝ) : 
  (45 * a^3 - 70 * a^2 + 28 * a - 2 = 0) →
  (45 * b^3 - 70 * b^2 + 28 * b - 2 = 0) →
  (45 * c^3 - 70 * c^2 + 28 * c - 2 = 0) →
  a ≠ b → b ≠ c → a ≠ c →
  -1 < a → a < 1 →
  -1 < b → b < 1 →
  -1 < c → c < 1 →
  1/(1-a) + 1/(1-b) + 1/(1-c) = 13/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3277_327791


namespace NUMINAMATH_CALUDE_giyun_distance_to_school_l3277_327762

/-- The distance between Giyun's house and school -/
def distance_to_school (step_length : ℝ) (steps_per_minute : ℕ) (time_taken : ℕ) : ℝ :=
  step_length * (steps_per_minute : ℝ) * time_taken

/-- Theorem stating the distance between Giyun's house and school -/
theorem giyun_distance_to_school :
  distance_to_school 0.75 70 13 = 682.5 := by
  sorry

end NUMINAMATH_CALUDE_giyun_distance_to_school_l3277_327762


namespace NUMINAMATH_CALUDE_mom_has_enough_money_l3277_327755

/-- Proves that the amount of money mom brought is sufficient to buy the discounted clothing item -/
theorem mom_has_enough_money (mom_money : ℝ) (original_price : ℝ) 
  (h1 : mom_money = 230)
  (h2 : original_price = 268)
  : mom_money ≥ 0.8 * original_price := by
  sorry

end NUMINAMATH_CALUDE_mom_has_enough_money_l3277_327755


namespace NUMINAMATH_CALUDE_sum_five_consecutive_squares_not_perfect_square_l3277_327723

theorem sum_five_consecutive_squares_not_perfect_square (n : ℤ) : 
  ∃ (k : ℤ), (5 * n^2 + 10) ≠ k^2 := by
sorry

end NUMINAMATH_CALUDE_sum_five_consecutive_squares_not_perfect_square_l3277_327723


namespace NUMINAMATH_CALUDE_drews_lawn_width_l3277_327717

def lawn_problem (bag_coverage : ℝ) (length : ℝ) (num_bags : ℕ) (extra_coverage : ℝ) : Prop :=
  let total_coverage := bag_coverage * num_bags
  let actual_lawn_area := total_coverage - extra_coverage
  let width := actual_lawn_area / length
  width = 36

theorem drews_lawn_width :
  lawn_problem 250 22 4 208 := by
  sorry

end NUMINAMATH_CALUDE_drews_lawn_width_l3277_327717


namespace NUMINAMATH_CALUDE_intersection_M_N_l3277_327725

def M : Set ℝ := {0, 1, 2}

def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3277_327725


namespace NUMINAMATH_CALUDE_half_percent_is_point_zero_zero_five_l3277_327737

/-- Converts a percentage to its decimal representation -/
def percent_to_decimal (p : ℚ) : ℚ := p / 100

/-- States that 1/2 % is equal to 0.005 -/
theorem half_percent_is_point_zero_zero_five :
  percent_to_decimal (1/2) = 5/1000 := by sorry

end NUMINAMATH_CALUDE_half_percent_is_point_zero_zero_five_l3277_327737


namespace NUMINAMATH_CALUDE_antiderivative_increment_l3277_327785

-- Define the function f(x) = 2x + 4
def f (x : ℝ) : ℝ := 2 * x + 4

-- Define what it means for F to be an antiderivative of f on the interval [-2, 0]
def is_antiderivative (F : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) 0, (deriv F) x = f x

-- Theorem statement
theorem antiderivative_increment (F : ℝ → ℝ) (h : is_antiderivative F) :
  F 0 - F (-2) = 4 := by sorry

end NUMINAMATH_CALUDE_antiderivative_increment_l3277_327785


namespace NUMINAMATH_CALUDE_ball_probabilities_l3277_327796

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  total : ℕ
  red : ℕ
  black : ℕ
  white : ℕ
  green : ℕ

/-- The given ball counts in the problem -/
def problemCounts : BallCounts := {
  total := 12,
  red := 5,
  black := 4,
  white := 2,
  green := 1
}

/-- Calculates the probability of drawing a red or black ball -/
def probRedOrBlack (counts : BallCounts) : ℚ :=
  (counts.red + counts.black : ℚ) / counts.total

/-- Calculates the probability of drawing at least one red ball when two balls are drawn -/
def probAtLeastOneRed (counts : BallCounts) : ℚ :=
  let totalWays := counts.total * (counts.total - 1) / 2
  let oneRedWays := counts.red * (counts.total - counts.red)
  let twoRedWays := counts.red * (counts.red - 1) / 2
  (oneRedWays + twoRedWays : ℚ) / totalWays

theorem ball_probabilities (counts : BallCounts) 
    (h_total : counts.total = 12)
    (h_red : counts.red = 5)
    (h_black : counts.black = 4)
    (h_white : counts.white = 2)
    (h_green : counts.green = 1) :
    probRedOrBlack counts = 3/4 ∧ probAtLeastOneRed counts = 15/22 := by
  sorry

#eval probRedOrBlack problemCounts
#eval probAtLeastOneRed problemCounts

end NUMINAMATH_CALUDE_ball_probabilities_l3277_327796


namespace NUMINAMATH_CALUDE_sarah_tic_tac_toe_wins_l3277_327706

/-- Represents the outcome of Sarah's tic-tac-toe games -/
structure TicTacToeOutcome where
  wins : ℕ
  ties : ℕ
  losses : ℕ
  total_games : ℕ
  net_earnings : ℤ

/-- Calculates the net earnings based on game outcomes -/
def calculate_earnings (outcome : TicTacToeOutcome) : ℤ :=
  4 * outcome.wins + outcome.ties - 3 * outcome.losses

theorem sarah_tic_tac_toe_wins : 
  ∀ (outcome : TicTacToeOutcome),
    outcome.total_games = 200 →
    outcome.ties = 60 →
    outcome.net_earnings = -84 →
    calculate_earnings outcome = outcome.net_earnings →
    outcome.wins + outcome.ties + outcome.losses = outcome.total_games →
    outcome.wins = 39 := by
  sorry


end NUMINAMATH_CALUDE_sarah_tic_tac_toe_wins_l3277_327706


namespace NUMINAMATH_CALUDE_ceiling_bounds_l3277_327784

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ :=
  Int.ceil x

-- State the theorem
theorem ceiling_bounds (m : ℚ) : m ≤ ceiling m ∧ (ceiling m : ℚ) < m + 1 := by
  sorry

-- Define the property of ceiling function
axiom ceiling_property (a : ℚ) : ∃ b : ℚ, 0 ≤ b ∧ b < 1 ∧ a = ceiling a - b

end NUMINAMATH_CALUDE_ceiling_bounds_l3277_327784


namespace NUMINAMATH_CALUDE_rectangle_ratio_theorem_l3277_327764

theorem rectangle_ratio_theorem (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0) : 
  (s + 2*y = 3*s) → (x + y = 3*s) → (x / y = 2) := by
  sorry

#check rectangle_ratio_theorem

end NUMINAMATH_CALUDE_rectangle_ratio_theorem_l3277_327764


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_squared_l3277_327711

theorem x_squared_plus_reciprocal_squared (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_squared_l3277_327711


namespace NUMINAMATH_CALUDE_x_minus_y_equals_three_l3277_327765

theorem x_minus_y_equals_three (x y : ℝ) 
  (h1 : x + y = 8) 
  (h2 : x^2 - y^2 = 24) : 
  x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_three_l3277_327765


namespace NUMINAMATH_CALUDE_union_of_sets_l3277_327799

theorem union_of_sets : 
  let P : Set Int := {-2, 2}
  let Q : Set Int := {-1, 0, 2, 3}
  P ∪ Q = {-2, -1, 0, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l3277_327799


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_270_l3277_327746

theorem rectangle_area (square_area : ℝ) (rectangle_length : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_breadth := (3 / 5) * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_is_270 :
  rectangle_area 2025 10 = 270 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_270_l3277_327746


namespace NUMINAMATH_CALUDE_weekly_caloric_deficit_l3277_327757

def monday_calories : ℕ := 2500
def tuesday_calories : ℕ := 2600
def wednesday_calories : ℕ := 2400
def thursday_calories : ℕ := 2700
def friday_calories : ℕ := 2300
def saturday_calories : ℕ := 3500
def sunday_calories : ℕ := 2400

def monday_exercise : ℕ := 1000
def tuesday_exercise : ℕ := 1200
def wednesday_exercise : ℕ := 1300
def thursday_exercise : ℕ := 1600
def friday_exercise : ℕ := 1000
def saturday_exercise : ℕ := 0
def sunday_exercise : ℕ := 1200

def total_weekly_calories : ℕ := monday_calories + tuesday_calories + wednesday_calories + thursday_calories + friday_calories + saturday_calories + sunday_calories

def total_weekly_net_calories : ℕ := 
  (monday_calories - monday_exercise) + 
  (tuesday_calories - tuesday_exercise) + 
  (wednesday_calories - wednesday_exercise) + 
  (thursday_calories - thursday_exercise) + 
  (friday_calories - friday_exercise) + 
  (saturday_calories - saturday_exercise) + 
  (sunday_calories - sunday_exercise)

theorem weekly_caloric_deficit : 
  total_weekly_calories - total_weekly_net_calories = 6800 := by
  sorry

end NUMINAMATH_CALUDE_weekly_caloric_deficit_l3277_327757


namespace NUMINAMATH_CALUDE_part_one_part_two_l3277_327793

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x ≤ 10}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}

-- Part 1
theorem part_one : 
  M ∩ (Set.univ \ N 2) = {x : ℝ | -2 ≤ x ∧ x < 3} := by sorry

-- Part 2
theorem part_two : 
  ∀ a : ℝ, M ∪ N a = M → a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3277_327793


namespace NUMINAMATH_CALUDE_car_trip_duration_l3277_327702

/-- Represents the duration of a car trip with varying speeds. -/
def CarTrip (initial_speed initial_time additional_speed average_speed : ℝ) : Prop :=
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) / total_time = average_speed

/-- The car trip lasts 12 hours given the specified conditions. -/
theorem car_trip_duration :
  CarTrip 45 4 75 65 → ∃ (total_time : ℝ), total_time = 12 :=
by sorry

end NUMINAMATH_CALUDE_car_trip_duration_l3277_327702


namespace NUMINAMATH_CALUDE_bananas_shared_l3277_327783

theorem bananas_shared (initial : ℕ) (remaining : ℕ) (shared : ℕ) : 
  initial = 88 → remaining = 84 → shared = initial - remaining → shared = 4 := by
sorry

end NUMINAMATH_CALUDE_bananas_shared_l3277_327783


namespace NUMINAMATH_CALUDE_tourist_guide_distribution_l3277_327727

theorem tourist_guide_distribution :
  let n_tourists : ℕ := 8
  let n_guides : ℕ := 3
  let total_distributions := n_guides ^ n_tourists
  let at_least_one_empty := n_guides * (n_guides - 1) ^ n_tourists
  let at_least_two_empty := n_guides * 1 ^ n_tourists
  total_distributions - at_least_one_empty + at_least_two_empty = 5796 :=
by sorry

end NUMINAMATH_CALUDE_tourist_guide_distribution_l3277_327727


namespace NUMINAMATH_CALUDE_key_cleaning_time_l3277_327701

/-- The time it takes to clean one key -/
def clean_time : ℝ := 3

theorem key_cleaning_time :
  let assignment_time : ℝ := 10
  let remaining_keys : ℕ := 14
  let total_time : ℝ := 52
  clean_time * remaining_keys + assignment_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_key_cleaning_time_l3277_327701


namespace NUMINAMATH_CALUDE_ainsley_win_probability_l3277_327774

/-- A fair six-sided die -/
inductive Die : Type
| one | two | three | four | five | six

/-- The probability of rolling a specific outcome on a fair six-sided die -/
def prob_roll (outcome : Die) : ℚ := 1 / 6

/-- Whether a roll is a multiple of 3 -/
def is_multiple_of_three (roll : Die) : Prop :=
  roll = Die.three ∨ roll = Die.six

/-- The probability of rolling a multiple of 3 -/
def prob_multiple_of_three : ℚ :=
  (prob_roll Die.three) + (prob_roll Die.six)

/-- The probability of rolling a non-multiple of 3 -/
def prob_non_multiple_of_three : ℚ :=
  1 - prob_multiple_of_three

/-- The probability of Ainsley winning the game -/
theorem ainsley_win_probability :
  prob_multiple_of_three * prob_multiple_of_three = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ainsley_win_probability_l3277_327774


namespace NUMINAMATH_CALUDE_water_speed_swimming_problem_l3277_327735

/-- Proves that the speed of water is 2 km/h given the conditions of the swimming problem. -/
theorem water_speed_swimming_problem (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) (water_speed : ℝ) :
  still_water_speed = 4 →
  distance = 12 →
  time = 6 →
  distance = (still_water_speed - water_speed) * time →
  water_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_speed_swimming_problem_l3277_327735


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_2_l3277_327703

theorem factorization_of_2m_squared_minus_2 (m : ℝ) : 2 * m^2 - 2 = 2 * (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_2_l3277_327703


namespace NUMINAMATH_CALUDE_heart_nested_calculation_l3277_327789

def heart (a b : ℝ) : ℝ := (a + 2*b) * (a - b)

theorem heart_nested_calculation : heart 2 (heart 3 4) = -260 := by
  sorry

end NUMINAMATH_CALUDE_heart_nested_calculation_l3277_327789


namespace NUMINAMATH_CALUDE_bridge_length_l3277_327767

/-- Given a train of length 150 meters traveling at 45 km/hr that crosses a bridge in 30 seconds,
    the length of the bridge is 225 meters. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 225 := by
  sorry

#check bridge_length

end NUMINAMATH_CALUDE_bridge_length_l3277_327767


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3277_327772

/-- Arithmetic sequence a_i -/
def a (d i : ℕ) : ℕ := 1 + 2 * (i - 1) * d

/-- Arithmetic sequence b_i -/
def b (d i : ℕ) : ℕ := 1 + (i - 1) * d

/-- Sum of first k terms of a_i -/
def s (d k : ℕ) : ℕ := k + k * (k - 1) * d

/-- Sum of first k terms of b_i -/
def t (d k : ℕ) : ℕ := k + k * (k - 1) * (d / 2)

/-- A_n sequence -/
def A (d n : ℕ) : ℕ := s d (t d n)

/-- B_n sequence -/
def B (d n : ℕ) : ℕ := t d (s d n)

/-- Main theorem -/
theorem arithmetic_sequence_difference (d n : ℕ) :
  A d (n + 1) - A d n = (1 + n * d)^3 ∧
  B d (n + 1) - B d n = (n * d)^3 + (1 + n * d)^3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3277_327772


namespace NUMINAMATH_CALUDE_largest_sum_is_five_sixths_l3277_327747

theorem largest_sum_is_five_sixths :
  let sums : List ℚ := [1/3 + 1/2, 1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/7, 1/3 + 1/9]
  ∀ x ∈ sums, x ≤ 5/6 ∧ (5/6 ∈ sums) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_is_five_sixths_l3277_327747


namespace NUMINAMATH_CALUDE_boris_neighbors_l3277_327733

/-- Represents the six people in the circle -/
inductive Person : Type
  | Arkady : Person
  | Boris : Person
  | Vera : Person
  | Galya : Person
  | Danya : Person
  | Egor : Person

/-- Represents the circular arrangement of people -/
def Circle := List Person

/-- Check if two people are standing next to each other in the circle -/
def are_adjacent (c : Circle) (p1 p2 : Person) : Prop :=
  ∃ i : Nat, (c.get? i = some p1 ∧ c.get? ((i + 1) % c.length) = some p2) ∨
             (c.get? i = some p2 ∧ c.get? ((i + 1) % c.length) = some p1)

/-- Check if two people are standing opposite each other in the circle -/
def are_opposite (c : Circle) (p1 p2 : Person) : Prop :=
  ∃ i : Nat, c.get? i = some p1 ∧ c.get? ((i + c.length / 2) % c.length) = some p2

theorem boris_neighbors (c : Circle) :
  c.length = 6 →
  are_adjacent c Person.Danya Person.Vera →
  are_adjacent c Person.Danya Person.Egor →
  are_opposite c Person.Galya Person.Egor →
  ¬ are_adjacent c Person.Arkady Person.Galya →
  (are_adjacent c Person.Boris Person.Arkady ∧ are_adjacent c Person.Boris Person.Galya) :=
by sorry

end NUMINAMATH_CALUDE_boris_neighbors_l3277_327733


namespace NUMINAMATH_CALUDE_parabola_midpoint_locus_ratio_l3277_327768

/-- A parabola with vertex and focus -/
structure Parabola where
  vertex : ℝ × ℝ
  focus : ℝ × ℝ

/-- The locus of midpoints of right-angled chords of a parabola -/
def midpoint_locus (P : Parabola) : Parabola :=
  sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  sorry

theorem parabola_midpoint_locus_ratio (P : Parabola) :
  let Q := midpoint_locus P
  (distance P.focus Q.focus) / (distance P.vertex Q.vertex) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_midpoint_locus_ratio_l3277_327768


namespace NUMINAMATH_CALUDE_birds_total_distance_l3277_327792

/-- Calculates the total distance flown by six birds given their speeds and flight times -/
def total_distance_flown (eagle_speed falcon_speed pelican_speed hummingbird_speed hawk_speed swallow_speed : ℝ)
  (eagle_time falcon_time pelican_time hummingbird_time hawk_time swallow_time : ℝ) : ℝ :=
  eagle_speed * eagle_time +
  falcon_speed * falcon_time +
  pelican_speed * pelican_time +
  hummingbird_speed * hummingbird_time +
  hawk_speed * hawk_time +
  swallow_speed * swallow_time

/-- The total distance flown by all birds is 482.5 miles -/
theorem birds_total_distance :
  total_distance_flown 15 46 33 30 45 25 2.5 2.5 2.5 2.5 3 1.5 = 482.5 := by
  sorry

end NUMINAMATH_CALUDE_birds_total_distance_l3277_327792


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3277_327700

theorem quadratic_transformation (x : ℝ) : 
  (4 * x^2 - 16 * x - 400 = 0) → 
  (∃ p q : ℝ, (x + p)^2 = q ∧ q = 104) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3277_327700


namespace NUMINAMATH_CALUDE_hexagon_division_divisible_by_three_l3277_327751

/-- A regular hexagon divided into congruent parallelograms -/
structure RegularHexagonDivision where
  /-- The number of congruent parallelograms -/
  N : ℕ
  /-- The hexagon is divided into N congruent parallelograms -/
  is_division : N > 0

/-- Theorem: The number of congruent parallelograms in a regular hexagon division is divisible by 3 -/
theorem hexagon_division_divisible_by_three (h : RegularHexagonDivision) : 
  ∃ k : ℕ, h.N = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_hexagon_division_divisible_by_three_l3277_327751


namespace NUMINAMATH_CALUDE_opposite_numbers_abs_l3277_327729

theorem opposite_numbers_abs (m n : ℝ) : m + n = 0 → |m + n - 1| = 1 := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_abs_l3277_327729


namespace NUMINAMATH_CALUDE_m_range_l3277_327777

def f (m : ℝ) (x : ℝ) := 2*x^2 - 2*(m-2)*x + 3*m - 1

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

def is_ellipse_with_foci_on_y_axis (m : ℝ) : Prop :=
  m + 1 > 0 ∧ 9 - m > m + 1

def prop_p (m : ℝ) : Prop := is_increasing (f m) 1 2

def prop_q (m : ℝ) : Prop := is_ellipse_with_foci_on_y_axis m

theorem m_range (m : ℝ) 
  (h1 : prop_p m ∨ prop_q m) 
  (h2 : ¬(prop_p m ∧ prop_q m)) 
  (h3 : ¬¬(prop_p m)) : 
  m ≤ -1 ∨ m = 4 := by sorry

end NUMINAMATH_CALUDE_m_range_l3277_327777


namespace NUMINAMATH_CALUDE_calorie_allowance_for_longevity_l3277_327739

/-- Calculates the weekly calorie allowance for a person in their 60s aiming to live to 100 years old -/
def weeklyCalorieAllowance (averageDailyAllowance : ℕ) (reduction : ℕ) (daysInWeek : ℕ) : ℕ :=
  (averageDailyAllowance - reduction) * daysInWeek

/-- Theorem stating the weekly calorie allowance for a person in their 60s aiming to live to 100 years old -/
theorem calorie_allowance_for_longevity :
  weeklyCalorieAllowance 2000 500 7 = 10500 := by
  sorry

#eval weeklyCalorieAllowance 2000 500 7

end NUMINAMATH_CALUDE_calorie_allowance_for_longevity_l3277_327739


namespace NUMINAMATH_CALUDE_courtyard_tile_cost_l3277_327718

/-- Calculates the total cost of tiles for a courtyard --/
def total_tile_cost (length width : ℝ) (tiles_per_sqft : ℝ) 
  (green_tile_percentage : ℝ) (green_tile_cost red_tile_cost : ℝ) : ℝ :=
  let area := length * width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := total_tiles * green_tile_percentage
  let red_tiles := total_tiles - green_tiles
  (green_tiles * green_tile_cost) + (red_tiles * red_tile_cost)

/-- Theorem stating the total cost of tiles for the given courtyard specifications --/
theorem courtyard_tile_cost :
  total_tile_cost 10 25 4 0.4 3 1.5 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_tile_cost_l3277_327718


namespace NUMINAMATH_CALUDE_ten_candies_distribution_l3277_327745

/-- The number of ways to distribute n candies over days, with at least one candy per day -/
def candy_distribution (n : ℕ) : ℕ := 2^(n - 1)

/-- Theorem: The number of ways to distribute 10 candies over days, with at least one candy per day, is 512 -/
theorem ten_candies_distribution : candy_distribution 10 = 512 := by
  sorry

end NUMINAMATH_CALUDE_ten_candies_distribution_l3277_327745


namespace NUMINAMATH_CALUDE_roots_of_p_l3277_327761

def p (x : ℝ) : ℝ := x * (x + 3)^2 * (5 - x)

theorem roots_of_p :
  ∀ x : ℝ, p x = 0 ↔ x = 0 ∨ x = -3 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_p_l3277_327761


namespace NUMINAMATH_CALUDE_parabola_directrix_tangent_to_circle_l3277_327769

/-- The value of p for which the directrix of the parabola y² = 2px (p > 0) 
    is tangent to the circle x² + y² - 4x + 2y - 4 = 0 -/
theorem parabola_directrix_tangent_to_circle :
  ∀ p : ℝ, p > 0 →
  (∀ x y : ℝ, y^2 = 2*p*x) →
  (∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y - 4 = 0) →
  (∃ x y : ℝ, x = -p/2 ∧ (x - 2)^2 + (y + 1)^2 = 9) →
  p = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_tangent_to_circle_l3277_327769


namespace NUMINAMATH_CALUDE_valid_a_values_l3277_327763

/-- Set A defined by the quadratic equation x^2 - 2x - 3 = 0 -/
def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}

/-- Set B defined by the linear equation ax - 1 = 0, parameterized by a -/
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

/-- The set of values for a such that B is a subset of A -/
def valid_a : Set ℝ := {a | B a ⊆ A}

/-- Theorem stating that the set of valid a values is {-1, 0, 1/3} -/
theorem valid_a_values : valid_a = {-1, 0, 1/3} := by sorry

end NUMINAMATH_CALUDE_valid_a_values_l3277_327763


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3277_327719

theorem polynomial_simplification (y : ℝ) : 
  (4 * y^10 + 6 * y^9 + 3 * y^8) + (2 * y^12 + 5 * y^10 + y^9 + y^7 + 4 * y^4 + 7 * y + 9) = 
  2 * y^12 + 9 * y^10 + 7 * y^9 + 3 * y^8 + y^7 + 4 * y^4 + 7 * y + 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3277_327719


namespace NUMINAMATH_CALUDE_total_cost_is_24_l3277_327710

/-- The cost of one gold ring in dollars -/
def ring_cost : ℕ := 12

/-- The number of index fingers a person has -/
def index_fingers : ℕ := 2

/-- The total cost of buying gold rings for all index fingers -/
def total_cost : ℕ := ring_cost * index_fingers

/-- Theorem: The total cost of buying gold rings for all index fingers is 24 dollars -/
theorem total_cost_is_24 : total_cost = 24 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_24_l3277_327710


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3277_327712

theorem trigonometric_identity (x : ℝ) : 
  8.435 * (Real.sin (3 * x))^10 + (Real.cos (3 * x))^10 = 
  4 * ((Real.sin (3 * x))^6 + (Real.cos (3 * x))^6) / 
  (4 * (Real.cos (6 * x))^2 + (Real.sin (6 * x))^2) ↔ 
  ∃ k : ℤ, x = k * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3277_327712


namespace NUMINAMATH_CALUDE_angle_A_measure_l3277_327749

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (sum_angles : A + B + C = 180)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Theorem statement
theorem angle_A_measure (abc : Triangle) (h1 : abc.C = 3 * abc.B) (h2 : abc.B = 15) : 
  abc.A = 120 := by
sorry

end NUMINAMATH_CALUDE_angle_A_measure_l3277_327749


namespace NUMINAMATH_CALUDE_cara_right_neighbors_l3277_327705

/-- The number of Cara's friends -/
def num_friends : ℕ := 7

/-- The number of different friends who can sit immediately to Cara's right -/
def num_right_neighbors : ℕ := num_friends

theorem cara_right_neighbors :
  num_right_neighbors = num_friends :=
by sorry

end NUMINAMATH_CALUDE_cara_right_neighbors_l3277_327705


namespace NUMINAMATH_CALUDE_robins_hair_length_l3277_327752

/-- Calculates the final hair length after growth and cutting -/
def finalHairLength (initial growth cut : ℝ) : ℝ :=
  initial + growth - cut

/-- Theorem stating that given the initial conditions, the final hair length is 2 inches -/
theorem robins_hair_length :
  finalHairLength 14 8 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robins_hair_length_l3277_327752


namespace NUMINAMATH_CALUDE_max_non_managers_proof_l3277_327743

/-- Represents the number of managers in department A -/
def managers : ℕ := 9

/-- Represents the ratio of managers to non-managers in department A -/
def manager_ratio : ℚ := 7 / 37

/-- Represents the ratio of specialists to generalists in department A -/
def specialist_ratio : ℚ := 2 / 1

/-- Calculates the maximum number of non-managers in department A -/
def max_non_managers : ℕ := 39

/-- Theorem stating that 39 is the maximum number of non-managers in department A -/
theorem max_non_managers_proof :
  ∀ n : ℕ, 
    (n : ℚ) / managers > manager_ratio ∧ 
    n % 3 = 0 ∧
    (2 * n / 3 : ℚ) / (n / 3 : ℚ) = specialist_ratio →
    n ≤ max_non_managers :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_proof_l3277_327743


namespace NUMINAMATH_CALUDE_calc_expression_equality_simplify_fraction_equality_l3277_327740

-- Part 1
theorem calc_expression_equality : 
  (-1/2)⁻¹ + Real.sqrt 2 * Real.sqrt 6 - (π - 3)^0 + abs (Real.sqrt 3 - 2) = -1 + Real.sqrt 3 := by sorry

-- Part 2
theorem simplify_fraction_equality (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) : 
  (x^2 - 1) / (x + 1) / ((x^2 - 2*x + 1) / (x^2 - x)) = x / (x - 1) := by sorry

end NUMINAMATH_CALUDE_calc_expression_equality_simplify_fraction_equality_l3277_327740


namespace NUMINAMATH_CALUDE_correct_schedule_count_l3277_327788

/-- Represents a club with members and scheduling constraints -/
structure Club where
  totalMembers : Nat
  daysToSchedule : Nat
  membersPerDay : Nat

/-- Represents the scheduling constraints for specific members -/
structure SchedulingConstraints where
  mustBeTogetherPair : Fin 2 → Nat
  cannotBeTogether : Fin 2 → Nat

/-- Calculates the total number of possible schedules given the club and constraints -/
def totalPossibleSchedules (club : Club) (constraints : SchedulingConstraints) : Nat :=
  sorry

/-- The main theorem stating the correct number of schedules -/
theorem correct_schedule_count :
  let club := Club.mk 10 5 2
  let constraints := SchedulingConstraints.mk
    (fun i => if i.val = 0 then 0 else 1)  -- A and B (represented as 0 and 1)
    (fun i => if i.val = 0 then 2 else 3)  -- C and D (represented as 2 and 3)
  totalPossibleSchedules club constraints = 5400 := by sorry

end NUMINAMATH_CALUDE_correct_schedule_count_l3277_327788


namespace NUMINAMATH_CALUDE_bus_profit_analysis_l3277_327728

/-- Represents the daily profit of a bus company -/
def daily_profit (x : ℕ) : ℤ :=
  2 * x - 600

theorem bus_profit_analysis :
  (∀ x : ℕ, x ≥ 300 → daily_profit x ≥ 0) ∧
  (∀ x : ℕ, daily_profit x = 2 * x - 600) ∧
  (daily_profit 800 = 1000) :=
sorry

end NUMINAMATH_CALUDE_bus_profit_analysis_l3277_327728


namespace NUMINAMATH_CALUDE_function_through_point_l3277_327714

/-- Given a function f(x) = x^α that passes through the point (2, √2), prove that f(9) = 3 -/
theorem function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  f 2 = Real.sqrt 2 →
  f 9 = 3 := by
sorry

end NUMINAMATH_CALUDE_function_through_point_l3277_327714


namespace NUMINAMATH_CALUDE_polynomial_coefficient_value_l3277_327748

/-- Given a polynomial equation, prove the value of a specific coefficient -/
theorem polynomial_coefficient_value 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^2 + x^10 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₉ = -10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_value_l3277_327748


namespace NUMINAMATH_CALUDE_base4_subtraction_l3277_327775

/-- Converts a natural number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Subtracts two lists of digits in base 4 -/
def subtractBase4 (a b : List ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem base4_subtraction :
  let a := 207
  let b := 85
  let a_base4 := toBase4 a
  let b_base4 := toBase4 b
  let diff_base4 := subtractBase4 a_base4 b_base4
  fromBase4 diff_base4 = fromBase4 [1, 2, 3, 2] :=
by sorry

end NUMINAMATH_CALUDE_base4_subtraction_l3277_327775


namespace NUMINAMATH_CALUDE_min_cars_in_group_l3277_327726

/-- Represents the group of cars -/
structure CarGroup where
  total : ℕ
  withAC : ℕ
  withStripes : ℕ

/-- The conditions of the car group -/
def validCarGroup (g : CarGroup) : Prop :=
  g.total - g.withAC = 49 ∧
  g.withStripes ≥ 51 ∧
  g.withAC - g.withStripes ≤ 49

/-- The theorem stating that the minimum number of cars in a valid group is 100 -/
theorem min_cars_in_group (g : CarGroup) (h : validCarGroup g) : g.total ≥ 100 := by
  sorry

#check min_cars_in_group

end NUMINAMATH_CALUDE_min_cars_in_group_l3277_327726


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3277_327781

theorem complex_modulus_problem (z : ℂ) : 
  z * (1 + Complex.I) = 4 - 2 * Complex.I → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3277_327781


namespace NUMINAMATH_CALUDE_calvins_haircuts_l3277_327759

/-- The number of haircuts Calvin has gotten so far -/
def haircuts_gotten : ℕ := 8

/-- The number of additional haircuts Calvin needs to reach his goal -/
def haircuts_needed : ℕ := 2

/-- The percentage of progress Calvin has made towards his goal -/
def progress_percentage : ℚ := 80 / 100

theorem calvins_haircuts : 
  (haircuts_gotten : ℚ) / (haircuts_gotten + haircuts_needed) = progress_percentage := by
  sorry

end NUMINAMATH_CALUDE_calvins_haircuts_l3277_327759


namespace NUMINAMATH_CALUDE_max_value_3sin2x_l3277_327741

theorem max_value_3sin2x :
  ∀ x : ℝ, 3 * Real.sin (2 * x) ≤ 3 ∧ ∃ x₀ : ℝ, 3 * Real.sin (2 * x₀) = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_3sin2x_l3277_327741


namespace NUMINAMATH_CALUDE_factor_polynomial_l3277_327756

theorem factor_polynomial (x : ℝ) :
  x^4 - 36*x^2 + 25 = (x^2 - 6*x + 5) * (x^2 + 6*x + 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3277_327756


namespace NUMINAMATH_CALUDE_production_growth_rate_l3277_327709

theorem production_growth_rate (initial_volume : ℝ) (final_volume : ℝ) (years : ℕ) (growth_rate : ℝ) : 
  initial_volume = 1000000 → 
  final_volume = 1210000 → 
  years = 2 →
  initial_volume * (1 + growth_rate) ^ years = final_volume →
  growth_rate = 0.1 := by
sorry

end NUMINAMATH_CALUDE_production_growth_rate_l3277_327709


namespace NUMINAMATH_CALUDE_num_paths_A_to_B_l3277_327704

/-- Represents the number of red arrows from Point A -/
def num_red_arrows : ℕ := 3

/-- Represents the number of blue arrows connected to each red arrow -/
def blue_per_red : ℕ := 2

/-- Represents the number of green arrows connected to each blue arrow -/
def green_per_blue : ℕ := 2

/-- Represents the number of orange arrows connected to each green arrow -/
def orange_per_green : ℕ := 1

/-- Represents the number of ways to reach each blue arrow from a red arrow -/
def ways_to_blue : ℕ := 3

/-- Represents the number of ways to reach each green arrow from a blue arrow -/
def ways_to_green : ℕ := 4

/-- Represents the number of ways to reach each orange arrow from a green arrow -/
def ways_to_orange : ℕ := 5

/-- Theorem stating that the number of paths from A to B is 1440 -/
theorem num_paths_A_to_B : 
  num_red_arrows * blue_per_red * green_per_blue * orange_per_green * 
  ways_to_blue * ways_to_green * ways_to_orange = 1440 := by
  sorry

#check num_paths_A_to_B

end NUMINAMATH_CALUDE_num_paths_A_to_B_l3277_327704


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3277_327790

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : -4*a - b + 1 = 0) :
  (1/a + 4/b) ≥ 16 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3277_327790


namespace NUMINAMATH_CALUDE_triangle_area_with_ratio_l3277_327787

/-- Given a triangle ABC with sides a, b, c and corresponding angles A, B, C,
    if (b+c):(c+a):(a+b) = 4:5:6 and b+c = 8, then the area of triangle ABC is 15√3/4 -/
theorem triangle_area_with_ratio (a b c : ℝ) (A B C : ℝ) :
  (b + c) / (c + a) = 4 / 5 →
  (c + a) / (a + b) = 5 / 6 →
  b + c = 8 →
  (a + b + c) / 2 > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a * a = b * b + c * c - 2 * b * c * Real.cos A →
  b * b = a * a + c * c - 2 * a * c * Real.cos B →
  c * c = a * a + b * b - 2 * a * b * Real.cos C →
  (1 / 2) * b * c * Real.sin A = 15 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_ratio_l3277_327787


namespace NUMINAMATH_CALUDE_original_decimal_l3277_327776

theorem original_decimal (x : ℝ) : (1000 * x) / 100 = 12.5 → x = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_original_decimal_l3277_327776


namespace NUMINAMATH_CALUDE_task_completion_probability_l3277_327778

theorem task_completion_probability (p_task1 p_task1_not_task2 : ℝ) 
  (h1 : p_task1 = 3/8)
  (h2 : p_task1_not_task2 = 0.15)
  (h3 : 0 ≤ p_task1 ∧ p_task1 ≤ 1)
  (h4 : 0 ≤ p_task1_not_task2 ∧ p_task1_not_task2 ≤ 1) :
  ∃ p_task2 : ℝ, p_task2 = 0.6 ∧ 0 ≤ p_task2 ∧ p_task2 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_task_completion_probability_l3277_327778


namespace NUMINAMATH_CALUDE_fraction_addition_l3277_327780

theorem fraction_addition (x : ℝ) (h : x ≠ 1) : 
  (1 : ℝ) / (x - 1) + (3 : ℝ) / (x - 1) = (4 : ℝ) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3277_327780


namespace NUMINAMATH_CALUDE_roots_product_l3277_327715

theorem roots_product (b c : ℤ) : 
  (∀ s : ℝ, s^2 - 2*s - 1 = 0 → s^5 - b*s^3 - c*s^2 = 0) → 
  b * c = 348 := by
sorry

end NUMINAMATH_CALUDE_roots_product_l3277_327715


namespace NUMINAMATH_CALUDE_sales_amount_is_194_l3277_327773

/-- Represents the total sales amount from pencils in a stationery store. -/
def total_sales (eraser_price regular_price short_price : ℚ) 
                (eraser_sold regular_sold short_sold : ℕ) : ℚ :=
  eraser_price * eraser_sold + regular_price * regular_sold + short_price * short_sold

/-- Theorem stating that the total sales amount is $194 given the specific conditions. -/
theorem sales_amount_is_194 :
  total_sales 0.8 0.5 0.4 200 40 35 = 194 := by
  sorry

#eval total_sales 0.8 0.5 0.4 200 40 35

end NUMINAMATH_CALUDE_sales_amount_is_194_l3277_327773


namespace NUMINAMATH_CALUDE_min_value_expression_l3277_327753

theorem min_value_expression (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^2 / (y - 1)) + (y^2 / (z - 1)) + (z^2 / (x - 1)) ≥ 12 ∧
  ((x^2 / (y - 1)) + (y^2 / (z - 1)) + (z^2 / (x - 1)) = 12 ↔ x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3277_327753


namespace NUMINAMATH_CALUDE_double_real_value_interest_rate_l3277_327734

/-- Proves that the given formula for the annual compound interest rate 
    results in doubling the real value of an initial sum after 22 years, 
    considering inflation and taxes. -/
theorem double_real_value_interest_rate 
  (X : ℝ) -- Annual inflation rate
  (Y : ℝ) -- Annual tax rate on earned interest
  (r : ℝ) -- Annual compound interest rate
  (h_X : X > 0)
  (h_Y : 0 ≤ Y ∧ Y < 1)
  (h_r : r = ((2 * (1 + X)) ^ (1 / 22) - 1) / (1 - Y)) :
  (∀ P : ℝ, P > 0 → 
    P * (1 + r * (1 - Y))^22 / (1 + X)^22 = 2 * P) :=
sorry

end NUMINAMATH_CALUDE_double_real_value_interest_rate_l3277_327734


namespace NUMINAMATH_CALUDE_range_of_f_minus_x_l3277_327786

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x < -3 then -4
  else if x < -2 then -2
  else if x < -1 then -1
  else if x < 0 then 0
  else if x < 1 then 1
  else if x < 2 then 2
  else if x < 3 then 3
  else 4

-- Define the domain
def domain : Set ℝ := Set.Icc (-4) 4

-- State the theorem
theorem range_of_f_minus_x :
  Set.range (fun x => f x - x) ∩ (Set.Icc 0 1) = Set.Icc 0 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_minus_x_l3277_327786


namespace NUMINAMATH_CALUDE_monthly_compound_greater_than_annual_l3277_327736

def annual_rate : ℝ := 0.03

theorem monthly_compound_greater_than_annual (t : ℝ) (h : t > 0) :
  (1 + annual_rate)^t < (1 + annual_rate / 12)^(12 * t) := by
  sorry


end NUMINAMATH_CALUDE_monthly_compound_greater_than_annual_l3277_327736


namespace NUMINAMATH_CALUDE_digits_after_decimal_point_l3277_327779

theorem digits_after_decimal_point : ∃ (n : ℕ), 
  (5^8 : ℚ) / (2^5 * 10^6) = (n : ℚ) / 10^11 ∧ 
  0 < n ∧ 
  n < 10^11 := by
sorry

end NUMINAMATH_CALUDE_digits_after_decimal_point_l3277_327779


namespace NUMINAMATH_CALUDE_gcd_equality_from_division_l3277_327771

theorem gcd_equality_from_division (a b q r : ℤ) :
  b > 0 →
  0 ≤ r →
  r < b →
  a = b * q + r →
  Int.gcd a b = Int.gcd b r := by
  sorry

end NUMINAMATH_CALUDE_gcd_equality_from_division_l3277_327771


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3277_327794

theorem min_value_reciprocal_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 2 * x + y = 1) :
  (2 / x + 1 / y) ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3277_327794


namespace NUMINAMATH_CALUDE_play_dough_quantity_l3277_327760

def lego_price : ℕ := 250
def sword_price : ℕ := 120
def dough_price : ℕ := 35
def lego_quantity : ℕ := 3
def sword_quantity : ℕ := 7
def total_paid : ℕ := 1940

theorem play_dough_quantity :
  (total_paid - (lego_price * lego_quantity + sword_price * sword_quantity)) / dough_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_play_dough_quantity_l3277_327760


namespace NUMINAMATH_CALUDE_stratified_sampling_under_35_l3277_327713

/-- Calculates the number of people to be drawn from a stratum in stratified sampling -/
def stratifiedSampleSize (totalPopulation : ℕ) (stratumSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (stratumSize * sampleSize) / totalPopulation

/-- The problem statement -/
theorem stratified_sampling_under_35 :
  let totalPopulation : ℕ := 500
  let under35 : ℕ := 125
  let between35and49 : ℕ := 280
  let over50 : ℕ := 95
  let sampleSize : ℕ := 100
  stratifiedSampleSize totalPopulation under35 sampleSize = 25 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_under_35_l3277_327713


namespace NUMINAMATH_CALUDE_fraction_nonnegative_l3277_327770

theorem fraction_nonnegative (x : ℝ) (h : x ≠ -2) : x^2 / (x + 2)^2 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_fraction_nonnegative_l3277_327770


namespace NUMINAMATH_CALUDE_tan_sum_with_product_l3277_327720

theorem tan_sum_with_product (x y : Real) (h1 : x + y = π / 3) 
  (h2 : Real.sqrt 3 = Real.tan (π / 3)) : 
  Real.tan x + Real.tan y + Real.sqrt 3 * Real.tan x * Real.tan y = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_with_product_l3277_327720


namespace NUMINAMATH_CALUDE_prob_first_ace_equal_l3277_327722

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_aces : ℕ)

/-- Represents the card game -/
structure CardGame :=
  (num_players : ℕ)
  (deck : Deck)

/-- The probability of a player getting the first ace -/
def prob_first_ace (game : CardGame) (player : ℕ) : ℚ :=
  1 / game.num_players

theorem prob_first_ace_equal (game : CardGame) (player : ℕ) 
  (h1 : game.num_players = 4)
  (h2 : game.deck.total_cards = 32)
  (h3 : game.deck.num_aces = 4)
  (h4 : player ≤ game.num_players) :
  prob_first_ace game player = 1/8 :=
sorry

#check prob_first_ace_equal

end NUMINAMATH_CALUDE_prob_first_ace_equal_l3277_327722


namespace NUMINAMATH_CALUDE_train_speed_conversion_l3277_327738

theorem train_speed_conversion (speed_kmph : ℝ) (speed_ms : ℝ) : 
  speed_kmph = 216 → speed_ms = 60 → speed_kmph * (1000 / 3600) = speed_ms :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l3277_327738


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3277_327716

/-- Given a geometric sequence where the first three terms are x, 3x+3, and 6x+6,
    this theorem proves that the fourth term is -24. -/
theorem geometric_sequence_fourth_term :
  ∀ x : ℝ,
  (3*x + 3)^2 = x*(6*x + 6) →
  ∃ (a r : ℝ),
    (a = x) ∧
    (a * r = 3*x + 3) ∧
    (a * r^2 = 6*x + 6) ∧
    (a * r^3 = -24) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3277_327716


namespace NUMINAMATH_CALUDE_partition_set_exists_l3277_327758

theorem partition_set_exists (n : ℕ) (h : n ≥ 3) :
  ∃ (S : Finset ℕ), 
    (Finset.card S = 2 * n) ∧ 
    (∀ m : ℕ, 2 ≤ m ∧ m ≤ n → 
      ∃ (A : Finset ℕ), 
        A ⊆ S ∧ 
        Finset.card A = m ∧ 
        (∃ (B : Finset ℕ), B = S \ A ∧ Finset.sum A id = Finset.sum B id)) :=
by sorry

end NUMINAMATH_CALUDE_partition_set_exists_l3277_327758
