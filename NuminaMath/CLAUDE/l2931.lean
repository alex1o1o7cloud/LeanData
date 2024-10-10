import Mathlib

namespace seashell_sale_theorem_l2931_293138

/-- Calculates the total money earned from selling items collected over two days -/
def total_money (day1_items : ℕ) (price_per_item : ℚ) : ℚ :=
  let day2_items := day1_items / 2
  let total_items := day1_items + day2_items
  total_items * price_per_item

/-- Proves that collecting 30 items on day 1, half as many on day 2, 
    and selling each for $1.20 results in $54 total -/
theorem seashell_sale_theorem :
  total_money 30 (6/5) = 54 := by
  sorry

end seashell_sale_theorem_l2931_293138


namespace smallest_common_multiple_of_9_and_6_l2931_293190

theorem smallest_common_multiple_of_9_and_6 :
  ∃ n : ℕ+, (n : ℕ) % 9 = 0 ∧ (n : ℕ) % 6 = 0 ∧
  ∀ m : ℕ+, (m : ℕ) % 9 = 0 → (m : ℕ) % 6 = 0 → n ≤ m :=
by sorry

end smallest_common_multiple_of_9_and_6_l2931_293190


namespace money_division_l2931_293102

theorem money_division (a b c : ℝ) : 
  a = (1/3) * (b + c) →
  b = (2/7) * (a + c) →
  a = b + 30 →
  a + b + c = 280 :=
by sorry

end money_division_l2931_293102


namespace acute_angle_alpha_l2931_293139

theorem acute_angle_alpha (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α = 1 - Real.sqrt 3 * Real.tan (π / 18) * Real.sin α) : 
  α = π / 3.6 := by sorry

end acute_angle_alpha_l2931_293139


namespace ariana_flowers_l2931_293113

theorem ariana_flowers (total : ℕ) 
  (h1 : 2 * total = 5 * (total - 10 - 14)) -- 2/5 of flowers were roses
  (h2 : 10 ≤ total) -- 10 flowers were tulips
  (h3 : 14 ≤ total - 10) -- 14 flowers were carnations
  : total = 40 := by sorry

end ariana_flowers_l2931_293113


namespace team_wins_l2931_293177

theorem team_wins (current_percentage : ℚ) (future_wins future_games : ℕ) 
  (new_percentage : ℚ) (h1 : current_percentage = 45/100) 
  (h2 : future_wins = 6) (h3 : future_games = 8) (h4 : new_percentage = 1/2) : 
  ∃ (total_games : ℕ) (current_wins : ℕ), 
    (current_wins : ℚ) / total_games = current_percentage ∧
    ((current_wins + future_wins) : ℚ) / (total_games + future_games) = new_percentage ∧
    current_wins = 18 := by
  sorry

end team_wins_l2931_293177


namespace f_derivative_at_two_l2931_293194

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_derivative_at_two (a b : ℝ) :
  (f a b 1 = -2) →
  (∀ x, HasDerivAt (f a b) ((a * x + b) / x^2) x) →
  (HasDerivAt (f a b) 0 1) →
  (∀ x, HasDerivAt (f a b) ((-2 * x + 2) / x^2) x) →
  HasDerivAt (f a b) (-1/2) 2 := by sorry

end f_derivative_at_two_l2931_293194


namespace crackers_distribution_l2931_293163

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) : 
  total_crackers = 45 → num_friends = 15 → crackers_per_friend = total_crackers / num_friends → 
  crackers_per_friend = 3 := by
  sorry

end crackers_distribution_l2931_293163


namespace g_seven_value_l2931_293132

theorem g_seven_value (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, g (x + y) = g x + g y) 
  (h2 : g 6 = 7) : 
  g 7 = 49 / 6 := by
sorry

end g_seven_value_l2931_293132


namespace more_boys_than_girls_is_two_l2931_293192

/-- The number of more boys than girls in a field day competition -/
def more_boys_than_girls : ℕ :=
  let fourth_grade_class1_girls := 12
  let fourth_grade_class1_boys := 13
  let fourth_grade_class2_girls := 15
  let fourth_grade_class2_boys := 11
  let fifth_grade_class1_girls := 9
  let fifth_grade_class1_boys := 13
  let fifth_grade_class2_girls := 10
  let fifth_grade_class2_boys := 11

  let total_girls := fourth_grade_class1_girls + fourth_grade_class2_girls +
                     fifth_grade_class1_girls + fifth_grade_class2_girls
  let total_boys := fourth_grade_class1_boys + fourth_grade_class2_boys +
                    fifth_grade_class1_boys + fifth_grade_class2_boys

  total_boys - total_girls

theorem more_boys_than_girls_is_two :
  more_boys_than_girls = 2 := by
  sorry

end more_boys_than_girls_is_two_l2931_293192


namespace min_value_bound_l2931_293189

noncomputable section

variable (a : ℝ) (x₀ : ℝ)

def f (x : ℝ) := Real.exp x - (a * x) / (x + 1)

theorem min_value_bound (h1 : a > 0) (h2 : x₀ > -1) 
  (h3 : ∀ x > -1, f a x ≥ f a x₀) : f a x₀ ≤ 1 := by
  sorry

end

end min_value_bound_l2931_293189


namespace prob_same_length_regular_hexagon_l2931_293152

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of elements in T -/
def n : ℕ := Finset.card T

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of short diagonals in a regular hexagon -/
def num_short_diag : ℕ := 6

/-- The number of long diagonals in a regular hexagon -/
def num_long_diag : ℕ := 3

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ :=
  (num_sides * (num_sides - 1) + num_short_diag * (num_short_diag - 1) + num_long_diag * (num_long_diag - 1)) /
  (n * (n - 1))

theorem prob_same_length_regular_hexagon :
  prob_same_length = 22 / 35 := by sorry

end prob_same_length_regular_hexagon_l2931_293152


namespace line_up_permutations_l2931_293109

def number_of_people : ℕ := 5

theorem line_up_permutations :
  let youngest_not_first := number_of_people - 1
  let eldest_not_last := number_of_people - 1
  let remaining_positions := number_of_people - 2
  youngest_not_first * eldest_not_last * (remaining_positions.factorial) = 96 :=
by sorry

end line_up_permutations_l2931_293109


namespace system_solution_l2931_293145

theorem system_solution (a b c x y z T : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  x = Real.sqrt (y^2 - a^2) + Real.sqrt (z^2 - a^2) →
  y = Real.sqrt (z^2 - b^2) + Real.sqrt (x^2 - b^2) →
  z = Real.sqrt (x^2 - c^2) + Real.sqrt (y^2 - c^2) →
  1 / T^2 = 2 / (a^2 * b^2) + 2 / (b^2 * c^2) + 2 / (c^2 * a^2) - 1 / a^4 - 1 / b^4 - 1 / c^4 →
  1 / T^2 > 0 →
  x = 2 * T / a ∧ y = 2 * T / b ∧ z = 2 * T / c :=
by sorry

end system_solution_l2931_293145


namespace min_max_quadratic_form_l2931_293123

theorem min_max_quadratic_form (x y : ℝ) (h : 2 * x^2 + 3 * x * y + y^2 = 2) :
  (∀ a b : ℝ, 2 * a^2 + 3 * a * b + b^2 = 2 → 4 * a^2 + 4 * a * b + 3 * b^2 ≥ 4) ∧
  (∀ a b : ℝ, 2 * a^2 + 3 * a * b + b^2 = 2 → 4 * a^2 + 4 * a * b + 3 * b^2 ≤ 6) ∧
  (∃ a b : ℝ, 2 * a^2 + 3 * a * b + b^2 = 2 ∧ 4 * a^2 + 4 * a * b + 3 * b^2 = 4) ∧
  (∃ a b : ℝ, 2 * a^2 + 3 * a * b + b^2 = 2 ∧ 4 * a^2 + 4 * a * b + 3 * b^2 = 6) :=
by sorry

end min_max_quadratic_form_l2931_293123


namespace circle_range_theorem_l2931_293133

/-- The range of 'a' for a circle (x-a)^2 + (y-a)^2 = 8 with a point at distance √2 from origin -/
theorem circle_range_theorem (a : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y - a)^2 = 8 ∧ x^2 + y^2 = 2) ↔ 
  (a ∈ Set.Icc (-3) (-1) ∪ Set.Icc 1 3) :=
sorry

end circle_range_theorem_l2931_293133


namespace power_of_eight_sum_equals_power_of_two_l2931_293143

theorem power_of_eight_sum_equals_power_of_two : ∃ x : ℕ, 8^4 + 8^4 + 8^4 = 2^x ∧ x = 13 := by
  sorry

end power_of_eight_sum_equals_power_of_two_l2931_293143


namespace married_women_fraction_l2931_293156

theorem married_women_fraction (total_men : ℕ) (total_women : ℕ) (single_men : ℕ) :
  (single_men : ℚ) / total_men = 3 / 7 →
  total_women = total_men - single_men →
  (total_women : ℚ) / (total_men + total_women) = 4 / 11 :=
by sorry

end married_women_fraction_l2931_293156


namespace triangle_side_length_l2931_293155

theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  a = 3 →
  b - c = 2 →
  Real.cos B = -1/2 →
  b = 7 := by
  sorry

end triangle_side_length_l2931_293155


namespace parking_lot_problem_l2931_293188

/-- Represents the number of wheels for each vehicle type -/
structure VehicleWheels where
  car : Nat
  bicycle : Nat
  motorcycle : Nat

/-- Represents the count of each vehicle type in the parking lot -/
structure VehicleCount where
  cars : Nat
  bicycles : Nat
  motorcycles : Nat

/-- The theorem stating the relationship between the number of cars and motorcycles -/
theorem parking_lot_problem (wheels : VehicleWheels) (count : VehicleCount) :
  wheels.car = 4 →
  wheels.bicycle = 2 →
  wheels.motorcycle = 2 →
  count.bicycles = 2 * count.motorcycles →
  wheels.car * count.cars + wheels.bicycle * count.bicycles + wheels.motorcycle * count.motorcycles = 196 →
  count.cars = (98 - 3 * count.motorcycles) / 2 := by
  sorry

end parking_lot_problem_l2931_293188


namespace min_value_reciprocal_sum_l2931_293193

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 3 / b₀ = 16 :=
by sorry

end min_value_reciprocal_sum_l2931_293193


namespace quadratic_inequality_solution_set_l2931_293126

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, (3 + 5 * x - 2 * x^2 > 0) ↔ (-1/2 < x ∧ x < 3) :=
by sorry

end quadratic_inequality_solution_set_l2931_293126


namespace sixth_quiz_score_achieves_target_mean_l2931_293107

def quiz_scores : List ℕ := [75, 80, 85, 90, 100]
def target_mean : ℕ := 95
def num_quizzes : ℕ := 6
def sixth_score : ℕ := 140

theorem sixth_quiz_score_achieves_target_mean :
  (List.sum quiz_scores + sixth_score) / num_quizzes = target_mean := by
  sorry

end sixth_quiz_score_achieves_target_mean_l2931_293107


namespace guaranteed_win_for_given_odds_l2931_293111

/-- Represents the odds for a team as a pair of natural numbers -/
def Odds := Nat × Nat

/-- Calculates the return multiplier for given odds -/
def returnMultiplier (odds : Odds) : Rat :=
  1 + odds.2 / odds.1

/-- Represents the odds for all teams in the tournament -/
structure TournamentOdds where
  team1 : Odds
  team2 : Odds
  team3 : Odds
  team4 : Odds

/-- Checks if a betting strategy exists that guarantees a win -/
def guaranteedWinExists (odds : TournamentOdds) : Prop :=
  ∃ (bet1 bet2 bet3 bet4 : Rat),
    bet1 > 0 ∧ bet2 > 0 ∧ bet3 > 0 ∧ bet4 > 0 ∧
    bet1 + bet2 + bet3 + bet4 = 1 ∧
    bet1 * returnMultiplier odds.team1 > 1 ∧
    bet2 * returnMultiplier odds.team2 > 1 ∧
    bet3 * returnMultiplier odds.team3 > 1 ∧
    bet4 * returnMultiplier odds.team4 > 1

/-- The main theorem stating that a guaranteed win exists for the given odds -/
theorem guaranteed_win_for_given_odds :
  let odds : TournamentOdds := {
    team1 := (1, 5)
    team2 := (1, 1)
    team3 := (1, 8)
    team4 := (1, 7)
  }
  guaranteedWinExists odds := by sorry

end guaranteed_win_for_given_odds_l2931_293111


namespace complex_modulus_problem_l2931_293130

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the problem statement
theorem complex_modulus_problem (z : ℂ) (h : (1 + i) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l2931_293130


namespace expression_evaluation_l2931_293110

theorem expression_evaluation : 2 - (-3) * 2 - 4 - (-5) * 2 - 6 = 8 := by
  sorry

end expression_evaluation_l2931_293110


namespace progression_ratio_l2931_293162

/-- Given an arithmetic progression and a geometric progression with shared elements,
    prove that the ratio of the difference of middle terms of the arithmetic progression
    to the middle term of the geometric progression is either 1/2 or -1/2. -/
theorem progression_ratio (a₁ a₂ b : ℝ) : 
  ((-2 : ℝ) - a₁ = a₁ - a₂ ∧ a₂ - (-8) = a₁ - a₂) →  -- arithmetic progression condition
  (b^2 = (-2) * (-8)) →                              -- geometric progression condition
  (a₂ - a₁) / b = 1/2 ∨ (a₂ - a₁) / b = -1/2 := by
  sorry


end progression_ratio_l2931_293162


namespace intersection_eq_interval_l2931_293165

open Set

-- Define the sets M and N
def M : Set ℝ := {x | 2 - x > 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Define the interval [1, 2)
def interval_1_2 : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- Theorem statement
theorem intersection_eq_interval : M ∩ N = interval_1_2 := by
  sorry

end intersection_eq_interval_l2931_293165


namespace integer_solutions_count_l2931_293176

theorem integer_solutions_count : 
  ∃ (S : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ S ↔ x^2 - y^2 = 1988) ∧ 
    Finset.card S = 8 := by
  sorry

end integer_solutions_count_l2931_293176


namespace two_squares_share_vertices_l2931_293170

/-- A square in a plane. -/
structure Square where
  vertices : Fin 4 → ℝ × ℝ

/-- An isosceles right triangle in a plane. -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a square shares two vertices with a triangle. -/
def SharesTwoVertices (s : Square) (t : IsoscelesRightTriangle) : Prop :=
  ∃ (i j : Fin 4) (v w : ℝ × ℝ), i ≠ j ∧
    s.vertices i = v ∧ s.vertices j = w ∧
    (v = t.A ∨ v = t.B ∨ v = t.C) ∧
    (w = t.A ∨ w = t.B ∨ w = t.C)

/-- The main theorem stating that there are exactly two squares sharing two vertices
    with an isosceles right triangle. -/
theorem two_squares_share_vertices (t : IsoscelesRightTriangle) :
  ∃! (n : ℕ), ∃ (squares : Fin n → Square),
    (∀ i, SharesTwoVertices (squares i) t) ∧
    (∀ s, SharesTwoVertices s t → ∃ i, s = squares i) ∧
    n = 2 :=
sorry

end two_squares_share_vertices_l2931_293170


namespace quadratic_radical_sum_l2931_293173

theorem quadratic_radical_sum (m n : ℕ) : 
  (∃ k : ℕ, (m - 1 : ℕ) = 2 ∧ 7^k = 7) ∧ 
  (∃ l : ℕ, 4*n - 1 = 7^l) ∧
  (m - 1 : ℕ) = 2 → 
  m + n = 5 := by sorry

end quadratic_radical_sum_l2931_293173


namespace average_female_students_l2931_293159

theorem average_female_students (class_8A class_8B class_8C class_8D class_8E : ℕ) 
  (h1 : class_8A = 10)
  (h2 : class_8B = 14)
  (h3 : class_8C = 7)
  (h4 : class_8D = 9)
  (h5 : class_8E = 13) : 
  (class_8A + class_8B + class_8C + class_8D + class_8E : ℚ) / 5 = 10.6 := by
  sorry

end average_female_students_l2931_293159


namespace production_time_calculation_l2931_293128

/-- Given that 5 machines can produce 20 units in 10 hours, 
    prove that 10 machines will take 25 hours to produce 100 units. -/
theorem production_time_calculation 
  (machines_initial : ℕ) 
  (units_initial : ℕ) 
  (hours_initial : ℕ) 
  (machines_final : ℕ) 
  (units_final : ℕ) 
  (h1 : machines_initial = 5) 
  (h2 : units_initial = 20) 
  (h3 : hours_initial = 10) 
  (h4 : machines_final = 10) 
  (h5 : units_final = 100) : 
  (units_final : ℚ) * machines_initial * hours_initial / 
  (units_initial * machines_final) = 25 := by
  sorry


end production_time_calculation_l2931_293128


namespace book_reading_time_l2931_293186

/-- Given a book with a certain number of pages and initial reading pace,
    calculate the number of days needed to finish the book with an increased reading pace. -/
theorem book_reading_time (total_pages : ℕ) (initial_pages_per_day : ℕ) (initial_days : ℕ) 
    (increase : ℕ) (h1 : total_pages = initial_pages_per_day * initial_days)
    (h2 : initial_pages_per_day = 15) (h3 : initial_days = 24) (h4 : increase = 3) : 
    total_pages / (initial_pages_per_day + increase) = 20 := by
  sorry

end book_reading_time_l2931_293186


namespace henry_shells_l2931_293174

theorem henry_shells (broken_shells : ℕ) (perfect_non_spiral : ℕ) (spiral_difference : ℕ) :
  broken_shells = 52 →
  perfect_non_spiral = 12 →
  spiral_difference = 21 →
  ∃ (total_perfect : ℕ),
    total_perfect = (broken_shells / 2 - spiral_difference) + perfect_non_spiral ∧
    total_perfect = 17 := by
  sorry

end henry_shells_l2931_293174


namespace chocolates_remaining_theorem_l2931_293160

/-- Number of chocolates remaining after 4 days -/
def chocolates_remaining (total : ℕ) (day1 : ℕ) : ℕ :=
  let day2 := 2 * day1 - 3
  let day3 := day1 - 2
  let day4 := day3 - 1
  total - (day1 + day2 + day3 + day4)

/-- Theorem stating that 12 chocolates remain uneaten after 4 days -/
theorem chocolates_remaining_theorem :
  chocolates_remaining 24 4 = 12 := by
  sorry

#eval chocolates_remaining 24 4

end chocolates_remaining_theorem_l2931_293160


namespace units_digit_of_7_pow_2050_l2931_293161

-- Define the function that returns the units digit of 7^n
def units_digit_of_7_pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case should never occur

theorem units_digit_of_7_pow_2050 :
  units_digit_of_7_pow 2050 = 9 := by
  sorry

end units_digit_of_7_pow_2050_l2931_293161


namespace smallest_integer_with_remainders_l2931_293100

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 2 = 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 10 = 9) ∧ 
  (∀ m : ℕ, m > 0 → m % 2 = 1 → m % 3 = 2 → m % 10 = 9 → m ≥ n) ∧
  (n = 59) := by
  sorry

end smallest_integer_with_remainders_l2931_293100


namespace first_machine_copies_per_minute_l2931_293131

/-- Given two copy machines working together, prove that the first machine makes 25 copies per minute. -/
theorem first_machine_copies_per_minute :
  ∀ (x : ℝ),
  (∃ (rate₁ : ℝ), rate₁ = x) →  -- First machine works at a constant rate x
  (∃ (rate₂ : ℝ), rate₂ = 55) →  -- Second machine works at 55 copies per minute
  (x + 55) * 30 = 2400 →  -- Together they make 2400 copies in 30 minutes
  x = 25 := by
sorry

end first_machine_copies_per_minute_l2931_293131


namespace sum_of_powers_divisibility_l2931_293175

theorem sum_of_powers_divisibility 
  (a₁ a₂ a₃ a₄ : ℤ) 
  (h : a₁^3 + a₂^3 + a₃^3 + a₄^3 = 0) :
  ∀ k : ℕ, k % 2 = 1 → (6 : ℤ) ∣ (a₁^k + a₂^k + a₃^k + a₄^k) := by
  sorry

end sum_of_powers_divisibility_l2931_293175


namespace cookies_left_after_sales_l2931_293171

/-- Calculates the number of cookies left after sales throughout the day -/
theorem cookies_left_after_sales (initial : ℕ) (morning_dozens : ℕ) (lunch : ℕ) (afternoon : ℕ) :
  initial = 120 →
  morning_dozens = 3 →
  lunch = 57 →
  afternoon = 16 →
  initial - (morning_dozens * 12 + lunch + afternoon) = 11 :=
by
  sorry

end cookies_left_after_sales_l2931_293171


namespace sophia_next_test_score_l2931_293135

def current_scores : List ℕ := [95, 85, 75, 65, 95]
def desired_increase : ℕ := 5

def minimum_required_score (scores : List ℕ) (increase : ℕ) : ℕ :=
  let current_sum := scores.sum
  let current_count := scores.length
  let current_average := current_sum / current_count
  let target_average := current_average + increase
  let total_count := current_count + 1
  target_average * total_count - current_sum

theorem sophia_next_test_score :
  minimum_required_score current_scores desired_increase = 113 := by
  sorry

end sophia_next_test_score_l2931_293135


namespace problem_statement_l2931_293182

theorem problem_statement (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : 1 / a^3 = 512 / b^3 ∧ 1 / a^3 = 125 / c^3 ∧ 1 / a^3 = d / (a + b + c)^3) : 
  d = 2744 := by
sorry

end problem_statement_l2931_293182


namespace travelers_checks_theorem_l2931_293141

/-- Represents the number of travelers checks of each denomination -/
structure TravelersChecks where
  fifty : ℕ
  hundred : ℕ

/-- The problem setup for the travelers checks -/
def travelersProblem (tc : TravelersChecks) : Prop :=
  tc.fifty + tc.hundred = 30 ∧
  50 * tc.fifty + 100 * tc.hundred = 1800

/-- The result of spending some $50 checks -/
def spendFiftyChecks (tc : TravelersChecks) (spent : ℕ) : TravelersChecks :=
  { fifty := tc.fifty - spent, hundred := tc.hundred }

/-- Calculate the average value of the remaining checks -/
def averageValue (tc : TravelersChecks) : ℚ :=
  (50 * tc.fifty + 100 * tc.hundred) / (tc.fifty + tc.hundred)

/-- The main theorem to prove -/
theorem travelers_checks_theorem (tc : TravelersChecks) :
  travelersProblem tc →
  averageValue (spendFiftyChecks tc 15) = 70 := by
  sorry


end travelers_checks_theorem_l2931_293141


namespace altitude_equation_median_equation_l2931_293103

/-- Triangle ABC with vertices A(-2,-1), B(2,1), and C(1,3) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The specific triangle given in the problem -/
def given_triangle : Triangle :=
  { A := (-2, -1),
    B := (2, 1),
    C := (1, 3) }

/-- Equation of a line in point-slope form -/
structure PointSlopeLine :=
  (m : ℝ)  -- slope
  (x₀ : ℝ) -- x-coordinate of point
  (y₀ : ℝ) -- y-coordinate of point

/-- Equation of a line in general form -/
structure GeneralLine :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- The altitude from side AB of the triangle -/
def altitude (t : Triangle) : PointSlopeLine :=
  { m := -2,
    x₀ := 1,
    y₀ := 3 }

/-- The median from side AB of the triangle -/
def median (t : Triangle) : GeneralLine :=
  { a := 3,
    b := -1,
    c := 0 }

theorem altitude_equation (t : Triangle) :
  t = given_triangle →
  altitude t = { m := -2, x₀ := 1, y₀ := 3 } :=
by sorry

theorem median_equation (t : Triangle) :
  t = given_triangle →
  median t = { a := 3, b := -1, c := 0 } :=
by sorry

end altitude_equation_median_equation_l2931_293103


namespace line_equation_l2931_293178

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop :=
  m.x = (a.x + b.x) / 2 ∧ m.y = (a.y + b.y) / 2

/-- The main theorem -/
theorem line_equation (l : Line) (m a b : Point) : 
  pointOnLine m l → 
  m.x = -1 ∧ m.y = 2 →
  a.y = 0 →
  b.x = 0 →
  isMidpoint m a b →
  l.a = 2 ∧ l.b = -1 ∧ l.c = 4 := by
  sorry

end line_equation_l2931_293178


namespace parabola_line_intersection_l2931_293137

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line with 60° inclination passing through the focus
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

-- Define a point in the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem parabola_line_intersection :
  ∀ (x y : ℝ),
  parabola x y →
  line x y →
  first_quadrant x y →
  Real.sqrt ((x - 1)^2 + y^2) = 4 :=
sorry

end parabola_line_intersection_l2931_293137


namespace identity_condition_l2931_293142

theorem identity_condition (a b c : ℝ) : 
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) ↔ 
  ((a = 0 ∧ b = 0 ∧ c = 1) ∨ 
   (a = 0 ∧ b = 0 ∧ c = -1) ∨ 
   (a = 0 ∧ b = 1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
   (a = 1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = -1 ∧ b = 0 ∧ c = 0)) :=
by sorry

end identity_condition_l2931_293142


namespace linda_savings_l2931_293116

theorem linda_savings (savings : ℝ) : 
  savings > 0 →
  savings * (3/4) + savings * (1/8) + 250 = savings * (7/8) →
  250 = (savings * (1/8)) * 0.9 →
  savings = 2222.24 := by
sorry

end linda_savings_l2931_293116


namespace valid_arrangements_count_l2931_293127

/-- The number of cards --/
def n : ℕ := 7

/-- The special card that must be at the beginning or end --/
def special_card : ℕ := 7

/-- The number of cards that will remain after removal --/
def remaining_cards : ℕ := 5

/-- The number of possible positions for the special card --/
def special_card_positions : ℕ := 2

/-- The number of ways to choose a card to remove from the non-special cards --/
def removal_choices : ℕ := n - 1

/-- The number of permutations of the remaining cards --/
def remaining_permutations : ℕ := remaining_cards.factorial

/-- The number of possible orderings (ascending or descending) --/
def possible_orderings : ℕ := 2

/-- The total number of valid arrangements --/
def valid_arrangements : ℕ := 
  special_card_positions * removal_choices * remaining_permutations * possible_orderings

theorem valid_arrangements_count : valid_arrangements = 2880 := by
  sorry

end valid_arrangements_count_l2931_293127


namespace quadratic_roots_sum_reciprocals_l2931_293184

theorem quadratic_roots_sum_reciprocals (a b : ℝ) : 
  (a^2 + 8*a + 4 = 0) → (b^2 + 8*b + 4 = 0) → (a / b + b / a = 14) :=
by sorry

end quadratic_roots_sum_reciprocals_l2931_293184


namespace factor_expression_l2931_293154

theorem factor_expression (a : ℝ) : 189 * a^2 + 27 * a - 54 = 9 * (7 * a - 3) * (3 * a + 2) := by
  sorry

end factor_expression_l2931_293154


namespace consecutive_sum_39_l2931_293181

theorem consecutive_sum_39 (n m : ℕ) : 
  m = n + 1 → n + m = 39 → m = 20 := by
  sorry

end consecutive_sum_39_l2931_293181


namespace total_students_in_high_school_l2931_293124

-- Define the number of students in each grade
def freshman_students : ℕ := sorry
def sophomore_students : ℕ := sorry
def senior_students : ℕ := 1200

-- Define the sample sizes
def freshman_sample : ℕ := 75
def sophomore_sample : ℕ := 60
def senior_sample : ℕ := 50

-- Define the total sample size
def total_sample : ℕ := 185

-- Theorem statement
theorem total_students_in_high_school :
  freshman_students + sophomore_students + senior_students = 4440 :=
by
  -- Assuming the stratified sampling method ensures equal ratios
  have h1 : (freshman_sample : ℚ) / freshman_students = (senior_sample : ℚ) / senior_students := sorry
  have h2 : (sophomore_sample : ℚ) / sophomore_students = (senior_sample : ℚ) / senior_students := sorry
  
  -- The total sample size is the sum of individual sample sizes
  have h3 : freshman_sample + sophomore_sample + senior_sample = total_sample := sorry

  sorry -- Complete the proof

end total_students_in_high_school_l2931_293124


namespace return_trip_speed_l2931_293115

/-- Given a round trip between two cities, prove the speed of the return trip -/
theorem return_trip_speed 
  (distance : ℝ) 
  (outbound_speed : ℝ) 
  (average_speed : ℝ) :
  distance = 150 →
  outbound_speed = 75 →
  average_speed = 50 →
  (2 * distance) / (distance / outbound_speed + distance / ((2 * distance) / (2 * average_speed) - distance / outbound_speed)) = average_speed →
  (2 * distance) / (2 * average_speed) - distance / outbound_speed = distance / 37.5 :=
by sorry

end return_trip_speed_l2931_293115


namespace church_female_adults_l2931_293108

/-- Calculates the number of female adults in a church given the total number of people,
    number of children, and number of male adults. -/
def female_adults (total : ℕ) (children : ℕ) (male_adults : ℕ) : ℕ :=
  total - (children + male_adults)

/-- Theorem stating that the number of female adults in the church is 60. -/
theorem church_female_adults :
  female_adults 200 80 60 = 60 := by
  sorry

end church_female_adults_l2931_293108


namespace anna_age_when_married_l2931_293125

/-- Represents the ages and marriage duration of Josh and Anna -/
structure Couple where
  josh_age_at_marriage : ℕ
  years_married : ℕ
  combined_age_factor : ℕ

/-- Calculates Anna's age when they got married -/
def anna_age_at_marriage (c : Couple) : ℕ :=
  c.combined_age_factor * c.josh_age_at_marriage - (c.josh_age_at_marriage + c.years_married)

/-- Theorem stating Anna's age when they got married -/
theorem anna_age_when_married (c : Couple) 
    (h1 : c.josh_age_at_marriage = 22)
    (h2 : c.years_married = 30)
    (h3 : c.combined_age_factor = 5) :
  anna_age_at_marriage c = 28 := by
  sorry

#eval anna_age_at_marriage ⟨22, 30, 5⟩

end anna_age_when_married_l2931_293125


namespace problem_solution_l2931_293148

def A : Set ℝ := {1, 2}

def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*a*x + a = 0}

def C (m : ℝ) : Set ℝ := {x : ℝ | x^2 - m*x + 3 > 0}

theorem problem_solution :
  (∀ a : ℝ, (∀ x ∈ B a, x ∈ A) ↔ a ∈ Set.Ioo 0 1) ∧
  (∀ m : ℝ, (A ⊆ C m) ↔ m ∈ Set.Iic (7/2)) := by sorry

end problem_solution_l2931_293148


namespace final_ethanol_percentage_l2931_293149

/-- Calculates the final ethanol percentage in a fuel mixture after adding pure ethanol -/
theorem final_ethanol_percentage
  (initial_volume : ℝ)
  (initial_ethanol_percentage : ℝ)
  (added_ethanol : ℝ)
  (h1 : initial_volume = 27)
  (h2 : initial_ethanol_percentage = 0.05)
  (h3 : added_ethanol = 1.5)
  : (initial_volume * initial_ethanol_percentage + added_ethanol) / (initial_volume + added_ethanol) = 0.1 := by
  sorry

#check final_ethanol_percentage

end final_ethanol_percentage_l2931_293149


namespace expected_value_of_twelve_sided_die_l2931_293158

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of rolling the die -/
def expected_value : ℚ :=
  (Finset.sum twelve_sided_die (fun i => i + 1)) / 12

/-- Theorem: The expected value of rolling a fair 12-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_of_twelve_sided_die :
  expected_value = 13/2 := by sorry

end expected_value_of_twelve_sided_die_l2931_293158


namespace staircase_climbing_l2931_293197

/-- Number of ways to ascend n steps by jumping 1 or 2 steps at a time -/
def ascend (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n+2 => ascend (n+1) + ascend n

/-- Number of ways to descend n steps with option to skip steps -/
def descend (n : ℕ) : ℕ := 2^(n-1)

theorem staircase_climbing :
  (ascend 10 = 89) ∧ (descend 10 = 512) := by
  sorry


end staircase_climbing_l2931_293197


namespace luke_paint_area_l2931_293112

/-- Calculates the area to be painted on a wall with a bookshelf -/
def area_to_paint (wall_height wall_length bookshelf_width bookshelf_height : ℝ) : ℝ :=
  wall_height * wall_length - bookshelf_width * bookshelf_height

/-- Proves that Luke needs to paint 135 square feet -/
theorem luke_paint_area :
  area_to_paint 10 15 3 5 = 135 := by
  sorry

end luke_paint_area_l2931_293112


namespace sandy_marbles_multiple_l2931_293134

def melanie_marbles : ℕ := 84
def sandy_dozens : ℕ := 56

def marbles_in_dozen : ℕ := 12

theorem sandy_marbles_multiple : 
  (sandy_dozens * marbles_in_dozen) / melanie_marbles = 8 := by
  sorry

end sandy_marbles_multiple_l2931_293134


namespace simplification_proof_equation_solution_proof_l2931_293144

-- Problem 1: Simplification
theorem simplification_proof (a : ℝ) (ha : a ≠ 0 ∧ a ≠ 1) :
  (a - 1/a) / ((a^2 - 2*a + 1) / a) = (a + 1) / (a - 1) := by sorry

-- Problem 2: Equation Solving
theorem equation_solution_proof :
  ∀ x : ℝ, x = -1 ↔ 2*x/(x-2) = 1 - 1/(2-x) := by sorry

end simplification_proof_equation_solution_proof_l2931_293144


namespace sixth_term_term_1994_l2931_293195

-- Define the sequence
def a (n : ℕ+) : ℕ := n * (n + 1)

-- Theorem for the 6th term
theorem sixth_term : a 6 = 42 := by sorry

-- Theorem for the 1994th term
theorem term_1994 : a 1994 = 3978030 := by sorry

end sixth_term_term_1994_l2931_293195


namespace tan_product_pi_eighths_l2931_293118

theorem tan_product_pi_eighths : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = 2 * Real.sqrt 2 := by
  sorry

end tan_product_pi_eighths_l2931_293118


namespace polynomial_differential_equation_l2931_293146

/-- A polynomial of the form a(x + b)^n satisfies (p'(x))^2 = c * p(x) * p''(x) for some constant c -/
theorem polynomial_differential_equation (a b : ℝ) (n : ℕ) (hn : n > 1) (ha : a ≠ 0) :
  ∃ c : ℝ, ∀ x : ℝ,
    let p := fun x => a * (x + b) ^ n
    let p' := fun x => n * a * (x + b) ^ (n - 1)
    let p'' := fun x => n * (n - 1) * a * (x + b) ^ (n - 2)
    (p' x) ^ 2 = c * (p x) * (p'' x) := by
  sorry

end polynomial_differential_equation_l2931_293146


namespace unique_three_digit_number_with_digit_property_l2931_293119

/-- Calculate the total number of digits used to write all integers from 1 to n -/
def totalDigits (n : ℕ) : ℕ :=
  if n < 10 then n
  else if n < 100 then 9 + 2 * (n - 9)
  else 189 + 3 * (n - 99)

/-- The property that a number, when doubled, equals the total digits required to write all numbers up to itself -/
def hasDigitProperty (n : ℕ) : Prop :=
  2 * n = totalDigits n

theorem unique_three_digit_number_with_digit_property :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ hasDigitProperty n ∧ n = 108 :=
by sorry

end unique_three_digit_number_with_digit_property_l2931_293119


namespace ones_digit_largest_power_of_3_dividing_27_factorial_l2931_293129

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largestPowerOf3DividingFactorial (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1) / 3) 0

def onesDigit (n : ℕ) : ℕ := n % 10

theorem ones_digit_largest_power_of_3_dividing_27_factorial :
  onesDigit (3^(largestPowerOf3DividingFactorial 27)) = 3 := by
  sorry

end ones_digit_largest_power_of_3_dividing_27_factorial_l2931_293129


namespace cross_spectral_density_symmetry_l2931_293140

/-- Cross-spectral density of two random functions -/
noncomputable def cross_spectral_density (X Y : ℝ → ℂ) (ω : ℝ) : ℂ := sorry

/-- Stationarity property for a random function -/
def stationary (X : ℝ → ℂ) : Prop := sorry

/-- Joint stationarity property for two random functions -/
def jointly_stationary (X Y : ℝ → ℂ) : Prop := sorry

/-- Theorem: For stationary and jointly stationary random functions, 
    the cross-spectral densities satisfy s_xy(-ω) = s_yx(ω) -/
theorem cross_spectral_density_symmetry 
  (X Y : ℝ → ℂ) (ω : ℝ) 
  (h1 : stationary X) (h2 : stationary Y) (h3 : jointly_stationary X Y) : 
  cross_spectral_density X Y (-ω) = cross_spectral_density Y X ω := by
  sorry

end cross_spectral_density_symmetry_l2931_293140


namespace tan_alpha_value_l2931_293150

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (Real.sin α + Real.cos α) = -1) : 
  Real.tan α = 1/2 := by
  sorry

end tan_alpha_value_l2931_293150


namespace area_of_composite_rectangle_l2931_293104

/-- The area of a rectangle formed by four identical smaller rectangles --/
theorem area_of_composite_rectangle (short_side : ℝ) : 
  short_side = 7 →
  (2 * short_side) * (2 * short_side) = 392 := by
  sorry

end area_of_composite_rectangle_l2931_293104


namespace orange_calories_l2931_293106

theorem orange_calories
  (num_oranges : ℕ)
  (pieces_per_orange : ℕ)
  (num_people : ℕ)
  (calories_per_person : ℕ)
  (h1 : num_oranges = 5)
  (h2 : pieces_per_orange = 8)
  (h3 : num_people = 4)
  (h4 : calories_per_person = 100)
  : calories_per_person = num_oranges * calories_per_person / num_oranges :=
by
  sorry

end orange_calories_l2931_293106


namespace cloth_selling_price_l2931_293114

/-- Calculates the total selling price of cloth given the length, profit per meter, and cost price per meter. -/
def total_selling_price (length : ℕ) (profit_per_meter : ℕ) (cost_per_meter : ℕ) : ℕ :=
  length * (profit_per_meter + cost_per_meter)

/-- Proves that the total selling price of 45 meters of cloth is 4500 rupees,
    given a profit of 12 rupees per meter and a cost price of 88 rupees per meter. -/
theorem cloth_selling_price :
  total_selling_price 45 12 88 = 4500 := by
  sorry

end cloth_selling_price_l2931_293114


namespace baker_initial_cakes_l2931_293151

theorem baker_initial_cakes (total_cakes : ℕ) (extra_cakes : ℕ) (initial_cakes : ℕ) : 
  total_cakes = 87 → extra_cakes = 9 → initial_cakes = total_cakes - extra_cakes → initial_cakes = 78 := by
  sorry

end baker_initial_cakes_l2931_293151


namespace team_point_difference_l2931_293185

/-- The difference in points between two teams -/
def pointDifference (beth_score jan_score judy_score angel_score : ℕ) : ℕ :=
  (beth_score + jan_score) - (judy_score + angel_score)

/-- Theorem stating the point difference between the two teams -/
theorem team_point_difference :
  pointDifference 12 10 8 11 = 3 := by
  sorry

end team_point_difference_l2931_293185


namespace inequality_proof_l2931_293179

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (a*b + 1) / (a + b)^2 + (b*c + 1) / (b + c)^2 + (c*a + 1) / (c + a)^2 ≥ 3 := by
  sorry

end inequality_proof_l2931_293179


namespace shekar_biology_score_l2931_293164

/-- Represents a student's scores in five subjects -/
structure StudentScores where
  mathematics : ℕ
  science : ℕ
  socialStudies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average score for a student -/
def averageScore (scores : StudentScores) : ℚ :=
  (scores.mathematics + scores.science + scores.socialStudies + scores.english + scores.biology) / 5

/-- Theorem: Given Shekar's scores in four subjects and the average, his Biology score must be 75 -/
theorem shekar_biology_score :
  ∀ (scores : StudentScores),
    scores.mathematics = 76 →
    scores.science = 65 →
    scores.socialStudies = 82 →
    scores.english = 67 →
    averageScore scores = 73 →
    scores.biology = 75 := by
  sorry

end shekar_biology_score_l2931_293164


namespace sum_with_reverse_has_even_digit_l2931_293183

/-- A type representing a 17-digit number -/
def Digit17 := Fin 10 → Fin 10

/-- Reverses a 17-digit number -/
def reverse (n : Digit17) : Digit17 :=
  fun i => n (16 - i)

/-- Adds two 17-digit numbers -/
def add (a b : Digit17) : Digit17 :=
  sorry

/-- Checks if a number has at least one even digit -/
def hasEvenDigit (n : Digit17) : Prop :=
  ∃ i, (n i).val % 2 = 0

/-- Main theorem: For any 17-digit number, when added to its reverse, 
    the resulting sum contains at least one even digit -/
theorem sum_with_reverse_has_even_digit (n : Digit17) : 
  hasEvenDigit (add n (reverse n)) := by
  sorry

end sum_with_reverse_has_even_digit_l2931_293183


namespace sum_and_fraction_relation_l2931_293122

theorem sum_and_fraction_relation (a b : ℝ) 
  (sum_eq : a + b = 507)
  (frac_eq : (a - b) / b = 1 / 7) : 
  b - a = -34.428571 := by
  sorry

end sum_and_fraction_relation_l2931_293122


namespace bride_age_at_silver_anniversary_l2931_293199

theorem bride_age_at_silver_anniversary (age_difference : ℕ) (combined_age : ℕ) : 
  age_difference = 19 → combined_age = 185 → ∃ (bride_age groom_age : ℕ), 
    bride_age = groom_age + age_difference ∧ 
    bride_age + groom_age = combined_age ∧ 
    bride_age = 102 := by
  sorry

end bride_age_at_silver_anniversary_l2931_293199


namespace beef_weight_problem_l2931_293167

theorem beef_weight_problem (initial_weight : ℝ) : 
  initial_weight > 0 →
  initial_weight * (1 - 0.3) * (1 - 0.2) * (1 - 0.5) = 315 →
  initial_weight = 1125 := by
sorry

end beef_weight_problem_l2931_293167


namespace logarithm_expression_equality_algebraic_expression_equality_l2931_293101

-- Part 1
theorem logarithm_expression_equality : 
  Real.log 5 * Real.log 20 - Real.log 2 * Real.log 50 - Real.log 25 = -1 := by sorry

-- Part 2
theorem algebraic_expression_equality (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  (2 * a^(2/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = 4 * a := by sorry

end logarithm_expression_equality_algebraic_expression_equality_l2931_293101


namespace original_cube_volume_l2931_293121

theorem original_cube_volume (s : ℝ) (h : (2 * s) ^ 3 = 1728) : s ^ 3 = 216 := by
  sorry

#check original_cube_volume

end original_cube_volume_l2931_293121


namespace min_value_of_function_l2931_293196

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  4 * x + 9 / x^2 ≥ 3 * (36 : ℝ)^(1/3) ∧ 
  ∃ y > 0, 4 * y + 9 / y^2 = 3 * (36 : ℝ)^(1/3) :=
by sorry

end min_value_of_function_l2931_293196


namespace right_triangle_area_l2931_293187

theorem right_triangle_area (a b : ℝ) (h1 : a = 3) (h2 : b = 5) : 
  (1/2) * a * b = 7.5 := by
  sorry

end right_triangle_area_l2931_293187


namespace calculation_proof_l2931_293147

theorem calculation_proof : (1 / Real.sqrt 3) - (1 / 4)⁻¹ + 4 * Real.sin (60 * π / 180) + |1 - Real.sqrt 3| = (10 / 3) * Real.sqrt 3 - 5 := by
  sorry

end calculation_proof_l2931_293147


namespace matts_future_age_l2931_293157

theorem matts_future_age (bush_age : ℕ) (age_difference : ℕ) (years_from_now : ℕ) :
  bush_age = 12 →
  age_difference = 3 →
  years_from_now = 10 →
  bush_age + age_difference + years_from_now = 25 :=
by
  sorry

end matts_future_age_l2931_293157


namespace equation_three_holds_l2931_293168

theorem equation_three_holds (square : ℚ) (h : square = 3 + 1/20) : 
  ((6.5 - 2/3) / (3 + 1/2) - (1 + 8/15)) * (square + 71.95) = 1 := by
  sorry

end equation_three_holds_l2931_293168


namespace product_divisible_by_60_l2931_293120

theorem product_divisible_by_60 (a : ℤ) : 
  60 ∣ (a^2 - 1) * a^2 * (a^2 + 1) := by sorry

end product_divisible_by_60_l2931_293120


namespace regression_properties_l2931_293105

/-- A dataset of two variables -/
structure Dataset where
  x : List ℝ
  y : List ℝ

/-- Properties of a linear regression model -/
structure RegressionModel (d : Dataset) where
  x_mean : ℝ
  y_mean : ℝ
  r : ℝ
  b_hat : ℝ
  a_hat : ℝ

/-- The regression line passes through the mean point -/
def passes_through_mean (m : RegressionModel d) : Prop :=
  m.y_mean = m.b_hat * m.x_mean + m.a_hat

/-- Strong correlation between variables -/
def strong_correlation (m : RegressionModel d) : Prop :=
  abs m.r > 0.75

/-- Negative slope of the regression line -/
def negative_slope (m : RegressionModel d) : Prop :=
  m.b_hat < 0

/-- Main theorem -/
theorem regression_properties (d : Dataset) (m : RegressionModel d)
  (h1 : m.r = -0.8) :
  passes_through_mean m ∧ strong_correlation m ∧ negative_slope m := by
  sorry

end regression_properties_l2931_293105


namespace home_to_school_distance_proof_l2931_293117

/-- The distance from Xiao Hong's home to her school -/
def home_to_school_distance : ℝ := 12000

/-- The distance the father drives Xiao Hong -/
def father_driving_distance : ℝ := 1000

/-- The time it takes Xiao Hong to get from home to school by car and walking -/
def car_and_walking_time : ℝ := 22.5

/-- The time it takes Xiao Hong to ride her bike from home to school -/
def bike_riding_time : ℝ := 40

/-- Xiao Hong's walking speed in meters per minute -/
def walking_speed : ℝ := 80

/-- The difference between father's driving speed and Xiao Hong's bike speed -/
def speed_difference : ℝ := 800

theorem home_to_school_distance_proof :
  home_to_school_distance = 12000 :=
sorry

end home_to_school_distance_proof_l2931_293117


namespace commercial_length_l2931_293166

theorem commercial_length 
  (total_time : ℕ) 
  (long_commercial_count : ℕ) 
  (long_commercial_length : ℕ) 
  (short_commercial_count : ℕ) : 
  total_time = 37 ∧ 
  long_commercial_count = 3 ∧ 
  long_commercial_length = 5 ∧ 
  short_commercial_count = 11 → 
  (total_time - long_commercial_count * long_commercial_length) / short_commercial_count = 2 := by
  sorry

end commercial_length_l2931_293166


namespace log_sum_equality_l2931_293169

theorem log_sum_equality : 21 * Real.log 2 + Real.log 25 = 2 := by
  sorry

end log_sum_equality_l2931_293169


namespace min_stamps_l2931_293191

def stamp_problem (n_010 n_020 n_050 n_200 : ℕ) (total : ℚ) : Prop :=
  n_010 ≥ 2 ∧
  n_020 ≥ 5 ∧
  n_050 ≥ 3 ∧
  n_200 ≥ 1 ∧
  total = 10 ∧
  0.1 * n_010 + 0.2 * n_020 + 0.5 * n_050 + 2 * n_200 = total

theorem min_stamps :
  ∃ (n_010 n_020 n_050 n_200 : ℕ),
    stamp_problem n_010 n_020 n_050 n_200 10 ∧
    (∀ (m_010 m_020 m_050 m_200 : ℕ),
      stamp_problem m_010 m_020 m_050 m_200 10 →
      n_010 + n_020 + n_050 + n_200 ≤ m_010 + m_020 + m_050 + m_200) ∧
    n_010 + n_020 + n_050 + n_200 = 17 :=
by
  sorry

end min_stamps_l2931_293191


namespace sin_plus_cos_from_double_angle_l2931_293136

theorem sin_plus_cos_from_double_angle (A : ℝ) (h1 : 0 < A) (h2 : A < π / 2) (h3 : Real.sin (2 * A) = 2 / 3) :
  Real.sin A + Real.cos A = Real.sqrt 15 / 3 := by
sorry

end sin_plus_cos_from_double_angle_l2931_293136


namespace john_garage_sale_games_l2931_293172

/-- The number of games John bought from a friend -/
def games_from_friend : ℕ := 21

/-- The number of games that didn't work -/
def bad_games : ℕ := 23

/-- The number of good games John ended up with -/
def good_games : ℕ := 6

/-- The number of games John bought at the garage sale -/
def games_from_garage_sale : ℕ := (good_games + bad_games) - games_from_friend

theorem john_garage_sale_games :
  games_from_garage_sale = 8 := by sorry

end john_garage_sale_games_l2931_293172


namespace roses_mary_added_l2931_293198

/-- The number of roses Mary put in the vase -/
def roses_added : ℕ := sorry

/-- The initial number of roses in the vase -/
def initial_roses : ℕ := 6

/-- The final number of roses in the vase -/
def final_roses : ℕ := 22

theorem roses_mary_added : roses_added = 16 := by
  sorry

end roses_mary_added_l2931_293198


namespace parabola_properties_l2931_293153

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the point M on the parabola
def point_M (p x y : ℝ) : Prop := parabola p x y ∧ x + 2 = x + p/2

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define a line passing through the focus
def line_through_focus (p m : ℝ) (x y : ℝ) : Prop := x = m*y + p/2

-- Define the intersection points of the line and the parabola
def intersection_points (p m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  line_through_focus p m x₁ y₁ ∧ parabola p x₁ y₁ ∧
  line_through_focus p m x₂ y₂ ∧ parabola p x₂ y₂ ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

theorem parabola_properties (p : ℝ) :
  (∃ x y : ℝ, point_M p x y) →
  (p = 4 ∧
   ∀ m x₁ y₁ x₂ y₂ : ℝ, intersection_points p m x₁ y₁ x₂ y₂ → y₁ * y₂ = -16) :=
by sorry

end

end parabola_properties_l2931_293153


namespace sunday_school_three_year_olds_l2931_293180

/-- The number of 4-year-olds in the Sunday school -/
def four_year_olds : ℕ := 20

/-- The number of 5-year-olds in the Sunday school -/
def five_year_olds : ℕ := 15

/-- The number of 6-year-olds in the Sunday school -/
def six_year_olds : ℕ := 22

/-- The average class size -/
def average_class_size : ℕ := 35

/-- The number of classes -/
def num_classes : ℕ := 2

theorem sunday_school_three_year_olds :
  ∃ (three_year_olds : ℕ),
    (three_year_olds + four_year_olds + five_year_olds + six_year_olds) / num_classes = average_class_size ∧
    three_year_olds = 13 := by
  sorry

end sunday_school_three_year_olds_l2931_293180
