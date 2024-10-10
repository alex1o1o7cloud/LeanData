import Mathlib

namespace total_jumps_l2837_283778

def hattie_first_round : ℕ := 180

def lorelei_first_round : ℕ := (3 * hattie_first_round) / 4

def hattie_second_round : ℕ := (2 * hattie_first_round) / 3

def lorelei_second_round : ℕ := hattie_second_round + 50

def hattie_third_round : ℕ := hattie_second_round + hattie_second_round / 3

def lorelei_third_round : ℕ := (4 * lorelei_first_round) / 5

theorem total_jumps :
  hattie_first_round + lorelei_first_round +
  hattie_second_round + lorelei_second_round +
  hattie_third_round + lorelei_third_round = 873 := by
  sorry

end total_jumps_l2837_283778


namespace sixth_term_of_sequence_l2837_283769

/-- Given a sequence {a_n} where a_1 = 1 and a_{n+1} = a_n + 2 for n ≥ 1, prove that a_6 = 11 -/
theorem sixth_term_of_sequence (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2) : 
  a 6 = 11 := by
sorry

end sixth_term_of_sequence_l2837_283769


namespace tan_value_fourth_quadrant_l2837_283752

theorem tan_value_fourth_quadrant (α : Real) :
  (α ∈ Set.Icc (3 * π / 2) (2 * π)) →  -- α is in the fourth quadrant
  (Real.sin α + Real.cos α = 1 / 5) →  -- given condition
  Real.tan α = -3 / 4 := by
  sorry

end tan_value_fourth_quadrant_l2837_283752


namespace power_equation_solution_l2837_283701

theorem power_equation_solution : ∃ x : ℕ, 27^3 + 27^3 + 27^3 + 27^3 = 3^x :=
by
  use 11
  have h1 : 27 = 3^3 := by sorry
  -- Proof steps would go here
  sorry

end power_equation_solution_l2837_283701


namespace price_decrease_l2837_283715

theorem price_decrease (original_price reduced_price : ℝ) 
  (h1 : reduced_price = original_price * (1 - 0.24))
  (h2 : reduced_price = 532) : original_price = 700 := by
  sorry

end price_decrease_l2837_283715


namespace three_pipes_fill_time_l2837_283745

/-- Represents the time taken to fill a tank given a number of pipes -/
def fill_time (num_pipes : ℕ) (time : ℝ) : Prop :=
  num_pipes > 0 ∧ time > 0 ∧ num_pipes * time = 36

theorem three_pipes_fill_time :
  fill_time 2 18 → fill_time 3 12 := by
  sorry

end three_pipes_fill_time_l2837_283745


namespace pentagon_percentage_is_fifty_percent_l2837_283759

/-- Represents a tiling of the plane with squares and pentagons -/
structure PlaneTiling where
  /-- The number of smaller squares in each large square tile -/
  smallSquaresPerTile : ℕ
  /-- The number of smaller squares that form parts of pentagons -/
  smallSquaresInPentagons : ℕ

/-- Calculates the percentage of the plane enclosed by pentagons -/
def pentagonPercentage (tiling : PlaneTiling) : ℚ :=
  (tiling.smallSquaresInPentagons : ℚ) / (tiling.smallSquaresPerTile : ℚ) * 100

/-- Theorem stating that the percentage of the plane enclosed by pentagons is 50% -/
theorem pentagon_percentage_is_fifty_percent (tiling : PlaneTiling) 
  (h1 : tiling.smallSquaresPerTile = 16)
  (h2 : tiling.smallSquaresInPentagons = 8) : 
  pentagonPercentage tiling = 50 := by
  sorry

#eval pentagonPercentage { smallSquaresPerTile := 16, smallSquaresInPentagons := 8 }

end pentagon_percentage_is_fifty_percent_l2837_283759


namespace square_area_from_perimeter_l2837_283725

theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 40) :
  (perimeter / 4) ^ 2 = 100 := by
  sorry

end square_area_from_perimeter_l2837_283725


namespace symmetric_points_on_parabola_l2837_283723

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define symmetry about the line x + y = m
def symmetric_about_line (x₁ y₁ x₂ y₂ m : ℝ) : Prop :=
  (x₁ + y₁ + x₂ + y₂) / 2 = m

-- Main theorem
theorem symmetric_points_on_parabola (x₁ y₁ x₂ y₂ m : ℝ) :
  is_on_parabola x₁ y₁ →
  is_on_parabola x₂ y₂ →
  symmetric_about_line x₁ y₁ x₂ y₂ m →
  y₁ * y₂ = -1/2 →
  m = 9/4 := by sorry

end symmetric_points_on_parabola_l2837_283723


namespace sum_reciprocal_squares_inequality_l2837_283795

theorem sum_reciprocal_squares_inequality 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d = 3) :
  1/a^2 + 1/b^2 + 1/c^2 + 1/d^2 ≤ 1/(a^2*b^2*c^2*d^2) := by
sorry

end sum_reciprocal_squares_inequality_l2837_283795


namespace average_of_three_numbers_l2837_283706

theorem average_of_three_numbers (A B C : ℝ) 
  (sum_AB : A + B = 147)
  (sum_BC : B + C = 123)
  (sum_AC : A + C = 132) :
  (A + B + C) / 3 = 67 := by
  sorry

end average_of_three_numbers_l2837_283706


namespace complex_fraction_simplification_l2837_283765

theorem complex_fraction_simplification :
  (2 * Complex.I) / (1 + 2 * Complex.I) = 4/5 + (2/5) * Complex.I :=
by sorry

end complex_fraction_simplification_l2837_283765


namespace cousins_ages_sum_l2837_283775

theorem cousins_ages_sum : 
  ∀ (a b c d : ℕ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  0 < a ∧ a < 10 ∧ 0 < b ∧ b < 10 ∧ 0 < c ∧ c < 10 ∧ 0 < d ∧ d < 10 →
  (a * b = 24 ∧ c * d = 30) ∨ (a * c = 24 ∧ b * d = 30) ∨ (a * d = 24 ∧ b * c = 30) →
  a + b + c + d = 22 :=
by sorry

end cousins_ages_sum_l2837_283775


namespace cube_edge_length_l2837_283718

-- Define the volume of the cube in milliliters
def cube_volume : ℝ := 729

-- Define the edge length of the cube in centimeters
def edge_length : ℝ := 9

-- Theorem: The edge length of a cube with volume 729 ml is 9 cm
theorem cube_edge_length : 
  edge_length ^ 3 * 1000 = cube_volume ∧ edge_length = 9 := by
  sorry

end cube_edge_length_l2837_283718


namespace complex_square_one_plus_i_l2837_283779

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_square_one_plus_i : (1 + i)^2 = 2*i := by sorry

end complex_square_one_plus_i_l2837_283779


namespace min_value_theorem_l2837_283707

theorem min_value_theorem (x y : ℝ) (h1 : x * y + 3 * x = 3) (h2 : 0 < x) (h3 : x < 1/2) :
  (3 / x) + (1 / (y - 3)) ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ * y₀ + 3 * x₀ = 3 ∧ 0 < x₀ ∧ x₀ < 1/2 ∧ (3 / x₀) + (1 / (y₀ - 3)) = 8 := by
  sorry

end min_value_theorem_l2837_283707


namespace yule_log_surface_area_increase_l2837_283738

/-- Proves that cutting a cylindrical Yule log into 9 slices increases its surface area by 100π -/
theorem yule_log_surface_area_increase :
  let h : ℝ := 10  -- height of the log
  let d : ℝ := 5   -- diameter of the log
  let n : ℕ := 9   -- number of slices
  let r : ℝ := d / 2  -- radius of the log
  let original_surface_area : ℝ := 2 * π * r * h + 2 * π * r^2
  let slice_height : ℝ := h / n
  let slice_surface_area : ℝ := 2 * π * r * slice_height + 2 * π * r^2
  let total_sliced_surface_area : ℝ := n * slice_surface_area
  let surface_area_increase : ℝ := total_sliced_surface_area - original_surface_area
  surface_area_increase = 100 * π := by
  sorry

end yule_log_surface_area_increase_l2837_283738


namespace consecutive_pages_sum_l2837_283730

theorem consecutive_pages_sum (n : ℕ) : 
  n * (n + 1) * (n + 2) = 479160 → n + (n + 1) + (n + 2) = 234 := by
  sorry

end consecutive_pages_sum_l2837_283730


namespace farm_has_55_cows_l2837_283762

/-- Given information about husk consumption by cows on a dairy farm -/
structure DairyFarm where
  totalBags : ℕ -- Total bags of husk consumed by the group
  totalDays : ℕ -- Total days for group consumption
  singleCowDays : ℕ -- Days for one cow to consume one bag

/-- Calculate the number of cows on the farm -/
def numberOfCows (farm : DairyFarm) : ℕ :=
  farm.totalBags * farm.singleCowDays / farm.totalDays

/-- Theorem stating that the number of cows is 55 under given conditions -/
theorem farm_has_55_cows (farm : DairyFarm)
  (h1 : farm.totalBags = 55)
  (h2 : farm.totalDays = 55)
  (h3 : farm.singleCowDays = 55) :
  numberOfCows farm = 55 := by
  sorry

end farm_has_55_cows_l2837_283762


namespace max_value_of_f_l2837_283750

noncomputable def f (x : ℝ) : ℝ := (-x^2 + x - 4) / x

theorem max_value_of_f :
  ∃ (x_max : ℝ), x_max > 0 ∧
  (∀ (x : ℝ), x > 0 → f x ≤ f x_max) ∧
  f x_max = -3 ∧
  x_max = 2 :=
sorry

end max_value_of_f_l2837_283750


namespace range_of_omega_l2837_283785

/-- Given a function f and its shifted version g, prove the range of ω -/
theorem range_of_omega (f g : ℝ → ℝ) (ω : ℝ) : 
  (ω > 0) →
  (∀ x, f x = Real.sin (π / 3 - ω * x)) →
  (∀ x, g x = Real.sin (ω * x - π / 3)) →
  (∀ x ∈ Set.Icc 0 π, -Real.sqrt 3 / 2 ≤ g x ∧ g x ≤ 1) →
  (5 / 6 : ℝ) ≤ ω ∧ ω ≤ (5 / 3 : ℝ) := by
  sorry

end range_of_omega_l2837_283785


namespace taxi_fare_equality_l2837_283754

/-- Taxi fare calculation problem -/
theorem taxi_fare_equality (mike_start_fee annie_start_fee annie_toll_fee : ℚ)
  (per_mile_rate : ℚ) (annie_miles : ℚ) :
  mike_start_fee = 2.5 ∧
  annie_start_fee = 2.5 ∧
  annie_toll_fee = 5 ∧
  per_mile_rate = 0.25 ∧
  annie_miles = 22 →
  ∃ (mike_miles : ℚ),
    mike_start_fee + per_mile_rate * mike_miles =
    annie_start_fee + annie_toll_fee + per_mile_rate * annie_miles ∧
    mike_miles = 42 :=
by sorry

end taxi_fare_equality_l2837_283754


namespace simplify_expression_evaluate_expression_l2837_283777

-- Problem 1
theorem simplify_expression (x y : ℝ) : 
  x - (2 * x - y) + (3 * x - 2 * y) = 2 * x - y := by sorry

-- Problem 2
theorem evaluate_expression : 
  -(1^4) + |3 - 5| - 8 + (-2) * (1/2) = -8 := by sorry

end simplify_expression_evaluate_expression_l2837_283777


namespace sqrt_xy_eq_three_halves_l2837_283797

theorem sqrt_xy_eq_three_halves (x y : ℝ) (h : |2*x + 1| + Real.sqrt (9 + 2*y) = 0) :
  Real.sqrt (x * y) = 3/2 := by
  sorry

end sqrt_xy_eq_three_halves_l2837_283797


namespace jackie_pushups_l2837_283744

/-- Calculates the number of push-ups Jackie can do in one minute given her initial rate,
    rate of decrease, break times, and rate recovery during breaks. -/
def pushups_in_one_minute (initial_rate : ℕ) (decrease_rate : ℚ) 
                          (break_times : List ℕ) (recovery_rate : ℚ) : ℕ :=
  sorry

/-- Theorem stating that Jackie can do 15 push-ups in one minute under the given conditions. -/
theorem jackie_pushups : 
  pushups_in_one_minute 5 (1/5) [22, 38] (1/10) = 15 := by sorry

end jackie_pushups_l2837_283744


namespace scientific_notation_of_twelve_million_l2837_283733

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def to_scientific_notation (n : ℝ) : ScientificNotation :=
  sorry

/-- The number we want to convert to scientific notation -/
def target_number : ℝ := 12000000

/-- Theorem stating that the scientific notation of 12,000,000 is 1.2 × 10^7 -/
theorem scientific_notation_of_twelve_million :
  (to_scientific_notation target_number).coefficient = 1.2 ∧
  (to_scientific_notation target_number).exponent = 7 :=
sorry

end scientific_notation_of_twelve_million_l2837_283733


namespace cubic_function_root_sum_squares_l2837_283746

/-- Given f(x) = x³ - 2x² - 3x + 4, if there exist distinct a, b, c such that f(a) = f(b) = f(c),
    then a² + b² + c² = 10 -/
theorem cubic_function_root_sum_squares (f : ℝ → ℝ) (a b c : ℝ) :
  f = (λ x => x^3 - 2*x^2 - 3*x + 4) →
  a < b →
  b < c →
  f a = f b →
  f b = f c →
  a^2 + b^2 + c^2 = 10 := by
  sorry

end cubic_function_root_sum_squares_l2837_283746


namespace f_pi_plus_3_l2837_283735

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.tan x + 1

theorem f_pi_plus_3 (a b : ℝ) :
  f a b (-3) = 5 → f a b (Real.pi + 3) = -3 := by
  sorry

end f_pi_plus_3_l2837_283735


namespace price_reduction_percentage_l2837_283719

theorem price_reduction_percentage (last_year_price : ℝ) : 
  let this_year_price := last_year_price * (1 + 0.25)
  let next_year_target := last_year_price * (1 + 0.10)
  ∃ (reduction_percentage : ℝ), 
    this_year_price * (1 - reduction_percentage) = next_year_target ∧ 
    reduction_percentage = 0.12 := by
  sorry

end price_reduction_percentage_l2837_283719


namespace or_and_not_implies_false_and_true_l2837_283737

theorem or_and_not_implies_false_and_true (p q : Prop) :
  (p ∨ q) → (¬p) → (¬p ∧ q) := by
  sorry

end or_and_not_implies_false_and_true_l2837_283737


namespace point_on_line_l2837_283757

/-- Given two points (m, n) and (m + p, n + 21) on the line x = (y / 7) - (2 / 5),
    prove that p = 3 -/
theorem point_on_line (m n p : ℝ) : 
  (m = n / 7 - 2 / 5) ∧ (m + p = (n + 21) / 7 - 2 / 5) → p = 3 := by
  sorry

end point_on_line_l2837_283757


namespace complex_number_problem_l2837_283755

theorem complex_number_problem (z ω : ℂ) :
  (((1 : ℂ) + 3*Complex.I) * z).re = 0 →
  ω = z / ((2 : ℂ) + Complex.I) →
  Complex.abs ω = 5 * Real.sqrt 2 →
  ω = 7 - Complex.I ∨ ω = -7 + Complex.I := by
  sorry

end complex_number_problem_l2837_283755


namespace kylies_daisies_l2837_283772

/-- Proves that Kylie's initial number of daisies is 5 given the problem conditions -/
theorem kylies_daisies (initial : ℕ) (sister_gift : ℕ) (remaining : ℕ) : 
  sister_gift = 9 → 
  remaining = 7 → 
  (initial + sister_gift) / 2 = remaining → 
  initial = 5 := by
sorry

end kylies_daisies_l2837_283772


namespace no_month_with_five_mondays_and_thursdays_l2837_283705

/-- Represents the possible number of days in a month -/
inductive MonthDays : Type where
  | days28 : MonthDays
  | days29 : MonthDays
  | days30 : MonthDays
  | days31 : MonthDays

/-- Converts MonthDays to a natural number -/
def monthDaysToNat (md : MonthDays) : Nat :=
  match md with
  | MonthDays.days28 => 28
  | MonthDays.days29 => 29
  | MonthDays.days30 => 30
  | MonthDays.days31 => 31

/-- Represents a day of the week -/
inductive Weekday : Type where
  | monday : Weekday
  | tuesday : Weekday
  | wednesday : Weekday
  | thursday : Weekday
  | friday : Weekday
  | saturday : Weekday
  | sunday : Weekday

/-- The number of days in a week -/
def daysInWeek : Nat := 7

/-- Counts the number of occurrences of a specific weekday in a month -/
def countWeekday (startDay : Weekday) (monthLength : MonthDays) (day : Weekday) : Nat :=
  sorry  -- Implementation details omitted

theorem no_month_with_five_mondays_and_thursdays :
  ∀ (md : MonthDays) (start : Weekday),
    ¬(countWeekday start md Weekday.monday = 5 ∧ countWeekday start md Weekday.thursday = 5) :=
by sorry


end no_month_with_five_mondays_and_thursdays_l2837_283705


namespace rectangle_area_unchanged_l2837_283721

theorem rectangle_area_unchanged (A l w : ℝ) (h1 : A = l * w) (h2 : A > 0) :
  let l' := 0.8 * l
  let w' := 1.25 * w
  l' * w' = A := by
  sorry

end rectangle_area_unchanged_l2837_283721


namespace sqrt_expressions_equality_l2837_283711

theorem sqrt_expressions_equality :
  (Real.sqrt 75 - Real.sqrt 54 + Real.sqrt 96 - Real.sqrt 108 = -Real.sqrt 3 + Real.sqrt 6) ∧
  (Real.sqrt 24 / Real.sqrt 3 + Real.sqrt (1/2) * Real.sqrt 18 - Real.sqrt 50 = 3 - 3 * Real.sqrt 2) :=
by sorry

end sqrt_expressions_equality_l2837_283711


namespace batsman_average_l2837_283712

/-- Calculates the new average score after an additional inning -/
def new_average (prev_avg : ℚ) (prev_innings : ℕ) (new_score : ℕ) : ℚ :=
  (prev_avg * prev_innings + new_score) / (prev_innings + 1)

/-- Theorem: Given the conditions, the batsman's new average is 18 -/
theorem batsman_average : new_average 19 17 1 = 18 := by
  sorry

end batsman_average_l2837_283712


namespace smallest_stable_triangle_side_l2837_283700

/-- A stable triangle is a scalene triangle with positive integer side lengths that are multiples of 5, 80, and 112 respectively. -/
def StableTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧  -- scalene
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive
  ∃ (x y z : ℕ), a = 5 * x ∧ b = 80 * y ∧ c = 112 * z  -- multiples of 5, 80, 112

/-- The smallest possible side length in any stable triangle is 20. -/
theorem smallest_stable_triangle_side : 
  (∃ (a b c : ℕ), StableTriangle a b c) → 
  (∀ (a b c : ℕ), StableTriangle a b c → min a (min b c) ≥ 20) ∧
  (∃ (a b c : ℕ), StableTriangle a b c ∧ min a (min b c) = 20) :=
sorry

end smallest_stable_triangle_side_l2837_283700


namespace sum_of_squares_of_roots_l2837_283704

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 6 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 6 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 6 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -32/9 := by
sorry

end sum_of_squares_of_roots_l2837_283704


namespace tournament_games_played_l2837_283764

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  num_teams : ℕ
  no_ties : Bool

/-- The number of games played in a single-elimination tournament -/
def games_played (t : SingleEliminationTournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 24 teams and no ties,
    the number of games played to declare a winner is 23 -/
theorem tournament_games_played :
  ∀ (t : SingleEliminationTournament),
    t.num_teams = 24 → t.no_ties = true →
    games_played t = 23 := by
  sorry

end tournament_games_played_l2837_283764


namespace roses_apple_sharing_l2837_283774

/-- Given that Rose has 9 apples and each friend receives 3 apples,
    prove that the number of friends Rose shares her apples with is 3. -/
theorem roses_apple_sharing :
  let total_apples : ℕ := 9
  let apples_per_friend : ℕ := 3
  total_apples / apples_per_friend = 3 :=
by sorry

end roses_apple_sharing_l2837_283774


namespace larger_number_proof_l2837_283726

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 2500) (h3 : L = 6 * S + 15) : L = 2997 := by
  sorry

end larger_number_proof_l2837_283726


namespace solve_equation_l2837_283741

-- Define the function F
def F (a b c : ℚ) : ℚ := a * b^3 + c

-- Theorem statement
theorem solve_equation :
  ∃ a : ℚ, F a 3 8 = F a 5 12 ∧ a = -2/49 := by
  sorry

end solve_equation_l2837_283741


namespace b_is_largest_l2837_283702

/-- Represents a number with a finite or repeating decimal expansion -/
structure DecimalNumber where
  whole : ℕ
  finite : List ℕ
  repeating : List ℕ

/-- Converts a DecimalNumber to a real number -/
noncomputable def toReal (d : DecimalNumber) : ℝ :=
  sorry

/-- The five numbers we're comparing -/
def a : DecimalNumber := { whole := 8, finite := [1, 2, 3, 6, 6], repeating := [] }
def b : DecimalNumber := { whole := 8, finite := [1, 2, 3], repeating := [6] }
def c : DecimalNumber := { whole := 8, finite := [1, 2], repeating := [3, 6] }
def d : DecimalNumber := { whole := 8, finite := [1], repeating := [2, 3, 6] }
def e : DecimalNumber := { whole := 8, finite := [], repeating := [1, 2, 3, 6] }

/-- Theorem stating that b is the largest among the given numbers -/
theorem b_is_largest :
  (toReal b > toReal a) ∧
  (toReal b > toReal c) ∧
  (toReal b > toReal d) ∧
  (toReal b > toReal e) :=
by
  sorry

end b_is_largest_l2837_283702


namespace stratified_sampling_car_inspection_l2837_283728

theorem stratified_sampling_car_inspection
  (total_sample : ℕ)
  (type_a_production type_b_production type_c_production : ℕ)
  (h_total_sample : total_sample = 47)
  (h_type_a : type_a_production = 1400)
  (h_type_b : type_b_production = 6000)
  (h_type_c : type_c_production = 2000) :
  ∃ (sample_a sample_b sample_c : ℕ),
    sample_a + sample_b + sample_c = total_sample ∧
    sample_a = 7 ∧
    sample_b = 30 ∧
    sample_c = 10 :=
by sorry

end stratified_sampling_car_inspection_l2837_283728


namespace parabola_parameter_l2837_283770

/-- Theorem: For a parabola y^2 = 2px (p > 0) with focus F, if a line through F makes an angle of π/3
    with the x-axis and intersects the parabola at points A and B with |AB| = 8, then p = 3. -/
theorem parabola_parameter (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  (∀ x y, y^2 = 2*p*x) →
  (∃ m b, ∀ x y, y = m*x + b ∧ m = Real.sqrt 3) →
  (∀ x y, y^2 = 2*p*x → (∃ t, x = t ∧ y = Real.sqrt 3 * (t - p/2))) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 →
  p = 3 := by
  sorry

end parabola_parameter_l2837_283770


namespace goldfish_equality_l2837_283732

/-- The number of months after which Alice and Bob have the same number of goldfish -/
def same_goldfish_month : ℕ := 7

/-- Alice's initial number of goldfish -/
def alice_initial : ℕ := 3

/-- Bob's initial number of goldfish -/
def bob_initial : ℕ := 256

/-- Alice's goldfish growth rate per month -/
def alice_growth_rate : ℕ := 3

/-- Bob's goldfish growth rate per month -/
def bob_growth_rate : ℕ := 4

/-- Alice's number of goldfish after n months -/
def alice_goldfish (n : ℕ) : ℕ := alice_initial * (alice_growth_rate ^ n)

/-- Bob's number of goldfish after n months -/
def bob_goldfish (n : ℕ) : ℕ := bob_initial * (bob_growth_rate ^ n)

theorem goldfish_equality :
  alice_goldfish same_goldfish_month = bob_goldfish same_goldfish_month ∧
  ∀ m : ℕ, m < same_goldfish_month → alice_goldfish m ≠ bob_goldfish m :=
by sorry

end goldfish_equality_l2837_283732


namespace tire_usage_proof_l2837_283761

/-- Represents the number of miles each tire is used when seven tires are used equally over a total distance --/
def miles_per_tire (total_miles : ℕ) : ℚ :=
  (4 * total_miles : ℚ) / 7

/-- Proves that given the conditions of the problem, each tire is used for 25,714 miles --/
theorem tire_usage_proof (total_miles : ℕ) (h1 : total_miles = 45000) :
  ⌊miles_per_tire total_miles⌋ = 25714 := by
  sorry

#eval ⌊miles_per_tire 45000⌋

end tire_usage_proof_l2837_283761


namespace solve_exponential_equation_l2837_283703

theorem solve_exponential_equation :
  ∃ x : ℝ, (3 : ℝ)^4 * (3 : ℝ)^x = 81 ∧ x = 0 :=
by
  sorry

end solve_exponential_equation_l2837_283703


namespace area_intersection_approx_l2837_283740

/-- The elliptical region D₁ -/
def D₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 ≤ 1

/-- The circular region D₂ -/
def D₂ (x y : ℝ) : Prop := x^2 + y^2 ≤ 2

/-- The intersection of D₁ and D₂ -/
def D_intersection (x y : ℝ) : Prop := D₁ x y ∧ D₂ x y

/-- The area of the intersection of D₁ and D₂ -/
noncomputable def area_intersection : ℝ := sorry

theorem area_intersection_approx :
  abs (area_intersection - 5.88) < 0.01 := by sorry

end area_intersection_approx_l2837_283740


namespace least_number_for_divisibility_l2837_283798

theorem least_number_for_divisibility : ∃! x : ℕ, x < 25 ∧ (1056 + x) % 25 = 0 ∧ ∀ y : ℕ, y < x → (1056 + y) % 25 ≠ 0 := by
  sorry

end least_number_for_divisibility_l2837_283798


namespace perimeter_ratio_l2837_283793

/-- Triangle PQR with sides of length 6, 8, and 10 units -/
def PQR : Fin 3 → ℝ := ![6, 8, 10]

/-- Triangle STU with sides of length 9, 12, and 15 units -/
def STU : Fin 3 → ℝ := ![9, 12, 15]

/-- Perimeter of a triangle given its side lengths -/
def perimeter (triangle : Fin 3 → ℝ) : ℝ :=
  triangle 0 + triangle 1 + triangle 2

/-- The ratio of the perimeter of triangle PQR to the perimeter of triangle STU is 2/3 -/
theorem perimeter_ratio :
  perimeter PQR / perimeter STU = 2 / 3 := by
  sorry

end perimeter_ratio_l2837_283793


namespace equilateral_triangle_product_l2837_283720

/-- Given that (0, 0), (a, 8), and (b, 20) form an equilateral triangle,
    prove that ab = 320/3 -/
theorem equilateral_triangle_product (a b : ℝ) : 
  (∃ (θ : ℝ), θ = π/3 ∨ θ = -π/3) →
  (Complex.abs (Complex.I * 8 - 0) = Complex.abs (b + Complex.I * 20 - 0)) →
  (Complex.abs (b + Complex.I * 20 - (a + Complex.I * 8)) = Complex.abs (Complex.I * 8 - 0)) →
  (b + Complex.I * 20 = (a + Complex.I * 8) * Complex.exp (Complex.I * θ)) →
  a * b = 320 / 3 := by
sorry

end equilateral_triangle_product_l2837_283720


namespace solution_inequality1_solution_inequality_system_l2837_283790

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 2 * x + 3 ≤ 5 * x
def inequality2 (x : ℝ) : Prop := 5 * x - 1 ≤ 3 * (x + 1)
def inequality3 (x : ℝ) : Prop := (2 * x - 1) / 2 - (5 * x - 1) / 4 < 1

-- Theorem for the first inequality
theorem solution_inequality1 :
  {x : ℝ | inequality1 x} = {x : ℝ | x ≥ 1} :=
sorry

-- Theorem for the system of inequalities
theorem solution_inequality_system :
  {x : ℝ | inequality2 x ∧ inequality3 x} = {x : ℝ | -5 < x ∧ x ≤ 2} :=
sorry

end solution_inequality1_solution_inequality_system_l2837_283790


namespace infinite_series_sum_l2837_283781

theorem infinite_series_sum : 
  (∑' n : ℕ, (n : ℝ) / (5 ^ n)) = 5 / 16 := by sorry

end infinite_series_sum_l2837_283781


namespace min_triangles_is_eighteen_l2837_283734

/-- Represents a non-convex hexagon formed by removing one corner square from an 8x8 chessboard -/
structure ChessboardHexagon where
  area : ℝ
  side_length : ℝ

/-- Calculates the minimum number of congruent triangles needed to partition the ChessboardHexagon -/
def min_congruent_triangles (h : ChessboardHexagon) : ℕ :=
  sorry

/-- The theorem stating that the minimum number of congruent triangles is 18 -/
theorem min_triangles_is_eighteen (h : ChessboardHexagon) 
  (h_area : h.area = 63)
  (h_side : h.side_length = 8) : 
  min_congruent_triangles h = 18 := by
  sorry

end min_triangles_is_eighteen_l2837_283734


namespace nested_cube_root_l2837_283748

theorem nested_cube_root (M : ℝ) (h : M > 1) :
  (M * (M * (M * M^(1/3))^(1/3))^(1/3))^(1/3) = M^(40/81) := by
  sorry

end nested_cube_root_l2837_283748


namespace circle_chord_length_l2837_283791

theorem circle_chord_length (AB CD : ℝ) (h1 : AB = 13) (h2 : CD = 6) :
  let AD := (x : ℝ)
  (x = 4 ∨ x = 9) ↔ x^2 - AB*x + CD^2 = 0 := by
sorry

end circle_chord_length_l2837_283791


namespace parabola_y_intercepts_l2837_283717

/-- The number of y-intercepts of the parabola x = 3y^2 - 2y + 1 -/
theorem parabola_y_intercepts : 
  let f : ℝ → ℝ := fun y => 3 * y^2 - 2 * y + 1
  (∃ y, f y = 0) = False :=
by sorry

end parabola_y_intercepts_l2837_283717


namespace intersection_point_product_range_l2837_283766

theorem intersection_point_product_range (k x₀ y₀ : ℝ) :
  x₀ + y₀ = 2 * k - 1 →
  x₀^2 + y₀^2 = k^2 + 2 * k - 3 →
  (11 - 6 * Real.sqrt 2) / 4 ≤ x₀ * y₀ ∧ x₀ * y₀ ≤ (11 + 6 * Real.sqrt 2) / 4 := by
  sorry

end intersection_point_product_range_l2837_283766


namespace smallest_positive_angle_l2837_283747

theorem smallest_positive_angle (y : Real) : 
  (4 * Real.sin y * (Real.cos y)^3 - 4 * (Real.sin y)^3 * Real.cos y = Real.cos y) →
  (y > 0) →
  (∀ z, z > 0 ∧ 4 * Real.sin z * (Real.cos z)^3 - 4 * (Real.sin z)^3 * Real.cos z = Real.cos z → y ≤ z) →
  y = 18 * π / 180 := by
sorry

end smallest_positive_angle_l2837_283747


namespace flagpole_break_height_l2837_283753

theorem flagpole_break_height (h : ℝ) (b : ℝ) (break_height : ℝ) :
  h = 8 →
  b = 3 →
  break_height = (Real.sqrt (h^2 + b^2)) / 2 →
  break_height = Real.sqrt 73 / 2 :=
by
  sorry

end flagpole_break_height_l2837_283753


namespace factorization_condition_l2837_283789

def is_factorizable (m : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ),
    ∀ (x y : ℤ),
      x^2 + 3*x*y + x + m*y - m = (a*x + b*y + c) * (d*x + e*y + f)

theorem factorization_condition (m : ℤ) :
  is_factorizable m ↔ (m = 0 ∨ m = 12) :=
sorry

end factorization_condition_l2837_283789


namespace stamps_per_book_is_15_l2837_283713

/-- The number of stamps in each book of the second type -/
def stamps_per_book : ℕ := sorry

/-- The total number of stamps Ruel has -/
def total_stamps : ℕ := 130

/-- The number of books of the first type (10 stamps each) -/
def books_type1 : ℕ := 4

/-- The number of stamps in each book of the first type -/
def stamps_per_book_type1 : ℕ := 10

/-- The number of books of the second type -/
def books_type2 : ℕ := 6

theorem stamps_per_book_is_15 : 
  stamps_per_book = 15 ∧ 
  total_stamps = books_type1 * stamps_per_book_type1 + books_type2 * stamps_per_book :=
by sorry

end stamps_per_book_is_15_l2837_283713


namespace complex_equation_ratio_l2837_283710

theorem complex_equation_ratio (a b : ℝ) : 
  (a - 2*Complex.I)*Complex.I = b + a*Complex.I → a/b = 1/2 := by
  sorry

end complex_equation_ratio_l2837_283710


namespace triangle_ABC_point_C_l2837_283729

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)

-- Define the line on which C lies
def line_C (x : ℝ) : ℝ := 3 * x + 3

-- Define the area of the triangle
def triangle_area : ℝ := 10

-- Theorem statement
theorem triangle_ABC_point_C :
  ∀ (C : ℝ × ℝ),
  (C.2 = line_C C.1) →  -- C lies on the line y = 3x + 3
  (abs ((C.1 - A.1) * (B.2 - A.2) - (C.2 - A.2) * (B.1 - A.1)) / 2 = triangle_area) →  -- Area of triangle ABC is 10
  (C = (-1, 0) ∨ C = (5/3, 14)) :=
by sorry

end triangle_ABC_point_C_l2837_283729


namespace total_people_in_program_l2837_283780

theorem total_people_in_program (parents pupils : ℕ) 
  (h1 : parents = 105) 
  (h2 : pupils = 698) : 
  parents + pupils = 803 := by
  sorry

end total_people_in_program_l2837_283780


namespace product_sale_loss_l2837_283722

/-- Represents the pricing and sale of a product -/
def ProductSale (cost_price : ℝ) : Prop :=
  let initial_markup := 1.20
  let price_reduction := 0.80
  let sale_price := 96
  initial_markup * cost_price * price_reduction = sale_price ∧
  cost_price > sale_price ∧
  cost_price - sale_price = 4

/-- Theorem stating the loss in the product sale -/
theorem product_sale_loss :
  ∃ (cost_price : ℝ), ProductSale cost_price :=
sorry

end product_sale_loss_l2837_283722


namespace function_extrema_l2837_283786

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a*x - (a+1) * Real.log x

theorem function_extrema (a : ℝ) (h1 : a < -1) :
  (∀ x : ℝ, x > 0 → (deriv (f a)) 2 = 0) →
  (a = -3 ∧ 
   (∀ x : ℝ, x > 0 → f a x ≤ f a 1) ∧
   (∀ x : ℝ, x > 0 → f a x ≥ f a 2) ∧
   f a 1 = -5/2 ∧
   f a 2 = -4 + 2 * Real.log 2) := by
  sorry

end

end function_extrema_l2837_283786


namespace first_number_calculation_l2837_283760

theorem first_number_calculation (average : ℝ) (num1 num2 added_num : ℝ) :
  average = 13 ∧ num1 = 16 ∧ num2 = 8 ∧ added_num = 22 →
  ∃ x : ℝ, (x + num1 + num2 + added_num) / 4 = average ∧ x = 6 := by
  sorry

end first_number_calculation_l2837_283760


namespace larger_number_in_ratio_l2837_283749

theorem larger_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / b = 2 / 3 →
  Nat.lcm a b = 120 →
  b = 72 := by
sorry

end larger_number_in_ratio_l2837_283749


namespace sector_angle_measure_l2837_283783

/-- Given a circular sector with radius 10 and area 50π/3, 
    prove that its central angle measures π/3 radians. -/
theorem sector_angle_measure (r : ℝ) (S : ℝ) (α : ℝ) 
  (h_radius : r = 10)
  (h_area : S = 50 * Real.pi / 3)
  (h_sector_area : S = 1/2 * r^2 * α) :
  α = Real.pi / 3 := by
  sorry

end sector_angle_measure_l2837_283783


namespace intersection_of_A_and_B_l2837_283796

def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by sorry

end intersection_of_A_and_B_l2837_283796


namespace smallest_denominator_for_repeating_2015_l2837_283792

/-- Given positive integers a and b where a/b is a repeating decimal with the sequence 2015,
    the smallest possible value of b is 129. -/
theorem smallest_denominator_for_repeating_2015 (a b : ℕ+) :
  (∃ k : ℕ, (a : ℚ) / b = 2015 / (10000 ^ k - 1)) →
  (∀ c : ℕ+, c < b → ¬∃ d : ℕ+, (d : ℚ) / c = 2015 / 9999) →
  b = 129 := by
  sorry


end smallest_denominator_for_repeating_2015_l2837_283792


namespace population_decrease_percentage_l2837_283763

/-- Calculates the percentage of population that moved away after a growth spurt -/
def percentage_moved_away (initial_population : ℕ) (growth_rate : ℚ) (final_population : ℕ) : ℚ :=
  let population_after_growth := initial_population * (1 + growth_rate)
  let people_moved_away := population_after_growth - final_population
  people_moved_away / population_after_growth

theorem population_decrease_percentage 
  (initial_population : ℕ) 
  (growth_rate : ℚ) 
  (final_population : ℕ) 
  (h1 : initial_population = 684) 
  (h2 : growth_rate = 1/4) 
  (h3 : final_population = 513) : 
  percentage_moved_away initial_population growth_rate final_population = 2/5 := by
  sorry

#eval percentage_moved_away 684 (1/4) 513

end population_decrease_percentage_l2837_283763


namespace function_positive_implies_a_bound_l2837_283773

/-- Given a function f(x) = x^2 - ax + 2 that is positive for all x > 2,
    prove that a ≤ 3. -/
theorem function_positive_implies_a_bound (a : ℝ) :
  (∀ x > 2, x^2 - a*x + 2 > 0) → a ≤ 3 := by
  sorry

end function_positive_implies_a_bound_l2837_283773


namespace sum_of_fractions_inequality_l2837_283736

theorem sum_of_fractions_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + c * d * a / (1 - b)^2 + 
  d * a * b / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1/9 := by
  sorry

end sum_of_fractions_inequality_l2837_283736


namespace reflect_P_across_y_axis_l2837_283787

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The theorem stating that reflecting P(1, -2) across the y-axis results in (-1, -2) -/
theorem reflect_P_across_y_axis :
  let P : Point := { x := 1, y := -2 }
  reflectAcrossYAxis P = { x := -1, y := -2 } := by
  sorry

end reflect_P_across_y_axis_l2837_283787


namespace max_passable_levels_l2837_283716

/-- Represents the maximum number of points obtainable from a single dice throw -/
def max_dice_points : ℕ := 6

/-- Represents the pass condition for a level in the "pass-through game" -/
def pass_condition (n : ℕ) : ℕ := 2^n

/-- Represents the maximum sum of points obtainable from n dice throws -/
def max_sum_points (n : ℕ) : ℕ := n * max_dice_points

/-- Theorem stating the maximum number of levels that can be passed in the "pass-through game" -/
theorem max_passable_levels : 
  ∃ (max_level : ℕ), 
    (∀ n : ℕ, n ≤ max_level → max_sum_points n > pass_condition n) ∧ 
    (∀ n : ℕ, n > max_level → max_sum_points n ≤ pass_condition n) :=
sorry

end max_passable_levels_l2837_283716


namespace negation_equivalence_l2837_283731

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) := by sorry

end negation_equivalence_l2837_283731


namespace speedster_convertible_fraction_l2837_283799

theorem speedster_convertible_fraction (T S : ℕ) (h1 : S = 3 * T / 4) (h2 : T - S = 30) : 
  54 / S = 3 / 5 := by
  sorry

end speedster_convertible_fraction_l2837_283799


namespace calculator_sale_result_l2837_283788

def calculator_transaction (price : ℝ) (profit_rate : ℝ) (loss_rate : ℝ) : Prop :=
  let profit_calculator_cost : ℝ := price / (1 + profit_rate)
  let loss_calculator_cost : ℝ := price / (1 - loss_rate)
  let total_cost : ℝ := profit_calculator_cost + loss_calculator_cost
  let total_revenue : ℝ := 2 * price
  total_revenue - total_cost = -7.5

theorem calculator_sale_result :
  calculator_transaction 90 0.2 0.2 := by
  sorry

end calculator_sale_result_l2837_283788


namespace distance_traveled_l2837_283724

theorem distance_traveled (speed : ℝ) (time : ℝ) : 
  speed = 57 → time = 30 / 3600 → speed * time = 0.475 := by
  sorry

end distance_traveled_l2837_283724


namespace solve_for_y_l2837_283742

theorem solve_for_y (x : ℝ) (y : ℝ) 
  (h1 : x = 101) 
  (h2 : x^3 * y - 2 * x^2 * y + x * y = 101000) : 
  y = 1/10 := by
sorry

end solve_for_y_l2837_283742


namespace largest_minimum_uniform_output_l2837_283758

def black_box (n : ℕ) : ℕ :=
  if n % 2 = 1 then 4 * n + 1 else n / 2

def series_black_box (n : ℕ) : ℕ :=
  black_box (black_box (black_box n))

def is_valid_input (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < n ∧ b < n ∧ c < n ∧
  series_black_box a = series_black_box b ∧
  series_black_box b = series_black_box c ∧
  series_black_box c = series_black_box n

theorem largest_minimum_uniform_output :
  ∃ (n : ℕ), is_valid_input n ∧
  (∀ m, is_valid_input m → m ≤ n) ∧
  (∀ k, k < n → ¬is_valid_input k) ∧
  n = 680 :=
sorry

end largest_minimum_uniform_output_l2837_283758


namespace equation_solutions_l2837_283756

theorem equation_solutions : 
  let f (x : ℝ) := 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 
                   1 / ((x - 4) * (x - 5)) + 1 / ((x - 5) * (x - 6))
  ∀ x : ℝ, f x = 1 / 12 ↔ x = 12 ∨ x = -4 :=
by sorry

end equation_solutions_l2837_283756


namespace cricket_innings_count_l2837_283767

-- Define the problem parameters
def current_average : ℝ := 32
def runs_next_innings : ℝ := 116
def average_increase : ℝ := 4

-- Theorem statement
theorem cricket_innings_count :
  ∀ n : ℝ,
  (n > 0) →
  (current_average * n + runs_next_innings) / (n + 1) = current_average + average_increase →
  n = 20 := by
sorry

end cricket_innings_count_l2837_283767


namespace solution_to_equation_l2837_283727

theorem solution_to_equation : ∃! (x : ℝ), x ≠ 0 ∧ (7 * x)^4 = (14 * x)^3 ∧ x = 8/7 := by
  sorry

end solution_to_equation_l2837_283727


namespace area_enclosed_by_line_and_curve_l2837_283751

/-- Given that the binomial coefficients of the third and fourth terms 
    in the expansion of (x - 2/x)^n are equal, prove that the area enclosed 
    by the line y = nx and the curve y = x^2 is 125/6 -/
theorem area_enclosed_by_line_and_curve (n : ℕ) : 
  (Nat.choose n 2 = Nat.choose n 3) → 
  (∫ (x : ℝ) in (0)..(5), n * x - x^2) = 125 / 6 := by
  sorry

end area_enclosed_by_line_and_curve_l2837_283751


namespace erasers_lost_l2837_283709

def initial_erasers : ℕ := 95
def final_erasers : ℕ := 53

theorem erasers_lost : initial_erasers - final_erasers = 42 := by
  sorry

end erasers_lost_l2837_283709


namespace hyperbola_eccentricity_l2837_283794

/-- Given a hyperbola and a parabola with specific properties, prove that the eccentricity of the hyperbola is √2 + 1 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c = Real.sqrt (a^2 + b^2)) :
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let parabola := fun (x y : ℝ) => y^2 = 4 * c * x
  let A := (c, 2 * c)
  let B := (-c, -2 * c)
  (hyperbola A.1 A.2 ∧ parabola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ parabola B.1 B.2) →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4 * c)^2 →
  let e := c / a
  e = Real.sqrt 2 + 1 := by
sorry

end hyperbola_eccentricity_l2837_283794


namespace remaining_payment_l2837_283782

def deposit_percentage : ℝ := 0.1
def deposit_amount : ℝ := 120

theorem remaining_payment (total : ℝ) (h1 : total * deposit_percentage = deposit_amount) :
  total - deposit_amount = 1080 := by sorry

end remaining_payment_l2837_283782


namespace rhombus_properties_l2837_283771

/-- Given a rhombus with diagonals of 18 inches and 24 inches, this theorem proves:
    1. The perimeter of the rhombus is 60 inches.
    2. The area of a triangle formed by one side of the rhombus and half of each diagonal is 67.5 square inches. -/
theorem rhombus_properties (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 24) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  (4 * s = 60) ∧ ((s * (d1 / 2)) / 2 = 67.5) :=
by sorry

end rhombus_properties_l2837_283771


namespace min_distance_point_triangle_l2837_283739

/-- Given a triangle ABC with vertices (x₁, y₁), (x₂, y₂), and (x₃, y₃), 
    this theorem states that the point P which minimizes the sum of squared distances 
    to the vertices of triangle ABC has coordinates ((x₁ + x₂ + x₃)/3, (y₁ + y₂ + y₃)/3). -/
theorem min_distance_point_triangle (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) :
  let vertices := [(x₁, y₁), (x₂, y₂), (x₃, y₃)]
  let sum_squared_distances (px py : ℝ) := 
    (vertices.map (fun (x, y) => (px - x)^2 + (py - y)^2)).sum
  let p := ((x₁ + x₂ + x₃)/3, (y₁ + y₂ + y₃)/3)
  ∀ q : ℝ × ℝ, sum_squared_distances p.1 p.2 ≤ sum_squared_distances q.1 q.2 :=
by sorry

end min_distance_point_triangle_l2837_283739


namespace silverware_probability_l2837_283743

def forks : ℕ := 8
def spoons : ℕ := 10
def knives : ℕ := 6

def total_silverware : ℕ := forks + spoons + knives

def favorable_outcomes : ℕ := forks * spoons * knives

def total_outcomes : ℕ := Nat.choose total_silverware 3

theorem silverware_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 120 / 506 := by
  sorry

end silverware_probability_l2837_283743


namespace jordan_running_time_l2837_283784

/-- Given that Jordan ran 4 miles in one-third the time it took Steve to run 6 miles,
    and Steve took 36 minutes to run 6 miles, prove that Jordan would take 21 minutes
    to run 7 miles. -/
theorem jordan_running_time
  (steve_time : ℝ)
  (steve_distance : ℝ)
  (jordan_distance : ℝ)
  (jordan_time_fraction : ℝ)
  (jordan_new_distance : ℝ)
  (h1 : steve_time = 36)
  (h2 : steve_distance = 6)
  (h3 : jordan_distance = 4)
  (h4 : jordan_time_fraction = 1 / 3)
  (h5 : jordan_new_distance = 7)
  : (jordan_new_distance * jordan_time_fraction * steve_time) / jordan_distance = 21 := by
  sorry

#check jordan_running_time

end jordan_running_time_l2837_283784


namespace hospital_bill_ambulance_cost_l2837_283768

/-- Given a hospital bill with specified percentages for various services and fixed costs,
    calculate the cost of the ambulance ride. -/
theorem hospital_bill_ambulance_cost 
  (total_bill : ℝ) 
  (medication_percent : ℝ) 
  (imaging_percent : ℝ) 
  (surgical_percent : ℝ) 
  (overnight_percent : ℝ) 
  (food_cost : ℝ) 
  (consultation_cost : ℝ) 
  (h1 : total_bill = 12000)
  (h2 : medication_percent = 0.40)
  (h3 : imaging_percent = 0.15)
  (h4 : surgical_percent = 0.20)
  (h5 : overnight_percent = 0.25)
  (h6 : food_cost = 300)
  (h7 : consultation_cost = 80)
  (h8 : medication_percent + imaging_percent + surgical_percent + overnight_percent = 1) :
  total_bill - (food_cost + consultation_cost) = 11620 := by
  sorry


end hospital_bill_ambulance_cost_l2837_283768


namespace fibonacci_fifth_is_s_plus_one_l2837_283708

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def s : ℕ := 4

theorem fibonacci_fifth_is_s_plus_one :
  fibonacci 5 = s + 1 ∧ ∀ k < 5, fibonacci k ≠ s + 1 := by sorry

end fibonacci_fifth_is_s_plus_one_l2837_283708


namespace correct_num_cups_l2837_283776

/-- The number of cups of coffee on the tray -/
def num_cups : ℕ := 5

/-- The initial volume of coffee in each cup (in ounces) -/
def initial_volume : ℝ := 8

/-- The shrink factor of the ray -/
def shrink_factor : ℝ := 0.5

/-- The total volume of coffee after shrinking (in ounces) -/
def final_total_volume : ℝ := 20

/-- Theorem stating that the number of cups is correct given the conditions -/
theorem correct_num_cups :
  initial_volume * shrink_factor * num_cups = final_total_volume :=
by sorry

end correct_num_cups_l2837_283776


namespace star_one_two_l2837_283714

-- Define the * operation
def star (a b : ℝ) : ℝ := a + b + a * b

-- State the theorem
theorem star_one_two (a : ℝ) : star (star a 1) 2 = 6 * a + 5 := by
  sorry

end star_one_two_l2837_283714
