import Mathlib

namespace NUMINAMATH_CALUDE_altitudes_constructible_l1798_179857

/-- Represents a point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle in a 2D plane -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents a circle in a 2D plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents construction tools -/
inductive ConstructionTool
  | Straightedge
  | Protractor

/-- Represents an altitude of a triangle -/
structure Altitude :=
  (base : Point)
  (apex : Point)

/-- Function to construct altitudes of a triangle -/
def constructAltitudes (t : Triangle) (c : Circle) (tools : List ConstructionTool) : 
  List Altitude :=
  sorry

/-- Theorem stating that altitudes can be constructed -/
theorem altitudes_constructible (t : Triangle) (c : Circle) : 
  ∃ (tools : List ConstructionTool), 
    (ConstructionTool.Straightedge ∈ tools) ∧ 
    (ConstructionTool.Protractor ∈ tools) ∧ 
    (constructAltitudes t c tools).length = 3 :=
  sorry

end NUMINAMATH_CALUDE_altitudes_constructible_l1798_179857


namespace NUMINAMATH_CALUDE_trash_can_problem_l1798_179819

/-- Represents the unit price of trash can A -/
def price_A : ℝ := 60

/-- Represents the unit price of trash can B -/
def price_B : ℝ := 100

/-- Represents the total number of trash cans needed -/
def total_cans : ℕ := 200

/-- Represents the maximum total cost allowed -/
def max_cost : ℝ := 15000

theorem trash_can_problem :
  (3 * price_A + 4 * price_B = 580) ∧
  (6 * price_A + 5 * price_B = 860) ∧
  (∀ a : ℕ, a ≥ 125 → 
    (price_A * a + price_B * (total_cans - a) ≤ max_cost)) ∧
  (∀ a : ℕ, a < 125 → 
    (price_A * a + price_B * (total_cans - a) > max_cost)) :=
by sorry

end NUMINAMATH_CALUDE_trash_can_problem_l1798_179819


namespace NUMINAMATH_CALUDE_min_value_of_f_on_interval_l1798_179817

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the interval [0, 3]
def interval : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem min_value_of_f_on_interval :
  ∃ (min : ℝ), min = 0 ∧ ∀ x ∈ interval, f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_on_interval_l1798_179817


namespace NUMINAMATH_CALUDE_anya_wins_l1798_179856

/-- Represents the possible choices in rock-paper-scissors game -/
inductive Choice
  | Rock
  | Paper
  | Scissors

/-- Determines if the first player wins against the second player -/
def wins (a b : Choice) : Bool :=
  match a, b with
  | Choice.Rock, Choice.Scissors => true
  | Choice.Paper, Choice.Rock => true
  | Choice.Scissors, Choice.Paper => true
  | _, _ => false

/-- Represents the game state -/
structure GameState where
  totalRounds : Nat
  anyaRock : Nat
  anyaScissors :Nat
  anyaPaper : Nat
  boryaRock : Nat
  boryaScissors : Nat
  boryaPaper : Nat

/-- Theorem stating that Anya won exactly 19 games -/
theorem anya_wins (g : GameState) : 
  g.totalRounds = 25 ∧ 
  g.anyaRock = 12 ∧ 
  g.anyaScissors = 6 ∧ 
  g.anyaPaper = 7 ∧
  g.boryaRock = 13 ∧ 
  g.boryaScissors = 9 ∧ 
  g.boryaPaper = 3 ∧
  g.totalRounds = g.anyaRock + g.anyaScissors + g.anyaPaper ∧
  g.totalRounds = g.boryaRock + g.boryaScissors + g.boryaPaper →
  g.anyaRock + g.anyaScissors + g.anyaPaper = 25 ∧
  (g.anyaRock.min g.boryaScissors) + 
  (g.anyaScissors.min g.boryaPaper) + 
  (g.anyaPaper.min g.boryaRock) = 19 := by
  sorry


end NUMINAMATH_CALUDE_anya_wins_l1798_179856


namespace NUMINAMATH_CALUDE_two_digit_multiples_of_five_mean_l1798_179810

/-- The smallest positive two-digit multiple of 5 -/
def smallest : ℕ := 10

/-- The largest positive two-digit multiple of 5 -/
def largest : ℕ := 95

/-- The number of positive two-digit multiples of 5 -/
def count : ℕ := (largest - smallest) / 5 + 1

/-- The arithmetic mean of all positive two-digit multiples of 5 -/
def arithmetic_mean : ℚ := (count : ℚ)⁻¹ * ((smallest + largest : ℚ) * (count : ℚ) / 2)

/-- Theorem stating that the arithmetic mean of all positive two-digit multiples of 5 is 52.5 -/
theorem two_digit_multiples_of_five_mean : arithmetic_mean = 52.5 := by sorry

end NUMINAMATH_CALUDE_two_digit_multiples_of_five_mean_l1798_179810


namespace NUMINAMATH_CALUDE_jim_reading_pages_l1798_179830

/-- Calculates the number of pages Jim reads per week after changing his reading speed and time --/
def pages_read_per_week (
  regular_rate : ℝ)
  (technical_rate : ℝ)
  (regular_time : ℝ)
  (technical_time : ℝ)
  (regular_speed_increase : ℝ)
  (technical_speed_increase : ℝ)
  (regular_time_reduction : ℝ)
  (technical_time_reduction : ℝ) : ℝ :=
  let new_regular_rate := regular_rate * regular_speed_increase
  let new_technical_rate := technical_rate * technical_speed_increase
  let new_regular_time := regular_time - regular_time_reduction
  let new_technical_time := technical_time - technical_time_reduction
  (new_regular_rate * new_regular_time) + (new_technical_rate * new_technical_time)

theorem jim_reading_pages : 
  pages_read_per_week 40 30 10 5 1.5 1.3 4 2 = 477 := by
  sorry

end NUMINAMATH_CALUDE_jim_reading_pages_l1798_179830


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l1798_179887

theorem sqrt_2x_minus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 4) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l1798_179887


namespace NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_8_times_sqrt_4_l1798_179806

theorem fourth_root_256_times_cube_root_8_times_sqrt_4 : 
  (256 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (4 : ℝ) ^ (1/2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_8_times_sqrt_4_l1798_179806


namespace NUMINAMATH_CALUDE_max_min_values_l1798_179875

theorem max_min_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 4 * a + b = 3) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = 3 → b - 1 / a ≥ y - 1 / x) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = 3 → 1 / (3 * a + 1) + 1 / (a + b) ≤ 1 / (3 * x + 1) + 1 / (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_l1798_179875


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1798_179838

def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {1, 2, 3}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1798_179838


namespace NUMINAMATH_CALUDE_painting_time_equation_l1798_179832

theorem painting_time_equation (t : ℝ) : t > 0 → t = 5/2 := by
  intro h
  have alice_rate : ℝ := 1/4
  have bob_rate : ℝ := 1/6
  have charlie_rate : ℝ := 1/12
  have combined_rate : ℝ := alice_rate + bob_rate + charlie_rate
  have break_time : ℝ := 1/2
  have painting_equation : (combined_rate * (t - break_time) = 1) := by sorry
  sorry

end NUMINAMATH_CALUDE_painting_time_equation_l1798_179832


namespace NUMINAMATH_CALUDE_power_function_k_values_l1798_179885

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

theorem power_function_k_values (k : ℝ) :
  is_power_function (λ x => (k^2 - k - 5) * x^3) → k = 3 ∨ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_k_values_l1798_179885


namespace NUMINAMATH_CALUDE_cosine_one_third_irrational_l1798_179850

theorem cosine_one_third_irrational (a : ℝ) (h : Real.cos (π * a) = (1 : ℝ) / 3) : 
  Irrational a := by sorry

end NUMINAMATH_CALUDE_cosine_one_third_irrational_l1798_179850


namespace NUMINAMATH_CALUDE_sine_cosine_ratio_equals_tangent_l1798_179843

theorem sine_cosine_ratio_equals_tangent :
  (Real.sin (10 * π / 180) + Real.sin (20 * π / 180)) / 
  (Real.cos (10 * π / 180) + Real.cos (20 * π / 180)) = 
  Real.tan (15 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_ratio_equals_tangent_l1798_179843


namespace NUMINAMATH_CALUDE_probability_second_odd_given_first_odd_l1798_179864

theorem probability_second_odd_given_first_odd (n : ℕ) (odds evens : ℕ) 
  (h1 : n = odds + evens)
  (h2 : n = 9)
  (h3 : odds = 5)
  (h4 : evens = 4) :
  (odds - 1) / (n - 1) = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_probability_second_odd_given_first_odd_l1798_179864


namespace NUMINAMATH_CALUDE_equation_solution_l1798_179846

theorem equation_solution : 
  let f (x : ℝ) := 1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
                   1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5))
  ∀ x : ℝ, f x = 1 / 10 ↔ x = 10 ∨ x = -3.5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1798_179846


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l1798_179890

theorem fifteenth_student_age
  (total_students : Nat)
  (average_age : ℚ)
  (group1_size group2_size : Nat)
  (group1_average group2_average : ℚ)
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_size = 7)
  (h4 : group2_size = 7)
  (h5 : group1_average = 14)
  (h6 : group2_average = 16) :
  (total_students * average_age - (group1_size * group1_average + group2_size * group2_average)) / (total_students - group1_size - group2_size) = 15 := by
  sorry


end NUMINAMATH_CALUDE_fifteenth_student_age_l1798_179890


namespace NUMINAMATH_CALUDE_acid_dilution_l1798_179880

/-- Given a p% solution of acid with volume p ounces (where p > 45),
    adding y ounces of water to create a (2p/3)% solution results in y = p/2 -/
theorem acid_dilution (p : ℝ) (y : ℝ) (h₁ : p > 45) :
  (p^2 / 100 = (2 * p / 300) * (p + y)) → y = p / 2 := by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l1798_179880


namespace NUMINAMATH_CALUDE_cube_of_negative_half_x_squared_y_l1798_179803

theorem cube_of_negative_half_x_squared_y (x y : ℝ) : 
  (-1/2 * x^2 * y)^3 = -1/8 * x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_half_x_squared_y_l1798_179803


namespace NUMINAMATH_CALUDE_difference_sum_of_powers_of_three_l1798_179839

def S : Finset ℕ := Finset.range 11

def difference_sum (S : Finset ℕ) : ℕ :=
  S.sum (λ i => S.sum (λ j => if i < j then 3^j - 3^i else 0))

theorem difference_sum_of_powers_of_three : difference_sum S = 787484 := by
  sorry

end NUMINAMATH_CALUDE_difference_sum_of_powers_of_three_l1798_179839


namespace NUMINAMATH_CALUDE_expression_simplification_l1798_179813

theorem expression_simplification (a b : ℝ) (h : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b - b^2) / (a - b) = (a^3 - 3*a*b + b^3) / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1798_179813


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l1798_179837

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_consecutive_nonprime (start : ℕ) : Prop :=
  ∀ i : ℕ, i < 7 → ¬(is_prime (start + i))

theorem smallest_prime_after_seven_nonprimes :
  ∃ start : ℕ,
    is_consecutive_nonprime start ∧
    is_prime 97 ∧
    (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ p > start + 6)) :=
  sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l1798_179837


namespace NUMINAMATH_CALUDE_first_last_gender_l1798_179860

/-- Represents the gender of a person in line -/
inductive Gender
  | Man
  | Woman

/-- Represents the state of bottle passing -/
structure BottlePassing where
  total_people : Nat
  woman_to_woman : Nat
  woman_to_man : Nat
  man_to_man : Nat

/-- Theorem stating the first and last person's gender based on bottle passing information -/
theorem first_last_gender (bp : BottlePassing) 
  (h1 : bp.total_people = 16)
  (h2 : bp.woman_to_woman = 4)
  (h3 : bp.woman_to_man = 3)
  (h4 : bp.man_to_man = 6) :
  (Gender.Woman, Gender.Man) = 
    (match bp.total_people with
      | 0 => (Gender.Woman, Gender.Man)  -- Arbitrary choice for empty line
      | n + 1 => 
        let first := if bp.woman_to_woman + bp.woman_to_man > bp.man_to_man + (n - (bp.woman_to_woman + bp.woman_to_man + bp.man_to_man)) 
                     then Gender.Woman else Gender.Man
        let last := if bp.man_to_man + (n - (bp.woman_to_woman + bp.woman_to_man + bp.man_to_man)) > bp.woman_to_woman + bp.woman_to_man 
                    then Gender.Man else Gender.Woman
        (first, last)
    ) :=
by
  sorry


end NUMINAMATH_CALUDE_first_last_gender_l1798_179860


namespace NUMINAMATH_CALUDE_spanish_test_score_difference_l1798_179896

theorem spanish_test_score_difference (average_score : ℝ) (marco_percentage : ℝ) (margaret_score : ℝ) :
  average_score = 90 ∧
  marco_percentage = 10 ∧
  margaret_score = 86 →
  margaret_score - (average_score * (1 - marco_percentage / 100)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_spanish_test_score_difference_l1798_179896


namespace NUMINAMATH_CALUDE_correct_calculation_l1798_179845

theorem correct_calculation : (-9)^2 / (-3)^2 = -9 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1798_179845


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1798_179828

theorem rectangle_diagonal (l w : ℝ) (h_area : l * w = 20) (h_perimeter : 2 * l + 2 * w = 18) :
  l^2 + w^2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1798_179828


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l1798_179879

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (36^2 + 45^2) ∧ ∀ q : ℕ, Nat.Prime q → q ∣ (36^2 + 45^2) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l1798_179879


namespace NUMINAMATH_CALUDE_line_intersection_with_x_axis_l1798_179821

/-- A line parallel to y = -3x that passes through (0, -2) intersects the x-axis at (-2/3, 0) -/
theorem line_intersection_with_x_axis :
  ∀ (k b : ℝ),
  (∀ x y : ℝ, y = k * x + b ↔ y = -3 * x + b) →  -- Line is parallel to y = -3x
  -2 = k * 0 + b →                               -- Line passes through (0, -2)
  ∃ x : ℝ, x = -2/3 ∧ 0 = k * x + b :=           -- Intersection point with x-axis
by sorry

end NUMINAMATH_CALUDE_line_intersection_with_x_axis_l1798_179821


namespace NUMINAMATH_CALUDE_big_eighteen_game_count_l1798_179852

/-- Calculates the total number of games in a basketball conference. -/
def total_conference_games (num_divisions : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let total_teams := num_divisions * teams_per_division
  let intra_division_total := num_divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_division_total := (total_teams * (total_teams - teams_per_division) * inter_division_games) / 2
  intra_division_total + inter_division_total

/-- The Big Eighteen Basketball Conference game count theorem -/
theorem big_eighteen_game_count : 
  total_conference_games 3 6 3 2 = 486 := by
  sorry

end NUMINAMATH_CALUDE_big_eighteen_game_count_l1798_179852


namespace NUMINAMATH_CALUDE_emilys_friends_with_color_boxes_l1798_179883

def rainbow_colors : ℕ := 7
def total_pencils : ℕ := 56

theorem emilys_friends_with_color_boxes :
  ∀ (pencils_per_box : ℕ) (total_boxes : ℕ),
    pencils_per_box = rainbow_colors →
    total_pencils = pencils_per_box * total_boxes →
    total_boxes - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_emilys_friends_with_color_boxes_l1798_179883


namespace NUMINAMATH_CALUDE_valid_numbers_l1798_179834

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000 ∧ n < 1000000) ∧
  (∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6] → ∃! p : ℕ, p < 6 ∧ (n / 10^p) % 10 = d) ∧
  (n / 10000) % 2 = 0 ∧
  (n / 10000) % 3 = 0 ∧
  (n / 100) % 4 = 0 ∧
  (n / 10) % 5 = 0 ∧
  n % 6 = 0

theorem valid_numbers : 
  {n : ℕ | is_valid_number n} = {123654, 321654} :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_l1798_179834


namespace NUMINAMATH_CALUDE_stephanie_orange_spending_l1798_179854

def num_visits : Nat := 8
def oranges_per_visit : Nat := 2

def prices : List Float := [0.50, 0.60, 0.55, 0.65, 0.70, 0.55, 0.50, 0.60]

theorem stephanie_orange_spending :
  prices.length = num_visits →
  (prices.map (· * oranges_per_visit.toFloat)).sum = 9.30 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_orange_spending_l1798_179854


namespace NUMINAMATH_CALUDE_degree_of_g_l1798_179835

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := -9 * x^5 + 4 * x^3 + 2 * x - 6

-- Define a proposition for the degree of a polynomial
def hasDegree (p : ℝ → ℝ) (n : ℕ) : Prop := sorry

-- State the theorem
theorem degree_of_g 
  (g : ℝ → ℝ) 
  (h : hasDegree (fun x => f x + g x) 2) : 
  hasDegree g 5 := by sorry

end NUMINAMATH_CALUDE_degree_of_g_l1798_179835


namespace NUMINAMATH_CALUDE_divisibility_conditions_l1798_179891

theorem divisibility_conditions (a b : ℕ) : 
  (∃ k : ℤ, a^3 * b - 1 = k * (a + 1)) ∧ 
  (∃ m : ℤ, a * b^3 + 1 = m * (b - 1)) → 
  ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_conditions_l1798_179891


namespace NUMINAMATH_CALUDE_inequality_proof_l1798_179859

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hx1 : x < 1) (hy : 0 < y) (hy1 : y < 1) :
  (x^2 / (x + y)) + (y^2 / (1 - x)) + ((1 - x - y)^2 / (1 - y)) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1798_179859


namespace NUMINAMATH_CALUDE_motel_monthly_charge_l1798_179865

theorem motel_monthly_charge 
  (weeks_per_month : ℕ)
  (num_months : ℕ)
  (weekly_rate : ℕ)
  (total_savings : ℕ)
  (h1 : weeks_per_month = 4)
  (h2 : num_months = 3)
  (h3 : weekly_rate = 280)
  (h4 : total_savings = 360) :
  (num_months * weeks_per_month * weekly_rate - total_savings) / num_months = 1000 := by
  sorry

end NUMINAMATH_CALUDE_motel_monthly_charge_l1798_179865


namespace NUMINAMATH_CALUDE_even_expression_l1798_179847

theorem even_expression (n : ℕ) (h : n = 101) : Even (2 * n - 2) := by
  sorry

end NUMINAMATH_CALUDE_even_expression_l1798_179847


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l1798_179841

theorem cubic_polynomials_common_roots :
  ∃! (a b : ℝ), 
    (∃ (r s : ℝ) (h : r ≠ s), 
      (∀ x : ℝ, x^3 + a*x^2 + 14*x + 7 = 0 ↔ x = r ∨ x = s ∨ x^3 + a*x^2 + 14*x + 7 = 0) ∧
      (∀ x : ℝ, x^3 + b*x^2 + 21*x + 15 = 0 ↔ x = r ∨ x = s ∨ x^3 + b*x^2 + 21*x + 15 = 0)) ∧
    a = 5 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l1798_179841


namespace NUMINAMATH_CALUDE_hawks_percentage_l1798_179867

/-- Represents the percentages of different bird types in a nature reserve -/
structure BirdReserve where
  hawks : ℝ
  paddyfieldWarblers : ℝ
  kingfishers : ℝ
  others : ℝ

/-- The conditions of the bird reserve problem -/
def validBirdReserve (b : BirdReserve) : Prop :=
  b.paddyfieldWarblers = 0.4 * (100 - b.hawks) ∧
  b.kingfishers = 0.25 * b.paddyfieldWarblers ∧
  b.others = 35 ∧
  b.hawks + b.paddyfieldWarblers + b.kingfishers + b.others = 100

/-- The theorem stating that hawks make up 30% of the birds in a valid bird reserve -/
theorem hawks_percentage (b : BirdReserve) (h : validBirdReserve b) : b.hawks = 30 := by
  sorry

end NUMINAMATH_CALUDE_hawks_percentage_l1798_179867


namespace NUMINAMATH_CALUDE_smallest_perfect_cube_divisor_l1798_179876

theorem smallest_perfect_cube_divisor (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) : 
  let n := p^2 * q^3 * r^5
  ∀ m : ℕ, m > 0 → (∃ k : ℕ, m = k^3) → n ∣ m → p^6 * q^9 * r^15 ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_cube_divisor_l1798_179876


namespace NUMINAMATH_CALUDE_function_zero_in_interval_l1798_179840

theorem function_zero_in_interval (a : ℝ) : 
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ a^2 * x - 2*a + 1 = 0) ↔ 
  a ∈ (Set.Ioo (1/2) 1) ∪ (Set.Ioi 1) := by
sorry

end NUMINAMATH_CALUDE_function_zero_in_interval_l1798_179840


namespace NUMINAMATH_CALUDE_first_group_size_l1798_179804

/-- Represents the number of questions Cameron answers per tourist -/
def questions_per_tourist : ℕ := 2

/-- Represents the total number of tour groups -/
def total_groups : ℕ := 4

/-- Represents the number of people in the second group -/
def second_group : ℕ := 11

/-- Represents the number of people in the third group -/
def third_group : ℕ := 8

/-- Represents the number of people in the fourth group -/
def fourth_group : ℕ := 7

/-- Represents the total number of questions Cameron answered -/
def total_questions : ℕ := 68

/-- Proves that the number of people in the first tour group is 8 -/
theorem first_group_size : ℕ := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l1798_179804


namespace NUMINAMATH_CALUDE_min_value_of_f_l1798_179899

-- Define the function f
def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

-- State the theorem
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1798_179899


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1798_179888

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 2 → x^2 ≥ 4) ↔ (∃ x : ℝ, x ≥ 2 ∧ x^2 < 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1798_179888


namespace NUMINAMATH_CALUDE_convex_regular_polygon_integer_angles_l1798_179895

/-- The number of positive integers n ≥ 3 such that 360 is divisible by n -/
def count_divisors : Nat :=
  (Finset.filter (fun n => n ≥ 3 ∧ 360 % n = 0) (Finset.range 361)).card

/-- Theorem stating that there are exactly 22 positive integers n ≥ 3 
    such that 360 is divisible by n -/
theorem convex_regular_polygon_integer_angles : count_divisors = 22 := by
  sorry

end NUMINAMATH_CALUDE_convex_regular_polygon_integer_angles_l1798_179895


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_prime_angles_l1798_179823

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def consecutive_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p < q ∧ ∀ m, p < m → m < q → ¬is_prime m

theorem right_triangle_consecutive_prime_angles (p q : ℕ) :
  p < q →
  consecutive_primes p q →
  p + q = 90 →
  (∀ p' q' : ℕ, p' < q' → consecutive_primes p' q' → p' + q' = 90 → p ≤ p') →
  p = 43 := by sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_prime_angles_l1798_179823


namespace NUMINAMATH_CALUDE_sum_x_coordinates_invariant_l1798_179897

/-- Represents a polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Creates a new polygon by finding midpoints of edges of the given polygon -/
def midpointTransform (p : Polygon) : Polygon :=
  sorry

/-- Calculates the sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

/-- Theorem: The sum of x-coordinates remains constant after two midpoint transformations -/
theorem sum_x_coordinates_invariant (Q₁ : Polygon) 
  (h : sumXCoordinates Q₁ = 132) 
  (h_vertices : Q₁.vertices.length = 44) : 
  sumXCoordinates (midpointTransform (midpointTransform Q₁)) = 132 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_invariant_l1798_179897


namespace NUMINAMATH_CALUDE_tangent_lines_theorem_l1798_179884

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

theorem tangent_lines_theorem :
  let l1 : ℝ → ℝ → Prop := λ x y => x - y - 2 = 0
  let l2 : ℝ → ℝ → Prop := λ x y => x + y + 3 = 0
  (∀ x, deriv f x = 2*x + 1) ∧
  (l1 0 (-2)) ∧
  (∃ a b, f a = b ∧ l2 a b) ∧
  (∀ x y, l1 x y → ∀ x' y', l2 x' y' → (y - (-2)) / (x - 0) * (y' - y) / (x' - x) = -1) →
  (∀ x y, l1 x y ↔ (x - y - 2 = 0)) ∧
  (∀ x y, l2 x y ↔ (x + y + 3 = 0))
:= by sorry

end NUMINAMATH_CALUDE_tangent_lines_theorem_l1798_179884


namespace NUMINAMATH_CALUDE_expand_expression_l1798_179816

theorem expand_expression (x : ℝ) : (1 + x^2) * (1 - x^4) = 1 + x^2 - x^4 - x^6 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1798_179816


namespace NUMINAMATH_CALUDE_uncle_bradley_money_l1798_179807

theorem uncle_bradley_money (M : ℚ) (F H : ℕ) : 
  F + H = 13 →
  50 * F = (3 / 10) * M →
  100 * H = (7 / 10) * M →
  M = 1300 := by
sorry

end NUMINAMATH_CALUDE_uncle_bradley_money_l1798_179807


namespace NUMINAMATH_CALUDE_not_perfect_square_l1798_179829

theorem not_perfect_square (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) :
  ¬ ∃ k : ℕ, 3^m + 3^n + 1 = k^2 := by
sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1798_179829


namespace NUMINAMATH_CALUDE_course_selection_ways_l1798_179815

theorem course_selection_ways (type_a : ℕ) (type_b : ℕ) (total_selection : ℕ) : 
  type_a = 4 → type_b = 2 → total_selection = 3 →
  (Nat.choose type_a 1 * Nat.choose type_b 2) + (Nat.choose type_a 2 * Nat.choose type_b 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_ways_l1798_179815


namespace NUMINAMATH_CALUDE_glenn_total_expenditure_l1798_179881

/-- Represents the cost of movie tickets and concessions -/
structure MovieCosts where
  monday_ticket : ℕ
  wednesday_ticket : ℕ
  saturday_ticket : ℕ
  concession : ℕ

/-- Represents discount percentages -/
structure Discounts where
  wednesday : ℕ
  group : ℕ

/-- Represents the number of people in Glenn's group for each day -/
structure GroupSize where
  wednesday : ℕ
  saturday : ℕ

/-- Calculates the total cost of Glenn's movie outings -/
def calculate_total_cost (costs : MovieCosts) (discounts : Discounts) (group : GroupSize) : ℕ :=
  let wednesday_cost := costs.wednesday_ticket * (100 - discounts.wednesday) / 100 * group.wednesday
  let saturday_cost := costs.saturday_ticket * group.saturday + costs.concession
  wednesday_cost + saturday_cost

/-- Theorem stating that Glenn's total expenditure is $93 -/
theorem glenn_total_expenditure (costs : MovieCosts) (discounts : Discounts) (group : GroupSize) :
  costs.monday_ticket = 5 →
  costs.wednesday_ticket = 2 * costs.monday_ticket →
  costs.saturday_ticket = 5 * costs.monday_ticket →
  costs.concession = 7 →
  discounts.wednesday = 10 →
  discounts.group = 20 →
  group.wednesday = 4 →
  group.saturday = 2 →
  calculate_total_cost costs discounts group = 93 := by
  sorry


end NUMINAMATH_CALUDE_glenn_total_expenditure_l1798_179881


namespace NUMINAMATH_CALUDE_L_intersects_C_twice_L_min_chord_correct_l1798_179826

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line L
def L (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Statement 1: L always intersects C at two points for any real m
theorem L_intersects_C_twice : ∀ m : ℝ, ∃! (p q : ℝ × ℝ), 
  p ≠ q ∧ C p.1 p.2 ∧ C q.1 q.2 ∧ L m p.1 p.2 ∧ L m q.1 q.2 :=
sorry

-- Statement 2: Equation of L with minimum chord length
def L_min_chord (x y : ℝ) : Prop := 2*x - y - 5 = 0

theorem L_min_chord_correct : 
  (∀ m : ℝ, ∃ (p q : ℝ × ℝ), p ≠ q ∧ C p.1 p.2 ∧ C q.1 q.2 ∧ L m p.1 p.2 ∧ L m q.1 q.2 ∧ 
    ∀ (r s : ℝ × ℝ), r ≠ s ∧ C r.1 r.2 ∧ C s.1 s.2 ∧ L_min_chord r.1 r.2 ∧ L_min_chord s.1 s.2 →
      (p.1 - q.1)^2 + (p.2 - q.2)^2 ≥ (r.1 - s.1)^2 + (r.2 - s.2)^2) ∧
  (∃ (p q : ℝ × ℝ), p ≠ q ∧ C p.1 p.2 ∧ C q.1 q.2 ∧ L_min_chord p.1 p.2 ∧ L_min_chord q.1 q.2) :=
sorry

end NUMINAMATH_CALUDE_L_intersects_C_twice_L_min_chord_correct_l1798_179826


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1798_179811

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (a 1 + a n) * n / 2

/-- Theorem: For an arithmetic sequence where S₁₅ - S₁₀ = 1, S₂₅ = 5 -/
theorem arithmetic_sequence_sum
  (seq : ArithmeticSequence)
  (h : seq.S 15 - seq.S 10 = 1) :
  seq.S 25 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1798_179811


namespace NUMINAMATH_CALUDE_remainder_theorem_l1798_179873

theorem remainder_theorem (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 19) → 
  (∃ m : ℤ, N = 13 * m + 6) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1798_179873


namespace NUMINAMATH_CALUDE_max_min_value_l1798_179814

theorem max_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let s := min x (min (y + 1/x) (1/y))
  ∃ (max_s : ℝ), max_s = Real.sqrt 2 ∧ 
    (∀ x' y' : ℝ, x' > 0 → y' > 0 → min x' (min (y' + 1/x') (1/y')) ≤ max_s) ∧
    (s = max_s ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_min_value_l1798_179814


namespace NUMINAMATH_CALUDE_arithmetic_mean_odd_eq_n_l1798_179818

/-- The sum of the first n odd positive integers -/
def sum_first_n_odd (n : ℕ) : ℕ := n^2

/-- The arithmetic mean of the first n odd positive integers -/
def arithmetic_mean_odd (n : ℕ) : ℚ := (sum_first_n_odd n : ℚ) / n

/-- Theorem: The arithmetic mean of the first n odd positive integers is equal to n -/
theorem arithmetic_mean_odd_eq_n (n : ℕ) (h : n > 0) : 
  arithmetic_mean_odd n = n := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_odd_eq_n_l1798_179818


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l1798_179868

noncomputable def f (x : ℝ) := Real.sin x - Real.cos x + x + 1

theorem f_monotonicity_and_extrema :
  ∀ x : ℝ, 0 < x → x < 2 * Real.pi →
  (∀ y : ℝ, 0 < y → y < Real.pi → HasDerivAt f (Real.cos y + Real.sin y + 1) y) ∧
  (∀ y : ℝ, Real.pi < y → y < 3 * Real.pi / 2 → HasDerivAt f (Real.cos y + Real.sin y + 1) y) ∧
  (∀ y : ℝ, 3 * Real.pi / 2 < y → y < 2 * Real.pi → HasDerivAt f (Real.cos y + Real.sin y + 1) y) ∧
  (f (3 * Real.pi / 2) = 3 * Real.pi / 2) ∧
  (f Real.pi = Real.pi + 2) ∧
  (∀ y : ℝ, 0 < y → y < 2 * Real.pi → f y ≥ 3 * Real.pi / 2) ∧
  (∀ y : ℝ, 0 < y → y < 2 * Real.pi → f y ≤ Real.pi + 2) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l1798_179868


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1798_179831

/-- The complex number (i-3)/(1+i) corresponds to a point in the second quadrant of the complex plane -/
theorem complex_number_in_second_quadrant :
  ∃ (z : ℂ), z = (Complex.I - 3) / (1 + Complex.I) ∧
  Complex.re z < 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1798_179831


namespace NUMINAMATH_CALUDE_exists_triangle_area_not_greater_than_two_l1798_179872

/-- A lattice point in a 2D coordinate system -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Checks if a lattice point is within the 5x5 grid centered at the origin -/
def isWithinGrid (p : LatticePoint) : Prop :=
  |p.x| ≤ 2 ∧ |p.y| ≤ 2

/-- Checks if three points are collinear -/
def areCollinear (p1 p2 p3 : LatticePoint) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Calculates the area of a triangle formed by three lattice points -/
def triangleArea (p1 p2 p3 : LatticePoint) : ℚ :=
  |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)| / 2

/-- Main theorem statement -/
theorem exists_triangle_area_not_greater_than_two 
  (points : Fin 6 → LatticePoint)
  (h_within_grid : ∀ i, isWithinGrid (points i))
  (h_not_collinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬areCollinear (points i) (points j) (points k)) :
  ∃ i j k, i ≠ j → j ≠ k → i ≠ k → triangleArea (points i) (points j) (points k) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_triangle_area_not_greater_than_two_l1798_179872


namespace NUMINAMATH_CALUDE_girls_count_in_school_l1798_179874

/-- Proves that in a school with a given total number of students and a ratio of boys to girls,
    the number of girls is as calculated. -/
theorem girls_count_in_school (total : ℕ) (boys_ratio girls_ratio : ℕ) 
    (h_total : total = 480) 
    (h_ratio : boys_ratio = 3 ∧ girls_ratio = 5) : 
    (girls_ratio * total) / (boys_ratio + girls_ratio) = 300 := by
  sorry

end NUMINAMATH_CALUDE_girls_count_in_school_l1798_179874


namespace NUMINAMATH_CALUDE_existence_of_least_t_for_geometric_progression_l1798_179824

open Real

theorem existence_of_least_t_for_geometric_progression :
  ∃ t : ℝ, t > 0 ∧
  ∃ α : ℝ, 0 < α ∧ α < π / 3 ∧
  ∃ r : ℝ, r > 0 ∧
  (arcsin (sin α) = α) ∧
  (arcsin (sin (3 * α)) = r * α) ∧
  (arcsin (sin (8 * α)) = r^2 * α) ∧
  (arcsin (sin (t * α)) = r^3 * α) ∧
  ∀ s : ℝ, s > 0 →
    (∃ β : ℝ, 0 < β ∧ β < π / 3 ∧
    ∃ q : ℝ, q > 0 ∧
    (arcsin (sin β) = β) ∧
    (arcsin (sin (3 * β)) = q * β) ∧
    (arcsin (sin (8 * β)) = q^2 * β) ∧
    (arcsin (sin (s * β)) = q^3 * β)) →
    t ≤ s :=
by sorry

end NUMINAMATH_CALUDE_existence_of_least_t_for_geometric_progression_l1798_179824


namespace NUMINAMATH_CALUDE_mountain_paths_l1798_179882

/-- Given a mountain with paths from east and west sides, calculate the total number of ways to ascend and descend -/
theorem mountain_paths (east_paths west_paths : ℕ) : 
  east_paths = 3 → west_paths = 2 → (east_paths + west_paths) * (east_paths + west_paths) = 25 := by
  sorry

#check mountain_paths

end NUMINAMATH_CALUDE_mountain_paths_l1798_179882


namespace NUMINAMATH_CALUDE_no_positive_solution_l1798_179855

theorem no_positive_solution :
  ¬ ∃ (x : ℝ), x > 0 ∧ (Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = 2 * Real.log 9 / Real.log 4 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_solution_l1798_179855


namespace NUMINAMATH_CALUDE_art_gallery_visitors_prove_initial_girls_l1798_179877

theorem art_gallery_visitors : ℕ → ℕ → Prop :=
  fun girls boys =>
    -- After 15 girls left, there were twice as many boys as girls remaining
    boys = 2 * (girls - 15) ∧
    -- After 45 boys left, there were five times as many girls as boys remaining
    (girls - 15) = 5 * (boys - 45) ∧
    -- The number of girls initially in the gallery is 40
    girls = 40

-- The theorem to prove
theorem prove_initial_girls : ∃ (girls boys : ℕ), art_gallery_visitors girls boys :=
  sorry

end NUMINAMATH_CALUDE_art_gallery_visitors_prove_initial_girls_l1798_179877


namespace NUMINAMATH_CALUDE_complete_square_sum_l1798_179870

theorem complete_square_sum (x : ℝ) : ∃ (a b c : ℤ), 
  (49 * x^2 + 70 * x - 81 = 0 ↔ (a * x + b)^2 = c) ∧ 
  a > 0 ∧ 
  a + b + c = -44 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1798_179870


namespace NUMINAMATH_CALUDE_solve_equation_l1798_179825

theorem solve_equation (y : ℚ) (h : (2 / 7) * (1 / 5) * y = 4) : y = 70 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1798_179825


namespace NUMINAMATH_CALUDE_m_range_l1798_179863

def f (x : ℝ) := x^3 + x

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < π/2 → f (m * Real.sin θ) + f (1 - m) > 0) →
  m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l1798_179863


namespace NUMINAMATH_CALUDE_fraction_integer_iff_p_in_set_l1798_179827

theorem fraction_integer_iff_p_in_set (p : ℕ) (hp : p > 0) :
  (∃ k : ℕ, k > 0 ∧ (4 * p + 34 : ℚ) / (3 * p - 8 : ℚ) = k) ↔ p ∈ ({3, 4, 5, 12} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_p_in_set_l1798_179827


namespace NUMINAMATH_CALUDE_root_product_zero_l1798_179866

theorem root_product_zero (x₁ x₂ x₃ : ℝ) :
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (Real.sqrt 2025) * x₁^3 - 4050 * x₁^2 - 4 = 0 ∧
  (Real.sqrt 2025) * x₂^3 - 4050 * x₂^2 - 4 = 0 ∧
  (Real.sqrt 2025) * x₃^3 - 4050 * x₃^2 - 4 = 0 →
  x₂ * (x₁ + x₃) = 0 := by
sorry

end NUMINAMATH_CALUDE_root_product_zero_l1798_179866


namespace NUMINAMATH_CALUDE_well_depth_rope_length_l1798_179898

/-- 
Given a well of unknown depth and a rope of unknown length, prove that if:
1) Folding the rope three times and lowering it into the well leaves 4 feet outside
2) Folding the rope four times and lowering it into the well leaves 1 foot outside
Then the depth of the well (x) and the length of the rope (h) satisfy the system of equations:
{h/3 = x + 4, h/4 = x + 1}
-/
theorem well_depth_rope_length (x h : ℝ) 
  (h_positive : h > 0) 
  (fold_three : h / 3 = x + 4) 
  (fold_four : h / 4 = x + 1) : 
  h / 3 = x + 4 ∧ h / 4 = x + 1 := by
sorry


end NUMINAMATH_CALUDE_well_depth_rope_length_l1798_179898


namespace NUMINAMATH_CALUDE_f_two_l1798_179892

/-- A linear function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The inverse function of f -/
def f_inv (x : ℝ) : ℝ := sorry

/-- f is a linear function -/
axiom f_linear : ∃ (a b : ℝ), ∀ x, f x = a * x + b

/-- f satisfies the equation f(x) = 3f^(-1)(x) + 5 -/
axiom f_equation : ∀ x, f x = 3 * f_inv x + 5

/-- f(1) = 5 -/
axiom f_one : f 1 = 5

/-- The main theorem: f(2) = 3 -/
theorem f_two : f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_f_two_l1798_179892


namespace NUMINAMATH_CALUDE_bus_breakdown_time_correct_l1798_179878

/-- Represents the scenario of a school trip with a bus breakdown -/
structure BusBreakdown where
  S : ℝ  -- Distance between school and county town in km
  x : ℝ  -- Walking speed in km/minute
  t : ℝ  -- Walking time of teachers and students in minutes
  a : ℝ  -- Bus breakdown time in minutes

/-- The bus speed is 5 times the walking speed -/
def bus_speed (bd : BusBreakdown) : ℝ := 5 * bd.x

/-- The walking time satisfies the equation derived from the problem conditions -/
def walking_time_equation (bd : BusBreakdown) : Prop :=
  bd.t = bd.S / (5 * bd.x) + 20 - (bd.S - bd.x * bd.t) / (5 * bd.x)

/-- The bus breakdown time satisfies the equation derived from the problem conditions -/
def breakdown_time_equation (bd : BusBreakdown) : Prop :=
  bd.a + (2 * (bd.S - bd.x * bd.t)) / (5 * bd.x) = (2 * bd.S) / (5 * bd.x) + 30

/-- Theorem stating that given the conditions, the bus breakdown time equation holds -/
theorem bus_breakdown_time_correct (bd : BusBreakdown) 
  (h_walking_time : walking_time_equation bd) :
  breakdown_time_equation bd :=
sorry

end NUMINAMATH_CALUDE_bus_breakdown_time_correct_l1798_179878


namespace NUMINAMATH_CALUDE_molly_rode_3285_miles_l1798_179812

/-- The number of miles Molly rode her bike from her 13th to 16th birthday -/
def molly_bike_miles : ℕ :=
  let start_age : ℕ := 13
  let end_age : ℕ := 16
  let years_riding : ℕ := end_age - start_age
  let days_per_year : ℕ := 365
  let miles_per_day : ℕ := 3
  years_riding * days_per_year * miles_per_day

/-- Theorem stating that Molly rode her bike for 3285 miles -/
theorem molly_rode_3285_miles : molly_bike_miles = 3285 := by
  sorry

end NUMINAMATH_CALUDE_molly_rode_3285_miles_l1798_179812


namespace NUMINAMATH_CALUDE_homologous_pair_from_both_parents_l1798_179889

/-- Represents a parent (mother or father) -/
inductive Parent : Type
| mother : Parent
| father : Parent

/-- Represents a chromosome -/
structure Chromosome : Type :=
  (source : Parent)

/-- Represents a pair of homologous chromosomes -/
structure HomologousPair : Type :=
  (chromosome1 : Chromosome)
  (chromosome2 : Chromosome)

/-- Represents a diploid cell -/
structure DiploidCell : Type :=
  (chromosomePairs : List HomologousPair)

/-- Theorem: In a diploid organism, each pair of homologous chromosomes
    is contributed jointly by the two parents -/
theorem homologous_pair_from_both_parents (cell : DiploidCell) :
  ∀ pair ∈ cell.chromosomePairs,
    (pair.chromosome1.source = Parent.mother ∧ pair.chromosome2.source = Parent.father) ∨
    (pair.chromosome1.source = Parent.father ∧ pair.chromosome2.source = Parent.mother) :=
sorry

end NUMINAMATH_CALUDE_homologous_pair_from_both_parents_l1798_179889


namespace NUMINAMATH_CALUDE_f_sum_zero_l1798_179820

-- Define the function f(x) = ax^2 + bx
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem f_sum_zero (a b x₁ x₂ : ℝ) 
  (h₁ : a * b ≠ 0) 
  (h₂ : f a b x₁ = f a b x₂) 
  (h₃ : x₁ ≠ x₂) : 
  f a b (x₁ + x₂) = 0 := by
sorry

end NUMINAMATH_CALUDE_f_sum_zero_l1798_179820


namespace NUMINAMATH_CALUDE_eggs_per_basket_l1798_179805

theorem eggs_per_basket (purple_eggs blue_eggs min_eggs : ℕ) 
  (h1 : purple_eggs = 30)
  (h2 : blue_eggs = 42)
  (h3 : min_eggs = 5) : 
  ∃ (eggs_per_basket : ℕ), 
    eggs_per_basket ≥ min_eggs ∧ 
    purple_eggs % eggs_per_basket = 0 ∧
    blue_eggs % eggs_per_basket = 0 ∧
    eggs_per_basket = 6 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_basket_l1798_179805


namespace NUMINAMATH_CALUDE_school_population_l1798_179862

theorem school_population (total_students : ℕ) : total_students = 400 :=
  sorry

end NUMINAMATH_CALUDE_school_population_l1798_179862


namespace NUMINAMATH_CALUDE_correct_swap_l1798_179836

def swap_values (a b : ℕ) : ℕ × ℕ :=
  let c := b
  let b' := a
  let a' := c
  (a', b')

theorem correct_swap :
  swap_values 6 5 = (5, 6) := by
sorry

end NUMINAMATH_CALUDE_correct_swap_l1798_179836


namespace NUMINAMATH_CALUDE_hispanic_west_percentage_l1798_179800

def hispanic_ne : ℕ := 10
def hispanic_mw : ℕ := 8
def hispanic_south : ℕ := 22
def hispanic_west : ℕ := 15

def total_hispanic : ℕ := hispanic_ne + hispanic_mw + hispanic_south + hispanic_west

def percent_in_west : ℚ := hispanic_west / total_hispanic * 100

theorem hispanic_west_percentage :
  round percent_in_west = 27 :=
sorry

end NUMINAMATH_CALUDE_hispanic_west_percentage_l1798_179800


namespace NUMINAMATH_CALUDE_third_card_value_l1798_179871

def sum_of_permutations (a b x : ℕ) : ℕ :=
  100000 * a + 10000 * b + 100 * x +
  100000 * a + 10000 * x + b +
  100000 * b + 10000 * a + x +
  100000 * b + 10000 * x + a +
  100000 * x + 10000 * b + a +
  100000 * x + 10000 * a + b

theorem third_card_value (x : ℕ) :
  x < 100 →
  sum_of_permutations 18 75 x = 2606058 →
  x = 36 := by
sorry

end NUMINAMATH_CALUDE_third_card_value_l1798_179871


namespace NUMINAMATH_CALUDE_stratified_sampling_red_balls_l1798_179869

/-- Given a set of 100 balls with 20 red balls, prove that a stratified sample of 10 balls should contain 2 red balls. -/
theorem stratified_sampling_red_balls 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (sample_size : ℕ) 
  (h_total : total_balls = 100) 
  (h_red : red_balls = 20) 
  (h_sample : sample_size = 10) : 
  (red_balls : ℚ) / total_balls * sample_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_red_balls_l1798_179869


namespace NUMINAMATH_CALUDE_solution_range_l1798_179802

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, 2 * (x + a) = x + 3 ∧ 2 * x - 10 > 8 * a) → 
  a < -1/3 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l1798_179802


namespace NUMINAMATH_CALUDE_binomial_150_150_l1798_179842

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_l1798_179842


namespace NUMINAMATH_CALUDE_bridge_length_l1798_179833

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 265 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l1798_179833


namespace NUMINAMATH_CALUDE_square_area_is_two_l1798_179858

/-- A complex number z is a vertex of a square with z^2 and z^4 if it satisfies the equation z^3 - iz + i - 1 = 0 -/
def is_square_vertex (z : ℂ) : Prop :=
  z ≠ 0 ∧ z^3 - Complex.I * z + Complex.I - 1 = 0

/-- The area of a square formed by z, z^2, and z^4 in the complex plane -/
noncomputable def square_area (z : ℂ) : ℝ :=
  (1/2) * Complex.abs (z^4 - z)^2

theorem square_area_is_two (z : ℂ) (h : is_square_vertex z) :
  square_area z = 2 :=
sorry

end NUMINAMATH_CALUDE_square_area_is_two_l1798_179858


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_l1798_179886

def base : ℕ := 3 + 4
def exponent : ℕ := 21

def last_two_digits (n : ℕ) : ℕ := n % 100

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_of_power :
  tens_digit (last_two_digits (base ^ exponent)) + ones_digit (last_two_digits (base ^ exponent)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_l1798_179886


namespace NUMINAMATH_CALUDE_money_distribution_l1798_179809

theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 → 
  a + c = 200 → 
  b + c = 350 → 
  c = 50 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1798_179809


namespace NUMINAMATH_CALUDE_time_to_install_one_window_l1798_179844

theorem time_to_install_one_window
  (total_windows : ℕ)
  (installed_windows : ℕ)
  (time_for_remaining : ℕ)
  (h1 : total_windows = 14)
  (h2 : installed_windows = 5)
  (h3 : time_for_remaining = 36)
  : (time_for_remaining : ℚ) / (total_windows - installed_windows : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_time_to_install_one_window_l1798_179844


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l1798_179848

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (x ≠ 1) →  -- ensure fraction is defined
  ((|x| - 1) / (x - 1) = 0) →
  x = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l1798_179848


namespace NUMINAMATH_CALUDE_share_calculation_l1798_179893

theorem share_calculation (total : ℚ) (a b c : ℚ) 
  (h_total : total = 578)
  (h_a : a = (2/3) * b)
  (h_b : b = (1/4) * c)
  (h_sum : a + b + c = total) :
  b = 102 := by
  sorry

end NUMINAMATH_CALUDE_share_calculation_l1798_179893


namespace NUMINAMATH_CALUDE_solution_system1_solution_system2_l1798_179851

-- Define the systems of equations
def system1 (x y : ℝ) : Prop := (3 * x + 2 * y = 5) ∧ (y = 2 * x - 8)
def system2 (x y : ℝ) : Prop := (2 * x - y = 10) ∧ (2 * x + 3 * y = 2)

-- Theorem for System 1
theorem solution_system1 : ∃ x y : ℝ, system1 x y ∧ x = 3 ∧ y = -2 := by
  sorry

-- Theorem for System 2
theorem solution_system2 : ∃ x y : ℝ, system2 x y ∧ x = 4 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_system1_solution_system2_l1798_179851


namespace NUMINAMATH_CALUDE_room_width_is_two_l1798_179849

-- Define the room's properties
def room_area : ℝ := 10
def room_length : ℝ := 5

-- Theorem statement
theorem room_width_is_two : 
  ∃ (width : ℝ), room_area = room_length * width ∧ width = 2 := by
  sorry

end NUMINAMATH_CALUDE_room_width_is_two_l1798_179849


namespace NUMINAMATH_CALUDE_elevator_is_translation_l1798_179861

/-- A structure representing a movement in space -/
structure Movement where
  is_straight_line : Bool

/-- Definition of translation in mathematics -/
def is_translation (m : Movement) : Prop :=
  m.is_straight_line = true

/-- Representation of an elevator's movement -/
def elevator_movement : Movement where
  is_straight_line := true

/-- Theorem stating that an elevator's movement is a translation -/
theorem elevator_is_translation : is_translation elevator_movement := by
  sorry

end NUMINAMATH_CALUDE_elevator_is_translation_l1798_179861


namespace NUMINAMATH_CALUDE_max_sum_of_vertex_products_l1798_179894

/-- Represents the set of numbers that can be assigned to cube faces -/
def CubeNumbers : Finset ℕ := {0, 1, 2, 3, 8, 9}

/-- A function that assigns numbers to cube faces -/
def FaceAssignment := Fin 6 → ℕ

/-- Predicate to check if a face assignment is valid -/
def ValidAssignment (f : FaceAssignment) : Prop :=
  (∀ i : Fin 6, f i ∈ CubeNumbers) ∧ (∀ i j : Fin 6, i ≠ j → f i ≠ f j)

/-- Calculate the product at a vertex given three face numbers -/
def VertexProduct (a b c : ℕ) : ℕ := a * b * c

/-- Calculate the sum of all vertex products for a given face assignment -/
def SumOfVertexProducts (f : FaceAssignment) : ℕ :=
  VertexProduct (f 0) (f 1) (f 2) +
  VertexProduct (f 0) (f 1) (f 3) +
  VertexProduct (f 0) (f 2) (f 4) +
  VertexProduct (f 0) (f 3) (f 4) +
  VertexProduct (f 1) (f 2) (f 5) +
  VertexProduct (f 1) (f 3) (f 5) +
  VertexProduct (f 2) (f 4) (f 5) +
  VertexProduct (f 3) (f 4) (f 5)

/-- The main theorem stating that the maximum sum of vertex products is 405 -/
theorem max_sum_of_vertex_products :
  ∃ (f : FaceAssignment), ValidAssignment f ∧
  SumOfVertexProducts f = 405 ∧
  ∀ (g : FaceAssignment), ValidAssignment g → SumOfVertexProducts g ≤ 405 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_vertex_products_l1798_179894


namespace NUMINAMATH_CALUDE_circle_properties_l1798_179822

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 6 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 6 = 0

-- Define the common chord equation
def common_chord (x y : ℝ) : Prop := 3*x - 2*y = 0

-- Theorem stating the properties of the circles
theorem circle_properties :
  -- The circles intersect
  (∃ x y : ℝ, C₁ x y ∧ C₂ x y) ∧
  -- The common chord equation is correct
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → common_chord x y) ∧
  -- The length of the common chord is (2√1182) / 13
  (let chord_length := (2 * Real.sqrt 1182) / 13
   ∃ x₁ y₁ x₂ y₂ : ℝ,
     C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧
     common_chord x₁ y₁ ∧ common_chord x₂ y₂ ∧
     Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = chord_length) :=
by sorry


end NUMINAMATH_CALUDE_circle_properties_l1798_179822


namespace NUMINAMATH_CALUDE_total_bird_wings_l1798_179808

/-- The number of birds in the sky -/
def num_birds : ℕ := 10

/-- The number of wings each bird has -/
def wings_per_bird : ℕ := 2

/-- Theorem: The total number of bird wings in the sky is 20 -/
theorem total_bird_wings : num_birds * wings_per_bird = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_bird_wings_l1798_179808


namespace NUMINAMATH_CALUDE_new_student_weight_l1798_179853

/-- Given 5 students, if replacing a 92 kg student with a new student
    causes the average weight to decrease by 4 kg,
    then the new student's weight is 72 kg. -/
theorem new_student_weight
  (n : Nat)
  (old_weight : Nat)
  (weight_decrease : Nat)
  (h1 : n = 5)
  (h2 : old_weight = 92)
  (h3 : weight_decrease = 4)
  : n * weight_decrease = old_weight - (old_weight - n * weight_decrease) :=
by
  sorry

#check new_student_weight

end NUMINAMATH_CALUDE_new_student_weight_l1798_179853


namespace NUMINAMATH_CALUDE_b_40_mod_49_l1798_179801

def b (n : ℕ) : ℤ := 5^n - 7^n

theorem b_40_mod_49 : b 40 ≡ 2 [ZMOD 49] := by sorry

end NUMINAMATH_CALUDE_b_40_mod_49_l1798_179801
