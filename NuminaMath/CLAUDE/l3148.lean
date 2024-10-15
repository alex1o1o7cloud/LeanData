import Mathlib

namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_l3148_314890

theorem coefficient_x3y5_in_expansion : 
  (Finset.range 9).sum (fun k => 
    if k = 3 then (Nat.choose 8 k : ℕ) 
    else 0) = 56 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_l3148_314890


namespace NUMINAMATH_CALUDE_dark_tiles_fraction_l3148_314832

/-- Represents a 4x4 block of tiles on the floor -/
structure Block where
  size : Nat
  dark_tiles : Nat

/-- Represents the entire tiled floor -/
structure Floor where
  block : Block

/-- The fraction of dark tiles in the floor -/
def dark_fraction (f : Floor) : Rat :=
  f.block.dark_tiles / (f.block.size * f.block.size)

theorem dark_tiles_fraction (f : Floor) 
  (h1 : f.block.size = 4)
  (h2 : f.block.dark_tiles = 12) : 
  dark_fraction f = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_dark_tiles_fraction_l3148_314832


namespace NUMINAMATH_CALUDE_adjacent_numbers_selection_l3148_314841

theorem adjacent_numbers_selection (n : ℕ) (k : ℕ) : 
  n = 49 → k = 6 → 
  (Nat.choose n k) - (Nat.choose (n - k + 1) k) = 
  (Nat.choose n k) - (Nat.choose 44 k) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_numbers_selection_l3148_314841


namespace NUMINAMATH_CALUDE_puzzle_solution_l3148_314837

theorem puzzle_solution (D E F : ℕ) 
  (h1 : D + E + F = 16)
  (h2 : F + D + 1 = 16)
  (h3 : E - 1 = D)
  (h4 : D ≠ E ∧ D ≠ F ∧ E ≠ F)
  (h5 : D < 10 ∧ E < 10 ∧ F < 10) : E = 1 := by
  sorry

#check puzzle_solution

end NUMINAMATH_CALUDE_puzzle_solution_l3148_314837


namespace NUMINAMATH_CALUDE_escalator_time_l3148_314804

theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) :
  escalator_speed = 8 →
  person_speed = 2 →
  escalator_length = 160 →
  escalator_length / (escalator_speed + person_speed) = 16 := by
  sorry

end NUMINAMATH_CALUDE_escalator_time_l3148_314804


namespace NUMINAMATH_CALUDE_age_difference_l3148_314828

theorem age_difference (a b c : ℕ) : 
  b = 10 →
  b = 2 * c →
  a + b + c = 27 →
  a = b + 2 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l3148_314828


namespace NUMINAMATH_CALUDE_equal_share_money_l3148_314800

theorem equal_share_money (emani_money : ℕ) (difference : ℕ) : 
  emani_money = 150 →
  difference = 30 →
  (emani_money + (emani_money - difference)) / 2 = 135 := by
  sorry

end NUMINAMATH_CALUDE_equal_share_money_l3148_314800


namespace NUMINAMATH_CALUDE_divisibility_by_five_l3148_314844

theorem divisibility_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : 5 ∣ a ∨ 5 ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l3148_314844


namespace NUMINAMATH_CALUDE_value_of_x_l3148_314815

theorem value_of_x (x y z : ℝ) : x = 3 * y ∧ y = z / 3 ∧ z = 90 → x = 90 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l3148_314815


namespace NUMINAMATH_CALUDE_real_part_of_complex_expression_l3148_314807

theorem real_part_of_complex_expression : Complex.re (1 + 2 / (Complex.I + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_expression_l3148_314807


namespace NUMINAMATH_CALUDE_f_sum_theorem_l3148_314806

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

theorem f_sum_theorem : f π + (deriv f) (π / 2) = -3 / π := by
  sorry

end NUMINAMATH_CALUDE_f_sum_theorem_l3148_314806


namespace NUMINAMATH_CALUDE_video_game_lives_l3148_314865

/-- The total number of lives for a group of friends in a video game -/
def totalLives (numFriends : ℕ) (livesPerFriend : ℕ) : ℕ :=
  numFriends * livesPerFriend

/-- Theorem: Given 15 friends, each with 25 lives, the total number of lives is 375 -/
theorem video_game_lives : totalLives 15 25 = 375 := by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l3148_314865


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3148_314801

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 2*m*x + 16 = (x - a)^2) → (m = 4 ∨ m = -4) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3148_314801


namespace NUMINAMATH_CALUDE_unique_solution_exists_l3148_314878

theorem unique_solution_exists : 
  ∃! (x y z : ℝ), x + y = 3 ∧ x * y - z^3 = 0 ∧ x = 1.5 ∧ y = 1.5 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l3148_314878


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l3148_314889

-- Define the arithmetic-geometric sequence and its properties
def arithmetic_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

-- Define S_n as the sum of the first n terms of a_n
def S (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S a n + a (n + 1)

-- Define T_n as the sum of the first n terms of S_n
def T (S : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => T S n + S (n + 1)

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_ag : arithmetic_geometric_sequence a)
  (h_S3 : S a 3 = 7)
  (h_S6 : S a 6 = 63) :
  (∀ n : ℕ, a n = 2^(n - 1)) ∧
  (∀ n : ℕ, S a n = 2^n - 1) ∧
  (∀ n : ℕ, T (S a) n = 2^(n + 1) - n - 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l3148_314889


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l3148_314876

theorem min_value_sum_fractions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) ∧
  (a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l3148_314876


namespace NUMINAMATH_CALUDE_students_in_lunchroom_l3148_314822

theorem students_in_lunchroom 
  (students_per_table : ℕ) 
  (number_of_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : number_of_tables = 34) : 
  students_per_table * number_of_tables = 204 := by
  sorry

end NUMINAMATH_CALUDE_students_in_lunchroom_l3148_314822


namespace NUMINAMATH_CALUDE_potato_cooking_time_l3148_314862

theorem potato_cooking_time (total_potatoes : ℕ) (cooked_potatoes : ℕ) (remaining_time : ℕ) :
  total_potatoes = 15 →
  cooked_potatoes = 8 →
  remaining_time = 63 →
  (remaining_time / (total_potatoes - cooked_potatoes) : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_potato_cooking_time_l3148_314862


namespace NUMINAMATH_CALUDE_sum_last_two_digits_modified_fibonacci_factorial_series_l3148_314858

def modifiedFibonacciFactorialSeries : List Nat := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 144]

def lastTwoDigits (n : Nat) : Nat :=
  n % 100

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_last_two_digits_modified_fibonacci_factorial_series :
  (modifiedFibonacciFactorialSeries.map (λ x => lastTwoDigits (factorial x))).sum % 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_last_two_digits_modified_fibonacci_factorial_series_l3148_314858


namespace NUMINAMATH_CALUDE_compare_values_l3148_314872

theorem compare_values : 0.5^(1/10) > 0.4^(1/10) ∧ 0.4^(1/10) > Real.log 0.1 / Real.log 4 := by
  sorry

end NUMINAMATH_CALUDE_compare_values_l3148_314872


namespace NUMINAMATH_CALUDE_iron_content_calculation_l3148_314897

theorem iron_content_calculation (initial_mass : ℝ) (impurities_mass : ℝ) 
  (impurities_iron_percent : ℝ) (iron_content_increase : ℝ) :
  initial_mass = 500 →
  impurities_mass = 200 →
  impurities_iron_percent = 12.5 →
  iron_content_increase = 20 →
  ∃ (remaining_iron : ℝ),
    remaining_iron = 187.5 ∧
    remaining_iron = 
      (initial_mass * ((impurities_mass * impurities_iron_percent / 100) / 
      (initial_mass - impurities_mass) + iron_content_increase / 100) / 100) * 
      (initial_mass - impurities_mass) -
      (impurities_mass * impurities_iron_percent / 100) := by
  sorry

end NUMINAMATH_CALUDE_iron_content_calculation_l3148_314897


namespace NUMINAMATH_CALUDE_zain_coin_count_l3148_314880

/-- Represents the count of each coin type --/
structure CoinCount where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ

/-- Calculates the total number of coins --/
def totalCoins (coins : CoinCount) : ℕ :=
  coins.quarters + coins.dimes + coins.nickels

/-- Zain's coin count given Emerie's coin count --/
def zainCoins (emerieCoins : CoinCount) : CoinCount :=
  { quarters := emerieCoins.quarters + 10
  , dimes := emerieCoins.dimes + 10
  , nickels := emerieCoins.nickels + 10 }

theorem zain_coin_count (emerieCoins : CoinCount) 
  (h1 : emerieCoins.quarters = 6)
  (h2 : emerieCoins.dimes = 7)
  (h3 : emerieCoins.nickels = 5) : 
  totalCoins (zainCoins emerieCoins) = 48 := by
  sorry

end NUMINAMATH_CALUDE_zain_coin_count_l3148_314880


namespace NUMINAMATH_CALUDE_range_of_m_when_a_is_one_range_of_a_for_sufficient_condition_l3148_314879

-- Define propositions p and q
def p (m a : ℝ) : Prop := m^2 - 7*a*m + 12*a^2 < 0 ∧ a > 0

def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2/(m-1) + y^2/(6-m) = 1 ∧ 1 < m ∧ m < 6

-- Theorem for part 1
theorem range_of_m_when_a_is_one :
  ∀ m : ℝ, (p m 1 ∧ q m) → (3 < m ∧ m < 7/2) :=
sorry

-- Theorem for part 2
theorem range_of_a_for_sufficient_condition :
  (∀ m a : ℝ, ¬(q m) → ¬(p m a)) ∧ (∃ m a : ℝ, ¬(p m a) ∧ q m) →
  (∀ a : ℝ, 1/3 ≤ a ∧ a ≤ 7/8) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_when_a_is_one_range_of_a_for_sufficient_condition_l3148_314879


namespace NUMINAMATH_CALUDE_ramon_twice_loui_in_twenty_years_loui_age_is_23_l3148_314835

/-- Ramon's current age -/
def ramon_current_age : ℕ := 26

/-- Loui's current age -/
def loui_current_age : ℕ := 23

/-- In twenty years, Ramon will be twice as old as Loui today -/
theorem ramon_twice_loui_in_twenty_years :
  ramon_current_age + 20 = 2 * loui_current_age := by sorry

theorem loui_age_is_23 : loui_current_age = 23 := by sorry

end NUMINAMATH_CALUDE_ramon_twice_loui_in_twenty_years_loui_age_is_23_l3148_314835


namespace NUMINAMATH_CALUDE_starting_lineup_selection_l3148_314886

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The total number of players in the team -/
def total_players : ℕ := 16

/-- The number of quadruplets -/
def num_quadruplets : ℕ := 4

/-- The number of starters to be selected -/
def num_starters : ℕ := 7

/-- The number of quadruplets that must be in the starting lineup -/
def quadruplets_in_lineup : ℕ := 3

/-- The number of ways to select the starting lineup -/
def num_ways : ℕ := 
  binomial num_quadruplets quadruplets_in_lineup * 
  binomial (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup)

theorem starting_lineup_selection :
  num_ways = 1980 := by sorry

end NUMINAMATH_CALUDE_starting_lineup_selection_l3148_314886


namespace NUMINAMATH_CALUDE_log_lt_x_div_one_minus_x_l3148_314827

theorem log_lt_x_div_one_minus_x (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  Real.log (1 + x) < x / (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_log_lt_x_div_one_minus_x_l3148_314827


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3148_314881

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3148_314881


namespace NUMINAMATH_CALUDE_investment_division_l3148_314842

/-- 
Given a total amount of 3200 divided into two parts, where one part is invested at 3% 
and the other at 5%, and the total annual interest is 144, prove that the amount 
of the first part (invested at 3%) is 800.
-/
theorem investment_division (x : ℝ) : 
  x ≥ 0 ∧ 
  3200 - x ≥ 0 ∧ 
  0.03 * x + 0.05 * (3200 - x) = 144 → 
  x = 800 := by
  sorry

#check investment_division

end NUMINAMATH_CALUDE_investment_division_l3148_314842


namespace NUMINAMATH_CALUDE_min_max_sum_l3148_314883

theorem min_max_sum (a b c d e : ℕ+) (h : a + b + c + d + e = 2020) :
  (max (a + b) (max (a + d) (max (b + e) (c + d)))) ≥ 1011 :=
sorry

end NUMINAMATH_CALUDE_min_max_sum_l3148_314883


namespace NUMINAMATH_CALUDE_adam_tickets_left_l3148_314836

/-- The number of tickets Adam had left after riding the ferris wheel -/
def tickets_left (initial_tickets : ℕ) (ticket_cost : ℕ) (spent_on_ride : ℕ) : ℕ :=
  initial_tickets - (spent_on_ride / ticket_cost)

/-- Theorem stating that Adam had 4 tickets left after riding the ferris wheel -/
theorem adam_tickets_left :
  tickets_left 13 9 81 = 4 := by
  sorry

#eval tickets_left 13 9 81

end NUMINAMATH_CALUDE_adam_tickets_left_l3148_314836


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l3148_314861

theorem fraction_sum_difference (p q r s : ℚ) 
  (h1 : p / q = 4 / 5) 
  (h2 : r / s = 3 / 7) : 
  (18 / 7) + ((2 * q - p) / (2 * q + p)) - ((3 * s + r) / (3 * s - r)) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l3148_314861


namespace NUMINAMATH_CALUDE_logical_equivalence_l3148_314826

theorem logical_equivalence (P Q R S : Prop) :
  ((P ∨ ¬R) → (¬Q ∧ S)) ↔ ((Q ∨ ¬S) → (¬P ∧ R)) :=
by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l3148_314826


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l3148_314870

-- Define a function f with the given property
def f : ℝ → ℝ := sorry

-- State the condition that f(x) = f(4-x) for all x
axiom f_symmetry (x : ℝ) : f x = f (4 - x)

-- Define what it means for a line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Theorem statement
theorem axis_of_symmetry :
  is_axis_of_symmetry 2 :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l3148_314870


namespace NUMINAMATH_CALUDE_sophie_wallet_problem_l3148_314882

theorem sophie_wallet_problem :
  ∃ (x y z : ℕ), 
    x + y + z = 60 ∧
    x + 2*y + 5*z = 175 ∧
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_sophie_wallet_problem_l3148_314882


namespace NUMINAMATH_CALUDE_other_communities_count_l3148_314830

theorem other_communities_count (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ)
  (h_total : total = 650)
  (h_muslim : muslim_percent = 44/100)
  (h_hindu : hindu_percent = 28/100)
  (h_sikh : sikh_percent = 10/100) :
  ⌊(1 - (muslim_percent + hindu_percent + sikh_percent)) * total⌋ = 117 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l3148_314830


namespace NUMINAMATH_CALUDE_minimum_value_problem_l3148_314874

theorem minimum_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 6) : 
  (9 / x) + (25 / y) + (49 / z) ≥ 37.5 ∧ 
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' + y' + z' = 6 ∧ 
    (9 / x') + (25 / y') + (49 / z') = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_problem_l3148_314874


namespace NUMINAMATH_CALUDE_auditorium_seating_l3148_314887

/-- The number of ways to seat people in an auditorium with the given conditions -/
def seatingArrangements (totalPeople : ℕ) (rowSeats : ℕ) : ℕ :=
  Nat.choose totalPeople rowSeats * 2^(totalPeople - 2)

/-- Theorem stating the number of seating arrangements for the given problem -/
theorem auditorium_seating :
  seatingArrangements 100 50 = Nat.choose 100 50 * 2^98 := by
  sorry

end NUMINAMATH_CALUDE_auditorium_seating_l3148_314887


namespace NUMINAMATH_CALUDE_lcm_problem_l3148_314820

theorem lcm_problem (a b c : ℕ+) (h1 : a = 24) (h2 : b = 42) 
  (h3 : Nat.lcm a (Nat.lcm b c) = 504) : c = 3 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3148_314820


namespace NUMINAMATH_CALUDE_spanish_only_count_l3148_314821

/-- Represents the number of students in different language classes -/
structure LanguageClasses where
  total : ℕ
  french : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of students taking Spanish only -/
def spanishOnly (lc : LanguageClasses) : ℕ :=
  lc.total - lc.french - lc.neither + lc.both

/-- Theorem stating the number of students taking Spanish only -/
theorem spanish_only_count (lc : LanguageClasses) 
  (h1 : lc.total = 28)
  (h2 : lc.french = 5)
  (h3 : lc.both = 4)
  (h4 : lc.neither = 13) :
  spanishOnly lc = 10 := by
  sorry

#check spanish_only_count

end NUMINAMATH_CALUDE_spanish_only_count_l3148_314821


namespace NUMINAMATH_CALUDE_paper_length_proof_l3148_314860

/-- Given a rectangular sheet of paper with specific dimensions and margins,
    prove that the length of the sheet is 10 inches. -/
theorem paper_length_proof (paper_width : Real) (margin : Real) (picture_area : Real) :
  paper_width = 8.5 →
  margin = 1.5 →
  picture_area = 38.5 →
  ∃ (paper_length : Real),
    paper_length = 10 ∧
    picture_area = (paper_length - 2 * margin) * (paper_width - 2 * margin) :=
by sorry

end NUMINAMATH_CALUDE_paper_length_proof_l3148_314860


namespace NUMINAMATH_CALUDE_circle_circumference_from_chord_l3148_314851

/-- Given a circular path with 8 evenly spaced trees, where the direct distance
    between two trees separated by 3 intervals is 100 feet, the total
    circumference of the circle is 175 feet. -/
theorem circle_circumference_from_chord (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  let interval := d / 4
  let circumference := interval * 7
  circumference = 175 := by
sorry

end NUMINAMATH_CALUDE_circle_circumference_from_chord_l3148_314851


namespace NUMINAMATH_CALUDE_number_of_men_l3148_314894

theorem number_of_men (M W B : ℕ) (Ww Wb : ℚ) : 
  M * 6 = W * Ww ∧ 
  W * Ww = 7 * B * Wb ∧ 
  M * 6 + W * Ww + B * Wb = 90 →
  M = 5 := by
sorry

end NUMINAMATH_CALUDE_number_of_men_l3148_314894


namespace NUMINAMATH_CALUDE_positive_square_harmonic_properties_l3148_314803

/-- Definition of a positive square harmonic function -/
def PositiveSquareHarmonic (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧
  (f 1 = 1) ∧
  (∀ x₁ x₂, x₁ + x₂ ∈ Set.Icc 0 1 → f x₁ + f x₂ ≤ f (x₁ + x₂))

theorem positive_square_harmonic_properties :
  ∀ f : ℝ → ℝ, PositiveSquareHarmonic f →
    (∀ x ∈ Set.Icc 0 1, f x = x^2) ∧
    (f 0 = 0) ∧
    (∀ x ∈ Set.Icc 0 1, f x ≤ 2*x) :=
by sorry

end NUMINAMATH_CALUDE_positive_square_harmonic_properties_l3148_314803


namespace NUMINAMATH_CALUDE_worksheets_to_memorize_l3148_314859

/-- Calculate the number of worksheets that can be memorized given study conditions --/
theorem worksheets_to_memorize (
  chapters : ℕ)
  (hours_per_chapter : ℝ)
  (hours_per_worksheet : ℝ)
  (max_hours_per_day : ℝ)
  (break_duration : ℝ)
  (breaks_per_day : ℕ)
  (snack_breaks : ℕ)
  (snack_break_duration : ℝ)
  (lunch_duration : ℝ)
  (study_days : ℕ)
  (h1 : chapters = 2)
  (h2 : hours_per_chapter = 3)
  (h3 : hours_per_worksheet = 1.5)
  (h4 : max_hours_per_day = 4)
  (h5 : break_duration = 1/6)  -- 10 minutes in hours
  (h6 : breaks_per_day = 4)
  (h7 : snack_breaks = 3)
  (h8 : snack_break_duration = 1/6)  -- 10 minutes in hours
  (h9 : lunch_duration = 0.5)  -- 30 minutes in hours
  (h10 : study_days = 4) :
  ⌊(study_days * (max_hours_per_day - (breaks_per_day * break_duration + snack_breaks * snack_break_duration + lunch_duration)) - chapters * hours_per_chapter) / hours_per_worksheet⌋ = 2 :=
by sorry

end NUMINAMATH_CALUDE_worksheets_to_memorize_l3148_314859


namespace NUMINAMATH_CALUDE_min_value_expression_l3148_314875

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  (1 / (2 * a)) + (1 / (2 * b)) + (8 / (a + b)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3148_314875


namespace NUMINAMATH_CALUDE_carrie_work_duration_l3148_314843

def hourly_wage : ℝ := 8
def weekly_hours : ℝ := 35
def bike_cost : ℝ := 400
def money_left : ℝ := 720

theorem carrie_work_duration :
  (money_left + bike_cost) / (hourly_wage * weekly_hours) = 4 := by
  sorry

end NUMINAMATH_CALUDE_carrie_work_duration_l3148_314843


namespace NUMINAMATH_CALUDE_picture_distribution_l3148_314840

theorem picture_distribution (total : ℕ) (main_album : ℕ) (other_albums : ℕ) 
  (h1 : total = 33) 
  (h2 : main_album = 27) 
  (h3 : other_albums = 3) :
  (total - main_album) / other_albums = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_picture_distribution_l3148_314840


namespace NUMINAMATH_CALUDE_min_value_n_over_m_l3148_314848

theorem min_value_n_over_m (m n : ℝ) :
  (∀ x : ℝ, Real.exp x - m * x + n - 1 ≥ 0) →
  (∃ k : ℝ, k = n / m ∧ k ≥ 0 ∧ ∀ j : ℝ, (∀ x : ℝ, Real.exp x - m * x + j * m - 1 ≥ 0) → j ≥ k) :=
by sorry

end NUMINAMATH_CALUDE_min_value_n_over_m_l3148_314848


namespace NUMINAMATH_CALUDE_lindas_cookies_l3148_314823

theorem lindas_cookies (classmates : Nat) (cookies_per_student : Nat) 
  (cookies_per_batch : Nat) (oatmeal_batches : Nat) (additional_batches : Nat) :
  classmates = 24 →
  cookies_per_student = 10 →
  cookies_per_batch = 48 →
  oatmeal_batches = 1 →
  additional_batches = 2 →
  ∃ (chocolate_chip_batches : Nat),
    chocolate_chip_batches * cookies_per_batch + 
    oatmeal_batches * cookies_per_batch + 
    additional_batches * cookies_per_batch = 
    classmates * cookies_per_student ∧
    chocolate_chip_batches = 2 :=
by sorry

end NUMINAMATH_CALUDE_lindas_cookies_l3148_314823


namespace NUMINAMATH_CALUDE_jennifer_book_fraction_l3148_314864

theorem jennifer_book_fraction (total : ℚ) (sandwich_fraction : ℚ) (museum_fraction : ℚ) (leftover : ℚ) :
  total = 90 →
  sandwich_fraction = 1 / 5 →
  museum_fraction = 1 / 6 →
  leftover = 12 →
  let spent := total - leftover
  let sandwich_cost := total * sandwich_fraction
  let museum_cost := total * museum_fraction
  let book_cost := spent - sandwich_cost - museum_cost
  book_cost / total = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_jennifer_book_fraction_l3148_314864


namespace NUMINAMATH_CALUDE_slope_range_theorem_l3148_314877

-- Define a line by its slope and a point it passes through
def Line (k : ℝ) (x₀ y₀ : ℝ) :=
  {(x, y) : ℝ × ℝ | y - y₀ = k * (x - x₀)}

-- Define the translation of a line
def translate (L : Set (ℝ × ℝ)) (dx dy : ℝ) :=
  {(x, y) : ℝ × ℝ | (x - dx, y - dy) ∈ L}

-- Define the fourth quadrant
def fourthQuadrant := {(x, y) : ℝ × ℝ | x > 0 ∧ y < 0}

theorem slope_range_theorem (k : ℝ) :
  let l := Line k 1 (-1)
  let m := translate l 3 (-2)
  (∀ p ∈ m, p ∉ fourthQuadrant) → 0 ≤ k ∧ k ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_slope_range_theorem_l3148_314877


namespace NUMINAMATH_CALUDE_correct_savings_amount_l3148_314849

/-- Represents a bank with its interest calculation method -/
structure Bank where
  name : String
  calculateInterest : (principal : ℝ) → ℝ

/-- Calculates the amount needed to save given a bank's interest calculation -/
def amountToSave (initialFunds : ℝ) (totalExpenses : ℝ) (bank : Bank) : ℝ :=
  totalExpenses - initialFunds - bank.calculateInterest initialFunds

/-- Theorem stating the correct amount to save for each bank -/
theorem correct_savings_amount 
  (initialFunds : ℝ) 
  (totalExpenses : ℝ) 
  (bettaBank gammaBank omegaBank epsilonBank : Bank) 
  (h1 : initialFunds = 150000)
  (h2 : totalExpenses = 182200)
  (h3 : bettaBank.calculateInterest initialFunds = 2720.33)
  (h4 : gammaBank.calculateInterest initialFunds = 3375)
  (h5 : omegaBank.calculateInterest initialFunds = 2349.13)
  (h6 : epsilonBank.calculateInterest initialFunds = 2264.11) :
  (amountToSave initialFunds totalExpenses bettaBank = 29479.67) ∧
  (amountToSave initialFunds totalExpenses gammaBank = 28825) ∧
  (amountToSave initialFunds totalExpenses omegaBank = 29850.87) ∧
  (amountToSave initialFunds totalExpenses epsilonBank = 29935.89) :=
by sorry


end NUMINAMATH_CALUDE_correct_savings_amount_l3148_314849


namespace NUMINAMATH_CALUDE_sum_of_three_squares_mod_8_l3148_314825

theorem sum_of_three_squares_mod_8 (a b c : ℤ) : (a^2 + b^2 + c^2) % 8 ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_mod_8_l3148_314825


namespace NUMINAMATH_CALUDE_total_amount_l3148_314847

/-- Represents the division of money among three people -/
structure MoneyDivision where
  x : ℝ  -- X's share
  y : ℝ  -- Y's share
  z : ℝ  -- Z's share

/-- The conditions of the money division problem -/
def problem_conditions (d : MoneyDivision) : Prop :=
  d.y = 0.75 * d.x ∧ 
  d.z = (2/3) * d.x ∧ 
  d.y = 48

/-- The theorem stating the total amount -/
theorem total_amount (d : MoneyDivision) 
  (h : problem_conditions d) : d.x + d.y + d.z = 154.67 := by
  sorry

#check total_amount

end NUMINAMATH_CALUDE_total_amount_l3148_314847


namespace NUMINAMATH_CALUDE_smallest_among_given_numbers_l3148_314873

theorem smallest_among_given_numbers :
  let numbers : List ℚ := [-6/7, 2, 0, -1]
  ∀ x ∈ numbers, -1 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_among_given_numbers_l3148_314873


namespace NUMINAMATH_CALUDE_abc_inequality_l3148_314812

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 ∧
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3148_314812


namespace NUMINAMATH_CALUDE_einstein_born_on_friday_l3148_314809

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Checks if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  year % 400 == 0 || (year % 4 == 0 && year % 100 ≠ 0)

/-- Einstein's birth year -/
def einsteinBirthYear : Nat := 1865

/-- Einstein's 160th anniversary year -/
def anniversaryYear : Nat := 2025

/-- Day of the week of Einstein's 160th anniversary -/
def anniversaryDayOfWeek : DayOfWeek := DayOfWeek.Friday

/-- Calculates the day of the week Einstein was born -/
def einsteinBirthDayOfWeek : DayOfWeek := sorry

theorem einstein_born_on_friday :
  einsteinBirthDayOfWeek = DayOfWeek.Friday := by sorry

end NUMINAMATH_CALUDE_einstein_born_on_friday_l3148_314809


namespace NUMINAMATH_CALUDE_identity_proof_l3148_314824

theorem identity_proof (a b : ℝ) : a^4 + b^4 + (a+b)^4 = 2*(a^2 + a*b + b^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l3148_314824


namespace NUMINAMATH_CALUDE_x_plus_y_fifth_power_l3148_314885

theorem x_plus_y_fifth_power (x y : ℝ) 
  (sum_eq : x + y = 3)
  (frac_eq : 1 / (x + y^2) + 1 / (x^2 + y) = 1 / 2) :
  x^5 + y^5 = 243 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_fifth_power_l3148_314885


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3148_314868

/-- The line y = kx + 1 and the parabola y^2 = 4x have only one common point -/
def has_one_common_point (k : ℝ) : Prop :=
  ∃! x y, y = k * x + 1 ∧ y^2 = 4 * x

theorem sufficient_not_necessary :
  (∀ k, k = 0 → has_one_common_point k) ∧
  (∃ k, k ≠ 0 ∧ has_one_common_point k) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3148_314868


namespace NUMINAMATH_CALUDE_sally_cut_six_orchids_l3148_314819

/-- The number of red orchids Sally cut from her garden -/
def orchids_cut (initial_red : ℕ) (final_red : ℕ) : ℕ :=
  final_red - initial_red

/-- Theorem stating that Sally cut 6 red orchids -/
theorem sally_cut_six_orchids (initial_red : ℕ) (final_red : ℕ) 
  (h1 : initial_red = 9)
  (h2 : final_red = 15) : 
  orchids_cut initial_red final_red = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_cut_six_orchids_l3148_314819


namespace NUMINAMATH_CALUDE_smallest_y_with_24_factors_l3148_314833

theorem smallest_y_with_24_factors (y : ℕ) 
  (h1 : (Nat.divisors y).card = 24)
  (h2 : 20 ∣ y)
  (h3 : 35 ∣ y) :
  y ≥ 1120 ∧ ∃ (z : ℕ), z ≥ 1120 ∧ (Nat.divisors z).card = 24 ∧ 20 ∣ z ∧ 35 ∣ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_with_24_factors_l3148_314833


namespace NUMINAMATH_CALUDE_basic_operation_time_scientific_notation_l3148_314802

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  norm : 1 ≤ coefficient ∧ coefficient < 10

/-- The time taken for one basic operation in seconds -/
def basicOperationTime : ℝ := 0.000000001

/-- The scientific notation representation of the basic operation time -/
def basicOperationTimeScientific : ScientificNotation :=
  { coefficient := 1
  , exponent := -9
  , norm := by sorry }

/-- Theorem stating that the basic operation time is correctly represented in scientific notation -/
theorem basic_operation_time_scientific_notation :
  basicOperationTime = basicOperationTimeScientific.coefficient * (10 : ℝ) ^ basicOperationTimeScientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_basic_operation_time_scientific_notation_l3148_314802


namespace NUMINAMATH_CALUDE_dave_tshirts_l3148_314845

def white_packs : ℕ := 3
def blue_packs : ℕ := 2
def red_packs : ℕ := 4
def green_packs : ℕ := 1

def white_per_pack : ℕ := 6
def blue_per_pack : ℕ := 4
def red_per_pack : ℕ := 5
def green_per_pack : ℕ := 3

def total_tshirts : ℕ := 
  white_packs * white_per_pack + 
  blue_packs * blue_per_pack + 
  red_packs * red_per_pack + 
  green_packs * green_per_pack

theorem dave_tshirts : total_tshirts = 49 := by
  sorry

end NUMINAMATH_CALUDE_dave_tshirts_l3148_314845


namespace NUMINAMATH_CALUDE_chef_apples_l3148_314869

theorem chef_apples (apples_left apples_used : ℕ) 
  (h1 : apples_left = 2) 
  (h2 : apples_used = 41) : 
  apples_left + apples_used = 43 := by
  sorry

end NUMINAMATH_CALUDE_chef_apples_l3148_314869


namespace NUMINAMATH_CALUDE_power_equality_solution_l3148_314831

theorem power_equality_solution : ∃ x : ℝ, x^5 = 5^10 ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_solution_l3148_314831


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3148_314814

def z₁ : ℂ := Complex.I
def z₂ : ℂ := 1 + Complex.I

theorem z_in_second_quadrant : 
  let z : ℂ := z₁ * z₂
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3148_314814


namespace NUMINAMATH_CALUDE_original_ratio_l3148_314896

theorem original_ratio (x y : ℝ) (h1 : y = 40) (h2 : (x + 10) / (y + 10) = 4/5) :
  x / y = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_original_ratio_l3148_314896


namespace NUMINAMATH_CALUDE_activity_ranking_l3148_314850

def fishing_popularity : ℚ := 13/36
def hiking_popularity : ℚ := 8/27
def painting_popularity : ℚ := 7/18

theorem activity_ranking :
  painting_popularity > fishing_popularity ∧
  fishing_popularity > hiking_popularity := by
  sorry

end NUMINAMATH_CALUDE_activity_ranking_l3148_314850


namespace NUMINAMATH_CALUDE_perfect_cube_base9_last_digit_l3148_314846

/-- Represents a number in base 9 of the form ab4c -/
structure Base9Number where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a ≠ 0
  h2 : c ≤ 8

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.b + 36 + n.c

/-- Predicate to check if a natural number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem perfect_cube_base9_last_digit 
  (n : Base9Number) 
  (h : isPerfectCube (toDecimal n)) : 
  n.c = 1 ∨ n.c = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_base9_last_digit_l3148_314846


namespace NUMINAMATH_CALUDE_after_school_program_enrollment_l3148_314829

theorem after_school_program_enrollment (drama_students music_students both_students : ℕ) 
  (h1 : drama_students = 41)
  (h2 : music_students = 28)
  (h3 : both_students = 15) :
  drama_students + music_students - both_students = 54 := by
sorry

end NUMINAMATH_CALUDE_after_school_program_enrollment_l3148_314829


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_two_times_sqrt_three_eq_sqrt_six_l3148_314892

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b := by
  sorry

theorem sqrt_two_times_sqrt_three_eq_sqrt_six : 
  Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_two_times_sqrt_three_eq_sqrt_six_l3148_314892


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l3148_314898

-- Define the line
def line (k x : ℝ) : ℝ := k * x - k + 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Theorem statement
theorem line_intersects_ellipse :
  ∀ k : ℝ, ∃ x y : ℝ, line k x = y ∧ ellipse x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l3148_314898


namespace NUMINAMATH_CALUDE_dinosaur_book_cost_l3148_314838

def dictionary_cost : ℕ := 5
def cookbook_cost : ℕ := 5
def saved_amount : ℕ := 19
def additional_needed : ℕ := 2

theorem dinosaur_book_cost :
  dictionary_cost + cookbook_cost + (saved_amount + additional_needed - (dictionary_cost + cookbook_cost)) = 11 := by
  sorry

end NUMINAMATH_CALUDE_dinosaur_book_cost_l3148_314838


namespace NUMINAMATH_CALUDE_negative_reciprocal_equality_l3148_314856

theorem negative_reciprocal_equality (a b : ℝ) : 
  (-1 / a = 8) → (-1 / (-b) = 8) → a = b := by sorry

end NUMINAMATH_CALUDE_negative_reciprocal_equality_l3148_314856


namespace NUMINAMATH_CALUDE_polygonal_number_theorem_l3148_314888

/-- The n-th k-sided polygonal number -/
def N (n k : ℕ) : ℚ :=
  (k - 2) / 2 * n^2 + (4 - k) / 2 * n

/-- Theorem stating the formula for the n-th k-sided polygonal number and the value of N(8,12) -/
theorem polygonal_number_theorem (n k : ℕ) (h1 : k ≥ 3) (h2 : n ≥ 1) : 
  N n k = (k - 2) / 2 * n^2 + (4 - k) / 2 * n ∧ N 8 12 = 288 := by
  sorry

end NUMINAMATH_CALUDE_polygonal_number_theorem_l3148_314888


namespace NUMINAMATH_CALUDE_sum_of_digits_of_9N_l3148_314863

/-- A function that checks if each digit of a natural number is strictly greater than the digit to its left -/
def is_strictly_increasing_digits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get i < (n.digits 10).get j

/-- A function that calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem: For any natural number N where each digit is strictly greater than the digit to its left,
    the sum of the digits of 9N is equal to 9 -/
theorem sum_of_digits_of_9N (N : ℕ) (h : is_strictly_increasing_digits N) :
  sum_of_digits (9 * N) = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_9N_l3148_314863


namespace NUMINAMATH_CALUDE_find_a_value_l3148_314867

theorem find_a_value (x y a : ℝ) 
  (h1 : x / (2 * y) = 3 / 2)
  (h2 : (a * x + 6 * y) / (x - 2 * y) = 27) :
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l3148_314867


namespace NUMINAMATH_CALUDE_birdseed_mix_problem_l3148_314816

/-- Represents the composition of a birdseed brand -/
structure BirdseedBrand where
  millet : Float
  sunflower : Float

/-- Represents a mix of two birdseed brands -/
structure BirdseedMix where
  brandA : BirdseedBrand
  brandB : BirdseedBrand
  proportionA : Float

theorem birdseed_mix_problem (mixA : BirdseedBrand) (mixB : BirdseedBrand) (mix : BirdseedMix) :
  mixA.millet = 0.4 →
  mixA.sunflower = 0.6 →
  mixB.millet = 0.65 →
  mix.brandA = mixA →
  mix.brandB = mixB →
  mix.proportionA = 0.6 →
  (mix.proportionA * mixA.sunflower + (1 - mix.proportionA) * mixB.sunflower = 0.5) →
  mixB.sunflower = 0.35 := by
  sorry

#check birdseed_mix_problem

end NUMINAMATH_CALUDE_birdseed_mix_problem_l3148_314816


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3148_314891

theorem simplify_square_roots : 
  (Real.sqrt 507 / Real.sqrt 48) - (Real.sqrt 175 / Real.sqrt 112) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3148_314891


namespace NUMINAMATH_CALUDE_total_fruits_l3148_314813

def papaya_trees : ℕ := 2
def mango_trees : ℕ := 3
def papayas_per_tree : ℕ := 10
def mangos_per_tree : ℕ := 20

theorem total_fruits : 
  papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_l3148_314813


namespace NUMINAMATH_CALUDE_at_most_one_greater_than_one_l3148_314893

theorem at_most_one_greater_than_one (x y : ℝ) (h : x + y < 2) :
  ¬(x > 1 ∧ y > 1) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_greater_than_one_l3148_314893


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3148_314855

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  man : ℝ  -- Speed of the man in still water
  stream : ℝ  -- Speed of the stream

/-- Calculates the effective speed when swimming downstream. -/
def downstream_speed (s : SwimmerSpeed) : ℝ := s.man + s.stream

/-- Calculates the effective speed when swimming upstream. -/
def upstream_speed (s : SwimmerSpeed) : ℝ := s.man - s.stream

/-- Theorem stating that given the conditions of the problem, the man's speed in still water is 12 km/h. -/
theorem swimmer_speed_in_still_water :
  ∀ (s : SwimmerSpeed),
    54 = downstream_speed s * 3 →
    18 = upstream_speed s * 3 →
    s.man = 12 := by
  sorry


end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3148_314855


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3148_314805

theorem quadratic_equation_solution (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x - 2 = 0 ∧ (x + 1) / (x - 1) = 3) → 
  k = -1 ∧ ∃ y : ℝ, y ≠ 2 ∧ y^2 + k*y - 2 = 0 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3148_314805


namespace NUMINAMATH_CALUDE_ratio_exists_l3148_314810

theorem ratio_exists : ∃ (m n : ℤ), 
  m > 100 ∧ 
  n > 100 ∧ 
  m + n = 300 ∧ 
  3 * n = 2 * m := by
sorry

end NUMINAMATH_CALUDE_ratio_exists_l3148_314810


namespace NUMINAMATH_CALUDE_subset_intersection_count_l3148_314811

-- Define the set S with n elements
variable (n : ℕ)
variable (S : Finset (Fin n))

-- Define k subsets of S
variable (k : ℕ)
variable (A : Fin k → Finset (Fin n))

-- Conditions
variable (h1 : ∀ i, A i ⊆ S)
variable (h2 : ∀ i j, i ≠ j → (A i ∩ A j).Nonempty)
variable (h3 : ∀ X, X ⊆ S → (∀ i, (X ∩ A i).Nonempty) → ∃ i, X = A i)

-- Theorem statement
theorem subset_intersection_count : k = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_subset_intersection_count_l3148_314811


namespace NUMINAMATH_CALUDE_probability_all_white_or_all_black_l3148_314871

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def total_balls : ℕ := white_balls + black_balls
def drawn_balls : ℕ := 5

theorem probability_all_white_or_all_black :
  (Nat.choose white_balls drawn_balls + Nat.choose black_balls drawn_balls) / Nat.choose total_balls drawn_balls = 77 / 3003 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_white_or_all_black_l3148_314871


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l3148_314854

theorem x_fourth_plus_inverse_x_fourth (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l3148_314854


namespace NUMINAMATH_CALUDE_inverse_function_point_l3148_314818

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + Real.log x / Real.log a

def has_inverse_point (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  ∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ Function.RightInverse g f ∧ g x = y

theorem inverse_function_point (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  has_inverse_point (f a) 2 4 →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_inverse_function_point_l3148_314818


namespace NUMINAMATH_CALUDE_yearly_savings_ratio_l3148_314808

-- Define the fraction of salary spent each month
def fraction_spent : ℚ := 0.6666666666666667

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Theorem statement
theorem yearly_savings_ratio :
  (1 - fraction_spent) * months_in_year = 4 := by
  sorry

end NUMINAMATH_CALUDE_yearly_savings_ratio_l3148_314808


namespace NUMINAMATH_CALUDE_min_value_of_squares_l3148_314895

theorem min_value_of_squares (t : ℝ) :
  ∃ (a b : ℝ), 2 * a + 3 * b = t ∧
  ∀ (x y : ℝ), 2 * x + 3 * y = t → a^2 + b^2 ≤ x^2 + y^2 ∧
  a^2 + b^2 = (13 * t^2) / 169 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_squares_l3148_314895


namespace NUMINAMATH_CALUDE_max_popsicles_for_zoe_l3148_314857

/-- Represents the pricing options for popsicles -/
structure PopsicleOptions where
  single_price : ℕ
  four_pack_price : ℕ
  seven_pack_price : ℕ

/-- Calculates the maximum number of popsicles that can be bought with a given budget -/
def max_popsicles (options : PopsicleOptions) (budget : ℕ) : ℕ :=
  sorry

/-- The store's pricing options -/
def store_options : PopsicleOptions :=
  { single_price := 2
  , four_pack_price := 3
  , seven_pack_price := 5 }

/-- Zoe's budget -/
def zoe_budget : ℕ := 11

/-- Theorem: The maximum number of popsicles Zoe can buy with $11 is 14 -/
theorem max_popsicles_for_zoe :
  max_popsicles store_options zoe_budget = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_popsicles_for_zoe_l3148_314857


namespace NUMINAMATH_CALUDE_find_x_l3148_314817

theorem find_x (a b : ℝ) (x y r : ℝ) (h1 : b ≠ 0) (h2 : r = (3*a)^(3*b)) (h3 : r = a^b * (x + y)^b) (h4 : y = 3*a) :
  x = 27*a^2 - 3*a := by
sorry

end NUMINAMATH_CALUDE_find_x_l3148_314817


namespace NUMINAMATH_CALUDE_decagon_diagonals_l3148_314853

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- Formula for calculating the number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem stating that the number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l3148_314853


namespace NUMINAMATH_CALUDE_cube_root_seven_to_sixth_l3148_314899

theorem cube_root_seven_to_sixth (x : ℝ) (h : x = 7^(1/3)) : x^6 = 49 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_seven_to_sixth_l3148_314899


namespace NUMINAMATH_CALUDE_delegates_without_badges_l3148_314834

theorem delegates_without_badges (total : Nat) (preprinted : Nat) : 
  total = 36 → preprinted = 16 → (total - preprinted - (total - preprinted) / 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_delegates_without_badges_l3148_314834


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3148_314839

theorem complex_magnitude_problem (x y : ℝ) (h : (5 : ℂ) - x * I = y + 1 - 3 * I) : 
  Complex.abs (x - y * I) = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3148_314839


namespace NUMINAMATH_CALUDE_investment_total_l3148_314884

/-- Represents the investment scenario with two parts at different interest rates -/
structure Investment where
  total : ℝ
  part1 : ℝ
  part2 : ℝ
  rate1 : ℝ
  rate2 : ℝ
  total_interest : ℝ

/-- The investment satisfies the given conditions -/
def valid_investment (i : Investment) : Prop :=
  i.total = i.part1 + i.part2 ∧
  i.part1 = 2800 ∧
  i.rate1 = 0.03 ∧
  i.rate2 = 0.05 ∧
  i.total_interest = 144 ∧
  i.part1 * i.rate1 + i.part2 * i.rate2 = i.total_interest

/-- Theorem: Given the conditions, the total amount divided is 4000 -/
theorem investment_total (i : Investment) (h : valid_investment i) : i.total = 4000 := by
  sorry

end NUMINAMATH_CALUDE_investment_total_l3148_314884


namespace NUMINAMATH_CALUDE_x_equals_cos_alpha_l3148_314866

/-- Given two squares with side length 1/2 inclined at an angle 2α, 
    x is the length of the line segment connecting the midpoints of 
    the non-intersecting sides of the squares -/
def x (α : Real) : Real :=
  sorry

theorem x_equals_cos_alpha (α : Real) : x α = Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_x_equals_cos_alpha_l3148_314866


namespace NUMINAMATH_CALUDE_remainder_after_adding_2040_l3148_314852

theorem remainder_after_adding_2040 (n : ℤ) (h : n % 8 = 3) : (n + 2040) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2040_l3148_314852
