import Mathlib

namespace complex_expression_value_l2045_204544

theorem complex_expression_value :
  ((6^2 - 4^2) + 2)^3 / 2 = 5324 :=
by
  sorry

end complex_expression_value_l2045_204544


namespace part_I_part_II_l2045_204510

theorem part_I (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_adbc: a * d = b * c) (h_ineq1: a + d > b + c): |a - d| > |b - c| :=
sorry

theorem part_II (a b c d t: ℝ) 
(h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
(h_eq: t * (Real.sqrt (a^2 + b^2) * Real.sqrt (c^2 + d^2)) = Real.sqrt (a^4 + c^4) + Real.sqrt (b^4 + d^4)):
t >= Real.sqrt 2 :=
sorry

end part_I_part_II_l2045_204510


namespace train_length_l2045_204595

theorem train_length (L : ℝ) :
  (∀ t₁ t₂ : ℝ, t₁ = t₂ → L = t₁ / 2) →
  (∀ t : ℝ, t = (8 / 3600) * 36 → L * 2 = t) →
  44 - 36 = 8 →
  L = 40 :=
by
  sorry

end train_length_l2045_204595


namespace find_x_l2045_204572

theorem find_x {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0)
    (h1 : x + 1/y = 10) (h2 : y + 1/x = 5/12) : x = 4 ∨ x = 6 :=
by
  sorry

end find_x_l2045_204572


namespace subway_train_speed_l2045_204557

open Nat

-- Define the speed function
def speed (s : ℕ) : ℕ := s^2 + 2*s

-- Define the theorem to be proved
theorem subway_train_speed (t : ℕ) (ht : 0 ≤ t ∧ t ≤ 7) (h_speed : speed 7 - speed t = 28) : t = 5 :=
by
  sorry

end subway_train_speed_l2045_204557


namespace polynomial_root_sum_eq_48_l2045_204547

theorem polynomial_root_sum_eq_48 {r s t : ℕ} (h1 : r * s * t = 2310) 
  (h2 : r > 0) (h3 : s > 0) (h4 : t > 0) : r + s + t = 48 :=
sorry

end polynomial_root_sum_eq_48_l2045_204547


namespace bart_trees_needed_l2045_204562

-- Define the constants and conditions given
def firewood_per_tree : Nat := 75
def logs_burned_per_day : Nat := 5
def days_in_november : Nat := 30
def days_in_december : Nat := 31
def days_in_january : Nat := 31
def days_in_february : Nat := 28

-- Calculate the total number of days from November 1 through February 28
def total_days : Nat := days_in_november + days_in_december + days_in_january + days_in_february

-- Calculate the total number of pieces of firewood needed
def total_firewood_needed : Nat := total_days * logs_burned_per_day

-- Calculate the number of trees needed
def trees_needed : Nat := total_firewood_needed / firewood_per_tree

-- The proof statement
theorem bart_trees_needed : trees_needed = 8 := 
by
  -- Placeholder for the proof
  sorry

end bart_trees_needed_l2045_204562


namespace proof_problem_l2045_204522

variables {m n : ℝ}

theorem proof_problem (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 2 * m * n) :
  (mn : ℝ) ≥ 1 ∧ (m^2 + n^2 ≥ 2) :=
  sorry

end proof_problem_l2045_204522


namespace findYearsForTwiceAge_l2045_204589

def fatherSonAges : ℕ := 33

def fatherAge : ℕ := fatherSonAges + 35

def yearsForTwiceAge (x : ℕ) : Prop :=
  fatherAge + x = 2 * (fatherSonAges + x)

theorem findYearsForTwiceAge : ∃ x, yearsForTwiceAge x :=
  ⟨2, sorry⟩

end findYearsForTwiceAge_l2045_204589


namespace symmetric_line_equation_l2045_204525

-- Definitions of the given conditions.
def original_line_equation (x y : ℝ) : Prop := 2 * x + 3 * y + 6 = 0
def line_of_symmetry (x y : ℝ) : Prop := y = x

-- The theorem statement to prove:
theorem symmetric_line_equation (x y : ℝ) : original_line_equation y x ↔ (3 * x + 2 * y + 6 = 0) :=
sorry

end symmetric_line_equation_l2045_204525


namespace quadrilateral_area_l2045_204598

noncomputable def area_of_quadrilateral (a : ℝ) : ℝ :=
  let sqrt3 := Real.sqrt 3
  let num := a^2 * (9 - 5 * sqrt3)
  let denom := 12
  num / denom

theorem quadrilateral_area (a : ℝ) : area_of_quadrilateral a = (a^2 * (9 - 5 * Real.sqrt 3)) / 12 := by
  sorry

end quadrilateral_area_l2045_204598


namespace concert_people_count_l2045_204504

variable {W M : ℕ}

theorem concert_people_count (h1 : W * 2 = M) (h2 : (W - 12) * 3 = M - 29) : W + M = 21 := 
sorry

end concert_people_count_l2045_204504


namespace minnie_mounts_time_period_l2045_204538

theorem minnie_mounts_time_period (M D : ℕ) 
  (mickey_daily_mounts_eq : 2 * M - 6 = 14)
  (minnie_mounts_per_day_eq : M = D + 3) : 
  D = 7 := 
by
  sorry

end minnie_mounts_time_period_l2045_204538


namespace three_pairs_exist_l2045_204517

theorem three_pairs_exist :
  ∃! S P : ℕ, 5 * S + 7 * P = 90 :=
by
  sorry

end three_pairs_exist_l2045_204517


namespace intersection_eq_zero_l2045_204559

def M := { x : ℤ | abs (x - 3) < 4 }
def N := { x : ℤ | x^2 + x - 2 < 0 }

theorem intersection_eq_zero : M ∩ N = {0} := 
  by
    sorry

end intersection_eq_zero_l2045_204559


namespace multiply_seven_l2045_204573

variable (x : ℕ)

theorem multiply_seven (h : 8 * x = 64) : 7 * x = 56 := by
  sorry


end multiply_seven_l2045_204573


namespace min_value_l2045_204509

noncomputable def min_value_of_expression (a b: ℝ) :=
    a > 0 ∧ b > 0 ∧ a + b = 1 → (∃ (m : ℝ), (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (1 / x + 2 / y) ≥ m) ∧ m = 3 + 2 * Real.sqrt 2)

theorem min_value (a b: ℝ) (h₀: a > 0) (h₁: b > 0) (h₂: a + b = 1) :
    ∃ m, (∀ x y, x > 0 → y > 0 → x + y = 1 → (1 / x + 2 / y) ≥ m) ∧ m = 3 + 2 * Real.sqrt 2 := 
by
    sorry

end min_value_l2045_204509


namespace unique_triad_l2045_204580

theorem unique_triad (x y z : ℕ) 
  (h_distinct: x ≠ y ∧ y ≠ z ∧ z ≠ x) 
  (h_gcd: Nat.gcd (Nat.gcd x y) z = 1)
  (h_div_properties: (z ∣ x + y) ∧ (x ∣ y + z) ∧ (y ∣ z + x)) :
  (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
sorry

end unique_triad_l2045_204580


namespace part_a_part_b_l2045_204561
open Set

def fantastic (n : ℕ) : Prop :=
  ∃ a b : ℚ, a > 0 ∧ b > 0 ∧ n = a + 1 / a + b + 1 / b

theorem part_a : ∃ᶠ p in at_top, Prime p ∧ ∀ k, ¬ fantastic (k * p) := 
  sorry

theorem part_b : ∃ᶠ p in at_top, Prime p ∧ ∃ k, fantastic (k * p) :=
  sorry

end part_a_part_b_l2045_204561


namespace intersection_P_Q_l2045_204502

def P : Set ℝ := { x | x^2 - x = 0 }
def Q : Set ℝ := { x | x^2 + x = 0 }

theorem intersection_P_Q : (P ∩ Q) = {0} := 
by
  sorry

end intersection_P_Q_l2045_204502


namespace factorization_of_polynomial_l2045_204512

theorem factorization_of_polynomial :
  (x : ℤ) → x^10 + x^5 + 1 = (x^2 + x + 1) * (x^8 - x^7 + x^5 - x^4 + x^3 - x + 1) :=
by
  sorry

end factorization_of_polynomial_l2045_204512


namespace sqrt_31_minus_2_in_range_l2045_204501

-- Defining the conditions based on the problem statements
def five_squared : ℤ := 5 * 5
def six_squared : ℤ := 6 * 6
def thirty_one : ℤ := 31

theorem sqrt_31_minus_2_in_range : 
  (5 * 5 < thirty_one) ∧ (thirty_one < 6 * 6) →
  3 < (Real.sqrt thirty_one) - 2 ∧ (Real.sqrt thirty_one) - 2 < 4 :=
by
  sorry

end sqrt_31_minus_2_in_range_l2045_204501


namespace max_PA_PB_l2045_204592

noncomputable def max_distance (PA PB : ℝ) : ℝ :=
  PA + PB

theorem max_PA_PB {A B : ℝ × ℝ} (m : ℝ) :
  A = (0, 0) ∧
  B = (1, 3) ∧
  dist A B = 10 →
  max_distance (dist A B) (dist (1, 3) B) = 2 * Real.sqrt 5 :=
by
  sorry

end max_PA_PB_l2045_204592


namespace trigonometric_identity_proof_l2045_204581

open Real

theorem trigonometric_identity_proof (x y : ℝ) (hx : sin x / sin y = 4) (hy : cos x / cos y = 1 / 3) :
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = 169 / 381 :=
by
  sorry

end trigonometric_identity_proof_l2045_204581


namespace smallest_prime_divisor_of_sum_first_100_is_5_l2045_204529

-- Conditions: The sum of the first 100 natural numbers
def sum_first_n_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Prime checking function to identify the smallest prime divisor
def smallest_prime_divisor (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 else
  if n % 3 = 0 then 3 else
  if n % 5 = 0 then 5 else
  n -- Such a simplification works because we know the answer must be within the first few primes.

-- Proof statement
theorem smallest_prime_divisor_of_sum_first_100_is_5 : smallest_prime_divisor (sum_first_n_numbers 100) = 5 :=
by
  -- Proof steps would follow here.
  sorry

end smallest_prime_divisor_of_sum_first_100_is_5_l2045_204529


namespace total_swordfish_catch_l2045_204541

-- Definitions
def S_c : ℝ := 5 - 2
def S_m : ℝ := S_c - 1
def S_a : ℝ := 2 * S_m

def W_s : ℕ := 3  -- Number of sunny days
def W_r : ℕ := 2  -- Number of rainy days

-- Sunny and rainy day adjustments
def Shelly_sunny_catch : ℝ := S_c + 0.20 * S_c
def Sam_sunny_catch : ℝ := S_m + 0.20 * S_m
def Sara_sunny_catch : ℝ := S_a + 0.20 * S_a

def Shelly_rainy_catch : ℝ := S_c - 0.10 * S_c
def Sam_rainy_catch : ℝ := S_m - 0.10 * S_m
def Sara_rainy_catch : ℝ := S_a - 0.10 * S_a

-- Total catch calculations
def Shelly_total_catch : ℝ := W_s * Shelly_sunny_catch + W_r * Shelly_rainy_catch
def Sam_total_catch : ℝ := W_s * Sam_sunny_catch + W_r * Sam_rainy_catch
def Sara_total_catch : ℝ := W_s * Sara_sunny_catch + W_r * Sara_rainy_catch

def Total_catch : ℝ := Shelly_total_catch + Sam_total_catch + Sara_total_catch

-- Proof statement
theorem total_swordfish_catch : ⌊Total_catch⌋ = 48 := 
  by sorry

end total_swordfish_catch_l2045_204541


namespace location_determined_l2045_204540

def determine_location(p : String) : Prop :=
  p = "Longitude 118°E, Latitude 40°N"

axiom row_2_in_cinema : ¬determine_location "Row 2 in a cinema"
axiom daqiao_south_road_nanjing : ¬determine_location "Daqiao South Road in Nanjing"
axiom thirty_degrees_northeast : ¬determine_location "30° northeast"
axiom longitude_latitude : determine_location "Longitude 118°E, Latitude 40°N"

theorem location_determined : determine_location "Longitude 118°E, Latitude 40°N" :=
longitude_latitude

end location_determined_l2045_204540


namespace k_cannot_be_zero_l2045_204575

theorem k_cannot_be_zero (k : ℝ) (h₁ : k ≠ 0) (h₂ : 4 - 2 * k > 0) : k ≠ 0 :=
by 
  exact h₁

end k_cannot_be_zero_l2045_204575


namespace shifted_parabola_expression_l2045_204588

theorem shifted_parabola_expression (x y x' y' : ℝ) 
  (h_initial : y = (x + 2)^2 + 3)
  (h_shift_right : x' = x - 3)
  (h_shift_down : y' = y - 2)
  : y' = (x' - 1)^2 + 1 := 
sorry

end shifted_parabola_expression_l2045_204588


namespace find_missing_number_l2045_204596

theorem find_missing_number (x : ℝ) (h : 0.00375 * x = 153.75) : x = 41000 :=
sorry

end find_missing_number_l2045_204596


namespace solution_of_fractional_equation_l2045_204535

theorem solution_of_fractional_equation :
  (∃ x, x ≠ 3 ∧ (x / (x - 3) - 2 = (m - 1) / (x - 3))) → m = 4 := by
  sorry

end solution_of_fractional_equation_l2045_204535


namespace ellipse_standard_equation_midpoint_trajectory_equation_l2045_204518

theorem ellipse_standard_equation :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ (∀ x y, (x, y) = (2, 0) → x^2 / a^2 + y^2 / b^2 = 1) → (a = 2 ∧ b = 1) :=
sorry

theorem midpoint_trajectory_equation :
  ∀ x y : ℝ,
  (∃ x0 y0 : ℝ, x0 = 2 * x - 1 ∧ y0 = 2 * y - 1 / 2 ∧ (x0^2 / 4 + y0^2 = 1)) →
  (x - 1 / 2)^2 + 4 * (y - 1 / 4)^2 = 1 :=
sorry

end ellipse_standard_equation_midpoint_trajectory_equation_l2045_204518


namespace bob_age_l2045_204500

variable {b j : ℝ}

theorem bob_age (h1 : b = 3 * j - 20) (h2 : b + j = 75) : b = 51 := by
  sorry

end bob_age_l2045_204500


namespace percent_daffodils_is_57_l2045_204552

-- Condition 1: Four-sevenths of the flowers are yellow
def fraction_yellow : ℚ := 4 / 7

-- Condition 2: Two-thirds of the red flowers are daffodils
def fraction_red_daffodils_given_red : ℚ := 2 / 3

-- Condition 3: Half of the yellow flowers are tulips
def fraction_yellow_tulips_given_yellow : ℚ := 1 / 2

-- Calculate fractions of yellow and red flowers
def fraction_red : ℚ := 1 - fraction_yellow

-- Calculate fractions of daffodils
def fraction_yellow_daffodils : ℚ := fraction_yellow * (1 - fraction_yellow_tulips_given_yellow)
def fraction_red_daffodils : ℚ := fraction_red * fraction_red_daffodils_given_red

-- Total fraction of daffodils
def fraction_daffodils : ℚ := fraction_yellow_daffodils + fraction_red_daffodils

-- Proof statement
theorem percent_daffodils_is_57 :
  fraction_daffodils * 100 = 57 := by
  sorry

end percent_daffodils_is_57_l2045_204552


namespace sqrt7_problem_l2045_204511

theorem sqrt7_problem (x y : ℝ) (h1 : 2 < Real.sqrt 7) (h2 : Real.sqrt 7 < 3) (hx : x = 2) (hy : y = Real.sqrt 7 - 2) :
  (x + Real.sqrt 7) * y = 3 :=
by
  sorry

end sqrt7_problem_l2045_204511


namespace total_apples_eaten_l2045_204599

def Apples_Tuesday : ℕ := 4
def Apples_Wednesday : ℕ := 2 * Apples_Tuesday
def Apples_Thursday : ℕ := Apples_Tuesday / 2

theorem total_apples_eaten : Apples_Tuesday + Apples_Wednesday + Apples_Thursday = 14 := by
  sorry

end total_apples_eaten_l2045_204599


namespace range_of_m_iff_l2045_204583

noncomputable def range_of_m (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (0 < x) → (0 < y) → ((2 / x) + (1 / y) = 1) → (x + 2 * y > m^2 + 2 * m)

theorem range_of_m_iff : (range_of_m m) ↔ (-4 < m ∧ m < 2) :=
  sorry

end range_of_m_iff_l2045_204583


namespace sufficient_but_not_necessary_l2045_204543

variable (p q : Prop)

theorem sufficient_but_not_necessary (h : p ∧ q) : ¬¬p :=
  by sorry -- Proof not required

end sufficient_but_not_necessary_l2045_204543


namespace area_of_rectangle_l2045_204556

theorem area_of_rectangle (x : ℝ) (hx : 0 < x) :
  let length := 3 * x - 1
  let width := 2 * x + 1 / 2
  let area := length * width
  area = 6 * x^2 - 1 / 2 * x - 1 / 2 :=
by
  sorry

end area_of_rectangle_l2045_204556


namespace smallest_n_div_75_has_75_divisors_l2045_204566

theorem smallest_n_div_75_has_75_divisors :
  ∃ n : ℕ, (n % 75 = 0) ∧ (n.factors.length = 75) ∧ (n / 75 = 432) :=
by
  sorry

end smallest_n_div_75_has_75_divisors_l2045_204566


namespace subway_ways_l2045_204528

theorem subway_ways (total_ways : ℕ) (bus_ways : ℕ) (h1 : total_ways = 7) (h2 : bus_ways = 4) :
  total_ways - bus_ways = 3 :=
by
  sorry

end subway_ways_l2045_204528


namespace tan_alpha_minus_pi_over_4_l2045_204533

noncomputable def alpha : ℝ := sorry
axiom alpha_in_range : -Real.pi / 2 < alpha ∧ alpha < 0
axiom cos_alpha : Real.cos alpha = (Real.sqrt 5) / 5

theorem tan_alpha_minus_pi_over_4 : Real.tan (alpha - Real.pi / 4) = 3 := by
  sorry

end tan_alpha_minus_pi_over_4_l2045_204533


namespace log_expression_l2045_204513

variable (a : ℝ) (log3 : ℝ → ℝ)
axiom h_a : a = log3 2
axiom log3_8_eq : log3 8 = 3 * log3 2
axiom log3_6_eq : log3 6 = log3 2 + 1

theorem log_expression (log_def : log3 8 - 2 * log3 6 = a - 2) :
  log3 8 - 2 * log3 6 = a - 2 := by
  sorry

end log_expression_l2045_204513


namespace subset_iff_l2045_204587

open Set

noncomputable def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | 0 < x ∧ x < a}

theorem subset_iff (a : ℝ) : A ⊆ B a ↔ 2 ≤ a :=
by sorry

end subset_iff_l2045_204587


namespace geometric_sequence_value_a3_l2045_204524

-- Define the geometric sequence
def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

-- Conditions given in the problem
variable (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 2)
variable (h₂ : (geometric_sequence a₁ q 4) * (geometric_sequence a₁ q 6) = 4 * (geometric_sequence a₁ q 7) ^ 2)

-- The goal is to prove that a₃ = 1
theorem geometric_sequence_value_a3 : geometric_sequence a₁ q 3 = 1 :=
by
  sorry

end geometric_sequence_value_a3_l2045_204524


namespace michael_remaining_books_l2045_204567

theorem michael_remaining_books (total_books : ℕ) (read_percentage : ℚ) 
  (H1 : total_books = 210) (H2 : read_percentage = 0.60) : 
  (total_books - (read_percentage * total_books) : ℚ) = 84 :=
by
  sorry

end michael_remaining_books_l2045_204567


namespace least_clock_equiv_square_l2045_204526

def clock_equiv (h k : ℕ) : Prop := (h - k) % 24 = 0

theorem least_clock_equiv_square : ∃ (h : ℕ), h > 6 ∧ (h^2) % 24 = h % 24 ∧ (∀ (k : ℕ), k > 6 ∧ clock_equiv k (k^2) → h ≤ k) :=
sorry

end least_clock_equiv_square_l2045_204526


namespace quadratic_inequality_solution_range_l2045_204549

theorem quadratic_inequality_solution_range (k : ℝ) :
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3 / 8 < 0) ↔ (-3 / 2 < k ∧ k < 0) := sorry

end quadratic_inequality_solution_range_l2045_204549


namespace min_students_changed_l2045_204546

-- Define the initial percentage of "Yes" and "No" at the beginning of the year
def initial_yes_percentage : ℝ := 0.40
def initial_no_percentage : ℝ := 0.60

-- Define the final percentage of "Yes" and "No" at the end of the year
def final_yes_percentage : ℝ := 0.80
def final_no_percentage : ℝ := 0.20

-- Define the minimum possible percentage of students that changed their mind
def min_changed_percentage : ℝ := 0.40

-- Prove that the minimum possible percentage of students that changed their mind is 40%
theorem min_students_changed :
  (final_yes_percentage - initial_yes_percentage = min_changed_percentage) ∧
  (initial_yes_percentage = final_yes_percentage - min_changed_percentage) ∧
  (initial_no_percentage - min_changed_percentage = final_no_percentage) :=
by
  sorry

end min_students_changed_l2045_204546


namespace diana_took_six_candies_l2045_204503

-- Define the initial number of candies in the box
def initial_candies : ℕ := 88

-- Define the number of candies left in the box after Diana took some
def remaining_candies : ℕ := 82

-- Define the number of candies taken by Diana
def candies_taken : ℕ := initial_candies - remaining_candies

-- The theorem we need to prove
theorem diana_took_six_candies : candies_taken = 6 := by
  sorry

end diana_took_six_candies_l2045_204503


namespace problem_statement_l2045_204507

theorem problem_statement (a b c d : ℤ) (h1 : a - b = -3) (h2 : c + d = 2) : (b + c) - (a - d) = 5 :=
by
  -- Proof steps skipped.
  sorry

end problem_statement_l2045_204507


namespace fraction_problem_l2045_204545

theorem fraction_problem (N D : ℚ) (h1 : 1.30 * N / (0.85 * D) = 25 / 21) : 
  N / D = 425 / 546 :=
sorry

end fraction_problem_l2045_204545


namespace yuebao_scientific_notation_l2045_204542

-- Definition of converting a number to scientific notation
def scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10 ^ n

-- The specific problem statement
theorem yuebao_scientific_notation :
  scientific_notation (1853 * 10 ^ 9) 1.853 11 :=
by
  sorry

end yuebao_scientific_notation_l2045_204542


namespace part_a_part_b_part_c_l2045_204584

def transformable (w1 w2 : String) : Prop :=
∀ q : String → String → Prop,
  (q "xy" "yyx") →
  (q "xt" "ttx") →
  (q "yt" "ty") →
  (q w1 w2)

theorem part_a : ¬ transformable "xy" "xt" :=
sorry

theorem part_b : ¬ transformable "xytx" "txyt" :=
sorry

theorem part_c : transformable "xtxyy" "ttxyyyyx" :=
sorry

end part_a_part_b_part_c_l2045_204584


namespace binom_7_4_l2045_204569

theorem binom_7_4 : Nat.choose 7 4 = 35 := 
by
  sorry

end binom_7_4_l2045_204569


namespace water_formed_on_combining_l2045_204550

theorem water_formed_on_combining (molar_mass_water : ℝ) (n_NaOH : ℝ) (n_HCl : ℝ) :
  n_NaOH = 1 ∧ n_HCl = 1 ∧ molar_mass_water = 18.01528 → 
  n_NaOH * molar_mass_water = 18.01528 :=
by sorry

end water_formed_on_combining_l2045_204550


namespace solve_system_l2045_204539

def eq1 (x y : ℝ) : Prop := x^2 * y - x * y^2 - 5 * x + 5 * y + 3 = 0
def eq2 (x y : ℝ) : Prop := x^3 * y - x * y^3 - 5 * x^2 + 5 * y^2 + 15 = 0

theorem solve_system :
  ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧ x = 4 ∧ y = 1 := by
  sorry

end solve_system_l2045_204539


namespace absent_children_l2045_204551

/-- On a school's annual day, sweets were to be equally distributed amongst 112 children. 
But on that particular day, some children were absent. Thus, the remaining children got 6 extra sweets. 
Each child was originally supposed to get 15 sweets. Prove that 32 children were absent. -/
theorem absent_children (A : ℕ) 
  (total_children : ℕ := 112) 
  (sweets_per_child : ℕ := 15) 
  (extra_sweets : ℕ := 6)
  (absent_eq : (total_children - A) * (sweets_per_child + extra_sweets) = total_children * sweets_per_child) : 
  A = 32 := 
by
  sorry

end absent_children_l2045_204551


namespace extended_morse_code_symbols_l2045_204521

def symbol_count (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 2
  else if n = 3 then 1
  else if n = 4 then 1 + 4 + 1
  else if n = 5 then 1 + 8
  else 0

theorem extended_morse_code_symbols : 
  (symbol_count 1 + symbol_count 2 + symbol_count 3 + symbol_count 4 + symbol_count 5) = 20 :=
by sorry

end extended_morse_code_symbols_l2045_204521


namespace find_base_l2045_204523

theorem find_base (b x y : ℝ) (h₁ : b^x * 4^y = 59049) (h₂ : x = 10) (h₃ : x - y = 10) : b = 3 :=
by
  sorry

end find_base_l2045_204523


namespace new_solid_edges_l2045_204534

-- Definitions based on conditions
def original_vertices : ℕ := 8
def original_edges : ℕ := 12
def new_edges_per_vertex : ℕ := 3
def number_of_vertices : ℕ := original_vertices

-- Conclusion to prove
theorem new_solid_edges : 
  (original_edges + new_edges_per_vertex * number_of_vertices) = 36 := 
by
  sorry

end new_solid_edges_l2045_204534


namespace exists_ab_negated_l2045_204571

theorem exists_ab_negated :
  ¬ (∀ a b : ℝ, (a + b = 0 → a^2 + b^2 = 0)) ↔ 
  ∃ a b : ℝ, (a + b = 0 ∧ a^2 + b^2 ≠ 0) :=
by
  sorry

end exists_ab_negated_l2045_204571


namespace smallest_integer_n_l2045_204585

theorem smallest_integer_n (n : ℕ) (h : ∃ k : ℕ, 432 * n = k ^ 2) : n = 3 := 
sorry

end smallest_integer_n_l2045_204585


namespace maximal_value_fraction_l2045_204563

noncomputable def maximum_value_ratio (a b c : ℝ) (S : ℝ) : ℝ :=
  if S = c^2 / 4 then 2 * Real.sqrt 2 else 0

theorem maximal_value_fraction (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (area_cond : 1/2 * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = c^2 / 4) :
  maximum_value_ratio a b c (c^2/4) = 2 * Real.sqrt 2 :=
sorry

end maximal_value_fraction_l2045_204563


namespace larger_exceeds_smaller_times_l2045_204593

theorem larger_exceeds_smaller_times {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h_diff : a ≠ b)
  (h_eq : a^3 - b^3 = 3 * (2 * a^2 * b - 3 * a * b^2 + b^3)) : a = 4 * b :=
sorry

end larger_exceeds_smaller_times_l2045_204593


namespace parabola_has_one_x_intercept_l2045_204506

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- State the theorem that proves the number of x-intercepts
theorem parabola_has_one_x_intercept : ∃! x, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
by
  -- Proof goes here, but it's omitted
  sorry

end parabola_has_one_x_intercept_l2045_204506


namespace remainder_of_3a_minus_b_divided_by_5_l2045_204531

theorem remainder_of_3a_minus_b_divided_by_5 (a b : ℕ) (m n : ℤ) 
(h1 : 3 * a > b) 
(h2 : a = 5 * m + 1) 
(h3 : b = 5 * n + 4) : 
(3 * a - b) % 5 = 4 := 
sorry

end remainder_of_3a_minus_b_divided_by_5_l2045_204531


namespace probability_of_rain_l2045_204554

theorem probability_of_rain {p : ℝ} (h : p = 0.95) :
  ∃ (q : ℝ), q = (1 - p) ∧ q < p :=
by
  sorry

end probability_of_rain_l2045_204554


namespace average_of_first_5_subjects_l2045_204577

theorem average_of_first_5_subjects (avg_6_subjects : ℚ) (marks_6th_subject : ℚ) (total_subjects : ℕ) (total_marks_6_subjects : ℚ) (total_marks_5_subjects : ℚ) (avg_5_subjects : ℚ) :
  avg_6_subjects = 77 ∧ marks_6th_subject = 92 ∧ total_subjects = 6 ∧ total_marks_6_subjects = avg_6_subjects * total_subjects ∧ total_marks_5_subjects = total_marks_6_subjects - marks_6th_subject ∧ avg_5_subjects = total_marks_5_subjects / 5
  → avg_5_subjects = 74 := by
  sorry

end average_of_first_5_subjects_l2045_204577


namespace speed_in_kmph_l2045_204527

noncomputable def speed_conversion (speed_mps: ℝ) : ℝ :=
  speed_mps * 3.6

theorem speed_in_kmph : speed_conversion 18.334799999999998 = 66.00528 :=
by
  -- proof steps would go here
  sorry

end speed_in_kmph_l2045_204527


namespace cds_total_l2045_204591

theorem cds_total (dawn_cds : ℕ) (h1 : dawn_cds = 10) (h2 : ∀ kristine_cds : ℕ, kristine_cds = dawn_cds + 7) :
  dawn_cds + (dawn_cds + 7) = 27 :=
by
  sorry

end cds_total_l2045_204591


namespace simplify_and_evaluate_l2045_204516

-- Define the expression as a function of a and b
def expr (a b : ℚ) : ℚ := 5 * a * b - 2 * (3 * a * b - (4 * a * b^2 + (1/2) * a * b)) - 5 * a * b^2

-- State the condition and the target result
theorem simplify_and_evaluate : 
  let a : ℚ := -1
  let b : ℚ := 1 / 2
  expr a b = -3 / 4 :=
by
  -- Proof goes here
  sorry

end simplify_and_evaluate_l2045_204516


namespace find_m_l2045_204582

theorem find_m (m : ℤ) (x y : ℤ) (h1 : x = 1) (h2 : y = m) (h3 : 3 * x - 4 * y = 7) : m = -1 :=
by
  sorry

end find_m_l2045_204582


namespace find_n_l2045_204590

theorem find_n :
  ∃ n : ℕ, 50 ≤ n ∧ n ≤ 150 ∧
          n % 7 = 0 ∧
          n % 9 = 3 ∧
          n % 6 = 3 ∧
          n = 75 :=
by
  sorry

end find_n_l2045_204590


namespace fraction_identity_l2045_204519

variables {a b c x : ℝ}

theorem fraction_identity (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ c) (h4 : c ≠ a) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a + 2 * b + 3 * c) / (a - b - 3 * c) = (b * (x + 2) + 3 * c) / (b * (x - 1) - 3 * c) :=
by {
  sorry
}

end fraction_identity_l2045_204519


namespace important_emails_l2045_204537

theorem important_emails (total_emails : ℕ) (spam_frac : ℚ) (promotional_frac : ℚ) (spam_email_count : ℕ) (remaining_emails : ℕ) (promotional_email_count : ℕ) (important_email_count : ℕ) :
  total_emails = 800 ∧ spam_frac = 3 / 7 ∧ promotional_frac = 5 / 11 ∧ spam_email_count = 343 ∧ remaining_emails = 457 ∧ promotional_email_count = 208 →
sorry

end important_emails_l2045_204537


namespace volume_of_cube_is_correct_l2045_204579

-- Define necessary constants and conditions
def cost_in_paise : ℕ := 34398
def rate_per_sq_cm : ℕ := 13
def surface_area : ℕ := cost_in_paise / rate_per_sq_cm
def face_area : ℕ := surface_area / 6
def side_length : ℕ := Nat.sqrt face_area
def volume : ℕ := side_length ^ 3

-- Prove the volume of the cube
theorem volume_of_cube_is_correct : volume = 9261 := by
  -- Using given conditions and basic arithmetic 
  sorry

end volume_of_cube_is_correct_l2045_204579


namespace problem1_problem2_l2045_204514

open Real

theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  sqrt a + sqrt b ≤ 2 :=
sorry

theorem problem2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (a + b^3) * (a^3 + b) ≥ 4 :=
sorry

end problem1_problem2_l2045_204514


namespace tan_20_add_4sin_20_eq_sqrt3_l2045_204565

theorem tan_20_add_4sin_20_eq_sqrt3 : Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180) = Real.sqrt 3 := 
by
  sorry

end tan_20_add_4sin_20_eq_sqrt3_l2045_204565


namespace base8_subtraction_correct_l2045_204532

noncomputable def base8_subtraction (x y : Nat) : Nat :=
  if y > x then 0 else x - y

theorem base8_subtraction_correct :
  base8_subtraction 546 321 - 105 = 120 :=
by
  -- Given the condition that all arithmetic is in base 8
  sorry

end base8_subtraction_correct_l2045_204532


namespace cos_210_eq_neg_sqrt_3_div_2_l2045_204536

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end cos_210_eq_neg_sqrt_3_div_2_l2045_204536


namespace cost_per_load_is_25_cents_l2045_204576

def washes_per_bottle := 80
def price_per_bottle_on_sale := 20
def bottles := 2
def total_cost := bottles * price_per_bottle_on_sale -- 2 * 20 = 40
def total_loads := bottles * washes_per_bottle -- 2 * 80 = 160
def cost_per_load_in_dollars := total_cost / total_loads -- 40 / 160 = 0.25
def cost_per_load_in_cents := cost_per_load_in_dollars * 100

theorem cost_per_load_is_25_cents :
  cost_per_load_in_cents = 25 :=
by 
  sorry

end cost_per_load_is_25_cents_l2045_204576


namespace height_inequality_triangle_l2045_204578

theorem height_inequality_triangle (a b c h_a h_b h_c Δ : ℝ) (n : ℝ) 
  (ha : h_a = 2 * Δ / a)
  (hb : h_b = 2 * Δ / b)
  (hc : h_c = 2 * Δ / c)
  (n_pos : n > 0) :
  (a * h_b)^n + (b * h_c)^n + (c * h_a)^n ≥ 3 * 2^n * Δ^n := 
sorry

end height_inequality_triangle_l2045_204578


namespace reciprocal_is_correct_l2045_204568

-- Define the initial number
def num : ℚ := -1 / 2023

-- Define the expected reciprocal
def reciprocal : ℚ := -2023

-- Theorem stating the reciprocal of the given number is the expected reciprocal
theorem reciprocal_is_correct : 1 / num = reciprocal :=
  by
    -- The actual proof can be filled in here
    sorry

end reciprocal_is_correct_l2045_204568


namespace range_of_m_satisfies_inequality_l2045_204586

theorem range_of_m_satisfies_inequality (m : ℝ) :
  ((∀ x : ℝ, (1 - m^2) * x^2 - (1 + m) * x - 1 < 0) ↔ (m ≤ -1 ∨ m > 5/3)) :=
sorry

end range_of_m_satisfies_inequality_l2045_204586


namespace average_productivity_l2045_204560

theorem average_productivity (T : ℕ) (total_words : ℕ) (increased_time_fraction : ℚ) (increased_productivity_fraction : ℚ) :
  T = 100 →
  total_words = 60000 →
  increased_time_fraction = 0.2 →
  increased_productivity_fraction = 1.5 →
  (total_words / T : ℚ) = 600 :=
by
  sorry

end average_productivity_l2045_204560


namespace round_to_nearest_hundredth_l2045_204508

noncomputable def recurring_decimal (n : ℕ) : ℝ :=
  if n = 87 then 87 + 36 / 99 else 0 -- Defines 87.3636... for n = 87

theorem round_to_nearest_hundredth : recurring_decimal 87 = 87.36 :=
by sorry

end round_to_nearest_hundredth_l2045_204508


namespace point_divides_segment_in_ratio_l2045_204597

theorem point_divides_segment_in_ratio (A B C C1 A1 P : Type) 
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] 
  [AddCommGroup C1] [AddCommGroup A1] [AddCommGroup P]
  (h1 : AP / PA1 = 3 / 2)
  (h2 : CP / PC1 = 2 / 1) :
  AC1 / C1B = 2 / 3 :=
sorry

end point_divides_segment_in_ratio_l2045_204597


namespace jasmine_pies_l2045_204530

-- Definitions based on the given conditions
def total_pies : Nat := 30
def raspberry_part : Nat := 2
def peach_part : Nat := 5
def plum_part : Nat := 3
def total_parts : Nat := raspberry_part + peach_part + plum_part

-- Calculate pies per part
def pies_per_part : Nat := total_pies / total_parts

-- Prove the statement
theorem jasmine_pies :
  (plum_part * pies_per_part = 9) :=
by
  -- The statement and proof will go here, but we are skipping the proof part.
  sorry

end jasmine_pies_l2045_204530


namespace hyperbola_eccentricity_b_value_l2045_204520

theorem hyperbola_eccentricity_b_value (b : ℝ) (a : ℝ) (e : ℝ) 
  (h1 : a^2 = 1) (h2 : e = 2) 
  (h3 : b > 0) (h4 : b^2 = 4 - 1) : 
  b = Real.sqrt 3 := 
by 
  sorry

end hyperbola_eccentricity_b_value_l2045_204520


namespace inequality_holds_for_all_x_l2045_204594

theorem inequality_holds_for_all_x (m : ℝ) :
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ -9 < m ∧ m < -1 :=
by
  sorry

end inequality_holds_for_all_x_l2045_204594


namespace blocks_found_l2045_204564

def initial_blocks : ℕ := 2
def final_blocks : ℕ := 86

theorem blocks_found : (final_blocks - initial_blocks) = 84 :=
by
  sorry

end blocks_found_l2045_204564


namespace ratio_of_x_and_y_l2045_204505

theorem ratio_of_x_and_y (x y : ℝ) (h : 0.80 * x = 0.20 * y) : x / y = 0.25 :=
by
  sorry

end ratio_of_x_and_y_l2045_204505


namespace range_of_a_l2045_204570

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

theorem range_of_a (a : ℝ) :
  (∀ (b : ℝ), (b ≤ 0) → ∀ (x : ℝ), (x > Real.exp 1 ∧ x ≤ Real.exp 2) → f a b x ≥ x) →
  a ≥ Real.exp 2 / 2 :=
by
  sorry

end range_of_a_l2045_204570


namespace marie_messages_days_l2045_204558

theorem marie_messages_days (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) (days : ℕ) :
  initial_messages = 98 ∧ read_per_day = 20 ∧ new_per_day = 6 → days = 7 :=
by
  sorry

end marie_messages_days_l2045_204558


namespace problem1_l2045_204515

theorem problem1 (α β : ℝ) 
  (tan_sum : Real.tan (α + β) = 2 / 5) 
  (tan_diff : Real.tan (β - Real.pi / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 := 
sorry

end problem1_l2045_204515


namespace Julia_played_with_11_kids_on_Monday_l2045_204553

theorem Julia_played_with_11_kids_on_Monday
  (kids_on_Tuesday : ℕ)
  (kids_on_Monday : ℕ) 
  (h1 : kids_on_Tuesday = 12)
  (h2 : kids_on_Tuesday = kids_on_Monday + 1) : 
  kids_on_Monday = 11 := 
by
  sorry

end Julia_played_with_11_kids_on_Monday_l2045_204553


namespace circle_transformation_l2045_204555

theorem circle_transformation (c : ℝ × ℝ) (v : ℝ × ℝ) (h_center : c = (8, -3)) (h_vector : v = (2, -5)) :
  let reflected := (c.2, c.1)
  let translated := (reflected.1 + v.1, reflected.2 + v.2)
  translated = (-1, 3) :=
by
  sorry

end circle_transformation_l2045_204555


namespace factorization_result_l2045_204548

theorem factorization_result (a b : ℤ) (h : (16:ℚ) * x^2 - 106 * x - 105 = (8 * x + a) * (2 * x + b)) : a + 2 * b = -23 := by
  sorry

end factorization_result_l2045_204548


namespace product_of_last_two_digits_l2045_204574

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 11) (h2 : ∃ (n : ℕ), 10 * A + B = 6 * n) : A * B = 24 :=
sorry

end product_of_last_two_digits_l2045_204574
