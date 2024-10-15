import Mathlib

namespace NUMINAMATH_GPT_pam_age_l767_76762

-- Given conditions:
-- 1. Pam is currently twice as young as Rena.
-- 2. In 10 years, Rena will be 5 years older than Pam.

variable (Pam Rena : ℕ)

theorem pam_age
  (h1 : 2 * Pam = Rena)
  (h2 : Rena + 10 = Pam + 15)
  : Pam = 5 := 
sorry

end NUMINAMATH_GPT_pam_age_l767_76762


namespace NUMINAMATH_GPT_find_m_range_l767_76778

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | -1 ≤ x ∧ x ≤ m + 1 }

theorem find_m_range (m : ℝ) : (B m ⊆ A) ↔ (-2 ≤ m ∧ m ≤ 3) := by
  sorry

end NUMINAMATH_GPT_find_m_range_l767_76778


namespace NUMINAMATH_GPT_min_value_fract_ineq_l767_76786

theorem min_value_fract_ineq (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (1 / a + 9 / b) ≥ 16 := 
sorry

end NUMINAMATH_GPT_min_value_fract_ineq_l767_76786


namespace NUMINAMATH_GPT_at_least_two_equal_l767_76742

theorem at_least_two_equal (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b^2 + c^2 = b + a^2 + c^2) (h2 : b + a^2 + c^2 = c + a^2 + b^2) : 
  (a = b) ∨ (a = c) ∨ (b = c) :=
sorry

end NUMINAMATH_GPT_at_least_two_equal_l767_76742


namespace NUMINAMATH_GPT_intersecting_lines_l767_76749

theorem intersecting_lines (c d : ℝ)
  (h1 : 16 = 2 * 4 + c)
  (h2 : 16 = 5 * 4 + d) :
  c + d = 4 :=
sorry

end NUMINAMATH_GPT_intersecting_lines_l767_76749


namespace NUMINAMATH_GPT_Bryce_grapes_l767_76709

theorem Bryce_grapes : 
  ∃ x : ℝ, (∀ y : ℝ, y = (1/3) * x → y = x - 7) → x = 21 / 2 :=
by
  sorry

end NUMINAMATH_GPT_Bryce_grapes_l767_76709


namespace NUMINAMATH_GPT_cone_radius_l767_76745

theorem cone_radius
    (l : ℝ) (n : ℝ) (r : ℝ)
    (h1 : l = 2 * Real.pi)
    (h2 : n = 120)
    (h3 : l = (n * Real.pi * r) / 180 ) :
    r = 3 :=
sorry

end NUMINAMATH_GPT_cone_radius_l767_76745


namespace NUMINAMATH_GPT_number_of_five_digit_numbers_with_at_least_one_zero_l767_76737

-- Definitions for the conditions
def total_five_digit_numbers : ℕ := 90000
def five_digit_numbers_with_no_zeros : ℕ := 59049

-- Theorem stating that the number of 5-digit numbers with at least one zero is 30,951
theorem number_of_five_digit_numbers_with_at_least_one_zero : 
    total_five_digit_numbers - five_digit_numbers_with_no_zeros = 30951 :=
by
  sorry

end NUMINAMATH_GPT_number_of_five_digit_numbers_with_at_least_one_zero_l767_76737


namespace NUMINAMATH_GPT_total_salaries_l767_76716

variable (A_salary B_salary : ℝ)

def A_saves : ℝ := 0.05 * A_salary
def B_saves : ℝ := 0.15 * B_salary

theorem total_salaries (h1 : A_salary = 5250) 
                       (h2 : A_saves = B_saves) : 
    A_salary + B_salary = 7000 := by
  sorry

end NUMINAMATH_GPT_total_salaries_l767_76716


namespace NUMINAMATH_GPT_juniper_initial_bones_l767_76799

theorem juniper_initial_bones (B : ℕ) (h : 2 * B - 2 = 6) : B = 4 := 
by
  sorry

end NUMINAMATH_GPT_juniper_initial_bones_l767_76799


namespace NUMINAMATH_GPT_bicycle_distance_l767_76797

theorem bicycle_distance (P_b P_f : ℝ) (h1 : P_b = 9) (h2 : P_f = 7) (h3 : ∀ D : ℝ, D / P_f = D / P_b + 10) :
  315 = 315 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_distance_l767_76797


namespace NUMINAMATH_GPT_binary_representation_of_38_l767_76732

theorem binary_representation_of_38 : ∃ binary : ℕ, binary = 0b100110 ∧ binary = 38 :=
by
  sorry

end NUMINAMATH_GPT_binary_representation_of_38_l767_76732


namespace NUMINAMATH_GPT_unique_prime_sum_diff_l767_76730

-- Define that p is a prime number that satisfies both conditions
def sum_two_primes (p a b : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime a ∧ Nat.Prime b ∧ p = a + b

def diff_two_primes (p c d : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime c ∧ Nat.Prime d ∧ p = c - d

-- Main theorem to prove: The only prime p that satisfies both conditions is 5
theorem unique_prime_sum_diff (p : ℕ) :
  (∃ a b, sum_two_primes p a b) ∧ (∃ c d, diff_two_primes p c d) → p = 5 :=
by
  sorry

end NUMINAMATH_GPT_unique_prime_sum_diff_l767_76730


namespace NUMINAMATH_GPT_amanda_average_speed_l767_76788

def amanda_distance1 : ℝ := 450
def amanda_time1 : ℝ := 7.5
def amanda_distance2 : ℝ := 420
def amanda_time2 : ℝ := 7

def total_distance : ℝ := amanda_distance1 + amanda_distance2
def total_time : ℝ := amanda_time1 + amanda_time2
def expected_average_speed : ℝ := 60

theorem amanda_average_speed :
  (total_distance / total_time) = expected_average_speed := by
  sorry

end NUMINAMATH_GPT_amanda_average_speed_l767_76788


namespace NUMINAMATH_GPT_find_x_y_z_l767_76718

theorem find_x_y_z (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x * y = x + y) (h2 : y * z = 3 * (y + z)) (h3 : z * x = 2 * (z + x)) : 
  x + y + z = 12 :=
sorry

end NUMINAMATH_GPT_find_x_y_z_l767_76718


namespace NUMINAMATH_GPT_Tammy_average_speed_second_day_l767_76779

theorem Tammy_average_speed_second_day : 
  ∀ (t v : ℝ), 
    (t + (t - 2) + (t + 1) = 20) → 
    (7 * v + 5 * (v + 0.5) + 8 * (v + 1.5) = 85) → 
    (v + 0.5 = 4.025) := 
by 
  intros t v ht hv 
  sorry

end NUMINAMATH_GPT_Tammy_average_speed_second_day_l767_76779


namespace NUMINAMATH_GPT_vasya_numbers_l767_76712

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end NUMINAMATH_GPT_vasya_numbers_l767_76712


namespace NUMINAMATH_GPT_jorge_spent_amount_l767_76707

theorem jorge_spent_amount
  (num_tickets : ℕ)
  (price_per_ticket : ℕ)
  (discount_percentage : ℚ)
  (h1 : num_tickets = 24)
  (h2 : price_per_ticket = 7)
  (h3 : discount_percentage = 0.5) :
  num_tickets * price_per_ticket * (1 - discount_percentage) = 84 := 
by
  simp [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_jorge_spent_amount_l767_76707


namespace NUMINAMATH_GPT_calculate_expression_l767_76714

theorem calculate_expression : 
  (3^2 - 2 * 3) - (5^2 - 2 * 5) + (7^2 - 2 * 7) = 23 := 
by sorry

end NUMINAMATH_GPT_calculate_expression_l767_76714


namespace NUMINAMATH_GPT_reduced_price_per_kg_l767_76704

-- Assume the constants in the conditions
variables (P R : ℝ)
variables (h1 : R = P - 0.40 * P) -- R = 0.60P
variables (h2 : 2000 / P + 10 = 2000 / R) -- extra 10 kg for the same 2000 rs

-- State the target we want to prove
theorem reduced_price_per_kg : R = 80 :=
by
  -- The steps and details of the proof
  sorry

end NUMINAMATH_GPT_reduced_price_per_kg_l767_76704


namespace NUMINAMATH_GPT_maximize_tables_eqn_l767_76723

theorem maximize_tables_eqn :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 12 → 400 * x = 20 * (12 - x) * 4 :=
by
  sorry

end NUMINAMATH_GPT_maximize_tables_eqn_l767_76723


namespace NUMINAMATH_GPT_xy_sum_l767_76775

theorem xy_sum (x y : ℝ) (h1 : x^3 - 6 * x^2 + 15 * x = 12) (h2 : y^3 - 6 * y^2 + 15 * y = 16) : x + y = 4 := 
sorry

end NUMINAMATH_GPT_xy_sum_l767_76775


namespace NUMINAMATH_GPT_proof_problem_l767_76780

theorem proof_problem (a b : ℝ) (H1 : ∀ x : ℝ, (ax^2 - 3*x + 6 > 4) ↔ (x < 1 ∨ x > b)) :
  a = 1 ∧ b = 2 ∧
  (∀ c : ℝ, (ax^2 - (a*c + b)*x + b*c < 0) ↔ 
   (if c > 2 then 2 < x ∧ x < c
    else if c < 2 then c < x ∧ x < 2
    else false)) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l767_76780


namespace NUMINAMATH_GPT_concrete_pillars_correct_l767_76721

-- Definitions based on conditions
def concrete_for_roadway := 1600
def concrete_for_one_anchor := 700
def total_concrete_for_bridge := 4800

-- Total concrete for both anchors
def concrete_for_both_anchors := 2 * concrete_for_one_anchor

-- Total concrete needed for the roadway and anchors
def concrete_for_roadway_and_anchors := concrete_for_roadway + concrete_for_both_anchors

-- Concrete needed for the supporting pillars
def concrete_for_pillars := total_concrete_for_bridge - concrete_for_roadway_and_anchors

-- Proof problem statement, verify that the concrete for the supporting pillars is 1800 tons
theorem concrete_pillars_correct : concrete_for_pillars = 1800 := by
  sorry

end NUMINAMATH_GPT_concrete_pillars_correct_l767_76721


namespace NUMINAMATH_GPT_expression_value_l767_76761

theorem expression_value (x y z w : ℝ) (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  (2 * x + y) * (2 * z + w) = 15 := 
sorry

end NUMINAMATH_GPT_expression_value_l767_76761


namespace NUMINAMATH_GPT_find_a_l767_76715

theorem find_a
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 2 * (b * Real.cos A + a * Real.cos B) = c^2)
  (h2 : b = 3)
  (h3 : 3 * Real.cos A = 1) :
  a = 3 :=
sorry

end NUMINAMATH_GPT_find_a_l767_76715


namespace NUMINAMATH_GPT_new_persons_joined_l767_76738

theorem new_persons_joined (initial_avg_age new_avg_age initial_total new_avg_age_total final_avg_age final_total : ℝ) 
  (n_initial n_new : ℕ) 
  (h1 : initial_avg_age = 16)
  (h2 : n_initial = 20)
  (h3 : new_avg_age = 15)
  (h4 : final_avg_age = 15.5)
  (h5 : initial_total = initial_avg_age * n_initial)
  (h6 : new_avg_age_total = new_avg_age * (n_new : ℝ))
  (h7 : final_total = initial_total + new_avg_age_total)
  (h8 : final_total = final_avg_age * (n_initial + n_new)) 
  : n_new = 20 :=
by
  sorry

end NUMINAMATH_GPT_new_persons_joined_l767_76738


namespace NUMINAMATH_GPT_cups_of_rice_in_afternoon_l767_76772

-- Definitions for conditions
def morning_cups : ℕ := 3
def evening_cups : ℕ := 5
def fat_per_cup : ℕ := 10
def weekly_total_fat : ℕ := 700

-- Theorem statement
theorem cups_of_rice_in_afternoon (morning_cups evening_cups fat_per_cup weekly_total_fat : ℕ) :
  (weekly_total_fat - (morning_cups + evening_cups) * fat_per_cup * 7) / fat_per_cup = 14 :=
by
  sorry

end NUMINAMATH_GPT_cups_of_rice_in_afternoon_l767_76772


namespace NUMINAMATH_GPT_spending_after_drink_l767_76767

variable (X : ℝ)
variable (Y : ℝ)

theorem spending_after_drink (h : X - 1.75 - Y = 6) : Y = X - 7.75 :=
by sorry

end NUMINAMATH_GPT_spending_after_drink_l767_76767


namespace NUMINAMATH_GPT_stickers_initial_count_l767_76740

theorem stickers_initial_count (S : ℕ) 
  (h1 : (3 / 5 : ℝ) * (2 / 3 : ℝ) * S = 54) : S = 135 := 
by
  sorry

end NUMINAMATH_GPT_stickers_initial_count_l767_76740


namespace NUMINAMATH_GPT_a_is_minus_one_l767_76771

theorem a_is_minus_one (a : ℤ) (h1 : 2 * a + 1 < 0) (h2 : 2 + a > 0) : a = -1 := 
by
  sorry

end NUMINAMATH_GPT_a_is_minus_one_l767_76771


namespace NUMINAMATH_GPT_length_of_second_platform_l767_76703

-- Definitions
def length_train : ℝ := 230
def time_first_platform : ℝ := 15
def length_first_platform : ℝ := 130
def total_distance_first_platform : ℝ := length_train + length_first_platform
def time_second_platform : ℝ := 20

-- Statement to prove
theorem length_of_second_platform : 
  ∃ L : ℝ, (total_distance_first_platform / time_first_platform) = ((length_train + L) / time_second_platform) ∧ L = 250 :=
by
  sorry

end NUMINAMATH_GPT_length_of_second_platform_l767_76703


namespace NUMINAMATH_GPT_pi_approx_by_jews_l767_76734

theorem pi_approx_by_jews (S D C : ℝ) (h1 : 4 * S = (5 / 4) * C) (h2 : D = S) (h3 : C = π * D) : π = 3 := by
  sorry

end NUMINAMATH_GPT_pi_approx_by_jews_l767_76734


namespace NUMINAMATH_GPT_intersection_eq_l767_76711

-- Define the sets M and N
def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := { x | -1 ≤ x ∧ x ≤ 1 }

-- The statement to prove
theorem intersection_eq : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l767_76711


namespace NUMINAMATH_GPT_frood_points_l767_76764

theorem frood_points (n : ℕ) (h : n > 29) : (n * (n + 1) / 2) > 15 * n := by
  sorry

end NUMINAMATH_GPT_frood_points_l767_76764


namespace NUMINAMATH_GPT_sum_of_digits_l767_76798

theorem sum_of_digits :
  ∃ (a b : ℕ), (4 * 100 + a * 10 + 5) + 457 = (9 * 100 + b * 10 + 2) ∧
                (((9 + 2) - b) % 11 = 0) ∧
                (a + b = 4) :=
sorry

end NUMINAMATH_GPT_sum_of_digits_l767_76798


namespace NUMINAMATH_GPT_sum_of_coefficients_of_expansion_l767_76746

theorem sum_of_coefficients_of_expansion (x y : ℝ) :
  (3*x - 4*y) ^ 20 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_of_expansion_l767_76746


namespace NUMINAMATH_GPT_binom_eight_three_l767_76787

theorem binom_eight_three : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_GPT_binom_eight_three_l767_76787


namespace NUMINAMATH_GPT_simplify_fraction_l767_76719

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l767_76719


namespace NUMINAMATH_GPT_distribution_plans_equiv_210_l767_76735

noncomputable def number_of_distribution_plans : ℕ := sorry -- we will skip the proof

theorem distribution_plans_equiv_210 :
  number_of_distribution_plans = 210 := by
  sorry

end NUMINAMATH_GPT_distribution_plans_equiv_210_l767_76735


namespace NUMINAMATH_GPT_min_value_reciprocals_l767_76751

theorem min_value_reciprocals (a b : ℝ) 
  (h1 : 2 * a + 2 * b = 2) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  (1 / a + 1 / b) ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_reciprocals_l767_76751


namespace NUMINAMATH_GPT_distinct_book_arrangements_l767_76777

def num_books := 7
def num_identical_books := 3
def num_unique_books := num_books - num_identical_books

theorem distinct_book_arrangements :
  (Nat.factorial num_books) / (Nat.factorial num_identical_books) = 840 := 
  by 
  sorry

end NUMINAMATH_GPT_distinct_book_arrangements_l767_76777


namespace NUMINAMATH_GPT_probability_divisible_by_five_l767_76796

def is_three_digit_number (n: ℕ) : Prop := 100 ≤ n ∧ n < 1000

def ends_with_five (n: ℕ) : Prop := n % 10 = 5

def divisible_by_five (n: ℕ) : Prop := n % 5 = 0

theorem probability_divisible_by_five {N : ℕ} (h1: is_three_digit_number N) (h2: ends_with_five N) : 
  ∃ p : ℚ, p = 1 ∧ ∀ n, (is_three_digit_number n ∧ ends_with_five n) → (divisible_by_five n) :=
by
  sorry

end NUMINAMATH_GPT_probability_divisible_by_five_l767_76796


namespace NUMINAMATH_GPT_journey_time_l767_76706

theorem journey_time
  (speed1 speed2 : ℝ)
  (distance total_time : ℝ)
  (h1 : speed1 = 40)
  (h2 : speed2 = 60)
  (h3 : distance = 240)
  (h4 : total_time = 5) :
  ∃ (t1 t2 : ℝ), (t1 + t2 = total_time) ∧ (speed1 * t1 + speed2 * t2 = distance) ∧ (t1 = 3) := 
by
  use (3 : ℝ), (2 : ℝ)
  simp [h1, h2, h3, h4]
  norm_num
  -- Additional steps to finish the proof would go here, but are omitted as per the requirements
  -- sorry

end NUMINAMATH_GPT_journey_time_l767_76706


namespace NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l767_76722

-- Define the first problem
theorem solve_quadratic_1 (x : ℝ) : 3 * x^2 - 4 * x = 2 * x → x = 0 ∨ x = 2 := by
  -- Proof step will go here
  sorry

-- Define the second problem
theorem solve_quadratic_2 (x : ℝ) : x * (x + 8) = 16 → x = -4 + 4 * Real.sqrt 2 ∨ x = -4 - 4 * Real.sqrt 2 := by
  -- Proof step will go here
  sorry

end NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l767_76722


namespace NUMINAMATH_GPT_multiply_fractions_l767_76766

theorem multiply_fractions :
  (2 / 3) * (5 / 7) * (8 / 9) = 80 / 189 :=
by sorry

end NUMINAMATH_GPT_multiply_fractions_l767_76766


namespace NUMINAMATH_GPT_initial_apples_l767_76731

-- Definitions based on the given conditions
def apples_given_away : ℕ := 88
def apples_left : ℕ := 39

-- Statement to prove
theorem initial_apples : apples_given_away + apples_left = 127 :=
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_initial_apples_l767_76731


namespace NUMINAMATH_GPT_max_min_S_l767_76784

theorem max_min_S (x y : ℝ) (h : (x - 1)^2 + (y + 2)^2 = 4) : 
  (∃ S_max S_min : ℝ, S_max = 4 + 2 * Real.sqrt 5 ∧ S_min = 4 - 2 * Real.sqrt 5 ∧ 
  (∀ S : ℝ, (∃ (x y : ℝ), (x - 1)^2 + (y + 2)^2 = 4 ∧ S = 2 * x + y) → S ≤ S_max ∧ S ≥ S_min)) :=
sorry

end NUMINAMATH_GPT_max_min_S_l767_76784


namespace NUMINAMATH_GPT_manufacturer_A_more_cost_effective_manufacturer_B_more_cost_effective_l767_76758

-- Definitions of costs and the problem conditions.
def cost_manufacturer_A (desks chairs : ℕ) : ℝ :=
  200 * desks + 50 * (chairs - desks)

def cost_manufacturer_B (desks chairs : ℕ) : ℝ :=
  0.9 * (200 * desks + 50 * chairs)

-- Given condition: School needs 60 desks.
def desks : ℕ := 60

-- (1) Prove manufacturer A is more cost-effective when x < 360.
theorem manufacturer_A_more_cost_effective (chairs : ℕ) (h : chairs ≥ 60) :
  chairs < 360 → cost_manufacturer_A desks chairs < cost_manufacturer_B desks chairs :=
by sorry

-- (2) Prove manufacturer B is more cost-effective when x > 360.
theorem manufacturer_B_more_cost_effective (chairs : ℕ) (h : chairs ≥ 60) :
  chairs > 360 → cost_manufacturer_A desks chairs > cost_manufacturer_B desks chairs :=
by sorry

end NUMINAMATH_GPT_manufacturer_A_more_cost_effective_manufacturer_B_more_cost_effective_l767_76758


namespace NUMINAMATH_GPT_next_correct_time_l767_76727

def clock_shows_correct_time (start_date : String) (start_time : String) (time_lost_per_hour : Int) : String :=
  if start_date = "March 21" ∧ start_time = "12:00 PM" ∧ time_lost_per_hour = 25 then
    "June 1, 12:00 PM"
  else
    "unknown"

theorem next_correct_time :
  clock_shows_correct_time "March 21" "12:00 PM" 25 = "June 1, 12:00 PM" :=
by sorry

end NUMINAMATH_GPT_next_correct_time_l767_76727


namespace NUMINAMATH_GPT_election_proof_l767_76733

noncomputable def election_problem : Prop :=
  ∃ (V : ℝ) (votesA votesB votesC : ℝ),
  (votesA = 0.35 * V) ∧
  (votesB = votesA + 1800) ∧
  (votesC = 0.5 * votesA) ∧
  (V = votesA + votesB + votesC) ∧
  (V = 14400) ∧
  ((votesA / V) * 100 = 35) ∧
  ((votesB / V) * 100 = 47.5) ∧
  ((votesC / V) * 100 = 17.5)

theorem election_proof : election_problem := sorry

end NUMINAMATH_GPT_election_proof_l767_76733


namespace NUMINAMATH_GPT_total_days_2000_to_2003_correct_l767_76781

-- Define the days in each type of year
def days_in_leap_year : ℕ := 366
def days_in_common_year : ℕ := 365

-- Define each year and its corresponding number of days
def year_2000 := days_in_leap_year
def year_2001 := days_in_common_year
def year_2002 := days_in_common_year
def year_2003 := days_in_common_year

-- Calculate the total number of days from 2000 to 2003
def total_days_2000_to_2003 : ℕ := year_2000 + year_2001 + year_2002 + year_2003

theorem total_days_2000_to_2003_correct : total_days_2000_to_2003 = 1461 := 
by
  unfold total_days_2000_to_2003 year_2000 year_2001 year_2002 year_2003 
        days_in_leap_year days_in_common_year 
  exact rfl

end NUMINAMATH_GPT_total_days_2000_to_2003_correct_l767_76781


namespace NUMINAMATH_GPT_determinant_example_l767_76739

noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)
noncomputable def sin_deg (θ : ℝ) : ℝ := Real.sin (θ * Real.pi / 180)

-- Define the determinant of a 2x2 matrix in terms of its entries
def determinant_2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Proposed theorem statement in Lean 4
theorem determinant_example : 
  determinant_2x2 (cos_deg 45) (sin_deg 75) (sin_deg 135) (cos_deg 105) = - (Real.sqrt 3 / 2) := 
by sorry

end NUMINAMATH_GPT_determinant_example_l767_76739


namespace NUMINAMATH_GPT_net_change_in_salary_l767_76717

variable (S : ℝ)

theorem net_change_in_salary : 
  let increased_salary := S + (0.1 * S)
  let final_salary := increased_salary - (0.1 * increased_salary)
  final_salary - S = -0.01 * S :=
by
  sorry

end NUMINAMATH_GPT_net_change_in_salary_l767_76717


namespace NUMINAMATH_GPT_min_value_expression_l767_76743

/-- Prove that for integers a, b, c satisfying 1 ≤ a ≤ b ≤ c ≤ 5, the minimum value of the expression 
  (a - 2)^2 + ((b + 1) / a - 1)^2 + ((c + 1) / b - 1)^2 + (5 / c - 1)^2 is 1.2595. -/
theorem min_value_expression (a b c : ℤ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  ∃ (min_val : ℝ), min_val = ((a - 2)^2 + ((b + 1) / a - 1)^2 + ((c + 1) / b - 1)^2 + (5 / c - 1)^2) ∧ min_val = 1.2595 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l767_76743


namespace NUMINAMATH_GPT_binary_to_base5_1101_l767_76752

-- Definition of the binary to decimal conversion for the given number
def binary_to_decimal (b: Nat): Nat :=
  match b with
  | 0    => 0
  | 1101 => 1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 1 * 2^3
  | _    => 0  -- This is a specific case for the given problem

-- Definition of the decimal to base-5 conversion method
def decimal_to_base5 (d: Nat): Nat :=
  match d with
  | 0    => 0
  | 13   =>
    let rem1 := 13 % 5
    let div1 := 13 / 5
    let rem2 := div1 % 5
    let div2 := div1 / 5
    rem2 * 10 + rem1  -- Assemble the base-5 number from remainders
  | _    => 0  -- This is a specific case for the given problem

-- Proof statement: conversion of 1101 in binary to base-5 yields 23
theorem binary_to_base5_1101 : decimal_to_base5 (binary_to_decimal 1101) = 23 := by
  sorry

end NUMINAMATH_GPT_binary_to_base5_1101_l767_76752


namespace NUMINAMATH_GPT_find_value_of_expression_l767_76729

variable (a b c : ℝ)

def parabola_symmetry (a b c : ℝ) :=
  (36 * a + 6 * b + c = 2) ∧ 
  (25 * a + 5 * b + c = 6) ∧ 
  (49 * a + 7 * b + c = -4)

theorem find_value_of_expression :
  (∃ a b c : ℝ, parabola_symmetry a b c) →
  3 * a + 3 * c + b = -8 :=  sorry

end NUMINAMATH_GPT_find_value_of_expression_l767_76729


namespace NUMINAMATH_GPT_common_diff_necessary_sufficient_l767_76791

section ArithmeticSequence

variable {α : Type*} [OrderedAddCommGroup α] {a : ℕ → α} {d : α}

-- Define an arithmetic sequence with common difference d
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Prove that d > 0 is the necessary and sufficient condition for a_2 > a_1
theorem common_diff_necessary_sufficient (a : ℕ → α) (d : α) :
    (is_arithmetic_sequence a d) → (d > 0 ↔ a 2 > a 1) :=
by
  sorry

end ArithmeticSequence

end NUMINAMATH_GPT_common_diff_necessary_sufficient_l767_76791


namespace NUMINAMATH_GPT_Clea_ride_time_l767_76755

theorem Clea_ride_time
  (c s d t : ℝ)
  (h1 : d = 80 * c)
  (h2 : d = 30 * (c + s))
  (h3 : s = 5 / 3 * c)
  (h4 : t = d / s) :
  t = 48 := by sorry

end NUMINAMATH_GPT_Clea_ride_time_l767_76755


namespace NUMINAMATH_GPT_number_of_elements_l767_76725

theorem number_of_elements (n : ℕ) (S : ℕ) (sum_first_six : ℕ) (sum_last_six : ℕ) (sixth_number : ℕ)
    (h1 : S = 22 * n) 
    (h2 : sum_first_six = 6 * 19) 
    (h3 : sum_last_six = 6 * 27) 
    (h4 : sixth_number = 34) 
    (h5 : S = sum_first_six + sum_last_six - sixth_number) : 
    n = 11 := 
by
  sorry

end NUMINAMATH_GPT_number_of_elements_l767_76725


namespace NUMINAMATH_GPT_angle_C_exceeds_120_degrees_l767_76793

theorem angle_C_exceeds_120_degrees 
  (a b : ℝ) (h_a : a = Real.sqrt 3) (h_b : b = Real.sqrt 3) (c : ℝ) (h_c : c > 3) :
  ∀ (C : ℝ), C = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) 
             → C > 120 :=
by
  sorry

end NUMINAMATH_GPT_angle_C_exceeds_120_degrees_l767_76793


namespace NUMINAMATH_GPT_probability_even_toys_l767_76748

theorem probability_even_toys:
  let total_toys := 21
  let even_toys := 10
  let probability_first_even := (even_toys : ℚ) / total_toys
  let probability_second_even := (even_toys - 1 : ℚ) / (total_toys - 1)
  let probability_both_even := probability_first_even * probability_second_even
  probability_both_even = 3 / 14 :=
by
  sorry

end NUMINAMATH_GPT_probability_even_toys_l767_76748


namespace NUMINAMATH_GPT_first_player_always_wins_l767_76756

theorem first_player_always_wins :
  ∃ A B : ℤ, A ≠ 0 ∧ B ≠ 0 ∧
  (A = 1998 ∧ B = -2 * 1998) ∧
  (∀ a b c : ℤ, (a = A ∨ a = B ∨ a = 1998) ∧ 
                (b = A ∨ b = B ∨ b = 1998) ∧ 
                (c = A ∨ c = B ∨ c = 1998) ∧ 
                a ≠ b ∧ b ≠ c ∧ a ≠ c → 
                ∃ x1 x2 : ℚ, x1 ≠ x2 ∧ 
                (a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0)) :=
by
  sorry

end NUMINAMATH_GPT_first_player_always_wins_l767_76756


namespace NUMINAMATH_GPT_sum_of_all_possible_values_of_z_l767_76744

noncomputable def sum_of_z_values (w x y z : ℚ) : ℚ :=
if h : w < x ∧ x < y ∧ y < z ∧ 
       (w + x = 1 ∧ w + y = 2 ∧ w + z = 3 ∧ x + y = 4 ∨ 
        w + x = 1 ∧ w + y = 2 ∧ w + z = 4 ∧ x + y = 3) ∧ 
       ((w + x) ≠ (w + y) ∧ (w + x) ≠ (w + z) ∧ (w + x) ≠ (x + y) ∧ (w + x) ≠ (x + z) ∧ (w + x) ≠ (y + z)) ∧ 
       ((w + y) ≠ (w + z) ∧ (w + y) ≠ (x + y) ∧ (w + y) ≠ (x + z) ∧ (w + y) ≠ (y + z)) ∧ 
       ((w + z) ≠ (x + y) ∧ (w + z) ≠ (x + z) ∧ (w + z) ≠ (y + z)) ∧ 
       ((x + y) ≠ (x + z) ∧ (x + y) ≠ (y + z)) ∧ 
       ((x + z) ≠ (y + z)) then
  if w + z = 4 then
    4 + 7/2
  else 0
else
  0

theorem sum_of_all_possible_values_of_z : sum_of_z_values w x y z = 15 / 2 :=
by sorry

end NUMINAMATH_GPT_sum_of_all_possible_values_of_z_l767_76744


namespace NUMINAMATH_GPT_monotonic_intervals_slope_tangent_line_inequality_condition_l767_76785

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * (a + 2) * x^2 + 2 * a * x
noncomputable def g (a x : ℝ) : ℝ := (1/2) * (a - 5) * x^2

theorem monotonic_intervals (a : ℝ) (h : a ≥ 4) :
  (∀ x, deriv (f a) x = x^2 - (a + 2) * x + 2 * a) ∧
  ((∀ x, x < 2 → deriv (f a) x > 0) ∧ (∀ x, x > a → deriv (f a) x > 0)) ∧
  (∀ x, 2 < x ∧ x < a → deriv (f a) x < 0) :=
sorry

theorem slope_tangent_line (a : ℝ) (h : a ≥ 4) :
  (∀ x, deriv (f a) x = x^2 - (a + 2) * x + 2 * a) ∧
  (∀ x_0 y_0 k, y_0 = f a x_0 ∧ k = deriv (f a) x_0 ∧ k ≥ -(25/4) →
    4 ≤ a ∧ a ≤ 7) :=
sorry

theorem inequality_condition (a : ℝ) (h : a ≥ 4) :
  (∀ x_1 x_2, 3 ≤ x_1 ∧ x_1 < x_2 ∧ x_2 ≤ 4 →
    abs (f a x_1 - f a x_2) > abs (g a x_1 - g a x_2)) →
  (14/3 ≤ a ∧ a ≤ 6) :=
sorry

end NUMINAMATH_GPT_monotonic_intervals_slope_tangent_line_inequality_condition_l767_76785


namespace NUMINAMATH_GPT_quadratic_polynomial_solution_is_zero_l767_76736

-- Definitions based on given conditions
variables (a b c r s : ℝ)
variables (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
variables (h2 : a ≠ b ∧ a ≠ c ∧ b ≠ c)
variables (h3 : r + s = -b / a)
variables (h4 : r * s = c / a)

-- Proposition matching the equivalent proof problem
theorem quadratic_polynomial_solution_is_zero :
  ¬ ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  (∃ r s : ℝ, (r + s = -b / a) ∧ (r * s = c / a) ∧ (c = r * s ∨ b = r * s ∨ a = r * s) ∧
  (a = r ∨ a = s)) :=
sorry

end NUMINAMATH_GPT_quadratic_polynomial_solution_is_zero_l767_76736


namespace NUMINAMATH_GPT_selling_price_per_book_l767_76741

noncomputable def fixed_costs : ℝ := 35630
noncomputable def variable_cost_per_book : ℝ := 11.50
noncomputable def num_books : ℕ := 4072
noncomputable def total_production_costs : ℝ := fixed_costs + variable_cost_per_book * num_books

theorem selling_price_per_book :
  (total_production_costs / num_books : ℝ) = 20.25 := by
  sorry

end NUMINAMATH_GPT_selling_price_per_book_l767_76741


namespace NUMINAMATH_GPT_digit_difference_is_7_l767_76708

def local_value (d : Nat) (place : Nat) : Nat :=
  d * (10^place)

def face_value (d : Nat) : Nat :=
  d

def difference (d : Nat) (place : Nat) : Nat :=
  local_value d place - face_value d

def numeral : Nat := 65793

theorem digit_difference_is_7 :
  ∃ d place, 0 ≤ d ∧ d < 10 ∧ difference d place = 693 ∧ d ∈ [6, 5, 7, 9, 3] ∧ numeral = 65793 ∧
  (local_value 6 4 = 60000 ∧ local_value 5 3 = 5000 ∧ local_value 7 2 = 700 ∧ local_value 9 1 = 90 ∧ local_value 3 0 = 3 ∧
   face_value 6 = 6 ∧ face_value 5 = 5 ∧ face_value 7 = 7 ∧ face_value 9 = 9 ∧ face_value 3 = 3) ∧ 
  d = 7 :=
sorry

end NUMINAMATH_GPT_digit_difference_is_7_l767_76708


namespace NUMINAMATH_GPT_solve_abs_equation_l767_76726

theorem solve_abs_equation (y : ℤ) : (|y - 8| + 3 * y = 12) ↔ (y = 2) :=
by
  sorry  -- skip the proof steps.

end NUMINAMATH_GPT_solve_abs_equation_l767_76726


namespace NUMINAMATH_GPT_clock_ticks_six_times_l767_76776

-- Define the conditions
def time_between_ticks (ticks : Nat) : Nat :=
  ticks - 1

def interval_duration (total_time : Nat) (ticks : Nat) : Nat :=
  total_time / time_between_ticks ticks

def number_of_ticks (total_time : Nat) (interval_time : Nat) : Nat :=
  total_time / interval_time + 1

-- Given conditions
def specific_time_intervals : Nat := 30
def eight_oclock_intervals : Nat := 42

-- Proven result
theorem clock_ticks_six_times : number_of_ticks specific_time_intervals (interval_duration eight_oclock_intervals 8) = 6 := 
sorry

end NUMINAMATH_GPT_clock_ticks_six_times_l767_76776


namespace NUMINAMATH_GPT_not_net_of_cuboid_l767_76713

noncomputable def cuboid_closed_path (c : Type) (f : c → c) :=
∀ (x1 x2 : c), ∃ (y : c), f x1 = y ∧ f x2 = y

theorem not_net_of_cuboid (c : Type) [Nonempty c] [DecidableEq c] (net : c → Set c) (f : c → c) :
  cuboid_closed_path c f → ¬ (∀ x, net x = {x}) :=
by
  sorry

end NUMINAMATH_GPT_not_net_of_cuboid_l767_76713


namespace NUMINAMATH_GPT_largest_square_perimeter_l767_76795

-- Define the conditions
def rectangle_length : ℕ := 80
def rectangle_width : ℕ := 60

-- Define the theorem to prove
theorem largest_square_perimeter : 4 * rectangle_width = 240 := by
  -- The proof steps are omitted
  sorry

end NUMINAMATH_GPT_largest_square_perimeter_l767_76795


namespace NUMINAMATH_GPT_new_stamps_ratio_l767_76782

theorem new_stamps_ratio (x : ℕ) (h1 : 7 * x = P) (h2 : 4 * x = Q)
  (h3 : P - 8 = 8 + (Q + 8)) : (P - 8) / gcd (P - 8) (Q + 8) = 6 ∧ (Q + 8) / gcd (P - 8) (Q + 8) = 5 :=
by
  sorry

end NUMINAMATH_GPT_new_stamps_ratio_l767_76782


namespace NUMINAMATH_GPT_perimeter_of_regular_nonagon_l767_76774

def regular_nonagon_side_length := 3
def number_of_sides := 9

theorem perimeter_of_regular_nonagon (h1 : number_of_sides = 9) (h2 : regular_nonagon_side_length = 3) :
  9 * 3 = 27 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_regular_nonagon_l767_76774


namespace NUMINAMATH_GPT_numbers_in_circle_are_zero_l767_76724

theorem numbers_in_circle_are_zero (a : Fin 55 → ℤ) 
  (h : ∀ i, a i = a ((i + 54) % 55) + a ((i + 1) % 55)) : 
  ∀ i, a i = 0 := 
by
  sorry

end NUMINAMATH_GPT_numbers_in_circle_are_zero_l767_76724


namespace NUMINAMATH_GPT_hypotenuse_length_l767_76747

theorem hypotenuse_length {a b c : ℕ} (ha : a = 8) (hb : b = 15) (hc : c = (8^2 + 15^2).sqrt) : c = 17 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l767_76747


namespace NUMINAMATH_GPT_arithmetic_seq_a4_l767_76759

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions and the goal to prove
theorem arithmetic_seq_a4 (h₁ : is_arithmetic_sequence a d) (h₂ : a 2 + a 6 = 10) : 
  a 4 = 5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a4_l767_76759


namespace NUMINAMATH_GPT_cube_surface_area_equals_353_l767_76754

noncomputable def volume_of_prism : ℝ := 5 * 3 * 30
noncomputable def edge_length_of_cube (volume : ℝ) : ℝ := (volume)^(1/3)
noncomputable def surface_area_of_cube (edge_length : ℝ) : ℝ := 6 * edge_length^2

theorem cube_surface_area_equals_353 :
  surface_area_of_cube (edge_length_of_cube volume_of_prism) = 353 := by
sorry

end NUMINAMATH_GPT_cube_surface_area_equals_353_l767_76754


namespace NUMINAMATH_GPT_inequality_solution_set_empty_l767_76765

theorem inequality_solution_set_empty (a : ℝ) :
  (∀ x : ℝ, ¬ (|x - 2| + |x + 3| < a)) → a ≤ 5 :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_empty_l767_76765


namespace NUMINAMATH_GPT_sales_first_month_l767_76710

theorem sales_first_month (S1 S2 S3 S4 S5 S6 : ℝ) 
  (h2 : S2 = 7000) (h3 : S3 = 6800) (h4 : S4 = 7200) (h5 : S5 = 6500) (h6 : S6 = 5100)
  (avg : (S1 + S2 + S3 + S4 + S5 + S6) / 6 = 6500) : S1 = 6400 := by
  sorry

end NUMINAMATH_GPT_sales_first_month_l767_76710


namespace NUMINAMATH_GPT_x_intercept_of_line_l767_76728

open Real

theorem x_intercept_of_line : 
  ∃ x : ℝ, 
  (∃ m : ℝ, m = (3 - -5) / (10 - -6) ∧ (∀ y : ℝ, y = m * (x - 10) + 3)) ∧ 
  (∀ y : ℝ, y = 0 → x = 4) :=
sorry

end NUMINAMATH_GPT_x_intercept_of_line_l767_76728


namespace NUMINAMATH_GPT_brian_cards_after_waine_takes_l767_76763

-- Define the conditions
def brian_initial_cards : ℕ := 76
def wayne_takes_away : ℕ := 59

-- Define the expected result
def brian_remaining_cards : ℕ := 17

-- The statement of the proof problem
theorem brian_cards_after_waine_takes : brian_initial_cards - wayne_takes_away = brian_remaining_cards := 
by 
-- the proof would be provided here 
sorry

end NUMINAMATH_GPT_brian_cards_after_waine_takes_l767_76763


namespace NUMINAMATH_GPT_janeth_balloons_l767_76773

/-- Janeth's total remaining balloons after accounting for burst ones. -/
def total_remaining_balloons (round_bags : Nat) (round_per_bag : Nat) (burst_round : Nat)
    (long_bags : Nat) (long_per_bag : Nat) (burst_long : Nat)
    (heart_bags : Nat) (heart_per_bag : Nat) (burst_heart : Nat) : Nat :=
  let total_round := round_bags * round_per_bag - burst_round
  let total_long := long_bags * long_per_bag - burst_long
  let total_heart := heart_bags * heart_per_bag - burst_heart
  total_round + total_long + total_heart

theorem janeth_balloons :
  total_remaining_balloons 5 25 5 4 35 7 3 40 3 = 370 :=
by
  let round_bags := 5
  let round_per_bag := 25
  let burst_round := 5
  let long_bags := 4
  let long_per_bag := 35
  let burst_long := 7
  let heart_bags := 3
  let heart_per_bag := 40
  let burst_heart := 3
  show total_remaining_balloons round_bags round_per_bag burst_round long_bags long_per_bag burst_long heart_bags heart_per_bag burst_heart = 370
  sorry

end NUMINAMATH_GPT_janeth_balloons_l767_76773


namespace NUMINAMATH_GPT_tetrahedron_volume_formula_l767_76789

variables (r₀ S₀ S₁ S₂ S₃ V : ℝ)

theorem tetrahedron_volume_formula
  (h : V = (1/3) * (S₁ + S₂ + S₃ - S₀) * r₀) :
  V = (1/3) * (S₁ + S₂ + S₃ - S₀) * r₀ :=
by { sorry }

end NUMINAMATH_GPT_tetrahedron_volume_formula_l767_76789


namespace NUMINAMATH_GPT_man_alone_days_l767_76768

-- Conditions from the problem
variables (M : ℕ) (h1 : (1 / (↑M : ℝ)) + (1 / 12) = 1 / 3)  -- Combined work rate condition

-- The proof statement we need to show
theorem man_alone_days : M = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_man_alone_days_l767_76768


namespace NUMINAMATH_GPT_combined_percentage_basketball_l767_76783

theorem combined_percentage_basketball (N_students : ℕ) (S_students : ℕ) 
  (N_percent_basketball : ℚ) (S_percent_basketball : ℚ) :
  N_students = 1800 → S_students = 3000 →
  N_percent_basketball = 0.25 → S_percent_basketball = 0.35 →
  ((N_students * N_percent_basketball) + (S_students * S_percent_basketball)) / (N_students + S_students) * 100 = 31 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  norm_num
  sorry

end NUMINAMATH_GPT_combined_percentage_basketball_l767_76783


namespace NUMINAMATH_GPT_power_function_expression_l767_76700

theorem power_function_expression (α : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x ^ α) (h_point : f 2 = 4) :
  α = 2 ∧ (∀ x, f x = x ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_power_function_expression_l767_76700


namespace NUMINAMATH_GPT_value_of_M_l767_76760

theorem value_of_M (M : ℝ) :
  (20 / 100) * M = (60 / 100) * 1500 → M = 4500 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_M_l767_76760


namespace NUMINAMATH_GPT_marcy_total_spears_l767_76792

-- Define the conditions
def can_make_spears_from_sapling (spears_per_sapling : ℕ) (saplings : ℕ) : ℕ :=
  spears_per_sapling * saplings

def can_make_spears_from_log (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  spears_per_log * logs

-- Number of spears Marcy can make from 6 saplings and 1 log
def total_spears (spears_per_sapling : ℕ) (saplings : ℕ) (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  can_make_spears_from_sapling spears_per_sapling saplings + can_make_spears_from_log spears_per_log logs

-- Given conditions
theorem marcy_total_spears (saplings : ℕ) (logs : ℕ) : 
  total_spears 3 6 9 1 = 27 :=
by
  sorry

end NUMINAMATH_GPT_marcy_total_spears_l767_76792


namespace NUMINAMATH_GPT_find_w_value_l767_76753

theorem find_w_value : 
  (2^5 * 9^2) / (8^2 * 243) = 0.16666666666666666 := 
by
  sorry

end NUMINAMATH_GPT_find_w_value_l767_76753


namespace NUMINAMATH_GPT_triangles_with_perimeter_20_l767_76757

theorem triangles_with_perimeter_20 (sides : Finset (Finset ℕ)) : 
  (∀ {a b c : ℕ}, (a + b + c = 20) → (a > 0) → (b > 0) → (c > 0) 
  → (a + b > c) → (a + c > b) → (b + c > a) → ({a, b, c} ∈ sides)) 
  → sides.card = 8 := 
by
  sorry

end NUMINAMATH_GPT_triangles_with_perimeter_20_l767_76757


namespace NUMINAMATH_GPT_non_union_employees_women_percent_l767_76701

-- Define the conditions
variables (total_employees men_percent women_percent unionized_percent unionized_men_percent : ℕ)
variables (total_men total_women total_unionized total_non_unionized unionized_men non_unionized_men non_unionized_women : ℕ)

axiom condition1 : men_percent = 52
axiom condition2 : unionized_percent = 60
axiom condition3 : unionized_men_percent = 70

axiom calc1 : total_employees = 100
axiom calc2 : total_men = total_employees * men_percent / 100
axiom calc3 : total_women = total_employees - total_men
axiom calc4 : total_unionized = total_employees * unionized_percent / 100
axiom calc5 : unionized_men = total_unionized * unionized_men_percent / 100
axiom calc6 : non_unionized_men = total_men - unionized_men
axiom calc7 : total_non_unionized = total_employees - total_unionized
axiom calc8 : non_unionized_women = total_non_unionized - non_unionized_men

-- Define the proof statement
theorem non_union_employees_women_percent : 
  (non_unionized_women / total_non_unionized) * 100 = 75 :=
by 
  sorry

end NUMINAMATH_GPT_non_union_employees_women_percent_l767_76701


namespace NUMINAMATH_GPT_oil_bill_for_January_l767_76720

variable {F J : ℕ}

theorem oil_bill_for_January (h1 : 2 * F = 3 * J) (h2 : 3 * (F + 20) = 5 * J) : J = 120 := by
  sorry

end NUMINAMATH_GPT_oil_bill_for_January_l767_76720


namespace NUMINAMATH_GPT_total_games_played_is_53_l767_76770

theorem total_games_played_is_53 :
  ∃ (ken_wins dave_wins jerry_wins larry_wins total_ties total_games_played : ℕ),
  jerry_wins = 7 ∧
  dave_wins = jerry_wins + 3 ∧
  ken_wins = dave_wins + 5 ∧
  larry_wins = 2 * jerry_wins ∧
  5 ≤ ken_wins ∧ 5 ≤ dave_wins ∧ 5 ≤ jerry_wins ∧ 5 ≤ larry_wins ∧
  total_ties = jerry_wins ∧
  total_games_played = ken_wins + dave_wins + jerry_wins + larry_wins + total_ties ∧
  total_games_played = 53 :=
by
  sorry

end NUMINAMATH_GPT_total_games_played_is_53_l767_76770


namespace NUMINAMATH_GPT_pencil_eraser_cost_l767_76769

theorem pencil_eraser_cost (p e : ℕ) (hp : p > e) (he : e > 0)
  (h : 20 * p + 4 * e = 160) : p + e = 12 :=
sorry

end NUMINAMATH_GPT_pencil_eraser_cost_l767_76769


namespace NUMINAMATH_GPT_smallest_number_divide_perfect_cube_l767_76705

theorem smallest_number_divide_perfect_cube (n : ℕ):
  n = 450 → (∃ m : ℕ, n * m = k ∧ ∃ k : ℕ, k ^ 3 = n * m) ∧ (∀ m₂ : ℕ, (n * m₂ = l ∧ ∃ l : ℕ, l ^ 3 = n * m₂) → m ≤ m₂) → m = 60 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_divide_perfect_cube_l767_76705


namespace NUMINAMATH_GPT_find_k_perpendicular_l767_76790

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (2, -3)

-- Define a function for the vector k * a - 2 * b
def vec_expression (k : ℝ) : ℝ × ℝ :=
  (k * vec_a.1 - 2 * vec_b.1, k * vec_a.2 - 2 * vec_b.2)

-- Prove that if the dot product of vec_expression k and vec_a is zero, then k = -1
theorem find_k_perpendicular (k : ℝ) :
  ((vec_expression k).1 * vec_a.1 + (vec_expression k).2 * vec_a.2 = 0) → k = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_perpendicular_l767_76790


namespace NUMINAMATH_GPT_father_present_age_l767_76702

theorem father_present_age (S F : ℕ) 
  (h1 : F = 3 * S + 3) 
  (h2 : F + 3 = 2 * (S + 3) + 10) : 
  F = 33 :=
by
  sorry

end NUMINAMATH_GPT_father_present_age_l767_76702


namespace NUMINAMATH_GPT_percentage_of_masters_is_76_l767_76750

variable (x y : ℕ)  -- Let x be the number of junior players, y be the number of master players
variable (junior_avg master_avg team_avg : ℚ)

-- The conditions given in the problem
def juniors_avg_points : Prop := junior_avg = 22
def masters_avg_points : Prop := master_avg = 47
def team_avg_points (x y : ℕ) (junior_avg master_avg team_avg : ℚ) : Prop :=
  (22 * x + 47 * y) / (x + y) = 41

def proportion_of_masters (x y : ℕ) : ℚ := (y : ℚ) / (x + y)

-- The theorem to be proved
theorem percentage_of_masters_is_76 (x y : ℕ) (junior_avg master_avg team_avg : ℚ) :
  juniors_avg_points junior_avg →
  masters_avg_points master_avg →
  team_avg_points x y junior_avg master_avg team_avg →
  proportion_of_masters x y = 19 / 25 := 
sorry

end NUMINAMATH_GPT_percentage_of_masters_is_76_l767_76750


namespace NUMINAMATH_GPT_min_value_l767_76794

noncomputable def min_value_expr (a b c d : ℝ) : ℝ :=
  (a + b) / c + (b + c) / a + (c + d) / b

theorem min_value 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  min_value_expr a b c d ≥ 6 
  := sorry

end NUMINAMATH_GPT_min_value_l767_76794
