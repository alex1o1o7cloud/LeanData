import Mathlib

namespace NUMINAMATH_GPT_cost_of_rice_l736_73651

theorem cost_of_rice (x : ℝ) 
  (h : 5 * x + 3 * 5 = 25) : x = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_of_rice_l736_73651


namespace NUMINAMATH_GPT_derivative_at_two_l736_73609

def f (x : ℝ) : ℝ := x^3 + 4 * x - 5

noncomputable def derivative_f (x : ℝ) : ℝ := 3 * x^2 + 4

theorem derivative_at_two : derivative_f 2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_two_l736_73609


namespace NUMINAMATH_GPT_solve_system_nat_l736_73648

open Nat

theorem solve_system_nat (x y z t : ℕ) :
  (x + y = z * t ∧ z + t = x * y) ↔ (x, y, z, t) = (1, 5, 2, 3) ∨ (x, y, z, t) = (2, 2, 2, 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_nat_l736_73648


namespace NUMINAMATH_GPT_soccer_league_teams_l736_73630

theorem soccer_league_teams (n : ℕ) (h : n * (n - 1) / 2 = 55) : n = 11 := 
sorry

end NUMINAMATH_GPT_soccer_league_teams_l736_73630


namespace NUMINAMATH_GPT_part1_part2_part3_l736_73623

def A (x y : ℝ) := 2*x^2 + 3*x*y + 2*y
def B (x y : ℝ) := x^2 - x*y + x

theorem part1 (x y : ℝ) : A x y - 2 * B x y = 5*x*y - 2*x + 2*y := by
  sorry

theorem part2 (x y : ℝ) (h1 : x^2 = 9) (h2 : |y| = 2) :
  A x y - 2 * B x y = 28 ∨ A x y - 2 * B x y = -40 ∨ A x y - 2 * B x y = -20 ∨ A x y - 2 * B x y = 32 := by
  sorry

theorem part3 (y : ℝ) : (∀ x : ℝ, A x y - 2 * B x y = A 0 y - 2 * B 0 y) → y = 2/5 := by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l736_73623


namespace NUMINAMATH_GPT_trains_crossing_time_correct_l736_73619

def convert_kmph_to_mps (speed_kmph : ℕ) : ℚ := (speed_kmph * 5) / 18

def time_to_cross_each_other 
  (length_train1 length_train2 speed_kmph_train1 speed_kmph_train2 : ℕ) : ℚ :=
  let speed_train1 := convert_kmph_to_mps speed_kmph_train1
  let speed_train2 := convert_kmph_to_mps speed_kmph_train2
  let relative_speed := speed_train2 - speed_train1
  let total_distance := length_train1 + length_train2
  (total_distance : ℚ) / relative_speed

theorem trains_crossing_time_correct :
  time_to_cross_each_other 200 150 40 46 = 210 := by
  sorry

end NUMINAMATH_GPT_trains_crossing_time_correct_l736_73619


namespace NUMINAMATH_GPT_gauss_company_percent_five_years_or_more_l736_73665

def num_employees_less_1_year (x : ℕ) : ℕ := 5 * x
def num_employees_1_to_2_years (x : ℕ) : ℕ := 5 * x
def num_employees_2_to_3_years (x : ℕ) : ℕ := 8 * x
def num_employees_3_to_4_years (x : ℕ) : ℕ := 3 * x
def num_employees_4_to_5_years (x : ℕ) : ℕ := 2 * x
def num_employees_5_to_6_years (x : ℕ) : ℕ := 2 * x
def num_employees_6_to_7_years (x : ℕ) : ℕ := 2 * x
def num_employees_7_to_8_years (x : ℕ) : ℕ := x
def num_employees_8_to_9_years (x : ℕ) : ℕ := x
def num_employees_9_to_10_years (x : ℕ) : ℕ := x

def total_employees (x : ℕ) : ℕ :=
  num_employees_less_1_year x +
  num_employees_1_to_2_years x +
  num_employees_2_to_3_years x +
  num_employees_3_to_4_years x +
  num_employees_4_to_5_years x +
  num_employees_5_to_6_years x +
  num_employees_6_to_7_years x +
  num_employees_7_to_8_years x +
  num_employees_8_to_9_years x +
  num_employees_9_to_10_years x

def employees_with_5_years_or_more (x : ℕ) : ℕ :=
  num_employees_5_to_6_years x +
  num_employees_6_to_7_years x +
  num_employees_7_to_8_years x +
  num_employees_8_to_9_years x +
  num_employees_9_to_10_years x

theorem gauss_company_percent_five_years_or_more (x : ℕ) :
  (employees_with_5_years_or_more x : ℝ) / (total_employees x : ℝ) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_gauss_company_percent_five_years_or_more_l736_73665


namespace NUMINAMATH_GPT_calculate_expression_l736_73635

theorem calculate_expression :
  (10^4 - 9^4 + 8^4 - 7^4 + 6^4 - 5^4 + 4^4 - 3^4 + 2^4 - 1^4) +
  (10^2 + 9^2 + 5 * 8^2 + 5 * 7^2 + 9 * 6^2 + 9 * 5^2 + 13 * 4^2 + 13 * 3^2) = 7615 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l736_73635


namespace NUMINAMATH_GPT_area_PVZ_is_correct_l736_73663

noncomputable def area_triangle_PVZ : ℝ :=
  let PQ : ℝ := 8
  let QR : ℝ := 4
  let RV : ℝ := 2
  let WS : ℝ := 3
  let VW : ℝ := PQ - (RV + WS)  -- VW is calculated as 3
  let base_PV : ℝ := PQ
  let height_PVZ : ℝ := QR
  1 / 2 * base_PV * height_PVZ

theorem area_PVZ_is_correct : area_triangle_PVZ = 16 :=
  sorry

end NUMINAMATH_GPT_area_PVZ_is_correct_l736_73663


namespace NUMINAMATH_GPT_percentage_increase_in_ear_piercing_l736_73682

def cost_of_nose_piercing : ℕ := 20
def noses_pierced : ℕ := 6
def ears_pierced : ℕ := 9
def total_amount_made : ℕ := 390

def cost_of_ear_piercing : ℕ := (total_amount_made - (noses_pierced * cost_of_nose_piercing)) / ears_pierced

def percentage_increase (original new : ℕ) : ℚ := ((new - original : ℚ) / original) * 100

theorem percentage_increase_in_ear_piercing : 
  percentage_increase cost_of_nose_piercing cost_of_ear_piercing = 50 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_in_ear_piercing_l736_73682


namespace NUMINAMATH_GPT_problem_a_b_c_relationship_l736_73668

theorem problem_a_b_c_relationship (u v a b c : ℝ)
  (h1 : u - v = a)
  (h2 : u^2 - v^2 = b)
  (h3 : u^3 - v^3 = c) :
  3 * b^2 + a^4 = 4 * a * c := by
  sorry

end NUMINAMATH_GPT_problem_a_b_c_relationship_l736_73668


namespace NUMINAMATH_GPT_opposite_of_neg_3_is_3_l736_73620

theorem opposite_of_neg_3_is_3 : ∀ (x : ℤ), x = -3 → -x = 3 :=
by
  intro x
  intro h
  rw [h]
  simp

end NUMINAMATH_GPT_opposite_of_neg_3_is_3_l736_73620


namespace NUMINAMATH_GPT_cells_that_remain_open_l736_73607

/-- A cell q remains open after iterative toggling if and only if it is a perfect square. -/
theorem cells_that_remain_open (n : ℕ) (h : n > 0) : 
  (∃ k : ℕ, k ^ 2 = n) ↔ 
  (∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → (∃ m : ℕ, i = m ^ 2)) := 
sorry

end NUMINAMATH_GPT_cells_that_remain_open_l736_73607


namespace NUMINAMATH_GPT_least_possible_c_l736_73636

theorem least_possible_c 
  (a b c : ℕ) 
  (h_avg : (a + b + c) / 3 = 20)
  (h_median : b = a + 13)
  (h_ord : a ≤ b ∧ b ≤ c)
  : c = 45 :=
sorry

end NUMINAMATH_GPT_least_possible_c_l736_73636


namespace NUMINAMATH_GPT_exists_between_elements_l736_73692

noncomputable def M : Set ℝ :=
  { x | ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ x = (m + n) / Real.sqrt (m^2 + n^2) }

theorem exists_between_elements (x y : ℝ) (hx : x ∈ M) (hy : y ∈ M) (hxy : x < y) :
  ∃ z ∈ M, x < z ∧ z < y :=
by
  sorry

end NUMINAMATH_GPT_exists_between_elements_l736_73692


namespace NUMINAMATH_GPT_john_took_11_more_chickens_than_ray_l736_73694

noncomputable def chickens_taken_by_john (mary_chickens : ℕ) : ℕ := mary_chickens + 5
noncomputable def chickens_taken_by_ray (mary_chickens : ℕ) : ℕ := mary_chickens - 6
def ray_chickens : ℕ := 10

-- The theorem to prove:
theorem john_took_11_more_chickens_than_ray :
  ∃ (mary_chickens : ℕ), chickens_taken_by_john mary_chickens - ray_chickens = 11 :=
by
  -- Initial assumptions and derivation steps should be provided here.
  sorry

end NUMINAMATH_GPT_john_took_11_more_chickens_than_ray_l736_73694


namespace NUMINAMATH_GPT_percentage_change_area_l736_73622

theorem percentage_change_area (L B : ℝ) :
  let A_original := L * B
  let A_new := (L / 2) * (3 * B)
  (A_new - A_original) / A_original * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_change_area_l736_73622


namespace NUMINAMATH_GPT_probability_A_miss_at_least_once_probability_A_2_hits_B_3_hits_l736_73659

variable {p q : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1)

theorem probability_A_miss_at_least_once :
  1 - p^4 = (1 - p^4) := by
sorry

theorem probability_A_2_hits_B_3_hits :
  24 * p^2 * q^3 * (1 - p)^2 * (1 - q) = 24 * p^2 * q^3 * (1 - p)^2 * (1 - q) := by
sorry

end NUMINAMATH_GPT_probability_A_miss_at_least_once_probability_A_2_hits_B_3_hits_l736_73659


namespace NUMINAMATH_GPT_tutors_work_together_again_in_360_days_l736_73656

theorem tutors_work_together_again_in_360_days :
  Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 :=
by
  sorry

end NUMINAMATH_GPT_tutors_work_together_again_in_360_days_l736_73656


namespace NUMINAMATH_GPT_find_other_percentage_l736_73671

noncomputable def percentage_other_investment
  (total_investment : ℝ)
  (investment_10_percent : ℝ)
  (total_interest : ℝ)
  (interest_rate_10_percent : ℝ)
  (other_investment_interest : ℝ) : ℝ :=
  let interest_10_percent := investment_10_percent * interest_rate_10_percent
  let interest_other_investment := total_interest - interest_10_percent
  let amount_other_percentage := total_investment - investment_10_percent
  interest_other_investment / amount_other_percentage

theorem find_other_percentage :
  ∀ (total_investment : ℝ)
    (investment_10_percent : ℝ)
    (total_interest : ℝ)
    (interest_rate_10_percent : ℝ),
    total_investment = 31000 ∧
    investment_10_percent = 12000 ∧
    total_interest = 1390 ∧
    interest_rate_10_percent = 0.1 →
    percentage_other_investment total_investment investment_10_percent total_interest interest_rate_10_percent 190 = 0.01 :=
by
  intros total_investment investment_10_percent total_interest interest_rate_10_percent h
  sorry

end NUMINAMATH_GPT_find_other_percentage_l736_73671


namespace NUMINAMATH_GPT_possible_six_digit_numbers_divisible_by_3_l736_73679

theorem possible_six_digit_numbers_divisible_by_3 (missing_digit_condition : ∀ k : Nat, (8 + 5 + 5 + 2 + 2 + k) % 3 = 0) : 
  ∃ count : Nat, count = 13 := by
  sorry

end NUMINAMATH_GPT_possible_six_digit_numbers_divisible_by_3_l736_73679


namespace NUMINAMATH_GPT_cos_theta_neg_three_fifths_l736_73637

theorem cos_theta_neg_three_fifths 
  (θ : ℝ)
  (h1 : Real.sin θ = -4 / 5)
  (h2 : Real.tan θ > 0) : 
  Real.cos θ = -3 / 5 := 
sorry

end NUMINAMATH_GPT_cos_theta_neg_three_fifths_l736_73637


namespace NUMINAMATH_GPT_remainder_div_29_l736_73689

theorem remainder_div_29 (k : ℤ) (N : ℤ) (h : N = 899 * k + 63) : N % 29 = 10 :=
  sorry

end NUMINAMATH_GPT_remainder_div_29_l736_73689


namespace NUMINAMATH_GPT_max_three_cards_l736_73600

theorem max_three_cards (n m p : ℕ) (h : n + m + p = 8) (sum : 3 * n + 4 * m + 5 * p = 33) 
  (n_le_10 : n ≤ 10) (m_le_10 : m ≤ 10) (p_le_10 : p ≤ 10) : n ≤ 3 := 
sorry

end NUMINAMATH_GPT_max_three_cards_l736_73600


namespace NUMINAMATH_GPT_remaining_students_l736_73677

def groups := 3
def students_per_group := 8
def students_left_early := 2

theorem remaining_students : (groups * students_per_group) - students_left_early = 22 := by
  --Proof skipped
  sorry

end NUMINAMATH_GPT_remaining_students_l736_73677


namespace NUMINAMATH_GPT_total_students_correct_l736_73646

-- Definitions based on the conditions
def students_germain : Nat := 13
def students_newton : Nat := 10
def students_young : Nat := 12
def overlap_germain_newton : Nat := 2
def overlap_germain_young : Nat := 1

-- Total distinct students (using inclusion-exclusion principle)
def total_distinct_students : Nat :=
  students_germain + students_newton + students_young - overlap_germain_newton - overlap_germain_young

-- The theorem we want to prove
theorem total_students_correct : total_distinct_students = 32 :=
  by
    -- We state the computation directly; proof is omitted
    sorry

end NUMINAMATH_GPT_total_students_correct_l736_73646


namespace NUMINAMATH_GPT_rods_in_mile_l736_73673

theorem rods_in_mile (mile_to_furlongs : 1 = 12) (furlong_to_rods : 1 = 50) : 1 * 12 * 50 = 600 :=
by
  sorry

end NUMINAMATH_GPT_rods_in_mile_l736_73673


namespace NUMINAMATH_GPT_flag_count_l736_73686

def colors := 3

def stripes := 3

noncomputable def number_of_flags (colors stripes : ℕ) : ℕ :=
  colors ^ stripes

theorem flag_count : number_of_flags colors stripes = 27 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end NUMINAMATH_GPT_flag_count_l736_73686


namespace NUMINAMATH_GPT_probability_of_even_distinct_digits_l736_73625

noncomputable def probability_even_distinct_digits : ℚ :=
  let total_numbers := 9000
  let favorable_numbers := 2744
  favorable_numbers / total_numbers

theorem probability_of_even_distinct_digits : 
  probability_even_distinct_digits = 343 / 1125 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_even_distinct_digits_l736_73625


namespace NUMINAMATH_GPT_find_w_when_x_is_six_l736_73681

variable {x w : ℝ}
variable (h1 : x = 3)
variable (h2 : w = 16)
variable (h3 : ∀ (x w : ℝ), x^4 * w^(1 / 4) = 162)

theorem find_w_when_x_is_six : x = 6 → w = 1 / 4096 :=
by
  intro hx
  sorry

end NUMINAMATH_GPT_find_w_when_x_is_six_l736_73681


namespace NUMINAMATH_GPT_compound_interest_l736_73696

variables {a r : ℝ}

theorem compound_interest (a r : ℝ) :
  (a * (1 + r)^10) = a * (1 + r)^(2020 - 2010) :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_l736_73696


namespace NUMINAMATH_GPT_find_x_l736_73655

theorem find_x (x : ℝ) : 9 - (x / (1 / 3)) + 3 = 3 → x = 3 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l736_73655


namespace NUMINAMATH_GPT_factorize_expression_l736_73675

theorem factorize_expression (m : ℝ) : m^3 - 4 * m^2 + 4 * m = m * (m - 2)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l736_73675


namespace NUMINAMATH_GPT_triangles_satisfying_equation_l736_73624

theorem triangles_satisfying_equation (a b c : ℝ) (h₂ : a + b > c) (h₃ : a + c > b) (h₄ : b + c > a) :
  (c ^ 2 - a ^ 2) / b + (b ^ 2 - c ^ 2) / a = b - a →
  (a = b ∨ c ^ 2 = a ^ 2 + b ^ 2) := 
sorry

end NUMINAMATH_GPT_triangles_satisfying_equation_l736_73624


namespace NUMINAMATH_GPT_perpendicular_lines_condition_l736_73641

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * (m + 1) * x + (m - 3) * y + 7 - 5 * m = 0) ↔ (∀ x y : ℝ, (m - 3) * x + 2 * y - 5 = 0) →
  (m = 3 ∨ m = -2) :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_l736_73641


namespace NUMINAMATH_GPT_total_people_expression_l736_73618

variable {X : ℕ}

def men (X : ℕ) := 24 * X
def women (X : ℕ) := 12 * X
def teenagers (X : ℕ) := 4 * X
def children (X : ℕ) := X

def total_people (X : ℕ) := men X + women X + teenagers X + children X

theorem total_people_expression (X : ℕ) : total_people X = 41 * X :=
by 
  unfold total_people
  unfold men women teenagers children
  sorry

end NUMINAMATH_GPT_total_people_expression_l736_73618


namespace NUMINAMATH_GPT_fraction_squares_sum_l736_73661

theorem fraction_squares_sum (x a y b z c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h1 : x / a + y / b + z / c = 3) (h2 : a / x + b / y + c / z = -3) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 15 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_squares_sum_l736_73661


namespace NUMINAMATH_GPT_sum_of_primes_eq_24_l736_73627

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

variable (a b c : ℕ)

theorem sum_of_primes_eq_24 (h1 : is_prime a) (h2 : is_prime b) (h3 : is_prime c)
    (h4 : a * b + b * c = 119) : a + b + c = 24 :=
sorry

end NUMINAMATH_GPT_sum_of_primes_eq_24_l736_73627


namespace NUMINAMATH_GPT_probability_of_divisibility_by_7_l736_73639

noncomputable def count_valid_numbers : Nat :=
  -- Implementation of the count of all five-digit numbers 
  -- such that the sum of the digits is 30 
  sorry

noncomputable def count_divisible_by_7 : Nat :=
  -- Implementation of the count of numbers among these 
  -- which are divisible by 7
  sorry

theorem probability_of_divisibility_by_7 :
  count_divisible_by_7 * 5 = count_valid_numbers :=
sorry

end NUMINAMATH_GPT_probability_of_divisibility_by_7_l736_73639


namespace NUMINAMATH_GPT_can_form_sets_l736_73602

def clearly_defined (s : Set α) : Prop := ∀ x ∈ s, True
def not_clearly_defined (s : Set α) : Prop := ¬clearly_defined s

def cubes := {x : Type | True} -- Placeholder for the actual definition
def major_supermarkets := {x : Type | True} -- Placeholder for the actual definition
def difficult_math_problems := {x : Type | True} -- Placeholder for the actual definition
def famous_dancers := {x : Type | True} -- Placeholder for the actual definition
def products_2012 := {x : Type | True} -- Placeholder for the actual definition
def points_on_axes := {x : ℝ × ℝ | x.1 = 0 ∨ x.2 = 0}

theorem can_form_sets :
  (clearly_defined cubes) ∧
  (not_clearly_defined major_supermarkets) ∧
  (not_clearly_defined difficult_math_problems) ∧
  (not_clearly_defined famous_dancers) ∧
  (clearly_defined products_2012) ∧
  (clearly_defined points_on_axes) →
  True := 
by {
  -- Your proof goes here
  sorry
}

end NUMINAMATH_GPT_can_form_sets_l736_73602


namespace NUMINAMATH_GPT_closest_ratio_l736_73654

theorem closest_ratio (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : (x + y) / 2 = 3 * Real.sqrt (x * y)) :
  abs (x / y - 34) < abs (x / y - n) :=
by sorry

end NUMINAMATH_GPT_closest_ratio_l736_73654


namespace NUMINAMATH_GPT_problem1_problem2_l736_73687

-- Problem 1
theorem problem1 (x y : ℤ) (h1 : x = 2) (h2 : y = 2016) :
  (3*x + 2*y)*(3*x - 2*y) - (x + 2*y)*(5*x - 2*y) / (8*x) = -2015 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (h1 : x = 2) :
  ((x - 3) / (x^2 - 1)) * ((x^2 + 2*x + 1) / (x - 3)) - (1 / (x - 1) + 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l736_73687


namespace NUMINAMATH_GPT_sequence_of_perfect_squares_l736_73643

theorem sequence_of_perfect_squares (A B C D: ℕ)
(h1: 10 ≤ 10 * A + B) 
(h2 : 10 * A + B < 100) 
(h3 : (10 * A + B) % 3 = 0 ∨ (10 * A + B) % 3 = 1)
(hC : 1 ≤ C ∧ C ≤ 9)
(hD : 1 ≤ D ∧ D ≤ 9)
(hCD : (C + D) % 3 = 0)
(hAB_square : ∃ k₁ : ℕ, k₁^2 = 10 * A + B) 
(hACDB_square : ∃ k₂ : ℕ, k₂^2 = 1000 * A + 100 * C + 10 * D + B) 
(hACCDDB_square : ∃ k₃ : ℕ, k₃^2 = 100000 * A + 10000 * C + 1000 * C + 100 * D + 10 * D + B) :
∀ n: ℕ, ∃ k : ℕ, k^2 = (10^n * A + (10^(n/2) * C) + (10^(n/2) * D) + B) := 
by
  sorry

end NUMINAMATH_GPT_sequence_of_perfect_squares_l736_73643


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l736_73660

theorem necessary_and_sufficient_condition (a : ℝ) : (a > 1) ↔ ∀ x : ℝ, (x^2 - 2*x + a > 0) :=
by 
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l736_73660


namespace NUMINAMATH_GPT_value_of_k_l736_73631

theorem value_of_k :
  3^1999 - 3^1998 - 3^1997 + 3^1996 = 16 * 3^1996 :=
by sorry

end NUMINAMATH_GPT_value_of_k_l736_73631


namespace NUMINAMATH_GPT_product_of_odd_primes_mod_sixteen_l736_73617

-- Define the set of odd primes less than 16
def odd_primes_less_than_sixteen : List ℕ := [3, 5, 7, 11, 13]

-- Define the product of all odd primes less than 16
def N : ℕ := odd_primes_less_than_sixteen.foldl (· * ·) 1

-- Proposition to prove: N ≡ 7 (mod 16)
theorem product_of_odd_primes_mod_sixteen :
  (N % 16) = 7 :=
  sorry

end NUMINAMATH_GPT_product_of_odd_primes_mod_sixteen_l736_73617


namespace NUMINAMATH_GPT_fundraiser_brownies_l736_73657

-- Definitions derived from the conditions in the problem statement
def brownie_price := 2
def cookie_price := 2
def donut_price := 2

def students_bringing_brownies (B : Nat) := B
def students_bringing_cookies := 20
def students_bringing_donuts := 15

def brownies_per_student := 12
def cookies_per_student := 24
def donuts_per_student := 12

def total_amount_raised := 2040

theorem fundraiser_brownies (B : Nat) :
  24 * B + 20 * 24 * 2 + 15 * 12 * 2 = total_amount_raised → B = 30 :=
by
  sorry

end NUMINAMATH_GPT_fundraiser_brownies_l736_73657


namespace NUMINAMATH_GPT_neg_p_sufficient_for_neg_q_l736_73666

def p (a : ℝ) := a ≤ 2
def q (a : ℝ) := a * (a - 2) ≤ 0

theorem neg_p_sufficient_for_neg_q (a : ℝ) : ¬ p a → ¬ q a :=
sorry

end NUMINAMATH_GPT_neg_p_sufficient_for_neg_q_l736_73666


namespace NUMINAMATH_GPT_iron_per_horseshoe_l736_73683

def num_farms := 2
def num_horses_per_farm := 2
def num_stables := 2
def num_horses_per_stable := 5
def num_horseshoes_per_horse := 4
def iron_available := 400
def num_horses_riding_school := 36

-- Lean theorem statement
theorem iron_per_horseshoe : 
  (iron_available / (num_farms * num_horses_per_farm * num_horseshoes_per_horse 
  + num_stables * num_horses_per_stable * num_horseshoes_per_horse 
  + num_horses_riding_school * num_horseshoes_per_horse)) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_iron_per_horseshoe_l736_73683


namespace NUMINAMATH_GPT_colton_stickers_final_count_l736_73698

-- Definitions based on conditions
def initial_stickers := 200
def stickers_given_to_7_friends := 6 * 7
def stickers_given_to_mandy := stickers_given_to_7_friends + 8
def remaining_after_mandy := initial_stickers - stickers_given_to_7_friends - stickers_given_to_mandy
def stickers_distributed_to_4_friends := remaining_after_mandy / 2
def remaining_after_4_friends := remaining_after_mandy - stickers_distributed_to_4_friends
def given_to_justin := 2 * remaining_after_4_friends / 3
def remaining_after_justin := remaining_after_4_friends - given_to_justin
def given_to_karen := remaining_after_justin / 5
def final_stickers := remaining_after_justin - given_to_karen

-- Theorem to state the proof problem
theorem colton_stickers_final_count : final_stickers = 15 := by
  sorry

end NUMINAMATH_GPT_colton_stickers_final_count_l736_73698


namespace NUMINAMATH_GPT_ab_range_l736_73647

theorem ab_range (a b : ℝ) : (a + b = 1/2) → ab ≤ 1/16 :=
by
  sorry

end NUMINAMATH_GPT_ab_range_l736_73647


namespace NUMINAMATH_GPT_geometric_sequence_sum_l736_73605

theorem geometric_sequence_sum (n : ℕ) (a : ℕ → ℕ := λ k => 2^k) 
  (S : ℕ → ℕ := λ k => (1 - 2^k) / (1 - 2)) :
  S (n + 1) = 2 * a n - 1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l736_73605


namespace NUMINAMATH_GPT_total_children_in_school_l736_73640

theorem total_children_in_school (B : ℕ) (C : ℕ) 
  (h1 : B = 2 * C)
  (h2 : B = 4 * (C - 350)) :
  C = 700 :=
by sorry

end NUMINAMATH_GPT_total_children_in_school_l736_73640


namespace NUMINAMATH_GPT_solution_set_of_inequality_l736_73638

-- Definitions for the problem
def inequality (x : ℝ) : Prop := (1 + x) * (2 - x) * (3 + x^2) > 0

-- Statement of the theorem
theorem solution_set_of_inequality :
  {x : ℝ | inequality x} = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l736_73638


namespace NUMINAMATH_GPT_product_of_two_integers_l736_73674

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 18) (h2 : x^2 - y^2 = 36) : x * y = 80 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_integers_l736_73674


namespace NUMINAMATH_GPT_inequality_solution_l736_73672

open Set

theorem inequality_solution :
  {x : ℝ | |x + 1| - 2 > 0} = {x : ℝ | x < -3} ∪ {x : ℝ | x > 1} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l736_73672


namespace NUMINAMATH_GPT_stored_energy_in_doubled_square_l736_73614

noncomputable def energy (q : ℝ) (d : ℝ) : ℝ := q^2 / d

theorem stored_energy_in_doubled_square (q d : ℝ) (h : energy q d * 4 = 20) :
  energy q (2 * d) * 4 = 10 := by
  -- Add steps: Show that energy proportional to 1/d means energy at 2d is half compared to at d
  sorry

end NUMINAMATH_GPT_stored_energy_in_doubled_square_l736_73614


namespace NUMINAMATH_GPT_first_term_formula_correct_l736_73629

theorem first_term_formula_correct
  (S n d a : ℝ) 
  (h_sum_formula : S = (n / 2) * (2 * a + (n - 1) * d)) :
  a = (S / n) + (n - 1) * (d / 2) := 
sorry

end NUMINAMATH_GPT_first_term_formula_correct_l736_73629


namespace NUMINAMATH_GPT_income_percentage_less_l736_73606

-- Definitions representing the conditions
variables (T M J : ℝ)
variables (h1 : M = 1.60 * T) (h2 : M = 1.12 * J)

-- The theorem stating the problem
theorem income_percentage_less : (100 - (T / J) * 100) = 30 :=
by
  sorry

end NUMINAMATH_GPT_income_percentage_less_l736_73606


namespace NUMINAMATH_GPT_distinct_intersection_points_l736_73603

theorem distinct_intersection_points : 
  ∃! (x y : ℝ), (x + 2*y = 6 ∧ x - 3*y = 2) ∨ (x + 2*y = 6 ∧ 4*x + y = 14) :=
by
  -- proof would be here
  sorry

end NUMINAMATH_GPT_distinct_intersection_points_l736_73603


namespace NUMINAMATH_GPT_base7_product_digit_sum_l736_73644

noncomputable def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 350 => 3 * 7 + 5
  | 217 => 2 * 7 + 1
  | _ => 0

noncomputable def base10_to_base7 (n : Nat) : Nat := 
  if n = 390 then 1065 else 0

noncomputable def digit_sum_in_base7 (n : Nat) : Nat :=
  if n = 1065 then 1 + 0 + 6 + 5 else 0

noncomputable def sum_to_base7 (n : Nat) : Nat :=
  if n = 12 then 15 else 0

theorem base7_product_digit_sum :
  digit_sum_in_base7 (base10_to_base7 (base7_to_base10 350 * base7_to_base10 217)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_base7_product_digit_sum_l736_73644


namespace NUMINAMATH_GPT_frances_towel_weight_in_ounces_l736_73699

theorem frances_towel_weight_in_ounces :
  (∀ Mary_towels Frances_towels : ℕ,
    Mary_towels = 4 * Frances_towels →
    Mary_towels = 24 →
    (Mary_towels + Frances_towels) * 2 = 60 →
    Frances_towels * 2 * 16 = 192) :=
by
  intros Mary_towels Frances_towels h1 h2 h3
  sorry

end NUMINAMATH_GPT_frances_towel_weight_in_ounces_l736_73699


namespace NUMINAMATH_GPT_triangle_angle_R_measure_l736_73676

theorem triangle_angle_R_measure :
  ∀ (P Q R : ℝ),
  P + Q + R = 180 ∧ P = 70 ∧ Q = 2 * R + 15 → R = 95 / 3 :=
by
  intros P Q R h
  sorry

end NUMINAMATH_GPT_triangle_angle_R_measure_l736_73676


namespace NUMINAMATH_GPT_middle_part_of_proportion_l736_73610

theorem middle_part_of_proportion (x : ℚ) (h : x + (1/4) * x + (1/8) * x = 104) : (1/4) * x = 208 / 11 :=
by
  sorry

end NUMINAMATH_GPT_middle_part_of_proportion_l736_73610


namespace NUMINAMATH_GPT_eve_total_spend_l736_73612

def hand_mitts_cost : ℝ := 14.00
def apron_cost : ℝ := 16.00
def utensils_cost : ℝ := 10.00
def knife_cost : ℝ := 2 * utensils_cost
def discount_percent : ℝ := 0.25
def nieces_count : ℕ := 3

def total_cost_before_discount : ℝ :=
  (hand_mitts_cost + apron_cost + utensils_cost + knife_cost) * nieces_count

def discount_amount : ℝ :=
  discount_percent * total_cost_before_discount

def total_cost_after_discount : ℝ :=
  total_cost_before_discount - discount_amount

theorem eve_total_spend : total_cost_after_discount = 135.00 := by
  sorry

end NUMINAMATH_GPT_eve_total_spend_l736_73612


namespace NUMINAMATH_GPT_number_of_geese_more_than_ducks_l736_73695

theorem number_of_geese_more_than_ducks (geese ducks : ℝ) (h1 : geese = 58.0) (h2 : ducks = 37.0) :
  geese - ducks = 21.0 :=
by
  sorry

end NUMINAMATH_GPT_number_of_geese_more_than_ducks_l736_73695


namespace NUMINAMATH_GPT_value_of_w_over_y_l736_73615

theorem value_of_w_over_y (w x y : ℝ) (h1 : w / x = 1 / 3) (h2 : (x + y) / y = 3) : w / y = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_w_over_y_l736_73615


namespace NUMINAMATH_GPT_quadratic_roots_l736_73664

theorem quadratic_roots {α p q : ℝ} (hα : 0 < α ∧ α ≤ 1) (hroots : ∃ x : ℝ, x^2 + p * x + q = 0) :
  ∃ x : ℝ, α * x^2 + p * x + q = 0 :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_l736_73664


namespace NUMINAMATH_GPT_ellipse_parabola_common_point_l736_73690

theorem ellipse_parabola_common_point (a : ℝ) :
  (∃ (x y : ℝ), x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔ -1 ≤ a ∧ a ≤ 17 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_ellipse_parabola_common_point_l736_73690


namespace NUMINAMATH_GPT_combined_percentage_increase_l736_73628

def initial_interval_days : ℝ := 50
def additive_A_effect : ℝ := 0.20
def additive_B_effect : ℝ := 0.30
def additive_C_effect : ℝ := 0.40

theorem combined_percentage_increase :
  ((1 + additive_A_effect) * (1 + additive_B_effect) * (1 + additive_C_effect) - 1) * 100 = 118.4 :=
by
  norm_num
  sorry

end NUMINAMATH_GPT_combined_percentage_increase_l736_73628


namespace NUMINAMATH_GPT_remainder_of_N_mod_16_is_7_l736_73669

-- Let N be the product of all odd primes less than 16
def odd_primes : List ℕ := [3, 5, 7, 11, 13]

-- Calculate the product N of these primes
def N : ℕ := odd_primes.foldr (· * ·) 1

-- Prove the remainder of N when divided by 16 is 7
theorem remainder_of_N_mod_16_is_7 : N % 16 = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_of_N_mod_16_is_7_l736_73669


namespace NUMINAMATH_GPT_minimum_omega_l736_73670

theorem minimum_omega (ω : ℝ) (hω_pos : ω > 0)
  (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin (ω * x + Real.pi / 3))
  (C : ℝ → ℝ) (hC : ∀ x, C x = Real.sin (ω * (x + Real.pi / 2) + Real.pi / 3)) :
  (∀ x, C x = C (-x)) ↔ ω = 1 / 3 := by
sorry

end NUMINAMATH_GPT_minimum_omega_l736_73670


namespace NUMINAMATH_GPT_notebook_cost_l736_73645

-- Define the conditions
def cost_pen := 1
def num_pens := 3
def num_notebooks := 4
def cost_folder := 5
def num_folders := 2
def initial_bill := 50
def change_back := 25

-- Calculate derived values
def total_spent := initial_bill - change_back
def total_cost_pens := num_pens * cost_pen
def total_cost_folders := num_folders * cost_folder
def total_cost_notebooks := total_spent - total_cost_pens - total_cost_folders

-- Calculate the cost per notebook
def cost_per_notebook := total_cost_notebooks / num_notebooks

-- Proof statement
theorem notebook_cost : cost_per_notebook = 3 := by
  sorry

end NUMINAMATH_GPT_notebook_cost_l736_73645


namespace NUMINAMATH_GPT_range_of_solutions_l736_73684

-- Define the function f(x) = x^2 - bx - 5
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b * x - 5

theorem range_of_solutions (b : ℝ) :
  (f b (-2) = 5) ∧ 
  (f b (-1) = -1) ∧ 
  (f b 4 = -1) ∧ 
  (f b 5 = 5) →
  ∃ x1 x2, (-2 < x1 ∧ x1 < -1) ∨ (4 < x2 ∧ x2 < 5) ∧ f b x1 = 0 ∧ f b x2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_solutions_l736_73684


namespace NUMINAMATH_GPT_original_price_of_cycle_l736_73697

/--
A man bought a cycle for some amount and sold it at a loss of 20%.
The selling price of the cycle is Rs. 1280.
What was the original price of the cycle?
-/
theorem original_price_of_cycle
    (loss_percent : ℝ)
    (selling_price : ℝ)
    (original_price : ℝ)
    (h_loss_percent : loss_percent = 0.20)
    (h_selling_price : selling_price = 1280)
    (h_selling_eqn : selling_price = (1 - loss_percent) * original_price) :
    original_price = 1600 :=
sorry

end NUMINAMATH_GPT_original_price_of_cycle_l736_73697


namespace NUMINAMATH_GPT_determine_b_value_l736_73653

theorem determine_b_value 
  (a : ℝ) 
  (b : ℝ) 
  (h₀ : a > 0) 
  (h₁ : a ≠ 1) 
  (h₂ : 2 * a^(2 - b) + 1 = 3) : 
  b = 2 := 
by 
  sorry

end NUMINAMATH_GPT_determine_b_value_l736_73653


namespace NUMINAMATH_GPT_monotonicity_and_extremum_of_f_l736_73601

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem monotonicity_and_extremum_of_f :
  (∀ x, 1 < x → ∀ y, x < y → f x < f y) ∧
  (∀ x, 0 < x → x < 1 → ∀ y, x < y → y < 1 → f x > f y) ∧
  (f 1 = -1) :=
by
  sorry

end NUMINAMATH_GPT_monotonicity_and_extremum_of_f_l736_73601


namespace NUMINAMATH_GPT_total_payroll_calc_l736_73611

theorem total_payroll_calc
  (h : ℕ := 129)          -- pay per day for heavy operators
  (l : ℕ := 82)           -- pay per day for general laborers
  (n : ℕ := 31)           -- total number of people hired
  (g : ℕ := 1)            -- number of general laborers employed
  : (h * (n - g) + l * g) = 3952 := 
by
  sorry

end NUMINAMATH_GPT_total_payroll_calc_l736_73611


namespace NUMINAMATH_GPT_abs_c_eq_116_l736_73632

theorem abs_c_eq_116 (a b c : ℤ) (h : Int.gcd a (Int.gcd b c) = 1) 
  (h_eq : a * (Complex.ofReal 3 + Complex.I) ^ 4 + 
          b * (Complex.ofReal 3 + Complex.I) ^ 3 + 
          c * (Complex.ofReal 3 + Complex.I) ^ 2 + 
          b * (Complex.ofReal 3 + Complex.I) + 
          a = 0) : 
  |c| = 116 :=
sorry

end NUMINAMATH_GPT_abs_c_eq_116_l736_73632


namespace NUMINAMATH_GPT_symmetric_line_b_value_l736_73685

theorem symmetric_line_b_value (b : ℝ) : 
  (∃ l1 l2 : ℝ × ℝ → Prop, 
    (∀ (x y : ℝ), l1 (x, y) ↔ y = -2 * x + b) ∧ 
    (∃ p2 : ℝ × ℝ, p2 = (1, 6) ∧ l2 p2) ∧
    l2 (-1, 6) ∧ 
    (∀ (x y : ℝ), l1 (x, y) ↔ l2 (-x, y))) →
  b = 4 := 
by
  sorry

end NUMINAMATH_GPT_symmetric_line_b_value_l736_73685


namespace NUMINAMATH_GPT_find_added_number_l736_73652

theorem find_added_number (a : ℕ → ℝ) (x : ℝ) (h_init : a 1 = 2) (h_a3 : a 3 = 6)
  (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)  -- arithmetic sequence condition
  (h_geom : (a 4 + x)^2 = (a 1 + x) * (a 5 + x)) : 
  x = -11 := 
sorry

end NUMINAMATH_GPT_find_added_number_l736_73652


namespace NUMINAMATH_GPT_major_minor_axis_lengths_foci_vertices_coordinates_l736_73691

-- Given conditions
def ellipse_eq (x y : ℝ) : Prop := 16 * x^2 + 25 * y^2 = 400

-- Proof Tasks
theorem major_minor_axis_lengths : 
  (∃ a b : ℝ, a = 5 ∧ b = 4 ∧ 2 * a = 10) :=
by sorry

theorem foci_vertices_coordinates : 
  (∃ c : ℝ, 
    (c = 3) ∧ 
    (∀ x y : ℝ, ellipse_eq x y → (x = 0 → y = 4 ∨ y = -4) ∧ (y = 0 → x = 5 ∨ x = -5))) :=
by sorry

end NUMINAMATH_GPT_major_minor_axis_lengths_foci_vertices_coordinates_l736_73691


namespace NUMINAMATH_GPT_cricket_team_members_l736_73658

-- Define variables and conditions
variable (n : ℕ) -- let n be the number of team members
variable (T : ℕ) -- let T be the total age of the team
variable (average_team_age : ℕ := 24) -- given average age of the team
variable (wicket_keeper_age : ℕ := average_team_age + 3) -- wicket keeper is 3 years older
variable (remaining_players_average_age : ℕ := average_team_age - 1) -- remaining players' average age

-- Given condition which relates to the total age
axiom total_age_condition : T = average_team_age * n

-- Given condition for the total age of remaining players
axiom remaining_players_total_age : T - 24 - 27 = remaining_players_average_age * (n - 2)

-- Prove the number of members in the cricket team
theorem cricket_team_members : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_cricket_team_members_l736_73658


namespace NUMINAMATH_GPT_proof_inequality_l736_73693

noncomputable def proof_problem (p q : ℝ) (m n : ℕ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) : Prop :=
  (1 - p^m)^n + (1 - q^n)^m ≥ 1

theorem proof_inequality (p q : ℝ) (m n : ℕ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_proof_inequality_l736_73693


namespace NUMINAMATH_GPT_work_completion_time_extension_l736_73604

theorem work_completion_time_extension
    (total_men : ℕ) (initial_days : ℕ) (remaining_men : ℕ) (man_days : ℕ) :
    total_men = 100 →
    initial_days = 20 →
    remaining_men = 50 →
    man_days = total_men * initial_days →
    (man_days / remaining_men) - initial_days = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_work_completion_time_extension_l736_73604


namespace NUMINAMATH_GPT_original_photo_dimensions_l736_73642

theorem original_photo_dimensions (squares_before : ℕ) 
    (squares_after : ℕ) 
    (vertical_length : ℕ) 
    (horizontal_length : ℕ) 
    (side_length : ℕ)
    (h1 : squares_before = 1812)
    (h2 : squares_after = 2018)
    (h3 : side_length = 1) :
    vertical_length = 101 ∧ horizontal_length = 803 :=
by
    sorry

end NUMINAMATH_GPT_original_photo_dimensions_l736_73642


namespace NUMINAMATH_GPT_gum_pack_size_is_5_l736_73634
noncomputable def find_gum_pack_size (x : ℕ) : Prop :=
  let cherry_initial := 25
  let grape_initial := 40
  let cherry_lost := cherry_initial - 2 * x
  let grape_found := grape_initial + 4 * x
  (cherry_lost * grape_found) = (cherry_initial * grape_initial)

theorem gum_pack_size_is_5 : find_gum_pack_size 5 :=
by
  sorry

end NUMINAMATH_GPT_gum_pack_size_is_5_l736_73634


namespace NUMINAMATH_GPT_movie_ticket_vs_popcorn_difference_l736_73667

variable (P : ℝ) -- cost of a bucket of popcorn
variable (d : ℝ) -- cost of a drink
variable (c : ℝ) -- cost of a candy
variable (t : ℝ) -- cost of a movie ticket

-- Given conditions
axiom h1 : t = 8
axiom h2 : d = P + 1
axiom h3 : c = (P + 1) / 2
axiom h4 : t + P + d + c = 22

-- Question rewritten: Prove that the difference between the normal cost of a movie ticket and the cost of a bucket of popcorn is 3.
theorem movie_ticket_vs_popcorn_difference : t - P = 3 :=
by
  sorry

end NUMINAMATH_GPT_movie_ticket_vs_popcorn_difference_l736_73667


namespace NUMINAMATH_GPT_alex_height_l736_73621

theorem alex_height
  (tree_height: ℚ) (tree_shadow: ℚ) (alex_shadow_in_inches: ℚ)
  (h_tree: tree_height = 50)
  (h_shadow_tree: tree_shadow = 25)
  (h_shadow_alex: alex_shadow_in_inches = 20) :
  ∃ alex_height_in_feet: ℚ, alex_height_in_feet = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_alex_height_l736_73621


namespace NUMINAMATH_GPT_function_parity_l736_73649

noncomputable def f : ℝ → ℝ := sorry

-- Condition: f satisfies the functional equation for all x, y in Real numbers
axiom functional_eqn (x y : ℝ) : f (x + y) + f (x - y) = 2 * f x * f y

-- Prove that the function could be either odd or even.
theorem function_parity : (∀ x, f (-x) = f x) ∨ (∀ x, f (-x) = -f x) := 
sorry

end NUMINAMATH_GPT_function_parity_l736_73649


namespace NUMINAMATH_GPT_eating_time_l736_73626

-- Define the eating rates of Mr. Fat, Mr. Thin, and Mr. Medium
def mrFat_rate := 1 / 15
def mrThin_rate := 1 / 35
def mrMedium_rate := 1 / 25

-- Define the combined eating rate
def combined_rate := mrFat_rate + mrThin_rate + mrMedium_rate

-- Define the amount of cereal to be eaten
def amount_cereal := 5

-- Prove that the time taken to eat the cereal is 2625 / 71 minutes
theorem eating_time : amount_cereal / combined_rate = 2625 / 71 :=
by 
  -- Here should be the proof, but it is skipped
  sorry

end NUMINAMATH_GPT_eating_time_l736_73626


namespace NUMINAMATH_GPT_sheets_in_stack_l736_73680

theorem sheets_in_stack (n : ℕ) (thickness : ℝ) (height : ℝ) 
  (h1 : n = 400) (h2 : thickness = 4) (h3 : height = 10) : 
  n * height / thickness = 1000 := 
by 
  sorry

end NUMINAMATH_GPT_sheets_in_stack_l736_73680


namespace NUMINAMATH_GPT_polynomial_divisible_by_a_plus_1_l736_73613

theorem polynomial_divisible_by_a_plus_1 (a : ℤ) : (3 * a + 5) ^ 2 - 4 ∣ a + 1 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_divisible_by_a_plus_1_l736_73613


namespace NUMINAMATH_GPT_inlet_rate_480_l736_73688

theorem inlet_rate_480 (capacity : ℕ) (T_outlet : ℕ) (T_outlet_inlet : ℕ) (R_i : ℕ) :
  capacity = 11520 →
  T_outlet = 8 →
  T_outlet_inlet = 12 →
  R_i = 480 :=
by
  intros
  sorry

end NUMINAMATH_GPT_inlet_rate_480_l736_73688


namespace NUMINAMATH_GPT_initial_erasers_in_box_l736_73678

-- Definitions based on the conditions
def erasers_in_bag_jane := 15
def erasers_taken_out_doris := 54
def erasers_left_in_box := 15

-- Theorem statement
theorem initial_erasers_in_box : ∃ B_i : ℕ, B_i = erasers_taken_out_doris + erasers_left_in_box ∧ B_i = 69 :=
by
  use 69
  -- omitted proof steps
  sorry

end NUMINAMATH_GPT_initial_erasers_in_box_l736_73678


namespace NUMINAMATH_GPT_david_money_left_l736_73650

noncomputable def david_trip (S H : ℝ) : Prop :=
  S + H = 3200 ∧ H = 0.65 * S

theorem david_money_left : ∃ H, david_trip 1939.39 H ∧ |H - 1260.60| < 0.01 := by
  sorry

end NUMINAMATH_GPT_david_money_left_l736_73650


namespace NUMINAMATH_GPT_sin_2x_plus_one_equals_9_over_5_l736_73608

theorem sin_2x_plus_one_equals_9_over_5 (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : Real.sin (2 * x) + 1 = 9 / 5 :=
sorry

end NUMINAMATH_GPT_sin_2x_plus_one_equals_9_over_5_l736_73608


namespace NUMINAMATH_GPT_inequality_holds_l736_73662

theorem inequality_holds (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l736_73662


namespace NUMINAMATH_GPT_total_marbles_in_bag_l736_73633

theorem total_marbles_in_bag 
  (r b p : ℕ) 
  (h1 : 32 = r)
  (h2 : b = (7 * r) / 4) 
  (h3 : p = (3 * b) / 2) 
  : r + b + p = 172 := 
sorry

end NUMINAMATH_GPT_total_marbles_in_bag_l736_73633


namespace NUMINAMATH_GPT_max_angle_position_l736_73616

-- Definitions for points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

-- Definitions for points A and B on the X-axis
def A (a : ℝ) : Point := { x := -a, y := 0 }
def B (a : ℝ) : Point := { x := a, y := 0 }

-- Definition for point C moving along the line y = 10 - x
def moves_along_line (C : Point) : Prop :=
  C.y = 10 - C.x

-- Definition for calculating the angle ACB (gamma)
def angle_ACB (A B C : Point) : ℝ := sorry -- The detailed function to calculate angle is omitted for brevity

-- Main statement to prove
theorem max_angle_position (a : ℝ) (C : Point) (ha : 0 ≤ a ∧ a ≤ 10) (hC : moves_along_line C) :
  (C = { x := 4, y := 6 } ∨ C = { x := 16, y := -6 }) ↔ (∀ C', moves_along_line C' → (angle_ACB (A a) (B a) C') ≤ angle_ACB (A a) (B a) C) :=
sorry

end NUMINAMATH_GPT_max_angle_position_l736_73616
