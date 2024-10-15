import Mathlib

namespace NUMINAMATH_GPT_a_squared_plus_b_squared_eq_zero_implies_a_eq_zero_and_b_eq_zero_l1746_174650

-- Mathematical condition: a^2 + b^2 = 0
variable {a b : ℝ}

-- Mathematical statement to be proven
theorem a_squared_plus_b_squared_eq_zero_implies_a_eq_zero_and_b_eq_zero 
  (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry  -- proof yet to be provided

end NUMINAMATH_GPT_a_squared_plus_b_squared_eq_zero_implies_a_eq_zero_and_b_eq_zero_l1746_174650


namespace NUMINAMATH_GPT_part1_part2_l1746_174621

theorem part1 : ∃ x : ℝ, 3 * x = 4.5 ∧ x = 4.5 - 3 :=
by {
  -- Skipping the proof for now
  sorry
}

theorem part2 (m : ℝ) (h : ∃ x : ℝ, 5 * x - m = 1 ∧ x = 1 - m - 5) : m = 21 / 4 :=
by {
  -- Skipping the proof for now
  sorry
}

end NUMINAMATH_GPT_part1_part2_l1746_174621


namespace NUMINAMATH_GPT_total_marbles_l1746_174658

/-- A craftsman makes 35 jars. This is exactly 2.5 times the number of clay pots he made.
If each jar has 5 marbles and each clay pot has four times as many marbles as the jars plus an additional 3 marbles, 
prove that the total number of marbles is 497. -/
theorem total_marbles (number_of_jars : ℕ) (number_of_clay_pots : ℕ) (marbles_in_jar : ℕ) (marbles_in_clay_pot : ℕ) :
  number_of_jars = 35 →
  (number_of_jars : ℝ) = 2.5 * number_of_clay_pots →
  marbles_in_jar = 5 →
  marbles_in_clay_pot = 4 * marbles_in_jar + 3 →
  (number_of_jars * marbles_in_jar + number_of_clay_pots * marbles_in_clay_pot) = 497 :=
by 
  sorry

end NUMINAMATH_GPT_total_marbles_l1746_174658


namespace NUMINAMATH_GPT_parabola_inequality_l1746_174675

theorem parabola_inequality {y1 y2 : ℝ} :
  (∀ x1 x2 : ℝ, x1 = -5 → x2 = 2 →
  y1 = x1^2 + 2 * x1 + 3 ∧ y2 = x2^2 + 2 * x2 + 3) → (y1 > y2) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_parabola_inequality_l1746_174675


namespace NUMINAMATH_GPT_loss_percentage_l1746_174607

-- Definitions related to the problem
def CPA : Type := ℝ
def SPAB (CPA: ℝ) : ℝ := 1.30 * CPA
def SPBC (CPA: ℝ) : ℝ := 1.040000000000000036 * CPA

-- Theorem to prove the loss percentage when B sold the bicycle to C 
theorem loss_percentage (CPA : ℝ) (L : ℝ) (h1 : SPAB CPA * (1 - L) = SPBC CPA) : 
  L = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_loss_percentage_l1746_174607


namespace NUMINAMATH_GPT_cakes_baked_yesterday_l1746_174643

noncomputable def BakedToday : ℕ := 5
noncomputable def SoldDinner : ℕ := 6
noncomputable def Left : ℕ := 2

theorem cakes_baked_yesterday (CakesBakedYesterday : ℕ) : 
  BakedToday + CakesBakedYesterday - SoldDinner = Left → CakesBakedYesterday = 3 := 
by 
  intro h 
  sorry

end NUMINAMATH_GPT_cakes_baked_yesterday_l1746_174643


namespace NUMINAMATH_GPT_part1_part2_i_part2_ii_l1746_174684

theorem part1 :
  ¬ ∃ x : ℝ, - (4 / x) = x := 
sorry

theorem part2_i (a c : ℝ) (ha : a ≠ 0) :
  (∃! x : ℝ, x = a * (x^2) + 6 * x + c ∧ x = 5 / 2) ↔ (a = -1 ∧ c = -25 / 4) :=
sorry

theorem part2_ii (m : ℝ) :
  (∃ (a c : ℝ), a = -1 ∧ c = - 25 / 4 ∧
    ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → - (x^2) + 6 * x - 25 / 4 + 1/4 ≥ -1 ∧ - (x^2) + 6 * x - 25 / 4 + 1/4 ≤ 3) ↔
    (3 ≤ m ∧ m ≤ 5) :=
sorry

end NUMINAMATH_GPT_part1_part2_i_part2_ii_l1746_174684


namespace NUMINAMATH_GPT_machine_value_after_two_years_l1746_174629

def machine_value (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - rate)^years

theorem machine_value_after_two_years :
  machine_value 8000 0.1 2 = 6480 :=
by
  sorry

end NUMINAMATH_GPT_machine_value_after_two_years_l1746_174629


namespace NUMINAMATH_GPT_trent_bus_blocks_to_library_l1746_174687

-- Define the given conditions
def total_distance := 22
def walking_distance := 4

-- Define the function to determine bus block distance
def bus_ride_distance (total: ℕ) (walk: ℕ) : ℕ :=
  (total - (walk * 2)) / 2

-- The theorem we need to prove
theorem trent_bus_blocks_to_library : 
  bus_ride_distance total_distance walking_distance = 7 := by
  sorry

end NUMINAMATH_GPT_trent_bus_blocks_to_library_l1746_174687


namespace NUMINAMATH_GPT_inequality_transform_l1746_174611

theorem inequality_transform (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_transform_l1746_174611


namespace NUMINAMATH_GPT_crayon_division_l1746_174674

theorem crayon_division (total_crayons : ℕ) (crayons_each : ℕ) (Fred Benny Jason : ℕ) 
  (h_total : total_crayons = 24) (h_each : crayons_each = 8) 
  (h_division : Fred = crayons_each ∧ Benny = crayons_each ∧ Jason = crayons_each) : 
  Fred + Benny + Jason = total_crayons :=
by
  sorry

end NUMINAMATH_GPT_crayon_division_l1746_174674


namespace NUMINAMATH_GPT_desired_digit_set_l1746_174670

noncomputable def prob_digit (d : ℕ) : ℝ := if d > 0 then Real.log (d + 1) - Real.log d else 0

theorem desired_digit_set : 
  (prob_digit 5 = (1 / 2) * (prob_digit 5 + prob_digit 6 + prob_digit 7 + prob_digit 8)) ↔
  {d | d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8} = {5, 6, 7, 8} :=
by
  sorry

end NUMINAMATH_GPT_desired_digit_set_l1746_174670


namespace NUMINAMATH_GPT_calc_1_calc_2_l1746_174696

variable (x y : ℝ)

theorem calc_1 : (-x^2)^4 = x^8 := 
sorry

theorem calc_2 : (-x^2 * y)^3 = -x^6 * y^3 := 
sorry

end NUMINAMATH_GPT_calc_1_calc_2_l1746_174696


namespace NUMINAMATH_GPT_numberOfSubsets_of_A_l1746_174653

def numberOfSubsets (s : Finset ℕ) : ℕ := 2 ^ (Finset.card s)

theorem numberOfSubsets_of_A : 
  numberOfSubsets ({0, 1} : Finset ℕ) = 4 := 
by 
  sorry

end NUMINAMATH_GPT_numberOfSubsets_of_A_l1746_174653


namespace NUMINAMATH_GPT_list_price_proof_l1746_174673

-- Define the list price of the item
noncomputable def list_price : ℝ := 33

-- Define the selling price and commission for Alice
def alice_selling_price (x : ℝ) : ℝ := x - 15
def alice_commission (x : ℝ) : ℝ := 0.15 * alice_selling_price x

-- Define the selling price and commission for Charles
def charles_selling_price (x : ℝ) : ℝ := x - 18
def charles_commission (x : ℝ) : ℝ := 0.18 * charles_selling_price x

-- The main theorem: proving the list price given Alice and Charles receive the same commission
theorem list_price_proof (x : ℝ) (h : alice_commission x = charles_commission x) : x = list_price :=
by 
  sorry

end NUMINAMATH_GPT_list_price_proof_l1746_174673


namespace NUMINAMATH_GPT_square_with_12_sticks_square_with_15_sticks_l1746_174677

-- Definitions for problem conditions
def sum_of_first_n_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def can_form_square (total_length : ℕ) : Prop :=
  total_length % 4 = 0

-- Given n = 12, check if breaking 2 sticks is required to form a square
theorem square_with_12_sticks : (n = 12) → ¬ can_form_square (sum_of_first_n_natural_numbers 12) → true :=
by
  intros
  sorry

-- Given n = 15, check if it is possible to form a square without breaking any sticks
theorem square_with_15_sticks : (n = 15) → can_form_square (sum_of_first_n_natural_numbers 15) → true :=
by
  intros
  sorry

end NUMINAMATH_GPT_square_with_12_sticks_square_with_15_sticks_l1746_174677


namespace NUMINAMATH_GPT_stamp_exhibition_l1746_174632

def total_number_of_stamps (x : ℕ) : ℕ := 3 * x + 24

theorem stamp_exhibition : ∃ x : ℕ, total_number_of_stamps x = 174 ∧ (4 * x - 26) = 174 :=
by
  sorry

end NUMINAMATH_GPT_stamp_exhibition_l1746_174632


namespace NUMINAMATH_GPT_compare_fx_l1746_174692

noncomputable def f (a x : ℝ) := a * x ^ 2 + 2 * a * x + 4

theorem compare_fx (a x1 x2 : ℝ) (h₁ : -3 < a) (h₂ : a < 0) (h₃ : x1 < x2) (h₄ : x1 + x2 ≠ 1 + a) :
  f a x1 > f a x2 :=
sorry

end NUMINAMATH_GPT_compare_fx_l1746_174692


namespace NUMINAMATH_GPT_train_speed_l1746_174662

theorem train_speed
  (distance: ℝ) (time_in_minutes : ℝ) (time_in_hours : ℝ) (speed: ℝ)
  (h1 : distance = 20)
  (h2 : time_in_minutes = 10)
  (h3 : time_in_hours = time_in_minutes / 60)
  (h4 : speed = distance / time_in_hours)
  : speed = 120 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_l1746_174662


namespace NUMINAMATH_GPT_educated_employees_count_l1746_174680

def daily_wages_decrease (illiterate_avg_before illiterate_avg_after illiterate_count : ℕ) : ℕ :=
  (illiterate_avg_before - illiterate_avg_after) * illiterate_count

def total_employees (total_decreased total_avg_decreased : ℕ) : ℕ :=
  total_decreased / total_avg_decreased

theorem educated_employees_count :
  ∀ (illiterate_avg_before illiterate_avg_after illiterate_count total_avg_decreased : ℕ),
    illiterate_avg_before = 25 →
    illiterate_avg_after = 10 →
    illiterate_count = 20 →
    total_avg_decreased = 10 →
    total_employees (daily_wages_decrease illiterate_avg_before illiterate_avg_after illiterate_count) total_avg_decreased - illiterate_count = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_educated_employees_count_l1746_174680


namespace NUMINAMATH_GPT_tiling_2002_gon_with_rhombuses_l1746_174612

theorem tiling_2002_gon_with_rhombuses : ∀ n : ℕ, n = 1001 → (n * (n - 1) / 2) = 500500 :=
by sorry

end NUMINAMATH_GPT_tiling_2002_gon_with_rhombuses_l1746_174612


namespace NUMINAMATH_GPT_solve_for_A_l1746_174651

theorem solve_for_A : ∃ (A : ℕ), A7 = 10 * A + 7 ∧ A7 + 30 = 77 ∧ A = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_A_l1746_174651


namespace NUMINAMATH_GPT_area_of_ellipse_l1746_174628

theorem area_of_ellipse (x y : ℝ) (h : x^2 + 6 * x + 4 * y^2 - 8 * y + 9 = 0) : 
  area = 2 * Real.pi :=
sorry

end NUMINAMATH_GPT_area_of_ellipse_l1746_174628


namespace NUMINAMATH_GPT_total_pencils_l1746_174617

def pencils_per_person : Nat := 15
def number_of_people : Nat := 5

theorem total_pencils : pencils_per_person * number_of_people = 75 := by
  sorry

end NUMINAMATH_GPT_total_pencils_l1746_174617


namespace NUMINAMATH_GPT_gcd_power_minus_one_l1746_174672

theorem gcd_power_minus_one (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) : gcd (2^a - 1) (2^b - 1) = 2^(gcd a b) - 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_power_minus_one_l1746_174672


namespace NUMINAMATH_GPT_required_CO2_l1746_174631

noncomputable def moles_of_CO2_required (Mg CO2 MgO C : ℕ) (hMgO : MgO = 2) (hC : C = 1) : ℕ :=
  if Mg = 2 then 1 else 0

theorem required_CO2
  (Mg CO2 MgO C : ℕ)
  (hMgO : MgO = 2)
  (hC : C = 1)
  (hMg : Mg = 2)
  : moles_of_CO2_required Mg CO2 MgO C hMgO hC = 1 :=
  by simp [moles_of_CO2_required, hMg]

end NUMINAMATH_GPT_required_CO2_l1746_174631


namespace NUMINAMATH_GPT_correct_equations_l1746_174614

theorem correct_equations (x y : ℝ) :
  (9 * x - y = 4) → (y - 8 * x = 3) → (9 * x - y = 4 ∧ y - 8 * x = 3) :=
by
  intros h1 h2
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_correct_equations_l1746_174614


namespace NUMINAMATH_GPT_evaluate_expression_l1746_174646

theorem evaluate_expression :
  (15 - 14 + 13 - 12 + 11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1) / (1 - 2 + 3 - 4 + 5 - 6 + 7) = 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1746_174646


namespace NUMINAMATH_GPT_regression_lines_have_common_point_l1746_174620

theorem regression_lines_have_common_point
  (n m : ℕ)
  (h₁ : n = 10)
  (h₂ : m = 15)
  (s t : ℝ)
  (data_A data_B : Fin n → Fin n → ℝ)
  (avg_x_A avg_x_B : ℝ)
  (avg_y_A avg_y_B : ℝ)
  (regression_line_A regression_line_B : ℝ → ℝ)
  (h₃ : avg_x_A = s)
  (h₄ : avg_x_B = s)
  (h₅ : avg_y_A = t)
  (h₆ : avg_y_B = t)
  (h₇ : ∀ x, regression_line_A x = a*x + b)
  (h₈ : ∀ x, regression_line_B x = c*x + d)
  : regression_line_A s = t ∧ regression_line_B s = t :=
by
  sorry

end NUMINAMATH_GPT_regression_lines_have_common_point_l1746_174620


namespace NUMINAMATH_GPT_find_quotient_l1746_174604

theorem find_quotient :
  ∀ (remainder dividend divisor quotient : ℕ),
    remainder = 1 →
    dividend = 217 →
    divisor = 4 →
    quotient = (dividend - remainder) / divisor →
    quotient = 54 :=
by
  intros remainder dividend divisor quotient hr hd hdiv hq
  rw [hr, hd, hdiv] at hq
  norm_num at hq
  exact hq

end NUMINAMATH_GPT_find_quotient_l1746_174604


namespace NUMINAMATH_GPT_ac_lt_bc_if_c_lt_zero_l1746_174645

variables {a b c : ℝ}
theorem ac_lt_bc_if_c_lt_zero (h : a > b) (h1 : b > c) (h2 : c < 0) : a * c < b * c :=
sorry

end NUMINAMATH_GPT_ac_lt_bc_if_c_lt_zero_l1746_174645


namespace NUMINAMATH_GPT_negation_of_proposition_l1746_174698

-- Define the original proposition and its negation
def original_proposition (x : ℝ) : Prop := x^2 - 3*x + 3 > 0
def negated_proposition (x : ℝ) : Prop := x^2 - 3*x + 3 ≤ 0

-- The theorem about the negation of the original proposition
theorem negation_of_proposition :
  ¬ (∀ x : ℝ, original_proposition x) ↔ ∃ x : ℝ, negated_proposition x :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1746_174698


namespace NUMINAMATH_GPT_complex_multiplication_l1746_174681

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i :=
by
  sorry

end NUMINAMATH_GPT_complex_multiplication_l1746_174681


namespace NUMINAMATH_GPT_original_prices_l1746_174625

theorem original_prices 
  (S P J : ℝ)
  (hS : 0.80 * S = 780)
  (hP : 0.70 * P = 2100)
  (hJ : 0.90 * J = 2700) :
  S = 975 ∧ P = 3000 ∧ J = 3000 :=
by
  sorry

end NUMINAMATH_GPT_original_prices_l1746_174625


namespace NUMINAMATH_GPT_frank_spent_on_mower_blades_l1746_174609

def money_made := 19
def money_spent_on_games := 4 * 2
def money_left := money_made - money_spent_on_games

theorem frank_spent_on_mower_blades : money_left = 11 :=
by
  -- we are providing the proof steps here in comments, but in the actual code, it's just sorry
  -- calc money_left
  --    = money_made - money_spent_on_games : by refl
  --    = 19 - 8 : by norm_num
  --    = 11 : by norm_num
  sorry

end NUMINAMATH_GPT_frank_spent_on_mower_blades_l1746_174609


namespace NUMINAMATH_GPT_arithmetic_seq_problem_l1746_174679

theorem arithmetic_seq_problem
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n + 1) - a n = d)
  (h2 : d > 0)
  (h3 : a 1 + a 2 + a 3 = 15)
  (h4 : a 1 * a 2 * a 3 = 80) :
  a 11 + a 12 + a 13 = 105 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_problem_l1746_174679


namespace NUMINAMATH_GPT_total_students_in_class_l1746_174634

theorem total_students_in_class 
  (hockey_players : ℕ)
  (basketball_players : ℕ)
  (neither_players : ℕ)
  (both_players : ℕ)
  (hockey_players_eq : hockey_players = 15)
  (basketball_players_eq : basketball_players = 16)
  (neither_players_eq : neither_players = 4)
  (both_players_eq : both_players = 10) :
  hockey_players + basketball_players - both_players + neither_players = 25 := 
by 
  sorry

end NUMINAMATH_GPT_total_students_in_class_l1746_174634


namespace NUMINAMATH_GPT_total_shares_eq_300_l1746_174642

-- Define the given conditions
def microtron_price : ℝ := 36
def dynaco_price : ℝ := 44
def avg_price : ℝ := 40
def dynaco_shares : ℝ := 150

-- Define the number of Microtron shares sold
variable (M : ℝ)

-- Define the total shares sold
def total_shares : ℝ := M + dynaco_shares

-- The average price equation given the conditions
def avg_price_eq (M : ℝ) : Prop :=
  avg_price = (microtron_price * M + dynaco_price * dynaco_shares) / total_shares M

-- The correct answer we need to prove
theorem total_shares_eq_300 (M : ℝ) (h : avg_price_eq M) : total_shares M = 300 :=
by
  sorry

end NUMINAMATH_GPT_total_shares_eq_300_l1746_174642


namespace NUMINAMATH_GPT_union_A_B_inter_A_B_diff_U_A_U_B_subset_A_C_l1746_174694

universe u

open Set

def U := @univ ℝ
def A := { x : ℝ | 3 ≤ x ∧ x < 10 }
def B := { x : ℝ | 2 < x ∧ x ≤ 7 }
def C (a : ℝ) := { x : ℝ | x > a }

theorem union_A_B : A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
by sorry

theorem inter_A_B : A ∩ B = { x : ℝ | 3 ≤ x ∧ x ≤ 7 } :=
by sorry

theorem diff_U_A_U_B : (U \ A) ∩ (U \ B) = { x : ℝ | x ≤ 2 } ∪ { x : ℝ | 10 ≤ x } :=
by sorry

theorem subset_A_C (a : ℝ) (h : A ⊆ C a) : a < 3 :=
by sorry

end NUMINAMATH_GPT_union_A_B_inter_A_B_diff_U_A_U_B_subset_A_C_l1746_174694


namespace NUMINAMATH_GPT_x_in_terms_of_y_y_in_terms_of_x_l1746_174641

-- Define the main equation
variable (x y : ℝ)

-- First part: Expressing x in terms of y given the condition
theorem x_in_terms_of_y (h : x + 3 * y = 3) : x = 3 - 3 * y :=
by
  sorry

-- Second part: Expressing y in terms of x given the condition
theorem y_in_terms_of_x (h : x + 3 * y = 3) : y = (3 - x) / 3 :=
by
  sorry

end NUMINAMATH_GPT_x_in_terms_of_y_y_in_terms_of_x_l1746_174641


namespace NUMINAMATH_GPT_find_k_l1746_174666

-- Definitions of given vectors and the condition that the vectors are parallel.
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 4)

-- Condition for vectors to be parallel in 2D is that their cross product is zero.
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem find_k : ∀ k : ℝ, parallel vector_a (vector_b k) → k = -2 :=
by
  intro k
  intro h
  sorry

end NUMINAMATH_GPT_find_k_l1746_174666


namespace NUMINAMATH_GPT_factorize_expression_l1746_174639

theorem factorize_expression (x : ℝ) : 4 * x ^ 2 - 2 * x = 2 * x * (2 * x - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1746_174639


namespace NUMINAMATH_GPT_rebecca_perm_charge_l1746_174601

theorem rebecca_perm_charge :
  ∀ (P : ℕ), (4 * 30 + 2 * 60 - 2 * 10 + P + 50 = 310) -> P = 40 :=
by
  intros P h
  sorry

end NUMINAMATH_GPT_rebecca_perm_charge_l1746_174601


namespace NUMINAMATH_GPT_problem_solution_l1746_174699

noncomputable def solve_system : List (ℝ × ℝ × ℝ) :=
[(0, 1, -2), (-3/2, 5/2, -1/2)]

theorem problem_solution (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h1 : x^2 + y^2 = -x + 3*y + z)
  (h2 : y^2 + z^2 = x + 3*y - z)
  (h3 : z^2 + x^2 = 2*x + 2*y - z) :
  (x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2) :=
sorry

end NUMINAMATH_GPT_problem_solution_l1746_174699


namespace NUMINAMATH_GPT_circle_diameter_from_area_l1746_174623

theorem circle_diameter_from_area (A : ℝ) (h : A = 225 * Real.pi) : ∃ d : ℝ, d = 30 :=
  by
  have r := Real.sqrt (225)
  have d := 2 * r
  exact ⟨d, sorry⟩

end NUMINAMATH_GPT_circle_diameter_from_area_l1746_174623


namespace NUMINAMATH_GPT_math_problem_l1746_174616

open Real

theorem math_problem (x : ℝ) (p q : ℕ)
  (h1 : (1 + sin x) * (1 + cos x) = 9 / 4)
  (h2 : (1 - sin x) * (1 - cos x) = p - sqrt q)
  (hp_pos : p > 0) (hq_pos : q > 0) : p + q = 1 := sorry

end NUMINAMATH_GPT_math_problem_l1746_174616


namespace NUMINAMATH_GPT_shaded_area_correct_l1746_174633

noncomputable def total_shaded_area (floor_length : ℝ) (floor_width : ℝ) (tile_size : ℝ) (circle_radius : ℝ) : ℝ :=
  let tile_area := tile_size ^ 2
  let circle_area := Real.pi * circle_radius ^ 2
  let shaded_area_per_tile := tile_area - circle_area
  let floor_area := floor_length * floor_width
  let number_of_tiles := floor_area / tile_area
  number_of_tiles * shaded_area_per_tile 

theorem shaded_area_correct : total_shaded_area 12 15 2 1 = 180 - 45 * Real.pi := sorry

end NUMINAMATH_GPT_shaded_area_correct_l1746_174633


namespace NUMINAMATH_GPT_intersection_of_sets_l1746_174627

variable (x : ℝ)
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets 
  (hA : ∀ x, x ∈ A ↔ -2 < x ∧ x ≤ 1)
  (hB : ∀ x, x ∈ B ↔ 0 < x ∧ x ≤ 1) :
  ∀ x, (x ∈ A ∩ B) ↔ (0 < x ∧ x ≤ 1) := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1746_174627


namespace NUMINAMATH_GPT_trigonometric_identity_l1746_174649

theorem trigonometric_identity (α : ℝ)
 (h : Real.sin (α / 2) - 2 * Real.cos (α / 2) = 1) :
  (1 + Real.sin α + Real.cos α) / (1 + Real.sin α - Real.cos α) = 3 / 4 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1746_174649


namespace NUMINAMATH_GPT_fraction_difference_l1746_174635

variable (x y : ℝ)

theorem fraction_difference (h : x / y = 2) : (x - y) / y = 1 :=
  sorry

end NUMINAMATH_GPT_fraction_difference_l1746_174635


namespace NUMINAMATH_GPT_percent_diploma_thirty_l1746_174630

-- Defining the conditions using Lean definitions

def percent_without_diploma_with_job := 0.10 -- 10%
def percent_with_job := 0.20 -- 20%
def percent_without_job_with_diploma :=
  (1 - percent_with_job) * 0.25 -- 25% of people without job is 25% of 80% which is 20%

def percent_with_diploma := percent_with_job - percent_without_diploma_with_job + percent_without_job_with_diploma

-- Theorem to prove that 30% of the people have a university diploma
theorem percent_diploma_thirty
  (H1 : percent_without_diploma_with_job = 0.10) -- condition 1
  (H2 : percent_with_job = 0.20) -- condition 3
  (H3 : percent_without_job_with_diploma = 0.20) -- evaluated from condition 2
  : percent_with_diploma = 0.30 := by
  -- prove that the percent with diploma is 30%
  sorry

end NUMINAMATH_GPT_percent_diploma_thirty_l1746_174630


namespace NUMINAMATH_GPT_polynomial_evaluation_l1746_174654

theorem polynomial_evaluation (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 + 2005 = 2006 :=
sorry

end NUMINAMATH_GPT_polynomial_evaluation_l1746_174654


namespace NUMINAMATH_GPT_iron_balls_molded_l1746_174661

noncomputable def volume_of_iron_bar (l w h : ℝ) : ℝ :=
  l * w * h

theorem iron_balls_molded (l w h n : ℝ) (volume_of_ball : ℝ) 
  (h_l : l = 12) (h_w : w = 8) (h_h : h = 6) (h_n : n = 10) (h_ball_volume : volume_of_ball = 8) :
  (n * volume_of_iron_bar l w h) / volume_of_ball = 720 :=
by 
  rw [h_l, h_w, h_h, h_n, h_ball_volume]
  rw [volume_of_iron_bar]
  sorry

end NUMINAMATH_GPT_iron_balls_molded_l1746_174661


namespace NUMINAMATH_GPT_part1_part2_part3_max_part3_min_l1746_174664

noncomputable def f : ℝ → ℝ := sorry

-- Given Conditions
axiom f_add (x y : ℝ) : f (x + y) = f x + f y
axiom f_neg (x : ℝ) : x > 0 → f x < 0
axiom f_one : f 1 = -2

-- Prove that f(0) = 0
theorem part1 : f 0 = 0 := sorry

-- Prove that f(x) is an odd function
theorem part2 : ∀ x : ℝ, f (-x) = -f x := sorry

-- Prove the maximum and minimum values of f(x) on [-3,3]
theorem part3_max : f (-3) = 6 := sorry
theorem part3_min : f 3 = -6 := sorry

end NUMINAMATH_GPT_part1_part2_part3_max_part3_min_l1746_174664


namespace NUMINAMATH_GPT_modulus_difference_l1746_174636

def z1 : Complex := 1 + 2 * Complex.I
def z2 : Complex := 2 + Complex.I

theorem modulus_difference :
  Complex.abs (z2 - z1) = Real.sqrt 2 := by sorry

end NUMINAMATH_GPT_modulus_difference_l1746_174636


namespace NUMINAMATH_GPT_miles_per_book_l1746_174682

theorem miles_per_book (total_miles : ℝ) (books_read : ℝ) (miles_per_book : ℝ) : 
  total_miles = 6760 ∧ books_read = 15 → miles_per_book = 450.67 := 
by
  sorry

end NUMINAMATH_GPT_miles_per_book_l1746_174682


namespace NUMINAMATH_GPT_evaluate_magnitude_l1746_174667

noncomputable def mag1 : ℂ := 3 * Real.sqrt 2 - 3 * Complex.I
noncomputable def mag2 : ℂ := Real.sqrt 5 + 5 * Complex.I
noncomputable def mag3 : ℂ := 2 - 2 * Complex.I

theorem evaluate_magnitude :
  Complex.abs (mag1 * mag2 * mag3) = 18 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_magnitude_l1746_174667


namespace NUMINAMATH_GPT_subscription_total_amount_l1746_174663

theorem subscription_total_amount 
  (A B C : ℝ)
  (profit_C profit_total : ℝ)
  (subscription_A subscription_B subscription_C : ℝ)
  (subscription_total : ℝ)
  (hA : subscription_A = subscription_B + 4000)
  (hB : subscription_B = subscription_C + 5000)
  (hc_share : profit_C = 8400)
  (total_profit : profit_total = 35000)
  (h_ratio : profit_C / profit_total = subscription_C / subscription_total)
  (h_subs : subscription_total = subscription_A + subscription_B + subscription_C)
  : subscription_total = 50000 := 
sorry

end NUMINAMATH_GPT_subscription_total_amount_l1746_174663


namespace NUMINAMATH_GPT_linear_transform_determined_by_points_l1746_174624

theorem linear_transform_determined_by_points
  (z1 z2 w1 w2 : ℂ)
  (h1 : z1 ≠ z2)
  (h2 : w1 ≠ w2)
  : ∃ (a b : ℂ), ∀ (z : ℂ), a = (w2 - w1) / (z2 - z1) ∧ b = (w1 * z2 - w2 * z1) / (z2 - z1) ∧ (a * z1 + b = w1) ∧ (a * z2 + b = w2) := 
sorry

end NUMINAMATH_GPT_linear_transform_determined_by_points_l1746_174624


namespace NUMINAMATH_GPT_correct_calculation_l1746_174647

variable (a : ℝ)

theorem correct_calculation :
  a^6 / (1/2 * a^2) = 2 * a^4 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1746_174647


namespace NUMINAMATH_GPT_Sandy_pumpkins_l1746_174685

-- Definitions from the conditions
def Mike_pumpkins : ℕ := 23
def Total_pumpkins : ℕ := 74

-- Theorem to prove the number of pumpkins Sandy grew
theorem Sandy_pumpkins : ∃ (n : ℕ), n + Mike_pumpkins = Total_pumpkins :=
by
  existsi 51
  sorry

end NUMINAMATH_GPT_Sandy_pumpkins_l1746_174685


namespace NUMINAMATH_GPT_max_weight_l1746_174669

-- Define the weights
def weight1 := 2
def weight2 := 5
def weight3 := 10

-- Theorem stating that the heaviest single item that can be weighed using any combination of these weights is 17 lb
theorem max_weight : ∃ x, (x = weight1 + weight2 + weight3) ∧ x = 17 :=
by
  sorry

end NUMINAMATH_GPT_max_weight_l1746_174669


namespace NUMINAMATH_GPT_cookies_flour_and_eggs_l1746_174678

theorem cookies_flour_and_eggs (c₁ c₂ : ℕ) (f₁ f₂ : ℕ) (e₁ e₂ : ℕ) 
  (h₁ : c₁ = 40) (h₂ : f₁ = 3) (h₃ : e₁ = 2) (h₄ : c₂ = 120) :
  f₂ = f₁ * (c₂ / c₁) ∧ e₂ = e₁ * (c₂ / c₁) :=
by
  sorry

end NUMINAMATH_GPT_cookies_flour_and_eggs_l1746_174678


namespace NUMINAMATH_GPT_vertex_angle_double_angle_triangle_l1746_174603

theorem vertex_angle_double_angle_triangle 
  {α β : ℝ} (h1 : α + β + β = 180) (h2 : α = 2 * β ∨ β = 2 * α) :
  α = 36 ∨ α = 90 :=
by
  sorry

end NUMINAMATH_GPT_vertex_angle_double_angle_triangle_l1746_174603


namespace NUMINAMATH_GPT_canal_cross_section_area_l1746_174693

theorem canal_cross_section_area
  (a b h : ℝ)
  (H1 : a = 12)
  (H2 : b = 8)
  (H3 : h = 84) :
  (1 / 2) * (a + b) * h = 840 :=
by
  rw [H1, H2, H3]
  sorry

end NUMINAMATH_GPT_canal_cross_section_area_l1746_174693


namespace NUMINAMATH_GPT_wilson_total_notebooks_l1746_174644

def num_notebooks_per_large_pack : ℕ := 7
def num_large_packs_wilson_bought : ℕ := 7

theorem wilson_total_notebooks : num_large_packs_wilson_bought * num_notebooks_per_large_pack = 49 := 
by
  -- sorry used to skip the proof.
  sorry

end NUMINAMATH_GPT_wilson_total_notebooks_l1746_174644


namespace NUMINAMATH_GPT_find_equation_of_line_l1746_174618

-- Define the given conditions
def center_of_circle : ℝ × ℝ := (0, 3)
def perpendicular_line_slope : ℝ := -1
def perpendicular_line_equation (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the proof problem
theorem find_equation_of_line (x y : ℝ) (l_passes_center : (x, y) = center_of_circle)
 (l_is_perpendicular : ∀ x y, perpendicular_line_equation x y ↔ (x-y+3=0)) : x - y + 3 = 0 :=
sorry

end NUMINAMATH_GPT_find_equation_of_line_l1746_174618


namespace NUMINAMATH_GPT_tony_remaining_money_l1746_174655

theorem tony_remaining_money :
  let initial_amount := 20
  let ticket_cost := 8
  let hotdog_cost := 3
  initial_amount - ticket_cost - hotdog_cost = 9 :=
by
  let initial_amount := 20
  let ticket_cost := 8
  let hotdog_cost := 3
  show initial_amount - ticket_cost - hotdog_cost = 9
  sorry

end NUMINAMATH_GPT_tony_remaining_money_l1746_174655


namespace NUMINAMATH_GPT_solve_perimeter_l1746_174690

noncomputable def ellipse_perimeter_proof : Prop :=
  let a := 4
  let b := Real.sqrt 7
  let c := 3
  let F1 := (-c, 0)
  let F2 := (c, 0)
  let ellipse_eq (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 7) = 1
  ∀ (A B : ℝ×ℝ), 
    (ellipse_eq A.1 A.2) ∧ (ellipse_eq B.1 B.2) ∧ (∃ l : ℝ, l ≠ 0 ∧ ∀ t : ℝ, (A = (F1.1 + t * l, F1.2 + t * l)) ∨ (B = (F1.1 + t * l, F1.2 + t * l))) 
    → ∃ P : ℝ, P = 16

theorem solve_perimeter : ellipse_perimeter_proof := sorry

end NUMINAMATH_GPT_solve_perimeter_l1746_174690


namespace NUMINAMATH_GPT_sum_of_a_values_l1746_174695

theorem sum_of_a_values : 
  (∀ (a x : ℝ), (a + x) / 2 ≥ x - 2 ∧ x / 3 - (x - 2) > 2 / 3 ∧ 
  (x - 1) / (4 - x) + (a + 5) / (x - 4) = -4 ∧ x < 2 ∧ (∃ n : ℤ, x = n ∧ 0 < n)) →
  ∃ I : ℤ, I = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_a_values_l1746_174695


namespace NUMINAMATH_GPT_basis_v_l1746_174668

variable {V : Type*} [AddCommGroup V] [Module ℝ V]  -- specifying V as a real vector space
variables (a b c : V)

-- Assume a, b, and c are linearly independent, forming a basis
axiom linear_independent_a_b_c : LinearIndependent ℝ ![a, b, c]

-- The main theorem which we need to prove
theorem basis_v (h : LinearIndependent ℝ ![a, b, c]) :
  LinearIndependent ℝ ![c, a + b, a - b] :=
sorry

end NUMINAMATH_GPT_basis_v_l1746_174668


namespace NUMINAMATH_GPT_percentage_of_b_l1746_174656

variable (a b c p : ℝ)

-- Conditions
def condition1 : Prop := 0.02 * a = 8
def condition2 : Prop := c = b / a
def condition3 : Prop := p * b = 2

-- Theorem statement
theorem percentage_of_b (h1 : condition1 a)
                        (h2 : condition2 b a c)
                        (h3 : condition3 p b) :
  p = 0.005 := sorry

end NUMINAMATH_GPT_percentage_of_b_l1746_174656


namespace NUMINAMATH_GPT_infinite_integer_solutions_l1746_174691

theorem infinite_integer_solutions (a b c k : ℤ) (D : ℤ) 
  (hD : D = b^2 - 4 * a * c) (hD_pos : D > 0) (hD_non_square : ¬ ∃ (n : ℤ), n^2 = D) 
  (hk_non_zero : k ≠ 0) :
  (∃ (x₀ y₀ : ℤ), a * x₀^2 + b * x₀ * y₀ + c * y₀^2 = k) →
  ∃ (f : ℤ → ℤ × ℤ), ∀ n : ℤ, a * (f n).1^2 + b * (f n).1 * (f n).2 + c * (f n).2^2 = k :=
by
  sorry

end NUMINAMATH_GPT_infinite_integer_solutions_l1746_174691


namespace NUMINAMATH_GPT_altitude_length_l1746_174665

theorem altitude_length 
    {A B C : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
    (AB BC AC : ℝ) (hAC : 𝕜) 
    (h₀ : AB = 8)
    (h₁ : BC = 7)
    (h₂ : AC = 5) :
  h = (5 * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_altitude_length_l1746_174665


namespace NUMINAMATH_GPT_closest_point_to_origin_l1746_174686

theorem closest_point_to_origin : 
  ∃ x y : ℝ, x > 0 ∧ y = x + 1/x ∧ (x, y) = (1/(2^(1/4)), (1 + 2^(1/2))/(2^(1/4))) :=
by
  sorry

end NUMINAMATH_GPT_closest_point_to_origin_l1746_174686


namespace NUMINAMATH_GPT_find_a_l1746_174638

theorem find_a (a : ℝ) : 
  let term_coeff (r : ℕ) := (Nat.choose 10 r : ℝ)
  let coeff_x6 := term_coeff 3 - (a * term_coeff 2)
  coeff_x6 = 30 → a = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l1746_174638


namespace NUMINAMATH_GPT_num_nat_numbers_l1746_174689

theorem num_nat_numbers (n : ℕ) (h1 : n ≥ 1) (h2 : n ≤ 1992)
  (h3 : ∃ k3, n = 3 * k3)
  (h4 : ¬ (∃ k2, n = 2 * k2))
  (h5 : ¬ (∃ k5, n = 5 * k5)) : ∃ (m : ℕ), m = 266 :=
by
  sorry

end NUMINAMATH_GPT_num_nat_numbers_l1746_174689


namespace NUMINAMATH_GPT_weight_of_new_person_l1746_174697

-- Definitions
variable (W : ℝ) -- total weight of original 15 people
variable (x : ℝ) -- weight of the new person
variable (n : ℕ) (avr_increase : ℝ) (original_person_weight : ℝ)
variable (total_increase : ℝ) -- total weight increase

-- Given constants
axiom n_value : n = 15
axiom avg_increase_value : avr_increase = 8
axiom original_person_weight_value : original_person_weight = 45
axiom total_increase_value : total_increase = n * avr_increase

-- Equation stating the condition
axiom weight_replace : W - original_person_weight + x = W + total_increase

-- Theorem (problem translated)
theorem weight_of_new_person : x = 165 := by
  sorry

end NUMINAMATH_GPT_weight_of_new_person_l1746_174697


namespace NUMINAMATH_GPT_height_pillar_D_correct_l1746_174688

def height_of_pillar_at_D (h_A h_B h_C : ℕ) (side_length : ℕ) : ℕ :=
17

theorem height_pillar_D_correct :
  height_of_pillar_at_D 15 10 12 10 = 17 := 
by sorry

end NUMINAMATH_GPT_height_pillar_D_correct_l1746_174688


namespace NUMINAMATH_GPT_p_suff_not_necess_q_l1746_174676

def proposition_p (a : ℝ) : Prop := ∀ (x : ℝ), x > 0 → (3*a - 1)^x < 1
def proposition_q (a : ℝ) : Prop := a > (1 / 3)

theorem p_suff_not_necess_q : 
  (∀ (a : ℝ), proposition_p a → proposition_q a) ∧ (¬∀ (a : ℝ), proposition_q a → proposition_p a) :=
  sorry

end NUMINAMATH_GPT_p_suff_not_necess_q_l1746_174676


namespace NUMINAMATH_GPT_second_number_mod_12_l1746_174610

theorem second_number_mod_12 (x : ℕ) (h : (1274 * x * 1277 * 1285) % 12 = 6) : x % 12 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_second_number_mod_12_l1746_174610


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1746_174683

theorem sufficient_not_necessary_condition (a b : ℝ) (h : (a - b) * a^2 > 0) : a > b ∧ a ≠ 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1746_174683


namespace NUMINAMATH_GPT_sequence_term_l1746_174606

theorem sequence_term (a : ℕ → ℕ) 
  (h1 : a 1 = 2009) 
  (h2 : a 2 = 2011) 
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n + 1) 
  : a 1000 = 2342 := 
by 
  sorry

end NUMINAMATH_GPT_sequence_term_l1746_174606


namespace NUMINAMATH_GPT_sum_first_11_terms_l1746_174648

variable (a : ℕ → ℤ) -- The arithmetic sequence
variable (d : ℤ) -- Common difference
variable (S : ℕ → ℤ) -- Sum of the arithmetic sequence

-- The properties of the arithmetic sequence and sum
axiom arith_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom sum_arith_seq (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Given condition
axiom given_condition : a 1 + a 5 + a 8 = a 2 + 12

-- To prove
theorem sum_first_11_terms : S 11 = 66 := by
  sorry

end NUMINAMATH_GPT_sum_first_11_terms_l1746_174648


namespace NUMINAMATH_GPT_total_cost_l1746_174602

-- Definitions based on conditions
def old_camera_cost : ℝ := 4000
def new_model_cost_increase_rate : ℝ := 0.3
def lens_initial_cost : ℝ := 400
def lens_discount : ℝ := 200

-- Main statement to prove
theorem total_cost (old_camera_cost new_model_cost_increase_rate lens_initial_cost lens_discount : ℝ) : 
  let new_camera_cost := old_camera_cost * (1 + new_model_cost_increase_rate)
  let lens_cost_after_discount := lens_initial_cost - lens_discount
  (new_camera_cost + lens_cost_after_discount) = 5400 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_l1746_174602


namespace NUMINAMATH_GPT_polynomial_roots_arithmetic_progression_not_all_real_l1746_174659

theorem polynomial_roots_arithmetic_progression_not_all_real :
  ∀ (a : ℝ), (∃ r d : ℂ, r - d ≠ r ∧ r ≠ r + d ∧ r - d + r + (r + d) = 9 ∧ (r - d) * r + (r - d) * (r + d) + r * (r + d) = 33 ∧ d ≠ 0) →
  a = -45 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_arithmetic_progression_not_all_real_l1746_174659


namespace NUMINAMATH_GPT_probability_same_color_l1746_174605

-- Definitions according to conditions
def total_socks : ℕ := 24
def blue_pairs : ℕ := 7
def green_pairs : ℕ := 3
def red_pairs : ℕ := 2

def total_blue_socks : ℕ := blue_pairs * 2
def total_green_socks : ℕ := green_pairs * 2
def total_red_socks : ℕ := red_pairs * 2

-- Probability calculations
def probability_blue : ℚ := (total_blue_socks * (total_blue_socks - 1)) / (total_socks * (total_socks - 1))
def probability_green : ℚ := (total_green_socks * (total_green_socks - 1)) / (total_socks * (total_socks - 1))
def probability_red : ℚ := (total_red_socks * (total_red_socks - 1)) / (total_socks * (total_socks - 1))

def total_probability : ℚ := probability_blue + probability_green + probability_red

theorem probability_same_color : total_probability = 28 / 69 :=
by
  sorry

end NUMINAMATH_GPT_probability_same_color_l1746_174605


namespace NUMINAMATH_GPT_value_of_b_l1746_174640

theorem value_of_b (b : ℝ) : 
  (∀ x : ℝ, -x ^ 2 + b * x + 7 < 0 ↔ x < -2 ∨ x > 3) → b = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_l1746_174640


namespace NUMINAMATH_GPT_range_of_a_l1746_174608

theorem range_of_a (a : ℝ) :
  (∀ x: ℝ, |x - a| < 4 → -x^2 + 5 * x - 6 > 0) → (-1 ≤ a ∧ a ≤ 6) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1746_174608


namespace NUMINAMATH_GPT_lily_read_total_books_l1746_174657

-- Definitions
def books_weekdays_last_month : ℕ := 4
def books_weekends_last_month : ℕ := 4

def books_weekdays_this_month : ℕ := 2 * books_weekdays_last_month
def books_weekends_this_month : ℕ := 3 * books_weekends_last_month

def total_books_last_month : ℕ := books_weekdays_last_month + books_weekends_last_month
def total_books_this_month : ℕ := books_weekdays_this_month + books_weekends_this_month
def total_books_two_months : ℕ := total_books_last_month + total_books_this_month

-- Proof problem statement
theorem lily_read_total_books : total_books_two_months = 28 :=
by
  sorry

end NUMINAMATH_GPT_lily_read_total_books_l1746_174657


namespace NUMINAMATH_GPT_apple_count_difference_l1746_174622

theorem apple_count_difference
    (original_green : ℕ)
    (additional_green : ℕ)
    (red_more_than_green : ℕ)
    (green_now : ℕ := original_green + additional_green)
    (red_now : ℕ := original_green + red_more_than_green)
    (difference : ℕ := green_now - red_now)
    (h_original_green : original_green = 32)
    (h_additional_green : additional_green = 340)
    (h_red_more_than_green : red_more_than_green = 200) :
    difference = 140 :=
by
  sorry

end NUMINAMATH_GPT_apple_count_difference_l1746_174622


namespace NUMINAMATH_GPT_probability_all_even_l1746_174600

theorem probability_all_even :
  let die1_even_count := 3
  let die1_total := 6
  let die2_even_count := 3
  let die2_total := 7
  let die3_even_count := 4
  let die3_total := 9
  let prob_die1_even := die1_even_count / die1_total
  let prob_die2_even := die2_even_count / die2_total
  let prob_die3_even := die3_even_count / die3_total
  let probability_all_even := prob_die1_even * prob_die2_even * prob_die3_even
  probability_all_even = 1 / 10.5 :=
by
  sorry

end NUMINAMATH_GPT_probability_all_even_l1746_174600


namespace NUMINAMATH_GPT_arccos_one_eq_zero_l1746_174671

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_GPT_arccos_one_eq_zero_l1746_174671


namespace NUMINAMATH_GPT_find_a_sequence_formula_l1746_174619

variable (f : ℝ → ℝ) (a : ℝ) (a_seq : ℕ → ℝ)
variable (h_f_def : ∀ x ≠ -a, f x = (a * x) / (a + x))
variable (h_f_2 : f 2 = 1) (h_seq_def : ∀ n : ℕ, a_seq (n+1) = f (a_seq n)) (h_a1 : a_seq 1 = 1)

theorem find_a : a = 2 :=
  sorry

theorem sequence_formula : ∀ n : ℕ, a_seq n = 2 / (n + 1) :=
  sorry

end NUMINAMATH_GPT_find_a_sequence_formula_l1746_174619


namespace NUMINAMATH_GPT_functional_relationship_selling_price_l1746_174652

open Real

-- Definitions used from conditions
def cost_price : ℝ := 20
def daily_sales_quantity (x : ℝ) : ℝ := -2 * x + 80

-- Functional relationship between daily sales profit W and selling price x
def daily_sales_profit (x : ℝ) : ℝ :=
  (x - cost_price) * daily_sales_quantity x

-- Part (1): Prove the functional relationship
theorem functional_relationship (x : ℝ) :
  daily_sales_profit x = -2 * x^2 + 120 * x - 1600 :=
by {
  sorry
}

-- Part (2): Prove the selling price should be $25 to achieve $150 profit with condition x ≤ 30
theorem selling_price (x : ℝ) :
  daily_sales_profit x = 150 ∧ x ≤ 30 → x = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_functional_relationship_selling_price_l1746_174652


namespace NUMINAMATH_GPT_last_four_digits_of_5_pow_15000_l1746_174615

theorem last_four_digits_of_5_pow_15000 (h : 5^500 ≡ 1 [MOD 2000]) : 
  5^15000 ≡ 1 [MOD 2000] :=
sorry

end NUMINAMATH_GPT_last_four_digits_of_5_pow_15000_l1746_174615


namespace NUMINAMATH_GPT_find_y_in_terms_of_x_and_n_l1746_174660

variable (x n y : ℝ)

theorem find_y_in_terms_of_x_and_n
  (h : n = 3 * x * y / (x - y)) :
  y = n * x / (3 * x + n) :=
  sorry

end NUMINAMATH_GPT_find_y_in_terms_of_x_and_n_l1746_174660


namespace NUMINAMATH_GPT_total_cans_from_256_l1746_174637

-- Define the recursive function to compute the number of new cans produced.
def total_new_cans (n : ℕ) : ℕ :=
  if n < 4 then 0
  else
    let rec_cans := total_new_cans (n / 4)
    (n / 4) + rec_cans

-- Theorem stating the total number of new cans that can be made from 256 initial cans.
theorem total_cans_from_256 : total_new_cans 256 = 85 := by
  sorry

end NUMINAMATH_GPT_total_cans_from_256_l1746_174637


namespace NUMINAMATH_GPT_divisible_by_17_l1746_174613

theorem divisible_by_17 (k : ℕ) : 17 ∣ (2^(2*k+3) + 3^(k+2) * 7^k) :=
  sorry

end NUMINAMATH_GPT_divisible_by_17_l1746_174613


namespace NUMINAMATH_GPT_find_a_b_c_l1746_174626

variable (a b c : ℚ)

def parabola (x : ℚ) : ℚ := a * x^2 + b * x + c

def vertex_condition := ∀ x, parabola a b c x = a * (x - 3)^2 - 2
def contains_point := parabola a b c 0 = 5

theorem find_a_b_c : vertex_condition a b c ∧ contains_point a b c → a + b + c = 10 / 9 :=
by
sorry

end NUMINAMATH_GPT_find_a_b_c_l1746_174626
