import Mathlib

namespace NUMINAMATH_GPT_houses_built_during_boom_l40_4077

theorem houses_built_during_boom :
  let original_houses := 20817
  let current_houses := 118558
  let houses_built := current_houses - original_houses
  houses_built = 97741 := by
  sorry

end NUMINAMATH_GPT_houses_built_during_boom_l40_4077


namespace NUMINAMATH_GPT_solve_inequality_l40_4025

theorem solve_inequality (x : ℝ) :
  abs (x + 3) + abs (2 * x - 1) < 7 ↔ -3 ≤ x ∧ x < 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l40_4025


namespace NUMINAMATH_GPT_friends_attended_l40_4055

theorem friends_attended (total_guests bride_couples groom_couples : ℕ)
                         (bride_guests groom_guests family_guests friends : ℕ)
                         (h1 : total_guests = 300)
                         (h2 : bride_couples = 30)
                         (h3 : groom_couples = 30)
                         (h4 : bride_guests = bride_couples * 2)
                         (h5 : groom_guests = groom_couples * 2)
                         (h6 : family_guests = bride_guests + groom_guests)
                         (h7 : friends = total_guests - family_guests) :
  friends = 180 :=
by sorry

end NUMINAMATH_GPT_friends_attended_l40_4055


namespace NUMINAMATH_GPT_cos_beta_value_cos_2alpha_plus_beta_value_l40_4047

-- Definitions of the conditions
variables (α β : ℝ)
variable (condition1 : 0 < α ∧ α < π / 2)
variable (condition2 : π / 2 < β ∧ β < π)
variable (condition3 : Real.cos (α + π / 4) = 1 / 3)
variable (condition4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3)

-- Proof problem (1)
theorem cos_beta_value :
  ∀ α β, (0 < α ∧ α < π / 2) →
  (π / 2 < β ∧ β < π) →
  (Real.cos (α + π / 4) = 1 / 3) →
  (Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) →
  Real.cos β = - 4 * Real.sqrt 2 / 9 :=
by
  intros α β condition1 condition2 condition3 condition4
  sorry

-- Proof problem (2)
theorem cos_2alpha_plus_beta_value :
  ∀ α β, (0 < α ∧ α < π / 2) →
  (π / 2 < β ∧ β < π) →
  (Real.cos (α + π / 4) = 1 / 3) →
  (Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) →
  Real.cos (2 * α + β) = -1 :=
by
  intros α β condition1 condition2 condition3 condition4
  sorry

end NUMINAMATH_GPT_cos_beta_value_cos_2alpha_plus_beta_value_l40_4047


namespace NUMINAMATH_GPT_inequality_proof_l40_4022

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^2 / (a + b)) + (b^2 / (b + c)) + (c^2 / (c + a)) ≥ (a + b + c) / 2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l40_4022


namespace NUMINAMATH_GPT_inscribed_circle_radius_l40_4059

noncomputable def semiPerimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def areaUsingHeron (a b c : ℝ) : ℝ :=
  let s := semiPerimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def inscribedCircleRadius (a b c : ℝ) : ℝ :=
  let s := semiPerimeter a b c
  let K := areaUsingHeron a b c
  K / s

theorem inscribed_circle_radius : inscribedCircleRadius 26 18 20 = Real.sqrt 31 :=
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l40_4059


namespace NUMINAMATH_GPT_halogens_have_solid_liquid_gas_l40_4039

def at_25C_and_1atm (element : String) : String :=
  match element with
  | "Li" | "Na" | "K" | "Rb" | "Cs" => "solid"
  | "N" => "gas"
  | "P" | "As" | "Sb" | "Bi" => "solid"
  | "O" => "gas"
  | "S" | "Se" | "Te" => "solid"
  | "F" | "Cl" => "gas"
  | "Br" => "liquid"
  | "I" | "At" => "solid"
  | _ => "unknown"

def family_has_solid_liquid_gas (family : List String) : Prop :=
  "solid" ∈ family.map at_25C_and_1atm ∧
  "liquid" ∈ family.map at_25C_and_1atm ∧
  "gas" ∈ family.map at_25C_and_1atm

theorem halogens_have_solid_liquid_gas :
  family_has_solid_liquid_gas ["F", "Cl", "Br", "I", "At"] :=
by
  sorry

end NUMINAMATH_GPT_halogens_have_solid_liquid_gas_l40_4039


namespace NUMINAMATH_GPT_range_of_k_l40_4086

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x^2 + 2 * x - 1 = 0) ↔ (k ≥ 0 ∧ k ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l40_4086


namespace NUMINAMATH_GPT_problem_solution_l40_4020

-- Definitions and assumptions
variables (priceA priceB : ℕ)
variables (numBooksA numBooksB totalBooks : ℕ)
variables (costPriceA : priceA = 45)
variables (costPriceB : priceB = 65)
variables (totalCost : priceA * numBooksA + priceB * numBooksB ≤ 3550)
variables (totalBooksEq : numBooksA + numBooksB = 70)

-- Proof problem
theorem problem_solution :
  priceA = 45 ∧ priceB = 65 ∧ ∃ (numBooksA : ℕ), numBooksA ≥ 50 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l40_4020


namespace NUMINAMATH_GPT_sum_of_integers_l40_4095

theorem sum_of_integers (x y : ℤ) (h_pos : 0 < y) (h_gt : x > y) (h_diff : x - y = 14) (h_prod : x * y = 48) : x + y = 20 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_l40_4095


namespace NUMINAMATH_GPT_Q_over_P_l40_4032

theorem Q_over_P :
  (∀ (x : ℝ), x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 6 → 
    (P / (x + 6) + Q / (x^2 - 6*x) = (x^2 - 3*x + 12) / (x^3 + x^2 - 24*x))) →
  Q / P = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_Q_over_P_l40_4032


namespace NUMINAMATH_GPT_domain_of_tan_sub_pi_over_4_l40_4015

theorem domain_of_tan_sub_pi_over_4 :
  ∀ x : ℝ, (∃ k : ℤ, x = k * π + 3 * π / 4) ↔ ∃ y : ℝ, y = (x - π / 4) ∧ (∃ k : ℤ, y = (2 * k + 1) * π / 2) := 
sorry

end NUMINAMATH_GPT_domain_of_tan_sub_pi_over_4_l40_4015


namespace NUMINAMATH_GPT_probability_at_least_three_prime_dice_l40_4076

-- Definitions from the conditions
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def p := 5 / 12
def q := 7 / 12
def binomial (n k : ℕ) := Nat.choose n k

-- The probability of at least three primes
theorem probability_at_least_three_prime_dice :
  (binomial 5 3 * p ^ 3 * q ^ 2) +
  (binomial 5 4 * p ^ 4 * q ^ 1) +
  (binomial 5 5 * p ^ 5 * q ^ 0) = 40625 / 622080 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_three_prime_dice_l40_4076


namespace NUMINAMATH_GPT_f_constant_1_l40_4006

theorem f_constant_1 (f : ℕ → ℕ) (h1 : ∀ n : ℕ, 0 < n → f (n + f n) = f n)
  (h2 : ∃ n0 : ℕ, 0 < n0 ∧ f n0 = 1) : ∀ n : ℕ, f n = 1 := 
by
  sorry

end NUMINAMATH_GPT_f_constant_1_l40_4006


namespace NUMINAMATH_GPT_find_radius_of_circle_B_l40_4044

noncomputable def radius_of_circle_B : Real :=
  sorry

theorem find_radius_of_circle_B :
  let A := 2
  let R := 4
  -- Define x as the horizontal distance (FG) and y as the vertical distance (GH)
  ∃ (x y : Real), 
  (y = x + (x^2 / 2)) ∧
  (y = 2 - (x^2 / 4)) ∧
  (5 * x^2 + 4 * x - 8 = 0) ∧
  -- Contains only the positive solution among possible valid radii
  (radius_of_circle_B = (22 / 25) + (2 * Real.sqrt 11 / 25))
:= 
sorry

end NUMINAMATH_GPT_find_radius_of_circle_B_l40_4044


namespace NUMINAMATH_GPT_bob_distance_when_they_meet_l40_4034

-- Define the conditions
def distance_XY : ℝ := 10
def yolanda_rate : ℝ := 3
def bob_rate : ℝ := 4
def yolanda_start_time : ℝ := 0
def bob_start_time : ℝ := 1

-- The statement we want to prove
theorem bob_distance_when_they_meet : 
  ∃ t : ℝ, (yolanda_rate * (t + 1) + bob_rate * t = distance_XY) ∧ (bob_rate * t = 4) :=
sorry

end NUMINAMATH_GPT_bob_distance_when_they_meet_l40_4034


namespace NUMINAMATH_GPT_inequality_am_gm_l40_4060

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a^3 / (b * c) + b^3 / (c * a) + c^3 / (a * b) ≥ a + b + c :=
by {
    sorry
}

end NUMINAMATH_GPT_inequality_am_gm_l40_4060


namespace NUMINAMATH_GPT_halfway_between_one_sixth_and_one_twelfth_is_one_eighth_l40_4068

theorem halfway_between_one_sixth_and_one_twelfth_is_one_eighth : 
  (1 / 6 + 1 / 12) / 2 = 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_halfway_between_one_sixth_and_one_twelfth_is_one_eighth_l40_4068


namespace NUMINAMATH_GPT_f_leq_binom_l40_4052

-- Define the function f with given conditions
def f (m n : ℕ) : ℕ := if m = 1 ∨ n = 1 then 1 else sorry

-- State the property to be proven
theorem f_leq_binom (m n : ℕ) (h2 : 2 ≤ m) (h2' : 2 ≤ n) :
  f m n ≤ Nat.choose (m + n) n := 
sorry

end NUMINAMATH_GPT_f_leq_binom_l40_4052


namespace NUMINAMATH_GPT_chameleons_all_white_l40_4099

theorem chameleons_all_white :
  ∀ (a b c : ℕ), a = 800 → b = 1000 → c = 1220 → 
  (a + b + c = 3020) → (a % 3 = 2) → (b % 3 = 1) → (c % 3 = 2) →
    ∃ k : ℕ, (k = 3020 ∧ (k % 3 = 1)) ∧ 
    (if k = b then a = 0 ∧ c = 0 else false) :=
by
  sorry

end NUMINAMATH_GPT_chameleons_all_white_l40_4099


namespace NUMINAMATH_GPT_find_value_of_E_l40_4066

variables (Q U I E T Z : ℤ)

theorem find_value_of_E (hZ : Z = 15) (hQUIZ : Q + U + I + Z = 60) (hQUIET : Q + U + I + E + T = 75) (hQUIT : Q + U + I + T = 50) : E = 25 :=
by
  have hQUIZ_val : Q + U + I = 45 := by linarith [hZ, hQUIZ]
  have hQUIET_val : E + T = 30 := by linarith [hQUIZ_val, hQUIET]
  have hQUIT_val : T = 5 := by linarith [hQUIZ_val, hQUIT]
  linarith [hQUIET_val, hQUIT_val]

end NUMINAMATH_GPT_find_value_of_E_l40_4066


namespace NUMINAMATH_GPT_total_students_in_class_l40_4012

def number_of_girls := 9
def number_of_boys := 16
def total_students := number_of_girls + number_of_boys

theorem total_students_in_class : total_students = 25 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_total_students_in_class_l40_4012


namespace NUMINAMATH_GPT_minutes_spent_calling_clients_l40_4037

theorem minutes_spent_calling_clients
    (C : ℕ)
    (H1 : 7 * C + C = 560) :
    C = 70 :=
sorry

end NUMINAMATH_GPT_minutes_spent_calling_clients_l40_4037


namespace NUMINAMATH_GPT_green_more_than_blue_l40_4081

theorem green_more_than_blue (B Y G : Nat) (h1 : B + Y + G = 108) (h2 : B * 7 = Y * 3) (h3 : B * 8 = G * 3) : G - B = 30 := by
  sorry

end NUMINAMATH_GPT_green_more_than_blue_l40_4081


namespace NUMINAMATH_GPT_first_term_of_geometric_sequence_l40_4070

theorem first_term_of_geometric_sequence (a r : ℕ) :
  (a * r ^ 3 = 54) ∧ (a * r ^ 4 = 162) → a = 2 :=
by
  -- Provided conditions and the goal
  sorry

end NUMINAMATH_GPT_first_term_of_geometric_sequence_l40_4070


namespace NUMINAMATH_GPT_coefficient_of_y_in_first_equation_is_minus_1_l40_4010

variable (x y z : ℝ)

def equation1 : Prop := 6 * x - y + 3 * z = 22 / 5
def equation2 : Prop := 4 * x + 8 * y - 11 * z = 7
def equation3 : Prop := 5 * x - 6 * y + 2 * z = 12
def sum_xyz : Prop := x + y + z = 10

theorem coefficient_of_y_in_first_equation_is_minus_1 :
  equation1 x y z → equation2 x y z → equation3 x y z → sum_xyz x y z → (-1 : ℝ) = -1 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_y_in_first_equation_is_minus_1_l40_4010


namespace NUMINAMATH_GPT_ram_account_balance_first_year_l40_4042

theorem ram_account_balance_first_year :
  let initial_deposit := 1000
  let interest_first_year := 100
  initial_deposit + interest_first_year = 1100 :=
by
  sorry

end NUMINAMATH_GPT_ram_account_balance_first_year_l40_4042


namespace NUMINAMATH_GPT_intersection_of_sets_l40_4026

def SetA : Set ℝ := { x | |x| ≤ 1 }
def SetB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_of_sets : (SetA ∩ SetB) = { x | 0 ≤ x ∧ x ≤ 1 } := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l40_4026


namespace NUMINAMATH_GPT_intersection_points_count_l40_4056

theorem intersection_points_count (B : ℝ) (hB : 0 < B) :
  ∃ p : ℕ, p = 4 ∧ (∀ x y : ℝ, (y = B * x^2 ∧ y^2 + 4 * y - 2 = x^2 + 5 * y) ↔ p = 4) := by
sorry

end NUMINAMATH_GPT_intersection_points_count_l40_4056


namespace NUMINAMATH_GPT_value_of_m_l40_4058

theorem value_of_m (x m : ℝ) (h_positive_root : x > 0) (h_eq : x / (x - 1) - m / (1 - x) = 2) : m = -1 := by
  sorry

end NUMINAMATH_GPT_value_of_m_l40_4058


namespace NUMINAMATH_GPT_tan_identity_15_eq_sqrt3_l40_4035

theorem tan_identity_15_eq_sqrt3 :
  (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_identity_15_eq_sqrt3_l40_4035


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l40_4046

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x + 3 ≤ x + 2) ∧ ((x + 1) / 3 > x - 1) → x ≤ -1 := by
  sorry

end NUMINAMATH_GPT_solve_system_of_inequalities_l40_4046


namespace NUMINAMATH_GPT_sin_square_eq_c_div_a2_plus_b2_l40_4094

theorem sin_square_eq_c_div_a2_plus_b2 
  (a b c : ℝ) (α β : ℝ)
  (h1 : a * Real.cos α + b * Real.sin α = c)
  (h2 : a * Real.cos β + b * Real.sin β = 0)
  (not_all_zero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  Real.sin (α - β) ^ 2 = c ^ 2 / (a ^ 2 + b ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_square_eq_c_div_a2_plus_b2_l40_4094


namespace NUMINAMATH_GPT_find_m_over_n_l40_4050

variable (a b : ℝ × ℝ)
variable (m n : ℝ)
variable (n_nonzero : n ≠ 0)

axiom a_def : a = (1, 2)
axiom b_def : b = (-2, 3)
axiom collinear : ∃ k : ℝ, m • a - n • b = k • (a + 2 • b)

theorem find_m_over_n : m / n = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_find_m_over_n_l40_4050


namespace NUMINAMATH_GPT_product_of_largest_integer_digits_l40_4014

theorem product_of_largest_integer_digits (u v : ℕ) :
  u^2 + v^2 = 45 ∧ u < v → u * v = 18 :=
sorry

end NUMINAMATH_GPT_product_of_largest_integer_digits_l40_4014


namespace NUMINAMATH_GPT_probability_sum_of_two_dice_is_4_l40_4063

noncomputable def fair_dice_probability_sum_4 : ℚ :=
  let total_outcomes := 6 * 6 -- Total outcomes for two dice
  let favorable_outcomes := 3 -- Outcomes that sum to 4: (1, 3), (3, 1), (2, 2)
  favorable_outcomes / total_outcomes

theorem probability_sum_of_two_dice_is_4 : fair_dice_probability_sum_4 = 1 / 12 := 
by
  sorry

end NUMINAMATH_GPT_probability_sum_of_two_dice_is_4_l40_4063


namespace NUMINAMATH_GPT_smallest_positive_integer_n_l40_4019

theorem smallest_positive_integer_n (n : ℕ) (h : n > 0) : 3^n ≡ n^3 [MOD 5] ↔ n = 3 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_n_l40_4019


namespace NUMINAMATH_GPT_parallel_lines_k_value_l40_4031

-- Define the lines and the condition of parallelism
def line1 (x y : ℝ) := x + 2 * y - 1 = 0
def line2 (k x y : ℝ) := k * x - y = 0

-- Define the parallelism condition
def lines_parallel (k : ℝ) := (1 / k) = (2 / -1)

-- Prove that given the parallelism condition, k equals -1/2
theorem parallel_lines_k_value (k : ℝ) (h : lines_parallel k) : k = (-1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_k_value_l40_4031


namespace NUMINAMATH_GPT_sum_of_distinct_prime_factors_of_number_is_10_l40_4054

-- Define the constant number 9720
def number : ℕ := 9720

-- Define the distinct prime factors of 9720
def distinct_prime_factors_of_number : List ℕ := [2, 3, 5]

-- Sum function for the list of distinct prime factors
def sum_of_distinct_prime_factors (lst : List ℕ) : ℕ :=
  lst.foldr (.+.) 0

-- The main theorem to prove
theorem sum_of_distinct_prime_factors_of_number_is_10 :
  sum_of_distinct_prime_factors distinct_prime_factors_of_number = 10 := by
  sorry

end NUMINAMATH_GPT_sum_of_distinct_prime_factors_of_number_is_10_l40_4054


namespace NUMINAMATH_GPT_tan_half_sum_eq_third_l40_4029

theorem tan_half_sum_eq_third
  (x y : ℝ)
  (h1 : Real.cos x + Real.cos y = 3/5)
  (h2 : Real.sin x + Real.sin y = 1/5) :
  Real.tan ((x + y) / 2) = 1/3 :=
by sorry

end NUMINAMATH_GPT_tan_half_sum_eq_third_l40_4029


namespace NUMINAMATH_GPT_find_m_n_l40_4021

theorem find_m_n (m n : ℕ) (positive_m : 0 < m) (positive_n : 0 < n)
  (h1 : m = 3) (h2 : n = 4) :
    Real.arctan (1 / 3) + Real.arctan (1 / 4) + Real.arctan (1 / m) + Real.arctan (1 / n) = π / 2 :=
  by 
    -- Placeholder for the proof
    sorry

end NUMINAMATH_GPT_find_m_n_l40_4021


namespace NUMINAMATH_GPT_total_trees_after_planting_l40_4001

-- Definitions based on conditions
def initial_trees : ℕ := 34
def trees_to_plant : ℕ := 49

-- Statement to prove the total number of trees after planting
theorem total_trees_after_planting : initial_trees + trees_to_plant = 83 := 
by 
  sorry

end NUMINAMATH_GPT_total_trees_after_planting_l40_4001


namespace NUMINAMATH_GPT_code_word_MEET_l40_4093

def translate_GREAT_TIME : String → ℕ 
| "G" => 0
| "R" => 1
| "E" => 2
| "A" => 3
| "T" => 4
| "I" => 5
| "M" => 6
| _   => 0 -- Default case for simplicity, not strictly necessary

theorem code_word_MEET : translate_GREAT_TIME "M" = 6 ∧ translate_GREAT_TIME "E" = 2 ∧ translate_GREAT_TIME "T" = 4 →
  let MEET : ℕ := (translate_GREAT_TIME "M" * 1000) + 
                  (translate_GREAT_TIME "E" * 100) + 
                  (translate_GREAT_TIME "E" * 10) + 
                  (translate_GREAT_TIME "T")
  MEET = 6224 :=
sorry

end NUMINAMATH_GPT_code_word_MEET_l40_4093


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l40_4051

variable (x : ℝ)

def p : Prop := x > 0
def q : Prop := x > -1

theorem p_sufficient_not_necessary_for_q : (p x → q x) ∧ ¬ (q x → p x) :=
by
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l40_4051


namespace NUMINAMATH_GPT_max_alligators_in_days_l40_4041

noncomputable def days := 616
noncomputable def weeks := 88  -- derived from 616 / 7
noncomputable def alligators_per_week := 1

theorem max_alligators_in_days
  (h1 : weeks = days / 7)
  (h2 : ∀ (w : ℕ), alligators_per_week = 1) :
  weeks * alligators_per_week = 88 := by
  sorry

end NUMINAMATH_GPT_max_alligators_in_days_l40_4041


namespace NUMINAMATH_GPT_solve_for_t_l40_4004

theorem solve_for_t (s t : ℚ) (h1 : 8 * s + 6 * t = 160) (h2 : s = t + 3) : t = 68 / 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_t_l40_4004


namespace NUMINAMATH_GPT_persistence_of_2_persistence_iff_2_l40_4028

def is_persistent (T : ℝ) : Prop :=
  ∀ (a b c d : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
                    a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ d ≠ 1) →
    (a + b + c + d = T) →
    (1 / a + 1 / b + 1 / c + 1 / d = T) →
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) + 1 / (1 - d) = T)

theorem persistence_of_2 : is_persistent 2 :=
by
  -- The proof is omitted as per instructions
  sorry

theorem persistence_iff_2 (T : ℝ) : is_persistent T ↔ T = 2 :=
by
  -- The proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_persistence_of_2_persistence_iff_2_l40_4028


namespace NUMINAMATH_GPT_num_of_int_solutions_l40_4033

/-- 
  The number of integer solutions to the equation 
  \((x^3 - x - 1)^{2015} = 1\) is 3.
-/
theorem num_of_int_solutions :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℤ, (x ^ 3 - x - 1) ^ 2015 = 1 ↔ x = 0 ∨ x = 1 ∨ x = -1 := 
sorry

end NUMINAMATH_GPT_num_of_int_solutions_l40_4033


namespace NUMINAMATH_GPT_area_of_lune_l40_4005

theorem area_of_lune :
  let d1 := 2
  let d2 := 4
  let r1 := d1 / 2
  let r2 := d2 / 2
  let height := r2 - r1
  let area_triangle := (1 / 2) * d1 * height
  let area_semicircle_small := (1 / 2) * π * r1^2
  let area_combined := area_triangle + area_semicircle_small
  let area_sector_large := (1 / 4) * π * r2^2
  let area_lune := area_combined - area_sector_large
  area_lune = 1 - (1 / 2) * π := 
by
  sorry

end NUMINAMATH_GPT_area_of_lune_l40_4005


namespace NUMINAMATH_GPT_value_of_f_at_2019_l40_4083

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_positive : ∀ x : ℝ, f x > 0)
variable (h_functional : ∀ x : ℝ, f (x + 2) = 1 / (f x))

theorem value_of_f_at_2019 : f 2019 = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_at_2019_l40_4083


namespace NUMINAMATH_GPT_fraction_of_income_from_tips_l40_4082

variable (S T I : ℝ)

/- Definition of the conditions -/
def tips_condition : Prop := T = (3 / 4) * S
def income_condition : Prop := I = S + T

/- The proof problem statement, asserting the desired result -/
theorem fraction_of_income_from_tips (h1 : tips_condition S T) (h2 : income_condition S T I) : T / I = 3 / 7 := by
  sorry

end NUMINAMATH_GPT_fraction_of_income_from_tips_l40_4082


namespace NUMINAMATH_GPT_calculate_fraction_l40_4088

theorem calculate_fraction : (1 / (1 + 1 / (4 + 1 / 5))) = (21 / 26) :=
by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l40_4088


namespace NUMINAMATH_GPT_complement_of_M_with_respect_to_U_l40_4061

noncomputable def U : Set ℕ := {1, 2, 3, 4}
noncomputable def M : Set ℕ := {1, 2, 3}

theorem complement_of_M_with_respect_to_U :
  (U \ M) = {4} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_with_respect_to_U_l40_4061


namespace NUMINAMATH_GPT_jack_finishes_in_16_days_l40_4084

noncomputable def pages_in_book : ℕ := 285
noncomputable def weekday_reading_rate : ℕ := 23
noncomputable def weekend_reading_rate : ℕ := 35
noncomputable def weekdays_per_week : ℕ := 5
noncomputable def weekends_per_week : ℕ := 2
noncomputable def weekday_skipped : ℕ := 1
noncomputable def weekend_skipped : ℕ := 1

noncomputable def pages_per_week : ℕ :=
  (weekdays_per_week - weekday_skipped) * weekday_reading_rate + 
  (weekends_per_week - weekend_skipped) * weekend_reading_rate

noncomputable def weeks_needed : ℕ :=
  pages_in_book / pages_per_week

noncomputable def pages_left_after_weeks : ℕ :=
  pages_in_book % pages_per_week

noncomputable def extra_days_needed (pages_left : ℕ) : ℕ :=
  if pages_left > weekend_reading_rate then 2
  else if pages_left > weekday_reading_rate then 2
  else 1

noncomputable def total_days_needed : ℕ :=
  weeks_needed * 7 + extra_days_needed (pages_left_after_weeks)

theorem jack_finishes_in_16_days : total_days_needed = 16 := by
  sorry

end NUMINAMATH_GPT_jack_finishes_in_16_days_l40_4084


namespace NUMINAMATH_GPT_absolute_value_of_slope_l40_4045

noncomputable def circle_center1 : ℝ × ℝ := (14, 92)
noncomputable def circle_center2 : ℝ × ℝ := (17, 76)
noncomputable def circle_center3 : ℝ × ℝ := (19, 84)
noncomputable def radius : ℝ := 3
noncomputable def point_on_line : ℝ × ℝ := (17, 76)

theorem absolute_value_of_slope :
  ∃ m : ℝ, ∀ line : ℝ × ℝ → Prop,
    (line point_on_line) ∧ 
    (∀ p, (line p) → true) → 
    abs m = 24 := 
  sorry

end NUMINAMATH_GPT_absolute_value_of_slope_l40_4045


namespace NUMINAMATH_GPT_Cody_total_bill_l40_4078

-- Definitions for the problem
def cost_per_child : ℝ := 7.5
def cost_per_adult : ℝ := 12.0

variables (A C : ℕ)

-- Conditions
def condition1 : Prop := C = A + 8
def condition2 : Prop := A + C = 12

-- Total bill
def total_cost := (A * cost_per_adult) + (C * cost_per_child)

-- The proof statement
theorem Cody_total_bill (h1 : condition1 A C) (h2 : condition2 A C) : total_cost A C = 99.0 := by
  sorry

end NUMINAMATH_GPT_Cody_total_bill_l40_4078


namespace NUMINAMATH_GPT_triangle_area_l40_4079

theorem triangle_area (x : ℝ) (h1 : 6 * x = 6) (h2 : 8 * x = 8) (h3 : 10 * x = 2 * 5) : 
  1 / 2 * 6 * 8 = 24 := 
sorry

end NUMINAMATH_GPT_triangle_area_l40_4079


namespace NUMINAMATH_GPT_midpoint_of_segment_l40_4018

theorem midpoint_of_segment :
  let p1 := (12, -8)
  let p2 := (-4, 10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint = (4, 1) :=
by
  let p1 := (12, -8)
  let p2 := (-4, 10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  show midpoint = (4, 1)
  sorry

end NUMINAMATH_GPT_midpoint_of_segment_l40_4018


namespace NUMINAMATH_GPT_avg_starting_with_d_l40_4090

-- Define c and d as positive integers
variables (c d : ℤ) (hc : c > 0) (hd : d > 0)

-- Define d as the average of the seven consecutive integers starting with c
def avg_starting_with_c (c : ℤ) : ℤ := (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7

-- Define the condition that d is the average of the seven consecutive integers starting with c
axiom d_is_avg_starting_with_c : d = avg_starting_with_c c

-- Prove that the average of the seven consecutive integers starting with d equals c + 6
theorem avg_starting_with_d (c d : ℤ) (hc : c > 0) (hd : d > 0) (h : d = avg_starting_with_c c) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 6 := by
  sorry

end NUMINAMATH_GPT_avg_starting_with_d_l40_4090


namespace NUMINAMATH_GPT_find_k_square_binomial_l40_4038

theorem find_k_square_binomial (k : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 - 16 * x + k = (x + b)^2) ↔ k = 64 :=
by
  sorry

end NUMINAMATH_GPT_find_k_square_binomial_l40_4038


namespace NUMINAMATH_GPT_solve_system_l40_4092

def system_of_equations_solution : Prop :=
  ∃ (x y : ℚ), 4 * x - 7 * y = -9 ∧ 5 * x + 3 * y = -11 ∧ (x, y) = (-(104 : ℚ) / 47, (1 : ℚ) / 47)

theorem solve_system : system_of_equations_solution :=
sorry

end NUMINAMATH_GPT_solve_system_l40_4092


namespace NUMINAMATH_GPT_train_speed_is_correct_l40_4065

-- Definitions based on the conditions
def length_of_train : ℝ := 120       -- Train is 120 meters long
def time_to_cross : ℝ := 16          -- The train takes 16 seconds to cross the post

-- Conversion constants
def seconds_to_hours : ℝ := 3600
def meters_to_kilometers : ℝ := 1000

-- The speed of the train in km/h
noncomputable def speed_of_train (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (seconds_to_hours / meters_to_kilometers)

-- Theorem: The speed of the train is 27 km/h
theorem train_speed_is_correct : speed_of_train length_of_train time_to_cross = 27 :=
by
  -- This is where the proof should be, but we leave it as sorry as instructed
  sorry

end NUMINAMATH_GPT_train_speed_is_correct_l40_4065


namespace NUMINAMATH_GPT_find_x2_times_sum_roots_l40_4030

noncomputable def sqrt2015 := Real.sqrt 2015

theorem find_x2_times_sum_roots
  (x1 x2 x3 : ℝ)
  (h_eq : ∀ x : ℝ, sqrt2015 * x^3 - 4030 * x^2 + 2 = 0 → x = x1 ∨ x = x2 ∨ x = x3)
  (h_ineq : x1 < x2 ∧ x2 < x3) :
  x2 * (x1 + x3) = 2 := by
  sorry

end NUMINAMATH_GPT_find_x2_times_sum_roots_l40_4030


namespace NUMINAMATH_GPT_compute_series_sum_l40_4024

noncomputable def term (n : ℕ) : ℝ := (5 * n - 2) / (3 ^ n)

theorem compute_series_sum : 
  ∑' n, term n = 11 / 4 := 
sorry

end NUMINAMATH_GPT_compute_series_sum_l40_4024


namespace NUMINAMATH_GPT_mary_regular_hours_l40_4008

theorem mary_regular_hours (x y : ℕ) (h1 : 8 * x + 10 * y = 560) (h2 : x + y = 60) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_mary_regular_hours_l40_4008


namespace NUMINAMATH_GPT_maximum_cars_quotient_l40_4002

theorem maximum_cars_quotient
  (car_length : ℕ) (m_speed : ℕ) (half_hour_distance : ℕ) 
  (unit_length : ℕ) (max_units : ℕ) (N : ℕ) :
  (car_length = 5) →
  (half_hour_distance = 10000) →
  (unit_length = 5 * (m_speed + 1)) →
  (max_units = half_hour_distance / unit_length) →
  (N = max_units) →
  (N / 10 = 200) :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_maximum_cars_quotient_l40_4002


namespace NUMINAMATH_GPT_find_integer_n_l40_4057

theorem find_integer_n :
  ∃ n : ℕ, 0 ≤ n ∧ n < 201 ∧ 200 * n ≡ 144 [MOD 101] ∧ n = 29 := 
by
  sorry

end NUMINAMATH_GPT_find_integer_n_l40_4057


namespace NUMINAMATH_GPT_quadratic_b_value_l40_4040

theorem quadratic_b_value (b : ℝ) (n : ℝ) (h_b_neg : b < 0) 
  (h_equiv : ∀ x : ℝ, (x + n)^2 + 1 / 16 = x^2 + b * x + 1 / 4) : 
  b = - (Real.sqrt 3) / 2 := 
sorry

end NUMINAMATH_GPT_quadratic_b_value_l40_4040


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l40_4069

-- Define the condition: all internal angles of the triangle are less than 60 degrees.
def condition (α β γ : ℝ) (h: α + β + γ = 180): Prop :=
  α < 60 ∧ β < 60 ∧ γ < 60

-- The proof statement
theorem triangle_angle_contradiction (α β γ : ℝ) (h_sum : α + β + γ = 180) (h: condition α β γ h_sum) : false :=
sorry

end NUMINAMATH_GPT_triangle_angle_contradiction_l40_4069


namespace NUMINAMATH_GPT_solution_set_inequality_l40_4043

theorem solution_set_inequality (m : ℝ) (h : 3 - m < 0) :
  { x : ℝ | (2 - m) * x + 2 > m } = { x : ℝ | x < -1 } :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l40_4043


namespace NUMINAMATH_GPT_candy_profit_l40_4011

theorem candy_profit :
  let num_bars := 800
  let cost_per_4_bars := 3
  let sell_per_3_bars := 2
  let cost_price := (cost_per_4_bars / 4) * num_bars
  let sell_price := (sell_per_3_bars / 3) * num_bars
  let profit := sell_price - cost_price
  profit = -66.67 :=
by
  sorry

end NUMINAMATH_GPT_candy_profit_l40_4011


namespace NUMINAMATH_GPT_lcm_of_18_and_36_l40_4073

theorem lcm_of_18_and_36 : Nat.lcm 18 36 = 36 := 
by 
  sorry

end NUMINAMATH_GPT_lcm_of_18_and_36_l40_4073


namespace NUMINAMATH_GPT_find_k_l40_4062

-- The function that computes the sum of the digits for the known form of the product (9 * 999...9) with k digits.
def sum_of_digits (k : ℕ) : ℕ :=
  8 + 9 * (k - 1) + 1

theorem find_k (k : ℕ) : sum_of_digits k = 2000 ↔ k = 222 := by
  sorry

end NUMINAMATH_GPT_find_k_l40_4062


namespace NUMINAMATH_GPT_sum_a_b_eq_neg2_l40_4048

theorem sum_a_b_eq_neg2 (a b : ℝ) (h : (a - 2)^2 + |b + 4| = 0) : a + b = -2 := 
by 
  sorry

end NUMINAMATH_GPT_sum_a_b_eq_neg2_l40_4048


namespace NUMINAMATH_GPT_fourth_proportional_segment_l40_4085

theorem fourth_proportional_segment 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  : ∃ x : ℝ, x = (b * c) / a := 
by
  sorry

end NUMINAMATH_GPT_fourth_proportional_segment_l40_4085


namespace NUMINAMATH_GPT_sum_of_positive_integer_factors_of_24_l40_4089

-- Define the number 24
def n : ℕ := 24

-- Define the list of positive factors of 24
def pos_factors_of_24 : List ℕ := [1, 2, 4, 8, 3, 6, 12, 24]

-- Define the sum of the factors
def sum_of_factors : ℕ := pos_factors_of_24.sum

-- The theorem statement
theorem sum_of_positive_integer_factors_of_24 : sum_of_factors = 60 := by
  sorry

end NUMINAMATH_GPT_sum_of_positive_integer_factors_of_24_l40_4089


namespace NUMINAMATH_GPT_Louie_monthly_payment_l40_4003

noncomputable def monthly_payment (P : ℕ) (r : ℚ) (n t : ℕ) : ℚ :=
  (P : ℚ) * (1 + r / n)^(n * t) / t

theorem Louie_monthly_payment : 
  monthly_payment 2000 0.10 1 3 = 887 := 
by
  sorry

end NUMINAMATH_GPT_Louie_monthly_payment_l40_4003


namespace NUMINAMATH_GPT_smallest_int_rel_prime_150_l40_4087

theorem smallest_int_rel_prime_150 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 150 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 150 = 1) → x ≤ y :=
by
  sorry

end NUMINAMATH_GPT_smallest_int_rel_prime_150_l40_4087


namespace NUMINAMATH_GPT_min_value_fraction_l40_4096

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_ln : Real.log (a + b) = 0) :
  (2 / a + 3 / b) = 5 + 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l40_4096


namespace NUMINAMATH_GPT_courtyard_paving_l40_4075

noncomputable def length_of_brick (L : ℕ) := L = 12

theorem courtyard_paving  (courtyard_length : ℕ) (courtyard_width : ℕ) 
                           (brick_width : ℕ) (total_bricks : ℕ) 
                           (H1 : courtyard_length = 18) (H2 : courtyard_width = 12) 
                           (H3 : brick_width = 6) (H4 : total_bricks = 30000) 
                           : length_of_brick 12 := 
by 
  sorry

end NUMINAMATH_GPT_courtyard_paving_l40_4075


namespace NUMINAMATH_GPT_virginia_eggs_l40_4036

theorem virginia_eggs (initial_eggs : ℕ) (taken_eggs : ℕ) (result_eggs : ℕ) 
  (h_initial : initial_eggs = 200) 
  (h_taken : taken_eggs = 37) 
  (h_calculation: result_eggs = initial_eggs - taken_eggs) :
result_eggs = 163 :=
by {
  sorry
}

end NUMINAMATH_GPT_virginia_eggs_l40_4036


namespace NUMINAMATH_GPT_value_a_plus_c_l40_4091

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c
noncomputable def g (a b c : ℝ) (x : ℝ) := c * x^2 + b * x + a

theorem value_a_plus_c (a b c : ℝ) (h : ∀ x : ℝ, f a b c (g a b c x) = x) : a + c = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_a_plus_c_l40_4091


namespace NUMINAMATH_GPT_determine_n_from_average_l40_4009

-- Definitions derived from conditions
def total_cards (n : ℕ) : ℕ := n * (n + 1) / 2
def sum_of_values (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6
def average_value (n : ℕ) : ℚ := sum_of_values n / total_cards n

-- Main statement for proving equivalence
theorem determine_n_from_average :
  (∃ n : ℕ, average_value n = 2023) ↔ (n = 3034) :=
by
  sorry

end NUMINAMATH_GPT_determine_n_from_average_l40_4009


namespace NUMINAMATH_GPT_andrew_total_homeless_shelter_donation_l40_4074

-- Given constants and conditions
def bake_sale_total : ℕ := 400
def ingredients_cost : ℕ := 100
def piggy_bank_donation : ℕ := 10

-- Intermediate calculated values
def remaining_total : ℕ := bake_sale_total - ingredients_cost
def shelter_donation_from_bake_sale : ℕ := remaining_total / 2

-- Final goal statement
theorem andrew_total_homeless_shelter_donation :
  shelter_donation_from_bake_sale + piggy_bank_donation = 160 :=
by
  -- Proof to be provided.
  sorry

end NUMINAMATH_GPT_andrew_total_homeless_shelter_donation_l40_4074


namespace NUMINAMATH_GPT_moles_of_KHSO4_formed_l40_4000

-- Chemical reaction definition
def reaction (n_KOH n_H2SO4 : ℕ) : ℕ :=
  if n_KOH = n_H2SO4 then n_KOH else 0

-- Given conditions
def moles_KOH : ℕ := 2
def moles_H2SO4 : ℕ := 2

-- Proof statement to be proved
theorem moles_of_KHSO4_formed : reaction moles_KOH moles_H2SO4 = 2 :=
by sorry

end NUMINAMATH_GPT_moles_of_KHSO4_formed_l40_4000


namespace NUMINAMATH_GPT_division_result_l40_4053

theorem division_result : 3486 / 189 = 18.444444444444443 := 
by sorry

end NUMINAMATH_GPT_division_result_l40_4053


namespace NUMINAMATH_GPT_fraction_problem_l40_4017

theorem fraction_problem
    (q r s u : ℚ)
    (h1 : q / r = 8)
    (h2 : s / r = 4)
    (h3 : s / u = 1 / 3) :
    u / q = 3 / 2 :=
  sorry

end NUMINAMATH_GPT_fraction_problem_l40_4017


namespace NUMINAMATH_GPT_simplify_expression_l40_4007

theorem simplify_expression :
  ((1 + 2 + 3 + 6) / 3) + ((3 * 6 + 9) / 4) = 43 / 4 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l40_4007


namespace NUMINAMATH_GPT_dog_cat_food_difference_l40_4098

theorem dog_cat_food_difference :
  let dogFood := 600
  let catFood := 327
  dogFood - catFood = 273 :=
by
  let dogFood := 600
  let catFood := 327
  show dogFood - catFood = 273
  sorry

end NUMINAMATH_GPT_dog_cat_food_difference_l40_4098


namespace NUMINAMATH_GPT_simplify_expression_is_3_l40_4049

noncomputable def simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) : ℝ :=
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)

theorem simplify_expression_is_3 (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) :
  simplify_expression x y z hx hy hz h = 3 :=
  sorry

end NUMINAMATH_GPT_simplify_expression_is_3_l40_4049


namespace NUMINAMATH_GPT_domain_of_f_l40_4097

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log x + Real.sqrt (2 - x)

theorem domain_of_f :
  { x : ℝ | 0 < x ∧ x ≤ 2 ∧ x ≠ 1 } = { x : ℝ | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) } :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l40_4097


namespace NUMINAMATH_GPT_tan_20_plus_4sin_20_eq_sqrt3_l40_4064

theorem tan_20_plus_4sin_20_eq_sqrt3 :
  (Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_20_plus_4sin_20_eq_sqrt3_l40_4064


namespace NUMINAMATH_GPT_pages_revised_only_once_l40_4013

theorem pages_revised_only_once 
  (total_pages : ℕ)
  (cost_per_page_first_time : ℝ)
  (cost_per_page_revised : ℝ)
  (revised_twice_pages : ℕ)
  (total_cost : ℝ)
  (pages_revised_only_once : ℕ) :
  total_pages = 100 →
  cost_per_page_first_time = 10 →
  cost_per_page_revised = 5 →
  revised_twice_pages = 30 →
  total_cost = 1400 →
  10 * (total_pages - pages_revised_only_once - revised_twice_pages) + 
  15 * pages_revised_only_once + 
  20 * revised_twice_pages = total_cost →
  pages_revised_only_once = 20 :=
by
  intros 
  sorry

end NUMINAMATH_GPT_pages_revised_only_once_l40_4013


namespace NUMINAMATH_GPT_solve_problem_l40_4072

noncomputable def problem_expression : ℝ :=
  4^(1/2) + Real.log (3^2) / Real.log 3

theorem solve_problem : problem_expression = 4 := by
  sorry

end NUMINAMATH_GPT_solve_problem_l40_4072


namespace NUMINAMATH_GPT_problem_equivalent_proof_l40_4071

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end NUMINAMATH_GPT_problem_equivalent_proof_l40_4071


namespace NUMINAMATH_GPT_postman_speeds_l40_4016

-- Define constants for the problem
def d1 : ℝ := 2 -- distance uphill in km
def d2 : ℝ := 4 -- distance on flat ground in km
def d3 : ℝ := 3 -- distance downhill in km
def time1 : ℝ := 2.267 -- time from A to B in hours
def time2 : ℝ := 2.4 -- time from B to A in hours
def half_time_round_trip : ℝ := 2.317 -- round trip to halfway point in hours

-- Define the speeds
noncomputable def V1 : ℝ := 3 -- speed uphill in km/h
noncomputable def V2 : ℝ := 4 -- speed on flat ground in km/h
noncomputable def V3 : ℝ := 5 -- speed downhill in km/h

-- The mathematically equivalent proof statement
theorem postman_speeds :
  (d1 / V1 + d2 / V2 + d3 / V3 = time1) ∧
  (d3 / V1 + d2 / V2 + d1 / V3 = time2) ∧
  (1 / V1 + 2 / V2 + 1.5 / V3 = half_time_round_trip / 2) :=
by 
  -- Equivalence holds because the speeds satisfy the given conditions
  sorry

end NUMINAMATH_GPT_postman_speeds_l40_4016


namespace NUMINAMATH_GPT_problem_abc_value_l40_4027

theorem problem_abc_value 
  (a b c : ℤ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > 0)
  (h4 : Int.gcd b c = 1)
  (h5 : (b + c) % a = 0)
  (h6 : (a + c) % b = 0) :
  a * b * c = 6 :=
sorry

end NUMINAMATH_GPT_problem_abc_value_l40_4027


namespace NUMINAMATH_GPT_evaluate_custom_operation_l40_4080

def custom_operation (x y : ℕ) : ℕ := 2 * x - 4 * y

theorem evaluate_custom_operation :
  custom_operation 7 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_custom_operation_l40_4080


namespace NUMINAMATH_GPT_probability_of_all_selected_l40_4023

theorem probability_of_all_selected :
  let p_x := 1 / 7
  let p_y := 2 / 9
  let p_z := 3 / 11
  p_x * p_y * p_z = 1 / 115.5 :=
by
  let p_x := 1 / 7
  let p_y := 2 / 9
  let p_z := 3 / 11
  sorry

end NUMINAMATH_GPT_probability_of_all_selected_l40_4023


namespace NUMINAMATH_GPT_inequality_3a3_2b3_3a2b_2ab2_l40_4067

theorem inequality_3a3_2b3_3a2b_2ab2 (a b : ℝ) (h₁ : a ≥ b) (h₂ : b > 0) : 
  3 * a ^ 3 + 2 * b ^ 3 ≥ 3 * a ^ 2 * b + 2 * a * b ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_3a3_2b3_3a2b_2ab2_l40_4067
