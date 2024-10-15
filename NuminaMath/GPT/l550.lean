import Mathlib

namespace NUMINAMATH_GPT_bus_driver_total_compensation_l550_55013

theorem bus_driver_total_compensation :
  let regular_rate := 16
  let regular_hours := 40
  let overtime_hours := 60 - regular_hours
  let overtime_rate := regular_rate + 0.75 * regular_rate
  let regular_pay := regular_hours * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  let total_compensation := regular_pay + overtime_pay
  total_compensation = 1200 := by
  sorry

end NUMINAMATH_GPT_bus_driver_total_compensation_l550_55013


namespace NUMINAMATH_GPT_initial_speed_l550_55082

variable (D T : ℝ) -- Total distance D and total time T
variable (S : ℝ)   -- Initial speed S

theorem initial_speed :
  (2 * D / 3) = (S * T / 3) →
  (35 = (D / (2 * T))) →
  S = 70 :=
by
  intro h1 h2
  -- Skipping the proof with 'sorry'
  sorry

end NUMINAMATH_GPT_initial_speed_l550_55082


namespace NUMINAMATH_GPT_solution_of_inequality_l550_55000

theorem solution_of_inequality (x : ℝ) : 3 * x > 2 * x + 4 ↔ x > 4 := 
sorry

end NUMINAMATH_GPT_solution_of_inequality_l550_55000


namespace NUMINAMATH_GPT_tan_beta_minus_pi_over_4_l550_55098

theorem tan_beta_minus_pi_over_4 (α β : ℝ) 
  (h1 : Real.tan (α + β) = 1/2) 
  (h2 : Real.tan (α + π/4) = -1/3) : 
  Real.tan (β - π/4) = 1 := 
sorry

end NUMINAMATH_GPT_tan_beta_minus_pi_over_4_l550_55098


namespace NUMINAMATH_GPT_arccos_cos_eight_l550_55070

theorem arccos_cos_eight : Real.arccos (Real.cos 8) = 8 - 2 * Real.pi :=
by sorry

end NUMINAMATH_GPT_arccos_cos_eight_l550_55070


namespace NUMINAMATH_GPT_coefficient_of_x4_l550_55085

theorem coefficient_of_x4 (a : ℝ) (h : 15 * a^4 = 240) : a = 2 ∨ a = -2 := 
sorry

end NUMINAMATH_GPT_coefficient_of_x4_l550_55085


namespace NUMINAMATH_GPT_greatest_integer_lesser_200_gcd_45_eq_9_l550_55044

theorem greatest_integer_lesser_200_gcd_45_eq_9 :
  ∃ n : ℕ, n < 200 ∧ Int.gcd n 45 = 9 ∧ ∀ m : ℕ, (m < 200 ∧ Int.gcd m 45 = 9) → m ≤ n :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_lesser_200_gcd_45_eq_9_l550_55044


namespace NUMINAMATH_GPT_class_tree_total_l550_55033

theorem class_tree_total
  (trees_A : ℕ)
  (trees_B : ℕ)
  (hA : trees_A = 8)
  (hB : trees_B = 7)
  : trees_A + trees_B = 15 := 
by
  sorry

end NUMINAMATH_GPT_class_tree_total_l550_55033


namespace NUMINAMATH_GPT_find_remainder_l550_55055

theorem find_remainder : 
    ∃ (d q r : ℕ), 472 = d * q + r ∧ 427 = d * (q - 5) + r ∧ r = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_remainder_l550_55055


namespace NUMINAMATH_GPT_heat_of_reaction_correct_l550_55057

def delta_H_f_NH4Cl : ℝ := -314.43  -- Enthalpy of formation of NH4Cl in kJ/mol
def delta_H_f_H2O : ℝ := -285.83    -- Enthalpy of formation of H2O in kJ/mol
def delta_H_f_HCl : ℝ := -92.31     -- Enthalpy of formation of HCl in kJ/mol
def delta_H_f_NH4OH : ℝ := -80.29   -- Enthalpy of formation of NH4OH in kJ/mol

def delta_H_rxn : ℝ :=
  ((2 * delta_H_f_NH4OH) + (2 * delta_H_f_HCl)) -
  ((2 * delta_H_f_NH4Cl) + (2 * delta_H_f_H2O))

theorem heat_of_reaction_correct :
  delta_H_rxn = 855.32 :=
  by
    -- Calculation and proof steps go here
    sorry

end NUMINAMATH_GPT_heat_of_reaction_correct_l550_55057


namespace NUMINAMATH_GPT_triangle_AX_length_l550_55050

noncomputable def length_AX (AB AC BC : ℝ) (h1 : AB = 60) (h2 : AC = 34) (h3 : BC = 52) : ℝ :=
  1020 / 43

theorem triangle_AX_length 
  (AB AC BC AX : ℝ)
  (h1 : AB = 60)
  (h2 : AC = 34)
  (h3 : BC = 52)
  (h4 : AX + (AB - AX) = AB)
  (h5 : AX / (AB - AX) = AC / BC) :
  AX = 1020 / 43 := 
sorry

end NUMINAMATH_GPT_triangle_AX_length_l550_55050


namespace NUMINAMATH_GPT_comparison_theorem_l550_55065

open Real

noncomputable def comparison (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) : Prop :=
  let a := log (sin x)
  let b := sin x
  let c := exp (sin x)
  a < b ∧ b < c

theorem comparison_theorem (x : ℝ) (h : 0 < x ∧ x < π / 2) : comparison x h.1 h.2 :=
by { sorry }

end NUMINAMATH_GPT_comparison_theorem_l550_55065


namespace NUMINAMATH_GPT_percentage_passed_l550_55022

-- Definitions corresponding to the conditions
def F_H : ℝ := 25
def F_E : ℝ := 35
def F_B : ℝ := 40

-- Main theorem stating the question's proof.
theorem percentage_passed :
  (100 - (F_H + F_E - F_B)) = 80 :=
by
  -- we can transcribe the remaining process here if needed.
  sorry

end NUMINAMATH_GPT_percentage_passed_l550_55022


namespace NUMINAMATH_GPT_total_collected_funds_l550_55058

theorem total_collected_funds (A B T : ℕ) (hA : A = 5) (hB : B = 3 * A + 3) (h_quotient : B / 3 = 6) (hT : T = B * (B / 3) + A) : 
  T = 113 := 
by 
  sorry

end NUMINAMATH_GPT_total_collected_funds_l550_55058


namespace NUMINAMATH_GPT_intersection_complement_M_and_N_l550_55064
open Set

def U := @univ ℝ
def M := {x : ℝ | x^2 + 2*x - 8 ≤ 0}
def N := {x : ℝ | -1 < x ∧ x < 3}
def complement_M := {x : ℝ | ¬ (x ∈ M)}

theorem intersection_complement_M_and_N :
  (complement_M ∩ N) = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_M_and_N_l550_55064


namespace NUMINAMATH_GPT_roots_poly_eq_l550_55089

theorem roots_poly_eq (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : d = 0) (root1_eq : 64 * a + 16 * b + 4 * c = 0) (root2_eq : -27 * a + 9 * b - 3 * c = 0) :
  (b + c) / a = -13 :=
by {
  sorry
}

end NUMINAMATH_GPT_roots_poly_eq_l550_55089


namespace NUMINAMATH_GPT_find_decreased_value_l550_55018

theorem find_decreased_value (x v : ℝ) (hx : x = 7)
  (h : x - v = 21 * (1 / x)) : v = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_decreased_value_l550_55018


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l550_55060

theorem quadratic_inequality_solution_set (m : ℝ) (h : m * (m - 1) < 0) : 
  ∀ x : ℝ, (x^2 - (m + 1/m) * x + 1 < 0) ↔ m < x ∧ x < 1/m :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l550_55060


namespace NUMINAMATH_GPT_total_sticks_needed_l550_55029

theorem total_sticks_needed (simon_sticks gerry_sticks micky_sticks darryl_sticks : ℕ):
  simon_sticks = 36 →
  gerry_sticks = (2 * simon_sticks) / 3 →
  micky_sticks = simon_sticks + gerry_sticks + 9 →
  darryl_sticks = simon_sticks + gerry_sticks + micky_sticks + 1 →
  simon_sticks + gerry_sticks + micky_sticks + darryl_sticks = 259 :=
by
  intros h_simon h_gerry h_micky h_darryl
  rw [h_simon, h_gerry, h_micky, h_darryl]
  norm_num
  sorry

end NUMINAMATH_GPT_total_sticks_needed_l550_55029


namespace NUMINAMATH_GPT_population_after_10_years_l550_55040

def initial_population : ℕ := 100000
def birth_increase_percent : ℝ := 0.6
def emigration_per_year : ℕ := 2000
def immigration_per_year : ℕ := 2500
def years : ℕ := 10

theorem population_after_10_years :
  let birth_increase := initial_population * birth_increase_percent
  let total_emigration := emigration_per_year * years
  let total_immigration := immigration_per_year * years
  let net_movement := total_immigration - total_emigration
  let final_population := initial_population + birth_increase + net_movement
  final_population = 165000 :=
by
  sorry

end NUMINAMATH_GPT_population_after_10_years_l550_55040


namespace NUMINAMATH_GPT_supplementary_angle_ratio_l550_55062

theorem supplementary_angle_ratio (x : ℝ) (hx : 4 * x + x = 180) : x = 36 :=
by sorry

end NUMINAMATH_GPT_supplementary_angle_ratio_l550_55062


namespace NUMINAMATH_GPT_calculate_expression_l550_55049

theorem calculate_expression : 
  (-6)^6 / 6^4 - 2^5 + 9^2 = 85 := 
by sorry

end NUMINAMATH_GPT_calculate_expression_l550_55049


namespace NUMINAMATH_GPT_ella_emma_hotdogs_l550_55023

-- Definitions based on the problem conditions
def hotdogs_each_sister_wants (E : ℕ) :=
  let luke := 2 * E
  let hunter := 3 * E
  E + E + luke + hunter = 14

-- Statement we need to prove
theorem ella_emma_hotdogs (E : ℕ) (h : hotdogs_each_sister_wants E) : E = 2 :=
by
  sorry

end NUMINAMATH_GPT_ella_emma_hotdogs_l550_55023


namespace NUMINAMATH_GPT_range_x_satisfies_inequality_l550_55067

theorem range_x_satisfies_inequality (x : ℝ) : (x^2 < |x|) ↔ (-1 < x ∧ x < 1 ∧ x ≠ 0) :=
sorry

end NUMINAMATH_GPT_range_x_satisfies_inequality_l550_55067


namespace NUMINAMATH_GPT_roots_form_parallelogram_l550_55087

theorem roots_form_parallelogram :
  let polynomial := fun (z : ℂ) (a : ℝ) =>
    z^4 - 8*z^3 + 13*a*z^2 - 2*(3*a^2 + 2*a - 4)*z - 2
  let a1 := 7.791
  let a2 := -8.457
  ∀ z1 z2 z3 z4 : ℂ,
    ( (polynomial z1 a1 = 0) ∧ (polynomial z2 a1 = 0) ∧ (polynomial z3 a1 = 0) ∧ (polynomial z4 a1 = 0)
    ∨ (polynomial z1 a2 = 0) ∧ (polynomial z2 a2 = 0) ∧ (polynomial z3 a2 = 0) ∧ (polynomial z4 a2 = 0) )
    → ( (z1 + z2 + z3 + z4) / 4 = 2 )
    → ( Complex.abs (z1 - z2) = Complex.abs (z3 - z4) 
      ∧ Complex.abs (z1 - z3) = Complex.abs (z2 - z4) ) := sorry

end NUMINAMATH_GPT_roots_form_parallelogram_l550_55087


namespace NUMINAMATH_GPT_cash_price_eq_8000_l550_55006

noncomputable def cash_price (d m s : ℕ) : ℕ :=
  d + 30 * m - s

theorem cash_price_eq_8000 :
  cash_price 3000 300 4000 = 8000 :=
by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_cash_price_eq_8000_l550_55006


namespace NUMINAMATH_GPT_prob_green_is_correct_l550_55047

-- Define the probability of picking any container
def prob_pick_container : ℚ := 1 / 4

-- Define the probability of drawing a green ball from each container
def prob_green_A : ℚ := 6 / 10
def prob_green_B : ℚ := 3 / 10
def prob_green_C : ℚ := 3 / 10
def prob_green_D : ℚ := 5 / 10

-- Define the individual probabilities for a green ball, accounting for container selection
def prob_green_given_A : ℚ := prob_pick_container * prob_green_A
def prob_green_given_B : ℚ := prob_pick_container * prob_green_B
def prob_green_given_C : ℚ := prob_pick_container * prob_green_C
def prob_green_given_D : ℚ := prob_pick_container * prob_green_D

-- Calculate the total probability of selecting a green ball
def prob_green_total : ℚ := prob_green_given_A + prob_green_given_B + prob_green_given_C + prob_green_given_D

-- Theorem statement: The probability of selecting a green ball is 17/40
theorem prob_green_is_correct : prob_green_total = 17 / 40 :=
by
  -- Proof will be provided here.
  sorry

end NUMINAMATH_GPT_prob_green_is_correct_l550_55047


namespace NUMINAMATH_GPT_eugene_boxes_needed_l550_55045

-- Define the number of cards in the deck
def total_cards : ℕ := 52

-- Define the number of cards not used
def unused_cards : ℕ := 16

-- Define the number of toothpicks per card
def toothpicks_per_card : ℕ := 75

-- Define the number of toothpicks in a box
def toothpicks_per_box : ℕ := 450

-- Calculate the number of cards used
def cards_used : ℕ := total_cards - unused_cards

-- Calculate the number of cards a single box can support
def cards_per_box : ℕ := toothpicks_per_box / toothpicks_per_card

-- Theorem statement
theorem eugene_boxes_needed : cards_used / cards_per_box = 6 := by
  -- The proof steps are not provided as per the instructions. 
  sorry

end NUMINAMATH_GPT_eugene_boxes_needed_l550_55045


namespace NUMINAMATH_GPT_cost_of_first_15_kgs_l550_55063

def cost_33_kg := 333
def cost_36_kg := 366
def kilo_33 := 33
def kilo_36 := 36
def first_limit := 30
def extra_3kg := 3  -- 33 - 30
def extra_6kg := 6  -- 36 - 30

theorem cost_of_first_15_kgs (l q : ℕ) 
  (h1 : first_limit * l + extra_3kg * q = cost_33_kg)
  (h2 : first_limit * l + extra_6kg * q = cost_36_kg) :
  15 * l = 150 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_first_15_kgs_l550_55063


namespace NUMINAMATH_GPT_two_pow_58_plus_one_factored_l550_55025

theorem two_pow_58_plus_one_factored :
  ∃ (a b c : ℕ), 2 < a ∧ 2 < b ∧ 2 < c ∧ 2 ^ 58 + 1 = a * b * c :=
sorry

end NUMINAMATH_GPT_two_pow_58_plus_one_factored_l550_55025


namespace NUMINAMATH_GPT_ben_fraction_of_taxes_l550_55093

theorem ben_fraction_of_taxes 
  (gross_income : ℝ) (car_payment : ℝ) (fraction_spend_on_car : ℝ) (after_tax_income_fraction : ℝ) 
  (h1 : gross_income = 3000) (h2 : car_payment = 400) (h3 : fraction_spend_on_car = 0.2) :
  after_tax_income_fraction = (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_ben_fraction_of_taxes_l550_55093


namespace NUMINAMATH_GPT_brendan_taxes_l550_55039

def total_hours (num_8hr_shifts : ℕ) (num_12hr_shifts : ℕ) : ℕ :=
  (num_8hr_shifts * 8) + (num_12hr_shifts * 12)

def total_wage (hourly_wage : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_wage * hours_worked

def total_tips (hourly_tips : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_tips * hours_worked

def reported_tips (total_tips : ℕ) (report_fraction : ℕ) : ℕ :=
  total_tips / report_fraction

def reported_income (wage : ℕ) (tips : ℕ) : ℕ :=
  wage + tips

def taxes (income : ℕ) (tax_rate : ℚ) : ℚ :=
  income * tax_rate

theorem brendan_taxes (num_8hr_shifts num_12hr_shifts : ℕ)
    (hourly_wage hourly_tips report_fraction : ℕ) (tax_rate : ℚ) :
    (hourly_wage = 6) →
    (hourly_tips = 12) →
    (report_fraction = 3) →
    (tax_rate = 0.2) →
    (num_8hr_shifts = 2) →
    (num_12hr_shifts = 1) →
    taxes (reported_income (total_wage hourly_wage (total_hours num_8hr_shifts num_12hr_shifts))
            (reported_tips (total_tips hourly_tips (total_hours num_8hr_shifts num_12hr_shifts))
            report_fraction))
          tax_rate = 56 :=
by
  intros
  sorry

end NUMINAMATH_GPT_brendan_taxes_l550_55039


namespace NUMINAMATH_GPT_equation_result_l550_55096

theorem equation_result : 
  ∀ (n : ℝ), n = 5.0 → (4 * n + 7 * n) = 55.0 :=
by
  intro n h
  rw [h]
  norm_num

end NUMINAMATH_GPT_equation_result_l550_55096


namespace NUMINAMATH_GPT_find_a12_l550_55072

variable (a : ℕ → ℝ) (q : ℝ)
variable (h1 : ∀ n, a (n + 1) = a n * q)
variable (h2 : abs q > 1)
variable (h3 : a 1 + a 6 = 2)
variable (h4 : a 3 * a 4 = -15)

theorem find_a12 : a 11 = -25 / 3 :=
by sorry

end NUMINAMATH_GPT_find_a12_l550_55072


namespace NUMINAMATH_GPT_prime_condition_l550_55046

theorem prime_condition (p : ℕ) (h_prime: Nat.Prime p) :
  (∃ m n : ℤ, p = m^2 + n^2 ∧ (m^3 + n^3 - 4) % p = 0) ↔ p = 2 ∨ p = 5 :=
by
  sorry

end NUMINAMATH_GPT_prime_condition_l550_55046


namespace NUMINAMATH_GPT_remainder_equality_l550_55016

theorem remainder_equality
  (P P' K D R R' r r' : ℕ)
  (h1 : P > P')
  (h2 : P % K = 0)
  (h3 : P' % K = 0)
  (h4 : P % D = R)
  (h5 : P' % D = R')
  (h6 : (P * K - P') % D = r)
  (h7 : (R * K - R') % D = r') :
  r = r' :=
sorry

end NUMINAMATH_GPT_remainder_equality_l550_55016


namespace NUMINAMATH_GPT_hyperbola_asymptote_eccentricity_l550_55020

-- Problem statement: We need to prove that the eccentricity of hyperbola 
-- given the specific asymptote is sqrt(5).

noncomputable def calc_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_asymptote_eccentricity 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_asymptote : b = 2 * a) :
  calc_eccentricity a b = Real.sqrt 5 := 
by
  -- Insert the proof step here
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_eccentricity_l550_55020


namespace NUMINAMATH_GPT_average_speed_problem_l550_55034

noncomputable def average_speed (d₁ d₂ d₃ d₄ t₁ t₂ t₃ t₄ : ℝ) : ℝ :=
  (d₁ + d₂ + d₃ + d₄) / (t₁ + t₂ + t₃ + t₄)

theorem average_speed_problem :
  average_speed 30 40 37.5 7 (30 / 35) (40 / 55) 0.5 (10 / 60) = 51 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_average_speed_problem_l550_55034


namespace NUMINAMATH_GPT_DE_eq_DF_l550_55035

variable {Point : Type}
variable {E A B C D F : Point}
variable (square : Π (A B C D : Point), Prop ) 
variable (is_parallel : Π (A B : Point), Prop) 
variable (E_outside_square : Prop)
variable (BE_eq_BD : Prop)
variable (BE_intersects_AD_at_F : Prop)

theorem DE_eq_DF
  (H1 : square A B C D)
  (H2 : is_parallel AE BD)
  (H3 : BE_eq_BD)
  (H4 : BE_intersects_AD_at_F) :
  DE = DF := 
sorry

end NUMINAMATH_GPT_DE_eq_DF_l550_55035


namespace NUMINAMATH_GPT_twenty_first_term_is_4641_l550_55090

def nthGroupStart (n : ℕ) : ℕ :=
  1 + (n * (n - 1)) / 2

def sumGroup (start n : ℕ) : ℕ :=
  (n * (start + (start + n - 1))) / 2

theorem twenty_first_term_is_4641 : sumGroup (nthGroupStart 21) 21 = 4641 := by
  sorry

end NUMINAMATH_GPT_twenty_first_term_is_4641_l550_55090


namespace NUMINAMATH_GPT_hyperbola_foci_coordinates_l550_55053

theorem hyperbola_foci_coordinates :
  ∀ (x y : ℝ), x^2 - (y^2 / 3) = 1 → (∃ c : ℝ, c = 2 ∧ (x = c ∨ x = -c) ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_foci_coordinates_l550_55053


namespace NUMINAMATH_GPT_sum_of_remainders_l550_55037

theorem sum_of_remainders
  (a b c : ℕ)
  (h₁ : a % 36 = 15)
  (h₂ : b % 36 = 22)
  (h₃ : c % 36 = 9) :
  (a + b + c) % 36 = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l550_55037


namespace NUMINAMATH_GPT_smallest_pieces_left_l550_55019

theorem smallest_pieces_left (m n : ℕ) (h1 : 1 < m) (h2 : 1 < n) : 
    ∃ k, (k = 2 ∧ (m * n) % 3 = 0) ∨ (k = 1 ∧ (m * n) % 3 ≠ 0) :=
by
    sorry

end NUMINAMATH_GPT_smallest_pieces_left_l550_55019


namespace NUMINAMATH_GPT_Adam_final_amount_l550_55010

def initial_amount : ℝ := 5.25
def spent_on_game : ℝ := 2.30
def spent_on_snacks : ℝ := 1.75
def found_dollar : ℝ := 1.00
def allowance : ℝ := 5.50

theorem Adam_final_amount :
  (initial_amount - spent_on_game - spent_on_snacks + found_dollar + allowance) = 7.70 :=
by
  sorry

end NUMINAMATH_GPT_Adam_final_amount_l550_55010


namespace NUMINAMATH_GPT_people_per_column_in_second_scenario_l550_55043

def total_people (num_people_per_column_1 : ℕ) (num_columns_1 : ℕ) : ℕ :=
  num_people_per_column_1 * num_columns_1

def people_per_column_second_scenario (P: ℕ) (num_columns_2 : ℕ) : ℕ :=
  P / num_columns_2

theorem people_per_column_in_second_scenario
  (num_people_per_column_1 : ℕ)
  (num_columns_1 : ℕ)
  (num_columns_2 : ℕ)
  (P : ℕ)
  (h1 : total_people num_people_per_column_1 num_columns_1 = P) :
  people_per_column_second_scenario P num_columns_2 = 48 :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_people_per_column_in_second_scenario_l550_55043


namespace NUMINAMATH_GPT_sum_of_products_two_at_a_time_l550_55068

theorem sum_of_products_two_at_a_time (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a + b + c = 23) : 
  ab + bc + ac = 131 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_products_two_at_a_time_l550_55068


namespace NUMINAMATH_GPT_min_value_of_squares_l550_55080

theorem min_value_of_squares (a b : ℝ) 
  (h_cond : a^2 - 2015 * a = b^2 - 2015 * b)
  (h_neq : a ≠ b)
  (h_positive_a : 0 < a)
  (h_positive_b : 0 < b) : 
  a^2 + b^2 ≥ 2015^2 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_squares_l550_55080


namespace NUMINAMATH_GPT_outer_boundary_diameter_l550_55048

theorem outer_boundary_diameter (d_pond : ℝ) (w_picnic : ℝ) (w_track : ℝ)
  (h_pond_diam : d_pond = 16) (h_picnic_width : w_picnic = 10) (h_track_width : w_track = 4) :
  2 * (d_pond / 2 + w_picnic + w_track) = 44 :=
by
  -- We avoid the entire proof, we only assert the statement in Lean
  sorry

end NUMINAMATH_GPT_outer_boundary_diameter_l550_55048


namespace NUMINAMATH_GPT_find_integer_l550_55036

theorem find_integer (x : ℕ) (h1 : (4 * x)^2 + 2 * x = 3528) : x = 14 := by
  sorry

end NUMINAMATH_GPT_find_integer_l550_55036


namespace NUMINAMATH_GPT_both_firms_participate_number_of_firms_participate_social_optimality_l550_55041

-- Definitions for general conditions
variable (α V IC : ℝ)
variable (hα : 0 < α ∧ α < 1)

-- Condition for both firms to participate
def condition_to_participate (V : ℝ) (α : ℝ) (IC : ℝ) : Prop :=
  V * α * (1 - 0.5 * α) ≥ IC

-- Part (a): Under what conditions will both firms participate?
theorem both_firms_participate (α V IC : ℝ) (hα : 0 < α ∧ α < 1) :
  condition_to_participate V α IC → (V * α * (1 - 0.5 * α) ≥ IC) :=
by sorry

-- Part (b): Given V=16, α=0.5, and IC=5, determine the number of firms participating
theorem number_of_firms_participate :
  (condition_to_participate 16 0.5 5) :=
by sorry

-- Part (c): To determine if the number of participating firms is socially optimal
def total_profit (α V IC : ℝ) (both : Bool) :=
  if both then 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC)
  else α * V - IC

theorem social_optimality :
   (total_profit 0.5 16 5 true ≠ max (total_profit 0.5 16 5 true) (total_profit 0.5 16 5 false)) :=
by sorry

end NUMINAMATH_GPT_both_firms_participate_number_of_firms_participate_social_optimality_l550_55041


namespace NUMINAMATH_GPT_least_integer_x_l550_55077

theorem least_integer_x (x : ℤ) : (2 * |x| + 7 < 17) → x = -4 := by
  sorry

end NUMINAMATH_GPT_least_integer_x_l550_55077


namespace NUMINAMATH_GPT_optimal_response_l550_55069

theorem optimal_response (n : ℕ) (m : ℕ) (s : ℕ) (a_1 : ℕ) (a_2 : ℕ -> ℕ) (a_opt : ℕ):
  n = 100 → 
  m = 107 →
  (∀ i, i ≥ 1 ∧ i ≤ 99 → a_2 i = a_opt) →
  a_1 = 7 :=
by
  sorry

end NUMINAMATH_GPT_optimal_response_l550_55069


namespace NUMINAMATH_GPT_exists_coprime_less_than_100_l550_55074

theorem exists_coprime_less_than_100 (a b c : ℕ) (ha : a < 1000000) (hb : b < 1000000) (hc : c < 1000000) :
  ∃ d, d < 100 ∧ gcd d a = 1 ∧ gcd d b = 1 ∧ gcd d c = 1 :=
by sorry

end NUMINAMATH_GPT_exists_coprime_less_than_100_l550_55074


namespace NUMINAMATH_GPT_shortest_minor_arc_line_equation_l550_55081

noncomputable def pointM : (ℝ × ℝ) := (1, -2)
noncomputable def circleC (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

theorem shortest_minor_arc_line_equation :
  (∀ x y : ℝ, (x + 2 * y + 3 = 0) ↔ 
  ((x = 1 ∧ y = -2) ∨ ∃ (k_l : ℝ), (k_l * (2) = -1) ∧ (y + 2 = -k_l * (x - 1)))) :=
sorry

end NUMINAMATH_GPT_shortest_minor_arc_line_equation_l550_55081


namespace NUMINAMATH_GPT_number_of_hexagonal_faces_geq_2_l550_55021

noncomputable def polyhedron_condition (P H : ℕ) : Prop :=
  ∃ V E : ℕ, 
    V - E + (P + H) = 2 ∧ 
    3 * V = 2 * E ∧ 
    E = (5 * P + 6 * H) / 2 ∧
    P > 0 ∧ H > 0

theorem number_of_hexagonal_faces_geq_2 (P H : ℕ) (h : polyhedron_condition P H) : H ≥ 2 :=
sorry

end NUMINAMATH_GPT_number_of_hexagonal_faces_geq_2_l550_55021


namespace NUMINAMATH_GPT_minimal_abs_diff_l550_55038

theorem minimal_abs_diff (a b : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b - 8 * a + 7 * b = 569) : abs (a - b) = 23 :=
sorry

end NUMINAMATH_GPT_minimal_abs_diff_l550_55038


namespace NUMINAMATH_GPT_minimum_value_of_x_plus_y_l550_55061

noncomputable def minValueSatisfies (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y + x * y = 2 → x + y ≥ 2 * Real.sqrt 3 - 2

theorem minimum_value_of_x_plus_y (x y : ℝ) : minValueSatisfies x y :=
by sorry

end NUMINAMATH_GPT_minimum_value_of_x_plus_y_l550_55061


namespace NUMINAMATH_GPT_equilateral_triangle_side_length_l550_55091

theorem equilateral_triangle_side_length (perimeter : ℕ) (h_perimeter : perimeter = 69) : 
  ∃ (side_length : ℕ), side_length = perimeter / 3 := 
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_side_length_l550_55091


namespace NUMINAMATH_GPT_smallest_x_for_1980_power4_l550_55007

theorem smallest_x_for_1980_power4 (M : ℤ) (x : ℕ) (hx : x > 0) :
  (1980 * (x : ℤ)) = M^4 → x = 6006250 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_smallest_x_for_1980_power4_l550_55007


namespace NUMINAMATH_GPT_irene_total_income_l550_55002

noncomputable def irene_income (weekly_hours : ℕ) (base_pay : ℕ) (overtime_pay : ℕ) (hours_worked : ℕ) : ℕ :=
  base_pay + (if hours_worked > weekly_hours then (hours_worked - weekly_hours) * overtime_pay else 0)

theorem irene_total_income :
  irene_income 40 500 20 50 = 700 :=
by
  sorry

end NUMINAMATH_GPT_irene_total_income_l550_55002


namespace NUMINAMATH_GPT_max_value_of_expression_l550_55094

theorem max_value_of_expression (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ 7 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l550_55094


namespace NUMINAMATH_GPT_men_build_walls_l550_55009

-- Define the variables
variables (a b d y : ℕ)

-- Define the work rate based on given conditions
def rate := d / (a * b)

-- Theorem to prove that y equals (a * a) / d given the conditions
theorem men_build_walls (h : a * b * y = a * a * d / a) : 
  y = a * a / d :=
by sorry

end NUMINAMATH_GPT_men_build_walls_l550_55009


namespace NUMINAMATH_GPT_pine_tree_next_one_in_between_l550_55079

theorem pine_tree_next_one_in_between (n : ℕ) (p s : ℕ) (trees : n = 2019) (pines : p = 1009) (spruces : s = 1010)
    (equal_intervals : true) : 
    ∃ (i : ℕ), (i < n) ∧ ((i + 1) % n ∈ {j | j < p}) ∧ ((i + 3) % n ∈ {j | j < p}) :=
  sorry

end NUMINAMATH_GPT_pine_tree_next_one_in_between_l550_55079


namespace NUMINAMATH_GPT_soap_box_missing_dimension_l550_55086

theorem soap_box_missing_dimension
  (x : ℕ) -- The missing dimension of the soap box
  (Volume_carton : ℕ := 25 * 48 * 60)
  (Volume_soap_box : ℕ := 8 * x * 5)
  (Max_soap_boxes : ℕ := 300)
  (condition : Max_soap_boxes * Volume_soap_box ≤ Volume_carton) :
  x ≤ 6 := by
sorry

end NUMINAMATH_GPT_soap_box_missing_dimension_l550_55086


namespace NUMINAMATH_GPT_rectangle_coloring_problem_l550_55073

theorem rectangle_coloring_problem :
  let n := 3
  let m := 4
  ∃ n, ∃ m, n = 3 ∧ m = 4 := sorry

end NUMINAMATH_GPT_rectangle_coloring_problem_l550_55073


namespace NUMINAMATH_GPT_find_q_r_s_l550_55008

noncomputable def is_valid_geometry 
  (AD : ℝ) (AL : ℝ) (AM : ℝ) (AN : ℝ) (q : ℕ) (r : ℕ) (s : ℕ) : Prop :=
  AD = 10 ∧ AL = 3 ∧ AM = 3 ∧ AN = 3 ∧ ¬(∃ p : ℕ, p^2 ∣ s)

theorem find_q_r_s : ∃ (q r s : ℕ), is_valid_geometry 10 3 3 3 q r s ∧ q + r + s = 711 :=
by
  sorry

end NUMINAMATH_GPT_find_q_r_s_l550_55008


namespace NUMINAMATH_GPT_max_cos_a_l550_55017

theorem max_cos_a (a b c : ℝ) 
  (h1 : Real.sin a = Real.cos b) 
  (h2 : Real.sin b = Real.cos c) 
  (h3 : Real.sin c = Real.cos a) : 
  Real.cos a = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_GPT_max_cos_a_l550_55017


namespace NUMINAMATH_GPT_simplify_fraction_l550_55083

theorem simplify_fraction :
  (4 * 6) / (12 * 15) * (5 * 12 * 15^2) / (2 * 6 * 5) = 2.5 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l550_55083


namespace NUMINAMATH_GPT_brick_height_calculation_l550_55097

theorem brick_height_calculation :
  ∀ (num_bricks : ℕ) (brick_length brick_width brick_height : ℝ)
    (wall_length wall_height wall_width : ℝ),
    num_bricks = 1600 →
    brick_length = 100 →
    brick_width = 11.25 →
    wall_length = 800 →
    wall_height = 600 →
    wall_width = 22.5 →
    wall_length * wall_height * wall_width = 
    num_bricks * brick_length * brick_width * brick_height →
    brick_height = 60 :=
by
  sorry

end NUMINAMATH_GPT_brick_height_calculation_l550_55097


namespace NUMINAMATH_GPT_square_area_l550_55059

theorem square_area {d : ℝ} (h : d = 12 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 144 ∧ ( ∃ s : ℝ, s = d / Real.sqrt 2 ∧ A = s^2 ) :=
by
  sorry

end NUMINAMATH_GPT_square_area_l550_55059


namespace NUMINAMATH_GPT_molecular_weight_BaCl2_l550_55084

def molecular_weight_one_mole (w_four_moles : ℕ) (n : ℕ) : ℕ := 
    w_four_moles / n

theorem molecular_weight_BaCl2 
    (w_four_moles : ℕ)
    (H : w_four_moles = 828) :
  molecular_weight_one_mole w_four_moles 4 = 207 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_molecular_weight_BaCl2_l550_55084


namespace NUMINAMATH_GPT_total_balls_l550_55027

theorem total_balls (jungkook_balls : ℕ) (yoongi_balls : ℕ) (h1 : jungkook_balls = 3) (h2 : yoongi_balls = 4) : 
  jungkook_balls + yoongi_balls = 7 :=
by
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_balls_l550_55027


namespace NUMINAMATH_GPT_painted_cells_l550_55075

theorem painted_cells (k l : ℕ) (h : k * l = 74) :
    (2 * k + 1) * (2 * l + 1) - k * l = 301 ∨ 
    (2 * k + 1) * (2 * l + 1) - k * l = 373 :=
sorry

end NUMINAMATH_GPT_painted_cells_l550_55075


namespace NUMINAMATH_GPT_price_increase_percentage_l550_55031

theorem price_increase_percentage (c : ℝ) (r : ℝ) (p : ℝ) 
  (h1 : r = 1.4 * c) 
  (h2 : p = 1.15 * r) : 
  (p - c) / c * 100 = 61 := 
sorry

end NUMINAMATH_GPT_price_increase_percentage_l550_55031


namespace NUMINAMATH_GPT_min_product_of_three_numbers_l550_55001

def SetOfNumbers : Set ℤ := {-9, -5, -1, 1, 3, 5, 8}

theorem min_product_of_three_numbers : 
  ∃ (a b c : ℤ), a ∈ SetOfNumbers ∧ b ∈ SetOfNumbers ∧ c ∈ SetOfNumbers ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b * c = -360 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_product_of_three_numbers_l550_55001


namespace NUMINAMATH_GPT_angle_sum_straight_line_l550_55012

theorem angle_sum_straight_line (x : ℝ) (h : 4 * x + x = 180) : x = 36 :=
sorry

end NUMINAMATH_GPT_angle_sum_straight_line_l550_55012


namespace NUMINAMATH_GPT_even_perfect_square_factors_l550_55042

theorem even_perfect_square_factors : 
  (∃ count : ℕ, count = 3 * 2 * 3 ∧ 
    (∀ (a b c : ℕ), 
      (1 ≤ a ∧ a ≤ 6 ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ b ≤ 3 ∧ c % 2 = 0 ∧ c ≤ 4) → 
      (2^a * 7^b * 3^c ∣ 2^6 * 7^3 * 3^4))) :=
sorry

end NUMINAMATH_GPT_even_perfect_square_factors_l550_55042


namespace NUMINAMATH_GPT_option_c_incorrect_l550_55003

theorem option_c_incorrect (a : ℝ) : a + a^2 ≠ a^3 :=
sorry

end NUMINAMATH_GPT_option_c_incorrect_l550_55003


namespace NUMINAMATH_GPT_angle_A_measure_l550_55078

theorem angle_A_measure 
  (B : ℝ) 
  (angle_in_smaller_triangle : ℝ) 
  (sum_of_triangle_angles_eq_180 : ∀ (x y z : ℝ), x + y + z = 180)
  (C : ℝ) 
  (angle_pair_linear : ∀ (x y : ℝ), x + y = 180) 
  (A : ℝ) 
  (C_eq_180_minus_B : C = 180 - B) 
  (A_eq_180_minus_angle_in_smaller_triangle_minus_C : 
    A = 180 - angle_in_smaller_triangle - C) :
  A = 70 :=
by
  sorry

end NUMINAMATH_GPT_angle_A_measure_l550_55078


namespace NUMINAMATH_GPT_triangle_area_l550_55088

def point := ℝ × ℝ

def A : point := (2, -3)
def B : point := (8, 1)
def C : point := (2, 3)

def area_triangle (A B C : point) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area : area_triangle A B C = 18 :=
  sorry

end NUMINAMATH_GPT_triangle_area_l550_55088


namespace NUMINAMATH_GPT_tiffany_daily_miles_l550_55015

-- Definitions for running schedule
def billy_sunday_miles := 1
def billy_monday_miles := 1
def billy_tuesday_miles := 1
def billy_wednesday_miles := 1
def billy_thursday_miles := 1
def billy_friday_miles := 1
def billy_saturday_miles := 1

def tiffany_wednesday_miles := 1 / 3
def tiffany_thursday_miles := 1 / 3
def tiffany_friday_miles := 1 / 3

-- Total miles is the sum of miles for the week
def billy_total_miles := billy_sunday_miles + billy_monday_miles + billy_tuesday_miles +
                         billy_wednesday_miles + billy_thursday_miles + billy_friday_miles +
                         billy_saturday_miles

def tiffany_total_miles (T : ℝ) := T * 3 + 
                                   tiffany_wednesday_miles + tiffany_thursday_miles + tiffany_friday_miles

-- Proof problem: show that Tiffany runs 2 miles each day on Sunday, Monday, and Tuesday
theorem tiffany_daily_miles : ∃ T : ℝ, (tiffany_total_miles T = billy_total_miles) ∧ T = 2 :=
by
  sorry

end NUMINAMATH_GPT_tiffany_daily_miles_l550_55015


namespace NUMINAMATH_GPT_driver_net_pay_rate_l550_55011

theorem driver_net_pay_rate
    (hours : ℕ) (distance_per_hour : ℕ) (distance_per_gallon : ℕ) 
    (pay_per_mile : ℝ) (gas_cost_per_gallon : ℝ) :
    hours = 3 →
    distance_per_hour = 50 →
    distance_per_gallon = 25 →
    pay_per_mile = 0.75 →
    gas_cost_per_gallon = 2.50 →
    (pay_per_mile * (distance_per_hour * hours) - gas_cost_per_gallon * ((distance_per_hour * hours) / distance_per_gallon)) / hours = 32.5 :=
by
  intros h_hours h_dph h_dpg h_ppm h_gcpg
  sorry

end NUMINAMATH_GPT_driver_net_pay_rate_l550_55011


namespace NUMINAMATH_GPT_aarti_completes_work_multiple_l550_55076

-- Define the condition that Aarti can complete one piece of work in 9 days.
def aarti_work_rate (work_size : ℕ) : ℕ := 9

-- Define the task to find how many times she will complete the work in 27 days.
def aarti_work_multiple (total_days : ℕ) (work_size: ℕ) : ℕ :=
  total_days / (aarti_work_rate work_size)

-- The theorem to prove the number of times Aarti will complete the work.
theorem aarti_completes_work_multiple : aarti_work_multiple 27 1 = 3 := by
  sorry

end NUMINAMATH_GPT_aarti_completes_work_multiple_l550_55076


namespace NUMINAMATH_GPT_elizabeth_needs_to_borrow_more_money_l550_55092

-- Define the costs of the items
def pencil_cost : ℝ := 6.00 
def notebook_cost : ℝ := 3.50 
def pen_cost : ℝ := 2.25 

-- Define the amount of money Elizabeth initially has and what she borrowed
def elizabeth_money : ℝ := 5.00 
def borrowed_money : ℝ := 0.53 

-- Define the total cost of the items
def total_cost : ℝ := pencil_cost + notebook_cost + pen_cost

-- Define the total amount of money Elizabeth has
def total_money : ℝ := elizabeth_money + borrowed_money

-- Define the additional amount Elizabeth needs to borrow
def amount_needed_to_borrow : ℝ := total_cost - total_money

-- The theorem to prove that Elizabeth needs to borrow an additional $6.22
theorem elizabeth_needs_to_borrow_more_money : 
  amount_needed_to_borrow = 6.22 := by 
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_elizabeth_needs_to_borrow_more_money_l550_55092


namespace NUMINAMATH_GPT_isosceles_right_triangle_area_l550_55005

/--
Given an isosceles right triangle with a hypotenuse of 6√2 units, prove that the area
of this triangle is 18 square units.
-/
theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (hyp : h = 6 * Real.sqrt 2) 
  (isosceles : h = l * Real.sqrt 2) : 
  (1/2) * l^2 = 18 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_area_l550_55005


namespace NUMINAMATH_GPT_verify_toothpick_count_l550_55026

def toothpick_problem : Prop :=
  let L := 45
  let W := 25
  let Mv := 8
  let Mh := 5
  -- Calculate the total number of vertical toothpicks
  let verticalToothpicks := (L + 1 - Mv) * W
  -- Calculate the total number of horizontal toothpicks
  let horizontalToothpicks := (W + 1 - Mh) * L
  -- Calculate the total number of toothpicks
  let totalToothpicks := verticalToothpicks + horizontalToothpicks
  -- Ensure the total matches the expected result
  totalToothpicks = 1895

theorem verify_toothpick_count : toothpick_problem :=
by
  sorry

end NUMINAMATH_GPT_verify_toothpick_count_l550_55026


namespace NUMINAMATH_GPT_N_def_M_intersection_CU_N_def_M_union_N_def_l550_55054

section Sets

variable {α : Type}

-- Declarations of conditions
def U := {x : ℝ | -3 ≤ x ∧ x ≤ 3}
def M := {x : ℝ | -1 < x ∧ x < 1}
def CU (N : Set ℝ) := {x : ℝ | 0 < x ∧ x < 2}

-- Problem statements
theorem N_def (N : Set ℝ) : N = {x : ℝ | (-3 ≤ x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x ≤ 3)} ↔ CU N = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

theorem M_intersection_CU_N_def (N : Set ℝ) : (M ∩ CU N) = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

theorem M_union_N_def (N : Set ℝ) : (M ∪ N) = {x : ℝ | (-3 ≤ x ∧ x < 1) ∨ (2 ≤ x ∧ x ≤ 3)} :=
by sorry

end Sets

end NUMINAMATH_GPT_N_def_M_intersection_CU_N_def_M_union_N_def_l550_55054


namespace NUMINAMATH_GPT_find_other_number_l550_55051

theorem find_other_number (a b : ℕ) (hcf_ab : Nat.gcd a b = 14) (lcm_ab : Nat.lcm a b = 396) (h : a = 36) : b = 154 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l550_55051


namespace NUMINAMATH_GPT_exists_a_b_divisible_l550_55066

theorem exists_a_b_divisible (n : ℕ) (hn : 0 < n) : 
  ∃ a b : ℤ, (4 * a^2 + 9 * b^2 - 1) % n = 0 := 
sorry

end NUMINAMATH_GPT_exists_a_b_divisible_l550_55066


namespace NUMINAMATH_GPT_largest_possible_s_l550_55071

theorem largest_possible_s 
  (r s : ℕ) 
  (hr : r ≥ s) 
  (hs : s ≥ 3) 
  (hangles : (r - 2) * 60 * s = (s - 2) * 61 * r) : 
  s = 121 := 
sorry

end NUMINAMATH_GPT_largest_possible_s_l550_55071


namespace NUMINAMATH_GPT_problem_statement_l550_55052

-- Definitions based on the conditions
def P : Prop := ∀ x : ℝ, (0 < x ∧ x < 1) ↔ (x / (x - 1) < 0)
def Q : Prop := ∀ (A B : ℝ), (A > B) → (A > 90 ∨ B < 90)

-- The proof problem statement
theorem problem_statement : P ∧ ¬Q := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l550_55052


namespace NUMINAMATH_GPT_find_number_l550_55095

variable (x : ℝ)

theorem find_number (hx : 5100 - (102 / x) = 5095) : x = 20.4 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l550_55095


namespace NUMINAMATH_GPT_number_of_football_players_l550_55024

theorem number_of_football_players
  (cricket_players : ℕ)
  (hockey_players : ℕ)
  (softball_players : ℕ)
  (total_players : ℕ) :
  cricket_players = 22 →
  hockey_players = 15 →
  softball_players = 19 →
  total_players = 77 →
  total_players - (cricket_players + hockey_players + softball_players) = 21 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_number_of_football_players_l550_55024


namespace NUMINAMATH_GPT_cheryl_bill_cost_correct_l550_55030

def cheryl_electricity_bill_cost : Prop :=
  ∃ (E : ℝ), 
    (E + 400) + 0.20 * (E + 400) = 1440 ∧ 
    E = 800

theorem cheryl_bill_cost_correct : cheryl_electricity_bill_cost :=
by
  sorry

end NUMINAMATH_GPT_cheryl_bill_cost_correct_l550_55030


namespace NUMINAMATH_GPT_not_constant_expression_l550_55028

noncomputable def is_centroid (A B C G : ℝ × ℝ) : Prop :=
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

noncomputable def squared_distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem not_constant_expression (A B C P G : ℝ × ℝ)
  (hG : is_centroid A B C G)
  (hP_on_AB : ∃ x, P = (x, A.2) ∧ A.2 = B.2) :
  ∃ dPA dPB dPC dPG : ℝ,
    dPA = squared_distance P A ∧
    dPB = squared_distance P B ∧
    dPC = squared_distance P C ∧
    dPG = squared_distance P G ∧
    (dPA + dPB + dPC - dPG) ≠ dPA + dPB + dPC - dPG := by
  sorry

end NUMINAMATH_GPT_not_constant_expression_l550_55028


namespace NUMINAMATH_GPT_compute_paths_in_grid_l550_55032

def grid : List (List Char) := [
  [' ', ' ', ' ', ' ', ' ', ' ', 'C', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', 'C', 'O', 'C', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', 'C', 'O', 'M', 'O', 'C', ' ', ' ', ' '],
  [' ', ' ', ' ', 'C', 'O', 'M', 'P', 'M', 'O', 'C', ' ', ' '],
  [' ', ' ', 'C', 'O', 'M', 'P', 'U', 'P', 'M', 'O', 'C', ' '],
  [' ', 'C', 'O', 'M', 'P', 'U', 'T', 'U', 'P', 'M', 'O', 'C'],
  ['C', 'O', 'M', 'P', 'U', 'T', 'E', 'T', 'U', 'P', 'M', 'O', 'C']
]

def is_valid_path (path : List (Nat × Nat)) : Bool :=
  -- This function checks if a given path is valid according to the problem's grid and rules.
  sorry

def count_paths_from_C_to_E (grid: List (List Char)) : Nat :=
  -- This function would count the number of valid paths from a 'C' in the leftmost column to an 'E' in the rightmost column.
  sorry

theorem compute_paths_in_grid : count_paths_from_C_to_E grid = 64 :=
by
  sorry

end NUMINAMATH_GPT_compute_paths_in_grid_l550_55032


namespace NUMINAMATH_GPT_total_bouquets_sold_l550_55004

-- defining the sale conditions
def monday_bouquets := 12
def tuesday_bouquets := 3 * monday_bouquets
def wednesday_bouquets := tuesday_bouquets / 3

-- defining the total sale
def total_bouquets := monday_bouquets + tuesday_bouquets + wednesday_bouquets

-- stating the theorem
theorem total_bouquets_sold : total_bouquets = 60 := by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_total_bouquets_sold_l550_55004


namespace NUMINAMATH_GPT_fraction_addition_l550_55056

theorem fraction_addition :
  (3 / 4) / (5 / 8) + (1 / 2) = 17 / 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_addition_l550_55056


namespace NUMINAMATH_GPT_find_monic_polynomial_l550_55014

-- Define the original polynomial
def polynomial_1 (x : ℝ) := x^3 - 4 * x^2 + 9

-- Define the monic polynomial we are seeking
def polynomial_2 (x : ℝ) := x^3 - 12 * x^2 + 243

theorem find_monic_polynomial :
  ∀ (r1 r2 r3 : ℝ), 
    polynomial_1 r1 = 0 → 
    polynomial_1 r2 = 0 → 
    polynomial_1 r3 = 0 → 
    polynomial_2 (3 * r1) = 0 ∧ polynomial_2 (3 * r2) = 0 ∧ polynomial_2 (3 * r3) = 0 :=
by
  intros r1 r2 r3 h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_monic_polynomial_l550_55014


namespace NUMINAMATH_GPT_expression_increase_l550_55099

variable {x y : ℝ}

theorem expression_increase (hx : x > 0) (hy : y > 0) :
  let original_expr := 3 * x^2 * y
  let new_x := 1.2 * x
  let new_y := 2.4 * y
  let new_expr := 3 * new_x ^ 2 * new_y
  (new_expr / original_expr) = 3.456 :=
by
-- original_expr is 3 * x^2 * y
-- new_x = 1.2 * x
-- new_y = 2.4 * y
-- new_expr = 3 * (1.2 * x)^2 * (2.4 * y)
-- (new_expr / original_expr) = (10.368 * x^2 * y) / (3 * x^2 * y)
-- (new_expr / original_expr) = 10.368 / 3
-- (new_expr / original_expr) = 3.456
sorry

end NUMINAMATH_GPT_expression_increase_l550_55099
