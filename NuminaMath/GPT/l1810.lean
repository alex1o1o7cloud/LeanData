import Mathlib

namespace NUMINAMATH_GPT_probability_king_then_ten_l1810_181098

-- Define the conditions
def standard_deck_size : ℕ := 52
def num_kings : ℕ := 4
def num_tens : ℕ := 4

-- Define the event probabilities
def prob_first_card_king : ℚ := num_kings / standard_deck_size
def prob_second_card_ten (remaining_deck_size : ℕ) : ℚ := num_tens / remaining_deck_size

-- The theorem statement to be proved
theorem probability_king_then_ten : 
  prob_first_card_king * prob_second_card_ten (standard_deck_size - 1) = 4 / 663 :=
by
  sorry

end NUMINAMATH_GPT_probability_king_then_ten_l1810_181098


namespace NUMINAMATH_GPT_value_independent_of_a_value_when_b_is_neg_2_l1810_181014

noncomputable def algebraic_expression (a b : ℝ) : ℝ :=
  3 * a^2 + (4 * a * b - a^2) - 2 * (a^2 + 2 * a * b - b^2)

theorem value_independent_of_a (a b : ℝ) : algebraic_expression a b = 2 * b^2 :=
by
  sorry

theorem value_when_b_is_neg_2 (a : ℝ) : algebraic_expression a (-2) = 8 :=
by
  sorry

end NUMINAMATH_GPT_value_independent_of_a_value_when_b_is_neg_2_l1810_181014


namespace NUMINAMATH_GPT_Fk_same_implies_eq_l1810_181056

def Q (n: ℕ) : ℕ :=
  -- Implementation of the square part of n
  sorry

def N (n: ℕ) : ℕ :=
  -- Implementation of the non-square part of n
  sorry

def Fk (k: ℕ) (n: ℕ) : ℕ :=
  -- Implementation of Fk function calculating the smallest positive integer bigger than kn such that Fk(n) * n is a perfect square
  sorry

theorem Fk_same_implies_eq (k: ℕ) (n m: ℕ) (hk: 0 < k) : Fk k n = Fk k m → n = m :=
  sorry

end NUMINAMATH_GPT_Fk_same_implies_eq_l1810_181056


namespace NUMINAMATH_GPT_range_of_expression_l1810_181008

variable {a b : ℝ}

theorem range_of_expression 
  (h₁ : -1 < a + b) (h₂ : a + b < 3)
  (h₃ : 2 < a - b) (h₄ : a - b < 4) :
  -9 / 2 < 2 * a + 3 * b ∧ 2 * a + 3 * b < 13 / 2 := 
sorry

end NUMINAMATH_GPT_range_of_expression_l1810_181008


namespace NUMINAMATH_GPT_value_of_e_l1810_181064

variable (e : ℝ)
noncomputable def eq1 : Prop :=
  ((10 * 0.3 + 2) / 4 - (3 * 0.3 - e) / 18 = (2 * 0.3 + 4) / 3)

theorem value_of_e : eq1 e → e = 6 := by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_e_l1810_181064


namespace NUMINAMATH_GPT_addition_example_l1810_181087

theorem addition_example : 248 + 64 = 312 := by
  sorry

end NUMINAMATH_GPT_addition_example_l1810_181087


namespace NUMINAMATH_GPT_A_share_of_profit_l1810_181017

-- Define the conditions
def A_investment : ℕ := 100
def A_months : ℕ := 12
def B_investment : ℕ := 200
def B_months : ℕ := 6
def total_profit : ℕ := 100

-- Calculate the weighted investments (directly from conditions)
def A_weighted_investment : ℕ := A_investment * A_months
def B_weighted_investment : ℕ := B_investment * B_months
def total_weighted_investment : ℕ := A_weighted_investment + B_weighted_investment

-- Prove A's share of the profit
theorem A_share_of_profit : (A_weighted_investment / total_weighted_investment : ℚ) * total_profit = 50 := by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_A_share_of_profit_l1810_181017


namespace NUMINAMATH_GPT_distribute_coins_l1810_181027

theorem distribute_coins (x y : ℕ) (h₁ : x + y = 16) (h₂ : x^2 - y^2 = 16 * (x - y)) :
  x = 8 ∧ y = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_distribute_coins_l1810_181027


namespace NUMINAMATH_GPT_solve_for_y_l1810_181018

theorem solve_for_y (y : ℤ) : (2 / 3 - 3 / 5 : ℚ) = 5 / y → y = 75 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1810_181018


namespace NUMINAMATH_GPT_dan_has_more_balloons_l1810_181035

-- Constants representing the number of balloons Dan and Tim have
def dans_balloons : ℝ := 29.0
def tims_balloons : ℝ := 4.142857143

-- Theorem: The ratio of Dan's balloons to Tim's balloons is 7
theorem dan_has_more_balloons : dans_balloons / tims_balloons = 7 := 
by
  sorry

end NUMINAMATH_GPT_dan_has_more_balloons_l1810_181035


namespace NUMINAMATH_GPT_cade_marbles_left_l1810_181004

theorem cade_marbles_left (initial_marbles : ℕ) (given_away : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 350 → given_away = 175 → remaining_marbles = initial_marbles - given_away → remaining_marbles = 175 :=
by
  intros h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining

end NUMINAMATH_GPT_cade_marbles_left_l1810_181004


namespace NUMINAMATH_GPT_maximize_x_minus_y_plus_z_l1810_181081

-- Define the given condition as a predicate
def given_condition (x y z : ℝ) : Prop :=
  2 * x^2 + y^2 + z^2 = 2 * x - 4 * y + 2 * x * z - 5

-- Define the statement we want to prove
theorem maximize_x_minus_y_plus_z :
  ∃ x y z : ℝ, given_condition x y z ∧ (x - y + z = 4) :=
by
  sorry

end NUMINAMATH_GPT_maximize_x_minus_y_plus_z_l1810_181081


namespace NUMINAMATH_GPT_relationship_among_g_a_0_f_b_l1810_181030

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x + x^2 - 3

theorem relationship_among_g_a_0_f_b (a b : ℝ) (h1 : f a = 0) (h2 : g b = 0) : g a < 0 ∧ 0 < f b :=
by
  -- Function properties are non-trivial and are omitted.
  sorry

end NUMINAMATH_GPT_relationship_among_g_a_0_f_b_l1810_181030


namespace NUMINAMATH_GPT_sum_first_n_terms_arithmetic_sequence_l1810_181047

theorem sum_first_n_terms_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (a 2 + a 4 = 10) ∧ (∀ n : ℕ, a (n + 1) - a n = 2) → 
  (∀ n : ℕ, S n = n^2) := by
  intro h
  sorry

end NUMINAMATH_GPT_sum_first_n_terms_arithmetic_sequence_l1810_181047


namespace NUMINAMATH_GPT_find_integer_n_l1810_181092

theorem find_integer_n : ∃ n : ℤ, 0 ≤ n ∧ n < 151 ∧ (150 * n) % 151 = 93 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_n_l1810_181092


namespace NUMINAMATH_GPT_harkamal_total_payment_l1810_181020

def grapes_quantity : ℕ := 10
def grapes_rate : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_rate : ℕ := 55

def cost_of_grapes : ℕ := grapes_quantity * grapes_rate
def cost_of_mangoes : ℕ := mangoes_quantity * mangoes_rate

def total_amount_paid : ℕ := cost_of_grapes + cost_of_mangoes

theorem harkamal_total_payment : total_amount_paid = 1195 := by
  sorry

end NUMINAMATH_GPT_harkamal_total_payment_l1810_181020


namespace NUMINAMATH_GPT_quadrilateral_area_inequality_equality_condition_l1810_181046

theorem quadrilateral_area_inequality 
  (a b c d S : ℝ) 
  (hS : S = 0.5 * a * c + 0.5 * b * d) 
  : S ≤ 0.5 * (a * c + b * d) :=
sorry

theorem equality_condition 
  (a b c d S : ℝ) 
  (hS : S = 0.5 * a * c + 0.5 * b * d)
  (h_perpendicular : ∃ (α β : ℝ), α = 90 ∧ β = 90) 
  : S = 0.5 * (a * c + b * d) :=
sorry

end NUMINAMATH_GPT_quadrilateral_area_inequality_equality_condition_l1810_181046


namespace NUMINAMATH_GPT_phase_shift_correct_l1810_181053

-- Given the function y = 3 * sin (x - π / 5)
-- We need to prove that the phase shift is π / 5.

theorem phase_shift_correct :
  ∀ x : ℝ, 3 * Real.sin (x - Real.pi / 5) = 3 * Real.sin (x - C) →
  C = Real.pi / 5 :=
by
  sorry

end NUMINAMATH_GPT_phase_shift_correct_l1810_181053


namespace NUMINAMATH_GPT_sum_of_roots_of_equation_l1810_181048

theorem sum_of_roots_of_equation : 
  (∀ x, 5 = (x^3 - 2*x^2 - 8*x) / (x + 2)) → 
  (∃ x1 x2, (5 = x1) ∧ (5 = x2) ∧ (x1 + x2 = 4)) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_equation_l1810_181048


namespace NUMINAMATH_GPT_conditionD_necessary_not_sufficient_l1810_181022

variable (a b : ℝ)

-- Define each of the conditions as separate variables
def conditionA : Prop := |a| < |b|
def conditionB : Prop := 2 * a < 2 * b
def conditionC : Prop := a < b - 1
def conditionD : Prop := a < b + 1

-- Prove that condition D is necessary but not sufficient for a < b
theorem conditionD_necessary_not_sufficient : conditionD a b → (¬ conditionA a b ∨ ¬ conditionB a b ∨ ¬ conditionC a b) ∧ ¬(conditionD a b ↔ a < b) :=
by sorry

end NUMINAMATH_GPT_conditionD_necessary_not_sufficient_l1810_181022


namespace NUMINAMATH_GPT_possible_values_of_a_l1810_181042

def A (a : ℝ) : Set ℝ := { x | 0 < x ∧ x < a }
def B : Set ℝ := { x | 1 < x ∧ x < 2 }
def complement_R (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem possible_values_of_a (a : ℝ) :
  (∃ x, x ∈ A a) →
  B ⊆ complement_R (A a) →
  0 < a ∧ a ≤ 1 :=
by 
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l1810_181042


namespace NUMINAMATH_GPT_probability_of_defective_l1810_181083

theorem probability_of_defective (p_first_grade p_second_grade : ℝ) (h_fg : p_first_grade = 0.65) (h_sg : p_second_grade = 0.3) : (1 - (p_first_grade + p_second_grade) = 0.05) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_defective_l1810_181083


namespace NUMINAMATH_GPT_bike_cost_l1810_181003

theorem bike_cost (h1: 8 > 0) (h2: 35 > 0) (weeks_in_month: ℕ := 4) (saved: ℕ := 720):
  let hourly_wage := 8
  let weekly_hours := 35
  let weekly_earnings := weekly_hours * hourly_wage
  let monthly_earnings := weekly_earnings * weeks_in_month
  let cost_of_bike := monthly_earnings - saved
  cost_of_bike = 400 :=
by
  sorry

end NUMINAMATH_GPT_bike_cost_l1810_181003


namespace NUMINAMATH_GPT_exists_Q_R_l1810_181012

noncomputable def P (x : ℚ) : ℚ := x^4 + x^3 + x^2 + x + 1

theorem exists_Q_R : ∃ (Q R : Polynomial ℚ), 
  (Q.degree > 0 ∧ R.degree > 0) ∧
  (∀ (y : ℚ), (Q.eval y) * (R.eval y) = P (5 * y^2)) :=
sorry

end NUMINAMATH_GPT_exists_Q_R_l1810_181012


namespace NUMINAMATH_GPT_flowers_per_bouquet_l1810_181040

-- Defining the problem parameters
def total_flowers : ℕ := 66
def wilted_flowers : ℕ := 10
def num_bouquets : ℕ := 7

-- The goal is to prove that the number of flowers per bouquet is 8
theorem flowers_per_bouquet :
  (total_flowers - wilted_flowers) / num_bouquets = 8 :=
by
  sorry

end NUMINAMATH_GPT_flowers_per_bouquet_l1810_181040


namespace NUMINAMATH_GPT_triangle_least_perimeter_l1810_181009

noncomputable def least_perimeter_of_triangle : ℕ :=
  let a := 7
  let b := 17
  let c := 13
  a + b + c

theorem triangle_least_perimeter :
  let a := 7
  let b := 17
  let c := 13
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  4 ∣ (a^2 + b^2 + c^2) - 2 * c^2 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) →
  least_perimeter_of_triangle = 37 :=
by
  intros _ _ _ h
  sorry

end NUMINAMATH_GPT_triangle_least_perimeter_l1810_181009


namespace NUMINAMATH_GPT_mean_of_six_numbers_l1810_181063

theorem mean_of_six_numbers (sum : ℚ) (h : sum = 1/3) : (sum / 6 = 1/18) :=
by
  sorry

end NUMINAMATH_GPT_mean_of_six_numbers_l1810_181063


namespace NUMINAMATH_GPT_number_of_strawberries_in_each_basket_l1810_181096

variable (x : ℕ) (Lilibeth_picks : 6 * x)
variable (total_strawberries : 4 * 6 * x = 1200)

theorem number_of_strawberries_in_each_basket : x = 50 := by
  sorry

end NUMINAMATH_GPT_number_of_strawberries_in_each_basket_l1810_181096


namespace NUMINAMATH_GPT_correct_systematic_sampling_l1810_181033

-- Definitions for conditions in a)
def num_bags := 50
def num_selected := 5
def interval := num_bags / num_selected

-- We encode the systematic sampling selection process
def systematic_sampling (n : Nat) (start : Nat) (interval: Nat) (count : Nat) : List Nat :=
  List.range count |>.map (λ i => start + i * interval)

-- Theorem to prove that the selection of bags should have an interval of 10
theorem correct_systematic_sampling :
  ∃ (start : Nat), systematic_sampling num_selected start interval num_selected = [7, 17, 27, 37, 47] := sorry

end NUMINAMATH_GPT_correct_systematic_sampling_l1810_181033


namespace NUMINAMATH_GPT_regular_ticket_price_l1810_181016

variable (P : ℝ) -- Define the regular ticket price as a real number

-- Condition: Travis pays $1400 for his ticket after a 30% discount on a regular price P
axiom h : 0.70 * P = 1400

-- Theorem statement: Proving that the regular ticket price P equals $2000
theorem regular_ticket_price : P = 2000 :=
by 
  sorry

end NUMINAMATH_GPT_regular_ticket_price_l1810_181016


namespace NUMINAMATH_GPT_find_de_l1810_181072

namespace MagicSquare

variables (a b c d e : ℕ)

-- Hypotheses based on the conditions provided.
axiom H1 : 20 + 15 + a = 57
axiom H2 : 25 + b + a = 57
axiom H3 : 18 + c + a = 57
axiom H4 : 20 + c + b = 57
axiom H5 : d + c + a = 57
axiom H6 : d + e + 18 = 57
axiom H7 : e + 25 + 15 = 57

def magicSum := 57

theorem find_de :
  ∃ d e, d + e = 42 :=
by sorry

end MagicSquare

end NUMINAMATH_GPT_find_de_l1810_181072


namespace NUMINAMATH_GPT_day_of_week_150th_day_of_year_N_minus_1_l1810_181011

/-- Given that the 250th day of year N is a Friday and year N is a leap year,
    prove that the 150th day of year N-1 is a Friday. -/
theorem day_of_week_150th_day_of_year_N_minus_1
  (N : ℕ) 
  (H1 : (250 % 7 = 5) → true)  -- Condition that 250th day is five days after Sunday (Friday).
  (H2 : 366 % 7 = 2)           -- Condition that year N is a leap year with 366 days.
  (H3 : (N - 1) % 7 = (N - 1) % 7) -- Used for year transition check.
  : 150 % 7 = 5 := sorry       -- Proving that the 150th of year N-1 is Friday.

end NUMINAMATH_GPT_day_of_week_150th_day_of_year_N_minus_1_l1810_181011


namespace NUMINAMATH_GPT_existence_of_x2_with_sum_ge_2_l1810_181054

variables (a b c x1 x2 : ℝ) (h_root1 : a * x1^2 + b * x1 + c = 0) (h_x1_pos : x1 > 0)

theorem existence_of_x2_with_sum_ge_2 :
  ∃ x2, (c * x2^2 + b * x2 + a = 0) ∧ (x1 + x2 ≥ 2) :=
sorry

end NUMINAMATH_GPT_existence_of_x2_with_sum_ge_2_l1810_181054


namespace NUMINAMATH_GPT_total_sampled_students_is_80_l1810_181055

-- Given conditions
variables (total_students num_freshmen num_sampled_freshmen : ℕ)
variables (total_students := 2400) (num_freshmen := 600) (num_sampled_freshmen := 20)

-- Define the proportion for stratified sampling.
def stratified_sampling (total_students num_freshmen num_sampled_freshmen total_sampled_students : ℕ) : Prop :=
  num_freshmen / total_students = num_sampled_freshmen / total_sampled_students

-- State the theorem: Prove the total number of students to be sampled from the entire school is 80.
theorem total_sampled_students_is_80 : ∃ n, stratified_sampling total_students num_freshmen num_sampled_freshmen n ∧ n = 80 := 
sorry

end NUMINAMATH_GPT_total_sampled_students_is_80_l1810_181055


namespace NUMINAMATH_GPT_solve_system_nat_l1810_181073

theorem solve_system_nat (a b c d : ℕ) :
  (a * b = c + d ∧ c * d = a + b) →
  (a = 1 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
  (a = 1 ∧ b = 5 ∧ c = 3 ∧ d = 2) ∨
  (a = 5 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
  (a = 5 ∧ b = 1 ∧ c = 3 ∧ d = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2) ∨
  (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5) ∨
  (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 1) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∨
  (a = 3 ∧ b = 2 ∧ c = 5 ∧ d = 1) :=
sorry

end NUMINAMATH_GPT_solve_system_nat_l1810_181073


namespace NUMINAMATH_GPT_fruit_salad_weight_l1810_181036

theorem fruit_salad_weight (melon berries : ℝ) (h_melon : melon = 0.25) (h_berries : berries = 0.38) : melon + berries = 0.63 :=
by
  sorry

end NUMINAMATH_GPT_fruit_salad_weight_l1810_181036


namespace NUMINAMATH_GPT_triangular_difference_30_28_l1810_181061

noncomputable def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

theorem triangular_difference_30_28 : triangular 30 - triangular 28 = 59 :=
by
  sorry

end NUMINAMATH_GPT_triangular_difference_30_28_l1810_181061


namespace NUMINAMATH_GPT_no_solution_if_and_only_if_l1810_181024

theorem no_solution_if_and_only_if (n : ℝ) : 
  ¬ ∃ (x y z : ℝ), 
    (n * x + y = 1) ∧ 
    (n * y + z = 1) ∧ 
    (x + n * z = 1) ↔ n = -1 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_if_and_only_if_l1810_181024


namespace NUMINAMATH_GPT_arccos_neg_one_eq_pi_l1810_181043

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := 
by
  sorry

end NUMINAMATH_GPT_arccos_neg_one_eq_pi_l1810_181043


namespace NUMINAMATH_GPT_min_t_of_BE_CF_l1810_181051

theorem min_t_of_BE_CF (A B C E F: ℝ)
  (hE_midpoint_AC : ∃ D, D = (A + C) / 2 ∧ E = D)
  (hF_midpoint_AB : ∃ D, D = (A + B) / 2 ∧ F = D)
  (h_AB_AC_ratio : B - A = 2 / 3 * (C - A)) :
  ∃ t : ℝ, t = 7 / 8 ∧ ∀ (BE CF : ℝ), BE = dist B E ∧ CF = dist C F → BE / CF < t := by
  sorry

end NUMINAMATH_GPT_min_t_of_BE_CF_l1810_181051


namespace NUMINAMATH_GPT_locus_midpoint_l1810_181090

-- Conditions
def hyperbola_eq (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1

def perpendicular_rays (OA OB : ℝ × ℝ) : Prop := (OA.1 * OB.1 + OA.2 * OB.2) = 0 -- Dot product zero for perpendicularity

-- Given the hyperbola and perpendicularity conditions, prove the locus equation
theorem locus_midpoint (x y : ℝ) :
  (∃ A B : ℝ × ℝ, hyperbola_eq A.1 A.2 ∧ hyperbola_eq B.1 B.2 ∧ perpendicular_rays A B ∧
  x = (A.1 + B.1) / 2 ∧ y = (A.2 + B.2) / 2) → 3 * (4 * x^2 - y^2)^2 = 4 * (16 * x^2 + y^2) :=
sorry

end NUMINAMATH_GPT_locus_midpoint_l1810_181090


namespace NUMINAMATH_GPT_mr_bhaskar_tour_duration_l1810_181078

theorem mr_bhaskar_tour_duration :
  ∃ d : Nat, 
    (d > 0) ∧ 
    (∃ original_daily_expense new_daily_expense : ℕ,
      original_daily_expense = 360 / d ∧
      new_daily_expense = original_daily_expense - 3 ∧
      360 = new_daily_expense * (d + 4)) ∧
      d = 20 :=
by
  use 20
  -- Here would come the proof steps to verify the conditions and reach the conclusion.
  sorry

end NUMINAMATH_GPT_mr_bhaskar_tour_duration_l1810_181078


namespace NUMINAMATH_GPT_mosquito_drops_per_feed_l1810_181041

-- Defining the constants and conditions.
def drops_per_liter : ℕ := 5000
def liters_to_die : ℕ := 3
def mosquitoes_to_kill : ℕ := 750

-- The assertion we want to prove.
theorem mosquito_drops_per_feed :
  (drops_per_liter * liters_to_die) / mosquitoes_to_kill = 20 :=
by
  sorry

end NUMINAMATH_GPT_mosquito_drops_per_feed_l1810_181041


namespace NUMINAMATH_GPT_person_speed_in_kmph_l1810_181013

noncomputable def speed_calculation (distance_meters : ℕ) (time_minutes : ℕ) : ℝ :=
  let distance_km := (distance_meters : ℝ) / 1000
  let time_hours := (time_minutes : ℝ) / 60
  distance_km / time_hours

theorem person_speed_in_kmph :
  speed_calculation 1080 12 = 5.4 :=
by
  sorry

end NUMINAMATH_GPT_person_speed_in_kmph_l1810_181013


namespace NUMINAMATH_GPT_num_int_values_satisfying_inequality_l1810_181010

theorem num_int_values_satisfying_inequality (x : ℤ) :
  (x^2 < 9 * x) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8) := 
sorry

end NUMINAMATH_GPT_num_int_values_satisfying_inequality_l1810_181010


namespace NUMINAMATH_GPT_sum_first_12_terms_l1810_181025

theorem sum_first_12_terms (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n * a n)
  (h2 : a 6 + a 7 = 18) : 
  S 12 = 108 :=
sorry

end NUMINAMATH_GPT_sum_first_12_terms_l1810_181025


namespace NUMINAMATH_GPT_scaling_matrix_unique_l1810_181088

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

noncomputable def matrix_N : Matrix (Fin 4) (Fin 4) ℝ := ![![3, 0, 0, 0], ![0, 3, 0, 0], ![0, 0, 3, 0], ![0, 0, 0, 3]]

theorem scaling_matrix_unique (N : Matrix (Fin 4) (Fin 4) ℝ) :
  (∀ (w : Fin 4 → ℝ), N.mulVec w = 3 • w) → N = matrix_N :=
by
  intros h
  sorry

end NUMINAMATH_GPT_scaling_matrix_unique_l1810_181088


namespace NUMINAMATH_GPT_positive_divisors_8_fact_l1810_181028

-- Factorial function definition
def factorial : Nat → Nat
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Function to compute the number of divisors from prime factors
def numDivisors (factors : List (Nat × Nat)) : Nat :=
  factors.foldl (fun acc (p, k) => acc * (k + 1)) 1

-- Known prime factorization of 8!
noncomputable def factors_8_fact : List (Nat × Nat) :=
  [(2, 7), (3, 2), (5, 1), (7, 1)]

-- Theorem statement
theorem positive_divisors_8_fact : numDivisors factors_8_fact = 96 :=
  sorry

end NUMINAMATH_GPT_positive_divisors_8_fact_l1810_181028


namespace NUMINAMATH_GPT_square_side_length_exists_l1810_181015

theorem square_side_length_exists
    (k : ℕ)
    (n : ℕ)
    (h_side_length_condition : n * n = k * (k - 7))
    (h_grid_lines : k > 7) :
    n = 12 ∨ n = 24 :=
by sorry

end NUMINAMATH_GPT_square_side_length_exists_l1810_181015


namespace NUMINAMATH_GPT_mod_equiv_solution_l1810_181084

theorem mod_equiv_solution (a b : ℤ) (n : ℤ) 
  (h₁ : a ≡ 22 [ZMOD 50])
  (h₂ : b ≡ 78 [ZMOD 50])
  (h₃ : 150 ≤ n ∧ n ≤ 201)
  (h₄ : n = 194) :
  a - b ≡ n [ZMOD 50] :=
by
  sorry

end NUMINAMATH_GPT_mod_equiv_solution_l1810_181084


namespace NUMINAMATH_GPT_cos_pi_over_2_minus_l1810_181031

theorem cos_pi_over_2_minus (A : ℝ) (h : Real.sin A = 1 / 2) : Real.cos (3 * Real.pi / 2 - A) = -1 / 2 :=
  sorry

end NUMINAMATH_GPT_cos_pi_over_2_minus_l1810_181031


namespace NUMINAMATH_GPT_min_disks_required_l1810_181034

/-- A structure to hold information about the file storage problem -/
structure FileStorageConditions where
  total_files : ℕ
  disk_capacity : ℝ
  num_files_1_6MB : ℕ
  num_files_1MB : ℕ
  num_files_0_5MB : ℕ

/-- Define specific conditions given in the problem -/
def storage_conditions : FileStorageConditions := {
  total_files := 42,
  disk_capacity := 2.88,
  num_files_1_6MB := 8,
  num_files_1MB := 16,
  num_files_0_5MB := 18 -- Derived from total_files - num_files_1_6MB - num_files_1MB
}

/-- Theorem stating the minimum number of disks required to store all files is 16 -/
theorem min_disks_required (c : FileStorageConditions)
  (h1 : c.total_files = 42)
  (h2 : c.disk_capacity = 2.88)
  (h3 : c.num_files_1_6MB = 8)
  (h4 : c.num_files_1MB = 16)
  (h5 : c.num_files_0_5MB = 18) :
  ∃ n : ℕ, n = 16 := by
  sorry

end NUMINAMATH_GPT_min_disks_required_l1810_181034


namespace NUMINAMATH_GPT_simplify_fraction_l1810_181068

variable (x y : ℝ)

theorem simplify_fraction (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 * x^2 / y) * (y^2 / (2 * x)) = 3 * x * y / 2 :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1810_181068


namespace NUMINAMATH_GPT_number_of_students_l1810_181091

/-- 
We are given that 36 students are selected from three grades: 
15 from the first grade, 12 from the second grade, and the rest from the third grade. 
Additionally, there are 900 students in the third grade.
We need to prove: the total number of students in the high school is 3600
-/
theorem number_of_students (x y z : ℕ) (s_total : ℕ) (x_sel : ℕ) (y_sel : ℕ) (z_students : ℕ) 
  (h1 : x_sel = 15) 
  (h2 : y_sel = 12) 
  (h3 : x_sel + y_sel + (s_total - (x_sel + y_sel)) = s_total) 
  (h4 : s_total = 36) 
  (h5 : z_students = 900) 
  (h6 : (s_total - (x_sel + y_sel)) = 9) 
  (h7 : 9 / 900 = 1 / 100) : 
  (36 * 100 = 3600) :=
by sorry

end NUMINAMATH_GPT_number_of_students_l1810_181091


namespace NUMINAMATH_GPT_Julia_played_kids_on_Monday_l1810_181075

theorem Julia_played_kids_on_Monday
  (t : ℕ) (w : ℕ) (h1 : t = 18) (h2 : w = 97) (h3 : t + m = 33) :
  ∃ m : ℕ, m = 15 :=
by
  sorry

end NUMINAMATH_GPT_Julia_played_kids_on_Monday_l1810_181075


namespace NUMINAMATH_GPT_larger_cookie_sugar_l1810_181000

theorem larger_cookie_sugar :
  let initial_cookies := 40
  let initial_sugar_per_cookie := 1 / 8
  let total_sugar := initial_cookies * initial_sugar_per_cookie
  let larger_cookies := 25
  let sugar_per_larger_cookie := total_sugar / larger_cookies
  sugar_per_larger_cookie = 1 / 5 := by
sorry

end NUMINAMATH_GPT_larger_cookie_sugar_l1810_181000


namespace NUMINAMATH_GPT_smallest_natural_with_properties_l1810_181038

theorem smallest_natural_with_properties :
  ∃ n : ℕ, (∃ N : ℕ, n = 10 * N + 6) ∧ 4 * (10 * N + 6) = 6 * 10^(5 : ℕ) + N ∧ n = 153846 := sorry

end NUMINAMATH_GPT_smallest_natural_with_properties_l1810_181038


namespace NUMINAMATH_GPT_prob_at_least_one_head_l1810_181057

theorem prob_at_least_one_head (n : ℕ) (hn : n = 3) : 
  1 - (1 / (2^n)) = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_prob_at_least_one_head_l1810_181057


namespace NUMINAMATH_GPT_Adam_teaches_students_l1810_181097

-- Define the conditions
def students_first_year : ℕ := 40
def students_per_year : ℕ := 50
def total_years : ℕ := 10
def remaining_years : ℕ := total_years - 1

-- Define the statement we are proving
theorem Adam_teaches_students (total_students : ℕ) :
  total_students = students_first_year + (students_per_year * remaining_years) :=
sorry

end NUMINAMATH_GPT_Adam_teaches_students_l1810_181097


namespace NUMINAMATH_GPT_even_increasing_ordering_l1810_181077

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

-- Theorem to prove
theorem even_increasing_ordering (h_even : is_even_function f) (h_increasing : is_increasing_on_pos f) : 
  f 1 < f (-2) ∧ f (-2) < f 3 :=
by
  sorry

end NUMINAMATH_GPT_even_increasing_ordering_l1810_181077


namespace NUMINAMATH_GPT_emery_total_alteration_cost_l1810_181062

-- Definition of the initial conditions
def num_pairs_of_shoes := 17
def cost_per_shoe := 29
def shoes_per_pair := 2

-- Proving the total cost
theorem emery_total_alteration_cost : num_pairs_of_shoes * shoes_per_pair * cost_per_shoe = 986 := by
  sorry

end NUMINAMATH_GPT_emery_total_alteration_cost_l1810_181062


namespace NUMINAMATH_GPT_three_number_relationship_l1810_181007

theorem three_number_relationship :
  let a := (0.7 : ℝ) ^ 6
  let b := 6 ^ (0.7 : ℝ)
  let c := Real.log 6 / Real.log 0.7
  c < a ∧ a < b :=
sorry

end NUMINAMATH_GPT_three_number_relationship_l1810_181007


namespace NUMINAMATH_GPT_fifteenth_term_is_correct_l1810_181089

-- Define the initial conditions of the arithmetic sequence
def firstTerm : ℕ := 4
def secondTerm : ℕ := 9

-- Calculate the common difference
def commonDifference : ℕ := secondTerm - firstTerm

-- Define the nth term formula of the arithmetic sequence
def nthTerm (a d n : ℕ) : ℕ := a + (n - 1) * d

-- The main statement: proving that the 15th term of the given sequence is 74
theorem fifteenth_term_is_correct : nthTerm firstTerm commonDifference 15 = 74 :=
by
  sorry

end NUMINAMATH_GPT_fifteenth_term_is_correct_l1810_181089


namespace NUMINAMATH_GPT_move_3m_left_is_neg_3m_l1810_181082

-- Define the notation for movements
def move_right (distance : Int) : Int := distance
def move_left (distance : Int) : Int := -distance

-- Define the specific condition
def move_1m_right : Int := move_right 1

-- Define the assertion for moving 3m to the left
def move_3m_left : Int := move_left 3

-- State the proof problem
theorem move_3m_left_is_neg_3m : move_3m_left = -3 := by
  unfold move_3m_left
  unfold move_left
  rfl

end NUMINAMATH_GPT_move_3m_left_is_neg_3m_l1810_181082


namespace NUMINAMATH_GPT_unique_solution_l1810_181001

def s (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem unique_solution (m n : ℕ) (h : n * (n + 1) = 3 ^ m + s n + 1182) : (m, n) = (0, 34) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l1810_181001


namespace NUMINAMATH_GPT_center_of_image_circle_l1810_181066

def point := ℝ × ℝ

def reflect_about_y_eq_neg_x (p : point) : point :=
  let (a, b) := p
  (-b, -a)

theorem center_of_image_circle :
  reflect_about_y_eq_neg_x (8, -3) = (3, -8) :=
by
  sorry

end NUMINAMATH_GPT_center_of_image_circle_l1810_181066


namespace NUMINAMATH_GPT_total_items_in_quiz_l1810_181006

theorem total_items_in_quiz (score_percent : ℝ) (mistakes : ℕ) (total_items : ℕ) 
  (h1 : score_percent = 80) 
  (h2 : mistakes = 5) :
  total_items = 25 :=
sorry

end NUMINAMATH_GPT_total_items_in_quiz_l1810_181006


namespace NUMINAMATH_GPT_isosceles_triangle_base_angle_l1810_181026

theorem isosceles_triangle_base_angle (a b c : ℝ) (h_triangle_isosceles : a = b ∨ b = c ∨ c = a)
  (h_angle_sum : a + b + c = 180) (h_one_angle : a = 50 ∨ b = 50 ∨ c = 50) :
  a = 50 ∨ b = 50 ∨ c = 50 ∨ a = 65 ∨ b = 65 ∨ c = 65 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angle_l1810_181026


namespace NUMINAMATH_GPT_quadratic_one_solution_l1810_181059

theorem quadratic_one_solution (b d : ℝ) (h1 : b + d = 35) (h2 : b < d) (h3 : (24 : ℝ)^2 - 4 * b * d = 0) :
  (b, d) = (35 - Real.sqrt 649 / 2, 35 + Real.sqrt 649 / 2) := 
sorry

end NUMINAMATH_GPT_quadratic_one_solution_l1810_181059


namespace NUMINAMATH_GPT_sum_divided_among_xyz_l1810_181002

noncomputable def total_amount (x_share y_share z_share : ℝ) : ℝ :=
  x_share + y_share + z_share

theorem sum_divided_among_xyz
    (x_share : ℝ) (y_share : ℝ) (z_share : ℝ)
    (y_gets_45_paisa : y_share = 0.45 * x_share)
    (z_gets_50_paisa : z_share = 0.50 * x_share)
    (y_share_is_18 : y_share = 18) :
    total_amount x_share y_share z_share = 78 := by
  sorry

end NUMINAMATH_GPT_sum_divided_among_xyz_l1810_181002


namespace NUMINAMATH_GPT_percent_decrease_correct_l1810_181094

def original_price_per_pack : ℚ := 7 / 3
def promotional_price_per_pack : ℚ := 8 / 4
def percent_decrease_in_price (old_price new_price : ℚ) : ℚ := 
  ((old_price - new_price) / old_price) * 100

theorem percent_decrease_correct :
  percent_decrease_in_price original_price_per_pack promotional_price_per_pack = 14 := by
  sorry

end NUMINAMATH_GPT_percent_decrease_correct_l1810_181094


namespace NUMINAMATH_GPT_number_of_positive_integers_l1810_181019

theorem number_of_positive_integers (n : ℕ) : ∃! k : ℕ, k = 5 ∧
  (∀ n : ℕ, (1 ≤ n) → (12 % (n + 1) = 0)) :=
sorry

end NUMINAMATH_GPT_number_of_positive_integers_l1810_181019


namespace NUMINAMATH_GPT_bridge_length_correct_l1810_181032

def train_length : ℕ := 256
def train_speed_kmh : ℕ := 72
def crossing_time : ℕ := 20

noncomputable def convert_speed (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600 -- Conversion from km/h to m/s

noncomputable def bridge_length (train_length : ℕ) (speed_m : ℕ) (time_s : ℕ) : ℕ :=
  (speed_m * time_s) - train_length

theorem bridge_length_correct :
  bridge_length train_length (convert_speed train_speed_kmh) crossing_time = 144 :=
by
  sorry

end NUMINAMATH_GPT_bridge_length_correct_l1810_181032


namespace NUMINAMATH_GPT_total_tank_capacity_l1810_181071

-- Definitions based on conditions
def initial_condition (w c : ℝ) : Prop := w / c = 1 / 3
def after_adding_five (w c : ℝ) : Prop := (w + 5) / c = 1 / 2

-- The problem statement
theorem total_tank_capacity (w c : ℝ) (h1 : initial_condition w c) (h2 : after_adding_five w c) : c = 30 :=
sorry

end NUMINAMATH_GPT_total_tank_capacity_l1810_181071


namespace NUMINAMATH_GPT_star_value_l1810_181093

def star (a b : ℝ) : ℝ := a^3 + 3 * a^2 * b + 3 * a * b^2 + b^3

theorem star_value : star 3 2 = 125 :=
by
  sorry

end NUMINAMATH_GPT_star_value_l1810_181093


namespace NUMINAMATH_GPT_alpha_nonneg_integer_l1810_181080

theorem alpha_nonneg_integer (α : ℝ) 
  (h : ∀ n : ℕ, ∃ k : ℕ, n = k * α) : α ≥ 0 ∧ ∃ k : ℤ, α = k := 
sorry

end NUMINAMATH_GPT_alpha_nonneg_integer_l1810_181080


namespace NUMINAMATH_GPT_sum_lent_l1810_181039

theorem sum_lent (P : ℝ) (R : ℝ := 4) (T : ℝ := 8) (I : ℝ) (H1 : I = P - 204) (H2 : I = (P * R * T) / 100) : 
  P = 300 :=
by 
  sorry

end NUMINAMATH_GPT_sum_lent_l1810_181039


namespace NUMINAMATH_GPT_james_meditation_time_is_30_l1810_181067

noncomputable def james_meditation_time_per_session 
  (sessions_per_day : ℕ) 
  (days_per_week : ℕ) 
  (hours_per_week : ℕ) 
  (minutes_per_hour : ℕ) : ℕ :=
  (hours_per_week * minutes_per_hour) / (sessions_per_day * days_per_week)

theorem james_meditation_time_is_30
  (sessions_per_day : ℕ) 
  (days_per_week : ℕ) 
  (hours_per_week : ℕ) 
  (minutes_per_hour : ℕ) 
  (h_sessions : sessions_per_day = 2) 
  (h_days : days_per_week = 7) 
  (h_hours : hours_per_week = 7) 
  (h_minutes : minutes_per_hour = 60) : 
  james_meditation_time_per_session sessions_per_day days_per_week hours_per_week minutes_per_hour = 30 := by
  sorry

end NUMINAMATH_GPT_james_meditation_time_is_30_l1810_181067


namespace NUMINAMATH_GPT_Steven_more_than_Jill_l1810_181058

variable (Jill Jake Steven : ℕ)

def Jill_peaches : Jill = 87 := by sorry
def Jake_peaches_more : Jake = Jill + 13 := by sorry
def Steven_peaches_more : Steven = Jake + 5 := by sorry

theorem Steven_more_than_Jill : Steven - Jill = 18 := by
  -- Proof steps to be filled
  sorry

end NUMINAMATH_GPT_Steven_more_than_Jill_l1810_181058


namespace NUMINAMATH_GPT_average_of_three_quantities_l1810_181005

theorem average_of_three_quantities (a b c d e : ℝ) 
  (h_avg_5 : (a + b + c + d + e) / 5 = 11)
  (h_avg_2 : (d + e) / 2 = 21.5) :
  (a + b + c) / 3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_of_three_quantities_l1810_181005


namespace NUMINAMATH_GPT_area_of_YZW_l1810_181085

-- Definitions from conditions
def area_of_triangle_XYZ := 36
def base_XY := 8
def base_YW := 32

-- The theorem to prove
theorem area_of_YZW : 1/2 * base_YW * (2 * area_of_triangle_XYZ / base_XY) = 144 := 
by
  -- Placeholder for the proof  
  sorry

end NUMINAMATH_GPT_area_of_YZW_l1810_181085


namespace NUMINAMATH_GPT_cubic_eq_factorization_l1810_181021

theorem cubic_eq_factorization (a b c : ℝ) :
  (∃ m n : ℝ, (x^3 + a * x^2 + b * x + c = (x^2 + m) * (x + n))) ↔ (c = a * b) :=
sorry

end NUMINAMATH_GPT_cubic_eq_factorization_l1810_181021


namespace NUMINAMATH_GPT_domain_of_log_base_half_l1810_181076

noncomputable def domain_log_base_half : Set ℝ := { x : ℝ | x > 5 }

theorem domain_of_log_base_half :
  (∀ x : ℝ, x > 5 ↔ x - 5 > 0) →
  (domain_log_base_half = { x : ℝ | x - 5 > 0 }) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_log_base_half_l1810_181076


namespace NUMINAMATH_GPT_range_of_f_l1810_181079

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((1 - x) / (1 + x)) + Real.arctan (2 * x)

theorem range_of_f : Set.Ioo (-(Real.pi / 2)) (Real.pi / 2) = Set.range f :=
  sorry

end NUMINAMATH_GPT_range_of_f_l1810_181079


namespace NUMINAMATH_GPT_binom_np_p_div_p4_l1810_181099

theorem binom_np_p_div_p4 (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (h3 : 3 < p) (hn : n % p = 1) : p^4 ∣ Nat.choose (n * p) p - n := 
sorry

end NUMINAMATH_GPT_binom_np_p_div_p4_l1810_181099


namespace NUMINAMATH_GPT_distinct_solution_count_l1810_181045

theorem distinct_solution_count
  (n : ℕ)
  (x y : ℕ)
  (h1 : x ≠ y)
  (h2 : x ≠ 2 * y)
  (h3 : y ≠ 2 * x)
  (h4 : x^2 - x * y + y^2 = n) :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card ≥ 12 ∧ ∀ (a b : ℕ), (a, b) ∈ pairs → a^2 - a * b + b^2 = n :=
sorry

end NUMINAMATH_GPT_distinct_solution_count_l1810_181045


namespace NUMINAMATH_GPT_find_j_l1810_181050

def original_number (a b k : ℕ) : ℕ := 10 * a + b
def sum_of_digits (a b : ℕ) : ℕ := a + b
def modified_number (b a : ℕ) : ℕ := 20 * b + a

theorem find_j
  (a b k j : ℕ)
  (h1 : original_number a b k = k * sum_of_digits a b)
  (h2 : modified_number b a = j * sum_of_digits a b) :
  j = (199 + k) / 10 :=
sorry

end NUMINAMATH_GPT_find_j_l1810_181050


namespace NUMINAMATH_GPT_simplify_fraction_l1810_181060

theorem simplify_fraction : 
    (3 ^ 1011 + 3 ^ 1009) / (3 ^ 1011 - 3 ^ 1009) = 5 / 4 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1810_181060


namespace NUMINAMATH_GPT_fisherman_sale_l1810_181074

/-- 
If the price of the radio is both the 4th highest price and the 13th lowest price 
among the prices of the fishes sold at a sale, then the total number of fishes 
sold at the fisherman sale is 16. 
-/
theorem fisherman_sale (h4_highest : ∃ price : ℕ, ∀ p : ℕ, p > price → p ∈ {a | a ≠ price} ∧ p > 3)
                       (h13_lowest : ∃ price : ℕ, ∀ p : ℕ, p < price → p ∈ {a | a ≠ price} ∧ p < 13) :
  ∃ n : ℕ, n = 16 :=
sorry

end NUMINAMATH_GPT_fisherman_sale_l1810_181074


namespace NUMINAMATH_GPT_expansion_simplification_l1810_181029

variable (x y : ℝ)

theorem expansion_simplification :
  let a := 3 * x + 4
  let b := 2 * x + 6 * y + 7
  a * b = 6 * x ^ 2 + 18 * x * y + 29 * x + 24 * y + 28 :=
by
  sorry

end NUMINAMATH_GPT_expansion_simplification_l1810_181029


namespace NUMINAMATH_GPT_evaluate_expression_l1810_181086

theorem evaluate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5) + 1) = 107 :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1810_181086


namespace NUMINAMATH_GPT_susie_large_rooms_count_l1810_181049

theorem susie_large_rooms_count:
  (∀ small_rooms medium_rooms large_rooms : ℕ,  
    (small_rooms = 4) → 
    (medium_rooms = 3) → 
    (large_rooms = x) → 
    (225 = small_rooms * 15 + medium_rooms * 25 + large_rooms * 35) → 
    x = 2) :=
by
  intros small_rooms medium_rooms large_rooms
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_susie_large_rooms_count_l1810_181049


namespace NUMINAMATH_GPT_fourth_power_of_cube_third_smallest_prime_l1810_181069

-- Define the third smallest prime number
def third_smallest_prime : Nat := 5

-- Define a function that calculates the fourth power of a number
def fourth_power (x : Nat) : Nat := x * x * x * x

-- Define a function that calculates the cube of a number
def cube (x : Nat) : Nat := x * x * x

-- The proposition stating the fourth power of the cube of the third smallest prime number is 244140625
theorem fourth_power_of_cube_third_smallest_prime : 
  fourth_power (cube third_smallest_prime) = 244140625 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_fourth_power_of_cube_third_smallest_prime_l1810_181069


namespace NUMINAMATH_GPT_math_books_together_l1810_181037

theorem math_books_together (math_books english_books : ℕ) (h_math_books : math_books = 2) (h_english_books : english_books = 2) : 
  ∃ ways, ways = 12 := by
  sorry

end NUMINAMATH_GPT_math_books_together_l1810_181037


namespace NUMINAMATH_GPT_find_X_l1810_181052

theorem find_X : ∃ X : ℝ, 0.60 * X = 0.30 * 800 + 370 ∧ X = 1016.67 := by
  sorry

end NUMINAMATH_GPT_find_X_l1810_181052


namespace NUMINAMATH_GPT_apple_lovers_l1810_181044

theorem apple_lovers :
  ∃ (x y : ℕ), 22 * x = 1430 ∧ 13 * (x + y) = 1430 ∧ y = 45 :=
by
  sorry

end NUMINAMATH_GPT_apple_lovers_l1810_181044


namespace NUMINAMATH_GPT_ticket_difference_l1810_181095

theorem ticket_difference (V G : ℕ) (h1 : V + G = 320) (h2 : 45 * V + 20 * G = 7500) :
  G - V = 232 :=
by
  sorry

end NUMINAMATH_GPT_ticket_difference_l1810_181095


namespace NUMINAMATH_GPT_find_x_l1810_181070

def star (p q : Int × Int) : Int × Int :=
  (p.1 + q.2, p.2 - q.1)

theorem find_x : ∀ (x y : Int), star (x, y) (4, 2) = (5, 4) → x = 3 :=
by
  intros x y h
  -- The statement is correct, just add a placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_x_l1810_181070


namespace NUMINAMATH_GPT_minimum_focal_chord_length_l1810_181023

theorem minimum_focal_chord_length (p : ℝ) (hp : p > 0) :
  ∃ l, (l = 2 * p) ∧ (∀ y x1 x2, y^2 = 2 * p * x1 ∧ y^2 = 2 * p * x2 → l = x2 - x1) := 
sorry

end NUMINAMATH_GPT_minimum_focal_chord_length_l1810_181023


namespace NUMINAMATH_GPT_no_six_consecutive010101_l1810_181065

def unit_digit (n: ℕ) : ℕ := n % 10

def sequence : ℕ → ℕ
| 0     => 1
| 1     => 0
| 2     => 1
| 3     => 0
| 4     => 1
| 5     => 0
| (n + 6) => unit_digit (sequence n + sequence (n + 1) + sequence (n + 2) + sequence (n + 3) + sequence (n + 4) + sequence (n + 5))

theorem no_six_consecutive010101 : ∀ n, ¬ (sequence n = 0 ∧ sequence (n + 1) = 1 ∧ sequence (n + 2) = 0 ∧ sequence (n + 3) = 1 ∧ sequence (n + 4) = 0 ∧ sequence (n + 5) = 1) :=
sorry

end NUMINAMATH_GPT_no_six_consecutive010101_l1810_181065
