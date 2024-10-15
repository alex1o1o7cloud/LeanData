import Mathlib

namespace NUMINAMATH_GPT_initial_teach_count_l1178_117868

theorem initial_teach_count :
  ∃ (x y : ℕ), (x + x * y + (x + x * y) * (y + x * y) = 195) ∧
               (y + x * y + (y + x * y) * (x + x * y) = 192) ∧
               x = 5 ∧ y = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_teach_count_l1178_117868


namespace NUMINAMATH_GPT_similar_triangle_shortest_side_l1178_117875

theorem similar_triangle_shortest_side (a b c : ℕ) (H1 : a^2 + b^2 = c^2) (H2 : a = 15) (H3 : c = 34) (H4 : b = Int.sqrt 931) : 
  ∃ d : ℝ, d = 3 * Int.sqrt 931 ∧ d = 102  :=
by
  sorry

end NUMINAMATH_GPT_similar_triangle_shortest_side_l1178_117875


namespace NUMINAMATH_GPT_calc_expression_l1178_117819

noncomputable def x := (3 + Real.sqrt 5) / 2 -- chosen from one of the roots of the quadratic equation x^2 - 3x + 1

theorem calc_expression (h : x + 1 / x = 3) : 
  (x - 1) ^ 2 + 16 / (x - 1) ^ 2 = 7 + 3 * Real.sqrt 5 := 
by 
  sorry

end NUMINAMATH_GPT_calc_expression_l1178_117819


namespace NUMINAMATH_GPT_f_relation_l1178_117880

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem f_relation :
  f (-Real.pi / 3) > f 1 ∧ f 1 > f (Real.pi / 5) :=
by
  sorry

end NUMINAMATH_GPT_f_relation_l1178_117880


namespace NUMINAMATH_GPT_range_of_a_minus_b_l1178_117879

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : 0 < b ∧ b < 1) : -1 < a - b ∧ a - b < 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_minus_b_l1178_117879


namespace NUMINAMATH_GPT_find_f1_l1178_117814

theorem find_f1 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f (3 * x + 1) = x^2 + 3*x + 2) :
  f 1 = 2 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_find_f1_l1178_117814


namespace NUMINAMATH_GPT_zero_neither_positive_nor_negative_l1178_117831

def is_positive (n : ℤ) : Prop := n > 0
def is_negative (n : ℤ) : Prop := n < 0
def is_rational (n : ℤ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ n = p / q

theorem zero_neither_positive_nor_negative : ¬is_positive 0 ∧ ¬is_negative 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_neither_positive_nor_negative_l1178_117831


namespace NUMINAMATH_GPT_first_worker_time_l1178_117832

theorem first_worker_time
  (T : ℝ) 
  (hT : T ≠ 0)
  (h_comb : (T + 8) / (8 * T) = 1 / 3.428571428571429) :
  T = 8 / 7 :=
by
  sorry

end NUMINAMATH_GPT_first_worker_time_l1178_117832


namespace NUMINAMATH_GPT_Francie_remaining_money_l1178_117840

theorem Francie_remaining_money :
  let weekly_allowance_8_weeks : ℕ := 5 * 8
  let weekly_allowance_6_weeks : ℕ := 6 * 6
  let cash_gift : ℕ := 20
  let initial_total_savings := weekly_allowance_8_weeks + weekly_allowance_6_weeks + cash_gift

  let investment_amount : ℕ := 10
  let expected_return_investment_1 : ℚ := 0.05 * 10
  let expected_return_investment_2 : ℚ := (0.5 * 0.10 * 10) + (0.5 * 0.02 * 10)
  let best_investment_return := max expected_return_investment_1 expected_return_investment_2
  let final_savings_after_investment : ℚ := initial_total_savings - investment_amount + best_investment_return

  let amount_for_clothes : ℚ := final_savings_after_investment / 2
  let remaining_after_clothes := final_savings_after_investment - amount_for_clothes
  let cost_of_video_game : ℕ := 35
  
  remaining_after_clothes.sub cost_of_video_game = 8.30 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Francie_remaining_money_l1178_117840


namespace NUMINAMATH_GPT_total_hours_worked_l1178_117897

-- Definition of the given conditions.
def hours_software : ℕ := 24
def hours_help_user : ℕ := 17
def percentage_other_services : ℚ := 0.4

-- Statement to prove.
theorem total_hours_worked : ∃ (T : ℕ), hours_software + hours_help_user + percentage_other_services * T = T ∧ T = 68 :=
by {
  -- The proof will go here.
  sorry
}

end NUMINAMATH_GPT_total_hours_worked_l1178_117897


namespace NUMINAMATH_GPT_huanhuan_initial_coins_l1178_117898

theorem huanhuan_initial_coins :
  ∃ (H L n : ℕ), H = 7 * L ∧ (H + n = 6 * (L + n)) ∧ (H + 2 * n = 5 * (L + 2 * n)) ∧ H = 70 :=
by
  sorry

end NUMINAMATH_GPT_huanhuan_initial_coins_l1178_117898


namespace NUMINAMATH_GPT_listK_consecutive_integers_count_l1178_117856

-- Given conditions
def listK := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] -- A list K consisting of consecutive integers
def leastInt : Int := -5 -- The least integer in list K
def rangePosInt : Nat := 5 -- The range of the positive integers in list K

-- The theorem to prove
theorem listK_consecutive_integers_count : listK.length = 11 := by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_listK_consecutive_integers_count_l1178_117856


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_proof_l1178_117891

noncomputable def necessary_but_not_sufficient_condition (x : ℝ) : Prop :=
  2 * x ^ 2 - 5 * x - 3 ≥ 0

theorem necessary_but_not_sufficient_condition_proof (x : ℝ) :
  (x < 0 ∨ x > 2) → necessary_but_not_sufficient_condition x :=
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_proof_l1178_117891


namespace NUMINAMATH_GPT_fraction_shaded_is_one_tenth_l1178_117842

theorem fraction_shaded_is_one_tenth :
  ∀ (A L S: ℕ), A = 300 → L = 5 → S = 2 → 
  ((15 * 20 = A) → (A / L = 60) → (60 / S = 30) → (30 / A = 1 / 10)) :=
by sorry

end NUMINAMATH_GPT_fraction_shaded_is_one_tenth_l1178_117842


namespace NUMINAMATH_GPT_find_n_mod_10_l1178_117869

theorem find_n_mod_10 :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n % 10 = (-2023) % 10 ∧ n = 7 :=
sorry

end NUMINAMATH_GPT_find_n_mod_10_l1178_117869


namespace NUMINAMATH_GPT_smallest_natural_number_divisible_l1178_117878

theorem smallest_natural_number_divisible :
  ∃ n : ℕ, (n^2 + 14 * n + 13) % 68 = 0 ∧ 
          ∀ m : ℕ, (m^2 + 14 * m + 13) % 68 = 0 → 21 ≤ m :=
by 
  sorry

end NUMINAMATH_GPT_smallest_natural_number_divisible_l1178_117878


namespace NUMINAMATH_GPT_find_savings_l1178_117813

theorem find_savings (income expenditure : ℕ) (ratio_income_expenditure : ℕ × ℕ) (income_value : income = 40000)
    (ratio_condition : ratio_income_expenditure = (8, 7)) :
    income - expenditure = 5000 :=
by
  sorry

end NUMINAMATH_GPT_find_savings_l1178_117813


namespace NUMINAMATH_GPT_subsequence_sum_q_l1178_117859

theorem subsequence_sum_q (S : Fin 1995 → ℕ) (m : ℕ) (hS_pos : ∀ i : Fin 1995, 0 < S i)
  (hS_sum : (Finset.univ : Finset (Fin 1995)).sum S = m) (h_m_lt : m < 3990) :
  ∀ q : ℕ, 1 ≤ q → q ≤ m → ∃ (I : Finset (Fin 1995)), I.sum S = q := 
sorry

end NUMINAMATH_GPT_subsequence_sum_q_l1178_117859


namespace NUMINAMATH_GPT_clock_equiv_to_square_l1178_117801

theorem clock_equiv_to_square : ∃ h : ℕ, h > 5 ∧ (h^2 - h) % 24 = 0 ∧ h = 9 :=
by 
  let h := 9
  use h
  refine ⟨by decide, by decide, rfl⟩ 

end NUMINAMATH_GPT_clock_equiv_to_square_l1178_117801


namespace NUMINAMATH_GPT_factorization_identity_l1178_117837

theorem factorization_identity (a b : ℝ) : 3 * a^2 + 6 * a * b + 3 * b^2 = 3 * (a + b)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorization_identity_l1178_117837


namespace NUMINAMATH_GPT_arithmetic_mean_is_correct_l1178_117839

variable (x a : ℝ)
variable (hx : x ≠ 0)

theorem arithmetic_mean_is_correct : 
  (1/2 * ((x + 2 * a) / x - 1 + (x - 3 * a) / x + 1)) = (1 - a / (2 * x)) := 
  sorry

end NUMINAMATH_GPT_arithmetic_mean_is_correct_l1178_117839


namespace NUMINAMATH_GPT_maximize_log_power_l1178_117818

theorem maximize_log_power (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (hab : a * b = 100) :
  ∃ x : ℝ, (a ^ (Real.logb 10 b)^2 = 10^x) ∧ x = 32 / 27 :=
by
  sorry

end NUMINAMATH_GPT_maximize_log_power_l1178_117818


namespace NUMINAMATH_GPT_block3_reaches_target_l1178_117850

-- Type representing the position of a block on a 3x7 grid
structure Position where
  row : Nat
  col : Nat
  deriving DecidableEq, Repr

-- Defining the initial positions of blocks
def Block1Start : Position := ⟨2, 2⟩
def Block2Start : Position := ⟨3, 5⟩
def Block3Start : Position := ⟨1, 4⟩

-- The target position in the center of the board
def TargetPosition : Position := ⟨3, 5⟩

-- A function to represent if blocks collide or not
def canMove (current : Position) (target : Position) (blocks : List Position) : Prop :=
  target.row < 3 ∧ target.col < 7 ∧ ¬(target ∈ blocks)

-- Main theorem stating the goal
theorem block3_reaches_target : ∃ (steps : Nat → Position), steps 0 = Block3Start ∧ steps 7 = TargetPosition :=
  sorry

end NUMINAMATH_GPT_block3_reaches_target_l1178_117850


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1178_117824

-- Definitions based on the conditions
def hyperbola (a b : ℝ) : Prop := (a > 0) ∧ (b > 0)

def distance_from_focus_to_asymptote (a b c : ℝ) : Prop :=
  (b^2 * c) / (a^2 + b^2).sqrt = b ∧ b = 2 * Real.sqrt 3

def minimum_distance_point_to_focus (a c : ℝ) : Prop :=
  c - a = 2

def eccentricity (a c e : ℝ) : Prop :=
  e = c / a

-- Problem statement
theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h_hyperbola : hyperbola a b)
  (h_dist_asymptote : distance_from_focus_to_asymptote a b c)
  (h_min_dist_focus : minimum_distance_point_to_focus a c)
  (h_eccentricity : eccentricity a c e) :
  e = 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1178_117824


namespace NUMINAMATH_GPT_find_original_number_l1178_117816

def original_number_divide_multiply (x : ℝ) : Prop :=
  (x / 12) * 24 = x + 36

theorem find_original_number (x : ℝ) (h : original_number_divide_multiply x) : x = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l1178_117816


namespace NUMINAMATH_GPT_horner_method_v3_value_l1178_117829

theorem horner_method_v3_value :
  let f (x : ℤ) := 3 * x^6 + 5 * x^5 + 6 * x^4 + 79 * x^3 - 8 * x^2 + 35 * x + 12
  let v : ℤ := 3
  let v1 (x : ℤ) : ℤ := v * x + 5
  let v2 (x : ℤ) (v1x : ℤ) : ℤ := v1x * x + 6
  let v3 (x : ℤ) (v2x : ℤ) : ℤ := v2x * x + 79
  x = -4 →
  v3 x (v2 x (v1 x)) = -57 :=
by
  sorry

end NUMINAMATH_GPT_horner_method_v3_value_l1178_117829


namespace NUMINAMATH_GPT_average_weight_decrease_l1178_117874

theorem average_weight_decrease 
  (num_persons : ℕ)
  (avg_weight_initial : ℕ)
  (new_person_weight : ℕ)
  (new_avg_weight : ℚ)
  (weight_decrease : ℚ)
  (h1 : num_persons = 20)
  (h2 : avg_weight_initial = 60)
  (h3 : new_person_weight = 45)
  (h4 : new_avg_weight = (1200 + 45) / 21) : 
  weight_decrease = avg_weight_initial - new_avg_weight :=
by
  sorry

end NUMINAMATH_GPT_average_weight_decrease_l1178_117874


namespace NUMINAMATH_GPT_pow_two_ge_square_l1178_117854

theorem pow_two_ge_square {n : ℕ} (hn : n ≥ 4) : 2^n ≥ n^2 :=
sorry

end NUMINAMATH_GPT_pow_two_ge_square_l1178_117854


namespace NUMINAMATH_GPT_min_value_quadratic_l1178_117835

noncomputable def quadratic_min (a c : ℝ) : ℝ :=
  (2 / a) + (2 / c)

theorem min_value_quadratic {a c : ℝ} (ha : a > 0) (hc : c > 0) (hac : a * c = 1/4) : 
  quadratic_min a c = 8 :=
sorry

end NUMINAMATH_GPT_min_value_quadratic_l1178_117835


namespace NUMINAMATH_GPT_inequality_solution_1_inequality_system_solution_2_l1178_117817

theorem inequality_solution_1 (x : ℝ) : 
  (2 * x - 1) / 2 ≥ 1 - (x + 1) / 3 ↔ x ≥ 7 / 8 := 
sorry

theorem inequality_system_solution_2 (x : ℝ) : 
  (-2 * x ≤ -3) ∧ (x / 2 < 2) ↔ (3 / 2 ≤ x) ∧ (x < 4) :=
sorry

end NUMINAMATH_GPT_inequality_solution_1_inequality_system_solution_2_l1178_117817


namespace NUMINAMATH_GPT_average_speed_to_first_summit_l1178_117828

theorem average_speed_to_first_summit 
  (time_first_summit : ℝ := 3)
  (time_descend_partially : ℝ := 1)
  (time_second_uphill : ℝ := 2)
  (time_descend_back : ℝ := 2)
  (avg_speed_whole_journey : ℝ := 3) :
  avg_speed_whole_journey = 3 →
  time_first_summit = 3 →
  avg_speed_whole_journey * (time_first_summit + time_descend_partially + time_second_uphill + time_descend_back) = 24 →
  avg_speed_whole_journey = 3 := 
by
  intros h_avg_speed h_time_first_summit h_total_distance
  sorry

end NUMINAMATH_GPT_average_speed_to_first_summit_l1178_117828


namespace NUMINAMATH_GPT_dana_hours_sunday_l1178_117882

-- Define the constants given in the problem
def hourly_rate : ℝ := 13
def hours_worked_friday : ℝ := 9
def hours_worked_saturday : ℝ := 10
def total_earnings : ℝ := 286

-- Define the function to compute total earnings from worked hours and hourly rate
def earnings (hours : ℝ) (rate : ℝ) : ℝ := hours * rate

-- Define the proof problem to show the number of hours worked on Sunday
theorem dana_hours_sunday (hours_sunday : ℝ) :
  earnings hours_worked_friday hourly_rate
  + earnings hours_worked_saturday hourly_rate
  + earnings hours_sunday hourly_rate = total_earnings ->
  hours_sunday = 3 :=
by
  sorry -- proof to be filled in

end NUMINAMATH_GPT_dana_hours_sunday_l1178_117882


namespace NUMINAMATH_GPT_cost_formula_correct_l1178_117823

def cost_of_ride (T : ℤ) : ℤ :=
  if T > 5 then 10 + 5 * T - 10 else 10 + 5 * T

theorem cost_formula_correct (T : ℤ) : cost_of_ride T = 10 + 5 * T - (if T > 5 then 10 else 0) := by
  sorry

end NUMINAMATH_GPT_cost_formula_correct_l1178_117823


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1178_117863

-- Defining the given conditions
def is_isosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ c = a
def triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

-- Stating the problem and goal
theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h_iso: is_isosceles a b c)
  (h_len1: a = 3 ∨ a = 6)
  (h_len2: b = 3 ∨ b = 6)
  (h_triangle: triangle a b c): a + b + c = 15 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1178_117863


namespace NUMINAMATH_GPT_find_points_on_number_line_l1178_117894

noncomputable def numbers_are_opposite (x y : ℝ) : Prop :=
  x = -y

theorem find_points_on_number_line (A B : ℝ) 
  (h1 : numbers_are_opposite A B) 
  (h2 : |A - B| = 8) 
  (h3 : A < B) : 
  (A = -4 ∧ B = 4) :=
by
  sorry

end NUMINAMATH_GPT_find_points_on_number_line_l1178_117894


namespace NUMINAMATH_GPT_law_I_law_II_l1178_117838

section
variable (x y z : ℝ)

def op_at (a b : ℝ) : ℝ := a + 2 * b
def op_hash (a b : ℝ) : ℝ := 2 * a - b

theorem law_I (x y z : ℝ) : op_at x (op_hash y z) = op_hash (op_at x y) (op_at x z) := 
by
  unfold op_at op_hash
  sorry

theorem law_II (x y z : ℝ) : x + op_at y z ≠ op_at (x + y) (x + z) := 
by
  unfold op_at
  sorry

end

end NUMINAMATH_GPT_law_I_law_II_l1178_117838


namespace NUMINAMATH_GPT_micah_has_seven_fish_l1178_117861

-- Definitions from problem conditions
def micahFish (M : ℕ) : Prop :=
  let kennethFish := 3 * M
  let matthiasFish := kennethFish - 15
  M + kennethFish + matthiasFish = 34

-- Main statement: prove that the number of fish Micah has is 7
theorem micah_has_seven_fish : ∃ M : ℕ, micahFish M ∧ M = 7 :=
by
  sorry

end NUMINAMATH_GPT_micah_has_seven_fish_l1178_117861


namespace NUMINAMATH_GPT_percent_of_x_is_y_minus_z_l1178_117815

variable (x y z : ℝ)

axiom condition1 : 0.60 * (x - y) = 0.30 * (x + y + z)
axiom condition2 : 0.40 * (y - z) = 0.20 * (y + x - z)

theorem percent_of_x_is_y_minus_z :
  (y - z) = x := by
  sorry

end NUMINAMATH_GPT_percent_of_x_is_y_minus_z_l1178_117815


namespace NUMINAMATH_GPT_find_a_l1178_117808

theorem find_a (a : ℝ) (U A CU: Set ℝ) (hU : U = {2, 3, a^2 - a - 1}) (hA : A = {2, 3}) (hCU : CU = {1}) (hComplement : CU = U \ A) :
  a = -1 ∨ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1178_117808


namespace NUMINAMATH_GPT_correct_factorization_option_A_l1178_117899

variable (x y : ℝ)

theorem correct_factorization_option_A :
  (2 * x^2 + 3 * x + 1 = (2 * x + 1) * (x + 1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_factorization_option_A_l1178_117899


namespace NUMINAMATH_GPT_missing_number_in_proportion_l1178_117866

theorem missing_number_in_proportion (x : ℝ) :
  (2 / x) = ((4 / 3) / (10 / 3)) → x = 5 :=
by sorry

end NUMINAMATH_GPT_missing_number_in_proportion_l1178_117866


namespace NUMINAMATH_GPT_Robert_photo_count_l1178_117872

theorem Robert_photo_count (k : ℕ) (hLisa : ∃ n : ℕ, k = 8 * n) : k = 24 - 16 → k = 24 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Robert_photo_count_l1178_117872


namespace NUMINAMATH_GPT_shirley_cases_l1178_117876

-- Given conditions
def T : ℕ := 54  -- boxes of Trefoils sold
def S : ℕ := 36  -- boxes of Samoas sold
def M : ℕ := 48  -- boxes of Thin Mints sold
def t_per_case : ℕ := 4  -- boxes of Trefoils per case
def s_per_case : ℕ := 3  -- boxes of Samoas per case
def m_per_case : ℕ := 5  -- boxes of Thin Mints per case

-- Amount of boxes delivered per case should meet the required demand
theorem shirley_cases : ∃ (n_cases : ℕ), 
  n_cases * t_per_case ≥ T ∧ 
  n_cases * s_per_case ≥ S ∧ 
  n_cases * m_per_case ≥ M :=
by
  use 14
  sorry

end NUMINAMATH_GPT_shirley_cases_l1178_117876


namespace NUMINAMATH_GPT_correct_option_is_B_l1178_117822

def satisfy_triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem correct_option_is_B :
  satisfy_triangle_inequality 3 4 5 ∧
  ¬ satisfy_triangle_inequality 1 1 2 ∧
  ¬ satisfy_triangle_inequality 1 4 6 ∧
  ¬ satisfy_triangle_inequality 2 3 7 :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_B_l1178_117822


namespace NUMINAMATH_GPT_log_50_between_integers_l1178_117841

open Real

-- Declaration of the proof problem
theorem log_50_between_integers (a b : ℤ) (h1 : log 10 = 1) (h2 : log 100 = 2) (h3 : 10 < 50) (h4 : 50 < 100) :
  a + b = 3 :=
by
  sorry

end NUMINAMATH_GPT_log_50_between_integers_l1178_117841


namespace NUMINAMATH_GPT_Jose_age_proof_l1178_117857

-- Definitions based on the conditions
def Inez_age : ℕ := 15
def Zack_age : ℕ := Inez_age + 5
def Jose_age : ℕ := Zack_age - 7

theorem Jose_age_proof : Jose_age = 13 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_Jose_age_proof_l1178_117857


namespace NUMINAMATH_GPT_find_prime_p_l1178_117812

def is_prime (p: ℕ) : Prop := Nat.Prime p

def is_product_of_three_distinct_primes (n: ℕ) : Prop :=
  ∃ (p1 p2 p3: ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ n = p1 * p2 * p3

theorem find_prime_p (p: ℕ) (hp: is_prime p) :
  (∃ x y z: ℕ, x^p + y^p + z^p - x - y - z = 30) ↔ (p = 2 ∨ p = 3 ∨ p = 5) := 
sorry

end NUMINAMATH_GPT_find_prime_p_l1178_117812


namespace NUMINAMATH_GPT_prove_x3_y3_le_2_l1178_117848

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

axiom positive_x : 0 < x
axiom positive_y : 0 < y
axiom condition : x^3 + y^4 ≤ x^2 + y^3

theorem prove_x3_y3_le_2 : x^3 + y^3 ≤ 2 := 
by
  sorry

end NUMINAMATH_GPT_prove_x3_y3_le_2_l1178_117848


namespace NUMINAMATH_GPT_calculate_LN_l1178_117870

theorem calculate_LN (sinN : ℝ) (LM LN : ℝ) (h1 : sinN = 4 / 5) (h2 : LM = 20) : LN = 25 :=
by
  sorry

end NUMINAMATH_GPT_calculate_LN_l1178_117870


namespace NUMINAMATH_GPT_polynomial_expansion_abs_sum_l1178_117865

theorem polynomial_expansion_abs_sum :
  let a_0 := 1
  let a_1 := -8
  let a_2 := 24
  let a_3 := -32
  let a_4 := 16
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| = 81 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_abs_sum_l1178_117865


namespace NUMINAMATH_GPT_no_possible_arrangement_l1178_117867

theorem no_possible_arrangement :
  ¬ ∃ (a : Fin 9 → ℕ),
    (∀ i, 1 ≤ a i ∧ a i ≤ 9) ∧
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    (∀ i, (a i + a ((i + 1) % 9) + a ((i + 2) % 9)) % 3 = 0) ∧
    (∀ i, (a i + a ((i + 1) % 9) + a ((i + 2) % 9)) > 12) :=
  sorry

end NUMINAMATH_GPT_no_possible_arrangement_l1178_117867


namespace NUMINAMATH_GPT_gcd_of_840_and_1764_l1178_117893

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := 
by {
  sorry
}

end NUMINAMATH_GPT_gcd_of_840_and_1764_l1178_117893


namespace NUMINAMATH_GPT_smallest_n_divisible_by_13_l1178_117851

theorem smallest_n_divisible_by_13 : ∃ (n : ℕ), 5^n + n^5 ≡ 0 [MOD 13] ∧ ∀ (m : ℕ), m < n → ¬(5^m + m^5 ≡ 0 [MOD 13]) :=
sorry

end NUMINAMATH_GPT_smallest_n_divisible_by_13_l1178_117851


namespace NUMINAMATH_GPT_greatest_area_difference_l1178_117804

theorem greatest_area_difference 
    (a b c d : ℕ) 
    (H1 : 2 * (a + b) = 100)
    (H2 : 2 * (c + d) = 100)
    (H3 : ∀i j : ℕ, 2 * (i + j) = 100 → i * j ≤ a * b)
    : 373 ≤ a * b - (c * d) := 
sorry

end NUMINAMATH_GPT_greatest_area_difference_l1178_117804


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1178_117877

def A : Set ℝ := { x | -2 < x ∧ x < 2 }
def B : Set ℝ := { x | x ≤ 1 ∨ x ≥ 3 }

theorem intersection_of_A_and_B : 
  (A ∩ B) = { x : ℝ | -2 < x ∧ x ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1178_117877


namespace NUMINAMATH_GPT_grandfather_age_5_years_back_l1178_117807

variable (F S G : ℕ)

-- Conditions
def father_age : Prop := F = 58
def son_current_age : Prop := S = 58 - S
def son_grandfather_age_relation : Prop := S - 5 = 1 / 2 * (G - 5)

-- Theorem: Prove the grandfather's age 5 years back given the conditions.
theorem grandfather_age_5_years_back (h1 : father_age F) (h2 : son_current_age S) (h3 : son_grandfather_age_relation S G) : G = 2 * S - 5 :=
sorry

end NUMINAMATH_GPT_grandfather_age_5_years_back_l1178_117807


namespace NUMINAMATH_GPT_determine_a_b_l1178_117800

-- Define the polynomial expression
def poly (x a b : ℝ) : ℝ := x^2 + a * x + b

-- Define the factored form
def factored_poly (x : ℝ) : ℝ := (x + 1) * (x - 3)

-- State the theorem
theorem determine_a_b (a b : ℝ) (h : ∀ x, poly x a b = factored_poly x) : a = -2 ∧ b = -3 :=
by 
  sorry

end NUMINAMATH_GPT_determine_a_b_l1178_117800


namespace NUMINAMATH_GPT_discarded_number_l1178_117871

theorem discarded_number (S S_48 : ℝ) (h1 : S = 1000) (h2 : S_48 = 900) (h3 : ∃ x : ℝ, S - S_48 = 45 + x): 
  ∃ x : ℝ, x = 55 :=
by {
  -- Using the conditions provided to derive the theorem.
  sorry 
}

end NUMINAMATH_GPT_discarded_number_l1178_117871


namespace NUMINAMATH_GPT_no_perfect_square_l1178_117810

-- Define the given polynomial
def poly (n : ℕ) : ℤ := n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3

-- The theorem to prove
theorem no_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, poly n = k^2 := by
  sorry

end NUMINAMATH_GPT_no_perfect_square_l1178_117810


namespace NUMINAMATH_GPT_ratio_of_time_spent_l1178_117853

theorem ratio_of_time_spent {total_minutes type_a_minutes type_b_minutes : ℝ}
  (h1 : total_minutes = 180)
  (h2 : type_a_minutes = 32.73)
  (h3 : type_b_minutes = total_minutes - type_a_minutes) :
  type_a_minutes / type_a_minutes = 1 ∧ type_b_minutes / type_a_minutes = 4.5 := by
  sorry

end NUMINAMATH_GPT_ratio_of_time_spent_l1178_117853


namespace NUMINAMATH_GPT_walkway_area_l1178_117881

/--
Tara has four rows of three 8-feet by 3-feet flower beds in her garden. The beds are separated
and surrounded by 2-feet-wide walkways. Prove that the total area of the walkways is 416 square feet.
-/
theorem walkway_area :
  let flower_bed_width := 8
  let flower_bed_height := 3
  let num_rows := 4
  let num_columns := 3
  let walkway_width := 2
  let total_width := (num_columns * flower_bed_width) + (num_columns + 1) * walkway_width
  let total_height := (num_rows * flower_bed_height) + (num_rows + 1) * walkway_width
  let total_garden_area := total_width * total_height
  let flower_bed_area := flower_bed_width * flower_bed_height * num_rows * num_columns
  total_garden_area - flower_bed_area = 416 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_walkway_area_l1178_117881


namespace NUMINAMATH_GPT_min_a_value_l1178_117847

theorem min_a_value {a b : ℕ} (h : 1998 * a = b^4) : a = 1215672 :=
sorry

end NUMINAMATH_GPT_min_a_value_l1178_117847


namespace NUMINAMATH_GPT_eval_g_at_neg2_l1178_117855

def g (x : ℝ) : ℝ := 5 * x + 2

theorem eval_g_at_neg2 : g (-2) = -8 := by
  sorry

end NUMINAMATH_GPT_eval_g_at_neg2_l1178_117855


namespace NUMINAMATH_GPT_sum_f_values_l1178_117845

noncomputable def f : ℝ → ℝ := sorry

axiom odd_property (x : ℝ) : f (-x) = -f (x)
axiom periodicity (x : ℝ) : f (x) = f (x + 4)
axiom f1 : f 1 = -1

theorem sum_f_values : f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry

end NUMINAMATH_GPT_sum_f_values_l1178_117845


namespace NUMINAMATH_GPT_simplest_square_root_l1178_117821

theorem simplest_square_root : 
  let a1 := Real.sqrt 20
  let a2 := Real.sqrt 2
  let a3 := Real.sqrt (1 / 2)
  let a4 := Real.sqrt 0.2
  a2 = Real.sqrt 2 ∧
  (a1 ≠ Real.sqrt 2 ∧ a3 ≠ Real.sqrt 2 ∧ a4 ≠ Real.sqrt 2) :=
by {
  -- Here, we fill in the necessary proof steps, but it's omitted for now.
  sorry
}

end NUMINAMATH_GPT_simplest_square_root_l1178_117821


namespace NUMINAMATH_GPT_original_number_of_girls_l1178_117883

theorem original_number_of_girls (b g : ℕ) (h1 : b = g)
                                (h2 : 3 * (g - 25) = b)
                                (h3 : 6 * (b - 60) = g - 25) :
  g = 67 :=
by sorry

end NUMINAMATH_GPT_original_number_of_girls_l1178_117883


namespace NUMINAMATH_GPT_Olivia_hours_worked_on_Monday_l1178_117806

/-- Olivia works on multiple days in a week with given wages per hour and total income -/
theorem Olivia_hours_worked_on_Monday 
  (M : ℕ)  -- Hours worked on Monday
  (rate_per_hour : ℕ := 9) -- Olivia’s earning rate per hour
  (hours_Wednesday : ℕ := 3)  -- Hours worked on Wednesday
  (hours_Friday : ℕ := 6)  -- Hours worked on Friday
  (total_income : ℕ := 117)  -- Total income earned this week
  (hours_total : ℕ := hours_Wednesday + hours_Friday + M)
  (income_calc : ℕ := rate_per_hour * hours_total) :
  -- Prove that the hours worked on Monday is 4 given the conditions
  income_calc = total_income → M = 4 :=
by
  sorry

end NUMINAMATH_GPT_Olivia_hours_worked_on_Monday_l1178_117806


namespace NUMINAMATH_GPT_two_digit_integer_plus_LCM_of_3_4_5_l1178_117852

theorem two_digit_integer_plus_LCM_of_3_4_5 (x : ℕ) (h1 : 9 < x) (h2 : x < 100) (h3 : ∃ k, x = 60 * k + 2) :
  x = 62 :=
by {
  sorry
}

end NUMINAMATH_GPT_two_digit_integer_plus_LCM_of_3_4_5_l1178_117852


namespace NUMINAMATH_GPT_min_sum_of_primes_l1178_117820

open Classical

theorem min_sum_of_primes (k m n p : ℕ) (h1 : 47 + m = k) (h2 : 53 + n = k) (h3 : 71 + p = k)
  (pm : Prime m) (pn : Prime n) (pp : Prime p) :
  m + n + p = 57 ↔ (k = 76 ∧ m = 29 ∧ n = 23 ∧ p = 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_sum_of_primes_l1178_117820


namespace NUMINAMATH_GPT_complement_event_A_l1178_117860

def is_at_least_two_defective (n : ℕ) : Prop :=
  n ≥ 2

def is_at_most_one_defective (n : ℕ) : Prop :=
  n ≤ 1

theorem complement_event_A (n : ℕ) :
  (¬ is_at_least_two_defective n) ↔ is_at_most_one_defective n :=
by
  sorry

end NUMINAMATH_GPT_complement_event_A_l1178_117860


namespace NUMINAMATH_GPT_ratio_of_games_played_to_losses_l1178_117895

-- Conditions
def games_played : ℕ := 10
def games_won : ℕ := 5
def games_lost : ℕ := games_played - games_won

-- Prove the ratio of games played to games lost is 2:1
theorem ratio_of_games_played_to_losses
  (h_played : games_played = 10)
  (h_won : games_won = 5) :
  (games_played / Nat.gcd games_played games_lost : ℕ) /
  (games_lost / Nat.gcd games_played games_lost : ℕ) = 2 / 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_games_played_to_losses_l1178_117895


namespace NUMINAMATH_GPT_find_constants_l1178_117836

theorem find_constants (A B : ℚ) : 
  (∀ x : ℚ, x ≠ 10 ∧ x ≠ -5 → (8 * x - 3) / (x^2 - 5 * x - 50) = A / (x - 10) + B / (x + 5)) 
  → (A = 77 / 15 ∧ B = 43 / 15) := by 
  sorry

end NUMINAMATH_GPT_find_constants_l1178_117836


namespace NUMINAMATH_GPT_trigonometric_solution_l1178_117885

theorem trigonometric_solution (x : Real) :
  (2 * Real.sin x * Real.cos (3 * Real.pi / 2 + x) 
  - 3 * Real.sin (Real.pi - x) * Real.cos x 
  + Real.sin (Real.pi / 2 + x) * Real.cos x = 0) ↔ 
  (∃ k : Int, x = Real.arctan ((3 + Real.sqrt 17) / -4) + k * Real.pi) ∨ 
  (∃ n : Int, x = Real.arctan ((3 - Real.sqrt 17) / -4) + n * Real.pi) :=
sorry

end NUMINAMATH_GPT_trigonometric_solution_l1178_117885


namespace NUMINAMATH_GPT_cos_alpha_plus_beta_l1178_117890

variable (α β : ℝ)
variable (hα : Real.sin α = (Real.sqrt 5) / 5)
variable (hβ : Real.sin β = (Real.sqrt 10) / 10)
variable (hα_obtuse : π / 2 < α ∧ α < π)
variable (hβ_obtuse : π / 2 < β ∧ β < π)

theorem cos_alpha_plus_beta : Real.cos (α + β) = Real.sqrt 2 / 2 ∧ α + β = 7 * π / 4 := by
  sorry

end NUMINAMATH_GPT_cos_alpha_plus_beta_l1178_117890


namespace NUMINAMATH_GPT_rope_cut_ratio_l1178_117846

theorem rope_cut_ratio (L : ℕ) (a b : ℕ) (hL : L = 40) (ha : a = 2) (hb : b = 3) :
  L / (a + b) * a = 16 :=
by
  sorry

end NUMINAMATH_GPT_rope_cut_ratio_l1178_117846


namespace NUMINAMATH_GPT_chosen_number_eq_l1178_117803

-- Given a number x, if (x / 2) - 100 = 4, then x = 208.
theorem chosen_number_eq (x : ℝ) (h : (x / 2) - 100 = 4) : x = 208 := 
by
  sorry

end NUMINAMATH_GPT_chosen_number_eq_l1178_117803


namespace NUMINAMATH_GPT_find_a_squared_l1178_117811

-- Defining the conditions for the problem
structure RectangleConditions :=
  (a : ℝ) 
  (side_length : ℝ := 36)
  (hinges_vertex : Bool := true)
  (hinges_midpoint : Bool := true)
  (pressed_distance : ℝ := 24)
  (hexagon_area_equiv : Bool := true)

-- Stating the theorem
theorem find_a_squared (cond : RectangleConditions) (ha : 36 * cond.a = 
  (24 * cond.a) + 2 * 15 * Real.sqrt (cond.a^2 - 36)) : 
  cond.a^2 = 720 :=
sorry

end NUMINAMATH_GPT_find_a_squared_l1178_117811


namespace NUMINAMATH_GPT_find_integer_part_of_m_l1178_117873

theorem find_integer_part_of_m {m : ℝ} (h_lecture_duration : m > 0) 
    (h_swap_positions : ∃ k : ℤ, 120 + m = 60 + k * 12 * 60 / 13 ∧ (120 + m) % 60 = 60 * (120 + m) / 720) : 
    ⌊m⌋ = 46 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_part_of_m_l1178_117873


namespace NUMINAMATH_GPT_ratio_prikya_ladonna_l1178_117858

def total_cans : Nat := 85
def ladonna_cans : Nat := 25
def yoki_cans : Nat := 10
def prikya_cans : Nat := total_cans - ladonna_cans - yoki_cans

theorem ratio_prikya_ladonna : prikya_cans.toFloat / ladonna_cans.toFloat = 2 / 1 := 
by sorry

end NUMINAMATH_GPT_ratio_prikya_ladonna_l1178_117858


namespace NUMINAMATH_GPT_student_fraction_mistake_l1178_117886

theorem student_fraction_mistake (n : ℕ) (h_n : n = 576) 
(h_mistake : ∃ r : ℚ, r * n = (5 / 16) * n + 300) : ∃ r : ℚ, r = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_student_fraction_mistake_l1178_117886


namespace NUMINAMATH_GPT_sum_of_integers_l1178_117849

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 168) : x + y = 32 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l1178_117849


namespace NUMINAMATH_GPT_ryan_more_hours_english_than_chinese_l1178_117884

-- Definitions for the time Ryan spends on subjects
def weekday_hours_english : ℕ := 6 * 5
def weekend_hours_english : ℕ := 2 * 2
def total_hours_english : ℕ := weekday_hours_english + weekend_hours_english

def weekday_hours_chinese : ℕ := 3 * 5
def weekend_hours_chinese : ℕ := 1 * 2
def total_hours_chinese : ℕ := weekday_hours_chinese + weekend_hours_chinese

-- Theorem stating the difference in hours spent on English vs Chinese
theorem ryan_more_hours_english_than_chinese :
  (total_hours_english - total_hours_chinese) = 17 := by
  sorry

end NUMINAMATH_GPT_ryan_more_hours_english_than_chinese_l1178_117884


namespace NUMINAMATH_GPT_initial_apples_value_l1178_117889

-- Definitions for the conditions
def picked_apples : ℤ := 105
def total_apples : ℤ := 161

-- Statement to prove
theorem initial_apples_value : ∀ (initial_apples : ℤ), 
  initial_apples + picked_apples = total_apples → 
  initial_apples = total_apples - picked_apples := 
by 
  sorry

end NUMINAMATH_GPT_initial_apples_value_l1178_117889


namespace NUMINAMATH_GPT_probability_of_victory_l1178_117896

theorem probability_of_victory (p_A p_B : ℝ) (h_A : p_A = 0.3) (h_B : p_B = 0.6) (independent : true) :
  p_A * p_B = 0.18 :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_probability_of_victory_l1178_117896


namespace NUMINAMATH_GPT_lauren_change_l1178_117827

-- Define the given conditions as Lean terms.
def price_meat_per_pound : ℝ := 3.5
def pounds_meat : ℝ := 2.0
def price_buns : ℝ := 1.5
def price_lettuce : ℝ := 1.0
def pounds_tomato : ℝ := 1.5
def price_tomato_per_pound : ℝ := 2.0
def price_pickles : ℝ := 2.5
def coupon_value : ℝ := 1.0
def amount_paid : ℝ := 20.0

-- Define the total cost of each item.
def cost_meat : ℝ := pounds_meat * price_meat_per_pound
def cost_tomato : ℝ := pounds_tomato * price_tomato_per_pound
def total_cost_before_coupon : ℝ := cost_meat + price_buns + price_lettuce + cost_tomato + price_pickles

-- Define the final total cost after applying the coupon.
def final_total_cost : ℝ := total_cost_before_coupon - coupon_value

-- Define the expected change.
def expected_change : ℝ := amount_paid - final_total_cost

-- Prove that the expected change is $6.00.
theorem lauren_change : expected_change = 6.0 := by
  sorry

end NUMINAMATH_GPT_lauren_change_l1178_117827


namespace NUMINAMATH_GPT_ratio_a_c_l1178_117825

variable (a b c d : ℕ)

/-- The given conditions -/
axiom ratio_a_b : a / b = 5 / 2
axiom ratio_c_d : c / d = 4 / 1
axiom ratio_d_b : d / b = 1 / 3

/-- The proof problem -/
theorem ratio_a_c : a / c = 15 / 8 := by
  sorry

end NUMINAMATH_GPT_ratio_a_c_l1178_117825


namespace NUMINAMATH_GPT_greatest_value_a_maximum_value_a_l1178_117862

-- Define the quadratic polynomial
def quadratic (a : ℝ) : ℝ := -a^2 + 9 * a - 20

-- The statement to be proven:
theorem greatest_value_a : ∀ a : ℝ, (quadratic a ≥ 0) → a ≤ 5 := 
sorry

theorem maximum_value_a : quadratic 5 = 0 :=
sorry

end NUMINAMATH_GPT_greatest_value_a_maximum_value_a_l1178_117862


namespace NUMINAMATH_GPT_tan_sum_trig_identity_l1178_117834

variable {A B C : ℝ} -- Angles
variable {a b c : ℝ} -- Sides opposite to angles A, B and C

-- Acute triangle implies A, B, C are all less than π/2 and greater than 0
variable (hAcute : 0 < A ∧ A < pi / 2 ∧ 0 < B ∧ B < pi / 2 ∧ 0 < C ∧ C < pi / 2)

-- Given condition in the problem
variable (hCondition : b / a + a / b = 6 * Real.cos C)

theorem tan_sum_trig_identity : 
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 4 :=
sorry

end NUMINAMATH_GPT_tan_sum_trig_identity_l1178_117834


namespace NUMINAMATH_GPT_max_k_inequality_l1178_117802

theorem max_k_inequality (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 1) 
                                      (h₂ : 0 ≤ b) (h₃ : b ≤ 1) 
                                      (h₄ : 0 ≤ c) (h₅ : c ≤ 1) 
                                      (h₆ : 0 ≤ d) (h₇ : d ≤ 1) :
  a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^2 + b^2 + c^2 + d^2) :=
sorry

end NUMINAMATH_GPT_max_k_inequality_l1178_117802


namespace NUMINAMATH_GPT_chef_meals_prepared_for_dinner_l1178_117843

theorem chef_meals_prepared_for_dinner (lunch_meals_prepared lunch_meals_sold dinner_meals_total : ℕ) 
  (h1 : lunch_meals_prepared = 17)
  (h2 : lunch_meals_sold = 12)
  (h3 : dinner_meals_total = 10) :
  (dinner_meals_total - (lunch_meals_prepared - lunch_meals_sold)) = 5 :=
by
  -- Lean proof code to proceed from here
  sorry

end NUMINAMATH_GPT_chef_meals_prepared_for_dinner_l1178_117843


namespace NUMINAMATH_GPT_tim_income_percent_less_than_juan_l1178_117888

theorem tim_income_percent_less_than_juan (T M J : ℝ) (h1 : M = 1.5 * T) (h2 : M = 0.9 * J) :
  (J - T) / J = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_tim_income_percent_less_than_juan_l1178_117888


namespace NUMINAMATH_GPT_john_growth_l1178_117892

theorem john_growth 
  (InitialHeight : ℤ)
  (GrowthRate : ℤ)
  (FinalHeight : ℤ)
  (h1 : InitialHeight = 66)
  (h2 : GrowthRate = 2)
  (h3 : FinalHeight = 72) :
  (FinalHeight - InitialHeight) / GrowthRate = 3 :=
by
  sorry

end NUMINAMATH_GPT_john_growth_l1178_117892


namespace NUMINAMATH_GPT_compare_magnitudes_l1178_117830

noncomputable def A : ℝ := Real.sin (Real.sin (3 * Real.pi / 8))
noncomputable def B : ℝ := Real.sin (Real.cos (3 * Real.pi / 8))
noncomputable def C : ℝ := Real.cos (Real.sin (3 * Real.pi / 8))
noncomputable def D : ℝ := Real.cos (Real.cos (3 * Real.pi / 8))

theorem compare_magnitudes : B < C ∧ C < A ∧ A < D :=
by
  sorry

end NUMINAMATH_GPT_compare_magnitudes_l1178_117830


namespace NUMINAMATH_GPT_equation_of_circle_correct_l1178_117864

open Real

def equation_of_circle_through_points (x y : ℝ) :=
  x^2 + y^2 - 4 * x - 6 * y

theorem equation_of_circle_correct :
  ∀ (x y : ℝ),
    (equation_of_circle_through_points (0 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (4 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (-1 : ℝ) (1 : ℝ) = 0) →
    (equation_of_circle_through_points x y = 0) :=
by 
  sorry

end NUMINAMATH_GPT_equation_of_circle_correct_l1178_117864


namespace NUMINAMATH_GPT_common_difference_l1178_117809

def Sn (S : Nat → ℝ) (n : Nat) : ℝ := S n

theorem common_difference (S : Nat → ℝ) (H : Sn S 2016 / 2016 = Sn S 2015 / 2015 + 1) : 2 = 2 := 
by
  sorry

end NUMINAMATH_GPT_common_difference_l1178_117809


namespace NUMINAMATH_GPT_balls_drawn_ensure_single_color_ge_20_l1178_117826

theorem balls_drawn_ensure_single_color_ge_20 (r g y b w bl : ℕ) (h_r : r = 34) (h_g : g = 28) (h_y : y = 23) (h_b : b = 18) (h_w : w = 12) (h_bl : bl = 11) : 
  ∃ (n : ℕ), n ≥ 20 →
    (r + g + y + b + w + bl - n) + 1 > 20 :=
by
  sorry

end NUMINAMATH_GPT_balls_drawn_ensure_single_color_ge_20_l1178_117826


namespace NUMINAMATH_GPT_volume_ratio_of_cube_cut_l1178_117833

/-
  The cube ABCDEFGH has its side length assumed to be 1.
  The points K, L, M divide the vertical edges AA', BB', CC'
  respectively, in the ratios 1:2, 1:3, 1:4. 
  We need to prove that the plane KLM cuts the cube into
  two parts such that the volume ratio of the two parts is 4:11.
-/
theorem volume_ratio_of_cube_cut (s : ℝ) (K L M : ℝ) :
  ∃ (Vbelow Vabove : ℝ), 
    s = 1 → 
    K = 1/3 → 
    L = 1/4 → 
    M = 1/5 → 
    Vbelow / Vabove = 4 / 11 :=
sorry

end NUMINAMATH_GPT_volume_ratio_of_cube_cut_l1178_117833


namespace NUMINAMATH_GPT_candy_from_sister_is_5_l1178_117887

noncomputable def candy_received_from_sister (candy_from_neighbors : ℝ) (pieces_per_day : ℝ) (days : ℕ) : ℝ :=
  pieces_per_day * days - candy_from_neighbors

theorem candy_from_sister_is_5 :
  candy_received_from_sister 11.0 8.0 2 = 5.0 :=
by
  sorry

end NUMINAMATH_GPT_candy_from_sister_is_5_l1178_117887


namespace NUMINAMATH_GPT_series_evaluation_l1178_117805

noncomputable def series_sum : ℝ :=
  ∑' m : ℕ, (∑' n : ℕ, (m^2 * n) / (3^m * (n * 3^m + m * 3^n)))

theorem series_evaluation : series_sum = 9 / 32 :=
by
  sorry

end NUMINAMATH_GPT_series_evaluation_l1178_117805


namespace NUMINAMATH_GPT_exists_sum_of_150_consecutive_integers_l1178_117844

theorem exists_sum_of_150_consecutive_integers :
  ∃ a : ℕ, 1627395075 = 150 * a + 11175 :=
by
  sorry

end NUMINAMATH_GPT_exists_sum_of_150_consecutive_integers_l1178_117844
