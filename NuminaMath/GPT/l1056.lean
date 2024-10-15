import Mathlib

namespace NUMINAMATH_GPT_sum_a1_a3_a5_l1056_105606

-- Definitions
variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)

-- Conditions
axiom initial_condition : a 1 = 16
axiom relationship_ak_bk : ∀ k, b k = a k / 2
axiom ak_next : ∀ k, a (k + 1) = a k + 2 * (b k)

-- Theorem Statement
theorem sum_a1_a3_a5 : a 1 + a 3 + a 5 = 336 :=
by
  sorry

end NUMINAMATH_GPT_sum_a1_a3_a5_l1056_105606


namespace NUMINAMATH_GPT_tom_sold_price_l1056_105611

noncomputable def original_price : ℝ := 200
noncomputable def tripled_price (price : ℝ) : ℝ := 3 * price
noncomputable def sold_price (price : ℝ) : ℝ := 0.4 * price

theorem tom_sold_price : sold_price (tripled_price original_price) = 240 := 
by
  sorry

end NUMINAMATH_GPT_tom_sold_price_l1056_105611


namespace NUMINAMATH_GPT_intersection_M_N_l1056_105619

-- Define the set M based on the given condition
def M : Set ℝ := { x | x^2 > 1 }

-- Define the set N based on the given elements
def N : Set ℝ := { x | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 }

-- Prove that the intersection of M and N is {-2, 2}
theorem intersection_M_N : M ∩ N = { -2, 2 } := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1056_105619


namespace NUMINAMATH_GPT_length_of_hypotenuse_l1056_105653

theorem length_of_hypotenuse (a b : ℝ) (h1 : a = 15) (h2 : b = 21) : 
hypotenuse_length = Real.sqrt (a^2 + b^2) :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_length_of_hypotenuse_l1056_105653


namespace NUMINAMATH_GPT_min_abs_diff_l1056_105669

theorem min_abs_diff (a b c d : ℝ) (h1 : |a - b| = 5) (h2 : |b - c| = 8) (h3 : |c - d| = 10) : 
  ∃ m, m = |a - d| ∧ m = 3 := 
by 
  sorry

end NUMINAMATH_GPT_min_abs_diff_l1056_105669


namespace NUMINAMATH_GPT_simplify_fraction_l1056_105661

variable (x y : ℝ)

theorem simplify_fraction (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + 1/y ≠ 0) (h2 : y + 1/x ≠ 0) : 
  (x + 1/y) / (y + 1/x) = x / y :=
sorry

end NUMINAMATH_GPT_simplify_fraction_l1056_105661


namespace NUMINAMATH_GPT_find_k_l1056_105627

def a : ℝ × ℝ := (2, 1)
def b (k : ℝ) : ℝ × ℝ := (-2, k)
def vec_op (a b : ℝ × ℝ) : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

noncomputable def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_k (k : ℝ) : dot_prod a (vec_op a (b k)) = 0 → k = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1056_105627


namespace NUMINAMATH_GPT_division_correct_l1056_105689

theorem division_correct (x : ℝ) (h : 10 / x = 2) : 20 / x = 4 :=
by
  sorry

end NUMINAMATH_GPT_division_correct_l1056_105689


namespace NUMINAMATH_GPT_parking_lot_wheels_l1056_105695

noncomputable def total_car_wheels (guest_cars : Nat) (guest_car_wheels : Nat) (parent_cars : Nat) (parent_car_wheels : Nat) : Nat :=
  guest_cars * guest_car_wheels + parent_cars * parent_car_wheels

theorem parking_lot_wheels :
  total_car_wheels 10 4 2 4 = 48 :=
by
  sorry

end NUMINAMATH_GPT_parking_lot_wheels_l1056_105695


namespace NUMINAMATH_GPT_find_four_numbers_l1056_105601

theorem find_four_numbers (a b c d : ℕ) : 
  a + b + c + d = 45 ∧ (∃ k : ℕ, a + 2 = k ∧ b - 2 = k ∧ 2 * c = k ∧ d / 2 = k) → (a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20) :=
by
  sorry

end NUMINAMATH_GPT_find_four_numbers_l1056_105601


namespace NUMINAMATH_GPT_remainder_of_polynomial_l1056_105694

-- Define the polynomial and the divisor
def f (x : ℝ) := x^3 - 4 * x + 6
def a := -3

-- State the theorem
theorem remainder_of_polynomial :
  f a = -9 := by
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_l1056_105694


namespace NUMINAMATH_GPT_determine_a_l1056_105626

theorem determine_a (a : ℕ)
  (h1 : 2 / (2 + 3 + a) = 1 / 3) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l1056_105626


namespace NUMINAMATH_GPT_steak_weight_in_ounces_l1056_105678

-- Definitions from conditions
def pounds : ℕ := 15
def ounces_per_pound : ℕ := 16
def steaks : ℕ := 20

-- The theorem to prove
theorem steak_weight_in_ounces : 
  (pounds * ounces_per_pound) / steaks = 12 := by
  sorry

end NUMINAMATH_GPT_steak_weight_in_ounces_l1056_105678


namespace NUMINAMATH_GPT_matrix_pow_A_50_l1056_105632

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![5, 2], ![-16, -6]]

theorem matrix_pow_A_50 :
  A ^ 50 = ![![301, 100], ![-800, -249]] :=
by
  sorry

end NUMINAMATH_GPT_matrix_pow_A_50_l1056_105632


namespace NUMINAMATH_GPT_box_volume_is_correct_l1056_105691

noncomputable def box_volume (length width cut_side : ℝ) : ℝ :=
  (length - 2 * cut_side) * (width - 2 * cut_side) * cut_side

theorem box_volume_is_correct : box_volume 48 36 5 = 9880 := by
  sorry

end NUMINAMATH_GPT_box_volume_is_correct_l1056_105691


namespace NUMINAMATH_GPT_mean_score_for_exam_l1056_105620

variable (M SD : ℝ)

-- Define the conditions
def condition1 : Prop := 58 = M - 2 * SD
def condition2 : Prop := 98 = M + 3 * SD

-- The problem statement
theorem mean_score_for_exam (h1 : condition1 M SD) (h2 : condition2 M SD) : M = 74 :=
sorry

end NUMINAMATH_GPT_mean_score_for_exam_l1056_105620


namespace NUMINAMATH_GPT_find_y_l1056_105630

noncomputable def x : Real := 1.6666666666666667
def y : Real := 5

theorem find_y (h : x ≠ 0) (h1 : (x * y) / 3 = x^2) : y = 5 := 
by sorry

end NUMINAMATH_GPT_find_y_l1056_105630


namespace NUMINAMATH_GPT_radius_of_circle_l1056_105608

theorem radius_of_circle (r : ℝ) (h : 3 * (2 * Real.pi * r) = 2 * (Real.pi * r ^ 2)) : r = 3 := by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l1056_105608


namespace NUMINAMATH_GPT_number_of_elements_l1056_105614

theorem number_of_elements
  (init_avg : ℕ → ℝ)
  (correct_avg : ℕ → ℝ)
  (incorrect_num correct_num : ℝ)
  (h1 : ∀ n : ℕ, init_avg n = 17)
  (h2 : ∀ n : ℕ, correct_avg n = 20)
  (h3 : incorrect_num = 26)
  (h4 : correct_num = 56)
  : ∃ n : ℕ, n = 10 := sorry

end NUMINAMATH_GPT_number_of_elements_l1056_105614


namespace NUMINAMATH_GPT_geometric_sequence_sum_S8_l1056_105642

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_S8_l1056_105642


namespace NUMINAMATH_GPT_range_j_l1056_105638

def h (x : ℝ) : ℝ := 2 * x + 3

def j (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_j : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 61 ≤ j x ∧ j x ≤ 93) := 
by 
  sorry

end NUMINAMATH_GPT_range_j_l1056_105638


namespace NUMINAMATH_GPT_mother_to_grandfather_age_ratio_l1056_105698

theorem mother_to_grandfather_age_ratio
  (rachel_age : ℕ)
  (grandfather_ratio : ℕ)
  (father_mother_gap : ℕ) 
  (future_rachel_age: ℕ) 
  (future_father_age : ℕ)
  (current_father_age current_mother_age current_grandfather_age : ℕ) 
  (h1 : rachel_age = 12)
  (h2 : grandfather_ratio = 7)
  (h3 : father_mother_gap = 5)
  (h4 : future_rachel_age = 25)
  (h5 : future_father_age = 60)
  (h6 : current_father_age = future_father_age - (future_rachel_age - rachel_age))
  (h7 : current_mother_age = current_father_age - father_mother_gap)
  (h8 : current_grandfather_age = grandfather_ratio * rachel_age) :
  current_mother_age = current_grandfather_age / 2 :=
by
  sorry

end NUMINAMATH_GPT_mother_to_grandfather_age_ratio_l1056_105698


namespace NUMINAMATH_GPT_correct_exponentiation_l1056_105658

theorem correct_exponentiation : ∀ (x : ℝ), (x^(4/5))^(5/4) = x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_correct_exponentiation_l1056_105658


namespace NUMINAMATH_GPT_nate_distance_after_resting_l1056_105696

variables (length_of_field total_distance : ℕ)

def distance_before_resting (length_of_field : ℕ) := 4 * length_of_field

def distance_after_resting (total_distance length_of_field : ℕ) : ℕ := 
  total_distance - distance_before_resting length_of_field

theorem nate_distance_after_resting
  (length_of_field_val : length_of_field = 168)
  (total_distance_val : total_distance = 1172) :
  distance_after_resting total_distance length_of_field = 500 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_nate_distance_after_resting_l1056_105696


namespace NUMINAMATH_GPT_pencils_per_row_l1056_105666

-- Define the conditions
def total_pencils := 25
def number_of_rows := 5

-- Theorem statement: The number of pencils per row is 5 given the conditions
theorem pencils_per_row : total_pencils / number_of_rows = 5 :=
by
  -- The proof should go here
  sorry

end NUMINAMATH_GPT_pencils_per_row_l1056_105666


namespace NUMINAMATH_GPT_tina_husband_brownies_days_l1056_105699

variable (d : Nat)

theorem tina_husband_brownies_days : 
  (exists (d : Nat), 
    let total_brownies := 24
    let tina_daily := 2
    let husband_daily := 1
    let total_daily := tina_daily + husband_daily
    let shared_with_guests := 4
    let remaining_brownies := total_brownies - shared_with_guests
    let final_leftover := 5
    let brownies_eaten := remaining_brownies - final_leftover
    brownies_eaten = d * total_daily) → d = 5 := 
by
  sorry

end NUMINAMATH_GPT_tina_husband_brownies_days_l1056_105699


namespace NUMINAMATH_GPT_constant_function_solution_l1056_105648

theorem constant_function_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, 2 * f x = f (x + y) + f (x + 2 * y)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end NUMINAMATH_GPT_constant_function_solution_l1056_105648


namespace NUMINAMATH_GPT_functional_ineq_l1056_105657

noncomputable def f : ℝ → ℝ := sorry

theorem functional_ineq (h1 : ∀ x > 1400^2021, x * f x ≤ 2021) (h2 : ∀ x : ℝ, 0 < x → f x = f (x + 2) + 2 * f (x * (x + 2))) : 
  ∀ x : ℝ, 0 < x → x * f x ≤ 2021 :=
sorry

end NUMINAMATH_GPT_functional_ineq_l1056_105657


namespace NUMINAMATH_GPT_sum_even_squares_sum_odd_squares_l1056_105623

open scoped BigOperators

def sumOfSquaresEven (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, (2 * (i + 1))^2

def sumOfSquaresOdd (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, (2 * i + 1)^2

theorem sum_even_squares (n : ℕ) :
  sumOfSquaresEven n = (2 * n * (n - 1) * (2 * n - 1)) / 3 := by
    sorry

theorem sum_odd_squares (n : ℕ) :
  sumOfSquaresOdd n = (n * (4 * n^2 - 1)) / 3 := by
    sorry

end NUMINAMATH_GPT_sum_even_squares_sum_odd_squares_l1056_105623


namespace NUMINAMATH_GPT_new_bookstore_acquisition_l1056_105622

theorem new_bookstore_acquisition (x : ℝ) 
  (h1 : (1 / 2) * x + (1 / 4) * x + 50 = x - 200) : x = 1000 :=
by {
  sorry
}

end NUMINAMATH_GPT_new_bookstore_acquisition_l1056_105622


namespace NUMINAMATH_GPT_average_age_of_new_men_is_30_l1056_105629

noncomputable def average_age_of_two_new_men (A : ℝ) : ℝ :=
  let total_age_before : ℝ := 8 * A
  let total_age_after : ℝ := 8 * (A + 2)
  let age_of_replaced_men : ℝ := 21 + 23
  let total_age_of_new_men : ℝ := total_age_after - total_age_before + age_of_replaced_men
  total_age_of_new_men / 2

theorem average_age_of_new_men_is_30 (A : ℝ) : 
  average_age_of_two_new_men A = 30 :=
by 
  sorry

end NUMINAMATH_GPT_average_age_of_new_men_is_30_l1056_105629


namespace NUMINAMATH_GPT_dima_always_wins_l1056_105692

theorem dima_always_wins (n : ℕ) (P : Prop) : 
  (∀ (gosha dima : ℕ → Prop), 
    (∀ k : ℕ, k < n → (gosha k ∨ dima k))
    ∧ (∀ i : ℕ, i < 14 → (gosha i ∨ dima i))
    ∧ (∃ j : ℕ, j ≤ n ∧ (∃ k ≤ j + 7, dima k))
    ∧ (∃ l : ℕ, l ≤ 14 ∧ (∃ m ≤ l + 7, dima m))
    → P) → P := sorry

end NUMINAMATH_GPT_dima_always_wins_l1056_105692


namespace NUMINAMATH_GPT_total_age_difference_is_twelve_l1056_105635

variable {A B C : ℕ}

theorem total_age_difference_is_twelve (h1 : A + B > B + C) (h2 : C = A - 12) :
  (A + B) - (B + C) = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_age_difference_is_twelve_l1056_105635


namespace NUMINAMATH_GPT_system_sum_of_squares_l1056_105656

theorem system_sum_of_squares :
  (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    9*y1^2 - 4*x1^2 = 144 - 48*x1 ∧ 9*y1^2 + 4*x1^2 = 144 + 18*x1*y1 ∧
    9*y2^2 - 4*x2^2 = 144 - 48*x2 ∧ 9*y2^2 + 4*x2^2 = 144 + 18*x2*y2 ∧
    9*y3^2 - 4*x3^2 = 144 - 48*x3 ∧ 9*y3^2 + 4*x3^2 = 144 + 18*x3*y3 ∧
    (x1^2 + x2^2 + x3^2 + y1^2 + y2^2 + y3^2 = 68)) :=
by sorry

end NUMINAMATH_GPT_system_sum_of_squares_l1056_105656


namespace NUMINAMATH_GPT_speed_of_current_l1056_105664

-- Conditions translated into Lean definitions
def initial_time : ℝ := 13 -- 1:00 PM is represented as 13:00 hours
def boat1_time_turnaround : ℝ := 14 -- Boat 1 turns around at 2:00 PM
def boat2_time_turnaround : ℝ := 15 -- Boat 2 turns around at 3:00 PM
def meeting_time : ℝ := 16 -- Boats meet at 4:00 PM
def raft_drift_distance : ℝ := 7.5 -- Raft drifted 7.5 km from the pier

-- The problem statement to prove
theorem speed_of_current:
  ∃ v : ℝ, (v * (meeting_time - initial_time) = raft_drift_distance) ∧ v = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_current_l1056_105664


namespace NUMINAMATH_GPT_edge_length_in_mm_l1056_105602

-- Definitions based on conditions
def cube_volume (a : ℝ) : ℝ := a^3

axiom volume_of_dice : cube_volume 2 = 8

-- Statement of the theorem to be proved
theorem edge_length_in_mm : ∃ (a : ℝ), cube_volume a = 8 ∧ a * 10 = 20 := sorry

end NUMINAMATH_GPT_edge_length_in_mm_l1056_105602


namespace NUMINAMATH_GPT_Matilda_correct_age_l1056_105607

def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

theorem Matilda_correct_age : Matilda_age = 35 :=
by
  -- Proof needs to be filled here
  sorry

end NUMINAMATH_GPT_Matilda_correct_age_l1056_105607


namespace NUMINAMATH_GPT_saree_original_price_l1056_105624

theorem saree_original_price :
  ∃ P : ℝ, (0.95 * 0.88 * P = 334.4) ∧ (P = 400) :=
by
  sorry

end NUMINAMATH_GPT_saree_original_price_l1056_105624


namespace NUMINAMATH_GPT_tim_meditation_time_l1056_105637

-- Definitions of the conditions:
def time_reading_week (t_reading : ℕ) : Prop := t_reading = 14
def twice_as_much_reading (t_reading t_meditate : ℕ) : Prop := t_reading = 2 * t_meditate

-- The theorem to prove:
theorem tim_meditation_time (t_reading t_meditate_per_day : ℕ) 
  (h1 : time_reading_week t_reading)
  (h2 : twice_as_much_reading t_reading (7 * t_meditate_per_day)) :
  t_meditate_per_day = 1 :=
by
  sorry

end NUMINAMATH_GPT_tim_meditation_time_l1056_105637


namespace NUMINAMATH_GPT_zero_in_M_l1056_105651

def M : Set Int := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M :=
  by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_zero_in_M_l1056_105651


namespace NUMINAMATH_GPT_jills_daily_earnings_first_month_l1056_105677

-- Definitions based on conditions
variable (x : ℕ) -- daily earnings in the first month
def total_earnings_first_month := 30 * x
def total_earnings_second_month := 30 * (2 * x)
def total_earnings_third_month := 15 * (2 * x)
def total_earnings_three_months := total_earnings_first_month x + total_earnings_second_month x + total_earnings_third_month x

-- The theorem we need to prove
theorem jills_daily_earnings_first_month
  (h : total_earnings_three_months x = 1200) : x = 10 :=
sorry

end NUMINAMATH_GPT_jills_daily_earnings_first_month_l1056_105677


namespace NUMINAMATH_GPT_difference_of_values_l1056_105662

theorem difference_of_values (num : Nat) : 
  (num = 96348621) →
  let face_value := 8
  let local_value := 8 * 10000
  local_value - face_value = 79992 := 
by
  intros h_eq
  have face_value := 8
  have local_value := 8 * 10000
  sorry

end NUMINAMATH_GPT_difference_of_values_l1056_105662


namespace NUMINAMATH_GPT_time_period_simple_interest_l1056_105647

theorem time_period_simple_interest 
  (P : ℝ) (R18 R12 : ℝ) (additional_interest : ℝ) (T : ℝ) :
  P = 2500 →
  R18 = 0.18 →
  R12 = 0.12 →
  additional_interest = 300 →
  P * R18 * T = P * R12 * T + additional_interest →
  T = 2 :=
by
  intros P_val R18_val R12_val add_int_val interest_eq
  rw [P_val, R18_val, R12_val, add_int_val] at interest_eq
  -- Continue the proof here
  sorry

end NUMINAMATH_GPT_time_period_simple_interest_l1056_105647


namespace NUMINAMATH_GPT_amount_C_l1056_105668

theorem amount_C (A_amt B_amt C_amt : ℚ)
  (h1 : A_amt + B_amt + C_amt = 527)
  (h2 : A_amt = (2 / 3) * B_amt)
  (h3 : B_amt = (1 / 4) * C_amt) :
  C_amt = 372 :=
sorry

end NUMINAMATH_GPT_amount_C_l1056_105668


namespace NUMINAMATH_GPT_menu_choices_l1056_105687

theorem menu_choices :
  let lunchChinese := 5 
  let lunchJapanese := 4 
  let dinnerChinese := 3 
  let dinnerJapanese := 5 
  let lunchOptions := lunchChinese + lunchJapanese
  let dinnerOptions := dinnerChinese + dinnerJapanese
  lunchOptions * dinnerOptions = 72 :=
by
  let lunchChinese := 5
  let lunchJapanese := 4
  let dinnerChinese := 3
  let dinnerJapanese := 5
  let lunchOptions := lunchChinese + lunchJapanese
  let dinnerOptions := dinnerChinese + dinnerJapanese
  have h : lunchOptions * dinnerOptions = 72 :=
    by 
      sorry
  exact h

end NUMINAMATH_GPT_menu_choices_l1056_105687


namespace NUMINAMATH_GPT_union_dues_proof_l1056_105682

noncomputable def h : ℕ := 42
noncomputable def r : ℕ := 10
noncomputable def tax_rate : ℝ := 0.20
noncomputable def insurance_rate : ℝ := 0.05
noncomputable def take_home_pay : ℝ := 310

noncomputable def gross_earnings : ℝ := h * r
noncomputable def tax_deduction : ℝ := tax_rate * gross_earnings
noncomputable def insurance_deduction : ℝ := insurance_rate * gross_earnings
noncomputable def total_deductions : ℝ := tax_deduction + insurance_deduction
noncomputable def net_earnings_before_union_dues : ℝ := gross_earnings - total_deductions
noncomputable def union_dues_deduction : ℝ := net_earnings_before_union_dues - take_home_pay

theorem union_dues_proof : union_dues_deduction = 5 := 
by sorry

end NUMINAMATH_GPT_union_dues_proof_l1056_105682


namespace NUMINAMATH_GPT_min_distance_between_intersections_range_of_a_l1056_105640

variable {a : ℝ}

/-- Given the function f(x) = x^2 - 2ax - 2(a + 1), 
1. Prove that the graph of function f(x) always intersects the x-axis at two distinct points.
2. For all x in the interval (-1, ∞), prove that f(x) + 3 ≥ 0 implies a ≤ sqrt 2 - 1. --/

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 2 * (a + 1)

theorem min_distance_between_intersections (a : ℝ) : 
  ∃ x₁ x₂ : ℝ, (f x₁ a = 0) ∧ (f x₂ a = 0) ∧ (x₁ ≠ x₂) ∧ (dist x₁ x₂ = 2) := sorry

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, -1 < x → f x a + 3 ≥ 0) → a ≤ Real.sqrt 2 - 1 := sorry

end NUMINAMATH_GPT_min_distance_between_intersections_range_of_a_l1056_105640


namespace NUMINAMATH_GPT_initial_kittens_l1056_105673

-- Define the number of kittens given to Jessica and Sara, and the number of kittens currently Tim has.
def kittens_given_to_Jessica : ℕ := 3
def kittens_given_to_Sara : ℕ := 6
def kittens_left_with_Tim : ℕ := 9

-- Define the theorem to prove the initial number of kittens Tim had.
theorem initial_kittens (kittens_given_to_Jessica kittens_given_to_Sara kittens_left_with_Tim : ℕ) 
    (h1 : kittens_given_to_Jessica = 3)
    (h2 : kittens_given_to_Sara = 6)
    (h3 : kittens_left_with_Tim = 9) :
    (kittens_given_to_Jessica + kittens_given_to_Sara + kittens_left_with_Tim) = 18 := 
    sorry

end NUMINAMATH_GPT_initial_kittens_l1056_105673


namespace NUMINAMATH_GPT_average_age_new_students_l1056_105676

theorem average_age_new_students (A : ℚ)
    (avg_original_age : ℚ := 48)
    (num_new_students : ℚ := 120)
    (new_avg_age : ℚ := 44)
    (total_students : ℚ := 160) :
    let num_original_students := total_students - num_new_students
    let total_age_original := num_original_students * avg_original_age
    let total_age_all := total_students * new_avg_age
    total_age_original + (num_new_students * A) = total_age_all → A = 42.67 := 
by
  intros
  sorry

end NUMINAMATH_GPT_average_age_new_students_l1056_105676


namespace NUMINAMATH_GPT_segment_proportionality_l1056_105616

variable (a b c x : ℝ)

theorem segment_proportionality (ha : a ≠ 0) (hc : c ≠ 0) 
  (h : x = a * (b / c)) : 
  (x / a) = (b / c) := 
by
  sorry

end NUMINAMATH_GPT_segment_proportionality_l1056_105616


namespace NUMINAMATH_GPT_compound_interest_comparison_l1056_105693

theorem compound_interest_comparison :
  (1 + 0.04) < (1 + 0.04 / 12) ^ 12 := sorry

end NUMINAMATH_GPT_compound_interest_comparison_l1056_105693


namespace NUMINAMATH_GPT_candy_bar_calories_l1056_105688

theorem candy_bar_calories
  (miles_walked : ℕ)
  (calories_per_mile : ℕ)
  (net_calorie_deficit : ℕ)
  (total_calories_burned : ℕ)
  (candy_bar_calories : ℕ)
  (h1 : miles_walked = 3)
  (h2 : calories_per_mile = 150)
  (h3 : net_calorie_deficit = 250)
  (h4 : total_calories_burned = miles_walked * calories_per_mile)
  (h5 : candy_bar_calories = total_calories_burned - net_calorie_deficit) :
  candy_bar_calories = 200 := 
by
  sorry

end NUMINAMATH_GPT_candy_bar_calories_l1056_105688


namespace NUMINAMATH_GPT_sum_of_m_and_n_l1056_105659

theorem sum_of_m_and_n (m n : ℚ) (h : (m - 3) * (Real.sqrt 5) + 2 - n = 0) : m + n = 5 :=
sorry

end NUMINAMATH_GPT_sum_of_m_and_n_l1056_105659


namespace NUMINAMATH_GPT_solve_equation_l1056_105610

theorem solve_equation (x : ℝ) (h1 : x + 2 ≠ 0) (h2 : 3 - x ≠ 0) :
  (3 * x - 5) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -15 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1056_105610


namespace NUMINAMATH_GPT_tan_subtraction_l1056_105649

theorem tan_subtraction (α β : ℝ) (h₁ : Real.tan α = 11) (h₂ : Real.tan β = 5) : 
  Real.tan (α - β) = 3 / 28 := 
  sorry

end NUMINAMATH_GPT_tan_subtraction_l1056_105649


namespace NUMINAMATH_GPT_tickets_needed_to_ride_l1056_105613

noncomputable def tickets_required : Float :=
let ferris_wheel := 3.5
let roller_coaster := 8.0
let bumper_cars := 5.0
let additional_ride_discount := 0.5
let newspaper_coupon := 1.5
let teacher_discount := 2.0

let total_cost_without_discounts := ferris_wheel + roller_coaster + bumper_cars
let total_additional_discounts := additional_ride_discount * 2
let total_coupons_discounts := newspaper_coupon + teacher_discount

let total_cost_with_discounts := total_cost_without_discounts - total_additional_discounts - total_coupons_discounts
total_cost_with_discounts

theorem tickets_needed_to_ride : tickets_required = 12.0 := by
  sorry

end NUMINAMATH_GPT_tickets_needed_to_ride_l1056_105613


namespace NUMINAMATH_GPT_certain_number_is_60_l1056_105615

theorem certain_number_is_60 
  (A J C : ℕ) 
  (h1 : A = 4) 
  (h2 : C = 8) 
  (h3 : A = (1 / 2) * J) :
  3 * (A + J + C) = 60 :=
by sorry

end NUMINAMATH_GPT_certain_number_is_60_l1056_105615


namespace NUMINAMATH_GPT_sum_reciprocals_factors_12_l1056_105686

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_reciprocals_factors_12_l1056_105686


namespace NUMINAMATH_GPT_problem_statement_l1056_105618

theorem problem_statement :
  (∀ x : ℝ, |x| < 2 → x < 3) ∧
  (∀ x : ℝ, ¬ (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ∧
  (-1 < m ∧ m < 0 → ∀ a b : ℝ, a ≠ b → (a * b > 0)) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1056_105618


namespace NUMINAMATH_GPT_find_m_l1056_105685

noncomputable def g (d e f x : ℤ) : ℤ := d * x * x + e * x + f

theorem find_m (d e f m : ℤ) (h₁ : g d e f 2 = 0)
    (h₂ : 60 < g d e f 6 ∧ g d e f 6 < 70) 
    (h₃ : 80 < g d e f 9 ∧ g d e f 9 < 90)
    (h₄ : 10000 * m < g d e f 100 ∧ g d e f 100 < 10000 * (m + 1)) :
  m = -1 :=
sorry

end NUMINAMATH_GPT_find_m_l1056_105685


namespace NUMINAMATH_GPT_min_expr_l1056_105665

theorem min_expr (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : a * b = 1) :
  ∃ s : ℝ, (s = a + b) ∧ (s ≥ 2) ∧ (a^2 + b^2 + 4/(s^2) = 3) :=
by sorry

end NUMINAMATH_GPT_min_expr_l1056_105665


namespace NUMINAMATH_GPT_ratio_a_c_l1056_105646

theorem ratio_a_c {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : ∀ x : ℝ, x^2 + a * x + b = 0 → 2 * x^2 + 2 * a * x + 2 * b = 0)
  (h2 : ∀ x : ℝ, x^2 + b * x + c = 0 → x^2 + b * x + c = 0) :
  a / c = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_a_c_l1056_105646


namespace NUMINAMATH_GPT_total_acorns_l1056_105697

theorem total_acorns (x y : ℝ) :
  let sheila_acorns := 5.3 * x
  let danny_acorns := sheila_acorns + y
  x + sheila_acorns + danny_acorns = 11.6 * x + y :=
by
  sorry

end NUMINAMATH_GPT_total_acorns_l1056_105697


namespace NUMINAMATH_GPT_man_l1056_105675

theorem man's_speed_with_stream
  (V_m V_s : ℝ)
  (h1 : V_m = 6)
  (h2 : V_m - V_s = 4) :
  V_m + V_s = 8 :=
sorry

end NUMINAMATH_GPT_man_l1056_105675


namespace NUMINAMATH_GPT_workers_are_280_women_l1056_105617

variables (W : ℕ) 
          (workers_without_retirement_plan : ℕ := W / 3)
          (women_without_retirement_plan : ℕ := (workers_without_retirement_plan * 1) / 10)
          (workers_with_retirement_plan : ℕ := W * 2 / 3)
          (men_with_retirement_plan : ℕ := (workers_with_retirement_plan * 4) / 10)
          (total_men : ℕ := (workers_without_retirement_plan * 9) / 30)
          (total_workers := total_men / (9 / 30))
          (number_of_women : ℕ := total_workers - 120)

theorem workers_are_280_women : total_workers = 400 ∧ number_of_women = 280 :=
by sorry

end NUMINAMATH_GPT_workers_are_280_women_l1056_105617


namespace NUMINAMATH_GPT_part1_part2_l1056_105600

open Real

def f (x a : ℝ) : ℝ :=
  x^2 + a * x + 3

theorem part1 (x : ℝ) (h : x^2 - 4 * x + 3 < 0) :
  1 < x ∧ x < 3 :=
  sorry

theorem part2 (a : ℝ) (h : ∀ x, f x a > 0) :
  -2 * sqrt 3 < a ∧ a < 2 * sqrt 3 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1056_105600


namespace NUMINAMATH_GPT_N_subset_M_l1056_105641

-- Definitions of sets M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | x * x - x < 0 }

-- Proof statement: N is a subset of M
theorem N_subset_M : N ⊆ M :=
sorry

end NUMINAMATH_GPT_N_subset_M_l1056_105641


namespace NUMINAMATH_GPT_chromosome_structure_l1056_105636

-- Definitions related to the conditions of the problem
def chromosome : Type := sorry  -- Define type for chromosome (hypothetical representation)
def has_centromere (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome has centromere
def contains_one_centromere (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome contains one centromere
def has_one_chromatid (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome has one chromatid
def has_two_chromatids (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome has two chromatids
def is_chromatin (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome is chromatin

-- Define the problem statement
theorem chromosome_structure (c : chromosome) :
  contains_one_centromere c ∧ ¬has_one_chromatid c ∧ ¬has_two_chromatids c ∧ ¬is_chromatin c := sorry

end NUMINAMATH_GPT_chromosome_structure_l1056_105636


namespace NUMINAMATH_GPT_total_pamphlets_correct_l1056_105625

-- Define the individual printing rates and hours
def Mike_pre_break_rate := 600
def Mike_pre_break_hours := 9
def Mike_post_break_rate := Mike_pre_break_rate / 3
def Mike_post_break_hours := 2

def Leo_pre_break_rate := 2 * Mike_pre_break_rate
def Leo_pre_break_hours := Mike_pre_break_hours / 3
def Leo_post_first_break_rate := Leo_pre_break_rate / 2
def Leo_post_second_break_rate := Leo_post_first_break_rate / 2

def Sally_pre_break_rate := 3 * Mike_pre_break_rate
def Sally_pre_break_hours := Mike_post_break_hours / 2
def Sally_post_break_rate := Leo_post_first_break_rate
def Sally_post_break_hours := 1

-- Calculate the total number of pamphlets printed by each person
def Mike_pamphlets := 
  (Mike_pre_break_rate * Mike_pre_break_hours) + (Mike_post_break_rate * Mike_post_break_hours)

def Leo_pamphlets := 
  (Leo_pre_break_rate * 1) + (Leo_post_first_break_rate * 1) + (Leo_post_second_break_rate * 1)

def Sally_pamphlets := 
  (Sally_pre_break_rate * Sally_pre_break_hours) + (Sally_post_break_rate * Sally_post_break_hours)

-- Calculate the total number of pamphlets printed by all three
def total_pamphlets := Mike_pamphlets + Leo_pamphlets + Sally_pamphlets

theorem total_pamphlets_correct : total_pamphlets = 10700 := by
  sorry

end NUMINAMATH_GPT_total_pamphlets_correct_l1056_105625


namespace NUMINAMATH_GPT_sets_given_to_friend_l1056_105650

theorem sets_given_to_friend (total_cards : ℕ) (total_given_away : ℕ) (sets_brother : ℕ) 
  (sets_sister : ℕ) (cards_per_set : ℕ) (sets_friend : ℕ) 
  (h1 : total_cards = 365) 
  (h2 : total_given_away = 195) 
  (h3 : sets_brother = 8) 
  (h4 : sets_sister = 5) 
  (h5 : cards_per_set = 13) 
  (h6 : total_given_away = (sets_brother + sets_sister + sets_friend) * cards_per_set) : 
  sets_friend = 2 :=
by
  sorry

end NUMINAMATH_GPT_sets_given_to_friend_l1056_105650


namespace NUMINAMATH_GPT_smallest_positive_n_l1056_105633

theorem smallest_positive_n (n : ℕ) (h1 : 0 < n) (h2 : gcd (8 * n - 3) (6 * n + 4) > 1) : n = 1 :=
sorry

end NUMINAMATH_GPT_smallest_positive_n_l1056_105633


namespace NUMINAMATH_GPT_min_value_5_5_l1056_105643

noncomputable def given_expression (x y z : ℝ) : ℝ :=
  (6 * z) / (2 * x + y) + (6 * x) / (y + 2 * z) + (4 * y) / (x + z)

theorem min_value_5_5 (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z = 1) :
  given_expression x y z ≥ 5.5 :=
sorry

end NUMINAMATH_GPT_min_value_5_5_l1056_105643


namespace NUMINAMATH_GPT_subtraction_proof_l1056_105628

theorem subtraction_proof :
  2000000000000 - 1111111111111 - 222222222222 = 666666666667 :=
by sorry

end NUMINAMATH_GPT_subtraction_proof_l1056_105628


namespace NUMINAMATH_GPT_count_perfect_cubes_l1056_105670

theorem count_perfect_cubes (a b : ℕ) (h1 : a = 200) (h2 : b = 1600) :
  ∃ (n : ℕ), n = 6 :=
by
  sorry

end NUMINAMATH_GPT_count_perfect_cubes_l1056_105670


namespace NUMINAMATH_GPT_area_ratio_l1056_105609

noncomputable def AreaOfTrapezoid (AD BC : ℝ) (R : ℝ) : ℝ :=
  let s_π := Real.pi
  let height1 := 2 -- One of the heights considered
  let height2 := 14 -- Another height considered
  (AD + BC) / 2 * height1  -- First case area
  -- Here we assume the area uses sine which is arc-related, but provide fixed coefficients for area representation

noncomputable def AreaOfRectangle (R : ℝ) : ℝ :=
  let d := 2 * R
  -- Using the equation for area discussed
  d * d / 2

theorem area_ratio (AD BC : ℝ) (R : ℝ) (hAD : AD = 16) (hBC : BC = 12) (hR : R = 10) :
  let area_trap := AreaOfTrapezoid AD BC R
  let area_rect := AreaOfRectangle R
  area_trap / area_rect = 1 / 2 ∨ area_trap / area_rect = 49 / 50 :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_l1056_105609


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l1056_105672

theorem ratio_of_x_to_y (x y : ℝ) (R : ℝ) (h1 : x = R * y) (h2 : x - y = 0.909090909090909 * x) : R = 11 := by
  sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l1056_105672


namespace NUMINAMATH_GPT_system_no_solution_l1056_105683

theorem system_no_solution (n : ℝ) :
  ∃ x y z : ℝ, (n * x + y = 1) ∧ (1 / 2 * n * y + z = 1) ∧ (x + 1 / 2 * n * z = 2) ↔ n = -1 := 
sorry

end NUMINAMATH_GPT_system_no_solution_l1056_105683


namespace NUMINAMATH_GPT_fg_sqrt2_eq_neg5_l1056_105621

noncomputable def f (x : ℝ) : ℝ := 4 - 3 * x
noncomputable def g (x : ℝ) : ℝ := x^2 + 1

theorem fg_sqrt2_eq_neg5 : f (g (Real.sqrt 2)) = -5 := by
  sorry

end NUMINAMATH_GPT_fg_sqrt2_eq_neg5_l1056_105621


namespace NUMINAMATH_GPT_product_of_roots_l1056_105660

theorem product_of_roots :
  ∃ x₁ x₂ : ℝ, (x₁ * x₂ = -4) ∧ (x₁ ^ 2 + 2 * x₁ - 4 = 0) ∧ (x₂ ^ 2 + 2 * x₂ - 4 = 0) := by
  sorry

end NUMINAMATH_GPT_product_of_roots_l1056_105660


namespace NUMINAMATH_GPT_cost_price_percentage_l1056_105645

theorem cost_price_percentage (MP CP : ℝ) 
  (h1 : MP * 0.9 = CP * (72 / 70))
  (h2 : CP / MP * 100 = 87.5) :
  CP / MP = 0.875 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_price_percentage_l1056_105645


namespace NUMINAMATH_GPT_ab_sum_l1056_105605

theorem ab_sum (a b : ℝ) (h₁ : ∀ x : ℝ, (x + a) * (x + 8) = x^2 + b * x + 24) (h₂ : 8 * a = 24) : a + b = 14 :=
by
  sorry

end NUMINAMATH_GPT_ab_sum_l1056_105605


namespace NUMINAMATH_GPT_part1_l1056_105634

variables {a b c : ℝ}
theorem part1 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : a / (b + c) = b / (c + a) - c / (a + b)) : 
    b / (c + a) ≥ (Real.sqrt 17 - 1) / 4 :=
sorry

end NUMINAMATH_GPT_part1_l1056_105634


namespace NUMINAMATH_GPT_peter_reads_more_books_l1056_105655

-- Definitions and conditions
def total_books : ℕ := 20
def peter_percentage_read : ℕ := 40
def brother_percentage_read : ℕ := 10

def percentage_to_count (percentage : ℕ) (total : ℕ) : ℕ := (percentage * total) / 100

-- Main statement to prove
theorem peter_reads_more_books :
  percentage_to_count peter_percentage_read total_books - percentage_to_count brother_percentage_read total_books = 6 :=
by
  sorry

end NUMINAMATH_GPT_peter_reads_more_books_l1056_105655


namespace NUMINAMATH_GPT_technician_round_trip_percentage_l1056_105667

theorem technician_round_trip_percentage (D: ℝ) (hD: D ≠ 0): 
  let round_trip_distance := 2 * D
  let distance_to_center := D
  let distance_back_10_percent := 0.10 * D
  let total_distance_completed := distance_to_center + distance_back_10_percent
  let percentage_completed := (total_distance_completed / round_trip_distance) * 100
  percentage_completed = 55 := 
by
  simp
  sorry -- Proof is not required per instructions

end NUMINAMATH_GPT_technician_round_trip_percentage_l1056_105667


namespace NUMINAMATH_GPT_range_of_a_l1056_105631

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (ax^2 - ax + 1 ≤ 0)) ↔ 0 ≤ a ∧ a < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1056_105631


namespace NUMINAMATH_GPT_triangular_number_30_sum_of_first_30_triangular_numbers_l1056_105690

theorem triangular_number_30 
  (T : ℕ → ℕ)
  (hT : ∀ n : ℕ, T n = n * (n + 1) / 2) : 
  T 30 = 465 :=
by
  -- Skipping proof with sorry
  sorry

theorem sum_of_first_30_triangular_numbers 
  (S : ℕ → ℕ)
  (hS : ∀ n : ℕ, S n = n * (n + 1) * (n + 2) / 6) : 
  S 30 = 4960 :=
by
  -- Skipping proof with sorry
  sorry

end NUMINAMATH_GPT_triangular_number_30_sum_of_first_30_triangular_numbers_l1056_105690


namespace NUMINAMATH_GPT_algebraic_expression_identity_l1056_105680

theorem algebraic_expression_identity (a b x : ℕ) (h : x * 3 * a * b = 3 * a * a * b) : x = a :=
sorry

end NUMINAMATH_GPT_algebraic_expression_identity_l1056_105680


namespace NUMINAMATH_GPT_complex_exp_cos_l1056_105674

theorem complex_exp_cos (z : ℂ) (α : ℂ) (n : ℕ) (h : z + z⁻¹ = 2 * Complex.cos α) : 
  z^n + z⁻¹^n = 2 * Complex.cos (n * α) :=
by
  sorry

end NUMINAMATH_GPT_complex_exp_cos_l1056_105674


namespace NUMINAMATH_GPT_complete_square_proof_l1056_105644

def complete_square (x : ℝ) : Prop :=
  x^2 - 2 * x - 8 = 0 -> (x - 1)^2 = 9

theorem complete_square_proof (x : ℝ) :
  complete_square x :=
sorry

end NUMINAMATH_GPT_complete_square_proof_l1056_105644


namespace NUMINAMATH_GPT_geometric_condition_l1056_105671

def Sn (p : ℤ) (n : ℕ) : ℤ := p * 2^n + 2

def an (p : ℤ) (n : ℕ) : ℤ :=
  if n = 1 then Sn p n
  else Sn p n - Sn p (n - 1)

def is_geometric_progression (p : ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → ∃ r : ℤ, an p n = an p (n - 1) * r

theorem geometric_condition (p : ℤ) :
  is_geometric_progression p ↔ p = -2 :=
sorry

end NUMINAMATH_GPT_geometric_condition_l1056_105671


namespace NUMINAMATH_GPT_functional_equation_solution_l1056_105684

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 2 * y * f x) :
  ∀ x : ℝ, f x = 0 := 
sorry

end NUMINAMATH_GPT_functional_equation_solution_l1056_105684


namespace NUMINAMATH_GPT_landmark_distance_l1056_105681

theorem landmark_distance (d : ℝ) : 
  (d >= 7 → d < 7) ∨ (d <= 8 → d > 8) ∨ (d <= 10 → d > 10) → d > 10 :=
by
  sorry

end NUMINAMATH_GPT_landmark_distance_l1056_105681


namespace NUMINAMATH_GPT_team_A_wins_2_1_team_B_wins_l1056_105663

theorem team_A_wins_2_1 (p_a p_b : ℝ)
  (h1 : p_a = 0.6)
  (h2 : p_b = 0.4)
  (h3 : ∀ {x y: ℝ}, x + y = 1)
  (h4 : ∃ n : ℕ, n = 3) : (2 * p_a * p_b) * p_a = 0.288 := by
  sorry

theorem team_B_wins (p_a p_b : ℝ)
  (h1 : p_a = 0.6)
  (h2 : p_b = 0.4)
  (h3 : ∀ {x y: ℝ}, x + y = 1)
  (h4 : ∃ n : ℕ, n = 3) : (p_b * p_b) + (2 * p_a * p_b * p_b) = 0.352 := by
  sorry

end NUMINAMATH_GPT_team_A_wins_2_1_team_B_wins_l1056_105663


namespace NUMINAMATH_GPT_students_not_yet_pictured_l1056_105652

def students_in_class : ℕ := 24
def students_before_lunch : ℕ := students_in_class / 3
def students_after_lunch_before_gym : ℕ := 10
def total_students_pictures_taken : ℕ := students_before_lunch + students_after_lunch_before_gym

theorem students_not_yet_pictured : total_students_pictures_taken = 18 → students_in_class - total_students_pictures_taken = 6 := by
  intros h
  rw [h]
  rfl

end NUMINAMATH_GPT_students_not_yet_pictured_l1056_105652


namespace NUMINAMATH_GPT_inequality_transform_l1056_105604

variable {x y : ℝ}

theorem inequality_transform (h : x < y) : - (x / 2) > - (y / 2) :=
sorry

end NUMINAMATH_GPT_inequality_transform_l1056_105604


namespace NUMINAMATH_GPT_average_weight_of_removed_onions_l1056_105603

theorem average_weight_of_removed_onions (total_weight_40_onions : ℝ := 7680)
    (average_weight_35_onions : ℝ := 190)
    (number_of_onions_removed : ℕ := 5)
    (total_onions_initial : ℕ := 40)
    (total_number_of_remaining_onions : ℕ := 35) :
    (total_weight_40_onions - total_number_of_remaining_onions * average_weight_35_onions) / number_of_onions_removed = 206 :=
by
    sorry

end NUMINAMATH_GPT_average_weight_of_removed_onions_l1056_105603


namespace NUMINAMATH_GPT_min_number_of_trials_sum_15_min_number_of_trials_sum_at_least_15_l1056_105639

noncomputable def min_trials_sum_of_15 : ℕ :=
  15

noncomputable def min_trials_sum_at_least_15 : ℕ :=
  8

theorem min_number_of_trials_sum_15 (x : ℕ) :
  (∀ (x : ℕ), (103/108 : ℝ)^x < (1/2 : ℝ) → x >= min_trials_sum_of_15) := sorry

theorem min_number_of_trials_sum_at_least_15 (x : ℕ) :
  (∀ (x : ℕ), (49/54 : ℝ)^x < (1/2 : ℝ) → x >= min_trials_sum_at_least_15) := sorry

end NUMINAMATH_GPT_min_number_of_trials_sum_15_min_number_of_trials_sum_at_least_15_l1056_105639


namespace NUMINAMATH_GPT_common_roots_correct_l1056_105654

noncomputable section
def common_roots_product (A B : ℝ) : ℝ :=
  let p := sorry
  let q := sorry
  p * q

theorem common_roots_correct (A B : ℝ) (h1 : ∀ x, x^3 + 2*A*x + 20 = 0 → x = p ∨ x = q ∨ x = r) 
    (h2 : ∀ x, x^3 + B*x^2 + 100 = 0 → x = p ∨ x = q ∨ x = s)
    (h_sum1 : p + q + r = 0) 
    (h_sum2 : p + q + s = -B)
    (h_prod1 : p * q * r = -20) 
    (h_prod2 : p * q * s = -100) : 
    common_roots_product A B = 10 * (2000)^(1/3) ∧ 15 = 10 + 3 + 2 :=
by
  sorry

end NUMINAMATH_GPT_common_roots_correct_l1056_105654


namespace NUMINAMATH_GPT_water_park_children_l1056_105679

theorem water_park_children (cost_adult cost_child total_cost : ℝ) (c : ℕ) 
  (h1 : cost_adult = 1)
  (h2 : cost_child = 0.75)
  (h3 : total_cost = 3.25) :
  c = 3 :=
by
  sorry

end NUMINAMATH_GPT_water_park_children_l1056_105679


namespace NUMINAMATH_GPT_real_roots_prime_equation_l1056_105612

noncomputable def has_rational_roots (p q : ℕ) : Prop :=
  ∃ x : ℚ, x^2 + p^2 * x + q^3 = 0

theorem real_roots_prime_equation (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  has_rational_roots p q ↔ (p = 3 ∧ q = 2) :=
sorry

end NUMINAMATH_GPT_real_roots_prime_equation_l1056_105612
