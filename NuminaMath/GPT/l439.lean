import Mathlib

namespace NUMINAMATH_GPT_ratio_a_b_is_zero_l439_43910

-- Setting up the conditions
variables (a y b : ℝ)
variable (d : ℝ)
-- Condition for arithmetic sequence
axiom h1 : a + d = y
axiom h2 : y + d = b
axiom h3 : b + d = 3 * y

-- The Lean statement to prove
theorem ratio_a_b_is_zero (h1 : a + d = y) (h2 : y + d = b) (h3 : b + d = 3 * y) : a / b = 0 :=
sorry

end NUMINAMATH_GPT_ratio_a_b_is_zero_l439_43910


namespace NUMINAMATH_GPT_find_ratio_l439_43904

def celsius_to_fahrenheit_ratio (ratio : ℝ) (c f : ℝ) : Prop :=
  f = ratio * c + 32

theorem find_ratio (ratio : ℝ) :
  (∀ c f, celsius_to_fahrenheit_ratio ratio c f ∧ ((f = 58) → (c = 14.444444444444445)) → f = 1.8 * c + 32) ∧ 
  (f - 32 = ratio * (c - 0)) ∧
  (c = 14.444444444444445 → f = 32 + 26) ∧
  (f = 58 → c = 14.444444444444445) ∧ 
  (ratio = 1.8)
  → ratio = 1.8 := 
sorry 


end NUMINAMATH_GPT_find_ratio_l439_43904


namespace NUMINAMATH_GPT_percentage_not_red_roses_l439_43941

-- Definitions for the conditions
def roses : Nat := 25
def tulips : Nat := 40
def daisies : Nat := 60
def lilies : Nat := 15
def sunflowers : Nat := 10
def totalFlowers : Nat := roses + tulips + daisies + lilies + sunflowers -- 150
def redRoses : Nat := roses / 2 -- 12 (considering integer division)

-- Statement to prove
theorem percentage_not_red_roses : 
  ((totalFlowers - redRoses) * 100 / totalFlowers) = 92 := by
  sorry

end NUMINAMATH_GPT_percentage_not_red_roses_l439_43941


namespace NUMINAMATH_GPT_find_angle_D_l439_43908

variable (A B C D : ℝ)
variable (h1 : A + B = 180)
variable (h2 : C = D)
variable (h3 : C + 50 + 60 = 180)

theorem find_angle_D : D = 70 := by
  sorry

end NUMINAMATH_GPT_find_angle_D_l439_43908


namespace NUMINAMATH_GPT_circle_radius_l439_43959

theorem circle_radius (x y : ℝ) :
  (∃ r, r > 0 ∧ (∀ x y, x^2 - 8*x + y^2 - 4*y + 16 = 0 → r = 2)) :=
sorry

end NUMINAMATH_GPT_circle_radius_l439_43959


namespace NUMINAMATH_GPT_y_value_when_x_is_20_l439_43940

theorem y_value_when_x_is_20 :
  ∀ (x : ℝ), (∀ m c : ℝ, m = 2.5 → c = 3 → (y = m * x + c) → x = 20 → y = 53) :=
by
  sorry

end NUMINAMATH_GPT_y_value_when_x_is_20_l439_43940


namespace NUMINAMATH_GPT_age_of_son_l439_43903

theorem age_of_son (S F : ℕ) (h1 : F = S + 28) (h2 : F + 2 = 2 * (S + 2)) : S = 26 := 
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_age_of_son_l439_43903


namespace NUMINAMATH_GPT_find_number_to_be_multiplied_l439_43961

def correct_multiplier := 43
def incorrect_multiplier := 34
def difference := 1224

theorem find_number_to_be_multiplied (x : ℕ) : correct_multiplier * x - incorrect_multiplier * x = difference → x = 136 :=
by
  sorry

end NUMINAMATH_GPT_find_number_to_be_multiplied_l439_43961


namespace NUMINAMATH_GPT_shiela_paintings_l439_43995

theorem shiela_paintings (h1 : 18 % 2 = 0) : 18 / 2 = 9 := 
by sorry

end NUMINAMATH_GPT_shiela_paintings_l439_43995


namespace NUMINAMATH_GPT_remainder_is_three_l439_43918

def dividend : ℕ := 15
def divisor : ℕ := 3
def quotient : ℕ := 4

theorem remainder_is_three : dividend = (divisor * quotient) + Nat.mod dividend divisor := by
  sorry

end NUMINAMATH_GPT_remainder_is_three_l439_43918


namespace NUMINAMATH_GPT_num_zeros_in_interval_l439_43934

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 6 * x ^ 2 + 7

theorem num_zeros_in_interval : 
    (∃ (a b : ℝ), a < b ∧ a = 0 ∧ b = 2 ∧
     (∀ x, f x = 0 → (0 < x ∧ x < 2)) ∧
     (∃! x, (0 < x ∧ x < 2) ∧ f x = 0)) :=
by
    sorry

end NUMINAMATH_GPT_num_zeros_in_interval_l439_43934


namespace NUMINAMATH_GPT_find_number_of_lines_l439_43942

theorem find_number_of_lines (n : ℕ) (h : (n * (n - 1) / 2) * 8 = 280) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_lines_l439_43942


namespace NUMINAMATH_GPT_impossible_to_arrange_circle_l439_43991

theorem impossible_to_arrange_circle : 
  ¬∃ (f : Fin 10 → Fin 10), 
    (∀ i : Fin 10, (abs ((f i).val - (f (i + 1)).val : Int) = 3 
                ∨ abs ((f i).val - (f (i + 1)).val : Int) = 4 
                ∨ abs ((f i).val - (f (i + 1)).val : Int) = 5)) :=
sorry

end NUMINAMATH_GPT_impossible_to_arrange_circle_l439_43991


namespace NUMINAMATH_GPT_smallest_m_l439_43980

theorem smallest_m (m : ℕ) (p q : ℤ) (h_eq : 10*(p:ℤ)^2 - m*(p:ℤ) + 360 = 0) (h_cond : q = 2 * p) :
  p * q = 36 → 3 * p + 3 * q = m → m = 90 :=
by sorry

end NUMINAMATH_GPT_smallest_m_l439_43980


namespace NUMINAMATH_GPT_cost_price_correct_l439_43968

variables (sp : ℕ) (profitPerMeter : ℕ) (metersSold : ℕ)

def total_profit (profitPerMeter metersSold : ℕ) : ℕ := profitPerMeter * metersSold
def total_cost_price (sp total_profit : ℕ) : ℕ := sp - total_profit
def cost_price_per_meter (total_cost_price metersSold : ℕ) : ℕ := total_cost_price / metersSold

theorem cost_price_correct (h1 : sp = 8925) (h2 : profitPerMeter = 10) (h3 : metersSold = 85) :
  cost_price_per_meter (total_cost_price sp (total_profit profitPerMeter metersSold)) metersSold = 95 :=
by
  rw [h1, h2, h3];
  sorry

end NUMINAMATH_GPT_cost_price_correct_l439_43968


namespace NUMINAMATH_GPT_loss_percentage_l439_43945

/--
A man sells a car to his friend at a certain loss percentage. The friend then sells it 
for Rs. 54000 and gains 20%. The original cost price of the car was Rs. 52941.17647058824.
Prove that the loss percentage when the man sold the car to his friend was 15%.
-/
theorem loss_percentage (CP SP_2 : ℝ) (gain_percent : ℝ) (h_CP : CP = 52941.17647058824) 
(h_SP2 : SP_2 = 54000) (h_gain : gain_percent = 20) : (CP - SP_2 / (1 + gain_percent / 100)) / CP * 100 = 15 := by
  sorry

end NUMINAMATH_GPT_loss_percentage_l439_43945


namespace NUMINAMATH_GPT_two_pow_a_plus_two_pow_neg_a_l439_43921

theorem two_pow_a_plus_two_pow_neg_a (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 1) :
  2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_two_pow_a_plus_two_pow_neg_a_l439_43921


namespace NUMINAMATH_GPT_first_term_geometric_sequence_l439_43973

theorem first_term_geometric_sequence (a r : ℕ) (h1 : r = 3) (h2 : a * r^4 = 81) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_first_term_geometric_sequence_l439_43973


namespace NUMINAMATH_GPT_work_days_l439_43905

theorem work_days (hp : ℝ) (hq : ℝ) (fraction_left : ℝ) (d : ℝ) :
  hp = 1 / 20 → hq = 1 / 10 → fraction_left = 0.7 → (3 / 20) * d = (1 - fraction_left) → d = 2 :=
  by
  intros hp_def hq_def fraction_def work_eq
  sorry

end NUMINAMATH_GPT_work_days_l439_43905


namespace NUMINAMATH_GPT_afternoon_sales_l439_43974

theorem afternoon_sales (x : ℕ) (H1 : 2 * x + x = 390) : 2 * x = 260 :=
by
  sorry

end NUMINAMATH_GPT_afternoon_sales_l439_43974


namespace NUMINAMATH_GPT_find_constant_b_l439_43906

theorem find_constant_b 
  (a b c : ℝ)
  (h1 : 3 * a = 9) 
  (h2 : (-2 * a + 3 * b) = -5) 
  : b = 1 / 3 :=
by 
  have h_a : a = 3 := by linarith
  
  have h_b : -2 * 3 + 3 * b = -5 := by linarith [h2]
  
  linarith

end NUMINAMATH_GPT_find_constant_b_l439_43906


namespace NUMINAMATH_GPT_max_vertex_sum_l439_43972

theorem max_vertex_sum
  (a U : ℤ)
  (hU : U ≠ 0)
  (hA : 0 = a * 0 * (0 - 3 * U))
  (hB : 0 = a * (3 * U) * ((3 * U) - 3 * U))
  (hC : 12 = a * (3 * U - 1) * ((3 * U - 1) - 3 * U))
  : ∃ N : ℝ, N = (3 * U) / 2 - (9 * a * U^2) / 4 ∧ N ≤ 17.75 :=
by sorry

end NUMINAMATH_GPT_max_vertex_sum_l439_43972


namespace NUMINAMATH_GPT_least_six_digit_divisible_by_198_l439_43984

/-- The least 6-digit natural number that is divisible by 198 is 100188. -/
theorem least_six_digit_divisible_by_198 : 
  ∃ n : ℕ, n ≥ 100000 ∧ n % 198 = 0 ∧ n = 100188 :=
by
  use 100188
  sorry

end NUMINAMATH_GPT_least_six_digit_divisible_by_198_l439_43984


namespace NUMINAMATH_GPT_small_disks_radius_l439_43962

theorem small_disks_radius (r : ℝ) (h : r > 0) :
  (2 * r ≥ 1 + r) → (r ≥ 1 / 2) := by
  intro hr
  linarith

end NUMINAMATH_GPT_small_disks_radius_l439_43962


namespace NUMINAMATH_GPT_train_length_l439_43986

theorem train_length (speed_km_per_hr : ℕ) (time_sec : ℕ) (h_speed : speed_km_per_hr = 80) (h_time : time_sec = 9) :
  ∃ length_m : ℕ, length_m = 200 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l439_43986


namespace NUMINAMATH_GPT_geometric_number_difference_l439_43938

-- Definitions
def is_geometric_sequence (a b c d : ℕ) : Prop := ∃ r : ℚ, b = a * r ∧ c = a * r^2 ∧ d = a * r^3

def is_valid_geometric_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧  -- 4-digit number
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ -- distinct digits
    is_geometric_sequence a b c d ∧ -- geometric sequence
    n = a * 1000 + b * 100 + c * 10 + d -- digits form the number

-- Theorem statement
theorem geometric_number_difference : 
  ∃ (m M : ℕ), is_valid_geometric_number m ∧ is_valid_geometric_number M ∧ (M - m = 7173) :=
sorry

end NUMINAMATH_GPT_geometric_number_difference_l439_43938


namespace NUMINAMATH_GPT_sequence_sum_l439_43936

-- Define the arithmetic sequence and conditions
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific values used in the problem
def specific_condition (a : ℕ → ℝ) : Prop :=
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450)

-- The proof goal that needs to be established
theorem sequence_sum (a : ℕ → ℝ) (h1 : arithmetic_seq a) (h2 : specific_condition a) : a 2 + a 8 = 180 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l439_43936


namespace NUMINAMATH_GPT_quadratic_other_root_l439_43919

theorem quadratic_other_root (m : ℝ) :
  (2 * 1^2 - m * 1 + 6 = 0) →
  ∃ y : ℝ, y ≠ 1 ∧ (2 * y^2 - m * y + 6 = 0) ∧ (1 * y = 3) :=
by
  intros h
  -- using sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_quadratic_other_root_l439_43919


namespace NUMINAMATH_GPT_total_weight_of_5_moles_of_cai2_l439_43963

-- Definitions based on the conditions
def weight_of_calcium : Real := 40.08
def weight_of_iodine : Real := 126.90
def iodine_atoms_in_cai2 : Nat := 2
def moles_of_calcium_iodide : Nat := 5

-- Lean 4 statement for the proof problem
theorem total_weight_of_5_moles_of_cai2 :
  (weight_of_calcium + (iodine_atoms_in_cai2 * weight_of_iodine)) * moles_of_calcium_iodide = 1469.4 := by
  sorry

end NUMINAMATH_GPT_total_weight_of_5_moles_of_cai2_l439_43963


namespace NUMINAMATH_GPT_fraction_of_work_left_l439_43939

theorem fraction_of_work_left 
  (A_days : ℕ) (B_days : ℕ) (work_days : ℕ) 
  (A_rate : ℚ := 1 / A_days) (B_rate : ℚ := 1 / B_days) (combined_rate : ℚ := 1 / A_days + 1 / B_days) 
  (work_completed : ℚ := combined_rate * work_days) (fraction_left : ℚ := 1 - work_completed)
  (hA : A_days = 15) (hB : B_days = 20) (hW : work_days = 4) 
  : fraction_left = 8 / 15 :=
sorry

end NUMINAMATH_GPT_fraction_of_work_left_l439_43939


namespace NUMINAMATH_GPT_machine_A_production_is_4_l439_43907

noncomputable def machine_production (A : ℝ) (B : ℝ) (T_A : ℝ) (T_B : ℝ) := 
  (440 / A = T_A) ∧
  (440 / B = T_B) ∧
  (T_A = T_B + 10) ∧
  (B = 1.10 * A)

theorem machine_A_production_is_4 {A B T_A T_B : ℝ}
  (h : machine_production A B T_A T_B) : 
  A = 4 :=
by
  sorry

end NUMINAMATH_GPT_machine_A_production_is_4_l439_43907


namespace NUMINAMATH_GPT_ratio_gold_to_green_horses_l439_43926

theorem ratio_gold_to_green_horses (blue_horses purple_horses green_horses gold_horses : ℕ)
    (h1 : blue_horses = 3)
    (h2 : purple_horses = 3 * blue_horses)
    (h3 : green_horses = 2 * purple_horses)
    (h4 : blue_horses + purple_horses + green_horses + gold_horses = 33) :
  gold_horses / gcd gold_horses green_horses = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_gold_to_green_horses_l439_43926


namespace NUMINAMATH_GPT_fraction_sum_of_two_reciprocals_fraction_sum_of_equal_reciprocals_fraction_difference_of_two_reciprocals_l439_43924

theorem fraction_sum_of_two_reciprocals (n : ℕ) (hn : n > 0) : 
  ∃ a b : ℕ, (a ≠ b) ∧ (3 * 5 * n * (a + b) = a * b) :=
sorry

theorem fraction_sum_of_equal_reciprocals (n : ℕ) : 
  ∃ a : ℕ, 3 * 5 * n * 2 = a * a ↔ (∃ k : ℕ, n = 2 * k) :=
sorry

theorem fraction_difference_of_two_reciprocals (n : ℕ) (hn : n > 0) : 
  ∃ a b : ℕ, (a ≠ b) ∧ 3 * 5 * n * (a - b) = a * b :=
sorry

end NUMINAMATH_GPT_fraction_sum_of_two_reciprocals_fraction_sum_of_equal_reciprocals_fraction_difference_of_two_reciprocals_l439_43924


namespace NUMINAMATH_GPT_sum_of_roots_of_equation_l439_43925

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_equation_l439_43925


namespace NUMINAMATH_GPT_mats_weaved_by_mat_weavers_l439_43954

variable (M : ℕ)

theorem mats_weaved_by_mat_weavers :
  -- 10 mat-weavers can weave 25 mats in 10 days
  (10 * 10) * M / (4 * 4) = 25 / (10 / 4)  →
  -- number of mats woven by 4 mat-weavers in 4 days
  M = 4 :=
sorry

end NUMINAMATH_GPT_mats_weaved_by_mat_weavers_l439_43954


namespace NUMINAMATH_GPT_find_initial_girls_l439_43930

variable (b g : ℕ)

theorem find_initial_girls 
  (h1 : 3 * (g - 18) = b)
  (h2 : 4 * (b - 36) = g - 18) :
  g = 31 := 
by
  sorry

end NUMINAMATH_GPT_find_initial_girls_l439_43930


namespace NUMINAMATH_GPT_arithmetic_sequence_a20_l439_43914

theorem arithmetic_sequence_a20 (a : Nat → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + a 3 + a 5 = 18)
  (h3 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a20_l439_43914


namespace NUMINAMATH_GPT_math_problem_l439_43929

theorem math_problem
  (x y : ℚ)
  (h1 : x + y = 11 / 17)
  (h2 : x - y = 1 / 143) :
  x^2 - y^2 = 11 / 2431 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l439_43929


namespace NUMINAMATH_GPT_seunghye_saw_number_l439_43923

theorem seunghye_saw_number (x : ℝ) (h : 10 * x - x = 37.35) : x = 4.15 :=
by
  sorry

end NUMINAMATH_GPT_seunghye_saw_number_l439_43923


namespace NUMINAMATH_GPT_regular_polygon_sides_l439_43964

theorem regular_polygon_sides (n : ℕ) (h₁ : n > 2) (h₂ : ∀ i, 1 ≤ i ∧ i ≤ n → True) (h₃ : (360 / n : ℝ) = 30) : n = 12 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l439_43964


namespace NUMINAMATH_GPT_total_pages_read_correct_l439_43992

-- Definition of the problem conditions
def first_week_books := 5
def first_week_book_pages := 300
def first_week_magazines := 3
def first_week_magazine_pages := 120
def first_week_newspapers := 2
def first_week_newspaper_pages := 50

def second_week_books := 2 * first_week_books
def second_week_book_pages := 350
def second_week_magazines := 4
def second_week_magazine_pages := 150
def second_week_newspapers := 1
def second_week_newspaper_pages := 60

def third_week_books := 3 * first_week_books
def third_week_book_pages := 400
def third_week_magazines := 5
def third_week_magazine_pages := 125
def third_week_newspapers := 1
def third_week_newspaper_pages := 70

-- Total pages read in each week
def first_week_total_pages : Nat :=
  (first_week_books * first_week_book_pages) +
  (first_week_magazines * first_week_magazine_pages) +
  (first_week_newspapers * first_week_newspaper_pages)

def second_week_total_pages : Nat :=
  (second_week_books * second_week_book_pages) +
  (second_week_magazines * second_week_magazine_pages) +
  (second_week_newspapers * second_week_newspaper_pages)

def third_week_total_pages : Nat :=
  (third_week_books * third_week_book_pages) +
  (third_week_magazines * third_week_magazine_pages) +
  (third_week_newspapers * third_week_newspaper_pages)

-- Grand total pages read over three weeks
def total_pages_read : Nat :=
  first_week_total_pages + second_week_total_pages + third_week_total_pages

-- Theorem statement to be proven
theorem total_pages_read_correct :
  total_pages_read = 12815 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_total_pages_read_correct_l439_43992


namespace NUMINAMATH_GPT_min_value_frac_l439_43915

theorem min_value_frac (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : a + b = 1) : 
  (1 / a) + (4 / b) ≥ 9 :=
by sorry

end NUMINAMATH_GPT_min_value_frac_l439_43915


namespace NUMINAMATH_GPT_pandas_and_bamboo_l439_43997

-- Definitions for the conditions
def number_of_pandas (x : ℕ) :=
  (∃ y : ℕ, y = 5 * x + 11 ∧ y = 2 * (3 * x - 5) - 8)

-- Theorem stating the solution
theorem pandas_and_bamboo (x y : ℕ) (h1 : y = 5 * x + 11) (h2 : y = 2 * (3 * x - 5) - 8) : x = 29 ∧ y = 156 :=
by {
  sorry
}

end NUMINAMATH_GPT_pandas_and_bamboo_l439_43997


namespace NUMINAMATH_GPT_somu_age_relation_l439_43993

-- Somu’s present age (S) is 20 years
def somu_present_age : ℕ := 20

-- Somu’s age is one-third of his father’s age (F)
def father_present_age : ℕ := 3 * somu_present_age

-- Proof statement: Y years ago, Somu's age was one-fifth of his father's age
theorem somu_age_relation : ∃ (Y : ℕ), somu_present_age - Y = (1 : ℕ) / 5 * (father_present_age - Y) ∧ Y = 10 :=
by
  have h := "" -- Placeholder for the proof steps
  sorry

end NUMINAMATH_GPT_somu_age_relation_l439_43993


namespace NUMINAMATH_GPT_distance_from_focus_l439_43912

theorem distance_from_focus (x : ℝ) (A : ℝ × ℝ) (hA_on_parabola : A.1^2 = 4 * A.2) (hA_coord : A.2 = 4) : 
  dist A (0, 1) = 5 := 
by
  sorry

end NUMINAMATH_GPT_distance_from_focus_l439_43912


namespace NUMINAMATH_GPT_breakfast_time_correct_l439_43996

noncomputable def breakfast_time_calc (x : ℚ) : ℚ :=
  (7 * 60) + (300 / 13)

noncomputable def coffee_time_calc (y : ℚ) : ℚ :=
  (7 * 60) + (420 / 11)

noncomputable def total_breakfast_time : ℚ :=
  coffee_time_calc ((420 : ℚ) / 11) - breakfast_time_calc ((300 : ℚ) / 13)

theorem breakfast_time_correct :
  total_breakfast_time = 15 + (6 / 60) :=
by
  sorry

end NUMINAMATH_GPT_breakfast_time_correct_l439_43996


namespace NUMINAMATH_GPT_grown_ups_in_milburg_l439_43976

def number_of_children : ℕ := 2987
def total_population : ℕ := 8243

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 := 
by 
  sorry

end NUMINAMATH_GPT_grown_ups_in_milburg_l439_43976


namespace NUMINAMATH_GPT_find_x_for_g_inv_eq_3_l439_43967

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 + 5

-- State the theorem
theorem find_x_for_g_inv_eq_3 : ∃ x : ℝ, g x = 113 :=
by
  exists 3
  unfold g
  norm_num

end NUMINAMATH_GPT_find_x_for_g_inv_eq_3_l439_43967


namespace NUMINAMATH_GPT_favorite_movies_hours_l439_43952

theorem favorite_movies_hours (J M N R : ℕ) (h1 : J = M + 2) (h2 : N = 3 * M) (h3 : R = (4 * N) / 5) (h4 : N = 30) : 
  J + M + N + R = 76 :=
by
  sorry

end NUMINAMATH_GPT_favorite_movies_hours_l439_43952


namespace NUMINAMATH_GPT_spring_membership_decrease_l439_43969

theorem spring_membership_decrease (init_members : ℝ) (increase_percent : ℝ) (total_change_percent : ℝ) 
  (fall_members := init_members * (1 + increase_percent / 100)) 
  (spring_members := init_members * (1 + total_change_percent / 100)) :
  increase_percent = 8 → total_change_percent = -12.52 → 
  (fall_members - spring_members) / fall_members * 100 = 19 :=
by
  intros h1 h2
  -- The complicated proof goes here.
  sorry

end NUMINAMATH_GPT_spring_membership_decrease_l439_43969


namespace NUMINAMATH_GPT_turnip_pulled_by_mice_l439_43960

theorem turnip_pulled_by_mice :
  ∀ (M B G D J C : ℕ),
    D = 2 * B →
    B = 3 * G →
    G = 4 * J →
    J = 5 * C →
    C = 6 * M →
    (D + B + G + J + C + M) ≥ (D + B + G + J + C) + M → 
    1237 * M ≤ (D + B + G + J + C + M) :=
by
  intros M B G D J C h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5]
  linarith

end NUMINAMATH_GPT_turnip_pulled_by_mice_l439_43960


namespace NUMINAMATH_GPT_joan_kittens_count_correct_l439_43979

def joan_initial_kittens : Nat := 8
def kittens_from_friends : Nat := 2
def joan_total_kittens (initial: Nat) (added: Nat) : Nat := initial + added

theorem joan_kittens_count_correct : joan_total_kittens joan_initial_kittens kittens_from_friends = 10 := 
by
  sorry

end NUMINAMATH_GPT_joan_kittens_count_correct_l439_43979


namespace NUMINAMATH_GPT_least_positive_integer_div_conditions_l439_43927

theorem least_positive_integer_div_conditions :
  ∃ n > 1, (n % 4 = 3) ∧ (n % 5 = 3) ∧ (n % 7 = 3) ∧ (n % 10 = 3) ∧ (n % 11 = 3) ∧ n = 1543 := 
by 
  sorry

end NUMINAMATH_GPT_least_positive_integer_div_conditions_l439_43927


namespace NUMINAMATH_GPT_proof_statements_imply_negation_l439_43987

-- Define propositions p, q, and r
variables (p q r : Prop)

-- Statement (1): p, q, and r are all true.
def statement_1 : Prop := p ∧ q ∧ r

-- Statement (2): p is true, q is false, and r is true.
def statement_2 : Prop := p ∧ ¬ q ∧ r

-- Statement (3): p is false, q is true, and r is false.
def statement_3 : Prop := ¬ p ∧ q ∧ ¬ r

-- Statement (4): p and r are false, q is true.
def statement_4 : Prop := ¬ p ∧ q ∧ ¬ r

-- The negation of "p and q are true, and r is false" is "¬(p ∧ q) ∨ r"
def negation : Prop := ¬(p ∧ q) ∨ r

-- Proof statement that each of the 4 statements implies the negation
theorem proof_statements_imply_negation :
  (statement_1 p q r → negation p q r) ∧
  (statement_2 p q r → negation p q r) ∧
  (statement_3 p q r → negation p q r) ∧
  (statement_4 p q r → negation p q r) :=
by
  sorry

end NUMINAMATH_GPT_proof_statements_imply_negation_l439_43987


namespace NUMINAMATH_GPT_initial_bags_count_l439_43978

theorem initial_bags_count
  (points_per_bag : ℕ)
  (non_recycled_bags : ℕ)
  (total_possible_points : ℕ)
  (points_earned : ℕ)
  (B : ℕ)
  (h1 : points_per_bag = 5)
  (h2 : non_recycled_bags = 2)
  (h3 : total_possible_points = 45)
  (h4 : points_earned = 5 * (B - non_recycled_bags))
  : B = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_bags_count_l439_43978


namespace NUMINAMATH_GPT_arithmetic_sequence_lemma_l439_43928

theorem arithmetic_sequence_lemma (a : ℕ → ℝ) (h_arith_seq : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0)
  (h_condition : a 3 + a 11 = 22) : a 7 = 11 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_lemma_l439_43928


namespace NUMINAMATH_GPT_dance_lessons_l439_43975

theorem dance_lessons (cost_per_lesson : ℕ) (free_lessons : ℕ) (amount_paid : ℕ) 
  (H1 : cost_per_lesson = 10) 
  (H2 : free_lessons = 2) 
  (H3 : amount_paid = 80) : 
  (amount_paid / cost_per_lesson + free_lessons = 10) :=
by
  sorry

end NUMINAMATH_GPT_dance_lessons_l439_43975


namespace NUMINAMATH_GPT_sum_of_squares_l439_43946

theorem sum_of_squares (a b c : ℝ) (h1 : ab + bc + ca = 4) (h2 : a + b + c = 17) : a^2 + b^2 + c^2 = 281 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l439_43946


namespace NUMINAMATH_GPT_Tim_age_l439_43994

theorem Tim_age (T t : ℕ) (h1 : T = 22) (h2 : T = 2 * t + 6) : t = 8 := by
  sorry

end NUMINAMATH_GPT_Tim_age_l439_43994


namespace NUMINAMATH_GPT_triangle_areas_l439_43948

theorem triangle_areas (r s : ℝ) (h1 : s = (1/2) * r + 6)
                       (h2 : (12 + r) * ((1/2) * r + 6) = 18) :
  r + s = -3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_areas_l439_43948


namespace NUMINAMATH_GPT_find_x_l439_43933

theorem find_x (x m n : ℤ) 
  (h₁ : 15 + x = m^2) 
  (h₂ : x - 74 = n^2) :
  x = 2010 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l439_43933


namespace NUMINAMATH_GPT_exponent_equality_l439_43937

theorem exponent_equality (M : ℕ) : 32^4 * 4^6 = 2^M → M = 32 :=
by
  sorry

end NUMINAMATH_GPT_exponent_equality_l439_43937


namespace NUMINAMATH_GPT_volume_of_wedge_l439_43911

theorem volume_of_wedge (h : 2 * Real.pi * r = 18 * Real.pi) :
  let V := (4 / 3) * Real.pi * (r ^ 3)
  let V_wedge := V / 6
  V_wedge = 162 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_volume_of_wedge_l439_43911


namespace NUMINAMATH_GPT_range_of_a_l439_43966

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then -x^2 - 1 else Real.log (x + 1)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x ≤ a * x) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l439_43966


namespace NUMINAMATH_GPT_train_speed_45_kmph_l439_43956

variable (length_train length_bridge time_passed : ℕ)

def total_distance (length_train length_bridge : ℕ) : ℕ :=
  length_train + length_bridge

def speed_m_per_s (length_train length_bridge time_passed : ℕ) : ℚ :=
  (total_distance length_train length_bridge) / time_passed

def speed_km_per_h (length_train length_bridge time_passed : ℕ) : ℚ :=
  (speed_m_per_s length_train length_bridge time_passed) * 3.6

theorem train_speed_45_kmph :
  length_train = 360 → length_bridge = 140 → time_passed = 40 → speed_km_per_h length_train length_bridge time_passed = 45 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_45_kmph_l439_43956


namespace NUMINAMATH_GPT_xiao_wang_programming_methods_l439_43944

theorem xiao_wang_programming_methods :
  ∃ (n : ℕ), n = 20 :=
by sorry

end NUMINAMATH_GPT_xiao_wang_programming_methods_l439_43944


namespace NUMINAMATH_GPT_simultaneous_equations_solution_l439_43931

theorem simultaneous_equations_solution (x y : ℚ) (h1 : 3 * x - 4 * y = 11) (h2 : 9 * x + 6 * y = 33) : 
  x = 11 / 3 ∧ y = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_simultaneous_equations_solution_l439_43931


namespace NUMINAMATH_GPT_find_a2_given_conditions_l439_43943

variable (a : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) := ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

theorem find_a2_given_conditions
  {a : ℕ → ℤ}
  (h_seq : is_arithmetic_sequence a)
  (h1 : a 3 + a 5 = 24)
  (h2 : a 7 - a 3 = 24) :
  a 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_a2_given_conditions_l439_43943


namespace NUMINAMATH_GPT_point_and_sum_of_coordinates_l439_43985

-- Definitions
def point_on_graph_of_g_over_3 (g : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = (g p.1) / 3

def point_on_graph_of_inv_g_over_3 (g : ℝ → ℝ) (q : ℝ × ℝ) : Prop :=
  q.2 = (g⁻¹ q.1) / 3

-- Main statement
theorem point_and_sum_of_coordinates {g : ℝ → ℝ} (h : point_on_graph_of_g_over_3 g (2, 3)) :
  point_on_graph_of_inv_g_over_3 g (9, 2 / 3) ∧ (9 + 2 / 3 = 29 / 3) :=
by
  sorry

end NUMINAMATH_GPT_point_and_sum_of_coordinates_l439_43985


namespace NUMINAMATH_GPT_A_equals_4_of_rounded_to_tens_9430_l439_43990

variable (A B : ℕ)

theorem A_equals_4_of_rounded_to_tens_9430
  (h1 : 9430 = 9000 + 100 * A + 10 * 3 + B)
  (h2 : B < 5)
  (h3 : 0 ≤ A ∧ A ≤ 9)
  (h4 : 0 ≤ B ∧ B ≤ 9) :
  A = 4 :=
by
  sorry

end NUMINAMATH_GPT_A_equals_4_of_rounded_to_tens_9430_l439_43990


namespace NUMINAMATH_GPT_Terry_has_20_more_stickers_than_Steven_l439_43913

theorem Terry_has_20_more_stickers_than_Steven :
  let Ryan_stickers := 30
  let Steven_stickers := 3 * Ryan_stickers
  let Total_stickers := 230
  let Ryan_Steven_Total := Ryan_stickers + Steven_stickers
  let Terry_stickers := Total_stickers - Ryan_Steven_Total
  (Terry_stickers - Steven_stickers) = 20 := 
by 
  sorry

end NUMINAMATH_GPT_Terry_has_20_more_stickers_than_Steven_l439_43913


namespace NUMINAMATH_GPT_line_slope_intercept_product_l439_43999

theorem line_slope_intercept_product :
  ∃ (m b : ℝ), (b = -1) ∧ ((1 - (m * -1 + b) = 0) ∧ (mb = m * b)) ∧ (mb = 2) :=
by sorry

end NUMINAMATH_GPT_line_slope_intercept_product_l439_43999


namespace NUMINAMATH_GPT_total_ants_correct_l439_43922

def abe_ants : ℕ := 4
def beth_ants : ℕ := abe_ants + (abe_ants / 2)
def cece_ants : ℕ := 2 * abe_ants
def duke_ants : ℕ := abe_ants / 2
def total_ants : ℕ := abe_ants + beth_ants + cece_ants + duke_ants

theorem total_ants_correct : total_ants = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_ants_correct_l439_43922


namespace NUMINAMATH_GPT_find_y_in_similar_triangles_l439_43970

-- Define the variables and conditions of the problem
def is_similar (a1 b1 a2 b2 : ℚ) : Prop :=
  a1 / b1 = a2 / b2

-- Problem statement
theorem find_y_in_similar_triangles
  (a1 b1 a2 b2 : ℚ)
  (h1 : a1 = 15)
  (h2 : b1 = 12)
  (h3 : b2 = 10)
  (similarity_condition : is_similar a1 b1 a2 b2) :
  a2 = 25 / 2 :=
by
  rw [h1, h2, h3, is_similar] at similarity_condition
  sorry

end NUMINAMATH_GPT_find_y_in_similar_triangles_l439_43970


namespace NUMINAMATH_GPT_train_speed_l439_43917

theorem train_speed (L1 L2 : ℕ) (V2 : ℕ) (t : ℝ) (V1 : ℝ) : 
  L1 = 200 → 
  L2 = 280 → 
  V2 = 30 → 
  t = 23.998 → 
  (0.001 * (L1 + L2)) / (t / 3600) = V1 + V2 → 
  V1 = 42 :=
by 
  intros
  sorry

end NUMINAMATH_GPT_train_speed_l439_43917


namespace NUMINAMATH_GPT_ratio_surfer_malibu_santa_monica_l439_43958

theorem ratio_surfer_malibu_santa_monica (M S : ℕ) (hS : S = 20) (hTotal : M + S = 60) : M / S = 2 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_surfer_malibu_santa_monica_l439_43958


namespace NUMINAMATH_GPT_typeA_cloth_typeB_cloth_typeC_cloth_l439_43983

section ClothPrices

variables (CPA CPB CPC : ℝ)

theorem typeA_cloth :
  (300 * CPA * 0.90 = 9000) → CPA = 33.33 :=
by
  intro hCPA
  sorry

theorem typeB_cloth :
  (250 * CPB * 1.05 = 7000) → CPB = 26.67 :=
by
  intro hCPB
  sorry

theorem typeC_cloth :
  (400 * (CPC + 8) = 12000) → CPC = 22 :=
by
  intro hCPC
  sorry

end ClothPrices

end NUMINAMATH_GPT_typeA_cloth_typeB_cloth_typeC_cloth_l439_43983


namespace NUMINAMATH_GPT_minimum_distance_from_mars_l439_43950

noncomputable def distance_function (a b c t : ℝ) : ℝ :=
  a * t^2 + b * t + c

theorem minimum_distance_from_mars :
  ∃ t₀ : ℝ, distance_function (11/54) (-1/18) 4 t₀ = (9:ℝ) :=
  sorry

end NUMINAMATH_GPT_minimum_distance_from_mars_l439_43950


namespace NUMINAMATH_GPT_alpha_sufficient_but_not_necessary_condition_of_beta_l439_43900
open Classical

variable (x : ℝ)
def α := x = -1
def β := x ≤ 0

theorem alpha_sufficient_but_not_necessary_condition_of_beta :
  (α x → β x) ∧ ¬(β x → α x) :=
by
  sorry

end NUMINAMATH_GPT_alpha_sufficient_but_not_necessary_condition_of_beta_l439_43900


namespace NUMINAMATH_GPT_domain_of_function_l439_43982

def quadratic_inequality (x : ℝ) : Prop := -8 * x^2 - 14 * x + 9 ≥ 0

theorem domain_of_function :
  {x : ℝ | quadratic_inequality x} = {x : ℝ | x ≤ -1} ∪ {x : ℝ | x ≥ 9 / 8} :=
by
  -- The detailed proof would go here, but we're focusing on the statement structure.
  sorry

end NUMINAMATH_GPT_domain_of_function_l439_43982


namespace NUMINAMATH_GPT_find_k_l439_43981

def geom_seq (c : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = c * (a n)

def sum_first_n_terms (S : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n, S n = 3^n + k

theorem find_k {c : ℝ} {a : ℕ → ℝ} {S : ℕ → ℝ} {k : ℝ} (hGeom : geom_seq c a) (hSum : sum_first_n_terms S k) :
  k = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l439_43981


namespace NUMINAMATH_GPT_find_x_for_equation_l439_43951

theorem find_x_for_equation :
  ∃ x : ℝ, 9 - 3 / (1 / x) + 3 = 3 ↔ x = 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_for_equation_l439_43951


namespace NUMINAMATH_GPT_number_of_nephews_l439_43949

def total_jellybeans : ℕ := 70
def jellybeans_per_child : ℕ := 14
def number_of_nieces : ℕ := 2

theorem number_of_nephews : total_jellybeans / jellybeans_per_child - number_of_nieces = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_nephews_l439_43949


namespace NUMINAMATH_GPT_compute_g_x_h_l439_43955

def g (x : ℝ) : ℝ := 6 * x^2 - 3 * x + 4

theorem compute_g_x_h (x h : ℝ) : 
  g (x + h) - g x = h * (12 * x + 6 * h - 3) := by
  sorry

end NUMINAMATH_GPT_compute_g_x_h_l439_43955


namespace NUMINAMATH_GPT_sum_of_four_primes_is_prime_l439_43901

theorem sum_of_four_primes_is_prime
    (A B : ℕ)
    (hA_prime : Prime A)
    (hB_prime : Prime B)
    (hA_minus_B_prime : Prime (A - B))
    (hA_plus_B_prime : Prime (A + B)) :
    Prime (A + B + (A - B) + A) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_four_primes_is_prime_l439_43901


namespace NUMINAMATH_GPT_heads_count_l439_43965

theorem heads_count (H T : ℕ) (h1 : H + T = 128) (h2 : H = T + 12) : H = 70 := by
  sorry

end NUMINAMATH_GPT_heads_count_l439_43965


namespace NUMINAMATH_GPT_calculate_expression_l439_43989

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_calculate_expression_l439_43989


namespace NUMINAMATH_GPT_value_sq_dist_OP_OQ_l439_43932

-- Definitions from problem conditions
def origin : ℝ × ℝ := (0, 0)
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1
def perpendicular (p q : ℝ × ℝ) : Prop := p.1 * q.1 + p.2 * q.2 = 0

-- The proof statement
theorem value_sq_dist_OP_OQ 
  (P Q : ℝ × ℝ) 
  (hP : ellipse P.1 P.2) 
  (hQ : ellipse Q.1 Q.2) 
  (h_perp : perpendicular P Q)
  : (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) = 48 / 7 := 
sorry

end NUMINAMATH_GPT_value_sq_dist_OP_OQ_l439_43932


namespace NUMINAMATH_GPT_pow_mod_79_l439_43998

theorem pow_mod_79 (a : ℕ) (h : a = 7) : a^79 % 11 = 6 := by
  sorry

end NUMINAMATH_GPT_pow_mod_79_l439_43998


namespace NUMINAMATH_GPT_square_area_increase_l439_43971

variable (s : ℝ)

theorem square_area_increase (h : s > 0) : 
  let s_new := 1.30 * s
  let A_original := s^2
  let A_new := s_new^2
  let percentage_increase := ((A_new - A_original) / A_original) * 100
  percentage_increase = 69 := by
sorry

end NUMINAMATH_GPT_square_area_increase_l439_43971


namespace NUMINAMATH_GPT_geom_seq_a11_l439_43977

variable {α : Type*} [LinearOrderedField α]

def geom_seq (a : ℕ → α) (q : α) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geom_seq_a11
  (a : ℕ → α)
  (q : α)
  (ha3 : a 3 = 3)
  (ha7 : a 7 = 6)
  (hgeom : geom_seq a q) :
  a 11 = 12 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_a11_l439_43977


namespace NUMINAMATH_GPT_radius_of_inner_tangent_circle_l439_43957

theorem radius_of_inner_tangent_circle (side_length : ℝ) (num_semicircles_per_side : ℝ) (semicircle_radius : ℝ)
  (h_side_length : side_length = 4) (h_num_semicircles_per_side : num_semicircles_per_side = 3) 
  (h_semicircle_radius : semicircle_radius = side_length / (2 * num_semicircles_per_side)) :
  ∃ (inner_circle_radius : ℝ), inner_circle_radius = 7 / 6 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_inner_tangent_circle_l439_43957


namespace NUMINAMATH_GPT_fraction_increase_l439_43909

theorem fraction_increase (m n a : ℕ) (h1 : m > n) (h2 : a > 0) : 
  (n : ℚ) / m < (n + a : ℚ) / (m + a) :=
by
  sorry

end NUMINAMATH_GPT_fraction_increase_l439_43909


namespace NUMINAMATH_GPT_other_denominations_l439_43935

theorem other_denominations :
  ∀ (total_checks : ℕ) (total_value : ℝ) (fifty_denomination_checks : ℕ) (remaining_avg : ℝ),
    total_checks = 30 →
    total_value = 1800 →
    fifty_denomination_checks = 15 →
    remaining_avg = 70 →
    ∃ (other_denomination : ℝ), other_denomination = 70 :=
by
  intros total_checks total_value fifty_denomination_checks remaining_avg
  intros h1 h2 h3 h4
  let other_denomination := 70
  use other_denomination
  sorry

end NUMINAMATH_GPT_other_denominations_l439_43935


namespace NUMINAMATH_GPT_sum_of_x_coordinates_mod13_intersection_l439_43920

theorem sum_of_x_coordinates_mod13_intersection :
  (∀ x y : ℕ, y ≡ 3 * x + 5 [MOD 13] → y ≡ 7 * x + 4 [MOD 13]) → (x ≡ 10 [MOD 13]) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_coordinates_mod13_intersection_l439_43920


namespace NUMINAMATH_GPT_day_crew_fraction_correct_l439_43902

variable (D Wd : ℕ) -- D = number of boxes loaded by each worker on the day crew, Wd = number of workers on the day crew

-- fraction of all boxes loaded by day crew
def fraction_loaded_by_day_crew (D Wd : ℕ) : ℚ :=
  (D * Wd) / (D * Wd + (3 / 4 * D) * (2 / 3 * Wd))

theorem day_crew_fraction_correct (h1 : D > 0) (h2 : Wd > 0) :
  fraction_loaded_by_day_crew D Wd = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_day_crew_fraction_correct_l439_43902


namespace NUMINAMATH_GPT_irene_age_is_46_l439_43953

def eddie_age : ℕ := 92

def becky_age (e_age : ℕ) : ℕ := e_age / 4

def irene_age (b_age : ℕ) : ℕ := 2 * b_age

theorem irene_age_is_46 : irene_age (becky_age eddie_age) = 46 := 
  by
    sorry

end NUMINAMATH_GPT_irene_age_is_46_l439_43953


namespace NUMINAMATH_GPT_cost_of_plastering_l439_43916

def length := 25
def width := 12
def depth := 6
def cost_per_sq_meter_paise := 75

def surface_area_of_two_walls_one := 2 * (length * depth)
def surface_area_of_two_walls_two := 2 * (width * depth)
def surface_area_of_bottom := length * width

def total_surface_area := surface_area_of_two_walls_one + surface_area_of_two_walls_two + surface_area_of_bottom

def cost_per_sq_meter_rupees := cost_per_sq_meter_paise / 100
def total_cost := total_surface_area * cost_per_sq_meter_rupees

theorem cost_of_plastering : total_cost = 558 := by
  sorry

end NUMINAMATH_GPT_cost_of_plastering_l439_43916


namespace NUMINAMATH_GPT_product_seqFrac_l439_43988

def seqFrac (n : ℕ) : ℚ := (n : ℚ) / (n + 5 : ℚ)

theorem product_seqFrac :
  ((List.range 53).map seqFrac).prod = 1 / 27720 := by
  sorry

end NUMINAMATH_GPT_product_seqFrac_l439_43988


namespace NUMINAMATH_GPT_zoo_guides_children_total_l439_43947

theorem zoo_guides_children_total :
  let num_guides := 22
  let num_english_guides := 10
  let num_french_guides := 6
  let num_spanish_guides := num_guides - num_english_guides - num_french_guides
  let children_english_friday := 10 * 20
  let children_french_friday := 6 * 25
  let children_spanish_friday := num_spanish_guides * 30
  let children_english_saturday := 10 * 22
  let children_french_saturday := 6 * 24
  let children_spanish_saturday := num_spanish_guides * 32
  let children_english_sunday := 10 * 24
  let children_french_sunday := 6 * 23
  let children_spanish_sunday := num_spanish_guides * 35
  let total_children := children_english_friday + children_french_friday + children_spanish_friday + children_english_saturday + children_french_saturday + children_spanish_saturday + children_english_sunday + children_french_sunday + children_spanish_sunday
  total_children = 1674 :=
by
  let num_guides := 22
  let num_english_guides := 10
  let num_french_guides := 6
  let num_spanish_guides := num_guides - num_english_guides - num_french_guides
  let children_english_friday := 10 * 20
  let children_french_friday := 6 * 25
  let children_spanish_friday := num_spanish_guides * 30
  let children_english_saturday := 10 * 22
  let children_french_saturday := 6 * 24
  let children_spanish_saturday := num_spanish_guides * 32
  let children_english_sunday := 10 * 24
  let children_french_sunday := 6 * 23
  let children_spanish_sunday := num_spanish_guides * 35
  let total_children := children_english_friday + children_french_friday + children_spanish_friday + children_english_saturday + children_french_saturday + children_spanish_saturday + children_english_sunday + children_french_sunday + children_spanish_sunday
  sorry

end NUMINAMATH_GPT_zoo_guides_children_total_l439_43947
