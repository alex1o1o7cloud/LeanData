import Mathlib

namespace joan_jogged_3563_miles_l143_14324

noncomputable def steps_per_mile : ℕ := 1200

noncomputable def flips_per_year : ℕ := 28

noncomputable def steps_per_full_flip : ℕ := 150000

noncomputable def final_day_steps : ℕ := 75000

noncomputable def total_steps_in_year := flips_per_year * steps_per_full_flip + final_day_steps

noncomputable def miles_jogged := total_steps_in_year / steps_per_mile

theorem joan_jogged_3563_miles :
  miles_jogged = 3563 :=
by
  sorry

end joan_jogged_3563_miles_l143_14324


namespace sequence_a_n_l143_14398

theorem sequence_a_n (a : ℕ → ℚ)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n + 1) * (a n + 1) = a n) :
  a 6 = 1 / 6 :=
  sorry

end sequence_a_n_l143_14398


namespace number_of_other_workers_l143_14325

theorem number_of_other_workers (N : ℕ) (h1 : N ≥ 2) (h2 : 1 / ((N * (N - 1)) / 2) = 1 / 6) : N - 2 = 2 :=
by
  sorry

end number_of_other_workers_l143_14325


namespace train_speed_is_54_kmh_l143_14335

noncomputable def train_length_m : ℝ := 285
noncomputable def train_length_km : ℝ := train_length_m / 1000
noncomputable def time_seconds : ℝ := 19
noncomputable def time_hours : ℝ := time_seconds / 3600
noncomputable def speed : ℝ := train_length_km / time_hours

theorem train_speed_is_54_kmh :
  speed = 54 := by
sorry

end train_speed_is_54_kmh_l143_14335


namespace remainder_2n_div_14_l143_14316

theorem remainder_2n_div_14 (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 :=
sorry

end remainder_2n_div_14_l143_14316


namespace number_of_students_l143_14340

variables (m d r : ℕ) (k : ℕ)

theorem number_of_students :
  (30 < m + d ∧ m + d < 40) → (r = 3 * m) → (r = 5 * d) → m + d = 32 :=
by 
  -- The proof body is not necessary here according to instructions.
  sorry

end number_of_students_l143_14340


namespace tan_5105_eq_tan_85_l143_14390

noncomputable def tan_deg (d : ℝ) := Real.tan (d * Real.pi / 180)

theorem tan_5105_eq_tan_85 :
  tan_deg 5105 = tan_deg 85 := by
  have eq_265 : tan_deg 5105 = tan_deg 265 := by sorry
  have eq_neg : tan_deg 265 = tan_deg 85 := by sorry
  exact Eq.trans eq_265 eq_neg

end tan_5105_eq_tan_85_l143_14390


namespace average_tickets_per_day_l143_14321

def total_revenue : ℕ := 960
def price_per_ticket : ℕ := 4
def number_of_days : ℕ := 3

theorem average_tickets_per_day :
  (total_revenue / price_per_ticket) / number_of_days = 80 := 
sorry

end average_tickets_per_day_l143_14321


namespace line_through_point_equal_intercepts_l143_14384

theorem line_through_point_equal_intercepts (P : ℝ × ℝ) (hP : P = (1, 1)) :
  (∀ x y : ℝ, (x - y = 0 ∨ x + y - 2 = 0) → ∃ k : ℝ, k = 1 ∧ k = 2) :=
by
  sorry

end line_through_point_equal_intercepts_l143_14384


namespace correlation_is_1_3_4_l143_14334

def relationship1 := "The relationship between a person's age and their wealth"
def relationship2 := "The relationship between a point on a curve and its coordinates"
def relationship3 := "The relationship between apple production and climate"
def relationship4 := "The relationship between the diameter of the cross-section and the height of the same type of tree in a forest"

def isCorrelation (rel: String) : Bool :=
  if rel == relationship1 ∨ rel == relationship3 ∨ rel == relationship4 then true else false

theorem correlation_is_1_3_4 :
  {relationship1, relationship3, relationship4} = {r | isCorrelation r = true} := 
by
  sorry

end correlation_is_1_3_4_l143_14334


namespace find_original_comic_books_l143_14367

def comic_books (X : ℕ) : Prop :=
  X / 2 + 6 = 13

theorem find_original_comic_books (X : ℕ) (h : comic_books X) : X = 14 :=
by
  sorry

end find_original_comic_books_l143_14367


namespace martha_cakes_l143_14303

theorem martha_cakes :
  ∀ (n : ℕ), (∀ (c : ℕ), c = 3 → (∀ (k : ℕ), k = 6 → n = c * k)) → n = 18 :=
by
  intros n h
  specialize h 3 rfl 6 rfl
  exact h

end martha_cakes_l143_14303


namespace length_of_each_stone_l143_14302

-- Define the dimensions of the hall in decimeters
def hall_length_dm : ℕ := 36 * 10
def hall_breadth_dm : ℕ := 15 * 10

-- Define the width of each stone in decimeters
def stone_width_dm : ℕ := 5

-- Define the number of stones
def number_of_stones : ℕ := 1350

-- Define the total area of the hall
def hall_area : ℕ := hall_length_dm * hall_breadth_dm

-- Define the area of one stone
def stone_area : ℕ := hall_area / number_of_stones

-- Define the length of each stone and state the theorem
theorem length_of_each_stone : (stone_area / stone_width_dm) = 8 :=
by
  sorry

end length_of_each_stone_l143_14302


namespace ratio_first_part_l143_14361

theorem ratio_first_part (x : ℕ) (h1 : x / 3 = 2) : x = 6 :=
by
  sorry

end ratio_first_part_l143_14361


namespace angle_halving_quadrant_l143_14389

theorem angle_halving_quadrant (k : ℤ) (α : ℝ) 
  (h : k * 360 + 180 < α ∧ α < k * 360 + 270) : 
  k * 180 + 90 < α / 2 ∧ α / 2 < k * 180 + 135 :=
sorry

end angle_halving_quadrant_l143_14389


namespace cylinder_twice_volume_l143_14306

theorem cylinder_twice_volume :
  let r := 8
  let h1 := 10
  let h2 := 20
  let V := (pi * r^2 * h1)
  let V_desired := 2 * V
  V_desired = pi * r^2 * h2 :=
by
  let r := 8
  let h1 := 10
  let h2 := 20
  let V := (pi * r^2 * h1)
  let V_desired := 2 * V
  show V_desired = pi * r^2 * h2
  sorry

end cylinder_twice_volume_l143_14306


namespace correct_proposition_is_B_l143_14317

variables {m n : Type} {α β : Type}

-- Define parallel and perpendicular relationships
def parallel (l₁ l₂ : Type) : Prop := sorry
def perpendicular (l₁ l₂ : Type) : Prop := sorry

def lies_in (l : Type) (p : Type) : Prop := sorry

-- The problem statement
theorem correct_proposition_is_B
  (H1 : perpendicular m α)
  (H2 : perpendicular n β)
  (H3 : perpendicular α β) :
  perpendicular m n :=
sorry

end correct_proposition_is_B_l143_14317


namespace triangle_inequality_cubed_l143_14339

theorem triangle_inequality_cubed
  (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  (a^3 / c^3) + (b^3 / c^3) + (3 * a * b / c^2) > 1 := 
sorry

end triangle_inequality_cubed_l143_14339


namespace abs_diff_probs_l143_14382

def numRedMarbles := 1000
def numBlackMarbles := 1002
def totalMarbles := numRedMarbles + numBlackMarbles

def probSame : ℚ := 
  ((numRedMarbles * (numRedMarbles - 1)) / 2 + (numBlackMarbles * (numBlackMarbles - 1)) / 2) / (totalMarbles * (totalMarbles - 1) / 2)

def probDiff : ℚ :=
  (numRedMarbles * numBlackMarbles) / (totalMarbles * (totalMarbles - 1) / 2)

theorem abs_diff_probs : |probSame - probDiff| = 999 / 2003001 := 
by {
  sorry
}

end abs_diff_probs_l143_14382


namespace solution_in_quadrant_IV_l143_14373

theorem solution_in_quadrant_IV (k : ℝ) :
  (∃ x y : ℝ, x + 2 * y = 4 ∧ k * x - y = 1 ∧ x > 0 ∧ y < 0) ↔ (-1 / 2 < k ∧ k < 2) :=
by
  sorry

end solution_in_quadrant_IV_l143_14373


namespace solve_ordered_pair_l143_14360

theorem solve_ordered_pair (x y : ℝ) (h1 : x + y = (5 - x) + (5 - y)) (h2 : x - y = (x - 1) + (y - 1)) : (x, y) = (4, 1) :=
by
  sorry

end solve_ordered_pair_l143_14360


namespace range_of_a_l143_14396

variable {f : ℝ → ℝ} {a : ℝ}
open Real

-- Conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x
def f_positive_at_2 (f : ℝ → ℝ) : Prop := f 2 > 1
def f_value_at_2014 (f : ℝ → ℝ) (a : ℝ) : Prop := f 2014 = (a + 3) / (a - 3)

-- Proof Problem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : odd_function f)
  (h2 : periodic_function f 7)
  (h3 : f_positive_at_2 f)
  (h4 : f_value_at_2014 f a) :
  0 < a ∧ a < 3 :=
sorry

end range_of_a_l143_14396


namespace range_of_a_l143_14366

open Classical

noncomputable def parabola_line_common_point_range (a : ℝ) : Prop :=
  ∃ (k : ℝ), ∃ (x : ℝ), ∃ (y : ℝ), 
  (y = a * x ^ 2) ∧ ((y + 2 = k * (x - 1)) ∨ (y + 2 = - (1 / k) * (x - 1)))

theorem range_of_a (a : ℝ) : 
  (∃ k : ℝ, ∃ x : ℝ, ∃ y : ℝ, 
    y = a * x ^ 2 ∧ (y + 2 = k * (x - 1) ∨ y + 2 = - (1 / k) * (x - 1))) ↔ 
  0 < a ∧ a <= 1 / 8 :=
sorry

end range_of_a_l143_14366


namespace ratio_c_d_l143_14341

theorem ratio_c_d (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) : 
  c / d = -4 / 5 :=
by
  sorry

end ratio_c_d_l143_14341


namespace inequality_350_l143_14330

theorem inequality_350 (a b c d : ℝ) : 
  (a - b) * (b - c) * (c - d) * (d - a) + (a - c)^2 * (b - d)^2 ≥ 0 :=
by
  sorry

end inequality_350_l143_14330


namespace smallest_value_of_a_for_polynomial_l143_14329

theorem smallest_value_of_a_for_polynomial (r1 r2 r3 : ℕ) (h_prod : r1 * r2 * r3 = 30030) :
  (r1 + r2 + r3 = 54) ∧ (r1 * r2 * r3 = 30030) → 
  (∀ a, a = r1 + r2 + r3 → a ≥ 54) :=
by
  sorry

end smallest_value_of_a_for_polynomial_l143_14329


namespace problem_statement_l143_14395

variable (F : ℕ → Prop)

theorem problem_statement (h1 : ∀ k : ℕ, F k → F (k + 1)) (h2 : ¬F 7) : ¬F 6 ∧ ¬F 5 := by
  sorry

end problem_statement_l143_14395


namespace find_a99_l143_14305

def seq (a : ℕ → ℕ) :=
  a 1 = 2 ∧ ∀ n ≥ 2, a n - a (n-1) = n + 1

theorem find_a99 (a : ℕ → ℕ) (h : seq a) : a 99 = 5049 :=
by
  have : seq a := h
  sorry

end find_a99_l143_14305


namespace multiply_3_6_and_0_3_l143_14348

theorem multiply_3_6_and_0_3 : 3.6 * 0.3 = 1.08 :=
by
  sorry

end multiply_3_6_and_0_3_l143_14348


namespace train_crossing_time_l143_14375

theorem train_crossing_time (length_of_train : ℝ) (speed_kmh : ℝ) :
  length_of_train = 180 →
  speed_kmh = 72 →
  (180 / (72 * (1000 / 3600))) = 9 :=
by 
  intros h1 h2
  sorry

end train_crossing_time_l143_14375


namespace perpendicular_lines_with_foot_l143_14355

theorem perpendicular_lines_with_foot (n : ℝ) : 
  (∀ x y, 10 * x + 4 * y - 2 = 0 ↔ 2 * x - 5 * y + n = 0) ∧
  (2 * 1 - 5 * (-2) + n = 0) → n = -12 := 
by sorry

end perpendicular_lines_with_foot_l143_14355


namespace geom_seq_product_l143_14333

noncomputable def geom_seq (a : ℕ → ℝ) := 
∀ n m: ℕ, ∃ r : ℝ, a (n + m) = a n * r ^ m

theorem geom_seq_product (a : ℕ → ℝ) 
  (h_seq : geom_seq a) 
  (h_pos : ∀ n, 0 < a n) 
  (h_log_sum : Real.log (a 3) + Real.log (a 6) + Real.log (a 9) = 3) : 
  a 1 * a 11 = 100 := 
sorry

end geom_seq_product_l143_14333


namespace num_cars_can_be_parked_l143_14353

theorem num_cars_can_be_parked (length width : ℝ) (useable_percentage : ℝ) (area_per_car : ℝ) 
  (h_length : length = 400) (h_width : width = 500) (h_useable_percentage : useable_percentage = 0.80) 
  (h_area_per_car : area_per_car = 10) : 
  length * width * useable_percentage / area_per_car = 16000 := 
by 
  sorry

end num_cars_can_be_parked_l143_14353


namespace find_angle_C_l143_14301

variable {A B C a b c : ℝ}
variable (hAcute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
variable (hTriangle : A + B + C = π)
variable (hSides : a > 0 ∧ b > 0 ∧ c > 0)
variable (hCondition : Real.sqrt 3 * a = 2 * c * Real.sin A)

theorem find_angle_C (hA_pos : A ≠ 0) : C = π / 3 :=
  sorry

end find_angle_C_l143_14301


namespace digit_B_divisibility_l143_14346

theorem digit_B_divisibility (B : ℕ) (h : 4 * 1000 + B * 100 + B * 10 + 6 % 11 = 0) : B = 5 :=
sorry

end digit_B_divisibility_l143_14346


namespace customer_total_payment_l143_14311

def Riqing_Beef_Noodles_quantity : ℕ := 24
def Riqing_Beef_Noodles_price_per_bag : ℝ := 1.80
def Riqing_Beef_Noodles_discount : ℝ := 0.8

def Kang_Shifu_Ice_Red_Tea_quantity : ℕ := 6
def Kang_Shifu_Ice_Red_Tea_price_per_box : ℝ := 1.70
def Kang_Shifu_Ice_Red_Tea_discount : ℝ := 0.8

def Shanlin_Purple_Cabbage_Soup_quantity : ℕ := 5
def Shanlin_Purple_Cabbage_Soup_price_per_bag : ℝ := 3.40

def Shuanghui_Ham_Sausage_quantity : ℕ := 3
def Shuanghui_Ham_Sausage_price_per_bag : ℝ := 11.20
def Shuanghui_Ham_Sausage_discount : ℝ := 0.9

def total_price : ℝ :=
  (Riqing_Beef_Noodles_quantity * Riqing_Beef_Noodles_price_per_bag * Riqing_Beef_Noodles_discount) +
  (Kang_Shifu_Ice_Red_Tea_quantity * Kang_Shifu_Ice_Red_Tea_price_per_box * Kang_Shifu_Ice_Red_Tea_discount) +
  (Shanlin_Purple_Cabbage_Soup_quantity * Shanlin_Purple_Cabbage_Soup_price_per_bag) +
  (Shuanghui_Ham_Sausage_quantity * Shuanghui_Ham_Sausage_price_per_bag * Shuanghui_Ham_Sausage_discount)

theorem customer_total_payment :
  total_price = 89.96 :=
by
  unfold total_price
  sorry

end customer_total_payment_l143_14311


namespace intersection_points_count_l143_14386

variables {R : Type*} [LinearOrderedField R]

def line1 (x y : R) : Prop := 3 * y - 2 * x = 1
def line2 (x y : R) : Prop := x + 2 * y = 2
def line3 (x y : R) : Prop := 4 * x - 6 * y = 5

theorem intersection_points_count : 
  ∃ p1 p2 : R × R, 
   (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧ 
   (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧ 
   p1 ≠ p2 ∧ 
   (∀ p : R × R, (line1 p.1 p.2 ∧ line3 p.1 p.2) → False) := 
sorry

end intersection_points_count_l143_14386


namespace cost_of_one_pencil_and_one_pen_l143_14376

variables (x y : ℝ)

def eq1 := 4 * x + 3 * y = 3.70
def eq2 := 3 * x + 4 * y = 4.20

theorem cost_of_one_pencil_and_one_pen (h₁ : eq1 x y) (h₂ : eq2 x y) :
  x + y = 1.1286 :=
sorry

end cost_of_one_pencil_and_one_pen_l143_14376


namespace sequence_satisfies_conditions_l143_14362

theorem sequence_satisfies_conditions : 
  let seq1 := [4, 1, 3, 1, 2, 4, 3, 2]
  let seq2 := [2, 3, 4, 2, 1, 3, 1, 4]
  (seq1[0] = 4 ∧ seq1[1] = 1 ∧ seq1[2] = 3 ∧ seq1[3] = 1 ∧ seq1[4] = 2 ∧ seq1[5] = 4 ∧ seq1[6] = 3 ∧ seq1[7] = 2)
  ∨ (seq2[0] = 2 ∧ seq2[1] = 3 ∧ seq2[2] = 4 ∧ seq2[3] = 2 ∧ seq2[4] = 1 ∧ seq2[5] = 3 ∧ seq2[6] = 1 ∧ seq2[7] = 4)
  ∧ (seq1[1] = 1 ∧ seq1[3] - seq1[1] = 2 ∧ seq1[4] - seq1[2] = 3 ∧ seq1[5] - seq1[2] = 4) := 
  sorry

end sequence_satisfies_conditions_l143_14362


namespace intersection_range_l143_14331

noncomputable def function1 (x : ℝ) : ℝ := abs (x^2 - 1) / (x - 1)
noncomputable def function2 (k x : ℝ) : ℝ := k * x - 2

theorem intersection_range (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ function1 x₁ = function2 k x₁ ∧ function1 x₂ = function2 k x₂) ↔ 
  (0 < k ∧ k < 1) ∨ (1 < k ∧ k < 4) := 
sorry

end intersection_range_l143_14331


namespace decimal_to_fraction_equiv_l143_14322

theorem decimal_to_fraction_equiv : (0.38 : ℝ) = 19 / 50 :=
by
  sorry

end decimal_to_fraction_equiv_l143_14322


namespace vector_b_value_l143_14359

theorem vector_b_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (1, -2)
  2 • a + b = (3, 2) → b = (1, -2) :=
by
  intros
  sorry

end vector_b_value_l143_14359


namespace total_shirts_made_l143_14369

def shirtsPerMinute := 6
def minutesWorkedYesterday := 12
def shirtsMadeToday := 14

theorem total_shirts_made : shirtsPerMinute * minutesWorkedYesterday + shirtsMadeToday = 86 := by
  sorry

end total_shirts_made_l143_14369


namespace alcohol_percentage_in_new_solution_l143_14399

theorem alcohol_percentage_in_new_solution :
  let original_volume := 40 -- liters
  let original_percentage_alcohol := 0.05
  let added_alcohol := 5.5 -- liters
  let added_water := 4.5 -- liters
  let original_alcohol := original_percentage_alcohol * original_volume
  let new_alcohol := original_alcohol + added_alcohol
  let new_volume := original_volume + added_alcohol + added_water
  (new_alcohol / new_volume) * 100 = 15 := by
  sorry

end alcohol_percentage_in_new_solution_l143_14399


namespace pyramid_dihedral_angle_l143_14315

theorem pyramid_dihedral_angle 
  (k : ℝ) 
  (h_k_pos : 0 < k) :
  ∃ α : ℝ, α = 2 * Real.arccos (1 / Real.sqrt (Real.sqrt (4 * k))) :=
sorry

end pyramid_dihedral_angle_l143_14315


namespace range_of_a3_l143_14385

theorem range_of_a3 (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) + a n = 4 * n + 3)
  (h2 : ∀ n : ℕ, n > 0 → a n + 2 * n^2 ≥ 0) 
  : 2 ≤ a 3 ∧ a 3 ≤ 19 := 
sorry

end range_of_a3_l143_14385


namespace total_cows_in_ranch_l143_14338

theorem total_cows_in_ranch :
  ∀ (WTP_cows : ℕ) (HGHF_cows : ℕ), WTP_cows = 17 → HGHF_cows = 3 * WTP_cows + 2 → (HGHF_cows + WTP_cows) = 70 :=
by 
  intros WTP_cows HGHF_cows WTP_cows_def HGHF_cows_def
  rw [WTP_cows_def, HGHF_cows_def]
  sorry

end total_cows_in_ranch_l143_14338


namespace plane_divided_into_four_regions_l143_14372

-- Definition of the conditions
def line1 (x y : ℝ) : Prop := y = 3 * x
def line2 (x y : ℝ) : Prop := y = (1 / 3) * x

-- Proof statement
theorem plane_divided_into_four_regions :
  (∃ f g : ℝ → ℝ, ∀ x, line1 x (f x) ∧ line2 x (g x)) →
  ∃ n : ℕ, n = 4 :=
by sorry

end plane_divided_into_four_regions_l143_14372


namespace cube_volume_l143_14312

theorem cube_volume (A : ℝ) (V : ℝ) (h : A = 64) : V = 512 :=
by
  sorry

end cube_volume_l143_14312


namespace larry_gave_52_apples_l143_14320

-- Define the initial and final count of Joyce's apples
def initial_apples : ℝ := 75.0
def final_apples : ℝ := 127.0

-- Define the number of apples Larry gave Joyce
def apples_given : ℝ := final_apples - initial_apples

-- The theorem stating that Larry gave Joyce 52 apples
theorem larry_gave_52_apples : apples_given = 52 := by
  sorry

end larry_gave_52_apples_l143_14320


namespace inequality_not_always_hold_l143_14363

theorem inequality_not_always_hold (a b : ℕ) 
  (ha : a > 0) (hb : b > 0) : ¬(∀ a b, a^3 + b^3 ≥ 2 * a * b^2) :=
sorry

end inequality_not_always_hold_l143_14363


namespace Mark_speeding_ticket_owed_amount_l143_14318

theorem Mark_speeding_ticket_owed_amount :
  let base_fine := 50
  let additional_penalty_per_mph := 2
  let mph_over_limit := 45
  let school_zone_multiplier := 2
  let court_costs := 300
  let lawyer_fee_per_hour := 80
  let lawyer_hours := 3
  let additional_penalty := additional_penalty_per_mph * mph_over_limit
  let pre_school_zone_fine := base_fine + additional_penalty
  let doubled_fine := pre_school_zone_fine * school_zone_multiplier
  let total_fine_with_court_costs := doubled_fine + court_costs
  let lawyer_total_fee := lawyer_fee_per_hour * lawyer_hours
  let total_owed := total_fine_with_court_costs + lawyer_total_fee
  total_owed = 820 :=
by
  sorry

end Mark_speeding_ticket_owed_amount_l143_14318


namespace problem_statement_l143_14370

theorem problem_statement (x y z t : ℝ) (h : (x + y) / (y + z) = (z + t) / (t + x)) : x * (z + t + y) = z * (x + y + t) :=
sorry

end problem_statement_l143_14370


namespace weeks_project_lasts_l143_14344

-- Definition of the conditions
def meal_cost : ℤ := 4
def people : ℤ := 4
def days_per_week : ℤ := 5
def total_spent : ℤ := 1280
def weekly_cost : ℤ := meal_cost * people * days_per_week

-- Problem statement: prove that the number of weeks the project will last equals 16 weeks.
theorem weeks_project_lasts : total_spent / weekly_cost = 16 := by 
  sorry

end weeks_project_lasts_l143_14344


namespace truck_capacity_l143_14327

theorem truck_capacity (x y : ℝ)
  (h1 : 3 * x + 4 * y = 22)
  (h2 : 5 * x + 2 * y = 25) :
  4 * x + 3 * y = 23.5 :=
sorry

end truck_capacity_l143_14327


namespace sum_of_interior_diagonals_of_box_l143_14371

theorem sum_of_interior_diagonals_of_box (a b c : ℝ) 
  (h_edges : 4 * (a + b + c) = 60)
  (h_surface_area : 2 * (a * b + b * c + c * a) = 150) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 20 * Real.sqrt 3 := 
by
  sorry

end sum_of_interior_diagonals_of_box_l143_14371


namespace hyperbola_eccentricity_l143_14358

-- Define the conditions given in the problem
def asymptote_equation_related (a b : ℝ) : Prop := a / b = 3 / 4
def hyperbola_eccentricity_relation (a c : ℝ) : Prop := c^2 / a^2 = 25 / 9

-- Define the proof problem
theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : asymptote_equation_related a b)
  (h2 : hyperbola_eccentricity_relation a c)
  (he : e = c / a) :
  e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l143_14358


namespace reflection_xy_plane_reflection_across_point_l143_14394

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def reflect_across_xy_plane (p : Point3D) : Point3D :=
  {x := p.x, y := p.y, z := -p.z}

def reflect_across_point (a p : Point3D) : Point3D :=
  {x := 2 * a.x - p.x, y := 2 * a.y - p.y, z := 2 * a.z - p.z}

theorem reflection_xy_plane :
  reflect_across_xy_plane {x := -2, y := 1, z := 4} = {x := -2, y := 1, z := -4} :=
by sorry

theorem reflection_across_point :
  reflect_across_point {x := 1, y := 0, z := 2} {x := -2, y := 1, z := 4} = {x := -5, y := -1, z := 0} :=
by sorry

end reflection_xy_plane_reflection_across_point_l143_14394


namespace number_of_rows_containing_53_l143_14354

theorem number_of_rows_containing_53 (h_prime_53 : Nat.Prime 53) : 
  ∃! n, (n = 53 ∧ ∃ k, k ≥ 0 ∧ k ≤ n ∧ Nat.choose n k = 53) :=
by 
  sorry

end number_of_rows_containing_53_l143_14354


namespace boys_from_other_communities_l143_14388

theorem boys_from_other_communities :
  ∀ (total_boys : ℕ) (percentage_muslims percentage_hindus percentage_sikhs : ℕ),
  total_boys = 400 →
  percentage_muslims = 44 →
  percentage_hindus = 28 →
  percentage_sikhs = 10 →
  (total_boys * (100 - (percentage_muslims + percentage_hindus + percentage_sikhs)) / 100) = 72 := 
by
  intros total_boys percentage_muslims percentage_hindus percentage_sikhs h1 h2 h3 h4
  sorry

end boys_from_other_communities_l143_14388


namespace product_of_solutions_eq_neg_35_l143_14378

theorem product_of_solutions_eq_neg_35 :
  ∀ (x : ℝ), -35 = -x^2 - 2 * x → ∃ (p : ℝ), p = -35 :=
by
  intro x h
  sorry

end product_of_solutions_eq_neg_35_l143_14378


namespace opposite_negative_nine_l143_14356

theorem opposite_negative_nine : 
  (∃ (y : ℤ), -9 + y = 0 ∧ y = 9) :=
by sorry

end opposite_negative_nine_l143_14356


namespace Jake_watched_hours_on_Friday_l143_14349

theorem Jake_watched_hours_on_Friday :
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  total_show_hours - total_hours_before_Friday = 19 :=
by
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  sorry

end Jake_watched_hours_on_Friday_l143_14349


namespace radian_measure_15_degrees_l143_14393

theorem radian_measure_15_degrees : (15 * (Real.pi / 180)) = (Real.pi / 12) :=
by
  sorry

end radian_measure_15_degrees_l143_14393


namespace unique_intersection_point_l143_14323

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (Real.log 3 / Real.log x)
noncomputable def h (x : ℝ) : ℝ := 3 - (1 / Real.sqrt (Real.log 3 / Real.log x))

theorem unique_intersection_point : (∃! (x : ℝ), (x > 0) ∧ (f x = g x ∨ f x = h x ∨ g x = h x)) :=
sorry

end unique_intersection_point_l143_14323


namespace solve_equation_l143_14308

theorem solve_equation (x : ℝ) : 
  x^2 + 2 * x + 4 * Real.sqrt (x^2 + 2 * x) - 5 = 0 →
  (x = Real.sqrt 2 - 1) ∨ (x = - (Real.sqrt 2 + 1)) :=
by 
  sorry

end solve_equation_l143_14308


namespace find_b_l143_14309

theorem find_b (b : ℚ) : (-4 : ℚ) * (45 / 4) = -45 → (-4 + 45 / 4) = -b → b = -29 / 4 := by
  intros h1 h2
  sorry

end find_b_l143_14309


namespace total_trees_planted_total_trees_when_a_100_l143_14343

-- Define the number of trees planted by each team based on 'a'
def trees_first_team (a : ℕ) : ℕ := a
def trees_second_team (a : ℕ) : ℕ := 2 * a + 8
def trees_third_team (a : ℕ) : ℕ := (2 * a + 8) / 2 - 6

-- Define the total number of trees
def total_trees (a : ℕ) : ℕ := 
  trees_first_team a + trees_second_team a + trees_third_team a

-- The main theorem
theorem total_trees_planted (a : ℕ) : total_trees a = 4 * a + 6 :=
by
  sorry

-- The specific calculation when a = 100
theorem total_trees_when_a_100 : total_trees 100 = 406 :=
by
  sorry

end total_trees_planted_total_trees_when_a_100_l143_14343


namespace calc_x_equals_condition_l143_14350

theorem calc_x_equals_condition (m n p q x : ℝ) :
  x^2 + (2 * m * p + 2 * n * q) ^ 2 + (2 * m * q - 2 * n * p) ^ 2 = (m ^ 2 + n ^ 2 + p ^ 2 + q ^ 2) ^ 2 →
  x = m ^ 2 + n ^ 2 - p ^ 2 - q ^ 2 ∨ x = - m ^ 2 - n ^ 2 + p ^ 2 + q ^ 2 :=
by
  sorry

end calc_x_equals_condition_l143_14350


namespace range_of_m_l143_14379

theorem range_of_m (m n : ℝ) (h1 : n = 2 / m) (h2 : n ≥ -1) :
  m ≤ -2 ∨ m > 0 := 
sorry

end range_of_m_l143_14379


namespace sum_of_coordinates_l143_14313

theorem sum_of_coordinates (f : ℝ → ℝ) (h : f 2 = 4) :
  let x := 4
  let y := (f⁻¹ x) / 4
  x + y = 9 / 2 :=
by
  sorry

end sum_of_coordinates_l143_14313


namespace tan_of_11pi_over_4_l143_14332

theorem tan_of_11pi_over_4 :
  Real.tan (11 * Real.pi / 4) = -1 := by
  sorry

end tan_of_11pi_over_4_l143_14332


namespace find_ratio_l143_14337

variable (x y : ℝ)

-- Hypotheses: x and y are distinct real numbers and the given equation holds
variable (h₁ : x ≠ y)
variable (h₂ : x / y + (x + 15 * y) / (y + 15 * x) = 3)

-- We aim to prove that x / y = 0.8
theorem find_ratio (h₁ : x ≠ y) (h₂ : x / y + (x + 15 * y) / (y + 15 * x) = 3) : x / y = 0.8 :=
sorry

end find_ratio_l143_14337


namespace simplify_expression_l143_14307

variable (x y : ℕ)
variable (h_x : x = 5)
variable (h_y : y = 2)

theorem simplify_expression : (10 * x^2 * y^3) / (15 * x * y^2) = 20 / 3 := by
  sorry

end simplify_expression_l143_14307


namespace magnitude_of_a_l143_14365

open Real

-- Assuming the standard inner product space for vectors in Euclidean space

variables (a b : ℝ) -- Vectors in R^n (could be general but simplified to real numbers for this example)
variable (θ : ℝ)    -- Angle between vectors
axiom angle_ab : θ = 60 -- Given angle between vectors

-- Conditions:
axiom non_zero_a : a ≠ 0
axiom non_zero_b : b ≠ 0
axiom norm_b : abs b = 1
axiom norm_2a_minus_b : abs (2 * a - b) = 1

-- To prove:
theorem magnitude_of_a : abs a = 1 / 2 :=
sorry

end magnitude_of_a_l143_14365


namespace intersection_A_B_l143_14381

def A (x : ℝ) : Prop := 0 < x ∧ x < 2
def B (x : ℝ) : Prop := -1 < x ∧ x < 1
def C (x : ℝ) : Prop := 0 < x ∧ x < 1

theorem intersection_A_B : ∀ x, A x ∧ B x ↔ C x := by
  sorry

end intersection_A_B_l143_14381


namespace circle_center_and_radius_l143_14351

-- Define a circle in the plane according to the given equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x = 0

-- Define the center of the circle
def center (x : ℝ) (y : ℝ) : Prop := x = -2 ∧ y = 0

-- Define the radius of the circle
def radius (r : ℝ) : Prop := r = 2

-- The theorem statement
theorem circle_center_and_radius :
  (∀ x y, circle_eq x y → center x y) ∧ radius 2 :=
sorry

end circle_center_and_radius_l143_14351


namespace find_x2_plus_y2_l143_14357

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + 2 * x + 2 * y = 88) :
    x^2 + y^2 = 304 / 9 := sorry

end find_x2_plus_y2_l143_14357


namespace isosceles_triangle_exists_l143_14368

-- Definitions for a triangle vertex and side lengths
structure Triangle :=
  (A B C : ℝ × ℝ) -- Vertices A, B, C
  (AB AC BC : ℝ)  -- Sides AB, AC, BC

-- Definition for all sides being less than 1 unit
def sides_less_than_one (T : Triangle) : Prop :=
  T.AB < 1 ∧ T.AC < 1 ∧ T.BC < 1

-- Definition for isosceles triangle containing the original one
def exists_isosceles_containing (T : Triangle) : Prop :=
  ∃ (T' : Triangle), 
    (T'.AB = T'.AC ∨ T'.AB = T'.BC ∨ T'.AC = T'.BC) ∧
    T'.A = T.A ∧ -- T'.A vertex is same as T.A
    (T'.AB < 1 ∧ T'.AC < 1 ∧ T'.BC < 1) ∧
    (∃ (B1 : ℝ × ℝ), -- There exists point B1 such that new triangle T' incorporates B1
      T'.B = B1 ∧
      T'.C = T.C) -- T' also has vertex C of original triangle

-- Complete theorem statement
theorem isosceles_triangle_exists (T : Triangle) (hT : sides_less_than_one T) : exists_isosceles_containing T :=
by 
  sorry

end isosceles_triangle_exists_l143_14368


namespace find_f_2006_l143_14392

variable (f g : ℝ → ℝ)

-- Conditions
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x
def g_def (f : ℝ → ℝ) (g : ℝ → ℝ) := ∀ x : ℝ, g x = f (x - 1)
def f_at_2 (f : ℝ → ℝ) := f 2 = 2

-- The theorem to prove
theorem find_f_2006 (f g : ℝ → ℝ) 
  (even_f : is_even f) 
  (odd_g : is_odd g) 
  (g_eq_f_shift : g_def f g) 
  (f_eq_2 : f_at_2 f) : 
  f 2006 = 2 := 
sorry

end find_f_2006_l143_14392


namespace quadrilateral_inequality_l143_14345

variable (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)

def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ := (x1 - x2)^2 + (y1 - y2)^2

-- Given a convex quadrilateral ABCD with vertices at (x1, y1), (x2, y2), (x3, y3), and (x4, y4), prove:
theorem quadrilateral_inequality :
  (distance_squared x1 y1 x3 y3 + distance_squared x2 y2 x4 y4) ≤
  (distance_squared x1 y1 x2 y2 + distance_squared x2 y2 x3 y3 +
   distance_squared x3 y3 x4 y4 + distance_squared x4 y4 x1 y1) :=
sorry

end quadrilateral_inequality_l143_14345


namespace reciprocal_geometric_sum_l143_14383

/-- The sum of the new geometric progression formed by taking the reciprocal of each term in the original progression,
    where the original progression has 10 terms, the first term is 2, and the common ratio is 3, is \( \frac{29524}{59049} \). -/
theorem reciprocal_geometric_sum :
  let a := 2
  let r := 3
  let n := 10
  let sn := (2 * (1 - r^n)) / (1 - r)
  let sn_reciprocal := (1 / a) * (1 - (1/r)^n) / (1 - 1/r)
  (sn_reciprocal = 29524 / 59049) :=
by
  sorry

end reciprocal_geometric_sum_l143_14383


namespace numerical_puzzle_solution_l143_14364

theorem numerical_puzzle_solution (A B V : ℕ) (h_diff_digits : A ≠ B) (h_two_digit : 10 ≤ A * 10 + B ∧ A * 10 + B < 100) :
  (A * 10 + B = B^V) → (A = 3 ∧ B = 2 ∧ V = 5) ∨ (A = 3 ∧ B = 6 ∧ V = 2) ∨ (A = 6 ∧ B = 4 ∧ V = 3) :=
sorry

end numerical_puzzle_solution_l143_14364


namespace sum_solutions_eq_16_l143_14342

theorem sum_solutions_eq_16 (x y : ℝ) 
  (h1 : |x - 5| = |y - 11|)
  (h2 : |x - 11| = 2 * |y - 5|)
  (h3 : x + y = 16) :
  x + y = 16 :=
by
  sorry

end sum_solutions_eq_16_l143_14342


namespace probability_one_each_item_l143_14326

theorem probability_one_each_item :
  let num_items := 32
  let total_ways := Nat.choose num_items 4
  let favorable_outcomes := 8 * 8 * 8 * 8
  total_ways = 35960 →
  let probability := favorable_outcomes / total_ways
  probability = (128 : ℚ) / 1125 :=
by
  sorry

end probability_one_each_item_l143_14326


namespace find_range_of_a_l143_14374

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^3) / 3 - (a / 2) * x^2 + x + 1

def is_monotonically_decreasing_in (a : ℝ) (x : ℝ) : Prop := 
  ∀ s t : ℝ, (s ∈ Set.Ioo (3 / 2) 4) ∧ (t ∈ Set.Ioo (3 / 2) 4) ∧ s < t → 
  f a t ≤ f a s

theorem find_range_of_a :
  ∀ a : ℝ, is_monotonically_decreasing_in a x → 
  a ∈ Set.Ici (17/4)
:= sorry

end find_range_of_a_l143_14374


namespace sum_x_y_650_l143_14347

theorem sum_x_y_650 (x y : ℤ) (h1 : x - y = 200) (h2 : y = 225) : x + y = 650 :=
by
  sorry

end sum_x_y_650_l143_14347


namespace system1_solution_system2_solution_l143_14352

-- Problem 1
theorem system1_solution (x z : ℤ) (h1 : 3 * x - 5 * z = 6) (h2 : x + 4 * z = -15) : x = -3 ∧ z = -3 :=
by
  sorry

-- Problem 2
theorem system2_solution (x y : ℚ) 
 (h1 : ((2 * x - 1) / 5) + ((3 * y - 2) / 4) = 2) 
 (h2 : ((3 * x + 1) / 5) - ((3 * y + 2) / 4) = 0) : x = 3 ∧ y = 2 :=
by
  sorry

end system1_solution_system2_solution_l143_14352


namespace fractional_exponent_calculation_l143_14380

variables (a b : ℝ) -- Define a and b as real numbers
variable (ha : a > 0) -- Condition a > 0
variable (hb : b > 0) -- Condition b > 0

theorem fractional_exponent_calculation :
  (a^(2 * b^(1/4)) / (a * b^(1/2))^(1/2)) = a^(1/2) :=
by
  sorry -- Proof is not required, skip with sorry

end fractional_exponent_calculation_l143_14380


namespace parabola_focus_l143_14391

theorem parabola_focus :
  ∀ (x y : ℝ), x^2 = 4 * y → (0, 1) = (0, (2 / 2)) :=
by
  intros x y h
  sorry

end parabola_focus_l143_14391


namespace solve_for_a_l143_14319

def op (a b : ℝ) : ℝ := 3 * a - 2 * b ^ 2

theorem solve_for_a (a : ℝ) : op a 3 = 15 → a = 11 :=
by
  intro h
  rw [op] at h
  sorry

end solve_for_a_l143_14319


namespace necessary_but_not_sufficient_l143_14304

theorem necessary_but_not_sufficient (a : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + a < 0) → (a < 11) ∧ ¬((a < 11) → (∃ x : ℝ, x^2 - 2*x + a < 0)) :=
by
  -- Sorry to bypass proof below, which is correct as per the problem statement requirements.
  sorry

end necessary_but_not_sufficient_l143_14304


namespace complex_number_in_first_quadrant_l143_14397

open Complex

theorem complex_number_in_first_quadrant 
    (z : ℂ) 
    (h : z = (3 + I) / (1 - 3 * I) + 2) : 
    0 < z.re ∧ 0 < z.im :=
by
  sorry

end complex_number_in_first_quadrant_l143_14397


namespace pollywogs_disappear_in_44_days_l143_14314

theorem pollywogs_disappear_in_44_days :
  ∀ (initial_count rate_mature rate_caught first_period_days : ℕ),
  initial_count = 2400 →
  rate_mature = 50 →
  rate_caught = 10 →
  first_period_days = 20 →
  (initial_count - first_period_days * (rate_mature + rate_caught)) / rate_mature + first_period_days = 44 := 
by
  intros initial_count rate_mature rate_caught first_period_days h1 h2 h3 h4
  sorry

end pollywogs_disappear_in_44_days_l143_14314


namespace john_spent_fraction_l143_14310

theorem john_spent_fraction (initial_money snacks_left necessities_left snacks_fraction : ℝ)
  (h1 : initial_money = 20)
  (h2 : snacks_fraction = 1/5)
  (h3 : snacks_left = initial_money * snacks_fraction)
  (h4 : necessities_left = 4)
  (remaining_money : ℝ) (h5 : remaining_money = initial_money - snacks_left)
  (spent_on_necessities : ℝ) (h6 : spent_on_necessities = remaining_money - necessities_left) 
  (fraction_spent : ℝ) (h7 : fraction_spent = spent_on_necessities / remaining_money) : 
  fraction_spent = 3/4 := 
sorry

end john_spent_fraction_l143_14310


namespace cause_of_polarization_by_electronegativity_l143_14387

-- Definition of the problem conditions as hypotheses
def strong_polarization_of_CH_bond (C_H_bond : Prop) (electronegativity : Prop) : Prop 
  := C_H_bond ∧ electronegativity

-- Given conditions: Carbon atom is in sp hybridization and C-H bond shows strong polarization
axiom carbon_sp_hybridized : Prop
axiom CH_bond_strong_polarization : Prop

-- Question: The cause of strong polarization of the C-H bond at the carbon atom in sp hybridization in alkynes
def cause_of_strong_polarization (sp_hybridization : Prop) : Prop 
  := true  -- This definition will hold as a placeholder, to indicate there is a causal connection

-- Correct answer: high electronegativity of the carbon atom in sp-hybrid state causes strong polarization
theorem cause_of_polarization_by_electronegativity 
  (high_electronegativity : Prop) 
  (sp_hybridized : Prop) 
  (polarized : Prop) 
  (H : strong_polarization_of_CH_bond polarized high_electronegativity) 
  : sp_hybridized ∧ polarized := 
  sorry

end cause_of_polarization_by_electronegativity_l143_14387


namespace find_k_l143_14300

noncomputable def expr_to_complete_square (x : ℝ) : ℝ :=
  x^2 - 6 * x

theorem find_k (x : ℝ) : ∃ a h k, expr_to_complete_square x = a * (x - h)^2 + k ∧ k = -9 :=
by
  use 1, 3, -9
  -- detailed steps of the proof would go here
  sorry

end find_k_l143_14300


namespace total_cost_charlotte_l143_14377

noncomputable def regular_rate : ℝ := 40.00
noncomputable def discount_rate : ℝ := 0.25
noncomputable def number_of_people : ℕ := 5

theorem total_cost_charlotte :
  number_of_people * (regular_rate * (1 - discount_rate)) = 150.00 := by
  sorry

end total_cost_charlotte_l143_14377


namespace georges_final_score_l143_14336

theorem georges_final_score :
  (6 + 4) * 3 = 30 := 
by
  sorry

end georges_final_score_l143_14336


namespace right_triangle_of_altitude_ratios_l143_14328

theorem right_triangle_of_altitude_ratios
  (h1 h2 h3 : ℝ) 
  (h1_pos : h1 > 0) 
  (h2_pos : h2 > 0) 
  (h3_pos : h3 > 0) 
  (H : (h1 / h2)^2 + (h1 / h3)^2 = 1) : 
  ∃ a b c : ℝ, a^2 = b^2 + c^2 ∧ h1 = 1 / a ∧ h2 = 1 / b ∧ h3 = 1 / c :=
sorry

end right_triangle_of_altitude_ratios_l143_14328
