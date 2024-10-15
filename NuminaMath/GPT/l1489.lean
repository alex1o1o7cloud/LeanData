import Mathlib

namespace NUMINAMATH_GPT_relation_xy_l1489_148994

theorem relation_xy (a c b d : ℝ) (x y p : ℝ) 
  (h1 : a^x = c^(3 * p))
  (h2 : c^(3 * p) = b^2)
  (h3 : c^y = b^p)
  (h4 : b^p = d^3) :
  y = 3 * p^2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_relation_xy_l1489_148994


namespace NUMINAMATH_GPT_problem_statement_l1489_148931

theorem problem_statement : 
  (777 % 4 = 1) ∧ 
  (555 % 4 = 3) ∧ 
  (999 % 4 = 3) → 
  ( (999^2021 * 555^2021 - 1) % 4 = 0 ∧ 
    (777^2021 * 999^2021 - 1) % 4 ≠ 0 ∧ 
    (555^2021 * 777^2021 - 1) % 4 ≠ 0 ) := 
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1489_148931


namespace NUMINAMATH_GPT_weight_of_each_bag_of_flour_l1489_148970

theorem weight_of_each_bag_of_flour
  (flour_weight_needed : ℕ)
  (cost_per_bag : ℕ)
  (salt_weight_needed : ℕ)
  (salt_cost_per_pound : ℚ)
  (promotion_cost : ℕ)
  (ticket_price : ℕ)
  (tickets_sold : ℕ)
  (total_made : ℕ)
  (profit : ℕ)
  (total_flour_cost : ℕ)
  (num_bags : ℕ)
  (weight_per_bag : ℕ)
  (calc_salt_cost : ℚ := salt_weight_needed * salt_cost_per_pound)
  (calc_total_earnings : ℕ := tickets_sold * ticket_price)
  (calc_total_cost : ℚ := calc_total_earnings - profit)
  (calc_flour_cost : ℚ := calc_total_cost - calc_salt_cost - promotion_cost)
  (calc_num_bags : ℚ := calc_flour_cost / cost_per_bag)
  (calc_weight_per_bag : ℚ := flour_weight_needed / calc_num_bags) :
  flour_weight_needed = 500 ∧
  cost_per_bag = 20 ∧
  salt_weight_needed = 10 ∧
  salt_cost_per_pound = 0.2 ∧
  promotion_cost = 1000 ∧
  ticket_price = 20 ∧
  tickets_sold = 500 ∧
  total_made = 8798 ∧
  profit = 10000 - total_made ∧
  calc_salt_cost = 2 ∧
  calc_total_earnings = 10000 ∧
  calc_total_cost = 1202 ∧
  calc_flour_cost = 200 ∧
  calc_num_bags = 10 ∧
  calc_weight_per_bag = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_weight_of_each_bag_of_flour_l1489_148970


namespace NUMINAMATH_GPT_not_all_angles_less_than_60_l1489_148904

-- Definitions relating to interior angles of a triangle
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180

theorem not_all_angles_less_than_60 (α β γ : ℝ) 
(h_triangle : triangle α β γ) 
(h1 : α < 60) 
(h2 : β < 60) 
(h3 : γ < 60) : False :=
    -- The proof steps would be placed here
sorry

end NUMINAMATH_GPT_not_all_angles_less_than_60_l1489_148904


namespace NUMINAMATH_GPT_time_for_B_alone_to_complete_work_l1489_148903

theorem time_for_B_alone_to_complete_work :
  (∃ (A B C : ℝ), A = 1 / 4 
  ∧ B + C = 1 / 3
  ∧ A + C = 1 / 2)
  → 1 / B = 12 :=
by
  sorry

end NUMINAMATH_GPT_time_for_B_alone_to_complete_work_l1489_148903


namespace NUMINAMATH_GPT_jolyn_older_than_leon_l1489_148952

open Nat

def Jolyn := Nat
def Therese := Nat
def Aivo := Nat
def Leon := Nat

-- Conditions
variable (jolyn therese aivo leon : Nat)
variable (h1 : jolyn = therese + 2) -- Jolyn is 2 months older than Therese
variable (h2 : therese = aivo + 5) -- Therese is 5 months older than Aivo
variable (h3 : leon = aivo + 2) -- Leon is 2 months older than Aivo

theorem jolyn_older_than_leon :
  jolyn = leon + 5 := by
  sorry

end NUMINAMATH_GPT_jolyn_older_than_leon_l1489_148952


namespace NUMINAMATH_GPT_original_rent_l1489_148913

theorem original_rent {avg_rent_before avg_rent_after : ℝ} (total_before total_after increase_percentage diff_increase : ℝ) :
  avg_rent_before = 800 → 
  avg_rent_after = 880 → 
  total_before = 4 * avg_rent_before → 
  total_after = 4 * avg_rent_after → 
  diff_increase = total_after - total_before → 
  increase_percentage = 0.20 → 
  diff_increase = increase_percentage * R → 
  R = 1600 :=
by sorry

end NUMINAMATH_GPT_original_rent_l1489_148913


namespace NUMINAMATH_GPT_largest_fraction_l1489_148959

theorem largest_fraction (p q r s : ℕ) (hp : 0 < p) (hpq : p < q) (hqr : q < r) (hrs : r < s) : 
  max (max (max (max (↑(p + q) / ↑(r + s)) (↑(p + s) / ↑(q + r))) 
              (↑(q + r) / ↑(p + s))) 
          (↑(q + s) / ↑(p + r))) 
      (↑(r + s) / ↑(p + q)) = (↑(r + s) / ↑(p + q)) :=
sorry

end NUMINAMATH_GPT_largest_fraction_l1489_148959


namespace NUMINAMATH_GPT_percentage_neither_l1489_148921

theorem percentage_neither (total_teachers high_blood_pressure heart_trouble both_conditions : ℕ)
  (h1 : total_teachers = 150)
  (h2 : high_blood_pressure = 90)
  (h3 : heart_trouble = 60)
  (h4 : both_conditions = 30) :
  100 * (total_teachers - (high_blood_pressure + heart_trouble - both_conditions)) / total_teachers = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_neither_l1489_148921


namespace NUMINAMATH_GPT_total_journey_time_eq_5_l1489_148928

-- Define constants for speed and times
def speed1 : ℕ := 40
def speed2 : ℕ := 60
def total_distance : ℕ := 240
def time1 : ℕ := 3

-- Noncomputable definition to avoid computation issues
noncomputable def journey_time : ℕ :=
  let distance1 := speed1 * time1
  let distance2 := total_distance - distance1
  let time2 := distance2 / speed2
  time1 + time2

-- Theorem to state the total journey time
theorem total_journey_time_eq_5 : journey_time = 5 := by
  sorry

end NUMINAMATH_GPT_total_journey_time_eq_5_l1489_148928


namespace NUMINAMATH_GPT_tan_alpha_minus_beta_value_l1489_148957

theorem tan_alpha_minus_beta_value (α β : Real) 
  (h1 : Real.sin α = 3 / 5) 
  (h2 : α ∈ Set.Ioo (π / 2) π) 
  (h3 : Real.tan (π - β) = 1 / 2) : 
  Real.tan (α - β) = -2 / 11 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_minus_beta_value_l1489_148957


namespace NUMINAMATH_GPT_squares_equal_l1489_148958

theorem squares_equal (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) 
    : ∃ (k : ℤ), a^2 + b^2 - c^2 = k^2 := 
by 
  sorry

end NUMINAMATH_GPT_squares_equal_l1489_148958


namespace NUMINAMATH_GPT_value_of_m_l1489_148920

theorem value_of_m (a b m : ℚ) (h1 : 2 * a = m) (h2 : 5 * b = m) (h3 : a + b = 2) : m = 20 / 7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l1489_148920


namespace NUMINAMATH_GPT_largest_angle_of_triangle_l1489_148916

theorem largest_angle_of_triangle (A B C : ℝ) :
  A + B + C = 180 ∧ A + B = 126 ∧ abs (A - B) = 45 → max A (max B C) = 85.5 :=
by sorry

end NUMINAMATH_GPT_largest_angle_of_triangle_l1489_148916


namespace NUMINAMATH_GPT_fraction_given_to_emma_is_7_over_36_l1489_148976

-- Define initial quantities
def stickers_noah : ℕ := sorry
def stickers_emma : ℕ := 3 * stickers_noah
def stickers_liam : ℕ := 12 * stickers_noah

-- Define required number of stickers for equal distribution
def total_stickers := stickers_noah + stickers_emma + stickers_liam
def equal_stickers := total_stickers / 3

-- Define the number of stickers to be given to Emma and the fraction of Liam's stickers he should give to Emma
def stickers_given_to_emma := equal_stickers - stickers_emma
def fraction_liams_stickers_given_to_emma := stickers_given_to_emma / stickers_liam

-- Theorem statement
theorem fraction_given_to_emma_is_7_over_36 :
  fraction_liams_stickers_given_to_emma = 7 / 36 :=
sorry

end NUMINAMATH_GPT_fraction_given_to_emma_is_7_over_36_l1489_148976


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1489_148938

theorem neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a + b > 0) ↔ (ab > 0)) := 
sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1489_148938


namespace NUMINAMATH_GPT_number_of_balloons_Allan_bought_l1489_148989

theorem number_of_balloons_Allan_bought 
  (initial_balloons final_balloons : ℕ) 
  (h1 : initial_balloons = 5) 
  (h2 : final_balloons = 8) : 
  final_balloons - initial_balloons = 3 := 
  by 
  sorry

end NUMINAMATH_GPT_number_of_balloons_Allan_bought_l1489_148989


namespace NUMINAMATH_GPT_banana_permutations_l1489_148930

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end NUMINAMATH_GPT_banana_permutations_l1489_148930


namespace NUMINAMATH_GPT_find_a_l1489_148978

theorem find_a (a : ℝ) : (∃ p : ℝ × ℝ, p = (2 - a, a - 3) ∧ p.fst = 0) → a = 2 := by
  sorry

end NUMINAMATH_GPT_find_a_l1489_148978


namespace NUMINAMATH_GPT_proof1_proof2_l1489_148997

-- Definitions based on the conditions given in the problem description.

def f1 (x : ℝ) (a : ℝ) : ℝ := abs (3 * x - 1) + a * x + 3

theorem proof1 (x : ℝ) : (f1 x 1 ≤ 4) ↔ 0 ≤ x ∧ x ≤ 1 / 2 := 
by
  sorry

theorem proof2 (a : ℝ) : (-3 ≤ a ∧ a ≤ 3) ↔ 
  ∃ (x : ℝ), ∀ y : ℝ, f1 x a ≤ f1 y a := 
by
  sorry

end NUMINAMATH_GPT_proof1_proof2_l1489_148997


namespace NUMINAMATH_GPT_average_speed_comparison_l1489_148902

variables (u v : ℝ) (hu : u > 0) (hv : v > 0)

theorem average_speed_comparison (x y : ℝ) 
  (hx : x = 2 * u * v / (u + v)) 
  (hy : y = (u + v) / 2) : x ≤ y := 
sorry

end NUMINAMATH_GPT_average_speed_comparison_l1489_148902


namespace NUMINAMATH_GPT_trigonometric_identity_l1489_148980

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (1 - Real.sin (2 * θ)) / (2 * (Real.cos θ)^2) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1489_148980


namespace NUMINAMATH_GPT_exists_monotonic_subsequence_l1489_148909

open Function -- For function related definitions
open Finset -- For finite set operations

-- Defining the theorem with the given conditions and the goal to be proved
theorem exists_monotonic_subsequence (a : Fin 10 → ℝ) (h : ∀ i j : Fin 10, i ≠ j → a i ≠ a j) :
  ∃ (i1 i2 i3 i4 : Fin 10), i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧
  ((a i1 < a i2 ∧ a i2 < a i3 ∧ a i3 < a i4) ∨ (a i1 > a i2 ∧ a i2 > a i3 ∧ a i3 > a i4)) :=
by
  sorry -- Proof is omitted as per the instructions

end NUMINAMATH_GPT_exists_monotonic_subsequence_l1489_148909


namespace NUMINAMATH_GPT_rational_non_positive_l1489_148946

variable (a : ℚ)

theorem rational_non_positive (h : ∃ a : ℚ, True) : 
  -a^2 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_rational_non_positive_l1489_148946


namespace NUMINAMATH_GPT_range_of_x_l1489_148900

variable {x p : ℝ}

theorem range_of_x (H : 0 ≤ p ∧ p ≤ 4) : 
  (x^2 + p * x > 4 * x + p - 3) ↔ (x ∈ Set.Iio (-1) ∪ Set.Ioi 3) := 
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1489_148900


namespace NUMINAMATH_GPT_six_digit_start_5_not_possible_l1489_148906

theorem six_digit_start_5_not_possible :
  ∀ n : ℕ, (n ≥ 500000 ∧ n < 600000) → (¬ ∃ m : ℕ, (n * 10^6 + m) ^ 2 < 10^12 ∧ (n * 10^6 + m) ^ 2 ≥ 5 * 10^11 ∧ (n * 10^6 + m) ^ 2 < 6 * 10^11) :=
sorry

end NUMINAMATH_GPT_six_digit_start_5_not_possible_l1489_148906


namespace NUMINAMATH_GPT_find_base_of_exponential_l1489_148949

theorem find_base_of_exponential (a : ℝ) 
  (h₁ : a > 0) 
  (h₂ : a ≠ 1) 
  (h₃ : a ^ 2 = 1 / 16) : 
  a = 1 / 4 := 
sorry

end NUMINAMATH_GPT_find_base_of_exponential_l1489_148949


namespace NUMINAMATH_GPT_find_f_2008_l1489_148948

noncomputable def f : ℝ → ℝ := sorry

axiom f_zero (f : ℝ → ℝ) : f 0 = 2008
axiom f_inequality_1 (f : ℝ → ℝ) (x : ℝ) : f (x + 2) - f x ≤ 3 * 2^x
axiom f_inequality_2 (f : ℝ → ℝ) (x : ℝ) : f (x + 6) - f x ≥ 63 * 2^x

theorem find_f_2008 (f : ℝ → ℝ) : f 2008 = 2^2008 + 2007 :=
by
  apply sorry

end NUMINAMATH_GPT_find_f_2008_l1489_148948


namespace NUMINAMATH_GPT_factor_tree_value_l1489_148998

theorem factor_tree_value :
  let Q := 5 * 3
  let R := 11 * 2
  let Y := 2 * Q
  let Z := 7 * R
  let X := Y * Z
  X = 4620 :=
by
  sorry

end NUMINAMATH_GPT_factor_tree_value_l1489_148998


namespace NUMINAMATH_GPT_value_of_expression_eq_33_l1489_148911

theorem value_of_expression_eq_33 : (3^2 + 7^2 - 5^2 = 33) := by
  sorry

end NUMINAMATH_GPT_value_of_expression_eq_33_l1489_148911


namespace NUMINAMATH_GPT_recording_incorrect_l1489_148917

-- Definitions for given conditions
def qualifying_standard : ℝ := 1.5
def xiao_ming_jump : ℝ := 1.95
def xiao_liang_jump : ℝ := 1.23
def xiao_ming_recording : ℝ := 0.45
def xiao_liang_recording : ℝ := -0.23

-- The proof statement to verify the correctness of the recordings
theorem recording_incorrect :
  (xiao_ming_jump - qualifying_standard = xiao_ming_recording) ∧ 
  (xiao_liang_jump - qualifying_standard ≠ xiao_liang_recording) :=
by
  sorry

end NUMINAMATH_GPT_recording_incorrect_l1489_148917


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1489_148934

variable (a : ℝ)

theorem sufficient_not_necessary_condition :
  (1 < a ∧ a < 2) → (a^2 - 3 * a ≤ 0) := by
  intro h
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1489_148934


namespace NUMINAMATH_GPT_banana_cost_is_2_l1489_148983

noncomputable def bananas_cost (B : ℝ) : Prop :=
  let cost_oranges : ℝ := 10 * 1.5
  let total_cost : ℝ := 25
  let cost_bananas : ℝ := total_cost - cost_oranges
  let num_bananas : ℝ := 5
  B = cost_bananas / num_bananas

theorem banana_cost_is_2 : bananas_cost 2 :=
by
  unfold bananas_cost
  sorry

end NUMINAMATH_GPT_banana_cost_is_2_l1489_148983


namespace NUMINAMATH_GPT_number_of_students_with_at_least_two_pets_l1489_148910

-- Definitions for the sets of students
def total_students := 50
def dog_students := 35
def cat_students := 40
def rabbit_students := 10
def dog_and_cat_students := 20
def dog_and_rabbit_students := 5
def cat_and_rabbit_students := 0  -- Assuming minimal overlap

-- Problem Statement
theorem number_of_students_with_at_least_two_pets :
  (dog_and_cat_students + dog_and_rabbit_students + cat_and_rabbit_students) = 25 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_with_at_least_two_pets_l1489_148910


namespace NUMINAMATH_GPT_undefined_count_expression_l1489_148969

theorem undefined_count_expression : 
  let expr (x : ℝ) := (x^2 - 16) / ((x^2 + 3*x - 10) * (x - 4))
  ∃ u v w : ℝ, (u = 2 ∨ v = -5 ∨ w = 4) ∧
  (u ≠ v ∧ u ≠ w ∧ v ≠ w) :=
by
  sorry

end NUMINAMATH_GPT_undefined_count_expression_l1489_148969


namespace NUMINAMATH_GPT_ratio_first_part_l1489_148999

theorem ratio_first_part (x : ℝ) (h1 : 180 / 100 * 5 = x) : x = 9 :=
by sorry

end NUMINAMATH_GPT_ratio_first_part_l1489_148999


namespace NUMINAMATH_GPT_arithmetic_seq_problem_l1489_148961

variable {a : Nat → ℝ}  -- a_n represents the value at index n
variable {d : ℝ} -- The common difference in the arithmetic sequence

-- Define the general term of the arithmetic sequence
def arithmeticSeq (a : Nat → ℝ) (a1 : ℝ) (d : ℝ) : Prop :=
  ∀ n : Nat, a n = a1 + n * d

-- The main proof problem
theorem arithmetic_seq_problem
  (a1 : ℝ)
  (d : ℝ)
  (a : Nat → ℝ)
  (h_arithmetic: arithmeticSeq a a1 d)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) : 
  a 7 - (1 / 2) * a 8 = 8 := 
  by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_problem_l1489_148961


namespace NUMINAMATH_GPT_solve_floor_eq_l1489_148967

theorem solve_floor_eq (x : ℝ) (hx_pos : 0 < x) (h : (⌊x⌋ : ℝ) * x = 110) : x = 11 := 
sorry

end NUMINAMATH_GPT_solve_floor_eq_l1489_148967


namespace NUMINAMATH_GPT_tom_books_read_in_may_l1489_148990

def books_read_in_june := 6
def books_read_in_july := 10
def total_books_read := 18

theorem tom_books_read_in_may : total_books_read - (books_read_in_june + books_read_in_july) = 2 :=
by sorry

end NUMINAMATH_GPT_tom_books_read_in_may_l1489_148990


namespace NUMINAMATH_GPT_smallest_number_divided_into_18_and_60_groups_l1489_148945

theorem smallest_number_divided_into_18_and_60_groups : ∃ n : ℕ, (∀ m : ℕ, (m % 18 = 0 ∧ m % 60 = 0) → n ≤ m) ∧ (n % 18 = 0 ∧ n % 60 = 0) ∧ n = 180 :=
by
  use 180
  sorry

end NUMINAMATH_GPT_smallest_number_divided_into_18_and_60_groups_l1489_148945


namespace NUMINAMATH_GPT_exists_negative_value_of_f_l1489_148950

noncomputable def f : ℝ → ℝ := sorry

axiom f_monotonic (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) : f x < f y
axiom f_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f ((2 * x * y) / (x + y)) ≥ (f x + f y) / 2

theorem exists_negative_value_of_f : ∃ x > 0, f x < 0 := 
sorry

end NUMINAMATH_GPT_exists_negative_value_of_f_l1489_148950


namespace NUMINAMATH_GPT_ribbon_left_l1489_148924

theorem ribbon_left (gifts : ℕ) (ribbon_per_gift Tom_ribbon_total : ℝ) (h1 : gifts = 8) (h2 : ribbon_per_gift = 1.5) (h3 : Tom_ribbon_total = 15) : Tom_ribbon_total - (gifts * ribbon_per_gift) = 3 := 
by
  sorry

end NUMINAMATH_GPT_ribbon_left_l1489_148924


namespace NUMINAMATH_GPT_sachin_age_is_49_l1489_148929

open Nat

-- Let S be Sachin's age and R be Rahul's age
def Sachin_age : ℕ := 49
def Rahul_age (S : ℕ) := S + 14

theorem sachin_age_is_49 (S R : ℕ) (h1 : R = S + 14) (h2 : S * 9 = R * 7) : S = 49 :=
by sorry

end NUMINAMATH_GPT_sachin_age_is_49_l1489_148929


namespace NUMINAMATH_GPT_product_of_two_numbers_l1489_148947

theorem product_of_two_numbers (x y : ℝ)
  (h1 : x + y = 25)
  (h2 : x - y = 3)
  : x * y = 154 := by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1489_148947


namespace NUMINAMATH_GPT_total_stickers_purchased_l1489_148936

-- Definitions for the number of sheets and stickers per sheet for each folder
def num_sheets_per_folder := 10
def stickers_per_sheet_red := 3
def stickers_per_sheet_green := 2
def stickers_per_sheet_blue := 1

-- Theorem stating that the total number of stickers is 60
theorem total_stickers_purchased : 
  num_sheets_per_folder * (stickers_per_sheet_red + stickers_per_sheet_green + stickers_per_sheet_blue) = 60 := 
  by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_total_stickers_purchased_l1489_148936


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_for_positive_sum_l1489_148953

variable (q : ℤ) (a1 : ℤ)

def geometric_sequence (n : ℕ) : ℤ := a1 * q ^ (n - 1)

def sum_of_first_n_terms (n : ℕ) : ℤ :=
  if q = 1 then a1 * n else (a1 * (1 - q ^ n)) / (1 - q)

theorem sufficient_and_necessary_condition_for_positive_sum :
  (a1 > 0) ↔ (sum_of_first_n_terms q a1 2017 > 0) :=
sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_for_positive_sum_l1489_148953


namespace NUMINAMATH_GPT_triangle_height_from_area_l1489_148915

theorem triangle_height_from_area {A b h : ℝ} (hA : A = 36) (hb : b = 8) 
    (formula : A = 1 / 2 * b * h) : h = 9 := 
by
  sorry

end NUMINAMATH_GPT_triangle_height_from_area_l1489_148915


namespace NUMINAMATH_GPT_bobby_initial_pieces_l1489_148974

-- Definitions based on the conditions
def pieces_eaten_1 := 17
def pieces_eaten_2 := 15
def pieces_left := 4

-- Definition based on the question and answer
def initial_pieces (pieces_eaten_1 pieces_eaten_2 pieces_left : ℕ) : ℕ :=
  pieces_eaten_1 + pieces_eaten_2 + pieces_left

-- Theorem stating the problem and the expected answer
theorem bobby_initial_pieces : 
  initial_pieces pieces_eaten_1 pieces_eaten_2 pieces_left = 36 :=
by 
  sorry

end NUMINAMATH_GPT_bobby_initial_pieces_l1489_148974


namespace NUMINAMATH_GPT_parallel_lines_perpendicular_lines_l1489_148933

theorem parallel_lines (a : ℝ) :
  (∃ b c : ℝ, (ax - y + b = 0) ∧ ((a + 2) * x - ay - c = 0)) →
  (∀ s1 s2 : ℝ, s1 = a → s2 = (a + 2) / a → s1 = s2) →
  a = 2 :=
by
  intros
  -- Proof goes here
  sorry

theorem perpendicular_lines (a : ℝ) :
  (∃ b c : ℝ, (ax - y + b = 0) ∧ ((a + 2) * x - ay - c = 0)) →
  (∀ s1 s2 : ℝ, s1 = a → s2 = (a + 2) / a → s1 * s2 = -1) →
  a = 0 ∨ a = -3 :=
by
  intros
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_parallel_lines_perpendicular_lines_l1489_148933


namespace NUMINAMATH_GPT_num_perfect_squares_diff_consecutive_under_20000_l1489_148922

theorem num_perfect_squares_diff_consecutive_under_20000 : 
  ∃ n, n = 71 ∧ ∀ a, a ^ 2 < 20000 → ∃ b, a ^ 2 = (b + 1) ^ 2 - b ^ 2 ↔ a ^ 2 % 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_num_perfect_squares_diff_consecutive_under_20000_l1489_148922


namespace NUMINAMATH_GPT_rationalize_denominator_l1489_148907

theorem rationalize_denominator :
  ∃ (A B C : ℤ), 
  (A + B * Real.sqrt C) = (2 + Real.sqrt 5) / (3 - Real.sqrt 5) 
  ∧ A = 11 ∧ B = 5 ∧ C = 5 ∧ A * B * C = 275 := by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l1489_148907


namespace NUMINAMATH_GPT_find_slope_of_line_q_l1489_148935

theorem find_slope_of_line_q
  (k : ℝ)
  (h₁ : ∀ (x y : ℝ), (y = 3 * x + 5) → (y = k * x + 3) → (x = -4 ∧ y = -7))
  : k = 2.5 :=
sorry

end NUMINAMATH_GPT_find_slope_of_line_q_l1489_148935


namespace NUMINAMATH_GPT_find_difference_l1489_148965

noncomputable def expr (a b : ℝ) : ℝ :=
  |a - b| / (|a| + |b|)

def min_val (a b : ℝ) : ℝ := 0

def max_val (a b : ℝ) : ℝ := 1

theorem find_difference (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  max_val a b - min_val a b = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_difference_l1489_148965


namespace NUMINAMATH_GPT_find_number_l1489_148991

theorem find_number :
  ∃ x : ℤ, 27 * (x + 143) = 9693 ∧ x = 216 :=
by 
  existsi 216
  sorry

end NUMINAMATH_GPT_find_number_l1489_148991


namespace NUMINAMATH_GPT_factorization_eq_l1489_148955

theorem factorization_eq :
  ∀ (a : ℝ), a^2 + 4 * a - 21 = (a - 3) * (a + 7) := by
  intro a
  sorry

end NUMINAMATH_GPT_factorization_eq_l1489_148955


namespace NUMINAMATH_GPT_original_acid_percentage_zero_l1489_148977

theorem original_acid_percentage_zero (a w : ℝ) 
  (h1 : (a + 1) / (a + w + 1) = 1 / 4) 
  (h2 : (a + 2) / (a + w + 2) = 2 / 5) : 
  a / (a + w) = 0 := 
by
  sorry

end NUMINAMATH_GPT_original_acid_percentage_zero_l1489_148977


namespace NUMINAMATH_GPT_three_times_two_to_the_n_minus_one_gt_n_squared_plus_three_l1489_148940

theorem three_times_two_to_the_n_minus_one_gt_n_squared_plus_three (n : ℕ) (h : n ≥ 4) : 3 * 2^(n-1) > n^2 + 3 := by
  sorry

end NUMINAMATH_GPT_three_times_two_to_the_n_minus_one_gt_n_squared_plus_three_l1489_148940


namespace NUMINAMATH_GPT_largest_n_divisibility_l1489_148942

theorem largest_n_divisibility (n : ℕ) (h : n + 12 ∣ n^3 + 144) : n ≤ 132 :=
  sorry

end NUMINAMATH_GPT_largest_n_divisibility_l1489_148942


namespace NUMINAMATH_GPT_parabola_line_slope_l1489_148925

theorem parabola_line_slope (y1 y2 x1 x2 : ℝ) (h1 : y1 ^ 2 = 6 * x1) (h2 : y2 ^ 2 = 6 * x2) 
    (midpoint_condition : (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = 2) :
  (y1 - y2) / (x1 - x2) = 3 / 2 :=
by
  -- here will be the actual proof using the given hypothesis
  sorry

end NUMINAMATH_GPT_parabola_line_slope_l1489_148925


namespace NUMINAMATH_GPT_bill_score_l1489_148979

variable {J B S : ℕ}

theorem bill_score (h1 : B = J + 20) (h2 : B = S / 2) (h3 : J + B + S = 160) : B = 45 :=
sorry

end NUMINAMATH_GPT_bill_score_l1489_148979


namespace NUMINAMATH_GPT_how_many_lassis_l1489_148972

def lassis_per_mango : ℕ := 15 / 3

def lassis15mangos : ℕ := 15

theorem how_many_lassis (H : lassis_per_mango = 5) : lassis15mangos * lassis_per_mango = 75 :=
by
  rw [H]
  sorry

end NUMINAMATH_GPT_how_many_lassis_l1489_148972


namespace NUMINAMATH_GPT_greatest_number_of_rented_trucks_l1489_148975

-- Define the conditions
def total_trucks_on_monday : ℕ := 24
def trucks_returned_percentage : ℕ := 50
def trucks_on_lot_saturday (R : ℕ) (P : ℕ) : ℕ := (R * P) / 100
def min_trucks_on_lot_saturday : ℕ := 12

-- Define the theorem
theorem greatest_number_of_rented_trucks : ∃ R, R = total_trucks_on_monday ∧ trucks_returned_percentage = 50 ∧ min_trucks_on_lot_saturday = 12 → R = 24 :=
by
  sorry

end NUMINAMATH_GPT_greatest_number_of_rented_trucks_l1489_148975


namespace NUMINAMATH_GPT_opposite_numbers_add_l1489_148919

theorem opposite_numbers_add : ∀ {a b : ℤ}, a + b = 0 → a + b + 3 = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_opposite_numbers_add_l1489_148919


namespace NUMINAMATH_GPT_cos_2theta_plus_pi_l1489_148971

-- Given condition
def tan_theta_eq_2 (θ : ℝ) : Prop := Real.tan θ = 2

-- The mathematical statement to prove
theorem cos_2theta_plus_pi (θ : ℝ) (h : tan_theta_eq_2 θ) : Real.cos (2 * θ + Real.pi) = 3 / 5 := 
sorry

end NUMINAMATH_GPT_cos_2theta_plus_pi_l1489_148971


namespace NUMINAMATH_GPT_ratio_AD_DC_l1489_148956

-- Definitions based on conditions
variable (A B C D : Point)
variable (AB BC AD DB : ℝ)
variable (h1 : AB = 2 * BC)
variable (h2 : AD = 3 / 5 * AB)
variable (h3 : DB = 2 / 5 * AB)

-- Lean statement for the problem
theorem ratio_AD_DC (h1 : AB = 2 * BC) (h2 : AD = 3 / 5 * AB) (h3 : DB = 2 / 5 * AB) :
  AD / (DB + BC) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_AD_DC_l1489_148956


namespace NUMINAMATH_GPT_none_of_the_choices_sum_of_150_consecutive_integers_l1489_148984

theorem none_of_the_choices_sum_of_150_consecutive_integers :
  ¬(∃ k : ℕ, 678900 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1136850 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1000000 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 2251200 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1876800 = 150 * k + 11325) :=
by
  sorry

end NUMINAMATH_GPT_none_of_the_choices_sum_of_150_consecutive_integers_l1489_148984


namespace NUMINAMATH_GPT_vip_seat_cost_l1489_148968

theorem vip_seat_cost
  (V : ℝ)
  (G V_T : ℕ)
  (h1 : 20 * G + V * V_T = 7500)
  (h2 : G + V_T = 320)
  (h3 : V_T = G - 276) :
  V = 70 := by
sorry

end NUMINAMATH_GPT_vip_seat_cost_l1489_148968


namespace NUMINAMATH_GPT_problem_solution_l1489_148963

noncomputable def problem_expr : ℝ :=
  (64 + 5 * 12) / (180 / 3) + Real.sqrt 49 - 2^3 * Nat.factorial 4

theorem problem_solution : problem_expr = -182.93333333 :=
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l1489_148963


namespace NUMINAMATH_GPT_number_of_right_triangles_with_hypotenuse_is_12_l1489_148941

theorem number_of_right_triangles_with_hypotenuse_is_12 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b : ℕ), 
     (b < 150) →
     (a^2 + b^2 = (b + 2)^2) →
     ∃ (k : ℕ), a = 2 * k ∧ k^2 = b + 1) := 
  sorry

end NUMINAMATH_GPT_number_of_right_triangles_with_hypotenuse_is_12_l1489_148941


namespace NUMINAMATH_GPT_find_c_l1489_148987

theorem find_c (c : ℝ) 
    (h : ∀ x, (x - 4) ∣ (c * x^3 + 16 * x^2 - 5 * c * x + 40)) : 
    c = -74 / 11 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1489_148987


namespace NUMINAMATH_GPT_find_m_of_symmetry_l1489_148905

-- Define the conditions for the parabola and the axis of symmetry
theorem find_m_of_symmetry (m : ℝ) :
  let a := (1 : ℝ)
  let b := (m - 2 : ℝ)
  let axis_of_symmetry := (0 : ℝ)
  (-b / (2 * a)) = axis_of_symmetry → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_of_symmetry_l1489_148905


namespace NUMINAMATH_GPT_train_speed_l1489_148988

theorem train_speed
    (train_length : ℕ := 800)
    (tunnel_length : ℕ := 500)
    (time_minutes : ℕ := 1)
    : (train_length + tunnel_length) * (60 / time_minutes) / 1000 = 78 := by
  sorry

end NUMINAMATH_GPT_train_speed_l1489_148988


namespace NUMINAMATH_GPT_first_ant_arrives_first_l1489_148993

noncomputable def time_crawling (d v : ℝ) : ℝ := d / v

noncomputable def time_riding_caterpillar (d v : ℝ) : ℝ := (d / 2) / (v / 2)

noncomputable def time_riding_grasshopper (d v : ℝ) : ℝ := (d / 2) / (10 * v)

noncomputable def time_ant1 (d v : ℝ) : ℝ := time_crawling d v

noncomputable def time_ant2 (d v : ℝ) : ℝ := time_riding_caterpillar d v + time_riding_grasshopper d v

theorem first_ant_arrives_first (d v : ℝ) (h_v_pos : 0 < v): time_ant1 d v < time_ant2 d v := by
  -- provide the justification for the theorem here
  sorry

end NUMINAMATH_GPT_first_ant_arrives_first_l1489_148993


namespace NUMINAMATH_GPT_time_for_c_l1489_148926

theorem time_for_c (a b work_completion: ℝ) (ha : a = 16) (hb : b = 6) (habc : work_completion = 3.2) : 
  (12 : ℝ) = 
  (48 * work_completion - 48) / 4 := 
sorry

end NUMINAMATH_GPT_time_for_c_l1489_148926


namespace NUMINAMATH_GPT_probability_of_x_gt_5y_l1489_148943

theorem probability_of_x_gt_5y :
  let rectangle := {(x, y) | 0 ≤ x ∧ x ≤ 3000 ∧ 0 ≤ y ∧ y ≤ 2500}
  let area_of_rectangle := 3000 * 2500
  let triangle := {(x, y) | 0 ≤ x ∧ x ≤ 3000 ∧ 0 ≤ y ∧ y < x / 5}
  let area_of_triangle := (3000 * 600) / 2
  ∃ prob : ℚ, (area_of_triangle / area_of_rectangle = prob) ∧ prob = 3 / 25 := by
  sorry

end NUMINAMATH_GPT_probability_of_x_gt_5y_l1489_148943


namespace NUMINAMATH_GPT_find_x2_plus_y2_l1489_148982

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 99) : 
  x^2 + y^2 = 5745 / 169 := 
sorry

end NUMINAMATH_GPT_find_x2_plus_y2_l1489_148982


namespace NUMINAMATH_GPT_avg_weight_increase_l1489_148923

theorem avg_weight_increase (A : ℝ) (X : ℝ) (hp1 : 8 * A - 65 + 105 = 8 * A + 40)
  (hp2 : 8 * (A + X) = 8 * A + 40) : X = 5 := 
by sorry

end NUMINAMATH_GPT_avg_weight_increase_l1489_148923


namespace NUMINAMATH_GPT_problem_l1489_148960

theorem problem 
  {a1 a2 : ℝ}
  (h1 : 0 ≤ a1)
  (h2 : 0 ≤ a2)
  (h3 : a1 + a2 = 1) :
  ∃ (b1 b2 : ℝ), 0 ≤ b1 ∧ 0 ≤ b2 ∧ b1 + b2 = 1 ∧ ((5/4 - a1) * b1 + 3 * (5/4 - a2) * b2 > 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_l1489_148960


namespace NUMINAMATH_GPT_A_elements_l1489_148939

open Set -- Open the Set namespace for easy access to set operations

def A : Set ℕ := {x | ∃ (n : ℕ), 12 = n * (6 - x)}

theorem A_elements : A = {0, 2, 3, 4, 5} :=
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_A_elements_l1489_148939


namespace NUMINAMATH_GPT_ratio_of_perimeters_of_squares_l1489_148944

theorem ratio_of_perimeters_of_squares (A B : ℝ) (h: A / B = 16 / 25) : ∃ (P1 P2 : ℝ), P1 / P2 = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_perimeters_of_squares_l1489_148944


namespace NUMINAMATH_GPT_ellipse_equation_l1489_148966

theorem ellipse_equation 
  (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m ≠ n)
  (h4 : ∀ A B : ℝ × ℝ, (m * A.1^2 + n * A.2^2 = 1) ∧ (m * B.1^2 + n * B.2^2 = 1) ∧ (A.1 + A.2 = 1) ∧ (B.1 + B.2 = 1) → dist A B = 2 * (2:ℝ).sqrt)
  (h5 : ∀ A B : ℝ × ℝ, (m * A.1^2 + n * A.2^2 = 1) ∧ (m * B.1^2 + n * B.2^2 = 1) ∧ (A.1 + A.2 = 1) ∧ (B.1 + B.2 = 1) → 
    (A.2 + B.2) / (A.1 + B.1) = (2:ℝ).sqrt / 2) :
  m = 1 / 3 → n = (2:ℝ).sqrt / 3 → 
  (∀ x y : ℝ, (1 / 3) * x^2 + ((2:ℝ).sqrt / 3) * y^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_l1489_148966


namespace NUMINAMATH_GPT_judgement_only_b_correct_l1489_148973

theorem judgement_only_b_correct
  (A_expr : Int := 11 + (-14) + 19 - (-6))
  (A_computed : Int := 11 + 19 + ((-14) + (-6)))
  (A_result_incorrect : A_computed ≠ 10)
  (B_expr : ℚ := -2/3 - 1/5 + (-1/3))
  (B_computed : ℚ := (-2/3 + -1/3) + -1/5)
  (B_result_correct : B_computed = -6/5) :
  (A_computed ≠ 10 ∧ B_computed = -6/5) :=
by
  sorry

end NUMINAMATH_GPT_judgement_only_b_correct_l1489_148973


namespace NUMINAMATH_GPT_decrypt_probability_l1489_148981

theorem decrypt_probability (p1 p2 p3 : ℚ) (h1 : p1 = 1/5) (h2 : p2 = 2/5) (h3 : p3 = 1/2) : 
  1 - ((1 - p1) * (1 - p2) * (1 - p3)) = 19/25 :=
by
  sorry

end NUMINAMATH_GPT_decrypt_probability_l1489_148981


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1489_148964

-- Problem (I)
theorem problem1 (x : ℝ) (hx : x > 1) : 2 * Real.log x < x - 1/x :=
sorry

-- Problem (II)
theorem problem2 (a : ℝ) : (∀ t : ℝ, t > 0 → (1 + a / t) * Real.log (1 + t) > a) → 0 < a ∧ a ≤ 2 :=
sorry

-- Problem (III)
theorem problem3 : (9/10 : ℝ)^19 < 1 / (Real.exp 2) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1489_148964


namespace NUMINAMATH_GPT_max_value_of_P_l1489_148986

noncomputable def P (a b c : ℝ) : ℝ :=
  (2 / (a^2 + 1)) - (2 / (b^2 + 1)) + (3 / (c^2 + 1))

theorem max_value_of_P (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c + a + c = b) :
  ∃ x, x = 1 ∧ ∀ y, (y = P a b c) → y ≤ x :=
sorry

end NUMINAMATH_GPT_max_value_of_P_l1489_148986


namespace NUMINAMATH_GPT_shaded_rectangle_area_l1489_148908

-- Define the square PQRS and its properties
def is_square (s : ℝ) := ∃ (PQ QR RS SP : ℝ), PQ = s ∧ QR = s ∧ RS = s ∧ SP = s

-- Define the conditions for the side lengths and segments
def side_length := 11
def top_left_height := 6
def top_right_height := 2
def width_bottom_right := 11 - 10
def width_top_right := 8

-- Calculate necessary dimensions
def shaded_rectangle_height := top_left_height - top_right_height
def shaded_rectangle_width := width_top_right - width_bottom_right

-- Proof statement
theorem shaded_rectangle_area (s : ℝ) (h1 : is_square s)
  (h2 : s = side_length)
  (h3 : shaded_rectangle_height = 4)
  (h4 : shaded_rectangle_width = 7) :
  4 * 7 = 28 := by
  sorry

end NUMINAMATH_GPT_shaded_rectangle_area_l1489_148908


namespace NUMINAMATH_GPT_sum_of_y_values_l1489_148995

def g (x : ℚ) : ℚ := 2 * x^2 - x + 3

theorem sum_of_y_values (y1 y2 : ℚ) (hy : g (4 * y1) = 10 ∧ g (4 * y2) = 10) :
  y1 + y2 = 1 / 16 :=
sorry

end NUMINAMATH_GPT_sum_of_y_values_l1489_148995


namespace NUMINAMATH_GPT_football_cost_correct_l1489_148914

variable (total_spent_on_toys : ℝ := 12.30)
variable (spent_on_marbles : ℝ := 6.59)

theorem football_cost_correct :
  (total_spent_on_toys - spent_on_marbles = 5.71) :=
by
  sorry

end NUMINAMATH_GPT_football_cost_correct_l1489_148914


namespace NUMINAMATH_GPT_min_value_arithmetic_sequence_l1489_148912

theorem min_value_arithmetic_sequence (d : ℝ) (n : ℕ) (hd : d ≠ 0) (a1 : ℝ) (ha1 : a1 = 1)
(geo : (1 + 2 * d)^2 = 1 + 12 * d) (Sn : ℝ) (hSn : Sn = n^2) (an : ℝ) (han : an = 2 * n - 1) :
  ∀ (n : ℕ), n > 0 → (2 * Sn + 8) / (an + 3) ≥ 5 / 2 :=
by sorry

end NUMINAMATH_GPT_min_value_arithmetic_sequence_l1489_148912


namespace NUMINAMATH_GPT_average_score_of_class_l1489_148937

-- Definitions based on the conditions
def class_size : ℕ := 20
def group1_size : ℕ := 10
def group2_size : ℕ := 10
def group1_avg_score : ℕ := 80
def group2_avg_score : ℕ := 60

-- Average score of the whole class
theorem average_score_of_class : 
  (group1_size * group1_avg_score + group2_size * group2_avg_score) / class_size = 70 := 
by sorry

end NUMINAMATH_GPT_average_score_of_class_l1489_148937


namespace NUMINAMATH_GPT_min_distance_squared_l1489_148927

noncomputable def graph_function1 (x : ℝ) : ℝ := -x^2 + 3 * Real.log x

noncomputable def point_on_graph1 (a b : ℝ) : Prop := b = graph_function1 a

noncomputable def graph_function2 (x : ℝ) : ℝ := x + 2

noncomputable def point_on_graph2 (c d : ℝ) : Prop := d = graph_function2 c

theorem min_distance_squared (a b c d : ℝ) 
  (hP : point_on_graph1 a b)
  (hQ : point_on_graph2 c d) :
  (a - c)^2 + (b - d)^2 = 8 := 
sorry

end NUMINAMATH_GPT_min_distance_squared_l1489_148927


namespace NUMINAMATH_GPT_min_value_1_a_plus_2_b_l1489_148996

open Real

theorem min_value_1_a_plus_2_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (∀ a b, 0 < a → 0 < b → a + b = 1 → 3 + 2 * sqrt 2 ≤ 1 / a + 2 / b) := sorry

end NUMINAMATH_GPT_min_value_1_a_plus_2_b_l1489_148996


namespace NUMINAMATH_GPT_xiaohua_distance_rounds_l1489_148932

def length := 5
def width := 3
def perimeter (a b : ℕ) := (a + b) * 2
def total_distance (perimeter : ℕ) (laps : ℕ) := perimeter * laps

theorem xiaohua_distance_rounds :
  total_distance (perimeter length width) 3 = 30 :=
by sorry

end NUMINAMATH_GPT_xiaohua_distance_rounds_l1489_148932


namespace NUMINAMATH_GPT_probability_at_most_one_incorrect_l1489_148985

variable (p : ℝ)

theorem probability_at_most_one_incorrect (h : 0 ≤ p ∧ p ≤ 1) :
  p^9 * (10 - 9*p) = p^10 + 10 * (1 - p) * p^9 := by
  sorry

end NUMINAMATH_GPT_probability_at_most_one_incorrect_l1489_148985


namespace NUMINAMATH_GPT_double_transmission_yellow_twice_double_transmission_less_single_l1489_148954

variables {α : ℝ} (hα : 0 < α ∧ α < 1)

-- Statement B
theorem double_transmission_yellow_twice (hα : 0 < α ∧ α < 1) :
  probability_displays_yellow_twice = α^2 :=
sorry

-- Statement D
theorem double_transmission_less_single (hα : 0 < α ∧ α < 1) :
  (1 - α)^2 < (1 - α) :=
sorry

end NUMINAMATH_GPT_double_transmission_yellow_twice_double_transmission_less_single_l1489_148954


namespace NUMINAMATH_GPT_milk_price_same_after_reductions_l1489_148962

theorem milk_price_same_after_reductions (x : ℝ) (h1 : 0 < x) :
  (x - 0.4 * x) = ((x - 0.2 * x) - 0.25 * (x - 0.2 * x)) :=
by
  sorry

end NUMINAMATH_GPT_milk_price_same_after_reductions_l1489_148962


namespace NUMINAMATH_GPT_nat_number_36_sum_of_digits_l1489_148901

-- Define the function that represents the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main statement
theorem nat_number_36_sum_of_digits (x : ℕ) (hx : x = 36 * sum_of_digits x) : x = 324 ∨ x = 648 := 
by 
  sorry

end NUMINAMATH_GPT_nat_number_36_sum_of_digits_l1489_148901


namespace NUMINAMATH_GPT_soldier_initial_consumption_l1489_148918

theorem soldier_initial_consumption :
  ∀ (s d1 n : ℕ) (c2 d2 : ℝ), 
    s = 1200 → d1 = 30 → n = 528 → c2 = 2.5 → d2 = 25 → 
    36000 * (x : ℝ) = 108000 → x = 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_soldier_initial_consumption_l1489_148918


namespace NUMINAMATH_GPT_third_number_in_sequence_l1489_148992

theorem third_number_in_sequence (n : ℕ) (h_sum : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 63) : n + 2 = 8 :=
by
  -- the proof would be written here
  sorry

end NUMINAMATH_GPT_third_number_in_sequence_l1489_148992


namespace NUMINAMATH_GPT_cyclic_quadrilateral_AD_correct_l1489_148951

noncomputable def cyclic_quadrilateral_AD_length : ℝ :=
  let R := 200 * Real.sqrt 2
  let AB := 200
  let BC := 200
  let CD := 200
  let AD := 500
  sorry

theorem cyclic_quadrilateral_AD_correct (R AB BC CD AD : ℝ) (hR : R = 200 * Real.sqrt 2) 
  (hAB : AB = 200) (hBC : BC = 200) (hCD : CD = 200) : AD = 500 :=
by
  have hRABBCDC: R = 200 * Real.sqrt 2 ∧ AB = 200 ∧ BC = 200 ∧ CD = 200 := ⟨hR, hAB, hBC, hCD⟩
  sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_AD_correct_l1489_148951
