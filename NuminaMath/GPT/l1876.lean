import Mathlib

namespace NUMINAMATH_GPT_crackers_per_person_l1876_187628

theorem crackers_per_person:
  ∀ (total_crackers friends : ℕ), total_crackers = 36 → friends = 18 → total_crackers / friends = 2 :=
by
  intros total_crackers friends h1 h2
  sorry

end NUMINAMATH_GPT_crackers_per_person_l1876_187628


namespace NUMINAMATH_GPT_books_read_l1876_187661

-- Given conditions
def chapters_per_book : ℕ := 17
def total_chapters_read : ℕ := 68

-- Statement to prove
theorem books_read : (total_chapters_read / chapters_per_book) = 4 := 
by sorry

end NUMINAMATH_GPT_books_read_l1876_187661


namespace NUMINAMATH_GPT_equal_piles_l1876_187611

theorem equal_piles (initial_rocks final_piles : ℕ) (moves : ℕ) (total_rocks : ℕ) (rocks_per_pile : ℕ) :
  initial_rocks = 36 →
  final_piles = 7 →
  moves = final_piles - 1 →
  total_rocks = initial_rocks + moves →
  rocks_per_pile = total_rocks / final_piles →
  rocks_per_pile = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_equal_piles_l1876_187611


namespace NUMINAMATH_GPT_candy_store_spending_l1876_187650

variable (weekly_allowance : ℝ) (arcade_fraction : ℝ) (toy_store_fraction : ℝ)

def remaining_after_arcade (weekly_allowance arcade_fraction : ℝ) : ℝ :=
  weekly_allowance * (1 - arcade_fraction)

def remaining_after_toy_store (remaining_allowance toy_store_fraction : ℝ) : ℝ :=
  remaining_allowance * (1 - toy_store_fraction)

theorem candy_store_spending
  (h1 : weekly_allowance = 3.30)
  (h2 : arcade_fraction = 3 / 5)
  (h3 : toy_store_fraction = 1 / 3) :
  remaining_after_toy_store (remaining_after_arcade weekly_allowance arcade_fraction) toy_store_fraction = 0.88 := 
sorry

end NUMINAMATH_GPT_candy_store_spending_l1876_187650


namespace NUMINAMATH_GPT_sum_of_coefficients_l1876_187666

theorem sum_of_coefficients (s : ℕ → ℝ) (a b c : ℝ) : 
  s 0 = 3 ∧ s 1 = 7 ∧ s 2 = 17 ∧ 
  (∀ k ≥ 2, s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)) → 
  a + b + c = 12 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1876_187666


namespace NUMINAMATH_GPT_total_travel_time_in_minutes_l1876_187643

def riding_rate : ℝ := 10 -- 10 miles per hour
def initial_riding_time : ℝ := 30 -- 30 minutes
def another_riding_distance : ℝ := 15 -- 15 miles
def resting_time : ℝ := 30 -- 30 minutes
def remaining_distance : ℝ := 20 -- 20 miles

theorem total_travel_time_in_minutes :
  initial_riding_time +
  (another_riding_distance / riding_rate * 60) +
  resting_time +
  (remaining_distance / riding_rate * 60) = 270 :=
by
  sorry

end NUMINAMATH_GPT_total_travel_time_in_minutes_l1876_187643


namespace NUMINAMATH_GPT_reeya_average_score_l1876_187659

theorem reeya_average_score :
  let scores := [50, 60, 70, 80, 80]
  let sum_scores := scores.sum
  let num_scores := scores.length
  sum_scores / num_scores = 68 :=
by
  sorry

end NUMINAMATH_GPT_reeya_average_score_l1876_187659


namespace NUMINAMATH_GPT_power_multiplication_result_l1876_187670

theorem power_multiplication_result :
  ( (8 / 9)^3 * (1 / 3)^3 * (2 / 5)^3 = (4096 / 2460375) ) :=
by
  sorry

end NUMINAMATH_GPT_power_multiplication_result_l1876_187670


namespace NUMINAMATH_GPT_increasing_function_of_a_l1876_187635

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (2 - (a / 2)) * x + 2

theorem increasing_function_of_a (a : ℝ) : (∀ x y, x < y → f a x ≤ f a y) ↔ 
  (8 / 3 ≤ a ∧ a < 4) :=
sorry

end NUMINAMATH_GPT_increasing_function_of_a_l1876_187635


namespace NUMINAMATH_GPT_fraction_division_correct_l1876_187604

theorem fraction_division_correct :
  (2 / 5) / 3 = 2 / 15 :=
by sorry

end NUMINAMATH_GPT_fraction_division_correct_l1876_187604


namespace NUMINAMATH_GPT_find_number_l1876_187649

variable (x : ℝ)

theorem find_number 
  (h1 : 0.20 * x + 0.25 * 60 = 23) :
  x = 40 :=
sorry

end NUMINAMATH_GPT_find_number_l1876_187649


namespace NUMINAMATH_GPT_dog_biscuit_cost_l1876_187637

open Real

theorem dog_biscuit_cost :
  (∀ (x : ℝ),
    (4 * x + 2) * 7 = 21 →
    x = 1 / 4) :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_dog_biscuit_cost_l1876_187637


namespace NUMINAMATH_GPT_value_of_c_l1876_187644

theorem value_of_c (a b c d w x y z : ℕ) (primes : ∀ p ∈ [w, x, y, z], Prime p)
  (h1 : w < x) (h2 : x < y) (h3 : y < z) 
  (h4 : (w^a) * (x^b) * (y^c) * (z^d) = 660) 
  (h5 : (a + b) - (c + d) = 1) : c = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_c_l1876_187644


namespace NUMINAMATH_GPT_smallest_x_solution_l1876_187679

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end NUMINAMATH_GPT_smallest_x_solution_l1876_187679


namespace NUMINAMATH_GPT_log_expression_value_l1876_187653

theorem log_expression_value : 
  (Real.logb 10 (Real.sqrt 2) + Real.logb 10 (Real.sqrt 5) + 2 ^ 0 + (5 ^ (1 / 3)) ^ 2 * Real.sqrt 5 = 13 / 2) := 
by 
  -- The proof is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_log_expression_value_l1876_187653


namespace NUMINAMATH_GPT_Rachel_father_age_when_Rachel_is_25_l1876_187662

-- Define the problem conditions:
def Rachel_age : ℕ := 12
def Grandfather_age : ℕ := 7 * Rachel_age
def Mother_age : ℕ := Grandfather_age / 2
def Father_age : ℕ := Mother_age + 5

-- Prove the age of Rachel's father when she is 25 years old:
theorem Rachel_father_age_when_Rachel_is_25 : 
  Father_age + (25 - Rachel_age) = 60 := by
    sorry

end NUMINAMATH_GPT_Rachel_father_age_when_Rachel_is_25_l1876_187662


namespace NUMINAMATH_GPT_four_digit_numbers_starting_with_1_l1876_187610

theorem four_digit_numbers_starting_with_1 
: ∃ n : ℕ, (n = 234) ∧ 
  (∀ (x y z : ℕ), 
    (x ≠ y → x ≠ z → y ≠ z → -- ensuring these constraints
    x ≠ 1 → y ≠ 1 → z = 1 → -- exactly two identical digits which include 1
    (x * 1000 + y * 100 + z * 10 + 1) / 1000 = 1 ∨ (x * 1000 + z * 100 + y * 10 + 1) / 1000 = 1) ∨ 
    (∃ (x y : ℕ),  
    (x ≠ y → x ≠ 1 → y = 1 → 
    (x * 110 + y * 10 + 1) + (x * 11 + y * 10 + 1) + (x * 100 + y * 10 + 1) + (x * 110 + 1) = n))) := sorry

end NUMINAMATH_GPT_four_digit_numbers_starting_with_1_l1876_187610


namespace NUMINAMATH_GPT_matrix_vector_combination_l1876_187689

variables {α : Type*} [AddCommGroup α] [Module ℝ α]
variables (M : α →ₗ[ℝ] ℝ × ℝ)
variables (u v w : α)
variables (h1 : M u = (-3, 4))
variables (h2 : M v = (2, -7))
variables (h3 : M w = (9, 0))

theorem matrix_vector_combination :
  M (3 • u - 4 • v + 2 • w) = (1, 40) :=
by sorry

end NUMINAMATH_GPT_matrix_vector_combination_l1876_187689


namespace NUMINAMATH_GPT_yearly_feeding_cost_l1876_187686

-- Defining the conditions
def num_geckos := 3
def num_iguanas := 2
def num_snakes := 4

def cost_per_snake_per_month := 10
def cost_per_iguana_per_month := 5
def cost_per_gecko_per_month := 15

-- Statement of the proof problem
theorem yearly_feeding_cost : 
  (num_snakes * cost_per_snake_per_month + num_iguanas * cost_per_iguana_per_month + num_geckos * cost_per_gecko_per_month) * 12 = 1140 := 
  by 
    sorry

end NUMINAMATH_GPT_yearly_feeding_cost_l1876_187686


namespace NUMINAMATH_GPT_intersection_point_l1876_187682

theorem intersection_point :
  (∃ (x y : ℝ), 5 * x - 3 * y = 15 ∧ 4 * x + 2 * y = 14)
  → (∃ (x y : ℝ), x = 3 ∧ y = 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_intersection_point_l1876_187682


namespace NUMINAMATH_GPT_B_can_complete_alone_l1876_187692

-- Define the given conditions
def A_work_rate := 1 / 20
def total_days := 21
def A_quit_days := 15
def B_completion_days := 30

-- Define the problem statement in Lean
theorem B_can_complete_alone (x : ℝ) (h₁ : A_work_rate = 1 / 20) (h₂ : total_days = 21)
  (h₃ : A_quit_days = 15) (h₄ : (21 - A_quit_days) * (1 / 20 + 1 / x) + A_quit_days * (1 / x) = 1) :
  x = B_completion_days :=
  sorry

end NUMINAMATH_GPT_B_can_complete_alone_l1876_187692


namespace NUMINAMATH_GPT_integral_solutions_l1876_187601

theorem integral_solutions (a b c : ℤ) (h : a^2 + b^2 + c^2 = a^2 * b^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end NUMINAMATH_GPT_integral_solutions_l1876_187601


namespace NUMINAMATH_GPT_montana_more_than_ohio_l1876_187639

-- Define the total number of combinations for Ohio and Montana
def ohio_combinations : ℕ := 26^4 * 10^3
def montana_combinations : ℕ := 26^5 * 10^2

-- The total number of combinations from both states
def ohio_total : ℕ := ohio_combinations
def montana_total : ℕ := montana_combinations

-- Prove the difference
theorem montana_more_than_ohio : montana_total - ohio_total = 731161600 := by
  sorry

end NUMINAMATH_GPT_montana_more_than_ohio_l1876_187639


namespace NUMINAMATH_GPT_chords_and_circle_l1876_187605

theorem chords_and_circle (R : ℝ) (A B C D : ℝ) 
  (hAB : 0 < A - B) (hCD : 0 < C - D) (hR : R > 0) 
  (h_perp : (A - B) * (C - D) = 0) 
  (h_radA : A ^ 2 + B ^ 2 = R ^ 2) 
  (h_radC : C ^ 2 + D ^ 2 = R ^ 2) :
  (A - C)^2 + (B - D)^2 = 4 * R^2 :=
by
  sorry

end NUMINAMATH_GPT_chords_and_circle_l1876_187605


namespace NUMINAMATH_GPT_fraction_ratio_l1876_187619

variable {α : Type*} [DivisionRing α] (a b : α)

theorem fraction_ratio (h1 : 2 * a = 3 * b) (h2 : b ≠ 0) : a / b = 3 / 2 := 
by sorry

end NUMINAMATH_GPT_fraction_ratio_l1876_187619


namespace NUMINAMATH_GPT_gcd_expression_multiple_of_456_l1876_187615

theorem gcd_expression_multiple_of_456 (a : ℤ) (h : ∃ k : ℤ, a = 456 * k) : 
  Int.gcd (3 * a^3 + a^2 + 4 * a + 57) a = 57 := by
  sorry

end NUMINAMATH_GPT_gcd_expression_multiple_of_456_l1876_187615


namespace NUMINAMATH_GPT_find_a_l1876_187694

def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_a (a : ℝ) : 
  are_perpendicular a (a + 2) → 
  a = -1 :=
by
  intro h
  unfold are_perpendicular at h
  have h_eq : a * (a + 2) = -1 := h
  have eq_zero : a * a + 2 * a + 1 = 0 := by linarith
  sorry

end NUMINAMATH_GPT_find_a_l1876_187694


namespace NUMINAMATH_GPT_number_of_males_who_listen_l1876_187633

theorem number_of_males_who_listen (females_listen : ℕ) (males_dont_listen : ℕ) (total_listen : ℕ) (total_dont_listen : ℕ) (total_females : ℕ) :
  females_listen = 72 →
  males_dont_listen = 88 →
  total_listen = 160 →
  total_dont_listen = 180 →
  (total_females = total_listen + total_dont_listen - (females_listen + males_dont_listen)) →
  (total_females + males_dont_listen + 92 = total_listen + total_dont_listen) →
  total_listen + total_dont_listen = females_listen + males_dont_listen + (total_females - females_listen) + 92 :=
sorry

end NUMINAMATH_GPT_number_of_males_who_listen_l1876_187633


namespace NUMINAMATH_GPT_problem1_problem2_real_problem2_complex_problem3_l1876_187652

-- Problem 1: Prove that if 2 ∈ A, then {-1, 1/2} ⊆ A
theorem problem1 (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : 2 ∈ A) : -1 ∈ A ∧ (1/2) ∈ A := sorry

-- Problem 2: Prove that A cannot be a singleton set for real numbers, but can for complex numbers.
theorem problem2_real (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : ∃ a ∈ A, a ≠ 1) : ¬(∃ a, A = {a}) := sorry

theorem problem2_complex (A : Set ℂ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : ∃ a ∈ A, a ≠ 1) : (∃ a, A = {a}) := sorry

-- Problem 3: Prove that 1 - 1/a ∈ A given a ∈ A
theorem problem3 (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (a : ℝ) (ha : a ∈ A) : (1 - 1/a) ∈ A := sorry

end NUMINAMATH_GPT_problem1_problem2_real_problem2_complex_problem3_l1876_187652


namespace NUMINAMATH_GPT_inequality_proof_l1876_187676

theorem inequality_proof (a b : Real) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1876_187676


namespace NUMINAMATH_GPT_number_of_members_l1876_187667

theorem number_of_members (n h : ℕ) (h1 : n * n * h = 362525) : n = 5 :=
sorry

end NUMINAMATH_GPT_number_of_members_l1876_187667


namespace NUMINAMATH_GPT_snow_on_second_day_l1876_187672

-- Definition of conditions as variables in Lean
def snow_on_first_day := 6 -- in inches
def snow_melted := 2 -- in inches
def additional_snow_fifth_day := 12 -- in inches
def total_snow := 24 -- in inches

-- The variable for snow on the second day
variable (x : ℕ)

-- Proof goal
theorem snow_on_second_day : snow_on_first_day + x - snow_melted + additional_snow_fifth_day = total_snow → x = 8 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_snow_on_second_day_l1876_187672


namespace NUMINAMATH_GPT_brick_height_l1876_187656

theorem brick_height (h : ℝ) : 
    let wall_length := 900
    let wall_width := 600
    let wall_height := 22.5
    let num_bricks := 7200
    let brick_length := 25
    let brick_width := 11.25
    wall_length * wall_width * wall_height = num_bricks * (brick_length * brick_width * h) -> 
    h = 67.5 := 
by
  intros
  sorry

end NUMINAMATH_GPT_brick_height_l1876_187656


namespace NUMINAMATH_GPT_older_brother_catches_up_l1876_187690

theorem older_brother_catches_up (D : ℝ) (t : ℝ) :
  let vy := D / 25
  let vo := D / 15
  let time := 20
  15 * time = 25 * (time - 8) → (15 * time = 25 * (time - 8) → t = 20)
:= by
  sorry

end NUMINAMATH_GPT_older_brother_catches_up_l1876_187690


namespace NUMINAMATH_GPT_jeff_total_distance_l1876_187618

-- Define the conditions as constants
def speed1 : ℝ := 80
def time1 : ℝ := 3

def speed2 : ℝ := 50
def time2 : ℝ := 2

def speed3 : ℝ := 70
def time3 : ℝ := 1

def speed4 : ℝ := 60
def time4 : ℝ := 2

def speed5 : ℝ := 45
def time5 : ℝ := 3

def speed6 : ℝ := 40
def time6 : ℝ := 2

def speed7 : ℝ := 30
def time7 : ℝ := 2.5

-- Define the equation for the total distance traveled
def total_distance : ℝ :=
  speed1 * time1 + 
  speed2 * time2 + 
  speed3 * time3 + 
  speed4 * time4 + 
  speed5 * time5 + 
  speed6 * time6 + 
  speed7 * time7

-- Prove that the total distance is equal to 820 miles
theorem jeff_total_distance : total_distance = 820 := by
  sorry

end NUMINAMATH_GPT_jeff_total_distance_l1876_187618


namespace NUMINAMATH_GPT_compute_abc_l1876_187640

theorem compute_abc (a b c : ℤ) (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h₁ : a + b + c = 30) 
  (h₂ : (1 : ℚ)/a + (1 : ℚ)/b + (1 : ℚ)/c + 300/(a * b * c) = 1) : a * b * c = 768 := 
by 
  sorry

end NUMINAMATH_GPT_compute_abc_l1876_187640


namespace NUMINAMATH_GPT_find_y_l1876_187669

open Real

theorem find_y : ∃ y : ℝ, (sqrt ((3 - (-5))^2 + (y - 4)^2) = 12) ∧ (y > 0) ∧ (y = 4 + 4 * sqrt 5) :=
by
  use 4 + 4 * sqrt 5
  -- The proof steps would go here.
  sorry

end NUMINAMATH_GPT_find_y_l1876_187669


namespace NUMINAMATH_GPT_fraction_simplification_l1876_187621

theorem fraction_simplification :
  ∃ (p q : ℕ), p = 2021 ∧ q ≠ 0 ∧ gcd p q = 1 ∧ (1011 / 1010) - (1010 / 1011) = (p : ℚ) / q := 
sorry

end NUMINAMATH_GPT_fraction_simplification_l1876_187621


namespace NUMINAMATH_GPT_target_run_correct_l1876_187681

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_10 : ℝ := 10
def run_rate_remaining_22_overs : ℝ := 11.363636363636363
def overs_remaining_22 : ℝ := 22

-- Initialize the target run calculation using the given conditions
def runs_first_10_overs := overs_first_10 * run_rate_first_10_overs
def runs_remaining_22_overs := overs_remaining_22 * run_rate_remaining_22_overs
def target_run := runs_first_10_overs + runs_remaining_22_overs 

-- The goal is to prove that the target run is 282
theorem target_run_correct : target_run = 282 := by
  sorry  -- The proof is not required as per the instructions.

end NUMINAMATH_GPT_target_run_correct_l1876_187681


namespace NUMINAMATH_GPT_pieces_of_fudge_l1876_187675

def pan_length : ℝ := 27.5
def pan_width : ℝ := 17.5
def pan_height : ℝ := 2.5
def cube_side : ℝ := 2.3

def volume (l w h : ℝ) : ℝ := l * w * h

def V_pan : ℝ := volume pan_length pan_width pan_height
def V_cube : ℝ := volume cube_side cube_side cube_side

theorem pieces_of_fudge : ⌊V_pan / V_cube⌋ = 98 := by
  -- calculation can be filled in here in the actual proof
  sorry

end NUMINAMATH_GPT_pieces_of_fudge_l1876_187675


namespace NUMINAMATH_GPT_four_digits_sum_l1876_187634

theorem four_digits_sum (A B C D : ℕ) 
  (A_neq_B : A ≠ B) (A_neq_C : A ≠ C) (A_neq_D : A ≠ D) 
  (B_neq_C : B ≠ C) (B_neq_D : B ≠ D) 
  (C_neq_D : C ≠ D)
  (digits_A : A ≤ 9) (digits_B : B ≤ 9) (digits_C : C ≤ 9) (digits_D : D ≤ 9)
  (A_lt_B : A < B) 
  (minimize_fraction : ∃ k : ℕ, (A + B) = k ∧ k ≤ (A + B) ∧ (C + D) ≥ (C + D)) :
  C + D = 17 := 
by
  sorry

end NUMINAMATH_GPT_four_digits_sum_l1876_187634


namespace NUMINAMATH_GPT_boxes_containing_neither_l1876_187696

theorem boxes_containing_neither
  (total_boxes : ℕ := 15)
  (boxes_with_markers : ℕ := 9)
  (boxes_with_crayons : ℕ := 5)
  (boxes_with_both : ℕ := 4) :
  (total_boxes - ((boxes_with_markers - boxes_with_both) + (boxes_with_crayons - boxes_with_both) + boxes_with_both)) = 5 := by
  sorry

end NUMINAMATH_GPT_boxes_containing_neither_l1876_187696


namespace NUMINAMATH_GPT_problem1_problem2_l1876_187641

theorem problem1 : -3 + (-2) * 5 - (-3) = -10 :=
by
  sorry

theorem problem2 : -1^4 + ((-5)^2 - 3) / |(-2)| = 10 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1876_187641


namespace NUMINAMATH_GPT_intersection_eq_l1876_187632

def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6 * x + 8 < 0}

theorem intersection_eq : M ∩ N = {x | 2 < x ∧ x < 3} := 
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1876_187632


namespace NUMINAMATH_GPT_cost_of_snake_toy_l1876_187614

-- Given conditions
def cost_of_cage : ℝ := 14.54
def dollar_bill_found : ℝ := 1.00
def total_cost : ℝ := 26.30

-- Theorem to find the cost of the snake toy
theorem cost_of_snake_toy : 
  (total_cost + dollar_bill_found - cost_of_cage) = 12.76 := 
  by sorry

end NUMINAMATH_GPT_cost_of_snake_toy_l1876_187614


namespace NUMINAMATH_GPT_range_of_a_l1876_187658

theorem range_of_a (x a : ℝ) :
  (∀ x : ℝ, x - 1 < 0 ∧ x < a + 3 → x < 1) → a ≥ -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1876_187658


namespace NUMINAMATH_GPT_parallel_lines_l1876_187616

theorem parallel_lines (a : ℝ) (h : ∀ x y : ℝ, 2*x - a*y - 1 = 0 → a*x - y = 0) : a = Real.sqrt 2 ∨ a = -Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_parallel_lines_l1876_187616


namespace NUMINAMATH_GPT_James_comics_l1876_187668

theorem James_comics (days_in_year : ℕ) (years : ℕ) (writes_every_other_day : ℕ) (no_leap_years : ℕ) 
  (h1 : days_in_year = 365) (h2 : years = 4) (h3 : writes_every_other_day = 2) : 
  (days_in_year * years) / writes_every_other_day = 730 := 
by
  sorry

end NUMINAMATH_GPT_James_comics_l1876_187668


namespace NUMINAMATH_GPT_range_of_m_l1876_187630

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.exp (2 * x)
noncomputable def g (m x : ℝ) : ℝ := m * x + 1

def exists_x0 (x1 : ℝ) (m : ℝ) : Prop :=
  ∃ (x0 : ℝ), -1 ≤ x0 ∧ x0 ≤ 1 ∧ g m x0 = f x1

theorem range_of_m (m : ℝ) (cond : ∀ (x1 : ℝ), -1 ≤ x1 → x1 ≤ 1 → exists_x0 x1 m) :
  m ∈ Set.Iic (1 - Real.exp 2) ∨ m ∈ Set.Ici (Real.exp 2 - 1) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1876_187630


namespace NUMINAMATH_GPT_range_of_a_l1876_187660

theorem range_of_a (x y a : ℝ) (h1 : 3 * x + y = a + 1) (h2 : x + 3 * y = 3) (h3 : x + y > 5) : a > 16 := 
sorry 

end NUMINAMATH_GPT_range_of_a_l1876_187660


namespace NUMINAMATH_GPT_maximize_profit_l1876_187631

-- Define constants for purchase and selling prices
def priceA_purchase : ℝ := 16
def priceA_selling : ℝ := 20
def priceB_purchase : ℝ := 20
def priceB_selling : ℝ := 25

-- Define constant for total weight
def total_weight : ℝ := 200

-- Define profit function
def profit (weightA weightB : ℝ) : ℝ :=
  (priceA_selling - priceA_purchase) * weightA + (priceB_selling - priceB_purchase) * weightB

-- Define constraints
def constraint1 (weightA weightB : ℝ) : Prop :=
  weightA + weightB = total_weight

def constraint2 (weightA weightB : ℝ) : Prop :=
  weightA >= 3 * weightB

open Real

-- Define the maximum profit we aim to prove
def max_profit : ℝ := 850

-- The main theorem to prove
theorem maximize_profit : 
  ∃ weightA weightB : ℝ, constraint1 weightA weightB ∧ constraint2 weightA weightB ∧ profit weightA weightB = max_profit :=
by {
  sorry
}

end NUMINAMATH_GPT_maximize_profit_l1876_187631


namespace NUMINAMATH_GPT_expression_non_negative_l1876_187693

theorem expression_non_negative (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b)) + (1 / (b - c)) + (4 / (c - a)) ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_expression_non_negative_l1876_187693


namespace NUMINAMATH_GPT_monotonic_intervals_extremum_values_l1876_187651

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 8

theorem monotonic_intervals :
  (∀ x, x < -1 → deriv f x > 0) ∧
  (∀ x, x > 2 → deriv f x > 0) ∧
  (∀ x, -1 < x ∧ x < 2 → deriv f x < 0) := sorry

theorem extremum_values :
  ∃ a b : ℝ, (a = -12) ∧ (b = 15) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 3 → f x ≥ b → f x = b) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 3 → f x ≤ a → f x = a) := sorry

end NUMINAMATH_GPT_monotonic_intervals_extremum_values_l1876_187651


namespace NUMINAMATH_GPT_find_g_2_l1876_187609

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_2
  (H : ∀ (x : ℝ), x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x ^ 2):
  g 2 = 67 / 14 :=
by
  sorry

end NUMINAMATH_GPT_find_g_2_l1876_187609


namespace NUMINAMATH_GPT_car_distance_ratio_l1876_187687

theorem car_distance_ratio (t : ℝ) (h₁ : t > 0)
    (speed_A speed_B : ℝ)
    (h₂ : speed_A = 70)
    (h₃ : speed_B = 35)
    (ratio : ℝ)
    (h₄ : ratio = 2)
    (h_time : ∀ a b : ℝ, a * t = b * t → a = b) :
  (speed_A * t) / (speed_B * t) = ratio := by
  sorry

end NUMINAMATH_GPT_car_distance_ratio_l1876_187687


namespace NUMINAMATH_GPT_min_abs_phi_l1876_187699

open Real

theorem min_abs_phi {k : ℤ} :
  ∃ (φ : ℝ), ∀ (k : ℤ), φ = - (5 * π) / 6 + k * π ∧ |φ| = π / 6 := sorry

end NUMINAMATH_GPT_min_abs_phi_l1876_187699


namespace NUMINAMATH_GPT_owen_sleep_hours_l1876_187622

-- Define the time spent by Owen in various activities
def hours_work : ℕ := 6
def hours_chores : ℕ := 7
def total_hours_day : ℕ := 24

-- The proposition to be proven
theorem owen_sleep_hours : (total_hours_day - (hours_work + hours_chores) = 11) := by
  sorry

end NUMINAMATH_GPT_owen_sleep_hours_l1876_187622


namespace NUMINAMATH_GPT_proof_x_plus_y_equals_30_l1876_187627

variable (x y : ℝ) (h_distinct : x ≠ y)
variable (h_det : Matrix.det ![
  ![2, 5, 10],
  ![4, x, y],
  ![4, y, x]
  ] = 0)

theorem proof_x_plus_y_equals_30 :
  x + y = 30 :=
sorry

end NUMINAMATH_GPT_proof_x_plus_y_equals_30_l1876_187627


namespace NUMINAMATH_GPT_complete_square_to_d_l1876_187665

-- Conditions given in the problem
def quadratic_eq (x : ℝ) : Prop := x^2 + 10 * x + 7 = 0

-- Equivalent Lean 4 statement of the problem
theorem complete_square_to_d (x : ℝ) (c d : ℝ) (h : quadratic_eq x) (hc : c = 5) : (x + c)^2 = d → d = 18 :=
by sorry

end NUMINAMATH_GPT_complete_square_to_d_l1876_187665


namespace NUMINAMATH_GPT_ratio_of_speeds_l1876_187606

noncomputable def speed_ratios (d t_b t : ℚ) : ℚ × ℚ  :=
  let d_b := t_b * t
  let d_a := d - d_b
  let t_h := t / 60
  let s_a := d_a / t_h
  let s_b := t_b
  (s_a / 15, s_b / 15)

theorem ratio_of_speeds
  (d : ℚ) (s_b : ℚ) (t : ℚ)
  (h : d = 88) (h1 : s_b = 90) (h2 : t = 32) :
  speed_ratios d s_b t = (5, 6) :=
  by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l1876_187606


namespace NUMINAMATH_GPT_area_of_square_not_covered_by_circles_l1876_187671

theorem area_of_square_not_covered_by_circles :
  let side : ℝ := 10
  let radius : ℝ := 5
  (side^2 - 4 * (π * radius^2) + 4 * (π * (radius^2) / 2)) = (100 - 50 * π) := 
sorry

end NUMINAMATH_GPT_area_of_square_not_covered_by_circles_l1876_187671


namespace NUMINAMATH_GPT_books_left_l1876_187608

namespace PaulBooksExample

-- Defining the initial conditions as given in the problem
def initial_books : ℕ := 134
def books_given : ℕ := 39
def books_sold : ℕ := 27

-- Proving that the final number of books Paul has is 68
theorem books_left : initial_books - (books_given + books_sold) = 68 := by
  sorry

end PaulBooksExample

end NUMINAMATH_GPT_books_left_l1876_187608


namespace NUMINAMATH_GPT_number_of_schools_is_23_l1876_187698

-- Conditions and definitions
noncomputable def number_of_students_per_school : ℕ := 3
def beth_rank : ℕ := 37
def carla_rank : ℕ := 64

-- Statement of the proof problem
theorem number_of_schools_is_23
  (n : ℕ)
  (h1 : ∀ i < n, ∃ r1 r2 r3: ℕ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)
  (h2 : ∀ i < n, ∃ A B C: ℕ, A = (2 * B + 1) ∧ C = A ∧ B = 35 ∧ A < beth_rank ∧ beth_rank < carla_rank):
  n = 23 :=
by
  sorry

end NUMINAMATH_GPT_number_of_schools_is_23_l1876_187698


namespace NUMINAMATH_GPT_subset_exists_l1876_187697

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {x + 2, 1}

-- Statement of the theorem
theorem subset_exists (x : ℝ) : B 2 ⊆ A 2 :=
by
  sorry

end NUMINAMATH_GPT_subset_exists_l1876_187697


namespace NUMINAMATH_GPT_eq_system_correct_l1876_187663

theorem eq_system_correct (x y : ℤ) : 
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) :=
sorry

end NUMINAMATH_GPT_eq_system_correct_l1876_187663


namespace NUMINAMATH_GPT_complex_fraction_simplification_l1876_187642

theorem complex_fraction_simplification :
  ((10^4 + 324) * (22^4 + 324) * (34^4 + 324) * (46^4 + 324) * (58^4 + 324)) /
  ((4^4 + 324) * (16^4 + 324) * (28^4 + 324) * (40^4 + 324) * (52^4 + 324)) = 373 :=
by
  sorry

end NUMINAMATH_GPT_complex_fraction_simplification_l1876_187642


namespace NUMINAMATH_GPT_number_of_strictly_increasing_sequences_l1876_187617

def strictly_increasing_sequences (n : ℕ) : ℕ :=
if n = 0 then 1 else if n = 1 then 1 else strictly_increasing_sequences (n - 1) + strictly_increasing_sequences (n - 2)

theorem number_of_strictly_increasing_sequences :
  strictly_increasing_sequences 12 = 144 :=
by
  sorry

end NUMINAMATH_GPT_number_of_strictly_increasing_sequences_l1876_187617


namespace NUMINAMATH_GPT_structure_of_S_l1876_187680

def set_S (x y : ℝ) : Prop :=
  (5 >= x + 1 ∧ 5 >= y - 5) ∨
  (x + 1 >= 5 ∧ x + 1 >= y - 5) ∨
  (y - 5 >= 5 ∧ y - 5 >= x + 1)

theorem structure_of_S :
  ∃ (a b c : ℝ), set_S x y ↔ (y <= x + 6) ∧ (x <= 4) ∧ (y <= 10) 
:= sorry

end NUMINAMATH_GPT_structure_of_S_l1876_187680


namespace NUMINAMATH_GPT_find_eccentricity_l1876_187626

def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

def eccentricity_conic_section (m : ℝ) (e : ℝ) : Prop :=
  (m = 6 → e = (Real.sqrt 30) / 6) ∧
  (m = -6 → e = Real.sqrt 7)

theorem find_eccentricity (m : ℝ) :
  geometric_sequence 4 m 9 →
  eccentricity_conic_section m ((Real.sqrt 30) / 6) ∨
  eccentricity_conic_section m (Real.sqrt 7) :=
by
  sorry

end NUMINAMATH_GPT_find_eccentricity_l1876_187626


namespace NUMINAMATH_GPT_complement_intersection_l1876_187603

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def M : Set ℕ := {1, 4}
noncomputable def N : Set ℕ := {2, 3}

theorem complement_intersection :
  ((U \ M) ∩ N) = {2, 3} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1876_187603


namespace NUMINAMATH_GPT_power_of_i_l1876_187647

theorem power_of_i : (Complex.I ^ 2018) = -1 := by
  sorry

end NUMINAMATH_GPT_power_of_i_l1876_187647


namespace NUMINAMATH_GPT_proof1_proof2_proof3_proof4_l1876_187636

-- Define variables.
variable (m n x y z : ℝ)

-- Prove the expressions equalities.
theorem proof1 : (m + 2 * n) - (m - 2 * n) = 4 * n := sorry
theorem proof2 : 2 * (x - 3) - (-x + 4) = 3 * x - 10 := sorry
theorem proof3 : 2 * x - 3 * (x - 2 * y + 3 * x) + 2 * (3 * x - 3 * y + 2 * z) = -4 * x + 4 * z := sorry
theorem proof4 : 8 * m^2 - (4 * m^2 - 2 * m - 4 * (2 * m^2 - 5 * m)) = 12 * m^2 - 18 * m := sorry

end NUMINAMATH_GPT_proof1_proof2_proof3_proof4_l1876_187636


namespace NUMINAMATH_GPT_num_perfect_square_factors_l1876_187678

def prime_factors_9600 (n : ℕ) : Prop :=
  n = 9600

theorem num_perfect_square_factors (n : ℕ) (h : prime_factors_9600 n) : 
  let cond := h
  (n = 9600) → 9600 = 2^6 * 5^2 * 3^1 → (∃ factors_count: ℕ, factors_count = 8) := by 
  sorry

end NUMINAMATH_GPT_num_perfect_square_factors_l1876_187678


namespace NUMINAMATH_GPT_average_sum_problem_l1876_187673

theorem average_sum_problem (avg : ℝ) (n : ℕ) (h_avg : avg = 5.3) (h_n : n = 10) : ∃ sum : ℝ, sum = avg * n ∧ sum = 53 :=
by
  sorry

end NUMINAMATH_GPT_average_sum_problem_l1876_187673


namespace NUMINAMATH_GPT_radius_of_circumscribed_circle_of_right_triangle_l1876_187684

theorem radius_of_circumscribed_circle_of_right_triangle 
  (a b c : ℝ)
  (h_area : (1 / 2) * a * b = 10)
  (h_inradius : (a + b - c) / 2 = 1)
  (h_hypotenuse : c = Real.sqrt (a^2 + b^2)) :
  c / 2 = 4.5 := 
sorry

end NUMINAMATH_GPT_radius_of_circumscribed_circle_of_right_triangle_l1876_187684


namespace NUMINAMATH_GPT_function_satisfies_conditions_l1876_187625

theorem function_satisfies_conditions :
  (∃ f : ℤ × ℤ → ℝ,
    (∀ x y z : ℤ, f (x, y) * f (y, z) * f (z, x) = 1) ∧
    (∀ x : ℤ, f (x + 1, x) = 2) ∧
    (∀ x y : ℤ, f (x, y) = 2 ^ (x - y))) :=
by
  sorry

end NUMINAMATH_GPT_function_satisfies_conditions_l1876_187625


namespace NUMINAMATH_GPT_oranges_savings_l1876_187657

-- Definitions for the conditions
def liam_oranges : Nat := 40
def liam_price_per_set : Real := 2.50
def oranges_per_set : Nat := 2

def claire_oranges : Nat := 30
def claire_price_per_orange : Real := 1.20

-- Statement of the problem to be proven
theorem oranges_savings : 
  liam_oranges / oranges_per_set * liam_price_per_set + 
  claire_oranges * claire_price_per_orange = 86 := 
by 
  sorry

end NUMINAMATH_GPT_oranges_savings_l1876_187657


namespace NUMINAMATH_GPT_solution_set_linear_inequalities_l1876_187612

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_linear_inequalities_l1876_187612


namespace NUMINAMATH_GPT_find_m_l1876_187607

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 36 m = 108) (h2 : Nat.lcm 45 m = 180) : m = 72 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_l1876_187607


namespace NUMINAMATH_GPT_tan_triple_angle_l1876_187685

theorem tan_triple_angle (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 :=
by
  -- to be completed
  sorry

end NUMINAMATH_GPT_tan_triple_angle_l1876_187685


namespace NUMINAMATH_GPT_div_eq_eight_fifths_l1876_187674

theorem div_eq_eight_fifths (a b : ℚ) (hb : b ≠ 0) (h : (a - b) / b = 3 / 5) : a / b = 8 / 5 :=
by
  sorry

end NUMINAMATH_GPT_div_eq_eight_fifths_l1876_187674


namespace NUMINAMATH_GPT_percentage_increase_is_50_l1876_187691

-- Define the conditions
variables {P : ℝ} {x : ℝ}

-- Define the main statement (goal)
theorem percentage_increase_is_50 (h : 0.80 * P + (0.008 * x * P) = 1.20 * P) : x = 50 :=
sorry  -- Skip the proof as per instruction

end NUMINAMATH_GPT_percentage_increase_is_50_l1876_187691


namespace NUMINAMATH_GPT_cases_needed_to_raise_funds_l1876_187600

-- Define conditions as lemmas that will be used in the main theorem.
lemma packs_per_case : ℕ := 3
lemma muffins_per_pack : ℕ := 4
lemma muffin_price : ℕ := 2
lemma fundraising_goal : ℕ := 120

-- Calculate muffins per case
noncomputable def muffins_per_case : ℕ := packs_per_case * muffins_per_pack

-- Calculate money earned per case
noncomputable def money_per_case : ℕ := muffins_per_case * muffin_price

-- The main theorem to prove the number of cases needed
theorem cases_needed_to_raise_funds : 
  (fundraising_goal / money_per_case) = 5 :=
by
  sorry

end NUMINAMATH_GPT_cases_needed_to_raise_funds_l1876_187600


namespace NUMINAMATH_GPT_smallest_a1_l1876_187648

theorem smallest_a1 (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_rec : ∀ n > 1, a n = 7 * a (n - 1) - n) :
  a 1 ≥ 13 / 36 :=
by
  sorry

end NUMINAMATH_GPT_smallest_a1_l1876_187648


namespace NUMINAMATH_GPT_cheryl_used_material_l1876_187654

theorem cheryl_used_material
    (material1 : ℚ) (material2 : ℚ) (leftover : ℚ)
    (h1 : material1 = 5/9)
    (h2 : material2 = 1/3)
    (h_lf : leftover = 8/24) :
    material1 + material2 - leftover = 5/9 :=
by
  sorry

end NUMINAMATH_GPT_cheryl_used_material_l1876_187654


namespace NUMINAMATH_GPT_remainder_sand_amount_l1876_187688

def total_sand : ℝ := 2548726
def bag_capacity : ℝ := 85741.2
def full_bags : ℝ := 29
def not_full_bag_sand : ℝ := 62231.2

theorem remainder_sand_amount :
  total_sand - (full_bags * bag_capacity) = not_full_bag_sand :=
by
  sorry

end NUMINAMATH_GPT_remainder_sand_amount_l1876_187688


namespace NUMINAMATH_GPT_compound_interest_rate_l1876_187655

theorem compound_interest_rate :
  ∀ (A P : ℝ) (t : ℕ),
  A = 4840.000000000001 ->
  P = 4000 ->
  t = 2 ->
  A = P * (1 + 0.1)^t :=
by
  intros A P t hA hP ht
  rw [hA, hP, ht]
  norm_num
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l1876_187655


namespace NUMINAMATH_GPT_part_time_employees_l1876_187623

theorem part_time_employees (total_employees : ℕ) (full_time_employees : ℕ) (part_time_employees : ℕ) 
  (h1 : total_employees = 65134) 
  (h2 : full_time_employees = 63093) 
  (h3 : total_employees = full_time_employees + part_time_employees) : 
  part_time_employees = 2041 :=
by 
  sorry

end NUMINAMATH_GPT_part_time_employees_l1876_187623


namespace NUMINAMATH_GPT_problem1_l1876_187645

theorem problem1 (x y : ℝ) (h1 : 2^(x + y) = x + 7) (h2 : x + y = 3) : (x = 1 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_problem1_l1876_187645


namespace NUMINAMATH_GPT_percentage_increase_l1876_187683

variable (S : ℝ) (P : ℝ)
variable (h1 : S + 0.10 * S = 330)
variable (h2 : S + P * S = 324)

theorem percentage_increase : P = 0.08 := sorry

end NUMINAMATH_GPT_percentage_increase_l1876_187683


namespace NUMINAMATH_GPT_valid_choice_count_l1876_187664

def is_valid_base_7_digit (n : ℕ) : Prop := n < 7
def is_valid_base_8_digit (n : ℕ) : Prop := n < 8
def to_base_10_base_7 (c3 c2 c1 c0 : ℕ) : ℕ := 2401 * c3 + 343 * c2 + 49 * c1 + 7 * c0
def to_base_10_base_8 (d3 d2 d1 d0 : ℕ) : ℕ := 4096 * d3 + 512 * d2 + 64 * d1 + 8 * d0
def is_four_digit_number (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

theorem valid_choice_count :
  ∃ (N : ℕ), is_four_digit_number N →
  ∀ (c3 c2 c1 c0 d3 d2 d1 d0 : ℕ),
    is_valid_base_7_digit c3 → is_valid_base_7_digit c2 → is_valid_base_7_digit c1 → is_valid_base_7_digit c0 →
    is_valid_base_8_digit d3 → is_valid_base_8_digit d2 → is_valid_base_8_digit d1 → is_valid_base_8_digit d0 →
    to_base_10_base_7 c3 c2 c1 c0 = N →
    to_base_10_base_8 d3 d2 d1 d0 = N →
    (to_base_10_base_7 c3 c2 c1 c0 + to_base_10_base_8 d3 d2 d1 d0) % 1000 = (2 * N) % 1000 → N = 20 :=
sorry

end NUMINAMATH_GPT_valid_choice_count_l1876_187664


namespace NUMINAMATH_GPT_question_1_question_2_l1876_187613

variable (a b c : ℝ × ℝ)
variable (k : ℝ)

def vect_a : ℝ × ℝ := (3, 2)
def vect_b : ℝ × ℝ := (-1, 2)
def vect_c : ℝ × ℝ := (4, 1)

theorem question_1 :
  3 • vect_a + vect_b - 2 • vect_c = (0, 6) := 
by
  sorry

theorem question_2 (k : ℝ) : 
  let lhs := (3 + 4 * k) * 2
  let rhs := -5 * (2 + k)
  (lhs = rhs) → k = -16 / 13 := 
by
  sorry

end NUMINAMATH_GPT_question_1_question_2_l1876_187613


namespace NUMINAMATH_GPT_not_minimum_on_l1876_187677

noncomputable def f (x m : ℝ) : ℝ :=
  x * Real.exp x - (m / 2) * x ^ 2 - m * x

theorem not_minimum_on (m : ℝ) : 
  ¬ (∃ x ∈ Set.Icc 1 2, f x m = Real.exp 2 - 2 * m ∧ 
  ∀ y ∈ Set.Icc 1 2, f y m ≥ f x m) :=
sorry

end NUMINAMATH_GPT_not_minimum_on_l1876_187677


namespace NUMINAMATH_GPT_smallest_positive_period_centers_of_symmetry_maximum_value_minimum_value_l1876_187646

noncomputable def f (x : ℝ) : ℝ := -2 * (Real.sin x)^2 + 2 * (Real.sin x) * (Real.cos x) + 1

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem centers_of_symmetry :
  ∀ k : ℤ, ∃ x, x = -Real.pi / 4 + k * Real.pi ∧ f (-x) = f x := sorry

theorem maximum_value :
  ∀ x : ℝ, f x ≤ 2 := sorry

theorem minimum_value :
  ∀ x : ℝ, f x ≥ -1 := sorry

end NUMINAMATH_GPT_smallest_positive_period_centers_of_symmetry_maximum_value_minimum_value_l1876_187646


namespace NUMINAMATH_GPT_simplify_expression_l1876_187620

theorem simplify_expression :
  (2 + 1 / 2) / (1 - 3 / 4) = 10 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1876_187620


namespace NUMINAMATH_GPT_negation_of_exists_l1876_187624

theorem negation_of_exists {x : ℝ} (h : ∃ x : ℝ, 3^x + x < 0) : ∀ x : ℝ, 3^x + x ≥ 0 :=
sorry

end NUMINAMATH_GPT_negation_of_exists_l1876_187624


namespace NUMINAMATH_GPT_probability_shaded_is_one_third_l1876_187638

-- Define the total number of regions as a constant
def total_regions : ℕ := 12

-- Define the number of shaded regions as a constant
def shaded_regions : ℕ := 4

-- The probability that the tip of a spinner stopping in a shaded region
def probability_shaded : ℚ := shaded_regions / total_regions

-- Main theorem stating the probability calculation is correct
theorem probability_shaded_is_one_third : probability_shaded = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_shaded_is_one_third_l1876_187638


namespace NUMINAMATH_GPT_ab_cd_eq_zero_l1876_187695

theorem ab_cd_eq_zero  
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1)
  (h2 : c^2 + d^2 = 1)
  (h3 : ad - bc = -1) :
  ab + cd = 0 :=
by
  sorry

end NUMINAMATH_GPT_ab_cd_eq_zero_l1876_187695


namespace NUMINAMATH_GPT_find_f_4_l1876_187602

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_4 : (∀ x : ℝ, f (x / 2 - 1) = 2 * x + 3) → f 4 = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_f_4_l1876_187602


namespace NUMINAMATH_GPT_algebraic_expression_value_l1876_187629

-- Given conditions as definitions and assumption
variables (a b : ℝ)
def expression1 (x : ℝ) := 2 * a * x^3 - 3 * b * x + 8
def expression2 := 9 * b - 6 * a + 2

theorem algebraic_expression_value
  (h1 : expression1 (-1) = 18) :
  expression2 = 32 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1876_187629
