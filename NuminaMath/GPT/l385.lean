import Mathlib

namespace NUMINAMATH_GPT_mean_days_correct_l385_38553

noncomputable def mean_days (a1 a2 a3 a4 a5 d1 d2 d3 d4 d5 : ℕ) : ℚ :=
  (a1 * d1 + a2 * d2 + a3 * d3 + a4 * d4 + a5 * d5 : ℚ) / (a1 + a2 + a3 + a4 + a5)

theorem mean_days_correct : mean_days 2 4 5 7 4 1 2 4 5 6 = 4.05 := by
  sorry

end NUMINAMATH_GPT_mean_days_correct_l385_38553


namespace NUMINAMATH_GPT_selection_count_Group3_selection_count_Group4_selection_count_Group5_probability_A_or_B_l385_38557

/-
  Conditions:
-/
def Group3 : ℕ := 18
def Group4 : ℕ := 12
def Group5 : ℕ := 6
def TotalParticipantsToSelect : ℕ := 12
def TotalFromGroups345 : ℕ := Group3 + Group4 + Group5

/-
  Questions:
  1. Prove that the number of people to be selected from each group using stratified sampling:
\ 2. Prove that the probability of selecting at least one of A or B from Group 5 is 3/5.
-/

theorem selection_count_Group3 : 
  (Group3 * TotalParticipantsToSelect / TotalFromGroups345) = 6 := 
  by sorry

theorem selection_count_Group4 : 
  (Group4 * TotalParticipantsToSelect / TotalFromGroups345) = 4 := 
  by sorry

theorem selection_count_Group5 : 
  (Group5 * TotalParticipantsToSelect / TotalFromGroups345) = 2 := 
  by sorry

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_A_or_B : 
  (combination 6 2 - combination 4 2) / combination 6 2 = 3 / 5 := 
  by sorry

end NUMINAMATH_GPT_selection_count_Group3_selection_count_Group4_selection_count_Group5_probability_A_or_B_l385_38557


namespace NUMINAMATH_GPT_malcolm_followers_l385_38586

theorem malcolm_followers :
  let instagram_followers := 240
  let facebook_followers := 500
  let twitter_followers := (instagram_followers + facebook_followers) / 2
  let tiktok_followers := 3 * twitter_followers
  let youtube_followers := tiktok_followers + 510
  instagram_followers + facebook_followers + twitter_followers + tiktok_followers + youtube_followers = 3840 :=
by {
  sorry
}

end NUMINAMATH_GPT_malcolm_followers_l385_38586


namespace NUMINAMATH_GPT_valid_six_digit_numbers_l385_38589

def is_divisible_by_4 (n : Nat) : Prop :=
  n % 4 = 0

def digit_sum (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

def is_divisible_by_9 (n : Nat) : Prop :=
  digit_sum n % 9 = 0

def is_valid_six_digit_number (n : Nat) : Prop :=
  ∃ (a b : Nat), n = b * 100000 + 20140 + a ∧ is_divisible_by_4 (10 * 2014 + a) ∧ is_divisible_by_9 (b * 100000 + 20140 + a)

theorem valid_six_digit_numbers :
  { n | is_valid_six_digit_number n } = {220140, 720144, 320148} :=
by
  sorry

end NUMINAMATH_GPT_valid_six_digit_numbers_l385_38589


namespace NUMINAMATH_GPT_men_employed_l385_38530

theorem men_employed (M : ℕ) (W : ℕ)
  (h1 : W = M * 9)
  (h2 : W = (M + 10) * 6) : M = 20 := by
  sorry

end NUMINAMATH_GPT_men_employed_l385_38530


namespace NUMINAMATH_GPT_ant_probability_after_10_minutes_l385_38529

-- Definitions based on the conditions given in the problem
def ant_start_at_A := true
def moves_each_minute (n : ℕ) := n == 10
def blue_dots (x y : ℤ) : Prop := 
  (x == 0 ∨ y == 0) ∧ (x + y) % 2 == 0
def A_at_center (x y : ℤ) : Prop := x == 0 ∧ y == 0
def B_north_of_A (x y : ℤ) : Prop := x == 0 ∧ y == 1

-- The probability we need to prove
def probability_ant_at_B_after_10_minutes := 1 / 9

-- We state our proof problem
theorem ant_probability_after_10_minutes :
  ant_start_at_A ∧ moves_each_minute 10 ∧ blue_dots 0 0 ∧ blue_dots 0 1 ∧ A_at_center 0 0 ∧ B_north_of_A 0 1
  → probability_ant_at_B_after_10_minutes = 1 / 9 := 
sorry

end NUMINAMATH_GPT_ant_probability_after_10_minutes_l385_38529


namespace NUMINAMATH_GPT_find_root_equation_l385_38506

theorem find_root_equation : ∃ x : ℤ, x - (5 / (x - 4)) = 2 - (5 / (x - 4)) ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_root_equation_l385_38506


namespace NUMINAMATH_GPT_work_hours_to_pay_off_debt_l385_38558

theorem work_hours_to_pay_off_debt (initial_debt paid_amount hourly_rate remaining_debt work_hours : ℕ) 
  (h₁ : initial_debt = 100) 
  (h₂ : paid_amount = 40) 
  (h₃ : hourly_rate = 15) 
  (h₄ : remaining_debt = initial_debt - paid_amount) 
  (h₅ : work_hours = remaining_debt / hourly_rate) : 
  work_hours = 4 :=
by
  sorry

end NUMINAMATH_GPT_work_hours_to_pay_off_debt_l385_38558


namespace NUMINAMATH_GPT_max_divisor_f_l385_38531

-- Given definition
def f (n : ℕ) : ℕ := (2 * n + 7) * 3 ^ n + 9

-- Main theorem to be proved
theorem max_divisor_f :
  ∃ m : ℕ, (∀ n : ℕ, 0 < n → m ∣ f n) ∧ m = 36 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_max_divisor_f_l385_38531


namespace NUMINAMATH_GPT_a_squared_plus_b_squared_gt_one_over_four_sequence_is_arithmetic_l385_38597

-- For Question 1
theorem a_squared_plus_b_squared_gt_one_over_four (a b : ℝ) (h : a + b = 1) : a^2 + b^2 > 1/4 :=
sorry

-- For Question 2
theorem sequence_is_arithmetic (n : ℕ) (S : ℕ → ℝ) (h : ∀ n, S n = 2 * (n:ℝ)^2 - 3 * (n:ℝ) - 2) :
  ∃ d, ∀ n, (S n / (2 * (n:ℝ) + 1)) = (S (n + 1) / (2 * (n + 1:ℝ) + 1)) + d :=
sorry

end NUMINAMATH_GPT_a_squared_plus_b_squared_gt_one_over_four_sequence_is_arithmetic_l385_38597


namespace NUMINAMATH_GPT_partI_solution_set_l385_38551

def f (x : ℝ) (a : ℝ) : ℝ := abs (x + a) - abs (x - a^2 - a)

theorem partI_solution_set (x : ℝ) : 
  (f x 1 ≤ 1) ↔ (x ≤ -1) :=
sorry

end NUMINAMATH_GPT_partI_solution_set_l385_38551


namespace NUMINAMATH_GPT_eighth_L_prime_is_31_l385_38501

def setL := {n : ℕ | n > 0 ∧ n % 3 = 1}

def isLPrime (n : ℕ) : Prop :=
  n ∈ setL ∧ n ≠ 1 ∧ ∀ m ∈ setL, (m ∣ n) → (m = 1 ∨ m = n)

theorem eighth_L_prime_is_31 : 
  ∃ n ∈ setL, isLPrime n ∧ 
  (∀ k, (∃ m ∈ setL, isLPrime m ∧ m < n) → k < 8 → m ≠ n) :=
by sorry

end NUMINAMATH_GPT_eighth_L_prime_is_31_l385_38501


namespace NUMINAMATH_GPT_faye_has_62_pieces_of_candy_l385_38580

-- Define initial conditions
def initialCandy : Nat := 47
def eatenCandy : Nat := 25
def receivedCandy : Nat := 40

-- Define the resulting number of candies after eating and receiving more candies
def resultingCandy : Nat := initialCandy - eatenCandy + receivedCandy

-- State the theorem and provide the proof
theorem faye_has_62_pieces_of_candy :
  resultingCandy = 62 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_faye_has_62_pieces_of_candy_l385_38580


namespace NUMINAMATH_GPT_maximize_quadratic_expression_l385_38521

theorem maximize_quadratic_expression :
  ∃ x : ℝ, (∀ y : ℝ, -2 * y^2 - 8 * y + 10 ≤ -2 * x^2 - 8 * x + 10) ∧ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_maximize_quadratic_expression_l385_38521


namespace NUMINAMATH_GPT_score_order_l385_38560

variable (A B C D : ℕ)

-- Condition 1: B + D = A + C
axiom h1 : B + D = A + C
-- Condition 2: A + B > C + D + 10
axiom h2 : A + B > C + D + 10
-- Condition 3: D > B + C + 20
axiom h3 : D > B + C + 20
-- Condition 4: A + B + C + D = 200
axiom h4 : A + B + C + D = 200

-- Question to prove: Order is Donna > Alice > Brian > Cindy
theorem score_order : D > A ∧ A > B ∧ B > C :=
by
  sorry

end NUMINAMATH_GPT_score_order_l385_38560


namespace NUMINAMATH_GPT_Jake_has_more_peaches_than_Jill_l385_38505

variables (Jake Steven Jill : ℕ)
variable (h1 : Jake = Steven - 5)
variable (h2 : Steven = Jill + 18)
variable (h3 : Jill = 87)

theorem Jake_has_more_peaches_than_Jill (Jake Steven Jill : ℕ) (h1 : Jake = Steven - 5) (h2 : Steven = Jill + 18) (h3 : Jill = 87) :
  Jake - Jill = 13 :=
by
  sorry

end NUMINAMATH_GPT_Jake_has_more_peaches_than_Jill_l385_38505


namespace NUMINAMATH_GPT_eggs_per_hen_per_day_l385_38533

theorem eggs_per_hen_per_day
  (hens : ℕ) (days : ℕ) (neighborTaken : ℕ) (dropped : ℕ) (finalEggs : ℕ) (E : ℕ) 
  (h1 : hens = 3) 
  (h2 : days = 7) 
  (h3 : neighborTaken = 12) 
  (h4 : dropped = 5) 
  (h5 : finalEggs = 46) 
  (totalEggs : ℕ := hens * E * days) 
  (afterNeighbor : ℕ := totalEggs - neighborTaken) 
  (beforeDropping : ℕ := finalEggs + dropped) : 
  totalEggs = beforeDropping + neighborTaken → E = 3 := sorry

end NUMINAMATH_GPT_eggs_per_hen_per_day_l385_38533


namespace NUMINAMATH_GPT_quadratic_inequality_false_range_l385_38537

theorem quadratic_inequality_false_range (a : ℝ) :
  (¬ ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (a < 0 ∨ a ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_false_range_l385_38537


namespace NUMINAMATH_GPT_find_m_l385_38571

theorem find_m (x y m : ℝ) (h1 : 2 * x + y = 1) (h2 : x + 2 * y = 2) (h3 : x + y = 2 * m - 1) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l385_38571


namespace NUMINAMATH_GPT_terminal_side_quadrant_l385_38572

theorem terminal_side_quadrant (k : ℤ) : 
  ∃ quadrant, quadrant = 1 ∨ quadrant = 3 ∧
  ∀ (α : ℝ), α = k * 180 + 45 → 
  (quadrant = 1 ∧ (∃ n : ℕ, k = 2 * n)) ∨ (quadrant = 3 ∧ (∃ n : ℕ, k = 2 * n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_terminal_side_quadrant_l385_38572


namespace NUMINAMATH_GPT_max_value_m_l385_38522

variable {a b m : ℝ}

theorem max_value_m (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b, (3 / a) + (1 / b) ≥ m / (a + 3 * b)) : m ≤ 12 :=
by 
  sorry

end NUMINAMATH_GPT_max_value_m_l385_38522


namespace NUMINAMATH_GPT_carvings_per_shelf_l385_38509

def total_wood_carvings := 56
def num_shelves := 7

theorem carvings_per_shelf : total_wood_carvings / num_shelves = 8 := by
  sorry

end NUMINAMATH_GPT_carvings_per_shelf_l385_38509


namespace NUMINAMATH_GPT_rectangular_plot_perimeter_l385_38593

theorem rectangular_plot_perimeter (w : ℝ) (P : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  (cost_per_meter = 6.5) →
  (total_cost = 1430) →
  (P = 2 * (w + (w + 10))) →
  (cost_per_meter * P = total_cost) →
  P = 220 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_plot_perimeter_l385_38593


namespace NUMINAMATH_GPT_percentage_of_stock_l385_38511

-- Definitions based on conditions
def income := 500  -- I
def investment := 1500  -- Inv
def price := 90  -- Price

-- Initiate the Lean 4 statement for the proof
theorem percentage_of_stock (P : ℝ) (h : income = (investment * P) / price) : P = 30 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_stock_l385_38511


namespace NUMINAMATH_GPT_union_of_A_and_B_l385_38559

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x < 4}

theorem union_of_A_and_B : A ∪ B = {x | x > 1} := 
by 
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l385_38559


namespace NUMINAMATH_GPT_additional_tobacco_acres_l385_38546

def original_land : ℕ := 1350
def original_ratio_units : ℕ := 9
def new_ratio_units : ℕ := 9

def acres_per_unit := original_land / original_ratio_units

def tobacco_old := 2 * acres_per_unit
def tobacco_new := 5 * acres_per_unit

theorem additional_tobacco_acres :
  tobacco_new - tobacco_old = 450 := by
  sorry

end NUMINAMATH_GPT_additional_tobacco_acres_l385_38546


namespace NUMINAMATH_GPT_Paula_needs_52_tickets_l385_38526

theorem Paula_needs_52_tickets :
  let g := 2
  let b := 4
  let r := 3
  let f := 1
  let t_g := 4
  let t_b := 5
  let t_r := 7
  let t_f := 3
  g * t_g + b * t_b + r * t_r + f * t_f = 52 := by
  intros
  sorry

end NUMINAMATH_GPT_Paula_needs_52_tickets_l385_38526


namespace NUMINAMATH_GPT_distance_AB_bounds_l385_38565

noncomputable def distance_AC : ℕ := 10
noncomputable def distance_AD : ℕ := 10
noncomputable def distance_BE : ℕ := 10
noncomputable def distance_BF : ℕ := 10
noncomputable def distance_AE : ℕ := 12
noncomputable def distance_AF : ℕ := 12
noncomputable def distance_BC : ℕ := 12
noncomputable def distance_BD : ℕ := 12
noncomputable def distance_CD : ℕ := 11
noncomputable def distance_EF : ℕ := 11
noncomputable def distance_CE : ℕ := 5
noncomputable def distance_DF : ℕ := 5

theorem distance_AB_bounds (AB : ℝ) :
  8.8 < AB ∧ AB < 19.2 :=
sorry

end NUMINAMATH_GPT_distance_AB_bounds_l385_38565


namespace NUMINAMATH_GPT_sheena_weeks_to_complete_dresses_l385_38585

/- Sheena is sewing the bridesmaid's dresses for her sister's wedding.
There are 7 bridesmaids in the wedding.
Each bridesmaid's dress takes a different number of hours to sew due to different styles and sizes.
The hours needed to sew the bridesmaid's dresses are as follows: 15 hours, 18 hours, 20 hours, 22 hours, 24 hours, 26 hours, and 28 hours.
If Sheena sews the dresses 5 hours each week, prove that it will take her 31 weeks to complete all the dresses. -/

def bridesmaid_hours : List ℕ := [15, 18, 20, 22, 24, 26, 28]

def total_hours_needed (hours : List ℕ) : ℕ :=
  hours.sum

def weeks_needed (total_hours : ℕ) (hours_per_week : ℕ) : ℕ :=
  (total_hours + hours_per_week - 1) / hours_per_week

theorem sheena_weeks_to_complete_dresses :
  weeks_needed (total_hours_needed bridesmaid_hours) 5 = 31 := by
  sorry

end NUMINAMATH_GPT_sheena_weeks_to_complete_dresses_l385_38585


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l385_38525

theorem sufficient_but_not_necessary_condition (a : ℝ) : 
  (a > 0) → (|2 * a + 1| > 1) ∧ ¬((|2 * a + 1| > 1) → (a > 0)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l385_38525


namespace NUMINAMATH_GPT_heartsuit_value_l385_38504

def heartsuit (x y : ℝ) := 4 * x + 6 * y

theorem heartsuit_value : heartsuit 3 4 = 36 := by
  sorry

end NUMINAMATH_GPT_heartsuit_value_l385_38504


namespace NUMINAMATH_GPT_three_digit_number_is_275_l385_38590

noncomputable def digits (n : ℕ) : ℕ × ℕ × ℕ :=
  (n / 100 % 10, n / 10 % 10, n % 10)

theorem three_digit_number_is_275 :
  ∃ (n : ℕ), n / 100 % 10 + n % 10 = n / 10 % 10 ∧
              7 * (n / 100 % 10) = n % 10 + n / 10 % 10 + 2 ∧
              n / 100 % 10 + n / 10 % 10 + n % 10 = 14 ∧
              n = 275 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_is_275_l385_38590


namespace NUMINAMATH_GPT_emily_sold_toys_l385_38583

theorem emily_sold_toys (initial_toys : ℕ) (remaining_toys : ℕ) (sold_toys : ℕ) 
  (h_initial : initial_toys = 7) 
  (h_remaining : remaining_toys = 4) 
  (h_sold : sold_toys = initial_toys - remaining_toys) :
  sold_toys = 3 :=
by sorry

end NUMINAMATH_GPT_emily_sold_toys_l385_38583


namespace NUMINAMATH_GPT_total_yen_l385_38594

/-- 
Abe's family has a checking account with 6359 yen
and a savings account with 3485 yen.
-/
def checking_account : ℕ := 6359
def savings_account : ℕ := 3485

/-- 
Prove that the total amount of yen Abe's family has
is equal to 9844 yen.
-/
theorem total_yen : checking_account + savings_account = 9844 :=
by
  sorry

end NUMINAMATH_GPT_total_yen_l385_38594


namespace NUMINAMATH_GPT_find_line_equation_l385_38561

theorem find_line_equation : 
  ∃ (m : ℝ), (∀ (x y : ℝ), (2 * x + y - 5 = 0) → (m = -2)) → 
  ∀ (x₀ y₀ : ℝ), (x₀ = -2) ∧ (y₀ = 3) → 
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * x₀ + b * y₀ + c = 0) ∧ (a = 1 ∧ b = -2 ∧ c = 8) := 
by
  sorry

end NUMINAMATH_GPT_find_line_equation_l385_38561


namespace NUMINAMATH_GPT_median_of_consecutive_integers_l385_38573

theorem median_of_consecutive_integers (a n : ℤ) (N : ℕ) (h1 : (a + (n - 1)) + (a + (N - n)) = 110) : 
  (2 * a + N - 1) / 2 = 55 := 
by {
  -- The proof goes here.
  sorry
}

end NUMINAMATH_GPT_median_of_consecutive_integers_l385_38573


namespace NUMINAMATH_GPT_project_completion_time_l385_38554

def work_rate_A : ℚ := 1 / 20
def work_rate_B : ℚ := 1 / 30
def total_project_days (x : ℚ) : Prop := (work_rate_A * (x - 10) + work_rate_B * x = 1)

theorem project_completion_time (x : ℚ) (h : total_project_days x) : x = 13 := 
sorry

end NUMINAMATH_GPT_project_completion_time_l385_38554


namespace NUMINAMATH_GPT_number_chosen_l385_38555

theorem number_chosen (x : ℤ) (h : x / 4 - 175 = 10) : x = 740 := by
  sorry

end NUMINAMATH_GPT_number_chosen_l385_38555


namespace NUMINAMATH_GPT_zander_stickers_l385_38516

theorem zander_stickers (total_stickers andrew_ratio bill_ratio : ℕ) (initial_stickers: total_stickers = 100) (andrew_fraction : andrew_ratio = 1 / 5) (bill_fraction : bill_ratio = 3 / 10) :
  let andrew_give_away := total_stickers * andrew_ratio
  let remaining_stickers := total_stickers - andrew_give_away
  let bill_give_away := remaining_stickers * bill_ratio
  let total_given_away := andrew_give_away + bill_give_away
  total_given_away = 44 :=
by
  sorry

end NUMINAMATH_GPT_zander_stickers_l385_38516


namespace NUMINAMATH_GPT_radius_of_circle_area_of_sector_l385_38517

theorem radius_of_circle (L : ℝ) (θ : ℝ) (hL : L = 50) (hθ : θ = 200) : 
  ∃ r : ℝ, r = 45 / Real.pi := 
by
  sorry

theorem area_of_sector (L : ℝ) (r : ℝ) (hL : L = 50) (hr : r = 45 / Real.pi) : 
  ∃ S : ℝ, S = 1125 / Real.pi := 
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_area_of_sector_l385_38517


namespace NUMINAMATH_GPT_triangle_type_l385_38500

-- Definitions given in the problem
def is_not_equal (a : ℝ) (b : ℝ) : Prop := a ≠ b
def log_eq (b x : ℝ) : Prop := Real.log x = Real.log 4 / Real.log b + Real.log (4 * x - 4) / Real.log b

-- Main theorem stating the type of triangle ABC
theorem triangle_type (a b c A B C : ℝ) (h_b_ne_1 : is_not_equal b 1) (h_C_over_A_root : log_eq b (C / A)) (h_sin_B_over_sin_A_root : log_eq b (Real.sin B / Real.sin A)) : (B = 90) ∧ (A ≠ C) :=
by
  sorry

end NUMINAMATH_GPT_triangle_type_l385_38500


namespace NUMINAMATH_GPT_missing_jar_size_l385_38550

theorem missing_jar_size (total_ounces jars_16 jars_28 jars_unknown m n p: ℕ) (h1 : m = 3) (h2 : n = 3) (h3 : p = 3)
    (total_jars : m + n + p = 9)
    (total_peanut_butter : 16 * m + 28 * n + jars_unknown * p = 252)
    : jars_unknown = 40 := by
  sorry

end NUMINAMATH_GPT_missing_jar_size_l385_38550


namespace NUMINAMATH_GPT_max_wooden_pencils_l385_38503

theorem max_wooden_pencils (m w : ℕ) (p : ℕ) (h1 : m + w = 72) (h2 : m = w + p) (hp : Nat.Prime p) : w = 35 :=
by
  sorry

end NUMINAMATH_GPT_max_wooden_pencils_l385_38503


namespace NUMINAMATH_GPT_difference_of_squares_65_35_l385_38569

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := 
  sorry

end NUMINAMATH_GPT_difference_of_squares_65_35_l385_38569


namespace NUMINAMATH_GPT_find_h_of_root_l385_38588

theorem find_h_of_root :
  ∀ h : ℝ, (-3)^3 + h * (-3) - 10 = 0 → h = -37/3 := by
  sorry

end NUMINAMATH_GPT_find_h_of_root_l385_38588


namespace NUMINAMATH_GPT_jane_reads_105_pages_in_a_week_l385_38539

-- Define the pages read in the morning and evening
def pages_morning := 5
def pages_evening := 10

-- Define the number of pages read in a day
def pages_per_day := pages_morning + pages_evening

-- Define the number of days in a week
def days_per_week := 7

-- Define the total number of pages read in a week
def pages_per_week := pages_per_day * days_per_week

-- The theorem that sums up the proof
theorem jane_reads_105_pages_in_a_week : pages_per_week = 105 := by
  sorry

end NUMINAMATH_GPT_jane_reads_105_pages_in_a_week_l385_38539


namespace NUMINAMATH_GPT_ellipse_equation_hyperbola_vertices_and_foci_exists_point_P_on_x_axis_angles_complementary_l385_38570

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 - y^2 / 2 = 1

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 2 = 1

def point_on_x_axis (P : ℝ × ℝ) : Prop :=
  P.snd = 0

def angles_complementary (P A B : ℝ × ℝ) : Prop :=
  let kPA := (A.snd - P.snd) / (A.fst - P.fst)
  let kPB := (B.snd - P.snd) / (B.fst - P.fst)
  kPA + kPB = 0

theorem ellipse_equation_hyperbola_vertices_and_foci :
  (∀ x y : ℝ, hyperbola_eq x y → ellipse_eq x y) :=
sorry

theorem exists_point_P_on_x_axis_angles_complementary (F2 A B : ℝ × ℝ) :
  F2 = (1, 0) → (∃ P : ℝ × ℝ, point_on_x_axis P ∧ angles_complementary P A B) :=
sorry

end NUMINAMATH_GPT_ellipse_equation_hyperbola_vertices_and_foci_exists_point_P_on_x_axis_angles_complementary_l385_38570


namespace NUMINAMATH_GPT_minimum_f_l385_38552

def f (x : ℝ) : ℝ := |x - 2| + |5 - x|

theorem minimum_f : ∃ x, f x = 3 :=
by
  use 3
  unfold f
  sorry

end NUMINAMATH_GPT_minimum_f_l385_38552


namespace NUMINAMATH_GPT_evaluate_expression_l385_38576

theorem evaluate_expression :
  let a := 3 * 4 * 5
  let b := (1 : ℝ) / 3
  let c := (1 : ℝ) / 4
  let d := (1 : ℝ) / 5
  (a : ℝ) * (b + c - d) = 23 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l385_38576


namespace NUMINAMATH_GPT_seven_lines_divide_into_29_regions_l385_38541

open Function

theorem seven_lines_divide_into_29_regions : 
  ∀ n : ℕ, (∀ l m : ℕ, l ≠ m → l < n ∧ m < n) → 1 + n + (n.choose 2) = 29 :=
by
  sorry

end NUMINAMATH_GPT_seven_lines_divide_into_29_regions_l385_38541


namespace NUMINAMATH_GPT_at_least_one_is_half_l385_38538

theorem at_least_one_is_half (x y z : ℝ) (h : x + y + z - 2 * (x * y + y * z + z * x) + 4 * x * y * z = 1 / 2) :
  x = 1 / 2 ∨ y = 1 / 2 ∨ z = 1 / 2 :=
sorry

end NUMINAMATH_GPT_at_least_one_is_half_l385_38538


namespace NUMINAMATH_GPT_count_numbers_with_digit_7_count_numbers_divisible_by_3_or_5_l385_38562

-- Statement for Question 1
theorem count_numbers_with_digit_7 :
  ∃ n, n = 19 ∧ (∀ k, (k < 100 → (k / 10 = 7 ∨ k % 10 = 7) ↔ k ≠ 77)) :=
sorry

-- Statement for Question 2
theorem count_numbers_divisible_by_3_or_5 :
  ∃ n, n = 47 ∧ (∀ k, (k < 100 → (k % 3 = 0 ∨ k % 5 = 0)) ↔ (k % 15 = 0)) :=
sorry

end NUMINAMATH_GPT_count_numbers_with_digit_7_count_numbers_divisible_by_3_or_5_l385_38562


namespace NUMINAMATH_GPT_minimum_phi_l385_38512

noncomputable def initial_function (x : ℝ) (ϕ : ℝ) : ℝ :=
  2 * Real.sin (4 * x + ϕ)

noncomputable def translated_function (x : ℝ) (ϕ : ℝ) : ℝ :=
  2 * Real.sin (4 * (x - (Real.pi / 6)) + ϕ)

theorem minimum_phi (ϕ : ℝ) :
  (∃ k : ℤ, ϕ = k * Real.pi + 7 * Real.pi / 6) →
  (∃ ϕ_min : ℝ, (ϕ_min = ϕ ∧ ϕ_min = Real.pi / 6)) :=
by
  sorry

end NUMINAMATH_GPT_minimum_phi_l385_38512


namespace NUMINAMATH_GPT_perimeter_of_quadrilateral_eq_fifty_l385_38548

theorem perimeter_of_quadrilateral_eq_fifty
  (a b : ℝ)
  (h1 : a = 10)
  (h2 : b = 15)
  (h3 : ∀ (p q r s : ℝ), p + q = r + s) : 
  2 * a + 2 * b = 50 := 
by
  sorry

end NUMINAMATH_GPT_perimeter_of_quadrilateral_eq_fifty_l385_38548


namespace NUMINAMATH_GPT_vince_bus_ride_distance_l385_38540

/-- 
  Vince's bus ride to school is 0.625 mile, 
  given that Zachary's bus ride is 0.5 mile 
  and Vince's bus ride is 0.125 mile longer than Zachary's.
--/
theorem vince_bus_ride_distance (zachary_ride : ℝ) (vince_longer : ℝ) 
  (h1 : zachary_ride = 0.5) (h2 : vince_longer = 0.125) 
  : zachary_ride + vince_longer = 0.625 :=
by sorry

end NUMINAMATH_GPT_vince_bus_ride_distance_l385_38540


namespace NUMINAMATH_GPT_purple_balls_correct_l385_38568

-- Define the total number of balls and individual counts
def total_balls : ℕ := 100
def white_balls : ℕ := 20
def green_balls : ℕ := 30
def yellow_balls : ℕ := 10
def red_balls : ℕ := 37

-- Probability that a ball chosen is neither red nor purple
def prob_neither_red_nor_purple : ℚ := 0.6

-- The number of purple balls to be proven
def purple_balls : ℕ := 3

-- The condition used for the proof
def condition : Prop := prob_neither_red_nor_purple = (white_balls + green_balls + yellow_balls) / total_balls

-- The proof problem statement
theorem purple_balls_correct (h : condition) : 
  ∃ P : ℕ, P = purple_balls ∧ P + red_balls = total_balls - (white_balls + green_balls + yellow_balls) :=
by
  have P := total_balls - (white_balls + green_balls + yellow_balls + red_balls)
  existsi P
  sorry

end NUMINAMATH_GPT_purple_balls_correct_l385_38568


namespace NUMINAMATH_GPT_sufficient_not_necessary_l385_38507

-- Definitions based on the conditions
def f1 (x y : ℝ) : Prop := x^2 + y^2 = 0
def f2 (x y : ℝ) : Prop := x * y = 0

-- The theorem we need to prove
theorem sufficient_not_necessary (x y : ℝ) : f1 x y → f2 x y ∧ ¬ (f2 x y → f1 x y) := 
by sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l385_38507


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l385_38549

theorem find_x_squared_plus_y_squared (x y : ℝ) 
  (h1 : (x - y)^2 = 49) (h2 : x * y = -12) : x^2 + y^2 = 25 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l385_38549


namespace NUMINAMATH_GPT_average_marks_of_all_students_l385_38523

theorem average_marks_of_all_students (n₁ n₂ a₁ a₂ : ℕ) (h₁ : n₁ = 30) (h₂ : a₁ = 40) (h₃ : n₂ = 50) (h₄ : a₂ = 80) :
  ((n₁ * a₁ + n₂ * a₂) / (n₁ + n₂) = 65) :=
by
  sorry

end NUMINAMATH_GPT_average_marks_of_all_students_l385_38523


namespace NUMINAMATH_GPT_find_adult_buffet_price_l385_38515

variable {A : ℝ} -- Let A be the price for the adult buffet
variable (children_cost : ℝ := 45) -- Total cost for the children's buffet
variable (senior_discount : ℝ := 0.9) -- Discount for senior citizens
variable (total_cost : ℝ := 159) -- Total amount spent by Mr. Smith
variable (num_adults : ℕ := 2) -- Number of adults (Mr. Smith and his wife)
variable (num_seniors : ℕ := 2) -- Number of senior citizens

theorem find_adult_buffet_price (h1 : children_cost = 45)
    (h2 : total_cost = 159)
    (h3 : ∀ x, num_adults * x + num_seniors * (senior_discount * x) + children_cost = total_cost)
    : A = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_adult_buffet_price_l385_38515


namespace NUMINAMATH_GPT_find_first_group_men_l385_38519

variable (M : ℕ)

def first_group_men := M
def days_for_first_group := 20
def men_in_second_group := 12
def days_for_second_group := 30

theorem find_first_group_men (h1 : first_group_men * days_for_first_group = men_in_second_group * days_for_second_group) :
  first_group_men = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_first_group_men_l385_38519


namespace NUMINAMATH_GPT_omega_min_value_l385_38578

def min_omega (ω : ℝ) : Prop :=
  ω > 0 ∧ ∃ k : ℤ, (k ≠ 0 ∧ ω = 8)

theorem omega_min_value (ω : ℝ) (h1 : ω > 0) (h2 : ∃ k : ℤ, k ≠ 0 ∧ (k * 2 * π) / ω = π / 4) : 
  ω = 8 :=
by
  sorry

end NUMINAMATH_GPT_omega_min_value_l385_38578


namespace NUMINAMATH_GPT_unique_solution_l385_38582

noncomputable def solve_system (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ) (x1 x2 x3 : ℝ) : Prop :=
  (a11 * x1 + a12 * x2 + a13 * x3 = 0) ∧
  (a21 * x1 + a22 * x2 + a23 * x3 = 0) ∧
  (a31 * x1 + a32 * x2 + a33 * x3 = 0)

theorem unique_solution 
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ)
  (h1 : 0 < a11) (h2 : 0 < a22) (h3 : 0 < a33)
  (h4 : a12 < 0) (h5 : a13 < 0) (h6 : a21 < 0)
  (h7 : a23 < 0) (h8 : a31 < 0) (h9 : a32 < 0)
  (h10 : 0 < a11 + a12 + a13) (h11 : 0 < a21 + a22 + a23) (h12 : 0 < a31 + a32 + a33) :
  ∀ (x1 x2 x3 : ℝ), solve_system a11 a12 a13 a21 a22 a23 a31 a32 a33 x1 x2 x3 → (x1 = 0 ∧ x2 = 0 ∧ x3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l385_38582


namespace NUMINAMATH_GPT_even_digits_count_1998_l385_38527

-- Define the function for counting the total number of digits used in the first n positive even integers
def totalDigitsEvenIntegers (n : ℕ) : ℕ :=
  let totalSingleDigit := 4 -- 2, 4, 6, 8
  let numDoubleDigit := 45 -- 10 to 98
  let digitsDoubleDigit := numDoubleDigit * 2
  let numTripleDigit := 450 -- 100 to 998
  let digitsTripleDigit := numTripleDigit * 3
  let numFourDigit := 1499 -- 1000 to 3996
  let digitsFourDigit := numFourDigit * 4
  totalSingleDigit + digitsDoubleDigit + digitsTripleDigit + digitsFourDigit

-- Theorem: The total number of digits used when the first 1998 positive even integers are written is 7440.
theorem even_digits_count_1998 : totalDigitsEvenIntegers 1998 = 7440 :=
  sorry

end NUMINAMATH_GPT_even_digits_count_1998_l385_38527


namespace NUMINAMATH_GPT_incorrect_statement_l385_38563

theorem incorrect_statement :
  let statementA := "The shortest distance between two points is a line segment."
  let statementB := "Vertical angles are congruent."
  let statementC := "Complementary angles of the same measure are congruent."
  let statementD := "There is only one line passing through a point outside a given line that is parallel to the given line."
  (statementA = "correct") ∧ 
  (statementB = "correct") ∧ 
  (statementC = "correct") ∧ 
  (statementD = "incorrect") :=
by
  let statementA := "The shortest distance between two points is a line segment."
  let statementB := "Vertical angles are congruent."
  let statementC := "Complementary angles of the same measure are congruent."
  let statementD := "There is only one line passing through a point outside a given line that is parallel to the given line."
  have hA : statementA = "correct" := sorry
  have hB : statementB = "correct" := sorry
  have hC : statementC = "correct" := sorry
  have hD : statementD = "incorrect" := sorry
  exact ⟨hA, hB, hC, hD⟩

end NUMINAMATH_GPT_incorrect_statement_l385_38563


namespace NUMINAMATH_GPT_lengths_of_triangle_sides_l385_38520

open Real

noncomputable def triangle_side_lengths (a b c : ℝ) (A B C : ℝ) :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ A + B + C = π ∧ A = 60 * π / 180 ∧
  10 * sqrt 3 = 0.5 * a * b * sin A ∧
  a + b = 13 ∧
  c = sqrt (a^2 + b^2 - 2 * a * b * cos A)

theorem lengths_of_triangle_sides
  (a b c : ℝ) (A B C : ℝ)
  (h : triangle_side_lengths a b c A B C) :
  (a = 5 ∧ b = 8 ∧ c = 7) ∨ (a = 8 ∧ b = 5 ∧ c = 7) :=
sorry

end NUMINAMATH_GPT_lengths_of_triangle_sides_l385_38520


namespace NUMINAMATH_GPT_set_union_l385_38513

theorem set_union :
  let M := {x | x^2 + 2 * x - 3 = 0}
  let N := {-1, 2, 3}
  M ∪ N = {-1, 1, 2, -3, 3} :=
by
  sorry

end NUMINAMATH_GPT_set_union_l385_38513


namespace NUMINAMATH_GPT_num_students_earning_B_l385_38542

open Real

theorem num_students_earning_B (total_students : ℝ) (pA : ℝ) (pB : ℝ) (pC : ℝ) (students_A : ℝ) (students_B : ℝ) (students_C : ℝ) :
  total_students = 31 →
  pA = 0.7 * pB →
  pC = 1.4 * pB →
  students_A = 0.7 * students_B →
  students_C = 1.4 * students_B →
  students_A + students_B + students_C = total_students →
  students_B = 10 :=
by
  intros h_total_students h_pa h_pc h_students_A h_students_C h_total_eq
  sorry

end NUMINAMATH_GPT_num_students_earning_B_l385_38542


namespace NUMINAMATH_GPT_B_and_C_complementary_l385_38595

def EventA (selected : List String) : Prop :=
  selected.count "boy" = 1

def EventB (selected : List String) : Prop :=
  selected.count "boy" ≥ 1

def EventC (selected : List String) : Prop :=
  selected.count "girl" = 2

theorem B_and_C_complementary :
  ∀ selected : List String,
    (selected.length = 2 ∧ (EventB selected ∨ EventC selected)) ∧ 
    (¬ (EventB selected ∧ EventC selected)) →
    (EventB selected → ¬ EventC selected) ∧ (EventC selected → ¬ EventB selected) :=
  sorry

end NUMINAMATH_GPT_B_and_C_complementary_l385_38595


namespace NUMINAMATH_GPT_non_zero_number_is_nine_l385_38579

theorem non_zero_number_is_nine (x : ℝ) (h1 : x ≠ 0) (h2 : (x + x^2) / 2 = 5 * x) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_non_zero_number_is_nine_l385_38579


namespace NUMINAMATH_GPT_soccer_tournament_probability_l385_38508

noncomputable def prob_teamA_more_points : ℚ :=
  (163 : ℚ) / 256

theorem soccer_tournament_probability :
  m + n = 419 ∧ prob_teamA_more_points = 163 / 256 := sorry

end NUMINAMATH_GPT_soccer_tournament_probability_l385_38508


namespace NUMINAMATH_GPT_jessica_total_cost_l385_38591

-- Define the costs
def cost_cat_toy : ℝ := 10.22
def cost_cage : ℝ := 11.73

-- Define the total cost
def total_cost : ℝ := cost_cat_toy + cost_cage

-- State the theorem
theorem jessica_total_cost : total_cost = 21.95 := by
  sorry

end NUMINAMATH_GPT_jessica_total_cost_l385_38591


namespace NUMINAMATH_GPT_red_lettuce_cost_l385_38514

-- Define the known conditions
def cost_per_pound : Nat := 2
def total_pounds : Nat := 7
def cost_green_lettuce : Nat := 8

-- Define the total cost calculation
def total_cost : Nat := total_pounds * cost_per_pound
def cost_red_lettuce : Nat := total_cost - cost_green_lettuce

-- Statement to prove: cost_red_lettuce = 6
theorem red_lettuce_cost :
  cost_red_lettuce = 6 :=
by
  sorry

end NUMINAMATH_GPT_red_lettuce_cost_l385_38514


namespace NUMINAMATH_GPT_values_of_b_for_real_root_l385_38574

noncomputable def polynomial_has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^5 + b * x^4 - x^3 + b * x^2 - x + b = 0

theorem values_of_b_for_real_root :
  {b : ℝ | polynomial_has_real_root b} = {b : ℝ | b ≤ -1 ∨ b ≥ 1} :=
sorry

end NUMINAMATH_GPT_values_of_b_for_real_root_l385_38574


namespace NUMINAMATH_GPT_set_characteristics_l385_38518

-- Define the characteristics of elements in a set
def characteristic_definiteness := true
def characteristic_distinctness := true
def characteristic_unorderedness := true
def characteristic_reality := false -- We aim to prove this

-- The problem statement in Lean
theorem set_characteristics :
  ¬ characteristic_reality :=
by
  -- Here would be the proof, but we add sorry as indicated.
  sorry

end NUMINAMATH_GPT_set_characteristics_l385_38518


namespace NUMINAMATH_GPT_kiwi_count_l385_38535

theorem kiwi_count (s b o k : ℕ)
  (h1 : s + b + o + k = 340)
  (h2 : s = 3 * b)
  (h3 : o = 2 * k)
  (h4 : k = 5 * s) :
  k = 104 :=
sorry

end NUMINAMATH_GPT_kiwi_count_l385_38535


namespace NUMINAMATH_GPT_ellipse_standard_equation_parabola_standard_equation_l385_38510

theorem ellipse_standard_equation (x y : ℝ) (a b : ℝ) (h₁ : a > b ∧ b > 0)
  (h₂ : 2 * a = Real.sqrt ((3 + 2) ^ 2 + (-2 * Real.sqrt 6) ^ 2) 
      + Real.sqrt ((3 - 2) ^ 2 + (-2 * Real.sqrt 6) ^ 2))
  (h₃ : b^2 = a^2 - 4) 
  : (x^2 / 36 + y^2 / 32 = 1) :=
by sorry

theorem parabola_standard_equation (y : ℝ) (p : ℝ) (h₁ : p > 0)
  (h₂ : -p / 2 = -1 / 2) 
  : (y^2 = 2 * p * 1) :=
by sorry

end NUMINAMATH_GPT_ellipse_standard_equation_parabola_standard_equation_l385_38510


namespace NUMINAMATH_GPT_cubes_sum_identity_l385_38532

variable {a b : ℝ}

theorem cubes_sum_identity (h : (a / (1 + b) + b / (1 + a) = 1)) : a^3 + b^3 = a + b :=
sorry

end NUMINAMATH_GPT_cubes_sum_identity_l385_38532


namespace NUMINAMATH_GPT_AM_GM_inequality_equality_case_of_AM_GM_l385_38584

theorem AM_GM_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : (x / y) + (y / x) ≥ 2 :=
by
  sorry

theorem equality_case_of_AM_GM (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : ((x / y) + (y / x) = 2) ↔ (x = y) :=
by
  sorry

end NUMINAMATH_GPT_AM_GM_inequality_equality_case_of_AM_GM_l385_38584


namespace NUMINAMATH_GPT_stamps_ratio_l385_38534

noncomputable def number_of_stamps_bought := 300
noncomputable def total_stamps_after_purchase := 450
noncomputable def number_of_stamps_before_purchase := total_stamps_after_purchase - number_of_stamps_bought

theorem stamps_ratio : (number_of_stamps_before_purchase : ℚ) / number_of_stamps_bought = 1 / 2 := by
  have h : number_of_stamps_before_purchase = total_stamps_after_purchase - number_of_stamps_bought := rfl
  rw [h]
  norm_num
  sorry

end NUMINAMATH_GPT_stamps_ratio_l385_38534


namespace NUMINAMATH_GPT_NorrisSavings_l385_38528

theorem NorrisSavings : 
  let saved_september := 29
  let saved_october := 25
  let saved_november := 31
  let saved_december := 35
  let saved_january := 40
  saved_september + saved_october + saved_november + saved_december + saved_january = 160 :=
by
  sorry

end NUMINAMATH_GPT_NorrisSavings_l385_38528


namespace NUMINAMATH_GPT_number_of_chairs_l385_38547

theorem number_of_chairs (x t c b T C B: ℕ) (r1 r2 r3: ℕ)
  (h1: x = 2250) (h2: t = 18) (h3: c = 12) (h4: b = 30) 
  (h5: r1 = 2) (h6: r2 = 3) (h7: r3 = 1) 
  (h_ratio1: T / C = r1 / r2) (h_ratio2: B / C = r3 / r2) 
  (h_eq: t * T + c * C + b * B = x) : C = 66 :=
by
  sorry

end NUMINAMATH_GPT_number_of_chairs_l385_38547


namespace NUMINAMATH_GPT_range_of_m_l385_38545

def P (m : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ (x1 ^ 2 + m * x1 + 1 = 0) ∧ (x2 ^ 2 + m * x2 + 1 = 0) ∧ (x1 < 0) ∧ (x2 < 0)

def Q (m : ℝ) : Prop :=
  ∀ (x : ℝ), 4 * x ^ 2 + 4 * (m - 2) * x + 1 ≠ 0

def P_or_Q (m : ℝ) : Prop :=
  P m ∨ Q m

def P_and_Q (m : ℝ) : Prop :=
  P m ∧ Q m

theorem range_of_m (m : ℝ) : P_or_Q m ∧ ¬P_and_Q m ↔ m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_m_l385_38545


namespace NUMINAMATH_GPT_elevation_angle_second_ship_l385_38543

-- Assume h is the height of the lighthouse.
def h : ℝ := 100

-- Assume d_total is the distance between the two ships.
def d_total : ℝ := 273.2050807568877

-- Assume θ₁ is the angle of elevation from the first ship.
def θ₁ : ℝ := 30

-- Assume θ₂ is the angle of elevation from the second ship.
def θ₂ : ℝ := 45

-- Prove that angle of elevation from the second ship is 45 degrees.
theorem elevation_angle_second_ship : θ₂ = 45 := by
  sorry

end NUMINAMATH_GPT_elevation_angle_second_ship_l385_38543


namespace NUMINAMATH_GPT_quadratic_trinomial_int_l385_38502

theorem quadratic_trinomial_int (a b c x : ℤ) (h : y = (x - a) * (x - 6) + 1) :
  ∃ (b c : ℤ), (x + b) * (x + c) = (x - 8) * (x - 6) + 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_trinomial_int_l385_38502


namespace NUMINAMATH_GPT_twice_as_many_juniors_as_seniors_l385_38592

theorem twice_as_many_juniors_as_seniors (j s : ℕ) (h : (1/3 : ℝ) * j = (2/3 : ℝ) * s) : j = 2 * s :=
by
  --proof steps here
  sorry

end NUMINAMATH_GPT_twice_as_many_juniors_as_seniors_l385_38592


namespace NUMINAMATH_GPT_intersection_A_B_l385_38596

def A : Set ℝ := { x | x^2 - 2*x < 0 }
def B : Set ℝ := { x | |x| > 1 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l385_38596


namespace NUMINAMATH_GPT_largest_rectangle_area_l385_38587

theorem largest_rectangle_area (x y : ℝ) (h1 : 2*x + 2*y = 60) (h2 : x ≥ 2*y) : ∃ A, A = x*y ∧ A ≤ 200 := by
  sorry

end NUMINAMATH_GPT_largest_rectangle_area_l385_38587


namespace NUMINAMATH_GPT_sheila_weekly_earnings_is_288_l385_38575

-- Define the conditions as constants.
def sheilaWorksHoursPerDay (d : String) : ℕ :=
  if d = "Monday" ∨ d = "Wednesday" ∨ d = "Friday" then 8
  else if d = "Tuesday" ∨ d = "Thursday" then 6
  else 0

def hourlyWage : ℕ := 8

-- Calculate total weekly earnings based on conditions.
def weeklyEarnings : ℕ :=
  (sheilaWorksHoursPerDay "Monday" + sheilaWorksHoursPerDay "Wednesday" + sheilaWorksHoursPerDay "Friday") * hourlyWage +
  (sheilaWorksHoursPerDay "Tuesday" + sheilaWorksHoursPerDay "Thursday") * hourlyWage

-- The Lean statement for the proof.
theorem sheila_weekly_earnings_is_288 : weeklyEarnings = 288 :=
  by
    sorry

end NUMINAMATH_GPT_sheila_weekly_earnings_is_288_l385_38575


namespace NUMINAMATH_GPT_trigonometric_identity_l385_38564

theorem trigonometric_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.cos (A / 2)) ^ 2 = (Real.cos (B / 2)) ^ 2 + (Real.cos (C / 2)) ^ 2 - 2 * (Real.cos (B / 2)) * (Real.cos (C / 2)) * (Real.sin (A / 2)) :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l385_38564


namespace NUMINAMATH_GPT_spend_amount_7_l385_38599

variable (x y z w : ℕ) (k : ℕ)

theorem spend_amount_7 
  (h1 : 10 * x + 15 * y + 25 * z + 40 * w = 100 * k)
  (h2 : x + y + z + w = 30)
  (h3 : (x = 5 ∨ x = 10) ∧ (y = 5 ∨ y = 10) ∧ (z = 5 ∨ z = 10) ∧ (w = 5 ∨ w = 10)) : 
  k = 7 := 
sorry

end NUMINAMATH_GPT_spend_amount_7_l385_38599


namespace NUMINAMATH_GPT_reduction_for_1750_yuan_max_daily_profit_not_1900_l385_38577

def average_shirts_per_day : ℕ := 40 
def profit_per_shirt_initial : ℕ := 40 
def price_reduction_increase_shirts (reduction : ℝ) : ℝ := reduction * 2 
def daily_profit (reduction : ℝ) : ℝ := (profit_per_shirt_initial - reduction) * (average_shirts_per_day + price_reduction_increase_shirts reduction)

-- Part 1: Proving the reduction that results in 1750 yuan profit
theorem reduction_for_1750_yuan : ∃ x : ℝ, daily_profit x = 1750 ∧ x = 15 := 
by {
  sorry
}

-- Part 2: Proving that the maximum cannot reach 1900 yuan
theorem max_daily_profit_not_1900 : ∀ x : ℝ, daily_profit x ≤ 1800 ∧ (∀ y : ℝ, y ≥ daily_profit x → y < 1900) :=
by {
  sorry
}

end NUMINAMATH_GPT_reduction_for_1750_yuan_max_daily_profit_not_1900_l385_38577


namespace NUMINAMATH_GPT_explicit_form_correct_l385_38556

-- Define the original function form
def f (a b x : ℝ) := 4*x^3 + a*x^2 + b*x + 5

-- Given tangent line slope condition at x = 1
axiom tangent_slope : ∀ (a b : ℝ), (12 * 1^2 + 2 * a * 1 + b = -12)

-- Given the point (1, f(1)) lies on the tangent line y = -12x
axiom tangent_point : ∀ (a b : ℝ), (4 * 1^3 + a * 1^2 + b * 1 + 5 = -12)

-- Definition for the specific f(x) found in solution
def f_explicit (x : ℝ) := 4*x^3 - 3*x^2 - 18*x + 5

-- Finding maximum and minimum values on interval [-3, 1]
def max_value : ℝ := -76
def min_value : ℝ := 16

theorem explicit_form_correct : 
  ∃ a b : ℝ, 
  (∀ x, f a b x = f_explicit x) ∧ 
  (max_value = 16) ∧ 
  (min_value = -76) := 
by
  sorry

end NUMINAMATH_GPT_explicit_form_correct_l385_38556


namespace NUMINAMATH_GPT_income_expenditure_ratio_l385_38524

theorem income_expenditure_ratio (I E S : ℝ) (h1 : I = 20000) (h2 : S = 4000) (h3 : S = I - E) :
    I / E = 5 / 4 :=
sorry

end NUMINAMATH_GPT_income_expenditure_ratio_l385_38524


namespace NUMINAMATH_GPT_evaluate_expression_l385_38567

theorem evaluate_expression : abs (abs (abs (-2 + 2) - 2) * 2) = 4 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l385_38567


namespace NUMINAMATH_GPT_total_dividend_received_l385_38581

noncomputable def investmentAmount : Nat := 14400
noncomputable def faceValue : Nat := 100
noncomputable def premium : Real := 0.20
noncomputable def declaredDividend : Real := 0.07

theorem total_dividend_received :
  let cost_per_share := faceValue * (1 + premium)
  let number_of_shares := investmentAmount / cost_per_share
  let dividend_per_share := faceValue * declaredDividend
  let total_dividend := number_of_shares * dividend_per_share
  total_dividend = 840 := 
by 
  sorry

end NUMINAMATH_GPT_total_dividend_received_l385_38581


namespace NUMINAMATH_GPT_students_speak_both_l385_38536

theorem students_speak_both (total E T N : ℕ) (h1 : total = 150) (h2 : E = 55) (h3 : T = 85) (h4 : N = 30) :
  E + T - (total - N) = 20 := by
  -- Main proof logic
  sorry

end NUMINAMATH_GPT_students_speak_both_l385_38536


namespace NUMINAMATH_GPT_cos_theta_four_times_l385_38598

theorem cos_theta_four_times (theta : ℝ) (h : Real.cos theta = 1 / 3) : 
  Real.cos (4 * theta) = 17 / 81 := 
sorry

end NUMINAMATH_GPT_cos_theta_four_times_l385_38598


namespace NUMINAMATH_GPT_harry_total_hours_l385_38566

variable (x h y : ℕ)

theorem harry_total_hours :
  ((h + 2 * y) = 42) → ∃ t, t = h + y :=
  by
    sorry -- Proof is omitted as per the instructions

end NUMINAMATH_GPT_harry_total_hours_l385_38566


namespace NUMINAMATH_GPT_fifteenth_term_l385_38544

variable (a b : ℤ)

def sum_first_n_terms (n : ℕ) : ℤ := n * (2 * a + (n - 1) * b) / 2

axiom sum_first_10 : sum_first_n_terms 10 = 60
axiom sum_first_20 : sum_first_n_terms 20 = 320

def nth_term (n : ℕ) : ℤ := a + (n - 1) * b

theorem fifteenth_term : nth_term 15 = 25 :=
by
  sorry

end NUMINAMATH_GPT_fifteenth_term_l385_38544
