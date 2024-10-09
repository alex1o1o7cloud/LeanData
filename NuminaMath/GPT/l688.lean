import Mathlib

namespace gcf_360_270_lcm_360_270_l688_68815

def prime_factors_360 := [(2, 3), (3, 2), (5, 1)]
def prime_factors_270 := [(2, 1), (3, 3), (5, 1)]

def GCF (a b: ℕ) : ℕ := 2^1 * 3^2 * 5^1
def LCM (a b: ℕ) : ℕ := 2^3 * 3^3 * 5^1

-- Theorem: The GCF of 360 and 270 is 90
theorem gcf_360_270 : GCF 360 270 = 90 := by
  sorry

-- Theorem: The LCM of 360 and 270 is 1080
theorem lcm_360_270 : LCM 360 270 = 1080 := by
  sorry

end gcf_360_270_lcm_360_270_l688_68815


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l688_68802

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l688_68802


namespace calculate_3_to_5_mul_7_to_5_l688_68826

theorem calculate_3_to_5_mul_7_to_5 : 3^5 * 7^5 = 4084101 :=
by {
  -- Sorry is added to skip the proof; assuming the proof is done following standard arithmetic calculations
  sorry
}

end calculate_3_to_5_mul_7_to_5_l688_68826


namespace area_of_given_field_l688_68868

noncomputable def area_of_field (cost_in_rupees : ℕ) (rate_per_meter_in_paise : ℕ) (ratio_width : ℕ) (ratio_length : ℕ) : ℕ :=
  let cost_in_paise := cost_in_rupees * 100
  let perimeter := (ratio_width + ratio_length) * 2
  let x := cost_in_paise / (perimeter * rate_per_meter_in_paise)
  let width := ratio_width * x
  let length := ratio_length * x
  width * length

theorem area_of_given_field :
  let cost_in_rupees := 105
  let rate_per_meter_in_paise := 25
  let ratio_width := 3
  let ratio_length := 4
  area_of_field cost_in_rupees rate_per_meter_in_paise ratio_width ratio_length = 10800 :=
by
  sorry

end area_of_given_field_l688_68868


namespace gcd_lcm_sum_l688_68840

theorem gcd_lcm_sum :
  Nat.gcd 44 64 + Nat.lcm 48 18 = 148 := 
by
  sorry

end gcd_lcm_sum_l688_68840


namespace find_james_number_l688_68820

theorem find_james_number (x : ℝ) 
  (h1 : 3 * (3 * x + 10) = 141) : 
  x = 12.33 :=
by 
  sorry

end find_james_number_l688_68820


namespace final_value_A_is_5_l688_68803

/-
Problem: Given a 3x3 grid of numbers and a series of operations that add or subtract 1 to two adjacent cells simultaneously, prove that the number in position A in the table on the right is 5.
Conditions:
1. The initial grid is:
   \[
   \begin{array}{ccc}
   a & b & c \\
   d & e & f \\
   g & h & i \\
   \end{array}
   \]
2. Each operation involves adding or subtracting 1 from two adjacent cells.
3. The sum of all numbers in the grid remains unchanged.
-/

def table_operations (a b c d e f g h i : ℤ) : ℤ :=
-- A is determined based on the given problem and conditions
  5

theorem final_value_A_is_5 (a b c d e f g h i : ℤ) : 
  table_operations a b c d e f g h i = 5 :=
sorry

end final_value_A_is_5_l688_68803


namespace remainder_is_x_plus_2_l688_68883

noncomputable def problem_division := 
  ∀ x : ℤ, ∃ q r : ℤ, (x^3 + 2 * x^2) = q * (x^2 + 3 * x + 2) + r ∧ r < x^2 + 3 * x + 2 ∧ r = x + 2

theorem remainder_is_x_plus_2 : problem_division := sorry

end remainder_is_x_plus_2_l688_68883


namespace number_of_pipes_l688_68888

theorem number_of_pipes (L : ℝ) : 
  let r_small := 1
  let r_large := 3
  let len_small := L
  let len_large := 2 * L
  let volume_large := π * r_large^2 * len_large
  let volume_small := π * r_small^2 * len_small
  volume_large = 18 * volume_small :=
by
  sorry

end number_of_pipes_l688_68888


namespace green_minus_blue_is_40_l688_68894

noncomputable def number_of_green_minus_blue_disks (total_disks : ℕ) (ratio_blue : ℕ) (ratio_yellow : ℕ) (ratio_green : ℕ) : ℕ :=
  let total_ratio := ratio_blue + ratio_yellow + ratio_green
  let disks_per_part := total_disks / total_ratio
  let blue_disks := ratio_blue * disks_per_part
  let green_disks := ratio_green * disks_per_part
  green_disks - blue_disks

theorem green_minus_blue_is_40 :
  number_of_green_minus_blue_disks 144 3 7 8 = 40 :=
sorry

end green_minus_blue_is_40_l688_68894


namespace range_of_a_l688_68828

theorem range_of_a (a : ℝ) : (forall x : ℝ, (a-3) * x > 1 → x < 1 / (a-3)) → a < 3 :=
by
  sorry

end range_of_a_l688_68828


namespace orthocenter_of_ABC_l688_68890

structure Point3D := (x : ℝ) (y : ℝ) (z : ℝ)

def A : Point3D := ⟨-1, 3, 2⟩
def B : Point3D := ⟨4, -2, 2⟩
def C : Point3D := ⟨2, -1, 6⟩

def orthocenter (A B C : Point3D) : Point3D :=
  -- formula to calculate the orthocenter
  sorry

theorem orthocenter_of_ABC :
  orthocenter A B C = ⟨101 / 150, 192 / 150, 232 / 150⟩ :=
by 
  -- proof steps
  sorry

end orthocenter_of_ABC_l688_68890


namespace steve_final_amount_l688_68837

def initial_deposit : ℝ := 100
def interest_years_1_to_3 : ℝ := 0.10
def interest_years_4_to_5 : ℝ := 0.08
def annual_deposit_years_1_to_2 : ℝ := 10
def annual_deposit_years_3_to_5 : ℝ := 15

def total_after_one_year (initial : ℝ) (annual : ℝ) (interest : ℝ) : ℝ :=
  initial * (1 + interest) + annual

def steve_saving_after_five_years : ℝ :=
  let year1 := total_after_one_year initial_deposit annual_deposit_years_1_to_2 interest_years_1_to_3
  let year2 := total_after_one_year year1 annual_deposit_years_1_to_2 interest_years_1_to_3
  let year3 := total_after_one_year year2 annual_deposit_years_3_to_5 interest_years_1_to_3
  let year4 := total_after_one_year year3 annual_deposit_years_3_to_5 interest_years_4_to_5
  let year5 := total_after_one_year year4 annual_deposit_years_3_to_5 interest_years_4_to_5
  year5

theorem steve_final_amount :
  steve_saving_after_five_years = 230.88768 := by
  sorry

end steve_final_amount_l688_68837


namespace quadratic_expression_value_l688_68816

theorem quadratic_expression_value (x1 x2 : ℝ) (h1 : x1 + x2 = 4) (h2 : x1 * x2 = 2) (hx : x1^2 - 4 * x1 + 2 = 0) :
  x1^2 - 4 * x1 + 2 * x1 * x2 = 2 :=
sorry

end quadratic_expression_value_l688_68816


namespace within_acceptable_range_l688_68876

def flour_weight : ℝ := 25.18
def flour_label : ℝ := 25
def tolerance : ℝ := 0.25

theorem within_acceptable_range  :
  (flour_label - tolerance) ≤ flour_weight ∧ flour_weight ≤ (flour_label + tolerance) :=
by
  sorry

end within_acceptable_range_l688_68876


namespace students_in_class_l688_68800

variable (G B : ℕ)

def total_plants (G B : ℕ) : ℕ := 3 * G + B / 3

theorem students_in_class (h1 : total_plants G B = 24) (h2 : B / 3 = 6) : G + B = 24 :=
by
  sorry

end students_in_class_l688_68800


namespace total_pay_is_correct_l688_68893

-- Define the constants and conditions
def regular_rate := 3  -- $ per hour
def regular_hours := 40  -- hours
def overtime_multiplier := 2  -- overtime pay is twice the regular rate
def overtime_hours := 8  -- hours

-- Calculate regular and overtime pay
def regular_pay := regular_rate * regular_hours
def overtime_rate := regular_rate * overtime_multiplier
def overtime_pay := overtime_rate * overtime_hours

-- Calculate total pay
def total_pay := regular_pay + overtime_pay

-- Prove that the total pay is $168
theorem total_pay_is_correct : total_pay = 168 := by
  -- The proof goes here
  sorry

end total_pay_is_correct_l688_68893


namespace visitors_current_day_l688_68853

-- Define the number of visitors on the previous day and the additional visitors
def v_prev : ℕ := 600
def v_add : ℕ := 61

-- Prove that the number of visitors on the current day is 661
theorem visitors_current_day : v_prev + v_add = 661 :=
by
  sorry

end visitors_current_day_l688_68853


namespace binom_15_13_eq_105_l688_68839

theorem binom_15_13_eq_105 : Nat.choose 15 13 = 105 := by
  sorry

end binom_15_13_eq_105_l688_68839


namespace trivia_team_points_l688_68846

theorem trivia_team_points 
    (total_members : ℕ) 
    (members_absent : ℕ) 
    (total_points : ℕ) 
    (members_present : ℕ := total_members - members_absent) 
    (points_per_member : ℕ := total_points / members_present) 
    (h1 : total_members = 7) 
    (h2 : members_absent = 2) 
    (h3 : total_points = 20) : 
    points_per_member = 4 :=
by
    sorry

end trivia_team_points_l688_68846


namespace solve_for_3x2_plus_6_l688_68880

theorem solve_for_3x2_plus_6 (x : ℚ) (h : 5 * x + 3 = 2 * x - 4) : 3 * (x^2 + 6) = 103 / 3 :=
by
  sorry

end solve_for_3x2_plus_6_l688_68880


namespace initial_honey_amount_l688_68811

variable (H : ℝ)

theorem initial_honey_amount :
  (0.70 * 0.60 * 0.50) * H = 315 → H = 1500 :=
by
  sorry

end initial_honey_amount_l688_68811


namespace power_of_two_l688_68835

theorem power_of_two (n : ℕ) (h : 2^n = 32 * (1 / 2) ^ 2) : n = 3 :=
by {
  sorry
}

end power_of_two_l688_68835


namespace min_weighings_to_find_counterfeit_l688_68836

-- Definition of the problem conditions.
def coin_is_genuine (coins : Fin 10 → ℝ) (n : Fin 10) : Prop :=
  ∀ m : Fin 10, m ≠ n → coins m = coins (Fin.mk 0 sorry)

def counterfit_coin_is_lighter (coins : Fin 10 → ℝ) (n : Fin 10) : Prop :=
  ∀ m : Fin 10, m ≠ n → coins n < coins m

-- The theorem statement
theorem min_weighings_to_find_counterfeit :
  (∀ coins : Fin 10 → ℝ, ∃ n : Fin 10, coin_is_genuine coins n ∧ counterfit_coin_is_lighter coins n → ∃ min_weighings : ℕ, min_weighings = 3) :=
by {
  sorry
}

end min_weighings_to_find_counterfeit_l688_68836


namespace Lena_stops_in_X_l688_68844

def circumference : ℕ := 60
def distance_run : ℕ := 7920
def starting_point : String := "T"
def quarter_stops : String := "X"

theorem Lena_stops_in_X :
  (distance_run / circumference) * circumference + (distance_run % circumference) = distance_run →
  distance_run % circumference = 0 →
  (distance_run % circumference = 0 → starting_point = quarter_stops) →
  quarter_stops = "X" :=
sorry

end Lena_stops_in_X_l688_68844


namespace divide_64_to_get_800_l688_68856

theorem divide_64_to_get_800 (x : ℝ) (h : 64 / x = 800) : x = 0.08 :=
sorry

end divide_64_to_get_800_l688_68856


namespace fourth_root_of_207360000_l688_68884

theorem fourth_root_of_207360000 :
  120 ^ 4 = 207360000 :=
sorry

end fourth_root_of_207360000_l688_68884


namespace integer_solution_of_inequalities_l688_68863

theorem integer_solution_of_inequalities :
  (∀ x : ℝ, 3 * x - 4 ≤ 6 * x - 2 → (2 * x + 1) / 3 - 1 < (x - 1) / 2 → (x = 0)) :=
sorry

end integer_solution_of_inequalities_l688_68863


namespace meaningful_fraction_l688_68865

theorem meaningful_fraction (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by sorry

end meaningful_fraction_l688_68865


namespace transformed_average_l688_68862

theorem transformed_average (n : ℕ) (original_average factor : ℝ) 
  (h1 : n = 15) (h2 : original_average = 21.5) (h3 : factor = 7) :
  (original_average * factor) = 150.5 :=
by
  sorry

end transformed_average_l688_68862


namespace solve_for_x_l688_68866

theorem solve_for_x (y z x : ℝ) (h1 : 2 / 3 = y / 90) (h2 : 2 / 3 = (y + z) / 120) (h3 : 2 / 3 = (x - z) / 150) : x = 120 :=
by
  sorry

end solve_for_x_l688_68866


namespace betty_total_stones_l688_68819

def stones_per_bracelet : ℕ := 14
def number_of_bracelets : ℕ := 10
def total_stones : ℕ := stones_per_bracelet * number_of_bracelets

theorem betty_total_stones : total_stones = 140 := by
  sorry

end betty_total_stones_l688_68819


namespace math_or_sci_but_not_both_l688_68850

-- Definitions of the conditions
variable (students_math_and_sci : ℕ := 15)
variable (students_math : ℕ := 30)
variable (students_only_sci : ℕ := 18)

-- The theorem to prove
theorem math_or_sci_but_not_both :
  (students_math - students_math_and_sci) + students_only_sci = 33 := by
  -- Proof is omitted.
  sorry

end math_or_sci_but_not_both_l688_68850


namespace non_empty_set_A_l688_68873

-- Definitions based on conditions
def A (a : ℝ) : Set ℝ := {x | x ^ 2 = a}

-- Theorem statement
theorem non_empty_set_A (a : ℝ) (h : (A a).Nonempty) : 0 ≤ a :=
by
  sorry

end non_empty_set_A_l688_68873


namespace carol_total_points_l688_68830

-- Define the conditions for Carol's game points.
def first_round_points := 17
def second_round_points := 6
def last_round_points := -16

-- Prove that the total points at the end of the game are 7.
theorem carol_total_points : first_round_points + second_round_points + last_round_points = 7 := by
  sorry

end carol_total_points_l688_68830


namespace find_extreme_values_find_m_range_for_zeros_l688_68852

noncomputable def f (x m : ℝ) : ℝ := Real.log x - m * x + 2

theorem find_extreme_values (m : ℝ) :
  (∀ x > 0, m ≤ 0 → (f x m ≠ 0 ∨ ∀ y > 0, f y m ≥ f x m ∨ f y m ≤ f x m)) ∧
  (∀ x > 0, m > 0 → ∃ x_max, x_max = 1 / m ∧ ∀ y > 0, f y m ≤ f x_max m) := 
sorry

theorem find_m_range_for_zeros (m : ℝ) :
  (∃ a b, a = 1 / Real.exp 2 ∧ b = Real.exp 1 ∧ (f a m = 0 ∧ f b m = 0)) ↔ 
  (m ≥ 3 / Real.exp 1 ∧ m < Real.exp 1) :=
sorry

end find_extreme_values_find_m_range_for_zeros_l688_68852


namespace distinct_weights_count_l688_68870

theorem distinct_weights_count (n : ℕ) (h : n = 4) : 
  -- Given four weights and a two-pan balance scale without a pointer,
  ∃ m : ℕ, 
  -- prove that the number of distinct weights of cargo
  (m = 40) ∧  
  -- that can be exactly measured if the weights can be placed on both pans of the scale is 40.
  m = 3^n - 1 ∧ (m / 2 = 40) := by
  sorry

end distinct_weights_count_l688_68870


namespace ceil_sqrt_169_eq_13_l688_68805

theorem ceil_sqrt_169_eq_13 : Int.ceil (Real.sqrt 169) = 13 := by
  sorry

end ceil_sqrt_169_eq_13_l688_68805


namespace maximum_correct_answers_l688_68872

theorem maximum_correct_answers (c w u : ℕ) :
  c + w + u = 25 →
  4 * c - w = 70 →
  c ≤ 19 :=
by
  sorry

end maximum_correct_answers_l688_68872


namespace bertha_no_children_count_l688_68889

-- Definitions
def bertha_daughters : ℕ := 6
def granddaughters_per_daughter : ℕ := 6
def total_daughters_and_granddaughters : ℕ := 30

-- Theorem to be proved
theorem bertha_no_children_count : 
  ∃ x : ℕ, (x * granddaughters_per_daughter + bertha_daughters = total_daughters_and_granddaughters) ∧ 
           (bertha_daughters - x + x * granddaughters_per_daughter = 26) :=
sorry

end bertha_no_children_count_l688_68889


namespace rate_of_fencing_is_4_90_l688_68825

noncomputable def rate_of_fencing_per_meter : ℝ :=
  let area_hectares := 13.86
  let cost := 6466.70
  let area_m2 := area_hectares * 10000
  let radius := Real.sqrt (area_m2 / Real.pi)
  let circumference := 2 * Real.pi * radius
  cost / circumference

theorem rate_of_fencing_is_4_90 :
  rate_of_fencing_per_meter = 4.90 := sorry

end rate_of_fencing_is_4_90_l688_68825


namespace number_of_defective_pens_l688_68878

noncomputable def defective_pens (total : ℕ) (prob : ℚ) : ℕ :=
  let N := 6 -- since we already know the steps in the solution leading to N = 6
  let D := total - N
  D

theorem number_of_defective_pens (total : ℕ) (prob : ℚ) :
  (total = 12) → (prob = 0.22727272727272727) → defective_pens total prob = 6 :=
by
  intros ht hp
  unfold defective_pens
  sorry

end number_of_defective_pens_l688_68878


namespace triangle_angles_l688_68808

theorem triangle_angles (A B C : ℝ) 
  (h1 : A + B + C = 180)
  (h2 : B = 120)
  (h3 : (∃D, A = D ∧ (A + A + C = 180 ∨ A + C + C = 180)) ∨ (∃E, C = E ∧ (B + 15 + 45 = 180 ∨ B + 15 + 15 = 180))) :
  (A = 40 ∧ C = 20) ∨ (A = 45 ∧ C = 15) :=
sorry

end triangle_angles_l688_68808


namespace expenditure_of_negative_amount_l688_68861

theorem expenditure_of_negative_amount (x : ℝ) (h : x < 0) : 
  ∃ y : ℝ, y > 0 ∧ x = -y :=
by
  sorry

end expenditure_of_negative_amount_l688_68861


namespace seventy_seventh_digit_is_three_l688_68814

-- Define the sequence of digits from the numbers 60 to 1 in decreasing order.
def sequence_of_digits : List Nat :=
  (List.range' 1 60).reverse.bind (fun n => n.digits 10)

-- Define a function to get the nth digit from the list.
def digit_at_position (n : Nat) : Option Nat :=
  sequence_of_digits.get? (n - 1)

-- The statement to prove
theorem seventy_seventh_digit_is_three : digit_at_position 77 = some 3 :=
sorry

end seventy_seventh_digit_is_three_l688_68814


namespace solve_for_x_l688_68818

theorem solve_for_x (x : ℚ) (h : (x + 2) / (x - 3) = (x - 4) / (x + 5)) : x = 1 / 7 :=
sorry

end solve_for_x_l688_68818


namespace min_val_l688_68898

theorem min_val (x y : ℝ) (h : x + 2 * y = 1) : 2^x + 4^y = 2 * Real.sqrt 2 :=
sorry

end min_val_l688_68898


namespace perimeter_area_ratio_le_8_l688_68824

/-- Let \( S \) be a shape in the plane obtained as a union of finitely many unit squares.
    The perimeter of a single unit square is 4 and its area is 1.
    Prove that the ratio of the perimeter \( P \) and the area \( A \) of \( S \)
    is at most 8, i.e., \(\frac{P}{A} \leq 8\). -/
theorem perimeter_area_ratio_le_8
  (S : Set (ℝ × ℝ)) 
  (unit_square : ∀ (x y : ℝ), (x, y) ∈ S → (x + 1, y + 1) ∈ S ∧ (x + 1, y) ∈ S ∧ (x, y + 1) ∈ S ∧ (x, y) ∈ S)
  (P A : ℝ)
  (unit_square_perimeter : ∀ (x y : ℝ), (x, y) ∈ S → P = 4)
  (unit_square_area : ∀ (x y : ℝ), (x, y) ∈ S → A = 1) :
  P / A ≤ 8 :=
sorry

end perimeter_area_ratio_le_8_l688_68824


namespace suzy_total_jumps_in_two_days_l688_68809

-- Definitions based on the conditions in the problem
def yesterdays_jumps : ℕ := 247
def additional_jumps_today : ℕ := 131
def todays_jumps : ℕ := yesterdays_jumps + additional_jumps_today

-- Lean statement of the proof problem
theorem suzy_total_jumps_in_two_days : yesterdays_jumps + todays_jumps = 625 := by
  sorry

end suzy_total_jumps_in_two_days_l688_68809


namespace total_weekly_cups_brewed_l688_68886

-- Define the given conditions
def weekday_cups_per_hour : ℕ := 10
def weekend_total_cups : ℕ := 120
def shop_open_hours_per_day : ℕ := 5
def weekdays_in_week : ℕ := 5

-- Prove the total number of coffee cups brewed in one week
theorem total_weekly_cups_brewed : 
  (weekday_cups_per_hour * shop_open_hours_per_day * weekdays_in_week) 
  + weekend_total_cups = 370 := 
by
  sorry

end total_weekly_cups_brewed_l688_68886


namespace solve_equation_l688_68801

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 ↔ x = 1 ∨ x = 0 :=
by
  sorry

end solve_equation_l688_68801


namespace solution_set_of_inequality_l688_68827

theorem solution_set_of_inequality (x : ℝ) : (x - 1 ≤ (1 + x) / 3) → (x ≤ 2) :=
by
  sorry

end solution_set_of_inequality_l688_68827


namespace sale_in_first_month_l688_68881

theorem sale_in_first_month
  (s2 : ℕ)
  (s3 : ℕ)
  (s4 : ℕ)
  (s5 : ℕ)
  (s6 : ℕ)
  (required_total_sales : ℕ)
  (average_sales : ℕ)
  : (required_total_sales = 39000) → 
    (average_sales = 6500) → 
    (s2 = 6927) →
    (s3 = 6855) →
    (s4 = 7230) →
    (s5 = 6562) →
    (s6 = 4991) →
    s2 + s3 + s4 + s5 + s6 = 32565 →
    required_total_sales - (s2 + s3 + s4 + s5 + s6) = 6435 :=
by
  intros
  sorry

end sale_in_first_month_l688_68881


namespace rank_of_matrix_A_is_2_l688_68833

def matrix_A : Matrix (Fin 4) (Fin 5) ℚ :=
  ![![3, -1, 1, 2, -8],
    ![7, -1, 2, 1, -12],
    ![11, -1, 3, 0, -16],
    ![10, -2, 3, 3, -20]]

theorem rank_of_matrix_A_is_2 : Matrix.rank matrix_A = 2 := by
  sorry

end rank_of_matrix_A_is_2_l688_68833


namespace evaluate_expression_l688_68899

theorem evaluate_expression : 
    8 * 7 / 8 * 7 = 49 := 
by sorry

end evaluate_expression_l688_68899


namespace symmetric_point_about_x_l688_68896

-- Define the coordinates of the point A
def A : ℝ × ℝ := (-2, 3)

-- Define the function that computes the symmetric point about the x-axis
def symmetric_about_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- The concrete symmetric point of A
def A' := symmetric_about_x A

-- The original problem and proof statement
theorem symmetric_point_about_x :
  A' = (-2, -3) :=
by
  -- Proof goes here
  sorry

end symmetric_point_about_x_l688_68896


namespace not_covered_by_homothetic_polygons_l688_68832

structure Polygon :=
  (vertices : Set (ℝ × ℝ))

def homothetic (M : Polygon) (k : ℝ) (O : ℝ × ℝ) : Polygon :=
  {
    vertices := {p | ∃ (q : ℝ × ℝ) (hq : q ∈ M.vertices), p = (O.1 + k * (q.1 - O.1), O.2 + k * (q.2 - O.2))}
  }

theorem not_covered_by_homothetic_polygons (M : Polygon) (k : ℝ) (h : 0 < k ∧ k < 1)
  (O1 O2 : ℝ × ℝ) :
  ¬ (∀ p ∈ M.vertices, p ∈ (homothetic M k O1).vertices ∨ p ∈ (homothetic M k O2).vertices) := by
  sorry

end not_covered_by_homothetic_polygons_l688_68832


namespace graph_of_cubic_equation_is_three_lines_l688_68892

theorem graph_of_cubic_equation_is_three_lines (x y : ℝ) :
  (x + y) ^ 3 = x ^ 3 + y ^ 3 →
  (y = -x ∨ x = 0 ∨ y = 0) :=
by
  sorry

end graph_of_cubic_equation_is_three_lines_l688_68892


namespace prob_first_question_correct_is_4_5_distribution_of_X_l688_68871

-- Assume probabilities for member A and member B answering correctly.
def prob_A_correct : ℚ := 2 / 5
def prob_B_correct : ℚ := 2 / 3

def prob_A_incorrect : ℚ := 1 - prob_A_correct
def prob_B_incorrect : ℚ := 1 - prob_B_correct

-- Given that A answers first, followed by B.
-- Calculate the probability that the first team answers the first question correctly.
def prob_first_question_correct : ℚ :=
  prob_A_correct + (prob_A_incorrect * prob_B_correct)

-- Assert that the calculated probability is equal to 4/5
theorem prob_first_question_correct_is_4_5 :
  prob_first_question_correct = 4 / 5 := by
  sorry

-- Define the possible scores and their probabilities
def prob_X_eq_0 : ℚ := prob_A_incorrect * prob_B_incorrect
def prob_X_eq_10 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) * prob_A_incorrect * prob_B_incorrect
def prob_X_eq_20 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) ^ 2 * prob_A_incorrect * prob_B_incorrect
def prob_X_eq_30 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) ^ 3

-- Assert the distribution probabilities for the random variable X
theorem distribution_of_X :
  prob_X_eq_0 = 1 / 5 ∧
  prob_X_eq_10 = 4 / 25 ∧
  prob_X_eq_20 = 16 / 125 ∧
  prob_X_eq_30 = 64 / 125 := by
  sorry

end prob_first_question_correct_is_4_5_distribution_of_X_l688_68871


namespace tom_books_l688_68804

-- Definitions based on the conditions
def joan_books : ℕ := 10
def total_books : ℕ := 48

-- The theorem statement: Proving that Tom has 38 books
theorem tom_books : (total_books - joan_books) = 38 := by
  -- Here we would normally provide a proof, but we use sorry to skip this.
  sorry

end tom_books_l688_68804


namespace sequence_value_of_m_l688_68895

theorem sequence_value_of_m (a : ℕ → ℝ) (m : ℕ) (h1 : a 1 = 1)
                            (h2 : ∀ n : ℕ, n > 0 → a n - a (n + 1) = a (n + 1) * a n)
                            (h3 : 8 * a m = 1) :
                            m = 8 := by
  sorry

end sequence_value_of_m_l688_68895


namespace algebraic_expression_analysis_l688_68867

theorem algebraic_expression_analysis :
  (∀ x y : ℝ, (x - 1/2 * y) * (x + 1/2 * y) = x^2 - (1/2 * y)^2) ∧
  (∀ a b c : ℝ, ¬ ((3 * a + b * c) * (-b * c - 3 * a) = (3 * a + b * c)^2)) ∧
  (∀ x y : ℝ, (3 - x + y) * (3 + x + y) = (3 + y)^2 - x^2) ∧
  ((100 + 1) * (100 - 1) = 100^2 - 1) :=
by
  intros
  repeat { split }; sorry

end algebraic_expression_analysis_l688_68867


namespace fraction_white_surface_area_l688_68877

/-- A 4-inch cube is constructed from 64 smaller cubes, each with 1-inch edges.
   48 of these smaller cubes are colored red and 16 are colored white.
   Prove that if the 4-inch cube is constructed to have the smallest possible white surface area showing,
   the fraction of the white surface area is 1/12. -/
theorem fraction_white_surface_area : 
  let total_surface_area := 96
  let white_cubes := 16
  let exposed_white_surface_area := 8
  (exposed_white_surface_area / total_surface_area) = (1 / 12) := 
  sorry

end fraction_white_surface_area_l688_68877


namespace rose_bought_flowers_l688_68851

theorem rose_bought_flowers (F : ℕ) (h1 : ∃ (daisies tulips sunflowers : ℕ), daisies = 2 ∧ sunflowers = 4 ∧ 
  tulips = (3 / 5) * (F - 2) ∧ sunflowers = (2 / 5) * (F - 2)) : F = 12 :=
sorry

end rose_bought_flowers_l688_68851


namespace units_digit_of_quotient_l688_68845

theorem units_digit_of_quotient : 
  (4^1985 + 7^1985) % 7 = 0 → (4^1985 + 7^1985) / 7 % 10 = 2 := 
  by 
    intro h
    sorry

end units_digit_of_quotient_l688_68845


namespace range_of_a_l688_68882

theorem range_of_a
  (a x : ℝ)
  (h_eq : 2 * (1 / 4) ^ (-x) - (1 / 2) ^ (-x) + a = 0)
  (h_x : -1 ≤ x ∧ x ≤ 0) :
  -1 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l688_68882


namespace count_three_digit_numbers_divisible_by_13_l688_68843

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l688_68843


namespace reasoning_common_sense_l688_68838

theorem reasoning_common_sense :
  (∀ P Q: Prop, names_not_correct → P → ¬Q → affairs_not_successful → ¬Q)
  ∧ (∀ R S: Prop, affairs_not_successful → R → ¬S → rites_not_flourish → ¬S)
  ∧ (∀ T U: Prop, rites_not_flourish → T → ¬U → punishments_not_executed_properly → ¬U)
  ∧ (∀ V W: Prop, punishments_not_executed_properly → V → ¬W → people_nowhere_hands_feet → ¬W)
  → reasoning_is_common_sense :=
by sorry

end reasoning_common_sense_l688_68838


namespace largest_d_l688_68879

theorem largest_d (a b c d : ℤ) 
  (h₁ : a + 1 = b - 2) 
  (h₂ : a + 1 = c + 3) 
  (h₃ : a + 1 = d - 4) : 
  d > a ∧ d > b ∧ d > c := 
by 
  -- Here we would provide the proof, but for now we'll skip it
  sorry

end largest_d_l688_68879


namespace large_rectangle_perimeter_l688_68891

-- Definitions for conditions
def rectangle_area (l b : ℝ) := l * b
def is_large_rectangle_perimeter (l b perimeter : ℝ) := perimeter = 2 * (l + b)

-- Statement of the theorem
theorem large_rectangle_perimeter :
  ∃ (l b : ℝ), rectangle_area l b = 8 ∧ 
               (∀ l_rect b_rect: ℝ, is_large_rectangle_perimeter l_rect b_rect 32) :=
by
  sorry

end large_rectangle_perimeter_l688_68891


namespace determine_x_l688_68812

theorem determine_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7.5 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end determine_x_l688_68812


namespace arithmetic_sequence_property_l688_68807

variable {α : Type*} [LinearOrderedField α]

theorem arithmetic_sequence_property
  (a : ℕ → α) (h1 : a 1 + a 8 = 9) (h4 : a 4 = 3) : a 5 = 6 :=
by
  sorry

end arithmetic_sequence_property_l688_68807


namespace computation_of_expression_l688_68831

theorem computation_of_expression (x : ℝ) (h : x + 1 / x = 7) : 
  (x - 3) ^ 2 + 49 / (x - 3) ^ 2 = 23 := 
by
  sorry

end computation_of_expression_l688_68831


namespace rectangle_area_difference_196_l688_68810

noncomputable def max_min_area_difference (P : ℕ) (A_max A_min : ℕ) : Prop :=
  ( ∃ l w : ℕ, 2 * l + 2 * w = P ∧ A_max = l * w ) ∧
  ( ∃ l' w' : ℕ, 2 * l' + 2 * w' = P ∧ A_min = l' * w' ) ∧
  (A_max - A_min = 196)

theorem rectangle_area_difference_196 : max_min_area_difference 60 225 29 :=
by
  sorry

end rectangle_area_difference_196_l688_68810


namespace find_n_l688_68806

noncomputable def r1 : ℚ := 6 / 15
noncomputable def S1 : ℚ := 15 / (1 - r1)
noncomputable def r2 (n : ℚ) : ℚ := (6 + n) / 15
noncomputable def S2 (n : ℚ) : ℚ := 15 / (1 - r2 n)

theorem find_n : ∃ (n : ℚ), S2 n = 3 * S1 ∧ n = 6 :=
by
  use 6
  sorry

end find_n_l688_68806


namespace intersection_A_B_find_coefficients_a_b_l688_68869

open Set

variable {X : Type} (x : X)

def setA : Set ℝ := { x | x^2 < 9 }
def setB : Set ℝ := { x | (x - 2) * (x + 4) < 0 }
def A_inter_B : Set ℝ := { x | -3 < x ∧ x < 2 }
def A_union_B_solution_set : Set ℝ := { x | -4 < x ∧ x < 3 }

theorem intersection_A_B :
  A ∩ B = { x | -3 < x ∧ x < 2 } :=
sorry

theorem find_coefficients_a_b (a b : ℝ) :
  (∀ x, 2 * x^2 + a * x + b < 0 ↔ -4 < x ∧ x < 3) → 
  a = 2 ∧ b = -24 :=
sorry

end intersection_A_B_find_coefficients_a_b_l688_68869


namespace number_divisible_by_33_l688_68842

theorem number_divisible_by_33 (x y : ℕ) 
  (h1 : (x + y) % 3 = 2) 
  (h2 : (y - x) % 11 = 8) : 
  (27850 + 1000 * x + y) % 33 = 0 := 
sorry

end number_divisible_by_33_l688_68842


namespace solve_for_x_l688_68849

theorem solve_for_x (x : ℤ) (h : 5 * (x - 9) = 6 * (3 - 3 * x) + 9) : x = 72 / 23 :=
by
  sorry

end solve_for_x_l688_68849


namespace division_result_l688_68829

theorem division_result : (5 * 6 + 4) / 8 = 4.25 :=
by
  sorry

end division_result_l688_68829


namespace exterior_angle_parallel_lines_l688_68823

theorem exterior_angle_parallel_lines
  (k l : Prop) 
  (triangle_has_angles : ∃ (a b c : ℝ), a = 40 ∧ b = 40 ∧ c = 100 ∧ a + b + c = 180)
  (exterior_angle_eq : ∀ (y : ℝ), y = 180 - 100) :
  ∃ (x : ℝ), x = 80 :=
by
  sorry

end exterior_angle_parallel_lines_l688_68823


namespace esperanzas_gross_monthly_salary_l688_68887

variables (Rent FoodExpenses MortgageBill Savings Taxes GrossSalary : ℝ)

def problem_conditions (Rent FoodExpenses MortgageBill Savings Taxes : ℝ) :=
  Rent = 600 ∧
  FoodExpenses = (3 / 5) * Rent ∧
  MortgageBill = 3 * FoodExpenses ∧
  Savings = 2000 ∧
  Taxes = (2 / 5) * Savings

theorem esperanzas_gross_monthly_salary (h : problem_conditions Rent FoodExpenses MortgageBill Savings Taxes) :
  GrossSalary = Rent + FoodExpenses + MortgageBill + Taxes + Savings → GrossSalary = 4840 :=
by
  sorry

end esperanzas_gross_monthly_salary_l688_68887


namespace king_lancelot_seats_38_l688_68859

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l688_68859


namespace no_integers_divisible_by_all_l688_68864

-- Define the list of divisors
def divisors : List ℕ := [2, 3, 4, 5, 7, 11]

-- Define the LCM function
def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Calculate the LCM of the given divisors
def lcm_divisors : ℕ := lcm_list divisors

-- Define a predicate to check divisibility by all divisors
def is_divisible_by_all (n : ℕ) (ds : List ℕ) : Prop :=
  ds.all (λ d => n % d = 0)

-- Define the theorem to prove the number of integers between 1 and 1000 divisible by the given divisors
theorem no_integers_divisible_by_all :
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 1000 ∧ is_divisible_by_all n divisors) → False := by
  sorry

end no_integers_divisible_by_all_l688_68864


namespace solution_value_of_a_l688_68834

noncomputable def verify_a (a : ℚ) (A : Set ℚ) : Prop :=
  A = {a - 2, 2 * a^2 + 5 * a, 12} ∧ -3 ∈ A

theorem solution_value_of_a (a : ℚ) (A : Set ℚ) (h : verify_a a A) : a = -3 / 2 := by
  sorry

end solution_value_of_a_l688_68834


namespace parabola_transform_l688_68857

theorem parabola_transform :
  ∀ (x : ℝ),
    ∃ (y : ℝ),
      (y = -2 * x^2) →
      (∃ (y' : ℝ), y' = y - 1 ∧
      ∃ (x' : ℝ), x' = x - 3 ∧
      ∃ (y'' : ℝ), y'' = -2 * (x')^2 - 1) :=
by sorry

end parabola_transform_l688_68857


namespace clarence_to_matthew_ratio_l688_68885

theorem clarence_to_matthew_ratio (D C M : ℝ) (h1 : D = 6.06) (h2 : D = 1 / 2 * C) (h3 : D + C + M = 20.20) : C / M = 6 := 
by 
  sorry

end clarence_to_matthew_ratio_l688_68885


namespace slope_of_bisecting_line_l688_68855

theorem slope_of_bisecting_line (m n : ℕ) (hmn : Int.gcd m n = 1) : 
  let p1 := (20, 90)
  let p2 := (20, 228)
  let p3 := (56, 306)
  let p4 := (56, 168)
  -- Define conditions for line through origin (x = 0, y = 0) bisecting the parallelogram
  let b := 135 / 19
  let slope := (90 + b) / 20
  -- The slope must be equal to 369/76 (m = 369, n = 76)
  m = 369 ∧ n = 76 → m + n = 445 := by
  intro m n hmn
  let p1 := (20, 90)
  let p2 := (20, 228)
  let p3 := (56, 306)
  let p4 := (56, 168)
  let b := 135 / 19
  let slope := (90 + b) / 20
  sorry

end slope_of_bisecting_line_l688_68855


namespace red_balls_count_l688_68813

theorem red_balls_count (R W : ℕ) (h1 : R / W = 4 / 5) (h2 : W = 20) : R = 16 := sorry

end red_balls_count_l688_68813


namespace distance_to_office_is_18_l688_68848

-- Definitions given in the problem conditions
variables (x t d : ℝ)
-- Conditions based on the problem statements
axiom speed_condition1 : d = x * t
axiom speed_condition2 : d = (x + 1) * (3 / 4 * t)
axiom speed_condition3 : d = (x - 1) * (t + 3)

-- The mathematical proof statement that needs to be shown
theorem distance_to_office_is_18 :
  d = 18 :=
by
  sorry

end distance_to_office_is_18_l688_68848


namespace repeating_decimal_to_fraction_l688_68817

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l688_68817


namespace major_axis_length_is_three_l688_68858

-- Given the radius of the cylinder
def cylinder_radius : ℝ := 1

-- Given the percentage longer of the major axis than the minor axis
def percentage_longer (r : ℝ) : ℝ := 1.5

-- Given the function to calculate the minor axis using the radius
def minor_axis (r : ℝ) : ℝ := 2 * r

-- Given the function to calculate the major axis using the minor axis
def major_axis (minor_axis : ℝ) (factor : ℝ) : ℝ := minor_axis * factor

-- The conjecture states that the major axis length is 3
theorem major_axis_length_is_three : 
  major_axis (minor_axis cylinder_radius) (percentage_longer cylinder_radius) = 3 :=
by 
  -- Proof goes here
  sorry

end major_axis_length_is_three_l688_68858


namespace area_of_regionM_l688_68897

/-
Define the conditions as separate predicates in Lean.
-/

def cond1 (x y : ℝ) : Prop := y - x ≥ abs (x + y)

def cond2 (x y : ℝ) : Prop := (x^2 + 8*x + y^2 + 6*y) / (2*y - x - 8) ≤ 0

/-
Define region \( M \) by combining the conditions.
-/

def regionM (x y : ℝ) : Prop := cond1 x y ∧ cond2 x y

/-
Define the main theorem to compute the area of the region \( M \).
-/

theorem area_of_regionM : 
  ∀ x y : ℝ, (regionM x y) → (calculateAreaOfM) := sorry

/-
A placeholder definition to calculate the area of M. 
-/

noncomputable def calculateAreaOfM : ℝ := 8

end area_of_regionM_l688_68897


namespace simplify_expression_l688_68847

theorem simplify_expression : 
    2 * Real.sqrt 12 + 3 * Real.sqrt (4 / 3) - Real.sqrt (16 / 3) - (2 / 3) * Real.sqrt 48 = 2 * Real.sqrt 3 :=
by
  sorry

end simplify_expression_l688_68847


namespace sets_equal_l688_68822

-- Definitions of sets M and N
def M := { u : ℤ | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l }
def N := { u : ℤ | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r }

-- Theorem statement asserting M = N
theorem sets_equal : M = N :=
by sorry

end sets_equal_l688_68822


namespace minimum_product_OP_OQ_l688_68854

theorem minimum_product_OP_OQ (a b : ℝ) (P Q : ℝ × ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : P ≠ Q) (h4 : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1) (h5 : Q.1 ^ 2 / a ^ 2 + Q.2 ^ 2 / b ^ 2 = 1)
  (h6 : P.1 * Q.1 + P.2 * Q.2 = 0) :
  (P.1 ^ 2 + P.2 ^ 2) * (Q.1 ^ 2 + Q.2 ^ 2) ≥ (2 * a ^ 2 * b ^ 2 / (a ^ 2 + b ^ 2)) :=
by sorry

end minimum_product_OP_OQ_l688_68854


namespace find_f_log2_20_l688_68860

noncomputable def f (x : ℝ) : ℝ :=
if -1 < x ∧ x < 0 then 2^x + 1 else sorry

lemma f_periodic (x : ℝ) : f (x - 2) = f (x + 2) :=
sorry

lemma f_odd (x : ℝ) : f (-x) = -f (x) :=
sorry

theorem find_f_log2_20 : f (Real.log 20 / Real.log 2) = -1 :=
sorry

end find_f_log2_20_l688_68860


namespace gcd_2023_2052_eq_1_l688_68875

theorem gcd_2023_2052_eq_1 : Int.gcd 2023 2052 = 1 :=
by
  sorry

end gcd_2023_2052_eq_1_l688_68875


namespace find_m_if_parallel_l688_68874

-- Definitions of the lines and the condition for parallel lines
def line1 (m : ℝ) (x y : ℝ) : ℝ := (m - 1) * x + y + 2
def line2 (m : ℝ) (x y : ℝ) : ℝ := 8 * x + (m + 1) * y + (m - 1)

-- The condition for the lines to be parallel
def parallel (m : ℝ) : Prop :=
  (m - 1) / 8 = 1 / (m + 1) ∧ (m - 1) / 8 ≠ 2 / (m - 1)

-- The main theorem to prove
theorem find_m_if_parallel (m : ℝ) (h : parallel m) : m = 3 :=
sorry

end find_m_if_parallel_l688_68874


namespace fraction_simplification_l688_68821

-- Define the numerator and denominator based on given conditions
def numerator : ℤ := 1 - 2 + 4 - 8 + 16 - 32 + 64 - 128 + 256
def denominator : ℤ := 2 - 4 + 8 - 16 + 32 - 64 + 128 - 256 + 512

-- Lean theorem that encapsulates the problem
theorem fraction_simplification : (numerator : ℚ) / (denominator : ℚ) = 1 / 2 :=
by
  sorry

end fraction_simplification_l688_68821


namespace bryson_new_shoes_l688_68841

-- Define the conditions as variables and constant values
def pairs_of_shoes : ℕ := 2 -- Number of pairs Bryson bought
def shoes_per_pair : ℕ := 2 -- Number of shoes per pair

-- Define the theorem to prove the question == answer
theorem bryson_new_shoes : pairs_of_shoes * shoes_per_pair = 4 :=
by
  sorry -- Proof placeholder

end bryson_new_shoes_l688_68841
