import Mathlib

namespace range_of_a_l316_31623

variable (a : ℝ)
variable (x y : ℝ)

def system_of_equations := 
  (5 * x + 2 * y = 11 * a + 18) ∧ 
  (2 * x - 3 * y = 12 * a - 8) ∧
  (x > 0) ∧ 
  (y > 0)

theorem range_of_a (h : system_of_equations a x y) : 
  - (2:ℝ) / 3 < a ∧ a < 2 :=
sorry

end range_of_a_l316_31623


namespace conversion_correct_l316_31684

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.enum.foldl (λ acc ⟨i, digit⟩ => acc + digit * 2^i) 0

def n : List ℕ := [1, 0, 1, 1, 1, 1, 0, 1, 1]

theorem conversion_correct :
  binary_to_decimal n = 379 :=
by 
  sorry

end conversion_correct_l316_31684


namespace john_total_spent_l316_31694

/-- John's expenditure calculations -/
theorem john_total_spent :
  let computer_cost := 1500
  let peripherals_cost := computer_cost / 5
  let original_video_card_cost := 300
  let upgraded_video_card_cost := original_video_card_cost * 2
  let additional_upgrade_cost := upgraded_video_card_cost - original_video_card_cost
  let total_spent := computer_cost + peripherals_cost + additional_upgrade_cost
  total_spent = 2100 :=
by
  sorry

end john_total_spent_l316_31694


namespace train_speed_40_l316_31614

-- Definitions for the conditions
def passes_pole (L V : ℝ) := V = L / 8
def passes_stationary_train (L V : ℝ) := V = (L + 400) / 18

-- The theorem we want to prove
theorem train_speed_40 (L V : ℝ) (h1 : passes_pole L V) (h2 : passes_stationary_train L V) : V = 40 := 
sorry

end train_speed_40_l316_31614


namespace rainfall_difference_l316_31636

noncomputable def r₁ : ℝ := 26
noncomputable def r₂ : ℝ := 34
noncomputable def r₃ : ℝ := r₂ - 12
noncomputable def avg : ℝ := 140

theorem rainfall_difference : (avg - (r₁ + r₂ + r₃)) = 58 := 
by
  sorry

end rainfall_difference_l316_31636


namespace nonneg_integer_representation_l316_31627

theorem nonneg_integer_representation (n : ℕ) : 
  ∃ x y : ℕ, n = (x + y) * (x + y) + 3 * x + y / 2 := 
sorry

end nonneg_integer_representation_l316_31627


namespace articles_selling_price_to_cost_price_eq_l316_31660

theorem articles_selling_price_to_cost_price_eq (C N : ℝ) (h_gain : 2 * C * N = 20 * C) : N = 10 :=
by
  sorry

end articles_selling_price_to_cost_price_eq_l316_31660


namespace initial_members_count_l316_31674

theorem initial_members_count (n : ℕ) (W : ℕ)
  (h1 : W = n * 48)
  (h2 : W + 171 = (n + 2) * 51) : 
  n = 23 :=
by sorry

end initial_members_count_l316_31674


namespace two_square_numbers_difference_133_l316_31612

theorem two_square_numbers_difference_133 : 
  ∃ (x y : ℤ), x^2 - y^2 = 133 ∧ ((x = 67 ∧ y = 66) ∨ (x = 13 ∧ y = 6)) :=
by {
  sorry
}

end two_square_numbers_difference_133_l316_31612


namespace part1_proof_l316_31690

variable (a r : ℝ) (f : ℝ → ℝ)

axiom a_gt_1 : a > 1
axiom r_gt_1 : r > 1

axiom f_condition : ∀ x > 0, f x * f x ≤ a * x * f (x / a)
axiom f_bound : ∀ x, 0 < x ∧ x < 1 / 2^2005 → f x < 2^2005

theorem part1_proof : ∀ x > 0, f x ≤ a^(1 - r) * x := 
by 
  sorry

end part1_proof_l316_31690


namespace perpendicular_lines_condition_l316_31604

theorem perpendicular_lines_condition (m : ℝ) : (m = -1) ↔ ∀ (x y : ℝ), (x + y = 0) ∧ (x + m * y = 0) → 
  ((m ≠ 0) ∧ (-1) * (-1 / m) = 1) :=
by 
  sorry

end perpendicular_lines_condition_l316_31604


namespace polynomial_identity_l316_31659

theorem polynomial_identity (a : ℝ) (h₁ : a^5 + 5 * a^4 + 10 * a^3 + 3 * a^2 - 9 * a - 6 = 0) (h₂ : a ≠ -1) : (a + 1)^3 = 7 :=
sorry

end polynomial_identity_l316_31659


namespace total_money_made_l316_31683

structure Building :=
(floors : Nat)
(rooms_per_floor : Nat)

def cleaning_time_per_room : Nat := 8

structure CleaningRates :=
(first_4_hours_rate : Int)
(next_4_hours_rate : Int)
(unpaid_break_hours : Nat)

def supply_cost : Int := 1200

def total_earnings (b : Building) (c : CleaningRates) : Int :=
  let rooms := b.floors * b.rooms_per_floor
  let earnings_per_room := (4 * c.first_4_hours_rate + 4 * c.next_4_hours_rate)
  rooms * earnings_per_room - supply_cost

theorem total_money_made (b : Building) (c : CleaningRates) : 
  b.floors = 12 →
  b.rooms_per_floor = 25 →
  cleaning_time_per_room = 8 →
  c.first_4_hours_rate = 20 →
  c.next_4_hours_rate = 25 →
  c.unpaid_break_hours = 1 →
  total_earnings b c = 52800 := 
by
  intros
  sorry

end total_money_made_l316_31683


namespace cost_per_vent_l316_31655

/--
Given that:
1. The total cost of the HVAC system is $20,000.
2. The system includes 2 conditioning zones.
3. Each zone has 5 vents.

Prove that the cost per vent is $2000.
-/
theorem cost_per_vent (total_cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h1 : total_cost = 20000) (h2 : zones = 2) (h3 : vents_per_zone = 5) :
  total_cost / (zones * vents_per_zone) = 2000 := 
sorry

end cost_per_vent_l316_31655


namespace gcd_problem_l316_31699

theorem gcd_problem 
  (b : ℤ) 
  (hb_odd : b % 2 = 1) 
  (hb_multiples_of_8723 : ∃ (k : ℤ), b = 8723 * k) : 
  Int.gcd (8 * b ^ 2 + 55 * b + 144) (4 * b + 15) = 3 := 
by 
  sorry

end gcd_problem_l316_31699


namespace brody_battery_fraction_l316_31680

theorem brody_battery_fraction (full_battery : ℕ) (battery_left_after_exam : ℕ) (exam_duration : ℕ) 
  (battery_before_exam : ℕ) (battery_used : ℕ) (fraction_used : ℚ) 
  (h1 : full_battery = 60)
  (h2 : battery_left_after_exam = 13)
  (h3 : exam_duration = 2)
  (h4 : battery_before_exam = battery_left_after_exam + exam_duration)
  (h5 : battery_used = full_battery - battery_before_exam)
  (h6 : fraction_used = battery_used / full_battery) :
  fraction_used = 3 / 4 := 
sorry

end brody_battery_fraction_l316_31680


namespace fraction_of_historical_fiction_new_releases_l316_31635

theorem fraction_of_historical_fiction_new_releases (total_books : ℕ) (p1 p2 p3 : ℕ) (frac_hist_fic : Rat) (frac_new_hist_fic : Rat) (frac_new_non_hist_fic : Rat) 
  (h1 : total_books > 0) (h2 : frac_hist_fic = 40 / 100) (h3 : frac_new_hist_fic = 40 / 100) (h4 : frac_new_non_hist_fic = 40 / 100) 
  (h5 : p1 = frac_hist_fic * total_books) (h6 : p2 = frac_new_hist_fic * p1) (h7 : p3 = frac_new_non_hist_fic * (total_books - p1)) :
  p2 / (p2 + p3) = 2 / 5 :=
by
  sorry

end fraction_of_historical_fiction_new_releases_l316_31635


namespace solve_for_x_l316_31616

theorem solve_for_x (x : ℚ) (h : (3 * x + 5) / 7 = 13) : x = 86 / 3 :=
sorry

end solve_for_x_l316_31616


namespace f_even_l316_31652

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_even (a : ℝ) (h1 : is_even f) (h2 : ∀ x, -1 ≤ x ∧ x ≤ a) : f a = 2 :=
  sorry

end f_even_l316_31652


namespace work_completion_days_l316_31672

theorem work_completion_days (a b : Type) (T : ℕ) (ha : T = 12) (hb : T = 6) : 
  (T = 4) :=
sorry

end work_completion_days_l316_31672


namespace twice_brother_age_l316_31642

theorem twice_brother_age (current_my_age : ℕ) (current_brother_age : ℕ) (years : ℕ) :
  current_my_age = 20 →
  (current_my_age + years) + (current_brother_age + years) = 45 →
  current_my_age + years = 2 * (current_brother_age + years) →
  years = 10 :=
by 
  intros h1 h2 h3
  sorry

end twice_brother_age_l316_31642


namespace find_AD_l316_31692

noncomputable def A := 0
noncomputable def C := 3
noncomputable def B (x : ℝ) := C - x
noncomputable def D (x : ℝ) := A + 3 + x

-- conditions
def AC := 3
def BD := 4
def ratio_condition (x : ℝ) := (A + C - x - (A + 3)) / x = (A + 3 + x) / x

-- theorem statement
theorem find_AD (x : ℝ) (h1 : AC = 3) (h2 : BD = 4) (h3 : ratio_condition x) :
  D x = 6 :=
sorry

end find_AD_l316_31692


namespace remainder_div_P_by_D_plus_D_l316_31698

theorem remainder_div_P_by_D_plus_D' 
  (P Q D R D' Q' R' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  P % (D + D') = R :=
by
  -- Proof is not required.
  sorry

end remainder_div_P_by_D_plus_D_l316_31698


namespace third_term_arithmetic_sequence_l316_31667

variable (a d : ℤ)
variable (h1 : a + 20 * d = 12)
variable (h2 : a + 21 * d = 15)

theorem third_term_arithmetic_sequence : a + 2 * d = -42 := by
  sorry

end third_term_arithmetic_sequence_l316_31667


namespace fraction_of_yellow_balls_l316_31666

theorem fraction_of_yellow_balls
  (total_balls : ℕ)
  (fraction_green : ℚ)
  (fraction_blue : ℚ)
  (number_blue : ℕ)
  (number_white : ℕ)
  (total_balls_eq : total_balls = number_blue * (1 / fraction_blue))
  (fraction_green_eq : fraction_green = 1 / 4)
  (fraction_blue_eq : fraction_blue = 1 / 8)
  (number_white_eq : number_white = 26)
  (number_blue_eq : number_blue = 6) :
  (total_balls - (total_balls * fraction_green + number_blue + number_white)) / total_balls = 1 / 12 :=
by
  sorry

end fraction_of_yellow_balls_l316_31666


namespace math_proof_problem_l316_31606

variable (a_n : ℕ → ℤ) (S_n : ℕ → ℤ)
variable (a_1 d : ℤ)
variable (n : ℕ)

def arith_seq : Prop := ∀ n, a_n n = a_1 + (n - 1) * d

def sum_arith_seq : Prop := ∀ n, S_n n = n * (a_1 + (n - 1) * d / 2)

def condition1 : Prop := a_n 5 + a_n 9 = -2

def condition2 : Prop := S_n 3 = 57

noncomputable def general_formula : Prop := ∀ n, a_n n = 27 - 4 * n

noncomputable def max_S_n : Prop := ∀ n, S_n n ≤ 78 ∧ ∃ n, S_n n = 78

theorem math_proof_problem : 
  arith_seq a_n a_1 d ∧ sum_arith_seq S_n a_1 d ∧ condition1 a_n ∧ condition2 S_n 
  → general_formula a_n ∧ max_S_n S_n := 
sorry

end math_proof_problem_l316_31606


namespace midpoint_of_segment_l316_31681

theorem midpoint_of_segment (a b : ℝ) : (a + b) / 2 = (a + b) / 2 :=
sorry

end midpoint_of_segment_l316_31681


namespace tailor_cut_difference_l316_31625

theorem tailor_cut_difference :
  (7 / 8 + 11 / 12) - (5 / 6 + 3 / 4) = 5 / 24 :=
by
  sorry

end tailor_cut_difference_l316_31625


namespace circle_equation_line_equation_l316_31611

theorem circle_equation (a b r x y : ℝ) (h1 : a + b = 2 * x + y)
  (h2 : (a, 2*a - 2) = ((1, 2) : ℝ × ℝ))
  (h3 : (a, 2*a - 2) = ((2, 1) : ℝ × ℝ)) :
  (x - 2) ^ 2 + (y - 2) ^ 2 = 1 := sorry

theorem line_equation (x y m : ℝ) (h1 : y + 3 = (x - (-3)) * ((-3) - 0) / (m - (-3)))
  (h2 : (x, y) = (m, 0) ∨ (x, y) = (m, 0))
  (h3 : (m = 1 ∨ m = - 3 / 4)) :
  (3 * x + 4 * y - 3 = 0) ∨ (4 * x + 3 * y + 3 = 0) := sorry

end circle_equation_line_equation_l316_31611


namespace lattice_point_condition_l316_31638

theorem lattice_point_condition (b : ℚ) :
  (∀ (m : ℚ), (1 / 3 < m ∧ m < b) →
    ∀ x : ℤ, (0 < x ∧ x ≤ 200) →
      ¬ ∃ y : ℤ, y = m * x + 3) →
  b = 68 / 203 := 
sorry

end lattice_point_condition_l316_31638


namespace students_speaking_Gujarati_l316_31634

theorem students_speaking_Gujarati 
  (total_students : ℕ)
  (students_Hindi : ℕ)
  (students_Marathi : ℕ)
  (students_two_languages : ℕ)
  (students_all_three_languages : ℕ)
  (students_total_set: 22 = total_students)
  (students_H_set: 15 = students_Hindi)
  (students_M_set: 6 = students_Marathi)
  (students_two_set: 2 = students_two_languages)
  (students_all_three_set: 1 = students_all_three_languages) :
  ∃ (students_Gujarati : ℕ), 
  22 = students_Gujarati + 15 + 6 - 2 + 1 ∧ students_Gujarati = 2 :=
by
  sorry

end students_speaking_Gujarati_l316_31634


namespace robot_trajectory_no_intersection_l316_31647

noncomputable def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x
noncomputable def line_equation (x y k : ℝ) : Prop := y = k * (x + 1)

theorem robot_trajectory_no_intersection (k : ℝ) :
  (∀ x y : ℝ, parabola_equation x y → ¬ line_equation x y k) →
  (k > 1 ∨ k < -1) :=
by
  sorry

end robot_trajectory_no_intersection_l316_31647


namespace allison_marbles_l316_31664

theorem allison_marbles (A B C : ℕ) (h1 : B = A + 8) (h2 : C = 3 * B) (h3 : C + A = 136) : 
  A = 28 :=
by
  sorry

end allison_marbles_l316_31664


namespace evaluate_polynomial_at_5_l316_31689

def polynomial (x : ℕ) : ℕ := 3*x^5 - 4*x^4 + 6*x^3 - 2*x^2 - 5*x - 2

theorem evaluate_polynomial_at_5 : polynomial 5 = 7548 := by
  sorry

end evaluate_polynomial_at_5_l316_31689


namespace final_number_is_correct_l316_31695

def initial_number := 9
def doubled_number (x : ℕ) := x * 2
def added_number (x : ℕ) := x + 13
def trebled_number (x : ℕ) := x * 3

theorem final_number_is_correct : trebled_number (added_number (doubled_number initial_number)) = 93 := by
  sorry

end final_number_is_correct_l316_31695


namespace number_of_triangles_for_second_star_l316_31613

theorem number_of_triangles_for_second_star (a b : ℝ) (h₁ : a + b + 90 = 180) (h₂ : 5 * (360 / 5) = 360) :
  360 / (180 - 90 - (360 / 5)) = 20 :=
by
  sorry

end number_of_triangles_for_second_star_l316_31613


namespace five_op_two_l316_31675

-- Definition of the operation
def op (a b : ℝ) := 3 * a + 4 * b

-- The theorem statement
theorem five_op_two : op 5 2 = 23 := by
  sorry

end five_op_two_l316_31675


namespace bacteria_count_correct_l316_31637

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 800

-- Define the doubling time in hours
def doubling_time : ℕ := 3

-- Define the function that calculates the number of bacteria after t hours
noncomputable def bacteria_after (t : ℕ) : ℕ :=
  initial_bacteria * 2 ^ (t / doubling_time)

-- Define the target number of bacteria
def target_bacteria : ℕ := 51200

-- Define the specific time we want to prove the bacteria count equals the target
def specific_time : ℕ := 18

-- Prove that after 18 hours, there will be exactly 51,200 bacteria
theorem bacteria_count_correct : bacteria_after specific_time = target_bacteria :=
  sorry

end bacteria_count_correct_l316_31637


namespace sqrt_72_eq_6_sqrt_2_l316_31662

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end sqrt_72_eq_6_sqrt_2_l316_31662


namespace natural_number_1981_l316_31682

theorem natural_number_1981 (x : ℕ) 
  (h1 : ∃ a : ℕ, x - 45 = a^2)
  (h2 : ∃ b : ℕ, x + 44 = b^2) :
  x = 1981 :=
sorry

end natural_number_1981_l316_31682


namespace max_value_of_x_minus_y_l316_31678

theorem max_value_of_x_minus_y
  (x y : ℝ)
  (h : 2 * (x ^ 2 + y ^ 2 - x * y) = x + y) :
  x - y ≤ 1 / 2 := 
sorry

end max_value_of_x_minus_y_l316_31678


namespace profit_rate_l316_31610

variables (list_price : ℝ)
          (discount : ℝ := 0.95)
          (selling_increase : ℝ := 1.6)
          (inflation_rate : ℝ := 1.4)

theorem profit_rate (list_price : ℝ) : 
  (selling_increase / (discount * inflation_rate)) - 1 = 0.203 :=
by 
  sorry

end profit_rate_l316_31610


namespace opposite_of_neg_half_is_half_l316_31644

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end opposite_of_neg_half_is_half_l316_31644


namespace part_one_part_two_part_three_l316_31658

def f(x : ℝ) := x^2 - 1
def g(a x : ℝ) := a * |x - 1|

-- (I)
theorem part_one (a : ℝ) : 
  ((∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |f x₁| = g a x₁ ∧ |f x₂| = g a x₂) ↔ (a = 0 ∨ a = 2)) :=
sorry

-- (II)
theorem part_two (a : ℝ) : 
  (∀ x : ℝ, f x ≥ g a x) ↔ (a <= -2) :=
sorry

-- (III)
def G(a x : ℝ) := |f x| + g a x

theorem part_three (a : ℝ) (h : a < 0) : 
  (∀ x ∈ [-2, 2], G a x ≤ if a <= -3 then 0 else 3 + a) :=
sorry

end part_one_part_two_part_three_l316_31658


namespace area_of_large_square_l316_31669

theorem area_of_large_square (s : ℝ) (h : 2 * s^2 = 14) : 9 * s^2 = 63 := by
  sorry

end area_of_large_square_l316_31669


namespace Mark_same_color_opposite_foot_l316_31602

variable (shoes : Finset (Σ _ : Fin (14), Bool))

def same_color_opposite_foot_probability (shoes : Finset (Σ _ : Fin (14), Bool)) : ℚ := 
  let total_shoes : ℚ := 28
  let num_black_pairs := 7
  let num_brown_pairs := 4
  let num_gray_pairs := 2
  let num_white_pairs := 1
  let black_pair_prob  := (14 / total_shoes) * (7 / (total_shoes - 1))
  let brown_pair_prob  := (8 / total_shoes) * (4 / (total_shoes - 1))
  let gray_pair_prob   := (4 / total_shoes) * (2 / (total_shoes - 1))
  let white_pair_prob  := (2 / total_shoes) * (1 / (total_shoes - 1))
  black_pair_prob + brown_pair_prob + gray_pair_prob + white_pair_prob

theorem Mark_same_color_opposite_foot (shoes : Finset (Σ _ : Fin (14), Bool)) :
  same_color_opposite_foot_probability shoes = 35 / 189 := 
sorry

end Mark_same_color_opposite_foot_l316_31602


namespace ratio_a_b_c_l316_31639

-- Given condition 14(a^2 + b^2 + c^2) = (a + 2b + 3c)^2
theorem ratio_a_b_c (a b c : ℝ) (h : 14 * (a^2 + b^2 + c^2) = (a + 2 * b + 3 * c)^2) : 
  a / b = 1 / 2 ∧ b / c = 2 / 3 :=
by 
  sorry

end ratio_a_b_c_l316_31639


namespace max_value_is_27_l316_31633

noncomputable def max_value_of_expression (a b c : ℝ) : ℝ :=
  (a - b)^2 + (b - c)^2 + (c - a)^2

theorem max_value_is_27 (a b c : ℝ)
  (h : a^2 + b^2 + c^2 = 9) : max_value_of_expression a b c = 27 :=
by
  sorry

end max_value_is_27_l316_31633


namespace l_shape_area_l316_31696

theorem l_shape_area (P : ℝ) (L : ℝ) (x : ℝ)
  (hP : P = 52) 
  (hL : L = 16) 
  (h_x : L + (L - x) + 2 * (16 - x) = P)
  (h_split : 2 * (16 - x) * x = 120) :
  2 * ((16 - x) * x) = 120 :=
by
  -- This is the proof problem statement
  sorry

end l_shape_area_l316_31696


namespace password_encryption_l316_31615

variables (a b x : ℝ)

theorem password_encryption :
  3 * a * (x^2 - 1) - 3 * b * (x^2 - 1) = 3 * (x + 1) * (x - 1) * (a - b) :=
by sorry

end password_encryption_l316_31615


namespace ones_digit_seven_consecutive_integers_l316_31668

theorem ones_digit_seven_consecutive_integers (k : ℕ) (hk : k % 5 = 1) :
  (k * (k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5) * (k + 6)) % 10 = 0 :=
by
  sorry

end ones_digit_seven_consecutive_integers_l316_31668


namespace repeating_decimal_to_fraction_l316_31653

theorem repeating_decimal_to_fraction : (0.36 : ℝ) = (11 / 30 : ℝ) :=
sorry

end repeating_decimal_to_fraction_l316_31653


namespace length_of_goods_train_l316_31691

/-- The length of the goods train given the conditions of the problem --/
theorem length_of_goods_train
  (speed_passenger_train : ℝ) (speed_goods_train : ℝ) 
  (time_taken_to_pass : ℝ) (length_goods_train : ℝ) :
  speed_passenger_train = 80 / 3.6 →  -- Convert 80 km/h to m/s
  speed_goods_train    = 32 / 3.6 →  -- Convert 32 km/h to m/s
  time_taken_to_pass   = 9 →
  length_goods_train   = 280 → 
  length_goods_train = (speed_passenger_train + speed_goods_train) * time_taken_to_pass := by
    sorry

end length_of_goods_train_l316_31691


namespace items_in_storeroom_l316_31621

-- Conditions definitions
def restocked_items : ℕ := 4458
def sold_items : ℕ := 1561
def total_items_left : ℕ := 3472

-- Statement of the proof
theorem items_in_storeroom : (total_items_left - (restocked_items - sold_items)) = 575 := 
by
  sorry

end items_in_storeroom_l316_31621


namespace arithmetic_sequence_sum_l316_31607

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- The sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (h_arith : is_arithmetic_sequence a d) (h_condition : a 2 + a 10 = 16) :
  a 4 + a 8 = 16 :=
sorry

end arithmetic_sequence_sum_l316_31607


namespace find_p_l316_31618

theorem find_p (f p : ℂ) (w : ℂ) (h1 : f * p - w = 15000) (h2 : f = 8) (h3 : w = 10 + 200 * Complex.I) : 
  p = 1876.25 + 25 * Complex.I := 
sorry

end find_p_l316_31618


namespace floor_of_neg_seven_fourths_l316_31624

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end floor_of_neg_seven_fourths_l316_31624


namespace true_discount_is_52_l316_31640

/-- The banker's gain on a bill due 3 years hence at 15% per annum is Rs. 23.4. -/
def BG : ℝ := 23.4

/-- The rate of interest per annum is 15%. -/
def R : ℝ := 15

/-- The time in years is 3. -/
def T : ℝ := 3

/-- The true discount is Rs. 52. -/
theorem true_discount_is_52 : BG * 100 / (R * T) = 52 :=
by
  -- Placeholder for proof. This needs proper calculation.
  sorry

end true_discount_is_52_l316_31640


namespace exist_non_special_symmetric_concat_l316_31685

-- Define the notion of a binary series being symmetric
def is_symmetric (xs : List Bool) : Prop :=
  ∀ i, i < xs.length → xs.get? i = xs.get? (xs.length - 1 - i)

-- Define the notion of a binary series being special
def is_special (xs : List Bool) : Prop :=
  (∀ x ∈ xs, x) ∨ (∀ x ∈ xs, ¬x)

-- The main theorem statement
theorem exist_non_special_symmetric_concat (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  ∃ (A B : List Bool), A.length = m ∧ B.length = n ∧ ¬is_special A ∧ ¬is_special B ∧ is_symmetric (A ++ B) :=
sorry

end exist_non_special_symmetric_concat_l316_31685


namespace sqrt_condition_l316_31673

theorem sqrt_condition (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
sorry

end sqrt_condition_l316_31673


namespace total_songs_isabel_bought_l316_31632

theorem total_songs_isabel_bought
  (country_albums pop_albums : ℕ)
  (songs_per_album : ℕ)
  (h1 : country_albums = 6)
  (h2 : pop_albums = 2)
  (h3 : songs_per_album = 9) : 
  (country_albums + pop_albums) * songs_per_album = 72 :=
by
  -- We provide only the statement, no proof as per the instruction
  sorry

end total_songs_isabel_bought_l316_31632


namespace range_of_b_over_a_l316_31693

-- Define the problem conditions and conclusion
theorem range_of_b_over_a 
  (a b c : ℝ) (A B C : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) 
  (h_sum_angles : A + B + C = π) 
  (h_sides_relation : ∀ x, (x^2 + c^2 - a^2 - ab = 0 ↔ x = 0)) : 
  1 < b / a ∧ b / a < 2 := 
sorry

end range_of_b_over_a_l316_31693


namespace simplify_fraction_l316_31630

theorem simplify_fraction :
  (144 : ℤ) / (1296 : ℤ) = 1 / 9 := 
by sorry

end simplify_fraction_l316_31630


namespace area_of_triangle_KBC_l316_31629

noncomputable def length_FE := 7
noncomputable def length_BC := 7
noncomputable def length_JB := 5
noncomputable def length_BK := 5

theorem area_of_triangle_KBC : (1 / 2 : ℝ) * length_BC * length_BK = 17.5 := by
  -- conditions: 
  -- 1. Hexagon ABCDEF is equilateral with each side of length s.
  -- 2. Squares ABJI and FEHG are formed outside the hexagon with areas 25 and 49 respectively.
  -- 3. Triangle JBK is equilateral.
  -- 4. FE = BC.
  sorry

end area_of_triangle_KBC_l316_31629


namespace problem_l316_31650

variable (x y : ℝ)

-- Define the given condition
def condition : Prop := |x + 5| + (y - 4)^2 = 0

-- State the theorem we need to prove
theorem problem (h : condition x y) : (x + y)^99 = -1 := sorry

end problem_l316_31650


namespace circle_representation_l316_31631

theorem circle_representation (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2 * m * x + 2 * m^2 + 2 * m - 3 = 0) ↔ m ∈ Set.Ioo (-3 : ℝ) (1 / 2 : ℝ) :=
by
  sorry

end circle_representation_l316_31631


namespace trajectory_of_complex_point_l316_31646

open Complex Topology

theorem trajectory_of_complex_point (z : ℂ) (hz : ‖z‖ ≤ 1) : 
  {w : ℂ | ‖w‖ ≤ 1} = {w : ℂ | w.re * w.re + w.im * w.im ≤ 1} :=
sorry

end trajectory_of_complex_point_l316_31646


namespace population_at_300pm_l316_31648

namespace BacteriaProblem

def initial_population : ℕ := 50
def time_increments_to_220pm : ℕ := 4   -- 4 increments of 5 mins each till 2:20 p.m.
def time_increments_to_300pm : ℕ := 2   -- 2 increments of 10 mins each till 3:00 p.m.

def growth_factor_before_220pm : ℕ := 3
def growth_factor_after_220pm : ℕ := 2

theorem population_at_300pm :
  initial_population * growth_factor_before_220pm^time_increments_to_220pm *
  growth_factor_after_220pm^time_increments_to_300pm = 16200 :=
by
  sorry

end BacteriaProblem

end population_at_300pm_l316_31648


namespace average_weight_whole_class_l316_31677

def sectionA_students : Nat := 36
def sectionB_students : Nat := 44
def avg_weight_sectionA : Float := 40.0 
def avg_weight_sectionB : Float := 35.0
def total_weight_sectionA := avg_weight_sectionA * Float.ofNat sectionA_students
def total_weight_sectionB := avg_weight_sectionB * Float.ofNat sectionB_students
def total_students := sectionA_students + sectionB_students
def total_weight := total_weight_sectionA + total_weight_sectionB
def avg_weight_class := total_weight / Float.ofNat total_students

theorem average_weight_whole_class :
  avg_weight_class = 37.25 := by
  sorry

end average_weight_whole_class_l316_31677


namespace sandy_shopping_l316_31643

variable (X : ℝ)

theorem sandy_shopping (h : 0.70 * X = 210) : X = 300 := by
  sorry

end sandy_shopping_l316_31643


namespace calculate_expr_l316_31601

theorem calculate_expr (h1 : Real.sin (30 * Real.pi / 180) = 1 / 2)
    (h2 : Real.cos (30 * Real.pi / 180) = Real.sqrt (3) / 2) :
    3 * Real.tan (30 * Real.pi / 180) + 6 * Real.sin (30 * Real.pi / 180) = 3 + Real.sqrt 3 :=
  sorry

end calculate_expr_l316_31601


namespace simplify_and_evaluate_l316_31651

noncomputable def expr (x : ℝ) : ℝ :=
  (x + 3) * (x - 2) + x * (4 - x)

theorem simplify_and_evaluate (x : ℝ) (hx : x = 2) : expr x = 4 :=
by
  rw [hx]
  show expr 2 = 4
  sorry

end simplify_and_evaluate_l316_31651


namespace kittens_remaining_l316_31620

theorem kittens_remaining (original_kittens : ℕ) (kittens_given_away : ℕ) 
  (h1 : original_kittens = 8) (h2 : kittens_given_away = 4) : 
  original_kittens - kittens_given_away = 4 := by
  sorry

end kittens_remaining_l316_31620


namespace inequality_solution_l316_31676

open Set

def f (x : ℝ) : ℝ := |x| + x^2 + 2

def solution_set : Set ℝ := { x | x < -2 ∨ x > 4 / 3 }

theorem inequality_solution :
  { x : ℝ | f (2 * x - 1) > f (3 - x) } = solution_set := by
  sorry

end inequality_solution_l316_31676


namespace mike_total_cans_l316_31649

theorem mike_total_cans (monday_cans : ℕ) (tuesday_cans : ℕ) (total_cans : ℕ) : 
  monday_cans = 71 ∧ tuesday_cans = 27 ∧ total_cans = monday_cans + tuesday_cans → total_cans = 98 :=
by
  sorry

end mike_total_cans_l316_31649


namespace combined_total_l316_31609

-- Definitions for the problem conditions
def marks_sandcastles : ℕ := 20
def towers_per_marks_sandcastle : ℕ := 10

def jeffs_multiplier : ℕ := 3
def towers_per_jeffs_sandcastle : ℕ := 5

-- Definitions derived from conditions
def jeffs_sandcastles : ℕ := jeffs_multiplier * marks_sandcastles
def marks_towers : ℕ := marks_sandcastles * towers_per_marks_sandcastle
def jeffs_towers : ℕ := jeffs_sandcastles * towers_per_jeffs_sandcastle

-- Question translated to a Lean theorem
theorem combined_total : 
  (marks_sandcastles + jeffs_sandcastles) + (marks_towers + jeffs_towers) = 580 :=
by
  -- The proof would go here
  sorry

end combined_total_l316_31609


namespace mats_length_l316_31665

open Real

theorem mats_length (r : ℝ) (n : ℤ) (w : ℝ) (y : ℝ) (h₁ : r = 6) (h₂ : n = 8) (h₃ : w = 1):
  y = 6 * sqrt (2 - sqrt 2) :=
sorry

end mats_length_l316_31665


namespace age_of_50th_student_l316_31603

theorem age_of_50th_student (avg_50_students : ℝ) (total_students : ℕ)
                           (avg_15_students : ℝ) (group_1_count : ℕ)
                           (avg_15_students_2 : ℝ) (group_2_count : ℕ)
                           (avg_10_students : ℝ) (group_3_count : ℕ)
                           (avg_9_students : ℝ) (group_4_count : ℕ) :
                           avg_50_students = 20 → total_students = 50 →
                           avg_15_students = 18 → group_1_count = 15 →
                           avg_15_students_2 = 22 → group_2_count = 15 →
                           avg_10_students = 25 → group_3_count = 10 →
                           avg_9_students = 24 → group_4_count = 9 →
                           ∃ (age_50th_student : ℝ), age_50th_student = 66 := by
                           sorry

end age_of_50th_student_l316_31603


namespace connie_initial_marbles_l316_31656

theorem connie_initial_marbles (marbles_given : ℕ) (marbles_left : ℕ) (initial_marbles : ℕ) 
    (h1 : marbles_given = 183) (h2 : marbles_left = 593) : initial_marbles = 776 :=
by
  sorry

end connie_initial_marbles_l316_31656


namespace geometric_sequence_sum_l316_31605

theorem geometric_sequence_sum (S : ℕ → ℚ) (a : ℕ → ℚ)
  (h1 : S 4 = 1)
  (h2 : S 8 = 3)
  (h3 : ∀ n, S (n + 4) - S n = a (n + 1) + a (n + 2) + a (n + 3) + a (n + 4)) :
  a 17 + a 18 + a 19 + a 20 = 16 :=
by
  -- Insert your proof here.
  sorry

end geometric_sequence_sum_l316_31605


namespace tire_mileage_l316_31670

theorem tire_mileage (total_miles_driven : ℕ) (x : ℕ) (spare_tire_miles : ℕ):
  total_miles_driven = 40000 →
  spare_tire_miles = 2 * x →
  4 * x + spare_tire_miles = total_miles_driven →
  x = 6667 := 
by
  intros h_total h_spare h_eq
  sorry

end tire_mileage_l316_31670


namespace john_remaining_money_l316_31619

variable (q : ℝ)
variable (number_of_small_pizzas number_of_large_pizzas number_of_drinks : ℕ)
variable (cost_of_drink cost_of_small_pizza cost_of_large_pizza dollars_left : ℝ)

def john_purchases := number_of_small_pizzas = 2 ∧
                      number_of_large_pizzas = 1 ∧
                      number_of_drinks = 4 ∧
                      cost_of_drink = q ∧
                      cost_of_small_pizza = q ∧
                      cost_of_large_pizza = 4 * q ∧
                      dollars_left = 50 - (4 * q + 2 * q + 4 * q)

theorem john_remaining_money : john_purchases q 2 1 4 q q (4 * q) (50 - 10 * q) :=
by
  sorry

end john_remaining_money_l316_31619


namespace Seohyeon_l316_31628

-- Define the distances in their respective units
def d_Kunwoo_km : ℝ := 3.97
def d_Seohyeon_m : ℝ := 4028

-- Convert Kunwoo's distance to meters
def d_Kunwoo_m : ℝ := d_Kunwoo_km * 1000

-- The main theorem we need to prove
theorem Seohyeon's_distance_longer_than_Kunwoo's :
  d_Seohyeon_m > d_Kunwoo_m :=
by
  sorry

end Seohyeon_l316_31628


namespace complex_solution_l316_31688

theorem complex_solution (z : ℂ) (h : z * (0 + 1 * I) = (0 + 1 * I) - 1) : z = 1 + I :=
by
  sorry

end complex_solution_l316_31688


namespace wall_ratio_l316_31661

theorem wall_ratio (V : ℝ) (B : ℝ) (H : ℝ) (x : ℝ) (L : ℝ) :
  V = 12.8 →
  B = 0.4 →
  H = 5 * B →
  L = x * H →
  V = B * H * L →
  x = 4 ∧ L / H = 4 :=
by
  intros hV hB hH hL hVL
  sorry

end wall_ratio_l316_31661


namespace distance_center_to_line_circle_l316_31645

noncomputable def circle_center : ℝ × ℝ := (2, Real.pi / 2)

noncomputable def distance_from_center_to_line (radius : ℝ) (center : ℝ × ℝ) : ℝ :=
  radius * Real.sin (center.snd - Real.pi / 3)

theorem distance_center_to_line_circle : distance_from_center_to_line 2 circle_center = 1 := by
  sorry

end distance_center_to_line_circle_l316_31645


namespace total_cost_28_oranges_avg_cost_per_orange_cost_6_oranges_l316_31600

-- Initial conditions
def cost_4_oranges : Nat := 12
def cost_7_oranges : Nat := 28
def total_oranges : Nat := 28

-- Calculate the total cost for 28 oranges
theorem total_cost_28_oranges
  (x y : Nat) 
  (h1 : 4 * x + 7 * y = total_oranges) 
  (h2 : total_oranges = 28) 
  (h3 : x = 7) 
  (h4 : y = 0) : 
  7 * cost_4_oranges = 84 := 
by sorry

-- Calculate the average cost per orange
theorem avg_cost_per_orange 
  (total_cost : Nat) 
  (h1 : total_cost = 84)
  (h2 : total_oranges = 28) : 
  total_cost / total_oranges = 3 := 
by sorry

-- Calculate the cost for 6 oranges
theorem cost_6_oranges 
  (avg_cost : Nat)
  (h1 : avg_cost = 3)
  (n : Nat) 
  (h2 : n = 6) : 
  n * avg_cost = 18 := 
by sorry

end total_cost_28_oranges_avg_cost_per_orange_cost_6_oranges_l316_31600


namespace yoki_cans_l316_31679

-- Definitions of the conditions
def total_cans_collected : ℕ := 85
def ladonna_cans : ℕ := 25
def prikya_cans : ℕ := 2 * ladonna_cans
def avi_initial_cans : ℕ := 8
def avi_cans := avi_initial_cans / 2

-- Statement that needs to be proved
theorem yoki_cans : ∀ (total_cans_collected ladonna_cans : ℕ) 
  (prikya_cans : ℕ := 2 * ladonna_cans) 
  (avi_initial_cans : ℕ := 8) 
  (avi_cans : ℕ := avi_initial_cans / 2), 
  (total_cans_collected = 85) → 
  (ladonna_cans = 25) → 
  (prikya_cans = 2 * ladonna_cans) →
  (avi_initial_cans = 8) → 
  (avi_cans = avi_initial_cans / 2) → 
  total_cans_collected - (ladonna_cans + prikya_cans + avi_cans) = 6 :=
by
  intros total_cans_collected ladonna_cans prikya_cans avi_initial_cans avi_cans H1 H2 H3 H4 H5
  sorry

end yoki_cans_l316_31679


namespace percentage_more_than_l316_31617

variable (P Q : ℝ)

-- P gets 20% more than Q
def getsMoreThan (P Q : ℝ) : Prop :=
  P = 1.20 * Q

-- Q gets 20% less than P
def getsLessThan (Q P : ℝ) : Prop :=
  Q = 0.80 * P

theorem percentage_more_than :
  getsLessThan Q P → getsMoreThan P Q := 
sorry

end percentage_more_than_l316_31617


namespace remaining_homes_proof_l316_31671

-- Define the total number of homes
def total_homes : ℕ := 200

-- Distributed homes after the first hour
def homes_distributed_first_hour : ℕ := (2 * total_homes) / 5

-- Remaining homes after the first hour
def remaining_homes_first_hour : ℕ := total_homes - homes_distributed_first_hour

-- Distributed homes in the next 2 hours
def homes_distributed_next_two_hours : ℕ := (60 * remaining_homes_first_hour) / 100

-- Remaining homes after the next 2 hours
def homes_remaining : ℕ := remaining_homes_first_hour - homes_distributed_next_two_hours

theorem remaining_homes_proof : homes_remaining = 48 := by
  sorry

end remaining_homes_proof_l316_31671


namespace find_x_l316_31608

theorem find_x :
  ∃ x : ℝ, (0 < x) ∧ (⌊x⌋ * x + x^2 = 93) ∧ (x = 7.10) :=
by {
   sorry
}

end find_x_l316_31608


namespace train_speed_l316_31622

theorem train_speed (v : ℕ) :
    let distance_between_stations := 155
    let speed_of_train_from_A := 20
    let start_time_train_A := 7
    let start_time_train_B := 8
    let meet_time := 11
    let distance_traveled_by_A := speed_of_train_from_A * (meet_time - start_time_train_A)
    let remaining_distance := distance_between_stations - distance_traveled_by_A
    let traveling_time_train_B := meet_time - start_time_train_B
    v * traveling_time_train_B = remaining_distance → v = 25 :=
by
  intros
  sorry

end train_speed_l316_31622


namespace find_a1_a10_value_l316_31687

variable {α : Type} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
∃ r : α, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a1_a10_value (a : ℕ → α) (h1 : is_geometric_sequence a)
    (h2 : a 4 + a 7 = 2) (h3 : a 5 * a 6 = -8) : a 1 + a 10 = -7 := by
  sorry

end find_a1_a10_value_l316_31687


namespace largest_sum_of_three_faces_l316_31686

theorem largest_sum_of_three_faces (faces : Fin 6 → ℕ)
  (h_unique : ∀ i j, i ≠ j → faces i ≠ faces j)
  (h_range : ∀ i, 1 ≤ faces i ∧ faces i ≤ 6)
  (h_opposite_sum : ∀ i, faces i + faces (5 - i) = 10) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ faces i + faces j + faces k = 12 :=
by sorry

end largest_sum_of_three_faces_l316_31686


namespace distance_AB_polar_l316_31626

open Real

theorem distance_AB_polar (A B : ℝ × ℝ) (θ₁ θ₂ : ℝ) (hA : A = (4, θ₁)) (hB : B = (12, θ₂))
  (hθ : θ₁ - θ₂ = π / 3) : dist (4 * cos θ₁, 4 * sin θ₁) (12 * cos θ₂, 12 * sin θ₂) = 4 * sqrt 13 :=
by
  sorry

end distance_AB_polar_l316_31626


namespace average_weight_increase_l316_31697

noncomputable def average_increase (A : ℝ) : ℝ :=
  let initial_total := 10 * A
  let new_total := initial_total + 25
  let new_average := new_total / 10
  new_average - A

theorem average_weight_increase (A : ℝ) : average_increase A = 2.5 := by
  sorry

end average_weight_increase_l316_31697


namespace max_value_of_quadratic_l316_31654

theorem max_value_of_quadratic :
  ∀ z : ℝ, -6*z^2 + 24*z - 12 ≤ 12 :=
by
  sorry

end max_value_of_quadratic_l316_31654


namespace a_must_not_be_zero_l316_31663

theorem a_must_not_be_zero (a b c d : ℝ) (h₁ : a / b < -3 * (c / d)) (h₂ : b ≠ 0) (h₃ : d ≠ 0) (h₄ : c = 2 * a) : a ≠ 0 :=
sorry

end a_must_not_be_zero_l316_31663


namespace equal_area_of_second_square_l316_31657

/-- 
In an isosceles right triangle with legs of length 25√2 cm, if a square is inscribed such that two 
of its vertices lie on one leg and one vertex on each of the hypotenuse and the other leg, 
and the area of the square is 625 cm², prove that the area of another inscribed square 
(with one vertex each on the hypotenuse and one leg, and two vertices on the other leg) is also 625 cm².
-/
theorem equal_area_of_second_square 
  (a b : ℝ) (h1 : a = 25 * Real.sqrt 2)  
  (h2 : b = 625) :
  ∃ c : ℝ, c = 625 :=
by
  sorry

end equal_area_of_second_square_l316_31657


namespace percent_decrease_first_year_l316_31641

theorem percent_decrease_first_year (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 100) 
  (h_second_year : 0.9 * (100 - x) = 54) : x = 40 :=
by sorry

end percent_decrease_first_year_l316_31641
