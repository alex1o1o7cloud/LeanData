import Mathlib

namespace complete_collection_prob_l1608_160828

noncomputable def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem complete_collection_prob : 
  let n := 18
  let k := 10
  let needed := 6
  let uncollected_combinations := C 6 6
  let collected_combinations := C 12 4
  let total_combinations := C 18 10
  (uncollected_combinations * collected_combinations) / total_combinations = 5 / 442 :=
by 
  sorry

end complete_collection_prob_l1608_160828


namespace wheat_flour_used_l1608_160800

-- Conditions and definitions
def total_flour_used : ℝ := 0.3
def white_flour_used : ℝ := 0.1

-- Statement of the problem
theorem wheat_flour_used : 
  (total_flour_used - white_flour_used) = 0.2 :=
by
  sorry

end wheat_flour_used_l1608_160800


namespace trigonometric_identity_l1608_160804

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 2 / 7 :=
by
  sorry

end trigonometric_identity_l1608_160804


namespace value_of_expression_l1608_160893

theorem value_of_expression (a b : ℤ) (h : a - b = 1) : 3 * a - 3 * b - 4 = -1 :=
by {
  sorry
}

end value_of_expression_l1608_160893


namespace admission_fee_for_children_l1608_160834

theorem admission_fee_for_children (x : ℝ) :
  (∀ (admission_fee_adult : ℝ) (total_people : ℝ) (total_fees_collected : ℝ) (children_admitted : ℝ) (adults_admitted : ℝ),
    admission_fee_adult = 4 ∧
    total_people = 315 ∧
    total_fees_collected = 810 ∧
    children_admitted = 180 ∧
    adults_admitted = total_people - children_admitted ∧
    total_fees_collected = children_admitted * x + adults_admitted * admission_fee_adult
  ) → x = 1.5 := sorry

end admission_fee_for_children_l1608_160834


namespace midpoint_AB_find_Q_find_H_l1608_160801

-- Problem 1: Midpoint of AB
theorem midpoint_AB (x1 y1 x2 y2 : ℝ) : 
  let A := (x1, y1)
  let B := (x2, y2)
  let M := ( (x1 + x2) / 2, (y1 + y2) / 2 )
  M = ( (x1 + x2) / 2, (y1 + y2) / 2 )
:= 
  -- The lean statement that shows the midpoint formula is correct.
  sorry

-- Problem 2: Coordinates of Q given midpoint
theorem find_Q (px py mx my : ℝ) : 
  let P := (px, py)
  let M := (mx, my)
  let Q := (2 * mx - px, 2 * my - py)
  ( (px + Q.1) / 2 = mx ∧ (py + Q.2) / 2 = my )
:= 
  -- Lean statement to find Q
  sorry

-- Problem 3: Coordinates of H given midpoints coinciding
theorem find_H (xE yE xF yF xG yG : ℝ) :
  let E := (xE, yE)
  let F := (xF, yF)
  let G := (xG, yG)
  ∃ xH yH : ℝ, 
    ( (xE + xH) / 2 = (xF + xG) / 2 ∧ (yE + yH) / 2 = (yF + yG) / 2 ) ∨
    ( (xF + xH) / 2 = (xE + xG) / 2 ∧ (yF + yH) / 2 = (yE + yG) / 2 ) ∨
    ( (xG + xH) / 2 = (xE + xF) / 2 ∧ (yG + yH) / 2 = (yE + yF) / 2 )
:=
  -- Lean statement to find H
  sorry

end midpoint_AB_find_Q_find_H_l1608_160801


namespace problem_statement_l1608_160822

noncomputable def a_b (a b : ℚ) : Prop :=
  a + b = 6 ∧ a / b = 6

theorem problem_statement (a b : ℚ) (h : a_b a b) : 
  (a * b - (a - b)) = 6 / 49 :=
by
  sorry

end problem_statement_l1608_160822


namespace fraction_unchanged_when_multiplied_by_3_l1608_160814

variable (x y : ℚ)

theorem fraction_unchanged_when_multiplied_by_3 (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 * x) / (3 * (3 * x + y)) = x / (3 * x + y) :=
by
  sorry

end fraction_unchanged_when_multiplied_by_3_l1608_160814


namespace solve_for_y_l1608_160805

theorem solve_for_y (x : ℝ) (y : ℝ) (h1 : x = 8) (h2 : x^(2*y) = 16) : y = 2/3 :=
by
  sorry

end solve_for_y_l1608_160805


namespace max_value_of_x_plus_y_l1608_160840

variable (x y : ℝ)

-- Conditions
def conditions (x y : ℝ) : Prop := 
  x > 0 ∧ y > 0 ∧ x + y + (1/x) + (1/y) = 5

-- Theorem statement
theorem max_value_of_x_plus_y (x y : ℝ) (h : conditions x y) : x + y ≤ 4 := 
sorry

end max_value_of_x_plus_y_l1608_160840


namespace complex_sum_l1608_160898

open Complex

theorem complex_sum (w : ℂ) (h : w^2 - w + 1 = 0) :
  w^103 + w^104 + w^105 + w^106 + w^107 = -1 :=
sorry

end complex_sum_l1608_160898


namespace track_length_l1608_160850

theorem track_length (y : ℝ) 
  (H1 : ∀ b s : ℝ, b + s = y ∧ b = y / 2 - 120 ∧ s = 120)
  (H2 : ∀ b s : ℝ, b + s = y + 180 ∧ b = y / 2 + 60 ∧ s = y / 2 - 60) :
  y = 600 :=
by 
  sorry

end track_length_l1608_160850


namespace cos_beta_value_l1608_160865

noncomputable def cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) : Real :=
  Real.cos β

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) :
  Real.cos β = 56 / 65 :=
by
  sorry

end cos_beta_value_l1608_160865


namespace inequality_change_l1608_160859

theorem inequality_change (a b : ℝ) (h : a < b) : -2 * a > -2 * b :=
sorry

end inequality_change_l1608_160859


namespace arithmetic_question_l1608_160866

theorem arithmetic_question :
  ((3.25 - 1.57) * 2) = 3.36 :=
by 
  sorry

end arithmetic_question_l1608_160866


namespace necessary_but_not_sufficient_condition_l1608_160883

variable (x y : ℤ)

def p : Prop := x ≠ 2 ∨ y ≠ 4
def q : Prop := x + y ≠ 6

theorem necessary_but_not_sufficient_condition :
  (p x y → q x y) ∧ (¬q x y → ¬p x y) :=
sorry

end necessary_but_not_sufficient_condition_l1608_160883


namespace eggs_eaten_in_afternoon_l1608_160808

theorem eggs_eaten_in_afternoon (initial : ℕ) (morning : ℕ) (final : ℕ) (afternoon : ℕ) :
  initial = 20 → morning = 4 → final = 13 → afternoon = initial - morning - final → afternoon = 3 :=
by
  intros h_initial h_morning h_final h_afternoon
  rw [h_initial, h_morning, h_final] at h_afternoon
  linarith

end eggs_eaten_in_afternoon_l1608_160808


namespace purely_imaginary_z_l1608_160838

open Complex

theorem purely_imaginary_z (b : ℝ) (h : z = (1 + b * I) / (2 + I) ∧ im z = 0) : z = -I :=
by
  sorry

end purely_imaginary_z_l1608_160838


namespace total_earnings_per_week_correct_l1608_160853

noncomputable def weekday_fee_kid : ℝ := 3
noncomputable def weekday_fee_adult : ℝ := 6
noncomputable def weekend_surcharge_ratio : ℝ := 0.5

noncomputable def num_kids_weekday : ℕ := 8
noncomputable def num_adults_weekday : ℕ := 10

noncomputable def num_kids_weekend : ℕ := 12
noncomputable def num_adults_weekend : ℕ := 15

noncomputable def weekday_earnings_kids : ℝ := (num_kids_weekday : ℝ) * weekday_fee_kid
noncomputable def weekday_earnings_adults : ℝ := (num_adults_weekday : ℝ) * weekday_fee_adult

noncomputable def weekday_earnings_total : ℝ := weekday_earnings_kids + weekday_earnings_adults

noncomputable def weekday_earning_per_week : ℝ := weekday_earnings_total * 5

noncomputable def weekend_fee_kid : ℝ := weekday_fee_kid * (1 + weekend_surcharge_ratio)
noncomputable def weekend_fee_adult : ℝ := weekday_fee_adult * (1 + weekend_surcharge_ratio)

noncomputable def weekend_earnings_kids : ℝ := (num_kids_weekend : ℝ) * weekend_fee_kid
noncomputable def weekend_earnings_adults : ℝ := (num_adults_weekend : ℝ) * weekend_fee_adult

noncomputable def weekend_earnings_total : ℝ := weekend_earnings_kids + weekend_earnings_adults

noncomputable def weekend_earning_per_week : ℝ := weekend_earnings_total * 2

noncomputable def total_weekly_earnings : ℝ := weekday_earning_per_week + weekend_earning_per_week

theorem total_earnings_per_week_correct : total_weekly_earnings = 798 := by
  sorry

end total_earnings_per_week_correct_l1608_160853


namespace relationship_between_a_b_l1608_160884

theorem relationship_between_a_b (a b x : ℝ) 
  (h₁ : x = (a + b) / 2)
  (h₂ : x^2 = (a^2 - b^2) / 2):
  a = -b ∨ a = 3 * b :=
sorry

end relationship_between_a_b_l1608_160884


namespace floor_pi_plus_four_l1608_160846

theorem floor_pi_plus_four : Int.floor (Real.pi + 4) = 7 := by
  sorry

end floor_pi_plus_four_l1608_160846


namespace shooting_prob_l1608_160810

theorem shooting_prob (p q : ℚ) (h: p + q = 1) (n : ℕ) 
  (cond1: p = 2/3) 
  (cond2: q = 1 - p) 
  (cond3: n = 5) : 
  (q ^ (n-1)) = 1/81 := 
by 
  sorry

end shooting_prob_l1608_160810


namespace measure_angle_BCA_l1608_160836

theorem measure_angle_BCA 
  (BCD_angle : ℝ)
  (CBA_angle : ℝ)
  (sum_angles : BCD_angle + CBA_angle + BCA_angle = 190)
  (BCD_right : BCD_angle = 90)
  (CBA_given : CBA_angle = 70) :
  BCA_angle = 30 :=
by
  sorry

end measure_angle_BCA_l1608_160836


namespace problem1_problem2_l1608_160835

-- Define the conditions as noncomputable definitions
noncomputable def A : Real := sorry
noncomputable def tan_A : Real := 2
noncomputable def sin_A_plus_cos_A : Real := 1 / 5

-- Define the trigonometric identities
noncomputable def sin (x : Real) : Real := sorry
noncomputable def cos (x : Real) : Real := sorry
noncomputable def tan (x : Real) : Real := sin x / cos x

-- Ensure the conditions
axiom tan_A_condition : tan A = tan_A
axiom sin_A_plus_cos_A_condition : sin A + cos A = sin_A_plus_cos_A

-- Proof problem 1:
theorem problem1 : 
  (sin (π - A) + cos (-A)) / (sin A - sin (π / 2 + A)) = 3 := by
  sorry

-- Proof problem 2:
theorem problem2 : 
  sin A - cos A = 7 / 5 := by
  sorry

end problem1_problem2_l1608_160835


namespace opposite_of_two_is_negative_two_l1608_160885

theorem opposite_of_two_is_negative_two : -2 = -2 :=
by
  sorry

end opposite_of_two_is_negative_two_l1608_160885


namespace coda_password_combinations_l1608_160870

open BigOperators

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19 
  ∨ n = 23 ∨ n = 29

def is_power_of_two (n : ℕ) : Prop :=
  n = 2 ∨ n = 4 ∨ n = 8 ∨ n = 16

def is_multiple_of_three (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n ≥ 1 ∧ n ≤ 30

def count_primes : ℕ :=
  10
def count_powers_of_two : ℕ :=
  4
def count_multiples_of_three : ℕ :=
  10

theorem coda_password_combinations : count_primes * count_powers_of_two * count_multiples_of_three = 400 := by
  sorry

end coda_password_combinations_l1608_160870


namespace investment_initial_amount_l1608_160852

theorem investment_initial_amount (P : ℝ) (h1 : ∀ (x : ℝ), 0 < x → (1 + 0.10) * x = 1.10 * x) (h2 : 1.21 * P = 363) : P = 300 :=
sorry

end investment_initial_amount_l1608_160852


namespace breadth_of_plot_l1608_160871

theorem breadth_of_plot (b l : ℝ) (h1 : l * b = 18 * b) (h2 : l - b = 10) : b = 8 :=
by
  sorry

end breadth_of_plot_l1608_160871


namespace complementary_angle_problem_l1608_160858

theorem complementary_angle_problem 
  (A B : ℝ) 
  (h1 : A + B = 90) 
  (h2 : A / B = 2 / 3) 
  (increase : A' = A * 1.20) 
  (new_sum : A' + B' = 90) 
  (B' : ℝ)
  (h3 : B' = B - B * 0.1333) :
  true := 
sorry

end complementary_angle_problem_l1608_160858


namespace frac_not_suff_nec_l1608_160890

theorem frac_not_suff_nec {a b : ℝ} (hab : a / b > 1) : 
  ¬ ((∀ a b : ℝ, a / b > 1 → a > b) ∧ (∀ a b : ℝ, a > b → a / b > 1)) :=
sorry

end frac_not_suff_nec_l1608_160890


namespace flute_cost_is_correct_l1608_160848

-- Define the conditions
def total_spent : ℝ := 158.35
def stand_cost : ℝ := 8.89
def songbook_cost : ℝ := 7.0

-- Calculate the cost to be subtracted
def accessories_cost : ℝ := stand_cost + songbook_cost

-- Define the target cost of the flute
def flute_cost : ℝ := total_spent - accessories_cost

-- Prove that the flute cost is $142.46
theorem flute_cost_is_correct : flute_cost = 142.46 :=
by
  -- Here we would provide the proof
  sorry

end flute_cost_is_correct_l1608_160848


namespace pencil_probability_l1608_160860

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem pencil_probability : 
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := 6
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination non_defective_pencils selected_pencils
  let probability := non_defective_ways / total_ways
  probability = 5 / 14 :=
by
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := 6
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination non_defective_pencils selected_pencils
  let probability := non_defective_ways / total_ways
  have h : probability = 5 / 14 := sorry
  exact h

end pencil_probability_l1608_160860


namespace winning_percentage_l1608_160856

/-- A soccer team played 158 games and won 63.2 games. 
    Prove that the winning percentage of the team is 40%. --/
theorem winning_percentage (total_games : ℕ) (won_games : ℝ) (h1 : total_games = 158) (h2 : won_games = 63.2) :
  (won_games / total_games) * 100 = 40 :=
sorry

end winning_percentage_l1608_160856


namespace max_marks_equals_l1608_160897

/-
  Pradeep has to obtain 45% of the total marks to pass.
  He got 250 marks and failed by 50 marks.
  Prove that the maximum marks is 667.
-/

-- Define the passing percentage
def passing_percentage : ℝ := 0.45

-- Define Pradeep's marks and the marks he failed by
def pradeep_marks : ℝ := 250
def failed_by : ℝ := 50

-- Passing marks is the sum of Pradeep's marks and the marks he failed by
def passing_marks : ℝ := pradeep_marks + failed_by

-- Prove that the maximum marks M is 667
theorem max_marks_equals : ∃ M : ℝ, passing_percentage * M = passing_marks ∧ M = 667 :=
sorry

end max_marks_equals_l1608_160897


namespace latest_start_time_l1608_160847

-- Define the times for each activity
def homework_time : ℕ := 30
def clean_room_time : ℕ := 30
def take_out_trash_time : ℕ := 5
def empty_dishwasher_time : ℕ := 10
def dinner_time : ℕ := 45

-- Define the total time required to finish everything in minutes
def total_time_needed : ℕ := homework_time + clean_room_time + take_out_trash_time + empty_dishwasher_time + dinner_time

-- Define the equivalent time in hours
def total_time_needed_hours : ℕ := total_time_needed / 60

-- Define movie start time and the time Justin gets home
def movie_start_time : ℕ := 20 -- (8 PM in 24-hour format)
def justin_home_time : ℕ := 17 -- (5 PM in 24-hour format)

-- Prove the latest time Justin can start his chores and homework
theorem latest_start_time : movie_start_time - total_time_needed_hours = 18 := by
  sorry

end latest_start_time_l1608_160847


namespace solve_for_x_l1608_160826

theorem solve_for_x (x : ℤ) (h : 15 * 2 = x - 3 + 5) : x = 28 :=
sorry

end solve_for_x_l1608_160826


namespace find_common_difference_l1608_160875

variable {a : ℕ → ℝ} -- Define the sequence as a function from natural numbers to real numbers
variable {d : ℝ} -- Define the common difference as a real number

-- Sequence is arithmetic means there exists a common difference such that a_{n+1} = a_n + d
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Conditions from the problem
variable (h1 : a 3 = 5)
variable (h2 : a 15 = 41)
variable (h3 : is_arithmetic_sequence a d)

-- Theorem statement
theorem find_common_difference : d = 3 :=
by
  sorry

end find_common_difference_l1608_160875


namespace intersection_eq_l1608_160876

def A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }
def B : Set ℝ := { x | Real.log (1 - x) > 0 }

theorem intersection_eq : A ∩ B = Set.Icc (-1) 0 :=
by
  sorry

end intersection_eq_l1608_160876


namespace sum_of_squares_l1608_160895

variables (x y z w : ℝ)

def condition1 := (x^2 / (2^2 - 1^2)) + (y^2 / (2^2 - 3^2)) + (z^2 / (2^2 - 5^2)) + (w^2 / (2^2 - 7^2)) = 1
def condition2 := (x^2 / (4^2 - 1^2)) + (y^2 / (4^2 - 3^2)) + (z^2 / (4^2 - 5^2)) + (w^2 / (4^2 - 7^2)) = 1
def condition3 := (x^2 / (6^2 - 1^2)) + (y^2 / (6^2 - 3^2)) + (z^2 / (6^2 - 5^2)) + (w^2 / (6^2 - 7^2)) = 1
def condition4 := (x^2 / (8^2 - 1^2)) + (y^2 / (8^2 - 3^2)) + (z^2 / (8^2 - 5^2)) + (w^2 / (8^2 - 7^2)) = 1

theorem sum_of_squares : condition1 x y z w → condition2 x y z w → 
                          condition3 x y z w → condition4 x y z w →
                          (x^2 + y^2 + z^2 + w^2 = 36) :=
by
  intros h1 h2 h3 h4
  sorry

end sum_of_squares_l1608_160895


namespace square_area_from_inscribed_circle_l1608_160868

theorem square_area_from_inscribed_circle (r : ℝ) (π_pos : 0 < Real.pi) (circle_area : Real.pi * r^2 = 9 * Real.pi) : 
  (2 * r)^2 = 36 :=
by
  -- Proof goes here
  sorry

end square_area_from_inscribed_circle_l1608_160868


namespace necessary_but_not_sufficient_l1608_160819

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a n = a 0 * q^n

def is_increasing_sequence (s : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, s n < s (n + 1)

def sum_first_n_terms (a : ℕ → ℝ) (q : ℝ) : ℕ → ℝ 
| 0 => a 0
| (n+1) => (sum_first_n_terms a q n) + (a 0 * q ^ n)

theorem necessary_but_not_sufficient (a : ℕ → ℝ) (q : ℝ) (h_geometric: is_geometric_sequence a q) :
  (q > 0) ∧ is_increasing_sequence (sum_first_n_terms a q) ↔ (q > 0)
:= sorry

end necessary_but_not_sufficient_l1608_160819


namespace total_cost_of_goods_l1608_160844

theorem total_cost_of_goods :
  ∃ (M R F : ℝ),
    (10 * M = 24 * R) ∧
    (6 * F = 2 * R) ∧
    (F = 20.50) ∧
    (4 * M + 3 * R + 5 * F = 877.40) :=
by {
  sorry
}

end total_cost_of_goods_l1608_160844


namespace ratio_of_side_length_to_brush_width_l1608_160869

theorem ratio_of_side_length_to_brush_width (s w : ℝ) (h : (w^2 + ((s - w)^2) / 2) = s^2 / 3) : s / w = 3 :=
by
  sorry

end ratio_of_side_length_to_brush_width_l1608_160869


namespace miles_ridden_further_l1608_160891

theorem miles_ridden_further (distance_ridden distance_walked : ℝ) (h1 : distance_ridden = 3.83) (h2 : distance_walked = 0.17) :
  distance_ridden - distance_walked = 3.66 := 
by sorry

end miles_ridden_further_l1608_160891


namespace factor_expression_l1608_160873

theorem factor_expression (x : ℝ) : 45 * x^2 + 135 * x = 45 * x * (x + 3) := 
by
  sorry

end factor_expression_l1608_160873


namespace min_red_chips_l1608_160820

theorem min_red_chips (w b r : ℕ) (h1 : b ≥ w / 3) (h2 : b ≤ r / 4) (h3 : w + b ≥ 75) : r ≥ 76 :=
sorry

end min_red_chips_l1608_160820


namespace height_of_larger_box_l1608_160878

/-- Define the dimensions of the larger box and smaller boxes, 
    and show that given the constraints, the height of the larger box must be 4 meters.-/
theorem height_of_larger_box 
  (L H : ℝ) (V_small : ℝ) (N_small : ℕ) (h : ℝ) 
  (dim_large : L = 6) (width_large : H = 5)
  (vol_small : V_small = 0.6 * 0.5 * 0.4) 
  (num_boxes : N_small = 1000) 
  (vol_large : 6 * 5 * h = N_small * V_small) : 
  h = 4 :=
by 
  sorry

end height_of_larger_box_l1608_160878


namespace arithmetic_seq_property_l1608_160862

-- Define the arithmetic sequence {a_n}
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the conditions
variable (a d : ℤ)
variable (h1 : arithmetic_seq a d 3 + arithmetic_seq a d 9 + arithmetic_seq a d 15 = 30)

-- Define the statement to be proved
theorem arithmetic_seq_property : 
  arithmetic_seq a d 17 - 2 * arithmetic_seq a d 13 = -10 :=
by
  sorry

end arithmetic_seq_property_l1608_160862


namespace range_of_m_l1608_160811

variable {x m : ℝ}

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (m : ℝ) : (¬ p m ∨ ¬ q m) → m ≥ 2 := 
sorry

end range_of_m_l1608_160811


namespace cone_volume_with_same_radius_and_height_l1608_160831

theorem cone_volume_with_same_radius_and_height (r h : ℝ) 
  (Vcylinder : ℝ) (Vcone : ℝ) (h1 : Vcylinder = 54 * Real.pi) 
  (h2 : Vcone = (1 / 3) * Vcylinder) : Vcone = 18 * Real.pi :=
by sorry

end cone_volume_with_same_radius_and_height_l1608_160831


namespace distribute_coins_l1608_160806

/-- The number of ways to distribute 25 identical coins among 4 schoolchildren -/
theorem distribute_coins :
  (Nat.choose 28 3) = 3276 :=
by
  sorry

end distribute_coins_l1608_160806


namespace backpack_price_equation_l1608_160874

-- Define the original price of the backpack
variable (x : ℝ)

-- Define the conditions
def discount1 (x : ℝ) : ℝ := 0.8 * x
def discount2 (d : ℝ) : ℝ := d - 10
def final_price (p : ℝ) : Prop := p = 90

-- Final statement to be proved
theorem backpack_price_equation : final_price (discount2 (discount1 x)) ↔ 0.8 * x - 10 = 90 := sorry

end backpack_price_equation_l1608_160874


namespace group_size_systematic_sampling_l1608_160877

-- Define the total number of viewers
def total_viewers : ℕ := 10000

-- Define the number of viewers to be selected
def selected_viewers : ℕ := 10

-- Lean statement to prove the group size for systematic sampling
theorem group_size_systematic_sampling (n_total n_selected : ℕ) : n_total = total_viewers → n_selected = selected_viewers → (n_total / n_selected) = 1000 :=
by
  intros h_total h_selected
  rw [h_total, h_selected]
  sorry

end group_size_systematic_sampling_l1608_160877


namespace price_percentage_combined_assets_l1608_160815

variable (A B P : ℝ)

-- Conditions
axiom h1 : P = 1.20 * A
axiom h2 : P = 2 * B

-- Statement
theorem price_percentage_combined_assets : (P / (A + B)) * 100 = 75 := by
  sorry

end price_percentage_combined_assets_l1608_160815


namespace quadratic_identity_l1608_160879

theorem quadratic_identity (x : ℝ) : 
  (3*x + 1)^2 + 2*(3*x + 1)*(x - 3) + (x - 3)^2 = 16*x^2 - 16*x + 4 :=
by
  sorry

end quadratic_identity_l1608_160879


namespace range_of_a_l1608_160833

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 3) : -2 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l1608_160833


namespace power_24_eq_one_l1608_160899

theorem power_24_eq_one (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^24 = 1 :=
by
  sorry

end power_24_eq_one_l1608_160899


namespace maximum_rabbits_l1608_160882

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l1608_160882


namespace ones_digit_largest_power_of_3_dividing_18_factorial_l1608_160830

theorem ones_digit_largest_power_of_3_dividing_18_factorial :
  (3^8 % 10) = 1 :=
by sorry

end ones_digit_largest_power_of_3_dividing_18_factorial_l1608_160830


namespace additional_oil_needed_l1608_160802

def car_cylinders := 6
def car_oil_per_cylinder := 8
def truck_cylinders := 8
def truck_oil_per_cylinder := 10
def motorcycle_cylinders := 4
def motorcycle_oil_per_cylinder := 6

def initial_car_oil := 16
def initial_truck_oil := 20
def initial_motorcycle_oil := 8

theorem additional_oil_needed :
  let car_total_oil := car_cylinders * car_oil_per_cylinder
  let truck_total_oil := truck_cylinders * truck_oil_per_cylinder
  let motorcycle_total_oil := motorcycle_cylinders * motorcycle_oil_per_cylinder
  let car_additional_oil := car_total_oil - initial_car_oil
  let truck_additional_oil := truck_total_oil - initial_truck_oil
  let motorcycle_additional_oil := motorcycle_total_oil - initial_motorcycle_oil
  car_additional_oil = 32 ∧
  truck_additional_oil = 60 ∧
  motorcycle_additional_oil = 16 :=
by
  repeat (exact sorry)

end additional_oil_needed_l1608_160802


namespace fraction_subtraction_l1608_160841

theorem fraction_subtraction :
  (15 / 45) - (1 + (2 / 9)) = - (8 / 9) :=
by
  sorry

end fraction_subtraction_l1608_160841


namespace vacation_cost_division_l1608_160880

theorem vacation_cost_division (n : ℕ) (h1 : 360 = 4 * (120 - 30)) (h2 : 360 = n * 120) : n = 3 := 
sorry

end vacation_cost_division_l1608_160880


namespace number_of_unique_intersections_l1608_160813

-- Definitions for the given lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 3
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 5 * x - 3 * y = 6

-- The problem is to show the number of unique intersection points is 2
theorem number_of_unique_intersections : ∃ p1 p2 : ℝ × ℝ,
  p1 ≠ p2 ∧
  (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
  (line1 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
  (p1 ≠ p2 → ∀ p : ℝ × ℝ, (line1 p.1 p.2 ∨ line2 p.1 p.2 ∨ line3 p.1 p.2) →
    (p = p1 ∨ p = p2)) :=
sorry

end number_of_unique_intersections_l1608_160813


namespace find_all_triplets_l1608_160861

theorem find_all_triplets (a b c : ℕ)
  (h₀_a : a > 0)
  (h₀_b : b > 0)
  (h₀_c : c > 0) :
  6^a = 1 + 2^b + 3^c ↔ 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 5 ∧ c = 1) :=
by
  sorry

end find_all_triplets_l1608_160861


namespace dean_taller_than_ron_l1608_160816

theorem dean_taller_than_ron (d h r : ℕ) (h1 : d = 15 * h) (h2 : r = 13) (h3 : d = 255) : h - r = 4 := 
by 
  sorry

end dean_taller_than_ron_l1608_160816


namespace expected_value_of_monicas_winnings_l1608_160843

def die_outcome (n : ℕ) : ℤ :=
  if n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 then n else if n = 1 ∨ n = 4 ∨ n = 6 ∨ n = 8 then 0 else -5

noncomputable def expected_winnings : ℚ :=
  (1/2 : ℚ) * 0 + (1/8 : ℚ) * 2 + (1/8 : ℚ) * 3 + (1/8 : ℚ) * 5 + (1/8 : ℚ) * 7 + (1/8 : ℚ) * (-5)

theorem expected_value_of_monicas_winnings : expected_winnings = 3/2 := by
  sorry

end expected_value_of_monicas_winnings_l1608_160843


namespace solve_f_sqrt_2009_l1608_160849

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_never_zero : ∀ x : ℝ, f x ≠ 0
axiom functional_eq : ∀ x y : ℝ, f (x - y) = 2009 * f x * f y

theorem solve_f_sqrt_2009 :
  f (sqrt 2009) = 1 / 2009 := sorry

end solve_f_sqrt_2009_l1608_160849


namespace other_number_l1608_160832

theorem other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 2310) (h_hcf : Nat.gcd A B = 30) (h_A : A = 770) : B = 90 :=
by
  -- The proof is omitted here.
  sorry

end other_number_l1608_160832


namespace exists_divisible_diff_l1608_160894

theorem exists_divisible_diff (l : List ℤ) (h_len : l.length = 2022) :
  ∃ i j, i ≠ j ∧ (l.nthLe i sorry - l.nthLe j sorry) % 2021 = 0 :=
by
  apply sorry -- Placeholder for proof

end exists_divisible_diff_l1608_160894


namespace solve_inequality_l1608_160812

theorem solve_inequality (x : ℝ) : (1 ≤ |x + 3| ∧ |x + 3| ≤ 4) ↔ (-7 ≤ x ∧ x ≤ -4) ∨ (-2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_l1608_160812


namespace height_of_windows_l1608_160824

theorem height_of_windows
  (L W H d_l d_w w_w : ℕ)
  (C T : ℕ)
  (hl : L = 25)
  (hw : W = 15)
  (hh : H = 12)
  (hdl : d_l = 6)
  (hdw : d_w = 3)
  (hww : w_w = 3)
  (hc : C = 3)
  (ht : T = 2718):
  ∃ h : ℕ, 960 - (18 + 9 * h) = 906 ∧ 
  (T = C * (960 - (18 + 9 * h))) ∧
  (960 = 2 * (L * H) + 2 * (W * H)) ∧ 
  (18 = d_l * d_w) ∧ 
  (9 * h = 3 * (h * w_w)) := 
sorry

end height_of_windows_l1608_160824


namespace mean_proportional_l1608_160842

theorem mean_proportional (x : ℝ) (h : (72.5:ℝ) = Real.sqrt (x * 81)): x = 64.9 := by
  sorry

end mean_proportional_l1608_160842


namespace orchid_bushes_total_l1608_160825

def current_orchid_bushes : ℕ := 22
def new_orchid_bushes : ℕ := 13

theorem orchid_bushes_total : current_orchid_bushes + new_orchid_bushes = 35 := 
by 
  sorry

end orchid_bushes_total_l1608_160825


namespace area_of_square_l1608_160829

theorem area_of_square (r s l : ℕ) (h1 : l = (2 * r) / 5) (h2 : r = s) (h3 : l * 10 = 240) : s * s = 3600 :=
by
  sorry

end area_of_square_l1608_160829


namespace smallest_possible_n_l1608_160839

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def n_is_three_digit (n : ℕ) : Prop := 
  n ≥ 100 ∧ n < 1000

def prime_digits_less_than_10 (p : ℕ) : Prop :=
  p ∈ [2, 3, 5, 7]

def three_distinct_prime_factors (n a b : ℕ) : Prop :=
  a ≠ b ∧ is_prime a ∧ is_prime b ∧ is_prime (10 * a + b) ∧ n = a * b * (10 * a + b)

theorem smallest_possible_n :
  ∃ (n a b : ℕ), n_is_three_digit n ∧ prime_digits_less_than_10 a ∧ prime_digits_less_than_10 b ∧ three_distinct_prime_factors n a b ∧ n = 138 :=
by {
  sorry
}

end smallest_possible_n_l1608_160839


namespace weight_of_new_girl_l1608_160803

theorem weight_of_new_girl (W N : ℝ) (h_weight_replacement: (20 * W / 20 + 40 - 40 + 40) / 20 = W / 20 + 2) :
  N = 80 :=
by
  sorry

end weight_of_new_girl_l1608_160803


namespace number_of_people_only_went_to_aquarium_is_5_l1608_160889

-- Define the conditions
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group_size : ℕ := 10
def total_earnings : ℕ := 240

-- Define the problem in Lean
theorem number_of_people_only_went_to_aquarium_is_5 :
  ∃ x : ℕ, (total_earnings - (group_size * (admission_fee + tour_fee)) = x * admission_fee) → x = 5 :=
by
  sorry

end number_of_people_only_went_to_aquarium_is_5_l1608_160889


namespace cost_of_each_pair_of_jeans_l1608_160896

-- Conditions
def costWallet : ℕ := 50
def costSneakers : ℕ := 100
def pairsSneakers : ℕ := 2
def costBackpack : ℕ := 100
def totalSpent : ℕ := 450
def pairsJeans : ℕ := 2

-- Definitions
def totalSpentLeonard := costWallet + pairsSneakers * costSneakers
def totalSpentMichaelWithoutJeans := costBackpack

-- Goal: Prove the cost of each pair of jeans
theorem cost_of_each_pair_of_jeans :
  let totalCostJeans := totalSpent - (totalSpentLeonard + totalSpentMichaelWithoutJeans)
  let costPerPairJeans := totalCostJeans / pairsJeans
  costPerPairJeans = 50 :=
by
  intros
  let totalCostJeans := totalSpent - (totalSpentLeonard + totalSpentMichaelWithoutJeans)
  let costPerPairJeans := totalCostJeans / pairsJeans
  show costPerPairJeans = 50
  sorry

end cost_of_each_pair_of_jeans_l1608_160896


namespace determine_b_l1608_160817

variable (a b c : ℝ)

theorem determine_b
  (h1 : -a / 3 = -c)
  (h2 : 1 + a + b + c = -c)
  (h3 : c = 5) :
  b = -26 :=
by
  sorry

end determine_b_l1608_160817


namespace total_seashells_found_intact_seashells_found_l1608_160863

-- Define the constants for seashells found
def tom_seashells : ℕ := 15
def fred_seashells : ℕ := 43

-- Define total_intercept
def total_intercept : ℕ := 29

-- Statement that the total seashells found by Tom and Fred is 58
theorem total_seashells_found : tom_seashells + fred_seashells = 58 := by
  sorry

-- Statement that the intact seashells are obtained by subtracting cracked ones
theorem intact_seashells_found : tom_seashells + fred_seashells - total_intercept = 29 := by
  sorry

end total_seashells_found_intact_seashells_found_l1608_160863


namespace age_difference_l1608_160872

theorem age_difference
  (A B C : ℕ)
  (h1 : B = 12)
  (h2 : B = 2 * C)
  (h3 : A + B + C = 32) :
  A - B = 2 :=
sorry

end age_difference_l1608_160872


namespace original_flow_rate_l1608_160867

theorem original_flow_rate :
  ∃ F : ℚ, 
  (F * 0.75 * 0.4 * 0.6 - 1 = 2) ∧
  (F = 50/3) :=
by
  sorry

end original_flow_rate_l1608_160867


namespace solve_quadratic_difference_l1608_160837

theorem solve_quadratic_difference :
  ∀ x : ℝ, (x^2 - 7*x - 48 = 0) → 
  let x1 := (7 + Real.sqrt 241) / 2
  let x2 := (7 - Real.sqrt 241) / 2
  abs (x1 - x2) = Real.sqrt 241 :=
by
  sorry

end solve_quadratic_difference_l1608_160837


namespace additional_machines_l1608_160892

theorem additional_machines (r : ℝ) (M : ℝ) : 
  (5 * r * 20 = 1) ∧ (M * r * 10 = 1) → (M - 5 = 95) :=
by
  sorry

end additional_machines_l1608_160892


namespace geometric_sequence_sum_l1608_160857

theorem geometric_sequence_sum :
  ∀ (a : ℕ → ℝ) (q : ℝ),
    a 1 * q + a 1 * q ^ 3 = 20 →
    a 1 * q ^ 2 + a 1 * q ^ 4 = 40 →
    a 1 * q ^ 4 + a 1 * q ^ 6 = 160 :=
by
  sorry

end geometric_sequence_sum_l1608_160857


namespace solve_system_l1608_160807

def F (t : ℝ) : ℝ := 32 * t ^ 5 + 48 * t ^ 3 + 17 * t - 15

def system_of_equations (x y z : ℝ) : Prop :=
  (1 / x = (32 / y ^ 5) + (48 / y ^ 3) + (17 / y) - 15) ∧
  (1 / y = (32 / z ^ 5) + (48 / z ^ 3) + (17 / z) - 15) ∧
  (1 / z = (32 / x ^ 5) + (48 / x ^ 3) + (17 / x) - 15)

theorem solve_system : ∃ (x y z : ℝ), system_of_equations x y z ∧ x = 2 ∧ y = 2 ∧ z = 2 :=
by
  sorry -- Proof not included

end solve_system_l1608_160807


namespace man_speed_against_current_eq_l1608_160886

-- Definitions
def downstream_speed : ℝ := 22 -- Man's speed with the current in km/hr
def current_speed : ℝ := 5 -- Speed of the current in km/hr

-- Man's speed in still water
def man_speed_in_still_water : ℝ := downstream_speed - current_speed

-- Man's speed against the current
def speed_against_current : ℝ := man_speed_in_still_water - current_speed

-- Theorem: The man's speed against the current is 12 km/hr.
theorem man_speed_against_current_eq : speed_against_current = 12 := by
  sorry

end man_speed_against_current_eq_l1608_160886


namespace abs_neg_five_not_eq_five_l1608_160845

theorem abs_neg_five_not_eq_five : -(abs (-5)) ≠ 5 := by
  sorry

end abs_neg_five_not_eq_five_l1608_160845


namespace largest_integer_satisfying_condition_l1608_160864

-- Definition of the conditions
def has_four_digits_in_base_10 (n : ℕ) : Prop :=
  10^3 ≤ n^2 ∧ n^2 < 10^4

-- Proof statement: N is the largest integer satisfying the condition
theorem largest_integer_satisfying_condition : ∃ (N : ℕ), 
  has_four_digits_in_base_10 N ∧ (∀ (m : ℕ), has_four_digits_in_base_10 m → m ≤ N) ∧ N = 99 := 
sorry

end largest_integer_satisfying_condition_l1608_160864


namespace range_of_x_l1608_160881

theorem range_of_x : ∀ x : ℝ, (¬ (x + 3 = 0)) ∧ (4 - x ≥ 0) ↔ x ≤ 4 ∧ x ≠ -3 := by
  sorry

end range_of_x_l1608_160881


namespace smaller_angle_of_parallelogram_l1608_160823

theorem smaller_angle_of_parallelogram (x : ℝ) (h : x + 3 * x = 180) : x = 45 :=
sorry

end smaller_angle_of_parallelogram_l1608_160823


namespace multiplier_of_difference_l1608_160827

variable (x y : ℕ)
variable (h : x + y = 49) (h1 : x > y)

theorem multiplier_of_difference (h2 : x^2 - y^2 = k * (x - y)) : k = 49 :=
by sorry

end multiplier_of_difference_l1608_160827


namespace length_of_AB_l1608_160851

theorem length_of_AB {A B P Q : ℝ} (h1 : P = 3 / 5 * B)
                    (h2 : Q = 2 / 5 * A + 3 / 5 * B)
                    (h3 : dist P Q = 5) :
  dist A B = 25 :=
by sorry

end length_of_AB_l1608_160851


namespace a_plus_b_l1608_160887

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x)

theorem a_plus_b (a b : ℝ) (h : f (a - 1) + f b = 0) : a + b = 1 :=
by
  sorry

end a_plus_b_l1608_160887


namespace initial_bird_count_l1608_160855

theorem initial_bird_count (B : ℕ) (h₁ : B + 7 = 12) : B = 5 :=
by
  sorry

end initial_bird_count_l1608_160855


namespace v3_at_2_is_15_l1608_160854

-- Define the polynomial f(x)
def f (x : ℝ) := x^4 + 2*x^3 + x^2 - 3*x - 1

-- Define v3 using Horner's Rule at x
def v3 (x : ℝ) := ((x + 2) * x + 1) * x - 3

-- Prove that v3 at x = 2 equals 15
theorem v3_at_2_is_15 : v3 2 = 15 :=
by
  -- Skipping the proof with sorry
  sorry

end v3_at_2_is_15_l1608_160854


namespace Total_points_proof_l1608_160809

noncomputable def Samanta_points (Mark_points : ℕ) : ℕ := Mark_points + 8
noncomputable def Mark_points (Eric_points : ℕ) : ℕ := Eric_points + (Eric_points / 2)
def Eric_points : ℕ := 6
noncomputable def Daisy_points (Total_points_Samanta_Mark_Eric : ℕ) : ℕ := Total_points_Samanta_Mark_Eric - (Total_points_Samanta_Mark_Eric / 4)

def Total_points_Samanta_Mark_Eric (Samanta_points Mark_points Eric_points : ℕ) : ℕ := Samanta_points + Mark_points + Eric_points

theorem Total_points_proof :
  let Mk_pts := Mark_points Eric_points
  let Sm_pts := Samanta_points Mk_pts
  let Tot_SME := Total_points_Samanta_Mark_Eric Sm_pts Mk_pts Eric_points
  let D_pts := Daisy_points Tot_SME
  Sm_pts + Mk_pts + Eric_points + D_pts = 56 := by
  sorry

end Total_points_proof_l1608_160809


namespace max_min_values_l1608_160821

def f (x : ℝ) : ℝ := 6 - 12 * x + x^3

theorem max_min_values :
  let a := (-1 / 3 : ℝ)
  let b := (1 : ℝ)
  max (f a) (f b) = f a ∧ f a = 269 / 27 ∧ min (f a) (f b) = f b ∧ f b = -5 :=
by
  let a := (-1 / 3 : ℝ)
  let b := (1 : ℝ)
  have ha : f a = 269 / 27 := sorry
  have hb : f b = -5 := sorry
  have max_eq : max (f a) (f b) = f a := by sorry
  have min_eq : min (f a) (f b) = f b := by sorry
  exact ⟨max_eq, ha, min_eq, hb⟩

end max_min_values_l1608_160821


namespace sum_digits_of_consecutive_numbers_l1608_160888

-- Define the sum of digits function
def sum_digits (n : ℕ) : ℕ := sorry -- Placeholder, define the sum of digits function

-- Given conditions
variables (N : ℕ)
axiom h1 : sum_digits N + sum_digits (N + 1) = 200
axiom h2 : sum_digits (N + 2) + sum_digits (N + 3) = 105

-- Theorem statement to be proved
theorem sum_digits_of_consecutive_numbers : 
  sum_digits (N + 1) + sum_digits (N + 2) = 103 := 
sorry  -- Proof to be provided

end sum_digits_of_consecutive_numbers_l1608_160888


namespace most_pieces_day_and_maximum_number_of_popular_days_l1608_160818

-- Definitions for conditions:
def a_n (n : ℕ) : ℕ :=
if h : n ≤ 13 then 3 * n
else 65 - 2 * n

def S_n (n : ℕ) : ℕ :=
if h : n ≤ 13 then (3 + 3 * n) * n / 2
else 273 + (51 - n) * (n - 13)

-- Propositions to prove:
theorem most_pieces_day_and_maximum :
  ∃ k a_k, (1 ≤ k ∧ k ≤ 31) ∧
           (a_k = a_n k) ∧
           (∀ n, 1 ≤ n ∧ n ≤ 31 → a_n n ≤ a_k) ∧
           k = 13 ∧ a_k = 39 := 
sorry

theorem number_of_popular_days :
  ∃ days_popular,
    (∃ n1, 1 ≤ n1 ∧ n1 ≤ 13 ∧ S_n n1 > 200) ∧
    (∃ n2, 14 ≤ n2 ∧ n2 ≤ 31 ∧ a_n n2 < 20) ∧
    days_popular = (22 - 12 + 1) :=
sorry

end most_pieces_day_and_maximum_number_of_popular_days_l1608_160818
