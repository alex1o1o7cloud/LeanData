import Mathlib

namespace total_time_naomi_30webs_l217_217319

-- Define the constants based on the given conditions
def time_katherine : ℕ := 20
def factor_naomi : ℚ := 5/4
def websites : ℕ := 30

-- Define the time taken by Naomi to build one website based on the conditions
def time_naomi (time_katherine : ℕ) (factor_naomi : ℚ) : ℚ :=
  factor_naomi * time_katherine

-- Define the total time Naomi took to build all websites
def total_time_naomi (time_naomi : ℚ) (websites : ℕ) : ℚ :=
  time_naomi * websites

-- Statement: Proving that the total number of hours Naomi took to create 30 websites is 750
theorem total_time_naomi_30webs : 
  total_time_naomi (time_naomi time_katherine factor_naomi) websites = 750 := 
sorry

end total_time_naomi_30webs_l217_217319


namespace probability_of_yellow_face_l217_217963

theorem probability_of_yellow_face :
  let total_faces : ℕ := 10
  let yellow_faces : ℕ := 4
  (yellow_faces : ℚ) / (total_faces : ℚ) = 2 / 5 :=
by
  sorry

end probability_of_yellow_face_l217_217963


namespace solve_eq1_solve_eq2_l217_217691

noncomputable def eq1 (x : ℝ) : Prop := x - 2 = 4 * (x - 2)^2
noncomputable def eq2 (x : ℝ) : Prop := x * (2 * x + 1) = 8 * x - 3

theorem solve_eq1 (x : ℝ) : eq1 x ↔ x = 2 ∨ x = 9 / 4 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : eq2 x ↔ x = 1 / 2 ∨ x = 3 :=
by
  sorry

end solve_eq1_solve_eq2_l217_217691


namespace probability_of_rolling_2_4_6_l217_217233

open Set Classical

noncomputable def fair_six_sided_die_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

def successful_outcomes : Finset ℕ := {2, 4, 6}

theorem probability_of_rolling_2_4_6 : 
  (successful_outcomes.card : ℚ) / (fair_six_sided_die_outcomes.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_rolling_2_4_6_l217_217233


namespace max_souls_guaranteed_l217_217955

def initial_nuts : ℕ := 1001

def valid_N (N : ℕ) : Prop :=
  1 ≤ N ∧ N ≤ 1001

def nuts_transferred (N : ℕ) (T : ℕ) : Prop :=
  valid_N N ∧ T ≤ 71

theorem max_souls_guaranteed : (∀ N, valid_N N → ∃ T, nuts_transferred N T) :=
sorry

end max_souls_guaranteed_l217_217955


namespace percent_decrease_correct_l217_217196

def original_price_per_pack : ℚ := 7 / 3
def promotional_price_per_pack : ℚ := 8 / 4
def percent_decrease_in_price (old_price new_price : ℚ) : ℚ := 
  ((old_price - new_price) / old_price) * 100

theorem percent_decrease_correct :
  percent_decrease_in_price original_price_per_pack promotional_price_per_pack = 14 := by
  sorry

end percent_decrease_correct_l217_217196


namespace length_BC_l217_217639

theorem length_BC (AB AC AM : ℝ)
  (hAB : AB = 5)
  (hAC : AC = 7)
  (hAM : AM = 4)
  (M_midpoint_of_BC : ∃ (BM MC : ℝ), BM = MC ∧ ∀ (BC: ℝ), BC = BM + MC) :
  ∃ (BC : ℝ), BC = 2 * Real.sqrt 21 := by
  sorry

end length_BC_l217_217639


namespace most_likely_sitting_people_l217_217644

theorem most_likely_sitting_people :
  let num_people := 100
  let seats := 100
  let favorite_seats : Fin num_people → Fin seats := sorry
  -- Conditions related to people sitting behavior
  let sits_in_row (i : Fin num_people) : Prop :=
    ∀ j : Fin num_people, j < i → favorite_seats j ≠ favorite_seats i
  let num_sitting_in_row := Finset.card (Finset.filter sits_in_row (Finset.univ : Finset (Fin num_people)))
  -- Prove
  num_sitting_in_row = 10 := 
sorry

end most_likely_sitting_people_l217_217644


namespace purse_multiple_of_wallet_l217_217480

theorem purse_multiple_of_wallet (W P : ℤ) (hW : W = 22) (hc : W + P = 107) : ∃ n : ℤ, n * W > P ∧ n = 4 :=
by
  sorry

end purse_multiple_of_wallet_l217_217480


namespace jill_and_bob_payment_l217_217444

-- Definitions of the conditions
def price_of_first_house (X : ℝ) := X
def price_of_second_house (Y X : ℝ) := 2 * X

theorem jill_and_bob_payment :
  ∃ X, ∃ Y, Y = 2 * X ∧ X + Y = 600000 ∧ X = 200000 :=
by
  sorry

end jill_and_bob_payment_l217_217444


namespace cost_per_book_eq_three_l217_217724

-- Let T be the total amount spent, B be the number of books, and C be the cost per book
variables (T B C : ℕ)
-- Conditions: Edward spent $6 (T = 6) to buy 2 books (B = 2)
-- Each book costs the same amount (C = T / B)
axiom total_amount : T = 6
axiom number_of_books : B = 2

-- We need to prove that each book cost $3
theorem cost_per_book_eq_three (h1 : T = 6) (h2 : B = 2) : (T / B) = 3 := by
  sorry

end cost_per_book_eq_three_l217_217724


namespace system_solution_correct_l217_217333

theorem system_solution_correct (b : ℝ) : (∃ x y : ℝ, (y = 3 * x - 5) ∧ (y = 2 * x + b) ∧ (x = 1) ∧ (y = -2)) ↔ b = -4 :=
by
  sorry

end system_solution_correct_l217_217333


namespace no_such_number_l217_217844

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def productOfDigitsIsPerfectSquare (n : ℕ) : Prop :=
  ∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ isPerfectSquare (d1 * d2)

theorem no_such_number :
  ¬ ∃ (N : ℕ),
    (N > 9) ∧ (N < 100) ∧ -- N is a two-digit number
    (N % 2 = 0) ∧        -- N is even
    (N % 13 = 0) ∧       -- N is a multiple of 13
    productOfDigitsIsPerfectSquare N := -- The product of digits of N is a perfect square
by
  sorry

end no_such_number_l217_217844


namespace complement_intersection_example_l217_217238

open Set

theorem complement_intersection_example
  (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3, 4})
  (hB : B = {2, 3}) :
  (U \ A) ∩ B = {2} :=
by
  sorry

end complement_intersection_example_l217_217238


namespace minnie_lucy_time_difference_is_66_minutes_l217_217779

noncomputable def minnie_time_uphill : ℚ := 12 / 6
noncomputable def minnie_time_downhill : ℚ := 18 / 25
noncomputable def minnie_time_flat : ℚ := 15 / 15

noncomputable def minnie_total_time : ℚ := minnie_time_uphill + minnie_time_downhill + minnie_time_flat

noncomputable def lucy_time_flat : ℚ := 15 / 25
noncomputable def lucy_time_uphill : ℚ := 12 / 8
noncomputable def lucy_time_downhill : ℚ := 18 / 35

noncomputable def lucy_total_time : ℚ := lucy_time_flat + lucy_time_uphill + lucy_time_downhill

-- Convert hours to minutes
noncomputable def minnie_total_time_minutes : ℚ := minnie_total_time * 60
noncomputable def lucy_total_time_minutes : ℚ := lucy_total_time * 60

-- Difference in minutes
noncomputable def time_difference : ℚ := minnie_total_time_minutes - lucy_total_time_minutes

theorem minnie_lucy_time_difference_is_66_minutes : time_difference = 66 := by
  sorry

end minnie_lucy_time_difference_is_66_minutes_l217_217779


namespace k_value_l217_217356

theorem k_value (k : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |k * x - 4| ≤ 2) → k = 2 := 
by
  intros h
  sorry

end k_value_l217_217356


namespace consumer_installment_credit_l217_217009

theorem consumer_installment_credit : 
  ∃ C : ℝ, 
    (0.43 * C = 200) ∧ 
    (C = 465.116) :=
by
  sorry

end consumer_installment_credit_l217_217009


namespace sum_from_one_to_twelve_l217_217454

-- Define the sum of an arithmetic series
def sum_arithmetic_series (n : ℕ) (a : ℕ) (l : ℕ) : ℕ :=
  (n * (a + l)) / 2

-- Theorem stating the sum of numbers from 1 to 12
theorem sum_from_one_to_twelve : sum_arithmetic_series 12 1 12 = 78 := by
  sorry

end sum_from_one_to_twelve_l217_217454


namespace ones_digit_of_largest_power_of_3_dividing_factorial_l217_217157

theorem ones_digit_of_largest_power_of_3_dividing_factorial (n : ℕ) (h : 27 = 3^3) : 
  (fun x => x % 10) (3^13) = 3 := by
  sorry

end ones_digit_of_largest_power_of_3_dividing_factorial_l217_217157


namespace unique_solution_exists_l217_217307

theorem unique_solution_exists (a x y z : ℝ) 
  (h1 : z = a * (x + 2 * y + 5 / 2)) 
  (h2 : x^2 + y^2 + 2 * x - y + a * (x + 2 * y + 5 / 2) = 0) :
  a = 1 → x = -3 / 2 ∧ y = -1 / 2 ∧ z = 0 := 
by
  sorry

end unique_solution_exists_l217_217307


namespace floor_expression_equality_l217_217937

theorem floor_expression_equality :
  ⌊((2023^3 : ℝ) / (2021 * 2022) - (2021^3 : ℝ) / (2022 * 2023))⌋ = 8 := 
sorry

end floor_expression_equality_l217_217937


namespace ratio_x_to_y_l217_217947

theorem ratio_x_to_y (x y : ℤ) (h : (10*x - 3*y) / (13*x - 2*y) = 3 / 5) : x / y = 9 / 11 := 
by sorry

end ratio_x_to_y_l217_217947


namespace valid_votes_other_candidate_l217_217078

theorem valid_votes_other_candidate (total_votes : ℕ) (invalid_percentage : ℕ) (candidate1_percentage : ℕ) (valid_votes_other_candidate : ℕ) : 
  total_votes = 7500 → 
  invalid_percentage = 20 → 
  candidate1_percentage = 55 → 
  valid_votes_other_candidate = 2700 :=
by
  sorry

end valid_votes_other_candidate_l217_217078


namespace tomatoes_picked_yesterday_l217_217621

-- Definitions corresponding to the conditions in the problem.
def initial_tomatoes : Nat := 160
def tomatoes_left_after_yesterday : Nat := 104

-- Statement of the problem proving the number of tomatoes picked yesterday.
theorem tomatoes_picked_yesterday : initial_tomatoes - tomatoes_left_after_yesterday = 56 :=
by
  sorry

end tomatoes_picked_yesterday_l217_217621


namespace cost_price_of_toy_l217_217734

-- Define the conditions
def sold_toys := 18
def selling_price := 23100
def gain_toys := 3

-- Define the cost price of one toy 
noncomputable def C := 1100

-- Lean 4 statement to prove the cost price
theorem cost_price_of_toy (C : ℝ) (sold_toys selling_price gain_toys : ℕ) (h1 : selling_price = (sold_toys + gain_toys) * C) : 
  C = 1100 := 
by
  sorry


end cost_price_of_toy_l217_217734


namespace power_sum_l217_217564

theorem power_sum : 1^234 + 4^6 / 4^4 = 17 :=
by
  sorry

end power_sum_l217_217564


namespace value_of_x_l217_217859

theorem value_of_x (x y : ℕ) (h1 : y = 864) (h2 : x^3 * 6^3 / 432 = y) : x = 12 :=
sorry

end value_of_x_l217_217859


namespace solve_inequality_l217_217331

theorem solve_inequality :
  { x : ℝ | x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 7 ∧ 
    (2 / (x - 1) - 3 / (x - 3) + 5 / (x - 5) - 2 / (x - 7) < 1 / 15) } = 
  { x : ℝ | (x < -8) ∨ (-7 < x ∧ x < -1) ∨ (1 < x ∧ x < 3) ∨ (5 < x ∧ x < 7) ∨ (x > 8) } := sorry

end solve_inequality_l217_217331


namespace range_of_a_l217_217910

-- Define the quadratic inequality
def quadratic_inequality (a x : ℝ) : ℝ := (a-1)*x^2 + (a-1)*x + 1

theorem range_of_a :
  (∀ x : ℝ, quadratic_inequality a x > 0) ↔ (1 ≤ a ∧ a < 5) :=
by
  sorry

end range_of_a_l217_217910


namespace check_sufficient_condition_for_eq_l217_217020

theorem check_sufficient_condition_for_eq (a b c : ℤ) (h : a = c - 1 ∧ b = a - 1) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 = 1 := 
by
  sorry

end check_sufficient_condition_for_eq_l217_217020


namespace part1_part2_l217_217667

noncomputable def determinant (a b c d : ℤ) : ℤ :=
  a * d - b * c

-- Lean statement for Question (1)
theorem part1 :
  determinant 2022 2023 2021 2022 = 1 :=
by sorry

-- Lean statement for Question (2)
theorem part2 (m : ℤ) :
  determinant (m + 2) (m - 2) (m - 2) (m + 2) = 32 → m = 4 :=
by sorry

end part1_part2_l217_217667


namespace range_of_g_l217_217241

noncomputable def g (x : ℝ) : ℝ := (3 * x + 8 - 2 * x ^ 2) / (x + 4)

theorem range_of_g : 
  (∀ y : ℝ, ∃ x : ℝ, x ≠ -4 ∧ y = (3 * x + 8 - 2 * x^2) / (x + 4)) :=
by
  sorry

end range_of_g_l217_217241


namespace wine_count_l217_217757

theorem wine_count (S B total W : ℕ) (hS : S = 22) (hB : B = 17) (htotal : S - B + W = total) (htotal_val : total = 31) : W = 26 :=
by
  sorry

end wine_count_l217_217757


namespace problem_solution_l217_217051

theorem problem_solution (x m : ℝ) (h1 : x ≠ 0) (h2 : x / (x^2 - m*x + 1) = 1) :
  x^3 / (x^6 - m^3 * x^3 + 1) = 1 / (3 * m^2 - 2) :=
by
  sorry

end problem_solution_l217_217051


namespace find_total_amount_l217_217252

theorem find_total_amount (x : ℝ) (h₁ : 1.5 * x = 40) : x + 1.5 * x + 0.5 * x = 80.01 :=
by
  sorry

end find_total_amount_l217_217252


namespace degree_measure_OC1D_l217_217421

/-- Define points on the sphere -/
structure Point (latitude longitude : ℝ) :=
(lat : ℝ := latitude)
(long : ℝ := longitude)

noncomputable def cos_deg (deg : ℝ) : ℝ := Real.cos (deg * Real.pi / 180)

noncomputable def angle_OC1D : ℝ :=
  Real.arccos ((cos_deg 44) * (cos_deg (-123)))

/-- The main theorem: the degree measure of ∠OC₁D is 113 -/
theorem degree_measure_OC1D :
  angle_OC1D = 113 := sorry

end degree_measure_OC1D_l217_217421


namespace train_length_is_correct_l217_217308

noncomputable def speed_km_per_hr := 60
noncomputable def time_seconds := 15
noncomputable def speed_m_per_s : ℝ := (60 * 1000) / 3600
noncomputable def expected_length : ℝ := 250.05

theorem train_length_is_correct : (speed_m_per_s * time_seconds) = expected_length := by
  sorry

end train_length_is_correct_l217_217308


namespace matrix_determinant_zero_l217_217712

theorem matrix_determinant_zero (a b c : ℝ) : 
  Matrix.det (Matrix.of ![![1, a + b, b + c], ![1, a + 2 * b, b + 2 * c], ![1, a + 3 * b, b + 3 * c]]) = 0 := 
by
  sorry

end matrix_determinant_zero_l217_217712


namespace corresponding_angles_equal_l217_217148

-- Definition of corresponding angles (this should be previously defined, so here we assume it is just a predicate)
def CorrespondingAngles (a b : Angle) : Prop := sorry

-- The main theorem to be proven
theorem corresponding_angles_equal (a b : Angle) (h : CorrespondingAngles a b) : a = b := 
sorry

end corresponding_angles_equal_l217_217148


namespace expressions_equality_l217_217726

-- Assumptions that expressions (1) and (2) are well-defined (denominators are non-zero)
variable {a b c m n p : ℝ}
variable (h1 : m ≠ 0)
variable (h2 : bp + cn ≠ 0)
variable (h3 : n ≠ 0)
variable (h4 : ap + cm ≠ 0)

-- Main theorem statement
theorem expressions_equality
  (hS : (a / m) + (bc + np) / (bp + cn) = 0) :
  (b / n) + (ac + mp) / (ap + cm) = 0 :=
  sorry

end expressions_equality_l217_217726


namespace upload_time_l217_217489

theorem upload_time (file_size upload_speed : ℕ) (h_file_size : file_size = 160) (h_upload_speed : upload_speed = 8) : file_size / upload_speed = 20 :=
by
  sorry

end upload_time_l217_217489


namespace min_cos_C_l217_217106

theorem min_cos_C (a b c : ℝ) (A B C : ℝ) (h1 : a^2 + b^2 = (5 / 2) * c^2) 
  (h2 : ∃ (A B C : ℝ), a ≠ b ∧ 
    c = (a ^ 2 + b ^ 2 - 2 * a * b * (Real.cos C))) : 
  ∃ (C : ℝ), Real.cos C = 3 / 5 :=
by
  sorry

end min_cos_C_l217_217106


namespace intersection_A_B_l217_217284

-- Conditions
def A : Set ℝ := {1, 2, 0.5}
def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = x^2}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {1} :=
sorry

end intersection_A_B_l217_217284


namespace mixed_bead_cost_per_box_l217_217006

-- Definitions based on given conditions
def red_bead_cost : ℝ := 1.30
def yellow_bead_cost : ℝ := 2.00
def total_boxes : ℕ := 10
def red_boxes_used : ℕ := 4
def yellow_boxes_used : ℕ := 4

-- Theorem statement
theorem mixed_bead_cost_per_box :
  ((red_boxes_used * red_bead_cost) + (yellow_boxes_used * yellow_bead_cost)) / total_boxes = 1.32 :=
  by sorry

end mixed_bead_cost_per_box_l217_217006


namespace int_to_fourth_power_l217_217750

theorem int_to_fourth_power:
  3^4 * 9^8 = 243^4 :=
by 
  sorry

end int_to_fourth_power_l217_217750


namespace polygon_six_sides_l217_217598

theorem polygon_six_sides (n : ℕ) (h1 : (n - 2) * 180 = 2 * 360) : n = 6 := by
  sorry

end polygon_six_sides_l217_217598


namespace mike_spent_total_l217_217818

-- Define the prices of the items
def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84

-- Define the total price calculation
def total_price : ℝ := trumpet_price + song_book_price

-- The theorem statement asserting the total price
theorem mike_spent_total : total_price = 151.00 :=
by
  sorry

end mike_spent_total_l217_217818


namespace cos_diff_proof_l217_217475

noncomputable def cos_diff (α β : ℝ) : ℝ := Real.cos (α - β)

theorem cos_diff_proof (α β : ℝ) 
  (h1 : Real.cos α - Real.cos β = 1 / 2)
  (h2 : Real.sin α - Real.sin β = 1 / 3) :
  cos_diff α β = 59 / 72 := by
  sorry

end cos_diff_proof_l217_217475


namespace james_meditation_time_is_30_l217_217169

noncomputable def james_meditation_time_per_session 
  (sessions_per_day : ℕ) 
  (days_per_week : ℕ) 
  (hours_per_week : ℕ) 
  (minutes_per_hour : ℕ) : ℕ :=
  (hours_per_week * minutes_per_hour) / (sessions_per_day * days_per_week)

theorem james_meditation_time_is_30
  (sessions_per_day : ℕ) 
  (days_per_week : ℕ) 
  (hours_per_week : ℕ) 
  (minutes_per_hour : ℕ) 
  (h_sessions : sessions_per_day = 2) 
  (h_days : days_per_week = 7) 
  (h_hours : hours_per_week = 7) 
  (h_minutes : minutes_per_hour = 60) : 
  james_meditation_time_per_session sessions_per_day days_per_week hours_per_week minutes_per_hour = 30 := by
  sorry

end james_meditation_time_is_30_l217_217169


namespace inequality_proof_l217_217870

theorem inequality_proof (a b c : ℝ)
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |c * x^2 + b * x + a| ≤ 2 :=
by
  sorry

end inequality_proof_l217_217870


namespace relationship_of_variables_l217_217898

variable {a b c d : ℝ}

theorem relationship_of_variables 
  (h1 : d - a < c - b) 
  (h2 : c - b < 0) 
  (h3 : d - b = c - a) : 
  d < c ∧ c < b ∧ b < a := 
sorry

end relationship_of_variables_l217_217898


namespace intersection_of_A_and_B_l217_217586

def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 < x ∧ x < 5}

theorem intersection_of_A_and_B :
  A ∩ B = {3, 4} := 
by 
  sorry

end intersection_of_A_and_B_l217_217586


namespace gymnast_scores_difference_l217_217519

theorem gymnast_scores_difference
  (s1 s2 s3 s4 s5 : ℝ)
  (h1 : (s2 + s3 + s4 + s5) / 4 = 9.46)
  (h2 : (s1 + s2 + s3 + s4) / 4 = 9.66)
  (h3 : (s2 + s3 + s4) / 3 = 9.58)
  : |s5 - s1| = 8.3 :=
sorry

end gymnast_scores_difference_l217_217519


namespace arithmetic_mean_124_4_31_l217_217891

theorem arithmetic_mean_124_4_31 :
  let numbers := [12, 25, 39, 48]
  let total := 124
  let count := 4
  (total / count : ℝ) = 31 := by
  sorry

end arithmetic_mean_124_4_31_l217_217891


namespace ratio_of_time_charged_l217_217432

theorem ratio_of_time_charged (P K M : ℕ) (r : ℚ) 
  (h1 : P + K + M = 144) 
  (h2 : P = r * K)
  (h3 : P = 1/3 * M)
  (h4 : M = K + 80) : 
  r = 2 := 
  sorry

end ratio_of_time_charged_l217_217432


namespace truck_transportation_l217_217524

theorem truck_transportation
  (x y t : ℕ) 
  (h1 : xt - yt = 60)
  (h2 : (x - 4) * (t + 10) = xt)
  (h3 : (y - 3) * (t + 10) = yt)
  (h4 : xt = x * t)
  (h5 : yt = y * t) : 
  x - 4 = 8 ∧ y - 3 = 6 ∧ t + 10 = 30 := 
by
  sorry

end truck_transportation_l217_217524


namespace math_problem_l217_217453

noncomputable def f (x : ℝ) := (x^9 - 27*x^6 + 216*x^3 - 512) / (x^3 - 8)

theorem math_problem : f 6 = 43264 := by
  sorry

end math_problem_l217_217453


namespace alex_new_salary_in_may_l217_217686

def initial_salary : ℝ := 50000
def february_increase (s : ℝ) : ℝ := s * 1.10
def april_bonus (s : ℝ) : ℝ := s + 2000
def may_pay_cut (s : ℝ) : ℝ := s * 0.95

theorem alex_new_salary_in_may : may_pay_cut (april_bonus (february_increase initial_salary)) = 54150 :=
by
  sorry

end alex_new_salary_in_may_l217_217686


namespace algebraic_expression_value_l217_217176

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x + 3 = 7) : 3 * x^2 + 3 * x + 7 = 19 :=
sorry

end algebraic_expression_value_l217_217176


namespace total_prairie_area_l217_217842

theorem total_prairie_area (A B C : ℕ) (Z1 Z2 Z3 : ℚ) (unaffected : ℕ) (total_area : ℕ) : 
  A = 55000 →
  B = 35000 →
  C = 45000 →
  Z1 = 0.80 →
  Z2 = 0.60 →
  Z3 = 0.95 →
  unaffected = 1500 →
  total_area = Z1 * A + Z2 * B + Z3 * C + unaffected →
  total_area = 109250 := sorry

end total_prairie_area_l217_217842


namespace greatest_possible_percentage_of_airlines_both_services_l217_217688

noncomputable def maxPercentageOfAirlinesWithBothServices (percentageInternet percentageSnacks : ℝ) : ℝ :=
  if percentageInternet <= percentageSnacks then percentageInternet else percentageSnacks

theorem greatest_possible_percentage_of_airlines_both_services:
  let p_internet := 0.35
  let p_snacks := 0.70
  maxPercentageOfAirlinesWithBothServices p_internet p_snacks = 0.35 :=
by
  sorry

end greatest_possible_percentage_of_airlines_both_services_l217_217688


namespace simplify_fraction_to_9_l217_217853

-- Define the necessary terms and expressions
def problem_expr := (3^12)^2 - (3^10)^2
def problem_denom := (3^11)^2 - (3^9)^2
def simplified_expr := problem_expr / problem_denom

-- State the theorem we want to prove
theorem simplify_fraction_to_9 : simplified_expr = 9 := 
by sorry

end simplify_fraction_to_9_l217_217853


namespace range_of_a_l217_217743

variable (A B : Set ℝ) (a : ℝ)

def setA : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def setB : Set ℝ := {x | (2^(1 - x) + a ≤ 0) ∧ (x^2 - 2*(a + 7)*x + 5 ≤ 0)}

theorem range_of_a :
  A ⊆ B ↔ (-4 ≤ a) ∧ (a ≤ -1) :=
by
  sorry

end range_of_a_l217_217743


namespace geometric_sequence_reciprocals_sum_l217_217150

theorem geometric_sequence_reciprocals_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), 
    (a 1 = 2) ∧ 
    (a 1 + a 3 + a 5 = 14) ∧ 
    (∀ n : ℕ, a (n + 1) = a n * q) → 
      (1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8) :=
sorry

end geometric_sequence_reciprocals_sum_l217_217150


namespace factor_theorem_solution_l217_217675

theorem factor_theorem_solution (t : ℝ) :
  (∃ p q : ℝ, 10 * p * q = 10 * t * t + 21 * t - 10 ∧ (x - q) = (x - t)) →
  t = 2 / 5 ∨ t = -5 / 2 := by
  sorry

end factor_theorem_solution_l217_217675


namespace minimum_positive_period_minimum_value_l217_217463

noncomputable def f (x : Real) : Real :=
  Real.sin (x / 5) - Real.cos (x / 5)

theorem minimum_positive_period (T : Real) : (∀ x, f (x + T) = f x) ∧ T > 0 → T = 10 * Real.pi :=
  sorry

theorem minimum_value : ∃ x, f x = -Real.sqrt 2 :=
  sorry

end minimum_positive_period_minimum_value_l217_217463


namespace maria_original_number_25_3_l217_217293

theorem maria_original_number_25_3 (x : ℚ) 
  (h : ((3 * (x + 3) - 4) / 3) = 10) : 
  x = 25 / 3 := 
by 
  sorry

end maria_original_number_25_3_l217_217293


namespace pen_price_relationship_l217_217897

variable (x : ℕ) -- x represents the number of pens
variable (y : ℝ) -- y represents the total selling price in dollars
variable (p : ℝ) -- p represents the price per pen

-- Each box contains 10 pens
def pens_per_box := 10

-- Each box is sold for $16
def price_per_box := 16

-- Given the conditions, prove the relationship between y and x
theorem pen_price_relationship (hx : x = 10) (hp : p = 16) :
  y = 1.6 * x := sorry

end pen_price_relationship_l217_217897


namespace sum_algebra_values_l217_217826

def alphabet_value (n : ℕ) : ℤ :=
  match n % 8 with
  | 1 => 3
  | 2 => 1
  | 3 => 0
  | 4 => -1
  | 5 => -3
  | 6 => -1
  | 7 => 0
  | _ => 1

theorem sum_algebra_values : 
  alphabet_value 1 + 
  alphabet_value 12 + 
  alphabet_value 7 +
  alphabet_value 5 +
  alphabet_value 2 +
  alphabet_value 18 +
  alphabet_value 1 
  = 5 := by
  sorry

end sum_algebra_values_l217_217826


namespace Jessie_lost_7_kilograms_l217_217387

def Jessie_previous_weight : ℕ := 74
def Jessie_current_weight : ℕ := 67
def Jessie_weight_lost : ℕ := Jessie_previous_weight - Jessie_current_weight

theorem Jessie_lost_7_kilograms : Jessie_weight_lost = 7 :=
by
  sorry

end Jessie_lost_7_kilograms_l217_217387


namespace sum_of_first_41_terms_is_94_l217_217723

def equal_product_sequence (a : ℕ → ℕ) (k : ℕ) : Prop := 
∀ (n : ℕ), a (n+1) * a (n+2) * a (n+3) = k

theorem sum_of_first_41_terms_is_94
  (a : ℕ → ℕ)
  (h1 : equal_product_sequence a 8)
  (h2 : a 1 = 1)
  (h3 : a 2 = 2) :
  (Finset.range 41).sum a = 94 :=
by
  sorry

end sum_of_first_41_terms_is_94_l217_217723


namespace quadratic_inequality_solution_l217_217819

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + 3 * x - 18 > 0) ↔ (x < -6 ∨ x > 3) := 
sorry

end quadratic_inequality_solution_l217_217819


namespace total_flour_l217_217537

theorem total_flour (original_flour extra_flour : Real) (h_orig : original_flour = 7.0) (h_extra : extra_flour = 2.0) : original_flour + extra_flour = 9.0 :=
sorry

end total_flour_l217_217537


namespace transition_algebraic_expression_l217_217997

theorem transition_algebraic_expression (k : ℕ) (hk : k > 0) :
  (k + 1 + k) * (k + 1 + k + 1) / (k + 1) = 4 * k + 2 :=
sorry

end transition_algebraic_expression_l217_217997


namespace batsman_average_after_11th_inning_l217_217074

theorem batsman_average_after_11th_inning (x : ℝ) (h : 10 * x + 110 = 11 * (x + 5)) : 
    (10 * x + 110) / 11 = 60 := by
  sorry

end batsman_average_after_11th_inning_l217_217074


namespace evaluate_expression_l217_217201

theorem evaluate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5) + 1) = 107 :=
by
  -- The proof will go here.
  sorry

end evaluate_expression_l217_217201


namespace fib_inequality_l217_217094

def Fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => Fib n + Fib (n + 1)

theorem fib_inequality {n : ℕ} (h : 2 ≤ n) : Fib (n + 5) > 10 * Fib n :=
  sorry

end fib_inequality_l217_217094


namespace cube_sum_l217_217722

theorem cube_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 15) : x^3 + y^3 = 550 := by
  sorry

end cube_sum_l217_217722


namespace num_triangles_l217_217540

def vertices := 10
def chosen_vertices := 3

theorem num_triangles : (Nat.choose vertices chosen_vertices) = 120 := by
  sorry

end num_triangles_l217_217540


namespace net_profit_is_correct_l217_217660

-- Define the purchase price, markup, and overhead percentage
def purchase_price : ℝ := 48
def markup : ℝ := 55
def overhead_percentage : ℝ := 0.30

-- Define the overhead cost calculation
def overhead_cost : ℝ := overhead_percentage * purchase_price

-- Define the net profit calculation
def net_profit : ℝ := markup - overhead_cost

-- State the theorem
theorem net_profit_is_correct : net_profit = 40.60 :=
by
  sorry

end net_profit_is_correct_l217_217660


namespace students_who_won_first_prize_l217_217907

theorem students_who_won_first_prize :
  ∃ x : ℤ, 30 ≤ x ∧ x ≤ 55 ∧ (x % 3 = 2) ∧ (x % 5 = 4) ∧ (x % 7 = 2) ∧ x = 44 :=
by
  sorry

end students_who_won_first_prize_l217_217907


namespace value_of_expression_l217_217771

theorem value_of_expression (x : ℝ) : 
  let a := 2000 * x + 2001
  let b := 2000 * x + 2002
  let c := 2000 * x + 2003
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end value_of_expression_l217_217771


namespace sqrt_meaningful_value_x_l217_217516

theorem sqrt_meaningful_value_x (x : ℝ) (h : x-1 ≥ 0) : x = 2 :=
by
  sorry

end sqrt_meaningful_value_x_l217_217516


namespace truck_travel_distance_l217_217927

theorem truck_travel_distance (b t : ℝ) (ht : t > 0) (ht30 : t + 30 > 0) : 
  let converted_feet := 4 * 60
  let time_half := converted_feet / 2
  let speed_first_half := b / 4
  let speed_second_half := b / 4
  let distance_first_half := speed_first_half * time_half / t
  let distance_second_half := speed_second_half * time_half / (t + 30)
  let total_distance_feet := distance_first_half + distance_second_half
  let result_yards := total_distance_feet / 3
  result_yards = (10 * b / t) + (10 * b / (t + 30))
:= by
  -- proof skipped
  sorry

end truck_travel_distance_l217_217927


namespace value_of_M_l217_217004

theorem value_of_M (M : ℝ) (h : (20 / 100) * M = (60 / 100) * 1500) : M = 4500 :=
by {
    sorry
}

end value_of_M_l217_217004


namespace find_a_l217_217969

noncomputable def A (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }
def B : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }

theorem find_a (a : ℝ) :
  A a ∪ B = B ↔ a = 0 ∨ a = 1 ∨ a = 1 / 2 :=
sorry

end find_a_l217_217969


namespace remainder_eq_one_l217_217925

theorem remainder_eq_one (n : ℤ) (h : n % 6 = 1) : (n + 150) % 6 = 1 := 
by
  sorry

end remainder_eq_one_l217_217925


namespace min_value_of_expression_l217_217697

/-- 
Given α and β are the two real roots of the quadratic equation x^2 - 2a * x + a + 6 = 0,
prove that the minimum value of (α - 1)^2 + (β - 1)^2 is 8.
-/
theorem min_value_of_expression (a α β : ℝ) (h1 : α ^ 2 - 2 * a * α + a + 6 = 0) (h2 : β ^ 2 - 2 * a * β + a + 6 = 0) :
  (α - 1)^2 + (β - 1)^2 ≥ 8 := 
sorry

end min_value_of_expression_l217_217697


namespace RSA_next_challenge_digits_l217_217523

theorem RSA_next_challenge_digits (previous_digits : ℕ) (prize_increase : ℕ) :
  previous_digits = 193 ∧ prize_increase > 10000 → ∃ N : ℕ, N = 212 :=
by {
  sorry -- Proof is omitted
}

end RSA_next_challenge_digits_l217_217523


namespace geometric_sequence_problem_l217_217698

theorem geometric_sequence_problem (a : ℕ → ℤ)
  (q : ℤ)
  (h1 : a 2 * a 5 = -32)
  (h2 : a 3 + a 4 = 4)
  (hq : ∃ (k : ℤ), q = k) :
  a 9 = -256 := 
sorry

end geometric_sequence_problem_l217_217698


namespace smallest_class_size_l217_217808

theorem smallest_class_size (n : ℕ) 
  (eight_students_scored_120 : 8 * 120 ≤ n * 92)
  (three_students_scored_115 : 3 * 115 ≤ n * 92)
  (min_score_70 : 70 * n ≤ n * 92)
  (mean_score_92 : (8 * 120 + 3 * 115 + 70 * (n - 11)) / n = 92) :
  n = 25 :=
by
  sorry

end smallest_class_size_l217_217808


namespace actual_height_is_191_l217_217996

theorem actual_height_is_191 :
  ∀ (n incorrect_avg correct_avg incorrect_height x : ℝ),
  n = 20 ∧ incorrect_avg = 175 ∧ correct_avg = 173 ∧ incorrect_height = 151 ∧
  (n * incorrect_avg - n * correct_avg = x - incorrect_height) →
  x = 191 :=
by
  intros n incorrect_avg correct_avg incorrect_height x h
  -- skip the proof part
  sorry

end actual_height_is_191_l217_217996


namespace principal_amount_l217_217336

theorem principal_amount (P : ℝ) (SI : ℝ) (R : ℝ) (T : ℝ)
  (h1 : SI = (P * R * T) / 100)
  (h2 : SI = 640)
  (h3 : R = 8)
  (h4 : T = 2) :
  P = 4000 :=
sorry

end principal_amount_l217_217336


namespace books_brought_back_l217_217916

def initial_books : ℕ := 235
def taken_out_tuesday : ℕ := 227
def taken_out_friday : ℕ := 35
def books_remaining : ℕ := 29

theorem books_brought_back (B : ℕ) :
  B = 56 ↔ (initial_books - taken_out_tuesday + B - taken_out_friday = books_remaining) :=
by
  -- proof steps would go here
  sorry

end books_brought_back_l217_217916


namespace find_g1_gneg1_l217_217111

variables {f g : ℝ → ℝ}

theorem find_g1_gneg1 (h1 : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y)
                      (h2 : f (-2) = f 1 ∧ f 1 ≠ 0) :
  g 1 + g (-1) = -1 :=
sorry

end find_g1_gneg1_l217_217111


namespace sqrt_square_eq_self_l217_217878

theorem sqrt_square_eq_self (a : ℝ) (h : a ≥ 1/2) :
  Real.sqrt ((2 * a - 1) ^ 2) = 2 * a - 1 :=
by
  sorry

end sqrt_square_eq_self_l217_217878


namespace gcd_square_of_difference_l217_217492

theorem gcd_square_of_difference (x y z : ℕ) (h : 1/x - 1/y = 1/z) :
  ∃ k : ℕ, (Nat.gcd (Nat.gcd x y) z) * (y - x) = k^2 :=
by
  sorry

end gcd_square_of_difference_l217_217492


namespace locus_midpoint_l217_217204

-- Conditions
def hyperbola_eq (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1

def perpendicular_rays (OA OB : ℝ × ℝ) : Prop := (OA.1 * OB.1 + OA.2 * OB.2) = 0 -- Dot product zero for perpendicularity

-- Given the hyperbola and perpendicularity conditions, prove the locus equation
theorem locus_midpoint (x y : ℝ) :
  (∃ A B : ℝ × ℝ, hyperbola_eq A.1 A.2 ∧ hyperbola_eq B.1 B.2 ∧ perpendicular_rays A B ∧
  x = (A.1 + B.1) / 2 ∧ y = (A.2 + B.2) / 2) → 3 * (4 * x^2 - y^2)^2 = 4 * (16 * x^2 + y^2) :=
sorry

end locus_midpoint_l217_217204


namespace no_positive_rational_solutions_l217_217278

theorem no_positive_rational_solutions (n : ℕ) (h_pos_n : 0 < n) : 
  ¬ ∃ (x y : ℚ) (h_x_pos : 0 < x) (h_y_pos : 0 < y), x + y + (1/x) + (1/y) = 3 * n :=
by
  sorry

end no_positive_rational_solutions_l217_217278


namespace scatter_plot_variable_placement_l217_217565

theorem scatter_plot_variable_placement
  (forecast explanatory : Type)
  (scatter_plot : explanatory → forecast → Prop) : 
  ∀ (x : explanatory) (y : forecast), scatter_plot x y → (True -> True) := 
by
  intros x y h
  sorry

end scatter_plot_variable_placement_l217_217565


namespace prod_sum_reciprocal_bounds_l217_217713

-- Define the product of the sum of three positive numbers and the sum of their reciprocals.
theorem prod_sum_reciprocal_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  9 ≤ (a + b + c) * (1 / a + 1 / b + 1 / c) :=
by
  sorry

end prod_sum_reciprocal_bounds_l217_217713


namespace necklace_cost_l217_217545

theorem necklace_cost (N : ℕ) (h1 : N + (N + 5) = 73) : N = 34 := by
  sorry

end necklace_cost_l217_217545


namespace sheila_hourly_wage_is_correct_l217_217728

-- Definitions based on conditions
def works_hours_per_day_mwf : ℕ := 8
def works_days_mwf : ℕ := 3
def works_hours_per_day_tt : ℕ := 6
def works_days_tt : ℕ := 2
def weekly_earnings : ℕ := 216

-- Total calculated hours based on the problem conditions
def total_weekly_hours : ℕ := (works_hours_per_day_mwf * works_days_mwf) + (works_hours_per_day_tt * works_days_tt)

-- Target wage per hour
def wage_per_hour : ℕ := weekly_earnings / total_weekly_hours

-- The theorem stating the proof problem
theorem sheila_hourly_wage_is_correct : wage_per_hour = 6 := by
  sorry

end sheila_hourly_wage_is_correct_l217_217728


namespace find_divisor_l217_217409

variable (r q d v : ℕ)
variable (h1 : r = 8)
variable (h2 : q = 43)
variable (h3 : d = 997)

theorem find_divisor : d = v * q + r → v = 23 :=
by
  sorry

end find_divisor_l217_217409


namespace fraction_zero_condition_l217_217866

theorem fraction_zero_condition (x : ℝ) (h1 : (3 - |x|) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 3 :=
by
  sorry

end fraction_zero_condition_l217_217866


namespace largest_gcd_sum_1089_l217_217325

theorem largest_gcd_sum_1089 (c d : ℕ) (h₁ : 0 < c) (h₂ : 0 < d) (h₃ : c + d = 1089) : ∃ k, k = Nat.gcd c d ∧ k = 363 :=
by
  sorry

end largest_gcd_sum_1089_l217_217325


namespace richmond_population_l217_217684

theorem richmond_population (R V B : ℕ) (h0 : R = V + 1000) (h1 : V = 4 * B) (h2 : B = 500) : R = 3000 :=
by
  -- skipping proof
  sorry

end richmond_population_l217_217684


namespace lines_are_perpendicular_l217_217267

noncomputable def line1 := {x : ℝ | ∃ y : ℝ, x + y - 1 = 0}
noncomputable def line2 := {x : ℝ | ∃ y : ℝ, x - y + 1 = 0}

theorem lines_are_perpendicular : 
  let slope1 := -1
  let slope2 := 1
  slope1 * slope2 = -1 := sorry

end lines_are_perpendicular_l217_217267


namespace minimum_cars_with_racing_stripes_l217_217597

-- Definitions and conditions
variable (numberOfCars : ℕ) (withoutAC : ℕ) (maxWithACWithoutStripes : ℕ)

axiom total_number_of_cars : numberOfCars = 100
axiom cars_without_ac : withoutAC = 49
axiom max_ac_without_stripes : maxWithACWithoutStripes = 49    

-- Proposition
theorem minimum_cars_with_racing_stripes 
  (total_number_of_cars : numberOfCars = 100) 
  (cars_without_ac : withoutAC = 49)
  (max_ac_without_stripes : maxWithACWithoutStripes = 49) :
  ∃ (R : ℕ), R = 2 :=
by
  sorry

end minimum_cars_with_racing_stripes_l217_217597


namespace line_slope_intercept_sum_l217_217011

theorem line_slope_intercept_sum (m b : ℝ)
    (h1 : m = 4)
    (h2 : ∃ b, ∀ x y : ℝ, y = mx + b → y = 5 ∧ x = -2)
    : m + b = 17 := by
  sorry

end line_slope_intercept_sum_l217_217011


namespace point_to_focus_distance_l217_217274

def parabola : Set (ℝ × ℝ) := { p | p.2^2 = 4 * p.1 }

def point_P : ℝ × ℝ := (3, 2) -- Since y^2 = 4*3 hence y = ±2 and we choose one of the (3, 2) or (3, -2)

def focus_F : ℝ × ℝ := (1, 0) -- Focus of y^2 = 4x is (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem point_to_focus_distance : distance point_P focus_F = 4 := by
  sorry -- Proof goes here

end point_to_focus_distance_l217_217274


namespace cosine_square_plus_alpha_sine_l217_217753

variable (α : ℝ)

theorem cosine_square_plus_alpha_sine (h1 : 0 ≤ α) (h2 : α ≤ Real.pi / 2) : 
  Real.cos α * Real.cos α + α * Real.sin α ≥ 1 :=
sorry

end cosine_square_plus_alpha_sine_l217_217753


namespace sequence_formula_l217_217949

theorem sequence_formula (a : ℕ → ℚ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 = -1/2) (h3 : a 3 = 1/3) (h4 : a 4 = -1/4) :
  a n = (-1)^(n+1) * (1/n) :=
sorry

end sequence_formula_l217_217949


namespace nelly_part_payment_is_875_l217_217882

noncomputable def part_payment (total_cost remaining_amount : ℝ) :=
  0.25 * total_cost

theorem nelly_part_payment_is_875 (total_cost : ℝ) (remaining_amount : ℝ)
  (h1 : remaining_amount = 2625)
  (h2 : remaining_amount = 0.75 * total_cost) :
  part_payment total_cost remaining_amount = 875 :=
by
  sorry

end nelly_part_payment_is_875_l217_217882


namespace find_x_value_l217_217725

theorem find_x_value (x : ℝ) 
  (h₁ : 1 / (x + 8) + 2 / (3 * x) + 1 / (2 * x) = 1 / (2 * x)) :
  x = (-1 + Real.sqrt 97) / 6 :=
sorry

end find_x_value_l217_217725


namespace ab_difference_l217_217483

theorem ab_difference (a b : ℤ) (h1 : |a| = 5) (h2 : |b| = 3) (h3 : a + b > 0) : a - b = 2 ∨ a - b = 8 :=
sorry

end ab_difference_l217_217483


namespace john_age_l217_217137

/-
Problem statement:
John is 24 years younger than his dad. The sum of their ages is 68 years.
We need to prove that John is 22 years old.
-/

theorem john_age:
  ∃ (j d : ℕ), (j = d - 24 ∧ j + d = 68) → j = 22 :=
by
  sorry

end john_age_l217_217137


namespace books_sold_l217_217872

def initial_books : ℕ := 134
def given_books : ℕ := 39
def books_left : ℕ := 68

theorem books_sold : (initial_books - given_books - books_left = 27) := 
by 
  sorry

end books_sold_l217_217872


namespace inequality_solution_real_l217_217699

theorem inequality_solution_real (x : ℝ) :
  (x + 1) * (2 - x) < 4 ↔ true :=
by
  sorry

end inequality_solution_real_l217_217699


namespace solve_eq1_solve_eq2_l217_217344

-- Proof problem 1: Prove that under the condition 6x - 4 = 3x + 2, x = 2
theorem solve_eq1 : ∀ x : ℝ, 6 * x - 4 = 3 * x + 2 → x = 2 :=
by
  intro x
  intro h
  sorry

-- Proof problem 2: Prove that under the condition (x / 4) - (3 / 5) = (x + 1) / 2, x = -22/5
theorem solve_eq2 : ∀ x : ℝ, (x / 4) - (3 / 5) = (x + 1) / 2 → x = -(22 / 5) :=
by
  intro x
  intro h
  sorry

end solve_eq1_solve_eq2_l217_217344


namespace smallest_possible_X_l217_217246

theorem smallest_possible_X (T : ℕ) (h1 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) (h2 : T % 24 = 0) :
  ∃ (X : ℕ), X = T / 24 ∧ X = 4625 :=
  sorry

end smallest_possible_X_l217_217246


namespace angle_in_fourth_quadrant_l217_217428

theorem angle_in_fourth_quadrant (α : ℝ) (h : 0 < α ∧ α < 90) : 270 < 360 - α ∧ 360 - α < 360 :=
by
  sorry

end angle_in_fourth_quadrant_l217_217428


namespace proof_product_eq_l217_217186

theorem proof_product_eq (a b c d : ℚ) (h1 : 2 * a + 3 * b + 5 * c + 7 * d = 42)
    (h2 : 4 * (d + c) = b) (h3 : 2 * b + 2 * c = a) (h4 : c - 2 = d) :
    a * b * c * d = -26880 / 729 := by
  sorry

end proof_product_eq_l217_217186


namespace largest_positive_integer_solution_l217_217832

theorem largest_positive_integer_solution (x : ℕ) (h₁ : 1 ≤ x) (h₂ : x + 3 ≤ 6) : 
  x = 3 := by
  sorry

end largest_positive_integer_solution_l217_217832


namespace min_flowers_for_bouquets_l217_217407

open Classical

noncomputable def minimum_flowers (types : ℕ) (flowers_for_bouquet : ℕ) (bouquets : ℕ) : ℕ := 
  sorry

theorem min_flowers_for_bouquets : minimum_flowers 6 5 10 = 70 := 
  sorry

end min_flowers_for_bouquets_l217_217407


namespace sum_of_legs_of_right_triangle_l217_217095

theorem sum_of_legs_of_right_triangle (y : ℤ) (hyodd : y % 2 = 1) (hyp : y ^ 2 + (y + 2) ^ 2 = 17 ^ 2) :
  y + (y + 2) = 24 :=
sorry

end sum_of_legs_of_right_triangle_l217_217095


namespace pieces_cut_from_rod_l217_217560

theorem pieces_cut_from_rod (rod_length_m : ℝ) (piece_length_cm : ℝ) (rod_length_cm_eq : rod_length_m * 100 = 4250) (piece_length_eq : piece_length_cm = 85) :
  (4250 / 85) = 50 :=
by sorry

end pieces_cut_from_rod_l217_217560


namespace intersection_is_2_l217_217905

-- Define the sets A and B
def A : Set ℝ := { x | x < 1 }
def B : Set ℝ := { -1, 0, 2 }

-- Define the complement of A
def A_complement : Set ℝ := { x | x ≥ 1 }

-- Define the intersection of the complement of A and B
def intersection : Set ℝ := A_complement ∩ B

-- Prove that the intersection is {2}
theorem intersection_is_2 : intersection = {2} := by
  sorry

end intersection_is_2_l217_217905


namespace ab_divisibility_l217_217458

theorem ab_divisibility (a b : ℕ) (h_a : a ≥ 2) (h_b : b ≥ 2) : 
  (ab - 1) % ((a - 1) * (b - 1)) = 0 ↔ (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) :=
sorry

end ab_divisibility_l217_217458


namespace sum_of_three_numbers_l217_217525

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h : a + (b * c) = (a + b) * (a + c)) : a + b + c = 1 :=
by
  sorry

end sum_of_three_numbers_l217_217525


namespace find_AC_find_angle_A_l217_217541

noncomputable def triangle_AC (AB BC : ℝ) (sinC_over_sinB : ℝ) : ℝ :=
  if h : sinC_over_sinB = 3 / 5 ∧ AB = 3 ∧ BC = 7 then 5 else 0

noncomputable def triangle_angle_A (AB AC BC : ℝ) : ℝ :=
  if h : AB = 3 ∧ AC = 5 ∧ BC = 7 then 120 else 0

theorem find_AC (BC AB : ℝ) (sinC_over_sinB : ℝ) (h : BC = 7 ∧ AB = 3 ∧ sinC_over_sinB = 3 / 5) : 
  triangle_AC AB BC sinC_over_sinB = 5 := by
  sorry

theorem find_angle_A (BC AB AC : ℝ) (h : BC = 7 ∧ AB = 3 ∧ AC = 5) : 
  triangle_angle_A AB AC BC = 120 := by
  sorry

end find_AC_find_angle_A_l217_217541


namespace sufficient_condition_m_ge_4_range_of_x_for_m5_l217_217803

variable (x m : ℝ)

-- Problem (1)
theorem sufficient_condition_m_ge_4 (h : m > 0)
  (hpq : ∀ x, ((x + 2) * (x - 6) ≤ 0) → (2 - m ≤ x ∧ x ≤ 2 + m)) : m ≥ 4 := by
  sorry

-- Problem (2)
theorem range_of_x_for_m5 (h : m = 5)
  (hp_or_q : ∀ x, ((x + 2) * (x - 6) ≤ 0 ∨ (-3 ≤ x ∧ x ≤ 7)) )
  (hp_and_not_q : ∀ x, ¬(((x + 2) * (x - 6) ≤ 0) ∧ (-3 ≤ x ∧ x ≤ 7))):
  ∀ x, x ∈ Set.Ico (-3) (-2) ∨ x ∈ Set.Ioc (6) (7) := by
  sorry

end sufficient_condition_m_ge_4_range_of_x_for_m5_l217_217803


namespace becky_necklaces_count_l217_217259

-- Define the initial conditions
def initial_necklaces := 50
def broken_necklaces := 3
def new_necklaces := 5
def given_away_necklaces := 15

-- Define the final number of necklaces
def final_necklaces (initial : Nat) (broken : Nat) (bought : Nat) (given_away : Nat) : Nat :=
  initial - broken + bought - given_away

-- The theorem stating that after performing the series of operations,
-- Becky should have 37 necklaces.
theorem becky_necklaces_count :
  final_necklaces initial_necklaces broken_necklaces new_necklaces given_away_necklaces = 37 :=
  by
    -- This proof is just a placeholder to ensure the code can be built successfully.
    -- Actual proof logic needs to be filled in to complete the theorem.
    sorry

end becky_necklaces_count_l217_217259


namespace debby_pictures_l217_217464

theorem debby_pictures : 
  let zoo_pics := 24
  let museum_pics := 12
  let pics_deleted := 14
  zoo_pics + museum_pics - pics_deleted = 22 := 
by
  sorry

end debby_pictures_l217_217464


namespace range_of_a_l217_217016

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (2 * x - 1 < 3) ∧ (x - a < 0) → (x < a)) → (a ≤ 2) :=
by
  intro h
  sorry

end range_of_a_l217_217016


namespace ratio_of_ages_l217_217678

theorem ratio_of_ages (Sandy_age : ℕ) (Molly_age : ℕ)
  (h1 : Sandy_age = 56)
  (h2 : Molly_age = Sandy_age + 16) :
  (Sandy_age : ℚ) / Molly_age = 7 / 9 :=
by
  -- Proof goes here
  sorry

end ratio_of_ages_l217_217678


namespace find_constants_l217_217248

theorem find_constants (a b c : ℝ) (h_neq_0_a : a ≠ 0) (h_neq_0_b : b ≠ 0) 
(h_neq_0_c : c ≠ 0) 
(h_eq1 : a * b = 3 * (a + b)) 
(h_eq2 : b * c = 4 * (b + c)) 
(h_eq3 : a * c = 5 * (a + c)) : 
a = 120 / 17 ∧ b = 120 / 23 ∧ c = 120 / 7 := 
  sorry

end find_constants_l217_217248


namespace original_price_is_100_l217_217973

variable (P : ℝ) -- Declare the original price P as a real number
variable (h : 0.10 * P = 10) -- The condition given in the problem

theorem original_price_is_100 (P : ℝ) (h : 0.10 * P = 10) : P = 100 := by
  sorry

end original_price_is_100_l217_217973


namespace evaluate_expression_l217_217855

def f (x : ℕ) : ℕ :=
  match x with
  | 3 => 10
  | 4 => 17
  | 5 => 26
  | 6 => 37
  | 7 => 50
  | _ => 0  -- for any x not in the table, f(x) is undefined and defaults to 0

def f_inv (y : ℕ) : ℕ :=
  match y with
  | 10 => 3
  | 17 => 4
  | 26 => 5
  | 37 => 6
  | 50 => 7
  | _ => 0  -- for any y not in the table, f_inv(y) is undefined and defaults to 0

theorem evaluate_expression :
  f_inv (f_inv 50 * f_inv 10 + f_inv 26) = 5 :=
by
  sorry

end evaluate_expression_l217_217855


namespace sequence_sum_l217_217558

-- Definitions for the sequences
def a (n : ℕ) : ℕ := n + 1
def b (n : ℕ) : ℕ := 2^(n-1)

-- The theorem we need to prove
theorem sequence_sum : a (b 1) + a (b 2) + a (b 3) + a (b 4) = 19 := by
  sorry

end sequence_sum_l217_217558


namespace max_n_value_l217_217029

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem max_n_value (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n → (2 * (n + 0.5) = a n + a (n + 1))) 
  (h2 : S a 63 = 2020) (h3 : a 2 < 3) : 63 ∈ { n : ℕ | S a n = 2020 } :=
sorry

end max_n_value_l217_217029


namespace quarterly_business_tax_cost_l217_217124

theorem quarterly_business_tax_cost
    (price_federal : ℕ := 50)
    (price_state : ℕ := 30)
    (Q : ℕ)
    (num_federal : ℕ := 60)
    (num_state : ℕ := 20)
    (num_quart_business : ℕ := 10)
    (total_revenue : ℕ := 4400)
    (revenue_equation : num_federal * price_federal + num_state * price_state + num_quart_business * Q = total_revenue) :
    Q = 80 :=
by 
  sorry

end quarterly_business_tax_cost_l217_217124


namespace negation_of_exists_l217_217034

-- Lean definition of the proposition P
def P (a : ℝ) : Prop :=
  ∃ x0 : ℝ, x0 > 0 ∧ 2^x0 * (x0 - a) > 1

-- The negation of the proposition P
def neg_P (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → 2^x * (x - a) ≤ 1

-- Theorem stating that the negation of P is neg_P
theorem negation_of_exists (a : ℝ) : ¬ P a ↔ neg_P a :=
by
  -- (Proof to be provided)
  sorry

end negation_of_exists_l217_217034


namespace ticket_difference_l217_217217

theorem ticket_difference (V G : ℕ) (h1 : V + G = 320) (h2 : 45 * V + 20 * G = 7500) :
  G - V = 232 :=
by
  sorry

end ticket_difference_l217_217217


namespace repayment_days_least_integer_l217_217354

theorem repayment_days_least_integer:
  ∀ (x : ℤ), (20 + 2 * x ≥ 60) → (x ≥ 20) :=
by
  intro x
  intro h
  sorry

end repayment_days_least_integer_l217_217354


namespace area_of_sector_l217_217271

theorem area_of_sector (s θ : ℝ) (r : ℝ) (h_s : s = 4) (h_θ : θ = 2) (h_r : r = s / θ) :
  (1 / 2) * r^2 * θ = 4 :=
by
  sorry

end area_of_sector_l217_217271


namespace find_2a_plus_b_l217_217033

open Function

noncomputable def f (a b : ℝ) (x : ℝ) := 2 * a * x - 3 * b
noncomputable def g (x : ℝ) := 5 * x + 4
noncomputable def h (a b : ℝ) (x : ℝ) := g (f a b x)
noncomputable def h_inv (x : ℝ) := 2 * x - 9

theorem find_2a_plus_b (a b : ℝ) (h_comp_inv_eq_id : ∀ x, h a b (h_inv x) = x) :
  2 * a + b = 1 / 15 := 
sorry

end find_2a_plus_b_l217_217033


namespace gcd_lcm_product_l217_217265

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 90) (h₂ : b = 135) : 
  Nat.gcd a b * Nat.lcm a b = 12150 :=
by
  -- Using given assumptions
  rw [h₁, h₂]
  -- Lean's definition of gcd and lcm in Nat
  sorry

end gcd_lcm_product_l217_217265


namespace calc_f_18_48_l217_217437

def f (x y : ℕ) : ℕ := sorry

axiom f_self (x : ℕ) : f x x = x
axiom f_symm (x y : ℕ) : f x y = f y x
axiom f_third_cond (x y : ℕ) : (x + y) * f x y = x * f x (x + y)

theorem calc_f_18_48 : f 18 48 = 48 := sorry

end calc_f_18_48_l217_217437


namespace find_integer_n_l217_217195

theorem find_integer_n : ∃ n : ℤ, 0 ≤ n ∧ n < 151 ∧ (150 * n) % 151 = 93 :=
by
  sorry

end find_integer_n_l217_217195


namespace heavy_cream_cost_l217_217073

theorem heavy_cream_cost
  (cost_strawberries : ℕ)
  (cost_raspberries : ℕ)
  (total_cost : ℕ)
  (cost_heavy_cream : ℕ) :
  (cost_strawberries = 3 * 2) →
  (cost_raspberries = 5 * 2) →
  (total_cost = 20) →
  (cost_heavy_cream = total_cost - (cost_strawberries + cost_raspberries)) →
  cost_heavy_cream = 4 :=
by
  sorry

end heavy_cream_cost_l217_217073


namespace geometric_sequence_arithmetic_l217_217572

theorem geometric_sequence_arithmetic (a₁ q : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) 
  (h2 : 2 * S 6 = S 3 + S 9) : 
  q^3 = -1 := 
sorry

end geometric_sequence_arithmetic_l217_217572


namespace ratio_male_whales_l217_217456

def num_whales_first_trip_males : ℕ := 28
def num_whales_first_trip_females : ℕ := 56
def num_whales_second_trip_babies : ℕ := 8
def num_whales_second_trip_parents_males : ℕ := 8
def num_whales_second_trip_parents_females : ℕ := 8
def num_whales_third_trip_females : ℕ := 56
def total_whales : ℕ := 178

theorem ratio_male_whales (M : ℕ) (ratio : ℕ × ℕ) 
  (h_total_whales : num_whales_first_trip_males + num_whales_first_trip_females 
    + num_whales_second_trip_babies + num_whales_second_trip_parents_males 
    + num_whales_second_trip_parents_females + M + num_whales_third_trip_females = total_whales) 
  (h_ratio : ratio = ((M : ℕ) / Nat.gcd M num_whales_first_trip_males, 
                       num_whales_first_trip_males / Nat.gcd M num_whales_first_trip_males)) 
  : ratio = (1, 2) :=
by
  sorry

end ratio_male_whales_l217_217456


namespace platform_length_l217_217037

theorem platform_length (train_length : ℝ) (time_cross_pole : ℝ) (time_cross_platform : ℝ) (speed : ℝ) 
  (h1 : train_length = 300) 
  (h2 : time_cross_pole = 18) 
  (h3 : time_cross_platform = 54)
  (h4 : speed = train_length / time_cross_pole) :
  train_length + (speed * time_cross_platform) - train_length = 600 := 
by
  sorry

end platform_length_l217_217037


namespace initial_apples_l217_217175

-- Defining the conditions
def apples_handed_out := 8
def pies_made := 6
def apples_per_pie := 9
def apples_for_pies := pies_made * apples_per_pie

-- Prove the initial number of apples
theorem initial_apples : apples_handed_out + apples_for_pies = 62 :=
by
  sorry

end initial_apples_l217_217175


namespace dot_product_of_a_and_b_is_correct_l217_217357

-- Define vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -1)

-- Define dot product for ℝ × ℝ vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Theorem statement (proof can be omitted with sorry)
theorem dot_product_of_a_and_b_is_correct : dot_product a b = -4 :=
by
  -- proof goes here, omitted for now
  sorry

end dot_product_of_a_and_b_is_correct_l217_217357


namespace simplify_fraction_l217_217163

variable (x y : ℝ)

theorem simplify_fraction (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 * x^2 / y) * (y^2 / (2 * x)) = 3 * x * y / 2 :=
by sorry

end simplify_fraction_l217_217163


namespace smallest_n_div_75_eq_432_l217_217988

theorem smallest_n_div_75_eq_432 :
  ∃ n k : ℕ, (n ∣ 75 ∧ (∃ (d : ℕ), d ∣ n → d ≠ 1 → d ≠ n → n = 75 * k ∧ ∀ x: ℕ, (x ∣ n) → (x ≠ 1 ∧ x ≠ n) → False)) → ( k =  432 ) :=
by
  sorry

end smallest_n_div_75_eq_432_l217_217988


namespace arithmetic_geometric_means_l217_217551

theorem arithmetic_geometric_means (a b : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 110) : 
  a^2 + b^2 = 1380 :=
sorry

end arithmetic_geometric_means_l217_217551


namespace calculation_proof_l217_217103

theorem calculation_proof : 
  2 * Real.tan (Real.pi / 3) - (-2023) ^ 0 + (1 / 2) ^ (-1 : ℤ) + abs (Real.sqrt 3 - 1) = 3 * Real.sqrt 3 := 
by
  sorry

end calculation_proof_l217_217103


namespace fermats_little_theorem_l217_217787

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Prime p) (hgcd : gcd a p = 1) : (a^(p-1) - 1) % p = 0 := by
  sorry

end fermats_little_theorem_l217_217787


namespace sample_size_l217_217837

variable (total_employees : ℕ) (young_employees : ℕ) (middle_aged_employees : ℕ) (elderly_employees : ℕ) (young_in_sample : ℕ)

theorem sample_size (h1 : total_employees = 750) (h2 : young_employees = 350) (h3 : middle_aged_employees = 250) (h4 : elderly_employees = 150) (h5 : young_in_sample = 7) :
  ∃ sample_size, young_in_sample * total_employees / young_employees = sample_size ∧ sample_size = 15 :=
by
  sorry

end sample_size_l217_217837


namespace Kolya_walking_speed_l217_217282

theorem Kolya_walking_speed
  (x : ℝ) 
  (h1 : x > 0) 
  (t_closing : ℝ := (3 * x) / 10) 
  (t_travel : ℝ := ((x / 10) + (x / 20))) 
  (remaining_time : ℝ := t_closing - t_travel)
  (walking_speed : ℝ := x / remaining_time)
  (correct_speed : ℝ := 20 / 3) :
  walking_speed = correct_speed := 
by 
  sorry

end Kolya_walking_speed_l217_217282


namespace right_triangle_cosine_l217_217989

theorem right_triangle_cosine (XY XZ YZ : ℝ) (hXY_pos : XY > 0) (hXZ_pos : XZ > 0) (hYZ_pos : YZ > 0)
  (angle_XYZ : angle_1 = 90) (tan_Z : XY / XZ = 5 / 12) : (XZ / YZ = 12 / 13) :=
by
  sorry

end right_triangle_cosine_l217_217989


namespace velocity_of_current_correct_l217_217945

-- Definitions based on the conditions in the problem
def rowing_speed_in_still_water : ℝ := 10
def distance_to_place : ℝ := 24
def total_time_round_trip : ℝ := 5

-- Define the velocity of the current
def velocity_of_current : ℝ := 2

-- Main theorem statement
theorem velocity_of_current_correct :
  ∃ (v : ℝ), (v = 2) ∧ 
  (total_time_round_trip = (distance_to_place / (rowing_speed_in_still_water + v) + 
                            distance_to_place / (rowing_speed_in_still_water - v))) :=
by {
  sorry
}

end velocity_of_current_correct_l217_217945


namespace x_range_l217_217934

theorem x_range (x : ℝ) : (x + 2) > 0 → (3 - x) ≥ 0 → (-2 < x ∧ x ≤ 3) :=
by
  intro h1 h2
  constructor
  { linarith }
  { linarith }

end x_range_l217_217934


namespace treasure_chest_l217_217367

theorem treasure_chest (n : ℕ) 
  (h1 : n % 8 = 2)
  (h2 : n % 7 = 6)
  (h3 : ∀ m : ℕ, (m % 8 = 2 → m % 7 = 6 → m ≥ n)) :
  n % 9 = 7 :=
sorry

end treasure_chest_l217_217367


namespace probability_all_white_is_correct_l217_217061

-- Define the total number of balls
def total_balls : ℕ := 25

-- Define the number of white balls
def white_balls : ℕ := 10

-- Define the number of black balls
def black_balls : ℕ := 15

-- Define the number of balls drawn
def balls_drawn : ℕ := 4

-- Define combination function
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to choose 4 balls from 25
def total_ways : ℕ := C total_balls balls_drawn

-- Ways to choose 4 white balls from 10 white balls
def white_ways : ℕ := C white_balls balls_drawn

-- Probability that all 4 drawn balls are white
def prob_all_white : ℚ := white_ways / total_ways

theorem probability_all_white_is_correct :
  prob_all_white = (3 : ℚ) / 181 := by
  -- Proof statements go here
  sorry

end probability_all_white_is_correct_l217_217061


namespace number_without_daughters_l217_217651

-- Given conditions
def Marilyn_daughters : Nat := 10
def total_women : Nat := 40
def daughters_with_daughters_women_have_each : Nat := 5

-- Helper definition representing the computation of granddaughters
def Marilyn_granddaughters : Nat := total_women - Marilyn_daughters

-- Proving the main statement
theorem number_without_daughters : 
  (Marilyn_daughters - (Marilyn_granddaughters / daughters_with_daughters_women_have_each)) + Marilyn_granddaughters = 34 := by
  sorry

end number_without_daughters_l217_217651


namespace sqrt_expression_l217_217915

theorem sqrt_expression :
  Real.sqrt 18 - 3 * Real.sqrt (1 / 2) + Real.sqrt 2 = (5 * Real.sqrt 2) / 2 :=
by
  sorry

end sqrt_expression_l217_217915


namespace second_time_apart_l217_217824

theorem second_time_apart 
  (glen_speed : ℕ) 
  (hannah_speed : ℕ)
  (initial_distance : ℕ) 
  (initial_time : ℕ)
  (relative_speed : ℕ)
  (hours_later : ℕ) :
  glen_speed = 37 →
  hannah_speed = 15 →
  initial_distance = 130 →
  initial_time = 6 →
  relative_speed = glen_speed + hannah_speed →
  hours_later = initial_distance / relative_speed →
  initial_time + hours_later = 8 + 30 / 60 :=
by
  intros
  sorry

end second_time_apart_l217_217824


namespace difference_in_amount_paid_l217_217203

variable (P Q : ℝ)

def original_price := P
def intended_quantity := Q

def new_price := P * 1.10
def new_quantity := Q * 0.80

theorem difference_in_amount_paid :
  ((new_price P * new_quantity Q) - (original_price P * intended_quantity Q)) = -0.12 * (original_price P * intended_quantity Q) :=
by
  sorry

end difference_in_amount_paid_l217_217203


namespace min_box_coeff_l217_217476

theorem min_box_coeff (a b c d : ℤ) (h_ac : a * c = 40) (h_bd : b * d = 40) : 
  ∃ (min_val : ℤ), min_val = 89 ∧ (a * d + b * c) ≥ min_val :=
sorry

end min_box_coeff_l217_217476


namespace perpendicular_lines_sin_2alpha_l217_217635

theorem perpendicular_lines_sin_2alpha (α : ℝ) 
  (l1 : ∀ (x y : ℝ), x * (Real.sin α) + y - 1 = 0) 
  (l2 : ∀ (x y : ℝ), x - 3 * y * Real.cos α + 1 = 0) 
  (perp : ∀ (x1 y1 x2 y2 : ℝ), 
        (x1 * (Real.sin α) + y1 - 1 = 0) ∧ 
        (x2 - 3 * y2 * Real.cos α + 1 = 0) → 
        ((-Real.sin α) * (1 / (3 * Real.cos α)) = -1)) :
  Real.sin (2 * α) = (3/5) :=
sorry

end perpendicular_lines_sin_2alpha_l217_217635


namespace solution_set_of_inequality_l217_217685

theorem solution_set_of_inequality : {x : ℝ | x^2 - 2 * x ≤ 0} = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_of_inequality_l217_217685


namespace intersection_of_sets_l217_217965

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {0, 1, 2}

theorem intersection_of_sets :
  C = A ∩ B :=
sorry

end intersection_of_sets_l217_217965


namespace operation_evaluation_l217_217830

theorem operation_evaluation : 65 + 5 * 12 / (180 / 3) = 66 :=
by
  -- Parentheses
  have h1 : 180 / 3 = 60 := by sorry
  -- Multiplication and Division
  have h2 : 5 * 12 = 60 := by sorry
  have h3 : 60 / 60 = 1 := by sorry
  -- Addition
  exact sorry

end operation_evaluation_l217_217830


namespace solve_investment_problem_l217_217592

def remaining_rate_proof (A I A1 R1 A2 R2 x : ℚ) : Prop :=
  let income1 := A1 * (R1 / 100)
  let income2 := A2 * (R2 / 100)
  let remaining := A - A1 - A2
  let required_income := I - (income1 + income2)
  let expected_rate_in_float := (required_income / remaining) * 100
  expected_rate_in_float = x

theorem solve_investment_problem :
  remaining_rate_proof 15000 800 5000 3 6000 4.5 9.5 :=
by
  -- proof goes here
  sorry

end solve_investment_problem_l217_217592


namespace graduate_degree_ratio_l217_217273

theorem graduate_degree_ratio (G C N : ℕ) (h1 : C = (2 / 3 : ℚ) * N)
  (h2 : (G : ℚ) / (G + C) = 0.15789473684210525) :
  (G : ℚ) / N = 1 / 8 :=
  sorry

end graduate_degree_ratio_l217_217273


namespace work_days_by_a_l217_217055

-- Given
def work_days_by_b : ℕ := 10  -- B can do the work alone in 10 days
def combined_work_days : ℕ := 5  -- A and B together can do the work in 5 days

-- Question: In how many days can A do the work alone?
def days_for_a_work_alone : ℕ := 10  -- The correct answer from the solution

-- Proof statement
theorem work_days_by_a (x : ℕ) : 
  ((1 : ℝ) / (x : ℝ) + (1 : ℝ) / (work_days_by_b : ℝ) = (1 : ℝ) / (combined_work_days : ℝ)) → 
  x = days_for_a_work_alone :=
by 
  sorry

end work_days_by_a_l217_217055


namespace find_a7_l217_217448

variable {a : ℕ → ℝ}

-- Conditions
def is_increasing_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 1 < q ∧ ∀ n : ℕ, a (n + 1) = a n * q

axiom a3_eq_4 : a 3 = 4
axiom harmonic_condition : (1 / a 1 + 1 / a 5 = 5 / 8)
axiom increasing_geometric : is_increasing_geometric_sequence a

-- The problem is to prove that a 7 = 16 given the above conditions.
theorem find_a7 : a 7 = 16 :=
by
  -- Proof goes here
  sorry

end find_a7_l217_217448


namespace arc_length_of_sector_l217_217846

theorem arc_length_of_sector (r θ : ℝ) (A : ℝ) (h₁ : r = 4)
  (h₂ : A = 7) : (1 / 2) * r^2 * θ = A → r * θ = 3.5 :=
by
  sorry

end arc_length_of_sector_l217_217846


namespace no_integer_roots_l217_217298
open Polynomial

theorem no_integer_roots {p : ℤ[X]} (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_pa : p.eval a = 1) (h_pb : p.eval b = 1) (h_pc : p.eval c = 1) : 
  ∀ m : ℤ, p.eval m ≠ 0 :=
by
  sorry

end no_integer_roots_l217_217298


namespace find_original_number_l217_217646

theorem find_original_number (n : ℝ) (h : n / 2 = 9) : n = 18 :=
sorry

end find_original_number_l217_217646


namespace find_x_axis_intercept_l217_217329

theorem find_x_axis_intercept : ∃ x, 5 * 0 - 6 * x = 15 ∧ x = -2.5 := by
  -- The theorem states that there exists an x-intercept such that substituting y = 0 in the equation results in x = -2.5.
  sorry

end find_x_axis_intercept_l217_217329


namespace number_of_rectangles_l217_217966

theorem number_of_rectangles (a b : ℝ) (ha_lt_b : a < b) :
  ∃! (x y : ℝ), (x < b ∧ y < b ∧ 2 * (x + y) = a + b ∧ x * y = (a * b) / 4) := 
sorry

end number_of_rectangles_l217_217966


namespace find_y_l217_217977

variable (x y z : ℚ)

theorem find_y
  (h1 : x + y + z = 150)
  (h2 : x + 7 = y - 12)
  (h3 : x + 7 = 4 * z) :
  y = 688 / 9 :=
sorry

end find_y_l217_217977


namespace guaranteed_winning_strategy_l217_217884

variable (a b : ℝ)

theorem guaranteed_winning_strategy (h : a ≠ b) : (a^3 + b^3) > (a^2 * b + a * b^2) :=
by 
  sorry

end guaranteed_winning_strategy_l217_217884


namespace chelsea_sugar_bags_l217_217845

variable (n : ℕ)

-- Defining the conditions as hypotheses
def initial_sugar : ℕ := 24
def remaining_sugar : ℕ := 21
def sugar_lost : ℕ := initial_sugar - remaining_sugar
def torn_bag_sugar : ℕ := 2 * sugar_lost

-- Define the statement to prove
theorem chelsea_sugar_bags :
  n = initial_sugar / torn_bag_sugar → n = 4 :=
by
  sorry

end chelsea_sugar_bags_l217_217845


namespace nina_money_l217_217309

theorem nina_money (W : ℝ) (P: ℝ) (Q : ℝ) 
  (h1 : P = 6 * W)
  (h2 : Q = 8 * (W - 1))
  (h3 : P = Q) 
  : P = 24 := 
by 
  sorry

end nina_money_l217_217309


namespace not_directly_or_inversely_proportional_l217_217849

theorem not_directly_or_inversely_proportional
  (P : ∀ x y : ℝ, x + y = 0 → (∃ k : ℝ, x = k * y))
  (Q : ∀ x y : ℝ, 3 * x * y = 10 → ∃ k : ℝ, x * y = k)
  (R : ∀ x y : ℝ, x = 5 * y → (∃ k : ℝ, x = k * y))
  (S : ∀ x y : ℝ, 3 * x + y = 10 → ¬ (∃ k : ℝ, x * y = k) ∧ ¬ (∃ k : ℝ, x = k * y))
  (T : ∀ x y : ℝ, x / y = Real.sqrt 3 → (∃ k : ℝ, x = k * y)) :
  ∀ x y : ℝ, 3 * x + y = 10 → ¬ (∃ k : ℝ, x * y = k) ∧ ¬ (∃ k : ℝ, x = k * y) := by
  sorry

end not_directly_or_inversely_proportional_l217_217849


namespace rabbits_total_distance_l217_217300

theorem rabbits_total_distance :
  let white_speed := 15
  let brown_speed := 12
  let grey_speed := 18
  let black_speed := 10
  let time := 7
  let white_distance := white_speed * time
  let brown_distance := brown_speed * time
  let grey_distance := grey_speed * time
  let black_distance := black_speed * time
  let total_distance := white_distance + brown_distance + grey_distance + black_distance
  total_distance = 385 :=
by
  sorry

end rabbits_total_distance_l217_217300


namespace gcd_324_243_l217_217223

-- Define the numbers involved in the problem.
def a : ℕ := 324
def b : ℕ := 243

-- State the theorem that the GCD of a and b is 81.
theorem gcd_324_243 : Nat.gcd a b = 81 := by
  sorry

end gcd_324_243_l217_217223


namespace range_of_m_l217_217098

noncomputable def f (x m : ℝ) : ℝ := (1 / 4) * x^4 - (2 / 3) * x^3 + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x m + (1 / 3) ≥ 0) ↔ m ≥ 1 := 
sorry

end range_of_m_l217_217098


namespace range_of_m_l217_217012

variable {x y m : ℝ}

theorem range_of_m (hx : 0 < x) (hy : 0 < y) (h : 4 / x + 1 / y = 1) :
  x + y ≥ m^2 + m + 3 ↔ -3 ≤ m ∧ m ≤ 2 := sorry

end range_of_m_l217_217012


namespace road_construction_equation_l217_217406

theorem road_construction_equation (x : ℝ) (hx : x > 0) :
  (9 / x) - (12 / (x + 1)) = 1 / 2 :=
sorry

end road_construction_equation_l217_217406


namespace find_b_perpendicular_lines_l217_217921

theorem find_b_perpendicular_lines (b : ℚ)
  (line1 : (3 : ℚ) * x + 4 * y - 6 = 0)
  (line2 : b * x + 4 * y - 6 = 0)
  (perpendicular : ( - (3 : ℚ) / 4 ) * ( - (b / 4) ) = -1) :
  b = - (16 : ℚ) / 3 := 
sorry

end find_b_perpendicular_lines_l217_217921


namespace log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6_l217_217244

theorem log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6 
  (h1 : 5^0.6 > 1)
  (h2 : 0 < 0.6^5 ∧ 0.6^5 < 1)
  (h3 : Real.logb 0.6 5 < 0) :
  Real.logb 0.6 5 < 0.6^5 ∧ 0.6^5 < 5^0.6 :=
sorry

end log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6_l217_217244


namespace functions_not_necessarily_equal_l217_217115

-- Define the domain and range
variables {α β : Type*}

-- Define two functions f and g with the same domain and range
variables (f g : α → β)

-- Lean statement for the given mathematical problem
theorem functions_not_necessarily_equal (h_domain : ∀ x : α, (∃ x : α, true))
  (h_range : ∀ y : β, (∃ y : β, true)) : ¬(f = g) :=
sorry

end functions_not_necessarily_equal_l217_217115


namespace train_speed_is_117_l217_217718

noncomputable def train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let man_speed_mps := man_speed_kmph * 1000 / 3600
  let relative_speed := train_length / crossing_time
  (relative_speed - man_speed_mps) * 3.6

theorem train_speed_is_117 :
  train_speed 300 9 3 = 117 :=
by
  -- We leave the proof as sorry since only the statement is needed
  sorry

end train_speed_is_117_l217_217718


namespace ratio_of_a_b_l217_217546

variable (x y a b : ℝ)

theorem ratio_of_a_b (h₁ : 4 * x - 2 * y = a)
                     (h₂ : 6 * y - 12 * x = b)
                     (hb : b ≠ 0)
                     (ha_solution : ∃ x y, 4 * x - 2 * y = a ∧ 6 * y - 12 * x = b) :
                     a / b = 1 / 3 :=
by sorry

end ratio_of_a_b_l217_217546


namespace birds_percentage_hawks_l217_217885

-- Define the conditions and the main proof problem
theorem birds_percentage_hawks (H : ℝ) :
  (0.4 * (1 - H) + 0.25 * 0.4 * (1 - H) + H = 0.65) → (H = 0.3) :=
by
  intro h
  sorry

end birds_percentage_hawks_l217_217885


namespace triangle_condition_proof_l217_217556

variables {A B C D M K : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M] [MetricSpace K]
variables (AB AC AD : ℝ)

-- Definitions based on the conditions
def is_isosceles (A B C : Type*) (AB AC : ℝ) : Prop :=
  AB = AC

def is_altitude (A D B C : Type*) : Prop :=
  true -- Ideally, this condition is more complex and involves perpendicular projection

def is_midpoint (M A D : Type*) : Prop :=
  true -- Ideally, this condition is more specific and involves equality of segments

def extends_to (C M A B K : Type*) : Prop :=
  true -- Represents the extension relationship

-- The theorem to be proved
theorem triangle_condition_proof (A B C D M K : Type*)
  (h_iso : is_isosceles A B C AB AC)
  (h_alt : is_altitude A D B C)
  (h_mid : is_midpoint M A D)
  (h_ext : extends_to C M A B K)
  : AB = 3 * AK :=
  sorry

end triangle_condition_proof_l217_217556


namespace num_kids_eq_3_l217_217133

def mom_eyes : ℕ := 1
def dad_eyes : ℕ := 3
def kid_eyes : ℕ := 4
def total_eyes : ℕ := 16

theorem num_kids_eq_3 : ∃ k : ℕ, 1 + 3 + 4 * k = 16 ∧ k = 3 := by
  sorry

end num_kids_eq_3_l217_217133


namespace ratio_of_a_to_b_l217_217431

variable (a b x m : ℝ)
variable (h_a_pos : a > 0) (h_b_pos : b > 0)
variable (h_x : x = 1.25 * a) (h_m : m = 0.6 * b)
variable (h_ratio : m / x = 0.6)

theorem ratio_of_a_to_b (h_x : x = 1.25 * a) (h_m : m = 0.6 * b) (h_ratio : m / x = 0.6) : a / b = 0.8 :=
by
  sorry

end ratio_of_a_to_b_l217_217431


namespace additional_days_needed_is_15_l217_217443

-- Definitions and conditions from the problem statement
def good_days_2013 : ℕ := 365 * 479 / 100  -- Number of good air quality days in 2013
def target_increase : ℕ := 20              -- Target increase in percentage for 2014
def additional_days_first_half_2014 : ℕ := 20 -- Additional good air quality days in first half of 2014 compared to 2013
def half_good_days_2013 : ℕ := good_days_2013 / 2 -- Good air quality days in first half of 2013

-- Target number of good air quality days for 2014
def target_days_2014 : ℕ := good_days_2013 * (100 + target_increase) / 100

-- Good air quality days in the first half of 2014
def good_days_first_half_2014 : ℕ := half_good_days_2013 + additional_days_first_half_2014

-- Additional good air quality days needed in the second half of 2014
def additional_days_2014_second_half (target_days good_days_first_half_2014 : ℕ) : ℕ := 
  target_days - good_days_first_half_2014 - half_good_days_2013

-- Final theorem verifying the number of additional days needed in the second half of 2014 is 15
theorem additional_days_needed_is_15 : 
  additional_days_2014_second_half target_days_2014 good_days_first_half_2014 = 15 :=
sorry

end additional_days_needed_is_15_l217_217443


namespace angle_A_is_70_l217_217496

-- Definitions of angles given as conditions in the problem
variables (BAD BAC ACB : ℝ)

def angle_BAD := 150
def angle_BAC := 80

-- The Lean 4 statement to prove the measure of angle ACB
theorem angle_A_is_70 (h1 : BAD = 150) (h2 : BAC = 80) : ACB = 70 :=
by {
  sorry
}

end angle_A_is_70_l217_217496


namespace carol_age_l217_217374

theorem carol_age (B C : ℕ) (h1 : B + C = 66) (h2 : C = 3 * B + 2) : C = 50 :=
sorry

end carol_age_l217_217374


namespace Q_2_plus_Q_neg_2_l217_217522

noncomputable def cubic_polynomial (a b c k : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + k

theorem Q_2_plus_Q_neg_2 (a b c k : ℝ) 
  (h0 : cubic_polynomial a b c k 0 = k)
  (h1 : cubic_polynomial a b c k 1 = 3 * k)
  (hneg1 : cubic_polynomial a b c k (-1) = 4 * k) :
  cubic_polynomial a b c k 2 + cubic_polynomial a b c k (-2) = 22 * k :=
sorry

end Q_2_plus_Q_neg_2_l217_217522


namespace mk_div_km_l217_217821

theorem mk_div_km 
  (m n k : ℕ) 
  (hm : 0 < m) 
  (hn : 0 < n) 
  (hk : 0 < k) 
  (h1 : m^n ∣ n^m) 
  (h2 : n^k ∣ k^n) : 
  m^k ∣ k^m := 
  sorry

end mk_div_km_l217_217821


namespace decreased_value_of_expression_l217_217760

theorem decreased_value_of_expression (x y z : ℝ) :
  let x' := 0.6 * x
  let y' := 0.6 * y
  let z' := 0.6 * z
  (x' * y' * z'^2) = 0.1296 * (x * y * z^2) :=
by
  sorry

end decreased_value_of_expression_l217_217760


namespace intersection_points_l217_217530

open Real

def parabola1 (x : ℝ) : ℝ := x^2 - 3 * x + 2
def parabola2 (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

theorem intersection_points : 
  ∃ x y : ℝ, 
  (parabola1 x = y ∧ parabola2 x = y) ∧ 
  ((x = 1/2 ∧ y = 3/4) ∨ (x = -3 ∧ y = 20)) :=
by sorry

end intersection_points_l217_217530


namespace vehicle_value_last_year_l217_217800

theorem vehicle_value_last_year (value_this_year : ℝ) (ratio : ℝ) (value_this_year_cond : value_this_year = 16000) (ratio_cond : ratio = 0.8) :
  ∃ (value_last_year : ℝ), value_this_year = ratio * value_last_year ∧ value_last_year = 20000 :=
by
  use 20000
  sorry

end vehicle_value_last_year_l217_217800


namespace sequence_periodicity_l217_217662

theorem sequence_periodicity (a : ℕ → ℤ) 
  (h1 : a 1 = 3) 
  (h2 : a 2 = 6) 
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n): 
  a 2015 = -6 := 
sorry

end sequence_periodicity_l217_217662


namespace max_c_friendly_value_l217_217187

def is_c_friendly (c : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → |f x - f y| ≤ c * |x - y|

theorem max_c_friendly_value (c : ℝ) (f : ℝ → ℝ) (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  c > 1 → is_c_friendly c f → |f x - f y| ≤ (c + 1) / 2 :=
sorry

end max_c_friendly_value_l217_217187


namespace quadratic_function_fixed_points_range_l217_217550

def has_two_distinct_fixed_points (c : ℝ) : Prop := 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
               (x1 = x1^2 - x1 + c) ∧ 
               (x2 = x2^2 - x2 + c) ∧ 
               x1 < 2 ∧ 2 < x2

theorem quadratic_function_fixed_points_range (c : ℝ) :
  has_two_distinct_fixed_points c ↔ c < 0 :=
sorry

end quadratic_function_fixed_points_range_l217_217550


namespace measure_of_U_is_120_l217_217113

variable {α β γ δ ε ζ : ℝ}
variable (h1 : α = γ) (h2 : α = ζ) (h3 : β + δ = 180) (h4 : ε + ζ = 180)

noncomputable def measure_of_U : ℝ :=
  let total_sum := 720
  have sum_of_angles : α + β + γ + δ + ζ + ε = total_sum := by
    sorry
  have subs_suppl_G_R : β + δ = 180 := h3
  have subs_suppl_E_U : ε + ζ = 180 := h4
  have congruent_F_I_U : α = γ ∧ α = ζ := ⟨h1, h2⟩
  let α : ℝ := sorry
  α

theorem measure_of_U_is_120 : measure_of_U h1 h2 h3 h4 = 120 :=
  sorry

end measure_of_U_is_120_l217_217113


namespace sum_arithmetic_sequence_n_ge_52_l217_217720

theorem sum_arithmetic_sequence_n_ge_52 (n : ℕ) : 
  (∃ k, k = n) → 22 - 3 * (n - 1) = 22 - 3 * (n - 1) ∧ n ∈ { k | 3 ≤ k ∧ k ≤ 13 } :=
by
  sorry

end sum_arithmetic_sequence_n_ge_52_l217_217720


namespace total_cost_l217_217487

theorem total_cost (a b : ℕ) : 30 * a + 20 * b = 30 * a + 20 * b :=
by
  sorry

end total_cost_l217_217487


namespace llesis_more_rice_l217_217317

theorem llesis_more_rice :
  let total_rice := 50
  let llesis_fraction := 7 / 10
  let llesis_rice := total_rice * llesis_fraction
  let everest_rice := total_rice - llesis_rice
  llesis_rice - everest_rice = 20 := by
    sorry

end llesis_more_rice_l217_217317


namespace lisa_flight_distance_l217_217348

-- Define the given speed and time
def speed : ℝ := 32
def time : ℝ := 8

-- Define the distance formula
def distance (v : ℝ) (t : ℝ) : ℝ := v * t

-- State the theorem to be proved
theorem lisa_flight_distance : distance speed time = 256 := by
sorry

end lisa_flight_distance_l217_217348


namespace anita_smallest_number_of_candies_l217_217270

theorem anita_smallest_number_of_candies :
  ∃ x : ℕ, x ≡ 5 [MOD 6] ∧ x ≡ 3 [MOD 8] ∧ x ≡ 7 [MOD 9] ∧ ∀ y : ℕ,
  (y ≡ 5 [MOD 6] ∧ y ≡ 3 [MOD 8] ∧ y ≡ 7 [MOD 9]) → x ≤ y :=
  ⟨203, by sorry⟩

end anita_smallest_number_of_candies_l217_217270


namespace intersection_of_sets_l217_217600

/-- Given the definitions of sets A and B, prove that A ∩ B equals {1, 2}. -/
theorem intersection_of_sets :
  let A := {x : ℝ | 0 < x}
  let B := {-2, -1, 1, 2}
  A ∩ B = {1, 2} :=
sorry

end intersection_of_sets_l217_217600


namespace basil_plants_count_l217_217681

-- Define the number of basil plants and the number of oregano plants
variables (B O : ℕ)

-- Define the conditions
def condition1 : Prop := O = 2 * B + 2
def condition2 : Prop := B + O = 17

-- The proof statement
theorem basil_plants_count (h1 : condition1 B O) (h2 : condition2 B O) : B = 5 := by
  sorry

end basil_plants_count_l217_217681


namespace price_per_half_pound_of_basil_l217_217922

theorem price_per_half_pound_of_basil
    (cost_per_pound_eggplant : ℝ)
    (pounds_eggplant : ℝ)
    (cost_per_pound_zucchini : ℝ)
    (pounds_zucchini : ℝ)
    (cost_per_pound_tomato : ℝ)
    (pounds_tomato : ℝ)
    (cost_per_pound_onion : ℝ)
    (pounds_onion : ℝ)
    (quarts_ratatouille : ℝ)
    (cost_per_quart : ℝ) :
    pounds_eggplant = 5 → cost_per_pound_eggplant = 2 →
    pounds_zucchini = 4 → cost_per_pound_zucchini = 2 →
    pounds_tomato = 4 → cost_per_pound_tomato = 3.5 →
    pounds_onion = 3 → cost_per_pound_onion = 1 →
    quarts_ratatouille = 4 → cost_per_quart = 10 →
    (cost_per_quart * quarts_ratatouille - 
    (cost_per_pound_eggplant * pounds_eggplant + 
    cost_per_pound_zucchini * pounds_zucchini + 
    cost_per_pound_tomato * pounds_tomato + 
    cost_per_pound_onion * pounds_onion)) / 2 = 2.5 :=
by
    intros h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈ h₉ h₀
    rw [h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈, h₉, h₀]
    sorry

end price_per_half_pound_of_basil_l217_217922


namespace map_distance_correct_l217_217050

noncomputable def distance_on_map : ℝ :=
  let speed := 60  -- miles per hour
  let time := 6.5  -- hours
  let scale := 0.01282051282051282 -- inches per mile
  let actual_distance := speed * time -- in miles
  actual_distance * scale -- convert to inches

theorem map_distance_correct :
  distance_on_map = 5 :=
by 
  sorry

end map_distance_correct_l217_217050


namespace paperclip_day_l217_217895

theorem paperclip_day:
  ∃ k : ℕ, 5 * 3 ^ k > 500 ∧ ∀ m : ℕ, m < k → 5 * 3 ^ m ≤ 500 ∧ k % 7 = 5 :=
sorry

end paperclip_day_l217_217895


namespace find_real_parts_l217_217532

theorem find_real_parts (a b : ℝ) (i : ℂ) (hi : i*i = -1) 
(h : a + b*i = (1 - i) * i) : a = 1 ∧ b = -1 :=
sorry

end find_real_parts_l217_217532


namespace number_of_cakes_sold_l217_217399

-- Definitions based on the conditions provided
def cakes_made : ℕ := 173
def cakes_bought : ℕ := 103
def cakes_left : ℕ := 190

-- Calculate the initial total number of cakes
def initial_cakes : ℕ := cakes_made + cakes_bought

-- Calculate the number of cakes sold
def cakes_sold : ℕ := initial_cakes - cakes_left

-- The proof statement
theorem number_of_cakes_sold : cakes_sold = 86 :=
by
  unfold cakes_sold initial_cakes cakes_left cakes_bought cakes_made
  rfl

end number_of_cakes_sold_l217_217399


namespace no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares_l217_217833

theorem no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares :
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ a b c : ℕ, 2 * n^2 + 1 = a^2 ∧ 3 * n^2 + 1 = b^2 ∧ 6 * n^2 + 1 = c^2 := by
  sorry

end no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares_l217_217833


namespace farmer_total_acres_l217_217024

-- Define the problem conditions and the total acres
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) : 5 * x + 2 * x + 4 * x = 1034 :=
by
  -- Proof is omitted
  sorry

end farmer_total_acres_l217_217024


namespace children_playing_both_sports_l217_217108

variable (total_children : ℕ) (T : ℕ) (S : ℕ) (N : ℕ)

theorem children_playing_both_sports 
  (h1 : total_children = 38) 
  (h2 : T = 19) 
  (h3 : S = 21) 
  (h4 : N = 10) : 
  (T + S) - (total_children - N) = 12 := 
by
  sorry

end children_playing_both_sports_l217_217108


namespace relationship_abc_l217_217980

noncomputable def a : ℝ := 4 / 5
noncomputable def b : ℝ := Real.sin (2 / 3)
noncomputable def c : ℝ := Real.cos (1 / 3)

theorem relationship_abc : b < a ∧ a < c := by
  sorry

end relationship_abc_l217_217980


namespace smallest_sector_angle_l217_217338

def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem smallest_sector_angle 
  (a : ℕ) (d : ℕ) (n : ℕ := 15) (sum_angles : ℕ := 360) 
  (angles_arith_seq : arithmetic_sequence_sum a d n = sum_angles) 
  (h_poses : ∀ m : ℕ, arithmetic_sequence_sum a d m = sum_angles -> m = n) 
  : a = 3 := 
by 
  sorry

end smallest_sector_angle_l217_217338


namespace regular_polygon_sides_l217_217240

theorem regular_polygon_sides (θ : ℝ) (hθ : θ = 45) : 360 / θ = 8 := by
  sorry

end regular_polygon_sides_l217_217240


namespace spinner_prob_l217_217773

theorem spinner_prob:
  let sections := 4
  let prob := 1 / sections
  let prob_not_e := 1 - prob
  (prob_not_e * prob_not_e) = 9 / 16 :=
by
  sorry

end spinner_prob_l217_217773


namespace inequality_a_cube_less_b_cube_l217_217264

theorem inequality_a_cube_less_b_cube (a b : ℝ) (ha : a < 0) (hb : b > 0) : a^3 < b^3 :=
by
  sorry

end inequality_a_cube_less_b_cube_l217_217264


namespace hyperbola_asymptote_equation_l217_217512

variable (a b : ℝ)
variable (x y : ℝ)

def arithmetic_mean := (a + b) / 2 = 5
def geometric_mean := (a * b) ^ (1 / 2) = 4
def a_greater_b := a > b
def hyperbola_asymptote := (y = (1 / 2) * x) ∨ (y = -(1 / 2) * x)

theorem hyperbola_asymptote_equation :
  arithmetic_mean a b ∧ geometric_mean a b ∧ a_greater_b a b → hyperbola_asymptote x y :=
by
  sorry

end hyperbola_asymptote_equation_l217_217512


namespace circle_area_l217_217383

-- Condition: Given the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 10 * x + 4 * y + 20 = 0

-- Theorem: The area enclosed by the given circle equation is 9π
theorem circle_area : ∀ x y : ℝ, circle_eq x y → ∃ A : ℝ, A = 9 * Real.pi :=
by
  intros
  sorry

end circle_area_l217_217383


namespace find_unknown_l217_217913

theorem find_unknown (x : ℝ) :
  300 * 2 + (x + 4) * (1 / 8) = 602 → x = 12 :=
by 
  sorry

end find_unknown_l217_217913


namespace find_2x_2y_2z_l217_217096

theorem find_2x_2y_2z (x y z : ℝ) 
  (h1 : y + z = 10 - 2 * x)
  (h2 : x + z = -12 - 4 * y)
  (h3 : x + y = 5 - 2 * z) : 
  2 * x + 2 * y + 2 * z = 3 :=
by
  sorry

end find_2x_2y_2z_l217_217096


namespace boys_in_art_class_l217_217510

noncomputable def number_of_boys (ratio_girls_to_boys : ℕ × ℕ) (total_students : ℕ) : ℕ :=
  let (g, b) := ratio_girls_to_boys
  let k := total_students / (g + b)
  b * k

theorem boys_in_art_class (h : number_of_boys (4, 3) 35 = 15) : true := 
  sorry

end boys_in_art_class_l217_217510


namespace square_area_with_circles_l217_217952

theorem square_area_with_circles
  (radius : ℝ) 
  (side_length : ℝ)
  (area : ℝ)
  (h_radius : radius = 7) 
  (h_side_length : side_length = 2 * (2 * radius)) 
  (h_area : area = side_length ^ 2) : 
  area = 784 := by
  sorry

end square_area_with_circles_l217_217952


namespace find_number_l217_217527

noncomputable def S (x : ℝ) : ℝ :=
  -- Assuming S(x) is a non-trivial function that sums the digits
  sorry

theorem find_number (x : ℝ) (hx_nonzero : x ≠ 0) (h_cond : x = (S x) / 5) : x = 1.8 :=
by
  sorry

end find_number_l217_217527


namespace problem_a_problem_b_problem_c_problem_d_problem_e_l217_217589

section problem_a
  -- Conditions
  def rainbow_russian_first_letters_sequence := ["к", "о", "ж", "з", "г", "с", "ф"]
  
  -- Theorem (question == answer)
  theorem problem_a : rainbow_russian_first_letters_sequence[4] = "г" ∧
                      rainbow_russian_first_letters_sequence[5] = "с" ∧
                      rainbow_russian_first_letters_sequence[6] = "ф" :=
  by
    -- Skip proof: sorry
    sorry
end problem_a

section problem_b
  -- Conditions
  def russian_alphabet_alternating_sequence := ["а", "в", "г", "ё", "ж", "з", "л", "м", "н", "о", "п", "т", "у"]
 
  -- Theorem (question == answer)
  theorem problem_b : russian_alphabet_alternating_sequence[10] = "п" ∧
                      russian_alphabet_alternating_sequence[11] = "т" ∧
                      russian_alphabet_alternating_sequence[12] = "у" :=
  by
    -- Skip proof: sorry
    sorry
end problem_b

section problem_c
  -- Conditions
  def russian_number_of_letters_sequence := ["один", "четыре", "шесть", "пять", "семь", "восемь"]
  
  -- Theorem (question == answer)
  theorem problem_c : russian_number_of_letters_sequence[4] = "семь" ∧
                      russian_number_of_letters_sequence[5] = "восемь" :=
  by
    -- Skip proof: sorry
    sorry
end problem_c

section problem_d
  -- Conditions
  def approximate_symmetry_letters_sequence := ["Ф", "Х", "Ш", "В"]

  -- Theorem (question == answer)
  theorem problem_d : approximate_symmetry_letters_sequence[3] = "В" :=
  by
    -- Skip proof: sorry
    sorry
end problem_d

section problem_e
  -- Conditions
  def russian_loops_in_digit_sequence := ["0", "д", "т", "ч", "п", "ш", "с", "в", "д"]

  -- Theorem (question == answer)
  theorem problem_e : russian_loops_in_digit_sequence[7] = "в" ∧
                      russian_loops_in_digit_sequence[8] = "д" :=
  by
    -- Skip proof: sorry
    sorry
end problem_e

end problem_a_problem_b_problem_c_problem_d_problem_e_l217_217589


namespace exists_natural_multiple_of_2015_with_digit_sum_2015_l217_217053

-- Definition of sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Proposition that we need to prove
theorem exists_natural_multiple_of_2015_with_digit_sum_2015 :
  ∃ n : ℕ, (2015 ∣ n) ∧ sum_of_digits n = 2015 :=
sorry

end exists_natural_multiple_of_2015_with_digit_sum_2015_l217_217053


namespace divides_2pow18_minus_1_l217_217056

theorem divides_2pow18_minus_1 (n : ℕ) : 20 ≤ n ∧ n < 30 ∧ (n ∣ 2^18 - 1) ↔ (n = 19 ∨ n = 27) := by
  sorry

end divides_2pow18_minus_1_l217_217056


namespace remainder_mod_8_l217_217575

theorem remainder_mod_8 (x : ℤ) (h : x % 63 = 25) : x % 8 = 1 := 
sorry

end remainder_mod_8_l217_217575


namespace cos_squared_value_l217_217930

theorem cos_squared_value (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + π / 4) ^ 2 = 1 / 6 :=
sorry

end cos_squared_value_l217_217930


namespace total_apples_l217_217740

def green_apples : ℕ := 2
def red_apples : ℕ := 3
def yellow_apples : ℕ := 14

theorem total_apples : green_apples + red_apples + yellow_apples = 19 :=
by
  -- Placeholder for the proof
  sorry

end total_apples_l217_217740


namespace square_area_l217_217877

theorem square_area (A : ℝ) (s : ℝ) (prob_not_in_B : ℝ)
  (h1 : s * 4 = 32)
  (h2 : prob_not_in_B = 0.20987654320987653)
  (h3 : A - s^2 = prob_not_in_B * A) :
  A = 81 :=
by
  sorry

end square_area_l217_217877


namespace average_large_basket_weight_l217_217310

-- Definitions derived from the conditions
def small_basket_capacity := 25  -- Capacity of each small basket in kilograms
def num_small_baskets := 28      -- Number of small baskets used
def num_large_baskets := 10      -- Number of large baskets used
def leftover_weight := 50        -- Leftover weight in kilograms

-- Statement of the problem
theorem average_large_basket_weight :
  (small_basket_capacity * num_small_baskets - leftover_weight) / num_large_baskets = 65 :=
by
  sorry

end average_large_basket_weight_l217_217310


namespace combined_annual_income_eq_correct_value_l217_217811

theorem combined_annual_income_eq_correct_value :
  let A_income := 5 / 2 * 17000
  let B_income := 1.12 * 17000
  let C_income := 17000
  let D_income := 0.85 * A_income
  (A_income + B_income + C_income + D_income) * 12 = 1375980 :=
by
  sorry

end combined_annual_income_eq_correct_value_l217_217811


namespace probability_same_color_l217_217979

-- Definitions with conditions
def totalPlates := 11
def redPlates := 6
def bluePlates := 5

-- Calculate combinations
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main statement
theorem probability_same_color (totalPlates redPlates bluePlates : ℕ) (h1 : totalPlates = 11) 
(h2 : redPlates = 6) (h3 : bluePlates = 5) : 
  (2 / 11 : ℚ) = ((choose redPlates 3 + choose bluePlates 3) : ℚ) / (choose totalPlates 3) :=
by
  -- Proof steps will be inserted here
  sorry

end probability_same_color_l217_217979


namespace polynomial_factorization_l217_217466

noncomputable def polynomial_expr (a b c : ℝ) :=
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2)

noncomputable def factored_form (a b c : ℝ) :=
  (a - b) * (b - c) * (c - a) * (b^2 + c^2 + a^2)

theorem polynomial_factorization (a b c : ℝ) :
  polynomial_expr a b c = factored_form a b c :=
by {
  sorry
}

end polynomial_factorization_l217_217466


namespace Xiaoming_speed_l217_217631

theorem Xiaoming_speed (x xiaohong_speed_xiaoming_diff : ℝ) :
  (50 * (2 * x + 2) = 600) →
  (xiaohong_speed_xiaoming_diff = 2) →
  x + xiaohong_speed_xiaoming_diff = 7 :=
by
  intros h₁ h₂
  sorry

end Xiaoming_speed_l217_217631


namespace second_car_distance_l217_217852

variables 
  (distance_apart : ℕ := 105)
  (d1 d2 d3 : ℕ := 25) -- distances 25 km, 15 km, 25 km respectively
  (d_road_back : ℕ := 15)
  (final_distance : ℕ := 20)

theorem second_car_distance 
  (car1_total_distance := d1 + d2 + d3 + d_road_back)
  (car2_distance : ℕ) :
  distance_apart - (car1_total_distance + car2_distance) = final_distance →
  car2_distance = 5 :=
sorry

end second_car_distance_l217_217852


namespace viewers_watching_program_A_l217_217526

theorem viewers_watching_program_A (T : ℕ) (hT : T = 560) (x : ℕ)
  (h_ratio : 1 * x + (2 * x - x) + (3 * x - x) = T) : 2 * x = 280 :=
by
  -- by solving the given equation, we find x = 140
  -- substituting x = 140 in 2 * x gives 2 * x = 280
  sorry

end viewers_watching_program_A_l217_217526


namespace maximum_area_l217_217814

-- Define necessary variables and conditions
variables (x y : ℝ)
variable (A : ℝ)
variable (peri : ℝ := 30)

-- Provide the premise that defines the perimeter condition
axiom perimeter_condition : 2 * x + 2 * y = peri

-- Define y in terms of x based on the perimeter condition
def y_in_terms_of_x (x : ℝ) : ℝ := 15 - x

-- Define the area of the rectangle in terms of x
def area (x : ℝ) : ℝ := x * (y_in_terms_of_x x)

-- The statement that needs to be proved
theorem maximum_area : A = 56.25 :=
by sorry

end maximum_area_l217_217814


namespace classroom_students_count_l217_217759

-- Definitions of given conditions
def total_students : ℕ := 1260

def aud_students : ℕ := (7 * total_students) / 18

def non_aud_students : ℕ := total_students - aud_students

def classroom_students : ℕ := (6 * non_aud_students) / 11

-- Theorem statement
theorem classroom_students_count : classroom_students = 420 := by
  sorry

end classroom_students_count_l217_217759


namespace range_of_x_l217_217663

theorem range_of_x :
  (∀ t : ℝ, |t - 3| + |2 * t + 1| ≥ |2 * x - 1| + |x + 2|) →
  (-1/2 ≤ x ∧ x ≤ 5/6) :=
by
  intro h 
  sorry

end range_of_x_l217_217663


namespace projectile_first_reaches_70_feet_l217_217501

theorem projectile_first_reaches_70_feet :
  ∃ t : ℝ, t = 7/4 ∧ 0 < t ∧ ∀ s : ℝ, s < t → -16 * s^2 + 80 * s < 70 :=
by 
  sorry

end projectile_first_reaches_70_feet_l217_217501


namespace minimize_function_l217_217810

noncomputable def f (x : ℝ) : ℝ := x - 4 + 9 / (x + 1)

theorem minimize_function : 
  (∀ x : ℝ, x > -1 → f x ≥ 1) ∧ (f 2 = 1) :=
by 
  sorry

end minimize_function_l217_217810


namespace sum_of_numbers_l217_217438

theorem sum_of_numbers : 217 + 2.017 + 0.217 + 2.0017 = 221.2357 :=
by
  sorry

end sum_of_numbers_l217_217438


namespace three_layer_rug_area_l217_217797

theorem three_layer_rug_area 
  (A B C D : ℕ) 
  (hA : A = 350) 
  (hB : B = 250) 
  (hC : C = 45) 
  (h_formula : A = B + C + D) : 
  D = 55 :=
by
  sorry

end three_layer_rug_area_l217_217797


namespace remainder_n_squared_plus_3n_plus_4_l217_217502

theorem remainder_n_squared_plus_3n_plus_4 (n : ℤ) (h : n % 100 = 99) : (n^2 + 3*n + 4) % 100 = 2 := 
by sorry

end remainder_n_squared_plus_3n_plus_4_l217_217502


namespace value_of_x_l217_217863

variable (x y : ℕ)

-- Conditions
axiom cond1 : x / y = 15 / 3
axiom cond2 : y = 27

-- Lean statement for the problem
theorem value_of_x : x = 135 :=
by
  have h1 := cond1
  have h2 := cond2
  sorry

end value_of_x_l217_217863


namespace fliers_left_l217_217003

theorem fliers_left (total : ℕ) (morning_fraction afternoon_fraction : ℚ) 
  (h1 : total = 1000)
  (h2 : morning_fraction = 1/5)
  (h3 : afternoon_fraction = 1/4) :
  let morning_sent := total * morning_fraction
  let remaining_after_morning := total - morning_sent
  let afternoon_sent := remaining_after_morning * afternoon_fraction
  let remaining_after_afternoon := remaining_after_morning - afternoon_sent
  remaining_after_afternoon = 600 :=
by
  sorry

end fliers_left_l217_217003


namespace selection_ways_l217_217462

-- The statement of the problem in Lean 4
theorem selection_ways :
  (Nat.choose 50 4) - (Nat.choose 47 4) = 
  (Nat.choose 3 1) * (Nat.choose 47 3) + 
  (Nat.choose 3 2) * (Nat.choose 47 2) + 
  (Nat.choose 3 3) * (Nat.choose 47 1) := 
sorry

end selection_ways_l217_217462


namespace johny_total_travel_distance_l217_217028

def TravelDistanceSouth : ℕ := 40
def TravelDistanceEast : ℕ := TravelDistanceSouth + 20
def TravelDistanceNorth : ℕ := 2 * TravelDistanceEast
def TravelDistanceWest : ℕ := TravelDistanceNorth / 2

theorem johny_total_travel_distance
    (hSouth : TravelDistanceSouth = 40)
    (hEast  : TravelDistanceEast = 60)
    (hNorth : TravelDistanceNorth = 120)
    (hWest  : TravelDistanceWest = 60)
    (totalDistance : ℕ := TravelDistanceSouth + TravelDistanceEast + TravelDistanceNorth + TravelDistanceWest) :
    totalDistance = 280 := by
  sorry

end johny_total_travel_distance_l217_217028


namespace find_f_neg2007_l217_217457

variable (f : ℝ → ℝ)

-- Conditions
axiom cond1 (x y w : ℝ) (hx : x > y) (hw : f x + x ≥ w ∧ w ≥ f y + y) : 
  ∃ z ∈ Set.Icc y x, f z = w - z

axiom cond2 : ∃ u, f u = 0 ∧ ∀ v, f v = 0 → u ≤ v

axiom cond3 : f 0 = 1

axiom cond4 : f (-2007) ≤ 2008

axiom cond5 (x y : ℝ) : f x * f y = f (x * f y + y * f x + x * y)

theorem find_f_neg2007 : f (-2007) = 2008 := 
sorry

end find_f_neg2007_l217_217457


namespace percentage_error_in_calculated_area_l217_217495

theorem percentage_error_in_calculated_area :
  let initial_length_error := 0.03 -- 3%
  let initial_width_error := -0.02 -- 2% deficit
  let temperature_change := 15 -- °C
  let humidity_increase := 20 -- %
  let length_error_temp_increase := (temperature_change / 5) * 0.01
  let width_error_humidity_increase := (humidity_increase / 10) * 0.005
  let total_length_error := initial_length_error + length_error_temp_increase
  let total_width_error := initial_width_error + width_error_humidity_increase
  let total_percentage_error := total_length_error + total_width_error
  total_percentage_error * 100 = 3 -- 3%
:= by
  sorry

end percentage_error_in_calculated_area_l217_217495


namespace apple_allocation_proof_l217_217508

theorem apple_allocation_proof : 
    ∃ (ann mary jane kate ned tom bill jack : ℕ), 
    ann = 1 ∧
    mary = 2 ∧
    jane = 3 ∧
    kate = 4 ∧
    ned = jane ∧
    tom = 2 * kate ∧
    bill = 3 * ann ∧
    jack = 4 * mary ∧
    ann + mary + jane + ned + kate + tom + bill + jack = 32 :=
by {
    sorry
}

end apple_allocation_proof_l217_217508


namespace find_m_l217_217843

def hyperbola_focus (x y : ℝ) (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a^2 = 9 ∧ b^2 = -m ∧ (x - 0)^2 / a^2 - (y - 0)^2 / b^2 = 1

theorem find_m (m : ℝ) (H : hyperbola_focus 5 0 m) : m = -16 :=
by
  sorry

end find_m_l217_217843


namespace calculate_years_l217_217993

variable {P R T SI : ℕ}

-- Conditions translations
def simple_interest_one_fifth (P SI : ℕ) : Prop :=
  SI = P / 5

def rate_of_interest (R : ℕ) : Prop :=
  R = 4

-- Proof of the number of years T
theorem calculate_years (h1 : simple_interest_one_fifth P SI)
                        (h2 : rate_of_interest R)
                        (h3 : SI = (P * R * T) / 100) : T = 5 :=
by
  sorry

end calculate_years_l217_217993


namespace existence_of_ab_l217_217593

theorem existence_of_ab (n : ℕ) (hn : 0 < n) : ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ n ∣ (4 * a^2 + 9 * b^2 - 1) :=
by 
  sorry

end existence_of_ab_l217_217593


namespace range_of_independent_variable_l217_217222

theorem range_of_independent_variable (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 :=
sorry

end range_of_independent_variable_l217_217222


namespace probability_of_events_l217_217303

noncomputable def total_types : ℕ := 8

noncomputable def fever_reducing_types : ℕ := 3

noncomputable def cough_suppressing_types : ℕ := 5

noncomputable def total_ways_to_choose_two : ℕ := Nat.choose total_types 2

noncomputable def event_A_ways : ℕ := total_ways_to_choose_two - Nat.choose cough_suppressing_types 2

noncomputable def P_A : ℚ := event_A_ways / total_ways_to_choose_two

noncomputable def event_B_ways : ℕ := fever_reducing_types * cough_suppressing_types

noncomputable def P_B_given_A : ℚ := event_B_ways / event_A_ways

theorem probability_of_events :
  P_A = 9 / 14 ∧ P_B_given_A = 5 / 6 := by
  sorry

end probability_of_events_l217_217303


namespace remaining_money_l217_217404

-- Defining the conditions
def orchids_qty := 20
def price_per_orchid := 50
def chinese_money_plants_qty := 15
def price_per_plant := 25
def worker_qty := 2
def salary_per_worker := 40
def cost_of_pots := 150

-- Earnings and expenses calculations
def earnings_from_orchids := orchids_qty * price_per_orchid
def earnings_from_plants := chinese_money_plants_qty * price_per_plant
def total_earnings := earnings_from_orchids + earnings_from_plants

def worker_expenses := worker_qty * salary_per_worker
def total_expenses := worker_expenses + cost_of_pots

-- The proof problem
theorem remaining_money : total_earnings - total_expenses = 1145 :=
by
  sorry

end remaining_money_l217_217404


namespace length_of_paving_stone_l217_217349

theorem length_of_paving_stone (courtyard_length courtyard_width : ℝ)
  (num_paving_stones : ℕ) (paving_stone_width : ℝ) (total_area : ℝ)
  (paving_stone_length : ℝ) : 
  courtyard_length = 70 ∧ courtyard_width = 16.5 ∧ num_paving_stones = 231 ∧ paving_stone_width = 2 ∧ total_area = courtyard_length * courtyard_width ∧ total_area = num_paving_stones * paving_stone_length * paving_stone_width → 
  paving_stone_length = 2.5 :=
by
  sorry

end length_of_paving_stone_l217_217349


namespace inscribable_quadrilateral_l217_217985

theorem inscribable_quadrilateral
  (a b c d : ℝ)
  (A : ℝ)
  (circumscribable : Prop)
  (area_condition : A = Real.sqrt (a * b * c * d))
  (A := Real.sqrt (a * b * c * d)) : 
  circumscribable → ∃ B D : ℝ, B + D = 180 :=
sorry

end inscribable_quadrilateral_l217_217985


namespace range_of_a_l217_217931

open Real

theorem range_of_a (a : ℝ) :
  (∀ x, |x - 1| < 3 → (x + 2) * (x + a) < 0) ∧ ¬ (∀ x, (x + 2) * (x + a) < 0 → |x - 1| < 3) →
  a < -4 :=
by
  sorry

end range_of_a_l217_217931


namespace laura_has_435_dollars_l217_217125

-- Define the monetary values and relationships
def darwin_money := 45
def mia_money := 2 * darwin_money + 20
def combined_money := mia_money + darwin_money
def laura_money := 3 * combined_money - 30

-- The theorem to prove: Laura's money is $435
theorem laura_has_435_dollars : laura_money = 435 := by
  sorry

end laura_has_435_dollars_l217_217125


namespace symmetric_circle_eq_l217_217591

/-- The definition of the original circle equation. -/
def original_circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

/-- The definition of the line of symmetry equation. -/
def line_eq (x y : ℝ) : Prop := x - y - 2 = 0

/-- The statement that the equation of the circle that is symmetric to the original circle 
    about the given line is (x - 4)^2 + (y + 1)^2 = 1. -/
theorem symmetric_circle_eq : 
  (∃ x y : ℝ, original_circle_eq x y ∧ line_eq x y) →
  (∀ x y : ℝ, (x - 4)^2 + (y + 1)^2 = 1) :=
by sorry

end symmetric_circle_eq_l217_217591


namespace order_of_a_b_c_l217_217749

noncomputable def a : ℝ := Real.log 4 / Real.log 5
noncomputable def b : ℝ := (Real.log 3 / Real.log 5)^2
noncomputable def c : ℝ := Real.log 5 / Real.log 4

theorem order_of_a_b_c : b < a ∧ a < c :=
by
  sorry

end order_of_a_b_c_l217_217749


namespace find_a_of_tangent_area_l217_217775

theorem find_a_of_tangent_area (a : ℝ) (h : a > 0) (h_area : (a^3 / 4) = 2) : a = 2 :=
by
  -- Proof is omitted as it's not required.
  sorry

end find_a_of_tangent_area_l217_217775


namespace maximum_third_height_l217_217807

theorem maximum_third_height 
  (A B C : Type)
  (h1 h2 : ℕ)
  (h1_pos : h1 = 4) 
  (h2_pos : h2 = 12) 
  (h3_pos : ℕ)
  (triangle_inequality : ∀ a b c : ℕ, a + b > c ∧ a + c > b ∧ b + c > a)
  (scalene : ∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c)
  : (3 < h3_pos ∧ h3_pos < 6) → h3_pos = 5 := 
sorry

end maximum_third_height_l217_217807


namespace tom_initial_amount_l217_217039

variables (t s j : ℝ)

theorem tom_initial_amount :
  t + s + j = 1200 →
  t - 200 + 3 * s + 2 * j = 1800 →
  t = 400 :=
by
  intros h1 h2
  sorry

end tom_initial_amount_l217_217039


namespace maximum_achievable_score_l217_217932

def robot_initial_iq : Nat := 25
def problem_scores : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

theorem maximum_achievable_score 
  (initial_iq : Nat := robot_initial_iq) 
  (scores : List Nat := problem_scores) 
  : Nat :=
  31

end maximum_achievable_score_l217_217932


namespace factorize_polynomial_value_of_x_cubed_l217_217018

-- Problem 1: Factorization
theorem factorize_polynomial (x : ℝ) : 42 * x^2 - 33 * x + 6 = 3 * (2 * x - 1) * (7 * x - 2) :=
sorry

-- Problem 2: Given condition and proof of x^3 + 1/x^3
theorem value_of_x_cubed (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^3 + 1 / x^3 = 18 :=
sorry

end factorize_polynomial_value_of_x_cubed_l217_217018


namespace find_n_l217_217815

noncomputable def arithmetic_sequence (a : ℕ → ℕ) := 
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_n (a : ℕ → ℕ) (n d : ℕ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1)
  (h3 : a 2 + a 5 = 12)
  (h4 : a n = 25) : 
  n = 13 := 
sorry

end find_n_l217_217815


namespace simple_interest_correct_l217_217609

-- Define the given conditions
def Principal : ℝ := 9005
def Rate : ℝ := 0.09
def Time : ℝ := 5

-- Define the simple interest function
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- State the theorem to prove the total interest earned
theorem simple_interest_correct : simple_interest Principal Rate Time = 4052.25 := sorry

end simple_interest_correct_l217_217609


namespace algebraic_expression_value_l217_217569

namespace MathProof

variables {α β : ℝ} 

-- Given conditions
def is_root (a : ℝ) : Prop := a^2 - a - 1 = 0
def roots_of_quadratic (α β : ℝ) : Prop := is_root α ∧ is_root β

-- The proof problem statement
theorem algebraic_expression_value (h : roots_of_quadratic α β) : α^2 + α * (β^2 - 2) = 0 := 
by sorry

end MathProof

end algebraic_expression_value_l217_217569


namespace prize_distribution_l217_217362

theorem prize_distribution : 
  ∃ (n1 n2 n3 : ℕ), -- The number of 1st, 2nd, and 3rd prize winners
  n1 + n2 + n3 = 7 ∧ -- Total number of winners is 7
  n1 * 800 + n2 * 700 + n3 * 300 = 4200 ∧ -- Total prize money distributed is $4200
  n1 = 1 ∧ -- Number of 1st prize winners
  n2 = 4 ∧ -- Number of 2nd prize winners
  n3 = 2 -- Number of 3rd prize winners
:= sorry

end prize_distribution_l217_217362


namespace sum_of_first_9_terms_arithmetic_sequence_l217_217794

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem sum_of_first_9_terms_arithmetic_sequence
  (h_arith_seq : is_arithmetic_sequence a)
  (h_condition : a 2 + a 8 = 8) :
  (Finset.range 9).sum a = 36 :=
sorry

end sum_of_first_9_terms_arithmetic_sequence_l217_217794


namespace noemi_start_amount_l217_217829

/-
  Conditions:
    lost_roulette = -600
    won_blackjack = 400
    lost_poker = -400
    won_baccarat = 500
    meal_cost = 200
    purse_end = 1800

  Prove: start_amount == 2300
-/

noncomputable def lost_roulette : Int := -600
noncomputable def won_blackjack : Int := 400
noncomputable def lost_poker : Int := -400
noncomputable def won_baccarat : Int := 500
noncomputable def meal_cost : Int := 200
noncomputable def purse_end : Int := 1800

noncomputable def net_gain : Int := lost_roulette + won_blackjack + lost_poker + won_baccarat

noncomputable def start_amount : Int := net_gain + meal_cost + purse_end

theorem noemi_start_amount : start_amount = 2300 :=
by
  sorry

end noemi_start_amount_l217_217829


namespace bananas_count_l217_217447

theorem bananas_count 
  (total_oranges : ℕ)
  (total_percentage_good : ℝ)
  (percentage_rotten_oranges : ℝ)
  (percentage_rotten_bananas : ℝ)
  (total_good_fruits_percentage : ℝ)
  (B : ℝ) :
  total_oranges = 600 →
  total_percentage_good = 0.85 →
  percentage_rotten_oranges = 0.15 →
  percentage_rotten_bananas = 0.03 →
  total_good_fruits_percentage = 0.898 →
  B = 400  :=
by
  intros h_oranges h_good_percentage h_rotten_oranges h_rotten_bananas h_good_fruits_percentage
  sorry

end bananas_count_l217_217447


namespace digging_project_depth_l217_217715

theorem digging_project_depth : 
  ∀ (P : ℕ) (D : ℝ), 
  (12 * P) * (25 * 30 * D) / 12 = (12 * P) * (75 * 20 * 50) / 12 → 
  D = 100 :=
by
  intros P D h
  sorry

end digging_project_depth_l217_217715


namespace part_a_part_b_l217_217767

variable {p q n : ℕ}

-- Conditions
def coprime (a b : ℕ) : Prop := gcd a b = 1
def differ_by_more_than_one (p q : ℕ) : Prop := (q > p + 1) ∨ (p > q + 1)

-- Part (a): Prove there exists a natural number n such that p + n and q + n are not coprime
theorem part_a (coprime_pq : coprime p q) (diff : differ_by_more_than_one p q) : 
  ∃ n : ℕ, ¬ coprime (p + n) (q + n) :=
sorry

-- Part (b): Prove the smallest such n is 41 for p = 2 and q = 2023
theorem part_b (h : p = 2) (h1 : q = 2023) : 
  ∃ n : ℕ, (n = 41) ∧ (¬ coprime (2 + n) (2023 + n)) :=
sorry

end part_a_part_b_l217_217767


namespace longer_diagonal_length_l217_217719

-- Conditions
def rhombus_side_length := 65
def shorter_diagonal_length := 72

-- Prove that the length of the longer diagonal is 108
theorem longer_diagonal_length : 
  (2 * (Real.sqrt ((rhombus_side_length: ℝ)^2 - (shorter_diagonal_length / 2)^2))) = 108 := 
by 
  sorry

end longer_diagonal_length_l217_217719


namespace sasha_quarters_l217_217687

theorem sasha_quarters (h₁ : 2.10 = 0.35 * q) : q = 6 := 
sorry

end sasha_quarters_l217_217687


namespace dune_buggy_speed_l217_217327

theorem dune_buggy_speed (S : ℝ) :
  (1/3 * S + 1/3 * (S + 12) + 1/3 * (S - 18) = 58) → S = 60 :=
by
  sorry

end dune_buggy_speed_l217_217327


namespace two_pow_65537_mod_19_l217_217736

theorem two_pow_65537_mod_19 : (2 ^ 65537) % 19 = 2 := by
  -- We will use Fermat's Little Theorem and given conditions.
  sorry

end two_pow_65537_mod_19_l217_217736


namespace kernels_needed_for_movie_night_l217_217474

structure PopcornPreferences where
  caramel_popcorn: ℝ
  butter_popcorn: ℝ
  cheese_popcorn: ℝ
  kettle_corn_popcorn: ℝ

noncomputable def total_kernels_needed (preferences: PopcornPreferences) : ℝ :=
  (preferences.caramel_popcorn / 6) * 3 +
  (preferences.butter_popcorn / 4) * 2 +
  (preferences.cheese_popcorn / 8) * 4 +
  (preferences.kettle_corn_popcorn / 3) * 1

theorem kernels_needed_for_movie_night :
  let preferences := PopcornPreferences.mk 3 4 6 3
  total_kernels_needed preferences = 7.5 :=
sorry

end kernels_needed_for_movie_night_l217_217474


namespace difference_of_squares_l217_217436

theorem difference_of_squares (x y : ℕ) (h₁ : x + y = 22) (h₂ : x * y = 120) (h₃ : x > y) : 
  x^2 - y^2 = 44 :=
sorry

end difference_of_squares_l217_217436


namespace max_a_value_l217_217706

def f (a x : ℝ) : ℝ := x^3 - a*x^2 + (a^2 - 2)*x + 1

theorem max_a_value (a : ℝ) :
  (∃ m : ℝ, m > 0 ∧ f a m ≤ 0) → a ≤ 1 :=
by
  intro h
  sorry

end max_a_value_l217_217706


namespace find_value_of_expression_l217_217001

theorem find_value_of_expression :
  3 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2400 :=
by
  sorry

end find_value_of_expression_l217_217001


namespace Adam_teaches_students_l217_217198

-- Define the conditions
def students_first_year : ℕ := 40
def students_per_year : ℕ := 50
def total_years : ℕ := 10
def remaining_years : ℕ := total_years - 1

-- Define the statement we are proving
theorem Adam_teaches_students (total_students : ℕ) :
  total_students = students_first_year + (students_per_year * remaining_years) :=
sorry

end Adam_teaches_students_l217_217198


namespace total_tourists_proof_l217_217568

noncomputable def calculate_total_tourists : ℕ :=
  let start_time := 8  
  let end_time := 17   -- 5 PM in 24-hour format
  let initial_tourists := 120
  let increment := 2
  let number_of_trips := end_time - start_time  -- total number of trips including both start and end
  let first_term := initial_tourists
  let last_term := initial_tourists + increment * (number_of_trips - 1)
  (number_of_trips * (first_term + last_term)) / 2

theorem total_tourists_proof : calculate_total_tourists = 1290 := by
  sorry

end total_tourists_proof_l217_217568


namespace three_layer_carpet_area_l217_217805

-- Define the dimensions of the carpets and the hall
structure Carpet := (width : ℕ) (height : ℕ)

def principal_carpet : Carpet := ⟨6, 8⟩
def caretaker_carpet : Carpet := ⟨6, 6⟩
def parent_committee_carpet : Carpet := ⟨5, 7⟩
def hall : Carpet := ⟨10, 10⟩

-- Define the area function
def area (c : Carpet) : ℕ := c.width * c.height

-- Prove the area of the part of the hall covered by all three carpets
theorem three_layer_carpet_area : area ⟨3, 2⟩ = 6 :=
by
  sorry

end three_layer_carpet_area_l217_217805


namespace dima_story_telling_l217_217424

theorem dima_story_telling (initial_spoons final_spoons : ℕ) 
  (h1 : initial_spoons = 26) (h2 : final_spoons = 33696)
  (h3 : ∃ (n : ℕ), final_spoons = initial_spoons * (2^5 * 3^4) * 13) : 
  ∃ n : ℕ, n = 9 := 
sorry

end dima_story_telling_l217_217424


namespace greatest_integer_jean_thinks_of_l217_217089

theorem greatest_integer_jean_thinks_of :
  ∃ n : ℕ, n < 150 ∧ (∃ a : ℤ, n + 2 = 9 * a) ∧ (∃ b : ℤ, n + 3 = 11 * b) ∧ n = 142 :=
by
  sorry

end greatest_integer_jean_thinks_of_l217_217089


namespace cost_of_song_book_l217_217998

-- Definitions of the constants:
def cost_of_flute : ℝ := 142.46
def cost_of_music_stand : ℝ := 8.89
def total_spent : ℝ := 158.35

-- Definition of the combined cost of the flute and music stand:
def combined_cost := cost_of_flute + cost_of_music_stand

-- The final theorem to prove that the cost of the song book is $7.00:
theorem cost_of_song_book : total_spent - combined_cost = 7.00 := by
  sorry

end cost_of_song_book_l217_217998


namespace hayden_ironing_weeks_l217_217557

variable (total_daily_minutes : Nat := 5 + 3)
variable (days_per_week : Nat := 5)
variable (total_minutes : Nat := 160)

def calculate_weeks (total_daily_minutes : Nat) (days_per_week : Nat) (total_minutes : Nat) : Nat :=
  total_minutes / (total_daily_minutes * days_per_week)

theorem hayden_ironing_weeks :
  calculate_weeks (5 + 3) 5 160 = 4 := 
by
  sorry

end hayden_ironing_weeks_l217_217557


namespace same_sign_iff_product_positive_different_sign_iff_product_negative_l217_217393

variable (a b : ℝ)

theorem same_sign_iff_product_positive :
  ((a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0)) ↔ (a * b > 0) :=
sorry

theorem different_sign_iff_product_negative :
  ((a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)) ↔ (a * b < 0) :=
sorry

end same_sign_iff_product_positive_different_sign_iff_product_negative_l217_217393


namespace sum_first_100_odd_l217_217901

-- Define the sequence of odd numbers.
def odd (n : ℕ) : ℕ := 2 * n + 1

-- Define the sum of the first n odd natural numbers.
def sumOdd (n : ℕ) : ℕ := (n * (n + 1))

-- State the theorem.
theorem sum_first_100_odd : sumOdd 100 = 10000 :=
by
  -- Skipping the proof as per the instructions
  sorry

end sum_first_100_odd_l217_217901


namespace value_of_m_over_q_l217_217716

-- Definitions for the given conditions
variables (n m p q : ℤ) 

-- Main theorem statement
theorem value_of_m_over_q (h1 : m = 10 * n) (h2 : p = 2 * n) (h3 : p = q / 5) :
  m / q = 1 :=
sorry

end value_of_m_over_q_l217_217716


namespace total_pages_to_read_l217_217290

theorem total_pages_to_read 
  (total_books : ℕ)
  (pages_per_book : ℕ)
  (books_read_first_month : ℕ)
  (books_remaining_second_month : ℕ) :
  total_books = 14 →
  pages_per_book = 200 →
  books_read_first_month = 4 →
  books_remaining_second_month = (total_books - books_read_first_month) / 2 →
  ((total_books * pages_per_book) - ((books_read_first_month + books_remaining_second_month) * pages_per_book) = 1000) :=
by
  sorry

end total_pages_to_read_l217_217290


namespace A_in_terms_of_B_l217_217669

-- Definitions based on conditions
def f (A B x : ℝ) : ℝ := A * x^2 - 3 * B^3
def g (B x : ℝ) : ℝ := B * x^2

-- Theorem statement
theorem A_in_terms_of_B (A B : ℝ) (hB : B ≠ 0) (h : f A B (g B 2) = 0) : A = 3 * B / 16 :=
by
  -- Proof omitted
  sorry

end A_in_terms_of_B_l217_217669


namespace number_of_students_l217_217205

/-- 
We are given that 36 students are selected from three grades: 
15 from the first grade, 12 from the second grade, and the rest from the third grade. 
Additionally, there are 900 students in the third grade.
We need to prove: the total number of students in the high school is 3600
-/
theorem number_of_students (x y z : ℕ) (s_total : ℕ) (x_sel : ℕ) (y_sel : ℕ) (z_students : ℕ) 
  (h1 : x_sel = 15) 
  (h2 : y_sel = 12) 
  (h3 : x_sel + y_sel + (s_total - (x_sel + y_sel)) = s_total) 
  (h4 : s_total = 36) 
  (h5 : z_students = 900) 
  (h6 : (s_total - (x_sel + y_sel)) = 9) 
  (h7 : 9 / 900 = 1 / 100) : 
  (36 * 100 = 3600) :=
by sorry

end number_of_students_l217_217205


namespace positive_difference_complementary_angles_l217_217209

theorem positive_difference_complementary_angles (a b : ℝ) 
  (h1 : a + b = 90) 
  (h2 : 3 * b = a) :
  |a - b| = 45 :=
by
  sorry

end positive_difference_complementary_angles_l217_217209


namespace calculate_f_at_2x_l217_217031

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem using the given condition and the desired result
theorem calculate_f_at_2x (x : ℝ) : f (2 * x) = 4 * x^2 - 1 :=
by
  sorry

end calculate_f_at_2x_l217_217031


namespace inequality_proof_l217_217385

theorem inequality_proof (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) :
  (a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1)) ≤ 1 / 2 :=
sorry

end inequality_proof_l217_217385


namespace events_per_coach_l217_217130

theorem events_per_coach {students events_per_student coaches events total_participations total_events : ℕ} 
  (h1 : students = 480) 
  (h2 : events_per_student = 4) 
  (h3 : (students * events_per_student) = total_participations) 
  (h4 : ¬ students * events_per_student ≠ total_participations)
  (h5 : total_participations = 1920) 
  (h6 : (total_participations / 20) = total_events) 
  (h7 : ¬ total_participations / 20 ≠ total_events)
  (h8 : total_events = 96)
  (h9 : coaches = 16) :
  (total_events / coaches) = 6 := sorry

end events_per_coach_l217_217130


namespace inverse_sum_l217_217798

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then 3 - x else 2 * x - x^2

theorem inverse_sum :
  let f_inv_2 := (1 + Real.sqrt 3)
  let f_inv_1 := 2
  let f_inv_4 := -1
  f_inv_2 + f_inv_1 + f_inv_4 = 2 + Real.sqrt 3 :=
by
  sorry

end inverse_sum_l217_217798


namespace composite_solid_volume_l217_217747

theorem composite_solid_volume :
  let V_prism := 2 * 2 * 1
  let V_cylinder := Real.pi * 1^2 * 3
  let V_overlap := Real.pi / 2
  V_prism + V_cylinder - V_overlap = 4 + 5 * Real.pi / 2 :=
by
  sorry

end composite_solid_volume_l217_217747


namespace Jeff_total_ounces_of_peanut_butter_l217_217741

theorem Jeff_total_ounces_of_peanut_butter
    (jars : ℕ)
    (equal_count : ℕ)
    (total_jars : jars = 9)
    (j16 : equal_count = 3) 
    (j28 : equal_count = 3)
    (j40 : equal_count = 3) :
    (3 * 16 + 3 * 28 + 3 * 40 = 252) :=
by
  sorry

end Jeff_total_ounces_of_peanut_butter_l217_217741


namespace initial_speed_is_7_l217_217059

-- Definitions based on conditions
def distance_travelled (S : ℝ) (T : ℝ) : ℝ := S * T

-- Constants from problem
def time_initial : ℝ := 6
def time_final : ℝ := 3
def speed_final : ℝ := 14

-- Theorem statement
theorem initial_speed_is_7 : ∃ S : ℝ, distance_travelled S time_initial = distance_travelled speed_final time_final ∧ S = 7 := by
  sorry

end initial_speed_is_7_l217_217059


namespace x_zero_necessary_but_not_sufficient_l217_217585

-- Definitions based on conditions
def x_eq_zero (x : ℝ) := x = 0
def xsq_plus_ysq_eq_zero (x y : ℝ) := x^2 + y^2 = 0

-- Statement that x = 0 is a necessary but not sufficient condition for x^2 + y^2 = 0
theorem x_zero_necessary_but_not_sufficient (x y : ℝ) : (x = 0 ↔ x^2 + y^2 = 0) → False :=
by sorry

end x_zero_necessary_but_not_sufficient_l217_217585


namespace soccer_game_goals_l217_217038

theorem soccer_game_goals (A1_first_half A2_first_half B1_first_half B2_first_half : ℕ) 
  (h1 : A1_first_half = 8)
  (h2 : B1_first_half = A1_first_half / 2)
  (h3 : B2_first_half = A1_first_half)
  (h4 : A2_first_half = B2_first_half - 2) : 
  A1_first_half + A2_first_half + B1_first_half + B2_first_half = 26 :=
by
  -- The proof is not needed, so we use sorry to skip it.
  sorry

end soccer_game_goals_l217_217038


namespace sqrt_defined_value_l217_217992

theorem sqrt_defined_value (x : ℝ) (h : x ≥ 4) : x = 5 → true := 
by 
  intro hx
  sorry

end sqrt_defined_value_l217_217992


namespace calculate_expression_l217_217784

variable (y : ℝ) (π : ℝ) (Q : ℝ)

theorem calculate_expression (h : 5 * (3 * y - 7 * π) = Q) : 
  10 * (6 * y - 14 * π) = 4 * Q := by
  sorry

end calculate_expression_l217_217784


namespace triangle_angle_ratio_arbitrary_convex_quadrilateral_angle_ratio_not_arbitrary_convex_pentagon_angle_ratio_not_arbitrary_l217_217652

theorem triangle_angle_ratio_arbitrary (k1 k2 k3 : ℕ) :
  ∃ (A B C : ℝ), A + B + C = 180 ∧ (A / B = k1 / k2) ∧ (A / C = k1 / k3) :=
  sorry

theorem convex_quadrilateral_angle_ratio_not_arbitrary (k1 k2 k3 k4 : ℕ) :
  ¬(∃ (A B C D : ℝ), A + B + C + D = 360 ∧
  A < B + C + D ∧
  B < A + C + D ∧
  C < A + B + D ∧
  D < A + B + C) :=
  sorry

theorem convex_pentagon_angle_ratio_not_arbitrary (k1 k2 k3 k4 k5 : ℕ) :
  ¬(∃ (A B C D E : ℝ), A + B + C + D + E = 540 ∧
  A < (B + C + D + E) / 2 ∧
  B < (A + C + D + E) / 2 ∧
  C < (A + B + D + E) / 2 ∧
  D < (A + B + C + E) / 2 ∧
  E < (A + B + C + D) / 2) :=
  sorry

end triangle_angle_ratio_arbitrary_convex_quadrilateral_angle_ratio_not_arbitrary_convex_pentagon_angle_ratio_not_arbitrary_l217_217652


namespace contrapositive_of_square_root_l217_217655

theorem contrapositive_of_square_root (a b : ℝ) :
  (a^2 < b → -Real.sqrt b < a ∧ a < Real.sqrt b) ↔ (a ≥ Real.sqrt b ∨ a ≤ -Real.sqrt b → a^2 ≥ b) := 
sorry

end contrapositive_of_square_root_l217_217655


namespace range_of_m_l217_217764

def p (x : ℝ) : Prop := abs (1 - (x - 1) / 3) ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : 
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) → 9 ≤ m :=
by
  sorry

end range_of_m_l217_217764


namespace leftover_value_is_5_30_l217_217473

variable (q_per_roll d_per_roll : ℕ)
variable (j_quarters j_dimes l_quarters l_dimes : ℕ)
variable (value_per_quarter value_per_dime : ℝ)

def total_leftover_value (q_per_roll d_per_roll : ℕ) 
  (j_quarters l_quarters j_dimes l_dimes : ℕ)
  (value_per_quarter value_per_dime : ℝ) : ℝ :=
  let total_quarters := j_quarters + l_quarters
  let total_dimes := j_dimes + l_dimes
  let leftover_quarters := total_quarters % q_per_roll
  let leftover_dimes := total_dimes % d_per_roll
  (leftover_quarters * value_per_quarter) + (leftover_dimes * value_per_dime)

theorem leftover_value_is_5_30 :
  total_leftover_value 45 55 95 140 173 285 0.25 0.10 = 5.3 := 
by
  sorry

end leftover_value_is_5_30_l217_217473


namespace fountain_pen_price_l217_217538

theorem fountain_pen_price
  (n_fpens : ℕ) (n_mpens : ℕ) (total_cost : ℕ) (avg_cost_mpens : ℝ)
  (hpens : n_fpens = 450) (mpens : n_mpens = 3750) 
  (htotal : total_cost = 11250) (havg_mpens : avg_cost_mpens = 2.25) : 
  (total_cost - n_mpens * avg_cost_mpens) / n_fpens = 6.25 :=
by
  sorry

end fountain_pen_price_l217_217538


namespace shorter_leg_of_right_triangle_l217_217088

theorem shorter_leg_of_right_triangle {a b : ℕ} (h : a^2 + b^2 = 65^2) (ha : a ≤ b) : a = 25 :=
by sorry

end shorter_leg_of_right_triangle_l217_217088


namespace not_universally_better_l217_217337

-- Definitions based on the implicitly given conditions
def can_show_quantity (chart : Type) : Prop := sorry
def can_reflect_changes (chart : Type) : Prop := sorry

-- Definitions of bar charts and line charts
inductive BarChart
| mk : BarChart

inductive LineChart
| mk : LineChart

-- Assumptions based on characteristics of the charts
axiom bar_chart_shows_quantity : can_show_quantity BarChart 
axiom line_chart_shows_quantity : can_show_quantity LineChart 
axiom line_chart_reflects_changes : can_reflect_changes LineChart 

-- Proof problem statement
theorem not_universally_better : ¬(∀ (c1 c2 : Type), can_show_quantity c1 → can_reflect_changes c1 → ¬can_show_quantity c2 → ¬can_reflect_changes c2) :=
  sorry

end not_universally_better_l217_217337


namespace sufficient_but_not_necessary_condition_l217_217505

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.tan (ω * x + φ)
def P (f : ℝ → ℝ) : Prop := f 0 = 0
def Q (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem sufficient_but_not_necessary_condition (ω : ℝ) (φ : ℝ) (hω : ω > 0) :
  (P (f ω φ) → Q (f ω φ)) ∧ ¬(Q (f ω φ) → P (f ω φ)) := by
  sorry

end sufficient_but_not_necessary_condition_l217_217505


namespace amy_money_left_l217_217613

-- Definitions for item prices
def stuffed_toy_price : ℝ := 2
def hot_dog_price : ℝ := 3.5
def candy_apple_price : ℝ := 1.5
def soda_price : ℝ := 1.75
def ferris_wheel_ticket_price : ℝ := 2.5

-- Tax rate
def tax_rate : ℝ := 0.1 

-- Initial amount Amy had
def initial_amount : ℝ := 15

-- Function to calculate price including tax
def price_with_tax (price : ℝ) (tax_rate : ℝ) : ℝ := price * (1 + tax_rate)

-- Prices including tax
def stuffed_toy_price_with_tax := price_with_tax stuffed_toy_price tax_rate
def hot_dog_price_with_tax := price_with_tax hot_dog_price tax_rate
def candy_apple_price_with_tax := price_with_tax candy_apple_price tax_rate
def soda_price_with_tax := price_with_tax soda_price tax_rate
def ferris_wheel_ticket_price_with_tax := price_with_tax ferris_wheel_ticket_price tax_rate

-- Discount rates
def discount_most_expensive : ℝ := 0.5
def discount_second_most_expensive : ℝ := 0.25

-- Applying discounts
def discounted_hot_dog_price := hot_dog_price_with_tax * (1 - discount_most_expensive)
def discounted_ferris_wheel_ticket_price := ferris_wheel_ticket_price_with_tax * (1 - discount_second_most_expensive)

-- Total cost with discounts
def total_cost_with_discounts : ℝ := 
  stuffed_toy_price_with_tax + discounted_hot_dog_price + candy_apple_price_with_tax +
  soda_price_with_tax + discounted_ferris_wheel_ticket_price

-- Amount left after purchases
def amount_left : ℝ := initial_amount - total_cost_with_discounts

theorem amy_money_left : amount_left = 5.23 := by
  -- Here the proof will be provided.
  sorry

end amy_money_left_l217_217613


namespace negation_log2_property_l217_217873

theorem negation_log2_property :
  ¬(∃ x₀ : ℝ, Real.log x₀ / Real.log 2 ≤ 0) ↔ ∀ x : ℝ, Real.log x / Real.log 2 > 0 :=
by
  sorry

end negation_log2_property_l217_217873


namespace vote_proportion_inequality_l217_217673

theorem vote_proportion_inequality
  (a b k : ℕ)
  (hb_odd : b % 2 = 1)
  (hb_min : 3 ≤ b)
  (vote_same : ∀ (i j : ℕ) (hi hj : i ≠ j) (votes : ℕ → ℕ), ∃ (k_max : ℕ), ∀ (cont : ℕ), votes cont ≤ k_max) :
  (k : ℚ) / a ≥ (b - 1) / (2 * b) := sorry

end vote_proportion_inequality_l217_217673


namespace proof_l217_217694

-- Define proposition p as negated form: ∀ x < 1, log_3 x ≤ 0
def p : Prop := ∀ x : ℝ, x < 1 → Real.log x / Real.log 3 ≤ 0

-- Define proposition q: ∃ x_0 ∈ ℝ, x_0^2 ≥ 2^x_0
def q : Prop := ∃ x_0 : ℝ, x_0^2 ≥ Real.exp (x_0 * Real.log 2)

-- State we need to prove: p ∨ q
theorem proof : p ∨ q := sorry

end proof_l217_217694


namespace complement_of_angle_l217_217765

theorem complement_of_angle (supplement : ℝ) (h_supp : supplement = 130) (original_angle : ℝ) (h_orig : original_angle = 180 - supplement) : 
  (90 - original_angle) = 40 := 
by 
  -- proof goes here
  sorry

end complement_of_angle_l217_217765


namespace matilda_father_chocolates_left_l217_217636

-- definitions for each condition
def initial_chocolates : ℕ := 20
def persons : ℕ := 5
def chocolates_per_person := initial_chocolates / persons
def half_chocolates_per_person := chocolates_per_person / 2
def total_given_to_father := half_chocolates_per_person * persons
def chocolates_given_to_mother := 3
def chocolates_eaten_by_father := 2

-- statement to prove
theorem matilda_father_chocolates_left :
  total_given_to_father - chocolates_given_to_mother - chocolates_eaten_by_father = 5 :=
by
  sorry

end matilda_father_chocolates_left_l217_217636


namespace rectangle_breadth_l217_217054

theorem rectangle_breadth (l b : ℕ) (hl : l = 15) (h : l * b = 15 * b) (h2 : l - b = 10) : b = 5 := 
sorry

end rectangle_breadth_l217_217054


namespace arithmetic_seq_sum_l217_217579

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 3 = 9)
  (h3 : a 5 = 5) :
  S 9 / S 5 = 1 :=
by
  sorry

end arithmetic_seq_sum_l217_217579


namespace problem_statement_l217_217674

noncomputable def c := 3 + Real.sqrt 21
noncomputable def d := 3 - Real.sqrt 21

theorem problem_statement : 
  (c + 2 * d) = 9 - Real.sqrt 21 :=
by
  sorry

end problem_statement_l217_217674


namespace carson_circles_theorem_l217_217517

-- Define the dimensions of the warehouse
def warehouse_length : ℕ := 600
def warehouse_width : ℕ := 400

-- Define the perimeter calculation
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- Define the distance Carson walked
def distance_walked : ℕ := 16000

-- Define the number of circles Carson skipped
def circles_skipped : ℕ := 2

-- Define the expected number of circles Carson was supposed to circle
def expected_circles :=
  let actual_circles := distance_walked / (perimeter warehouse_length warehouse_width)
  actual_circles + circles_skipped

-- The theorem we want to prove
theorem carson_circles_theorem : expected_circles = 10 := by
  sorry

end carson_circles_theorem_l217_217517


namespace sum_of_fifth_powers_l217_217221

variables {a b c d : ℝ}

theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l217_217221


namespace part1_daily_sales_profit_final_max_daily_sales_profit_l217_217929

-- Conditions from part (a)
def original_selling_price : ℚ := 30
def cost_price : ℚ := 15
def original_sales_volume : ℚ := 60
def sales_increase_per_yuan : ℚ := 10

-- Part (1): Daily sales profit if the price is reduced by 2 yuan
def new_selling_price1 : ℚ := original_selling_price - 2
def new_sales_volume1 : ℚ := original_sales_volume + (2 * sales_increase_per_yuan)
def profit_per_kilogram1 : ℚ := new_selling_price1 - cost_price
def daily_sales_profit1 : ℚ := profit_per_kilogram1 * new_sales_volume1

theorem part1_daily_sales_profit : daily_sales_profit1 = 1040 := by
  sorry

-- Part (2): Maximum daily sales profit and corresponding selling price
def selling_price_at_max_profit : ℚ := 51 / 2

def daily_profit (x : ℚ) : ℚ :=
  (x - cost_price) * (original_sales_volume + (original_selling_price - x) * sales_increase_per_yuan)

theorem final_max_daily_sales_profit :
  (∀ x : ℚ, daily_profit x ≤ daily_profit selling_price_at_max_profit) ∧ daily_profit selling_price_at_max_profit = 1102.5 := by
  sorry

end part1_daily_sales_profit_final_max_daily_sales_profit_l217_217929


namespace Rachel_spent_on_lunch_fraction_l217_217172

variable {MoneyEarned MoneySpentOnDVD MoneyLeft MoneySpentOnLunch : ℝ}

-- Given conditions
axiom Rachel_earnings : MoneyEarned = 200
axiom Rachel_spent_on_DVD : MoneySpentOnDVD = MoneyEarned / 2
axiom Rachel_leftover : MoneyLeft = 50
axiom Rachel_total_spent : MoneyEarned - MoneyLeft = MoneySpentOnLunch + MoneySpentOnDVD

-- Prove that Rachel spent 1/4 of her money on lunch
theorem Rachel_spent_on_lunch_fraction :
  MoneySpentOnLunch / MoneyEarned = 1 / 4 :=
sorry

end Rachel_spent_on_lunch_fraction_l217_217172


namespace find_number_l217_217005

theorem find_number (number : ℚ) 
  (H1 : 8 * 60 = 480)
  (H2 : number / 6 = 16 / 480) :
  number = 1 / 5 := 
by
  sorry

end find_number_l217_217005


namespace impossible_to_use_up_components_l217_217485

theorem impossible_to_use_up_components 
  (p q r x y z : ℕ) 
  (condition1 : 2 * x + 2 * z = 2 * p + 2 * r + 2)
  (condition2 : 2 * x + y = 2 * p + q + 1)
  (condition3 : y + z = q + r) : 
  False :=
by sorry

end impossible_to_use_up_components_l217_217485


namespace range_of_m_l217_217392

def p (m : ℝ) : Prop :=
  ∀ (x : ℝ), 2^x - m + 1 > 0

def q (m : ℝ) : Prop :=
  5 - 2 * m > 1

theorem range_of_m (m : ℝ) (hpq : p m ∧ q m) : m ≤ 1 := sorry

end range_of_m_l217_217392


namespace total_expenditure_l217_217860

-- Definitions of costs and purchases
def bracelet_cost : ℕ := 4
def keychain_cost : ℕ := 5
def coloring_book_cost : ℕ := 3

def paula_bracelets : ℕ := 2
def paula_keychains : ℕ := 1

def olive_coloring_books : ℕ := 1
def olive_bracelets : ℕ := 1

-- Hypothesis stating the total expenditure for Paula and Olive
theorem total_expenditure
  (bracelet_cost keychain_cost coloring_book_cost : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ) :
  paula_bracelets * bracelet_cost + paula_keychains * keychain_cost + olive_coloring_books * coloring_book_cost + olive_bracelets * bracelet_cost = 20 := 
  by
  -- Applying the given costs
  let bracelet_cost := 4
  let keychain_cost := 5
  let coloring_book_cost := 3 

  -- Applying the purchases made by Paula and Olive
  let paula_bracelets := 2
  let paula_keychains := 1
  let olive_coloring_books := 1
  let olive_bracelets := 1

  sorry

end total_expenditure_l217_217860


namespace parallel_line_slope_y_intercept_l217_217261

theorem parallel_line_slope_y_intercept (x y : ℝ) (h : 3 * x - 6 * y = 12) :
  ∃ (m b : ℝ), m = 1 / 2 ∧ b = -2 := 
by { sorry }

end parallel_line_slope_y_intercept_l217_217261


namespace trigonometric_identity_l217_217041

theorem trigonometric_identity :
  (1 / Real.cos 80) - (Real.sqrt 3 / Real.cos 10) = 4 :=
by
  sorry

end trigonometric_identity_l217_217041


namespace perimeter_of_flowerbed_l217_217864

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem perimeter_of_flowerbed : perimeter length width = 22 := by
  sorry

end perimeter_of_flowerbed_l217_217864


namespace determinant_transformation_l217_217482

theorem determinant_transformation 
  (p q r s : ℝ)
  (h : Matrix.det ![![p, q], ![r, s]] = 6) :
  Matrix.det ![![p, 9 * p + 4 * q], ![r, 9 * r + 4 * s]] = 24 := 
sorry

end determinant_transformation_l217_217482


namespace jessica_total_cost_l217_217230

def price_of_cat_toy : ℝ := 10.22
def price_of_cage : ℝ := 11.73
def price_of_cat_food : ℝ := 5.65
def price_of_catnip : ℝ := 2.30
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.07

def discounted_price_of_cat_toy : ℝ := price_of_cat_toy * (1 - discount_rate)
def total_cost_before_tax : ℝ := discounted_price_of_cat_toy + price_of_cage + price_of_cat_food + price_of_catnip
def sales_tax : ℝ := total_cost_before_tax * tax_rate
def total_cost_after_discount_and_tax : ℝ := total_cost_before_tax + sales_tax

theorem jessica_total_cost : total_cost_after_discount_and_tax = 30.90 := by
  sorry

end jessica_total_cost_l217_217230


namespace division_of_monomials_l217_217948

variable (x : ℝ) -- ensure x is defined as a variable, here assuming x is a real number

theorem division_of_monomials (x : ℝ) : (2 * x^3 / x^2) = 2 * x := 
by 
  sorry

end division_of_monomials_l217_217948


namespace volume_of_largest_sphere_from_cube_l217_217218

theorem volume_of_largest_sphere_from_cube : 
  (∃ (V : ℝ), 
    (∀ (l : ℝ), l = 1 → (V = (4 / 3) * π * ((l / 2)^3)) → V = π / 6)) :=
sorry

end volume_of_largest_sphere_from_cube_l217_217218


namespace probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40_l217_217679

/-- 
There are 30 tiles in box C numbered from 1 to 30 and 30 tiles in box D numbered from 21 to 50. 
We want to prove that the probability of drawing a tile less than 20 from box C and a tile that 
is either odd or greater than 40 from box D is 19/45. 
-/
theorem probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40 :
  (19 / 30) * (2 / 3) = (19 / 45) :=
by sorry

end probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40_l217_217679


namespace cost_of_each_nose_spray_l217_217376

def total_nose_sprays : ℕ := 10
def total_cost : ℝ := 15
def buy_one_get_one_free : Bool := true

theorem cost_of_each_nose_spray :
  buy_one_get_one_free = true →
  total_nose_sprays = 10 →
  total_cost = 15 →
  (total_cost / (total_nose_sprays / 2)) = 3 :=
by
  intros h1 h2 h3
  sorry

end cost_of_each_nose_spray_l217_217376


namespace problem_statement_l217_217422

theorem problem_statement (m n : ℝ) 
  (h₁ : m^2 - 1840 * m + 2009 = 0)
  (h₂ : n^2 - 1840 * n + 2009 = 0) : 
  (m^2 - 1841 * m + 2009) * (n^2 - 1841 * n + 2009) = 2009 :=
sorry

end problem_statement_l217_217422


namespace shelves_used_l217_217594

theorem shelves_used (initial_books : ℕ) (sold_books : ℕ) (books_per_shelf : ℕ) (remaining_books : ℕ) (total_shelves : ℕ) :
  initial_books = 120 → sold_books = 39 → books_per_shelf = 9 → remaining_books = initial_books - sold_books → total_shelves = remaining_books / books_per_shelf → total_shelves = 9 :=
by
  intros h_initial_books h_sold_books h_books_per_shelf h_remaining_books h_total_shelves
  rw [h_initial_books, h_sold_books] at h_remaining_books
  rw [h_books_per_shelf, h_remaining_books] at h_total_shelves
  exact h_total_shelves

end shelves_used_l217_217594


namespace reflected_ray_equation_l217_217657

-- Define the initial point
def point_of_emanation : (ℝ × ℝ) := (-1, 3)

-- Define the point after reflection which the ray passes through
def point_after_reflection : (ℝ × ℝ) := (4, 6)

-- Define the expected equation of the line in general form
def expected_line_equation (x y : ℝ) : Prop := 9 * x - 5 * y - 6 = 0

-- The theorem we need to prove
theorem reflected_ray_equation :
  ∃ (m b : ℝ), ∀ x y : ℝ, (y = m * x + b) → expected_line_equation x y :=
sorry

end reflected_ray_equation_l217_217657


namespace sum_ninth_power_l217_217994

theorem sum_ninth_power (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) 
                        (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7)
                        (h5 : a^5 + b^5 = 11)
                        (h_ind : ∀ n, n ≥ 3 → a^n + b^n = a^(n-1) + b^(n-1) + a^(n-2) + b^(n-2)) :
  a^9 + b^9 = 76 :=
by
  sorry

end sum_ninth_power_l217_217994


namespace right_triangle_excircle_incircle_l217_217351

theorem right_triangle_excircle_incircle (a b c r r_a : ℝ) (h : a^2 + b^2 = c^2) :
  (r = (a + b - c) / 2) → (r_a = (b + c - a) / 2) → r_a = 2 * r :=
by
  intros hr hra
  sorry

end right_triangle_excircle_incircle_l217_217351


namespace largest_number_l217_217656

def HCF (a b c d : ℕ) : Prop := d ∣ a ∧ d ∣ b ∧ d ∣ c ∧ 
                                ∀ e, (e ∣ a ∧ e ∣ b ∧ e ∣ c) → e ≤ d
def LCM (a b c m : ℕ) : Prop := m % a = 0 ∧ m % b = 0 ∧ m % c = 0 ∧ 
                                ∀ n, (n % a = 0 ∧ n % b = 0 ∧ n % c = 0) → m ≤ n

theorem largest_number (a b c : ℕ)
  (hcf: HCF a b c 210)
  (lcm_has_factors: ∃ k1 k2 k3, k1 = 11 ∧ k2 = 17 ∧ k3 = 23 ∧
                                LCM a b c (210 * k1 * k2 * k3)) :
  max a (max b c) = 4830 := 
by
  sorry

end largest_number_l217_217656


namespace calculate_expression_l217_217607

variables (x y : ℝ)

theorem calculate_expression (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x - y) / (Real.sqrt x + Real.sqrt y) - (x - 2 * Real.sqrt (x * y) + y) / (Real.sqrt x - Real.sqrt y) = 0 :=
by
  sorry

end calculate_expression_l217_217607


namespace geometric_sequence_common_ratio_l217_217242

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (a_mono : ∀ n, a n < a (n+1))
    (a2a5_eq_6 : a 2 * a 5 = 6)
    (a3a4_eq_5 : a 3 + a 4 = 5) 
    (q : ℝ) (hq : ∀ n, a n = a 1 * q ^ (n - 1)) :
    q = 3 / 2 :=
by
    sorry

end geometric_sequence_common_ratio_l217_217242


namespace chris_sick_weeks_l217_217085

theorem chris_sick_weeks :
  ∀ (h1 : ∀ w : ℕ, w = 4 → 2 * w = 8),
    ∀ (h2 : ∀ h w : ℕ, h = 20 → ∀ m : ℕ, 2 * (w * m) = 160),
    ∀ (h3 : ∀ h : ℕ, h = 180 → 180 - 160 = 20),
    ∀ (h4 : ∀ h w : ℕ, h = 20 → w = 20 → 20 / 20 = 1),
    180 - 160 = (20 / 20) * 20 :=
by
  intros
  sorry

end chris_sick_weeks_l217_217085


namespace total_tank_capacity_l217_217174

-- Definitions based on conditions
def initial_condition (w c : ℝ) : Prop := w / c = 1 / 3
def after_adding_five (w c : ℝ) : Prop := (w + 5) / c = 1 / 2

-- The problem statement
theorem total_tank_capacity (w c : ℝ) (h1 : initial_condition w c) (h2 : after_adding_five w c) : c = 30 :=
sorry

end total_tank_capacity_l217_217174


namespace points_on_equation_correct_l217_217851

theorem points_on_equation_correct (x y : ℝ) :
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) :=
by 
  sorry

end points_on_equation_correct_l217_217851


namespace unit_conversion_factor_l217_217013

theorem unit_conversion_factor (u : ℝ) (h₁ : u = 5) (h₂ : (u * 0.9)^2 = 20.25) : u = 5 → (1 : ℝ) = 0.9  :=
sorry

end unit_conversion_factor_l217_217013


namespace total_amount_shared_l217_217145

theorem total_amount_shared (a b c d : ℝ) (h1 : a = (1/3) * (b + c + d)) 
    (h2 : b = (2/7) * (a + c + d)) (h3 : c = (4/9) * (a + b + d)) 
    (h4 : d = (5/11) * (a + b + c)) (h5 : a = b + 20) (h6 : c = d - 15) 
    (h7 : (a + b + c + d) % 10 = 0) : a + b + c + d = 1330 :=
by
  sorry

end total_amount_shared_l217_217145


namespace simplify_expr_at_sqrt6_l217_217048

noncomputable def simplifyExpression (x : ℝ) : ℝ :=
  (1 / (Real.sqrt (3 + x) * Real.sqrt (x + 2)) + 1 / (Real.sqrt (3 - x) * Real.sqrt (x - 2))) /
  (1 / (Real.sqrt (3 + x) * Real.sqrt (x + 2)) - 1 / (Real.sqrt (3 - x) * Real.sqrt (x - 2)))

theorem simplify_expr_at_sqrt6 : simplifyExpression (Real.sqrt 6) = - (Real.sqrt 6) / 2 :=
by
  sorry

end simplify_expr_at_sqrt6_l217_217048


namespace mean_cat_weights_l217_217791

-- Define a list representing the weights of the cats from the stem-and-leaf plot
def cat_weights : List ℕ := [12, 13, 14, 20, 21, 21, 25, 25, 28, 30, 31, 32, 32, 36, 38, 39, 39]

-- Function to calculate the sum of elements in a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Function to calculate the mean of a list of natural numbers
def mean_list (l : List ℕ) : ℚ := (sum_list l : ℚ) / l.length

-- The theorem we need to prove
theorem mean_cat_weights : mean_list cat_weights = 27 := by 
  sorry

end mean_cat_weights_l217_217791


namespace ratio_h_w_l217_217112

-- Definitions from conditions
variables (h w : ℝ)
variables (XY YZ : ℝ)
variables (h_pos : 0 < h) (w_pos : 0 < w) -- heights and widths are positive
variables (XY_pos : 0 < XY) (YZ_pos : 0 < YZ) -- segment lengths are positive

-- Given that in the right-angled triangle ∆XYZ, YZ = 2 * XY
axiom YZ_eq_2XY : YZ = 2 * XY

-- Prove that h / w = 3 / 8
theorem ratio_h_w (H : XY / YZ = 4 * h / (3 * w)) : h / w = 3 / 8 :=
by {
  -- Use the axioms and given conditions here to prove H == ratio
  sorry
}

end ratio_h_w_l217_217112


namespace Mille_suckers_l217_217893

theorem Mille_suckers:
  let pretzels := 64
  let goldfish := 4 * pretzels
  let baggies := 16
  let items_per_baggie := 22
  let total_items_needed := baggies * items_per_baggie
  let total_pretzels_and_goldfish := pretzels + goldfish
  let suckers := total_items_needed - total_pretzels_and_goldfish
  suckers = 32 := 
by sorry

end Mille_suckers_l217_217893


namespace petes_original_number_l217_217372

theorem petes_original_number (x : ℤ) (h : 5 * (3 * x - 6) = 195) : x = 15 :=
sorry

end petes_original_number_l217_217372


namespace number_of_boys_l217_217692

theorem number_of_boys 
  (B G : ℕ) 
  (h1 : B + G = 650) 
  (h2 : G = B + 106) :
  B = 272 :=
sorry

end number_of_boys_l217_217692


namespace common_divisor_of_differences_l217_217497

theorem common_divisor_of_differences 
  (a1 a2 b1 b2 c1 c2 d : ℤ) 
  (h1: d ∣ (a1 - a2)) 
  (h2: d ∣ (b1 - b2)) 
  (h3: d ∣ (c1 - c2)) : 
  d ∣ (a1 * b1 * c1 - a2 * b2 * c2) := 
by sorry

end common_divisor_of_differences_l217_217497


namespace base7_to_base10_conversion_l217_217063

theorem base7_to_base10_conversion (n : ℤ) (h : n = 2 * 7^2 + 4 * 7^1 + 6 * 7^0) : n = 132 := by
  sorry

end base7_to_base10_conversion_l217_217063


namespace total_selling_price_l217_217647

theorem total_selling_price (total_commissions : ℝ) (number_of_appliances : ℕ) (fixed_commission_rate_per_appliance : ℝ) (percentage_commission_rate : ℝ) :
  total_commissions = number_of_appliances * fixed_commission_rate_per_appliance + percentage_commission_rate * S →
  total_commissions = 662 →
  number_of_appliances = 6 →
  fixed_commission_rate_per_appliance = 50 →
  percentage_commission_rate = 0.10 →
  S = 3620 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end total_selling_price_l217_217647


namespace domain_of_log_base_half_l217_217207

noncomputable def domain_log_base_half : Set ℝ := { x : ℝ | x > 5 }

theorem domain_of_log_base_half :
  (∀ x : ℝ, x > 5 ↔ x - 5 > 0) →
  (domain_log_base_half = { x : ℝ | x - 5 > 0 }) :=
by
  sorry

end domain_of_log_base_half_l217_217207


namespace alex_minus_sam_eq_negative_2_50_l217_217491

def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.07
def discount_rate : ℝ := 0.15
def packaging_fee : ℝ := 2.50

def alex_total (original_price tax_rate discount_rate : ℝ) : ℝ :=
  let price_with_tax := original_price * (1 + tax_rate)
  let final_price := price_with_tax * (1 - discount_rate)
  final_price

def sam_total (original_price tax_rate discount_rate packaging_fee : ℝ) : ℝ :=
  let price_with_discount := original_price * (1 - discount_rate)
  let price_with_tax := price_with_discount * (1 + tax_rate)
  let final_price := price_with_tax + packaging_fee
  final_price

theorem alex_minus_sam_eq_negative_2_50 :
  alex_total original_price tax_rate discount_rate - sam_total original_price tax_rate discount_rate packaging_fee = -2.50 := by
  sorry

end alex_minus_sam_eq_negative_2_50_l217_217491


namespace square_area_inscribed_in_parabola_l217_217346

def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

theorem square_area_inscribed_in_parabola :
  ∃ s : ℝ, s = (-1 + Real.sqrt 5) ∧ (2 * s)^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end square_area_inscribed_in_parabola_l217_217346


namespace fraction_of_work_left_correct_l217_217991

-- Define the conditions for p, q, and r
def p_one_day_work : ℚ := 1 / 15
def q_one_day_work : ℚ := 1 / 20
def r_one_day_work : ℚ := 1 / 30

-- Define the total work done in one day by p, q, and r
def total_one_day_work : ℚ := p_one_day_work + q_one_day_work + r_one_day_work

-- Define the work done in 4 days
def work_done_in_4_days : ℚ := total_one_day_work * 4

-- Define the fraction of work left after 4 days
def fraction_of_work_left : ℚ := 1 - work_done_in_4_days

-- Statement to prove
theorem fraction_of_work_left_correct : fraction_of_work_left = 2 / 5 := by
  sorry

end fraction_of_work_left_correct_l217_217991


namespace arithmetic_sequence_a7_l217_217886

theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) (h3 : a 3 = 3) (h5 : a 5 = -3) : a 7 = -9 := 
sorry

end arithmetic_sequence_a7_l217_217886


namespace fisherman_sale_l217_217216

/-- 
If the price of the radio is both the 4th highest price and the 13th lowest price 
among the prices of the fishes sold at a sale, then the total number of fishes 
sold at the fisherman sale is 16. 
-/
theorem fisherman_sale (h4_highest : ∃ price : ℕ, ∀ p : ℕ, p > price → p ∈ {a | a ≠ price} ∧ p > 3)
                       (h13_lowest : ∃ price : ℕ, ∀ p : ℕ, p < price → p ∈ {a | a ≠ price} ∧ p < 13) :
  ∃ n : ℕ, n = 16 :=
sorry

end fisherman_sale_l217_217216


namespace red_more_than_yellow_l217_217650

-- Define the total number of marbles
def total_marbles : ℕ := 19

-- Define the number of yellow marbles
def yellow_marbles : ℕ := 5

-- Calculate the number of remaining marbles
def remaining_marbles : ℕ := total_marbles - yellow_marbles

-- Define the ratio of blue to red marbles
def blue_ratio : ℕ := 3
def red_ratio : ℕ := 4

-- Calculate the sum of ratio parts
def sum_ratio : ℕ := blue_ratio + red_ratio

-- Calculate the number of shares per ratio part
def share_per_part : ℕ := remaining_marbles / sum_ratio

-- Calculate the number of red marbles
def red_marbles : ℕ := red_ratio * share_per_part

-- Theorem to prove: the difference between red marbles and yellow marbles is 3
theorem red_more_than_yellow : red_marbles - yellow_marbles = 3 :=
by
  sorry

end red_more_than_yellow_l217_217650


namespace problem_statement_l217_217738

noncomputable def f (x : ℝ) : ℝ :=
  if (0 < x ∧ x < 1) then 4^x
  else if (-1 < x ∧ x < 0) then -4^(-x)
  else if (-2 < x ∧ x < -1) then -4^(x + 2)
  else if (1 < x ∧ x < 2) then 4^(x - 2)
  else 0

theorem problem_statement :
  (f (-5 / 2) + f 1) = -2 :=
sorry

end problem_statement_l217_217738


namespace probability_of_winning_quiz_l217_217816

theorem probability_of_winning_quiz :
  let n := 4 -- number of questions
  let choices := 3 -- number of choices per question
  let probability_correct := 1 / choices -- probability of answering correctly
  let probability_incorrect := 1 - probability_correct -- probability of answering incorrectly
  let probability_all_correct := probability_correct^n -- probability of getting all questions correct
  let probability_exactly_three_correct := 4 * probability_correct^3 * probability_incorrect -- probability of getting exactly 3 questions correct
  probability_all_correct + probability_exactly_three_correct = 1 / 9 :=
by
  sorry

end probability_of_winning_quiz_l217_217816


namespace son_age_l217_217131

theorem son_age:
  ∃ S M : ℕ, 
  (M = S + 20) ∧ 
  (M + 2 = 2 * (S + 2)) ∧ 
  (S = 18) := 
by
  sorry

end son_age_l217_217131


namespace edward_initial_money_l217_217143

variable (spent_books : ℕ) (spent_pens : ℕ) (money_left : ℕ)

theorem edward_initial_money (h_books : spent_books = 6) 
                             (h_pens : spent_pens = 16)
                             (h_left : money_left = 19) : 
                             spent_books + spent_pens + money_left = 41 := by
  sorry

end edward_initial_money_l217_217143


namespace gasoline_needed_l217_217774

theorem gasoline_needed (D : ℕ) 
    (fuel_efficiency : ℕ) 
    (fuel_efficiency_proof : fuel_efficiency = 20)
    (gallons_for_130km : ℕ) 
    (gallons_for_130km_proof : gallons_for_130km = 130 / 20) :
    (D : ℕ) / fuel_efficiency = (D : ℕ) / 20 :=
by
  -- The proof is omitted as per the instruction
  sorry

end gasoline_needed_l217_217774


namespace exists_integers_for_prime_l217_217637

theorem exists_integers_for_prime (p : ℕ) (hp : Nat.Prime p) : 
  ∃ x y z w : ℤ, x^2 + y^2 + z^2 = w * p ∧ 0 < w ∧ w < p :=
by 
  sorry

end exists_integers_for_prime_l217_217637


namespace price_arun_paid_l217_217289

theorem price_arun_paid 
  (original_price : ℝ)
  (standard_concession_rate : ℝ) 
  (additional_concession_rate : ℝ)
  (reduced_price : ℝ)
  (final_price : ℝ) 
  (h1 : original_price = 2000)
  (h2 : standard_concession_rate = 0.30)
  (h3 : additional_concession_rate = 0.20)
  (h4 : reduced_price = original_price * (1 - standard_concession_rate))
  (h5 : final_price = reduced_price * (1 - additional_concession_rate)) :
  final_price = 1120 :=
by
  sorry

end price_arun_paid_l217_217289


namespace terrier_to_poodle_grooming_ratio_l217_217782

-- Definitions and conditions
def time_to_groom_poodle : ℕ := 30
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8
def total_grooming_time : ℕ := 210
def time_to_groom_terrier := total_grooming_time - (num_poodles * time_to_groom_poodle) / num_terriers

-- Theorem statement
theorem terrier_to_poodle_grooming_ratio :
  time_to_groom_terrier / time_to_groom_poodle = 1 / 2 :=
by
  sorry

end terrier_to_poodle_grooming_ratio_l217_217782


namespace simplify_fraction_l217_217069

theorem simplify_fraction : (3 ^ 100 + 3 ^ 98) / (3 ^ 100 - 3 ^ 98) = 5 / 4 := 
by sorry

end simplify_fraction_l217_217069


namespace find_positive_integer_pair_l217_217359

noncomputable def quadratic_has_rational_solutions (d : ℤ) : Prop :=
  ∃ x : ℚ, 7 * x^2 + 13 * x + d = 0

theorem find_positive_integer_pair :
  ∃ (d1 d2 : ℕ), 
  d1 > 0 ∧ d2 > 0 ∧ 
  quadratic_has_rational_solutions d1 ∧ quadratic_has_rational_solutions d2 ∧ 
  d1 * d2 = 2 := 
sorry -- Proof left as an exercise

end find_positive_integer_pair_l217_217359


namespace rectangle_width_l217_217518

theorem rectangle_width (length : ℕ) (perimeter : ℕ) (h1 : length = 20) (h2 : perimeter = 70) :
  2 * (length + width) = perimeter → width = 15 :=
by
  intro h
  rw [h1, h2] at h
  -- Continue the steps to solve for width (can be simplified if not requesting the whole proof)
  sorry

end rectangle_width_l217_217518


namespace domain_of_f_2x_minus_1_l217_217867

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) (dom : ∀ x, f x ≠ 0 → (0 < x ∧ x < 1)) :
  ∀ x, f (2*x - 1) ≠ 0 → (1/2 < x ∧ x < 1) :=
by
  sorry

end domain_of_f_2x_minus_1_l217_217867


namespace distance_between_cities_l217_217224

variable (a b : Nat)

theorem distance_between_cities :
  (a = (10 * a + b) - (10 * b + a)) ∧ (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → 10 * a + b = 98 := by
  sorry

end distance_between_cities_l217_217224


namespace fraction_of_teeth_removed_l217_217735

theorem fraction_of_teeth_removed
  (total_teeth : ℕ)
  (initial_teeth : ℕ)
  (second_fraction : ℚ)
  (third_fraction : ℚ)
  (second_removed : ℕ)
  (third_removed : ℕ)
  (fourth_removed : ℕ)
  (total_removed : ℕ)
  (first_removed : ℕ)
  (fraction_first_removed : ℚ) :
  total_teeth = 32 →
  initial_teeth = 32 →
  second_fraction = 3 / 8 →
  third_fraction = 1 / 2 →
  second_removed = 12 →
  third_removed = 16 →
  fourth_removed = 4 →
  total_removed = 40 →
  first_removed + second_removed + third_removed + fourth_removed = total_removed →
  first_removed = 8 →
  fraction_first_removed = first_removed / initial_teeth →
  fraction_first_removed = 1 / 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end fraction_of_teeth_removed_l217_217735


namespace star_value_l217_217199

def star (a b : ℝ) : ℝ := a^3 + 3 * a^2 * b + 3 * a * b^2 + b^3

theorem star_value : star 3 2 = 125 :=
by
  sorry

end star_value_l217_217199


namespace eccentricities_proof_l217_217666

variable (e1 e2 m n c : ℝ)
variable (h1 : e1 = 2 * c / (m + n))
variable (h2 : e2 = 2 * c / (m - n))
variable (h3 : m ^ 2 + n ^ 2 = 4 * c ^ 2)

theorem eccentricities_proof :
  (e1 * e2) / (Real.sqrt (e1 ^ 2 + e2 ^ 2)) = (Real.sqrt 2) / 2 :=
by sorry

end eccentricities_proof_l217_217666


namespace fractions_product_equals_54_l217_217332

theorem fractions_product_equals_54 :
  (4 / 5) * (9 / 6) * (12 / 4) * (20 / 15) * (14 / 21) * (35 / 28) * (48 / 32) * (24 / 16) = 54 :=
by
  -- Add the proof here
  sorry

end fractions_product_equals_54_l217_217332


namespace problem_statement_l217_217914

theorem problem_statement (a b : ℝ) (h0 : 0 < b) (h1 : b < 1/2) (h2 : 1/2 < a) (h3 : a < 1) :
  (0 < a - b) ∧ (a - b < 1) ∧ (ab < a^2) ∧ (a - 1/b < b - 1/a) :=
by 
  sorry

end problem_statement_l217_217914


namespace pieces_per_box_l217_217543

theorem pieces_per_box 
  (a : ℕ) -- Adam bought 13 boxes of chocolate candy 
  (g : ℕ) -- Adam gave 7 boxes to his little brother 
  (p : ℕ) -- Adam still has 36 pieces 
  (n : ℕ) (b : ℕ) 
  (h₁ : a = 13) 
  (h₂ : g = 7) 
  (h₃ : p = 36) 
  (h₄ : n = a - g) 
  (h₅ : p = n * b) 
  : b = 6 :=
by 
  sorry

end pieces_per_box_l217_217543


namespace even_increasing_ordering_l217_217208

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

-- Theorem to prove
theorem even_increasing_ordering (h_even : is_even_function f) (h_increasing : is_increasing_on_pos f) : 
  f 1 < f (-2) ∧ f (-2) < f 3 :=
by
  sorry

end even_increasing_ordering_l217_217208


namespace city_growth_rate_order_l217_217232

theorem city_growth_rate_order 
  (Dover Eden Fairview : Type) 
  (highest lowest : Type)
  (h1 : Dover = highest → ¬(Eden = highest) ∧ (Fairview = lowest))
  (h2 : ¬(Dover = highest) ∧ Eden = highest ∧ Fairview = lowest → Eden = highest ∧ Dover = lowest ∧ Fairview = highest)
  (h3 : ¬(Fairview = lowest) → ¬(Eden = highest) ∧ ¬(Dover = highest)) : 
  Eden = highest ∧ Dover = lowest ∧ Fairview = highest ∧ Eden ≠ lowest :=
by
  sorry

end city_growth_rate_order_l217_217232


namespace tax_free_amount_is_600_l217_217000

variable (X : ℝ) -- X is the tax-free amount

-- Given conditions
variable (total_value : ℝ := 1720)
variable (tax_paid : ℝ := 89.6)
variable (tax_rate : ℝ := 0.08)

-- Proof problem
theorem tax_free_amount_is_600
  (h1 : 0.08 * (total_value - X) = tax_paid) :
  X = 600 :=
by
  sorry

end tax_free_amount_is_600_l217_217000


namespace provisions_last_for_girls_l217_217796

theorem provisions_last_for_girls (P : ℝ) (G : ℝ) (h1 : P / (50 * G) = P / (250 * (G + 20))) : G = 25 := 
by
  sorry

end provisions_last_for_girls_l217_217796


namespace extra_discount_percentage_l217_217619

theorem extra_discount_percentage 
  (initial_price : ℝ)
  (first_discount : ℝ)
  (new_price : ℝ)
  (final_price : ℝ)
  (extra_discount_amount : ℝ)
  (x : ℝ)
  (discount_formula : x = (extra_discount_amount * 100) / new_price) :
  initial_price = 50 ∧ 
  first_discount = 2.08 ∧ 
  new_price = 47.92 ∧ 
  final_price = 46 ∧ 
  extra_discount_amount = new_price - final_price → 
  x = 4 :=
by
  -- The proof will go here
  sorry

end extra_discount_percentage_l217_217619


namespace base_4_last_digit_of_389_l217_217152

theorem base_4_last_digit_of_389 : (389 % 4) = 1 :=
by {
  sorry
}

end base_4_last_digit_of_389_l217_217152


namespace positive_integer_solutions_of_inequality_system_l217_217481

theorem positive_integer_solutions_of_inequality_system :
  {x : ℤ | 2 * (x - 1) < x + 1 ∧ 1 - (2 * x + 5) / 3 ≤ x ∧ x > 0} = {1, 2} :=
by
  sorry

end positive_integer_solutions_of_inequality_system_l217_217481


namespace martian_year_length_ratio_l217_217858

theorem martian_year_length_ratio :
  let EarthDay := 24 -- hours
  let MarsDay := EarthDay + 2 / 3 -- hours (since 40 minutes is 2/3 of an hour)
  let MartianYearDays := 668
  let EarthYearDays := 365.25
  (MartianYearDays * MarsDay) / EarthYearDays = 1.88 := by
{
  sorry
}

end martian_year_length_ratio_l217_217858


namespace min_val_of_q_l217_217777

theorem min_val_of_q (p q : ℕ) (h1 : 72 / 487 < p / q) (h2 : p / q < 18 / 121) : 
  ∃ p q : ℕ, (72 / 487 < p / q) ∧ (p / q < 18 / 121) ∧ q = 27 :=
sorry

end min_val_of_q_l217_217777


namespace geometric_sequence_from_second_term_l217_217049

open Nat

-- Define the sequence S_n
def S (n : ℕ) : ℕ := 
  match n with
  | 0 => 0 -- to handle the 0th term which is typically not used here
  | 1 => 1
  | 2 => 2
  | n + 3 => 3 * S (n + 2) - 2 * S (n + 1) -- given recurrence relation

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 
  match n with
  | 0 => 0 -- Define a_0 as 0 since it's not used in the problem
  | 1 => 1 -- a1
  | n + 2 => S (n + 2) - S (n + 1) -- a_n = S_n - S_(n-1)

theorem geometric_sequence_from_second_term :
  ∀ n ≥ 2, a (n + 1) = 2 * a n := by
  -- Proof step not provided
  sorry

end geometric_sequence_from_second_term_l217_217049


namespace movie_ticket_cost_l217_217900

variable (x : ℝ)
variable (h1 : x * 2 + 1.59 + 13.95 = 36.78)

theorem movie_ticket_cost : x = 10.62 :=
by
  sorry

end movie_ticket_cost_l217_217900


namespace domain_of_f_l217_217015

open Real

noncomputable def f (x : ℝ) : ℝ := log ((2 - x) / (2 + x))

theorem domain_of_f : ∀ x : ℝ, (2 - x) / (2 + x) > 0 ∧ 2 + x ≠ 0 ↔ -2 < x ∧ x < 2 :=
by
  intro x
  sorry

end domain_of_f_l217_217015


namespace truck_wheels_l217_217801

theorem truck_wheels (t x : ℝ) (wheels_front : ℕ) (wheels_other : ℕ) :
  (t = 1.50 + 1.50 * (x - 2)) → (t = 6) → (wheels_front = 2) → (wheels_other = 4) → x = 5 → 
  (wheels_front + wheels_other * (x - 1) = 18) :=
by
  intros h1 h2 h3 h4 h5
  rw [h5] at *
  sorry

end truck_wheels_l217_217801


namespace agency_comparison_l217_217439

variable (days m : ℝ)

theorem agency_comparison (h : 20.25 * days + 0.14 * m < 18.25 * days + 0.22 * m) : m > 25 * days :=
by
  sorry

end agency_comparison_l217_217439


namespace shaded_square_area_l217_217249

theorem shaded_square_area (a b s : ℝ) (h : a * b = 40) :
  ∃ s, s^2 = 2500 / 441 :=
by
  sorry

end shaded_square_area_l217_217249


namespace average_height_31_students_l217_217770

theorem average_height_31_students (avg1 avg2 : ℝ) (n1 n2 : ℕ) (h1 : avg1 = 20) (h2 : avg2 = 20) (h3 : n1 = 20) (h4 : n2 = 11) : ((avg1 * n1 + avg2 * n2) / (n1 + n2)) = 20 :=
by
  sorry

end average_height_31_students_l217_217770


namespace initial_provisions_last_l217_217906

theorem initial_provisions_last (x : ℕ) (h : 2000 * (x - 20) = 4000 * 10) : x = 40 :=
by sorry

end initial_provisions_last_l217_217906


namespace no_linear_term_in_product_l217_217780

theorem no_linear_term_in_product (a : ℝ) (h : ∀ x : ℝ, (x + 4) * (x + a) - x^2 - 4 * a = 0) : a = -4 :=
sorry

end no_linear_term_in_product_l217_217780


namespace work_done_on_gas_in_process_1_2_l217_217382

variables (V₁ V₂ V₃ V₄ A₁₂ A₃₄ T n R : ℝ)

-- Both processes 1-2 and 3-4 are isothermal.
def is_isothermal_process := true -- Placeholder

-- Volumes relationship: for any given pressure, the volume in process 1-2 is exactly twice the volume in process 3-4.
def volumes_relation (V₁ V₂ V₃ V₄ : ℝ) : Prop :=
  V₁ = 2 * V₃ ∧ V₂ = 2 * V₄

-- Work done on a gas during an isothermal process can be represented as: A = 2 * A₃₄
def work_relation (A₁₂ A₃₄ : ℝ) : Prop :=
  A₁₂ = 2 * A₃₄

theorem work_done_on_gas_in_process_1_2
  (h_iso : is_isothermal_process)
  (h_vol : volumes_relation V₁ V₂ V₃ V₄)
  (h_work : work_relation A₁₂ A₃₄) :
  A₁₂ = 2 * A₃₄ :=
by 
  sorry

end work_done_on_gas_in_process_1_2_l217_217382


namespace total_spent_l217_217856

def spending (A B C : ℝ) : Prop :=
  (A = (13 / 10) * B) ∧
  (C = (4 / 5) * B) ∧
  (A = C + 15)

theorem total_spent (A B C : ℝ) (h : spending A B C) : A + B + C = 93 :=
by
  sorry

end total_spent_l217_217856


namespace five_digit_palindromes_count_l217_217970

theorem five_digit_palindromes_count : 
  ∃ (a b c : Fin 10), (a ≠ 0) ∧ (∃ (count : Nat), count = 9 * 10 * 10 ∧ count = 900) :=
by
  sorry

end five_digit_palindromes_count_l217_217970


namespace pos_real_ineq_l217_217379

theorem pos_real_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c)/3) :=
by 
  sorry

end pos_real_ineq_l217_217379


namespace solve_inequality_l217_217471

theorem solve_inequality (a : ℝ) : 
  (if a = 0 ∨ a = 1 then { x : ℝ | false }
   else if a < 0 ∨ a > 1 then { x : ℝ | a < x ∧ x < a^2 }
   else if 0 < a ∧ a < 1 then { x : ℝ | a^2 < x ∧ x < a }
   else ∅) = 
  { x : ℝ | (x - a) / (x - a^2) < 0 } :=
by sorry

end solve_inequality_l217_217471


namespace non_rent_extra_expenses_is_3000_l217_217733

-- Define the constants
def cost_parts : ℕ := 800
def markup : ℝ := 1.4
def num_computers : ℕ := 60
def rent : ℕ := 5000
def profit : ℕ := 11200

-- Calculate the selling price per computer
def selling_price : ℝ := cost_parts * markup

-- Calculate the total revenue from selling 60 computers
def total_revenue : ℝ := selling_price * num_computers

-- Calculate the total cost of components for 60 computers
def total_cost_components : ℕ := cost_parts * num_computers

-- Calculate the total expenses
def total_expenses : ℝ := total_revenue - profit

-- Define the non-rent extra expenses
def non_rent_extra_expenses : ℝ := total_expenses - rent - total_cost_components

-- Prove that the non-rent extra expenses equal to $3000
theorem non_rent_extra_expenses_is_3000 : non_rent_extra_expenses = 3000 := sorry

end non_rent_extra_expenses_is_3000_l217_217733


namespace remainder_when_x_plus_4uy_div_y_l217_217498

theorem remainder_when_x_plus_4uy_div_y (x y u v : ℕ) (h₀: x = u * y + v) (h₁: 0 ≤ v) (h₂: v < y) : 
  ((x + 4 * u * y) % y) = v := 
by 
  sorry

end remainder_when_x_plus_4uy_div_y_l217_217498


namespace ratio_of_capital_l217_217302

variable (C A B : ℝ)
variable (h1 : B = 4 * C)
variable (h2 : B / (A + 5 * C) = 6000 / 16500)

theorem ratio_of_capital : A / B = 17 / 4 :=
by
  sorry

end ratio_of_capital_l217_217302


namespace function_form_l217_217535

noncomputable def f : ℕ → ℕ := sorry

theorem function_form (c d a : ℕ) (h1 : c > 1) (h2 : a - c > 1)
  (hf : ∀ n : ℕ, f n + f (n + 1) = f (n + 2) + f (n + 3) - 168) :
  (∀ n : ℕ, f (2 * n) = c + n * d) ∧ (∀ n : ℕ, f (2 * n + 1) = (168 - d) * n + a - c) :=
sorry

end function_form_l217_217535


namespace purely_imaginary_iff_l217_217151

theorem purely_imaginary_iff (a : ℝ) :
  (a^2 - a - 2 = 0 ∧ (|a - 1| - 1 ≠ 0)) ↔ a = -1 :=
by
  sorry

end purely_imaginary_iff_l217_217151


namespace number_of_boys_l217_217121

noncomputable def numGirls : Nat := 46
noncomputable def numGroups : Nat := 8
noncomputable def groupSize : Nat := 9
noncomputable def totalMembers : Nat := numGroups * groupSize
noncomputable def numBoys : Nat := totalMembers - numGirls

theorem number_of_boys :
  numBoys = 26 := by
  sorry

end number_of_boys_l217_217121


namespace find_smaller_number_l217_217378

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 18) (h2 : a * b = 45) : a = 3 ∨ b = 3 := 
by 
  -- Proof would go here
  sorry

end find_smaller_number_l217_217378


namespace family_total_weight_gain_l217_217155

def orlando_gain : ℕ := 5
def jose_gain : ℕ := 2 * orlando_gain + 2
def fernando_gain : ℕ := (jose_gain / 2) - 3
def total_weight_gain : ℕ := orlando_gain + jose_gain + fernando_gain

theorem family_total_weight_gain : total_weight_gain = 20 := by
  -- proof omitted
  sorry

end family_total_weight_gain_l217_217155


namespace common_difference_value_l217_217534

-- Define the arithmetic sequence and the sum of the first n terms
def sum_of_arithmetic_sequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

-- Define the given condition in terms of the arithmetic sequence
def given_condition (a1 d : ℚ) : Prop :=
  (sum_of_arithmetic_sequence a1 d 2017) / 2017 - (sum_of_arithmetic_sequence a1 d 17) / 17 = 100

-- Prove the common difference d is 1/10 given the condition
theorem common_difference_value (a1 d : ℚ) :
  given_condition a1 d → d = 1/10 :=
by
  sorry

end common_difference_value_l217_217534


namespace max_possible_percent_error_in_garden_area_l217_217703

open Real

theorem max_possible_percent_error_in_garden_area :
  ∃ (error_max : ℝ), error_max = 21 :=
by
  -- Given conditions
  let accurate_diameter := 30
  let max_error_percent := 10

  -- Defining lower and upper bounds for the diameter
  let lower_diameter := accurate_diameter - accurate_diameter * (max_error_percent / 100)
  let upper_diameter := accurate_diameter + accurate_diameter * (max_error_percent / 100)

  -- Calculating the exact and potential extreme areas
  let exact_area := π * (accurate_diameter / 2) ^ 2
  let lower_area := π * (lower_diameter / 2) ^ 2
  let upper_area := π * (upper_diameter / 2) ^ 2

  -- Calculating the percent errors
  let lower_error_percent := ((exact_area - lower_area) / exact_area) * 100
  let upper_error_percent := ((upper_area - exact_area) / exact_area) * 100

  -- We need to show the maximum error is 21%
  use upper_error_percent -- which should be 21% according to the problem statement
  sorry -- proof goes here

end max_possible_percent_error_in_garden_area_l217_217703


namespace same_terminal_side_angle_l217_217234

theorem same_terminal_side_angle (k : ℤ) : 
  0 ≤ (k * 360 - 35) ∧ (k * 360 - 35) < 360 → (k * 360 - 35) = 325 :=
by
  sorry

end same_terminal_side_angle_l217_217234


namespace sphere_in_cone_volume_l217_217040

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem sphere_in_cone_volume :
  let d := 12
  let θ := 45
  let r := 3 * Real.sqrt 2
  let V := volume_of_sphere r
  d = 12 → θ = 45 → V = 72 * Real.sqrt 2 * Real.pi := by
  intros h1 h2
  sorry

end sphere_in_cone_volume_l217_217040


namespace math_problem_l217_217173

theorem math_problem : 3 * 13 + 3 * 14 + 3 * 17 + 11 = 143 := by
  sorry

end math_problem_l217_217173


namespace number_is_minus_72_l217_217554

noncomputable def find_number (x : ℝ) : Prop :=
  0.833 * x = -60

theorem number_is_minus_72 : ∃ x : ℝ, find_number x ∧ x = -72 :=
by
  sorry

end number_is_minus_72_l217_217554


namespace james_drinks_per_day_l217_217101

-- condition: James buys 5 packs of sodas, each contains 12 sodas
def num_packs : Nat := 5
def sodas_per_pack : Nat := 12
def sodas_bought : Nat := num_packs * sodas_per_pack

-- condition: James already had 10 sodas
def sodas_already_had : Nat := 10

-- condition: James finishes all the sodas in 1 week (7 days)
def days_in_week : Nat := 7

-- total sodas
def total_sodas : Nat := sodas_bought + sodas_already_had

-- number of sodas james drinks per day
def sodas_per_day : Nat := 10

-- proof problem
theorem james_drinks_per_day : (total_sodas / days_in_week) = sodas_per_day :=
  sorry

end james_drinks_per_day_l217_217101


namespace son_working_alone_l217_217277

theorem son_working_alone (M S : ℝ) (h1: M = 1 / 5) (h2: M + S = 1 / 3) : 1 / S = 7.5 :=
  by
  sorry

end son_working_alone_l217_217277


namespace least_possible_value_of_squares_l217_217883

theorem least_possible_value_of_squares (a b x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h1 : 15 * a + 16 * b = x^2) (h2 : 16 * a - 15 * b = y^2) : 
  ∃ (x : ℕ) (y : ℕ), min (x^2) (y^2) = 231361 := 
sorry

end least_possible_value_of_squares_l217_217883


namespace find_number_l217_217206

theorem find_number (n x : ℕ) (h1 : n * (x - 1) = 21) (h2 : x = 4) : n = 7 :=
by
  sorry

end find_number_l217_217206


namespace circle_line_distance_difference_l217_217340

/-- We define the given circle and line and prove the difference between maximum and minimum distances
    from any point on the circle to the line is 5√2. -/
theorem circle_line_distance_difference :
  (∀ (x y : ℝ), x^2 + y^2 - 4*x - 4*y - 10 = 0) →
  (∀ (x y : ℝ), x + y - 8 = 0) →
  ∃ (d : ℝ), d = 5 * Real.sqrt 2 :=
by
  sorry

end circle_line_distance_difference_l217_217340


namespace area_of_triangle_DEF_l217_217091

theorem area_of_triangle_DEF :
  let s := 2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let radius := s
  let distance_between_centers := 2 * radius
  let side_of_triangle_DEF := distance_between_centers
  let triangle_area := (Real.sqrt 3 / 4) * side_of_triangle_DEF^2
  triangle_area = 4 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_DEF_l217_217091


namespace value_of_expression_l217_217682

theorem value_of_expression (a b : ℝ) (h : 2 * a + 4 * b = 3) : 4 * a + 8 * b - 2 = 4 := 
by 
  sorry

end value_of_expression_l217_217682


namespace paint_quantity_l217_217580

variable (totalPaint : ℕ) (blueRatio greenRatio whiteRatio : ℕ)

theorem paint_quantity 
  (h_total_paint : totalPaint = 45)
  (h_ratio_blue : blueRatio = 5)
  (h_ratio_green : greenRatio = 3)
  (h_ratio_white : whiteRatio = 7) :
  let totalRatio := blueRatio + greenRatio + whiteRatio
  let partQuantity := totalPaint / totalRatio
  let bluePaint := blueRatio * partQuantity
  let greenPaint := greenRatio * partQuantity
  let whitePaint := whiteRatio * partQuantity
  bluePaint = 15 ∧ greenPaint = 9 ∧ whitePaint = 21 :=
by
  sorry

end paint_quantity_l217_217580


namespace ellipse_equation_l217_217943

-- Definitions of the tangents given as conditions
def tangent1 (x y : ℝ) : Prop := 4 * x + 5 * y = 25
def tangent2 (x y : ℝ) : Prop := 9 * x + 20 * y = 75

-- The statement we need to prove
theorem ellipse_equation :
  (∀ (x y : ℝ), tangent1 x y → tangent2 x y → 
  (∃ (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0), a = 5 ∧ b = 3 ∧ 
  (x^2 / a^2 + y^2 / b^2 = 1))) :=
sorry

end ellipse_equation_l217_217943


namespace expected_value_of_fair_8_sided_die_l217_217850

-- Define the outcomes of the fair 8-sided die
def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define the probability of each outcome for a fair die
def prob (n : ℕ) : ℚ := 1 / 8

-- Calculate the expected value of the outcomes
noncomputable def expected_value : ℚ :=
  (outcomes.map (λ x => prob x * x)).sum

-- State the theorem that the expected value is 4.5
theorem expected_value_of_fair_8_sided_die : expected_value = 4.5 :=
  sorry

end expected_value_of_fair_8_sided_die_l217_217850


namespace positive_integer_solutions_l217_217967

theorem positive_integer_solutions (n x y z t : ℕ) (h_n : n > 0) (h_n_neq_1 : n ≠ 1) (h_x : x > 0) (h_y : y > 0) (h_z : z > 0) (h_t : t > 0) :
  (n ^ x ∣ n ^ y + n ^ z ↔ n ^ x = n ^ t) →
  ((n = 2 ∧ y = x ∧ z = x + 1 ∧ t = x + 2) ∨ (n = 3 ∧ y = x ∧ z = x ∧ t = x + 1)) :=
by
  sorry

end positive_integer_solutions_l217_217967


namespace sum_of_factors_of_30_l217_217670

/--
Given the positive integer factors of 30, prove that their sum is 72.
-/
theorem sum_of_factors_of_30 : 
  (1 + 2 + 3 + 5 + 6 + 10 + 15 + 30) = 72 := 
by 
  sorry

end sum_of_factors_of_30_l217_217670


namespace tissues_used_l217_217583

-- Define the conditions
def box_tissues : ℕ := 160
def boxes_bought : ℕ := 3
def tissues_left : ℕ := 270

-- Define the theorem that needs to be proven
theorem tissues_used (total_tissues := boxes_bought * box_tissues) : total_tissues - tissues_left = 210 := by
  sorry

end tissues_used_l217_217583


namespace polar_coordinates_full_circle_l217_217027

theorem polar_coordinates_full_circle :
  ∀ (r : ℝ) (θ : ℝ), (r = 3 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) → (r = 3 ∧ ∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi ↔ r = 3) :=
by
  intros r θ h
  sorry

end polar_coordinates_full_circle_l217_217027


namespace find_m_l217_217146

def triangle (x y : ℤ) := x * y + x + y

theorem find_m (m : ℤ) (h : triangle 2 m = -16) : m = -6 :=
by
  sorry

end find_m_l217_217146


namespace range_of_a_l217_217427

def p (a m : ℝ) : Prop := m^2 + 12 * a^2 < 7 * a * m ∧ a > 0
def q (m : ℝ) : Prop := 1 < m ∧ m < 3 / 2

theorem range_of_a (a : ℝ) :
  (∀ m : ℝ, p a m → q m) → 
  (∃ (a_lower a_upper : ℝ), a_lower ≤ a ∧ a ≤ a_upper ∧ a_lower = 1 / 3 ∧ a_upper = 3 / 8) :=
sorry

end range_of_a_l217_217427


namespace solve_equation_l217_217683

noncomputable def f (x : ℝ) := (1 / (x^2 + 17 * x + 20)) + (1 / (x^2 + 12 * x + 20)) + (1 / (x^2 - 15 * x + 20))

theorem solve_equation :
  {x : ℝ | f x = 0} = {-1, -4, -5, -20} :=
by
  sorry

end solve_equation_l217_217683


namespace second_markdown_percentage_l217_217228

theorem second_markdown_percentage (P : ℝ) (h1 : P > 0)
    (h2 : ∃ x : ℝ, x = 0.50 * P) -- First markdown
    (h3 : ∃ y : ℝ, y = 0.45 * P) -- Final price
    : ∃ X : ℝ, X = 10 := 
sorry

end second_markdown_percentage_l217_217228


namespace combined_weight_loss_l217_217573

theorem combined_weight_loss (a_weekly_loss : ℝ) (a_weeks : ℕ) (x_weekly_loss : ℝ) (x_weeks : ℕ)
  (h1 : a_weekly_loss = 1.5) (h2 : a_weeks = 10) (h3 : x_weekly_loss = 2.5) (h4 : x_weeks = 8) :
  a_weekly_loss * a_weeks + x_weekly_loss * x_weeks = 35 := 
by
  -- We will not provide the proof body; the goal is to ensure the statement compiles.
  sorry

end combined_weight_loss_l217_217573


namespace polygon_sides_diagonals_l217_217313

theorem polygon_sides_diagonals (n : ℕ) 
  (h1 : 4 * (n * (n - 3)) = 14 * n)
  (h2 : (n + (n * (n - 3)) / 2) % 2 = 0)
  (h3 : n + n * (n - 3) / 2 > 50) : n = 12 := 
by 
  sorry

end polygon_sides_diagonals_l217_217313


namespace tax_refund_l217_217616

-- Definitions based on the problem conditions
def monthly_salary : ℕ := 9000
def treatment_cost : ℕ := 100000
def medication_cost : ℕ := 20000
def tax_rate : ℚ := 0.13

-- Annual salary calculation
def annual_salary := monthly_salary * 12

-- Total spending on treatment and medications
def total_spending := treatment_cost + medication_cost

-- Possible tax refund based on total spending
def possible_tax_refund := total_spending * tax_rate

-- Income tax paid on the annual salary
def income_tax_paid := annual_salary * tax_rate

-- Prove statement that the actual tax refund is equal to income tax paid
theorem tax_refund : income_tax_paid = 14040 := by
  sorry

end tax_refund_l217_217616


namespace find_x_l217_217185

def star (p q : Int × Int) : Int × Int :=
  (p.1 + q.2, p.2 - q.1)

theorem find_x : ∀ (x y : Int), star (x, y) (4, 2) = (5, 4) → x = 3 :=
by
  intros x y h
  -- The statement is correct, just add a placeholder for the proof
  sorry

end find_x_l217_217185


namespace brokerage_percentage_l217_217007

theorem brokerage_percentage
  (f : ℝ) (d : ℝ) (c : ℝ) 
  (hf : f = 100)
  (hd : d = 0.08)
  (hc : c = 92.2)
  (h_disc_price : f - f * d = 92) :
  (c - (f - f * d)) / f * 100 = 0.2 := 
by
  sorry

end brokerage_percentage_l217_217007


namespace geometric_sequence_eighth_term_l217_217776

theorem geometric_sequence_eighth_term (a r : ℝ) (h1 : a * r ^ 3 = 12) (h2 : a * r ^ 11 = 3) : 
  a * r ^ 7 = 6 * Real.sqrt 2 :=
sorry

end geometric_sequence_eighth_term_l217_217776


namespace find_m_l217_217171

theorem find_m {m : ℝ} :
  (∃ x y : ℝ, y = x + 1 ∧ y = -x ∧ y = mx + 3) → m = 5 :=
by
  sorry

end find_m_l217_217171


namespace excircle_opposite_side_b_l217_217763

-- Definition of the terms and assumptions
variables {a b c : ℝ} -- sides of the triangle
variables {r r1 : ℝ}  -- radii of the circles

-- Given conditions
def touches_side_c_and_extensions_of_a_b (r : ℝ) (a b c : ℝ) : Prop :=
  r = (a + b + c) / 2

-- The goal to be proved
theorem excircle_opposite_side_b (a b c : ℝ) (r1 : ℝ) (h1 : touches_side_c_and_extensions_of_a_b r a b c) :
  r1 = (a + c - b) / 2 := 
by
  sorry

end excircle_opposite_side_b_l217_217763


namespace percentage_of_part_over_whole_l217_217946

theorem percentage_of_part_over_whole (Part Whole : ℕ) (h1 : Part = 120) (h2 : Whole = 50) :
  (Part / Whole : ℚ) * 100 = 240 := by
  sorry

end percentage_of_part_over_whole_l217_217946


namespace M_subset_P_l217_217408

universe u

-- Definitions of the sets
def U : Set ℝ := {x : ℝ | True}
def M : Set ℝ := {x : ℝ | x > 1}
def P : Set ℝ := {x : ℝ | x^2 > 1}

-- Proof statement
theorem M_subset_P : M ⊆ P := by
  sorry

end M_subset_P_l217_217408


namespace frac_equiv_l217_217429

-- Define the given values of x and y.
def x : ℚ := 2 / 7
def y : ℚ := 8 / 11

-- Define the statement to prove.
theorem frac_equiv : (7 * x + 11 * y) / (77 * x * y) = 5 / 8 :=
by
  -- The proof will go here (use 'sorry' for now)
  sorry

end frac_equiv_l217_217429


namespace correct_mean_251_l217_217119

theorem correct_mean_251
  (n : ℕ) (incorrect_mean : ℕ) (wrong_val : ℕ) (correct_val : ℕ)
  (h1 : n = 30) (h2 : incorrect_mean = 250) (h3 : wrong_val = 135) (h4 : correct_val = 165) :
  ((incorrect_mean * n + (correct_val - wrong_val)) / n) = 251 :=
by
  sorry

end correct_mean_251_l217_217119


namespace calculate_expression_l217_217559

theorem calculate_expression :
  |(-Real.sqrt 3)| - (1/3)^(-1/2 : ℝ) + 2 / (Real.sqrt 3 - 1) - 12^(1/2 : ℝ) = 1 - Real.sqrt 3 :=
by
  sorry

end calculate_expression_l217_217559


namespace problem_solution_l217_217588

theorem problem_solution : (324^2 - 300^2) / 24 = 624 :=
by 
  -- The proof will be inserted here.
  sorry

end problem_solution_l217_217588


namespace missing_score_and_variance_l217_217192

theorem missing_score_and_variance (score_A score_B score_D score_E : ℕ) (avg_score : ℕ)
  (h_scores : score_A = 81 ∧ score_B = 79 ∧ score_D = 80 ∧ score_E = 82)
  (h_avg : avg_score = 80):
  ∃ (score_C variance : ℕ), score_C = 78 ∧ variance = 2 := by
  sorry

end missing_score_and_variance_l217_217192


namespace intersection_M_N_l217_217352

open Set

-- Definitions from conditions
def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {x | x < 1}

-- Proof statement
theorem intersection_M_N : M ∩ N = {-1} := 
by sorry

end intersection_M_N_l217_217352


namespace max_side_range_of_triangle_l217_217253

-- Define the requirement on the sides a and b
def side_condition (a b : ℝ) : Prop :=
  |a - 3| + (b - 7)^2 = 0

-- Prove the range of side c
theorem max_side_range_of_triangle (a b c : ℝ) (h : side_condition a b) (hc : c = max a (max b c)) :
  7 ≤ c ∧ c < 10 :=
sorry

end max_side_range_of_triangle_l217_217253


namespace compound_interest_calculation_l217_217334

-- Define the variables used in the problem
def principal : ℝ := 8000
def annual_rate : ℝ := 0.05
def compound_frequency : ℕ := 1
def final_amount : ℝ := 9261
def years : ℝ := 3

-- Statement we need to prove
theorem compound_interest_calculation :
  final_amount = principal * (1 + annual_rate / compound_frequency) ^ (compound_frequency * years) :=
by 
  sorry

end compound_interest_calculation_l217_217334


namespace profit_without_discount_l217_217762

noncomputable def cost_price : ℝ := 100
noncomputable def discount_percentage : ℝ := 0.05
noncomputable def profit_with_discount_percentage : ℝ := 0.387
noncomputable def selling_price_with_discount : ℝ := cost_price * (1 + profit_with_discount_percentage)

noncomputable def profit_without_discount_percentage : ℝ :=
  let selling_price_without_discount := selling_price_with_discount / (1 - discount_percentage)
  ((selling_price_without_discount - cost_price) / cost_price) * 100

theorem profit_without_discount :
  profit_without_discount_percentage = 45.635 := by
  sorry

end profit_without_discount_l217_217762


namespace peter_takes_last_stone_l217_217415

theorem peter_takes_last_stone (n : ℕ) (h : ∀ p, Nat.Prime p → p < n) :
  ∃ P, ∀ stones: ℕ, stones > n^2 → (∃ k : ℕ, 
  ((k = 1 ∨ (∃ p : ℕ, Nat.Prime p ∧ p < n ∧ k = p) ∨ (∃ m : ℕ, k = m * n)) ∧
  stones ≥ k ∧ stones - k > n^2) →
  P = stones - k) := 
sorry

end peter_takes_last_stone_l217_217415


namespace negation_of_proposition_l217_217689

open Nat 

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, n > 0 ∧ n^2 > 2^n) ↔ ∀ n : ℕ, n > 0 → n^2 ≤ 2^n :=
by
  sorry

end negation_of_proposition_l217_217689


namespace find_a_l217_217165

noncomputable def l1 (a : ℝ) (x y : ℝ) : ℝ := a * x + (a + 1) * y + 1
noncomputable def l2 (a : ℝ) (x y : ℝ) : ℝ := x + a * y + 2

def perp_lines (a : ℝ) : Prop :=
  let m1 := -a
  let m2 := -1 / a
  m1 * m2 = -1

theorem find_a (a : ℝ) : (perp_lines a) ↔ (a = 0 ∨ a = -2) := 
sorry

end find_a_l217_217165


namespace rectangle_perimeter_l217_217917

variable (a b : ℝ)
variable (h1 : a * b = 24)
variable (h2 : a^2 + b^2 = 121)

theorem rectangle_perimeter : 2 * (a + b) = 26 := 
by
  sorry

end rectangle_perimeter_l217_217917


namespace Alice_has_3_more_dimes_than_quarters_l217_217214

-- Definitions of the conditions given in the problem
variable (n d : ℕ) -- number of 5-cent and 10-cent coins
def q : ℕ := 10
def total_coins : ℕ := 30
def total_value : ℕ := 435
def extra_dimes : ℕ := 6

-- Conditions translated to Lean
axiom total_coin_count : n + d + q = total_coins
axiom total_value_count : 5 * n + 10 * d + 25 * q = total_value
axiom dime_difference : d = n + extra_dimes

-- The theorem that needs to be proven: Alice has 3 more 10-cent coins than 25-cent coins.
theorem Alice_has_3_more_dimes_than_quarters :
  d - q = 3 :=
sorry

end Alice_has_3_more_dimes_than_quarters_l217_217214


namespace range_k_l217_217623

theorem range_k (k : ℝ) :
  (∀ x : ℝ, (3/8 - k*x - 2*k*x^2) ≥ 0) ↔ (-3 ≤ k ∧ k ≤ 0) :=
sorry

end range_k_l217_217623


namespace value_of_a_l217_217275

theorem value_of_a (x y a : ℝ) (h1 : x - 2 * y = a - 6) (h2 : 2 * x + 5 * y = 2 * a) (h3 : x + y = 9) : a = 11 := 
by
  sorry

end value_of_a_l217_217275


namespace quadratic_complete_square_l217_217739

open Real

theorem quadratic_complete_square (d e : ℝ) :
  (∀ x, x^2 - 24 * x + 50 = (x + d)^2 + e) → d + e = -106 :=
by
  intros h
  have h_eq := h 12
  sorry

end quadratic_complete_square_l217_217739


namespace tori_needs_more_correct_answers_l217_217210

theorem tori_needs_more_correct_answers :
  let total_questions := 80
  let arithmetic_questions := 20
  let algebra_questions := 25
  let geometry_questions := 35
  let arithmetic_correct := 0.60 * arithmetic_questions
  let algebra_correct := Float.round (0.50 * algebra_questions)
  let geometry_correct := Float.round (0.70 * geometry_questions)
  let correct_answers := arithmetic_correct + algebra_correct + geometry_correct
  let passing_percentage := 0.65
  let required_correct := passing_percentage * total_questions
-- assertion
  required_correct - correct_answers = 2 := 
by 
  sorry

end tori_needs_more_correct_answers_l217_217210


namespace origami_papers_per_cousin_l217_217082

/-- Haley has 48 origami papers and 6 cousins. Each cousin should receive the same number of papers. -/
theorem origami_papers_per_cousin : ∀ (total_papers : ℕ) (number_of_cousins : ℕ),
  total_papers = 48 → number_of_cousins = 6 → total_papers / number_of_cousins = 8 :=
by
  intros total_papers number_of_cousins
  sorry

end origami_papers_per_cousin_l217_217082


namespace undefined_values_l217_217836

-- Define the expression to check undefined values
noncomputable def is_undefined (x : ℝ) : Prop :=
  x^3 - 9 * x = 0

-- Statement: For which real values of x is the expression undefined?
theorem undefined_values (x : ℝ) : is_undefined x ↔ x = 0 ∨ x = -3 ∨ x = 3 :=
sorry

end undefined_values_l217_217836


namespace least_number_added_1054_l217_217653

theorem least_number_added_1054 (x d: ℕ) (h_cond: 1054 + x = 1058) (h_div: d = 2) : 1058 % d = 0 :=
by
  sorry

end least_number_added_1054_l217_217653


namespace fifteenth_term_is_correct_l217_217167

-- Define the initial conditions of the arithmetic sequence
def firstTerm : ℕ := 4
def secondTerm : ℕ := 9

-- Calculate the common difference
def commonDifference : ℕ := secondTerm - firstTerm

-- Define the nth term formula of the arithmetic sequence
def nthTerm (a d n : ℕ) : ℕ := a + (n - 1) * d

-- The main statement: proving that the 15th term of the given sequence is 74
theorem fifteenth_term_is_correct : nthTerm firstTerm commonDifference 15 = 74 :=
by
  sorry

end fifteenth_term_is_correct_l217_217167


namespace statement_B_statement_C_statement_D_l217_217555

-- Statement B
theorem statement_B (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a^3 * c < b^3 * c :=
sorry

-- Statement C
theorem statement_C (a b c : ℝ) (h1 : c > a) (h2 : a > b) (h3 : b > 0) : (a / (c - a)) > (b / (c - b)) :=
sorry

-- Statement D
theorem statement_D (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a > 0 ∧ b < 0 :=
sorry

end statement_B_statement_C_statement_D_l217_217555


namespace systematic_sampling_eighth_group_l217_217423

theorem systematic_sampling_eighth_group
  (total_employees : ℕ)
  (target_sample : ℕ)
  (third_group_value : ℕ)
  (group_count : ℕ)
  (common_difference : ℕ)
  (eighth_group_value : ℕ) :
  total_employees = 840 →
  target_sample = 42 →
  third_group_value = 44 →
  group_count = total_employees / target_sample →
  common_difference = group_count →
  eighth_group_value = third_group_value + (8 - 3) * common_difference →
  eighth_group_value = 144 :=
sorry

end systematic_sampling_eighth_group_l217_217423


namespace smallest_lcm_l217_217581

theorem smallest_lcm (m n : ℕ) (hm : 10000 ≤ m ∧ m < 100000) (hn : 10000 ≤ n ∧ n < 100000) (h : Nat.gcd m n = 5) : Nat.lcm m n = 20030010 :=
sorry

end smallest_lcm_l217_217581


namespace find_a_of_even_function_l217_217926

-- Define the function f
def f (x a : ℝ) := (x + 1) * (x + a)

-- State the theorem to be proven
theorem find_a_of_even_function (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = -1 :=
by
  -- The actual proof goes here
  sorry

end find_a_of_even_function_l217_217926


namespace positive_divisors_3k1_ge_3k_minus_1_l217_217405

theorem positive_divisors_3k1_ge_3k_minus_1 (n : ℕ) (h : 0 < n) :
  (∃ k : ℕ, (3 * k + 1) ∣ n) → (∃ k : ℕ, ¬ (3 * k - 1) ∣ n) :=
  sorry

end positive_divisors_3k1_ge_3k_minus_1_l217_217405


namespace johnson_potatoes_left_l217_217708

theorem johnson_potatoes_left :
  ∀ (initial gina tom anne remaining : Nat),
  initial = 300 →
  gina = 69 →
  tom = 2 * gina →
  anne = tom / 3 →
  remaining = initial - (gina + tom + anne) →
  remaining = 47 := by
sorry

end johnson_potatoes_left_l217_217708


namespace supplement_of_complement_of_75_degree_angle_l217_217924

def angle : ℕ := 75
def complement_angle (a : ℕ) := 90 - a
def supplement_angle (a : ℕ) := 180 - a

theorem supplement_of_complement_of_75_degree_angle : supplement_angle (complement_angle angle) = 165 :=
by
  sorry

end supplement_of_complement_of_75_degree_angle_l217_217924


namespace percentage_increase_of_x_compared_to_y_l217_217425

-- We are given that y = 0.5 * z and x = 0.6 * z
-- We need to prove that the percentage increase of x compared to y is 20%

theorem percentage_increase_of_x_compared_to_y (x y z : ℝ) 
  (h1 : y = 0.5 * z) 
  (h2 : x = 0.6 * z) : 
  (x / y - 1) * 100 = 20 :=
by 
  -- Placeholder for actual proof
  sorry

end percentage_increase_of_x_compared_to_y_l217_217425


namespace general_term_formula_exponential_seq_l217_217254

variable (n : ℕ)

def exponential_sequence (a1 r : ℕ) (n : ℕ) : ℕ := a1 * r^(n-1)

theorem general_term_formula_exponential_seq :
  exponential_sequence 2 3 n = 2 * 3^(n-1) :=
by
  sorry

end general_term_formula_exponential_seq_l217_217254


namespace total_distance_traveled_l217_217889

noncomputable def row_speed_still_water : ℝ := 8
noncomputable def river_speed : ℝ := 2

theorem total_distance_traveled (h : (3.75 / (row_speed_still_water - river_speed)) + (3.75 / (row_speed_still_water + river_speed)) = 1) : 
  2 * 3.75 = 7.5 :=
by
  sorry

end total_distance_traveled_l217_217889


namespace find_x_l217_217602

variables (z y x : Int)

def condition1 : Prop := z + 1 = 0
def condition2 : Prop := y - 1 = 1
def condition3 : Prop := x + 2 = -1

theorem find_x (h1 : condition1 z) (h2 : condition2 y) (h3 : condition3 x) : x = -3 :=
by
  sorry

end find_x_l217_217602


namespace man_climbing_out_of_well_l217_217972

theorem man_climbing_out_of_well (depth climb slip : ℕ) (h1 : depth = 30) (h2 : climb = 4) (h3 : slip = 3) : 
  let effective_climb_per_day := climb - slip
  let total_days := if depth % effective_climb_per_day = 0 then depth / effective_climb_per_day else depth / effective_climb_per_day + 1
  total_days = 30 :=
by
  sorry

end man_climbing_out_of_well_l217_217972


namespace robot_handling_capacity_l217_217060

variables (x : ℝ) (A B : ℝ)

def robot_speed_condition1 : Prop :=
  A = B + 30

def robot_speed_condition2 : Prop :=
  1000 / A = 800 / B

theorem robot_handling_capacity
  (h1 : robot_speed_condition1 A B)
  (h2 : robot_speed_condition2 A B) :
  B = 120 ∧ A = 150 :=
by
  sorry

end robot_handling_capacity_l217_217060


namespace max_value_in_range_l217_217297

noncomputable def x_range : Set ℝ := {x | -5 * Real.pi / 12 ≤ x ∧ x ≤ -Real.pi / 3}

noncomputable def expression (x : ℝ) : ℝ :=
  Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)

theorem max_value_in_range :
  ∀ x ∈ x_range, expression x ≤ (11 / 6) * Real.sqrt 3 :=
sorry

end max_value_in_range_l217_217297


namespace john_sublets_to_3_people_l217_217793

def monthly_income (n : ℕ) : ℕ := 400 * n
def monthly_cost : ℕ := 900
def annual_profit (n : ℕ) : ℕ := 12 * (monthly_income n - monthly_cost)

theorem john_sublets_to_3_people
  (h1 : forall n : ℕ, monthly_income n - monthly_cost > 0)
  (h2 : annual_profit 3 = 3600) :
  3 = 3 := by
  sorry

end john_sublets_to_3_people_l217_217793


namespace number_of_questionnaires_from_unit_D_l217_217046

theorem number_of_questionnaires_from_unit_D 
  (a d : ℕ) 
  (total : ℕ) 
  (samples : ℕ → ℕ) 
  (h_seq : samples 0 = a ∧ samples 1 = a + d ∧ samples 2 = a + 2 * d ∧ samples 3 = a + 3 * d)
  (h_total : samples 0 + samples 1 + samples 2 + samples 3 = total)
  (h_stratified : ∀ (i : ℕ), i < 4 → samples i * 100 / total = 20 → i = 1) 
  : samples 3 = 40 := sorry

end number_of_questionnaires_from_unit_D_l217_217046


namespace john_finances_l217_217384

theorem john_finances :
  let total_first_year := 10000
  let tuition_percent := 0.4
  let room_board_percent := 0.35
  let textbook_transport_percent := 0.25
  let tuition_increase := 0.06
  let room_board_increase := 0.03
  let aid_first_year := 0.25
  let aid_increase := 0.02

  let tuition_first_year := total_first_year * tuition_percent
  let room_board_first_year := total_first_year * room_board_percent
  let textbook_transport_first_year := total_first_year * textbook_transport_percent

  let tuition_second_year := tuition_first_year * (1 + tuition_increase)
  let room_board_second_year := room_board_first_year * (1 + room_board_increase)
  let financial_aid_second_year := tuition_second_year * (aid_first_year + aid_increase)

  let tuition_third_year := tuition_second_year * (1 + tuition_increase)
  let room_board_third_year := room_board_second_year * (1 + room_board_increase)
  let financial_aid_third_year := tuition_third_year * (aid_first_year + 2 * aid_increase)

  let total_cost_first_year := 
      (tuition_first_year - tuition_first_year * aid_first_year) +
      room_board_first_year + 
      textbook_transport_first_year

  let total_cost_second_year :=
      (tuition_second_year - financial_aid_second_year) +
      room_board_second_year +
      textbook_transport_first_year

  let total_cost_third_year :=
      (tuition_third_year - financial_aid_third_year) +
      room_board_third_year +
      textbook_transport_first_year

  total_cost_first_year = 9000 ∧
  total_cost_second_year = 9200.20 ∧
  total_cost_third_year = 9404.17 := 
by
  sorry

end john_finances_l217_217384


namespace sushi_downstream_distance_l217_217742

variable (sushi_speed : ℕ)
variable (stream_speed : ℕ := 12)
variable (upstream_distance : ℕ := 27)
variable (upstream_time : ℕ := 9)
variable (downstream_time : ℕ := 9)

theorem sushi_downstream_distance (h : upstream_distance = (sushi_speed - stream_speed) * upstream_time) : 
  ∃ (D_d : ℕ), D_d = (sushi_speed + stream_speed) * downstream_time ∧ D_d = 243 :=
by {
  -- We assume the given condition for upstream_distance
  sorry
}

end sushi_downstream_distance_l217_217742


namespace quadratic_condition_l217_217625

theorem quadratic_condition (m : ℤ) (x : ℝ) :
  (m + 1) * x^(m^2 + 1) - 2 * x - 5 = 0 ∧ m^2 + 1 = 2 ∧ m + 1 ≠ 0 ↔ m = 1 := 
by
  sorry

end quadratic_condition_l217_217625


namespace find_function_l217_217369

def satisfies_condition (f : ℕ+ → ℕ+) :=
  ∀ a b : ℕ+, f a + b ∣ a^2 + f a * f b

theorem find_function :
  ∀ f : ℕ+ → ℕ+, satisfies_condition f → (∀ a : ℕ+, f a = a) :=
by
  intros f h
  sorry

end find_function_l217_217369


namespace ambulance_ride_cost_is_correct_l217_217944

-- Define all the constants and conditions
def daily_bed_cost : ℝ := 900
def bed_days : ℕ := 3
def specialist_rate_per_hour : ℝ := 250
def specialist_minutes_per_day : ℕ := 15
def specialists_count : ℕ := 2
def total_bill : ℝ := 4625

noncomputable def ambulance_cost : ℝ :=
  total_bill - ((daily_bed_cost * bed_days) + (specialist_rate_per_hour * (specialist_minutes_per_day / 60) * specialists_count))

-- The proof statement
theorem ambulance_ride_cost_is_correct : ambulance_cost = 1675 := by
  sorry

end ambulance_ride_cost_is_correct_l217_217944


namespace exists_perpendicular_line_l217_217243

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure DirectionVector :=
  (dx : ℝ)
  (dy : ℝ)
  (dz : ℝ)

noncomputable def parametric_line_through_point 
  (P : Point3D) 
  (d : DirectionVector) : Prop :=
  ∀ t : ℝ, ∃ x y z : ℝ, 
  x = P.x + d.dx * t ∧
  y = P.y + d.dy * t ∧
  z = P.z + d.dz * t

theorem exists_perpendicular_line : 
  ∃ d : DirectionVector, 
    (d.dx * 2 + d.dy * 3 - d.dz = 0) ∧ 
    (d.dx * 4 - d.dy * -1 + d.dz * 3 = 0) ∧ 
    parametric_line_through_point 
      ⟨3, -2, 1⟩ d :=
  sorry

end exists_perpendicular_line_l217_217243


namespace probability_even_sum_l217_217990

open Nat

def balls : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def even_sum_probability : ℚ :=
  let total_outcomes := 12 * 11
  let even_balls := balls.filter (λ n => n % 2 = 0)
  let odd_balls := balls.filter (λ n => n % 2 = 1)
  let even_outcomes := even_balls.length * (even_balls.length - 1)
  let odd_outcomes := odd_balls.length * (odd_balls.length - 1)
  let favorable_outcomes := even_outcomes + odd_outcomes
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_even_sum :
  even_sum_probability = 5 / 11 := by
  sorry

end probability_even_sum_l217_217990


namespace binom_np_p_div_p4_l217_217184

theorem binom_np_p_div_p4 (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (h3 : 3 < p) (hn : n % p = 1) : p^4 ∣ Nat.choose (n * p) p - n := 
sorry

end binom_np_p_div_p4_l217_217184


namespace ring_worth_l217_217295

theorem ring_worth (R : ℝ) (h1 : (R + 2000 + 2 * R = 14000)) : R = 4000 :=
by 
  sorry

end ring_worth_l217_217295


namespace ratio_area_II_to_III_l217_217983

-- Define the properties of the squares as given in the conditions
def perimeter_region_I : ℕ := 16
def perimeter_region_II : ℕ := 32
def side_length_region_I := perimeter_region_I / 4
def side_length_region_II := perimeter_region_II / 4
def side_length_region_III := 2 * side_length_region_II
def area_region_II := side_length_region_II ^ 2
def area_region_III := side_length_region_III ^ 2

-- Prove that the ratio of the area of region II to the area of region III is 1/4
theorem ratio_area_II_to_III : (area_region_II : ℚ) / (area_region_III : ℚ) = 1 / 4 := 
by sorry

end ratio_area_II_to_III_l217_217983


namespace proof_negation_l217_217389

-- Definitions of rational and real numbers
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- Proposition stating the existence of an irrational number that is rational
def original_proposition : Prop :=
  ∃ x : ℝ, is_irrational x ∧ is_rational x

-- Negation of the original proposition
def negated_proposition : Prop :=
  ∀ x : ℝ, is_irrational x → ¬ is_rational x

theorem proof_negation : ¬ original_proposition = negated_proposition := 
sorry

end proof_negation_l217_217389


namespace no_solution_for_x6_eq_2y2_plus_2_l217_217043

theorem no_solution_for_x6_eq_2y2_plus_2 :
  ¬ ∃ (x y : ℤ), x^6 = 2 * y^2 + 2 :=
sorry

end no_solution_for_x6_eq_2y2_plus_2_l217_217043


namespace trailing_zeros_of_9_pow_999_plus_1_l217_217920

theorem trailing_zeros_of_9_pow_999_plus_1 :
  ∃ n : ℕ, n = 999 ∧ (9^n + 1) % 10 = 0 ∧ (9^n + 1) % 100 ≠ 0 :=
by
  sorry

end trailing_zeros_of_9_pow_999_plus_1_l217_217920


namespace proof_problem_l217_217940

noncomputable def problem : ℕ :=
  let p := 588
  let q := 0
  let r := 1
  p + q + r

theorem proof_problem
  (AB : ℝ) (P Q : ℝ) (AP BP PQ : ℝ) (angle_POQ : ℝ) 
  (h1 : AB = 1200)
  (h2 : AP + PQ = BP)
  (h3 : BP - Q = 600)
  (h4 : angle_POQ = 30)
  (h5 : PQ = 500)
  : problem = 589 := by
    sorry

end proof_problem_l217_217940


namespace handshakes_at_gathering_l217_217070

noncomputable def total_handshakes : Nat :=
  let twins := 16
  let triplets := 15
  let handshakes_among_twins := twins * 14 / 2
  let handshakes_among_triplets := 0
  let cross_handshakes := twins * triplets
  handshakes_among_twins + handshakes_among_triplets + cross_handshakes

theorem handshakes_at_gathering : total_handshakes = 352 := 
by
  -- By substituting the values, we can solve and show that the total handshakes equal to 352.
  sorry

end handshakes_at_gathering_l217_217070


namespace ratio_of_speeds_l217_217087

variables (v_A v_B v_C : ℝ)

-- Conditions definitions
def condition1 : Prop := v_A - v_B = 5
def condition2 : Prop := v_A + v_C = 15

-- Theorem statement (the mathematically equivalent proof problem)
theorem ratio_of_speeds (h1 : condition1 v_A v_B) (h2 : condition2 v_A v_C) : (v_A / v_B) = 3 :=
sorry

end ratio_of_speeds_l217_217087


namespace disjoint_sets_condition_l217_217093

theorem disjoint_sets_condition (A B : Set ℕ) (h_disjoint: Disjoint A B) (h_union: A ∪ B = Set.univ) :
  ∀ n : ℕ, ∃ a b : ℕ, a > n ∧ b > n ∧ a ≠ b ∧ 
             ((a ∈ A ∧ b ∈ A ∧ a + b ∈ A) ∨ (a ∈ B ∧ b ∈ B ∧ a + b ∈ B)) := 
by
  sorry

end disjoint_sets_condition_l217_217093


namespace find_C_l217_217488

theorem find_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 340) : C = 40 :=
by sorry

end find_C_l217_217488


namespace measure_of_angle_y_l217_217828

theorem measure_of_angle_y (m n : ℝ) (A B C D F G H : ℝ) :
  (m = n) → (A = 40) → (B = 90) → (B = 40) → (y = 80) :=
by
  -- proof steps to be filled in
  sorry

end measure_of_angle_y_l217_217828


namespace find_x_l217_217987

theorem find_x (x y : ℚ) (h1 : 3 * x - y = 9) (h2 : x + 4 * y = 11) : x = 47 / 13 :=
sorry

end find_x_l217_217987


namespace pythagorean_triple_transformation_l217_217756

theorem pythagorean_triple_transformation
  (a b c α β γ s p q r : ℝ)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : α^2 + β^2 - γ^2 = 2)
  (h₃ : s = a * α + b * β - c * γ)
  (h₄ : p = a - α * s)
  (h₅ : q = b - β * s)
  (h₆ : r = c - γ * s) :
  p^2 + q^2 = r^2 :=
by
  sorry

end pythagorean_triple_transformation_l217_217756


namespace correct_operation_l217_217792

theorem correct_operation :
  ¬(a^2 * a^3 = a^6) ∧ (2 * a^3 / a = 2 * a^2) ∧ ¬((a * b)^2 = a * b^2) ∧ ¬((-a^3)^3 = -a^6) :=
by
  sorry

end correct_operation_l217_217792


namespace find_two_digit_number_l217_217584

theorem find_two_digit_number :
  ∃ x y : ℕ, 10 * x + y = 78 ∧ 10 * x + y < 100 ∧ y ≠ 0 ∧ (10 * x + y) / y = 9 ∧ (10 * x + y) % y = 6 :=
by
  sorry

end find_two_digit_number_l217_217584


namespace false_statement_of_quadratic_l217_217936

-- Define the function f and the conditions
def f (a b c x : ℝ) := a * x^2 + b * x + c

theorem false_statement_of_quadratic (a b c x0 : ℝ) (h₀ : a > 0) (h₁ : 2 * a * x0 + b = 0) :
  ¬ ∀ x : ℝ, f a b c x ≤ f a b c x0 := by
  sorry

end false_statement_of_quadratic_l217_217936


namespace remainder_of_product_modulo_12_l217_217627

theorem remainder_of_product_modulo_12 : (1625 * 1627 * 1629) % 12 = 3 := by
  sorry

end remainder_of_product_modulo_12_l217_217627


namespace sum_of_digits_ABCED_l217_217677

theorem sum_of_digits_ABCED {A B C D E : ℕ} (hABCED : 3 * (10000 * A + 1000 * B + 100 * C + 10 * D + E) = 111111) :
  A + B + C + D + E = 20 := 
by
  sorry

end sum_of_digits_ABCED_l217_217677


namespace find_numbers_l217_217960

theorem find_numbers (a b c : ℕ) (h : a + b = 2015) (h' : a = 10 * b + c) (hc : 0 ≤ c ∧ c ≤ 9) :
  (a = 1832 ∧ b = 183) :=
sorry

end find_numbers_l217_217960


namespace intersection_points_rectangular_coords_l217_217553

theorem intersection_points_rectangular_coords :
  ∃ (x y : ℝ),
    (∃ (ρ θ : ℝ), ρ = 2 * Real.cos θ ∧ ρ^2 * (Real.cos θ)^2 - 4 * ρ^2 * (Real.sin θ)^2 = 4 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
    (x = (1 + Real.sqrt 13) / 3 ∧ y = 0) := 
sorry

end intersection_points_rectangular_coords_l217_217553


namespace minimize_expression_l217_217766

theorem minimize_expression (n : ℕ) (h : 0 < n) : 
  (n = 10) ↔ (∀ m : ℕ, 0 < m → ((n / 2) + (50 / n) ≤ (m / 2) + (50 / m))) :=
sorry

end minimize_expression_l217_217766


namespace shopkeeper_profit_percentage_goal_l217_217617

-- Definitions for CP, MP and discount percentage
variable (CP : ℝ)
noncomputable def MP : ℝ := CP * 1.32
noncomputable def discount_percentage : ℝ := 0.18939393939393938
noncomputable def SP : ℝ := MP CP - (discount_percentage * MP CP)
noncomputable def profit : ℝ := SP CP - CP
noncomputable def profit_percentage : ℝ := (profit CP / CP) * 100

-- Theorem stating that the profit percentage is approximately 7%
theorem shopkeeper_profit_percentage_goal :
  abs (profit_percentage CP - 7) < 0.01 := sorry

end shopkeeper_profit_percentage_goal_l217_217617


namespace basketball_lineup_count_l217_217058

theorem basketball_lineup_count :
  (∃ (players : Finset ℕ), players.card = 15) → 
  ∃ centers power_forwards small_forwards shooting_guards point_guards sixth_men : ℕ,
  ∃ b : Fin (15) → Fin (15),
  15 * 14 * 13 * 12 * 11 * 10 = 360360 
:= by sorry

end basketball_lineup_count_l217_217058


namespace Grace_pool_water_capacity_l217_217360

theorem Grace_pool_water_capacity :
  let rate1 := 50 -- gallons per hour of the first hose
  let rate2 := 70 -- gallons per hour of the second hose
  let hours1 := 3 -- hours the first hose was used alone
  let hours2 := 2 -- hours both hoses were used together
  let water1 := rate1 * hours1 -- water from the first hose in the first period
  let water2 := rate2 * hours2 -- water from the second hose in the second period
  let water3 := rate1 * hours2 -- water from the first hose in the second period
  let total_water := water1 + water2 + water3 -- total water in the pool
  total_water = 390 :=
by
  sorry

end Grace_pool_water_capacity_l217_217360


namespace smallest_clock_equivalent_number_l217_217645

theorem smallest_clock_equivalent_number :
  ∃ h : ℕ, h > 4 ∧ h^2 % 24 = h % 24 ∧ h = 12 := by
  sorry

end smallest_clock_equivalent_number_l217_217645


namespace smaller_angle_at_7_15_l217_217286

theorem smaller_angle_at_7_15 
  (hour_hand_rate : ℕ → ℝ)
  (minute_hand_rate : ℕ → ℝ)
  (hour_time : ℕ)
  (minute_time : ℕ)
  (top_pos : ℝ)
  (smaller_angle : ℝ) 
  (h1 : hour_hand_rate hour_time + (minute_time/60) * hour_hand_rate hour_time = 217.5)
  (h2 : minute_hand_rate minute_time = 90.0)
  (h3 : |217.5 - 90.0| = smaller_angle) :
  smaller_angle = 127.5 :=
by
  sorry

end smaller_angle_at_7_15_l217_217286


namespace part_a_part_b_l217_217322

-- Part (a)
theorem part_a (a b : ℕ) (h : Nat.lcm a (a + 5) = Nat.lcm b (b + 5)) : a = b :=
sorry

-- Part (b)
theorem part_b (a b c : ℕ) (gcd_abc : Nat.gcd a (Nat.gcd b c) = 1) :
  Nat.lcm a b = Nat.lcm (a + c) (b + c) → False :=
sorry

end part_a_part_b_l217_217322


namespace scientific_notation_conversion_l217_217381

theorem scientific_notation_conversion :
  216000 = 2.16 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l217_217381


namespace part1_solution_set_part2_range_a_l217_217442

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 1)

theorem part1_solution_set :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3 / 2} ∪ {x : ℝ | x ≥ 3 / 2} := 
sorry

theorem part2_range_a (a : ℝ) : 
  (∀ x : ℝ, f x ≥ a^2 - a) ↔ (-1 ≤ a ∧ a ≤ 2) := 
sorry

end part1_solution_set_part2_range_a_l217_217442


namespace candy_left_l217_217477

-- Define the number of candies each sibling has
def debbyCandy : ℕ := 32
def sisterCandy : ℕ := 42
def brotherCandy : ℕ := 48

-- Define the total candies collected
def totalCandy : ℕ := debbyCandy + sisterCandy + brotherCandy

-- Define the number of candies eaten
def eatenCandy : ℕ := 56

-- Define the remaining candies after eating some
def remainingCandy : ℕ := totalCandy - eatenCandy

-- The hypothesis stating the initial condition
theorem candy_left (h1 : debbyCandy = 32) (h2 : sisterCandy = 42) (h3 : brotherCandy = 48) (h4 : eatenCandy = 56) : remainingCandy = 66 :=
by
  -- Proof can be filled in here
  sorry

end candy_left_l217_217477


namespace triangle_angle_and_area_l217_217958

section Geometry

variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def triangle_sides_opposite_angles (a b c : ℝ) (A B C : ℝ) : Prop := 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

def vectors_parallel (a b : ℝ) (A B : ℝ) : Prop := 
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A

-- Problem statement
theorem triangle_angle_and_area (A B C a b c : ℝ) : 
  triangle_sides_opposite_angles a b c A B C ∧ vectors_parallel a b A B ∧ a = Real.sqrt 7 ∧ b = 2 ∧ A = Real.pi / 3
  → A = Real.pi / 3 ∧ (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
sorry

end Geometry

end triangle_angle_and_area_l217_217958


namespace more_valley_than_humpy_l217_217737

def is_humpy (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 > d4 ∧ d4 > d5

def is_valley (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 > d2 ∧ d2 > d3 ∧ d3 < d4 ∧ d4 < d5

def starts_with_5 (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  d1 = 5

theorem more_valley_than_humpy :
  (∃ m, starts_with_5 m ∧ is_humpy m) → (∃ n, starts_with_5 n ∧ is_valley n) ∧ 
  (∀ x, starts_with_5 x → is_humpy x → ∃ y, starts_with_5 y ∧ is_valley y ∧ y ≠ x) :=
by sorry

end more_valley_than_humpy_l217_217737


namespace kelly_single_shot_decrease_l217_217761

def kelly_salary_decrease (s : ℝ) : ℝ :=
  let first_cut := s * 0.92
  let second_cut := first_cut * 0.86
  let third_cut := second_cut * 0.82
  third_cut

theorem kelly_single_shot_decrease :
  let original_salary := 1.0 -- Assume original salary is 1 for percentage calculation
  let final_salary := kelly_salary_decrease original_salary
  (100 : ℝ) - (final_salary * 100) = 34.8056 :=
by
  sorry

end kelly_single_shot_decrease_l217_217761


namespace not_all_inequalities_hold_l217_217077

theorem not_all_inequalities_hold (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
(hlt_a : a < 1) (hlt_b : b < 1) (hlt_c : c < 1) :
  ¬(a * (1 - b) > 1 / 4 ∧ b * (1 - c) > 1 / 4 ∧ c * (1 - a) > 1 / 4) :=
by
  sorry

end not_all_inequalities_hold_l217_217077


namespace triangle_side_range_l217_217548

theorem triangle_side_range (x : ℝ) (hx1 : 8 + 10 > x) (hx2 : 10 + x > 8) (hx3 : x + 8 > 10) : 2 < x ∧ x < 18 :=
by
  sorry

end triangle_side_range_l217_217548


namespace find_value_l217_217745

variables (a b c d : ℝ)

theorem find_value
  (h1 : a - b = 3)
  (h2 : c + d = 2) :
  (a + c) - (b - d) = 5 :=
by sorry

end find_value_l217_217745


namespace total_weight_of_2_meters_l217_217680

def tape_measure_length : ℚ := 5
def tape_measure_weight : ℚ := 29 / 8
def computer_length : ℚ := 4
def computer_weight : ℚ := 2.8

noncomputable def weight_per_meter_tape_measure : ℚ := tape_measure_weight / tape_measure_length
noncomputable def weight_per_meter_computer : ℚ := computer_weight / computer_length

noncomputable def total_weight : ℚ :=
  2 * weight_per_meter_tape_measure + 2 * weight_per_meter_computer

theorem total_weight_of_2_meters (h1 : tape_measure_length = 5)
    (h2 : tape_measure_weight = 29 / 8) 
    (h3 : computer_length = 4) 
    (h4 : computer_weight = 2.8): 
    total_weight = 57 / 20 := by 
  unfold total_weight
  sorry

end total_weight_of_2_meters_l217_217680


namespace fraction_of_sum_l217_217746

theorem fraction_of_sum (n S : ℕ) 
  (h1 : S = (n-1) * ((n:ℚ) / 3))
  (h2 : n > 0) : 
  (n:ℚ) / (S + n) = 3 / (n + 2) := 
by 
  sorry

end fraction_of_sum_l217_217746


namespace find_starting_number_l217_217035

theorem find_starting_number (S : ℤ) (n : ℤ) (sum_eq : 10 = S) (consec_eq : S = (20 / 2) * (n + (n + 19))) : 
  n = -9 := 
by
  sorry

end find_starting_number_l217_217035


namespace prop1_prop2_prop3_l217_217312

variables (a b c d : ℝ)

-- Proposition 1: ab > 0 ∧ bc - ad > 0 → (c/a - d/b > 0)
theorem prop1 (h1 : a * b > 0) (h2 : b * c - a * d > 0) : c / a - d / b > 0 :=
sorry

-- Proposition 2: ab > 0 ∧ (c/a - d/b > 0) → bc - ad > 0
theorem prop2 (h1 : a * b > 0) (h2 : c / a - d / b > 0) : b * c - a * d > 0 :=
sorry

-- Proposition 3: (bc - ad > 0) ∧ (c/a - d/b > 0) → ab > 0
theorem prop3 (h1 : b * c - a * d > 0) (h2 : c / a - d / b > 0) : a * b > 0 :=
sorry

end prop1_prop2_prop3_l217_217312


namespace find_m_l217_217731

theorem find_m (C D m : ℤ) (h1 : C = D + m) (h2 : C - 1 = 6 * (D - 1)) (h3 : C = D^3) : m = 0 :=
by sorry

end find_m_l217_217731


namespace gcd_18_30_l217_217123

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l217_217123


namespace evans_family_children_count_l217_217823

-- Let the family consist of the mother, the father, two grandparents, and children.
-- This proof aims to show x, the number of children, is 1.

theorem evans_family_children_count
  (m g y : ℕ) -- m = mother's age, g = average age of two grandparents, y = average age of children
  (x : ℕ) -- x = number of children
  (avg_family_age : (m + 50 + 2 * g + x * y) / (4 + x) = 30)
  (father_age : 50 = 50)
  (avg_non_father_age : (m + 2 * g + x * y) / (3 + x) = 25) :
  x = 1 :=
sorry

end evans_family_children_count_l217_217823


namespace flagpole_proof_l217_217875

noncomputable def flagpole_height (AC AD DE : ℝ) (h_ABC_DEC : (AC ≠ 0) ∧ (AC - AD ≠ 0) ∧ (DE ≠ 0)) : ℝ :=
  let DC := AC - AD
  let h_ratio := DE / DC
  h_ratio * AC

theorem flagpole_proof (AC AD DE : ℝ) (h_AC : AC = 4) (h_AD : AD = 3) (h_DE : DE = 1.8) 
  (h_ABC_DEC : (AC ≠ 0) ∧ (AC - AD ≠ 0) ∧ (DE ≠ 0)) :
  flagpole_height AC AD DE h_ABC_DEC = 7.2 := by
  sorry

end flagpole_proof_l217_217875


namespace solution_set_of_inequality_l217_217888

theorem solution_set_of_inequality :
  { x : ℝ // (x - 2)^2 ≤ 2 * x + 11 } = { x : ℝ | -1 ≤ x ∧ x ≤ 7 } :=
sorry

end solution_set_of_inequality_l217_217888


namespace cost_price_of_radio_l217_217953

theorem cost_price_of_radio (SP : ℝ) (L_p : ℝ) (C : ℝ) (h₁ : SP = 3200) (h₂ : L_p = 0.28888888888888886) 
  (h₃ : SP = C - (C * L_p)) : C = 4500 :=
by
  sorry

end cost_price_of_radio_l217_217953


namespace g_of_f_eq_l217_217283

def f (A B x : ℝ) : ℝ := A * x^2 - B^2
def g (B x : ℝ) : ℝ := B * x + B^2

theorem g_of_f_eq (A B : ℝ) (hB : B ≠ 0) : 
  g B (f A B 1) = B * A - B^3 + B^2 := 
by
  sorry

end g_of_f_eq_l217_217283


namespace class_funding_reached_l217_217276

-- Definition of the conditions
def students : ℕ := 45
def goal : ℝ := 3000
def full_payment_students : ℕ := 25
def full_payment_amount : ℝ := 60
def merit_students : ℕ := 10
def merit_payment_per_student_euro : ℝ := 40
def euro_to_usd : ℝ := 1.20
def financial_needs_students : ℕ := 7
def financial_needs_payment_per_student_pound : ℝ := 30
def pound_to_usd : ℝ := 1.35
def discount_students : ℕ := 3
def discount_payment_per_student_cad : ℝ := 68
def cad_to_usd : ℝ := 0.80
def administrative_fee_yen : ℝ := 10000
def yen_to_usd : ℝ := 0.009

-- Definitions of amounts
def full_payment_amount_total : ℝ := full_payment_students * full_payment_amount
def merit_payment_amount_total : ℝ := merit_students * merit_payment_per_student_euro * euro_to_usd
def financial_needs_payment_amount_total : ℝ := financial_needs_students * financial_needs_payment_per_student_pound * pound_to_usd
def discount_payment_amount_total : ℝ := discount_students * discount_payment_per_student_cad * cad_to_usd
def administrative_fee_usd : ℝ := administrative_fee_yen * yen_to_usd

-- Definition of total collected
def total_collected : ℝ := 
  full_payment_amount_total + 
  merit_payment_amount_total + 
  financial_needs_payment_amount_total + 
  discount_payment_amount_total - 
  administrative_fee_usd

-- The final theorem statement
theorem class_funding_reached : total_collected = 2427.70 ∧ goal - total_collected = 572.30 := by
  sorry

end class_funding_reached_l217_217276


namespace min_books_borrowed_l217_217067

theorem min_books_borrowed 
    (h1 : 12 * 1 = 12) 
    (h2 : 10 * 2 = 20) 
    (h3 : 2 = 2) 
    (h4 : 32 = 32) 
    (h5 : (32 * 2 = 64))
    (h6 : ∀ x, x ≤ 11) :
    ∃ (x : ℕ), (8 * x = 32) ∧ x ≤ 11 := 
  sorry

end min_books_borrowed_l217_217067


namespace side_lengths_are_10_and_50_l217_217064

-- Define variables used in the problem
variables {s t : ℕ}

-- Define the conditions
def condition1 (s t : ℕ) : Prop := 4 * s = 20 * t
def condition2 (s t : ℕ) : Prop := s + t = 60

-- Prove that given the conditions, the side lengths of the squares are 10 and 50
theorem side_lengths_are_10_and_50 (s t : ℕ) (h1 : condition1 s t) (h2 : condition2 s t) : (s = 50 ∧ t = 10) ∨ (s = 10 ∧ t = 50) :=
by sorry

end side_lengths_are_10_and_50_l217_217064


namespace simplify_fraction_l217_217047

theorem simplify_fraction (x y : ℕ) : (x + y)^3 / (x + y) = (x + y)^2 := by
  sorry

end simplify_fraction_l217_217047


namespace middle_digit_is_3_l217_217789

theorem middle_digit_is_3 (d e f : ℕ) (hd : 0 ≤ d ∧ d ≤ 7) (he : 0 ≤ e ∧ e ≤ 7) (hf : 0 ≤ f ∧ f ≤ 7)
    (h_eq : 64 * d + 8 * e + f = 100 * f + 10 * e + d) : e = 3 :=
sorry

end middle_digit_is_3_l217_217789


namespace corrected_mean_35_25_l217_217711

theorem corrected_mean_35_25 (n : ℕ) (mean : ℚ) (x_wrong x_correct : ℚ) :
  n = 20 → mean = 36 → x_wrong = 40 → x_correct = 25 → 
  ( (mean * n - x_wrong + x_correct) / n = 35.25) :=
by
  intros h1 h2 h3 h4
  sorry

end corrected_mean_35_25_l217_217711


namespace exists_num_with_digit_sum_div_by_11_l217_217542

-- Helper function to sum the digits of a natural number
def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Main theorem statement
theorem exists_num_with_digit_sum_div_by_11 (N : ℕ) :
  ∃ k : ℕ, k < 39 ∧ (digit_sum (N + k)) % 11 = 0 :=
sorry

end exists_num_with_digit_sum_div_by_11_l217_217542


namespace sufficient_not_necessary_condition_not_necessary_condition_l217_217785

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  a^2 + b^2 = 1 → (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) :=
by
  sorry

theorem not_necessary_condition (a b : ℝ) : 
  (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) → ¬(a^2 + b^2 = 1) :=
by
  sorry

end sufficient_not_necessary_condition_not_necessary_condition_l217_217785


namespace value_of_2_Z_6_l217_217062

def Z (a b : ℝ) : ℝ := b + 10 * a - a^2

theorem value_of_2_Z_6 : Z 2 6 = 22 :=
by
  sorry

end value_of_2_Z_6_l217_217062


namespace no_solutions_l217_217147

theorem no_solutions (x : ℝ) (hx : x ≠ 0): ¬ (12 * Real.sin x + 5 * Real.cos x = 13 + 1 / |x|) := 
by 
  sorry

end no_solutions_l217_217147


namespace johns_age_l217_217995

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
sorry

end johns_age_l217_217995


namespace probability_first_head_second_tail_l217_217504

-- Conditions
def fair_coin := true
def prob_heads := 1 / 2
def prob_tails := 1 / 2
def independent_events (A B : Prop) := true

-- Statement
theorem probability_first_head_second_tail :
  fair_coin →
  independent_events (prob_heads = 1/2) (prob_tails = 1/2) →
  (prob_heads * prob_tails) = 1/4 :=
by
  sorry

end probability_first_head_second_tail_l217_217504


namespace insurance_covers_80_percent_of_medical_bills_l217_217941

theorem insurance_covers_80_percent_of_medical_bills 
    (vaccine_cost : ℕ) (num_vaccines : ℕ) (doctor_visit_cost trip_cost : ℕ) (amount_tom_pays : ℕ) 
    (total_cost := num_vaccines * vaccine_cost + doctor_visit_cost) 
    (total_trip_cost := trip_cost + total_cost)
    (insurance_coverage := total_trip_cost - amount_tom_pays)
    (percent_covered := (insurance_coverage * 100) / total_cost) :
    vaccine_cost = 45 → num_vaccines = 10 → doctor_visit_cost = 250 → trip_cost = 1200 → amount_tom_pays = 1340 →
    percent_covered = 80 := 
by
  sorry

end insurance_covers_80_percent_of_medical_bills_l217_217941


namespace circle_center_and_radius_sum_l217_217066

theorem circle_center_and_radius_sum :
  let a := -4
  let b := -8
  let r := Real.sqrt 17
  a + b + r = -12 + Real.sqrt 17 :=
by
  sorry

end circle_center_and_radius_sum_l217_217066


namespace rectangle_ratio_l217_217671

theorem rectangle_ratio (s x y : ℝ) (h1 : 4 * (x * y) + s * s = 9 * s * s) (h2 : s + 2 * y = 3 * s) (h3 : x + y = 3 * s): x / y = 2 :=
by sorry

end rectangle_ratio_l217_217671


namespace prob_four_children_at_least_one_boy_one_girl_l217_217744

-- Define the probability of a single birth being a boy or a girl
def prob_boy_or_girl : ℚ := 1/2

-- Calculate the probability of all children being boys or all girls
def prob_all_boys : ℚ := (prob_boy_or_girl)^4
def prob_all_girls : ℚ := (prob_boy_or_girl)^4

-- Calculate the probability of having neither all boys nor all girls
def prob_at_least_one_boy_one_girl : ℚ := 1 - (prob_all_boys + prob_all_girls)

-- The theorem to prove
theorem prob_four_children_at_least_one_boy_one_girl : 
  prob_at_least_one_boy_one_girl = 7/8 := 
by 
  sorry

end prob_four_children_at_least_one_boy_one_girl_l217_217744


namespace binom_600_eq_1_l217_217825

theorem binom_600_eq_1 : Nat.choose 600 600 = 1 :=
by sorry

end binom_600_eq_1_l217_217825


namespace evaluate_expression_l217_217871

theorem evaluate_expression : 2 * (2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3) = 24240542 :=
by
  let a := 2009
  let b := 2010
  sorry

end evaluate_expression_l217_217871


namespace inequality_not_always_true_l217_217772

theorem inequality_not_always_true (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬(∀ a > 0, ∀ b > 0, (2 / ((1 / a) + (1 / b)) ≥ Real.sqrt (a * b))) :=
sorry

end inequality_not_always_true_l217_217772


namespace speed_of_stream_l217_217848

theorem speed_of_stream
  (v_a v_s : ℝ)
  (h1 : v_a - v_s = 4)
  (h2 : v_a + v_s = 6) :
  v_s = 1 :=
by {
  sorry
}

end speed_of_stream_l217_217848


namespace combined_share_a_c_l217_217023

-- Define the conditions
def total_money : ℕ := 15800
def ratio_a : ℕ := 5
def ratio_b : ℕ := 9
def ratio_c : ℕ := 6
def ratio_d : ℕ := 5

-- The total parts in the ratio
def total_parts : ℕ := ratio_a + ratio_b + ratio_c + ratio_d

-- The value of each part
def value_per_part : ℕ := total_money / total_parts

-- The shares of a and c
def share_a : ℕ := ratio_a * value_per_part
def share_c : ℕ := ratio_c * value_per_part

-- Prove that the combined share of a + c equals 6952
theorem combined_share_a_c : share_a + share_c = 6952 :=
by
  -- This is the proof placeholder
  sorry

end combined_share_a_c_l217_217023


namespace artist_paints_total_exposed_surface_area_l217_217730

def num_cubes : Nat := 18
def edge_length : Nat := 1

-- Define the configuration of cubes
def bottom_layer_grid : Nat := 9 -- Number of cubes in the 3x3 grid (bottom layer)
def top_layer_cross : Nat := 9 -- Number of cubes in the cross shape (top layer)

-- Exposed surfaces in bottom layer
def bottom_layer_exposed_surfaces : Nat :=
  let top_surfaces := 9 -- 9 top surfaces for 9 cubes
  let corner_cube_sides := 4 * 3 -- 4 corners, 3 exposed sides each
  let edge_cube_sides := 4 * 2 -- 4 edge (non-corner) cubes, 2 exposed sides each
  top_surfaces + corner_cube_sides + edge_cube_sides

-- Exposed surfaces in top layer
def top_layer_exposed_surfaces : Nat :=
  let top_surfaces := 5 -- 5 top surfaces for 5 cubes in the cross
  let side_surfaces_of_cross_arms := 4 * 3 -- 4 arms, 3 exposed sides each
  top_surfaces + side_surfaces_of_cross_arms

-- Total exposed surface area
def total_exposed_surface_area : Nat :=
  bottom_layer_exposed_surfaces + top_layer_exposed_surfaces

-- Problem statement
theorem artist_paints_total_exposed_surface_area :
  total_exposed_surface_area = 46 := by
    sorry

end artist_paints_total_exposed_surface_area_l217_217730


namespace problem_statement_l217_217478

theorem problem_statement (x y : ℝ) (h : x * y < 0) : abs (x + y) < abs (x - y) :=
sorry

end problem_statement_l217_217478


namespace sum_of_other_endpoint_coordinates_l217_217528

theorem sum_of_other_endpoint_coordinates (x y : ℝ) (hx : (x + 5) / 2 = 3) (hy : (y - 2) / 2 = 4) :
  x + y = 11 :=
sorry

end sum_of_other_endpoint_coordinates_l217_217528


namespace remainder_15_plus_3y_l217_217676

theorem remainder_15_plus_3y (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (15 + 3 * y) % 31 = 11 :=
by
  sorry

end remainder_15_plus_3y_l217_217676


namespace problem_statement_l217_217315

variable {P : ℕ → Prop}

theorem problem_statement
  (h1 : ∀ k, P k → P (k + 1))
  (h2 : ¬P 4)
  (n : ℕ) (hn : 1 ≤ n → n ≤ 4 → n ∈ Set.Icc 1 4) :
  ¬P n :=
by
  sorry

end problem_statement_l217_217315


namespace range_of_3x_plus_2y_l217_217350

theorem range_of_3x_plus_2y (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 3) (hy : -1 ≤ y ∧ y ≤ 4) :
  1 ≤ 3 * x + 2 * y ∧ 3 * x + 2 * y ≤ 17 :=
sorry

end range_of_3x_plus_2y_l217_217350


namespace angle_sum_at_F_l217_217959

theorem angle_sum_at_F (x y z w v : ℝ) (h : x + y + z + w + v = 360) : 
  x = 360 - y - z - w - v := by
  sorry

end angle_sum_at_F_l217_217959


namespace divisor_between_l217_217460

theorem divisor_between (n a b : ℕ) (h_n_gt_8 : n > 8) (h_a_dvd_n : a ∣ n) (h_b_dvd_n : b ∣ n) 
    (h_a_lt_b : a < b) (h_n_eq_asq_plus_b : n = a^2 + b) (h_a_ne_b : a ≠ b) :
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b :=
sorry

end divisor_between_l217_217460


namespace percent_red_prob_l217_217403

-- Define the conditions
def initial_red := 2
def initial_blue := 4
def additional_red := 2
def additional_blue := 2
def total_balloons := initial_red + initial_blue + additional_red + additional_blue
def total_red := initial_red + additional_red

-- State the theorem
theorem percent_red_prob : (total_red.toFloat / total_balloons.toFloat) * 100 = 40 :=
by
  sorry

end percent_red_prob_l217_217403


namespace ryan_hours_on_english_l217_217486

-- Given the conditions
def hours_on_chinese := 2
def hours_on_spanish := 4
def extra_hours_between_english_and_spanish := 3

-- We want to find out the hours on learning English
def hours_on_english := hours_on_spanish + extra_hours_between_english_and_spanish

-- Proof statement
theorem ryan_hours_on_english : hours_on_english = 7 := by
  -- This is where the proof would normally go.
  sorry

end ryan_hours_on_english_l217_217486


namespace dislike_both_tv_and_video_games_l217_217236

theorem dislike_both_tv_and_video_games (total_people : ℕ) (percent_dislike_tv : ℝ) (percent_dislike_tv_and_games : ℝ) :
  let people_dislike_tv := percent_dislike_tv * total_people
  let people_dislike_both := percent_dislike_tv_and_games * people_dislike_tv
  total_people = 1800 ∧ percent_dislike_tv = 0.4 ∧ percent_dislike_tv_and_games = 0.25 →
  people_dislike_both = 180 :=
by {
  sorry
}

end dislike_both_tv_and_video_games_l217_217236


namespace tic_tac_toe_tie_probability_l217_217391

theorem tic_tac_toe_tie_probability (john_wins martha_wins : ℚ) 
  (hj : john_wins = 4 / 9) 
  (hm : martha_wins = 5 / 12) : 
  1 - (john_wins + martha_wins) = 5 / 36 := 
by {
  /- insert proof here -/
  sorry
}

end tic_tac_toe_tie_probability_l217_217391


namespace find_abc_l217_217938

theorem find_abc (a b c : ℕ) (h_coprime_ab : gcd a b = 1) (h_coprime_ac : gcd a c = 1) 
  (h_coprime_bc : gcd b c = 1) (h1 : ab + bc + ac = 431) (h2 : a + b + c = 39) 
  (h3 : a + b + (ab / c) = 18) : 
  a = 7 ∧ b = 9 ∧ c = 23 := 
sorry

end find_abc_l217_217938


namespace polynomial_simplification_l217_217566

variable (x : ℝ)

theorem polynomial_simplification : 
  ((3 * x - 2) * (5 * x ^ 12 + 3 * x ^ 11 - 4 * x ^ 9 + x ^ 8)) = 
  (15 * x ^ 13 - x ^ 12 - 6 * x ^ 11 - 12 * x ^ 10 + 11 * x ^ 9 - 2 * x ^ 8) := by
  sorry

end polynomial_simplification_l217_217566


namespace amy_points_per_treasure_l217_217643

theorem amy_points_per_treasure (treasures_first_level treasures_second_level total_score : ℕ) (h1 : treasures_first_level = 6) (h2 : treasures_second_level = 2) (h3 : total_score = 32) :
  total_score / (treasures_first_level + treasures_second_level) = 4 := by
  sorry

end amy_points_per_treasure_l217_217643


namespace abs_neg_two_l217_217144

theorem abs_neg_two : abs (-2) = 2 :=
by
  sorry

end abs_neg_two_l217_217144


namespace unique_n_in_range_satisfying_remainders_l217_217036

theorem unique_n_in_range_satisfying_remainders : 
  ∃! (n : ℤ) (k r : ℤ), 150 < n ∧ n < 250 ∧ 0 <= r ∧ r <= 6 ∧ n = 7 * k + r ∧ n = 9 * r + r :=
sorry

end unique_n_in_range_satisfying_remainders_l217_217036


namespace measure_of_angle_XPM_l217_217400

-- Definitions based on given conditions
variables (X Y Z L M N P : Type)
variables (a b c : ℝ) -- Angles are represented in degrees
variables [DecidableEq X] [DecidableEq Y] [DecidableEq Z]

-- Triangle XYZ with angle bisectors XL, YM, and ZN meeting at incenter P
-- Given angle XYZ in degrees
def angle_XYZ : ℝ := 46

-- Incenter angle properties
axiom angle_bisector_XL (angle_XYP : ℝ) : angle_XYP = angle_XYZ / 2
axiom angle_bisector_YM (angle_YXP : ℝ) : ∃ (angle_YXZ : ℝ), angle_YXP = angle_YXZ / 2

-- The proposition we need to prove
theorem measure_of_angle_XPM : ∃ (angle_XPM : ℝ), angle_XPM = 67 := 
by {
  sorry
}

end measure_of_angle_XPM_l217_217400


namespace compare_A_B_l217_217820

-- Definitions based on conditions from part a)
def A (n : ℕ) : ℕ := 2 * n^2
def B (n : ℕ) : ℕ := 3^n

-- The theorem that needs to be proven
theorem compare_A_B (n : ℕ) (h : n > 0) : A n < B n := 
by sorry

end compare_A_B_l217_217820


namespace John_surveyed_total_people_l217_217226

theorem John_surveyed_total_people :
  ∃ P D : ℝ, 
  0 ≤ P ∧ 
  D = 0.868 * P ∧ 
  21 = 0.457 * D ∧ 
  P = 53 :=
by
  sorry

end John_surveyed_total_people_l217_217226


namespace possibleValues_set_l217_217984

noncomputable def possibleValues (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 3) : Set ℝ :=
  {x | x = 1/a + 1/b}

theorem possibleValues_set :
  ∀ a b : ℝ, (0 < a ∧ 0 < b) → (a + b = 3) → possibleValues a b (by sorry) (by sorry) = {x | ∃ y, y ≥ 4/3 ∧ x = y} :=
by
  sorry

end possibleValues_set_l217_217984


namespace machine_x_produces_40_percent_l217_217835

theorem machine_x_produces_40_percent (T X Y : ℝ) 
  (h1 : X + Y = T)
  (h2 : 0.009 * X + 0.004 * Y = 0.006 * T) :
  X = 0.4 * T :=
by
  sorry

end machine_x_produces_40_percent_l217_217835


namespace percentage_trucks_returned_l217_217068

theorem percentage_trucks_returned (total_trucks rented_trucks returned_trucks : ℕ)
  (h1 : total_trucks = 24)
  (h2 : rented_trucks = total_trucks)
  (h3 : returned_trucks ≥ 12)
  (h4 : returned_trucks ≤ total_trucks) :
  (returned_trucks / rented_trucks) * 100 = 50 :=
by sorry

end percentage_trucks_returned_l217_217068


namespace problem_statement_l217_217490

theorem problem_statement (x y z : ℝ) (h1 : x = 2) (h2 : y = -1) (h3 : z = 3) :
  x^2 + y^2 + z^2 + 2*x*z = 26 :=
by
  rw [h1, h2, h3]
  norm_num

end problem_statement_l217_217490


namespace inequality_reciprocal_l217_217470

theorem inequality_reciprocal (a b : ℝ) (hab : a < b) (hb : b < 0) : (1 / a) > (1 / b) :=
by
  sorry

end inequality_reciprocal_l217_217470


namespace drug_price_reduction_l217_217964

theorem drug_price_reduction (x : ℝ) :
    36 * (1 - x)^2 = 25 :=
sorry

end drug_price_reduction_l217_217964


namespace fraction_value_l217_217410

theorem fraction_value : (5 * 7 : ℝ) / 10 = 3.5 := by
  sorry

end fraction_value_l217_217410


namespace train_speed_l217_217084

theorem train_speed 
  (length : ℝ)
  (time : ℝ)
  (relative_speed : ℝ)
  (conversion_factor : ℝ)
  (h_length : length = 120)
  (h_time : time = 4)
  (h_relative_speed : relative_speed = 60)
  (h_conversion_factor : conversion_factor = 3.6) :
  (relative_speed / 2) * conversion_factor = 108 :=
by
  sorry

end train_speed_l217_217084


namespace range_of_a_l217_217257

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 :=
sorry

end range_of_a_l217_217257


namespace distinct_real_numbers_a_l217_217494

theorem distinct_real_numbers_a (a x y z : ℝ) (h_distinct: x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  (a = x + 1 / y ∧ a = y + 1 / z ∧ a = z + 1 / x) ↔ (a = 1 ∨ a = -1) :=
by sorry

end distinct_real_numbers_a_l217_217494


namespace determine_real_pairs_l217_217397

theorem determine_real_pairs (a b : ℝ) :
  (∀ n : ℕ+, a * ⌊ b * n ⌋ = b * ⌊ a * n ⌋) →
  (∃ c : ℝ, (a = 0 ∧ b = c) ∨ (a = c ∧ b = 0) ∨ (a = c ∧ b = c) ∨ (∃ k l : ℤ, a = k ∧ b = l)) :=
by
  sorry

end determine_real_pairs_l217_217397


namespace christian_age_in_years_l217_217375

theorem christian_age_in_years (B C x : ℕ) (h1 : C = 2 * B) (h2 : B + x = 40) (h3 : C + x = 72) :
    x = 8 := 
sorry

end christian_age_in_years_l217_217375


namespace axis_of_symmetry_center_of_symmetry_g_max_value_g_increasing_intervals_g_decreasing_intervals_l217_217705

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sqrt 3 * Real.sin (Real.pi - x) + 5 * Real.sin (Real.pi / 2 + x) + 5

theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, f x = f (Real.pi / 3 + k * Real.pi) :=
sorry

theorem center_of_symmetry :
  ∃ k : ℤ, f (k * Real.pi - Real.pi / 6) = 5 :=
sorry

noncomputable def g (x : ℝ) : ℝ := 10 * Real.sin (2 * x) - 8

theorem g_max_value :
  ∀ x : ℝ, g x ≤ 2 :=
sorry

theorem g_increasing_intervals :
  ∀ k : ℤ, -Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 4 + k * Real.pi → ∀ x : ℝ, g x ≤ g (x + 1) :=
sorry

theorem g_decreasing_intervals :
  ∀ k : ℤ, Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ 3 * Real.pi / 4 + k * Real.pi → ∀ x : ℝ, g x ≥ g (x + 1) :=
sorry

end axis_of_symmetry_center_of_symmetry_g_max_value_g_increasing_intervals_g_decreasing_intervals_l217_217705


namespace domestic_probability_short_haul_probability_long_haul_probability_l217_217484

variable (P_internet_domestic P_snacks_domestic P_entertainment_domestic P_legroom_domestic : ℝ)
variable (P_internet_short_haul P_snacks_short_haul P_entertainment_short_haul P_legroom_short_haul : ℝ)
variable (P_internet_long_haul P_snacks_long_haul P_entertainment_long_haul P_legroom_long_haul : ℝ)

noncomputable def P_domestic :=
  P_internet_domestic * P_snacks_domestic * P_entertainment_domestic * P_legroom_domestic

theorem domestic_probability :
  P_domestic 0.40 0.60 0.70 0.50 = 0.084 := by
  sorry

noncomputable def P_short_haul :=
  P_internet_short_haul * P_snacks_short_haul * P_entertainment_short_haul * P_legroom_short_haul

theorem short_haul_probability :
  P_short_haul 0.50 0.75 0.55 0.60 = 0.12375 := by
  sorry

noncomputable def P_long_haul :=
  P_internet_long_haul * P_snacks_long_haul * P_entertainment_long_haul * P_legroom_long_haul

theorem long_haul_probability :
  P_long_haul 0.65 0.80 0.75 0.70 = 0.273 := by
  sorry

end domestic_probability_short_haul_probability_long_haul_probability_l217_217484


namespace find_x_l217_217339

theorem find_x (x : ℝ) (h : 65 + 5 * 12 / (x / 3) = 66) : x = 180 :=
by
  sorry

end find_x_l217_217339


namespace total_hamburger_varieties_l217_217239

def num_condiments : ℕ := 9
def num_condiment_combinations : ℕ := 2 ^ num_condiments
def num_patties_choices : ℕ := 4
def num_bread_choices : ℕ := 2

theorem total_hamburger_varieties : num_condiment_combinations * num_patties_choices * num_bread_choices = 4096 :=
by
  -- conditions
  have h1 : num_condiments = 9 := rfl
  have h2 : num_condiment_combinations = 2 ^ num_condiments := rfl
  have h3 : num_patties_choices = 4 := rfl
  have h4 : num_bread_choices = 2 := rfl

  -- correct answer
  sorry

end total_hamburger_varieties_l217_217239


namespace find_marksman_hit_rate_l217_217014

-- Define the conditions
def independent_shots (p : ℝ) (n : ℕ) : Prop :=
  0 ≤ p ∧ p ≤ 1 ∧ (n ≥ 1)

def hit_probability (p : ℝ) (n : ℕ) : ℝ :=
  1 - (1 - p) ^ n

-- Stating the proof problem in Lean
theorem find_marksman_hit_rate (p : ℝ) (n : ℕ) 
  (h_independent : independent_shots p n) 
  (h_prob : hit_probability p n = 80 / 81) : 
  p = 2 / 3 :=
sorry

end find_marksman_hit_rate_l217_217014


namespace tank_plastering_cost_proof_l217_217075

/-- 
Given a tank with the following dimensions:
length = 35 meters,
width = 18 meters,
depth = 10 meters.
The cost of plastering per square meter is ₹135.
Prove that the total cost of plastering the walls and bottom of the tank is ₹228,150.
-/
theorem tank_plastering_cost_proof (length width depth cost_per_sq_meter : ℕ)
  (h_length : length = 35)
  (h_width : width = 18)
  (h_depth : depth = 10)
  (h_cost_per_sq_meter : cost_per_sq_meter = 135) : 
  (2 * (length * depth) + 2 * (width * depth) + length * width) * cost_per_sq_meter = 228150 := 
by 
  -- The proof is not required as per the problem statement
  sorry

end tank_plastering_cost_proof_l217_217075


namespace probability_point_in_circle_l217_217411

theorem probability_point_in_circle (r : ℝ) (h: r = 2) :
  let side_length := 2 * r
  let area_square := side_length ^ 2
  let area_circle := Real.pi * r ^ 2
  (area_circle / area_square) = Real.pi / 4 :=
by
  sorry

end probability_point_in_circle_l217_217411


namespace Arrow_velocity_at_impact_l217_217605

def Edward_initial_distance := 1875 -- \(\text{ft}\)
def Edward_initial_velocity := 0 -- \(\text{ft/s}\)
def Edward_acceleration := 1 -- \(\text{ft/s}^2\)
def Arrow_initial_distance := 0 -- \(\text{ft}\)
def Arrow_initial_velocity := 100 -- \(\text{ft/s}\)
def Arrow_deceleration := -1 -- \(\text{ft/s}^2\)
def time_impact := 25 -- \(\text{s}\)

theorem Arrow_velocity_at_impact : 
  (Arrow_initial_velocity + Arrow_deceleration * time_impact) = 75 := 
by
  sorry

end Arrow_velocity_at_impact_l217_217605


namespace find_original_price_l217_217634

-- Definitions based on Conditions
def original_price (P : ℝ) : Prop :=
  let increased_price := 1.25 * P
  let final_price := increased_price * 0.75
  final_price = 187.5

theorem find_original_price (P : ℝ) (h : original_price P) : P = 200 :=
  by sorry

end find_original_price_l217_217634


namespace least_common_addition_of_primes_l217_217912

theorem least_common_addition_of_primes (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (hxy : x < y) (h : 4 * x + y = 87) : x + y = 81 := 
sorry

end least_common_addition_of_primes_l217_217912


namespace negation_of_p_l217_217710

-- Define the proposition p
def proposition_p : Prop := ∀ x : ℝ, x^2 + 1 > 0

-- State the theorem: the negation of proposition p
theorem negation_of_p : ¬ proposition_p ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by 
  sorry

end negation_of_p_l217_217710


namespace nine_a_eq_frac_minus_eighty_one_over_eleven_l217_217622

theorem nine_a_eq_frac_minus_eighty_one_over_eleven (a b : ℚ) 
  (h1 : 8 * a + 3 * b = 0) 
  (h2 : a = b - 3) : 
  9 * a = -81 / 11 := 
sorry

end nine_a_eq_frac_minus_eighty_one_over_eleven_l217_217622


namespace circle_equation_exists_l217_217262

noncomputable def point (α : Type*) := {p : α × α // ∃ x y : α, p = (x, y)}

structure Circle (α : Type*) :=
(center : α × α)
(radius : α)

def passes_through (c : Circle ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1) ^ 2 + (p.2 - c.center.2) ^ 2 = c.radius ^ 2

theorem circle_equation_exists :
  ∃ (c : Circle ℝ),
    c.center = (-4, 3) ∧ c.radius = 5 ∧ passes_through c (-1, -1) ∧ passes_through c (-8, 0) ∧ passes_through c (0, 6) :=
by { sorry }

end circle_equation_exists_l217_217262


namespace perfect_square_trinomial_l217_217982

theorem perfect_square_trinomial (y : ℝ) (m : ℝ) : 
  (∃ b : ℝ, y^2 - m*y + 9 = (y + b)^2) → (m = 6 ∨ m = -6) :=
by
  intro h
  sorry

end perfect_square_trinomial_l217_217982


namespace bobby_pizzas_l217_217042

theorem bobby_pizzas (B : ℕ) (h_slices : (1 / 4 : ℝ) * B = 3) (h_slices_per_pizza : 6 > 0) :
  B / 6 = 2 := by
  sorry

end bobby_pizzas_l217_217042


namespace video_games_expenditure_l217_217831

theorem video_games_expenditure (allowance : ℝ) (books_expense : ℝ) (snacks_expense : ℝ) (clothes_expense : ℝ) 
    (initial_allowance : allowance = 50)
    (books_fraction : books_expense = 1 / 7 * allowance)
    (snacks_fraction : snacks_expense = 1 / 2 * allowance)
    (clothes_fraction : clothes_expense = 3 / 14 * allowance) :
    50 - (books_expense + snacks_expense + clothes_expense) = 7.15 :=
by
  sorry

end video_games_expenditure_l217_217831


namespace min_value_of_expression_l217_217610

open Real

theorem min_value_of_expression (x y z : ℝ) (h₁ : x + y + z = 1) (h₂ : x > 0) (h₃ : y > 0) (h₄ : z > 0) :
  (∃ a, (∀ x y z, a ≤ (1 / (x + y) + (x + y) / z)) ∧ a = 3) :=
by
  sorry

end min_value_of_expression_l217_217610


namespace a8_value_l217_217057

variable {an : ℕ → ℕ}

def S (n : ℕ) : ℕ := n ^ 2

theorem a8_value : an 8 = S 8 - S 7 := by
  sorry

end a8_value_l217_217057


namespace square_perimeter_l217_217809

theorem square_perimeter (s : ℝ) (h1 : ∀ r : ℝ, r = 2 * (s + s / 4)) (r_perimeter_eq_40 : r = 40) :
  4 * s = 64 := by
  sorry

end square_perimeter_l217_217809


namespace problem_statement_l217_217795

def f (x : ℝ) : ℝ := x^5 - x^3 + 1
def g (x : ℝ) : ℝ := x^2 - 2

theorem problem_statement (x1 x2 x3 x4 x5 : ℝ) 
  (h_roots : ∀ x, f x = 0 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5) :
  g x1 * g x2 * g x3 * g x4 * g x5 = -7 := 
sorry

end problem_statement_l217_217795


namespace boric_acid_solution_l217_217323

theorem boric_acid_solution
  (amount_first_solution: ℝ) (percentage_first_solution: ℝ)
  (amount_second_solution: ℝ) (percentage_second_solution: ℝ)
  (final_amount: ℝ) (final_percentage: ℝ)
  (h1: amount_first_solution = 15)
  (h2: percentage_first_solution = 0.01)
  (h3: amount_second_solution = 15)
  (h4: final_amount = 30)
  (h5: final_percentage = 0.03)
  : percentage_second_solution = 0.05 := 
by
  sorry

end boric_acid_solution_l217_217323


namespace can_measure_all_weights_l217_217567

def weights : List ℕ := [1, 3, 9, 27]

theorem can_measure_all_weights :
  (∀ n, 1 ≤ n ∧ n ≤ 40 → ∃ (a b c d : ℕ), a * 1 + b * 3 + c * 9 + d * 27 = n) ∧ 
  (∃ (a b c d : ℕ), a * 1 + b * 3 + c * 9 + d * 27 = 40) :=
by
  sorry

end can_measure_all_weights_l217_217567


namespace four_digit_div_90_count_l217_217386

theorem four_digit_div_90_count :
  ∃ n : ℕ, n = 10 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 → ab % 9 = 0 → 
  (10 * ab + 90) % 90 = 0 ∧ 1000 ≤ 10 * ab + 90 ∧ 10 * ab + 90 < 10000) :=
sorry

end four_digit_div_90_count_l217_217386


namespace number_of_liars_on_the_island_l217_217281

-- Definitions for the conditions
def isKnight (person : ℕ) : Prop := sorry -- Placeholder, we know knights always tell the truth
def isLiar (person : ℕ) : Prop := sorry -- Placeholder, we know liars always lie
def population := 1000
def villages := 10
def minInhabitantsPerVillage := 2

-- Definitional property: each islander claims that all other villagers in their village are liars
def claimsAllOthersAreLiars (islander : ℕ) (village : ℕ) : Prop := 
  ∀ (other : ℕ), (other ≠ islander) → (isLiar other)

-- Main statement in Lean
theorem number_of_liars_on_the_island : ∃ liars, liars = 990 :=
by
  have total_population := population
  have number_of_villages := villages
  have min_people_per_village := minInhabitantsPerVillage
  have knight_prop := isKnight
  have liar_prop := isLiar
  have claim_prop := claimsAllOthersAreLiars
  -- Proof will be filled here
  sorry

end number_of_liars_on_the_island_l217_217281


namespace odd_function_increasing_on_negative_interval_l217_217612

theorem odd_function_increasing_on_negative_interval {f : ℝ → ℝ}
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, 3 ≤ x → x ≤ 7 → 3 ≤ y → y ≤ 7 → x < y → f x < f y)
  (h_min_value : f 3 = 1) :
  (∀ x y, -7 ≤ x → x ≤ -3 → -7 ≤ y → y ≤ -3 → x < y → f x < f y) ∧ f (-3) = -1 := 
sorry

end odd_function_increasing_on_negative_interval_l217_217612


namespace circle_equation_through_ABC_circle_equation_with_center_and_points_l217_217854

-- Define points
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 0⟩
def B : Point := ⟨4, 0⟩
def C : Point := ⟨6, -2⟩

-- First problem: proof of the circle equation given points A, B, and C
theorem circle_equation_through_ABC :
  ∃ (D E F : ℝ), 
  (∀ (P : Point), (P = A ∨ P = B ∨ P = C) → P.x^2 + P.y^2 + D * P.x + E * P.y + F = 0) 
  ↔ (D = -5 ∧ E = 7 ∧ F = 4) := sorry

-- Second problem: proof of the circle equation given the y-coordinate of the center and points A and B
theorem circle_equation_with_center_and_points :
  ∃ (h k r : ℝ), 
  (h = (A.x + B.x) / 2 ∧ k = 2) ∧
  ∀ (P : Point), (P = A ∨ P = B) → (P.x - h)^2 + (P.y - k)^2 = r^2
  ↔ (h = 5 / 2 ∧ k = 2 ∧ r = 5 / 2) := sorry

end circle_equation_through_ABC_circle_equation_with_center_and_points_l217_217854


namespace expected_value_winnings_l217_217606

def probability_heads : ℚ := 2 / 5
def probability_tails : ℚ := 3 / 5
def win_amount_heads : ℚ := 5
def lose_amount_tails : ℚ := -4

theorem expected_value_winnings : 
  probability_heads * win_amount_heads + probability_tails * lose_amount_tails = -2 / 5 := 
by 
  sorry

end expected_value_winnings_l217_217606


namespace base_five_product_l217_217962

theorem base_five_product (n1 n2 : ℕ) (h1 : n1 = 1 * 5^2 + 3 * 5^1 + 1 * 5^0) 
                          (h2 : n2 = 1 * 5^1 + 2 * 5^0) :
  let product_dec := (n1 * n2 : ℕ)
  let product_base5 := 2 * 125 + 1 * 25 + 2 * 5 + 2 * 1
  product_dec = 287 ∧ product_base5 = 2122 := by
                                -- calculations to verify statement omitted
                                sorry

end base_five_product_l217_217962


namespace correct_average_is_26_l217_217122

noncomputable def initial_average : ℕ := 20
noncomputable def number_of_numbers : ℕ := 10
noncomputable def incorrect_number : ℕ := 26
noncomputable def correct_number : ℕ := 86
noncomputable def incorrect_total_sum : ℕ := initial_average * number_of_numbers
noncomputable def correct_total_sum : ℕ := incorrect_total_sum + (correct_number - incorrect_number)
noncomputable def correct_average : ℕ := correct_total_sum / number_of_numbers

theorem correct_average_is_26 :
  correct_average = 26 := by
  sorry

end correct_average_is_26_l217_217122


namespace cab_driver_income_l217_217446

theorem cab_driver_income (x : ℕ)
  (h1 : 50 + 60 + 65 + 70 + x = 5 * 58) :
  x = 45 :=
by
  sorry

end cab_driver_income_l217_217446


namespace orlie_age_l217_217179

theorem orlie_age (O R : ℕ) (h1 : R = 9) (h2 : R = (3 * O) / 4)
  (h3 : R - 4 = ((O - 4) / 2) + 1) : O = 12 :=
by
  sorry

end orlie_age_l217_217179


namespace minimum_value_of_f_l217_217321

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem minimum_value_of_f :
  f 2 = -3 ∧ (∀ x : ℝ, f x ≥ -3) :=
by
  sorry

end minimum_value_of_f_l217_217321


namespace product_decrease_increase_fifteenfold_l217_217576

theorem product_decrease_increase_fifteenfold (a1 a2 a3 a4 a5 : ℕ) :
  ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * a1 * a2 * a3 * a4 * a5) → true :=
by
  sorry

end product_decrease_increase_fifteenfold_l217_217576


namespace tiling_polygons_l217_217126

theorem tiling_polygons (n : ℕ) (h1 : 2 < n) (h2 : ∃ x : ℕ, x * (((n - 2) * 180 : ℝ) / n) = 360) :
  n = 3 ∨ n = 4 ∨ n = 6 := 
by
  sorry

end tiling_polygons_l217_217126


namespace non_sophomores_is_75_percent_l217_217292

def students_not_sophomores_percentage (total_students : ℕ) 
                                       (percent_juniors : ℚ)
                                       (num_seniors : ℕ)
                                       (freshmen_more_than_sophomores : ℕ) : ℚ :=
  let num_juniors := total_students * percent_juniors 
  let s := (total_students - num_juniors - num_seniors - freshmen_more_than_sophomores) / 2
  let f := s + freshmen_more_than_sophomores
  let non_sophomores := total_students - s
  (non_sophomores / total_students) * 100

theorem non_sophomores_is_75_percent : students_not_sophomores_percentage 800 0.28 160 16 = 75 := by
  sorry

end non_sophomores_is_75_percent_l217_217292


namespace mod_sum_l217_217577

theorem mod_sum : 
  (5432 + 5433 + 5434 + 5435) % 7 = 2 := 
by
  sorry

end mod_sum_l217_217577


namespace trapezoid_area_l217_217701

theorem trapezoid_area (x : ℝ) (y : ℝ) :
  (∀ x, y = x + 1) →
  (∀ y, y = 12) →
  (∀ y, y = 7) →
  (∀ x, x = 0) →
  ∃ area,
  area = (1/2) * (6 + 11) * 5 ∧ area = 42.5 :=
by {
  sorry
}

end trapezoid_area_l217_217701


namespace questionnaires_drawn_l217_217975

theorem questionnaires_drawn
  (units : ℕ → ℕ)
  (h_arithmetic : ∀ n, units (n + 1) - units n = units 1 - units 0)
  (h_total : units 0 + units 1 + units 2 + units 3 = 100)
  (h_unitB : units 1 = 20) :
  units 3 = 40 :=
by
  -- Proof would go here
  -- Establish that the arithmetic sequence difference is 10, then compute unit D (units 3)
  sorry

end questionnaires_drawn_l217_217975


namespace range_of_a_in_second_quadrant_l217_217861

theorem range_of_a_in_second_quadrant :
  (∀ (x y : ℝ), x^2 + y^2 + 6*x - 4*a*y + 3*a^2 + 9 = 0 → x < 0 ∧ y > 0) → (0 < a ∧ a < 3) :=
by
  sorry

end range_of_a_in_second_quadrant_l217_217861


namespace lcm_16_24_l217_217280

/-
  Prove that the least common multiple (LCM) of 16 and 24 is 48.
-/
theorem lcm_16_24 : Nat.lcm 16 24 = 48 :=
by
  sorry

end lcm_16_24_l217_217280


namespace six_digit_number_l217_217974

/-- 
Find a six-digit number that starts with the digit 1 and such that if this digit is moved to the end, the resulting number is three times the original number.
-/
theorem six_digit_number (N : ℕ) (h₁ : 100000 ≤ N ∧ N < 1000000) (h₂ : ∃ x : ℕ, N = 1 * 10^5 + x ∧ 10 * x + 1 = 3 * N) : N = 142857 :=
by sorry

end six_digit_number_l217_217974


namespace smallest_scalene_triangle_perimeter_l217_217976

-- Define what it means for a number to be a prime number greater than 3
def prime_gt_3 (n : ℕ) : Prop := Prime n ∧ 3 < n

-- Define the main theorem
theorem smallest_scalene_triangle_perimeter : ∃ (a b c : ℕ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  prime_gt_3 a ∧ prime_gt_3 b ∧ prime_gt_3 c ∧
  Prime (a + b + c) ∧ 
  (∀ (x y z : ℕ), 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    prime_gt_3 x ∧ prime_gt_3 y ∧ prime_gt_3 z ∧
    Prime (x + y + z) → (a + b + c) ≤ (x + y + z)) ∧
  a + b + c = 23 := by
    sorry

end smallest_scalene_triangle_perimeter_l217_217976


namespace range_of_m_l217_217620

variable {x m : ℝ}

theorem range_of_m (h1 : x + 2 < 2 * m) (h2 : x - m < 0) (h3 : x < 2 * m - 2) : m ≤ 2 :=
sorry

end range_of_m_l217_217620


namespace kim_initial_classes_l217_217071

-- Necessary definitions for the problem
def hours_per_class := 2
def total_hours_after_dropping := 6
def classes_after_dropping := total_hours_after_dropping / hours_per_class
def initial_classes := classes_after_dropping + 1

theorem kim_initial_classes : initial_classes = 4 :=
by
  -- Proof will be derived here
  sorry

end kim_initial_classes_l217_217071


namespace number_of_blue_candles_l217_217956

def total_candles : ℕ := 79
def yellow_candles : ℕ := 27
def red_candles : ℕ := 14
def blue_candles : ℕ := total_candles - (yellow_candles + red_candles)

theorem number_of_blue_candles : blue_candles = 38 :=
by
  unfold blue_candles
  unfold total_candles yellow_candles red_candles
  sorry

end number_of_blue_candles_l217_217956


namespace rectangle_relationships_l217_217465

theorem rectangle_relationships (x y S : ℝ) (h1 : 2 * x + 2 * y = 10) (h2 : S = x * y) :
  y = 5 - x ∧ S = 5 * x - x ^ 2 :=
by
  sorry

end rectangle_relationships_l217_217465


namespace roller_skate_wheels_l217_217752

theorem roller_skate_wheels (number_of_people : ℕ)
  (feet_per_person : ℕ)
  (skates_per_foot : ℕ)
  (wheels_per_skate : ℕ)
  (h_people : number_of_people = 40)
  (h_feet : feet_per_person = 2)
  (h_skates : skates_per_foot = 1)
  (h_wheels : wheels_per_skate = 4)
  : (number_of_people * feet_per_person * skates_per_foot * wheels_per_skate) = 320 := 
by
  sorry

end roller_skate_wheels_l217_217752


namespace find_a_2018_l217_217105

noncomputable def a : ℕ → ℕ
| n => if n > 0 then 2 * n else sorry

theorem find_a_2018 (a : ℕ → ℕ) 
  (h : ∀ m n : ℕ, m > 0 ∧ n > 0 → a m + a n = a (m + n)) 
  (h1 : a 1 = 2) : a 2018 = 4036 := by
  sorry

end find_a_2018_l217_217105


namespace k_ge_1_l217_217177

theorem k_ge_1 (k : ℝ) : 
  (∀ x : ℝ, 2 * x + 9 > 6 * x + 1 ∧ x - k < 1 → x < 2) → k ≥ 1 :=
by 
  sorry

end k_ge_1_l217_217177


namespace number_difference_l217_217120

theorem number_difference:
  ∀ (number : ℝ), 0.30 * number = 63.0000000000001 →
  (3 / 7) * number - 0.40 * number = 6.00000000000006 := by
  sorry

end number_difference_l217_217120


namespace leila_total_cakes_l217_217021

def cakes_monday : ℕ := 6
def cakes_friday : ℕ := 9
def cakes_saturday : ℕ := 3 * cakes_monday
def total_cakes : ℕ := cakes_monday + cakes_friday + cakes_saturday

theorem leila_total_cakes : total_cakes = 33 := by
  sorry

end leila_total_cakes_l217_217021


namespace sequence_b_l217_217301

theorem sequence_b (b : ℕ → ℕ) 
  (h1 : b 1 = 2) 
  (h2 : ∀ m n : ℕ, b (m + n) = b m + b n + 2 * m * n) : 
  b 10 = 110 :=
sorry

end sequence_b_l217_217301


namespace count_distinct_m_values_l217_217026

theorem count_distinct_m_values : 
  ∃ m_values : Finset ℤ, 
  (∀ x1 x2 : ℤ, x1 * x2 = 30 → (m_values : Set ℤ) = { x1 + x2 }) ∧ 
  m_values.card = 8 :=
by
  sorry

end count_distinct_m_values_l217_217026


namespace female_democrats_count_l217_217806

-- Define the parameters and conditions
variables (F M D_f D_m D_total : ℕ)
variables (h1 : F + M = 840)
variables (h2 : D_total = 1/3 * (F + M))
variables (h3 : D_f = 1/2 * F)
variables (h4 : D_m = 1/4 * M)
variables (h5 : D_total = D_f + D_m)

-- State the theorem
theorem female_democrats_count : D_f = 140 :=
by
  sorry

end female_democrats_count_l217_217806


namespace market_value_of_stock_l217_217263

theorem market_value_of_stock (dividend_rate : ℝ) (yield_rate : ℝ) (face_value : ℝ) :
  dividend_rate = 0.12 → yield_rate = 0.08 → face_value = 100 → (dividend_rate * face_value / yield_rate * 100) = 150 :=
by
  intros h1 h2 h3
  sorry

end market_value_of_stock_l217_217263


namespace find_a_l217_217533

theorem find_a (x a a1 a2 a3 a4 : ℝ) :
  (x + a) ^ 4 = x ^ 4 + a1 * x ^ 3 + a2 * x ^ 2 + a3 * x + a4 → 
  a1 + a2 + a3 = 64 → a = 2 :=
by
  sorry

end find_a_l217_217533


namespace consecutive_integers_product_l217_217778

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l217_217778


namespace fourth_number_is_2_eighth_number_is_2_l217_217933

-- Conditions as given in the problem
def initial_board := [1]

/-- Medians recorded in Mitya's notebook for the first 10 numbers -/
def medians := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

/-- Prove that the fourth number written on the board is 2 given initial conditions. -/
theorem fourth_number_is_2 (board : ℕ → ℤ)  
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 3 = 2 :=
sorry

/-- Prove that the eighth number written on the board is 2 given initial conditions. -/
theorem eighth_number_is_2 (board : ℕ → ℤ) 
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 7 = 2 :=
sorry

end fourth_number_is_2_eighth_number_is_2_l217_217933


namespace remainder_when_c_divided_by_b_eq_2_l217_217081

theorem remainder_when_c_divided_by_b_eq_2 
(a b c : ℕ) 
(hb : b = 3 * a + 3) 
(hc : c = 9 * a + 11) : 
  c % b = 2 := 
sorry

end remainder_when_c_divided_by_b_eq_2_l217_217081


namespace ratio_Y_to_Z_l217_217390

variables (X Y Z : ℕ)

def population_relation1 (X Y : ℕ) : Prop := X = 3 * Y
def population_relation2 (X Z : ℕ) : Prop := X = 6 * Z

theorem ratio_Y_to_Z (h1 : population_relation1 X Y) (h2 : population_relation2 X Z) : Y / Z = 2 :=
  sorry

end ratio_Y_to_Z_l217_217390


namespace dots_not_visible_l217_217570

-- Define the sum of numbers on a single die
def sum_die_faces : ℕ := 1 + 2 + 3 + 4 + 5 + 6

-- Define the sum of numbers on four dice
def total_dots_on_four_dice : ℕ := 4 * sum_die_faces

-- List the visible numbers
def visible_numbers : List ℕ := [1, 2, 2, 3, 3, 4, 5, 5, 6]

-- Calculate the sum of visible numbers
def sum_visible_numbers : ℕ := (visible_numbers.sum)

-- Define the math proof problem
theorem dots_not_visible : total_dots_on_four_dice - sum_visible_numbers = 53 := by
  sorry

end dots_not_visible_l217_217570


namespace time_to_overflow_equals_correct_answer_l217_217017

-- Definitions based on conditions
def pipeA_fill_time : ℚ := 32
def pipeB_fill_time : ℚ := pipeA_fill_time / 5

-- Derived rates from the conditions
def pipeA_rate : ℚ := 1 / pipeA_fill_time
def pipeB_rate : ℚ := 1 / pipeB_fill_time
def combined_rate : ℚ := pipeA_rate + pipeB_rate

-- The time to overflow when both pipes are filling the tank simultaneously
def time_to_overflow : ℚ := 1 / combined_rate

-- The statement we are going to prove
theorem time_to_overflow_equals_correct_answer : time_to_overflow = 16 / 3 :=
by sorry

end time_to_overflow_equals_correct_answer_l217_217017


namespace solve_inequality_l217_217574

theorem solve_inequality (a x : ℝ) :
  (a - x) * (x - 1) < 0 ↔
  (a > 1 ∧ (x < 1 ∨ x > a)) ∨
  (a < 1 ∧ (x < a ∨ x > 1)) ∨
  (a = 1 ∧ x ≠ 1) :=
by
  sorry

end solve_inequality_l217_217574


namespace A_inter_B_is_correct_l217_217468

def set_A : Set ℤ := { x : ℤ | x^2 - x - 2 ≤ 0 }
def set_B : Set ℤ := { x : ℤ | True }

theorem A_inter_B_is_correct : set_A ∩ set_B = { -1, 0, 1, 2 } := by
  sorry

end A_inter_B_is_correct_l217_217468


namespace percentage_problem_l217_217751

theorem percentage_problem
    (x : ℕ) (h1 : (x:ℝ) / 100 * 20 = 8) :
    x = 40 :=
by
    sorry

end percentage_problem_l217_217751


namespace greatest_prime_factor_of_341_l217_217536

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l217_217536


namespace xy_value_l217_217110

theorem xy_value (x y : ℝ) (h : x * (x - y) = x^2 - 6) : x * y = 6 := 
by 
  sorry

end xy_value_l217_217110


namespace exponentiation_identity_l217_217629

variable {a : ℝ}

theorem exponentiation_identity : (-a) ^ 2 * a ^ 3 = a ^ 5 := sorry

end exponentiation_identity_l217_217629


namespace proper_subsets_B_l217_217099

theorem proper_subsets_B (A B : Set ℝ) (a : ℝ)
  (hA : A = {x | x^2 + 2*x + 1 = 0})
  (hA_singleton : A = {a})
  (hB : B = {x | x^2 + a*x = 0}) :
  a = -1 ∧ 
  B = {0, 1} ∧
  (∀ S, S ∈ ({∅, {0}, {1}} : Set (Set ℝ)) ↔ S ⊂ B) :=
by
  -- Proof not provided, only statement required.
  sorry

end proper_subsets_B_l217_217099


namespace average_difference_l217_217398

theorem average_difference (F1 L1 F2 L2 : ℤ) (H1 : F1 = 200) (H2 : L1 = 400) (H3 : F2 = 100) (H4 : L2 = 200) :
  (F1 + L1) / 2 - (F2 + L2) / 2 = 150 := 
by 
  sorry

end average_difference_l217_217398


namespace number_of_workers_in_each_block_is_200_l217_217665

-- Conditions
def total_amount : ℕ := 6000
def worth_of_each_gift : ℕ := 2
def number_of_blocks : ℕ := 15

-- Question and answer to be proven
def number_of_workers_in_each_block : ℕ := total_amount / worth_of_each_gift / number_of_blocks

theorem number_of_workers_in_each_block_is_200 :
  number_of_workers_in_each_block = 200 :=
by
  -- Skip the proof with sorry
  sorry

end number_of_workers_in_each_block_is_200_l217_217665


namespace work_completion_days_l217_217748

theorem work_completion_days (A B : ℕ) (h1 : A = 2 * B) (h2 : 6 * (A + B) = 18) : B = 1 → 18 = 18 :=
by
  sorry

end work_completion_days_l217_217748


namespace elizabeth_stickers_l217_217599

def total_stickers (initial_bottles lost_bottles stolen_bottles stickers_per_bottle : ℕ) : ℕ :=
  let remaining_bottles := initial_bottles - lost_bottles - stolen_bottles
  remaining_bottles * stickers_per_bottle

theorem elizabeth_stickers :
  total_stickers 10 2 1 3 = 21 :=
by
  sorry

end elizabeth_stickers_l217_217599


namespace yuan_exchange_l217_217117

theorem yuan_exchange : 
  ∃ (n : ℕ), n = 5 ∧ ∀ (x y : ℕ), x + 5 * y = 20 → x ≥ 0 ∧ y ≥ 0 :=
by {
  sorry
}

end yuan_exchange_l217_217117


namespace side_length_of_square_l217_217986

theorem side_length_of_square (d : ℝ) (h : d = 4) : ∃ (s : ℝ), s = 2 * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l217_217986


namespace remaining_distance_proof_l217_217008

-- Define the conditions
def pascal_current_speed : ℝ := 8
def pascal_reduced_speed : ℝ := pascal_current_speed - 4
def pascal_increased_speed : ℝ := pascal_current_speed * 1.5

-- Define the remaining distance in terms of the current speed and time taken
noncomputable def remaining_distance (T : ℝ) : ℝ := pascal_current_speed * T

-- Define the times with the increased and reduced speeds
noncomputable def time_with_increased_speed (T : ℝ) : ℝ := T - 16
noncomputable def time_with_reduced_speed (T : ℝ) : ℝ := T + 16

-- Define the distances using increased and reduced speeds
noncomputable def distance_increased_speed (T : ℝ) : ℝ := pascal_increased_speed * (time_with_increased_speed T)
noncomputable def distance_reduced_speed (T : ℝ) : ℝ := pascal_reduced_speed * (time_with_reduced_speed T)

-- Main theorem stating that the remaining distance is 256 miles
theorem remaining_distance_proof (T : ℝ) (ht_eq: pascal_current_speed * T = 256) : 
  remaining_distance T = 256 := by
  sorry

end remaining_distance_proof_l217_217008


namespace range_of_f_l217_217189

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((1 - x) / (1 + x)) + Real.arctan (2 * x)

theorem range_of_f : Set.Ioo (-(Real.pi / 2)) (Real.pi / 2) = Set.range f :=
  sorry

end range_of_f_l217_217189


namespace dennis_floor_l217_217072

theorem dennis_floor :
  ∃ d c b f e: ℕ, 
  (d = c + 2) ∧ 
  (c = b + 1) ∧ 
  (c = f / 4) ∧ 
  (f = 16) ∧ 
  (e = d / 2) ∧ 
  (d = 6) :=
by
  sorry

end dennis_floor_l217_217072


namespace circle_tangent_to_xaxis_at_origin_l217_217668

theorem circle_tangent_to_xaxis_at_origin (G E F : ℝ)
  (h : ∀ x y: ℝ, x^2 + y^2 + G*x + E*y + F = 0 → y = 0 ∧ x = 0 ∧ 0 < E) :
  G = 0 ∧ F = 0 ∧ E ≠ 0 :=
by
  sorry

end circle_tangent_to_xaxis_at_origin_l217_217668


namespace range_of_a_l217_217260

noncomputable def f (a x : ℝ) : ℝ := Real.log (a * x ^ 2 + 2 * x + 1)

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, a * x ^ 2 + 2 * x + 1 > 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end range_of_a_l217_217260


namespace area_of_rectangle_is_108_l217_217434

-- Define the conditions and parameters
variables (P Q R S : Type) (diameter : ℝ) (height : ℝ) (width : ℝ) (area : ℝ)
variable (isTangentToSides : Prop)
variable (centersFormLineParallelToLongerSide : Prop)

-- Assume the given conditions
axiom h1 : diameter = 6
axiom h2 : isTangentToSides
axiom h3 : centersFormLineParallelToLongerSide

-- Define the goal to prove
theorem area_of_rectangle_is_108 (P Q R S : Type) (diameter : ℝ) (height : ℝ) (width : ℝ) (area : ℝ)
    (isTangentToSides : Prop) (centersFormLineParallelToLongerSide : Prop)
    (h1 : diameter = 6)
    (h2 : isTangentToSides)
    (h3 : centersFormLineParallelToLongerSide) :
    area = 108 :=
by
  -- Lean code requires an actual proof here, but for now, we'll use sorry.
  sorry

end area_of_rectangle_is_108_l217_217434


namespace union_A_B_m_eq_3_range_of_m_l217_217178

def A (x : ℝ) : Prop := x^2 - x - 12 ≤ 0
def B (x m : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem union_A_B_m_eq_3 :
  A x ∨ B x 3 ↔ (-3 : ℝ) ≤ x ∧ x ≤ 5 := sorry

theorem range_of_m (h : ∀ x, A x ∨ B x m ↔ A x) : m ≤ (5 / 2) := sorry

end union_A_B_m_eq_3_range_of_m_l217_217178


namespace simple_interest_borrowed_rate_l217_217894

theorem simple_interest_borrowed_rate
  (P_borrowed P_lent : ℝ)
  (n_years : ℕ)
  (gain_per_year : ℝ)
  (simple_interest_lent_rate : ℝ)
  (SI_lending : ℝ := P_lent * simple_interest_lent_rate * n_years / 100)
  (total_gain : ℝ := gain_per_year * n_years) :
  SI_lending = 1000 →
  total_gain = 100 →
  ∀ (SI_borrowing : ℝ), SI_borrowing = SI_lending - total_gain →
  ∀ (R_borrowed : ℝ), SI_borrowing = P_borrowed * R_borrowed * n_years / 100 →
  R_borrowed = 9 := 
by
  sorry

end simple_interest_borrowed_rate_l217_217894


namespace largest_inscribed_square_length_l217_217876

noncomputable def inscribed_square_length (s : ℝ) (n : ℕ) : ℝ :=
  let t := s / n
  let h := (Real.sqrt 3 / 2) * t
  s - 2 * h

theorem largest_inscribed_square_length :
  inscribed_square_length 12 3 = 12 - 4 * Real.sqrt 3 :=
by
  sorry

end largest_inscribed_square_length_l217_217876


namespace percentage_gain_is_20_percent_l217_217138

theorem percentage_gain_is_20_percent (manufacturing_cost transportation_cost total_shoes selling_price : ℝ)
(h1 : manufacturing_cost = 220)
(h2 : transportation_cost = 500)
(h3 : total_shoes = 100)
(h4 : selling_price = 270) :
  let cost_per_shoe := manufacturing_cost + transportation_cost / total_shoes
  let profit_per_shoe := selling_price - cost_per_shoe
  let percentage_gain := (profit_per_shoe / cost_per_shoe) * 100
  percentage_gain = 20 :=
by
  sorry

end percentage_gain_is_20_percent_l217_217138


namespace weight_of_b_l217_217632

theorem weight_of_b (a b c : ℝ) (h1 : (a + b + c) / 3 = 45) (h2 : (a + b) / 2 = 40) (h3 : (b + c) / 2 = 43) : b = 31 :=
by
  sorry

end weight_of_b_l217_217632


namespace tan_plus_pi_over_4_l217_217380

variable (θ : ℝ)

-- Define the conditions
def condition_θ_interval : Prop := θ ∈ Set.Ioo (Real.pi / 2) Real.pi
def condition_sin_θ : Prop := Real.sin θ = 3 / 5

-- Define the theorem to be proved
theorem tan_plus_pi_over_4 (h1 : condition_θ_interval θ) (h2 : condition_sin_θ θ) :
  Real.tan (θ + Real.pi / 4) = 7 :=
sorry

end tan_plus_pi_over_4_l217_217380


namespace jane_sandwich_count_l217_217942

noncomputable def total_sandwiches : ℕ := 5 * 7 * 4

noncomputable def turkey_swiss_reduction : ℕ := 5 * 1 * 1

noncomputable def salami_bread_reduction : ℕ := 5 * 1 * 4

noncomputable def correct_sandwich_count : ℕ := 115

theorem jane_sandwich_count : total_sandwiches - turkey_swiss_reduction - salami_bread_reduction = correct_sandwich_count :=
by
  sorry

end jane_sandwich_count_l217_217942


namespace linear_equation_solution_l217_217402

theorem linear_equation_solution (a : ℝ) (x y : ℝ) 
    (h : (a - 2) * x^(|a| - 1) + 3 * y = 1) 
    (h1 : ∀ (x y : ℝ), (a - 2) ≠ 0)
    (h2 : |a| - 1 = 1) : a = -2 :=
by
  sorry

end linear_equation_solution_l217_217402


namespace a2_a3_equals_20_l217_217100

-- Sequence definition
def a_n (n : ℕ) : ℕ :=
  if n % 2 = 1 then 3 * n + 1 else 2 * n - 2

-- Proof that a_2 * a_3 = 20
theorem a2_a3_equals_20 :
  a_n 2 * a_n 3 = 20 :=
by
  sorry

end a2_a3_equals_20_l217_217100


namespace measure_angle_A_l217_217857

open Real

def triangle_area (a b c S : ℝ) (A B C : ℝ) : Prop :=
  S = (1 / 2) * b * c * sin A

def sides_and_angles (a b c A B C : ℝ) : Prop :=
  A = 2 * B

theorem measure_angle_A (a b c S A B C : ℝ)
  (h1 : triangle_area a b c S A B C)
  (h2 : sides_and_angles a b c A B C)
  (h3 : S = (a ^ 2) / 4) :
  A = π / 2 ∨ A = π / 4 :=
  sorry

end measure_angle_A_l217_217857


namespace complementary_event_probability_l217_217503

-- Define A and B as events such that B is the complement of A.
section
variables (A B : Prop) -- A and B are propositions representing events.
variable (P : Prop → ℝ) -- P is a function that gives the probability of an event.

-- Define the conditions for the problem.
variable (h_complementary : ∀ A B, A ∧ B = false ∧ A ∨ B = true) 
variable (h_PA : P A = 1 / 5)

-- The statement to be proved.
theorem complementary_event_probability : P B = 4 / 5 :=
by
  -- Here we would provide the proof, but for now, we use 'sorry' to bypass it.
  sorry
end

end complementary_event_probability_l217_217503


namespace total_spent_correct_l217_217090

def cost_gifts : ℝ := 561.00
def cost_giftwrapping : ℝ := 139.00
def total_spent : ℝ := cost_gifts + cost_giftwrapping

theorem total_spent_correct : total_spent = 700.00 := by
  sorry

end total_spent_correct_l217_217090


namespace fill_time_first_and_fourth_taps_l217_217109

noncomputable def pool_filling_time (m x y z u : ℝ) (h₁ : 2 * (x + y) = m) (h₂ : 3 * (y + z) = m) (h₃ : 4 * (z + u) = m) : ℝ :=
  m / (x + u)

theorem fill_time_first_and_fourth_taps (m x y z u : ℝ) (h₁ : 2 * (x + y) = m) (h₂ : 3 * (y + z) = m) (h₃ : 4 * (z + u) = m) :
  pool_filling_time m x y z u h₁ h₂ h₃ = 12 / 5 :=
sorry

end fill_time_first_and_fourth_taps_l217_217109


namespace find_a_value_l217_217918

theorem find_a_value
  (a : ℝ)
  (h : ∀ x, 0 ≤ x ∧ x ≤ (π / 2) → a * Real.sin x + Real.cos x ≤ 2)
  (h_max : ∃ x, 0 ≤ x ∧ x ≤ (π / 2) ∧ a * Real.sin x + Real.cos x = 2) :
  a = Real.sqrt 3 :=
sorry

end find_a_value_l217_217918


namespace gcd_calculation_l217_217364

def gcd_36_45_495 : ℕ :=
  Int.gcd 36 (Int.gcd 45 495)

theorem gcd_calculation : gcd_36_45_495 = 9 := by
  sorry

end gcd_calculation_l217_217364


namespace constant_term_in_binomial_expansion_max_coef_sixth_term_l217_217330

theorem constant_term_in_binomial_expansion_max_coef_sixth_term 
  (n : ℕ) (h : n = 10) : 
  (∃ C : ℕ → ℕ → ℕ, C 10 2 * (Nat.sqrt 2) ^ 8 = 720) :=
sorry

end constant_term_in_binomial_expansion_max_coef_sixth_term_l217_217330


namespace unique_parallel_line_in_beta_l217_217790

-- Define the basic geometrical entities.
axiom Plane : Type
axiom Line : Type
axiom Point : Type

-- Definitions relating entities.
def contains (P : Plane) (l : Line) : Prop := sorry
def parallel (A B : Plane) : Prop := sorry
def in_plane (p : Point) (P : Plane) : Prop := sorry
def parallel_lines (a b : Line) : Prop := sorry

-- Statements derived from the conditions in problem.
variables (α β : Plane) (a : Line) (B : Point)
-- Given conditions
axiom plane_parallel : parallel α β
axiom line_in_plane : contains α a
axiom point_in_plane : in_plane B β

-- The ultimate goal derived from the question.
theorem unique_parallel_line_in_beta : 
  ∃! b : Line, (in_plane B β) ∧ (parallel_lines a b) :=
sorry

end unique_parallel_line_in_beta_l217_217790


namespace area_of_triangle_AEB_l217_217193

noncomputable def rectangle_area_AEB : ℝ :=
  let AB := 8
  let BC := 4
  let DF := 2
  let GC := 2
  let FG := 8 - DF - GC -- DC (8 units) minus DF and GC.
  let ratio := AB / FG
  let altitude_AEB := BC * ratio
  let area_AEB := 0.5 * AB * altitude_AEB
  area_AEB

theorem area_of_triangle_AEB : rectangle_area_AEB = 32 :=
by
  -- placeholder for detailed proof
  sorry

end area_of_triangle_AEB_l217_217193


namespace dishes_combinations_is_correct_l217_217079

-- Define the number of dishes
def num_dishes : ℕ := 15

-- Define the number of appetizers
def num_appetizers : ℕ := 5

-- Compute the total number of combinations
def combinations_of_dishes : ℕ :=
  num_dishes * num_dishes * num_appetizers

-- The theorem that states the total number of combinations is 1125
theorem dishes_combinations_is_correct :
  combinations_of_dishes = 1125 := by
  sorry

end dishes_combinations_is_correct_l217_217079


namespace average_growth_rate_of_second_brand_l217_217345

theorem average_growth_rate_of_second_brand 
  (init1 : ℝ) (rate1 : ℝ) (init2 : ℝ) (t : ℝ) (r : ℝ)
  (h1 : init1 = 4.9) (h2 : rate1 = 0.275) (h3 : init2 = 2.5) (h4 : t = 5.647)
  (h_eq : init1 + rate1 * t = init2 + r * t) : 
  r = 0.7 :=
by 
  -- proof steps would go here
  sorry

end average_growth_rate_of_second_brand_l217_217345


namespace rationalize_denominator_correct_l217_217950

noncomputable def rationalize_denominator : ℚ := 
  let A := 5
  let B := 49
  let C := 21
  -- Form is (5 * ∛49) / 21
  A + B + C

theorem rationalize_denominator_correct : rationalize_denominator = 75 :=
  by 
    -- The proof steps are omitted, as they are not required for this task
    sorry

end rationalize_denominator_correct_l217_217950


namespace smallest_even_natural_number_l217_217328

theorem smallest_even_natural_number (a : ℕ) :
  ( ∃ a, a % 2 = 0 ∧
    (a + 1) % 3 = 0 ∧
    (a + 2) % 5 = 0 ∧
    (a + 3) % 7 = 0 ∧
    (a + 4) % 11 = 0 ∧
    (a + 5) % 13 = 0 ) → 
  a = 788 := by
  sorry

end smallest_even_natural_number_l217_217328


namespace average_remaining_two_numbers_l217_217420

theorem average_remaining_two_numbers 
  (a1 a2 a3 a4 a5 a6 : ℝ)
  (h_avg_6 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 2.80)
  (h_avg_2_1 : (a1 + a2) / 2 = 2.4)
  (h_avg_2_2 : (a3 + a4) / 2 = 2.3) :
  (a5 + a6) / 2 = 3.7 :=
by
  sorry

end average_remaining_two_numbers_l217_217420


namespace hyperbola_eccentricity_correct_l217_217834

noncomputable def hyperbola_eccentricity : Real :=
  let a := 5
  let b := 4
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  c / a

theorem hyperbola_eccentricity_correct :
  hyperbola_eccentricity = Real.sqrt 41 / 5 :=
by
  sorry

end hyperbola_eccentricity_correct_l217_217834


namespace factor_expression_l217_217128

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l217_217128


namespace determinant_zero_l217_217971

noncomputable def matrix_A (θ φ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2 * Real.sin θ, -Real.cos θ],
    ![-2 * Real.sin θ, 0, Real.sin φ],
    ![Real.cos θ, -Real.sin φ, 0]]

theorem determinant_zero (θ φ : ℝ) : Matrix.det (matrix_A θ φ) = 0 := by
  sorry

end determinant_zero_l217_217971


namespace cattle_train_left_6_hours_before_l217_217092

theorem cattle_train_left_6_hours_before 
  (Vc : ℕ) (Vd : ℕ) (T : ℕ) 
  (h1 : Vc = 56)
  (h2 : Vd = Vc - 33)
  (h3 : 12 * Vd + 12 * Vc + T * Vc = 1284) : 
  T = 6 := 
by
  sorry

end cattle_train_left_6_hours_before_l217_217092


namespace distinct_real_roots_a1_l217_217304

theorem distinct_real_roots_a1 {x : ℝ} :
  ∀ a : ℝ, a = 1 →
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^2 + (1 - a) * x1 - 1 = 0) ∧ (a * x2^2 + (1 - a) * x2 - 1 = 0) :=
by sorry

end distinct_real_roots_a1_l217_217304


namespace total_screens_sold_l217_217717

variable (J F M : ℕ)
variable (feb_eq_fourth_of_march : F = M / 4)
variable (feb_eq_double_of_jan : F = 2 * J)
variable (march_sales : M = 8800)

theorem total_screens_sold (J F M : ℕ)
  (feb_eq_fourth_of_march : F = M / 4)
  (feb_eq_double_of_jan : F = 2 * J)
  (march_sales : M = 8800) :
  J + F + M = 12100 :=
by
  sorry

end total_screens_sold_l217_217717


namespace total_profit_is_8800_l217_217116

variable (A B C : Type) [CommRing A] [CommRing B] [CommRing C]

variable (investment_A investment_B investment_C : ℝ)
variable (total_profit : ℝ)

-- Conditions
def A_investment_three_times_B (investment_A investment_B : ℝ) : Prop :=
  investment_A = 3 * investment_B

def B_invest_two_thirds_C (investment_B investment_C : ℝ) : Prop :=
  investment_B = 2 / 3 * investment_C

def B_share_is_1600 (investment_B total_profit : ℝ) : Prop :=
  1600 = (2 / 11) * total_profit

theorem total_profit_is_8800 :
  A_investment_three_times_B investment_A investment_B →
  B_invest_two_thirds_C investment_B investment_C →
  B_share_is_1600 investment_B total_profit →
  total_profit = 8800 :=
by
  intros
  sorry

end total_profit_is_8800_l217_217116


namespace find_a2_and_sum_l217_217590

theorem find_a2_and_sum (a a1 a2 a3 a4 : ℝ) (x : ℝ) (h1 : (1 + 2 * x)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) :
  a2 = 24 ∧ a + a1 + a2 + a3 + a4 = 81 :=
by
  sorry

end find_a2_and_sum_l217_217590


namespace find_polynomial_l217_217911

theorem find_polynomial
  (M : ℝ → ℝ)
  (h : ∀ x, M x + 5 * x^2 - 4 * x - 3 = -1 * x^2 - 3 * x) :
  ∀ x, M x = -6 * x^2 + x + 3 :=
sorry

end find_polynomial_l217_217911


namespace total_cost_l217_217704

-- Define the given conditions.
def coffee_pounds : ℕ := 4
def coffee_cost_per_pound : ℝ := 2.50
def milk_gallons : ℕ := 2
def milk_cost_per_gallon : ℝ := 3.50

-- The total cost Jasmine will pay is $17.00
theorem total_cost : coffee_pounds * coffee_cost_per_pound + milk_gallons * milk_cost_per_gallon = 17.00 := by
  sorry

end total_cost_l217_217704


namespace max_ab_real_positive_l217_217025

theorem max_ab_real_positive (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 2) : 
  ab ≤ 1 :=
sorry

end max_ab_real_positive_l217_217025


namespace area_of_YZW_l217_217200

-- Definitions from conditions
def area_of_triangle_XYZ := 36
def base_XY := 8
def base_YW := 32

-- The theorem to prove
theorem area_of_YZW : 1/2 * base_YW * (2 * area_of_triangle_XYZ / base_XY) = 144 := 
by
  -- Placeholder for the proof  
  sorry

end area_of_YZW_l217_217200


namespace find_x_l217_217449

-- Let \( x \) be a real number.
variable (x : ℝ)

-- Condition given in the problem.
def condition : Prop := x = (3 / 7) * x + 200

-- The main statement to be proved.
theorem find_x (h : condition x) : x = 350 :=
  sorry

end find_x_l217_217449


namespace length_of_platform_l217_217140

noncomputable def train_length : ℝ := 450
noncomputable def signal_pole_time : ℝ := 18
noncomputable def platform_time : ℝ := 39

theorem length_of_platform : 
  ∃ (L : ℝ), 
    (train_length / signal_pole_time = (train_length + L) / platform_time) → 
    L = 525 := 
by
  sorry

end length_of_platform_l217_217140


namespace total_amount_spent_l217_217220

variables (P J I T : ℕ)

-- Given conditions
def Pauline_dress : P = 30 := sorry
def Jean_dress : J = P - 10 := sorry
def Ida_dress : I = J + 30 := sorry
def Patty_dress : T = I + 10 := sorry

-- Theorem to prove the total amount spent
theorem total_amount_spent :
  P + J + I + T = 160 :=
by
  -- Placeholder for proof
  sorry

end total_amount_spent_l217_217220


namespace problem_equivalence_l217_217902

theorem problem_equivalence : (7^2 - 3^2)^4 = 2560000 :=
by
  sorry

end problem_equivalence_l217_217902


namespace gilbert_herb_plants_count_l217_217114

variable (initial_basil : Nat) (initial_parsley : Nat) (initial_mint : Nat)
variable (dropped_basil_seeds : Nat) (rabbit_ate_all_mint : Bool)

def total_initial_plants (initial_basil initial_parsley initial_mint : Nat) : Nat :=
  initial_basil + initial_parsley + initial_mint

def total_plants_after_dropping_seeds (initial_basil initial_parsley initial_mint dropped_basil_seeds : Nat) : Nat :=
  total_initial_plants initial_basil initial_parsley initial_mint + dropped_basil_seeds

def total_plants_after_rabbit (initial_basil initial_parsley initial_mint dropped_basil_seeds : Nat) (rabbit_ate_all_mint : Bool) : Nat :=
  if rabbit_ate_all_mint then 
    total_plants_after_dropping_seeds initial_basil initial_parsley initial_mint dropped_basil_seeds - initial_mint 
  else 
    total_plants_after_dropping_seeds initial_basil initial_parsley initial_mint dropped_basil_seeds

theorem gilbert_herb_plants_count
  (h1 : initial_basil = 3)
  (h2 : initial_parsley = 1)
  (h3 : initial_mint = 2)
  (h4 : dropped_basil_seeds = 1)
  (h5 : rabbit_ate_all_mint = true) :
  total_plants_after_rabbit initial_basil initial_parsley initial_mint dropped_basil_seeds rabbit_ate_all_mint = 5 := by
  sorry

end gilbert_herb_plants_count_l217_217114


namespace greatest_integer_inequality_l217_217441

theorem greatest_integer_inequality : 
  ⌊ (3 ^ 100 + 2 ^ 100 : ℝ) / (3 ^ 96 + 2 ^ 96) ⌋ = 80 :=
by
  sorry

end greatest_integer_inequality_l217_217441


namespace exponentiation_correct_l217_217696

theorem exponentiation_correct (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 :=
sorry

end exponentiation_correct_l217_217696


namespace david_reading_time_l217_217355

theorem david_reading_time
  (total_time : ℕ)
  (math_time : ℕ)
  (spelling_time : ℕ)
  (reading_time : ℕ)
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : spelling_time = 18)
  (h4 : reading_time = total_time - (math_time + spelling_time)) :
  reading_time = 27 := 
by {
  sorry
}

end david_reading_time_l217_217355


namespace average_balance_correct_l217_217571

-- Define the monthly balances
def january_balance : ℕ := 120
def february_balance : ℕ := 240
def march_balance : ℕ := 180
def april_balance : ℕ := 180
def may_balance : ℕ := 210
def june_balance : ℕ := 300

-- List of all balances
def balances : List ℕ := [january_balance, february_balance, march_balance, april_balance, may_balance, june_balance]

-- Define the function to calculate the average balance
def average_balance (balances : List ℕ) : ℕ :=
  (balances.sum / balances.length)

-- Define the target average balance
def target_average_balance : ℕ := 205

-- The theorem we need to prove
theorem average_balance_correct :
  average_balance balances = target_average_balance :=
by
  sorry

end average_balance_correct_l217_217571


namespace quadratic_increasing_implies_m_gt_1_l217_217180

theorem quadratic_increasing_implies_m_gt_1 (m : ℝ) (x : ℝ) 
(h1 : x > 1) 
(h2 : ∀ x, (y = x^2 + (m-3) * x + m + 1) → (∀ z > x, y < z^2 + (m-3) * z + m + 1)) 
: m > 1 := 
sorry

end quadratic_increasing_implies_m_gt_1_l217_217180


namespace fractional_eq_no_solution_l217_217227

theorem fractional_eq_no_solution (m : ℝ) :
  ¬ ∃ x, (x - 2) / (x + 2) - (m * x) / (x^2 - 4) = 1 ↔ m = -4 :=
by
  sorry

end fractional_eq_no_solution_l217_217227


namespace compute_expression_l217_217658

theorem compute_expression : (3 + 9)^3 + (3^3 + 9^3) = 2484 := by
  sorry

end compute_expression_l217_217658


namespace part1_part2_l217_217865

-- Definition of the function
def f (a x : ℝ) := |x - a|

-- Proof statement for question 1
theorem part1 (a : ℝ)
  (h : ∀ x : ℝ, f a x ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) :
  a = 3 := by
  sorry

-- Auxiliary function for question 2
def g (a x : ℝ) := f a (2 * x) + f a (x + 2)

-- Proof statement for question 2
theorem part2 (m : ℝ)
  (h : ∀ x : ℝ, g 3 x ≥ m) :
  m ≤ 1/2 := by
  sorry

end part1_part2_l217_217865


namespace vasya_no_purchase_days_l217_217664

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l217_217664


namespace tangent_line_condition_l217_217659

-- statement only, no proof required
theorem tangent_line_condition {m n u v x y : ℝ}
  (hm : m > 1)
  (curve_eq : x^m + y^m = 1)
  (line_eq : u * x + v * y = 1)
  (u_v_condition : u^n + v^n = 1)
  (mn_condition : 1/m + 1/n = 1)
  : (u * x + v * y = 1) ↔ (u^n + v^n = 1 ∧ 1/m + 1/n = 1) :=
sorry

end tangent_line_condition_l217_217659


namespace not_perfect_square_for_n_greater_than_11_l217_217368

theorem not_perfect_square_for_n_greater_than_11 (n : ℤ) (h1 : n > 11) :
  ∀ m : ℤ, n^2 - 19 * n + 89 ≠ m^2 :=
sorry

end not_perfect_square_for_n_greater_than_11_l217_217368


namespace crayons_more_than_erasers_l217_217628

-- Definitions of the conditions
def initial_crayons := 531
def initial_erasers := 38
def final_crayons := 391
def final_erasers := initial_erasers -- no erasers lost

-- Theorem statement
theorem crayons_more_than_erasers :
  final_crayons - final_erasers = 102 :=
by
  -- Placeholder for the proof
  sorry

end crayons_more_than_erasers_l217_217628


namespace total_spectators_l217_217595

-- Definitions of conditions
def num_men : Nat := 7000
def num_children : Nat := 2500
def num_women := num_children / 5

-- Theorem stating the total number of spectators
theorem total_spectators : (num_men + num_children + num_women) = 10000 := by
  sorry

end total_spectators_l217_217595


namespace scaling_matrix_unique_l217_217164

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

noncomputable def matrix_N : Matrix (Fin 4) (Fin 4) ℝ := ![![3, 0, 0, 0], ![0, 3, 0, 0], ![0, 0, 3, 0], ![0, 0, 0, 3]]

theorem scaling_matrix_unique (N : Matrix (Fin 4) (Fin 4) ℝ) :
  (∀ (w : Fin 4 → ℝ), N.mulVec w = 3 • w) → N = matrix_N :=
by
  intros h
  sorry

end scaling_matrix_unique_l217_217164


namespace middle_number_consecutive_even_l217_217132

theorem middle_number_consecutive_even (a b c : ℤ) 
  (h1 : a = b - 2) 
  (h2 : c = b + 2) 
  (h3 : a + b = 18) 
  (h4 : a + c = 22) 
  (h5 : b + c = 28) : 
  b = 11 :=
by sorry

end middle_number_consecutive_even_l217_217132


namespace bureaucrats_total_l217_217417

-- Define the parameters and conditions as stated in the problem
variables (a b c : ℕ)

-- Conditions stated in the problem
def condition_1 : Prop :=
  ∀ (i j : ℕ) (h1 : i ≠ j), 
    (10 * a * b = 10 * a * c ∧ 10 * b * c = 10 * a * b)

-- The main goal: proving the total number of bureaucrats
theorem bureaucrats_total (h1 : a = b) (h2 : b = c) (h3 : condition_1 a b c) : 
  3 * a = 120 :=
by sorry

end bureaucrats_total_l217_217417


namespace puppy_food_cost_l217_217154

theorem puppy_food_cost :
  let puppy_cost : ℕ := 10
  let days_in_week : ℕ := 7
  let total_number_of_weeks : ℕ := 3
  let cups_per_day : ℚ := 1 / 3
  let cups_per_bag : ℚ := 3.5
  let cost_per_bag : ℕ := 2
  let total_days := total_number_of_weeks * days_in_week
  let total_cups := total_days * cups_per_day
  let total_bags := total_cups / cups_per_bag
  let food_cost := total_bags * cost_per_bag
  let total_cost := puppy_cost + food_cost
  total_cost = 14 := by
  sorry

end puppy_food_cost_l217_217154


namespace max_b_value_l217_217547

theorem max_b_value (a b c : ℕ) (h_volume : a * b * c = 360) (h_conditions : 1 < c ∧ c < b ∧ b < a) : b = 12 :=
  sorry

end max_b_value_l217_217547


namespace roots_of_equation_l217_217887

theorem roots_of_equation :
  {x : ℝ | -x * (x + 3) = x * (x + 3)} = {0, -3} :=
by
  sorry

end roots_of_equation_l217_217887


namespace parabola_vertex_example_l217_217896

noncomputable def parabola_vertex (a b c : ℝ) := (-b / (2 * a), (4 * a * c - b^2) / (4 * a))

theorem parabola_vertex_example : parabola_vertex (-4) (-16) (-20) = (-2, -4) :=
by
  sorry

end parabola_vertex_example_l217_217896


namespace junk_items_count_l217_217272

variable (total_items : ℕ)
variable (useful_percentage : ℚ := 0.20)
variable (heirloom_percentage : ℚ := 0.10)
variable (junk_percentage : ℚ := 0.70)
variable (useful_items : ℕ := 8)

theorem junk_items_count (huseful : useful_percentage * total_items = useful_items) : 
  junk_percentage * total_items = 28 :=
by
  sorry

end junk_items_count_l217_217272


namespace minimal_degree_of_g_l217_217707

noncomputable def g_degree_minimal (f g h : Polynomial ℝ) (deg_f : ℕ) (deg_h : ℕ) (h_eq : (5 : ℝ) • f + (7 : ℝ) • g = h) : Prop :=
  Polynomial.degree f = deg_f ∧ Polynomial.degree h = deg_h → Polynomial.degree g = 12

theorem minimal_degree_of_g (f g h : Polynomial ℝ) (h_eq : (5 : ℝ) • f + (7 : ℝ) • g = h)
    (deg_f : Polynomial.degree f = 5) (deg_h : Polynomial.degree h = 12) :
    Polynomial.degree g = 12 := by
  sorry

end minimal_degree_of_g_l217_217707


namespace total_hours_driven_l217_217618

def total_distance : ℝ := 55.0
def distance_in_one_hour : ℝ := 1.527777778

theorem total_hours_driven : (total_distance / distance_in_one_hour) = 36.00 :=
by
  sorry

end total_hours_driven_l217_217618


namespace N_subset_M_l217_217377

open Set

def M : Set (ℝ × ℝ) := { p | ∃ x, p = (x, 2*x + 1) }
def N : Set (ℝ × ℝ) := { p | ∃ x, p = (x, -x^2) }

theorem N_subset_M : N ⊆ M :=
by
  sorry

end N_subset_M_l217_217377


namespace proof_problem_l217_217587

theorem proof_problem (x : ℝ) 
    (h1 : (x - 1) * (x + 1) = x^2 - 1)
    (h2 : (x - 1) * (x^2 + x + 1) = x^3 - 1)
    (h3 : (x - 1) * (x^3 + x^2 + x + 1) = x^4 - 1)
    (h4 : (x - 1) * (x^4 + x^3 + x^2 + x + 1) = -2) :
    x^2023 = -1 := 
by 
  sorry -- Proof is omitted

end proof_problem_l217_217587


namespace star_three_four_eq_zero_l217_217847

def star (a b : ℕ) : ℕ := 4 * a + 3 * b - 2 * a * b

theorem star_three_four_eq_zero : star 3 4 = 0 := sorry

end star_three_four_eq_zero_l217_217847


namespace find_x_l217_217159

-- Define that x is a real number and positive
variable (x : ℝ)
variable (hx_pos : 0 < x)

-- Define the floor function and the main equation
variable (hx_eq : ⌊x⌋ * x = 90)

theorem find_x (h : ⌊x⌋ * x = 90) (hx_pos : 0 < x) : ⌊x⌋ = 9 ∧ x = 10 :=
by
  sorry

end find_x_l217_217159


namespace fourth_power_of_cube_third_smallest_prime_l217_217213

-- Define the third smallest prime number
def third_smallest_prime : Nat := 5

-- Define a function that calculates the fourth power of a number
def fourth_power (x : Nat) : Nat := x * x * x * x

-- Define a function that calculates the cube of a number
def cube (x : Nat) : Nat := x * x * x

-- The proposition stating the fourth power of the cube of the third smallest prime number is 244140625
theorem fourth_power_of_cube_third_smallest_prime : 
  fourth_power (cube third_smallest_prime) = 244140625 :=
by
  -- skip the proof
  sorry

end fourth_power_of_cube_third_smallest_prime_l217_217213


namespace smallest_number_after_operations_n_111_smallest_number_after_operations_n_110_l217_217188

theorem smallest_number_after_operations_n_111 :
  ∀ (n : ℕ), n = 111 → 
  (∃ (f : List ℕ → ℕ), -- The function f represents the sequence of operations
     (∀ (l : List ℕ), l = List.range 111 →
       (f l) = 0)) :=
by 
  sorry

theorem smallest_number_after_operations_n_110 :
  ∀ (n : ℕ), n = 110 → 
  (∃ (f : List ℕ → ℕ), -- The function f represents the sequence of operations
     (∀ (l : List ℕ), l = List.range 110 →
       (f l) = 1)) :=
by 
  sorry

end smallest_number_after_operations_n_111_smallest_number_after_operations_n_110_l217_217188


namespace simplify_expression_l217_217225

theorem simplify_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (b * c + a * c + a * b) :=
by
  sorry

end simplify_expression_l217_217225


namespace find_interesting_numbers_l217_217968

def is_interesting (A B : ℕ) : Prop :=
  A > B ∧ (∃ p : ℕ, Nat.Prime p ∧ A - B = p) ∧ ∃ n : ℕ, A * B = n ^ 2

theorem find_interesting_numbers :
  {A | (∃ B : ℕ, is_interesting A B) ∧ 200 < A ∧ A < 400} = {225, 256, 361} :=
by
  sorry

end find_interesting_numbers_l217_217968


namespace avg_math_chem_l217_217416

variables (M P C : ℕ)

def total_marks (M P : ℕ) := M + P = 50
def chemistry_marks (P C : ℕ) := C = P + 20

theorem avg_math_chem (M P C : ℕ) (h1 : total_marks M P) (h2 : chemistry_marks P C) :
  (M + C) / 2 = 35 :=
by
  sorry

end avg_math_chem_l217_217416


namespace probability_space_diagonal_l217_217732

theorem probability_space_diagonal : 
  let vertices := 8
  let space_diagonals := 4
  let total_pairs := Nat.choose vertices 2
  4 / total_pairs = 1 / 7 :=
by
  sorry

end probability_space_diagonal_l217_217732


namespace times_faster_l217_217229

theorem times_faster (A B W : ℝ) (h1 : A = 3 * B) (h2 : (A + B) * 21 = A * 28) : A = 3 * B :=
by sorry

end times_faster_l217_217229


namespace fertilizer_needed_per_acre_l217_217083

-- Definitions for the conditions
def horse_daily_fertilizer : ℕ := 5 -- Each horse produces 5 gallons of fertilizer per day.
def horses : ℕ := 80 -- Janet has 80 horses.
def days : ℕ := 25 -- It takes 25 days until all her fields are fertilized.
def total_acres : ℕ := 20 -- Janet's farmland is 20 acres.

-- Calculated intermediate values
def total_fertilizer : ℕ := horse_daily_fertilizer * horses * days -- Total fertilizer produced
def fertilizer_per_acre : ℕ := total_fertilizer / total_acres -- Fertilizer needed per acre

-- Theorem to prove
theorem fertilizer_needed_per_acre : fertilizer_per_acre = 500 := by
  sorry

end fertilizer_needed_per_acre_l217_217083


namespace common_fraction_equiv_l217_217649

noncomputable def decimal_equivalent_frac : Prop :=
  ∃ (x : ℚ), x = 413 / 990 ∧ x = 0.4 + (7/10^2 + 1/10^3) / (1 - 1/10^2)

theorem common_fraction_equiv : decimal_equivalent_frac :=
by
  sorry

end common_fraction_equiv_l217_217649


namespace find_b_l217_217804

noncomputable def point (x y : Float) : Float × Float := (x, y)

def line_y_eq_b_plus_x (b x : Float) : Float := b + x

def intersects_y_axis (b : Float) : Float × Float := (0, b)

def intersects_x_axis (b : Float) : Float × Float := (-b, 0)

def intersects_x_eq_5 (b : Float) : Float × Float := (5, b + 5)

def area_triangle_qrs (b : Float) : Float :=
  0.5 * (5 + b) * (b + 5)

def area_triangle_qop (b : Float) : Float :=
  0.5 * b * b

theorem find_b (b : Float) (h : b > 0) (h_area_ratio : area_triangle_qrs b / area_triangle_qop b = 4 / 9) : b = 5 :=
by
  sorry

end find_b_l217_217804


namespace nancy_first_album_pictures_l217_217509

theorem nancy_first_album_pictures (total_pics : ℕ) (total_albums : ℕ) (pics_per_album : ℕ)
    (h1 : total_pics = 51) (h2 : total_albums = 8) (h3 : pics_per_album = 5) :
    (total_pics - total_albums * pics_per_album = 11) :=
by
    sorry

end nancy_first_album_pictures_l217_217509


namespace find_S12_l217_217279

variable {a : Nat → Int} -- representing the arithmetic sequence {a_n}
variable {S : Nat → Int} -- representing the sums of the first n terms, S_n

-- Condition: a_1 = -9
axiom a1_def : a 1 = -9

-- Condition: (S_n / n) forms an arithmetic sequence
axiom arithmetic_s : ∃ d : Int, ∀ n : Nat, S n / n = -9 + (n - 1) * d

-- Condition: 2 = S9 / 9 - S7 / 7
axiom condition : S 9 / 9 - S 7 / 7 = 2

-- We want to prove: S_12 = 36
theorem find_S12 : S 12 = 36 := 
sorry

end find_S12_l217_217279


namespace speed_of_man_is_approx_4_99_l217_217802

noncomputable def train_length : ℝ := 110  -- meters
noncomputable def train_speed : ℝ := 50  -- km/h
noncomputable def time_to_pass_man : ℝ := 7.2  -- seconds

def mps_to_kmph (speed : ℝ) : ℝ := speed * 3.6

noncomputable def relative_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

noncomputable def relative_speed_kmph : ℝ :=
  mps_to_kmph (relative_speed train_length time_to_pass_man)

noncomputable def speed_of_man (relative_speed_kmph : ℝ) (train_speed : ℝ) : ℝ :=
  relative_speed_kmph - train_speed

theorem speed_of_man_is_approx_4_99 :
  abs (speed_of_man relative_speed_kmph train_speed - 4.99) < 0.01 :=
by
  sorry

end speed_of_man_is_approx_4_99_l217_217802


namespace situationD_not_represented_l217_217231

def situationA := -2 + 10 = 8

def situationB := -2 + 10 = 8

def situationC := 10 - 2 = 8 ∧ -2 + 10 = 8

def situationD := |10 - (-2)| = 12

theorem situationD_not_represented : ¬ (|10 - (-2)| = -2 + 10) := 
by
  sorry

end situationD_not_represented_l217_217231


namespace no_nat_m_n_square_diff_2014_l217_217258

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l217_217258


namespace intersection_product_l217_217786

noncomputable def line_l (t : ℝ) := (1 + (Real.sqrt 3 / 2) * t, 1 + (1/2) * t)

def curve_C (x y : ℝ) : Prop := y^2 = 8 * x

theorem intersection_product :
  ∀ (t1 t2 : ℝ), 
  (1 + (1/2) * t1)^2 = 8 * (1 + (Real.sqrt 3 / 2) * t1) →
  (1 + (1/2) * t2)^2 = 8 * (1 + (Real.sqrt 3 / 2) * t2) →
  (1 + (1/2) * t1) * (1 + (1/2) * t2) = 28 := 
  sorry

end intersection_product_l217_217786


namespace mod_equiv_solution_l217_217182

theorem mod_equiv_solution (a b : ℤ) (n : ℤ) 
  (h₁ : a ≡ 22 [ZMOD 50])
  (h₂ : b ≡ 78 [ZMOD 50])
  (h₃ : 150 ≤ n ∧ n ≤ 201)
  (h₄ : n = 194) :
  a - b ≡ n [ZMOD 50] :=
by
  sorry

end mod_equiv_solution_l217_217182


namespace solution_set_for_inequality_l217_217394

theorem solution_set_for_inequality :
  {x : ℝ | -x^2 + 4 * x + 5 < 0} = {x : ℝ | x > 5 ∨ x < -1} :=
by
  sorry

end solution_set_for_inequality_l217_217394


namespace first_number_Harold_says_l217_217401

/-
  Define each student's sequence of numbers.
  - Alice skips every 4th number.
  - Barbara says numbers that Alice didn't say, skipping every 4th in her sequence.
  - Subsequent students follow the same rule.
  - Harold picks the smallest prime number not said by any student.
-/

def is_skipped_by_Alice (n : Nat) : Prop :=
  n % 4 ≠ 0

def is_skipped_by_Barbara (n : Nat) : Prop :=
  is_skipped_by_Alice n ∧ (n / 4) % 4 ≠ 3

def is_skipped_by_Candice (n : Nat) : Prop :=
  is_skipped_by_Barbara n ∧ (n / 16) % 4 ≠ 3

def is_skipped_by_Debbie (n : Nat) : Prop :=
  is_skipped_by_Candice n ∧ (n / 64) % 4 ≠ 3

def is_skipped_by_Eliza (n : Nat) : Prop :=
  is_skipped_by_Debbie n ∧ (n / 256) % 4 ≠ 3

def is_skipped_by_Fatima (n : Nat) : Prop :=
  is_skipped_by_Eliza n ∧ (n / 1024) % 4 ≠ 3

def is_skipped_by_Grace (n : Nat) : Prop :=
  is_skipped_by_Fatima n

def is_skipped_by_anyone (n : Nat) : Prop :=
  ¬ is_skipped_by_Alice n ∨ ¬ is_skipped_by_Barbara n ∨ ¬ is_skipped_by_Candice n ∨
  ¬ is_skipped_by_Debbie n ∨ ¬ is_skipped_by_Eliza n ∨ ¬ is_skipped_by_Fatima n ∨
  ¬ is_skipped_by_Grace n

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ (m : Nat), m ∣ n → m = 1 ∨ m = n

theorem first_number_Harold_says : ∃ n : Nat, is_prime n ∧ ¬ is_skipped_by_anyone n ∧ n = 11 := by
  sorry

end first_number_Harold_says_l217_217401


namespace find_number_l217_217951

def sum : ℕ := 2468 + 1375
def diff : ℕ := 2468 - 1375
def first_quotient : ℕ := 3 * diff
def second_quotient : ℕ := 5 * diff
def remainder : ℕ := 150

theorem find_number (N : ℕ) (h1 : sum = 3843) (h2 : diff = 1093) 
                    (h3 : first_quotient = 3279) (h4 : second_quotient = 5465)
                    (h5 : remainder = 150) (h6 : N = sum * first_quotient + remainder)
                    (h7 : N = sum * second_quotient + remainder) :
  N = 12609027 := 
by 
  sorry

end find_number_l217_217951


namespace insufficient_pharmacies_l217_217104

/-- Define the grid size and conditions --/
def grid_size : ℕ := 9
def total_intersections : ℕ := 100
def pharmacy_walking_distance : ℕ := 3
def number_of_pharmacies : ℕ := 12

/-- Prove that 12 pharmacies are not enough for the given conditions. --/
theorem insufficient_pharmacies : ¬ (number_of_pharmacies ≥ grid_size + 3) → 
  ∃(pharmacies : ℕ), pharmacies = number_of_pharmacies ∧
  ∀(x y : ℕ), (x ≤ grid_size ∧ y ≤ grid_size) → 
  ¬ exists (p : ℕ × ℕ), p.1 ≤ x + pharmacy_walking_distance ∧
  p.2 ≤ y + pharmacy_walking_distance ∧ 
  p.1 < total_intersections ∧ 
  p.2 < total_intersections := 
by sorry

end insufficient_pharmacies_l217_217104


namespace rectangular_garden_width_l217_217624

theorem rectangular_garden_width (w : ℕ) (h1 : ∃ l : ℕ, l = 3 * w) (h2 : w * (3 * w) = 507) : w = 13 := 
by 
  sorry

end rectangular_garden_width_l217_217624


namespace find_ac_find_a_and_c_l217_217156

variables (A B C a b c : ℝ)

-- Condition: Angles A, B, C form an arithmetic sequence.
def arithmetic_sequence := 2 * B = A + C

-- Condition: Area of the triangle is sqrt(3)/2.
def area_triangle := (1/2) * a * c * (Real.sin B) = (Real.sqrt 3) / 2

-- Condition: b = sqrt(3)
def b_sqrt3 := b = Real.sqrt 3

-- Goal 1: To prove that ac = 2.
theorem find_ac (h1 : arithmetic_sequence A B C) (h2 : area_triangle a c B) : a * c = 2 :=
sorry

-- Goal 2: To prove a = 2 and c = 1 given the additional condition.
theorem find_a_and_c (h1 : arithmetic_sequence A B C) (h2 : area_triangle a c B) (h3 : b_sqrt3 b) (h4 : a > c) : a = 2 ∧ c = 1 :=
sorry

end find_ac_find_a_and_c_l217_217156


namespace total_tubes_in_consignment_l217_217890

theorem total_tubes_in_consignment (N : ℕ) 
  (h : (5 / (N : ℝ)) * (4 / (N - 1 : ℝ)) = 0.05263157894736842) : 
  N = 20 := 
sorry

end total_tubes_in_consignment_l217_217890


namespace part_a_part_b_l217_217343

theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) : x + y + z ≤ 4 :=
sorry

theorem part_b : ∃ (S : Set (ℚ × ℚ × ℚ)), S.Countable ∧
  (∀ (x y z : ℚ), (x, y, z) ∈ S → 0 < x ∧ 0 < y ∧ 0 < z ∧ 16 * x * y * z = (x + y)^2 * (x + z)^2 ∧ x + y + z = 4) ∧ 
  Infinite S :=
sorry

end part_a_part_b_l217_217343


namespace ineq_x4_y4_l217_217414

theorem ineq_x4_y4 (x y : ℝ) (h1 : x > Real.sqrt 2) (h2 : y > Real.sqrt 2) :
    x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by
  sorry

end ineq_x4_y4_l217_217414


namespace solve_inequality_system_simplify_expression_l217_217127

-- Part 1: System of Inequalities

theorem solve_inequality_system : 
  ∀ (x : ℝ), (x + 2) / 5 < 1 ∧ 3 * x - 1 ≥ 2 * x → 1 ≤ x ∧ x < 3 :=  by
  sorry

-- Part 2: Expression Simplification

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) : 
  (m - 1 / m) * ((m^2 - m) / (m^2 - 2 * m + 1)) = m + 1 :=
  by
  sorry

end solve_inequality_system_simplify_expression_l217_217127


namespace find_de_l217_217160

namespace MagicSquare

variables (a b c d e : ℕ)

-- Hypotheses based on the conditions provided.
axiom H1 : 20 + 15 + a = 57
axiom H2 : 25 + b + a = 57
axiom H3 : 18 + c + a = 57
axiom H4 : 20 + c + b = 57
axiom H5 : d + c + a = 57
axiom H6 : d + e + 18 = 57
axiom H7 : e + 25 + 15 = 57

def magicSum := 57

theorem find_de :
  ∃ d e, d + e = 42 :=
by sorry

end MagicSquare

end find_de_l217_217160


namespace grasshopper_jump_distance_l217_217479

theorem grasshopper_jump_distance (frog_jump grasshopper_jump : ℝ) (h_frog : frog_jump = 40) (h_difference : frog_jump = grasshopper_jump + 15) : grasshopper_jump = 25 :=
by sorry

end grasshopper_jump_distance_l217_217479


namespace gecko_bug_eating_l217_217603

theorem gecko_bug_eating (G L F T : ℝ) (hL : L = G / 2)
                                      (hF : F = 3 * L)
                                      (hT : T = 1.5 * F)
                                      (hTotal : G + L + F + T = 63) :
  G = 15 :=
by
  sorry

end gecko_bug_eating_l217_217603


namespace intersection_M_N_l217_217251

-- Define the sets M and N based on given conditions
def M : Set ℝ := { x : ℝ | x^2 < 4 }
def N : Set ℝ := { x : ℝ | x < 1 }

-- State the theorem to prove the intersection of M and N
theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l217_217251


namespace find_cost_price_l217_217840

/-- 
Given:
- SP = 1290 (selling price)
- LossP = 14.000000000000002 (loss percentage)
Prove that: CP = 1500 (cost price)
--/
theorem find_cost_price (SP : ℝ) (LossP : ℝ) (CP : ℝ) (h1 : SP = 1290) (h2 : LossP = 14.000000000000002) : CP = 1500 :=
sorry

end find_cost_price_l217_217840


namespace intersection_A_B_l217_217862

noncomputable def A : Set ℝ := {x | 9 * x ^ 2 < 1}

noncomputable def B : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2 - 2 * x + 5 / 4}

theorem intersection_A_B :
  (A ∩ B) = {y | y ∈ Set.Ico (1/4 : ℝ) (1/3 : ℝ)} :=
by
  sorry

end intersection_A_B_l217_217862


namespace corrected_mean_l217_217690

theorem corrected_mean (n : ℕ) (obs_mean : ℝ) (obs_count : ℕ) (wrong_val correct_val : ℝ) :
  obs_count = 40 →
  obs_mean = 100 →
  wrong_val = 75 →
  correct_val = 50 →
  (obs_count * obs_mean - (wrong_val - correct_val)) / obs_count = 3975 / 40 :=
by
  sorry

end corrected_mean_l217_217690


namespace solution_set_of_inequality_l217_217373

theorem solution_set_of_inequality :
  {x : ℝ | |x + 1| - |x - 5| < 4} = {x : ℝ | x < 4} :=
sorry

end solution_set_of_inequality_l217_217373


namespace algebraic_expression_value_l217_217529

variable (a b : ℝ)
axiom h1 : a = 3
axiom h2 : a - b = 1

theorem algebraic_expression_value :
  a^2 - a * b = 3 :=
by
  sorry

end algebraic_expression_value_l217_217529


namespace expression_equals_one_l217_217347

def evaluate_expression : ℕ :=
  3 * (3 * (3 * (3 * (3 - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1

theorem expression_equals_one : evaluate_expression = 1 := by
  sorry

end expression_equals_one_l217_217347


namespace compare_neg_fractions_l217_217413

theorem compare_neg_fractions : - (3 / 5 : ℚ) < - (1 / 5 : ℚ) :=
by
  sorry

end compare_neg_fractions_l217_217413


namespace robert_has_2_more_years_l217_217642

theorem robert_has_2_more_years (R P T Rb M : ℕ) 
                                 (h1 : R = P + T + Rb + M)
                                 (h2 : R = 42)
                                 (h3 : P = 12)
                                 (h4 : T = 2 * Rb)
                                 (h5 : Rb = P - 4) : Rb - M = 2 := 
by 
-- skipped proof
  sorry

end robert_has_2_more_years_l217_217642


namespace mark_sold_8_boxes_less_l217_217899

theorem mark_sold_8_boxes_less (T M A x : ℕ) (hT : T = 9) 
    (hM : M = T - x) (hA : A = T - 2) 
    (hM_ge_1 : 1 ≤ M) (hA_ge_1 : 1 ≤ A) 
    (h_sum_lt_T : M + A < T) : x = 8 := 
by
  sorry

end mark_sold_8_boxes_less_l217_217899


namespace sequence_sum_square_l217_217450

-- Definition of the sum of the symmetric sequence.
def sequence_sum (n : ℕ) : ℕ :=
  (List.range' 1 (n+1)).sum + (List.range' 1 n).sum

-- The conjecture that the sum of the sequence equals n^2.
theorem sequence_sum_square (n : ℕ) : sequence_sum n = n^2 := by
  sorry

end sequence_sum_square_l217_217450


namespace kevin_total_miles_l217_217440

theorem kevin_total_miles : 
  ∃ (d1 d2 d3 d4 d5 : ℕ), 
  d1 = 60 / 6 ∧ 
  d2 = 60 / (6 + 6 * 1) ∧ 
  d3 = 60 / (6 + 6 * 2) ∧ 
  d4 = 60 / (6 + 6 * 3) ∧ 
  d5 = 60 / (6 + 6 * 4) ∧ 
  (d1 + d2 + d3 + d4 + d5) = 13 := 
by
  sorry

end kevin_total_miles_l217_217440


namespace coefficient_of_determination_indicates_better_fit_l217_217318

theorem coefficient_of_determination_indicates_better_fit (R_squared : ℝ) (h1 : 0 ≤ R_squared) (h2 : R_squared ≤ 1) :
  R_squared = 1 → better_fitting_effect_of_regression_model :=
by
  sorry

end coefficient_of_determination_indicates_better_fit_l217_217318


namespace solve_inequality_l217_217141

theorem solve_inequality (x : ℝ) :
  (x ^ 2 - 2 * x - 3) * (x ^ 2 - 4 * x + 4) < 0 ↔ (-1 < x ∧ x < 3 ∧ x ≠ 2) := by
  sorry

end solve_inequality_l217_217141


namespace minimum_value_f_maximum_value_f_l217_217459

-- Problem 1: Minimum value of f(x) = 12/x + 3x for x > 0
theorem minimum_value_f (x : ℝ) (h : x > 0) : 
  (12 / x + 3 * x) ≥ 12 :=
sorry

-- Problem 2: Maximum value of f(x) = x(1 - 3x) for 0 < x < 1/3
theorem maximum_value_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1 / 3) :
  x * (1 - 3 * x) ≤ 1 / 12 :=
sorry

end minimum_value_f_maximum_value_f_l217_217459


namespace range_of_a_l217_217601

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 < x → (a / x - 4 / x^2 < 1)) → a < 4 := 
by
  sorry

end range_of_a_l217_217601


namespace prism_surface_area_l217_217097

theorem prism_surface_area (P : ℝ) (h : ℝ) (S : ℝ) (s: ℝ) 
  (hP : P = 4)
  (hh : h = 2) 
  (hs : s = 1) 
  (h_surf_top : S = s * s) 
  (h_lat : S = 8) : 
  S = 10 := 
sorry

end prism_surface_area_l217_217097


namespace systematic_sampling_probability_l217_217549

/-- Given a population of 1002 individuals, if we remove 2 randomly and then pick 50 out of the remaining 1000, then the probability of picking each individual is 50/1002. 
This is because the process involves two independent steps: not being removed initially and then being chosen in the sample of size 50. --/
theorem systematic_sampling_probability :
  let population_size := 1002
  let removal_count := 2
  let sample_size := 50
  ∀ p : ℕ, p = 50 / (1002 : ℚ) := sorry

end systematic_sampling_probability_l217_217549


namespace mushrooms_safe_to_eat_l217_217709

theorem mushrooms_safe_to_eat (S : ℕ) (Total_mushrooms Poisonous_mushrooms Uncertain_mushrooms : ℕ)
  (h1: Total_mushrooms = 32)
  (h2: Poisonous_mushrooms = 2 * S)
  (h3: Uncertain_mushrooms = 5)
  (h4: S + Poisonous_mushrooms + Uncertain_mushrooms = Total_mushrooms) :
  S = 9 :=
sorry

end mushrooms_safe_to_eat_l217_217709


namespace percent_pension_participation_l217_217266

-- Define the conditions provided
def total_first_shift_members : ℕ := 60
def total_second_shift_members : ℕ := 50
def total_third_shift_members : ℕ := 40

def first_shift_pension_percentage : ℚ := 20 / 100
def second_shift_pension_percentage : ℚ := 40 / 100
def third_shift_pension_percentage : ℚ := 10 / 100

-- Calculate participation in the pension program for each shift
def first_shift_pension_members := total_first_shift_members * first_shift_pension_percentage
def second_shift_pension_members := total_second_shift_members * second_shift_pension_percentage
def third_shift_pension_members := total_third_shift_members * third_shift_pension_percentage

-- Calculate total participation in the pension program and total number of workers
def total_pension_members := first_shift_pension_members + second_shift_pension_members + third_shift_pension_members
def total_workers := total_first_shift_members + total_second_shift_members + total_third_shift_members

-- Lean proof statement
theorem percent_pension_participation : (total_pension_members / total_workers * 100) = 24 := by
  sorry

end percent_pension_participation_l217_217266


namespace simplify_expression_l217_217754

theorem simplify_expression (x : ℝ) : (x + 2)^2 + x * (x - 4) = 2 * x^2 + 4 := by
  sorry

end simplify_expression_l217_217754


namespace triangle_area_given_conditions_l217_217981

theorem triangle_area_given_conditions (a b c A B S : ℝ) (h₁ : (2 * c - b) * Real.cos A = a * Real.cos B) (h₂ : b = 1) (h₃ : c = 2) :
  S = (1 / 2) * b * c * Real.sin A → S = Real.sqrt 3 / 2 := 
by
  intros
  sorry

end triangle_area_given_conditions_l217_217981


namespace avg_rest_students_l217_217153

/- Definitions based on conditions -/
def total_students : ℕ := 28
def students_scored_95 : ℕ := 4
def students_scored_0 : ℕ := 3
def avg_whole_class : ℚ := 47.32142857142857
def total_marks_95 : ℚ := students_scored_95 * 95
def total_marks_0 : ℚ := students_scored_0 * 0
def marks_whole_class : ℚ := total_students * avg_whole_class
def rest_students : ℕ := total_students - students_scored_95 - students_scored_0

/- Theorem to prove the average of the rest students given the conditions -/
theorem avg_rest_students : (total_marks_95 + total_marks_0 + rest_students * 45) = marks_whole_class :=
by
  sorry

end avg_rest_students_l217_217153


namespace min_distance_AB_tangent_line_circle_l217_217702

theorem min_distance_AB_tangent_line_circle 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h_tangent : a^2 + b^2 = 1) :
  ∃ A B : ℝ × ℝ, (A = (0, 1/b) ∧ B = (2/a, 0)) ∧ dist A B = 3 :=
by
  sorry

end min_distance_AB_tangent_line_circle_l217_217702


namespace exists_c_same_digit_occurrences_l217_217908

theorem exists_c_same_digit_occurrences (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  ∃ c : ℕ, c > 0 ∧ ∀ d : ℕ, d ≠ 0 → 
    (Nat.digits 10 (c * m)).count d = (Nat.digits 10 (c * n)).count d := sorry

end exists_c_same_digit_occurrences_l217_217908


namespace simplify_expression_l217_217430

theorem simplify_expression : (2^3002 * 3^3004) / 6^3003 = 3 / 4 := by
  sorry

end simplify_expression_l217_217430


namespace triangle_height_l217_217445

theorem triangle_height (x y : ℝ) :
  let area := (x^3 * y)^2
  let base := (2 * x * y)^2
  base ≠ 0 →
  (2 * area) / base = x^4 / 2 :=
by
  sorry

end triangle_height_l217_217445


namespace coeffs_divisible_by_5_l217_217316

theorem coeffs_divisible_by_5
  (a b c d : ℤ)
  (h1 : a + b + c + d ≡ 0 [ZMOD 5])
  (h2 : -a + b - c + d ≡ 0 [ZMOD 5])
  (h3 : 8 * a + 4 * b + 2 * c + d ≡ 0 [ZMOD 5])
  (h4 : d ≡ 0 [ZMOD 5]) :
  a ≡ 0 [ZMOD 5] ∧ b ≡ 0 [ZMOD 5] ∧ c ≡ 0 [ZMOD 5] ∧ d ≡ 0 [ZMOD 5] :=
sorry

end coeffs_divisible_by_5_l217_217316


namespace quadratic_real_roots_iff_l217_217582

theorem quadratic_real_roots_iff (k : ℝ) : 
  (∃ x : ℝ, (k-1) * x^2 + 3 * x - 1 = 0) ↔ k ≥ -5 / 4 ∧ k ≠ 1 := sorry

end quadratic_real_roots_iff_l217_217582


namespace simplify_polynomial_l217_217388

theorem simplify_polynomial (y : ℝ) :
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + y ^ 10 + 2 * y ^ 9) =
  15 * y ^ 13 - y ^ 12 - 3 * y ^ 11 + 4 * y ^ 10 - 4 * y ^ 9 := 
by
  sorry

end simplify_polynomial_l217_217388


namespace Annette_Caitlin_total_weight_l217_217299

variable (A C S : ℕ)

-- Conditions
axiom cond1 : C + S = 87
axiom cond2 : A = S + 8

-- Theorem
theorem Annette_Caitlin_total_weight : A + C = 95 := by
  sorry

end Annette_Caitlin_total_weight_l217_217299


namespace Mike_exercises_l217_217904

theorem Mike_exercises :
  let pull_ups_per_visit := 2
  let push_ups_per_visit := 5
  let squats_per_visit := 10
  let office_visits_per_day := 5
  let kitchen_visits_per_day := 8
  let living_room_visits_per_day := 7
  let days_per_week := 7
  let total_pull_ups := pull_ups_per_visit * office_visits_per_day * days_per_week
  let total_push_ups := push_ups_per_visit * kitchen_visits_per_day * days_per_week
  let total_squats := squats_per_visit * living_room_visits_per_day * days_per_week
  total_pull_ups = 70 ∧ total_push_ups = 280 ∧ total_squats = 490 :=
by
  let pull_ups_per_visit := 2
  let push_ups_per_visit := 5
  let squats_per_visit := 10
  let office_visits_per_day := 5
  let kitchen_visits_per_day := 8
  let living_room_visits_per_day := 7
  let days_per_week := 7
  let total_pull_ups := pull_ups_per_visit * office_visits_per_day * days_per_week
  let total_push_ups := push_ups_per_visit * kitchen_visits_per_day * days_per_week
  let total_squats := squats_per_visit * living_room_visits_per_day * days_per_week
  have h1 : total_pull_ups = 2 * 5 * 7 := rfl
  have h2 : total_push_ups = 5 * 8 * 7 := rfl
  have h3 : total_squats = 10 * 7 * 7 := rfl
  show total_pull_ups = 70 ∧ total_push_ups = 280 ∧ total_squats = 490
  sorry

end Mike_exercises_l217_217904


namespace sum_m_n_l217_217135

theorem sum_m_n (m n : ℤ) (h1 : m^2 - n^2 = 18) (h2 : m - n = 9) : m + n = 2 := 
by
  sorry

end sum_m_n_l217_217135


namespace problem_A_problem_C_problem_D_problem_E_l217_217287

variable {a b c : ℝ}
variable (ha : a < 0) (hab : a < b) (hb : b < 0) (hc : 0 < c)

theorem problem_A (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a * b > a * c :=
by sorry

theorem problem_C (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a * c < b * c :=
by sorry

theorem problem_D (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a + c < b + c :=
by sorry

theorem problem_E (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : c / a > 1 :=
by sorry

end problem_A_problem_C_problem_D_problem_E_l217_217287


namespace polygon_eq_quadrilateral_l217_217341

theorem polygon_eq_quadrilateral (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 := 
sorry

end polygon_eq_quadrilateral_l217_217341


namespace baker_sold_more_pastries_l217_217781

theorem baker_sold_more_pastries {cakes_made pastries_made pastries_sold cakes_sold : ℕ}
    (h1 : cakes_made = 105)
    (h2 : pastries_made = 275)
    (h3 : pastries_sold = 214)
    (h4 : cakes_sold = 163) :
    pastries_sold - cakes_sold = 51 := by
  sorry

end baker_sold_more_pastries_l217_217781


namespace everton_college_payment_l217_217149

theorem everton_college_payment :
  let num_sci_calculators := 20
  let num_graph_calculators := 25
  let price_sci_calculator := 10
  let price_graph_calculator := 57
  let total_payment := num_sci_calculators * price_sci_calculator + num_graph_calculators * price_graph_calculator
  total_payment = 1625 :=
by
  let num_sci_calculators := 20
  let num_graph_calculators := 25
  let price_sci_calculator := 10
  let price_graph_calculator := 57
  let total_payment := num_sci_calculators * price_sci_calculator + num_graph_calculators * price_graph_calculator
  sorry

end everton_college_payment_l217_217149


namespace form_triangle_condition_right_angled_triangle_condition_l217_217255

def vector (α : Type*) := α × α
noncomputable def oa : vector ℝ := ⟨2, -1⟩
noncomputable def ob : vector ℝ := ⟨3, 2⟩
noncomputable def oc (m : ℝ) : vector ℝ := ⟨m, 2 * m + 1⟩

def vector_sub (v1 v2 : vector ℝ) : vector ℝ := ⟨v1.1 - v2.1, v1.2 - v2.2⟩
def vector_dot (v1 v2 : vector ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem form_triangle_condition (m : ℝ) : 
  ¬ ((vector_sub ob oa).1 * (vector_sub (oc m) oa).2 = (vector_sub ob oa).2 * (vector_sub (oc m) oa).1) ↔ m ≠ 8 :=
sorry

theorem right_angled_triangle_condition (m : ℝ) : 
  (vector_dot (vector_sub ob oa) (vector_sub (oc m) oa) = 0 ∨ 
   vector_dot (vector_sub ob oa) (vector_sub (oc m) ob) = 0 ∨ 
   vector_dot (vector_sub (oc m) oa) (vector_sub (oc m) ob) = 0) ↔ 
  (m = -4/7 ∨ m = 6/7) :=
sorry

end form_triangle_condition_right_angled_triangle_condition_l217_217255


namespace problem_a_problem_b_l217_217361

-- Problem (a)
theorem problem_a (n : ℕ) (hn : n > 0) :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

-- Problem (b)
theorem problem_b (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (coprime_ab : Nat.gcd a b = 1)
  (coprime_ac_or_bc : Nat.gcd c a = 1 ∨ Nat.gcd c b = 1) :
  ∃ᶠ x : ℕ in Filter.atTop, ∃ (y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x^a + y^b = z^c :=
sorry

end problem_a_problem_b_l217_217361


namespace number_of_strawberries_in_each_basket_l217_217211

variable (x : ℕ) (Lilibeth_picks : 6 * x)
variable (total_strawberries : 4 * 6 * x = 1200)

theorem number_of_strawberries_in_each_basket : x = 50 := by
  sorry

end number_of_strawberries_in_each_basket_l217_217211


namespace minimum_yellow_marbles_l217_217326

theorem minimum_yellow_marbles :
  ∀ (n y : ℕ), 
  (3 ∣ n) ∧ (4 ∣ n) ∧ 
  (9 + y + 2 * y ≤ n) ∧ 
  (n = n / 3 + n / 4 + 9 + y + 2 * y) → 
  y = 4 :=
by
  sorry

end minimum_yellow_marbles_l217_217326


namespace spike_hunts_20_crickets_per_day_l217_217102

/-- Spike the bearded dragon hunts 5 crickets every morning -/
def spike_morning_crickets : ℕ := 5

/-- Spike hunts three times the morning amount in the afternoon and evening -/
def spike_afternoon_evening_multiplier : ℕ := 3

/-- Total number of crickets Spike hunts per day -/
def spike_total_crickets_per_day : ℕ := spike_morning_crickets + spike_morning_crickets * spike_afternoon_evening_multiplier

/-- Prove that the total number of crickets Spike hunts per day is 20 -/
theorem spike_hunts_20_crickets_per_day : spike_total_crickets_per_day = 20 := 
by
  sorry

end spike_hunts_20_crickets_per_day_l217_217102


namespace spherical_to_rectangular_coordinates_l217_217561

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z)

theorem spherical_to_rectangular_coordinates :
  sphericalToRectangular 10 (5 * Real.pi / 4) (Real.pi / 4) = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_coordinates_l217_217561


namespace solve_for_x_l217_217695

theorem solve_for_x (x : ℝ) (h : 3 * x - 7 = 2 * x + 5) : x = 12 :=
sorry

end solve_for_x_l217_217695


namespace radius_of_regular_polygon_l217_217080

theorem radius_of_regular_polygon :
  ∃ (p : ℝ), 
        (∀ n : ℕ, 3 ≤ n → (n : ℝ) = 6) ∧ 
        (∀ s : ℝ, s = 2 → s = 2) → 
        (∀ i : ℝ, i = 720 → i = 720) →
        (∀ e : ℝ, e = 360 → e = 360) →
        p = 2 :=
by
  sorry

end radius_of_regular_polygon_l217_217080


namespace question1_question2_l217_217305

def energy_cost (units: ℕ) : ℝ :=
  if units <= 100 then
    units * 0.5
  else
    100 * 0.5 + (units - 100) * 0.8

theorem question1 :
  energy_cost 130 = 74 := by
  sorry

theorem question2 (units: ℕ) (H: energy_cost units = 90) :
  units = 150 := by
  sorry

end question1_question2_l217_217305


namespace equilateral_triangle_condition_l217_217961

-- We define points in a plane and vectors between these points
structure Point where
  x : ℝ
  y : ℝ

-- Vector subtraction
def vector (p q : Point) : Point :=
  { x := q.x - p.x, y := q.y - p.y }

-- The equation required to hold for certain type of triangles
def bisector_eq_zero (A B C A1 B1 C1 : Point) : Prop :=
  let AA1 := vector A A1
  let BB1 := vector B B1
  let CC1 := vector C C1
  AA1.x + BB1.x + CC1.x = 0 ∧ AA1.y + BB1.y + CC1.y = 0

-- Property of equilateral triangle
def is_equilateral (A B C : Point) : Prop :=
  let AB := vector A B
  let BC := vector B C
  let CA := vector C A
  (AB.x^2 + AB.y^2 = BC.x^2 + BC.y^2 ∧ BC.x^2 + BC.y^2 = CA.x^2 + CA.y^2)

-- Main theorem statement
theorem equilateral_triangle_condition (A B C A1 B1 C1 : Point)
  (h : bisector_eq_zero A B C A1 B1 C1) :
  is_equilateral A B C :=
sorry

end equilateral_triangle_condition_l217_217961


namespace jeremy_age_l217_217247

theorem jeremy_age
  (A J C : ℕ)
  (h1 : A + J + C = 132)
  (h2 : A = 1 / 3 * J)
  (h3 : C = 2 * A) :
  J = 66 :=
sorry

end jeremy_age_l217_217247


namespace jonathans_and_sisters_total_letters_l217_217166

theorem jonathans_and_sisters_total_letters:
  (jonathan_first: Nat) = 8 ∧
  (jonathan_surname: Nat) = 10 ∧
  (sister_first: Nat) = 5 ∧
  (sister_surname: Nat) = 10 →
  jonathan_first + jonathan_surname + sister_first + sister_surname = 33 := by
  intros
  sorry

end jonathans_and_sisters_total_letters_l217_217166


namespace sqrt_mul_l217_217615

theorem sqrt_mul (h₁ : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3) : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 :=
by
  sorry

end sqrt_mul_l217_217615


namespace emily_has_7_times_more_oranges_than_sandra_l217_217288

theorem emily_has_7_times_more_oranges_than_sandra
  (B S E : ℕ)
  (h1 : S = 3 * B)
  (h2 : B = 12)
  (h3 : E = 252) :
  ∃ k : ℕ, E = k * S ∧ k = 7 :=
by
  use 7
  sorry

end emily_has_7_times_more_oranges_than_sandra_l217_217288


namespace solve_mod_equation_l217_217507

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem solve_mod_equation (u : ℕ) (h1 : is_two_digit_positive_integer u) (h2 : 13 * u % 100 = 52) : u = 4 :=
sorry

end solve_mod_equation_l217_217507


namespace original_rectangle_length_l217_217935

-- Define the problem conditions
def length_three_times_width (l w : ℕ) : Prop :=
  l = 3 * w

def length_decreased_width_increased (l w : ℕ) : Prop :=
  l - 5 = w + 5

-- Define the proof problem
theorem original_rectangle_length (l w : ℕ) (H1 : length_three_times_width l w) (H2 : length_decreased_width_increased l w) : l = 15 :=
sorry

end original_rectangle_length_l217_217935


namespace Julia_played_kids_on_Monday_l217_217194

theorem Julia_played_kids_on_Monday
  (t : ℕ) (w : ℕ) (h1 : t = 18) (h2 : w = 97) (h3 : t + m = 33) :
  ∃ m : ℕ, m = 15 :=
by
  sorry

end Julia_played_kids_on_Monday_l217_217194


namespace max_area_square_pen_l217_217783

theorem max_area_square_pen (P : ℝ) (h1 : P = 64) : ∃ A : ℝ, A = 256 := 
by
  sorry

end max_area_square_pen_l217_217783


namespace price_difference_pc_sm_l217_217245

-- Definitions based on given conditions
def S : ℕ := 300
def x : ℕ := sorry -- This is what we are trying to find
def PC : ℕ := S + x
def AT : ℕ := S + PC
def total_cost : ℕ := S + PC + AT

-- Theorem to be proved
theorem price_difference_pc_sm (h : total_cost = 2200) : x = 500 :=
by
  -- We would prove the theorem here
  sorry

end price_difference_pc_sm_l217_217245


namespace sum_of_n_values_l217_217544

theorem sum_of_n_values (sum_n : ℕ) : (∀ n : ℕ, 0 < n ∧ 24 % (2 * n - 1) = 0) → sum_n = 3 :=
by
  sorry

end sum_of_n_values_l217_217544


namespace sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C_l217_217268

theorem sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C
  {A B C : ℝ}
  (h : A + B + C = π) :
  Real.sin (4 * A) + Real.sin (4 * B) + Real.sin (4 * C) = -4 * Real.sin (2 * A) * Real.sin (2 * B) * Real.sin (2 * C) :=
sorry

end sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C_l217_217268


namespace hexagon_side_lengths_l217_217903

theorem hexagon_side_lengths (n m : ℕ) (AB BC : ℕ) (P : ℕ) :
  n + m = 6 ∧ n * 4 + m * 7 = 38 ∧ AB = 4 ∧ BC = 7 → m = 4 :=
by
  sorry

end hexagon_side_lengths_l217_217903


namespace gcd_12012_18018_l217_217552

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l217_217552


namespace coeff_fourth_term_expansion_l217_217513

theorem coeff_fourth_term_expansion :
  (3 : ℚ) ^ 2 * (-1 : ℚ) / 8 * (Nat.choose 8 3) = -63 :=
by
  sorry

end coeff_fourth_term_expansion_l217_217513


namespace linear_condition_l217_217879

theorem linear_condition (m : ℝ) : ¬ (m = 2) ↔ (∃ f : ℝ → ℝ, ∀ x, f x = (m - 2) * x + 2) :=
by
  sorry

end linear_condition_l217_217879


namespace stating_martha_painting_time_l217_217520

/-- 
  Theorem stating the time it takes for Martha to paint the kitchen is 42 hours.
-/
theorem martha_painting_time :
  let width1 := 12
  let width2 := 16
  let height := 10
  let area_pair1 := 2 * width1 * height
  let area_pair2 := 2 * width2 * height
  let total_area := area_pair1 + area_pair2
  let coats := 3
  let total_paint_area := total_area * coats
  let painting_speed := 40
  let time_required := total_paint_area / painting_speed
  time_required = 42 := by
    -- Since we are asked not to provide the proof steps, we use sorry to skip the proof.
    sorry

end stating_martha_painting_time_l217_217520


namespace solution_set_for_rational_inequality_l217_217515

theorem solution_set_for_rational_inequality (x : ℝ) :
  (x - 2) / (x - 1) > 0 ↔ x < 1 ∨ x > 2 := 
sorry

end solution_set_for_rational_inequality_l217_217515


namespace find_A_l217_217661

theorem find_A (d q r A : ℕ) (h1 : d = 7) (h2 : q = 5) (h3 : r = 3) (h4 : A = d * q + r) : A = 38 := 
by 
  { sorry }

end find_A_l217_217661


namespace find_square_number_divisible_by_three_between_90_and_150_l217_217721

theorem find_square_number_divisible_by_three_between_90_and_150 :
  ∃ x : ℕ, 90 < x ∧ x < 150 ∧ ∃ y : ℕ, x = y * y ∧ 3 ∣ x ∧ x = 144 := 
by 
  sorry

end find_square_number_divisible_by_three_between_90_and_150_l217_217721


namespace calculate_c_from_law_of_cosines_l217_217799

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : Prop :=
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B

theorem calculate_c_from_law_of_cosines 
  (a b c : ℝ) (B : ℝ)
  (ha : a = 8) (hb : b = 7) (hB : B = Real.pi / 3) : 
  (c = 3) ∨ (c = 5) :=
sorry

end calculate_c_from_law_of_cosines_l217_217799


namespace Elizabeth_lost_bottles_l217_217324

theorem Elizabeth_lost_bottles :
  ∃ (L : ℕ), (10 - L - 1) * 3 = 21 ∧ L = 2 := by
  sorry

end Elizabeth_lost_bottles_l217_217324


namespace impossibility_of_transition_l217_217076

theorem impossibility_of_transition 
  {a b c : ℤ}
  (h1 : a = 2)
  (h2 : b = 2)
  (h3 : c = 2) :
  ¬(∃ x y z : ℤ, x = 19 ∧ y = 1997 ∧ z = 1999 ∧
    (∃ n : ℕ, ∀ i < n, ∃ a' b' c' : ℤ, 
      if i = 0 then a' = 2 ∧ b' = 2 ∧ c' = 2 
      else (a', b', c') = 
        if i % 3 = 0 then (b + c - 1, b, c)
        else if i % 3 = 1 then (a, a + c - 1, c)
        else (a, b, a + b - 1) 
  )) :=
sorry

end impossibility_of_transition_l217_217076


namespace John_height_l217_217506

open Real

variable (John Mary Tom Angela Helen Amy Becky Carl : ℝ)

axiom h1 : John = 1.5 * Mary
axiom h2 : Mary = 2 * Tom
axiom h3 : Tom = Angela - 70
axiom h4 : Angela = Helen + 4
axiom h5 : Helen = Amy + 3
axiom h6 : Amy = 1.2 * Becky
axiom h7 : Becky = 2 * Carl
axiom h8 : Carl = 120

theorem John_height : John = 675 := by
  sorry

end John_height_l217_217506


namespace inequality_always_holds_l217_217874

theorem inequality_always_holds (a : ℝ) (h : a ≥ -2) : ∀ (x : ℝ), x^2 + a * |x| + 1 ≥ 0 :=
by
  sorry

end inequality_always_holds_l217_217874


namespace min_value_expr_l217_217353

-- Definition of the expression given a real constant k
def expr (k : ℝ) (x y : ℝ) : ℝ := 9 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9

-- The proof problem statement
theorem min_value_expr (k : ℝ) (h : k = 2 / 9) : ∃ x y : ℝ, expr k x y = 1 ∧ ∀ x y : ℝ, expr k x y ≥ 1 :=
by
  sorry

end min_value_expr_l217_217353


namespace tangent_condition_l217_217881

theorem tangent_condition (a b : ℝ) :
  (4 * a^2 + b^2 = 1) ↔ 
  ∀ x y : ℝ, (y = 2 * x + 1) → ((x^2 / a^2) + (y^2 / b^2) = 1) → (∃! y, y = 2 * x + 1 ∧ (x^2 / a^2) + (y^2 / b^2) = 1) :=
sorry

end tangent_condition_l217_217881


namespace find_x0_l217_217630

/-- Given that the tangent line to the curve y = x^2 - 1 at the point x = x0 is parallel 
to the tangent line to the curve y = 1 - x^3 at the point x = x0, prove that x0 = 0 
or x0 = -2/3. -/
theorem find_x0 (x0 : ℝ) (h : (∃ x0, (2 * x0) = (-3 * x0 ^ 2))) : x0 = 0 ∨ x0 = -2/3 := 
sorry

end find_x0_l217_217630


namespace odometer_problem_l217_217435

theorem odometer_problem
    (x a b c : ℕ)
    (h_dist : 60 * x = (100 * b + 10 * c + a) - (100 * a + 10 * b + c))
    (h_b_ge_1 : b ≥ 1)
    (h_sum_le_9 : a + b + c ≤ 9) :
    a^2 + b^2 + c^2 = 29 :=
sorry

end odometer_problem_l217_217435


namespace arithmetic_sequence_squares_l217_217136

theorem arithmetic_sequence_squares (a b c : ℝ) :
  (1 / (a + b) - 1 / (b + c) = 1 / (c + a) - 1 / (b + c)) →
  (2 * b^2 = a^2 + c^2) :=
by
  intro h
  sorry

end arithmetic_sequence_squares_l217_217136


namespace red_light_adds_3_minutes_l217_217954

-- Definitions (conditions)
def first_route_time_if_all_green := 10
def second_route_time := 14
def additional_time_if_all_red := 5

-- Given that the first route is 5 minutes longer when all stoplights are red
def first_route_time_if_all_red := second_route_time + additional_time_if_all_red

-- Define red_light_time as the time each stoplight adds if it is red
def red_light_time := (first_route_time_if_all_red - first_route_time_if_all_green) / 3

-- Theorem (question == answer)
theorem red_light_adds_3_minutes :
  red_light_time = 3 :=
by
  -- proof goes here
  sorry

end red_light_adds_3_minutes_l217_217954


namespace abc_inequality_l217_217335

theorem abc_inequality 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
by
  sorry

end abc_inequality_l217_217335


namespace points_3_units_away_from_origin_l217_217269

theorem points_3_units_away_from_origin (a : ℝ) (h : |a| = 3) : a = 3 ∨ a = -3 := by
  sorry

end points_3_units_away_from_origin_l217_217269


namespace speed_of_current_l217_217638

-- Define the conditions in Lean
theorem speed_of_current (c : ℝ) (r : ℝ) 
  (hu : c - r = 12 / 6) -- upstream speed equation
  (hd : c + r = 12 / 0.75) -- downstream speed equation
  : r = 7 := 
sorry

end speed_of_current_l217_217638


namespace find_xyz_l217_217909

theorem find_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 17) 
  (h3 : x^3 + y^3 + z^3 = 27) : 
  x * y * z = 32 / 3 :=
  sorry

end find_xyz_l217_217909


namespace angle_AM_BN_60_degrees_area_triangle_ABP_eq_area_quadrilateral_MDNP_l217_217285

-- Definitions according to the given conditions
variables (A B C D E F M N P : Point)
  (hexagon_regular : is_regular_hexagon A B C D E F)
  (is_midpoint_M : is_midpoint M C D)
  (is_midpoint_N : is_midpoint N D E)
  (intersection_P : intersection_point P (line_through A M) (line_through B N))

-- Angle between AM and BN is 60 degrees
theorem angle_AM_BN_60_degrees 
  (h1 : hexagon_regular)
  (h2 : is_midpoint_M)
  (h3 : is_midpoint_N)
  (h4 : intersection_P) :
  angle (line_through A M) (line_through B N) = 60 := 
sorry

-- Area of triangle ABP is equal to the area of quadrilateral MDNP
theorem area_triangle_ABP_eq_area_quadrilateral_MDNP 
  (h1 : hexagon_regular)
  (h2 : is_midpoint_M)
  (h3 : is_midpoint_N)
  (h4 : intersection_P) :
  area (triangle A B P) = area (quadrilateral M D N P) := 
sorry

end angle_AM_BN_60_degrees_area_triangle_ABP_eq_area_quadrilateral_MDNP_l217_217285


namespace sqrt_infinite_series_eq_two_l217_217426

theorem sqrt_infinite_series_eq_two (m : ℝ) (hm : 0 < m) :
  (m ^ 2 = 2 + m) → m = 2 :=
by {
  sorry
}

end sqrt_infinite_series_eq_two_l217_217426


namespace smallest_n_for_multiple_of_7_l217_217086

theorem smallest_n_for_multiple_of_7 (x y : ℤ) (h1 : x % 7 = -1 % 7) (h2 : y % 7 = 2 % 7) :
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n) % 7 = 0 ∧ n = 4 :=
sorry

end smallest_n_for_multiple_of_7_l217_217086


namespace move_line_upwards_l217_217769

theorem move_line_upwards (x y : ℝ) :
  (y = -x + 1) → (y + 5 = -x + 6) :=
by
  intro h
  sorry

end move_line_upwards_l217_217769


namespace common_ratio_of_geometric_sequence_l217_217817

variable (a₁ q : ℝ)

def geometric_sequence (n : ℕ) := a₁ * q^n

theorem common_ratio_of_geometric_sequence
  (h_sum : geometric_sequence a₁ q 0 + geometric_sequence a₁ q 1 + geometric_sequence a₁ q 2 = 3 * a₁) :
  q = 1 ∨ q = -2 :=
sorry

end common_ratio_of_geometric_sequence_l217_217817


namespace scientific_notation_of_0_0000000005_l217_217358

theorem scientific_notation_of_0_0000000005 : 0.0000000005 = 5 * 10^(-10) :=
by {
  sorry
}

end scientific_notation_of_0_0000000005_l217_217358


namespace paint_walls_l217_217514

theorem paint_walls (d h e : ℕ) : 
  ∃ (x : ℕ), (d * d * e = 2 * h * h * x) ↔ x = (d^2 * e) / (2 * h^2) := by
  sorry

end paint_walls_l217_217514


namespace largest_sum_is_5_over_6_l217_217511

def sum_1 := (1/3) + (1/7)
def sum_2 := (1/3) + (1/8)
def sum_3 := (1/3) + (1/2)
def sum_4 := (1/3) + (1/9)
def sum_5 := (1/3) + (1/4)

theorem largest_sum_is_5_over_6 : (sum_3 = 5/6) ∧ ((sum_3 > sum_1) ∧ (sum_3 > sum_2) ∧ (sum_3 > sum_4) ∧ (sum_3 > sum_5)) :=
by
  sorry

end largest_sum_is_5_over_6_l217_217511


namespace ratio_unit_price_l217_217412

theorem ratio_unit_price
  (v : ℝ) (p : ℝ) (h_v : v > 0) (h_p : p > 0)
  (vol_A : ℝ := 1.25 * v)
  (price_A : ℝ := 0.85 * p) :
  (price_A / vol_A) / (p / v) = 17 / 25 :=
by
  sorry

end ratio_unit_price_l217_217412


namespace part1_l217_217768

def purchase_price (x y : ℕ) : Prop := 25 * x + 30 * y = 1500
def quantity_relation (x y : ℕ) : Prop := x = 2 * y - 4

theorem part1 (x y : ℕ) (h1 : purchase_price x y) (h2 : quantity_relation x y) : x = 36 ∧ y = 20 :=
sorry

end part1_l217_217768


namespace negation_of_p_implies_a_gt_one_half_l217_217294

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, a * x^2 + x + 1 / 2 ≤ 0

-- Define the statement that negation of p implies a > 1/2
theorem negation_of_p_implies_a_gt_one_half (a : ℝ) (h : ¬ p a) : a > 1 / 2 :=
by
  sorry

end negation_of_p_implies_a_gt_one_half_l217_217294


namespace problem_statement_l217_217999

variable {x y z : ℝ}

-- Lean 4 statement of the problem
theorem problem_statement (h₀ : 0 ≤ x) (h₁ : x ≤ 1) (h₂ : 0 ≤ y) (h₃ : y ≤ 1) (h₄ : 0 ≤ z) (h₅ : z ≤ 1) :
  (x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
  sorry

end problem_statement_l217_217999


namespace optimal_hospital_location_l217_217578

-- Define the coordinates for points A, B, and C
def A : ℝ × ℝ := (0, 12)
def B : ℝ × ℝ := (-5, 0)
def C : ℝ × ℝ := (5, 0)

-- Define the distance function
def dist_sq (p q : ℝ × ℝ) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Define the statement to be proved: minimizing sum of squares of distances
theorem optimal_hospital_location : ∃ y : ℝ, 
  (∀ (P : ℝ × ℝ), P = (0, y) → (dist_sq P A + dist_sq P B + dist_sq P C) = 146) ∧ y = 4 :=
by sorry

end optimal_hospital_location_l217_217578


namespace tiger_speed_l217_217452

variable (v_t : ℝ) (hours_head_start : ℝ := 5) (hours_zebra_to_catch : ℝ := 6) (speed_zebra : ℝ := 55)

-- Define the distance covered by the tiger and the zebra
def distance_tiger (v_t : ℝ) (hours : ℝ) : ℝ := v_t * hours
def distance_zebra (hours : ℝ) (speed_zebra : ℝ) : ℝ := speed_zebra * hours

theorem tiger_speed :
  v_t * hours_head_start + v_t * hours_zebra_to_catch = distance_zebra hours_zebra_to_catch speed_zebra →
  v_t = 30 :=
by
  sorry

end tiger_speed_l217_217452


namespace total_number_of_apples_l217_217314

namespace Apples

def red_apples : ℕ := 7
def green_apples : ℕ := 2
def total_apples : ℕ := red_apples + green_apples

theorem total_number_of_apples : total_apples = 9 := by
  -- Definition of total_apples is used directly from conditions.
  -- Conditions state there are 7 red apples and 2 green apples.
  -- Therefore, total_apples = 7 + 2 = 9.
  sorry

end Apples

end total_number_of_apples_l217_217314


namespace rationalize_fraction_l217_217461

open BigOperators

theorem rationalize_fraction :
  (3 : ℝ) / (Real.sqrt 50 + 2) = (15 * Real.sqrt 2 - 6) / 46 :=
by
  -- Our proof intention will be inserted here.
  sorry

end rationalize_fraction_l217_217461


namespace part1_part2_l217_217158

def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

theorem part1 (x : ℝ) (h : f x 2 ≥ 2) : x ≤ 1/2 ∨ x ≥ 2.5 := by
  sorry

theorem part2 (a : ℝ) (h_even : ∀ x : ℝ, f (-x) a = f x a) : a = -1 := by
  sorry

end part1_part2_l217_217158


namespace remove_green_balls_l217_217455

theorem remove_green_balls (total_balls green_balls yellow_balls x : ℕ) 
  (h1 : total_balls = 600)
  (h2 : green_balls = 420)
  (h3 : yellow_balls = 180)
  (h4 : green_balls = 70 * total_balls / 100)
  (h5 : yellow_balls = total_balls - green_balls)
  (h6 : (green_balls - x) = 60 * (total_balls - x) / 100) :
  x = 150 := 
by {
  -- sorry placeholder for proof.
  sorry
}

end remove_green_balls_l217_217455


namespace eval_expression_l217_217596

theorem eval_expression : 4^3 - 2 * 4^2 + 2 * 4 - 1 = 39 :=
by 
  -- Here we would write the proof, but according to the instructions we skip it with sorry.
  sorry

end eval_expression_l217_217596


namespace external_tangent_twice_internal_tangent_l217_217320

noncomputable def distance_between_centers (r R : ℝ) : ℝ :=
  Real.sqrt (R^2 + r^2 + (10/3) * R * r)

theorem external_tangent_twice_internal_tangent 
  (r R O₁O₂ AB CD : ℝ)
  (h₁ : AB = 2 * CD)
  (h₂ : AB^2 = O₁O₂^2 - (R - r)^2)
  (h₃ : CD^2 = O₁O₂^2 - (R + r)^2) :
  O₁O₂ = distance_between_centers r R :=
by
  sorry

end external_tangent_twice_internal_tangent_l217_217320


namespace total_amount_proof_l217_217562

-- Definitions of the base 8 numbers
def silks_base8 := 5267
def stones_base8 := 6712
def spices_base8 := 327

-- Conversion function from base 8 to base 10
def base8_to_base10 (n : ℕ) : ℕ := sorry -- Assume this function converts a base 8 number to base 10

-- Converted values
def silks_base10 := base8_to_base10 silks_base8
def stones_base10 := base8_to_base10 stones_base8
def spices_base10 := base8_to_base10 spices_base8

-- Total amount calculation in base 10
def total_amount_base10 := silks_base10 + stones_base10 + spices_base10

-- The theorem that we want to prove
theorem total_amount_proof : total_amount_base10 = 6488 :=
by
  -- The proof is omitted here.
  sorry

end total_amount_proof_l217_217562


namespace part1_part2_l217_217788

-- Definitions for the problem
def f (x a : ℝ) : ℝ := |x - a| + 3 * x

-- Part (1)
theorem part1 (x : ℝ) (h : f x 1 ≥ 3 * x + 2) : x ≥ 3 ∨ x ≤ -1 :=
sorry

-- Part (2)
theorem part2 (h : ∀ x, f x a ≤ 0 → x ≤ -1) : a = 2 :=
sorry

end part1_part2_l217_217788


namespace problem_l217_217395

-- Conditions
def a_n (n : ℕ) : ℚ := (1/3)^(n-1)

def b_n (n : ℕ) : ℚ := n * (1/3)^n

-- Sums over the first n terms
def S_n (n : ℕ) : ℚ := (3/2) - (1/2) * (1/3)^n

def T_n (n : ℕ) : ℚ := (3/4) - (1/4) * (1/3)^n - (n/2) * (1/3)^n

-- Problem: Prove T_n < S_n / 2
theorem problem (n : ℕ) : T_n n < S_n n / 2 :=
by sorry

end problem_l217_217395


namespace radius_of_circle_l217_217648

theorem radius_of_circle
  (r : ℝ) (r_pos : r > 0)
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1^2 + y1^2 = r^2)
  (h2 : x2^2 + y2^2 = r^2)
  (h3 : x1 + y1 = 3)
  (h4 : x2 + y2 = 3)
  (h5 : x1 * x2 + y1 * y2 = -0.5 * r^2) : 
  r = 3 * Real.sqrt 2 :=
by
  sorry

end radius_of_circle_l217_217648


namespace fill_pipe_half_cistern_time_l217_217197

theorem fill_pipe_half_cistern_time (time_to_fill_half : ℕ) 
  (H : time_to_fill_half = 10) : 
  time_to_fill_half = 10 := 
by
  -- Proof is omitted
  sorry

end fill_pipe_half_cistern_time_l217_217197


namespace factorization_of_polynomial_l217_217633

theorem factorization_of_polynomial (x : ℝ) : 2 * x^2 - 12 * x + 18 = 2 * (x - 3)^2 := by
  sorry

end factorization_of_polynomial_l217_217633


namespace floor_sqrt_sum_eq_floor_sqrt_expr_l217_217129

-- Proof problem definition
theorem floor_sqrt_sum_eq_floor_sqrt_expr (n : ℕ) : 
  (Int.floor (Real.sqrt n + Real.sqrt (n + 1))) = (Int.floor (Real.sqrt (4 * n + 2))) := 
sorry

end floor_sqrt_sum_eq_floor_sqrt_expr_l217_217129


namespace number_of_red_balls_l217_217672

-- Conditions
variables (w r : ℕ)
variable (ratio_condition : 4 * r = 3 * w)
variable (white_balls : w = 8)

-- Prove the number of red balls
theorem number_of_red_balls : r = 6 :=
by
  sorry

end number_of_red_balls_l217_217672


namespace f_const_one_l217_217365

-- Mathematical Translation of the Definitions
variable (f g h : ℕ → ℕ)

-- Given conditions
axiom h_injective : Function.Injective h
axiom g_surjective : Function.Surjective g
axiom f_eq : ∀ n, f n = g n - h n + 1

-- Theorem to Prove
theorem f_const_one : ∀ n, f n = 1 :=
by
  sorry

end f_const_one_l217_217365


namespace f_nested_seven_l217_217219

-- Definitions for the given conditions
variables (f : ℝ → ℝ) (odd_f : ∀ x, f (-x) = -f x)
variables (period_f : ∀ x, f (x + 4) = f x)
variables (f_one : f 1 = 4)

theorem f_nested_seven (f : ℝ → ℝ)
  (odd_f : ∀ x, f (-x) = -f x)
  (period_f : ∀ x, f (x + 4) = f x)
  (f_one : f 1 = 4) :
  f (f 7) = 0 :=
sorry

end f_nested_seven_l217_217219


namespace find_original_number_l217_217626

theorem find_original_number (a b c : ℕ) (h : 100 * a + 10 * b + c = 390) 
  (N : ℕ) (hN : N = 4326) : a = 3 ∧ b = 9 ∧ c = 0 :=
by 
  sorry

end find_original_number_l217_217626


namespace busy_squirrels_count_l217_217363

variable (B : ℕ)
variable (busy_squirrel_nuts_per_day : ℕ := 30)
variable (sleepy_squirrel_nuts_per_day : ℕ := 20)
variable (days : ℕ := 40)
variable (total_nuts : ℕ := 3200)

theorem busy_squirrels_count : busy_squirrel_nuts_per_day * days * B + sleepy_squirrel_nuts_per_day * days = total_nuts → B = 2 := by
  sorry

end busy_squirrels_count_l217_217363


namespace find_c_l217_217342

theorem find_c (a b c : ℤ) (h1 : a + b * c = 2017) (h2 : b + c * a = 8) :
  c = -6 ∨ c = 0 ∨ c = 2 ∨ c = 8 :=
by 
  sorry

end find_c_l217_217342


namespace triangle_inscribed_circle_area_l217_217306

noncomputable def circle_radius (circumference : ℝ) : ℝ :=
  circumference / (2 * Real.pi)

noncomputable def triangle_area (r : ℝ) : ℝ :=
  (1 / 2) * r^2 * (Real.sin (Real.pi / 2) + Real.sin (2 * Real.pi / 3) + Real.sin (5 * Real.pi / 6))

theorem triangle_inscribed_circle_area (a b c : ℝ) (h : a + b + c = 24) :
  ∀ (r : ℝ) (h_r : r = circle_radius 24),
  triangle_area r = 72 / Real.pi^2 * (Real.sqrt 3 + 1) :=
by
  intro r h_r
  rw [h_r, circle_radius, triangle_area]
  sorry

end triangle_inscribed_circle_area_l217_217306


namespace probability_king_then_ten_l217_217183

-- Define the conditions
def standard_deck_size : ℕ := 52
def num_kings : ℕ := 4
def num_tens : ℕ := 4

-- Define the event probabilities
def prob_first_card_king : ℚ := num_kings / standard_deck_size
def prob_second_card_ten (remaining_deck_size : ℕ) : ℚ := num_tens / remaining_deck_size

-- The theorem statement to be proved
theorem probability_king_then_ten : 
  prob_first_card_king * prob_second_card_ten (standard_deck_size - 1) = 4 / 663 :=
by
  sorry

end probability_king_then_ten_l217_217183


namespace abs_neg_number_l217_217640

theorem abs_neg_number : abs (-2023) = 2023 := sorry

end abs_neg_number_l217_217640


namespace max_colors_for_valid_coloring_l217_217654

-- Define the 4x4 grid as a type synonym for a set of cells
def Grid4x4 := Fin 4 × Fin 4

-- Condition: Define a valid coloring function for a 4x4 grid
def valid_coloring (colors : ℕ) (f : Grid4x4 → Fin colors) : Prop :=
  ∀ i j : Fin 3, ∃ c : Fin colors, (f (i, j) = c ∨ f (i+1, j) = c) ∧ (f (i+1, j) = c ∨ f (i, j+1) = c)

-- The main theorem to prove
theorem max_colors_for_valid_coloring : 
  ∃ (colors : ℕ), colors = 11 ∧ ∀ f : Grid4x4 → Fin colors, valid_coloring colors f :=
sorry

end max_colors_for_valid_coloring_l217_217654


namespace opposite_sides_of_line_l217_217822

theorem opposite_sides_of_line (a : ℝ) (h1 : 0 < a) (h2 : a < 2) : (-a) * (2 - a) < 0 :=
sorry

end opposite_sides_of_line_l217_217822


namespace ana_bonita_age_difference_l217_217002

theorem ana_bonita_age_difference (A B n : ℕ) 
  (h1 : A = B + n)
  (h2 : A - 1 = 7 * (B - 1))
  (h3 : A = B^3) : 
  n = 6 :=
sorry

end ana_bonita_age_difference_l217_217002


namespace solution_m_in_interval_l217_217928

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x < 1 then -x^2 + 2 * m * x - 2 else 1 + Real.log x

theorem solution_m_in_interval :
  ∃ m : ℝ, (1 ≤ m ∧ m ≤ 2) ∧
  (∀ x < 1, ∀ y < 1, x ≤ y → f x m ≤ f y m) ∧
  (∀ x ≥ 1, ∀ y ≥ 1, x ≤ y → f x m ≤ f y m) ∧
  (∀ x < 1, ∀ y ≥ 1, f x m ≤ f y m) :=
by
  sorry

end solution_m_in_interval_l217_217928


namespace maximize_x_minus_y_plus_z_l217_217191

-- Define the given condition as a predicate
def given_condition (x y z : ℝ) : Prop :=
  2 * x^2 + y^2 + z^2 = 2 * x - 4 * y + 2 * x * z - 5

-- Define the statement we want to prove
theorem maximize_x_minus_y_plus_z :
  ∃ x y z : ℝ, given_condition x y z ∧ (x - y + z = 4) :=
by
  sorry

end maximize_x_minus_y_plus_z_l217_217191


namespace pool_length_l217_217919

theorem pool_length (r : ℕ) (t : ℕ) (w : ℕ) (d : ℕ) (L : ℕ) 
  (H1 : r = 60)
  (H2 : t = 2000)
  (H3 : w = 80)
  (H4 : d = 10)
  (H5 : L = (r * t) / (w * d)) : L = 150 :=
by
  rw [H1, H2, H3, H4] at H5
  exact H5


end pool_length_l217_217919


namespace common_chord_line_l217_217065

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 4 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 3 = 0

-- Definition of the line equation for the common chord
def line (x y : ℝ) : Prop := 2*x - 2*y + 7 = 0

theorem common_chord_line (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : line x y :=
by
  sorry

end common_chord_line_l217_217065


namespace fraction_of_juniors_equals_seniors_l217_217052

theorem fraction_of_juniors_equals_seniors (J S : ℕ) (h1 : 0 < J) (h2 : 0 < S) (h3 : J * 7 = 4 * (J + S)) : J / S = 4 / 3 :=
sorry

end fraction_of_juniors_equals_seniors_l217_217052


namespace addition_example_l217_217212

theorem addition_example : 248 + 64 = 312 := by
  sorry

end addition_example_l217_217212


namespace find_angle_sum_l217_217499

theorem find_angle_sum
  {α β : ℝ}
  (hα_acute : 0 < α ∧ α < π / 2)
  (hβ_acute : 0 < β ∧ β < π / 2)
  (h_tan_α : Real.tan α = 1 / 3)
  (h_cos_β : Real.cos β = 3 / 5) :
  α + 2 * β = π - Real.arctan (13 / 9) :=
sorry

end find_angle_sum_l217_217499


namespace interval_length_l217_217433

theorem interval_length (c : ℝ) (h : ∀ x : ℝ, 3 ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ c → 
                             (3 * (x) + 4 ≤ c ∧ 3 ≤ 3 * x + 4)) :
  (∃ c : ℝ, ((c - 4) / 3) - ((-1) / 3) = 15) → (c - 3 = 45) :=
sorry

end interval_length_l217_217433


namespace find_amount_l217_217729

theorem find_amount (x : ℝ) (A : ℝ) (h1 : 0.65 * x = 0.20 * A) (h2 : x = 230) : A = 747.5 := by
  sorry

end find_amount_l217_217729


namespace study_group_members_l217_217139

theorem study_group_members (x : ℕ) (h : x * (x - 1) = 90) : x = 10 :=
sorry

end study_group_members_l217_217139


namespace sqrt_seven_to_six_power_eq_343_l217_217812

theorem sqrt_seven_to_six_power_eq_343 : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_six_power_eq_343_l217_217812


namespace alice_cell_phone_cost_l217_217939

theorem alice_cell_phone_cost
  (base_cost : ℕ)
  (included_hours : ℕ)
  (text_cost_per_message : ℕ)
  (extra_minute_cost : ℕ)
  (messages_sent : ℕ)
  (hours_spent : ℕ) :
  base_cost = 25 →
  included_hours = 40 →
  text_cost_per_message = 4 →
  extra_minute_cost = 5 →
  messages_sent = 150 →
  hours_spent = 42 →
  (base_cost + (messages_sent * text_cost_per_message) / 100 + ((hours_spent - included_hours) * 60 * extra_minute_cost) / 100) = 37 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end alice_cell_phone_cost_l217_217939


namespace total_granola_bars_l217_217022

-- Problem conditions
def oatmeal_raisin_bars : ℕ := 6
def peanut_bars : ℕ := 8

-- Statement to prove
theorem total_granola_bars : oatmeal_raisin_bars + peanut_bars = 14 := 
by 
  sorry

end total_granola_bars_l217_217022


namespace shaniqua_earnings_correct_l217_217841

noncomputable def calc_earnings : ℝ :=
  let haircut_tuesday := 5 * 10
  let haircut_normal := 5 * 12
  let styling_vip := (6 * 25) * (1 - 0.2)
  let styling_regular := 4 * 25
  let coloring_friday := (7 * 35) * (1 - 0.15)
  let coloring_normal := 3 * 35
  let treatment_senior := (3 * 50) * (1 - 0.1)
  let treatment_other := 4 * 50
  haircut_tuesday + haircut_normal + styling_vip + styling_regular + coloring_friday + coloring_normal + treatment_senior + treatment_other

theorem shaniqua_earnings_correct : calc_earnings = 978.25 := by
  sorry

end shaniqua_earnings_correct_l217_217841


namespace largest_composite_in_five_consecutive_ints_l217_217641

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_of_five_composite_ints : ℕ :=
  36

theorem largest_composite_in_five_consecutive_ints (a b c d e : ℕ) :
  a < 40 ∧ b < 40 ∧ c < 40 ∧ d < 40 ∧ e < 40 ∧ 
  ¬is_prime a ∧ ¬is_prime b ∧ ¬is_prime c ∧ ¬is_prime d ∧ ¬is_prime e ∧ 
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a = 32 ∧ b = 33 ∧ c = 34 ∧ d = 35 ∧ e = 36 →
  e = largest_of_five_composite_ints :=
by 
  sorry

end largest_composite_in_five_consecutive_ints_l217_217641


namespace rachel_study_time_l217_217727

-- Define the conditions
def pages_math := 2
def pages_reading := 3
def pages_biology := 10
def pages_history := 4
def pages_physics := 5
def pages_chemistry := 8

def total_pages := pages_math + pages_reading + pages_biology + pages_history + pages_physics + pages_chemistry

def percent_study_time_biology := 30
def percent_study_time_reading := 30

-- State the theorem
theorem rachel_study_time :
  percent_study_time_biology = 30 ∧ 
  percent_study_time_reading = 30 →
  (100 - (percent_study_time_biology + percent_study_time_reading)) = 40 :=
by
  sorry

end rachel_study_time_l217_217727


namespace value_of_7_prime_prime_l217_217827

-- Define the function q' (written as q_prime in Lean)
def q_prime (q : ℕ) : ℕ := 3 * q - 3

-- Define the specific value problem
theorem value_of_7_prime_prime : q_prime (q_prime 7) = 51 := by
  sorry

end value_of_7_prime_prime_l217_217827


namespace ratio_problem_l217_217868

theorem ratio_problem (a b c d : ℝ) (h1 : a / b = 5) (h2 : b / c = 1 / 2) (h3 : c / d = 6) : 
  d / a = 1 / 15 :=
by sorry

end ratio_problem_l217_217868


namespace sandy_ordered_three_cappuccinos_l217_217838

-- Definitions and conditions
def cost_cappuccino : ℝ := 2
def cost_iced_tea : ℝ := 3
def cost_cafe_latte : ℝ := 1.5
def cost_espresso : ℝ := 1
def num_iced_teas : ℕ := 2
def num_cafe_lattes : ℕ := 2
def num_espressos : ℕ := 2
def change_received : ℝ := 3
def amount_paid : ℝ := 20

-- Calculation of costs
def total_cost_iced_teas : ℝ := num_iced_teas * cost_iced_tea
def total_cost_cafe_lattes : ℝ := num_cafe_lattes * cost_cafe_latte
def total_cost_espressos : ℝ := num_espressos * cost_espresso
def total_cost_other_drinks : ℝ := total_cost_iced_teas + total_cost_cafe_lattes + total_cost_espressos
def total_spent : ℝ := amount_paid - change_received
def cost_cappuccinos := total_spent - total_cost_other_drinks

-- Proof statement
theorem sandy_ordered_three_cappuccinos (num_cappuccinos : ℕ) : cost_cappuccinos = num_cappuccinos * cost_cappuccino → num_cappuccinos = 3 :=
by sorry

end sandy_ordered_three_cappuccinos_l217_217838


namespace tan_theta_cos_double_angle_minus_pi_over_3_l217_217693

open Real

-- Given conditions
variable (θ : ℝ)
axiom sin_theta : sin θ = 3 / 5
axiom theta_in_second_quadrant : π / 2 < θ ∧ θ < π

-- Questions and answers to prove:
theorem tan_theta : tan θ = - 3 / 4 :=
sorry

theorem cos_double_angle_minus_pi_over_3 : cos (2 * θ - π / 3) = (7 - 24 * Real.sqrt 3) / 50 :=
sorry

end tan_theta_cos_double_angle_minus_pi_over_3_l217_217693


namespace Carlton_button_up_shirts_l217_217700

/-- 
Given that the number of sweater vests V is twice the number of button-up shirts S, 
and the total number of unique outfits (each combination of a sweater vest and a button-up shirt) is 18, 
prove that the number of button-up shirts S is 3. 
-/
theorem Carlton_button_up_shirts (V S : ℕ) (h1 : V = 2 * S) (h2 : V * S = 18) : S = 3 := by
  sorry

end Carlton_button_up_shirts_l217_217700


namespace center_of_image_circle_l217_217215

def point := ℝ × ℝ

def reflect_about_y_eq_neg_x (p : point) : point :=
  let (a, b) := p
  (-b, -a)

theorem center_of_image_circle :
  reflect_about_y_eq_neg_x (8, -3) = (3, -8) :=
by
  sorry

end center_of_image_circle_l217_217215


namespace algebraic_expression_value_l217_217539

theorem algebraic_expression_value (x : ℝ) :
  let a := 2003 * x + 2001
  let b := 2003 * x + 2002
  let c := 2003 * x + 2003
  a^2 + b^2 + c^2 - a * b - a * c - b * c = 3 :=
by
  sorry

end algebraic_expression_value_l217_217539


namespace car_arrives_first_and_earlier_l217_217142

-- Define the conditions
def total_intersections : ℕ := 11
def total_blocks : ℕ := 12
def green_time : ℕ := 3
def red_time : ℕ := 1
def car_block_time : ℕ := 1
def bus_block_time : ℕ := 2

-- Define the functions that compute the travel times
def car_travel_time (blocks : ℕ) : ℕ :=
  (blocks / 3) * (green_time + red_time) + (blocks % 3 * car_block_time)

def bus_travel_time (blocks : ℕ) : ℕ :=
  blocks * bus_block_time

-- Define the theorem to prove
theorem car_arrives_first_and_earlier :
  car_travel_time total_blocks < bus_travel_time total_blocks ∧
  bus_travel_time total_blocks - car_travel_time total_blocks = 9 := 
by
  sorry

end car_arrives_first_and_earlier_l217_217142


namespace ratio_between_house_and_park_l217_217296

theorem ratio_between_house_and_park (w x y : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0)
    (h : y / w = x / w + (x + y) / (10 * w)) : x / y = 9 / 11 :=
by 
  sorry

end ratio_between_house_and_park_l217_217296


namespace prove_moles_of_C2H6_l217_217202

def moles_of_CCl4 := 4
def moles_of_Cl2 := 14
def moles_of_C2H6 := 2

theorem prove_moles_of_C2H6
  (h1 : moles_of_Cl2 = 14)
  (h2 : moles_of_CCl4 = 4)
  : moles_of_C2H6 = 2 := 
sorry

end prove_moles_of_C2H6_l217_217202


namespace midpoint_integer_of_five_points_l217_217045

theorem midpoint_integer_of_five_points 
  (P : Fin 5 → ℤ × ℤ) 
  (distinct : Function.Injective P) :
  ∃ i j : Fin 5, i ≠ j ∧ (P i).1 + (P j).1 % 2 = 0 ∧ (P i).2 + (P j).2 % 2 = 0 :=
by
  sorry

end midpoint_integer_of_five_points_l217_217045


namespace count_valid_outfits_l217_217467

/-
Problem:
I have 5 shirts, 3 pairs of pants, and 5 hats. The pants come in red, green, and blue. 
The shirts and hats come in those colors, plus orange and purple. 
I refuse to wear an outfit where the shirt and the hat are the same color. 
How many choices for outfits, consisting of one shirt, one hat, and one pair of pants, do I have?
-/

def num_shirts := 5
def num_pants := 3
def num_hats := 5
def valid_outfits := 66

-- The set of colors available for shirts and hats
inductive color
| red | green | blue | orange | purple

-- Conditions and properties translated into Lean
def pants_colors : List color := [color.red, color.green, color.blue]
def shirt_hat_colors : List color := [color.red, color.green, color.blue, color.orange, color.purple]

theorem count_valid_outfits (h1 : num_shirts = 5) 
                            (h2 : num_pants = 3) 
                            (h3 : num_hats = 5) 
                            (h4 : ∀ (s : color), s ∈ shirt_hat_colors) 
                            (h5 : ∀ (p : color), p ∈ pants_colors) 
                            (h6 : ∀ (s h : color), s ≠ h) :
  valid_outfits = 66 :=
by
  sorry

end count_valid_outfits_l217_217467


namespace probability_of_defective_l217_217181

theorem probability_of_defective (p_first_grade p_second_grade : ℝ) (h_fg : p_first_grade = 0.65) (h_sg : p_second_grade = 0.3) : (1 - (p_first_grade + p_second_grade) = 0.05) :=
by
  sorry

end probability_of_defective_l217_217181


namespace smallest_n_l217_217604

theorem smallest_n (j c g : ℕ) (n : ℕ) (total_cost : ℕ) 
  (h_condition : total_cost = 10 * j ∧ total_cost = 16 * c ∧ total_cost = 18 * g ∧ total_cost = 24 * n) 
  (h_lcm : Nat.lcm (Nat.lcm 10 16) 18 = 720) : n = 30 :=
by
  sorry

end smallest_n_l217_217604


namespace ellipse_equation_l217_217813

theorem ellipse_equation (a b c : ℝ) 
  (h1 : 0 < b) (h2 : b < a) 
  (h3 : c = 3 * Real.sqrt 3) 
  (h4 : a = 6) 
  (h5 : b^2 = a^2 - c^2) :
  (∀ x y : ℝ, x^2 / 36 + y^2 / 9 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) :=
by
  sorry

end ellipse_equation_l217_217813


namespace combined_age_of_siblings_l217_217366

theorem combined_age_of_siblings (a s h : ℕ) (h1 : a = 15) (h2 : s = 3 * a) (h3 : h = 4 * s) : a + s + h = 240 :=
by
  sorry

end combined_age_of_siblings_l217_217366


namespace mark_parking_tickets_eq_l217_217500

def total_tickets : ℕ := 24
def sarah_speeding_tickets : ℕ := 6
def mark_speeding_tickets : ℕ := 6
def sarah_parking_tickets (S : ℕ) := S
def mark_parking_tickets (S : ℕ) := 2 * S
def total_traffic_tickets (S : ℕ) := S + 2 * S + sarah_speeding_tickets + mark_speeding_tickets

theorem mark_parking_tickets_eq (S : ℕ) (h1 : total_traffic_tickets S = total_tickets)
  (h2 : sarah_speeding_tickets = 6) (h3 : mark_speeding_tickets = 6) :
  mark_parking_tickets S = 8 :=
sorry

end mark_parking_tickets_eq_l217_217500


namespace largest_three_digit_divisible_and_prime_sum_l217_217256

theorem largest_three_digit_divisible_and_prime_sum :
  ∃ n : ℕ, 900 ≤ n ∧ n < 1000 ∧
           (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d ≠ 0 ∧ n % d = 0) ∧
           Prime (n / 100 + (n / 10) % 10 + n % 10) ∧
           n = 963 ∧
           ∀ m : ℕ, 900 ≤ m ∧ m < 1000 ∧
           (∀ d ∈ [m / 100, (m / 10) % 10, m % 10], d ≠ 0 ∧ m % d = 0) ∧
           Prime (m / 100 + (m / 10) % 10 + m % 10) →
           m ≤ 963 :=
by
  sorry

end largest_three_digit_divisible_and_prime_sum_l217_217256


namespace weight_of_7th_person_l217_217755

-- Defining the constants and conditions
def num_people_initial : ℕ := 6
def avg_weight_initial : ℝ := 152
def num_people_total : ℕ := 7
def avg_weight_total : ℝ := 151

-- Calculating the total weights from the given average weights
def total_weight_initial := num_people_initial * avg_weight_initial
def total_weight_total := num_people_total * avg_weight_total

-- Theorem stating the weight of the 7th person
theorem weight_of_7th_person : total_weight_total - total_weight_initial = 145 := 
sorry

end weight_of_7th_person_l217_217755


namespace person_savings_l217_217291

theorem person_savings (income expenditure savings : ℝ) 
  (h1 : income = 18000)
  (h2 : income / expenditure = 5 / 4)
  (h3 : savings = income - expenditure) : 
  savings = 3600 := 
sorry

end person_savings_l217_217291


namespace rectangular_field_perimeter_l217_217044

theorem rectangular_field_perimeter (A L : ℝ) (h1 : A = 300) (h2 : L = 15) : 
  let W := A / L 
  let P := 2 * (L + W)
  P = 70 := by
  sorry

end rectangular_field_perimeter_l217_217044


namespace number_of_three_digit_multiples_of_7_l217_217869

theorem number_of_three_digit_multiples_of_7 : 
  let smallest_multiple := 7 * Nat.ceil (100 / 7)
  let largest_multiple := 7 * Nat.floor (999 / 7)
  (largest_multiple - smallest_multiple) / 7 + 1 = 128 :=
by
  sorry

end number_of_three_digit_multiples_of_7_l217_217869


namespace no_four_digit_with_five_units_divisible_by_ten_l217_217978

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def units_place_is_five (n : ℕ) : Prop :=
  n % 10 = 5

def divisible_by_ten (n : ℕ) : Prop :=
  n % 10 = 0

theorem no_four_digit_with_five_units_divisible_by_ten : ∀ n : ℕ, 
  is_four_digit n → units_place_is_five n → ¬ divisible_by_ten n :=
by
  intro n h1 h2
  rw [units_place_is_five] at h2
  rw [divisible_by_ten, h2]
  sorry

end no_four_digit_with_five_units_divisible_by_ten_l217_217978


namespace original_volume_of_cube_l217_217892

theorem original_volume_of_cube (a : ℕ) 
  (h1 : (a + 2) * (a - 2) * (a + 3) = a^3 - 7) : 
  a = 3 :=
by sorry

end original_volume_of_cube_l217_217892


namespace determine_t_l217_217839

theorem determine_t (t : ℝ) : 
  (3 * t - 9) * (4 * t - 3) = (4 * t - 16) * (3 * t - 9) → t = 7.8 :=
by
  intros h
  sorry

end determine_t_l217_217839


namespace solve_system_nat_l217_217168

theorem solve_system_nat (a b c d : ℕ) :
  (a * b = c + d ∧ c * d = a + b) →
  (a = 1 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
  (a = 1 ∧ b = 5 ∧ c = 3 ∧ d = 2) ∨
  (a = 5 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
  (a = 5 ∧ b = 1 ∧ c = 3 ∧ d = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2) ∨
  (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5) ∨
  (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 1) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∨
  (a = 3 ∧ b = 2 ∧ c = 5 ∧ d = 1) :=
sorry

end solve_system_nat_l217_217168


namespace find_c_minus_a_l217_217030

variable (a b c d e : ℝ)

-- Conditions
axiom avg_ab : (a + b) / 2 = 40
axiom avg_bc : (b + c) / 2 = 60
axiom avg_de : (d + e) / 2 = 80
axiom geom_mean : (a * b * d) = (b * c * e)

theorem find_c_minus_a : c - a = 40 := by
  sorry

end find_c_minus_a_l217_217030


namespace min_abs_value_sum_l217_217714

theorem min_abs_value_sum (x : ℚ) : (min (|x - 1| + |x + 3|) = 4) :=
sorry

end min_abs_value_sum_l217_217714


namespace initial_storks_count_l217_217608

-- Definitions based on the conditions provided
def initialBirds : ℕ := 3
def additionalStorks : ℕ := 6
def totalBirdsAndStorks : ℕ := 13

-- The mathematical statement to be proved
theorem initial_storks_count (S : ℕ) (h : initialBirds + S + additionalStorks = totalBirdsAndStorks) : S = 4 :=
by
  sorry

end initial_storks_count_l217_217608


namespace cos_135_eq_neg_sqrt_2_div_2_point_Q_coordinates_l217_217311

noncomputable def cos_135_deg : Real := - (Real.sqrt 2) / 2

theorem cos_135_eq_neg_sqrt_2_div_2 : Real.cos (135 * Real.pi / 180) = cos_135_deg := sorry

noncomputable def point_Q : Real × Real :=
  (- (Real.sqrt 2) / 2, (Real.sqrt 2) / 2)

theorem point_Q_coordinates :
  ∃ (Q : Real × Real), Q = point_Q ∧ Q = (Real.cos (135 * Real.pi / 180), Real.sin (135 * Real.pi / 180)) := sorry

end cos_135_eq_neg_sqrt_2_div_2_point_Q_coordinates_l217_217311


namespace five_x_ge_seven_y_iff_exists_abcd_l217_217469

theorem five_x_ge_seven_y_iff_exists_abcd (x y : ℕ) :
  (5 * x ≥ 7 * y) ↔ ∃ (a b c d : ℕ), x = a + 2 * b + 3 * c + 7 * d ∧ y = b + 2 * c + 5 * d :=
by sorry

end five_x_ge_seven_y_iff_exists_abcd_l217_217469


namespace circle_B_area_l217_217010

theorem circle_B_area
  (r R : ℝ)
  (h1 : ∀ (x : ℝ), x = 5)  -- derived from r = 5
  (h2 : R = 2 * r)
  (h3 : 25 * Real.pi = Real.pi * r^2)
  (h4 : R = 10)  -- derived from diameter relation
  : ∃ A_B : ℝ, A_B = 100 * Real.pi :=
by
  sorry

end circle_B_area_l217_217010


namespace quadrilateral_diagonal_length_l217_217758

theorem quadrilateral_diagonal_length (D A₁ A₂ : ℝ) (hA₁ : A₁ = 9) (hA₂ : A₂ = 6) (Area : ℝ) (hArea : Area = 165) :
  (1/2) * D * (A₁ + A₂) = Area → D = 22 :=
by
  -- Use the given conditions and solve to obtain D = 22
  intros
  sorry

end quadrilateral_diagonal_length_l217_217758


namespace probability_same_color_opposite_foot_l217_217923

def total_shoes := 28

def black_pairs := 7
def brown_pairs := 4
def gray_pairs := 2
def red_pair := 1

def total_pairs := black_pairs + brown_pairs + gray_pairs + red_pair

theorem probability_same_color_opposite_foot : 
  (7 + 4 + 2 + 1) * 2 = total_shoes →
  (14 / 28 * (7 / 27) + 8 / 28 * (4 / 27) + 4 / 28 * (2 / 27) + 2 / 28 * (1 / 27)) = (20 / 63) :=
by
  sorry

end probability_same_color_opposite_foot_l217_217923


namespace minimum_value_of_f_l217_217250

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) * (x^2 - 4 * x + 3)

theorem minimum_value_of_f : ∃ m : ℝ, m = -16 ∧ ∀ x : ℝ, f x ≥ m :=
by
  use -16
  sorry

end minimum_value_of_f_l217_217250


namespace total_asphalt_used_1520_tons_l217_217418

noncomputable def asphalt_used (L W : ℕ) (asphalt_per_100m2 : ℕ) : ℕ :=
  (L * W / 100) * asphalt_per_100m2

theorem total_asphalt_used_1520_tons :
  asphalt_used 800 50 3800 = 1520000 := by
  sorry

end total_asphalt_used_1520_tons_l217_217418


namespace mr_bhaskar_tour_duration_l217_217161

theorem mr_bhaskar_tour_duration :
  ∃ d : Nat, 
    (d > 0) ∧ 
    (∃ original_daily_expense new_daily_expense : ℕ,
      original_daily_expense = 360 / d ∧
      new_daily_expense = original_daily_expense - 3 ∧
      360 = new_daily_expense * (d + 4)) ∧
      d = 20 :=
by
  use 20
  -- Here would come the proof steps to verify the conditions and reach the conclusion.
  sorry

end mr_bhaskar_tour_duration_l217_217161


namespace percent_increase_in_pizza_area_l217_217611

theorem percent_increase_in_pizza_area (r : ℝ) (h : 0 < r) :
  let r_large := 1.10 * r
  let A_medium := π * r^2
  let A_large := π * r_large^2
  let percent_increase := ((A_large - A_medium) / A_medium) * 100 
  percent_increase = 21 := 
by sorry

end percent_increase_in_pizza_area_l217_217611


namespace remaining_books_l217_217614

def initial_books : Nat := 500
def num_people_donating : Nat := 10
def books_per_person : Nat := 8
def borrowed_books : Nat := 220

theorem remaining_books :
  (initial_books + num_people_donating * books_per_person - borrowed_books) = 360 := 
by 
  -- This will contain the mathematical proof
  sorry

end remaining_books_l217_217614


namespace range_of_k_for_quadratic_inequality_l217_217521

theorem range_of_k_for_quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, k * x^2 + 2 * k * x - 1 < 0) ↔ (-1 < k ∧ k ≤ 0) :=
  sorry

end range_of_k_for_quadratic_inequality_l217_217521


namespace margaret_spends_on_croissants_l217_217237

theorem margaret_spends_on_croissants :
  (∀ (people : ℕ) (sandwiches_per_person : ℕ) (croissants_per_sandwich : ℕ) (croissants_per_set : ℕ) (cost_per_set : ℝ),
    people = 24 →
    sandwiches_per_person = 2 →
    croissants_per_sandwich = 1 →
    croissants_per_set = 12 →
    cost_per_set = 8 →
    (people * sandwiches_per_person * croissants_per_sandwich) / croissants_per_set * cost_per_set = 32) := sorry

end margaret_spends_on_croissants_l217_217237


namespace average_age_of_team_l217_217531

theorem average_age_of_team
    (A : ℝ)
    (captain_age : ℝ)
    (wicket_keeper_age : ℝ)
    (bowlers_count : ℝ)
    (batsmen_count : ℝ)
    (team_members_count : ℝ)
    (avg_bowlers_age : ℝ)
    (avg_batsmen_age : ℝ)
    (total_age_team : ℝ) :
    captain_age = 28 →
    wicket_keeper_age = 31 →
    bowlers_count = 5 →
    batsmen_count = 4 →
    avg_bowlers_age = A - 2 →
    avg_batsmen_age = A + 3 →
    total_age_team = 28 + 31 + 5 * (A - 2) + 4 * (A + 3) →
    team_members_count * A = total_age_team →
    team_members_count = 11 →
    A = 30.5 :=
by
  intros
  sorry

end average_age_of_team_l217_217531


namespace move_3m_left_is_neg_3m_l217_217170

-- Define the notation for movements
def move_right (distance : Int) : Int := distance
def move_left (distance : Int) : Int := -distance

-- Define the specific condition
def move_1m_right : Int := move_right 1

-- Define the assertion for moving 3m to the left
def move_3m_left : Int := move_left 3

-- State the proof problem
theorem move_3m_left_is_neg_3m : move_3m_left = -3 := by
  unfold move_3m_left
  unfold move_left
  rfl

end move_3m_left_is_neg_3m_l217_217170


namespace tennis_tournament_cycle_l217_217371

noncomputable def exists_cycle_of_three_players (P : Type) [Fintype P] (G : P → P → Bool) : Prop :=
  (∃ (a b c : P), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ G a b ∧ G b c ∧ G c a)

theorem tennis_tournament_cycle (P : Type) [Fintype P] (n : ℕ) (hp : 3 ≤ n) 
  (G : P → P → Bool) (H : ∀ a b : P, a ≠ b → (G a b ∨ G b a))
  (Hw : ∀ a : P, ∃ b : P, a ≠ b ∧ G a b) : exists_cycle_of_three_players P G :=
by 
  sorry

end tennis_tournament_cycle_l217_217371


namespace problem1_problem2_l217_217107

-- Define variables
variables {x y m : ℝ}
variables (h1 : x + y > 0) (h2 : xy ≠ 0)

-- Problem (1): Prove that x^3 + y^3 ≥ x^2 y + y^2 x
theorem problem1 (h1 : x + y > 0) (h2 : xy ≠ 0) : x^3 + y^3 ≥ x^2 * y + y^2 * x :=
sorry

-- Problem (2): Given the conditions, the range of m is [-6, 2]
theorem problem2 (h1 : x + y > 0) (h2 : xy ≠ 0) (h3 : (x / y^2) + (y / x^2) ≥ (m / 2) * ((1 / x) + (1 / y))) : m ∈ Set.Icc (-6 : ℝ) 2 :=
sorry

end problem1_problem2_l217_217107


namespace alpha_nonneg_integer_l217_217190

theorem alpha_nonneg_integer (α : ℝ) 
  (h : ∀ n : ℕ, ∃ k : ℕ, n = k * α) : α ≥ 0 ∧ ∃ k : ℤ, α = k := 
sorry

end alpha_nonneg_integer_l217_217190


namespace sum_of_xyz_l217_217563

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : 1/x + y + z = 3) 
  (h2 : x + 1/y + z = 3) 
  (h3 : x + y + 1/z = 3) : 
  ∃ m n : ℕ, m = 9 ∧ n = 2 ∧ Nat.gcd m n = 1 ∧ 100 * m + n = 902 := 
sorry

end sum_of_xyz_l217_217563


namespace students_transferred_l217_217370

theorem students_transferred (initial_students : ℝ) (students_left : ℝ) (end_students : ℝ) :
  initial_students = 42.0 →
  students_left = 4.0 →
  end_students = 28.0 →
  initial_students - students_left - end_students = 10.0 :=
by
  intros
  sorry

end students_transferred_l217_217370


namespace money_difference_l217_217957

-- Given conditions
def packs_per_hour_peak : Nat := 6
def packs_per_hour_low : Nat := 4
def price_per_pack : Nat := 60
def hours_per_day : Nat := 15

-- Calculate total sales in peak and low seasons
def total_sales_peak : Nat :=
  packs_per_hour_peak * price_per_pack * hours_per_day

def total_sales_low : Nat :=
  packs_per_hour_low * price_per_pack * hours_per_day

-- The Lean statement proving the correct answer
theorem money_difference :
  total_sales_peak - total_sales_low = 1800 :=
by
  sorry

end money_difference_l217_217957


namespace machine_parts_probabilities_l217_217396

-- Define the yield rates for the two machines
def yield_rate_A : ℝ := 0.8
def yield_rate_B : ℝ := 0.9

-- Define the probabilities of defectiveness for each machine
def defective_probability_A := 1 - yield_rate_A
def defective_probability_B := 1 - yield_rate_B

theorem machine_parts_probabilities :
  (defective_probability_A * defective_probability_B = 0.02) ∧
  (((yield_rate_A * defective_probability_B) + (defective_probability_A * yield_rate_B)) = 0.26) ∧
  (defective_probability_A * defective_probability_B + (1 - (defective_probability_A * defective_probability_B)) = 1) :=
by
  sorry

end machine_parts_probabilities_l217_217396


namespace geometric_progression_term_count_l217_217134

theorem geometric_progression_term_count
  (q : ℝ) (b4 : ℝ) (S : ℝ) (b1 : ℝ)
  (h1 : q = 1 / 3)
  (h2 : b4 = b1 * (q ^ 3))
  (h3 : S = b1 * (1 - q ^ 5) / (1 - q))
  (h4 : b4 = 1 / 54)
  (h5 : S = 121 / 162) :
  5 = 5 := sorry

end geometric_progression_term_count_l217_217134


namespace value_of_a8_l217_217235

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

noncomputable def arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n : ℕ, ∃ d : α, a (n + 1) = a n + d

variable {a : ℕ → ℝ}

axiom seq_is_arithmetic : arithmetic_sequence a

axiom initial_condition :
  a 1 + 3 * a 8 + a 15 = 120

axiom arithmetic_property :
  a 1 + a 15 = 2 * a 8

theorem value_of_a8 : a 8 = 24 :=
by {
  sorry
}

end value_of_a8_l217_217235


namespace solve_system_of_equations_l217_217032

theorem solve_system_of_equations (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x^4 + y^4 - x^2 * y^2 = 13)
  (h2 : x^2 - y^2 + 2 * x * y = 1) :
  x = 1 ∧ y = 2 :=
sorry

end solve_system_of_equations_l217_217032


namespace intersection_equiv_l217_217880

def A : Set ℝ := { x : ℝ | x > 1 }
def B : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }
def C : Set ℝ := { x : ℝ | 1 < x ∧ x < 2 }

theorem intersection_equiv : A ∩ B = C :=
by
  sorry

end intersection_equiv_l217_217880


namespace karen_group_size_l217_217451

theorem karen_group_size (total_students : ℕ) (zack_group_size number_of_groups : ℕ) (karen_group_size : ℕ) (h1 : total_students = 70) (h2 : zack_group_size = 14) (h3 : number_of_groups = total_students / zack_group_size) (h4 : number_of_groups = total_students / karen_group_size) : karen_group_size = 14 :=
by
  sorry

end karen_group_size_l217_217451


namespace unit_digit_product_l217_217019

theorem unit_digit_product (n1 n2 n3 : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) :
  (n1 = 68) ∧ (n2 = 59) ∧ (n3 = 71) ∧ (a = 3) ∧ (b = 6) ∧ (c = 7) →
  (a ^ n1 * b ^ n2 * c ^ n3) % 10 = 8 := by
  sorry

end unit_digit_product_l217_217019


namespace part1_part2_l217_217162

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 2) - abs (x + 1)

theorem part1 (x : ℝ) : f x ≤ 3 ↔ -2/3 ≤ x ∧ x ≤ 6 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, f x ≤ abs (x + 1) + a^2) ↔ a ≤ -2 ∨ 2 ≤ a :=
by
  sorry

end part1_part2_l217_217162


namespace find_treasure_island_l217_217493

-- Define the types for the three islands
inductive Island : Type
| A | B | C

-- Define the possible inhabitants of island A
inductive Inhabitant : Type
| Knight  -- always tells the truth
| Liar    -- always lies
| Normal  -- might tell the truth or lie

-- Define the conditions
def no_treasure_on_A : Prop := ¬ ∃ (x : Island), x = Island.A ∧ (x = Island.A)
def normal_people_on_A_two_treasures : Prop := ∀ (h : Inhabitant), h = Inhabitant.Normal → (∃ (x y : Island), x ≠ y ∧ (x ≠ Island.A ∧ y ≠ Island.A))

-- The question to ask
def question_to_ask (h : Inhabitant) : Prop :=
  (h = Inhabitant.Knight) ↔ (∃ (x : Island), (x = Island.B) ∧ (¬ ∃ (y : Island), (y = Island.A) ∧ (y = Island.A)))

-- The theorem statement
theorem find_treasure_island (inh : Inhabitant) :
  no_treasure_on_A ∧ normal_people_on_A_two_treasures →
  (question_to_ask inh → (∃ (x : Island), x = Island.B)) ∧ (¬ question_to_ask inh → (∃ (x : Island), x = Island.C)) :=
by
  intro h
  sorry

end find_treasure_island_l217_217493


namespace proj_a_b_l217_217419

open Real

def vector (α : Type*) := (α × α)

noncomputable def dot_product (a b: vector ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (v: vector ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

noncomputable def projection (a b: vector ℝ) : ℝ := (dot_product a b) / (magnitude b)

-- Define the vectors a and b
def a : vector ℝ := (-1, 3)
def b : vector ℝ := (3, 4)

-- The projection of a in the direction of b
theorem proj_a_b : projection a b = 9 / 5 := 
  by sorry

end proj_a_b_l217_217419


namespace range_of_m_l217_217118

theorem range_of_m (x y m : ℝ) (h1 : 2 / x + 1 / y = 1) (h2 : x + y = 2 + 2 * m) : -4 < m ∧ m < 2 :=
sorry

end range_of_m_l217_217118


namespace impossibility_of_4_level_ideal_interval_tan_l217_217472

def has_ideal_interval (f : ℝ → ℝ) (D : Set ℝ) (k : ℝ) :=
  ∃ (a b : ℝ), a ≤ b ∧ Set.Icc a b ⊆ D ∧ (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y ∨ f y ≤ f x) ∧
  (Set.image f (Set.Icc a b) = Set.Icc (k * a) (k * b))

def option_D_incorrect : Prop :=
  ¬ has_ideal_interval (fun x => Real.tan x) (Set.Ioc (-(Real.pi / 2)) (Real.pi / 2)) 4

theorem impossibility_of_4_level_ideal_interval_tan :
  option_D_incorrect :=
sorry

end impossibility_of_4_level_ideal_interval_tan_l217_217472
