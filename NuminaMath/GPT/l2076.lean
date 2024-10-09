import Mathlib

namespace soda_ratio_l2076_207628

theorem soda_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) : 
  let v_z := 1.3 * v
  let p_z := 0.85 * p
  (p_z / v_z) / (p / v) = 17 / 26 :=
by sorry

end soda_ratio_l2076_207628


namespace range_of_m_for_decreasing_interval_l2076_207690

def function_monotonically_decreasing_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x → x < y → y < b → f y ≤ f x

def f (x : ℝ) : ℝ := x ^ 3 - 12 * x

theorem range_of_m_for_decreasing_interval :
  ∀ m : ℝ, function_monotonically_decreasing_in_interval f (2 * m) (m + 1) → -1 ≤ m ∧ m < 1 :=
by
  sorry

end range_of_m_for_decreasing_interval_l2076_207690


namespace train_cross_time_l2076_207611

def train_length := 100
def bridge_length := 275
def train_speed_kmph := 45

noncomputable def train_speed_mps : ℝ :=
  (train_speed_kmph * 1000.0) / 3600.0

theorem train_cross_time :
  let total_distance := train_length + bridge_length
  let speed := train_speed_mps
  let time := total_distance / speed
  time = 30 :=
by 
  -- Introduce definitions to make sure they align with the initial conditions
  let total_distance := train_length + bridge_length
  let speed := train_speed_mps
  let time := total_distance / speed
  -- Prove time = 30
  sorry

end train_cross_time_l2076_207611


namespace votes_to_win_l2076_207631

theorem votes_to_win (total_votes : ℕ) (geoff_votes_percent : ℝ) (additional_votes : ℕ) (x : ℝ) 
(h1 : total_votes = 6000)
(h2 : geoff_votes_percent = 0.5)
(h3 : additional_votes = 3000)
(h4 : x = 50.5) :
  ((geoff_votes_percent / 100 * total_votes) + additional_votes) / total_votes * 100 = x :=
by
  sorry

end votes_to_win_l2076_207631


namespace opposite_of_5_is_neg_5_l2076_207635

def opposite_number (x y : ℤ) : Prop := x + y = 0

theorem opposite_of_5_is_neg_5 : opposite_number 5 (-5) := by
  sorry

end opposite_of_5_is_neg_5_l2076_207635


namespace unique_b_positive_solution_l2076_207678

theorem unique_b_positive_solution (c : ℝ) (h : c ≠ 0) : 
  (∃ b : ℝ, b > 0 ∧ ∀ b : ℝ, b ≠ 0 → 
    ∀ x : ℝ, x^2 + (b + 1 / b) * x + c = 0 → x = - (b + 1 / b) / 2) 
  ↔ c = (5 + Real.sqrt 21) / 2 ∨ c = (5 - Real.sqrt 21) / 2 := 
by {
  sorry
}

end unique_b_positive_solution_l2076_207678


namespace find_a_l2076_207638

noncomputable def A (a : ℝ) : Set ℝ :=
  {a + 2, (a + 1)^2, a^2 + 3 * a + 3}

theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 :=
  sorry

end find_a_l2076_207638


namespace part_a_part_b_part_c_l2076_207668

def f (x : ℝ) := x^2
def g (x : ℝ) := 3 * x - 8
def h (r : ℝ) (x : ℝ) := 3 * x - r

theorem part_a :
  f 2 = 4 ∧ g (f 2) = 4 :=
by {
  sorry
}

theorem part_b :
  ∀ x : ℝ, f (g x) = g (f x) → (x = 2 ∨ x = 6) :=
by {
  sorry
}

theorem part_c :
  ∀ r : ℝ, f (h r 2) = h r (f 2) → (r = 3 ∨ r = 8) :=
by {
  sorry
}

end part_a_part_b_part_c_l2076_207668


namespace expression_value_is_one_l2076_207617

theorem expression_value_is_one :
  let a1 := 121
  let b1 := 19
  let a2 := 91
  let b2 := 13
  (a1^2 - b1^2) / (a2^2 - b2^2) * ((a2 - b2) * (a2 + b2)) / ((a1 - b1) * (a1 + b1)) = 1 := by
  sorry

end expression_value_is_one_l2076_207617


namespace gcd_of_17420_23826_36654_l2076_207653

theorem gcd_of_17420_23826_36654 : Nat.gcd (Nat.gcd 17420 23826) 36654 = 2 := 
by 
  sorry

end gcd_of_17420_23826_36654_l2076_207653


namespace students_not_in_either_l2076_207684

theorem students_not_in_either (total_students chemistry_students biology_students both_subjects neither_subjects : ℕ) 
  (h1 : total_students = 120) 
  (h2 : chemistry_students = 75) 
  (h3 : biology_students = 50) 
  (h4 : both_subjects = 15) 
  (h5 : neither_subjects = total_students - (chemistry_students - both_subjects + biology_students - both_subjects + both_subjects)) : 
  neither_subjects = 10 := 
by 
  sorry

end students_not_in_either_l2076_207684


namespace reflection_across_x_axis_l2076_207632

theorem reflection_across_x_axis (x y : ℝ) : (x, -y) = (-2, 3) ↔ (x, y) = (-2, -3) :=
by sorry

end reflection_across_x_axis_l2076_207632


namespace S6_geometric_sum_l2076_207605

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem S6_geometric_sum (a r : ℝ)
    (sum_n : ℕ → ℝ)
    (geo_seq : ∀ n, sum_n n = geometric_sequence_sum a r n)
    (S2 : sum_n 2 = 6)
    (S4 : sum_n 4 = 30) :
    sum_n 6 = 126 := 
by
  sorry

end S6_geometric_sum_l2076_207605


namespace donation_amount_per_person_l2076_207642

theorem donation_amount_per_person (m n : ℕ) 
  (h1 : m + 11 = n + 9) 
  (h2 : ∃ d : ℕ, (m * n + 9 * m + 11 * n + 145) = d * (m + 11)) 
  (h3 : ∃ d : ℕ, (m * n + 9 * m + 11 * n + 145) = d * (n + 9))
  : ∃ k : ℕ, k = 25 ∨ k = 47 :=
by
  sorry

end donation_amount_per_person_l2076_207642


namespace equation_of_ellipse_equation_of_line_AB_l2076_207669

-- Step 1: Given conditions for the ellipse and related hyperbola.
def condition_eccentricity (a b c : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ c / a = Real.sqrt 2 / 2

def condition_distance_focus_asymptote (c : ℝ) : Prop :=
  abs c / Real.sqrt (1 + 2) = Real.sqrt 3 / 3

-- Step 2: Given conditions for the line AB.
def condition_line_A_B (k m : ℝ) : Prop :=
  k < 0 ∧ m^2 = 4 / 5 * (1 + k^2) ∧
  ∃ (x1 x2 y1 y2 : ℝ), 
  (1 + 2 * k^2) * x1^2 + 4 * k * m * x1 + 2 * m^2 - 2 = 0 ∧ 
  (1 + 2 * k^2) * x2^2 + 4 * k * m * x2 + 2 * m^2 - 2 = 0 ∧
  x1 + x2 = -4 * k * m / (1 + 2*k^2) ∧ 
  x1 * x2 = (2 * m^2 - 2) / (1 + 2*k^2)

def condition_circle_passes_F2 (x1 x2 k m : ℝ) : Prop :=
  (1 + k^2) * x1 * x2 + (k * m - 1) * (x1 + x2) + m^2 + 1 = 0

noncomputable def problem_data : Prop :=
  ∃ (a b c k m x1 x2 : ℝ),
    condition_eccentricity a b c ∧
    condition_distance_focus_asymptote c ∧
    condition_line_A_B k m ∧
    condition_circle_passes_F2 x1 x2 k m

-- Step 3: Statements to be proven.
theorem equation_of_ellipse : problem_data → 
  ∃ (a b : ℝ), a = Real.sqrt 2 ∧ b = 1 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 2 + y^2 = 1) :=
by sorry

theorem equation_of_line_AB : problem_data → 
  ∃ (k m : ℝ), m = 1 ∧ k = -1/2 ∧ ∀ x y : ℝ, (y = k * x + m) ↔ (y = -0.5 * x + 1) :=
by sorry

end equation_of_ellipse_equation_of_line_AB_l2076_207669


namespace infinite_nested_radical_l2076_207603

theorem infinite_nested_radical : ∀ (x : ℝ), (x > 0) → (x = Real.sqrt (12 + x)) → x = 4 :=
by
  intro x
  intro hx_pos
  intro hx_eq
  sorry

end infinite_nested_radical_l2076_207603


namespace inequality_of_negatives_l2076_207656

theorem inequality_of_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  sorry

end inequality_of_negatives_l2076_207656


namespace part_a_part_b_part_c_l2076_207658

-- Part (a)
theorem part_a : (7 * (2 / 3) + 16 * (5 / 12)) = (34 / 3) :=
by
  sorry

-- Part (b)
theorem part_b : (5 - (2 / (5 / 3))) = (19 / 5) :=
by
  sorry

-- Part (c)
theorem part_c : (1 + (2 / (1 + (3 / (1 + 4))))) = (9 / 4) :=
by
  sorry

end part_a_part_b_part_c_l2076_207658


namespace gcd_polynomial_l2076_207696

theorem gcd_polynomial {b : ℤ} (h1 : ∃ k : ℤ, b = 2 * 7786 * k) : 
  Int.gcd (8 * b^2 + 85 * b + 200) (2 * b + 10) = 10 :=
by
  sorry

end gcd_polynomial_l2076_207696


namespace empty_square_exists_in_4x4_l2076_207666

theorem empty_square_exists_in_4x4  :
  ∀ (points: Finset (Fin 4 × Fin 4)), points.card = 15 → 
  ∃ (i j : Fin 4), (i, j) ∉ points :=
by
  sorry

end empty_square_exists_in_4x4_l2076_207666


namespace savings_after_increase_l2076_207686

-- Conditions
def salary : ℕ := 5000
def initial_savings_ratio : ℚ := 0.20
def expense_increase_ratio : ℚ := 1.20

-- Derived initial values
def initial_savings : ℚ := initial_savings_ratio * salary
def initial_expenses : ℚ := ((1 : ℚ) - initial_savings_ratio) * salary

-- New expenses after increase
def new_expenses : ℚ := expense_increase_ratio * initial_expenses

-- Savings after expense increase
def final_savings : ℚ := salary - new_expenses

theorem savings_after_increase : final_savings = 200 := by
  sorry

end savings_after_increase_l2076_207686


namespace problem_statement_l2076_207644

theorem problem_statement (a b : ℤ) (h1 : b = 7) (h2: a * b = 2 * (a + b) + 1) :
  b - a = 4 := by
  sorry

end problem_statement_l2076_207644


namespace A_greater_than_B_l2076_207606

theorem A_greater_than_B (A B : ℝ) (h₁ : A * 4 = B * 5) (h₂ : A ≠ 0) (h₃ : B ≠ 0) : A > B :=
by
  sorry

end A_greater_than_B_l2076_207606


namespace calculate_avg_l2076_207600

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem calculate_avg :
  avg3 (avg3 1 2 0) (avg2 0 2) 0 = 2 / 3 :=
by
  sorry

end calculate_avg_l2076_207600


namespace product_variation_l2076_207619

theorem product_variation (a b c : ℕ) (h1 : a * b = c) (h2 : b' = 10 * b) (h3 : ∃ d : ℕ, d = a * b') : d = 720 :=
by
  sorry

end product_variation_l2076_207619


namespace fourth_person_height_l2076_207694

theorem fourth_person_height 
  (height1 height2 height3 height4 : ℝ)
  (diff12 : height2 = height1 + 2)
  (diff23 : height3 = height2 + 2)
  (diff34 : height4 = height3 + 6)
  (avg_height : (height1 + height2 + height3 + height4) / 4 = 76) :
  height4 = 82 :=
by
  sorry

end fourth_person_height_l2076_207694


namespace smallest_five_digit_divisible_by_15_32_54_l2076_207682

theorem smallest_five_digit_divisible_by_15_32_54 : 
  ∃ n : ℤ, n >= 10000 ∧ n < 100000 ∧ (15 ∣ n) ∧ (32 ∣ n) ∧ (54 ∣ n) ∧ n = 17280 :=
  sorry

end smallest_five_digit_divisible_by_15_32_54_l2076_207682


namespace janet_earns_more_as_freelancer_l2076_207650

-- Definitions for the problem conditions
def current_job_weekly_hours : ℕ := 40
def current_job_hourly_rate : ℕ := 30

def freelance_client_a_hours_per_week : ℕ := 15
def freelance_client_a_hourly_rate : ℕ := 45

def freelance_client_b_hours_project1_per_week : ℕ := 5
def freelance_client_b_hours_project2_per_week : ℕ := 10
def freelance_client_b_hourly_rate : ℕ := 40

def freelance_client_c_hours_per_week : ℕ := 20
def freelance_client_c_rate_range : ℕ × ℕ := (35, 42)

def weekly_fica_taxes : ℕ := 25
def monthly_healthcare_premiums : ℕ := 400
def monthly_increased_rent : ℕ := 750
def monthly_business_phone_internet : ℕ := 150
def business_expense_percentage : ℕ := 10

def weeks_in_month : ℕ := 4

-- Define the calculations
def current_job_monthly_earnings := current_job_weekly_hours * current_job_hourly_rate * weeks_in_month

def freelance_client_a_weekly_earnings := freelance_client_a_hours_per_week * freelance_client_a_hourly_rate
def freelance_client_b_weekly_earnings := (freelance_client_b_hours_project1_per_week + freelance_client_b_hours_project2_per_week) * freelance_client_b_hourly_rate
def freelance_client_c_weekly_earnings := freelance_client_c_hours_per_week * ((freelance_client_c_rate_range.1 + freelance_client_c_rate_range.2) / 2)

def total_freelance_weekly_earnings := freelance_client_a_weekly_earnings + freelance_client_b_weekly_earnings + freelance_client_c_weekly_earnings
def total_freelance_monthly_earnings := total_freelance_weekly_earnings * weeks_in_month

def total_additional_expenses := (weekly_fica_taxes * weeks_in_month) + monthly_healthcare_premiums + monthly_increased_rent + monthly_business_phone_internet

def business_expense_deduction := (total_freelance_monthly_earnings * business_expense_percentage) / 100
def adjusted_freelance_earnings_after_deduction := total_freelance_monthly_earnings - business_expense_deduction
def adjusted_freelance_earnings_after_expenses := adjusted_freelance_earnings_after_deduction - total_additional_expenses

def earnings_difference := adjusted_freelance_earnings_after_expenses - current_job_monthly_earnings

-- The theorem to be proved
theorem janet_earns_more_as_freelancer :
  earnings_difference = 1162 :=
sorry

end janet_earns_more_as_freelancer_l2076_207650


namespace circle_equation_exists_shortest_chord_line_l2076_207622

-- Condition 1: Points A and B
def point_A : (ℝ × ℝ) := (1, -2)
def point_B : (ℝ × ℝ) := (-1, 0)

-- Condition 2: Circle passes through A and B and sum of intercepts is 2
def passes_through (x y : ℝ) (D E F : ℝ) : Prop := 
  (x^2 + y^2 + D * x + E * y + F = 0)

def satisfies_intercepts (D E : ℝ) : Prop := (-D - E = 2)

-- Prove
theorem circle_equation_exists : 
  ∃ D E F, passes_through 1 (-2) D E F ∧ passes_through (-1) 0 D E F ∧ satisfies_intercepts D E :=
sorry

-- Given that P(2, 0.5) is inside the circle from above theorem
def point_P : (ℝ × ℝ) := (2, 0.5)

-- Prove the equation of the shortest chord line l
theorem shortest_chord_line :
  ∃ m b, m = -2 ∧ point_P.2 = m * (point_P.1 - 2) + b ∧ (∀ (x y : ℝ), 4 * x + 2 * y - 9 = 0) :=
sorry

end circle_equation_exists_shortest_chord_line_l2076_207622


namespace ariana_average_speed_l2076_207625

theorem ariana_average_speed
  (sadie_speed : ℝ)
  (sadie_time : ℝ)
  (ariana_time : ℝ)
  (sarah_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (sadie_speed_eq : sadie_speed = 3)
  (sadie_time_eq : sadie_time = 2)
  (ariana_time_eq : ariana_time = 0.5)
  (sarah_speed_eq : sarah_speed = 4)
  (total_time_eq : total_time = 4.5)
  (total_distance_eq : total_distance = 17) :
  ∃ ariana_speed : ℝ, ariana_speed = 6 :=
by {
  sorry
}

end ariana_average_speed_l2076_207625


namespace intersection_M_N_l2076_207643

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := 
by
  -- Proof to be provided
  sorry

end intersection_M_N_l2076_207643


namespace interior_surface_area_is_correct_l2076_207697

-- Define the original dimensions of the rectangular sheet
def original_length : ℕ := 40
def original_width : ℕ := 50

-- Define the side length of the square corners
def corner_side : ℕ := 10

-- Define the area of the original sheet
def area_original : ℕ := original_length * original_width

-- Define the area of one square corner
def area_corner : ℕ := corner_side * corner_side

-- Define the total area removed by all four corners
def area_removed : ℕ := 4 * area_corner

-- Define the remaining area after the corners are removed
def area_remaining : ℕ := area_original - area_removed

-- The theorem to be proved
theorem interior_surface_area_is_correct : area_remaining = 1600 := by
  sorry

end interior_surface_area_is_correct_l2076_207697


namespace ratio_fraction_4A3B_5C2A_l2076_207688

def ratio (a b c : ℝ) := a / b = 3 / 2 ∧ b / c = 2 / 6 ∧ a / c = 3 / 6

theorem ratio_fraction_4A3B_5C2A (A B C : ℝ) (h : ratio A B C) : (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 := 
  sorry

end ratio_fraction_4A3B_5C2A_l2076_207688


namespace ratio_of_weights_l2076_207660

def initial_weight : ℝ := 2
def weight_after_brownies (w : ℝ) : ℝ := w * 3
def weight_after_more_jelly_beans (w : ℝ) : ℝ := w + 2
def final_weight : ℝ := 16
def weight_before_adding_gummy_worms : ℝ := weight_after_more_jelly_beans (weight_after_brownies initial_weight)

theorem ratio_of_weights :
  final_weight / weight_before_adding_gummy_worms = 2 := 
by
  sorry

end ratio_of_weights_l2076_207660


namespace bob_first_six_probability_l2076_207657

noncomputable def probability_bob_first_six (p : ℚ) : ℚ :=
  (1 - p) * p / (1 - ( (1 - p) * (1 - p)))

theorem bob_first_six_probability :
  probability_bob_first_six (1/6) = 5/11 :=
by
  sorry

end bob_first_six_probability_l2076_207657


namespace find_real_number_l2076_207689

theorem find_real_number :
    (∃ y : ℝ, y = 3 + (5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + sorry)))))))))) ∧ 
    y = (3 + Real.sqrt 29) / 2 :=
by
  sorry

end find_real_number_l2076_207689


namespace parabola_hyperbola_coincide_directrix_l2076_207677

noncomputable def parabola_directrix (p : ℝ) : ℝ := -p / 2
noncomputable def hyperbola_directrix : ℝ := -3 / 2

theorem parabola_hyperbola_coincide_directrix (p : ℝ) (hp : 0 < p) 
  (h_eq : parabola_directrix p = hyperbola_directrix) : p = 3 :=
by
  have hp_directrix : parabola_directrix p = -p / 2 := rfl
  have h_directrix : hyperbola_directrix = -3 / 2 := rfl
  rw [hp_directrix, h_directrix] at h_eq
  sorry

end parabola_hyperbola_coincide_directrix_l2076_207677


namespace find_value_of_10n_l2076_207613

theorem find_value_of_10n (n : ℝ) (h : 2 * n = 14) : 10 * n = 70 :=
sorry

end find_value_of_10n_l2076_207613


namespace percentage_increase_correct_l2076_207692

-- Define the highest and lowest scores as given conditions.
def highest_score : ℕ := 92
def lowest_score : ℕ := 65

-- State that the percentage increase calculation will result in 41.54%
theorem percentage_increase_correct :
  ((highest_score - lowest_score) * 100) / lowest_score = 4154 / 100 :=
by sorry

end percentage_increase_correct_l2076_207692


namespace polynomial_terms_equal_l2076_207641

theorem polynomial_terms_equal (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (h : p + q = 1) :
  (9 * p^8 * q = 36 * p^7 * q^2) → p = 4 / 5 :=
by
  sorry

end polynomial_terms_equal_l2076_207641


namespace union_of_A_and_B_l2076_207683

-- Define the sets A and B as given in the problem
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 4}

-- State the theorem to prove that A ∪ B = {0, 1, 2, 4}
theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 4} := by
  sorry

end union_of_A_and_B_l2076_207683


namespace student_correct_answers_l2076_207604

theorem student_correct_answers 
  (c w : ℕ) 
  (h1 : c + w = 60) 
  (h2 : 4 * c - w = 130) : 
  c = 38 :=
by
  sorry

end student_correct_answers_l2076_207604


namespace base_b_representation_l2076_207685

theorem base_b_representation (b : ℕ) : (2 * b + 9)^2 = 7 * b^2 + 3 * b + 4 → b = 14 := 
sorry

end base_b_representation_l2076_207685


namespace ferris_wheel_cost_per_child_l2076_207663

namespace AmusementPark

def num_children := 5
def daring_children := 3
def merry_go_round_cost_per_child := 3
def ice_cream_cones_per_child := 2
def ice_cream_cost_per_cone := 8
def total_spent := 110

theorem ferris_wheel_cost_per_child (F : ℝ) :
  (daring_children * F + num_children * merry_go_round_cost_per_child +
   num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone = total_spent) →
  F = 5 :=
by
  -- Here we would proceed with the proof steps, but adding sorry to skip it.
  sorry

end AmusementPark

end ferris_wheel_cost_per_child_l2076_207663


namespace cardinals_count_l2076_207698

theorem cardinals_count (C R B S : ℕ) 
  (hR : R = 4 * C)
  (hB : B = 2 * C)
  (hS : S = 3 * C + 1)
  (h_total : C + R + B + S = 31) :
  C = 3 :=
by
  sorry

end cardinals_count_l2076_207698


namespace perimeter_of_original_rectangle_l2076_207648

-- Define the rectangle's dimensions based on the given condition
def length_of_rectangle := 2 * 8 -- because it forms two squares of side 8 cm each
def width_of_rectangle := 8 -- side of the squares

-- Using the formula for the perimeter of a rectangle: P = 2 * (length + width)
def perimeter_of_rectangle := 2 * (length_of_rectangle + width_of_rectangle)

-- The statement we need to prove
theorem perimeter_of_original_rectangle : perimeter_of_rectangle = 48 := by
  sorry

end perimeter_of_original_rectangle_l2076_207648


namespace arithmetic_seq_problem_l2076_207699

open Nat

def arithmetic_sequence (a : ℕ → ℚ) (a1 d : ℚ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

theorem arithmetic_seq_problem :
  ∃ (a : ℕ → ℚ) (a1 d : ℚ),
    (arithmetic_sequence a a1 d) ∧
    (a 2 + a 3 + a 4 = 3) ∧
    (a 7 = 8) ∧
    (a 11 = 15) :=
  sorry

end arithmetic_seq_problem_l2076_207699


namespace chess_tournament_num_players_l2076_207691

theorem chess_tournament_num_players (n : ℕ) :
  (∀ k, k ≠ n → exists m, m ≠ n ∧ (k = m)) ∧ 
  ((1 / 2 * (n - 1)) + (1 / 4 * (n - 1))) = (1 / 13 * ((1 / 2 * n * (n - 1)) - ((1 / 2 * (n - 1)) + (1 / 4 * (n - 1))))) →
  n = 21 :=
by
  sorry

end chess_tournament_num_players_l2076_207691


namespace Panthers_total_games_l2076_207609

/-
Given:
1) The Panthers had won 60% of their basketball games before the district play.
2) During district play, they won four more games and lost four.
3) They finished the season having won half of their total games.
Prove that the total number of games they played in all is 48.
-/

theorem Panthers_total_games
  (y : ℕ) -- total games before district play
  (x : ℕ) -- games won before district play
  (h1 : x = 60 * y / 100) -- they won 60% of the games before district play
  (h2 : (x + 4) = 50 * (y + 8) / 100) -- they won half of the total games including district play
  : (y + 8) = 48 := -- total games they played in all
sorry

end Panthers_total_games_l2076_207609


namespace ratio_of_numbers_l2076_207629

theorem ratio_of_numbers (a b : ℝ) (h1 : 0 < b) (h2 : 0 < a) (h3 : b < a) (h4 : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
sorry

end ratio_of_numbers_l2076_207629


namespace sum_eq_zero_l2076_207607

variable {R : Type} [Field R]

-- Define the conditions
def cond1 (a b c : R) : Prop := (a + b) / c = (b + c) / a
def cond2 (a b c : R) : Prop := (b + c) / a = (a + c) / b
def neq (b c : R) : Prop := b ≠ c

-- State the theorem
theorem sum_eq_zero (a b c : R) (h1 : cond1 a b c) (h2 : cond2 a b c) (h3 : neq b c) : a + b + c = 0 := 
by sorry

end sum_eq_zero_l2076_207607


namespace relationship_of_ys_l2076_207654

variables {k y1 y2 y3 : ℝ}

theorem relationship_of_ys (h : k < 0) 
  (h1 : y1 = k / -4) 
  (h2 : y2 = k / 2) 
  (h3 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 :=
by 
  sorry

end relationship_of_ys_l2076_207654


namespace factor_expression_l2076_207676

theorem factor_expression (x a b c : ℝ) :
  (x - a) ^ 2 * (b - c) + (x - b) ^ 2 * (c - a) + (x - c) ^ 2 * (a - b) = -(a - b) * (b - c) * (c - a) :=
by
  sorry

end factor_expression_l2076_207676


namespace speed_of_man_in_still_water_l2076_207626

theorem speed_of_man_in_still_water 
  (v_m v_s : ℝ)
  (h1 : 32 = 4 * (v_m + v_s))
  (h2 : 24 = 4 * (v_m - v_s)) :
  v_m = 7 :=
by
  sorry

end speed_of_man_in_still_water_l2076_207626


namespace find_multiplying_number_l2076_207665

variable (a b : ℤ)

theorem find_multiplying_number (h : a^2 * b = 3 * (4 * a + 2)) (ha : a = 1) :
  b = 18 := by
  sorry

end find_multiplying_number_l2076_207665


namespace workout_goal_l2076_207673

def monday_situps : ℕ := 12
def tuesday_situps : ℕ := 19
def wednesday_situps_needed : ℕ := 59

theorem workout_goal : monday_situps + tuesday_situps + wednesday_situps_needed = 90 := by
  sorry

end workout_goal_l2076_207673


namespace fourth_vertex_of_tetrahedron_exists_l2076_207687

theorem fourth_vertex_of_tetrahedron_exists (x y z : ℤ) :
  (∃ (x y z : ℤ), 
     ((x - 1) ^ 2 + y ^ 2 + (z - 3) ^ 2 = 26) ∧ 
     ((x - 5) ^ 2 + (y - 3) ^ 2 + (z - 2) ^ 2 = 26) ∧ 
     ((x - 4) ^ 2 + y ^ 2 + (z - 6) ^ 2 = 26)) :=
sorry

end fourth_vertex_of_tetrahedron_exists_l2076_207687


namespace distance_to_second_museum_l2076_207680

theorem distance_to_second_museum (d x : ℕ) (h1 : d = 5) (h2 : 2 * d + 2 * x = 40) : x = 15 :=
by
  sorry

end distance_to_second_museum_l2076_207680


namespace complex_number_on_line_l2076_207624

theorem complex_number_on_line (a : ℝ) (h : (3 : ℝ) = (a - 1) + 2) : a = 2 :=
by
  sorry

end complex_number_on_line_l2076_207624


namespace ravi_money_l2076_207681

theorem ravi_money (n q d : ℕ) (h1 : q = n + 2) (h2 : d = q + 4) (h3 : n = 6) :
  (n * 5 + q * 25 + d * 10) = 350 := by
  sorry

end ravi_money_l2076_207681


namespace value_of_expression_l2076_207634

theorem value_of_expression (n : ℝ) (h : n + 1/n = 6) : n^2 + 1/n^2 + 9 = 43 :=
by
  sorry

end value_of_expression_l2076_207634


namespace find_c_l2076_207679

theorem find_c (c : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Iio (-2) ∪ Set.Ioi 3 → x^2 - c * x + 6 > 0) → c = 1 :=
by
  sorry

end find_c_l2076_207679


namespace total_bowling_balls_is_66_l2076_207649

-- Define the given conditions
def red_bowling_balls := 30
def difference_green_red := 6
def green_bowling_balls := red_bowling_balls + difference_green_red

-- The statement to prove
theorem total_bowling_balls_is_66 :
  red_bowling_balls + green_bowling_balls = 66 := by
  sorry

end total_bowling_balls_is_66_l2076_207649


namespace pair_C_product_not_36_l2076_207620

-- Definitions of the pairs
def pair_A : ℤ × ℤ := (-4, -9)
def pair_B : ℤ × ℤ := (-3, -12)
def pair_C : ℚ × ℚ := (1/2, -72)
def pair_D : ℤ × ℤ := (1, 36)
def pair_E : ℚ × ℚ := (3/2, 24)

-- Mathematical statement for the proof problem
theorem pair_C_product_not_36 :
  pair_C.fst * pair_C.snd ≠ 36 :=
by
  sorry

end pair_C_product_not_36_l2076_207620


namespace find_positive_A_l2076_207695

theorem find_positive_A (A : ℕ) : (A^2 + 7^2 = 130) → A = 9 :=
by
  intro h
  sorry

end find_positive_A_l2076_207695


namespace range_f_neg2_l2076_207659

noncomputable def f (a b x : ℝ): ℝ := a * x^2 + b * x

theorem range_f_neg2 (a b : ℝ) (h1 : 1 ≤ f a b (-1)) (h2 : f a b (-1) ≤ 2)
  (h3 : 3 ≤ f a b 1) (h4 : f a b 1 ≤ 4) : 6 ≤ f a b (-2) ∧ f a b (-2) ≤ 10 :=
by
  sorry

end range_f_neg2_l2076_207659


namespace maximize_lower_houses_l2076_207614

theorem maximize_lower_houses (x y : ℕ) 
    (h1 : x + 2 * y = 30)
    (h2 : 0 < y)
    (h3 : (∃ k, k = 112)) :
  ∃ x y, (x + 2 * y = 30) ∧ ((x * y)) = 112 :=
by
  sorry

end maximize_lower_houses_l2076_207614


namespace scientific_notation_of_425000_l2076_207662

def scientific_notation (x : ℝ) : ℝ × ℤ := sorry

theorem scientific_notation_of_425000 :
  scientific_notation 425000 = (4.25, 5) := sorry

end scientific_notation_of_425000_l2076_207662


namespace remainder_13_pow_51_mod_5_l2076_207610

theorem remainder_13_pow_51_mod_5 : 13^51 % 5 = 2 := by
  sorry

end remainder_13_pow_51_mod_5_l2076_207610


namespace trig_identity_solution_l2076_207674

open Real

theorem trig_identity_solution :
  sin (15 * (π / 180)) * cos (45 * (π / 180)) + sin (105 * (π / 180)) * sin (135 * (π / 180)) = sqrt 3 / 2 :=
by
  -- Placeholder for the proof
  sorry

end trig_identity_solution_l2076_207674


namespace factor_between_l2076_207672

theorem factor_between (n a b : ℕ) (h1 : 10 < n) 
(h2 : n = a * a + b) 
(h3 : a ∣ n) 
(h4 : b ∣ n) 
(h5 : a ≠ b) 
(h6 : 1 < a) 
(h7 : 1 < b) : 
    ∃ m : ℕ, b = m * a ∧ 1 < m ∧ a < a + m ∧ a + m < b  :=
by
  -- proof to be filled in
  sorry

end factor_between_l2076_207672


namespace red_marbles_in_bag_l2076_207618

theorem red_marbles_in_bag (T R : ℕ) (hT : T = 84)
    (probability_not_red : ((T - R : ℚ) / T)^2 = 36 / 49) : 
    R = 12 := 
sorry

end red_marbles_in_bag_l2076_207618


namespace trevor_quarters_counted_l2076_207661

-- Define the conditions from the problem
variable (Q D : ℕ) 
variable (total_coins : ℕ := 77)
variable (excess : ℕ := 48)

-- Use the conditions to assert the existence of quarters and dimes such that the totals align with the given constraints
theorem trevor_quarters_counted : (Q + D = total_coins) ∧ (D = Q + excess) → Q = 29 :=
by
  -- Add sorry to skip the actual proof, as we are only writing the statement
  sorry

end trevor_quarters_counted_l2076_207661


namespace candy_game_solution_l2076_207608

open Nat

theorem candy_game_solution 
  (total_candies : ℕ) 
  (nick_candies : ℕ) 
  (tim_candies : ℕ)
  (tim_wins : ℕ)
  (m n : ℕ)
  (htotal : total_candies = 55) 
  (hnick : nick_candies = 30) 
  (htim : tim_candies = 25)
  (htim_wins : tim_wins = 2)
  (hrounds_total : total_candies = nick_candies + tim_candies)
  (hwinner_condition1 : m > n) 
  (hwinner_condition2 : n > 0) 
  (hwinner_candies_total : total_candies = tim_wins * m + (total_candies / (m + n) - tim_wins) * n)
: m = 8 := 
sorry

end candy_game_solution_l2076_207608


namespace vectors_opposite_directions_l2076_207675

variable {V : Type*} [AddCommGroup V]

theorem vectors_opposite_directions (a b : V) (h : a + 4 • b = 0) (ha : a ≠ 0) (hb : b ≠ 0) : a = -4 • b :=
by sorry

end vectors_opposite_directions_l2076_207675


namespace johns_minutes_billed_l2076_207670

theorem johns_minutes_billed 
  (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) 
  (h1 : monthly_fee = 5) (h2 : cost_per_minute = 0.25) (h3 : total_bill = 12.02) :
  ⌊(total_bill - monthly_fee) / cost_per_minute⌋ = 28 :=
by
  sorry

end johns_minutes_billed_l2076_207670


namespace total_marks_prove_total_marks_l2076_207627

def average_marks : ℝ := 40
def number_of_candidates : ℕ := 50

theorem total_marks (average_marks : ℝ) (number_of_candidates : ℕ) : Real :=
  average_marks * number_of_candidates

theorem prove_total_marks : total_marks average_marks number_of_candidates = 2000 := 
by
  sorry

end total_marks_prove_total_marks_l2076_207627


namespace day_of_week_100_days_from_wednesday_l2076_207667

theorem day_of_week_100_days_from_wednesday (today_is_wed : ∃ i : ℕ, i % 7 = 3) : 
  (100 % 7 + 3) % 7 = 5 := 
by
  sorry

end day_of_week_100_days_from_wednesday_l2076_207667


namespace brendan_threw_back_l2076_207652

-- Brendan's catches in the morning, throwing back x fish and catching more in the afternoon
def brendan_morning (x : ℕ) : ℕ := 8 - x
def brendan_afternoon : ℕ := 5

-- Brendan's and his dad's total catches
def brendan_total (x : ℕ) : ℕ := brendan_morning x + brendan_afternoon
def dad_total : ℕ := 13

-- Combined total fish caught by both
def total_fish (x : ℕ) : ℕ := brendan_total x + dad_total

-- The number of fish thrown back by Brendan
theorem brendan_threw_back : ∃ x : ℕ, total_fish x = 23 ∧ x = 3 :=
by
  sorry

end brendan_threw_back_l2076_207652


namespace part1_solution_set_part2_solution_l2076_207647

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)

theorem part1_solution_set :
  {x : ℝ | f x > 2} = {x | x > 1} ∪ {x | x < -5} :=
by
  sorry

theorem part2_solution (t : ℝ) :
  (∀ x, f x ≥ t^2 - (11 / 2) * t) ↔ (1 / 2 ≤ t ∧ t ≤ 5) :=
by
  sorry

end part1_solution_set_part2_solution_l2076_207647


namespace find_tax_percentage_l2076_207655

noncomputable def net_income : ℝ := 12000
noncomputable def total_income : ℝ := 13000
noncomputable def non_taxable_income : ℝ := 3000
noncomputable def taxable_income : ℝ := total_income - non_taxable_income
noncomputable def tax_percentage (T : ℝ) := total_income - (T * taxable_income)

theorem find_tax_percentage : ∃ T : ℝ, tax_percentage T = net_income :=
by
  sorry

end find_tax_percentage_l2076_207655


namespace quadratic_discriminant_l2076_207664

theorem quadratic_discriminant (k : ℝ) :
  (∃ x : ℝ, k*x^2 + 2*x - 1 = 0) ∧ (∀ a b, (a*x + b) ^ 2 = a^2 * x^2 + 2 * a * b * x + b^2) ∧
  (a = k) ∧ (b = 2) ∧ (c = -1) ∧ ((b^2 - 4 * a * c = 0) → (4 + 4 * k = 0)) → k = -1 :=
sorry

end quadratic_discriminant_l2076_207664


namespace square_divided_into_40_smaller_squares_l2076_207645

theorem square_divided_into_40_smaller_squares : ∃ squares : ℕ, squares = 40 :=
by
  sorry

end square_divided_into_40_smaller_squares_l2076_207645


namespace algebraic_expression_evaluation_l2076_207602

theorem algebraic_expression_evaluation (a b : ℤ) (h : a - 3 * b = -3) : 5 - a + 3 * b = 8 :=
by 
  sorry

end algebraic_expression_evaluation_l2076_207602


namespace ratio_kittens_to_breeding_rabbits_l2076_207615

def breeding_rabbits : ℕ := 10
def kittens_first_spring (k : ℕ) : ℕ := k * breeding_rabbits
def adopted_kittens_first_spring (k : ℕ) : ℕ := 5 * k
def returned_kittens : ℕ := 5
def remaining_kittens_first_spring (k : ℕ) : ℕ := (k * breeding_rabbits) / 2 + returned_kittens

def kittens_second_spring : ℕ := 60
def adopted_kittens_second_spring : ℕ := 4
def remaining_kittens_second_spring : ℕ := kittens_second_spring - adopted_kittens_second_spring

def total_rabbits (k : ℕ) : ℕ := 
  breeding_rabbits + remaining_kittens_first_spring k + remaining_kittens_second_spring

theorem ratio_kittens_to_breeding_rabbits (k : ℕ) (h : total_rabbits k = 121) :
  k = 10 :=
sorry

end ratio_kittens_to_breeding_rabbits_l2076_207615


namespace part1_part2_l2076_207636

def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y
def B (x y : ℝ) : ℝ := x^2 - x * y + x

def difference (x y : ℝ) : ℝ := A x y - 2 * B x y

theorem part1 : difference (-2) 3 = -20 :=
by
  -- Proving that difference (-2) 3 = -20
  sorry

theorem part2 (y : ℝ) : (∀ (x : ℝ), difference x y = 2 * y) → y = 2 / 5 :=
by
  -- Proving that if difference x y is independent of x, then y = 2 / 5
  sorry

end part1_part2_l2076_207636


namespace find_B_from_period_l2076_207671

theorem find_B_from_period (A B C D : ℝ) (h : B ≠ 0) (period_condition : 2 * |2 * π / B| = 4 * π) : B = 1 := sorry

end find_B_from_period_l2076_207671


namespace square_area_l2076_207693

theorem square_area (x : ℝ) (h1 : BG = GH) (h2 : GH = HD) (h3 : BG = 20 * Real.sqrt 2) : x = 40 * Real.sqrt 2 → x^2 = 3200 :=
by
  sorry

end square_area_l2076_207693


namespace product_of_three_numbers_l2076_207616

-- Define the problem conditions as variables and assumptions
variables (a b c : ℚ)
axiom h1 : a + b + c = 30
axiom h2 : a = 3 * (b + c)
axiom h3 : b = 6 * c

-- State the theorem to be proven
theorem product_of_three_numbers : a * b * c = 10125 / 14 :=
by
  sorry

end product_of_three_numbers_l2076_207616


namespace union_of_M_and_Q_is_correct_l2076_207640

-- Given sets M and Q
def M : Set ℕ := {0, 2, 4, 6}
def Q : Set ℕ := {0, 1, 3, 5}

-- Statement to prove
theorem union_of_M_and_Q_is_correct : M ∪ Q = {0, 1, 2, 3, 4, 5, 6} :=
by
  sorry

end union_of_M_and_Q_is_correct_l2076_207640


namespace socks_ratio_l2076_207630

/-- Alice ordered 6 pairs of green socks and some additional pairs of red socks. The price per pair
of green socks was three times that of the red socks. During the delivery, the quantities of the 
pairs were accidentally swapped. This mistake increased the bill by 40%. Prove that the ratio of the 
number of pairs of green socks to red socks in Alice's original order is 1:2. -/
theorem socks_ratio (r y : ℕ) (h1 : y * r ≠ 0) (h2 : 6 * 3 * y + r * y = (r * 3 * y + 6 * y) * 10 / 7) :
  6 / r = 1 / 2 :=
by
  sorry

end socks_ratio_l2076_207630


namespace Tommy_Ratio_Nickels_to_Dimes_l2076_207633

def TommyCoinsProblem :=
  ∃ (P D N Q : ℕ), 
    (D = P + 10) ∧ 
    (Q = 4) ∧ 
    (P = 10 * Q) ∧ 
    (N = 100) ∧ 
    (N / D = 2)

theorem Tommy_Ratio_Nickels_to_Dimes : TommyCoinsProblem := by
  sorry

end Tommy_Ratio_Nickels_to_Dimes_l2076_207633


namespace value_of_x_l2076_207646

theorem value_of_x : (2015^2 + 2015 - 1) / (2015 : ℝ) = 2016 - 1 / 2015 := 
  sorry

end value_of_x_l2076_207646


namespace pipe_B_fill_time_l2076_207651

theorem pipe_B_fill_time (t : ℝ) :
  (1/10) + (2/t) - (2/15) = 1 ↔ t = 60/31 :=
by
  sorry

end pipe_B_fill_time_l2076_207651


namespace determine_f_1789_l2076_207612

theorem determine_f_1789
  (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, 0 < n → f (f n) = 4 * n + 9)
  (h2 : ∀ k : ℕ, f (2^k) = 2^(k+1) + 3) :
  f 1789 = 3581 :=
sorry

end determine_f_1789_l2076_207612


namespace quadratic_residue_l2076_207623

theorem quadratic_residue (a : ℤ) (p : ℕ) (hp : p > 2) (ha_nonzero : a ≠ 0) :
  (∃ b : ℤ, b^2 ≡ a [ZMOD p] → a^((p - 1) / 2) ≡ 1 [ZMOD p]) ∧
  (¬ ∃ b : ℤ, b^2 ≡ a [ZMOD p] → a^((p - 1) / 2) ≡ -1 [ZMOD p]) :=
sorry

end quadratic_residue_l2076_207623


namespace remainder_four_times_plus_six_l2076_207639

theorem remainder_four_times_plus_six (n : ℤ) (h : n % 5 = 3) : (4 * n + 6) % 5 = 3 :=
by
  sorry

end remainder_four_times_plus_six_l2076_207639


namespace new_light_wattage_l2076_207601

theorem new_light_wattage (w_old : ℕ) (p : ℕ) (w_new : ℕ) (h1 : w_old = 110) (h2 : p = 30) (h3 : w_new = w_old + (p * w_old / 100)) : w_new = 143 :=
by
  -- Using the conditions provided
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end new_light_wattage_l2076_207601


namespace sum_of_products_mod_7_l2076_207621

-- Define the numbers involved
def a := 1789
def b := 1861
def c := 1945
def d := 1533
def e := 1607
def f := 1688

-- Define the sum of products
def sum_of_products := a * b * c + d * e * f

-- The statement to prove:
theorem sum_of_products_mod_7 : sum_of_products % 7 = 3 := 
by sorry

end sum_of_products_mod_7_l2076_207621


namespace spherical_to_rectangular_example_l2076_207637

noncomputable def spherical_to_rectangular (ρ θ ϕ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin ϕ * Real.cos θ, ρ * Real.sin ϕ * Real.sin θ, ρ * Real.cos ϕ)

theorem spherical_to_rectangular_example :
  spherical_to_rectangular 4 (Real.pi / 4) (Real.pi / 6) = (Real.sqrt 2, Real.sqrt 2, 2 * Real.sqrt 3) :=
by
  sorry

end spherical_to_rectangular_example_l2076_207637
