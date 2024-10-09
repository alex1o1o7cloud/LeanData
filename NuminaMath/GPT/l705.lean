import Mathlib

namespace triangle_smallest_side_l705_70545

theorem triangle_smallest_side (a b c : ℝ) (h : b^2 + c^2 ≥ 5 * a^2) : 
    (a ≤ b ∧ a ≤ c) := 
sorry

end triangle_smallest_side_l705_70545


namespace roof_area_l705_70502

-- Definitions based on conditions
variables (l w : ℝ)
def length_eq_five_times_width : Prop := l = 5 * w
def length_minus_width_eq_48 : Prop := l - w = 48

-- Proof goal
def area_of_roof : Prop := l * w = 720

-- Lean 4 statement asserting the mathematical problem
theorem roof_area (l w : ℝ) 
  (H1 : length_eq_five_times_width l w)
  (H2 : length_minus_width_eq_48 l w) : 
  area_of_roof l w := 
  by sorry

end roof_area_l705_70502


namespace describe_set_T_l705_70506

theorem describe_set_T:
  ( ∀ (x y : ℝ), ((x + 2 = 4 ∧ y - 5 ≤ 4) ∨ (y - 5 = 4 ∧ x + 2 ≤ 4) ∨ (x + 2 = y - 5 ∧ 4 ≤ x + 2)) →
    ( ∃ (x y : ℝ), x = 2 ∧ y ≤ 9 ∨ y = 9 ∧ x ≤ 2 ∨ y = x + 7 ∧ x ≥ 2 ∧ y ≥ 9) ) :=
sorry

end describe_set_T_l705_70506


namespace sin_alpha_plus_beta_alpha_plus_two_beta_l705_70551

variables {α β : ℝ} (hα_acute : 0 < α ∧ α < π / 2) (hβ_acute : 0 < β ∧ β < π / 2)
          (h_tan_α : Real.tan α = 1 / 7) (h_sin_β : Real.sin β = Real.sqrt 10 / 10)

theorem sin_alpha_plus_beta : 
    Real.sin (α + β) = Real.sqrt 5 / 5 :=
by
  sorry

theorem alpha_plus_two_beta : 
    α + 2 * β = π / 4 :=
by
  sorry

end sin_alpha_plus_beta_alpha_plus_two_beta_l705_70551


namespace converse_equivalence_l705_70521

-- Definition of the original proposition
def original_proposition : Prop := ∀ (x : ℝ), x < 0 → x^2 > 0

-- Definition of the converse proposition
def converse_proposition : Prop := ∀ (x : ℝ), x^2 > 0 → x < 0

-- Theorem statement asserting the equivalence
theorem converse_equivalence : (converse_proposition = ¬ original_proposition) :=
sorry

end converse_equivalence_l705_70521


namespace find_x_l705_70525

-- Definitions to capture angles and triangle constraints
def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

def perpendicular (A B : ℝ) : Prop := A + B = 90

-- Given conditions
axiom angle_ABC : ℝ
axiom angle_BAC : ℝ
axiom angle_BCA : ℝ
axiom angle_DCE : ℝ
axiom angle_x : ℝ

-- Specific values for the angles provided in the problem
axiom angle_ABC_is_70 : angle_ABC = 70
axiom angle_BAC_is_50 : angle_BAC = 50

-- Angle BCA in triangle ABC
axiom angle_sum_ABC : angle_sum_triangle angle_ABC angle_BAC angle_BCA

-- Conditional relationships in triangle CDE
axiom angle_DCE_equals_BCA : angle_DCE = angle_BCA
axiom angle_sum_CDE : perpendicular angle_DCE angle_x

-- The theorem we need to prove
theorem find_x : angle_x = 30 := sorry

end find_x_l705_70525


namespace machine_work_hours_l705_70505

theorem machine_work_hours (A B : ℝ) (x : ℝ) (hA : A = 1 / 8) (hB : B = A / 4)
  (hB_rate : B = 1 / 32) (B_time : B * 8 = 1 - x / 8) : x = 6 :=
by
  sorry

end machine_work_hours_l705_70505


namespace min_small_bottles_needed_l705_70544

theorem min_small_bottles_needed (small_capacity large_capacity : ℕ) 
    (h_small_capacity : small_capacity = 35) (h_large_capacity : large_capacity = 500) : 
    ∃ n, n = 15 ∧ large_capacity <= n * small_capacity :=
by 
  sorry

end min_small_bottles_needed_l705_70544


namespace bridge_length_is_correct_l705_70579

noncomputable def length_of_bridge (length_of_train : ℝ) (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  let total_distance := speed_ms * time_sec
  total_distance - length_of_train

theorem bridge_length_is_correct :
  length_of_bridge 160 45 30 = 215 := by
  sorry

end bridge_length_is_correct_l705_70579


namespace razorback_shop_jersey_revenue_l705_70550

theorem razorback_shop_jersey_revenue :
  let price_per_tshirt := 67
  let price_per_jersey := 165
  let tshirts_sold := 74
  let jerseys_sold := 156
  jerseys_sold * price_per_jersey = 25740 := by
  sorry

end razorback_shop_jersey_revenue_l705_70550


namespace g_constant_term_l705_70536

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := f * g

-- Conditions from the problem
def f_has_constant_term_5 : f.coeff 0 = 5 := sorry
def h_has_constant_term_neg_10 : h.coeff 0 = -10 := sorry
def g_is_quadratic : g.degree ≤ 2 := sorry

-- Statement of the problem
theorem g_constant_term : g.coeff 0 = -2 :=
by
  have h_eq_fg : h = f * g := rfl
  have f_const := f_has_constant_term_5
  have h_const := h_has_constant_term_neg_10
  have g_quad := g_is_quadratic
  sorry

end g_constant_term_l705_70536


namespace find_x_l705_70513

theorem find_x
  (x : ℝ)
  (h : (x + 1) / (x + 5) = (x + 5) / (x + 13)) :
  x = 3 :=
sorry

end find_x_l705_70513


namespace find_sum_of_xyz_l705_70519

theorem find_sum_of_xyz : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  (151 / 44 : ℚ) = 3 + 1 / (x + 1 / (y + 1 / z)) ∧ x + y + z = 11 :=
by 
  sorry

end find_sum_of_xyz_l705_70519


namespace count_valid_orderings_l705_70556

-- Define the houses and conditions
inductive HouseColor where
  | Green
  | Purple
  | Blue
  | Pink
  | X -- Representing the fifth unspecified house

open HouseColor

def validOrderings : List (List HouseColor) :=
  [
    [Green, Blue, Purple, Pink, X], 
    [Green, Blue, X, Purple, Pink],
    [Green, X, Purple, Blue, Pink],
    [X, Pink, Purple, Blue, Green],
    [X, Purple, Pink, Blue, Green],
    [X, Pink, Blue, Purple, Green]
  ] 

-- Prove that there are exactly 6 valid orderings
theorem count_valid_orderings : (validOrderings.length = 6) :=
by
  -- Since we list all possible valid orderings above, just compute the length
  sorry

end count_valid_orderings_l705_70556


namespace find_b_l705_70593

noncomputable def given_c := 3
noncomputable def given_C := Real.pi / 3
noncomputable def given_cos_C := 1 / 2
noncomputable def given_a (b : ℝ) := 2 * b

theorem find_b (b : ℝ) (h1 : given_c = 3) (h2 : given_cos_C = Real.cos (given_C)) (h3 : given_a b = 2 * b) : b = Real.sqrt 3 := 
by
  sorry

end find_b_l705_70593


namespace evaluation_of_expression_l705_70590

theorem evaluation_of_expression : (3^2 - 2^2 + 1^2) = 6 :=
by
  sorry

end evaluation_of_expression_l705_70590


namespace length_of_one_side_of_square_l705_70558

variable (total_ribbon_length : ℕ) (triangle_perimeter : ℕ)

theorem length_of_one_side_of_square (h1 : total_ribbon_length = 78)
                                    (h2 : triangle_perimeter = 46) :
  (total_ribbon_length - triangle_perimeter) / 4 = 8 :=
by
  sorry

end length_of_one_side_of_square_l705_70558


namespace angle_between_clock_hands_at_3_05_l705_70575

theorem angle_between_clock_hands_at_3_05 :
  let minute_angle := 5 * 6
  let hour_angle := (5 / 60) * 30
  let initial_angle := 3 * 30
  initial_angle - minute_angle + hour_angle = 62.5 := by
  sorry

end angle_between_clock_hands_at_3_05_l705_70575


namespace solve_fraction_problem_l705_70581

theorem solve_fraction_problem (n : ℝ) (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by
  sorry

end solve_fraction_problem_l705_70581


namespace part1_part2_l705_70570

-- The quadratic equation of interest
def quadratic_eq (k x : ℝ) : ℝ :=
  x^2 + (2 * k - 1) * x + k^2 - k

-- Part 1: Proof that the equation has two distinct real roots
theorem part1 (k : ℝ) : (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic_eq k x1 = 0 ∧ quadratic_eq k x2 = 0) := 
  sorry

-- Part 2: Given x = 2 is a root, prove the value of the expression
theorem part2 (k : ℝ) (h : quadratic_eq k 2 = 0) : -2 * k^2 - 6 * k - 5 = -1 :=
  sorry

end part1_part2_l705_70570


namespace lamp_cost_l705_70580

def saved : ℕ := 500
def couch : ℕ := 750
def table : ℕ := 100
def remaining_owed : ℕ := 400

def total_cost_without_lamp : ℕ := couch + table

theorem lamp_cost :
  total_cost_without_lamp - saved + lamp = remaining_owed → lamp = 50 := by
  sorry

end lamp_cost_l705_70580


namespace initial_amount_l705_70503

theorem initial_amount (M : ℝ) 
  (H1 : M * (2/3) * (4/5) * (3/4) * (5/7) * (5/6) = 200) : 
  M = 840 :=
by
  -- Proof to be provided
  sorry

end initial_amount_l705_70503


namespace inequality_proof_inequality_equality_conditions_l705_70546

theorem inequality_proof
  (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hxy1 : x1 * y1 - z1 ^ 2 > 0) (hxy2 : x2 * y2 - z2 ^ 2 > 0) :
  (x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2 ≤ (1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2)) :=
sorry

theorem inequality_equality_conditions
  (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hxy1 : x1 * y1 - z1 ^ 2 > 0) (hxy2 : x2 * y2 - z2 ^ 2 > 0) :
  ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2 = (1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2))
  ↔ (x1 = x2 ∧ y1 = y2 ∧ z1 = z2)) :=
sorry

end inequality_proof_inequality_equality_conditions_l705_70546


namespace degrees_of_remainder_division_l705_70508

theorem degrees_of_remainder_division (f g : Polynomial ℝ) (h : g = Polynomial.C 3 * Polynomial.X ^ 3 + Polynomial.C (-4) * Polynomial.X ^ 2 + Polynomial.C 1 * Polynomial.X + Polynomial.C (-8)) :
  ∃ r q : Polynomial ℝ, f = g * q + r ∧ (r.degree < 3) := 
sorry

end degrees_of_remainder_division_l705_70508


namespace elvis_recording_time_l705_70591

theorem elvis_recording_time :
  ∀ (total_studio_time writing_time_per_song editing_time number_of_songs : ℕ),
  total_studio_time = 300 →
  writing_time_per_song = 15 →
  editing_time = 30 →
  number_of_songs = 10 →
  (total_studio_time - (number_of_songs * writing_time_per_song + editing_time)) / number_of_songs = 12 :=
by
  intros total_studio_time writing_time_per_song editing_time number_of_songs
  intros h1 h2 h3 h4
  sorry

end elvis_recording_time_l705_70591


namespace nontrivial_solution_exists_l705_70557

theorem nontrivial_solution_exists 
  (a b : ℤ) 
  (h_square_a : ∀ k : ℤ, a ≠ k^2) 
  (h_square_b : ∀ k : ℤ, b ≠ k^2) 
  (h_nontrivial : ∃ (x y z w : ℤ), x^2 - a * y^2 - b * z^2 + a * b * w^2 = 0 ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) : 
  ∃ (x y z : ℤ), x^2 - a * y^2 - b * z^2 = 0 ∧ (x, y, z) ≠ (0, 0, 0) :=
by
  sorry

end nontrivial_solution_exists_l705_70557


namespace cubes_difference_l705_70564

theorem cubes_difference :
  let a := 642
  let b := 641
  a^3 - b^3 = 1234567 :=
by
  let a := 642
  let b := 641
  have h : a^3 - b^3 = 264609288 - 263374721 := sorry
  have h_correct : 264609288 - 263374721 = 1234567 := sorry
  exact Eq.trans h h_correct

end cubes_difference_l705_70564


namespace total_weekly_earnings_l705_70568

-- Define the total weekly hours and earnings
def weekly_hours_weekday : ℕ := 5 * 5
def weekday_rate : ℕ := 3
def weekday_earnings : ℕ := weekly_hours_weekday * weekday_rate

-- Define the total weekend hours and earnings
def weekend_days : ℕ := 2
def weekend_hours_per_day : ℕ := 3
def weekend_rate : ℕ := 3 * 2
def weekend_hours : ℕ := weekend_days * weekend_hours_per_day
def weekend_earnings : ℕ := weekend_hours * weekend_rate

-- Prove that Mitch's total earnings per week are $111
theorem total_weekly_earnings : weekday_earnings + weekend_earnings = 111 := by
  sorry

end total_weekly_earnings_l705_70568


namespace tea_in_each_box_initially_l705_70528

theorem tea_in_each_box_initially (x : ℕ) 
  (h₁ : 4 * (x - 9) = x) : 
  x = 12 := 
sorry

end tea_in_each_box_initially_l705_70528


namespace amount_c_gets_l705_70537

theorem amount_c_gets (total_amount : ℕ) (ratio_b ratio_c : ℕ) (h_total_amount : total_amount = 2000) (h_ratio : ratio_b = 4 ∧ ratio_c = 16) : ∃ (c_amount: ℕ), c_amount = 1600 :=
by
  sorry

end amount_c_gets_l705_70537


namespace wheel_radius_l705_70538

theorem wheel_radius 
(D: ℝ) (N: ℕ) (r: ℝ) 
(hD: D = 88 * 1000) 
(hN: N = 1000) 
(hC: 2 * Real.pi * r * N = D) : 
r = 88 / (2 * Real.pi) :=
by
  sorry

end wheel_radius_l705_70538


namespace problem_statement_l705_70571

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem problem_statement :
  (∀ x : ℝ, f (x) = 0 → x = - Real.pi / 6) ∧ (∀ x : ℝ, f (x) = 4 * Real.cos (2 * x - Real.pi / 6)) := sorry

end problem_statement_l705_70571


namespace max_fraction_l705_70541

theorem max_fraction (x y : ℝ) (h1 : -6 ≤ x) (h2 : x ≤ -3) (h3 : 3 ≤ y) (h4 : y ≤ 5) :
  (∀ x y, -6 ≤ x → x ≤ -3 → 3 ≤ y → y ≤ 5 → (x - y) / y ≤ -2) :=
by
  sorry

end max_fraction_l705_70541


namespace probability_of_selection_l705_70530

/-- A school selects 80 students for a discussion from a total of 883 students. First, 3 people are eliminated using simple random sampling, and then 80 are selected from the remaining 880 using systematic sampling. Prove that the probability of each person being selected is 80/883. -/
theorem probability_of_selection (total_students : ℕ) (students_eliminated : ℕ) (students_selected : ℕ) 
  (h_total : total_students = 883) (h_eliminated : students_eliminated = 3) (h_selected : students_selected = 80) :
  ((total_students - students_eliminated) * students_selected) / (total_students * (total_students - students_eliminated)) = 80 / 883 :=
by
  sorry

end probability_of_selection_l705_70530


namespace no_real_solutions_if_discriminant_neg_one_real_solution_if_discriminant_zero_more_than_one_real_solution_if_discriminant_pos_l705_70569

noncomputable def system_discriminant (a b c : ℝ) : ℝ := (b - 1)^2 - 4 * a * c

theorem no_real_solutions_if_discriminant_neg (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c < 0) :
  ¬∃ (x₁ x₂ x₃ : ℝ), (a * x₁^2 + b * x₁ + c = x₂) ∧
                      (a * x₂^2 + b * x₂ + c = x₃) ∧
                      (a * x₃^2 + b * x₃ + c = x₁) :=
sorry

theorem one_real_solution_if_discriminant_zero (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c = 0) :
  ∃ (x : ℝ), ∀ (x₁ x₂ x₃ : ℝ), (x₁ = x) ∧ (x₂ = x) ∧ (x₃ = x) ∧
                              (a * x₁^2 + b * x₁ + c = x₂) ∧
                              (a * x₂^2 + b * x₂ + c = x₃) ∧
                              (a * x₃^2 + b * x₃ + c = x₁)  :=
sorry

theorem more_than_one_real_solution_if_discriminant_pos (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c > 0) :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = x₂) ∧
                      (a * x₂^2 + b * x₂ + c = x₃) ∧
                      (a * x₃^2 + b * x₃ + c = x₁) :=
sorry

end no_real_solutions_if_discriminant_neg_one_real_solution_if_discriminant_zero_more_than_one_real_solution_if_discriminant_pos_l705_70569


namespace dryer_runtime_per_dryer_l705_70576

-- Definitions for the given conditions
def washer_cost : ℝ := 4
def dryer_cost_per_10min : ℝ := 0.25
def loads_of_laundry : ℕ := 2
def num_dryers : ℕ := 3
def total_spent : ℝ := 11

-- Statement to prove
theorem dryer_runtime_per_dryer : 
  (2 * washer_cost + ((total_spent - 2 * washer_cost) / dryer_cost_per_10min) * 10) / num_dryers = 40 :=
by
  sorry

end dryer_runtime_per_dryer_l705_70576


namespace sin_range_l705_70522

theorem sin_range (x : ℝ) (h : x ∈ Set.Icc (Real.pi / 6) (Real.pi / 2)) : 
  Set.range (fun x => Real.sin x) = Set.Icc (1/2 : ℝ) 1 :=
sorry

end sin_range_l705_70522


namespace height_difference_l705_70524

def burj_khalifa_height : ℝ := 830
def sears_tower_height : ℝ := 527

theorem height_difference : burj_khalifa_height - sears_tower_height = 303 := 
by
  sorry

end height_difference_l705_70524


namespace unique_representation_l705_70529

theorem unique_representation {p x y : ℕ} 
  (hp : p > 2 ∧ Prime p) 
  (h : 2 * y = p * (x + y)) 
  (hx : x ≠ y) : 
  ∃ x y : ℕ, (1/x + 1/y = 2/p) ∧ x ≠ y := 
sorry

end unique_representation_l705_70529


namespace solution_set_of_inequality_l705_70548

theorem solution_set_of_inequality :
  { x : ℝ | abs (x - 4) + abs (3 - x) < 2 } = { x : ℝ | 2.5 < x ∧ x < 4.5 } := sorry

end solution_set_of_inequality_l705_70548


namespace total_consumer_installment_credit_l705_70587

-- Conditions
def auto_instalment_credit (C : ℝ) : ℝ := 0.2 * C
def auto_finance_extends_1_third (auto_installment : ℝ) : ℝ := 57
def student_loans (C : ℝ) : ℝ := 0.15 * C
def credit_card_debt (C : ℝ) (auto_installment : ℝ) : ℝ := 0.25 * C
def other_loans (C : ℝ) : ℝ := 0.4 * C

-- Correct Answer
theorem total_consumer_installment_credit (C : ℝ) :
  auto_instalment_credit C / 3 = auto_finance_extends_1_third (auto_instalment_credit C) ∧
  student_loans C = 80 ∧
  credit_card_debt C (auto_instalment_credit C) = auto_instalment_credit C + 100 ∧
  credit_card_debt C (auto_instalment_credit C) = 271 →
  C = 1084 := 
by
  sorry

end total_consumer_installment_credit_l705_70587


namespace solve_fractional_equation_l705_70512

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 2) : 
  (4 * x ^ 2 + 3 * x + 2) / (x - 2) = 4 * x + 5 ↔ x = -2 := by 
  sorry

end solve_fractional_equation_l705_70512


namespace batsman_average_after_17th_l705_70510

def runs_17th_inning : ℕ := 87
def increase_in_avg : ℕ := 4
def num_innings : ℕ := 17

theorem batsman_average_after_17th (A : ℕ) (H : A + increase_in_avg = (16 * A + runs_17th_inning) / num_innings) : 
  (A + increase_in_avg) = 23 := sorry

end batsman_average_after_17th_l705_70510


namespace volume_of_original_cube_l705_70595

theorem volume_of_original_cube (s : ℝ) (h : (s + 2) * (s - 3) * s - s^3 = 26) : s^3 = 343 := 
sorry

end volume_of_original_cube_l705_70595


namespace probability_of_target_destroyed_l705_70509

theorem probability_of_target_destroyed :
  let p1 := 0.9
  let p2 := 0.9
  let p3 := 0.8
  (p1 * p2 * p3) + (p1 * p2 * (1 - p3)) + (p1 * (1 - p2) * p3) + ((1 - p1) * p2 * p3) = 0.954 :=
by
  let p1 := 0.9
  let p2 := 0.9
  let p3 := 0.8
  sorry

end probability_of_target_destroyed_l705_70509


namespace no_real_intersections_l705_70572

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l705_70572


namespace constants_sum_l705_70586

theorem constants_sum (A B C D : ℕ) 
  (h : ∀ n : ℕ, n ≥ 4 → n^4 = A * (n.choose 4) + B * (n.choose 3) + C * (n.choose 2) + D * (n.choose 1)) 
  : A + B + C + D = 75 :=
by
  sorry

end constants_sum_l705_70586


namespace find_x_l705_70540

variable (x : ℤ)

-- Define the conditions based on the problem
def adjacent_sum_condition := 
  (x + 15) + (x + 8) + (x - 7) = x

-- State the goal, which is to prove x = -8
theorem find_x : x = -8 :=
by
  have h : adjacent_sum_condition x := sorry
  sorry

end find_x_l705_70540


namespace monomial_2023_eq_l705_70566

def monomial (n : ℕ) : ℤ × ℕ :=
  ((-1)^(n+1) * (2*n - 1), n)

theorem monomial_2023_eq : monomial 2023 = (4045, 2023) :=
by
  sorry

end monomial_2023_eq_l705_70566


namespace mike_sold_song_book_for_correct_amount_l705_70507

-- Define the constants for the cost of the trumpet and the net amount spent
def cost_of_trumpet : ℝ := 145.16
def net_amount_spent : ℝ := 139.32

-- Define the amount received from selling the song book
def amount_received_from_selling_song_book : ℝ :=
  cost_of_trumpet - net_amount_spent

-- The theorem stating the amount Mike sold the song book for
theorem mike_sold_song_book_for_correct_amount :
  amount_received_from_selling_song_book = 5.84 :=
sorry

end mike_sold_song_book_for_correct_amount_l705_70507


namespace f_div_36_l705_70592

open Nat

def f (n : ℕ) : ℕ :=
  (2 * n + 7) * 3^n + 9

theorem f_div_36 (n : ℕ) : (f n) % 36 = 0 := 
  sorry

end f_div_36_l705_70592


namespace taishan_maiden_tea_prices_l705_70518

theorem taishan_maiden_tea_prices (x y : ℝ) 
  (h1 : 30 * x + 20 * y = 6000)
  (h2 : 24 * x + 18 * y = 5100) :
  x = 100 ∧ y = 150 :=
by
  sorry

end taishan_maiden_tea_prices_l705_70518


namespace hotdogs_per_hour_l705_70527

-- Define the necessary conditions
def price_per_hotdog : ℝ := 2
def total_hours : ℝ := 10
def total_sales : ℝ := 200

-- Prove that the number of hot dogs sold per hour equals 10
theorem hotdogs_per_hour : (total_sales / total_hours) / price_per_hotdog = 10 :=
by
  sorry

end hotdogs_per_hour_l705_70527


namespace total_spent_on_toys_l705_70520

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59
def total_spent : ℝ := 12.30

theorem total_spent_on_toys : football_cost + marbles_cost = total_spent :=
by sorry

end total_spent_on_toys_l705_70520


namespace min_value_condition_l705_70589

open Real

theorem min_value_condition 
  (m n : ℝ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : 2 * m + n = 1) : 
  (1 / m + 2 / n) ≥ 8 :=
sorry

end min_value_condition_l705_70589


namespace axis_of_symmetry_l705_70554

theorem axis_of_symmetry (x : ℝ) (h : x = -Real.pi / 12) :
  ∃ k : ℤ, 2 * x - Real.pi / 3 = k * Real.pi + Real.pi / 2 :=
sorry

end axis_of_symmetry_l705_70554


namespace solve_system_eq_l705_70511

theorem solve_system_eq (x y : ℝ) (h1 : 2 * x - y = 3) (h2 : 3 * x + 2 * y = 8) :
  x = 2 ∧ y = 1 :=
by
  sorry

end solve_system_eq_l705_70511


namespace find_stream_speed_l705_70596

variable (D : ℝ) (v : ℝ)

theorem find_stream_speed 
  (h1 : ∀D v, D / (63 - v) = 2 * (D / (63 + v)))
  (h2 : v = 21) :
  true := 
  by
  sorry

end find_stream_speed_l705_70596


namespace travel_A_to_D_l705_70599

-- Definitions for the number of roads between each pair of cities
def roads_A_to_B : ℕ := 3
def roads_A_to_C : ℕ := 1
def roads_B_to_C : ℕ := 2
def roads_B_to_D : ℕ := 1
def roads_C_to_D : ℕ := 3

-- Theorem stating the total number of ways to travel from A to D visiting each city exactly once
theorem travel_A_to_D : roads_A_to_B * roads_B_to_C * roads_C_to_D + roads_A_to_C * roads_B_to_C * roads_B_to_D = 20 :=
by
  -- Formal proof goes here
  sorry

end travel_A_to_D_l705_70599


namespace brad_started_after_maxwell_l705_70594

theorem brad_started_after_maxwell :
  ∀ (distance maxwell_speed brad_speed maxwell_time : ℕ),
  distance = 94 →
  maxwell_speed = 4 →
  brad_speed = 6 →
  maxwell_time = 10 →
  (distance - maxwell_speed * maxwell_time) / brad_speed = 9 := 
by
  intros distance maxwell_speed brad_speed maxwell_time h_dist h_m_speed h_b_speed h_m_time
  sorry

end brad_started_after_maxwell_l705_70594


namespace smallest_three_digit_integer_solution_l705_70543

theorem smallest_three_digit_integer_solution :
  ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    (∃ a b c : ℕ,
      n = 100 * a + 10 * b + c ∧
      1 ≤ a ∧ a ≤ 9 ∧
      0 ≤ b ∧ b ≤ 9 ∧ 
      0 ≤ c ∧ c ≤ 9 ∧
      2 * n = 100 * c + 10 * b + a + 5) ∧ 
    n = 102 := by
{
  sorry
}

end smallest_three_digit_integer_solution_l705_70543


namespace at_least_one_zero_l705_70533

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem at_least_one_zero (p q : ℝ) (h_zero : ∃ m : ℝ, f m p q = 0 ∧ f (f (f m p q) p q) p q = 0) :
  f 0 p q = 0 ∨ f 1 p q = 0 :=
sorry

end at_least_one_zero_l705_70533


namespace simplify_expression_l705_70565

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) : (a - 2) * (b - 2) = -2 * m := 
by
  sorry

end simplify_expression_l705_70565


namespace inequality_must_hold_l705_70504

theorem inequality_must_hold (x y : ℝ) (h : x > y) : -2 * x < -2 * y :=
sorry

end inequality_must_hold_l705_70504


namespace average_monthly_growth_rate_proof_profit_in_may_proof_l705_70585

theorem average_monthly_growth_rate_proof :
  ∃ r : ℝ, 2400 * (1 + r)^2 = 3456 ∧ r = 0.2 := sorry

theorem profit_in_may_proof (r : ℝ) (h_r : r = 0.2) :
  3456 * (1 + r) = 4147.2 := sorry

end average_monthly_growth_rate_proof_profit_in_may_proof_l705_70585


namespace arithmetic_expression_eval_l705_70516

theorem arithmetic_expression_eval :
  ((26.3 * 12 * 20) / 3) + 125 = 2229 :=
sorry

end arithmetic_expression_eval_l705_70516


namespace negation_implication_l705_70574

theorem negation_implication (a b c : ℝ) : 
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) :=
by 
  sorry

end negation_implication_l705_70574


namespace simplify_expression_l705_70531

variable (x : ℝ)
variable (h₁ : x ≠ 2)
variable (h₂ : x ≠ 3)
variable (h₃ : x ≠ 4)
variable (h₄ : x ≠ 5)

theorem simplify_expression : 
  ( (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 9) / (x^2 - 8*x + 15)) 
  = ( (x - 1) * (x - 5) ) / ( (x - 4) * (x - 2) * (x - 3) ) ) :=
by sorry

end simplify_expression_l705_70531


namespace slope_acute_l705_70549

noncomputable def curve (a : ℤ) : ℝ → ℝ := λ x => x^3 - 2 * a * x^2 + 2 * a * x

noncomputable def tangent_slope (a : ℤ) : ℝ → ℝ := λ x => 3 * x^2 - 4 * a * x + 2 * a

theorem slope_acute (a : ℤ) : (∀ x : ℝ, (tangent_slope a x > 0)) ↔ (a = 1) := sorry

end slope_acute_l705_70549


namespace find_missing_number_l705_70515

theorem find_missing_number (n : ℝ) : n * 120 = 173 * 240 → n = 345.6 :=
by
  intros h
  sorry

end find_missing_number_l705_70515


namespace sequence_arithmetic_l705_70514

theorem sequence_arithmetic (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h : ∀ n, S n = 2 * n^2 - 3 * n)
  (h₀ : S 0 = 0) 
  (h₁ : ∀ n, S (n+1) = S n + a (n+1)) :
  ∀ n, a n = 4 * n - 1 := sorry

end sequence_arithmetic_l705_70514


namespace problem_3_equals_answer_l705_70573

variable (a : ℝ)

theorem problem_3_equals_answer :
  (-2 * a^2)^3 / (2 * a^2) = -4 * a^4 :=
by
  sorry

end problem_3_equals_answer_l705_70573


namespace part1_part2_l705_70559

def p (x a : ℝ) : Prop := x - a < 0
def q (x : ℝ) : Prop := x * x - 4 * x + 3 ≤ 0

theorem part1 (a : ℝ) (h : a = 2) (hpq : ∀ x : ℝ, p x a ∧ q x) :
  Set.Ico 1 (2 : ℝ) = {x : ℝ | p x a ∧ q x} :=
by {
  sorry
}

theorem part2 (hp : ∀ (x a : ℝ), p x a → ¬ q x) : {a : ℝ | ∀ x : ℝ, q x → p x a} = Set.Ioi 3 :=
by {
  sorry
}

end part1_part2_l705_70559


namespace problem_real_numbers_inequality_l705_70562

open Real

theorem problem_real_numbers_inequality 
  (a1 b1 a2 b2 : ℝ) :
  a1 * b1 + a2 * b2 ≤ sqrt (a1^2 + a2^2) * sqrt (b1^2 + b2^2) :=
by 
  sorry

end problem_real_numbers_inequality_l705_70562


namespace friends_lunch_spending_l705_70539

-- Problem conditions and statement to prove
theorem friends_lunch_spending (x : ℝ) (h1 : x + (x + 15) + (x - 20) + 2 * x = 100) : 
  x = 21 :=
by sorry

end friends_lunch_spending_l705_70539


namespace remainder_of_power_modulo_l705_70555

theorem remainder_of_power_modulo : (3^2048) % 11 = 5 := by
  sorry

end remainder_of_power_modulo_l705_70555


namespace service_center_location_l705_70501

-- Definitions from conditions
def third_exit := 30
def twelfth_exit := 195
def seventh_exit := 90

-- Concept of distance and service center location
def distance := seventh_exit - third_exit
def service_center_milepost := third_exit + 2 * distance / 3

-- The theorem to prove
theorem service_center_location : service_center_milepost = 70 := by
  -- Sorry is used to skip the proof details.
  sorry

end service_center_location_l705_70501


namespace other_team_members_points_l705_70560

theorem other_team_members_points :
  ∃ (x : ℕ), ∃ (y : ℕ), (y ≤ 9 * 3) ∧ (x = y + 18 + x / 3 + x / 5) ∧ y = 24 :=
by
  sorry

end other_team_members_points_l705_70560


namespace triangular_region_area_l705_70561

def line (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def y (x : ℝ) := x

theorem triangular_region_area : 
  ∀ (x y: ℝ),
  (y = line 1 2 x ∧ y = 3) ∨ 
  (y = line (-1) 8 x ∧ y = 3) ∨ 
  (y = line 1 2 x ∧ y = line (-1) 8 x)
  →
  ∃ (area: ℝ), area = 4.00 := 
by
  sorry

end triangular_region_area_l705_70561


namespace solution_set_of_inequality_l705_70584

theorem solution_set_of_inequality (x : ℝ) :
  (x - 1) * (x - 2) ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end solution_set_of_inequality_l705_70584


namespace nick_paints_wall_in_fraction_l705_70597

theorem nick_paints_wall_in_fraction (nick_paint_time wall_paint_time : ℕ) (h1 : wall_paint_time = 60) (h2 : nick_paint_time = 12) : (nick_paint_time * 1 / wall_paint_time = 1 / 5) :=
by
  sorry

end nick_paints_wall_in_fraction_l705_70597


namespace geometric_sequence_common_ratio_l705_70598

theorem geometric_sequence_common_ratio :
  (∃ q : ℝ, 1 + q + q^2 = 13 ∧ (q = 3 ∨ q = -4)) :=
by
  sorry

end geometric_sequence_common_ratio_l705_70598


namespace ratio_a_to_c_l705_70588

variable (a b c : ℚ)

theorem ratio_a_to_c (h1 : a / b = 7 / 3) (h2 : b / c = 1 / 5) : a / c = 7 / 15 := 
sorry

end ratio_a_to_c_l705_70588


namespace solve_trig_equation_l705_70583

theorem solve_trig_equation (x : ℝ) : 
  2 * Real.cos (13 * x) + 3 * Real.cos (3 * x) + 3 * Real.cos (5 * x) - 8 * Real.cos x * (Real.cos (4 * x))^3 = 0 ↔ 
  ∃ (k : ℤ), x = (k * Real.pi) / 12 :=
sorry

end solve_trig_equation_l705_70583


namespace cube_cut_possible_l705_70577

theorem cube_cut_possible (a b : ℝ) (unit_a : a = 1) (unit_b : b = 1) : 
  ∃ (cut : ℝ → ℝ → Prop), (∀ x y, cut x y → (∃ q r : ℝ, q > 0 ∧ r > 0 ∧ q * r > 1)) :=
sorry

end cube_cut_possible_l705_70577


namespace abs_sub_eq_five_l705_70523

theorem abs_sub_eq_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 :=
sorry

end abs_sub_eq_five_l705_70523


namespace floor_tiling_l705_70532

-- Define that n can be expressed as 7k for some integer k.
theorem floor_tiling (n : ℕ) (h : ∃ x : ℕ, n^2 = 7 * x) : ∃ k : ℕ, n = 7 * k := by
  sorry

end floor_tiling_l705_70532


namespace number_of_adult_dogs_l705_70535

theorem number_of_adult_dogs (x : ℕ) (h : 2 * 50 + x * 100 + 2 * 150 = 700) : x = 3 :=
by
  -- Definitions from conditions
  have cost_cats := 2 * 50
  have cost_puppies := 2 * 150
  have total_cost := 700
  
  -- Using the provided hypothesis to assert our proof
  sorry

end number_of_adult_dogs_l705_70535


namespace find_a_b_extreme_values_l705_70582

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + b * x - (2/3)

theorem find_a_b_extreme_values : 
  ∃ (a b : ℝ), 
    (a = -2) ∧ 
    (b = 3) ∧ 
    (f 1 (-2) 3 = 2/3) ∧ 
    (f 3 (-2) 3 = -2/3) :=
by
  sorry

end find_a_b_extreme_values_l705_70582


namespace adam_age_l705_70542

theorem adam_age (x : ℤ) :
  (∃ m : ℤ, x - 2 = m^2) ∧ (∃ n : ℤ, x + 2 = n^3) → x = 6 :=
by
  sorry

end adam_age_l705_70542


namespace p_at_zero_l705_70563

-- Definitions according to given conditions
def p (x : ℝ) : ℝ := sorry  -- Polynomial of degree 6 with specific values

-- Given condition: Degree of polynomial
def degree_p : Prop := (∀ n : ℕ, (n ≤ 6) → p (3 ^ n) = 1 / 3 ^ n)

-- Theorem that needs to be proved
theorem p_at_zero : degree_p → p 0 = 6560 / 2187 := 
by
  sorry

end p_at_zero_l705_70563


namespace f_value_l705_70547

noncomputable def f : ℝ → ℝ
| x => if x > 1 then 2^(x-1) else Real.tan (Real.pi * x / 3)

theorem f_value : f (1 / f 2) = Real.sqrt 3 / 3 := by
  sorry

end f_value_l705_70547


namespace longest_collection_pages_l705_70526

theorem longest_collection_pages 
    (pages_per_inch_miles : ℕ := 5) 
    (pages_per_inch_daphne : ℕ := 50) 
    (height_miles : ℕ := 240) 
    (height_daphne : ℕ := 25) : 
  max (pages_per_inch_miles * height_miles) (pages_per_inch_daphne * height_daphne) = 1250 := 
by
  sorry

end longest_collection_pages_l705_70526


namespace fraction_of_b_eq_three_tenths_a_l705_70553

theorem fraction_of_b_eq_three_tenths_a (a b : ℝ) (h1 : a + b = 100) (h2 : b = 60) :
  (3 / 10) * a = (1 / 5) * b :=
by 
  have h3 : a = 40 := by linarith [h1, h2]
  rw [h2, h3]
  linarith

end fraction_of_b_eq_three_tenths_a_l705_70553


namespace find_abcd_from_N_l705_70552

theorem find_abcd_from_N (N : ℕ) (hN1 : N ≥ 10000) (hN2 : N < 100000)
  (hN3 : N % 100000 = (N ^ 2) % 100000) : (N / 10) / 10 / 10 / 10 = 2999 := by
  sorry

end find_abcd_from_N_l705_70552


namespace sequence_inequality_l705_70578

def F : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n+2) => F (n+1) + F n

theorem sequence_inequality (n : ℕ) :
  (F (n+1) : ℝ)^(1 / n) ≥ 1 + 1 / ((F n : ℝ)^(1 / n)) :=
by
  sorry

end sequence_inequality_l705_70578


namespace travis_takes_home_money_l705_70500

-- Define the conditions
def total_apples : ℕ := 10000
def apples_per_box : ℕ := 50
def price_per_box : ℕ := 35

-- Define the main theorem to be proved
theorem travis_takes_home_money : (total_apples / apples_per_box) * price_per_box = 7000 := by
  sorry

end travis_takes_home_money_l705_70500


namespace energy_consumption_correct_l705_70517

def initial_wattages : List ℕ := [60, 80, 100, 120]

def increased_wattages : List ℕ := initial_wattages.map (λ x => x + (x * 25 / 100))

def combined_wattage (ws : List ℕ) : ℕ := ws.sum

def daily_energy_consumption (cw : ℕ) : ℕ := cw * 6 / 1000

def total_energy_consumption (dec : ℕ) : ℕ := dec * 30

-- Main theorem statement
theorem energy_consumption_correct :
  total_energy_consumption (daily_energy_consumption (combined_wattage increased_wattages)) = 81 := 
sorry

end energy_consumption_correct_l705_70517


namespace shift_graph_sin_cos_l705_70567

open Real

theorem shift_graph_sin_cos :
  ∀ x : ℝ, sin (2 * x + π / 3) = cos (2 * (x + π / 12) - π / 3) :=
by
  sorry

end shift_graph_sin_cos_l705_70567


namespace original_price_of_cycle_l705_70534

variable (P : ℝ)

theorem original_price_of_cycle (h1 : 0.75 * P = 1050) : P = 1400 :=
sorry

end original_price_of_cycle_l705_70534
