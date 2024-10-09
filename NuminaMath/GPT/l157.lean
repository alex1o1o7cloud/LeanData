import Mathlib

namespace quadratic_positive_difference_l157_15726
open Real

theorem quadratic_positive_difference :
  ∀ (x : ℝ), (2*x^2 - 7*x + 1 = x + 31) →
    (abs ((2 + sqrt 19) - (2 - sqrt 19)) = 2 * sqrt 19) :=
by intros x h
   sorry

end quadratic_positive_difference_l157_15726


namespace midpoint_coordinates_l157_15770

theorem midpoint_coordinates (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 2) (hy1 : y1 = 9) (hx2 : x2 = 8) (hy2 : y2 = 3) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx, my) = (5, 6) :=
by
  rw [hx1, hy1, hx2, hy2]
  sorry

end midpoint_coordinates_l157_15770


namespace temperature_equivalence_l157_15716

theorem temperature_equivalence (x : ℝ) (h : x = (9 / 5) * x + 32) : x = -40 :=
sorry

end temperature_equivalence_l157_15716


namespace sales_percentage_l157_15779

theorem sales_percentage (pens_sales pencils_sales notebooks_sales : ℕ) 
  (h1 : pens_sales = 25)
  (h2 : pencils_sales = 20)
  (h3 : notebooks_sales = 30) :
  100 - (pens_sales + pencils_sales + notebooks_sales) = 25 :=
by
  sorry

end sales_percentage_l157_15779


namespace angies_monthly_salary_l157_15721

theorem angies_monthly_salary 
    (necessities_expense : ℕ)
    (taxes_expense : ℕ)
    (left_over : ℕ)
    (monthly_salary : ℕ) :
  necessities_expense = 42 → 
  taxes_expense = 20 → 
  left_over = 18 → 
  monthly_salary = necessities_expense + taxes_expense + left_over → 
  monthly_salary = 80 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end angies_monthly_salary_l157_15721


namespace projectiles_meet_in_90_minutes_l157_15754

theorem projectiles_meet_in_90_minutes
  (d : ℝ) (v1 : ℝ) (v2 : ℝ) (time_in_minutes : ℝ)
  (h_d : d = 1455)
  (h_v1 : v1 = 470)
  (h_v2 : v2 = 500)
  (h_time : time_in_minutes = 90) :
  d / (v1 + v2) * 60 = time_in_minutes :=
by
  sorry

end projectiles_meet_in_90_minutes_l157_15754


namespace min_value_proof_l157_15798

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a^2 + 5 * a + 2) * (b^2 + 5 * b + 2) * (c^2 + 5 * c + 2) / (a * b * c)

theorem min_value_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c ≥ 343 :=
sorry

end min_value_proof_l157_15798


namespace miles_run_by_harriet_l157_15781

def miles_run_by_all_runners := 285
def miles_run_by_katarina := 51
def miles_run_by_adriana := 74
def miles_run_by_tomas_tyler_harriet (total_run: ℝ) := (total_run - (miles_run_by_katarina + miles_run_by_adriana))

theorem miles_run_by_harriet : (miles_run_by_tomas_tyler_harriet miles_run_by_all_runners) / 3 = 53.33 := by
  sorry

end miles_run_by_harriet_l157_15781


namespace infinite_coprime_pairs_with_divisibility_l157_15743

theorem infinite_coprime_pairs_with_divisibility :
  ∃ (A : ℕ → ℕ) (B : ℕ → ℕ), (∀ n, gcd (A n) (B n) = 1) ∧
    ∀ n, (A n ∣ (B n)^2 - 5) ∧ (B n ∣ (A n)^2 - 5) :=
sorry

end infinite_coprime_pairs_with_divisibility_l157_15743


namespace solve_for_k_l157_15797

theorem solve_for_k : {k : ℕ | ∀ x : ℝ, (x^2 - 1)^(2*k) + (x^2 + 2*x)^(2*k) + (2*x + 1)^(2*k) = 2*(1 + x + x^2)^(2*k)} = {1, 2} :=
sorry

end solve_for_k_l157_15797


namespace inequality_solution_l157_15706

theorem inequality_solution (a c : ℝ) (h : ∀ x : ℝ, (1/3 < x ∧ x < 1/2) ↔ ax^2 + 5*x + c > 0) : a + c = -7 :=
sorry

end inequality_solution_l157_15706


namespace values_of_a_l157_15709

axiom exists_rat : (x y a : ℚ) → Prop

theorem values_of_a (a : ℚ) (h1 : ∀ x y : ℚ, (x/2 - (2*x - 3*y)/5 = a - 1)) (h2 : ∀ x y : ℚ, (x + 3 = y/3)) :
  0.7 < a ∧ a < 6.4 ↔ (∃ x y : ℚ, x < 0 ∧ y > 0) :=
by
  sorry

end values_of_a_l157_15709


namespace choose_blue_pair_l157_15722

/-- In a drawer, there are 12 distinguishable socks: 5 white, 3 brown, and 4 blue socks.
    Prove that the number of ways to choose a pair of socks such that both socks are blue is 6. -/
theorem choose_blue_pair (total_socks white_socks brown_socks blue_socks : ℕ)
  (h_total : total_socks = 12) (h_white : white_socks = 5) (h_brown : brown_socks = 3) (h_blue : blue_socks = 4) :
  (blue_socks.choose 2) = 6 :=
by
  sorry

end choose_blue_pair_l157_15722


namespace not_necessarily_divisor_sixty_four_l157_15736

theorem not_necessarily_divisor_sixty_four (k : ℤ) (h : (k * (k + 1) * (k + 2)) % 8 = 0) :
  ¬ ((k * (k + 1) * (k + 2)) % 64 = 0) := 
sorry

end not_necessarily_divisor_sixty_four_l157_15736


namespace product_of_fractions_l157_15769

theorem product_of_fractions :
  (1 / 5) * (3 / 7) = 3 / 35 :=
sorry

end product_of_fractions_l157_15769


namespace range_of_a_l157_15758

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 4^x - (a + 3) * 2^x + 1 = 0) → a ≥ -1 := sorry

end range_of_a_l157_15758


namespace fraction_unchanged_l157_15750

-- Define the digit rotation
def rotate (d : ℕ) : ℕ :=
  match d with
  | 0 => 0
  | 1 => 1
  | 6 => 9
  | 8 => 8
  | 9 => 6
  | _ => d  -- for completeness, though we assume d only takes {0, 1, 6, 8, 9}

-- Define the condition for a fraction to be unchanged when flipped
def unchanged_when_flipped (numerator denominator : ℕ) : Prop :=
  let rotated_numerator := rotate numerator
  let rotated_denominator := rotate denominator
  rotated_numerator * denominator = rotated_denominator * numerator

-- Define the specific fraction 6/9
def specific_fraction_6_9 : Prop :=
  unchanged_when_flipped 6 9 ∧ 6 < 9

-- Theorem stating 6/9 is unchanged when its digits are flipped and it's a valid fraction
theorem fraction_unchanged : specific_fraction_6_9 :=
by
  sorry

end fraction_unchanged_l157_15750


namespace part1_tangent_line_at_x2_part2_inequality_l157_15778

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x + Real.exp 2 - 7

theorem part1_tangent_line_at_x2 (a : ℝ) (h_a : a = 2) :
  ∃ m b : ℝ, (∀ x : ℝ, f x a = m * x + b) ∧ m = Real.exp 2 - 2 ∧ b = -(2 * Real.exp 2 - 7) := by
  sorry

theorem part2_inequality (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x a ≥ (7 / 4) * x^2) → a ≤ Real.exp 2 - 7 := by
  sorry

end part1_tangent_line_at_x2_part2_inequality_l157_15778


namespace SugarWeightLoss_l157_15730

noncomputable def sugar_fraction_lost : Prop :=
  let green_beans_weight := 60
  let rice_weight := green_beans_weight - 30
  let sugar_weight := green_beans_weight - 10
  let rice_lost := (1 / 3) * rice_weight
  let remaining_weight := 120
  let total_initial_weight := green_beans_weight + rice_weight + sugar_weight
  let total_lost := total_initial_weight - remaining_weight
  let sugar_lost := total_lost - rice_lost
  let expected_fraction := (sugar_lost / sugar_weight)
  expected_fraction = (1 / 5)

theorem SugarWeightLoss : sugar_fraction_lost := by
  sorry

end SugarWeightLoss_l157_15730


namespace factorize_quadratic_trinomial_l157_15718

theorem factorize_quadratic_trinomial (t : ℝ) : t^2 - 10 * t + 25 = (t - 5)^2 :=
by
  sorry

end factorize_quadratic_trinomial_l157_15718


namespace mandy_total_shirts_l157_15711

-- Condition definitions
def black_packs : ℕ := 6
def black_shirts_per_pack : ℕ := 7
def yellow_packs : ℕ := 8
def yellow_shirts_per_pack : ℕ := 4

theorem mandy_total_shirts : 
  (black_packs * black_shirts_per_pack + yellow_packs * yellow_shirts_per_pack) = 74 :=
by
  sorry

end mandy_total_shirts_l157_15711


namespace cube_root_of_5_irrational_l157_15746

theorem cube_root_of_5_irrational : ¬ ∃ (a b : ℚ), (b ≠ 0) ∧ (a / b)^3 = 5 := 
by
  sorry

end cube_root_of_5_irrational_l157_15746


namespace Rajesh_Spend_Salary_on_Food_l157_15739

theorem Rajesh_Spend_Salary_on_Food
    (monthly_salary : ℝ)
    (percentage_medicines : ℝ)
    (savings_percentage : ℝ)
    (savings : ℝ) :
    monthly_salary = 15000 ∧
    percentage_medicines = 0.20 ∧
    savings_percentage = 0.60 ∧
    savings = 4320 →
    (32 : ℝ) = ((monthly_salary * percentage_medicines + monthly_salary * (1 - (percentage_medicines + savings_percentage))) / monthly_salary) * 100 :=
by
  sorry

end Rajesh_Spend_Salary_on_Food_l157_15739


namespace ratio_of_common_differences_l157_15720

variable (a b d1 d2 : ℝ)

theorem ratio_of_common_differences
  (h1 : a + 4 * d1 = b)
  (h2 : a + 5 * d2 = b) :
  d1 / d2 = 5 / 4 := 
by
  sorry

end ratio_of_common_differences_l157_15720


namespace age_of_oldest_child_l157_15737

theorem age_of_oldest_child
  (a b c d : ℕ)
  (h1 : a = 6)
  (h2 : b = 8)
  (h3 : c = 10)
  (h4 : (a + b + c + d) / 4 = 9) :
  d = 12 :=
sorry

end age_of_oldest_child_l157_15737


namespace solve_inequality_l157_15738

noncomputable def inequality_solution : Set ℝ :=
  { x | x^2 / (x + 2) ≥ 3 / (x - 2) + 7 / 4 }

theorem solve_inequality :
  inequality_solution = { x | -2 < x ∧ x < 2 } ∪ { x | 3 ≤ x } :=
by
  sorry

end solve_inequality_l157_15738


namespace revenue_difference_l157_15753

theorem revenue_difference {x z : ℕ} (hx : 10 ≤ x ∧ x ≤ 96) (hz : z = x + 3) :
  1000 * z + 10 * x - (1000 * x + 10 * z) = 2920 :=
by
  sorry

end revenue_difference_l157_15753


namespace ratio_of_sides_product_of_areas_and_segments_l157_15780

variable (S S' S'' : ℝ) (a a' : ℝ)

-- Given condition
axiom proportion_condition : S / S'' = a / a'

-- Proofs that need to be verified
theorem ratio_of_sides (S S' : ℝ) (a a' : ℝ) (h : S / S'' = a / a') :
  S / a = S' / a' :=
sorry

theorem product_of_areas_and_segments (S S' : ℝ) (a a' : ℝ) (h: S / S'' = a / a') :
  S * a' = S' * a :=
sorry

end ratio_of_sides_product_of_areas_and_segments_l157_15780


namespace number_of_children_l157_15782

-- Definition of the conditions
def pencils_per_child : ℕ := 2
def total_pencils : ℕ := 30

-- Theorem statement
theorem number_of_children (n : ℕ) (h1 : pencils_per_child = 2) (h2 : total_pencils = 30) :
  n = total_pencils / pencils_per_child :=
by
  have h : n = 30 / 2 := sorry
  exact h

end number_of_children_l157_15782


namespace smallest_number_of_set_s_l157_15745

theorem smallest_number_of_set_s : 
  ∀ (s : Set ℕ),
    (∃ n : ℕ, s = {k | ∃ m : ℕ, k = 5 * (m+n) ∧ m < 45}) ∧ 
    (275 ∈ s) → 
      (∃ min_elem : ℕ, min_elem ∈ s ∧ min_elem = 55) 
  :=
by
  sorry

end smallest_number_of_set_s_l157_15745


namespace intersection_of_lines_l157_15775

theorem intersection_of_lines : ∃ (x y : ℚ), y = -3 * x + 1 ∧ y + 5 = 15 * x - 2 ∧ x = 1 / 3 ∧ y = 0 :=
by
  sorry

end intersection_of_lines_l157_15775


namespace parents_without_full_time_jobs_l157_15786

theorem parents_without_full_time_jobs
  {total_parents mothers fathers : ℕ}
  (h_total_parents : total_parents = 100)
  (h_mothers_percentage : mothers = 60)
  (h_fathers_percentage : fathers = 40)
  (h_mothers_full_time : ℕ)
  (h_fathers_full_time : ℕ)
  (h_mothers_ratio : h_mothers_full_time = (5 * mothers) / 6)
  (h_fathers_ratio : h_fathers_full_time = (3 * fathers) / 4) :
  ((total_parents - (h_mothers_full_time + h_fathers_full_time)) * 100 / total_parents = 20) := sorry

end parents_without_full_time_jobs_l157_15786


namespace student_arrangement_l157_15710

theorem student_arrangement :
  let total_arrangements := 720
  let ab_violations := 240
  let cd_violations := 240
  let ab_cd_overlap := 96
  let restricted_arrangements := ab_violations + cd_violations - ab_cd_overlap
  let valid_arrangements := total_arrangements - restricted_arrangements
  valid_arrangements = 336 :=
by
  let total_arrangements := 720
  let ab_violations := 240
  let cd_violations := 240
  let ab_cd_overlap := 96
  let restricted_arrangements := ab_violations + cd_violations - ab_cd_overlap
  let valid_arrangements := total_arrangements - restricted_arrangements
  exact sorry

end student_arrangement_l157_15710


namespace exists_sequence_satisfying_conditions_l157_15787

def F : ℕ → ℕ := sorry

theorem exists_sequence_satisfying_conditions :
  (∀ n, ∃ k, F k = n) ∧ 
  (∀ n, ∃ m > n, F m = n) ∧ 
  (∀ n ≥ 2, F (F (n ^ 163)) = F (F n) + F (F 361)) :=
sorry

end exists_sequence_satisfying_conditions_l157_15787


namespace complex_cube_root_identity_l157_15756

theorem complex_cube_root_identity (a b c : ℂ) (ω : ℂ)
  (h1 : ω^3 = 1)
  (h2 : 1 + ω + ω^2 = 0) :
  (a + b * ω + c * ω^2) * (a + b * ω^2 + c * ω) = a^2 + b^2 + c^2 - ab - ac - bc :=
by
  sorry

end complex_cube_root_identity_l157_15756


namespace Robert_ate_10_chocolates_l157_15785

def chocolates_eaten_by_Nickel : Nat := 5
def difference_between_Robert_and_Nickel : Nat := 5
def chocolates_eaten_by_Robert := chocolates_eaten_by_Nickel + difference_between_Robert_and_Nickel

theorem Robert_ate_10_chocolates : chocolates_eaten_by_Robert = 10 :=
by
  -- Proof omitted
  sorry

end Robert_ate_10_chocolates_l157_15785


namespace total_children_on_bus_after_stop_l157_15742

theorem total_children_on_bus_after_stop (initial : ℕ) (additional : ℕ) (total : ℕ) 
  (h1 : initial = 18) (h2 : additional = 7) : total = 25 :=
by sorry

end total_children_on_bus_after_stop_l157_15742


namespace division_problem_l157_15772

theorem division_problem : 240 / (12 + 14 * 2) = 6 := by
  sorry

end division_problem_l157_15772


namespace tuning_day_method_pi_l157_15777

variable (x : ℝ)

-- Initial bounds and approximations
def initial_bounds (π : ℝ) := 31 / 10 < π ∧ π < 49 / 15

-- Definition of the "Tuning Day Method"
def tuning_day_method (a b c d : ℕ) (a' b' : ℝ) := a' = (b + d) / (a + c)

theorem tuning_day_method_pi :
  ∀ π : ℝ, initial_bounds π →
  (31 / 10 < π ∧ π < 16 / 5) ∧ 
  (47 / 15 < π ∧ π < 63 / 20) ∧
  (47 / 15 < π ∧ π < 22 / 7) →
  22 / 7 = 22 / 7 :=
by
  sorry

end tuning_day_method_pi_l157_15777


namespace books_bought_l157_15765

theorem books_bought (math_price : ℕ) (hist_price : ℕ) (total_cost : ℕ) (math_books : ℕ) (hist_books : ℕ) 
  (H : math_price = 4) (H1 : hist_price = 5) (H2 : total_cost = 396) (H3 : math_books = 54) 
  (H4 : math_books * math_price + hist_books * hist_price = total_cost) :
  math_books + hist_books = 90 :=
by sorry

end books_bought_l157_15765


namespace fitted_bowling_ball_volume_l157_15789

theorem fitted_bowling_ball_volume :
  let r_bowl := 20 -- radius of the bowling ball in cm
  let r_hole1 := 1 -- radius of the first hole in cm
  let r_hole2 := 2 -- radius of the second hole in cm
  let r_hole3 := 2 -- radius of the third hole in cm
  let depth := 10 -- depth of each hole in cm
  let V_bowl := (4/3) * Real.pi * r_bowl^3
  let V_hole1 := Real.pi * r_hole1^2 * depth
  let V_hole2 := Real.pi * r_hole2^2 * depth
  let V_hole3 := Real.pi * r_hole3^2 * depth
  let V_holes := V_hole1 + V_hole2 + V_hole3
  let V_fitted := V_bowl - V_holes
  V_fitted = (31710 / 3) * Real.pi :=
by sorry

end fitted_bowling_ball_volume_l157_15789


namespace measure_of_unknown_angle_in_hexagon_l157_15759

theorem measure_of_unknown_angle_in_hexagon :
  let a1 := 135
  let a2 := 105
  let a3 := 87
  let a4 := 120
  let a5 := 78
  let total_internal_angles := 180 * (6 - 2)
  let known_sum := a1 + a2 + a3 + a4 + a5
  let Q := total_internal_angles - known_sum
  Q = 195 :=
by
  sorry

end measure_of_unknown_angle_in_hexagon_l157_15759


namespace number_of_arrangements_is_48_l157_15768

noncomputable def number_of_arrangements (students : List String) (boy_not_at_ends : String) (adjacent_girls : List String) : Nat :=
  sorry

theorem number_of_arrangements_is_48 : number_of_arrangements ["A", "B1", "B2", "G1", "G2", "G3"] "B1" ["G1", "G2", "G3"] = 48 :=
by
  sorry

end number_of_arrangements_is_48_l157_15768


namespace equivalent_problem_l157_15774

def f (x : ℤ) : ℤ := 9 - x

def g (x : ℤ) : ℤ := x - 9

theorem equivalent_problem : g (f 15) = -15 := sorry

end equivalent_problem_l157_15774


namespace kanul_total_amount_l157_15717

theorem kanul_total_amount (T : ℝ) (R : ℝ) (M : ℝ) (C : ℝ)
  (hR : R = 80000)
  (hM : M = 30000)
  (hC : C = 0.2 * T)
  (hT : T = R + M + C) : T = 137500 :=
by {
  sorry
}

end kanul_total_amount_l157_15717


namespace count_zero_expressions_l157_15734

/-- Given four specific vector expressions, prove that exactly two of them evaluate to the zero vector. --/
theorem count_zero_expressions
(AB BC CA MB BO OM AC BD CD OA OC CO : ℝ × ℝ)
(H1 : AB + BC + CA = 0)
(H2 : AB + (MB + BO + OM) ≠ 0)
(H3 : AB - AC + BD - CD = 0)
(H4 : OA + OC + BO + CO ≠ 0) :
  (∃ count, count = 2 ∧
      ((AB + BC + CA = 0) → count = count + 1) ∧
      ((AB + (MB + BO + OM) = 0) → count = count + 1) ∧
      ((AB - AC + BD - CD = 0) → count = count + 1) ∧
      ((OA + OC + BO + CO = 0) → count = count + 1)) :=
sorry

end count_zero_expressions_l157_15734


namespace arithmetic_sequence_common_difference_l157_15760

theorem arithmetic_sequence_common_difference
  (a_n : ℕ → ℤ) (h_arithmetic : ∀ n, (a_n (n + 1) = a_n n + d)) 
  (h_sum1 : a_n 1 + a_n 3 + a_n 5 = 105)
  (h_sum2 : a_n 2 + a_n 4 + a_n 6 = 99) : 
  d = -2 :=
by
  sorry

end arithmetic_sequence_common_difference_l157_15760


namespace Peter_finishes_all_tasks_at_5_30_PM_l157_15799

-- Definitions representing the initial conditions
def start_time : ℕ := 9 * 60 -- 9:00 AM in minutes
def third_task_completion_time : ℕ := 11 * 60 + 30 -- 11:30 AM in minutes
def task_durations : List ℕ :=
  [30, 30, 60, 120, 240] -- Durations of the 5 tasks in minutes
  
-- Statement for the proof problem
theorem Peter_finishes_all_tasks_at_5_30_PM :
  let total_duration := task_durations.sum 
  let finish_time := start_time + total_duration
  finish_time = 17 * 60 + 30 := -- 5:30 PM in minutes
  sorry

end Peter_finishes_all_tasks_at_5_30_PM_l157_15799


namespace wrongly_written_height_is_176_l157_15712

-- Definitions and given conditions
def average_height_incorrect := 182
def average_height_correct := 180
def num_boys := 35
def actual_height := 106

-- The difference in total height due to the error
def total_height_incorrect := num_boys * average_height_incorrect
def total_height_correct := num_boys * average_height_correct
def height_difference := total_height_incorrect - total_height_correct

-- The wrongly written height
def wrongly_written_height := actual_height + height_difference

-- Proof statement
theorem wrongly_written_height_is_176 : wrongly_written_height = 176 := by
  sorry

end wrongly_written_height_is_176_l157_15712


namespace weekly_cost_l157_15755

def cost_per_hour : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 7
def number_of_bodyguards : ℕ := 2

theorem weekly_cost :
  (cost_per_hour * hours_per_day * number_of_bodyguards * days_per_week) = 2240 := by
  sorry

end weekly_cost_l157_15755


namespace polar_to_rectangular_l157_15748

theorem polar_to_rectangular :
  ∀ (r θ : ℝ), r = 3 * Real.sqrt 2 → θ = (3 * Real.pi) / 4 → 
  (r * Real.cos θ, r * Real.sin θ) = (-3, 3) :=
by
  intro r θ hr hθ
  rw [hr, hθ]
  sorry

end polar_to_rectangular_l157_15748


namespace find_f_2015_l157_15763

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_2015
  (h1 : ∀ x, f (-x) = -f x) -- f is an odd function
  (h2 : ∀ x, f (x + 2) = -f x) -- f(x+2) = -f(x)
  (h3 : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) -- f(x) = 2x^2 for x in (0, 2)
  : f 2015 = -2 :=
sorry

end find_f_2015_l157_15763


namespace remainder_of_3x_minus_2y_mod_30_l157_15762

theorem remainder_of_3x_minus_2y_mod_30
  (p q : ℤ) (x y : ℤ)
  (hx : x = 60 * p + 53)
  (hy : y = 45 * q + 28) :
  (3 * x - 2 * y) % 30 = 13 :=
by 
  sorry

end remainder_of_3x_minus_2y_mod_30_l157_15762


namespace responses_needed_750_l157_15776

section Responses
  variable (q_min : ℕ) (response_rate : ℝ)

  def responses_needed : ℝ := response_rate * q_min

  theorem responses_needed_750 (h1 : q_min = 1250) (h2 : response_rate = 0.60) : responses_needed q_min response_rate = 750 :=
  by
    simp [responses_needed, h1, h2]
    sorry
end Responses

end responses_needed_750_l157_15776


namespace basketball_game_half_points_l157_15788

noncomputable def eagles_geometric_sequence (a r : ℕ) (n : ℕ) : ℕ :=
  a * r ^ n

noncomputable def lions_arithmetic_sequence (b d : ℕ) (n : ℕ) : ℕ :=
  b + n * d

noncomputable def total_first_half_points (a r b d : ℕ) : ℕ :=
  eagles_geometric_sequence a r 0 + eagles_geometric_sequence a r 1 +
  lions_arithmetic_sequence b d 0 + lions_arithmetic_sequence b d 1

theorem basketball_game_half_points (a r b d : ℕ) (h1 : a + a * r = b + (b + d)) (h2 : a + a * r + a * r^2 + a * r^3 = b + (b + d) + (b + 2*d) + (b + 3*d)) :
  total_first_half_points a r b d = 8 :=
by sorry

end basketball_game_half_points_l157_15788


namespace part_1_part_2_l157_15708

variables (a b c : ℝ) (A B C : ℝ)
variable (triangle_ABC : a = b ∧ b = c ∧ A + B + C = 180 ∧ A = 90 ∨ B = 90 ∨ C = 90)
variable (sin_condition : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C)

theorem part_1 (h : a = b) : Real.cos C = 7 / 8 :=
by { sorry }

theorem part_2 (h₁ : B = 90) (h₂ : a = Real.sqrt 2) : b = 2 :=
by { sorry }

end part_1_part_2_l157_15708


namespace range_f_pos_l157_15757

noncomputable def f : ℝ → ℝ := sorry
axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom increasing_f : ∀ x y : ℝ, x < y → x ≤ 0 → y ≤ 0 → f x ≤ f y
axiom f_at_neg_one : f (-1) = 0

theorem range_f_pos : {x : ℝ | f x > 0} = Set.Ioo (-1) 1 := 
by
  sorry

end range_f_pos_l157_15757


namespace floor_S_proof_l157_15771

noncomputable def floor_S (a b c d: ℝ) : ℝ :=
⌊a + b + c + d⌋

theorem floor_S_proof (a b c d : ℝ)
  (h1 : a ^ 2 + 2 * b ^ 2 = 2016)
  (h2 : c ^ 2 + 2 * d ^ 2 = 2016)
  (h3 : a * c = 1024)
  (h4 : b * d = 1024)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : floor_S a b c d = 129 := 
sorry

end floor_S_proof_l157_15771


namespace amount_brought_by_sisters_l157_15725

-- Definitions based on conditions
def cost_per_ticket : ℕ := 8
def number_of_tickets : ℕ := 2
def change_received : ℕ := 9

-- Statement to prove
theorem amount_brought_by_sisters :
  (cost_per_ticket * number_of_tickets + change_received) = 25 :=
by
  -- Using assumptions directly
  let total_cost := cost_per_ticket * number_of_tickets
  have total_cost_eq : total_cost = 16 := by sorry
  let amount_brought := total_cost + change_received
  have amount_brought_eq : amount_brought = 25 := by sorry
  exact amount_brought_eq

end amount_brought_by_sisters_l157_15725


namespace func_g_neither_even_nor_odd_l157_15713

noncomputable def func_g (x : ℝ) : ℝ := (⌈x⌉ : ℝ) - (1 / 3)

theorem func_g_neither_even_nor_odd :
  (¬ ∀ x, func_g (-x) = func_g x) ∧ (¬ ∀ x, func_g (-x) = -func_g x) :=
by
  sorry

end func_g_neither_even_nor_odd_l157_15713


namespace valid_digit_distribution_l157_15715

theorem valid_digit_distribution (n : ℕ) : 
  (∃ (d1 d2 d5 others : ℕ), 
    d1 = n / 2 ∧
    d2 = n / 5 ∧
    d5 = n / 5 ∧
    others = n / 10 ∧
    d1 + d2 + d5 + others = n) :=
by
  sorry

end valid_digit_distribution_l157_15715


namespace arccos_cos_three_l157_15751

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := 
  sorry

end arccos_cos_three_l157_15751


namespace color_set_no_arith_prog_same_color_l157_15796

def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 1987}

def colors : Fin 4 := sorry  -- Color indexing set (0, 1, 2, 3)

def valid_coloring (c : ℕ → Fin 4) : Prop :=
  ∀ (a d : ℕ) (h₁ : a ∈ M) (h₂ : d ≠ 0) (h₃ : ∀ k, a + k * d ∈ M ∧ k < 10), 
  ¬ ∀ k, c (a + k * d) = c a

theorem color_set_no_arith_prog_same_color :
  ∃ (c : ℕ → Fin 4), valid_coloring c :=
sorry

end color_set_no_arith_prog_same_color_l157_15796


namespace ordered_triple_unique_l157_15735

theorem ordered_triple_unique (a b c : ℝ) (h2 : a > 2) (h3 : b > 2) (h4 : c > 2)
    (h : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 49) :
    a = 7 ∧ b = 5 ∧ c = 3 :=
sorry

end ordered_triple_unique_l157_15735


namespace markup_percentage_l157_15794

-- Definitions coming from conditions
variables (C : ℝ) (M : ℝ) (S : ℝ)
-- Markup formula
def markup_formula : Prop := M = 0.10 * C
-- Selling price formula
def selling_price_formula : Prop := S = C + M

-- Given the conditions, we need to prove that the markup is 9.09% of the selling price
theorem markup_percentage (h1 : markup_formula C M) (h2 : selling_price_formula C M S) :
  (M / S) * 100 = 9.09 :=
sorry

end markup_percentage_l157_15794


namespace central_angle_of_sector_l157_15740

theorem central_angle_of_sector (R θ l : ℝ) (h1 : 2 * R + l = π * R) : θ = π - 2 := 
by
  sorry

end central_angle_of_sector_l157_15740


namespace proof_problem_l157_15767

variable {a b : ℝ}

theorem proof_problem (h₁ : a < b) (h₂ : b < 0) : (b/a) + (a/b) > 2 :=
by 
  sorry

end proof_problem_l157_15767


namespace negation_of_exists_gt_one_l157_15714

theorem negation_of_exists_gt_one : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
by 
  sorry

end negation_of_exists_gt_one_l157_15714


namespace p_is_necessary_not_sufficient_for_q_l157_15719

  variable (x : ℝ)

  def p := |x| ≤ 2
  def q := 0 ≤ x ∧ x ≤ 2

  theorem p_is_necessary_not_sufficient_for_q : (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬ q x) :=
  by
    sorry
  
end p_is_necessary_not_sufficient_for_q_l157_15719


namespace prime_square_mod_12_l157_15791

theorem prime_square_mod_12 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) : 
  (p ^ 2) % 12 = 1 :=
sorry

end prime_square_mod_12_l157_15791


namespace no_real_solution_l157_15747

theorem no_real_solution : ∀ x : ℝ, ¬ ((2*x - 3*x + 7)^2 + 4 = -|2*x|) :=
by
  intro x
  have h1 : (2*x - 3*x + 7)^2 + 4 ≥ 4 := by
    sorry
  have h2 : -|2*x| ≤ 0 := by
    sorry
  -- The main contradiction follows from comparing h1 and h2
  sorry

end no_real_solution_l157_15747


namespace cost_of_bananas_l157_15784

theorem cost_of_bananas (A B : ℝ) (n : ℝ) (Tcost: ℝ) (Acost: ℝ): 
  (A * n + B = Tcost) → (A * (1 / 2 * n) + B = Acost) → (Tcost = 7) → (Acost = 5) → B = 3 :=
by
  intros hTony hArnold hTcost hAcost
  sorry

end cost_of_bananas_l157_15784


namespace sin_30_eq_half_l157_15764

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l157_15764


namespace chromium_percentage_new_alloy_l157_15732

theorem chromium_percentage_new_alloy :
  let wA := 15
  let pA := 0.12
  let wB := 30
  let pB := 0.08
  let wC := 20
  let pC := 0.20
  let wD := 35
  let pD := 0.05
  let total_weight := wA + wB + wC + wD
  let total_chromium := (wA * pA) + (wB * pB) + (wC * pC) + (wD * pD)
  total_weight = 100 ∧ total_chromium = 9.95 → total_chromium / total_weight * 100 = 9.95 :=
by
  sorry

end chromium_percentage_new_alloy_l157_15732


namespace calculate_fg1_l157_15729

def f (x : ℝ) : ℝ := 4 - 3 * x
def g (x : ℝ) : ℝ := x^3 + 1

theorem calculate_fg1 : f (g 1) = -2 :=
by
  sorry

end calculate_fg1_l157_15729


namespace quadratic_real_root_m_l157_15744

theorem quadratic_real_root_m (m : ℝ) (h : 4 - 4 * m ≥ 0) : m = 0 ∨ m = 2 ∨ m = 4 ∨ m = 6 ↔ m = 0 :=
by
  sorry

end quadratic_real_root_m_l157_15744


namespace expand_expression_l157_15792

theorem expand_expression (x : ℝ) : 3 * (x - 6) * (x - 7) = 3 * x^2 - 39 * x + 126 := by
  sorry

end expand_expression_l157_15792


namespace min_value_expression_l157_15727

theorem min_value_expression (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : (a - b) * (b - c) * (c - a) = -16) : 
  ∃ x : ℝ, x = (1 / (a - b)) + (1 / (b - c)) - (1 / (c - a)) ∧ x = 5 / 4 :=
by
  sorry

end min_value_expression_l157_15727


namespace inequality_always_holds_l157_15703

theorem inequality_always_holds (m : ℝ) :
  (∀ x : ℝ, m * x ^ 2 - m * x - 1 < 0) → -4 < m ∧ m ≤ 0 :=
by
  sorry

end inequality_always_holds_l157_15703


namespace fire_fighting_max_saved_houses_l157_15761

noncomputable def max_houses_saved (n c : ℕ) : ℕ :=
  n^2 + c^2 - n * c - c

theorem fire_fighting_max_saved_houses (n c : ℕ) (h : c ≤ n / 2) :
    ∃ k, k = max_houses_saved n c :=
    sorry

end fire_fighting_max_saved_houses_l157_15761


namespace lisa_more_dresses_than_ana_l157_15707

theorem lisa_more_dresses_than_ana :
  ∀ (total_dresses ana_dresses : ℕ),
    total_dresses = 48 →
    ana_dresses = 15 →
    (total_dresses - ana_dresses) - ana_dresses = 18 :=
by
  intros total_dresses ana_dresses h1 h2
  sorry

end lisa_more_dresses_than_ana_l157_15707


namespace other_candidate_valid_votes_l157_15749

noncomputable def validVotes (totalVotes invalidPct : ℝ) : ℝ :=
  totalVotes * (1 - invalidPct)

noncomputable def otherCandidateVotes (validVotes oneCandidatePct : ℝ) : ℝ :=
  validVotes * (1 - oneCandidatePct)

theorem other_candidate_valid_votes :
  let totalVotes := 7500
  let invalidPct := 0.20
  let oneCandidatePct := 0.55
  validVotes totalVotes invalidPct = 6000 ∧
  otherCandidateVotes (validVotes totalVotes invalidPct) oneCandidatePct = 2700 :=
by
  sorry

end other_candidate_valid_votes_l157_15749


namespace proof_correct_judgments_l157_15795

def terms_are_like (t1 t2 : Expr) : Prop := sorry -- Define like terms
def is_polynomial (p : Expr) : Prop := sorry -- Define polynomial
def is_quadratic_trinomial (p : Expr) : Prop := sorry -- Define quadratic trinomial
def constant_term (p : Expr) : Expr := sorry -- Define extraction of constant term

theorem proof_correct_judgments :
  let t1 := (2 * Real.pi * (a ^ 2) * b)
  let t2 := ((1 / 3) * (a ^ 2) * b)
  let p1 := (5 * a + 4 * b - 1)
  let p2 := (x - 2 * x * y + y)
  let p3 := ((x + y) / 4)
  let p4 := (x / 2 + 1)
  let p5 := (a / 4)
  terms_are_like t1 t2 ∧ 
  constant_term p1 = 1 = False ∧
  is_quadratic_trinomial p2 ∧
  is_polynomial p3 ∧ is_polynomial p4 ∧ is_polynomial p5
  → ("①③④" = "C") :=
by
  sorry

end proof_correct_judgments_l157_15795


namespace area_increase_300_percent_l157_15704

noncomputable def percentage_increase_of_area (d : ℝ) : ℝ :=
  let d' := 2 * d
  let r := d / 2
  let r' := d' / 2
  let A := Real.pi * r^2
  let A' := Real.pi * (r')^2
  100 * (A' - A) / A

theorem area_increase_300_percent (d : ℝ) : percentage_increase_of_area d = 300 :=
by
  sorry

end area_increase_300_percent_l157_15704


namespace deductive_reasoning_l157_15728

theorem deductive_reasoning (
  deductive_reasoning_form : Prop
): ¬(deductive_reasoning_form → true → correct_conclusion) :=
by sorry

end deductive_reasoning_l157_15728


namespace rectangle_area_same_width_l157_15733

theorem rectangle_area_same_width
  (square_area : ℝ) (area_eq : square_area = 36)
  (rect_width_eq_side : ℝ → ℝ → Prop) (width_eq : ∀ s, rect_width_eq_side s s)
  (rect_length_eq_3_times_width : ℝ → ℝ → Prop) (length_eq : ∀ w, rect_length_eq_3_times_width w (3 * w)) :
  (∃ s l w, s = 6 ∧ w = s ∧ l = 3 * w ∧ square_area = s * s ∧ rect_width_eq_side w s ∧ rect_length_eq_3_times_width w l ∧ w * l = 108) :=
by {
  sorry
}

end rectangle_area_same_width_l157_15733


namespace add_in_base6_l157_15752

def add_base6 (a b : ℕ) : ℕ := (a + b) % 6 + (((a + b) / 6) * 10)

theorem add_in_base6 (x y : ℕ) (h1 : x = 5) (h2 : y = 23) : add_base6 x y = 32 :=
by
  rw [h1, h2]
  -- Explanation: here add_base6 interprets numbers as base 6 and then performs addition,
  -- taking care of the base conversion automatically. This avoids directly involving steps of the given solution.
  sorry

end add_in_base6_l157_15752


namespace claudia_groupings_l157_15724

-- Definition of combinations
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Given conditions
def candles_combinations : ℕ := combination 6 3
def flowers_combinations : ℕ := combination 15 12

-- Lean statement
theorem claudia_groupings : candles_combinations * flowers_combinations = 9100 :=
by
  sorry

end claudia_groupings_l157_15724


namespace sin_alpha_through_point_l157_15705

theorem sin_alpha_through_point (α : ℝ) (x y : ℝ) (h : x = -1 ∧ y = 2) (r : ℝ) (h_r : r = Real.sqrt (x^2 + y^2)) :
  Real.sin α = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end sin_alpha_through_point_l157_15705


namespace valentina_burger_length_l157_15700

-- Definitions and conditions
def share : ℕ := 6
def total_length (share : ℕ) : ℕ := 2 * share

-- Proof statement
theorem valentina_burger_length : total_length share = 12 := by
  sorry

end valentina_burger_length_l157_15700


namespace Angle_Not_Equivalent_l157_15702

theorem Angle_Not_Equivalent (θ : ℤ) : (θ = -750) → (680 % 360 ≠ θ % 360) :=
by
  intro h
  have h1 : 680 % 360 = 320 := by norm_num
  have h2 : -750 % 360 = -30 % 360 := by norm_num
  have h3 : -30 % 360 = 330 := by norm_num
  rw [h, h2, h3]
  sorry

end Angle_Not_Equivalent_l157_15702


namespace nina_money_l157_15731

variable (C : ℝ)

def original_widget_count : ℕ := 6
def new_widget_count : ℕ := 8
def price_reduction : ℝ := 1.5

theorem nina_money (h : original_widget_count * C = new_widget_count * (C - price_reduction)) :
  original_widget_count * C = 36 := by
  sorry

end nina_money_l157_15731


namespace compute_pounds_of_cotton_l157_15793

theorem compute_pounds_of_cotton (x : ℝ) :
  (5 * 30 + 10 * x = 640) → (x = 49) := by
  intro h
  sorry

end compute_pounds_of_cotton_l157_15793


namespace sequence_general_formula_l157_15723

theorem sequence_general_formula (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = n^2 - 2 * n + 2):
  (a 1 = 1) ∧ (∀ n, 1 < n → a n = S n - S (n - 1)) → 
  (∀ n, a n = if n = 1 then 1 else 2 * n - 3) :=
by
  intro h
  sorry

end sequence_general_formula_l157_15723


namespace fraction_red_after_tripling_l157_15741

-- Define the initial conditions
def initial_fraction_blue : ℚ := 4 / 7
def initial_fraction_red : ℚ := 1 - initial_fraction_blue
def triple_red_fraction (initial_red : ℚ) : ℚ := 3 * initial_red

-- Theorem statement
theorem fraction_red_after_tripling :
  let x := 1 -- Any number since it will cancel out
  let initial_red_marble := initial_fraction_red * x
  let total_marble := x
  let new_red_marble := triple_red_fraction initial_red_marble
  let new_total_marble := initial_fraction_blue * x + new_red_marble
  (new_red_marble / new_total_marble) = 9 / 13 :=
by
  sorry

end fraction_red_after_tripling_l157_15741


namespace neg_prop_p_l157_15773

theorem neg_prop_p :
  (¬ (∃ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end neg_prop_p_l157_15773


namespace weight_order_l157_15766

variable {P Q R S T : ℕ}

theorem weight_order
    (h1 : Q + S = 1200)
    (h2 : R + T = 2100)
    (h3 : Q + T = 800)
    (h4 : Q + R = 900)
    (h5 : P + T = 700)
    (hP : P < 1000)
    (hQ : Q < 1000)
    (hR : R < 1000)
    (hS : S < 1000)
    (hT : T < 1000) :
  S > R ∧ R > T ∧ T > Q ∧ Q > P :=
sorry

end weight_order_l157_15766


namespace problem1_problem2_l157_15790

-- For Problem (1)
theorem problem1 (x : ℝ) : 2 * x - 3 > x + 1 → x > 4 := 
by sorry

-- For Problem (2)
theorem problem2 (a b : ℝ) (h : a^2 + 3 * a * b = 5) : (a + b) * (a + 2 * b) - 2 * b^2 = 5 := 
by sorry

end problem1_problem2_l157_15790


namespace number_of_routes_l157_15783

structure RailwayStation :=
  (A B C D E F G H I J K L M : ℕ)

def initialize_station : RailwayStation :=
  ⟨1, 1, 1, 1, 2, 2, 3, 3, 3, 6, 9, 9, 18⟩

theorem number_of_routes (station : RailwayStation) : station.M = 18 :=
  by sorry

end number_of_routes_l157_15783


namespace Shelby_drive_time_in_rain_l157_15701

theorem Shelby_drive_time_in_rain (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 3) 
  (h3 : 40 * (3 - x) + 25 * x = 85) : x = 140 / 60 :=
  sorry

end Shelby_drive_time_in_rain_l157_15701
