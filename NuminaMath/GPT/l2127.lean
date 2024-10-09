import Mathlib

namespace problem_statement_l2127_212750

-- Define what it means for a number's tens and ones digits to have a sum of 13
def sum_of_tens_and_ones_equals (n : ℕ) (s : ℕ) : Prop :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit = s

-- State the theorem with the given conditions and correct answer
theorem problem_statement : sum_of_tens_and_ones_equals (6^11) 13 :=
sorry

end problem_statement_l2127_212750


namespace missed_questions_l2127_212796

theorem missed_questions (F M : ℕ) (h1 : M = 5 * F) (h2 : M + F = 216) : M = 180 :=
by
  sorry

end missed_questions_l2127_212796


namespace decrease_in_profit_when_one_loom_idles_l2127_212797

def num_looms : ℕ := 125
def total_sales_value : ℕ := 500000
def total_manufacturing_expenses : ℕ := 150000
def monthly_establishment_charges : ℕ := 75000
def sales_value_per_loom : ℕ := total_sales_value / num_looms
def manufacturing_expense_per_loom : ℕ := total_manufacturing_expenses / num_looms
def decrease_in_sales_value : ℕ := sales_value_per_loom
def decrease_in_manufacturing_expenses : ℕ := manufacturing_expense_per_loom
def net_decrease_in_profit : ℕ := decrease_in_sales_value - decrease_in_manufacturing_expenses

theorem decrease_in_profit_when_one_loom_idles : net_decrease_in_profit = 2800 := by
  sorry

end decrease_in_profit_when_one_loom_idles_l2127_212797


namespace sufficiency_not_necessity_l2127_212718

def l1 : Type := sorry
def l2 : Type := sorry

def skew_lines (l1 l2 : Type) : Prop := sorry
def do_not_intersect (l1 l2 : Type) : Prop := sorry

theorem sufficiency_not_necessity (p q : Prop) 
  (hp : p = skew_lines l1 l2)
  (hq : q = do_not_intersect l1 l2) :
  (p → q) ∧ ¬ (q → p) :=
by {
  sorry
}

end sufficiency_not_necessity_l2127_212718


namespace larger_cookie_sugar_l2127_212794

theorem larger_cookie_sugar :
  let initial_cookies := 40
  let initial_sugar_per_cookie := 1 / 8
  let total_sugar := initial_cookies * initial_sugar_per_cookie
  let larger_cookies := 25
  let sugar_per_larger_cookie := total_sugar / larger_cookies
  sugar_per_larger_cookie = 1 / 5 := by
sorry

end larger_cookie_sugar_l2127_212794


namespace function_cannot_be_decreasing_if_f1_lt_f2_l2127_212744

variable (f : ℝ → ℝ)

theorem function_cannot_be_decreasing_if_f1_lt_f2
  (h : f 1 < f 2) : ¬ (∀ x y, x < y → f y < f x) :=
by
  sorry

end function_cannot_be_decreasing_if_f1_lt_f2_l2127_212744


namespace product_of_two_numbers_l2127_212755

-- Define HCF function
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM function
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Define the conditions for the problem
def problem_conditions (x y : ℕ) : Prop :=
  HCF x y = 55 ∧ LCM x y = 1500

-- State the theorem that should be proven
theorem product_of_two_numbers (x y : ℕ) (h_conditions : problem_conditions x y) :
  x * y = 82500 :=
by
  sorry

end product_of_two_numbers_l2127_212755


namespace convert_to_dms_convert_to_decimal_degrees_l2127_212732

-- Problem 1: Conversion of 24.29 degrees to degrees, minutes, and seconds 
theorem convert_to_dms (d : ℝ) (h : d = 24.29) : 
  (∃ deg min sec, d = deg + min / 60 + sec / 3600 ∧ deg = 24 ∧ min = 17 ∧ sec = 24) :=
by
  sorry

-- Problem 2: Conversion of 36 degrees 40 minutes 30 seconds to decimal degrees
theorem convert_to_decimal_degrees (deg min sec : ℝ) (h : deg = 36 ∧ min = 40 ∧ sec = 30) : 
  (deg + min / 60 + sec / 3600) = 36.66 :=
by
  sorry

end convert_to_dms_convert_to_decimal_degrees_l2127_212732


namespace minimum_stool_height_l2127_212773

def ceiling_height : ℤ := 280
def alice_height : ℤ := 150
def reach : ℤ := alice_height + 30
def light_bulb_height : ℤ := ceiling_height - 15

theorem minimum_stool_height : 
  ∃ h : ℤ, reach + h = light_bulb_height ∧ h = 85 :=
by
  sorry

end minimum_stool_height_l2127_212773


namespace reflection_line_slope_intercept_l2127_212790

theorem reflection_line_slope_intercept (m b : ℝ) :
  let P1 := (2, 3)
  let P2 := (10, 7)
  let midpoint := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)
  midpoint = (6, 5) ∧
  ∃(m b : ℝ), 
    m = -2 ∧
    b = 17 ∧
    P2 = (2 * midpoint.1 - P1.1, 2 * midpoint.2 - P1.2)
→ m + b = 15 := by
  intros
  sorry

end reflection_line_slope_intercept_l2127_212790


namespace exists_Q_R_l2127_212799

noncomputable def P (x : ℚ) : ℚ := x^4 + x^3 + x^2 + x + 1

theorem exists_Q_R : ∃ (Q R : Polynomial ℚ), 
  (Q.degree > 0 ∧ R.degree > 0) ∧
  (∀ (y : ℚ), (Q.eval y) * (R.eval y) = P (5 * y^2)) :=
sorry

end exists_Q_R_l2127_212799


namespace percentage_of_600_equals_150_is_25_l2127_212760

theorem percentage_of_600_equals_150_is_25 : (150 / 600 * 100) = 25 := by
  sorry

end percentage_of_600_equals_150_is_25_l2127_212760


namespace _l2127_212716

noncomputable def angle_ACB_is_45_degrees (A B C D E F : Type) [LinearOrderedField A]
  (angle : A → A → A → A) (AB AC : A) (h1 : AB = 3 * AC)
  (BAE ACD : A) (h2 : BAE = ACD)
  (BCA : A) (h3 : BAE = 2 * BCA)
  (CF FE : A) (h4 : CF = FE)
  (is_isosceles : ∀ {X Y Z : Type} [LinearOrderedField X] (a b c : X), a = b → b = c → a = c)
  (triangle_sum : ∀ {X Y Z : Type} [LinearOrderedField X] (a b c : X), a + b + c = 180) :
  ∃ (angle_ACB : A), angle_ACB = 45 := 
by
  -- Here we assume we have the appropriate conditions from geometry
  -- Then you'd prove the theorem based on given hypotheses
  sorry

end _l2127_212716


namespace sum_solutions_eq_l2127_212765

theorem sum_solutions_eq : 
  let a := 12
  let b := -19
  let c := -21
  (4 * x + 3) * (3 * x - 7) = 0 → (b/a) = 19/12 :=
by
  sorry

end sum_solutions_eq_l2127_212765


namespace total_books_l2127_212787

theorem total_books (D Loris Lamont : ℕ) 
  (h1 : Loris + 3 = Lamont)
  (h2 : Lamont = 2 * D)
  (h3 : D = 20) : D + Loris + Lamont = 97 := 
by 
  sorry

end total_books_l2127_212787


namespace x_squared_minus_y_squared_l2127_212775

theorem x_squared_minus_y_squared {x y : ℚ} 
    (h1 : x + y = 3/8) 
    (h2 : x - y = 5/24) 
    : x^2 - y^2 = 5/64 := 
by 
    -- The proof would go here
    sorry

end x_squared_minus_y_squared_l2127_212775


namespace largest_share_received_l2127_212786

noncomputable def largest_share (total_profit : ℝ) (ratio : List ℝ) : ℝ :=
  let total_parts := ratio.foldl (· + ·) 0
  let part_value := total_profit / total_parts
  let max_part := ratio.foldl max 0
  max_part * part_value

theorem largest_share_received
  (total_profit : ℝ)
  (h_total_profit : total_profit = 42000)
  (ratio : List ℝ)
  (h_ratio : ratio = [2, 3, 4, 4, 6]) :
  largest_share total_profit ratio = 12600 :=
by
  sorry

end largest_share_received_l2127_212786


namespace find_e_l2127_212793

theorem find_e (b e : ℝ) (f g : ℝ → ℝ)
    (h1 : ∀ x, f x = 5 * x + b)
    (h2 : ∀ x, g x = b * x + 3)
    (h3 : ∀ x, f (g x) = 15 * x + e) : e = 18 :=
by
  sorry

end find_e_l2127_212793


namespace conditional_probability_l2127_212759

variables (A B : Prop)
variables (P : Prop → ℚ)
variables (h₁ : P A = 8 / 30) (h₂ : P (A ∧ B) = 7 / 30)

theorem conditional_probability : P (A → B) = 7 / 8 :=
by sorry

end conditional_probability_l2127_212759


namespace percent_time_in_meetings_l2127_212731

theorem percent_time_in_meetings
  (work_day_minutes : ℕ := 8 * 60)
  (first_meeting_minutes : ℕ := 30)
  (second_meeting_minutes : ℕ := 3 * 30) :
  (first_meeting_minutes + second_meeting_minutes) / work_day_minutes * 100 = 25 :=
by
  -- sorry to skip the actual proof
  sorry

end percent_time_in_meetings_l2127_212731


namespace circumscribed_sphere_radius_l2127_212768

theorem circumscribed_sphere_radius (a b c : ℝ) : 
  R = (1/2) * Real.sqrt (a^2 + b^2 + c^2) := sorry

end circumscribed_sphere_radius_l2127_212768


namespace remainingAreaCalculation_l2127_212788

noncomputable def totalArea : ℝ := 9500.0
noncomputable def lizzieGroupArea : ℝ := 2534.1
noncomputable def hilltownTeamArea : ℝ := 2675.95
noncomputable def greenValleyCrewArea : ℝ := 1847.57

theorem remainingAreaCalculation :
  (totalArea - (lizzieGroupArea + hilltownTeamArea + greenValleyCrewArea) = 2442.38) :=
by
  sorry

end remainingAreaCalculation_l2127_212788


namespace bus_driver_limit_of_hours_l2127_212725

theorem bus_driver_limit_of_hours (r o T H L : ℝ)
  (h_reg_rate : r = 16)
  (h_ot_rate : o = 1.75 * r)
  (h_total_comp : T = 752)
  (h_hours_worked : H = 44)
  (h_equation : r * L + o * (H - L) = T) :
  L = 40 :=
  sorry

end bus_driver_limit_of_hours_l2127_212725


namespace equation_D_has_two_distinct_real_roots_l2127_212743

def quadratic_has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem equation_D_has_two_distinct_real_roots : quadratic_has_two_distinct_real_roots 1 2 (-1) :=
by {
  sorry
}

end equation_D_has_two_distinct_real_roots_l2127_212743


namespace radius_of_smaller_base_of_truncated_cone_l2127_212722

theorem radius_of_smaller_base_of_truncated_cone 
  (r1 r2 r3 : ℕ) (touching : 2 * r1 = r2 ∧ r1 + r3 = r2 * 2):
  (∀ (R : ℕ), R = 6) :=
sorry

end radius_of_smaller_base_of_truncated_cone_l2127_212722


namespace average_of_three_quantities_l2127_212798

theorem average_of_three_quantities (a b c d e : ℝ) 
  (h_avg_5 : (a + b + c + d + e) / 5 = 11)
  (h_avg_2 : (d + e) / 2 = 21.5) :
  (a + b + c) / 3 = 4 :=
by
  sorry

end average_of_three_quantities_l2127_212798


namespace smallest_distance_l2127_212754

noncomputable def a : Complex := 2 + 4 * Complex.I
noncomputable def b : Complex := 5 + 2 * Complex.I

theorem smallest_distance 
  (z w : Complex) 
  (hz : Complex.abs (z - a) = 2) 
  (hw : Complex.abs (w - b) = 4) : 
  Complex.abs (z - w) ≥ 6 - Real.sqrt 13 :=
sorry

end smallest_distance_l2127_212754


namespace find_a7_coefficient_l2127_212727

theorem find_a7_coefficient (a_7 : ℤ) : 
    (∀ x : ℤ, (x+1)^5 * (2*x-1)^3 = a_8 * x^8 + a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) → a_7 = 28 :=
by
  sorry

end find_a7_coefficient_l2127_212727


namespace find_k_values_l2127_212738

/-- 
Prove that the values of k such that the positive difference between the 
roots of 3x^2 + 5x + k = 0 equals the sum of the squares of the roots 
are exactly (70 + 10sqrt(33))/8 and (70 - 10sqrt(33))/8.
-/
theorem find_k_values (k : ℝ) :
  (∀ (a b : ℝ), (3 * a^2 + 5 * a + k = 0 ∧ 3 * b^2 + 5 * b + k = 0 ∧ |a - b| = a^2 + b^2))
  ↔ (k = (70 + 10 * Real.sqrt 33) / 8 ∨ k = (70 - 10 * Real.sqrt 33) / 8) :=
sorry

end find_k_values_l2127_212738


namespace speed_of_ship_with_two_sails_l2127_212715

noncomputable def nautical_mile : ℝ := 1.15
noncomputable def land_miles_traveled : ℝ := 345
noncomputable def time_with_one_sail : ℝ := 4
noncomputable def time_with_two_sails : ℝ := 4
noncomputable def speed_with_one_sail : ℝ := 25

theorem speed_of_ship_with_two_sails :
  ∃ S : ℝ, 
    (S * time_with_two_sails + speed_with_one_sail * time_with_one_sail = land_miles_traveled / nautical_mile) → 
    S = 50  :=
by
  sorry

end speed_of_ship_with_two_sails_l2127_212715


namespace num_of_laborers_is_24_l2127_212756

def average_salary_all (L S : Nat) (avg_salary_ls : Nat) (avg_salary_l : Nat) (avg_salary_s : Nat) : Prop :=
  (avg_salary_l * L + avg_salary_s * S) / (L + S) = avg_salary_ls

def average_salary_supervisors (S : Nat) (avg_salary_s : Nat) : Prop :=
  (avg_salary_s * S) / S = avg_salary_s

theorem num_of_laborers_is_24 :
  ∀ (L S : Nat) (avg_salary_ls avg_salary_l avg_salary_s : Nat),
    average_salary_all L S avg_salary_ls avg_salary_l avg_salary_s →
    average_salary_supervisors S avg_salary_s →
    S = 6 → avg_salary_ls = 1250 → avg_salary_l = 950 → avg_salary_s = 2450 →
    L = 24 :=
by
  intros L S avg_salary_ls avg_salary_l avg_salary_s h1 h2 h3 h4 h5 h6
  sorry

end num_of_laborers_is_24_l2127_212756


namespace base_six_four_digit_odd_final_l2127_212730

theorem base_six_four_digit_odd_final :
  ∃ b : ℕ, (b^4 > 285 ∧ 285 ≥ b^3 ∧ (285 % b) % 2 = 1) :=
by 
  use 6
  sorry

end base_six_four_digit_odd_final_l2127_212730


namespace abc_not_8_l2127_212709

theorem abc_not_8 (a b c : ℕ) (h : 2^a * 3^b * 4^c = 192) : a + b + c ≠ 8 :=
sorry

end abc_not_8_l2127_212709


namespace augmented_wedge_volume_proof_l2127_212774

open Real

noncomputable def sphere_radius (circumference : ℝ) : ℝ :=
  circumference / (2 * π)

noncomputable def sphere_volume (r : ℝ) : ℝ :=
  (4/3) * π * r^3

noncomputable def wedge_volume (volume_sphere : ℝ) (number_of_wedges : ℕ) : ℝ :=
  volume_sphere / number_of_wedges

noncomputable def augmented_wedge_volume (original_wedge_volume : ℝ) : ℝ :=
  2 * original_wedge_volume

theorem augmented_wedge_volume_proof (circumference : ℝ) (number_of_wedges : ℕ) 
  (volume : ℝ) (augmented_volume : ℝ) :
  circumference = 18 * π →
  number_of_wedges = 6 →
  volume = sphere_volume (sphere_radius circumference) →
  augmented_volume = augmented_wedge_volume (wedge_volume volume number_of_wedges) →
  augmented_volume = 324 * π :=
by
  intros h_circ h_wedges h_vol h_aug_vol
  -- This is where the proof steps would go
  sorry

end augmented_wedge_volume_proof_l2127_212774


namespace range_of_z_l2127_212757

theorem range_of_z (x y : ℝ) (h1 : -4 ≤ x - y ∧ x - y ≤ -1) (h2 : -1 ≤ 4 * x - y ∧ 4 * x - y ≤ 5) :
  ∃ (z : ℝ), z = 9 * x - y ∧ -1 ≤ z ∧ z ≤ 20 :=
sorry

end range_of_z_l2127_212757


namespace equation_of_l2_l2127_212741

-- Define the initial line equation
def l1 (x : ℝ) : ℝ := -2 * x - 2

-- Define the transformed line equation after translation
def l2 (x : ℝ) : ℝ := l1 (x + 1) + 2

-- Statement to prove
theorem equation_of_l2 : ∀ x, l2 x = -2 * x - 2 := by
  sorry

end equation_of_l2_l2127_212741


namespace probability_larry_wins_l2127_212724

noncomputable def P_larry_wins_game : ℝ :=
  let p_hit := (1 : ℝ) / 3
  let p_miss := (2 : ℝ) / 3
  let r := p_miss^3
  (p_hit / (1 - r))

theorem probability_larry_wins :
  P_larry_wins_game = 9 / 19 :=
by
  -- Proof is omitted, but the outline and logic are given in the problem statement
  sorry

end probability_larry_wins_l2127_212724


namespace geometric_series_ratio_half_l2127_212736

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l2127_212736


namespace sam_initial_nickels_l2127_212726

variable (n_now n_given n_initial : Nat)

theorem sam_initial_nickels (h_now : n_now = 63) (h_given : n_given = 39) (h_relation : n_now = n_initial + n_given) : n_initial = 24 :=
by
  sorry

end sam_initial_nickels_l2127_212726


namespace abs_diff_of_two_numbers_l2127_212749

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 34) (h2 : x * y = 240) : abs (x - y) = 14 :=
by
  sorry

end abs_diff_of_two_numbers_l2127_212749


namespace sequence_98th_term_l2127_212785

-- Definitions of the rules
def rule1 (n : ℕ) : ℕ := n * 9
def rule2 (n : ℕ) : ℕ := n / 2
def rule3 (n : ℕ) : ℕ := n - 5

-- Function to compute the next term in the sequence based on the current term
def next_term (n : ℕ) : ℕ :=
  if n < 10 then rule1 n
  else if n % 2 = 0 then rule2 n
  else rule3 n

-- Function to compute the nth term of the sequence starting with the initial term
def nth_term (start : ℕ) (n : ℕ) : ℕ :=
  Nat.iterate next_term n start

-- Theorem to prove that the 98th term of the sequence starting at 98 is 27
theorem sequence_98th_term : nth_term 98 98 = 27 := by
  sorry

end sequence_98th_term_l2127_212785


namespace emily_total_spent_l2127_212763

def total_cost (art_supplies_cost skirt_cost : ℕ) (number_of_skirts : ℕ) : ℕ :=
  art_supplies_cost + (skirt_cost * number_of_skirts)

theorem emily_total_spent :
  total_cost 20 15 2 = 50 :=
by
  sorry

end emily_total_spent_l2127_212763


namespace max_cylinder_volume_in_cone_l2127_212781

theorem max_cylinder_volume_in_cone :
  ∃ x, (0 < x ∧ x < 1) ∧ ∀ y, (0 < y ∧ y < 1 → y ≠ x → ((π * (-2 * y^3 + 2 * y^2)) ≤ (π * (-2 * x^3 + 2 * x^2)))) ∧ 
  (π * (-2 * x^3 + 2 * x^2) = 8 * π / 27) := sorry

end max_cylinder_volume_in_cone_l2127_212781


namespace number_of_aluminum_atoms_l2127_212735

def molecular_weight (n : ℕ) : ℝ :=
  n * 26.98 + 30.97 + 4 * 16.0

theorem number_of_aluminum_atoms (n : ℕ) (h : molecular_weight n = 122) : n = 1 :=
by
  sorry

end number_of_aluminum_atoms_l2127_212735


namespace train_speed_l2127_212746

theorem train_speed (length1 length2 speed2 : ℝ) (time_seconds speed1 : ℝ)
    (h_length1 : length1 = 111)
    (h_length2 : length2 = 165)
    (h_speed2 : speed2 = 90)
    (h_time : time_seconds = 6.623470122390208)
    (h_speed1 : speed1 = 60) :
    (length1 / 1000.0) + (length2 / 1000.0) / (time_seconds / 3600) = speed1 + speed2 :=
by
  sorry

end train_speed_l2127_212746


namespace max_possible_value_l2127_212712

-- Define the expressions and the conditions
def expr1 := 10 * 10
def expr2 := 10 / 10
def expr3 := expr1 + 10
def expr4 := expr3 - expr2

-- Define our main statement that asserts the maximum value is 109
theorem max_possible_value: expr4 = 109 := by
  sorry

end max_possible_value_l2127_212712


namespace correct_statement_3_l2127_212703

-- Definitions
def acute_angles (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def angles_less_than_90 (θ : ℝ) : Prop := θ < 90
def angles_in_first_quadrant (θ : ℝ) : Prop := ∃ k : ℤ, k * 360 < θ ∧ θ < k * 360 + 90

-- Sets
def M := {θ | acute_angles θ}
def N := {θ | angles_less_than_90 θ}
def P := {θ | angles_in_first_quadrant θ}

-- Proof statement
theorem correct_statement_3 : M ⊆ P := sorry

end correct_statement_3_l2127_212703


namespace factorization_of_expression_l2127_212704

theorem factorization_of_expression (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) := by
  sorry

end factorization_of_expression_l2127_212704


namespace digits_in_8_20_3_30_base_12_l2127_212702

def digits_in_base (n b : ℕ) : ℕ :=
  if n = 0 then 1 else 1 + Nat.log b n

theorem digits_in_8_20_3_30_base_12 : digits_in_base (8^20 * 3^30) 12 = 31 :=
by
  sorry

end digits_in_8_20_3_30_base_12_l2127_212702


namespace least_trees_l2127_212772

theorem least_trees (N : ℕ) (h1 : N % 7 = 0) (h2 : N % 6 = 0) (h3 : N % 4 = 0) (h4 : N ≥ 100) : N = 168 :=
sorry

end least_trees_l2127_212772


namespace regular_polygon_perimeter_l2127_212791

def exterior_angle (n : ℕ) := 360 / n

theorem regular_polygon_perimeter
  (side_length : ℕ)
  (exterior_angle_deg : ℕ)
  (polygon_perimeter : ℕ)
  (h1 : side_length = 8)
  (h2 : exterior_angle_deg = 72)
  (h3 : ∃ n : ℕ, exterior_angle n = exterior_angle_deg)
  (h4 : ∀ n : ℕ, exterior_angle n = exterior_angle_deg → polygon_perimeter = n * side_length) :
  polygon_perimeter = 40 :=
sorry

end regular_polygon_perimeter_l2127_212791


namespace next_leap_year_visible_after_2017_l2127_212713

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0) ∧ ((y % 100 ≠ 0) ∨ (y % 400 = 0))

def stromquist_visible (start_year interval next_leap : ℕ) : Prop :=
  ∃ k : ℕ, next_leap = start_year + k * interval ∧ is_leap_year next_leap

theorem next_leap_year_visible_after_2017 :
  stromquist_visible 2017 61 2444 :=
  sorry

end next_leap_year_visible_after_2017_l2127_212713


namespace incorrect_statement_implies_m_eq_zero_l2127_212705

theorem incorrect_statement_implies_m_eq_zero
  (m : ℝ)
  (y : ℝ → ℝ)
  (h : ∀ x, y x = m * x + 4 * m - 2)
  (intersects_y_axis_at : y 0 = -2) :
  m = 0 :=
sorry

end incorrect_statement_implies_m_eq_zero_l2127_212705


namespace equilateral_triangle_ratio_correct_l2127_212723

noncomputable def equilateral_triangle_ratio : ℝ :=
  let side_length := 12
  let perimeter := 3 * side_length
  let altitude := side_length * (Real.sqrt 3 / 2)
  let area := (side_length * altitude) / 2
  area / perimeter

theorem equilateral_triangle_ratio_correct :
  equilateral_triangle_ratio = Real.sqrt 3 :=
sorry

end equilateral_triangle_ratio_correct_l2127_212723


namespace area_increase_factor_l2127_212748

theorem area_increase_factor (s : ℝ) :
  let A_original := s^2
  let A_new := (3 * s)^2
  A_new / A_original = 9 := by
  sorry

end area_increase_factor_l2127_212748


namespace height_of_fourth_person_l2127_212779

theorem height_of_fourth_person
  (h : ℝ)
  (cond : (h + (h + 2) + (h + 4) + (h + 10)) / 4 = 79) :
  (h + 10) = 85 :=
by 
  sorry

end height_of_fourth_person_l2127_212779


namespace find_t_l2127_212778

theorem find_t : ∀ (p j t x y a b c : ℝ),
  j = 0.75 * p →
  j = 0.80 * t →
  t = p - (t/100) * p →
  x = 0.10 * t →
  y = 0.50 * j →
  x + y = 12 →
  a = x + y →
  b = 0.15 * a →
  c = 2 * b →
  t = 24 := 
by
  intros p j t x y a b c hjp hjt htp hxt hyy hxy ha hb hc
  sorry

end find_t_l2127_212778


namespace no_integer_solutions_l2127_212720

theorem no_integer_solutions (x y z : ℤ) : ¬ (x^2 + y^2 = 3 * z^2) :=
sorry

end no_integer_solutions_l2127_212720


namespace water_remaining_l2127_212721

variable (initial_amount : ℝ) (leaked_amount : ℝ)

theorem water_remaining (h1 : initial_amount = 0.75)
                       (h2 : leaked_amount = 0.25) :
  initial_amount - leaked_amount = 0.50 :=
by
  sorry

end water_remaining_l2127_212721


namespace sam_age_two_years_ago_l2127_212711

variables (S J : ℕ)
variables (h1 : J = 3 * S) (h2 : J + 9 = 2 * (S + 9))

theorem sam_age_two_years_ago : S - 2 = 7 := by
  sorry

end sam_age_two_years_ago_l2127_212711


namespace part1_part2_l2127_212769

noncomputable def x : ℝ := 1 - Real.sqrt 2
noncomputable def y : ℝ := 1 + Real.sqrt 2

theorem part1 : x^2 + 3 * x * y + y^2 = 3 := by
  sorry

theorem part2 : (y / x) - (x / y) = -4 * Real.sqrt 2 := by
  sorry

end part1_part2_l2127_212769


namespace monotonic_decreasing_interval_l2127_212742

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < (1 / 2) → (0 < x ∧ x < (1 / 2)) ∧ (f (1 / 2) - f x) > 0 :=
sorry

end monotonic_decreasing_interval_l2127_212742


namespace youtube_dislikes_l2127_212717

theorem youtube_dislikes (x y : ℕ) 
  (h1 : x = 3 * y) 
  (h2 : x = 100 + 2 * y) 
  (h_y_increased : ∃ y' : ℕ, y' = 3 * y) :
  y' = 300 := by
  sorry

end youtube_dislikes_l2127_212717


namespace LCM_of_fractions_l2127_212761

theorem LCM_of_fractions (x : ℕ) (h : x > 0) : 
  lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (9 * x))) = 1 / (36 * x) :=
by
  sorry

end LCM_of_fractions_l2127_212761


namespace indefinite_integral_solution_l2127_212728

open Real

theorem indefinite_integral_solution (c : ℝ) : 
  ∫ x, (1 - cos x) / (x - sin x) ^ 2 = - 1 / (x - sin x) + c := 
sorry

end indefinite_integral_solution_l2127_212728


namespace find_z_l2127_212777

theorem find_z (x y z : ℝ) 
  (h1 : y = 2 * x + 3) 
  (h2 : x + 1 / x = 3.5 + (Real.sin (z * Real.exp (-z)))) :
  z = x^2 + 1 / x^2 := 
sorry

end find_z_l2127_212777


namespace students_more_than_rabbits_l2127_212700

-- Definitions of conditions
def classrooms : ℕ := 5
def students_per_classroom : ℕ := 22
def rabbits_per_classroom : ℕ := 2

-- Statement of the theorem
theorem students_more_than_rabbits :
  classrooms * students_per_classroom - classrooms * rabbits_per_classroom = 100 := 
  by
    sorry

end students_more_than_rabbits_l2127_212700


namespace cade_marbles_left_l2127_212752

theorem cade_marbles_left (initial_marbles : ℕ) (given_away : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 350 → given_away = 175 → remaining_marbles = initial_marbles - given_away → remaining_marbles = 175 :=
by
  intros h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining

end cade_marbles_left_l2127_212752


namespace cost_of_socks_l2127_212714

theorem cost_of_socks (x : ℝ) : 
  let initial_amount := 20
  let hat_cost := 7 
  let final_amount := 5
  let socks_pairs := 4
  let remaining_amount := initial_amount - hat_cost
  remaining_amount - socks_pairs * x = final_amount 
  -> x = 2 := 
by 
  sorry

end cost_of_socks_l2127_212714


namespace students_in_both_clubs_l2127_212751

theorem students_in_both_clubs (total_students drama_club art_club drama_or_art in_both_clubs : ℕ)
  (H1 : total_students = 300)
  (H2 : drama_club = 120)
  (H3 : art_club = 150)
  (H4 : drama_or_art = 220) :
  in_both_clubs = drama_club + art_club - drama_or_art :=
by
  -- this is the proof space
  sorry

end students_in_both_clubs_l2127_212751


namespace balloons_kept_by_Andrew_l2127_212747

theorem balloons_kept_by_Andrew :
  let blue := 303
  let purple := 453
  let red := 165
  let yellow := 324
  let blue_kept := (2/3 : ℚ) * blue
  let purple_kept := (3/5 : ℚ) * purple
  let red_kept := (4/7 : ℚ) * red
  let yellow_kept := (1/3 : ℚ) * yellow
  let total_kept := blue_kept.floor + purple_kept.floor + red_kept.floor + yellow_kept
  total_kept = 675 := by
  sorry

end balloons_kept_by_Andrew_l2127_212747


namespace university_diploma_percentage_l2127_212739

-- Define the conditions
variables (P N JD ND : ℝ)
-- P: total population assumed as 100% for simplicity
-- N: percentage of people with university diploma
-- JD: percentage of people who have the job of their choice
-- ND: percentage of people who do not have a university diploma but have the job of their choice
variables (A : ℝ) -- A: University diploma percentage of those who do not have the job of their choice
variable (total_diploma : ℝ)
axiom country_Z_conditions : 
  (P = 100) ∧ (ND = 18) ∧ (JD = 40) ∧ (A = 25)

-- Define the proof problem
theorem university_diploma_percentage :
  (N = ND + (JD - ND) + (total_diploma * (P - JD * (P / JD) / P))) →
  N = 37 :=
by
  sorry

end university_diploma_percentage_l2127_212739


namespace tangent_line_eq_at_x_is_1_range_of_sum_extreme_values_l2127_212745

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * Real.log x - x
noncomputable def g (x m : ℝ) : ℝ := f x + m * x^2
noncomputable def tangentLineEq (x y : ℝ) : Prop := x + 2 * y + 1 = 0
noncomputable def rangeCondition (x₁ x₂ m : ℝ) : Prop := g x₁ m + g x₂ m < -3 / 2

theorem tangent_line_eq_at_x_is_1 :
  tangentLineEq 1 (f 1) := 
sorry

theorem range_of_sum_extreme_values (h : 0 < m ∧ m < 1 / 4) (x₁ x₂ : ℝ) :
  rangeCondition x₁ x₂ m := 
sorry

end tangent_line_eq_at_x_is_1_range_of_sum_extreme_values_l2127_212745


namespace solution_set_of_inequality_l2127_212758

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 ≥ 0 } = {x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l2127_212758


namespace price_difference_correct_l2127_212771

-- Define the list price of Camera Y
def list_price : ℚ := 52.50

-- Define the discount at Mega Deals
def mega_deals_discount : ℚ := 12

-- Define the discount rate at Budget Buys
def budget_buys_discount_rate : ℚ := 0.30

-- Calculate the sale prices
def mega_deals_price : ℚ := list_price - mega_deals_discount
def budget_buys_price : ℚ := (1 - budget_buys_discount_rate) * list_price

-- Calculate the price difference in dollars and convert to cents
def price_difference_in_cents : ℚ := (mega_deals_price - budget_buys_price) * 100

-- Theorem to prove the computed price difference in cents equals 375
theorem price_difference_correct : price_difference_in_cents = 375 := by
  sorry

end price_difference_correct_l2127_212771


namespace sum_divided_among_xyz_l2127_212766

noncomputable def total_amount (x_share y_share z_share : ℝ) : ℝ :=
  x_share + y_share + z_share

theorem sum_divided_among_xyz
    (x_share : ℝ) (y_share : ℝ) (z_share : ℝ)
    (y_gets_45_paisa : y_share = 0.45 * x_share)
    (z_gets_50_paisa : z_share = 0.50 * x_share)
    (y_share_is_18 : y_share = 18) :
    total_amount x_share y_share z_share = 78 := by
  sorry

end sum_divided_among_xyz_l2127_212766


namespace largest_value_x_y_l2127_212737

theorem largest_value_x_y (x y : ℝ) (h1 : 5 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) : x + y ≤ 11 / 4 :=
sorry

end largest_value_x_y_l2127_212737


namespace distance_midpoint_chord_AB_to_y_axis_l2127_212753

theorem distance_midpoint_chord_AB_to_y_axis
  (k : ℝ)
  (A B : ℝ × ℝ)
  (hA : A.2 = k * A.1 - k)
  (hB : B.2 = k * B.1 - k)
  (hA_on_parabola : A.2 ^ 2 = 4 * A.1)
  (hB_on_parabola : B.2 ^ 2 = 4 * B.1)
  (h_distance_AB : dist A B = 4) :
  (abs ((A.1 + B.1) / 2)) = 1 :=
by
  sorry

end distance_midpoint_chord_AB_to_y_axis_l2127_212753


namespace integer_count_between_sqrt8_and_sqrt78_l2127_212770

theorem integer_count_between_sqrt8_and_sqrt78 :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℤ), (⌈Real.sqrt 8⌉ ≤ x ∧ x ≤ ⌊Real.sqrt 78⌋) ↔ (3 ≤ x ∧ x ≤ 8) := by
  sorry

end integer_count_between_sqrt8_and_sqrt78_l2127_212770


namespace values_of_z_l2127_212764

theorem values_of_z (x z : ℝ) 
  (h1 : 3 * x^2 + 9 * x + 7 * z + 2 = 0)
  (h2 : 3 * x + z + 4 = 0) : 
  z^2 + 20 * z - 14 = 0 := 
sorry

end values_of_z_l2127_212764


namespace min_chips_to_color_all_cells_l2127_212708

def min_chips_needed (n : ℕ) : ℕ := n

theorem min_chips_to_color_all_cells (n : ℕ) :
  min_chips_needed n = n :=
sorry

end min_chips_to_color_all_cells_l2127_212708


namespace points_earned_l2127_212729

-- Define the given conditions
def points_per_enemy := 5
def total_enemies := 8
def enemies_remaining := 6

-- Calculate the number of enemies defeated
def enemies_defeated := total_enemies - enemies_remaining

-- Calculate the points earned based on the enemies defeated
theorem points_earned : enemies_defeated * points_per_enemy = 10 := by
  -- Insert mathematical operations
  sorry

end points_earned_l2127_212729


namespace leftmost_digit_base9_l2127_212780

theorem leftmost_digit_base9 (x : ℕ) (h : x = 3^19 + 2*3^18 + 1*3^17 + 1*3^16 + 2*3^15 + 2*3^14 + 1*3^13 + 1*3^12 + 1*3^11 + 2*3^10 + 2*3^9 + 2*3^8 + 1*3^7 + 1*3^6 + 1*3^5 + 1*3^4 + 2*3^3 + 2*3^2 + 2*3^1 + 2) : ℕ :=
by
  sorry

end leftmost_digit_base9_l2127_212780


namespace number_of_diet_soda_l2127_212701

variable (d r : ℕ)

-- Define the conditions of the problem
def condition1 : Prop := r = d + 79
def condition2 : Prop := r = 83

-- State the theorem we want to prove
theorem number_of_diet_soda (h1 : condition1 d r) (h2 : condition2 r) : d = 4 :=
by
  sorry

end number_of_diet_soda_l2127_212701


namespace arith_to_geom_l2127_212783

noncomputable def a (n : ℕ) (d : ℝ) : ℝ := 1 + (n - 1) * d

theorem arith_to_geom (m n : ℕ) (d : ℝ) 
  (h_pos : d > 0)
  (h_arith_seq : ∀ k : ℕ, a k d > 0)
  (h_geo_seq : (a 4 d + 5 / 2)^2 = (a 3 d) * (a 11 d))
  (h_mn : m - n = 8) : 
  a m d - a n d = 12 := 
sorry

end arith_to_geom_l2127_212783


namespace expression_value_l2127_212734

theorem expression_value :
  ( (120^2 - 13^2) / (90^2 - 19^2) * ((90 - 19) * (90 + 19)) / ((120 - 13) * (120 + 13)) ) = 1 := by
  sorry

end expression_value_l2127_212734


namespace circumcircle_eqn_l2127_212719

variables (D E F : ℝ)

def point_A := (4, 0)
def point_B := (0, 3)
def point_C := (0, 0)

-- Define the system of equations for the circumcircle
def system : Prop :=
  (16 + 4*D + F = 0) ∧
  (9 + 3*E + F = 0) ∧
  (F = 0)

theorem circumcircle_eqn : system D E F → (D = -4 ∧ E = -3 ∧ F = 0) :=
sorry -- Proof omitted

end circumcircle_eqn_l2127_212719


namespace cost_price_of_table_l2127_212762

theorem cost_price_of_table (SP : ℝ) (CP : ℝ) (h1 : SP = 1.20 * CP) (h2 : SP = 3600) : CP = 3000 :=
by
  sorry

end cost_price_of_table_l2127_212762


namespace unique_solution_l2127_212795

def s (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem unique_solution (m n : ℕ) (h : n * (n + 1) = 3 ^ m + s n + 1182) : (m, n) = (0, 34) :=
by
  sorry

end unique_solution_l2127_212795


namespace bike_cost_l2127_212767

theorem bike_cost (h1: 8 > 0) (h2: 35 > 0) (weeks_in_month: ℕ := 4) (saved: ℕ := 720):
  let hourly_wage := 8
  let weekly_hours := 35
  let weekly_earnings := weekly_hours * hourly_wage
  let monthly_earnings := weekly_earnings * weeks_in_month
  let cost_of_bike := monthly_earnings - saved
  cost_of_bike = 400 :=
by
  sorry

end bike_cost_l2127_212767


namespace inequality_proof_l2127_212776

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
    a * (b^2 + c^2) + b * (c^2 + a^2) ≥ 4 * a * b * c :=
by
  sorry

end inequality_proof_l2127_212776


namespace walking_total_distance_l2127_212707

theorem walking_total_distance :
  let t1 := 1    -- first hour on level ground
  let t2 := 0.5  -- next 0.5 hour on level ground
  let t3 := 0.75 -- 45 minutes uphill
  let t4 := 0.5  -- 30 minutes uphill
  let t5 := 0.5  -- 30 minutes downhill
  let t6 := 0.25 -- 15 minutes downhill
  let t7 := 1.5  -- 1.5 hours on level ground
  let t8 := 0.75 -- 45 minutes on level ground
  let s1 := 4    -- speed for t1 (4 km/hr)
  let s2 := 5    -- speed for t2 (5 km/hr)
  let s3 := 3    -- speed for t3 (3 km/hr)
  let s4 := 2    -- speed for t4 (2 km/hr)
  let s5 := 6    -- speed for t5 (6 km/hr)
  let s6 := 7    -- speed for t6 (7 km/hr)
  let s7 := 4    -- speed for t7 (4 km/hr)
  let s8 := 6    -- speed for t8 (6 km/hr)
  s1 * t1 + s2 * t2 + s3 * t3 + s4 * t4 + s5 * t5 + s6 * t6 + s7 * t7 + s8 * t8 = 25 :=
by sorry

end walking_total_distance_l2127_212707


namespace line_through_P_with_equal_intercepts_line_through_A_with_inclination_90_l2127_212740

-- Definitions for the first condition
def P : ℝ × ℝ := (3, 2)
def passes_through_P (l : ℝ → ℝ) := l P.1 = P.2
def equal_intercepts (l : ℝ → ℝ) := ∃ a : ℝ, l a = 0 ∧ l (-a) = 0

-- Equation 1: Line passing through P with equal intercepts
theorem line_through_P_with_equal_intercepts :
  (∃ l : ℝ → ℝ, passes_through_P l ∧ equal_intercepts l ∧ 
   (∀ x y : ℝ, l x = y ↔ (2 * x - 3 * y = 0) ∨ (x + y - 5 = 0))) :=
sorry

-- Definitions for the second condition
def A : ℝ × ℝ := (-1, -3)
def passes_through_A (l : ℝ → ℝ) := l A.1 = A.2
def inclination_90 (l : ℝ → ℝ) := ∀ x : ℝ, l x = l 0

-- Equation 2: Line passing through A with inclination 90°
theorem line_through_A_with_inclination_90 :
  (∃ l : ℝ → ℝ, passes_through_A l ∧ inclination_90 l ∧ 
   (∀ x y : ℝ, l x = y ↔ (x + 1 = 0))) :=
sorry

end line_through_P_with_equal_intercepts_line_through_A_with_inclination_90_l2127_212740


namespace region_area_l2127_212792

-- Let x and y be real numbers
variables (x y : ℝ)

-- Define the inequality condition
def region_condition (x y : ℝ) : Prop := abs (4 * x - 20) + abs (3 * y + 9) ≤ 6

-- The statement that needs to be proved
theorem region_area : (∃ x y : ℝ, region_condition x y) → ∃ A : ℝ, A = 6 :=
by
  sorry

end region_area_l2127_212792


namespace sum_of_factors_636405_l2127_212784

theorem sum_of_factors_636405 :
  ∃ (a b c : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 10 ≤ c ∧ c < 100 ∧
    a * b * c = 636405 ∧ a + b + c = 259 :=
sorry

end sum_of_factors_636405_l2127_212784


namespace opposite_neg_two_is_two_l2127_212733

theorem opposite_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_neg_two_is_two_l2127_212733


namespace sum_of_squares_l2127_212710

theorem sum_of_squares (x y : ℤ) (h : ∃ k : ℤ, (x^2 + y^2) = 5 * k) : 
  ∃ a b : ℤ, (x^2 + y^2) / 5 = a^2 + b^2 :=
by sorry

end sum_of_squares_l2127_212710


namespace candidates_appeared_l2127_212789

theorem candidates_appeared (x : ℝ) (h1 : 0.07 * x = 0.06 * x + 82) : x = 8200 :=
by
  sorry

end candidates_appeared_l2127_212789


namespace find_z2_l2127_212782

theorem find_z2 (z1 z2 : ℂ) (h1 : z1 = 1 - I) (h2 : z1 * z2 = 1 + I) : z2 = I :=
sorry

end find_z2_l2127_212782


namespace distance_between_incenter_and_circumcenter_of_right_triangle_l2127_212706

theorem distance_between_incenter_and_circumcenter_of_right_triangle (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) (right_triangle : a^2 + b^2 = c^2) :
    ∃ (IO : ℝ), IO = Real.sqrt 5 :=
by
  rw [h1, h2, h3] at right_triangle
  have h_sum : 6^2 + 8^2 = 10^2 := by sorry
  exact ⟨Real.sqrt 5, by sorry⟩

end distance_between_incenter_and_circumcenter_of_right_triangle_l2127_212706
