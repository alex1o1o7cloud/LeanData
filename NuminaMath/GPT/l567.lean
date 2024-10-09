import Mathlib

namespace rhind_papyrus_problem_l567_56743

theorem rhind_papyrus_problem 
  (a1 a2 a3 a4 a5 : ℚ)
  (h1 : a2 = a1 + d)
  (h2 : a3 = a1 + 2 * d)
  (h3 : a4 = a1 + 3 * d)
  (h4 : a5 = a1 + 4 * d)
  (h_sum : a1 + a2 + a3 + a4 + a5 = 60)
  (h_condition : (a4 + a5) / 2 = a1 + a2 + a3) :
  a1 = 4 / 3 :=
by
  sorry

end rhind_papyrus_problem_l567_56743


namespace parity_implies_even_sum_l567_56750

theorem parity_implies_even_sum (n m : ℤ) (h : Even (n^2 + m^2 + n * m)) : ¬Odd (n + m) :=
sorry

end parity_implies_even_sum_l567_56750


namespace equal_profits_at_20000_end_month_more_profit_50000_l567_56756

noncomputable section

-- Define the conditions
def profit_beginning_month (x : ℝ) : ℝ := 0.15 * x + 1.15 * x * 0.1
def profit_end_month (x : ℝ) : ℝ := 0.3 * x - 700

-- Proof Problem 1: Prove that at x = 20000, the profits are equal
theorem equal_profits_at_20000 : profit_beginning_month 20000 = profit_end_month 20000 :=
by
  sorry

-- Proof Problem 2: Prove that at x = 50000, selling at end of month yields more profit than selling at beginning of month
theorem end_month_more_profit_50000 : profit_end_month 50000 > profit_beginning_month 50000 :=
by
  sorry

end equal_profits_at_20000_end_month_more_profit_50000_l567_56756


namespace tangent_line_eqn_extreme_values_l567_56777

/-- The tangent line to the function f at (0, 5) -/
theorem tangent_line_eqn (f : ℝ → ℝ) (h : ∀ x, f x = (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 - 2 * x + 5) :
  (∃ k b, (∀ x, f x = k * x + b) ∧ k = -2 ∧ b = 5) ∧ (2 * 0 + 5 - 5 = 0) := by
  sorry

/-- The function f has a local maximum at x = -1 and a local minimum at x = 2 -/
theorem extreme_values (f : ℝ → ℝ) (h : ∀ x, f x = (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 - 2 * x + 5) :
  (∃ x₁ x₂, x₁ = -1 ∧ f x₁ = 37 / 6 ∧ x₂ = 2 ∧ f x₂ = 5 / 3) := by
  sorry

end tangent_line_eqn_extreme_values_l567_56777


namespace English_family_information_l567_56776

-- Define the statements given by the family members.
variables (father_statement : Prop)
          (mother_statement : Prop)
          (daughter_statement : Prop)

-- Conditions provided in the problem
variables (going_to_Spain : Prop)
          (coming_from_Newcastle : Prop)
          (stopped_in_Paris : Prop)

-- Define what each family member said
axiom Father : father_statement ↔ (going_to_Spain ∨ coming_from_Newcastle)
axiom Mother : mother_statement ↔ ((¬going_to_Spain ∧ coming_from_Newcastle) ∨ (stopped_in_Paris ∧ ¬going_to_Spain))
axiom Daughter : daughter_statement ↔ (¬coming_from_Newcastle ∨ stopped_in_Paris)

-- The final theorem to be proved:
theorem English_family_information : (¬going_to_Spain ∧ coming_from_Newcastle ∧ stopped_in_Paris) :=
by
  -- steps to prove the theorem should go here, but they are skipped with sorry
  sorry

end English_family_information_l567_56776


namespace mia_socks_l567_56711

-- Defining the number of each type of socks
variables {a b c : ℕ}

-- Conditions and constraints
def total_pairs (a b c : ℕ) : Prop := a + b + c = 15
def total_cost (a b c : ℕ) : Prop := 2 * a + 3 * b + 5 * c = 35
def at_least_one (a b c : ℕ) : Prop := a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1

-- Main theorem to prove the number of 2-dollar pairs of socks
theorem mia_socks : 
  ∀ (a b c : ℕ), 
  total_pairs a b c → 
  total_cost a b c → 
  at_least_one a b c → 
  a = 12 :=
by
  sorry

end mia_socks_l567_56711


namespace simplify_expression_l567_56731

variable (y : ℝ)

theorem simplify_expression :
  4 * y^3 + 8 * y + 6 - (3 - 4 * y^3 - 8 * y) = 8 * y^3 + 16 * y + 3 :=
by
  sorry

end simplify_expression_l567_56731


namespace min_value_of_x2_plus_y2_min_value_of_reciprocal_sum_l567_56783

namespace MathProof

-- Definitions and conditions
variables {x y : ℝ}
axiom x_pos : x > 0
axiom y_pos : y > 0
axiom sum_eq_one : x + y = 1

-- Problem Statement 1: Prove the minimum value of x^2 + y^2 is 1/2
theorem min_value_of_x2_plus_y2 : ∃ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) ∧ (x^2 + y^2 = 1/2) :=
by
  sorry

-- Problem Statement 2: Prove the minimum value of 1/x + 1/y + 1/(xy) is 6
theorem min_value_of_reciprocal_sum : ∃ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) ∧ ((1/x + 1/y + 1/(x*y)) = 6) :=
by
  sorry

end MathProof

end min_value_of_x2_plus_y2_min_value_of_reciprocal_sum_l567_56783


namespace first_train_cross_time_l567_56763

noncomputable def length_first_train : ℝ := 800
noncomputable def speed_first_train_kmph : ℝ := 120
noncomputable def length_second_train : ℝ := 1000
noncomputable def speed_second_train_kmph : ℝ := 80
noncomputable def length_third_train : ℝ := 600
noncomputable def speed_third_train_kmph : ℝ := 150

noncomputable def speed_kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

noncomputable def speed_first_train_mps : ℝ := speed_kmph_to_mps speed_first_train_kmph
noncomputable def speed_second_train_mps : ℝ := speed_kmph_to_mps speed_second_train_kmph
noncomputable def speed_third_train_mps : ℝ := speed_kmph_to_mps speed_third_train_kmph

noncomputable def relative_speed_same_direction : ℝ := speed_first_train_mps - speed_second_train_mps
noncomputable def relative_speed_opposite_direction : ℝ := speed_first_train_mps + speed_third_train_mps

noncomputable def time_to_cross_second_train : ℝ := (length_first_train + length_second_train) / relative_speed_same_direction
noncomputable def time_to_cross_third_train : ℝ := (length_first_train + length_third_train) / relative_speed_opposite_direction

noncomputable def total_time_to_cross : ℝ := time_to_cross_second_train + time_to_cross_third_train

theorem first_train_cross_time : total_time_to_cross = 180.67 := by
  sorry

end first_train_cross_time_l567_56763


namespace area_of_figure_enclosed_by_curve_l567_56740

theorem area_of_figure_enclosed_by_curve (θ : ℝ) : 
  ∃ (A : ℝ), A = 4 * Real.pi ∧ (∀ θ, (4 * Real.cos θ)^2 = (4 * Real.cos θ) * 4 * Real.cos θ) :=
sorry

end area_of_figure_enclosed_by_curve_l567_56740


namespace weighted_average_salary_l567_56755

theorem weighted_average_salary :
  let num_managers := 9
  let salary_managers := 4500
  let num_associates := 18
  let salary_associates := 3500
  let num_lead_cashiers := 6
  let salary_lead_cashiers := 3000
  let num_sales_representatives := 45
  let salary_sales_representatives := 2500
  let total_salaries := 
    (num_managers * salary_managers) +
    (num_associates * salary_associates) +
    (num_lead_cashiers * salary_lead_cashiers) +
    (num_sales_representatives * salary_sales_representatives)
  let total_employees := 
    num_managers + num_associates + num_lead_cashiers + num_sales_representatives
  let weighted_avg_salary := total_salaries / total_employees
  weighted_avg_salary = 3000 := 
by
  sorry

end weighted_average_salary_l567_56755


namespace XY_sum_l567_56709

theorem XY_sum (A B C D X Y : ℕ) 
  (h1 : A + B + C + D = 22) 
  (h2 : X = A + B) 
  (h3 : Y = C + D) 
  : X + Y = 4 := 
  sorry

end XY_sum_l567_56709


namespace find_radius_of_circle_l567_56764

theorem find_radius_of_circle (C : ℝ) (h : C = 72 * Real.pi) : ∃ r : ℝ, 2 * Real.pi * r = C ∧ r = 36 :=
by
  sorry

end find_radius_of_circle_l567_56764


namespace range_of_a_l567_56708

-- Definition of sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B (a : ℝ) : Set ℝ := { x | x < a }

-- Condition of the union of A and B
theorem range_of_a (a : ℝ) : (A ∪ B a = { x | x < 1 }) ↔ -1 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l567_56708


namespace solve_for_x_l567_56789

-- Define the variables and conditions based on the problem statement
def equation (x : ℚ) := 5 * x - 3 * (x + 2) = 450 - 9 * (x - 4)

-- State the theorem to be proved, including the condition and the result
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = 44.72727272727273 := by
  sorry  -- The proof is omitted

end solve_for_x_l567_56789


namespace fraction_calls_by_team_B_l567_56716

-- Define the conditions
variables (A B C : ℝ)
axiom ratio_agents : A = (5 / 8) * B
axiom ratio_calls : ∀ (c : ℝ), c = (6 / 5) * C

-- Prove the fraction of the total calls processed by team B
theorem fraction_calls_by_team_B 
  (h1 : A = (5 / 8) * B)
  (h2 : ∀ (c : ℝ), c = (6 / 5) * C) :
  (B * C) / ((5 / 8) * B * (6 / 5) * C + B * C) = 4 / 7 :=
by {
  -- proof is omitted, so we use sorry
  sorry
}

end fraction_calls_by_team_B_l567_56716


namespace value_of_m_squared_plus_reciprocal_squared_l567_56744

theorem value_of_m_squared_plus_reciprocal_squared 
  (m : ℝ) 
  (h : m + 1/m = 10) :
  m^2 + 1/m^2 + 4 = 102 :=
by {
  sorry
}

end value_of_m_squared_plus_reciprocal_squared_l567_56744


namespace dice_sum_to_11_l567_56757

/-- Define the conditions for the outcomes of the dice rolls -/
def valid_outcomes (x : Fin 5 → ℕ) : Prop :=
  (∀ i, 1 ≤ x i ∧ x i ≤ 6) ∧ (x 0 + x 1 + x 2 + x 3 + x 4 = 11)

/-- Prove that there are exactly 205 ways to achieve a sum of 11 with five different colored dice -/
theorem dice_sum_to_11 : 
  (∃ (s : Finset (Fin 5 → ℕ)), (∀ x ∈ s, valid_outcomes x) ∧ s.card = 205) :=
  by
    sorry

end dice_sum_to_11_l567_56757


namespace number_of_students_l567_56738

theorem number_of_students (S : ℕ) (hS1 : S ≥ 2) (hS2 : S ≤ 80) 
                          (hO : ∀ n : ℕ, (n * S) % 120 = 0) : 
    S = 40 :=
sorry

end number_of_students_l567_56738


namespace steve_fraction_of_day_in_school_l567_56775

theorem steve_fraction_of_day_in_school :
  let total_hours : ℕ := 24
  let sleep_fraction : ℚ := 1 / 3
  let assignment_fraction : ℚ := 1 / 12
  let family_hours : ℕ := 10
  let sleep_hours : ℚ := sleep_fraction * total_hours
  let assignment_hours : ℚ := assignment_fraction * total_hours
  let accounted_hours : ℚ := sleep_hours + assignment_hours + family_hours
  let school_hours : ℚ := total_hours - accounted_hours
  (school_hours / total_hours) = (1 / 6) :=
by
  let total_hours : ℕ := 24
  let sleep_fraction : ℚ := 1 / 3
  let assignment_fraction : ℚ := 1 / 12
  let family_hours : ℕ := 10
  let sleep_hours : ℚ := sleep_fraction * total_hours
  let assignment_hours : ℚ := assignment_fraction * total_hours
  let accounted_hours : ℚ := sleep_hours + assignment_hours + family_hours
  let school_hours : ℚ := total_hours - accounted_hours
  have : (school_hours / total_hours) = (1 / 6) := sorry
  exact this

end steve_fraction_of_day_in_school_l567_56775


namespace third_side_not_twelve_l567_56741

theorem third_side_not_twelve (x : ℕ) (h1 : x > 5) (h2 : x < 11) (h3 : x % 2 = 0) : x ≠ 12 :=
by
  -- The proof is omitted
  sorry

end third_side_not_twelve_l567_56741


namespace cannot_fit_480_pictures_l567_56767

theorem cannot_fit_480_pictures 
  (A_capacity : ℕ) (B_capacity : ℕ) (C_capacity : ℕ) 
  (n_A : ℕ) (n_B : ℕ) (n_C : ℕ) 
  (total_pictures : ℕ) : 
  A_capacity = 12 → B_capacity = 18 → C_capacity = 24 → 
  n_A = 6 → n_B = 4 → n_C = 3 → 
  total_pictures = 480 → 
  A_capacity * n_A + B_capacity * n_B + C_capacity * n_C < total_pictures :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end cannot_fit_480_pictures_l567_56767


namespace original_selling_price_l567_56701

theorem original_selling_price (CP SP_original SP_loss : ℝ)
  (h1 : SP_original = CP * 1.25)
  (h2 : SP_loss = CP * 0.85)
  (h3 : SP_loss = 544) : SP_original = 800 :=
by
  -- The proof goes here, but we are skipping it with sorry
  sorry

end original_selling_price_l567_56701


namespace false_proposition_is_C_l567_56723

theorem false_proposition_is_C : ¬ (∀ x : ℝ, x^3 > 0) :=
sorry

end false_proposition_is_C_l567_56723


namespace parabola_equation_l567_56791

theorem parabola_equation (p : ℝ) (h : 0 < p) (Fₓ : ℝ) (Tₓ Tᵧ : ℝ) (Mₓ Mᵧ : ℝ)
  (eq_parabola : ∀ (y x : ℝ), y^2 = 2 * p * x → (y, x) = (Tᵧ, Tₓ))
  (F : (Fₓ, 0) = (p / 2, 0))
  (T_on_C : (Tᵧ, Tₓ) ∈ {(y, x) | y^2 = 2 * p * x})
  (FT_dist : dist (Fₓ, 0) (Tₓ, Tᵧ) = 5 / 2)
  (M : (Mₓ, Mᵧ) = (0, 1))
  (MF_MT_perp : ((Mᵧ - 0) / (Mₓ - Fₓ)) * ((Tᵧ - Mᵧ) / (Tₓ - Mᵧ)) = -1) :
  y^2 = 2 * x ∨ y^2 = 8 * x := 
sorry

end parabola_equation_l567_56791


namespace simplify_to_ellipse_l567_56728

theorem simplify_to_ellipse (x y : ℝ) :
  (Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10) →
  (x^2 / 25 + y^2 / 21 = 1) :=
by
  sorry

end simplify_to_ellipse_l567_56728


namespace sum_of_ages_in_10_years_l567_56737

-- Define the initial conditions about Ann's and Tom's ages
def AnnCurrentAge : ℕ := 6
def TomCurrentAge : ℕ := 2 * AnnCurrentAge

-- Define their ages 10 years later
def AnnAgeIn10Years : ℕ := AnnCurrentAge + 10
def TomAgeIn10Years : ℕ := TomCurrentAge + 10

-- The proof statement
theorem sum_of_ages_in_10_years : AnnAgeIn10Years + TomAgeIn10Years = 38 := by
  sorry

end sum_of_ages_in_10_years_l567_56737


namespace tangent_line_circle_l567_56782

theorem tangent_line_circle (a : ℝ) :
  (∀ (x y : ℝ), 4 * x - 3 * y = 0 → x^2 + y^2 - 2 * x + a * y + 1 = 0) →
  a = -1 ∨ a = 4 :=
sorry

end tangent_line_circle_l567_56782


namespace eval_expression_l567_56790

theorem eval_expression : (-1)^45 + 2^(3^2 + 5^2 - 4^2) = 262143 := by
  sorry

end eval_expression_l567_56790


namespace intersection_nonempty_implies_a_gt_neg1_l567_56720

def A := {x : ℝ | -1 ≤ x ∧ x < 2}
def B (a : ℝ) := {x : ℝ | x < a}

theorem intersection_nonempty_implies_a_gt_neg1 (a : ℝ) : (A ∩ B a).Nonempty → a > -1 :=
by
  sorry

end intersection_nonempty_implies_a_gt_neg1_l567_56720


namespace find_m_l567_56714

def U : Set ℕ := {1, 2, 3, 4}
def compl_U_A : Set ℕ := {1, 4}

theorem find_m (m : ℕ) (A : Set ℕ) (hA : A = {x | x ^ 2 - 5 * x + m = 0 ∧ x ∈ U}) :
  compl_U_A = U \ A → m = 6 :=
by
  sorry

end find_m_l567_56714


namespace pascal_current_speed_l567_56725

variable (v : ℝ)
variable (h₁ : v > 0) -- current speed is positive

-- Conditions
variable (h₂ : 96 / (v - 4) = 96 / (1.5 * v) + 16)

-- Proving the speed
theorem pascal_current_speed (h₁ : v > 0) (h₂ : 96 / (v - 4) = 96 / (1.5 * v) + 16) : v = 8 :=
sorry

end pascal_current_speed_l567_56725


namespace find_angle_BEC_l567_56796

-- Constants and assumptions
def angle_A : ℝ := 45
def angle_D : ℝ := 50
def angle_F : ℝ := 55
def E_above_C : Prop := true  -- This is a placeholder to represent the condition that E is directly above C.

-- Definition of the problem
theorem find_angle_BEC (angle_A_eq : angle_A = 45) 
                      (angle_D_eq : angle_D = 50) 
                      (angle_F_eq : angle_F = 55)
                      (triangle_BEC_formed : Prop)
                      (E_directly_above_C : E_above_C) 
                      : ∃ (BEC : ℝ), BEC = 10 :=
by sorry

end find_angle_BEC_l567_56796


namespace base_conversion_l567_56794

def baseThreeToBaseTen (n : List ℕ) : ℕ :=
  n.reverse.enumFrom 0 |>.map (λ ⟨i, d⟩ => d * 3^i) |>.sum

def baseTenToBaseFive (n : ℕ) : List ℕ :=
  let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else aux (n / 5) ((n % 5) :: acc)
  aux n []

theorem base_conversion (baseThreeNum : List ℕ) (baseTenNum : ℕ) (baseFiveNum : List ℕ) :
  baseThreeNum = [2, 0, 1, 2, 1] →
  baseTenNum = 178 →
  baseFiveNum = [1, 2, 0, 3] →
  baseThreeToBaseTen baseThreeNum = baseTenNum ∧ baseTenToBaseFive baseTenNum = baseFiveNum :=
by
  intros h1 h2 h3
  unfold baseThreeToBaseTen
  unfold baseTenToBaseFive
  sorry

end base_conversion_l567_56794


namespace negative_correction_is_correct_l567_56780

-- Define the constants given in the problem
def gain_per_day : ℚ := 13 / 4
def set_time : ℚ := 8 -- 8 A.M. on April 10
def end_time : ℚ := 15 -- 3 P.M. on April 19
def days_passed : ℚ := 9

-- Calculate the total time in hours from 8 A.M. on April 10 to 3 P.M. on April 19
def total_hours_passed : ℚ := days_passed * 24 + (end_time - set_time)

-- Calculate the gain in time per hour
def gain_per_hour : ℚ := gain_per_day / 24

-- Calculate the total gained time over the total hours passed
def total_gain : ℚ := total_hours_passed * gain_per_hour

-- The negative correction m to be subtracted
def correction : ℚ := 2899 / 96

theorem negative_correction_is_correct :
  total_gain = correction :=
by
-- skipping the proof
sorry

end negative_correction_is_correct_l567_56780


namespace range_of_x_for_positive_function_value_l567_56702

variable {R : Type*} [LinearOrderedField R]

def even_function (f : R → R) := ∀ x, f (-x) = f x

def monotonically_decreasing_on_nonnegatives (f : R → R) := ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem range_of_x_for_positive_function_value (f : R → R)
  (hf_even : even_function f)
  (hf_monotonic : monotonically_decreasing_on_nonnegatives f)
  (hf_at_2 : f 2 = 0)
  (hf_positive : ∀ x, f (x - 1) > 0) :
  ∀ x, -1 < x ∧ x < 3 := sorry

end range_of_x_for_positive_function_value_l567_56702


namespace x_can_be_any_sign_l567_56706

theorem x_can_be_any_sign
  (x y z w : ℤ)
  (h1 : (y - 1) * (w - 2) ≠ 0)
  (h2 : (x + 2)/(y - 1) < - (z + 3)/(w - 2)) :
  ∃ x : ℤ, True :=
by
  sorry

end x_can_be_any_sign_l567_56706


namespace find_kn_l567_56735

theorem find_kn (k n : ℕ) (h : k * n^2 - k * n - n^2 + n = 94) : k = 48 ∧ n = 2 := 
by 
  sorry

end find_kn_l567_56735


namespace g_value_at_49_l567_56761

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_value_at_49 :
  (∀ x y : ℝ, 0 < x → 0 < y → x * g y - y * g x = g (x^2 / y)) →
  g 49 = 0 :=
by
  -- Assuming the given condition holds for all positive real numbers x and y
  intro h
  -- sorry placeholder represents the proof process
  sorry

end g_value_at_49_l567_56761


namespace consecutive_odd_integers_l567_56787

theorem consecutive_odd_integers (n : ℤ) (h : (n - 2) + (n + 2) = 130) : n = 65 :=
sorry

end consecutive_odd_integers_l567_56787


namespace two_digit_integer_eq_55_l567_56781

theorem two_digit_integer_eq_55
  (c : ℕ)
  (h1 : c / 10 + c % 10 = 10)
  (h2 : (c / 10) * (c % 10) = 25) :
  c = 55 :=
  sorry

end two_digit_integer_eq_55_l567_56781


namespace polynomial_expansion_sum_l567_56722

theorem polynomial_expansion_sum :
  ∀ P Q R S : ℕ, ∀ x : ℕ, 
  (P = 4 ∧ Q = 10 ∧ R = 1 ∧ S = 21) → 
  ((x + 3) * (4 * x ^ 2 - 2 * x + 7) = P * x ^ 3 + Q * x ^ 2 + R * x + S) → 
  P + Q + R + S = 36 :=
by
  intros P Q R S x h1 h2
  sorry

end polynomial_expansion_sum_l567_56722


namespace liquid_in_cylinders_l567_56733

theorem liquid_in_cylinders (n : ℕ) (a : ℝ) (h1 : 2 ≤ n) :
  (∃ x : ℕ → ℝ, ∀ (k : ℕ), (1 ≤ k ∧ k ≤ n) → 
    (if k = 1 then 
      x k = a * n * (n - 2) / (n - 1) ^ 2 
    else if k = 2 then 
      x k = a * (n^2 - 2*n + 2) / (n - 1) ^ 2 
    else 
      x k = a)) :=
sorry

end liquid_in_cylinders_l567_56733


namespace calculate_x_l567_56726

def percentage (p : ℚ) (n : ℚ) := (p / 100) * n

theorem calculate_x : 
  (percentage 47 1442 - percentage 36 1412) + 65 = 234.42 := 
by 
  sorry

end calculate_x_l567_56726


namespace valid_number_count_is_300_l567_56799

-- Define the set of digits
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6]

-- Define the set of odd digits
def odd_digits : List ℕ := [1, 3, 5]

-- Define a function to count valid four-digit numbers
noncomputable def count_valid_numbers : ℕ :=
  (odd_digits.length * (digits.length - 2) * (digits.length - 2) * (digits.length - 3))

-- State the theorem
theorem valid_number_count_is_300 : count_valid_numbers = 300 :=
  sorry

end valid_number_count_is_300_l567_56799


namespace geometric_series_common_ratio_l567_56771

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 500) (hS : S = 2500) (h_series : S = a / (1 - r)) : r = 4 / 5 :=
by
  sorry

end geometric_series_common_ratio_l567_56771


namespace div3_of_div9_l567_56715

theorem div3_of_div9 (u v : ℤ) (h : 9 ∣ (u^2 + u * v + v^2)) : 3 ∣ u ∧ 3 ∣ v :=
sorry

end div3_of_div9_l567_56715


namespace inequality_condition_l567_56788

theorem inequality_condition
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 2015) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / Real.sqrt 2015 :=
by
  sorry

end inequality_condition_l567_56788


namespace tiles_count_l567_56707

variable (c r : ℕ)

-- given: r = 10
def initial_rows_eq : Prop := r = 10

-- assertion: number of tiles is conserved after rearrangement
def tiles_conserved : Prop := c * r = (c - 2) * (r + 4)

-- desired: total number of tiles is 70
def total_tiles : Prop := c * r = 70

theorem tiles_count (h1 : initial_rows_eq r) (h2 : tiles_conserved c r) : total_tiles c r :=
by
  subst h1
  sorry

end tiles_count_l567_56707


namespace sachin_younger_than_rahul_l567_56717

theorem sachin_younger_than_rahul
  (S R : ℝ)
  (h1 : S = 24.5)
  (h2 : S / R = 7 / 9) :
  R - S = 7 := 
by sorry

end sachin_younger_than_rahul_l567_56717


namespace sum_first_nine_primes_l567_56718

theorem sum_first_nine_primes : 
  2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 = 100 :=
by
  sorry

end sum_first_nine_primes_l567_56718


namespace area_of_triangle_ABC_is_1_l567_56751

-- Define the vertices A, B, and C
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (2, 1)

-- Define the function to compute the area of the triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The main theorem to prove that the area of triangle ABC is 1
theorem area_of_triangle_ABC_is_1 : triangle_area A B C = 1 := 
by
  sorry

end area_of_triangle_ABC_is_1_l567_56751


namespace dragons_legs_l567_56742

theorem dragons_legs :
  ∃ (n : ℤ), ∀ (x y : ℤ), x + 3 * y = 26
                       → 40 * x + n * y = 298
                       → n = 14 :=
by
  sorry

end dragons_legs_l567_56742


namespace find_x_l567_56786

theorem find_x (x : ℕ) : (x % 7 = 0) ∧ (x^2 > 200) ∧ (x < 30) ↔ (x = 21 ∨ x = 28) :=
by
  sorry

end find_x_l567_56786


namespace find_sisters_dolls_l567_56797

variable (H S : ℕ)

-- Conditions
def hannah_has_5_times_sisters_dolls : Prop :=
  H = 5 * S

def total_dolls_is_48 : Prop :=
  H + S = 48

-- Question: Prove S = 8
theorem find_sisters_dolls (h1 : hannah_has_5_times_sisters_dolls H S) (h2 : total_dolls_is_48 H S) : S = 8 :=
sorry

end find_sisters_dolls_l567_56797


namespace thirteen_percent_greater_than_80_l567_56773

theorem thirteen_percent_greater_than_80 (x : ℝ) (h : x = 1.13 * 80) : x = 90.4 :=
sorry

end thirteen_percent_greater_than_80_l567_56773


namespace minimum_cans_needed_l567_56729

theorem minimum_cans_needed (h : ∀ c, c * 10 ≥ 120) : ∃ c, c = 12 :=
by
  sorry

end minimum_cans_needed_l567_56729


namespace lucy_needs_more_distance_l567_56798

noncomputable def mary_distance : ℝ := (3 / 8) * 24
noncomputable def edna_distance : ℝ := (2 / 3) * mary_distance
noncomputable def lucy_distance : ℝ := (5 / 6) * edna_distance

theorem lucy_needs_more_distance :
  mary_distance - lucy_distance = 4 := by
  sorry

end lucy_needs_more_distance_l567_56798


namespace max_a1_value_l567_56712

theorem max_a1_value (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n+2) = a n + a (n+1))
    (h2 : ∀ n : ℕ, a n > 0) (h3 : a 5 = 60) : a 1 ≤ 11 :=
by 
  sorry

end max_a1_value_l567_56712


namespace symmetric_line_x_axis_l567_56704

theorem symmetric_line_x_axis (y : ℝ → ℝ) (x : ℝ) :
  (∀ x, y x = 2 * x + 1) → (∀ x, -y x = 2 * x + 1) → y x = -2 * x -1 :=
by
  intro h1 h2
  sorry

end symmetric_line_x_axis_l567_56704


namespace Bruce_paid_correct_amount_l567_56778

def grape_kg := 9
def grape_price_per_kg := 70
def mango_kg := 7
def mango_price_per_kg := 55
def orange_kg := 5
def orange_price_per_kg := 45
def apple_kg := 3
def apple_price_per_kg := 80

def total_cost := grape_kg * grape_price_per_kg + 
                  mango_kg * mango_price_per_kg + 
                  orange_kg * orange_price_per_kg + 
                  apple_kg * apple_price_per_kg

theorem Bruce_paid_correct_amount : total_cost = 1480 := by
  sorry

end Bruce_paid_correct_amount_l567_56778


namespace ratio_of_boys_to_girls_l567_56772

/-- 
  Given 200 girls and a total of 600 students in a college,
  the ratio of the number of boys to the number of girls is 2:1.
--/
theorem ratio_of_boys_to_girls 
  (num_girls : ℕ) (total_students : ℕ) (h_girls : num_girls = 200) 
  (h_total : total_students = 600) : 
  (total_students - num_girls) / num_girls = 2 :=
by
  sorry

end ratio_of_boys_to_girls_l567_56772


namespace max_showers_l567_56710

open Nat

variable (household water_limit water_for_drinking_and_cooking water_per_shower pool_length pool_width pool_height water_per_cubic_foot pool_leakage_rate days_in_july : ℕ)

def volume_of_pool (length width height: ℕ): ℕ :=
  length * width * height

def water_usage (drinking cooking pool leakage: ℕ): ℕ :=
  drinking + cooking + pool + leakage

theorem max_showers (h1: water_limit = 1000)
                    (h2: water_for_drinking_and_cooking = 100)
                    (h3: water_per_shower = 20)
                    (h4: pool_length = 10)
                    (h5: pool_width = 10)
                    (h6: pool_height = 6)
                    (h7: water_per_cubic_foot = 1)
                    (h8: pool_leakage_rate = 5)
                    (h9: days_in_july = 31) : 
  (water_limit - water_usage water_for_drinking_and_cooking
                                  (volume_of_pool pool_length pool_width pool_height) 
                                  ((pool_leakage_rate * days_in_july))) / water_per_shower = 7 := by
  sorry

end max_showers_l567_56710


namespace num_ordered_pairs_l567_56747

theorem num_ordered_pairs :
  ∃ n : ℕ, n = 49 ∧ ∀ (a b : ℕ), a + b = 50 → 0 < a ∧ 0 < b → (1 ≤ a ∧ a < 50) :=
by
  sorry

end num_ordered_pairs_l567_56747


namespace discount_rate_l567_56705

theorem discount_rate (cost_price marked_price desired_profit_margin selling_price : ℝ)
  (h1 : cost_price = 160)
  (h2 : marked_price = 240)
  (h3 : desired_profit_margin = 0.2)
  (h4 : selling_price = cost_price * (1 + desired_profit_margin)) :
  marked_price * (1 - ((marked_price - selling_price) / marked_price)) = selling_price :=
by
  sorry

end discount_rate_l567_56705


namespace random_event_is_eventA_l567_56793

-- Definitions of conditions
def eventA : Prop := true  -- Tossing a coin and it lands either heads up or tails up is a random event
def eventB : Prop := (∀ (a b : ℝ), (b * a = b * a))  -- The area of a rectangle with sides of length a and b is ab is a certain event
def eventC : Prop := ∃ (defective_items : ℕ), (defective_items / 100 = 10 / 100)  -- Drawing 2 defective items from 100 parts with 10% defective parts is uncertain
def eventD : Prop := false -- Scoring 105 points in a regular 100-point system exam is an impossible event

-- The proof problem statement
theorem random_event_is_eventA : eventA ∧ ¬eventB ∧ ¬eventC ∧ ¬eventD := 
sorry

end random_event_is_eventA_l567_56793


namespace find_a_l567_56792

theorem find_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 20) 
  (h3 : (4254253 % 53^1 - a) % 17 = 0): 
  a = 3 := 
sorry

end find_a_l567_56792


namespace selling_price_of_book_l567_56768

theorem selling_price_of_book (SP : ℝ) (CP : ℝ := 200) :
  (SP - CP) = (340 - CP) + 0.05 * CP → SP = 350 :=
by {
  sorry
}

end selling_price_of_book_l567_56768


namespace sqrt_diff_of_squares_l567_56719

theorem sqrt_diff_of_squares : (Real.sqrt 3 - 2) * (Real.sqrt 3 + 2) = -1 := by
  sorry

end sqrt_diff_of_squares_l567_56719


namespace casey_nail_decorating_time_l567_56730

theorem casey_nail_decorating_time :
  let coat_application_time := 20
  let coat_drying_time := 20
  let pattern_time := 40
  let total_time := 3 * (coat_application_time + coat_drying_time) + pattern_time
  total_time = 160 :=
by
  let coat_application_time := 20
  let coat_drying_time := 20
  let pattern_time := 40
  let total_time := 3 * (coat_application_time + coat_drying_time) + pattern_time
  trivial

end casey_nail_decorating_time_l567_56730


namespace mark_less_than_kate_and_laura_l567_56746

theorem mark_less_than_kate_and_laura (K : ℝ) (h : K + 2 * K + 3 * K + 4.5 * K = 360) :
  let Pat := 2 * K
  let Mark := 3 * K
  let Laura := 4.5 * K
  let Combined := K + Laura
  Mark - Combined = -85.72 :=
sorry

end mark_less_than_kate_and_laura_l567_56746


namespace sequences_count_l567_56759

theorem sequences_count (a_n b_n c_n : ℕ → ℕ) :
  (a_n 1 = 1) ∧ (b_n 1 = 1) ∧ (c_n 1 = 1) ∧ 
  (∀ n : ℕ, a_n (n + 1) = a_n n + b_n n) ∧ 
  (∀ n : ℕ, b_n (n + 1) = a_n n + b_n n + c_n n) ∧ 
  (∀ n : ℕ, c_n (n + 1) = b_n n + c_n n) → 
  ∀ n : ℕ, a_n n + b_n n + c_n n = 
            (1/2 * ((1 + Real.sqrt 2)^(n+1) + (1 - Real.sqrt 2)^(n+1))) :=
by
  intro h
  sorry

end sequences_count_l567_56759


namespace stratified_sampling_l567_56727

-- Definition of the given variables and conditions
def total_students_grade10 : ℕ := 30
def total_students_grade11 : ℕ := 40
def selected_students_grade11 : ℕ := 8

-- Implementation of the stratified sampling proportion requirement
theorem stratified_sampling (x : ℕ) (hx : (x : ℚ) / total_students_grade10 = (selected_students_grade11 : ℚ) / total_students_grade11) :
  x = 6 :=
by
  sorry

end stratified_sampling_l567_56727


namespace find_a_n_l567_56734

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_sum : ∀ n : ℕ, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n : ℕ, a n = Real.sqrt ↑n - Real.sqrt (↑n - 1) :=
by
  sorry

end find_a_n_l567_56734


namespace line_slope_intercept_through_points_l567_56721

theorem line_slope_intercept_through_points (a b : ℝ) :
  (∀ x y : ℝ, (x, y) = (3, 7) ∨ (x, y) = (7, 19) → y = a * x + b) →
  a - b = 5 :=
by
  sorry

end line_slope_intercept_through_points_l567_56721


namespace silver_coin_worth_l567_56754

theorem silver_coin_worth :
  ∀ (g : ℕ) (S : ℕ) (n_gold n_silver cash : ℕ), 
  g = 50 →
  n_gold = 3 →
  n_silver = 5 →
  cash = 30 →
  n_gold * g + n_silver * S + cash = 305 →
  S = 25 :=
by
  intros g S n_gold n_silver cash
  intros hg hng hnsi hcash htotal
  sorry

end silver_coin_worth_l567_56754


namespace share_of_C_l567_56760

/-- Given the conditions:
  - Total investment is Rs. 120,000.
  - A's investment is Rs. 6,000 more than B's.
  - B's investment is Rs. 8,000 more than C's.
  - Profit distribution ratio among A, B, and C is 4:3:2.
  - Total profit is Rs. 50,000.
Prove that C's share of the profit is Rs. 11,111.11. -/
theorem share_of_C (total_investment : ℝ)
  (A_more_than_B : ℝ)
  (B_more_than_C : ℝ)
  (profit_distribution : ℝ)
  (total_profit : ℝ) :
  total_investment = 120000 →
  A_more_than_B = 6000 →
  B_more_than_C = 8000 →
  profit_distribution = 4 / 9 →
  total_profit = 50000 →
  ∃ (C_share : ℝ), C_share = 11111.11 :=
by
  sorry

end share_of_C_l567_56760


namespace value_of_x_l567_56769

theorem value_of_x : 
  ∀ (x : ℕ), x = (2011^2 + 2011) / 2011 → x = 2012 :=
by
  intro x
  intro h
  sorry

end value_of_x_l567_56769


namespace root_one_value_of_m_real_roots_range_of_m_l567_56749

variables {m x : ℝ}

-- Part 1: Prove that if 1 is a root of 'mx^2 - 4x + 1 = 0', then m = 3
theorem root_one_value_of_m (h : m * 1^2 - 4 * 1 + 1 = 0) : m = 3 :=
  by sorry

-- Part 2: Prove that 'mx^2 - 4x + 1 = 0' has real roots iff 'm ≤ 4 ∧ m ≠ 0'
theorem real_roots_range_of_m : (∃ x : ℝ, m * x^2 - 4 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 0) :=
  by sorry

end root_one_value_of_m_real_roots_range_of_m_l567_56749


namespace business_value_l567_56784

-- Define the conditions
variable (V : ℝ) -- Total value of the business
variable (man_shares : ℝ := (2/3) * V) -- Man's share in the business
variable (sold_shares_value : ℝ := (3/4) * man_shares) -- Value of sold shares
variable (sale_price : ℝ := 45000) -- Price the shares were sold for

-- State the theorem to be proven
theorem business_value (h : (3/4) * (2/3) * V = 45000) : V = 90000 := by
  sorry

end business_value_l567_56784


namespace range_of_a_l567_56736

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x + 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 * Real.log x) / x
noncomputable def h (x : ℝ) : ℝ := (x^2 - x - 2) / x^3

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, 0 < x1 ∧ x1 ≤ 1 ∧ 0 < x2 ∧ x2 ≤ 1 → f a x1 ≥ g x2) ↔ a ≥ -2 :=
by
  sorry

end range_of_a_l567_56736


namespace quadratic_function_passing_origin_l567_56779

theorem quadratic_function_passing_origin (a : ℝ) (h : ∃ x y, y = ax^2 + x + a * (a - 2) ∧ (x, y) = (0, 0)) : a = 2 := by
  sorry

end quadratic_function_passing_origin_l567_56779


namespace remaining_insects_is_twenty_one_l567_56753

-- Define the initial counts of each type of insect
def spiders := 3
def ants := 12
def ladybugs := 8

-- Define the number of ladybugs that flew away
def ladybugs_flew_away := 2

-- Define the total initial number of insects
def total_insects_initial := spiders + ants + ladybugs

-- Define the total number of insects that remain after some ladybugs fly away
def total_insects_remaining := total_insects_initial - ladybugs_flew_away

-- Theorem statement: proving that the number of insects remaining is 21
theorem remaining_insects_is_twenty_one : total_insects_remaining = 21 := sorry

end remaining_insects_is_twenty_one_l567_56753


namespace maxwell_age_l567_56774

theorem maxwell_age (M : ℕ) (h1 : ∃ n : ℕ, n = M + 2) (h2 : ∃ k : ℕ, k = 4) (h3 : (M + 2) = 2 * 4) : M = 6 :=
sorry

end maxwell_age_l567_56774


namespace find_ratio_l567_56700

-- Given that the tangent of angle θ (inclination angle) is -2
def tan_theta (θ : Real) : Prop := Real.tan θ = -2

theorem find_ratio (θ : Real) (h : tan_theta θ) :
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 1 / 3 := by
  sorry

end find_ratio_l567_56700


namespace complex_number_imaginary_l567_56739

theorem complex_number_imaginary (x : ℝ) 
  (h1 : x^2 - 2*x - 3 = 0)
  (h2 : x + 1 ≠ 0) : x = 3 := sorry

end complex_number_imaginary_l567_56739


namespace box_weight_without_balls_l567_56785

theorem box_weight_without_balls :
  let number_of_balls := 30
  let weight_per_ball := 0.36
  let total_weight_with_balls := 11.26
  let total_weight_of_balls := number_of_balls * weight_per_ball
  let weight_of_box := total_weight_with_balls - total_weight_of_balls
  weight_of_box = 0.46 :=
by 
  sorry

end box_weight_without_balls_l567_56785


namespace x_squared_minus_y_squared_l567_56770

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9 / 13) (h2 : x - y = 5 / 13) : x^2 - y^2 = 45 / 169 := 
by 
  -- proof omitted 
  sorry

end x_squared_minus_y_squared_l567_56770


namespace no_valid_k_values_l567_56748

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def roots_are_primes (k : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 57 ∧ p * q = k

theorem no_valid_k_values : ∀ k : ℕ, ¬ roots_are_primes k := by
  sorry

end no_valid_k_values_l567_56748


namespace find_m_l567_56752

theorem find_m (m : ℕ) (h : m * (Nat.factorial m) + 2 * (Nat.factorial m) = 5040) : m = 5 :=
by
  sorry

end find_m_l567_56752


namespace binomial_1300_2_eq_844350_l567_56713

theorem binomial_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 := 
by
  sorry

end binomial_1300_2_eq_844350_l567_56713


namespace find_positive_integer_l567_56795

theorem find_positive_integer (n : ℕ) (h1 : 100 % n = 3) (h2 : 197 % n = 3) : n = 97 := 
sorry

end find_positive_integer_l567_56795


namespace geometric_sequence_new_product_l567_56703

theorem geometric_sequence_new_product 
  (a r : ℝ) (n : ℕ) (h_even : n % 2 = 0)
  (P S S' : ℝ)
  (hP : P = a^n * r^(n * (n-1) / 2))
  (hS : S = a * (1 - r^n) / (1 - r))
  (hS' : S' = (1 - r^n) / (a * (1 - r))) :
  (2^n * a^n * r^(n * (n-1) / 2)) = (S * S')^(n / 2) :=
sorry

end geometric_sequence_new_product_l567_56703


namespace last_score_is_65_l567_56724

-- Define the scores and the problem conditions
def scores := [65, 72, 75, 80, 85, 88, 92]
def total_sum := 557
def remaining_sum (score : ℕ) : ℕ := total_sum - score

-- Define a property to check divisibility
def divisible_by (n d : ℕ) : Prop := n % d = 0

-- The main theorem statement
theorem last_score_is_65 :
  (∀ s ∈ scores, divisible_by (remaining_sum s) 6) ∧ divisible_by total_sum 7 ↔ scores = [65, 72, 75, 80, 85, 88, 92] :=
sorry

end last_score_is_65_l567_56724


namespace xiao_yun_age_l567_56758

theorem xiao_yun_age (x : ℕ) (h1 : ∀ x, x + 25 = Xiao_Yun_fathers_current_age)
                     (h2 : ∀ x, Xiao_Yun_fathers_age_in_5_years = 2 * (x+5) - 10) :
  x = 30 := by
  sorry

end xiao_yun_age_l567_56758


namespace three_pow_2023_mod_17_l567_56745

theorem three_pow_2023_mod_17 : (3 ^ 2023) % 17 = 7 := by
  sorry

end three_pow_2023_mod_17_l567_56745


namespace fraction_of_15_smaller_by_20_l567_56732

/-- Define 80% of 40 -/
def eighty_percent_of_40 : ℝ := 0.80 * 40

/-- Define the fraction of 15 that we are looking for -/
def fraction_of_15 (x : ℝ) : ℝ := x * 15

/-- Define the problem statement -/
theorem fraction_of_15_smaller_by_20 : ∃ x : ℝ, fraction_of_15 x = eighty_percent_of_40 - 20 ∧ x = 4 / 5 :=
by
  sorry

end fraction_of_15_smaller_by_20_l567_56732


namespace student_sums_l567_56765

theorem student_sums (x y : ℕ) (h1 : y = 3 * x) (h2 : x + y = 48) : y = 36 :=
by
  sorry

end student_sums_l567_56765


namespace point_on_number_line_l567_56762

theorem point_on_number_line (a : ℤ) (h : abs (a + 3) = 4) : a = 1 ∨ a = -7 := 
sorry

end point_on_number_line_l567_56762


namespace computation_result_l567_56766

theorem computation_result : 8 * (2 / 17) * 34 * (1 / 4) = 8 := by
  sorry

end computation_result_l567_56766
