import Mathlib

namespace find_value_x_y_cube_l968_96880

variables (x y k c m : ℝ)

theorem find_value_x_y_cube
  (h1 : x^3 * y^3 = k)
  (h2 : 1 / x^3 + 1 / y^3 = c)
  (h3 : x + y = m) :
  (x + y)^3 = c * k + 3 * k^(1/3) * m :=
by
  sorry

end find_value_x_y_cube_l968_96880


namespace julie_monthly_salary_l968_96846

def hourly_rate : ℕ := 5
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 6
def missed_days : ℕ := 1
def weeks_per_month : ℕ := 4

theorem julie_monthly_salary :
  (hourly_rate * hours_per_day * (days_per_week - missed_days) * weeks_per_month) = 920 :=
by
  sorry

end julie_monthly_salary_l968_96846


namespace cost_of_milkshake_is_correct_l968_96876

-- Definitions related to the problem conditions
def initial_amount : ℕ := 15
def spent_on_cupcakes : ℕ := initial_amount * (1 / 3)
def remaining_after_cupcakes : ℕ := initial_amount - spent_on_cupcakes
def spent_on_sandwich : ℕ := remaining_after_cupcakes * (20 / 100)
def remaining_after_sandwich : ℕ := remaining_after_cupcakes - spent_on_sandwich
def remaining_after_milkshake : ℕ := 4
def cost_of_milkshake : ℕ := remaining_after_sandwich - remaining_after_milkshake

-- The theorem stating the equivalent proof problem
theorem cost_of_milkshake_is_correct :
  cost_of_milkshake = 4 :=
sorry

end cost_of_milkshake_is_correct_l968_96876


namespace cost_of_fencing_correct_l968_96867

noncomputable def cost_of_fencing (d : ℝ) (r : ℝ) : ℝ :=
  Real.pi * d * r

theorem cost_of_fencing_correct : cost_of_fencing 30 5 = 471 :=
by
  sorry

end cost_of_fencing_correct_l968_96867


namespace min_value_collinear_l968_96899

theorem min_value_collinear (x y : ℝ) (h₁ : 2 * x + 3 * y = 3) (h₂ : 0 < x) (h₃ : 0 < y) : 
  (3 / x + 2 / y) = 8 :=
sorry

end min_value_collinear_l968_96899


namespace revenue_from_full_price_tickets_l968_96884

noncomputable def full_price_ticket_revenue (f h p : ℕ) : ℕ := f * p

theorem revenue_from_full_price_tickets (f h p : ℕ) (total_tickets total_revenue : ℕ) 
  (tickets_eq : f + h = total_tickets)
  (revenue_eq : f * p + h * (p / 2) = total_revenue) 
  (total_tickets_value : total_tickets = 180)
  (total_revenue_value : total_revenue = 2652) :
  full_price_ticket_revenue f h p = 984 :=
by {
  sorry
}

end revenue_from_full_price_tickets_l968_96884


namespace more_oil_l968_96887

noncomputable def original_price (P : ℝ) :=
  P - 0.3 * P = 70

noncomputable def amount_of_oil_before (P : ℝ) :=
  700 / P

noncomputable def amount_of_oil_after :=
  700 / 70

theorem more_oil (P : ℝ) (h1 : original_price P) :
  (amount_of_oil_after - amount_of_oil_before P) = 3 :=
  sorry

end more_oil_l968_96887


namespace time_after_2004_hours_l968_96840

variable (h : ℕ) 

-- Current time is represented as an integer from 0 to 11 (9 o'clock).
def current_time : ℕ := 9

-- 12-hour clock cycles every 12 hours.
def cycle : ℕ := 12

-- Time after 2004 hours.
def hours_after : ℕ := 2004

-- Proof statement
theorem time_after_2004_hours (h : ℕ) :
  (current_time + hours_after) % cycle = current_time := 
sorry

end time_after_2004_hours_l968_96840


namespace solution_set_for_a1_find_a_if_min_value_is_4_l968_96872

noncomputable def f (a x : ℝ) : ℝ := |2 * x - 1| + |a * x - 5|

theorem solution_set_for_a1 : 
  { x : ℝ | f 1 x ≥ 9 } = { x : ℝ | x ≤ -1 ∨ x > 5 } :=
sorry

theorem find_a_if_min_value_is_4 :
  ∃ a : ℝ, (0 < a ∧ a < 5) ∧ (∀ x : ℝ, f a x ≥ 4) ∧ (∃ x : ℝ, f a x = 4) ∧ a = 2 :=
sorry

end solution_set_for_a1_find_a_if_min_value_is_4_l968_96872


namespace rational_sum_p_q_l968_96863

noncomputable def x := (Real.sqrt 5 - 1) / 2

theorem rational_sum_p_q :
  ∃ (p q : ℚ), x^3 + p * x + q = 0 ∧ p + q = -1 := by
  sorry

end rational_sum_p_q_l968_96863


namespace triangular_region_area_l968_96897

theorem triangular_region_area :
  let x_intercept := 4
  let y_intercept := 6
  let area := (1 / 2) * x_intercept * y_intercept
  area = 12 :=
by
  sorry

end triangular_region_area_l968_96897


namespace seating_solution_l968_96815

/-- 
Imagine Abby, Bret, Carl, and Dana are seated in a row of four seats numbered from 1 to 4.
Joe observes them and declares:

- "Bret is sitting next to Dana" (False)
- "Carl is between Abby and Dana" (False)

Further, it is known that Abby is in seat #2.

Who is seated in seat #3? 
-/

def seating_problem : Prop :=
  ∃ (seats : ℕ → ℕ),
  (¬ (seats 1 = 1 ∧ seats 1 = 4 ∨ seats 4 = 1 ∧ seats 4 = 4)) ∧
  (¬ (seats 3 > seats 1 ∧ seats 3 < seats 2 ∨ seats 3 > seats 2 ∧ seats 3 < seats 1)) ∧
  (seats 2 = 2) →
  (seats 3 = 3)

theorem seating_solution : seating_problem :=
sorry

end seating_solution_l968_96815


namespace usual_time_is_24_l968_96848

variable (R T : ℝ)
variable (usual_rate fraction_of_rate early_min : ℝ)
variable (h1 : fraction_of_rate = 6 / 7)
variable (h2 : early_min = 4)
variable (h3 : (R / (fraction_of_rate * R)) = 7 / 6)
variable (h4 : ((T - early_min) / T) = fraction_of_rate)

theorem usual_time_is_24 {R T : ℝ} (fraction_of_rate := 6/7) (early_min := 4) :
  fraction_of_rate = 6 / 7 ∧ early_min = 4 → 
  (T - early_min) / T = fraction_of_rate → 
  T = 24 :=
by
  intros hfraction_hearly htime_eq_fraction
  sorry

end usual_time_is_24_l968_96848


namespace xiaoning_pe_comprehensive_score_l968_96819

def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.7
def midterm_score : ℝ := 80
def final_score : ℝ := 90

theorem xiaoning_pe_comprehensive_score : midterm_score * midterm_weight + final_score * final_weight = 87 :=
by
  sorry

end xiaoning_pe_comprehensive_score_l968_96819


namespace expr_1989_eval_expr_1990_eval_l968_96807

def nestedExpr : ℕ → ℤ
| 0     => 0
| (n+1) => -1 - (nestedExpr n)

-- Conditions translated into Lean definitions:
def expr_1989 := nestedExpr 1989
def expr_1990 := nestedExpr 1990

-- The proof statements:
theorem expr_1989_eval : expr_1989 = -1 := sorry
theorem expr_1990_eval : expr_1990 = 0 := sorry

end expr_1989_eval_expr_1990_eval_l968_96807


namespace find_q_revolutions_per_minute_l968_96871

variable (p_rpm : ℕ) (q_rpm : ℕ) (t : ℕ)

def revolutions_per_minute_q : Prop :=
  (p_rpm = 10) → (t = 4) → (q_rpm = (10 / 60 * 4 + 2) * 60 / 4) → (q_rpm = 120)

theorem find_q_revolutions_per_minute (p_rpm q_rpm t : ℕ) :
  revolutions_per_minute_q p_rpm q_rpm t :=
by
  unfold revolutions_per_minute_q
  sorry

end find_q_revolutions_per_minute_l968_96871


namespace range_of_a_l968_96891

noncomputable def f (a x : ℝ) := (Real.exp x - a * x^2) 

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), 0 ≤ x → f a x ≥ x + 1) ↔ a ∈ Set.Iic (1/2) :=
by
  sorry

end range_of_a_l968_96891


namespace unique_real_solution_between_consecutive_integers_l968_96808

theorem unique_real_solution_between_consecutive_integers (k : ℕ) (h : k > 0) :
  ∃! x : ℝ, k < x ∧ x < k + 1 ∧ (⌊x⌋ : ℝ) * (x^2 + 1) = x^3 := sorry

end unique_real_solution_between_consecutive_integers_l968_96808


namespace Anne_is_15_pounds_heavier_l968_96874

def Anne_weight : ℕ := 67
def Douglas_weight : ℕ := 52

theorem Anne_is_15_pounds_heavier : Anne_weight - Douglas_weight = 15 := by
  sorry

end Anne_is_15_pounds_heavier_l968_96874


namespace calculation_correct_l968_96830

def f (x : ℚ) := (2 * x^2 + 6 * x + 9) / (x^2 + 3 * x + 5)
def g (x : ℚ) := 2 * x + 1

theorem calculation_correct : f (g 2) + g (f 2) = 308 / 45 := by
  sorry

end calculation_correct_l968_96830


namespace probability_intersecting_diagonals_l968_96850

def number_of_vertices := 10

def number_of_diagonals : ℕ := Nat.choose number_of_vertices 2 - number_of_vertices

def number_of_ways_choose_two_diagonals := Nat.choose number_of_diagonals 2

def number_of_sets_of_intersecting_diagonals : ℕ := Nat.choose number_of_vertices 4

def intersection_probability : ℚ :=
  (number_of_sets_of_intersecting_diagonals : ℚ) / (number_of_ways_choose_two_diagonals : ℚ)

theorem probability_intersecting_diagonals :
  intersection_probability = 42 / 119 :=
by
  sorry

end probability_intersecting_diagonals_l968_96850


namespace linear_price_item_func_l968_96858

noncomputable def price_item_func (x : ℝ) : Prop :=
  ∃ (y : ℝ), y = - (1/4) * x + 50 ∧ 0 < x ∧ x < 200

theorem linear_price_item_func : ∀ x, price_item_func x ↔ (∃ y, y = - (1/4) * x + 50 ∧ 0 < x ∧ x < 200) :=
by
  sorry

end linear_price_item_func_l968_96858


namespace smallest_natural_number_l968_96886

open Nat

theorem smallest_natural_number (n : ℕ) :
  (n + 1) % 4 = 0 ∧ (n + 1) % 6 = 0 ∧ (n + 1) % 10 = 0 ∧ (n + 1) % 12 = 0 →
  n = 59 :=
by
  sorry

end smallest_natural_number_l968_96886


namespace largest_angle_measures_203_l968_96885

-- Define the angles of the hexagon
def angle1 (x : ℚ) : ℚ := x + 2
def angle2 (x : ℚ) : ℚ := 2 * x + 1
def angle3 (x : ℚ) : ℚ := 3 * x
def angle4 (x : ℚ) : ℚ := 4 * x - 1
def angle5 (x : ℚ) : ℚ := 5 * x + 2
def angle6 (x : ℚ) : ℚ := 6 * x - 2

-- Define the sum of interior angles for a hexagon
def hexagon_angle_sum : ℚ := 720

-- Prove that the largest angle is equal to 203 degrees given the conditions
theorem largest_angle_measures_203 (x : ℚ) (h : angle1 x + angle2 x + angle3 x + angle4 x + angle5 x + angle6 x = hexagon_angle_sum) :
  (6 * x - 2) = 203 := by
  sorry

end largest_angle_measures_203_l968_96885


namespace polynomial_coeffs_identity_l968_96864

theorem polynomial_coeffs_identity : 
  (∀ a b c : ℝ, (2 * x^4 + x^3 - 41 * x^2 + 83 * x - 45 = 
                (a * x^2 + b * x + c) * (x^2 + 4 * x + 9))
                  → a = 2 ∧ b = -7 ∧ c = -5) :=
by
  intros a b c h
  have h₁ : a = 2 := 
    sorry-- prove that a = 2
  have h₂ : b = -7 := 
    sorry-- prove that b = -7
  have h₃ : c = -5 := 
    sorry-- prove that c = -5
  exact ⟨h₁, h₂, h₃⟩

end polynomial_coeffs_identity_l968_96864


namespace symmetric_point_of_M_origin_l968_96843

-- Define the point M with given coordinates
def M : (ℤ × ℤ) := (-3, -5)

-- The theorem stating that the symmetric point of M about the origin is (3, 5)
theorem symmetric_point_of_M_origin :
  let symmetric_point : (ℤ × ℤ) := (-M.1, -M.2)
  symmetric_point = (3, 5) :=
by
  -- (Proof should be filled)
  sorry

end symmetric_point_of_M_origin_l968_96843


namespace angle_terminal_side_eq_l968_96816

theorem angle_terminal_side_eq (α : ℝ) : 
  (α = -4 * Real.pi / 3 + 2 * Real.pi) → (0 ≤ α ∧ α < 2 * Real.pi) → α = 2 * Real.pi / 3 := 
by 
  sorry

end angle_terminal_side_eq_l968_96816


namespace cubic_roots_proof_l968_96824

noncomputable def cubic_roots_reciprocal (a b c : ℝ) (h1 : a + b + c = 7) 
  (h2 : a * b + b * c + c * a = 3) (h3 : a * b * c = -4) : ℝ :=
  (1 / a^2) + (1 / b^2) + (1 / c^2)

theorem cubic_roots_proof (a b c : ℝ) (h1 : a + b + c = 7) 
  (h2 : a * b + b * c + c * a = 3) (h3 : a * b * c = -4) : 
  cubic_roots_reciprocal a b c h1 h2 h3 = 65 / 16 :=
sorry

end cubic_roots_proof_l968_96824


namespace prob_all_pass_prob_at_least_one_pass_most_likely_event_l968_96844

noncomputable def probability_A := 2 / 5
noncomputable def probability_B := 3 / 4
noncomputable def probability_C := 1 / 3
noncomputable def prob_none_pass := (1 - probability_A) * (1 - probability_B) * (1 - probability_C)
noncomputable def prob_one_pass := 
  (probability_A * (1 - probability_B) * (1 - probability_C)) +
  ((1 - probability_A) * probability_B * (1 - probability_C)) +
  ((1 - probability_A) * (1 - probability_B) * probability_C)
noncomputable def prob_two_pass := 
  (probability_A * probability_B * (1 - probability_C)) +
  (probability_A * (1 - probability_B) * probability_C) +
  ((1 - probability_A) * probability_B * probability_C)

-- Prove that the probability that all three candidates pass is 1/10
theorem prob_all_pass : probability_A * probability_B * probability_C = 1 / 10 := by
  sorry

-- Prove that the probability that at least one candidate passes is 9/10
theorem prob_at_least_one_pass : 1 - prob_none_pass = 9 / 10 := by
  sorry

-- Prove that the most likely event of passing is exactly one candidate passing with probability 5/12
theorem most_likely_event : prob_one_pass > prob_two_pass ∧ prob_one_pass > probability_A * probability_B * probability_C ∧ prob_one_pass > prob_none_pass ∧ prob_one_pass = 5 / 12 := by
  sorry

end prob_all_pass_prob_at_least_one_pass_most_likely_event_l968_96844


namespace pelicans_among_non_egrets_is_47_percent_l968_96896

-- Definitions for the percentage of each type of bird.
def pelican_percentage : ℝ := 0.4
def cormorant_percentage : ℝ := 0.2
def egret_percentage : ℝ := 0.15
def osprey_percentage : ℝ := 0.25

-- Calculate the percentage of pelicans among the non-egret birds.
theorem pelicans_among_non_egrets_is_47_percent :
  (pelican_percentage / (1 - egret_percentage)) * 100 = 47 :=
by
  -- Detailed proof goes here
  sorry

end pelicans_among_non_egrets_is_47_percent_l968_96896


namespace number_of_possible_n_l968_96812

theorem number_of_possible_n :
  ∃ (a : ℕ), (∀ n, (n = a^3) ∧ 
  ((∃ b c : ℕ, b ≠ c ∧ b ≠ a ∧ c ≠ a ∧ a = b * c)) ∧ 
  (a + b + c = 2010) ∧ 
  (a > 0) ∧
  (b > 0) ∧
  (c > 0)) → 
  ∃ (num_n : ℕ), num_n = 2009 :=
  sorry

end number_of_possible_n_l968_96812


namespace evaluate_expression_l968_96894

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end evaluate_expression_l968_96894


namespace problem1_problem2_l968_96814

namespace MathProofs

theorem problem1 : (0.25 * 4 - ((5 / 6) + (1 / 12)) * (6 / 5)) = (1 / 10) := by
  sorry

theorem problem2 : ((5 / 12) - (5 / 16)) * (4 / 5) + (2 / 3) - (3 / 4) = 0 := by
  sorry

end MathProofs

end problem1_problem2_l968_96814


namespace price_of_each_orange_l968_96851

theorem price_of_each_orange 
  (x : ℕ)
  (a o : ℕ)
  (h1 : a + o = 20)
  (h2 : 40 * a + x * o = 1120)
  (h3 : (a + o - 10) * 52 = 1120 - 10 * x) :
  x = 60 :=
sorry

end price_of_each_orange_l968_96851


namespace pins_after_one_month_l968_96822

def avg_pins_per_day : ℕ := 10
def delete_pins_per_week_per_person : ℕ := 5
def group_size : ℕ := 20
def initial_pins : ℕ := 1000

theorem pins_after_one_month
  (avg_pins_per_day_pos : avg_pins_per_day = 10)
  (delete_pins_per_week_per_person_pos : delete_pins_per_week_per_person = 5)
  (group_size_pos : group_size = 20)
  (initial_pins_pos : initial_pins = 1000) : 
  1000 + (avg_pins_per_day * group_size * 30) - (delete_pins_per_week_per_person * group_size * 4) = 6600 :=
by
  sorry

end pins_after_one_month_l968_96822


namespace tiling_tromino_l968_96857

theorem tiling_tromino (m n : ℕ) : (∀ t : ℕ, (t = 3) → (3 ∣ m * n)) → (m * n % 6 = 0) → (m * n % 6 = 0) :=
by
  sorry

end tiling_tromino_l968_96857


namespace divides_by_3_l968_96828

theorem divides_by_3 (a b c : ℕ) (h : 9 ∣ a ^ 3 + b ^ 3 + c ^ 3) : 3 ∣ a ∨ 3 ∣ b ∨ 3 ∣ c :=
sorry

end divides_by_3_l968_96828


namespace not_equiv_2_pi_six_and_11_pi_six_l968_96829

def polar_equiv (r θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * ↑k * Real.pi

theorem not_equiv_2_pi_six_and_11_pi_six :
  ¬ polar_equiv 2 (Real.pi / 6) (11 * Real.pi / 6) := 
sorry

end not_equiv_2_pi_six_and_11_pi_six_l968_96829


namespace remainder_of_3_pow_2023_mod_7_l968_96865

theorem remainder_of_3_pow_2023_mod_7 :
  3 ^ 2023 % 7 = 3 :=
by
  sorry

end remainder_of_3_pow_2023_mod_7_l968_96865


namespace decreasing_function_range_l968_96801

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 1) * x + 7 * a - 2 else a ^ x

theorem decreasing_function_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (3 / 8 ≤ a ∧ a < 1 / 2) := 
by
  intro a
  sorry

end decreasing_function_range_l968_96801


namespace circle_radius_tangent_to_circumcircles_l968_96806

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  (a * b * c) / (4 * (Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))))

theorem circle_radius_tangent_to_circumcircles (AB BC CA : ℝ) (H : Point) 
  (h_AB : AB = 13) (h_BC : BC = 14) (h_CA : CA = 15) : 
  (radius : ℝ) = 65 / 16 :=
by
  sorry

end circle_radius_tangent_to_circumcircles_l968_96806


namespace find_b_l968_96805

theorem find_b (a b c : ℝ) (h₁ : c = 3)
  (h₂ : -a / 3 = c)
  (h₃ : -a / 3 = 1 + a + b + c) :
  b = -16 :=
by
  -- The solution steps are not necessary to include here.
  sorry

end find_b_l968_96805


namespace fgf_3_is_299_l968_96890

def f (x : ℕ) : ℕ := 5 * x + 4
def g (x : ℕ) : ℕ := 3 * x + 2
def h : ℕ := 3

theorem fgf_3_is_299 : f (g (f h)) = 299 :=
by
  sorry

end fgf_3_is_299_l968_96890


namespace inequality_proof_l968_96804

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a / b + b / c + c / a) ^ 2 ≥ 3 * (a / c + c / b + b / a) :=
  sorry

end inequality_proof_l968_96804


namespace find_a_b_and_tangent_lines_l968_96877

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 1

theorem find_a_b_and_tangent_lines (a b : ℝ) :
  (3 * (-2 / 3)^2 + 2 * a * (-2 / 3) + b = 0) ∧
  (3 * 1^2 + 2 * a * 1 + b = 0) →
  a = -1 / 2 ∧ b = -2 ∧
  (∀ t : ℝ, f t a b = (t^3 + (a - 1 / 2) * t^2 - 2 * t + 1) → 
     (f t a b - (3 * t^2 - t - 2) * (0 - t) = 1) →
       (3 * t^2 - t - 2 = (t * (3 * (t - t))) ) → 
          ((2 * 0 + f 0 a b) = 1) ∨ (33 * 0 + 16 * 1 - 16 = 1)) :=
sorry

end find_a_b_and_tangent_lines_l968_96877


namespace perfect_square_polynomial_l968_96845

-- Define the polynomial and the conditions
def polynomial (a b : ℚ) := fun x : ℚ => x^4 + x^3 + 2 * x^2 + a * x + b

-- The expanded form of a quadratic trinomial squared
def quadratic_square (p q : ℚ) := fun x : ℚ =>
  x^4 + 2 * p * x^3 + (p^2 + 2 * q) * x^2 + 2 * p * q * x + q^2

-- Main theorem statement
theorem perfect_square_polynomial :
  ∃ (a b : ℚ), 
  (∀ x : ℚ, polynomial a b x = (quadratic_square (1/2 : ℚ) (7/8 : ℚ) x)) ↔ 
  a = 7/8 ∧ b = 49/64 :=
by
  sorry

end perfect_square_polynomial_l968_96845


namespace jackie_apples_l968_96833

variable (A J : ℕ)

-- Condition: Adam has 3 more apples than Jackie.
axiom h1 : A = J + 3

-- Condition: Adam has 9 apples.
axiom h2 : A = 9

-- Question: How many apples does Jackie have?
theorem jackie_apples : J = 6 :=
by
  -- We would normally the proof steps here, but we'll skip to the answer
  sorry

end jackie_apples_l968_96833


namespace proof_problem_l968_96853

theorem proof_problem (a b c : ℝ) (h : a > b) (h1 : b > c) :
  (1 / (a - b) + 1 / (b - c) + 4 / (c - a) ≥ 0) :=
sorry

end proof_problem_l968_96853


namespace max_k_guarded_l968_96826

-- Define the size of the board
def board_size : ℕ := 8

-- Define the directions a guard can look
inductive Direction
| up | down | left | right

-- Define a guard's position on the board as a pair of Fin 8
def Position := Fin board_size × Fin board_size

-- Guard record that contains its position and direction
structure Guard where
  pos : Position
  dir : Direction

-- Function to determine if guard A is guarding guard B
def is_guarding (a b : Guard) : Bool :=
  match a.dir with
  | Direction.up    => a.pos.1 < b.pos.1 ∧ a.pos.2 = b.pos.2
  | Direction.down  => a.pos.1 > b.pos.1 ∧ a.pos.2 = b.pos.2
  | Direction.left  => a.pos.1 = b.pos.1 ∧ a.pos.2 > b.pos.2
  | Direction.right => a.pos.1 = b.pos.1 ∧ a.pos.2 < b.pos.2

-- The main theorem states that the maximum k is 5
theorem max_k_guarded : ∃ k : ℕ, (∀ g : Guard, ∃ S : Finset Guard, (S.card ≥ k) ∧ (∀ s ∈ S, is_guarding s g)) ∧ k = 5 :=
by
  sorry

end max_k_guarded_l968_96826


namespace train_length_is_350_meters_l968_96825

noncomputable def length_of_train (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let time_hr := time_sec / 3600
  speed_kmh * time_hr * 1000

theorem train_length_is_350_meters :
  length_of_train 60 21 = 350 :=
by
  sorry

end train_length_is_350_meters_l968_96825


namespace point_A_2019_pos_l968_96810

noncomputable def A : ℕ → ℤ
| 0       => 2
| (n + 1) =>
    if (n + 1) % 2 = 1 then A n - (n + 1)
    else A n + (n + 1)

theorem point_A_2019_pos : A 2019 = -1008 := by
  sorry

end point_A_2019_pos_l968_96810


namespace point_inside_circle_l968_96802

theorem point_inside_circle (m : ℝ) : (1 - 2)^2 + (-3 + 1)^2 < m → m > 5 :=
by
  sorry

end point_inside_circle_l968_96802


namespace tan_problem_l968_96820

theorem tan_problem (m : ℝ) (α : ℝ) (h1 : Real.tan α = m / 3) (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
sorry

end tan_problem_l968_96820


namespace sarah_age_l968_96873

variable (s m : ℕ)

theorem sarah_age (h1 : s = m - 18) (h2 : s + m = 50) : s = 16 :=
by {
  -- The proof will go here
  sorry
}

end sarah_age_l968_96873


namespace minimum_participants_l968_96892

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l968_96892


namespace jon_buys_2_coffees_each_day_l968_96868

-- Define the conditions
def cost_per_coffee : ℕ := 2
def total_spent : ℕ := 120
def days_in_april : ℕ := 30

-- Define the total number of coffees bought
def total_coffees_bought : ℕ := total_spent / cost_per_coffee

-- Prove that Jon buys 2 coffees each day
theorem jon_buys_2_coffees_each_day : total_coffees_bought / days_in_april = 2 := by
  sorry

end jon_buys_2_coffees_each_day_l968_96868


namespace sqrt_of_8_l968_96883

-- Definition of square root
def isSquareRoot (x : ℝ) (a : ℝ) : Prop := x * x = a

-- Theorem statement: The square root of 8 is ±√8
theorem sqrt_of_8 :
  ∃ x : ℝ, isSquareRoot x 8 ∧ (x = Real.sqrt 8 ∨ x = -Real.sqrt 8) :=
by
  sorry

end sqrt_of_8_l968_96883


namespace isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm_l968_96882

def isosceles_right_triangle_initial_leg_length (x : ℝ) (h : ℝ) : Prop :=
  x + 4 * ((x + 4) / 2) ^ 2 = x * x / 2 + 112 

def isosceles_right_triangle_legs_correct (a b : ℝ) (h : ℝ) : Prop :=
  a = 26 ∧ b = 26 * Real.sqrt 2

theorem isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm :
  ∃ (x : ℝ) (h : ℝ), isosceles_right_triangle_initial_leg_length x h ∧ 
                       isosceles_right_triangle_legs_correct x (x * Real.sqrt 2) h := 
by
  sorry

end isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm_l968_96882


namespace six_letter_vowel_words_count_l968_96821

noncomputable def vowel_count_six_letter_words : Nat := 27^6

theorem six_letter_vowel_words_count :
  vowel_count_six_letter_words = 531441 :=
  by
    sorry

end six_letter_vowel_words_count_l968_96821


namespace ab_square_value_l968_96809

noncomputable def cyclic_quadrilateral (AX AY BX BY CX CY AB2 : ℝ) : Prop :=
  AX * AY = 6 ∧
  BX * BY = 5 ∧
  CX * CY = 4 ∧
  AB2 = 122 / 15

theorem ab_square_value :
  ∃ (AX AY BX BY CX CY : ℝ), cyclic_quadrilateral AX AY BX BY CX CY (122 / 15) :=
by
  sorry

end ab_square_value_l968_96809


namespace increasing_sequence_a1_range_l968_96811

theorem increasing_sequence_a1_range
  (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) = (4 * a n - 2) / (a n + 1))
  (strictly_increasing : ∀ n, a (n + 1) > a n) :
  1 < a 1 ∧ a 1 < 2 :=
sorry

end increasing_sequence_a1_range_l968_96811


namespace focus_of_parabola_l968_96803

theorem focus_of_parabola (x y : ℝ) : x^2 = 4 * y → (0, 1) = (0, (4 / 4)) :=
by
  sorry

end focus_of_parabola_l968_96803


namespace tile_ratio_l968_96889

theorem tile_ratio (original_black_tiles : ℕ) (original_white_tiles : ℕ) (original_width : ℕ) (original_height : ℕ) (border_width : ℕ) (border_height : ℕ) :
  original_black_tiles = 10 ∧ original_white_tiles = 22 ∧ original_width = 8 ∧ original_height = 4 ∧ border_width = 2 ∧ border_height = 2 →
  (original_black_tiles + ( (original_width + 2 * border_width) * (original_height + 2 * border_height) - original_width * original_height ) ) / original_white_tiles = 19 / 11 :=
by
  -- sorry to skip the proof
  sorry

end tile_ratio_l968_96889


namespace smallest_number_ending_in_9_divisible_by_13_l968_96869

theorem smallest_number_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), (n % 10 = 9) ∧ (13 ∣ n) ∧ (∀ (m : ℕ), (m % 10 = 9) ∧ (13 ∣ m) ∧ (m < n) -> false) :=
sorry

end smallest_number_ending_in_9_divisible_by_13_l968_96869


namespace faye_age_l968_96817

def ages (C D E F : ℕ) :=
  D = E - 2 ∧
  E = C + 3 ∧
  F = C + 4 ∧
  D = 15

theorem faye_age (C D E F : ℕ) (h : ages C D E F) : F = 18 :=
by
  unfold ages at h
  sorry

end faye_age_l968_96817


namespace son_present_age_l968_96800

variable (S F : ℕ)

-- Define the conditions
def fatherAgeCondition := F = S + 35
def twoYearsCondition := F + 2 = 2 * (S + 2)

-- The proof theorem
theorem son_present_age : 
  fatherAgeCondition S F → 
  twoYearsCondition S F → 
  S = 33 :=
by
  intros h1 h2
  sorry

end son_present_age_l968_96800


namespace balance_pitcher_with_saucers_l968_96881

-- Define the weights of the cup (C), pitcher (P), and saucer (S)
variables (C P S : ℝ)

-- Conditions provided in the problem
axiom cond1 : 2 * C + 2 * P = 14 * S
axiom cond2 : P = C + S

-- The statement to prove
theorem balance_pitcher_with_saucers : P = 4 * S :=
by
  sorry

end balance_pitcher_with_saucers_l968_96881


namespace find_y_for_slope_l968_96847

theorem find_y_for_slope (y : ℝ) :
  let R := (-3, 9)
  let S := (3, y)
  let slope := (S.2 - R.2) / (S.1 - R.1)
  slope = -2 ↔ y = -3 :=
by
  simp [slope]
  sorry

end find_y_for_slope_l968_96847


namespace daily_wage_of_c_l968_96832

theorem daily_wage_of_c 
  (a_days : ℕ) (b_days : ℕ) (c_days : ℕ) 
  (wage_ratio_a_b : ℚ) (wage_ratio_b_c : ℚ) 
  (total_earnings : ℚ) 
  (A : ℚ) (C : ℚ) :
  a_days = 6 →
  b_days = 9 →
  c_days = 4 →
  wage_ratio_a_b = 3 / 4 →
  wage_ratio_b_c = 4 / 5 →
  total_earnings = 1850 →
  A = 75 →
  C = 208.33 := 
sorry

end daily_wage_of_c_l968_96832


namespace part1_part2_l968_96831

open Set Real

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem part1 (m : ℝ) (h : Disjoint A (B m)) : m ∈ Iio 2 ∪ Ioi 4 := 
sorry

theorem part2 (m : ℝ) (h : A ∪ (univ \ (B m)) = univ) : m ∈ Iic 3 := 
sorry

end part1_part2_l968_96831


namespace initial_blue_balls_proof_l968_96898

-- Define the main problem parameters and condition
def initial_jars (total_balls initial_blue_balls removed_blue probability remaining_balls : ℕ) :=
  total_balls = 18 ∧
  removed_blue = 3 ∧
  remaining_balls = total_balls - removed_blue ∧
  probability = 1/5 → 
  (initial_blue_balls - removed_blue) / remaining_balls = probability

-- Define the proof problem
theorem initial_blue_balls_proof (total_balls initial_blue_balls removed_blue probability remaining_balls : ℕ) :
  initial_jars total_balls initial_blue_balls removed_blue probability remaining_balls →
  initial_blue_balls = 6 :=
by
  sorry

end initial_blue_balls_proof_l968_96898


namespace m_squared_divisible_by_64_l968_96856

theorem m_squared_divisible_by_64 (m : ℕ) (h : 8 ∣ m) : 64 ∣ m * m :=
sorry

end m_squared_divisible_by_64_l968_96856


namespace books_per_shelf_l968_96836

theorem books_per_shelf (total_books : ℕ) (books_taken : ℕ) (shelves : ℕ) (books_left : ℕ) (books_per_shelf : ℕ) :
  total_books = 46 →
  books_taken = 10 →
  shelves = 9 →
  books_left = total_books - books_taken →
  books_per_shelf = books_left / shelves →
  books_per_shelf = 4 :=
by
  sorry

end books_per_shelf_l968_96836


namespace number_of_possible_outcomes_l968_96888

theorem number_of_possible_outcomes : 
  ∃ n : ℕ, n = 30 ∧
  ∀ (total_shots successful_shots consecutive_hits : ℕ),
  total_shots = 8 ∧ successful_shots = 3 ∧ consecutive_hits = 2 →
  n = 30 := 
by
  sorry

end number_of_possible_outcomes_l968_96888


namespace interval_x_2x_3x_l968_96839

theorem interval_x_2x_3x (x : ℝ) :
  (2 * x > 1) ∧ (2 * x < 2) ∧ (3 * x > 1) ∧ (3 * x < 2) ↔ (x > 1 / 2) ∧ (x < 2 / 3) :=
by
  sorry

end interval_x_2x_3x_l968_96839


namespace minimize_side_length_of_triangle_l968_96875

-- Define a triangle with sides a, b, and c and angle C
structure Triangle :=
  (a b c : ℝ)
  (C : ℝ) -- angle C in radians
  (area : ℝ) -- area of the triangle

-- Define the conditions for the problem
def conditions (T : Triangle) : Prop :=
  T.area > 0 ∧ T.C > 0 ∧ T.C < Real.pi

-- Define the desired result
def min_side_length (T : Triangle) : Prop :=
  T.a = T.b ∧ T.a = Real.sqrt ((2 * T.area) / Real.sin T.C)

-- The theorem to be proven
theorem minimize_side_length_of_triangle (T : Triangle) (h : conditions T) : min_side_length T :=
  sorry

end minimize_side_length_of_triangle_l968_96875


namespace modular_units_l968_96870

theorem modular_units (U N S : ℕ) 
  (h1 : N = S / 4)
  (h2 : (S : ℚ) / (S + U * N) = 0.14285714285714285) : 
  U = 24 :=
by
  sorry

end modular_units_l968_96870


namespace fraction_transformation_l968_96861

theorem fraction_transformation (a b : ℝ) (h : a ≠ b) : 
  (-a) / (a - b) = a / (b - a) :=
sorry

end fraction_transformation_l968_96861


namespace part1_part2_l968_96879

theorem part1 : 2 * (-1)^3 - (-2)^2 / 4 + 10 = 7 := by
  sorry

theorem part2 : abs (-3) - (-6 + 4) / (-1 / 2)^3 + (-1)^2013 = -14 := by
  sorry

end part1_part2_l968_96879


namespace time_to_drain_l968_96835

theorem time_to_drain (V R C : ℝ) (hV : V = 75000) (hR : R = 60) (hC : C = 0.80) : 
  (V * C) / R = 1000 := by
  sorry

end time_to_drain_l968_96835


namespace cos_B_in_third_quadrant_l968_96854

theorem cos_B_in_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB: Real.sin B = 5 / 13) : Real.cos B = - 12 / 13 := by
  sorry

end cos_B_in_third_quadrant_l968_96854


namespace point_in_second_quadrant_l968_96842

-- Define the point coordinates in the Cartesian plane
def x_coord : ℤ := -8
def y_coord : ℤ := 2

-- Define the quadrants based on coordinate conditions
def first_quadrant : Prop := x_coord > 0 ∧ y_coord > 0
def second_quadrant : Prop := x_coord < 0 ∧ y_coord > 0
def third_quadrant : Prop := x_coord < 0 ∧ y_coord < 0
def fourth_quadrant : Prop := x_coord > 0 ∧ y_coord < 0

-- Proof statement: The point (-8, 2) lies in the second quadrant
theorem point_in_second_quadrant : second_quadrant :=
by
  sorry

end point_in_second_quadrant_l968_96842


namespace total_protest_days_l968_96893

theorem total_protest_days (d1 : ℕ) (increase_percent : ℕ) (d2 : ℕ) (total_days : ℕ) (h1 : d1 = 4) (h2 : increase_percent = 25) (h3 : d2 = d1 + (d1 * increase_percent / 100)) : total_days = d1 + d2 → total_days = 9 :=
by
  intros
  sorry

end total_protest_days_l968_96893


namespace exponent_property_l968_96866

theorem exponent_property : 3000 * 3000^2500 = 3000^2501 := 
by sorry

end exponent_property_l968_96866


namespace students_attended_game_l968_96855

variable (s n : ℕ)

theorem students_attended_game (h1 : s + n = 3000) (h2 : 10 * s + 15 * n = 36250) : s = 1750 := by
  sorry

end students_attended_game_l968_96855


namespace gcd_subtraction_method_gcd_euclidean_algorithm_l968_96860

theorem gcd_subtraction_method (a b : ℕ) (h₁ : a = 72) (h₂ : b = 168) : Int.gcd a b = 24 := by
  sorry

theorem gcd_euclidean_algorithm (a b : ℕ) (h₁ : a = 98) (h₂ : b = 280) : Int.gcd a b = 14 := by
  sorry

end gcd_subtraction_method_gcd_euclidean_algorithm_l968_96860


namespace quadrant_of_angle_l968_96852

-- Definitions for conditions
def sin_pos_cos_pos (α : ℝ) : Prop := (Real.sin α) * (Real.cos α) > 0

-- The theorem to prove
theorem quadrant_of_angle (α : ℝ) (h : sin_pos_cos_pos α) : 
  (0 < α ∧ α < π / 2) ∨ (π < α ∧ α < 3 * π / 2) :=
sorry

end quadrant_of_angle_l968_96852


namespace problem_solution_l968_96813

noncomputable def f1 (x : ℝ) : ℝ := -2 * x + 2 * Real.sqrt 2
noncomputable def f2 (x : ℝ) : ℝ := Real.sin x
noncomputable def f3 (x : ℝ) : ℝ := x + (1 / x)
noncomputable def f4 (x : ℝ) : ℝ := Real.exp x
noncomputable def f5 (x : ℝ) : ℝ := -2 * Real.log x

def has_inverse_proportion_point (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∃ x ∈ domain, x * f x = 1

theorem problem_solution :
  (has_inverse_proportion_point f1 univ) ∧
  (has_inverse_proportion_point f2 (Set.Icc 0 (2 * Real.pi))) ∧
  ¬ (has_inverse_proportion_point f3 (Set.Ioi 0)) ∧
  (has_inverse_proportion_point f4 univ) ∧
  ¬ (has_inverse_proportion_point f5 (Set.Ioi 0)) :=
by
  sorry

end problem_solution_l968_96813


namespace equivalent_statements_l968_96878

variables (P Q : Prop)

theorem equivalent_statements (h : P → Q) : 
  ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) ↔ (P → Q) := by
sorry

end equivalent_statements_l968_96878


namespace minimum_empty_cells_face_move_minimum_empty_cells_diagonal_move_l968_96862

-- Definition for Problem Part (a)
def box_dimensions := (3, 5, 7)
def initial_cockchafers := 3 * 5 * 7 -- or 105

-- Defining the theorem for part (a)
theorem minimum_empty_cells_face_move (d : (ℕ × ℕ × ℕ)) (n : ℕ) :
  d = box_dimensions →
  n = initial_cockchafers →
  ∃ k ≥ 1, k = 1 :=
by
  intros hdim hn
  sorry

-- Definition for Problem Part (b)
def row_odd_cells := 2 * 5 * 7  
def row_even_cells := 1 * 5 * 7  

-- Defining the theorem for part (b)
theorem minimum_empty_cells_diagonal_move (r_odd r_even : ℕ) :
  r_odd = row_odd_cells →
  r_even = row_even_cells →
  ∃ m ≥ 35, m = 35 :=
by
  intros ho he
  sorry

end minimum_empty_cells_face_move_minimum_empty_cells_diagonal_move_l968_96862


namespace age_of_b_l968_96895

theorem age_of_b (A B C : ℕ) (h₁ : (A + B + C) / 3 = 25) (h₂ : (A + C) / 2 = 29) : B = 17 := 
by
  sorry

end age_of_b_l968_96895


namespace vectors_coplanar_l968_96838

/-- Vectors defined as 3-dimensional Euclidean space vectors. --/
def vector3 := (ℝ × ℝ × ℝ)

/-- Definitions for vectors a, b, c as given in the problem conditions. --/
def a : vector3 := (3, 1, -1)
def b : vector3 := (1, 0, -1)
def c : vector3 := (8, 3, -2)

/-- The scalar triple product of vectors a, b, c is the determinant of the matrix formed. --/
noncomputable def scalarTripleProduct (u v w : vector3) : ℝ :=
  let (u1, u2, u3) := u
  let (v1, v2, v3) := v
  let (w1, w2, w3) := w
  u1 * (v2 * w3 - v3 * w2) - u2 * (v1 * w3 - v3 * w1) + u3 * (v1 * w2 - v2 * w1)

/-- Statement to prove that vectors a, b, c are coplanar (i.e., their scalar triple product is zero). --/
theorem vectors_coplanar : scalarTripleProduct a b c = 0 :=
  by sorry

end vectors_coplanar_l968_96838


namespace part_I_part_II_l968_96837

noncomputable def f (x : ℝ) := x * (Real.log x - 1) + Real.log x + 1

theorem part_I :
  let f_tangent (x y : ℝ) := x - y - 1
  (∀ x y, f_tangent x y = 0 ↔ y = x - 1) ∧ f_tangent 1 (f 1) = 0 :=
by
  sorry

theorem part_II (m : ℝ) :
  (∀ x : ℝ, x > 0 → x^2 + x * (m - (Real.log x + 1 / x)) + 1 ≥ 0) → m ≥ -1 :=
by
  sorry

end part_I_part_II_l968_96837


namespace binom_solution_l968_96818

theorem binom_solution (x y : ℕ) (hxy : x > 0 ∧ y > 0) (bin_eq : Nat.choose x y = 1999000) : x = 1999000 ∨ x = 2000 := 
by
  sorry

end binom_solution_l968_96818


namespace scaling_factor_is_2_l968_96849

-- Define the volumes of the original and scaled cubes
def V1 : ℕ := 343
def V2 : ℕ := 2744

-- Assume s1 cubed equals V1 and s2 cubed equals V2
def s1 : ℕ := 7  -- because 7^3 = 343
def s2 : ℕ := 14 -- because 14^3 = 2744

-- Scaling factor between the cubes
def scaling_factor : ℕ := s2 / s1 

-- The theorem stating the scaling factor is 2 given the volumes
theorem scaling_factor_is_2 (h1 : s1 ^ 3 = V1) (h2 : s2 ^ 3 = V2) : scaling_factor = 2 := by
  sorry

end scaling_factor_is_2_l968_96849


namespace find_other_number_l968_96827

theorem find_other_number (b : ℕ) (lcm_val gcd_val : ℕ)
  (h_lcm : Nat.lcm 240 b = 2520)
  (h_gcd : Nat.gcd 240 b = 24) :
  b = 252 :=
sorry

end find_other_number_l968_96827


namespace B_plus_C_is_330_l968_96823

-- Definitions
def A : ℕ := 170
def B : ℕ := 300
def C : ℕ := 30

axiom h1 : A + B + C = 500
axiom h2 : A + C = 200
axiom h3 : C = 30

-- Theorem statement
theorem B_plus_C_is_330 : B + C = 330 :=
by
  sorry

end B_plus_C_is_330_l968_96823


namespace possible_values_of_m_l968_96859

def F1 := (-3, 0)
def F2 := (3, 0)
def possible_vals := [2, -1, 4, -3, 1/2]

noncomputable def is_valid_m (m : ℝ) : Prop :=
  abs (2 * m - 1) < 6 ∧ m ≠ 1/2

theorem possible_values_of_m : {m ∈ possible_vals | is_valid_m m} = {2, -1} := by
  sorry

end possible_values_of_m_l968_96859


namespace maximum_profit_l968_96834

noncomputable def R (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then
  10.8 - (1/30) * x^2
else
  108 / x - 1000 / (3 * x^2)

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then
  x * R x - (10 + 2.7 * x)
else
  x * R x - (10 + 2.7 * x)

theorem maximum_profit : 
  ∃ x : ℝ, (0 < x ∧ x ≤ 10 → W x = 8.1 * x - (x^3 / 30) - 10) ∧ 
           (x > 10 → W x = 98 - 1000 / (3 * x) - 2.7 * x) ∧ 
           (∃ xmax : ℝ, xmax = 9 ∧ W 9 = 38.6) := 
sorry

end maximum_profit_l968_96834


namespace salary_increase_l968_96841

variable (S : ℝ) -- Robert's original salary
variable (P : ℝ) -- Percentage increase after decrease in decimal form

theorem salary_increase (h1 : 0.5 * S * (1 + P) = 0.75 * S) : P = 0.5 := 
by 
  sorry

end salary_increase_l968_96841
