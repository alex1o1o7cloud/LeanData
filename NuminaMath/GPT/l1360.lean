import Mathlib

namespace geom_seq_sum_l1360_136040

variable {a : ℕ → ℝ}

theorem geom_seq_sum (h : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) : a 5 + a 7 = 6 ∨ a 5 + a 7 = -6 := by
  sorry

end geom_seq_sum_l1360_136040


namespace tangent_line_at_point_l1360_136093

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = (2 * x - 1)^3) (h_point : (x, y) = (1, 1)) :
  ∃ m b : ℝ, y = m * x + b ∧ m = 6 ∧ b = -5 :=
by
  sorry

end tangent_line_at_point_l1360_136093


namespace fraction_simplification_addition_l1360_136054

theorem fraction_simplification_addition :
  (∃ a b : ℕ, 0.4375 = (a : ℚ) / b ∧ Nat.gcd a b = 1 ∧ a + b = 23) :=
by
  sorry

end fraction_simplification_addition_l1360_136054


namespace infinite_six_consecutive_epsilon_squarish_l1360_136047

def is_epsilon_squarish (ε : ℝ) (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 < a ∧ a < b ∧ b < (1 + ε) * a ∧ n = a * b

theorem infinite_six_consecutive_epsilon_squarish (ε : ℝ) (hε : 0 < ε) : 
  ∃ (N : ℕ), ∃ (n : ℕ), N ≤ n ∧
  (is_epsilon_squarish ε n) ∧ 
  (is_epsilon_squarish ε (n + 1)) ∧ 
  (is_epsilon_squarish ε (n + 2)) ∧ 
  (is_epsilon_squarish ε (n + 3)) ∧ 
  (is_epsilon_squarish ε (n + 4)) ∧ 
  (is_epsilon_squarish ε (n + 5)) :=
  sorry

end infinite_six_consecutive_epsilon_squarish_l1360_136047


namespace total_pizzas_two_days_l1360_136058

-- Declare variables representing the number of pizzas made by Craig and Heather
variables (c1 c2 h1 h2 : ℕ)

-- Given conditions as definitions
def condition_1 : Prop := h1 = 4 * c1
def condition_2 : Prop := h2 = c2 - 20
def condition_3 : Prop := c1 = 40
def condition_4 : Prop := c2 = c1 + 60

-- Prove that the total number of pizzas made in two days equals 380
theorem total_pizzas_two_days (c1 c2 h1 h2 : ℕ)
  (h_cond1 : condition_1 c1 h1) (h_cond2 : condition_2 h2 c2)
  (h_cond3 : condition_3 c1) (h_cond4 : condition_4 c1 c2) :
  (h1 + c1) + (h2 + c2) = 380 :=
by
  -- As we don't need to provide the proof here, just insert sorry
  sorry

end total_pizzas_two_days_l1360_136058


namespace sneakers_cost_l1360_136019

theorem sneakers_cost (rate_per_yard : ℝ) (num_yards_cut : ℕ) (total_earnings : ℝ) :
  rate_per_yard = 2.15 ∧ num_yards_cut = 6 ∧ total_earnings = rate_per_yard * num_yards_cut → 
  total_earnings = 12.90 :=
by
  sorry

end sneakers_cost_l1360_136019


namespace fourth_number_second_set_l1360_136025

theorem fourth_number_second_set :
  (∃ (x y : ℕ), (28 + x + 42 + 78 + 104) / 5 = 90 ∧ (128 + 255 + 511 + y + x) / 5 = 423 ∧ x = 198) →
  (y = 1023) :=
by
  sorry

end fourth_number_second_set_l1360_136025


namespace percent_fewer_than_50000_is_75_l1360_136084

-- Define the given conditions as hypotheses
variables {P_1 P_2 P_3 P_4 : ℝ}
variable (h1 : P_1 = 0.35)
variable (h2 : P_2 = 0.40)
variable (h3 : P_3 = 0.15)
variable (h4 : P_4 = 0.10)

-- Define the percentage of counties with fewer than 50,000 residents
def percent_fewer_than_50000 (P_1 P_2 : ℝ) : ℝ :=
  P_1 + P_2

-- The theorem statement we need to prove
theorem percent_fewer_than_50000_is_75 (h1 : P_1 = 0.35) (h2 : P_2 = 0.40) :
  percent_fewer_than_50000 P_1 P_2 = 0.75 :=
by
  sorry

end percent_fewer_than_50000_is_75_l1360_136084


namespace incorrect_step_l1360_136038

-- Given conditions
variables {a b : ℝ} (hab : a < b)

-- Proof statement of the incorrect step ③
theorem incorrect_step : ¬ (2 * (a - b) ^ 2 < (a - b) ^ 2) :=
by sorry

end incorrect_step_l1360_136038


namespace candy_problem_l1360_136027

-- Define the given conditions
def numberOfStudents : Nat := 43
def piecesOfCandyPerStudent : Nat := 8

-- Formulate the problem statement
theorem candy_problem : numberOfStudents * piecesOfCandyPerStudent = 344 := by
  sorry

end candy_problem_l1360_136027


namespace find_roots_and_m_l1360_136013

theorem find_roots_and_m (m a : ℝ) (h_root : (-2)^2 - 4 * (-2) + m = 0) :
  m = -12 ∧ a = 6 :=
by
  sorry

end find_roots_and_m_l1360_136013


namespace parallel_vectors_l1360_136000

variables (x : ℝ)

theorem parallel_vectors (h : (1 + x) / 2 = (1 - 3 * x) / -1) : x = 3 / 5 :=
by {
  sorry
}

end parallel_vectors_l1360_136000


namespace number_of_subjects_l1360_136023

variable (P C M : ℝ)

-- Given conditions
def conditions (P C M : ℝ) : Prop :=
  (P + C + M) / 3 = 75 ∧
  (P + M) / 2 = 90 ∧
  (P + C) / 2 = 70 ∧
  P = 95

-- Proposition with given conditions and the conclusion
theorem number_of_subjects (P C M : ℝ) (h : conditions P C M) : 
  (∃ n : ℕ, n = 3) :=
by
  sorry

end number_of_subjects_l1360_136023


namespace polynomial_roots_l1360_136045

theorem polynomial_roots (d e : ℤ) :
  (∀ r, r^2 - 2 * r - 1 = 0 → r^5 - d * r - e = 0) ↔ (d = 29 ∧ e = 12) := by
  sorry

end polynomial_roots_l1360_136045


namespace parabola_equation_l1360_136071

theorem parabola_equation (d : ℝ) (p : ℝ) (x y : ℝ) (h1 : d = 2) (h2 : y = 2) (h3 : x = 1) :
  y^2 = 4 * x :=
sorry

end parabola_equation_l1360_136071


namespace condition_a_condition_b_condition_c_l1360_136066

-- Definitions for conditions
variable {ι : Type*} (f₁ f₂ f₃ f₄ : ι → ℝ) (x : ι)

-- First part: Condition to prove second equation is a consequence of first
theorem condition_a :
  (∀ x, f₁ x * f₄ x = f₂ x * f₃ x) →
  ((f₂ x ≠ 0) ∧ (f₂ x + f₄ x ≠ 0)) →
  (f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) :=
sorry

-- Second part: Condition to prove first equation is a consequence of second
theorem condition_b :
  (∀ x, f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) →
  ((f₄ x ≠ 0) ∧ (f₂ x ≠ 0)) →
  (f₁ x * f₄ x = f₂ x * f₃ x) :=
sorry

-- Third part: Condition for equivalence of the equations
theorem condition_c :
  (∀ x, (f₁ x * f₄ x = f₂ x * f₃ x) ∧ (x ∉ {x | f₂ x + f₄ x = 0})) ↔
  (∀ x, (f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) ∧ (x ∉ {x | f₄ x = 0})) :=
sorry

end condition_a_condition_b_condition_c_l1360_136066


namespace cos_half_pi_minus_2alpha_l1360_136061

open Real

theorem cos_half_pi_minus_2alpha (α : ℝ) (h : sin α - cos α = 1 / 3) : cos (π / 2 - 2 * α) = 8 / 9 :=
sorry

end cos_half_pi_minus_2alpha_l1360_136061


namespace pow_eq_of_pow_sub_eq_l1360_136064

theorem pow_eq_of_pow_sub_eq (a : ℝ) (m n : ℕ) (h1 : a^m = 6) (h2 : a^(m-n) = 2) : a^n = 3 := 
by
  sorry

end pow_eq_of_pow_sub_eq_l1360_136064


namespace find_x_l1360_136021

noncomputable def a : ℝ × ℝ := (3, 4)
noncomputable def b : ℝ × ℝ := (2, 1)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem find_x (x : ℝ) (h : dot_product (a.1 + x * b.1, a.2 + x * b.2) (a.1 - b.1, a.2 - b.2) = 0) : x = -3 :=
  sorry

end find_x_l1360_136021


namespace two_f_x_eq_8_over_4_plus_x_l1360_136041

variable (f : ℝ → ℝ)
variable (x : ℝ)
variables (hx : 0 < x)
variable (h : ∀ x, 0 < x → f (2 * x) = 2 / (2 + x))

theorem two_f_x_eq_8_over_4_plus_x : 2 * f x = 8 / (4 + x) :=
by sorry

end two_f_x_eq_8_over_4_plus_x_l1360_136041


namespace residue_mod_17_l1360_136096

theorem residue_mod_17 : (230 * 15 - 20 * 9 + 5) % 17 = 0 :=
  by
  sorry

end residue_mod_17_l1360_136096


namespace no_non_negative_solutions_l1360_136003

theorem no_non_negative_solutions (a b : ℕ) (h_diff : a ≠ b) (d := Nat.gcd a b) 
                                 (a' := a / d) (b' := b / d) (n := d * (a' * b' - a' - b')) :
  ¬ ∃ x y : ℕ, a * x + b * y = n := 
by
  sorry

end no_non_negative_solutions_l1360_136003


namespace monster_perimeter_l1360_136056

theorem monster_perimeter (r : ℝ) (theta : ℝ) (h₁ : r = 2) (h₂ : theta = 90 * π / 180) :
  2 * r + (3 / 4) * (2 * π * r) = 3 * π + 4 := by
  -- Sorry to skip the proof.
  sorry

end monster_perimeter_l1360_136056


namespace minimum_height_for_surface_area_geq_120_l1360_136081

noncomputable def box_surface_area (x : ℝ) : ℝ :=
  6 * x^2 + 20 * x

theorem minimum_height_for_surface_area_geq_120 :
  ∃ (x : ℝ), (x ≥ 0) ∧ (box_surface_area x ≥ 120) ∧ (x + 5 = 9) := by
  sorry

end minimum_height_for_surface_area_geq_120_l1360_136081


namespace find_range_of_a_l1360_136034

-- Definitions and conditions
def pointA : ℝ × ℝ := (0, 3)
def lineL (x : ℝ) : ℝ := 2 * x - 4
def circleCenter (a : ℝ) : ℝ × ℝ := (a, 2 * a - 4)
def circleRadius : ℝ := 1

-- The range to prove
def valid_range (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 12 / 5

-- Main theorem
theorem find_range_of_a (a : ℝ) (M : ℝ × ℝ)
  (on_circle : (M.1 - (circleCenter a).1)^2 + (M.2 - (circleCenter a).2)^2 = circleRadius^2)
  (condition_MA_MD : (M.1 - pointA.1)^2 + (M.2 - pointA.2)^2 = 4 * M.1^2 + 4 * M.2^2) :
  valid_range a :=
sorry

end find_range_of_a_l1360_136034


namespace circle_area_with_radius_8_l1360_136029

noncomputable def circle_radius : ℝ := 8
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circle_area_with_radius_8 :
  circle_area circle_radius = 64 * Real.pi :=
by
  sorry

end circle_area_with_radius_8_l1360_136029


namespace plain_chips_count_l1360_136088

theorem plain_chips_count (total_chips : ℕ) (BBQ_chips : ℕ)
  (hyp1 : total_chips = 9) (hyp2 : BBQ_chips = 5)
  (hyp3 : (5 * 4 / (2 * 1) : ℚ) / ((9 * 8 * 7) / (3 * 2 * 1)) = 0.11904761904761904) :
  total_chips - BBQ_chips = 4 := by
sorry

end plain_chips_count_l1360_136088


namespace find_coordinates_of_symmetric_point_l1360_136044

def point_on_parabola (A : ℝ × ℝ) : Prop :=
  A.2 = (A.1 - 1)^2 + 2

def symmetric_with_respect_to_axis (A A' : ℝ × ℝ) : Prop :=
  A'.1 = 2 * 1 - A.1 ∧ A'.2 = A.2

def correct_coordinates_of_A' (A' : ℝ × ℝ) : Prop :=
  A' = (3, 6)

theorem find_coordinates_of_symmetric_point (A A' : ℝ × ℝ)
  (hA : A = (-1, 6))
  (h_parabola : point_on_parabola A)
  (h_symmetric : symmetric_with_respect_to_axis A A') :
  correct_coordinates_of_A' A' :=
sorry

end find_coordinates_of_symmetric_point_l1360_136044


namespace harriet_smallest_stickers_l1360_136039

theorem harriet_smallest_stickers 
  (S : ℕ) (a b c : ℕ)
  (h1 : S = 5 * a + 3)
  (h2 : S = 11 * b + 3)
  (h3 : S = 13 * c + 3)
  (h4 : S > 3) :
  S = 718 :=
by
  sorry

end harriet_smallest_stickers_l1360_136039


namespace last_digit_of_a_power_b_l1360_136017

-- Define the constants from the problem
def a : ℕ := 954950230952380948328708
def b : ℕ := 470128749397540235934750230

-- Define a helper function to calculate the last digit of a natural number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Define the main statement to be proven
theorem last_digit_of_a_power_b : last_digit ((last_digit a) ^ (b % 4)) = 4 :=
by
  -- Here go the proof steps if we were to provide them
  sorry

end last_digit_of_a_power_b_l1360_136017


namespace min_visible_pairs_l1360_136048

-- Define the problem conditions
def bird_circle_flock (P : ℕ) : Prop :=
  P = 155

def mutual_visibility_condition (θ : ℝ) : Prop :=
  θ ≤ 10

-- Define the minimum number of mutually visible pairs
def min_mutual_visible_pairs (P_pairs : ℕ) : Prop :=
  P_pairs = 270

-- The main theorem statement
theorem min_visible_pairs (n : ℕ) (θ : ℝ) (P_pairs : ℕ)
  (H1 : bird_circle_flock n)
  (H2 : mutual_visibility_condition θ) :
  min_mutual_visible_pairs P_pairs :=
by
  sorry

end min_visible_pairs_l1360_136048


namespace triangle_property_l1360_136011

theorem triangle_property
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a > b)
  (h2 : a = 5)
  (h3 : c = 6)
  (h4 : Real.sin B = 3 / 5) :
  (b = Real.sqrt 13 ∧ Real.sin A = 3 * Real.sqrt 13 / 13) →
  Real.sin (2 * A + π / 4) = 7 * Real.sqrt 2 / 26 :=
sorry

end triangle_property_l1360_136011


namespace PQ_is_10_5_l1360_136001

noncomputable def PQ_length_proof_problem : Prop := 
  ∃ (PQ : ℝ),
    PQ = 10.5 ∧ 
    ∃ (ST : ℝ) (SU : ℝ),
      ST = 4.5 ∧ SU = 7.5 ∧ 
      ∃ (QR : ℝ) (PR : ℝ),
        QR = 21 ∧ PR = 15 ∧ 
        ∃ (angle_PQR angle_STU : ℝ),
          angle_PQR = 120 ∧ angle_STU = 120 ∧ 
          PQ / ST = PR / SU

theorem PQ_is_10_5 :
  PQ_length_proof_problem := sorry

end PQ_is_10_5_l1360_136001


namespace savings_equal_after_25_weeks_l1360_136007

theorem savings_equal_after_25_weeks (x : ℝ) :
  (160 + 25 * x = 210 + 125) → x = 7 :=
by 
  apply sorry

end savings_equal_after_25_weeks_l1360_136007


namespace domain_of_function_l1360_136077

-- Definitions of the conditions

def sqrt_condition (x : ℝ) : Prop := -x^2 - 3*x + 4 ≥ 0
def log_condition (x : ℝ) : Prop := x + 1 > 0 ∧ x + 1 ≠ 1

-- Statement of the problem

theorem domain_of_function :
  {x : ℝ | sqrt_condition x ∧ log_condition x} = { x | -1 < x ∧ x < 0 ∨ 0 < x ∧ x ≤ 1 } :=
sorry

end domain_of_function_l1360_136077


namespace possible_double_roots_l1360_136067

theorem possible_double_roots (b₃ b₂ b₁ : ℤ) (s : ℤ) :
  s^2 ∣ 50 →
  (Polynomial.eval s (Polynomial.C 50 + Polynomial.C b₁ * Polynomial.X + Polynomial.C b₂ * Polynomial.X^2 + Polynomial.C b₃ * Polynomial.X^3 + Polynomial.X^4) = 0) →
  (Polynomial.eval s (Polynomial.derivative (Polynomial.C 50 + Polynomial.C b₁ * Polynomial.X + Polynomial.C b₂ * Polynomial.X^2 + Polynomial.C b₃ * Polynomial.X^3 + Polynomial.X^4)) = 0) →
  s = 1 ∨ s = -1 ∨ s = 5 ∨ s = -5 :=
by
  sorry

end possible_double_roots_l1360_136067


namespace tan_subtraction_inequality_l1360_136095

theorem tan_subtraction_inequality (x y : ℝ) 
  (hx : 0 < x ∧ x < (π / 2)) 
  (hy : 0 < y ∧ y < (π / 2)) 
  (h : Real.tan x = 3 * Real.tan y) : 
  x - y ≤ π / 6 ∧ (x - y = π / 6 ↔ (x = π / 3 ∧ y = π / 6)) := 
sorry

end tan_subtraction_inequality_l1360_136095


namespace sales_discount_l1360_136062

theorem sales_discount
  (P N : ℝ)  -- original price and number of items sold
  (H1 : (1 - D / 100) * 1.3 = 1.17) -- condition when discount D is applied
  (D : ℝ)  -- sales discount percentage
  : D = 10 := by
  sorry

end sales_discount_l1360_136062


namespace simplify_expression_l1360_136022

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1/3 : ℝ)

theorem simplify_expression :
  (cube_root 512) * (cube_root 343) = 56 := by
  -- conditions
  let h1 : 512 = 2^9 := by rfl
  let h2 : 343 = 7^3 := by rfl
  -- goal
  sorry

end simplify_expression_l1360_136022


namespace solve_for_k_l1360_136016

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k-1) * x + 2

theorem solve_for_k (k : ℝ) :
  (∀ x : ℝ, f k x = f k (-x)) ↔ k = 1 :=
by
  sorry

end solve_for_k_l1360_136016


namespace evaluate_expression_l1360_136060

theorem evaluate_expression : (2^3002 * 3^3004) / (6^3003) = (3 / 2) := by
  sorry

end evaluate_expression_l1360_136060


namespace not_enough_money_l1360_136015

-- Define the prices of the books
def price_animal_world : Real := 21.8
def price_fairy_tale_stories : Real := 19.5

-- Define the total amount of money Xiao Ming has
def xiao_ming_money : Real := 40.0

-- Define the statement we want to prove
theorem not_enough_money : (price_animal_world + price_fairy_tale_stories) > xiao_ming_money := by
  sorry

end not_enough_money_l1360_136015


namespace train_speed_l1360_136078

theorem train_speed (L : ℝ) (T : ℝ) (V_m : ℝ) (V_t : ℝ) : (L = 500) → (T = 29.997600191984642) → (V_m = 5 / 6) → (V_t = (L / T) + V_m) → (V_t * 3.6 = 63) :=
by
  intros hL hT hVm hVt
  simp at hL hT hVm hVt
  sorry

end train_speed_l1360_136078


namespace mass_percentage_H_correct_l1360_136063

noncomputable def mass_percentage_H_in_CaH2 : ℝ :=
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_H : ℝ := 1.01
  let molar_mass_CaH2 : ℝ := molar_mass_Ca + 2 * molar_mass_H
  (2 * molar_mass_H / molar_mass_CaH2) * 100

theorem mass_percentage_H_correct :
  |mass_percentage_H_in_CaH2 - 4.80| < 0.01 :=
by
  sorry

end mass_percentage_H_correct_l1360_136063


namespace rectangle_area_90_l1360_136006

theorem rectangle_area_90 {x y : ℝ} (h1 : (x + 3) * (y - 1) = x * y) (h2 : (x - 3) * (y + 1.5) = x * y) : x * y = 90 := 
  sorry

end rectangle_area_90_l1360_136006


namespace intersection_complement_eq_l1360_136037

def U : Set Int := { -2, -1, 0, 1, 2, 3 }
def M : Set Int := { 0, 1, 2 }
def N : Set Int := { 0, 1, 2, 3 }

noncomputable def C_U (A : Set Int) := U \ A

theorem intersection_complement_eq :
  (C_U M ∩ N) = {3} :=
by
  sorry

end intersection_complement_eq_l1360_136037


namespace Leah_lost_11_dollars_l1360_136086

-- Define the conditions
def LeahEarned : ℕ := 28
def MilkshakeCost : ℕ := LeahEarned / 7
def RemainingAfterMilkshake : ℕ := LeahEarned - MilkshakeCost
def Savings : ℕ := RemainingAfterMilkshake / 2
def WalletAfterSavings : ℕ := RemainingAfterMilkshake - Savings
def WalletAfterDog : ℕ := 1

-- Define the theorem to prove Leah's loss
theorem Leah_lost_11_dollars : WalletAfterSavings - WalletAfterDog = 11 := 
by 
  sorry

end Leah_lost_11_dollars_l1360_136086


namespace jerry_feathers_left_l1360_136082

def hawk_feathers : ℕ := 37
def eagle_feathers : ℝ := 17.5 * hawk_feathers
def total_feathers : ℝ := hawk_feathers + eagle_feathers
def feathers_to_sister : ℝ := 0.45 * total_feathers
def remaining_feathers_after_sister : ℝ := total_feathers - feathers_to_sister
def feathers_sold : ℝ := 0.85 * remaining_feathers_after_sister
def final_remaining_feathers : ℝ := remaining_feathers_after_sister - feathers_sold

theorem jerry_feathers_left : ⌊final_remaining_feathers⌋₊ = 56 := by
  sorry

end jerry_feathers_left_l1360_136082


namespace fabric_nguyen_needs_l1360_136014

-- Definitions for conditions
def fabric_per_pant : ℝ := 8.5
def total_pants : ℝ := 7
def yards_to_feet (yards : ℝ) : ℝ := yards * 3
def fabric_nguyen_has_yards : ℝ := 3.5

-- The proof we need to establish
theorem fabric_nguyen_needs : (total_pants * fabric_per_pant) - (yards_to_feet fabric_nguyen_has_yards) = 49 :=
by
  sorry

end fabric_nguyen_needs_l1360_136014


namespace incorrect_pair_l1360_136035

def roots_of_polynomial (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

theorem incorrect_pair : ¬ ∃ x : ℝ, (y = x - 1 ∧ y = x + 1 ∧ roots_of_polynomial x) :=
by
  sorry

end incorrect_pair_l1360_136035


namespace picture_distance_from_right_end_l1360_136026

def distance_from_right_end_of_wall (wall_width picture_width position_from_left : ℕ) : ℕ := 
  wall_width - (position_from_left + picture_width)

theorem picture_distance_from_right_end :
  ∀ (wall_width picture_width position_from_left : ℕ), 
  wall_width = 24 -> 
  picture_width = 4 -> 
  position_from_left = 5 -> 
  distance_from_right_end_of_wall wall_width picture_width position_from_left = 15 :=
by
  intros wall_width picture_width position_from_left hw hp hp_left
  rw [hw, hp, hp_left]
  sorry

end picture_distance_from_right_end_l1360_136026


namespace smallest_n_exists_square_smallest_n_exists_cube_l1360_136031

open Nat

-- Statement for part (a)
theorem smallest_n_exists_square (n x y : ℕ) : (∀ n x y, (x * (x + n) = y^2) → (∃ (x y : ℕ), n = 3 ∧ (x * (x + 3) = y^2))) := sorry

-- Statement for part (b)
theorem smallest_n_exists_cube (n x y : ℕ) : (∀ n x y, (x * (x + n) = y^3) → (∃ (x y : ℕ), n = 2 ∧ (x * (x + 2) = y^3))) := sorry

end smallest_n_exists_square_smallest_n_exists_cube_l1360_136031


namespace Vasya_birthday_on_Thursday_l1360_136070

-- Define the days of the week enumeration
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define the conditions
def Sunday_is_the_day_after_tomorrow (today : DayOfWeek) : Prop :=
  match today with
  | Friday => true
  | _      => false

def Vasya_birthday_not_on_Sunday (birthday : DayOfWeek) : Prop :=
  birthday ≠ Sunday

def Today_is_next_day_after_birthday (today birthday : DayOfWeek) : Prop :=
  match (today, birthday) with
  | (Friday, Thursday) => true
  | _                  => false

-- The main theorem to prove Vasya's birthday was on Thursday
theorem Vasya_birthday_on_Thursday (birthday today : DayOfWeek) 
  (H1: Sunday_is_the_day_after_tomorrow today) 
  (H2: Vasya_birthday_not_on_Sunday birthday) 
  (H3: Today_is_next_day_after_birthday today birthday) : 
  birthday = Thursday :=
by
  sorry

end Vasya_birthday_on_Thursday_l1360_136070


namespace transform_polynomial_l1360_136080

theorem transform_polynomial (x y : ℝ) (h1 : y = x + 1 / x) (h2 : x^4 - x^3 - 6 * x^2 - x + 1 = 0) :
  x^2 * (y^2 - y - 6) = 0 := 
  sorry

end transform_polynomial_l1360_136080


namespace one_odd_one_even_l1360_136065

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

theorem one_odd_one_even (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prime : is_prime a) (h_eq : a^2 + b^2 = c^2) : 
(is_odd b ∧ is_even c) ∨ (is_even b ∧ is_odd c) :=
sorry

end one_odd_one_even_l1360_136065


namespace min_value_of_function_l1360_136012

theorem min_value_of_function (x : ℝ) (hx : x > 3) :
  (x + (1 / (x - 3))) ≥ 5 :=
sorry

end min_value_of_function_l1360_136012


namespace nina_shoe_payment_l1360_136052

theorem nina_shoe_payment :
  let first_pair_original := 22
  let first_pair_discount := 0.10 * first_pair_original
  let first_pair_discounted := first_pair_original - first_pair_discount
  let first_pair_tax := 0.05 * first_pair_discounted
  let first_pair_final := first_pair_discounted + first_pair_tax

  let second_pair_original := first_pair_original * 1.50
  let second_pair_discount := 0.15 * second_pair_original
  let second_pair_discounted := second_pair_original - second_pair_discount
  let second_pair_tax := 0.07 * second_pair_discounted
  let second_pair_final := second_pair_discounted + second_pair_tax

  let total_payment := first_pair_final + second_pair_final
  total_payment = 50.80 :=
by 
  sorry

end nina_shoe_payment_l1360_136052


namespace test_question_total_l1360_136085

theorem test_question_total
  (total_points : ℕ)
  (points_2q : ℕ)
  (points_4q : ℕ)
  (num_2q : ℕ)
  (num_4q : ℕ)
  (H1 : total_points = 100)
  (H2 : points_2q = 2)
  (H3 : points_4q = 4)
  (H4 : num_2q = 30)
  (H5 : total_points = num_2q * points_2q + num_4q * points_4q) :
  num_2q + num_4q = 40 := 
sorry

end test_question_total_l1360_136085


namespace trig_identity_l1360_136068

theorem trig_identity (α m : ℝ) (h : Real.tan α = m) :
  (Real.sin (π / 4 + α))^2 - (Real.sin (π / 6 - α))^2 - Real.cos (5 * π / 12) * Real.sin (5 * π / 12 - 2 * α) = 2 * m / (1 + m^2) :=
by
  sorry

end trig_identity_l1360_136068


namespace scientific_notation_500_billion_l1360_136079

theorem scientific_notation_500_billion :
  ∃ (a : ℝ), 500000000000 = a * 10 ^ 10 ∧ 1 ≤ a ∧ a < 10 :=
by
  sorry

end scientific_notation_500_billion_l1360_136079


namespace football_team_total_progress_l1360_136055

theorem football_team_total_progress :
  let play1 := -5
  let play2 := 13
  let play3 := -2 * play1
  let play4 := play3 / 2
  play1 + play2 + play3 + play4 = 3 :=
by
  sorry

end football_team_total_progress_l1360_136055


namespace positive_difference_l1360_136020

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 := by
  sorry

end positive_difference_l1360_136020


namespace cos_arcsin_l1360_136072

theorem cos_arcsin (h : 8^2 + 15^2 = 17^2) : Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
by
  have opp_hyp_pos : 0 ≤ 8 / 17 := by norm_num
  have hyp_pos : 0 < 17 := by norm_num
  have opp_le_hyp : 8 / 17 ≤ 1 := by norm_num
  sorry

end cos_arcsin_l1360_136072


namespace inequality_solution_l1360_136049

theorem inequality_solution (x : ℝ) : 
  (x ∈ Set.Iio (-3/4) ∪ Set.Ioc 4 5 ∪ Set.Ioi 5) ↔ 
  (x+2) ≠ 0 ∧ (x-2) ≠ 0 ∧ (4 * (x^2 - 1) * (x-2) - (x+2) * (7 * x - 6)) / (4 * (x+2) * (x-2)) ≥ 0 := 
by
  sorry

end inequality_solution_l1360_136049


namespace tethered_dog_area_comparison_l1360_136089

theorem tethered_dog_area_comparison :
  let fence_radius := 20
  let rope_length := 30
  let arrangement1_area := π * (rope_length ^ 2)
  let tether_distance := 12
  let arrangement2_effective_radius := rope_length - tether_distance
  let arrangement2_full_circle_area := π * (arrangement2_effective_radius ^ 2)
  let arrangement2_additional_area := (1 / 4) * π * (tether_distance ^ 2)
  let arrangement2_total_area := arrangement2_full_circle_area + arrangement2_additional_area
  (arrangement1_area - arrangement2_total_area) = 540 * π := 
by
  sorry

end tethered_dog_area_comparison_l1360_136089


namespace solve_for_k_l1360_136046

theorem solve_for_k : ∃ k : ℤ, 2^5 - 10 = 5^2 + k ∧ k = -3 := by
  sorry

end solve_for_k_l1360_136046


namespace triangle_area_is_18_l1360_136074

noncomputable def area_of_triangle (y_8 y_2_2x y_2_minus_2x : ℝ) : ℝ :=
  let intersect1 : ℝ × ℝ := (3, 8)
  let intersect2 : ℝ × ℝ := (-3, 8)
  let intersect3 : ℝ × ℝ := (0, 2)
  let base := 3 - -3
  let height := 8 - 2
  (1 / 2 ) * base * height

theorem triangle_area_is_18 : 
  area_of_triangle (8) (2 + 2 * x) (2 - 2 * x) = 18 := 
  by
    sorry

end triangle_area_is_18_l1360_136074


namespace interest_rate_per_annum_l1360_136033

-- Given conditions
variables (BG TD t : ℝ) (FV r : ℝ)
axiom bg_eq : BG = 6
axiom td_eq : TD = 50
axiom t_eq : t = 1
axiom bankers_gain_eq : BG = FV * r * t - (FV - TD) * r * t

-- Proof problem
theorem interest_rate_per_annum : r = 0.12 :=
by sorry

end interest_rate_per_annum_l1360_136033


namespace fraction_simplification_l1360_136092

/-- Given x and y, under the conditions x ≠ 3y and x ≠ -3y, 
we want to prove that (2 * x) / (x ^ 2 - 9 * y ^ 2) - 1 / (x - 3 * y) = 1 / (x + 3 * y). -/
theorem fraction_simplification (x y : ℝ) (h1 : x ≠ 3 * y) (h2 : x ≠ -3 * y) :
  (2 * x) / (x ^ 2 - 9 * y ^ 2) - 1 / (x - 3 * y) = 1 / (x + 3 * y) :=
by
  sorry

end fraction_simplification_l1360_136092


namespace first_part_is_7613_l1360_136091

theorem first_part_is_7613 :
  ∃ (n : ℕ), ∃ (d : ℕ), d = 3 ∧ (761 * 10 + d) * 1000 + 829 = n ∧ (n % 9 = 0) ∧ (761 * 10 + d = 7613) := 
by
  sorry

end first_part_is_7613_l1360_136091


namespace intersection_A_B_l1360_136053

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end intersection_A_B_l1360_136053


namespace number_of_cats_l1360_136075

theorem number_of_cats (c d : ℕ) (h1 : c = 20 + d) (h2 : c + d = 60) : c = 40 :=
sorry

end number_of_cats_l1360_136075


namespace cindy_age_l1360_136090

-- Define the ages involved
variables (C J M G : ℕ)

-- Define the conditions
def jan_age_condition : Prop := J = C + 2
def marcia_age_condition : Prop := M = 2 * J
def greg_age_condition : Prop := G = M + 2
def greg_age_known : Prop := G = 16

-- The statement we need to prove
theorem cindy_age : 
  jan_age_condition C J → 
  marcia_age_condition J M → 
  greg_age_condition M G → 
  greg_age_known G → 
  C = 5 := 
by 
  -- Sorry is used here to skip the proof
  sorry

end cindy_age_l1360_136090


namespace cherries_eaten_l1360_136042

-- Define the number of cherries Oliver had initially
def initial_cherries : ℕ := 16

-- Define the number of cherries Oliver had left after eating some
def left_cherries : ℕ := 6

-- Prove that the difference between the initial and left cherries is 10
theorem cherries_eaten : initial_cherries - left_cherries = 10 := by
  sorry

end cherries_eaten_l1360_136042


namespace sea_horses_count_l1360_136008

theorem sea_horses_count (S P : ℕ) 
  (h1 : S / P = 5 / 11) 
  (h2 : P = S + 85) 
  : S = 70 := sorry

end sea_horses_count_l1360_136008


namespace rational_solutions_for_k_l1360_136076

theorem rational_solutions_for_k :
  ∀ (k : ℕ), k > 0 → 
  (∃ x : ℚ, k * x^2 + 16 * x + k = 0) ↔ k = 8 :=
by
  sorry

end rational_solutions_for_k_l1360_136076


namespace minimum_positive_difference_contains_amounts_of_numbers_on_strips_l1360_136083

theorem minimum_positive_difference_contains_amounts_of_numbers_on_strips (a b c d e f : ℕ) 
  (h1 : a + f = 7) (h2 : b + e = 7) (h3 : c + d = 7) :
  ∃ (min_diff : ℕ), min_diff = 1 :=
by {
  -- The problem guarantees the minimum difference given the conditions.
  sorry
}

end minimum_positive_difference_contains_amounts_of_numbers_on_strips_l1360_136083


namespace find_a_value_l1360_136097

-- Problem statement
theorem find_a_value (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) :
  (∀ x : ℝ, x^2 + 2 * x^10 = a + a1 * (x+1) + a2 * (x+1)^2 + a3 * (x+1)^3 + a4 * (x+1)^4 + a5 * (x+1)^5 + a6 * (x+1)^6 + a7 * (x+1)^7 + a8 * (x+1)^8 + a9 * (x+1)^9 + a10 * (x+1)^(10)) → a = 3 :=
by sorry

end find_a_value_l1360_136097


namespace ratio_x_y_l1360_136032

theorem ratio_x_y (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) : x / y = 11 / 6 :=
by
  sorry

end ratio_x_y_l1360_136032


namespace angles_satisfy_system_l1360_136087

theorem angles_satisfy_system (k : ℤ) : 
  let x := Real.pi / 3 + k * Real.pi
  let y := k * Real.pi
  x - y = Real.pi / 3 ∧ Real.tan x - Real.tan y = Real.sqrt 3 := 
by 
  sorry

end angles_satisfy_system_l1360_136087


namespace inequality_holds_l1360_136002

-- Given conditions
variables {a b x y : ℝ}
variables (pos_a : 0 < a) (pos_b : 0 < b) (pos_x : 0 < x) (pos_y : 0 < y)
variable (h : a + b = 1)

-- Goal/Question
theorem inequality_holds : (a * x + b * y) * (b * x + a * y) ≥ x * y :=
by sorry

end inequality_holds_l1360_136002


namespace at_least_one_not_greater_than_minus_four_l1360_136073

theorem at_least_one_not_greater_than_minus_four {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4 :=
sorry

end at_least_one_not_greater_than_minus_four_l1360_136073


namespace gcd_256_180_720_l1360_136028

theorem gcd_256_180_720 : Int.gcd (Int.gcd 256 180) 720 = 36 := by
  sorry

end gcd_256_180_720_l1360_136028


namespace sam_total_money_spent_l1360_136010

def value_of_pennies (n : ℕ) : ℝ := n * 0.01
def value_of_nickels (n : ℕ) : ℝ := n * 0.05
def value_of_dimes (n : ℕ) : ℝ := n * 0.10
def value_of_quarters (n : ℕ) : ℝ := n * 0.25

def total_money_spent : ℝ :=
  (value_of_pennies 5 + value_of_nickels 3) +  -- Monday
  (value_of_dimes 8 + value_of_quarters 4) +   -- Tuesday
  (value_of_nickels 7 + value_of_dimes 10 + value_of_quarters 2) +  -- Wednesday
  (value_of_pennies 20 + value_of_nickels 15 + value_of_dimes 12 + value_of_quarters 6) +  -- Thursday
  (value_of_pennies 45 + value_of_nickels 20 + value_of_dimes 25 + value_of_quarters 10)  -- Friday

theorem sam_total_money_spent : total_money_spent = 14.05 :=
by
  sorry

end sam_total_money_spent_l1360_136010


namespace berries_from_fourth_bush_l1360_136024

def number_of_berries (n : ℕ) : ℕ :=
  match n with
  | 1 => 3
  | 2 => 4
  | 3 => 7
  | 5 => 19
  | _ => sorry  -- Assume the given pattern

theorem berries_from_fourth_bush : number_of_berries 4 = 12 :=
by sorry

end berries_from_fourth_bush_l1360_136024


namespace three_pos_reals_inequality_l1360_136057

open Real

theorem three_pos_reals_inequality 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_eq : a + b + c = a^2 + b^2 + c^2) :
  ((a^2) / (a^2 + b * c) + (b^2) / (b^2 + c * a) + (c^2) / (c^2 + a * b)) ≥ (a + b + c) / 2 :=
by
  sorry

end three_pos_reals_inequality_l1360_136057


namespace which_is_negative_l1360_136050

theorem which_is_negative
    (A : ℤ := 2023)
    (B : ℤ := -2023)
    (C : ℚ := 1/2023)
    (D : ℤ := 0) :
    B < 0 :=
by
  sorry

end which_is_negative_l1360_136050


namespace log3_of_7_eq_ab_l1360_136004

noncomputable def log3_of_2_eq_a (a : ℝ) : Prop := Real.log 2 / Real.log 3 = a
noncomputable def log2_of_7_eq_b (b : ℝ) : Prop := Real.log 7 / Real.log 2 = b

theorem log3_of_7_eq_ab (a b : ℝ) (h1 : log3_of_2_eq_a a) (h2 : log2_of_7_eq_b b) :
  Real.log 7 / Real.log 3 = a * b :=
sorry

end log3_of_7_eq_ab_l1360_136004


namespace difference_is_correct_l1360_136069

-- Definition of the given numbers
def numbers : List ℕ := [44, 16, 2, 77, 241]

-- Define the sum of the numbers
def sum_numbers := numbers.sum

-- Define the average of the numbers
def average := sum_numbers / numbers.length

-- Define the difference between sum and average
def difference := sum_numbers - average

-- The theorem we need to prove
theorem difference_is_correct : difference = 304 := by
  sorry

end difference_is_correct_l1360_136069


namespace part_1_part_2_max_min_part_3_length_AC_part_4_range_a_l1360_136099

-- Conditions
def quadratic (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2 * a * x + 2 * a
def point_A (a : ℝ) : ℝ × ℝ := (-1, quadratic a (-1))
def point_B (a : ℝ) : ℝ × ℝ := (3, quadratic a 3)
def line_EF (a : ℝ) : ℝ × ℝ × ℝ × ℝ := ((a - 1), -1, (2 * a + 3), -1)

-- Statements based on solution
theorem part_1 (a : ℝ) :
  (quadratic a (-1)) = -1 := sorry

theorem part_2_max_min (a : ℝ) : 
  a = 1 → 
  (∀ x, -2 ≤ x ∧ x ≤ 3 → 
    (quadratic 1 1 = 3 ∧ 
     quadratic 1 (-2) = -6 ∧ 
     quadratic 1 3 = -1)) := sorry

theorem part_3_length_AC (a : ℝ) (h : a > -1) :
  abs ((2 * a + 1) - (-1)) = abs ((2 * a + 2)) := sorry

theorem part_4_range_a (a : ℝ) : 
  quadratic a (a-1) = -1 ∧ quadratic a (2 * a + 3) = -1 → 
  a ∈ ({-2, -1} ∪ {b : ℝ | b ≥ 0}) := sorry

end part_1_part_2_max_min_part_3_length_AC_part_4_range_a_l1360_136099


namespace common_area_of_equilateral_triangles_in_unit_square_l1360_136094

theorem common_area_of_equilateral_triangles_in_unit_square
  (unit_square_side_length : ℝ)
  (triangle_side_length : ℝ)
  (common_area : ℝ)
  (h_unit_square : unit_square_side_length = 1)
  (h_triangle_side : triangle_side_length = 1) :
  common_area = -1 :=
by
  sorry

end common_area_of_equilateral_triangles_in_unit_square_l1360_136094


namespace repeating_decimal_to_fraction_l1360_136030

theorem repeating_decimal_to_fraction :
  (0.3 + 0.206) = (5057 / 9990) :=
sorry

end repeating_decimal_to_fraction_l1360_136030


namespace Leroy_min_bail_rate_l1360_136036

noncomputable def min_bailing_rate
    (distance_to_shore : ℝ)
    (leak_rate : ℝ)
    (max_tolerable_water : ℝ)
    (rowing_speed : ℝ)
    : ℝ :=
  let time_to_shore := distance_to_shore / rowing_speed * 60
  let total_water_intake := leak_rate * time_to_shore
  let required_bailing := total_water_intake - max_tolerable_water
  required_bailing / time_to_shore

theorem Leroy_min_bail_rate
    (distance_to_shore : ℝ := 2)
    (leak_rate : ℝ := 15)
    (max_tolerable_water : ℝ := 60)
    (rowing_speed : ℝ := 4)
    : min_bailing_rate 2 15 60 4 = 13 := 
by
  simp [min_bailing_rate]
  sorry

end Leroy_min_bail_rate_l1360_136036


namespace neg_p_neither_sufficient_nor_necessary_l1360_136098

-- Definitions of p and q as described
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

-- Proving that ¬p is neither a sufficient nor necessary condition for q
theorem neg_p_neither_sufficient_nor_necessary (x : ℝ) : 
  ( ¬ p x → q x ) = false ∧ ( q x → ¬ p x ) = false := by
  sorry

end neg_p_neither_sufficient_nor_necessary_l1360_136098


namespace area_ratio_l1360_136051

-- Definitions for the conditions in the problem
variables (PQ QR RP : ℝ) (p q r : ℝ)

-- Conditions
def pq_condition := PQ = 18
def qr_condition := QR = 24
def rp_condition := RP = 30
def pqr_sum := p + q + r = 3 / 4
def pqr_squaresum := p^2 + q^2 + r^2 = 1 / 2

-- Goal statement that the area ratio of triangles XYZ to PQR is 23/32
theorem area_ratio (h1 : PQ = 18) (h2 : QR = 24) (h3 : RP = 30) 
  (h4 : p + q + r = 3 / 4) (h5 : p^2 + q^2 + r^2 = 1 / 2) : 
  ∃ (m n : ℕ), (m + n = 55) ∧ (m / n = 23 / 32) := 
sorry

end area_ratio_l1360_136051


namespace calculate_3_pow_5_mul_6_pow_5_l1360_136009

theorem calculate_3_pow_5_mul_6_pow_5 :
  3^5 * 6^5 = 34012224 := 
by 
  sorry

end calculate_3_pow_5_mul_6_pow_5_l1360_136009


namespace david_average_marks_l1360_136005

-- Define the individual marks
def english_marks : ℕ := 74
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 90
def total_marks : ℕ := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

-- Define the average marks
def average_marks : ℚ := total_marks / num_subjects

-- Assert the average marks calculation
theorem david_average_marks : average_marks = 75.6 := by
  sorry

end david_average_marks_l1360_136005


namespace smaller_investment_value_l1360_136043

theorem smaller_investment_value :
  ∃ (x : ℝ), 0.07 * x + 0.27 * 1500 = 0.22 * (x + 1500) ∧ x = 500 :=
by
  sorry

end smaller_investment_value_l1360_136043


namespace number_of_students_playing_soccer_l1360_136018

variable (total_students boys playing_soccer_girls not_playing_soccer_girls : ℕ)
variable (percentage_boys_playing_soccer : ℕ)

-- Conditions
axiom h1 : total_students = 470
axiom h2 : boys = 300
axiom h3 : not_playing_soccer_girls = 135
axiom h4 : percentage_boys_playing_soccer = 86
axiom h5 : playing_soccer_girls = 470 - 300 - not_playing_soccer_girls

-- Question: Prove that the number of students playing soccer is 250
theorem number_of_students_playing_soccer : 
  (playing_soccer_girls * 100) / (100 - percentage_boys_playing_soccer) = 250 :=
sorry

end number_of_students_playing_soccer_l1360_136018


namespace parrots_count_l1360_136059

theorem parrots_count (p r : ℕ) : 2 * p + 4 * r = 26 → p + r = 10 → p = 7 := by
  intros h1 h2
  sorry

end parrots_count_l1360_136059
