import Mathlib

namespace sin_315_eq_neg_sqrt2_div_2_l47_47515

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l47_47515


namespace problem_statement_l47_47141

variable (x y : ℝ)
variable (h_cond1 : 1 / x + 1 / y = 4)
variable (h_cond2 : x * y - x - y = -7)

theorem problem_statement (h_cond1 : 1 / x + 1 / y = 4) (h_cond2 : x * y - x - y = -7) : 
  x^2 * y + x * y^2 = 196 / 9 := 
sorry

end problem_statement_l47_47141


namespace series_sum_equals_9_over_4_l47_47861

noncomputable def series_sum : ℝ := ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3))

theorem series_sum_equals_9_over_4 :
  series_sum = 9 / 4 :=
sorry

end series_sum_equals_9_over_4_l47_47861


namespace number_of_children_l47_47047

-- Definitions based on the conditions provided in the problem
def total_population : ℕ := 8243
def grown_ups : ℕ := 5256

-- Statement of the proof problem
theorem number_of_children : total_population - grown_ups = 2987 := by
-- This placeholder 'sorry' indicates that the proof is omitted.
sorry

end number_of_children_l47_47047


namespace remainder_when_x_minus_y_div_18_l47_47609

variable (k m : ℤ)
variable (x y : ℤ)
variable (h1 : x = 72 * k + 65)
variable (h2 : y = 54 * m + 22)

theorem remainder_when_x_minus_y_div_18 :
  (x - y) % 18 = 7 := by
sorry

end remainder_when_x_minus_y_div_18_l47_47609


namespace div_add_fraction_l47_47862

theorem div_add_fraction :
  (-75) / (-25) + 1/2 = 7/2 := by
  sorry

end div_add_fraction_l47_47862


namespace total_ninja_stars_l47_47865

variable (e c j : ℕ)
variable (H1 : e = 4) -- Eric has 4 ninja throwing stars
variable (H2 : c = 2 * e) -- Chad has twice as many ninja throwing stars as Eric
variable (H3 : j = c - 2) -- Chad sells 2 ninja stars to Jeff
variable (H4 : j = 6) -- Jeff now has 6 ninja throwing stars

theorem total_ninja_stars :
  e + (c - 2) + 6 = 16 :=
by
  sorry

end total_ninja_stars_l47_47865


namespace solve_system_l47_47037

theorem solve_system (x y : ℝ) :
  (x + 3*y + 3*x*y = -1) ∧ (x^2*y + 3*x*y^2 = -4) →
  (x = -3 ∧ y = -1/3) ∨ (x = -1 ∧ y = -1) ∨ (x = -1 ∧ y = 4/3) ∨ (x = 4 ∧ y = -1/3) :=
by
  sorry

end solve_system_l47_47037


namespace a4_equals_9_l47_47781

variable {a : ℕ → ℝ}

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a4_equals_9 (h_geom : geometric_sequence a)
  (h_roots : ∃ a2 a6 : ℝ, a2^2 - 34 * a2 + 81 = 0 ∧ a6^2 - 34 * a6 + 81 = 0 ∧ a 2 = a2 ∧ a 6 = a6) :
  a 4 = 9 :=
sorry

end a4_equals_9_l47_47781


namespace estimated_watched_students_l47_47200

-- Definitions for the problem conditions
def total_students : ℕ := 3600
def surveyed_students : ℕ := 200
def watched_students : ℕ := 160

-- Problem statement (proof not included yet)
theorem estimated_watched_students :
  total_students * (watched_students / surveyed_students : ℝ) = 2880 := by
  -- skipping proof step
  sorry

end estimated_watched_students_l47_47200


namespace sin_315_degree_l47_47486

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l47_47486


namespace tan_shifted_value_l47_47186

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l47_47186


namespace largest_multiple_of_9_less_than_75_is_72_l47_47415

theorem largest_multiple_of_9_less_than_75_is_72 : 
  ∃ n : ℕ, 9 * n < 75 ∧ ∀ m : ℕ, 9 * m < 75 → 9 * m ≤ 9 * n :=
sorry

end largest_multiple_of_9_less_than_75_is_72_l47_47415


namespace hydras_never_die_l47_47061

def two_hydras_survive (a b : ℕ) : Prop :=
  ∀ n : ℕ, ∀ (a_heads b_heads : ℕ),
    (a_heads = a + n ∗ (5 ∨ 7) - 4 ∗ n) ∧
    (b_heads = b + n ∗ (5 ∨ 7) - 4 ∗ n) → a_heads ≠ b_heads

theorem hydras_never_die :
  two_hydras_survive 2016 2017 :=
by sorry

end hydras_never_die_l47_47061


namespace find_m_l47_47598

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l47_47598


namespace min_small_bottles_needed_l47_47975

theorem min_small_bottles_needed (small_capacity large_capacity : ℕ) 
    (h_small_capacity : small_capacity = 35) (h_large_capacity : large_capacity = 500) : 
    ∃ n, n = 15 ∧ large_capacity <= n * small_capacity :=
by 
  sorry

end min_small_bottles_needed_l47_47975


namespace necessary_condition_l47_47219

theorem necessary_condition (A B C D : Prop) (h1 : A > B → C < D) : A > B → C < D := by
  exact h1 -- This is just a placeholder for the actual hypothesis, a required assumption in our initial problem statement

end necessary_condition_l47_47219


namespace divisibility_by_120_l47_47389

theorem divisibility_by_120 (n : ℕ) : 120 ∣ (n^7 - n^3) :=
sorry

end divisibility_by_120_l47_47389


namespace sufficient_but_not_necessary_condition_l47_47960

theorem sufficient_but_not_necessary_condition (x : ℝ) : 
  (x > 2 → (x-1)^2 > 1) ∧ (∃ (y : ℝ), y ≤ 2 ∧ (y-1)^2 > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l47_47960


namespace initial_rate_of_interest_l47_47818

theorem initial_rate_of_interest (P : ℝ) (R : ℝ) 
  (h1 : 1680 = (P * R * 5) / 100) 
  (h2 : 1680 = (P * 5 * 4) / 100) : 
  R = 4 := 
by 
  sorry

end initial_rate_of_interest_l47_47818


namespace initial_amount_of_money_l47_47437

-- Definitions based on conditions in a)
variables (n : ℚ) -- Bert left the house with n dollars
def after_hardware_store := (3 / 4) * n
def after_dry_cleaners := after_hardware_store - 9
def after_grocery_store := (1 / 2) * after_dry_cleaners
def after_bookstall := (2 / 3) * after_grocery_store
def after_donation := (4 / 5) * after_bookstall

-- Theorem statement
theorem initial_amount_of_money : after_donation = 27 → n = 72 :=
by
  sorry

end initial_amount_of_money_l47_47437


namespace projections_concyclic_l47_47367

noncomputable def concyclicity (A B C D A' C' B' D' : Point) :=
  let A' = orthogonal_projection A B D
  let C' = orthogonal_projection C B D
  let B' = orthogonal_projection B A C
  let D' = orthogonal_projection D A C
  affine_span_circumscribes : ∀ (G : affine_subspace ℝ V), is_regular_polygon G 4 → ∀ (w : ℝ), w > 0 → affine_line.basic (A', B', C', D',G,w)

-- The main theorem
theorem projections_concyclic (A B C D A' C' B' D' : Point) :
  (are_cyclic_points A B C D) → 
  (A' = orthogonal_projection A B D) → 
  (C' = orthogonal_projection C B D) → 
  (B' = orthogonal_projection B A C) → 
  (D' = orthogonal_projection D A C) → 
  are_cyclic_points A' B' C' D' :=
begin
  sorry
end

end projections_concyclic_l47_47367


namespace irreducible_fractions_properties_l47_47562

theorem irreducible_fractions_properties : 
  let f1 := 11 / 2
  let f2 := 11 / 6
  let f3 := 11 / 3
  let reciprocal_sum := (2 / 11) + (6 / 11) + (3 / 11)
  (f1 + f2 + f3 = 11) ∧ (reciprocal_sum = 1) :=
by
  sorry

end irreducible_fractions_properties_l47_47562


namespace grayson_travels_further_l47_47892

noncomputable def grayson_first_part_distance : ℝ := 25 * 1
noncomputable def grayson_second_part_distance : ℝ := 20 * 0.5
noncomputable def total_distance_grayson : ℝ := grayson_first_part_distance + grayson_second_part_distance

noncomputable def total_distance_rudy : ℝ := 10 * 3

theorem grayson_travels_further : (total_distance_grayson - total_distance_rudy) = 5 := by
  sorry

end grayson_travels_further_l47_47892


namespace maximize_profit_maximize_tax_revenue_l47_47344

open Real

-- Conditions
def inverse_demand_function (P Q : ℝ) : Prop :=
  P = 310 - 3 * Q

def production_cost_per_jar : ℝ := 10

-- Part (a): Prove that Q = 50 maximizes profit
theorem maximize_profit : (Q : ℝ) (P : ℝ) (∏ : ℝ -> ℝ) (C : ℝ -> ℝ) 
  (h_inv_demand : inverse_demand_function P Q)
  (h_revenue : (R : ℝ -> ℝ), R Q = P * Q)
  (h_cost : (C : ℝ -> ℝ), C Q = production_cost_per_jar * Q)
  (h_profit : (∏ : ℝ -> ℝ), ∏ Q = R Q - C Q)
  (h_derive : ∀ Q, deriv ∏ Q = 300 - 6 * Q) :
  Q = 50 := sorry

-- Part (b): Prove that t = 150 maximizes tax revenue for the government
theorem maximize_tax_revenue : (Q t : ℝ) (T : ℝ -> ℝ)
  (h_inv_demand : inverse_demand_function (310 - 3 * Q) Q)
  (h_tax_revenue : (T : ℝ -> ℝ), T t = Q * t)
  (h_tax_revenue_simplified : (T : ℝ -> ℝ), T t = (300 * t - t ^ 2) / 6)
  (h_derive_tax_revenue : ∀ t, deriv T t = 50 - t / 3) :
  t = 150 := sorry

end maximize_profit_maximize_tax_revenue_l47_47344


namespace max_value_of_expression_l47_47363

theorem max_value_of_expression (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h_sum : x + y + z = 3) 
  (h_order : x ≥ y ∧ y ≥ z) : 
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 12 :=
sorry

end max_value_of_expression_l47_47363


namespace jerry_age_l47_47015

theorem jerry_age (M J : ℕ) (h1 : M = 4 * J - 8) (h2 : M = 24) : J = 8 :=
by
  sorry

end jerry_age_l47_47015


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l47_47460

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l47_47460


namespace intersection_eq_singleton_l47_47144

-- Defining the sets M and N
def M : Set ℤ := {-1, 1, -2, 2}
def N : Set ℤ := {1, 4}

-- Stating the intersection problem
theorem intersection_eq_singleton :
  M ∩ N = {1} := 
by 
  sorry

end intersection_eq_singleton_l47_47144


namespace sin_cos_75_eq_quarter_l47_47124

theorem sin_cos_75_eq_quarter : (Real.sin (75 * Real.pi / 180)) * (Real.cos (75 * Real.pi / 180)) = 1 / 4 :=
by
  sorry

end sin_cos_75_eq_quarter_l47_47124


namespace f_one_equals_half_f_increasing_l47_47737

noncomputable def f : ℝ → ℝ := sorry

axiom f_add_half (x y : ℝ) : f (x + y) = f x + f y + 1/2

axiom f_half     : f (1/2) = 0

axiom f_positive (x : ℝ) (hx : x > 1/2) : f x > 0

theorem f_one_equals_half : f 1 = 1/2 := 
by 
  sorry

theorem f_increasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 > f x2 := 
by 
  sorry

end f_one_equals_half_f_increasing_l47_47737


namespace simplify_expr_l47_47719

-- Define the condition on b
def condition (b : ℚ) : Prop :=
  b ≠ -1 / 2

-- Define the expression to be evaluated
def expression (b : ℚ) : ℚ :=
  1 - 1 / (1 + b / (1 + b))

-- Define the simplified form
def simplified_expr (b : ℚ) : ℚ :=
  b / (1 + 2 * b)

-- The theorem statement showing the equivalence
theorem simplify_expr (b : ℚ) (h : condition b) : expression b = simplified_expr b :=
by
  sorry

end simplify_expr_l47_47719


namespace husband_additional_payment_l47_47092

theorem husband_additional_payment (total_medical_cost : ℝ) (total_salary : ℝ) 
                                  (half_medical_cost : ℝ) (deduction_from_salary : ℝ) 
                                  (remaining_salary : ℝ) (total_payment : ℝ)
                                  (each_share : ℝ) (amount_paid_by_husband : ℝ) : 
                                  
                                  total_medical_cost = 128 →
                                  total_salary = 160 →
                                  half_medical_cost = total_medical_cost / 2 →
                                  deduction_from_salary = half_medical_cost →
                                  remaining_salary = total_salary - deduction_from_salary →
                                  total_payment = remaining_salary + half_medical_cost →
                                  each_share = total_payment / 2 →
                                  amount_paid_by_husband = 64 →
                                  (each_share - amount_paid_by_husband) = 16 := by
  sorry

end husband_additional_payment_l47_47092


namespace option_C_is_nonnegative_rational_l47_47674

def isNonNegativeRational (x : ℚ) : Prop :=
  x ≥ 0

theorem option_C_is_nonnegative_rational :
  isNonNegativeRational (-( - (4^2 : ℚ))) :=
by
  sorry

end option_C_is_nonnegative_rational_l47_47674


namespace find_x_l47_47303

-- Define x as a function of n, where n is an odd natural number
def x (n : ℕ) (h_odd : n % 2 = 1) : ℕ :=
  6^n + 1

-- Define the main theorem
theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_prime_div : ∀ p, p.Prime → p ∣ x n h_odd → (p = 11 ∨ p = 7 ∨ p = 101)) : x 1 (by norm_num) = 7777 :=
  sorry

end find_x_l47_47303


namespace kyle_money_left_l47_47354

-- Define variables and conditions
variables (d k : ℕ)
variables (has_kyle : k = 3 * d - 12) (has_dave : d = 46)

-- State the theorem to prove 
theorem kyle_money_left (d k : ℕ) (has_kyle : k = 3 * d - 12) (has_dave : d = 46) :
  k - k / 3 = 84 :=
by
  -- Sorry to complete the proof block
  sorry

end kyle_money_left_l47_47354


namespace find_m_l47_47596

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l47_47596


namespace inequality_2_inequality_1_9_l47_47640

variables {a : ℕ → ℝ}

-- Conditions
def non_negative (a : ℕ → ℝ) : Prop := ∀ n, a n ≥ 0
def boundary_zero (a : ℕ → ℝ) : Prop := a 1 = 0 ∧ a 9 = 0
def non_zero_interior (a : ℕ → ℝ) : Prop := ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a i ≠ 0

-- Proof problems
theorem inequality_2 (a : ℕ → ℝ) (h1 : non_negative a) (h2 : boundary_zero a) (h3 : non_zero_interior a) :
  ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i - 1) + a (i + 1) < 2 * a i := sorry

theorem inequality_1_9 (a : ℕ → ℝ) (h1 : non_negative a) (h2 : boundary_zero a) (h3 : non_zero_interior a) :
  ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i - 1) + a (i + 1) < 1.9 * a i := sorry

end inequality_2_inequality_1_9_l47_47640


namespace mass_percentage_of_nitrogen_in_N2O5_l47_47072

noncomputable def atomic_mass_N : ℝ := 14.01
noncomputable def atomic_mass_O : ℝ := 16.00
noncomputable def molar_mass_N2O5 : ℝ := (2 * atomic_mass_N) + (5 * atomic_mass_O)

theorem mass_percentage_of_nitrogen_in_N2O5 : 
  (2 * atomic_mass_N / molar_mass_N2O5 * 100) = 25.94 := 
by 
  sorry

end mass_percentage_of_nitrogen_in_N2O5_l47_47072


namespace factorizable_trinomial_l47_47905

theorem factorizable_trinomial (k : ℤ) : (∃ a b : ℤ, a + b = k ∧ a * b = 5) ↔ (k = 6 ∨ k = -6) :=
by
  sorry

end factorizable_trinomial_l47_47905


namespace fraction_addition_l47_47448

theorem fraction_addition : (2 / 5) + (3 / 8) = 31 / 40 := 
by {
  sorry
}

end fraction_addition_l47_47448


namespace solve_fractional_equation_l47_47390

theorem solve_fractional_equation (x : ℝ) (h : (3 * x + 6) / (x ^ 2 + 5 * x - 6) = (3 - x) / (x - 1)) (hx : x ≠ 1) : x = -4 := 
sorry

end solve_fractional_equation_l47_47390


namespace divisor_of_a_l47_47209

namespace MathProofProblem

-- Define the given problem
variable (a b c d : ℕ) -- Variables representing positive integers

-- Given conditions
variables (h_gcd_ab : Nat.gcd a b = 30)
variables (h_gcd_bc : Nat.gcd b c = 42)
variables (h_gcd_cd : Nat.gcd c d = 66)
variables (h_lcm_cd : Nat.lcm c d = 2772)
variables (h_gcd_da : 100 < Nat.gcd d a ∧ Nat.gcd d a < 150)

-- Target statement to prove
theorem divisor_of_a : 13 ∣ a :=
by
  sorry

end MathProofProblem

end divisor_of_a_l47_47209


namespace find_percent_l47_47335

theorem find_percent (x y z : ℝ) (h1 : z * (x - y) = 0.15 * (x + y)) (h2 : y = 0.25 * x) : 
  z = 0.25 := 
sorry

end find_percent_l47_47335


namespace alice_min_speed_l47_47814

open Real

theorem alice_min_speed (d : ℝ) (bob_speed : ℝ) (alice_delay : ℝ) (alice_time : ℝ) :
  d = 180 → bob_speed = 40 → alice_delay = 0.5 → alice_time = 4 → d / alice_time > (d / bob_speed) - alice_delay →
  d / alice_time > 45 := by
  sorry


end alice_min_speed_l47_47814


namespace valentino_farm_total_birds_l47_47376

-- The definitions/conditions from the problem statement
def chickens := 200
def ducks := 2 * chickens
def turkeys := 3 * ducks

-- The theorem to prove the total number of birds
theorem valentino_farm_total_birds : 
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof is not required, so we use 'sorry'
  sorry

end valentino_farm_total_birds_l47_47376


namespace Amanda_notebooks_l47_47980

theorem Amanda_notebooks (initial ordered lost final : ℕ) 
  (h_initial: initial = 65) 
  (h_ordered: ordered = 23) 
  (h_lost: lost = 14) : 
  final = 74 := 
by 
  -- calculation and proof will go here
  sorry 

end Amanda_notebooks_l47_47980


namespace prob1_prob2_l47_47985

-- Definition and theorems related to the calculations of the given problem.
theorem prob1 : ((-12) - 5 + (-14) - (-39)) = 8 := by 
  sorry

theorem prob2 : (-2^2 * 5 - (-12) / 4 - 4) = -21 := by
  sorry

end prob1_prob2_l47_47985


namespace find_m_l47_47582

open Nat

theorem find_m (m : ℕ) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 60 := 
by
  sorry  -- Proof goes here

end find_m_l47_47582


namespace sin_315_degree_l47_47487

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l47_47487


namespace tan_add_pi_div_three_l47_47173

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l47_47173


namespace periodic_function_property_l47_47361

theorem periodic_function_property
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_period : ∀ x, f (x + 2) = f x)
  (h_def1 : ∀ x, -1 ≤ x ∧ x < 0 → f x = a * x + 1)
  (h_def2 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = (b * x + 2) / (x + 1))
  (h_eq : f (1 / 2) = f (3 / 2)) :
  3 * a + 2 * b = -8 := by
  sorry

end periodic_function_property_l47_47361


namespace find_number_of_girls_l47_47345

variable (G : ℕ)

-- Given conditions
def avg_weight_girls (total_weight_girls : ℕ) : Prop := total_weight_girls = 45 * G
def avg_weight_boys (total_weight_boys : ℕ) : Prop := total_weight_boys = 275
def avg_weight_students (total_weight_students : ℕ) : Prop := total_weight_students = 500

-- Proposition to prove
theorem find_number_of_girls 
  (total_weight_girls : ℕ) 
  (total_weight_boys : ℕ) 
  (total_weight_students : ℕ) 
  (h1 : avg_weight_girls G total_weight_girls)
  (h2 : avg_weight_boys total_weight_boys)
  (h3 : avg_weight_students total_weight_students) : 
  G = 5 :=
by sorry

end find_number_of_girls_l47_47345


namespace rationalize_denominator_l47_47024

theorem rationalize_denominator :
  ∃ A B C : ℤ, A * B * C = 180 ∧
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

end rationalize_denominator_l47_47024


namespace hydras_will_live_l47_47066

noncomputable def hydras_live : Prop :=
  let A_initial := 2016
  let B_initial := 2017
  let possible_growth := {5, 7}
  let weekly_death := 4
  ∀ (weeks : ℕ), 
    let A_heads := A_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    let B_heads := B_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    A_heads ≠ B_heads

theorem hydras_will_live : hydras_live :=
sorry

end hydras_will_live_l47_47066


namespace archibald_percentage_games_won_l47_47107

theorem archibald_percentage_games_won
  (A B F1 F2 : ℝ) -- number of games won by Archibald, his brother, and his two friends
  (total_games : ℝ)
  (A_eq_1_1B : A = 1.1 * B)
  (F_eq_2_1B : F1 + F2 = 2.1 * B)
  (total_games_eq : A + B + F1 + F2 = total_games)
  (total_games_val : total_games = 280) :
  (A / total_games * 100) = 26.19 :=
by
  sorry

end archibald_percentage_games_won_l47_47107


namespace possibleValues_set_l47_47914

noncomputable def possibleValues (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 3) : Set ℝ :=
  {x | x = 1/a + 1/b}

theorem possibleValues_set :
  ∀ a b : ℝ, (0 < a ∧ 0 < b) → (a + b = 3) → possibleValues a b (by sorry) (by sorry) = {x | ∃ y, y ≥ 4/3 ∧ x = y} :=
by
  sorry

end possibleValues_set_l47_47914


namespace bowl_weight_after_refill_l47_47798

-- Define the problem conditions
def empty_bowl_weight : ℕ := 420
def day1_consumption : ℕ := 53
def day2_consumption : ℕ := 76
def day3_consumption : ℕ := 65
def day4_consumption : ℕ := 14

-- Define the total consumption over 4 days
def total_consumption : ℕ :=
  day1_consumption + day2_consumption + day3_consumption + day4_consumption

-- Define the final weight of the bowl after refilling
def final_bowl_weight : ℕ :=
  empty_bowl_weight + total_consumption

-- Statement to prove
theorem bowl_weight_after_refill : final_bowl_weight = 628 := by
  sorry

end bowl_weight_after_refill_l47_47798


namespace sum_of_digits_of_number_of_rows_l47_47851

theorem sum_of_digits_of_number_of_rows :
  ∃ N, (3 * (N * (N + 1) / 2) = 1575) ∧ (Nat.digits 10 N).sum = 8 :=
by
  sorry

end sum_of_digits_of_number_of_rows_l47_47851


namespace alice_age_multiple_sum_l47_47279

theorem alice_age_multiple_sum (B : ℕ) (C : ℕ := 3) (A : ℕ := B + 2) (next_multiple_age : ℕ := A + (3 - (A % 3))) :
  B % C = 0 ∧ A = B + 2 ∧ C = 3 → 
  (next_multiple_age % 3 = 0 ∧
   (next_multiple_age / 10) + (next_multiple_age % 10) = 6) := 
by
  intros h
  sorry

end alice_age_multiple_sum_l47_47279


namespace fraction_addition_l47_47442

theorem fraction_addition (a b c d : ℚ) (ha : a = 2/5) (hb : b = 3/8) (hc : c = 31/40) :
  a + b = c :=
by
  rw [ha, hb, hc]
  -- The proof part is skipped here as per instructions
  sorry

end fraction_addition_l47_47442


namespace stickers_total_l47_47388

theorem stickers_total (ryan_has : Ryan_stickers = 30)
  (steven_has : Steven_stickers = 3 * Ryan_stickers)
  (terry_has : Terry_stickers = Steven_stickers + 20) :
  Ryan_stickers + Steven_stickers + Terry_stickers = 230 :=
sorry

end stickers_total_l47_47388


namespace find_S12_l47_47359

theorem find_S12 (S : ℕ → ℕ) (h1 : S 3 = 6) (h2 : S 9 = 15) : S 12 = 18 :=
by
  sorry

end find_S12_l47_47359


namespace max_band_members_l47_47928

theorem max_band_members (k n m : ℕ) : m = k^2 + 11 → m = n * (n + 9) → m ≤ 112 :=
by
  sorry

end max_band_members_l47_47928


namespace Heather_heavier_than_Emily_l47_47896

def Heather_weight := 87
def Emily_weight := 9

theorem Heather_heavier_than_Emily : (Heather_weight - Emily_weight = 78) :=
by sorry

end Heather_heavier_than_Emily_l47_47896


namespace complement_union_eq_l47_47756

variable (U : Set Int := {-2, -1, 0, 1, 2, 3}) 
variable (A : Set Int := {-1, 0, 1}) 
variable (B : Set Int := {1, 2}) 

theorem complement_union_eq :
  U \ (A ∪ B) = {-2, 3} := by 
  sorry

end complement_union_eq_l47_47756


namespace isosceles_triangle_angles_l47_47342

theorem isosceles_triangle_angles (A B C : ℝ) (h_iso: (A = B) ∨ (B = C) ∨ (A = C)) (angle_A : A = 50) :
  (B = 50) ∨ (B = 65) ∨ (B = 80) :=
by
  sorry

end isosceles_triangle_angles_l47_47342


namespace ben_owes_rachel_l47_47384

theorem ben_owes_rachel :
  let dollars_per_lawn := (13 : ℚ) / 3
  let lawns_mowed := (8 : ℚ) / 5
  let total_owed := (104 : ℚ) / 15
  dollars_per_lawn * lawns_mowed = total_owed := 
by 
  sorry

end ben_owes_rachel_l47_47384


namespace sin_315_degree_is_neg_sqrt_2_div_2_l47_47495

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l47_47495


namespace dust_particles_calculation_l47_47032

theorem dust_particles_calculation (D : ℕ) (swept : ℝ) (left_by_shoes : ℕ) (total_after_walk : ℕ)  
  (h_swept : swept = 9 / 10)
  (h_left_by_shoes : left_by_shoes = 223)
  (h_total_after_walk : total_after_walk = 331)
  (h_equation : (1 - swept) * D + left_by_shoes = total_after_walk) : 
  D = 1080 := 
by
  sorry

end dust_particles_calculation_l47_47032


namespace math_problem_james_ladder_l47_47350

def convertFeetToInches(feet : ℕ) : ℕ :=
  feet * 12

def totalRungSpace(rungLength inchesApart : ℕ) : ℕ :=
  rungLength + inchesApart

def totalRungsRequired(totalHeight rungSpace : ℕ) : ℕ :=
  totalHeight / rungSpace

def totalWoodRequiredInInches(rungsRequired rungLength : ℕ) : ℕ :=
  rungsRequired * rungLength

def convertInchesToFeet(inches : ℕ) : ℕ :=
  inches / 12

def woodRequiredForRungs(feetToClimb rungLength inchesApart : ℕ) : ℕ :=
   convertInchesToFeet
    (totalWoodRequiredInInches
      (totalRungsRequired
        (convertFeetToInches feetToClimb) 
        (totalRungSpace rungLength inchesApart))
      rungLength)

theorem math_problem_james_ladder : 
  woodRequiredForRungs 50 18 6 = 37.5 :=
sorry

end math_problem_james_ladder_l47_47350


namespace grayson_vs_rudy_distance_l47_47895

-- Definitions based on the conditions
def grayson_first_part_distance : Real := 25 * 1
def grayson_second_part_distance : Real := 20 * 0.5
def total_grayson_distance : Real := grayson_first_part_distance + grayson_second_part_distance
def rudy_distance : Real := 10 * 3

-- Theorem stating the problem to be proved
theorem grayson_vs_rudy_distance : total_grayson_distance - rudy_distance = 5 := by
  -- Proof would go here
  sorry

end grayson_vs_rudy_distance_l47_47895


namespace fraction_addition_l47_47453

theorem fraction_addition : (2 / 5 + 3 / 8) = 31 / 40 :=
by
  sorry

end fraction_addition_l47_47453


namespace shaded_area_l47_47780

theorem shaded_area (r : ℝ) (α : ℝ) (β : ℝ) (h1 : r = 4) (h2 : α = 1/2) :
  β = 64 - 16 * Real.pi := by sorry

end shaded_area_l47_47780


namespace inscribed_sphere_radius_l47_47663

theorem inscribed_sphere_radius 
  (a : ℝ) 
  (h_angle : ∀ (lateral_face : ℝ), lateral_face = 60) : 
  ∃ (r : ℝ), r = a * (Real.sqrt 3) / 6 :=
by
  sorry

end inscribed_sphere_radius_l47_47663


namespace find_given_number_l47_47979

theorem find_given_number (x : ℕ) : 10 * x + 2 = 3 * (x + 200000) → x = 85714 :=
by
  sorry

end find_given_number_l47_47979


namespace sin_315_eq_neg_sqrt2_div_2_l47_47527

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47527


namespace katie_more_games_l47_47636

noncomputable def katie_games : ℕ := 57 + 39
noncomputable def friends_games : ℕ := 34
noncomputable def games_difference : ℕ := katie_games - friends_games

theorem katie_more_games : games_difference = 62 :=
by
  -- Proof omitted
  sorry

end katie_more_games_l47_47636


namespace sin_315_degree_is_neg_sqrt_2_div_2_l47_47492

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l47_47492


namespace direction_vector_of_line_l47_47927

noncomputable def direction_vector_of_line_eq : Prop :=
  ∃ u v, ∀ x y, (x / 4) + (y / 2) = 1 → (u, v) = (-2, 1)

theorem direction_vector_of_line :
  direction_vector_of_line_eq := sorry

end direction_vector_of_line_l47_47927


namespace purple_shoes_count_l47_47238

-- Define the conditions
def total_shoes : ℕ := 1250
def blue_shoes : ℕ := 540
def remaining_shoes : ℕ := total_shoes - blue_shoes
def green_shoes := remaining_shoes / 2
def purple_shoes := green_shoes

-- State the theorem to be proven
theorem purple_shoes_count : purple_shoes = 355 := 
by
-- Proof can be filled in here (not needed for the task)
sorry

end purple_shoes_count_l47_47238


namespace no_integer_solutions_for_2891_l47_47336

theorem no_integer_solutions_for_2891 (x y : ℤ) : ¬ (x^3 - 3 * x * y^2 + y^3 = 2891) :=
sorry

end no_integer_solutions_for_2891_l47_47336


namespace complement_union_l47_47751

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- The proof problem statement
theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3}) (hA : A = {-1, 0, 1}) (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := sorry

end complement_union_l47_47751


namespace valentino_farm_total_birds_l47_47377

-- The definitions/conditions from the problem statement
def chickens := 200
def ducks := 2 * chickens
def turkeys := 3 * ducks

-- The theorem to prove the total number of birds
theorem valentino_farm_total_birds : 
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof is not required, so we use 'sorry'
  sorry

end valentino_farm_total_birds_l47_47377


namespace sin_315_eq_neg_sqrt2_div_2_l47_47504

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47504


namespace min_n_constant_term_exists_l47_47362

theorem min_n_constant_term_exists (n : ℕ) (h : 0 < n) :
  (∃ r : ℕ, (2 * n = 3 * r) ∧ n > 0) ↔ n = 3 :=
by
  sorry

end min_n_constant_term_exists_l47_47362


namespace range_of_a_l47_47324

open Real

-- The quadratic expression
def quadratic (a x : ℝ) : ℝ := a*x^2 + 2*x + a

-- The condition of the problem
def quadratic_nonnegative_for_all (a : ℝ) := ∀ x : ℝ, quadratic a x ≥ 0

-- The theorem to be proven
theorem range_of_a (a : ℝ) (h : quadratic_nonnegative_for_all a) : a ≥ 1 :=
sorry

end range_of_a_l47_47324


namespace sin_315_degree_l47_47482

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l47_47482


namespace tan_add_pi_over_3_l47_47165

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l47_47165


namespace prove_side_c_prove_sin_B_prove_area_circumcircle_l47_47347

-- Define the given conditions
def triangle_ABC (a b A : ℝ) : Prop :=
  a = Real.sqrt 7 ∧ b = 2 ∧ A = Real.pi / 3

-- Prove that side 'c' is equal to 3
theorem prove_side_c (h : triangle_ABC a b A) : c = 3 := by
  sorry

-- Prove that sin B is equal to \frac{\sqrt{21}}{7}
theorem prove_sin_B (h : triangle_ABC a b A) : Real.sin B = Real.sqrt 21 / 7 := by
  sorry

-- Prove that the area of the circumcircle is \frac{7\pi}{3}
theorem prove_area_circumcircle (h : triangle_ABC a b A) (R : ℝ) : 
  let circumcircle_area := Real.pi * R^2
  circumcircle_area = 7 * Real.pi / 3 := by
  sorry

end prove_side_c_prove_sin_B_prove_area_circumcircle_l47_47347


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47512

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47512


namespace kyle_money_l47_47356

theorem kyle_money (dave_money : ℕ) (kyle_initial : ℕ) (kyle_remaining : ℕ)
  (h1 : dave_money = 46)
  (h2 : kyle_initial = 3 * dave_money - 12)
  (h3 : kyle_remaining = kyle_initial - kyle_initial / 3) :
  kyle_remaining = 84 :=
by
  -- Define Dave's money and provide the assumption
  let dave_money := 46
  have h1 : dave_money = 46 := rfl

  -- Define Kyle's initial money based on Dave's money
  let kyle_initial := 3 * dave_money - 12
  have h2 : kyle_initial = 3 * dave_money - 12 := rfl

  -- Define Kyle's remaining money after spending one third on snowboarding
  let kyle_remaining := kyle_initial - kyle_initial / 3
  have h3 : kyle_remaining = kyle_initial - kyle_initial / 3 := rfl

  -- Now we prove that Kyle's remaining money is 84
  sorry -- Proof steps omitted

end kyle_money_l47_47356


namespace problem_I_problem_II_l47_47572

def setA : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≤ 0}

theorem problem_I (a : ℝ) : (setB a ⊆ setA) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

theorem problem_II (a : ℝ) : (setA ∩ setB a = {1}) ↔ a ≤ 1 := by
  sorry

end problem_I_problem_II_l47_47572


namespace impossible_result_l47_47870

noncomputable def f (a b : ℝ) (c : ℤ) (x : ℝ) : ℝ :=
  a * Real.sin x + b * x + c

theorem impossible_result (a b : ℝ) (c : ℤ) :
  ¬(f a b c 1 = 1 ∧ f a b c (-1) = 2) :=
by {
  sorry
}

end impossible_result_l47_47870


namespace speed_of_stream_l47_47046

variable (x : ℝ) -- Let the speed of the stream be x kmph

-- Conditions
variable (speed_of_boat_in_still_water : ℝ)
variable (time_upstream_twice_time_downstream : Prop)

-- Given conditions
axiom h1 : speed_of_boat_in_still_water = 48
axiom h2 : time_upstream_twice_time_downstream → 1 / (speed_of_boat_in_still_water - x) = 2 * (1 / (speed_of_boat_in_still_water + x))

-- Theorem to prove
theorem speed_of_stream (h2: time_upstream_twice_time_downstream) : x = 16 := by
  sorry

end speed_of_stream_l47_47046


namespace rationalize_denominator_equals_ABC_product_l47_47027

theorem rationalize_denominator_equals_ABC_product :
  let A := -9
  let B := -4
  let C := 5
  ∀ (x y : ℚ), (x + y * real.sqrt 5) = (2 + real.sqrt 5) * (2 + real.sqrt 5) / ((2 - real.sqrt 5) * (2 + real.sqrt 5)) →
    A * B * C = 180 := sorry

end rationalize_denominator_equals_ABC_product_l47_47027


namespace average_of_remaining_two_numbers_l47_47254

theorem average_of_remaining_two_numbers
  (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 3.9)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.45 :=
sorry

end average_of_remaining_two_numbers_l47_47254


namespace total_handshakes_l47_47944

def gremlins := 30
def pixies := 12
def unfriendly_gremlins := 15
def friendly_gremlins := 15

def handshake_count : Nat :=
  let handshakes_friendly_gremlins := friendly_gremlins * (friendly_gremlins - 1) / 2
  let handshakes_friendly_unfriendly := friendly_gremlins * unfriendly_gremlins
  let handshakes_gremlins_pixies := gremlins * pixies
  handshakes_friendly_gremlins + handshakes_friendly_unfriendly + handshakes_gremlins_pixies

theorem total_handshakes : handshake_count = 690 := by
  sorry

end total_handshakes_l47_47944


namespace exist_directed_graph_two_step_l47_47735

theorem exist_directed_graph_two_step {n : ℕ} (h : n > 4) :
  ∃ G : SimpleGraph (Fin n), 
    (∀ u v : Fin n, u ≠ v → 
      (G.Adj u v ∨ (∃ w : Fin n, u ≠ w ∧ w ≠ v ∧ G.Adj u w ∧ G.Adj w v))) :=
sorry

end exist_directed_graph_two_step_l47_47735


namespace oscar_bus_ride_length_l47_47800

/-- Oscar's bus ride to school is some distance, and Charlie's bus ride is 0.25 mile.
Oscar's bus ride is 0.5 mile longer than Charlie's. Prove that Oscar's bus ride is 0.75 mile. -/
theorem oscar_bus_ride_length (charlie_ride : ℝ) (h1 : charlie_ride = 0.25) 
  (oscar_ride : ℝ) (h2 : oscar_ride = charlie_ride + 0.5) : oscar_ride = 0.75 :=
by sorry

end oscar_bus_ride_length_l47_47800


namespace solution_set_inequality_l47_47121

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ x : ℝ, deriv f x < 1 / 2

theorem solution_set_inequality : {x : ℝ | 0 < x ∧ x < 2} = {x : ℝ | f (Real.log x / Real.log 2) > (Real.log x / Real.log 2 + 1) / 2} :=
by
  sorry

end solution_set_inequality_l47_47121


namespace previous_year_profit_percentage_l47_47908

variables {R P: ℝ}

theorem previous_year_profit_percentage (h1: R > 0)
    (h2: P = 0.1 * R)
    (h3: 0.7 * P = 0.07 * R) :
    (P / R) * 100 = 10 :=
by
  -- Since we have P = 0.1 * R from the conditions and definitions,
  -- it follows straightforwardly that (P / R) * 100 = 10.
  -- We'll continue the proof from here.
  sorry

end previous_year_profit_percentage_l47_47908


namespace opposite_number_l47_47690

theorem opposite_number (n : ℕ) (hn : n = 100) (N : Fin n) (hN : N = 83) :
    ∃ M : Fin n, M = 84 :=
by
  sorry
 
end opposite_number_l47_47690


namespace reinforcement_left_after_days_l47_47427

theorem reinforcement_left_after_days
  (initial_men : ℕ) (initial_days : ℕ) (remaining_days : ℕ) (men_left : ℕ)
  (remaining_men : ℕ) (x : ℕ) :
  initial_men = 400 ∧
  initial_days = 31 ∧
  remaining_days = 8 ∧
  men_left = initial_men - remaining_men ∧
  remaining_men = 200 ∧
  400 * 31 - 400 * x = 200 * 8 →
  x = 27 :=
by
  intros h
  sorry

end reinforcement_left_after_days_l47_47427


namespace ratio_of_cube_sides_l47_47816

theorem ratio_of_cube_sides {a b : ℝ} (h : (6 * a^2) / (6 * b^2) = 16) : a / b = 4 :=
by
  sorry

end ratio_of_cube_sides_l47_47816


namespace ratio_of_interior_to_exterior_angle_in_regular_octagon_l47_47776

theorem ratio_of_interior_to_exterior_angle_in_regular_octagon
  (n : ℕ) (regular_polygon : n = 8) : 
  let interior_angle := ((n - 2) * 180) / n
  let exterior_angle := 360 / n
  (interior_angle / exterior_angle) = 3 :=
by
  sorry

end ratio_of_interior_to_exterior_angle_in_regular_octagon_l47_47776


namespace total_weight_moved_l47_47205

-- Given conditions as definitions
def initial_back_squat : ℝ := 200
def back_squat_increase : ℝ := 50
def front_squat_ratio : ℝ := 0.8
def triple_ratio : ℝ := 0.9

-- Proving the total weight moved for three triples
theorem total_weight_moved : 
  let new_back_squat := initial_back_squat + back_squat_increase in
  let front_squat := new_back_squat * front_squat_ratio in
  let weight_per_triple := front_squat * triple_ratio in
  3 * weight_per_triple = 540 :=
by
  sorry

end total_weight_moved_l47_47205


namespace orange_ratio_l47_47088

theorem orange_ratio (total_oranges : ℕ) (brother_fraction : ℚ) (friend_receives : ℕ)
  (H1 : total_oranges = 12)
  (H2 : friend_receives = 2)
  (H3 : 1 / 4 * ((1 - brother_fraction) * total_oranges) = friend_receives) :
  brother_fraction * total_oranges / total_oranges = 1 / 3 :=
by
  sorry

end orange_ratio_l47_47088


namespace no_such_integers_exist_l47_47071

theorem no_such_integers_exist : ¬ ∃ (n k : ℕ), n > 0 ∧ k > 0 ∧ (n ∣ (k ^ n - 1)) ∧ (n.gcd (k - 1) = 1) :=
by
  sorry

end no_such_integers_exist_l47_47071


namespace sin_315_eq_neg_sqrt2_div_2_l47_47517

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l47_47517


namespace doctor_is_correct_l47_47063

noncomputable theory

def hydra_heads_never_equal : Prop :=
  ∀ (a b : ℕ), 
    a = 2016 ∧ b = 2017 ∧ 
    (∀ n : ℕ, ∃ (a_new b_new : ℕ), 
      (a_new = a + 5 ∨ a_new = a + 7) ∧ 
      (b_new = b + 5 ∨ b_new = b + 7) ∧
      (∀ m : ℕ, m < n → a_new + b_new - 4 * (m + 1) ≠ (a_new + b_new) / 2 * 2)
    ) → 
    ∀ n : ℕ, (a + b) % 2 = 1 ∧ a ≠ b

theorem doctor_is_correct : hydra_heads_never_equal :=
by sorry

end doctor_is_correct_l47_47063


namespace second_dog_miles_per_day_l47_47109

-- Definitions describing conditions
section DogWalk
variable (total_miles_week : ℕ)
variable (first_dog_miles_day : ℕ)
variable (days_in_week : ℕ)

-- Assert conditions given in the problem
def condition1 := total_miles_week = 70
def condition2 := first_dog_miles_day = 2
def condition3 := days_in_week = 7

-- The theorem to prove
theorem second_dog_miles_per_day
  (h1 : condition1 total_miles_week)
  (h2 : condition2 first_dog_miles_day)
  (h3 : condition3 days_in_week) :
  (total_miles_week - days_in_week * first_dog_miles_day) / days_in_week = 8 :=
sorry
end DogWalk

end second_dog_miles_per_day_l47_47109


namespace probability_sum_of_dice_gt_8_l47_47825

noncomputable def probability_sum_gt_8 (event_space : set (ℕ × ℕ)) (desired_event : set (ℕ × ℕ)) (prob : ℝ) : Prop :=
  Prob.event_space event_space →
  Prob.event desired_event →
  Prob.P event_space desired_event = prob

theorem probability_sum_of_dice_gt_8 :
  probability_sum_gt_8 
    ({(r, b) | r ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {3, 6}}) 
    ({(r, b) | (r + b > 8) ∧ (b ∈ {3, 6})}) 
    (5 / 12) :=
sorry

end probability_sum_of_dice_gt_8_l47_47825


namespace scientific_notation_integer_l47_47698

theorem scientific_notation_integer (x : ℝ) (h1 : x > 10) :
  ∃ (A : ℝ) (N : ℤ), (1 ≤ A ∧ A < 10) ∧ x = A * 10^N :=
by
  sorry

end scientific_notation_integer_l47_47698


namespace cost_of_items_l47_47806

theorem cost_of_items (e t b : ℝ) 
    (h1 : 3 * e + 4 * t = 3.20)
    (h2 : 4 * e + 3 * t = 3.50)
    (h3 : 5 * e + 5 * t + 2 * b = 5.70) :
    4 * e + 4 * t + 3 * b = 5.20 :=
by
  sorry

end cost_of_items_l47_47806


namespace oranges_to_friend_is_two_l47_47265

-- Definitions based on the conditions.

def initial_oranges : ℕ := 12

def oranges_to_brother (n : ℕ) : ℕ := n / 3

def remainder_after_brother (n : ℕ) : ℕ := n - oranges_to_brother n

def oranges_to_friend (n : ℕ) : ℕ := remainder_after_brother n / 4

-- Theorem stating the problem to be proven.
theorem oranges_to_friend_is_two : oranges_to_friend initial_oranges = 2 :=
sorry

end oranges_to_friend_is_two_l47_47265


namespace problem200_squared_minus_399_composite_l47_47203

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  ¬ is_prime n

theorem problem200_squared_minus_399_composite : is_composite (200^2 - 399) :=
sorry

end problem200_squared_minus_399_composite_l47_47203


namespace find_m_l47_47597

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l47_47597


namespace wheel_circumferences_satisfy_conditions_l47_47681

def C_f : ℝ := 24
def C_r : ℝ := 18

theorem wheel_circumferences_satisfy_conditions:
  360 / C_f = 360 / C_r + 4 ∧ 360 / (C_f - 3) = 360 / (C_r - 3) + 6 :=
by 
  have h1: 360 / C_f = 360 / C_r + 4 := sorry
  have h2: 360 / (C_f - 3) = 360 / (C_r - 3) + 6 := sorry
  exact ⟨h1, h2⟩

end wheel_circumferences_satisfy_conditions_l47_47681


namespace tom_finishes_in_6_years_l47_47242

/-- Combined program years for BS and Ph.D. -/
def BS_years : ℕ := 3
def PhD_years : ℕ := 5

/-- Total combined program time -/
def total_program_years : ℕ := BS_years + PhD_years

/-- Tom's time multiplier -/
def tom_time_multiplier : ℚ := 3 / 4

/-- Tom's total time to finish the program -/
def tom_total_time : ℚ := tom_time_multiplier * total_program_years

theorem tom_finishes_in_6_years : tom_total_time = 6 := 
by 
  -- implementation of the proof is to be filled in here
  sorry

end tom_finishes_in_6_years_l47_47242


namespace max_distance_circle_to_line_l47_47659

-- Definitions for the circle and line
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 2 * y = 0
def line_eq (x y : ℝ) : Prop := x + y + 2 = 0

-- Proof statement
theorem max_distance_circle_to_line 
  (x y : ℝ)
  (h_circ : circle_eq x y)
  (h_line : ∀ (x y : ℝ), line_eq x y → true) :
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 :=
sorry

end max_distance_circle_to_line_l47_47659


namespace Smiths_Backery_Pies_l47_47229

theorem Smiths_Backery_Pies : 
  ∀ (p : ℕ), (∃ (q : ℕ), q = 16 ∧ p = 4 * q + 6) → p = 70 :=
by
  intros p h
  cases' h with q hq
  cases' hq with hq1 hq2
  rw [hq1] at hq2
  have h_eq : p = 4 * 16 + 6 := hq2
  norm_num at h_eq
  exact h_eq

end Smiths_Backery_Pies_l47_47229


namespace probability_passing_through_C_l47_47924

theorem probability_passing_through_C :
  ∀ (A B C : Finset (Nat × Nat)) (paths_A_to_B via paths_A_to_C plus paths_C_to_B : ℕ),
  paths_A_to_B = Nat.choose 6 3 ∧
  paths_A_to_C = Nat.choose 3 3 2 ∧
  paths_C_to_B = Nat.choose 3 3 1 →
  (paths_A_to_C * paths_C_to_B / paths_A_to_B) = 21 / 32 :=
by
  intros
  sorry

end probability_passing_through_C_l47_47924


namespace problem_series_sum_l47_47548

noncomputable def series_sum : ℝ := ∑' n : ℕ, (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem problem_series_sum :
  series_sum = 1 / 200 :=
sorry

end problem_series_sum_l47_47548


namespace original_selling_price_l47_47677

theorem original_selling_price (P : ℝ) (S : ℝ) (h1 : S = 1.10 * P) (h2 : 1.17 * P = 1.10 * P + 35) : S = 550 := 
by
  sorry

end original_selling_price_l47_47677


namespace sin_315_eq_neg_sqrt2_div_2_l47_47467

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47467


namespace circle_repr_eq_l47_47192

theorem circle_repr_eq (a : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 1 + a = 0) ↔ a < 4 :=
by
  sorry

end circle_repr_eq_l47_47192


namespace pizza_slices_left_l47_47106

theorem pizza_slices_left (initial_slices : ℕ) (people : ℕ) (slices_per_person : ℕ) 
  (h1 : initial_slices = 16) (h2 : people = 6) (h3 : slices_per_person = 2) : 
  initial_slices - people * slices_per_person = 4 := 
by
  sorry

end pizza_slices_left_l47_47106


namespace find_remainder_l47_47638

noncomputable def q (x : ℝ) : ℝ := (x^2010 + x^2009 + x^2008 + x + 1)
noncomputable def s (x : ℝ) := (q x) % (x^3 + 2*x^2 + 3*x + 1)

theorem find_remainder (x : ℝ) : (|s 2011| % 500) = 357 := by
    sorry

end find_remainder_l47_47638


namespace simplify_expr1_simplify_expr2_simplify_expr3_l47_47226

-- 1. Proving (1)(2x^{2})^{3}-x^{2}·x^{4} = 7x^{6}
theorem simplify_expr1 (x : ℝ) : (1 : ℝ) * (2 * x^2)^3 - x^2 * x^4 = 7 * x^6 := 
by 
  sorry

-- 2. Proving (a+b)^{2}-b(2a+b) = a^{2}
theorem simplify_expr2 (a b : ℝ) : (a + b)^2 - b * (2 * a + b) = a^2 := 
by 
  sorry

-- 3. Proving (x+1)(x-1)-x^{2} = -1
theorem simplify_expr3 (x : ℝ) : (x + 1) * (x - 1) - x^2 = -1 :=
by 
  sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l47_47226


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47521

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47521


namespace distinct_ways_to_place_digits_l47_47333

theorem distinct_ways_to_place_digits : 
  let boxes := finset.finrange 5
  let digits := {1, 2, 3, 4}
  ∃ (f : boxes → option (fin 5)), 
    (∀ b, b ∈ digits ∪ {∅} ∧ 
    ∀ d₁ d₂, d₁ ≠ d₂ → f d₁ ≠ f d₂) → (boxes.card.factorial = 120) := sorry

end distinct_ways_to_place_digits_l47_47333


namespace find_b_of_roots_condition_l47_47615

theorem find_b_of_roots_condition
  (α β : ℝ)
  (h1 : α * β = -1)
  (h2 : α + β = -b)
  (h3 : α * β - 2 * α - 2 * β = -11) :
  b = -5 := 
  sorry

end find_b_of_roots_condition_l47_47615


namespace triangle_constructibility_l47_47987

variables (a b c γ : ℝ)

-- definition of the problem conditions
def valid_triangle_constructibility_conditions (a b_c_diff γ : ℝ) : Prop :=
  γ < 90 ∧ b_c_diff < a * Real.cos γ

-- constructibility condition
def is_constructible (a b c γ : ℝ) : Prop :=
  b - c < a * Real.cos γ

-- final theorem statement
theorem triangle_constructibility (a b c γ : ℝ) (h1 : γ < 90) (h2 : b > c) :
  (b - c < a * Real.cos γ) ↔ valid_triangle_constructibility_conditions a (b - c) γ :=
by sorry

end triangle_constructibility_l47_47987


namespace runner_injury_point_l47_47849

-- Define the initial setup conditions
def total_distance := 40
def second_half_time := 10
def first_half_additional_time := 5

-- Prove that given the conditions, the runner injured her foot at 20 miles.
theorem runner_injury_point : 
  ∃ (d v : ℝ), (d = 5 * v) ∧ (total_distance - d = 5 * v) ∧ (10 = second_half_time) ∧ (first_half_additional_time = 5) ∧ (d = 20) :=
by
  sorry

end runner_injury_point_l47_47849


namespace sin_315_eq_neg_sqrt2_div_2_l47_47539

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47539


namespace volume_of_increased_box_l47_47973

theorem volume_of_increased_box {l w h : ℝ} (vol : l * w * h = 4860) (sa : l * w + w * h + l * h = 930) (sum_dim : l + w + h = 56) :
  (l + 2) * (w + 3) * (h + 1) = 5964 :=
by
  sorry

end volume_of_increased_box_l47_47973


namespace find_m_l47_47573

open Nat  

-- Definition of lcm in Lean, if it's not already provided in Mathlib
def lcm (a b : Nat) : Nat := (a * b) / gcd a b

theorem find_m (m : ℕ) (h1 : 0 < m)
    (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180):
    m = 60 :=
sorry

end find_m_l47_47573


namespace find_pairs_l47_47947

/-
Define the conditions:
1. The number of three-digit phone numbers consisting of only odd digits.
2. The number of three-digit phone numbers consisting of only even digits excluding 0.
3. Revenue difference is given by a specific equation.
4. \(X\) and \(Y\) are integers less than 250.
-/
def N₁ : ℕ := 5 * 5 * 5  -- Number of combinations with odd digits (1, 3, 5, 7, 9)
def N₂ : ℕ := 4 * 4 * 4  -- Number of combinations with even digits (2, 4, 6, 8)

-- Main theorem: finding pairs (X, Y) that satisfy the given conditions.
theorem find_pairs (X Y : ℕ) (hX : X < 250) (hY : Y < 250) :
  N₁ * X - N₂ * Y = 5 ↔ (X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205) := 
by {
  sorry
}

end find_pairs_l47_47947


namespace polynomial_expansion_sum_constants_l47_47328

theorem polynomial_expansion_sum_constants :
  ∃ (A B C D : ℤ), ((x - 3) * (4 * x ^ 2 + 2 * x - 7) = A * x ^ 3 + B * x ^ 2 + C * x + D) → A + B + C + D = 2 := 
by
  sorry

end polynomial_expansion_sum_constants_l47_47328


namespace total_canvas_area_l47_47643

theorem total_canvas_area (rect_length rect_width tri1_base tri1_height tri2_base tri2_height : ℕ)
    (h1 : rect_length = 5) (h2 : rect_width = 8)
    (h3 : tri1_base = 3) (h4 : tri1_height = 4)
    (h5 : tri2_base = 4) (h6 : tri2_height = 6) :
    (rect_length * rect_width) + ((tri1_base * tri1_height) / 2) + ((tri2_base * tri2_height) / 2) = 58 := by
  sorry

end total_canvas_area_l47_47643


namespace parametric_inclination_l47_47291

noncomputable def angle_of_inclination (x y : ℝ) : ℝ := 50

theorem parametric_inclination (t : ℝ) (x y : ℝ) :
  x = t * Real.sin 40 → y = -1 + t * Real.cos 40 → angle_of_inclination x y = 50 :=
by
  intros hx hy
  -- This is where the proof would go, but we skip it.
  sorry

end parametric_inclination_l47_47291


namespace preceding_integer_binary_l47_47767

--- The conditions as definitions in Lean 4

def M := 0b101100 -- M is defined as binary '101100' which is decimal 44
def preceding_binary (n : Nat) : Nat := n - 1 -- Define a function to get the preceding integer in binary

--- The proof problem statement in Lean 4
theorem preceding_integer_binary :
  preceding_binary M = 0b101011 :=
by
  sorry

end preceding_integer_binary_l47_47767


namespace kyle_money_left_l47_47353

-- Define variables and conditions
variables (d k : ℕ)
variables (has_kyle : k = 3 * d - 12) (has_dave : d = 46)

-- State the theorem to prove 
theorem kyle_money_left (d k : ℕ) (has_kyle : k = 3 * d - 12) (has_dave : d = 46) :
  k - k / 3 = 84 :=
by
  -- Sorry to complete the proof block
  sorry

end kyle_money_left_l47_47353


namespace Ryan_learning_days_l47_47647

theorem Ryan_learning_days
  (hours_english_per_day : ℕ)
  (hours_chinese_per_day : ℕ)
  (total_hours : ℕ)
  (h1 : hours_english_per_day = 6)
  (h2 : hours_chinese_per_day = 7)
  (h3 : total_hours = 65) :
  total_hours / (hours_english_per_day + hours_chinese_per_day) = 5 := by
  sorry

end Ryan_learning_days_l47_47647


namespace books_ratio_l47_47404

theorem books_ratio (c e : ℕ) (h_ratio : c / e = 2 / 5) (h_sampled : c = 10) : e = 25 :=
by
  sorry

end books_ratio_l47_47404


namespace distance_between_centers_of_circles_l47_47011

theorem distance_between_centers_of_circles (C_1 C_2 : ℝ) : 
  (∀ a : ℝ, (C_1 = a ∧ C_2 = a ∧ (4- a)^2 + (1 - a)^2 = a^2)) → 
  |C_1 - C_2| = 8 :=
by
  sorry

end distance_between_centers_of_circles_l47_47011


namespace rationalize_denominator_l47_47030

theorem rationalize_denominator :
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = -9 - 4 * Real.sqrt 5 :=
by
  -- Commutative field properties and algebraic manipulation will be used here.
  sorry

end rationalize_denominator_l47_47030


namespace train_passes_jogger_in_37_seconds_l47_47966

-- Define the parameters
def jogger_speed_kmph : ℝ := 9
def train_speed_kmph : ℝ := 45
def headstart : ℝ := 250
def train_length : ℝ := 120

-- Convert speeds from km/h to m/s
noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

-- Calculate relative speed in m/s
noncomputable def relative_speed : ℝ :=
  train_speed_mps - jogger_speed_mps

-- Calculate total distance to be covered in meters
def total_distance : ℝ :=
  headstart + train_length

-- Calculate time taken to pass the jogger in seconds
noncomputable def time_to_pass : ℝ :=
  total_distance / relative_speed

theorem train_passes_jogger_in_37_seconds :
  time_to_pass = 37 :=
by
  -- Proof would be here
  sorry

end train_passes_jogger_in_37_seconds_l47_47966


namespace tan_add_pi_over_3_l47_47160

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l47_47160


namespace minimum_transfers_required_l47_47667

def initial_quantities : List ℕ := [2, 12, 12, 12, 12]
def target_quantity := 10
def min_transfers := 4

theorem minimum_transfers_required :
  ∃ transfers : ℕ, transfers = min_transfers ∧
  ∀ quantities : List ℕ, List.sum initial_quantities = List.sum quantities →
  (∀ q ∈ quantities, q = target_quantity) :=
by
  sorry

end minimum_transfers_required_l47_47667


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l47_47461

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l47_47461


namespace carol_first_round_points_l47_47712

theorem carol_first_round_points (P : ℤ) (h1 : P + 6 - 16 = 7) : P = 17 :=
by
  sorry

end carol_first_round_points_l47_47712


namespace tan_shifted_value_l47_47182

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l47_47182


namespace find_softball_players_l47_47774

def total_players : ℕ := 51
def cricket_players : ℕ := 10
def hockey_players : ℕ := 12
def football_players : ℕ := 16

def softball_players : ℕ := total_players - (cricket_players + hockey_players + football_players)

theorem find_softball_players : softball_players = 13 := 
by {
  sorry
}

end find_softball_players_l47_47774


namespace tan_add_pi_over_3_l47_47152

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l47_47152


namespace range_of_a_l47_47658

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ), |x + 3| - |x - 1| ≤ a ^ 2 - 3 * a) ↔ a ≤ -1 ∨ a ≥ 4 :=
by
  sorry

end range_of_a_l47_47658


namespace total_people_on_boats_l47_47260

-- Definitions based on the given conditions
def boats : Nat := 5
def people_per_boat : Nat := 3

-- Theorem statement to prove the total number of people on boats in the lake
theorem total_people_on_boats : boats * people_per_boat = 15 :=
by
  sorry

end total_people_on_boats_l47_47260


namespace fraction_spent_on_raw_material_l47_47269

variable (C : ℝ)
variable (x : ℝ)

theorem fraction_spent_on_raw_material :
  C - x * C - (1/10) * (C * (1 - x)) = 0.675 * C → x = 1/4 :=
by
  sorry

end fraction_spent_on_raw_material_l47_47269


namespace fraction_addition_l47_47447

theorem fraction_addition : (2 / 5) + (3 / 8) = 31 / 40 := 
by {
  sorry
}

end fraction_addition_l47_47447


namespace milburg_children_count_l47_47051

theorem milburg_children_count : 
  ∀ (total_population grown_ups : ℕ), 
  total_population = 8243 ∧ grown_ups = 5256 → 
  total_population - grown_ups = 2987 :=
by 
  intros total_population grown_ups h,
  rcases h with ⟨h1, h2⟩,
  rw [h1, h2],
  exact rfl

end milburg_children_count_l47_47051


namespace sin_315_eq_neg_sqrt2_div_2_l47_47529

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47529


namespace find_x_value_l47_47857

def average_eq_condition (x : ℝ) : Prop :=
  (5050 + x) / 101 = 50 * (x + 1)

theorem find_x_value : ∃ x : ℝ, average_eq_condition x ∧ x = 0 :=
by
  use 0
  sorry

end find_x_value_l47_47857


namespace chair_and_desk_prices_l47_47934

theorem chair_and_desk_prices (c d : ℕ) 
  (h1 : c + d = 115)
  (h2 : d - c = 45) :
  c = 35 ∧ d = 80 := 
by
  sorry

end chair_and_desk_prices_l47_47934


namespace disjunction_false_implies_neg_p_true_neg_p_true_does_not_imply_disjunction_false_l47_47763

variable (p q : Prop)

theorem disjunction_false_implies_neg_p_true (hpq : ¬(p ∨ q)) : ¬p :=
by 
  sorry

theorem neg_p_true_does_not_imply_disjunction_false (hnp : ¬p) : ¬(¬(p ∨ q)) :=
by 
  sorry

end disjunction_false_implies_neg_p_true_neg_p_true_does_not_imply_disjunction_false_l47_47763


namespace complement_union_l47_47750

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- The proof problem statement
theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3}) (hA : A = {-1, 0, 1}) (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := sorry

end complement_union_l47_47750


namespace complement_union_eq_l47_47746

open Set

-- Definition of sets U, A, and B
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- Statement of the problem
theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 3} :=
by sorry

end complement_union_eq_l47_47746


namespace complement_union_eq_l47_47757

variable (U : Set Int := {-2, -1, 0, 1, 2, 3}) 
variable (A : Set Int := {-1, 0, 1}) 
variable (B : Set Int := {1, 2}) 

theorem complement_union_eq :
  U \ (A ∪ B) = {-2, 3} := by 
  sorry

end complement_union_eq_l47_47757


namespace sweaters_to_wash_l47_47920

theorem sweaters_to_wash (pieces_per_load : ℕ) (total_loads : ℕ) (shirts_to_wash : ℕ) 
  (h1 : pieces_per_load = 5) (h2 : total_loads = 9) (h3 : shirts_to_wash = 43) : ℕ :=
  if total_loads * pieces_per_load - shirts_to_wash = 2 then 2 else 0

end sweaters_to_wash_l47_47920


namespace arrangement_count_correct_l47_47327

-- Definitions for conditions
def valid_first_section (arrangement : List Char) : Prop :=
  ∀ (c : Char), c ∈ arrangement.take 4 → c = 'B' ∨ c = 'C' ∨ c = 'D'

def valid_middle_section (arrangement : List Char) : Prop :=
  ∀ (c : Char), c ∈ arrangement.drop 4.take 4 → c = 'A' ∨ c = 'C' ∨ c = 'D'

def valid_last_section (arrangement : List Char) : Prop :=
  ∀ (c : Char), c ∈ arrangement.drop 8 → c = 'A' ∨ c = 'B'

def is_valid_arrangement (arrangement : List Char) : Prop :=
  valid_first_section arrangement ∧ valid_middle_section arrangement ∧ valid_last_section arrangement

def count_valid_arrangements : ℕ :=
  -- Expression calculating the total valid arrangements

-- Main statement to be proven
theorem arrangement_count_correct : count_valid_arrangements = 192 := by
  sorry

end arrangement_count_correct_l47_47327


namespace opposite_point_83_is_84_l47_47692

theorem opposite_point_83_is_84 :
  ∀ (n : ℕ) (k : ℕ) (points : Fin n) 
  (H : n = 100) 
  (H1 : k = 83) 
  (H2 : ∀ k, 1 ≤ k ∧ k ≤ 100 ∧ (∑ i in range k, i < k ↔ ∑ i in range 100, i = k)), 
  ∃ m, m = 84 ∧ diametrically_opposite k m := 
begin
  sorry
end

end opposite_point_83_is_84_l47_47692


namespace evaluate_expression_l47_47990

variable (a : ℝ)
variable (x : ℝ)

theorem evaluate_expression (h : x = a + 9) : x - a + 6 = 15 := by
  sorry

end evaluate_expression_l47_47990


namespace find_m_l47_47601

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l47_47601


namespace framed_painting_ratio_l47_47968

-- Define the conditions and the problem
theorem framed_painting_ratio:
  ∀ (x : ℝ),
    (30 + 2 * x) * (20 + 4 * x) = 1500 →
    (20 + 4 * x) / (30 + 2 * x) = 4 / 5 := 
by sorry

end framed_painting_ratio_l47_47968


namespace simplify_expression_l47_47142

variable (a b : ℤ)

theorem simplify_expression : (a - b) - (3 * (a + b)) - b = a - 8 * b := 
by sorry

end simplify_expression_l47_47142


namespace plates_used_l47_47671

def plates_per_course : ℕ := 2
def courses_breakfast : ℕ := 2
def courses_lunch : ℕ := 2
def courses_dinner : ℕ := 3
def courses_late_snack : ℕ := 3
def courses_per_day : ℕ := courses_breakfast + courses_lunch + courses_dinner + courses_late_snack
def plates_per_day : ℕ := courses_per_day * plates_per_course

def parents_and_siblings_stay : ℕ := 6
def grandparents_stay : ℕ := 4
def cousins_stay : ℕ := 3

def parents_and_siblings_count : ℕ := 5
def grandparents_count : ℕ := 2
def cousins_count : ℕ := 4

def plates_parents_and_siblings : ℕ := parents_and_siblings_count * plates_per_day * parents_and_siblings_stay
def plates_grandparents : ℕ := grandparents_count * plates_per_day * grandparents_stay
def plates_cousins : ℕ := cousins_count * plates_per_day * cousins_stay

def total_plates_used : ℕ := plates_parents_and_siblings + plates_grandparents + plates_cousins

theorem plates_used (expected : ℕ) : total_plates_used = expected :=
by
  sorry

end plates_used_l47_47671


namespace complex_number_in_first_quadrant_l47_47874

open Complex

theorem complex_number_in_first_quadrant 
    (z : ℂ) 
    (h : z = (3 + I) / (1 - 3 * I) + 2) : 
    0 < z.re ∧ 0 < z.im :=
by
  sorry

end complex_number_in_first_quadrant_l47_47874


namespace father_and_daughter_age_l47_47846

-- A father's age is 5 times the daughter's age.
-- In 30 years, the father will be 3 times as old as the daughter.
-- Prove that the daughter's current age is 30 and the father's current age is 150.

theorem father_and_daughter_age :
  ∃ (d f : ℤ), (f = 5 * d) ∧ (f + 30 = 3 * (d + 30)) ∧ (d = 30 ∧ f = 150) :=
by
  sorry

end father_and_daughter_age_l47_47846


namespace rationalize_denominator_ABC_l47_47029

theorem rationalize_denominator_ABC :
  (let A := -9 in let B := -4 in let C := 5 in A * B * C) = 180 :=
by
  let expr := (2 + Real.sqrt 5) / (2 - Real.sqrt 5)
  let numerator := (2 + Real.sqrt 5) * (2 + Real.sqrt 5)
  let denominator := (2 - Real.sqrt 5) * (2 + Real.sqrt 5)
  let simplified_expr := numerator / denominator
  have h1 : numerator = 9 + 4 * Real.sqrt 5 := sorry
  have h2 : denominator = -1 := sorry
  have h3 : simplified_expr = -9 - 4 * Real.sqrt 5 := by
    rw [h1, h2]
    simp
  have hA : -9 = A := rfl
  have hB : -4 = B := rfl
  have hC : 5 = C := rfl
  have hABC : -9 * -4 * 5 = 180 := by 
    rw [hA, hB, hC]
    ring
  exact hABC

end rationalize_denominator_ABC_l47_47029


namespace simple_interest_l47_47701

theorem simple_interest (P R T : ℝ) (hP : P = 8965) (hR : R = 9) (hT : T = 5) : 
    (P * R * T) / 100 = 806.85 := 
by 
  sorry

end simple_interest_l47_47701


namespace find_y_at_neg3_l47_47873

noncomputable def quadratic_solution (y x a b : ℝ) : Prop :=
  y = x ^ 2 + a * x + b

theorem find_y_at_neg3
    (a b : ℝ)
    (h1 : 1 + a + b = 2)
    (h2 : 4 - 2 * a + b = -1)
    : quadratic_solution 2 (-3) a b :=
by
  sorry

end find_y_at_neg3_l47_47873


namespace A_remaining_time_equals_B_remaining_time_l47_47428

variable (d_A d_B remaining_Distance_A remaining_Time_A remaining_Distance_B remaining_Time_B total_Distance : ℝ)

-- Given conditions as definitions
def A_traveled_more : d_A = d_B + 180 := sorry
def total_distance_between_X_Y : total_Distance = 900 := sorry
def sum_distance_traveled : d_A + d_B = total_Distance := sorry
def B_remaining_time : remaining_Time_B = 4.5 := sorry
def B_remaining_distance : remaining_Distance_B = total_Distance - d_B := sorry

-- Prove that: A travels the same remaining distance in the same time as B
theorem A_remaining_time_equals_B_remaining_time :
  remaining_Distance_A = remaining_Distance_B ∧ remaining_Time_A = remaining_Time_B := sorry

end A_remaining_time_equals_B_remaining_time_l47_47428


namespace difference_of_smallest_integers_l47_47995

theorem difference_of_smallest_integers (n_1 n_2: ℕ) (h1 : ∀ k, 2 ≤ k ∧ k ≤ 6 → (n_1 > 1 ∧ n_1 % k = 1)) (h2 : ∀ k, 2 ≤ k ∧ k ≤ 6 → (n_2 > 1 ∧ n_2 % k = 1)) (h_smallest : n_1 = 61) (h_second_smallest : n_2 = 121) : n_2 - n_1 = 60 :=
by
  sorry

end difference_of_smallest_integers_l47_47995


namespace complement_union_l47_47752

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- The proof problem statement
theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3}) (hA : A = {-1, 0, 1}) (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := sorry

end complement_union_l47_47752


namespace exchange_rate_5_CAD_to_JPY_l47_47423

theorem exchange_rate_5_CAD_to_JPY :
  (1 : ℝ) * 85 * 5 = 425 :=
by
  sorry

end exchange_rate_5_CAD_to_JPY_l47_47423


namespace cookie_cost_proof_l47_47300

def cost_per_cookie (total_spent : ℕ) (days : ℕ) (cookies_per_day : ℕ) : ℕ :=
  total_spent / (days * cookies_per_day)

theorem cookie_cost_proof : cost_per_cookie 1395 31 3 = 15 := by
  sorry

end cookie_cost_proof_l47_47300


namespace eleven_twelve_divisible_by_133_l47_47803

theorem eleven_twelve_divisible_by_133 (n : ℕ) (h : n > 0) : 133 ∣ (11^(n+2) + 12^(2*n+1)) := 
by 
  sorry

end eleven_twelve_divisible_by_133_l47_47803


namespace milburg_children_count_l47_47052

theorem milburg_children_count : 
  ∀ (total_population grown_ups : ℕ), 
  total_population = 8243 ∧ grown_ups = 5256 → 
  total_population - grown_ups = 2987 :=
by 
  intros total_population grown_ups h,
  rcases h with ⟨h1, h2⟩,
  rw [h1, h2],
  exact rfl

end milburg_children_count_l47_47052


namespace place_pawns_distinct_5x5_l47_47569

noncomputable def number_of_ways_place_pawns : ℕ :=
  5 * 4 * 3 * 2 * 1 * 120

theorem place_pawns_distinct_5x5 : number_of_ways_place_pawns = 14400 := by
  sorry

end place_pawns_distinct_5x5_l47_47569


namespace max_value_of_expression_l47_47364

theorem max_value_of_expression (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h_sum : x + y + z = 3) 
  (h_order : x ≥ y ∧ y ≥ z) : 
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 12 :=
sorry

end max_value_of_expression_l47_47364


namespace sin_315_degree_is_neg_sqrt_2_div_2_l47_47497

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l47_47497


namespace boy_lap_time_l47_47148

noncomputable def muddy_speed : ℝ := 5 * 1000 / 3600
noncomputable def sandy_speed : ℝ := 7 * 1000 / 3600
noncomputable def uphill_speed : ℝ := 4 * 1000 / 3600

noncomputable def muddy_distance : ℝ := 10
noncomputable def sandy_distance : ℝ := 15
noncomputable def uphill_distance : ℝ := 10

noncomputable def time_for_muddy : ℝ := muddy_distance / muddy_speed
noncomputable def time_for_sandy : ℝ := sandy_distance / sandy_speed
noncomputable def time_for_uphill : ℝ := uphill_distance / uphill_speed

noncomputable def total_time_for_one_side : ℝ := time_for_muddy + time_for_sandy + time_for_uphill
noncomputable def total_time_for_lap : ℝ := 4 * total_time_for_one_side

theorem boy_lap_time : total_time_for_lap = 95.656 := by
  sorry

end boy_lap_time_l47_47148


namespace number_of_results_l47_47054

theorem number_of_results (n : ℕ)
  (avg_all : (summation : ℤ) → summation / n = 42)
  (avg_first_5 : (sum_first_5 : ℤ) → sum_first_5 / 5 = 49)
  (avg_last_7 : (sum_last_7 : ℤ) → sum_last_7 / 7 = 52)
  (fifth_result : (r5 : ℤ) → r5 = 147) :
  n = 11 :=
by
  -- Conditions
  let sum_first_5 := 5 * 49
  let sum_last_7 := 7 * 52
  let summed_results := sum_first_5 + sum_last_7 - 147
  let sum_all := 42 * n 
  -- Since sum of all results = 42n
  exact sorry

end number_of_results_l47_47054


namespace find_m_l47_47574

open Nat  

-- Definition of lcm in Lean, if it's not already provided in Mathlib
def lcm (a b : Nat) : Nat := (a * b) / gcd a b

theorem find_m (m : ℕ) (h1 : 0 < m)
    (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180):
    m = 60 :=
sorry

end find_m_l47_47574


namespace trapezoid_shorter_base_l47_47817

theorem trapezoid_shorter_base (L : ℝ) (S : ℝ) (m : ℝ)
  (hL : L = 100)
  (hm : m = 4)
  (h : m = (L - S) / 2) :
  S = 92 :=
by {
  sorry -- Proof is not required
}

end trapezoid_shorter_base_l47_47817


namespace maximize_S_l47_47397

-- Define the sets and the problem statement
def a1 := 2
def a2 := 3
def a3 := 4
def b1 := 5
def b2 := 6
def b3 := 7
def c1 := 8
def c2 := 9
def c3 := 10

def maxS (a1 a2 a3 b1 b2 b3 c1 c2 c3: ℕ) : ℕ :=
  a1 * a2 * a3 + b1 * b2 * b3 + c1 * c2 * c3

theorem maximize_S : 
  maxS 8 9 10 5 6 7 2 3 4 = 954 :=
by sorry

end maximize_S_l47_47397


namespace count_seven_digit_palindromes_l47_47128

theorem count_seven_digit_palindromes : 
  let palindromes := { n : ℤ | ∃ (a b c d : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
    n = 1000000 * a + 100000 * b + 10000 * c + 1000 * d + 100 * c + 10 * b + a } in
  ∃ (count : ℕ), count = 9000 ∧ count = palindromes.card :=
by
  -- the proof goes here
  sorry

end count_seven_digit_palindromes_l47_47128


namespace wendy_points_earned_l47_47827

-- Define the conditions
def points_per_bag : ℕ := 5
def total_bags : ℕ := 11
def bags_not_recycled : ℕ := 2

-- Define the statement to be proved
theorem wendy_points_earned : (total_bags - bags_not_recycled) * points_per_bag = 45 :=
by
  sorry

end wendy_points_earned_l47_47827


namespace value_of_expression_l47_47879

variables (m n c d : ℝ)
variables (h1 : m = -n) (h2 : c * d = 1)

theorem value_of_expression : m + n + 3 * c * d - 10 = -7 :=
by sorry

end value_of_expression_l47_47879


namespace shares_proportion_l47_47432

theorem shares_proportion (C D : ℕ) (h1 : D = 1500) (h2 : C = D + 500) : C / Nat.gcd C D = 4 ∧ D / Nat.gcd C D = 3 := by
  sorry

end shares_proportion_l47_47432


namespace divisible_by_9_l47_47953

theorem divisible_by_9 (k : ℕ) (h : k > 0) : 9 ∣ 3 * (2 + 7^k) :=
sorry

end divisible_by_9_l47_47953


namespace sandwich_cost_proof_l47_47783

/-- Definitions of ingredient costs and quantities. --/
def bread_cost : ℝ := 0.15
def ham_cost : ℝ := 0.25
def cheese_cost : ℝ := 0.35
def mayo_cost : ℝ := 0.10
def lettuce_cost : ℝ := 0.05
def tomato_cost : ℝ := 0.08

def num_bread_slices : ℕ := 2
def num_ham_slices : ℕ := 2
def num_cheese_slices : ℕ := 2
def num_mayo_tbsp : ℕ := 1
def num_lettuce_leaf : ℕ := 1
def num_tomato_slices : ℕ := 2

/-- Calculation of the total cost in dollars and conversion to cents. --/
def sandwich_cost_in_dollars : ℝ :=
  (num_bread_slices * bread_cost) + 
  (num_ham_slices * ham_cost) + 
  (num_cheese_slices * cheese_cost) + 
  (num_mayo_tbsp * mayo_cost) + 
  (num_lettuce_leaf * lettuce_cost) + 
  (num_tomato_slices * tomato_cost)

def sandwich_cost_in_cents : ℝ :=
  sandwich_cost_in_dollars * 100

/-- Prove that the cost of the sandwich in cents is 181. --/
theorem sandwich_cost_proof : sandwich_cost_in_cents = 181 := by
  sorry

end sandwich_cost_proof_l47_47783


namespace pet_food_weight_in_ounces_l47_47645

-- Define the given conditions
def cat_food_bags := 2
def cat_food_weight_per_bag := 3 -- in pounds
def dog_food_bags := 2
def additional_dog_food_weight := 2 -- additional weight per bag compared to cat food
def pounds_to_ounces := 16

-- Calculate the total weight of cat food in pounds
def total_cat_food_weight := cat_food_bags * cat_food_weight_per_bag

-- Calculate the weight of each bag of dog food in pounds
def dog_food_weight_per_bag := cat_food_weight_per_bag + additional_dog_food_weight

-- Calculate the total weight of dog food in pounds
def total_dog_food_weight := dog_food_bags * dog_food_weight_per_bag

-- Calculate the total weight of pet food in pounds
def total_pet_food_weight_pounds := total_cat_food_weight + total_dog_food_weight

-- Convert the total weight to ounces
def total_pet_food_weight_ounces := total_pet_food_weight_pounds * pounds_to_ounces

-- Statement of the problem in Lean 4
theorem pet_food_weight_in_ounces : total_pet_food_weight_ounces = 256 := by
  sorry

end pet_food_weight_in_ounces_l47_47645


namespace range_of_b_l47_47323

theorem range_of_b {b : ℝ} (h_b_ne_zero : b ≠ 0) :
  (∃ x : ℝ, (0 ≤ x ∧ x ≤ 3) ∧ (2 * x + b = 3)) ↔ -3 ≤ b ∧ b ≤ 3 ∧ b ≠ 0 :=
by
  sorry

end range_of_b_l47_47323


namespace jerry_age_proof_l47_47915

variable (J : ℝ)

/-- Mickey's age is 4 years less than 400% of Jerry's age. Mickey is 18 years old. Prove that Jerry is 5.5 years old. -/
theorem jerry_age_proof (h : 18 = 4 * J - 4) : J = 5.5 :=
by
  sorry

end jerry_age_proof_l47_47915


namespace sum_infinite_series_l47_47456

theorem sum_infinite_series : 
  ∑' n : ℕ, (2 * (n + 1) + 3) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2) * ((n + 1) + 3)) = 9 / 4 := by
  sorry

end sum_infinite_series_l47_47456


namespace equation_B_is_quadratic_l47_47077

theorem equation_B_is_quadratic : ∀ y : ℝ, ∃ A B C : ℝ, (5 * y ^ 2 - 5 * y = 0) ∧ A ≠ 0 :=
by
  sorry

end equation_B_is_quadratic_l47_47077


namespace fraction_addition_l47_47451

theorem fraction_addition : (2 / 5 + 3 / 8) = 31 / 40 :=
by
  sorry

end fraction_addition_l47_47451


namespace central_angle_l47_47624

-- Definition: percentage corresponds to central angle
def percentage_equal_ratio (P : ℝ) (θ : ℝ) : Prop :=
  P = θ / 360

-- Theorem statement: Given that P = θ / 360, we want to prove θ = 360 * P
theorem central_angle (P θ : ℝ) (h : percentage_equal_ratio P θ) : θ = 360 * P :=
sorry

end central_angle_l47_47624


namespace prime_number_five_greater_than_perfect_square_l47_47248

theorem prime_number_five_greater_than_perfect_square 
(p x : ℤ) (h1 : p - 5 = x^2) (h2 : p + 9 = (x + 1)^2) : 
  p = 41 :=
sorry

end prime_number_five_greater_than_perfect_square_l47_47248


namespace hoursWorkedPerDay_l47_47634

-- Define the conditions
def widgetsPerHour := 20
def daysPerWeek := 5
def totalWidgetsPerWeek := 800

-- Theorem statement
theorem hoursWorkedPerDay : (totalWidgetsPerWeek / widgetsPerHour) / daysPerWeek = 8 := 
  sorry

end hoursWorkedPerDay_l47_47634


namespace sin_315_eq_neg_sqrt2_div_2_l47_47518

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l47_47518


namespace find_prob_real_roots_l47_47101

-- Define the polynomial q(x)
def q (a : ℝ) (x : ℝ) : ℝ := x^4 + 3*a*x^3 + (3*a - 5)*x^2 + (-6*a + 4)*x - 3

-- Define the conditions for a to ensure all roots of the polynomial are real
noncomputable def all_roots_real_condition (a : ℝ) : Prop :=
  a ≤ -1/3 ∨ 1 ≤ a

-- Define the probability that given a in the interval [-12, 32] all q's roots are real
noncomputable def probability_real_roots : ℝ :=
  let total_length := 32 - (-12)
  let excluded_interval_length := 1 - (-1/3)
  let valid_interval_length := total_length - excluded_interval_length
  valid_interval_length / total_length

-- State the theorem
theorem find_prob_real_roots :
  probability_real_roots = 32 / 33 :=
sorry

end find_prob_real_roots_l47_47101


namespace division_theorem_l47_47723

variable (x : ℤ)

def dividend := 8 * x ^ 4 + 7 * x ^ 3 + 3 * x ^ 2 - 5 * x - 8
def divisor := x - 1
def quotient := 8 * x ^ 3 + 15 * x ^ 2 + 18 * x + 13
def remainder := 5

theorem division_theorem : dividend x = divisor x * quotient x + remainder := by
  sorry

end division_theorem_l47_47723


namespace cannot_determine_number_of_pens_l47_47098

theorem cannot_determine_number_of_pens 
  (P : ℚ) -- marked price of one pen
  (N : ℕ) -- number of pens = 46
  (discount : ℚ := 0.01) -- 1% discount
  (profit_percent : ℚ := 11.91304347826087) -- given profit percent
  : ¬ ∃ (N : ℕ), 
        profit_percent = ((N * P * (1 - discount) - N * P) / (N * P)) * 100 :=
by
  sorry

end cannot_determine_number_of_pens_l47_47098


namespace pavan_travel_distance_l47_47802

theorem pavan_travel_distance (t : ℝ) (v1 v2 : ℝ) (D : ℝ) (h₁ : t = 15) (h₂ : v1 = 30) (h₃ : v2 = 25):
  (D / 2) / v1 + (D / 2) / v2 = t → D = 2250 / 11 :=
by
  intro h
  rw [h₁, h₂, h₃] at h
  sorry

end pavan_travel_distance_l47_47802


namespace remainder_mod_7_l47_47956

theorem remainder_mod_7 (n m p : ℕ) 
  (h₁ : n % 4 = 3)
  (h₂ : m % 7 = 5)
  (h₃ : p % 5 = 2) :
  (7 * n + 3 * m - p) % 7 = 6 :=
by
  sorry

end remainder_mod_7_l47_47956


namespace number_of_rows_of_red_notes_l47_47628

theorem number_of_rows_of_red_notes (R : ℕ) :
  let red_notes_in_each_row := 6
  let blue_notes_per_red_note := 2
  let additional_blue_notes := 10
  let total_notes := 100
  (6 * R + 12 * R + 10 = 100) → R = 5 :=
by
  intros
  sorry

end number_of_rows_of_red_notes_l47_47628


namespace math_problem_l47_47212

-- Conditions
def ellipse_eq (a b : ℝ) : Prop := ∀ x y : ℝ, x^2 / (a^2) + y^2 / (b^2) = 1
def eccentricity (a c : ℝ) : Prop := c / a = (Real.sqrt 2) / 2
def major_axis_length (a : ℝ) : Prop := 2 * a = 6 * Real.sqrt 2

-- Equations and properties to be proven
def ellipse_equation : Prop := ∃ a b : ℝ, a = 3 * Real.sqrt 2 ∧ b = 3 ∧ ellipse_eq a b
def length_AB (θ : ℝ) : Prop := ∃ AB : ℝ, AB = (6 * Real.sqrt 2) / (1 + (Real.sin θ)^2)
def min_AB_CD : Prop := ∃ θ : ℝ, (Real.sin (2 * θ) = 1) ∧ (6 * Real.sqrt 2) / (1 + (Real.sin θ)^2) + (6 * Real.sqrt 2) / (1 + (Real.cos θ)^2) = 8 * Real.sqrt 2

-- The complete proof problem
theorem math_problem : ellipse_equation ∧
                       (∀ θ : ℝ, length_AB θ) ∧
                       min_AB_CD := by
  sorry

end math_problem_l47_47212


namespace total_stickers_l47_47386

theorem total_stickers (r s t : ℕ) (h1 : r = 30) (h2 : s = 3 * r) (h3 : t = s + 20) : r + s + t = 230 :=
by sorry

end total_stickers_l47_47386


namespace melanie_trout_catch_l47_47372

theorem melanie_trout_catch (T M : ℕ) 
  (h1 : T = 2 * M) 
  (h2 : T = 16) : 
  M = 8 :=
by
  sorry

end melanie_trout_catch_l47_47372


namespace three_lines_pass_through_point_and_intersect_parabola_l47_47722

-- Define the point (0,1)
def point : ℝ × ℝ := (0, 1)

-- Define the parabola y^2 = 4x as a set of points
def parabola (p : ℝ × ℝ) : Prop :=
  (p.snd)^2 = 4 * (p.fst)

-- Define the condition for the line passing through (0,1)
def line_through_point (line_eq : ℝ → ℝ) : Prop :=
  line_eq 0 = 1

-- Define the condition for the line intersecting the parabola at only one point
def intersects_once (line_eq : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, parabola (x, line_eq x)

-- The main theorem statement
theorem three_lines_pass_through_point_and_intersect_parabola :
  ∃ (f1 f2 f3 : ℝ → ℝ), 
    line_through_point f1 ∧ line_through_point f2 ∧ line_through_point f3 ∧
    intersects_once f1 ∧ intersects_once f2 ∧ intersects_once f3 ∧
    (∀ (f : ℝ → ℝ), (line_through_point f ∧ intersects_once f) ->
      (f = f1 ∨ f = f2 ∨ f = f3)) :=
sorry

end three_lines_pass_through_point_and_intersect_parabola_l47_47722


namespace candle_burning_time_l47_47824

theorem candle_burning_time :
  ∃ t : ℚ, (1 - t / 5) = 3 * (1 - t / 4) ∧ t = 40 / 11 :=
by {
  sorry
}

end candle_burning_time_l47_47824


namespace pencils_sold_is_correct_l47_47935

-- Define the conditions
def first_two_students_pencils : Nat := 2 * 2
def next_six_students_pencils : Nat := 6 * 3
def last_two_students_pencils : Nat := 2 * 1
def total_pencils_sold : Nat := first_two_students_pencils + next_six_students_pencils + last_two_students_pencils

-- Prove that all pencils sold equals 24
theorem pencils_sold_is_correct : total_pencils_sold = 24 :=
by 
  -- Add the statement to be proved here
  sorry

end pencils_sold_is_correct_l47_47935


namespace calculate_expression_l47_47115

theorem calculate_expression : (-1^4 + |1 - Real.sqrt 2| - (Real.pi - 3.14)^0) = Real.sqrt 2 - 3 :=
by
  sorry

end calculate_expression_l47_47115


namespace correct_ordering_of_fractions_l47_47413

theorem correct_ordering_of_fractions :
  let a := (6 : ℚ) / 17
  let b := (8 : ℚ) / 25
  let c := (10 : ℚ) / 31
  let d := (1 : ℚ) / 3
  b < d ∧ d < c ∧ c < a :=
by
  sorry

end correct_ordering_of_fractions_l47_47413


namespace base_six_equals_base_b_l47_47813

noncomputable def base_six_to_decimal (n : ℕ) : ℕ :=
  6 * 6 + 2

noncomputable def base_b_to_decimal (b : ℕ) : ℕ :=
  b^2 + 2 * b + 4

theorem base_six_equals_base_b (b : ℕ) : b^2 + 2 * b - 34 = 0 → b = 4 := 
by sorry

end base_six_equals_base_b_l47_47813


namespace candy_left_proof_l47_47836

def candy_left (d_candy : ℕ) (s_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  d_candy + s_candy - eaten_candy

theorem candy_left_proof :
  candy_left 32 42 35 = 39 :=
by
  sorry

end candy_left_proof_l47_47836


namespace compare_sqrt_l47_47714

noncomputable def a : ℝ := 3 * Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 15

theorem compare_sqrt : a > b :=
by
  sorry

end compare_sqrt_l47_47714


namespace james_bought_100_cattle_l47_47348

noncomputable def number_of_cattle (purchase_price : ℝ) (feeding_ratio : ℝ) (weight_per_cattle : ℝ) (price_per_pound : ℝ) (profit : ℝ) : ℝ :=
  let feeding_cost := purchase_price * feeding_ratio
  let total_feeding_cost := purchase_price + feeding_cost
  let total_cost := purchase_price + total_feeding_cost
  let selling_price_per_cattle := weight_per_cattle * price_per_pound
  let total_revenue := total_cost + profit
  total_revenue / selling_price_per_cattle

theorem james_bought_100_cattle :
  number_of_cattle 40000 0.20 1000 2 112000 = 100 :=
by {
  sorry
}

end james_bought_100_cattle_l47_47348


namespace money_equations_l47_47779

theorem money_equations (x y : ℝ) (h1 : x + (1 / 2) * y = 50) (h2 : y + (2 / 3) * x = 50) :
  x + (1 / 2) * y = 50 ∧ y + (2 / 3) * x = 50 :=
by
  exact ⟨h1, h2⟩

-- Please note that by stating the theorem this way, we have restated the conditions and conclusion
-- in Lean 4. The proof uses the given conditions directly without the need for intermediate steps.

end money_equations_l47_47779


namespace pineapples_sold_l47_47240

/-- 
There were initially 86 pineapples in the store. After selling some pineapples,
9 of the remaining pineapples were rotten and were discarded. Given that there 
are 29 fresh pineapples left, prove that the number of pineapples sold is 48.
-/
theorem pineapples_sold (initial_pineapples : ℕ) (rotten_pineapples : ℕ) (remaining_fresh_pineapples : ℕ)
  (h_init : initial_pineapples = 86)
  (h_rotten : rotten_pineapples = 9)
  (h_fresh : remaining_fresh_pineapples = 29) :
  initial_pineapples - (remaining_fresh_pineapples + rotten_pineapples) = 48 :=
sorry

end pineapples_sold_l47_47240


namespace companies_increase_profitability_l47_47993

-- Define the three main arguments as conditions
def increased_retention_and_attraction : Prop := 
  ∀ (company : Type), ∃ (environment : Type),
    (company = Google ∨ company = Facebook) → 
    environment provides_comfortable_and_flexible_workplace → 
    increases_profitability company

def enhanced_productivity_and_creativity : Prop := 
  ∀ (company : Type), ∃ (environment : Type),
    (company = Google ∨ company = Facebook) → 
    environment provides_work_and_leisure_integration → 
    increases_profitability company

def better_work_life_integration : Prop := 
  ∀ (company : Type), ∃ (environment : Type),
    (company = Google ∨ company = Facebook) → 
    environment provides_work_life_integration → 
    increases_profitability company

-- Define the overall theorem we need to prove
theorem companies_increase_profitability 
  (company : Type)
  (h1 : increased_retention_and_attraction)
  (h2 : enhanced_productivity_and_creativity)
  (h3 : better_work_life_integration) 
: ∃ (environment : Type), 
    (company = Google ∨ company = Facebook) → 
    environment enhances_work_environment → 
    increases_profitability company :=
by
  sorry

end companies_increase_profitability_l47_47993


namespace binom_19_9_l47_47315

variable (n k : ℕ)

theorem binom_19_9
  (h₁ : nat.choose 17 7 = 19448)
  (h₂ : nat.choose 17 8 = 24310)
  (h₃ : nat.choose 17 9 = 24310) :
  nat.choose 19 9 = 92378 := by
  sorry

end binom_19_9_l47_47315


namespace cosine_lt_sine_neg_four_l47_47568

theorem cosine_lt_sine_neg_four : ∀ (m n : ℝ), m = Real.cos (-4) → n = Real.sin (-4) → m < n :=
by
  intros m n hm hn
  rw [hm, hn]
  sorry

end cosine_lt_sine_neg_four_l47_47568


namespace percent_x_of_w_l47_47195

theorem percent_x_of_w (x y z w : ℝ)
  (h1 : x = 1.2 * y)
  (h2 : y = 0.7 * z)
  (h3 : w = 1.5 * z) : (x / w) * 100 = 56 :=
by
  sorry

end percent_x_of_w_l47_47195


namespace hyperbola_tangent_circle_asymptote_l47_47618

theorem hyperbola_tangent_circle_asymptote (m : ℝ) (hm : m > 0) :
  (∀ x y : ℝ, y^2 - (x^2 / m^2) = 1 → 
   ∃ (circle_center : ℝ × ℝ) (circle_radius : ℝ), 
   circle_center = (0, 2) ∧ circle_radius = 1 ∧ 
   ((∃ k : ℝ, k = m ∨ k = -m) → 
    ((k * 0 - 2 + 0) / real.sqrt ((k)^2 + (-1)^2) = 1))) →
  m = real.sqrt 3 / 3 :=
by sorry

end hyperbola_tangent_circle_asymptote_l47_47618


namespace num_sets_of_consecutive_integers_sum_to_30_l47_47611

theorem num_sets_of_consecutive_integers_sum_to_30 : 
  let S_n (n a : ℕ) := (n * (2 * a + n - 1)) / 2 
  ∃! (s : ℕ), s = 3 ∧ ∀ n, n ≥ 2 → ∃ a, S_n n a = 30 :=
by
  sorry

end num_sets_of_consecutive_integers_sum_to_30_l47_47611


namespace archer_probability_l47_47280

noncomputable def prob_hit : ℝ := 0.9
noncomputable def prob_miss : ℝ := 1 - prob_hit

theorem archer_probability :
  (prob_miss * prob_hit * prob_hit * prob_hit) = 0.0729 :=
by
  have h_prob_miss : prob_miss = 0.1 := by norm_num
  have h_prob_seq : prob_miss * prob_hit * prob_hit * prob_hit 
                    = 0.1 * 0.9 * 0.9 * 0.9 := by rw [h_prob_miss]
  rw [h_prob_seq]
  norm_num
  sorry

end archer_probability_l47_47280


namespace joyce_gave_apples_l47_47206

theorem joyce_gave_apples : 
  ∀ (initial_apples final_apples given_apples : ℕ), (initial_apples = 75) ∧ (final_apples = 23) → (given_apples = initial_apples - final_apples) → (given_apples = 52) :=
by
  intros
  sorry

end joyce_gave_apples_l47_47206


namespace rectangle_dimensions_l47_47102

theorem rectangle_dimensions (w l : ℚ) (h1 : 2 * l + 2 * w = 2 * l * w) (h2 : l = 3 * w) :
  w = 4 / 3 ∧ l = 4 :=
by
  sorry

end rectangle_dimensions_l47_47102


namespace find_m_l47_47599

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l47_47599


namespace sum_digits_probability_l47_47436

noncomputable def sumOfDigits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def numInRange : ℕ := 1000000

noncomputable def coefficient : ℕ :=
  Nat.choose 24 5 - 6 * Nat.choose 14 5

noncomputable def probability : ℚ :=
  coefficient / numInRange

theorem sum_digits_probability :
  probability = 7623 / 250000 :=
by
  sorry

end sum_digits_probability_l47_47436


namespace minimum_value_of_a_l47_47736

theorem minimum_value_of_a (x y a : ℝ) (h1 : y = (1 / (x - 2)) * (x^2))
(h2 : x = a * y) : a = 3 :=
sorry

end minimum_value_of_a_l47_47736


namespace roots_ratio_sum_l47_47766

theorem roots_ratio_sum (α β : ℝ) (hαβ : α > β) (h1 : 3*α^2 + α - 1 = 0) (h2 : 3*β^2 + β - 1 = 0) :
  α / β + β / α = -7 / 3 :=
sorry

end roots_ratio_sum_l47_47766


namespace no_prime_in_sum_100_consecutive_l47_47557

noncomputable def sum_100_consecutive (n : ℕ) : ℕ :=
  100 * n + 4950

theorem no_prime_in_sum_100_consecutive :
  ∀ n : ℕ, ¬ Prime (sum_100_consecutive n) :=
by
  intro n
  have h : sum_100_consecutive n = 50 * (2 * n + 99) := by
    rw [sum_100_consecutive, mul_add, mul_comm 100 n, mul_assoc, add_comm]
  rw h
  exact Prime.not_prime_mul (prime_two, 50) sorry

end no_prime_in_sum_100_consecutive_l47_47557


namespace valentino_farm_birds_total_l47_47378

theorem valentino_farm_birds_total :
  let chickens := 200
  let ducks := 2 * chickens
  let turkeys := 3 * ducks
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof here
  sorry

end valentino_farm_birds_total_l47_47378


namespace probability_of_unique_color_and_number_l47_47053

-- Defining the sets of colors and numbers
inductive Color
| red
| yellow
| blue

inductive Number
| one
| two
| three

-- Defining a ball as a combination of a Color and a Number
structure Ball :=
(color : Color)
(number : Number)

-- Setting up the list of 9 balls
def allBalls : List Ball :=
  [⟨Color.red, Number.one⟩, ⟨Color.red, Number.two⟩, ⟨Color.red, Number.three⟩,
   ⟨Color.yellow, Number.one⟩, ⟨Color.yellow, Number.two⟩, ⟨Color.yellow, Number.three⟩,
   ⟨Color.blue, Number.one⟩, ⟨Color.blue, Number.two⟩, ⟨Color.blue, Number.three⟩]

-- Proving the probability calculation as a theorem
noncomputable def probability_neither_same_color_nor_number : ℕ → ℕ → ℚ :=
  λ favorable total => favorable / total

theorem probability_of_unique_color_and_number :
  probability_neither_same_color_nor_number
    (6) -- favorable outcomes
    (84) -- total outcomes
  = 1 / 14 := by
  sorry

end probability_of_unique_color_and_number_l47_47053


namespace q_is_false_l47_47904

-- Given conditions
variables (p q : Prop)
axiom h1 : ¬ (p ∧ q)
axiom h2 : ¬ ¬ p

-- Proof that q is false
theorem q_is_false : q = False :=
by
  sorry

end q_is_false_l47_47904


namespace union_sets_l47_47564

def setA : Set ℝ := { x | -1 ≤ x ∧ x < 3 }

def setB : Set ℝ := { x | x^2 - 7 * x + 10 ≤ 0 }

theorem union_sets : setA ∪ setB = { x | -1 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end union_sets_l47_47564


namespace sin_315_eq_neg_sqrt2_div_2_l47_47537

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47537


namespace no_perfect_squares_l47_47134

theorem no_perfect_squares (x y z t : ℕ) (h1 : xy - zt = k) (h2 : x + y = k) (h3 : z + t = k) :
  ¬ (∃ m n : ℕ, x * y = m^2 ∧ z * t = n^2) := by
  sorry

end no_perfect_squares_l47_47134


namespace largest_possible_3_digit_sum_l47_47769

theorem largest_possible_3_digit_sum (X Y Z : ℕ) (h_diff : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) 
(h_digit_X : 0 ≤ X ∧ X ≤ 9) (h_digit_Y : 0 ≤ Y ∧ Y ≤ 9) (h_digit_Z : 0 ≤ Z ∧ Z ≤ 9) :
  (100 * X + 10 * X + X) + (10 * Y + X) + X = 994 → (X, Y, Z) = (8, 9, 0) := by
  sorry

end largest_possible_3_digit_sum_l47_47769


namespace sum_of_numbers_l47_47337

theorem sum_of_numbers (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 24) : 
  x + y + z = 10 :=
by
  sorry

end sum_of_numbers_l47_47337


namespace solution_set_of_inequality_system_l47_47937

theorem solution_set_of_inequality_system (x : ℝ) :
  (4 * x + 1 > 2) ∧ (1 - 2 * x < 7) ↔ (x > 1 / 4) :=
by
  sorry

end solution_set_of_inequality_system_l47_47937


namespace diminished_gcd_equals_100_l47_47123

theorem diminished_gcd_equals_100 : Nat.gcd 7800 360 - 20 = 100 := by
  sorry

end diminished_gcd_equals_100_l47_47123


namespace problem_seven_integers_l47_47648

theorem problem_seven_integers (a b c d e f g : ℕ) 
  (h1 : b = a + 1) 
  (h2 : c = b + 1) 
  (h3 : d = c + 1) 
  (h4 : e = d + 1) 
  (h5 : f = e + 1) 
  (h6 : g = f + 1) 
  (h_sum : a + b + c + d + e + f + g = 2017) : 
  a = 286 ∨ g = 286 :=
sorry

end problem_seven_integers_l47_47648


namespace find_d_value_l47_47150

theorem find_d_value (d : ℝ) :
  (∀ x, (8 * x^3 + 27 * x^2 + d * x + 55 = 0) → (2 * x + 5 = 0)) → d = 39.5 :=
by
  sorry

end find_d_value_l47_47150


namespace equation_of_line_BC_l47_47891

/-
Given:
1. Point A(3, -1)
2. The line containing the median from A to side BC: 6x + 10y - 59 = 0
3. The line containing the angle bisector of ∠B: x - 4y + 10 = 0

Prove:
The equation of the line containing side BC is 2x + 9y - 65 = 0.
-/

noncomputable def point_A : (ℝ × ℝ) := (3, -1)

noncomputable def median_line (x y : ℝ) : Prop := 6 * x + 10 * y - 59 = 0

noncomputable def angle_bisector_line_B (x y : ℝ) : Prop := x - 4 * y + 10 = 0

theorem equation_of_line_BC :
  ∃ (x y : ℝ), 2 * x + 9 * y - 65 = 0 :=
sorry

end equation_of_line_BC_l47_47891


namespace topsoil_cost_proof_l47_47823

-- Definitions
def cost_per_cubic_foot : ℕ := 8
def cubic_feet_per_cubic_yard : ℕ := 27
def amount_in_cubic_yards : ℕ := 7

-- Theorem
theorem topsoil_cost_proof : cost_per_cubic_foot * cubic_feet_per_cubic_yard * amount_in_cubic_yards = 1512 := by
  -- proof logic goes here
  sorry

end topsoil_cost_proof_l47_47823


namespace min_value_l47_47790

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
3 * a + 6 * b + 12 * c

theorem min_value (a b c : ℝ) (h : 9 * a ^ 2 + 4 * b ^ 2 + 36 * c ^ 2 = 4) :
  minimum_value a b c = -2 * Real.sqrt 14 := sorry

end min_value_l47_47790


namespace fraction_addition_l47_47452

theorem fraction_addition : (2 / 5 + 3 / 8) = 31 / 40 :=
by
  sorry

end fraction_addition_l47_47452


namespace speed_first_half_proof_l47_47844

noncomputable def speed_first_half
  (total_time: ℕ) 
  (distance: ℕ) 
  (second_half_speed: ℕ) 
  (first_half_time: ℕ) :
  ℕ :=
  distance / first_half_time

theorem speed_first_half_proof
  (total_time: ℕ)
  (distance: ℕ)
  (second_half_speed: ℕ)
  (half_distance: ℕ)
  (second_half_time: ℕ)
  (first_half_time: ℕ) :
  total_time = 12 →
  distance = 560 →
  second_half_speed = 40 →
  half_distance = distance / 2 →
  second_half_time = half_distance / second_half_speed →
  first_half_time = total_time - second_half_time →
  speed_first_half total_time half_distance second_half_speed first_half_time = 56 :=
by
  sorry

end speed_first_half_proof_l47_47844


namespace stratified_sampling_sample_size_l47_47425

-- Definitions based on conditions
def total_employees : ℕ := 120
def male_employees : ℕ := 90
def female_employees_in_sample : ℕ := 3

-- Proof statement
theorem stratified_sampling_sample_size : total_employees = 120 ∧ male_employees = 90 ∧ female_employees_in_sample = 3 → 
  (female_employees_in_sample + female_employees_in_sample * (male_employees / (total_employees - male_employees))) = 12 :=
sorry

end stratified_sampling_sample_size_l47_47425


namespace toys_produced_each_day_l47_47695

theorem toys_produced_each_day (weekly_production : ℕ) (days_worked : ℕ) 
  (h1 : weekly_production = 5500) (h2 : days_worked = 4) : 
  (weekly_production / days_worked = 1375) :=
sorry

end toys_produced_each_day_l47_47695


namespace terminal_side_quadrant_l47_47902

theorem terminal_side_quadrant (α : ℝ) (k : ℤ) (hk : α = 45 + k * 180) :
  (∃ n : ℕ, k = 2 * n ∧ α = 45) ∨ (∃ n : ℕ, k = 2 * n + 1 ∧ α = 225) :=
sorry

end terminal_side_quadrant_l47_47902


namespace tan_shifted_value_l47_47181

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l47_47181


namespace largest_n_exists_l47_47297

theorem largest_n_exists :
  ∃ (n : ℕ), (∀ (x : ℕ → ℝ), (∀ i j : ℕ, 1 ≤ i → i < j → j ≤ n → (1 + x i * x j)^2 ≤ 0.99 * (1 + x i^2) * (1 + x j^2))) ∧ n = 31 :=
sorry

end largest_n_exists_l47_47297


namespace CoinRun_ProcGen_ratio_l47_47147

theorem CoinRun_ProcGen_ratio
  (greg_ppo_reward: ℝ)
  (maximum_procgen_reward: ℝ)
  (ppo_ratio: ℝ)
  (maximum_coinrun_reward: ℝ)
  (coinrun_to_procgen_ratio: ℝ)
  (greg_ppo_reward_eq: greg_ppo_reward = 108)
  (maximum_procgen_reward_eq: maximum_procgen_reward = 240)
  (ppo_ratio_eq: ppo_ratio = 0.90)
  (coinrun_equation: maximum_coinrun_reward = greg_ppo_reward / ppo_ratio)
  (ratio_definition: coinrun_to_procgen_ratio = maximum_coinrun_reward / maximum_procgen_reward) :
  coinrun_to_procgen_ratio = 0.5 :=
sorry

end CoinRun_ProcGen_ratio_l47_47147


namespace _l47_47713

noncomputable def charlesPictures : Prop :=
  ∀ (bought : ℕ) (drew_today : ℕ) (drew_yesterday_after_work : ℕ) (left : ℕ),
    (bought = 20) →
    (drew_today = 6) →
    (drew_yesterday_after_work = 6) →
    (left = 2) →
    (bought - left - drew_today - drew_yesterday_after_work = 6)

-- We can use this statement "charlesPictures" to represent the theorem to be proved in Lean 4.

end _l47_47713


namespace ratio_of_circle_areas_l47_47346

variable (S L A : ℝ)

theorem ratio_of_circle_areas 
  (h1 : A = (3 / 5) * S)
  (h2 : A = (6 / 25) * L)
  : S / L = 2 / 5 :=
by
  sorry

end ratio_of_circle_areas_l47_47346


namespace part1_3kg_part2_5kg_part2_function_part3_compare_l47_47018

noncomputable def supermarket_A_cost (x : ℝ) : ℝ :=
if x <= 4 then 10 * x
else 6 * x + 16

noncomputable def supermarket_B_cost (x : ℝ) : ℝ :=
8 * x

-- Proof that supermarket_A_cost 3 = 30
theorem part1_3kg : supermarket_A_cost 3 = 30 :=
by sorry

-- Proof that supermarket_A_cost 5 = 46
theorem part2_5kg : supermarket_A_cost 5 = 46 :=
by sorry

-- Proof that the cost function is correct
theorem part2_function (x : ℝ) : 
(0 < x ∧ x <= 4 → supermarket_A_cost x = 10 * x) ∧ 
(x > 4 → supermarket_A_cost x = 6 * x + 16) :=
by sorry

-- Proof that supermarket A is cheaper for 10 kg apples
theorem part3_compare : supermarket_A_cost 10 < supermarket_B_cost 10 :=
by sorry

end part1_3kg_part2_5kg_part2_function_part3_compare_l47_47018


namespace hydras_survive_l47_47068

theorem hydras_survive (A_heads : ℕ) (B_heads : ℕ) (growthA growthB : ℕ → ℕ) (a b : ℕ)
    (hA : A_heads = 2016) (hB : B_heads = 2017)
    (growthA_conds : ∀ n, growthA n ∈ {5, 7})
    (growthB_conds : ∀ n, growthB n ∈ {5, 7}) :
  ∀ n, let total_heads := (A_heads + growthA n - 2 * n) + (B_heads + growthB n - 2 * n);
       total_heads % 2 = 1 :=
by intro n
   sorry

end hydras_survive_l47_47068


namespace sufficient_but_not_necessary_condition_l47_47187

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, |2*x - 1| ≤ x → x^2 + x - 2 ≤ 0) ∧ 
  ¬(∀ x : ℝ, x^2 + x - 2 ≤ 0 → |2 * x - 1| ≤ x) := sorry

end sufficient_but_not_necessary_condition_l47_47187


namespace sqrt_pow_mul_l47_47715

theorem sqrt_pow_mul (a b : ℝ) : (a = 3) → (b = 5) → (Real.sqrt (a^2 * b^6) = 375) :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end sqrt_pow_mul_l47_47715


namespace find_k_solution_l47_47322

theorem find_k_solution 
  (k : ℝ)
  (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |k * x - 4| ≤ 2) : 
  k = 2 :=
sorry

end find_k_solution_l47_47322


namespace find_m_l47_47586

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l47_47586


namespace sum_divisors_36_48_l47_47727

open Finset

noncomputable def sum_common_divisors (a b : ℕ) : ℕ :=
  let divisors_a := (range (a + 1)).filter (λ x => a % x = 0)
  let divisors_b := (range (b + 1)).filter (λ x => b % x = 0)
  let common_divisors := divisors_a ∩ divisors_b
  common_divisors.sum id

theorem sum_divisors_36_48 : sum_common_divisors 36 48 = 28 := by
  sorry

end sum_divisors_36_48_l47_47727


namespace simplify_expression_l47_47034

theorem simplify_expression (a : ℤ) (h_range : -3 < a ∧ a ≤ 0) (h_notzero : a ≠ 0) (h_notone : a ≠ 1 ∧ a ≠ -1) :
  (a - (2 * a - 1) / a) / (1 / a - a) = -3 :=
by
  have h_eq : (a - (2 * a - 1) / a) / (1 / a - a) = (1 - a) / (1 + a) :=
    sorry
  have h_a_neg_two : a = -2 :=
    sorry
  rw [h_eq, h_a_neg_two]
  sorry


end simplify_expression_l47_47034


namespace not_equiv_2_pi_six_and_11_pi_six_l47_47419

def polar_equiv (r θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * ↑k * Real.pi

theorem not_equiv_2_pi_six_and_11_pi_six :
  ¬ polar_equiv 2 (Real.pi / 6) (11 * Real.pi / 6) := 
sorry

end not_equiv_2_pi_six_and_11_pi_six_l47_47419


namespace find_number_satisfy_equation_l47_47772

theorem find_number_satisfy_equation (x : ℝ) :
  9 - x / 7 * 5 + 10 = 13.285714285714286 ↔ x = -20 := sorry

end find_number_satisfy_equation_l47_47772


namespace sin_315_eq_neg_sqrt2_div_2_l47_47535

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47535


namespace remaining_players_average_points_l47_47807

-- Define the conditions
def total_points : ℕ := 270
def total_players : ℕ := 9
def players_averaged_50 : ℕ := 5
def average_points_50 : ℕ := 50

-- Define the query
theorem remaining_players_average_points :
  (total_points - players_averaged_50 * average_points_50) / (total_players - players_averaged_50) = 5 :=
by
  sorry

end remaining_players_average_points_l47_47807


namespace num_people_on_boats_l47_47258

-- Definitions based on the conditions
def boats := 5
def people_per_boat := 3

-- Theorem stating the problem to be solved
theorem num_people_on_boats : boats * people_per_boat = 15 :=
by sorry

end num_people_on_boats_l47_47258


namespace find_x_l47_47308

theorem find_x (n : ℕ) (hn : Odd n) (hx : ∃ p1 p2 p3 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 11 ∈ {p1, p2, p3} ∧ (6^n + 1) = p1 * p2 * p3) :
  6^n + 1 = 7777 :=
by
  sorry

end find_x_l47_47308


namespace cosine_theorem_a_cosine_theorem_b_cosine_theorem_c_l47_47120

theorem cosine_theorem_a (a b c A : ℝ) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A := sorry

theorem cosine_theorem_b (a b c B : ℝ) :
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B := sorry

theorem cosine_theorem_c (a b c C : ℝ) :
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C := sorry

end cosine_theorem_a_cosine_theorem_b_cosine_theorem_c_l47_47120


namespace sin_315_eq_neg_sqrt2_div_2_l47_47505

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47505


namespace probability_all_pairs_successful_expected_successful_pairs_greater_than_half_l47_47410

-- Define the problem's parameters and conditions
def n : ℕ  -- number of pairs of socks

-- Define the successful pair probability calculation
theorem probability_all_pairs_successful (n : ℕ) :
  (∑ s in Finset.range (n + 1), s.factorial * 2 ^ s) / ((2 * n).factorial) = 2^n * n.factorial := sorry

-- Define the expected value calculation
theorem expected_successful_pairs_greater_than_half (n : ℕ) (h : n > 0) :
  (∑ i in Finset.range n, (1 : ℝ) / (2 * n - 1) : ℝ) / n > 0.5 := sorry

end probability_all_pairs_successful_expected_successful_pairs_greater_than_half_l47_47410


namespace sin_315_eq_neg_sqrt2_div_2_l47_47466

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47466


namespace value_of_expression_at_x_4_l47_47246

theorem value_of_expression_at_x_4 :
  ∀ (x : ℝ), x = 4 → (x^2 - 2 * x - 8) / (x - 4) = 6 :=
by
  intro x hx
  sorry

end value_of_expression_at_x_4_l47_47246


namespace union_complement_eq_set_l47_47215

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {3, 5, 6}
def comp_U_N : Set ℕ := U \ N  -- complement of N with respect to U

theorem union_complement_eq_set :
  M ∪ comp_U_N = {1, 2, 3, 4} :=
by
  simp [U, M, N, comp_U_N]
  sorry

end union_complement_eq_set_l47_47215


namespace range_of_a_l47_47040

noncomputable def f (x a : ℝ) := Real.log x + 1 / 2 * x^2 + a * x

theorem range_of_a
  (a : ℝ)
  (h : ∃ x : ℝ, x > 0 ∧ (1/x + x + a = 3)) :
  a ≤ 1 :=
by
  sorry

end range_of_a_l47_47040


namespace find_m_l47_47584

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l47_47584


namespace exists_subset_A_l47_47138

theorem exists_subset_A (n k : ℕ) (h1 : 2 ≤ n) (h2 : n < 2^k) :
  ∃ A : Finset ℕ, (∀ x y ∈ A, x ≠ y → Nat.choose y x % 2 = 0) ∧
  (A.card ≥ (Nat.choose k (k / 2) / 2^k) * (n + 1)) :=
by
  sorry

end exists_subset_A_l47_47138


namespace least_possible_value_z_minus_x_l47_47678

theorem least_possible_value_z_minus_x
  (x y z : ℤ)
  (h1 : x < y)
  (h2 : y < z)
  (h3 : y - x > 5)
  (hx_even : x % 2 = 0)
  (hy_odd : y % 2 = 1)
  (hz_odd : z % 2 = 1) :
  z - x = 9 :=
  sorry

end least_possible_value_z_minus_x_l47_47678


namespace angle_between_vectors_l47_47326

noncomputable def vec_a : ℝ × ℝ := (-2 * Real.sqrt 3, 2)
noncomputable def vec_b : ℝ × ℝ := (1, - Real.sqrt 3)

-- Define magnitudes
noncomputable def mag_a : ℝ := Real.sqrt ((-2 * Real.sqrt 3) ^ 2 + 2^2)
noncomputable def mag_b : ℝ := Real.sqrt (1^2 + (- Real.sqrt 3) ^ 2)

-- Define the dot product
noncomputable def dot_product : ℝ := (-2 * Real.sqrt 3) * 1 + 2 * (- Real.sqrt 3)

-- Define cosine of the angle theta
-- We use mag_a and mag_b defined above
noncomputable def cos_theta : ℝ := dot_product / (mag_a * mag_b)

-- Define the angle theta, within the range [0, π]
noncomputable def theta : ℝ := Real.arccos cos_theta

-- The expected result is θ = 5π / 6
theorem angle_between_vectors : theta = (5 * Real.pi) / 6 :=
by
  sorry

end angle_between_vectors_l47_47326


namespace car_Z_probability_l47_47341

theorem car_Z_probability :
  let P_X := 1/6
  let P_Y := 1/10
  let P_XYZ := 0.39166666666666666
  ∃ P_Z : ℝ, P_X + P_Y + P_Z = P_XYZ ∧ P_Z = 0.125 :=
by
  sorry

end car_Z_probability_l47_47341


namespace integral_of_reciprocal_l47_47991

theorem integral_of_reciprocal (a b : ℝ) (h_eq : a = 1) (h_eb : b = Real.exp 1) : ∫ x in a..b, 1/x = 1 :=
by 
  rw [h_eq, h_eb]
  sorry

end integral_of_reciprocal_l47_47991


namespace cost_of_eraser_l47_47272

theorem cost_of_eraser 
  (s n c : ℕ)
  (h1 : s > 18)
  (h2 : n > 2)
  (h3 : c > n)
  (h4 : s * c * n = 3978) : 
  c = 17 :=
sorry

end cost_of_eraser_l47_47272


namespace inverse_propositions_l47_47683

-- Given conditions
lemma right_angles_equal : ∀ θ1 θ2 : ℝ, θ1 = θ2 → (θ1 = 90 ∧ θ2 = 90) :=
sorry

lemma equal_angles_right : ∀ θ1 θ2 : ℝ, (θ1 = 90 ∧ θ2 = 90) → (θ1 = θ2) :=
sorry

-- Theorem to be proven
theorem inverse_propositions :
  (∀ θ1 θ2 : ℝ, θ1 = θ2 → (θ1 = 90 ∧ θ2 = 90)) ↔
  (∀ θ1 θ2 : ℝ, (θ1 = 90 ∧ θ2 = 90) → (θ1 = θ2)) :=
sorry

end inverse_propositions_l47_47683


namespace terms_are_equal_l47_47641

theorem terms_are_equal (n : ℕ) (a b : ℕ → ℕ)
  (h_n : n ≥ 2018)
  (h_a : ∀ i j : ℕ, i ≠ j → a i ≠ a j)
  (h_b : ∀ i j : ℕ, i ≠ j → b i ≠ b j)
  (h_a_pos : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i > 0)
  (h_b_pos : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → b i > 0)
  (h_a_le : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ≤ 5 * n)
  (h_b_le : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → b i ≤ 5 * n)
  (h_arith : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → (a j * b i - a i * b j) * (j - i) = 0):
  ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → a i * b j = a j * b i :=
by
  sorry

end terms_are_equal_l47_47641


namespace complement_intersection_l47_47890

open Set

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection :
  (U \ M) ∩ N = {3} :=
sorry

end complement_intersection_l47_47890


namespace smallest_prime_dividing_sum_l47_47073

theorem smallest_prime_dividing_sum (a b : ℕ) (h₁ : a = 7^15) (h₂ : b = 9^17) (h₃ : a % 2 = 1) (h₄ : b % 2 = 1) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (a + b) ∧ ∀ q : ℕ, (Nat.Prime q ∧ q ∣ (a + b)) → q ≥ p := by
  sorry

end smallest_prime_dividing_sum_l47_47073


namespace mr_valentino_birds_l47_47381

theorem mr_valentino_birds : 
  ∀ (chickens ducks turkeys : ℕ), 
  chickens = 200 → 
  ducks = 2 * chickens → 
  turkeys = 3 * ducks → 
  chickens + ducks + turkeys = 1800 :=
by
  intros chickens ducks turkeys
  assume h1 : chickens = 200
  assume h2 : ducks = 2 * chickens
  assume h3 : turkeys = 3 * ducks
  sorry

end mr_valentino_birds_l47_47381


namespace eccentricity_of_hyperbola_l47_47143

theorem eccentricity_of_hyperbola (a b c : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_c : c = Real.sqrt (a^2 + b^2))
  (F1 : ℝ × ℝ := (-c, 0))
  (A B : ℝ × ℝ)
  (slope_of_AB : ∀ (x y : ℝ), y = x + c)
  (asymptotes_eqn : ∀ (x : ℝ), x = a ∨ x = -a)
  (intersections : A = (-(a * c / (a - b)), -(b * c / (a - b))) ∧ B = (-(a * c / (a + b)), (b * c / (a + b))))
  (AB_eq_2BF1 : 2 * (F1 - B) = A - B) :
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 5 :=
sorry

end eccentricity_of_hyperbola_l47_47143


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47523

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47523


namespace minimum_spend_on_boxes_l47_47421

noncomputable def box_length : ℕ := 20
noncomputable def box_width : ℕ := 20
noncomputable def box_height : ℕ := 12
noncomputable def cost_per_box : ℝ := 0.40
noncomputable def total_volume : ℕ := 2400000

theorem minimum_spend_on_boxes : 
  (total_volume / (box_length * box_width * box_height)) * cost_per_box = 200 :=
by
  sorry

end minimum_spend_on_boxes_l47_47421


namespace points_lost_calculation_l47_47859

variable (firstRound secondRound finalScore : ℕ)
variable (pointsLost : ℕ)

theorem points_lost_calculation 
  (h1 : firstRound = 40) 
  (h2 : secondRound = 50) 
  (h3 : finalScore = 86) 
  (h4 : pointsLost = firstRound + secondRound - finalScore) :
  pointsLost = 4 := 
sorry

end points_lost_calculation_l47_47859


namespace tan_angle_addition_l47_47166

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l47_47166


namespace find_m_l47_47594

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l47_47594


namespace coeff_of_quadratic_term_eq_neg5_l47_47926

theorem coeff_of_quadratic_term_eq_neg5 (a b c : ℝ) (h_eq : -5 * x^2 + 5 * x + 6 = a * x^2 + b * x + c) :
  a = -5 :=
by
  sorry

end coeff_of_quadratic_term_eq_neg5_l47_47926


namespace arrangement_plans_count_l47_47556

noncomputable def number_of_arrangement_plans (num_teachers : ℕ) (num_students : ℕ) : ℕ :=
if num_teachers = 2 ∧ num_students = 4 then 12 else 0

theorem arrangement_plans_count :
  number_of_arrangement_plans 2 4 = 12 :=
by 
  sorry

end arrangement_plans_count_l47_47556


namespace kyle_money_l47_47355

theorem kyle_money (dave_money : ℕ) (kyle_initial : ℕ) (kyle_remaining : ℕ)
  (h1 : dave_money = 46)
  (h2 : kyle_initial = 3 * dave_money - 12)
  (h3 : kyle_remaining = kyle_initial - kyle_initial / 3) :
  kyle_remaining = 84 :=
by
  -- Define Dave's money and provide the assumption
  let dave_money := 46
  have h1 : dave_money = 46 := rfl

  -- Define Kyle's initial money based on Dave's money
  let kyle_initial := 3 * dave_money - 12
  have h2 : kyle_initial = 3 * dave_money - 12 := rfl

  -- Define Kyle's remaining money after spending one third on snowboarding
  let kyle_remaining := kyle_initial - kyle_initial / 3
  have h3 : kyle_remaining = kyle_initial - kyle_initial / 3 := rfl

  -- Now we prove that Kyle's remaining money is 84
  sorry -- Proof steps omitted

end kyle_money_l47_47355


namespace ab_value_is_3360_l47_47392

noncomputable def find_ab (a b : ℤ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0) ∧
  (∃ r s : ℤ, 
    (x : ℤ) → 
      (x^3 + a * x^2 + b * x + 16 * a = (x - r)^2 * (x - s)) ∧ 
      (2 * r + s = -a) ∧ 
      (r^2 + 2 * r * s = b) ∧ 
      (r^2 * s = -16 * a))

theorem ab_value_is_3360 (a b : ℤ) (h : find_ab a b) : |a * b| = 3360 :=
sorry

end ab_value_is_3360_l47_47392


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47525

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47525


namespace g_2023_eq_0_l47_47616

noncomputable def g (x : ℕ) : ℝ := sorry

axiom g_defined (x : ℕ) : ∃ y : ℝ, g x = y

axiom g_initial : g 1 = 1

axiom g_functional (a b : ℕ) : g (a + b) = g a + g b - 2 * g (a * b + 1)

theorem g_2023_eq_0 : g 2023 = 0 :=
sorry

end g_2023_eq_0_l47_47616


namespace scientific_notation_135000_l47_47256

theorem scientific_notation_135000 :
  135000 = 1.35 * 10^5 := sorry

end scientific_notation_135000_l47_47256


namespace sin_315_degree_l47_47483

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l47_47483


namespace find_m_l47_47592

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l47_47592


namespace sin_315_eq_neg_sqrt2_div_2_l47_47516

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l47_47516


namespace find_m_l47_47577

open Nat  

-- Definition of lcm in Lean, if it's not already provided in Mathlib
def lcm (a b : Nat) : Nat := (a * b) / gcd a b

theorem find_m (m : ℕ) (h1 : 0 < m)
    (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180):
    m = 60 :=
sorry

end find_m_l47_47577


namespace simplify_expression_l47_47127

theorem simplify_expression :
  (3 * (Real.sqrt 3 + Real.sqrt 5)) / (4 * Real.sqrt (3 + Real.sqrt 4))
  = (3 * Real.sqrt 15 + 3 * Real.sqrt 5) / 20 :=
by
  sorry

end simplify_expression_l47_47127


namespace range_of_m_min_value_a2_2b2_3c2_l47_47888

theorem range_of_m (x m : ℝ) (h : ∀ x : ℝ, abs (x + 3) + abs (x + m) ≥ 2 * m) : m ≤ 1 :=
sorry

theorem min_value_a2_2b2_3c2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  ∃ (a b c : ℝ), a = 6/11 ∧ b = 3/11 ∧ c = 2/11 ∧ a^2 + 2 * b^2 + 3 * c^2 = 6/11 :=
sorry

end range_of_m_min_value_a2_2b2_3c2_l47_47888


namespace circle_radius_proof_l47_47884

def circle_radius : Prop :=
  let D := -2
  let E := 3
  let F := -3 / 4
  let r := 1 / 2 * Real.sqrt (D^2 + E^2 - 4 * F)
  r = 2

theorem circle_radius_proof : circle_radius :=
  sorry

end circle_radius_proof_l47_47884


namespace sum_of_first_9_terms_l47_47137

variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)
variable (a1 : ℤ)
variable (d : ℤ)

-- Given is that the sequence is arithmetic.
-- Given a1 is the first term, and d is the common difference, we can define properties based on the conditions.
def is_arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a1 + (n - 1) * d

def sum_first_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Given condition: 2a_1 + a_13 = -9.
def given_condition (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ) : Prop :=
  2 * a1 + (a1 + 12 * d) = -9

theorem sum_of_first_9_terms (a : ℕ → ℤ) (S : ℕ → ℤ) (a1 d : ℤ)
  (h_arith : is_arithmetic_sequence a a1 d)
  (h_sum : sum_first_n_terms S a)
  (h_cond : given_condition a a1 d) :
  S 9 = -27 :=
sorry

end sum_of_first_9_terms_l47_47137


namespace diametrically_opposite_number_l47_47694

theorem diametrically_opposite_number (n k : ℕ) (h1 : n = 100) (h2 : k = 83) 
  (h3 : ∀ k ∈ finset.range n, 
        let m := (k - 1) / 2 in 
        let points_lt_k := finset.range k in 
        points_lt_k.card = 2 * m) : 
  True → 
  (k + 1 = 84) :=
by
  sorry

end diametrically_opposite_number_l47_47694


namespace cornbread_pieces_count_l47_47207

def cornbread_pieces (pan_length pan_width piece_length piece_width : ℕ) : ℕ := 
  (pan_length * pan_width) / (piece_length * piece_width)

theorem cornbread_pieces_count :
  cornbread_pieces 24 20 3 3 = 53 :=
by
  -- The definitions and the equivalence transformation tell us that this is true
  sorry

end cornbread_pieces_count_l47_47207


namespace find_f_neg1_l47_47316

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def f : ℝ → ℝ
| x => if 0 < x then x^2 + 2 else if x = 0 then 2 else -(x^2 + 2)

axiom odd_f : is_odd_function f

theorem find_f_neg1 : f (-1) = -3 := by
  sorry

end find_f_neg1_l47_47316


namespace num_dimes_is_3_l47_47082

noncomputable def num_dimes (pennies nickels dimes quarters : ℕ) : ℕ :=
  dimes

theorem num_dimes_is_3 (h_total_coins : pennies + nickels + dimes + quarters = 11)
  (h_total_value : pennies + 5 * nickels + 10 * dimes + 25 * quarters = 118)
  (h_at_least_one_each : 0 < pennies ∧ 0 < nickels ∧ 0 < dimes ∧ 0 < quarters) :
  num_dimes pennies nickels dimes quarters = 3 :=
sorry

end num_dimes_is_3_l47_47082


namespace number_of_children_l47_47048

-- Definitions based on the conditions provided in the problem
def total_population : ℕ := 8243
def grown_ups : ℕ := 5256

-- Statement of the proof problem
theorem number_of_children : total_population - grown_ups = 2987 := by
-- This placeholder 'sorry' indicates that the proof is omitted.
sorry

end number_of_children_l47_47048


namespace fraction_addition_l47_47445

theorem fraction_addition (a b c d : ℚ) (ha : a = 2/5) (hb : b = 3/8) (hc : c = 31/40) :
  a + b = c :=
by
  rw [ha, hb, hc]
  -- The proof part is skipped here as per instructions
  sorry

end fraction_addition_l47_47445


namespace factorization_25x2_minus_155x_minus_150_l47_47395

theorem factorization_25x2_minus_155x_minus_150 :
  ∃ (a b : ℤ), (a + b) * 5 = -155 ∧ a * b = -150 ∧ a + 2 * b = 27 :=
by
  sorry

end factorization_25x2_minus_155x_minus_150_l47_47395


namespace num_people_on_boats_l47_47257

-- Definitions based on the conditions
def boats := 5
def people_per_boat := 3

-- Theorem stating the problem to be solved
theorem num_people_on_boats : boats * people_per_boat = 15 :=
by sorry

end num_people_on_boats_l47_47257


namespace product_a2_a3_a4_l47_47741

open Classical

noncomputable def geometric_sequence (a : ℕ → ℚ) (a1 : ℚ) (q : ℚ) : Prop :=
∀ n : ℕ, a n = a1 * q^(n - 1)

theorem product_a2_a3_a4 (a : ℕ → ℚ) (q : ℚ) 
  (h_seq : geometric_sequence a 1 q)
  (h_a1 : a 1 = 1)
  (h_a5 : a 5 = 1 / 9) :
  a 2 * a 3 * a 4 = 1 / 27 :=
sorry

end product_a2_a3_a4_l47_47741


namespace second_dog_average_miles_l47_47112

theorem second_dog_average_miles
  (total_miles_week : ℕ)
  (first_dog_miles_day : ℕ)
  (days_in_week : ℕ)
  (h1 : total_miles_week = 70)
  (h2 : first_dog_miles_day = 2)
  (h3 : days_in_week = 7) :
  (total_miles_week - (first_dog_miles_day * days_in_week)) / days_in_week = 8 := sorry

end second_dog_average_miles_l47_47112


namespace train_length_l47_47278

def speed_kmph := 72   -- Speed in kilometers per hour
def time_sec := 14     -- Time in seconds

/-- Function to convert speed from km/hr to m/s -/
def convert_speed (speed : ℕ) : ℕ :=
  speed * 1000 / 3600

/-- Function to calculate distance given speed and time -/
def calculate_distance (speed : ℕ) (time : ℕ) : ℕ :=
  speed * time

theorem train_length :
  calculate_distance (convert_speed speed_kmph) time_sec = 280 :=
by
  sorry

end train_length_l47_47278


namespace ratio_B_to_A_l47_47090

def work_together_rate : Real := 0.75
def days_for_A : Real := 4

theorem ratio_B_to_A : 
  ∃ (days_for_B : Real), 
    (1/days_for_A + 1/days_for_B = work_together_rate) → 
    (days_for_B / days_for_A = 0.5) :=
by 
  sorry

end ratio_B_to_A_l47_47090


namespace geometric_sequence_sufficient_and_necessary_l47_47360

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_sufficient_and_necessary (a : ℕ → ℝ) (h1 : a 0 > 0) :
  (a 0 < a 1) ↔ (is_geometric_sequence a ∧ is_increasing_sequence a) :=
sorry

end geometric_sequence_sufficient_and_necessary_l47_47360


namespace geometric_sequence_second_term_l47_47270

theorem geometric_sequence_second_term (a r : ℕ) (h1 : a = 5) (h2 : a * r^4 = 1280) : a * r = 20 :=
by
  sorry

end geometric_sequence_second_term_l47_47270


namespace fraction_addition_l47_47440

def fraction_sum : ℚ := (2 : ℚ)/5 + (3 : ℚ)/8

theorem fraction_addition : fraction_sum = 31/40 := by
  sorry

end fraction_addition_l47_47440


namespace rice_grain_difference_l47_47375

theorem rice_grain_difference :
  (3^8) - (3^1 + 3^2 + 3^3 + 3^4 + 3^5) = 6198 :=
by
  sorry

end rice_grain_difference_l47_47375


namespace sin_315_eq_neg_sqrt2_div_2_l47_47501

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47501


namespace jessie_problem_l47_47005

def round_to_nearest_five (n : ℤ) : ℤ :=
  if n % 5 = 0 then n
  else if n % 5 < 3 then n - (n % 5)
  else n - (n % 5) + 5

theorem jessie_problem :
  round_to_nearest_five ((82 + 56) - 15) = 125 :=
by
  sorry

end jessie_problem_l47_47005


namespace total_pencils_sold_l47_47936

theorem total_pencils_sold:
  let first_two := 2 * 2 in
  let next_six := 6 * 3 in
  let last_two := 2 * 1 in
  first_two + next_six + last_two = 24 :=
by
  let first_two := 2 * 2
  let next_six := 6 * 3
  let last_two := 2 * 1
  calc
    first_two + next_six + last_two = 4 + 18 + 2 := by sorry
    ... = 24 := by sorry

end total_pencils_sold_l47_47936


namespace shaded_triangle_ratio_is_correct_l47_47097

noncomputable def ratio_of_shaded_triangle_to_large_square (total_area : ℝ) 
  (midpoint_area_ratio : ℝ := 1 / 24) : ℝ :=
  midpoint_area_ratio * total_area

theorem shaded_triangle_ratio_is_correct 
  (shaded_area total_area : ℝ)
  (n : ℕ)
  (h1 : n = 36)
  (grid_area : ℝ)
  (condition1 : grid_area = total_area / n)
  (condition2 : shaded_area = grid_area / 2 * 3)
  : shaded_area / total_area = 1 / 24 :=
by
  sorry

end shaded_triangle_ratio_is_correct_l47_47097


namespace no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares_l47_47805

theorem no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares :
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ a b c : ℕ, 2 * n^2 + 1 = a^2 ∧ 3 * n^2 + 1 = b^2 ∧ 6 * n^2 + 1 = c^2 := by
  sorry

end no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares_l47_47805


namespace annual_profit_growth_rate_l47_47603

variable (a : ℝ)

theorem annual_profit_growth_rate (ha : a > -1) : 
  (1 + a) ^ 12 - 1 = (1 + a) ^ 12 - 1 := 
by 
  sorry

end annual_profit_growth_rate_l47_47603


namespace sin_315_eq_neg_sqrt2_div_2_l47_47503

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47503


namespace find_m_l47_47581

open Nat

theorem find_m (m : ℕ) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 60 := 
by
  sorry  -- Proof goes here

end find_m_l47_47581


namespace pizzeria_large_pizzas_l47_47661

theorem pizzeria_large_pizzas (price_small : ℕ) (price_large : ℕ) (total_revenue : ℕ) (small_pizzas_sold : ℕ) (L : ℕ) 
    (h1 : price_small = 2) 
    (h2 : price_large = 8) 
    (h3 : total_revenue = 40) 
    (h4 : small_pizzas_sold = 8) 
    (h5 : price_small * small_pizzas_sold + price_large * L = total_revenue) :
    L = 3 := 
by 
  -- Lean will expect a proof here; add sorry for now
  sorry

end pizzeria_large_pizzas_l47_47661


namespace min_small_bottles_needed_l47_47974

theorem min_small_bottles_needed (small_capacity large_capacity : ℕ) 
    (h_small_capacity : small_capacity = 35) (h_large_capacity : large_capacity = 500) : 
    ∃ n, n = 15 ∧ large_capacity <= n * small_capacity :=
by 
  sorry

end min_small_bottles_needed_l47_47974


namespace second_dog_average_miles_l47_47111

theorem second_dog_average_miles
  (total_miles_week : ℕ)
  (first_dog_miles_day : ℕ)
  (days_in_week : ℕ)
  (h1 : total_miles_week = 70)
  (h2 : first_dog_miles_day = 2)
  (h3 : days_in_week = 7) :
  (total_miles_week - (first_dog_miles_day * days_in_week)) / days_in_week = 8 := sorry

end second_dog_average_miles_l47_47111


namespace find_m_l47_47589

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l47_47589


namespace sin_315_eq_neg_sqrt2_div_2_l47_47531

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47531


namespace probability_smallest_divides_larger_two_l47_47821

noncomputable def number_of_ways := 20

noncomputable def successful_combinations := 11

theorem probability_smallest_divides_larger_two : (successful_combinations : ℚ) / number_of_ways = 11 / 20 :=
by
  sorry

end probability_smallest_divides_larger_two_l47_47821


namespace time_after_classes_l47_47000

def time_after_maths : Nat := 60
def time_after_history : Nat := 60 + 90
def time_after_break1 : Nat := time_after_history + 25
def time_after_geography : Nat := time_after_break1 + 45
def time_after_break2 : Nat := time_after_geography + 15
def time_after_science : Nat := time_after_break2 + 75

theorem time_after_classes (start_time : Nat := 12 * 60) : (start_time + time_after_science) % 1440 = 17 * 60 + 10 :=
by
  sorry

end time_after_classes_l47_47000


namespace meet_probability_zero_l47_47218

/-- Define the setup conditions for objects A and B. -/
def movement_condition (A B : ℕ × ℕ) : Prop :=
  (A = (0, 0) ∧ B = (6, 8) ∧ 
  (∀ n, A.1 + A.2 = 4 → B.1 + B.2 = 4) ∧ 
  (A.1 = 6 - B.1) ∧ (A.2 = 8 - B.2))

/-- Define the function to calculate the probability. -/
def meeting_probability (steps : ℕ) : ℚ :=
  if ∃ (A B : ℕ × ℕ), movement_condition A B then 1 / (2 ^ (2 * steps)) else 0

/-- The probability that A and B meet after moving exactly four steps is zero. -/
theorem meet_probability_zero : meeting_probability 4 = 0 :=
sorry

end meet_probability_zero_l47_47218


namespace range_mn_squared_l47_47863

-- Let's define the conditions in Lean

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f is strictly increasing
axiom h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0

-- Condition 2: f(x-1) is centrally symmetric about (1,0)
axiom h2 : ∀ x : ℝ, f (x - 1) = - f (2 - (x - 1))

-- Condition 3: Given inequality
axiom h3 : ∀ m n : ℝ, f (m^2 - 6*m + 21) + f (n^2 - 8*n) < 0

-- Prove the range for m^2 + n^2 is (9, 49)
theorem range_mn_squared : ∀ m n : ℝ, f (m^2 - 6*m + 21) + f (n^2 - 8*n) < 0 →
  9 < m^2 + n^2 ∧ m^2 + n^2 < 49 :=
sorry

end range_mn_squared_l47_47863


namespace simplify_and_evaluate_l47_47225

theorem simplify_and_evaluate (a : ℕ) (h : a = 2023) : (a + 1) / a / (a - 1 / a) = 1 / 2022 :=
by
  sorry

end simplify_and_evaluate_l47_47225


namespace house_number_count_l47_47126

noncomputable def count_valid_house_numbers : Nat :=
  let two_digit_primes := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let valid_combinations := two_digit_primes.product two_digit_primes |>.filter (λ (WX, YZ) => WX ≠ YZ)
  valid_combinations.length

theorem house_number_count : count_valid_house_numbers = 110 :=
  by
    sorry

end house_number_count_l47_47126


namespace jess_father_first_round_l47_47004

theorem jess_father_first_round (initial_blocks : ℕ)
  (players : ℕ)
  (blocks_before_jess_turn : ℕ)
  (jess_falls_tower_round : ℕ)
  (h1 : initial_blocks = 54)
  (h2 : players = 5)
  (h3 : blocks_before_jess_turn = 28)
  (h4 : ∀ rounds : ℕ, rounds * players ≥ 26 → jess_falls_tower_round = rounds + 1) :
  jess_falls_tower_round = 6 := 
by
  sorry

end jess_father_first_round_l47_47004


namespace find_x_y_z_l47_47639

theorem find_x_y_z (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x * y = x + y) (h2 : y * z = 3 * (y + z)) (h3 : z * x = 2 * (z + x)) : 
  x + y + z = 12 :=
sorry

end find_x_y_z_l47_47639


namespace find_legs_of_triangle_l47_47274

theorem find_legs_of_triangle (a b : ℝ) (h : a / b = 3 / 4) (h_sum : a^2 + b^2 = 70^2) : 
  (a = 42) ∧ (b = 56) :=
sorry

end find_legs_of_triangle_l47_47274


namespace cos_alpha_value_l47_47565

-- Define our conditions
variables (α : ℝ)
axiom sin_alpha : Real.sin α = -5 / 13
axiom tan_alpha_pos : Real.tan α > 0

-- State our goal
theorem cos_alpha_value : Real.cos α = -12 / 13 :=
by
  sorry

end cos_alpha_value_l47_47565


namespace sin_315_eq_neg_sqrt2_div_2_l47_47470

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47470


namespace problem_equivalent_l47_47318

variable (f : ℝ → ℝ)

theorem problem_equivalent (h₁ : ∀ x, deriv f x = deriv (deriv f) x)
                            (h₂ : ∀ x, deriv (deriv f) x < f x) : 
                            f 2 < Real.exp 2 * f 0 ∧ f 2017 < Real.exp 2017 * f 0 := sorry

end problem_equivalent_l47_47318


namespace cost_price_watch_l47_47251

variable (cost_price : ℚ)

-- Conditions
def sold_at_loss (cost_price : ℚ) := 0.90 * cost_price
def sold_at_gain (cost_price : ℚ) := 1.03 * cost_price
def price_difference (cost_price : ℚ) := sold_at_gain cost_price - sold_at_loss cost_price = 140

-- Theorem
theorem cost_price_watch (h : price_difference cost_price) : cost_price = 1076.92 := by
  sorry

end cost_price_watch_l47_47251


namespace intersection_of_A_and_B_l47_47742

def A : Set ℝ := {x | x - 1 > 1}
def B : Set ℝ := {x | x < 3}

theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_of_A_and_B_l47_47742


namespace find_m_l47_47578

open Nat

theorem find_m (m : ℕ) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 60 := 
by
  sorry  -- Proof goes here

end find_m_l47_47578


namespace rationalize_denominator_l47_47031

theorem rationalize_denominator :
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = -9 - 4 * Real.sqrt 5 :=
by
  -- Commutative field properties and algebraic manipulation will be used here.
  sorry

end rationalize_denominator_l47_47031


namespace both_selected_prob_l47_47957

-- Given conditions
def prob_Ram := 6 / 7
def prob_Ravi := 1 / 5

-- The mathematically equivalent proof problem statement
theorem both_selected_prob : (prob_Ram * prob_Ravi) = 6 / 35 := by
  sorry

end both_selected_prob_l47_47957


namespace max_f_geq_fraction_3_sqrt3_over_2_l47_47998

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (2 * x) + Real.sin (3 * x)

theorem max_f_geq_fraction_3_sqrt3_over_2 : ∃ x : ℝ, f x ≥ (3 + Real.sqrt 3) / 2 := 
sorry

end max_f_geq_fraction_3_sqrt3_over_2_l47_47998


namespace soap_remaining_days_l47_47962

theorem soap_remaining_days 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0)
  (daily_consumption : ℝ)
  (h4 : daily_consumption = a * b * c / 8) 
  (h5 : ∀ t : ℝ, t > 0 → t ≤ 7 → daily_consumption = (a * b * c - (a * b * c) * (1 / 8))) :
  ∃ t : ℝ, t = 1 :=
by 
  sorry

end soap_remaining_days_l47_47962


namespace sin_315_degree_is_neg_sqrt_2_div_2_l47_47493

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l47_47493


namespace min_students_with_same_score_l47_47197

noncomputable def highest_score : ℕ := 83
noncomputable def lowest_score : ℕ := 30
noncomputable def total_students : ℕ := 8000
noncomputable def range_scores : ℕ := (highest_score - lowest_score + 1)

theorem min_students_with_same_score :
  ∃ k : ℕ, k = Nat.ceil (total_students / range_scores) ∧ k = 149 :=
by
  sorry

end min_students_with_same_score_l47_47197


namespace negation_proposition_l47_47660

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) :=
sorry

end negation_proposition_l47_47660


namespace mr_valentino_birds_l47_47380

theorem mr_valentino_birds : 
  ∀ (chickens ducks turkeys : ℕ), 
  chickens = 200 → 
  ducks = 2 * chickens → 
  turkeys = 3 * ducks → 
  chickens + ducks + turkeys = 1800 :=
by
  intros chickens ducks turkeys
  assume h1 : chickens = 200
  assume h2 : ducks = 2 * chickens
  assume h3 : turkeys = 3 * ducks
  sorry

end mr_valentino_birds_l47_47380


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47511

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47511


namespace sum_of_common_divisors_36_48_l47_47725

-- Definitions based on the conditions
def is_divisor (n d : ℕ) : Prop := d ∣ n

-- List of divisors for 36 and 48
def divisors_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]
def divisors_48 : List ℕ := [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]

-- Definition of common divisors
def common_divisors_36_48 : List ℕ := [1, 2, 3, 4, 6, 12]

-- Sum of common divisors
def sum_common_divisors_36_48 := common_divisors_36_48.sum

-- The statement of the theorem
theorem sum_of_common_divisors_36_48 : sum_common_divisors_36_48 = 28 := by
  sorry

end sum_of_common_divisors_36_48_l47_47725


namespace rationalize_denominator_ABC_l47_47028

theorem rationalize_denominator_ABC :
  (let A := -9 in let B := -4 in let C := 5 in A * B * C) = 180 :=
by
  let expr := (2 + Real.sqrt 5) / (2 - Real.sqrt 5)
  let numerator := (2 + Real.sqrt 5) * (2 + Real.sqrt 5)
  let denominator := (2 - Real.sqrt 5) * (2 + Real.sqrt 5)
  let simplified_expr := numerator / denominator
  have h1 : numerator = 9 + 4 * Real.sqrt 5 := sorry
  have h2 : denominator = -1 := sorry
  have h3 : simplified_expr = -9 - 4 * Real.sqrt 5 := by
    rw [h1, h2]
    simp
  have hA : -9 = A := rfl
  have hB : -4 = B := rfl
  have hC : 5 = C := rfl
  have hABC : -9 * -4 * 5 = 180 := by 
    rw [hA, hB, hC]
    ring
  exact hABC

end rationalize_denominator_ABC_l47_47028


namespace sin_315_eq_neg_sqrt2_div_2_l47_47513

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l47_47513


namespace tan_add_pi_over_3_l47_47157

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l47_47157


namespace trip_to_Atlanta_equals_Boston_l47_47672

def distance_to_Boston : ℕ := 840
def daily_distance : ℕ := 40
def num_days (distance : ℕ) (daily : ℕ) : ℕ := distance / daily
def distance_to_Atlanta (days : ℕ) (daily : ℕ) : ℕ := days * daily

theorem trip_to_Atlanta_equals_Boston :
  distance_to_Atlanta (num_days distance_to_Boston daily_distance) daily_distance = distance_to_Boston :=
by
  -- Here we would insert the proof.
  sorry

end trip_to_Atlanta_equals_Boston_l47_47672


namespace repeating_decimal_sum_numerator_denominator_l47_47952

open Rat

theorem repeating_decimal_sum_numerator_denominator : 
  let x := 0.\overline{134} in
  let f := { ratNum := 134, ratDenom := 999, H := by norm_num } in -- Define the fraction 134/999
  Rat.add f.num f.denom = 1133 := 
by sorry

end repeating_decimal_sum_numerator_denominator_l47_47952


namespace sale_in_second_month_l47_47965

theorem sale_in_second_month 
  (sale_first_month: ℕ := 2500)
  (sale_third_month: ℕ := 3540)
  (sale_fourth_month: ℕ := 1520)
  (average_sale: ℕ := 2890)
  (total_sales: ℕ := 11560) :
  sale_first_month + sale_third_month + sale_fourth_month + (sale_second_month: ℕ) = total_sales → 
  sale_second_month = 4000 := 
by
  intros h
  sorry

end sale_in_second_month_l47_47965


namespace tan_add_pi_div_three_l47_47174

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l47_47174


namespace expression_value_l47_47114

theorem expression_value : 200 * (200 - 8) - (200 * 200 + 8) = -1608 := 
by 
  -- We will put the proof here
  sorry

end expression_value_l47_47114


namespace minimum_small_bottles_l47_47976

-- Define the capacities of the bottles
def small_bottle_capacity : ℕ := 35
def large_bottle_capacity : ℕ := 500

-- Define the number of small bottles needed to fill a large bottle
def small_bottles_needed_to_fill_large : ℕ := 
  (large_bottle_capacity + small_bottle_capacity - 1) / small_bottle_capacity

-- Statement of the theorem
theorem minimum_small_bottles : small_bottles_needed_to_fill_large = 15 := by
  sorry

end minimum_small_bottles_l47_47976


namespace graph_passes_through_point_l47_47140

theorem graph_passes_through_point (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) :
  ∃ x y, (x, y) = (0, 3) ∧ (∀ f : ℝ → ℝ, (∀ y, (f y = a ^ y) → (0, f 0 + 2) = (0, 3))) :=
by
  sorry

end graph_passes_through_point_l47_47140


namespace opposite_of_neg_six_is_six_l47_47932

theorem opposite_of_neg_six_is_six : 
  ∃ (x : ℝ), (-6 + x = 0) ∧ x = 6 := by
  sorry

end opposite_of_neg_six_is_six_l47_47932


namespace sum_digits_of_consecutive_numbers_l47_47958

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

end sum_digits_of_consecutive_numbers_l47_47958


namespace Vinnie_exceeded_word_limit_l47_47412

theorem Vinnie_exceeded_word_limit :
  let words_limit := 1000
  let words_saturday := 450
  let words_sunday := 650
  let total_words := words_saturday + words_sunday
  total_words - words_limit = 100 :=
by
  sorry

end Vinnie_exceeded_word_limit_l47_47412


namespace hydra_survival_l47_47056

-- The initial number of heads for both hydras
def initial_heads_hydra_A : ℕ := 2016
def initial_heads_hydra_B : ℕ := 2017

-- Weekly head growth possibilities
def growth_values : set ℕ := {5, 7}

-- Death condition: The hydras die if their head counts become equal.
def death_condition (heads_A heads_B : ℕ) : Prop := heads_A = heads_B

-- The problem statement to prove
theorem hydra_survival : ∀ (weeks : ℕ) (growth_A growth_B : ℕ),
  growth_A ∈ growth_values →
  growth_B ∈ growth_values →
  ¬ death_condition 
    (initial_heads_hydra_A + weeks * growth_A - 2 * weeks)
    (initial_heads_hydra_B + weeks * growth_B - 2 * weeks) :=
by
  sorry

end hydra_survival_l47_47056


namespace pool_fill_time_with_P_and_Q_l47_47718

noncomputable def pool_fill_time_estimation (p q r : ℝ) (h1 : p + q + r = 1/2) 
  (h2 : p + r = 1/3) (h3 : q + r = 1/4) : ℝ :=
1 / (p + q)

theorem pool_fill_time_with_P_and_Q (p q r : ℝ) (h1 : p + q + r = 1/2) 
  (h2 : p + r = 1/3) (h3 : q + r = 1/4) : pool_fill_time_estimation p q r h1 h2 h3 = 2.4 :=
by 
  let q := 1/6
  let p := 1/4
  show pool_fill_time_estimation p q r h1 h2 h3 = 2.4
  sorry

end pool_fill_time_with_P_and_Q_l47_47718


namespace tan_angle_addition_l47_47170

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l47_47170


namespace expectation_and_variance_of_affine_transformation_l47_47768

theorem expectation_and_variance_of_affine_transformation
  (X : Type) [probability_space X] 
  (f : X → ℝ) [is_discrete f] :
  let Y := λ x, 3 * f x + 2 in
  E(Y) = 3 * E(f) + 2 ∧ D(Y) = 9 * D(f) :=
by
  unfold is_discrete
  sorry

end expectation_and_variance_of_affine_transformation_l47_47768


namespace sum_of_six_consecutive_integers_l47_47287

theorem sum_of_six_consecutive_integers (m : ℤ) : 
  (m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5)) = 6 * m + 15 :=
by
  sorry

end sum_of_six_consecutive_integers_l47_47287


namespace hydra_survival_l47_47057

-- The initial number of heads for both hydras
def initial_heads_hydra_A : ℕ := 2016
def initial_heads_hydra_B : ℕ := 2017

-- Weekly head growth possibilities
def growth_values : set ℕ := {5, 7}

-- Death condition: The hydras die if their head counts become equal.
def death_condition (heads_A heads_B : ℕ) : Prop := heads_A = heads_B

-- The problem statement to prove
theorem hydra_survival : ∀ (weeks : ℕ) (growth_A growth_B : ℕ),
  growth_A ∈ growth_values →
  growth_B ∈ growth_values →
  ¬ death_condition 
    (initial_heads_hydra_A + weeks * growth_A - 2 * weeks)
    (initial_heads_hydra_B + weeks * growth_B - 2 * weeks) :=
by
  sorry

end hydra_survival_l47_47057


namespace max_value_on_interval_l47_47042

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_value_on_interval : 
  ∀ (x : ℝ), x ∈ Icc 0 3 → f x ≤ 5 :=
begin
  sorry
end

end max_value_on_interval_l47_47042


namespace number_of_pens_l47_47662

theorem number_of_pens (x y : ℝ) (h1 : 60 * (x + 2 * y) = 50 * (x + 3 * y)) (h2 : x = 3 * y) : 
  (60 * (x + 2 * y)) / x = 100 :=
by
  sorry

end number_of_pens_l47_47662


namespace students_not_next_each_other_l47_47729

open Nat

theorem students_not_next_each_other (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 5) (h2 : k = 2) (h3 : m = 3)
  (h4 : ∀ (A B : ℕ), A ≠ B) : 
  ∃ (total : ℕ), total = 3! * (choose (5-3+1) 2) := 
by
  sorry

end students_not_next_each_other_l47_47729


namespace daniel_initial_noodles_l47_47988

theorem daniel_initial_noodles (noodles_given noodles_left : ℕ) 
  (h_given : noodles_given = 12)
  (h_left : noodles_left = 54) :
  ∃ x, x - noodles_given = noodles_left ∧ x = 66 := 
by 
sory

end daniel_initial_noodles_l47_47988


namespace hydrae_never_equal_heads_l47_47059

theorem hydrae_never_equal_heads :
  ∀ (a b : ℕ), a = 2016 → b = 2017 →
  (∀ (a' b' : ℕ), a' ∈ {5, 7} → b' ∈ {5, 7} → 
  ∀ n : ℕ, let aa := a + n * 5 + (n - a / 7) * 2 - n in
           let bb := b + n * 5 + (n - b / 7) * 2 - n in
  aa + bb ≠ 2 * (aa / 2)) → 
  true :=
begin
  -- Sorry, the proof is left as an exercise
  sorry,
end

end hydrae_never_equal_heads_l47_47059


namespace units_digit_of_expression_l47_47869

noncomputable def C : ℝ := 7 + Real.sqrt 50
noncomputable def D : ℝ := 7 - Real.sqrt 50

theorem units_digit_of_expression (C D : ℝ) (hC : C = 7 + Real.sqrt 50) (hD : D = 7 - Real.sqrt 50) : 
  ((C ^ 21 + D ^ 21) % 10) = 4 :=
  sorry

end units_digit_of_expression_l47_47869


namespace total_students_l47_47096

theorem total_students (boys girls : ℕ) (h_boys : boys = 127) (h_girls : girls = boys + 212) : boys + girls = 466 :=
by
  sorry

end total_students_l47_47096


namespace probability_of_green_l47_47289

open Classical

-- Define the total number of balls in each container
def balls_A := 12
def balls_B := 14
def balls_C := 12

-- Define the number of green balls in each container
def green_balls_A := 7
def green_balls_B := 6
def green_balls_C := 9

-- Define the probability of selecting each container
def prob_select_container := (1:ℚ) / 3

-- Define the probability of drawing a green ball from each container
def prob_green_A := green_balls_A / balls_A
def prob_green_B := green_balls_B / balls_B
def prob_green_C := green_balls_C / balls_C

-- Define the total probability of drawing a green ball
def total_prob_green := prob_select_container * prob_green_A +
                        prob_select_container * prob_green_B +
                        prob_select_container * prob_green_C

-- Create the proof statement
theorem probability_of_green : total_prob_green = 127 / 252 := 
by
  -- Skip the proof
  sorry

end probability_of_green_l47_47289


namespace area_triangle_PZQ_l47_47778

/-- 
In rectangle PQRS, side PQ measures 8 units and side QR measures 4 units.
Points X and Y are on side RS such that segment RX measures 2 units and
segment SY measures 3 units. Lines PX and QY intersect at point Z.
Prove the area of triangle PZQ is 128/3 square units.
-/

theorem area_triangle_PZQ {PQ QR RX SY : ℝ} (h1 : PQ = 8) (h2 : QR = 4) (h3 : RX = 2) (h4 : SY = 3) :
  let area_PZQ : ℝ := 8 * 4 / 2 * 8 / (3 * 2)
  area_PZQ = 128 / 3 :=
by
  sorry

end area_triangle_PZQ_l47_47778


namespace minimum_value_on_line_l47_47382

theorem minimum_value_on_line : ∃ (x y : ℝ), (x + y = 4) ∧ (∀ x' y', (x' + y' = 4) → (x^2 + y^2 ≤ x'^2 + y'^2)) ∧ (x^2 + y^2 = 8) :=
sorry

end minimum_value_on_line_l47_47382


namespace sin_315_degree_l47_47479

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l47_47479


namespace tan_ratio_l47_47370

theorem tan_ratio (p q : ℝ) 
  (h1: Real.sin (p + q) = 5 / 8)
  (h2: Real.sin (p - q) = 3 / 8) : Real.tan p / Real.tan q = 4 := 
by
  sorry

end tan_ratio_l47_47370


namespace star_vertex_angle_l47_47842

-- Defining a function that calculates the star vertex angle for odd n-sided concave regular polygon
theorem star_vertex_angle (n : ℕ) (hn_odd : n % 2 = 1) (hn_gt3 : 3 < n) : 
  (180 - 360 / n) = (n - 2) * 180 / n := 
sorry

end star_vertex_angle_l47_47842


namespace sin_315_equals_minus_sqrt2_div_2_l47_47477

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l47_47477


namespace tan_add_pi_div_three_l47_47179

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l47_47179


namespace num_seven_digit_palindromes_l47_47130

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_seven_digit_palindrome (x : ℕ) : Prop :=
  let a := x / 1000000 % 10 in
  let b := x / 100000 % 10 in
  let c := x / 10000 % 10 in
  let d := x / 1000 % 10 in
  let e := x / 100 % 10 in
  let f := x / 10 % 10 in
  let g := x % 10 in
  a ≠ 0 ∧ a = g ∧ b = f ∧ c = e

theorem num_seven_digit_palindromes : 
  ∃ n : ℕ, (∀ x : ℕ, is_seven_digit_palindrome x → 1 ≤ a (x / 1000000 % 10) ∧ is_digit (b) ∧ is_digit (c) ∧ is_digit (d)) → n = 9000 :=
sorry

end num_seven_digit_palindromes_l47_47130


namespace solution_quadrant_I_l47_47210

theorem solution_quadrant_I (c x y : ℝ) :
  (x - y = 2 ∧ c * x + y = 3 ∧ x > 0 ∧ y > 0) ↔ (-1 < c ∧ c < 3/2) := by
  sorry

end solution_quadrant_I_l47_47210


namespace grayson_travels_further_l47_47893

noncomputable def grayson_first_part_distance : ℝ := 25 * 1
noncomputable def grayson_second_part_distance : ℝ := 20 * 0.5
noncomputable def total_distance_grayson : ℝ := grayson_first_part_distance + grayson_second_part_distance

noncomputable def total_distance_rudy : ℝ := 10 * 3

theorem grayson_travels_further : (total_distance_grayson - total_distance_rudy) = 5 := by
  sorry

end grayson_travels_further_l47_47893


namespace random_event_is_B_l47_47830

axiom isCertain (event : Prop) : Prop
axiom isImpossible (event : Prop) : Prop
axiom isRandom (event : Prop) : Prop

def A : Prop := ∀ t, t is certain (the sun rises from the east at time t)
def B : Prop := ∃ t, t is random (encountering a red light at time t ∧ passing through traffic light intersection)
def C : Prop := ∀ (p1 p2 p3 : ℝ²), isCertain (non-collinear points p1 p2 p3 → ∃! c, c is circle passing through p1 p2 p3)
def D : Prop := ∀ (T : Triangle), isImpossible (sum_of_interior_angles T = 540)

theorem random_event_is_B : isRandom B :=
by
  sorry

end random_event_is_B_l47_47830


namespace compute_expression_l47_47118

/-- Definitions of parts of the expression --/
def expr1 := 6 ^ 2
def expr2 := 4 * 5
def expr3 := 2 ^ 3
def expr4 := 4 ^ 2 / 2

/-- Main statement to prove --/
theorem compute_expression : expr1 + expr2 - expr3 + expr4 = 56 := 
by
  sorry

end compute_expression_l47_47118


namespace sin_315_degree_l47_47488

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l47_47488


namespace sum_series_eq_11_div_18_l47_47860

theorem sum_series_eq_11_div_18 :
  (∑' n : ℕ, if n = 0 then 0 else 1 / (n * (n + 3))) = 11 / 18 :=
by
  sorry

end sum_series_eq_11_div_18_l47_47860


namespace solve_abs_ineq_l47_47724

theorem solve_abs_ineq (x : ℝ) : |(8 - x) / 4| < 3 ↔ 4 < x ∧ x < 20 := by
  sorry

end solve_abs_ineq_l47_47724


namespace distance_to_focus_l47_47653

theorem distance_to_focus (d : ℝ) (a : ℝ) (b : ℝ) :
  a = 2 ∧ b = 1 ∧ |12 - d| = 4 → d = 8 ∨ d = 16 :=
by
  intro h
  cases h with ha hb
  cases hb with hb1 hb2
  sorry

end distance_to_focus_l47_47653


namespace sin_315_degree_l47_47490

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l47_47490


namespace quadrilateral_side_length_eq_12_l47_47777

-- Definitions
def EF : ℝ := 7
def FG : ℝ := 15
def GH : ℝ := 7
def HE : ℝ := 12
def EH : ℝ := 12

-- Statement to prove that EH = 12 given the definition and conditions
theorem quadrilateral_side_length_eq_12
  (EF_eq : EF = 7)
  (FG_eq : FG = 15)
  (GH_eq : GH = 7)
  (HE_eq : HE = 12)
  (EH_eq : EH = 12) : 
  EH = 12 :=
sorry

end quadrilateral_side_length_eq_12_l47_47777


namespace smallest_value_A_B_C_D_l47_47933

theorem smallest_value_A_B_C_D :
  ∃ (A B C D : ℕ), 
  (A < B) ∧ (B < C) ∧ (C < D) ∧ -- A, B, C are in arithmetic sequence and B, C, D in geometric sequence
  (C = B + (B - A)) ∧  -- A, B, C form an arithmetic sequence with common difference d = B - A
  (C = (4 * B) / 3) ∧  -- Given condition
  (D = (4 * C) / 3) ∧ -- B, C, D form geometric sequence with common ratio 4/3
  ((∃ k, D = k * 9) ∧ -- D must be an integer, ensuring B must be divisible by 9
   A + B + C + D = 43) := 
sorry

end smallest_value_A_B_C_D_l47_47933


namespace find_angle_x_l47_47951

theorem find_angle_x (x : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ)
  (h₁ : α = 45)
  (h₂ : β = 3 * x)
  (h₃ : γ = x)
  (h₄ : α + β + γ = 180) :
  x = 33.75 :=
sorry

end find_angle_x_l47_47951


namespace probability_two_white_balls_is_4_over_15_l47_47261

-- Define the conditions of the problem
def total_balls : ℕ := 15
def white_balls_initial : ℕ := 8
def black_balls : ℕ := 7
def balls_drawn : ℕ := 2 -- Note: Even though not explicitly required, it's part of the context

-- Calculate the probability of drawing two white balls without replacement
noncomputable def probability_two_white_balls : ℚ :=
  (white_balls_initial / total_balls) * ((white_balls_initial - 1) / (total_balls - 1))

-- The theorem to prove
theorem probability_two_white_balls_is_4_over_15 :
  probability_two_white_balls = 4 / 15 := by
  sorry

end probability_two_white_balls_is_4_over_15_l47_47261


namespace straight_flush_probability_l47_47948

open Classical

noncomputable def number_of_possible_hands : ℕ := Nat.choose 52 5

noncomputable def number_of_straight_flushes : ℕ := 40 

noncomputable def probability_of_straight_flush : ℚ := number_of_straight_flushes / number_of_possible_hands

theorem straight_flush_probability :
  probability_of_straight_flush = 1 / 64974 := by
  sorry

end straight_flush_probability_l47_47948


namespace exists_convex_polygon_diagonals_l47_47417

theorem exists_convex_polygon_diagonals :
  ∃ n : ℕ, n * (n - 3) / 2 = 54 :=
by
  sorry

end exists_convex_polygon_diagonals_l47_47417


namespace chord_line_eq_l47_47882

open Real

def ellipse (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 9 = 1

def bisecting_point (a b : ℝ) : Prop :=
  a = 4 ∧ b = 2

theorem chord_line_eq :
  (∃ (k : ℝ), ∀ (x y : ℝ), ellipse x y → bisecting_point ((x + y) / 2) ((x + y) / 2) → y - 2 = k * (x - 4)) →
  (∃ (x y : ℝ), ellipse x y ∧ x + 2 * y - 8 = 0) :=
by
  sorry

end chord_line_eq_l47_47882


namespace find_n_if_roots_opposite_signs_l47_47125

theorem find_n_if_roots_opposite_signs :
  ∃ n : ℝ, (∀ x : ℝ, (x^2 + (n-2)*x) / (2*n*x - 4) = (n+1) / (n-1) → x = -x) →
    (n = (-1 + Real.sqrt 5) / 2 ∨ n = (-1 - Real.sqrt 5) / 2) :=
by
  sorry

end find_n_if_roots_opposite_signs_l47_47125


namespace count_lattice_right_triangles_with_incenter_l47_47646

def is_lattice_point (p : ℤ × ℤ) : Prop := ∃ x y : ℤ, p = (x, y)

def is_right_triangle (O A B : ℤ × ℤ) : Prop :=
  O = (0, 0) ∧ (O.1 = A.1 ∨ O.2 = A.2) ∧ (O.1 = B.1 ∨ O.2 = B.2) ∧
  (A.1 * B.2 - A.2 * B.1 ≠ 0) -- Ensure A and B are not collinear with O

def incenter (O A B : ℤ × ℤ) : ℤ × ℤ :=
  ((A.1 + B.1 - O.1) / 2, (A.2 + B.2 - O.2) / 2)

theorem count_lattice_right_triangles_with_incenter :
  let I := (2015, 7 * 2015)
  ∃ (O A B : ℤ × ℤ), is_right_triangle O A B ∧ incenter O A B = I :=
sorry

end count_lattice_right_triangles_with_incenter_l47_47646


namespace fitted_bowling_ball_volume_l47_47963

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

end fitted_bowling_ball_volume_l47_47963


namespace arithmetic_mean_difference_l47_47191

theorem arithmetic_mean_difference (p q r : ℝ)
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 20) : 
  r - p = 20 := 
by sorry

end arithmetic_mean_difference_l47_47191


namespace tan_add_pi_div_three_l47_47178

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l47_47178


namespace cat_mouse_position_after_299_moves_l47_47198

-- Definitions based on conditions
def cat_position (move : Nat) : Nat :=
  let active_moves := move - (move / 100)
  active_moves % 4

def mouse_position (move : Nat) : Nat :=
  move % 8

-- Main theorem
theorem cat_mouse_position_after_299_moves :
  cat_position 299 = 0 ∧ mouse_position 299 = 3 :=
by
  sorry

end cat_mouse_position_after_299_moves_l47_47198


namespace total_wait_time_l47_47076

def customs_wait : ℕ := 20
def quarantine_days : ℕ := 14
def hours_per_day : ℕ := 24

theorem total_wait_time :
  customs_wait + quarantine_days * hours_per_day = 356 := 
by
  sorry

end total_wait_time_l47_47076


namespace prism_volume_l47_47925

theorem prism_volume (x y z : ℝ) 
  (h1 : x * y = 24) 
  (h2 : x * z = 32) 
  (h3 : y * z = 48) : 
  x * y * z = 192 :=
by
  sorry

end prism_volume_l47_47925


namespace maximum_value_of_expression_l47_47366

noncomputable def calc_value (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2)

theorem maximum_value_of_expression :
  ∃ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 ∧ x ≥ y ∧ y ≥ z ∧
  calc_value x y z = 2916 / 729 :=
by
  sorry

end maximum_value_of_expression_l47_47366


namespace number_of_throwers_l47_47374

theorem number_of_throwers (total_players throwers right_handed : ℕ) 
  (h1 : total_players = 64)
  (h2 : right_handed = 55) 
  (h3 : ∀ T N, T + N = total_players → 
  T + (2/3 : ℚ) * N = right_handed) : 
  throwers = 37 := 
sorry

end number_of_throwers_l47_47374


namespace tan_angle_addition_l47_47169

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l47_47169


namespace evaluate_expression_l47_47720

theorem evaluate_expression : (3^3)^4 = 531441 :=
by sorry

end evaluate_expression_l47_47720


namespace diametrically_opposite_to_83_l47_47693

variable (n : ℕ) (k : ℕ)
variable (h1 : n = 100)
variable (h2 : 1 ≤ k ∧ k ≤ n)
variable (h_dist : ∀ k, k < 83 → (number_of_points_less_than_k_on_left + number_of_points_less_than_k_on_right = k - 1) ∧ number_of_points_less_than_k_on_left = number_of_points_less_than_k_on_right)

-- define what's "diametrically opposite" means
def opposite_number (n k : ℕ) : ℕ := 
  if (k + n/2) ≤ n then k + n/2 else (k + n/2) - n

-- the exact statement to demonstrate the problem above
theorem diametrically_opposite_to_83 (h1 : n = 100) (h2 : 1 ≤ k ∧ k ≤ n) (h3 : k = 83) : 
  opposite_number n k = 84 :=
by
  have h_opposite : opposite_number n 83 = 84 := sorry
  exact h_opposite


end diametrically_opposite_to_83_l47_47693


namespace find_x_l47_47307

theorem find_x (n : ℕ) (hn : Odd n) (hx : ∃ p1 p2 p3 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 11 ∈ {p1, p2, p3} ∧ (6^n + 1) = p1 * p2 * p3) :
  6^n + 1 = 7777 :=
by
  sorry

end find_x_l47_47307


namespace subset_condition_l47_47314

theorem subset_condition (m : ℝ) (A : Set ℝ) (B : Set ℝ) :
  A = {1, 3} ∧ B = {1, 2, m} ∧ A ⊆ B → m = 3 :=
by
  sorry

end subset_condition_l47_47314


namespace geometric_sequence_common_ratio_l47_47775

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} 
    (h1 : a 1 = 1) 
    (h4 : a 4 = 1 / 64) 
    (geom_seq : ∀ n, ∃ r, a (n + 1) = a n * r) : 
       
    ∃ q, (∀ n, a n = 1 * (q ^ (n - 1))) ∧ (a 4 = 1 * (q ^ 3)) ∧ q = 1 / 4 := 
by
    sorry

end geometric_sequence_common_ratio_l47_47775


namespace inserted_number_sq_property_l47_47237

noncomputable def inserted_number (n : ℕ) : ℕ :=
  (5 * 10^n - 1) * 10^(n+1) + 1

theorem inserted_number_sq_property (n : ℕ) : (inserted_number n)^2 = (10^(n+1) - 1)^2 :=
by sorry

end inserted_number_sq_property_l47_47237


namespace verify_conditions_l47_47250

-- Define the conditions as expressions
def condition_A (a : ℝ) : Prop := 2 * a * 3 * a = 6 * a
def condition_B (a b : ℝ) : Prop := 3 * a^2 * b - 3 * a * b^2 = 0
def condition_C (a : ℝ) : Prop := 6 * a / (2 * a) = 3
def condition_D (a : ℝ) : Prop := (-2 * a) ^ 3 = -6 * a^3

-- Prove which condition is correct
theorem verify_conditions (a b : ℝ) (h : a ≠ 0) : 
  ¬ condition_A a ∧ ¬ condition_B a b ∧ condition_C a ∧ ¬ condition_D a :=
by 
  sorry

end verify_conditions_l47_47250


namespace hydras_will_live_l47_47067

noncomputable def hydras_live : Prop :=
  let A_initial := 2016
  let B_initial := 2017
  let possible_growth := {5, 7}
  let weekly_death := 4
  ∀ (weeks : ℕ), 
    let A_heads := A_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    let B_heads := B_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    A_heads ≠ B_heads

theorem hydras_will_live : hydras_live :=
sorry

end hydras_will_live_l47_47067


namespace polynomial_expansion_sum_constants_l47_47329

theorem polynomial_expansion_sum_constants :
  ∃ (A B C D : ℤ), ((x - 3) * (4 * x ^ 2 + 2 * x - 7) = A * x ^ 3 + B * x ^ 2 + C * x + D) → A + B + C + D = 2 := 
by
  sorry

end polynomial_expansion_sum_constants_l47_47329


namespace tan_angle_addition_l47_47171

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l47_47171


namespace length_QF_l47_47607

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 8 * x

def focus : ℝ × ℝ := (2, 0)

def directrix (x y : ℝ) : Prop := x = 1 -- Directrix of the given parabola

def point_on_directrix (P : ℝ × ℝ) : Prop := directrix P.1 P.2

def point_on_parabola (Q : ℝ × ℝ) : Prop := parabola Q.1 Q.2

def point_on_line_PF (P F Q : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m ≠ 0 ∧ (Q.2 = m * (Q.1 - F.1) + F.2) ∧ point_on_parabola Q

def vector_equality (F P Q : ℝ × ℝ) : Prop :=
  (4 * (Q.1 - F.1), 4 * (Q.2 - F.2)) = (P.1 - F.1, P.2 - F.2)

theorem length_QF 
  (P Q : ℝ × ℝ)
  (hPd : point_on_directrix P)
  (hPQ : point_on_line_PF P focus Q)
  (hVec : vector_equality focus P Q) : 
  dist Q focus = 3 :=
by
  sorry

end length_QF_l47_47607


namespace complement_union_l47_47748

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- The proof problem statement
theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3}) (hA : A = {-1, 0, 1}) (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := sorry

end complement_union_l47_47748


namespace raft_capacity_l47_47941

theorem raft_capacity (total_without_life_jackets : ℕ) (reduction_with_life_jackets : ℕ)
  (people_needing_life_jackets : ℕ) (total_capacity_with_life_jackets : ℕ)
  (no_life_jackets_capacity : total_without_life_jackets = 21)
  (life_jackets_reduction : reduction_with_life_jackets = 7)
  (life_jackets_needed : people_needing_life_jackets = 8) :
  total_capacity_with_life_jackets = 14 :=
by
  -- Proof should be here
  sorry

end raft_capacity_l47_47941


namespace last_person_is_knight_l47_47352

def KnightLiarsGame1 (n : ℕ) : Prop :=
  let m := 10
  let p := 13
  (∀ i < n, (i % 2 = 0 → true) ∧ (i % 2 = 1 → true)) ∧ 
  (m % 2 ≠ p % 2 → (n - 1) % 2 = 1)

def KnightLiarsGame2 (n : ℕ) : Prop :=
  let m := 12
  let p := 9
  (∀ i < n, (i % 2 = 0 → true) ∧ (i % 2 = 1 → true)) ∧ 
  (m % 2 ≠ p % 2 → (n - 1) % 2 = 1)

theorem last_person_is_knight :
  ∃ n, KnightLiarsGame1 n ∧ KnightLiarsGame2 n :=
by 
  sorry

end last_person_is_knight_l47_47352


namespace part_I_part_II_l47_47885

def f (x : ℝ) : ℝ := x ^ 3 - 3 * x

theorem part_I (m : ℝ) (h : ∀ x : ℝ, f x ≠ m) : m < -2 ∨ m > 2 :=
sorry

theorem part_II (P : ℝ × ℝ) (hP : P = (2, -6)) :
  (∃ m b : ℝ, ∀ x : ℝ, (m * x + b = 0 ∧ m = 3 ∧ b = 0) ∨ 
                 (m * x + b = 24 * x - 54 ∧ P.2 = 24 * P.1 - 54)) :=
sorry

end part_I_part_II_l47_47885


namespace find_number_l47_47707

theorem find_number (x : Real) (h1 : (2 / 5) * 300 = 120) (h2 : 120 - (3 / 5) * x = 45) : x = 125 :=
by
  sorry

end find_number_l47_47707


namespace complement_union_l47_47749

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- The proof problem statement
theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3}) (hA : A = {-1, 0, 1}) (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := sorry

end complement_union_l47_47749


namespace probability_log2_x_between_1_and_2_l47_47971

noncomputable def probability_log_between : ℝ :=
  let favorable_range := (4:ℝ) - (2:ℝ)
  let total_range := (6:ℝ) - (0:ℝ)
  favorable_range / total_range

theorem probability_log2_x_between_1_and_2 :
  probability_log_between = 1 / 3 :=
sorry

end probability_log2_x_between_1_and_2_l47_47971


namespace minimum_small_bottles_l47_47977

-- Define the capacities of the bottles
def small_bottle_capacity : ℕ := 35
def large_bottle_capacity : ℕ := 500

-- Define the number of small bottles needed to fill a large bottle
def small_bottles_needed_to_fill_large : ℕ := 
  (large_bottle_capacity + small_bottle_capacity - 1) / small_bottle_capacity

-- Statement of the theorem
theorem minimum_small_bottles : small_bottles_needed_to_fill_large = 15 := by
  sorry

end minimum_small_bottles_l47_47977


namespace stream_speed_l47_47084

variable (D : ℝ) -- Distance rowed

theorem stream_speed (v : ℝ) (h : D / (60 - v) = 2 * (D / (60 + v))) : v = 20 :=
by
  sorry

end stream_speed_l47_47084


namespace complement_union_l47_47760

variable (U : Set ℤ)
variable (A : Set ℤ)
variable (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} :=
sorry

end complement_union_l47_47760


namespace inequality_solution_l47_47231

theorem inequality_solution (x : ℝ) :
  (6*x^2 + 24*x - 63) / ((3*x - 4)*(x + 5)) < 4 ↔ x ∈ Set.Ioo (-(5:ℝ)) (4 / 3) ∪ Set.Iio (5) ∪ Set.Ioi (4 / 3) := by
  sorry

end inequality_solution_l47_47231


namespace find_m_l47_47600

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l47_47600


namespace trigonometric_unique_solution_l47_47612

theorem trigonometric_unique_solution :
  (∃ x : ℝ, 0 ≤ x ∧ x < (π / 2) ∧ Real.sin x = 0.6 ∧ Real.cos x = 0.8) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < (π / 2) ∧ 0 ≤ y ∧ y < (π / 2) ∧ Real.sin x = 0.6 ∧ Real.cos x = 0.8 ∧
    Real.sin y = 0.6 ∧ Real.cos y = 0.8 → x = y) :=
by
  sorry

end trigonometric_unique_solution_l47_47612


namespace sin_315_eq_neg_sqrt2_over_2_l47_47542

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l47_47542


namespace competition_score_l47_47339

theorem competition_score
    (x : ℕ)
    (h1 : 20 ≥ x)
    (h2 : 5 * x - (20 - x) = 70) :
    x = 15 :=
sorry

end competition_score_l47_47339


namespace total_people_on_boats_l47_47259

-- Definitions based on the given conditions
def boats : Nat := 5
def people_per_boat : Nat := 3

-- Theorem statement to prove the total number of people on boats in the lake
theorem total_people_on_boats : boats * people_per_boat = 15 :=
by
  sorry

end total_people_on_boats_l47_47259


namespace intersection_of_lines_l47_47296

theorem intersection_of_lines : ∃ (x y : ℝ), (9 * x - 4 * y = 30) ∧ (7 * x + y = 11) ∧ (x = 2) ∧ (y = -3) := 
by
  sorry

end intersection_of_lines_l47_47296


namespace maximum_alpha_l47_47008

noncomputable def is_in_F (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f (3 * x) ≥ f (f (2 * x)) + x

theorem maximum_alpha :
  (∀ f : ℝ → ℝ, is_in_F f → ∀ x > 0, f x ≥ (1 / 2) * x) := 
by
  sorry

end maximum_alpha_l47_47008


namespace people_not_in_pool_l47_47006

noncomputable def total_people_karen_donald : ℕ := 2
noncomputable def children_karen_donald : ℕ := 6
noncomputable def total_people_tom_eva : ℕ := 2
noncomputable def children_tom_eva : ℕ := 4
noncomputable def legs_in_pool : ℕ := 16

theorem people_not_in_pool : total_people_karen_donald + children_karen_donald + total_people_tom_eva + children_tom_eva - (legs_in_pool / 2) = 6 := by
  sorry

end people_not_in_pool_l47_47006


namespace complement_union_eq_l47_47745

open Set

-- Definition of sets U, A, and B
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- Statement of the problem
theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 3} :=
by sorry

end complement_union_eq_l47_47745


namespace find_m_l47_47575

open Nat  

-- Definition of lcm in Lean, if it's not already provided in Mathlib
def lcm (a b : Nat) : Nat := (a * b) / gcd a b

theorem find_m (m : ℕ) (h1 : 0 < m)
    (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180):
    m = 60 :=
sorry

end find_m_l47_47575


namespace sam_investment_l47_47809

noncomputable def compound_interest (P: ℝ) (r: ℝ) (n: ℕ) (t: ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem sam_investment :
  compound_interest 3000 0.10 4 1 = 3311.44 :=
by
  sorry

end sam_investment_l47_47809


namespace project_completion_equation_l47_47091

variables (x : ℕ)

-- Project completion conditions
def person_A_time : ℕ := 12
def person_B_time : ℕ := 8
def A_initial_work_days : ℕ := 3

-- Work done by Person A when working alone for 3 days
def work_A_initial := (A_initial_work_days:ℚ) / person_A_time

-- Work done by Person A and B after the initial 3 days until completion
def combined_work_remaining := 
  (λ x:ℕ => ((x - A_initial_work_days):ℚ) * (1/person_A_time + 1/person_B_time))

-- The equation representing the total work done equals 1
theorem project_completion_equation (x : ℕ) : 
  (x:ℚ) / person_A_time + (x - A_initial_work_days:ℚ) / person_B_time = 1 :=
sorry

end project_completion_equation_l47_47091


namespace bride_older_than_groom_l47_47820

-- Define the ages of the bride and groom
variables (B G : ℕ)

-- Given conditions
def groom_age : Prop := G = 83
def total_age : Prop := B + G = 185

-- Theorem to prove how much older the bride is than the groom
theorem bride_older_than_groom (h1 : groom_age G) (h2 : total_age B G) : B - G = 19 :=
sorry

end bride_older_than_groom_l47_47820


namespace random_event_is_B_l47_47831

variable (isCertain : Event → Prop)
variable (isImpossible : Event → Prop)
variable (isRandom : Event → Prop)

variable (A : Event)
variable (B : Event)
variable (C : Event)
variable (D : Event)

-- Here we set the conditions as definitions in Lean 4:
def condition_A : isCertain A := sorry
def condition_B : isRandom B := sorry
def condition_C : isCertain C := sorry
def condition_D : isImpossible D := sorry

-- The theorem we need to prove:
theorem random_event_is_B : isRandom B := 
by
-- adding sorry to skip the proof
sorry

end random_event_is_B_l47_47831


namespace find_m_l47_47591

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l47_47591


namespace sin_315_eq_neg_sqrt2_over_2_l47_47546

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l47_47546


namespace johns_chore_homework_time_l47_47294

-- Definitions based on problem conditions
def cartoons_time : ℕ := 150  -- John's cartoon watching time in minutes
def chores_homework_per_10 : ℕ := 13  -- 13 minutes combined chores and homework per 10 minutes of cartoons
def cartoon_period : ℕ := 10  -- Per 10 minutes period

-- Theorem statement
theorem johns_chore_homework_time :
  cartoons_time / cartoon_period * chores_homework_per_10 = 195 :=
by sorry

end johns_chore_homework_time_l47_47294


namespace car_speed_l47_47420

theorem car_speed (v : ℝ) (h : (1 / v) = (1 / 100 + 2 / 3600)) : v = 3600 / 38 := 
by
  sorry

end car_speed_l47_47420


namespace people_in_room_l47_47945

theorem people_in_room (P C : ℚ) (H1 : (3 / 5) * P = (2 / 3) * C) (H2 : C / 3 = 5) : 
  P = 50 / 3 :=
by
  -- The proof would go here
  sorry

end people_in_room_l47_47945


namespace fraction_addition_l47_47441

def fraction_sum : ℚ := (2 : ℚ)/5 + (3 : ℚ)/8

theorem fraction_addition : fraction_sum = 31/40 := by
  sorry

end fraction_addition_l47_47441


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47508

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47508


namespace negative_exponent_example_l47_47838

theorem negative_exponent_example : 3^(-2 : ℤ) = (1 : ℚ) / (3^2) :=
by sorry

end negative_exponent_example_l47_47838


namespace topology_on_X_l47_47330

-- Define the universal set X
def X : Set ℕ := {1, 2, 3}

-- Sequences of candidate sets v
def v1 : Set (Set ℕ) := {∅, {1}, {3}, {1, 2, 3}}
def v2 : Set (Set ℕ) := {∅, {2}, {3}, {2, 3}, {1, 2, 3}}
def v3 : Set (Set ℕ) := {∅, {1}, {1, 2}, {1, 3}}
def v4 : Set (Set ℕ) := {∅, {1, 3}, {2, 3}, {3}, {1, 2, 3}}

-- Define the conditions that determine a topology
def isTopology (X : Set ℕ) (v : Set (Set ℕ)) : Prop :=
  X ∈ v ∧ ∅ ∈ v ∧ 
  (∀ (s : Set (Set ℕ)), s ⊆ v → ⋃₀ s ∈ v) ∧
  (∀ (s : Set (Set ℕ)), s ⊆ v → ⋂₀ s ∈ v)

-- The statement we want to prove
theorem topology_on_X : 
  isTopology X v2 ∧ isTopology X v4 :=
by
  sorry

end topology_on_X_l47_47330


namespace quadratic_range_l47_47325

open Real

theorem quadratic_range (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 4 * x1 - 2 = 0 ∧ a * x2^2 + 4 * x2 - 2 = 0) : 
  a > -2 ∧ a ≠ 0 :=
by 
  sorry

end quadratic_range_l47_47325


namespace circle_radius_tangent_l47_47789

theorem circle_radius_tangent (A B O M X : Type) (AB AM MB r : ℝ)
  (hL1 : AB = 2) (hL2 : AM = 1) (hL3 : MB = 1) (hMX : MX = 1/2)
  (hTangent1 : OX = 1/2 + r) (hTangent2 : OM = 1 - r)
  (hPythagorean : OM^2 + MX^2 = OX^2) :
  r = 1/3 :=
by
  sorry

end circle_radius_tangent_l47_47789


namespace linear_function_does_not_pass_first_quadrant_l47_47815

theorem linear_function_does_not_pass_first_quadrant (k b : ℝ) (h : ∀ x : ℝ, y = k * x + b) :
  k = -1 → b = -2 → ¬∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x + b :=
by
  sorry

end linear_function_does_not_pass_first_quadrant_l47_47815


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l47_47457

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l47_47457


namespace work_done_l47_47839

theorem work_done (m : ℕ) : 18 * 30 = m * 36 → m = 15 :=
by
  intro h  -- assume the equality condition
  have h1 : m = 15 := by
    -- We would solve for m here similarly to the solution given to derive 15
    sorry
  exact h1

end work_done_l47_47839


namespace flagpole_proof_l47_47847

noncomputable def flagpole_height (AC AD DE : ℝ) (h_ABC_DEC : (AC ≠ 0) ∧ (AC - AD ≠ 0) ∧ (DE ≠ 0)) : ℝ :=
  let DC := AC - AD
  let h_ratio := DE / DC
  h_ratio * AC

theorem flagpole_proof (AC AD DE : ℝ) (h_AC : AC = 4) (h_AD : AD = 3) (h_DE : DE = 1.8) 
  (h_ABC_DEC : (AC ≠ 0) ∧ (AC - AD ≠ 0) ∧ (DE ≠ 0)) :
  flagpole_height AC AD DE h_ABC_DEC = 7.2 := by
  sorry

end flagpole_proof_l47_47847


namespace sin_315_eq_neg_sqrt2_div_2_l47_47530

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47530


namespace sin_315_eq_neg_sqrt2_div_2_l47_47502

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47502


namespace original_painting_width_l47_47100

theorem original_painting_width {W : ℝ} 
  (orig_height : ℝ) (print_height : ℝ) (print_width : ℝ)
  (h1 : orig_height = 10) 
  (h2 : print_height = 25)
  (h3 : print_width = 37.5) :
  W = 15 :=
  sorry

end original_painting_width_l47_47100


namespace fib_100_mod_5_l47_47651

def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem fib_100_mod_5 : fib 100 % 5 = 0 := by
  sorry

end fib_100_mod_5_l47_47651


namespace find_number_l47_47840

theorem find_number (x : ℝ) (h : 0.20 * x = 0.20 * 650 + 190) : x = 1600 :=
sorry

end find_number_l47_47840


namespace candy_mixture_price_l47_47983

theorem candy_mixture_price
  (price_first_per_kg : ℝ) (price_second_per_kg : ℝ) (weight_ratio : ℝ) (weight_second : ℝ) 
  (h1 : price_first_per_kg = 10) 
  (h2 : price_second_per_kg = 15) 
  (h3 : weight_ratio = 3) 
  : (price_first_per_kg * weight_ratio * weight_second + price_second_per_kg * weight_second) / 
    (weight_ratio * weight_second + weight_second) = 11.25 :=
by
  sorry

end candy_mixture_price_l47_47983


namespace new_ratio_is_three_half_l47_47398

theorem new_ratio_is_three_half (F J : ℕ) (h1 : F * 4 = J * 5) (h2 : J = 120) :
  ((F + 30) : ℚ) / J = 3 / 2 :=
by
  sorry

end new_ratio_is_three_half_l47_47398


namespace eight_letter_good_words_l47_47555

-- Definition of a good word sequence (only using A, B, and C)
inductive Letter
| A | B | C

-- Define the restriction condition for a good word
def is_valid_transition (a b : Letter) : Prop :=
  match a, b with
  | Letter.A, Letter.B => False
  | Letter.B, Letter.C => False
  | Letter.C, Letter.A => False
  | _, _ => True

-- Count the number of 8-letter good words
def count_good_words : ℕ :=
  let letters := [Letter.A, Letter.B, Letter.C]
  -- Initial 3 choices for the first letter
  let first_choices := letters.length
  -- Subsequent 7 letters each have 2 valid previous choices
  let subsequent_choices := 2 ^ 7
  first_choices * subsequent_choices

theorem eight_letter_good_words : count_good_words = 384 :=
by
  sorry

end eight_letter_good_words_l47_47555


namespace find_number_l47_47283

theorem find_number (x : ℕ) (h : (537 - x) / (463 + x) = 1 / 9) : x = 437 :=
sorry

end find_number_l47_47283


namespace find_m_l47_47587

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l47_47587


namespace sin_315_eq_neg_sqrt2_over_2_l47_47545

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l47_47545


namespace sin_315_equals_minus_sqrt2_div_2_l47_47476

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l47_47476


namespace sum_of_ages_l47_47652

theorem sum_of_ages 
  (a1 a2 a3 : ℕ) 
  (h1 : a1 ≠ a2) 
  (h2 : a1 ≠ a3) 
  (h3 : a2 ≠ a3) 
  (h4 : 1 ≤ a1 ∧ a1 ≤ 9) 
  (h5 : 1 ≤ a2 ∧ a2 ≤ 9) 
  (h6 : 1 ≤ a3 ∧ a3 ≤ 9) 
  (h7 : a1 * a2 = 18) 
  (h8 : a3 * min a1 a2 = 28) : 
  a1 + a2 + a3 = 18 := 
sorry

end sum_of_ages_l47_47652


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47526

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47526


namespace pirate_schooner_problem_l47_47970

theorem pirate_schooner_problem (p : ℕ) (h1 : 10 < p) 
  (h2 : 0.54 * (p - 10) = (54 : ℝ) / 100 * (p - 10)) 
  (h3 : 0.34 * (p - 10) = (34 : ℝ) / 100 * (p - 10)) 
  (h4 : 2 / 3 * p = (2 : ℝ) / 3 * p) : 
  p = 60 := 
sorry

end pirate_schooner_problem_l47_47970


namespace edge_labeling_large_count_l47_47275

-- Define a regular dodecahedron
def dodecahedron : SimpleGraph ℕ := sorry

-- Each face must have exactly 2 edges labeled as 1, and the remaining edges as 0
def valid_edge_labeling (labels : Finset (ℕ × ℕ)) : Prop :=
  ∀ f ∈ faces, (∑ e in f.edges, labels.count (1)) = 2

-- Main statement, asserting the number of valid labelings is very large
theorem edge_labeling_large_count : 
  ∃ N : ℕ, N ≥ 10^12 ∧ 
  (∃ labels : Finset (ℕ × ℕ), valid_edge_labeling labels) :=
sorry

end edge_labeling_large_count_l47_47275


namespace tan_angle_addition_l47_47172

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l47_47172


namespace unique_solution_triple_l47_47994

theorem unique_solution_triple (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xy / z : ℚ) + (yz / x) + (zx / y) = 3 → (x = 1 ∧ y = 1 ∧ z = 1) := 
by 
  sorry

end unique_solution_triple_l47_47994


namespace cannot_form_right_triangle_l47_47853

theorem cannot_form_right_triangle (a b c : ℝ) (h₁ : a = 2) (h₂ : b = 2) (h₃ : c = 3) :
  a^2 + b^2 ≠ c^2 :=
by
  rw [h₁, h₂, h₃]
  -- Next step would be to simplify and show the inequality, but we skip the proof
  -- 2^2 + 2^2 = 4 + 4 = 8 
  -- 3^2 = 9 
  -- 8 ≠ 9
  sorry

end cannot_form_right_triangle_l47_47853


namespace solve_equation_l47_47819

noncomputable def equation_solution (x : ℝ) : Prop :=
  (3 / x = 2 / (x - 2)) ∧ x ≠ 0 ∧ x - 2 ≠ 0

theorem solve_equation : (equation_solution 6) :=
  by
    sorry

end solve_equation_l47_47819


namespace cat_can_pass_through_gap_l47_47855

theorem cat_can_pass_through_gap (R : ℝ) (h : ℝ) (π : ℝ) (hπ : π = Real.pi)
  (L₀ : ℝ) (L₁ : ℝ)
  (hL₀ : L₀ = 2 * π * R)
  (hL₁ : L₁ = L₀ + 1)
  (hL₁' : L₁ = 2 * π * (R + h)) :
  h = 1 / (2 * π) :=
by
  sorry

end cat_can_pass_through_gap_l47_47855


namespace river_flow_volume_l47_47833

noncomputable def river_depth : ℝ := 2
noncomputable def river_width : ℝ := 45
noncomputable def flow_rate_kmph : ℝ := 4
noncomputable def flow_rate_mpm := flow_rate_kmph * 1000 / 60
noncomputable def cross_sectional_area := river_depth * river_width
noncomputable def volume_per_minute := cross_sectional_area * flow_rate_mpm

theorem river_flow_volume :
  volume_per_minute = 6000.3 := by
  sorry

end river_flow_volume_l47_47833


namespace no_positive_integer_solution_l47_47023

theorem no_positive_integer_solution (p x y : ℕ) (hp : Nat.Prime p) (hp_gt3 : p > 3) 
  (h_p_div_x : p ∣ x) (hx_pos : 0 < x) (hy_pos : 0 < y) : x^2 - 1 ≠ y^p :=
sorry

end no_positive_integer_solution_l47_47023


namespace chessboard_number_determination_l47_47422

theorem chessboard_number_determination (d_n : ℤ) (a_n b_n a_1 b_1 c_0 d_0 : ℤ) :
  (∀ i j : ℤ, d_n + a_n = b_n + a_1 + b_1 - (c_0 + d_0) → 
   a_n + b_n = c_0 + d_0 + d_n) →
  ∃ x : ℤ, x = a_1 + b_1 - d_n ∧ 
  x = d_n + (a_1 - c_0) + (b_1 - d_0) :=
by
  sorry

end chessboard_number_determination_l47_47422


namespace stickers_total_l47_47387

theorem stickers_total (ryan_has : Ryan_stickers = 30)
  (steven_has : Steven_stickers = 3 * Ryan_stickers)
  (terry_has : Terry_stickers = Steven_stickers + 20) :
  Ryan_stickers + Steven_stickers + Terry_stickers = 230 :=
sorry

end stickers_total_l47_47387


namespace simplify_expression_l47_47617

theorem simplify_expression (m n : ℝ) (h : m^2 + 3 * m * n = 5) : 
  5 * m^2 - 3 * m * n - (-9 * m * n + 3 * m^2) = 10 :=
by 
  sorry

end simplify_expression_l47_47617


namespace necessary_but_not_sufficient_l47_47010

def M := {x : ℝ | 0 < x ∧ x ≤ 3}
def N := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient (a : ℝ) (haM : a ∈ M) : (a ∈ N → a ∈ M) ∧ ¬(a ∈ M → a ∈ N) :=
by {
  sorry
}

end necessary_but_not_sufficient_l47_47010


namespace well_depth_l47_47700

theorem well_depth :
  (∃ t₁ t₂ : ℝ, t₁ + t₂ = 9.5 ∧ 20 * t₁ ^ 2 = d ∧ t₂ = d / 1000 ∧ d = 1332.25) :=
by
  sorry

end well_depth_l47_47700


namespace greatest_integer_x_l47_47414

theorem greatest_integer_x (x : ℤ) : 
  (∃ k : ℤ, (x - 4) = k ∧ x^2 - 3 * x + 4 = k * (x - 4) + 8) →
  x ≤ 12 :=
by
  sorry

end greatest_integer_x_l47_47414


namespace tan_add_pi_over_3_l47_47153

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l47_47153


namespace greatest_number_of_sets_l47_47080

-- Definitions based on conditions
def whitney_tshirts := 5
def whitney_buttons := 24
def whitney_stickers := 12
def buttons_per_set := 2
def stickers_per_set := 1

-- The statement to prove the greatest number of identical sets Whitney can make
theorem greatest_number_of_sets : 
  ∃ max_sets : ℕ, 
  max_sets = whitney_tshirts ∧ 
  max_sets ≤ (whitney_buttons / buttons_per_set) ∧
  max_sets ≤ (whitney_stickers / stickers_per_set) :=
sorry

end greatest_number_of_sets_l47_47080


namespace tan_add_pi_div_three_l47_47177

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l47_47177


namespace complement_union_eq_l47_47753

variable (U : Set Int := {-2, -1, 0, 1, 2, 3}) 
variable (A : Set Int := {-1, 0, 1}) 
variable (B : Set Int := {1, 2}) 

theorem complement_union_eq :
  U \ (A ∪ B) = {-2, 3} := by 
  sorry

end complement_union_eq_l47_47753


namespace sum_of_roots_quadratic_eq_m_plus_n_l47_47288

theorem sum_of_roots_quadratic_eq_m_plus_n :
  let a := 5
  let b := -11
  let c := 2
  let sum_roots := (11 + 9) / 10 + (11 - 9) / 10
  let m := 121
  let n := 5
  sum_roots = 2.2 ∧ (Int.sqrt m = 11 ∧ n = 5) → m + n = 126 :=
by
  sorry

end sum_of_roots_quadratic_eq_m_plus_n_l47_47288


namespace problem_simplify_l47_47649

variable (a : ℝ)

theorem problem_simplify (h1 : a ≠ 3) (h2 : a ≠ -3) :
  (1 / (a - 3) - 6 / (a^2 - 9) = 1 / (a + 3)) :=
sorry

end problem_simplify_l47_47649


namespace find_speed_first_car_l47_47946

noncomputable def speed_first_car (v : ℝ) : Prop :=
  let t := (14 : ℝ) / 3
  let d_total := 490
  let d_second_car := 60 * t
  let d_first_car := v * t
  d_second_car + d_first_car = d_total

theorem find_speed_first_car : ∃ v : ℝ, speed_first_car v ∧ v = 45 :=
by
  sorry

end find_speed_first_car_l47_47946


namespace domain_of_f_equals_l47_47122

noncomputable def domain_of_function := {x : ℝ | x > -1 ∧ -(x+4) * (x-1) > 0}

theorem domain_of_f_equals : domain_of_function = { x : ℝ | -1 < x ∧ x < 1 } :=
by
  sorry

end domain_of_f_equals_l47_47122


namespace Donggil_cleaning_time_l47_47249

-- Define the total area of the school as A.
variable (A : ℝ)

-- Define the cleaning rates of Daehyeon (D) and Donggil (G).
variable (D G : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (D + G) * 8 = (7 / 12) * A
def condition2 : Prop := D * 10 = (5 / 12) * A

-- The goal is to prove that Donggil can clean the entire area alone in 32 days.
theorem Donggil_cleaning_time : condition1 A D G ∧ condition2 A D → 32 * G = A :=
by
  sorry

end Donggil_cleaning_time_l47_47249


namespace perfect_square_expression_l47_47563

theorem perfect_square_expression (n : ℕ) : ∃ t : ℕ, n^2 - 4 * n + 11 = t^2 ↔ n = 5 :=
by
  sorry

end perfect_square_expression_l47_47563


namespace find_x_l47_47189

theorem find_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 5 = 5 * y + 2) :
  x = (685 + 25 * Real.sqrt 745) / 6 :=
by
  sorry

end find_x_l47_47189


namespace g_difference_l47_47637

def g (n : ℕ) : ℚ :=
  1/4 * n * (n + 1) * (n + 2) * (n + 3)

theorem g_difference (r : ℕ) : g r - g (r - 1) = r * (r + 1) * (r + 2) :=
  sorry

end g_difference_l47_47637


namespace tan_add_pi_over_3_l47_47161

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l47_47161


namespace employee_payment_correct_l47_47276

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the percentage markup for retail price
def markup_percentage : ℝ := 0.20

-- Define the retail_price based on wholesale cost and markup percentage
def retail_price : ℝ := wholesale_cost + (markup_percentage * wholesale_cost)

-- Define the employee discount percentage
def discount_percentage : ℝ := 0.20

-- Define the discount amount based on retail price and discount percentage
def discount_amount : ℝ := retail_price * discount_percentage

-- Define the final price the employee pays after applying the discount
def employee_price : ℝ := retail_price - discount_amount

-- State the theorem to prove
theorem employee_payment_correct :
  employee_price = 192 :=
  by
    sorry

end employee_payment_correct_l47_47276


namespace parallel_vectors_eq_l47_47880

theorem parallel_vectors_eq (t : ℝ) : ∀ (m n : ℝ × ℝ), m = (2, 8) → n = (-4, t) → (∃ k : ℝ, n = k • m) → t = -16 :=
by 
  intros m n hm hn h_parallel
  -- proof goes here
  sorry

end parallel_vectors_eq_l47_47880


namespace sin_315_equals_minus_sqrt2_div_2_l47_47471

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l47_47471


namespace find_list_price_l47_47434

noncomputable def list_price (x : ℝ) (alice_price_diff bob_price_diff : ℝ) (alice_comm_fraction bob_comm_fraction : ℝ) : Prop :=
  alice_comm_fraction * (x - alice_price_diff) = bob_comm_fraction * (x - bob_price_diff)

theorem find_list_price : list_price 40 15 25 0.15 0.25 :=
by
  sorry

end find_list_price_l47_47434


namespace length_of_scale_parts_l47_47954

theorem length_of_scale_parts (total_length_ft : ℕ) (remaining_inches : ℕ) (parts : ℕ) : 
  total_length_ft = 6 ∧ remaining_inches = 8 ∧ parts = 2 →
  ∃ ft inches, ft = 3 ∧ inches = 4 :=
by
  sorry

end length_of_scale_parts_l47_47954


namespace carnival_activity_order_l47_47402

theorem carnival_activity_order :
  let dodgeball := 3 / 8
  let magic_show := 9 / 24
  let petting_zoo := 1 / 3
  let face_painting := 5 / 12
  let ordered_activities := ["Face Painting", "Dodgeball", "Magic Show", "Petting Zoo"]
  (face_painting > dodgeball) ∧ (dodgeball = magic_show) ∧ (magic_show > petting_zoo) →
  ordered_activities = ["Face Painting", "Dodgeball", "Magic Show", "Petting Zoo"] :=
by {
  sorry
}

end carnival_activity_order_l47_47402


namespace find_value_of_x_l47_47211

theorem find_value_of_x (b : ℕ) (x : ℝ) (h_b_pos : b > 0) (h_x_pos : x > 0) 
  (h_r1 : r = 4 ^ (2 * b)) (h_r2 : r = 2 ^ b * x ^ b) : x = 8 :=
by
  -- Proof omitted for brevity
  sorry

end find_value_of_x_l47_47211


namespace min_k_value_l47_47931

variable (p q r s k : ℕ)

/-- Prove the smallest value of k for which p, q, r, and s are positive integers and 
    satisfy the given equations is 77
-/
theorem min_k_value (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
  (eq1 : p + 2 * q + 3 * r + 4 * s = k)
  (eq2 : 4 * p = 3 * q)
  (eq3 : 4 * p = 2 * r)
  (eq4 : 4 * p = s) : k = 77 :=
sorry

end min_k_value_l47_47931


namespace range_of_a_l47_47771

-- Define the function f and its derivative f'
def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1
def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + 3 * (a + 2)

-- We are given that for f to have both maximum and minimum values, f' must have two distinct roots
-- Thus we translate the mathematical condition to the discriminant of f' being greater than 0
def discriminant_greater_than_zero (a : ℝ) : Prop :=
  (6 * a)^2 - 4 * 3 * 3 * (a + 2) > 0

-- Finally, we want to prove that this simplifies to a condition on a
theorem range_of_a (a : ℝ) : discriminant_greater_than_zero a ↔ (a > 2 ∨ a < -1) :=
by
  -- Write the proof here
  sorry

end range_of_a_l47_47771


namespace abs_eq_neg_of_nonpos_l47_47981

theorem abs_eq_neg_of_nonpos (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
by
  have ha : |a| ≥ 0 := abs_nonneg a
  rw [h] at ha
  exact neg_nonneg.mp ha

end abs_eq_neg_of_nonpos_l47_47981


namespace pos_divisors_180_l47_47900

theorem pos_divisors_180 : 
  (∃ a b c : ℕ, 180 = (2^a) * (3^b) * (5^c) ∧ a = 2 ∧ b = 2 ∧ c = 1) →
  (∃ n : ℕ, n = 18 ∧ n = (a + 1) * (b + 1) * (c + 1)) := by
  sorry

end pos_divisors_180_l47_47900


namespace opposite_of_83_is_84_l47_47688

-- Define the problem context
def circle_with_points (n : ℕ) := { p : ℕ // p < n }

-- Define the even distribution condition
def even_distribution (n k : ℕ) (points : circle_with_points n → ℕ) :=
  ∀ k < n, let diameter := set_of (λ p, points p < k) in
           cardinal.mk diameter % 2 = 0

-- Define the diametrically opposite point function
def diametrically_opposite (n k : ℕ) (points : circle_with_points n → ℕ) : ℕ :=
  (points ⟨k + (n / 2), sorry⟩) -- We consider n is always even, else modulus might be required

-- Problem statement to prove
theorem opposite_of_83_is_84 :
  ∀ (points : circle_with_points 100 → ℕ)
    (hne : even_distribution 100 83 points),
    diametrically_opposite 100 83 points = 84 := 
sorry

end opposite_of_83_is_84_l47_47688


namespace no_solution_for_equation_l47_47923

theorem no_solution_for_equation (x : ℝ) (hx : x ≠ -1) :
  (5 * x + 2) / (x^2 + x) ≠ 3 / (x + 1) := 
sorry

end no_solution_for_equation_l47_47923


namespace overall_average_score_l47_47622

def students_monday := 24
def students_tuesday := 4
def total_students := 28
def mean_score_monday := 82
def mean_score_tuesday := 90

theorem overall_average_score :
  (students_monday * mean_score_monday + students_tuesday * mean_score_tuesday) / total_students = 83 := by
sorry

end overall_average_score_l47_47622


namespace sin_315_degree_l47_47481

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l47_47481


namespace exam_max_marks_l47_47290

theorem exam_max_marks (M : ℝ) (h1: 0.30 * M = 66) : M = 220 :=
by
  sorry

end exam_max_marks_l47_47290


namespace division_result_l47_47131

open Polynomial

noncomputable def dividend := (X ^ 6 - 5 * X ^ 4 + 3 * X ^ 3 - 7 * X ^ 2 + 2 * X - 8 : Polynomial ℤ)
noncomputable def divisor := (X - 3 : Polynomial ℤ)
noncomputable def expected_quotient := (X ^ 5 + 3 * X ^ 4 + 4 * X ^ 3 + 15 * X ^ 2 + 38 * X + 116 : Polynomial ℤ)
noncomputable def expected_remainder := (340 : ℤ)

theorem division_result : (dividend /ₘ divisor) = expected_quotient ∧ (dividend %ₘ divisor) = C expected_remainder := by
  sorry

end division_result_l47_47131


namespace angle_is_30_degrees_l47_47770

theorem angle_is_30_degrees (A : ℝ) (h_acute : A > 0 ∧ A < π / 2) (h_sin : Real.sin A = 1/2) : A = π / 6 := 
by 
  sorry

end angle_is_30_degrees_l47_47770


namespace union_complement_eq_set_l47_47214

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {3, 5, 6}
def comp_U_N : Set ℕ := U \ N  -- complement of N with respect to U

theorem union_complement_eq_set :
  M ∪ comp_U_N = {1, 2, 3, 4} :=
by
  simp [U, M, N, comp_U_N]
  sorry

end union_complement_eq_set_l47_47214


namespace second_dog_miles_per_day_l47_47110

-- Definitions describing conditions
section DogWalk
variable (total_miles_week : ℕ)
variable (first_dog_miles_day : ℕ)
variable (days_in_week : ℕ)

-- Assert conditions given in the problem
def condition1 := total_miles_week = 70
def condition2 := first_dog_miles_day = 2
def condition3 := days_in_week = 7

-- The theorem to prove
theorem second_dog_miles_per_day
  (h1 : condition1 total_miles_week)
  (h2 : condition2 first_dog_miles_day)
  (h3 : condition3 days_in_week) :
  (total_miles_week - days_in_week * first_dog_miles_day) / days_in_week = 8 :=
sorry
end DogWalk

end second_dog_miles_per_day_l47_47110


namespace option_B_is_quadratic_l47_47078

-- Definitions of the equations provided in conditions
def eqA (x y : ℝ) := x + 3 * y = 4
def eqB (y : ℝ) := 5 * y = 5 * y^2
def eqC (x : ℝ) := 4 * x - 4 = 0
def eqD (x a : ℝ) := a * x^2 - x = 1

-- Quadratic equation definition
def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ (A B C : ℝ) (x : ℝ), A ≠ 0 ∧ eq = (A * x^2 + B * x + C = 0)

-- Theorem: Prove that option B (eqB) is definitely a quadratic equation
theorem option_B_is_quadratic : is_quadratic eqB :=
  sorry

end option_B_is_quadratic_l47_47078


namespace right_triangle_area_integer_l47_47222

theorem right_triangle_area_integer (a b c : ℤ) (h : a * a + b * b = c * c) : ∃ (n : ℤ), (1 / 2 : ℚ) * a * b = ↑n := 
sorry

end right_triangle_area_integer_l47_47222


namespace CarlosBooksInJune_l47_47710

def CarlosBooksInJuly := 28
def CarlosBooksInAugust := 30
def CarlosTotalBooksGoal := 100

theorem CarlosBooksInJune :
  ∃ x : ℕ, x = CarlosTotalBooksGoal - (CarlosBooksInJuly + CarlosBooksInAugust) :=
begin
  use 42,
  dsimp [CarlosTotalBooksGoal, CarlosBooksInJuly, CarlosBooksInAugust],
  norm_num,
  sorry
end

end CarlosBooksInJune_l47_47710


namespace sheep_count_l47_47909

-- Define the conditions
def TotalAnimals : ℕ := 200
def NumberCows : ℕ := 40
def NumberGoats : ℕ := 104

-- Define the question and its corresponding answer
def NumberSheep : ℕ := TotalAnimals - (NumberCows + NumberGoats)

-- State the theorem
theorem sheep_count : NumberSheep = 56 := by
  -- Skipping the proof
  sorry

end sheep_count_l47_47909


namespace total_spending_l47_47811

-- Conditions
def pop_spending : ℕ := 15
def crackle_spending : ℕ := 3 * pop_spending
def snap_spending : ℕ := 2 * crackle_spending

-- Theorem stating the total spending
theorem total_spending : snap_spending + crackle_spending + pop_spending = 150 :=
by
  sorry

end total_spending_l47_47811


namespace sin_315_eq_neg_sqrt2_div_2_l47_47536

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47536


namespace milburg_children_count_l47_47049

theorem milburg_children_count : 
  ∀ (total_population grown_ups : ℕ), 
  total_population = 8243 → grown_ups = 5256 → 
  (total_population - grown_ups) = 2987 :=
by
  intros total_population grown_ups h1 h2
  sorry

end milburg_children_count_l47_47049


namespace coffee_cups_per_week_l47_47267

theorem coffee_cups_per_week 
  (cups_per_hour_weekday : ℕ)
  (total_weekend_cups : ℕ)
  (hours_per_day : ℕ)
  (days_in_week : ℕ)
  (weekdays : ℕ)
  : cups_per_hour_weekday * hours_per_day * weekdays + total_weekend_cups = 370 :=
by
  -- Assume values based on given conditions
  have h1 : cups_per_hour_weekday = 10 := sorry
  have h2 : total_weekend_cups = 120 := sorry
  have h3 : hours_per_day = 5 := sorry
  have h4 : weekdays = 5 := sorry
  have h5 : days_in_week = 7 := sorry
  
  -- Calculate total weekday production
  have weekday_production : ℕ := cups_per_hour_weekday * hours_per_day * weekdays
  
  -- Substitute and calculate total weekly production
  have total_production : ℕ := weekday_production + total_weekend_cups
  
  -- The total production should be equal to 370
  show total_production = 370 from by
    rw [weekday_production, h1, h3, h4]
    exact sorry

  sorry

end coffee_cups_per_week_l47_47267


namespace tan_add_pi_div_three_l47_47175

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l47_47175


namespace sin_315_eq_neg_sqrt2_div_2_l47_47519

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l47_47519


namespace value_of_f1_l47_47656

noncomputable def f (x : ℝ) (m : ℝ) := 2 * x^2 - m * x + 3

theorem value_of_f1 (m : ℝ) (h_increasing : ∀ x : ℝ, x ≥ -2 → 2 * x^2 - m * x + 3 ≤ 2 * (x + 1)^2 - m * (x + 1) + 3)
  (h_decreasing : ∀ x : ℝ, x ≤ -2 → 2 * (x - 1)^2 - m * (x - 1) + 3 ≤ 2 * x^2 - m * x + 3) : 
  f 1 (-8) = 13 := 
sorry

end value_of_f1_l47_47656


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47520

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47520


namespace derivative_f_cos2x_l47_47734

variable {f : ℝ → ℝ} {x : ℝ}

theorem derivative_f_cos2x :
  f (Real.cos (2 * x)) = 1 - 2 * (Real.sin x) ^ 2 →
  deriv f x = -2 * Real.sin (2 * x) :=
by sorry

end derivative_f_cos2x_l47_47734


namespace intersection_A_B_l47_47213

def A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem intersection_A_B : (A ∩ B) = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_A_B_l47_47213


namespace find_m_l47_47579

open Nat

theorem find_m (m : ℕ) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 60 := 
by
  sorry  -- Proof goes here

end find_m_l47_47579


namespace angle_in_third_quadrant_l47_47331

theorem angle_in_third_quadrant (k : ℤ) (α : ℝ) 
  (h : 180 + k * 360 < α ∧ α < 270 + k * 360) : 
  180 - α > -90 - k * 360 ∧ 180 - α < -k * 360 := 
by sorry

end angle_in_third_quadrant_l47_47331


namespace numbers_with_digit_one_are_more_numerous_l47_47852

theorem numbers_with_digit_one_are_more_numerous :
  let n := 9999998
  let total_numbers := 10^7
  let numbers_without_one := 9^7 
  total_numbers - numbers_without_one > numbers_without_one :=
by
  let n := 9999998
  let total_numbers := 10^7
  let numbers_without_one := 9^7 
  show total_numbers - numbers_without_one > numbers_without_one
  sorry

end numbers_with_digit_one_are_more_numerous_l47_47852


namespace find_m_l47_47595

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l47_47595


namespace max_profit_jars_max_tax_value_l47_47343

-- Part a: Prove the optimal number of jars for maximum profit
theorem max_profit_jars (Q : ℝ) 
  (h : ∀ Q, Q >= 0 → (310 - 3 * Q) * Q - 10 * Q ≤ (310 - 3 * 50) * 50 - 10 * 50):
  Q = 50 :=
sorry

-- Part b: Prove the optimal tax for maximum tax revenue
theorem max_tax_value (t : ℝ) 
  (h : ∀ t, t >= 0 → ((300 * t - t^2) / 6) ≤ ((300 * 150 - 150^2) / 6)):
  t = 150 :=
sorry

end max_profit_jars_max_tax_value_l47_47343


namespace find_prime_power_solutions_l47_47295

theorem find_prime_power_solutions (p n m : ℕ) (hp : Nat.Prime p) (hn : n > 0) (hm : m > 0) 
  (h : p^n + 144 = m^2) :
  (p = 2 ∧ n = 9 ∧ m = 36) ∨ (p = 3 ∧ n = 4 ∧ m = 27) :=
by sorry

end find_prime_power_solutions_l47_47295


namespace initial_card_distribution_l47_47702

variables {A B C D : ℕ}

theorem initial_card_distribution 
  (total_cards : A + B + C + D = 32)
  (alfred_final : ∀ c, c = A → ((c / 2) + (c / 2)) + B + C + D = 8)
  (bruno_final : ∀ c, c = B → ((c / 2) + (c / 2)) + A + C + D = 8)
  (christof_final : ∀ c, c = C → ((c / 2) + (c / 2)) + A + B + D = 8)
  : A = 7 ∧ B = 7 ∧ C = 10 ∧ D = 8 :=
by sorry

end initial_card_distribution_l47_47702


namespace original_equation_proof_l47_47224

theorem original_equation_proof :
  ∃ (A O H M J : ℕ),
  A ≠ O ∧ A ≠ H ∧ A ≠ M ∧ A ≠ J ∧
  O ≠ H ∧ O ≠ M ∧ O ≠ J ∧
  H ≠ M ∧ H ≠ J ∧
  M ≠ J ∧
  A + 8 * (10 * O + H) = 10 * M + J ∧
  (O = 1) ∧ (H = 2) ∧ (M = 9) ∧ (J = 6) ∧ (A = 0) :=
by
  sorry

end original_equation_proof_l47_47224


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47507

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47507


namespace carlos_books_in_june_l47_47711

theorem carlos_books_in_june
  (books_july : ℕ)
  (books_august : ℕ)
  (total_books_needed : ℕ)
  (books_june : ℕ) : 
  books_july = 28 →
  books_august = 30 →
  total_books_needed = 100 →
  books_june = total_books_needed - (books_july + books_august) →
  books_june = 42 :=
by intros books_july books_august total_books_needed books_june h1 h2 h3 h4
   sorry

end carlos_books_in_june_l47_47711


namespace complement_union_l47_47761

variable (U : Set ℤ)
variable (A : Set ℤ)
variable (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} :=
sorry

end complement_union_l47_47761


namespace sum_of_squares_l47_47403

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 22) (h2 : x * y = 40) : x^2 + y^2 = 404 :=
  sorry

end sum_of_squares_l47_47403


namespace table_height_is_five_l47_47978

def height_of_table (l h w : ℕ) : Prop :=
  l + h + w = 45 ∧ 2 * w + h = 40

theorem table_height_is_five (l w : ℕ) : height_of_table l 5 w :=
by
  sorry

end table_height_is_five_l47_47978


namespace complement_union_eq_l47_47755

variable (U : Set Int := {-2, -1, 0, 1, 2, 3}) 
variable (A : Set Int := {-1, 0, 1}) 
variable (B : Set Int := {1, 2}) 

theorem complement_union_eq :
  U \ (A ∪ B) = {-2, 3} := by 
  sorry

end complement_union_eq_l47_47755


namespace slope_probability_l47_47883

def line_equation (a x y : ℝ) : Prop := a * x + 2 * y - 3 = 0

def in_interval (a : ℝ) : Prop := -5 ≤ a ∧ a ≤ 4

def slope_not_less_than_1 (a : ℝ) : Prop := - a / 2 ≥ 1

noncomputable def probability_slope_not_less_than_1 : ℝ :=
  (2 - (-5)) / (4 - (-5))

theorem slope_probability :
  ∀ (a : ℝ), in_interval a → slope_not_less_than_1 a → probability_slope_not_less_than_1 = 1 / 3 :=
by
  intros a h_in h_slope
  sorry

end slope_probability_l47_47883


namespace number_of_men_for_2km_road_l47_47822

noncomputable def men_for_1km_road : ℕ := 30
noncomputable def days_for_1km_road : ℕ := 12
noncomputable def hours_per_day_for_1km_road : ℕ := 8
noncomputable def length_of_1st_road : ℕ := 1
noncomputable def length_of_2nd_road : ℕ := 2
noncomputable def working_hours_per_day_2nd_road : ℕ := 14
noncomputable def days_for_2km_road : ℝ := 20.571428571428573

theorem number_of_men_for_2km_road (total_man_hours_1km : ℕ := men_for_1km_road * days_for_1km_road * hours_per_day_for_1km_road):
  (men_for_1km_road * length_of_2nd_road * days_for_1km_road * hours_per_day_for_1km_road = 5760) →
  ∃ (men_for_2nd_road : ℕ), men_for_1km_road * 2 * days_for_1km_road * hours_per_day_for_1km_road = 5760 ∧  men_for_2nd_road * days_for_2km_road * working_hours_per_day_2nd_road = 5760 ∧ men_for_2nd_road = 20 :=
by {
  sorry
}

end number_of_men_for_2km_road_l47_47822


namespace mixture_problem_l47_47899

theorem mixture_problem :
  ∀ (x P : ℝ), 
    let initial_solution := 70
    let initial_percentage := 0.20
    let final_percentage := 0.40
    let final_amount := 70
    (x = 70) →
    (initial_percentage * initial_solution + P * x = final_percentage * (initial_solution + x)) →
    (P = 0.60) :=
by
  intros x P initial_solution initial_percentage final_percentage final_amount hx h_eq
  sorry

end mixture_problem_l47_47899


namespace smiths_bakery_pies_l47_47230

theorem smiths_bakery_pies (m : ℕ) (h : m = 16) : 
  let s := 4 * m + 6 in 
  s = 70 := 
by
  unfold s
  sorry

end smiths_bakery_pies_l47_47230


namespace problem_I_II_l47_47740

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

def seq_a (a : ℕ → ℝ) (a1 : ℝ) : Prop :=
  a 0 = a1 ∧ (∀ n, a (n + 1) = f (a n))

theorem problem_I_II (a : ℕ → ℝ) (a1 : ℝ) (h_a1 : 0 < a1 ∧ a1 < 1) (h_seq : seq_a a a1) :
  (∀ n, 0 < a (n + 1) ∧ a (n + 1) < a n ∧ a n < 1) ∧
  (∀ n, a (n + 1) < (1 / 6) * (a n) ^ 3) :=
  sorry

end problem_I_II_l47_47740


namespace smallest_c_for_exponential_inequality_l47_47416

theorem smallest_c_for_exponential_inequality : ∃ c : ℤ, 27^c > 3^24 ∧ ∀ n : ℤ, 27^n > 3^24 → c ≤ n := sorry

end smallest_c_for_exponential_inequality_l47_47416


namespace volleyball_lineup_ways_l47_47105

def num_ways_lineup (team_size : ℕ) (positions : ℕ) : ℕ :=
  if positions ≤ team_size then
    Nat.descFactorial team_size positions
  else
    0

theorem volleyball_lineup_ways :
  num_ways_lineup 10 5 = 30240 :=
by
  rfl

end volleyball_lineup_ways_l47_47105


namespace find_a3_l47_47606

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_a3
  (a : ℕ → α) (q : α)
  (h_geom : geometric_sequence a q)
  (h_a1 : a 1 = 2)
  (h_cond : a 3 * a 5 = 4 * (a 6) ^ 2) :
  a 3 = 1 :=
by
  sorry

end find_a3_l47_47606


namespace jon_awake_hours_per_day_l47_47911

def regular_bottle_size : ℕ := 16
def larger_bottle_size : ℕ := 20
def weekly_fluid_intake : ℕ := 728
def larger_bottle_daily_intake : ℕ := 40
def larger_bottle_weekly_intake : ℕ := 280
def regular_bottle_weekly_intake : ℕ := 448
def regular_bottles_per_week : ℕ := 28
def regular_bottles_per_day : ℕ := 4
def hours_per_bottle : ℕ := 4

theorem jon_awake_hours_per_day
  (h1 : jon_drinks_regular_bottle_every_4_hours)
  (h2 : jon_drinks_two_larger_bottles_daily)
  (h3 : jon_drinks_728_ounces_per_week) :
  jon_is_awake_hours_per_day = 16 :=
by
  sorry

def jon_drinks_regular_bottle_every_4_hours : Prop :=
  ∀ hours : ℕ, hours * regular_bottle_size / hours_per_bottle = 1

def jon_drinks_two_larger_bottles_daily : Prop :=
  larger_bottle_size = (regular_bottle_size * 5) / 4 ∧ 
  larger_bottle_daily_intake = 2 * larger_bottle_size

def jon_drinks_728_ounces_per_week : Prop :=
  weekly_fluid_intake = 728

def jon_is_awake_hours_per_day : ℕ :=
  regular_bottles_per_day * hours_per_bottle

end jon_awake_hours_per_day_l47_47911


namespace max_discount_l47_47850

theorem max_discount (C : ℝ) (x : ℝ) (h1 : 1.8 * C = 360) (h2 : ∀ y, y ≥ 1.3 * C → 360 - x ≥ y) : x ≤ 100 :=
by
  have hC : C = 360 / 1.8 := by sorry
  have hMinPrice : 1.3 * C = 1.3 * (360 / 1.8) := by sorry
  have hDiscount : 360 - x ≥ 1.3 * (360 / 1.8) := by sorry
  sorry

end max_discount_l47_47850


namespace milburg_children_count_l47_47050

theorem milburg_children_count : 
  ∀ (total_population grown_ups : ℕ), 
  total_population = 8243 → grown_ups = 5256 → 
  (total_population - grown_ups) = 2987 :=
by
  intros total_population grown_ups h1 h2
  sorry

end milburg_children_count_l47_47050


namespace jane_stopped_babysitting_l47_47629

noncomputable def stopped_babysitting_years_ago := 12

-- Definitions for the problem conditions
def jane_age_started_babysitting := 20
def jane_current_age := 32
def oldest_child_current_age := 22

-- Final statement to prove the equivalence
theorem jane_stopped_babysitting : 
    ∃ (x : ℕ), 
    (jane_current_age - x = stopped_babysitting_years_ago) ∧
    (oldest_child_current_age - x ≤ 1/2 * (jane_current_age - x)) := 
sorry

end jane_stopped_babysitting_l47_47629


namespace find_m_l47_47602

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l47_47602


namespace new_person_weight_l47_47234

theorem new_person_weight (n : ℕ) (k : ℝ) (w_old w_new : ℝ) 
  (h_n : n = 6) 
  (h_k : k = 4.5) 
  (h_w_old : w_old = 75) 
  (h_avg_increase : w_new - w_old = n * k) : 
  w_new = 102 := 
sorry

end new_person_weight_l47_47234


namespace divides_trans_l47_47368

theorem divides_trans (m n : ℤ) (h : n ∣ m * (n + 1)) : n ∣ m :=
by
  sorry

end divides_trans_l47_47368


namespace ways_to_place_7_balls_in_3_boxes_l47_47020

theorem ways_to_place_7_balls_in_3_boxes : ∃ n : ℕ, n = 8 ∧ (∀ x y z : ℕ, x + y + z = 7 → x ≥ y → y ≥ z → z ≥ 0) := 
by
  sorry

end ways_to_place_7_balls_in_3_boxes_l47_47020


namespace arc_length_correct_l47_47232

noncomputable def radius : ℝ :=
  5

noncomputable def area_of_sector : ℝ :=
  8.75

noncomputable def arc_length (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * 2 * Real.pi * r

theorem arc_length_correct :
  ∃ θ, arc_length θ radius = 3.5 ∧ (θ / 360) * Real.pi * radius^2 = area_of_sector :=
by
  sorry

end arc_length_correct_l47_47232


namespace evaluate_fraction_l47_47559

theorem evaluate_fraction (a b : ℤ) (h1 : a = 5) (h2 : b = -2) : (5 : ℝ) / (a + b) = 5 / 3 :=
by
  sorry

end evaluate_fraction_l47_47559


namespace rationalize_denominator_equals_ABC_product_l47_47026

theorem rationalize_denominator_equals_ABC_product :
  let A := -9
  let B := -4
  let C := 5
  ∀ (x y : ℚ), (x + y * real.sqrt 5) = (2 + real.sqrt 5) * (2 + real.sqrt 5) / ((2 - real.sqrt 5) * (2 + real.sqrt 5)) →
    A * B * C = 180 := sorry

end rationalize_denominator_equals_ABC_product_l47_47026


namespace sofia_running_time_l47_47035

theorem sofia_running_time :
  ∃ t : ℤ, t = 8 * 60 + 20 ∧ 
  (∀ (laps : ℕ) (d1 d2 v1 v2 : ℤ),
    laps = 5 →
    d1 = 200 →
    v1 = 4 →
    d2 = 300 →
    v2 = 6 →
    t = laps * ((d1 / v1 + d2 / v2))) :=
by
  sorry

end sofia_running_time_l47_47035


namespace three_digit_palindrome_probability_divisible_by_11_l47_47277

theorem three_digit_palindrome_probability_divisible_by_11 :
  (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 101 * a + 10 * b % 11 = 0) →
  (∃ p : ℚ, p = 1 / 5) :=
by
  sorry

end three_digit_palindrome_probability_divisible_by_11_l47_47277


namespace part_I_part_II_l47_47571

-- Part (I)
theorem part_I (a₁ : ℝ) (d : ℝ) (S : ℕ → ℝ) (k : ℕ) :
  a₁ = 3 / 2 →
  d = 1 →
  (∀ n, S n = (n / 2 : ℝ) * (n + 2)) →
  S (k ^ 2) = S k ^ 2 →
  k = 4 :=
by
  intros ha₁ hd hSn hSeq
  sorry

-- Part (II)
theorem part_II (a : ℝ) (d : ℝ) (S : ℕ → ℝ) :
  (∀ k : ℕ, S (k ^ 2) = (S k) ^ 2) →
  ( (∀ n, a = 0 ∧ d = 0 ∧ a + d * (n - 1) = 0) ∨
    (∀ n, a = 1 ∧ d = 0 ∧ a + d * (n - 1) = 1) ∨
    (∀ n, a = 1 ∧ d = 2 ∧ a + d * (n - 1) = 2 * n - 1) ) :=
by
  intros hSeq
  sorry

end part_I_part_II_l47_47571


namespace chord_length_of_curve_by_line_l47_47967

theorem chord_length_of_curve_by_line :
  let x (t : ℝ) := 2 + 2 * t
  let y (t : ℝ) := -t
  let curve_eq (θ : ℝ) := 4 * Real.cos θ
  ∃ a b : ℝ, (x a = 2 + 2 * a ∧ y a = -a) ∧ (x b = 2 + 2 * b ∧ y b = -b) ∧
  ((x a - x b)^2 + (y a - y b)^2 = 4^2) :=
by
  sorry

end chord_length_of_curve_by_line_l47_47967


namespace point_outside_circle_l47_47334

theorem point_outside_circle (D E F x0 y0 : ℝ) (h : (x0 + D / 2)^2 + (y0 + E / 2)^2 > (D^2 + E^2 - 4 * F) / 4) :
  x0^2 + y0^2 + D * x0 + E * y0 + F > 0 :=
sorry

end point_outside_circle_l47_47334


namespace find_cos_alpha_l47_47739

theorem find_cos_alpha (α : ℝ) (h0 : 0 ≤ α ∧ α ≤ π / 2) (h1 : Real.sin (α - π / 6) = 3 / 5) : 
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 :=
sorry

end find_cos_alpha_l47_47739


namespace snack_cost_is_five_l47_47632

-- Define the cost of one ticket
def ticket_cost : ℕ := 18

-- Define the total number of people
def total_people : ℕ := 4

-- Define the total cost for tickets and snacks
def total_cost : ℕ := 92

-- Define the unknown cost of one set of snacks
def snack_cost := 92 - 4 * 18

-- Statement asserting that the cost of one set of snacks is $5
theorem snack_cost_is_five : snack_cost = 5 := by
  sorry

end snack_cost_is_five_l47_47632


namespace total_cans_collected_l47_47959

-- Definitions based on conditions
def bags_on_saturday : ℕ := 6
def bags_on_sunday : ℕ := 3
def cans_per_bag : ℕ := 8

-- The theorem statement
theorem total_cans_collected : bags_on_saturday + bags_on_sunday * cans_per_bag = 72 :=
by
  sorry

end total_cans_collected_l47_47959


namespace leftmost_rectangle_is_B_l47_47717

def isLeftmostRectangle (wA wB wC wD wE : ℕ) : Prop := 
  wB < wD ∧ wB < wE

theorem leftmost_rectangle_is_B :
  let wA := 5
  let wB := 2
  let wC := 4
  let wD := 9
  let wE := 10
  let xA := 2
  let xB := 1
  let xC := 7
  let xD := 6
  let xE := 4
  let yA := 8
  let yB := 6
  let yC := 3
  let yD := 5
  let yE := 7
  let zA := 10
  let zB := 9
  let zC := 0
  let zD := 11
  let zE := 2
  isLeftmostRectangle wA wB wC wD wE :=
by
  simp only
  sorry

end leftmost_rectangle_is_B_l47_47717


namespace hyperbola_imaginary_axis_twice_real_axis_l47_47041

theorem hyperbola_imaginary_axis_twice_real_axis (m : ℝ) : 
  (exists (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0), mx^2 + b^2 * y^2 = b^2) ∧
  (b = 2 * a) ∧ (m < 0) → 
  m = -1 / 4 := 
sorry

end hyperbola_imaginary_axis_twice_real_axis_l47_47041


namespace ball_returns_velocity_required_initial_velocity_to_stop_l47_47841

-- Define the conditions.
def distance_A_to_wall : ℝ := 5
def distance_wall_to_B : ℝ := 2
def distance_AB : ℝ := 9
def initial_velocity_v0 : ℝ := 5
def acceleration_a : ℝ := -0.4

-- Hypothesize that the velocity when the ball returns to A is 3 m/s.
theorem ball_returns_velocity (t : ℝ) :
  distance_A_to_wall + distance_wall_to_B + distance_AB = 2 * (distance_A_to_wall + distance_wall_to_B) ∧
  initial_velocity_v0 * t + (1 / 2) * acceleration_a * t^2 = distance_AB + distance_A_to_wall →
  initial_velocity_v0 + acceleration_a * t = 3 := sorry

-- Hypothesize that to stop exactly at A, the initial speed should be 4 m/s.
theorem required_initial_velocity_to_stop (t' : ℝ) :
  distance_A_to_wall + distance_wall_to_B + distance_AB = 2 * (distance_A_to_wall + distance_wall_to_B) ∧
  (0.4 * t') * t' + (1 / 2) * acceleration_a * t'^2 = distance_AB + distance_A_to_wall →
  0.4 * t' = 4 := sorry

end ball_returns_velocity_required_initial_velocity_to_stop_l47_47841


namespace composite_numbers_equal_l47_47012

-- Define composite natural number
def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

-- Define principal divisors
def principal_divisors (n : ℕ) (principal1 principal2 : ℕ) : Prop :=
  is_composite n ∧ 
  (1 < principal1 ∧ principal1 < n) ∧ 
  (1 < principal2 ∧ principal2 < n) ∧
  principal1 * principal2 = n

-- Problem statement to prove
theorem composite_numbers_equal (a b p1 p2 : ℕ) :
  is_composite a → is_composite b →
  principal_divisors a p1 p2 → principal_divisors b p1 p2 →
  a = b :=
by
  sorry

end composite_numbers_equal_l47_47012


namespace sin_315_eq_neg_sqrt2_div_2_l47_47465

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47465


namespace triangle_is_right_triangle_l47_47782

theorem triangle_is_right_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : a ≠ b)
  (h₂ : (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B))
  (A_ne_B : A ≠ B)
  (hABC : A + B + C = Real.pi) :
  C = Real.pi / 2 :=
by
  sorry

end triangle_is_right_triangle_l47_47782


namespace prop_range_a_l47_47313

theorem prop_range_a (a : ℝ) 
  (p : ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 4 → x^2 ≥ a)
  (q : ∃ (x : ℝ), x^2 + 2 * a * x + (2 - a) = 0)
  : a = 1 ∨ a ≤ -2 :=
sorry

end prop_range_a_l47_47313


namespace total_legs_camden_dogs_l47_47286

variable (c r j : ℕ) -- c: Camden's dogs, r: Rico's dogs, j: Justin's dogs

theorem total_legs_camden_dogs :
  (r = j + 10) ∧ (j = 14) ∧ (c = (3 * r) / 4) → 4 * c = 72 :=
by
  sorry

end total_legs_camden_dogs_l47_47286


namespace smallest_d_l47_47394

theorem smallest_d (c d : ℕ) (h1 : c - d = 8)
  (h2 : Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16) : d = 4 := by
  sorry

end smallest_d_l47_47394


namespace quadratic_discriminant_l47_47804

theorem quadratic_discriminant {a b c : ℝ} (h : (a + b + c) * c ≤ 0) : b^2 ≥ 4 * a * c :=
sorry

end quadratic_discriminant_l47_47804


namespace sin_315_eq_neg_sqrt2_div_2_l47_47514

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l47_47514


namespace distance_from_mountains_l47_47017

/-- Given distances and scales from the problem description -/
def distance_between_mountains_map : ℤ := 312 -- in inches
def actual_distance_between_mountains : ℤ := 136 -- in km
def scale_A : ℤ := 1 -- 1 inch represents 1 km
def scale_B : ℤ := 2 -- 1 inch represents 2 km
def distance_from_mountain_A_map : ℤ := 25 -- in inches
def distance_from_mountain_B_map : ℤ := 40 -- in inches

/-- Prove the actual distances from Ram's camp to the mountains -/
theorem distance_from_mountains (dA dB : ℤ) :
  (dA = distance_from_mountain_A_map * scale_A) ∧ 
  (dB = distance_from_mountain_B_map * scale_B) :=
by {
  sorry -- Proof placeholder
}

end distance_from_mountains_l47_47017


namespace sin_315_equals_minus_sqrt2_div_2_l47_47475

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l47_47475


namespace total_pet_food_weight_ounces_l47_47644

-- Define the conditions
def cat_food_bag_weight : ℕ := 3 -- each cat food bag weighs 3 pounds
def cat_food_bags : ℕ := 2 -- number of cat food bags
def dog_food_extra_weight : ℕ := 2 -- each dog food bag weighs 2 pounds more than each cat food bag
def dog_food_bags : ℕ := 2 -- number of dog food bags
def pounds_to_ounces : ℕ := 16 -- number of ounces in each pound

-- Calculate the total weight of pet food in ounces
theorem total_pet_food_weight_ounces :
  let total_cat_food_weight := cat_food_bags * cat_food_bag_weight,
      dog_food_bag_weight := cat_food_bag_weight + dog_food_extra_weight,
      total_dog_food_weight := dog_food_bags * dog_food_bag_weight,
      total_weight_pounds := total_cat_food_weight + total_dog_food_weight,
      total_weight_ounces := total_weight_pounds * pounds_to_ounces
  in total_weight_ounces = 256 :=
by 
  -- The proof is not required, so we leave it as sorry.
  sorry

end total_pet_food_weight_ounces_l47_47644


namespace curve_of_constant_width_l47_47022

structure Curve :=
  (is_convex : Prop)

structure Point := 
  (x : ℝ) 
  (y : ℝ)

def rotate_180 (K : Curve) (O : Point) : Curve := sorry

def sum_curves (K1 K2 : Curve) : Curve := sorry

def is_circle_with_radius (K : Curve) (r : ℝ) : Prop := sorry

def constant_width (K : Curve) (w : ℝ) : Prop := sorry

theorem curve_of_constant_width {K : Curve} {O : Point} {h : ℝ} :
  K.is_convex →
  (K' : Curve) → K' = rotate_180 K O →
  is_circle_with_radius (sum_curves K K') h →
  constant_width K h :=
by 
  sorry

end curve_of_constant_width_l47_47022


namespace probability_all_successful_pairs_expected_successful_pairs_gt_half_l47_47409

-- Definition of conditions
def num_pairs_socks : ℕ := n

def successful_pair (socks : List (ℕ × ℕ)) (k : ℕ) : Prop :=
  (socks[n - k]).fst = (socks[n - k]).snd

-- Part (a): Prove the probability that all pairs are successful is \(\frac{2^n n!}{(2n)!}\)
theorem probability_all_successful_pairs (n : ℕ) :
  ∃ p : ℚ, p = (2^n * nat.factorial n) / nat.factorial (2 * n) :=
sorry

-- Part (b): Prove the expected number of successful pairs is greater than 0.5
theorem expected_successful_pairs_gt_half (n : ℕ) :
  (n / (2 * n - 1)) > 1 / 2 :=
sorry

end probability_all_successful_pairs_expected_successful_pairs_gt_half_l47_47409


namespace inequality_solution_l47_47236

theorem inequality_solution (x : ℝ) : x * |x| ≤ 1 ↔ x ≤ 1 := 
sorry

end inequality_solution_l47_47236


namespace increasing_on_iff_decreasing_on_periodic_even_l47_47317

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f x = f (x + p)
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x

theorem increasing_on_iff_decreasing_on_periodic_even :
  (is_even f ∧ is_periodic f 2 ∧ is_increasing_on f 0 1) ↔ is_decreasing_on f 3 4 := 
by
  sorry

end increasing_on_iff_decreasing_on_periodic_even_l47_47317


namespace percentage_in_quarters_l47_47081

theorem percentage_in_quarters:
  let dimes : ℕ := 40
  let quarters : ℕ := 30
  let value_dimes : ℕ := dimes * 10
  let value_quarters : ℕ := quarters * 25
  let total_value : ℕ := value_dimes + value_quarters
  let percentage_quarters : ℚ := (value_quarters : ℚ) / total_value * 100
  percentage_quarters = 65.22 := sorry

end percentage_in_quarters_l47_47081


namespace productivity_increase_l47_47086

theorem productivity_increase :
  (∃ d : ℝ, 
   (∀ n : ℕ, 0 < n → n ≤ 30 → 
      (5 + (n - 1) * d ≥ 0) ∧ 
      (30 * 5 + (30 * 29 / 2) * d = 390) ∧ 
      1 / 100 < d ∧ d < 1) ∧
      d = 0.52) :=
sorry

end productivity_increase_l47_47086


namespace rectangle_area_increase_l47_47848

theorem rectangle_area_increase
  (l w : ℝ)
  (h₀ : l > 0) -- original length is positive
  (h₁ : w > 0) -- original width is positive
  (length_increase : l' = 1.3 * l) -- new length after increase
  (width_increase : w' = 1.15 * w) -- new width after increase
  (new_area : A' = l' * w') -- new area after increase
  (original_area : A = l * w) -- original area
  :
  ((A' / A) * 100 - 100) = 49.5 := by
  sorry

end rectangle_area_increase_l47_47848


namespace sin_315_eq_neg_sqrt2_div_2_l47_47464

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47464


namespace negation_of_universal_prop_l47_47396

-- Define the conditions
variable (f : ℝ → ℝ)

-- Theorem statement
theorem negation_of_universal_prop : 
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) :=
by
  sorry

end negation_of_universal_prop_l47_47396


namespace toys_produced_each_day_l47_47095

theorem toys_produced_each_day (total_weekly_production : ℕ) (days_per_week : ℕ) (H1 : total_weekly_production = 6500) (H2 : days_per_week = 5) : (total_weekly_production / days_per_week = 1300) :=
by {
  sorry
}

end toys_produced_each_day_l47_47095


namespace ben_paints_area_l47_47435

variable (allen_ratio : ℕ) (ben_ratio : ℕ) (total_area : ℕ)
variable (total_ratio : ℕ := allen_ratio + ben_ratio)
variable (part_size : ℕ := total_area / total_ratio)

theorem ben_paints_area 
  (h1 : allen_ratio = 2)
  (h2 : ben_ratio = 6)
  (h3 : total_area = 360) : 
  ben_ratio * part_size = 270 := sorry

end ben_paints_area_l47_47435


namespace grayson_vs_rudy_distance_l47_47894

-- Definitions based on the conditions
def grayson_first_part_distance : Real := 25 * 1
def grayson_second_part_distance : Real := 20 * 0.5
def total_grayson_distance : Real := grayson_first_part_distance + grayson_second_part_distance
def rudy_distance : Real := 10 * 3

-- Theorem stating the problem to be proved
theorem grayson_vs_rudy_distance : total_grayson_distance - rudy_distance = 5 := by
  -- Proof would go here
  sorry

end grayson_vs_rudy_distance_l47_47894


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47524

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47524


namespace fraction_addition_l47_47450

theorem fraction_addition : (2 / 5 + 3 / 8) = 31 / 40 :=
by
  sorry

end fraction_addition_l47_47450


namespace inequality_holds_l47_47358

theorem inequality_holds (a b : ℕ) (ha : a > 1) (hb : b > 2) : a ^ b + 1 ≥ b * (a + 1) :=
sorry

end inequality_holds_l47_47358


namespace intersection_M_N_l47_47139

def setM : Set ℝ := {x | x^2 - 1 ≤ 0}
def setN : Set ℝ := {x | x^2 - 3 * x > 0}

theorem intersection_M_N :
  {x | -1 ≤ x ∧ x < 0} = setM ∩ setN :=
by
  sorry

end intersection_M_N_l47_47139


namespace probability_rolls_more_ones_than_eights_l47_47188

noncomputable def probability_more_ones_than_eights (n : ℕ) := 10246 / 32768

theorem probability_rolls_more_ones_than_eights :
  (probability_more_ones_than_eights 5) = 10246 / 32768 :=
by
  sorry

end probability_rolls_more_ones_than_eights_l47_47188


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47522

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47522


namespace find_value_of_fraction_l47_47614

theorem find_value_of_fraction (x y z : ℝ)
  (h1 : 3 * x - 4 * y - z = 0)
  (h2 : x + 4 * y - 15 * z = 0)
  (h3 : z ≠ 0) :
  (x^2 + 3 * x * y - y * z) / (y^2 + z^2) = 2.4 :=
by
  sorry

end find_value_of_fraction_l47_47614


namespace symmetric_angles_y_axis_l47_47194

theorem symmetric_angles_y_axis (α β : ℝ) (k : ℤ)
  (h : ∃ k : ℤ, β = 2 * k * π + (π - α)) :
  α + β = (2 * k + 1) * π ∨ α = -β + (2 * k + 1) * π :=
by sorry

end symmetric_angles_y_axis_l47_47194


namespace intermission_length_l47_47001

def concert_duration : ℕ := 80
def song_duration_total : ℕ := 70

theorem intermission_length : 
  concert_duration - song_duration_total = 10 :=
by
  -- conditions are already defined above
  sorry

end intermission_length_l47_47001


namespace triangle_angle_contradiction_l47_47220

-- Define the condition: all internal angles of the triangle are less than 60 degrees.
def condition (α β γ : ℝ) (h: α + β + γ = 180): Prop :=
  α < 60 ∧ β < 60 ∧ γ < 60

-- The proof statement
theorem triangle_angle_contradiction (α β γ : ℝ) (h_sum : α + β + γ = 180) (h: condition α β γ h_sum) : false :=
sorry

end triangle_angle_contradiction_l47_47220


namespace original_number_l47_47699

theorem original_number (x : ℝ) (h1 : 1.5 * x = 135) : x = 90 :=
by
  sorry

end original_number_l47_47699


namespace percentage_difference_l47_47906

noncomputable def P : ℝ := 40
variables {w x y z : ℝ}
variables (H1 : w = x * (1 - P / 100))
variables (H2 : x = 0.6 * y)
variables (H3 : z = 0.54 * y)
variables (H4 : z = 1.5 * w)

-- Goal
theorem percentage_difference : P = 40 :=
by sorry -- Proof omitted

end percentage_difference_l47_47906


namespace find_number_l47_47252

-- Given conditions and declarations
variable (x : ℕ)
variable (h : x / 3 = x - 42)

-- Proof problem statement
theorem find_number : x = 63 := 
sorry

end find_number_l47_47252


namespace kathleen_money_left_l47_47786

def june_savings : ℕ := 21
def july_savings : ℕ := 46
def august_savings : ℕ := 45

def school_supplies_expenses : ℕ := 12
def new_clothes_expenses : ℕ := 54

def total_savings : ℕ := june_savings + july_savings + august_savings
def total_expenses : ℕ := school_supplies_expenses + new_clothes_expenses

def total_money_left : ℕ := total_savings - total_expenses

theorem kathleen_money_left : total_money_left = 46 :=
by
  sorry

end kathleen_money_left_l47_47786


namespace probability_13_knowers_probability_14_knowers_expected_knowers_l47_47682

noncomputable def scientist_count : ℕ := 18
noncomputable def initial_knowers : ℕ := 10

def pairs (n : ℕ) : ℕ := n / 2

-- Part (a)
theorem probability_13_knowers : 
  ∀ (scientists : ℕ) (initial_knowers : ℕ), 
  scientists = 18 → initial_knowers = 10 →
  probability (λ (x : ℕ), x = 13) = 0 := sorry

-- Part (b)
theorem probability_14_knowers : 
  ∀ (scientists : ℕ) (initial_knowers : ℕ),
  scientists = 18 → initial_knowers = 10 →
  probability (λ (x : ℕ), x = 14) ≈ 0.461 := sorry

-- Part (c)
theorem expected_knowers : 
  ∀ (scientists : ℕ) (initial_knowers : ℕ),
  scientists = 18 → initial_knowers = 10 →
  expectation (λ (x: ℕ), x) ≈ 14.705 := sorry

end probability_13_knowers_probability_14_knowers_expected_knowers_l47_47682


namespace calculate_expression_l47_47455

theorem calculate_expression : -4^2 * (-1)^2022 = -16 :=
by
  sorry

end calculate_expression_l47_47455


namespace symmetric_about_origin_l47_47301

theorem symmetric_about_origin (x y : ℝ) :
  (∀ (x y : ℝ), (x*y - x^2 = 1) → ((-x)*(-y) - (-x)^2 = 1)) :=
by
  intros x y h
  sorry

end symmetric_about_origin_l47_47301


namespace number_of_distinct_colorings_l47_47201

def disks := Fin 8

inductive Color
| Blue
| Red
| Green

def coloring := disks → Color

def is_symmetry (c : coloring): Prop :=
  ∃ f : disks → disks, ∃ g: coloring, (∀ x, g (f x) = c x ∧
    ( (equiv.rotate 8).symm f = f ∨ (equiv.reflect.disk 8).symm f = f))

theorem number_of_distinct_colorings : 
  {p : ∀ c : coloring, is_symmetry c, (finset.image c p).size} = 38 := 
sorry

end number_of_distinct_colorings_l47_47201


namespace cannot_achieve_55_cents_with_six_coins_l47_47922

theorem cannot_achieve_55_cents_with_six_coins :
  ¬∃ (a b c d e : ℕ), 
    a + b + c + d + e = 6 ∧ 
    a * 1 + b * 5 + c * 10 + d * 25 + e * 50 = 55 := 
sorry

end cannot_achieve_55_cents_with_six_coins_l47_47922


namespace tan_shifted_value_l47_47184

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l47_47184


namespace trigonometry_identity_l47_47320

theorem trigonometry_identity (α : ℝ) (P : ℝ × ℝ) (h : P = (4, -3)) :
  let x := P.1
  let y := P.2
  let r := Real.sqrt (x^2 + y^2)
  x = 4 →
  y = -3 →
  r = 5 →
  Real.tan α = y / x := by
  intros x y r hx hy hr
  rw [hx, hy]
  simp [Real.tan, div_eq_mul_inv, mul_comm]
  sorry

end trigonometry_identity_l47_47320


namespace hydras_never_die_l47_47064

theorem hydras_never_die (heads_A heads_B : ℕ) (grow_heads : ℕ → ℕ → Prop) : 
  (heads_A = 2016) → 
  (heads_B = 2017) →
  (∀ a b : ℕ, grow_heads a b → (a = 5 ∨ a = 7) ∧ (b = 5 ∨ b = 7)) →
  (∀ (a b : ℕ), grow_heads a b → (heads_A + a - 2) ≠ (heads_B + b - 2)) :=
by
  intros hA hB hGrow
  intro hEq
  sorry

end hydras_never_die_l47_47064


namespace complement_union_eq_l47_47754

variable (U : Set Int := {-2, -1, 0, 1, 2, 3}) 
variable (A : Set Int := {-1, 0, 1}) 
variable (B : Set Int := {1, 2}) 

theorem complement_union_eq :
  U \ (A ∪ B) = {-2, 3} := by 
  sorry

end complement_union_eq_l47_47754


namespace combined_average_age_l47_47812

noncomputable def roomA : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
noncomputable def roomB : Set ℕ := {11, 12, 13, 14}
noncomputable def average_age_A := 55
noncomputable def average_age_B := 35
noncomputable def total_people := (10 + 4)
noncomputable def total_age_A := 10 * average_age_A
noncomputable def total_age_B := 4 * average_age_B
noncomputable def combined_total_age := total_age_A + total_age_B

theorem combined_average_age :
  (combined_total_age / total_people : ℚ) = 49.29 :=
by sorry

end combined_average_age_l47_47812


namespace problem_solution_l47_47074

theorem problem_solution : (275^2 - 245^2) / 30 = 520 := by
  sorry

end problem_solution_l47_47074


namespace kinetic_energy_reduction_collisions_l47_47431

theorem kinetic_energy_reduction_collisions (E_0 : ℝ) (n : ℕ) :
  (1 / 2)^n * E_0 = E_0 / 64 → n = 6 :=
by
  sorry

end kinetic_energy_reduction_collisions_l47_47431


namespace probability_median_is_55_l47_47093

def weights_of_measured := [60, 55, 60, 55, 65, 50, 50]

def unmeasured_weight_range := set.Icc (50 : ℝ) (60 : ℝ)

def is_median_55 (unmeasured_weight : ℝ) : Prop :=
  let all_weights := unmeasured_weight :: weights_of_measured in
  let sorted_weights := all_weights |>.qsort (· < ·) in
  sorted_weights.get 3 = 55 

theorem probability_median_is_55 :
  ∃ (μ : MeasureTheory.Measure (set.Icc 50 60)),
  μ.unmeasured_weight_range = volume.unmeasured_weight_range ∧
  μ {weight | is_median_55 weight } = (1/2 : ℝ) :=
sorry

end probability_median_is_55_l47_47093


namespace jerry_reaches_four_l47_47633

/-- Jerry starts at 0 on the real number line. He tosses a fair coin 8 times. When he gets heads, he moves 1 unit in the positive direction; when he gets tails, he moves 1 unit in the negative direction. Prove that the probability that he reaches 4 at some time during this process is 7/32, and then show that the sum of the numerator and denominator of this probability in reduced form is 39. -/
theorem jerry_reaches_four :
  let a := 7
  let b := 32
  (∑ i in Finset.range 9, if (i - (8 - i)) = 4 then Nat.choose 8 i else 0) =
  7 / 32 ∧ (a + b = 39) := 
sorry

end jerry_reaches_four_l47_47633


namespace days_worked_per_week_l47_47843

theorem days_worked_per_week (toys_per_week toys_per_day : ℕ) (h1 : toys_per_week = 5500) (h2 : toys_per_day = 1375) : toys_per_week / toys_per_day = 4 := by
  sorry

end days_worked_per_week_l47_47843


namespace problem_statement_l47_47738

noncomputable def inequality_not_necessarily_true (a b c : ℝ) :=
  c < b ∧ b < a ∧ a * c < 0

theorem problem_statement (a b c : ℝ) (h : inequality_not_necessarily_true a b c) : ¬ (∃ a b c : ℝ, c < b ∧ b < a ∧ a * c < 0 ∧ ¬ (b^2/c > a^2/c)) :=
by sorry

end problem_statement_l47_47738


namespace num_possibilities_for_asima_integer_l47_47982

theorem num_possibilities_for_asima_integer (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 65) :
  ∃ (n : ℕ), n = 64 :=
by
  sorry

end num_possibilities_for_asima_integer_l47_47982


namespace painted_cubes_l47_47551

theorem painted_cubes (n : ℕ) (h1 : 3 < n)
  (h2 : 6 * (n - 2)^2 = 12 * (n - 2)) :
  n = 4 := by
  sorry

end painted_cubes_l47_47551


namespace sin_315_degree_l47_47484

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l47_47484


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47509

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47509


namespace platform_protection_l47_47871

noncomputable def max_distance (r : ℝ) (n : ℕ) : ℝ :=
  if n > 2 then r / (Real.sin (180.0 / n)) else 0

noncomputable def coverage_ring_area (r : ℝ) (w : ℝ) : ℝ :=
  let inner_radius := r * (Real.sin 20.0)
  let outer_radius := inner_radius + w
  Real.pi * (outer_radius^2 - inner_radius^2)

theorem platform_protection :
  let r := 61
  let w := 22
  let n := 9
  max_distance r n = 60 / Real.sin 20.0 ∧
  coverage_ring_area r w = 2640 * Real.pi / Real.tan 20.0 := by
  sorry

end platform_protection_l47_47871


namespace probability_of_two_white_balls_correct_l47_47263

noncomputable def probability_of_two_white_balls : ℚ :=
  let total_balls := 15
  let white_balls := 8
  let first_draw_white := (white_balls : ℚ) / total_balls
  let second_draw_white := (white_balls - 1 : ℚ) / (total_balls - 1)
  first_draw_white * second_draw_white

theorem probability_of_two_white_balls_correct :
  probability_of_two_white_balls = 4 / 15 :=
by
  sorry

end probability_of_two_white_balls_correct_l47_47263


namespace total_birds_times_types_l47_47795

-- Defining the number of adults and offspring for each type of bird.
def num_ducks1 : ℕ := 2
def num_ducklings1 : ℕ := 5
def num_ducks2 : ℕ := 6
def num_ducklings2 : ℕ := 3
def num_ducks3 : ℕ := 9
def num_ducklings3 : ℕ := 6

def num_geese : ℕ := 4
def num_goslings : ℕ := 7

def num_swans : ℕ := 3
def num_cygnets : ℕ := 4

-- Calculate total number of birds
def total_ducks := (num_ducks1 * num_ducklings1 + num_ducks1) + (num_ducks2 * num_ducklings2 + num_ducks2) +
                      (num_ducks3 * num_ducklings3 + num_ducks3)

def total_geese := num_geese * num_goslings + num_geese
def total_swans := num_swans * num_cygnets + num_swans

def total_birds := total_ducks + total_geese + total_swans

-- Calculate the number of different types of birds
def num_types_of_birds : ℕ := 3 -- ducks, geese, swans

-- The final Lean statement to be proven
theorem total_birds_times_types :
  total_birds * num_types_of_birds = 438 :=
  by sorry

end total_birds_times_types_l47_47795


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l47_47458

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l47_47458


namespace work_required_to_pump_liquid_l47_47454

/-- Calculation of work required to pump a liquid of density ρ out of a parabolic boiler. -/
theorem work_required_to_pump_liquid
  (ρ g H a : ℝ)
  (h_pos : 0 < H)
  (a_pos : 0 < a) :
  ∃ (A : ℝ), A = (π * ρ * g * H^3) / (6 * a^2) :=
by
  -- TODO: Provide the proof.
  sorry

end work_required_to_pump_liquid_l47_47454


namespace fraction_addition_l47_47443

theorem fraction_addition (a b c d : ℚ) (ha : a = 2/5) (hb : b = 3/8) (hc : c = 31/40) :
  a + b = c :=
by
  rw [ha, hb, hc]
  -- The proof part is skipped here as per instructions
  sorry

end fraction_addition_l47_47443


namespace meera_fraction_4kmh_l47_47216

noncomputable def fraction_of_time_at_4kmh (total_time : ℝ) (x : ℝ) : ℝ :=
  x / total_time

theorem meera_fraction_4kmh (total_time x : ℝ) (h1 : x = total_time / 14) :
  fraction_of_time_at_4kmh total_time x = 1 / 14 :=
by
  sorry

end meera_fraction_4kmh_l47_47216


namespace raft_capacity_l47_47942

theorem raft_capacity (total_without_life_jackets : ℕ) (reduction_with_life_jackets : ℕ)
  (people_needing_life_jackets : ℕ) (total_capacity_with_life_jackets : ℕ)
  (no_life_jackets_capacity : total_without_life_jackets = 21)
  (life_jackets_reduction : reduction_with_life_jackets = 7)
  (life_jackets_needed : people_needing_life_jackets = 8) :
  total_capacity_with_life_jackets = 14 :=
by
  -- Proof should be here
  sorry

end raft_capacity_l47_47942


namespace age_difference_l47_47196

-- Defining the necessary variables and their types
variables (A B : ℕ)

-- Given conditions: 
axiom B_current_age : B = 38
axiom future_age_relationship : A + 10 = 2 * (B - 10)

-- Proof goal statement
theorem age_difference : A - B = 8 :=
by
  sorry

end age_difference_l47_47196


namespace f_difference_l47_47878

noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom local_f : ∀ x : ℝ, -2 < x ∧ x < 0 → f x = 2^x

-- State the problem
theorem f_difference :
  f 2012 - f 2011 = -1 / 2 := sorry

end f_difference_l47_47878


namespace sum_of_roots_equals_18_l47_47371

-- Define the conditions
variable (f : ℝ → ℝ)
variable (h_symmetry : ∀ x : ℝ, f (3 + x) = f (3 - x))
variable (h_distinct_roots : ∃ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0))

-- The theorem statement
theorem sum_of_roots_equals_18 (f : ℝ → ℝ) 
  (h_symmetry : ∀ x : ℝ, f (3 + x) = f (3 - x)) 
  (h_distinct_roots : ∃ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0)) :
  ∀ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0) → xs.sum id = 18 :=
by
  sorry

end sum_of_roots_equals_18_l47_47371


namespace maximum_value_of_expression_l47_47365

noncomputable def calc_value (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2)

theorem maximum_value_of_expression :
  ∃ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 ∧ x ≥ y ∧ y ≥ z ∧
  calc_value x y z = 2916 / 729 :=
by
  sorry

end maximum_value_of_expression_l47_47365


namespace sin_315_eq_neg_sqrt2_div_2_l47_47468

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47468


namespace john_moves_540kg_l47_47204

-- Conditions
def used_to_back_squat : ℝ := 200
def increased_by : ℝ := 50
def front_squat_ratio : ℝ := 0.8
def triple_ratio : ℝ := 0.9

-- Definitions based on conditions
def current_back_squat : ℝ := used_to_back_squat + increased_by
def current_front_squat : ℝ := front_squat_ratio * current_back_squat
def one_triple : ℝ := triple_ratio * current_front_squat
def three_triples : ℝ := 3 * one_triple

-- The proof statement
theorem john_moves_540kg : three_triples = 540 := by
  sorry

end john_moves_540kg_l47_47204


namespace total_animals_correct_l47_47765

-- Define the number of aquariums and the number of animals per aquarium.
def num_aquariums : ℕ := 26
def animals_per_aquarium : ℕ := 2

-- Define the total number of saltwater animals.
def total_animals : ℕ := num_aquariums * animals_per_aquarium

-- The statement we want to prove.
theorem total_animals_correct : total_animals = 52 := by
  -- Proof is omitted.
  sorry

end total_animals_correct_l47_47765


namespace book_distribution_l47_47293

theorem book_distribution (x : ℕ) (books : ℕ) :
  (books = 3 * x + 8) ∧ (books < 5 * x - 5 + 2) → (x = 6 ∧ books = 26) :=
by
  sorry

end book_distribution_l47_47293


namespace sin_315_eq_neg_sqrt2_div_2_l47_47540

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47540


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l47_47462

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l47_47462


namespace tan_angle_addition_l47_47167

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l47_47167


namespace poem_lines_added_l47_47039

theorem poem_lines_added (x : ℕ) 
  (initial_lines : ℕ)
  (months : ℕ)
  (final_lines : ℕ)
  (h_init : initial_lines = 24)
  (h_months : months = 22)
  (h_final : final_lines = 90)
  (h_equation : initial_lines + months * x = final_lines) :
  x = 3 :=
by {
  -- Placeholder for the proof
  sorry
}

end poem_lines_added_l47_47039


namespace tan_add_pi_over_3_l47_47163

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l47_47163


namespace find_m_l47_47585

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l47_47585


namespace successful_pairs_probability_expected_successful_pairs_l47_47408

-- Part (a)
theorem successful_pairs_probability {n : ℕ} : 
  (2 * n)!! = ite (n = 0) 1 ((2 * n) * (2 * n - 2) * (2 * n - 4) * ... * 2) →
  (fact (2 * n) / (2 ^ n * fact n)) = (2 ^ n * fact n / fact (2 * n)) :=
sorry

-- Part (b)
theorem expected_successful_pairs {n : ℕ} :
  (∑ k in range n, (if (successful_pair k) then 1 else 0)) / n > 0.5 :=
sorry

end successful_pairs_probability_expected_successful_pairs_l47_47408


namespace domain_of_f_l47_47654

noncomputable def f (x: ℝ): ℝ := 1 / Real.sqrt (x - 2)

theorem domain_of_f:
  {x: ℝ | 2 < x} = {x: ℝ | f x = 1 / Real.sqrt (x - 2)} :=
by
  sorry

end domain_of_f_l47_47654


namespace order_of_a_b_c_l47_47733

noncomputable def a : ℝ := (Real.log (Real.sqrt 2)) / 2
noncomputable def b : ℝ := Real.log 3 / 6
noncomputable def c : ℝ := 1 / (2 * Real.exp 1)

theorem order_of_a_b_c : c > b ∧ b > a := by
  sorry

end order_of_a_b_c_l47_47733


namespace exists_triangle_l47_47912

variables {α : Type*} [affine_space α] [decidable_eq α]

open_locale affine

-- Given Sets and predicate conditions
def finite_set_points (s : set α) : Prop := set.finite s
def disjoint_sets (A B : finset α) : Prop := A ∩ B = ∅
def non_collinear_sets (S : finset α) : Prop := ∀ (x y z : α), x ≠ y → y ≠ z → x ≠ z → set.collinear ↥S {x, y, z} = false
def at_least_five_points_set (S : finset α) : Prop := S.card ≥ 5

theorem exists_triangle {A B : finset α} (h1 : disjoint_sets A B) (h2 : non_collinear_sets (A ∪ B)) 
(h3: at_least_five_points_set A ∨ at_least_five_points_set B) :
∃ T, (T ⊆ A ∨ T ⊆ B) ∧ T.card = 3 ∧ ∀ p ∈ (A ∪ B).filter (λ x, x ∉ T), ¬point_inside_triangle p T := 
sorry

end exists_triangle_l47_47912


namespace complement_union_eq_l47_47744

open Set

-- Definition of sets U, A, and B
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- Statement of the problem
theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 3} :=
by sorry

end complement_union_eq_l47_47744


namespace smallest_term_4_in_c_seq_l47_47877

noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

noncomputable def b_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else n * (n - 1) + 15

noncomputable def c_seq (n : ℕ) : ℚ :=
  if n = 0 then 0 else (b_seq n) / (a_seq n)

theorem smallest_term_4_in_c_seq : 
  ∀ n : ℕ, n > 0 → c_seq 4 ≤ c_seq n :=
sorry

end smallest_term_4_in_c_seq_l47_47877


namespace hexagonal_prism_surface_area_l47_47938

theorem hexagonal_prism_surface_area (h : ℝ) (a : ℝ) (H_h : h = 6) (H_a : a = 4) : 
  let base_area := 2 * (3 * a^2 * (Real.sqrt 3) / 2)
  let lateral_area := 6 * a * h
  let total_area := lateral_area + base_area
  total_area = 48 * (3 + Real.sqrt 3) :=
by
  -- let base_area := 2 * (3 * a^2 * (Real.sqrt 3) / 2)
  -- let lateral_area := 6 * a * h
  -- let total_area := lateral_area + base_area
  -- total_area = 48 * (3 + Real.sqrt 3)
  sorry

end hexagonal_prism_surface_area_l47_47938


namespace only_function_B_has_inverse_l47_47553

-- Definitions based on the problem conditions
def function_A (x : ℝ) : ℝ := 3 - x^2 -- Parabola opening downwards with vertex at (0,3)
def function_B (x : ℝ) : ℝ := x -- Straight line with slope 1 passing through (0,0) and (1,1)
def function_C (x y : ℝ) : Prop := x^2 + y^2 = 4 -- Circle centered at (0,0) with radius 2

-- Theorem stating that only function B has an inverse
theorem only_function_B_has_inverse :
  (∀ y : ℝ, ∃! x : ℝ, function_B x = y) ∧
  (¬∀ y : ℝ, ∃! x : ℝ, function_A x = y) ∧
  (¬∀ y : ℝ, ∃! x : ℝ, ∃ y1 y2 : ℝ, function_C x y1 ∧ function_C x y2 ∧ y1 ≠ y2) :=
  by 
  sorry -- Proof not required

end only_function_B_has_inverse_l47_47553


namespace sin_315_eq_neg_sqrt2_div_2_l47_47533

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47533


namespace total_accidents_l47_47357

-- Define the given vehicle counts for the highways
def total_vehicles_A : ℕ := 4 * 10^9
def total_vehicles_B : ℕ := 2 * 10^9
def total_vehicles_C : ℕ := 1 * 10^9

-- Define the accident ratios per highway
def accident_ratio_A : ℕ := 80
def accident_ratio_B : ℕ := 120
def accident_ratio_C : ℕ := 65

-- Define the number of vehicles in millions
def million := 10^6

-- Define the accident calculations per highway
def accidents_A : ℕ := (total_vehicles_A / (100 * million)) * accident_ratio_A
def accidents_B : ℕ := (total_vehicles_B / (200 * million)) * accident_ratio_B
def accidents_C : ℕ := (total_vehicles_C / (50 * million)) * accident_ratio_C

-- Prove the total number of accidents across all highways
theorem total_accidents : accidents_A + accidents_B + accidents_C = 5700 := by
  have : accidents_A = 3200 := by sorry
  have : accidents_B = 1200 := by sorry
  have : accidents_C = 1300 := by sorry
  sorry

end total_accidents_l47_47357


namespace find_m_l47_47590

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l47_47590


namespace find_function_l47_47794

-- Let f be a differentiable function over all real numbers
variable (f : ℝ → ℝ)
variable (k : ℝ)

-- Condition: f is differentiable over (-∞, ∞)
variable (h_diff : differentiable ℝ f)

-- Condition: f(0) = 1
variable (h_init : f 0 = 1)

-- Condition: for any x1, x2 in ℝ, f(x1 + x2) ≥ f(x1) f(x2)
variable (h_ineq : ∀ x1 x2 : ℝ, f (x1 + x2) ≥ f x1 * f x2)

-- We aim to prove: f(x) = e^(kx)
theorem find_function : ∃ k : ℝ, ∀ x : ℝ, f x = Real.exp (k * x) :=
sorry

end find_function_l47_47794


namespace least_value_of_x_for_divisibility_l47_47085

theorem least_value_of_x_for_divisibility (x : ℕ) (h : 1 + 8 + 9 + 4 = 22) :
  ∃ x : ℕ, (22 + x) % 3 = 0 ∧ x = 2 := by
sorry

end least_value_of_x_for_divisibility_l47_47085


namespace length_of_ladder_l47_47697

theorem length_of_ladder (a b : ℝ) (ha : a = 20) (hb : b = 15) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 25 := by
  sorry

end length_of_ladder_l47_47697


namespace sphere_radius_eq_three_l47_47338

theorem sphere_radius_eq_three (r : ℝ) (h : 4 / 3 * π * r ^ 3 = 4 * π * r ^ 2) : r = 3 :=
by
  sorry

end sphere_radius_eq_three_l47_47338


namespace angle_inequality_l47_47829

def valid_angles (x : ℝ) (k : ℤ) : Prop :=
  (-π/6 + 2*k*π < x ∧ x < π/6 + 2*k*π) ∨ (2*π/3 + 2*k*π < x ∧ x < 4*π/3 + 2*k*π)

theorem angle_inequality (x : ℝ) : 
  (cos (2 * x) - 4 * cos (π / 4) * cos (5 * π / 12) * cos x + cos (5 * π / 6) + 1 > 0) ↔ 
  ∃ k : ℤ, valid_angles x k :=
sorry

end angle_inequality_l47_47829


namespace max_min_ab_bc_ca_l47_47913

theorem max_min_ab_bc_ca (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 12) (h_ab_bc_ca : a * b + b * c + c * a = 30) :
  max (min (a * b) (min (b * c) (c * a))) = 9 :=
sorry

end max_min_ab_bc_ca_l47_47913


namespace polynomial_divisibility_l47_47986

-- Define the polynomial f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^3 - 8 * x^2 + m * x - 16

-- Prove that f(x) is divisible by x-2 if and only if m=8
theorem polynomial_divisibility (m : ℝ) :
  (∀ (x : ℝ), (x - 2) ∣ f x m) ↔ m = 8 := 
by
  sorry

end polynomial_divisibility_l47_47986


namespace jersey_to_shoes_ratio_l47_47002

theorem jersey_to_shoes_ratio
  (pairs_shoes: ℕ) (jerseys: ℕ) (total_cost: ℝ) (total_cost_shoes: ℝ) 
  (shoes: pairs_shoes = 6) (jer: jerseys = 4) (total: total_cost = 560) (cost_sh: total_cost_shoes = 480) :
  ((total_cost - total_cost_shoes) / jerseys) / (total_cost_shoes / pairs_shoes) = 1 / 4 := 
by 
  sorry

end jersey_to_shoes_ratio_l47_47002


namespace prob_neg2_leq_X_leq_2_l47_47311

noncomputable def normal_distribution (μ σ : ℝ) : ProbabilityMassFunction ℝ := sorry

variable {X : ℝ}
variable {σ : ℝ}
variable {a : ℝ}

-- Condition: X ~ N(0, σ^2)
axiom axiom_normal : X ∼ normal_distribution 0 σ^2
-- Condition: P(x > 2) = a and 0 < a < 1
axiom axiom_prob_gt_2 : 0 < a ∧ a < 1 ∧ P(X > 2) = a

theorem prob_neg2_leq_X_leq_2 : P(-2 ≤ X ∧ X ≤ 2) = 1 - 2 * a :=
by 
  sorry

end prob_neg2_leq_X_leq_2_l47_47311


namespace probability_of_two_white_balls_correct_l47_47264

noncomputable def probability_of_two_white_balls : ℚ :=
  let total_balls := 15
  let white_balls := 8
  let first_draw_white := (white_balls : ℚ) / total_balls
  let second_draw_white := (white_balls - 1 : ℚ) / (total_balls - 1)
  first_draw_white * second_draw_white

theorem probability_of_two_white_balls_correct :
  probability_of_two_white_balls = 4 / 15 :=
by
  sorry

end probability_of_two_white_balls_correct_l47_47264


namespace total_number_of_coins_l47_47668

theorem total_number_of_coins (x n : Nat) (h1 : 15 * 5 = 75) (h2 : 125 - 75 = 50)
  (h3 : x = 50 / 2) (h4 : n = x + 15) : n = 40 := by
  sorry

end total_number_of_coins_l47_47668


namespace tan_shifted_value_l47_47180

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l47_47180


namespace num_seven_digit_palindromes_l47_47129

theorem num_seven_digit_palindromes : 
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices = 9000 :=
by
  sorry

end num_seven_digit_palindromes_l47_47129


namespace surface_area_of_modified_structure_l47_47550

-- Define the given conditions
def initial_cube_side_length : ℕ := 12
def smaller_cube_side_length : ℕ := 2
def smaller_cubes_count : ℕ := 72
def face_center_cubes_count : ℕ := 6

-- Define the calculation of the surface area
def single_smaller_cube_surface_area : ℕ := 6 * (smaller_cube_side_length ^ 2)
def added_surface_from_removed_center_cube : ℕ := 4 * (smaller_cube_side_length ^ 2)
def modified_smaller_cube_surface_area : ℕ := single_smaller_cube_surface_area + added_surface_from_removed_center_cube
def unaffected_smaller_cubes : ℕ := smaller_cubes_count - face_center_cubes_count

-- Define the given surface area according to the problem
def correct_surface_area : ℕ := 1824

-- The equivalent proof problem statement
theorem surface_area_of_modified_structure : 
    66 * single_smaller_cube_surface_area + 6 * modified_smaller_cube_surface_area = correct_surface_area := 
by
    -- placeholders for the actual proof
    sorry

end surface_area_of_modified_structure_l47_47550


namespace simplify_expression_l47_47858
open Real

theorem simplify_expression (x y : ℝ) : -x + y - 2 * x - 3 * y = -3 * x - 2 * y :=
by
  sorry

end simplify_expression_l47_47858


namespace negation_prop_l47_47043

theorem negation_prop (x : ℝ) : (¬ (∀ x : ℝ, Real.exp x > x^2)) ↔ (∃ x : ℝ, Real.exp x ≤ x^2) :=
by
  sorry

end negation_prop_l47_47043


namespace r_needs_35_days_l47_47253

def work_rate (P Q R: ℚ) : Prop :=
  (P = Q + R) ∧ (P + Q = 1/10) ∧ (Q = 1/28)

theorem r_needs_35_days (P Q R: ℚ) (h: work_rate P Q R) : 1 / R = 35 :=
by 
  sorry

end r_needs_35_days_l47_47253


namespace num_cars_in_parking_lot_l47_47910

-- Define the conditions
variable (C : ℕ) -- Number of cars
def number_of_bikes := 5 -- Number of bikes given
def total_wheels := 66 -- Total number of wheels given
def wheels_per_bike := 2 -- Number of wheels per bike
def wheels_per_car := 4 -- Number of wheels per car

-- Define the proof statement
theorem num_cars_in_parking_lot 
  (h1 : total_wheels = 66) 
  (h2 : number_of_bikes = 5) 
  (h3 : wheels_per_bike = 2)
  (h4 : wheels_per_car = 4) 
  (h5 : C * wheels_per_car + number_of_bikes * wheels_per_bike = total_wheels) :
  C = 14 :=
by
  sorry

end num_cars_in_parking_lot_l47_47910


namespace costOfBrantsRoyalBananaSplitSundae_l47_47675

-- Define constants for the prices of the known sundaes
def yvette_sundae_cost : ℝ := 9.00
def alicia_sundae_cost : ℝ := 7.50
def josh_sundae_cost : ℝ := 8.50

-- Define the tip percentage
def tip_percentage : ℝ := 0.20

-- Define the final bill amount
def final_bill : ℝ := 42.00

-- Calculate the total known sundaes cost
def total_known_sundaes_cost : ℝ := yvette_sundae_cost + alicia_sundae_cost + josh_sundae_cost

-- Define a proof to show that the cost of Brant's sundae is $10.00
theorem costOfBrantsRoyalBananaSplitSundae : 
  total_known_sundaes_cost + b = final_bill / (1 + tip_percentage) → b = 10 :=
sorry

end costOfBrantsRoyalBananaSplitSundae_l47_47675


namespace hydrae_never_equal_heads_l47_47058

theorem hydrae_never_equal_heads :
  ∀ (a b : ℕ), a = 2016 → b = 2017 →
  (∀ (a' b' : ℕ), a' ∈ {5, 7} → b' ∈ {5, 7} → 
  ∀ n : ℕ, let aa := a + n * 5 + (n - a / 7) * 2 - n in
           let bb := b + n * 5 + (n - b / 7) * 2 - n in
  aa + bb ≠ 2 * (aa / 2)) → 
  true :=
begin
  -- Sorry, the proof is left as an exercise
  sorry,
end

end hydrae_never_equal_heads_l47_47058


namespace sin_315_degree_is_neg_sqrt_2_div_2_l47_47494

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l47_47494


namespace eliot_account_balance_l47_47834

theorem eliot_account_balance (A E : ℝ) 
  (h1 : A > E)
  (h2 : A - E = (1 / 12) * (A + E))
  (h3 : 1.10 * A - 1.15 * E = 22) : 
  E = 146.67 :=
by
  sorry

end eliot_account_balance_l47_47834


namespace multiplication_identity_l47_47373

theorem multiplication_identity (x y : ℝ) : 
  (2*x^3 - 5*y^2) * (4*x^6 + 10*x^3*y^2 + 25*y^4) = 8*x^9 - 125*y^6 := 
by
  sorry

end multiplication_identity_l47_47373


namespace birds_problem_l47_47961

-- Define the initial number of birds and the total number of birds as given conditions.
def initial_birds : ℕ := 2
def total_birds : ℕ := 6

-- Define the number of new birds that came to join.
def new_birds : ℕ := total_birds - initial_birds

-- State the theorem to be proved, asserting that the number of new birds is 4.
theorem birds_problem : new_birds = 4 := 
by
  -- required proof goes here
  sorry

end birds_problem_l47_47961


namespace feet_of_wood_required_l47_47349

def rung_length_in_inches : ℤ := 18
def spacing_between_rungs_in_inches : ℤ := 6
def height_to_climb_in_feet : ℤ := 50

def feet_per_rung := rung_length_in_inches / 12
def rungs_per_foot := 12 / spacing_between_rungs_in_inches
def total_rungs := height_to_climb_in_feet * rungs_per_foot
def total_feet_of_wood := total_rungs * feet_per_rung

theorem feet_of_wood_required :
  total_feet_of_wood = 150 :=
by
  sorry

end feet_of_wood_required_l47_47349


namespace at_least_one_ge_one_l47_47383

theorem at_least_one_ge_one (x y : ℝ) (h : x + y ≥ 2) : x ≥ 1 ∨ y ≥ 1 :=
sorry

end at_least_one_ge_one_l47_47383


namespace inequality_solution_l47_47401

theorem inequality_solution : { x : ℝ | (x - 1) / (x + 3) < 0 } = { x : ℝ | -3 < x ∧ x < 1 } :=
sorry

end inequality_solution_l47_47401


namespace calculate_lunch_break_duration_l47_47801

noncomputable def paula_rate (p : ℝ) : Prop := p > 0
noncomputable def helpers_rate (h : ℝ) : Prop := h > 0
noncomputable def apprentice_rate (a : ℝ) : Prop := a > 0
noncomputable def lunch_break_duration (L : ℝ) : Prop := L >= 0

-- Monday's work equation
noncomputable def monday_work (p h a L : ℝ) (monday_work_done : ℝ) :=
  0.6 = (9 - L) * (p + h + a)

-- Tuesday's work equation
noncomputable def tuesday_work (h a L : ℝ) (tuesday_work_done : ℝ) :=
  0.3 = (7 - L) * (h + a)

-- Wednesday's work equation
noncomputable def wednesday_work (p a L : ℝ) (wednesday_work_done : ℝ) :=
  0.1 = (1.2 - L) * (p + a)

-- Final proof statement
theorem calculate_lunch_break_duration (p h a L : ℝ)
  (H1 : paula_rate p)
  (H2 : helpers_rate h)
  (H3 : apprentice_rate a)
  (H4 : lunch_break_duration L)
  (H5 : monday_work p h a L 0.6)
  (H6 : tuesday_work h a L 0.3)
  (H7 : wednesday_work p a L 0.1) :
  L = 1.4 :=
sorry

end calculate_lunch_break_duration_l47_47801


namespace part1_part2_l47_47872

-- Define the conditions for p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 6) <= 0
def q (x m : ℝ) : Prop := (2 - m <= x) ∧ (x <= 2 + m)

-- Proof statement for part (1)
theorem part1 (m: ℝ) : 
  (∀ x : ℝ, p x → q x m) → 4 <= m :=
sorry

-- Proof statement for part (2)
theorem part2 (x : ℝ) (m : ℝ) : 
  (m = 5) → (p x ∨ q x m) ∧ ¬(p x ∧ q x m) → x ∈ Set.Ico (-3) (-2) ∪ Set.Ioc 6 7 :=
sorry

end part1_part2_l47_47872


namespace find_planes_l47_47625

-- Define the conditions given in the problem
def plane_intersecting (n : ℕ) : Prop :=
  ∀ (k : ℕ), k = 1999 → ∀ i, 1 ≤ i < n-1 → ∃ (p : ℕ), p = 1999 ∧ (p ≠ k)

theorem find_planes (n : ℕ) :
  plane_intersecting n →
  n = 2000 ∨ n = 3998 :=
by
  sorry

end find_planes_l47_47625


namespace highest_elevation_l47_47430

-- Define the function for elevation as per the conditions
def elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2

-- Prove that the highest elevation reached is 500 meters
theorem highest_elevation : (exists t : ℝ, elevation t = 500) ∧ (∀ t : ℝ, elevation t ≤ 500) := sorry

end highest_elevation_l47_47430


namespace paint_replacement_fractions_l47_47907

variables {r b g : ℚ}

/-- Given the initial and replacement intensities and the final intensities of red, blue,
and green paints respectively, prove the fractions of the original amounts of each paint color
that were replaced. -/
theorem paint_replacement_fractions :
  (0.6 * (1 - r) + 0.3 * r = 0.4) ∧
  (0.4 * (1 - b) + 0.15 * b = 0.25) ∧
  (0.25 * (1 - g) + 0.1 * g = 0.18) →
  (r = 2/3) ∧ (b = 3/5) ∧ (g = 7/15) :=
by
  sorry

end paint_replacement_fractions_l47_47907


namespace proof_problem_l47_47650

theorem proof_problem
  (a b c : ℂ)
  (h1 : ac / (a + b) + ba / (b + c) + cb / (c + a) = -4)
  (h2 : bc / (a + b) + ca / (b + c) + ab / (c + a) = 7) :
  b / (a + b) + c / (b + c) + a / (c + a) = 7 := 
sorry

end proof_problem_l47_47650


namespace find_a3_l47_47875

variable {α : Type} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) :=
  ∃ r : α, ∀ n, a (n + 1) = a n * r

theorem find_a3 (a : ℕ → α) (h : geometric_sequence a) (h1 : a 0 * a 4 = 16) :
  a 2 = 4 ∨ a 2 = -4 :=
by
  sorry

end find_a3_l47_47875


namespace problem1_problem2_problem3_problem4_l47_47284

-- Problem 1
theorem problem1 (a : ℝ) : -2 * a^3 * 3 * a^2 = -6 * a^5 := 
by
  sorry

-- Problem 2
theorem problem2 (m : ℝ) : m^4 * (m^2)^3 / m^8 = m^2 := 
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (-2 * x - 1) * (2 * x - 1) = 1 - 4 * x^2 := 
by
  sorry

-- Problem 4
theorem problem4 (x : ℝ) : (-3 * x + 2)^2 = 9 * x^2 - 12 * x + 4 := 
by
  sorry

end problem1_problem2_problem3_problem4_l47_47284


namespace why_build_offices_l47_47992

structure Company where
  name : String
  hasSkillfulEmployees : Prop
  uniqueComfortableWorkEnvironment : Prop
  integratedWorkLeisureSpaces : Prop
  reducedEmployeeStress : Prop
  flexibleWorkSchedules : Prop
  increasesProfit : Prop

theorem why_build_offices (goog_fb : Company)
  (h1 : goog_fb.hasSkillfulEmployees)
  (h2 : goog_fb.uniqueComfortableWorkEnvironment)
  (h3 : goog_fb.integratedWorkLeisureSpaces)
  (h4 : goog_fb.reducedEmployeeStress)
  (h5 : goog_fb.flexibleWorkSchedules) :
  goog_fb.increasesProfit := 
sorry

end why_build_offices_l47_47992


namespace instantaneous_velocity_at_t2_l47_47087

open Real

theorem instantaneous_velocity_at_t2 :
  (∃ (s : ℝ → ℝ), (∀ t, s t = (1 / 8) * t^2) ∧ deriv s 2 = 1 / 2) :=
by {
  let s := λ t : ℝ, (1 / 8) * t^2,
  use s,
  split,
  { intro t,
    refl, },
  { have h_deriv : deriv s 2 = (1 / 4) * 2,
    { dsimp [s],
      calc deriv (λ t, (1 / 8) * t^2) 2
          = (1 / 8) * deriv (λ t, t^2) 2 : by apply deriv_const_mul
      ... = (1 / 8) * (2 * 2) : by simp [deriv_pow],
    },
    rw h_deriv,
    norm_num,
  },
  sorry
}

end instantaneous_velocity_at_t2_l47_47087


namespace find_x_l47_47310

noncomputable def x (n : ℕ) := 6^n + 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_three_prime_divisors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ Prime a ∧ Prime b ∧ Prime c ∧ a * b * c ∣ x ∧ ∀ d, Prime d ∧ d ∣ x → d = a ∨ d = b ∨ d = c

theorem find_x (n : ℕ) (hodd : is_odd n) (hdiv : has_three_prime_divisors (x n)) (hprime : 11 ∣ (x n)) : x n = 7777 :=
by 
  sorry

end find_x_l47_47310


namespace prob_all_successful_pairs_l47_47407

theorem prob_all_successful_pairs (n : ℕ) : 
  let num_pairs := (2 * n)!
  let num_successful_pairs := 2^n * n!
  prob_all_successful_pairs := num_successful_pairs / num_pairs 
  prob_all_successful_pairs = (2^n * n!) / (2 * n)! := 
sorry

end prob_all_successful_pairs_l47_47407


namespace find_m_l47_47588

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l47_47588


namespace books_sold_correct_l47_47014

-- Define the number of books sold by Matias, Olivia, and Luke on each day
def matias_monday := 7
def olivia_monday := 5
def luke_monday := 12

def matias_tuesday := 2 * matias_monday
def olivia_tuesday := 3 * olivia_monday
def luke_tuesday := luke_monday / 2

def matias_wednesday := 3 * matias_tuesday
def olivia_wednesday := 4 * olivia_tuesday
def luke_wednesday := luke_tuesday

-- Calculate the total books sold by each person over three days
def matias_total := matias_monday + matias_tuesday + matias_wednesday
def olivia_total := olivia_monday + olivia_tuesday + olivia_wednesday
def luke_total := luke_monday + luke_tuesday + luke_wednesday

-- Calculate the combined total of books sold by Matias, Olivia, and Luke
def combined_total := matias_total + olivia_total + luke_total

-- Prove the combined total equals 167
theorem books_sold_correct : combined_total = 167 := by
  sorry

end books_sold_correct_l47_47014


namespace simple_interest_principal_l47_47433

theorem simple_interest_principal (R : ℝ) (P : ℝ) (h : P * 7 * (R + 2) / 100 = P * 7 * R / 100 + 140) : P = 1000 :=
by
  sorry

end simple_interest_principal_l47_47433


namespace complement_union_eq_l47_47747

open Set

-- Definition of sets U, A, and B
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- Statement of the problem
theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 3} :=
by sorry

end complement_union_eq_l47_47747


namespace contrapositive_l47_47038

theorem contrapositive (x : ℝ) : (x > 1 → x^2 + x > 2) ↔ (x^2 + x ≤ 2 → x ≤ 1) :=
sorry

end contrapositive_l47_47038


namespace enrollment_difference_l47_47070

theorem enrollment_difference :
  let M := 1500
  let S := 2100
  let L := 2700
  let R := 1800
  let B := 900
  max M (max S (max L (max R B))) - min M (min S (min L (min R B))) = 1800 := 
by 
  sorry

end enrollment_difference_l47_47070


namespace seahawks_final_score_l47_47241

def num_touchdowns : ℕ := 4
def num_field_goals : ℕ := 3
def points_per_touchdown : ℕ := 7
def points_per_fieldgoal : ℕ := 3

theorem seahawks_final_score : (num_touchdowns * points_per_touchdown) + (num_field_goals * points_per_fieldgoal) = 37 := by
  sorry

end seahawks_final_score_l47_47241


namespace no_solution_exists_l47_47921

theorem no_solution_exists (a b : ℤ) : ∃ c : ℤ, ∀ m n : ℤ, m^2 + a * m + b ≠ 2 * n^2 + 2 * n + c :=
by {
  -- Insert correct proof here
  sorry
}

end no_solution_exists_l47_47921


namespace find_pirates_l47_47969

def pirate_problem (p : ℕ) (nonfighters : ℕ) (arm_loss_percent both_loss_percent : ℝ) (leg_loss_fraction : ℚ) : Prop :=
  let participants := p - nonfighters in
  let arm_loss := arm_loss_percent * participants in
  let both_loss := both_loss_percent * participants in
  let leg_loss (p : ℕ) := leg_loss_fraction * p in
  let only_leg_loss (p : ℕ) := leg_loss p - both_loss in
  arm_loss + only_leg_loss p + both_loss = leg_loss p

theorem find_pirates : ∃ p : ℕ, pirate_problem p 10 0.54 0.34 (2/3) :=
sorry

end find_pirates_l47_47969


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l47_47463

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l47_47463


namespace spaces_per_tray_l47_47989

-- Conditions
def num_ice_cubes_glass : ℕ := 8
def num_ice_cubes_pitcher : ℕ := 2 * num_ice_cubes_glass
def total_ice_cubes_used : ℕ := num_ice_cubes_glass + num_ice_cubes_pitcher
def num_trays : ℕ := 2

-- Proof statement
theorem spaces_per_tray : total_ice_cubes_used / num_trays = 12 :=
by
  sorry

end spaces_per_tray_l47_47989


namespace remainder_of_product_mod_17_l47_47673

theorem remainder_of_product_mod_17 :
  (2005 * 2006 * 2007 * 2008 * 2009) % 17 = 0 :=
sorry

end remainder_of_product_mod_17_l47_47673


namespace smallest_d_l47_47393

theorem smallest_d (c d : ℕ) (h1 : c - d = 8)
  (h2 : Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16) : d = 4 := by
  sorry

end smallest_d_l47_47393


namespace trains_meet_1050_km_from_delhi_l47_47955

def distance_train_meet (t1_departure t2_departure : ℕ) (s1 s2 : ℕ) : ℕ :=
  let t_gap := t2_departure - t1_departure      -- Time difference between the departures in hours
  let d1 := s1 * t_gap                          -- Distance covered by the first train until the second train starts
  let relative_speed := s2 - s1                 -- Relative speed of the second train with respect to the first train
  d1 + s2 * (d1 / relative_speed)               -- Distance from Delhi where they meet

theorem trains_meet_1050_km_from_delhi :
  distance_train_meet 9 14 30 35 = 1050 := by
  -- Definitions based on the problem's conditions
  let t1 := 9          -- First train departs at 9 a.m.
  let t2 := 14         -- Second train departs at 2 p.m. (14:00 in 24-hour format)
  let s1 := 30         -- Speed of the first train in km/h
  let s2 := 35         -- Speed of the second train in km/h
  sorry -- proof to be filled in

end trains_meet_1050_km_from_delhi_l47_47955


namespace product_of_primes_sum_101_l47_47665

theorem product_of_primes_sum_101 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 101) : p * q = 194 := by
  sorry

end product_of_primes_sum_101_l47_47665


namespace complement_union_l47_47758

variable (U : Set ℤ)
variable (A : Set ℤ)
variable (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} :=
sorry

end complement_union_l47_47758


namespace total_spent_is_64_l47_47837

/-- Condition 1: The cost of each deck is 8 dollars -/
def deck_cost : ℕ := 8

/-- Condition 2: Tom bought 3 decks -/
def tom_decks : ℕ := 3

/-- Condition 3: Tom's friend bought 5 decks -/
def friend_decks : ℕ := 5

/-- Total amount spent by Tom and his friend -/
def total_amount_spent : ℕ := (tom_decks * deck_cost) + (friend_decks * deck_cost)

/-- Proof statement: Prove that total amount spent is 64 -/
theorem total_spent_is_64 : total_amount_spent = 64 := by
  sorry

end total_spent_is_64_l47_47837


namespace number_of_cows_l47_47282

theorem number_of_cows (H : ℕ) (C : ℕ) (h1 : H = 6) (h2 : C / H = 7 / 2) : C = 21 :=
by
  sorry

end number_of_cows_l47_47282


namespace find_m_l47_47580

open Nat

theorem find_m (m : ℕ) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 60 := 
by
  sorry  -- Proof goes here

end find_m_l47_47580


namespace sin_315_eq_neg_sqrt2_div_2_l47_47499

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47499


namespace tan_add_pi_div_three_l47_47176

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l47_47176


namespace number_of_elements_in_set_l47_47405

theorem number_of_elements_in_set 
  (S : ℝ) (n : ℝ) 
  (h_avg : S / n = 6.8) 
  (a : ℝ) (h_a : a = 6) 
  (h_new_avg : (S + 2 * a) / n = 9.2) : 
  n = 5 := 
  sorry

end number_of_elements_in_set_l47_47405


namespace tan_330_eq_neg_sqrt3_div_3_l47_47996

theorem tan_330_eq_neg_sqrt3_div_3 :
  Real.tan (330 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_330_eq_neg_sqrt3_div_3_l47_47996


namespace find_m_l47_47593

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l47_47593


namespace find_smallest_nat_with_remainder_2_l47_47561

noncomputable def smallest_nat_with_remainder_2 : Nat :=
    let x := 26
    if x > 0 ∧ x ≡ 2 [MOD 3] 
                 ∧ x ≡ 2 [MOD 4] 
                 ∧ x ≡ 2 [MOD 6] 
                 ∧ x ≡ 2 [MOD 8] then x
    else 0

theorem find_smallest_nat_with_remainder_2 :
    ∃ x : Nat, x > 0 ∧ x ≡ 2 [MOD 3] 
                 ∧ x ≡ 2 [MOD 4] 
                 ∧ x ≡ 2 [MOD 6] 
                 ∧ x ≡ 2 [MOD 8] ∧ x = smallest_nat_with_remainder_2 :=
    sorry

end find_smallest_nat_with_remainder_2_l47_47561


namespace sin_315_eq_neg_sqrt2_div_2_l47_47532

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47532


namespace find_a10_l47_47045

theorem find_a10 (a : ℕ → ℚ) 
  (h1 : a 1 = 1/2)
  (h2 : ∀ n : ℕ, 0 < n → (1 / (a (n + 1) - 1)) = (1 / (a n - 1)) - 1) : 
  a 10 = 10/11 := 
sorry

end find_a10_l47_47045


namespace storage_space_calc_correct_l47_47706

noncomputable def storage_space_available 
    (second_floor_total : ℕ)
    (box_space : ℕ)
    (one_quarter_second_floor : ℕ)
    (first_floor_ratio : ℕ) 
    (second_floor_ratio : ℕ) : ℕ :=
  if (box_space = one_quarter_second_floor ∧
      first_floor_ratio = 2 ∧
      second_floor_ratio = 1) then
    let total_building_space := second_floor_total + (first_floor_ratio * second_floor_total)
    in total_building_space - box_space
  else 0

theorem storage_space_calc_correct :
  storage_space_available 20000 5000 5000 2 1 = 55000 :=
by sorry

end storage_space_calc_correct_l47_47706


namespace fraction_addition_l47_47446

theorem fraction_addition : (2 / 5) + (3 / 8) = 31 / 40 := 
by {
  sorry
}

end fraction_addition_l47_47446


namespace valentino_farm_birds_total_l47_47379

theorem valentino_farm_birds_total :
  let chickens := 200
  let ducks := 2 * chickens
  let turkeys := 3 * ducks
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof here
  sorry

end valentino_farm_birds_total_l47_47379


namespace remainder_division_Q_l47_47369

noncomputable def Q_rest : Polynomial ℝ := -(Polynomial.X : Polynomial ℝ) + 125

theorem remainder_division_Q (Q : Polynomial ℝ) :
  Q.eval 20 = 105 ∧ Q.eval 105 = 20 →
  ∃ R : Polynomial ℝ, Q = (Polynomial.X - 20) * (Polynomial.X - 105) * R + Q_rest :=
by sorry

end remainder_division_Q_l47_47369


namespace total_steps_five_days_l47_47918

def steps_monday : ℕ := 150 + 170
def steps_tuesday : ℕ := 140 + 170
def steps_wednesday : ℕ := 160 + 210 + 25
def steps_thursday : ℕ := 150 + 140 + 30 + 15
def steps_friday : ℕ := 180 + 200 + 20

theorem total_steps_five_days :
  steps_monday + steps_tuesday + steps_wednesday + steps_thursday + steps_friday = 1760 :=
by
  have h1 : steps_monday = 320 := rfl
  have h2 : steps_tuesday = 310 := rfl
  have h3 : steps_wednesday = 395 := rfl
  have h4 : steps_thursday = 335 := rfl
  have h5 : steps_friday = 400 := rfl
  show 320 + 310 + 395 + 335 + 400 = 1760
  sorry

end total_steps_five_days_l47_47918


namespace find_m_l47_47619

theorem find_m (m : ℝ) (h1 : ∀ x y : ℝ, (x ^ 2 + (y - 2) ^ 2 = 1) → (y = x / m ∨ y = -x / m)) (h2 : 0 < m) :
  m = (Real.sqrt 3) / 3 :=
by
  sorry

end find_m_l47_47619


namespace right_triangle_area_l47_47930

theorem right_triangle_area
  (hypotenuse : ℝ) (angle : ℝ) (hyp_eq : hypotenuse = 12) (angle_eq : angle = 30) :
  ∃ area : ℝ, area = 18 * Real.sqrt 3 :=
by
  have side1 := hypotenuse / 2  -- Shorter leg = hypotenuse / 2
  have side2 := side1 * Real.sqrt 3  -- Longer leg = shorter leg * sqrt 3
  let area := (side1 * side2) / 2  -- Area calculation
  use area
  sorry

end right_triangle_area_l47_47930


namespace distinct_x_intercepts_l47_47149

theorem distinct_x_intercepts : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, (x - 5) * (x ^ 2 + 3 * x + 2) = 0) ∧ s.card = 3 :=
by {
  sorry
}

end distinct_x_intercepts_l47_47149


namespace range_of_m_l47_47730

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / (x + 1) + 4 / y = 1) (m : ℝ) :
  x + y / 4 > m^2 - 5 * m - 3 ↔ -1 < m ∧ m < 6 := sorry

end range_of_m_l47_47730


namespace find_smaller_number_l47_47679

theorem find_smaller_number (x y : ℕ) (h₁ : y - x = 1365) (h₂ : y = 6 * x + 15) : x = 270 :=
sorry

end find_smaller_number_l47_47679


namespace sum_divisors_36_48_l47_47728

open Finset

noncomputable def sum_common_divisors (a b : ℕ) : ℕ :=
  let divisors_a := (range (a + 1)).filter (λ x => a % x = 0)
  let divisors_b := (range (b + 1)).filter (λ x => b % x = 0)
  let common_divisors := divisors_a ∩ divisors_b
  common_divisors.sum id

theorem sum_divisors_36_48 : sum_common_divisors 36 48 = 28 := by
  sorry

end sum_divisors_36_48_l47_47728


namespace carlos_books_in_june_l47_47709

def books_in_july : ℕ := 28
def books_in_august : ℕ := 30
def goal_books : ℕ := 100

theorem carlos_books_in_june :
  let books_in_july_august := books_in_july + books_in_august
  let books_needed_june := goal_books - books_in_july_august
  books_needed_june = 42 := 
by
  sorry

end carlos_books_in_june_l47_47709


namespace integer_solutions_l47_47721

theorem integer_solutions :
  ∃ (a b c : ℤ), a + b + c = 24 ∧ a^2 + b^2 + c^2 = 210 ∧ a * b * c = 440 ∧
    (a = 5 ∧ b = 8 ∧ c = 11) ∨ (a = 5 ∧ b = 11 ∧ c = 8) ∨ 
    (a = 8 ∧ b = 5 ∧ c = 11) ∨ (a = 8 ∧ b = 11 ∧ c = 5) ∨
    (a = 11 ∧ b = 5 ∧ c = 8) ∨ (a = 11 ∧ b = 8 ∧ c = 5) :=
sorry

end integer_solutions_l47_47721


namespace change_given_back_l47_47217

theorem change_given_back
  (p s t a : ℕ)
  (hp : p = 140)
  (hs : s = 43)
  (ht : t = 15)
  (ha : a = 200) :
  (a - (p + s + t)) = 2 :=
by
  sorry

end change_given_back_l47_47217


namespace find_number_l47_47943

theorem find_number (x: ℝ) (h: (6 * x) / 2 - 5 = 25) : x = 10 :=
by
  sorry

end find_number_l47_47943


namespace original_area_of_triangle_l47_47235

theorem original_area_of_triangle (A : ℝ) (h1 : 4 * A * 16 = 64) : A = 4 :=
by
  sorry

end original_area_of_triangle_l47_47235


namespace sin_315_eq_neg_sqrt2_over_2_l47_47547

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l47_47547


namespace boys_bound_l47_47255

open Nat

noncomputable def num_students := 1650
noncomputable def num_rows := 22
noncomputable def num_cols := 75
noncomputable def max_pairs_same_sex := 11

-- Assume we have a function that gives the number of boys.
axiom number_of_boys : ℕ
axiom col_pairs_property : ∀ (c1 c2 : ℕ), ∀ (r : ℕ), c1 ≠ c2 → r ≤ num_rows → 
  (number_of_boys ≤ max_pairs_same_sex)

theorem boys_bound : number_of_boys ≤ 920 :=
sorry

end boys_bound_l47_47255


namespace find_r_l47_47773

theorem find_r (k r : ℝ) : 
  5 = k * 3^r ∧ 45 = k * 9^r → r = 2 :=
by 
  sorry

end find_r_l47_47773


namespace find_g3_l47_47791

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x^2 + b * x^3 + c * x + d

theorem find_g3 (a b c d : ℝ) (h : g (-3) a b c d = 2) : g 3 a b c d = 0 := 
by 
  sorry

end find_g3_l47_47791


namespace polygon_six_sides_l47_47193

theorem polygon_six_sides (n : ℕ) (h1 : (n - 2) * 180 = 2 * 360) : n = 6 := by
  sorry

end polygon_six_sides_l47_47193


namespace polynomial_arithmetic_sequence_roots_l47_47560

theorem polynomial_arithmetic_sequence_roots (p q : ℝ) (h : ∃ a b c d : ℝ, 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a + 3*(b - a) = b ∧ b + 3*(c - b) = c ∧ c + 3*(d - c) = d ∧ 
  (a^4 + p * a^2 + q = 0) ∧ (b^4 + p * b^2 + q = 0) ∧ 
  (c^4 + p * c^2 + q = 0) ∧ (d^4 + p * d^2 + q = 0)) :
  p ≤ 0 ∧ q = 0.09 * p^2 := 
sorry

end polynomial_arithmetic_sequence_roots_l47_47560


namespace total_spokes_is_60_l47_47281

def num_spokes_front : ℕ := 20
def num_spokes_back : ℕ := 2 * num_spokes_front
def total_spokes : ℕ := num_spokes_front + num_spokes_back

theorem total_spokes_is_60 : total_spokes = 60 :=
by
  sorry

end total_spokes_is_60_l47_47281


namespace range_of_k_l47_47190

theorem range_of_k (k : ℝ) :
  (∃ (x : ℝ), 2 < x ∧ x < 3 ∧ x^2 + (1 - k) * x - 2 * (k + 1) = 0) →
  1 < k ∧ k < 2 :=
by
  sorry

end range_of_k_l47_47190


namespace binom_7_4_plus_5_l47_47117

theorem binom_7_4_plus_5 : ((Nat.choose 7 4) + 5) = 40 := by
  sorry

end binom_7_4_plus_5_l47_47117


namespace value_of_expression_l47_47332

variables (u v w : ℝ)

theorem value_of_expression (h1 : u = 3 * v) (h2 : w = 5 * u) : 2 * v + u + w = 20 * v :=
by sorry

end value_of_expression_l47_47332


namespace total_stickers_l47_47385

theorem total_stickers (r s t : ℕ) (h1 : r = 30) (h2 : s = 3 * r) (h3 : t = s + 20) : r + s + t = 230 :=
by sorry

end total_stickers_l47_47385


namespace nesbitt_inequality_l47_47009

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem nesbitt_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := 
by
  sorry

end nesbitt_inequality_l47_47009


namespace cyclists_meeting_time_l47_47897

theorem cyclists_meeting_time :
  ∃ t : ℕ, t = Nat.lcm 7 (Nat.lcm 12 9) ∧ t = 252 :=
by
  use 252
  have h1 : Nat.lcm 7 (Nat.lcm 12 9) = 252 := sorry
  exact ⟨rfl, h1⟩

end cyclists_meeting_time_l47_47897


namespace fraction_expression_l47_47867

theorem fraction_expression :
  ((3 / 7) + (5 / 8)) / ((5 / 12) + (2 / 9)) = (531 / 322) :=
by
  sorry

end fraction_expression_l47_47867


namespace kathleen_remaining_money_l47_47785

-- Define the conditions
def saved_june := 21
def saved_july := 46
def saved_august := 45
def spent_school_supplies := 12
def spent_clothes := 54
def aunt_gift_threshold := 125
def aunt_gift := 25

-- Prove that Kathleen has the correct remaining amount of money
theorem kathleen_remaining_money : 
    (saved_june + saved_july + saved_august) - 
    (spent_school_supplies + spent_clothes) = 46 := 
by
  sorry

end kathleen_remaining_money_l47_47785


namespace ages_correct_l47_47917

-- Let A be Anya's age and P be Petya's age
def anya_age : ℕ := 4
def petya_age : ℕ := 12

-- The conditions
def condition1 (A P : ℕ) : Prop := P = 3 * A
def condition2 (A P : ℕ) : Prop := P - A = 8

-- The statement to be proven
theorem ages_correct : condition1 anya_age petya_age ∧ condition2 anya_age petya_age :=
by
  unfold condition1 condition2 anya_age petya_age -- Reveal the definitions
  have h1 : petya_age = 3 * anya_age := by
    sorry
  have h2 : petya_age - anya_age = 8 := by
    sorry
  exact ⟨h1, h2⟩ -- Combine both conditions into a single conjunction

end ages_correct_l47_47917


namespace multiply_powers_same_base_l47_47418

theorem multiply_powers_same_base (a : ℝ) : a^4 * a^2 = a^6 := by
  sorry

end multiply_powers_same_base_l47_47418


namespace probability_all_successful_pairs_expected_successful_pairs_l47_47411

-- Lean statement for part (a)
theorem probability_all_successful_pairs (n : ℕ) :
  let prob_all_successful := (2 ^ n * n.factorial) / (2 * n).factorial
  in prob_all_successful = (1 / (2 ^ n)) * (n.factorial / (2 * n).factorial)
:= by sorry

-- Lean statement for part (b)
theorem expected_successful_pairs (n : ℕ) :
  let expected_value := n / (2 * n - 1)
  in expected_value > 0.5
:= by sorry

end probability_all_successful_pairs_expected_successful_pairs_l47_47411


namespace find_x_l47_47306

theorem find_x :
  ∃ n : ℕ, odd n ∧ (6^n + 1).natPrimeFactors.length = 3 ∧ 11 ∈ (6^n + 1).natPrimeFactors ∧ 6^n + 1 = 7777 :=
by sorry

end find_x_l47_47306


namespace not_perfect_square_l47_47223

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, (3^n + 2 * 17^n) = k^2 :=
by
  sorry

end not_perfect_square_l47_47223


namespace simplify_polynomial_l47_47810

theorem simplify_polynomial :
  (3 * y - 2) * (6 * y ^ 12 + 3 * y ^ 11 + 6 * y ^ 10 + 3 * y ^ 9) =
  18 * y ^ 13 - 3 * y ^ 12 + 12 * y ^ 11 - 3 * y ^ 10 - 6 * y ^ 9 :=
by
  sorry

end simplify_polynomial_l47_47810


namespace f_is_periodic_l47_47570

-- Define the conditions for the function f
def f (x : ℝ) : ℝ := sorry
axiom f_defined : ∀ x : ℝ, f x ≠ 0
axiom f_property : ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f (x - a) = 1 / f x

-- Formal problem statement to be proven
theorem f_is_periodic : ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f x = f (x + 2 * a) :=
by {
  sorry
}

end f_is_periodic_l47_47570


namespace sin_315_eq_neg_sqrt2_div_2_l47_47500

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47500


namespace find_A_and_height_l47_47621

noncomputable def triangle_properties (a b : ℝ) (B : ℝ) (cos_B : ℝ) (h : ℝ) :=
  a = 7 ∧ b = 8 ∧ cos_B = -1 / 7 ∧ 
  h = (a : ℝ) * (Real.sqrt (1 - (cos_B)^2)) * (1 : ℝ) / b / 2

theorem find_A_and_height : 
  ∀ (a b : ℝ) (B : ℝ) (cos_B : ℝ) (h : ℝ), 
  triangle_properties a b B cos_B h → 
  ∃ A h1, A = Real.pi / 3 ∧ h1 = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end find_A_and_height_l47_47621


namespace circumcircle_radius_ABC_l47_47244

noncomputable def radius_of_circumcircle_of_ABC
  (R1 R2 : ℝ)
  (h_sum : R1 + R2 = 11)
  (d : ℝ := 5 * Real.sqrt 17)
  (h_dist : 5 * Real.sqrt 17 = d)
  (A : ℝ := 8)
  (h_tangent : ∀ {R3 : ℝ}, R3 = A → R3 = 8)
  : ℝ :=
2 * Real.sqrt 19

theorem circumcircle_radius_ABC (R1 R2 d A : ℝ)
  (h_sum : R1 + R2 = 11)
  (h_dist : d = 5 * Real.sqrt 17)
  (h_tangent : A = 8)
  : radius_of_circumcircle_of_ABC R1 R2 h_sum d h_dist A h_tangent = 2 * Real.sqrt 19 := by
  sorry

end circumcircle_radius_ABC_l47_47244


namespace tan_add_pi_over_3_l47_47156

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l47_47156


namespace gift_wrapping_combinations_l47_47687

theorem gift_wrapping_combinations :
  (10 * 5 * 6 * 2 = 600) :=
by
  sorry

end gift_wrapping_combinations_l47_47687


namespace ratio_of_pages_given_l47_47208

variable (Lana_initial_pages : ℕ) (Duane_initial_pages : ℕ) (Lana_final_pages : ℕ)

theorem ratio_of_pages_given
  (h1 : Lana_initial_pages = 8)
  (h2 : Duane_initial_pages = 42)
  (h3 : Lana_final_pages = 29) :
  (Lana_final_pages - Lana_initial_pages) / Duane_initial_pages = 1 / 2 :=
  by
  -- Placeholder for the proof
  sorry

end ratio_of_pages_given_l47_47208


namespace max_min_value_sum_l47_47604

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x) * Real.sin (x - 2) + x + 1

theorem max_min_value_sum (M m : ℝ) 
  (hM : ∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≤ M)
  (hm : ∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≥ m)
  (hM_max : ∃ x ∈ Set.Icc (-1 : ℝ) 5, f x = M)
  (hm_min : ∃ x ∈ Set.Icc (-1 : ℝ) 5, f x = m)
  : M + m = 6 :=
sorry

end max_min_value_sum_l47_47604


namespace reporters_cover_local_politics_l47_47266

-- Definitions of percentages and total reporters
def total_reporters : ℕ := 100
def politics_coverage_percent : ℕ := 20 -- Derived from 100 - 80
def politics_reporters : ℕ := (politics_coverage_percent * total_reporters) / 100
def not_local_politics_percent : ℕ := 40
def local_politics_percent : ℕ := 60 -- Derived from 100 - 40
def local_politics_reporters : ℕ := (local_politics_percent * politics_reporters) / 100

theorem reporters_cover_local_politics :
  (local_politics_reporters * 100) / total_reporters = 12 :=
by
  exact sorry

end reporters_cover_local_politics_l47_47266


namespace total_cost_is_660_l47_47003

def total_material_cost : ℝ :=
  let velvet_area := (12 * 4) * 3
  let velvet_cost := velvet_area * 3
  let silk_cost := 2 * 6
  let lace_cost := 5 * 2 * 10
  let bodice_cost := silk_cost + lace_cost
  let satin_area := 2.5 * 1.5
  let satin_cost := satin_area * 4
  let leather_area := 1 * 1.5 * 2
  let leather_cost := leather_area * 5
  let wool_area := 5 * 2
  let wool_cost := wool_area * 8
  let ribbon_cost := 3 * 2
  velvet_cost + bodice_cost + satin_cost + leather_cost + wool_cost + ribbon_cost

theorem total_cost_is_660 : total_material_cost = 660 := by
  sorry

end total_cost_is_660_l47_47003


namespace prove_l47_47793

variable (m n : ℕ)
variable (a b α β : ℝ)
variable hα : α = 3 / 4
variable hβ : β = 19 / 20

def height_ratio (m n : ℕ) (a b α β : ℝ) (hα : α = 3 / 4) (hβ : β = 19 / 20) 
    (h1 : a = α * b) (h2 : a = β * (a * m + b * n) / (m + n)) : (ℝ) :=
  m / n

theorem prove\_height\_ratio (m n : ℕ) (a b α β : ℝ) (hα : α = 3 / 4) (hβ : β = 19 / 20) 
    (h1 : a = α * b) (h2 : a = β * (a * m + b * n) / (m + n)) : 
  height_ratio m n a b α β hα hβ h1 h2 = 8 / 9 :=
by
  -- Proof omitted
  sorry

end prove_l47_47793


namespace total_wait_time_l47_47075

def customs_wait : ℕ := 20
def quarantine_days : ℕ := 14
def hours_per_day : ℕ := 24

theorem total_wait_time :
  customs_wait + quarantine_days * hours_per_day = 356 := 
by
  sorry

end total_wait_time_l47_47075


namespace tickets_used_to_buy_toys_l47_47703

-- Definitions for the conditions
def initial_tickets : ℕ := 13
def leftover_tickets : ℕ := 7

-- The theorem we want to prove
theorem tickets_used_to_buy_toys : initial_tickets - leftover_tickets = 6 :=
by
  sorry

end tickets_used_to_buy_toys_l47_47703


namespace total_birds_in_marsh_l47_47055

-- Define the number of geese and ducks as constants.
def geese : Nat := 58
def ducks : Nat := 37

-- The theorem that we need to prove.
theorem total_birds_in_marsh : geese + ducks = 95 :=
by
  -- Here, we add the sorry keyword to skip the proof part.
  sorry

end total_birds_in_marsh_l47_47055


namespace penguin_fish_consumption_l47_47429

-- Definitions based on the conditions
def initial_penguins : ℕ := 158
def total_fish_per_day : ℕ := 237
def fish_per_penguin_per_day : ℚ := 1.5

-- Lean statement for the conditional problem
theorem penguin_fish_consumption
  (P : ℕ)
  (h_initial_penguins : P = initial_penguins)
  (h_total_fish_per_day : total_fish_per_day = 237)
  (h_current_penguins : P * 2 * 3 + 129 = 1077)
  : total_fish_per_day / P = fish_per_penguin_per_day := by
  sorry

end penguin_fish_consumption_l47_47429


namespace exp_7pi_over_2_eq_i_l47_47554

theorem exp_7pi_over_2_eq_i : Complex.exp (7 * Real.pi * Complex.I / 2) = Complex.I :=
by
  sorry

end exp_7pi_over_2_eq_i_l47_47554


namespace complement_union_eq_l47_47743

open Set

-- Definition of sets U, A, and B
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- Statement of the problem
theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 3} :=
by sorry

end complement_union_eq_l47_47743


namespace find_y_l47_47901

def G (a b c d : ℕ) : ℕ := a ^ b + c * d

theorem find_y (y : ℕ) : G 3 y 5 10 = 350 ↔ y = 5 := by
  sorry

end find_y_l47_47901


namespace greatest_power_of_2_factor_l47_47949

theorem greatest_power_of_2_factor
    : ∃ k : ℕ, (2^k) ∣ (10^1503 - 4^752) ∧ ∀ m : ℕ, (2^(m+1)) ∣ (10^1503 - 4^752) → m < k :=
by
    sorry

end greatest_power_of_2_factor_l47_47949


namespace series_sum_eq_1_over_200_l47_47549

-- Definition of the nth term of the series
def nth_term (n : ℕ) : ℝ :=
  (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

-- Definition of the series sum
def series_sum : ℝ :=
  ∑' n, nth_term n

-- Final proposition to prove the sum of the series is 1/200
theorem series_sum_eq_1_over_200 : series_sum = 1 / 200 :=
  sorry

end series_sum_eq_1_over_200_l47_47549


namespace ball_arrangement_l47_47666

theorem ball_arrangement :
  (Nat.factorial 9) / ((Nat.factorial 2) * (Nat.factorial 3) * (Nat.factorial 4)) = 1260 := 
by
  sorry

end ball_arrangement_l47_47666


namespace cos_alpha_value_l47_47151

open Real

theorem cos_alpha_value (α : ℝ) : 
  (sin (α - (π / 3)) = 1 / 5) ∧ (0 < α) ∧ (α < π / 2) → 
  (cos α = (2 * sqrt 6 - sqrt 3) / 10) := 
by
  intros h
  sorry

end cos_alpha_value_l47_47151


namespace find_point_B_l47_47319

-- Definition of Point
structure Point where
  x : ℝ
  y : ℝ

-- Definitions of conditions
def A : Point := ⟨1, 2⟩
def d : ℝ := 3
def AB_parallel_x (A B : Point) : Prop := A.y = B.y

theorem find_point_B (B : Point) (h_parallel : AB_parallel_x A B) (h_dist : abs (B.x - A.x) = d) :
  (B = ⟨4, 2⟩) ∨ (B = ⟨-2, 2⟩) :=
by
  sorry

end find_point_B_l47_47319


namespace blithe_initial_toys_l47_47113

-- Define the conditions as given in the problem
def lost_toys : ℤ := 6
def found_toys : ℤ := 9
def final_toys : ℤ := 43

-- Define the problem statement to prove the initial number of toys
theorem blithe_initial_toys (T : ℤ) (h : T - lost_toys + found_toys = final_toys) : T = 40 :=
sorry

end blithe_initial_toys_l47_47113


namespace number_of_nonnegative_solutions_l47_47898

theorem number_of_nonnegative_solutions : ∃ (count : ℕ), count = 1 ∧ ∀ x : ℝ, x^2 + 9 * x = 0 → x ≥ 0 → x = 0 := by
  sorry

end number_of_nonnegative_solutions_l47_47898


namespace possible_values_a_l47_47886

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 - a * x - 7 else a / x

theorem possible_values_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → -2 * x - a ≥ 0) ∧
  (∀ x : ℝ, x > 1 → -a / (x^2) ≥ 0) ∧
  (-8 - a ≤ a) →
  a = -2 ∨ a = -3 ∨ a = -4 :=
sorry

end possible_values_a_l47_47886


namespace tan_add_pi_over_3_l47_47162

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l47_47162


namespace kaleb_gave_boxes_l47_47635

theorem kaleb_gave_boxes (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) (given_boxes : ℕ)
  (h1 : total_boxes = 14) 
  (h2 : pieces_per_box = 6) 
  (h3 : pieces_left = 54) :
  given_boxes = 5 :=
by
  -- Add your proof here
  sorry

end kaleb_gave_boxes_l47_47635


namespace train_cross_first_platform_in_15_seconds_l47_47104

noncomputable def length_of_train : ℝ := 100
noncomputable def length_of_second_platform : ℝ := 500
noncomputable def time_to_cross_second_platform : ℝ := 20
noncomputable def length_of_first_platform : ℝ := 350
noncomputable def speed_of_train := (length_of_train + length_of_second_platform) / time_to_cross_second_platform
noncomputable def time_to_cross_first_platform := (length_of_train + length_of_first_platform) / speed_of_train

theorem train_cross_first_platform_in_15_seconds : time_to_cross_first_platform = 15 := by
  sorry

end train_cross_first_platform_in_15_seconds_l47_47104


namespace eccentricity_is_sqrt2_div2_l47_47312

noncomputable def eccentricity_square_ellipse (a b c : ℝ) : ℝ :=
  c / (Real.sqrt (b ^ 2 + c ^ 2))

theorem eccentricity_is_sqrt2_div2 (a b c : ℝ) (h : b = c) : 
  eccentricity_square_ellipse a b c = Real.sqrt 2 / 2 :=
by
  -- The proof will show that the eccentricity calculation is correct given the conditions.
  sorry

end eccentricity_is_sqrt2_div2_l47_47312


namespace composite_divides_factorial_l47_47731

-- Define the factorial of a number
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Statement of the problem
theorem composite_divides_factorial (m : ℕ) (hm : m ≠ 4) (hcomposite : ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = m) :
  m ∣ factorial (m - 1) :=
by
  sorry

end composite_divides_factorial_l47_47731


namespace intersection_M_N_l47_47608

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l47_47608


namespace determine_n_l47_47007

theorem determine_n (n : ℕ) (hn : 0 < n) :
  (∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 3 * x + 2 * y + z = n) → (n = 15 ∨ n = 16) :=
  by sorry

end determine_n_l47_47007


namespace seeds_per_bed_l47_47019

theorem seeds_per_bed (total_seeds : ℕ) (flower_beds : ℕ) (h1 : total_seeds = 60) (h2 : flower_beds = 6) : total_seeds / flower_beds = 10 := by
  sorry

end seeds_per_bed_l47_47019


namespace factor_expression_l47_47929

variable {R : Type*} [CommRing R]

theorem factor_expression (a b c : R) :
    a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) =
    (a - b) * (b - c) * (c - a) * ((a + b) * a^2 * b^2 + (b + c) * b^2 * c^2 + (a + c) * c^2 * a) :=
sorry

end factor_expression_l47_47929


namespace find_x_l47_47304

-- Define x as a function of n, where n is an odd natural number
def x (n : ℕ) (h_odd : n % 2 = 1) : ℕ :=
  6^n + 1

-- Define the main theorem
theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_prime_div : ∀ p, p.Prime → p ∣ x n h_odd → (p = 11 ∨ p = 7 ∨ p = 101)) : x 1 (by norm_num) = 7777 :=
  sorry

end find_x_l47_47304


namespace sin_315_degree_is_neg_sqrt_2_div_2_l47_47496

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l47_47496


namespace complex_numbers_not_comparable_l47_47227

-- Definitions based on conditions
def is_real (z : ℂ) : Prop := ∃ r : ℝ, z = r
def is_not_entirely_real (z : ℂ) : Prop := ¬ is_real z

-- Proof problem statement
theorem complex_numbers_not_comparable (z1 z2 : ℂ) (h1 : is_not_entirely_real z1) (h2 : is_not_entirely_real z2) : 
  ¬ (z1.re = z2.re ∧ z1.im = z2.im) :=
sorry

end complex_numbers_not_comparable_l47_47227


namespace diametrically_opposite_number_is_84_l47_47689

-- Define the circular arrangement problem
def equal_arcs (n : ℕ) (k : ℕ → ℕ) (m : ℕ) : Prop :=
  ∀ i, i < n → ∃ j, j < n ∧ j ≠ i ∧ k j = k i + n / 2 ∨ k j = k i - n / 2

-- Define the specific number condition
def exactly_one_to_100 (k : ℕ) : Prop := k = 83

theorem diametrically_opposite_number_is_84 (n : ℕ) (k : ℕ → ℕ) (m : ℕ) :
  n = 100 →
  equal_arcs n k m →
  exactly_one_to_100 (k 83) →
  k 83 + 1 = 84 :=
by {
  intros,
  sorry
}

end diametrically_opposite_number_is_84_l47_47689


namespace find_m_l47_47583

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l47_47583


namespace camden_dogs_total_legs_l47_47285

theorem camden_dogs_total_legs :
  ∀ (Justin_dogs : ℕ), Justin_dogs = 14 →
  let Rico_dogs := Justin_dogs + 10 in
  let Camden_dogs := (3 * Rico_dogs) / 4 in
  let total_legs := Camden_dogs * 4 in
  total_legs = 72 :=
by
  intros Justin_dogs hJustin_dogs
  let Rico_dogs := Justin_dogs + 10
  let Camden_dogs := (3 * Rico_dogs) / 4
  let total_legs := Camden_dogs * 4
  have hJustin_dogs_14 : Justin_dogs = 14 := hJustin_dogs
  rw [hJustin_dogs_14] at *
  have hRico_dogs : Rico_dogs = 24 := by norm_num
  rw [hRico_dogs] at *
  have hCamden_dogs : Camden_dogs = 18 := by norm_num
  rw [hCamden_dogs] at *
  have hTotal_legs : total_legs = 72 := by norm_num
  exact hTotal_legs

end camden_dogs_total_legs_l47_47285


namespace min_frac_sum_l47_47567

theorem min_frac_sum (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_eq : 2 * a + b = 1) : 
  (3 / b + 2 / a) = 7 + 4 * Real.sqrt 3 := 
sorry

end min_frac_sum_l47_47567


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47510

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47510


namespace interior_diagonals_sum_l47_47972

theorem interior_diagonals_sum (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 52)
  (h2 : 2 * (a * b + b * c + c * a) = 118) :
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 51 := 
by
  sorry

end interior_diagonals_sum_l47_47972


namespace book_pages_l47_47613

theorem book_pages (x : ℕ) : 
  (x - (1/5 * x + 12)) - (1/4 * (x - (1/5 * x + 12)) + 15) - (1/3 * ((x - (1/5 * x + 12)) - (1/4 * (x - (1/5 * x + 12)) + 15)) + 18) = 62 →
  x = 240 :=
by
  -- This is where the proof would go, but it's omitted for this task.
  sorry

end book_pages_l47_47613


namespace permutation_by_transpositions_l47_47221

-- Formalizing the conditions in Lean
section permutations
  variable {n : ℕ}

  -- Define permutations
  def is_permutation (σ : Fin n → Fin n) : Prop :=
    ∃ σ_inv : Fin n → Fin n, 
      (∀ i, σ (σ_inv i) = i) ∧ 
      (∀ i, σ_inv (σ i) = i)

  -- Define transposition
  def transposition (σ : Fin n → Fin n) (i j : Fin n) : Fin n → Fin n :=
    fun x => if x = i then j else if x = j then i else σ x

  -- Main theorem stating that any permutation can be obtained through a series of transpositions
  theorem permutation_by_transpositions (σ : Fin n → Fin n) (h : is_permutation σ) :
    ∃ τ : ℕ → (Fin n → Fin n),
      (∀ i, is_permutation (τ i)) ∧
      (∀ m, ∃ k, τ m = transposition (τ (m - 1)) (⟨ k, sorry ⟩) (σ (⟨ k, sorry⟩))) ∧
      (∃ m, τ m = σ) :=
  sorry
end permutations

end permutation_by_transpositions_l47_47221


namespace number_of_students_run_red_light_l47_47670

theorem number_of_students_run_red_light :
  let total_students := 300
  let yes_responses := 90
  let odd_id_students := 75
  let coin_probability := 1/2
  -- Calculate using the conditions:
  total_students / 2 - odd_id_students / 2 * coin_probability + total_students / 2 * coin_probability = 30 :=
by
  sorry

end number_of_students_run_red_light_l47_47670


namespace max_happy_monkeys_l47_47099

-- Definitions for given problem
def pears := 20
def bananas := 30
def peaches := 40
def mandarins := 50
def fruits (x y : Nat) := x + y

-- The theorem to prove
theorem max_happy_monkeys : 
  ∃ (m : Nat), m = (pears + bananas + peaches) / 2 ∧ m ≤ mandarins :=
by
  sorry

end max_happy_monkeys_l47_47099


namespace no_square_with_odd_last_two_digits_l47_47116

def last_two_digits_odd (n : ℤ) : Prop :=
  (n % 10) % 2 = 1 ∧ ((n / 10) % 10) % 2 = 1

theorem no_square_with_odd_last_two_digits (n : ℤ) (k : ℤ) :
  (k^2 = n) → last_two_digits_odd n → False :=
by
  -- A placeholder for the proof
  sorry

end no_square_with_odd_last_two_digits_l47_47116


namespace train_length_proof_l47_47676

noncomputable def length_of_train : ℝ := 450.09

theorem train_length_proof
  (speed_kmh : ℝ := 60)
  (time_s : ℝ := 27) :
  (speed_kmh * (5 / 18) * time_s = length_of_train) :=
by
  sorry

end train_length_proof_l47_47676


namespace midpoint_sum_four_times_l47_47245

theorem midpoint_sum_four_times (x1 y1 x2 y2 : ℝ) (h1 : x1 = 8) (h2 : y1 = -4) (h3 : x2 = -2) (h4 : y2 = 10) :
  4 * ((x1 + x2) / 2 + (y1 + y2) / 2) = 24 :=
by
  rw [h1, h2, h3, h4]
  -- simplifying to get the desired result
  sorry

end midpoint_sum_four_times_l47_47245


namespace double_pythagorean_triple_l47_47079

theorem double_pythagorean_triple (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  (2*a)^2 + (2*b)^2 = (2*c)^2 :=
by
  sorry

end double_pythagorean_triple_l47_47079


namespace partition_Z_l47_47876

-- Define the set Z as points on the line whose distance from a given point is an integer.
def Z (p : ℝ) : Set ℝ := {x | ∃ n : ℤ, x = p + n}

-- Define a three-element subset H of Z.
def H (p : ℝ) (a b : ℤ) (h : 0 < a ∧ a ≤ b) : Set ℝ := {p, p + a, p + (a + b)}

-- The main theorem statement.
theorem partition_Z (p : ℝ) (a b : ℤ) (h : 0 < a ∧ a ≤ b) : 
  ∃ (S : Set ℝ) (T : Set ℝ → Set ℝ) (Hs : ∀ s ∈ S, ∃ (x : ℝ), T s = {x, x + a, x + (a + b)}), 
  ∀ (x : ℝ), x ∈ Z p → ∃ (t ∈ S), x ∈ T t :=
sorry

end partition_Z_l47_47876


namespace inequality_solution_set_l47_47400

theorem inequality_solution_set (x : ℝ) : (x - 4) * (x + 1) > 0 ↔ x > 4 ∨ x < -1 :=
by sorry

end inequality_solution_set_l47_47400


namespace ratio_m_over_n_l47_47792

theorem ratio_m_over_n : 
  ∀ (m n : ℕ) (a b : ℝ),
  let α := (3 : ℝ) / 4
  let β := (19 : ℝ) / 20
  (a = α * b) →
  (a = β * (a * m + b * n) / (m + n)) →
  (n ≠ 0) →
  m / n = 8 / 9 :=
by
  intros m n a b α β hα hβ hn
  sorry

end ratio_m_over_n_l47_47792


namespace smallest_k_l47_47788

theorem smallest_k (p : ℕ) (hp : p = 997) : 
  ∃ k : ℕ, (p^2 - k) % 10 = 0 ∧ k = 9 :=
by
  sorry

end smallest_k_l47_47788


namespace find_a_range_l47_47605

def f (a x : ℝ) : ℝ := x^2 + a * x

theorem find_a_range (a : ℝ) :
  (∃ x : ℝ, f a (f a x) ≤ f a x) → (a ≤ 0 ∨ a ≥ 2) :=
by
  sorry

end find_a_range_l47_47605


namespace electricity_cost_one_kilometer_minimum_electricity_kilometers_l47_47964

-- Part 1: Cost of traveling one kilometer using electricity only
theorem electricity_cost_one_kilometer (x : ℝ) (fuel_cost : ℝ) (electricity_cost : ℝ) 
  (total_fuel_cost : ℝ) (total_electricity_cost : ℝ) 
  (fuel_per_km_more_than_electricity : ℝ) (distance_fuel : ℝ) (distance_electricity : ℝ)
  (h1 : total_fuel_cost = distance_fuel * fuel_cost)
  (h2 : total_electricity_cost = distance_electricity * electricity_cost)
  (h3 : fuel_per_km_more_than_electricity = 0.5)
  (h4 : fuel_cost = electricity_cost + fuel_per_km_more_than_electricity)
  (h5 : distance_fuel = 76 / (electricity_cost + 0.5))
  (h6 : distance_electricity = 26 / electricity_cost) : 
  x = 0.26 :=
sorry

-- Part 2: Minimum kilometers traveled using electricity
theorem minimum_electricity_kilometers (total_trip_cost : ℝ) (electricity_per_km : ℝ) 
  (hybrid_total_km : ℝ) (max_total_cost : ℝ) (fuel_per_km : ℝ) (y : ℝ)
  (h1 : electricity_per_km = 0.26)
  (h2 : fuel_per_km = 0.26 + 0.5)
  (h3 : hybrid_total_km = 100)
  (h4 : max_total_cost = 39)
  (h5 : total_trip_cost = electricity_per_km * y + (hybrid_total_km - y) * fuel_per_km)
  (h6 : total_trip_cost ≤ max_total_cost) :
  y ≥ 74 :=
sorry

end electricity_cost_one_kilometer_minimum_electricity_kilometers_l47_47964


namespace nigella_sold_3_houses_l47_47016

noncomputable def houseA_cost : ℝ := 60000
noncomputable def houseB_cost : ℝ := 3 * houseA_cost
noncomputable def houseC_cost : ℝ := 2 * houseA_cost - 110000
noncomputable def commission_rate : ℝ := 0.02

noncomputable def houseA_commission : ℝ := houseA_cost * commission_rate
noncomputable def houseB_commission : ℝ := houseB_cost * commission_rate
noncomputable def houseC_commission : ℝ := houseC_cost * commission_rate

noncomputable def total_commission : ℝ := houseA_commission + houseB_commission + houseC_commission
noncomputable def base_salary : ℝ := 3000
noncomputable def total_earnings : ℝ := base_salary + total_commission

theorem nigella_sold_3_houses 
  (H1 : total_earnings = 8000) 
  (H2 : houseA_cost = 60000) 
  (H3 : houseB_cost = 3 * houseA_cost) 
  (H4 : houseC_cost = 2 * houseA_cost - 110000) 
  (H5 : commission_rate = 0.02) :
  3 = 3 :=
by 
  -- Proof not required
  sorry

end nigella_sold_3_houses_l47_47016


namespace scrap_cookie_radius_l47_47558

theorem scrap_cookie_radius (r: ℝ) (r_cookies: ℝ) (A_scrap: ℝ) (r_large: ℝ) (A_large: ℝ) (A_total_small: ℝ):
  r_cookies = 1.5 ∧
  r_large = r_cookies + 2 * r_cookies ∧
  A_large = π * r_large^2 ∧
  A_total_small = 8 * (π * r_cookies^2) ∧
  A_scrap = A_large - A_total_small ∧
  A_scrap = π * r^2
  → r = r_cookies
  :=
by
  intro h
  rcases h with ⟨hcookies, hrlarge, halarge, hatotalsmall, hascrap, hpi⟩
  sorry

end scrap_cookie_radius_l47_47558


namespace five_to_one_ratio_to_eleven_is_fifty_five_l47_47247

theorem five_to_one_ratio_to_eleven_is_fifty_five (y : ℚ) (h : 5 / 1 = y / 11) : y = 55 :=
by
  sorry

end five_to_one_ratio_to_eleven_is_fifty_five_l47_47247


namespace g_odd_l47_47627

def g (x : ℝ) : ℝ := x^3 - 2*x

theorem g_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_odd_l47_47627


namespace sin_315_degree_is_neg_sqrt_2_div_2_l47_47498

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l47_47498


namespace cyclists_meet_time_l47_47680

/-- 
  Two cyclists start on a circular track from a given point but in opposite directions with speeds of 7 m/s and 8 m/s.
  The circumference of the circle is 180 meters.
  After what time will they meet at the starting point? 
-/
theorem cyclists_meet_time :
  let speed1 := 7 -- m/s
  let speed2 := 8 -- m/s
  let circumference := 180 -- meters
  (circumference / (speed1 + speed2) = 12) :=
by
  let speed1 := 7 -- m/s
  let speed2 := 8 -- m/s
  let circumference := 180 -- meters
  sorry

end cyclists_meet_time_l47_47680


namespace tan_add_pi_over_3_l47_47154

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l47_47154


namespace num_orders_javier_constraint_l47_47631

noncomputable def num_valid_orders : ℕ :=
  Nat.factorial 5 / 2

theorem num_orders_javier_constraint : num_valid_orders = 60 := 
by
  sorry

end num_orders_javier_constraint_l47_47631


namespace solution_interval_log_eq_l47_47399

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2) + x - 3

theorem solution_interval_log_eq (h_mono : ∀ x y, (0 < x ∧ x < y) → f x < f y)
  (h_f2 : f 2 = 0)
  (h_f3 : f 3 > 0) :
  ∃ x, (2 ≤ x ∧ x < 3 ∧ f x = 0) :=
by
  sorry

end solution_interval_log_eq_l47_47399


namespace tan_add_pi_over_3_l47_47159

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l47_47159


namespace tan_add_pi_over_3_l47_47164

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l47_47164


namespace slab_cost_l47_47103

-- Define the conditions
def cubes_per_stick : ℕ := 4
def cubes_per_slab : ℕ := 80
def total_kabob_cost : ℕ := 50
def kabob_sticks_made : ℕ := 40
def total_cubes_needed := kabob_sticks_made * cubes_per_stick
def slabs_needed := total_cubes_needed / cubes_per_slab

-- Final proof problem statement in Lean 4
theorem slab_cost : (total_kabob_cost / slabs_needed) = 25 := by
  sorry

end slab_cost_l47_47103


namespace imaginary_part_of_conjugate_l47_47881

def z : Complex := Complex.mk 1 2

def z_conj : Complex := Complex.mk 1 (-2)

theorem imaginary_part_of_conjugate :
  z_conj.im = -2 := by
  sorry

end imaginary_part_of_conjugate_l47_47881


namespace count_zero_vectors_l47_47292

variable {V : Type} [AddCommGroup V]

variables (A B C D M O : V)

def vector_expressions_1 := (A - B) + (B - C) + (C - A) = 0
def vector_expressions_2 := (A - B) + (M - B) + (B - O) + (O - M) ≠ 0
def vector_expressions_3 := (A - B) - (A - C) + (B - D) - (C - D) = 0
def vector_expressions_4 := (O - A) + (O - C) + (B - O) + (C - O) ≠ 0

theorem count_zero_vectors :
  (vector_expressions_1 A B C) ∧
  (vector_expressions_2 A B M O) ∧
  (vector_expressions_3 A B C D) ∧
  (vector_expressions_4 O A C B) →
  (2 = 2) :=
sorry

end count_zero_vectors_l47_47292


namespace total_pages_in_book_l47_47119

def pagesReadMonday := 23
def pagesReadTuesday := 38
def pagesReadWednesday := 61
def pagesReadThursday := 12
def pagesReadFriday := 2 * pagesReadThursday

def totalPagesRead := pagesReadMonday + pagesReadTuesday + pagesReadWednesday + pagesReadThursday + pagesReadFriday

theorem total_pages_in_book :
  totalPagesRead = 158 :=
by
  sorry

end total_pages_in_book_l47_47119


namespace rationalize_denominator_l47_47025

theorem rationalize_denominator :
  ∃ A B C : ℤ, A * B * C = 180 ∧
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

end rationalize_denominator_l47_47025


namespace math_problem_l47_47866

theorem math_problem :
  (2^8 + 4^5) * (1^3 - (-1)^3)^2 = 5120 :=
by
  sorry

end math_problem_l47_47866


namespace freds_average_book_cost_l47_47999

theorem freds_average_book_cost :
  ∀ (initial_amount spent_amount num_books remaining_amount avg_cost : ℕ),
    initial_amount = 236 →
    remaining_amount = 14 →
    num_books = 6 →
    spent_amount = initial_amount - remaining_amount →
    avg_cost = spent_amount / num_books →
    avg_cost = 37 :=
by
  intros initial_amount spent_amount num_books remaining_amount avg_cost h_init h_rem h_books h_spent h_avg
  sorry

end freds_average_book_cost_l47_47999


namespace fraction_addition_l47_47439

def fraction_sum : ℚ := (2 : ℚ)/5 + (3 : ℚ)/8

theorem fraction_addition : fraction_sum = 31/40 := by
  sorry

end fraction_addition_l47_47439


namespace volume_of_bottle_l47_47669

theorem volume_of_bottle (r h : ℝ) (π : ℝ) (h₀ : π > 0)
  (h₁ : r^2 * h + (4 / 3) * r^3 = 625) :
  π * (r^2 * h + (4 / 3) * r^3) = 625 * π :=
by sorry

end volume_of_bottle_l47_47669


namespace fraction_addition_l47_47438

def fraction_sum : ℚ := (2 : ℚ)/5 + (3 : ℚ)/8

theorem fraction_addition : fraction_sum = 31/40 := by
  sorry

end fraction_addition_l47_47438


namespace sin_315_eq_neg_sqrt2_div_2_l47_47469

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 * π / 180) = - (Real.sqrt 2 / 2) := 
sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47469


namespace find_x_l47_47305

theorem find_x :
  ∃ n : ℕ, odd n ∧ (6^n + 1).natPrimeFactors.length = 3 ∧ 11 ∈ (6^n + 1).natPrimeFactors ∧ 6^n + 1 = 7777 :=
by sorry

end find_x_l47_47305


namespace doctor_is_correct_l47_47062

noncomputable theory

def hydra_heads_never_equal : Prop :=
  ∀ (a b : ℕ), 
    a = 2016 ∧ b = 2017 ∧ 
    (∀ n : ℕ, ∃ (a_new b_new : ℕ), 
      (a_new = a + 5 ∨ a_new = a + 7) ∧ 
      (b_new = b + 5 ∨ b_new = b + 7) ∧
      (∀ m : ℕ, m < n → a_new + b_new - 4 * (m + 1) ≠ (a_new + b_new) / 2 * 2)
    ) → 
    ∀ n : ℕ, (a + b) % 2 = 1 ∧ a ≠ b

theorem doctor_is_correct : hydra_heads_never_equal :=
by sorry

end doctor_is_correct_l47_47062


namespace calculate_sum_l47_47984

theorem calculate_sum : 5 * 12 + 7 * 15 + 13 * 4 + 6 * 9 = 271 :=
by
  sorry

end calculate_sum_l47_47984


namespace probability_two_white_balls_is_4_over_15_l47_47262

-- Define the conditions of the problem
def total_balls : ℕ := 15
def white_balls_initial : ℕ := 8
def black_balls : ℕ := 7
def balls_drawn : ℕ := 2 -- Note: Even though not explicitly required, it's part of the context

-- Calculate the probability of drawing two white balls without replacement
noncomputable def probability_two_white_balls : ℚ :=
  (white_balls_initial / total_balls) * ((white_balls_initial - 1) / (total_balls - 1))

-- The theorem to prove
theorem probability_two_white_balls_is_4_over_15 :
  probability_two_white_balls = 4 / 15 := by
  sorry

end probability_two_white_balls_is_4_over_15_l47_47262


namespace factor_expression_l47_47868

theorem factor_expression (x : ℝ) : 
  x^2 * (x + 3) + 2 * x * (x + 3) + (x + 3) = (x + 1)^2 * (x + 3) := by
  sorry

end factor_expression_l47_47868


namespace sin_315_degree_l47_47478

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l47_47478


namespace dress_total_price_correct_l47_47094

-- Define constants and variables
def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

-- Function to calculate sale price after discount
def sale_price (op : ℝ) (dr : ℝ) : ℝ := op - (op * dr)

-- Function to calculate total price including tax
def total_selling_price (sp : ℝ) (tr : ℝ) : ℝ := sp + (sp * tr)

-- The proof statement to be proven
theorem dress_total_price_correct :
  total_selling_price (sale_price original_price discount_rate) tax_rate = 96.6 :=
  by sorry

end dress_total_price_correct_l47_47094


namespace fraction_addition_l47_47444

theorem fraction_addition (a b c d : ℚ) (ha : a = 2/5) (hb : b = 3/8) (hc : c = 31/40) :
  a + b = c :=
by
  rw [ha, hb, hc]
  -- The proof part is skipped here as per instructions
  sorry

end fraction_addition_l47_47444


namespace find_m_l47_47576

open Nat  

-- Definition of lcm in Lean, if it's not already provided in Mathlib
def lcm (a b : Nat) : Nat := (a * b) / gcd a b

theorem find_m (m : ℕ) (h1 : 0 < m)
    (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180):
    m = 60 :=
sorry

end find_m_l47_47576


namespace gcd_triangular_number_l47_47299

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem gcd_triangular_number (n : ℕ) (h : n > 2) :
  ∃ k, n = 12 * k + 2 → gcd (6 * triangular_number n) (n - 2) = 12 :=
  sorry

end gcd_triangular_number_l47_47299


namespace sum_of_common_divisors_36_48_l47_47726

-- Definitions based on the conditions
def is_divisor (n d : ℕ) : Prop := d ∣ n

-- List of divisors for 36 and 48
def divisors_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]
def divisors_48 : List ℕ := [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]

-- Definition of common divisors
def common_divisors_36_48 : List ℕ := [1, 2, 3, 4, 6, 12]

-- Sum of common divisors
def sum_common_divisors_36_48 := common_divisors_36_48.sum

-- The statement of the theorem
theorem sum_of_common_divisors_36_48 : sum_common_divisors_36_48 = 28 := by
  sorry

end sum_of_common_divisors_36_48_l47_47726


namespace sin_315_degree_l47_47491

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l47_47491


namespace increasing_sequence_range_of_a_l47_47302

theorem increasing_sequence_range_of_a (a : ℝ) (a_n : ℕ → ℝ) (h : ∀ n : ℕ, a_n n = a * n ^ 2 + n) (increasing : ∀ n : ℕ, a_n (n + 1) > a_n n) : 0 ≤ a :=
by
  sorry

end increasing_sequence_range_of_a_l47_47302


namespace sin_315_degree_l47_47480

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l47_47480


namespace probability_of_one_black_ball_l47_47424

theorem probability_of_one_black_ball (total_balls black_balls white_balls drawn_balls : ℕ) 
  (h_total : total_balls = 4)
  (h_black : black_balls = 2)
  (h_white : white_balls = 2)
  (h_drawn : drawn_balls = 2) :
  ((Nat.choose black_balls 1) * (Nat.choose white_balls 1) : ℚ) / (Nat.choose total_balls drawn_balls) = 2 / 3 :=
by {
  -- Insert proof here
  sorry
}

end probability_of_one_black_ball_l47_47424


namespace sin_315_eq_neg_sqrt2_div_2_l47_47534

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47534


namespace quadratic_eq_two_distinct_real_roots_l47_47903

theorem quadratic_eq_two_distinct_real_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 2 * x₁ + a = 0) ∧ (x₂^2 - 2 * x₂ + a = 0)) ↔ a < 1 :=
by
  sorry

end quadratic_eq_two_distinct_real_roots_l47_47903


namespace vector_calculation_l47_47642

variables (a b : ℝ × ℝ)

def a_def : Prop := a = (3, 5)
def b_def : Prop := b = (-2, 1)

theorem vector_calculation (h1 : a_def a) (h2 : b_def b) : a - 2 • b = (7, 3) :=
sorry

end vector_calculation_l47_47642


namespace vectors_parallel_l47_47145

theorem vectors_parallel (m n : ℝ) (k : ℝ) (h1 : 2 = k * 1) (h2 : -1 = k * m) (h3 : 2 = k * n) : 
  m + n = 1 / 2 := 
by
  sorry

end vectors_parallel_l47_47145


namespace exchange_rate_lire_l47_47854

theorem exchange_rate_lire (x : ℕ) (h : 2500 / 2 = x / 5) : x = 6250 :=
by
  sorry

end exchange_rate_lire_l47_47854


namespace count_harmonic_vals_l47_47610

def floor (x : ℝ) : ℤ := sorry -- or use Mathlib function
def frac (x : ℝ) : ℝ := x - (floor x)

def is_harmonic_progression (a b c : ℝ) : Prop := 
  (1 / a) = (2 / b) - (1 / c)

theorem count_harmonic_vals :
  (∃ x, is_harmonic_progression x (floor x) (frac x)) ∧
  (∃! x1 x2, is_harmonic_progression x1 (floor x1) (frac x1) ∧
               is_harmonic_progression x2 (floor x2) (frac x2)) ∧
  x1 ≠ x2 :=
  sorry

end count_harmonic_vals_l47_47610


namespace interest_rate_is_4_percent_l47_47808

variable (P A n : ℝ)
variable (r : ℝ)
variable (n_pos : n ≠ 0)

-- Define the conditions
def principal : ℝ := P
def amount_after_n_years : ℝ := A
def years : ℝ := n
def interest_rate : ℝ := r

-- The compound interest formula
def compound_interest (P A r : ℝ) (n : ℝ) : Prop :=
  A = P * (1 + r) ^ n

-- The Lean theorem statement
theorem interest_rate_is_4_percent
  (P_val : principal = 7500)
  (A_val : amount_after_n_years = 8112)
  (n_val : years = 2)
  (h : compound_interest P A r n) :
  r = 0.04 :=
sorry

end interest_rate_is_4_percent_l47_47808


namespace westgate_high_school_chemistry_l47_47108

theorem westgate_high_school_chemistry :
  ∀ (total_players physics_both physics : ℕ),
    total_players = 15 →
    physics_both = 3 →
    physics = 8 →
    (total_players - (physics - physics_both)) - physics_both = 10 := by
  intros total_players physics_both physics h1 h2 h3
  sorry

end westgate_high_school_chemistry_l47_47108


namespace find_original_intensity_l47_47391

variable (I : ℝ)  -- Define intensity of the original red paint (in percentage).

-- Conditions:
variable (fractionReplaced : ℝ) (newIntensity : ℝ) (replacingIntensity : ℝ)
  (fractionReplaced_eq : fractionReplaced = 0.8)
  (newIntensity_eq : newIntensity = 30)
  (replacingIntensity_eq : replacingIntensity = 25)

-- Theorem statement:
theorem find_original_intensity :
  (1 - fractionReplaced) * I + fractionReplaced * replacingIntensity = newIntensity → I = 50 :=
sorry

end find_original_intensity_l47_47391


namespace total_weekly_cups_brewed_l47_47268

-- Define the given conditions
def weekday_cups_per_hour : ℕ := 10
def weekend_total_cups : ℕ := 120
def shop_open_hours_per_day : ℕ := 5
def weekdays_in_week : ℕ := 5

-- Prove the total number of coffee cups brewed in one week
theorem total_weekly_cups_brewed : 
  (weekday_cups_per_hour * shop_open_hours_per_day * weekdays_in_week) 
  + weekend_total_cups = 370 := 
by
  sorry

end total_weekly_cups_brewed_l47_47268


namespace ratio_of_gold_and_copper_l47_47146

theorem ratio_of_gold_and_copper
  (G C : ℝ)
  (hG : G = 11)
  (hC : C = 5)
  (hA : (11 * G + 5 * C) / (G + C) = 8) : G = C :=
by
  sorry

end ratio_of_gold_and_copper_l47_47146


namespace general_formula_sequence_l47_47657

-- Define the sequence as an arithmetic sequence with the given first term and common difference
def arithmetic_sequence (a_1 d : ℕ) (n : ℕ) : ℕ := a_1 + (n - 1) * d

-- Define given values
def a_1 : ℕ := 1
def d : ℕ := 2

-- State the theorem to be proved
theorem general_formula_sequence :
  ∀ n : ℕ, n > 0 → arithmetic_sequence a_1 d n = 2 * n - 1 :=
by
  intro n hn
  sorry

end general_formula_sequence_l47_47657


namespace sin_315_eq_neg_sqrt_2_div_2_l47_47506

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l47_47506


namespace tan_shifted_value_l47_47185

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l47_47185


namespace vertical_asymptote_l47_47132

theorem vertical_asymptote (x : ℝ) : 
  (∃ x, 4 * x + 5 = 0) → x = -5/4 :=
by 
  sorry

end vertical_asymptote_l47_47132


namespace sharp_triple_72_l47_47716

-- Definition of the transformation function
def sharp (N : ℝ) : ℝ := 0.4 * N + 3

-- Theorem statement
theorem sharp_triple_72 : sharp (sharp (sharp 72)) = 9.288 := by
  sorry

end sharp_triple_72_l47_47716


namespace product_of_slope_and_intercept_l47_47655

theorem product_of_slope_and_intercept {x1 y1 x2 y2 : ℝ} (h1 : x1 = -4) (h2 : y1 = -2) (h3 : x2 = 1) (h4 : y2 = 3) :
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m * b = 2 :=
by
  sorry

end product_of_slope_and_intercept_l47_47655


namespace find_initial_quarters_l47_47916

variables {Q : ℕ} -- Initial number of quarters

def quarters_to_dollars (q : ℕ) : ℝ := q * 0.25

noncomputable def initial_cash : ℝ := 40
noncomputable def cash_given_to_sister : ℝ := 5
noncomputable def quarters_given_to_sister : ℕ := 120
noncomputable def remaining_total : ℝ := 55

theorem find_initial_quarters (Q : ℕ) (h1 : quarters_to_dollars Q + 40 = 90) : Q = 200 :=
by { sorry }

end find_initial_quarters_l47_47916


namespace dust_particles_before_sweeping_l47_47033

theorem dust_particles_before_sweeping 
  (cleared_fraction : ℚ := 9/10)
  (dust_particles_added : ℕ := 223)
  (post_walking_dust_particles : ℕ := 331) :
  let remaining_fraction := 1 - cleared_fraction
  let pre_walking_dust_particles : ℕ := post_walking_dust_particles - dust_particles_added
  let pre_sweeping_dust_particles : ℕ := pre_walking_dust_particles / remaining_fraction
  pre_sweeping_dust_particles = 1080 :=
by
  sorry

end dust_particles_before_sweeping_l47_47033


namespace max_value_of_g_l47_47997

noncomputable def g (x : ℝ) : ℝ := min (min (3 * x + 3) ((1 / 3) * x + 1)) (-2 / 3 * x + 8)

theorem max_value_of_g : ∃ x : ℝ, g x = 10 / 3 :=
by
  sorry

end max_value_of_g_l47_47997


namespace two_f_two_plus_three_f_neg_two_eq_neg_107_l47_47552

noncomputable def f (x : ℤ) : ℤ := 3 * x^3 - 2 * x^2 + 4 * x - 7

theorem two_f_two_plus_three_f_neg_two_eq_neg_107 :
  2 * (f 2) + 3 * (f (-2)) = -107 :=
by sorry

end two_f_two_plus_three_f_neg_two_eq_neg_107_l47_47552


namespace sin_315_equals_minus_sqrt2_div_2_l47_47474

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l47_47474


namespace hydras_never_die_l47_47060

def two_hydras_survive (a b : ℕ) : Prop :=
  ∀ n : ℕ, ∀ (a_heads b_heads : ℕ),
    (a_heads = a + n ∗ (5 ∨ 7) - 4 ∗ n) ∧
    (b_heads = b + n ∗ (5 ∨ 7) - 4 ∗ n) → a_heads ≠ b_heads

theorem hydras_never_die :
  two_hydras_survive 2016 2017 :=
by sorry

end hydras_never_die_l47_47060


namespace kathleen_money_left_l47_47787

def june_savings : ℕ := 21
def july_savings : ℕ := 46
def august_savings : ℕ := 45

def school_supplies_expenses : ℕ := 12
def new_clothes_expenses : ℕ := 54

def total_savings : ℕ := june_savings + july_savings + august_savings
def total_expenses : ℕ := school_supplies_expenses + new_clothes_expenses

def total_money_left : ℕ := total_savings - total_expenses

theorem kathleen_money_left : total_money_left = 46 :=
by
  sorry

end kathleen_money_left_l47_47787


namespace arithmetic_geometric_sequence_l47_47233

theorem arithmetic_geometric_sequence (a1 d : ℝ) (h1 : a1 = 1) (h2 : d ≠ 0) (h_geom : (a1 + d) ^ 2 = a1 * (a1 + 4 * d)) :
  d = 2 :=
by
  sorry

end arithmetic_geometric_sequence_l47_47233


namespace length_of_side_of_pentagon_l47_47828

-- Assuming these conditions from the math problem:
-- 1. The perimeter of the regular polygon is 125.
-- 2. The polygon is a pentagon (5 sides).

-- Let's define the conditions:
def perimeter := 125
def sides := 5
def regular_polygon (perimeter : ℕ) (sides : ℕ) := (perimeter / sides : ℕ)

-- Statement to be proved:
theorem length_of_side_of_pentagon : regular_polygon perimeter sides = 25 := 
by sorry

end length_of_side_of_pentagon_l47_47828


namespace complement_union_l47_47759

variable (U : Set ℤ)
variable (A : Set ℤ)
variable (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} :=
sorry

end complement_union_l47_47759


namespace sin_315_equals_minus_sqrt2_div_2_l47_47473

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l47_47473


namespace hydras_survive_l47_47069

theorem hydras_survive (A_heads : ℕ) (B_heads : ℕ) (growthA growthB : ℕ → ℕ) (a b : ℕ)
    (hA : A_heads = 2016) (hB : B_heads = 2017)
    (growthA_conds : ∀ n, growthA n ∈ {5, 7})
    (growthB_conds : ∀ n, growthB n ∈ {5, 7}) :
  ∀ n, let total_heads := (A_heads + growthA n - 2 * n) + (B_heads + growthB n - 2 * n);
       total_heads % 2 = 1 :=
by intro n
   sorry

end hydras_survive_l47_47069


namespace fraction_addition_l47_47449

theorem fraction_addition : (2 / 5) + (3 / 8) = 31 / 40 := 
by {
  sorry
}

end fraction_addition_l47_47449


namespace number_of_ways_is_25_l47_47239

-- Define the number of books
def number_of_books : ℕ := 5

-- Define the function to calculate the number of ways
def number_of_ways_to_buy_books : ℕ :=
  number_of_books * number_of_books

-- Define the theorem to be proved
theorem number_of_ways_is_25 : 
  number_of_ways_to_buy_books = 25 :=
by
  sorry

end number_of_ways_is_25_l47_47239


namespace total_profit_l47_47685

theorem total_profit (investment_B : ℝ) (period_B : ℝ) (profit_B : ℝ) (investment_A : ℝ) (period_A : ℝ) (total_profit : ℝ)
  (h1 : investment_A = 3 * investment_B)
  (h2 : period_A = 2 * period_B)
  (h3 : profit_B = 6000)
  (h4 : profit_B / (profit_A * 6 + profit_B) = profit_B) : total_profit = 7 * 6000 :=
by 
  sorry

#print axioms total_profit

end total_profit_l47_47685


namespace storage_space_remaining_l47_47705

def total_space_remaining (first_floor second_floor: ℕ) (boxes: ℕ) : ℕ :=
  first_floor + second_floor - boxes

theorem storage_space_remaining :
  ∀ (first_floor second_floor boxes: ℕ),
  (first_floor = 2 * second_floor) →
  (boxes = 5000) →
  (boxes = second_floor / 4) →
  total_space_remaining first_floor second_floor boxes = 55000 :=
by
  intros first_floor second_floor boxes h1 h2 h3
  sorry

end storage_space_remaining_l47_47705


namespace tan_add_pi_over_3_l47_47158

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l47_47158


namespace find_divisor_l47_47623

-- Define the conditions
def dividend := 689
def quotient := 19
def remainder := 5

-- Define the division formula
def division_formula (divisor : ℕ) : Prop := 
  dividend = (divisor * quotient) + remainder

-- State the theorem to be proved
theorem find_divisor :
  ∃ divisor : ℕ, division_formula divisor ∧ divisor = 36 :=
by
  sorry

end find_divisor_l47_47623


namespace distance_to_mothers_house_l47_47832

theorem distance_to_mothers_house 
  (D : ℝ) 
  (h1 : (2 / 3) * D = 156.0) : 
  D = 234.0 := 
sorry

end distance_to_mothers_house_l47_47832


namespace apples_bought_is_28_l47_47044

-- Define the initial number of apples, number of apples used, and total number of apples after buying more
def initial_apples : ℕ := 38
def apples_used : ℕ := 20
def total_apples_after_buying : ℕ := 46

-- State the theorem: the number of apples bought is 28
theorem apples_bought_is_28 : (total_apples_after_buying - (initial_apples - apples_used)) = 28 := 
by sorry

end apples_bought_is_28_l47_47044


namespace sin_315_degree_l47_47489

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l47_47489


namespace total_distance_is_105_km_l47_47664

-- Define the boat's speed in still water
def boat_speed_still_water : ℝ := 50

-- Define the current speeds for each hour
def current_speed_first_hour : ℝ := 10
def current_speed_second_hour : ℝ := 20
def current_speed_third_hour : ℝ := 15

-- Calculate the effective speeds for each hour
def effective_speed_first_hour := boat_speed_still_water - current_speed_first_hour
def effective_speed_second_hour := boat_speed_still_water - current_speed_second_hour
def effective_speed_third_hour := boat_speed_still_water - current_speed_third_hour

-- Calculate the distance traveled in each hour
def distance_first_hour := effective_speed_first_hour * 1
def distance_second_hour := effective_speed_second_hour * 1
def distance_third_hour := effective_speed_third_hour * 1

-- Define the total distance
def total_distance_traveled := distance_first_hour + distance_second_hour + distance_third_hour

-- Prove that the total distance traveled is 105 km
theorem total_distance_is_105_km : total_distance_traveled = 105 := by
  sorry

end total_distance_is_105_km_l47_47664


namespace solve_for_x_l47_47406

theorem solve_for_x (a r s x : ℝ) (h1 : s > r) (h2 : r * (x + a) = s * (x - a)) :
  x = a * (s + r) / (s - r) :=
sorry

end solve_for_x_l47_47406


namespace rubber_bands_per_large_ball_l47_47796

open Nat

theorem rubber_bands_per_large_ball :
  let total_rubber_bands := 5000
  let small_bands := 50
  let small_balls := 22
  let large_balls := 13
  let used_bands := small_balls * small_bands
  let remaining_bands := total_rubber_bands - used_bands
  let large_bands := remaining_bands / large_balls
  large_bands = 300 :=
by
  sorry

end rubber_bands_per_large_ball_l47_47796


namespace raft_people_with_life_jackets_l47_47939

theorem raft_people_with_life_jackets (n m k : ℕ) (h1 : n = 21) (h2 : m = n - 7) (h3 : k = 8) :
  n - (k / (m / (n - m))) = 17 := 
by sorry

end raft_people_with_life_jackets_l47_47939


namespace a1964_eq_neg1_l47_47889

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ ∀ n ≥ 4, a n = a (n-1) * a (n-3)

theorem a1964_eq_neg1 (a : ℕ → ℤ) (h : seq a) : a 1964 = -1 :=
  by sorry

end a1964_eq_neg1_l47_47889


namespace complement_union_l47_47762

variable (U : Set ℤ)
variable (A : Set ℤ)
variable (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} :=
sorry

end complement_union_l47_47762


namespace mr_brown_no_calls_in_2020_l47_47797

noncomputable def number_of_days_with_no_calls (total_days : ℕ) (calls_niece1 : ℕ) (calls_niece2 : ℕ) (calls_niece3 : ℕ) : ℕ := 
  let calls_2 := total_days / calls_niece1
  let calls_3 := total_days / calls_niece2
  let calls_4 := total_days / calls_niece3
  let calls_6 := total_days / (Nat.lcm calls_niece1 calls_niece2)
  let calls_12_ := total_days / (Nat.lcm calls_niece1 (Nat.lcm calls_niece2 calls_niece3))
  total_days - (calls_2 + calls_3 + calls_4 - calls_6 - calls_4 - (total_days / calls_niece2 / 4) + calls_12_)

theorem mr_brown_no_calls_in_2020 : number_of_days_with_no_calls 365 2 3 4 = 122 := 
  by 
    -- Proof steps would go here
    sorry

end mr_brown_no_calls_in_2020_l47_47797


namespace tan_shifted_value_l47_47183

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l47_47183


namespace wall_width_is_4_l47_47083

structure Wall where
  width : ℝ
  height : ℝ
  length : ℝ
  volume : ℝ

theorem wall_width_is_4 (h_eq_6w : ∀ (wall : Wall), wall.height = 6 * wall.width)
                        (l_eq_7h : ∀ (wall : Wall), wall.length = 7 * wall.height)
                        (volume_16128 : ∀ (wall : Wall), wall.volume = 16128) :
  ∃ (wall : Wall), wall.width = 4 :=
by
  sorry

end wall_width_is_4_l47_47083


namespace hydras_never_die_l47_47065

theorem hydras_never_die (heads_A heads_B : ℕ) (grow_heads : ℕ → ℕ → Prop) : 
  (heads_A = 2016) → 
  (heads_B = 2017) →
  (∀ a b : ℕ, grow_heads a b → (a = 5 ∨ a = 7) ∧ (b = 5 ∨ b = 7)) →
  (∀ (a b : ℕ), grow_heads a b → (heads_A + a - 2) ≠ (heads_B + b - 2)) :=
by
  intros hA hB hGrow
  intro hEq
  sorry

end hydras_never_die_l47_47065


namespace tan_sum_formula_l47_47566

theorem tan_sum_formula (α β : ℝ) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -1 := by
sorry

end tan_sum_formula_l47_47566


namespace digit_divisible_by_3_l47_47133

theorem digit_divisible_by_3 (d : ℕ) (h : d < 10) : (15780 + d) % 3 = 0 ↔ d = 0 ∨ d = 3 ∨ d = 6 ∨ d = 9 := by
  sorry

end digit_divisible_by_3_l47_47133


namespace seeds_total_l47_47845

noncomputable def seeds_planted (x : ℕ) (y : ℕ) (z : ℕ) : ℕ :=
x + y + z

theorem seeds_total (x : ℕ) (H1 :  y = 5 * x) (H2 : x + y = 156) (z : ℕ) 
(H3 : z = 4) : seeds_planted x y z = 160 :=
by
  sorry

end seeds_total_l47_47845


namespace geom_progression_contra_l47_47708

theorem geom_progression_contra (q : ℝ) (p n : ℕ) (hp : p > 0) (hn : n > 0) :
  (11 = 10 * q^p) → (12 = 10 * q^n) → False :=
by
  -- proof steps should follow here
  sorry

end geom_progression_contra_l47_47708


namespace kathleen_remaining_money_l47_47784

-- Define the conditions
def saved_june := 21
def saved_july := 46
def saved_august := 45
def spent_school_supplies := 12
def spent_clothes := 54
def aunt_gift_threshold := 125
def aunt_gift := 25

-- Prove that Kathleen has the correct remaining amount of money
theorem kathleen_remaining_money : 
    (saved_june + saved_july + saved_august) - 
    (spent_school_supplies + spent_clothes) = 46 := 
by
  sorry

end kathleen_remaining_money_l47_47784


namespace opposite_of_83_is_84_l47_47691

noncomputable def opposite_number (k : ℕ) (n : ℕ) : ℕ :=
if k < n / 2 then k + n / 2 else k - n / 2

theorem opposite_of_83_is_84 : opposite_number 83 100 = 84 := 
by
  -- Assume the conditions of the problem here for the proof.
  sorry

end opposite_of_83_is_84_l47_47691


namespace arctg_inequality_l47_47826

theorem arctg_inequality (a b : ℝ) :
    |Real.arctan a - Real.arctan b| ≤ |b - a| := 
sorry

end arctg_inequality_l47_47826


namespace sin_315_degree_l47_47485

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l47_47485


namespace sin_315_eq_neg_sqrt2_over_2_l47_47544

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l47_47544


namespace sin_315_eq_neg_sqrt2_over_2_l47_47541

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l47_47541


namespace minimum_dot_product_l47_47199

noncomputable def min_AE_dot_AF : ℝ :=
  let AB : ℝ := 2
  let BC : ℝ := 1
  let AD : ℝ := 1
  let CD : ℝ := 1
  let angle_ABC : ℝ := 60 -- this is 60 degrees, which should be converted to radians if we need to use it
  sorry

theorem minimum_dot_product :
  let AB : ℝ := 2
  let BC : ℝ := 1
  let AD : ℝ := 1
  let CD : ℝ := 1
  let angle_ABC : ℝ := 60
  ∃ (E F : ℝ), (min_AE_dot_AF = 29 / 18) :=
    sorry

end minimum_dot_product_l47_47199


namespace initial_population_is_3162_l47_47684

noncomputable def initial_population (P : ℕ) : Prop :=
  let after_bombardment := 0.95 * (P : ℝ)
  let after_fear := 0.85 * after_bombardment
  after_fear = 2553

theorem initial_population_is_3162 : initial_population 3162 :=
  by
    -- By our condition setup, we need to prove:
    -- let after_bombardment := 0.95 * 3162
    -- let after_fear := 0.85 * after_bombardment
    -- after_fear = 2553

    -- This can be directly stated and verified through concrete calculations as in the problem steps.
    sorry

end initial_population_is_3162_l47_47684


namespace eqn_y_value_l47_47864

theorem eqn_y_value (y : ℝ) (h : (2 / y) + ((3 / y) / (6 / y)) = 1.5) : y = 2 :=
sorry

end eqn_y_value_l47_47864


namespace probability_of_purple_marble_l47_47686

theorem probability_of_purple_marble 
  (P_blue : ℝ) 
  (P_green : ℝ) 
  (P_purple : ℝ) 
  (h1 : P_blue = 0.25) 
  (h2 : P_green = 0.55) 
  (h3 : P_blue + P_green + P_purple = 1) 
  : P_purple = 0.20 := 
by 
  sorry

end probability_of_purple_marble_l47_47686


namespace raft_people_with_life_jackets_l47_47940

theorem raft_people_with_life_jackets (n m k : ℕ) (h1 : n = 21) (h2 : m = n - 7) (h3 : k = 8) :
  n - (k / (m / (n - m))) = 17 := 
by sorry

end raft_people_with_life_jackets_l47_47940


namespace time_to_reach_madison_l47_47704

-- Definitions based on the conditions
def map_distance : ℝ := 5 -- inches
def average_speed : ℝ := 60 -- miles per hour
def map_scale : ℝ := 0.016666666666666666 -- inches per mile

-- The time taken by Pete to arrive in Madison
noncomputable def time_to_madison := map_distance / map_scale / average_speed

-- The theorem to prove
theorem time_to_reach_madison : time_to_madison = 5 := 
by
  sorry

end time_to_reach_madison_l47_47704


namespace tan_add_pi_over_3_l47_47155

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l47_47155


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l47_47459

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l47_47459


namespace algebraic_expression_value_l47_47298

theorem algebraic_expression_value (a b : ℝ) (h1 : a = 1 + Real.sqrt 2) (h2 : b = Real.sqrt 3) : 
  a^2 + b^2 - 2 * a + 1 = 5 := 
by
  sorry

end algebraic_expression_value_l47_47298


namespace inequality_solution_l47_47036

theorem inequality_solution (x : ℝ) : (3 * x - 1) / (x - 2) > 0 ↔ x < 1 / 3 ∨ x > 2 :=
sorry

end inequality_solution_l47_47036


namespace lemon_juice_calculation_l47_47351

noncomputable def lemon_juice_per_lemon (table_per_dozen : ℕ) (dozens : ℕ) (lemons : ℕ) : ℕ :=
  (table_per_dozen * dozens) / lemons

theorem lemon_juice_calculation :
  lemon_juice_per_lemon 12 3 9 = 4 :=
by
  -- proof would be here
  sorry

end lemon_juice_calculation_l47_47351


namespace stream_speed_l47_47835

theorem stream_speed (C S : ℝ) 
    (h1 : C - S = 8) 
    (h2 : C + S = 12) : 
    S = 2 :=
sorry

end stream_speed_l47_47835


namespace range_of_a_l47_47136

-- Definitions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

-- Main theorem to prove
theorem range_of_a (a : ℝ) (h : a < 0)
  (h_necessary : ∀ x, ¬ p x a → ¬ q x) 
  (h_not_sufficient : ∃ x, ¬ p x a ∧ q x): 
  a ≤ -4 :=
sorry

end range_of_a_l47_47136


namespace number_of_licenses_l47_47273

-- We define the conditions for the problem
def number_of_letters : ℕ := 3  -- B, C, or D
def number_of_digits : ℕ := 4   -- Four digits following the letter
def choices_per_digit : ℕ := 10 -- Each digit can range from 0 to 9

-- We define the total number of licenses that can be generated
def total_licenses : ℕ := number_of_letters * (choices_per_digit ^ number_of_digits)

-- We now state the theorem to be proved
theorem number_of_licenses : total_licenses = 30000 :=
by
  sorry

end number_of_licenses_l47_47273


namespace multiple_rate_is_correct_l47_47013

-- Define Lloyd's standard working hours per day
def regular_hours_per_day : ℝ := 7.5

-- Define Lloyd's standard hourly rate
def regular_rate : ℝ := 3.5

-- Define the total hours worked on a specific day
def total_hours_worked : ℝ := 10.5

-- Define the total earnings for that specific day
def total_earnings : ℝ := 42.0

-- Define the function to calculate the multiple of the regular rate for excess hours
noncomputable def multiple_of_regular_rate (r_hours : ℝ) (r_rate : ℝ) (t_hours : ℝ) (t_earnings : ℝ) : ℝ :=
  let regular_earnings := r_hours * r_rate
  let excess_hours := t_hours - r_hours
  let excess_earnings := t_earnings - regular_earnings
  (excess_earnings / excess_hours) / r_rate

-- The statement to be proved
theorem multiple_rate_is_correct : 
  multiple_of_regular_rate regular_hours_per_day regular_rate total_hours_worked total_earnings = 1.5 :=
by
  sorry

end multiple_rate_is_correct_l47_47013


namespace correct_equation_for_annual_consumption_l47_47426

-- Definitions based on the problem conditions
-- average_monthly_consumption_first_half is the average monthly electricity consumption in the first half of the year, assumed to be x
def average_monthly_consumption_first_half (x : ℝ) := x

-- average_monthly_consumption_second_half is the average monthly consumption in the second half of the year, i.e., x - 2000
def average_monthly_consumption_second_half (x : ℝ) := x - 2000

-- total_annual_consumption is the total annual electricity consumption which is 150000 kWh
def total_annual_consumption (x : ℝ) := 6 * average_monthly_consumption_first_half x + 6 * average_monthly_consumption_second_half x

-- The main theorem statement which we need to prove
theorem correct_equation_for_annual_consumption (x : ℝ) : total_annual_consumption x = 150000 :=
by
  -- equation derivation
  sorry

end correct_equation_for_annual_consumption_l47_47426


namespace sequence_formula_l47_47202

theorem sequence_formula (a : ℕ → ℤ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, a n - a (n + 1) + 2 = 0) :
  ∀ n : ℕ, a n = 2 * n - 1 := 
sorry

end sequence_formula_l47_47202


namespace year_when_mother_age_is_twice_jack_age_l47_47799

noncomputable def jack_age_2010 := 12
noncomputable def mother_age_2010 := 3 * jack_age_2010

theorem year_when_mother_age_is_twice_jack_age :
  ∃ x : ℕ, mother_age_2010 + x = 2 * (jack_age_2010 + x) ∧ (2010 + x = 2022) :=
by
  sorry

end year_when_mother_age_is_twice_jack_age_l47_47799


namespace power_function_value_l47_47620

theorem power_function_value (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ (1 / 2)) (H : f 9 = 3) : f 25 = 5 :=
by
  sorry

end power_function_value_l47_47620


namespace guard_maximum_demand_l47_47696

-- Definitions based on conditions from part a)
-- A type for coins
def Coins := ℕ

-- Definitions based on question and conditions
def maximum_guard_demand (x : Coins) : Prop :=
  ∀ x, (x ≤ 199 ∧ x - 100 < 100) ∨ (x > 199 → 100 ≤ x - 100)

-- Statement of the problem in Lean 4
theorem guard_maximum_demand : ∃ x : Coins, maximum_guard_demand x ∧ x = 199 :=
by
  sorry

end guard_maximum_demand_l47_47696


namespace find_x_l47_47309

noncomputable def x (n : ℕ) := 6^n + 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_three_prime_divisors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ Prime a ∧ Prime b ∧ Prime c ∧ a * b * c ∣ x ∧ ∀ d, Prime d ∧ d ∣ x → d = a ∨ d = b ∨ d = c

theorem find_x (n : ℕ) (hodd : is_odd n) (hdiv : has_three_prime_divisors (x n)) (hprime : 11 ∣ (x n)) : x n = 7777 :=
by 
  sorry

end find_x_l47_47309


namespace bottle_caps_per_box_l47_47630

theorem bottle_caps_per_box (total_bottle_caps boxes : ℕ) (hb : total_bottle_caps = 316) (bn : boxes = 79) :
  total_bottle_caps / boxes = 4 :=
by
  sorry

end bottle_caps_per_box_l47_47630


namespace sin_315_eq_neg_sqrt2_div_2_l47_47538

theorem sin_315_eq_neg_sqrt2_div_2 : sin (315 * real.pi / 180) = -real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47538


namespace find_m_n_equal_probs_l47_47135

open Nat

theorem find_m_n_equal_probs (m n : ℕ) (hmn : 10 ≥ m ∧ m > n ∧ n ≥ 4) :
  (choose m 2 + choose n 2) * choose (m + n) 2 = m * n * choose (m + n) 2 →
  (m, n) = (10, 6) := by
  sorry

end find_m_n_equal_probs_l47_47135


namespace smiths_bakery_pies_l47_47228

theorem smiths_bakery_pies (m : ℕ) (h : m = 16) : 4 * m + 6 = 70 :=
by
  rw [h]
  norm_num
  sorry

end smiths_bakery_pies_l47_47228


namespace expressions_cannot_all_exceed_one_fourth_l47_47732

theorem expressions_cannot_all_exceed_one_fourth (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬ ((1 - a) * b > 1/4 ∧ (1 - b) * c > 1/4 ∧ (1 - c) * a > 1/4) := 
by
  sorry

end expressions_cannot_all_exceed_one_fourth_l47_47732


namespace max_value_of_y_over_x_l47_47764

theorem max_value_of_y_over_x
  (x y : ℝ)
  (h1 : x + y ≥ 3)
  (h2 : x - y ≥ -1)
  (h3 : 2 * x - y ≤ 3) :
  (∀ (x y : ℝ), (x + y ≥ 3) ∧ (x - y ≥ -1) ∧ (2 * x - y ≤ 3) → (∀ k, k = y / x → k ≤ 2)) :=
by
  sorry

end max_value_of_y_over_x_l47_47764


namespace sin_315_equals_minus_sqrt2_div_2_l47_47472

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l47_47472


namespace sin_315_eq_neg_sqrt2_over_2_l47_47543

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l47_47543


namespace min_value_of_quadratic_l47_47321

-- Define the given quadratic function
def quadratic (x : ℝ) : ℝ := 3 * x^2 + 8 * x + 15

-- Define the assertion that the minimum value of the quadratic function is 29/3
theorem min_value_of_quadratic : ∃ x : ℝ, quadratic x = 29/3 ∧ ∀ y : ℝ, quadratic y ≥ 29/3 :=
by
  sorry

end min_value_of_quadratic_l47_47321


namespace sin_315_eq_neg_sqrt2_div_2_l47_47528

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l47_47528


namespace cousin_age_result_l47_47919

-- Let define the ages
def rick_age : ℕ := 15
def oldest_brother_age : ℕ := 2 * rick_age
def middle_brother_age : ℕ := oldest_brother_age / 3
def smallest_brother_age : ℕ := middle_brother_age / 2
def youngest_brother_age : ℕ := smallest_brother_age - 2
def cousin_age : ℕ := 5 * youngest_brother_age

-- The theorem stating the cousin's age.
theorem cousin_age_result : cousin_age = 15 := by
  sorry

end cousin_age_result_l47_47919


namespace builder_needs_boards_l47_47089

theorem builder_needs_boards (packages : ℕ) (boards_per_package : ℕ) (total_boards : ℕ)
  (h1 : packages = 52)
  (h2 : boards_per_package = 3)
  (h3 : total_boards = packages * boards_per_package) : 
  total_boards = 156 :=
by
  rw [h1, h2] at h3
  exact h3

end builder_needs_boards_l47_47089


namespace fraction_inequality_l47_47021

theorem fraction_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / b) + (b / c) + (c / a) ≤ (a^2 / b^2) + (b^2 / c^2) + (c^2 / a^2) := 
by
  sorry

end fraction_inequality_l47_47021


namespace molecular_weight_correct_l47_47950

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def num_N_in_N2O3 : ℕ := 2
def num_O_in_N2O3 : ℕ := 3

def molecular_weight_N2O3 : ℝ :=
  (num_N_in_N2O3 * atomic_weight_N) + (num_O_in_N2O3 * atomic_weight_O)

theorem molecular_weight_correct :
  molecular_weight_N2O3 = 76.02 := by
  sorry

end molecular_weight_correct_l47_47950


namespace initial_masses_l47_47340

def area_of_base : ℝ := 15
def density_water : ℝ := 1
def density_ice : ℝ := 0.92
def change_in_water_level : ℝ := 5
def final_height_of_water : ℝ := 115

theorem initial_masses (m_ice m_water : ℝ) :
  m_ice = 675 ∧ m_water = 1050 :=
by
  -- Calculate the change in volume of water
  let delta_v := area_of_base * change_in_water_level

  -- Relate this volume change to the volume difference between ice and water
  let lhs := m_ice / density_ice - m_ice / density_water
  let eq1 := delta_v

  -- Solve for the mass of ice
  have h_ice : m_ice = 675 := 
  sorry

  -- Determine the final volume of water
  let final_volume_of_water := final_height_of_water * area_of_base

  -- Determine the initial mass of water
  let mass_of_water_total := density_water * final_volume_of_water
  let initial_mass_of_water :=
    mass_of_water_total - m_ice

  have h_water : m_water = 1050 := 
  sorry

  exact ⟨h_ice, h_water⟩

end initial_masses_l47_47340


namespace hexagonal_tile_difference_l47_47626

theorem hexagonal_tile_difference :
  let initial_blue_tiles := 15
  let initial_green_tiles := 9
  let new_green_border_tiles := 18
  let new_blue_border_tiles := 18
  let total_green_tiles := initial_green_tiles + new_green_border_tiles
  let total_blue_tiles := initial_blue_tiles + new_blue_border_tiles
  total_blue_tiles - total_green_tiles = 6 := by {
    sorry
  }

end hexagonal_tile_difference_l47_47626


namespace hyperbola_equation_l47_47887

theorem hyperbola_equation 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_asymptote : (b / a) = (Real.sqrt 3 / 2))
  (c : ℝ) (hc : c = Real.sqrt 7)
  (foci_directrix_condition : a^2 + b^2 = c^2) :
  (∀ x y : ℝ, (x^2 / 4 - y^2 / 3 = 1)) :=
by
  -- We do not provide the proof as per instructions
  sorry

end hyperbola_equation_l47_47887


namespace tan_angle_addition_l47_47168

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l47_47168


namespace radius_circumcircle_l47_47243

variables (R1 R2 R3 : ℝ)
variables (d : ℝ)
variables (R : ℝ)

noncomputable def sum_radii := R1 + R2 = 11
noncomputable def distance_centers := d = 5 * Real.sqrt 17
noncomputable def radius_third_sphere := R3 = 8
noncomputable def touching := R1 + R2 + 2 * R3 = d

theorem radius_circumcircle :
  R = 5 * Real.sqrt 17 / 2 :=
  by
  -- Use conditions here if necessary
  sorry

end radius_circumcircle_l47_47243


namespace beth_comic_books_percentage_l47_47856

/-- Definition of total books Beth owns -/
def total_books : ℕ := 120

/-- Definition of percentage novels in her collection -/
def percentage_novels : ℝ := 0.65

/-- Definition of number of graphic novels in her collection -/
def graphic_novels : ℕ := 18

/-- Calculation of the percentage of comic books she owns -/
theorem beth_comic_books_percentage (total_books : ℕ) (percentage_novels : ℝ) (graphic_novels : ℕ) : 
  (100 * ((total_books * (1 - percentage_novels) - graphic_novels) / total_books) = 20) :=
by
  let non_novel_books := total_books * (1 - percentage_novels)
  let comic_books := non_novel_books - graphic_novels
  let percentage_comic_books := 100 * (comic_books / total_books)
  have h : percentage_comic_books = 20 := sorry
  assumption

end beth_comic_books_percentage_l47_47856


namespace train_pass_jogger_in_36_sec_l47_47271

noncomputable def time_to_pass_jogger (speed_jogger speed_train : ℝ) (lead_jogger len_train : ℝ) : ℝ :=
  let speed_jogger_mps := speed_jogger * (1000 / 3600)
  let speed_train_mps := speed_train * (1000 / 3600)
  let relative_speed := speed_train_mps - speed_jogger_mps
  let total_distance := lead_jogger + len_train
  total_distance / relative_speed

theorem train_pass_jogger_in_36_sec :
  time_to_pass_jogger 9 45 240 120 = 36 := by
  sorry

end train_pass_jogger_in_36_sec_l47_47271
