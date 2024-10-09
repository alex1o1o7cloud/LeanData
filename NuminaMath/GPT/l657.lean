import Mathlib

namespace roots_order_l657_65733

theorem roots_order {a b m n : ℝ} (h1 : m < n) (h2 : a < b)
  (hm : 1 - (m - a) * (m - b) = 0) (hn : 1 - (n - a) * (n - b) = 0) :
  m < a ∧ a < b ∧ b < n :=
sorry

end roots_order_l657_65733


namespace Eric_eggs_collected_l657_65721

theorem Eric_eggs_collected : 
  (∀ (chickens : ℕ) (eggs_per_chicken_per_day : ℕ) (days : ℕ),
    chickens = 4 ∧ eggs_per_chicken_per_day = 3 ∧ days = 3 → 
    chickens * eggs_per_chicken_per_day * days = 36) :=
by
  sorry

end Eric_eggs_collected_l657_65721


namespace find_k_l657_65756

-- The function that computes the sum of the digits for the known form of the product (9 * 999...9) with k digits.
def sum_of_digits (k : ℕ) : ℕ :=
  8 + 9 * (k - 1) + 1

theorem find_k (k : ℕ) : sum_of_digits k = 2000 ↔ k = 222 := by
  sorry

end find_k_l657_65756


namespace solution_set_inequality_l657_65789

theorem solution_set_inequality (m : ℝ) (h : 3 - m < 0) :
  { x : ℝ | (2 - m) * x + 2 > m } = { x : ℝ | x < -1 } :=
sorry

end solution_set_inequality_l657_65789


namespace rational_solutions_equation_l657_65714

theorem rational_solutions_equation :
  ∃ x : ℚ, (|x - 19| + |x - 93| = 74 ∧ x ∈ {y : ℚ | 19 ≤ y ∨ 19 < y ∧ y < 93 ∨ y ≥ 93}) :=
sorry

end rational_solutions_equation_l657_65714


namespace find_radius_of_circle_B_l657_65792

noncomputable def radius_of_circle_B : Real :=
  sorry

theorem find_radius_of_circle_B :
  let A := 2
  let R := 4
  -- Define x as the horizontal distance (FG) and y as the vertical distance (GH)
  ∃ (x y : Real), 
  (y = x + (x^2 / 2)) ∧
  (y = 2 - (x^2 / 4)) ∧
  (5 * x^2 + 4 * x - 8 = 0) ∧
  -- Contains only the positive solution among possible valid radii
  (radius_of_circle_B = (22 / 25) + (2 * Real.sqrt 11 / 25))
:= 
sorry

end find_radius_of_circle_B_l657_65792


namespace problem_abc_value_l657_65736

theorem problem_abc_value 
  (a b c : ℤ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > 0)
  (h4 : Int.gcd b c = 1)
  (h5 : (b + c) % a = 0)
  (h6 : (a + c) % b = 0) :
  a * b * c = 6 :=
sorry

end problem_abc_value_l657_65736


namespace triangle_angle_contradiction_l657_65786

-- Define the condition: all internal angles of the triangle are less than 60 degrees.
def condition (α β γ : ℝ) (h: α + β + γ = 180): Prop :=
  α < 60 ∧ β < 60 ∧ γ < 60

-- The proof statement
theorem triangle_angle_contradiction (α β γ : ℝ) (h_sum : α + β + γ = 180) (h: condition α β γ h_sum) : false :=
sorry

end triangle_angle_contradiction_l657_65786


namespace tan_20_plus_4sin_20_eq_sqrt3_l657_65784

theorem tan_20_plus_4sin_20_eq_sqrt3 :
  (Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180)) = Real.sqrt 3 := by
  sorry

end tan_20_plus_4sin_20_eq_sqrt3_l657_65784


namespace page_number_added_twice_l657_65725

theorem page_number_added_twice (n p : ℕ) (Hn : 1 ≤ n) (Hsum : (n * (n + 1)) / 2 + p = 2630) : 
  p = 2 :=
sorry

end page_number_added_twice_l657_65725


namespace absolute_value_of_slope_l657_65791

noncomputable def circle_center1 : ℝ × ℝ := (14, 92)
noncomputable def circle_center2 : ℝ × ℝ := (17, 76)
noncomputable def circle_center3 : ℝ × ℝ := (19, 84)
noncomputable def radius : ℝ := 3
noncomputable def point_on_line : ℝ × ℝ := (17, 76)

theorem absolute_value_of_slope :
  ∃ m : ℝ, ∀ line : ℝ × ℝ → Prop,
    (line point_on_line) ∧ 
    (∀ p, (line p) → true) → 
    abs m = 24 := 
  sorry

end absolute_value_of_slope_l657_65791


namespace quadratic_b_value_l657_65744

theorem quadratic_b_value (b : ℝ) (n : ℝ) (h_b_neg : b < 0) 
  (h_equiv : ∀ x : ℝ, (x + n)^2 + 1 / 16 = x^2 + b * x + 1 / 4) : 
  b = - (Real.sqrt 3) / 2 := 
sorry

end quadratic_b_value_l657_65744


namespace max_alligators_in_days_l657_65779

noncomputable def days := 616
noncomputable def weeks := 88  -- derived from 616 / 7
noncomputable def alligators_per_week := 1

theorem max_alligators_in_days
  (h1 : weeks = days / 7)
  (h2 : ∀ (w : ℕ), alligators_per_week = 1) :
  weeks * alligators_per_week = 88 := by
  sorry

end max_alligators_in_days_l657_65779


namespace optimal_ticket_price_l657_65724

noncomputable def revenue (x : ℕ) : ℤ :=
  if x < 6 then -5750
  else if x ≤ 10 then 1000 * (x : ℤ) - 5750
  else if x ≤ 38 then -30 * (x : ℤ)^2 + 1300 * (x : ℤ) - 5750
  else -5750

theorem optimal_ticket_price :
  revenue 22 = 8330 :=
by
  sorry

end optimal_ticket_price_l657_65724


namespace inequality_3a3_2b3_3a2b_2ab2_l657_65750

theorem inequality_3a3_2b3_3a2b_2ab2 (a b : ℝ) (h₁ : a ≥ b) (h₂ : b > 0) : 
  3 * a ^ 3 + 2 * b ^ 3 ≥ 3 * a ^ 2 * b + 2 * a * b ^ 2 :=
by
  sorry

end inequality_3a3_2b3_3a2b_2ab2_l657_65750


namespace moles_of_KHSO4_formed_l657_65734

-- Chemical reaction definition
def reaction (n_KOH n_H2SO4 : ℕ) : ℕ :=
  if n_KOH = n_H2SO4 then n_KOH else 0

-- Given conditions
def moles_KOH : ℕ := 2
def moles_H2SO4 : ℕ := 2

-- Proof statement to be proved
theorem moles_of_KHSO4_formed : reaction moles_KOH moles_H2SO4 = 2 :=
by sorry

end moles_of_KHSO4_formed_l657_65734


namespace probability_sum_of_two_dice_is_4_l657_65757

noncomputable def fair_dice_probability_sum_4 : ℚ :=
  let total_outcomes := 6 * 6 -- Total outcomes for two dice
  let favorable_outcomes := 3 -- Outcomes that sum to 4: (1, 3), (3, 1), (2, 2)
  favorable_outcomes / total_outcomes

theorem probability_sum_of_two_dice_is_4 : fair_dice_probability_sum_4 = 1 / 12 := 
by
  sorry

end probability_sum_of_two_dice_is_4_l657_65757


namespace intersection_of_sets_l657_65739

def SetA : Set ℝ := { x | |x| ≤ 1 }
def SetB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_of_sets : (SetA ∩ SetB) = { x | 0 ≤ x ∧ x ≤ 1 } := 
by
  sorry

end intersection_of_sets_l657_65739


namespace perpendicular_lines_b_eq_neg_six_l657_65797

theorem perpendicular_lines_b_eq_neg_six
    (b : ℝ) :
    (∀ x y : ℝ, 3 * y + 2 * x - 4 = 0 → y = (-2/3) * x + 4/3) →
    (∀ x y : ℝ, 4 * y + b * x - 6 = 0 → y = (-b/4) * x + 3/2) →
    - (2/3) * (-b/4) = -1 →
    b = -6 := 
sorry

end perpendicular_lines_b_eq_neg_six_l657_65797


namespace pages_revised_only_once_l657_65780

theorem pages_revised_only_once 
  (total_pages : ℕ)
  (cost_per_page_first_time : ℝ)
  (cost_per_page_revised : ℝ)
  (revised_twice_pages : ℕ)
  (total_cost : ℝ)
  (pages_revised_only_once : ℕ) :
  total_pages = 100 →
  cost_per_page_first_time = 10 →
  cost_per_page_revised = 5 →
  revised_twice_pages = 30 →
  total_cost = 1400 →
  10 * (total_pages - pages_revised_only_once - revised_twice_pages) + 
  15 * pages_revised_only_once + 
  20 * revised_twice_pages = total_cost →
  pages_revised_only_once = 20 :=
by
  intros 
  sorry

end pages_revised_only_once_l657_65780


namespace sum_of_distinct_prime_factors_of_number_is_10_l657_65770

-- Define the constant number 9720
def number : ℕ := 9720

-- Define the distinct prime factors of 9720
def distinct_prime_factors_of_number : List ℕ := [2, 3, 5]

-- Sum function for the list of distinct prime factors
def sum_of_distinct_prime_factors (lst : List ℕ) : ℕ :=
  lst.foldr (.+.) 0

-- The main theorem to prove
theorem sum_of_distinct_prime_factors_of_number_is_10 :
  sum_of_distinct_prime_factors distinct_prime_factors_of_number = 10 := by
  sorry

end sum_of_distinct_prime_factors_of_number_is_10_l657_65770


namespace parallel_lines_k_value_l657_65753

-- Define the lines and the condition of parallelism
def line1 (x y : ℝ) := x + 2 * y - 1 = 0
def line2 (k x y : ℝ) := k * x - y = 0

-- Define the parallelism condition
def lines_parallel (k : ℝ) := (1 / k) = (2 / -1)

-- Prove that given the parallelism condition, k equals -1/2
theorem parallel_lines_k_value (k : ℝ) (h : lines_parallel k) : k = (-1 / 2) :=
by
  sorry

end parallel_lines_k_value_l657_65753


namespace solve_system_of_inequalities_l657_65794

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x + 3 ≤ x + 2) ∧ ((x + 1) / 3 > x - 1) → x ≤ -1 := by
  sorry

end solve_system_of_inequalities_l657_65794


namespace tan_identity_15_eq_sqrt3_l657_65748

theorem tan_identity_15_eq_sqrt3 :
  (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by
  sorry

end tan_identity_15_eq_sqrt3_l657_65748


namespace compute_series_sum_l657_65737

noncomputable def term (n : ℕ) : ℝ := (5 * n - 2) / (3 ^ n)

theorem compute_series_sum : 
  ∑' n, term n = 11 / 4 := 
sorry

end compute_series_sum_l657_65737


namespace infinite_series_sum_l657_65727

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l657_65727


namespace postman_speeds_l657_65766

-- Define constants for the problem
def d1 : ℝ := 2 -- distance uphill in km
def d2 : ℝ := 4 -- distance on flat ground in km
def d3 : ℝ := 3 -- distance downhill in km
def time1 : ℝ := 2.267 -- time from A to B in hours
def time2 : ℝ := 2.4 -- time from B to A in hours
def half_time_round_trip : ℝ := 2.317 -- round trip to halfway point in hours

-- Define the speeds
noncomputable def V1 : ℝ := 3 -- speed uphill in km/h
noncomputable def V2 : ℝ := 4 -- speed on flat ground in km/h
noncomputable def V3 : ℝ := 5 -- speed downhill in km/h

-- The mathematically equivalent proof statement
theorem postman_speeds :
  (d1 / V1 + d2 / V2 + d3 / V3 = time1) ∧
  (d3 / V1 + d2 / V2 + d1 / V3 = time2) ∧
  (1 / V1 + 2 / V2 + 1.5 / V3 = half_time_round_trip / 2) :=
by 
  -- Equivalence holds because the speeds satisfy the given conditions
  sorry

end postman_speeds_l657_65766


namespace binom_18_6_eq_18564_l657_65711

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binom_18_6_eq_18564 : binomial 18 6 = 18564 := by
  sorry

end binom_18_6_eq_18564_l657_65711


namespace find_value_of_E_l657_65749

variables (Q U I E T Z : ℤ)

theorem find_value_of_E (hZ : Z = 15) (hQUIZ : Q + U + I + Z = 60) (hQUIET : Q + U + I + E + T = 75) (hQUIT : Q + U + I + T = 50) : E = 25 :=
by
  have hQUIZ_val : Q + U + I = 45 := by linarith [hZ, hQUIZ]
  have hQUIET_val : E + T = 30 := by linarith [hQUIZ_val, hQUIET]
  have hQUIT_val : T = 5 := by linarith [hQUIZ_val, hQUIT]
  linarith [hQUIET_val, hQUIT_val]

end find_value_of_E_l657_65749


namespace division_result_l657_65769

theorem division_result : 3486 / 189 = 18.444444444444443 := 
by sorry

end division_result_l657_65769


namespace product_of_largest_integer_digits_l657_65751

theorem product_of_largest_integer_digits (u v : ℕ) :
  u^2 + v^2 = 45 ∧ u < v → u * v = 18 :=
sorry

end product_of_largest_integer_digits_l657_65751


namespace triangle_inradius_is_2_5_l657_65730

variable (A : ℝ) (p : ℝ) (r : ℝ)

def triangle_has_given_inradius (A p : ℝ) : Prop :=
  A = r * p / 2

theorem triangle_inradius_is_2_5 (h₁ : A = 25) (h₂ : p = 20) :
  triangle_has_given_inradius A p r → r = 2.5 := sorry

end triangle_inradius_is_2_5_l657_65730


namespace ram_account_balance_first_year_l657_65740

theorem ram_account_balance_first_year :
  let initial_deposit := 1000
  let interest_first_year := 100
  initial_deposit + interest_first_year = 1100 :=
by
  sorry

end ram_account_balance_first_year_l657_65740


namespace time_for_Q_l657_65713

-- Definitions of conditions
def time_for_P := 252
def meet_time := 2772

-- Main statement to prove
theorem time_for_Q : (∃ T : ℕ, lcm time_for_P T = meet_time) ∧ (lcm time_for_P meet_time = meet_time) :=
    by 
    sorry

end time_for_Q_l657_65713


namespace p_sufficient_not_necessary_for_q_l657_65781

variable (x : ℝ)

def p : Prop := x > 0
def q : Prop := x > -1

theorem p_sufficient_not_necessary_for_q : (p x → q x) ∧ ¬ (q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l657_65781


namespace all_propositions_imply_l657_65723

variables (p q r : Prop)

theorem all_propositions_imply (hpqr : p ∧ q ∧ r)
                               (hnpqr : ¬p ∧ q ∧ ¬r)
                               (hpnqr : p ∧ ¬q ∧ r)
                               (hnpnqr : ¬p ∧ ¬q ∧ ¬r) :
  (p → q) ∨ r :=
by { sorry }

end all_propositions_imply_l657_65723


namespace halogens_have_solid_liquid_gas_l657_65743

def at_25C_and_1atm (element : String) : String :=
  match element with
  | "Li" | "Na" | "K" | "Rb" | "Cs" => "solid"
  | "N" => "gas"
  | "P" | "As" | "Sb" | "Bi" => "solid"
  | "O" => "gas"
  | "S" | "Se" | "Te" => "solid"
  | "F" | "Cl" => "gas"
  | "Br" => "liquid"
  | "I" | "At" => "solid"
  | _ => "unknown"

def family_has_solid_liquid_gas (family : List String) : Prop :=
  "solid" ∈ family.map at_25C_and_1atm ∧
  "liquid" ∈ family.map at_25C_and_1atm ∧
  "gas" ∈ family.map at_25C_and_1atm

theorem halogens_have_solid_liquid_gas :
  family_has_solid_liquid_gas ["F", "Cl", "Br", "I", "At"] :=
by
  sorry

end halogens_have_solid_liquid_gas_l657_65743


namespace solution_of_inequality_l657_65732

theorem solution_of_inequality (x : ℝ) : x * (x - 1) < 2 ↔ -1 < x ∧ x < 2 :=
sorry

end solution_of_inequality_l657_65732


namespace probability_two_digit_between_21_and_30_l657_65726

theorem probability_two_digit_between_21_and_30 (dice1 dice2 : ℤ) (h1 : 1 ≤ dice1 ∧ dice1 ≤ 6) (h2 : 1 ≤ dice2 ∧ dice2 ≤ 6) :
∃ (p : ℚ), p = 11 / 36 := 
sorry

end probability_two_digit_between_21_and_30_l657_65726


namespace find_k_square_binomial_l657_65778

theorem find_k_square_binomial (k : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 - 16 * x + k = (x + b)^2) ↔ k = 64 :=
by
  sorry

end find_k_square_binomial_l657_65778


namespace virginia_eggs_l657_65765

theorem virginia_eggs (initial_eggs : ℕ) (taken_eggs : ℕ) (result_eggs : ℕ) 
  (h_initial : initial_eggs = 200) 
  (h_taken : taken_eggs = 37) 
  (h_calculation: result_eggs = initial_eggs - taken_eggs) :
result_eggs = 163 :=
by {
  sorry
}

end virginia_eggs_l657_65765


namespace find_x2_times_sum_roots_l657_65773

noncomputable def sqrt2015 := Real.sqrt 2015

theorem find_x2_times_sum_roots
  (x1 x2 x3 : ℝ)
  (h_eq : ∀ x : ℝ, sqrt2015 * x^3 - 4030 * x^2 + 2 = 0 → x = x1 ∨ x = x2 ∨ x = x3)
  (h_ineq : x1 < x2 ∧ x2 < x3) :
  x2 * (x1 + x3) = 2 := by
  sorry

end find_x2_times_sum_roots_l657_65773


namespace bejgli_slices_l657_65722

theorem bejgli_slices (x : ℕ) (hx : x ≤ 58) 
    (h1 : x * (x - 1) * (x - 2) = 3 * (58 - x) * (57 - x) * x) : 
    58 - x = 21 :=
by
  have hpos1 : 0 < x := sorry  -- x should be strictly positive since it's a count
  have hpos2 : 0 < 58 - x := sorry  -- the remaining slices should be strictly positive
  sorry

end bejgli_slices_l657_65722


namespace probability_of_all_selected_l657_65764

theorem probability_of_all_selected :
  let p_x := 1 / 7
  let p_y := 2 / 9
  let p_z := 3 / 11
  p_x * p_y * p_z = 1 / 115.5 :=
by
  let p_x := 1 / 7
  let p_y := 2 / 9
  let p_z := 3 / 11
  sorry

end probability_of_all_selected_l657_65764


namespace smallest_four_digit_mod_8_l657_65718

theorem smallest_four_digit_mod_8 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 8 = 5 ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ m % 8 = 5 → n ≤ m) → n = 1005 :=
by
  sorry

end smallest_four_digit_mod_8_l657_65718


namespace largest_alpha_l657_65717

theorem largest_alpha (a b : ℕ) (h1 : a < b) (h2 : b < 2 * a) (N : ℕ) :
  ∃ (α : ℝ), α = 1 / (2 * a^2 - 2 * a * b + b^2) ∧
  (∃ marked_cells : ℕ, marked_cells ≥ α * (N:ℝ)^2) :=
by
  sorry

end largest_alpha_l657_65717


namespace find_m_n_l657_65775

theorem find_m_n (m n : ℕ) (positive_m : 0 < m) (positive_n : 0 < n)
  (h1 : m = 3) (h2 : n = 4) :
    Real.arctan (1 / 3) + Real.arctan (1 / 4) + Real.arctan (1 / m) + Real.arctan (1 / n) = π / 2 :=
  by 
    -- Placeholder for the proof
    sorry

end find_m_n_l657_65775


namespace tan_half_sum_eq_third_l657_65746

theorem tan_half_sum_eq_third
  (x y : ℝ)
  (h1 : Real.cos x + Real.cos y = 3/5)
  (h2 : Real.sin x + Real.sin y = 1/5) :
  Real.tan ((x + y) / 2) = 1/3 :=
by sorry

end tan_half_sum_eq_third_l657_65746


namespace f_leq_binom_l657_65754

-- Define the function f with given conditions
def f (m n : ℕ) : ℕ := if m = 1 ∨ n = 1 then 1 else sorry

-- State the property to be proven
theorem f_leq_binom (m n : ℕ) (h2 : 2 ≤ m) (h2' : 2 ≤ n) :
  f m n ≤ Nat.choose (m + n) n := 
sorry

end f_leq_binom_l657_65754


namespace cos_beta_value_cos_2alpha_plus_beta_value_l657_65795

-- Definitions of the conditions
variables (α β : ℝ)
variable (condition1 : 0 < α ∧ α < π / 2)
variable (condition2 : π / 2 < β ∧ β < π)
variable (condition3 : Real.cos (α + π / 4) = 1 / 3)
variable (condition4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3)

-- Proof problem (1)
theorem cos_beta_value :
  ∀ α β, (0 < α ∧ α < π / 2) →
  (π / 2 < β ∧ β < π) →
  (Real.cos (α + π / 4) = 1 / 3) →
  (Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) →
  Real.cos β = - 4 * Real.sqrt 2 / 9 :=
by
  intros α β condition1 condition2 condition3 condition4
  sorry

-- Proof problem (2)
theorem cos_2alpha_plus_beta_value :
  ∀ α β, (0 < α ∧ α < π / 2) →
  (π / 2 < β ∧ β < π) →
  (Real.cos (α + π / 4) = 1 / 3) →
  (Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) →
  Real.cos (2 * α + β) = -1 :=
by
  intros α β condition1 condition2 condition3 condition4
  sorry

end cos_beta_value_cos_2alpha_plus_beta_value_l657_65795


namespace arith_seq_general_formula_geom_seq_sum_l657_65729

-- Problem 1
theorem arith_seq_general_formula (a : ℕ → ℕ) (d : ℕ) (h_d : d = 3) (h_a1 : a 1 = 4) :
  a n = 3 * n + 1 :=
sorry

-- Problem 2
theorem geom_seq_sum (b : ℕ → ℚ) (S : ℕ → ℚ) (h_b1 : b 1 = 1 / 3) (r : ℚ) (h_r : r = 1 / 3) :
  S n = (1 / 2) * (1 - (1 / 3 ^ n)) :=
sorry

end arith_seq_general_formula_geom_seq_sum_l657_65729


namespace find_circle_center_l657_65709

theorem find_circle_center : ∃ (h k : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 2*x + 12*y + 1 = 0 ↔ (x - h)^2 + (y - k)^2 = 36) ∧ h = 1 ∧ k = -6 := 
sorry

end find_circle_center_l657_65709


namespace inscribed_circle_radius_l657_65793

noncomputable def semiPerimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def areaUsingHeron (a b c : ℝ) : ℝ :=
  let s := semiPerimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def inscribedCircleRadius (a b c : ℝ) : ℝ :=
  let s := semiPerimeter a b c
  let K := areaUsingHeron a b c
  K / s

theorem inscribed_circle_radius : inscribedCircleRadius 26 18 20 = Real.sqrt 31 :=
  sorry

end inscribed_circle_radius_l657_65793


namespace binom_even_if_power_of_two_binom_odd_if_not_power_of_two_l657_65700

-- Definition of power of two
def is_power_of_two (n : ℕ) := ∃ m : ℕ, n = 2^m

-- Theorems to be proven
theorem binom_even_if_power_of_two (n : ℕ) (h : is_power_of_two n) :
  ∀ k : ℕ, 1 ≤ k ∧ k < n → Nat.choose n k % 2 = 0 := sorry

theorem binom_odd_if_not_power_of_two (n : ℕ) (h : ¬ is_power_of_two n) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ Nat.choose n k % 2 = 1 := sorry

end binom_even_if_power_of_two_binom_odd_if_not_power_of_two_l657_65700


namespace train_speed_is_correct_l657_65760

-- Definitions based on the conditions
def length_of_train : ℝ := 120       -- Train is 120 meters long
def time_to_cross : ℝ := 16          -- The train takes 16 seconds to cross the post

-- Conversion constants
def seconds_to_hours : ℝ := 3600
def meters_to_kilometers : ℝ := 1000

-- The speed of the train in km/h
noncomputable def speed_of_train (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (seconds_to_hours / meters_to_kilometers)

-- Theorem: The speed of the train is 27 km/h
theorem train_speed_is_correct : speed_of_train length_of_train time_to_cross = 27 :=
by
  -- This is where the proof should be, but we leave it as sorry as instructed
  sorry

end train_speed_is_correct_l657_65760


namespace friends_attended_l657_65782

theorem friends_attended (total_guests bride_couples groom_couples : ℕ)
                         (bride_guests groom_guests family_guests friends : ℕ)
                         (h1 : total_guests = 300)
                         (h2 : bride_couples = 30)
                         (h3 : groom_couples = 30)
                         (h4 : bride_guests = bride_couples * 2)
                         (h5 : groom_guests = groom_couples * 2)
                         (h6 : family_guests = bride_guests + groom_guests)
                         (h7 : friends = total_guests - family_guests) :
  friends = 180 :=
by sorry

end friends_attended_l657_65782


namespace num_three_digit_integers_with_zero_in_units_place_divisible_by_30_l657_65719

noncomputable def countThreeDigitMultiplesOf30WithZeroInUnitsPlace : ℕ :=
  let a := 120
  let d := 30
  let l := 990
  (l - a) / d + 1

theorem num_three_digit_integers_with_zero_in_units_place_divisible_by_30 :
  countThreeDigitMultiplesOf30WithZeroInUnitsPlace = 30 := by
  sorry

end num_three_digit_integers_with_zero_in_units_place_divisible_by_30_l657_65719


namespace sum_a_b_eq_neg2_l657_65747

theorem sum_a_b_eq_neg2 (a b : ℝ) (h : (a - 2)^2 + |b + 4| = 0) : a + b = -2 := 
by 
  sorry

end sum_a_b_eq_neg2_l657_65747


namespace minutes_spent_calling_clients_l657_65777

theorem minutes_spent_calling_clients
    (C : ℕ)
    (H1 : 7 * C + C = 560) :
    C = 70 :=
sorry

end minutes_spent_calling_clients_l657_65777


namespace fraction_problem_l657_65767

theorem fraction_problem
    (q r s u : ℚ)
    (h1 : q / r = 8)
    (h2 : s / r = 4)
    (h3 : s / u = 1 / 3) :
    u / q = 3 / 2 :=
  sorry

end fraction_problem_l657_65767


namespace solve_inequality_l657_65738

theorem solve_inequality (x : ℝ) :
  abs (x + 3) + abs (2 * x - 1) < 7 ↔ -3 ≤ x ∧ x < 5 / 3 :=
by
  sorry

end solve_inequality_l657_65738


namespace value_of_m_l657_65742

theorem value_of_m (x m : ℝ) (h_positive_root : x > 0) (h_eq : x / (x - 1) - m / (1 - x) = 2) : m = -1 := by
  sorry

end value_of_m_l657_65742


namespace total_students_in_class_l657_65755

def number_of_girls := 9
def number_of_boys := 16
def total_students := number_of_girls + number_of_boys

theorem total_students_in_class : total_students = 25 :=
by
  -- The proof will go here
  sorry

end total_students_in_class_l657_65755


namespace total_trees_after_planting_l657_65701

-- Definitions based on conditions
def initial_trees : ℕ := 34
def trees_to_plant : ℕ := 49

-- Statement to prove the total number of trees after planting
theorem total_trees_after_planting : initial_trees + trees_to_plant = 83 := 
by 
  sorry

end total_trees_after_planting_l657_65701


namespace complement_of_M_with_respect_to_U_l657_65796

noncomputable def U : Set ℕ := {1, 2, 3, 4}
noncomputable def M : Set ℕ := {1, 2, 3}

theorem complement_of_M_with_respect_to_U :
  (U \ M) = {4} :=
by
  sorry

end complement_of_M_with_respect_to_U_l657_65796


namespace max_value_expression_l657_65798

theorem max_value_expression : 
  ∃ x_max : ℝ, 
    (∀ x : ℝ, -3 * x^2 + 15 * x + 9 ≤ -3 * x_max^2 + 15 * x_max + 9) ∧
    (-3 * x_max^2 + 15 * x_max + 9 = 111 / 4) :=
by
  sorry

end max_value_expression_l657_65798


namespace determine_n_from_average_l657_65761

-- Definitions derived from conditions
def total_cards (n : ℕ) : ℕ := n * (n + 1) / 2
def sum_of_values (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6
def average_value (n : ℕ) : ℚ := sum_of_values n / total_cards n

-- Main statement for proving equivalence
theorem determine_n_from_average :
  (∃ n : ℕ, average_value n = 2023) ↔ (n = 3034) :=
by
  sorry

end determine_n_from_average_l657_65761


namespace sin_right_triangle_l657_65720

theorem sin_right_triangle (FG GH : ℝ) (h1 : FG = 13) (h2 : GH = 12) (h3 : FG^2 = FH^2 + GH^2) : 
  sin_H = 5 / 13 :=
by sorry

end sin_right_triangle_l657_65720


namespace persistence_of_2_persistence_iff_2_l657_65745

def is_persistent (T : ℝ) : Prop :=
  ∀ (a b c d : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
                    a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ d ≠ 1) →
    (a + b + c + d = T) →
    (1 / a + 1 / b + 1 / c + 1 / d = T) →
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) + 1 / (1 - d) = T)

theorem persistence_of_2 : is_persistent 2 :=
by
  -- The proof is omitted as per instructions
  sorry

theorem persistence_iff_2 (T : ℝ) : is_persistent T ↔ T = 2 :=
by
  -- The proof is omitted as per instructions
  sorry

end persistence_of_2_persistence_iff_2_l657_65745


namespace solve_for_t_l657_65706

theorem solve_for_t (s t : ℚ) (h1 : 8 * s + 6 * t = 160) (h2 : s = t + 3) : t = 68 / 7 :=
by
  sorry

end solve_for_t_l657_65706


namespace midpoint_of_segment_l657_65768

theorem midpoint_of_segment :
  let p1 := (12, -8)
  let p2 := (-4, 10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint = (4, 1) :=
by
  let p1 := (12, -8)
  let p2 := (-4, 10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  show midpoint = (4, 1)
  sorry

end midpoint_of_segment_l657_65768


namespace find_integer_n_l657_65741

theorem find_integer_n :
  ∃ n : ℕ, 0 ≤ n ∧ n < 201 ∧ 200 * n ≡ 144 [MOD 101] ∧ n = 29 := 
by
  sorry

end find_integer_n_l657_65741


namespace intersection_points_count_l657_65783

theorem intersection_points_count (B : ℝ) (hB : 0 < B) :
  ∃ p : ℕ, p = 4 ∧ (∀ x y : ℝ, (y = B * x^2 ∧ y^2 + 4 * y - 2 = x^2 + 5 * y) ↔ p = 4) := by
sorry

end intersection_points_count_l657_65783


namespace candy_profit_l657_65759

theorem candy_profit :
  let num_bars := 800
  let cost_per_4_bars := 3
  let sell_per_3_bars := 2
  let cost_price := (cost_per_4_bars / 4) * num_bars
  let sell_price := (sell_per_3_bars / 3) * num_bars
  let profit := sell_price - cost_price
  profit = -66.67 :=
by
  sorry

end candy_profit_l657_65759


namespace simplify_expression_l657_65707

theorem simplify_expression :
  ((1 + 2 + 3 + 6) / 3) + ((3 * 6 + 9) / 4) = 43 / 4 := 
sorry

end simplify_expression_l657_65707


namespace coefficient_of_y_in_first_equation_is_minus_1_l657_65758

variable (x y z : ℝ)

def equation1 : Prop := 6 * x - y + 3 * z = 22 / 5
def equation2 : Prop := 4 * x + 8 * y - 11 * z = 7
def equation3 : Prop := 5 * x - 6 * y + 2 * z = 12
def sum_xyz : Prop := x + y + z = 10

theorem coefficient_of_y_in_first_equation_is_minus_1 :
  equation1 x y z → equation2 x y z → equation3 x y z → sum_xyz x y z → (-1 : ℝ) = -1 :=
by
  sorry

end coefficient_of_y_in_first_equation_is_minus_1_l657_65758


namespace symmetric_point_l657_65731

theorem symmetric_point (x0 y0 : ℝ) (P : ℝ × ℝ) (line : ℝ → ℝ) 
  (hP : P = (-1, 3)) (hline : ∀ x, line x = x) :
  ((x0, y0) = (3, -1)) ↔
    ( ∃ M : ℝ × ℝ, M = ((x0 - -1) / 2, (y0 + 3) / 2) ∧ M.1 = M.2 ) ∧ 
    ( ∃ l : ℝ, l = (y0 - 3) / (x0 + 1) ∧ l = -1 ) :=
by
  sorry

end symmetric_point_l657_65731


namespace inequality_am_gm_l657_65787

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a^3 / (b * c) + b^3 / (c * a) + c^3 / (a * b) ≥ a + b + c :=
by {
    sorry
}

end inequality_am_gm_l657_65787


namespace total_population_l657_65799

-- Defining the populations of Springfield and the difference in population
def springfield_population : ℕ := 482653
def population_difference : ℕ := 119666

-- The definition of Greenville's population in terms of Springfield's population
def greenville_population : ℕ := springfield_population - population_difference

-- The statement that we want to prove: the total population of Springfield and Greenville
theorem total_population :
  springfield_population + greenville_population = 845640 := by
  sorry

end total_population_l657_65799


namespace simplify_expression_is_3_l657_65790

noncomputable def simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) : ℝ :=
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)

theorem simplify_expression_is_3 (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) :
  simplify_expression x y z hx hy hz h = 3 :=
  sorry

end simplify_expression_is_3_l657_65790


namespace domain_of_tan_sub_pi_over_4_l657_65752

theorem domain_of_tan_sub_pi_over_4 :
  ∀ x : ℝ, (∃ k : ℤ, x = k * π + 3 * π / 4) ↔ ∃ y : ℝ, y = (x - π / 4) ∧ (∃ k : ℤ, y = (2 * k + 1) * π / 2) := 
sorry

end domain_of_tan_sub_pi_over_4_l657_65752


namespace problem_solution_l657_65774

-- Definitions and assumptions
variables (priceA priceB : ℕ)
variables (numBooksA numBooksB totalBooks : ℕ)
variables (costPriceA : priceA = 45)
variables (costPriceB : priceB = 65)
variables (totalCost : priceA * numBooksA + priceB * numBooksB ≤ 3550)
variables (totalBooksEq : numBooksA + numBooksB = 70)

-- Proof problem
theorem problem_solution :
  priceA = 45 ∧ priceB = 65 ∧ ∃ (numBooksA : ℕ), numBooksA ≥ 50 :=
by
  sorry

end problem_solution_l657_65774


namespace bob_distance_when_they_meet_l657_65763

-- Define the conditions
def distance_XY : ℝ := 10
def yolanda_rate : ℝ := 3
def bob_rate : ℝ := 4
def yolanda_start_time : ℝ := 0
def bob_start_time : ℝ := 1

-- The statement we want to prove
theorem bob_distance_when_they_meet : 
  ∃ t : ℝ, (yolanda_rate * (t + 1) + bob_rate * t = distance_XY) ∧ (bob_rate * t = 4) :=
sorry

end bob_distance_when_they_meet_l657_65763


namespace smallest_positive_integer_n_l657_65771

theorem smallest_positive_integer_n (n : ℕ) (h : n > 0) : 3^n ≡ n^3 [MOD 5] ↔ n = 3 :=
sorry

end smallest_positive_integer_n_l657_65771


namespace halfway_between_one_sixth_and_one_twelfth_is_one_eighth_l657_65785

theorem halfway_between_one_sixth_and_one_twelfth_is_one_eighth : 
  (1 / 6 + 1 / 12) / 2 = 1 / 8 := 
by
  sorry

end halfway_between_one_sixth_and_one_twelfth_is_one_eighth_l657_65785


namespace abs_val_eq_two_l657_65710

theorem abs_val_eq_two (x : ℝ) (h : |x| = 2) : x = 2 ∨ x = -2 := 
sorry

end abs_val_eq_two_l657_65710


namespace Q_over_P_l657_65735

theorem Q_over_P :
  (∀ (x : ℝ), x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 6 → 
    (P / (x + 6) + Q / (x^2 - 6*x) = (x^2 - 3*x + 12) / (x^3 + x^2 - 24*x))) →
  Q / P = 5 / 3 :=
by
  sorry

end Q_over_P_l657_65735


namespace mary_regular_hours_l657_65708

theorem mary_regular_hours (x y : ℕ) (h1 : 8 * x + 10 * y = 560) (h2 : x + y = 60) : x = 20 :=
by
  sorry

end mary_regular_hours_l657_65708


namespace Louie_monthly_payment_l657_65703

noncomputable def monthly_payment (P : ℕ) (r : ℚ) (n t : ℕ) : ℚ :=
  (P : ℚ) * (1 + r / n)^(n * t) / t

theorem Louie_monthly_payment : 
  monthly_payment 2000 0.10 1 3 = 887 := 
by
  sorry

end Louie_monthly_payment_l657_65703


namespace margie_change_l657_65712

theorem margie_change : 
  let cost_per_apple := 0.30
  let cost_per_orange := 0.40
  let number_of_apples := 5
  let number_of_oranges := 4
  let total_money := 10.00
  let total_cost_of_apples := cost_per_apple * number_of_apples
  let total_cost_of_oranges := cost_per_orange * number_of_oranges
  let total_cost_of_fruits := total_cost_of_apples + total_cost_of_oranges
  let change_received := total_money - total_cost_of_fruits
  change_received = 6.90 :=
by
  sorry

end margie_change_l657_65712


namespace inequality_proof_l657_65776

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^2 / (a + b)) + (b^2 / (b + c)) + (c^2 / (c + a)) ≥ (a + b + c) / 2 := 
by
  sorry

end inequality_proof_l657_65776


namespace plains_routes_count_l657_65715

-- Defining the total number of cities and the number of cities in each region
def total_cities : Nat := 100
def mountainous_cities : Nat := 30
def plains_cities : Nat := total_cities - mountainous_cities

-- Defining the number of routes established each year and over three years
def routes_per_year : Nat := 50
def total_routes : Nat := routes_per_year * 3

-- Defining the number of routes connecting pairs of mountainous cities
def mountainous_routes : Nat := 21

-- The statement to prove the number of routes connecting pairs of plains cities
theorem plains_routes_count :
  plains_cities = 70 →
  total_routes = 150 →
  mountainous_routes = 21 →
  3 * mountainous_cities - 2 * mountainous_routes = 48 →
  3 * plains_cities - 48 = 162 →
  81 = 81 := sorry

end plains_routes_count_l657_65715


namespace fraction_of_oil_sent_to_production_l657_65705

-- Definitions based on the problem's conditions
def initial_concentration : ℝ := 0.02
def replacement_concentration1 : ℝ := 0.03
def replacement_concentration2 : ℝ := 0.015
def final_concentration : ℝ := 0.02

-- Main theorem stating the fraction x is 1/2
theorem fraction_of_oil_sent_to_production (x : ℝ) (hx : x > 0) :
  (initial_concentration + (replacement_concentration1 - initial_concentration) * x) * (1 - x) +
  replacement_concentration2 * x = final_concentration →
  x = 0.5 :=
  sorry

end fraction_of_oil_sent_to_production_l657_65705


namespace find_m_over_n_l657_65772

variable (a b : ℝ × ℝ)
variable (m n : ℝ)
variable (n_nonzero : n ≠ 0)

axiom a_def : a = (1, 2)
axiom b_def : b = (-2, 3)
axiom collinear : ∃ k : ℝ, m • a - n • b = k • (a + 2 • b)

theorem find_m_over_n : m / n = -1 / 2 := by
  sorry

end find_m_over_n_l657_65772


namespace option_B_shares_asymptotes_l657_65704

-- Define the given hyperbola equation
def given_hyperbola (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1

-- The asymptotes for the given hyperbola
def asymptotes_of_given_hyperbola (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Define the hyperbola for option B
def option_B_hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 16) = 1

-- The asymptotes for option B hyperbola
def asymptotes_of_option_B_hyperbola (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Theorem stating that the hyperbola in option B shares the same asymptotes as the given hyperbola
theorem option_B_shares_asymptotes :
  (∀ x y : ℝ, given_hyperbola x y → asymptotes_of_given_hyperbola x y) →
  (∀ x y : ℝ, option_B_hyperbola x y → asymptotes_of_option_B_hyperbola x y) :=
by
  intros h₁ h₂
  -- Here should be the proof to show they have the same asymptotes
  sorry

end option_B_shares_asymptotes_l657_65704


namespace find_a5_of_geom_seq_l657_65728

theorem find_a5_of_geom_seq 
  (a : ℕ → ℝ) (q : ℝ)
  (hgeom : ∀ n, a (n + 1) = a n * q)
  (S : ℕ → ℝ)
  (hS3 : S 3 = a 0 * (1 - q ^ 3) / (1 - q))
  (hS6 : S 6 = a 0 * (1 - q ^ 6) / (1 - q))
  (hS9 : S 9 = a 0 * (1 - q ^ 9) / (1 - q))
  (harith : S 3 + S 6 = 2 * S 9)
  (a8 : a 8 = 3) :
  a 5 = -6 :=
by
  sorry

end find_a5_of_geom_seq_l657_65728


namespace num_of_int_solutions_l657_65762

/-- 
  The number of integer solutions to the equation 
  \((x^3 - x - 1)^{2015} = 1\) is 3.
-/
theorem num_of_int_solutions :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℤ, (x ^ 3 - x - 1) ^ 2015 = 1 ↔ x = 0 ∨ x = 1 ∨ x = -1 := 
sorry

end num_of_int_solutions_l657_65762


namespace car_rental_cost_l657_65716

theorem car_rental_cost (daily_rent : ℕ) (rent_duration : ℕ) (mileage_rate : ℚ) (mileage : ℕ) (total_cost : ℕ) :
  daily_rent = 30 → rent_duration = 5 → mileage_rate = 0.25 → mileage = 500 → total_cost = 275 :=
by
  intros hd hr hm hl
  sorry

end car_rental_cost_l657_65716


namespace maximum_cars_quotient_l657_65702

theorem maximum_cars_quotient
  (car_length : ℕ) (m_speed : ℕ) (half_hour_distance : ℕ) 
  (unit_length : ℕ) (max_units : ℕ) (N : ℕ) :
  (car_length = 5) →
  (half_hour_distance = 10000) →
  (unit_length = 5 * (m_speed + 1)) →
  (max_units = half_hour_distance / unit_length) →
  (N = max_units) →
  (N / 10 = 200) :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end maximum_cars_quotient_l657_65702


namespace first_term_of_geometric_sequence_l657_65788

theorem first_term_of_geometric_sequence (a r : ℕ) :
  (a * r ^ 3 = 54) ∧ (a * r ^ 4 = 162) → a = 2 :=
by
  -- Provided conditions and the goal
  sorry

end first_term_of_geometric_sequence_l657_65788
